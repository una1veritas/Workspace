/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments AM1705
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: #2 $
 **
 ***************************************************************************/

#ifndef __IOAM1705_H
#define __IOAM1705_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4f = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    AM1705 SPECIAL FUNCTION REGISTERS
 **
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

/* MPU Configuration Register (CONFIG) */
typedef struct {
  __REG32 ASSUME_ALLOWED  : 1;
  __REG32                 :11;
  __REG32 NUM_AIDS        : 4;
  __REG32 NUM_PROG        : 4;
  __REG32 NUM_FIXED       : 4;
  __REG32 ADDR_WIDTH      : 8;
} __mpu_config_bits;

/* MPU Interrupt Raw Status/Set Register (IRAWSTAT) */
/* Interrupt Enable Status/Clear Register (IENSTAT) */
typedef struct {
  __REG32 PROTERR         : 1;
  __REG32 ADDRERR         : 1;
  __REG32                 :30;
} __mpu_irawstat_bits;

/* MPU Interrupt Enable Set Register (IENSET) */
typedef struct {
  __REG32 PROTERR_EN      : 1;
  __REG32 ADDRERR_EN      : 1;
  __REG32                 :30;
} __mpu_ienset_bits;

/* MPU Interrupt Enable Clear Register (IENCLR) */
typedef struct {
  __REG32 PROTERR_CLR     : 1;
  __REG32 ADDRERR_CLR     : 1;
  __REG32                 :30;
} __mpu_ienclr_bits;

/* MPU Fixed Range Memory Protection Page Attributes Register (FXD_MPPA) */
/* MPU Programmable Range Memory Protection Page Attributes Register (PROGn_MPPA) */
typedef struct {
  __REG32 UX              : 1;
  __REG32 UW              : 1;
  __REG32 UR              : 1;
  __REG32 SX              : 1;
  __REG32 SW              : 1;
  __REG32 SR              : 1;
  __REG32                 : 3;
  __REG32 AIDX            : 1;
  __REG32 AID0            : 1;
  __REG32 AID1            : 1;
  __REG32 AID2            : 1;
  __REG32 AID3            : 1;
  __REG32 AID4            : 1;
  __REG32 AID5            : 1;
  __REG32 AID6            : 1;
  __REG32 AID7            : 1;
  __REG32 AID8            : 1;
  __REG32 AID9            : 1;
  __REG32 AID10           : 1;
  __REG32 AID11           : 1;
  __REG32                 :10;
} __mpu_fxd_mppa_bits;

/* MPU Fault Status Register (FLTSTAT) */
typedef struct {
  __REG32 TYPE            : 6;
  __REG32                 : 3;
  __REG32 PRIVID          : 4;
  __REG32                 : 3;
  __REG32 MSTID           : 8;
  __REG32                 : 8;
} __mpu_fltstat_bits;

/* MPU Fault Clear Register (FLTCLR) */
typedef struct {
  __REG32 CLEAR           : 1;
  __REG32                 :31;
} __mpu_fltclr_bits;

/* PLLC0 Reset Type Status Register (RSTYPE) */
typedef struct {
  __REG32 POR             : 1;
  __REG32 XWRST           : 1;
  __REG32 PLLSWRST        : 1;
  __REG32                 :29;
} __pllc0_rstype_bits;

/* PLLC0 Control Register (PLLCTL) */
typedef struct {
  __REG32 PLLEN           : 1;
  __REG32 PLLPWRDN        : 1;
  __REG32                 : 1;
  __REG32 PLLRST          : 1;
  __REG32                 : 1;
  __REG32 PLLENSRC        : 1;
  __REG32                 : 2;
  __REG32 CLKMODE         : 1;
  __REG32                 :23;
} __pllc0_pllctl_bits;

/* PLLCx Multiplier Control Register (PLLM) */
typedef struct {
  __REG32 PLLM            : 5;
  __REG32                 :27;
} __pllc_pllm_bits;

/* PLLC0 Pre-Divider Control Register (PREDIV) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 PREDEN          : 1;
  __REG32                 :16;
} __pllc0_prediv_bits;

/* PLLCx Divider 1 Register (PLLDIV1) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D1EN            : 1;
  __REG32                 :16;
} __pllc_plldiv1_bits;

/* PLLCx Divider 2 Register (PLLDIV2) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D2EN            : 1;
  __REG32                 :16;
} __pllc_plldiv2_bits;

/* PLLCx Divider 3 Register (PLLDIV3) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D3EN            : 1;
  __REG32                 :16;
} __pllc_plldiv3_bits;

/* PLLC0 Divider 4 Register (PLLDIV4) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D4EN            : 1;
  __REG32                 :16;
} __pllc0_plldiv4_bits;

/* PLLC0 Divider 5 Register (PLLDIV5) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D5EN            : 1;
  __REG32                 :16;
} __pllc0_plldiv5_bits;

/* PLLC0 Divider 6 Register (PLLDIV6) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D6EN            : 1;
  __REG32                 :16;
} __pllc0_plldiv6_bits;

/* PLLC0 Divider 7 Register (PLLDIV7) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 D7EN            : 1;
  __REG32                 :16;
} __pllc0_plldiv7_bits;

/* PLLCx Post-Divider Control Register (POSTDIV) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 POSTDEN         : 1;
  __REG32                 :16;
} __pllc_postdiv_bits;

/* PLLCx Controller Command Register (PLLCMD) */
typedef struct {
  __REG32 GOSET           : 1;
  __REG32                 :31;
} __pllc_pllcmd_bits;

/* PLLCx Controller Status Register (PLLSTAT) */
typedef struct {
  __REG32 GOSTAT          : 1;
  __REG32                 : 1;
  __REG32 STABLE          : 1;
  __REG32                 :29;
} __pllc_pllstat_bits;

/* PLLC0 Clock Align Control Register (ALNCTL) */
typedef struct {
  __REG32 ALN1            : 1;
  __REG32 ALN2            : 1;
  __REG32 ALN3            : 1;
  __REG32 ALN4            : 1;
  __REG32 ALN5            : 1;
  __REG32 ALN6            : 1;
  __REG32 ALN7            : 1;
  __REG32                 :25;
} __pllc0_alnctl_bits;

/* PLLC0 PLLDIV Ratio Change Status Register (DCHANGE) */
typedef struct {
  __REG32 SYS1            : 1;
  __REG32 SYS2            : 1;
  __REG32 SYS3            : 1;
  __REG32 SYS4            : 1;
  __REG32 SYS5            : 1;
  __REG32 SYS6            : 1;
  __REG32 SYS7            : 1;
  __REG32                 :25;
} __pllc0_dchange_bits;

/* PLLC0 Clock Enable Control Register (CKEN) */
/* Clock Status Register (CKSTAT) */
typedef struct {
  __REG32 AUXEN           : 1;
  __REG32                 :31;
} __pllc0_cken_bits;

/* PLLC0 SYSCLK Status Register (SYSTAT) */
typedef struct {
  __REG32 SYS1ON          : 1;
  __REG32 SYS2ON          : 1;
  __REG32 SYS3ON          : 1;
  __REG32 SYS4ON          : 1;
  __REG32 SYS5ON          : 1;
  __REG32 SYS6ON          : 1;
  __REG32 SYS7ON          : 1;
  __REG32                 :25;
} __pllc0_systat_bits;

/* PSC Interrupt Evaluation Register (INTEVAL) */
typedef struct {
  __REG32 ALLEV           : 1;
  __REG32                 :31;
} __psc_inteval_bits;

/* PSC0 Module Error Pending Register 0 (modules 0-15) (MERRPR0) */
/* PSC0 Module Error Clear Register 0 (modules 0-15) (MERRCR0) */
typedef struct {
  __REG32                 :14;
  __REG32 M14             : 1;
  __REG32                 :17;
} __psc0_merrpr0_bits;

/* PSC Power Error Pending Register (PERRPR) */
/* PSC Power Error Clear Register (PERRCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 P1              : 1;
  __REG32                 :30;
} __psc_perrpr_bits;

/* PSC Power Domain Transition Command Register (PTCMD) */
typedef struct {
  __REG32 GO0             : 1;
  __REG32 GO1             : 1;
  __REG32                 :30;
} __psc_ptcmd_bits;

/* PSC Power Domain Transition Status Register (PTSTAT) */
typedef struct {
  __REG32 GOSTAT0         : 1;
  __REG32 GOSTAT1         : 1;
  __REG32                 :30;
} __psc_ptstat_bits;

/* PSC Power Domain 0/1 Status Register (PDSTAT0/1) */
typedef struct {
  __REG32 STATE           : 5;
  __REG32                 : 3;
  __REG32 POR             : 1;
  __REG32 PORDONE         : 1;
  __REG32                 : 1;
  __REG32 EMUIHB          : 1;
  __REG32                 :20;
} __psc_pdstat_bits;

/* PSC Power Domain 0/1 Control Register (PDCTL0/1) */
typedef struct {
  __REG32 NEXT            : 1;
  __REG32                 : 8;
  __REG32 EMUIHBIE        : 1;
  __REG32                 : 2;
  __REG32 PDMODE          : 4;
  __REG32 WAKECNT         : 8;
  __REG32                 : 8;
} __psc_pdctl_bits;

/* PSC Power Domain 0/1 Configuration Register (PDCFG0/1) */
typedef struct {
  __REG32 ALWAYSON        : 1;
  __REG32 RAM_PSM         : 1;
  __REG32 ICEPICK         : 1;
  __REG32 PD_LOCK         : 1;
  __REG32                 :28;
} __psc_pdcfg_bits;

/* PSC Module Status n Register (MDSTATn) */
typedef struct {
  __REG32 STATE           : 6;
  __REG32                 : 2;
  __REG32 LRST            : 1;
  __REG32 LRSTDONE        : 1;
  __REG32 MRST            : 1;
  __REG32                 : 1;
  __REG32 MCKOUT          : 1;
  __REG32                 : 3;
  __REG32 EMURST          : 1;
  __REG32 EMUIHB          : 1;
  __REG32                 :14;
} __psc_mdstat_bits;

/* PSC0 Module Control n Register (modules 0-15) (MDCTLn) */
typedef struct {
  __REG32 NEXT            : 3;
  __REG32                 : 5;
  __REG32 LRST            : 1;
  __REG32 EMURSTIE        : 1;
  __REG32 EMUIHBIE        : 1;
  __REG32                 :20;
  __REG32 FORCE           : 1;
} __psc0_mdctl_bits;

/* PSC1 Module Control n Register (modules 0-31) (MDCTLn) */
typedef struct {
  __REG32 NEXT            : 3;
  __REG32                 :28;
  __REG32 FORCE           : 1;
} __psc1_mdctl_bits;

/* SYSCFG Boot Configuration Register (BOOTCFG) */
typedef struct {
  __REG32 BOOTMODE        :16;
  __REG32                 :16;
} __syscfg_bootcfg_bits;

/* SYSCFG Host 0 Configuration Register (HOST0CFG) */
typedef struct {
  __REG32 BOOTRDY         : 1;
  __REG32                 :31;
} __syscfg_host0cfg_bits;

/* SYSCFG Interrupt Raw Status/Set Register (IRAWSTAT) */
/* SYSCFG Interrupt Enable Status/Clear Register (IENSTAT) */
typedef struct {
  __REG32 PROTERR         : 1;
  __REG32 ADDRERR         : 1;
  __REG32                 :30;
} __syscfg_irawstat_bits;

/* SYSCFG Interrupt Enable Register (IENSET) */
typedef struct {
  __REG32 PROTERR_EN      : 1;
  __REG32 ADDRERR_EN      : 1;
  __REG32                 :30;
} __syscfg_ienset_bits;

/* SYSCFG Interrupt Enable Clear Register (IENCLR) */
typedef struct {
  __REG32 PROTERR_CLR     : 1;
  __REG32 ADDRERR_CLR     : 1;
  __REG32                 :30;
} __syscfg_ienclr_bits;

/* SYSCFG End of Interrupt Register (EOI) */
typedef struct {
  __REG32 EOIVECT         : 8;
  __REG32                 :24;
} __syscfg_eoi_bits;

/* SYSCFG Fault Status Register (FLTSTAT) */
typedef struct {
  __REG32 TYPE            : 6;
  __REG32                 : 3;
  __REG32 PRIVID          : 4;
  __REG32                 : 3;
  __REG32 MSTID           : 8;
  __REG32 ID              : 8;
} __syscfg_fltstat_bits;

/* SYSCFG Master Priority 0 Register (MSTPRI0) */
typedef struct {
  __REG32 ARM_I           : 3;
  __REG32                 : 1;
  __REG32 ARM_D           : 3;
  __REG32                 :25;
} __syscfg_mstpri0_bits;

/* SYSCFG Master Priority 1 Register (MSTPRI1) */
typedef struct {
  __REG32 PRU0            : 3;
  __REG32                 : 1;
  __REG32 PRU1            : 3;
  __REG32                 : 1;
  __REG32 EDMA30TC0       : 3;
  __REG32                 : 1;
  __REG32 EDMA30TC1       : 3;
  __REG32                 :17;
} __syscfg_mstpri1_bits;

/* SYSCFG Master Priority 2 Register (MSTPRI2) */
typedef struct {
  __REG32 EMAC            : 3;
  __REG32                 : 5;
  __REG32 USB0CFG         : 3;
  __REG32                 : 1;
  __REG32 USB0CDMA        : 3;
  __REG32                 :17;
} __syscfg_mstpri2_bits;

/* SYSCFG Pin Multiplexing Control 0 Register (PINMUX0) */
typedef struct {
  __REG32 PINMUX0_3_0     : 4;
  __REG32 PINMUX0_7_4     : 4;
  __REG32 PINMUX0_11_8    : 4;
  __REG32 PINMUX0_15_12   : 4;
  __REG32 PINMUX0_19_16   : 4;
  __REG32 PINMUX0_23_20   : 4;
  __REG32 PINMUX0_27_24   : 4;
  __REG32 PINMUX0_31_28   : 4;
} __syscfg_pinmux0_bits;

/* SYSCFG Pin Multiplexing Control 1 Register (PINMUX1) */
typedef struct {
  __REG32 PINMUX1_3_0     : 4;
  __REG32 PINMUX1_7_4     : 4;
  __REG32 PINMUX1_11_8    : 4;
  __REG32 PINMUX1_15_12   : 4;
  __REG32 PINMUX1_19_16   : 4;
  __REG32 PINMUX1_23_20   : 4;
  __REG32 PINMUX1_27_24   : 4;
  __REG32 PINMUX1_31_28   : 4;
} __syscfg_pinmux1_bits;

/* SYSCFG Pin Multiplexing Control 2 Register (PINMUX2) */
typedef struct {
  __REG32 PINMUX2_3_0     : 4;
  __REG32 PINMUX2_7_4     : 4;
  __REG32 PINMUX2_11_8    : 4;
  __REG32 PINMUX2_15_12   : 4;
  __REG32 PINMUX2_19_16   : 4;
  __REG32 PINMUX2_23_20   : 4;
  __REG32 PINMUX2_27_24   : 4;
  __REG32 PINMUX2_31_28   : 4;
} __syscfg_pinmux2_bits;

/* SYSCFG Pin Multiplexing Control 3 Register (PINMUX3) */
typedef struct {
  __REG32 PINMUX3_3_0     : 4;
  __REG32 PINMUX3_7_4     : 4;
  __REG32 PINMUX3_11_8    : 4;
  __REG32 PINMUX3_15_12   : 4;
  __REG32 PINMUX3_19_16   : 4;
  __REG32 PINMUX3_23_20   : 4;
  __REG32 PINMUX3_27_24   : 4;
  __REG32 PINMUX3_31_28   : 4;
} __syscfg_pinmux3_bits;

/* SYSCFG Pin Multiplexing Control 4 Register (PINMUX4) */
typedef struct {
  __REG32 PINMUX4_3_0     : 4;
  __REG32 PINMUX4_7_4     : 4;
  __REG32 PINMUX4_11_8    : 4;
  __REG32 PINMUX4_15_12   : 4;
  __REG32 PINMUX4_19_16   : 4;
  __REG32 PINMUX4_23_20   : 4;
  __REG32 PINMUX4_27_24   : 4;
  __REG32 PINMUX4_31_28   : 4;
} __syscfg_pinmux4_bits;

/* SYSCFG Pin Multiplexing Control 5 Register (PINMUX5) */
typedef struct {
  __REG32 PINMUX5_3_0     : 4;
  __REG32 PINMUX5_7_4     : 4;
  __REG32 PINMUX5_11_8    : 4;
  __REG32 PINMUX5_15_12   : 4;
  __REG32 PINMUX5_19_16   : 4;
  __REG32 PINMUX5_23_20   : 4;
  __REG32 PINMUX5_27_24   : 4;
  __REG32 PINMUX5_31_28   : 4;
} __syscfg_pinmux5_bits;

/* SYSCFG Pin Multiplexing Control 6 Register (PINMUX6) */
typedef struct {
  __REG32 PINMUX6_3_0     : 4;
  __REG32 PINMUX6_7_4     : 4;
  __REG32 PINMUX6_11_8    : 4;
  __REG32 PINMUX6_15_12   : 4;
  __REG32 PINMUX6_19_16   : 4;
  __REG32 PINMUX6_23_20   : 4;
  __REG32 PINMUX6_27_24   : 4;
  __REG32 PINMUX6_31_28   : 4;
} __syscfg_pinmux6_bits;

/* SYSCFG Pin Multiplexing Control 7 Register (PINMUX7) */
typedef struct {
  __REG32 PINMUX7_3_0     : 4;
  __REG32 PINMUX7_7_4     : 4;
  __REG32 PINMUX7_11_8    : 4;
  __REG32 PINMUX7_15_12   : 4;
  __REG32 PINMUX7_19_16   : 4;
  __REG32 PINMUX7_23_20   : 4;
  __REG32 PINMUX7_27_24   : 4;
  __REG32 PINMUX7_31_28   : 4;
} __syscfg_pinmux7_bits;

/* SYSCFG Pin Multiplexing Control 8 Register (PINMUX8) */
typedef struct {
  __REG32 PINMUX8_3_0     : 4;
  __REG32 PINMUX8_7_4     : 4;
  __REG32 PINMUX8_11_8    : 4;
  __REG32 PINMUX8_15_12   : 4;
  __REG32 PINMUX8_19_16   : 4;
  __REG32 PINMUX8_23_20   : 4;
  __REG32 PINMUX8_27_24   : 4;
  __REG32 PINMUX8_31_28   : 4;
} __syscfg_pinmux8_bits;

/* SYSCFG Pin Multiplexing Control 9 Register (PINMUX9) */
typedef struct {
  __REG32 PINMUX9_3_0     : 4;
  __REG32 PINMUX9_7_4     : 4;
  __REG32 PINMUX9_11_8    : 4;
  __REG32 PINMUX9_15_12   : 4;
  __REG32 PINMUX9_19_16   : 4;
  __REG32 PINMUX9_23_20   : 4;
  __REG32 PINMUX9_27_24   : 4;
  __REG32 PINMUX9_31_28   : 4;
} __syscfg_pinmux9_bits;

/* SYSCFG Pin Multiplexing Control 10 Register (PINMUX10) */
typedef struct {
  __REG32 PINMUX10_3_0     : 4;
  __REG32 PINMUX10_7_4     : 4;
  __REG32 PINMUX10_11_8    : 4;
  __REG32 PINMUX10_15_12   : 4;
  __REG32 PINMUX10_19_16   : 4;
  __REG32 PINMUX10_23_20   : 4;
  __REG32 PINMUX10_27_24   : 4;
  __REG32 PINMUX10_31_28   : 4;
} __syscfg_pinmux10_bits;

/* SYSCFG Pin Multiplexing Control 11 Register (PINMUX11) */
typedef struct {
  __REG32 PINMUX11_3_0     : 4;
  __REG32 PINMUX11_7_4     : 4;
  __REG32 PINMUX11_11_8    : 4;
  __REG32 PINMUX11_15_12   : 4;
  __REG32 PINMUX11_19_16   : 4;
  __REG32 PINMUX11_23_20   : 4;
  __REG32 PINMUX11_27_24   : 4;
  __REG32 PINMUX11_31_28   : 4;
} __syscfg_pinmux11_bits;

/* SYSCFG Pin Multiplexing Control 12 Register (PINMUX12) */
typedef struct {
  __REG32 PINMUX12_3_0     : 4;
  __REG32 PINMUX12_7_4     : 4;
  __REG32 PINMUX12_11_8    : 4;
  __REG32 PINMUX12_15_12   : 4;
  __REG32 PINMUX12_19_16   : 4;
  __REG32 PINMUX12_23_20   : 4;
  __REG32 PINMUX12_27_24   : 4;
  __REG32 PINMUX12_31_28   : 4;
} __syscfg_pinmux12_bits;

/* SYSCFG Pin Multiplexing Control 13 Register (PINMUX13) */
typedef struct {
  __REG32 PINMUX13_3_0     : 4;
  __REG32 PINMUX13_7_4     : 4;
  __REG32 PINMUX13_11_8    : 4;
  __REG32 PINMUX13_15_12   : 4;
  __REG32 PINMUX13_19_16   : 4;
  __REG32 PINMUX13_23_20   : 4;
  __REG32 PINMUX13_27_24   : 4;
  __REG32 PINMUX13_31_28   : 4;
} __syscfg_pinmux13_bits;

/* SYSCFG Pin Multiplexing Control 14 Register (PINMUX14) */
typedef struct {
  __REG32 PINMUX14_3_0     : 4;
  __REG32 PINMUX14_7_4     : 4;
  __REG32 PINMUX14_11_8    : 4;
  __REG32 PINMUX14_15_12   : 4;
  __REG32 PINMUX14_19_16   : 4;
  __REG32 PINMUX14_23_20   : 4;
  __REG32 PINMUX14_27_24   : 4;
  __REG32 PINMUX14_31_28   : 4;
} __syscfg_pinmux14_bits;

/* SYSCFG Pin Multiplexing Control 15 Register (PINMUX15) */
typedef struct {
  __REG32 PINMUX15_3_0     : 4;
  __REG32 PINMUX15_7_4     : 4;
  __REG32 PINMUX15_11_8    : 4;
  __REG32 PINMUX15_15_12   : 4;
  __REG32 PINMUX15_19_16   : 4;
  __REG32 PINMUX15_23_20   : 4;
  __REG32 PINMUX15_27_24   : 4;
  __REG32 PINMUX15_31_28   : 4;
} __syscfg_pinmux15_bits;

/* SYSCFG Pin Multiplexing Control 16 Register (PINMUX16) */
typedef struct {
  __REG32 PINMUX16_3_0     : 4;
  __REG32 PINMUX16_7_4     : 4;
  __REG32 PINMUX16_11_8    : 4;
  __REG32 PINMUX16_15_12   : 4;
  __REG32 PINMUX16_19_16   : 4;
  __REG32 PINMUX16_23_20   : 4;
  __REG32 PINMUX16_27_24   : 4;
  __REG32 PINMUX16_31_28   : 4;
} __syscfg_pinmux16_bits;

/* SYSCFG Pin Multiplexing Control 17 Register (PINMUX17) */
typedef struct {
  __REG32 PINMUX17_3_0     : 4;
  __REG32 PINMUX17_7_4     : 4;
  __REG32 PINMUX17_11_8    : 4;
  __REG32 PINMUX17_15_12   : 4;
  __REG32 PINMUX17_19_16   : 4;
  __REG32 PINMUX17_23_20   : 4;
  __REG32 PINMUX17_27_24   : 4;
  __REG32 PINMUX17_31_28   : 4;
} __syscfg_pinmux17_bits;

/* SYSCFG Pin Multiplexing Control 18 Register (PINMUX18) */
typedef struct {
  __REG32 PINMUX18_3_0     : 4;
  __REG32 PINMUX18_7_4     : 4;
  __REG32 PINMUX18_11_8    : 4;
  __REG32 PINMUX18_15_12   : 4;
  __REG32 PINMUX18_19_16   : 4;
  __REG32 PINMUX18_23_20   : 4;
  __REG32 PINMUX18_27_24   : 4;
  __REG32 PINMUX18_31_28   : 4;
} __syscfg_pinmux18_bits;

/* SYSCFG Pin Multiplexing Control 19 Register (PINMUX19) */
typedef struct {
  __REG32 PINMUX19_3_0     : 4;
  __REG32                  :28;
} __syscfg_pinmux19_bits;

/* SYSCFG Suspend Source Register (SUSPSRC) */
typedef struct {
  __REG32 ECAP0SRC         : 1;
  __REG32 ECAP1SRC         : 1;
  __REG32 ECAP2SRC         : 1;
  __REG32 EQEP0SRC         : 1;
  __REG32 EQEP1SRC         : 1;
  __REG32 EMACSRC          : 1;
  __REG32 PRUSRC           : 1;
  __REG32                  : 2;
  __REG32 USB0SRC          : 1;
  __REG32                  : 6;
  __REG32 I2C0SRC          : 1;
  __REG32 I2C1SRC          : 1;
  __REG32 UART0SRC         : 1;
  __REG32 UART1SRC         : 1;
  __REG32 UART2SRC         : 1;
  __REG32 SPI0SRC          : 1;
  __REG32 SPI1SRC          : 1;
  __REG32 EPWM0SRC         : 1;
  __REG32 EPWM1SRC         : 1;
  __REG32 EPWM2SRC         : 1;
  __REG32                  : 1;
  __REG32 TIMER64P_0SRC    : 1;
  __REG32 TIMER64P_1SRC    : 1;
  __REG32                  : 3;
} __syscfg_suspsrc_bits;

/* SYSCFG Suspend Source Register (SUSPSRC) */
/* SYSCFG Chip Signal Clear Register (CHIPSIG_CLR) */
typedef struct {
  __REG32 CHIPSIG0         : 1;
  __REG32 CHIPSIG1         : 1;
  __REG32 CHIPSIG2         : 1;
  __REG32 CHIPSIG3         : 1;
  __REG32                  :28;
} __syscfg_chipsig_bits;

/* SYSCFG Chip Configuration 0 Register (CFGCHIP0) */
typedef struct {
  __REG32 TC0DBS           : 2;
  __REG32 TC1DBS           : 2;
  __REG32 PLL_MASTER_LOCK  : 1;
  __REG32                  :27;
} __syscfg_cfgchip0_bits;

/* SYSCFG Chip Configuration 1 Register (CFGCHIP1) */
typedef struct {
  __REG32 AMUTESEL0        : 4;
  __REG32 AMUTESEL1        : 4;
  __REG32                  : 4;
  __REG32 TBCLKSYNC        : 1;
  __REG32                  : 4;
  __REG32 CAP0SRC          : 5;
  __REG32 CAP1SRC          : 5;
  __REG32 CAP2SRC          : 5;
} __syscfg_cfgchip1_bits;

/* SYSCFG Chip Configuration 2 Register (CFGCHIP2) */
typedef struct {
  __REG32 USB0REF_FREQ     : 4;
  __REG32 USB0VBDTCTEN     : 1;
  __REG32 USB0SESNDEN      : 1;
  __REG32 USB0PHY_PLLON    : 1;
  __REG32                  : 1;
  __REG32 USB0DATPOL       : 1;
  __REG32 USB0OTGPWRDN     : 1;
  __REG32 USB0PHYPWDN      : 1;
  __REG32 USB0PHYCLKMUX    : 1;
  __REG32                  : 1;
  __REG32 USB0OTGMODE      : 2;
  __REG32 RESET            : 1;
  __REG32 USB0VBUSSENSE    : 1;
  __REG32 USB0PHYCLKGD     : 1;
  __REG32                  :14;
} __syscfg_cfgchip2_bits;

/* SYSCFG Chip Configuration 3 Register (CFGCHIP3) */
typedef struct {
  __REG32 EMB_CLKSRC       : 1;
  __REG32 EMA_CLKSRC       : 1;
  __REG32 DIV45PENA        : 1;
  __REG32                  :29;
} __syscfg_cfgchip3_bits;

/* SYSCFG Chip Configuration 4 Register (CFGCHIP4) */
typedef struct {
  __REG32 AMUTECLR0        : 1;
  __REG32 AMUTECLR1        : 1;
  __REG32                  :30;
} __syscfg_cfgchip4_bits;

/* AINTC Control Register (CR) */
typedef struct {
  __REG32                  : 2;
  __REG32 NESTMODE         : 2;
  __REG32 PRHOLDMODE       : 1;
  __REG32                  :27;
} __aintc_cr_bits;

/* AINTC Global Enable Register (GER) */
typedef struct {
  __REG32 ENABLE           : 1;
  __REG32                  :31;
} __aintc_ger_bits;

/* AINTC Global Nesting Level Register (GNLR) */
typedef struct {
  __REG32 NESTLVL          : 9;
  __REG32                  :22;
  __REG32 OVERRIDE         : 1;
} __aintc_gnlr_bits;

/* AINTC System Interrupt Status Indexed Set Register (SISR) */
/* AINTC System Interrupt Status Indexed Clear Register (SICR) */
/* AINTC System Interrupt Enable Indexed Set Register (EISR) */
/* AINTC System Interrupt Enable Indexed Clear Register (EICR) */
typedef struct {
  __REG32 INDEX            : 7;
  __REG32                  :25;
} __aintc_sisr_bits;

/* AINTC Host Interrupt Enable Indexed Set Register (HIEISR) */
/* AINTC Host Interrupt Enable Indexed Clear Register (HIEICR) */
typedef struct {
  __REG32 INDEX            : 1;
  __REG32                  :31;
} __aintc_hieisr_bits;

/* AINTC Vector Size Register (VSR) */
typedef struct {
  __REG32 SIZE             : 8;
  __REG32                  :24;
} __aintc_vsr_bits;

/* AINTC Global Prioritized Index Register (GPIR) */
typedef struct {
  __REG32 PRI_INDX         :10;
  __REG32                  :21;
  __REG32 NONE             : 1;
} __aintc_gpir_bits;

/* AINTC System Interrupt Status Raw/Set Register 0 (SRSR0) */
typedef struct {
  __REG32 RAW_STATUS0      : 1;
  __REG32 RAW_STATUS1      : 1;
  __REG32 RAW_STATUS2      : 1;
  __REG32 RAW_STATUS3      : 1;
  __REG32 RAW_STATUS4      : 1;
  __REG32 RAW_STATUS5      : 1;
  __REG32 RAW_STATUS6      : 1;
  __REG32 RAW_STATUS7      : 1;
  __REG32 RAW_STATUS8      : 1;
  __REG32 RAW_STATUS9      : 1;
  __REG32 RAW_STATUS10     : 1;
  __REG32 RAW_STATUS11     : 1;
  __REG32 RAW_STATUS12     : 1;
  __REG32 RAW_STATUS13     : 1;
  __REG32 RAW_STATUS14     : 1;
  __REG32 RAW_STATUS15     : 1;
  __REG32 RAW_STATUS16     : 1;
  __REG32 RAW_STATUS17     : 1;
  __REG32 RAW_STATUS18     : 1;
  __REG32 RAW_STATUS19     : 1;
  __REG32 RAW_STATUS20     : 1;
  __REG32 RAW_STATUS21     : 1;
  __REG32 RAW_STATUS22     : 1;
  __REG32 RAW_STATUS23     : 1;
  __REG32 RAW_STATUS24     : 1;
  __REG32 RAW_STATUS25     : 1;
  __REG32 RAW_STATUS26     : 1;
  __REG32 RAW_STATUS27     : 1;
  __REG32 RAW_STATUS28     : 1;
  __REG32 RAW_STATUS29     : 1;
  __REG32 RAW_STATUS30     : 1;
  __REG32 RAW_STATUS31     : 1;
} __aintc_srsr0_bits;

/* AINTC System Interrupt Status Raw/Set Register 1 (SRSR1) */
typedef struct {
  __REG32 RAW_STATUS32     : 1;
  __REG32 RAW_STATUS33     : 1;
  __REG32 RAW_STATUS34     : 1;
  __REG32 RAW_STATUS35     : 1;
  __REG32 RAW_STATUS36     : 1;
  __REG32 RAW_STATUS37     : 1;
  __REG32 RAW_STATUS38     : 1;
  __REG32 RAW_STATUS39     : 1;
  __REG32 RAW_STATUS40     : 1;
  __REG32 RAW_STATUS41     : 1;
  __REG32 RAW_STATUS42     : 1;
  __REG32 RAW_STATUS43     : 1;
  __REG32 RAW_STATUS44     : 1;
  __REG32 RAW_STATUS45     : 1;
  __REG32 RAW_STATUS46     : 1;
  __REG32 RAW_STATUS47     : 1;
  __REG32 RAW_STATUS48     : 1;
  __REG32 RAW_STATUS49     : 1;
  __REG32 RAW_STATUS50     : 1;
  __REG32 RAW_STATUS51     : 1;
  __REG32 RAW_STATUS52     : 1;
  __REG32 RAW_STATUS53     : 1;
  __REG32 RAW_STATUS54     : 1;
  __REG32 RAW_STATUS55     : 1;
  __REG32 RAW_STATUS56     : 1;
  __REG32 RAW_STATUS57     : 1;
  __REG32 RAW_STATUS58     : 1;
  __REG32 RAW_STATUS59     : 1;
  __REG32 RAW_STATUS60     : 1;
  __REG32 RAW_STATUS61     : 1;
  __REG32 RAW_STATUS62     : 1;
  __REG32 RAW_STATUS63     : 1;
} __aintc_srsr1_bits;

/* AINTC System Interrupt Status Raw/Set Register 2 (SRSR2) */
typedef struct {
  __REG32 RAW_STATUS64     : 1;
  __REG32 RAW_STATUS65     : 1;
  __REG32 RAW_STATUS66     : 1;
  __REG32 RAW_STATUS67     : 1;
  __REG32 RAW_STATUS68     : 1;
  __REG32 RAW_STATUS69     : 1;
  __REG32 RAW_STATUS70     : 1;
  __REG32 RAW_STATUS71     : 1;
  __REG32 RAW_STATUS72     : 1;
  __REG32 RAW_STATUS73     : 1;
  __REG32 RAW_STATUS74     : 1;
  __REG32 RAW_STATUS75     : 1;
  __REG32 RAW_STATUS76     : 1;
  __REG32 RAW_STATUS77     : 1;
  __REG32 RAW_STATUS78     : 1;
  __REG32 RAW_STATUS79     : 1;
  __REG32 RAW_STATUS80     : 1;
  __REG32 RAW_STATUS81     : 1;
  __REG32 RAW_STATUS82     : 1;
  __REG32 RAW_STATUS83     : 1;
  __REG32 RAW_STATUS84     : 1;
  __REG32 RAW_STATUS85     : 1;
  __REG32 RAW_STATUS86     : 1;
  __REG32 RAW_STATUS87     : 1;
  __REG32 RAW_STATUS88     : 1;
  __REG32 RAW_STATUS89     : 1;
  __REG32 RAW_STATUS90     : 1;
  __REG32                  : 5;
} __aintc_srsr2_bits;

/* AINTC System Interrupt Status Enabled/Clear Register 0 (SECR0) */
typedef struct {
  __REG32 ENBL_STATUS0      : 1;
  __REG32 ENBL_STATUS1      : 1;
  __REG32 ENBL_STATUS2      : 1;
  __REG32 ENBL_STATUS3      : 1;
  __REG32 ENBL_STATUS4      : 1;
  __REG32 ENBL_STATUS5      : 1;
  __REG32 ENBL_STATUS6      : 1;
  __REG32 ENBL_STATUS7      : 1;
  __REG32 ENBL_STATUS8      : 1;
  __REG32 ENBL_STATUS9      : 1;
  __REG32 ENBL_STATUS10     : 1;
  __REG32 ENBL_STATUS11     : 1;
  __REG32 ENBL_STATUS12     : 1;
  __REG32 ENBL_STATUS13     : 1;
  __REG32 ENBL_STATUS14     : 1;
  __REG32 ENBL_STATUS15     : 1;
  __REG32 ENBL_STATUS16     : 1;
  __REG32 ENBL_STATUS17     : 1;
  __REG32 ENBL_STATUS18     : 1;
  __REG32 ENBL_STATUS19     : 1;
  __REG32 ENBL_STATUS20     : 1;
  __REG32 ENBL_STATUS21     : 1;
  __REG32 ENBL_STATUS22     : 1;
  __REG32 ENBL_STATUS23     : 1;
  __REG32 ENBL_STATUS24     : 1;
  __REG32 ENBL_STATUS25     : 1;
  __REG32 ENBL_STATUS26     : 1;
  __REG32 ENBL_STATUS27     : 1;
  __REG32 ENBL_STATUS28     : 1;
  __REG32 ENBL_STATUS29     : 1;
  __REG32 ENBL_STATUS30     : 1;
  __REG32 ENBL_STATUS31     : 1;
} __aintc_secr0_bits;

/* AINTC System Interrupt Status Enabled/Clear Register 1 (SECR1) */
typedef struct {
  __REG32 ENBL_STATUS32     : 1;
  __REG32 ENBL_STATUS33     : 1;
  __REG32 ENBL_STATUS34     : 1;
  __REG32 ENBL_STATUS35     : 1;
  __REG32 ENBL_STATUS36     : 1;
  __REG32 ENBL_STATUS37     : 1;
  __REG32 ENBL_STATUS38     : 1;
  __REG32 ENBL_STATUS39     : 1;
  __REG32 ENBL_STATUS40     : 1;
  __REG32 ENBL_STATUS41     : 1;
  __REG32 ENBL_STATUS42     : 1;
  __REG32 ENBL_STATUS43     : 1;
  __REG32 ENBL_STATUS44     : 1;
  __REG32 ENBL_STATUS45     : 1;
  __REG32 ENBL_STATUS46     : 1;
  __REG32 ENBL_STATUS47     : 1;
  __REG32 ENBL_STATUS48     : 1;
  __REG32 ENBL_STATUS49     : 1;
  __REG32 ENBL_STATUS50     : 1;
  __REG32 ENBL_STATUS51     : 1;
  __REG32 ENBL_STATUS52     : 1;
  __REG32 ENBL_STATUS53     : 1;
  __REG32 ENBL_STATUS54     : 1;
  __REG32 ENBL_STATUS55     : 1;
  __REG32 ENBL_STATUS56     : 1;
  __REG32 ENBL_STATUS57     : 1;
  __REG32 ENBL_STATUS58     : 1;
  __REG32 ENBL_STATUS59     : 1;
  __REG32 ENBL_STATUS60     : 1;
  __REG32 ENBL_STATUS61     : 1;
  __REG32 ENBL_STATUS62     : 1;
  __REG32 ENBL_STATUS63     : 1;
} __aintc_secr1_bits;

/* AINTC System Interrupt Status Enabled/Clear Register 2 (SECR2) */
typedef struct {
  __REG32 ENBL_STATUS64     : 1;
  __REG32 ENBL_STATUS65     : 1;
  __REG32 ENBL_STATUS66     : 1;
  __REG32 ENBL_STATUS67     : 1;
  __REG32 ENBL_STATUS68     : 1;
  __REG32 ENBL_STATUS69     : 1;
  __REG32 ENBL_STATUS70     : 1;
  __REG32 ENBL_STATUS71     : 1;
  __REG32 ENBL_STATUS72     : 1;
  __REG32 ENBL_STATUS73     : 1;
  __REG32 ENBL_STATUS74     : 1;
  __REG32 ENBL_STATUS75     : 1;
  __REG32 ENBL_STATUS76     : 1;
  __REG32 ENBL_STATUS77     : 1;
  __REG32 ENBL_STATUS78     : 1;
  __REG32 ENBL_STATUS79     : 1;
  __REG32 ENBL_STATUS80     : 1;
  __REG32 ENBL_STATUS81     : 1;
  __REG32 ENBL_STATUS82     : 1;
  __REG32 ENBL_STATUS83     : 1;
  __REG32 ENBL_STATUS84     : 1;
  __REG32 ENBL_STATUS85     : 1;
  __REG32 ENBL_STATUS86     : 1;
  __REG32 ENBL_STATUS87     : 1;
  __REG32 ENBL_STATUS88     : 1;
  __REG32 ENBL_STATUS89     : 1;
  __REG32 ENBL_STATUS90     : 1;
  __REG32                   : 5;
} __aintc_secr2_bits;

/* AINTC System Interrupt Enable Set Register 0 (ESR0) */
typedef struct {
  __REG32 ENABLE0      : 1;
  __REG32 ENABLE1      : 1;
  __REG32 ENABLE2      : 1;
  __REG32 ENABLE3      : 1;
  __REG32 ENABLE4      : 1;
  __REG32 ENABLE5      : 1;
  __REG32 ENABLE6      : 1;
  __REG32 ENABLE7      : 1;
  __REG32 ENABLE8      : 1;
  __REG32 ENABLE9      : 1;
  __REG32 ENABLE10     : 1;
  __REG32 ENABLE11     : 1;
  __REG32 ENABLE12     : 1;
  __REG32 ENABLE13     : 1;
  __REG32 ENABLE14     : 1;
  __REG32 ENABLE15     : 1;
  __REG32 ENABLE16     : 1;
  __REG32 ENABLE17     : 1;
  __REG32 ENABLE18     : 1;
  __REG32 ENABLE19     : 1;
  __REG32 ENABLE20     : 1;
  __REG32 ENABLE21     : 1;
  __REG32 ENABLE22     : 1;
  __REG32 ENABLE23     : 1;
  __REG32 ENABLE24     : 1;
  __REG32 ENABLE25     : 1;
  __REG32 ENABLE26     : 1;
  __REG32 ENABLE27     : 1;
  __REG32 ENABLE28     : 1;
  __REG32 ENABLE29     : 1;
  __REG32 ENABLE30     : 1;
  __REG32 ENABLE31     : 1;
} __aintc_esr0_bits;

/* AINTC System Interrupt Enable Set Register 1 (ESR1) */
typedef struct {
  __REG32 ENABLE32     : 1;
  __REG32 ENABLE33     : 1;
  __REG32 ENABLE34     : 1;
  __REG32 ENABLE35     : 1;
  __REG32 ENABLE36     : 1;
  __REG32 ENABLE37     : 1;
  __REG32 ENABLE38     : 1;
  __REG32 ENABLE39     : 1;
  __REG32 ENABLE40     : 1;
  __REG32 ENABLE41     : 1;
  __REG32 ENABLE42     : 1;
  __REG32 ENABLE43     : 1;
  __REG32 ENABLE44     : 1;
  __REG32 ENABLE45     : 1;
  __REG32 ENABLE46     : 1;
  __REG32 ENABLE47     : 1;
  __REG32 ENABLE48     : 1;
  __REG32 ENABLE49     : 1;
  __REG32 ENABLE50     : 1;
  __REG32 ENABLE51     : 1;
  __REG32 ENABLE52     : 1;
  __REG32 ENABLE53     : 1;
  __REG32 ENABLE54     : 1;
  __REG32 ENABLE55     : 1;
  __REG32 ENABLE56     : 1;
  __REG32 ENABLE57     : 1;
  __REG32 ENABLE58     : 1;
  __REG32 ENABLE59     : 1;
  __REG32 ENABLE60     : 1;
  __REG32 ENABLE61     : 1;
  __REG32 ENABLE62     : 1;
  __REG32 ENABLE63     : 1;
} __aintc_esr1_bits;

/* AINTC System Interrupt Enable Set Register 2 (ESR2) */
typedef struct {
  __REG32 ENABLE64     : 1;
  __REG32 ENABLE65     : 1;
  __REG32 ENABLE66     : 1;
  __REG32 ENABLE67     : 1;
  __REG32 ENABLE68     : 1;
  __REG32 ENABLE69     : 1;
  __REG32 ENABLE70     : 1;
  __REG32 ENABLE71     : 1;
  __REG32 ENABLE72     : 1;
  __REG32 ENABLE73     : 1;
  __REG32 ENABLE74     : 1;
  __REG32 ENABLE75     : 1;
  __REG32 ENABLE76     : 1;
  __REG32 ENABLE77     : 1;
  __REG32 ENABLE78     : 1;
  __REG32 ENABLE79     : 1;
  __REG32 ENABLE80     : 1;
  __REG32 ENABLE81     : 1;
  __REG32 ENABLE82     : 1;
  __REG32 ENABLE83     : 1;
  __REG32 ENABLE84     : 1;
  __REG32 ENABLE85     : 1;
  __REG32 ENABLE86     : 1;
  __REG32 ENABLE87     : 1;
  __REG32 ENABLE88     : 1;
  __REG32 ENABLE89     : 1;
  __REG32 ENABLE90     : 1;
  __REG32              : 5;
} __aintc_esr2_bits;

/* AINTC System Interrupt Enable Clear Register 0 (ECR0) */
typedef struct {
  __REG32 DISABLE0      : 1;
  __REG32 DISABLE1      : 1;
  __REG32 DISABLE2      : 1;
  __REG32 DISABLE3      : 1;
  __REG32 DISABLE4      : 1;
  __REG32 DISABLE5      : 1;
  __REG32 DISABLE6      : 1;
  __REG32 DISABLE7      : 1;
  __REG32 DISABLE8      : 1;
  __REG32 DISABLE9      : 1;
  __REG32 DISABLE10     : 1;
  __REG32 DISABLE11     : 1;
  __REG32 DISABLE12     : 1;
  __REG32 DISABLE13     : 1;
  __REG32 DISABLE14     : 1;
  __REG32 DISABLE15     : 1;
  __REG32 DISABLE16     : 1;
  __REG32 DISABLE17     : 1;
  __REG32 DISABLE18     : 1;
  __REG32 DISABLE19     : 1;
  __REG32 DISABLE20     : 1;
  __REG32 DISABLE21     : 1;
  __REG32 DISABLE22     : 1;
  __REG32 DISABLE23     : 1;
  __REG32 DISABLE24     : 1;
  __REG32 DISABLE25     : 1;
  __REG32 DISABLE26     : 1;
  __REG32 DISABLE27     : 1;
  __REG32 DISABLE28     : 1;
  __REG32 DISABLE29     : 1;
  __REG32 DISABLE30     : 1;
  __REG32 DISABLE31     : 1;
} __aintc_ecr0_bits;

/* AINTC System Interrupt Enable Clear Register 1 (ECR1) */
typedef struct {
  __REG32 DISABLE32     : 1;
  __REG32 DISABLE33     : 1;
  __REG32 DISABLE34     : 1;
  __REG32 DISABLE35     : 1;
  __REG32 DISABLE36     : 1;
  __REG32 DISABLE37     : 1;
  __REG32 DISABLE38     : 1;
  __REG32 DISABLE39     : 1;
  __REG32 DISABLE40     : 1;
  __REG32 DISABLE41     : 1;
  __REG32 DISABLE42     : 1;
  __REG32 DISABLE43     : 1;
  __REG32 DISABLE44     : 1;
  __REG32 DISABLE45     : 1;
  __REG32 DISABLE46     : 1;
  __REG32 DISABLE47     : 1;
  __REG32 DISABLE48     : 1;
  __REG32 DISABLE49     : 1;
  __REG32 DISABLE50     : 1;
  __REG32 DISABLE51     : 1;
  __REG32 DISABLE52     : 1;
  __REG32 DISABLE53     : 1;
  __REG32 DISABLE54     : 1;
  __REG32 DISABLE55     : 1;
  __REG32 DISABLE56     : 1;
  __REG32 DISABLE57     : 1;
  __REG32 DISABLE58     : 1;
  __REG32 DISABLE59     : 1;
  __REG32 DISABLE60     : 1;
  __REG32 DISABLE61     : 1;
  __REG32 DISABLE62     : 1;
  __REG32 DISABLE63     : 1;
} __aintc_ecr1_bits;

/* AINTC System Interrupt Enable Clear Register 2 (ECR2) */
typedef struct {
  __REG32 DISABLE64     : 1;
  __REG32 DISABLE65     : 1;
  __REG32 DISABLE66     : 1;
  __REG32 DISABLE67     : 1;
  __REG32 DISABLE68     : 1;
  __REG32 DISABLE69     : 1;
  __REG32 DISABLE70     : 1;
  __REG32 DISABLE71     : 1;
  __REG32 DISABLE72     : 1;
  __REG32 DISABLE73     : 1;
  __REG32 DISABLE74     : 1;
  __REG32 DISABLE75     : 1;
  __REG32 DISABLE76     : 1;
  __REG32 DISABLE77     : 1;
  __REG32 DISABLE78     : 1;
  __REG32 DISABLE79     : 1;
  __REG32 DISABLE80     : 1;
  __REG32 DISABLE81     : 1;
  __REG32 DISABLE82     : 1;
  __REG32 DISABLE83     : 1;
  __REG32 DISABLE84     : 1;
  __REG32 DISABLE85     : 1;
  __REG32 DISABLE86     : 1;
  __REG32 DISABLE87     : 1;
  __REG32 DISABLE88     : 1;
  __REG32 DISABLE89     : 1;
  __REG32 DISABLE90     : 1;
  __REG32               : 5;
} __aintc_ecr2_bits;

/* AINTC Channel Map Registers (CMR0) */
typedef struct {
  __REG32 CHNL_0        : 8;
  __REG32 CHNL_1        : 8;
  __REG32 CHNL_2        : 8;
  __REG32 CHNL_3        : 8;
} __aintc_cmr0_bits;

/* AINTC Channel Map Registers (CMR1) */
typedef struct {
  __REG32 CHNL_4        : 8;
  __REG32 CHNL_5        : 8;
  __REG32 CHNL_6        : 8;
  __REG32 CHNL_7        : 8;
} __aintc_cmr1_bits;

/* AINTC Channel Map Registers (CMR2) */
typedef struct {
  __REG32 CHNL_8        : 8;
  __REG32 CHNL_9        : 8;
  __REG32 CHNL_10       : 8;
  __REG32 CHNL_11       : 8;
} __aintc_cmr2_bits;

/* AINTC Channel Map Registers (CMR3) */
typedef struct {
  __REG32 CHNL_12       : 8;
  __REG32 CHNL_13       : 8;
  __REG32 CHNL_14       : 8;
  __REG32 CHNL_15       : 8;
} __aintc_cmr3_bits;

/* AINTC Channel Map Registers (CMR4) */
typedef struct {
  __REG32 CHNL_16       : 8;
  __REG32 CHNL_17       : 8;
  __REG32 CHNL_18       : 8;
  __REG32 CHNL_19       : 8;
} __aintc_cmr4_bits;

/* AINTC Channel Map Registers (CMR5) */
typedef struct {
  __REG32 CHNL_20       : 8;
  __REG32 CHNL_21       : 8;
  __REG32 CHNL_22       : 8;
  __REG32 CHNL_23       : 8;
} __aintc_cmr5_bits;

/* AINTC Channel Map Registers (CMR6) */
typedef struct {
  __REG32 CHNL_24       : 8;
  __REG32 CHNL_25       : 8;
  __REG32 CHNL_26       : 8;
  __REG32 CHNL_27       : 8;
} __aintc_cmr6_bits;

/* AINTC Channel Map Registers (CMR7) */
typedef struct {
  __REG32 CHNL_28       : 8;
  __REG32 CHNL_29       : 8;
  __REG32 CHNL_30       : 8;
  __REG32 CHNL_31       : 8;
} __aintc_cmr7_bits;

/* AINTC Channel Map Registers (CMR8) */
typedef struct {
  __REG32 CHNL_32       : 8;
  __REG32 CHNL_33       : 8;
  __REG32 CHNL_34       : 8;
  __REG32 CHNL_35       : 8;
} __aintc_cmr8_bits;

/* AINTC Channel Map Registers (CMR9) */
typedef struct {
  __REG32 CHNL_36       : 8;
  __REG32 CHNL_37       : 8;
  __REG32 CHNL_38       : 8;
  __REG32 CHNL_39       : 8;
} __aintc_cmr9_bits;

/* AINTC Channel Map Registers (CMR10) */
typedef struct {
  __REG32 CHNL_40       : 8;
  __REG32 CHNL_41       : 8;
  __REG32 CHNL_42       : 8;
  __REG32 CHNL_43       : 8;
} __aintc_cmr10_bits;

/* AINTC Channel Map Registers (CMR11) */
typedef struct {
  __REG32 CHNL_44       : 8;
  __REG32 CHNL_45       : 8;
  __REG32 CHNL_46       : 8;
  __REG32 CHNL_47       : 8;
} __aintc_cmr11_bits;

/* AINTC Channel Map Registers (CMR12) */
typedef struct {
  __REG32 CHNL_48       : 8;
  __REG32 CHNL_49       : 8;
  __REG32 CHNL_50       : 8;
  __REG32 CHNL_51       : 8;
} __aintc_cmr12_bits;

/* AINTC Channel Map Registers (CMR13) */
typedef struct {
  __REG32 CHNL_52       : 8;
  __REG32 CHNL_53       : 8;
  __REG32 CHNL_54       : 8;
  __REG32 CHNL_55       : 8;
} __aintc_cmr13_bits;

/* AINTC Channel Map Registers (CMR14) */
typedef struct {
  __REG32 CHNL_56       : 8;
  __REG32 CHNL_57       : 8;
  __REG32 CHNL_58       : 8;
  __REG32 CHNL_59       : 8;
} __aintc_cmr14_bits;

/* AINTC Channel Map Registers (CMR15) */
typedef struct {
  __REG32 CHNL_60       : 8;
  __REG32 CHNL_61       : 8;
  __REG32 CHNL_62       : 8;
  __REG32 CHNL_63       : 8;
} __aintc_cmr15_bits;

/* AINTC Channel Map Registers (CMR16) */
typedef struct {
  __REG32 CHNL_64       : 8;
  __REG32 CHNL_65       : 8;
  __REG32 CHNL_66       : 8;
  __REG32 CHNL_67       : 8;
} __aintc_cmr16_bits;

/* AINTC Channel Map Registers (CMR17) */
typedef struct {
  __REG32 CHNL_68       : 8;
  __REG32 CHNL_69       : 8;
  __REG32 CHNL_70       : 8;
  __REG32 CHNL_71       : 8;
} __aintc_cmr17_bits;

/* AINTC Channel Map Registers (CMR18) */
typedef struct {
  __REG32 CHNL_72       : 8;
  __REG32 CHNL_73       : 8;
  __REG32 CHNL_74       : 8;
  __REG32 CHNL_75       : 8;
} __aintc_cmr18_bits;

/* AINTC Channel Map Registers (CMR19) */
typedef struct {
  __REG32 CHNL_76       : 8;
  __REG32 CHNL_77       : 8;
  __REG32 CHNL_78       : 8;
  __REG32 CHNL_79       : 8;
} __aintc_cmr19_bits;

/* AINTC Channel Map Registers (CMR20) */
typedef struct {
  __REG32 CHNL_80       : 8;
  __REG32 CHNL_81       : 8;
  __REG32 CHNL_82       : 8;
  __REG32 CHNL_83       : 8;
} __aintc_cmr20_bits;

/* AINTC Channel Map Registers (CMR21) */
typedef struct {
  __REG32 CHNL_84       : 8;
  __REG32 CHNL_85       : 8;
  __REG32 CHNL_86       : 8;
  __REG32 CHNL_87       : 8;
} __aintc_cmr21_bits;

/* AINTC Channel Map Registers (CMR22) */
typedef struct {
  __REG32 CHNL_88       : 8;
  __REG32 CHNL_89       : 8;
  __REG32 CHNL_90       : 8;
  __REG32               : 8;
} __aintc_cmr22_bits;

/* AINTC Host Interrupt Prioritized Index Register 0/1 (HIPIR0/1) */
typedef struct {
  __REG32 PRI_INDX      :10;
  __REG32               :21;
  __REG32 NONE          : 1;
} __aintc_hipir_bits;

/* AINTC Host Interrupt Nesting Level Register 0/1 (HINLR0/1) */
typedef struct {
  __REG32 NEST_LVL      : 9;
  __REG32               :22;
  __REG32 OVERRIDE      : 1;
} __aintc_hinlr_bits;

/* AINTC Host Interrupt Enable Register (HIER) */
typedef struct {
  __REG32 FIQ           : 1;
  __REG32 IRQ           : 1;
  __REG32               :30;
} __aintc_hier_bits;

/* USB OTG Control Register (CTRLR) */
typedef struct {
  __REG32 RESET         : 1;
  __REG32 CLKFACK       : 1;
  __REG32               : 1;
  __REG32 UINT          : 1;
  __REG32 RNDIS         : 1;
  __REG32               :27;
} __usb_ctrlr_bits;

/* USB OTG Status Register (STATR) */
typedef struct {
  __REG32 DRVVBUS       : 1;
  __REG32               :31;
} __usb_statr_bits;

/* USB OTG Emulation Register (EMUR) */
typedef struct {
  __REG32 FREERUN       : 1;
  __REG32 SOFT          : 1;
  __REG32 RT_SEL        : 1;
  __REG32               :29;
} __usb_emur_bits;

/* USB OTG Mode Register (MODE) */
typedef struct {
  __REG32 TX1_MODE      : 2;
  __REG32               : 2;
  __REG32 TX2_MODE      : 2;
  __REG32               : 2;
  __REG32 TX3_MODE      : 2;
  __REG32               : 2;
  __REG32 TX4_MODE      : 2;
  __REG32               : 2;
  __REG32 RX1_MODE      : 2;
  __REG32               : 2;
  __REG32 RX2_MODE      : 2;
  __REG32               : 2;
  __REG32 RX3_MODE      : 2;
  __REG32               : 2;
  __REG32 RX4_MODE      : 2;
  __REG32               : 2;
} __usb_mode_bits;

/* USB OTG Auto Request Register (AUTOREQ) */
typedef struct {
  __REG32 RX1_AUTREQ    : 2;
  __REG32 RX2_AUTREQ    : 2;
  __REG32 RX3_AUTREQ    : 2;
  __REG32 RX4_AUTREQ    : 2;
  __REG32               :24;
} __usb_autoreq_bits;

/* USB OTG Teardown Register (TEARDOWN) */
typedef struct {
  __REG32               : 1;
  __REG32 RX_TDOWN      : 4;
  __REG32               :12;
  __REG32 TX_TDOWN      : 4;
  __REG32               :11;
} __usb_teardown_bits;

/* USB OTG Interrupt Source Register (INTSRCR) */
/* USB OTG Interrupt Source Set Register (INTSETR) */
/* USB OTG Interrupt Source Clear Register (INTCLRR) */
/* USB OTG Interrupt Mask Register (INTMSKR) */
/* USB OTG Interrupt Mask Set Register (INTMSKSETR) */
/* USB OTG Interrupt Mask Clear Register (INTMSKCLRR) */
/* USB OTG Interrupt Source Masked Register (INTMASKEDR) */
typedef struct {
  __REG32 EP0           : 1;
  __REG32 TXEP1         : 1;
  __REG32 TXEP2         : 1;
  __REG32 TXEP3         : 1;
  __REG32 TXEP4         : 1;
  __REG32               : 4;
  __REG32 RXEP1         : 1;
  __REG32 RXEP2         : 1;
  __REG32 RXEP3         : 1;
  __REG32 RXEP4         : 1;
  __REG32               : 3;
  __REG32 USB           : 8;
  __REG32               : 8;
} __usb_intsrcr_bits;

/* USB OTG End of Interrupt Register (EOIR) */
typedef struct {
  __REG32 EOI_VECTOR    : 8;
  __REG32               :24;
} __usb_eoir_bits;

/* USB OTG Generic RNDIS EP1 Size Register (GENRNDISSZ1) */
typedef struct {
  __REG32 EP1_SIZE      :17;
  __REG32               :15;
} __usb_genrndissz1_bits;

/* USB OTG Generic RNDIS EP2 Size Register (GENRNDISSZ2) */
typedef struct {
  __REG32 EP2_SIZE      :17;
  __REG32               :15;
} __usb_genrndissz2_bits;

/* USB OTG Generic RNDIS EP3 Size Register (GENRNDISSZ3) */
typedef struct {
  __REG32 EP3_SIZE      :17;
  __REG32               :15;
} __usb_genrndissz3_bits;

/* USB OTG Generic RNDIS EP4 Size Register (GENRNDISSZ4) */
typedef struct {
  __REG32 EP4_SIZE      :17;
  __REG32               :15;
} __usb_genrndissz4_bits;

/* USB OTG Function Address Register (FADDR) */
typedef struct {
  __REG8 FUNCADDR      : 7;
  __REG8               : 1;
} __usb_faddr_bits;

/* USB OTG Power Management Register (POWER) */
typedef struct {
  __REG8 ENSUSPM       : 1;
  __REG8 SUSPENDM      : 1;
  __REG8 RESUME        : 1;
  __REG8 RESET         : 1;
  __REG8 HSMODE        : 1;
  __REG8 HSEN          : 1;
  __REG8 SOFTCONN      : 1;
  __REG8 ISOUPDATE     : 1;
} __usb_power_bits;

/* USB OTG Interrupt Register for Endpoint 0 Plus Transmit Endpoints 1 to 4 (INTRTX) */
/* USB OTG Interrupt Enable Register for INTRTX (INTRTXE) */
typedef struct {
  __REG16 EP0           : 1;
  __REG16 EP1TX         : 1;
  __REG16 EP2TX         : 1;
  __REG16 EP3TX         : 1;
  __REG16 EP4TX         : 1;
  __REG16               :11;
} __usb_intrtx_bits;

/* USB OTG Interrupt Register for Receive Endpoints 1 to 4 (INTRRX) */
/* USB OTG Interrupt Enable Register for INTRRX (INTRRXE) */
typedef struct {
  __REG16               : 1;
  __REG16 EP1RX         : 1;
  __REG16 EP2RX         : 1;
  __REG16 EP3RX         : 1;
  __REG16 EP4RX         : 1;
  __REG16               :11;
} __usb_intrrx_bits;

/* USB OTG Interrupt Register for Common USB Interrupts (INTRUSB) */
/* USB OTG Interrupt Enable Register for INTRUSB (INTRUSBE) */
typedef struct {
  __REG8 SUSPEND       : 1;
  __REG8 RESUME        : 1;
  __REG8 RESET_BABBLE  : 1;
  __REG8 SOF           : 1;
  __REG8 CONN          : 1;
  __REG8 DISCON        : 1;
  __REG8 SESSREQ       : 1;
  __REG8 VBUSERR       : 1;
} __usb_intrusb_bits;

/* USB OTG Frame Number Register (FRAME) */
typedef struct {
  __REG16 FRAMENUMBER   :11;
  __REG16               : 5;
} __usb_frame_bits;

/* USB OTG Index Register for Selecting the Endpoint Status and Control Registers (INDEX) */
typedef struct {
  __REG8 EPSEL         : 4;
  __REG8               : 4;
} __usb_index_bits;

/* USB OTG Register to Enable the USB 2.0 Test Modes (TESTMODE) */
typedef struct {
  __REG8 TEST_SE0_NAK  : 1;
  __REG8 TEST_J        : 1;
  __REG8 TEST_K        : 1;
  __REG8 TEST_PACKET   : 1;
  __REG8 FORCE_HS      : 1;
  __REG8 FORCE_FS      : 1;
  __REG8 FIFO_ACCESS   : 1;
  __REG8 FORCE_HOST    : 1;
} __usb_testmode_bits;

/* USB OTG Maximum Packet Size for Peripheral/Host Transmit Endpoint (TXMAXP) */
typedef struct {
  __REG16 MAXPAYLOAD    :11;
  __REG16               : 5;
} __usb_txmaxp_bits;

/* USB OTG Control Status Register for Endpoint 0 in Peripheral Mode (PERI_CSR0) */
/* USB OTG Control Status Register for Endpoint 0 in Host Mode (HOST_CSR0) */
/* USB OTG Control Status Register for Peripheral Transmit Endpoint (PERI_TXCSR) */
/* USB OTG Control Status Register for Host Transmit Endpoint (HOST_TXCSR) */
typedef union {
  /* USB0_PERI_CSR0 */
  struct {
  __REG16 RXPKTRDY      : 1;
  __REG16 TXPKTRDY      : 1;
  __REG16 SENTSTALL     : 1;
  __REG16 DATAEND       : 1;
  __REG16 SETUPEND      : 1;
  __REG16 SENDSTALL     : 1;
  __REG16 SERV_RXPKTRDY : 1;
  __REG16 SERV_SETUPEND : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16               : 7;
  };
  /* USB0_HOST_CSR0 */
  struct {
  __REG16 RXPKTRDY      : 1;
  __REG16 TXPKTRDY      : 1;
  __REG16 RXSTALL       : 1;
  __REG16 SETUPPKT      : 1;
  __REG16 ERROR         : 1;
  __REG16 REQPKT        : 1;
  __REG16 STATUSPKT     : 1;
  __REG16 NAK_TIMEOUT   : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16 DATATOG       : 1;
  __REG16 DATATOGWREN   : 1;
  __REG16 DISPING       : 1;
  __REG16               : 4;
  } host_csr0;
  /* USB0_PERI_TXCSR */
  struct {
  __REG16 TXPKTRDY      : 1;
  __REG16 FIFONOTEMPTY  : 1;
  __REG16 UNDERRUN      : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16 SENDSTALL     : 1;
  __REG16 SENTSTALL     : 1;
  __REG16 CLRDATATOG    : 1;
  __REG16               : 3;
  __REG16 DMAMODE       : 1;
  __REG16 FRCDATATOG    : 1;
  __REG16 DMAEN         : 1;
  __REG16 MODE          : 1;
  __REG16 ISO           : 1;
  __REG16 AUTOSET       : 1;
  } peri_txcsr;
  /* USB0_HOST_TXCSR */
  struct {
  __REG16 TXPKTRDY     : 1;
  __REG16 FIFONOTEMPTY : 1;
  __REG16 ERROR        : 1;
  __REG16 FLUSHFIFO    : 1;
  __REG16 SETUPPKT     : 1;
  __REG16 RXSTALL      : 1;
  __REG16 CLRDATATOG   : 1;
  __REG16 NAK_TIMEOUT  : 1;
  __REG16 DATATOG      : 1;
  __REG16 DATATOGWREN  : 1;
  __REG16 DMAMODE      : 1;
  __REG16 FRCDATATOG   : 1;
  __REG16 DMAEN        : 1;
  __REG16 MODE         : 1;
  __REG16              : 1;
  __REG16 AUTOSET      : 1;
  } host_txcsr;
} __usb_peri_csr0i_bits;

/* USB OTG Control Status Register for Peripheral Transmit Endpoint (PERI_TXCSR) */
/* USB OTG Control Status Register for Host Transmit Endpoint (HOST_TXCSR) */
typedef union {
  /* USB0_PERI_TXCSR_EPx */
  struct {
  __REG16 TXPKTRDY      : 1;
  __REG16 FIFONOTEMPTY  : 1;
  __REG16 UNDERRUN      : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16 SENDSTALL     : 1;
  __REG16 SENTSTALL     : 1;
  __REG16 CLRDATATOG    : 1;
  __REG16               : 3;
  __REG16 DMAMODE       : 1;
  __REG16 FRCDATATOG    : 1;
  __REG16 DMAEN         : 1;
  __REG16 MODE          : 1;
  __REG16 ISO           : 1;
  __REG16 AUTOSET       : 1;
  };
  /* USB0_HOST_TXCSR_EPx */
  struct {
  __REG16 TXPKTRDY     : 1;
  __REG16 FIFONOTEMPTY : 1;
  __REG16 ERROR        : 1;
  __REG16 FLUSHFIFO    : 1;
  __REG16 SETUPPKT     : 1;
  __REG16 RXSTALL      : 1;
  __REG16 CLRDATATOG   : 1;
  __REG16 NAK_TIMEOUT  : 1;
  __REG16 DATATOG      : 1;
  __REG16 DATATOGWREN  : 1;
  __REG16 DMAMODE      : 1;
  __REG16 FRCDATATOG   : 1;
  __REG16 DMAEN        : 1;
  __REG16 MODE         : 1;
  __REG16              : 1;
  __REG16 AUTOSET      : 1;
  } host_txcsr;
} __usb_peri_txcsr_bits;

/* USB OTG Control Status Register for Endpoint 0 in Peripheral Mode (PERI_CSR0) */
/* USB OTG Control Status Register for Endpoint 0 in Host Mode (HOST_CSR0) */
typedef union {
  /* USB0_PERI_CSR0 */
  struct {
  __REG16 RXPKTRDY      : 1;
  __REG16 TXPKTRDY      : 1;
  __REG16 SENTSTALL     : 1;
  __REG16 DATAEND       : 1;
  __REG16 SETUPEND      : 1;
  __REG16 SENDSTALL     : 1;
  __REG16 SERV_RXPKTRDY : 1;
  __REG16 SERV_SETUPEND : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16               : 7;
  };
  /* USB0_HOST_CSR0_EP0 */
  struct {
  __REG16 RXPKTRDY      : 1;
  __REG16 TXPKTRDY      : 1;
  __REG16 RXSTALL       : 1;
  __REG16 SETUPPKT      : 1;
  __REG16 ERROR         : 1;
  __REG16 REQPKT        : 1;
  __REG16 STATUSPKT     : 1;
  __REG16 NAK_TIMEOUT   : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16 DATATOG       : 1;
  __REG16 DATATOGWREN   : 1;
  __REG16 DISPING       : 1;
  __REG16               : 4;
  } host_csr0_ep0;
} __usb_peri_csr0_bits;

/* USB Count 0 Register (COUNT0) */
typedef struct {
  __REG16 EP0RXCOUNT    : 7;
  __REG16               : 9;
} __usb_count0_bits;

/* USB Type Register (Host mode only) (HOST_TYPE0) */
typedef struct {
  __REG8               : 6;
  __REG8 SPEED         : 2;
} __usb_host_type0_bits;

/* USB NAKLimit0 Register (Host mode only) (HOST_NAKLIMIT0) */
typedef struct {
  __REG8 EP0NAKLIMIT   : 5;
  __REG8               : 3;
} __usb_host_naklimit0_bits;

/* USB Maximum Packet Size for Peripheral Host Receive Endpoint (RXMAXP) */
typedef struct {
  __REG16 MAXPAYLOAD    :11;
  __REG16               : 5;
} __usb_rxmaxp_bits;

/* USB Control Status Register for Peripheral Receive Endpoint (PERI_RXCSR) */
/* USB Control Status Register for Host Receive Endpoint (HOST_RXCSR) */
typedef union {
/* USB0_PERI_RXCSR */
/* USB0_PERI_RXCSR_EPx */
 struct {
  __REG16 RXPKTRDY      : 1;
  __REG16 FIFOFULL      : 1;
  __REG16 OVERRUN       : 1;
  __REG16 DATAERROR     : 1;
  __REG16 FLUSHFIFO     : 1;
  __REG16 SENDSTALL     : 1;
  __REG16 SENTSTALL     : 1;
  __REG16 CLRDATATOG    : 1;
  __REG16               : 3;
  __REG16 DMAMODE       : 1;
  __REG16 DISNYET       : 1;
  __REG16 DMAEN         : 1;
  __REG16 ISO           : 1;
  __REG16 AUTOCLEAR     : 1;
  };
/* USB0_HOST_RXCSR */
/* USB0_HOST_RXCSR_EPx */
 struct {
  __REG16 RXPKTRDY            : 1;
  __REG16 FIFOFULL            : 1;
  __REG16 ERROR               : 1;
  __REG16 DATAERR_NAKTIMEOUT  : 1;
  __REG16 FLUSHFIFO           : 1;
  __REG16 REQPKT              : 1;
  __REG16 RXSTALL             : 1;
  __REG16 CLRDATATOG          : 1;
  __REG16                     : 1;
  __REG16 DATATOG             : 1;
  __REG16 DATATOGWREN         : 1;
  __REG16 DMAMODE             : 1;
  __REG16 DISNYET             : 1;
  __REG16 DMAEN               : 1;
  __REG16 AUTOREQ             : 1;
  __REG16 AUTOCLEAR           : 1;
  } host_rxcsr;
} __usb_peri_rxcsr_bits;

/* USB Count 0 Register (COUNT0) */
/* USB Receive Count Register (RXCOUNT) */
typedef union {
/* USB0_COUNT0 */
 struct {
  __REG16 EP0RXCOUNT    : 7;
  __REG16               : 9;
  };
/* USB0_RXCOUNT */
 struct {
  __REG16 EPRXCOUNT     :13;
  __REG16               : 3;
  };
} __usb_count0i_bits;

/* USB Receive Count Register (RXCOUNT) */
typedef struct {
  __REG16 EPRXCOUNT     :13;
  __REG16               : 3;
} __usb_rxcount_bits;

/* USB Type Register (Host mode only) (HOST_TYPE0) */
/* USB Transmit Type Register (Host mode only) (HOST_TXTYPE) */
typedef struct {
  __REG8  TENDPN   : 4;
  __REG8  PROT     : 2;
  __REG8  SPEED    : 2;
} __usb_host_txtype_bits;

/* USB NAKLimit0 Register (Host mode only) (HOST_NAKLIMIT0) */
typedef struct {
  __REG8  USB0_HOST_TXINTERVALI : 5;
  __REG8                        : 3;
} __usb_host_naklimit0i_bits;

/* USB Receive Type Register (Host mode only) (HOST_RXTYPE) */
typedef struct {
  __REG8  RENDPN                : 4;
  __REG8  PROT                  : 2;
  __REG8  SPEED                 : 2;
} __usb_host_rxtype_bits;

/* USB Configuration Data Register (CONFIGDATA) */
typedef struct {
  __REG8  UTMIDATAWIDTH         : 1;
  __REG8  SOFTCONE              : 1;
  __REG8  DYNFIFO               : 1;
  __REG8  HBTXE                 : 1;
  __REG8  HBRXE                 : 1;
  __REG8  BIGENDIAN             : 1;
  __REG8  MPTXE                 : 1;
  __REG8  MPRXE                 : 1;
} __usb_configdata_bits;

/* USB Device Control Register (DEVCTL) */
typedef struct {
  __REG8  SESSION               : 1;
  __REG8  HOSTREQ               : 1;
  __REG8  HOSTMODE              : 1;
  __REG8  VBUS                  : 2;
  __REG8  LSDEV                 : 1;
  __REG8  FSDEV                 : 1;
  __REG8  BDEVICE               : 1;
} __usb_devctl_bits;

/* USB Transmit Endpoint FIFO Size (TXFIFOSZ) */
/* USB Receive Endpoint FIFO Size (RXFIFOSZ) */
typedef struct {
  __REG8  SZ                    : 4;
  __REG8  DPB                   : 1;
  __REG8                        : 3;
} __usb_txfifosz_bits;

/* USB Transmit Endpoint FIFO Address (TXFIFOADDR) */
/* USB Receive Endpoint FIFO Address (RXFIFOADDR) */
typedef struct {
  __REG16 ADDR                  :13;
  __REG16                       : 3;
} __usb_txfifoaddr_bits;

/* USB OTG Transmit Function Address (TXFUNCADDR) */
/* USB OTG Receive Function Address (RXFUNCADDR) */
typedef struct {
  __REG8 FUNCADDR      : 7;
  __REG8               : 1;
} __usb_txfuncaddr_h_bits;

/* USB OTG Transmit Hub Address (TXHUBADDR) */
/* USB Receive Hub Address (RXHUBADDR) */
typedef struct {
  __REG8 HUBADDR       : 7;
  __REG8 MULT_TRANS    : 1;
} __usb_txhubaddr_h_bits;

/* USB OTG Transmit Hub Port (TXHUBPORT) */
/* USB OTG Receive Hub Port (RXHUBPORT) */
typedef struct {
  __REG8 HUBPORT       : 7;
  __REG8               : 1;
} __usb_txhubport_h_bits;

/* USB OTG CDMA Teardown Free Descriptor Queue Control Register (TDFDQ) */
typedef struct {
  __REG32 TD_DESC_QNUM  :12;
  __REG32 TD_DESC_QMGR  : 2;
  __REG32               :18;
} __usb_tdfdq_bits;

/* USB OTG CDMA Emulation Control Register (DMAEMU) */
typedef struct {
  __REG32 FREE          : 1;
  __REG32 SOFT          : 1;
  __REG32               :30;
} __usb_dmaemu_bits;

/* USB OTG CDMA Transmit Channel n Global Configuration Registers (TXGCR[0]-TXGCR[3]) */
typedef struct {
  __REG32 TX_DEFAULT_QNUM :12;
  __REG32 TX_DEFAULT_QMGR : 2;
  __REG32                 :16;
  __REG32 TX_TEARDOWN     : 1;
  __REG32 TX_ENABLE       : 1;
} __usb_txgcr_bits;

/* USB OTG CDMA Receive Channel n Global Configuration Registers (RXGCR[0]-RXGCR[3]) */
typedef struct {
  __REG32 RX_DEFAULT_RQ_QNUM    :12;
  __REG32 RX_DEFAULT_RQ_QMGR    : 2;
  __REG32 RX_DEFAULT_DESC_TYPE  : 2;
  __REG32 RX_SOP_OFFSET         : 8;
  __REG32 RX_ERROR_HANDLING     : 1;
  __REG32                       : 5;
  __REG32 RX_TEARDOWN           : 1;
  __REG32 RX_ENABLE             : 1;
} __usb_rxgcr_bits;

/* USB OTG CDMA Receive Channel n Host Packet Configuration Registers A (RXHPCRA[0]-RXHPCRA[3]) */
typedef struct {
  __REG32 RX_HOST_FDQ0_QNUM     :12;
  __REG32 RX_HOST_FDQ0_QMGR     : 2;
  __REG32                       : 2;
  __REG32 RX_HOST_FDQ1_QNUM     :12;
  __REG32 RX_HOST_FDQ1_QMGR     : 2;
  __REG32                       : 2;
} __usb_rxhpcra_bits;

/* USB OTG CDMA Receive Channel n Host Packet Configuration Registers B (RXHPCRB[0]-RXHPCRB[3]) */
typedef struct {
  __REG32 RX_HOST_FDQ0_QNUM     :12;
  __REG32 RX_HOST_FDQ0_QMGR     : 2;
  __REG32                       : 2;
  __REG32 RX_HOST_FDQ1_QNUM     :12;
  __REG32 RX_HOST_FDQ1_QMGR     : 2;
  __REG32                       : 2;
} __usb_rxhpcrb_bits;

/* USB OTG CDMA Scheduler Control Register (DMA_SCHED_CTRL) */
typedef struct {
  __REG32 LAST_ENTRY            : 8;
  __REG32                       :23;
  __REG32 ENABLE                : 1;
} __usb_dma_sched_ctrl_bits;

/* USB OTG CDMA Scheduler Table Word n Registers (WORD[0]-WORD[63]) */
typedef struct {
  __REG32 ENTRY0_CHANNEL        : 4;
  __REG32                       : 3;
  __REG32 ENTRY0_RXTX           : 1;
  __REG32 ENTRY1_CHANNEL        : 4;
  __REG32                       : 3;
  __REG32 ENTRY1_RXTX           : 1;
  __REG32 ENTRY2_CHANNEL        : 4;
  __REG32                       : 3;
  __REG32 ENTRY2_RXTX           : 1;
  __REG32 ENTRY3_CHANNEL        : 4;
  __REG32                       : 3;
  __REG32 ENTRY3_RXTX           : 1;
} __usb_entry_bits;

/* USB OTG Queue Manager Queue Diversion Register (DIVERSION) */
typedef struct {
  __REG32 SOURCE_QNUM           :14;
  __REG32                       : 2;
  __REG32 DEST_QNUM             :14;
  __REG32                       : 1;
  __REG32 HEAD_TAIL             : 1;
} __usb_diversion_bits;

/* USB OTG Queue Manager Free Descriptor/Buffer Starvation Count Register 0 (FDBSC0) */
typedef struct {
  __REG32 FDBQ0_STARVE_CNT      : 8;
  __REG32 FDBQ1_STARVE_CNT      : 8;
  __REG32 FDBQ2_STARVE_CNT      : 8;
  __REG32 FDBQ3_STARVE_CNT      : 8;
} __usb_fdbsc0_bits;

/* USB OTG Queue Manager Free Descriptor/Buffer Starvation Count Register 1 (FDBSC1) */
typedef struct {
  __REG32 FDBQ4_STARVE_CNT      : 8;
  __REG32 FDBQ5_STARVE_CNT      : 8;
  __REG32 FDBQ6_STARVE_CNT      : 8;
  __REG32 FDBQ7_STARVE_CNT      : 8;
} __usb_fdbsc1_bits;

/* USB OTG Queue Manager Free Descriptor/Buffer Starvation Count Register 2 (FDBSC2) */
typedef struct {
  __REG32 FDBQ8_STARVE_CNT      : 8;
  __REG32 FDBQ9_STARVE_CNT      : 8;
  __REG32 FDBQ10_STARVE_CNT     : 8;
  __REG32 FDBQ11_STARVE_CNT     : 8;
} __usb_fdbsc2_bits;

/* USB OTG Queue Manager Free Descriptor/Buffer Starvation Count Register 3 (FDBSC3) */
typedef struct {
  __REG32 FDBQ12_STARVE_CNT     : 8;
  __REG32 FDBQ13_STARVE_CNT     : 8;
  __REG32 FDBQ14_STARVE_CNT     : 8;
  __REG32 FDBQ15_STARVE_CNT     : 8;
} __usb_fdbsc3_bits;

/* USB OTG Queue Manager Linking RAM Region 0 Size Register (LRAM0SIZE) */
typedef struct {
  __REG32 REGION0_SIZE          :14;
  __REG32                       :18;
} __usb_lram0size_bits;

/* USB OTG Queue Manager Memory Region R Control Registers (QMEMRCTRL[0]-QMEMRCTRL[15]) */
typedef struct {
  __REG32 REG_SIZE              : 3;
  __REG32                       : 5;
  __REG32 DESC_SIZE             : 4;
  __REG32                       : 4;
  __REG32 START_INDEX           :14;
  __REG32                       : 2;
} __usb_qmemrctrl_bits;

/* USB OTG Queue Manager Queue N Control Register D (CTRLD[0]-CTRLD[63]) */
typedef struct {
  __REG32 DESC_SIZE             : 5;
  __REG32 DESC_PTR              :27;
} __usb_ctrld_bits;

/* USB OTG Queue Manager Queue N Status Register A (QSTATA[0]-QSTATA[63]) */
typedef struct {
  __REG32 QUEUE_ENTRY_COUNT     :14;
  __REG32                       :18;
} __usb_qstata_bits;

/* USB OTG Queue Manager Queue N Status Register B (QSTATB[0]-QSTATB[63]) */
typedef struct {
  __REG32 QUEUE_BYTE_COUNT      :28;
  __REG32                       : 4;
} __usb_qstatb_bits;

/* USB OTG Queue Manager Queue N Status Register C (QSTATC[0]-QSTATC[63]) */
typedef struct {
  __REG32 QUEUE_BYTE_COUNT      :28;
  __REG32                       : 4;
} __usb_qstatc_bits;

/* EMIF Asynchronous Wait Cycle Configuration Register (AWCC) */
typedef struct {
  __REG32 MAX_EXT_WAIT  : 8;
  __REG32               : 8;
  __REG32 CS2_WAIT      : 2;
  __REG32 CS3_WAIT      : 2;
  __REG32 CS4_WAIT      : 2;
  __REG32 CS5_WAIT      : 2;
  __REG32               : 4;
  __REG32 WP0           : 1;
  __REG32 WP1           : 1;
  __REG32               : 2;
} __awcc_bits;

/* EMIF SDRAM Configuration Register (SDCR) */
typedef struct {
  __REG32 PAGESIZE      : 3;
  __REG32               : 1;
  __REG32 IBANK         : 3;
  __REG32               : 1;
  __REG32 BIT11_9LOCK   : 1;
  __REG32 CL            : 3;
  __REG32               : 2;
  __REG32 NM            : 1;
  __REG32               :14;
  __REG32 PDWR          : 1;
  __REG32 PD            : 1;
  __REG32 SR            : 1;
} __sdcr_bits;

/* EMIF SDRAM Refresh Control Register (SDRCR) */
typedef struct {
  __REG32 RR            :13;
  __REG32               :19;
} __sdrcr_bits;

/* EMIF SDRAM Asynchronous n Configuration Registers (CE2CFG-CE5CFG) */
typedef struct {
  __REG32 ASIZE         : 2;
  __REG32 TA            : 2;
  __REG32 R_HOLD        : 3;
  __REG32 R_STROBE      : 6;
  __REG32 R_SETUP       : 4;
  __REG32 W_HOLD        : 3;
  __REG32 W_STROBE      : 6;
  __REG32 W_SETUP       : 4;
  __REG32 EW            : 1;
  __REG32 SS            : 1;
} __cecfg_bits;

/* EMIF SDRAM Timing Register (SDTIMR) */
typedef struct {
  __REG32               : 4;
  __REG32 T_RRD         : 3;
  __REG32               : 1;
  __REG32 T_RC          : 4;
  __REG32 T_RAS         : 4;
  __REG32 T_WR          : 3;
  __REG32               : 1;
  __REG32 T_RCD         : 3;
  __REG32               : 1;
  __REG32 T_RP          : 3;
  __REG32 T_RFC         : 5;
} __sdtimr_bits;

/* EMIF SDRAM Self Refresh Exit Timing Register (SDSRETR) */
typedef struct {
  __REG32 T_XS          : 5;
  __REG32               :27;
} __sdsretr_bits;

/* EMIF Interrupt Raw Register (INTRAW) */
typedef struct {
  __REG32 AT            : 1;
  __REG32 LT            : 1;
  __REG32 WR            : 1;
  __REG32               :29;
} __intraw_bits;

/* EMIF Interrupt Mask Register (INTMSK) */
typedef struct {
  __REG32 AT_MASKED     : 1;
  __REG32 LT_MASKED     : 1;
  __REG32 WR_MASKED     : 1;
  __REG32               :29;
} __intmsk_bits;

/* EMIF Interrupt Mask Set Register (INTMSKSET) */
typedef struct {
  __REG32 AT_MASK_SET   : 1;
  __REG32               : 1;
  __REG32 WR_MASK_SET   : 1;
  __REG32               :29;
} __intmskset_bits;

/* EMIF Interrupt Mask Clear Register (INTMSKCLR) */
typedef struct {
  __REG32 AT_MASK_CLR   : 1;
  __REG32               : 1;
  __REG32 WR_MASK_CLR   : 1;
  __REG32               :29;
} __intmskclr_bits;

/* EMIF NAND Flash Control Register (NANDFCR) */
typedef struct {
  __REG32 CS2NAND                 : 1;
  __REG32 CS3NAND                 : 1;
  __REG32 CS4NAND                 : 1;
  __REG32 CS5NAND                 : 1;
  __REG32 _4BITECCSEL             : 2;
  __REG32                         : 2;
  __REG32 CS2ECC                  : 1;
  __REG32 CS3ECC                  : 1;
  __REG32 CS4ECC                  : 1;
  __REG32 CS5ECC                  : 1;
  __REG32 _4BITECC_START          : 1;
  __REG32 _4BITECC_ADD_CALC_START : 1;
  __REG32                         :18;
} __nandfcr_bits;

/* EMIF NAND Flash Status Register (NANDFSR) */
typedef struct {
  __REG32 WAITST0                 : 1;
  __REG32 WAITST1                 : 1;
  __REG32                         : 6;
  __REG32 ECC_STATE               : 4;
  __REG32                         : 4;
  __REG32 ECC_ERRNUM              : 2;
  __REG32                         :14;
} __nandfsr_bits;

/* EMIF Page Mode Control Register (PMCR) */
typedef struct {
  __REG32 CS2_PG_MD_EN            : 1;
  __REG32 CS2_PG_SIZE             : 1;
  __REG32 CS2_PG_DEL              : 6;
  __REG32 CS3_PG_MD_EN            : 1;
  __REG32 CS3_PG_SIZE             : 1;
  __REG32 CS3_PG_DEL              : 6;
  __REG32 CS4_PG_MD_EN            : 1;
  __REG32 CS4_PG_SIZE             : 1;
  __REG32 CS4_PG_DEL              : 6;
  __REG32 CS5_PG_MD_EN            : 1;
  __REG32 CS5_PG_SIZE             : 1;
  __REG32 CS5_PG_DEL              : 6;
} __pmcr_bits;

/* EMIF NAND Flash n ECC Registers (NANDF1ECC-NANDF4ECC) */
typedef struct {
  __REG32 P1E                     : 1;
  __REG32 P2E                     : 1;
  __REG32 P4E                     : 1;
  __REG32 P8E                     : 1;
  __REG32 P16E                    : 1;
  __REG32 P32E                    : 1;
  __REG32 P64E                    : 1;
  __REG32 P128E                   : 1;
  __REG32 P256E                   : 1;
  __REG32 P512E                   : 1;
  __REG32 P1024E                  : 1;
  __REG32 P2048E                  : 1;
  __REG32                         : 4;
  __REG32 P1O                     : 1;
  __REG32 P2O                     : 1;
  __REG32 P4O                     : 1;
  __REG32 P8O                     : 1;
  __REG32 P16O                    : 1;
  __REG32 P32O                    : 1;
  __REG32 P64O                    : 1;
  __REG32 P128O                   : 1;
  __REG32 P256O                   : 1;
  __REG32 P512O                   : 1;
  __REG32 P1024O                  : 1;
  __REG32 P2048O                  : 1;
  __REG32                         : 4;
} __nandfecc_bits;

/* EMIF NAND Flash 4-Bit ECC LOAD Register (NAND4BITECCLOAD) */
typedef struct {
  __REG32 _4BITECCLOAD            :10;
  __REG32                         :22;
} __nand4biteccload_bits;

/* EMIF NAND Flash 4-Bit ECC Register 1 (NAND4BITECC1) */
typedef struct {
  __REG32 _4BITECCVAL1            :10;
  __REG32                         : 6;
  __REG32 _4BITECCVAL2            :10;
  __REG32                         : 6;
} __nand4bitecc1_bits;

/* EMIF NAND Flash 4-Bit ECC Register 2 (NAND4BITECC2) */
typedef struct {
  __REG32 _4BITECCVAL3            :10;
  __REG32                         : 6;
  __REG32 _4BITECCVAL4            :10;
  __REG32                         : 6;
} __nand4bitecc2_bits;

/* EMIF NAND Flash 4-Bit ECC Register 3 (NAND4BITECC3) */
typedef struct {
  __REG32 _4BITECCVAL5            :10;
  __REG32                         : 6;
  __REG32 _4BITECCVAL6            :10;
  __REG32                         : 6;
} __nand4bitecc3_bits;

/* EMIF NAND Flash 4-Bit ECC Register 4 (NAND4BITECC4) */
typedef struct {
  __REG32 _4BITECCVAL7            :10;
  __REG32                         : 6;
  __REG32 _4BITECCVAL8            :10;
  __REG32                         : 6;
} __nand4bitecc4_bits;

/* EMIF NAND Flash 4-Bit ECC Error Address Register 1 (NANDERRADD1) */
typedef struct {
  __REG32 _4BITECCERRADD1         :10;
  __REG32                         : 6;
  __REG32 _4BITECCERRADD2         :10;
  __REG32                         : 6;
} __nanderradd1_bits;

/* EMIF NAND Flash 4-Bit ECC Error Address Register 2 (NANDERRADD2) */
typedef struct {
  __REG32 _4BITECCERRADD3         :10;
  __REG32                         : 6;
  __REG32 _4BITECCERRADD4         :10;
  __REG32                         : 6;
} __nanderradd2_bits;

/* EMIF NAND Flash 4-Bit ECC Error Value Register 1 (NANDERRVAL1) */
typedef struct {
  __REG32 _4BITECCERRVAL1         :10;
  __REG32                         : 6;
  __REG32 _4BITECCERRVAL2         :10;
  __REG32                         : 6;
} __nanderrval1_bits;

/* EMIF NAND Flash 4-Bit ECC Error Value Register 2 (NANDERRVAL2) */
typedef struct {
  __REG32 _4BITECCERRVAL3         :10;
  __REG32                         : 6;
  __REG32 _4BITECCERRVAL4         :10;
  __REG32                         : 6;
} __nanderrval2_bits;

/* SDRAM Configuration Register (SDCFG) */
typedef struct {
  __REG32 PAGESIZE                : 3;
  __REG32 EBANK                   : 1;
  __REG32 IBANK                   : 3;
  __REG32                         : 2;
  __REG32 CL                      : 3;
  __REG32                         : 2;
  __REG32 NM                      : 1;
  __REG32 TIMUNLOCK               : 1;
  __REG32 SDREN                   : 1;
  __REG32                         : 6;
  __REG32 BOOT_UNLOCK             : 1;
  __REG32                         : 1;
  __REG32 MSDRAM_ENABLE           : 1;
  __REG32 IBANK_POS               : 1;
  __REG32                         : 5;
} __emifb_sdcfg_bits;

/* SDRAM Refresh Control Register (SDRFC) */
typedef struct {
  __REG32 REFRESH_RATE            :16;
  __REG32                         : 7;
  __REG32 SR_PD                   : 1;
  __REG32                         : 6;
  __REG32 MCLKSTOP_EN             : 1;
  __REG32 LP_MODE                 : 1;
} __emifb_sdrfc_bits;

/* SDRAM Timing 1 Register (SDTIM1) */
typedef struct {
  __REG32                         : 3;
  __REG32 T_RRD                   : 3;
  __REG32 T_RC                    : 5;
  __REG32 T_RAS                   : 5;
  __REG32 T_WR                    : 3;
  __REG32 T_RCD                   : 3;
  __REG32 T_RP                    : 3;
  __REG32 T_RFC                   : 7;
} __emifb_sdtim1_bits;

/* SDRAM Timing 2 Register (SDTIM2) */
typedef struct {
  __REG32 T_CKE                   : 5;
  __REG32                         :11;
  __REG32 T_XSR                   : 7;
  __REG32                         : 4;
  __REG32 T_RAS_MAX               : 4;
  __REG32                         : 1;
} __emifb_sdtim2_bits;

/* SDRAM Configuration 2 Register (SDCFG2) */
typedef struct {
  __REG32 ROWSIZE                 : 3;
  __REG32                         :13;
  __REG32 PASR                    : 3;
  __REG32                         :13;
} __emifb_sdcfg2_bits;

/* Peripheral Bus Burst Priority Register (BPRIO) */
typedef struct {
  __REG32 PRIO_RAISE              : 8;
  __REG32                         :24;
} __emifb_bprio_bits;

/* Performance Counter Configuration Register (PCC) */
typedef struct {
  __REG32 CNTR1_CFG               : 4;
  __REG32                         :10;
  __REG32 CNTR1_REGION_EN         : 1;
  __REG32 CNTR1_MSTID_EN          : 1;
  __REG32 CNTR2_CFG               : 4;
  __REG32                         :10;
  __REG32 CNTR2_REGION_EN         : 1;
  __REG32 CNTR2_MSTID_EN          : 1;
} __emifb_pcc_bits;

/* Performance Counter Master Region Select Register (PCMRS) */
typedef struct {
  __REG32 REGION_SEL1             : 4;
  __REG32                         : 4;
  __REG32 MST_ID1                 : 8;
  __REG32 REGION_SEL2             : 4;
  __REG32                         : 4;
  __REG32 MST_ID2                 : 8;
} __emifb_pcmrs_bits;

/* Interrupt Raw Register (IRR) */
/* Interrupt Mask Register (IMR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LT                      : 1;
  __REG32                         :29;
} __emifb_irr_bits;

/* Interrupt Mask Set Register (IMSR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LTMSET                  : 1;
  __REG32                         :29;
} __emifb_imsr_bits;

/* Interrupt Mask Clear Register (IMCR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LTMCLR                  : 1;
  __REG32                         :29;
} __emifb_imcr_bits;

/* MMC/SD Control Register (MMCCTL) */
typedef struct {
  __REG32 DATRST                  : 1;
  __REG32 CMDRST                  : 1;
  __REG32 WIDTH0                  : 1;
  __REG32                         : 3;
  __REG32 DATEG                   : 2;
  __REG32 WIDTH1                  : 1;
  __REG32 PERMDR                  : 1;
  __REG32 PERMDX                  : 1;
  __REG32                         :21;
} __mmcctl_bits;

/* MMC/SD Control Register (MMCCTL) */
typedef struct {
  __REG32 CLKRT                   : 8;
  __REG32 CLKEN                   : 1;
  __REG32 DIV4                    : 1;
  __REG32                         :22;
} __mmcclk_bits;

/* MMC/SD Status Register 0 (MMCST0) */
typedef struct {
  __REG32 DATDNE                  : 1;
  __REG32 BSYDNE                  : 1;
  __REG32 RSPDNE                  : 1;
  __REG32 TOUTRD                  : 1;
  __REG32 TOUTRS                  : 1;
  __REG32 CRCWR                   : 1;
  __REG32 CRCRD                   : 1;
  __REG32 CRCRS                   : 1;
  __REG32                         : 1;
  __REG32 DXRDY                   : 1;
  __REG32 DRRDY                   : 1;
  __REG32 DATED                   : 1;
  __REG32 TRNDNE                  : 1;
  __REG32 CCS                     : 1;
  __REG32                         :18;
} __mmcst0_bits;

/* MMC/SD Status Register 1 (MMCST1) */
typedef struct {
  __REG32 BUSY                    : 1;
  __REG32 CLKSTP                  : 1;
  __REG32 DXEMP                   : 1;
  __REG32 DRFUL                   : 1;
  __REG32 DAT3ST                  : 1;
  __REG32 FIFOEMP                 : 1;
  __REG32 FIFOFUL                 : 1;
  __REG32                         :25;
} __mmcst1_bits;

/* MMC/SD Interrupt Mask Register (MMCIM) */
typedef struct {
  __REG32 EDATDNE                 : 1;
  __REG32 EBSYDNE                 : 1;
  __REG32 ERSPDNE                 : 1;
  __REG32 ETOUTRD                 : 1;
  __REG32 ETOUTRS                 : 1;
  __REG32 ECRCWR                  : 1;
  __REG32 ECRCRD                  : 1;
  __REG32 ECRCRS                  : 1;
  __REG32                         : 1;
  __REG32 EDXRDY                  : 1;
  __REG32 EDRRDY                  : 1;
  __REG32 EDATED                  : 1;
  __REG32 ETRNDNE                 : 1;
  __REG32 ECCS                    : 1;
  __REG32                         :18;
} __mmcim_bits;

/* MMC/SD Response Time-Out Register (MMCTOR) */
typedef struct {
  __REG32 TOR                     : 8;
  __REG32 TOD_25_16               :10;
  __REG32                         :14;
} __mmctor_bits;

/* MMC/SD Data Read Time-Out Register (MMCTOD) */
typedef struct {
  __REG32 TOD_15_0                :16;
  __REG32                         :16;
} __mmctod_bits;

/* MMC/SD Block Length Register (MMCBLEN) */
typedef struct {
  __REG32 BLEN                    :12;
  __REG32                         :20;
} __mmcblen_bits;

/* MMC/SD Number of Blocks Register (MMCNBLK) */
typedef struct {
  __REG32 NBLK                    :16;
  __REG32                         :16;
} __mmcnblk_bits;

/* MMC/SD Number of Blocks Counter Register (MMCNBLC) */
typedef struct {
  __REG32 NBLC                    :16;
  __REG32                         :16;
} __mmcnblc_bits;

/* MMC/SD Command Register (MMCCMD) */
typedef struct {
  __REG32 CMD                     : 6;
  __REG32                         : 1;
  __REG32 PPLEN                   : 1;
  __REG32 BSYEXP                  : 1;
  __REG32 RSPFMT                  : 2;
  __REG32 DTRW                    : 1;
  __REG32 STRMTP                  : 1;
  __REG32 WDATX                   : 1;
  __REG32 INITCK                  : 1;
  __REG32 DCLR                    : 1;
  __REG32 DMATRIG                 : 1;
  __REG32                         :15;
} __mmccmd_bits;

/* MMC/SD Argument Register (MMCARGHL) */
typedef struct {
  __REG32 ARGL                    :16;
  __REG32 ARGH                    :16;
} __mmcarghl_bits;

/* MMC/SD Response Register 0 and 1 (MMCRSP01) */
typedef struct {
  __REG32 MMCRSP0                 :16;
  __REG32 MMCRSP1                 :16;
} __mmcrsp01_bits;

/* MMC/SD Response Register 2 and 3 (MMCRSP23) */
typedef struct {
  __REG32 MMCRSP2                 :16;
  __REG32 MMCRSP3                 :16;
} __mmcrsp23_bits;

/* MMC/SD Response Register 4 and 5 (MMCRSP45) */
typedef struct {
  __REG32 MMCRSP4                 :16;
  __REG32 MMCRSP5                 :16;
} __mmcrsp45_bits;

/* MMC/SD Response Register 6 and 7 (MMCRSP67) */
typedef struct {
  __REG32 MMCRSP6                 :16;
  __REG32 MMCRSP7                 :16;
} __mmcrsp67_bits;

/* MMC/SD Data Response Register (MMCDRSP) */
typedef struct {
  __REG32 DRSP                    : 8;
  __REG32                         :24;
} __mmcdrsp_bits;

/* MMC/SD Command Index Register (MMCCIDX) */
typedef struct {
  __REG32 CIDX                    : 6;
  __REG32 XMIT                    : 1;
  __REG32 STRT                    : 1;
  __REG32                         :24;
} __mmccidx_bits;

/* MMC/SD SDIO Control Register (SDIOCTL) */
typedef struct {
  __REG32 RDWTRQ                  : 1;
  __REG32 RDWTCR                  : 1;
  __REG32                         :30;
} __sdioctl_bits;

/* MMC/SD SDIO Status Register 0 (SDIOST0) */
typedef struct {
  __REG32 DAT1                    : 1;
  __REG32 INTPRD                  : 1;
  __REG32 RDWTST                  : 1;
  __REG32                         :29;
} __sdiost0_bits;

/* MMC/SD SDIO Interrupt Enable Register (SDIOIEN) */
typedef struct {
  __REG32 IOINTEN                 : 1;
  __REG32 RWSEN                   : 1;
  __REG32                         :30;
} __sdioien_bits;

/* MMC/SD SDIO Interrupt Status Register (SDIOIST) */
typedef struct {
  __REG32 IOINT                   : 1;
  __REG32 RWS                     : 1;
  __REG32                         :30;
} __sdioist_bits;

/* MMC/SD FIFO Control Register (MMCFIFOCTL) */
typedef struct {
  __REG32 FIFORST                 : 1;
  __REG32 FIFODIR                 : 1;
  __REG32 FIFOLEV                 : 1;
  __REG32 ACCWD                   : 2;
  __REG32                         :27;
} __mmcfifoctl_bits;

/* McASP Pin Function Register (PFUNC) */
typedef struct {
  __REG32 AXR0                    : 1;
  __REG32 AXR1                    : 1;
  __REG32 AXR2                    : 1;
  __REG32 AXR3                    : 1;
  __REG32 AXR4                    : 1;
  __REG32 AXR5                    : 1;
  __REG32 AXR6                    : 1;
  __REG32 AXR7                    : 1;
  __REG32 AXR8                    : 1;
  __REG32 AXR9                    : 1;
  __REG32 AXR10                   : 1;
  __REG32 AXR11                   : 1;
  __REG32 AXR12                   : 1;
  __REG32 AXR13                   : 1;
  __REG32 AXR14                   : 1;
  __REG32 AXR15                   : 1;
  __REG32                         : 9;
  __REG32 AMUTE                   : 1;
  __REG32 ACLKX                   : 1;
  __REG32 AHCLKX                  : 1;
  __REG32 AFSX                    : 1;
  __REG32 ACLKR                   : 1;
  __REG32 AHCLKR                  : 1;
  __REG32 AFSR                    : 1;
} __mcasp_pfunc_bits;

/* McASP Global Control Register (GBLCTL) */
typedef struct {
  __REG32 RCLKRST                 : 1;
  __REG32 RHCLKRST                : 1;
  __REG32 RSRCLR                  : 1;
  __REG32 RSMRST                  : 1;
  __REG32 RFRST                   : 1;
  __REG32                         : 3;
  __REG32 XCLKRST                 : 1;
  __REG32 XHCLKRST                : 1;
  __REG32 XSRCLR                  : 1;
  __REG32 XSMRST                  : 1;
  __REG32 XFRST                   : 1;
  __REG32                         :19;
} __mcasp_gblctl_bits;

/* McASP Audio Mute Control Register (AMUTE) */
typedef struct {
  __REG32 MUTEN                   : 2;
  __REG32 INPOL                   : 1;
  __REG32 INEN                    : 1;
  __REG32 INSTAT                  : 1;
  __REG32 ROVRN                   : 1;
  __REG32 XUNDRN                  : 1;
  __REG32 RSYNCERR                : 1;
  __REG32 XSYNCERR                : 1;
  __REG32 RCKFAIL                 : 1;
  __REG32 XCKFAIL                 : 1;
  __REG32 RDMAERR                 : 1;
  __REG32 XDMAERR                 : 1;
  __REG32                         :19;
} __mcasp_amute_bits;

/* McASP Digital Loopback Control Register (DLBCTL) */
typedef struct {
  __REG32 DLBEN                   : 1;
  __REG32 ORD                     : 1;
  __REG32 MODE                    : 2;
  __REG32                         :28;
} __mcasp_dlbctl_bits;

/* McASP Digital Mode Control Register (DITCTL) */
typedef struct {
  __REG32 DITEN                   : 1;
  __REG32                         : 1;
  __REG32 VA                      : 1;
  __REG32 VB                      : 1;
  __REG32                         :28;
} __mcasp_ditctl_bits;

/* McASP Receiver Global Control Register (RGBLCTL) */
typedef struct {
  __REG32 RCLKRST                 : 1;
  __REG32 RHCLKRST                : 1;
  __REG32 RSRCLR                  : 1;
  __REG32 RSMRST                  : 1;
  __REG32 RFRST                   : 1;
  __REG32                         : 3;
  __REG32 XCLKRST                 : 1;
  __REG32 XHCLKRST                : 1;
  __REG32 XSRCLR                  : 1;
  __REG32 XSMRST                  : 1;
  __REG32 XFRST                   : 1;
  __REG32                         :19;
} __mcasp_rgblctl_bits;

/* McASP Receive Format Unit Bit Mask Register (RMASK) */
typedef struct {
  __REG32 RMASK0                  : 1;
  __REG32 RMASK1                  : 1;
  __REG32 RMASK2                  : 1;
  __REG32 RMASK3                  : 1;
  __REG32 RMASK4                  : 1;
  __REG32 RMASK5                  : 1;
  __REG32 RMASK6                  : 1;
  __REG32 RMASK7                  : 1;
  __REG32 RMASK8                  : 1;
  __REG32 RMASK9                  : 1;
  __REG32 RMASK10                 : 1;
  __REG32 RMASK11                 : 1;
  __REG32 RMASK12                 : 1;
  __REG32 RMASK13                 : 1;
  __REG32 RMASK14                 : 1;
  __REG32 RMASK15                 : 1;
  __REG32 RMASK16                 : 1;
  __REG32 RMASK17                 : 1;
  __REG32 RMASK18                 : 1;
  __REG32 RMASK19                 : 1;
  __REG32 RMASK20                 : 1;
  __REG32 RMASK21                 : 1;
  __REG32 RMASK22                 : 1;
  __REG32 RMASK23                 : 1;
  __REG32 RMASK24                 : 1;
  __REG32 RMASK25                 : 1;
  __REG32 RMASK26                 : 1;
  __REG32 RMASK27                 : 1;
  __REG32 RMASK28                 : 1;
  __REG32 RMASK29                 : 1;
  __REG32 RMASK30                 : 1;
  __REG32 RMASK31                 : 1;
} __mcasp_rmask_bits;

/* McASP Receive Bit Stream Format Register (RFMT) */
typedef struct {
  __REG32 RROT                    : 3;
  __REG32 RBUSEL                  : 1;
  __REG32 RSSZ                    : 4;
  __REG32 RPBIT                   : 5;
  __REG32 RPAD                    : 2;
  __REG32 RRVRS                   : 1;
  __REG32 RDATDLY                 : 2;
  __REG32                         :14;
} __mcasp_rfmt_bits;

/* Receive Frame Sync Control Register (AFSRCTL) */
typedef struct {
  __REG32 FSRP                    : 1;
  __REG32 FSRM                    : 1;
  __REG32                         : 2;
  __REG32 FRWID                   : 1;
  __REG32                         : 2;
  __REG32 RMOD                    : 9;
  __REG32                         :16;
} __mcasp_afsrctl_bits;

/* McASP Receive Clock Control Register (ACLKRCTL) */
typedef struct {
  __REG32 CLKRDIV                 : 5;
  __REG32 CLKRM                   : 1;
  __REG32                         : 1;
  __REG32 CLKRP                   : 1;
  __REG32                         :24;
} __mcasp_aclkrctl_bits;

/* McASP Receive Clock Control Register (ACLKRCTL) */
typedef struct {
  __REG32 HCLKRDIV                :12;
  __REG32                         : 2;
  __REG32 HCLKRP                  : 1;
  __REG32 HCLKRM                  : 1;
  __REG32                         :16;
} __mcasp_ahclkrctl_bits;

/* McASP Receive TDM Time Slot Register (RTDM) */
typedef struct {
  __REG32 RTDMS0                  : 1;
  __REG32 RTDMS1                  : 1;
  __REG32 RTDMS2                  : 1;
  __REG32 RTDMS3                  : 1;
  __REG32 RTDMS4                  : 1;
  __REG32 RTDMS5                  : 1;
  __REG32 RTDMS6                  : 1;
  __REG32 RTDMS7                  : 1;
  __REG32 RTDMS8                  : 1;
  __REG32 RTDMS9                  : 1;
  __REG32 RTDMS10                 : 1;
  __REG32 RTDMS11                 : 1;
  __REG32 RTDMS12                 : 1;
  __REG32 RTDMS13                 : 1;
  __REG32 RTDMS14                 : 1;
  __REG32 RTDMS15                 : 1;
  __REG32 RTDMS16                 : 1;
  __REG32 RTDMS17                 : 1;
  __REG32 RTDMS18                 : 1;
  __REG32 RTDMS19                 : 1;
  __REG32 RTDMS20                 : 1;
  __REG32 RTDMS21                 : 1;
  __REG32 RTDMS22                 : 1;
  __REG32 RTDMS23                 : 1;
  __REG32 RTDMS24                 : 1;
  __REG32 RTDMS25                 : 1;
  __REG32 RTDMS26                 : 1;
  __REG32 RTDMS27                 : 1;
  __REG32 RTDMS28                 : 1;
  __REG32 RTDMS29                 : 1;
  __REG32 RTDMS30                 : 1;
  __REG32 RTDMS31                 : 1;
} __mcasp_rtdm_bits;

/* McASP Receiver Interrupt Control Register (RINTCTL) */
typedef struct {
  __REG32 ROVRN                   : 1;
  __REG32 RSYNCERR                : 1;
  __REG32 RCKFAIL                 : 1;
  __REG32 RDMAERR                 : 1;
  __REG32 RLAST                   : 1;
  __REG32 RDATA                   : 1;
  __REG32                         : 1;
  __REG32 RSTAFRM                 : 1;
  __REG32                         :24;
} __mcasp_rintctl_bits;

/* McASP Receiver Status Register (RSTAT) */
typedef struct {
  __REG32 ROVRN                   : 1;
  __REG32 RSYNCERR                : 1;
  __REG32 RCKFAIL                 : 1;
  __REG32 RTDMSLOT                : 1;
  __REG32 RLAST                   : 1;
  __REG32 RDATA                   : 1;
  __REG32 RSTAFRM                 : 1;
  __REG32 RDMAERR                 : 1;
  __REG32 RERR                    : 1;
  __REG32                         :23;
} __mcasp_rstat_bits;

/* McASP Current Receive TDM Time Slot Registers (RSLOT) */
typedef struct {
  __REG32 RSLOTCNT                : 9;
  __REG32                         :23;
} __mcasp_rslot_bits;

/* McASP Receive Clock Check Control Register (RCLKCHK) */
typedef struct {
  __REG32 RPS                     : 4;
  __REG32                         : 4;
  __REG32 RMIN                    : 8;
  __REG32 RMAX                    : 8;
  __REG32 RCNT                    : 8;
} __mcasp_rclkchk_bits;

/* McASP Receiver DMA Event Control Register (REVTCTL) */
typedef struct {
  __REG32 RDATDMA                 : 1;
  __REG32                         :31;
} __mcasp_revtctl_bits;

/* McASP Transmitter Global Control Register (XGBLCTL) */
typedef struct {
  __REG32 RCLKRST                 : 1;
  __REG32 RHCLKRST                : 1;
  __REG32 RSRCLR                  : 1;
  __REG32 RSMRST                  : 1;
  __REG32 RFRST                   : 1;
  __REG32                         : 3;
  __REG32 XCLKRST                 : 1;
  __REG32 XHCLKRST                : 1;
  __REG32 XSRCLR                  : 1;
  __REG32 XSMRST                  : 1;
  __REG32 XFRST                   : 1;
  __REG32                         :19;
} __mcasp_xgblctl_bits;

/* McASP Transmit Format Unit Bit Mask Register (XMASK) */
typedef struct {
  __REG32 XMASK0                  : 1;
  __REG32 XMASK1                  : 1;
  __REG32 XMASK2                  : 1;
  __REG32 XMASK3                  : 1;
  __REG32 XMASK4                  : 1;
  __REG32 XMASK5                  : 1;
  __REG32 XMASK6                  : 1;
  __REG32 XMASK7                  : 1;
  __REG32 XMASK8                  : 1;
  __REG32 XMASK9                  : 1;
  __REG32 XMASK10                 : 1;
  __REG32 XMASK11                 : 1;
  __REG32 XMASK12                 : 1;
  __REG32 XMASK13                 : 1;
  __REG32 XMASK14                 : 1;
  __REG32 XMASK15                 : 1;
  __REG32 XMASK16                 : 1;
  __REG32 XMASK17                 : 1;
  __REG32 XMASK18                 : 1;
  __REG32 XMASK19                 : 1;
  __REG32 XMASK20                 : 1;
  __REG32 XMASK21                 : 1;
  __REG32 XMASK22                 : 1;
  __REG32 XMASK23                 : 1;
  __REG32 XMASK24                 : 1;
  __REG32 XMASK25                 : 1;
  __REG32 XMASK26                 : 1;
  __REG32 XMASK27                 : 1;
  __REG32 XMASK28                 : 1;
  __REG32 XMASK29                 : 1;
  __REG32 XMASK30                 : 1;
  __REG32 XMASK31                 : 1;
} __mcasp_xmask_bits;

/* McASP Transmit Bit Stream Format Register (XFMT) */
typedef struct {
  __REG32 XROT                    : 3;
  __REG32 XBUSEL                  : 1;
  __REG32 XSSZ                    : 4;
  __REG32 XPBIT                   : 5;
  __REG32 XPAD                    : 2;
  __REG32 XRVRS                   : 1;
  __REG32 XDATDLY                 : 2;
  __REG32                         :14;
} __mcasp_xfmt_bits;

/* McASP Transmit Frame Sync Control Register (AFSXCTL) */
typedef struct {
  __REG32 FSXP                    : 1;
  __REG32 FSXM                    : 1;
  __REG32                         : 2;
  __REG32 FXWID                   : 1;
  __REG32                         : 2;
  __REG32 XMOD                    : 9;
  __REG32                         :16;
} __mcasp_afsxctl_bits;

/* McASP Transmit Clock Control Register (ACLKXCTL) */
typedef struct {
  __REG32 CLKXDIV                 : 5;
  __REG32 CLKXM                   : 1;
  __REG32 ASYNC                   : 1;
  __REG32 CLKXP                   : 1;
  __REG32                         :24;
} __mcasp_aclkxctl_bits;

/* McASP Transmit High-Frequency Clock Control Register (AHCLKXCTL) */
typedef struct {
  __REG32 HCLKXDIV                :12;
  __REG32                         : 2;
  __REG32 HCLKXP                  : 1;
  __REG32 HCLKXM                  : 1;
  __REG32                         :16;
} __mcasp_ahclkxctl_bits;

/* McASP Transmit TDM Time Slot Register (XTDM) */
typedef struct {
  __REG32 XTDMS0                  : 1;
  __REG32 XTDMS1                  : 1;
  __REG32 XTDMS2                  : 1;
  __REG32 XTDMS3                  : 1;
  __REG32 XTDMS4                  : 1;
  __REG32 XTDMS5                  : 1;
  __REG32 XTDMS6                  : 1;
  __REG32 XTDMS7                  : 1;
  __REG32 XTDMS8                  : 1;
  __REG32 XTDMS9                  : 1;
  __REG32 XTDMS10                 : 1;
  __REG32 XTDMS11                 : 1;
  __REG32 XTDMS12                 : 1;
  __REG32 XTDMS13                 : 1;
  __REG32 XTDMS14                 : 1;
  __REG32 XTDMS15                 : 1;
  __REG32 XTDMS16                 : 1;
  __REG32 XTDMS17                 : 1;
  __REG32 XTDMS18                 : 1;
  __REG32 XTDMS19                 : 1;
  __REG32 XTDMS20                 : 1;
  __REG32 XTDMS21                 : 1;
  __REG32 XTDMS22                 : 1;
  __REG32 XTDMS23                 : 1;
  __REG32 XTDMS24                 : 1;
  __REG32 XTDMS25                 : 1;
  __REG32 XTDMS26                 : 1;
  __REG32 XTDMS27                 : 1;
  __REG32 XTDMS28                 : 1;
  __REG32 XTDMS29                 : 1;
  __REG32 XTDMS30                 : 1;
  __REG32 XTDMS31                 : 1;
} __mcasp_xtdm_bits;

/* McASP Transmitter Interrupt Control Register (XINTCTL) */
typedef struct {
  __REG32 XUNDRN                  : 1;
  __REG32 XSYNCERR                : 1;
  __REG32 XCKFAIL                 : 1;
  __REG32 XDMAERR                 : 1;
  __REG32 XLAST                   : 1;
  __REG32 XDATA                   : 1;
  __REG32                         : 1;
  __REG32 XSTAFRM                 : 1;
  __REG32                         :24;
} __mcasp_xintctl_bits;

/* McASP Transmitter Status Register (XSTAT) */
typedef struct {
  __REG32 XUNDRN                  : 1;
  __REG32 XSYNCERR                : 1;
  __REG32 XCKFAIL                 : 1;
  __REG32 XTDMSLOT                : 1;
  __REG32 XLAST                   : 1;
  __REG32 XDATA                   : 1;
  __REG32 XSTAFRM                 : 1;
  __REG32 XDMAERR                 : 1;
  __REG32 XERR                    : 1;
  __REG32                         :23;
} __mcasp_xstat_bits;

/* McASP Current Transmit TDM Time Slot Register (XSLOT) */
typedef struct {
  __REG32 XSLOTCNT                : 9;
  __REG32                         :23;
} __mcasp_xslot_bits;

/* McASP Transmit Clock Check Control Register (XCLKCHK) */
typedef struct {
  __REG32 XPS                     : 4;
  __REG32                         : 4;
  __REG32 XMIN                    : 8;
  __REG32 XMAX                    : 8;
  __REG32 XCNT                    : 8;
} __mcasp_xclkchk_bits;

/* McASP Transmitter DMA Event Control Register (XEVTCTL) */
typedef struct {
  __REG32 XDATDMA                 : 1;
  __REG32                         :31;
} __mcasp_xevtctl_bits;

/* McASP Serializer Control Registers (SRCTLn) */
typedef struct {
  __REG32 SRMOD                   : 2;
  __REG32 DISMOD                  : 2;
  __REG32 XRDY                    : 1;
  __REG32 RRDY                    : 1;
  __REG32                         :26;
} __mcasp_srctl_bits;

/* McASP Write FIFO Control Register (WFIFOCTL) */
typedef struct {
  __REG32 WNUMDMA                 : 8;
  __REG32 WNUMEVT                 : 8;
  __REG32 WENA                    : 1;
  __REG32                         :15;
} __mcasp_wfifoctl_bits;

/* McASP Write FIFO Status Register (WFIFOSTS) */
typedef struct {
  __REG32 WLVL                    : 8;
  __REG32                         :24;
} __mcasp_wfifosts_bits;

/* McASP Read FIFO Control Register (RFIFOCTL) */
typedef struct {
  __REG32 RNUMDMA                 : 8;
  __REG32 RNUMEVT                 : 8;
  __REG32 RENA                    : 1;
  __REG32                         :15;
} __mcasp_rfifoctl_bits;

/* McASP Read FIFO Status Register (RFIFOSTS) */
typedef struct {
  __REG32 RLVL                    : 8;
  __REG32                         :24;
} __mcasp_rfifosts_bits;

/* SPI Global Control Register 0 (SPIGCR0) */
typedef struct {
  __REG32 RESET                   : 1;
  __REG32                         :31;
} __spigcr0_bits;

/* SPI Global Control Register 1 (SPIGCR1) */
typedef struct {
  __REG32 MASTER                  : 1;
  __REG32 CLKMOD                  : 1;
  __REG32                         : 6;
  __REG32 POWERDOWN               : 1;
  __REG32                         : 7;
  __REG32 LOOPBACK                : 1;
  __REG32                         : 7;
  __REG32 ENABLE                  : 1;
  __REG32                         : 7;
} __spigcr1_bits;

/* SPI Interrupt Register (SPIINT0) */
typedef struct {
  __REG32 DLENERRENA              : 1;
  __REG32 TIMEOUTENA              : 1;
  __REG32 PARERRENA               : 1;
  __REG32 DESYNCENA               : 1;
  __REG32 BITERRENA               : 1;
  __REG32                         : 1;
  __REG32 OVRNINTENA              : 1;
  __REG32                         : 1;
  __REG32 RXINTENA                : 1;
  __REG32 TXINTENA                : 1;
  __REG32                         : 6;
  __REG32 DMAREQEN                : 1;
  __REG32                         : 7;
  __REG32 ENABLEHIGHZ             : 1;
  __REG32                         : 7;
} __spiint0_bits;

/* SPI Interrupt Level Register (SPILVL) */
typedef struct {
  __REG32 DLENERRLVL              : 1;
  __REG32 TIMEOUTLVL              : 1;
  __REG32 PARERRLVL               : 1;
  __REG32 DESYNCLVL               : 1;
  __REG32 BITERRLVL               : 1;
  __REG32                         : 1;
  __REG32 OVRNINTLVL              : 1;
  __REG32                         : 1;
  __REG32 RXINTLVL                : 1;
  __REG32 TXINTLVL                : 1;
  __REG32                         :22;
} __spilvl_bits;

/* SPI Flag Register (SPIFLG) */
typedef struct {
  __REG32 DLENERRFLG              : 1;
  __REG32 TIMEOUTFLG              : 1;
  __REG32 PARERRFLG               : 1;
  __REG32 DESYNCFLG               : 1;
  __REG32 BITERRFLG               : 1;
  __REG32                         : 1;
  __REG32 OVRNINTFLG              : 1;
  __REG32                         : 1;
  __REG32 RXINTFLG                : 1;
  __REG32 TXINTFLG                : 1;
  __REG32                         :22;
} __spiflg_bits;

/* SPI Pin Control Register 0 (SPIPC0) */
typedef struct {
  __REG32 SCS0FU0                 : 1;
  __REG32 SCS0FU1                 : 1;
  __REG32 SCS0FU2                 : 1;
  __REG32 SCS0FU3                 : 1;
  __REG32 SCS0FU4                 : 1;
  __REG32 SCS0FU5                 : 1;
  __REG32 SCS0FU6                 : 1;
  __REG32 SCS0FU7                 : 1;
  __REG32 ENAFUN                  : 1;
  __REG32 CLKFUN                  : 1;
  __REG32 SIMOFUN                 : 1;
  __REG32 SOMIFUN                 : 1;
  __REG32                         :20;
} __spipc0_bits;

/* SPI Pin Control Register 1 (SPIPC1) */
typedef struct {
  __REG32 SCS0DIR0                : 1;
  __REG32 SCS0DIR1                : 1;
  __REG32 SCS0DIR2                : 1;
  __REG32 SCS0DIR3                : 1;
  __REG32 SCS0DIR4                : 1;
  __REG32 SCS0DIR5                : 1;
  __REG32 SCS0DIR6                : 1;
  __REG32 SCS0DIR7                : 1;
  __REG32 ENADIR                  : 1;
  __REG32 CLKDIR                  : 1;
  __REG32 SIMODIR                 : 1;
  __REG32 SOMIDIR                 : 1;
  __REG32                         :20;
} __spipc1_bits;

/* SPI Pin Control Register 2 (SPIPC2) */
typedef struct {
  __REG32 SCS0DIN0                : 1;
  __REG32 SCS0DIN1                : 1;
  __REG32 SCS0DIN2                : 1;
  __REG32 SCS0DIN3                : 1;
  __REG32 SCS0DIN4                : 1;
  __REG32 SCS0DIN5                : 1;
  __REG32 SCS0DIN6                : 1;
  __REG32 SCS0DIN7                : 1;
  __REG32 ENADIN                  : 1;
  __REG32 CLKDIN                  : 1;
  __REG32 SIMODIN                 : 1;
  __REG32 SOMIDIN                 : 1;
  __REG32                         :20;
} __spipc2_bits;

/* SPI Pin Control Register 4 (SPIPC3) */
typedef struct {
  __REG32 SCS0DOUT0               : 1;
  __REG32 SCS0DOUT1               : 1;
  __REG32 SCS0DOUT2               : 1;
  __REG32 SCS0DOUT3               : 1;
  __REG32 SCS0DOUT4               : 1;
  __REG32 SCS0DOUT5               : 1;
  __REG32 SCS0DOUT6               : 1;
  __REG32 SCS0DOUT7               : 1;
  __REG32 ENADOUT                 : 1;
  __REG32 CLKDOUT                 : 1;
  __REG32 SIMODOUT                : 1;
  __REG32 SOMIDOUT                : 1;
  __REG32                         :20;
} __spipc3_bits;

/* SPI Pin Control Register 4 (SPIPC4) */
typedef struct {
  __REG32 SCS0SET0                : 1;
  __REG32 SCS0SET1                : 1;
  __REG32 SCS0SET2                : 1;
  __REG32 SCS0SET3                : 1;
  __REG32 SCS0SET4                : 1;
  __REG32 SCS0SET5                : 1;
  __REG32 SCS0SET6                : 1;
  __REG32 SCS0SET7                : 1;
  __REG32 ENASET                  : 1;
  __REG32 CLKSET                  : 1;
  __REG32 SIMOSET                 : 1;
  __REG32 SOMISET                 : 1;
  __REG32                         :20;
} __spipc4_bits;

/* SPI Pin Control Register 5 (SPIPC5) */
typedef struct {
  __REG32 SCS0CLR0                : 1;
  __REG32 SCS0CLR1                : 1;
  __REG32 SCS0CLR2                : 1;
  __REG32 SCS0CLR3                : 1;
  __REG32 SCS0CLR4                : 1;
  __REG32 SCS0CLR5                : 1;
  __REG32 SCS0CLR6                : 1;
  __REG32 SCS0CLR7                : 1;
  __REG32 ENACLR                  : 1;
  __REG32 CLKCLR                  : 1;
  __REG32 SIMOCLR                 : 1;
  __REG32 SOMICLR                 : 1;
  __REG32                         :20;
} __spipc5_bits;

/* SPI Transmit Data Register 0 (SPIDAT0) */
typedef struct {
  __REG32 TXDATA                  :16;
  __REG32                         :16;
} __spidat0_bits;

/* SPI Transmit Data Register 1 (SPIDAT1) */
typedef struct {
  __REG32 TXDATA                  :16;
  __REG32 CSNR0                   : 1;
  __REG32 CSNR1                   : 1;
  __REG32 CSNR2                   : 1;
  __REG32 CSNR3                   : 1;
  __REG32 CSNR4                   : 1;
  __REG32 CSNR5                   : 1;
  __REG32 CSNR6                   : 1;
  __REG32 CSNR7                   : 1;
  __REG32 DFSEL                   : 2;
  __REG32 WDEL                    : 1;
  __REG32                         : 1;
  __REG32 CSHOLD                  : 1;
  __REG32                         : 3;
} __spidat1_bits;

/* SPI Receive Buffer Register (SPIBUF) */
typedef struct {
  __REG32 RXDATA                  :16;
  __REG32                         : 8;
  __REG32 DLENERR                 : 1;
  __REG32 TIMEOUT                 : 1;
  __REG32 PARERR                  : 1;
  __REG32 DESYNC                  : 1;
  __REG32 BITERR                  : 1;
  __REG32 TXFULL                  : 1;
  __REG32 RXOVR                   : 1;
  __REG32 RXEMPTY                 : 1;
} __spibuf_bits;

/* SPI Emulation Register (SPIEMU) */
typedef struct {
  __REG32 RXDATA                  :16;
  __REG32                         :16;
} __spiemu_bits;

/* SPI Delay Register (SPIDELAY) */
typedef struct {
  __REG32 C2EDELAY                : 8;
  __REG32 T2EDELAY                : 8;
  __REG32 T2CDELAY                : 8;
  __REG32 C2TDELAY                : 8;
} __spidelay_bits;

/* SPI Default Chip Select Register (SPIDEF) */
typedef struct {
  __REG32 CSDEF0                  : 1;
  __REG32 CSDEF1                  : 1;
  __REG32 CSDEF2                  : 1;
  __REG32 CSDEF3                  : 1;
  __REG32 CSDEF4                  : 1;
  __REG32 CSDEF5                  : 1;
  __REG32 CSDEF6                  : 1;
  __REG32 CSDEF7                  : 1;
  __REG32                         :24;
} __spidef_bits;

/* SPI Data Format Registers (SPIFMTn) */
typedef struct {
  __REG32 CHARLEN                 : 5;
  __REG32                         : 3;
  __REG32 PRESCALE                : 8;
  __REG32 PHASE                   : 1;
  __REG32 POLARITY                : 1;
  __REG32 DISCSTIMERS             : 1;
  __REG32                         : 1;
  __REG32 SHIFTDIR                : 1;
  __REG32 WAITENA                 : 1;
  __REG32 PARENA                  : 1;
  __REG32 PARPOL                  : 1;
  __REG32 WDELAY                  : 6;
  __REG32                         : 2;
} __spifmt_bits;

/* SPI Interrupt Vector Register 1 (INTVEC1) */
typedef struct {
  __REG32                         : 1;
  __REG32 INTVECT1                : 5;
  __REG32                         :26;
} __spiintvec1_bits;

/* I2C Own Address Register (ICOAR) */
typedef struct {
  __REG32 OADDR                   :10;
  __REG32                         :22;
} __icoar_bits;

/* I2C Interrupt Mask Register (ICIMR) */
typedef struct {
  __REG32 AL                      : 1;
  __REG32 NACK                    : 1;
  __REG32 ARDY                    : 1;
  __REG32 ICRRDY                  : 1;
  __REG32 ICXRDY                  : 1;
  __REG32 SCD                     : 1;
  __REG32 AAS                     : 1;
  __REG32                         :25;
} __icimr_bits;

/* I2C Interrupt Status Register (ICSTR) */
typedef struct {
  __REG32 AL                      : 1;
  __REG32 NACK                    : 1;
  __REG32 ARDY                    : 1;
  __REG32 ICRRDY                  : 1;
  __REG32 ICXRDY                  : 1;
  __REG32 SCD                     : 1;
  __REG32                         : 2;
  __REG32 AD0                     : 1;
  __REG32 AAS                     : 1;
  __REG32 XSMT                    : 1;
  __REG32 RSFULL                  : 1;
  __REG32 BB                      : 1;
  __REG32 NACKSNT                 : 1;
  __REG32 SDIR                    : 1;
  __REG32                         :17;
} __icstr_bits;

/* I2C Clock Low-Time Divider Register (ICCLKL) */
typedef struct {
  __REG32 ICCL                    :16;
  __REG32                         :16;
} __icclkl_bits;

/* I2C Clock High-Time Divider Register (ICCLKH) */
typedef struct {
  __REG32 ICCL                    :16;
  __REG32                         :16;
} __icclkh_bits;

/* I2C Data Count Register (ICCNT) */
typedef struct {
  __REG32 ICDC                    :16;
  __REG32                         :16;
} __iccnt_bits;

/* I2C Data Receive Register (ICDRR) */
typedef struct {
  __REG32 D                       : 8;
  __REG32                         :24;
} __icdrr_bits;

/* I2C Slave Address Register (ICSAR) */
typedef struct {
  __REG32 SADDR                   :10;
  __REG32                         :22;
} __icsar_bits;

/* I2C Mode Register (ICMDR) */
typedef struct {
  __REG32 BC                      : 3;
  __REG32 FDF                     : 1;
  __REG32 STB                     : 1;
  __REG32 IRS                     : 1;
  __REG32 DLB                     : 1;
  __REG32 RM                      : 1;
  __REG32 XA                      : 1;
  __REG32 TRX                     : 1;
  __REG32 MST                     : 1;
  __REG32 STP                     : 1;
  __REG32                         : 1;
  __REG32 STT                     : 1;
  __REG32 FREE                    : 1;
  __REG32 NACKMOD                 : 1;
  __REG32                         :16;
} __icmdr_bits;

/* I2C Interrupt Vector Register (ICIVR) */
typedef struct {
  __REG32 INTCODE                 : 3;
  __REG32                         :29;
} __icivr_bits;

/* I2C Extended Mode Register (ICEMDR) */
typedef struct {
  __REG32 BCM                     : 1;
  __REG32 IGNACK                  : 1;
  __REG32                         :30;
} __icemdr_bits;

/* I2C Prescaler Register (ICPSC) */
typedef struct {
  __REG32 IPSC                    : 8;
  __REG32                         :24;
} __icpsc_bits;

/* I2C Pin Function Register (ICPFUNC) */
typedef struct {
  __REG32 PFUNC0                  : 1;
  __REG32                         :31;
} __icpfunc_bits;

/* I2C Pin Direction Register (ICPDIR) */
typedef struct {
  __REG32 PDIR0                   : 1;
  __REG32 PDIR1                   : 1;
  __REG32                         :30;
} __icpdir_bits;

/* I2C Pin Data In Register (ICPDIN) */
typedef struct {
  __REG32 PDIN0                   : 1;
  __REG32 PDIN1                   : 1;
  __REG32                         :30;
} __icpdin_bits;

/* I2C Pin Data Out Register (ICPDOUT) */
typedef struct {
  __REG32 PDOUT0                  : 1;
  __REG32 PDOUT1                  : 1;
  __REG32                         :30;
} __icpdout_bits;

/* I2C Pin Data Set Register (ICPDSET) */
typedef struct {
  __REG32 PDSET0                  : 1;
  __REG32 PDSET1                  : 1;
  __REG32                         :30;
} __icpdset_bits;

/* I2C Pin Data Clear Register (ICPDCLR) */
typedef struct {
  __REG32 PDCLR0                  : 1;
  __REG32 PDCLR1                  : 1;
  __REG32                         :30;
} __icpdclr_bits;

/* UART Receiver Buffer Register (RBR) */
typedef struct {
  __REG32 DATA                    : 8;
  __REG32                         :24;
} __uartrbr_bits;

/* UART Interrupt Enable Register (IER) */
typedef struct {
  __REG32 ERBI                    : 1;
  __REG32 ETBEI                   : 1;
  __REG32 ELSI                    : 1;
  __REG32                         :29;
} __uartier_bits;

/* UART Interrupt Identification Register (IIR) */
typedef union{
/* UARTx_IIR */
  struct {
  __REG32 IPEND                   : 1;
  __REG32 INTID                   : 3;
  __REG32                         : 2;
  __REG32 FIFOEN                  : 2;
  __REG32                         :24;
  };
/* UARTx_FCR */
  struct {
  __REG32 FIFOEN                  : 1;
  __REG32 RXCLR                   : 1;
  __REG32 TXCLR                   : 1;
  __REG32 DMAMODE1                : 1;
  __REG32                         : 2;
  __REG32 RXFIFTL                 : 2;
  __REG32                         :24;
  }UARTx_FCR;
} __uartiir_bits;

/* UART Line Control Register (LCR) */
typedef struct {
  __REG32 WLS                     : 2;
  __REG32 STB                     : 1;
  __REG32 PEN                     : 1;
  __REG32 EPS                     : 1;
  __REG32 SP                      : 1;
  __REG32 BC                      : 1;
  __REG32 DLAB                    : 1;
  __REG32                         :24;
} __uartlcr_bits;

/* UART Modem Control Register (MCR) */
typedef struct {
  __REG32                         : 1;
  __REG32 RTS                     : 1;
  __REG32 OUT1                    : 1;
  __REG32 OUT2                    : 1;
  __REG32 LOOP                    : 1;
  __REG32 AFE                     : 1;
  __REG32                         :26;
} __uartmcr_bits;

/* UART Line Status Register (LSR) */
typedef struct {
  __REG32 DR                      : 1;
  __REG32 OE                      : 1;
  __REG32 PE                      : 1;
  __REG32 FE                      : 1;
  __REG32 BI                      : 1;
  __REG32 THRE                    : 1;
  __REG32 TEMT                    : 1;
  __REG32 RXFIFOE                 : 1;
  __REG32                         :24;
} __uartlsr_bits;

/* UART Modem Status Register (MSR) */
typedef struct {
  __REG32 DCTS                    : 1;
  __REG32 DDSR                    : 1;
  __REG32 TERI                    : 1;
  __REG32 DCD                     : 1;
  __REG32 CTS                     : 1;
  __REG32 DSR                     : 1;
  __REG32 RI                      : 1;
  __REG32 CD                      : 1;
  __REG32                         :24;
} __uartmsr_bits;

/* UART Scratch Pad Register (SCR) */
typedef struct {
  __REG32 SCR                     : 8;
  __REG32                         :24;
} __uartscr_bits;

/* UART Divisor LSB Latch (DLL) */
typedef struct {
  __REG32 DLL                     : 8;
  __REG32                         :24;
} __uartdll_bits;

/* UART Divisor MSB Latch (DLH) */
typedef struct {
  __REG32 DLH                     : 8;
  __REG32                         :24;
} __uartdlh_bits;

/* UART Power and Emulation Management Register (PWREMU_MGMT) */
typedef struct {
  __REG32 FREE                    : 1;
  __REG32                         :12;
  __REG32 URRST                   : 1;
  __REG32 UTRST                   : 1;
  __REG32                         :17;
} __uartpwremu_mgmt_bits;

/* UART Mode Definition Register (MDR) */
typedef struct {
  __REG32 OSM_SEL                 : 1;
  __REG32                         :31;
} __uartmdr_bits;

/* EMAC Control Module Software Reset Register (SOFTRESET) */
typedef struct {
  __REG32 RESET                   : 1;
  __REG32                         :31;
} __emac_msoftreset_bits;

/* EMAC Control Module Interrupt Control Register (INTCONTROL) */
typedef struct {
  __REG32 INTPRESCALE             :12;
  __REG32                         : 4;
  __REG32 C0RXPACEEN              : 1;
  __REG32 C0TXPACEEN              : 1;
  __REG32 C1RXPACEEN              : 1;
  __REG32 C1TXPACEEN              : 1;
  __REG32 C2RXPACEEN              : 1;
  __REG32 C2TXPACEEN              : 1;
  __REG32                         :10;
} __emac_intcontrol_bits;

/* EMAC Control Module Interrupt Core Receive Threshold Interrupt Enable Registers (C0RXTHRESHEN-C2RXTHRESHEN) */
typedef struct {
  __REG32 RXCH0THRESHEN           : 1;
  __REG32 RXCH1THRESHEN           : 1;
  __REG32 RXCH2THRESHEN           : 1;
  __REG32 RXCH3THRESHEN           : 1;
  __REG32 RXCH4THRESHEN           : 1;
  __REG32 RXCH5THRESHEN           : 1;
  __REG32 RXCH6THRESHEN           : 1;
  __REG32 RXCH7THRESHEN           : 1;
  __REG32                         :24;
} __emac_crxthreshen_bits;

/* EMAC Control Module Interrupt Core Receive Interrupt Enable Registers (C0RXEN-C2RXEN) */
typedef struct {
  __REG32 RXCH0EN                 : 1;
  __REG32 RXCH1EN                 : 1;
  __REG32 RXCH2EN                 : 1;
  __REG32 RXCH3EN                 : 1;
  __REG32 RXCH4EN                 : 1;
  __REG32 RXCH5EN                 : 1;
  __REG32 RXCH6EN                 : 1;
  __REG32 RXCH7EN                 : 1;
  __REG32                         :24;
} __emac_crxen_bits;

/* EMAC Control Module Interrupt Core Transmit Interrupt Enable Registers (C0TXEN-C2TXEN) */
typedef struct {
  __REG32 TXCH0EN                 : 1;
  __REG32 TXCH1EN                 : 1;
  __REG32 TXCH2EN                 : 1;
  __REG32 TXCH3EN                 : 1;
  __REG32 TXCH4EN                 : 1;
  __REG32 TXCH5EN                 : 1;
  __REG32 TXCH6EN                 : 1;
  __REG32 TXCH7EN                 : 1;
  __REG32                         :24;
} __emac_ctxen_bits;

/* EMAC Control Module Interrupt Core Miscellaneous Interrupt Enable Registers (C0MISCEN-C2MISCEN) */
typedef struct {
  __REG32 USERINT0EN              : 1;
  __REG32 LINKINT0EN              : 1;
  __REG32 HOSTPENDEN              : 1;
  __REG32 STATPENDEN              : 1;
  __REG32                         :28;
} __emac_cmiscen_bits;

/* EMAC Control Module Interrupt Core Receive Threshold Interrupt Status Registers (C0RXTHRESHSTAT-C2RXTHRESHSTAT) */
typedef struct {
  __REG32 RXCH0THRESHSTAT         : 1;
  __REG32 RXCH1THRESHSTAT         : 1;
  __REG32 RXCH2THRESHSTAT         : 1;
  __REG32 RXCH3THRESHSTAT         : 1;
  __REG32 RXCH4THRESHSTAT         : 1;
  __REG32 RXCH5THRESHSTAT         : 1;
  __REG32 RXCH6THRESHSTAT         : 1;
  __REG32 RXCH7THRESHSTAT         : 1;
  __REG32                         :24;
} __emac_crxthreshstat_bits;

/* EMAC Control Module Interrupt Core Receive Interrupt Status Registers (C0RXSTAT-C2RXSTAT) */
typedef struct {
  __REG32 RXCH0STAT               : 1;
  __REG32 RXCH1STAT               : 1;
  __REG32 RXCH2STAT               : 1;
  __REG32 RXCH3STAT               : 1;
  __REG32 RXCH4STAT               : 1;
  __REG32 RXCH5STAT               : 1;
  __REG32 RXCH6STAT               : 1;
  __REG32 RXCH7STAT               : 1;
  __REG32                         :24;
} __emac_crxstat_bits;

/* EMAC Control Module Interrupt Core Transmit Interrupt Status Registers (C0TXSTAT-C2TXSTAT) */
typedef struct {
  __REG32 TXCH0STAT               : 1;
  __REG32 TXCH1STAT               : 1;
  __REG32 TXCH2STAT               : 1;
  __REG32 TXCH3STAT               : 1;
  __REG32 TXCH4STAT               : 1;
  __REG32 TXCH5STAT               : 1;
  __REG32 TXCH6STAT               : 1;
  __REG32 TXCH7STAT               : 1;
  __REG32                         :24;
} __emac_ctxstat_bits;

/* EMAC Control Module Interrupt Core Miscellaneous Interrupt Status Registers (C0MISCSTAT-C2MISCSTAT) */
typedef struct {
  __REG32 USERINT0STAT            : 1;
  __REG32 LINKINT0STAT            : 1;
  __REG32 HOSTPENDSTAT            : 1;
  __REG32 STATPENDSTAT            : 1;
  __REG32                         :28;
} __emac_cmiscstat_bits;

/* EMAC Control Module Interrupt Core Receive Interrupts Per Millisecond Registers (C0RXIMAX-C2RXIMAX) */
typedef struct {
  __REG32 RXIMAX                  : 6;
  __REG32                         :26;
} __emac_crximax_bits;

/* EMAC Control Module Interrupt Core Transmit Interrupts Per Millisecond Registers (C0TXIMAX-C2TXIMAX) */
typedef struct {
  __REG32 TXIMAX                  : 6;
  __REG32                         :26;
} __emac_ctximax_bits;

/* EMAC MDIO Control Register (CONTROL) */
typedef struct {
  __REG32 CLKDIV                  :16;
  __REG32                         : 2;
  __REG32 FAULTENB                : 1;
  __REG32 FAULT                   : 1;
  __REG32 PREAMBLE                : 1;
  __REG32                         : 3;
  __REG32 HIGHEST_USER_CHANNEL    : 5;
  __REG32                         : 1;
  __REG32 ENABLE                  : 1;
  __REG32 IDLE                    : 1;
} __mdio_control_bits;

/* EMAC MDIO PHY Acknowledge Status Register (ALIVE)*/
typedef struct {
  __REG32 ALIVE0                  : 1;
  __REG32 ALIVE1                  : 1;
  __REG32 ALIVE2                  : 1;
  __REG32 ALIVE3                  : 1;
  __REG32 ALIVE4                  : 1;
  __REG32 ALIVE5                  : 1;
  __REG32 ALIVE6                  : 1;
  __REG32 ALIVE7                  : 1;
  __REG32 ALIVE8                  : 1;
  __REG32 ALIVE9                  : 1;
  __REG32 ALIVE10                 : 1;
  __REG32 ALIVE11                 : 1;
  __REG32 ALIVE12                 : 1;
  __REG32 ALIVE13                 : 1;
  __REG32 ALIVE14                 : 1;
  __REG32 ALIVE15                 : 1;
  __REG32 ALIVE16                 : 1;
  __REG32 ALIVE17                 : 1;
  __REG32 ALIVE18                 : 1;
  __REG32 ALIVE19                 : 1;
  __REG32 ALIVE20                 : 1;
  __REG32 ALIVE21                 : 1;
  __REG32 ALIVE22                 : 1;
  __REG32 ALIVE23                 : 1;
  __REG32 ALIVE24                 : 1;
  __REG32 ALIVE25                 : 1;
  __REG32 ALIVE26                 : 1;
  __REG32 ALIVE27                 : 1;
  __REG32 ALIVE28                 : 1;
  __REG32 ALIVE29                 : 1;
  __REG32 ALIVE30                 : 1;
  __REG32 ALIVE31                 : 1;
} __mdio_alive_bits;

/* EMAC PHY Link Status Register (LINK) */
typedef struct {
  __REG32 LINKE0                  : 1;
  __REG32 LINKE1                  : 1;
  __REG32 LINKE2                  : 1;
  __REG32 LINKE3                  : 1;
  __REG32 LINKE4                  : 1;
  __REG32 LINKE5                  : 1;
  __REG32 LINKE6                  : 1;
  __REG32 LINKE7                  : 1;
  __REG32 LINKE8                  : 1;
  __REG32 LINKE9                  : 1;
  __REG32 LINKE10                 : 1;
  __REG32 LINKE11                 : 1;
  __REG32 LINKE12                 : 1;
  __REG32 LINKE13                 : 1;
  __REG32 LINKE14                 : 1;
  __REG32 LINKE15                 : 1;
  __REG32 LINKE16                 : 1;
  __REG32 LINKE17                 : 1;
  __REG32 LINKE18                 : 1;
  __REG32 LINKE19                 : 1;
  __REG32 LINKE20                 : 1;
  __REG32 LINKE21                 : 1;
  __REG32 LINKE22                 : 1;
  __REG32 LINKE23                 : 1;
  __REG32 LINKE24                 : 1;
  __REG32 LINKE25                 : 1;
  __REG32 LINKE26                 : 1;
  __REG32 LINKE27                 : 1;
  __REG32 LINKE28                 : 1;
  __REG32 LINKE29                 : 1;
  __REG32 LINKE30                 : 1;
  __REG32 LINKE31                 : 1;
} __mdio_link_bits;

/* EMAC MDIO Link Status Change Interrupt (Unmasked) Register (LINKINTRAW) */
/* EMAC MDIO Link Status Change Interrupt (Masked) Register (LINKINTMASKED) */
typedef struct {
  __REG32 USERPHY0                : 1;
  __REG32 USERPHY1                : 1;
  __REG32                         :30;
} __mdio_linkintraw_bits;

/* EMAC MDIO User Command Complete Interrupt (Unmasked) Register (USERINTRAW) */
/* EMAC MDIO User Command Complete Interrupt (Masked) Register (USERINTMASKED) */
/* EMAC MDIO User Command Complete Interrupt Mask Set Register (USERINTMASKSET) */
/* EMAC MDIO User Command Complete Interrupt Mask Clear Register (USERINTMASKCLEAR) */
typedef struct {
  __REG32 USERACCESS0             : 1;
  __REG32 USERACCESS1             : 1;
  __REG32                         :30;
} __mdio_userintraw_bits;

/* EMAC MDIO User Access Register 0/1 (USERACCESS0/1) */
typedef struct {
  __REG32 DATA                    :16;
  __REG32 PHYADR                  : 5;
  __REG32 REGADR                  : 5;
  __REG32                         : 3;
  __REG32 ACK                     : 1;
  __REG32 WRITE                   : 1;
  __REG32 GO                      : 1;
} __mdio_useraccess_bits;

/* EMAC MDIO User PHY Select Register 0/1 (USERPHYSEL0/1) */
typedef struct {
  __REG32 PHYADRMON               : 5;
  __REG32                         : 1;
  __REG32 LINKINTENB              : 1;
  __REG32 LINKSEL                 : 1;
  __REG32                         :24;
} __mdio_userphysel_bits;

/* EMAC Transmit Control Register (TXCONTROL) */
typedef struct {
  __REG32 TXEN                    : 1;
  __REG32                         :31;
} __emac_txcontrol_bits;

/* EMAC Transmit Teardown Register (TXTEARDOWN) */
typedef struct {
  __REG32 TXTDNCH                 : 3;
  __REG32                         :29;
} __emac_txteardown_bits;

/* EMAC Receive Control Register (RXCONTROL) */
typedef struct {
  __REG32 RXEN                    : 1;
  __REG32                         :31;
} __emac_rxcontrol_bits;

/* EMAC Receive Teardown Register (RXTEARDOWN) */
typedef struct {
  __REG32 RXTDNCH                 : 3;
  __REG32                         :29;
} __emac_rxteardown_bits;

/* EMAC Transmit Interrupt Status (Unmasked) Register (TXINTSTATRAW) */
/* EMAC Transmit Interrupt Status (Masked) Register (TXINTSTATMASKED) */
typedef struct {
  __REG32 TX0PEND                 : 1;
  __REG32 TX1PEND                 : 1;
  __REG32 TX2PEND                 : 1;
  __REG32 TX3PEND                 : 1;
  __REG32 TX4PEND                 : 1;
  __REG32 TX5PEND                 : 1;
  __REG32 TX6PEND                 : 1;
  __REG32 TX7PEND                 : 1;
  __REG32                         :24;
} __emac_txintstatraw_bits;

/* EMAC Transmit Interrupt Mask Set Register (TXINTMASKSET) */
/* EMAC Transmit Interrupt Mask Clear Register (TXINTMASKCLEAR) */
typedef struct {
  __REG32 TX0MASK                 : 1;
  __REG32 TX1MASK                 : 1;
  __REG32 TX2MASK                 : 1;
  __REG32 TX3MASK                 : 1;
  __REG32 TX4MASK                 : 1;
  __REG32 TX5MASK                 : 1;
  __REG32 TX6MASK                 : 1;
  __REG32 TX7MASK                 : 1;
  __REG32                         :24;
} __emac_txintmaskset_bits;

/* EMAC MAC Input Vector Register (MACINVECTOR) */
typedef struct {
  __REG32 RXPEND                  : 8;
  __REG32 RXTHRESHPEND            : 8;
  __REG32 TXPEND                  : 8;
  __REG32 USERINT0                : 1;
  __REG32 LINKINT0                : 1;
  __REG32 HOSTPEND                : 1;
  __REG32 STATPEND                : 1;
  __REG32                         : 4;
} __emac_macinvector_bits;

/* EMAC MAC End Of Interrupt Vector Register (MACEOIVECTOR) */
typedef struct {
  __REG32 INTVECT                 : 5;
  __REG32                         :27;
} __emac_maceoivector_bits;

/* EMAC Receive Interrupt Status (Unmasked) Register (RXINTSTATRAW) */
/* EMAC Receive Interrupt Status (Masked) Register (RXINTSTATMASKED) */
typedef struct {
  __REG32 RX0PEND                 : 1;
  __REG32 RX1PEND                 : 1;
  __REG32 RX2PEND                 : 1;
  __REG32 RX3PEND                 : 1;
  __REG32 RX4PEND                 : 1;
  __REG32 RX5PEND                 : 1;
  __REG32 RX6PEND                 : 1;
  __REG32 RX7PEND                 : 1;
  __REG32 RX0THRESHPEND           : 1;
  __REG32 RX1THRESHPEND           : 1;
  __REG32 RX2THRESHPEND           : 1;
  __REG32 RX3THRESHPEND           : 1;
  __REG32 RX4THRESHPEND           : 1;
  __REG32 RX5THRESHPEND           : 1;
  __REG32 RX6THRESHPEND           : 1;
  __REG32 RX7THRESHPEND           : 1;
  __REG32                         :16;
} __emac_rxintstatraw_bits;

/* EMAC Receive Interrupt Mask Set Register (RXINTMASKSET) */
/* EMAC Receive Interrupt Mask Clear Register (RXINTMASKCLEAR) */
typedef struct {
  __REG32 RX0MASK                 : 1;
  __REG32 RX1MASK                 : 1;
  __REG32 RX2MASK                 : 1;
  __REG32 RX3MASK                 : 1;
  __REG32 RX4MASK                 : 1;
  __REG32 RX5MASK                 : 1;
  __REG32 RX6MASK                 : 1;
  __REG32 RX7MASK                 : 1;
  __REG32 RX0THRESHMASK           : 1;
  __REG32 RX1THRESHMASK           : 1;
  __REG32 RX2THRESHMASK           : 1;
  __REG32 RX3THRESHMASK           : 1;
  __REG32 RX4THRESHMASK           : 1;
  __REG32 RX5THRESHMASK           : 1;
  __REG32 RX6THRESHMASK           : 1;
  __REG32 RX7THRESHMASK           : 1;
  __REG32                         :16;
} __emac_rxintmaskset_bits;

/* EMAC MAC Interrupt Status (Unmasked) Register (MACINTSTATRAW) */
/* EMAC MAC Interrupt Status (Masked) Register (MACINTSTATMASKED) */
typedef struct {
  __REG32 STATPEND                : 1;
  __REG32 HOSTPEND                : 1;
  __REG32                         :30;
} __emac_macintstatraw_bits;

/* EMAC MAC Interrupt Mask Set Register (MACINTMASKSET) */
/* EMAC MAC Interrupt Mask Clear Register (MACINTMASKCLEAR) */
typedef struct {
  __REG32 STATMASK                : 1;
  __REG32 HOSTMASK                : 1;
  __REG32                         :30;
} __emac_macintmaskset_bits;

/* EMAC Receive Multicast/Broadcast/Promiscuous Channel Enable Register (RXMBPENABLE) */
typedef struct {
  __REG32 RXMULTCH                : 3;
  __REG32                         : 2;
  __REG32 RXMULTEN                : 1;
  __REG32                         : 2;
  __REG32 RXBROADCH               : 3;
  __REG32                         : 2;
  __REG32 RXBROADEN               : 1;
  __REG32                         : 2;
  __REG32 RXPROMCH                : 3;
  __REG32                         : 2;
  __REG32 RXCAFEN                 : 1;
  __REG32 RXCEFEN                 : 1;
  __REG32 RXCSFEN                 : 1;
  __REG32 RXCMFEN                 : 1;
  __REG32                         : 3;
  __REG32 RXNOCHAIN               : 1;
  __REG32 RXQOSEN                 : 1;
  __REG32 RXPASSCRC               : 1;
  __REG32                         : 1;
} __emac_rxmbpenable_bits;

/* EMAC Receive Unicast Enable Set Register (RXUNICASTSET) */
/* EMAC Receive Unicast Clear Register (RXUNICASTCLEAR) */
typedef struct {
  __REG32 RXCH0EN                 : 1;
  __REG32 RXCH1EN                 : 1;
  __REG32 RXCH2EN                 : 1;
  __REG32 RXCH3EN                 : 1;
  __REG32 RXCH4EN                 : 1;
  __REG32 RXCH5EN                 : 1;
  __REG32 RXCH6EN                 : 1;
  __REG32 RXCH7EN                 : 1;
  __REG32                         :24;
} __emac_rxunicastset_bits;

/* EMAC Receive Maximum Length Register (RXMAXLEN) */
typedef struct {
  __REG32 RXMAXLEN                :16;
  __REG32                         :16;
} __emac_rxmaxlen_bits;

/* EMAC Receive Buffer Offset Register (RXBUFFEROFFSET) */
typedef struct {
  __REG32 RXBUFFEROFFSET          :16;
  __REG32                         :16;
} __emac_rxbufferoffset_bits;

/* EMAC Receive Filter Low Priority Frame Threshold Register (RXFILTERLOWTHRESH) */
typedef struct {
  __REG32 RXFILTERTHRESH          : 8;
  __REG32                         :24;
} __emac_rxfilterlowthresh_bits;

/* EMAC Receive Channel Flow Control Threshold Registers (RX0FLOWTHRESH-RX7FLOWTHRESH) */
typedef struct {
  __REG32 RXFLOWTHRESH            : 8;
  __REG32                         :24;
} __emac_rxflowthresh_bits;

/* EMAC Receive Channel Free Buffer Count Registers (RX0FREEBUFFER-RX7FREEBUFFER) */
typedef struct {
  __REG32 RXFREEBUF               :16;
  __REG32                         :16;
} __emac_rxfreebuffer_bits;

/* EMAC MAC Control Register (MACCONTROL) */
typedef struct {
  __REG32 FULLDUPLEX              : 1;
  __REG32 LOOPBACK                : 1;
  __REG32                         : 1;
  __REG32 RXBUFFERFLOWEN          : 1;
  __REG32 TXFLOWEN                : 1;
  __REG32 GMIIEN                  : 1;
  __REG32 TXPACE                  : 1;
  __REG32                         : 2;
  __REG32 TXPTYPE                 : 1;
  __REG32 TXSHORTGAPEN            : 1;
  __REG32 CMDIDLE                 : 1;
  __REG32                         : 1;
  __REG32 RXOWNERSHIP             : 1;
  __REG32 RXOFFLENBLOCK           : 1;
  __REG32 RMIISPEED               : 1;
  __REG32                         :16;
} __emac_maccontrol_bits;

/* EMAC MAC Status Register (MACSTATUS) */
typedef struct {
  __REG32 TXFLOWACT               : 1;
  __REG32 RXFLOWACT               : 1;
  __REG32 RXQOSACT                : 1;
  __REG32                         : 5;
  __REG32 RXERRCH                 : 3;
  __REG32                         : 1;
  __REG32 RXERRCODE               : 4;
  __REG32 TXERRCH                 : 3;
  __REG32                         : 1;
  __REG32 TXERRCODE               : 4;
  __REG32                         : 7;
  __REG32 IDLE                    : 1;
} __emac_macstatus_bits;

/* EMAC Emulation Control Register (EMCONTROL) */
typedef struct {
  __REG32 FREE                    : 1;
  __REG32 SOFT                    : 1;
  __REG32                         :30;
} __emac_emcontrol_bits;

/* EMAC FIFO Control Register (FIFOCONTROL) */
typedef struct {
  __REG32 TXCELLTHRESH            : 2;
  __REG32                         :30;
} __emac_fifocontrol_bits;

/* EMAC MAC Configuration Register (MACCONFIG) */
typedef struct {
  __REG32 MACCFIG                 : 8;
  __REG32 ADDRESSTYPE             : 8;
  __REG32 RXCELLDEPTH             : 8;
  __REG32 TXCELLDEPTH             : 8;
} __emac_macconfig_bits;

/* EMAC Soft Reset Register (SOFTRESET) */
typedef struct {
  __REG32 SOFTRESET               : 1;
  __REG32                         :31;
} __emac_softreset_bits;

/* EMAC MAC Source Address Low Bytes Register (MACSRCADDRLO) */
typedef struct {
  __REG32 MACSRCADDR1             : 8;
  __REG32 MACSRCADDR0             : 8;
  __REG32                         :16;
} __emac_macsrcaddrlo_bits;

/* EMAC MAC Source Address High Bytes Register (MACSRCADDRHI) */
typedef struct {
  __REG32 MACSRCADDR5             : 8;
  __REG32 MACSRCADDR4             : 8;
  __REG32 MACSRCADDR3             : 8;
  __REG32 MACSRCADDR2             : 8;
} __emac_macsrcaddrhi_bits;

/* EMAC Back Off Test Register (BOFFTEST) */
typedef struct {
  __REG32 TXBACKOFF               :10;
  __REG32                         : 2;
  __REG32 COLLCOUNT               : 4;
  __REG32 RNDNUM                  :10;
  __REG32                         : 6;
} __emac_bofftest_bits;

/* EMAC Transmit Pacing Algorithm Test Register (TPACETEST) */
typedef struct {
  __REG32 PACEVAL                 : 5;
  __REG32                         :27;
} __emac_tpacetest_bits;

/* EMAC Receive Pause Timer Register (RXPAUSE) */
/* EMAC Transmit Pause Timer Register (TXPAUSE) */
typedef struct {
  __REG32 PAUSETIMER              :16;
  __REG32                         :16;
} __emac_rxpause_bits;

/* EMAC MAC Address Low Bytes Register (MACADDRLO) */
typedef struct {
  __REG32 MACADDR1                : 8;
  __REG32 MACADDR0                : 8;
  __REG32 CHANNEL                 : 3;
  __REG32 MATCHFILT               : 1;
  __REG32 VALID                   : 1;
  __REG32                         :11;
} __emac_macaddrlo_bits;

/* EMAC MAC Address High Bytes Register (MACADDRHI) */
typedef struct {
  __REG32 MACADDR5                : 8;
  __REG32 MACADDR4                : 8;
  __REG32 MACADDR3                : 8;
  __REG32 MACADDR2                : 8;
} __emac_macaddrhi_bits;

/* EMAC MAC Index Register (MACINDEX) */
typedef struct {
  __REG32 MACINDEX                : 3;
  __REG32                         :29;
} __emac_macindex_bits;

/* ECAP Control Register 1 (ECCTL1) */
typedef struct {
  __REG16 CAP1POL                 : 1;
  __REG16 CTRRST1                 : 1;
  __REG16 CAP2POL                 : 1;
  __REG16 CTRRST2                 : 1;
  __REG16 CAP3POL                 : 1;
  __REG16 CTRRST3                 : 1;
  __REG16 CAP4POL                 : 1;
  __REG16 CTRRST4                 : 1;
  __REG16 CAPLDEN                 : 1;
  __REG16 PRESCALE                : 5;
  __REG16 FREE_SOFT               : 2;
} __ecap_ecctl1_bits;

/* ECAP Control Register 2 (ECCTL2) */
typedef struct {
  __REG16 CONT_ONESHT             : 1;
  __REG16 STOP_WRAP               : 2;
  __REG16 REARM                   : 1;
  __REG16 TSCTRSTOP               : 1;
  __REG16 SYNCI_EN                : 1;
  __REG16 SYNCO_SEL               : 2;
  __REG16 SWSYNC                  : 1;
  __REG16 CAP_APWM                : 1;
  __REG16 APWMPOL                 : 1;
  __REG16                         : 5;
} __ecap_ecctl2_bits;

/* ECAP Interrupt Enable Register (ECEINT) */
/* ECAP Interrupt Flag Register (ECFLG) */
/* ECAP Interrupt Clear Register (ECCLR) */
typedef struct {
  __REG16                         : 1;
  __REG16 CETV1                   : 1;
  __REG16 CEVT2                   : 1;
  __REG16 CEVT3                   : 1;
  __REG16 CEVT4                   : 1;
  __REG16 CTROVF                  : 1;
  __REG16 CTR_PRD                 : 1;
  __REG16 CTR_CMP                 : 1;
  __REG16                         : 8;
} __ecap_eceint_bits;

/* QEP Decoder Control Register (QDECCTL) */
typedef struct {
  __REG16                         : 5;
  __REG16 QSP                     : 1;
  __REG16 QIP                     : 1;
  __REG16 QBP                     : 1;
  __REG16 QAP                     : 1;
  __REG16 IGATE                   : 1;
  __REG16 SWAP                    : 1;
  __REG16 XCR                     : 1;
  __REG16 SPSEL                   : 1;
  __REG16 SOEN                    : 1;
  __REG16 QSRC                    : 2;
} __eqep_qdecctl_bits;

/* eQEP Control Register (QEPCTL) */
typedef struct {
  __REG16 WDE                     : 1;
  __REG16 UTE                     : 1;
  __REG16 QCLM                    : 1;
  __REG16 PHEN                    : 1;
  __REG16 IEL                     : 2;
  __REG16 SEL                     : 1;
  __REG16 SWI                     : 1;
  __REG16 IEI                     : 2;
  __REG16 SEI                     : 2;
  __REG16 PCRM                    : 2;
  __REG16 FREE_SOFT               : 2;
} __eqep_qepctl_bits;

/* eQEP Capture Control Register (QCAPCTL) */
typedef struct {
  __REG16 UPPS                    : 4;
  __REG16 CCPS                    : 3;
  __REG16                         : 8;
  __REG16 CEN                     : 1;
} __eqep_qcapctl_bits;

/* eQEP Position-Compare Control Register (QPOSCTL) */
typedef struct {
  __REG16 PCSPW                   :12;
  __REG16 PCE                     : 1;
  __REG16 PCPOL                   : 1;
  __REG16 PCLOAD                  : 1;
  __REG16 PCSHDW                  : 1;
} __eqep_qposctl_bits;

/* eQEP Interrupt Enable Register (QEINT) */
/* eQEP Interrupt Force Register (QFRC) */
typedef struct {
  __REG16                         : 1;
  __REG16 PCE                     : 1;
  __REG16 PHE                     : 1;
  __REG16 QDC                     : 1;
  __REG16 WTO                     : 1;
  __REG16 PCU                     : 1;
  __REG16 PCO                     : 1;
  __REG16 PCR                     : 1;
  __REG16 PCM                     : 1;
  __REG16 SEL                     : 1;
  __REG16 IEL                     : 1;
  __REG16 UTO                     : 1;
  __REG16                         : 4;
} __eqep_qeint_bits;

/* eQEP Interrupt Flag Register (QFLG) */
/* eQEP Interrupt Clear Register (QCLR) */
typedef struct {
  __REG16 INT                     : 1;
  __REG16 PCE                     : 1;
  __REG16 PHE                     : 1;
  __REG16 QDC                     : 1;
  __REG16 WTO                     : 1;
  __REG16 PCU                     : 1;
  __REG16 PCO                     : 1;
  __REG16 PCR                     : 1;
  __REG16 PCM                     : 1;
  __REG16 SEL                     : 1;
  __REG16 IEL                     : 1;
  __REG16 UTO                     : 1;
  __REG16                         : 4;
} __eqep_qflg_bits;

/* eQEP Status Register (QEPSTS) */
typedef struct {
  __REG16 PCEF                    : 1;
  __REG16 FIMF                    : 1;
  __REG16 CDEF                    : 1;
  __REG16 COEF                    : 1;
  __REG16 QDLF                    : 1;
  __REG16 QDF                     : 1;
  __REG16 FIDF                    : 1;
  __REG16 UPEVNT                  : 1;
  __REG16                         : 8;
} __eqep_qepsts_bits;

/* eHRPWM Time-Base Control Register (TBCTL) */
typedef struct {
  __REG16 CTRMODE                 : 2;
  __REG16 PHSEN                   : 1;
  __REG16 PRDLD                   : 1;
  __REG16 SYNCOSEL                : 2;
  __REG16 SWFSYNC                 : 1;
  __REG16 HSPCLKDIV               : 3;
  __REG16 CLKDIV                  : 3;
  __REG16 PHSDIR                  : 1;
  __REG16 FREE_SOFT               : 2;
} __ehrpwm_tbctl_bits;

/* eHRPWM Time-Base Status Register (TBSTS) */
typedef struct {
  __REG16 CTRDIR                  : 1;
  __REG16 SYNCI                   : 1;
  __REG16 CTRMAX                  : 1;
  __REG16                         :13;
} __ehrpwm_tbsts_bits;

/* eHRPWM Counter-Compare Control Register (CMPCTL) */
typedef struct {
  __REG16 LOADAMODE               : 2;
  __REG16 LOADBMODE               : 2;
  __REG16 SHDWAMODE               : 1;
  __REG16                         : 1;
  __REG16 SHDWBMODE               : 1;
  __REG16                         : 1;
  __REG16 SHDWAFULL               : 1;
  __REG16 SHDWBFULL               : 1;
  __REG16                         : 6;
} __ehrpwm_cmpctl_bits;

/* eHRPWM Action-Qualifier Output A Control Register (AQCTLA) */
/* eHRPWM Action-Qualifier Output B Control Register (AQCTLB) */
typedef struct {
  __REG16 ZRO                     : 2;
  __REG16 PRD                     : 2;
  __REG16 CAU                     : 2;
  __REG16 CAD                     : 2;
  __REG16 CBU                     : 2;
  __REG16 CBD                     : 2;
  __REG16                         : 4;
} __ehrpwm_aqctla_bits;

/* eHRPWM Action-Qualifier Software Force Register (AQSFRC) */
typedef struct {
  __REG16 ACTSFA                  : 2;
  __REG16 OTSFA                   : 1;
  __REG16 ACTSFB                  : 2;
  __REG16 OTSFB                   : 1;
  __REG16 RLDCSF                  : 2;
  __REG16                         : 8;
} __ehrpwm_aqsfrc_bits;

/* eHRPWM Action-Qualifier Continuous Software Force Register (AQCSFRC) */
typedef struct {
  __REG16 CSFA                    : 2;
  __REG16 CSFB                    : 2;
  __REG16                         :12;
} __ehrpwm_aqcsfrc_bits;

/* eHRPWM Dead-Band Generator Control Register (DBCTL) */
typedef struct {
  __REG16 OUT_MODE                : 2;
  __REG16 POLSEL                  : 2;
  __REG16 IN_MODE                 : 2;
  __REG16                         :10;
} __ehrpwm_dbctl_bits;

/* eHRPWM Dead-Band Generator Rising Edge Delay Register (DBRED) */
/* eHRPWM Dead-Band Generator Falling Edge Delay Register (DBFED) */
typedef struct {
  __REG16 DEL                     :10;
  __REG16                         : 6;
} __ehrpwm_dbred_bits;

/* eHRPWM PWM-Chopper Submodule Register */
typedef struct {
  __REG16 CHPEN                   : 1;
  __REG16 OSHTWTH                 : 4;
  __REG16 CHPFREQ                 : 3;
  __REG16 CHPDUTY                 : 3;
  __REG16                         : 5;
} __ehrpwm_pcctl_bits;

/* eHRPWM Trip-Zone Select Register (TZSEL) */
typedef struct {
  __REG16 CBC1                    : 1;
  __REG16 CBC2                    : 1;
  __REG16 CBC3                    : 1;
  __REG16 CBC4                    : 1;
  __REG16 CBC5                    : 1;
  __REG16 CBC6                    : 1;
  __REG16 CBC7                    : 1;
  __REG16 CBC8                    : 1;
  __REG16 OSHT1                   : 1;
  __REG16 OSHT2                   : 1;
  __REG16 OSHT3                   : 1;
  __REG16 OSHT4                   : 1;
  __REG16 OSHT5                   : 1;
  __REG16 OSHT6                   : 1;
  __REG16 OSHT7                   : 1;
  __REG16 OSHT8                   : 1;
} __ehrpwm_tzsel_bits;

/* eHRPWM Trip-Zone Control Register (TZCTL) */
typedef struct {
  __REG16 TZA                     : 2;
  __REG16 TZB                     : 2;
  __REG16                         :12;
} __ehrpwm_tzctl_bits;

/* eHRPWM Trip-Zone Enable Interrupt Register (TZEINT) */
/* eHRPWM Trip-Zone Force Register (TZFRC) */
typedef struct {
  __REG16                         : 1;
  __REG16 CBC                     : 1;
  __REG16 OST                     : 1;
  __REG16                         :13;
} __ehrpwm_tzeint_bits;

/* eHRPWM Trip-Zone Flag Register (TZFLG) */
/* eHRPWM Trip-Zone Clear Register (TZCLR) */
typedef struct {
  __REG16 INT                     : 1;
  __REG16 CBC                     : 1;
  __REG16 OST                     : 1;
  __REG16                         :13;
} __ehrpwm_tzflg_bits;

/* eHRPWM Event-Trigger Selection Register (ETSEL) */
typedef struct {
  __REG16 INTSEL                  : 3;
  __REG16 INTEN                   : 1;
  __REG16                         :12;
} __ehrpwm_etsel_bits;

/* eHRPWM Event-Trigger Prescale Register (ETPS) */
typedef struct {
  __REG16 INTPRD                  : 2;
  __REG16 INTCNT                  : 2;
  __REG16                         :12;
} __ehrpwm_etps_bits;

/* eHRPWM Event-Trigger Prescale Register (ETPS) */
/* eHRPWM Event-Trigger Clear Register (ETCLR) */
/* eHRPWM Event-Trigger Force Register (ETFRC) */
typedef struct {
  __REG16 INT                     : 1;
  __REG16                         :15;
} __ehrpwm_etflg_bits;

/* eHRPWM Time-Base Phase High-Resolution Register (TBPHSHR) */
typedef struct {
  __REG16                         : 8;
  __REG16 TBPHSH                  : 8;
} __ehrpwm_tbphshr_bits;

/* eHRPWM Counter-Compare A High-Resolution Register (CMPAHR) */
typedef struct {
  __REG16                         : 8;
  __REG16 CMPAHR                  : 8;
} __ehrpwm_cmpahr_bits;

/* eHRPWM Configuration Register (HRCNFG) */
typedef struct {
  __REG16 EDGMODE                 : 2;
  __REG16 CTLMODE                 : 1;
  __REG16 HRLOAD                  : 1;
  __REG16                         :12;
} __ehrpwm_hrcnfg_bits;

/* TIMER64P Emulation Management Register (EMUMGT) */
typedef struct {
  __REG32 FREE                    : 1;
  __REG32 SOFT                    : 1;
  __REG32                         :30;
} __timer64p_emumgt_bits;

/* TIMER64P GPIO Interrupt Control and Enable Register (GPINTGPEN) */
typedef struct {
  __REG32 GPINT12ENI              : 1;
  __REG32 GPINT12ENO              : 1;
  __REG32                         : 2;
  __REG32 GPINT12INVI             : 1;
  __REG32 GPINT12INVO             : 1;
  __REG32                         : 2;
  __REG32 GPINT34ENI              : 1;
  __REG32 GPINT34ENO              : 1;
  __REG32                         : 2;
  __REG32 GPINT34INVI             : 1;
  __REG32 GPINT34INVO             : 1;
  __REG32                         : 2;
  __REG32 GPENI12                 : 1;
  __REG32 GPENO12                 : 1;
  __REG32                         : 6;
  __REG32 GPENI34                 : 1;
  __REG32 GPENO34                 : 1;
  __REG32                         : 6;
} __timer64p_gpintgpen_bits;

/* TIMER64P GPIO Data and Direction Register (GPDATGPDIR) */
typedef struct {
  __REG32 GPDATI12                : 1;
  __REG32 GPDATO12                : 1;
  __REG32                         : 6;
  __REG32 GPDATI34                : 1;
  __REG32 GPDATO34                : 1;
  __REG32                         : 6;
  __REG32 GPDIRI12                : 1;
  __REG32 GPDIRO12                : 1;
  __REG32                         : 6;
  __REG32 GPDIRI34                : 1;
  __REG32 GPDIRO34                : 1;
  __REG32                         : 6;
} __timer64p_gpdatgpdir_bits;

/* TIMER64P Timer Control Register (TCR) */
typedef struct {
  __REG32 TSTAT12                 : 1;
  __REG32 INVOUTP12               : 1;
  __REG32 INVINP12                : 1;
  __REG32 CP12                    : 1;
  __REG32 PWID12                  : 2;
  __REG32 ENAMODE12               : 2;
  __REG32 CLKSRC12                : 1;
  __REG32 TIEN12                  : 1;
  __REG32 READRSTMODE12           : 1;
  __REG32 CAPMODE12               : 1;
  __REG32 CAPVTMODE12             : 2;
  __REG32                         : 2;
  __REG32 TSTAT34                 : 1;
  __REG32 INVOUTP34               : 1;
  __REG32 INVINP34                : 1;
  __REG32 CP34                    : 1;
  __REG32 PWID34                  : 2;
  __REG32 ENAMODE34               : 2;
  __REG32 CLKSRC34                : 1;
  __REG32 TIEN34                  : 1;
  __REG32 READRSTMODE34           : 1;
  __REG32 CAPMODE34               : 1;
  __REG32 CAPEVTMODE34            : 2;
  __REG32                         : 2;
} __timer64p_tcr_bits;

/* TIMER64P Timer Global Control Register (TGCR) */
typedef struct {
  __REG32 TIM12RS                 : 1;
  __REG32 TIM34RS                 : 1;
  __REG32 TIMMODE                 : 2;
  __REG32 PLUSEN                  : 1;
  __REG32                         : 3;
  __REG32 PSC34                   : 4;
  __REG32 TDDR34                  : 4;
  __REG32                         :16;
} __timer64p_tgcr_bits;

/* TIMER64P Watchdog Timer Control Register (WDTCR) */
typedef struct {
  __REG32                         :14;
  __REG32 WDEN                    : 1;
  __REG32 WDFLAG                  : 1;
  __REG32 WDKEY                   :16;
} __timer64p_wdtcr_bits;

/* TIMER64P Timer Interrupt Control and Status Register (INTCTLSTAT) */
typedef struct {
  __REG32 PRDINTEN12              : 1;
  __REG32 PRDINTSTAT12            : 1;
  __REG32 EVTINTEN12              : 1;
  __REG32 EVTINTSTAT12            : 1;
  __REG32                         :12;
  __REG32 PRDINTEN34              : 1;
  __REG32 PRDINTSTAT34            : 1;
  __REG32 EVTINTEN34              : 1;
  __REG32 EVTINTSTAT34            : 1;
  __REG32                         :12;
} __timer64p_intctlstat_bits;

/* GPIO Interrupt Per-Bank Enable Register (BINTEN) */
typedef struct {
  __REG32 EN0                     : 1;
  __REG32 EN1                     : 1;
  __REG32 EN2                     : 1;
  __REG32 EN3                     : 1;
  __REG32 EN4                     : 1;
  __REG32 EN5                     : 1;
  __REG32 EN6                     : 1;
  __REG32 EN7                     : 1;
  __REG32                         :24;
} __gpio_binten_bits;

/* GPIO Banks 0 and 1 Direction Register (DIR01) */
typedef struct {
  __REG32 GP0P0                   : 1;
  __REG32 GP0P1                   : 1;
  __REG32 GP0P2                   : 1;
  __REG32 GP0P3                   : 1;
  __REG32 GP0P4                   : 1;
  __REG32 GP0P5                   : 1;
  __REG32 GP0P6                   : 1;
  __REG32 GP0P7                   : 1;
  __REG32 GP0P8                   : 1;
  __REG32 GP0P9                   : 1;
  __REG32 GP0P10                  : 1;
  __REG32 GP0P11                  : 1;
  __REG32 GP0P12                  : 1;
  __REG32 GP0P13                  : 1;
  __REG32 GP0P14                  : 1;
  __REG32 GP0P15                  : 1;
  __REG32 GP1P0                   : 1;
  __REG32 GP1P1                   : 1;
  __REG32 GP1P2                   : 1;
  __REG32 GP1P3                   : 1;
  __REG32 GP1P4                   : 1;
  __REG32 GP1P5                   : 1;
  __REG32 GP1P6                   : 1;
  __REG32 GP1P7                   : 1;
  __REG32 GP1P8                   : 1;
  __REG32 GP1P9                   : 1;
  __REG32 GP1P10                  : 1;
  __REG32 GP1P11                  : 1;
  __REG32 GP1P12                  : 1;
  __REG32 GP1P13                  : 1;
  __REG32 GP1P14                  : 1;
  __REG32 GP1P15                  : 1;
} __gpio_dir01_bits;

/* GPIO Banks 2 and 3 Direction Register (DIR23) */
typedef struct {
  __REG32 GP2P0                   : 1;
  __REG32 GP2P1                   : 1;
  __REG32 GP2P2                   : 1;
  __REG32 GP2P3                   : 1;
  __REG32 GP2P4                   : 1;
  __REG32 GP2P5                   : 1;
  __REG32 GP2P6                   : 1;
  __REG32 GP2P7                   : 1;
  __REG32 GP2P8                   : 1;
  __REG32 GP2P9                   : 1;
  __REG32 GP2P10                  : 1;
  __REG32 GP2P11                  : 1;
  __REG32 GP2P12                  : 1;
  __REG32 GP2P13                  : 1;
  __REG32 GP2P14                  : 1;
  __REG32 GP2P15                  : 1;
  __REG32 GP3P0                   : 1;
  __REG32 GP3P1                   : 1;
  __REG32 GP3P2                   : 1;
  __REG32 GP3P3                   : 1;
  __REG32 GP3P4                   : 1;
  __REG32 GP3P5                   : 1;
  __REG32 GP3P6                   : 1;
  __REG32 GP3P7                   : 1;
  __REG32 GP3P8                   : 1;
  __REG32 GP3P9                   : 1;
  __REG32 GP3P10                  : 1;
  __REG32 GP3P11                  : 1;
  __REG32 GP3P12                  : 1;
  __REG32 GP3P13                  : 1;
  __REG32 GP3P14                  : 1;
  __REG32 GP3P15                  : 1;
} __gpio_dir23_bits;

/* GPIO Banks 4 and 5 Direction Register (DIR45) */
typedef struct {
  __REG32 GP4P0                   : 1;
  __REG32 GP4P1                   : 1;
  __REG32 GP4P2                   : 1;
  __REG32 GP4P3                   : 1;
  __REG32 GP4P4                   : 1;
  __REG32 GP4P5                   : 1;
  __REG32 GP4P6                   : 1;
  __REG32 GP4P7                   : 1;
  __REG32 GP4P8                   : 1;
  __REG32 GP4P9                   : 1;
  __REG32 GP4P10                  : 1;
  __REG32 GP4P11                  : 1;
  __REG32 GP4P12                  : 1;
  __REG32 GP4P13                  : 1;
  __REG32 GP4P14                  : 1;
  __REG32 GP4P15                  : 1;
  __REG32 GP5P0                   : 1;
  __REG32 GP5P1                   : 1;
  __REG32 GP5P2                   : 1;
  __REG32 GP5P3                   : 1;
  __REG32 GP5P4                   : 1;
  __REG32 GP5P5                   : 1;
  __REG32 GP5P6                   : 1;
  __REG32 GP5P7                   : 1;
  __REG32 GP5P8                   : 1;
  __REG32 GP5P9                   : 1;
  __REG32 GP5P10                  : 1;
  __REG32 GP5P11                  : 1;
  __REG32 GP5P12                  : 1;
  __REG32 GP5P13                  : 1;
  __REG32 GP5P14                  : 1;
  __REG32 GP5P15                  : 1;
} __gpio_dir45_bits;

/* GPIO Banks 6 and 7 Direction Register (DIR67) */
typedef struct {
  __REG32 GP6P0                   : 1;
  __REG32 GP6P1                   : 1;
  __REG32 GP6P2                   : 1;
  __REG32 GP6P3                   : 1;
  __REG32 GP6P4                   : 1;
  __REG32 GP6P5                   : 1;
  __REG32 GP6P6                   : 1;
  __REG32 GP6P7                   : 1;
  __REG32 GP6P8                   : 1;
  __REG32 GP6P9                   : 1;
  __REG32 GP6P10                  : 1;
  __REG32 GP6P11                  : 1;
  __REG32 GP6P12                  : 1;
  __REG32 GP6P13                  : 1;
  __REG32 GP6P14                  : 1;
  __REG32 GP6P15                  : 1;
  __REG32 GP7P0                   : 1;
  __REG32 GP7P1                   : 1;
  __REG32 GP7P2                   : 1;
  __REG32 GP7P3                   : 1;
  __REG32 GP7P4                   : 1;
  __REG32 GP7P5                   : 1;
  __REG32 GP7P6                   : 1;
  __REG32 GP7P7                   : 1;
  __REG32 GP7P8                   : 1;
  __REG32 GP7P9                   : 1;
  __REG32 GP7P10                  : 1;
  __REG32 GP7P11                  : 1;
  __REG32 GP7P12                  : 1;
  __REG32 GP7P13                  : 1;
  __REG32 GP7P14                  : 1;
  __REG32 GP7P15                  : 1;
} __gpio_dir67_bits;

/* EDMA3CC Configuration Register (CCCFG) */
typedef struct {
  __REG32 NUM_DMACH               : 3;
  __REG32                         : 1;
  __REG32 NUM_QDMACH              : 3;
  __REG32                         : 1;
  __REG32 NUM_INTCH               : 3;
  __REG32                         : 1;
  __REG32 NUM_PAENTRY             : 3;
  __REG32                         : 1;
  __REG32 NUM_EVQUE               : 3;
  __REG32                         : 1;
  __REG32 NUM_REGN                : 2;
  __REG32                         : 2;
  __REG32 CHMAP_EXIST             : 1;
  __REG32 MP_EXIST                : 1;
  __REG32                         : 6;
} __edmacc_cccfg_bits;

/* QDMA Channel n Mapping Register (QCHMAPn) */
typedef struct {
  __REG32                         : 2;
  __REG32 TRWORD                  : 3;
  __REG32 PAENTRY                 : 9;
  __REG32                         :18;
} __edmacc_qchmap_bits;

/* DMA Channel Queue Number Register 0 (DMAQNUM0) */
typedef struct {
  __REG32 E0                      : 3;
  __REG32                         : 1;
  __REG32 E1                      : 3;
  __REG32                         : 1;
  __REG32 E2                      : 3;
  __REG32                         : 1;
  __REG32 E3                      : 3;
  __REG32                         : 1;
  __REG32 E4                      : 3;
  __REG32                         : 1;
  __REG32 E5                      : 3;
  __REG32                         : 1;
  __REG32 E6                      : 3;
  __REG32                         : 1;
  __REG32 E7                      : 3;
  __REG32                         : 1;
} __edmacc_dmaqnum0_bits;

/* DMA Channel Queue Number Register 1 (DMAQNUM1) */
typedef struct {
  __REG32 E8                      : 3;
  __REG32                         : 1;
  __REG32 E9                      : 3;
  __REG32                         : 1;
  __REG32 E10                     : 3;
  __REG32                         : 1;
  __REG32 E11                     : 3;
  __REG32                         : 1;
  __REG32 E12                     : 3;
  __REG32                         : 1;
  __REG32 E13                     : 3;
  __REG32                         : 1;
  __REG32 E14                     : 3;
  __REG32                         : 1;
  __REG32 E15                     : 3;
  __REG32                         : 1;
} __edmacc_dmaqnum1_bits;

/* DMA Channel Queue Number Register 2 (DMAQNUM2) */
typedef struct {
  __REG32 E16                     : 3;
  __REG32                         : 1;
  __REG32 E17                     : 3;
  __REG32                         : 1;
  __REG32 E18                     : 3;
  __REG32                         : 1;
  __REG32 E19                     : 3;
  __REG32                         : 1;
  __REG32 E20                     : 3;
  __REG32                         : 1;
  __REG32 E21                     : 3;
  __REG32                         : 1;
  __REG32 E22                     : 3;
  __REG32                         : 1;
  __REG32 E23                     : 3;
  __REG32                         : 1;
} __edmacc_dmaqnum2_bits;

/* DMA Channel Queue Number Register 3 (DMAQNUM3) */
typedef struct {
  __REG32 E24                     : 3;
  __REG32                         : 1;
  __REG32 E25                     : 3;
  __REG32                         : 1;
  __REG32 E26                     : 3;
  __REG32                         : 1;
  __REG32 E27                     : 3;
  __REG32                         : 1;
  __REG32 E28                     : 3;
  __REG32                         : 1;
  __REG32 E29                     : 3;
  __REG32                         : 1;
  __REG32 E30                     : 3;
  __REG32                         : 1;
  __REG32 E31                     : 3;
  __REG32                         : 1;
} __edmacc_dmaqnum3_bits;

/* QDMA Channel Queue Number Register (QDMAQNUM) */
typedef struct {
  __REG32 E0                      : 3;
  __REG32                         : 1;
  __REG32 E1                      : 3;
  __REG32                         : 1;
  __REG32 E2                      : 3;
  __REG32                         : 1;
  __REG32 E3                      : 3;
  __REG32                         : 1;
  __REG32 E4                      : 3;
  __REG32                         : 1;
  __REG32 E5                      : 3;
  __REG32                         : 1;
  __REG32 E6                      : 3;
  __REG32                         : 1;
  __REG32 E7                      : 3;
  __REG32                         : 1;
} __edmacc_qdmaqnum_bits;

/* Event Missed Registers (EMR) */
/* Event Missed Clear Registers (EMCR) */
/* DMA Region Access Enable for Region m (DRAEm) */
/* Event Register (ER) */
/* Event Clear Register (ECR) */
/* Event Set Register (ESR) */
/* Chained Event Register (CER) */
/* Event Enable Register (EER) */
/* Event Enable Clear Register (EECR) */
/* Event Enable Set Register (EESR) */
/* Secondary Event Register (SER) */
/* Secondary Event Clear Register (SECR) */
/* Secondary Event Clear Register (SECR) */
typedef struct {
  __REG32 E0                      : 1;
  __REG32 E1                      : 1;
  __REG32 E2                      : 1;
  __REG32 E3                      : 1;
  __REG32 E4                      : 1;
  __REG32 E5                      : 1;
  __REG32 E6                      : 1;
  __REG32 E7                      : 1;
  __REG32 E8                      : 1;
  __REG32 E9                      : 1;
  __REG32 E10                     : 1;
  __REG32 E11                     : 1;
  __REG32 E12                     : 1;
  __REG32 E13                     : 1;
  __REG32 E14                     : 1;
  __REG32 E15                     : 1;
  __REG32 E16                     : 1;
  __REG32 E17                     : 1;
  __REG32 E18                     : 1;
  __REG32 E19                     : 1;
  __REG32 E20                     : 1;
  __REG32 E21                     : 1;
  __REG32 E22                     : 1;
  __REG32 E23                     : 1;
  __REG32 E24                     : 1;
  __REG32 E25                     : 1;
  __REG32 E26                     : 1;
  __REG32 E27                     : 1;
  __REG32 E28                     : 1;
  __REG32 E29                     : 1;
  __REG32 E30                     : 1;
  __REG32 E31                     : 1;
} __edmacc_emr_bits;

/* QDMA Event Missed Register (QEMR) */
/* QDMA Event Missed Clear Register (QEMCR) */
/* QDMA Region Access Enable Registers (QRAEm) */
/* QDMA Event Register (QER) */
/* QDMA Event Enable Register (QEER) */
/* QDMA Event Enable Clear Register (QEECR) */
/* QDMA Event Enable Set Register (QEESR) */
/* QDMA Secondary Event Register (QSER) */
/* QDMA Secondary Event Clear Register (QSECR) */
typedef struct {
  __REG32 E0                      : 1;
  __REG32 E1                      : 1;
  __REG32 E2                      : 1;
  __REG32 E3                      : 1;
  __REG32 E4                      : 1;
  __REG32 E5                      : 1;
  __REG32 E6                      : 1;
  __REG32 E7                      : 1;
  __REG32                         :24;
} __edmacc_qemr_bits;

/* EDMA3CC Error Register (CCERR) */
/* EDMA3CC Error Clear Register (CCERRCLR) */
typedef struct {
  __REG32 QTHRXCD0                : 1;
  __REG32 QTHRXCD1                : 1;
  __REG32                         :14;
  __REG32 TCCERR                  : 1;
  __REG32                         :15;
} __edmacc_ccerr_bits;

/* Error Evaluate Register (EEVAL) */
typedef struct {
  __REG32 EVAL                    : 1;
  __REG32                         :31;
} __edmacc_eeval_bits;

/* Event Queue Entry Registers (QxEy) */
typedef struct {
  __REG32 ENUM                    : 5;
  __REG32                         : 1;
  __REG32 ETYPE                   : 2;
  __REG32                         :24;
} __edmacc_qe_bits;

/* Queue n Status Registers (QSTATn) */
typedef struct {
  __REG32 STRTPTR                 : 4;
  __REG32                         : 4;
  __REG32 NUMVAL                  : 5;
  __REG32                         : 3;
  __REG32 WM                      : 5;
  __REG32                         : 3;
  __REG32 THRXCD                  : 1;
  __REG32                         : 7;
} __edmacc_qstat_bits;

/* Queue Watermark Threshold A Register (QWMTHRA) */
typedef struct {
  __REG32 Q0                      : 5;
  __REG32                         : 3;
  __REG32 Q1                      : 5;
  __REG32                         :19;
} __edmacc_qwmthra_bits;

/* EDMA3CC Status Register (CCSTAT) */
typedef struct {
  __REG32 EVTACTV                 : 1;
  __REG32 QEVTACTV                : 1;
  __REG32 TRACTV                  : 1;
  __REG32 WSTATACTV               : 1;
  __REG32 ACTV                    : 1;
  __REG32                         : 3;
  __REG32 COMPACTV                : 6;
  __REG32                         : 2;
  __REG32 QUEACTV0                : 1;
  __REG32 QUEACTV1                : 1;
  __REG32                         :14;
} __edmacc_ccstat_bits;

/* Interrupt Enable Registers (IER) */
/* Interrupt Enable Clear Register (IECR) */
/* Interrupt Enable Set Register (IESR) */
/* Interrupt Pending Register (IPR) */
/* Interrupt Clear Register (ICR) */
typedef struct {
  __REG32 I0                      : 1;
  __REG32 I1                      : 1;
  __REG32 I2                      : 1;
  __REG32 I3                      : 1;
  __REG32 I4                      : 1;
  __REG32 I5                      : 1;
  __REG32 I6                      : 1;
  __REG32 I7                      : 1;
  __REG32 I8                      : 1;
  __REG32 I9                      : 1;
  __REG32 I10                     : 1;
  __REG32 I11                     : 1;
  __REG32 I12                     : 1;
  __REG32 I13                     : 1;
  __REG32 I14                     : 1;
  __REG32 I15                     : 1;
  __REG32 I16                     : 1;
  __REG32 I17                     : 1;
  __REG32 I18                     : 1;
  __REG32 I19                     : 1;
  __REG32 I20                     : 1;
  __REG32 I21                     : 1;
  __REG32 I22                     : 1;
  __REG32 I23                     : 1;
  __REG32 I24                     : 1;
  __REG32 I25                     : 1;
  __REG32 I26                     : 1;
  __REG32 I27                     : 1;
  __REG32 I28                     : 1;
  __REG32 I29                     : 1;
  __REG32 I30                     : 1;
  __REG32 I31                     : 1;
} __edmacc_ier_bits;

/* Interrupt Evaluate Register (IEVAL) */
typedef struct {
  __REG32 EVAL                    : 1;
  __REG32                         :31;
} __edmacc_ieval_bits;

/* EDMA3TC Configuration Register (TCCFG) */
typedef struct {
  __REG32 FIFOSIZE                : 3;
  __REG32                         : 1;
  __REG32 BUSWIDTH                : 2;
  __REG32                         : 2;
  __REG32 DREGDEPTH               : 2;
  __REG32                         :22;
} __edmatc_tccfg_bits;

/* EDMA3TC Channel Status Register (TCSTAT) */
typedef struct {
  __REG32 PROGBUSY                : 1;
  __REG32 SRCACTV                 : 1;
  __REG32 WSACTV                  : 1;
  __REG32                         : 1;
  __REG32 DSTACTV                 : 3;
  __REG32                         : 4;
  __REG32 DFSTRTPTR               : 2;
  __REG32                         :19;
} __edmatc_tcstat_bits;

/* Error Status Register (ERRSTAT) */
/* Error Enable Register (ERREN) */
/* Error Clear Register (ERRCLR) */
typedef struct {
  __REG32 BUSERR                  : 1;
  __REG32                         : 1;
  __REG32 TRERR                   : 1;
  __REG32 MMRAERR                 : 1;
  __REG32                         :28;
} __edmatc_errstat_bits;

/* Error Details Register (ERRDET) */
typedef struct {
  __REG32 STAT                    : 4;
  __REG32                         : 4;
  __REG32 TCC                     : 6;
  __REG32                         : 2;
  __REG32 TCINTEN                 : 1;
  __REG32 TCCHEN                  : 1;
  __REG32                         :14;
} __edmatc_errdet_bits;

/* Error Interrupt Command Register (ERRCMD) */
typedef struct {
  __REG32 EVAL                    : 1;
  __REG32                         :31;
} __edmatc_errcmd_bits;

/* Read Command Rate Register (RDRATE) */
typedef struct {
  __REG32 RDRATE                  : 3;
  __REG32                         :29;
} __edmatc_rdrate_bits;

/* Source Active Options Register (SAOPT) */
/* Destination FIFO Options Register n (DFOPTn) */
typedef struct {
  __REG32 SAM                     : 1;
  __REG32 DAM                     : 1;
  __REG32                         : 2;
  __REG32 PRI                     : 3;
  __REG32                         : 1;
  __REG32 FWID                    : 3;
  __REG32                         : 1;
  __REG32 TCC                     : 6;
  __REG32                         : 2;
  __REG32 TCINTEN                 : 1;
  __REG32                         : 1;
  __REG32 TCCHEN                  : 1;
  __REG32                         : 9;
} __edmatc_saopt_bits;

/* Source Active Count Register (SACNT) */
typedef struct {
  __REG32 ACNT                    :16;
  __REG32 BCNT                    :16;
} __edmatc_sacnt_bits;

/* Source Active B-Index Register (SABIDX) */
typedef struct {
  __REG32 SRCBIDX                 :16;
  __REG32 DSTBIDX                 :16;
} __edmatc_sabidx_bits;

/* Source Active Memory Protection Proxy Register (SAMPPRXY) */
typedef struct {
  __REG32 PRIVID                  : 4;
  __REG32                         : 4;
  __REG32 PRIV                    : 1;
  __REG32                         :23;
} __edmatc_sampprxy_bits;

/* Source Active Count Reload Register (SACNTRLD) */
/* Destination FIFO Set Count Reload Register (DFCNTRLD) */
typedef struct {
  __REG32 ACNTRLD                 :16;
  __REG32                         :16;
} __edmatc_sacntrld_bits;

/* Destination FIFO Options Register n (DFOPTn) */
typedef struct {
  __REG32 SAM                     : 1;
  __REG32 DAM                     : 1;
  __REG32                         : 2;
  __REG32 PRI                     : 3;
  __REG32                         : 1;
  __REG32 FWID                    : 3;
  __REG32                         : 1;
  __REG32 TCC                     : 6;
  __REG32                         : 2;
  __REG32 TCINTEN                 : 1;
  __REG32                         : 1;
  __REG32 TCCHEN                  : 1;
  __REG32                         : 9;
} __edmatc_dfopt_bits;


#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** MPU1
 **
 ***************************************************************************/
__IO_REG32(    MPU1_REVID,            0x01E14000,__READ       );
__IO_REG32_BIT(MPU1_CONFIG,           0x01E14004,__READ       ,__mpu_config_bits);
__IO_REG32_BIT(MPU1_IRAWSTAT,         0x01E14010,__READ_WRITE ,__mpu_irawstat_bits);
__IO_REG32_BIT(MPU1_IENSTAT,          0x01E14014,__READ_WRITE ,__mpu_irawstat_bits);
__IO_REG32_BIT(MPU1_IENSET,           0x01E14018,__READ_WRITE ,__mpu_ienset_bits);
__IO_REG32_BIT(MPU1_IENCLR,           0x01E1401C,__READ_WRITE ,__mpu_ienclr_bits);
__IO_REG32(    MPU1_PROG1_MPSAR,      0x01E14200,__READ_WRITE );
__IO_REG32(    MPU1_PROG1_MPEAR,      0x01E14204,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG1_MPPA,       0x01E14208,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG2_MPSAR,      0x01E14210,__READ_WRITE );
__IO_REG32(    MPU1_PROG2_MPEAR,      0x01E14214,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG2_MPPA,       0x01E14218,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG3_MPSAR,      0x01E14220,__READ_WRITE );
__IO_REG32(    MPU1_PROG3_MPEAR,      0x01E14224,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG3_MPPA,       0x01E14228,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG4_MPSAR,      0x01E14230,__READ_WRITE );
__IO_REG32(    MPU1_PROG4_MPEAR,      0x01E14234,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG4_MPPA,       0x01E14238,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG5_MPSAR,      0x01E14240,__READ_WRITE );
__IO_REG32(    MPU1_PROG5_MPEAR,      0x01E14244,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG5_MPPA,       0x01E14248,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG6_MPSAR,      0x01E14250,__READ_WRITE );
__IO_REG32(    MPU1_PROG6_MPEAR,      0x01E14254,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG6_MPPA,       0x01E14258,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_FLTADDRR,         0x01E14300,__READ       );
__IO_REG32_BIT(MPU1_FLTSTAT,          0x01E14304,__READ       ,__mpu_fltstat_bits);
__IO_REG32_BIT(MPU1_FLTCLR,           0x01E14308,__WRITE      ,__mpu_fltclr_bits);

/***************************************************************************
 **
 ** MPU2
 **
 ***************************************************************************/
__IO_REG32(    MPU2_REVID,            0x01E15000,__READ       );
__IO_REG32_BIT(MPU2_CONFIG,           0x01E15004,__READ       ,__mpu_config_bits);
__IO_REG32_BIT(MPU2_IRAWSTAT,         0x01E15010,__READ_WRITE ,__mpu_irawstat_bits);
__IO_REG32_BIT(MPU2_IENSTAT,          0x01E15014,__READ_WRITE ,__mpu_irawstat_bits);
__IO_REG32_BIT(MPU2_IENSET,           0x01E15018,__READ_WRITE ,__mpu_ienset_bits);
__IO_REG32_BIT(MPU2_IENCLR,           0x01E1501C,__READ_WRITE ,__mpu_ienclr_bits);
__IO_REG32(    MPU2_FXD_MPSAR,        0x01E15100,__READ       );
__IO_REG32(    MPU2_FXD_MPEAR,        0x01E15104,__READ       );
__IO_REG32_BIT(MPU2_FXD_MPPA,         0x01E15108,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG1_MPSAR,      0x01E15200,__READ_WRITE );
__IO_REG32(    MPU2_PROG1_MPEAR,      0x01E15204,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG1_MPPA,       0x01E15208,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG2_MPSAR,      0x01E15210,__READ_WRITE );
__IO_REG32(    MPU2_PROG2_MPEAR,      0x01E15214,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG2_MPPA,       0x01E15218,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG3_MPSAR,      0x01E15220,__READ_WRITE );
__IO_REG32(    MPU2_PROG3_MPEAR,      0x01E15224,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG3_MPPA,       0x01E15228,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG4_MPSAR,      0x01E15230,__READ_WRITE );
__IO_REG32(    MPU2_PROG4_MPEAR,      0x01E15234,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG4_MPPA,       0x01E15238,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG5_MPSAR,      0x01E15240,__READ_WRITE );
__IO_REG32(    MPU2_PROG5_MPEAR,      0x01E15244,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG5_MPPA,       0x01E15248,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG6_MPSAR,      0x01E15250,__READ_WRITE );
__IO_REG32(    MPU2_PROG6_MPEAR,      0x01E15254,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG6_MPPA,       0x01E15258,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG7_MPSAR,      0x01E15260,__READ_WRITE );
__IO_REG32(    MPU2_PROG7_MPEAR,      0x01E15264,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG7_MPPA,       0x01E15268,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG8_MPSAR,      0x01E15270,__READ_WRITE );
__IO_REG32(    MPU2_PROG8_MPEAR,      0x01E15274,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG8_MPPA,       0x01E15278,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG9_MPSAR,      0x01E15280,__READ_WRITE );
__IO_REG32(    MPU2_PROG9_MPEAR,      0x01E15284,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG9_MPPA,       0x01E15288,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG10_MPSAR,     0x01E15290,__READ_WRITE );
__IO_REG32(    MPU2_PROG10_MPEAR,     0x01E15294,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG10_MPPA,      0x01E15298,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG11_MPSAR,     0x01E152A0,__READ_WRITE );
__IO_REG32(    MPU2_PROG11_MPEAR,     0x01E152A4,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG11_MPPA,      0x01E152A8,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG12_MPSAR,     0x01E152B0,__READ_WRITE );
__IO_REG32(    MPU2_PROG12_MPEAR,     0x01E152B4,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG12_MPPA,      0x01E152B8,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_FLTADDRR,         0x01E15300,__READ       );
__IO_REG32_BIT(MPU2_FLTSTAT,          0x01E15304,__READ       ,__mpu_fltstat_bits);
__IO_REG32_BIT(MPU2_FLTCLR,           0x01E15308,__WRITE      ,__mpu_fltclr_bits);

/***************************************************************************
 **
 ** PLLC0
 **
 ***************************************************************************/
__IO_REG32(    PLLC0_REVID,           0x01C11000,__READ       );
__IO_REG32_BIT(PLLC0_RSTYPE,          0x01C110E4,__READ       ,__pllc0_rstype_bits);
__IO_REG32_BIT(PLLC0_PLLCTL,          0x01C11100,__READ_WRITE ,__pllc0_pllctl_bits);
__IO_REG32_BIT(PLLC0_PLLM,            0x01C11110,__READ_WRITE ,__pllc_pllm_bits);
__IO_REG32_BIT(PLLC0_PREDIV,          0x01C11114,__READ_WRITE ,__pllc0_prediv_bits);
__IO_REG32_BIT(PLLC0_PLLDIV1,         0x01C11118,__READ_WRITE ,__pllc_plldiv1_bits);
__IO_REG32_BIT(PLLC0_PLLDIV2,         0x01C1111C,__READ_WRITE ,__pllc_plldiv2_bits);
__IO_REG32_BIT(PLLC0_PLLDIV3,         0x01C11120,__READ_WRITE ,__pllc_plldiv3_bits);
__IO_REG32_BIT(PLLC0_POSTDIV,         0x01C11128,__READ_WRITE ,__pllc_postdiv_bits);
__IO_REG32_BIT(PLLC0_PLLCMD,          0x01C11138,__READ_WRITE ,__pllc_pllcmd_bits);
__IO_REG32_BIT(PLLC0_PLLSTAT,         0x01C1113C,__READ       ,__pllc_pllstat_bits);
__IO_REG32_BIT(PLLC0_ALNCTL,          0x01C11140,__READ_WRITE ,__pllc0_alnctl_bits);
__IO_REG32_BIT(PLLC0_DCHANGE,         0x01C11144,__READ       ,__pllc0_dchange_bits);
__IO_REG32_BIT(PLLC0_CKEN,            0x01C11148,__READ_WRITE ,__pllc0_cken_bits);
__IO_REG32_BIT(PLLC0_CKSTAT,          0x01C1114C,__READ       ,__pllc0_cken_bits);
__IO_REG32_BIT(PLLC0_SYSTAT,          0x01C11150,__READ       ,__pllc0_systat_bits);
__IO_REG32_BIT(PLLC0_PLLDIV4,         0x01C11160,__READ_WRITE ,__pllc0_plldiv4_bits);
__IO_REG32_BIT(PLLC0_PLLDIV5,         0x01C11164,__READ_WRITE ,__pllc0_plldiv5_bits);
__IO_REG32_BIT(PLLC0_PLLDIV6,         0x01C11168,__READ_WRITE ,__pllc0_plldiv6_bits);
__IO_REG32_BIT(PLLC0_PLLDIV7,         0x01C1116C,__READ_WRITE ,__pllc0_plldiv7_bits);
__IO_REG32(    PLLC0_EMUCNT0,         0x01C111F0,__READ       );
__IO_REG32(    PLLC0_EMUCNT1,         0x01C111F4,__READ       );

/***************************************************************************
 **
 ** PSC0
 **
 ***************************************************************************/
__IO_REG32(    PSC0_REVID,            0x01C10000,__READ       );
__IO_REG32_BIT(PSC0_INTEVAL,          0x01C10018,__WRITE      ,__psc_inteval_bits);
__IO_REG32_BIT(PSC0_MERRPR0,          0x01C10040,__READ       ,__psc0_merrpr0_bits);
__IO_REG32_BIT(PSC0_MERRCR0,          0x01C10050,__WRITE      ,__psc0_merrpr0_bits);
__IO_REG32_BIT(PSC0_PERRPR,           0x01C10060,__READ       ,__psc_perrpr_bits);
__IO_REG32_BIT(PSC0_PERRCR,           0x01C10068,__WRITE      ,__psc_perrpr_bits);
__IO_REG32_BIT(PSC0_PTCMD,            0x01C10120,__WRITE      ,__psc_ptcmd_bits);
__IO_REG32_BIT(PSC0_PTSTAT,           0x01C10128,__READ       ,__psc_ptstat_bits);
__IO_REG32_BIT(PSC0_PDSTAT0,          0x01C10200,__READ       ,__psc_pdstat_bits);
__IO_REG32_BIT(PSC0_PDSTAT1,          0x01C10204,__READ       ,__psc_pdstat_bits);
__IO_REG32_BIT(PSC0_PDCTL0,           0x01C10300,__READ_WRITE ,__psc_pdctl_bits);
__IO_REG32_BIT(PSC0_PDCTL1,           0x01C10304,__READ_WRITE ,__psc_pdctl_bits);
__IO_REG32_BIT(PSC0_PDCFG0,           0x01C10400,__READ       ,__psc_pdcfg_bits);
__IO_REG32_BIT(PSC0_PDCFG1,           0x01C10404,__READ       ,__psc_pdcfg_bits);
__IO_REG32_BIT(PSC0_MDSTAT0,          0x01C10800,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT1,          0x01C10804,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT2,          0x01C10808,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT3,          0x01C1080C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT4,          0x01C10810,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT5,          0x01C10814,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT6,          0x01C10818,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT7,          0x01C1081C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT8,          0x01C10820,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT9,          0x01C10824,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT10,         0x01C10828,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT11,         0x01C1082C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT12,         0x01C10830,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT13,         0x01C10834,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT14,         0x01C10838,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDSTAT15,         0x01C1083C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC0_MDCTL0,           0x01C10A00,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL1,           0x01C10A04,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL2,           0x01C10A08,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL3,           0x01C10A0C,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL4,           0x01C10A10,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL5,           0x01C10A14,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL6,           0x01C10A18,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL7,           0x01C10A1C,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL8,           0x01C10A20,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL9,           0x01C10A24,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL10,          0x01C10A28,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL11,          0x01C10A2C,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL12,          0x01C10A30,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL13,          0x01C10A34,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL14,          0x01C10A38,__READ_WRITE ,__psc0_mdctl_bits);
__IO_REG32_BIT(PSC0_MDCTL15,          0x01C10A3C,__READ_WRITE ,__psc0_mdctl_bits);

/***************************************************************************
 **
 ** PSC1
 **
 ***************************************************************************/
__IO_REG32(    PSC1_REVID,            0x01E27000,__READ       );
__IO_REG32_BIT(PSC1_INTEVAL,          0x01E27018,__WRITE      ,__psc_inteval_bits);
__IO_REG32_BIT(PSC1_PERRPR,           0x01E27060,__READ       ,__psc_perrpr_bits);
__IO_REG32_BIT(PSC1_PERRCR,           0x01E27068,__WRITE      ,__psc_perrpr_bits);
__IO_REG32_BIT(PSC1_PTCMD,            0x01E27120,__WRITE      ,__psc_ptcmd_bits);
__IO_REG32_BIT(PSC1_PTSTAT,           0x01E27128,__READ       ,__psc_ptstat_bits);
__IO_REG32_BIT(PSC1_PDSTAT0,          0x01E27200,__READ       ,__psc_pdstat_bits);
__IO_REG32_BIT(PSC1_PDSTAT1,          0x01E27204,__READ       ,__psc_pdstat_bits);
__IO_REG32_BIT(PSC1_PDCTL0,           0x01E27300,__READ_WRITE ,__psc_pdctl_bits);
__IO_REG32_BIT(PSC1_PDCTL1,           0x01E27304,__READ_WRITE ,__psc_pdctl_bits);
__IO_REG32_BIT(PSC1_PDCFG0,           0x01E27400,__READ       ,__psc_pdcfg_bits);
__IO_REG32_BIT(PSC1_PDCFG1,           0x01E27404,__READ       ,__psc_pdcfg_bits);
__IO_REG32_BIT(PSC1_MDSTAT0,          0x01E27800,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT1,          0x01E27804,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT2,          0x01E27808,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT3,          0x01E2780C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT4,          0x01E27810,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT5,          0x01E27814,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT6,          0x01E27818,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT7,          0x01E2781C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT8,          0x01E27820,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT9,          0x01E27824,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT10,         0x01E27828,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT11,         0x01E2782C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT12,         0x01E27830,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT13,         0x01E27834,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT14,         0x01E27838,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT15,         0x01E2783C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT16,         0x01E27840,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT17,         0x01E27844,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT18,         0x01E27848,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT19,         0x01E2784C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT20,         0x01E27850,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT21,         0x01E27854,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT22,         0x01E27858,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT23,         0x01E2785C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT24,         0x01E27860,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT25,         0x01E27864,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT26,         0x01E27868,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT27,         0x01E2786C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT28,         0x01E27870,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT29,         0x01E27874,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT30,         0x01E27878,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDSTAT31,         0x01E2787C,__READ       ,__psc_mdstat_bits);
__IO_REG32_BIT(PSC1_MDCTL0,           0x01E27A00,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL1,           0x01E27A04,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL2,           0x01E27A08,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL3,           0x01E27A0C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL4,           0x01E27A10,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL5,           0x01E27A14,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL6,           0x01E27A18,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL7,           0x01E27A1C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL8,           0x01E27A20,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL9,           0x01E27A24,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL10,          0x01E27A28,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL11,          0x01E27A2C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL12,          0x01E27A30,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL13,          0x01E27A34,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL14,          0x01E27A38,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL15,          0x01E27A3C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL16,          0x01E27A40,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL17,          0x01E27A44,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL18,          0x01E27A48,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL19,          0x01E27A4C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL20,          0x01E27A50,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL21,          0x01E27A54,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL22,          0x01E27A58,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL23,          0x01E27A5C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL24,          0x01E27A60,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL25,          0x01E27A64,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL26,          0x01E27A68,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL27,          0x01E27A6C,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL28,          0x01E27A70,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL29,          0x01E27A74,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL30,          0x01E27A78,__READ_WRITE ,__psc1_mdctl_bits);
__IO_REG32_BIT(PSC1_MDCTL31,          0x01E27A7C,__READ_WRITE ,__psc1_mdctl_bits);

/***************************************************************************
 **
 ** SYSCFG0
 **
 ***************************************************************************/
__IO_REG32(    SYSCFG0_REVID,         0x01C14000,__READ       );
__IO_REG32(    SYSCFG0_DIEIDR0,       0x01C14008,__READ       );
__IO_REG32(    SYSCFG0_DIEIDR1,       0x01C1400C,__READ       );
__IO_REG32(    SYSCFG0_DIEIDR2,       0x01C14010,__READ       );
__IO_REG32(    SYSCFG0_DIEIDR3,       0x01C14014,__READ       );
__IO_REG32_BIT(SYSCFG0_BOOTCFG,       0x01C14020,__READ       ,__syscfg_bootcfg_bits);
__IO_REG32(    SYSCFG0_KICK0R,        0x01C14038,__READ_WRITE );
__IO_REG32(    SYSCFG0_KICK1R,        0x01C1403C,__READ_WRITE );
__IO_REG32_BIT(SYSCFG0_HOST0CFG,      0x01C14040,__READ_WRITE ,__syscfg_host0cfg_bits);
__IO_REG32(    SYSCFG0_HOST1CFG,      0x01C14044,__READ_WRITE );
__IO_REG32_BIT(SYSCFG0_IRAWSTAT,      0x01C140E0,__READ_WRITE ,__syscfg_irawstat_bits);
__IO_REG32_BIT(SYSCFG0_IENSTAT,       0x01C140E4,__READ_WRITE ,__syscfg_irawstat_bits);
__IO_REG32_BIT(SYSCFG0_IENSET,        0x01C140E8,__READ_WRITE ,__syscfg_ienset_bits);
__IO_REG32_BIT(SYSCFG0_IENCLR,        0x01C140EC,__READ_WRITE ,__syscfg_ienclr_bits);
__IO_REG32_BIT(SYSCFG0_EOI,           0x01C140F0,__WRITE      ,__syscfg_eoi_bits);
__IO_REG32(    SYSCFG0_FLTADDRR,      0x01C140F4,__READ       );
__IO_REG32_BIT(SYSCFG0_FLTSTAT,       0x01C140F8,__READ       ,__syscfg_fltstat_bits);
__IO_REG32_BIT(SYSCFG0_MSTPRI0,       0x01C14110,__READ_WRITE ,__syscfg_mstpri0_bits);
__IO_REG32_BIT(SYSCFG0_MSTPRI1,       0x01C14114,__READ_WRITE ,__syscfg_mstpri1_bits);
__IO_REG32_BIT(SYSCFG0_MSTPRI2,       0x01C14118,__READ_WRITE ,__syscfg_mstpri2_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX0,       0x01C14120,__READ_WRITE ,__syscfg_pinmux0_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX1,       0x01C14124,__READ_WRITE ,__syscfg_pinmux1_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX2,       0x01C14128,__READ_WRITE ,__syscfg_pinmux2_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX3,       0x01C1412C,__READ_WRITE ,__syscfg_pinmux3_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX4,       0x01C14130,__READ_WRITE ,__syscfg_pinmux4_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX5,       0x01C14134,__READ_WRITE ,__syscfg_pinmux5_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX6,       0x01C14138,__READ_WRITE ,__syscfg_pinmux6_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX7,       0x01C1413C,__READ_WRITE ,__syscfg_pinmux7_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX8,       0x01C14140,__READ_WRITE ,__syscfg_pinmux8_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX9,       0x01C14144,__READ_WRITE ,__syscfg_pinmux9_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX10,      0x01C14148,__READ_WRITE ,__syscfg_pinmux10_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX11,      0x01C1414C,__READ_WRITE ,__syscfg_pinmux11_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX12,      0x01C14150,__READ_WRITE ,__syscfg_pinmux12_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX13,      0x01C14154,__READ_WRITE ,__syscfg_pinmux13_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX14,      0x01C14158,__READ_WRITE ,__syscfg_pinmux14_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX15,      0x01C1415C,__READ_WRITE ,__syscfg_pinmux15_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX16,      0x01C14160,__READ_WRITE ,__syscfg_pinmux16_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX17,      0x01C14164,__READ_WRITE ,__syscfg_pinmux17_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX18,      0x01C14168,__READ_WRITE ,__syscfg_pinmux18_bits);
__IO_REG32_BIT(SYSCFG0_PINMUX19,      0x01C1416C,__READ_WRITE ,__syscfg_pinmux19_bits);
__IO_REG32_BIT(SYSCFG0_SUSPSRC,       0x01C14170,__READ_WRITE ,__syscfg_suspsrc_bits);
__IO_REG32_BIT(SYSCFG0_CHIPSIG,       0x01C14174,__READ_WRITE ,__syscfg_chipsig_bits);
__IO_REG32_BIT(SYSCFG0_CHIPSIG_CLR,   0x01C14178,__READ_WRITE ,__syscfg_chipsig_bits);
__IO_REG32_BIT(SYSCFG0_CFGCHIP0,      0x01C1417C,__READ_WRITE ,__syscfg_cfgchip0_bits);
__IO_REG32_BIT(SYSCFG0_CFGCHIP1,      0x01C14180,__READ_WRITE ,__syscfg_cfgchip1_bits);
__IO_REG32_BIT(SYSCFG0_CFGCHIP2,      0x01C14184,__READ_WRITE ,__syscfg_cfgchip2_bits);
__IO_REG32_BIT(SYSCFG0_CFGCHIP3,      0x01C14188,__READ_WRITE ,__syscfg_cfgchip3_bits);
__IO_REG32_BIT(SYSCFG0_CFGCHIP4,      0x01C1418C,__READ_WRITE ,__syscfg_cfgchip4_bits);

/***************************************************************************
 **
 ** AINTC
 **
 ***************************************************************************/
__IO_REG32(    AINTC_REV,             0xFFFEE000,__READ_WRITE );
__IO_REG32_BIT(AINTC_CR,              0xFFFEE004,__READ_WRITE ,__aintc_cr_bits);
__IO_REG32_BIT(AINTC_GER,             0xFFFEE010,__READ_WRITE ,__aintc_ger_bits);
__IO_REG32_BIT(AINTC_GNLR,            0xFFFEE01C,__READ_WRITE ,__aintc_gnlr_bits);
__IO_REG32_BIT(AINTC_SISR,            0xFFFEE020,__WRITE      ,__aintc_sisr_bits);
__IO_REG32_BIT(AINTC_SICR,            0xFFFEE024,__WRITE      ,__aintc_sisr_bits);
__IO_REG32_BIT(AINTC_EISR,            0xFFFEE028,__WRITE      ,__aintc_sisr_bits);
__IO_REG32_BIT(AINTC_EICR,            0xFFFEE02C,__WRITE      ,__aintc_sisr_bits);
__IO_REG32_BIT(AINTC_HIEISR,          0xFFFEE034,__WRITE      ,__aintc_hieisr_bits);
__IO_REG32_BIT(AINTC_HIDISR,          0xFFFEE038,__READ_WRITE ,__aintc_hieisr_bits);
__IO_REG32(    AINTC_VBR,             0xFFFEE050,__READ_WRITE );
__IO_REG32_BIT(AINTC_VSR,             0xFFFEE054,__READ_WRITE ,__aintc_vsr_bits);
__IO_REG32(    AINTC_VNR,             0xFFFEE058,__READ_WRITE );
__IO_REG32_BIT(AINTC_GPIR,            0xFFFEE080,__READ       ,__aintc_gpir_bits);
__IO_REG32(    AINTC_GPVR,            0xFFFEE084,__READ_WRITE );
__IO_REG32_BIT(AINTC_SRSR0,           0xFFFEE200,__WRITE      ,__aintc_srsr0_bits);
__IO_REG32_BIT(AINTC_SRSR1,           0xFFFEE204,__WRITE      ,__aintc_srsr1_bits);
__IO_REG32_BIT(AINTC_SRSR2,           0xFFFEE208,__WRITE      ,__aintc_srsr2_bits);
__IO_REG32_BIT(AINTC_SECR0,           0xFFFEE280,__WRITE      ,__aintc_secr0_bits);
__IO_REG32_BIT(AINTC_SECR1,           0xFFFEE284,__WRITE      ,__aintc_secr1_bits);
__IO_REG32_BIT(AINTC_SECR2,           0xFFFEE288,__WRITE      ,__aintc_secr2_bits);
__IO_REG32_BIT(AINTC_ESR0,            0xFFFEE300,__WRITE      ,__aintc_esr0_bits);
__IO_REG32_BIT(AINTC_ESR1,            0xFFFEE304,__WRITE      ,__aintc_esr1_bits);
__IO_REG32_BIT(AINTC_ESR2,            0xFFFEE308,__WRITE      ,__aintc_esr2_bits);
__IO_REG32_BIT(AINTC_ECR0,            0xFFFEE380,__WRITE      ,__aintc_ecr0_bits);
__IO_REG32_BIT(AINTC_ECR1,            0xFFFEE384,__WRITE      ,__aintc_ecr1_bits);
__IO_REG32_BIT(AINTC_ECR2,            0xFFFEE388,__WRITE      ,__aintc_ecr2_bits);
__IO_REG32_BIT(AINTC_CMR0,            0xFFFEE400,__READ_WRITE ,__aintc_cmr0_bits);
__IO_REG32_BIT(AINTC_CMR1,            0xFFFEE404,__READ_WRITE ,__aintc_cmr1_bits);
__IO_REG32_BIT(AINTC_CMR2,            0xFFFEE408,__READ_WRITE ,__aintc_cmr2_bits);
__IO_REG32_BIT(AINTC_CMR3,            0xFFFEE40C,__READ_WRITE ,__aintc_cmr3_bits);
__IO_REG32_BIT(AINTC_CMR4,            0xFFFEE410,__READ_WRITE ,__aintc_cmr4_bits);
__IO_REG32_BIT(AINTC_CMR5,            0xFFFEE414,__READ_WRITE ,__aintc_cmr5_bits);
__IO_REG32_BIT(AINTC_CMR6,            0xFFFEE418,__READ_WRITE ,__aintc_cmr6_bits);
__IO_REG32_BIT(AINTC_CMR7,            0xFFFEE41C,__READ_WRITE ,__aintc_cmr7_bits);
__IO_REG32_BIT(AINTC_CMR8,            0xFFFEE420,__READ_WRITE ,__aintc_cmr8_bits);
__IO_REG32_BIT(AINTC_CMR9,            0xFFFEE424,__READ_WRITE ,__aintc_cmr9_bits);
__IO_REG32_BIT(AINTC_CMR10,           0xFFFEE428,__READ_WRITE ,__aintc_cmr10_bits);
__IO_REG32_BIT(AINTC_CMR11,           0xFFFEE42C,__READ_WRITE ,__aintc_cmr11_bits);
__IO_REG32_BIT(AINTC_CMR12,           0xFFFEE430,__READ_WRITE ,__aintc_cmr12_bits);
__IO_REG32_BIT(AINTC_CMR13,           0xFFFEE434,__READ_WRITE ,__aintc_cmr13_bits);
__IO_REG32_BIT(AINTC_CMR14,           0xFFFEE438,__READ_WRITE ,__aintc_cmr14_bits);
__IO_REG32_BIT(AINTC_CMR15,           0xFFFEE43C,__READ_WRITE ,__aintc_cmr15_bits);
__IO_REG32_BIT(AINTC_CMR16,           0xFFFEE440,__READ_WRITE ,__aintc_cmr16_bits);
__IO_REG32_BIT(AINTC_CMR17,           0xFFFEE444,__READ_WRITE ,__aintc_cmr17_bits);
__IO_REG32_BIT(AINTC_CMR18,           0xFFFEE448,__READ_WRITE ,__aintc_cmr18_bits);
__IO_REG32_BIT(AINTC_CMR19,           0xFFFEE44C,__READ_WRITE ,__aintc_cmr19_bits);
__IO_REG32_BIT(AINTC_CMR20,           0xFFFEE450,__READ_WRITE ,__aintc_cmr20_bits);
__IO_REG32_BIT(AINTC_CMR21,           0xFFFEE454,__READ_WRITE ,__aintc_cmr21_bits);
__IO_REG32_BIT(AINTC_CMR22,           0xFFFEE458,__READ_WRITE ,__aintc_cmr22_bits);
__IO_REG32_BIT(AINTC_HIPIR0,          0xFFFEE900,__READ       ,__aintc_hipir_bits);
__IO_REG32_BIT(AINTC_HIPIR1,          0xFFFEE904,__READ       ,__aintc_hipir_bits);
__IO_REG32_BIT(AINTC_HINLR0,          0xFFFEF100,__READ_WRITE ,__aintc_hinlr_bits);
__IO_REG32_BIT(AINTC_HINLR1,          0xFFFEF104,__READ_WRITE ,__aintc_hinlr_bits);
__IO_REG32_BIT(AINTC_HIER ,           0xFFFEF500,__READ_WRITE ,__aintc_hier_bits);
__IO_REG32(    AINTC_HIPVR0,          0xFFFEF600,__READ       );
__IO_REG32(    AINTC_HIPVR1,          0xFFFEF604,__READ       );

/***************************************************************************
 **
 ** EMIFA
 **
 ***************************************************************************/
__IO_REG32(    EMIFA_MIDR,            0x68000000,__READ       );
__IO_REG32_BIT(EMIFA_AWCC,            0x68000004,__READ_WRITE ,__awcc_bits);
__IO_REG32_BIT(EMIFA_SDCR,            0x68000008,__READ_WRITE ,__sdcr_bits);
__IO_REG32_BIT(EMIFA_SDRCR,           0x6800000C,__READ_WRITE ,__sdrcr_bits);
__IO_REG32_BIT(EMIFA_CE2CFG,          0x68000010,__READ_WRITE ,__cecfg_bits);
__IO_REG32_BIT(EMIFA_CE3CFG,          0x68000014,__READ_WRITE ,__cecfg_bits);
__IO_REG32_BIT(EMIFA_CE4CFG,          0x68000018,__READ_WRITE ,__cecfg_bits);
__IO_REG32_BIT(EMIFA_CE5CFG,          0x6800001C,__READ_WRITE ,__cecfg_bits);
__IO_REG32_BIT(EMIFA_SDTIMR,          0x68000020,__READ_WRITE ,__sdtimr_bits);
__IO_REG32_BIT(EMIFA_SDSRETR,         0x6800003C,__READ_WRITE ,__sdsretr_bits);
__IO_REG32_BIT(EMIFA_INTRAW,          0x68000040,__READ_WRITE ,__intraw_bits);
__IO_REG32_BIT(EMIFA_INTMSK,          0x68000044,__READ_WRITE ,__intmsk_bits);
__IO_REG32_BIT(EMIFA_INTMSKSET,       0x68000048,__READ_WRITE ,__intmskset_bits);
__IO_REG32_BIT(EMIFA_INTMSKCLR,       0x6800004C,__READ_WRITE ,__intmskclr_bits);
__IO_REG32_BIT(EMIFA_NANDFCR,         0x68000060,__READ_WRITE ,__nandfcr_bits);
__IO_REG32_BIT(EMIFA_NANDFSR,         0x68000064,__READ       ,__nandfsr_bits);
__IO_REG32_BIT(EMIFA_PMCR,            0x68000068,__READ_WRITE ,__pmcr_bits);
__IO_REG32_BIT(EMIFA_NANDF1ECC,       0x68000070,__READ_WRITE ,__nandfecc_bits);
__IO_REG32_BIT(EMIFA_NANDF2ECC,       0x68000074,__READ_WRITE ,__nandfecc_bits);
__IO_REG32_BIT(EMIFA_NANDF3ECC,       0x68000078,__READ_WRITE ,__nandfecc_bits);
__IO_REG32_BIT(EMIFA_NANDF4ECC,       0x6800007C,__READ_WRITE ,__nandfecc_bits);
__IO_REG32_BIT(EMIFA_NAND4BITECCLOAD, 0x680000BC,__READ_WRITE ,__nand4biteccload_bits);
__IO_REG32_BIT(EMIFA_NAND4BITECC1,    0x680000C0,__READ_WRITE ,__nand4bitecc1_bits);
__IO_REG32_BIT(EMIFA_NAND4BITECC2,    0x680000C4,__READ_WRITE ,__nand4bitecc2_bits);
__IO_REG32_BIT(EMIFA_NAND4BITECC3,    0x680000C8,__READ_WRITE ,__nand4bitecc3_bits);
__IO_REG32_BIT(EMIFA_NAND4BITECC4,    0x680000CC,__READ_WRITE ,__nand4bitecc4_bits);
__IO_REG32_BIT(EMIFA_NANDERRADD1,     0x680000D0,__READ_WRITE ,__nanderradd1_bits);
__IO_REG32_BIT(EMIFA_NANDERRADD2,     0x680000D4,__READ_WRITE ,__nanderradd2_bits);
__IO_REG32_BIT(EMIFA_NANDERRVAL1,     0x680000D8,__READ_WRITE ,__nanderrval1_bits);
__IO_REG32_BIT(EMIFA_NANDERRVAL2,     0x680000DC,__READ_WRITE ,__nanderrval2_bits);

/***************************************************************************
 **
 ** EMIFB
 **
 ***************************************************************************/
__IO_REG32(    EMIFB_MIDR,            0xB0000000,__READ       );
__IO_REG32_BIT(EMIFB_SDCFG,           0xB0000008,__READ_WRITE ,__emifb_sdcfg_bits);
__IO_REG32_BIT(EMIFB_SDRFC,           0xB000000C,__READ_WRITE ,__emifb_sdrfc_bits);
__IO_REG32_BIT(EMIFB_SDTIM1,          0xB0000010,__READ_WRITE ,__emifb_sdtim1_bits);
__IO_REG32_BIT(EMIFB_SDTIM2,          0xB0000014,__READ_WRITE ,__emifb_sdtim2_bits);
__IO_REG32_BIT(EMIFB_SDCFG2,          0xB000001C,__READ_WRITE ,__emifb_sdcfg2_bits);
__IO_REG32_BIT(EMIFB_BPRIO,           0xB0000020,__READ_WRITE ,__emifb_bprio_bits);
__IO_REG32(    EMIFB_PC1,             0xB0000040,__READ       );
__IO_REG32(    EMIFB_PC2,             0xB0000044,__READ       );
__IO_REG32_BIT(EMIFB_PCC,             0xB0000048,__READ_WRITE ,__emifb_pcc_bits);
__IO_REG32_BIT(EMIFB_PCMRS,           0xB000004C,__READ_WRITE ,__emifb_pcmrs_bits);
__IO_REG32(    EMIFB_PCT,             0xB0000050,__READ       );
__IO_REG32_BIT(EMIFB_IRR,             0xB00000C0,__READ_WRITE ,__emifb_irr_bits);
__IO_REG32_BIT(EMIFB_IMR,             0xB00000C4,__READ_WRITE ,__emifb_irr_bits);
__IO_REG32_BIT(EMIFB_IMSR,            0xB00000C8,__READ_WRITE ,__emifb_imsr_bits);
__IO_REG32_BIT(EMIFB_IMCR,            0xB00000CC,__READ_WRITE ,__emifb_imcr_bits);

/***************************************************************************
 **
 ** MMCSD0
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC0CTL,               0x01C40000,__READ_WRITE ,__mmcctl_bits);
__IO_REG32_BIT(MMC0CLK,               0x01C40004,__READ_WRITE ,__mmcclk_bits);
__IO_REG32_BIT(MMC0ST0,               0x01C40008,__READ       ,__mmcst0_bits);
__IO_REG32_BIT(MMC0ST1,               0x01C4000C,__READ       ,__mmcst1_bits);
__IO_REG32_BIT(MMC0IM,                0x01C40010,__READ_WRITE ,__mmcim_bits);
__IO_REG32_BIT(MMC0TOR,               0x01C40014,__READ_WRITE ,__mmctor_bits);
__IO_REG32_BIT(MMC0TOD,               0x01C40018,__READ_WRITE ,__mmctod_bits);
__IO_REG32_BIT(MMC0BLEN,              0x01C4001C,__READ_WRITE ,__mmcblen_bits);
__IO_REG32_BIT(MMC0NBLK,              0x01C40020,__READ_WRITE ,__mmcnblk_bits);
__IO_REG32_BIT(MMC0NBLC,              0x01C40024,__READ       ,__mmcnblc_bits);
__IO_REG32(    MMC0DRR,               0x01C40028,__READ_WRITE );
__IO_REG32(    MMC0DXR,               0x01C4002C,__READ_WRITE );
__IO_REG32_BIT(MMC0CMD,               0x01C40030,__READ_WRITE ,__mmccmd_bits);
__IO_REG32_BIT(MMC0ARGHL,             0x01C40034,__READ_WRITE ,__mmcarghl_bits);
__IO_REG32_BIT(MMC0RSP01,             0x01C40038,__READ_WRITE ,__mmcrsp01_bits);
__IO_REG32_BIT(MMC0RSP23,             0x01C4003C,__READ_WRITE ,__mmcrsp23_bits);
__IO_REG32_BIT(MMC0RSP45,             0x01C40040,__READ_WRITE ,__mmcrsp45_bits);
__IO_REG32_BIT(MMC0RSP67,             0x01C40044,__READ_WRITE ,__mmcrsp67_bits);
__IO_REG32_BIT(MMC0DRSP,              0x01C40048,__READ_WRITE ,__mmcdrsp_bits);
__IO_REG32_BIT(MMC0CIDX,              0x01C40050,__READ_WRITE ,__mmccidx_bits);
__IO_REG32_BIT(SDIO0CTL,              0x01C40064,__READ_WRITE ,__sdioctl_bits);
__IO_REG32_BIT(SDIO0ST0,              0x01C40068,__READ       ,__sdiost0_bits);
__IO_REG32_BIT(SDI0OIEN,              0x01C4006C,__READ_WRITE ,__sdioien_bits);
__IO_REG32_BIT(SDI0OIST,              0x01C40070,__READ_WRITE ,__sdioist_bits);
__IO_REG32_BIT(MMC0FIFOCTL,           0x01C40074,__READ_WRITE ,__mmcfifoctl_bits);

/***************************************************************************
 **
 ** McASP0
 **
 ***************************************************************************/
__IO_REG32(    McASP0_REV,            0x01D00000,__READ       );
__IO_REG32_BIT(McASP0_PFUNC,          0x01D00010,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP0_PDIR,           0x01D00014,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP0_PDOUT,          0x01D00018,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP0_PDIN,           0x01D0001C,__READ_WRITE ,__mcasp_pfunc_bits);
#define McASP0_PDSET     McASP0_PDIN
#define McASP0_PDSET_bit McASP0_PDIN_bit
__IO_REG32_BIT(McASP0_PDCLR,          0x01D00020,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP0_GBLCTL,         0x01D00044,__READ_WRITE ,__mcasp_gblctl_bits);
__IO_REG32_BIT(McASP0_AMUTE,          0x01D00048,__READ_WRITE ,__mcasp_amute_bits);
__IO_REG32_BIT(McASP0_DLBCTL,         0x01D0004C,__READ_WRITE ,__mcasp_dlbctl_bits);
__IO_REG32_BIT(McASP0_DITCTL,         0x01D00050,__READ_WRITE ,__mcasp_ditctl_bits);
__IO_REG32_BIT(McASP0_RGBLCTL,        0x01D00060,__READ_WRITE ,__mcasp_rgblctl_bits);
__IO_REG32_BIT(McASP0_RMASK,          0x01D00064,__READ_WRITE ,__mcasp_rmask_bits);
__IO_REG32_BIT(McASP0_RFMT,           0x01D00068,__READ_WRITE ,__mcasp_rfmt_bits);
__IO_REG32_BIT(McASP0_AFSRCTL,        0x01D0006C,__READ_WRITE ,__mcasp_afsrctl_bits);
__IO_REG32_BIT(McASP0_ACLKRCTL,       0x01D00070,__READ_WRITE ,__mcasp_aclkrctl_bits);
__IO_REG32_BIT(McASP0_AHCLKRCTL,      0x01D00074,__READ_WRITE ,__mcasp_ahclkrctl_bits);
__IO_REG32_BIT(McASP0_RTDM,           0x01D00078,__READ_WRITE ,__mcasp_rtdm_bits);
__IO_REG32_BIT(McASP0_RINTCTL,        0x01D0007C,__READ_WRITE ,__mcasp_rintctl_bits);
__IO_REG32_BIT(McASP0_RSTAT,          0x01D00080,__READ_WRITE ,__mcasp_rstat_bits);
__IO_REG32_BIT(McASP0_RSLOT,          0x01D00084,__READ       ,__mcasp_rslot_bits);
__IO_REG32_BIT(McASP0_RCLKCHK,        0x01D00088,__READ_WRITE ,__mcasp_rclkchk_bits);
__IO_REG32_BIT(McASP0_REVTCTL,        0x01D0008C,__READ_WRITE ,__mcasp_revtctl_bits);
__IO_REG32_BIT(McASP0_XGBLCTL,        0x01D000A0,__READ_WRITE ,__mcasp_xgblctl_bits);
__IO_REG32_BIT(McASP0_XMASK,          0x01D000A4,__READ_WRITE ,__mcasp_xmask_bits);
__IO_REG32_BIT(McASP0_XFMT,           0x01D000A8,__READ_WRITE ,__mcasp_xfmt_bits);
__IO_REG32_BIT(McASP0_AFSXCTL,        0x01D000AC,__READ_WRITE ,__mcasp_afsxctl_bits);
__IO_REG32_BIT(McASP0_ACLKXCTL,       0x01D000B0,__READ_WRITE ,__mcasp_aclkxctl_bits);
__IO_REG32_BIT(McASP0_AHCLKXCTL,      0x01D000B4,__READ_WRITE ,__mcasp_ahclkxctl_bits);
__IO_REG32_BIT(McASP0_XTDM,           0x01D000B8,__READ_WRITE ,__mcasp_xtdm_bits);
__IO_REG32_BIT(McASP0_XINTCTL,        0x01D000BC,__READ_WRITE ,__mcasp_xintctl_bits);
__IO_REG32_BIT(McASP0_XSTAT,          0x01D000C0,__READ_WRITE ,__mcasp_xstat_bits);
__IO_REG32_BIT(McASP0_XSLOT,          0x01D000C4,__READ       ,__mcasp_xslot_bits);
__IO_REG32_BIT(McASP0_XCLKCHK,        0x01D000C8,__READ_WRITE ,__mcasp_xclkchk_bits);
__IO_REG32_BIT(McASP0_XEVTCTL,        0x01D000CC,__READ_WRITE ,__mcasp_xevtctl_bits);
__IO_REG32(    McASP0_DITCSRA0,       0x01D00100,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRA1,       0x01D00104,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRA2,       0x01D00108,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRA3,       0x01D0010C,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRA4,       0x01D00110,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRA5,       0x01D00114,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB0,       0x01D00118,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB1,       0x01D0011C,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB2,       0x01D00120,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB3,       0x01D00124,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB4,       0x01D00128,__READ_WRITE );
__IO_REG32(    McASP0_DITCSRB5,       0x01D0012C,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA0,       0x01D00130,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA1,       0x01D00134,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA2,       0x01D00138,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA3,       0x01D0013C,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA4,       0x01D00140,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRA5,       0x01D00144,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB0,       0x01D00148,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB1,       0x01D0014C,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB2,       0x01D00150,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB3,       0x01D00154,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB4,       0x01D00158,__READ_WRITE );
__IO_REG32(    McASP0_DITUDRB5,       0x01D0015C,__READ_WRITE );
__IO_REG32_BIT(McASP0_SRCTL0,         0x01D00180,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL1,         0x01D00184,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL2,         0x01D00188,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL3,         0x01D0018C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL4,         0x01D00190,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL5,         0x01D00194,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL6,         0x01D00198,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL7,         0x01D0019C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL8,         0x01D001A0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL9,         0x01D001A4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL10,        0x01D001A8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL11,        0x01D001AC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL12,        0x01D001B0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL13,        0x01D001B4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL14,        0x01D001B8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP0_SRCTL15,        0x01D001BC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32(    McASP0_XBUF0,          0x01D00200,__READ_WRITE );
__IO_REG32(    McASP0_XBUF1,          0x01D00204,__READ_WRITE );
__IO_REG32(    McASP0_XBUF2,          0x01D00208,__READ_WRITE );
__IO_REG32(    McASP0_XBUF3,          0x01D0020C,__READ_WRITE );
__IO_REG32(    McASP0_XBUF4,          0x01D00210,__READ_WRITE );
__IO_REG32(    McASP0_XBUF5,          0x01D00214,__READ_WRITE );
__IO_REG32(    McASP0_XBUF6,          0x01D00218,__READ_WRITE );
__IO_REG32(    McASP0_XBUF7,          0x01D0021C,__READ_WRITE );
__IO_REG32(    McASP0_XBUF8,          0x01D00220,__READ_WRITE );
__IO_REG32(    McASP0_XBUF9,          0x01D00224,__READ_WRITE );
__IO_REG32(    McASP0_XBUF10,         0x01D00228,__READ_WRITE );
__IO_REG32(    McASP0_XBUF11,         0x01D0022C,__READ_WRITE );
__IO_REG32(    McASP0_XBUF12,         0x01D00230,__READ_WRITE );
__IO_REG32(    McASP0_XBUF13,         0x01D00234,__READ_WRITE );
__IO_REG32(    McASP0_XBUF14,         0x01D00238,__READ_WRITE );
__IO_REG32(    McASP0_XBUF15,         0x01D0023C,__READ_WRITE );
__IO_REG32(    McASP0_RBUF0,          0x01D00280,__READ_WRITE );
__IO_REG32(    McASP0_RBUF1,          0x01D00284,__READ_WRITE );
__IO_REG32(    McASP0_RBUF2,          0x01D00288,__READ_WRITE );
__IO_REG32(    McASP0_RBUF3,          0x01D0028C,__READ_WRITE );
__IO_REG32(    McASP0_RBUF4,          0x01D00290,__READ_WRITE );
__IO_REG32(    McASP0_RBUF5,          0x01D00294,__READ_WRITE );
__IO_REG32(    McASP0_RBUF6,          0x01D00298,__READ_WRITE );
__IO_REG32(    McASP0_RBUF7,          0x01D0029C,__READ_WRITE );
__IO_REG32(    McASP0_RBUF8,          0x01D002A0,__READ_WRITE );
__IO_REG32(    McASP0_RBUF9,          0x01D002A4,__READ_WRITE );
__IO_REG32(    McASP0_RBUF10,         0x01D002A8,__READ_WRITE );
__IO_REG32(    McASP0_RBUF11,         0x01D002AC,__READ_WRITE );
__IO_REG32(    McASP0_RBUF12,         0x01D002B0,__READ_WRITE );
__IO_REG32(    McASP0_RBUF13,         0x01D002B4,__READ_WRITE );
__IO_REG32(    McASP0_RBUF14,         0x01D002B8,__READ_WRITE );
__IO_REG32(    McASP0_RBUF15,         0x01D002BC,__READ_WRITE );
__IO_REG32(    McASP0_RBUF,           0x01D02000,__READ_WRITE );
#define McASP0_XBUF      McASP0_RBUF
__IO_REG32(    McASP0_AFIFOREV,       0x01D01000,__READ_WRITE );
__IO_REG32_BIT(McASP0_WFIFOCTL,       0x01D01010,__READ_WRITE ,__mcasp_wfifoctl_bits);
__IO_REG32_BIT(McASP0_WFIFOSTS,       0x01D01014,__READ       ,__mcasp_wfifosts_bits);
__IO_REG32_BIT(McASP0_RFIFOCTL,       0x01D01018,__READ_WRITE ,__mcasp_rfifoctl_bits);
__IO_REG32_BIT(McASP0_RFIFOSTS,       0x01D0101C,__READ       ,__mcasp_rfifosts_bits);

/***************************************************************************
 **
 ** McASP1
 **
 ***************************************************************************/
__IO_REG32(    McASP1_REV,            0x01D04000,__READ       );
__IO_REG32_BIT(McASP1_PFUNC,          0x01D04010,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP1_PDIR,           0x01D04014,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP1_PDOUT,          0x01D04018,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP1_PDIN,           0x01D0401C,__READ_WRITE ,__mcasp_pfunc_bits);
#define McASP1_PDSET     McASP1_PDIN
#define McASP1_PDSET_bit McASP1_PDIN_bit
__IO_REG32_BIT(McASP1_PDCLR,          0x01D04020,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP1_GBLCTL,         0x01D04044,__READ_WRITE ,__mcasp_gblctl_bits);
__IO_REG32_BIT(McASP1_AMUTE,          0x01D04048,__READ_WRITE ,__mcasp_amute_bits);
__IO_REG32_BIT(McASP1_DLBCTL,         0x01D0404C,__READ_WRITE ,__mcasp_dlbctl_bits);
__IO_REG32_BIT(McASP1_DITCTL,         0x01D04050,__READ_WRITE ,__mcasp_ditctl_bits);
__IO_REG32_BIT(McASP1_RGBLCTL,        0x01D04060,__READ_WRITE ,__mcasp_rgblctl_bits);
__IO_REG32_BIT(McASP1_RMASK,          0x01D04064,__READ_WRITE ,__mcasp_rmask_bits);
__IO_REG32_BIT(McASP1_RFMT,           0x01D04068,__READ_WRITE ,__mcasp_rfmt_bits);
__IO_REG32_BIT(McASP1_AFSRCTL,        0x01D0406C,__READ_WRITE ,__mcasp_afsrctl_bits);
__IO_REG32_BIT(McASP1_ACLKRCTL,       0x01D04070,__READ_WRITE ,__mcasp_aclkrctl_bits);
__IO_REG32_BIT(McASP1_AHCLKRCTL,      0x01D04074,__READ_WRITE ,__mcasp_ahclkrctl_bits);
__IO_REG32_BIT(McASP1_RTDM,           0x01D04078,__READ_WRITE ,__mcasp_rtdm_bits);
__IO_REG32_BIT(McASP1_RINTCTL,        0x01D0407C,__READ_WRITE ,__mcasp_rintctl_bits);
__IO_REG32_BIT(McASP1_RSTAT,          0x01D04080,__READ_WRITE ,__mcasp_rstat_bits);
__IO_REG32_BIT(McASP1_RSLOT,          0x01D04084,__READ       ,__mcasp_rslot_bits);
__IO_REG32_BIT(McASP1_RCLKCHK,        0x01D04088,__READ_WRITE ,__mcasp_rclkchk_bits);
__IO_REG32_BIT(McASP1_REVTCTL,        0x01D0408C,__READ_WRITE ,__mcasp_revtctl_bits);
__IO_REG32_BIT(McASP1_XGBLCTL,        0x01D040A0,__READ_WRITE ,__mcasp_xgblctl_bits);
__IO_REG32_BIT(McASP1_XMASK,          0x01D040A4,__READ_WRITE ,__mcasp_xmask_bits);
__IO_REG32_BIT(McASP1_XFMT,           0x01D040A8,__READ_WRITE ,__mcasp_xfmt_bits);
__IO_REG32_BIT(McASP1_AFSXCTL,        0x01D040AC,__READ_WRITE ,__mcasp_afsxctl_bits);
__IO_REG32_BIT(McASP1_ACLKXCTL,       0x01D040B0,__READ_WRITE ,__mcasp_aclkxctl_bits);
__IO_REG32_BIT(McASP1_AHCLKXCTL,      0x01D040B4,__READ_WRITE ,__mcasp_ahclkxctl_bits);
__IO_REG32_BIT(McASP1_XTDM,           0x01D040B8,__READ_WRITE ,__mcasp_xtdm_bits);
__IO_REG32_BIT(McASP1_XINTCTL,        0x01D040BC,__READ_WRITE ,__mcasp_xintctl_bits);
__IO_REG32_BIT(McASP1_XSTAT,          0x01D040C0,__READ_WRITE ,__mcasp_xstat_bits);
__IO_REG32_BIT(McASP1_XSLOT,          0x01D040C4,__READ       ,__mcasp_xslot_bits);
__IO_REG32_BIT(McASP1_XCLKCHK,        0x01D040C8,__READ_WRITE ,__mcasp_xclkchk_bits);
__IO_REG32_BIT(McASP1_XEVTCTL,        0x01D040CC,__READ_WRITE ,__mcasp_xevtctl_bits);
__IO_REG32(    McASP1_DITCSRA0,       0x01D04100,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRA1,       0x01D04104,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRA2,       0x01D04108,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRA3,       0x01D0410C,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRA4,       0x01D04110,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRA5,       0x01D04114,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB0,       0x01D04118,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB1,       0x01D0411C,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB2,       0x01D04120,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB3,       0x01D04124,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB4,       0x01D04128,__READ_WRITE );
__IO_REG32(    McASP1_DITCSRB5,       0x01D0412C,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA0,       0x01D04130,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA1,       0x01D04134,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA2,       0x01D04138,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA3,       0x01D0413C,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA4,       0x01D04140,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRA5,       0x01D04144,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB0,       0x01D04148,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB1,       0x01D0414C,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB2,       0x01D04150,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB3,       0x01D04154,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB4,       0x01D04158,__READ_WRITE );
__IO_REG32(    McASP1_DITUDRB5,       0x01D0415C,__READ_WRITE );
__IO_REG32_BIT(McASP1_SRCTL0,         0x01D04180,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL1,         0x01D04184,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL2,         0x01D04188,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL3,         0x01D0418C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL4,         0x01D04190,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL5,         0x01D04194,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL6,         0x01D04198,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL7,         0x01D0419C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL8,         0x01D041A0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL9,         0x01D041A4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL10,        0x01D041A8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL11,        0x01D041AC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL12,        0x01D041B0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL13,        0x01D041B4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL14,        0x01D041B8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP1_SRCTL15,        0x01D041BC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32(    McASP1_XBUF0,          0x01D04200,__READ_WRITE );
__IO_REG32(    McASP1_XBUF1,          0x01D04204,__READ_WRITE );
__IO_REG32(    McASP1_XBUF2,          0x01D04208,__READ_WRITE );
__IO_REG32(    McASP1_XBUF3,          0x01D0420C,__READ_WRITE );
__IO_REG32(    McASP1_XBUF4,          0x01D04210,__READ_WRITE );
__IO_REG32(    McASP1_XBUF5,          0x01D04214,__READ_WRITE );
__IO_REG32(    McASP1_XBUF6,          0x01D04218,__READ_WRITE );
__IO_REG32(    McASP1_XBUF7,          0x01D0421C,__READ_WRITE );
__IO_REG32(    McASP1_XBUF8,          0x01D04220,__READ_WRITE );
__IO_REG32(    McASP1_XBUF9,          0x01D04224,__READ_WRITE );
__IO_REG32(    McASP1_XBUF10,         0x01D04228,__READ_WRITE );
__IO_REG32(    McASP1_XBUF11,         0x01D0422C,__READ_WRITE );
__IO_REG32(    McASP1_XBUF12,         0x01D04230,__READ_WRITE );
__IO_REG32(    McASP1_XBUF13,         0x01D04234,__READ_WRITE );
__IO_REG32(    McASP1_XBUF14,         0x01D04238,__READ_WRITE );
__IO_REG32(    McASP1_XBUF15,         0x01D0423C,__READ_WRITE );
__IO_REG32(    McASP1_RBUF0,          0x01D04280,__READ_WRITE );
__IO_REG32(    McASP1_RBUF1,          0x01D04284,__READ_WRITE );
__IO_REG32(    McASP1_RBUF2,          0x01D04288,__READ_WRITE );
__IO_REG32(    McASP1_RBUF3,          0x01D0428C,__READ_WRITE );
__IO_REG32(    McASP1_RBUF4,          0x01D04290,__READ_WRITE );
__IO_REG32(    McASP1_RBUF5,          0x01D04294,__READ_WRITE );
__IO_REG32(    McASP1_RBUF6,          0x01D04298,__READ_WRITE );
__IO_REG32(    McASP1_RBUF7,          0x01D0429C,__READ_WRITE );
__IO_REG32(    McASP1_RBUF8,          0x01D042A0,__READ_WRITE );
__IO_REG32(    McASP1_RBUF9,          0x01D042A4,__READ_WRITE );
__IO_REG32(    McASP1_RBUF10,         0x01D042A8,__READ_WRITE );
__IO_REG32(    McASP1_RBUF11,         0x01D042AC,__READ_WRITE );
__IO_REG32(    McASP1_RBUF12,         0x01D042B0,__READ_WRITE );
__IO_REG32(    McASP1_RBUF13,         0x01D042B4,__READ_WRITE );
__IO_REG32(    McASP1_RBUF14,         0x01D042B8,__READ_WRITE );
__IO_REG32(    McASP1_RBUF15,         0x01D042BC,__READ_WRITE );
__IO_REG32(    McASP1_RBUF,           0x01D06000,__READ_WRITE );
#define McASP1_XBUF      McASP1_RBUF
__IO_REG32(    McASP1_AFIFOREV,       0x01D05000,__READ_WRITE );
__IO_REG32_BIT(McASP1_WFIFOCTL,       0x01D05010,__READ_WRITE ,__mcasp_wfifoctl_bits);
__IO_REG32_BIT(McASP1_WFIFOSTS,       0x01D05014,__READ       ,__mcasp_wfifosts_bits);
__IO_REG32_BIT(McASP1_RFIFOCTL,       0x01D05018,__READ_WRITE ,__mcasp_rfifoctl_bits);
__IO_REG32_BIT(McASP1_RFIFOSTS,       0x01D0501C,__READ       ,__mcasp_rfifosts_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0GCR0,              0x01C41000,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(SPI0GCR1,              0x01C41004,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(SPI0INT0,              0x01C41008,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(SPI0LVL,               0x01C4100C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(SPI0FLG,               0x01C41010,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(SPI0PC0,               0x01C41014,__READ_WRITE ,__spipc0_bits);
__IO_REG32_BIT(SPI0PC1,               0x01C41018,__READ_WRITE ,__spipc1_bits);
__IO_REG32_BIT(SPI0PC2,               0x01C4101C,__READ_WRITE ,__spipc2_bits);
__IO_REG32_BIT(SPI0PC3,               0x01C41020,__READ_WRITE ,__spipc3_bits);
__IO_REG32_BIT(SPI0PC4,               0x01C41024,__READ_WRITE ,__spipc4_bits);
__IO_REG32_BIT(SPI0PC5,               0x01C41028,__READ_WRITE ,__spipc5_bits);
__IO_REG32_BIT(SPI0DAT0,              0x01C41038,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(SPI0DAT1,              0x01C4103C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(SPI0BUF,               0x01C41040,__READ_WRITE ,__spibuf_bits);
__IO_REG32_BIT(SPI0EMU,               0x01C41044,__READ       ,__spiemu_bits);
__IO_REG32_BIT(SPI0DELAY,             0x01C41048,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(SPI0DEF,               0x01C4104C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(SPI0FMT0,              0x01C41050,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI0FMT1,              0x01C41054,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI0FMT2,              0x01C41058,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI0FMT3,              0x01C4105C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI0INTVEC1,           0x01C41064,__READ_WRITE ,__spiintvec1_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1GCR0,              0x01E12000,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(SPI1GCR1,              0x01E12004,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(SPI1INT0,              0x01E12008,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(SPI1LVL,               0x01E1200C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(SPI1FLG,               0x01E12010,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(SPI1PC0,               0x01E12014,__READ_WRITE ,__spipc0_bits);
__IO_REG32_BIT(SPI1PC1,               0x01E12018,__READ_WRITE ,__spipc1_bits);
__IO_REG32_BIT(SPI1PC2,               0x01E1201C,__READ_WRITE ,__spipc2_bits);
__IO_REG32_BIT(SPI1PC3,               0x01E12020,__READ_WRITE ,__spipc3_bits);
__IO_REG32_BIT(SPI1PC4,               0x01E12024,__READ_WRITE ,__spipc4_bits);
__IO_REG32_BIT(SPI1PC5,               0x01E12028,__READ_WRITE ,__spipc5_bits);
__IO_REG32_BIT(SPI1DAT0,              0x01E12038,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(SPI1DAT1,              0x01E1203C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(SPI1BUF,               0x01E12040,__READ_WRITE ,__spibuf_bits);
__IO_REG32_BIT(SPI1EMU,               0x01E12044,__READ       ,__spiemu_bits);
__IO_REG32_BIT(SPI1DELAY,             0x01E12048,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(SPI1DEF,               0x01E1204C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(SPI1FMT0,              0x01E12050,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT1,              0x01E12054,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT2,              0x01E12058,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT3,              0x01E1205C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1INTVEC1,           0x01E12064,__READ_WRITE ,__spiintvec1_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(IC0OAR,                0x01C22000,__READ_WRITE ,__icoar_bits);
__IO_REG32_BIT(IC0IMR,                0x01C22004,__READ_WRITE ,__icimr_bits);
__IO_REG32_BIT(IC0STR,                0x01C22008,__READ_WRITE ,__icstr_bits);
__IO_REG32_BIT(IC0CLKL,               0x01C2200C,__READ_WRITE ,__icclkl_bits);
__IO_REG32_BIT(IC0CLKH,               0x01C22010,__READ_WRITE ,__icclkh_bits);
__IO_REG32_BIT(IC0CNT,                0x01C22014,__READ_WRITE ,__iccnt_bits);
__IO_REG32_BIT(IC0DRR,                0x01C22018,__READ_WRITE ,__icdrr_bits);
__IO_REG32_BIT(IC0SAR,                0x01C2201C,__READ_WRITE ,__icsar_bits);
__IO_REG32_BIT(IC0DXR,                0x01C22020,__READ_WRITE ,__icdrr_bits);
__IO_REG32_BIT(IC0MDR,                0x01C22024,__READ_WRITE ,__icmdr_bits);
__IO_REG32_BIT(IC0IVR,                0x01C22028,__READ_WRITE ,__icivr_bits);
__IO_REG32_BIT(IC0EMDR,               0x01C2202C,__READ_WRITE ,__icemdr_bits);
__IO_REG32_BIT(IC0PSC,                0x01C22030,__READ_WRITE ,__icpsc_bits);
__IO_REG32(    IC0REVID1,             0x01C22034,__READ       );
__IO_REG32(    IC0REVID2,             0x01C22038,__READ       );
__IO_REG32_BIT(IC0PFUNC,              0x01C22048,__READ_WRITE ,__icpfunc_bits);
__IO_REG32_BIT(IC0PDIR,               0x01C2204C,__READ_WRITE ,__icpdir_bits);
__IO_REG32_BIT(IC0PDIN,               0x01C22050,__READ_WRITE ,__icpdin_bits);
__IO_REG32_BIT(IC0PDOUT,              0x01C22054,__READ_WRITE ,__icpdout_bits);
__IO_REG32_BIT(IC0PDSET,              0x01C22058,__READ_WRITE ,__icpdset_bits);
__IO_REG32_BIT(IC0PDCLR,              0x01C2205C,__READ_WRITE ,__icpdclr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(IC1OAR,                0x01E28000,__READ_WRITE ,__icoar_bits);
__IO_REG32_BIT(IC1IMR,                0x01E28004,__READ_WRITE ,__icimr_bits);
__IO_REG32_BIT(IC1STR,                0x01E28008,__READ_WRITE ,__icstr_bits);
__IO_REG32_BIT(IC1CLKL,               0x01E2800C,__READ_WRITE ,__icclkl_bits);
__IO_REG32_BIT(IC1CLKH,               0x01E28010,__READ_WRITE ,__icclkh_bits);
__IO_REG32_BIT(IC1CNT,                0x01E28014,__READ_WRITE ,__iccnt_bits);
__IO_REG32_BIT(IC1DRR,                0x01E28018,__READ_WRITE ,__icdrr_bits);
__IO_REG32_BIT(IC1SAR,                0x01E2801C,__READ_WRITE ,__icsar_bits);
__IO_REG32_BIT(IC1DXR,                0x01E28020,__READ_WRITE ,__icdrr_bits);
__IO_REG32_BIT(IC1MDR,                0x01E28024,__READ_WRITE ,__icmdr_bits);
__IO_REG32_BIT(IC1IVR,                0x01E28028,__READ_WRITE ,__icivr_bits);
__IO_REG32_BIT(IC1EMDR,               0x01E2802C,__READ_WRITE ,__icemdr_bits);
__IO_REG32_BIT(IC1PSC,                0x01E28030,__READ_WRITE ,__icpsc_bits);
__IO_REG32(    IC1REVID1,             0x01E28034,__READ       );
__IO_REG32(    IC1REVID2,             0x01E28038,__READ       );
__IO_REG32_BIT(IC1PFUNC,              0x01E28048,__READ_WRITE ,__icpfunc_bits);
__IO_REG32_BIT(IC1PDIR,               0x01E2804C,__READ_WRITE ,__icpdir_bits);
__IO_REG32_BIT(IC1PDIN,               0x01E28050,__READ_WRITE ,__icpdin_bits);
__IO_REG32_BIT(IC1PDOUT,              0x01E28054,__READ_WRITE ,__icpdout_bits);
__IO_REG32_BIT(IC1PDSET,              0x01E28058,__READ_WRITE ,__icpdset_bits);
__IO_REG32_BIT(IC1PDCLR,              0x01E2805C,__READ_WRITE ,__icpdclr_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0_RBR,             0x01C42000,__READ_WRITE ,__uartrbr_bits);
#define UART0_THR       UART0_RBR
#define UART0_THR_bit   UART0_RBR_bit
__IO_REG32_BIT(UART0_IER,             0x01C42004,__READ_WRITE ,__uartier_bits);
__IO_REG32_BIT(UART0_IIR,             0x01C42008,__READ_WRITE ,__uartiir_bits);
#define UART0_FCR     UART0_IIR
#define UART0_FCR_bit UART0_IIR_bit.UARTx_FCR
__IO_REG32_BIT(UART0_LCR,             0x01C4200C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(UART0_MCR,             0x01C42010,__READ_WRITE ,__uartmcr_bits);
__IO_REG32_BIT(UART0_LSR,             0x01C42014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(UART0_MSR,             0x01C42018,__READ       ,__uartmsr_bits);
__IO_REG32_BIT(UART0_SCR,             0x01C4201C,__READ_WRITE ,__uartscr_bits);
__IO_REG32_BIT(UART0_DLL,             0x01C42020,__READ_WRITE ,__uartdll_bits);
__IO_REG32_BIT(UART0_DLH,             0x01C42024,__READ_WRITE ,__uartdlh_bits);
__IO_REG32(    UART0_REVID1,          0x01C42028,__READ       );
__IO_REG32_BIT(UART0_PWREMU_MGMT,     0x01C42030,__READ_WRITE ,__uartpwremu_mgmt_bits);
__IO_REG32_BIT(UART0_MDR,             0x01C42034,__READ_WRITE ,__uartmdr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1_RBR,             0x01D0C000,__READ_WRITE ,__uartrbr_bits);
#define UART1_THR       UART1_RBR
#define UART1_THR_bit   UART1_RBR_bit
__IO_REG32_BIT(UART1_IER,             0x01D0C004,__READ_WRITE ,__uartier_bits);
__IO_REG32_BIT(UART1_IIR,             0x01D0C008,__READ_WRITE ,__uartiir_bits);
#define UART1_FCR     UART1_IIR
#define UART1_FCR_bit UART1_IIR_bit.UARTx_FCR
__IO_REG32_BIT(UART1_LCR,             0x01D0C00C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(UART1_MCR,             0x01D0C010,__READ_WRITE ,__uartmcr_bits);
__IO_REG32_BIT(UART1_LSR,             0x01D0C014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(UART1_MSR,             0x01D0C018,__READ       ,__uartmsr_bits);
__IO_REG32_BIT(UART1_SCR,             0x01D0C01C,__READ_WRITE ,__uartscr_bits);
__IO_REG32_BIT(UART1_DLL,             0x01D0C020,__READ_WRITE ,__uartdll_bits);
__IO_REG32_BIT(UART1_DLH,             0x01D0C024,__READ_WRITE ,__uartdlh_bits);
__IO_REG32(    UART1_REVID1,          0x01D0C028,__READ       );
__IO_REG32_BIT(UART1_PWREMU_MGMT,     0x01D0C030,__READ_WRITE ,__uartpwremu_mgmt_bits);
__IO_REG32_BIT(UART1_MDR,             0x01D0C034,__READ_WRITE ,__uartmdr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2_RBR,             0x01D0D000,__READ_WRITE ,__uartrbr_bits);
#define UART2_THR       UART2_RBR
#define UART2_THR_bit   UART2_RBR_bit
__IO_REG32_BIT(UART2_IER,             0x01D0D004,__READ_WRITE ,__uartier_bits);
__IO_REG32_BIT(UART2_IIR,             0x01D0D008,__READ_WRITE ,__uartiir_bits);
#define UART2_FCR     UART2_IIR
#define UART2_FCR_bit UART2_IIR_bit.UARTx_FCR
__IO_REG32_BIT(UART2_LCR,             0x01D0D00C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(UART2_MCR,             0x01D0D010,__READ_WRITE ,__uartmcr_bits);
__IO_REG32_BIT(UART2_LSR,             0x01D0D014,__READ       ,__uartlsr_bits);
__IO_REG32_BIT(UART2_MSR,             0x01D0D018,__READ       ,__uartmsr_bits);
__IO_REG32_BIT(UART2_SCR,             0x01D0D01C,__READ_WRITE ,__uartscr_bits);
__IO_REG32_BIT(UART2_DLL,             0x01D0D020,__READ_WRITE ,__uartdll_bits);
__IO_REG32_BIT(UART2_DLH,             0x01D0D024,__READ_WRITE ,__uartdlh_bits);
__IO_REG32(    UART2_REVID1,          0x01D0D028,__READ       );
__IO_REG32_BIT(UART2_PWREMU_MGMT,     0x01D0D030,__READ_WRITE ,__uartpwremu_mgmt_bits);
__IO_REG32_BIT(UART2_MDR,             0x01D0D034,__READ_WRITE ,__uartmdr_bits);

/***************************************************************************
 **
 ** USB0 (OTG)
 **
 ***************************************************************************/
__IO_REG32(    USB0_REVID,              0x01E00000,__READ       );
__IO_REG32_BIT(USB0_CTRLR,              0x01E00004,__READ_WRITE ,__usb_ctrlr_bits);
__IO_REG32_BIT(USB0_STATR,              0x01E00008,__READ       ,__usb_statr_bits);
__IO_REG32_BIT(USB0_EMUR,               0x01E0000C,__READ_WRITE ,__usb_emur_bits);
__IO_REG32_BIT(USB0_MODE,               0x01E00010,__READ_WRITE ,__usb_mode_bits);
__IO_REG32_BIT(USB0_AUTOREQ,            0x01E00014,__READ_WRITE ,__usb_autoreq_bits);
__IO_REG32(    USB0_SRPFIXTIME,         0x01E00018,__READ_WRITE );
__IO_REG32_BIT(USB0_TEARDOWN,           0x01E0001C,__READ_WRITE ,__usb_teardown_bits);
__IO_REG32_BIT(USB0_INTSRCR,            0x01E00020,__READ       ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTSETR,            0x01E00024,__READ_WRITE ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTCLRR,            0x01E00028,__READ_WRITE ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTMSKR,            0x01E0002C,__READ       ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTMSKSETR,         0x01E00030,__READ_WRITE ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTMSKCLRR,         0x01E00034,__READ_WRITE ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_INTMASKEDR,         0x01E00038,__READ       ,__usb_intsrcr_bits);
__IO_REG32_BIT(USB0_EOIR,               0x01E0003C,__READ_WRITE ,__usb_eoir_bits);
__IO_REG32_BIT(USB0_GENRNDISSZ1,        0x01E00050,__READ_WRITE ,__usb_genrndissz1_bits);
__IO_REG32_BIT(USB0_GENRNDISSZ2,        0x01E00054,__READ_WRITE ,__usb_genrndissz2_bits);
__IO_REG32_BIT(USB0_GENRNDISSZ3,        0x01E00058,__READ_WRITE ,__usb_genrndissz3_bits);
__IO_REG32_BIT(USB0_GENRNDISSZ4,        0x01E0005C,__READ_WRITE ,__usb_genrndissz4_bits);
__IO_REG8_BIT( USB0_FADDR,              0x01E00400,__READ_WRITE ,__usb_faddr_bits);
__IO_REG8_BIT( USB0_POWER,              0x01E00401,__READ_WRITE ,__usb_power_bits);
__IO_REG16_BIT(USB0_INTRTX,             0x01E00402,__READ       ,__usb_intrtx_bits);
__IO_REG16_BIT(USB0_INTRRX,             0x01E00404,__READ       ,__usb_intrrx_bits);
__IO_REG16_BIT(USB0_INTRTXE,            0x01E00406,__READ_WRITE ,__usb_intrtx_bits);
__IO_REG16_BIT(USB0_INTRRXE,            0x01E00408,__READ_WRITE ,__usb_intrrx_bits);
__IO_REG8_BIT( USB0_INTRUSB,            0x01E0040A,__READ       ,__usb_intrusb_bits);
__IO_REG8_BIT( USB0_INTRUSBE,           0x01E0040B,__READ_WRITE ,__usb_intrusb_bits);
__IO_REG16_BIT(USB0_FRAME,              0x01E0040C,__READ       ,__usb_frame_bits);
__IO_REG8_BIT( USB0_INDEX,              0x01E0040E,__READ_WRITE ,__usb_index_bits);
__IO_REG8_BIT( USB0_TESTMODE,           0x01E0040F,__READ_WRITE ,__usb_testmode_bits);
__IO_REG16_BIT(USB0_TXMAXP,             0x01E00410,__READ_WRITE ,__usb_txmaxp_bits);
__IO_REG16_BIT(USB0_PERI_CSR0,          0x01E00412,__READ_WRITE ,__usb_peri_csr0i_bits);
#define USB0_HOST_CSR0             USB0_PERI_CSR0
#define USB0_HOST_CSR0_bit         USB0_PERI_CSR0_bit.host_csr0
#define USB0_PERI_TXCSR            USB0_PERI_CSR0
#define USB0_PERI_TXCSR_bit        USB0_PERI_CSR0_bit.peri_txcsr
#define USB0_HOST_TXCSR            USB0_PERI_CSR0
#define USB0_HOST_TXCSR_bit        USB0_PERI_CSR0_bit.host_txcsr
__IO_REG16_BIT(USB0_RXMAXP,             0x01E00414,__READ_WRITE ,__usb_rxmaxp_bits);
__IO_REG16_BIT(USB0_PERI_RXCSR,         0x01E00416,__READ_WRITE ,__usb_peri_rxcsr_bits);
#define USB0_HOST_RXCSR            USB0_PERI_RXCSR
#define USB0_HOST_RXCSR_bit        USB0_PERI_RXCSR_bit.host_rxcsr
__IO_REG16_BIT(USB0_COUNT0,             0x01E00418,__READ_WRITE ,__usb_count0i_bits);
#define USB0_RXCOUNT               USB0_COUNT0
#define USB0_RXCOUNT_bit           USB0_COUNT0_bit
__IO_REG8_BIT( USB0_HOST_TYPE0,         0x01E0041A,__READ_WRITE ,__usb_host_txtype_bits);
#define USB0_HOST_TXTYPE           USB0_HOST_TYPE0
#define USB0_HOST_TXTYPE_bit       USB0_HOST_TYPE0_bit
__IO_REG8_BIT( USB0_HOST_NAKLIMIT0,    0x01E0041B,__READ_WRITE ,__usb_host_naklimit0i_bits);
#define USB0_HOST_TXINTERVAL       USB0_HOST_NAKLIMIT0
__IO_REG8_BIT( USB0_HOST_RXTYPE,        0x01E0041C,__READ_WRITE ,__usb_host_rxtype_bits);
__IO_REG8(     USB0_HOST_RXINTERVAL ,   0x01E0041D,__READ_WRITE );
__IO_REG8_BIT( USB0_CONFIGDATA,         0x01E0041F,__READ       ,__usb_configdata_bits);
__IO_REG32(    USB0_FIFO0,              0x01E00420,__READ_WRITE );
__IO_REG32(    USB0_FIFO1,              0x01E00424,__READ_WRITE );
__IO_REG32(    USB0_FIFO2,              0x01E00428,__READ_WRITE );
__IO_REG32(    USB0_FIFO3,              0x01E0042C,__READ_WRITE );
__IO_REG32(    USB0_FIFO4,              0x01E00430,__READ_WRITE );
__IO_REG8_BIT( USB0_DEVCTL,             0x01E00460,__READ_WRITE ,__usb_devctl_bits);
__IO_REG8_BIT( USB0_TXFIFOSZ,           0x01E00462,__READ_WRITE ,__usb_txfifosz_bits);
__IO_REG8_BIT( USB0_RXFIFOSZ,           0x01E00463,__READ_WRITE ,__usb_txfifosz_bits);
__IO_REG16_BIT(USB0_TXFIFOADDR,         0x01E00464,__READ_WRITE ,__usb_txfifoaddr_bits);
__IO_REG16_BIT(USB0_RXFIFOADDR,         0x01E00466,__READ_WRITE ,__usb_txfifoaddr_bits);
__IO_REG8_BIT( USB0_TXFUNCADDR_H_EP0,   0x01E00480,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBADDR_H_EP0,    0x01E00482,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBPORT_H_EP0,    0x01E00483,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_RXFUNCADDR_H_EP0,   0x01E00484,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBADDR_H_EP0,    0x01E00486,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBPORT_H_EP0,    0x01E00487,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_TXFUNCADDR_H_EP1,   0x01E00488,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBADDR_H_EP1,    0x01E0048A,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBPORT_H_EP1,    0x01E0048B,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_RXFUNCADDR_H_EP1,   0x01E0048C,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBADDR_H_EP1,    0x01E0048E,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBPORT_H_EP1,    0x01E0048F,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_TXFUNCADDR_H_EP2,   0x01E00490,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBADDR_H_EP2,    0x01E00492,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBPORT_H_EP2,    0x01E00493,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_RXFUNCADDR_H_EP2,   0x01E00494,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBADDR_H_EP2,    0x01E00496,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBPORT_H_EP2,    0x01E00497,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_TXFUNCADDR_H_EP3,   0x01E00498,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBADDR_H_EP3,    0x01E0049A,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBPORT_H_EP3,    0x01E0049B,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_RXFUNCADDR_H_EP3,   0x01E0049C,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBADDR_H_EP3,    0x01E0049E,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBPORT_H_EP3,    0x01E0049F,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_TXFUNCADDR_H_EP4,   0x01E004A0,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBADDR_H_EP4,    0x01E004A2,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_TXHUBPORT_H_EP4,    0x01E004A3,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG8_BIT( USB0_RXFUNCADDR_H_EP4,   0x01E004A4,__READ_WRITE ,__usb_txfuncaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBADDR_H_EP4,    0x01E004A6,__READ_WRITE ,__usb_txhubaddr_h_bits);
__IO_REG8_BIT( USB0_RXHUBPORT_H_EP4,    0x01E004A7,__READ_WRITE ,__usb_txhubport_h_bits);
__IO_REG16_BIT(USB0_PERI_CSR0_EP0,      0x01E00502,__READ_WRITE ,__usb_peri_csr0_bits);
#define USB0_HOST_CSR0_EP0            USB0_PERI_CSR0_EP0C
#define USB0_HOST_CSR0_EP0_bit        USB0_PERI_CSR0_EP0C_bit.host_csr0_ep0
__IO_REG16_BIT(USB0_COUNT0_EP0,         0x01E00508,__READ_WRITE ,__usb_count0_bits);
__IO_REG8_BIT( USB0_HOST_TYPE0_EP0,     0x01E0050A,__READ_WRITE ,__usb_host_type0_bits);
__IO_REG8_BIT( USB0_HOST_NAKLIMIT0_EP0, 0x01E0050B,__READ_WRITE ,__usb_host_naklimit0_bits);
__IO_REG8_BIT( USB0_CONFIGDATA_EP0,     0x01E0050F,__READ_WRITE ,__usb_configdata_bits);
__IO_REG16_BIT(USB0_TXMAXP_EP1,         0x01E00510,__READ_WRITE ,__usb_txmaxp_bits);
__IO_REG16_BIT(USB0_PERI_TXCSR_EP1,     0x01E00512,__READ_WRITE ,__usb_peri_txcsr_bits);
#define USB0_HOST_TXCSR_EP1           USB0_PERI_TXCSR_EP1
#define USB0_HOST_TXCSR_EP1_bit       USB0_PERI_TXCSR_EP1_bit.host_txcsr
__IO_REG16_BIT(USB0_RXMAXP_EP1,         0x01E00514,__READ_WRITE ,__usb_rxmaxp_bits);
__IO_REG16_BIT(USB0_PERI_RXCSR_EP1,     0x01E00516,__READ_WRITE ,__usb_peri_rxcsr_bits);
#define USB0_HOST_RXCSR_EP1           USB0_PERI_RXCSR_EP1
#define USB0_HOST_RXCSR_EP1_bit       USB0_PERI_RXCSR_EP1_bit.host_rxcsr
__IO_REG16_BIT(USB0_RXCOUNT_EP1,        0x01E00518,__READ_WRITE ,__usb_rxcount_bits);
__IO_REG8_BIT( USB0_HOST_TXTYPE_EP1,    0x01E0051A,__READ_WRITE ,__usb_host_txtype_bits);
__IO_REG8(     USB0_HOST_TXINTERVAL_EP1,0x01E0051B,__READ_WRITE );
__IO_REG8_BIT( USB0_HOST_RXTYPE_EP1,    0x01E0051C,__READ_WRITE ,__usb_host_rxtype_bits);
__IO_REG8(     USB0_HOST_RXINTERVAL_EP1,0x01E0051D,__READ_WRITE );
__IO_REG16_BIT(USB0_TXMAXP_EP2,         0x01E00520,__READ_WRITE ,__usb_txmaxp_bits);
__IO_REG16_BIT(USB0_PERI_TXCSR_EP2,     0x01E00522,__READ_WRITE ,__usb_peri_txcsr_bits);
#define USB0_HOST_TXCSR_EP2           USB0_PERI_TXCSR_EP2
#define USB0_HOST_TXCSR_EP2_bit       USB0_PERI_TXCSR_EP2_bit.host_txcsr
__IO_REG16_BIT(USB0_RXMAXP_EP2,         0x01E00524,__READ_WRITE ,__usb_rxmaxp_bits);
__IO_REG16_BIT(USB0_PERI_RXCSR_EP2,     0x01E00526,__READ_WRITE ,__usb_peri_rxcsr_bits);
#define USB0_HOST_RXCSR_EP2           USB0_PERI_RXCSR_EP2
#define USB0_HOST_RXCSR_EP2_bit       USB0_PERI_RXCSR_EP2_bit.host_rxcsr
__IO_REG16_BIT(USB0_RXCOUNT_EP2,        0x01E00528,__READ_WRITE ,__usb_rxcount_bits);
__IO_REG8_BIT( USB0_HOST_TXTYPE_EP2,    0x01E0052A,__READ_WRITE ,__usb_host_txtype_bits);
__IO_REG8(     USB0_HOST_TXINTERVAL_EP2,0x01E0052B,__READ_WRITE );
__IO_REG8_BIT( USB0_HOST_RXTYPE_EP2,    0x01E0052C,__READ_WRITE ,__usb_host_rxtype_bits);
__IO_REG8(     USB0_HOST_RXINTERVAL_EP2,0x01E0052D,__READ_WRITE );
__IO_REG16_BIT(USB0_TXMAXP_EP3,         0x01E00530,__READ_WRITE ,__usb_txmaxp_bits);
__IO_REG16_BIT(USB0_PERI_TXCSR_EP3,     0x01E00532,__READ_WRITE ,__usb_peri_txcsr_bits);
#define USB0_HOST_TXCSR_EP3           USB0_PERI_TXCSR_EP3C
#define USB0_HOST_TXCSR_EP3_bit       USB0_PERI_TXCSR_EP3C_bit.host_txcsr
__IO_REG16_BIT(USB0_RXMAXP_EP3,         0x01E00534,__READ_WRITE ,__usb_rxmaxp_bits);
__IO_REG16_BIT(USB0_PERI_RXCSR_EP3,     0x01E00536,__READ_WRITE ,__usb_peri_rxcsr_bits);
#define USB0_HOST_RXCSR_EP3           USB0_PERI_RXCSR_EP3
#define USB0_HOST_RXCSR_EP3_bit       USB0_PERI_RXCSR_EP3_bit.host_rxcsr
__IO_REG16_BIT(USB0_RXCOUNT_EP3,        0x01E00538,__READ_WRITE ,__usb_rxcount_bits);
__IO_REG8_BIT( USB0_HOST_TXTYPE_EP3,    0x01E0053A,__READ_WRITE ,__usb_host_txtype_bits);
__IO_REG8(     USB0_HOST_TXINTERVAL_EP3,0x01E0053B,__READ_WRITE );
__IO_REG8_BIT( USB0_HOST_RXTYPE_EP3,    0x01E0053C,__READ_WRITE ,__usb_host_rxtype_bits);
__IO_REG8(     USB0_HOST_RXINTERVAL_EP3,0x01E0053D,__READ_WRITE );
__IO_REG16_BIT(USB0_TXMAXP_EP4,         0x01E00540,__READ_WRITE ,__usb_txmaxp_bits);
__IO_REG16_BIT(USB0_PERI_TXCSR_EP4,     0x01E00542,__READ_WRITE ,__usb_peri_txcsr_bits);
#define USB0_HOST_TXCSR_EP4           USB0_PERI_TXCSR_EP4
#define USB0_HOST_TXCSR_EP4_bit       USB0_PERI_TXCSR_EP4_bit.host_txcsr
__IO_REG16_BIT(USB0_RXMAXP_EP4,         0x01E00544,__READ_WRITE ,__usb_rxmaxp_bits);
__IO_REG16_BIT(USB0_PERI_RXCSR_EP4,     0x01E00546,__READ_WRITE ,__usb_peri_rxcsr_bits);
#define USB0_HOST_RXCSR_EP4           USB0_PERI_RXCSR_EP4
#define USB0_HOST_RXCSR_EP4_bit       USB0_PERI_RXCSR_EP4_bit.host_rxcsr
__IO_REG16_BIT(USB0_RXCOUNT_EP4,        0x01E00548,__READ_WRITE ,__usb_rxcount_bits);
__IO_REG8_BIT( USB0_HOST_TXTYPE_EP4,    0x01E0054A,__READ_WRITE ,__usb_host_txtype_bits);
__IO_REG8(     USB0_HOST_TXINTERVAL_EP4,0x01E0054B,__READ_WRITE );
__IO_REG8_BIT( USB0_HOST_RXTYPE_EP4,    0x01E0054C,__READ_WRITE ,__usb_host_rxtype_bits);
__IO_REG8(     USB0_HOST_RXINTERVAL_EP4,0x01E0054D,__READ_WRITE );
__IO_REG32(    USB0_DMAREVID,           0x01E01000,__READ       );
__IO_REG32_BIT(USB0_TDFDQ,              0x01E01004,__READ_WRITE ,__usb_tdfdq_bits);
__IO_REG32_BIT(USB0_DMAEMU,             0x01E01008,__READ_WRITE ,__usb_dmaemu_bits);
__IO_REG32_BIT(USB0_TXGCR0,             0x01E01800,__READ_WRITE ,__usb_txgcr_bits);
__IO_REG32_BIT(USB0_RXGCR0,             0x01E01808,__READ_WRITE ,__usb_rxgcr_bits);
__IO_REG32_BIT(USB0_RXHPCRA0,           0x01E0180C,__READ_WRITE ,__usb_rxhpcra_bits);
__IO_REG32_BIT(USB0_RXHPCRB0,           0x01E01810,__READ_WRITE ,__usb_rxhpcrb_bits);
__IO_REG32_BIT(USB0_TXGCR1,             0x01E01820,__READ_WRITE ,__usb_txgcr_bits);
__IO_REG32_BIT(USB0_RXGCR1,             0x01E01828,__READ_WRITE ,__usb_rxgcr_bits);
__IO_REG32_BIT(USB0_RXHPCRA1,           0x01E0182C,__READ_WRITE ,__usb_rxhpcra_bits);
__IO_REG32_BIT(USB0_RXHPCRB1,           0x01E01830,__READ_WRITE ,__usb_rxhpcrb_bits);
__IO_REG32_BIT(USB0_TXGCR2,             0x01E01840,__READ_WRITE ,__usb_txgcr_bits);
__IO_REG32_BIT(USB0_RXGCR2,             0x01E01848,__READ_WRITE ,__usb_rxgcr_bits);
__IO_REG32_BIT(USB0_RXHPCRA2,           0x01E0184C,__READ_WRITE ,__usb_rxhpcra_bits);
__IO_REG32_BIT(USB0_RXHPCRB2,           0x01E01850,__READ_WRITE ,__usb_rxhpcrb_bits);
__IO_REG32_BIT(USB0_TXGCR3,             0x01E01860,__READ_WRITE ,__usb_txgcr_bits);
__IO_REG32_BIT(USB0_RXGCR3,             0x01E01868,__READ_WRITE ,__usb_rxgcr_bits);
__IO_REG32_BIT(USB0_RXHPCRA3,           0x01E0186C,__READ_WRITE ,__usb_rxhpcra_bits);
__IO_REG32_BIT(USB0_RXHPCRB3,           0x01E01870,__READ_WRITE ,__usb_rxhpcrb_bits);
__IO_REG32_BIT(USB0_DMA_SCHED_CTRL,     0x01E02000,__READ_WRITE ,__usb_dma_sched_ctrl_bits);
__IO_REG32_BIT(USB0_ENTRY0,             0x01E02800,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY1,             0x01E02804,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY2,             0x01E02808,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY3,             0x01E0280C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY4,             0x01E02810,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY5,             0x01E02814,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY6,             0x01E02818,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY7,             0x01E0281C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY8,             0x01E02820,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY9,             0x01E02824,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY10,            0x01E02828,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY11,            0x01E0282C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY12,            0x01E02830,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY13,            0x01E02834,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY14,            0x01E02838,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY15,            0x01E0283C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY16,            0x01E02840,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY17,            0x01E02844,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY18,            0x01E02848,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY19,            0x01E0284C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY20,            0x01E02850,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY21,            0x01E02854,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY22,            0x01E02858,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY23,            0x01E0285C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY24,            0x01E02860,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY25,            0x01E02864,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY26,            0x01E02868,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY27,            0x01E0286C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY28,            0x01E02870,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY29,            0x01E02874,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY30,            0x01E02878,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY31,            0x01E0287C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY32,            0x01E02880,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY33,            0x01E02884,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY34,            0x01E02888,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY35,            0x01E0288C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY36,            0x01E02890,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY37,            0x01E02894,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY38,            0x01E02898,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY39,            0x01E0289C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY40,            0x01E028A0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY41,            0x01E028A4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY42,            0x01E028A8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY43,            0x01E028AC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY44,            0x01E028B0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY45,            0x01E028B4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY46,            0x01E028B8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY47,            0x01E028BC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY48,            0x01E028C0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY49,            0x01E028C4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY50,            0x01E028C8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY51,            0x01E028CC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY52,            0x01E028D0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY53,            0x01E028D4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY54,            0x01E028D8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY55,            0x01E028DC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY56,            0x01E028E0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY57,            0x01E028E4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY58,            0x01E028E8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY59,            0x01E028EC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY60,            0x01E028F0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY61,            0x01E028F4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY62,            0x01E028F8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY63,            0x01E028FC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32(    USB0_QMGRREVID,          0x01E04000,__READ_WRITE );
__IO_REG32_BIT(USB0_DIVERSION,          0x01E04008,__WRITE      ,__usb_diversion_bits);
__IO_REG32_BIT(USB0_FDBSC0,             0x01E04020,__READ_WRITE ,__usb_fdbsc0_bits);
__IO_REG32_BIT(USB0_FDBSC1,             0x01E04024,__READ_WRITE ,__usb_fdbsc1_bits);
__IO_REG32_BIT(USB0_FDBSC2,             0x01E04028,__READ_WRITE ,__usb_fdbsc2_bits);
__IO_REG32_BIT(USB0_FDBSC3,             0x01E0402C,__READ_WRITE ,__usb_fdbsc3_bits);
__IO_REG32(    USB0_LRAM0BASE,          0x01E04080,__READ_WRITE );
__IO_REG32_BIT(USB0_LRAM0SIZE,          0x01E04084,__READ_WRITE ,__usb_lram0size_bits);
__IO_REG32(    USB0_LRAM1BASE,          0x01E04088,__READ_WRITE );
__IO_REG32(    USB0_PEND0,              0x01E04090,__READ       );
__IO_REG32(    USB0_PEND1,              0x01E04094,__READ       );
__IO_REG32(    USB0_QMEMRBASE0,         0x01E05000,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL0,         0x01E05004,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE1,         0x01E05010,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL1,         0x01E05014,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE2,         0x01E05020,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL2,         0x01E05024,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE3,         0x01E05030,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL3,         0x01E05034,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE4,         0x01E05040,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL4,         0x01E05044,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE5,         0x01E05050,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL5,         0x01E05054,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE6,         0x01E05060,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL6,         0x01E05064,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE7,         0x01E05070,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL7,         0x01E05074,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE8,         0x01E05080,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL8,         0x01E05084,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE9,         0x01E05090,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL9,         0x01E05094,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE10,        0x01E050A0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL10,        0x01E050A4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE11,        0x01E050B0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL11,        0x01E050B4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE12,        0x01E050C0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL12,        0x01E050C4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE13,        0x01E050D0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL13,        0x01E050D4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE14,        0x01E050E0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL14,        0x01E050E4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32(    USB0_QMEMRBASE15,        0x01E050F0,__READ_WRITE );
__IO_REG32_BIT(USB0_QMEMRCTRL15,        0x01E050F4,__READ_WRITE ,__usb_qmemrctrl_bits);
__IO_REG32_BIT(USB0_CTRLD0,             0x01E0600C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD1,             0x01E0601C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD2,             0x01E0602C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD3,             0x01E0603C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD4,             0x01E0604C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD5,             0x01E0605C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD6,             0x01E0606C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD7,             0x01E0607C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD8,             0x01E0608C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD9,             0x01E0609C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD10,            0x01E060AC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD11,            0x01E060BC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD12,            0x01E060CC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD13,            0x01E060DC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD14,            0x01E060EC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD15,            0x01E060FC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD16,            0x01E0610C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD17,            0x01E0611C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD18,            0x01E0612C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD19,            0x01E0613C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD20,            0x01E0614C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD21,            0x01E0615C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD22,            0x01E0616C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD23,            0x01E0617C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD24,            0x01E0618C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD25,            0x01E0619C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD26,            0x01E061AC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD27,            0x01E061BC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD28,            0x01E061CC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD29,            0x01E061DC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD30,            0x01E061EC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD31,            0x01E061FC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD32,            0x01E0620C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD33,            0x01E0621C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD34,            0x01E0622C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD35,            0x01E0623C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD36,            0x01E0624C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD37,            0x01E0625C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD38,            0x01E0626C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD39,            0x01E0627C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD40,            0x01E0628C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD41,            0x01E0629C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD42,            0x01E062AC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD43,            0x01E062BC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD44,            0x01E062CC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD45,            0x01E062DC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD46,            0x01E062EC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD47,            0x01E062FC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD48,            0x01E0630C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD49,            0x01E0631C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD50,            0x01E0632C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD51,            0x01E0633C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD52,            0x01E0634C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD53,            0x01E0635C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD54,            0x01E0636C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD55,            0x01E0637C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD56,            0x01E0638C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD57,            0x01E0639C,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD58,            0x01E063AC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD59,            0x01E063BC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD60,            0x01E063CC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD61,            0x01E063DC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD62,            0x01E063EC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_CTRLD63,            0x01E063FC,__READ_WRITE ,__usb_ctrld_bits);
__IO_REG32_BIT(USB0_QSTATA0,            0x01E06800,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB0,            0x01E06804,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC0,            0x01E06808,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA1,            0x01E06810,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB1,            0x01E06814,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC1,            0x01E06818,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA2,            0x01E06820,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB2,            0x01E06824,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC2,            0x01E06828,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA3,            0x01E06830,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB3,            0x01E06834,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC3,            0x01E06838,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA4,            0x01E06840,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB4,            0x01E06844,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC4,            0x01E06848,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA5,            0x01E06850,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB5,            0x01E06854,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC5,            0x01E06858,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA6,            0x01E06860,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB6,            0x01E06864,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC6,            0x01E06868,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA7,            0x01E06870,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB7,            0x01E06874,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC7,            0x01E06878,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA8,            0x01E06880,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB8,            0x01E06884,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC8,            0x01E06888,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA9,            0x01E06890,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB9,            0x01E06894,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC9,            0x01E06898,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA10,           0x01E068A0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB10,           0x01E068A4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC10,           0x01E068A8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA11,           0x01E068B0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB11,           0x01E068B4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC11,           0x01E068B8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA12,           0x01E068C0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB12,           0x01E068C4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC12,           0x01E068C8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA13,           0x01E068D0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB13,           0x01E068D4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC13,           0x01E068D8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA14,           0x01E068E0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB14,           0x01E068E4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC14,           0x01E068E8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA15,           0x01E068F0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB15,           0x01E068F4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC15,           0x01E068F8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA16,           0x01E06900,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB16,           0x01E06904,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC16,           0x01E06908,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA17,           0x01E06910,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB17,           0x01E06914,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC17,           0x01E06918,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA18,           0x01E06920,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB18,           0x01E06924,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC18,           0x01E06928,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA19,           0x01E06930,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB19,           0x01E06934,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC19,           0x01E06938,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA20,           0x01E06940,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB20,           0x01E06944,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC20,           0x01E06948,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA21,           0x01E06950,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB21,           0x01E06954,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC21,           0x01E06958,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA22,           0x01E06960,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB22,           0x01E06964,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC22,           0x01E06968,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA23,           0x01E06970,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB23,           0x01E06974,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC23,           0x01E06978,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA24,           0x01E06980,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB24,           0x01E06984,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC24,           0x01E06988,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA25,           0x01E06990,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB25,           0x01E06994,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC25,           0x01E06998,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA26,           0x01E069A0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB26,           0x01E069A4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC26,           0x01E069A8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA27,           0x01E069B0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB27,           0x01E069B4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC27,           0x01E069B8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA28,           0x01E069C0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB28,           0x01E069C4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC28,           0x01E069C8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA29,           0x01E069D0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB29,           0x01E069D4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC29,           0x01E069D8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA30,           0x01E069E0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB30,           0x01E069E4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC30,           0x01E069E8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA31,           0x01E069F0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB31,           0x01E069F4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC31,           0x01E069F8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA32,           0x01E06A00,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB32,           0x01E06A04,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC32,           0x01E06A08,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA33,           0x01E06A10,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB33,           0x01E06A14,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC33,           0x01E06A18,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA34,           0x01E06A20,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB34,           0x01E06A24,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC34,           0x01E06A28,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA35,           0x01E06A30,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB35,           0x01E06A34,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC35,           0x01E06A38,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA36,           0x01E06A40,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB36,           0x01E06A44,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC36,           0x01E06A48,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA37,           0x01E06A50,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB37,           0x01E06A54,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC37,           0x01E06A58,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA38,           0x01E06A60,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB38,           0x01E06A64,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC38,           0x01E06A68,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA39,           0x01E06A70,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB39,           0x01E06A74,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC39,           0x01E06A78,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA40,           0x01E06A80,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB40,           0x01E06A84,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC40,           0x01E06A88,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA41,           0x01E06A90,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB41,           0x01E06A94,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC41,           0x01E06A98,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA42,           0x01E06AA0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB42,           0x01E06AA4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC42,           0x01E06AA8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA43,           0x01E06AB0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB43,           0x01E06AB4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC43,           0x01E06AB8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA44,           0x01E06AC0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB44,           0x01E06AC4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC44,           0x01E06AC8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA45,           0x01E06AD0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB45,           0x01E06AD4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC45,           0x01E06AD8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA46,           0x01E06AE0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB46,           0x01E06AE4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC46,           0x01E06AE8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA47,           0x01E06AF0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB47,           0x01E06AF4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC47,           0x01E06AF8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA48,           0x01E06B00,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB48,           0x01E06B04,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC48,           0x01E06B08,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA49,           0x01E06B10,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB49,           0x01E06B14,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC49,           0x01E06B18,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA50,           0x01E06B20,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB50,           0x01E06B24,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC50,           0x01E06B28,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA51,           0x01E06B30,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB51,           0x01E06B34,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC51,           0x01E06B38,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA52,           0x01E06B40,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB52,           0x01E06B44,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC52,           0x01E06B48,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA53,           0x01E06B50,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB53,           0x01E06B54,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC53,           0x01E06B58,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA54,           0x01E06B60,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB54,           0x01E06B64,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC54,           0x01E06B68,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA55,           0x01E06B70,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB55,           0x01E06B74,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC55,           0x01E06B78,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA56,           0x01E06B80,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB56,           0x01E06B84,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC56,           0x01E06B88,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA57,           0x01E06B90,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB57,           0x01E06B94,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC57,           0x01E06B98,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA58,           0x01E06BA0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB58,           0x01E06BA4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC58,           0x01E06BA8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA59,           0x01E06BB0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB59,           0x01E06BB4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC59,           0x01E06BB8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA60,           0x01E06BC0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB60,           0x01E06BC4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC60,           0x01E06BC8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA61,           0x01E06BD0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB61,           0x01E06BD4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC61,           0x01E06BD8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA62,           0x01E06BE0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB62,           0x01E06BE4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC62,           0x01E06BE8,__READ       ,__usb_qstatc_bits);
__IO_REG32_BIT(USB0_QSTATA63,           0x01E06BF0,__READ       ,__usb_qstata_bits);
__IO_REG32_BIT(USB0_QSTATB63,           0x01E06BF4,__READ       ,__usb_qstatb_bits);
__IO_REG32_BIT(USB0_QSTATC63,           0x01E06BF8,__READ       ,__usb_qstatc_bits);

/***************************************************************************
 **
 ** EMAC
 **
 ***************************************************************************/
__IO_REG32(    EMAC_TXREVID,          0x01E23000,__READ       );
__IO_REG32_BIT(EMAC_TXCONTROL,        0x01E23004,__READ_WRITE ,__emac_txcontrol_bits);
__IO_REG32_BIT(EMAC_TXTEARDOWN,       0x01E23008,__READ_WRITE ,__emac_txteardown_bits);
__IO_REG32(    EMAC_RXREVID,          0x01E23010,__READ       );
__IO_REG32_BIT(EMAC_RXCONTROL,        0x01E23014,__READ_WRITE ,__emac_rxcontrol_bits);
__IO_REG32_BIT(EMAC_RXTEARDOWN,       0x01E23018,__READ_WRITE ,__emac_rxteardown_bits);
__IO_REG32_BIT(EMAC_TXINTSTATRAW,     0x01E23080,__READ       ,__emac_txintstatraw_bits);
__IO_REG32_BIT(EMAC_TXINTSTATMASKED,  0x01E23084,__READ       ,__emac_txintstatraw_bits);
__IO_REG32_BIT(EMAC_TXINTMASKSET,     0x01E23088,__READ_WRITE ,__emac_txintmaskset_bits);
__IO_REG32_BIT(EMAC_TXINTMASKCLEAR,   0x01E2308C,__READ_WRITE ,__emac_txintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINVECTOR,      0x01E23090,__READ       ,__emac_macinvector_bits);
__IO_REG32_BIT(EMAC_MACEOIVECTOR,     0x01E23094,__READ_WRITE ,__emac_maceoivector_bits);
__IO_REG32_BIT(EMAC_RXINTSTATRAW,     0x01E230A0,__READ       ,__emac_rxintstatraw_bits);
__IO_REG32_BIT(EMAC_RXINTSTATMASKED,  0x01E230A4,__READ       ,__emac_rxintstatraw_bits);
__IO_REG32_BIT(EMAC_RXINTMASKSET,     0x01E230A8,__READ_WRITE ,__emac_rxintmaskset_bits);
__IO_REG32_BIT(EMAC_RXINTMASKCLEAR,   0x01E230AC,__READ_WRITE ,__emac_rxintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINTSTATRAW,    0x01E230B0,__READ       ,__emac_macintstatraw_bits);
__IO_REG32_BIT(EMAC_MACINTSTATMASKED, 0x01E230B4,__READ       ,__emac_macintstatraw_bits);
__IO_REG32_BIT(EMAC_MACINTMASKSET,    0x01E230B8,__READ_WRITE ,__emac_macintmaskset_bits);
__IO_REG32_BIT(EMAC_MACINTMASKCLEAR,  0x01E230BC,__READ_WRITE ,__emac_macintmaskset_bits);
__IO_REG32_BIT(EMAC_RXMBPENABLE,      0x01E23100,__READ_WRITE ,__emac_rxmbpenable_bits);
__IO_REG32_BIT(EMAC_RXUNICASTSET,     0x01E23104,__READ_WRITE ,__emac_rxunicastset_bits);
__IO_REG32_BIT(EMAC_RXUNICASTCLEAR,   0x01E23108,__READ_WRITE ,__emac_rxunicastset_bits);
__IO_REG32_BIT(EMAC_RXMAXLEN,         0x01E2310C,__READ_WRITE ,__emac_rxmaxlen_bits);
__IO_REG32_BIT(EMAC_RXBUFFEROFFSET,   0x01E23110,__READ_WRITE ,__emac_rxbufferoffset_bits);
__IO_REG32_BIT(EMAC_RXFILTERLOWTHRESH,0x01E23114,__READ_WRITE ,__emac_rxfilterlowthresh_bits);
__IO_REG32_BIT(EMAC_RX0FLOWTHRESH,    0x01E23120,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX1FLOWTHRESH,    0x01E23124,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX2FLOWTHRESH,    0x01E23128,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX3FLOWTHRESH,    0x01E2312C,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX4FLOWTHRESH,    0x01E23130,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX5FLOWTHRESH,    0x01E23134,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX6FLOWTHRESH,    0x01E23138,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX7FLOWTHRESH,    0x01E2313C,__READ_WRITE ,__emac_rxflowthresh_bits);
__IO_REG32_BIT(EMAC_RX0FREEBUFFER,    0x01E23140,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX1FREEBUFFER,    0x01E23144,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX2FREEBUFFER,    0x01E23148,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX3FREEBUFFER,    0x01E2314C,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX4FREEBUFFER,    0x01E23150,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX5FREEBUFFER,    0x01E23154,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX6FREEBUFFER,    0x01E23158,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_RX7FREEBUFFER,    0x01E2315C,__READ_WRITE ,__emac_rxfreebuffer_bits);
__IO_REG32_BIT(EMAC_MACCONTROL,       0x01E23160,__READ_WRITE ,__emac_maccontrol_bits);
__IO_REG32_BIT(EMAC_MACSTATUS,        0x01E23164,__READ       ,__emac_macstatus_bits);
__IO_REG32_BIT(EMAC_EMCONTROL,        0x01E23168,__READ_WRITE ,__emac_emcontrol_bits);
__IO_REG32_BIT(EMAC_FIFOCONTROL,      0x01E2316C,__READ_WRITE ,__emac_fifocontrol_bits);
__IO_REG32_BIT(EMAC_MACCONFIG,        0x01E23170,__READ       ,__emac_macconfig_bits);
__IO_REG32_BIT(EMAC_MSOFTRESET,       0x01E23174,__READ_WRITE ,__emac_msoftreset_bits);
__IO_REG32_BIT(EMAC_MACSRCADDRLO,     0x01E231D0,__READ_WRITE ,__emac_macsrcaddrlo_bits);
__IO_REG32_BIT(EMAC_MACSRCADDRHI,     0x01E231D4,__READ_WRITE ,__emac_macsrcaddrhi_bits);
__IO_REG32(    EMAC_MACHASH1,         0x01E231D8,__READ_WRITE );
__IO_REG32(    EMAC_MACHASH2,         0x01E231DC,__READ_WRITE );
__IO_REG32_BIT(EMAC_BOFFTEST,         0x01E231E0,__READ       ,__emac_bofftest_bits);
__IO_REG32_BIT(EMAC_TPACETEST,        0x01E231E4,__READ       ,__emac_tpacetest_bits);
__IO_REG32_BIT(EMAC_RXPAUSE,          0x01E231E8,__READ       ,__emac_rxpause_bits);
__IO_REG32_BIT(EMAC_TXPAUSE,          0x01E231EC,__READ       ,__emac_rxpause_bits);
__IO_REG32_BIT(EMAC_MACADDRLO,        0x01E23500,__READ_WRITE ,__emac_macaddrlo_bits);
__IO_REG32_BIT(EMAC_MACADDRHI,        0x01E23504,__READ_WRITE ,__emac_macaddrhi_bits);
__IO_REG32_BIT(EMAC_MACINDEX,         0x01E23508,__READ_WRITE ,__emac_macindex_bits);
__IO_REG32(    EMAC_TX0HDP,           0x01E23600,__READ_WRITE );
__IO_REG32(    EMAC_TX1HDP,           0x01E23604,__READ_WRITE );
__IO_REG32(    EMAC_TX2HDP,           0x01E23608,__READ_WRITE );
__IO_REG32(    EMAC_TX3HDP,           0x01E2360C,__READ_WRITE );
__IO_REG32(    EMAC_TX4HDP,           0x01E23610,__READ_WRITE );
__IO_REG32(    EMAC_TX5HDP,           0x01E23614,__READ_WRITE );
__IO_REG32(    EMAC_TX6HDP,           0x01E23618,__READ_WRITE );
__IO_REG32(    EMAC_TX7HDP,           0x01E2361C,__READ_WRITE );
__IO_REG32(    EMAC_RX0HDP,           0x01E23620,__READ_WRITE );
__IO_REG32(    EMAC_RX1HDP,           0x01E23624,__READ_WRITE );
__IO_REG32(    EMAC_RX2HDP,           0x01E23628,__READ_WRITE );
__IO_REG32(    EMAC_RX3HDP,           0x01E2362C,__READ_WRITE );
__IO_REG32(    EMAC_RX4HDP,           0x01E23630,__READ_WRITE );
__IO_REG32(    EMAC_RX5HDP,           0x01E23634,__READ_WRITE );
__IO_REG32(    EMAC_RX6HDP,           0x01E23638,__READ_WRITE );
__IO_REG32(    EMAC_RX7HDP,           0x01E2363C,__READ_WRITE );
__IO_REG32(    EMAC_TX0CP,            0x01E23640,__READ_WRITE );
__IO_REG32(    EMAC_TX1CP,            0x01E23644,__READ_WRITE );
__IO_REG32(    EMAC_TX2CP,            0x01E23648,__READ_WRITE );
__IO_REG32(    EMAC_TX3CP,            0x01E2364C,__READ_WRITE );
__IO_REG32(    EMAC_TX4CP,            0x01E23650,__READ_WRITE );
__IO_REG32(    EMAC_TX5CP,            0x01E23654,__READ_WRITE );
__IO_REG32(    EMAC_TX6CP,            0x01E23658,__READ_WRITE );
__IO_REG32(    EMAC_TX7CP,            0x01E2365C,__READ_WRITE );
__IO_REG32(    EMAC_RX0CP,            0x01E23660,__READ_WRITE );
__IO_REG32(    EMAC_RX1CP,            0x01E23664,__READ_WRITE );
__IO_REG32(    EMAC_RX2CP,            0x01E23668,__READ_WRITE );
__IO_REG32(    EMAC_RX3CP,            0x01E2366C,__READ_WRITE );
__IO_REG32(    EMAC_RX4CP,            0x01E23670,__READ_WRITE );
__IO_REG32(    EMAC_RX5CP,            0x01E23674,__READ_WRITE );
__IO_REG32(    EMAC_RX6CP,            0x01E23678,__READ_WRITE );
__IO_REG32(    EMAC_RX7CP,            0x01E2367C,__READ_WRITE );
__IO_REG32(    EMAC_RXGOODFRAMES,     0x01E23200,__READ_WRITE );
__IO_REG32(    EMAC_RXBCASTFRAMES,    0x01E23204,__READ_WRITE );
__IO_REG32(    EMAC_RXMCASTFRAMES,    0x01E23208,__READ_WRITE );
__IO_REG32(    EMAC_RXPAUSEFRAMES,    0x01E2320C,__READ_WRITE );
__IO_REG32(    EMAC_RXCRCERRORS,      0x01E23210,__READ_WRITE );
__IO_REG32(    EMAC_RXALIGNCODEERRORS,0x01E23214,__READ_WRITE );
__IO_REG32(    EMAC_RXOVERSIZED,      0x01E23218,__READ_WRITE );
__IO_REG32(    EMAC_RXJABBER,         0x01E2321C,__READ_WRITE );
__IO_REG32(    EMAC_RXUNDERSIZED,     0x01E23220,__READ_WRITE );
__IO_REG32(    EMAC_RXFRAGMENTS,      0x01E23224,__READ_WRITE );
__IO_REG32(    EMAC_RXFILTERED,       0x01E23228,__READ_WRITE );
__IO_REG32(    EMAC_RXQOSFILTERED,    0x01E2322C,__READ_WRITE );
__IO_REG32(    EMAC_RXOCTETS,         0x01E23230,__READ_WRITE );
__IO_REG32(    EMAC_TXGOODFRAMES,     0x01E23234,__READ_WRITE );
__IO_REG32(    EMAC_TXBCASTFRAMES,    0x01E23238,__READ_WRITE );
__IO_REG32(    EMAC_TXMCASTFRAMES,    0x01E2323C,__READ_WRITE );
__IO_REG32(    EMAC_TXPAUSEFRAMES,    0x01E23240,__READ_WRITE );
__IO_REG32(    EMAC_TXDEFERRED,       0x01E23244,__READ_WRITE );
__IO_REG32(    EMAC_TXCOLLISION,      0x01E23248,__READ_WRITE );
__IO_REG32(    EMAC_TXSINGLECOLL,     0x01E2324C,__READ_WRITE );
__IO_REG32(    EMAC_TXMULTICOLL,      0x01E23250,__READ_WRITE );
__IO_REG32(    EMAC_TXEXCESSIVECOLL,  0x01E23254,__READ_WRITE );
__IO_REG32(    EMAC_TXLATECOLL,       0x01E23258,__READ_WRITE );
__IO_REG32(    EMAC_TXUNDERRUN,       0x01E2325C,__READ_WRITE );
__IO_REG32(    EMAC_TXCARRIERSENSE,   0x01E23260,__READ_WRITE );
__IO_REG32(    EMAC_TXOCTETS,         0x01E23264,__READ_WRITE );
__IO_REG32(    EMAC_FRAME64,          0x01E23268,__READ_WRITE );
__IO_REG32(    EMAC_FRAME65T127,      0x01E2326C,__READ_WRITE );
__IO_REG32(    EMAC_FRAME128T255,     0x01E23270,__READ_WRITE );
__IO_REG32(    EMAC_FRAME256T511,     0x01E23274,__READ_WRITE );
__IO_REG32(    EMAC_FRAME512T1023,    0x01E23278,__READ_WRITE );
__IO_REG32(    EMAC_FRAME1024TUP,     0x01E2327C,__READ_WRITE );
__IO_REG32(    EMAC_NETOCTETS,        0x01E23280,__READ_WRITE );
__IO_REG32(    EMAC_RXSOFOVERRUNS,    0x01E23284,__READ_WRITE );
__IO_REG32(    EMAC_RXMOFOVERRUNS,    0x01E23288,__READ_WRITE );
__IO_REG32(    EMAC_RXDMAOVERRUNS,    0x01E2328C,__READ_WRITE );
__IO_REG32(    EMAC_REVID,            0x01E22000,__READ       );
__IO_REG32_BIT(EMAC_SOFTRESET,        0x01E22004,__READ_WRITE ,__emac_softreset_bits);
__IO_REG32_BIT(EMAC_INTCONTROL,       0x01E2200C,__READ_WRITE ,__emac_intcontrol_bits);
__IO_REG32_BIT(EMAC_C0RXTHRESHEN,     0x01E22010,__READ_WRITE ,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C0RXEN,           0x01E22014,__READ_WRITE ,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C0TXEN,           0x01E22018,__READ_WRITE ,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C0MISCEN,         0x01E2201C,__READ_WRITE ,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C1RXTHRESHEN,     0x01E22020,__READ_WRITE ,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C1RXEN,           0x01E22024,__READ_WRITE ,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C1TXEN,           0x01E22028,__READ_WRITE ,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C1MISCEN,         0x01E2202C,__READ_WRITE ,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C2RXTHRESHEN,     0x01E22030,__READ_WRITE ,__emac_crxthreshen_bits);
__IO_REG32_BIT(EMAC_C2RXEN,           0x01E22034,__READ_WRITE ,__emac_crxen_bits);
__IO_REG32_BIT(EMAC_C2TXEN,           0x01E22038,__READ_WRITE ,__emac_ctxen_bits);
__IO_REG32_BIT(EMAC_C2MISCEN,         0x01E2203C,__READ_WRITE ,__emac_cmiscen_bits);
__IO_REG32_BIT(EMAC_C0RXTHRESHSTAT,   0x01E22040,__READ       ,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C0RXSTAT,         0x01E22044,__READ       ,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C0TXSTAT,         0x01E22048,__READ       ,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C0MISCSTAT,       0x01E2204C,__READ       ,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C1RXTHRESHSTAT,   0x01E22050,__READ       ,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C1RXSTAT,         0x01E22054,__READ       ,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C1TXSTAT,         0x01E22058,__READ       ,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C1MISCSTAT,       0x01E2205C,__READ       ,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C2RXTHRESHSTAT,   0x01E22060,__READ       ,__emac_crxthreshstat_bits);
__IO_REG32_BIT(EMAC_C2RXSTAT,         0x01E22064,__READ       ,__emac_crxstat_bits);
__IO_REG32_BIT(EMAC_C2TXSTAT,         0x01E22068,__READ       ,__emac_ctxstat_bits);
__IO_REG32_BIT(EMAC_C2MISCSTAT,       0x01E2206C,__READ       ,__emac_cmiscstat_bits);
__IO_REG32_BIT(EMAC_C0RXIMAX,         0x01E22070,__READ_WRITE ,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C0TXIMAX,         0x01E22074,__READ_WRITE ,__emac_ctximax_bits);
__IO_REG32_BIT(EMAC_C1RXIMAX,         0x01E22078,__READ_WRITE ,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C1TXIMAX,         0x01E2207C,__READ_WRITE ,__emac_ctximax_bits);
__IO_REG32_BIT(EMAC_C2RXIMAX,         0x01E22080,__READ_WRITE ,__emac_crximax_bits);
__IO_REG32_BIT(EMAC_C2TXIMAX,         0x01E22084,__READ_WRITE ,__emac_ctximax_bits);
__IO_REG32(    EMAC_DESC_BASE_ADDR,   0x01E20000,__READ_WRITE );

/***************************************************************************
 **
 ** MDIO
 **
 ***************************************************************************/
__IO_REG32(    MDIO_REV,              0x01E24000,__READ       );
__IO_REG32_BIT(MDIO_CONTROL,          0x01E24004,__READ_WRITE ,__mdio_control_bits);
__IO_REG32_BIT(MDIO_ALIVE,            0x01E24008,__READ_WRITE ,__mdio_alive_bits);
__IO_REG32_BIT(MDIO_LINK,             0x01E2400C,__READ_WRITE ,__mdio_link_bits);
__IO_REG32_BIT(MDIO_LINKINTRAW,       0x01E24010,__READ_WRITE ,__mdio_linkintraw_bits);
__IO_REG32_BIT(MDIO_LINKINTMASKED,    0x01E24014,__READ_WRITE ,__mdio_linkintraw_bits);
__IO_REG32_BIT(MDIO_USERINTRAW,       0x01E24020,__READ_WRITE ,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKED,    0x01E24024,__READ_WRITE ,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKSET,   0x01E24028,__READ_WRITE ,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERINTMASKCLEAR, 0x01E2402C,__READ_WRITE ,__mdio_userintraw_bits);
__IO_REG32_BIT(MDIO_USERACCESS0,      0x01E24080,__READ_WRITE ,__mdio_useraccess_bits);
__IO_REG32_BIT(MDIO_USERPHYSEL0,      0x01E24084,__READ_WRITE ,__mdio_userphysel_bits);
__IO_REG32_BIT(MDIO_USERACCESS1,      0x01E24088,__READ_WRITE ,__mdio_useraccess_bits);
__IO_REG32_BIT(MDIO_USERPHYSEL1,      0x01E2408C,__READ_WRITE ,__mdio_userphysel_bits);

/***************************************************************************
 **
 ** ECAP0
 **
 ***************************************************************************/
__IO_REG32(    ECAP0_TSCTR,           0x01F06000,__READ_WRITE );
__IO_REG32(    ECAP0_CTRPHS,          0x01F06004,__READ_WRITE );
__IO_REG32(    ECAP0_CAP1,            0x01F06008,__READ_WRITE );
__IO_REG32(    ECAP0_CAP2,            0x01F0600C,__READ_WRITE );
__IO_REG32(    ECAP0_CAP3,            0x01F06010,__READ_WRITE );
__IO_REG32(    ECAP0_CAP4,            0x01F06014,__READ_WRITE );
__IO_REG16_BIT(ECAP0_ECCTL1,          0x01F06028,__READ_WRITE ,__ecap_ecctl1_bits);
__IO_REG16_BIT(ECAP0_ECCTL2,          0x01F0602A,__READ_WRITE ,__ecap_ecctl2_bits);
__IO_REG16_BIT(ECAP0_ECEINT,          0x01F0602C,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP0_ECFLG,           0x01F0602E,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP0_ECCLR,           0x01F06030,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP0_ECFRC,           0x01F06032,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG32(    ECAP0_REVID,           0x01F0605C,__READ       );

/***************************************************************************
 **
 ** ECAP1
 **
 ***************************************************************************/
__IO_REG32(    ECAP1_TSCTR,           0x01F07000,__READ_WRITE );
__IO_REG32(    ECAP1_CTRPHS,          0x01F07004,__READ_WRITE );
__IO_REG32(    ECAP1_CAP1,            0x01F07008,__READ_WRITE );
__IO_REG32(    ECAP1_CAP2,            0x01F0700C,__READ_WRITE );
__IO_REG32(    ECAP1_CAP3,            0x01F07010,__READ_WRITE );
__IO_REG32(    ECAP1_CAP4,            0x01F07014,__READ_WRITE );
__IO_REG16_BIT(ECAP1_ECCTL1,          0x01F07028,__READ_WRITE ,__ecap_ecctl1_bits);
__IO_REG16_BIT(ECAP1_ECCTL2,          0x01F0702A,__READ_WRITE ,__ecap_ecctl2_bits);
__IO_REG16_BIT(ECAP1_ECEINT,          0x01F0702C,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP1_ECFLG,           0x01F0702E,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP1_ECCLR,           0x01F07030,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP1_ECFRC,           0x01F07032,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG32(    ECAP1_REVID,           0x01F0705C,__READ       );

/***************************************************************************
 **
 ** ECAP2
 **
 ***************************************************************************/
__IO_REG32(    ECAP2_TSCTR,           0x01F08000,__READ_WRITE );
__IO_REG32(    ECAP2_CTRPHS,          0x01F08004,__READ_WRITE );
__IO_REG32(    ECAP2_CAP1,            0x01F08008,__READ_WRITE );
__IO_REG32(    ECAP2_CAP2,            0x01F0800C,__READ_WRITE );
__IO_REG32(    ECAP2_CAP3,            0x01F08010,__READ_WRITE );
__IO_REG32(    ECAP2_CAP4,            0x01F08014,__READ_WRITE );
__IO_REG16_BIT(ECAP2_ECCTL1,          0x01F08028,__READ_WRITE ,__ecap_ecctl1_bits);
__IO_REG16_BIT(ECAP2_ECCTL2,          0x01F0802A,__READ_WRITE ,__ecap_ecctl2_bits);
__IO_REG16_BIT(ECAP2_ECEINT,          0x01F0802C,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP2_ECFLG,           0x01F0802E,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP2_ECCLR,           0x01F08030,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG16_BIT(ECAP2_ECFRC,           0x01F08032,__READ_WRITE ,__ecap_eceint_bits);
__IO_REG32(    ECAP2_REVID,           0x01F0805C,__READ       );

/***************************************************************************
 **
 ** EQEP0
 **
 ***************************************************************************/
__IO_REG32(    EQEP0_QPOSCNT,         0x01F09000,__READ_WRITE );
__IO_REG32(    EQEP0_QPOSINIT,        0x01F09004,__READ_WRITE );
__IO_REG32(    EQEP0_QPOSMAX,         0x01F09008,__READ_WRITE );
__IO_REG32(    EQEP0_QPOSCMP,         0x01F0900C,__READ_WRITE );
__IO_REG32(    EQEP0_QPOSILAT,        0x01F09010,__READ       );
__IO_REG32(    EQEP0_QPOSSLAT,        0x01F09014,__READ       );
__IO_REG32(    EQEP0_QPOSLAT,         0x01F09018,__READ       );
__IO_REG32(    EQEP0_QUTMR,           0x01F0901C,__READ_WRITE );
__IO_REG32(    EQEP0_QUPRD,           0x01F09020,__READ_WRITE );
__IO_REG16(    EQEP0_QWDTMR,          0x01F09024,__READ_WRITE );
__IO_REG16(    EQEP0_QWDPRD,          0x01F09026,__READ_WRITE );
__IO_REG16_BIT(EQEP0_QDECCTL,         0x01F09028,__READ_WRITE ,__eqep_qdecctl_bits);
__IO_REG16_BIT(EQEP0_QEPCTL,          0x01F0902A,__READ_WRITE ,__eqep_qepctl_bits);
__IO_REG16_BIT(EQEP0_QCAPCTL,         0x01F0902C,__READ_WRITE ,__eqep_qcapctl_bits);
__IO_REG16_BIT(EQEP0_QPOSCTL,         0x01F0902E,__READ_WRITE ,__eqep_qposctl_bits);
__IO_REG16_BIT(EQEP0_QEINT,           0x01F09030,__READ_WRITE ,__eqep_qeint_bits);
__IO_REG16_BIT(EQEP0_QFLG,            0x01F09032,__READ       ,__eqep_qflg_bits);
__IO_REG16_BIT(EQEP0_QCLR,            0x01F09034,__READ_WRITE ,__eqep_qflg_bits);
__IO_REG16_BIT(EQEP0_QFRC,            0x01F09036,__READ_WRITE ,__eqep_qeint_bits);
__IO_REG16_BIT(EQEP0_QEPSTS,          0x01F09038,__READ_WRITE ,__eqep_qepsts_bits);
__IO_REG16(    EQEP0_QCTMR,           0x01F0903A,__READ_WRITE );
__IO_REG16(    EQEP0_QCPRD,           0x01F0903C,__READ_WRITE );
__IO_REG16(    EQEP0_QCTMRLAT,        0x01F0903E,__READ       );
__IO_REG16(    EQEP0_QCPRDLAT,        0x01F09040,__READ_WRITE );
__IO_REG16(    EQEP0_REVID,           0x01F0905C,__READ       );

/***************************************************************************
 **
 ** EQEP1
 **
 ***************************************************************************/
__IO_REG32(    EQEP1_QPOSCNT,         0x01F0A000,__READ_WRITE );
__IO_REG32(    EQEP1_QPOSINIT,        0x01F0A004,__READ_WRITE );
__IO_REG32(    EQEP1_QPOSMAX,         0x01F0A008,__READ_WRITE );
__IO_REG32(    EQEP1_QPOSCMP,         0x01F0A00C,__READ_WRITE );
__IO_REG32(    EQEP1_QPOSILAT,        0x01F0A010,__READ       );
__IO_REG32(    EQEP1_QPOSSLAT,        0x01F0A014,__READ       );
__IO_REG32(    EQEP1_QPOSLAT,         0x01F0A018,__READ       );
__IO_REG32(    EQEP1_QUTMR,           0x01F0A01C,__READ_WRITE );
__IO_REG32(    EQEP1_QUPRD,           0x01F0A020,__READ_WRITE );
__IO_REG16(    EQEP1_QWDTMR,          0x01F0A024,__READ_WRITE );
__IO_REG16(    EQEP1_QWDPRD,          0x01F0A026,__READ_WRITE );
__IO_REG16_BIT(EQEP1_QDECCTL,         0x01F0A028,__READ_WRITE ,__eqep_qdecctl_bits);
__IO_REG16_BIT(EQEP1_QEPCTL,          0x01F0A02A,__READ_WRITE ,__eqep_qepctl_bits);
__IO_REG16_BIT(EQEP1_QCAPCTL,         0x01F0A02C,__READ_WRITE ,__eqep_qcapctl_bits);
__IO_REG16_BIT(EQEP1_QPOSCTL,         0x01F0A02E,__READ_WRITE ,__eqep_qposctl_bits);
__IO_REG16_BIT(EQEP1_QEINT,           0x01F0A030,__READ_WRITE ,__eqep_qeint_bits);
__IO_REG16_BIT(EQEP1_QFLG,            0x01F0A032,__READ       ,__eqep_qflg_bits);
__IO_REG16_BIT(EQEP1_QCLR,            0x01F0A034,__READ_WRITE ,__eqep_qflg_bits);
__IO_REG16_BIT(EQEP1_QFRC,            0x01F0A036,__READ_WRITE ,__eqep_qeint_bits);
__IO_REG16_BIT(EQEP1_QEPSTS,          0x01F0A038,__READ_WRITE ,__eqep_qepsts_bits);
__IO_REG16(    EQEP1_QCTMR,           0x01F0A03A,__READ_WRITE );
__IO_REG16(    EQEP1_QCPRD,           0x01F0A03C,__READ_WRITE );
__IO_REG16(    EQEP1_QCTMRLAT,        0x01F0A03E,__READ       );
__IO_REG16(    EQEP1_QCPRDLAT,        0x01F0A040,__READ_WRITE );
__IO_REG16(    EQEP1_REVID,           0x01F0A05C,__READ       );

/***************************************************************************
 **
 ** eHRPWM0
 **
 ***************************************************************************/
__IO_REG16_BIT(eHRPWM0_TBCTL,         0x01F00000,__READ_WRITE ,__ehrpwm_tbctl_bits);
__IO_REG16_BIT(eHRPWM0_TBSTS,         0x01F00002,__READ_WRITE ,__ehrpwm_tbsts_bits);
__IO_REG16_BIT(eHRPWM0_TBPHSHR,       0x01F00004,__READ_WRITE ,__ehrpwm_tbphshr_bits);
__IO_REG16(    eHRPWM0_TBPHS,         0x01F00006,__READ_WRITE );
__IO_REG16(    eHRPWM0_TBCNT,         0x01F00008,__READ_WRITE );
__IO_REG16(    eHRPWM0_TBPRD,         0x01F0000A,__READ_WRITE );
__IO_REG16_BIT(eHRPWM0_CMPCTL,        0x01F0000E,__READ_WRITE ,__ehrpwm_cmpctl_bits);
__IO_REG16_BIT(eHRPWM0_CMPAHR,        0x01F00010,__READ_WRITE ,__ehrpwm_cmpahr_bits);
__IO_REG16(    eHRPWM0_CMPA,          0x01F00012,__READ_WRITE );
__IO_REG16(    eHRPWM0_CMPB,          0x01F00014,__READ_WRITE );
__IO_REG16_BIT(eHRPWM0_AQCTLA,        0x01F00016,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM0_AQCTLB,        0x01F00018,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM0_AQSFRC,        0x01F0001A,__READ_WRITE ,__ehrpwm_aqsfrc_bits);
__IO_REG16_BIT(eHRPWM0_AQCSFRC,       0x01F0001C,__READ_WRITE ,__ehrpwm_aqcsfrc_bits);
__IO_REG16_BIT(eHRPWM0_DBCTL,         0x01F0001E,__READ_WRITE ,__ehrpwm_dbctl_bits);
__IO_REG16_BIT(eHRPWM0_DBRED,         0x01F00020,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM0_DBFED,         0x01F00022,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM0_TZSEL,         0x01F00024,__READ_WRITE ,__ehrpwm_tzsel_bits);
__IO_REG16_BIT(eHRPWM0_TZCTL,         0x01F00028,__READ_WRITE ,__ehrpwm_tzctl_bits);
__IO_REG16_BIT(eHRPWM0_TZEINT,        0x01F0002A,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM0_TZFLG,         0x01F0002C,__READ       ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM0_TZCLR,         0x01F0002E,__READ_WRITE ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM0_TZFRC,         0x01F00030,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM0_ETSEL,         0x01F00032,__READ_WRITE ,__ehrpwm_etsel_bits);
__IO_REG16_BIT(eHRPWM0_ETPS,          0x01F00034,__READ_WRITE ,__ehrpwm_etps_bits);
__IO_REG16_BIT(eHRPWM0_ETFLG,         0x01F00036,__READ       ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM0_ETCLR,         0x01F00038,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM0_ETFRC,         0x01F0003A,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM0_PCCTL,         0x01F0003C,__READ_WRITE ,__ehrpwm_pcctl_bits);
__IO_REG16_BIT(eHRPWM0_HRCNFG,        0x01F01020,__READ_WRITE ,__ehrpwm_hrcnfg_bits);

/***************************************************************************
 **
 ** eHRPWM1
 **
 ***************************************************************************/
__IO_REG16_BIT(eHRPWM1_TBCTL,         0x01F02000,__READ_WRITE ,__ehrpwm_tbctl_bits);
__IO_REG16_BIT(eHRPWM1_TBSTS,         0x01F02002,__READ_WRITE ,__ehrpwm_tbsts_bits);
__IO_REG16_BIT(eHRPWM1_TBPHSHR,       0x01F02004,__READ_WRITE ,__ehrpwm_tbphshr_bits);
__IO_REG16(    eHRPWM1_TBPHS,         0x01F02006,__READ_WRITE );
__IO_REG16(    eHRPWM1_TBCNT,         0x01F02008,__READ_WRITE );
__IO_REG16(    eHRPWM1_TBPRD,         0x01F0200A,__READ_WRITE );
__IO_REG16_BIT(eHRPWM1_CMPCTL,        0x01F0200E,__READ_WRITE ,__ehrpwm_cmpctl_bits);
__IO_REG16_BIT(eHRPWM1_CMPAHR,        0x01F02010,__READ_WRITE ,__ehrpwm_cmpahr_bits);
__IO_REG16(    eHRPWM1_CMPA,          0x01F02012,__READ_WRITE );
__IO_REG16(    eHRPWM1_CMPB,          0x01F02014,__READ_WRITE );
__IO_REG16_BIT(eHRPWM1_AQCTLA,        0x01F02016,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM1_AQCTLB,        0x01F02018,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM1_AQSFRC,        0x01F0201A,__READ_WRITE ,__ehrpwm_aqsfrc_bits);
__IO_REG16_BIT(eHRPWM1_AQCSFRC,       0x01F0201C,__READ_WRITE ,__ehrpwm_aqcsfrc_bits);
__IO_REG16_BIT(eHRPWM1_DBCTL,         0x01F0201E,__READ_WRITE ,__ehrpwm_dbctl_bits);
__IO_REG16_BIT(eHRPWM1_DBRED,         0x01F02020,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM1_DBFED,         0x01F02022,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM1_TZSEL,         0x01F02024,__READ_WRITE ,__ehrpwm_tzsel_bits);
__IO_REG16_BIT(eHRPWM1_TZCTL,         0x01F02028,__READ_WRITE ,__ehrpwm_tzctl_bits);
__IO_REG16_BIT(eHRPWM1_TZEINT,        0x01F0202A,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM1_TZFLG,         0x01F0202C,__READ       ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM1_TZCLR,         0x01F0202E,__READ_WRITE ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM1_TZFRC,         0x01F02030,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM1_ETSEL,         0x01F02032,__READ_WRITE ,__ehrpwm_etsel_bits);
__IO_REG16_BIT(eHRPWM1_ETPS,          0x01F02034,__READ_WRITE ,__ehrpwm_etps_bits);
__IO_REG16_BIT(eHRPWM1_ETFLG,         0x01F02036,__READ       ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM1_ETCLR,         0x01F02038,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM1_ETFRC,         0x01F0203A,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM1_PCCTL,         0x01F0203C,__READ_WRITE ,__ehrpwm_pcctl_bits);
__IO_REG16_BIT(eHRPWM1_HRCNFG,        0x01F03020,__READ_WRITE ,__ehrpwm_hrcnfg_bits);

/***************************************************************************
 **
 ** eHRPWM2
 **
 ***************************************************************************/
__IO_REG16_BIT(eHRPWM2_TBCTL,         0x01F04000,__READ_WRITE ,__ehrpwm_tbctl_bits);
__IO_REG16_BIT(eHRPWM2_TBSTS,         0x01F04002,__READ_WRITE ,__ehrpwm_tbsts_bits);
__IO_REG16_BIT(eHRPWM2_TBPHSHR,       0x01F04004,__READ_WRITE ,__ehrpwm_tbphshr_bits);
__IO_REG16(    eHRPWM2_TBPHS,         0x01F04006,__READ_WRITE );
__IO_REG16(    eHRPWM2_TBCNT,         0x01F04008,__READ_WRITE );
__IO_REG16(    eHRPWM2_TBPRD,         0x01F0400A,__READ_WRITE );
__IO_REG16_BIT(eHRPWM2_CMPCTL,        0x01F0400E,__READ_WRITE ,__ehrpwm_cmpctl_bits);
__IO_REG16_BIT(eHRPWM2_CMPAHR,        0x01F04010,__READ_WRITE ,__ehrpwm_cmpahr_bits);
__IO_REG16(    eHRPWM2_CMPA,          0x01F04012,__READ_WRITE );
__IO_REG16(    eHRPWM2_CMPB,          0x01F04014,__READ_WRITE );
__IO_REG16_BIT(eHRPWM2_AQCTLA,        0x01F04016,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM2_AQCTLB,        0x01F04018,__READ_WRITE ,__ehrpwm_aqctla_bits);
__IO_REG16_BIT(eHRPWM2_AQSFRC,        0x01F0401A,__READ_WRITE ,__ehrpwm_aqsfrc_bits);
__IO_REG16_BIT(eHRPWM2_AQCSFRC,       0x01F0401C,__READ_WRITE ,__ehrpwm_aqcsfrc_bits);
__IO_REG16_BIT(eHRPWM2_DBCTL,         0x01F0401E,__READ_WRITE ,__ehrpwm_dbctl_bits);
__IO_REG16_BIT(eHRPWM2_DBRED,         0x01F04020,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM2_DBFED,         0x01F04022,__READ_WRITE ,__ehrpwm_dbred_bits);
__IO_REG16_BIT(eHRPWM2_TZSEL,         0x01F04024,__READ_WRITE ,__ehrpwm_tzsel_bits);
__IO_REG16_BIT(eHRPWM2_TZCTL,         0x01F04028,__READ_WRITE ,__ehrpwm_tzctl_bits);
__IO_REG16_BIT(eHRPWM2_TZEINT,        0x01F0402A,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM2_TZFLG,         0x01F0402C,__READ       ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM2_TZCLR,         0x01F0402E,__READ_WRITE ,__ehrpwm_tzflg_bits);
__IO_REG16_BIT(eHRPWM2_TZFRC,         0x01F04030,__READ_WRITE ,__ehrpwm_tzeint_bits);
__IO_REG16_BIT(eHRPWM2_ETSEL,         0x01F04032,__READ_WRITE ,__ehrpwm_etsel_bits);
__IO_REG16_BIT(eHRPWM2_ETPS,          0x01F04034,__READ_WRITE ,__ehrpwm_etps_bits);
__IO_REG16_BIT(eHRPWM2_ETFLG,         0x01F04036,__READ       ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM2_ETCLR,         0x01F04038,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM2_ETFRC,         0x01F0403A,__READ_WRITE ,__ehrpwm_etflg_bits);
__IO_REG16_BIT(eHRPWM2_PCCTL,         0x01F0403C,__READ_WRITE ,__ehrpwm_pcctl_bits);
__IO_REG16_BIT(eHRPWM2_HRCNFG,        0x01F05020,__READ_WRITE ,__ehrpwm_hrcnfg_bits);

/***************************************************************************
 **
 ** TIMER64P0
 **
 ***************************************************************************/
__IO_REG32(    TIMER64P0_REVID,       0x01C20000,__READ       );
__IO_REG32_BIT(TIMER64P0_EMUMGT,      0x01C20004,__READ_WRITE ,__timer64p_emumgt_bits);
__IO_REG32_BIT(TIMER64P0_GPINTGPEN,   0x01C20008,__READ_WRITE ,__timer64p_gpintgpen_bits);
__IO_REG32_BIT(TIMER64P0_GPDATGPDIR,  0x01C2000C,__READ_WRITE ,__timer64p_gpdatgpdir_bits);
__IO_REG32(    TIMER64P0_TIM12,       0x01C20010,__READ_WRITE );
__IO_REG32(    TIMER64P0_TIM34,       0x01C20014,__READ_WRITE );
__IO_REG32(    TIMER64P0_PRD12,       0x01C20018,__READ_WRITE );
__IO_REG32(    TIMER64P0_PRD34,       0x01C2001C,__READ_WRITE );
__IO_REG32_BIT(TIMER64P0_TCR,         0x01C20020,__READ_WRITE ,__timer64p_tcr_bits);
__IO_REG32_BIT(TIMER64P0_TGCR,        0x01C20024,__READ_WRITE ,__timer64p_tgcr_bits);
__IO_REG32_BIT(TIMER64P0_WDTCR,       0x01C20028,__READ_WRITE ,__timer64p_wdtcr_bits);
__IO_REG32(    TIMER64P0_REL12,       0x01C20034,__READ_WRITE );
__IO_REG32(    TIMER64P0_REL34,       0x01C20038,__READ_WRITE );
__IO_REG32(    TIMER64P0_CAP12,       0x01C2003C,__READ_WRITE );
__IO_REG32(    TIMER64P0_CAP34,       0x01C20040,__READ_WRITE );
__IO_REG32_BIT(TIMER64P0_INTCTLSTAT,  0x01C20044,__READ_WRITE ,__timer64p_intctlstat_bits);
__IO_REG32(    TIMER64P0_CMP0,        0x01C20060,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP1,        0x01C20064,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP2,        0x01C20068,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP3,        0x01C2006C,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP4,        0x01C20070,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP5,        0x01C20074,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP6,        0x01C20078,__READ_WRITE );
__IO_REG32(    TIMER64P0_CMP7,        0x01C2007C,__READ_WRITE );

/***************************************************************************
 **
 ** TIMER64P1
 **
 ***************************************************************************/
__IO_REG32(    TIMER64P1_REVID,       0x01C21000,__READ       );
__IO_REG32_BIT(TIMER64P1_EMUMGT,      0x01C21004,__READ_WRITE ,__timer64p_emumgt_bits);
__IO_REG32_BIT(TIMER64P1_GPINTGPEN,   0x01C21008,__READ_WRITE ,__timer64p_gpintgpen_bits);
__IO_REG32_BIT(TIMER64P1_GPDATGPDIR,  0x01C2100C,__READ_WRITE ,__timer64p_gpdatgpdir_bits);
__IO_REG32(    TIMER64P1_TIM12,       0x01C21010,__READ_WRITE );
__IO_REG32(    TIMER64P1_TIM34,       0x01C21014,__READ_WRITE );
__IO_REG32(    TIMER64P1_PRD12,       0x01C21018,__READ_WRITE );
__IO_REG32(    TIMER64P1_PRD34,       0x01C2101C,__READ_WRITE );
__IO_REG32_BIT(TIMER64P1_TCR,         0x01C21020,__READ_WRITE ,__timer64p_tcr_bits);
__IO_REG32_BIT(TIMER64P1_TGCR,        0x01C21024,__READ_WRITE ,__timer64p_tgcr_bits);
__IO_REG32_BIT(TIMER64P1_WDTCR,       0x01C21028,__READ_WRITE ,__timer64p_wdtcr_bits);
__IO_REG32(    TIMER64P1_REL12,       0x01C21034,__READ_WRITE );
__IO_REG32(    TIMER64P1_REL34,       0x01C21038,__READ_WRITE );
__IO_REG32(    TIMER64P1_CAP12,       0x01C2103C,__READ_WRITE );
__IO_REG32(    TIMER64P1_CAP34,       0x01C21040,__READ_WRITE );
__IO_REG32_BIT(TIMER64P1_INTCTLSTAT,  0x01C21044,__READ_WRITE ,__timer64p_intctlstat_bits);
__IO_REG32(    TIMER64P1_CMP0,        0x01C21060,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP1,        0x01C21064,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP2,        0x01C21068,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP3,        0x01C2106C,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP4,        0x01C21070,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP5,        0x01C21074,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP6,        0x01C21078,__READ_WRITE );
__IO_REG32(    TIMER64P1_CMP7,        0x01C2107C,__READ_WRITE );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32(    GPIO_REVID,            0x01E26000,__READ       );
__IO_REG32_BIT(GPIO_BINTEN,           0x01E26008,__READ_WRITE ,__gpio_binten_bits);
__IO_REG32_BIT(GPIO_DIR01,            0x01E26010,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_OUT_DATA01,       0x01E26014,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_SET_DATA01,       0x01E26018,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_CLR_DATA01,       0x01E2601C,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_IN_DATA01,        0x01E26020,__READ       ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_SET_RIS_TRIG01,   0x01E26024,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_CLR_RIS_TRIG01,   0x01E26028,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_SET_FAL_TRIG01,   0x01E2602C,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_CLR_FAL_TRIG01,   0x01E26030,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_INTSTAT01,        0x01E26034,__READ_WRITE ,__gpio_dir01_bits);
__IO_REG32_BIT(GPIO_DIR23,            0x01E26038,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_OUT_DATA23,       0x01E2603C,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_SET_DATA23,       0x01E26040,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_CLR_DATA23,       0x01E26044,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_IN_DATA23,        0x01E26048,__READ       ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_SET_RIS_TRIG23,   0x01E2604C,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_CLR_RIS_TRIG23,   0x01E26050,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_SET_FAL_TRIG23,   0x01E26054,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_CLR_FAL_TRIG23,   0x01E26058,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_INTSTAT23,        0x01E2605C,__READ_WRITE ,__gpio_dir23_bits);
__IO_REG32_BIT(GPIO_DIR45,            0x01E26060,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_OUT_DATA45,       0x01E26064,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_SET_DATA45,       0x01E26068,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_CLR_DATA45,       0x01E2606C,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_IN_DATA45,        0x01E26070,__READ       ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_SET_RIS_TRIG45,   0x01E26074,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_CLR_RIS_TRIG45,   0x01E26078,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_SET_FAL_TRIG45,   0x01E2607C,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_CLR_FAL_TRIG45,   0x01E26080,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_INTSTAT45,        0x01E26084,__READ_WRITE ,__gpio_dir45_bits);
__IO_REG32_BIT(GPIO_DIR67,            0x01E26088,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_OUT_DATA67,       0x01E2608C,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_SET_DATA67,       0x01E26090,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_CLR_DATA67,       0x01E26094,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_IN_DATA67,        0x01E26098,__READ       ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_SET_RIS_TRIG67,   0x01E2609C,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_CLR_RIS_TRIG67,   0x01E260A0,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_SET_FAL_TRIG67,   0x01E260A4,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_CLR_FAL_TRIG67,   0x01E260A8,__READ_WRITE ,__gpio_dir67_bits);
__IO_REG32_BIT(GPIO_INTSTAT67,        0x01E260AC,__READ_WRITE ,__gpio_dir67_bits);

/***************************************************************************
 **
 ** EDMA30CC0
 **
 ***************************************************************************/
__IO_REG32(    EDMA0CC_PID,           0x01C00000,__READ       );
__IO_REG32_BIT(EDMA0CC_CCCFG,         0x01C00004,__READ_WRITE ,__edmacc_cccfg_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP0,       0x01C00200,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP1,       0x01C00204,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP2,       0x01C00208,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP3,       0x01C0020C,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP4,       0x01C00210,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP5,       0x01C00214,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP6,       0x01C00218,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_QCHMAP7,       0x01C0021C,__READ_WRITE ,__edmacc_qchmap_bits);
__IO_REG32_BIT(EDMA0CC_DMAQNUM0,      0x01C00240,__READ_WRITE ,__edmacc_dmaqnum0_bits);
__IO_REG32_BIT(EDMA0CC_DMAQNUM1,      0x01C00244,__READ_WRITE ,__edmacc_dmaqnum1_bits);
__IO_REG32_BIT(EDMA0CC_DMAQNUM2,      0x01C00248,__READ_WRITE ,__edmacc_dmaqnum2_bits);
__IO_REG32_BIT(EDMA0CC_DMAQNUM3,      0x01C0024C,__READ_WRITE ,__edmacc_dmaqnum3_bits);
__IO_REG32_BIT(EDMA0CC_QDMAQNUM,      0x01C00260,__READ_WRITE ,__edmacc_qdmaqnum_bits);
//__IO_REG32_BIT(EDMA0CC_QUEPRI,        0x01C00284,__READ_WRITE ,__edmacc_quepri_bits);
__IO_REG32_BIT(EDMA0CC_EMR,           0x01C00300,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_EMCR,          0x01C00308,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_QEMR,          0x01C00310,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QEMCR,         0x01C00314,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_CCERR,         0x01C00318,__READ_WRITE ,__edmacc_ccerr_bits);
__IO_REG32_BIT(EDMA0CC_CCERRCLR,      0x01C0031C,__READ_WRITE ,__edmacc_ccerr_bits);
__IO_REG32_BIT(EDMA0CC_EEVAL,         0x01C00320,__READ_WRITE ,__edmacc_eeval_bits);
__IO_REG32_BIT(EDMA0CC_DRAE0,         0x01C00340,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_DRAE1,         0x01C00348,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_DRAE2,         0x01C00350,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_DRAE3,         0x01C00358,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_QRAE0,         0x01C00380,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QRAE1,         0x01C00384,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QRAE2,         0x01C00388,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QRAE3,         0x01C0038C,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_Q0E0,          0x01C00400,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E1,          0x01C00404,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E2,          0x01C00408,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E3,          0x01C0040C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E4,          0x01C00410,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E5,          0x01C00414,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E6,          0x01C00418,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E7,          0x01C0041C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E8,          0x01C00420,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E9,          0x01C00424,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E10,         0x01C00428,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E11,         0x01C0042C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E12,         0x01C00430,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E13,         0x01C00434,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E14,         0x01C00438,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q0E15,         0x01C0043C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E0,          0x01C00440,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E1,          0x01C00444,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E2,          0x01C00448,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E3,          0x01C0044C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E4,          0x01C00450,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E5,          0x01C00454,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E6,          0x01C00458,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E7,          0x01C0045C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E8,          0x01C00460,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E9,          0x01C00464,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E10,         0x01C00468,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E11,         0x01C0046C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E12,         0x01C00470,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E13,         0x01C00474,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E14,         0x01C00478,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_Q1E15,         0x01C0047C,__READ_WRITE ,__edmacc_qe_bits);
__IO_REG32_BIT(EDMA0CC_QSTAT0,        0x01C00600,__READ_WRITE ,__edmacc_qstat_bits);
__IO_REG32_BIT(EDMA0CC_QSTAT1,        0x01C00604,__READ_WRITE ,__edmacc_qstat_bits);
__IO_REG32_BIT(EDMA0CC_QWMTHRA,       0x01C00620,__READ_WRITE ,__edmacc_qwmthra_bits);
__IO_REG32_BIT(EDMA0CC_CCSTAT,        0x01C00640,__READ_WRITE ,__edmacc_ccstat_bits);
__IO_REG32_BIT(EDMA0CC_ER,            0x01C01000,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_ECR,           0x01C01008,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_ESR,           0x01C01010,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_CER,           0x01C01018,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_EER,           0x01C01020,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_EECR,          0x01C01028,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_EESR,          0x01C01030,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SER,           0x01C01038,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SECR,          0x01C01040,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_IER,           0x01C01050,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_IECR,          0x01C01058,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_IESR,          0x01C01060,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_IPR,           0x01C01068,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_ICR,           0x01C01070,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_IEVAL,         0x01C01078,__READ_WRITE ,__edmacc_ieval_bits);
__IO_REG32_BIT(EDMA0CC_QER,           0x01C01080,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QEER,          0x01C01084,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QEECR,         0x01C01088,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QEESR,         0x01C0108C,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QSER,          0x01C01090,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_QSECR,         0x01C01094,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0ER,         0x01C02000,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0ECR,        0x01C02008,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0ESR,        0x01C02010,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0CER,        0x01C02018,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0EER,        0x01C02020,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0EECR,       0x01C02028,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0EESR,       0x01C02030,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0SER,        0x01C02038,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0SECR,       0x01C02040,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC0IER,        0x01C02050,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC0IECR,       0x01C02058,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC0IESR,       0x01C02060,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC0IPR,        0x01C02068,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC0ICR,        0x01C02070,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC0IEVAL,      0x01C02078,__READ_WRITE ,__edmacc_ieval_bits);
__IO_REG32_BIT(EDMA0CC_SC0QER,        0x01C02080,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0QEER,       0x01C02084,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0QEECR,      0x01C02088,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0QEESR,      0x01C0208C,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0QSER,       0x01C02090,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC0QSECR,      0x01C02094,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1ER,         0x01C02200,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1ECR,        0x01C02208,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1ESR,        0x01C02210,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1CER,        0x01C02218,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1EER,        0x01C02220,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1EECR,       0x01C02228,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1EESR,       0x01C02230,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1SER,        0x01C02238,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1SECR,       0x01C02240,__READ_WRITE ,__edmacc_emr_bits);
__IO_REG32_BIT(EDMA0CC_SC1IER,        0x01C02250,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1IECR,       0x01C02258,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1IESR,       0x01C02260,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1IPR,        0x01C02268,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1ICR,        0x01C02270,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1IEVAL,      0x01C02278,__READ_WRITE ,__edmacc_ier_bits);
__IO_REG32_BIT(EDMA0CC_SC1QER,        0x01C02280,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1QEER,       0x01C02284,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1QEECR,      0x01C02288,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1QEESR,      0x01C0228C,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1QSER,       0x01C02290,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32_BIT(EDMA0CC_SC1QSECR,      0x01C02294,__READ_WRITE ,__edmacc_qemr_bits);
__IO_REG32(    EDMA0CC_PaRAM_BA,      0x01C04000,__READ_WRITE );

/***************************************************************************
 **
 ** EDMA30TC0
 **
 ***************************************************************************/
__IO_REG32(    EDMA0TC0_PID,          0x01C08000,__READ       );
__IO_REG32_BIT(EDMA0TC0_TCCFG,        0x01C08004,__READ_WRITE ,__edmatc_tccfg_bits);
__IO_REG32_BIT(EDMA0TC0_TCSTAT,       0x01C08100,__READ_WRITE ,__edmatc_tcstat_bits);
__IO_REG32_BIT(EDMA0TC0_ERRSTAT,      0x01C08120,__READ_WRITE ,__edmatc_errstat_bits);
__IO_REG32_BIT(EDMA0TC0_ERREN,        0x01C08124,__READ_WRITE ,__edmatc_errstat_bits);
__IO_REG32_BIT(EDMA0TC0_ERRCLR,       0x01C08128,__READ_WRITE ,__edmatc_errstat_bits);
__IO_REG32_BIT(EDMA0TC0_ERRDET,       0x01C0812C,__READ_WRITE ,__edmatc_errdet_bits);
__IO_REG32_BIT(EDMA0TC0_ERRCMD,       0x01C08130,__READ_WRITE ,__edmatc_errcmd_bits);
__IO_REG32_BIT(EDMA0TC0_RDRATE,       0x01C08140,__READ_WRITE ,__edmatc_rdrate_bits);
__IO_REG32_BIT(EDMA0TC0_SAOPT,        0x01C08240,__READ_WRITE ,__edmatc_saopt_bits);
__IO_REG32(    EDMA0TC0_SASRC,        0x01C08244,__READ       );
__IO_REG32_BIT(EDMA0TC0_SACNT,        0x01C08248,__READ       ,__edmatc_sacnt_bits);
__IO_REG32(    EDMA0TC0_SADST,        0x01C0824C,__READ       );
__IO_REG32_BIT(EDMA0TC0_SABIDX,       0x01C08250,__READ       ,__edmatc_sabidx_bits);
__IO_REG32_BIT(EDMA0TC0_SAMPPRXY,     0x01C08254,__READ       ,__edmatc_sampprxy_bits);
__IO_REG32_BIT(EDMA0TC0_SACNTRLD,     0x01C08258,__READ       ,__edmatc_sacntrld_bits);
__IO_REG32(    EDMA0TC0_SASRCBREF,    0x01C0825C,__READ       );
__IO_REG32(    EDMA0TC0_SADSTBREF,    0x01C08260,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFCNTRLD,     0x01C08280,__READ       ,__edmatc_sacntrld_bits);
__IO_REG32(    EDMA0TC0_DFSRCBREF,    0x01C08284,__READ       );
__IO_REG32(    EDMA0TC0_DFDSTBREF,    0x01C08288,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFOPT0,       0x01C08300,__READ_WRITE ,__edmatc_saopt_bits);
__IO_REG32(    EDMA0TC0_DFSRC0,       0x01C08304,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFCNT0,       0x01C08308,__READ       ,__edmatc_sacnt_bits);
__IO_REG32(    EDMA0TC0_DFDST0,       0x01C0830C,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFBIDX0,      0x01C08310,__READ       ,__edmatc_sabidx_bits);
__IO_REG32_BIT(EDMA0TC0_DFMPPRXY0,    0x01C08314,__READ       ,__edmatc_sampprxy_bits);
__IO_REG32_BIT(EDMA0TC0_DFOPT1,       0x01C08340,__READ_WRITE ,__edmatc_saopt_bits);
__IO_REG32(    EDMA0TC0_DFSRC1,       0x01C08344,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFCNT1,       0x01C08348,__READ       ,__edmatc_sacnt_bits);
__IO_REG32(    EDMA0TC0_DFDST1,       0x01C0834C,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFBIDX1,      0x01C08350,__READ       ,__edmatc_sabidx_bits);
__IO_REG32_BIT(EDMA0TC0_DFMPPRXY1,    0x01C08354,__READ       ,__edmatc_sampprxy_bits);
__IO_REG32_BIT(EDMA0TC0_DFOPT2,       0x01C08380,__READ_WRITE ,__edmatc_saopt_bits);
__IO_REG32(    EDMA0TC0_DFSRC2,       0x01C08384,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFCNT2,       0x01C08388,__READ       ,__edmatc_sacnt_bits);
__IO_REG32(    EDMA0TC0_DFDST2,       0x01C0838C,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFBIDX2,      0x01C08390,__READ       ,__edmatc_sabidx_bits);
__IO_REG32_BIT(EDMA0TC0_DFMPPRXY2,    0x01C08394,__READ       ,__edmatc_sampprxy_bits);
__IO_REG32_BIT(EDMA0TC0_DFOPT3,       0x01C083C0,__READ_WRITE ,__edmatc_saopt_bits);
__IO_REG32(    EDMA0TC0_DFSRC3,       0x01C083C4,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFCNT3,       0x01C083C8,__READ       ,__edmatc_sacnt_bits);
__IO_REG32(    EDMA0TC0_DFDST3,       0x01C083CC,__READ       );
__IO_REG32_BIT(EDMA0TC0_DFBIDX3,      0x01C083D0,__READ_WRITE ,__edmatc_sabidx_bits);
__IO_REG32_BIT(EDMA0TC0_DFMPPRXY3,    0x01C083D4,__READ_WRITE ,__edmatc_sampprxy_bits);

/***************************************************************************
 **
 ** EDMA30TC1
 **
 ***************************************************************************/
__IO_REG32(    EDMA0TC1_PID,          0x01C08400,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_TCCFG,        0x01C08404,__READ_WRITE ,__edmatc_tccfg_bits);                
__IO_REG32_BIT(EDMA0TC1_TCSTAT,       0x01C08500,__READ_WRITE ,__edmatc_tcstat_bits);               
__IO_REG32_BIT(EDMA0TC1_ERRSTAT,      0x01C08520,__READ_WRITE ,__edmatc_errstat_bits);              
__IO_REG32_BIT(EDMA0TC1_ERREN,        0x01C08524,__READ_WRITE ,__edmatc_errstat_bits);              
__IO_REG32_BIT(EDMA0TC1_ERRCLR,       0x01C08528,__READ_WRITE ,__edmatc_errstat_bits);              
__IO_REG32_BIT(EDMA0TC1_ERRDET,       0x01C0852C,__READ_WRITE ,__edmatc_errdet_bits);               
__IO_REG32_BIT(EDMA0TC1_ERRCMD,       0x01C08530,__READ_WRITE ,__edmatc_errcmd_bits);               
__IO_REG32_BIT(EDMA0TC1_RDRATE,       0x01C08540,__READ_WRITE ,__edmatc_rdrate_bits);               
__IO_REG32_BIT(EDMA0TC1_SAOPT,        0x01C08640,__READ_WRITE ,__edmatc_saopt_bits);                
__IO_REG32(    EDMA0TC1_SASRC,        0x01C08644,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_SACNT,        0x01C08648,__READ       ,__edmatc_sacnt_bits);                
__IO_REG32(    EDMA0TC1_SADST,        0x01C0864C,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_SABIDX,       0x01C08650,__READ       ,__edmatc_sabidx_bits);               
__IO_REG32_BIT(EDMA0TC1_SAMPPRXY,     0x01C08654,__READ       ,__edmatc_sampprxy_bits);             
__IO_REG32_BIT(EDMA0TC1_SACNTRLD,     0x01C08658,__READ       ,__edmatc_sacntrld_bits);             
__IO_REG32(    EDMA0TC1_SASRCBREF,    0x01C0865C,__READ       );                                    
__IO_REG32(    EDMA0TC1_SADSTBREF,    0x01C08660,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFCNTRLD,     0x01C08680,__READ       ,__edmatc_sacntrld_bits);             
__IO_REG32(    EDMA0TC1_DFSRCBREF,    0x01C08684,__READ       );                                    
__IO_REG32(    EDMA0TC1_DFDSTBREF,    0x01C08688,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFOPT0,       0x01C08700,__READ_WRITE ,__edmatc_saopt_bits);                
__IO_REG32(    EDMA0TC1_DFSRC0,       0x01C08704,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFCNT0,       0x01C08708,__READ       ,__edmatc_sacnt_bits);                
__IO_REG32(    EDMA0TC1_DFDST0,       0x01C0870C,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFBIDX0,      0x01C08710,__READ       ,__edmatc_sabidx_bits);               
__IO_REG32_BIT(EDMA0TC1_DFMPPRXY0,    0x01C08714,__READ       ,__edmatc_sampprxy_bits);             
__IO_REG32_BIT(EDMA0TC1_DFOPT1,       0x01C08740,__READ_WRITE ,__edmatc_saopt_bits);                
__IO_REG32(    EDMA0TC1_DFSRC1,       0x01C08744,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFCNT1,       0x01C08748,__READ       ,__edmatc_sacnt_bits);                
__IO_REG32(    EDMA0TC1_DFDST1,       0x01C0874C,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFBIDX1,      0x01C08750,__READ       ,__edmatc_sabidx_bits);               
__IO_REG32_BIT(EDMA0TC1_DFMPPRXY1,    0x01C08754,__READ       ,__edmatc_sampprxy_bits);             
__IO_REG32_BIT(EDMA0TC1_DFOPT2,       0x01C08780,__READ_WRITE ,__edmatc_saopt_bits);                
__IO_REG32(    EDMA0TC1_DFSRC2,       0x01C08784,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFCNT2,       0x01C08788,__READ       ,__edmatc_sacnt_bits);                
__IO_REG32(    EDMA0TC1_DFDST2,       0x01C0878C,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFBIDX2,      0x01C08790,__READ       ,__edmatc_sabidx_bits);               
__IO_REG32_BIT(EDMA0TC1_DFMPPRXY2,    0x01C08794,__READ       ,__edmatc_sampprxy_bits);             
__IO_REG32_BIT(EDMA0TC1_DFOPT3,       0x01C087C0,__READ_WRITE ,__edmatc_saopt_bits);                
__IO_REG32(    EDMA0TC1_DFSRC3,       0x01C087C4,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFCNT3,       0x01C087C8,__READ       ,__edmatc_sacnt_bits);                
__IO_REG32(    EDMA0TC1_DFDST3,       0x01C087CC,__READ       );                                    
__IO_REG32_BIT(EDMA0TC1_DFBIDX3,      0x01C087D0,__READ_WRITE ,__edmatc_sabidx_bits);               
__IO_REG32_BIT(EDMA0TC1_DFMPPRXY3,    0x01C087D4,__READ_WRITE ,__edmatc_sampprxy_bits);             

/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

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
 **  AINT interrupt channels
 **
 ***************************************************************************/
#define AINT_COMMTX             0
#define AINT_COMMRX             1
#define AINT_NINT               2
#define AINT_PRU_EVTOUT0        3
#define AINT_PRU_EVTOUT1        4
#define AINT_PRU_EVTOUT2        5
#define AINT_PRU_EVTOUT3        6
#define AINT_PRU_EVTOUT4        7
#define AINT_PRU_EVTOUT5        8
#define AINT_PRU_EVTOUT6        9
#define AINT_PRU_EVTOUT7        10
#define AINT_EDMA3_0_CC0_INT0   11
#define AINT_EDMA3_0_CC0_ERRINT 12
#define AINT_EDMA3_0_TC0_ERRINT 13
#define AINT_EMIFA_INT          14
#define AINT_IIC0_INT           15
#define AINT_MMCSD0_INT0        16
#define AINT_MMCSD0_INT1        17
#define AINT_PSC0_ALLINT        18
#define AINT_SPI0_INT           20
#define AINT_T64P0_TINT12       21
#define AINT_T64P0_TINT34       22
#define AINT_T64P1_TINT12       23
#define AINT_T64P1_TINT34       24
#define AINT_UART0_INT          25
#define AINT_PROTERR            27
#define AINT_EDMA3_0_TC1_ERRINT 32
#define AINT_EMAC_C0RXTHRESH    33
#define AINT_EMAC_C0RX          34
#define AINT_EMAC_C0TX          35
#define AINT_EMAC_C0MISC        36
#define AINT_EMAC_C1RXTHRESH    37
#define AINT_EMAC_C1RX          38
#define AINT_EMAC_C1TX          39
#define AINT_EMAC_C1MISC        40
#define AINT_DDR2_MEMERR        41
#define AINT_GPIO_B0INT         42
#define AINT_GPIO_B1INT         43
#define AINT_GPIO_B2INT         44
#define AINT_GPIO_B3INT         45
#define AINT_GPIO_B4INT         46
#define AINT_GPIO_B5INT         47
#define AINT_GPIO_B6INT         48
#define AINT_GPIO_B7INT         49
#define AINT_IIC1_INT           51
#define AINT_UART_INT1          53
#define AINT_MCASP_INT          54
#define AINT_PSC1_ALLINT        55
#define AINT_SPI1_INT           56
#define AINT_USB0_INT           58
#define AINT_UART2_INT          61
#define AINT_EHRPWM0            63
#define AINT_EHRPWM0TZ          64
#define AINT_EHRPWM1            65
#define AINT_EHRPWM1TZ          66
#define AINT_EHRPWM2            67
#define AINT_EHRPWM2TZ          68
#define AINT_ECAP0              69
#define AINT_ECAP1              70
#define AINT_ECAP2              71
#define AINT_EQEP0              72
#define AINT_EQEP1              73
#define AINT_T64P0_CMPINT0      74
#define AINT_T64P0_CMPINT1      75
#define AINT_T64P0_CMPINT2      76
#define AINT_T64P0_CMPINT3      77
#define AINT_T64P0_CMPINT4      78
#define AINT_T64P0_CMPINT5      79
#define AINT_T64P0_CMPINT6      80
#define AINT_T64P0_CMPINT7      81
#define AINT_T64P1_CMPINT0      82
#define AINT_T64P1_CMPINT1      83
#define AINT_T64P1_CMPINT2      84
#define AINT_T64P1_CMPINT3      85
#define AINT_T64P1_CMPINT4      86
#define AINT_T64P1_CMPINT5      87
#define AINT_T64P1_CMPINT6      88
#define AINT_T64P1_CMPINT7      89
#define AINT_ARMCLKSTOPREQ      90

#endif    /* __IOAM1705_H */
