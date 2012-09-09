/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Texas Instruments OMAP-L138
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 54436 $
 **
 ***************************************************************************/

#ifndef __IOOMAPL138_H
#define __IOOMAPL138_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4f = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    OMAP-L138 SPECIAL FUNCTION REGISTERS
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
  __REG32 EXTCLKSRC       : 1;
  __REG32                 :22;
} __pllc0_pllctl_bits;

/* PLLC1 Control Register (PLLCTL) */
typedef struct {
  __REG32 PLLEN           : 1;
  __REG32 PLLPWRDN        : 1;
  __REG32                 : 1;
  __REG32 PLLRST          : 1;
  __REG32                 : 1;
  __REG32 PLLENSRC        : 1;
  __REG32                 :26;
} __pllc1_pllctl_bits;

/* PLLCx OBSCLK Select Register (OCSEL) */
typedef struct {
  __REG32 OCSRC           : 5;
  __REG32                 :27;
} __pllc_ocsel_bits;

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

/* PLLCx Oscillator Divider 1 Register (OSCDIV) */
typedef struct {
  __REG32 RATIO           : 5;
  __REG32                 :10;
  __REG32 OD1EN           : 1;
  __REG32                 :16;
} __pllc_oscdiv_bits;

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

/* PLLC1 Clock Align Control Register (ALNCTL) */
typedef struct {
  __REG32 ALN1            : 1;
  __REG32 ALN2            : 1;
  __REG32 ALN3            : 1;
  __REG32                 :29;
} __pllc1_alnctl_bits;

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

/* PLLC1 PLLDIV Ratio Change Status Register (DCHANGE) */
typedef struct {
  __REG32 SYS1            : 1;
  __REG32 SYS2            : 1;
  __REG32 SYS3            : 1;
  __REG32                 :29;
} __pllc1_dchange_bits;

/* PLLC0 Clock Enable Control Register (CKEN) */
typedef struct {
  __REG32 AUXEN           : 1;
  __REG32 OBSEN           : 1;
  __REG32                 :30;
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

/* PLLC1 SYSCLK Status Register (SYSTAT) */
typedef struct {
  __REG32 SYS1ON          : 1;
  __REG32 SYS2ON          : 1;
  __REG32 SYS3ON          : 1;
  __REG32                 :29;
} __pllc1_systat_bits;

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
  __REG32 M15             : 1;
  __REG32                 :16;
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
  __REG32                 : 1;
  __REG32 DSP_MDMA        : 3;
  __REG32                 : 1;
  __REG32 DSP_CFG         : 3;
  __REG32                 : 1;
  __REG32 UPP             : 3;
  __REG32                 : 1;
  __REG32 SATA            : 3;
  __REG32                 : 9;
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
  __REG32                 : 1;
  __REG32 EDMA31TC0       : 3;
  __REG32                 : 5;
  __REG32 VPIF_DMA_0      : 3;
  __REG32                 : 1;
  __REG32 VPIF_DMA_1      : 3;
  __REG32                 : 1;
} __syscfg_mstpri1_bits;

/* SYSCFG Master Priority 2 Register (MSTPRI2) */
typedef struct {
  __REG32 EMAC            : 3;
  __REG32                 : 5;
  __REG32 USB0CFG         : 3;
  __REG32                 : 1;
  __REG32 USB0CDMA        : 3;
  __REG32                 : 5;
  __REG32 UHPI            : 3;
  __REG32                 : 1;
  __REG32 USB1            : 3;
  __REG32                 : 1;
  __REG32 LCDC            : 3;
  __REG32                 : 1;
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
  __REG32 PINMUX19_7_4     : 4;
  __REG32 PINMUX19_11_8    : 4;
  __REG32 PINMUX19_15_12   : 4;
  __REG32 PINMUX19_19_16   : 4;
  __REG32 PINMUX19_23_20   : 4;
  __REG32 PINMUX19_27_24   : 4;
  __REG32 PINMUX19_31_28   : 4;
} __syscfg_pinmux19_bits;

/* SYSCFG Suspend Source Register (SUSPSRC) */
typedef struct {
  __REG32 ECAP0SRC         : 1;
  __REG32 ECAP1SRC         : 1;
  __REG32 ECAP2SRC         : 1;
  __REG32 TIMER64P_3SRC    : 1;
  __REG32 UPPSRC           : 1;
  __REG32 EMACSRC          : 1;
  __REG32 PRUSRC           : 1;
  __REG32 MCBSP0SRC        : 1;
  __REG32 MCBSP1SRC        : 1;
  __REG32 USB0SRC          : 1;
  __REG32                  : 2;
  __REG32 HPISRC           : 1;
  __REG32 SATASRC          : 1;
  __REG32 VPIFSRC          : 1;
  __REG32                  : 1;
  __REG32 I2C0SRC          : 1;
  __REG32 I2C1SRC          : 1;
  __REG32 UART0SRC         : 1;
  __REG32 UART1SRC         : 1;
  __REG32 UART2SRC         : 1;
  __REG32 SPI0SRC          : 1;
  __REG32 SPI1SRC          : 1;
  __REG32 EPWM0SRC         : 1;
  __REG32 EPWM1SRC         : 1;
  __REG32                  : 2;
  __REG32 TIMER64P_0SRC    : 1;
  __REG32 TIMER64P_1SRC    : 1;
  __REG32 TIMER64P_2SRC    : 1;
  __REG32                  : 2;
} __syscfg_suspsrc_bits;

/* SYSCFG Suspend Source Register (SUSPSRC) */
/* SYSCFG Chip Signal Clear Register (CHIPSIG_CLR) */
typedef struct {
  __REG32 CHIPSIG0         : 1;
  __REG32 CHIPSIG1         : 1;
  __REG32 CHIPSIG2         : 1;
  __REG32 CHIPSIG3         : 1;
  __REG32 CHIPSIG4         : 1;
  __REG32                  :27;
} __syscfg_chipsig_bits;

/* SYSCFG Chip Configuration 0 Register (CFGCHIP0) */
typedef struct {
  __REG32 EDMA30TC0DBS     : 2;
  __REG32 EDMA30TC1DBS     : 2;
  __REG32 PLL_MASTER_LOCK  : 1;
  __REG32                  :27;
} __syscfg_cfgchip0_bits;

/* SYSCFG Chip Configuration 1 Register (CFGCHIP1) */
typedef struct {
  __REG32 AMUTESEL0        : 4;
  __REG32                  : 8;
  __REG32 TBCLKSYNC        : 1;
  __REG32 EDMA31TC0DBS     : 2;
  __REG32 HPIENA           : 1;
  __REG32 HPIBYTEAD        : 1;
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
  __REG32 USB1SUSPENDM     : 1;
  __REG32 USB0DATPOL       : 1;
  __REG32 USB0OTGPWRDN     : 1;
  __REG32 USB0PHYPWDN      : 1;
  __REG32 USB0PHYCLKMUX    : 1;
  __REG32 USB1PHYCLKMUX    : 1;
  __REG32 USB0OTGMODE      : 2;
  __REG32 RESET            : 1;
  __REG32 USB0VBUSSENSE    : 1;
  __REG32 USB0PHYCLKGD     : 1;
  __REG32                  :14;
} __syscfg_cfgchip2_bits;

/* SYSCFG Chip Configuration 3 Register (CFGCHIP3) */
typedef struct {
  __REG32                  : 1;
  __REG32 EMA_CLKSRC       : 1;
  __REG32 DIV45PENA        : 1;
  __REG32 PRUEVTSEL        : 1;
  __REG32 ASYNC3_CLKSRC    : 1;
  __REG32 PLL1_MASTER_LOCK : 1;
  __REG32 UPP_TX_CLKSRC    : 1;
  __REG32                  : 1;
  __REG32 RMII_SEL         : 1;
  __REG32                  :23;
} __syscfg_cfgchip3_bits;

/* SYSCFG Chip Configuration 4 Register (CFGCHIP4) */
typedef struct {
  __REG32 AMUTECLR0        : 1;
  __REG32                  :31;
} __syscfg_cfgchip4_bits;

/* SYSCFG VTP I/O Control Register (VTPIO_CTL) */
typedef struct {
  __REG32 F2               : 1;
  __REG32 F1               : 1;
  __REG32 F0               : 1;
  __REG32 D2               : 1;
  __REG32 D1               : 1;
  __REG32 D0               : 1;
  __REG32 POWERDN          : 1;
  __REG32 LOCK             : 1;
  __REG32 PWRSAVE          : 1;
  __REG32 FORCEUPN         : 1;
  __REG32 FORCEUPP         : 1;
  __REG32 FORCEDNN         : 1;
  __REG32 FORCEDNP         : 1;
  __REG32 CLKRZ            : 1;
  __REG32 IOPWRDN          : 1;
  __REG32 READY            : 1;
  __REG32 VREFTAP          : 2;
  __REG32 VREFEN           : 1;
  __REG32                  :13;
} __syscfg_vtpio_ctl_bits;

/* SYSCFG DDR Slew Register (DDR_SLEW) */
typedef struct {
  __REG32 DDR_CMDSLEW      : 2;
  __REG32 DDR_DATASLEW     : 2;
  __REG32 CMOSEN           : 1;
  __REG32 DDR_PDENA        : 1;
  __REG32                  : 2;
  __REG32 ODT_TERMOFF      : 2;
  __REG32 ODT_TERMON       : 2;
  __REG32                  :20;
} __syscfg_ddr_slew_bits;

/* SYSCFG Deep Sleep Register (DEEPSLEEP) */
typedef struct {
  __REG32 SLEEPCOUNT       :16;
  __REG32                  :14;
  __REG32 SLEEPCOMPLETE    : 1;
  __REG32 SLEEPENABLE      : 1;
} __syscfg_deep_sleep_bits;

/* SYSCFG Pullup/Pulldown Enable Register (PUPD_ENA) */
typedef struct {
  __REG32 PUPDENA0         : 1;
  __REG32 PUPDENA1         : 1;
  __REG32 PUPDENA2         : 1;
  __REG32 PUPDENA3         : 1;
  __REG32 PUPDENA4         : 1;
  __REG32 PUPDENA5         : 1;
  __REG32 PUPDENA6         : 1;
  __REG32 PUPDENA7         : 1;
  __REG32 PUPDENA8         : 1;
  __REG32 PUPDENA9         : 1;
  __REG32 PUPDENA10        : 1;
  __REG32 PUPDENA11        : 1;
  __REG32 PUPDENA12        : 1;
  __REG32 PUPDENA13        : 1;
  __REG32 PUPDENA14        : 1;
  __REG32 PUPDENA15        : 1;
  __REG32 PUPDENA16        : 1;
  __REG32 PUPDENA17        : 1;
  __REG32 PUPDENA18        : 1;
  __REG32 PUPDENA19        : 1;
  __REG32 PUPDENA20        : 1;
  __REG32 PUPDENA21        : 1;
  __REG32 PUPDENA22        : 1;
  __REG32 PUPDENA23        : 1;
  __REG32 PUPDENA24        : 1;
  __REG32 PUPDENA25        : 1;
  __REG32 PUPDENA26        : 1;
  __REG32 PUPDENA27        : 1;
  __REG32 PUPDENA28        : 1;
  __REG32 PUPDENA29        : 1;
  __REG32 PUPDENA30        : 1;
  __REG32 PUPDENA31        : 1;
} __syscfg_pupd_ena_bits;

/* SYSCFG Pullup/Pulldown Select Register (PUPD_SEL) */
typedef struct {
  __REG32 PUPDSEL0         : 1;
  __REG32 PUPDSEL1         : 1;
  __REG32 PUPDSEL2         : 1;
  __REG32 PUPDSEL3         : 1;
  __REG32 PUPDSEL4         : 1;
  __REG32 PUPDSEL5         : 1;
  __REG32 PUPDSEL6         : 1;
  __REG32 PUPDSEL7         : 1;
  __REG32 PUPDSEL8         : 1;
  __REG32 PUPDSEL9         : 1;
  __REG32 PUPDSEL10        : 1;
  __REG32 PUPDSEL11        : 1;
  __REG32 PUPDSEL12        : 1;
  __REG32 PUPDSEL13        : 1;
  __REG32 PUPDSEL14        : 1;
  __REG32 PUPDSEL15        : 1;
  __REG32 PUPDSEL16        : 1;
  __REG32 PUPDSEL17        : 1;
  __REG32 PUPDSEL18        : 1;
  __REG32 PUPDSEL19        : 1;
  __REG32 PUPDSEL20        : 1;
  __REG32 PUPDSEL21        : 1;
  __REG32 PUPDSEL22        : 1;
  __REG32 PUPDSEL23        : 1;
  __REG32 PUPDSEL24        : 1;
  __REG32 PUPDSEL25        : 1;
  __REG32 PUPDSEL26        : 1;
  __REG32 PUPDSEL27        : 1;
  __REG32 PUPDSEL28        : 1;
  __REG32 PUPDSEL29        : 1;
  __REG32 PUPDSEL30        : 1;
  __REG32 PUPDSEL31        : 1;
} __syscfg_pupd_sel_bits;

/* SYSCFG RXACTIVE Control Register (RXACTIVE) */
typedef struct {
  __REG32 RXACTIVE0        : 1;
  __REG32 RXACTIVE1        : 1;
  __REG32 RXACTIVE2        : 1;
  __REG32 RXACTIVE3        : 1;
  __REG32 RXACTIVE4        : 1;
  __REG32 RXACTIVE5        : 1;
  __REG32 RXACTIVE6        : 1;
  __REG32 RXACTIVE7        : 1;
  __REG32 RXACTIVE8        : 1;
  __REG32 RXACTIVE9        : 1;
  __REG32 RXACTIVE10       : 1;
  __REG32 RXACTIVE11       : 1;
  __REG32 RXACTIVE12       : 1;
  __REG32 RXACTIVE13       : 1;
  __REG32 RXACTIVE14       : 1;
  __REG32 RXACTIVE15       : 1;
  __REG32 RXACTIVE16       : 1;
  __REG32 RXACTIVE17       : 1;
  __REG32 RXACTIVE18       : 1;
  __REG32 RXACTIVE19       : 1;
  __REG32 RXACTIVE20       : 1;
  __REG32 RXACTIVE21       : 1;
  __REG32 RXACTIVE22       : 1;
  __REG32 RXACTIVE23       : 1;
  __REG32 RXACTIVE24       : 1;
  __REG32 RXACTIVE25       : 1;
  __REG32 RXACTIVE26       : 1;
  __REG32 RXACTIVE27       : 1;
  __REG32 RXACTIVE28       : 1;
  __REG32 RXACTIVE29       : 1;
  __REG32 RXACTIVE30       : 1;
  __REG32 RXACTIVE31       : 1;
} __syscfg_rxactive_bits;

/* SYSCFG Power Down Control Register (PWRDN) */
typedef struct {
  __REG32 SATACLK_PWRDN    : 1;
  __REG32                  :31;
} __syscfg_pwrdn_bits;

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
  __REG32 RAW_STATUS91     : 1;
  __REG32 RAW_STATUS92     : 1;
  __REG32 RAW_STATUS93     : 1;
  __REG32 RAW_STATUS94     : 1;
  __REG32 RAW_STATUS95     : 1;
} __aintc_srsr2_bits;

/* AINTC System Interrupt Status Raw/Set Register 3 (SRSR3) */
typedef struct {
  __REG32 RAW_STATUS96     : 1;
  __REG32 RAW_STATUS97     : 1;
  __REG32 RAW_STATUS98     : 1;
  __REG32 RAW_STATUS99     : 1;
  __REG32 RAW_STATUS100    : 1;
  __REG32                  :27;
} __aintc_srsr3_bits;

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
  __REG32 ENBL_STATUS91     : 1;
  __REG32 ENBL_STATUS92     : 1;
  __REG32 ENBL_STATUS93     : 1;
  __REG32 ENBL_STATUS94     : 1;
  __REG32 ENBL_STATUS95     : 1;
} __aintc_secr2_bits;

/* AINTC System Interrupt Status Enabled/Clear Register 3 (SECR3) */
typedef struct {
  __REG32 ENBL_STATUS96     : 1;
  __REG32 ENBL_STATUS97     : 1;
  __REG32 ENBL_STATUS98     : 1;
  __REG32 ENBL_STATUS99     : 1;
  __REG32 ENBL_STATUS100    : 1;
  __REG32                  :27;
} __aintc_secr3_bits;

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
  __REG32 ENABLE91     : 1;
  __REG32 ENABLE92     : 1;
  __REG32 ENABLE93     : 1;
  __REG32 ENABLE94     : 1;
  __REG32 ENABLE95     : 1;
} __aintc_esr2_bits;

/* AINTC System Interrupt Enable Set Register 3 (ESR3) */
typedef struct {
  __REG32 ENABLE96     : 1;
  __REG32 ENABLE97     : 1;
  __REG32 ENABLE98     : 1;
  __REG32 ENABLE99     : 1;
  __REG32 ENABLE100    : 1;
  __REG32              :27;
} __aintc_esr3_bits;

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
  __REG32 DISABLE91     : 1;
  __REG32 DISABLE92     : 1;
  __REG32 DISABLE93     : 1;
  __REG32 DISABLE94     : 1;
  __REG32 DISABLE95     : 1;
} __aintc_ecr2_bits;

/* AINTC System Interrupt Enable Clear Register 3 (ECR3) */
typedef struct {
  __REG32 DISABLE96     : 1;
  __REG32 DISABLE97     : 1;
  __REG32 DISABLE98     : 1;
  __REG32 DISABLE99     : 1;
  __REG32 DISABLE100    : 1;
  __REG32               :27;
} __aintc_ecr3_bits;

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
  __REG32 CHNL_91       : 8;
} __aintc_cmr22_bits;

/* AINTC Channel Map Registers (CMR23) */
typedef struct {
  __REG32 CHNL_92       : 8;
  __REG32 CHNL_93       : 8;
  __REG32 CHNL_94       : 8;
  __REG32 CHNL_95       : 8;
} __aintc_cmr23_bits;

/* AINTC Channel Map Registers (CMR24) */
typedef struct {
  __REG32 CHNL_96       : 8;
  __REG32 CHNL_97       : 8;
  __REG32 CHNL_98       : 8;
  __REG32 CHNL_99       : 8;
} __aintc_cmr24_bits;

/* AINTC Channel Map Registers (CMR25) */
typedef struct {
  __REG32 CHNL_100      : 8;
  __REG32               :24;
} __aintc_cmr25_bits;

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

/* uPP Peripheral Control Register (UPPCR) */
typedef struct {
  __REG32 FREE          : 1;
  __REG32 SOFT          : 1;
  __REG32 RTEMU         : 1;
  __REG32 EN            : 1;
  __REG32 SWRST         : 1;
  __REG32               : 2;
  __REG32 DB            : 1;
  __REG32               :24;
} __uppcr_bits;

/* uPP Digital Loopback Register (UPDLB) */
typedef struct {
  __REG32               :12;
  __REG32 AB            : 1;
  __REG32 BA            : 1;
  __REG32               :18;
} __updlb_bits;

/* uPP Channel Control Register (UPCTL) */
typedef struct {
  __REG32 MODE          : 2;
  __REG32 CHN           : 1;
  __REG32 SDRTXIL       : 1;
  __REG32 DDRDEMUX      : 1;
  __REG32               :11;
  __REG32 DRA           : 1;
  __REG32 IWA           : 1;
  __REG32 DPWA          : 3;
  __REG32 DPFA          : 2;
  __REG32               : 1;
  __REG32 DRB           : 1;
  __REG32 IWB           : 1;
  __REG32 DPWB          : 3;
  __REG32 DPFB          : 2;
  __REG32               : 1;
} __upctl_bits;

/* uPP Interface Configuration Register (UPICR) */
typedef struct {
  __REG32 STARTPOLA     : 1;
  __REG32 ENAPOLA       : 1;
  __REG32 WAITPOLA      : 1;
  __REG32 STARTA        : 1;
  __REG32 ENAA          : 1;
  __REG32 WAITA         : 1;
  __REG32               : 2;
  __REG32 CLKDIVA       : 4;
  __REG32 CLKINVA       : 1;
  __REG32 TRISA         : 1;
  __REG32               : 2;
  __REG32 STARTPOLB     : 1;
  __REG32 ENAPOLB       : 1;
  __REG32 WAITPOLB      : 1;
  __REG32 STARTB        : 1;
  __REG32 ENAB          : 1;
  __REG32 WAITB         : 1;
  __REG32               : 2;
  __REG32 CLKDIVB       : 4;
  __REG32 CLKINVB       : 1;
  __REG32 TRISB         : 1;
  __REG32               : 2;
} __upicr_bits;

/* uPP Interface Idle Value Register (UPIVR) */
typedef struct {
  __REG32 VALA          :16;
  __REG32 VALB          :16;
} __upivr_bits;

/* uPP Threshold Configuration Register (UPTCR) */
typedef struct {
  __REG32 RDSIZEI       : 2;
  __REG32               : 6;
  __REG32 RDSIZEQ       : 2;
  __REG32               : 6;
  __REG32 TXSIZEA       : 2;
  __REG32               : 6;
  __REG32 TXSIZEB       : 2;
  __REG32               : 6;
} __uptcr_bits;

/* uPP Interrupt Raw Status Register (UPISR) */
typedef struct {
  __REG32 DPEI          : 1;
  __REG32 UORI          : 1;
  __REG32 ERRI          : 1;
  __REG32 EOWI          : 1;
  __REG32 EOLI          : 1;
  __REG32               : 3;
  __REG32 DPEQ          : 1;
  __REG32 UORQ          : 1;
  __REG32 ERRQ          : 1;
  __REG32 EOWQ          : 1;
  __REG32 EOLQ          : 1;
  __REG32               :19;
} __upisr_bits;

/* uPP End of Interrupt Register (UPEOI) */
typedef struct {
  __REG32 EOI           : 8;
  __REG32               :24;
} __upeoi_bits;

/* uPP DMA Channel I Descriptor 1 Register (UPID1) */
/* uPP DMA Channel I Status 1 Register (UPIS1) */
/* uPP DMA Channel Q Descriptor 1 Register (UPQD1) */
/* uPP DMA Channel Q Status 1 Register (UPQS1) */
typedef struct {
  __REG32 BCNT          :16;
  __REG32 LNCNT         :16;
} __upid1_bits;

/* uPP DMA Channel I Descriptor 2 Register (UPID2) */
/* uPP DMA Channel Q Descriptor 2 Register (UPID2) */
typedef struct {
  __REG32 LNOFFSET      :16;
  __REG32               :16;
} __upid2_bits;

/* uPP DMA Channel I Status 2 Register (UPIS2) */
/* uPP DMA Channel Q Status 2 Register (UPQS2) */
typedef struct {
  __REG32 ACT           : 1;
  __REG32 PEND          : 1;
  __REG32               : 2;
  __REG32 WM            : 4;
  __REG32               :24;
} __upis2_bits;

/* OHCI Revision Number Register (HCREVISION) */
typedef struct {
  __REG32 REV           : 8;
  __REG32               :24;
} __hcrevision_bits;

/* OHCI Operating Mode Register (HCCONTROL) */
typedef struct {
  __REG32 CBSR          : 2;
  __REG32 PLE           : 1;
  __REG32 IE            : 1;
  __REG32 CLE           : 1;
  __REG32 BLE           : 1;
  __REG32 HCFS          : 2;
  __REG32 IR            : 1;
  __REG32 RWC           : 1;
  __REG32 RWE           : 1;
  __REG32               :21;
} __hccontrol_bits;

/* OHCI Command and Status Register (HCCOMMANDSTATUS) */
typedef struct {
  __REG32 HCR           : 1;
  __REG32 CLF           : 1;
  __REG32 BLF           : 1;
  __REG32 OCR           : 1;
  __REG32               :12;
  __REG32 SOC           : 2;
  __REG32               :14;
} __hccommandstatus_bits;

/* OHCI Interrupt and Status Register (HCINTERRUPTSTATUS) */
typedef struct {
  __REG32 SO            : 1;
  __REG32 WDH           : 1;
  __REG32 SF            : 1;
  __REG32 RD            : 1;
  __REG32 UE            : 1;
  __REG32 FNO           : 1;
  __REG32 RHSC          : 1;
  __REG32               :23;
  __REG32 OC            : 1;
  __REG32               : 1;
} __hcinterruptstatus_bits;

/* OHCI Interrupt Enable Register (HCINTERRUPTENABLE) */
typedef struct {
  __REG32 SO            : 1;
  __REG32 WDH           : 1;
  __REG32 SF            : 1;
  __REG32 RD            : 1;
  __REG32 UE            : 1;
  __REG32 FNO           : 1;
  __REG32 RHSC          : 1;
  __REG32               :23;
  __REG32 OC            : 1;
  __REG32 MIE           : 1;
} __hcinterruptenable_bits;

/* OHCI Frame Interval Register (HCFMINTERVAL) */
typedef struct {
  __REG32 FRAMEINTERVAL :14;
  __REG32               : 2;
  __REG32 FSMPS         :15;
  __REG32 FIT           : 1;
} __hcfminterval_bits;

/* OHCI Frame Remaining Register (HCFMREMAINING) */
typedef struct {
  __REG32 FR            :14;
  __REG32               :17;
  __REG32 FRT           : 1;
} __hcfmremaining_bits;

/* OHCI Frame Number Register (HCFMNUMBER) */
typedef struct {
  __REG32 FR            :16;
  __REG32               :16;
} __hcfmnumber_bits;

/* OHCI Periodic Start Register (HCPERIODICSTART) */
typedef struct {
  __REG32 PS            :14;
  __REG32               :18;
} __hcperiodicstart_bits;

/* OHCI Low-Speed Threshold Register (HCLSTHRESHOLD) */
typedef struct {
  __REG32 PS            :14;
  __REG32               :18;
} __hclsthreshold_bits;

/* OHCI Root Hub A Register (HCRHDESCRIPTORA) */
typedef struct {
  __REG32 NDP           : 8;
  __REG32 PSM           : 1;
  __REG32 NPS           : 1;
  __REG32 DT            : 1;
  __REG32 OCPM          : 1;
  __REG32 NOCP          : 1;
  __REG32               :11;
  __REG32 POTPG         : 8;
} __hcrhdescriptora_bits;

/* OHCI Root Hub B Register (HCRHDESCRIPTORB) */
typedef struct {
  __REG32 DR0           : 1;
  __REG32 DR1           : 1;
  __REG32 DR2           : 1;
  __REG32 DR3           : 1;
  __REG32 DR4           : 1;
  __REG32 DR5           : 1;
  __REG32 DR6           : 1;
  __REG32 DR7           : 1;
  __REG32 DR8           : 1;
  __REG32 DR9           : 1;
  __REG32 DR10          : 1;
  __REG32 DR11          : 1;
  __REG32 DR12          : 1;
  __REG32 DR13          : 1;
  __REG32 DR14          : 1;
  __REG32 DR15          : 1;
  __REG32 PPCM0         : 1;
  __REG32 PPCM1         : 1;
  __REG32 PPCM2         : 1;
  __REG32 PPCM3         : 1;
  __REG32 PPCM4         : 1;
  __REG32 PPCM5         : 1;
  __REG32 PPCM6         : 1;
  __REG32 PPCM7         : 1;
  __REG32 PPCM8         : 1;
  __REG32 PPCM9         : 1;
  __REG32 PPCM10        : 1;
  __REG32 PPCM11        : 1;
  __REG32 PPCM12        : 1;
  __REG32 PPCM13        : 1;
  __REG32 PPCM14        : 1;
  __REG32 PPCM15        : 1;
} __hcrhdescriptorb_bits;

/* OHCI Root Hub Status Register (HCRHSTATUS) */
typedef struct {
  __REG32 LPS           : 1;
  __REG32 OCI           : 1;
  __REG32               :13;
  __REG32 DRWE          : 1;
  __REG32 LPSC          : 1;
  __REG32 OCIC          : 1;
  __REG32               :13;
  __REG32 CRWE          : 1;
} __hcrhstatus_bits;

/* OHCI Port 1/2 Status and Control Register (HCRHPORTSTATUS1/2) */
typedef struct {
  __REG32 CCS_CPE       : 1;
  __REG32 PES_SPE       : 1;
  __REG32 PSS_SPS       : 1;
  __REG32 POCI_CSS      : 1;
  __REG32 PRS_SPR       : 1;
  __REG32               : 3;
  __REG32 PPS_SPP       : 1;
  __REG32 LSDA_CPP      : 1;
  __REG32               : 6;
  __REG32 CSC           : 1;
  __REG32 PESC          : 1;
  __REG32 PSSC          : 1;
  __REG32 OCIC          : 1;
  __REG32 PRSC          : 1;
  __REG32               :11;
} __hcrhportstatus_bits;

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

/* USB Hardware Version Register (HWVERS) */
typedef struct {
  __REG16 REVMIN                :10;
  __REG16 REVMAJ                : 5;
  __REG16 RC                    : 1;
} __usb_hwvers_bits;

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

/* DDR2/mDDR SDRAM Status Register (SDRSTAT) */
typedef struct {
  __REG32                         : 1;
  __REG32 PHYRDY                  : 1;
  __REG32                         :28;
  __REG32 DUALCLK                 : 1;
  __REG32                         : 1;
} __ddr_sdrstat_bits;

/* DDR2/mDDR SDRAM Configuration Register (SDCR) */
typedef struct {
  __REG32 PAGESIZE                : 3;
  __REG32                         : 1;
  __REG32 IBANK                   : 3;
  __REG32                         : 2;
  __REG32 CL                      : 3;
  __REG32                         : 2;
  __REG32 NM                      : 1;
  __REG32 TIMUNLOCK               : 1;
  __REG32 SDRAMEN                 : 1;
  __REG32 DDREN                   : 1;
  __REG32 DDRDRIVE0               : 1;
  __REG32 DDRDLL_DIS              : 1;
  __REG32 DDR2EN                  : 1;
  __REG32 DDR2TERM0               : 1;
  __REG32 DDR2DDQS                : 1;
  __REG32 BOOTUNLOCK              : 1;
  __REG32 DDRDRIVE1               : 1;
  __REG32 MSDRAMEN                : 1;
  __REG32 IBANK_POS               : 1;
  __REG32 DDR2TERM1               : 1;
  __REG32                         : 4;
} __ddr_sdcr_bits;

/* DDR2/mDDR SDRAM Refresh Control Register (SDRCR) */
typedef struct {
  __REG32 RR                      :16;
  __REG32                         : 7;
  __REG32 SR_PD                   : 1;
  __REG32                         : 6;
  __REG32 MCLKSTOPEN              : 1;
  __REG32 LPMODEN                 : 1;
} __ddr_sdrcr_bits;

/* DDR2/mDDR SDRAM Timing Register 1 (SDTIMR1) */
typedef struct {
  __REG32 T_WTR                   : 2;
  __REG32                         : 1;
  __REG32 T_RRD                   : 3;
  __REG32 T_RC                    : 5;
  __REG32 T_RAS                   : 5;
  __REG32 T_WR                    : 3;
  __REG32 T_RCD                   : 3;
  __REG32 T_RP                    : 3;
  __REG32 T_RFC                   : 7;
} __ddr_sdtimr1_bits;

/* DDR2/mDDR SDRAM Timing Register 2 (SDTIMR2) */
typedef struct {
  __REG32 T_CKE                   : 5;
  __REG32 T_RTP                   : 3;
  __REG32 T_XSRD                  : 8;
  __REG32 T_XSNR                  : 7;
  __REG32 T_ODT                   : 2;
  __REG32 T_XP                    : 2;
  __REG32 T_RASMAX                : 4;
  __REG32                         : 1;
} __ddr_sdtimr2_bits;

/* DDR2/mDDR SDRAM Configuration Register 2 (SDCR2) */
typedef struct {
  __REG32 ROWSIZE                 : 3;
  __REG32                         :13;
  __REG32 PASR                    : 3;
  __REG32                         :13;
} __ddr_sdcr2_bits;

/* DDR2/mDDR Peripheral Bus Burst Priority Register (PBBPR) */
typedef struct {
  __REG32 PR_OLD_COUNT            : 8;
  __REG32                         :24;
} __ddr_pbbpr_bits;

/* DDR2/mDDR Performance Counter Configuration Register (PCC) */
typedef struct {
  __REG32 CNTR1_CFG               : 4;
  __REG32                         :10;
  __REG32 CNTR1_REGION_EN         : 1;
  __REG32 CNTR1_MSTID_EN          : 1;
  __REG32 CNTR2_CFG               : 4;
  __REG32                         :10;
  __REG32 CNTR2_REGION_EN         : 1;
  __REG32 CNTR2_MSTID_EN          : 1;
} __ddr_pcc_bits;

/* DDR2/mDDR Performance Counter Master Region Select Register (PCMRS) */
typedef struct {
  __REG32 REGION_SEL1             : 4;
  __REG32                         : 4;
  __REG32 MST_ID1                 : 8;
  __REG32 REGION_SEL2             : 4;
  __REG32                         : 4;
  __REG32 MST_ID2                 : 8;
} __ddr_pcmrs_bits;

/* DDR2/mDDR Interrupt Raw Register (IRR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LT                      : 1;
  __REG32                         :29;
} __ddr_irr_bits;

/* DDR2/mDDR Interrupt Masked Register (IMR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LTM                     : 1;
  __REG32                         :29;
} __ddr_imr_bits;

/* DDR2/mDDR Interrupt Mask Set Register (IMSR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LTMSET                  : 1;
  __REG32                         :29;
} __ddr_imsr_bits;

/* DDR2/mDDR Interrupt Mask Clear Register (IMCR) */
typedef struct {
  __REG32                         : 2;
  __REG32 LTMCLR                  : 1;
  __REG32                         :29;
} __ddr_imcr_bits;

/* DDR2/mDDR PHY Control Register 1 (DRPYC1R) */
typedef struct {
  __REG32 RL                      : 3;
  __REG32                         : 3;
  __REG32 PWRDNEN                 : 1;
  __REG32 EXT_STRBEN              : 1;
  __REG32                         :24;
} __ddr_drpyc1r_bits;

/* DDR2/mDDR VTP IO Control Register (VTPIO_CTL) */
typedef struct {
  __REG32 F2                      : 1;
  __REG32 F1                      : 1;
  __REG32 F0                      : 1;
  __REG32 D2                      : 1;
  __REG32 D1                      : 1;
  __REG32 D0                      : 1;
  __REG32 POWERDN                 : 1;
  __REG32 LOCK                    : 1;
  __REG32 PWRSAVE                 : 1;
  __REG32 FORCEUPN                : 1;
  __REG32 FORCEUPP                : 1;
  __REG32 FORCEDNN                : 1;
  __REG32 FORCEDNP                : 1;
  __REG32 CLKRZ                   : 1;
  __REG32 IOPWRDN                 : 1;
  __REG32 READY                   : 1;
  __REG32 VREFTAP                 : 2;
  __REG32 VREFEN                  : 1;
  __REG32                         :13;
} __vtpio_ctl_bits;

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

/* SATA HBA Capabilities Register (CAP) */
typedef struct {
  __REG32 NP                      : 5;
  __REG32 SXS                     : 1;
  __REG32 EMS                     : 1;
  __REG32 CCCS                    : 1;
  __REG32 NCS                     : 5;
  __REG32 PSC                     : 1;
  __REG32 SSC                     : 1;
  __REG32 PMD                     : 1;
  __REG32                         : 1;
  __REG32 SPM                     : 1;
  __REG32 SAM                     : 1;
  __REG32 SNZO                    : 1;
  __REG32 ISS                     : 4;
  __REG32 SCLO                    : 1;
  __REG32 SAL                     : 1;
  __REG32 SALP                    : 1;
  __REG32 SSS                     : 1;
  __REG32 SMPS                    : 1;
  __REG32 SSNTF                   : 1;
  __REG32 SNCQ                    : 1;
  __REG32 S64A                    : 1;
} __sata_cap_bits;

/* SATA Global HBA Control Register (GHC) */
typedef struct {
  __REG32 HR                      : 1;
  __REG32 IE                      : 1;
  __REG32                         :29;
  __REG32 AE                      : 1;
} __sata_ghc_bits;

/* SATA Interrupt Status Register (IS) */
typedef struct {
  __REG32 IPS0                    : 1;
  __REG32 IPS1                    : 1;
  __REG32                         :30;
} __sata_is_bits;

/* SATA Ports Implemented Register (PI) */
typedef struct {
  __REG32 PI0                     : 1;
  __REG32 PI1                     : 1;
  __REG32                         :30;
} __sata_pi_bits;

/* SATA AHCI Version Register (VS) */
typedef struct {
  __REG32 MNR                     :16;
  __REG32 MJR                     :16;
} __sata_vs_bits;

/* SATA Command Completion Coalescing Control Register (CCC_CTL) */
typedef struct {
  __REG32 EN                      : 1;
  __REG32                         : 2;
  __REG32 INT                     : 5;
  __REG32 CC                      : 8;
  __REG32 TV                      :16;
} __sata_ccc_ctl_bits;

/* SATA Command Completion Coalescing Ports Register (CCC_PORTS) */
typedef struct {
  __REG32 PRT0                    : 1;
  __REG32 PRT1                    : 1;
  __REG32 PRT2                    : 1;
  __REG32 PRT3                    : 1;
  __REG32 PRT4                    : 1;
  __REG32 PRT5                    : 1;
  __REG32 PRT6                    : 1;
  __REG32 PRT7                    : 1;
  __REG32 PRT8                    : 1;
  __REG32 PRT9                    : 1;
  __REG32 PRT10                   : 1;
  __REG32 PRT11                   : 1;
  __REG32 PRT12                   : 1;
  __REG32 PRT13                   : 1;
  __REG32 PRT14                   : 1;
  __REG32 PRT15                   : 1;
  __REG32 PRT16                   : 1;
  __REG32 PRT17                   : 1;
  __REG32 PRT18                   : 1;
  __REG32 PRT19                   : 1;
  __REG32 PRT20                   : 1;
  __REG32 PRT21                   : 1;
  __REG32 PRT22                   : 1;
  __REG32 PRT23                   : 1;
  __REG32 PRT24                   : 1;
  __REG32 PRT25                   : 1;
  __REG32 PRT26                   : 1;
  __REG32 PRT27                   : 1;
  __REG32 PRT28                   : 1;
  __REG32 PRT29                   : 1;
  __REG32 PRT30                   : 1;
  __REG32 PRT31                   : 1;
} __sata_ccc_ports_bits;

/* SATA BIST Active FIS Register (BISTAFR) */
typedef struct {
  __REG32 PD                      : 8;
  __REG32 NCP                     : 8;
  __REG32                         :16;
} __sata_bistafr_bits;

/* SATA BIST Control Register (BISTCR) */
typedef struct {
  __REG32 PATTERN                 : 4;
  __REG32 PV                      : 1;
  __REG32 FLIP                    : 1;
  __REG32 ERREN                   : 1;
  __REG32                         : 1;
  __REG32 LLC                     : 3;
  __REG32                         : 5;
  __REG32 NEALB                   : 1;
  __REG32 CNTCLR                  : 1;
  __REG32 TXO                     : 1;
  __REG32                         :13;
} __sata_bistcr_bits;

/* SATA BIST Status Register (BISTSR) */
typedef struct {
  __REG32 FRAMERR                 :16;
  __REG32 BRSTERR                 : 8;
  __REG32                         : 8;
} __sata_bistsr_bits;

/* SATA BIST DWORD Error Count Register (TIMER1MS) */
typedef struct {
  __REG32 TIMV                    :20;
  __REG32                         :12;
} __sata_timer1ms_bits;

/* SATA Global Parameter 1 Register (GPARAM1R) */
typedef struct {
  __REG32 M_HDATA                 : 3;
  __REG32 S_HDATA                 : 3;
  __REG32 M_HADDR                 : 1;
  __REG32 S_HADDR                 : 1;
  __REG32 AHB_ENDIA               : 2;
  __REG32 RETURN_ERR              : 1;
  __REG32                         : 1;
  __REG32 PHY_TYPE                : 1;
  __REG32 BIST_M                  : 1;
  __REG32 LATCH_M                 : 1;
  __REG32 PHY_STAT                : 6;
  __REG32 PHY_CTRL                : 6;
  __REG32 PHY_RST                 : 1;
  __REG32 PHY_DATA                : 2;
  __REG32 RX_BUFFER               : 1;
  __REG32 ALIGN_M                 : 1;
} __sata_gparam1r_bits;

/* SATA Global Parameter 2 Register (GPARAM2R) */
typedef struct {
  __REG32 RXOOB_CLK               : 9;
  __REG32 TX_OOB_M                : 1;
  __REG32 RXOOB_M                 : 1;
  __REG32 RXOOB_CLK_M             : 1;
  __REG32 ENCODE_M                : 1;
  __REG32 DEV_MP                  : 1;
  __REG32 DEV_CP                  : 1;
  __REG32                         :17;
} __sata_gparam2r_bits;

/* SATA Port Parameter Register (PPARAMR) */
typedef struct {
  __REG32 RX_FIFO_DEPTH           : 3;
  __REG32 X_FIFO_DEPTH            : 3;
  __REG32 RX_MEM_S                : 1;
  __REG32 RX_MEM_M                : 1;
  __REG32 TX_MEM_S                : 1;
  __REG32 TX_MEM_M                : 1;
  __REG32                         :22;
} __sata_pparamr_bits;

/* SATA Test Register (TESTR) */
typedef struct {
  __REG32 TEST_IF                 : 1;
  __REG32                         :15;
  __REG32 PSEL                    : 3;
  __REG32                         :13;
} __sata_testr_bits;

/* SATA Port Interrupt Status Register (P0IS) */
typedef struct {
  __REG32 DHRS                    : 1;
  __REG32 PSS                     : 1;
  __REG32 DSS                     : 1;
  __REG32 SDBS                    : 1;
  __REG32 UFS                     : 1;
  __REG32 DPS                     : 1;
  __REG32 PCS                     : 1;
  __REG32 DMPS                    : 1;
  __REG32                         :14;
  __REG32 PRCS                    : 1;
  __REG32 IPMS                    : 1;
  __REG32 OFS                     : 1;
  __REG32                         : 1;
  __REG32 INFS                    : 1;
  __REG32 IFS                     : 1;
  __REG32 HBDS                    : 1;
  __REG32 HBFS                    : 1;
  __REG32 TFES                    : 1;
  __REG32 CPDS                    : 1;
} __sata_p0is_bits;

/* SATA Port Interrupt Enable Register (P0IE) */
typedef struct {
  __REG32 DHRE                    : 1;
  __REG32 PSE                     : 1;
  __REG32 DSE                     : 1;
  __REG32 SDBE                    : 1;
  __REG32 UFE                     : 1;
  __REG32 DPE                     : 1;
  __REG32 PCE                     : 1;
  __REG32 DMPE                    : 1;
  __REG32                         :14;
  __REG32 PRCE                    : 1;
  __REG32 IPME                    : 1;
  __REG32 OFE                     : 1;
  __REG32                         : 1;
  __REG32 INFE                    : 1;
  __REG32 IFE                     : 1;
  __REG32 HBDE                    : 1;
  __REG32 HBFE                    : 1;
  __REG32 TFEE                    : 1;
  __REG32 CPDE                    : 1;
} __sata_p0ie_bits;

/* SATA Port Command Register (P0CMD) */
typedef struct {
  __REG32 ST                      : 1;
  __REG32 SUD                     : 1;
  __REG32 POD                     : 1;
  __REG32 CLO                     : 1;
  __REG32 FRE                     : 1;
  __REG32                         : 3;
  __REG32 CCS                     : 5;
  __REG32 MPSS                    : 1;
  __REG32 FR                      : 1;
  __REG32 CR                      : 1;
  __REG32 CPS                     : 1;
  __REG32 PMA                     : 1;
  __REG32 HPCP                    : 1;
  __REG32 MPSP                    : 1;
  __REG32 CPD                     : 1;
  __REG32 ESP                     : 1;
  __REG32                         : 2;
  __REG32 ATAPI                   : 1;
  __REG32 DLAE                    : 1;
  __REG32 ALPE                    : 1;
  __REG32 ASP                     : 1;
  __REG32                         : 4;
} __sata_p0cmd_bits;

/* SATA Port Task File Data Register (P0TFD) */
typedef struct {
  __REG32 STS                     : 8;
  __REG32 ERR                     : 8;
  __REG32                         :16;
} __sata_p0tfd_bits;

/* SATA Port Serial ATA Status (SStatus) Register (P0SSTS) */
typedef struct {
  __REG32 DET                     : 4;
  __REG32 SPD                     : 4;
  __REG32 IPM                     : 4;
  __REG32                         :20;
} __sata_p0ssts_bits;

/* SATA Port Serial ATA Error (SError) Register (P0SERR) */
typedef struct {
  __REG32 ERR_I                   : 1;
  __REG32 ERR_M                   : 1;
  __REG32                         : 6;
  __REG32 ERR_T                   : 1;
  __REG32 ERR_C                   : 1;
  __REG32 ERR_P                   : 1;
  __REG32 ERR_E                   : 1;
  __REG32                         : 4;
  __REG32 DIAG_N                  : 1;
  __REG32 DIAG_I                  : 1;
  __REG32 DIAG_W                  : 1;
  __REG32 DIAG_B                  : 1;
  __REG32 DIAG_D                  : 1;
  __REG32 DIAG_C                  : 1;
  __REG32 DIAG_H                  : 1;
  __REG32 DIAG_S                  : 1;
  __REG32 DIAG_T                  : 1;
  __REG32 DIAG_F                  : 1;
  __REG32 DIAG_X                  : 1;
  __REG32                         : 5;
} __sata_p0serr_bits;

/* SATA Port Serial ATA Active Register (P0SACT) */
typedef struct {
  __REG32 DS0                     : 1;
  __REG32 DS1                     : 1;
  __REG32 DS2                     : 1;
  __REG32 DS3                     : 1;
  __REG32 DS4                     : 1;
  __REG32 DS5                     : 1;
  __REG32 DS6                     : 1;
  __REG32 DS7                     : 1;
  __REG32 DS8                     : 1;
  __REG32 DS9                     : 1;
  __REG32 DS10                    : 1;
  __REG32 DS11                    : 1;
  __REG32 DS12                    : 1;
  __REG32 DS13                    : 1;
  __REG32 DS14                    : 1;
  __REG32 DS15                    : 1;
  __REG32 DS16                    : 1;
  __REG32 DS17                    : 1;
  __REG32 DS18                    : 1;
  __REG32 DS19                    : 1;
  __REG32 DS20                    : 1;
  __REG32 DS21                    : 1;
  __REG32 DS22                    : 1;
  __REG32 DS23                    : 1;
  __REG32 DS24                    : 1;
  __REG32 DS25                    : 1;
  __REG32 DS26                    : 1;
  __REG32 DS27                    : 1;
  __REG32 DS28                    : 1;
  __REG32 DS29                    : 1;
  __REG32 DS30                    : 1;
  __REG32 DS31                    : 1;
} __sata_p0sact_bits;

/* SATA Port Command Issue Register (P0CI) */
typedef struct {
  __REG32 CI0                     : 1;
  __REG32 CI1                     : 1;
  __REG32 CI2                     : 1;
  __REG32 CI3                     : 1;
  __REG32 CI4                     : 1;
  __REG32 CI5                     : 1;
  __REG32 CI6                     : 1;
  __REG32 CI7                     : 1;
  __REG32 CI8                     : 1;
  __REG32 CI9                     : 1;
  __REG32 CI10                    : 1;
  __REG32 CI11                    : 1;
  __REG32 CI12                    : 1;
  __REG32 CI13                    : 1;
  __REG32 CI14                    : 1;
  __REG32 CI15                    : 1;
  __REG32 CI16                    : 1;
  __REG32 CI17                    : 1;
  __REG32 CI18                    : 1;
  __REG32 CI19                    : 1;
  __REG32 CI20                    : 1;
  __REG32 CI21                    : 1;
  __REG32 CI22                    : 1;
  __REG32 CI23                    : 1;
  __REG32 CI24                    : 1;
  __REG32 CI25                    : 1;
  __REG32 CI26                    : 1;
  __REG32 CI27                    : 1;
  __REG32 CI28                    : 1;
  __REG32 CI29                    : 1;
  __REG32 CI30                    : 1;
  __REG32 CI31                    : 1;
} __sata_p0ci_bits;

/* SATA Port Serial ATA Notification Register (P0SNTF) */
typedef struct {
  __REG32 PMN0                    : 1;
  __REG32 PMN1                    : 1;
  __REG32 PMN2                    : 1;
  __REG32 PMN3                    : 1;
  __REG32 PMN4                    : 1;
  __REG32 PMN5                    : 1;
  __REG32 PMN6                    : 1;
  __REG32 PMN7                    : 1;
  __REG32 PMN8                    : 1;
  __REG32 PMN9                    : 1;
  __REG32 PMN10                   : 1;
  __REG32 PMN11                   : 1;
  __REG32 PMN12                   : 1;
  __REG32 PMN13                   : 1;
  __REG32 PMN14                   : 1;
  __REG32 PMN15                   : 1;
  __REG32                         :16;
} __sata_p0sntf_bits;

/* SATA Port DMA Control Register (P0DMACR) */
typedef struct {
  __REG32 TXTS                    : 4;
  __REG32 RXTS                    : 4;
  __REG32 TXABL                   : 4;
  __REG32 RXABL                   : 4;
  __REG32                         :16;
} __sata_p0dmacr_bits;

/* SATA Port PHY Control Register (P0PHYCR) */
typedef struct {
  __REG32 MPY                     : 4;
  __REG32 LB                      : 2;
  __REG32 LOS                     : 1;
  __REG32 RXINVPAIR               : 1;
  __REG32 RXTERM                  : 2;
  __REG32 RXCDR                   : 3;
  __REG32 RXEQ                    : 4;
  __REG32 TXINVPAIR               : 1;
  __REG32 TXCM                    : 1;
  __REG32 TXSWING                 : 3;
  __REG32 TXDE                    : 4;
  __REG32                         : 4;
  __REG32 OVERRIDE                : 1;
  __REG32 ENPLL                   : 1;
} __sata_p0phycr_bits;

/* SATA Port PHY Status Register (P0PHYSR) */
typedef struct {
  __REG32 LOCK                    : 1;
  __REG32 SIGDET                  : 1;
  __REG32                         :30;
} __sata_p0physr_bits;

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

/* McBSP Serial Port Control Register (SPCR) */
typedef struct {
  __REG32 RRST                    : 1;
  __REG32 RRDY                    : 1;
  __REG32 RFULL                   : 1;
  __REG32 RSYNCERR                : 1;
  __REG32 RINTM                   : 2;
  __REG32                         : 1;
  __REG32 DXENA                   : 1;
  __REG32                         : 3;
  __REG32 CLKSTP                  : 2;
  __REG32 RJUST                   : 2;
  __REG32 DLB                     : 1;
  __REG32 XRST                    : 1;
  __REG32 XRDY                    : 1;
  __REG32 XEMPTY                  : 1;
  __REG32 XSYNCERR                : 1;
  __REG32 XINTM                   : 2;
  __REG32 GRST                    : 1;
  __REG32 FRST                    : 1;
  __REG32 SOFT                    : 1;
  __REG32 FREE                    : 1;
  __REG32                         : 6;
} __mcbsp_spcr_bits;

/* McBSP Receive Control Register (RCR) */
typedef struct {
  __REG32                         : 4;
  __REG32 RWDREVRS                : 1;
  __REG32 RWDLEN1                 : 3;
  __REG32 RFRLEN1                 : 7;
  __REG32                         : 1;
  __REG32 RDATDLY                 : 2;
  __REG32 RFIG                    : 1;
  __REG32 RCOMPAND                : 2;
  __REG32 RWDLEN2                 : 3;
  __REG32 RFRLEN2                 : 7;
  __REG32 RPHASE                  : 1;
} __mcbsp_rcr_bits;

/* McBSP Transmit Control Register (XCR) */
typedef struct {
  __REG32                         : 4;
  __REG32 XWDREVRS                : 1;
  __REG32 XWDLEN1                 : 3;
  __REG32 XFRLEN1                 : 7;
  __REG32                         : 1;
  __REG32 XDATDLY                 : 2;
  __REG32 XFIG                    : 1;
  __REG32 XCOMPAND                : 2;
  __REG32 XWDLEN2                 : 3;
  __REG32 XFRLEN2                 : 7;
  __REG32 XPHASE                  : 1;
} __mcbsp_xcr_bits;

/* McBSP Sample Rate Generator Register (SRGR) */
typedef struct {
  __REG32 CLKGDV                  : 8;
  __REG32 FWID                    : 8;
  __REG32 FPER                    :12;
  __REG32 FSGM                    : 1;
  __REG32 CLKSM                   : 1;
  __REG32 CLKSP                   : 1;
  __REG32 GSYNC                   : 1;
} __mcbsp_srgr_bits;

/* McBSP Multichannel Control Register (MCR) */
typedef struct {
  __REG32 RMCM                    : 1;
  __REG32                         : 1;
  __REG32 RCBLK                   : 3;
  __REG32 RPABLK                  : 2;
  __REG32 RPBBLK                  : 2;
  __REG32 RMCME                   : 1;
  __REG32                         : 6;
  __REG32 XMCM                    : 2;
  __REG32 XCBLK                   : 3;
  __REG32 XPABLK                  : 2;
  __REG32 XPBBLK                  : 2;
  __REG32 XMCME                   : 1;
  __REG32                         : 6;
} __mcbsp_mcr_bits;

/* McBSP Enhanced Receive Channel Enable Registers (RCERE0-RCERE3) */
typedef struct {
  __REG32 RCE0                    : 1;
  __REG32 RCE1                    : 1;
  __REG32 RCE2                    : 1;
  __REG32 RCE3                    : 1;
  __REG32 RCE4                    : 1;
  __REG32 RCE5                    : 1;
  __REG32 RCE6                    : 1;
  __REG32 RCE7                    : 1;
  __REG32 RCE8                    : 1;
  __REG32 RCE9                    : 1;
  __REG32 RCE10                   : 1;
  __REG32 RCE11                   : 1;
  __REG32 RCE12                   : 1;
  __REG32 RCE13                   : 1;
  __REG32 RCE14                   : 1;
  __REG32 RCE15                   : 1;
  __REG32 RCE16                   : 1;
  __REG32 RCE17                   : 1;
  __REG32 RCE18                   : 1;
  __REG32 RCE19                   : 1;
  __REG32 RCE20                   : 1;
  __REG32 RCE21                   : 1;
  __REG32 RCE22                   : 1;
  __REG32 RCE23                   : 1;
  __REG32 RCE24                   : 1;
  __REG32 RCE25                   : 1;
  __REG32 RCE26                   : 1;
  __REG32 RCE27                   : 1;
  __REG32 RCE28                   : 1;
  __REG32 RCE29                   : 1;
  __REG32 RCE30                   : 1;
  __REG32 RCE31                   : 1;
} __mcbsp_rcere_bits;

/* McBSP Enhanced Transmit Channel Enable Registers (XCERE0-XCERE3) */
typedef struct {
  __REG32 XCE0                    : 1;
  __REG32 XCE1                    : 1;
  __REG32 XCE2                    : 1;
  __REG32 XCE3                    : 1;
  __REG32 XCE4                    : 1;
  __REG32 XCE5                    : 1;
  __REG32 XCE6                    : 1;
  __REG32 XCE7                    : 1;
  __REG32 XCE8                    : 1;
  __REG32 XCE9                    : 1;
  __REG32 XCE10                   : 1;
  __REG32 XCE11                   : 1;
  __REG32 XCE12                   : 1;
  __REG32 XCE13                   : 1;
  __REG32 XCE14                   : 1;
  __REG32 XCE15                   : 1;
  __REG32 XCE16                   : 1;
  __REG32 XCE17                   : 1;
  __REG32 XCE18                   : 1;
  __REG32 XCE19                   : 1;
  __REG32 XCE20                   : 1;
  __REG32 XCE21                   : 1;
  __REG32 XCE22                   : 1;
  __REG32 XCE23                   : 1;
  __REG32 XCE24                   : 1;
  __REG32 XCE25                   : 1;
  __REG32 XCE26                   : 1;
  __REG32 XCE27                   : 1;
  __REG32 XCE28                   : 1;
  __REG32 XCE29                   : 1;
  __REG32 XCE30                   : 1;
  __REG32 XCE31                   : 1;
} __mcbsp_xcere_bits;

/* McBSP Pin Control Register (PCR) */
typedef struct {
  __REG32 CLKRP                   : 1;
  __REG32 CLKXP                   : 1;
  __REG32 FSRP                    : 1;
  __REG32 FSXP                    : 1;
  __REG32                         : 3;
  __REG32 SCLKME                  : 1;
  __REG32 CLKRM                   : 1;
  __REG32 CLKXM                   : 1;
  __REG32 FSRM                    : 1;
  __REG32 FSXM                    : 1;
  __REG32                         :20;
} __mcbsp_pcr_bits;

/* McBSP Write FIFO Control Register (WFIFOCTL) */
typedef struct {
  __REG32 WNUMDMA                 : 8;
  __REG32 WNUMEVT                 : 8;
  __REG32 WENA                    : 1;
  __REG32                         :15;
} __mcbsp_wfifoctl_bits;

/* McBSP Write FIFO Status Register (WFIFOSTS) */
typedef struct {
  __REG32 WLVL                    : 8;
  __REG32                         :24;
} __mcbsp_wfifosts_bits;

/* McBSP Read FIFO Control Register (RFIFOCTL) */
typedef struct {
  __REG32 RNUMDMA                 : 8;
  __REG32 RNUMEVT                 : 8;
  __REG32 RENA                    : 1;
  __REG32                         :15;
} __mcbsp_rfifoctl_bits;

/* McBSP Write FIFO Status Register (WFIFOSTS) */
typedef struct {
  __REG32 RLVL                    : 8;
  __REG32                         :24;
} __mcbsp_rfifosts_bits;

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

/* UART Revision Identification Register 2 (REVID2) */
typedef struct {
  __REG32 REVID2                  : 8;
  __REG32                         :24;
} __uartrevid2_bits;

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

/* EMAC LCD Control Register (LCD_CTRL) */
typedef struct {
  __REG32 MODESEL                 : 1;
  __REG32                         : 7;
  __REG32 CLKDIV                  : 8;
  __REG32                         :16;
} __lcdc_ctrl_bits;

/* LCD Status Register (LCD_STAT) */
typedef struct {
  __REG32 DONE                    : 1;
  __REG32                         : 1;
  __REG32 SYNC                    : 1;
  __REG32 ABC                     : 1;
  __REG32                         : 1;
  __REG32 FUF                     : 1;
  __REG32 PL                      : 1;
  __REG32                         : 1;
  __REG32 EOF0                    : 1;
  __REG32 EOF1                    : 1;
  __REG32                         :22;
} __lcdc_stat_bits;

/* LCD LIDD Control Register (LIDD_CTRL) */
typedef struct {
  __REG32 LIDD_MODE_SEL           : 3;
  __REG32 ALEPOL                  : 1;
  __REG32 RS_EN_POL               : 1;
  __REG32 WS_DIR_POL              : 1;
  __REG32 CS0_E0_POL              : 1;
  __REG32 CS1_E1_POL              : 1;
  __REG32 LIDD_DMA_EN             : 1;
  __REG32 DMA_CS0_CS1             : 1;
  __REG32 DONE_INT_EN             : 1;
  __REG32                         :21;
} __lidd_ctrl_bits;

/* LCD LIDD CSn Configuration Registers (LIDD_CS0_CONF and LIDD_CS1_CONF) */
typedef struct {
  __REG32 TA                      : 2;
  __REG32 R_HOLD                  : 4;
  __REG32 R_STROBE                : 6;
  __REG32 R_SU                    : 5;
  __REG32 W_HOLD                  : 4;
  __REG32 W_STROBE                : 6;
  __REG32 W_SU                    : 5;
} __lcdc_cs_conf_bits;

/* LCD LIDD CSn Address Read/Write Registers (LIDD_CS0_ADDR and LIDD_CS1_ADDR) */
typedef struct {
  __REG32 ADR_INDX                :16;
  __REG32                         :16;
} __lcdc_cs_addr_bits;

/* LCD LIDD CSn Data Read/Write Register (LIDD_CSn_DATA) */
typedef struct {
  __REG32 DATA                    :16;
  __REG32                         :16;
} __lcdc_cs_data_bits;

/* LCD Raster Control Register (RASTER_CTRL) */
typedef struct {
  __REG32 RASTER_EN               : 1;
  __REG32 MONO_COLOR              : 1;
  __REG32 AC_EN                   : 1;
  __REG32 DONE_EN                 : 1;
  __REG32 PL_EN                   : 1;
  __REG32 SL_EN                   : 1;
  __REG32 FUF_EN                  : 1;
  __REG32 TFT_STN                 : 1;
  __REG32 RD_ORDER                : 1;
  __REG32 MONO8B                  : 1;
  __REG32                         : 2;
  __REG32 FIFO_DMA_DELAY          : 8;
  __REG32 PLM                     : 2;
  __REG32 NIB_MODE                : 1;
  __REG32 TFT_ALT_MAP             : 1;
  __REG32 STN_565                 : 1;
  __REG32                         : 7;
} __lcdc_raster_ctrl_bits;

/* LCD Raster Timing Register 0 (RASTER_TIMING_0) */
typedef struct {
  __REG32                         : 4;
  __REG32 PPL                     : 6;
  __REG32 HSW                     : 6;
  __REG32 HFP                     : 8;
  __REG32 HBP                     : 8;
} __lcdc_raster_timing_0_bits;

/* LCD Raster Timing Register 1 (RASTER_TIMING_1) */
typedef struct {
  __REG32 LPP                     :10;
  __REG32 VSW                     : 6;
  __REG32 VFP                     : 8;
  __REG32 VBP                     : 8;
} __lcdc_raster_timing_1_bits;

/* LCD Raster Timing Register 2 (RASTER_TIMING_2) */
typedef struct {
  __REG32                         : 8;
  __REG32 ACB                     : 8;
  __REG32 ACB_I                   : 4;
  __REG32 IVS                     : 1;
  __REG32 IHS                     : 1;
  __REG32 IPC                     : 1;
  __REG32 BIAS                    : 1;
  __REG32 SYNC_EDGE               : 1;
  __REG32 SYNC_CTRL               : 1;
  __REG32                         : 6;
} __lcdc_raster_timing_2_bits;

/* LCD Raster Subpanel Display Register (RASTER_SUBPANEL) */
typedef struct {
  __REG32                         : 4;
  __REG32 DPD                     :12;
  __REG32 LPPT                    :10;
  __REG32                         : 3;
  __REG32 HOLS                    : 1;
  __REG32                         : 1;
  __REG32 SPEN                    : 1;
} __lcdc_raster_subpanel_bits;

/* LCD DMA Control Register (LCDDMA_CTRL) */
typedef struct {
  __REG32 FRAME_MODE              : 1;
  __REG32 BIGENDIAN               : 1;
  __REG32 EOF_INTEN               : 1;
  __REG32                         : 1;
  __REG32 BURST_SIZE              : 3;
  __REG32                         : 1;
  __REG32 TH_FIFO_READY           : 3;
  __REG32                         :21;
} __lcdc_dma_ctrl_bits;

/* HPI Power and Emulation Management Register (PWREMU_MGMT) */
typedef struct {
  __REG32 FREE                    : 1;
  __REG32 SOFT                    : 1;
  __REG32                         :30;
} __hpipwremu_mgmt_bits;

/* HPI GPIO Enable Register (GPIO_EN) */
typedef struct {
  __REG32 GPIOEN0                 : 1;
  __REG32 GPIOEN1                 : 1;
  __REG32 GPIOEN2                 : 1;
  __REG32                         : 1;
  __REG32 GPIOEN4                 : 1;
  __REG32 GPIOEN5                 : 1;
  __REG32 GPIOEN6                 : 1;
  __REG32 GPIOEN7                 : 1;
  __REG32 GPIOEN8                 : 1;
  __REG32                         :23;
} __hpigpio_en_bits;

/* HPI GPIO Direction 1 Register (GPIO_DIR1) */
/* HPI GPIO Data 1 Register (GPIO_DAT1) */
typedef struct {
  __REG32 HD0                     : 1;
  __REG32 HD1                     : 1;
  __REG32 HD2                     : 1;
  __REG32 HD3                     : 1;
  __REG32 HD4                     : 1;
  __REG32 HD5                     : 1;
  __REG32 HD6                     : 1;
  __REG32 HD7                     : 1;
  __REG32 HD8                     : 1;
  __REG32 HD9                     : 1;
  __REG32 HD10                    : 1;
  __REG32 HD11                    : 1;
  __REG32 HD12                    : 1;
  __REG32 HD13                    : 1;
  __REG32 HD14                    : 1;
  __REG32 HD15                    : 1;
  __REG32                         :16;
} __hpigpio_dir1_bits;

/* HPI GPIO Direction 2 Register (GPIO_DIR2) */
/* HPI GPIO Data 2 Register (GPIO_DAT2) */
typedef struct {
  __REG32 HASZ                    : 1;
  __REG32 HCSZ                    : 1;
  __REG32 HDS1Z                   : 1;
  __REG32 HDS2Z                   : 1;
  __REG32 HRW                     : 1;
  __REG32 HHWIL                   : 1;
  __REG32 HCNTL1                  : 1;
  __REG32 HCNTL0                  : 1;
  __REG32 HINTZ                   : 1;
  __REG32 HRDY                    : 1;
  __REG32                         :22;
} __hpigpio_dir2_bits;

/* HPI Host Port Interface Control Register (HPIC) */
typedef struct {
  __REG32 HWOB                    : 1;
  __REG32 DSPINT                  : 1;
  __REG32 HINT                    : 1;
  __REG32                         : 1;
  __REG32 FETCH                   : 1;
  __REG32                         : 2;
  __REG32 HPIRST                  : 1;
  __REG32 HWOBSTAT                : 1;
  __REG32 DUALHPIA                : 1;
  __REG32                         : 1;
  __REG32 HPIASEL                 : 1;
  __REG32                         :20;
} __hpic_bits;

/* VPIF Channel 0 Control Register (C0CTRL) */
typedef struct {
  __REG32 CHANEN                  : 1;
  __REG32                         : 1;
  __REG32 CAPMODE                 : 1;
  __REG32 YCMUX                   : 1;
  __REG32                         : 1;
  __REG32 FID                     : 1;
  __REG32 INTFRAME                : 2;
  __REG32 HANC                    : 1;
  __REG32 VANC                    : 1;
  __REG32 INTRPROG                : 1;
  __REG32                         : 1;
  __REG32 FIELDFRAME              : 1;
  __REG32 HVINV                   : 1;
  __REG32 VVINV                   : 1;
  __REG32 FIDINV                  : 1;
  __REG32 INTLINE                 :12;
  __REG32 DATAWIDTH               : 2;
  __REG32                         : 1;
  __REG32 CLKEDGE                 : 1;
} __vpif_c0ctrl_bits;

/* VPIF Channel 1 Control Register (C1CTRL) */
typedef struct {
  __REG32 CHANEN                  : 1;
  __REG32                         : 1;
  __REG32 CAPMODE                 : 1;
  __REG32 YCMUX                   : 1;
  __REG32                         : 1;
  __REG32 FID                     : 1;
  __REG32 INTFRAME                : 2;
  __REG32 HANC                    : 1;
  __REG32 VANC                    : 1;
  __REG32 INTRPROG                : 1;
  __REG32                         :20;
  __REG32 CLKEDGE                 : 1;
} __vpif_c1ctrl_bits;

/* VPIF Channel 2 Control Register (C2CTRL) */
typedef struct {
  __REG32 CHANEN                  : 1;
  __REG32 CLKEN                   : 1;
  __REG32                         : 1;
  __REG32 YCMUX                   : 1;
  __REG32                         : 1;
  __REG32 FID                     : 1;
  __REG32 INTFRAME                : 2;
  __REG32 HANC                    : 1;
  __REG32 VANC                    : 1;
  __REG32 PIXEL                   : 1;
  __REG32 INTRPROG                : 1;
  __REG32 FIELDFRAME              : 1;
  __REG32 CLIPVID                 : 1;
  __REG32 CLIPANC                 : 1;
  __REG32                         :16;
  __REG32 CLKEDGE                 : 1;
} __vpif_c2ctrl_bits;

/* VPIF Channel 3 Control Register (C3CTRL) */
typedef struct {
  __REG32 CHANEN                  : 1;
  __REG32 CLKEN                   : 1;
  __REG32                         : 1;
  __REG32 YCMUX                   : 1;
  __REG32                         : 1;
  __REG32 FID                     : 1;
  __REG32 INTFRAME                : 2;
  __REG32 HANC                    : 1;
  __REG32 VANC                    : 1;
  __REG32 PIXEL                   : 1;
  __REG32 INTRPROG                : 1;
  __REG32                         : 1;
  __REG32 CLIPVID                 : 1;
  __REG32 CLIPANC                 : 1;
  __REG32                         :16;
  __REG32 CLKEDGE                 : 1;
} __vpif_c3ctrl_bits;

/* VPIF Interrupt Enable Register (INTEN) */
/* VPIF Interrupt Enable Clear Register (INTCLR) */
/* VPIF Interrupt Status Register (INTSTAT) */
/* VPIF Interrupt Status Clear Register (INTSTATCLR) */
typedef struct {
  __REG32 FRAME0                  : 1;
  __REG32 FRAME1                  : 1;
  __REG32 FRAME2                  : 1;
  __REG32 FRAME3                  : 1;
  __REG32 ERROR                   : 1;
  __REG32                         :27;
} __vpif_inten_bits;

/* VPIF Interrupt Enable Set Register (INTSET) */
typedef struct {
  __REG32 FRAME0                  : 1;
  __REG32 FRAME1                  : 1;
  __REG32 FRAME2                  : 1;
  __REG32 FRAME3                  : 1;
  __REG32 ERROR1                  : 1;
  __REG32                         :27;
} __vpif_intenset_bits;

/* VPIF Emulation Suspend Control Register (EMUCTRL) */
typedef struct {
  __REG32 FREE                    : 1;
  __REG32                         :31;
} __vpif_emuctrl_bits;

/* VPIF DMA Size Control Register (REQSIZE) */
typedef struct {
  __REG32 BYTES                   : 9;
  __REG32                         :23;
} __vpif_reqsize_bits;

/* VPIF Channel n Horizontal Size Configuration Register (C0HCFG and C1HCFG) */
typedef struct {
  __REG32 SAV2EAV                 :13;
  __REG32                         : 3;
  __REG32 EAV2SAV                 :13;
  __REG32                         : 3;
} __vpif_chcfg_bits;

/* VPIF Channel n Vertical Size Configuration 0 Register (C0VCFG0 and C1VCFG0) */
typedef struct {
  __REG32 L3                      :12;
  __REG32                         : 4;
  __REG32 L1                      :12;
  __REG32                         : 4;
} __vpif_cvcfg0_bits;

/* VPIF Channel n Vertical Size Configuration 1 Register (C0VCFG1 and C1VCFG1) */
typedef struct {
  __REG32 L7                      :12;
  __REG32                         : 4;
  __REG32 L5                      :12;
  __REG32                         : 4;
} __vpif_cvcfg1_bits;

/* VPIF Channel n Vertical Size Configuration 2 Register (C0VCFG2 and C1VCFG2) */
typedef struct {
  __REG32 L11                     :12;
  __REG32                         : 4;
  __REG32 L9                      :12;
  __REG32                         : 4;
} __vpif_cvcfg2_bits;

/* VPIF Channel n Vertical Image Size Register (C0VSIZE and C1VSIZE) */
typedef struct {
  __REG32 VSIZE                   :12;
  __REG32                         :20;
} __vpif_cvsize_bits;

/* VPIF Channel n Horizontal Size Configuration Register (C2HCFG and C3HCFG) */
typedef struct {
  __REG32 SAV2EAV                 :11;
  __REG32                         : 5;
  __REG32 EAV2SAV                 :11;
  __REG32                         : 5;
} __vpif_c2hcfg_bits;

/* VPIF Channel n Vertical Size Configuration 0 Register (C2VCFG0 and C3VCFG0) */
typedef struct {
  __REG32 L3                      :11;
  __REG32                         : 5;
  __REG32 L1                      :11;
  __REG32                         : 5;
} __vpif_c2vcfg0_bits;

/* VPIF Channel n Vertical Size Configuration 1 Register (C2VCFG1 and C3VCFG1) */
typedef struct {
  __REG32 L7                      :11;
  __REG32                         : 5;
  __REG32 L5                      :11;
  __REG32                         : 5;
} __vpif_c2vcfg1_bits;

/* VPIF Channel n Vertical Size Configuration 2 Register (C2VCFG2 and C3VCFG2) */
typedef struct {
  __REG32 L11                     :11;
  __REG32                         : 5;
  __REG32 L9                      :11;
  __REG32                         : 5;
} __vpif_c2vcfg2_bits;

/* VPIF Channel n Vertical Image Size Register (C2VSIZE and C3VSIZE) */
typedef struct {
  __REG32 VSIZE                   :11;
  __REG32                         :21;
} __vpif_c2vsize_bits;

/* VPIF Channel n Top Field Horizontal Ancillary Position Register (C2THANCPOS and C3THANCPOS) */
/* VPIF Channel n Bottom Field Horizontal Ancillary Position Register (C2BHANCPOS and C3BHANCPOS) */
/* VPIF Channel n Top Field Vertical Ancillary Position Register (C2TVANCPOS and C3TVANCPOS) */
/* VPIF Channel n Bottom Field Vertical Ancillary Position Register (C2BVANCPOS and C3BVANCPOS) */
typedef struct {
  __REG32 HPOS                    :11;
  __REG32                         : 5;
  __REG32 VPOS                    :11;
  __REG32                         : 5;
} __vpif_c2thancpos_bits;

/* VPIF Channel n Top Field Horizontal Ancillary Size Register (C2THANCSIZE and C3THANCSIZE) */
/* VPIF Channel n Bottom Field Horizontal Ancillary Size Register (C2BHANCSIZE and C3BHANCSIZE) */
/* VPIF Channel n Top Field Vertical Ancillary Size Register (C2TVANCSIZE and C3TVANCSIZE) */
/* VPIF Channel n Bottom Field Vertical Ancillary Size Register (C2BVANCSIZE and C3BVANCSIZE) */
typedef struct {
  __REG32 HSIZE                   :11;
  __REG32                         : 5;
  __REG32 VSIZE                   :11;
  __REG32                         : 5;
} __vpif_c2thancsize_bits;

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

/* RTC Second Register (SECOND) */
/* RTC Alarm Second Register (ALARMSECOND) */
typedef struct {
  __REG32 SEC0                    : 4;
  __REG32 SEC1                    : 3;
  __REG32                         :25;
} __rtc_second_bits;

/* RTC Minute Register (MINUTE) */
/* RTC Alarm Minute Register (ALARMMINUTE) */
typedef struct {
  __REG32 MIN0                    : 4;
  __REG32 MIN1                    : 3;
  __REG32                         :25;
} __rtc_minute_bits;

/* RTC Hour Register (HOUR) */
/* RTC Alarm Hour Register (ALARMHOUR) */
typedef struct {
  __REG32 HOUR0                   : 4;
  __REG32 HOUR1                   : 2;
  __REG32                         : 1;
  __REG32 MERIDIEM                : 1;
  __REG32                         :24;
} __rtc_hour_bits;

/* RTC Day of the Month Register (DAY) */
/* RTC Alarm Day of the Month Register (ALARMDAY) */
typedef struct {
  __REG32 DAY0                    : 4;
  __REG32 DAY1                    : 2;
  __REG32                         :26;
} __rtc_day_bits;

/* RTC Month Register (MONTH) */
/* RTC Alarm Month Register (ALARMMONTH) */
typedef struct {
  __REG32 MONTH0                  : 4;
  __REG32 MONTH1                  : 1;
  __REG32                         :27;
} __rtc_month_bits;

/* RTC Year Register (YEAR) */
/* RTC Alarm Year Register (ALARMYEAR) */
typedef struct {
  __REG32 YEAR0                   : 4;
  __REG32 YEAR1                   : 4;
  __REG32                         :24;
} __rtc_year_bits;

/* RTC Day of the Week Register (DOTW) */
typedef struct {
  __REG32 DOTW                    : 3;
  __REG32                         :29;
} __rtc_dotw_bits;

/* RTC Control Register (CTRL) */
typedef struct {
  __REG32 RUN                     : 1;
  __REG32 ROUNDMIN                : 1;
  __REG32 AUTOCOMP                : 1;
  __REG32 HOURMODE                : 1;
  __REG32                         : 1;
  __REG32 SET32COUNTER            : 1;
  __REG32 RTCDISABLE              : 1;
  __REG32 SPLITPOWER              : 1;
  __REG32                         :24;
} __rtc_ctrl_bits;

/* RTC Status Register (STATUS) */
typedef struct {
  __REG32 BUSY                    : 1;
  __REG32 RUN                     : 1;
  __REG32 SECEVT                  : 1;
  __REG32 MINEVT                  : 1;
  __REG32 HREVT                   : 1;
  __REG32 DAYEVT                  : 1;
  __REG32 ALARM                   : 1;
  __REG32                         :25;
} __rtc_status_bits;

/* RTC Interrupt Register (INTERRUPT) */
typedef struct {
  __REG32 EVERY                   : 2;
  __REG32 TIMER                   : 1;
  __REG32 ALARM                   : 1;
  __REG32                         :28;
} __rtc_interrupt_bits;

/* RTC Compensation (LSB) Register (COMPLSB) */
typedef struct {
  __REG32 COMPLSB                 : 8;
  __REG32                         :24;
} __rtc_complsb_bits;

/* RTC Compensation (MSB) Register (COMPMSB) */
typedef struct {
  __REG32 COMPMSB                 : 8;
  __REG32                         :24;
} __rtc_compmsb_bits;

/* RTC Oscillator Register (OSC) */
typedef struct {
  __REG32                         : 5;
  __REG32 SWRESET                 : 1;
  __REG32                         :26;
} __rtc_osc_bits;

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

/* GPIO Banks 8 Direction Register (DIR8) */
typedef struct {
  __REG32 GP8P0                   : 1;
  __REG32 GP8P1                   : 1;
  __REG32 GP8P2                   : 1;
  __REG32 GP8P3                   : 1;
  __REG32 GP8P4                   : 1;
  __REG32 GP8P5                   : 1;
  __REG32 GP8P6                   : 1;
  __REG32 GP8P7                   : 1;
  __REG32 GP8P8                   : 1;
  __REG32 GP8P9                   : 1;
  __REG32 GP8P10                  : 1;
  __REG32 GP8P11                  : 1;
  __REG32 GP8P12                  : 1;
  __REG32 GP8P13                  : 1;
  __REG32 GP8P14                  : 1;
  __REG32 GP8P15                  : 1;
  __REG32                         :16;
} __gpio_dir8_bits;

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
__IO_REG32(    MPU1_PROG0_MPSAR,      0x01E14200,__READ_WRITE );
__IO_REG32(    MPU1_PROG0_MPEAR,      0x01E14204,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG0_MPPA,       0x01E14208,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG1_MPSAR,      0x01E14210,__READ_WRITE );
__IO_REG32(    MPU1_PROG1_MPEAR,      0x01E14214,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG1_MPPA,       0x01E14218,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG2_MPSAR,      0x01E14220,__READ_WRITE );
__IO_REG32(    MPU1_PROG2_MPEAR,      0x01E14224,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG2_MPPA,       0x01E14228,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG3_MPSAR,      0x01E14230,__READ_WRITE );
__IO_REG32(    MPU1_PROG3_MPEAR,      0x01E14234,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG3_MPPA,       0x01E14238,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG4_MPSAR,      0x01E14240,__READ_WRITE );
__IO_REG32(    MPU1_PROG4_MPEAR,      0x01E14244,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG4_MPPA,       0x01E14248,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU1_PROG5_MPSAR,      0x01E14250,__READ_WRITE );
__IO_REG32(    MPU1_PROG5_MPEAR,      0x01E14254,__READ_WRITE );
__IO_REG32_BIT(MPU1_PROG5_MPPA,       0x01E14258,__READ_WRITE ,__mpu_fxd_mppa_bits);
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
__IO_REG32(    MPU2_PROG0_MPSAR,      0x01E15200,__READ_WRITE );
__IO_REG32(    MPU2_PROG0_MPEAR,      0x01E15204,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG0_MPPA,       0x01E15208,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG1_MPSAR,      0x01E15210,__READ_WRITE );
__IO_REG32(    MPU2_PROG1_MPEAR,      0x01E15214,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG1_MPPA,       0x01E15218,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG2_MPSAR,      0x01E15220,__READ_WRITE );
__IO_REG32(    MPU2_PROG2_MPEAR,      0x01E15224,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG2_MPPA,       0x01E15228,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG3_MPSAR,      0x01E15230,__READ_WRITE );
__IO_REG32(    MPU2_PROG3_MPEAR,      0x01E15234,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG3_MPPA,       0x01E15238,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG4_MPSAR,      0x01E15240,__READ_WRITE );
__IO_REG32(    MPU2_PROG4_MPEAR,      0x01E15244,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG4_MPPA,       0x01E15248,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG5_MPSAR,      0x01E15250,__READ_WRITE );
__IO_REG32(    MPU2_PROG5_MPEAR,      0x01E15254,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG5_MPPA,       0x01E15258,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG6_MPSAR,      0x01E15260,__READ_WRITE );
__IO_REG32(    MPU2_PROG6_MPEAR,      0x01E15264,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG6_MPPA,       0x01E15268,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG7_MPSAR,      0x01E15270,__READ_WRITE );
__IO_REG32(    MPU2_PROG7_MPEAR,      0x01E15274,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG7_MPPA,       0x01E15278,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG8_MPSAR,      0x01E15280,__READ_WRITE );
__IO_REG32(    MPU2_PROG8_MPEAR,      0x01E15284,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG8_MPPA,       0x01E15288,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG9_MPSAR,      0x01E15290,__READ_WRITE );
__IO_REG32(    MPU2_PROG9_MPEAR,      0x01E15294,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG9_MPPA,       0x01E15298,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG10_MPSAR,     0x01E152A0,__READ_WRITE );
__IO_REG32(    MPU2_PROG10_MPEAR,     0x01E152A4,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG10_MPPA,      0x01E152A8,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_PROG11_MPSAR,     0x01E152B0,__READ_WRITE );
__IO_REG32(    MPU2_PROG11_MPEAR,     0x01E152B4,__READ_WRITE );
__IO_REG32_BIT(MPU2_PROG11_MPPA,      0x01E152B8,__READ_WRITE ,__mpu_fxd_mppa_bits);
__IO_REG32(    MPU2_FLTADDRR,         0x01E15300,__READ       );
__IO_REG32_BIT(MPU2_FLTSTAT,          0x01E15304,__READ       ,__mpu_fltstat_bits);
__IO_REG32_BIT(MPU2_FLTCLR,           0x01E15308,__WRITE      ,__mpu_fltclr_bits);

/***************************************************************************
 **
 ** PLLC0
 **
 ***************************************************************************/
__IO_REG32(    PLLC0_REVID,           0x01C11000,__READ       );
__IO_REG32_BIT(PLLC0_RSTYPE,          0x01C11004,__READ       ,__pllc0_rstype_bits);
__IO_REG32_BIT(PLLC0_PLLCTL,          0x01C11100,__READ_WRITE ,__pllc0_pllctl_bits);
__IO_REG32_BIT(PLLC0_OCSEL,           0x01C11104,__READ_WRITE ,__pllc_ocsel_bits);
__IO_REG32_BIT(PLLC0_PLLM,            0x01C11110,__READ_WRITE ,__pllc_pllm_bits);
__IO_REG32_BIT(PLLC0_PREDIV,          0x01C11114,__READ_WRITE ,__pllc0_prediv_bits);
__IO_REG32_BIT(PLLC0_PLLDIV1,         0x01C11118,__READ_WRITE ,__pllc_plldiv1_bits);
__IO_REG32_BIT(PLLC0_PLLDIV2,         0x01C1111C,__READ_WRITE ,__pllc_plldiv2_bits);
__IO_REG32_BIT(PLLC0_PLLDIV3,         0x01C11120,__READ_WRITE ,__pllc_plldiv3_bits);
__IO_REG32_BIT(PLLC0_OSCDIV,          0x01C11124,__READ_WRITE ,__pllc_oscdiv_bits);
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
 ** PLLC1
 **
 ***************************************************************************/
__IO_REG32(    PLLC1_REVID,           0x01E1A000,__READ       );
__IO_REG32_BIT(PLLC1_PLLCTL,          0x01E1A100,__READ_WRITE ,__pllc1_pllctl_bits);
__IO_REG32_BIT(PLLC1_OCSEL,           0x01E1A104,__READ_WRITE ,__pllc_ocsel_bits);
__IO_REG32_BIT(PLLC1_PLLM,            0x01E1A110,__READ_WRITE ,__pllc_pllm_bits);
__IO_REG32_BIT(PLLC1_PLLDIV1,         0x01E1A118,__READ_WRITE ,__pllc_plldiv1_bits);
__IO_REG32_BIT(PLLC1_PLLDIV2,         0x01E1A11C,__READ_WRITE ,__pllc_plldiv2_bits);
__IO_REG32_BIT(PLLC1_PLLDIV3,         0x01E1A120,__READ_WRITE ,__pllc_plldiv3_bits);
__IO_REG32_BIT(PLLC1_OSCDIV,          0x01E1A124,__READ_WRITE ,__pllc_oscdiv_bits);
__IO_REG32_BIT(PLLC1_POSTDIV,         0x01E1A128,__READ_WRITE ,__pllc_postdiv_bits);
__IO_REG32_BIT(PLLC1_PLLCMD,          0x01E1A138,__READ_WRITE ,__pllc_pllcmd_bits);
__IO_REG32_BIT(PLLC1_PLLSTAT,         0x01E1A13C,__READ       ,__pllc_pllstat_bits);
__IO_REG32_BIT(PLLC1_ALNCTL,          0x01E1A140,__READ_WRITE ,__pllc1_alnctl_bits);
__IO_REG32_BIT(PLLC1_DCHANGE,         0x01E1A144,__READ       ,__pllc1_dchange_bits);
__IO_REG32_BIT(PLLC1_SYSTAT,          0x01E1A150,__READ       ,__pllc1_systat_bits);
__IO_REG32(    PLLC1_EMUCNT0,         0x01E1A1F0,__READ       );
__IO_REG32(    PLLC1_EMUCNT1,         0x01E1A1F4,__READ       );

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
//__IO_REG32_BIT(PSC1_MERRPR0,          0x01E27040,__READ       ,__psc_merrpr0_bits);
//__IO_REG32_BIT(PSC1_MERRCR0,          0x01E27050,__WRITE      ,__psc_merrcr0_bits);
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
 ** SYSCFG1
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSCFG1_VTPIO_CTL,     0x01E2C000,__READ_WRITE ,__syscfg_vtpio_ctl_bits);
__IO_REG32_BIT(SYSCFG1_DDR_SLEW,      0x01E2C004,__READ_WRITE ,__syscfg_ddr_slew_bits);
__IO_REG32_BIT(SYSCFG1_DEEP_SLEEP,    0x01E2C008,__READ_WRITE ,__syscfg_deep_sleep_bits);
__IO_REG32_BIT(SYSCFG1_PUPD_ENA,      0x01E2C00C,__READ_WRITE ,__syscfg_pupd_ena_bits);
__IO_REG32_BIT(SYSCFG1_PUPD_SEL,      0x01E2C010,__READ_WRITE ,__syscfg_pupd_sel_bits);
__IO_REG32_BIT(SYSCFG1_RXACTIVE,      0x01E2C014,__READ_WRITE ,__syscfg_rxactive_bits);
__IO_REG32_BIT(SYSCFG1_PWRDN,         0x01E2C018,__READ_WRITE ,__syscfg_pwrdn_bits);

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
__IO_REG32_BIT(AINTC_SRSR3,           0xFFFEE20C,__WRITE      ,__aintc_srsr3_bits);
__IO_REG32_BIT(AINTC_SECR0,           0xFFFEE280,__WRITE      ,__aintc_secr0_bits);
__IO_REG32_BIT(AINTC_SECR1,           0xFFFEE284,__WRITE      ,__aintc_secr1_bits);
__IO_REG32_BIT(AINTC_SECR2,           0xFFFEE288,__WRITE      ,__aintc_secr2_bits);
__IO_REG32_BIT(AINTC_SECR3,           0xFFFEE28C,__WRITE      ,__aintc_secr3_bits);
__IO_REG32_BIT(AINTC_ESR0,            0xFFFEE300,__WRITE      ,__aintc_esr0_bits);
__IO_REG32_BIT(AINTC_ESR1,            0xFFFEE304,__WRITE      ,__aintc_esr1_bits);
__IO_REG32_BIT(AINTC_ESR2,            0xFFFEE308,__WRITE      ,__aintc_esr2_bits);
__IO_REG32_BIT(AINTC_ESR3,            0xFFFEE30C,__WRITE      ,__aintc_esr3_bits);
__IO_REG32_BIT(AINTC_ECR0,            0xFFFEE380,__WRITE      ,__aintc_ecr0_bits);
__IO_REG32_BIT(AINTC_ECR1,            0xFFFEE384,__WRITE      ,__aintc_ecr1_bits);
__IO_REG32_BIT(AINTC_ECR2,            0xFFFEE388,__WRITE      ,__aintc_ecr2_bits);
__IO_REG32_BIT(AINTC_ECR3,            0xFFFEE38C,__WRITE      ,__aintc_ecr3_bits);
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
__IO_REG32_BIT(AINTC_CMR23,           0xFFFEE45C,__READ_WRITE ,__aintc_cmr23_bits);
__IO_REG32_BIT(AINTC_CMR24,           0xFFFEE460,__READ_WRITE ,__aintc_cmr24_bits);
__IO_REG32_BIT(AINTC_CMR25,           0xFFFEE464,__READ_WRITE ,__aintc_cmr25_bits);
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
 ** DDR2/mDDR
 **
 ***************************************************************************/
__IO_REG32(    DDR_REVID,             0xB0000000,__READ       );
__IO_REG32_BIT(DDR_SDRSTAT,           0xB0000004,__READ       ,__ddr_sdrstat_bits);
__IO_REG32_BIT(DDR_SDCR,              0xB0000008,__READ_WRITE ,__ddr_sdcr_bits);
__IO_REG32_BIT(DDR_SDRCR,             0xB000000C,__READ_WRITE ,__ddr_sdrcr_bits);
__IO_REG32_BIT(DDR_SDTIMR1,           0xB0000010,__READ_WRITE ,__ddr_sdtimr1_bits);
__IO_REG32_BIT(DDR_SDTIMR2,           0xB0000014,__READ_WRITE ,__ddr_sdtimr2_bits);
__IO_REG32_BIT(DDR_SDCR2,             0xB000001C,__READ_WRITE ,__ddr_sdcr2_bits);
__IO_REG32_BIT(DDR_PBBPR,             0xB0000020,__READ_WRITE ,__ddr_pbbpr_bits);
__IO_REG32(    DDR_PC1,               0xB0000040,__READ       );
__IO_REG32(    DDR_PC2,               0xB0000044,__READ       );
__IO_REG32_BIT(DDR_PCC,               0xB0000048,__READ_WRITE ,__ddr_pcc_bits);
__IO_REG32_BIT(DDR_PCMRS,             0xB000004C,__READ_WRITE ,__ddr_pcmrs_bits);
__IO_REG32(    DDR_PCT,               0xB0000050,__READ       );
__IO_REG32_BIT(DDR_IRR,               0xB00000C0,__READ_WRITE ,__ddr_irr_bits);
__IO_REG32_BIT(DDR_IMR,               0xB00000C4,__READ_WRITE ,__ddr_imr_bits);
__IO_REG32_BIT(DDR_IMSR,              0xB00000C8,__READ_WRITE ,__ddr_imsr_bits);
__IO_REG32_BIT(DDR_IMCR,              0xB00000CC,__READ_WRITE ,__ddr_imcr_bits);
__IO_REG32_BIT(DDR_DRPYC1R,           0xB00000E4,__READ_WRITE ,__ddr_drpyc1r_bits);

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
 ** MMCSD1
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC1CTL,               0x01E1B000,__READ_WRITE ,__mmcctl_bits);
__IO_REG32_BIT(MMC1CLK,               0x01E1B004,__READ_WRITE ,__mmcclk_bits);
__IO_REG32_BIT(MMC1ST0,               0x01E1B008,__READ       ,__mmcst0_bits);
__IO_REG32_BIT(MMC1ST1,               0x01E1B00C,__READ       ,__mmcst1_bits);
__IO_REG32_BIT(MMC1IM,                0x01E1B010,__READ_WRITE ,__mmcim_bits);
__IO_REG32_BIT(MMC1TOR,               0x01E1B014,__READ_WRITE ,__mmctor_bits);
__IO_REG32_BIT(MMC1TOD,               0x01E1B018,__READ_WRITE ,__mmctod_bits);
__IO_REG32_BIT(MMC1BLEN,              0x01E1B01C,__READ_WRITE ,__mmcblen_bits);
__IO_REG32_BIT(MMC1NBLK,              0x01E1B020,__READ_WRITE ,__mmcnblk_bits);
__IO_REG32_BIT(MMC1NBLC,              0x01E1B024,__READ       ,__mmcnblc_bits);
__IO_REG32(    MMC1DRR,               0x01E1B028,__READ_WRITE );
__IO_REG32(    MMC1DXR,               0x01E1B02C,__READ_WRITE );
__IO_REG32_BIT(MMC1CMD,               0x01E1B030,__READ_WRITE ,__mmccmd_bits);
__IO_REG32_BIT(MMC1ARGHL,             0x01E1B034,__READ_WRITE ,__mmcarghl_bits);
__IO_REG32_BIT(MMC1RSP01,             0x01E1B038,__READ_WRITE ,__mmcrsp01_bits);
__IO_REG32_BIT(MMC1RSP23,             0x01E1B03C,__READ_WRITE ,__mmcrsp23_bits);
__IO_REG32_BIT(MMC1RSP45,             0x01E1B040,__READ_WRITE ,__mmcrsp45_bits);
__IO_REG32_BIT(MMC1RSP67,             0x01E1B044,__READ_WRITE ,__mmcrsp67_bits);
__IO_REG32_BIT(MMC1DRSP,              0x01E1B048,__READ_WRITE ,__mmcdrsp_bits);
__IO_REG32_BIT(MMC1CIDX,              0x01E1B050,__READ_WRITE ,__mmccidx_bits);
__IO_REG32_BIT(SDIO1CTL,              0x01E1B064,__READ_WRITE ,__sdioctl_bits);
__IO_REG32_BIT(SDIO1ST0,              0x01E1B068,__READ       ,__sdiost0_bits);
__IO_REG32_BIT(SDI01IEN,              0x01E1B06C,__READ_WRITE ,__sdioien_bits);
__IO_REG32_BIT(SDI01IST,              0x01E1B070,__READ_WRITE ,__sdioist_bits);
__IO_REG32_BIT(MMC1FIFOCTL,           0x01E1B074,__READ_WRITE ,__mmcfifoctl_bits);

/***************************************************************************
 **
 ** SATA
 **
 ***************************************************************************/
__IO_REG32_BIT(SATA_CAP,              0x01E18000,__READ       ,__sata_cap_bits);
__IO_REG32_BIT(SATA_GHC,              0x01E18004,__READ_WRITE ,__sata_ghc_bits);
__IO_REG32_BIT(SATA_IS,               0x01E18008,__READ_WRITE ,__sata_is_bits);
__IO_REG32_BIT(SATA_PI,               0x01E1800C,__READ_WRITE ,__sata_pi_bits);
__IO_REG32_BIT(SATA_VS,               0x01E18010,__READ       ,__sata_vs_bits);
__IO_REG32_BIT(SATA_CCC_CTL,          0x01E18014,__READ_WRITE ,__sata_ccc_ctl_bits);
__IO_REG32_BIT(SATA_CCC_PORTS,        0x01E18018,__READ_WRITE ,__sata_ccc_ports_bits);
__IO_REG32_BIT(SATA_BISTAFR,          0x01E180A0,__READ_WRITE ,__sata_bistafr_bits);
__IO_REG32_BIT(SATA_BISTCR,           0x01E180A4,__READ_WRITE ,__sata_bistcr_bits);
__IO_REG32(    SATA_BISTFCTR,         0x01E180A8,__READ_WRITE );
__IO_REG32_BIT(SATA_BISTSR,           0x01E180AC,__READ_WRITE ,__sata_bistsr_bits);
__IO_REG32(    SATA_BISTDECR,         0x01E180B0,__READ       );
__IO_REG32_BIT(SATA_TIMER1MS,         0x01E180E0,__READ_WRITE ,__sata_timer1ms_bits);
__IO_REG32_BIT(SATA_GPARAM1R,         0x01E180E8,__READ       ,__sata_gparam1r_bits);
__IO_REG32_BIT(SATA_GPARAM2R,         0x01E180EC,__READ       ,__sata_gparam2r_bits);
__IO_REG32_BIT(SATA_PPARAMR,          0x01E180F0,__READ       ,__sata_pparamr_bits);
__IO_REG32_BIT(SATA_TESTR,            0x01E180F4,__READ_WRITE ,__sata_testr_bits);
__IO_REG32(    SATA_VERSIONR,         0x01E180F8,__READ       );
__IO_REG32(    SATA_IDR,              0x01E180FC,__READ       );
__IO_REG32(    SATA_P0CLB,            0x01E18100,__READ_WRITE );
__IO_REG32(    SATA_P0FB,             0x01E18108,__READ_WRITE );
__IO_REG32_BIT(SATA_P0IS,             0x01E18110,__READ_WRITE ,__sata_p0is_bits);
__IO_REG32_BIT(SATA_P0IE,             0x01E18114,__READ_WRITE ,__sata_p0ie_bits);
__IO_REG32_BIT(SATA_P0CMD,            0x01E18118,__READ_WRITE ,__sata_p0cmd_bits);
__IO_REG32_BIT(SATA_P0TFD,            0x01E18120,__READ       ,__sata_p0tfd_bits);
__IO_REG32(    SATA_P0SIG,            0x01E18124,__READ       );
__IO_REG32_BIT(SATA_P0SSTS,           0x01E18128,__READ       ,__sata_p0ssts_bits);
__IO_REG32_BIT(SATA_P0SCTL,           0x01E1812C,__READ_WRITE ,__sata_p0ssts_bits);
__IO_REG32_BIT(SATA_P0SERR,           0x01E18130,__READ_WRITE ,__sata_p0serr_bits);
__IO_REG32_BIT(SATA_P0SACT,           0x01E18134,__READ_WRITE ,__sata_p0sact_bits);
__IO_REG32_BIT(SATA_P0CI,             0x01E18138,__READ_WRITE ,__sata_p0ci_bits);
__IO_REG32_BIT(SATA_P0SNTF,           0x01E1813C,__READ_WRITE ,__sata_p0sntf_bits);
__IO_REG32_BIT(SATA_P0DMACR,          0x01E18170,__READ_WRITE ,__sata_p0dmacr_bits);
__IO_REG32_BIT(SATA_P0PHYCR,          0x01E18178,__READ_WRITE ,__sata_p0phycr_bits);
__IO_REG32_BIT(SATA_P0PHYSR,          0x01E1817C,__READ       ,__sata_p0physr_bits);

/***************************************************************************
 **
 ** McASP
 **
 ***************************************************************************/
__IO_REG32(    McASP_REV,             0x01D00000,__READ       );
__IO_REG32_BIT(McASP_PFUNC,           0x01D00010,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP_PDIR,            0x01D00014,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP_PDOUT,           0x01D00018,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP_PDIN,            0x01D0001C,__READ_WRITE ,__mcasp_pfunc_bits);
#define McASP_PDSET     McASP_PDIN
#define McASP_PDSET_bit McASP_PDIN_bit
__IO_REG32_BIT(McASP_PDCLR,           0x01D00020,__READ_WRITE ,__mcasp_pfunc_bits);
__IO_REG32_BIT(McASP_GBLCTL,          0x01D00044,__READ_WRITE ,__mcasp_gblctl_bits);
__IO_REG32_BIT(McASP_AMUTE,           0x01D00048,__READ_WRITE ,__mcasp_amute_bits);
__IO_REG32_BIT(McASP_DLBCTL,          0x01D0004C,__READ_WRITE ,__mcasp_dlbctl_bits);
__IO_REG32_BIT(McASP_DITCTL,          0x01D00050,__READ_WRITE ,__mcasp_ditctl_bits);
__IO_REG32_BIT(McASP_RGBLCTL,         0x01D00060,__READ_WRITE ,__mcasp_rgblctl_bits);
__IO_REG32_BIT(McASP_RMASK,           0x01D00064,__READ_WRITE ,__mcasp_rmask_bits);
__IO_REG32_BIT(McASP_RFMT,            0x01D00068,__READ_WRITE ,__mcasp_rfmt_bits);
__IO_REG32_BIT(McASP_AFSRCTL,         0x01D0006C,__READ_WRITE ,__mcasp_afsrctl_bits);
__IO_REG32_BIT(McASP_ACLKRCTL,        0x01D00070,__READ_WRITE ,__mcasp_aclkrctl_bits);
__IO_REG32_BIT(McASP_AHCLKRCTL,       0x01D00074,__READ_WRITE ,__mcasp_ahclkrctl_bits);
__IO_REG32_BIT(McASP_RTDM,            0x01D00078,__READ_WRITE ,__mcasp_rtdm_bits);
__IO_REG32_BIT(McASP_RINTCTL,         0x01D0007C,__READ_WRITE ,__mcasp_rintctl_bits);
__IO_REG32_BIT(McASP_RSTAT,           0x01D00080,__READ_WRITE ,__mcasp_rstat_bits);
__IO_REG32_BIT(McASP_RSLOT,           0x01D00084,__READ       ,__mcasp_rslot_bits);
__IO_REG32_BIT(McASP_RCLKCHK,         0x01D00088,__READ_WRITE ,__mcasp_rclkchk_bits);
__IO_REG32_BIT(McASP_REVTCTL,         0x01D0008C,__READ_WRITE ,__mcasp_revtctl_bits);
__IO_REG32_BIT(McASP_XGBLCTL,         0x01D000A0,__READ_WRITE ,__mcasp_xgblctl_bits);
__IO_REG32_BIT(McASP_XMASK,           0x01D000A4,__READ_WRITE ,__mcasp_xmask_bits);
__IO_REG32_BIT(McASP_XFMT,            0x01D000A8,__READ_WRITE ,__mcasp_xfmt_bits);
__IO_REG32_BIT(McASP_AFSXCTL,         0x01D000AC,__READ_WRITE ,__mcasp_afsxctl_bits);
__IO_REG32_BIT(McASP_ACLKXCTL,        0x01D000B0,__READ_WRITE ,__mcasp_aclkxctl_bits);
__IO_REG32_BIT(McASP_AHCLKXCTL,       0x01D000B4,__READ_WRITE ,__mcasp_ahclkxctl_bits);
__IO_REG32_BIT(McASP_XTDM,            0x01D000B8,__READ_WRITE ,__mcasp_xtdm_bits);
__IO_REG32_BIT(McASP_XINTCTL,         0x01D000BC,__READ_WRITE ,__mcasp_xintctl_bits);
__IO_REG32_BIT(McASP_XSTAT,           0x01D000C0,__READ_WRITE ,__mcasp_xstat_bits);
__IO_REG32_BIT(McASP_XSLOT,           0x01D000C4,__READ       ,__mcasp_xslot_bits);
__IO_REG32_BIT(McASP_XCLKCHK,         0x01D000C8,__READ_WRITE ,__mcasp_xclkchk_bits);
__IO_REG32_BIT(McASP_XEVTCTL,         0x01D000CC,__READ_WRITE ,__mcasp_xevtctl_bits);
__IO_REG32(    McASP_DITCSRA0,        0x01D00100,__READ_WRITE );
__IO_REG32(    McASP_DITCSRA1,        0x01D00104,__READ_WRITE );
__IO_REG32(    McASP_DITCSRA2,        0x01D00108,__READ_WRITE );
__IO_REG32(    McASP_DITCSRA3,        0x01D0010C,__READ_WRITE );
__IO_REG32(    McASP_DITCSRA4,        0x01D00110,__READ_WRITE );
__IO_REG32(    McASP_DITCSRA5,        0x01D00114,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB0,        0x01D00118,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB1,        0x01D0011C,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB2,        0x01D00120,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB3,        0x01D00124,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB4,        0x01D00128,__READ_WRITE );
__IO_REG32(    McASP_DITCSRB5,        0x01D0012C,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA0,        0x01D00130,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA1,        0x01D00134,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA2,        0x01D00138,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA3,        0x01D0013C,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA4,        0x01D00140,__READ_WRITE );
__IO_REG32(    McASP_DITUDRA5,        0x01D00144,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB0,        0x01D00148,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB1,        0x01D0014C,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB2,        0x01D00150,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB3,        0x01D00154,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB4,        0x01D00158,__READ_WRITE );
__IO_REG32(    McASP_DITUDRB5,        0x01D0015C,__READ_WRITE );
__IO_REG32_BIT(McASP_SRCTL0,          0x01D00180,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL1,          0x01D00184,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL2,          0x01D00188,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL3,          0x01D0018C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL4,          0x01D00190,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL5,          0x01D00194,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL6,          0x01D00198,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL7,          0x01D0019C,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL8,          0x01D001A0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL9,          0x01D001A4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL10,         0x01D001A8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL11,         0x01D001AC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL12,         0x01D001B0,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL13,         0x01D001B4,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL14,         0x01D001B8,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32_BIT(McASP_SRCTL15,         0x01D001BC,__READ_WRITE ,__mcasp_srctl_bits);
__IO_REG32(    McASP_XBUF0,           0x01D00200,__READ_WRITE );
__IO_REG32(    McASP_XBUF1,           0x01D00204,__READ_WRITE );
__IO_REG32(    McASP_XBUF2,           0x01D00208,__READ_WRITE );
__IO_REG32(    McASP_XBUF3,           0x01D0020C,__READ_WRITE );
__IO_REG32(    McASP_XBUF4,           0x01D00210,__READ_WRITE );
__IO_REG32(    McASP_XBUF5,           0x01D00214,__READ_WRITE );
__IO_REG32(    McASP_XBUF6,           0x01D00218,__READ_WRITE );
__IO_REG32(    McASP_XBUF7,           0x01D0021C,__READ_WRITE );
__IO_REG32(    McASP_XBUF8,           0x01D00220,__READ_WRITE );
__IO_REG32(    McASP_XBUF9,           0x01D00224,__READ_WRITE );
__IO_REG32(    McASP_XBUF10,          0x01D00228,__READ_WRITE );
__IO_REG32(    McASP_XBUF11,          0x01D0022C,__READ_WRITE );
__IO_REG32(    McASP_XBUF12,          0x01D00230,__READ_WRITE );
__IO_REG32(    McASP_XBUF13,          0x01D00234,__READ_WRITE );
__IO_REG32(    McASP_XBUF14,          0x01D00238,__READ_WRITE );
__IO_REG32(    McASP_XBUF15,          0x01D0023C,__READ_WRITE );
__IO_REG32(    McASP_RBUF0,           0x01D00280,__READ_WRITE );
__IO_REG32(    McASP_RBUF1,           0x01D00284,__READ_WRITE );
__IO_REG32(    McASP_RBUF2,           0x01D00288,__READ_WRITE );
__IO_REG32(    McASP_RBUF3,           0x01D0028C,__READ_WRITE );
__IO_REG32(    McASP_RBUF4,           0x01D00290,__READ_WRITE );
__IO_REG32(    McASP_RBUF5,           0x01D00294,__READ_WRITE );
__IO_REG32(    McASP_RBUF6,           0x01D00298,__READ_WRITE );
__IO_REG32(    McASP_RBUF7,           0x01D0029C,__READ_WRITE );
__IO_REG32(    McASP_RBUF8,           0x01D002A0,__READ_WRITE );
__IO_REG32(    McASP_RBUF9,           0x01D002A4,__READ_WRITE );
__IO_REG32(    McASP_RBUF10,          0x01D002A8,__READ_WRITE );
__IO_REG32(    McASP_RBUF11,          0x01D002AC,__READ_WRITE );
__IO_REG32(    McASP_RBUF12,          0x01D002B0,__READ_WRITE );
__IO_REG32(    McASP_RBUF13,          0x01D002B4,__READ_WRITE );
__IO_REG32(    McASP_RBUF14,          0x01D002B8,__READ_WRITE );
__IO_REG32(    McASP_RBUF15,          0x01D002BC,__READ_WRITE );
__IO_REG32(    McASP_RBUF,            0x01D02000,__READ_WRITE );
#define McASP_XBUF      McASP_RBUF
__IO_REG32(    McASP_AFIFOREV,        0x01D01000,__READ_WRITE );
__IO_REG32_BIT(McASP_WFIFOCTL,        0x01D01010,__READ_WRITE ,__mcasp_wfifoctl_bits);
__IO_REG32_BIT(McASP_WFIFOSTS,        0x01D01014,__READ       ,__mcasp_wfifosts_bits);
__IO_REG32_BIT(McASP_RFIFOCTL,        0x01D01018,__READ_WRITE ,__mcasp_rfifoctl_bits);
__IO_REG32_BIT(McASP_RFIFOSTS,        0x01D0101C,__READ       ,__mcasp_rfifosts_bits);

/***************************************************************************
 **
 ** McBSP0
 **
 ***************************************************************************/
__IO_REG32(    McBSP0_DRR,            0x01D10000,__READ       );
__IO_REG32(    McBSP0_DXR,            0x01D10004,__READ       );
__IO_REG32_BIT(McBSP0_SPCR,           0x01D10008,__READ_WRITE ,__mcbsp_spcr_bits);
__IO_REG32_BIT(McBSP0_RCR,            0x01D1000C,__READ_WRITE ,__mcbsp_rcr_bits);
__IO_REG32_BIT(McBSP0_XCR,            0x01D10010,__READ_WRITE ,__mcbsp_xcr_bits);
__IO_REG32_BIT(McBSP0_SRGR,           0x01D10014,__READ_WRITE ,__mcbsp_srgr_bits);
__IO_REG32_BIT(McBSP0_MCR,            0x01D10018,__READ_WRITE ,__mcbsp_mcr_bits);
__IO_REG32_BIT(McBSP0_RCERE0,         0x01D1001C,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP0_XCERE0,         0x01D10020,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP0_PCR,            0x01D10024,__READ_WRITE ,__mcbsp_pcr_bits);
__IO_REG32_BIT(McBSP0_RCERE1,         0x01D10028,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP0_XCERE1,         0x01D1002C,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP0_RCERE2,         0x01D10030,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP0_XCERE2,         0x01D10034,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP0_RCERE3,         0x01D10038,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP0_XCERE3,         0x01D1003C,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32(    McBSP0_BFIFOREV,       0x01D10800,__READ       );
__IO_REG32_BIT(McBSP0_WFIFOCTL,       0x01D10810,__READ_WRITE ,__mcbsp_wfifoctl_bits);
__IO_REG32_BIT(McBSP0_WFIFOSTS,       0x01D10814,__READ       ,__mcbsp_wfifosts_bits);
__IO_REG32_BIT(McBSP0_RFIFOCTL,       0x01D10818,__READ_WRITE ,__mcbsp_rfifoctl_bits);
__IO_REG32_BIT(McBSP0_RFIFOSTS,       0x01D1081C,__READ       ,__mcbsp_rfifosts_bits);
__IO_REG32(    McBSP0_RBUF,           0x01F10000,__READ_WRITE );
#define McBSP0_XBUF     McBSP0_RBUF

/***************************************************************************
 **
 ** McBSP1
 **
 ***************************************************************************/
__IO_REG32(    McBSP1_DRR,            0x01D11000,__READ       );
__IO_REG32(    McBSP1_DXR,            0x01D11004,__READ       );
__IO_REG32_BIT(McBSP1_SPCR,           0x01D11008,__READ_WRITE ,__mcbsp_spcr_bits);
__IO_REG32_BIT(McBSP1_RCR,            0x01D1100C,__READ_WRITE ,__mcbsp_rcr_bits);
__IO_REG32_BIT(McBSP1_XCR,            0x01D11010,__READ_WRITE ,__mcbsp_xcr_bits);
__IO_REG32_BIT(McBSP1_SRGR,           0x01D11014,__READ_WRITE ,__mcbsp_srgr_bits);
__IO_REG32_BIT(McBSP1_MCR,            0x01D11018,__READ_WRITE ,__mcbsp_mcr_bits);
__IO_REG32_BIT(McBSP1_RCERE0,         0x01D1101C,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP1_XCERE0,         0x01D11020,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP1_PCR,            0x01D11024,__READ_WRITE ,__mcbsp_pcr_bits);
__IO_REG32_BIT(McBSP1_RCERE1,         0x01D11028,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP1_XCERE1,         0x01D1102C,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP1_RCERE2,         0x01D11030,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP1_XCERE2,         0x01D11034,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32_BIT(McBSP1_RCERE3,         0x01D11038,__READ_WRITE ,__mcbsp_rcere_bits);
__IO_REG32_BIT(McBSP1_XCERE3,         0x01D1103C,__READ_WRITE ,__mcbsp_xcere_bits);
__IO_REG32(    McBSP1_BFIFOREV,       0x01D11800,__READ       );
__IO_REG32_BIT(McBSP1_WFIFOCTL,       0x01D11810,__READ_WRITE ,__mcbsp_wfifoctl_bits);
__IO_REG32_BIT(McBSP1_WFIFOSTS,       0x01D11814,__READ       ,__mcbsp_wfifosts_bits);
__IO_REG32_BIT(McBSP1_RFIFOCTL,       0x01D11818,__READ_WRITE ,__mcbsp_rfifoctl_bits);
__IO_REG32_BIT(McBSP1_RFIFOSTS,       0x01D1181C,__READ       ,__mcbsp_rfifosts_bits);
__IO_REG32(    McBSP1_RBUF,           0x01F11000,__READ_WRITE );
#define McBSP1_XBUF     McBSP1_RBUF

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
__IO_REG32_BIT(SPI1GCR0,              0x01F0E000,__READ_WRITE ,__spigcr0_bits);
__IO_REG32_BIT(SPI1GCR1,              0x01F0E004,__READ_WRITE ,__spigcr1_bits);
__IO_REG32_BIT(SPI1INT0,              0x01F0E008,__READ_WRITE ,__spiint0_bits);
__IO_REG32_BIT(SPI1LVL,               0x01F0E00C,__READ_WRITE ,__spilvl_bits);
__IO_REG32_BIT(SPI1FLG,               0x01F0E010,__READ_WRITE ,__spiflg_bits);
__IO_REG32_BIT(SPI1PC0,               0x01F0E014,__READ_WRITE ,__spipc0_bits);
__IO_REG32_BIT(SPI1PC1,               0x01F0E018,__READ_WRITE ,__spipc1_bits);
__IO_REG32_BIT(SPI1PC2,               0x01F0E01C,__READ_WRITE ,__spipc2_bits);
__IO_REG32_BIT(SPI1PC3,               0x01F0E020,__READ_WRITE ,__spipc3_bits);
__IO_REG32_BIT(SPI1PC4,               0x01F0E024,__READ_WRITE ,__spipc4_bits);
__IO_REG32_BIT(SPI1PC5,               0x01F0E028,__READ_WRITE ,__spipc5_bits);
__IO_REG32_BIT(SPI1DAT0,              0x01F0E038,__READ_WRITE ,__spidat0_bits);
__IO_REG32_BIT(SPI1DAT1,              0x01F0E03C,__READ_WRITE ,__spidat1_bits);
__IO_REG32_BIT(SPI1BUF,               0x01F0E040,__READ_WRITE ,__spibuf_bits);
__IO_REG32_BIT(SPI1EMU,               0x01F0E044,__READ       ,__spiemu_bits);
__IO_REG32_BIT(SPI1DELAY,             0x01F0E048,__READ_WRITE ,__spidelay_bits);
__IO_REG32_BIT(SPI1DEF,               0x01F0E04C,__READ_WRITE ,__spidef_bits);
__IO_REG32_BIT(SPI1FMT0,              0x01F0E050,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT1,              0x01F0E054,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT2,              0x01F0E058,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1FMT3,              0x01F0E05C,__READ_WRITE ,__spifmt_bits);
__IO_REG32_BIT(SPI1INTVEC1,           0x01F0E064,__READ_WRITE ,__spiintvec1_bits);

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
__IO_REG32_BIT(UART0_REVID2,          0x01C4202C,__READ       ,__uartrevid2_bits);
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
__IO_REG32_BIT(UART1_REVID2,          0x01D0C02C,__READ       ,__uartrevid2_bits);
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
__IO_REG32_BIT(UART2_REVID2,          0x01D0D02C,__READ       ,__uartrevid2_bits);
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
__IO_REG16_BIT(USB0_HWVERS,             0x01E0046C,__READ       ,__usb_hwvers_bits);
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
__IO_REG32_BIT(USB0_DMA_SCHED_CTRL,     0x01E02C00,__READ_WRITE ,__usb_dma_sched_ctrl_bits);
__IO_REG32_BIT(USB0_ENTRY0,             0x01E02D00,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY1,             0x01E02D04,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY2,             0x01E02D08,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY3,             0x01E02D0C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY4,             0x01E02D10,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY5,             0x01E02D14,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY6,             0x01E02D18,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY7,             0x01E02D1C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY8,             0x01E02D20,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY9,             0x01E02D24,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY10,            0x01E02D28,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY11,            0x01E02D2C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY12,            0x01E02D30,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY13,            0x01E02D34,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY14,            0x01E02D38,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY15,            0x01E02D3C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY16,            0x01E02D40,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY17,            0x01E02D44,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY18,            0x01E02D48,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY19,            0x01E02D4C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY20,            0x01E02D50,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY21,            0x01E02D54,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY22,            0x01E02D58,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY23,            0x01E02D5C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY24,            0x01E02D60,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY25,            0x01E02D64,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY26,            0x01E02D68,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY27,            0x01E02D6C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY28,            0x01E02D70,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY29,            0x01E02D74,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY30,            0x01E02D78,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY31,            0x01E02D7C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY32,            0x01E02D80,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY33,            0x01E02D84,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY34,            0x01E02D88,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY35,            0x01E02D8C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY36,            0x01E02D90,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY37,            0x01E02D94,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY38,            0x01E02D98,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY39,            0x01E02D9C,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY40,            0x01E02DA0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY41,            0x01E02DA4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY42,            0x01E02DA8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY43,            0x01E02DAC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY44,            0x01E02DB0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY45,            0x01E02DB4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY46,            0x01E02DB8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY47,            0x01E02DBC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY48,            0x01E02DC0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY49,            0x01E02DC4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY50,            0x01E02DC8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY51,            0x01E02DCC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY52,            0x01E02DD0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY53,            0x01E02DD4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY54,            0x01E02DD8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY55,            0x01E02DDC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY56,            0x01E02DE0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY57,            0x01E02DE4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY58,            0x01E02DE8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY59,            0x01E02DEC,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY60,            0x01E02DF0,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY61,            0x01E02DF4,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY62,            0x01E02DF8,__READ_WRITE ,__usb_entry_bits);
__IO_REG32_BIT(USB0_ENTRY63,            0x01E02DFC,__READ_WRITE ,__usb_entry_bits);
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
__IO_REG32(   USB0_QMEMRBASE2,         0x01E05020,__READ_WRITE );
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
 ** USB1 (HOST)
 **
 ***************************************************************************/
__IO_REG32_BIT(HCREVISION,            0x01E25000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(HCCONTROL,             0x01E25004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(HCCOMMANDSTATUS,       0x01E25008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(HCINTERRUPTSTATUS,     0x01E2500C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(HCINTERRUPTENABLE,     0x01E25010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCINTERRUPTDISABLE,    0x01E25014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32(    HCHCCA,                0x01E25018,__READ_WRITE );
__IO_REG32(    HCPERIODCURRENTED,     0x01E2501C,__READ       );
__IO_REG32(    HCCONTROLHEADED,       0x01E25020,__READ_WRITE );
__IO_REG32(    HCCONTROLCURRENTED,    0x01E25024,__READ_WRITE );
__IO_REG32(    HCBULKHEADED,          0x01E25028,__READ_WRITE );
__IO_REG32(    HCBULKCURRENTED,       0x01E2502C,__READ_WRITE );
__IO_REG32(    HCDONEHEAD,            0x01E25030,__READ_WRITE );
__IO_REG32_BIT(HCFMINTERVAL,          0x01E25034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(HCFMREMAINING,         0x01E25038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(HCFMNUMBER,            0x01E2503C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(HCPERIODICSTART,       0x01E25040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(HCLSTHRESHOLD,         0x01E25044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(HCRHDESCRIPTORA,       0x01E25048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(HCRHDESCRIPTORB,       0x01E2504C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HCRHSTATUS,            0x01E25050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS1,       0x01E25054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS2,       0x01E25058,__READ_WRITE ,__hcrhportstatus_bits);

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
__IO_REG32_BIT(EMAC_RX1FLOWTHRESH,    0x01E20124,__READ_WRITE ,__emac_rxflowthresh_bits);
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
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32(    LCDC_REVID,            0x01E13000,__READ       );
__IO_REG32_BIT(LCDC_CTRL,             0x01E13004,__READ_WRITE ,__lcdc_ctrl_bits);
__IO_REG32_BIT(LCDC_STAT,             0x01E13008,__READ_WRITE ,__lcdc_stat_bits);
__IO_REG32_BIT(LIDD_CTRL,             0x01E1300C,__READ_WRITE ,__lcdc_ctrl_bits);
__IO_REG32_BIT(LIDD_CS0_CONF,         0x01E13010,__READ_WRITE ,__lcdc_cs_conf_bits);
__IO_REG32_BIT(LIDD_CS0_ADDR,         0x01E13014,__READ_WRITE ,__lcdc_cs_addr_bits);
__IO_REG32_BIT(LIDD_CS0_DATA,         0x01E13018,__READ_WRITE ,__lcdc_cs_data_bits);
__IO_REG32_BIT(LIDD_CS1_CONF,         0x01E1301C,__READ_WRITE ,__lcdc_cs_conf_bits);
__IO_REG32_BIT(LIDD_CS1_ADDR,         0x01E13020,__READ_WRITE ,__lcdc_cs_addr_bits);
__IO_REG32_BIT(LIDD_CS1_DATA,         0x01E13024,__READ_WRITE ,__lcdc_cs_data_bits);
__IO_REG32_BIT(LCDC_RASTER_CTRL,      0x01E13028,__READ_WRITE ,__lcdc_raster_ctrl_bits);
__IO_REG32_BIT(LCDC_RASTER_TIMING_0,  0x01E1302C,__READ_WRITE ,__lcdc_raster_timing_0_bits);
__IO_REG32_BIT(LCDC_RASTER_TIMING_1,  0x01E13030,__READ_WRITE ,__lcdc_raster_timing_1_bits);
__IO_REG32_BIT(LCDC_RASTER_TIMING_2,  0x01E13034,__READ_WRITE ,__lcdc_raster_timing_2_bits);
__IO_REG32_BIT(LCDC_RASTER_SUBPANEL,  0x01E13038,__READ_WRITE ,__lcdc_raster_subpanel_bits);
__IO_REG32_BIT(LCDC_DMA_CTRL,         0x01E13040,__READ_WRITE ,__lcdc_dma_ctrl_bits);
__IO_REG32(    LCDC_DMA_FB0_BASE,     0x01E13044,__READ_WRITE );
__IO_REG32(    LCDC_DMA_FB0_CEILING,  0x01E13048,__READ_WRITE );
__IO_REG32(    LCDC_DMA_FB1_BASE,     0x01E1304C,__READ_WRITE );
__IO_REG32(    LCDC_DMA_FB1_CEILING,  0x01E13050,__READ_WRITE );

/***************************************************************************
 **
 ** HPI
 **
 ***************************************************************************/
__IO_REG32(    HPIREVID,              0x01E10000,__READ       );
__IO_REG32_BIT(HPIPWREMU_MGMT,        0x01E10004,__READ_WRITE ,__hpipwremu_mgmt_bits);
__IO_REG32_BIT(HPIGPIO_EN,            0x01E1000C,__READ_WRITE ,__hpigpio_en_bits);
__IO_REG32_BIT(HPIGPIO_DIR1,          0x01E10010,__READ_WRITE ,__hpigpio_dir1_bits);
__IO_REG32_BIT(HPIGPIO_DAT1,          0x01E10014,__READ_WRITE ,__hpigpio_dir1_bits);
__IO_REG32_BIT(HPIGPIO_DIR2,          0x01E10018,__READ_WRITE ,__hpigpio_dir2_bits);
__IO_REG32_BIT(HPIGPIO_DAT2,          0x01E1001C,__READ_WRITE ,__hpigpio_dir2_bits);
__IO_REG32_BIT(HPIC,                  0x01E10030,__READ_WRITE ,__hpic_bits);
__IO_REG32(    HPIAW,                 0x01E10034,__READ       );
__IO_REG32(    HPIAR,                 0x01E10038,__READ       );

/***************************************************************************
 **
 ** uPP
 **
 ***************************************************************************/
__IO_REG32(    UPPID,                 0x01E16000,__READ       );
__IO_REG32_BIT(UPPCR,                 0x01E16004,__READ_WRITE ,__uppcr_bits);
__IO_REG32_BIT(UPDLB,                 0x01E16008,__READ_WRITE ,__updlb_bits);
__IO_REG32_BIT(UPCTL,                 0x01E16010,__READ_WRITE ,__upctl_bits);
__IO_REG32_BIT(UPICR,                 0x01E16014,__READ_WRITE ,__upicr_bits);
__IO_REG32_BIT(UPIVR,                 0x01E16018,__READ_WRITE ,__upivr_bits);
__IO_REG32_BIT(UPTCR,                 0x01E1601C,__READ_WRITE ,__uptcr_bits);
__IO_REG32_BIT(UPISR,                 0x01E16020,__READ_WRITE ,__upisr_bits);
__IO_REG32_BIT(UPIER,                 0x01E16024,__READ_WRITE ,__upisr_bits);
__IO_REG32_BIT(UPIES,                 0x01E16028,__READ_WRITE ,__upisr_bits);
__IO_REG32_BIT(UPIEC,                 0x01E1602C,__READ_WRITE ,__upisr_bits);
__IO_REG32_BIT(UPEOI,                 0x01E16030,__READ_WRITE ,__upeoi_bits);
__IO_REG32(    UPID0,                 0x01E16040,__READ_WRITE );
__IO_REG32_BIT(UPID1,                 0x01E16044,__READ_WRITE ,__upid1_bits);
__IO_REG32_BIT(UPID2,                 0x01E16048,__READ_WRITE ,__upid2_bits);
__IO_REG32(    UPIS0,                 0x01E16050,__READ       );
__IO_REG32_BIT(UPIS1,                 0x01E16054,__READ       ,__upid1_bits);
__IO_REG32_BIT(UPIS2,                 0x01E16058,__READ       ,__upis2_bits);
__IO_REG32(    UPQD0,                 0x01E16060,__READ_WRITE );
__IO_REG32_BIT(UPQD1,                 0x01E16064,__READ_WRITE ,__upid1_bits);
__IO_REG32_BIT(UPQD2,                 0x01E16068,__READ_WRITE ,__upid2_bits);
__IO_REG32(    UPQS0,                 0x01E16070,__READ       );
__IO_REG32_BIT(UPQS1,                 0x01E16074,__READ       ,__upid1_bits);
__IO_REG32_BIT(UPQS2,                 0x01E16078,__READ       ,__upis2_bits);

/***************************************************************************
 **
 ** VPIF
 **
 ***************************************************************************/
__IO_REG32(    VPIF_REVID,            0x01E17000,__READ       );
__IO_REG32_BIT(VPIF_C0CTRL,           0x01E17004,__READ_WRITE ,__vpif_c0ctrl_bits);
__IO_REG32_BIT(VPIF_C1CTRL,           0x01E17008,__READ_WRITE ,__vpif_c1ctrl_bits);
__IO_REG32_BIT(VPIF_C2CTRL,           0x01E1700C,__READ_WRITE ,__vpif_c2ctrl_bits);
__IO_REG32_BIT(VPIF_C3CTRL,           0x01E17010,__READ_WRITE ,__vpif_c3ctrl_bits);
__IO_REG32_BIT(VPIF_INTEN,            0x01E17020,__READ_WRITE ,__vpif_inten_bits);
__IO_REG32_BIT(VPIF_INTENSET,         0x01E17024,__READ_WRITE ,__vpif_intenset_bits);
__IO_REG32_BIT(VPIF_INTENCLR,         0x01E17028,__READ_WRITE ,__vpif_inten_bits);
__IO_REG32_BIT(VPIF_INTSTAT,          0x01E1702C,__READ       ,__vpif_inten_bits);
__IO_REG32_BIT(VPIF_INTSTATCLR,       0x01E17030,__READ_WRITE ,__vpif_inten_bits);
__IO_REG32_BIT(VPIF_EMUCTRL,          0x01E17034,__READ_WRITE ,__vpif_emuctrl_bits);
__IO_REG32_BIT(VPIF_REQSIZE,          0x01E17038,__READ_WRITE ,__vpif_reqsize_bits);
__IO_REG32(    VPIF_C0TLUMA,          0x01E17040,__READ_WRITE );
__IO_REG32(    VPIF_C0BLUMA,          0x01E17044,__READ_WRITE );
__IO_REG32(    VPIF_C0TCHROMA,        0x01E17048,__READ_WRITE );
__IO_REG32(    VPIF_C0BCHROMA,        0x01E1704C,__READ_WRITE );
__IO_REG32(    VPIF_C0THANC,          0x01E17050,__READ_WRITE );
__IO_REG32(    VPIF_C0BHANC,          0x01E17054,__READ_WRITE );
__IO_REG32(    VPIF_C0TVANC,          0x01E17058,__READ_WRITE );
__IO_REG32(    VPIF_C0BVANC,          0x01E1705C,__READ_WRITE );
__IO_REG32(    VPIF_C0IMGOFFSET,      0x01E17064,__READ_WRITE );
__IO_REG32(    VPIF_C0HANCOFFSET,     0x01E17068,__READ_WRITE );
__IO_REG32_BIT(VPIF_C0HCFG,           0x01E1706C,__READ_WRITE ,__vpif_chcfg_bits);
__IO_REG32_BIT(VPIF_C0VCFG0,          0x01E17070,__READ_WRITE ,__vpif_cvcfg0_bits);
__IO_REG32_BIT(VPIF_C0VCFG1,          0x01E17074,__READ_WRITE ,__vpif_cvcfg1_bits);
__IO_REG32_BIT(VPIF_C0VCFG2,          0x01E17078,__READ_WRITE ,__vpif_cvcfg2_bits);
__IO_REG32_BIT(VPIF_C0VSIZE,          0x01E1707C,__READ_WRITE ,__vpif_cvsize_bits);
__IO_REG32(    VPIF_C1TLUMA,          0x01E17080,__READ_WRITE );
__IO_REG32(    VPIF_C1BLUMA,          0x01E17084,__READ_WRITE );
__IO_REG32(    VPIF_C1TCHROMA,        0x01E17088,__READ_WRITE );
__IO_REG32(    VPIF_C1BCHROMA,        0x01E1708C,__READ_WRITE );
__IO_REG32(    VPIF_C1THANC,          0x01E17090,__READ_WRITE );
__IO_REG32(    VPIF_C1BHANC,          0x01E17094,__READ_WRITE );
__IO_REG32(    VPIF_C1TVANC,          0x01E17098,__READ_WRITE );
__IO_REG32(    VPIF_C1BVANC,          0x01E1709C,__READ_WRITE );
__IO_REG32(    VPIF_C1IMGOFFSET,      0x01E170A4,__READ_WRITE );
__IO_REG32(    VPIF_C1HANCOFFSET,     0x01E170A8,__READ_WRITE );
__IO_REG32_BIT(VPIF_C1HCFG,           0x01E170AC,__READ_WRITE ,__vpif_chcfg_bits);
__IO_REG32_BIT(VPIF_C1VCFG0,          0x01E170B0,__READ_WRITE ,__vpif_cvcfg0_bits);
__IO_REG32_BIT(VPIF_C1VCFG1,          0x01E170B4,__READ_WRITE ,__vpif_cvcfg1_bits);
__IO_REG32_BIT(VPIF_C1VCFG2,          0x01E170B8,__READ_WRITE ,__vpif_cvcfg2_bits);
__IO_REG32_BIT(VPIF_C1VSIZE,          0x01E170BC,__READ_WRITE ,__vpif_cvsize_bits);
__IO_REG32(    VPIF_C2TLUMA,          0x01E170C0,__READ_WRITE );
__IO_REG32(    VPIF_C2BLUMA,          0x01E170C4,__READ_WRITE );
__IO_REG32(    VPIF_C2TCHROMA,        0x01E170C8,__READ_WRITE );
__IO_REG32(    VPIF_C2BCHROMA,        0x01E170CC,__READ_WRITE );
__IO_REG32(    VPIF_C2THANC,          0x01E170D0,__READ_WRITE );
__IO_REG32(    VPIF_C2BHANC,          0x01E170D4,__READ_WRITE );
__IO_REG32(    VPIF_C2TVANC,          0x01E170D8,__READ_WRITE );
__IO_REG32(    VPIF_C2BVANC,          0x01E170DC,__READ_WRITE );
__IO_REG32(    VPIF_C2IMGOFFSET,      0x01E170E4,__READ_WRITE );
__IO_REG32(    VPIF_C2HANCOFFSET,     0x01E170E8,__READ_WRITE );
__IO_REG32_BIT(VPIF_C2HCFG,           0x01E170EC,__READ_WRITE ,__vpif_c2hcfg_bits);
__IO_REG32_BIT(VPIF_C2VCFG0,          0x01E170F0,__READ_WRITE ,__vpif_c2vcfg0_bits);
__IO_REG32_BIT(VPIF_C2VCFG1,          0x01E170F4,__READ_WRITE ,__vpif_c2vcfg1_bits);
__IO_REG32_BIT(VPIF_C2VCFG2,          0x01E170F8,__READ_WRITE ,__vpif_c2vcfg2_bits);
__IO_REG32_BIT(VPIF_C2VSIZE,          0x01E170FC,__READ_WRITE ,__vpif_c2vsize_bits);
__IO_REG32_BIT(VPIF_C2THANCPOS,       0x01E17100,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C2THANCSIZE,      0x01E17104,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C2BHANCPOS,       0x01E17108,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C2BHANCSIZE,      0x01E1710C,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C2TVANCPOS,       0x01E17110,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C2TVANCSIZE,      0x01E17114,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C2BVANCPOS,       0x01E17118,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C2BVANCSIZE,      0x01E1711C,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32(    VPIF_C3TLUMA,          0x01E17140,__READ_WRITE );
__IO_REG32(    VPIF_C3BLUMA,          0x01E17144,__READ_WRITE );
__IO_REG32(    VPIF_C3TCHROMA,        0x01E17148,__READ_WRITE );
__IO_REG32(    VPIF_C3BCHROMA,        0x01E1714C,__READ_WRITE );
__IO_REG32(    VPIF_C3THANC,          0x01E17150,__READ_WRITE );
__IO_REG32(    VPIF_C3BHANC,          0x01E17154,__READ_WRITE );
__IO_REG32(    VPIF_C3TVANC,          0x01E17158,__READ_WRITE );
__IO_REG32(    VPIF_C3BVANC,          0x01E1715C,__READ_WRITE );
__IO_REG32(    VPIF_C3IMGOFFSET,      0x01E17160,__READ_WRITE );
__IO_REG32(    VPIF_C3HANCOFFSET,     0x01E17164,__READ_WRITE );
__IO_REG32_BIT(VPIF_C3HCFG,           0x01E17168,__READ_WRITE ,__vpif_c2hcfg_bits);
__IO_REG32_BIT(VPIF_C3VCFG0,          0x01E1716C,__READ_WRITE ,__vpif_c2vcfg0_bits);
__IO_REG32_BIT(VPIF_C3VCFG1,          0x01E17170,__READ_WRITE ,__vpif_c2vcfg1_bits);
__IO_REG32_BIT(VPIF_C3VCFG2,          0x01E17174,__READ_WRITE ,__vpif_c2vcfg2_bits);
__IO_REG32_BIT(VPIF_C3VSIZE,          0x01E17178,__READ_WRITE ,__vpif_c2vsize_bits);
__IO_REG32_BIT(VPIF_C3THANCPOS,       0x01E1717C,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C3THANCSIZE,      0x01E17180,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C3BHANCPOS,       0x01E17184,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C3BHANCSIZE,      0x01E17188,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C3TVANCPOS,       0x01E1718C,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C3TVANCSIZE,      0x01E17190,__READ_WRITE ,__vpif_c2thancsize_bits);
__IO_REG32_BIT(VPIF_C3BVANCPOS,       0x01E17194,__READ_WRITE ,__vpif_c2thancpos_bits);
__IO_REG32_BIT(VPIF_C3BVANCSIZE,      0x01E17198,__READ_WRITE ,__vpif_c2thancsize_bits);

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
 ** TIMER64P2
 **
 ***************************************************************************/
__IO_REG32(    TIMER64P2_REVID,       0x01F0C000,__READ       );
__IO_REG32_BIT(TIMER64P2_EMUMGT,      0x01F0C004,__READ_WRITE ,__timer64p_emumgt_bits);
__IO_REG32_BIT(TIMER64P2_GPINTGPEN,   0x01F0C008,__READ_WRITE ,__timer64p_gpintgpen_bits);
__IO_REG32_BIT(TIMER64P2_GPDATGPDIR,  0x01F0C00C,__READ_WRITE ,__timer64p_gpdatgpdir_bits);
__IO_REG32(    TIMER64P2_TIM12,       0x01F0C010,__READ_WRITE );
__IO_REG32(    TIMER64P2_TIM34,       0x01F0C014,__READ_WRITE );
__IO_REG32(    TIMER64P2_PRD12,       0x01F0C018,__READ_WRITE );
__IO_REG32(    TIMER64P2_PRD34,       0x01F0C01C,__READ_WRITE );
__IO_REG32_BIT(TIMER64P2_TCR,         0x01F0C020,__READ_WRITE ,__timer64p_tcr_bits);
__IO_REG32_BIT(TIMER64P2_TGCR,        0x01F0C024,__READ_WRITE ,__timer64p_tgcr_bits);
__IO_REG32_BIT(TIMER64P2_WDTCR,       0x01F0C028,__READ_WRITE ,__timer64p_wdtcr_bits);
__IO_REG32(    TIMER64P2_REL12,       0x01F0C034,__READ_WRITE );
__IO_REG32(    TIMER64P2_REL34,       0x01F0C038,__READ_WRITE );
__IO_REG32(    TIMER64P2_CAP12,       0x01F0C03C,__READ_WRITE );
__IO_REG32(    TIMER64P2_CAP34,       0x01F0C040,__READ_WRITE );
__IO_REG32_BIT(TIMER64P2_INTCTLSTAT,  0x01F0C044,__READ_WRITE ,__timer64p_intctlstat_bits);
__IO_REG32(    TIMER64P2_CMP0,        0x01F0C060,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP1,        0x01F0C064,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP2,        0x01F0C068,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP3,        0x01F0C06C,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP4,        0x01F0C070,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP5,        0x01F0C074,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP6,        0x01F0C078,__READ_WRITE );
__IO_REG32(    TIMER64P2_CMP7,        0x01F0C07C,__READ_WRITE );

/***************************************************************************
 **
 ** TIMER64P3
 **
 ***************************************************************************/
__IO_REG32(    TIMER64P3_REVID,       0x01F0D000,__READ       );
__IO_REG32_BIT(TIMER64P3_EMUMGT,      0x01F0D004,__READ_WRITE ,__timer64p_emumgt_bits);
__IO_REG32_BIT(TIMER64P3_GPINTGPEN,   0x01F0D008,__READ_WRITE ,__timer64p_gpintgpen_bits);
__IO_REG32_BIT(TIMER64P3_GPDATGPDIR,  0x01F0D00C,__READ_WRITE ,__timer64p_gpdatgpdir_bits);
__IO_REG32(    TIMER64P3_TIM12,       0x01F0D010,__READ_WRITE );
__IO_REG32(    TIMER64P3_TIM34,       0x01F0D014,__READ_WRITE );
__IO_REG32(    TIMER64P3_PRD12,       0x01F0D018,__READ_WRITE );
__IO_REG32(    TIMER64P3_PRD34,       0x01F0D01C,__READ_WRITE );
__IO_REG32_BIT(TIMER64P3_TCR,         0x01F0D020,__READ_WRITE ,__timer64p_tcr_bits);
__IO_REG32_BIT(TIMER64P3_TGCR,        0x01F0D024,__READ_WRITE ,__timer64p_tgcr_bits);
__IO_REG32_BIT(TIMER64P3_WDTCR,       0x01F0D028,__READ_WRITE ,__timer64p_wdtcr_bits);
__IO_REG32(    TIMER64P3_REL12,       0x01F0D034,__READ_WRITE );
__IO_REG32(    TIMER64P3_REL34,       0x01F0D038,__READ_WRITE );
__IO_REG32(    TIMER64P3_CAP12,       0x01F0D03C,__READ_WRITE );
__IO_REG32(    TIMER64P3_CAP34,       0x01F0D040,__READ_WRITE );
__IO_REG32_BIT(TIMER64P3_INTCTLSTAT,  0x01F0D044,__READ_WRITE ,__timer64p_intctlstat_bits);
__IO_REG32(    TIMER64P3_CMP0,        0x01F0D060,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP1,        0x01F0D064,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP2,        0x01F0D068,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP3,        0x01F0D06C,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP4,        0x01F0D070,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP5,        0x01F0D074,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP6,        0x01F0D078,__READ_WRITE );
__IO_REG32(    TIMER64P3_CMP7,        0x01F0D07C,__READ_WRITE );

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_SECOND,            0x01C23000,__READ_WRITE ,__rtc_second_bits);
__IO_REG32_BIT(RTC_MINUTE,            0x01C23004,__READ_WRITE ,__rtc_minute_bits);
__IO_REG32_BIT(RTC_HOUR,              0x01C23008,__READ_WRITE ,__rtc_hour_bits);
__IO_REG32_BIT(RTC_DAY,               0x01C2300C,__READ_WRITE ,__rtc_day_bits);
__IO_REG32_BIT(RTC_MONTH,             0x01C23010,__READ_WRITE ,__rtc_month_bits);
__IO_REG32_BIT(RTC_YEAR,              0x01C23014,__READ_WRITE ,__rtc_year_bits);
__IO_REG32_BIT(RTC_DOTW,              0x01C23018,__READ_WRITE ,__rtc_dotw_bits);
__IO_REG32_BIT(RTC_ALARMSECOND,       0x01C23020,__READ_WRITE ,__rtc_second_bits);
__IO_REG32_BIT(RTC_ALARMMINUTE,       0x01C23024,__READ_WRITE ,__rtc_minute_bits);
__IO_REG32_BIT(RTC_ALARMHOUR,         0x01C23028,__READ_WRITE ,__rtc_hour_bits);
__IO_REG32_BIT(RTC_ALARMDAY,          0x01C2302C,__READ_WRITE ,__rtc_day_bits);
__IO_REG32_BIT(RTC_ALARMMONTH,        0x01C23030,__READ_WRITE ,__rtc_month_bits);
__IO_REG32_BIT(RTC_ALARMYEAR,         0x01C23034,__READ_WRITE ,__rtc_year_bits);
__IO_REG32_BIT(RTC_CTRL,              0x01C23040,__READ_WRITE ,__rtc_ctrl_bits);
__IO_REG32_BIT(RTC_STATUS,            0x01C23044,__READ_WRITE ,__rtc_status_bits);
__IO_REG32_BIT(RTC_INTERRUPT,         0x01C23048,__READ_WRITE ,__rtc_interrupt_bits);
__IO_REG32_BIT(RTC_COMPLSB,           0x01C2304C,__READ_WRITE ,__rtc_complsb_bits);
__IO_REG32_BIT(RTC_COMPMSB,           0x01C23050,__READ_WRITE ,__rtc_compmsb_bits);
__IO_REG32_BIT(RTC_OSC,               0x01C23054,__WRITE      ,__rtc_osc_bits);
__IO_REG32(    RTC_SCRATCH0,          0x01C23060,__READ_WRITE );
__IO_REG32(    RTC_SCRATCH1,          0x01C23064,__READ_WRITE );
__IO_REG32(    RTC_SCRATCH2,          0x01C23068,__READ_WRITE );
__IO_REG32(    RTC_KICK0,             0x01C2306C,__READ_WRITE );
__IO_REG32(    RTC_KICK1,             0x01C23070,__READ_WRITE );

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
__IO_REG32_BIT(GPIO_DIR8,             0x01E260B0,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_OUT_DATA8,        0x01E260B4,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_SET_DATA8,        0x01E260B8,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_CLR_DATA8,        0x01E260BC,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_IN_DATA8,         0x01E260C0,__READ       ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_SET_RIS_TRIG8,    0x01E260C4,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_CLR_RIS_TRIG8,    0x01E260C8,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_SET_FAL_TRIG8,    0x01E260CC,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_CLR_FAL_TRIG8,    0x01E260D0,__READ_WRITE ,__gpio_dir8_bits);
__IO_REG32_BIT(GPIO_INTSTAT8,         0x01E260D4,__READ_WRITE ,__gpio_dir8_bits);

#if 0
/***************************************************************************
 **
 ** PRU
 **
 ***************************************************************************/
__IO_REG32_BIT(PRU_REVID,             0x01C34000,__READ_WRITE ,__pru_revid_bits);
__IO_REG32_BIT(PRU_CONTROL,           0x01C34004,__READ_WRITE ,__pru_control_bits);
__IO_REG32_BIT(PRU_GLBLEN,            0x01C34010,__READ_WRITE ,__pru_glblen_bits);
__IO_REG32_BIT(PRU_GLBLNSTLVL,        0x01C3401C,__READ_WRITE ,__pru_glblnstlvl_bits);
__IO_REG32_BIT(PRU_STATIDXSET,        0x01C34020,__READ_WRITE ,__pru_statidxset_bits);
__IO_REG32_BIT(PRU_STATIDXCLR,        0x01C34024,__READ_WRITE ,__pru_statidxclr_bits);
__IO_REG32_BIT(PRU_ENIDXSET,          0x01C34028,__READ_WRITE ,__pru_enidxset_bits);
__IO_REG32_BIT(PRU_ENIDXCLR,          0x01C3402C,__READ_WRITE ,__pru_enidxclr_bits);
__IO_REG32_BIT(PRU_HSTINTENIDXSET,    0x01C34034,__READ_WRITE ,__pru_hstintenidxset_bits);
__IO_REG32_BIT(PRU_HSTINTENIDXCLR,    0x01C34038,__READ_WRITE ,__pru_hstintenidxclr_bits);
__IO_REG32_BIT(PRU_GLBLPRIIDX,        0x01C34080,__READ_WRITE ,__pru_glblpriidx_bits);
__IO_REG32_BIT(PRU_STATSETINT0,       0x01C34200,__READ_WRITE ,__pru_statsetint0_bits);
__IO_REG32_BIT(PRU_STATSETINT1,       0x01C34204,__READ_WRITE ,__pru_statsetint1_bits);
__IO_REG32_BIT(PRU_STATCLRINT0,       0x01C34280,__READ_WRITE ,__pru_statclrint0_bits);
__IO_REG32_BIT(PRU_STATCLRINT1,       0x01C34284,__READ_WRITE ,__pru_statclrint1_bits);
__IO_REG32_BIT(PRU_ENABLESET0,        0x01C34300,__READ_WRITE ,__pru_enableset0_bits);
__IO_REG32_BIT(PRU_ENABLESET1,        0x01C34304,__READ_WRITE ,__pru_enableset1_bits);
__IO_REG32_BIT(PRU_ENABLECLR0,        0x01C34380,__READ_WRITE ,__pru_enableclr0_bits);
__IO_REG32_BIT(PRU_ENABLECLR1,        0x01C34384,__READ_WRITE ,__pru_enableclr1_bits);
__IO_REG32_BIT(PRU_CHANMAP0,          0x01C34400,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP1,          0x01C34404,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP2,          0x01C34408,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP3,          0x01C3440C,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP4,          0x01C34410,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP5,          0x01C34414,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP6,          0x01C34418,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP7,          0x01C3441C,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP8,          0x01C34420,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP9,          0x01C34424,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP10,         0x01C34428,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP11,         0x01C3442C,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP12,         0x01C34430,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP13,         0x01C34434,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP14,         0x01C34438,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_CHANMAP15,         0x01C3443C,__READ_WRITE ,__pru_chanmap_bits);
__IO_REG32_BIT(PRU_HOSTMAP0,          0x01C34800,__READ_WRITE ,__pru_hostmap_bits);
__IO_REG32_BIT(PRU_HOSTMAP1,          0x01C34804,__READ_WRITE ,__pru_hostmap_bits);
__IO_REG32_BIT(PRU_HOSTMAP2,          0x01C34808,__READ_WRITE ,__pru_hostmap_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX0,    0x01C34900,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX1,    0x01C34904,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX2,    0x01C34908,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX3,    0x01C3490C,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX4,    0x01C34910,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX5,    0x01C34914,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX6,    0x01C34918,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX7,    0x01C3491C,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX8,    0x01C34924,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTPRIIDX9,    0x01C34928,__READ_WRITE ,__pru_hostintpriidx_bits);
__IO_REG32_BIT(PRU_HOSTINTEN,         0x01C35500,__READ_WRITE ,__pru_hostinten_bits);

/***************************************************************************
 **
 ** PRU0
 **
 ***************************************************************************/
__IO_REG32_BIT(PRU0_CONTROL,          0x01C37000,__READ_WRITE ,__pru_control_bits);
__IO_REG32_BIT(PRU0_STATUS,           0x01C37004,__READ_WRITE ,__pru_status_bits);
__IO_REG32_BIT(PRU0_WAKEUP,           0x01C37008,__READ_WRITE ,__pru_wakeup_bits);
__IO_REG32_BIT(PRU0_CYCLCNT,          0x01C3700C,__READ_WRITE ,__pru_cyclcnt_bits);
__IO_REG32_BIT(PRU0_STALLCNT,         0x01C37010,__READ_WRITE ,__pru_stallcnt_bits);
__IO_REG32_BIT(PRU0_CONTABBLKIDX0,    0x01C37020,__READ_WRITE ,__pru_contabblkidx0_bits);
__IO_REG32_BIT(PRU0_CONTABPROPTR0,    0x01C37028,__READ_WRITE ,__pru_contabproptr0_bits);
__IO_REG32_BIT(PRU0_CONTABPROPTR1,    0x01C3702C,__READ_WRITE ,__pru_contabproptr1_bits);
__IO_REG32_BIT(PRU0_INTGPR0,          0x01C37400,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR1,          0x01C37404,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR2,          0x01C37408,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR3,          0x01C3740C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR4,          0x01C37410,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR5,          0x01C37414,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR6,          0x01C37418,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR7,          0x01C3741C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR8,          0x01C37420,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR9,          0x01C37424,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR10,         0x01C37428,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR11,         0x01C3742C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR12,         0x01C37430,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR13,         0x01C37434,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR14,         0x01C37438,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR15,         0x01C3743C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR16,         0x01C37440,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR17,         0x01C37444,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR18,         0x01C37448,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR19,         0x01C3744C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR20,         0x01C37450,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR21,         0x01C37454,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR22,         0x01C37458,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR23,         0x01C3745C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR24,         0x01C37460,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR25,         0x01C37464,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR26,         0x01C37468,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR27,         0x01C3746C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR28,         0x01C37470,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR29,         0x01C37474,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR30,         0x01C37478,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTGPR31,         0x01C3747C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU0_INTCTER0,         0x01C37480,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER1,         0x01C37484,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER2,         0x01C37488,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER3,         0x01C3748C,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER4,         0x01C37490,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER5,         0x01C37494,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER6,         0x01C37498,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER7,         0x01C3749C,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER8,         0x01C374A0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER9,         0x01C374A4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER10,        0x01C374A8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER11,        0x01C374AC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER12,        0x01C374B0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER13,        0x01C374B4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER14,        0x01C374B8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER15,        0x01C374BC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER16,        0x01C374C0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER17,        0x01C374C4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER18,        0x01C374C8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER19,        0x01C374CC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER20,        0x01C374D0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER21,        0x01C374D4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER22,        0x01C374D8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER23,        0x01C374DC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER24,        0x01C374E0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER25,        0x01C374E4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER26,        0x01C374E8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER27,        0x01C374EC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER28,        0x01C374F0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER29,        0x01C374F4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER30,        0x01C374F8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU0_INTCTER31,        0x01C374FC,__READ_WRITE ,__pru_intcter_bits);

/***************************************************************************
 **
 ** PRU1
 **
 ***************************************************************************/
__IO_REG32_BIT(PRU1_CONTROL,          0x01C37800,__READ_WRITE ,__pru_control_bits);
__IO_REG32_BIT(PRU1_STATUS,           0x01C37804,__READ_WRITE ,__pru_status_bits);
__IO_REG32_BIT(PRU1_WAKEUP,           0x01C37808,__READ_WRITE ,__pru_wakeup_bits);
__IO_REG32_BIT(PRU1_CYCLCNT,          0x01C3780C,__READ_WRITE ,__pru_cyclcnt_bits);
__IO_REG32_BIT(PRU1_STALLCNT,         0x01C37810,__READ_WRITE ,__pru_stallcnt_bits);
__IO_REG32_BIT(PRU1_CONTABBLKIDX0,    0x01C37820,__READ_WRITE ,__pru_contabblkidx0_bits);
__IO_REG32_BIT(PRU1_CONTABPROPTR0,    0x01C37828,__READ_WRITE ,__pru_contabproptr0_bits);
__IO_REG32_BIT(PRU1_CONTABPROPTR1,    0x01C3782C,__READ_WRITE ,__pru_contabproptr1_bits);
__IO_REG32_BIT(PRU1_INTGPR0,          0x01C37C00,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR1,          0x01C37C04,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR2,          0x01C37C08,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR3,          0x01C37C0C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR4,          0x01C37C10,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR5,          0x01C37C14,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR6,          0x01C37C18,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR7,          0x01C37C1C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR8,          0x01C37C20,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR9,          0x01C37C24,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR10,         0x01C37C28,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR11,         0x01C37C2C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR12,         0x01C37C30,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR13,         0x01C37C34,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR14,         0x01C37C38,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR15,         0x01C37C3C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR16,         0x01C37C40,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR17,         0x01C37C44,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR18,         0x01C37C48,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR19,         0x01C37C4C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR20,         0x01C37C50,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR21,         0x01C37C54,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR22,         0x01C37C58,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR23,         0x01C37C5C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR24,         0x01C37C60,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR25,         0x01C37C64,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR26,         0x01C37C68,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR27,         0x01C37C6C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR28,         0x01C37C70,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR29,         0x01C37C74,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR30,         0x01C37C78,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTGPR31,         0x01C37C7C,__READ_WRITE ,__pru_intgpr_bits);
__IO_REG32_BIT(PRU1_INTCTER0,         0x01C37C80,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER1,         0x01C37C84,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER2,         0x01C37C88,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER3,         0x01C37C8C,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER4,         0x01C37C90,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER5,         0x01C37C94,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER6,         0x01C37C98,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER7,         0x01C37C9C,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER8,         0x01C37CA0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER9,         0x01C37CA4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER10,        0x01C37CA8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER11,        0x01C37CAC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER12,        0x01C37CB0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER13,        0x01C37CB4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER14,        0x01C37CB8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER15,        0x01C37CBC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER16,        0x01C37CC0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER17,        0x01C37CC4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER18,        0x01C37CC8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER19,        0x01C37CCC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER20,        0x01C37CD0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER21,        0x01C37CD4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER22,        0x01C37CD8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER23,        0x01C37CDC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER24,        0x01C37CE0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER25,        0x01C37CE4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER26,        0x01C37CE8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER27,        0x01C37CEC,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER28,        0x01C37CF0,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER29,        0x01C37CF4,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER30,        0x01C37CF8,__READ_WRITE ,__pru_intcter_bits);
__IO_REG32_BIT(PRU1_INTCTER31,        0x01C37CFC,__READ_WRITE ,__pru_intcter_bits);
#endif

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
#define AINT_RTC_IRQS           19
#define AINT_SPI0_INT           20
#define AINT_T64P0_TINT12       21
#define AINT_T64P0_TINT34       22
#define AINT_T64P1_TINT12       23
#define AINT_T64P1_TINT34       24
#define AINT_UART0_INT          25
#define AINT_PROTERR            27
#define AINT_SYSCFG_CHIPINT0    28
#define AINT_SYSCFG_CHIPINT1    29
#define AINT_SYSCFG_CHIPINT2    30
#define AINT_SYSCFG_CHIPINT3    31
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
#define AINT_GPIO_B8INT         50
#define AINT_IIC1_INT           51
#define AINT_LCDC_INT           52
#define AINT_UART_INT1          53
#define AINT_MCASP_INT          54
#define AINT_PSC1_ALLINT        55
#define AINT_SPI1_INT           56
#define AINT_UHPI_ARMINT        57
#define AINT_USB0_INT           58
#define AINT_USB1_HCINT         59
#define AINT_USB1_RWAKEUP       60
#define AINT_UART2_INT          61
#define AINT_EHRPWM0            63
#define AINT_EHRPWM0TZ          64
#define AINT_EHRPWM1            65
#define AINT_EHRPWM1TZ          66
#define AINT_SATA_INT           67
#define AINT_T64P2_ALL          68
#define AINT_ECAP0              69
#define AINT_ECAP1              70
#define AINT_ECAP2              71
#define AINT_MMCSD1_INT0        72
#define AINT_MMCSD1_INT1        73
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
#define AINT_uPP_ALLINT         91
#define AINT_VPIF_ALLINT        92
#define AINT_EDMA3_1_CC0_INT0   93
#define AINT_EDMA3_1_CC0_ERRINT 94
#define AINT_EDMA3_1_TC0_ERRINT 95
#define AINT_T64P3_ALL          96
#define AINT_MCBSP0_RINT        97
#define AINT_MCBSP0_XINT        98
#define AINT_MCBSP1_RINT        99
#define AINT_MCBSP1_XINT        100

#endif    /* __IOOMAPL138_H */
