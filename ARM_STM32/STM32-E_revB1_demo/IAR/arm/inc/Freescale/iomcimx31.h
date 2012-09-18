/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Motorola MCIMX31 Dragonball
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2003
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIMX31_H
#define __MCIMX31_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX31 SPECIAL FUNCTION REGISTERS
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

/* Control Register (CCMR) */
typedef struct {
__REG32 FPME      : 1;
__REG32 PRCS      : 2;
__REG32 MPE       : 1;
__REG32 SBYCS     : 1;
__REG32 ROMW      : 2;
__REG32 MDS       : 1;
__REG32 SPE       : 1;
__REG32 UPE       : 1;
__REG32 WAMO      : 1;
__REG32 FIRS      : 2;
__REG32           : 1;
__REG32 LPM       : 2;
__REG32 RAMW      : 2;
__REG32 SSI1S     : 2;
__REG32           : 1;
__REG32 SSI2S     : 2;
__REG32           : 1;
__REG32 PERCS     : 1;
__REG32 CSCS      : 1;
__REG32 FPMF      : 1;
__REG32 WBEN      : 1;
__REG32 VSTBY     : 1;
__REG32 L2PG      : 1;
__REG32           : 2;
} __ccm_ccmr_bits;

/* Post Divider Register 0 (PDR0) */
typedef struct {
__REG32 MCU_PODF  : 3;
__REG32 MAX_PODF  : 3;
__REG32 IPG_PODF  : 2;
__REG32 NFC_PODF  : 3;
__REG32 HSP_PODF  : 3;
__REG32           : 2;
__REG32 PER_PODF  : 5;
__REG32           : 2;
__REG32 CSI_PODF  : 9;
} __ccm_pdr0_bits;

/* Post Divider Register 1 (PDR1) */
typedef struct {
__REG32 SSI1_PODF       : 6;
__REG32 SSI1_PRE_PODF   : 3;
__REG32 SSI2_PODF       : 6;
__REG32 SSI2_PRE_PODF   : 3;
__REG32 FIRI_PODF       : 6;
__REG32 FIRI_PRE_PODF   : 3;
__REG32 USB_PODF        : 3;
__REG32 USB_PRDF        : 2;
} __ccm_pdr1_bits;

/* Reset Control and Source Register (RCSR) */
typedef struct {
__REG32 REST            : 3;
__REG32                 : 1;
__REG32 WFIS            : 1;
__REG32 GPF             : 3;
__REG32                 : 4;
__REG32 SDM             : 2;
__REG32                 : 1;
__REG32 PERES           : 1;
__REG32 OSCNT           : 7;
__REG32 BTP             : 5;
__REG32                 : 2;
__REG32 NFMS            : 1;
__REG32 NF16B           : 1;
} __ccm_rcsr_bits;

/* MCU PLL Control Register (MPCTL) */
/* USB PLL Control Register (UPCTL) */
/* SR PLL Control Register (SPCTL) */
typedef struct {
__REG32 MFN             :10;
__REG32 MFI             : 4;
__REG32                 : 2;
__REG32 MFD             :10;
__REG32 PD              : 4;
__REG32                 : 1;
__REG32 BRMO            : 1;
} __ccm_mpctl_bits;

/* Clock Out Source Register (COSR) */
typedef struct {
__REG32 CLKOSEL         : 4;
__REG32                 : 2;
__REG32 CLKOUTDIV       : 3;
__REG32 CLKOEN          : 1;
__REG32                 :22;
} __ccm_cosr_bits;

/* Clock Gating Registers (CGR0–CGR2) */
typedef struct {
__REG32 CG0             : 2;
__REG32 CG1             : 2;
__REG32 CG2             : 2;
__REG32 CG3             : 2;
__REG32 CG4             : 2;
__REG32 CG5             : 2;
__REG32 CG6             : 2;
__REG32 CG7             : 2;
__REG32 CG8             : 2;
__REG32 CG9             : 2;
__REG32 CG10            : 2;
__REG32 CG11            : 2;
__REG32 CG12            : 2;
__REG32 CG13            : 2;
__REG32 CG14            : 2;
__REG32 CG15            : 2;
} __ccm_cgr_bits;

typedef struct {
__REG32 CG0             : 2;
__REG32 CG1             : 2;
__REG32 CG2             : 2;
__REG32 CG3             : 2;
__REG32 CG4             : 2;
__REG32 CG5             : 2;
__REG32 CG6             : 2;
__REG32                 :18;
} __ccm_cgr2_bits;

/* Wake-Up Interrupt Mask Register (WIMR0) */
typedef struct {
__REG32 WIM0            : 1;
__REG32 WIM1            : 1;
__REG32 WIM2            : 1;
__REG32 WIM3            : 1;
__REG32 WIM4            : 1;
__REG32 WIM5            : 1;
__REG32 WIM6            : 1;
__REG32 WIM7            : 1;
__REG32 WIM8            : 1;
__REG32 WIM9            : 1;
__REG32 WIM10           : 1;
__REG32 WIM11           : 1;
__REG32 WIM12           : 1;
__REG32 WIM13           : 1;
__REG32 WIM14           : 1;
__REG32 WIM15           : 1;
__REG32 WIM16           : 1;
__REG32 WIM17           : 1;
__REG32 WIM18           : 1;
__REG32 WIM19           : 1;
__REG32 WIM20           : 1;
__REG32 WIM21           : 1;
__REG32 WIM22           : 1;
__REG32 WIM23           : 1;
__REG32 WIM24           : 1;
__REG32 WIM25           : 1;
__REG32 WIM26           : 1;
__REG32 WIM27           : 1;
__REG32 WIM28           : 1;
__REG32 WIM29           : 1;
__REG32 WIM30           : 1;
__REG32 WIM31           : 1;
} __ccm_wimr0_bits;

/* DPTC Comparator Value Registers (DCVR0-DCVR3) */
typedef struct {
__REG32                 : 2;
__REG32 ELV             :10;
__REG32 LLV             :10;
__REG32 ULV             :10;
} __ccm_dcvr_bits;

/* Load Tracking Register (LTR0) */
typedef struct {
__REG32                 : 1;
__REG32 DIV3CK          : 2;
__REG32 SIGD0           : 1;
__REG32 SIGD1           : 1;
__REG32 SIGD2           : 1;
__REG32 SIGD3           : 1;
__REG32 SIGD4           : 1;
__REG32 SIGD5           : 1;
__REG32 SIGD6           : 1;
__REG32 SIGD7           : 1;
__REG32 SIGD8           : 1;
__REG32 SIGD9           : 1;
__REG32 SIGD10          : 1;
__REG32 SIGD11          : 1;
__REG32 SIGD12          : 1;
__REG32 DNTHR           : 6;
__REG32 UPTHR           : 6;
__REG32                 : 1;
__REG32 SIGD13          : 1;
__REG32 SIGD14          : 1;
__REG32 SIGD15          : 1;
} __ccm_ltr0_bits;

/* Load Tracking Register (LTR1) */
typedef struct {
__REG32 PNCTHR          : 6;
__REG32 UPCNT           : 8;
__REG32 DNCNT           : 8;
__REG32 LTBRSR          : 1;
__REG32 LTBRSH          : 1;
__REG32                 : 8;
} __ccm_ltr1_bits;

/* Load Tracking Register (LTR2) */
typedef struct {
__REG32 EMAC            : 9;
__REG32                 : 2;
__REG32 WSW9            : 3;
__REG32 WSW10           : 3;
__REG32 WSW11           : 3;
__REG32 WSW12           : 3;
__REG32 WSW13           : 3;
__REG32 WSW14           : 3;
__REG32 WSW15           : 3;
} __ccm_ltr2_bits;

/* Load Tracking Register (LTR3) */
typedef struct {
__REG32                 : 5;
__REG32 WSW0            : 3;
__REG32 WSW1            : 3;
__REG32 WSW2            : 3;
__REG32 WSW3            : 3;
__REG32 WSW4            : 3;
__REG32 WSW5            : 3;
__REG32 WSW6            : 3;
__REG32 WSW7            : 3;
__REG32 WSW8            : 3;
} __ccm_ltr3_bits;

/* Load Tracking Buffer Register (LTBR0) */
typedef struct {
__REG32 LTS0            : 4;
__REG32 LTS1            : 4;
__REG32 LTS2            : 4;
__REG32 LTS3            : 4;
__REG32 LTS4            : 4;
__REG32 LTS5            : 4;
__REG32 LTS6            : 4;
__REG32 LTS7            : 4;
} __ccm_ltbr0_bits;

/* Load Tracking Buffer Register (LTBR1) */
typedef struct {
__REG32 LTS8            : 4;
__REG32 LTS9            : 4;
__REG32 LTS10           : 4;
__REG32 LTS11           : 4;
__REG32 LTS12           : 4;
__REG32 LTS13           : 4;
__REG32 LTS14           : 4;
__REG32 LTS15           : 4;
} __ccm_ltbr1_bits;

/* Power Management Control Register 0 (PMCR0) */
typedef struct {
__REG32 DPTEN           : 1;
__REG32 PTVAI           : 2;
__REG32 PTVAIM          : 1;
__REG32 DVFEN           : 1;
__REG32 DCR             : 1;
__REG32 DRCE0           : 1;
__REG32 DRCE1           : 1;
__REG32 DRCE2           : 1;
__REG32 DRCE3           : 1;
__REG32 WFIM            : 1;
__REG32 DPVV            : 1;
__REG32 DPVCR           : 1;
__REG32 FSVAI           : 2;
__REG32 FSVAIM          : 1;
__REG32 UPDTEN          : 1;
__REG32 PTVIS           : 1;
__REG32 LBCF            : 2;
__REG32 LBFL            : 1;
__REG32 LBMI            : 1;
__REG32 DVFIS           : 1;
__REG32 DVFEV           : 1;
__REG32 VSCNT           : 3;
__REG32 UDSC            : 1;
__REG32 DVSUP           : 2;
__REG32 DFSUP           : 2;
} __ccm_pmcr0_bits;

/* Power Management Control Register 1 (PMCR1) */
typedef struct {
__REG32 DVGP            : 4;
__REG32                 : 2;
__REG32 CPFA            : 1;
__REG32 NWTS            : 1;
__REG32 PWTS            : 1;
__REG32 CPSPA           : 4;
__REG32                 : 3;
__REG32 WBCN            : 8;
__REG32                 : 8;
} __ccm_pmcr1_bits;

/* Post Divider Register 2 (PDR2) */
typedef struct {
__REG32 MST1_PDF        : 6;
__REG32                 : 1;
__REG32 MST2_PDF        : 6;
__REG32                 :19;
} __ccm_pdr2_bits;

/* General Purpose Register (GPR) */
typedef struct {
__REG32 PGP_FIRI          : 1;
__REG32 DDR_MODE          : 1;
__REG32 PGP_CSPI_BB       : 1;
__REG32 PGP_ATA_1         : 1;
__REG32 PGP_ATA_2         : 1;
__REG32 PGP_ATA_3         : 1;
__REG32 PGP_ATA_4         : 1;
__REG32 PGP_ATA_5         : 1;
__REG32 PGP_ATA_6         : 1;
__REG32 PGP_ATA_7         : 1;
__REG32 PGP_ATA_8         : 1;
__REG32 PGP_UH2           : 1;
__REG32 SDCTL_CSD0_SEL_B  : 1;
__REG32 SDCTL_CSD1_SEL_B  : 1;
__REG32 CSPI1_UART3       : 1;
__REG32 EXTDMAREQ2_MBX_SE : 1;
__REG32 TAMPER_DETECT_EN  : 1;
__REG32 PGP_USB_4WIRE     : 1;
__REG32 PGP_USB_COMMON    : 1;
__REG32 SDHC_MEMSTICK1    : 1;
__REG32 SDHC_MEMSTICK2    : 1;
__REG32 PGP_SPLL_BYP      : 1;
__REG32 PGP_UPLL_BYP      : 1;
__REG32 MAX_DRIVE_MUXED   : 1;
__REG32 SLEWRATE_MUXED    : 1;
__REG32 CSPI3_UART5_SEL   : 1;
__REG32 PGP_ATA_9         : 1;
__REG32 PGP_USB_SUSPEND   : 1;
__REG32 PGP_USB_OTG_LB    : 1;
__REG32 PGP_USB_HS1_LB    : 1;
__REG32 PGP_USB_HS2_LB    : 1;
__REG32 CLKO_DDR_MODE     : 1;
} __iomux_gpr_bits;

/* Software MUX Control Register (SW_MUX_CTL) */
typedef struct {
__REG32 SW_MUX_IN_EN0   : 4;
__REG32 SW_MUX_OUT_EN0  : 3;
__REG32                 : 1;
__REG32 SW_MUX_IN_EN1   : 4;
__REG32 SW_MUX_OUT_EN1  : 3;
__REG32                 : 1;
__REG32 SW_MUX_IN_EN2   : 4;
__REG32 SW_MUX_OUT_EN2  : 3;
__REG32                 : 1;
__REG32 SW_MUX_IN_EN3   : 4;
__REG32 SW_MUX_OUT_EN3  : 3;
__REG32                 : 1;
} __iomux_sw_mux_ctl_bits;

/* Software PAD Control Register (SW_PAD_CTL) */
typedef struct {
__REG32 IPP_SRE0        : 1;
__REG32 IPP_DSE0        : 2;
__REG32 IPP_ODE0        : 1;
__REG32 IPP_HYS0        : 1;
__REG32 IPP_PUS0        : 2;
__REG32 IPP_PUE0        : 1;
__REG32 IPP_PKE0        : 1;
__REG32 LOOPBACK0       : 1;
__REG32 IPP_SRE1        : 1;
__REG32 IPP_DSE1        : 2;
__REG32 IPP_ODE1        : 1;
__REG32 IPP_HYS1        : 1;
__REG32 IPP_PUS1        : 2;
__REG32 IPP_PUE1        : 1;
__REG32 IPP_PKE1        : 1;
__REG32 LOOPBACK1       : 1;
__REG32 IPP_SRE2        : 1;
__REG32 IPP_DSE2        : 2;
__REG32 IPP_ODE2        : 1;
__REG32 IPP_HYS2        : 1;
__REG32 IPP_PUS2        : 2;
__REG32 IPP_PUE2        : 1;
__REG32 IPP_PKE2        : 1;
__REG32 LOOPBACK2       : 1;
__REG32                 : 2;
} __iomux_sw_pad_ctl_bits;

/* GPIO Interrupt Configuration Register1 (ICR1) */
typedef struct {
__REG32 ICR0            : 2;
__REG32 ICR1            : 2;
__REG32 ICR2            : 2;
__REG32 ICR3            : 2;
__REG32 ICR4            : 2;
__REG32 ICR5            : 2;
__REG32 ICR6            : 2;
__REG32 ICR7            : 2;
__REG32 ICR8            : 2;
__REG32 ICR9            : 2;
__REG32 ICR10           : 2;
__REG32 ICR11           : 2;
__REG32 ICR12           : 2;
__REG32 ICR13           : 2;
__REG32 ICR14           : 2;
__REG32 ICR15           : 2;
} __icr1_bits;

/* GPIO Interrupt Configuration Register2 (ICR2) */
typedef struct {
__REG32 ICR16           : 2;
__REG32 ICR17           : 2;
__REG32 ICR18           : 2;
__REG32 ICR19           : 2;
__REG32 ICR20           : 2;
__REG32 ICR21           : 2;
__REG32 ICR22           : 2;
__REG32 ICR23           : 2;
__REG32 ICR24           : 2;
__REG32 ICR25           : 2;
__REG32 ICR26           : 2;
__REG32 ICR27           : 2;
__REG32 ICR28           : 2;
__REG32 ICR29           : 2;
__REG32 ICR30           : 2;
__REG32 ICR31           : 2;
} __icr2_bits;

typedef struct {        /* Interrupt Control Register (0x10040000) Reset (0x00000000)                   */
__REG32          :18;     /* Bits 17-0    - Reserved*/
__REG32 NM     	 : 1;     /* Bit 18*/
__REG32 FIAD     : 1;     /* Bit 19       - Fast Interrupt Arbiter Disable*/
__REG32 NIAD     : 1;     /* Bit 20       - Normal Interrupt Arbiter Disable*/
__REG32 FIDIS    : 1;     /* Bit 21       - Fast Interrupt Disable*/
__REG32 NIDIS    : 1;     /* Bit 22       - Normal Interrupt Disable*/
__REG32          : 1;     /* Bit 23       - Reserved*/
__REG32 ABFEN    : 1;     /* Bit 24       - ABFLAG Sticky Enable*/
__REG32 ABFLAG   : 1;     /* Bit 25       - Core Arbitration Prioritization Risen Flag*/
__REG32          : 6;     /* Bits 26-31   - Reserved*/
} __intcntl_bits;

typedef struct {        /* Normal Interrupt Mask Register (0x10040004) Reset (0x0000001F)                       */
__REG32 NIMASK  : 5;     /* Bits 0-4     - Normal Interrupt Mask (0 = Disable priority level 0 ints, 1 = disable priority 1 and lower... 16+ = disable all interrupts)*/
__REG32         :27;     /* Bits 5-31    - Reserved*/
} __nimask_bits;

typedef struct {        /* Interrupt Enable Number Register (0x10040008) Reset (0x00000000)                     */
__REG32 ENNUM  : 6;     /* Bits 0-5     - Interrupt Enable Number - Enables/Disables the interrupt source associated with this value.*/
__REG32        :26;     /* Bits 6-31    - Reserved*/
} __intennum_bits;

typedef struct {        /* Interrupt Disable Number Register (0x1004000C) Reset (0x00000000)                    */
__REG32 DISNUM  : 6;     /* Bits 0-5     - Interrupt Disable Number - Enables/Disables the interrupt source associated with this value.*/
__REG32         :26;     /* Bits 6-31    - Reserved*/
} __intdisnum_bits;

typedef struct {        /* Interrupt Enable Register High (0x10040010) Reset (0x00000000)                       */
__REG32 INTENABLE32  : 1;     /* Bit  0           - Interrupt Enable*/
__REG32 INTENABLE33  : 1;     /* Bit  1           - Interrupt Enable*/
__REG32 INTENABLE34  : 1;     /* Bit  2           - Interrupt Enable*/
__REG32 INTENABLE35  : 1;     /* Bit  3           - Interrupt Enable*/
__REG32 INTENABLE36  : 1;     /* Bit  4           - Interrupt Enable*/
__REG32 INTENABLE37  : 1;     /* Bit  5           - Interrupt Enable*/
__REG32 INTENABLE38  : 1;     /* Bit  6           - Interrupt Enable*/
__REG32 INTENABLE39  : 1;     /* Bit  7           - Interrupt Enable*/
__REG32 INTENABLE40  : 1;     /* Bit  8           - Interrupt Enable*/
__REG32 INTENABLE41  : 1;     /* Bit  9           - Interrupt Enable*/
__REG32 INTENABLE42  : 1;     /* Bit  10          - Interrupt Enable*/
__REG32 INTENABLE43  : 1;     /* Bit  11          - Interrupt Enable*/
__REG32 INTENABLE44  : 1;     /* Bit  12          - Interrupt Enable*/
__REG32 INTENABLE45  : 1;     /* Bit  13          - Interrupt Enable*/
__REG32 INTENABLE46  : 1;     /* Bit  14          - Interrupt Enable*/
__REG32 INTENABLE47  : 1;     /* Bit  15          - Interrupt Enable*/
__REG32 INTENABLE48  : 1;     /* Bit  16          - Interrupt Enable*/
__REG32 INTENABLE49  : 1;     /* Bit  17          - Interrupt Enable*/
__REG32 INTENABLE50  : 1;     /* Bit  18          - Interrupt Enable*/
__REG32 INTENABLE51  : 1;     /* Bit  19          - Interrupt Enable*/
__REG32 INTENABLE52  : 1;     /* Bit  20          - Interrupt Enable*/
__REG32 INTENABLE53  : 1;     /* Bit  21          - Interrupt Enable*/
__REG32 INTENABLE54  : 1;     /* Bit  22          - Interrupt Enable*/
__REG32 INTENABLE55  : 1;     /* Bit  23          - Interrupt Enable*/
__REG32 INTENABLE56  : 1;     /* Bit  24          - Interrupt Enable*/
__REG32 INTENABLE57  : 1;     /* Bit  25          - Interrupt Enable*/
__REG32 INTENABLE58  : 1;     /* Bit  26          - Interrupt Enable*/
__REG32 INTENABLE59  : 1;     /* Bit  27          - Interrupt Enable*/
__REG32 INTENABLE60  : 1;     /* Bit  28          - Interrupt Enable*/
__REG32 INTENABLE61  : 1;     /* Bit  29          - Interrupt Enable*/
__REG32 INTENABLE62  : 1;     /* Bit  30          - Interrupt Enable*/
__REG32 INTENABLE63  : 1;     /* Bit  31          - Interrupt Enable*/
} __intenableh_bits;

typedef struct {        /* Interrupt Enable Register Low (0x10040014) Reset (0x00000000)                        */
__REG32 INTENABLE0   : 1;     /* Bit  0           - Interrupt Enable*/
__REG32 INTENABLE1   : 1;     /* Bit  1           - Interrupt Enable*/
__REG32 INTENABLE2   : 1;     /* Bit  2           - Interrupt Enable*/
__REG32 INTENABLE3   : 1;     /* Bit  3           - Interrupt Enable*/
__REG32 INTENABLE4   : 1;     /* Bit  4           - Interrupt Enable*/
__REG32 INTENABLE5   : 1;     /* Bit  5           - Interrupt Enable*/
__REG32 INTENABLE6   : 1;     /* Bit  6           - Interrupt Enable*/
__REG32 INTENABLE7   : 1;     /* Bit  7           - Interrupt Enable*/
__REG32 INTENABLE8   : 1;     /* Bit  8           - Interrupt Enable*/
__REG32 INTENABLE9   : 1;     /* Bit  9           - Interrupt Enable*/
__REG32 INTENABLE10  : 1;     /* Bit  10          - Interrupt Enable*/
__REG32 INTENABLE11  : 1;     /* Bit  11          - Interrupt Enable*/
__REG32 INTENABLE12  : 1;     /* Bit  12          - Interrupt Enable*/
__REG32 INTENABLE13  : 1;     /* Bit  13          - Interrupt Enable*/
__REG32 INTENABLE14  : 1;     /* Bit  14          - Interrupt Enable*/
__REG32 INTENABLE15  : 1;     /* Bit  15          - Interrupt Enable*/
__REG32 INTENABLE16  : 1;     /* Bit  16          - Interrupt Enable*/
__REG32 INTENABLE17  : 1;     /* Bit  17          - Interrupt Enable*/
__REG32 INTENABLE18  : 1;     /* Bit  18          - Interrupt Enable*/
__REG32 INTENABLE19  : 1;     /* Bit  19          - Interrupt Enable*/
__REG32 INTENABLE20  : 1;     /* Bit  20          - Interrupt Enable*/
__REG32 INTENABLE21  : 1;     /* Bit  21          - Interrupt Enable*/
__REG32 INTENABLE22  : 1;     /* Bit  22          - Interrupt Enable*/
__REG32 INTENABLE23  : 1;     /* Bit  23          - Interrupt Enable*/
__REG32 INTENABLE24  : 1;     /* Bit  24          - Interrupt Enable*/
__REG32 INTENABLE25  : 1;     /* Bit  25          - Interrupt Enable*/
__REG32 INTENABLE26  : 1;     /* Bit  26          - Interrupt Enable*/
__REG32 INTENABLE27  : 1;     /* Bit  27          - Interrupt Enable*/
__REG32 INTENABLE28  : 1;     /* Bit  28          - Interrupt Enable*/
__REG32 INTENABLE29  : 1;     /* Bit  29          - Interrupt Enable*/
__REG32 INTENABLE30  : 1;     /* Bit  30          - Interrupt Enable*/
__REG32 INTENABLE31  : 1;     /* Bit  31          - Interrupt Enable*/
} __intenablel_bits;

typedef struct {        /* Interrupt Type Register High (0x10040018) Reset (0x00000000)                 */
__REG32 INTTYPE32  : 1;     /* Bit  0         - Interrupt Enable*/
__REG32 INTTYPE33  : 1;     /* Bit  1         - Interrupt Enable*/
__REG32 INTTYPE34  : 1;     /* Bit  2         - Interrupt Enable*/
__REG32 INTTYPE35  : 1;     /* Bit  3         - Interrupt Enable*/
__REG32 INTTYPE36  : 1;     /* Bit  4         - Interrupt Enable*/
__REG32 INTTYPE37  : 1;     /* Bit  5         - Interrupt Enable*/
__REG32 INTTYPE38  : 1;     /* Bit  6         - Interrupt Enable*/
__REG32 INTTYPE39  : 1;     /* Bit  7         - Interrupt Enable*/
__REG32 INTTYPE40  : 1;     /* Bit  8         - Interrupt Enable*/
__REG32 INTTYPE41  : 1;     /* Bit  9         - Interrupt Enable*/
__REG32 INTTYPE42  : 1;     /* Bit  10        - Interrupt Enable*/
__REG32 INTTYPE43  : 1;     /* Bit  11        - Interrupt Enable*/
__REG32 INTTYPE44  : 1;     /* Bit  12        - Interrupt Enable*/
__REG32 INTTYPE45  : 1;     /* Bit  13        - Interrupt Enable*/
__REG32 INTTYPE46  : 1;     /* Bit  14        - Interrupt Enable*/
__REG32 INTTYPE47  : 1;     /* Bit  15        - Interrupt Enable*/
__REG32 INTTYPE48  : 1;     /* Bit  16        - Interrupt Enable*/
__REG32 INTTYPE49  : 1;     /* Bit  17        - Interrupt Enable*/
__REG32 INTTYPE50  : 1;     /* Bit  18        - Interrupt Enable*/
__REG32 INTTYPE51  : 1;     /* Bit  19        - Interrupt Enable*/
__REG32 INTTYPE52  : 1;     /* Bit  20        - Interrupt Enable*/
__REG32 INTTYPE53  : 1;     /* Bit  21        - Interrupt Enable*/
__REG32 INTTYPE54  : 1;     /* Bit  22        - Interrupt Enable*/
__REG32 INTTYPE55  : 1;     /* Bit  23        - Interrupt Enable*/
__REG32 INTTYPE56  : 1;     /* Bit  24        - Interrupt Enable*/
__REG32 INTTYPE57  : 1;     /* Bit  25        - Interrupt Enable*/
__REG32 INTTYPE58  : 1;     /* Bit  26        - Interrupt Enable*/
__REG32 INTTYPE59  : 1;     /* Bit  27        - Interrupt Enable*/
__REG32 INTTYPE60  : 1;     /* Bit  28        - Interrupt Enable*/
__REG32 INTTYPE61  : 1;     /* Bit  29        - Interrupt Enable*/
__REG32 INTTYPE62  : 1;     /* Bit  30        - Interrupt Enable*/
__REG32 INTTYPE63  : 1;     /* Bit  31        - Interrupt Enable*/
} __inttypeh_bits;

typedef struct {        /* Interrupt Enable Register Low (0x1004001C) Reset (0x00000000)                        */
__REG32 INTTYPE0   : 1;     /* Bit  0         - Interrupt Enable*/
__REG32 INTTYPE1   : 1;     /* Bit  1         - Interrupt Enable*/
__REG32 INTTYPE2   : 1;     /* Bit  2         - Interrupt Enable*/
__REG32 INTTYPE3   : 1;     /* Bit  3         - Interrupt Enable*/
__REG32 INTTYPE4   : 1;     /* Bit  4         - Interrupt Enable*/
__REG32 INTTYPE5   : 1;     /* Bit  5         - Interrupt Enable*/
__REG32 INTTYPE6   : 1;     /* Bit  6         - Interrupt Enable*/
__REG32 INTTYPE7   : 1;     /* Bit  7         - Interrupt Enable*/
__REG32 INTTYPE8   : 1;     /* Bit  8         - Interrupt Enable*/
__REG32 INTTYPE9   : 1;     /* Bit  9         - Interrupt Enable*/
__REG32 INTTYPE10  : 1;     /* Bit  10        - Interrupt Enable*/
__REG32 INTTYPE11  : 1;     /* Bit  11        - Interrupt Enable*/
__REG32 INTTYPE12  : 1;     /* Bit  12        - Interrupt Enable*/
__REG32 INTTYPE13  : 1;     /* Bit  13        - Interrupt Enable*/
__REG32 INTTYPE14  : 1;     /* Bit  14        - Interrupt Enable*/
__REG32 INTTYPE15  : 1;     /* Bit  15        - Interrupt Enable*/
__REG32 INTTYPE16  : 1;     /* Bit  16        - Interrupt Enable*/
__REG32 INTTYPE17  : 1;     /* Bit  17        - Interrupt Enable*/
__REG32 INTTYPE18  : 1;     /* Bit  18        - Interrupt Enable*/
__REG32 INTTYPE19  : 1;     /* Bit  19        - Interrupt Enable*/
__REG32 INTTYPE20  : 1;     /* Bit  20        - Interrupt Enable*/
__REG32 INTTYPE21  : 1;     /* Bit  21        - Interrupt Enable*/
__REG32 INTTYPE22  : 1;     /* Bit  22        - Interrupt Enable*/
__REG32 INTTYPE23  : 1;     /* Bit  23        - Interrupt Enable*/
__REG32 INTTYPE24  : 1;     /* Bit  24        - Interrupt Enable*/
__REG32 INTTYPE25  : 1;     /* Bit  25        - Interrupt Enable*/
__REG32 INTTYPE26  : 1;     /* Bit  26        - Interrupt Enable*/
__REG32 INTTYPE27  : 1;     /* Bit  27        - Interrupt Enable*/
__REG32 INTTYPE28  : 1;     /* Bit  28        - Interrupt Enable*/
__REG32 INTTYPE29  : 1;     /* Bit  29        - Interrupt Enable*/
__REG32 INTTYPE30  : 1;     /* Bit  30        - Interrupt Enable*/
__REG32 INTTYPE31  : 1;     /* Bit  31        - Interrupt Enable*/
} __inttypel_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 7 (0x10040020) Reset (0x00000000)                   */
__REG32 NIPR56  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR57  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR58  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR59  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR60  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR61  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR62  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR63  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority7_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 6 (0x10040024) Reset (0x00000000)                   */
__REG32 NIPR48  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR49  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR50  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR51  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR52  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR53  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR54  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR55  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority6_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 5 (0x10040028) Reset (0x00000000)                   */
__REG32 NIPR40  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR41  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR42  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR43  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR44  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR45  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR46  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR47  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority5_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 4 (0x1004002C) Reset (0x00000000)                   */
__REG32 NIPR32  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR33  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR34  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR35  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR36  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR37  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR38  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR39  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority4_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 3 (0x10040030) Reset (0x00000000)                   */
__REG32 NIPR24  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR25  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR26  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR27  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR28  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR29  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR30  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR31  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority3_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 2 (0x10040034) Reset (0x00000000)                   */
__REG32 NIPR16  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR17  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR18  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR19  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR20  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR21  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR22  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR23  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority2_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 1 (0x10040038) Reset (0x00000000)                   */
__REG32 NIPR8   : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR9   : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR10  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR11  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR12  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR13  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR14  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR15  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority1_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 0 (0x1004003C) Reset (0x00000000)                   */
__REG32 NIPR0  : 4;     /* Bits 0  - 3  - Normal Interrupt Priority Level*/
__REG32 NIPR1  : 4;     /* Bits 4  - 7  - Normal Interrupt Priority Level*/
__REG32 NIPR2  : 4;     /* Bits 8  - 11 - Normal Interrupt Priority Level*/
__REG32 NIPR3  : 4;     /* Bits 12 - 15 - Normal Interrupt Priority Level*/
__REG32 NIPR4  : 4;     /* Bits 16 - 19 - Normal Interrupt Priority Level*/
__REG32 NIPR5  : 4;     /* Bits 20 - 23 - Normal Interrupt Priority Level*/
__REG32 NIPR6  : 4;     /* Bits 24 - 27 - Normal Interrupt Priority Level*/
__REG32 NIPR7  : 4;     /* Bits 28 - 31 - Normal Interrupt Priority Level*/
} __nipriority0_bits;

typedef struct {        /* Normal Interrupt Vector and Status Register (0x10040040) Reset (0xFFFFFFFF)  */
__REG32 NIPRILVL  :16;     /* Bits 0  - 15 - Normal Interrupt Priority Level - Indicates the priority level of the highest priority normal interrupt.*/
__REG32 NIVECTOR  :16;     /* Bits 16 - 31 - Normal Interrupt Vector - Indicates vector index for the highest pending normal interrupt.*/
} __nivecsr_bits;

typedef struct {        /* Interrupt Source Register High (0x10040048) Reset (0x00000000)       */
__REG32 INTIN32  : 1;     /* Bit  0       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN33  : 1;     /* Bit  1       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN34  : 1;     /* Bit  2       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN35  : 1;     /* Bit  3       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN36  : 1;     /* Bit  4       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN37  : 1;     /* Bit  5       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN38  : 1;     /* Bit  6       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN39  : 1;     /* Bit  7       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN40  : 1;     /* Bit  8       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN41  : 1;     /* Bit  9       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN42  : 1;     /* Bit  10      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN43  : 1;     /* Bit  11      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN44  : 1;     /* Bit  12      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN45  : 1;     /* Bit  13      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN46  : 1;     /* Bit  14      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN47  : 1;     /* Bit  15      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN48  : 1;     /* Bit  16      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN49  : 1;     /* Bit  17      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN50  : 1;     /* Bit  18      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN51  : 1;     /* Bit  19      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN52  : 1;     /* Bit  20      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN53  : 1;     /* Bit  21      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN54  : 1;     /* Bit  22      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN55  : 1;     /* Bit  23      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN56  : 1;     /* Bit  24      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN57  : 1;     /* Bit  25      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN58  : 1;     /* Bit  26      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN59  : 1;     /* Bit  27      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN60  : 1;     /* Bit  28      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN61  : 1;     /* Bit  29      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN62  : 1;     /* Bit  30      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN63  : 1;     /* Bit  31      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
} __intsrch_bits;

typedef struct {        /* Interrupt Source Register Low (0x1004004C) Reset (0x00000000)        */
__REG32 INTIN0   : 1;     /* Bit  0       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN1   : 1;     /* Bit  1       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN2   : 1;     /* Bit  2       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN3   : 1;     /* Bit  3       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN4   : 1;     /* Bit  4       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN5   : 1;     /* Bit  5       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN6   : 1;     /* Bit  6       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN7   : 1;     /* Bit  7       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN8   : 1;     /* Bit  8       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN9   : 1;     /* Bit  9       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN10  : 1;     /* Bit  10      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN11  : 1;     /* Bit  11      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN12  : 1;     /* Bit  12      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN13  : 1;     /* Bit  13      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN14  : 1;     /* Bit  14      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN15  : 1;     /* Bit  15      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN16  : 1;     /* Bit  16      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN17  : 1;     /* Bit  17      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN18  : 1;     /* Bit  18      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN19  : 1;     /* Bit  19      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN20  : 1;     /* Bit  20      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN21  : 1;     /* Bit  21      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN22  : 1;     /* Bit  22      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN23  : 1;     /* Bit  23      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN24  : 1;     /* Bit  24      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN25  : 1;     /* Bit  25      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN26  : 1;     /* Bit  26      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN27  : 1;     /* Bit  27      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN28  : 1;     /* Bit  28      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN29  : 1;     /* Bit  29      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN30  : 1;     /* Bit  30      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN31  : 1;     /* Bit  31      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
} __intsrcl_bits;

typedef struct {        /* Interrupt Force Register High (0x10040050) Reset (0x00000000)        */
__REG32 FORCE32  : 1;     /* Bit  0       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE33  : 1;     /* Bit  1       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE34  : 1;     /* Bit  2       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE35  : 1;     /* Bit  3       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE36  : 1;     /* Bit  4       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE37  : 1;     /* Bit  5       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE38  : 1;     /* Bit  6       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE39  : 1;     /* Bit  7       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE40  : 1;     /* Bit  8       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE41  : 1;     /* Bit  9       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE42  : 1;     /* Bit  10      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE43  : 1;     /* Bit  11      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE44  : 1;     /* Bit  12      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE45  : 1;     /* Bit  13      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE46  : 1;     /* Bit  14      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE47  : 1;     /* Bit  15      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE48  : 1;     /* Bit  16      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE49  : 1;     /* Bit  17      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE50  : 1;     /* Bit  18      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE51  : 1;     /* Bit  19      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE52  : 1;     /* Bit  20      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE53  : 1;     /* Bit  21      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE54  : 1;     /* Bit  22      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE55  : 1;     /* Bit  23      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE56  : 1;     /* Bit  24      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE57  : 1;     /* Bit  25      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE58  : 1;     /* Bit  26      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE59  : 1;     /* Bit  27      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE60  : 1;     /* Bit  28      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE61  : 1;     /* Bit  29      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE62  : 1;     /* Bit  30      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE63  : 1;     /* Bit  31      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
} __intfrch_bits;

typedef struct {        /* Interrupt Force Register Low (0x10040054) Reset (0x00000000) */
__REG32 FORCE0   : 1;     /* Bit  0       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE1   : 1;     /* Bit  1       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE2   : 1;     /* Bit  2       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE3   : 1;     /* Bit  3       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE4   : 1;     /* Bit  4       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE5   : 1;     /* Bit  5       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE6   : 1;     /* Bit  6       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE7   : 1;     /* Bit  7       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE8   : 1;     /* Bit  8       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE9   : 1;     /* Bit  9       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE10  : 1;     /* Bit  10      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE11  : 1;     /* Bit  11      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE12  : 1;     /* Bit  12      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE13  : 1;     /* Bit  13      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE14  : 1;     /* Bit  14      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE15  : 1;     /* Bit  15      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE16  : 1;     /* Bit  16      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE17  : 1;     /* Bit  17      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE18  : 1;     /* Bit  18      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE19  : 1;     /* Bit  19      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE20  : 1;     /* Bit  20      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE21  : 1;     /* Bit  21      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE22  : 1;     /* Bit  22      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE23  : 1;     /* Bit  23      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE24  : 1;     /* Bit  24      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE25  : 1;     /* Bit  25      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE26  : 1;     /* Bit  26      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE27  : 1;     /* Bit  27      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE28  : 1;     /* Bit  28      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE29  : 1;     /* Bit  29      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE30  : 1;     /* Bit  30      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE31  : 1;     /* Bit  31      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
} __intfrcl_bits;

typedef struct {        /* Normal Interrupt Pending Register High (0x10040058) Reset (0x00000000)       */
__REG32 NIPEND32  : 1;     /* Bit  0       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND33  : 1;     /* Bit  1       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND34  : 1;     /* Bit  2       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND35  : 1;     /* Bit  3       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND36  : 1;     /* Bit  4       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND37  : 1;     /* Bit  5       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND38  : 1;     /* Bit  6       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND39  : 1;     /* Bit  7       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND40  : 1;     /* Bit  8       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND41  : 1;     /* Bit  9       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND42  : 1;     /* Bit  10      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND43  : 1;     /* Bit  11      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND44  : 1;     /* Bit  12      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND45  : 1;     /* Bit  13      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND46  : 1;     /* Bit  14      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND47  : 1;     /* Bit  15      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND48  : 1;     /* Bit  16      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND49  : 1;     /* Bit  17      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND50  : 1;     /* Bit  18      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND51  : 1;     /* Bit  19      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND52  : 1;     /* Bit  20      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND53  : 1;     /* Bit  21      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND54  : 1;     /* Bit  22      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND55  : 1;     /* Bit  23      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND56  : 1;     /* Bit  24      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND57  : 1;     /* Bit  25      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND58  : 1;     /* Bit  26      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND59  : 1;     /* Bit  27      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND60  : 1;     /* Bit  28      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND61  : 1;     /* Bit  29      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND62  : 1;     /* Bit  30      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND63  : 1;     /* Bit  31      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
} __nipndh_bits;

typedef struct {        /* Normal Interrupt Pending Register Low (0x1004005C) Reset (0x00000000)        */
__REG32 NIPEND0   : 1;     /* Bit  0       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND1   : 1;     /* Bit  1       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND2   : 1;     /* Bit  2       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND3   : 1;     /* Bit  3       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND4   : 1;     /* Bit  4       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND5   : 1;     /* Bit  5       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND6   : 1;     /* Bit  6       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND7   : 1;     /* Bit  7       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND8   : 1;     /* Bit  8       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND9   : 1;     /* Bit  9       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND10  : 1;     /* Bit  10      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND11  : 1;     /* Bit  11      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND12  : 1;     /* Bit  12      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND13  : 1;     /* Bit  13      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND14  : 1;     /* Bit  14      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND15  : 1;     /* Bit  15      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND16  : 1;     /* Bit  16      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND17  : 1;     /* Bit  17      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND18  : 1;     /* Bit  18      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND19  : 1;     /* Bit  19      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND20  : 1;     /* Bit  20      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND21  : 1;     /* Bit  21      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND22  : 1;     /* Bit  22      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND23  : 1;     /* Bit  23      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND24  : 1;     /* Bit  24      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND25  : 1;     /* Bit  25      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND26  : 1;     /* Bit  26      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND27  : 1;     /* Bit  27      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND28  : 1;     /* Bit  28      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND29  : 1;     /* Bit  29      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND30  : 1;     /* Bit  30      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND31  : 1;     /* Bit  31      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
} __nipndl_bits;

typedef struct {        /* Fast Interrupt Pending Register High (0x10040060) Reset (0x00000000) */
__REG32 FIPEND32  : 1;     /* Bit  0       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND33  : 1;     /* Bit  1       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND34  : 1;     /* Bit  2       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND35  : 1;     /* Bit  3       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND36  : 1;     /* Bit  4       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND37  : 1;     /* Bit  5       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND38  : 1;     /* Bit  6       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND39  : 1;     /* Bit  7       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND40  : 1;     /* Bit  8       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND41  : 1;     /* Bit  9       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND42  : 1;     /* Bit  10      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND43  : 1;     /* Bit  11      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND44  : 1;     /* Bit  12      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND45  : 1;     /* Bit  13      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND46  : 1;     /* Bit  14      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND47  : 1;     /* Bit  15      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND48  : 1;     /* Bit  16      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND49  : 1;     /* Bit  17      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND50  : 1;     /* Bit  18      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND51  : 1;     /* Bit  19      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND52  : 1;     /* Bit  20      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND53  : 1;     /* Bit  21      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND54  : 1;     /* Bit  22      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND55  : 1;     /* Bit  23      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND56  : 1;     /* Bit  24      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND57  : 1;     /* Bit  25      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND58  : 1;     /* Bit  26      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND59  : 1;     /* Bit  27      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND60  : 1;     /* Bit  28      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND61  : 1;     /* Bit  29      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND62  : 1;     /* Bit  30      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND63  : 1;     /* Bit  31      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
} __fipndh_bits;

typedef struct {        /* Fast Interrupt Pending Register Low (0x10040064) Reset (0x00000000)  */
__REG32 FIPEND0   : 1;     /* Bit  0       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND1   : 1;     /* Bit  1       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND2   : 1;     /* Bit  2       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND3   : 1;     /* Bit  3       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND4   : 1;     /* Bit  4       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND5   : 1;     /* Bit  5       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND6   : 1;     /* Bit  6       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND7   : 1;     /* Bit  7       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND8   : 1;     /* Bit  8       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND9   : 1;     /* Bit  9       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND10  : 1;     /* Bit  10      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND11  : 1;     /* Bit  11      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND12  : 1;     /* Bit  12      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND13  : 1;     /* Bit  13      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND14  : 1;     /* Bit  14      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND15  : 1;     /* Bit  15      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND16  : 1;     /* Bit  16      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND17  : 1;     /* Bit  17      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND18  : 1;     /* Bit  18      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND19  : 1;     /* Bit  19      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND20  : 1;     /* Bit  20      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND21  : 1;     /* Bit  21      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND22  : 1;     /* Bit  22      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND23  : 1;     /* Bit  23      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND24  : 1;     /* Bit  24      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND25  : 1;     /* Bit  25      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND26  : 1;     /* Bit  26      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND27  : 1;     /* Bit  27      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND28  : 1;     /* Bit  28      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND29  : 1;     /* Bit  29      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND30  : 1;     /* Bit  30      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND31  : 1;     /* Bit  31      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
} __fipndl_bits;

/* Red Start Register (RSR) */
typedef struct{
__REG32 RED_START  : 7;
__REG32            :25;
} __scc_rsr_bits;

/* Black Start Register (BSR) */
typedef struct{
__REG32 BLACK_START  : 7;
__REG32              :25;
} __scc_bsr_bits;

/* Length Register (LENR) */
typedef struct{
__REG32 LENGTH  : 7;
__REG32         :25;
} __scc_lenr_bits;

/* Control Register (CTLR) */
typedef struct{
__REG32 CIPHERMODE      : 1;
__REG32 CHAININGMODE    : 1;
__REG32 CIPHERSTART     : 1;
__REG32                 :29;
} __scc_ctlr_bits;

/* SCM Status Register (SCMR) */
typedef struct{
__REG32 BUSY           : 1;
__REG32 ZEROIZING      : 1;
__REG32 CIPHERING      : 1;
__REG32 BLOCKACCESS    : 1;
__REG32 ZEROIZEFAILED  : 1;
__REG32 BADKEY         : 1;
__REG32 INTERNERROR    : 1;
__REG32 SECRETKEY      : 1;
__REG32 INTSTATUS      : 1;
__REG32 ZEROIZINGDONE  : 1;
__REG32 CIPHERINGDONE  : 1;
__REG32 ACCESSALLOWED  : 1;
__REG32 LENGTHERROR    : 1;
__REG32                :19;
} __scc_scmr_bits;

/* Error Status Register (ERRSR) */
typedef struct{
__REG32 BUSY            : 1;
__REG32 ZEROIZING       : 1;
__REG32 CIPHERING       : 1;
__REG32 BLOCKACCESS     : 1;
__REG32 ZEROIZEFAILED   : 1;
__REG32 BADKEY          : 1;
__REG32 INTERNERROR     : 1;
__REG32 SECRETKEY       : 1;
__REG32 USERACCESS      : 1;
__REG32 ILLEGALADDRESS  : 1;
__REG32 BYTEACCESS      : 1;
__REG32 UNALIGNEDACCESS : 1;
__REG32 ILLEGALMASTER   : 1;
__REG32 CACHEABLEACCESS : 1;
__REG32                 :18;
} __scc_errsr_bits;

/* Interrupt Control Register (INTCR) */
typedef struct{
__REG32 INTERRUPTMASK   : 1;
__REG32 CLEARINTERRUPT  : 1;
__REG32 ZEROIZEMEMOR    : 1;
__REG32                 :29;
} __scc_intcr_bits;

/* Configuration Register (CONFIGR) */
typedef struct{
__REG32 BLOCK_SIZE  : 7;
__REG32 RED_SIZE    :10;
__REG32 BLACK_SIZE  :10;
__REG32 VERSION     : 5;
} __scc_configr_bits;

/* SMN Status Register (SMNSR) */
typedef struct{
__REG32 STATE     : 5;
__REG32 IB        : 1;
__REG32 ZF        : 1;
__REG32 DA        : 1;
__REG32 SPE       : 1;
__REG32 ASCE      : 1;
__REG32 BBE       : 1;
__REG32 PCE       : 1;
__REG32 TE        : 1;
__REG32 SA        : 1;
__REG32 SMNIRQ    : 1;
__REG32 SCME      : 1;
__REG32 IA        : 1;
__REG32 BK        : 1;
__REG32 DK        : 1;
__REG32 USRA      : 1;
__REG32 IADDR     : 1;
__REG32 BY        : 1;
__REG32 UA        : 1;
__REG32 SE        : 1;
__REG32 IM        : 1;
__REG32 CA        : 1;
__REG32 VERSION   : 6;
} __scc_smnsr_bits;

/* Command Register (CMDR) */
typedef struct{
__REG32 SA  : 1;
__REG32 EI  : 1;
__REG32 CBB : 1;
__REG32 CI  : 1;
__REG32     :28;
} __scc_cmdr_bits;

/* Sequence Start Register (SEQSR) */
typedef struct{
__REG32 START_VALUE  :16;
__REG32              :16;
} __scc_seqsr_bits;

/* Sequence End Register (SEQER) */
typedef struct{
__REG32 END_VALUE  :16;
__REG32            :16;
} __scc_seqer_bits;

/* Sequence Check Register (SEQCR) */
typedef struct{
__REG32 CHECK_VALUE  :16;
__REG32              :16;
} __scc_seqcr_bits;

/* Bit Count Register (BCNTR) */
typedef struct{
__REG32 BITCNT  :11;
__REG32         :21;
} __scc_bcntr_bits;

/* Bit Bank Increment Size Register (BISR) */
typedef struct{
__REG32 INC_SIZE  :11;
__REG32           :21;
} __scc_bisr_bits;

/* Bit Bank Decrement Register (BBDR) */
typedef struct{
__REG32 DEC_AMT  :11;
__REG32          :21;
} __scc_bbdr_bits;

/* Compare Size Register (SIZER) */
typedef struct{
__REG32 SIZE  : 6;
__REG32       :26;
} __scc_sizer_bits;

/*Timer Control Register (TIMECTR) */
typedef struct{
__REG32 ST  : 1;
__REG32 LD  : 1;
__REG32     :30;
} __scc_timectr_bits;

/* Debug Detector Status Register (DEBUGR) */
typedef struct{
__REG32 D1  : 1;
__REG32 D2  : 1;
__REG32 D3  : 1;
__REG32 D4  : 1;
__REG32 D5  : 1;
__REG32 D6  : 1;
__REG32 D7  : 1;
__REG32 D8  : 1;
__REG32 D9  : 1;
__REG32 D10 : 1;
__REG32 D11 : 1;
__REG32 D12 : 1;
__REG32     :20;
} __scc_debugr_bits;

/* Control Register (RNGCR) */
typedef struct{
__REG32 GO        : 1;
__REG32 H_ASSURE  : 1;
__REG32 INT_MASK  : 1;
__REG32 CLR_INT   : 1;
__REG32 SLEEP     : 1;
__REG32           :27;
} __rngcr_bits;

/* Status Register (RNGSR) */
typedef struct{
__REG32 SECURITY        : 1;
__REG32 LAST_READ       : 1;
__REG32 FIFO_UNDER      : 1;
__REG32 ERROR_I         : 1;
__REG32 SLEEP           : 1;
__REG32                 : 3;
__REG32 OUT_FIFO_LEVEL  : 8;
__REG32 FIFO_SIZE       : 8;
__REG32                 : 7;
__REG32 OSC_DEAD        : 1;
} __rngsr_bits;

/* Mode Register (RNGMOD) */
typedef struct{
__REG32 VERIFY    : 1;
__REG32 OSC_TEST  : 1;
__REG32           :30;
} __rngmod_bits;

/* Verification Control Register (RNGVER) */
typedef struct{
__REG32 CLK_OFF       : 1;
__REG32 FORCE_SYSCLK  : 1;
__REG32 RST_SHFT      : 1;
__REG32               :29;
} __rngver_bits;

/* Oscillator Counter Control Register (RNGOSCCR) */
typedef struct{
__REG32 REMAINING_CLK_CYC  :18;
__REG32                    :14;
} __rngosccr_bits;

/* Oscillator 1,2 Counter (RNGOSCCT) */
typedef struct{
__REG32 COUNTER   :20;
__REG32           :12;
} __rngoscct_bits;

/* Oscillator Counter Status (RNGOSCSTAT) */
typedef struct{
__REG32 OSC1  : 1;
__REG32 OSC2  : 1;
__REG32       :30;
} __rngoscstat_bits;

/* RTIC Status Register (STATUS) */
typedef struct{
__REG32 BUSY            : 1;
__REG32 HASH_DONE       : 1;
__REG32 HASH_ERR        : 1;
__REG32 WD_ERR          : 1;
__REG32 MEMORY_INT_STA  : 4;
__REG32 ADDRESS_ERROR   : 4;
__REG32 LENGTH_ERROR    : 4;
__REG32 STATUS0         : 9;
__REG32 RTIC_STATE      : 2;
__REG32 STATUS1         : 5;
} __rtic_status_bits;

/* Command Register (COMMAND) */
typedef struct{
__REG32 CLR_IRQ       : 1;
__REG32 SW_RST        : 1;
__REG32 HASH_ONCE     : 1;
__REG32 RT_CHK  			: 1;
__REG32 RT_DISABLE    : 1;
__REG32               :27;
} __rtic_command_bits;

/* Control Register (CTRL) */
typedef struct{
__REG32 IRQ_EN            	: 1;
__REG32 DMA_BURST_SIZ     	: 3;
__REG32 HASH_ONCE_MEM_EN  	: 4;
__REG32 RUNTIME_MEM_EN    	: 4;
__REG32 BYTESWAP          	: 1;
__REG32 HALFWORDSWAP      	: 1;
__REG32                   	: 2;
__REG32 HASH_ONCE_MODE_DMA	: 8;
__REG32                   	: 8;
} __rtic_ctrl_bits;

/* Watchdog Time-out Register (WDOG) */
typedef struct{
__REG32 DMA_WDOG  :16;
__REG32           :16;
} __rtic_wdog_bits;

/* Status Register (STAT) */
typedef struct{
__REG8  SNSD      : 1;
__REG8  PRGD      : 1;
__REG8            : 5;
__REG8  BUSY      : 1;
} __iim_stat_bits;

/* Status IRQ Mask (STATM) */
typedef struct{
__REG8  SNSD_M    : 1;
__REG8  PRGD_M    : 1;
__REG8            : 6;
} __iim_statm_bits;

/* Module Errors Register (ERR) */
typedef struct{
__REG8            : 1;
__REG8  PARITYE   : 1;
__REG8  SNSE      : 1;
__REG8  WLRE      : 1;
__REG8  RPE       : 1;
__REG8  OPE       : 1;
__REG8  WPE       : 1;
__REG8  PRGE      : 1;
} __iim_err_bits;

/* Error IRQ Mask Register (EMASK) */
typedef struct{
__REG8            : 1;
__REG8  PARITYE_M : 1;
__REG8  SNSE_M    : 1;
__REG8  WLRE_M    : 1;
__REG8  RPE_M     : 1;
__REG8  OPE_M     : 1;
__REG8  WPE_M     : 1;
__REG8  PRGE_M    : 1;
} __iim_emask_bits;

/* Fuse Control Register (FCTL) */
typedef struct{
__REG8  PRG       : 1;
__REG8  ESNS_1    : 1;
__REG8  ESNS_0    : 1;
__REG8  ESNS_N    : 1;
__REG8  PRG_LENGTH: 3;
__REG8  DPC       : 1;
} __iim_fctl_bits;

/* Upper Address (UA) */
typedef struct{
__REG8  A         : 6;
__REG8            : 2;
} __iim_ua_bits;

/* Product Revision (PREV) */
typedef struct{
__REG8  PROD_VT   : 3;
__REG8  PROD_REV  : 5;
} __iim_prev_bits;

/* Software-Controllable Signals Register 0 (SCS0) */
typedef struct{
__REG8  SCS       : 6;
__REG8  HAB_JDE   : 1;
__REG8  LOCK      : 1;
} __iim_scs0_bits;

/* Software-Controllable Signals Registers 1–3 (SCS1–SCS3) */
typedef struct{
__REG8  SCS       : 7;
__REG8  LOCK      : 1;
} __iim_scs_bits;

/* Fuse Bank 0 Access Protection Register (FBAC0) */
typedef struct{
__REG8                  : 1;
__REG8  SJC_CHALL_LOCK  : 1;
__REG8                  : 1;
__REG8  FBESP           : 1;
__REG8  FBSP            : 1;
__REG8  FBRP            : 1;
__REG8  FBOP            : 1;
__REG8  FBWP            : 1;
} __iim_fbac0_bits;

/* JTAG Access Control (Hardware-Visible Word 0) JAC (HWAV0) */
typedef struct{
__REG8  JTAG_BP         : 1;
__REG8  SEC_JTAG_RE     : 1;
__REG8  KTE             : 1;
__REG8  JTAG_HEO        : 1;
__REG8  JTAG_SCC_DIS    : 1;
__REG8  JTAG_SMODE      : 2;
__REG8  WLOCK           : 1;
} __iim_hwv0_bits;

/* Hardware Visible Fuse Words 1 HWV1 */
typedef struct{
__REG8                    : 1;
__REG8  MEM_STICK_DISABLE : 1;
__REG8  HANTRO_DISABLE    : 1;
__REG8  SCC_DISABLE       : 1;
__REG8  BOOT_INT          : 1;
__REG8  DSP_ENDIAN        : 1;
__REG8  MCU_ENDIAN        : 1;
__REG8  WLOCK             : 1;
} __iim_hwv1_bits;

/* Hardware Visible Fuse Words 2 HWV2 */
typedef struct{
__REG8  BRG_TRIM        : 4;
__REG8                  : 1;
__REG8  SCM_DCM         : 1;
__REG8  BP_SDMA         : 1;
__REG8  WLOCK           : 1;
} __iim_hwv2_bits;

/* HAB0 Register (HAB0) */
typedef struct{
__REG8  HAB_SRS         : 5;
__REG8                  : 2;
__REG8  HAB_LOCK        : 1;
} __iim_hab0_bits;

/* HAB1 Register (HAB1) */
typedef struct{
__REG8  HAB_TYPE        : 3;
__REG8                  : 1;
__REG8  HAB_CUS         : 3;
__REG8                  : 1;
} __iim_hab1_bits;

/* Product Revision Defined in Fuse (PREV_FUSE) */
typedef struct{
__REG8  PROD_VT         : 3;
__REG8  PROD_REV        : 5;
} __iim_prev_fuse_bits;

/* Fuse Bank 1 Access Protection Register (FBAC1) */
typedef struct{
__REG8  SCC_LOCK        : 1;
__REG8  SJC_RESP_LOCK   : 1;
__REG8                  : 1;
__REG8  FBESP           : 1;
__REG8  FBSP            : 1;
__REG8  FBRP            : 1;
__REG8  FBOP            : 1;
__REG8  FBWP            : 1;
} __iim_fbac1_bits;

/* Fuse Bank 2-3 Access Protection Register (FBAC2-3) */
typedef struct{
__REG8                  : 3;
__REG8  FBESP           : 1;
__REG8  FBSP            : 1;
__REG8  FBRP            : 1;
__REG8  FBOP            : 1;
__REG8  FBWP            : 1;
} __iim_fbac_bits;

/* Register 0: L2CC Cache ID Register */
typedef struct{
__REG32 RTL_REL         : 6;
__REG32 PARTNUM         : 4;
__REG32 CACHEID         : 6;
__REG32                 : 8;
__REG32 RTL_IMPL        : 8;
} __l2cc_id_bits;

/* Register0: L2CC Cache Type Register */
typedef struct{
__REG32 L2CACHLEN0      : 2;
__REG32                 : 1;
__REG32 L2ASSOC0        : 5;
__REG32 L2CACHEWAY0     : 4;
__REG32 L2CACHLEN1      : 2;
__REG32                 : 1;
__REG32 L2              : 1;
__REG32 L2ASSOC1        : 4;
__REG32 L2CACHEWAY1     : 4;
__REG32 H               : 1;
__REG32 CTYPE           : 4;
__REG32                 : 3;
} __l2cc_type_bits;

/* Register 1: L2CC Control Register */
typedef struct{
__REG32 CACHEN          : 1;
__REG32                 :31;
} __l2cc_ctrl_bits;

/* Register 1: L2CC Auxiliary Control register */
typedef struct{
__REG32 LATENCY_READ            : 3;
__REG32 LATENCY_WRITE           : 3;
__REG32 LATENCY_TAG             : 3;
__REG32 LATENCY_DIRTY           : 3;
__REG32 WRAP_ACC_DIS            : 1;
__REG32 ASSOCIATIVITY           : 4;
__REG32 WAY_SIZE                : 3;
__REG32 EVENT_MONITOR           : 1;
__REG32 PARITY                  : 1;
__REG32 SHARED_ATTR_OVERRIDE    : 1;
__REG32 WRITE_ALLOCATE_OVERRIDE : 1;
__REG32 EXCLUSIVE_ABORT_DIS     : 1;
__REG32 WRAP2_DIS               : 1;
__REG32 IDLE_PIN_FORCE          : 1;
__REG32 IDLE8_DISABLE           : 1;
__REG32 CLKEN_CORE              : 1;
__REG32 INCRDISABLE             : 1;
__REG32 WRAP4_DIS               : 1;
__REG32 WRAP8_DIS               : 1;
} __l2cc_aux_ctrl_bits;

/* Test registers */
typedef struct{
__REG32 NRW             : 1;
__REG32                 : 4;
__REG32 INDEX           :24;
__REG32 WAY             : 3;
} __l2cc_test_bits;

/* L2 Line Tag Register */
typedef struct{
__REG32                 : 7;
__REG32 VICTIM_POINTER  : 3;
__REG32 DIRTY0          : 1;
__REG32 DIRTY1          : 1;
__REG32 VALID           : 1;
__REG32 TAG             :19;
} __l2cc_line_tag_bits;

/* L2CC Debug Control Register */
typedef struct{
__REG32 DCL             : 1;
__REG32 DWB             : 1;
__REG32                 :30;
} __l2cc_debug_ctrl_bits;

/* Monitor Control Register (EMMC) */
typedef struct{
__REG32 EVTMONEN        : 1;
__REG32 TYPE            : 1;
__REG32 POL             : 1;
__REG32 INTPULDUR       : 3;
__REG32                 : 2;
__REG32 EMC0RST         : 1;
__REG32 EMC1RST         : 1;
__REG32 EMC2RST         : 1;
__REG32 EMC3RST         : 1;
__REG32                 :20;
} __emmc_bits;

/* Counter Status Register (EMCS) */
typedef struct{
__REG32 EMC0            : 1;
__REG32 EMC1            : 1;
__REG32 EMC2            : 1;
__REG32 EMC3            : 1;
__REG32                 :28;
} __emcs_bits;

/* Counter Configuration Registers (EMCCx) */
typedef struct{
__REG32 INTEN           : 1;
__REG32 FLAG            : 1;
__REG32 SOURCE          : 5;
__REG32                 :25;
} __emcc_bits;

/* M3IF Control Register (M3IFCTL) */
typedef struct{
__REG32 MRRP            : 8;
__REG32 MLSD            : 3;
__REG32 MLSD_EN         : 1;
__REG32                 :19;
__REG32 SDA             : 1;
} __m3ifctl_bits;

/* M3IF Snooping Configuration Register 0 (M3IFSCFG0) */
typedef struct{
__REG32 SE              : 1;
__REG32 SWSZ            : 4;
__REG32                 : 6;
__REG32 SWBA            :21;
} __m3ifscfg0_bits;

/* M3IF Snooping Configuration Register 1 (M3IFSCFG1) */
typedef struct{
__REG32 SSE0_0          : 1;
__REG32 SSE0_1          : 1;
__REG32 SSE0_2          : 1;
__REG32 SSE0_3          : 1;
__REG32 SSE0_4          : 1;
__REG32 SSE0_5          : 1;
__REG32 SSE0_6          : 1;
__REG32 SSE0_7          : 1;
__REG32 SSE0_8          : 1;
__REG32 SSE0_9          : 1;
__REG32 SSE0_10         : 1;
__REG32 SSE0_11         : 1;
__REG32 SSE0_12         : 1;
__REG32 SSE0_13         : 1;
__REG32 SSE0_14         : 1;
__REG32 SSE0_15         : 1;
__REG32 SSE0_16         : 1;
__REG32 SSE0_17         : 1;
__REG32 SSE0_18         : 1;
__REG32 SSE0_19         : 1;
__REG32 SSE0_20         : 1;
__REG32 SSE0_21         : 1;
__REG32 SSE0_22         : 1;
__REG32 SSE0_23         : 1;
__REG32 SSE0_24         : 1;
__REG32 SSE0_25         : 1;
__REG32 SSE0_26         : 1;
__REG32 SSE0_27         : 1;
__REG32 SSE0_28         : 1;
__REG32 SSE0_29         : 1;
__REG32 SSE0_30         : 1;
__REG32 SSE0_31         : 1;
} __m3ifscfg1_bits;

/* M3IF Snooping Configuration Register 2 (M3IFSCFG2) */
typedef struct{
__REG32 SSE1_0          : 1;
__REG32 SSE1_1          : 1;
__REG32 SSE1_2          : 1;
__REG32 SSE1_3          : 1;
__REG32 SSE1_4          : 1;
__REG32 SSE1_5          : 1;
__REG32 SSE1_6          : 1;
__REG32 SSE1_7          : 1;
__REG32 SSE1_8          : 1;
__REG32 SSE1_9          : 1;
__REG32 SSE1_10         : 1;
__REG32 SSE1_11         : 1;
__REG32 SSE1_12         : 1;
__REG32 SSE1_13         : 1;
__REG32 SSE1_14         : 1;
__REG32 SSE1_15         : 1;
__REG32 SSE1_16         : 1;
__REG32 SSE1_17         : 1;
__REG32 SSE1_18         : 1;
__REG32 SSE1_19         : 1;
__REG32 SSE1_20         : 1;
__REG32 SSE1_21         : 1;
__REG32 SSE1_22         : 1;
__REG32 SSE1_23         : 1;
__REG32 SSE1_24         : 1;
__REG32 SSE1_25         : 1;
__REG32 SSE1_26         : 1;
__REG32 SSE1_27         : 1;
__REG32 SSE1_28         : 1;
__REG32 SSE1_29         : 1;
__REG32 SSE1_30         : 1;
__REG32 SSE1_31         : 1;
} __m3ifscfg2_bits;

/* M3IF Snooping Status Register 0 (M3IFSSR0) */
typedef struct{
__REG32 SSS0_0          : 1;
__REG32 SSS0_1          : 1;
__REG32 SSS0_2          : 1;
__REG32 SSS0_3          : 1;
__REG32 SSS0_4          : 1;
__REG32 SSS0_5          : 1;
__REG32 SSS0_6          : 1;
__REG32 SSS0_7          : 1;
__REG32 SSS0_8          : 1;
__REG32 SSS0_9          : 1;
__REG32 SSS0_10         : 1;
__REG32 SSS0_11         : 1;
__REG32 SSS0_12         : 1;
__REG32 SSS0_13         : 1;
__REG32 SSS0_14         : 1;
__REG32 SSS0_15         : 1;
__REG32 SSS0_16         : 1;
__REG32 SSS0_17         : 1;
__REG32 SSS0_18         : 1;
__REG32 SSS0_19         : 1;
__REG32 SSS0_20         : 1;
__REG32 SSS0_21         : 1;
__REG32 SSS0_22         : 1;
__REG32 SSS0_23         : 1;
__REG32 SSS0_24         : 1;
__REG32 SSS0_25         : 1;
__REG32 SSS0_26         : 1;
__REG32 SSS0_27         : 1;
__REG32 SSS0_28         : 1;
__REG32 SSS0_29         : 1;
__REG32 SSS0_30         : 1;
__REG32 SSS0_31         : 1;
} __m3ifssr0_bits;

/* M3IF Snooping Status Register 1 (M3IFSSR1) */
typedef struct{
__REG32 SSS1_0          : 1;
__REG32 SSS1_1          : 1;
__REG32 SSS1_2          : 1;
__REG32 SSS1_3          : 1;
__REG32 SSS1_4          : 1;
__REG32 SSS1_5          : 1;
__REG32 SSS1_6          : 1;
__REG32 SSS1_7          : 1;
__REG32 SSS1_8          : 1;
__REG32 SSS1_9          : 1;
__REG32 SSS1_10         : 1;
__REG32 SSS1_11         : 1;
__REG32 SSS1_12         : 1;
__REG32 SSS1_13         : 1;
__REG32 SSS1_14         : 1;
__REG32 SSS1_15         : 1;
__REG32 SSS1_16         : 1;
__REG32 SSS1_17         : 1;
__REG32 SSS1_18         : 1;
__REG32 SSS1_19         : 1;
__REG32 SSS1_20         : 1;
__REG32 SSS1_21         : 1;
__REG32 SSS1_22         : 1;
__REG32 SSS1_23         : 1;
__REG32 SSS1_24         : 1;
__REG32 SSS1_25         : 1;
__REG32 SSS1_26         : 1;
__REG32 SSS1_27         : 1;
__REG32 SSS1_28         : 1;
__REG32 SSS1_29         : 1;
__REG32 SSS1_30         : 1;
__REG32 SSS1_31         : 1;
} __m3ifssr1_bits;

/* M3IF Master Lock WEIM CSx Register (M3IFMLWEx) */
typedef struct{
__REG32 MLWE            : 3;
__REG32 MLWE_EN         : 1;
__REG32                 :27;
__REG32 WEMA            : 1;
} __m3ifmlwe_bits;

/* Chip Select x Upper Control Register (CSCRxU) */
typedef struct{
__REG32 EDC             : 4;
__REG32 WWS             : 3;
__REG32 EW              : 1;
__REG32 WSC             : 6;
__REG32 CNC             : 2;
__REG32 DOL             : 4;
__REG32 SYNC            : 1;
__REG32 PME             : 1;
__REG32 PSZ             : 2;
__REG32 BCS             : 4;
__REG32 BCD             : 2;
__REG32 WP              : 1;
__REG32 SP              : 1;
} __weim_cscru_bits;

/* Chip Select x Lower Control Register (CSCRxL) */
typedef struct{
__REG32 CSEN            : 1;
__REG32 WRAP            : 1;
__REG32 CRE             : 1;
__REG32 PSR             : 1;
__REG32 CSN             : 4;
__REG32 DSZ             : 3;
__REG32 EBC             : 1;
__REG32 CSA             : 4;
__REG32 EBWN            : 4;
__REG32 EBWA            : 4;
__REG32 OEN             : 4;
__REG32 OEA             : 4;
} __weim_cscrl_bits;

/* Chip Select x Additional Control Register (CSCRxA) */
typedef struct{
__REG32 FCE             : 1;
__REG32 CNC2            : 1;
__REG32 AGE             : 1;
__REG32 WWU             : 1;
__REG32 DCT             : 2;
__REG32 DWW             : 2;
__REG32 LBA             : 2;
__REG32 LBN             : 3;
__REG32 LAH             : 2;
__REG32 MUM             : 1;
__REG32 RWN             : 4;
__REG32 RWA             : 4;
__REG32 EBRN            : 4;
__REG32 EBRA            : 4;
} __weim_cscra_bits;

/* WEIM Configuration Register (WCR) */
typedef struct{
__REG32 MAS             : 1;
__REG32                 : 1;
__REG32 BCM             : 1;
__REG32                 :29;
} __weim_wcr_bits;

/* ESDCTL0-1 Control Registers */
typedef struct{
__REG32 PRCT            : 6;
__REG32                 : 1;
__REG32 BL              : 1;
__REG32 FP              : 1;
__REG32                 : 1;
__REG32 PWDT            : 2;
__REG32                 : 1;
__REG32 SREFR           : 3;
__REG32 DSIZ            : 2;
__REG32                 : 2;
__REG32 COL             : 2;
__REG32                 : 2;
__REG32 ROW             : 3;
__REG32 SP              : 1;
__REG32 SMODE           : 3;
__REG32 SDE             : 1;
} __esdctl_bits;

/* ESDCTL Configuration Registers (ESDCFG0/ESDCFG1) */
typedef struct{
__REG32 TRC             : 4;
__REG32 TRCD            : 3;
__REG32                 : 1;
__REG32 TCAS            : 2;
__REG32 TRRD            : 2;
__REG32 TRAS            : 3;
__REG32 TWR             : 1;
__REG32 TMRD            : 2;
__REG32 TRP             : 2;
__REG32 TWTR            : 1;
__REG32 TXP             : 2;
__REG32                 : 9;
} __esdcfg_bits;

/* ESDMISC Miscellaneous Register (ESDMISC) */
typedef struct{
__REG32                 : 1;
__REG32 RST             : 1;
__REG32 MDDREN          : 1;
__REG32 MDDR_DL_RST     : 1;
__REG32 MDDR_MDIS       : 1;
__REG32 LHD             : 1;
__REG32                 :25;
__REG32 SDRAMRDY        : 1;
} __esdmisc_bits;

/* MDDR Delay Line 1 Configuration Debug Register */
typedef struct{
__REG32 DLY_REG_1       :11;
__REG32                 : 5;
__REG32 DLY_CORR_1      :11;
__REG32                 : 4;
__REG32 SEL_DLY_REG_1   : 1;
} __esdcdly1_bits;

/* MDDR Delay Line 2 Configuration Debug Register */
typedef struct{
__REG32 DLY_REG_2       :11;
__REG32                 : 5;
__REG32 DLY_CORR_2      :11;
__REG32                 : 4;
__REG32 SEL_DLY_REG_2   : 1;
} __esdcdly2_bits;

/* MDDR Delay Line 3 Configuration Debug Register */
typedef struct{
__REG32 DLY_REG_3       :11;
__REG32                 : 5;
__REG32 DLY_CORR_3      :11;
__REG32                 : 4;
__REG32 SEL_DLY_REG_3   : 1;
} __esdcdly3_bits;

/* MDDR Delay Line 4 Configuration Debug Register */
typedef struct{
__REG32 DLY_REG_4       :11;
__REG32                 : 5;
__REG32 DLY_CORR_4      :11;
__REG32                 : 4;
__REG32 SEL_DLY_REG_4   : 1;
} __esdcdly4_bits;

/* MDDR Delay Line 5 Configuration Debug Register */
typedef struct{
__REG32 DLY_REG_5       :11;
__REG32                 : 5;
__REG32 DLY_CORR_5      :11;
__REG32                 : 4;
__REG32 SEL_DLY_REG_5   : 1;
} __esdcdly5_bits;

/* MDDR Delay Line Cycle Length Debug Register */
typedef struct{
__REG32 DLY_CYCLE_LENGTH  :11;
__REG32                   :21;
} __esdcdlyl_bits;

/* Internal SRAM SIZE (NFC_BUFSIZE) */
typedef struct{
__REG16 BUFSIZE  : 4;
__REG16          :12;
} __nfc_bufsize_bits;

/* Buffer Number for Page Data Transfer (RAM_BUFFER_ADDRESS) */
typedef struct{
__REG16 RBA  : 4;
__REG16      :12;
} __nfc_rba_bits;

/* NANDFC Internal Buffer Lock Control (NFC_CONFIGURATION) */
typedef struct{
__REG16 BLS  : 2;
__REG16      :14;
} __nfc_iblc_bits;

/* Controller Status and Result of Flash Operation (ECC_STATUS_RESULT) */
typedef struct{
__REG16 ERS  : 2;
__REG16 ERM  : 2;
__REG16      :12;
} __ecc_srr_bits;

/* ECC Error Position of Main Area Data Error (ECC_RSLT_MAIN_AREA) */
typedef union{
  /*ECC_RSLT_MA 8bit*/
  struct {
__REG16 ECC8_RESULT2   : 3;
__REG16 ECC8_RESULT1   : 9;
__REG16                : 4;
  };
  /*ECC_RSLT_MA 16bit*/
  struct {
__REG16 ECC16_RESULT2  : 4;
__REG16 ECC16_RESULT1  : 8;
__REG16                : 4;
  };
} __ecc_rslt_ma_bits;

/* ECC Error Position of Spare Area Data Error (ECC_RSLT_SPARE_AREA) */
typedef union{
  /*ECC_RSLT_SA 8bit*/
  struct {
__REG16 ECC8_RESULT3   : 3;
__REG16 ECC8_RESULT4   : 2;
__REG16                :11;
  };
  /*ECC_RSLT_SA 16bit*/
  struct {
__REG16 ECC16_RESULT3  : 4;
__REG16 ECC16_RESULT4  : 1;
__REG16                :11;
  };
} __ecc_rslt_sa_bits;

/* NAND Flash Write Protection (NF_WR_PROT) */
typedef struct{
__REG16 WPC  : 3;
__REG16      :13;
} __nf_wr_prot_bits;

/* NAND Flash Write Protection Status (NAND_FLASH_WR_PR_ST) */
typedef struct{
__REG16 LTS  : 1;
__REG16 LS   : 1;
__REG16 US   : 1;
__REG16      :13;
} __nf_wr_prot_sta_bits;

/* NAND Flash Operation Configuration (NAND_FLASH_CONFIG1) */
typedef struct{
__REG16           : 2;
__REG16 SP_EN     : 1;
__REG16 ECC_EN    : 1;
__REG16 INT_MASK  : 1;
__REG16 NF_BIG    : 1;
__REG16 NFC_RST   : 1;
__REG16 NNF_CE    : 1;
__REG16           : 8;
} __nand_fc1_bits;

/* NAND Flash Operation Configuration 2 (NAND_FLASH_CONFIG2) */
typedef struct{
__REG16 FCMD  : 1;
__REG16 FADD  : 1;
__REG16 FDI   : 1;
__REG16 FDO   : 3;
__REG16       : 9;
__REG16 INT   : 1;
} __nand_fc2_bits;

/* PCMCIA Input Pins Register */
typedef struct{
__REG32 VS       : 2;
__REG32 WP       : 1;
__REG32 CD       : 2;
__REG32 BVD1     : 1;
__REG32 BVD2     : 1;
__REG32 RDY      : 1;
__REG32 POWERON  : 1;
__REG32          :23;
} __pipr_bits;

/* PCMCIA Status Change Register */
typedef struct{
__REG32 VSC1   : 1;
__REG32 VSC2   : 1;
__REG32 WPC    : 1;
__REG32 CDC1   : 1;
__REG32 CDC2   : 1;
__REG32 BVDC1  : 1;
__REG32 BVDC2  : 1;
__REG32 RDYL   : 1;
__REG32 RDYH   : 1;
__REG32 RDYF   : 1;
__REG32 RDYR   : 1;
__REG32 POWC   : 1;
__REG32        :20;
} __pscr_bits;

/* PCMCIA Enable Register */
typedef struct{
__REG32 VSE1       : 1;
__REG32 VSE2       : 1;
__REG32 WPE        : 1;
__REG32 CDE1       : 1;
__REG32 CDE2       : 1;
__REG32 BVDE1      : 1;
__REG32 BVDE2      : 1;
__REG32 RDYLE      : 1;
__REG32 RDYHE      : 1;
__REG32 RDYFE      : 1;
__REG32 RDYRE      : 1;
__REG32 POWERONEN  : 1;
__REG32 ERRINTEN   : 1;
__REG32            :19;
} __per_bits;

/* PCMCIA Base Registers */
typedef struct{
__REG32 PBA  :26;
__REG32      : 6;
} __pbr_bits;

/* PCMCIA Option Registers */
typedef struct{
__REG32 BSIZE  : 5;
__REG32 PSHT   : 6;
__REG32 PSST   : 6;
__REG32 PSL    : 7;
__REG32 PPS    : 1;
__REG32 PRS    : 2;
__REG32 WP     : 1;
__REG32 WPEN   : 1;
__REG32 PV     : 1;
__REG32        : 2;
} __por_bits;

/* PCMCIA Offset Registers */
typedef struct{
__REG32 POFA  :26;
__REG32       : 6;
} __pofr_bits;

/* PCMCIA General Control Register */
typedef struct{
__REG32 RESET   : 1;
__REG32 POE     : 1;
__REG32 SPKREN  : 1;
__REG32 LPMEN   : 1;
__REG32         :28;
} __pgcr_bits;

/* PCMCIA General Control Register */
typedef struct{
__REG32 WPE    : 1;
__REG32 CDE    : 1;
__REG32 SE     : 1;
__REG32 LPE    : 1;
__REG32 NWINE  : 1;
__REG32        :27;
} __pgsr_bits;

/* O-Wire Control Register */
typedef struct{
__REG16       : 3;
__REG16 RDST  : 1;
__REG16 WR1   : 1;
__REG16 WR0   : 1;
__REG16 PST   : 1;
__REG16 RPP   : 1;
__REG16       : 8;
} __ow_control_bits;

/* O-Wire Time Divider Register */
typedef struct{
__REG16 DVDR  : 8;
__REG16       : 8;
} __ow_time_divider_bits;

/* O-Wire Reset Register */
typedef struct{
__REG16 RST  : 1;
__REG16      :15;
} __ow_reset_bits;

/* ATA_CONTROL Register */
typedef struct{
__REG8  IORDY_EN            : 1;
__REG8  DMA_WRITE           : 1;
__REG8  DMA_ULTRA_SELECTED  : 1;
__REG8  DMA_PENDING         : 1;
__REG8  FIFO_RCV_EN         : 1;
__REG8  FIFO_TX_EN          : 1;
__REG8  NATA_RST            : 1;
__REG8  NFIFO_RST           : 1;
} __ata_control_bits;

/* INTERRUPT_PENDING Register */
typedef struct{
__REG8                      : 3;
__REG8  ATA_IRTRQ2          : 1;
__REG8  CONTROLLER_IDLE     : 1;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8  ATA_INTRQ1          : 1;
} __ata_intr_pend_bits;

/* INTERRUPT_ENABLE Register */
typedef struct{
__REG8                      : 3;
__REG8  ATA_IRTRQ2          : 1;
__REG8  CONTROLLER_IDLE     : 1;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8  ATA_INTRQ1          : 1;
} __ata_intr_ena_bits;

/* INTERRUPT_CLEAR Register */
typedef struct{
__REG8                      : 5;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8                      : 1;
} __ata_intr_clr_bits;

/* CSPI Control Register (CONREG) */
typedef struct{
__REG32 EN          : 1;
__REG32 MODE        : 1;
__REG32 XCH         : 1;
__REG32 SMC         : 1;
__REG32 POL         : 1;
__REG32 PHA         : 1;
__REG32 SSCTL       : 1;
__REG32 SSPOL       : 1;
__REG32 BIT_COUNT   : 5;
__REG32             : 3;
__REG32 DATA_RATE   : 3;
__REG32             : 1;
__REG32 DRCTL       : 2;
__REG32             : 2;
__REG32 CS          : 2;
__REG32             : 6;
} __cspi_conreg_bits;

/* CSPI Interrupt Control Register (INTREG) */
typedef struct{
__REG32 TEEN     : 1;
__REG32 THEN     : 1;
__REG32 TFEN     : 1;
__REG32 RREN     : 1;
__REG32 RHEN     : 1;
__REG32 RFEN     : 1;
__REG32 ROEN     : 1;
__REG32 BOEN     : 1;
__REG32 TCEN     : 1;
__REG32          :23;
} __cspi_intreg_bits;

/* CSPI DMA Control Register (DMAREG) */
typedef struct{
__REG32 TEDEN  : 1;
__REG32 THDEN  : 1;
__REG32        : 2;
__REG32 RHDEN  : 1;
__REG32 RFDEN  : 1;
__REG32        :26;
} __cspi_dmareg_bits;

/* CSPI Status Register (STATREG) */
typedef struct{
__REG32 TE      : 1;
__REG32 TH      : 1;
__REG32 TF      : 1;
__REG32 RR      : 1;
__REG32 RH      : 1;
__REG32 RF      : 1;
__REG32 RO      : 1;
__REG32 BO      : 1;
__REG32 TC      : 1;
__REG32         :23;
} __cspi_statreg_bits;

/* CSPI Sample Period Control Register (PERIODREG) */
typedef struct{
__REG32 WAIT  :15;
__REG32 CSRC  : 1;
__REG32       :16;
} __cspi_periodreg_bits;

/* CSPI Test Control Register (TESTREG) */
typedef struct{
__REG32 TXCNT     : 4;
__REG32 RXCNT     : 4;
__REG32 SMSTATUS  : 4;
__REG32           : 2;
__REG32 LBC       : 1;
__REG32 SWAP      : 1;
__REG32           :16;
} __cspi_testreg_bits;

/* FIR Transmitter Control Register (FIRITCR) */
typedef struct{
__REG32 TE     : 1;
__REG32 TM     : 2;
__REG32 TPP    : 1;
__REG32 SIP    : 1;
__REG32 PC     : 1;
__REG32 PCF    : 1;
__REG32 TFUIE  : 1;
__REG32 TPEIE  : 1;
__REG32 TCIE   : 1;
__REG32 TDT    : 3;
__REG32 SRF    : 2;
__REG32        : 1;
__REG32 TPA    : 8;
__REG32 HAG    : 1;
__REG32        : 7;
} __firitcr_bits;

/* FIR Transmitter Count Register (FIRITCTR) */
typedef struct{
__REG32 TPL  :11;
__REG32      :21;
} __firitctr_bits;

/* FIR Receiver Control Register (FIRIRCR) */
typedef struct{
__REG32 RE     : 1;
__REG32 RM     : 2;
__REG32 RPP    : 1;
__REG32 RFOIE  : 1;
__REG32 PAIE   : 1;
__REG32 RPEIE  : 1;
__REG32 RPA    : 1;
__REG32 RDT    : 3;
__REG32 RPEDE  : 1;
__REG32        : 4;
__REG32 RA     : 8;
__REG32 RAM    : 1;
__REG32        : 7;
} __firircr_bits;

/* FIR Transmit Status Register (FIRITSR) */
typedef struct{
__REG32 TFU   : 1;
__REG32 TPE   : 1;
__REG32 SIPE  : 1;
__REG32 TC    : 1;
__REG32       : 4;
__REG32 TFP   : 8;
__REG32       :16;
} __firitsr_bits;

/* FIR Receive Status Register (FIRIRSR)*/
typedef struct{
__REG32 DDE   : 1;
__REG32 CRCE  : 1;
__REG32 BAM   : 1;
__REG32 RFO   : 1;
__REG32 RPE   : 1;
__REG32 PAS   : 1;
__REG32       : 2;
__REG32 RFP   : 8;
__REG32       :16;
} __firirsr_bits;

/* FIRI Control Register */
typedef struct{
__REG32 OSF  : 4;
__REG32      : 1;
__REG32 BL   : 7;
__REG32      :20;
} __firicr_bits;

/* I2C Address Register (IADR) */
typedef struct {
__REG16      : 1;
__REG16 ADR  : 7;
__REG16      : 8;
} __iadr_bits;

/* I2C Frequency Register (IFDR) */
typedef struct {
__REG16 IC  : 6;
__REG16     :10;
} __ifdr_bits;

/* I2C Control Register (I2CR) */
typedef struct {
__REG16       : 2;
__REG16 RSTA  : 1;
__REG16 TXAK  : 1;
__REG16 MTX   : 1;
__REG16 MSTA  : 1;
__REG16 IIEN  : 1;
__REG16 IEN   : 1;
__REG16       : 8;
} __i2cr_bits;

/* I2C Status Register (I2SR) */
typedef struct {
__REG16 RXAK  : 1;
__REG16 IIF   : 1;
__REG16 SRW   : 1;
__REG16       : 1;
__REG16 IAL   : 1;
__REG16 IBB   : 1;
__REG16 IAAS  : 1;
__REG16 ICF   : 1;
__REG16       : 8;
} __i2sr_bits;

/* I2C Data Register (I2DR) */
typedef struct {
__REG16 DATA  : 8;
__REG16       : 8;
} __i2dr_bits;

/* Keypad Control Register (KPCR) */
typedef struct{
__REG16 KRE  : 8;
__REG16 KCO  : 8;
} __kpcr_bits;

/* Keypad Status Register (KPSR) */
typedef struct{
__REG16 KPKD    : 1;
__REG16 KPKR    : 1;
__REG16 KDSC    : 1;
__REG16 KRSS    : 1;
__REG16         : 4;
__REG16 KDIE    : 1;
__REG16 KRIE    : 1;
__REG16         : 6;
} __kpsr_bits;

/* Keypad Data Direction Register (KDDR) */
typedef struct{
__REG16 KRDD  : 8;
__REG16 KCDD  : 8;
} __kddr_bits;

/* Keypad Data Register (KPDR) */
typedef struct{
__REG16 KRD  : 8;
__REG16 KCD  : 8;
} __kpdr_bits;

/* Gasket Interrupt Status/Clear Register */
typedef struct{
__REG8        : 2;
__REG8  REMPL : 1;
__REG8  WFUL  : 1;
__REG8        : 2;
__REG8  IXFR  : 1;
__REG8  IDA   : 1;
} __mshc_intr_stat_bits;

/* Gasket Interrupt Enable Register */
typedef struct{
__REG8              : 2;
__REG8  INTEN_REMP  : 1;
__REG8  INTEN_WFUL  : 1;
__REG8              : 2;
__REG8  INTEN_IXFR  : 1;
__REG8  INTEN_IDA   : 1;
} __mshc_intr_ena_bits;

/* SDHC Clock Control Register (STR_STP_CLK) */
typedef struct{
__REG32 STOP_CLK      : 1;
__REG32 START_CLK     : 1;
__REG32               : 1;
__REG32 SDHC_RESET    : 1;
__REG32               :28;
} __str_stp_clk_bits;

/* SDHC Status Register (STATUS) */
typedef struct{
__REG32 TIME_OUT_READ      : 1;
__REG32 TIME_OUT_RESP      : 1;
__REG32 WRITE_CRC_ERR      : 1;
__REG32 READ_CRC_ERR       : 1;
__REG32                    : 1;
__REG32 RESP_CRC_ERR       : 1;
__REG32 BUF_WRITE_READY    : 1;
__REG32 BUF_READ_READY     : 1;
__REG32 CARD_BUS_CLK_RUN   : 1;
__REG32 WR_CRC_ERROR_CODE  : 2;
__REG32 READ_OP_DONE       : 1;
__REG32 WRITE_OP_DONE      : 1;
__REG32 END_CMD_RESP       : 1;
__REG32 SDIO_INT_ACTIVE    : 1;
__REG32                    : 9;
__REG32 BUF_OVFL           : 1;
__REG32 BUF_UND_RUN        : 1;
__REG32 XBUF_FULL          : 1;
__REG32 YBUF_FULL          : 1;
__REG32 XBUF_EMPTY         : 1;
__REG32 YBUF_EMPTY         : 1;
__REG32 CARD_REMOVAL       : 1;
__REG32 CARD_INSERTION     : 1;
} __sd_status_bits;

/* SDHC Clock Rate Register (CLK_RATE) */
typedef struct{
__REG32 CLK_DIVIDER    : 4;
__REG32 CLK_PRESCALER  :12;
__REG32                :16;
} __clk_rate_bits;

/* SDHC Command and Data Control Register (CMD_DAT_CONT) */
typedef struct{
__REG32 FORMAT_OF_RESPONSE  : 3;
__REG32 DATA_ENABLE         : 1;
__REG32 WRITE_READ          : 1;
__REG32                     : 2;
__REG32 INIT                : 1;
__REG32 BUS_WIDTH           : 2;
__REG32 START_READ_WAIT     : 1;
__REG32 STOP_READ_WAIT      : 1;
__REG32 CMD_RESP_LONG_OFF   : 1;
__REG32                     : 2;
__REG32 CMD_RESUME          : 1;
__REG32                     :16;
} __cmd_dat_cont_bits;

/* SDHC Response Time Out Register (RES_TO) */
typedef struct{
__REG32 RESPONSE_TIME_OUT  : 8;
__REG32                    :24;
} __res_to_bits;

/* SDHC Read Time Out Register (READ_TO) */
typedef struct{
__REG32 DATA_READ_TIME_OUT  :16;
__REG32                     :16;
} __read_to_bits;

/* SDHC Block Length Register (BLK_LEN) */
typedef struct{
__REG32 BLOCK_LENGTH  :12;
__REG32               :20;
} __blk_len_bits;

/* SDHC Number of Blocks Register (NOB) */
typedef struct{
__REG32 NOB  :16;
__REG32      :16;
} __nob_bits;

/* SDHC Revision Number Register (REV_NO) */
typedef struct{
__REG32 REVISION_NUMBER  :16;
__REG32                  :16;
} __rev_no_bits;

/* SDHC Interrupt Control Register (INT_CNTR) */
typedef struct{
__REG32 READ_OP_DONE          : 1;
__REG32 WRITE_OP_DONE         : 1;
__REG32 END_CMD_RES           : 1;
__REG32 BUF_WRITE_EN          : 1;
__REG32 BUF_READ_EN           : 1;
__REG32                       : 7;
__REG32 DAT0_EN               : 1;
__REG32 SDIO_INT_EN           : 1;
__REG32 CARD_REMOVAL_EN       : 1;
__REG32 CARD_INSERTION_EN     : 1;
__REG32 CARD_REMOVAL_WKP_EN   : 1;
__REG32 CARD_INSERTION_WKP_EN : 1;
__REG32 SDIO_INT_WKP_EN       : 1;
__REG32                       :13;
} __mmcsd_int_ctrl_bits;

/* SDHC Command Number Register (CMD) */
typedef struct{
__REG32 COMMAND_NUMBER  : 6;
__REG32                 :26;
} __cmd_bits;

/* SDHC Response FIFO Access Register (RES_FIFO) */
typedef struct{
__REG32 RESPONSE_CONTENT  :16;
__REG32                   :16;
} __res_fifo_bits;

/* SIM Port1 Control Register (PORT1_CNTL) */
typedef struct{
__REG32 SAPD1             : 1;
__REG32 SVEN1             : 1;
__REG32 STEN1             : 1;
__REG32 SRST1             : 1;
__REG32 SCEN1             : 1;
__REG32 SCSP1             : 1;
__REG32 _3VOLT1           : 1;
__REG32 SFPD1             : 1;
__REG32                   :24;
} __sim_port1_cntl_bits;

/* SIM Setup Register (SETUP) */
typedef struct{
__REG32 AMODE             : 1;
__REG32 SPS               : 1;
__REG32                   :30;
} __sim_setup_bits;

/* SIM Port0,1 Detect Register (PORTn_DETECT) */
typedef struct{
__REG32 SDIM              : 1;
__REG32 SDI               : 1;
__REG32 SPDP              : 1;
__REG32 SPDS              : 1;
__REG32                   :28;
} __sim_port_detect_bits;

/* SIM Port0,1 Transmit Buffer Register (PORTn_XMT_BUF) */
typedef struct{
__REG32 PORT_XMT          : 8;
__REG32                   :24;
} __sim_port_xmt_buf_bits;

/* SIM Port0,1 Receive Buffer Register (PORTn_RCV_BUF) */
typedef struct{
__REG32 PORT_RCV          : 8;
__REG32 PE                : 1;
__REG32 FE                : 1;
__REG32 CWT               : 1;
__REG32                   :21;
} __sim_port_rcv_buf_bits;

/* SIM Port0 Control Register (PORT0_CNTL) */
typedef struct{
__REG32 SAPD0             : 1;
__REG32 SVEN0             : 1;
__REG32 STEN0             : 1;
__REG32 SRST0             : 1;
__REG32 SCEN0             : 1;
__REG32 SCSP0             : 1;
__REG32 _3VOLT0           : 1;
__REG32 SFPD0             : 1;
__REG32                   :24;
} __sim_port0_cntl_bits;

/* SIM Control Register (CNTL) */
typedef struct{
__REG32                   : 1;
__REG32 ICM               : 1;
__REG32 ANACK             : 1;
__REG32 ONACK             : 1;
__REG32 SAMPLE12          : 1;
__REG32                   : 1;
__REG32 BAUD_SEL          : 3;
__REG32 GPCNT_CLK_SEL     : 2;
__REG32 CWTEN             : 1;
__REG32 LRCEN             : 1;
__REG32 CRCEN             : 1;
__REG32 XMT_CRC_LRC       : 1;
__REG32 BWTEN             : 1;
__REG32                   :16;
} __sim_cntl_bits;

/* SIM Clock Select Register (CLOCK_SELECT) */
typedef struct{
__REG32 CLOCK_SELECT      : 4;
__REG32                   :28;
} __sim_clock_select_bits;

/* SIM Receive Threshold Register (RCV_THRESHOLD) */
typedef struct{
__REG32 RDT               : 5;
__REG32 RTH               : 4;
__REG32                   :23;
} __sim_rcv_threshold_bits;

/* SIM Enable Register (ENABLE) */
typedef struct{
__REG32 RCV_EN            : 1;
__REG32 XMT_EN            : 1;
__REG32                   :30;
} __sim_enable_bits;

/* SIM Transmit Status Register (XMT_STATUS) */
typedef struct{
__REG32 XTE               : 1;
__REG32                   : 2;
__REG32 TFE               : 1;
__REG32 ETC               : 1;
__REG32 TC                : 1;
__REG32 TFO               : 1;
__REG32 TDTF              : 1;
__REG32 GPCNT             : 1;
__REG32                   :23;
} __sim_xmt_status_bits;

/* SIM Receive Status Register (RCV_STATUS) */
typedef struct{
__REG32 OEF               : 1;
__REG32                   : 3;
__REG32 RFD               : 1;
__REG32 RDRF              : 1;
__REG32 LRCOK             : 1;
__REG32 CRCOK             : 1;
__REG32 CWT               : 1;
__REG32 RTE               : 1;
__REG32 BWT               : 1;
__REG32 BGT               : 1;
__REG32                   :20;
} __sim_rcv_status_bits;

/* SIM Interrupt Mask Register (INT_MASK) */
typedef struct{
__REG32 RIM               : 1;
__REG32 TCIM              : 1;
__REG32 OIM               : 1;
__REG32 ETCIM             : 1;
__REG32 TFEIM             : 1;
__REG32 XTM               : 1;
__REG32 TFOM              : 1;
__REG32 TDTFM             : 1;
__REG32 GPCNTM            : 1;
__REG32 CWTM              : 1;
__REG32 RTM               : 1;
__REG32 BWTM              : 1;
__REG32 BGTM              : 1;
__REG32                   :19;
} __sim_int_mask_bits;

/* SIM Data Format Register (DATA_FORMAT) */
typedef struct{
__REG32 IC                : 1;
__REG32                   :31;
} __sim_data_format_bits;

/* SIM Transmit Threshold Register (XMT_THRESHOLD) */
typedef struct{
__REG32 TDT               : 4;
__REG32 XTH               : 4;
__REG32                   :24;
} __sim_xmt_threshold_bits;

/* SIM Transmit Guard Control Register (GUARD_CNTL) */
typedef struct{
__REG32 GETU              : 8;
__REG32 RCVR11            : 1;
__REG32                   :23;
} __sim_guard_cntl_bits;

/* SIM Open Drain Configuration Control Register (OD_CONFIG) */
typedef struct{
__REG32 OD_P0             : 1;
__REG32 OD_P1             : 1;
__REG32                   :30;
} __sim_od_config_bits;

/* SIM Reset Control Register (RESET_CNTL) */
typedef struct{
__REG32 FLUSH_RCV         : 1;
__REG32 FLUSH_XMT         : 1;
__REG32 SOFT_REST         : 1;
__REG32 KILL_CLK          : 1;
__REG32 DOZE              : 1;
__REG32 STOP              : 1;
__REG32 DBUG              : 1;
__REG32                   :25;
} __sim_reset_cntl_bits;

/* SIM Divisor Register (DIVISOR) */
typedef struct{
__REG32 DIVISOR           : 7;
__REG32                   :25;
} __sim_divisor_bits;

/* UARTs Receiver Register */
typedef struct{
__REG32 RX_DATA  : 8;     /* Bits 0-7             - Recieve Data*/
__REG32          : 2;     /* Bits 8-9             - Reserved*/
__REG32 PRERR    : 1;     /* Bit  10              - Receive Parity Error 1=error*/
__REG32 BRK      : 1;     /* Bit  11              - Receive break Caracter detected 1 = detected*/
__REG32 FRMERR   : 1;     /* Bit  12              - Receive Framing Error 1=error*/
__REG32 OVRRUN   : 1;     /* Bit  13              - Receive Over run Error 1=error*/
__REG32 ERR      : 1;     /* Bit  14              - Receive Error Detect (OVR,FRM,BRK,PR 0=error*/
__REG32 CHARRDY  : 1;     /* Bit  15              - Receive Character Ready 1=character valid*/
__REG32          :16;     /* Bits 31-16           - Reserved*/
} __urxd_bits;

/* UARTs Transmitter Register */
typedef struct{
__REG32 TX_DATA  : 8;     /* Bits 7-0             - Transmit Data*/
__REG32          :24;     /* Bits 31-16           - Reserved*/
} __utxd_bits;

/* UARTs Control Register 1 */
typedef struct{
__REG32 UARTEN    : 1;     /* Bit  0       - UART Enable 1 = Enable the UART*/
__REG32 DOZE      : 1;     /* Bit  1       - DOZE 1 = The UART is disabled when in DOZE state*/
__REG32 ATDMAEN   : 1;     /* Bit  2*/
__REG32 TXDMAEN   : 1;     /* Bit  3       - Transmitter Ready DMA Enable 1 = enable*/
__REG32 SNDBRK    : 1;     /* Bit  4       - Send BREAK 1 = send break char continuous*/
__REG32 RTSDEN    : 1;     /* Bit  5       - RTS Delta Interrupt Enable 1 = enable*/
__REG32 TXMPTYEN  : 1;     /* Bit  6       - Transmitter Empty Interrupt Enable 1 = enable*/
__REG32 IREN      : 1;     /* Bit  7       - Infrared Interface Enable 1 = enable*/
__REG32 RXDMAEN   : 1;     /* Bit  8       - Receive Ready DMA Enable 1 = enable*/
__REG32 RRDYEN    : 1;     /* Bit  9       - Receiver Ready Interrupt Enable 1 = Enable*/
__REG32 ICD       : 2;     /* Bit  10-11   - Idle Condition Detect*/
                                /*              - 00 = Idle for more than 4 frames*/
                                /*              - 01 = Idle for more than 8 frames*/
                                /*              - 10 = Idle for more than 16 frames*/
                                /*              - 11 = Idle for more than 32 frames*/
__REG32 IDEN      : 1;     /* Bit  12      - Idle Condition Detected Interrupt en 1=en*/
__REG32 TRDYEN    : 1;     /* Bit  13      - Transmitter Ready Interrupt Enable 1=en*/
__REG32 ADBR      : 1;     /* Bit  14      - AutoBaud Rate Detection enable 1=en*/
__REG32 ADEN      : 1;     /* Bit  15      - AutoBaud Rate Detection Interrupt en 1=en*/
__REG32           :16;
} __ucr1_bits;

/* UARTs Control Register 2 */
typedef struct{
__REG32 NSRST  : 1;     /* Bit  0       -Software Reset 0 = Reset the tx and rx state machines*/
__REG32 RXEN   : 1;     /* Bit  1       -Receiver Enable 1 = Enable*/
__REG32 TXEN   : 1;     /* Bit  2       -Transmitter Enable 1= enable*/
__REG32 ATEN   : 1;     /* Bit  3       -Aging Timer Enable—This bit is used to mask the aging timer interrupt (triggered with AGTIM)*/
__REG32 RTSEN  : 1;     /* Bit  4       -Request to Send Interrupt Enable 1=enable*/
__REG32 WS     : 1;     /* Bit  5       -Word Size 0 = 7bit, 1= 8 bit*/
__REG32 STPB   : 1;     /* Bit  6       -Stop 0= 1 stop bits, 1= 2 stop bits*/
__REG32 PROE   : 1;     /* Bit  7       -Parity Odd/Even 1=Odd*/
__REG32 PREN   : 1;     /* Bit  8       -Parity Enable 1=enable parity generator*/
__REG32 RTEC   : 2;     /* Bits 9-10    -Request to Send Edge Control*/
                                /*              - 00 = Trigger interrupt on a rising edge*/
                                /*              - 01 = Trigger interrupt on a falling edge*/
                                /*              - 1X = Trigger interrupt on any edge*/
__REG32 ESCEN  : 1;     /* Bit  11      -Escape Enable 1 = Enable escape sequence detection*/
__REG32 CTS    : 1;     /* Bit  12      -Clear to Send 1 = The UARTx_CTS pin is low (active)*/
__REG32 CTSC   : 1;     /* Bit  13      -UARTx_CTS Pin controlled by 1= receiver 0= CTS bit*/
__REG32 IRTS   : 1;     /* Bit  14      -Ignore UARTx_RTS Pin 1=ignore*/
__REG32 ESCI   : 1;     /* Bit  15      -Escape Sequence Interrupt En 1=enable*/
__REG32        :16;
} __ucr2_bits;

/* UARTs Control Register 3 */
typedef struct{
__REG32 ACIEN      : 1;
__REG32 INVT       : 1;
__REG32 RXDMUXSEL  : 1;
__REG32 DTRDEN     : 1;
__REG32 AWAKEN     : 1;
__REG32 AIRINTEN   : 1;
__REG32 RXDSEN     : 1;
__REG32 ADNIMP     : 1;
__REG32 RI         : 1;
__REG32 DCD        : 1;
__REG32 DSR        : 1;
__REG32 FRAERREN   : 1;
__REG32 PARERREN   : 1;
__REG32 DTREN      : 1;
__REG32 DPEC       : 2;
__REG32            :16;
} __ucr3_bits;

/* UARTs Control Register 4 */
typedef struct{
__REG32 DREN   : 1;     /* Bit  0       -Receive Data Ready Interrupt Enable 1= enable*/
__REG32 OREN   : 1;     /* Bit  1       -Receiver Overrun Interrupt Enable 1= enable*/
__REG32 BKEN   : 1;     /* Bit  2       -BREAK Condition Detected Interrupt en 1= enable*/
__REG32 TCEN   : 1;     /* Bit  3       -Transmit Complete Interrupt Enable1 = Enable*/
__REG32 LPBYP  : 1;     /* Bit  4       -Low Power Bypass—Allows to bypass the low power new features in UART for i.MX31. To use during debug phase.*/
__REG32 IRSC   : 1;     /* Bit  5       -IR Special Case vote logic uses 1= uart ref clk*/
__REG32 IDDMAEN: 1;     /* Bit  6       -*/
__REG32 WKEN   : 1;     /* Bit  7       -WAKE Interrupt Enable 1= enable*/
__REG32 ENIRI  : 1;     /* Bit  8       -Serial Infrared Interrupt Enable 1= enable*/
__REG32 INVR   : 1;     /* Bit  9       -Inverted Infrared Reception 1= active high*/
__REG32 CTSTL  : 6;     /* Bits 10-15   -CTS Trigger Level*/
                                /*              000000 = 0 characters received*/
                                /*              000001 = 1 characters in the RxFIFO*/
                                /*              ...*/
                                /*              100000 = 32 characters in the RxFIFO (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32        :16;
} __ucr4_bits;

/* UARTs FIFO Control Register */
typedef struct{
__REG32 RXTL    : 6;     /* Bits 0-5     -Receiver Trigger Level*/
                                /*              000000 = 0 characters received*/
                                /*              000001 = RxFIFO has 1 character*/
                                /*              ...*/
                                /*              011111 = RxFIFO has 31 characters*/
                                /*              100000 = RxFIFO has 32 characters (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32 DCEDTE  : 1;     /* Bit  6       -*/
__REG32 RFDIV   : 3;     /* Bits 7-9     -Reference Frequency Divider*/
                                /*              000 = Divide input clock by 6*/
                                /*              001 = Divide input clock by 5*/
                                /*              010 = Divide input clock by 4*/
                                /*              011 = Divide input clock by 3*/
                                /*              100 = Divide input clock by 2*/
                                /*              101 = Divide input clock by 1*/
                                /*              110 = Divide input clock by 7*/
__REG32 TXTL    : 6;     /* Bits 10-15   -Transmitter Trigger Level*/
                                /*              000000 = Reserved*/
                                /*              000001 = Reserved*/
                                /*              000010 = TxFIFO has 2 or fewer characters*/
                                /*              ...*/
                                /*              011111 = TxFIFO has 31 or fewer characters*/
                                /*              100000 = TxFIFO has 32 characters (maximum)*/
                                /*              All Other Settings Reserved*/
__REG32         :16;
} __ufcr_bits;

/* UARTs Status Register 1 */
typedef struct{
__REG32            : 4;
__REG32 AWAKE      : 1;
__REG32 AIRINT     : 1;
__REG32 RXDS       : 1;
__REG32 DTRD       : 1;
__REG32 AGTIM      : 1;
__REG32 RRDY       : 1;
__REG32 FRAMER     : 1;
__REG32 ESCF       : 1;
__REG32 RTSD       : 1;
__REG32 TRDY       : 1;
__REG32 RTSS       : 1;
__REG32 PARITYERR  : 1;
__REG32            :16;
} __usr1_bits;

/* UARTs Status Register 2 */
typedef struct{
__REG32 RDR      : 1;
__REG32 ORE      : 1;
__REG32 BRCD     : 1;
__REG32 TXDC     : 1;
__REG32 RTSF     : 1;
__REG32 DCDIN    : 1;
__REG32 DCDDELT  : 1;
__REG32 WAKE     : 1;
__REG32 IRINT    : 1;
__REG32 RIIN     : 1;
__REG32 RIDELT   : 1;
__REG32 ACST     : 1;
__REG32 IDLE     : 1;
__REG32 DTRF     : 1;
__REG32 TXFE     : 1;
__REG32 ADET     : 1;
__REG32          :16;
} __usr2_bits;

/* UARTs Escape Character Register */
typedef struct{
__REG32 ESC_CHAR  : 8;     /* Bits 0-7     -UART Escape Character*/
__REG32           :24;
} __uesc_bits;

/* UARTs Escape Timer Register */
typedef struct{
__REG32 TIM  :12;     /* Bits 0-11    -UART Escape Timer*/
__REG32      :20;
} __utim_bits;

/* UARTs BRM Incremental Register */
typedef struct{
__REG32 INC  :16;     /* Bits 0-15    -Incremental Numerator*/
__REG32      :16;
} __ubir_bits;

/* UARTs BRM Modulator Register */
typedef struct{
__REG32 MOD  :16;     /* Bits 0-15    -Modulator Denominator*/
__REG32      :16;
} __ubmr_bits;

/* UARTs Baud Rate Count register */
typedef struct{
__REG32 BCNT  :16;     /* Bits 0-15    -Baud Rate Count Register*/
__REG32       :16;
} __ubrc_bits;

/* UARTs One Millisecond Registers */
typedef struct{
__REG32 BCNT  :16;     /* Bits 0-15    -Baud Rate Count Register*/
__REG32       :16;
} __onems_bits;

/* UARTS Test Register 1 */
typedef struct{
__REG32 SOFTRST  : 1;
__REG32          : 2;
__REG32 RXFULL   : 1;
__REG32 TXFULL   : 1;
__REG32 RXEMPTY  : 1;
__REG32 TXEMPTY  : 1;
__REG32          : 2;
__REG32 RXDBG    : 1;
__REG32 LOOPIR   : 1;
__REG32 DBGEN    : 1;
__REG32 LOOP     : 1;
__REG32 FRCPERR  : 1;
__REG32          :18;
} __uts_bits;

/* USBCONTROL—USB Control Register */
typedef struct{
__REG32 BPE       : 1;
__REG32           : 3;
__REG32 H1DT      : 1;
__REG32 H2DT      : 1;
__REG32           : 2;
__REG32 H1PM      : 1;
__REG32 RXDP      : 2;
__REG32 H1WIE     : 1;
__REG32           : 1;
__REG32 H1SIC     : 2;
__REG32 H1WIR     : 1;
__REG32 H2PM      : 1;
__REG32           : 2;
__REG32 H2WIE     : 1;
__REG32 H2UIE     : 1;
__REG32 H2SIC     : 2;
__REG32 H2WIR     : 1;
__REG32 OPM       : 1;
__REG32 RXDB      : 2;
__REG32 OWIE      : 1;
__REG32 OUIE      : 1;
__REG32 OSIC      : 2;
__REG32 OWIR      : 1;
} __usb_ctrl_bits;

/* OTGMIRROR—OTG Port Mirror Register */
typedef struct{
__REG8  IDDIG     : 1;
__REG8  ASESVLD   : 1;
__REG8  BSESVLD   : 1;
__REG8  VBUSVLD   : 1;
__REG8  SESEND    : 1;
__REG8            : 3;
} __usb_otg_mirror_bits;

/* ID Register */
typedef struct{
__REG32 ID        : 6;
__REG32           : 2;
__REG32 NID       : 6;
__REG32           : 2;
__REG32 REVISION  : 8;
__REG32           : 8;
} __uog_id_bits;

/* HWGENERAL—General Hardware Parameters */
typedef struct{
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 1;
__REG32           :22;
} __uog_hwgeneral_bits;

/* HWHOST—Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __uog_hwhost_bits;

/* HWDEVICE—Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __uog_hwdevice_bits;

/* HWTXBUF—TX Buffer Hardware Parameters */
typedef struct{
__REG32 TXBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __uog_hwtxbuf_bits;

/* HWRXBUF—RX Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __uog_hwrxbuf_bits;

/* HCSPARAMS—Host Control Structural Parameters */
typedef struct{
__REG32 N_PORTS   : 4;
__REG32 PPC       : 1;
__REG32           : 3;
__REG32 N_PCC     : 4;
__REG32 N_CC      : 4;
__REG32 PI        : 1;
__REG32           : 3;
__REG32 N_PTT     : 4;
__REG32 N_TT      : 4;
__REG32           : 4;
} __uog_hcsparams_bits;

/* HCCPARAMS—Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __uog_hccparams_bits;

/* HCCPARAMS—Host Control Capability Parameters */
typedef struct{
__REG32 DEN       : 5;
__REG32           : 2;
__REG32 D         : 1;
__REG32 H         : 1;
__REG32           :23;
} __uog_dccparams_bits;

/* USB Command Register (USBCMD) */
typedef struct{
__REG32 RS        : 1;
__REG32 RST       : 1;
__REG32 FS0       : 1;
__REG32 FS1       : 1;
__REG32 PSE       : 1;
__REG32 ASE       : 1;
__REG32 IAA       : 1;
__REG32 LR        : 1;
__REG32 ASP0      : 1;
__REG32 ASP1      : 1;
__REG32           : 1;
__REG32 ASPE      : 1;
__REG32 ATDTW     : 1;
__REG32 SUTW      : 1;
__REG32           : 1;
__REG32 FS2       : 1;
__REG32 ITC       : 8;
__REG32           : 8;
} __uog_usbcmd_bits;

/* USBSTS—USB Status */
typedef struct{
__REG32 UI        : 1;
__REG32 UEI       : 1;
__REG32 PCI       : 1;
__REG32 FRI       : 1;
__REG32 SEI       : 1;
__REG32 AAI       : 1;
__REG32 URI       : 1;
__REG32 SRI       : 1;
__REG32 SLI       : 1;
__REG32           : 1;
__REG32 ULPII     : 1;
__REG32           : 1;
__REG32 HCH       : 1;
__REG32 RCL       : 1;
__REG32 PS        : 1;
__REG32 AS        : 1;
__REG32           :16;
} __uog_usbsts_bits;

/* USBINTR—USB Interrupt Enable */
typedef struct{
__REG32 UE        : 1;
__REG32 UEE       : 1;
__REG32 PCE       : 1;
__REG32 FRE       : 1;
__REG32 SEE       : 1;
__REG32 AAE       : 1;
__REG32 URE       : 1;
__REG32 SRE       : 1;
__REG32 SLE       : 1;
__REG32           : 1;
__REG32 ULPIE     : 1;
__REG32           :21;
} __uog_usbintr_bits;

/* FRINDEX—USB Frame Index */
typedef struct{
__REG32 FRINDEX   :14;
__REG32           :18;
} __uog_frindex_bits;

/* PERIODICLISTBASE—Host Controller Frame List Base Address */
/* DEVICEADDR—Device Controller USB Device Address */
typedef union {
/* PERIODICLISTBASE*/
  struct{
    __REG32           :12;
    __REG32 PERBASE   :20;
  };
/* DEVICEADDR*/
  struct{
    __REG32           :25;
    __REG32 USBADR    : 7;
  };
} __uog_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
/* ASYNCLISTADDR*/
  struct{
    __REG32           : 6;
    __REG32 ASYBASE   :26;
  };
/* ENDPOINTLISTADDR*/
  struct{
    __REG32           :11;
    __REG32 EPBASE    :21;
  };
} __uog_asynclistaddr_bits;


/* BURSTSIZE—Host Controller Embedded TT Async. Buffer Status */
typedef struct{
__REG32 RXPBURST  : 8;
__REG32 TXPBURST  : 9;
__REG32           :15;
} __uog_burstsize_bits;

/* TXFILLTUNING Register */
typedef struct{
__REG32 TXSCHOH   : 8;
__REG32 TXSCHEAL  : 5;
__REG32           : 3;
__REG32 TXFIFOTHR : 6;
__REG32           :10;
} __uog_txfilltuning_bits;

/* TXFILLTUNING Register */
typedef struct{
__REG32 ULPIDATWR : 8;
__REG32 ULPIDATRD : 8;
__REG32 ULPIADDR  : 8;
__REG32 ULPIPORT  : 3;
__REG32 ULPIS     : 1;
__REG32           : 1;
__REG32 ULPIRW    : 1;
__REG32 ULPIRUN   : 1;
__REG32 ULPIWU    : 1;
} __uog_ulpiview_bits;

/* PORTSCx—Port Status Control[1:8] */
typedef struct{
__REG32 CCS       : 1;
__REG32 CSC       : 1;
__REG32 PE        : 1;
__REG32 PEC       : 1;
__REG32 OCA       : 1;
__REG32 OCC       : 1;
__REG32 FPR       : 1;
__REG32 SUSP      : 1;
__REG32 PR        : 1;
__REG32 HSP       : 1;
__REG32 LS        : 2;
__REG32 PP        : 1;
__REG32 PO        : 1;
__REG32 PIC       : 2;
__REG32 PTC       : 3;
__REG32 WKCN      : 1;
__REG32 WKDS      : 1;
__REG32 WKOC      : 1;
__REG32 PHCD      : 1;
__REG32 PFSC      : 1;
__REG32           : 1;
__REG32 PSPD      : 3;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __uog_portsc_bits;

/* OTGSC—OTG Status Control */
typedef struct{
__REG32 VD        : 1;
__REG32 VC        : 1;
__REG32           : 1;
__REG32 OT        : 1;
__REG32 DP        : 1;
__REG32 IDPU      : 1;
__REG32           : 2;
__REG32 ID        : 1;
__REG32 AVV       : 1;
__REG32 ASV       : 1;
__REG32 BSV       : 1;
__REG32 BSE       : 1;
__REG32 _1MST     : 1;
__REG32 DPS       : 1;
__REG32           : 1;
__REG32 IDIS      : 1;
__REG32 AVVIS     : 1;
__REG32 ASVIS     : 1;
__REG32 BSVIS     : 1;
__REG32 BSEIS     : 1;
__REG32 _1MSS     : 1;
__REG32 DPIS      : 1;
__REG32           : 1;
__REG32 IDIE      : 1;
__REG32 AVVIE     : 1;
__REG32 ASVIE     : 1;
__REG32 BSVIE     : 1;
__REG32 BSEIE     : 1;
__REG32 _1MSE     : 1;
__REG32 DPIE      : 1;
__REG32           : 1;
} __uog_otgsc_bits;

/* USBMODE—USB Device Mode */
typedef struct{
__REG32 CM        : 2;
__REG32 E         : 1;
__REG32 SL        : 1;
__REG32 S         : 1;
__REG32           :27;
} __uog_usbmode_bits;

/* ENDPTSETUPSTAT—Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT  :16;
__REG32                 :16;
} __uog_endptsetupstat_bits;

/* ENDPTPRIME—Endpoint Initialization */
typedef struct{
__REG32 PERB            :16;
__REG32 PETB            :16;
} __uog_endptprime_bits;

/* ENDPTFLUSH—Endpoint De-Initialize */
typedef struct{
__REG32 FERB            :16;
__REG32 FETB            :16;
} __uog_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR            :16;
__REG32 ETBR            :16;
} __uog_endptstat_bits;

/* ENDPTCOMPLETE—Endpoint Compete */
typedef struct{
__REG32 ERCE            :16;
__REG32 ETCE            :16;
} __uog_endptcomplete_bits;

/* ENDPTCOMPLETE—Endpoint Compete */
typedef struct{
__REG32 RXS             : 1;
__REG32 RXD             : 1;
__REG32 RXT             : 2;
__REG32                 : 1;
__REG32 RXI             : 1;
__REG32 RXR             : 1;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32 TXD             : 1;
__REG32 TXT             : 2;
__REG32                 : 1;
__REG32 TXI             : 1;
__REG32 TXR             : 1;
__REG32 TXE             : 1;
__REG32                 : 8;
} __uog_endptctrl_bits;

/* EPIT Control Register (EPITCR) */
typedef struct{
__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 OCIEN           : 1;
__REG32 RLD             : 1;
__REG32 PRESCALER       :12;
__REG32 SWR             : 1;
__REG32 IOVW            : 1;
__REG32 DBGEN           : 1;
__REG32 EPIT            : 1;
__REG32 DOZEN           : 1;
__REG32 STOPEN          : 1;
__REG32 OM              : 2;
__REG32 CLKSRC          : 2;
__REG32                 : 6;
} __epitcr_bits;

/* EPIT Status Register (EPITSR) */
typedef struct{
__REG32 OCIF            : 1;
__REG32                 :31;
} __epitsr_bits;

/* GPT Control Register (GPTCR) */
typedef struct{
__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 DBGEN           : 1;
__REG32 WAITEN          : 1;
__REG32 DOZEN           : 1;
__REG32 STOPEN          : 1;
__REG32 CLKSRC          : 3;
__REG32 FRR             : 1;
__REG32                 : 5;
__REG32 SWR             : 1;
__REG32 IM1             : 2;
__REG32 IM2             : 2;
__REG32 OM1             : 3;
__REG32 OM2             : 3;
__REG32 OM3             : 3;
__REG32 FO1             : 1;
__REG32 FO2             : 1;
__REG32 FO3             : 1;
} __gptcr_bits;

/* GPT Prescaler Register (GPTPR) */
typedef struct{
__REG32 PRESCALER       :12;
__REG32                 :20;
} __gptpr_bits;

/* GPT Status Register (GPTSR) */
typedef struct{
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32 OF3             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32 ROV             : 1;
__REG32                 :26;
} __gptsr_bits;

/* GPT Interrupt Register (GPTIR) */
typedef struct{
__REG32 OF1IE           : 1;
__REG32 OF2IE           : 1;
__REG32 OF3IE           : 1;
__REG32 IF1IE           : 1;
__REG32 IF2IE           : 1;
__REG32 ROVIE           : 1;
__REG32                 :26;
} __gptir_bits;

/* PWM Control Register (PWMCR) */
typedef struct{
__REG32 EN         : 1;
__REG32 REPEAT     : 2;
__REG32 SWR        : 1;
__REG32 PRESCALER  :12;
__REG32 CLKSRC     : 2;
__REG32 POUTC      : 2;
__REG32 HCTR       : 1;
__REG32 BCTR       : 1;
__REG32 DBGEN      : 1;
__REG32 WAITEN     : 1;
__REG32 DOZEN      : 1;
__REG32 STOPEN     : 1;
__REG32 FWM        : 2;
__REG32            : 4;
} __pwmcr_bits;

/* PWM Status Register (PWMSR) */
typedef struct{
__REG32 FIFOAV  : 3;
__REG32 FE      : 1;
__REG32 ROV     : 1;
__REG32 CMP     : 1;
__REG32 FWE     : 1;
__REG32         :25;
} __pwmsr_bits;

/* PWM Interrupt Register (PWMIR) */
typedef struct{
__REG32 FIE     : 1;
__REG32 RIE     : 1;
__REG32 CIE     : 1;
__REG32         :29;
} __pwmir_bits;

/* PWM Sample Register (PWMSAR) */
typedef struct{
__REG32 SAMPLE :16;
__REG32        :16;
} __pwmsar_bits;

/* PWM Period Register (PWMPR) */
typedef struct{
__REG32 PERIOD :16;
__REG32        :16;
} __pwmpr_bits;

/* PWM Counter Register (PWMCNR) */
typedef struct{
__REG32 COUNT  :16;
__REG32        :16;
} __pwmcnr_bits;

/* RTC Hours and Minutes Counter Register (HOURMIN) */
/* RTC Hours and Minutes Alarm Register (ALRM_HM) */
typedef struct{
__REG32 MINUTES  : 6;
__REG32          : 2;
__REG32 HOURS    : 5;
__REG32          :19;
} __hourmin_bits;

/* RTC Seconds Counter Register (SECONDS) */
/* RTC Seconds Alarm Register (ALRM_SEC) */
typedef struct{
__REG32 SECONDS  : 6;
__REG32          :26;
} __seconds_bits;

/* RTC Control Register (RTCCTL) */
typedef struct{
__REG32 SWR  : 1;
__REG32 GEN  : 1;
__REG32      : 3;
__REG32 XTL  : 2;
__REG32 EN   : 1;
__REG32      :24;
} __rcctl_bits;

/* RTC Interrupt Status Register (RTCISR) */
/* RTC Interrupt Enable Register (RTCIENR) */
typedef struct{
__REG32 SW    : 1;
__REG32 MIN   : 1;
__REG32 ALM   : 1;
__REG32 DAY   : 1;
__REG32 _1HZ  : 1;
__REG32 HR    : 1;
__REG32       : 1;
__REG32 _2HZ  : 1;
__REG32 SAM0  : 1;
__REG32 SAM1  : 1;
__REG32 SAM2  : 1;
__REG32 SAM3  : 1;
__REG32 SAM4  : 1;
__REG32 SAM5  : 1;
__REG32 SAM6  : 1;
__REG32 SAM7  : 1;
__REG32       :16;
} __rtcisr_bits;

/* RTC Stopwatch Minutes Register (STPWCH) */
typedef struct{
__REG32 CNT  : 6;
__REG32      :26;
} __stpwch_bits;

/* RTC Days Counter Register (DAYR) */
typedef struct{
__REG32 DAYS  :16;
__REG32       :16;
} __dayr_bits;

/* RTC Day Alarm Register (DAYALARM) */
typedef struct{
__REG32 DAYSAL  :16;
__REG32         :16;
} __dayalarm_bits;

/* Watchdog Control Register (WCR) */
typedef struct {
__REG16 WDZST  : 1;     /* Bit  0       - Watchdog Low Power*/
__REG16 WDBG   : 1;     /* Bits 1       - Watchdog DEBUG Enable*/
__REG16 WDE    : 1;     /* Bit  2       - Watchdog Enable*/
__REG16 WRE    : 1;     /* Bit  3       - ~WDOG or ~WDOG_RESET Enable*/
__REG16 SRS    : 1;     /* Bit  4       - Software Reset Signal*/
__REG16 WDA    : 1;     /* Bit  5       - Watchdog Assertion*/
__REG16 WOE    : 1;     /* Bit  6*/
__REG16        : 1;     /* Bit  7       - Reserved*/
__REG16 WT     : 8;     /* Bits 8 - 15  - Watchdog Time-Out Field*/
} __wcr_bits;

/* Watchdog Reset Status Register (WRSR) */
typedef struct {
__REG16 SFTW  : 1;     /* Bit  0       - Software Reset*/
__REG16 TOUT  : 1;     /* Bit  1       - Time-out*/
__REG16 CMON  : 1;     /* Bit  2*/
__REG16 EXT   : 1;     /* Bit  3       - External Reset*/
__REG16 PWR   : 1;     /* Bit  4       - Power-On Reset*/
__REG16 JRST  : 1;     /* Bit  5*/
__REG16       :10;     /* Bits 6  - 15 - Reserved*/
} __wrsr_bits;

/* Master Privilege Registers (MPR_1 and MPR_2) */
typedef struct{
__REG32 MPL7 : 1;
__REG32 MTW7 : 1;
__REG32 MTR7 : 1;
__REG32 MBW7 : 1;
__REG32 MPL6 : 1;
__REG32 MTW6 : 1;
__REG32 MTR6 : 1;
__REG32 MBW6 : 1;
__REG32 MPL5 : 1;
__REG32 MTW5 : 1;
__REG32 MTR5 : 1;
__REG32 MBW5 : 1;
__REG32 MPL4 : 1;
__REG32 MTW4 : 1;
__REG32 MTR4 : 1;
__REG32 MBW4 : 1;
__REG32 MPL3 : 1;
__REG32 MTW3 : 1;
__REG32 MTR3 : 1;
__REG32 MBW3 : 1;
__REG32 MPL2 : 1;
__REG32 MTW2 : 1;
__REG32 MTR2 : 1;
__REG32 MBW2 : 1;
__REG32 MPL1 : 1;
__REG32 MTW1 : 1;
__REG32 MTR1 : 1;
__REG32 MBW1 : 1;
__REG32 MPL0 : 1;
__REG32 MTW0 : 1;
__REG32 MTR0 : 1;
__REG32 MBW0 : 1;
} __aips_mpr_1_bits;

typedef struct{
__REG32 MPL15: 1;
__REG32 MTW15: 1;
__REG32 MTR15: 1;
__REG32 MBW15: 1;
__REG32 MPL14: 1;
__REG32 MTW14: 1;
__REG32 MTR14: 1;
__REG32 MBW14: 1;
__REG32 MPL13: 1;
__REG32 MTW13: 1;
__REG32 MTR13: 1;
__REG32 MBW13: 1;
__REG32 MPL12: 1;
__REG32 MTW12: 1;
__REG32 MTR12: 1;
__REG32 MBW12: 1;
__REG32 MPL11: 1;
__REG32 MTW11: 1;
__REG32 MTR11: 1;
__REG32 MBW11: 1;
__REG32 MPL10: 1;
__REG32 MTW10: 1;
__REG32 MTR10: 1;
__REG32 MBW10: 1;
__REG32 MPL9 : 1;
__REG32 MTW9 : 1;
__REG32 MTR9 : 1;
__REG32 MBW9 : 1;
__REG32 MPL8 : 1;
__REG32 MTW8 : 1;
__REG32 MTR8 : 1;
__REG32 MBW8 : 1;
} __aips_mpr_2_bits;

/* Peripheral Access Control Registers (PACR_1, PACR_2, PACR_3, and PACR_4) */
/* Off-Platform Peripheral Access Control Registers (OPACR_1, OPACR_2, OPACR_3, OPACR_4, and OPACR_5) */
typedef struct{
__REG32 TP7   : 1;
__REG32 WP7   : 1;
__REG32 SP7   : 1;
__REG32 BW7   : 1;
__REG32 TP6   : 1;
__REG32 WP6   : 1;
__REG32 SP6   : 1;
__REG32 BW6   : 1;
__REG32 TP5   : 1;
__REG32 WP5   : 1;
__REG32 SP5   : 1;
__REG32 BW5   : 1;
__REG32 TP4   : 1;
__REG32 WP4   : 1;
__REG32 SP4   : 1;
__REG32 BW4   : 1;
__REG32 TP3   : 1;
__REG32 WP3   : 1;
__REG32 SP3   : 1;
__REG32 BW3   : 1;
__REG32 TP2   : 1;
__REG32 WP2   : 1;
__REG32 SP2   : 1;
__REG32 BW2   : 1;
__REG32 TP1   : 1;
__REG32 WP1   : 1;
__REG32 SP1   : 1;
__REG32 BW1   : 1;
__REG32 TP0   : 1;
__REG32 WP0   : 1;
__REG32 SP0   : 1;
__REG32 BW0   : 1;
} __aips_pacr_1_bits;

typedef struct{
__REG32 TP15  : 1;
__REG32 WP15  : 1;
__REG32 SP15  : 1;
__REG32 BW15  : 1;
__REG32 TP14  : 1;
__REG32 WP14  : 1;
__REG32 SP14  : 1;
__REG32 BW14  : 1;
__REG32 TP13  : 1;
__REG32 WP13  : 1;
__REG32 SP13  : 1;
__REG32 BW13  : 1;
__REG32 TP12  : 1;
__REG32 WP12  : 1;
__REG32 SP12  : 1;
__REG32 BW12  : 1;
__REG32 TP11  : 1;
__REG32 WP11  : 1;
__REG32 SP11  : 1;
__REG32 BW11  : 1;
__REG32 TP10  : 1;
__REG32 WP10  : 1;
__REG32 SP10  : 1;
__REG32 BW10  : 1;
__REG32 TP9   : 1;
__REG32 WP9   : 1;
__REG32 SP9   : 1;
__REG32 BW9   : 1;
__REG32 TP8   : 1;
__REG32 WP8   : 1;
__REG32 SP8   : 1;
__REG32 BW8   : 1;
} __aips_pacr_2_bits;

typedef struct{
__REG32 TP23  : 1;
__REG32 WP23  : 1;
__REG32 SP23  : 1;
__REG32 BW23  : 1;
__REG32 TP22  : 1;
__REG32 WP22  : 1;
__REG32 SP22  : 1;
__REG32 BW22  : 1;
__REG32 TP21  : 1;
__REG32 WP21  : 1;
__REG32 SP21  : 1;
__REG32 BW21  : 1;
__REG32 TP20  : 1;
__REG32 WP20  : 1;
__REG32 SP20  : 1;
__REG32 BW20  : 1;
__REG32 TP19  : 1;
__REG32 WP19  : 1;
__REG32 SP19  : 1;
__REG32 BW19  : 1;
__REG32 TP18  : 1;
__REG32 WP18  : 1;
__REG32 SP18  : 1;
__REG32 BW18  : 1;
__REG32 TP17  : 1;
__REG32 WP17  : 1;
__REG32 SP17  : 1;
__REG32 BW17  : 1;
__REG32 TP16  : 1;
__REG32 WP16  : 1;
__REG32 SP16  : 1;
__REG32 BW16  : 1;
} __aips_pacr_3_bits;

typedef struct{
__REG32 TP31  : 1;
__REG32 WP31  : 1;
__REG32 SP31  : 1;
__REG32 BW31  : 1;
__REG32 TP30  : 1;
__REG32 WP30  : 1;
__REG32 SP30  : 1;
__REG32 BW30  : 1;
__REG32 TP29  : 1;
__REG32 WP29  : 1;
__REG32 SP29  : 1;
__REG32 BW29  : 1;
__REG32 TP28  : 1;
__REG32 WP28  : 1;
__REG32 SP28  : 1;
__REG32 BW28  : 1;
__REG32 TP27  : 1;
__REG32 WP27  : 1;
__REG32 SP27  : 1;
__REG32 BW27  : 1;
__REG32 TP26  : 1;
__REG32 WP26  : 1;
__REG32 SP26  : 1;
__REG32 BW26  : 1;
__REG32 TP25  : 1;
__REG32 WP25  : 1;
__REG32 SP25  : 1;
__REG32 BW25  : 1;
__REG32 TP24  : 1;
__REG32 WP24  : 1;
__REG32 SP24  : 1;
__REG32 BW24  : 1;
} __aips_pacr_4_bits;

typedef struct{
__REG32       : 24;
__REG32 TP33  : 1;
__REG32 WP33  : 1;
__REG32 SP33  : 1;
__REG32 BW33  : 1;
__REG32 TP32  : 1;
__REG32 WP32  : 1;
__REG32 SP32  : 1;
__REG32 BW32  : 1;
} __aips_pacr_5_bits;

/* Master Priority Register (MPR0–MPR4) */
typedef struct{
__REG32 MSTR_0  : 3;
__REG32         : 1;
__REG32 MSTR_1  : 3;
__REG32         : 1;
__REG32 MSTR_2  : 3;
__REG32         : 1;
__REG32 MSTR_3  : 3;
__REG32         : 1;
__REG32 MSTR_4  : 3;
__REG32         : 1;
__REG32 MSTR_5  : 3;
__REG32         : 9;
} __mpr_bits;

/* Slave General Purpose Control Register (SGPCR0–SGPCR4) */
typedef struct{
__REG32 PARK    : 3;
__REG32         : 1;
__REG32 PCTL    : 2;
__REG32         : 2;
__REG32 ARB     : 2;
__REG32         :20;
__REG32 HLP     : 1;
__REG32 RO      : 1;
} __sgpcr_bits;

/* Master General Purpose Control Register */
typedef struct{
__REG32 AULB    : 3;
__REG32         :29;
} __mgpcr_bits;


/* Channel Interrupts (INTR) */
typedef struct{
__REG32 HI0     : 1;
__REG32 HI1     : 1;
__REG32 HI2     : 1;
__REG32 HI3     : 1;
__REG32 HI4     : 1;
__REG32 HI5     : 1;
__REG32 HI6     : 1;
__REG32 HI7     : 1;
__REG32 HI8     : 1;
__REG32 HI9     : 1;
__REG32 HI10    : 1;
__REG32 HI11    : 1;
__REG32 HI12    : 1;
__REG32 HI13    : 1;
__REG32 HI14    : 1;
__REG32 HI15    : 1;
__REG32 HI16    : 1;
__REG32 HI17    : 1;
__REG32 HI18    : 1;
__REG32 HI19    : 1;
__REG32 HI20    : 1;
__REG32 HI21    : 1;
__REG32 HI22    : 1;
__REG32 HI23    : 1;
__REG32 HI24    : 1;
__REG32 HI25    : 1;
__REG32 HI26    : 1;
__REG32 HI27    : 1;
__REG32 HI28    : 1;
__REG32 HI29    : 1;
__REG32 HI30    : 1;
__REG32 HI31    : 1;
} __sdma_intr_bits;

/* Channel Stop/Channel Status (STOP_STAT) */
typedef struct{
__REG32 HE0     : 1;
__REG32 HE1     : 1;
__REG32 HE2     : 1;
__REG32 HE3     : 1;
__REG32 HE4     : 1;
__REG32 HE5     : 1;
__REG32 HE6     : 1;
__REG32 HE7     : 1;
__REG32 HE8     : 1;
__REG32 HE9     : 1;
__REG32 HE10    : 1;
__REG32 HE11    : 1;
__REG32 HE12    : 1;
__REG32 HE13    : 1;
__REG32 HE14    : 1;
__REG32 HE15    : 1;
__REG32 HE16    : 1;
__REG32 HE17    : 1;
__REG32 HE18    : 1;
__REG32 HE19    : 1;
__REG32 HE20    : 1;
__REG32 HE21    : 1;
__REG32 HE22    : 1;
__REG32 HE23    : 1;
__REG32 HE24    : 1;
__REG32 HE25    : 1;
__REG32 HE26    : 1;
__REG32 HE27    : 1;
__REG32 HE28    : 1;
__REG32 HE29    : 1;
__REG32 HE30    : 1;
__REG32 HE31    : 1;
} __sdma_stop_stat_bits;

/* Channel Start (HSTART) */
typedef struct{
__REG32 HSTART0 : 1;
__REG32 HSTART1 : 1;
__REG32 HSTART2 : 1;
__REG32 HSTART3 : 1;
__REG32 HSTART4 : 1;
__REG32 HSTART5 : 1;
__REG32 HSTART6 : 1;
__REG32 HSTART7 : 1;
__REG32 HSTART8 : 1;
__REG32 HSTART9 : 1;
__REG32 HSTART10: 1;
__REG32 HSTART11: 1;
__REG32 HSTART12: 1;
__REG32 HSTART13: 1;
__REG32 HSTART14: 1;
__REG32 HSTART15: 1;
__REG32 HSTART16: 1;
__REG32 HSTART17: 1;
__REG32 HSTART18: 1;
__REG32 HSTART19: 1;
__REG32 HSTART20: 1;
__REG32 HSTART21: 1;
__REG32 HSTART22: 1;
__REG32 HSTART23: 1;
__REG32 HSTART24: 1;
__REG32 HSTART25: 1;
__REG32 HSTART26: 1;
__REG32 HSTART27: 1;
__REG32 HSTART28: 1;
__REG32 HSTART29: 1;
__REG32 HSTART30: 1;
__REG32 HSTART31: 1;
} __sdma_hstart_bits;

/* Channel Event Override (EVTOVR) */
typedef struct{
__REG32 EO0     : 1;
__REG32 EO1     : 1;
__REG32 EO2     : 1;
__REG32 EO3     : 1;
__REG32 EO4     : 1;
__REG32 EO5     : 1;
__REG32 EO6     : 1;
__REG32 EO7     : 1;
__REG32 EO8     : 1;
__REG32 EO9     : 1;
__REG32 EO10    : 1;
__REG32 EO11    : 1;
__REG32 EO12    : 1;
__REG32 EO13    : 1;
__REG32 EO14    : 1;
__REG32 EO15    : 1;
__REG32 EO16    : 1;
__REG32 EO17    : 1;
__REG32 EO18    : 1;
__REG32 EO19    : 1;
__REG32 EO20    : 1;
__REG32 EO21    : 1;
__REG32 EO22    : 1;
__REG32 EO23    : 1;
__REG32 EO24    : 1;
__REG32 EO25    : 1;
__REG32 EO26    : 1;
__REG32 EO27    : 1;
__REG32 EO28    : 1;
__REG32 EO29    : 1;
__REG32 EO30    : 1;
__REG32 EO31    : 1;
} __sdma_evtovr_bits;

/* Channel DSP Override (DSPOVR) */
typedef struct{
__REG32 DO0     : 1;
__REG32 DO1     : 1;
__REG32 DO2     : 1;
__REG32 DO3     : 1;
__REG32 DO4     : 1;
__REG32 DO5     : 1;
__REG32 DO6     : 1;
__REG32 DO7     : 1;
__REG32 DO8     : 1;
__REG32 DO9     : 1;
__REG32 DO10    : 1;
__REG32 DO11    : 1;
__REG32 DO12    : 1;
__REG32 DO13    : 1;
__REG32 DO14    : 1;
__REG32 DO15    : 1;
__REG32 DO16    : 1;
__REG32 DO17    : 1;
__REG32 DO18    : 1;
__REG32 DO19    : 1;
__REG32 DO20    : 1;
__REG32 DO21    : 1;
__REG32 DO22    : 1;
__REG32 DO23    : 1;
__REG32 DO24    : 1;
__REG32 DO25    : 1;
__REG32 DO26    : 1;
__REG32 DO27    : 1;
__REG32 DO28    : 1;
__REG32 DO29    : 1;
__REG32 DO30    : 1;
__REG32 DO31    : 1;
} __sdma_dspovr_bits;

/* Channel AP Override (HOSTOVR) */
typedef struct{
__REG32 HO0     : 1;
__REG32 HO1     : 1;
__REG32 HO2     : 1;
__REG32 HO3     : 1;
__REG32 HO4     : 1;
__REG32 HO5     : 1;
__REG32 HO6     : 1;
__REG32 HO7     : 1;
__REG32 HO8     : 1;
__REG32 HO9     : 1;
__REG32 HO10    : 1;
__REG32 HO11    : 1;
__REG32 HO12    : 1;
__REG32 HO13    : 1;
__REG32 HO14    : 1;
__REG32 HO15    : 1;
__REG32 HO16    : 1;
__REG32 HO17    : 1;
__REG32 HO18    : 1;
__REG32 HO19    : 1;
__REG32 HO20    : 1;
__REG32 HO21    : 1;
__REG32 HO22    : 1;
__REG32 HO23    : 1;
__REG32 HO24    : 1;
__REG32 HO25    : 1;
__REG32 HO26    : 1;
__REG32 HO27    : 1;
__REG32 HO28    : 1;
__REG32 HO29    : 1;
__REG32 HO30    : 1;
__REG32 HO31    : 1;
} __sdma_hostovr_bits;

/* Channel Event Pending (EVTPEND) */
typedef struct{
__REG32 EP0     : 1;
__REG32 EP1     : 1;
__REG32 EP2     : 1;
__REG32 EP3     : 1;
__REG32 EP4     : 1;
__REG32 EP5     : 1;
__REG32 EP6     : 1;
__REG32 EP7     : 1;
__REG32 EP8     : 1;
__REG32 EP9     : 1;
__REG32 EP10    : 1;
__REG32 EP11    : 1;
__REG32 EP12    : 1;
__REG32 EP13    : 1;
__REG32 EP14    : 1;
__REG32 EP15    : 1;
__REG32 EP16    : 1;
__REG32 EP17    : 1;
__REG32 EP18    : 1;
__REG32 EP19    : 1;
__REG32 EP20    : 1;
__REG32 EP21    : 1;
__REG32 EP22    : 1;
__REG32 EP23    : 1;
__REG32 EP24    : 1;
__REG32 EP25    : 1;
__REG32 EP26    : 1;
__REG32 EP27    : 1;
__REG32 EP28    : 1;
__REG32 EP29    : 1;
__REG32 EP30    : 1;
__REG32 EP31    : 1;
} __sdma_evtpend_bits;

/* Reset Register (RESET) */
typedef struct{
__REG32 RESET   : 1;
__REG32 RESCHED : 1;
__REG32         :30;
} __sdma_reset_bits;

/* DMA Request Error Register (EVTERR) */
typedef struct{
__REG32 CHNERR0 : 1;
__REG32 CHNERR1 : 1;
__REG32 CHNERR2 : 1;
__REG32 CHNERR3 : 1;
__REG32 CHNERR4 : 1;
__REG32 CHNERR5 : 1;
__REG32 CHNERR6 : 1;
__REG32 CHNERR7 : 1;
__REG32 CHNERR8 : 1;
__REG32 CHNERR9 : 1;
__REG32 CHNERR10: 1;
__REG32 CHNERR11: 1;
__REG32 CHNERR12: 1;
__REG32 CHNERR13: 1;
__REG32 CHNERR14: 1;
__REG32 CHNERR15: 1;
__REG32 CHNERR16: 1;
__REG32 CHNERR17: 1;
__REG32 CHNERR18: 1;
__REG32 CHNERR19: 1;
__REG32 CHNERR20: 1;
__REG32 CHNERR21: 1;
__REG32 CHNERR22: 1;
__REG32 CHNERR23: 1;
__REG32 CHNERR24: 1;
__REG32 CHNERR25: 1;
__REG32 CHNERR26: 1;
__REG32 CHNERR27: 1;
__REG32 CHNERR28: 1;
__REG32 CHNERR29: 1;
__REG32 CHNERR30: 1;
__REG32 CHNERR31: 1;
} __sdma_evterr_bits;

/* Channel AP Interrupt Mask Flags (INTRMASK) */
typedef struct{
__REG32 HIMASK0 : 1;
__REG32 HIMASK1 : 1;
__REG32 HIMASK2 : 1;
__REG32 HIMASK3 : 1;
__REG32 HIMASK4 : 1;
__REG32 HIMASK5 : 1;
__REG32 HIMASK6 : 1;
__REG32 HIMASK7 : 1;
__REG32 HIMASK8 : 1;
__REG32 HIMASK9 : 1;
__REG32 HIMASK10: 1;
__REG32 HIMASK11: 1;
__REG32 HIMASK12: 1;
__REG32 HIMASK13: 1;
__REG32 HIMASK14: 1;
__REG32 HIMASK15: 1;
__REG32 HIMASK16: 1;
__REG32 HIMASK17: 1;
__REG32 HIMASK18: 1;
__REG32 HIMASK19: 1;
__REG32 HIMASK20: 1;
__REG32 HIMASK21: 1;
__REG32 HIMASK22: 1;
__REG32 HIMASK23: 1;
__REG32 HIMASK24: 1;
__REG32 HIMASK25: 1;
__REG32 HIMASK26: 1;
__REG32 HIMASK27: 1;
__REG32 HIMASK28: 1;
__REG32 HIMASK29: 1;
__REG32 HIMASK30: 1;
__REG32 HIMASK31: 1;
} __sdma_intrmask_bits;

/* Schedule Status (PSW) */
typedef struct{
__REG32 CCR     : 5;
__REG32 CCP     : 3;
__REG32 NCR     : 5;
__REG32 NCP     : 3;
__REG32         :16;
} __sdma_psw_bits;

/* Configuration Register (CONFIG) */
typedef struct{
__REG32 CSM     : 2;
__REG32         : 2;
__REG32 ACR     : 1;
__REG32         : 6;
__REG32 RTDOBS  : 1;
__REG32 DSPDMA  : 1;
__REG32         :19;
} __sdma_config_bits;

/* OnCE Enable (ONCE_ENB) */
typedef struct{
__REG32 ENB     : 1;
__REG32         :31;
} __sdma_once_enb_bits;

/* OnCE Instruction Register (ONCE_INSTR) */
typedef struct{
__REG32 INSTR   :16;
__REG32         :16;
} __sdma_once_instr_bits;

/* OnCE Status Register (ONCE_STAT) */
typedef struct{
__REG32 ECDR    : 3;
__REG32         : 4;
__REG32 MST     : 1;
__REG32 SWB     : 1;
__REG32 ODR     : 1;
__REG32 EDR     : 1;
__REG32 RCV     : 1;
__REG32 PST     : 4;
__REG32         :16;
} __sdma_once_stat_bits;

/* OnCE Command Register (ONCE_CMD) */
typedef struct{
__REG32 CMD     : 4;
__REG32         :28;
} __sdma_once_cmd_bits;

/* Illegal Instruction Trap Address (ILLINSTADDR) */
typedef struct{
__REG32 ILLINSTADDR :14;
__REG32             :18;
} __sdma_illinstaddr_bits;

/* Channel 0 Boot Address (CHN0ADDR) */
typedef struct{
__REG32 CHN0ADDR    :14;
__REG32 SMSZ        : 1;
__REG32             :17;
} __sdma_chn0addr_bits;

/* DMA Requests (EVT_MIRROR) */
typedef struct{
__REG32 EVENTS0 : 1;
__REG32 EVENTS1 : 1;
__REG32 EVENTS2 : 1;
__REG32 EVENTS3 : 1;
__REG32 EVENTS4 : 1;
__REG32 EVENTS5 : 1;
__REG32 EVENTS6 : 1;
__REG32 EVENTS7 : 1;
__REG32 EVENTS8 : 1;
__REG32 EVENTS9 : 1;
__REG32 EVENTS10: 1;
__REG32 EVENTS11: 1;
__REG32 EVENTS12: 1;
__REG32 EVENTS13: 1;
__REG32 EVENTS14: 1;
__REG32 EVENTS15: 1;
__REG32 EVENTS16: 1;
__REG32 EVENTS17: 1;
__REG32 EVENTS18: 1;
__REG32 EVENTS19: 1;
__REG32 EVENTS20: 1;
__REG32 EVENTS21: 1;
__REG32 EVENTS22: 1;
__REG32 EVENTS23: 1;
__REG32 EVENTS24: 1;
__REG32 EVENTS25: 1;
__REG32 EVENTS26: 1;
__REG32 EVENTS27: 1;
__REG32 EVENTS28: 1;
__REG32 EVENTS29: 1;
__REG32 EVENTS30: 1;
__REG32 EVENTS31: 1;
} __sdma_evt_mirror_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF1)*/
typedef struct{
__REG32 NUM0        : 5;
__REG32             : 1;
__REG32 CNF0        : 1;
__REG32             : 1;
__REG32 NUM1        : 5;
__REG32             : 1;
__REG32 CNF1        : 1;
__REG32             : 1;
__REG32 NUM2        : 5;
__REG32             : 1;
__REG32 CNF2        : 1;
__REG32             : 1;
__REG32 NUM3        : 5;
__REG32             : 1;
__REG32 CNF3        : 1;
__REG32             : 1;
} __sdma_xtrig1_conf_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF2)*/
typedef struct{
__REG32 NUM4        : 5;
__REG32             : 1;
__REG32 CNF4        : 1;
__REG32             : 1;
__REG32 NUM5        : 5;
__REG32             : 1;
__REG32 CNF5        : 1;
__REG32             : 1;
__REG32 NUM6        : 5;
__REG32             : 1;
__REG32 CNF6        : 1;
__REG32             : 1;
__REG32 NUM7        : 5;
__REG32             : 1;
__REG32 CNF7        : 1;
__REG32             : 1;
} __sdma_xtrig2_conf_bits;

/* Channel Priority Registers (CHNPRIn) */
typedef struct{
__REG32 CHNPRI      : 3;
__REG32             :29;
} __sdma_chnpri_bits;

/* Channel Priority Registers (CHNPRIn) */
typedef struct{
__REG32 RAR         : 3;
__REG32             :13;
__REG32 ROI         : 2;
__REG32             :12;
__REG32 RMO         : 2;
} __spba_prr_bits;

/* Port Timing Control Register x (PTCR) */
typedef struct{
__REG32             :11;
__REG32 SYN         : 1;
__REG32 RCSEL       : 4;
__REG32 RCLKDIR     : 1;
__REG32 RFSEL       : 4;
__REG32 RFSDIR      : 1;
__REG32 TCSEL       : 4;
__REG32 TCLKDIR     : 1;
__REG32 TFSEL       : 4;
__REG32 TFSDIR      : 1;
} __audmux_ptcr_bits;

/* Port Data Control Register x (PDCR) */
typedef struct{
__REG32 INMMASK     : 8;
__REG32 MODE        : 2;
__REG32             : 2;
__REG32 TXRXEN      : 1;
__REG32 RXDSEL      : 3;
__REG32             :16;
} __audmux_pdcr_bits;

/* CE Bus Network Mode Control Register (CNMCR) */
typedef struct{
__REG32 CNTLOW      : 8;
__REG32 CNTHI       : 8;
__REG32 CLKPOL      : 1;
__REG32 FSPOL       : 1;
__REG32 CEN         : 1;
__REG32             :13;
} __audmux_cnmcr_bits;

/* IPU Configuration Register (IPU_CONF) */
typedef struct{
__REG32 CSI_EN      : 1;
__REG32 IC_EN       : 1;
__REG32 ROT_EN      : 1;
__REG32 PF_EN       : 1;
__REG32 SDC_EN      : 1;
__REG32 ADC_EN      : 1;
__REG32 DI_EN       : 1;
__REG32 DU_EN       : 1;
__REG32 PXL_ENDIAN  : 1;
__REG32             :23;
} __ipu_conf_bits;

/* IPU Channels Buffer 0-1 Ready Register (IPU_CHA_BUFn_RDY) */
typedef struct{
__REG32  DMAIC_0_BUF_RDY   : 1;
__REG32  DMAIC_1_BUF_RDY   : 1;
__REG32  DMAIC_2_BUF_RDY   : 1;
__REG32  DMAIC_3_BUF_RDY   : 1;
__REG32  DMAIC_4_BUF_RDY   : 1;
__REG32  DMAIC_5_BUF_RDY   : 1;
__REG32  DMAIC_6_BUF_RDY   : 1;
__REG32  DMAIC_7_BUF_RDY   : 1;
__REG32  DMAIC_8_BUF_RDY   : 1;
__REG32  DMAIC_9_BUF_RDY   : 1;
__REG32 DMAIC_10_BUF_RDY   : 1;
__REG32 DMAIC_11_BUF_RDY   : 1;
__REG32 DMAIC_12_BUF_RDY   : 1;
__REG32 DMAIC_13_BUF_RDY   : 1;
__REG32 DMASDC_0_BUF_RDY   : 1;
__REG32 DMASDC_1_BUF_RDY   : 1;
__REG32 DMASDC_2_BUF_RDY   : 1;
__REG32 DMASDC_3_BUF_RDY   : 1;
__REG32 DMAADC_2_BUF_RDY   : 1;
__REG32 DMAADC_3_BUF_RDY   : 1;
__REG32 DMAADC_4_BUF_RDY   : 1;
__REG32 DMAADC_5_BUF_RDY   : 1;
__REG32 DMAADC_6_BUF_RDY   : 1;
__REG32 DMAADC_7_BUF_RDY   : 1;
__REG32  DMAPF_0_BUF_RDY   : 1;
__REG32  DMAPF_1_BUF_RDY   : 1;
__REG32  DMAPF_2_BUF_RDY   : 1;
__REG32  DMAPF_3_BUF_RDY   : 1;
__REG32  DMAPF_4_BUF_RDY   : 1;
__REG32  DMAPF_5_BUF_RDY   : 1;
__REG32  DMAPF_6_BUF_RDY   : 1;
__REG32  DMAPF_7_BUF_RDY   : 1;
} __ipu_cha_buf_rdy_bits;

/* IPU Channel Double Buffer Mode Select Register (IPU_CHA_DB_MODE_SEL) */
typedef struct{
__REG32  DMAIC_0_DBMS       : 1;
__REG32  DMAIC_1_DBMS       : 1;
__REG32  DMAIC_2_DBMS       : 1;
__REG32  DMAIC_3_DBMS       : 1;
__REG32  DMAIC_4_DBMS       : 1;
__REG32  DMAIC_5_DBMS       : 1;
__REG32  DMAIC_6_DBMS       : 1;
__REG32  DMAIC_7_DBMS       : 1;
__REG32  DMAIC_8_DBMS       : 1;
__REG32  DMAIC_9_DBMS       : 1;
__REG32 DMAIC_10_DBMS       : 1;
__REG32 DMAIC_11_DBMS       : 1;
__REG32 DMAIC_12_DBMS       : 1;
__REG32 DMAIC_13_DBMS       : 1;
__REG32 DMASDC_0_DBMS       : 1;
__REG32 DMASDC_1_DBMS       : 1;
__REG32 DMASDC_2_DBMS       : 1;
__REG32                     : 1;
__REG32 DMAADC_2_DBMS       : 1;
__REG32 DMAADC_3_DBMS       : 1;
__REG32                     : 6;
__REG32  DMAPF_2_DBMS       : 1;
__REG32                     : 2;
__REG32  DMAPF_5_DBMS       : 1;
__REG32                     : 2;
} __ipu_cha_db_mode_sel_bits;

/* IPU Channel Current Buffer Register (IPU_CHA_CUR_BUF) */
typedef struct{
__REG32  DMAIC_0_CUR_BUF    : 1;
__REG32  DMAIC_1_CUR_BUF    : 1;
__REG32  DMAIC_2_CUR_BUF    : 1;
__REG32  DMAIC_3_CUR_BUF    : 1;
__REG32  DMAIC_4_CUR_BUF    : 1;
__REG32  DMAIC_5_CUR_BUF    : 1;
__REG32  DMAIC_6_CUR_BUF    : 1;
__REG32  DMAIC_7_CUR_BUF    : 1;
__REG32  DMAIC_8_CUR_BUF    : 1;
__REG32  DMAIC_9_CUR_BUF    : 1;
__REG32 DMAIC_10_CUR_BUF    : 1;
__REG32 DMAIC_11_CUR_BUF    : 1;
__REG32 DMAIC_12_CUR_BUF    : 1;
__REG32 DMAIC_13_CUR_BUF    : 1;
__REG32 DMASDC_0_CUR_BUF    : 1;
__REG32 DMASDC_1_CUR_BUF    : 1;
__REG32 DMASDC_2_CUR_BUF    : 1;
__REG32 DMASDC_3_CUR_BUF    : 1;
__REG32 DMAADC_2_CUR_BUF    : 1;
__REG32 DMAADC_3_CUR_BUF    : 1;
__REG32 DMAADC_4_CUR_BUF    : 1;
__REG32 DMAADC_5_CUR_BUF    : 1;
__REG32 DMAADC_6_CUR_BUF    : 1;
__REG32 DMAADC_7_CUR_BUF    : 1;
__REG32  DMAPF_0_CUR_BUF    : 1;
__REG32  DMAPF_1_CUR_BUF    : 1;
__REG32  DMAPF_2_CUR_BUF    : 1;
__REG32  DMAPF_3_CUR_BUF    : 1;
__REG32  DMAPF_4_CUR_BUF    : 1;
__REG32  DMAPF_5_CUR_BUF    : 1;
__REG32  DMAPF_6_CUR_BUF    : 1;
__REG32  DMAPF_7_CUR_BUF    : 1;
} __ipu_cha_cur_buf_bits;

/* IPU Frame Synchronization Processing Flow Register (IPU_FS_PROC_FLOW)*/
typedef struct{
__REG32 ENC_IN_VALID        : 1;
__REG32 VF_IN_VALID         : 1;
__REG32                     : 2;
__REG32 PRPENC_DEST_SEL     : 1;
__REG32 PRPENC_ROT_SRC_SEL  : 1;
__REG32 PRPVF_ROT_SRC_SEL   : 1;
__REG32                     : 1;
__REG32 PP_SRC_SEL          : 2;
__REG32 PP_ROT_SRC_SEL      : 2;
__REG32 PF_DEST_SEL         : 2;
__REG32                     : 2;
__REG32 PRPVF_DEST_SEL      : 3;
__REG32                     : 1;
__REG32 PRPVF_ROT_DEST_SEL  : 3;
__REG32                     : 1;
__REG32 PP_DEST_SEL         : 3;
__REG32                     : 1;
__REG32 PP_ROT_DEST_SEL     : 3;
__REG32                     : 1;
} __ipu_fs_proc_flow_bits;

/* IPU Frame Synchronization Displaying Flow Register (IPU_FS_DISP_FLOW) */
typedef struct{
__REG32 SDC0_SRC_SEL        : 3;
__REG32                     : 1;
__REG32 SDC1_SRC_SEL        : 3;
__REG32                     : 1;
__REG32 ADC2_SRC_SEL        : 3;
__REG32                     : 1;
__REG32 ADC3_SRC_SEL        : 3;
__REG32                     : 1;
__REG32 AUTO_REF_PER        :10;
__REG32                     : 6;
} __ipu_fs_disp_flow_bits;

/* IPU Tasks Status Register (IPU_TASKS_STAT) */
typedef struct{
__REG32 CSI_SKIP_TSTAT      : 2;
__REG32 CSI2MEM_TSTAT       : 2;
__REG32 MEM2PRP_TSTAT       : 3;
__REG32 ENC_TSTAT           : 2;
__REG32 VF_TSTAT            : 2;
__REG32 PP_TSTAT            : 2;
__REG32 SDC_PIX_SKIP        : 1;
__REG32 PF_H264_Y_PAUSE     : 1;
__REG32                     : 1;
__REG32 ENC_ROT_TSTAT       : 2;
__REG32 VF_ROT_TSTAT        : 2;
__REG32 PP_ROT_TSTAT        : 2;
__REG32 PF_TSTAT            : 2;
__REG32 ADCSYS1_TSTAT       : 2;
__REG32 ADCSYS2_TSTAT       : 2;
__REG32 ADC_PRPCHAN_LOCK    : 1;
__REG32 ADC_PPCHAN_LOCK     : 1;
__REG32 ADC_SYS1CHAN_LOCK   : 1;
__REG32 ADC_SYS2CHAN_LOCK   : 1;
} __ipu_tasks_stat_bits;

/* IPU Internal Memory Access Address and Data Registers (IPU_IMA_ADDR) */
typedef struct{
__REG32 WORD_NU             : 3;
__REG32 ROW_NU              :13;
__REG32 MEM_NU              : 4;
__REG32                     :12;
} __ipu_ima_addr_bits;

/* IPU Interrupt Control Register 1 (IPU_INT_CTRL_1) */
typedef struct{
__REG32  DMAIC_0_EOF_EN     : 1;
__REG32  DMAIC_1_EOF_EN     : 1;
__REG32  DMAIC_2_EOF_EN     : 1;
__REG32  DMAIC_3_EOF_EN     : 1;
__REG32  DMAIC_4_EOF_EN     : 1;
__REG32  DMAIC_5_EOF_EN     : 1;
__REG32  DMAIC_6_EOF_EN     : 1;
__REG32  DMAIC_7_EOF_EN     : 1;
__REG32  DMAIC_8_EOF_EN     : 1;
__REG32  DMAIC_9_EOF_EN     : 1;
__REG32 DMAIC_10_EOF_EN     : 1;
__REG32 DMAIC_11_EOF_EN     : 1;
__REG32 DMAIC_12_EOF_EN     : 1;
__REG32 DMAIC_13_EOF_EN     : 1;
__REG32 DMASDC_0_EOF_EN     : 1;
__REG32 DMASDC_1_EOF_EN     : 1;
__REG32 DMASDC_2_EOF_EN     : 1;
__REG32 DMASDC_3_EOF_EN     : 1;
__REG32 DMAADC_2_EOF_EN     : 1;
__REG32 DMAADC_3_EOF_EN     : 1;
__REG32 DMAADC_4_EOF_EN     : 1;
__REG32 DMAADC_5_EOF_EN     : 1;
__REG32 DMAADC_6_EOF_EN     : 1;
__REG32 DMAADC_7_EOF_EN     : 1;
__REG32  DMAPF_0_EOF_EN     : 1;
__REG32  DMAPF_1_EOF_EN     : 1;
__REG32  DMAPF_2_EOF_EN     : 1;
__REG32  DMAPF_3_EOF_EN     : 1;
__REG32  DMAPF_4_EOF_EN     : 1;
__REG32  DMAPF_5_EOF_EN     : 1;
__REG32  DMAPF_6_EOF_EN     : 1;
__REG32  DMAPF_7_EOF_EN     : 1;
} __ipu_int_ctrl_1_bits;

/* IPU Interrupt Control Register 2 (IPU_INT_CTRL_2) */
typedef struct{
__REG32  DMAIC_0_NFACK_EN   : 1;
__REG32  DMAIC_1_NFACK_EN   : 1;
__REG32  DMAIC_2_NFACK_EN   : 1;
__REG32  DMAIC_3_NFACK_EN   : 1;
__REG32  DMAIC_4_NFACK_EN   : 1;
__REG32  DMAIC_5_NFACK_EN   : 1;
__REG32  DMAIC_6_NFACK_EN   : 1;
__REG32  DMAIC_7_NFACK_EN   : 1;
__REG32  DMAIC_8_NFACK_EN   : 1;
__REG32  DMAIC_9_NFACK_EN   : 1;
__REG32 DMAIC_10_NFACK_EN   : 1;
__REG32 DMAIC_11_NFACK_EN   : 1;
__REG32 DMAIC_12_NFACK_EN   : 1;
__REG32 DMAIC_13_NFACK_EN   : 1;
__REG32 DMASDC_0_NFACK_EN   : 1;
__REG32 DMASDC_1_NFACK_EN   : 1;
__REG32 DMASDC_2_NFACK_EN   : 1;
__REG32 DMASDC_3_NFACK_EN   : 1;
__REG32 DMAADC_2_NFACK_EN   : 1;
__REG32 DMAADC_3_NFACK_EN   : 1;
__REG32 DMAADC_4_NFACK_EN   : 1;
__REG32 DMAADC_5_NFACK_EN   : 1;
__REG32 DMAADC_6_NFACK_EN   : 1;
__REG32 DMAADC_7_NFACK_EN   : 1;
__REG32  DMAPF_0_NFACK_EN   : 1;
__REG32  DMAPF_1_NFACK_EN   : 1;
__REG32  DMAPF_2_NFACK_EN   : 1;
__REG32  DMAPF_3_NFACK_EN   : 1;
__REG32  DMAPF_4_NFACK_EN   : 1;
__REG32  DMAPF_5_NFACK_EN   : 1;
__REG32  DMAPF_6_NFACK_EN   : 1;
__REG32  DMAPF_7_NFACK_EN   : 1;
} __ipu_int_ctrl_2_bits;

/* IPU Interrupt Control Register 3 (IPU_INT_CTRL_3) */
typedef struct{
__REG32 BRK_RQ_STAT_EN        : 1;
__REG32 SDC_BG_EOF_EN         : 1;
__REG32 SDC_FG_EOF_EN         : 1;
__REG32 SDC_MSK_EOF_EN        : 1;
__REG32 SERIAL_DATA_FINISH_EN : 1;
__REG32 CSI_NF_EN             : 1;
__REG32 CSI_EOF_EN            : 1;
__REG32 DMAIC_3_SBUF_END_EN   : 1;
__REG32 DMAIC_4_SBUF_END_EN   : 1;
__REG32 DMAIC_5_SBUF_END_EN   : 1;
__REG32 DMAIC_6_SBUF_END_EN   : 1;
__REG32 DMASDC_0_SBUF_END_EN  : 1;
__REG32 DMASDC_1_SBUF_END_EN  : 1;
__REG32 DMASDC_2_SBUF_END_EN  : 1;
__REG32 DMAADC_2_SBUF_END_EN  : 1;
__REG32 DMAADC_3_SBUF_END_EN  : 1;
__REG32 SDC_DISP3_VSYNC_EN    : 1;
__REG32 ADC_DISP0_VSYNC_EN    : 1;
__REG32 ADC_DISP12_VSYNC_EN   : 1;
__REG32 ADC_PRP_EOF_EN        : 1;
__REG32 ADC_PP_EOF_EN         : 1;
__REG32 ADC_SYS1_EOF_EN       : 1;
__REG32 ADC_SYS2_EOF_EN       : 1;
__REG32 STOP_MODE_ACK_EN      : 1;
__REG32                       : 8;
} __ipu_int_ctrl_3_bits;

/* IPU Interrupt Control Register 4 (IPU_INT_CTRL_4) */
typedef struct{
__REG32  DMAIC_0_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_1_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_2_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_3_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_4_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_5_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_6_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_7_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_8_NFB4EOF_ERR_EN : 1;
__REG32  DMAIC_9_NFB4EOF_ERR_EN : 1;
__REG32 DMAIC_10_NFB4EOF_ERR_EN : 1;
__REG32 DMAIC_11_NFB4EOF_ERR_EN : 1;
__REG32 DMAIC_12_NFB4EOF_ERR_EN : 1;
__REG32 DMAIC_13_NFB4EOF_ERR_EN : 1;
__REG32 DMASDC_0_NFB4EOF_ERR_EN : 1;
__REG32 DMASDC_1_NFB4EOF_ERR_EN : 1;
__REG32 DMASDC_2_NFB4EOF_ERR_EN : 1;
__REG32 DMASDC_3_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_2_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_3_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_4_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_5_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_6_NFB4EOF_ERR_EN : 1;
__REG32 DMAADC_7_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_0_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_1_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_2_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_3_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_4_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_5_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_6_NFB4EOF_ERR_EN : 1;
__REG32  DMAPF_7_NFB4EOF_ERR_EN : 1;
} __ipu_int_ctrl_4_bits;

/* IPU Interrupt Control Register 5 (IPU_INT_CTRL_5) */
typedef struct{
__REG32 BAYER_BUF_OVF_ERR_EN      : 1;
__REG32 ENC_BUF_OVF_ERR_EN        : 1;
__REG32 VF_BUF_OVF_ERR_EN         : 1;
__REG32 ADC_PP_TEARING_ERR_EN     : 1;
__REG32 ADC_SYS1_TEARING_ERR_EN   : 1;
__REG32 ADC_SYS2_TEARING_ERR_EN   : 1;
__REG32 AHB_M1_ERR_EN             : 1;
__REG32 AHB_M2_ERR_EN             : 1;
__REG32 SDC_BGD_ERR_EN            : 1;
__REG32 SDC_FGD_ERR_EN            : 1;
__REG32 SDC_MSK_ERR_EN            : 1;
__REG32 BAYER_FRM_LOST_ERR_EN     : 1;
__REG32 ENC_FRM_LOST_ERR_EN       : 1;
__REG32 VF_FRM_LOST_ERR_EN        : 1;
__REG32 DI_ADC_LOCK_ERR_EN        : 1;
__REG32 DI_LLA_LOCK_ERR_EN        : 1;
__REG32 SAHB_ADDR_ERR_EN          : 1;
__REG32                           :15;
} __ipu_int_ctrl_5_bits;

/* IPU Interrupt Status Register 1 (IPU_INT_STAT_1) */
typedef struct{
__REG32  DMAIC_0_EOF  : 1;
__REG32  DMAIC_1_EOF  : 1;
__REG32  DMAIC_2_EOF  : 1;
__REG32  DMAIC_3_EOF  : 1;
__REG32  DMAIC_4_EOF  : 1;
__REG32  DMAIC_5_EOF  : 1;
__REG32  DMAIC_6_EOF  : 1;
__REG32  DMAIC_7_EOF  : 1;
__REG32  DMAIC_8_EOF  : 1;
__REG32  DMAIC_9_EOF  : 1;
__REG32 DMAIC_10_EOF  : 1;
__REG32 DMAIC_11_EOF  : 1;
__REG32 DMAIC_12_EOF  : 1;
__REG32 DMAIC_13_EOF  : 1;
__REG32 DMASDC_0_EOF  : 1;
__REG32 DMASDC_1_EOF  : 1;
__REG32 DMASDC_2_EOF  : 1;
__REG32 DMASDC_3_EOF  : 1;
__REG32 DMAADC_2_EOF  : 1;
__REG32 DMAADC_3_EOF  : 1;
__REG32 DMAADC_4_EOF  : 1;
__REG32 DMAADC_5_EOF  : 1;
__REG32 DMAADC_6_EOF  : 1;
__REG32 DMAADC_7_EOF  : 1;
__REG32  DMAPF_0_EOF  : 1;
__REG32  DMAPF_1_EOF  : 1;
__REG32  DMAPF_2_EOF  : 1;
__REG32  DMAPF_3_EOF  : 1;
__REG32  DMAPF_4_EOF  : 1;
__REG32  DMAPF_5_EOF  : 1;
__REG32  DMAPF_6_EOF  : 1;
__REG32  DMAPF_7_EOF  : 1;
} __ipu_int_stat_1_bits;

/* IPU Interrupt Status Register 2 (IPU_INT_STAT_2) */
typedef struct{
__REG32  DMAIC_0_NFACK  : 1;
__REG32  DMAIC_1_NFACK  : 1;
__REG32  DMAIC_2_NFACK  : 1;
__REG32  DMAIC_3_NFACK  : 1;
__REG32  DMAIC_4_NFACK  : 1;
__REG32  DMAIC_5_NFACK  : 1;
__REG32  DMAIC_6_NFACK  : 1;
__REG32  DMAIC_7_NFACK  : 1;
__REG32  DMAIC_8_NFACK  : 1;
__REG32  DMAIC_9_NFACK  : 1;
__REG32 DMAIC_10_NFACK  : 1;
__REG32 DMAIC_11_NFACK  : 1;
__REG32 DMAIC_12_NFACK  : 1;
__REG32 DMAIC_13_NFACK  : 1;
__REG32 DMASDC_0_NFACK  : 1;
__REG32 DMASDC_1_NFACK  : 1;
__REG32 DMASDC_2_NFACK  : 1;
__REG32 DMASDC_3_NFACK  : 1;
__REG32 DMAADC_2_NFACK  : 1;
__REG32 DMAADC_3_NFACK  : 1;
__REG32 DMAADC_4_NFACK  : 1;
__REG32 DMAADC_5_NFACK  : 1;
__REG32 DMAADC_6_NFACK  : 1;
__REG32 DMAADC_7_NFACK  : 1;
__REG32  DMAPF_0_NFACK  : 1;
__REG32  DMAPF_1_NFACK  : 1;
__REG32  DMAPF_2_NFACK  : 1;
__REG32  DMAPF_3_NFACK  : 1;
__REG32  DMAPF_4_NFACK  : 1;
__REG32  DMAPF_5_NFACK  : 1;
__REG32  DMAPF_6_NFACK  : 1;
__REG32  DMAPF_7_NFACK  : 1;
} __ipu_int_stat_2_bits;

/* IPU Interrupt Status Register 3 (IPU_INT_STAT_3) */
typedef struct{
__REG32 BRK_RQ_STAT         : 1;
__REG32 SDC_BG_EOF          : 1;
__REG32 SDC_FG_EOF          : 1;
__REG32 SDC_MSK_EOF         : 1;
__REG32 SERIAL_DATA_FINISH  : 1;
__REG32 CSI_NF              : 1;
__REG32 CSI_EOF             : 1;
__REG32 DMAIC_3_SBUF_END    : 1;
__REG32 DMAIC_4_SBUF_END    : 1;
__REG32 DMAIC_5_SBUF_END    : 1;
__REG32 DMAIC_6_SBUF_END    : 1;
__REG32 DMASDC_0_SBUF_END   : 1;
__REG32 DMASDC_1_SBUF_END   : 1;
__REG32 DMASDC_2_SBUF_END   : 1;
__REG32 DMAADC_2_SBUF_END   : 1;
__REG32 DMAADC_3_SBUF_END   : 1;
__REG32 SDC_DISP3_VSYNC     : 1;
__REG32 ADC_DISP0_VSYNC     : 1;
__REG32 ADC_DISP12_VSYNC    : 1;
__REG32 ADC_PRP_EOF         : 1;
__REG32 ADC_PP_EOF          : 1;
__REG32 ADC_SYS1_EOF        : 1;
__REG32 ADC_SYS2_EOF        : 1;
__REG32 STOP_MODE_ACK       : 1;
__REG32                     : 8;
} __ipu_int_stat_3_bits;

/* IPU Interrupt Status Register 4 (IPU_INT_STAT_4) */
typedef struct{
__REG32  DMAIC_0_NFB4EOF_ERR  : 1;
__REG32  DMAIC_1_NFB4EOF_ERR  : 1;
__REG32  DMAIC_2_NFB4EOF_ERR  : 1;
__REG32  DMAIC_3_NFB4EOF_ERR  : 1;
__REG32  DMAIC_4_NFB4EOF_ERR  : 1;
__REG32  DMAIC_5_NFB4EOF_ERR  : 1;
__REG32  DMAIC_6_NFB4EOF_ERR  : 1;
__REG32  DMAIC_7_NFB4EOF_ERR  : 1;
__REG32  DMAIC_8_NFB4EOF_ERR  : 1;
__REG32  DMAIC_9_NFB4EOF_ERR  : 1;
__REG32 DMAIC_10_NFB4EOF_ERR  : 1;
__REG32 DMAIC_11_NFB4EOF_ERR  : 1;
__REG32 DMAIC_12_NFB4EOF_ERR  : 1;
__REG32 DMAIC_13_NFB4EOF_ERR  : 1;
__REG32 DMASDC_0_NFB4EOF_ERR  : 1;
__REG32 DMASDC_1_NFB4EOF_ERR  : 1;
__REG32 DMASDC_2_NFB4EOF_ERR  : 1;
__REG32 DMASDC_3_NFB4EOF_ERR  : 1;
__REG32 DMAADC_2_NFB4EOF_ERR  : 1;
__REG32 DMAADC_3_NFB4EOF_ERR  : 1;
__REG32 DMAADC_4_NFB4EOF_ERR  : 1;
__REG32 DMAADC_5_NFB4EOF_ERR  : 1;
__REG32 DMAADC_6_NFB4EOF_ERR  : 1;
__REG32 DMAADC_7_NFB4EOF_ERR  : 1;
__REG32  DMAPF_0_NFB4EOF_ERR  : 1;
__REG32  DMAPF_1_NFB4EOF_ERR  : 1;
__REG32  DMAPF_2_NFB4EOF_ERR  : 1;
__REG32  DMAPF_3_NFB4EOF_ERR  : 1;
__REG32  DMAPF_4_NFB4EOF_ERR  : 1;
__REG32  DMAPF_5_NFB4EOF_ERR  : 1;
__REG32  DMAPF_6_NFB4EOF_ERR  : 1;
__REG32  DMAPF_7_NFB4EOF_ERR  : 1;
} __ipu_int_stat_4_bits;

/* IPU Interrupt Status Register 5 (IPU_INT_STAT_5) */
typedef struct{
__REG32 BAYER_BUF_OVF_ERR   : 1;
__REG32 ENC_BUF_OVF_ERR     : 1;
__REG32 VF_BUF_OVF_ERR      : 1;
__REG32 ADC_PP_TEARING_ERR  : 1;
__REG32 ADC_SYS1_TEARING_ERR: 1;
__REG32 ADC_SYS2_TEARING_ERR: 1;
__REG32 AHB_M1_ERR          : 1;
__REG32 AHB_M2_ERR          : 1;
__REG32 SDC_BGD_ERR         : 1;
__REG32 SDC_FGD_ERR         : 1;
__REG32 SDC_MSKD_ERR        : 1;
__REG32 BAYER_FRM_LOST_ERR  : 1;
__REG32 ENC_FRM_LOST_ERR    : 1;
__REG32 VF_FRM_LOST_ERR     : 1;
__REG32 DI_ADC_LOCK_ERR     : 1;
__REG32 DI_LLA_LOCK_ERR     : 1;
__REG32 SAHB_ADDR_ERR       : 1;
__REG32                     :15;
} __ipu_int_stat_5_bits;

/* IPU Break Control Register 1 (IPU_BRK_CTRL_1) */
typedef struct{
__REG32 BRK_EN              : 1;
__REG32 FRC_DBG             : 1;
__REG32 DBG_EXIT            : 1;
__REG32                     : 1;
__REG32 DBG_ENTER_MODE      : 2;
__REG32 BRK_RQ_MODE         : 2;
__REG32 BRK_CHA_COND_EN     : 1;
__REG32 BRK_ROW_COND_EN     : 1;
__REG32 BRK_COL_COND_EN     : 1;
__REG32 BRK_SIG_COND_EN     : 1;
__REG32 SDC_DBG_MASK_DIS    : 1;
__REG32                     : 3;
__REG32 BRK_SIG_SEL         : 5;
__REG32 BRK_GRP_SEL         : 3;
__REG32 BRK_EVNT_NUM        : 8;
} __ipu_brk_ctrl_1_bits;

/* IPU Break Control Register 2 (IPU_BRK_CTRL_2) */
typedef struct{
__REG32 BRK_ROW_NUM         :13;
__REG32 BRK_COL_NUM         :12;
__REG32                     : 1;
__REG32 BRK_CHA_NUM         : 5;
__REG32 CSI_CHA_EN          : 1;
} __ipu_brk_ctrl_2_bits;

/* IPU Break Status Register (IPU_BRK_STAT) */
typedef struct{
__REG32 IPU_BREAK_ACK       : 1;
__REG32 MCU_DBGRQ           : 1;
__REG32 BRK_SRC             : 1;
__REG32                     :29;
} __ipu_brk_stat_bits;

/* IPU Diagnostic Bus Control Register (IPU_DIAGB_CTRL) */
typedef struct{
__REG32 MON_GRP_SEL         : 5;
__REG32                     :27;
} __ipu_diagb_ctrl_bits;

/* CSI Sensor Configuration Register (CSI_SENS_CONF) */
typedef struct{
__REG32 VSYNC_POL           : 1;
__REG32 HSYNC_POL           : 1;
__REG32 DATA_POL            : 1;
__REG32 SENS_PIX_CLK_POL    : 1;
__REG32 SENS_PRTCL          : 2;
__REG32                     : 1;
__REG32 SENS_CLK_SRC        : 1;
__REG32 SENS_DATA_FORMAT    : 2;
__REG32 DATA_WIDTH          : 2;
__REG32                     : 3;
__REG32 EXT_VSYNC           : 1;
__REG32 DIV_RATIO           : 8;
__REG32                     : 8;
} __csi_sens_conf_bits;

/* CSI Sensor Frame Size Register (CSI_SENS_FRM_SIZE) */
typedef struct{
__REG32 SENS_FRM_WIDTH      :12;
__REG32                     : 4;
__REG32 SENS_FRM_HEIGHT     :12;
__REG32                     : 4;
} __csi_sens_frm_size_bits;

/* CSI Actual Frame Size Register (CSI_ACT_FRM_SIZE) */
typedef struct{
__REG32 ACT_FRM_WIDTH       :12;
__REG32                     : 4;
__REG32 ACT_FRM_HEIGHT      :12;
__REG32                     : 4;
} __csi_act_frm_size_bits;

/* CSI Output Frame Control Register (CSI_OUT_FRM_CTRL) */
typedef struct{
__REG32 VSC                 : 8;
__REG32 HSC                 : 8;
__REG32 SKIP_ENC            : 5;
__REG32 SKIP_W_VF           : 5;
__REG32 IC_TV_MODE          : 1;
__REG32                     : 1;
__REG32 VERT_DWNS           : 1;
__REG32 HORZ_DWNS           : 1;
__REG32                     : 2;
} __csi_out_frm_ctrl_bits;

/* CSI Test Control Register (CSI_TST_CTRL) */
typedef struct{
__REG32 PG_R_VALUE          : 8;
__REG32 PG_G_VALUE          : 8;
__REG32 PG_B_VALUE          : 8;
__REG32 TEST_GEN_MODE       : 1;
__REG32                     : 7;
} __csi_tst_ctrl_bits;

/* CSI CCIR Code Register 1 (CSI_CCIR_CODE_1) */
typedef struct{
__REG32 END_FLD0_BLNK_1ST   : 3;
__REG32 STRT_FLD0_BLNK_1ST  : 3;
__REG32 END_FLD0_BLNK_2ND   : 3;
__REG32 STRT_FLD0_BLNK_2ND  : 3;
__REG32                     : 4;
__REG32 END_FLD0_ACTV       : 3;
__REG32 STRT_FLD0_ACTV      : 3;
__REG32                     : 2;
__REG32 CCIR_ERR_DET_EN     : 1;
__REG32                     : 7;
} __csi_ccir_code_1_bits;

/* CSI CCIR Code Register 2 (CSI_CCIR_CODE_2) */
typedef struct{
__REG32 END_FLD1_BLNK_1ST   : 3;
__REG32 STRT_FLD1_BLNK_1ST  : 3;
__REG32 END_FLD1_BLNK_2ND   : 3;
__REG32 STRT_FLD1_BLNK_2ND  : 3;
__REG32                     : 4;
__REG32 END_FLD1_ACTV       : 3;
__REG32 STRT_FLD1_ACTV      : 3;
__REG32                     :10;
} __csi_ccir_code_2_bits;

/* CSI CCIR Code Register 3 (CSI_CCIR_CODE_3) */
typedef struct{
__REG32 CCIR_PRECOM         :24;
__REG32                     : 8;
} __csi_ccir_code_3_bits;

/* CSI Flash Strobe Register 1 (CSI_FLASH_STROBE_1) */
typedef struct{
__REG32 CLOCK_SEL           : 1;
__REG32                     :15;
__REG32 SENS_ROW_DURATION   :16;
} __csi_flash_strobe_1_bits;

/* CSI Flash Strobe Register 2 (CSI_FLASH_STROBE_2) */
typedef struct{
__REG32 STROBE_EN           : 1;
__REG32 STROBE_POL          : 1;
__REG32                     : 1;
__REG32 STROBE_START_TIME   :13;
__REG32 STROBE_DURATION     :16;
} __csi_flash_strobe_2_bits;

/* IC Configuration Register (IC_CONF) */
typedef struct{
__REG32 PRPENC_EN           : 1;
__REG32 PRPENC_CSC1         : 1;
__REG32 PRPENC_ROT_EN       : 1;
__REG32                     : 5;
__REG32 PRPVF_EN            : 1;
__REG32 PRPVF_CSC1          : 1;
__REG32 PRPVF_CSC2          : 1;
__REG32 PRPVF_CMB           : 1;
__REG32 PRPVF_ROT_EN        : 1;
__REG32                     : 3;
__REG32 PP_EN               : 1;
__REG32 PP_CSC1             : 1;
__REG32 PP_CSC2             : 1;
__REG32 PP_CMB              : 1;
__REG32 PP_ROT_EN           : 1;
__REG32                     : 7;
__REG32 IC_GLB_LOC_A        : 1;
__REG32 IC_KEY_COLOR_EN     : 1;
__REG32 RWS_EN              : 1;
__REG32 CSI_MEM_WR_EN       : 1;
} __ic_conf_bits;

/* IC Preprocessing Encoder Resizing Coefficients Register (IC_PRP_ENC_RSC) */
typedef struct{
__REG32 PRPENC_RS_R_H       :14;
__REG32 PRPENC_DS_R_H       : 2;
__REG32 PRPENC_RS_R_V       :14;
__REG32 PRPENC_DS_R_V       : 2;
} __ic_prp_enc_rsc_bits;

/* IC Preprocessing View-Finder Resizing Coefficients Register (IC_PRP_VF_RSC) */
typedef struct{
__REG32 PRPVF_RS_R_H        :14;
__REG32 PRPVF_DS_R_H        : 2;
__REG32 PRPVF_RS_R_V        :14;
__REG32 PRPVF_DS_R_V        : 2;
} __ic_prp_vf_rsc_bits;

/* IC Post-Processing Resizing Coefficients Register (IC_PP_RSC) */
typedef struct{
__REG32 PP_RS_R_H           :14;
__REG32 PP_DS_R_H           : 2;
__REG32 PP_RS_R_V           :14;
__REG32 PP_DS_R_V           : 2;
} __ic_pp_rsc_bits;

/* IC Combining Parameters Register 1 (IC_CMBP_1) */
typedef struct{
__REG32 IC_PRPVF_ALPHA_V    : 8;
__REG32 IC_PP_ALPHA_V       : 8;
__REG32                     :16;
} __ic_cmbp_1_bits;

/* IC Combining Parameters Register 2 (IC_CMBP_2) */
typedef struct{
__REG32 IC_KEY_COLOR_B      : 8;
__REG32 IC_KEY_COLOR_G      : 8;
__REG32 IC_KEY_COLOR_R      : 8;
__REG32                     : 8;
} __ic_cmbp_2_bits;

/* Post Filter (PF) Configuration Register (PF_CONF) */
typedef struct{
__REG32 PF_TYPE             : 3;
__REG32                     : 1;
__REG32 H264_Y_PAUSE_EN     : 1;
__REG32                     :11;
__REG32 H264_Y_PAUSE_ROW    : 6;
__REG32                     :10;
} __pf_conf_bits;

/* IDMAC Configuration Register (IDMAC_CONF) */
typedef struct{
__REG32 PRYM                : 2;
__REG32                     : 2;
__REG32 SRCNT               : 3;
__REG32                     : 1;
__REG32 SINGLE_AHB_M_EN     : 1;
__REG32                     :23;
} __idmac_conf_bits;

/* IDMAC Channel Enable Register (IDMAC_CHA_EN) */
typedef struct{
__REG32  DMAIC_0_EN : 1;
__REG32  DMAIC_1_EN : 1;
__REG32  DMAIC_2_EN : 1;
__REG32  DMAIC_3_EN : 1;
__REG32  DMAIC_4_EN : 1;
__REG32  DMAIC_5_EN : 1;
__REG32  DMAIC_6_EN : 1;
__REG32  DMAIC_7_EN : 1;
__REG32  DMAIC_8_EN : 1;
__REG32  DMAIC_9_EN : 1;
__REG32 DMAIC_10_EN : 1;
__REG32 DMAIC_11_EN : 1;
__REG32 DMAIC_12_EN : 1;
__REG32 DMAIC_13_EN : 1;
__REG32 DMASDC_0_EN : 1;
__REG32 DMASDC_1_EN : 1;
__REG32 DMASDC_2_EN : 1;
__REG32 DMASDC_3_EN : 1;
__REG32 DMAADC_2_EN : 1;
__REG32 DMAADC_3_EN : 1;
__REG32 DMAADC_4_EN : 1;
__REG32 DMAADC_5_EN : 1;
__REG32 DMAADC_6_EN : 1;
__REG32 DMAADC_7_EN : 1;
__REG32  DMAPF_0_EN : 1;
__REG32  DMAPF_1_EN : 1;
__REG32  DMAPF_2_EN : 1;
__REG32  DMAPF_3_EN : 1;
__REG32  DMAPF_4_EN : 1;
__REG32  DMAPF_5_EN : 1;
__REG32  DMAPF_6_EN : 1;
__REG32  DMAPF_7_EN : 1;
} __idmac_cha_en_bits;

/* IDMAC Channel Priority Register (IDMAC_CHA_PRI) */
typedef struct{
__REG32  DMAIC_0_PRI  : 1;
__REG32  DMAIC_1_PRI  : 1;
__REG32  DMAIC_2_PRI  : 1;
__REG32  DMAIC_3_PRI  : 1;
__REG32  DMAIC_4_PRI  : 1;
__REG32  DMAIC_5_PRI  : 1;
__REG32  DMAIC_6_PRI  : 1;
__REG32  DMAIC_7_PRI  : 1;
__REG32  DMAIC_8_PRI  : 1;
__REG32  DMAIC_9_PRI  : 1;
__REG32 DMAIC_10_PRI  : 1;
__REG32 DMAIC_11_PRI  : 1;
__REG32 DMAIC_12_PRI  : 1;
__REG32 DMAIC_13_PRI  : 1;
__REG32 DMASDC_0_PRI  : 1;
__REG32 DMASDC_1_PRI  : 1;
__REG32 DMASDC_2_PRI  : 1;
__REG32 DMASDC_3_PRI  : 1;
__REG32 DMAADC_2_PRI  : 1;
__REG32 DMAADC_3_PRI  : 1;
__REG32 DMAADC_4_PRI  : 1;
__REG32 DMAADC_5_PRI  : 1;
__REG32 DMAADC_6_PRI  : 1;
__REG32 DMAADC_7_PRI  : 1;
__REG32  DMAPF_0_PRI  : 1;
__REG32  DMAPF_1_PRI  : 1;
__REG32  DMAPF_2_PRI  : 1;
__REG32  DMAPF_3_PRI  : 1;
__REG32  DMAPF_4_PRI  : 1;
__REG32  DMAPF_5_PRI  : 1;
__REG32  DMAPF_6_PRI  : 1;
__REG32  DMAPF_7_PRI  : 1;
} __idmac_cha_pri_bits;

/* IDMAC Channel Busy Register (IDMAC_CHA_BUSY) */
typedef struct{
__REG32  DMAIC_0_BUSY : 1;
__REG32  DMAIC_1_BUSY : 1;
__REG32  DMAIC_2_BUSY : 1;
__REG32  DMAIC_3_BUSY : 1;
__REG32  DMAIC_4_BUSY : 1;
__REG32  DMAIC_5_BUSY : 1;
__REG32  DMAIC_6_BUSY : 1;
__REG32  DMAIC_7_BUSY : 1;
__REG32  DMAIC_8_BUSY : 1;
__REG32  DMAIC_9_BUSY : 1;
__REG32 DMAIC_10_BUSY : 1;
__REG32 DMAIC_11_BUSY : 1;
__REG32 DMAIC_12_BUSY : 1;
__REG32 DMAIC_13_BUSY : 1;
__REG32 DMASDC_0_BUSY : 1;
__REG32 DMASDC_1_BUSY : 1;
__REG32 DMASDC_2_BUSY : 1;
__REG32 DMASDC_3_BUSY : 1;
__REG32 DMAADC_2_BUSY : 1;
__REG32 DMAADC_3_BUSY : 1;
__REG32 DMAADC_4_BUSY : 1;
__REG32 DMAADC_5_BUSY : 1;
__REG32 DMAADC_6_BUSY : 1;
__REG32 DMAADC_7_BUSY : 1;
__REG32  DMAPF_0_BUSY : 1;
__REG32  DMAPF_1_BUSY : 1;
__REG32  DMAPF_2_BUSY : 1;
__REG32  DMAPF_3_BUSY : 1;
__REG32  DMAPF_4_BUSY : 1;
__REG32  DMAPF_5_BUSY : 1;
__REG32  DMAPF_6_BUSY : 1;
__REG32  DMAPF_7_BUSY : 1;
} __idmac_cha_busy_bits;

/* SDC Common Configuration Register (SDC_COM_CONF) */
typedef struct{
__REG32 SDC_MODE          : 2;
__REG32 BG_MCP_FORM       : 1;
__REG32 FG_MCP_FORM       : 1;
__REG32 FG_EN             : 1;
__REG32 GWSEL             : 1;
__REG32 SDC_GLB_LOC_A     : 1;
__REG32 SDC_KEY_COLOR_EN  : 1;
__REG32 MASK_EN           : 1;
__REG32 BG_EN             : 1;
__REG32                   : 2;
__REG32 SHARP             : 1;
__REG32                   : 1;
__REG32 SAVE_REFR_EN      : 1;
__REG32 DUAL_MODE         : 1;
__REG32 COC               : 3;
__REG32                   :13;
} __sdc_com_conf_bits;

/* SDC Graphic Window Control Register (SDC_GRAPH_WIND_CTRL) */
typedef struct{
__REG32 SDC_KEY_COLOR_B   : 8;
__REG32 SDC_KEY_COLOR_G   : 8;
__REG32 SDC_KEY_COLOR_R   : 8;
__REG32 SDC_ALPHA_V       : 8;
} __sdc_graph_wind_ctrl_bits;

/* SDC Foreground Window Position Register (SDC_FG_POS) */
typedef struct{
__REG32 FGYP              :10;
__REG32                   : 6;
__REG32 FGXP              :10;
__REG32                   : 6;
} __sdc_fg_pos_bits;

/* SDC Background Window Position Register (SDC_BG_POS) */
typedef struct{
__REG32 BGYP              :10;
__REG32                   : 6;
__REG32 BGXP              :10;
__REG32                   : 6;
} __sdc_bg_pos_bits;

/* SDC Cursor Position Register (SDC_CUR_POS) */
typedef struct{
__REG32 CYP               :10;
__REG32 CYH               : 5;
__REG32                   : 1;
__REG32 CXP               :10;
__REG32 CXW               : 5;
__REG32                   : 1;
} __sdc_cur_pos_bits;

/* SDC Cursor Blinking and PWM Contrast Control Register (SDC_CUR_BLINK_PWM_CTRL) */
typedef struct{
__REG32 BKDIV             : 8;
__REG32                   : 7;
__REG32 BK_EN             : 1;
__REG32 PWM               : 8;
__REG32 CC_EN             : 1;
__REG32 SCR               : 2;
__REG32                   : 5;
} __sdc_cur_blink_pwm_ctrl_bits;

/* SDC Color Cursor Mapping Register (SDC_CUR_MAP) */
typedef struct{
__REG32 CUR_COL_B         : 8;
__REG32 CUR_COL_G         : 8;
__REG32 CUR_COL_R         : 8;
__REG32                   : 8;
} __sdc_cur_map_bits;

/* SDC Horizontal Configuration Register (SDC_HOR_CONF) */
typedef struct{
__REG32                   :16;
__REG32 SCREEN_WIDTH      :10;
__REG32 H_SYNC_WIDTH      : 6;
} __sdc_hor_conf_bits;

/* SDC Vertical Configuration Register (SDC_VER_CONF) */
typedef struct{
__REG32 V_SYNC_WIDTH_L    : 1;
__REG32                   :15;
__REG32 SCREEN_HEIGHT     :10;
__REG32 V_SYNC_WIDTH      : 6;
} __sdc_ver_conf_bits;

/* SDC Sharp Configuration Register 1 (SDC_SHARP_CONF_1) */
typedef struct{
__REG32 CLS_RISE_DELAY    : 8;
__REG32 PS_FALL_DELAY     : 8;
__REG32 REV_TOGGLE_DELAY  :10;
__REG32                   : 6;
} __sdc_sharp_conf_1_bits;

/* SDC Sharp Configuration Register 2 (SDC_SHARP_CONF_2) */
typedef struct{
__REG32 CLS_FALL_DELAY    :10;
__REG32                   : 6;
__REG32 PS_RISE_DELAY     :10;
__REG32                   : 6;
} __sdc_sharp_conf_2_bits;

/* ADC Configuration Register (ADC_CONF) */
typedef struct{
__REG32 PRP_CHAN_EN       : 1;
__REG32 PP_CHAN_EN        : 1;
__REG32 MCU_CHAN_EN       : 1;
__REG32 PRP_DISP_NUM      : 2;
__REG32 PRP_ADDR_INC      : 2;
__REG32 PRP_DATA_MAP      : 1;
__REG32 PP_DISP_NUM       : 2;
__REG32 PP_ADDR_INC       : 2;
__REG32 PP_DATA_MAP       : 1;
__REG32 PP_NO_TEARING     : 1;
__REG32 SYS1_NO_TEARING   : 1;
__REG32 SYS2_NO_TEARING   : 1;
__REG32 SYS1_MODE         : 3;
__REG32 SYS1_DISP_NUM     : 2;
__REG32 SYS1_ADDR_INC     : 2;
__REG32 SYS1_DATA_MAP     : 1;
__REG32 SYS2_MODE         : 3;
__REG32 SYS2_DISP_NUM     : 2;
__REG32 SYS2_ADDR_INC     : 2;
__REG32 SYS2_DATA_MAP     : 1;
} __adc_conf_bits;

/* ADC System Channel 1-2 Start Address Register (ADC_SYSCHA1_SA) */
typedef struct{
__REG32 SYS_CHAN_SA       :23;
__REG32 SYS_START_TIME    : 9;
} __adc_syscha_sa_bits;

/* ADC Preprocessing Channel Start Address Register (ADC_PRPCHAN_SA) */
typedef struct{
__REG32 PRP_CHAN_SA       :23;
__REG32                   : 9;
} __adc_prpchan_sa_bits;

/* ADC Post-Processing Channel Start Address Register (ADC_PPCHAN_SA) */
typedef struct{
__REG32 PP_CHAN_SA        :23;
__REG32 PP_START_TIME     : 9;
} __adc_ppchan_sa_bits;

/* ADC Display 0,1,2 Configuration Register (ADC_DISPn_CONF) */
typedef struct{
__REG32 DISP_SL               :12;
__REG32 DISP_TYPE             : 2;
__REG32 MCU_DISP_DATA_WIDTH   : 1;
__REG32 MCU_DISP_DATA_MAP     : 1;
__REG32                       :16;
} __adc_disp_conf_bits;

/* ADC Display 0,1,2 Read Acknowledge Pattern Register (ADC_DISPn_RD_AP) */
typedef struct{
__REG32 DISP_ACK_PTRN         :25;
__REG32                       : 7;
} __adc_disp_rd_ap_bits;

/* ADC Display 0,1,2 Read Mask Register (ADC_DISPn_RDM) */
typedef struct{
__REG32 DISP_MASK_ACK_DATA    :24;
__REG32                       : 8;
} __adc_disp_rdm_bits;

/* ADC Display 0,1,2 Screen Size Register (ADC_DISPn_SS) */
typedef struct{
__REG32 SCREEN_WIDTH          :10;
__REG32                       : 6;
__REG32 SCREEN_HEIGHT         :10;
__REG32                       : 6;
} __adc_disp_ss_bits;

/* ADC Displays Vertical Synchronization Register (ADC_DISP_VSYNC) */
typedef struct{
__REG32 DISPL0_VSYNC_MODE     : 2;
__REG32 DISP12_VSYNC_MODE     : 2;
__REG32 DISP12_VSYNC_SEL      : 1;
__REG32                       : 1;
__REG32 DISP_LN_WT            :10;
__REG32 DISP0_VSYNC_WIDTH     : 6;
__REG32 DISP0_VSYNC_WIDTH_L   : 1;
__REG32                       : 1;
__REG32 DISP12_VSYNC_WIDTH    : 6;
__REG32 DISP12_VSYNC_WIDTH_L  : 1;
__REG32                       : 1;
} __adc_disp_vsync_bits;

/* DI Display Interface Configuration Register (DI_DISP_IF_CONF) */
typedef struct{
__REG32 DISP0_EN              : 1;
__REG32 DISP0_IF_MODE         : 2;
__REG32 DISP0_PAR_BURST_MODE  : 2;
__REG32                       : 3;
__REG32 DISP1_EN              : 1;
__REG32 DISP1_IF_MODE         : 3;
__REG32 DISP1_PAR_BURST_MODE  : 2;
__REG32                       : 2;
__REG32 DISP2_EN              : 1;
__REG32 DISP2_IF_MODE         : 3;
__REG32 DISP2_PAR_BURST_MODE  : 2;
__REG32                       : 2;
__REG32 DISP3_DATAMSK         : 1;
__REG32 DISP3_CLK_SEL         : 1;
__REG32 DISP3_CLK_IDLE        : 1;
__REG32 DISP012_DEAD_CLK_NUM  : 4;
__REG32                       : 1;
} __di_disp_if_conf_bits;

/* DI Display Signals Polarity Register (DI_DISP_SIG_POL) */
typedef struct{
__REG32 D0_DATA_POL           : 1;
__REG32 D0_CS_POL             : 1;
__REG32 D0_PAR_RS_POL         : 1;
__REG32 D0_WR_POL             : 1;
__REG32 D0_RD_POL             : 1;
__REG32 D0_VSYNC_POL          : 1;
__REG32 D12_VSYNC_POL         : 1;
__REG32                       : 1;
__REG32 D1_DATA_POL           : 1;
__REG32 D1_CS_POL             : 1;
__REG32 D1_PAR_RS_POL         : 1;
__REG32 D1_WR_POL             : 1;
__REG32 D1_RD_POL             : 1;
__REG32 D1_SD_D_POL           : 1;
__REG32 D1_SD_CLK_POL         : 1;
__REG32 D1_SER_RS_POL         : 1;
__REG32 D2_DATA_POL           : 1;
__REG32 D2_CS_POL             : 1;
__REG32 D2_PAR_RS_POL         : 1;
__REG32 D2_WR_POL             : 1;
__REG32 D2_RD_POL             : 1;
__REG32 D2_SD_D_POL           : 1;
__REG32 D2_SD_CLK_POL         : 1;
__REG32 D2_SER_RS_POL         : 1;
__REG32 D3_DATA_POL           : 1;
__REG32 D3_CLK_POL            : 1;
__REG32 D3_DRDY_SHARP_POL     : 1;
__REG32 D3_HSYNC_POL          : 1;
__REG32 D3_VSYNC_POL          : 1;
__REG32 D0_BCLK_POL           : 1;
__REG32 D1_BCLK_POL           : 1;
__REG32 D2_BCLK_POL           : 1;
} __di_disp_sig_pol_bits;

/* DI Serial Display 1,2 Configuration Register (DI_SER_DISPn_CONF) */
typedef struct{
__REG32 DISP_PREAMBLE_EN      : 1;
__REG32 DISP_RW_CONFIG        : 2;
__REG32                       : 1;
__REG32 DISP_PREAMBLE_LENGTH  : 3;
__REG32                       : 1;
__REG32 DISP_PREAMBLE         : 8;
__REG32 DISP_SER_BIT_NUM      : 5;
__REG32                       : 3;
__REG32 DISP_SER_BURST_MODE   : 1;
__REG32                       : 7;
} __di_ser_disp_conf_bits;

/* DI HSP_CLK Period Register (DI_HSP_CLK_PER) */
typedef struct{
__REG32 HSP_CLK_PERIOD_1      : 7;
__REG32                       : 9;
__REG32 HSP_CLK_PERIOD_2      : 7;
__REG32                       : 9;
} __di_hsp_clk_per_bits;

/* DI Display 0,1,2,3 Time Configuration Register 1 (DI_DISPn_TIME_CONF_1) */
typedef struct{
__REG32 DISP_IF_CLK_PER_WR    :12;
__REG32 DISP_IF_CLK_UP_WR     :10;
__REG32 DISP_IF_CLK_DOWN_WR   :10;
} __di_disp_time_conf_1_bits;

/* DI Display 0,1,2 Time Configuration Register 2 (DI_DISn_TIME_CONF_2) */
typedef struct{
__REG32 DISP_IF_CLK_PER_RD    :12;
__REG32 DISP_IF_CLK_UP_RD     :10;
__REG32 DISP_IF_CLK_DOWN_RD   :10;
} __di_disp_time_conf_2_bits;

/* DI Display 0,1 Time Configuration Register 3 (DI_DISP0_TIME_CONF_3) */
typedef struct{
__REG32 DISP_PIX_CLK_PER      :12;
__REG32                       : 4;
__REG32 DISP_READ_EN          :10;
__REG32                       : 2;
__REG32 DISP_RD_WAIT_ST       : 2;
__REG32                       : 2;
} __di_disp_time_conf_3_bits;

/* DI Display 2 Time Configuration Register 3 (DI_DISP2_TIME_CONF_3) */
typedef struct{
__REG32                       :16;
__REG32 DISP_READ_EN          :10;
__REG32                       : 2;
__REG32 DISP_RD_WAIT_ST       : 2;
__REG32                       : 2;
} __di_disp2_time_conf_3_bits;

/* DI Display 0,1,2 Data Byte 0,1,2 Mapping Register (DI_DISPn_DBm_MAP) */
typedef struct{
__REG32 M0     : 2;
__REG32 M1     : 2;
__REG32 M2     : 2;
__REG32 M3     : 2;
__REG32 M4     : 2;
__REG32 M5     : 2;
__REG32 M6     : 2;
__REG32 M7     : 2;
__REG32 OFFS0  : 5;
__REG32 OFFS1  : 5;
__REG32 OFFS2  : 5;
__REG32        : 1;
} __di_disp_map_bits;

/* DI Display Access Cycles Count Register (DI_DISP_ACC_CC) */
typedef struct{
__REG32 DISP0_IF_CLK_CNT_D  : 2;
__REG32 DISP0_IF_CLK_CNT_C  : 2;
__REG32 DISP1_IF_CLK_CNT_D  : 2;
__REG32 DISP1_IF_CLK_CNT_C  : 2;
__REG32 DISP2_IF_CLK_CNT_D  : 2;
__REG32 DISP2_IF_CLK_CNT_C  : 2;
__REG32 DISP3_IF_CLK_CNT_D  : 2;
__REG32                     :18;
} __di_disp_acc_cc_bits;

/* DI Display Low Level Access Configuration Register (DI_DISP_LLA_CONF) */
typedef struct{
__REG32 DRCT_RS             : 1;
__REG32 DRCT_DISP_NUM       : 2;
__REG32 DRCT_LOCK           : 1;
__REG32 DRCT_MAP_DC         : 1;
__REG32 DRCT_BE_MODE        : 1;
__REG32                     :26;
} __di_disp_lla_conf_bits;

/* DI Display Low Level Access Data Register (DI_DISP_LLA_DATA) */
typedef struct{
__REG32 LLA_DATA            :24;
__REG32                     : 8;
} __di_disp_lla_data_bits;

/* SSI Control Register (SCR) */
typedef struct{
__REG32 SSIEN               : 1;
__REG32 TE                  : 1;
__REG32 RE                  : 1;
__REG32 NET                 : 1;
__REG32 SYN                 : 1;
__REG32 I2S_MODE            : 2;
__REG32 SYS_CLK_EN          : 1;
__REG32 TCH_EN              : 1;
__REG32 CLK_IST             : 1;
__REG32                     :22;
} __scr_bits;

/* SSI Interrupt Status Register (SISR) */
typedef struct{
__REG32 TFE0                : 1;
__REG32 TFE1                : 1;
__REG32 RFF0                : 1;
__REG32 RFF1                : 1;
__REG32 RLS                 : 1;
__REG32 TLS                 : 1;
__REG32 RFS                 : 1;
__REG32 TFS                 : 1;
__REG32 TUE0                : 1;
__REG32 TUE1                : 1;
__REG32 ROE0                : 1;
__REG32 ROE1                : 1;
__REG32 TDE0                : 1;
__REG32 TDE1                : 1;
__REG32 RDR0                : 1;
__REG32 RDR1                : 1;
__REG32 RXT                 : 1;
__REG32 CMDDU               : 1;
__REG32 CMDAU               : 1;
__REG32                     :13;
} __sisr_bits;

/* SSI Interrupt Enable Register (SIER) */
typedef struct{
__REG32 TFE0_EN             : 1;
__REG32 TFE1_EN             : 1;
__REG32 RFF0_EN             : 1;
__REG32 RFF1_EN             : 1;
__REG32 RLS_EN              : 1;
__REG32 TLS_EN              : 1;
__REG32 RFS_EN              : 1;
__REG32 TFS_EN              : 1;
__REG32 TUE0_EN             : 1;
__REG32 TUE1_EN             : 1;
__REG32 ROE0_EN             : 1;
__REG32 ROE1_EN             : 1;
__REG32 TDE0_EN             : 1;
__REG32 TDE1_EN             : 1;
__REG32 RDR0_EN             : 1;
__REG32 RDR1_EN             : 1;
__REG32 RXT_EN              : 1;
__REG32 CMDDU_EN            : 1;
__REG32 CMDAU_EN            : 1;
__REG32 TIE                 : 1;
__REG32 TDMAE               : 1;
__REG32 RIE                 : 1;
__REG32 RDMAE               : 1;
__REG32                     : 9;
} __sier_bits;

/* SSI Transmit Configuration Register (STCR) */
typedef struct{
__REG32 TEFS                : 1;
__REG32 TFSL                : 1;
__REG32 TFSI                : 1;
__REG32 TSCKP               : 1;
__REG32 TSHFD               : 1;
__REG32 TXDIR               : 1;
__REG32 TFDIR               : 1;
__REG32 TFEN0               : 1;
__REG32 TFEN1               : 1;
__REG32 TXBIT0              : 1;
__REG32                     :22;
} __stcr_bits;

/* SSI Receive Configuration Register (SRCR) */
typedef struct{
__REG32 REFS                : 1;
__REG32 RFSL                : 1;
__REG32 RFSI                : 1;
__REG32 RSCKP               : 1;
__REG32 RSHFD               : 1;
__REG32 RXDIR               : 1;
__REG32 RFDIR               : 1;
__REG32 RFEN0               : 1;
__REG32 RFEN1               : 1;
__REG32 RXBIT0              : 1;
__REG32 RXEXT               : 1;
__REG32                     :21;
} __srcr_bits;

/* SSI Transmit and Receive Clock Control Registers (STCCR and SRCCR) */
typedef struct{
__REG32 PM0                 : 1;
__REG32 PM1                 : 1;
__REG32 PM2                 : 1;
__REG32 PM3                 : 1;
__REG32 PM4                 : 1;
__REG32 PM5                 : 1;
__REG32 PM6                 : 1;
__REG32 PM7                 : 1;
__REG32 DC0                 : 1;
__REG32 DC1                 : 1;
__REG32 DC2                 : 1;
__REG32 DC3                 : 1;
__REG32 DC4                 : 1;
__REG32 WL0                 : 1;
__REG32 WL1                 : 1;
__REG32 WL2                 : 1;
__REG32 WL3                 : 1;
__REG32 PSR                 : 1;
__REG32 DIV2                : 1;
__REG32                     :13;
} __stccr_bits;

/* SSI FIFO Control/Status Register (SFCSR) */
typedef struct{
__REG32 TFWM0               : 4;
__REG32 RFWM0               : 4;
__REG32 TFCNT0              : 4;
__REG32 RFCNT0              : 4;
__REG32 TFWM1               : 4;
__REG32 RFWM1               : 4;
__REG32 TFCNT1              : 4;
__REG32 RFCNT1              : 4;
} __sfcsr_bits;

/* SSI AC97 Control Register (SACNT) */
typedef struct{
__REG32 AC97EN              : 1;
__REG32 FV                  : 1;
__REG32 TIF                 : 1;
__REG32 RD                  : 1;
__REG32 WR                  : 1;
__REG32 FRDIV               : 6;
__REG32                     :21;
} __sacnt_bits;

/* SSI AC97 Command Address Register (SACADD) */
typedef struct{
__REG32 SACADD              :19;
__REG32                     :13;
} __sacadd_bits;

/* SSI AC97 Command Data Register (SACDAT) */
typedef struct{
__REG32 SACADD              :20;
__REG32                     :12;
} __sacdat_bits;

/* SSI Transmit Time Slot Mask Register (STMSK) */
typedef struct{
__REG32 STMSK0              : 1;
__REG32 STMSK1              : 1;
__REG32 STMSK2              : 1;
__REG32 STMSK3              : 1;
__REG32 STMSK4              : 1;
__REG32 STMSK5              : 1;
__REG32 STMSK6              : 1;
__REG32 STMSK7              : 1;
__REG32 STMSK8              : 1;
__REG32 STMSK9              : 1;
__REG32 STMSK10             : 1;
__REG32 STMSK11             : 1;
__REG32 STMSK12             : 1;
__REG32 STMSK13             : 1;
__REG32 STMSK14             : 1;
__REG32 STMSK15             : 1;
__REG32 STMSK16             : 1;
__REG32 STMSK17             : 1;
__REG32 STMSK18             : 1;
__REG32 STMSK19             : 1;
__REG32 STMSK20             : 1;
__REG32 STMSK21             : 1;
__REG32 STMSK22             : 1;
__REG32 STMSK23             : 1;
__REG32 STMSK24             : 1;
__REG32 STMSK25             : 1;
__REG32 STMSK26             : 1;
__REG32 STMSK27             : 1;
__REG32 STMSK28             : 1;
__REG32 STMSK29             : 1;
__REG32 STMSK30             : 1;
__REG32 STMSK31             : 1;
} __stmsk_bits;

/* SSI Receive Time Slot Mask Register (SRMSK) */
typedef struct{
__REG32 SRMSK0              : 1;
__REG32 SRMSK1              : 1;
__REG32 SRMSK2              : 1;
__REG32 SRMSK3              : 1;
__REG32 SRMSK4              : 1;
__REG32 SRMSK5              : 1;
__REG32 SRMSK6              : 1;
__REG32 SRMSK7              : 1;
__REG32 SRMSK8              : 1;
__REG32 SRMSK9              : 1;
__REG32 SRMSK10             : 1;
__REG32 SRMSK11             : 1;
__REG32 SRMSK12             : 1;
__REG32 SRMSK13             : 1;
__REG32 SRMSK14             : 1;
__REG32 SRMSK15             : 1;
__REG32 SRMSK16             : 1;
__REG32 SRMSK17             : 1;
__REG32 SRMSK18             : 1;
__REG32 SRMSK19             : 1;
__REG32 SRMSK20             : 1;
__REG32 SRMSK21             : 1;
__REG32 SRMSK22             : 1;
__REG32 SRMSK23             : 1;
__REG32 SRMSK24             : 1;
__REG32 SRMSK25             : 1;
__REG32 SRMSK26             : 1;
__REG32 SRMSK27             : 1;
__REG32 SRMSK28             : 1;
__REG32 SRMSK29             : 1;
__REG32 SRMSK30             : 1;
__REG32 SRMSK31             : 1;
} __srmsk_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  CCM
 **
 ***************************************************************************/
__IO_REG32_BIT(CCM_CCMR,                  0x53F80000,__READ_WRITE ,__ccm_ccmr_bits);
__IO_REG32_BIT(CCM_PDR0,                  0x53F80004,__READ_WRITE ,__ccm_pdr0_bits);
__IO_REG32_BIT(CCM_PDR1,                  0x53F80008,__READ_WRITE ,__ccm_pdr1_bits);
__IO_REG32_BIT(CCM_RCSR,                  0x53F8000C,__READ_WRITE ,__ccm_rcsr_bits);
__IO_REG32_BIT(CCM_MPCTL,                 0x53F80010,__READ_WRITE ,__ccm_mpctl_bits);
__IO_REG32_BIT(CCM_UPCTL,                 0x53F80014,__READ_WRITE ,__ccm_mpctl_bits);
__IO_REG32_BIT(CCM_SPCTL,                 0x53F80018,__READ_WRITE ,__ccm_mpctl_bits);
__IO_REG32_BIT(CCM_COSR,                  0x53F8001C,__READ_WRITE ,__ccm_cosr_bits);
__IO_REG32_BIT(CCM_CGR0,                  0x53F80020,__READ_WRITE ,__ccm_cgr_bits);
__IO_REG32_BIT(CCM_CGR1,                  0x53F80024,__READ_WRITE ,__ccm_cgr_bits);
__IO_REG32_BIT(CCM_CGR2,                  0x53F80028,__READ_WRITE ,__ccm_cgr2_bits);
__IO_REG32_BIT(CCM_WIMR0,                 0x53F8002C,__READ_WRITE ,__ccm_wimr0_bits);
__IO_REG32(    CCM_LDC,                   0x53F80030,__READ_WRITE );
__IO_REG32_BIT(CCM_DCVR0,                 0x53F80034,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR1,                 0x53F80038,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR2,                 0x53F8003C,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR3,                 0x53F80040,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_LTR0,                  0x53F80044,__READ_WRITE ,__ccm_ltr0_bits);
__IO_REG32_BIT(CCM_LTR1,                  0x53F80048,__READ_WRITE ,__ccm_ltr1_bits);
__IO_REG32_BIT(CCM_LTR2,                  0x53F8004C,__READ_WRITE ,__ccm_ltr2_bits);
__IO_REG32_BIT(CCM_LTR3,                  0x53F80050,__READ_WRITE ,__ccm_ltr3_bits);
__IO_REG32_BIT(CCM_LTBR0,                 0x53F80054,__READ       ,__ccm_ltbr0_bits);
__IO_REG32_BIT(CCM_LTBR1,                 0x53F80058,__READ       ,__ccm_ltbr1_bits);
__IO_REG32_BIT(CCM_PMCR0,                 0x53F8005C,__READ_WRITE ,__ccm_pmcr0_bits);
__IO_REG32_BIT(CCM_PMCR1,                 0x53F80060,__READ_WRITE ,__ccm_pmcr1_bits);
__IO_REG32_BIT(CCM_PDR2,                  0x53F80064,__READ_WRITE ,__ccm_pdr2_bits);

/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
__IO_REG32(    IOMUX_INT_OBS1,            0x43FAC000,__READ_WRITE );
__IO_REG32(    IOMUX_INT_OBS2,            0x43FAC004,__READ_WRITE );
__IO_REG32_BIT(IOMUX_GPR,                 0x43FAC008,__READ_WRITE ,__iomux_gpr_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL0,         0x43FAC00C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL1,         0x43FAC010,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL2,         0x43FAC014,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL3,         0x43FAC018,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL4,         0x43FAC01C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL5,         0x43FAC020,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL6,         0x43FAC024,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL7,         0x43FAC028,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL8,         0x43FAC02C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL9,         0x43FAC030,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL10,        0x43FAC034,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL11,        0x43FAC038,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL12,        0x43FAC03C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL13,        0x43FAC040,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL14,        0x43FAC044,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL15,        0x43FAC048,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL16,        0x43FAC04C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL17,        0x43FAC050,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL18,        0x43FAC054,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL19,        0x43FAC058,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL20,        0x43FAC05C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL21,        0x43FAC060,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL22,        0x43FAC064,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL23,        0x43FAC068,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL24,        0x43FAC06C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL25,        0x43FAC070,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL26,        0x43FAC074,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL27,        0x43FAC078,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL28,        0x43FAC07C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL29,        0x43FAC080,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL30,        0x43FAC084,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL31,        0x43FAC088,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL32,        0x43FAC08C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL33,        0x43FAC090,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL34,        0x43FAC094,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL35,        0x43FAC098,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL36,        0x43FAC09C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL37,        0x43FAC0A0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL38,        0x43FAC0A4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL39,        0x43FAC0A8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL40,        0x43FAC0AC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL41,        0x43FAC0B0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL42,        0x43FAC0B4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL43,        0x43FAC0B8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL44,        0x43FAC0BC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL45,        0x43FAC0C0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL46,        0x43FAC0C4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL47,        0x43FAC0C8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL48,        0x43FAC0CC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL49,        0x43FAC0D0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL50,        0x43FAC0D4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL51,        0x43FAC0D8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL52,        0x43FAC0DC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL53,        0x43FAC0E0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL54,        0x43FAC0E4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL55,        0x43FAC0E8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL56,        0x43FAC0EC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL57,        0x43FAC0F0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL58,        0x43FAC0F4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL59,        0x43FAC0F8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL60,        0x43FAC0FC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL61,        0x43FAC100,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL62,        0x43FAC104,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL63,        0x43FAC108,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL64,        0x43FAC10C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL65,        0x43FAC110,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL66,        0x43FAC114,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL67,        0x43FAC118,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL68,        0x43FAC11C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL69,        0x43FAC120,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL70,        0x43FAC124,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL71,        0x43FAC128,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL72,        0x43FAC12C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL73,        0x43FAC130,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL74,        0x43FAC134,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL75,        0x43FAC138,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL76,        0x43FAC13C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL77,        0x43FAC140,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL78,        0x43FAC144,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL79,        0x43FAC148,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL80,        0x43FAC14C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_MUX_CTL81,        0x43FAC150,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL0,         0x43FAC154,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL1,         0x43FAC158,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL2,         0x43FAC15C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL3,         0x43FAC160,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL4,         0x43FAC164,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL5,         0x43FAC168,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL6,         0x43FAC16C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL7,         0x43FAC170,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL8,         0x43FAC174,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL9,         0x43FAC178,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL10,        0x43FAC17C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL11,        0x43FAC180,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL12,        0x43FAC184,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL13,        0x43FAC188,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL14,        0x43FAC18C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL15,        0x43FAC190,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL16,        0x43FAC194,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL17,        0x43FAC198,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL18,        0x43FAC19C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL19,        0x43FAC1A0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL20,        0x43FAC1A4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL21,        0x43FAC1A8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL22,        0x43FAC1AC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL23,        0x43FAC1B0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL24,        0x43FAC1B4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL25,        0x43FAC1B8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL26,        0x43FAC1BC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL27,        0x43FAC1C0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL28,        0x43FAC1C4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL29,        0x43FAC1C8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL30,        0x43FAC1CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL32,        0x43FAC1D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL33,        0x43FAC1D4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL34,        0x43FAC1D8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL35,        0x43FAC1DC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL36,        0x43FAC1E0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL37,        0x43FAC1E4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL38,        0x43FAC1E8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL39,        0x43FAC1EC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL40,        0x43FAC1F0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL41,        0x43FAC1F4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL42,        0x43FAC1F8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL43,        0x43FAC1FC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL44,        0x43FAC200,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL45,        0x43FAC204,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL46,        0x43FAC208,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL47,        0x43FAC20C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL48,        0x43FAC210,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL49,        0x43FAC214,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL50,        0x43FAC218,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL51,        0x43FAC21C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL52,        0x43FAC220,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL53,        0x43FAC224,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL54,        0x43FAC228,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL55,        0x43FAC22C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL56,        0x43FAC230,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL57,        0x43FAC234,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL58,        0x43FAC238,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL59,        0x43FAC23C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL60,        0x43FAC240,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL61,        0x43FAC244,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL62,        0x43FAC248,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL63,        0x43FAC24C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL64,        0x43FAC250,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL65,        0x43FAC254,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL66,        0x43FAC258,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL67,        0x43FAC25C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL68,        0x43FAC260,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL69,        0x43FAC264,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL70,        0x43FAC268,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL71,        0x43FAC26C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL72,        0x43FAC270,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL73,        0x43FAC274,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL74,        0x43FAC278,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL75,        0x43FAC27C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL76,        0x43FAC280,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL77,        0x43FAC284,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL78,        0x43FAC288,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL79,        0x43FAC28C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL80,        0x43FAC290,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL81,        0x43FAC294,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL82,        0x43FAC298,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL83,        0x43FAC29C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL84,        0x43FAC2A0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL85,        0x43FAC2A4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL86,        0x43FAC2A8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL87,        0x43FAC2AC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL88,        0x43FAC2B0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL89,        0x43FAC2B4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL90,        0x43FAC2B8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL91,        0x43FAC2BC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL92,        0x43FAC2C0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL93,        0x43FAC2C4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL94,        0x43FAC2C8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL95,        0x43FAC2CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL96,        0x43FAC2D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL97,        0x43FAC2D4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL98,        0x43FAC2D8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL99,        0x43FAC2DC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL100,       0x43FAC2E0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL101,       0x43FAC2E4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL102,       0x43FAC2E8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL103,       0x43FAC2EC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL104,       0x43FAC2F0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL105,       0x43FAC2F4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL106,       0x43FAC2F8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL107,       0x43FAC2FC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL108,       0x43FAC300,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL109,       0x43FAC304,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUX_SW_PAD_CTL110,       0x43FAC308,__READ_WRITE ,__iomux_sw_pad_ctl_bits);

/***************************************************************************
 **
 **  GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_DR,                  0x53FCC000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_GDIR,                0x53FCC004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_PSR,                 0x53FCC008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO1_ICR1,                0x53FCC00C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO1_ICR2,                0x53FCC010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO1_IMR,                 0x53FCC014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_ISR,                 0x53FCC018,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_DR,                  0x53FD0000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_GDIR,                0x53FD0004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_PSR,                 0x53FD0008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO2_ICR1,                0x53FD000C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO2_ICR2,                0x53FD0010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO2_IMR,                 0x53FD0014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_ISR,                 0x53FD0018,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_DR,                  0x53FA4000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_GDIR,                0x53FA4004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_PSR,                 0x53FA4008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO3_ICR1,                0x53FA400C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO3_ICR2,                0x53FA4010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO3_IMR,                 0x53FA4014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_ISR,                 0x53FA4018,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  AVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(INTCNTL,              0x68000000,__READ_WRITE ,__intcntl_bits);
__IO_REG32_BIT(NIMASK,               0x68000004,__READ_WRITE ,__nimask_bits);
__IO_REG32_BIT(INTENNUM,             0x68000008,__READ_WRITE ,__intennum_bits);
__IO_REG32_BIT(INTDISNUM,            0x6800000C,__READ_WRITE ,__intdisnum_bits);
__IO_REG32_BIT(INTENABLEH,           0x68000010,__READ_WRITE ,__intenableh_bits);
__IO_REG32_BIT(INTENABLEL,           0x68000014,__READ_WRITE ,__intenablel_bits);
__IO_REG32_BIT(INTTYPEH,             0x68000018,__READ_WRITE ,__inttypeh_bits);
__IO_REG32_BIT(INTTYPEL,             0x6800001C,__READ_WRITE ,__inttypel_bits);
__IO_REG32_BIT(NIPRIORITY7,          0x68000020,__READ_WRITE ,__nipriority7_bits);
__IO_REG32_BIT(NIPRIORITY6,          0x68000024,__READ_WRITE ,__nipriority6_bits);
__IO_REG32_BIT(NIPRIORITY5,          0x68000028,__READ_WRITE ,__nipriority5_bits);
__IO_REG32_BIT(NIPRIORITY4,          0x6800002C,__READ_WRITE ,__nipriority4_bits);
__IO_REG32_BIT(NIPRIORITY3,          0x68000030,__READ_WRITE ,__nipriority3_bits);
__IO_REG32_BIT(NIPRIORITY2,          0x68000034,__READ_WRITE ,__nipriority2_bits);
__IO_REG32_BIT(NIPRIORITY1,          0x68000038,__READ_WRITE ,__nipriority1_bits);
__IO_REG32_BIT(NIPRIORITY0,          0x6800003C,__READ_WRITE ,__nipriority0_bits);
__IO_REG32_BIT(NIVECSR,              0x68000040,__READ       ,__nivecsr_bits);
__IO_REG32(    FIVECSR,              0x68000044,__READ       );
__IO_REG32_BIT(INTSRCH,              0x68000048,__READ       ,__intsrch_bits);
__IO_REG32_BIT(INTSRCL,              0x6800004C,__READ       ,__intsrcl_bits);
__IO_REG32_BIT(INTFRCH,              0x68000050,__READ_WRITE ,__intfrch_bits);
__IO_REG32_BIT(INTFRCL,              0x68000054,__READ_WRITE ,__intfrcl_bits);
__IO_REG32_BIT(NIPNDH,               0x68000058,__READ       ,__nipndh_bits);
__IO_REG32_BIT(NIPNDL,               0x6800005C,__READ       ,__nipndl_bits);
__IO_REG32_BIT(FIPNDH,               0x68000060,__READ       ,__fipndh_bits);
__IO_REG32_BIT(FIPNDL,               0x68000064,__READ       ,__fipndl_bits);
__IO_REG32(    VECTOR0,              0x68000100,__READ_WRITE );
__IO_REG32(    VECTOR1,              0x68000104,__READ_WRITE );
__IO_REG32(    VECTOR2,              0x68000108,__READ_WRITE );
__IO_REG32(    VECTOR3,              0x6800010C,__READ_WRITE );
__IO_REG32(    VECTOR4,              0x68000110,__READ_WRITE );
__IO_REG32(    VECTOR5,              0x68000114,__READ_WRITE );
__IO_REG32(    VECTOR6,              0x68000118,__READ_WRITE );
__IO_REG32(    VECTOR7,              0x6800011C,__READ_WRITE );
__IO_REG32(    VECTOR8,              0x68000120,__READ_WRITE );
__IO_REG32(    VECTOR9,              0x68000124,__READ_WRITE );
__IO_REG32(    VECTOR10,             0x68000128,__READ_WRITE );
__IO_REG32(    VECTOR11,             0x6800012C,__READ_WRITE );
__IO_REG32(    VECTOR12,             0x68000130,__READ_WRITE );
__IO_REG32(    VECTOR13,             0x68000134,__READ_WRITE );
__IO_REG32(    VECTOR14,             0x68000138,__READ_WRITE );
__IO_REG32(    VECTOR15,             0x6800013C,__READ_WRITE );
__IO_REG32(    VECTOR16,             0x68000140,__READ_WRITE );
__IO_REG32(    VECTOR17,             0x68000144,__READ_WRITE );
__IO_REG32(    VECTOR18,             0x68000148,__READ_WRITE );
__IO_REG32(    VECTOR19,             0x6800014C,__READ_WRITE );
__IO_REG32(    VECTOR20,             0x68000150,__READ_WRITE );
__IO_REG32(    VECTOR21,             0x68000154,__READ_WRITE );
__IO_REG32(    VECTOR22,             0x68000158,__READ_WRITE );
__IO_REG32(    VECTOR23,             0x6800015C,__READ_WRITE );
__IO_REG32(    VECTOR24,             0x68000160,__READ_WRITE );
__IO_REG32(    VECTOR25,             0x68000164,__READ_WRITE );
__IO_REG32(    VECTOR26,             0x68000168,__READ_WRITE );
__IO_REG32(    VECTOR27,             0x6800016C,__READ_WRITE );
__IO_REG32(    VECTOR28,             0x68000170,__READ_WRITE );
__IO_REG32(    VECTOR29,             0x68000174,__READ_WRITE );
__IO_REG32(    VECTOR30,             0x68000178,__READ_WRITE );
__IO_REG32(    VECTOR31,             0x6800017C,__READ_WRITE );
__IO_REG32(    VECTOR32,             0x68000180,__READ_WRITE );
__IO_REG32(    VECTOR33,             0x68000184,__READ_WRITE );
__IO_REG32(    VECTOR34,             0x68000188,__READ_WRITE );
__IO_REG32(    VECTOR35,             0x6800018C,__READ_WRITE );
__IO_REG32(    VECTOR36,             0x68000190,__READ_WRITE );
__IO_REG32(    VECTOR37,             0x68000194,__READ_WRITE );
__IO_REG32(    VECTOR38,             0x68000198,__READ_WRITE );
__IO_REG32(    VECTOR39,             0x6800019C,__READ_WRITE );
__IO_REG32(    VECTOR40,             0x680001A0,__READ_WRITE );
__IO_REG32(    VECTOR41,             0x680001A4,__READ_WRITE );
__IO_REG32(    VECTOR42,             0x680001A8,__READ_WRITE );
__IO_REG32(    VECTOR43,             0x680001AC,__READ_WRITE );
__IO_REG32(    VECTOR44,             0x680001B0,__READ_WRITE );
__IO_REG32(    VECTOR45,             0x680001B4,__READ_WRITE );
__IO_REG32(    VECTOR46,             0x680001B8,__READ_WRITE );
__IO_REG32(    VECTOR47,             0x680001BC,__READ_WRITE );
__IO_REG32(    VECTOR48,             0x680001C0,__READ_WRITE );
__IO_REG32(    VECTOR49,             0x680001C4,__READ_WRITE );
__IO_REG32(    VECTOR50,             0x680001C8,__READ_WRITE );
__IO_REG32(    VECTOR51,             0x680001CC,__READ_WRITE );
__IO_REG32(    VECTOR52,             0x680001D0,__READ_WRITE );
__IO_REG32(    VECTOR53,             0x680001D4,__READ_WRITE );
__IO_REG32(    VECTOR54,             0x680001D8,__READ_WRITE );
__IO_REG32(    VECTOR55,             0x680001DC,__READ_WRITE );
__IO_REG32(    VECTOR56,             0x680001E0,__READ_WRITE );
__IO_REG32(    VECTOR57,             0x680001E4,__READ_WRITE );
__IO_REG32(    VECTOR58,             0x680001E8,__READ_WRITE );
__IO_REG32(    VECTOR59,             0x680001EC,__READ_WRITE );
__IO_REG32(    VECTOR60,             0x680001F0,__READ_WRITE );
__IO_REG32(    VECTOR61,             0x680001F4,__READ_WRITE );
__IO_REG32(    VECTOR62,             0x680001F8,__READ_WRITE );
__IO_REG32(    VECTOR63,             0x680001FC,__READ_WRITE );

/***************************************************************************
 **
 **  SCC
 **
 ***************************************************************************/
__IO_REG32_BIT(SCC_RSR,                   0x53FAC000,__READ_WRITE ,__scc_rsr_bits);
__IO_REG32_BIT(SCC_BSR,                   0x53FAC004,__READ_WRITE ,__scc_bsr_bits);
__IO_REG32_BIT(SCC_LENR,                  0x53FAC008,__READ_WRITE ,__scc_lenr_bits);
__IO_REG32_BIT(SCC_CTLR,                  0x53FAC00C,__READ_WRITE ,__scc_ctlr_bits);
__IO_REG32_BIT(SCC_SCMR,                  0x53FAC010,__READ       ,__scc_scmr_bits);
__IO_REG32_BIT(SCC_ERRSR,                 0x53FAC014,__READ_WRITE ,__scc_errsr_bits);
__IO_REG32_BIT(SCC_INTCR,                 0x53FAC018,__READ_WRITE ,__scc_intcr_bits);
__IO_REG32_BIT(SCC_CONFIGR,               0x53FAC01C,__READ       ,__scc_configr_bits);
__IO_REG32(    SCC_INITVR0,               0x53FAC020,__READ_WRITE );
__IO_REG32(    SCC_INITVR1,               0x53FAC024,__READ_WRITE );
__IO_REG32(    SCC_RMEMR_BASE,            0x53FAC400,__READ_WRITE );
__IO_REG32(    SCC_BMEMR_BASE,            0x53FAC800,__READ_WRITE );
__IO_REG32_BIT(SCC_SMNSR,                 0x53FAD000,__READ_WRITE ,__scc_smnsr_bits);
__IO_REG32_BIT(SCC_CMDR,                  0x53FAD004,__READ_WRITE ,__scc_cmdr_bits);
__IO_REG32_BIT(SCC_SEQSR,                 0x53FAD008,__READ_WRITE ,__scc_seqsr_bits);
__IO_REG32_BIT(SCC_SEQER,                 0x53FAD00C,__READ_WRITE ,__scc_seqer_bits);
__IO_REG32_BIT(SCC_SEQCR,                 0x53FAD010,__READ_WRITE ,__scc_seqcr_bits);
__IO_REG32_BIT(SCC_BCNTR,                 0x53FAD014,__READ       ,__scc_bcntr_bits);
__IO_REG32_BIT(SCC_BISR,                  0x53FAD018,__READ_WRITE ,__scc_bisr_bits);
__IO_REG32_BIT(SCC_BBDR,                  0x53FAD01C,__WRITE      ,__scc_bbdr_bits);
__IO_REG32_BIT(SCC_SIZER,                 0x53FAD020,__READ_WRITE ,__scc_sizer_bits);
__IO_REG32(    SCC_PLAINR,                0x53FAD024,__READ_WRITE );
__IO_REG32(    SCC_CIPHER,                0x53FAD028,__READ_WRITE );
__IO_REG32(    SCC_TIMEIVR,               0x53FAD02C,__READ_WRITE );
__IO_REG32_BIT(SCC_TIMECTR,               0x53FAD030,__READ_WRITE ,__scc_timectr_bits);
__IO_REG32_BIT(SCC_DEBUGR,                0x53FAD034,__READ_WRITE ,__scc_debugr_bits);
__IO_REG32(    SCC_TIMERR,                0x53FAD038,__READ       );

/***************************************************************************
 **
 **  RNGA
 **
 ***************************************************************************/
__IO_REG32_BIT(RNGCR,                     0x53FB0000,__READ_WRITE ,__rngcr_bits);
__IO_REG32_BIT(RNGSR,                     0x53FB0004,__READ       ,__rngsr_bits);
__IO_REG32(    RNGENT,                    0x53FB0008,__WRITE      );
__IO_REG32(    RNGOFIFO,                  0x53FB000C,__READ       );
__IO_REG32_BIT(RNGMOD,                    0x53FB0010,__READ_WRITE ,__rngmod_bits);
__IO_REG32_BIT(RNGVER,                    0x53FB0014,__READ_WRITE ,__rngver_bits);
__IO_REG32_BIT(RNGOSCCR,                  0x53FB0018,__READ_WRITE ,__rngosccr_bits);
__IO_REG32_BIT(RNGOSC1CT,                 0x53FB001C,__READ       ,__rngoscct_bits);
__IO_REG32_BIT(RNGOSC2CT,                 0x53FB0020,__READ       ,__rngoscct_bits);
__IO_REG32_BIT(RNGOSCSTAT,                0x53FB0024,__READ       ,__rngoscstat_bits);

/***************************************************************************
 **
 **  RTIC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTIC_STATUS,               0x53FEC000,__READ       ,__rtic_status_bits);
__IO_REG32_BIT(RTIC_COMMAND,              0x53FEC004,__READ_WRITE ,__rtic_command_bits);
__IO_REG32_BIT(RTIC_CTRL,                 0x53FEC008,__READ_WRITE ,__rtic_ctrl_bits);
__IO_REG32(		 RTIC_DMATHROTTLE,          0x53FEC00C,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKA1,           0x53FEC010,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENA1,             0x53FEC014,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKA2,           0x53FEC018,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENA2,             0x53FEC01C,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKB1,           0x53FEC030,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENB1,             0x53FEC034,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKB2,           0x53FEC038,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENB2,             0x53FEC03C,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKC1,           0x53FEC050,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENC1,             0x53FEC054,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKC2,           0x53FEC058,__READ_WRITE );
__IO_REG32(    RTIC_MEMLENC2,             0x53FEC05C,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKD1,           0x53FEC070,__READ_WRITE );
__IO_REG32(    RTIC_MEMLEND1,             0x53FEC074,__READ_WRITE );
__IO_REG32(    RTIC_MEMBLOCKD2,           0x53FEC078,__READ_WRITE );
__IO_REG32(    RTIC_MEMLEND2,             0x53FEC07C,__READ_WRITE );
__IO_REG32(    RTIC_FAULT,                0x53FEC090,__READ       );
__IO_REG32_BIT(RTIC_WDOG,                 0x53FEC094,__READ_WRITE ,__rtic_wdog_bits);
__IO_REG32(    RTIC_HASHA1,               0x53FEC0A0,__READ       );
__IO_REG32(    RTIC_HASHA2,               0x53FEC0A4,__READ       );
__IO_REG32(    RTIC_HASHA3,               0x53FEC0A8,__READ       );
__IO_REG32(    RTIC_HASHA4,               0x53FEC0AC,__READ       );
__IO_REG32(    RTIC_HASHA5,               0x53FEC0B0,__READ       );
__IO_REG32(    RTIC_HASHB1,               0x53FEC0C0,__READ       );
__IO_REG32(    RTIC_HASHB2,               0x53FEC0C4,__READ       );
__IO_REG32(    RTIC_HASHB3,               0x53FEC0C8,__READ       );
__IO_REG32(    RTIC_HASHB4,               0x53FEC0CC,__READ       );
__IO_REG32(    RTIC_HASHB5,               0x53FEC0D0,__READ       );
__IO_REG32(    RTIC_HASHC1,               0x53FEC0E0,__READ       );
__IO_REG32(    RTIC_HASHC2,               0x53FEC0E4,__READ       );
__IO_REG32(    RTIC_HASHC3,               0x53FEC0E8,__READ       );
__IO_REG32(    RTIC_HASHC4,               0x53FEC0EC,__READ       );
__IO_REG32(    RTIC_HASHC5,               0x53FEC0F0,__READ       );
__IO_REG32(    RTIC_HASHD1,               0x53FEC100,__READ       );
__IO_REG32(    RTIC_HASHD2,               0x53FEC104,__READ       );
__IO_REG32(    RTIC_HASHD3,               0x53FEC108,__READ       );
__IO_REG32(    RTIC_HASHD4,               0x53FEC10C,__READ       );
__IO_REG32(    RTIC_HASHD5,               0x53FEC110,__READ       );

/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT( IMM_STAT,                  0x5001C000,__READ_WRITE ,__iim_stat_bits);
__IO_REG8_BIT( IMM_STATM,                 0x5001C004,__READ_WRITE ,__iim_statm_bits);
__IO_REG8_BIT( IMM_ERR,                   0x5001C008,__READ_WRITE ,__iim_err_bits);
__IO_REG8_BIT( IMM_EMASK,                 0x5001C00C,__READ_WRITE ,__iim_emask_bits);
__IO_REG8_BIT( IMM_FCTL,                  0x5001C010,__READ_WRITE ,__iim_fctl_bits);
__IO_REG8_BIT( IMM_UA,                    0x5001C014,__READ_WRITE ,__iim_ua_bits);
__IO_REG8(     IMM_LA,                    0x5001C018,__READ_WRITE );
__IO_REG8(     IMM_SDAT,                  0x5001C01C,__READ       );
__IO_REG8_BIT( IMM_PREV,                  0x5001C020,__READ       ,__iim_prev_bits);
__IO_REG8(     IMM_SREV,                  0x5001C024,__READ       );
__IO_REG8(     IMM_PREG_P,                0x5001C028,__READ_WRITE );
__IO_REG8_BIT( IMM_SCS0,                  0x5001C02C,__READ_WRITE ,__iim_scs0_bits);
__IO_REG8_BIT( IMM_SCS1,                  0x5001C030,__READ_WRITE ,__iim_scs_bits);
__IO_REG8_BIT( IMM_SCS2,                  0x5001C034,__READ_WRITE ,__iim_scs_bits);
__IO_REG8_BIT( IMM_SCS3,                  0x5001C038,__READ_WRITE ,__iim_scs_bits);
__IO_REG8_BIT( IMM_FBAC0,                 0x5001C800,__READ_WRITE ,__iim_fbac0_bits);
__IO_REG8_BIT( IMM_HWV0,                  0x5001C804,__READ_WRITE ,__iim_hwv0_bits);
__IO_REG8_BIT( IMM_HWV1,                  0x5001C808,__READ_WRITE ,__iim_hwv1_bits);
__IO_REG8_BIT( IMM_HWV2,                  0x5001C80C,__READ_WRITE ,__iim_hwv2_bits);
__IO_REG8_BIT( IMM_HAB0,                  0x5001C810,__READ_WRITE ,__iim_hab0_bits);
__IO_REG8_BIT( IMM_HAB1,                  0x5001C814,__READ_WRITE ,__iim_hab1_bits);
__IO_REG8_BIT( IMM_PREV_FUSE,             0x5001C818,__READ_WRITE ,__iim_prev_fuse_bits);
__IO_REG8(     IMM_SREV_FUSE,             0x5001C81C,__READ_WRITE);
__IO_REG8(     IMM_SJC_CHALL0,            0x5001C820,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL1,            0x5001C824,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL2,            0x5001C828,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL3,            0x5001C82C,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL4,            0x5001C830,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL5,            0x5001C834,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL6,            0x5001C838,__READ_WRITE );
__IO_REG8(     IMM_SJC_CHALL7,            0x5001C83C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC16,               0x5001C840,__READ_WRITE );
__IO_REG8(     IMM_FB0UC17,               0x5001C844,__READ_WRITE );
__IO_REG8(     IMM_FB0UC18,               0x5001C848,__READ_WRITE );
__IO_REG8(     IMM_FB0UC19,               0x5001C84C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC20,               0x5001C850,__READ_WRITE );
__IO_REG8(     IMM_FB0UC21,               0x5001C854,__READ_WRITE );
__IO_REG8(     IMM_FB0UC22,               0x5001C858,__READ_WRITE );
__IO_REG8(     IMM_FB0UC23,               0x5001C85C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC24,               0x5001C860,__READ_WRITE );
__IO_REG8(     IMM_FB0UC25,               0x5001C864,__READ_WRITE );
__IO_REG8(     IMM_FB0UC26,               0x5001C868,__READ_WRITE );
__IO_REG8(     IMM_FB0UC27,               0x5001C86C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC28,               0x5001C870,__READ_WRITE );
__IO_REG8(     IMM_FB0UC29,               0x5001C874,__READ_WRITE );
__IO_REG8(     IMM_FB0UC30,               0x5001C878,__READ_WRITE );
__IO_REG8(     IMM_FB0UC31,               0x5001C87C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC32,               0x5001C880,__READ_WRITE );
__IO_REG8(     IMM_FB0UC33,               0x5001C884,__READ_WRITE );
__IO_REG8(     IMM_FB0UC34,               0x5001C888,__READ_WRITE );
__IO_REG8(     IMM_FB0UC35,               0x5001C88C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC36,               0x5001C890,__READ_WRITE );
__IO_REG8(     IMM_FB0UC37,               0x5001C894,__READ_WRITE );
__IO_REG8(     IMM_FB0UC38,               0x5001C898,__READ_WRITE );
__IO_REG8(     IMM_FB0UC39,               0x5001C89C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC40,               0x5001C8A0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC41,               0x5001C8A4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC42,               0x5001C8A8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC43,               0x5001C8AC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC44,               0x5001C8B0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC45,               0x5001C8B4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC46,               0x5001C8B8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC47,               0x5001C8BC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC48,               0x5001C8C0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC49,               0x5001C8C4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC50,               0x5001C8C8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC51,               0x5001C8CC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC52,               0x5001C8D0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC53,               0x5001C8D4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC54,               0x5001C8D8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC55,               0x5001C8DC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC56,               0x5001C8E0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC57,               0x5001C8E4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC58,               0x5001C8E8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC59,               0x5001C8EC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC60,               0x5001C8F0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC61,               0x5001C8F4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC62,               0x5001C8F8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC63,               0x5001C8FC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC64,               0x5001C900,__READ_WRITE );
__IO_REG8(     IMM_FB0UC65,               0x5001C904,__READ_WRITE );
__IO_REG8(     IMM_FB0UC66,               0x5001C908,__READ_WRITE );
__IO_REG8(     IMM_FB0UC67,               0x5001C90C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC68,               0x5001C910,__READ_WRITE );
__IO_REG8(     IMM_FB0UC69,               0x5001C914,__READ_WRITE );
__IO_REG8(     IMM_FB0UC70,               0x5001C918,__READ_WRITE );
__IO_REG8(     IMM_FB0UC71,               0x5001C91C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC72,               0x5001C920,__READ_WRITE );
__IO_REG8(     IMM_FB0UC73,               0x5001C924,__READ_WRITE );
__IO_REG8(     IMM_FB0UC74,               0x5001C928,__READ_WRITE );
__IO_REG8(     IMM_FB0UC75,               0x5001C92C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC76,               0x5001C930,__READ_WRITE );
__IO_REG8(     IMM_FB0UC77,               0x5001C934,__READ_WRITE );
__IO_REG8(     IMM_FB0UC78,               0x5001C938,__READ_WRITE );
__IO_REG8(     IMM_FB0UC79,               0x5001C93C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC80,               0x5001C940,__READ_WRITE );
__IO_REG8(     IMM_FB0UC81,               0x5001C944,__READ_WRITE );
__IO_REG8(     IMM_FB0UC82,               0x5001C948,__READ_WRITE );
__IO_REG8(     IMM_FB0UC83,               0x5001C94C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC84,               0x5001C950,__READ_WRITE );
__IO_REG8(     IMM_FB0UC85,               0x5001C954,__READ_WRITE );
__IO_REG8(     IMM_FB0UC86,               0x5001C958,__READ_WRITE );
__IO_REG8(     IMM_FB0UC87,               0x5001C95C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC88,               0x5001C960,__READ_WRITE );
__IO_REG8(     IMM_FB0UC89,               0x5001C964,__READ_WRITE );
__IO_REG8(     IMM_FB0UC90,               0x5001C968,__READ_WRITE );
__IO_REG8(     IMM_FB0UC91,               0x5001C96C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC92,               0x5001C970,__READ_WRITE );
__IO_REG8(     IMM_FB0UC93,               0x5001C974,__READ_WRITE );
__IO_REG8(     IMM_FB0UC94,               0x5001C978,__READ_WRITE );
__IO_REG8(     IMM_FB0UC95,               0x5001C97C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC96,               0x5001C980,__READ_WRITE );
__IO_REG8(     IMM_FB0UC97,               0x5001C984,__READ_WRITE );
__IO_REG8(     IMM_FB0UC98,               0x5001C988,__READ_WRITE );
__IO_REG8(     IMM_FB0UC99,               0x5001C98C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC100,              0x5001C990,__READ_WRITE );
__IO_REG8(     IMM_FB0UC101,              0x5001C994,__READ_WRITE );
__IO_REG8(     IMM_FB0UC102,              0x5001C998,__READ_WRITE );
__IO_REG8(     IMM_FB0UC103,              0x5001C99C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC104,              0x5001C9A0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC105,              0x5001C9A4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC106,              0x5001C9A8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC107,              0x5001C9AC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC108,              0x5001C9B0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC109,              0x5001C9B4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC110,              0x5001C9B8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC111,              0x5001C9BC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC112,              0x5001C9C0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC113,              0x5001C9C4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC114,              0x5001C9C8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC115,              0x5001C9CC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC116,              0x5001C9D0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC117,              0x5001C9D4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC118,              0x5001C9D8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC119,              0x5001C9DC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC120,              0x5001C9E0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC121,              0x5001C9E4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC122,              0x5001C9E8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC123,              0x5001C9EC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC124,              0x5001C9F0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC125,              0x5001C9F4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC126,              0x5001C9F8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC127,              0x5001C9FC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC128,              0x5001CA00,__READ_WRITE );
__IO_REG8(     IMM_FB0UC129,              0x5001CA04,__READ_WRITE );
__IO_REG8(     IMM_FB0UC130,              0x5001CA08,__READ_WRITE );
__IO_REG8(     IMM_FB0UC131,              0x5001CA0C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC132,              0x5001CA10,__READ_WRITE );
__IO_REG8(     IMM_FB0UC133,              0x5001CA14,__READ_WRITE );
__IO_REG8(     IMM_FB0UC134,              0x5001CA18,__READ_WRITE );
__IO_REG8(     IMM_FB0UC135,              0x5001CA1C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC136,              0x5001CA20,__READ_WRITE );
__IO_REG8(     IMM_FB0UC137,              0x5001CA24,__READ_WRITE );
__IO_REG8(     IMM_FB0UC138,              0x5001CA28,__READ_WRITE );
__IO_REG8(     IMM_FB0UC139,              0x5001CA2C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC140,              0x5001CA30,__READ_WRITE );
__IO_REG8(     IMM_FB0UC141,              0x5001CA34,__READ_WRITE );
__IO_REG8(     IMM_FB0UC142,              0x5001CA38,__READ_WRITE );
__IO_REG8(     IMM_FB0UC143,              0x5001CA3C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC144,              0x5001CA40,__READ_WRITE );
__IO_REG8(     IMM_FB0UC145,              0x5001CA44,__READ_WRITE );
__IO_REG8(     IMM_FB0UC146,              0x5001CA48,__READ_WRITE );
__IO_REG8(     IMM_FB0UC147,              0x5001CA4C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC148,              0x5001CA50,__READ_WRITE );
__IO_REG8(     IMM_FB0UC149,              0x5001CA54,__READ_WRITE );
__IO_REG8(     IMM_FB0UC150,              0x5001CA58,__READ_WRITE );
__IO_REG8(     IMM_FB0UC151,              0x5001CA5C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC152,              0x5001CA60,__READ_WRITE );
__IO_REG8(     IMM_FB0UC153,              0x5001CA64,__READ_WRITE );
__IO_REG8(     IMM_FB0UC154,              0x5001CA68,__READ_WRITE );
__IO_REG8(     IMM_FB0UC155,              0x5001CA6C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC156,              0x5001CA70,__READ_WRITE );
__IO_REG8(     IMM_FB0UC157,              0x5001CA74,__READ_WRITE );
__IO_REG8(     IMM_FB0UC158,              0x5001CA78,__READ_WRITE );
__IO_REG8(     IMM_FB0UC159,              0x5001CA7C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC160,              0x5001CA80,__READ_WRITE );
__IO_REG8(     IMM_FB0UC161,              0x5001CA84,__READ_WRITE );
__IO_REG8(     IMM_FB0UC162,              0x5001CA88,__READ_WRITE );
__IO_REG8(     IMM_FB0UC163,              0x5001CA8C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC164,              0x5001CA90,__READ_WRITE );
__IO_REG8(     IMM_FB0UC165,              0x5001CA94,__READ_WRITE );
__IO_REG8(     IMM_FB0UC166,              0x5001CA98,__READ_WRITE );
__IO_REG8(     IMM_FB0UC167,              0x5001CA9C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC168,              0x5001CAA0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC169,              0x5001CAA4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC170,              0x5001CAA8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC171,              0x5001CAAC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC172,              0x5001CAB0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC173,              0x5001CAB4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC174,              0x5001CAB8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC175,              0x5001CABC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC176,              0x5001CAC0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC177,              0x5001CAC4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC178,              0x5001CAC8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC179,              0x5001CACC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC180,              0x5001CAD0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC181,              0x5001CAD4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC182,              0x5001CAD8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC183,              0x5001CADC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC184,              0x5001CAE0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC185,              0x5001CAE4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC186,              0x5001CAE8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC187,              0x5001CAEC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC188,              0x5001CAF0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC189,              0x5001CAF4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC190,              0x5001CAF8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC191,              0x5001CAFC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC192,              0x5001CB00,__READ_WRITE );
__IO_REG8(     IMM_FB0UC193,              0x5001CB04,__READ_WRITE );
__IO_REG8(     IMM_FB0UC194,              0x5001CB08,__READ_WRITE );
__IO_REG8(     IMM_FB0UC195,              0x5001CB0C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC196,              0x5001CB10,__READ_WRITE );
__IO_REG8(     IMM_FB0UC197,              0x5001CB14,__READ_WRITE );
__IO_REG8(     IMM_FB0UC198,              0x5001CB18,__READ_WRITE );
__IO_REG8(     IMM_FB0UC199,              0x5001CB1C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC200,              0x5001CB20,__READ_WRITE );
__IO_REG8(     IMM_FB0UC201,              0x5001CB24,__READ_WRITE );
__IO_REG8(     IMM_FB0UC202,              0x5001CB28,__READ_WRITE );
__IO_REG8(     IMM_FB0UC203,              0x5001CB2C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC204,              0x5001CB30,__READ_WRITE );
__IO_REG8(     IMM_FB0UC205,              0x5001CB34,__READ_WRITE );
__IO_REG8(     IMM_FB0UC206,              0x5001CB38,__READ_WRITE );
__IO_REG8(     IMM_FB0UC207,              0x5001CB3C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC208,              0x5001CB40,__READ_WRITE );
__IO_REG8(     IMM_FB0UC209,              0x5001CB44,__READ_WRITE );
__IO_REG8(     IMM_FB0UC210,              0x5001CB48,__READ_WRITE );
__IO_REG8(     IMM_FB0UC211,              0x5001CB4C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC212,              0x5001CB50,__READ_WRITE );
__IO_REG8(     IMM_FB0UC213,              0x5001CB54,__READ_WRITE );
__IO_REG8(     IMM_FB0UC214,              0x5001CB58,__READ_WRITE );
__IO_REG8(     IMM_FB0UC215,              0x5001CB5C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC216,              0x5001CB60,__READ_WRITE );
__IO_REG8(     IMM_FB0UC217,              0x5001CB64,__READ_WRITE );
__IO_REG8(     IMM_FB0UC218,              0x5001CB68,__READ_WRITE );
__IO_REG8(     IMM_FB0UC219,              0x5001CB6C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC220,              0x5001CB70,__READ_WRITE );
__IO_REG8(     IMM_FB0UC221,              0x5001CB74,__READ_WRITE );
__IO_REG8(     IMM_FB0UC222,              0x5001CB78,__READ_WRITE );
__IO_REG8(     IMM_FB0UC223,              0x5001CB7C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC224,              0x5001CB80,__READ_WRITE );
__IO_REG8(     IMM_FB0UC225,              0x5001CB84,__READ_WRITE );
__IO_REG8(     IMM_FB0UC226,              0x5001CB88,__READ_WRITE );
__IO_REG8(     IMM_FB0UC227,              0x5001CB8C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC228,              0x5001CB90,__READ_WRITE );
__IO_REG8(     IMM_FB0UC229,              0x5001CB94,__READ_WRITE );
__IO_REG8(     IMM_FB0UC230,              0x5001CB98,__READ_WRITE );
__IO_REG8(     IMM_FB0UC231,              0x5001CB9C,__READ_WRITE );
__IO_REG8(     IMM_FB0UC232,              0x5001CBA0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC233,              0x5001CBA4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC234,              0x5001CBA8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC235,              0x5001CBAC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC236,              0x5001CBB0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC237,              0x5001CBB4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC238,              0x5001CBB8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC239,              0x5001CBBC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC240,              0x5001CBC0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC241,              0x5001CBC4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC242,              0x5001CBC8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC243,              0x5001CBCC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC244,              0x5001CBD0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC245,              0x5001CBD4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC246,              0x5001CBD8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC247,              0x5001CBDC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC248,              0x5001CBE0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC249,              0x5001CBE4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC250,              0x5001CBE8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC251,              0x5001CBEC,__READ_WRITE );
__IO_REG8(     IMM_FB0UC252,              0x5001CBF0,__READ_WRITE );
__IO_REG8(     IMM_FB0UC253,              0x5001CBF4,__READ_WRITE );
__IO_REG8(     IMM_FB0UC254,              0x5001CBF8,__READ_WRITE );
__IO_REG8(     IMM_FB0UC255,              0x5001CBFC,__READ_WRITE );
__IO_REG8_BIT( IMM_FBAC1,                 0x5001CC00,__READ_WRITE ,__iim_fbac1_bits);
__IO_REG8(     IMM_KEY0,                  0x5001CC04,__WRITE      );
__IO_REG8(     IMM_KEY1,                  0x5001CC08,__WRITE      );
__IO_REG8(     IMM_KEY2,                  0x5001CC0C,__WRITE      );
__IO_REG8(     IMM_KEY3,                  0x5001CC10,__WRITE      );
__IO_REG8(     IMM_KEY4,                  0x5001CC14,__WRITE      );
__IO_REG8(     IMM_KEY5,                  0x5001CC18,__WRITE      );
__IO_REG8(     IMM_KEY6,                  0x5001CC1C,__WRITE      );
__IO_REG8(     IMM_KEY7,                  0x5001CC20,__WRITE      );
__IO_REG8(     IMM_KEY8,                  0x5001CC24,__WRITE      );
__IO_REG8(     IMM_KEY9,                  0x5001CC28,__WRITE      );
__IO_REG8(     IMM_KEY10,                 0x5001CC2C,__WRITE      );
__IO_REG8(     IMM_KEY11,                 0x5001CC30,__WRITE      );
__IO_REG8(     IMM_KEY12,                 0x5001CC34,__WRITE      );
__IO_REG8(     IMM_KEY13,                 0x5001CC38,__WRITE      );
__IO_REG8(     IMM_KEY14,                 0x5001CC3C,__WRITE      );
__IO_REG8(     IMM_KEY15,                 0x5001CC40,__WRITE      );
__IO_REG8(     IMM_KEY16,                 0x5001CC44,__WRITE      );
__IO_REG8(     IMM_KEY17,                 0x5001CC48,__WRITE      );
__IO_REG8(     IMM_KEY18,                 0x5001CC4C,__WRITE      );
__IO_REG8(     IMM_KEY19,                 0x5001CC50,__WRITE      );
__IO_REG8(     IMM_KEY20,                 0x5001CC54,__WRITE      );
__IO_REG8(     IMM_SJC_RESP_0,            0x5001CC58,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_1,            0x5001CC5C,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_2,            0x5001CC60,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_3,            0x5001CC64,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_4,            0x5001CC68,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_5,            0x5001CC6C,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_6,            0x5001CC70,__READ_WRITE );
__IO_REG8(     IMM_SJC_RESP_7,            0x5001CC74,__READ_WRITE );
__IO_REG8(     IMM_FBIUC30,               0x5001CC78,__READ_WRITE );
__IO_REG8(     IMM_FBIUC31,               0x5001CC7C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC32,               0x5001CC80,__READ_WRITE );
__IO_REG8(     IMM_FBIUC33,               0x5001CC84,__READ_WRITE );
__IO_REG8(     IMM_FBIUC34,               0x5001CC88,__READ_WRITE );
__IO_REG8(     IMM_FBIUC35,               0x5001CC8C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC36,               0x5001CC90,__READ_WRITE );
__IO_REG8(     IMM_FBIUC37,               0x5001CC94,__READ_WRITE );
__IO_REG8(     IMM_FBIUC38,               0x5001CC98,__READ_WRITE );
__IO_REG8(     IMM_FBIUC39,               0x5001CC9C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC40,               0x5001CCA0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC41,               0x5001CCA4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC42,               0x5001CCA8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC43,               0x5001CCAC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC44,               0x5001CCB0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC45,               0x5001CCB4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC46,               0x5001CCB8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC47,               0x5001CCBC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC48,               0x5001CCC0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC49,               0x5001CCC4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC50,               0x5001CCC8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC51,               0x5001CCCC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC52,               0x5001CCD0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC53,               0x5001CCD4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC54,               0x5001CCD8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC55,               0x5001CCDC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC56,               0x5001CCE0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC57,               0x5001CCE4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC58,               0x5001CCE8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC59,               0x5001CCEC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC60,               0x5001CCF0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC61,               0x5001CCF4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC62,               0x5001CCF8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC63,               0x5001CCFC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC64,               0x5001CD00,__READ_WRITE );
__IO_REG8(     IMM_FBIUC65,               0x5001CD04,__READ_WRITE );
__IO_REG8(     IMM_FBIUC66,               0x5001CD08,__READ_WRITE );
__IO_REG8(     IMM_FBIUC67,               0x5001CD0C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC68,               0x5001CD10,__READ_WRITE );
__IO_REG8(     IMM_FBIUC69,               0x5001CD14,__READ_WRITE );
__IO_REG8(     IMM_FBIUC70,               0x5001CD18,__READ_WRITE );
__IO_REG8(     IMM_FBIUC71,               0x5001CD1C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC72,               0x5001CD20,__READ_WRITE );
__IO_REG8(     IMM_FBIUC73,               0x5001CD24,__READ_WRITE );
__IO_REG8(     IMM_FBIUC74,               0x5001CD28,__READ_WRITE );
__IO_REG8(     IMM_FBIUC75,               0x5001CD2C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC76,               0x5001CD30,__READ_WRITE );
__IO_REG8(     IMM_FBIUC77,               0x5001CD34,__READ_WRITE );
__IO_REG8(     IMM_FBIUC78,               0x5001CD38,__READ_WRITE );
__IO_REG8(     IMM_FBIUC79,               0x5001CD3C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC80,               0x5001CD40,__READ_WRITE );
__IO_REG8(     IMM_FBIUC81,               0x5001CD44,__READ_WRITE );
__IO_REG8(     IMM_FBIUC82,               0x5001CD48,__READ_WRITE );
__IO_REG8(     IMM_FBIUC83,               0x5001CD4C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC84,               0x5001CD50,__READ_WRITE );
__IO_REG8(     IMM_FBIUC85,               0x5001CD54,__READ_WRITE );
__IO_REG8(     IMM_FBIUC86,               0x5001CD58,__READ_WRITE );
__IO_REG8(     IMM_FBIUC87,               0x5001CD5C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC88,               0x5001CD60,__READ_WRITE );
__IO_REG8(     IMM_FBIUC89,               0x5001CD64,__READ_WRITE );
__IO_REG8(     IMM_FBIUC90,               0x5001CD68,__READ_WRITE );
__IO_REG8(     IMM_FBIUC91,               0x5001CD6C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC92,               0x5001CD70,__READ_WRITE );
__IO_REG8(     IMM_FBIUC93,               0x5001CD74,__READ_WRITE );
__IO_REG8(     IMM_FBIUC94,               0x5001CD78,__READ_WRITE );
__IO_REG8(     IMM_FBIUC95,               0x5001CD7C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC96,               0x5001CD80,__READ_WRITE );
__IO_REG8(     IMM_FBIUC97,               0x5001CD84,__READ_WRITE );
__IO_REG8(     IMM_FBIUC98,               0x5001CD88,__READ_WRITE );
__IO_REG8(     IMM_FBIUC99,               0x5001CD8C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC100,              0x5001CD90,__READ_WRITE );
__IO_REG8(     IMM_FBIUC101,              0x5001CD94,__READ_WRITE );
__IO_REG8(     IMM_FBIUC102,              0x5001CD98,__READ_WRITE );
__IO_REG8(     IMM_FBIUC103,              0x5001CD9C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC104,              0x5001CDA0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC105,              0x5001CDA4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC106,              0x5001CDA8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC107,              0x5001CDAC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC108,              0x5001CDB0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC109,              0x5001CDB4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC110,              0x5001CDB8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC111,              0x5001CDBC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC112,              0x5001CDC0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC113,              0x5001CDC4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC114,              0x5001CDC8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC115,              0x5001CDCC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC116,              0x5001CDD0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC117,              0x5001CDD4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC118,              0x5001CDD8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC119,              0x5001CDDC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC120,              0x5001CDE0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC121,              0x5001CDE4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC122,              0x5001CDE8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC123,              0x5001CDEC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC124,              0x5001CDF0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC125,              0x5001CDF4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC126,              0x5001CDF8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC127,              0x5001CDFC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC128,              0x5001CE00,__READ_WRITE );
__IO_REG8(     IMM_FBIUC129,              0x5001CE04,__READ_WRITE );
__IO_REG8(     IMM_FBIUC130,              0x5001CE08,__READ_WRITE );
__IO_REG8(     IMM_FBIUC131,              0x5001CE0C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC132,              0x5001CE10,__READ_WRITE );
__IO_REG8(     IMM_FBIUC133,              0x5001CE14,__READ_WRITE );
__IO_REG8(     IMM_FBIUC134,              0x5001CE18,__READ_WRITE );
__IO_REG8(     IMM_FBIUC135,              0x5001CE1C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC136,              0x5001CE20,__READ_WRITE );
__IO_REG8(     IMM_FBIUC137,              0x5001CE24,__READ_WRITE );
__IO_REG8(     IMM_FBIUC138,              0x5001CE28,__READ_WRITE );
__IO_REG8(     IMM_FBIUC139,              0x5001CE2C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC140,              0x5001CE30,__READ_WRITE );
__IO_REG8(     IMM_FBIUC141,              0x5001CE34,__READ_WRITE );
__IO_REG8(     IMM_FBIUC142,              0x5001CE38,__READ_WRITE );
__IO_REG8(     IMM_FBIUC143,              0x5001CE3C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC144,              0x5001CE40,__READ_WRITE );
__IO_REG8(     IMM_FBIUC145,              0x5001CE44,__READ_WRITE );
__IO_REG8(     IMM_FBIUC146,              0x5001CE48,__READ_WRITE );
__IO_REG8(     IMM_FBIUC147,              0x5001CE4C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC148,              0x5001CE50,__READ_WRITE );
__IO_REG8(     IMM_FBIUC149,              0x5001CE54,__READ_WRITE );
__IO_REG8(     IMM_FBIUC150,              0x5001CE58,__READ_WRITE );
__IO_REG8(     IMM_FBIUC151,              0x5001CE5C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC152,              0x5001CE60,__READ_WRITE );
__IO_REG8(     IMM_FBIUC153,              0x5001CE64,__READ_WRITE );
__IO_REG8(     IMM_FBIUC154,              0x5001CE68,__READ_WRITE );
__IO_REG8(     IMM_FBIUC155,              0x5001CE6C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC156,              0x5001CE70,__READ_WRITE );
__IO_REG8(     IMM_FBIUC157,              0x5001CE74,__READ_WRITE );
__IO_REG8(     IMM_FBIUC158,              0x5001CE78,__READ_WRITE );
__IO_REG8(     IMM_FBIUC159,              0x5001CE7C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC160,              0x5001CE80,__READ_WRITE );
__IO_REG8(     IMM_FBIUC161,              0x5001CE84,__READ_WRITE );
__IO_REG8(     IMM_FBIUC162,              0x5001CE88,__READ_WRITE );
__IO_REG8(     IMM_FBIUC163,              0x5001CE8C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC164,              0x5001CE90,__READ_WRITE );
__IO_REG8(     IMM_FBIUC165,              0x5001CE94,__READ_WRITE );
__IO_REG8(     IMM_FBIUC166,              0x5001CE98,__READ_WRITE );
__IO_REG8(     IMM_FBIUC167,              0x5001CE9C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC168,              0x5001CEA0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC169,              0x5001CEA4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC170,              0x5001CEA8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC171,              0x5001CEAC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC172,              0x5001CEB0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC173,              0x5001CEB4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC174,              0x5001CEB8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC175,              0x5001CEBC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC176,              0x5001CEC0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC177,              0x5001CEC4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC178,              0x5001CEC8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC179,              0x5001CECC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC180,              0x5001CED0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC181,              0x5001CED4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC182,              0x5001CED8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC183,              0x5001CEDC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC184,              0x5001CEE0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC185,              0x5001CEE4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC186,              0x5001CEE8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC187,              0x5001CEEC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC188,              0x5001CEF0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC189,              0x5001CEF4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC190,              0x5001CEF8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC191,              0x5001CEFC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC192,              0x5001CF00,__READ_WRITE );
__IO_REG8(     IMM_FBIUC193,              0x5001CF04,__READ_WRITE );
__IO_REG8(     IMM_FBIUC194,              0x5001CF08,__READ_WRITE );
__IO_REG8(     IMM_FBIUC195,              0x5001CF0C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC196,              0x5001CF10,__READ_WRITE );
__IO_REG8(     IMM_FBIUC197,              0x5001CF14,__READ_WRITE );
__IO_REG8(     IMM_FBIUC198,              0x5001CF18,__READ_WRITE );
__IO_REG8(     IMM_FBIUC199,              0x5001CF1C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC200,              0x5001CF20,__READ_WRITE );
__IO_REG8(     IMM_FBIUC201,              0x5001CF24,__READ_WRITE );
__IO_REG8(     IMM_FBIUC202,              0x5001CF28,__READ_WRITE );
__IO_REG8(     IMM_FBIUC203,              0x5001CF2C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC204,              0x5001CF30,__READ_WRITE );
__IO_REG8(     IMM_FBIUC205,              0x5001CF34,__READ_WRITE );
__IO_REG8(     IMM_FBIUC206,              0x5001CF38,__READ_WRITE );
__IO_REG8(     IMM_FBIUC207,              0x5001CF3C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC208,              0x5001CF40,__READ_WRITE );
__IO_REG8(     IMM_FBIUC209,              0x5001CF44,__READ_WRITE );
__IO_REG8(     IMM_FBIUC210,              0x5001CF48,__READ_WRITE );
__IO_REG8(     IMM_FBIUC211,              0x5001CF4C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC212,              0x5001CF50,__READ_WRITE );
__IO_REG8(     IMM_FBIUC213,              0x5001CF54,__READ_WRITE );
__IO_REG8(     IMM_FBIUC214,              0x5001CF58,__READ_WRITE );
__IO_REG8(     IMM_FBIUC215,              0x5001CF5C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC216,              0x5001CF60,__READ_WRITE );
__IO_REG8(     IMM_FBIUC217,              0x5001CF64,__READ_WRITE );
__IO_REG8(     IMM_FBIUC218,              0x5001CF68,__READ_WRITE );
__IO_REG8(     IMM_FBIUC219,              0x5001CF6C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC220,              0x5001CF70,__READ_WRITE );
__IO_REG8(     IMM_FBIUC221,              0x5001CF74,__READ_WRITE );
__IO_REG8(     IMM_FBIUC222,              0x5001CF78,__READ_WRITE );
__IO_REG8(     IMM_FBIUC223,              0x5001CF7C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC224,              0x5001CF80,__READ_WRITE );
__IO_REG8(     IMM_FBIUC225,              0x5001CF84,__READ_WRITE );
__IO_REG8(     IMM_FBIUC226,              0x5001CF88,__READ_WRITE );
__IO_REG8(     IMM_FBIUC227,              0x5001CF8C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC228,              0x5001CF90,__READ_WRITE );
__IO_REG8(     IMM_FBIUC229,              0x5001CF94,__READ_WRITE );
__IO_REG8(     IMM_FBIUC230,              0x5001CF98,__READ_WRITE );
__IO_REG8(     IMM_FBIUC231,              0x5001CF9C,__READ_WRITE );
__IO_REG8(     IMM_FBIUC232,              0x5001CFA0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC233,              0x5001CFA4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC234,              0x5001CFA8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC235,              0x5001CFAC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC236,              0x5001CFB0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC237,              0x5001CFB4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC238,              0x5001CFB8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC239,              0x5001CFBC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC240,              0x5001CFC0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC241,              0x5001CFC4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC242,              0x5001CFC8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC243,              0x5001CFCC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC244,              0x5001CFD0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC245,              0x5001CFD4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC246,              0x5001CFD8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC247,              0x5001CFDC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC248,              0x5001CFE0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC249,              0x5001CFE4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC250,              0x5001CFE8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC251,              0x5001CFEC,__READ_WRITE );
__IO_REG8(     IMM_FBIUC252,              0x5001CFF0,__READ_WRITE );
__IO_REG8(     IMM_FBIUC253,              0x5001CFF4,__READ_WRITE );
__IO_REG8(     IMM_FBIUC254,              0x5001CFF8,__READ_WRITE );
__IO_REG8(     IMM_FBIUC255,              0x5001CFFC,__READ_WRITE );
__IO_REG8_BIT( IMM_FBAC2,                 0x5001D000,__READ_WRITE ,__iim_fbac_bits);
__IO_REG8(     IMM_XCORD,                 0x5001D004,__READ_WRITE );
__IO_REG8(     IMM_YCORD,                 0x5001D008,__READ_WRITE );
__IO_REG8(     IMM_FAB,                   0x5001D00C,__READ_WRITE );
__IO_REG8(     IMM_WAFER,                 0x5001D010,__READ_WRITE );
__IO_REG8(     IMM_LOT0,                  0x5001D014,__READ_WRITE );
__IO_REG8(     IMM_LOT1,                  0x5001D018,__READ_WRITE );
__IO_REG8(     IMM_LOT2,                  0x5001D01C,__READ_WRITE );
__IO_REG8(     IMM_PROB_SBIN,             0x5001D020,__READ_WRITE );
__IO_REG8(     IMM_FT_SBIN,               0x5001D024,__READ_WRITE );
__IO_REG8(     IMM_FB2UC10,               0x5001D028,__READ_WRITE );
__IO_REG8(     IMM_FB2UC11,               0x5001D02C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC12,               0x5001D030,__READ_WRITE );
__IO_REG8(     IMM_FB2UC13,               0x5001D034,__READ_WRITE );
__IO_REG8(     IMM_FB2UC14,               0x5001D038,__READ_WRITE );
__IO_REG8(     IMM_FB2UC15,               0x5001D03C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC16,               0x5001D040,__READ_WRITE );
__IO_REG8(     IMM_FB2UC17,               0x5001D044,__READ_WRITE );
__IO_REG8(     IMM_FB2UC18,               0x5001D048,__READ_WRITE );
__IO_REG8(     IMM_FB2UC19,               0x5001D04C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC20,               0x5001D050,__READ_WRITE );
__IO_REG8(     IMM_FB2UC21,               0x5001D054,__READ_WRITE );
__IO_REG8(     IMM_FB2UC22,               0x5001D058,__READ_WRITE );
__IO_REG8(     IMM_FB2UC23,               0x5001D05C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC24,               0x5001D060,__READ_WRITE );
__IO_REG8(     IMM_FB2UC25,               0x5001D064,__READ_WRITE );
__IO_REG8(     IMM_FB2UC26,               0x5001D068,__READ_WRITE );
__IO_REG8(     IMM_FB2UC27,               0x5001D06C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC28,               0x5001D070,__READ_WRITE );
__IO_REG8(     IMM_FB2UC29,               0x5001D074,__READ_WRITE );
__IO_REG8(     IMM_FB2UC30,               0x5001D078,__READ_WRITE );
__IO_REG8(     IMM_FB2UC31,               0x5001D07C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC32,               0x5001D080,__READ_WRITE );
__IO_REG8(     IMM_FB2UC33,               0x5001D084,__READ_WRITE );
__IO_REG8(     IMM_FB2UC34,               0x5001D088,__READ_WRITE );
__IO_REG8(     IMM_FB2UC35,               0x5001D08C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC36,               0x5001D090,__READ_WRITE );
__IO_REG8(     IMM_FB2UC37,               0x5001D094,__READ_WRITE );
__IO_REG8(     IMM_FB2UC38,               0x5001D098,__READ_WRITE );
__IO_REG8(     IMM_FB2UC39,               0x5001D09C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC40,               0x5001D0A0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC41,               0x5001D0A4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC42,               0x5001D0A8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC43,               0x5001D0AC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC44,               0x5001D0B0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC45,               0x5001D0B4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC46,               0x5001D0B8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC47,               0x5001D0BC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC48,               0x5001D0C0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC49,               0x5001D0C4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC50,               0x5001D0C8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC51,               0x5001D0CC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC52,               0x5001D0D0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC53,               0x5001D0D4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC54,               0x5001D0D8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC55,               0x5001D0DC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC56,               0x5001D0E0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC57,               0x5001D0E4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC58,               0x5001D0E8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC59,               0x5001D0EC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC60,               0x5001D0F0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC61,               0x5001D0F4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC62,               0x5001D0F8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC63,               0x5001D0FC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC64,               0x5001D100,__READ_WRITE );
__IO_REG8(     IMM_FB2UC65,               0x5001D104,__READ_WRITE );
__IO_REG8(     IMM_FB2UC66,               0x5001D108,__READ_WRITE );
__IO_REG8(     IMM_FB2UC67,               0x5001D10C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC68,               0x5001D110,__READ_WRITE );
__IO_REG8(     IMM_FB2UC69,               0x5001D114,__READ_WRITE );
__IO_REG8(     IMM_FB2UC70,               0x5001D118,__READ_WRITE );
__IO_REG8(     IMM_FB2UC71,               0x5001D11C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC72,               0x5001D120,__READ_WRITE );
__IO_REG8(     IMM_FB2UC73,               0x5001D124,__READ_WRITE );
__IO_REG8(     IMM_FB2UC74,               0x5001D128,__READ_WRITE );
__IO_REG8(     IMM_FB2UC75,               0x5001D12C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC76,               0x5001D130,__READ_WRITE );
__IO_REG8(     IMM_FB2UC77,               0x5001D134,__READ_WRITE );
__IO_REG8(     IMM_FB2UC78,               0x5001D138,__READ_WRITE );
__IO_REG8(     IMM_FB2UC79,               0x5001D13C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC80,               0x5001D140,__READ_WRITE );
__IO_REG8(     IMM_FB2UC81,               0x5001D144,__READ_WRITE );
__IO_REG8(     IMM_FB2UC82,               0x5001D148,__READ_WRITE );
__IO_REG8(     IMM_FB2UC83,               0x5001D14C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC84,               0x5001D150,__READ_WRITE );
__IO_REG8(     IMM_FB2UC85,               0x5001D154,__READ_WRITE );
__IO_REG8(     IMM_FB2UC86,               0x5001D158,__READ_WRITE );
__IO_REG8(     IMM_FB2UC87,               0x5001D15C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC88,               0x5001D160,__READ_WRITE );
__IO_REG8(     IMM_FB2UC89,               0x5001D164,__READ_WRITE );
__IO_REG8(     IMM_FB2UC90,               0x5001D168,__READ_WRITE );
__IO_REG8(     IMM_FB2UC91,               0x5001D16C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC92,               0x5001D170,__READ_WRITE );
__IO_REG8(     IMM_FB2UC93,               0x5001D174,__READ_WRITE );
__IO_REG8(     IMM_FB2UC94,               0x5001D178,__READ_WRITE );
__IO_REG8(     IMM_FB2UC95,               0x5001D17C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC96,               0x5001D180,__READ_WRITE );
__IO_REG8(     IMM_FB2UC97,               0x5001D184,__READ_WRITE );
__IO_REG8(     IMM_FB2UC98,               0x5001D188,__READ_WRITE );
__IO_REG8(     IMM_FB2UC99,               0x5001D18C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC100,              0x5001D190,__READ_WRITE );
__IO_REG8(     IMM_FB2UC101,              0x5001D194,__READ_WRITE );
__IO_REG8(     IMM_FB2UC102,              0x5001D198,__READ_WRITE );
__IO_REG8(     IMM_FB2UC103,              0x5001D19C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC104,              0x5001D1A0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC105,              0x5001D1A4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC106,              0x5001D1A8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC107,              0x5001D1AC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC108,              0x5001D1B0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC109,              0x5001D1B4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC110,              0x5001D1B8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC111,              0x5001D1BC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC112,              0x5001D1C0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC113,              0x5001D1C4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC114,              0x5001D1C8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC115,              0x5001D1CC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC116,              0x5001D1D0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC117,              0x5001D1D4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC118,              0x5001D1D8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC119,              0x5001D1DC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC120,              0x5001D1E0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC121,              0x5001D1E4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC122,              0x5001D1E8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC123,              0x5001D1EC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC124,              0x5001D1F0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC125,              0x5001D1F4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC126,              0x5001D1F8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC127,              0x5001D1FC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC128,              0x5001D200,__READ_WRITE );
__IO_REG8(     IMM_FB2UC129,              0x5001D204,__READ_WRITE );
__IO_REG8(     IMM_FB2UC130,              0x5001D208,__READ_WRITE );
__IO_REG8(     IMM_FB2UC131,              0x5001D20C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC132,              0x5001D210,__READ_WRITE );
__IO_REG8(     IMM_FB2UC133,              0x5001D214,__READ_WRITE );
__IO_REG8(     IMM_FB2UC134,              0x5001D218,__READ_WRITE );
__IO_REG8(     IMM_FB2UC135,              0x5001D21C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC136,              0x5001D220,__READ_WRITE );
__IO_REG8(     IMM_FB2UC137,              0x5001D224,__READ_WRITE );
__IO_REG8(     IMM_FB2UC138,              0x5001D228,__READ_WRITE );
__IO_REG8(     IMM_FB2UC139,              0x5001D22C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC140,              0x5001D230,__READ_WRITE );
__IO_REG8(     IMM_FB2UC141,              0x5001D234,__READ_WRITE );
__IO_REG8(     IMM_FB2UC142,              0x5001D238,__READ_WRITE );
__IO_REG8(     IMM_FB2UC143,              0x5001D23C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC144,              0x5001D240,__READ_WRITE );
__IO_REG8(     IMM_FB2UC145,              0x5001D244,__READ_WRITE );
__IO_REG8(     IMM_FB2UC146,              0x5001D248,__READ_WRITE );
__IO_REG8(     IMM_FB2UC147,              0x5001D24C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC148,              0x5001D250,__READ_WRITE );
__IO_REG8(     IMM_FB2UC149,              0x5001D254,__READ_WRITE );
__IO_REG8(     IMM_FB2UC150,              0x5001D258,__READ_WRITE );
__IO_REG8(     IMM_FB2UC151,              0x5001D25C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC152,              0x5001D260,__READ_WRITE );
__IO_REG8(     IMM_FB2UC153,              0x5001D264,__READ_WRITE );
__IO_REG8(     IMM_FB2UC154,              0x5001D268,__READ_WRITE );
__IO_REG8(     IMM_FB2UC155,              0x5001D26C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC156,              0x5001D270,__READ_WRITE );
__IO_REG8(     IMM_FB2UC157,              0x5001D274,__READ_WRITE );
__IO_REG8(     IMM_FB2UC158,              0x5001D278,__READ_WRITE );
__IO_REG8(     IMM_FB2UC159,              0x5001D27C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC160,              0x5001D280,__READ_WRITE );
__IO_REG8(     IMM_FB2UC161,              0x5001D284,__READ_WRITE );
__IO_REG8(     IMM_FB2UC162,              0x5001D288,__READ_WRITE );
__IO_REG8(     IMM_FB2UC163,              0x5001D28C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC164,              0x5001D290,__READ_WRITE );
__IO_REG8(     IMM_FB2UC165,              0x5001D294,__READ_WRITE );
__IO_REG8(     IMM_FB2UC166,              0x5001D298,__READ_WRITE );
__IO_REG8(     IMM_FB2UC167,              0x5001D29C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC168,              0x5001D2A0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC169,              0x5001D2A4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC170,              0x5001D2A8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC171,              0x5001D2AC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC172,              0x5001D2B0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC173,              0x5001D2B4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC174,              0x5001D2B8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC175,              0x5001D2BC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC176,              0x5001D2C0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC177,              0x5001D2C4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC178,              0x5001D2C8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC179,              0x5001D2CC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC180,              0x5001D2D0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC181,              0x5001D2D4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC182,              0x5001D2D8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC183,              0x5001D2DC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC184,              0x5001D2E0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC185,              0x5001D2E4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC186,              0x5001D2E8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC187,              0x5001D2EC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC188,              0x5001D2F0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC189,              0x5001D2F4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC190,              0x5001D2F8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC191,              0x5001D2FC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC192,              0x5001D300,__READ_WRITE );
__IO_REG8(     IMM_FB2UC193,              0x5001D304,__READ_WRITE );
__IO_REG8(     IMM_FB2UC194,              0x5001D308,__READ_WRITE );
__IO_REG8(     IMM_FB2UC195,              0x5001D30C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC196,              0x5001D310,__READ_WRITE );
__IO_REG8(     IMM_FB2UC197,              0x5001D314,__READ_WRITE );
__IO_REG8(     IMM_FB2UC198,              0x5001D318,__READ_WRITE );
__IO_REG8(     IMM_FB2UC199,              0x5001D31C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC200,              0x5001D320,__READ_WRITE );
__IO_REG8(     IMM_FB2UC201,              0x5001D324,__READ_WRITE );
__IO_REG8(     IMM_FB2UC202,              0x5001D328,__READ_WRITE );
__IO_REG8(     IMM_FB2UC203,              0x5001D32C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC204,              0x5001D330,__READ_WRITE );
__IO_REG8(     IMM_FB2UC205,              0x5001D334,__READ_WRITE );
__IO_REG8(     IMM_FB2UC206,              0x5001D338,__READ_WRITE );
__IO_REG8(     IMM_FB2UC207,              0x5001D33C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC208,              0x5001D340,__READ_WRITE );
__IO_REG8(     IMM_FB2UC209,              0x5001D344,__READ_WRITE );
__IO_REG8(     IMM_FB2UC210,              0x5001D348,__READ_WRITE );
__IO_REG8(     IMM_FB2UC211,              0x5001D34C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC212,              0x5001D350,__READ_WRITE );
__IO_REG8(     IMM_FB2UC213,              0x5001D354,__READ_WRITE );
__IO_REG8(     IMM_FB2UC214,              0x5001D358,__READ_WRITE );
__IO_REG8(     IMM_FB2UC215,              0x5001D35C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC216,              0x5001D360,__READ_WRITE );
__IO_REG8(     IMM_FB2UC217,              0x5001D364,__READ_WRITE );
__IO_REG8(     IMM_FB2UC218,              0x5001D368,__READ_WRITE );
__IO_REG8(     IMM_FB2UC219,              0x5001D36C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC220,              0x5001D370,__READ_WRITE );
__IO_REG8(     IMM_FB2UC221,              0x5001D374,__READ_WRITE );
__IO_REG8(     IMM_FB2UC222,              0x5001D378,__READ_WRITE );
__IO_REG8(     IMM_FB2UC223,              0x5001D37C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC224,              0x5001D380,__READ_WRITE );
__IO_REG8(     IMM_FB2UC225,              0x5001D384,__READ_WRITE );
__IO_REG8(     IMM_FB2UC226,              0x5001D388,__READ_WRITE );
__IO_REG8(     IMM_FB2UC227,              0x5001D38C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC228,              0x5001D390,__READ_WRITE );
__IO_REG8(     IMM_FB2UC229,              0x5001D394,__READ_WRITE );
__IO_REG8(     IMM_FB2UC230,              0x5001D398,__READ_WRITE );
__IO_REG8(     IMM_FB2UC231,              0x5001D39C,__READ_WRITE );
__IO_REG8(     IMM_FB2UC232,              0x5001D3A0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC233,              0x5001D3A4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC234,              0x5001D3A8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC235,              0x5001D3AC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC236,              0x5001D3B0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC237,              0x5001D3B4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC238,              0x5001D3B8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC239,              0x5001D3BC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC240,              0x5001D3C0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC241,              0x5001D3C4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC242,              0x5001D3C8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC243,              0x5001D3CC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC244,              0x5001D3D0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC245,              0x5001D3D4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC246,              0x5001D3D8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC247,              0x5001D3DC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC248,              0x5001D3E0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC249,              0x5001D3E4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC250,              0x5001D3E8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC251,              0x5001D3EC,__READ_WRITE );
__IO_REG8(     IMM_FB2UC252,              0x5001D3F0,__READ_WRITE );
__IO_REG8(     IMM_FB2UC253,              0x5001D3F4,__READ_WRITE );
__IO_REG8(     IMM_FB2UC254,              0x5001D3F8,__READ_WRITE );
__IO_REG8(     IMM_FB2UC255,              0x5001D3FC,__READ_WRITE );
__IO_REG8_BIT( IMM_FBAC3,                 0x5001D400,__READ_WRITE ,__iim_fbac_bits);
__IO_REG8(     IMM_FB3UC01,               0x5001D404,__READ_WRITE );
__IO_REG8(     IMM_FB3UC02,               0x5001D408,__READ_WRITE );
__IO_REG8(     IMM_FB3UC03,               0x5001D40C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC04,               0x5001D410,__READ_WRITE );
__IO_REG8(     IMM_FB3UC05,               0x5001D414,__READ_WRITE );
__IO_REG8(     IMM_FB3UC06,               0x5001D418,__READ_WRITE );
__IO_REG8(     IMM_FB3UC07,               0x5001D41C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC08,               0x5001D420,__READ_WRITE );
__IO_REG8(     IMM_FB3UC09,               0x5001D424,__READ_WRITE );
__IO_REG8(     IMM_FB3UC10,               0x5001D428,__READ_WRITE );
__IO_REG8(     IMM_FB3UC11,               0x5001D42C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC12,               0x5001D430,__READ_WRITE );
__IO_REG8(     IMM_FB3UC13,               0x5001D434,__READ_WRITE );
__IO_REG8(     IMM_FB3UC14,               0x5001D438,__READ_WRITE );
__IO_REG8(     IMM_FB3UC15,               0x5001D43C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC16,               0x5001D440,__READ_WRITE );
__IO_REG8(     IMM_FB3UC17,               0x5001D444,__READ_WRITE );
__IO_REG8(     IMM_FB3UC18,               0x5001D448,__READ_WRITE );
__IO_REG8(     IMM_FB3UC19,               0x5001D44C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC20,               0x5001D450,__READ_WRITE );
__IO_REG8(     IMM_FB3UC21,               0x5001D454,__READ_WRITE );
__IO_REG8(     IMM_FB3UC22,               0x5001D458,__READ_WRITE );
__IO_REG8(     IMM_FB3UC23,               0x5001D45C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC24,               0x5001D460,__READ_WRITE );
__IO_REG8(     IMM_FB3UC25,               0x5001D464,__READ_WRITE );
__IO_REG8(     IMM_FB3UC26,               0x5001D468,__READ_WRITE );
__IO_REG8(     IMM_FB3UC27,               0x5001D46C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC28,               0x5001D470,__READ_WRITE );
__IO_REG8(     IMM_FB3UC29,               0x5001D474,__READ_WRITE );
__IO_REG8(     IMM_FB3UC30,               0x5001D478,__READ_WRITE );
__IO_REG8(     IMM_FB3UC31,               0x5001D47C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC32,               0x5001D480,__READ_WRITE );
__IO_REG8(     IMM_FB3UC33,               0x5001D484,__READ_WRITE );
__IO_REG8(     IMM_FB3UC34,               0x5001D488,__READ_WRITE );
__IO_REG8(     IMM_FB3UC35,               0x5001D48C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC36,               0x5001D490,__READ_WRITE );
__IO_REG8(     IMM_FB3UC37,               0x5001D494,__READ_WRITE );
__IO_REG8(     IMM_FB3UC38,               0x5001D498,__READ_WRITE );
__IO_REG8(     IMM_FB3UC39,               0x5001D49C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC40,               0x5001D4A0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC41,               0x5001D4A4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC42,               0x5001D4A8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC43,               0x5001D4AC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC44,               0x5001D4B0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC45,               0x5001D4B4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC46,               0x5001D4B8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC47,               0x5001D4BC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC48,               0x5001D4C0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC49,               0x5001D4C4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC50,               0x5001D4C8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC51,               0x5001D4CC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC52,               0x5001D4D0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC53,               0x5001D4D4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC54,               0x5001D4D8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC55,               0x5001D4DC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC56,               0x5001D4E0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC57,               0x5001D4E4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC58,               0x5001D4E8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC59,               0x5001D4EC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC60,               0x5001D4F0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC61,               0x5001D4F4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC62,               0x5001D4F8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC63,               0x5001D4FC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC64,               0x5001D500,__READ_WRITE );
__IO_REG8(     IMM_FB3UC65,               0x5001D504,__READ_WRITE );
__IO_REG8(     IMM_FB3UC66,               0x5001D508,__READ_WRITE );
__IO_REG8(     IMM_FB3UC67,               0x5001D50C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC68,               0x5001D510,__READ_WRITE );
__IO_REG8(     IMM_FB3UC69,               0x5001D514,__READ_WRITE );
__IO_REG8(     IMM_FB3UC70,               0x5001D518,__READ_WRITE );
__IO_REG8(     IMM_FB3UC71,               0x5001D51C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC72,               0x5001D520,__READ_WRITE );
__IO_REG8(     IMM_FB3UC73,               0x5001D524,__READ_WRITE );
__IO_REG8(     IMM_FB3UC74,               0x5001D528,__READ_WRITE );
__IO_REG8(     IMM_FB3UC75,               0x5001D52C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC76,               0x5001D530,__READ_WRITE );
__IO_REG8(     IMM_FB3UC77,               0x5001D534,__READ_WRITE );
__IO_REG8(     IMM_FB3UC78,               0x5001D538,__READ_WRITE );
__IO_REG8(     IMM_FB3UC79,               0x5001D53C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC80,               0x5001D540,__READ_WRITE );
__IO_REG8(     IMM_FB3UC81,               0x5001D544,__READ_WRITE );
__IO_REG8(     IMM_FB3UC82,               0x5001D548,__READ_WRITE );
__IO_REG8(     IMM_FB3UC83,               0x5001D54C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC84,               0x5001D550,__READ_WRITE );
__IO_REG8(     IMM_FB3UC85,               0x5001D554,__READ_WRITE );
__IO_REG8(     IMM_FB3UC86,               0x5001D558,__READ_WRITE );
__IO_REG8(     IMM_FB3UC87,               0x5001D55C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC88,               0x5001D560,__READ_WRITE );
__IO_REG8(     IMM_FB3UC89,               0x5001D564,__READ_WRITE );
__IO_REG8(     IMM_FB3UC90,               0x5001D568,__READ_WRITE );
__IO_REG8(     IMM_FB3UC91,               0x5001D56C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC92,               0x5001D570,__READ_WRITE );
__IO_REG8(     IMM_FB3UC93,               0x5001D574,__READ_WRITE );
__IO_REG8(     IMM_FB3UC94,               0x5001D578,__READ_WRITE );
__IO_REG8(     IMM_FB3UC95,               0x5001D57C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC96,               0x5001D580,__READ_WRITE );
__IO_REG8(     IMM_FB3UC97,               0x5001D584,__READ_WRITE );
__IO_REG8(     IMM_FB3UC98,               0x5001D588,__READ_WRITE );
__IO_REG8(     IMM_FB3UC99,               0x5001D58C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC100,              0x5001D590,__READ_WRITE );
__IO_REG8(     IMM_FB3UC101,              0x5001D594,__READ_WRITE );
__IO_REG8(     IMM_FB3UC102,              0x5001D598,__READ_WRITE );
__IO_REG8(     IMM_FB3UC103,              0x5001D59C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC104,              0x5001D5A0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC105,              0x5001D5A4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC106,              0x5001D5A8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC107,              0x5001D5AC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC108,              0x5001D5B0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC109,              0x5001D5B4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC110,              0x5001D5B8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC111,              0x5001D5BC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC112,              0x5001D5C0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC113,              0x5001D5C4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC114,              0x5001D5C8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC115,              0x5001D5CC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC116,              0x5001D5D0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC117,              0x5001D5D4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC118,              0x5001D5D8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC119,              0x5001D5DC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC120,              0x5001D5E0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC121,              0x5001D5E4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC122,              0x5001D5E8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC123,              0x5001D5EC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC124,              0x5001D5F0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC125,              0x5001D5F4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC126,              0x5001D5F8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC127,              0x5001D5FC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC128,              0x5001D600,__READ_WRITE );
__IO_REG8(     IMM_FB3UC129,              0x5001D604,__READ_WRITE );
__IO_REG8(     IMM_FB3UC130,              0x5001D608,__READ_WRITE );
__IO_REG8(     IMM_FB3UC131,              0x5001D60C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC132,              0x5001D610,__READ_WRITE );
__IO_REG8(     IMM_FB3UC133,              0x5001D614,__READ_WRITE );
__IO_REG8(     IMM_FB3UC134,              0x5001D618,__READ_WRITE );
__IO_REG8(     IMM_FB3UC135,              0x5001D61C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC136,              0x5001D620,__READ_WRITE );
__IO_REG8(     IMM_FB3UC137,              0x5001D624,__READ_WRITE );
__IO_REG8(     IMM_FB3UC138,              0x5001D628,__READ_WRITE );
__IO_REG8(     IMM_FB3UC139,              0x5001D62C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC140,              0x5001D630,__READ_WRITE );
__IO_REG8(     IMM_FB3UC141,              0x5001D634,__READ_WRITE );
__IO_REG8(     IMM_FB3UC142,              0x5001D638,__READ_WRITE );
__IO_REG8(     IMM_FB3UC143,              0x5001D63C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC144,              0x5001D640,__READ_WRITE );
__IO_REG8(     IMM_FB3UC145,              0x5001D644,__READ_WRITE );
__IO_REG8(     IMM_FB3UC146,              0x5001D648,__READ_WRITE );
__IO_REG8(     IMM_FB3UC147,              0x5001D64C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC148,              0x5001D650,__READ_WRITE );
__IO_REG8(     IMM_FB3UC149,              0x5001D654,__READ_WRITE );
__IO_REG8(     IMM_FB3UC150,              0x5001D658,__READ_WRITE );
__IO_REG8(     IMM_FB3UC151,              0x5001D65C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC152,              0x5001D660,__READ_WRITE );
__IO_REG8(     IMM_FB3UC153,              0x5001D664,__READ_WRITE );
__IO_REG8(     IMM_FB3UC154,              0x5001D668,__READ_WRITE );
__IO_REG8(     IMM_FB3UC155,              0x5001D66C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC156,              0x5001D670,__READ_WRITE );
__IO_REG8(     IMM_FB3UC157,              0x5001D674,__READ_WRITE );
__IO_REG8(     IMM_FB3UC158,              0x5001D678,__READ_WRITE );
__IO_REG8(     IMM_FB3UC159,              0x5001D67C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC160,              0x5001D680,__READ_WRITE );
__IO_REG8(     IMM_FB3UC161,              0x5001D684,__READ_WRITE );
__IO_REG8(     IMM_FB3UC162,              0x5001D688,__READ_WRITE );
__IO_REG8(     IMM_FB3UC163,              0x5001D68C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC164,              0x5001D690,__READ_WRITE );
__IO_REG8(     IMM_FB3UC165,              0x5001D694,__READ_WRITE );
__IO_REG8(     IMM_FB3UC166,              0x5001D698,__READ_WRITE );
__IO_REG8(     IMM_FB3UC167,              0x5001D69C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC168,              0x5001D6A0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC169,              0x5001D6A4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC170,              0x5001D6A8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC171,              0x5001D6AC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC172,              0x5001D6B0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC173,              0x5001D6B4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC174,              0x5001D6B8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC175,              0x5001D6BC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC176,              0x5001D6C0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC177,              0x5001D6C4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC178,              0x5001D6C8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC179,              0x5001D6CC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC180,              0x5001D6D0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC181,              0x5001D6D4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC182,              0x5001D6D8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC183,              0x5001D6DC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC184,              0x5001D6E0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC185,              0x5001D6E4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC186,              0x5001D6E8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC187,              0x5001D6EC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC188,              0x5001D6F0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC189,              0x5001D6F4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC190,              0x5001D6F8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC191,              0x5001D6FC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC192,              0x5001D700,__READ_WRITE );
__IO_REG8(     IMM_FB3UC193,              0x5001D704,__READ_WRITE );
__IO_REG8(     IMM_FB3UC194,              0x5001D708,__READ_WRITE );
__IO_REG8(     IMM_FB3UC195,              0x5001D70C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC196,              0x5001D710,__READ_WRITE );
__IO_REG8(     IMM_FB3UC197,              0x5001D714,__READ_WRITE );
__IO_REG8(     IMM_FB3UC198,              0x5001D718,__READ_WRITE );
__IO_REG8(     IMM_FB3UC199,              0x5001D71C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC200,              0x5001D720,__READ_WRITE );
__IO_REG8(     IMM_FB3UC201,              0x5001D724,__READ_WRITE );
__IO_REG8(     IMM_FB3UC202,              0x5001D728,__READ_WRITE );
__IO_REG8(     IMM_FB3UC203,              0x5001D72C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC204,              0x5001D730,__READ_WRITE );
__IO_REG8(     IMM_FB3UC205,              0x5001D734,__READ_WRITE );
__IO_REG8(     IMM_FB3UC206,              0x5001D738,__READ_WRITE );
__IO_REG8(     IMM_FB3UC207,              0x5001D73C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC208,              0x5001D740,__READ_WRITE );
__IO_REG8(     IMM_FB3UC209,              0x5001D744,__READ_WRITE );
__IO_REG8(     IMM_FB3UC210,              0x5001D748,__READ_WRITE );
__IO_REG8(     IMM_FB3UC211,              0x5001D74C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC212,              0x5001D750,__READ_WRITE );
__IO_REG8(     IMM_FB3UC213,              0x5001D754,__READ_WRITE );
__IO_REG8(     IMM_FB3UC214,              0x5001D758,__READ_WRITE );
__IO_REG8(     IMM_FB3UC215,              0x5001D75C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC216,              0x5001D760,__READ_WRITE );
__IO_REG8(     IMM_FB3UC217,              0x5001D764,__READ_WRITE );
__IO_REG8(     IMM_FB3UC218,              0x5001D768,__READ_WRITE );
__IO_REG8(     IMM_FB3UC219,              0x5001D76C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC220,              0x5001D770,__READ_WRITE );
__IO_REG8(     IMM_FB3UC221,              0x5001D774,__READ_WRITE );
__IO_REG8(     IMM_FB3UC222,              0x5001D778,__READ_WRITE );
__IO_REG8(     IMM_FB3UC223,              0x5001D77C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC224,              0x5001D780,__READ_WRITE );
__IO_REG8(     IMM_FB3UC225,              0x5001D784,__READ_WRITE );
__IO_REG8(     IMM_FB3UC226,              0x5001D788,__READ_WRITE );
__IO_REG8(     IMM_FB3UC227,              0x5001D78C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC228,              0x5001D790,__READ_WRITE );
__IO_REG8(     IMM_FB3UC229,              0x5001D794,__READ_WRITE );
__IO_REG8(     IMM_FB3UC230,              0x5001D798,__READ_WRITE );
__IO_REG8(     IMM_FB3UC231,              0x5001D79C,__READ_WRITE );
__IO_REG8(     IMM_FB3UC232,              0x5001D7A0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC233,              0x5001D7A4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC234,              0x5001D7A8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC235,              0x5001D7AC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC236,              0x5001D7B0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC237,              0x5001D7B4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC238,              0x5001D7B8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC239,              0x5001D7BC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC240,              0x5001D7C0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC241,              0x5001D7C4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC242,              0x5001D7C8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC243,              0x5001D7CC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC244,              0x5001D7D0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC245,              0x5001D7D4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC246,              0x5001D7D8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC247,              0x5001D7DC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC248,              0x5001D7E0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC249,              0x5001D7E4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC250,              0x5001D7E8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC251,              0x5001D7EC,__READ_WRITE );
__IO_REG8(     IMM_FB3UC252,              0x5001D7F0,__READ_WRITE );
__IO_REG8(     IMM_FB3UC253,              0x5001D7F4,__READ_WRITE );
__IO_REG8(     IMM_FB3UC254,              0x5001D7F8,__READ_WRITE );
__IO_REG8(     IMM_FB3UC255,              0x5001D7FC,__READ_WRITE );

/***************************************************************************
 **
 **  L2CC
 **
 ***************************************************************************/
__IO_REG32_BIT(L2CC_ID,                   0x30000000,__READ       ,__l2cc_id_bits);
__IO_REG32_BIT(L2CC_TYPE,                 0x30000004,__READ       ,__l2cc_type_bits);
__IO_REG32_BIT(L2CC_CTRL,                 0x30000100,__READ_WRITE ,__l2cc_ctrl_bits);
__IO_REG32_BIT(L2CC_AUX_CTRL,             0x30000104,__READ_WRITE ,__l2cc_aux_ctrl_bits);
__IO_REG32(    L2CC_CACHE_SYNC,           0x30000730,__READ_WRITE );
__IO_REG32(    L2CC_INV_BY_PA,            0x30000770,__READ_WRITE );
__IO_REG32(    L2CC_INV_BY_WAY,           0x3000077C,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_LINE_BY_PA,     0x300007B0,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_LINE_BY_INDX,   0x300007B8,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_BY_WAY,         0x300007BC,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_INV_BY_PA,      0x300007F0,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_INV_BY_INDX,    0x300007F8,__READ_WRITE );
__IO_REG32(    L2CC_CLEAN_INV_BY_WAY,     0x300007FC,__READ_WRITE );
__IO_REG32(    L2CC_LOCKDOWN_DWAY,        0x30000900,__READ_WRITE );
__IO_REG32(    L2CC_LOCKDOWN_IWAY,        0x30000904,__READ_WRITE );
__IO_REG32_BIT(L2CC_TEST,                 0x30000F00,__READ_WRITE ,__l2cc_test_bits);
__IO_REG32(    L2CC_LINE_DATA0,           0x30000F10,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA1,           0x30000F14,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA2,           0x30000F18,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA3,           0x30000F1C,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA4,           0x30000F20,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA5,           0x30000F24,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA6,           0x30000F28,__READ_WRITE );
__IO_REG32(    L2CC_LINE_DATA7,           0x30000F2C,__READ_WRITE );
__IO_REG32_BIT(L2CC_LINE_TAG,             0x30000F30,__READ_WRITE ,__l2cc_line_tag_bits);
__IO_REG32_BIT(L2CC_DEBUG_CTRL,           0x30000F40,__READ_WRITE ,__l2cc_debug_ctrl_bits);

/***************************************************************************
 **
 **  EVTMON
 **
 ***************************************************************************/
__IO_REG32_BIT(EMMC,                      0x43F08000,__READ_WRITE ,__emmc_bits);
__IO_REG32_BIT(EMCS,                      0x43F08004,__READ_WRITE ,__emcs_bits);
__IO_REG32_BIT(EMCC0,                     0x43F08008,__READ_WRITE ,__emcc_bits);
__IO_REG32_BIT(EMCC1,                     0x43F0800C,__READ_WRITE ,__emcc_bits);
__IO_REG32_BIT(EMCC2,                     0x43F08010,__READ_WRITE ,__emcc_bits);
__IO_REG32_BIT(EMCC3,                     0x43F08014,__READ_WRITE ,__emcc_bits);
__IO_REG32(    EMC0,                      0x43F08020,__READ       );
__IO_REG32(    EMC1,                      0x43F08024,__READ       );
__IO_REG32(    EMC2,                      0x43F08028,__READ       );
__IO_REG32(    EMC3,                      0x43F0802C,__READ       );

/***************************************************************************
 **
 **  M3IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M3IFCTL,                   0xB8003000,__READ_WRITE ,__m3ifctl_bits);
__IO_REG32_BIT(M3IFSCFG0,                 0xB8003028,__READ_WRITE ,__m3ifscfg0_bits);
__IO_REG32_BIT(M3IFSCFG1,                 0xB800302C,__READ_WRITE ,__m3ifscfg1_bits);
__IO_REG32_BIT(M3IFSCFG2,                 0xB8003030,__READ_WRITE ,__m3ifscfg2_bits);
__IO_REG32_BIT(M3IFSSR0,                  0xB8003034,__READ_WRITE ,__m3ifssr0_bits);
__IO_REG32_BIT(M3IFSSR1,                  0xB8003038,__READ_WRITE ,__m3ifssr1_bits);
__IO_REG32_BIT(M3IFMLWE0,                 0xB8003040,__READ_WRITE ,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE1,                 0xB8003044,__READ_WRITE ,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE2,                 0xB8003048,__READ_WRITE ,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE3,                 0xB800304C,__READ_WRITE ,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE4,                 0xB8003050,__READ_WRITE ,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE5,                 0xB8003054,__READ_WRITE ,__m3ifmlwe_bits);

/***************************************************************************
 **
 **  WEIM
 **
 ***************************************************************************/
__IO_REG32_BIT(WEIM_CSCR0U,               0xB8002000,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR0L,               0xB8002004,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR0A,               0xB8002008,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_CSCR1U,               0xB8002010,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR1L,               0xB8002014,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR1A,               0xB8002018,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_CSCR2U,               0xB8002020,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR2L,               0xB8002024,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR2A,               0xB8002028,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_CSCR3U,               0xB8002030,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR3L,               0xB8002034,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR3A,               0xB8002038,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_CSCR4U,               0xB8002040,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR4L,               0xB8002044,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR4A,               0xB8002048,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_CSCR5U,               0xB8002050,__READ_WRITE ,__weim_cscru_bits);
__IO_REG32_BIT(WEIM_CSCR5L,               0xB8002054,__READ_WRITE ,__weim_cscrl_bits);
__IO_REG32_BIT(WEIM_CSCR5A,               0xB8002058,__READ_WRITE ,__weim_cscra_bits);
__IO_REG32_BIT(WEIM_WCR,                  0xB8002060,__READ_WRITE ,__weim_wcr_bits);

/***************************************************************************
 **
 **  ESDCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL0,                   0xB8001000,__READ_WRITE ,__esdctl_bits);
__IO_REG32_BIT(ESDCFG0,                   0xB8001004,__READ_WRITE ,__esdcfg_bits);
__IO_REG32_BIT(ESDCTL1,                   0xB8001008,__READ_WRITE ,__esdctl_bits);
__IO_REG32_BIT(ESDCFG1,                   0xB800100C,__READ_WRITE ,__esdcfg_bits);
__IO_REG32_BIT(ESDMISC,                   0xB8001010,__READ_WRITE ,__esdmisc_bits);
__IO_REG32_BIT(ESDCDLY1,                  0xB8001020,__READ_WRITE ,__esdcdly1_bits);
__IO_REG32_BIT(ESDCDLY2,                  0xB8001024,__READ_WRITE ,__esdcdly2_bits);
__IO_REG32_BIT(ESDCDLY3,                  0xB8001028,__READ_WRITE ,__esdcdly3_bits);
__IO_REG32_BIT(ESDCDLY4,                  0xB800102C,__READ_WRITE ,__esdcdly4_bits);
__IO_REG32_BIT(ESDCDLY5,                  0xB8001030,__READ_WRITE ,__esdcdly5_bits);
__IO_REG32_BIT(ESDCDLYL,                  0xB8001034,__READ       ,__esdcdlyl_bits);

/***************************************************************************
 **
 **  NANDFC
 **
 ***************************************************************************/
__IO_REG16_BIT(NFC_BUFSIZE,               0xB8000E00,__READ       ,__nfc_bufsize_bits);
__IO_REG16_BIT(NFC_RBA,                   0xB8000E04,__READ_WRITE ,__nfc_rba_bits);
__IO_REG16(    NAND_FLASH_ADD,            0xB8000E06,__READ_WRITE );
__IO_REG16(    NAND_FLASH_CMD,            0xB8000E08,__READ_WRITE );
__IO_REG16_BIT(NFC_IBLC,                  0xB8000E0A,__READ_WRITE ,__nfc_iblc_bits);
__IO_REG16_BIT(ECC_SRR,                   0xB8000E0C,__READ       ,__ecc_srr_bits);
__IO_REG16_BIT(ECC_RSLT_MA,               0xB8000E0E,__READ       ,__ecc_rslt_ma_bits);
__IO_REG16_BIT(ECC_RSLT_SA,               0xB8000E10,__READ       ,__ecc_rslt_sa_bits);
__IO_REG16_BIT(NF_WR_PROT,                0xB8000E12,__READ_WRITE ,__nf_wr_prot_bits);
__IO_REG16(    NFC_USBA,                  0xB8000E14,__READ_WRITE );
__IO_REG16(    NFC_UEBA,                  0xB8000E16,__READ_WRITE );
__IO_REG16_BIT(NF_WR_PROT_STA,            0xB8000E18,__READ       ,__nf_wr_prot_sta_bits);
__IO_REG16_BIT(NAND_FC1,                  0xB8000E1A,__READ_WRITE ,__nand_fc1_bits);
__IO_REG16_BIT(NAND_FC2,                  0xB8000E1C,__READ_WRITE ,__nand_fc2_bits);

/***************************************************************************
 **
 **  PCMCIA
 **
 ***************************************************************************/
__IO_REG32_BIT(PIPR,                      0xB8004000,__READ       ,__pipr_bits);
__IO_REG32_BIT(PSCR,                      0xB8004004,__READ_WRITE ,__pscr_bits);
__IO_REG32_BIT(PER,                       0xB8004008,__READ_WRITE ,__per_bits);
__IO_REG32_BIT(PBR0,                      0xB800400C,__READ_WRITE ,__pbr_bits);
__IO_REG32_BIT(PBR1,                      0xB8004010,__READ_WRITE ,__pbr_bits);
__IO_REG32_BIT(PBR2,                      0xB8004014,__READ_WRITE ,__pbr_bits);
__IO_REG32_BIT(PBR3,                      0xB8004018,__READ_WRITE ,__pbr_bits);
__IO_REG32_BIT(PBR4,                      0xB800401C,__READ_WRITE ,__pbr_bits);
__IO_REG32_BIT(POR0,                      0xB8004028,__READ_WRITE ,__por_bits);
__IO_REG32_BIT(POR1,                      0xB800402C,__READ_WRITE ,__por_bits);
__IO_REG32_BIT(POR2,                      0xB8004030,__READ_WRITE ,__por_bits);
__IO_REG32_BIT(POR3,                      0xB8004034,__READ_WRITE ,__por_bits);
__IO_REG32_BIT(POR4,                      0xB8004038,__READ_WRITE ,__por_bits);
__IO_REG32_BIT(POFR0,                     0xB8004044,__READ_WRITE ,__pofr_bits);
__IO_REG32_BIT(POFR1,                     0xB8004048,__READ_WRITE ,__pofr_bits);
__IO_REG32_BIT(POFR2,                     0xB800404C,__READ_WRITE ,__pofr_bits);
__IO_REG32_BIT(POFR3,                     0xB8004050,__READ_WRITE ,__pofr_bits);
__IO_REG32_BIT(POFR4,                     0xB8004054,__READ_WRITE ,__pofr_bits);
__IO_REG32_BIT(PGCR,                      0xB8004060,__READ_WRITE ,__pgcr_bits);
__IO_REG32_BIT(PGSR,                      0xB8004064,__READ_WRITE ,__pgsr_bits);

/***************************************************************************
 **
 **  One Wire
 **
 ***************************************************************************/
__IO_REG16_BIT(OW_CONTROL,                0x43F9C000,__READ_WRITE ,__ow_control_bits);
__IO_REG16_BIT(OW_TIME_DIVIDER,           0x43F9C002,__READ_WRITE ,__ow_time_divider_bits);
__IO_REG16_BIT(OW_RESET,                  0x43F9C004,__READ_WRITE ,__ow_reset_bits);

/***************************************************************************
 **
 **  ATA
 **
 ***************************************************************************/
__IO_REG8(     ATA_TIME_OFF,              0x43F8C000,__READ_WRITE );
__IO_REG8(     ATA_TIME_ON,               0x43F8C001,__READ_WRITE );
__IO_REG8(     ATA_TIME_1,                0x43F8C002,__READ_WRITE );
__IO_REG8(     ATA_TIME_2W,               0x43F8C003,__READ_WRITE );
__IO_REG8(     ATA_TIME_2R,               0x43F8C004,__READ_WRITE );
__IO_REG8(     ATA_TIME_AX,               0x43F8C005,__READ_WRITE );
__IO_REG8(     ATA_TIME_4,                0x43F8C007,__READ_WRITE );
__IO_REG8(     ATA_TIME_9,                0x43F8C008,__READ_WRITE );
__IO_REG8(     ATA_TIME_M,                0x43F8C009,__READ_WRITE );
__IO_REG8(     ATA_TIME_JN,               0x43F8C00A,__READ_WRITE );
__IO_REG8(     ATA_TIME_D,                0x43F8C00B,__READ_WRITE );
__IO_REG8(     ATA_TIME_K,                0x43F8C00C,__READ_WRITE );
__IO_REG8(     ATA_TIME_ACK,              0x43F8C00D,__READ_WRITE );
__IO_REG8(     ATA_TIME_ENV,              0x43F8C00E,__READ_WRITE );
__IO_REG8(     ATA_TIME_RPX,              0x43F8C00F,__READ_WRITE );
__IO_REG8(     ATA_TIME_ZAH,              0x43F8C010,__READ_WRITE );
__IO_REG8(     ATA_TIME_MLIX,             0x43F8C011,__READ_WRITE );
__IO_REG8(     ATA_TIME_DVH,              0x43F8C012,__READ_WRITE );
__IO_REG8(     ATA_TIME_DZFS,             0x43F8C013,__READ_WRITE );
__IO_REG8(     ATA_TIME_DVS,              0x43F8C014,__READ_WRITE );
__IO_REG8(     ATA_TIME_CVH,              0x43F8C015,__READ_WRITE );
__IO_REG8(     ATA_TIME_SS,               0x43F8C016,__READ_WRITE );
__IO_REG8(     ATA_TIME_CYC,              0x43F8C017,__READ_WRITE );
__IO_REG8(     ATA_FIFO_DATA_32,          0x43F8C018,__READ_WRITE );
__IO_REG8(     ATA_FIFO_DATA_16,          0x43F8C01C,__READ_WRITE );
__IO_REG8(     ATA_FIFO_FILL,             0x43F8C020,__READ       );
__IO_REG8_BIT( ATA_CONTROL,               0x43F8C024,__READ_WRITE ,__ata_control_bits);
__IO_REG8_BIT( ATA_INTR_PEND,             0x43F8C028,__READ       ,__ata_intr_pend_bits);
__IO_REG8_BIT( ATA_INTR_ENA,              0x43F8C02C,__READ_WRITE ,__ata_intr_ena_bits);
__IO_REG8_BIT( ATA_INTR_CLR,              0x43F8C030,__WRITE      ,__ata_intr_clr_bits);
__IO_REG8(     ATA_FIFO_ALARM,            0x43F8C034,__READ_WRITE );
__IO_REG16(    ATA_DRIVE_DATA,            0x43F8C0A0,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_FEATURES,        0x43F8C0A4,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_SECTOR_COUNT,    0x43F8C0A8,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_SECTOR_NUM,      0x43F8C0AC,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_CYL_LOW,         0x43F8C0B0,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_CYL_HIGH,        0x43F8C0B4,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_DEV_HEAD,        0x43F8C0B8,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_COMMAND,         0x43F8C0BC,__WRITE      );
__IO_REG32(    ATA_DRIVE_STATUS,          0x43F8C0C0,__READ       );
__IO_REG32(    ATA_DRIVE_ALT_STATUS,      0x43F8C0C4,__READ       );
__IO_REG32(    ATA_DRIVE_CONTROL,         0x43F8C0C8,__WRITE      );

/***************************************************************************
 **
 **  CSPI1
 **
 ***************************************************************************/
__IO_REG32(    CSPI1_RXDATA,              0x43FA4000,__READ      );
__IO_REG32(    CSPI1_TXDATA,              0x43FA4004,__WRITE     );
__IO_REG32_BIT(CSPI1_CONREG,              0x43FA4008,__READ_WRITE,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI1_INTREG,              0x43FA400C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI1_DMAREG,              0x43FA4010,__READ_WRITE,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI1_STATREG,             0x43FA4014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI1_PERIODREG,           0x43FA4018,__READ_WRITE,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI1_TESTREG,             0x43FA41C0,__READ_WRITE,__cspi_testreg_bits);

/***************************************************************************
 **
 **  CSPI2
 **
 ***************************************************************************/
__IO_REG32(    CSPI2_RXDATA,              0x50010000,__READ      );
__IO_REG32(    CSPI2_TXDATA,              0x50010004,__WRITE     );
__IO_REG32_BIT(CSPI2_CONREG,              0x50010008,__READ_WRITE,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI2_INTREG,              0x5001000C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI2_DMAREG,              0x50010010,__READ_WRITE,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI2_STATREG,             0x50010014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI2_PERIODREG,           0x50010018,__READ_WRITE,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI2_TESTREG,             0x500101C0,__READ_WRITE,__cspi_testreg_bits);

/***************************************************************************
 **
 **  CSPI3
 **
 ***************************************************************************/
__IO_REG32(    CSPI3_RXDATA,              0x53F84000,__READ      );
__IO_REG32(    CSPI3_TXDATA,              0x53F84004,__WRITE     );
__IO_REG32_BIT(CSPI3_CONREG,              0x53F84008,__READ_WRITE,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI3_INTREG,              0x53F8400C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI3_DMAREG,              0x53F84010,__READ_WRITE,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI3_STATREG,             0x53F84014,__READ_WRITE,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI3_PERIODREG,           0x53F84018,__READ_WRITE,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI3_TESTREG,             0x53F841C0,__READ_WRITE,__cspi_testreg_bits);

/***************************************************************************
 **
 ** FIR
 **
 ***************************************************************************/
__IO_REG32_BIT(FIRITCR,                   0x53F8C000,__READ_WRITE ,__firitcr_bits);
__IO_REG32_BIT(FIRITCTR,                  0x53F8C004,__READ_WRITE ,__firitctr_bits);
__IO_REG32_BIT(FIRIRCR,                   0x53F8C008,__READ_WRITE ,__firircr_bits);
__IO_REG32_BIT(FIRITSR,                   0x53F8C00C,__READ_WRITE ,__firitsr_bits);
__IO_REG32_BIT(FIRIRSR,                   0x53F8C010,__READ_WRITE ,__firirsr_bits);
__IO_REG32(    FIRITXD,                   0x53F8C014,__WRITE      );
__IO_REG32(    FIRIRXD,                   0x53F8C018,__READ       );
__IO_REG32_BIT(FIRICR,                    0x53F8C01C,__READ_WRITE ,__firicr_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR1,                     0x43F80000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR1,                     0x43F80004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR1,                     0x43F80008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR1,                     0x43F8000C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR1,                     0x43F80010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR2,                     0x43F98000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR2,                     0x43F98004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR2,                     0x43F98008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR2,                     0x43F9800C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR2,                     0x43F98010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C3
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR3,                     0x43F84000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR3,                     0x43F84004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR3,                     0x43F84008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR3,                     0x43F8400C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR3,                     0x43F84010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPCR,                      0x43FA8000,__READ_WRITE ,__kpcr_bits);
__IO_REG16_BIT(KPSR,                      0x43FA8002,__READ_WRITE ,__kpsr_bits);
__IO_REG16_BIT(KDDR,                      0x43FA8004,__READ_WRITE ,__kddr_bits);
__IO_REG16_BIT(KPDR,                      0x43FA8006,__READ_WRITE ,__kpdr_bits);

/***************************************************************************
 **
 **  MSHC1
 **
 ***************************************************************************/
__IO_REG8(     MSHC1_TIMEOUT,             0x50024000,__READ_WRITE );
__IO_REG8_BIT( MSHC1_INTR_STAT,           0x50024014,__READ_WRITE ,__mshc_intr_stat_bits);
__IO_REG8_BIT( MSHC1_INTR_ENA,            0x5002401C,__READ_WRITE ,__mshc_intr_ena_bits);

/***************************************************************************
 **
 **  MSHC2
 **
 ***************************************************************************/
__IO_REG8(     MSHC2_TIMEOUT,             0x50028000,__READ_WRITE );
__IO_REG8_BIT( MSHC2_INTR_STAT,           0x50028014,__READ_WRITE ,__mshc_intr_stat_bits);
__IO_REG8_BIT( MSHC2_INTR_ENA,            0x5002801C,__READ_WRITE ,__mshc_intr_ena_bits);

/***************************************************************************
 **
 **  SMSC
 **
 ***************************************************************************/


/***************************************************************************
 **
 **  SDHC1
 **
 ***************************************************************************/
__IO_REG32_BIT(SDHC1_STR_STP_CLK,         0x50004000,__READ_WRITE ,__str_stp_clk_bits);
__IO_REG32_BIT(SDHC1_STATUS,              0x50004004,__READ       ,__sd_status_bits);
__IO_REG32_BIT(SDHC1_CLK_RATE,            0x50004008,__READ_WRITE ,__clk_rate_bits);
__IO_REG32_BIT(SDHC1_CMD_DAT_CONT,        0x5000400C,__READ_WRITE ,__cmd_dat_cont_bits);
__IO_REG32_BIT(SDHC1_RES_TO,              0x50004010,__READ_WRITE ,__res_to_bits);
__IO_REG32_BIT(SDHC1_READ_TO,             0x50004014,__READ_WRITE ,__read_to_bits);
__IO_REG32_BIT(SDHC1_BLK_LEN,             0x50004018,__READ_WRITE ,__blk_len_bits);
__IO_REG32_BIT(SDHC1_NOB,                 0x5000401C,__READ_WRITE ,__nob_bits);
__IO_REG32_BIT(SDHC1_REV_NO,              0x50004020,__READ       ,__rev_no_bits);
__IO_REG32_BIT(SDHC1_INT_CTRL,            0x50004024,__READ_WRITE ,__mmcsd_int_ctrl_bits);
__IO_REG32_BIT(SDHC1_CMD,                 0x50004028,__READ_WRITE ,__cmd_bits);
__IO_REG32(    SDHC1_ARG,                 0x5000402C,__READ_WRITE );
__IO_REG32_BIT(SDHC1_RES_FIFO,            0x50004034,__READ       ,__res_fifo_bits);
__IO_REG32(    SDHC1_BUFFER_ACCESS,       0x50004038,__READ_WRITE );

/***************************************************************************
 **
 **  SDHC2
 **
 ***************************************************************************/
__IO_REG32_BIT(SDHC2_STR_STP_CLK,         0x50008000,__READ_WRITE ,__str_stp_clk_bits);
__IO_REG32_BIT(SDHC2_STATUS,              0x50008004,__READ       ,__sd_status_bits);
__IO_REG32_BIT(SDHC2_CLK_RATE,            0x50008008,__READ_WRITE ,__clk_rate_bits);
__IO_REG32_BIT(SDHC2_CMD_DAT_CONT,        0x5000800C,__READ_WRITE ,__cmd_dat_cont_bits);
__IO_REG32_BIT(SDHC2_RES_TO,              0x50008010,__READ_WRITE ,__res_to_bits);
__IO_REG32_BIT(SDHC2_READ_TO,             0x50008014,__READ_WRITE ,__read_to_bits);
__IO_REG32_BIT(SDHC2_BLK_LEN,             0x50008018,__READ_WRITE ,__blk_len_bits);
__IO_REG32_BIT(SDHC2_NOB,                 0x5000801C,__READ_WRITE ,__nob_bits);
__IO_REG32_BIT(SDHC2_REV_NO,              0x50008020,__READ       ,__rev_no_bits);
__IO_REG32_BIT(SDHC2_INT_CTRL,            0x50008024,__READ_WRITE ,__mmcsd_int_ctrl_bits);
__IO_REG32_BIT(SDHC2_CMD,                 0x50008028,__READ_WRITE ,__cmd_bits);
__IO_REG32(    SDHC2_ARG,                 0x5000802C,__READ_WRITE );
__IO_REG32_BIT(SDHC2_RES_FIFO,            0x50008034,__READ       ,__res_fifo_bits);
__IO_REG32(    SDHC2_BUFFER_ACCESS,       0x50008038,__READ_WRITE );

/***************************************************************************
 **
 **  SIM
 **
 ***************************************************************************/
__IO_REG32_BIT(SIM_PORT1_CNTL,            0x50018000,__READ_WRITE ,__sim_port1_cntl_bits);
__IO_REG32_BIT(SIM_SETUP,                 0x50018004,__READ_WRITE ,__sim_setup_bits);
__IO_REG32_BIT(SIM_PORT1_DETECT,          0x50018008,__READ_WRITE ,__sim_port_detect_bits);
__IO_REG32_BIT(SIM_PORT1_XMT_BUF,         0x5001800C,__READ_WRITE ,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM_PORT1_RCV_BUF,         0x50018010,__READ       ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM_PORT0_CNTL,            0x50018014,__READ_WRITE ,__sim_port0_cntl_bits);
__IO_REG32_BIT(SIM_CNTL,                  0x50018018,__READ_WRITE ,__sim_cntl_bits);
__IO_REG32_BIT(SIM_CLOCK_SELECT,          0x5001801C,__READ_WRITE ,__sim_clock_select_bits);
__IO_REG32_BIT(SIM_RCV_THRESHOLD,         0x50018020,__READ_WRITE ,__sim_rcv_threshold_bits);
__IO_REG32_BIT(SIM_ENABLE,                0x50018024,__READ_WRITE ,__sim_enable_bits);
__IO_REG32_BIT(SIM_XMT_STATUS,            0x50018028,__READ_WRITE ,__sim_xmt_status_bits);
__IO_REG32_BIT(SIM_RCV_STATUS,            0x5001802C,__READ_WRITE ,__sim_rcv_status_bits);
__IO_REG32_BIT(SIM_INT_MASK,              0x50018030,__READ_WRITE ,__sim_int_mask_bits);
__IO_REG32_BIT(SIM_PORT0_XMT_BUF,         0x50018034,__READ_WRITE ,__sim_port_xmt_buf_bits);
__IO_REG32_BIT(SIM_PORT0_RCV_BUF,         0x50018038,__READ       ,__sim_port_rcv_buf_bits);
__IO_REG32_BIT(SIM_PORT0_DETECT,          0x5001803C,__READ_WRITE ,__sim_port_detect_bits);
__IO_REG32_BIT(SIM_DATA_FORMAT,           0x50018040,__READ_WRITE ,__sim_data_format_bits);
__IO_REG32_BIT(SIM_XMT_THRESHOLD,         0x50018044,__READ_WRITE ,__sim_xmt_threshold_bits);
__IO_REG32_BIT(SIM_GUARD_CNTL,            0x50018048,__READ_WRITE ,__sim_guard_cntl_bits);
__IO_REG32_BIT(SIM_OD_CONFIG,             0x5001804C,__READ_WRITE ,__sim_od_config_bits);
__IO_REG32_BIT(SIM_RESET_CNTL,            0x50018050,__READ_WRITE ,__sim_reset_cntl_bits);
__IO_REG16(    SIM_CHAR_WAIT,             0x50018054,__READ_WRITE );
__IO_REG16(    SIM_GPCNT,                 0x50018058,__READ_WRITE );
__IO_REG32_BIT(SIM_DIVISOR,               0x5001805C,__READ_WRITE ,__sim_divisor_bits);
__IO_REG16(    SIM_BWT,                   0x50018060,__READ_WRITE );
__IO_REG16(    SIM_BGT,                   0x50018064,__READ_WRITE );
__IO_REG16(    SIM_BWT_H,                 0x50018068,__READ_WRITE );

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_1,                    0x43F90000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UTXD_1,                    0x43F90040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UCR1_1,                    0x43F90080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UCR2_1,                    0x43F90084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UCR3_1,                    0x43F90088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UCR4_1,                    0x43F9008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UFCR_1,                    0x43F90090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(USR1_1,                    0x43F90094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(USR2_1,                    0x43F90098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UESC_1,                    0x43F9009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UTIM_1,                    0x43F900A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UBIR_1,                    0x43F900A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UBMR_1,                    0x43F900A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UBRC_1,                    0x43F900AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_1,                   0x43F900B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UTS_1,                     0x43F900B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_2,                    0x43F94000,__READ       ,__urxd_bits);
__IO_REG32_BIT(UTXD_2,                    0x43F94040,__WRITE      ,__utxd_bits);
__IO_REG32_BIT(UCR1_2,                    0x43F94080,__READ_WRITE ,__ucr1_bits);
__IO_REG32_BIT(UCR2_2,                    0x43F94084,__READ_WRITE ,__ucr2_bits);
__IO_REG32_BIT(UCR3_2,                    0x43F94088,__READ_WRITE ,__ucr3_bits);
__IO_REG32_BIT(UCR4_2,                    0x43F9408C,__READ_WRITE ,__ucr4_bits);
__IO_REG32_BIT(UFCR_2,                    0x43F94090,__READ_WRITE ,__ufcr_bits);
__IO_REG32_BIT(USR1_2,                    0x43F94094,__READ_WRITE ,__usr1_bits);
__IO_REG32_BIT(USR2_2,                    0x43F94098,__READ_WRITE ,__usr2_bits);
__IO_REG32_BIT(UESC_2,                    0x43F9409C,__READ_WRITE ,__uesc_bits);
__IO_REG32_BIT(UTIM_2,                    0x43F940A0,__READ_WRITE ,__utim_bits);
__IO_REG32_BIT(UBIR_2,                    0x43F940A4,__READ_WRITE ,__ubir_bits);
__IO_REG32_BIT(UBMR_2,                    0x43F940A8,__READ_WRITE ,__ubmr_bits);
__IO_REG32_BIT(UBRC_2,                    0x43F940AC,__READ_WRITE ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_2,                   0x43F940B0,__READ_WRITE ,__onems_bits);
__IO_REG32_BIT(UTS_2,                     0x43F940B4,__READ_WRITE ,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_3,                    0x5000C000,__READ       ,__urxd_bits);
__IO_REG32_BIT(UTXD_3,                    0x5000C040,__WRITE      ,__utxd_bits);
__IO_REG32_BIT(UCR1_3,                    0x5000C080,__READ_WRITE ,__ucr1_bits);
__IO_REG32_BIT(UCR2_3,                    0x5000C084,__READ_WRITE ,__ucr2_bits);
__IO_REG32_BIT(UCR3_3,                    0x5000C088,__READ_WRITE ,__ucr3_bits);
__IO_REG32_BIT(UCR4_3,                    0x5000C08C,__READ_WRITE ,__ucr4_bits);
__IO_REG32_BIT(UFCR_3,                    0x5000C090,__READ_WRITE ,__ufcr_bits);
__IO_REG32_BIT(USR1_3,                    0x5000C094,__READ_WRITE ,__usr1_bits);
__IO_REG32_BIT(USR2_3,                    0x5000C098,__READ_WRITE ,__usr2_bits);
__IO_REG32_BIT(UESC_3,                    0x5000C09C,__READ_WRITE ,__uesc_bits);
__IO_REG32_BIT(UTIM_3,                    0x5000C0A0,__READ_WRITE ,__utim_bits);
__IO_REG32_BIT(UBIR_3,                    0x5000C0A4,__READ_WRITE ,__ubir_bits);
__IO_REG32_BIT(UBMR_3,                    0x5000C0A8,__READ_WRITE ,__ubmr_bits);
__IO_REG32_BIT(UBRC_3,                    0x5000C0AC,__READ_WRITE ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_3,                   0x5000C0B0,__READ_WRITE ,__onems_bits);
__IO_REG32_BIT(UTS_3,                     0x5000C0B4,__READ_WRITE ,__uts_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_4,                    0x43FB0000,__READ       ,__urxd_bits);
__IO_REG32_BIT(UTXD_4,                    0x43FB0040,__WRITE      ,__utxd_bits);
__IO_REG32_BIT(UCR1_4,                    0x43FB0080,__READ_WRITE ,__ucr1_bits);
__IO_REG32_BIT(UCR2_4,                    0x43FB0084,__READ_WRITE ,__ucr2_bits);
__IO_REG32_BIT(UCR3_4,                    0x43FB0088,__READ_WRITE ,__ucr3_bits);
__IO_REG32_BIT(UCR4_4,                    0x43FB008C,__READ_WRITE ,__ucr4_bits);
__IO_REG32_BIT(UFCR_4,                    0x43FB0090,__READ_WRITE ,__ufcr_bits);
__IO_REG32_BIT(USR1_4,                    0x43FB0094,__READ_WRITE ,__usr1_bits);
__IO_REG32_BIT(USR2_4,                    0x43FB0098,__READ_WRITE ,__usr2_bits);
__IO_REG32_BIT(UESC_4,                    0x43FB009C,__READ_WRITE ,__uesc_bits);
__IO_REG32_BIT(UTIM_4,                    0x43FB00A0,__READ_WRITE ,__utim_bits);
__IO_REG32_BIT(UBIR_4,                    0x43FB00A4,__READ_WRITE ,__ubir_bits);
__IO_REG32_BIT(UBMR_4,                    0x43FB00A8,__READ_WRITE ,__ubmr_bits);
__IO_REG32_BIT(UBRC_4,                    0x43FB00AC,__READ_WRITE ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_4,                   0x43FB00B0,__READ_WRITE ,__onems_bits);
__IO_REG32_BIT(UTS_4,                     0x43FB00B4,__READ_WRITE ,__uts_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_5,                    0x43FB4000,__READ       ,__urxd_bits);
__IO_REG32_BIT(UTXD_5,                    0x43FB4040,__WRITE      ,__utxd_bits);
__IO_REG32_BIT(UCR1_5,                    0x43FB4080,__READ_WRITE ,__ucr1_bits);
__IO_REG32_BIT(UCR2_5,                    0x43FB4084,__READ_WRITE ,__ucr2_bits);
__IO_REG32_BIT(UCR3_5,                    0x43FB4088,__READ_WRITE ,__ucr3_bits);
__IO_REG32_BIT(UCR4_5,                    0x43FB408C,__READ_WRITE ,__ucr4_bits);
__IO_REG32_BIT(UFCR_5,                    0x43FB4090,__READ_WRITE ,__ufcr_bits);
__IO_REG32_BIT(USR1_5,                    0x43FB4094,__READ_WRITE ,__usr1_bits);
__IO_REG32_BIT(USR2_5,                    0x43FB4098,__READ_WRITE ,__usr2_bits);
__IO_REG32_BIT(UESC_5,                    0x43FB409C,__READ_WRITE ,__uesc_bits);
__IO_REG32_BIT(UTIM_5,                    0x43FB40A0,__READ_WRITE ,__utim_bits);
__IO_REG32_BIT(UBIR_5,                    0x43FB40A4,__READ_WRITE ,__ubir_bits);
__IO_REG32_BIT(UBMR_5,                    0x43FB40A8,__READ_WRITE ,__ubmr_bits);
__IO_REG32_BIT(UBRC_5,                    0x43FB40AC,__READ_WRITE ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_5,                   0x43FB40B0,__READ_WRITE ,__onems_bits);
__IO_REG32_BIT(UTS_5,                     0x43FB40B4,__READ_WRITE ,__uts_bits);

/***************************************************************************
 **
 **  USB OTG
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x43F88000,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x43F88004,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x43F88008,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x43F8800C,__READ       ,__uog_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x43F88010,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x43F88014,__READ       ,__uog_hwrxbuf_bits);
__IO_REG8(     UOG_CAPLENGTH,             0x43F88100,__READ       );
__IO_REG16(    UOG_HCIVERSION,            0x43F88102,__READ       );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x43F88104,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x43F88108,__READ       ,__uog_hccparams_bits);
__IO_REG16(    UOG_DCIVERSION,            0x43F88120,__READ       );
__IO_REG32_BIT(UOG_DCCPARAMS,             0x43F88124,__READ       ,__uog_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x43F88140,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x43F88144,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x43F88148,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x43F8814C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x43F88154,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UOG_ASYNCLISTADDR,         0x43F88158,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UOG_BURSTSIZE,             0x43F88160,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x43F88164,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UOG_ULPIVIEW,              0x43F88170,__READ_WRITE ,__uog_ulpiview_bits);
__IO_REG32(    UOG_CFGFLAG,               0x43F88180,__READ       );
__IO_REG32_BIT(UOG_PORTSC1,               0x43F88184,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC2,               0x43F88188,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC3,               0x43F8818C,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC4,               0x43F88190,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC5,               0x43F88194,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC6,               0x43F88198,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC7,               0x43F8819C,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC8,               0x43F881A0,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x43F881A4,__READ_WRITE ,__uog_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x43F881A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x43F881AC,__READ_WRITE ,__uog_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x43F881B0,__READ_WRITE ,__uog_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x43F881B4,__READ_WRITE ,__uog_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x43F881B8,__READ       ,__uog_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x43F881BC,__READ_WRITE ,__uog_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x43F881C0,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x43F881C4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x43F881C8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x43F881CC,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x43F881D0,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x43F881D4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x43F881D8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x43F881DC,__READ_WRITE ,__uog_endptctrl_bits);

__IO_REG32_BIT(UH1_ID,                    0x43F88200,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH1_HWGENERAL,             0x43F88204,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH1_HWHOST,                0x43F88208,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH1_HWTXBUF,               0x43F88210,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH1_HWRXBUF,               0x43F88214,__READ       ,__uog_hwrxbuf_bits);
__IO_REG8(     UH1_CAPLENGTH,             0x43F88300,__READ       );
__IO_REG16(    UH1_HCIVERSION,            0x43F88302,__READ       );
__IO_REG32_BIT(UH1_HCSPARAMS,             0x43F88304,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH1_HCCPARAMS,             0x43F88308,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH1_USBCMD,                0x43F88340,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH1_USBSTS,                0x43F88344,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH1_USBINTR,               0x43F88348,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH1_FRINDEX,               0x43F8834C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH1_PERIODICLISTBASE,      0x43F88354,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH1_ASYNCLISTADDR,         0x43F88358,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH1_BURSTSIZE,             0x43F88360,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH1_TXFILLTUNING,          0x43F88364,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH1_PORTSC1,               0x43F88384,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH1_USBMODE,               0x43F883A8,__READ_WRITE ,__uog_usbmode_bits);

__IO_REG32_BIT(UH2_ID,                    0x43F88400,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH2_HWGENERAL,             0x43F88404,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH2_HWHOST,                0x43F88408,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH2_HWTXBUF,               0x43F88410,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH2_HWRXBUF,               0x43F88414,__READ       ,__uog_hwrxbuf_bits);
__IO_REG8(     UH2_CAPLENGTH,             0x43F88500,__READ       );
__IO_REG16(    UH2_HCIVERSION,            0x43F88502,__READ       );
__IO_REG32_BIT(UH2_HCSPARAMS,             0x43F88504,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH2_HCCPARAMS,             0x43F88508,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH2_USBCMD,                0x43F88540,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH2_USBSTS,                0x43F88544,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH2_USBINTR,               0x43F88548,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH2_FRINDEX,               0x43F8854C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH2_PERIODICLISTBASE,      0x43F88554,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH2_ASYNCLISTADDR,         0x43F88558,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH2_BURSTSIZE,             0x43F88560,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH2_TXFILLTUNING,          0x43F88564,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH2_PORTSC1,               0x43F88584,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH2_USBMODE,               0x43F885A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(USB_CTRL,                  0x43F88600,__READ_WRITE ,__usb_ctrl_bits);
__IO_REG8_BIT( USB_OTG_MIRROR,            0x43F88604,__READ_WRITE ,__usb_otg_mirror_bits);

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPITCR1,                   0x53F94000,__READ_WRITE ,__epitcr_bits);
__IO_REG32_BIT(EPITSR1,                   0x53F94004,__READ_WRITE ,__epitsr_bits);
__IO_REG32(    EPITLR1,                   0x53F94008,__READ_WRITE );
__IO_REG32(    EPITCMPR1,                 0x53F9400C,__READ_WRITE );
__IO_REG32(    EPITCNT1,                  0x53F94010,__READ       );

/***************************************************************************
 **
 **  EPIT2
 **
 ***************************************************************************/
__IO_REG32_BIT(EPITCR2,                   0x53F98000,__READ_WRITE ,__epitcr_bits);
__IO_REG32_BIT(EPITSR2,                   0x53F98004,__READ_WRITE ,__epitsr_bits);
__IO_REG32(    EPITLR2,                   0x53F98008,__READ_WRITE );
__IO_REG32(    EPITCMPR2,                 0x53F9800C,__READ_WRITE );
__IO_REG32(    EPITCNT2,                  0x53F98010,__READ       );

/***************************************************************************
 **
 **  GPT1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTCR,                     0x53F90000,__READ_WRITE ,__gptcr_bits);
__IO_REG32_BIT(GPTPR,                     0x53F90004,__READ_WRITE ,__gptpr_bits);
__IO_REG32_BIT(GPTSR,                     0x53F90008,__READ_WRITE ,__gptsr_bits);
__IO_REG32_BIT(GPTIR,                     0x53F9000C,__READ_WRITE ,__gptir_bits);
__IO_REG32(    GPTOCR1,                   0x53F90010,__READ_WRITE );
__IO_REG32(    GPTOCR2,                   0x53F90014,__READ_WRITE );
__IO_REG32(    GPTOCR3,                   0x53F90018,__READ_WRITE );
__IO_REG32(    GPTICR1,                   0x53F9001C,__READ       );
__IO_REG32(    GPTICR2,                   0x53F90020,__READ       );
__IO_REG32(    GPTCNT,                    0x53F90024,__READ       );

/***************************************************************************
 **
 **  PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR,                     0x53FE0000,__READ_WRITE ,__pwmcr_bits);
__IO_REG32_BIT(PWMSR,                     0x53FE0004,__READ_WRITE ,__pwmsr_bits);
__IO_REG32_BIT(PWMIR,                     0x53FE0008,__READ_WRITE ,__pwmir_bits);
__IO_REG32_BIT(PWMSAR,                    0x53FE000C,__READ_WRITE ,__pwmsar_bits);
__IO_REG32_BIT(PWMPR,                     0x53FE0010,__READ_WRITE ,__pwmpr_bits);
__IO_REG32_BIT(PWMCNR,                    0x53FE0014,__READ       ,__pwmcnr_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(HOURMIN,                   0x53FD8000,__READ_WRITE ,__hourmin_bits);
__IO_REG32_BIT(SECONDS,                   0x53FD8004,__READ_WRITE ,__seconds_bits);
__IO_REG32_BIT(ALRM_HM,                   0x53FD8008,__READ_WRITE ,__hourmin_bits);
__IO_REG32_BIT(ALRM_SEC,                  0x53FD800C,__READ_WRITE ,__seconds_bits);
__IO_REG32_BIT(RCCTL,                     0x53FD8010,__READ_WRITE ,__rcctl_bits);
__IO_REG32_BIT(RTCISR,                    0x53FD8014,__READ_WRITE ,__rtcisr_bits);
__IO_REG32_BIT(RTCIENR,                   0x53FD8018,__READ_WRITE ,__rtcisr_bits);
__IO_REG32_BIT(STPWCH,                    0x53FD801C,__READ_WRITE ,__stpwch_bits);
__IO_REG32_BIT(DAYR,                      0x53FD8020,__READ_WRITE ,__dayr_bits);
__IO_REG32_BIT(DAYALARM,                  0x53FD8024,__READ_WRITE ,__dayalarm_bits);

/***************************************************************************
 **
 **  WDOG
 **
 ***************************************************************************/
__IO_REG16_BIT(WCR,                       0x53FDC000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WSR,                       0x53FDC002,__READ_WRITE );
__IO_REG16_BIT(WRSR,                      0x53FDC004,__READ       ,__wrsr_bits);

/***************************************************************************
 **
 **  AIPS A
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPSA_MPR_1,               0x43F00000,__READ_WRITE ,__aips_mpr_1_bits);
__IO_REG32_BIT(AIPSA_MPR_2,               0x43F00004,__READ_WRITE ,__aips_mpr_2_bits);
__IO_REG32_BIT(AIPSA_PACR_1,              0x43F00020,__READ_WRITE ,__aips_pacr_1_bits);
__IO_REG32_BIT(AIPSA_PACR_2,              0x43F00024,__READ_WRITE ,__aips_pacr_2_bits);
__IO_REG32_BIT(AIPSA_PACR_3,              0x43F00028,__READ_WRITE ,__aips_pacr_3_bits);
__IO_REG32_BIT(AIPSA_PACR_4,              0x43F0002C,__READ_WRITE ,__aips_pacr_4_bits);
__IO_REG32_BIT(AIPSA_OPACR_1,             0x43F00040,__READ_WRITE ,__aips_pacr_1_bits);
__IO_REG32_BIT(AIPSA_OPACR_2,             0x43F00044,__READ_WRITE ,__aips_pacr_2_bits);
__IO_REG32_BIT(AIPSA_OPACR_3,             0x43F00048,__READ_WRITE ,__aips_pacr_3_bits);
__IO_REG32_BIT(AIPSA_OPACR_4,             0x43F0004C,__READ_WRITE ,__aips_pacr_4_bits);
__IO_REG32_BIT(AIPSA_OPACR_5,             0x43F00050,__READ_WRITE ,__aips_pacr_5_bits);

/***************************************************************************
 **
 **  AIPS B
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPSB_MPR_1,               0x53F00000,__READ_WRITE ,__aips_mpr_1_bits);
__IO_REG32_BIT(AIPSB_MPR_2,               0x53F00004,__READ_WRITE ,__aips_mpr_2_bits);
__IO_REG32_BIT(AIPSB_PACR_1,              0x53F00020,__READ_WRITE ,__aips_pacr_1_bits);
__IO_REG32_BIT(AIPSB_PACR_2,              0x53F00024,__READ_WRITE ,__aips_pacr_2_bits);
__IO_REG32_BIT(AIPSB_PACR_3,              0x53F00028,__READ_WRITE ,__aips_pacr_3_bits);
__IO_REG32_BIT(AIPSB_PACR_4,              0x53F0002C,__READ_WRITE ,__aips_pacr_4_bits);
__IO_REG32_BIT(AIPSB_OPACR_1,             0x53F00040,__READ_WRITE ,__aips_pacr_1_bits);
__IO_REG32_BIT(AIPSB_OPACR_2,             0x53F00044,__READ_WRITE ,__aips_pacr_2_bits);
__IO_REG32_BIT(AIPSB_OPACR_3,             0x53F00048,__READ_WRITE ,__aips_pacr_3_bits);
__IO_REG32_BIT(AIPSB_OPACR_4,             0x53F0004C,__READ_WRITE ,__aips_pacr_4_bits);
__IO_REG32_BIT(AIPSB_OPACR_5,             0x53F00050,__READ_WRITE ,__aips_pacr_5_bits);

/***************************************************************************
 **
 **  MAX
 **
 ***************************************************************************/
__IO_REG32_BIT(MPR0,                      0x43F04000,__READ_WRITE ,__mpr_bits);
__IO_REG32_BIT(SGPCR0,                    0x43F04010,__READ_WRITE ,__sgpcr_bits);
__IO_REG32_BIT(MPR1,                      0x43F04100,__READ_WRITE ,__mpr_bits);
__IO_REG32_BIT(SGPCR1,                    0x43F04110,__READ_WRITE ,__sgpcr_bits);
__IO_REG32_BIT(MPR2,                      0x43F04200,__READ_WRITE ,__mpr_bits);
__IO_REG32_BIT(SGPCR2,                    0x43F04210,__READ_WRITE ,__sgpcr_bits);
__IO_REG32_BIT(MPR3,                      0x43F04300,__READ_WRITE ,__mpr_bits);
__IO_REG32_BIT(SGPCR3,                    0x43F04310,__READ_WRITE ,__sgpcr_bits);
__IO_REG32_BIT(MPR4,                      0x43F04400,__READ_WRITE ,__mpr_bits);
__IO_REG32_BIT(SGPCR4,                    0x43F04410,__READ_WRITE ,__sgpcr_bits);
__IO_REG32_BIT(MGPCR0,                    0x43F04800,__READ_WRITE ,__mgpcr_bits);
__IO_REG32_BIT(MGPCR1,                    0x43F04900,__READ_WRITE ,__mgpcr_bits);
__IO_REG32_BIT(MGPCR2,                    0x43F04A00,__READ_WRITE ,__mgpcr_bits);
__IO_REG32_BIT(MGPCR3,                    0x43F04B00,__READ_WRITE ,__mgpcr_bits);
__IO_REG32_BIT(MGPCR4,                    0x43F04C00,__READ_WRITE ,__mgpcr_bits);
__IO_REG32_BIT(MGPCR5,                    0x43F04D00,__READ_WRITE ,__mgpcr_bits);

/***************************************************************************
 **
 **  SDMA
 **
 ***************************************************************************/
__IO_REG32(    SDMA_MC0PTR,               0x53FD4000,__READ_WRITE );
__IO_REG32_BIT(SDMA_INTR,                 0x53FD4004,__READ_WRITE ,__sdma_intr_bits);
__IO_REG32_BIT(SDMA_STOP_STAT,            0x53FD4008,__READ       ,__sdma_stop_stat_bits);
__IO_REG32_BIT(SDMA_HSTART,               0x53FD400C,__READ_WRITE ,__sdma_hstart_bits);
__IO_REG32_BIT(SDMA_EVTOVR,               0x53FD4010,__READ_WRITE ,__sdma_evtovr_bits);
__IO_REG32_BIT(SDMA_DSPOVR,               0x53FD4014,__READ_WRITE ,__sdma_dspovr_bits);
__IO_REG32_BIT(SDMA_HOSTOVR,              0x53FD4018,__READ_WRITE ,__sdma_hostovr_bits);
__IO_REG32_BIT(SDMA_EVTPEND,              0x53FD401C,__READ       ,__sdma_evtpend_bits);
__IO_REG32_BIT(SDMA_RESET,                0x53FD4024,__READ       ,__sdma_reset_bits);
__IO_REG32_BIT(SDMA_EVTERR,               0x53FD4028,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_INTRMASK,             0x53FD402C,__READ_WRITE ,__sdma_intrmask_bits);
__IO_REG32_BIT(SDMA_PSW,                  0x53FD4030,__READ       ,__sdma_psw_bits);
__IO_REG32_BIT(SDMA_EVTERRDBG,            0x53FD4034,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_CONFIG,               0x53FD4038,__READ_WRITE ,__sdma_config_bits);
__IO_REG32_BIT(SDMA_ONCE_ENB,             0x53FD4040,__READ_WRITE ,__sdma_once_enb_bits);
__IO_REG32(    SDMA_ONCE_DATA,            0x53FD4044,__READ_WRITE );
__IO_REG32_BIT(SDMA_ONCE_INSTR,           0x53FD4048,__READ_WRITE ,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMA_ONCE_STAT,            0x53FD404C,__READ       ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMA_ONCE_CMD,             0x53FD4050,__READ_WRITE ,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR,           0x53FD4054,__READ       ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMA_ILLINSTADDR,          0x53FD4058,__READ_WRITE ,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMA_CHN0ADDR,             0x53FD405C,__READ_WRITE ,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF1,          0x53FD4070,__READ_WRITE ,__sdma_xtrig1_conf_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF2,          0x53FD4074,__READ_WRITE ,__sdma_xtrig2_conf_bits);
__IO_REG32_BIT(SDMA_CHNPRI0,              0x53FD4100,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI1,              0x53FD4104,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI2,              0x53FD4108,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI3,              0x53FD410C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI4,              0x53FD4110,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI5,              0x53FD4114,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI6,              0x53FD4118,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI7,              0x53FD411C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI8,              0x53FD4120,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI9,              0x53FD4124,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI10,             0x53FD4128,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI11,             0x53FD412C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI12,             0x53FD4130,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI13,             0x53FD4134,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI14,             0x53FD4138,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI15,             0x53FD413C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI16,             0x53FD4140,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI17,             0x53FD4144,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI18,             0x53FD4148,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI19,             0x53FD414C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI20,             0x53FD4150,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI21,             0x53FD4154,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI22,             0x53FD4158,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI23,             0x53FD415C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI24,             0x53FD4160,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI25,             0x53FD4164,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI26,             0x53FD4168,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI27,             0x53FD416C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI28,             0x53FD4170,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI29,             0x53FD4174,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI30,             0x53FD4178,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI31,             0x53FD417C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32(    SDMA_CHNENBL0,             0x53FD4080,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL1,             0x53FD4084,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL2,             0x53FD4088,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL3,             0x53FD408C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL4,             0x53FD4090,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL5,             0x53FD4094,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL6,             0x53FD4098,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL7,             0x53FD409C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL8,             0x53FD40A0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL9,             0x53FD40A4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL10,            0x53FD40A8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL11,            0x53FD40AC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL12,            0x53FD40B0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL13,            0x53FD40B4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL14,            0x53FD40B8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL15,            0x53FD40BC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL16,            0x53FD40C0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL17,            0x53FD40C4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL18,            0x53FD40C8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL19,            0x53FD40CC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL20,            0x53FD40D0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL21,            0x53FD40D4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL22,            0x53FD40D8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL23,            0x53FD40DC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL24,            0x53FD40E0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL25,            0x53FD40E4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL26,            0x53FD40E8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL27,            0x53FD40EC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL28,            0x53FD40F0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL29,            0x53FD40F4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL30,            0x53FD40F8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL31,            0x53FD40FC,__READ_WRITE );

/***************************************************************************
 **
 **  SPBA
 **
 ***************************************************************************/
__IO_REG32_BIT(SPBA_PRR0,                 0x5003C000,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR1,                 0x5003C004,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR2,                 0x5003C008,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR3,                 0x5003C00C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR4,                 0x5003C010,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR5,                 0x5003C014,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR6,                 0x5003C018,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR7,                 0x5003C01C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR8,                 0x5003C020,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR9,                 0x5003C024,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR10,                0x5003C028,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR11,                0x5003C02C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR12,                0x5003C030,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR13,                0x5003C034,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR14,                0x5003C038,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR15,                0x5003C03C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR16,                0x5003C040,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR17,                0x5003C044,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR18,                0x5003C048,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR19,                0x5003C04C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR20,                0x5003C050,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR21,                0x5003C054,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR22,                0x5003C058,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR23,                0x5003C05C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR24,                0x5003C060,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR25,                0x5003C064,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR26,                0x5003C068,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR27,                0x5003C06C,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR28,                0x5003C070,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR29,                0x5003C074,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR30,                0x5003C078,__READ_WRITE ,__spba_prr_bits);
__IO_REG32_BIT(SPBA_PRR31,                0x5003C07C,__READ_WRITE ,__spba_prr_bits);

/***************************************************************************
 **
 **  AUDMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(AUDMUX_PTCR1,              0x53FC4000,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR1,              0x53FC4004,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR2,              0x53FC4008,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR2,              0x53FC400C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR3,              0x53FC4010,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR3,              0x53FC4014,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR4,              0x53FC4018,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR4,              0x53FC401C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR5,              0x53FC4020,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR5,              0x53FC4024,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR6,              0x53FC4028,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR6,              0x53FC402C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR7,              0x53FC4030,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR7,              0x53FC4034,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_CNMCR,              0x53FC4038,__READ_WRITE ,__audmux_cnmcr_bits);

/***************************************************************************
 **
 **  IPU
 **
 ***************************************************************************/
__IO_REG32_BIT(IPU_CONF,                  0x53FC0000,__READ_WRITE ,__ipu_conf_bits);
__IO_REG32_BIT(IPU_CHA_BUF0_RDY,          0x53FC0004,__READ_WRITE ,__ipu_cha_buf_rdy_bits);
__IO_REG32_BIT(IPU_CHA_BUF1_RDY,          0x53FC0008,__READ_WRITE ,__ipu_cha_buf_rdy_bits);
__IO_REG32_BIT(IPU_CHA_DB_MODE_SEL,       0x53FC000C,__READ_WRITE ,__ipu_cha_db_mode_sel_bits);
__IO_REG32_BIT(IPU_CHA_CUR_BUF,           0x53FC0010,__READ_WRITE ,__ipu_cha_cur_buf_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW,          0x53FC0014,__READ_WRITE ,__ipu_fs_proc_flow_bits);
__IO_REG32_BIT(IPU_FS_DISP_FLOW,          0x53FC0018,__READ_WRITE ,__ipu_fs_disp_flow_bits);
__IO_REG32_BIT(IPU_TASKS_STAT,            0x53FC001C,__READ       ,__ipu_tasks_stat_bits);
__IO_REG32_BIT(IPU_IMA_ADDR,              0x53FC0020,__READ_WRITE ,__ipu_ima_addr_bits);
__IO_REG32(    IPU_IMA_DATA,              0x53FC0024,__READ_WRITE );
__IO_REG32_BIT(IPU_INT_CTRL_1,            0x53FC0028,__READ_WRITE ,__ipu_int_ctrl_1_bits);
__IO_REG32_BIT(IPU_INT_CTRL_2,            0x53FC002C,__READ_WRITE ,__ipu_int_ctrl_2_bits);
__IO_REG32_BIT(IPU_INT_CTRL_3,            0x53FC0030,__READ_WRITE ,__ipu_int_ctrl_3_bits);
__IO_REG32_BIT(IPU_INT_CTRL_4,            0x53FC0034,__READ_WRITE ,__ipu_int_ctrl_4_bits);
__IO_REG32_BIT(IPU_INT_CTRL_5,            0x53FC0038,__READ_WRITE ,__ipu_int_ctrl_5_bits);
__IO_REG32_BIT(IPU_INT_STAT_1,            0x53FC003C,__READ_WRITE ,__ipu_int_stat_1_bits);
__IO_REG32_BIT(IPU_INT_STAT_2,            0x53FC0040,__READ_WRITE ,__ipu_int_stat_2_bits);
__IO_REG32_BIT(IPU_INT_STAT_3,            0x53FC0044,__READ_WRITE ,__ipu_int_stat_3_bits);
__IO_REG32_BIT(IPU_INT_STAT_4,            0x53FC0048,__READ_WRITE ,__ipu_int_stat_4_bits);
__IO_REG32_BIT(IPU_INT_STAT_5,            0x53FC004C,__READ_WRITE ,__ipu_int_stat_5_bits);
__IO_REG32_BIT(IPU_BRK_CTRL_1,            0x53FC0050,__READ_WRITE ,__ipu_brk_ctrl_1_bits);
__IO_REG32_BIT(IPU_BRK_CTRL_2,            0x53FC0054,__READ_WRITE ,__ipu_brk_ctrl_2_bits);
__IO_REG32_BIT(IPU_BRK_STAT,              0x53FC0058,__READ       ,__ipu_brk_stat_bits);
__IO_REG32_BIT(IPU_DIAGB_CTRL,            0x53FC005C,__READ_WRITE ,__ipu_diagb_ctrl_bits);

/***************************************************************************
 **
 **  CSI
 **
 ***************************************************************************/
__IO_REG32_BIT(CSI_SENS_CONF,             0x53FC0060,__READ_WRITE ,__csi_sens_conf_bits);
__IO_REG32_BIT(CSI_SENS_FRM_SIZE,         0x53FC0064,__READ_WRITE ,__csi_sens_frm_size_bits);
__IO_REG32_BIT(CSI_ACT_FRM_SIZE,          0x53FC0068,__READ_WRITE ,__csi_act_frm_size_bits);
__IO_REG32_BIT(CSI_OUT_FRM_CTRL,          0x53FC006C,__READ_WRITE ,__csi_out_frm_ctrl_bits);
__IO_REG32_BIT(CSI_TST_CTRL,              0x53FC0070,__READ_WRITE ,__csi_tst_ctrl_bits);
__IO_REG32_BIT(CSI_CCIR_CODE_1,           0x53FC0074,__READ_WRITE ,__csi_ccir_code_1_bits);
__IO_REG32_BIT(CSI_CCIR_CODE_2,           0x53FC0078,__READ_WRITE ,__csi_ccir_code_2_bits);
__IO_REG32_BIT(CSI_CCIR_CODE_3,           0x53FC007C,__READ_WRITE ,__csi_ccir_code_3_bits);
__IO_REG32_BIT(CSI_FLASH_STROBE_1,        0x53FC0080,__READ_WRITE ,__csi_flash_strobe_1_bits);
__IO_REG32_BIT(CSI_FLASH_STROBE_2,        0x53FC0084,__READ_WRITE ,__csi_flash_strobe_2_bits);

/***************************************************************************
 **
 **  IC
 **
 ***************************************************************************/
__IO_REG32_BIT(IC_CONF,                   0x53FC0088,__READ_WRITE ,__ic_conf_bits);
__IO_REG32_BIT(IC_PRP_ENC_RSC,            0x53FC008C,__READ_WRITE ,__ic_prp_enc_rsc_bits);
__IO_REG32_BIT(IC_PRP_VF_RSC,             0x53FC0090,__READ_WRITE ,__ic_prp_vf_rsc_bits);
__IO_REG32_BIT(IC_PP_RSC,                 0x53FC0094,__READ_WRITE ,__ic_pp_rsc_bits);
__IO_REG32_BIT(IC_CMBP_1,                 0x53FC0098,__READ_WRITE ,__ic_cmbp_1_bits);
__IO_REG32_BIT(IC_CMBP_2,                 0x53FC009C,__READ_WRITE ,__ic_cmbp_2_bits);

/***************************************************************************
 **
 **  PF
 **
 ***************************************************************************/
__IO_REG32_BIT(PF_CONF,                   0x53FC00A0,__READ_WRITE ,__pf_conf_bits);

/***************************************************************************
 **
 **  IDMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(IDMAC_CONF,                0x53FC00A4,__READ_WRITE ,__idmac_conf_bits);
__IO_REG32_BIT(IDMAC_CHA_EN,              0x53FC00A8,__READ_WRITE ,__idmac_cha_en_bits);
__IO_REG32_BIT(IDMAC_CHA_PRI,             0x53FC00AC,__READ_WRITE ,__idmac_cha_pri_bits);
__IO_REG32_BIT(IDMAC_CHA_BUSY,            0x53FC00B0,__READ_WRITE ,__idmac_cha_busy_bits);

/***************************************************************************
 **
 **  IDMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(SDC_COM_CONF,              0x53FC00B4,__READ_WRITE ,__sdc_com_conf_bits);
__IO_REG32_BIT(SDC_GRAPH_WIND_CTRL,       0x53FC00B8,__READ_WRITE ,__sdc_graph_wind_ctrl_bits);
__IO_REG32_BIT(SDC_FG_POS,                0x53FC00BC,__READ_WRITE ,__sdc_fg_pos_bits);
__IO_REG32_BIT(SDC_BG_POS,                0x53FC00C0,__READ_WRITE ,__sdc_bg_pos_bits);
__IO_REG32_BIT(SDC_CUR_POS,               0x53FC00C4,__READ_WRITE ,__sdc_cur_pos_bits);
__IO_REG32_BIT(SDC_CUR_BLINK_PWM_CTRL,    0x53FC00C8,__READ_WRITE ,__sdc_cur_blink_pwm_ctrl_bits);
__IO_REG32_BIT(SDC_CUR_MAP,               0x53FC00CC,__READ_WRITE ,__sdc_cur_map_bits);
__IO_REG32_BIT(SDC_HOR_CONF,              0x53FC00D0,__READ_WRITE ,__sdc_hor_conf_bits);
__IO_REG32_BIT(SDC_VER_CONF,              0x53FC00D4,__READ_WRITE ,__sdc_ver_conf_bits);
__IO_REG32_BIT(SDC_SHARP_CONF_1,          0x53FC00D8,__READ_WRITE ,__sdc_sharp_conf_1_bits);
__IO_REG32_BIT(SDC_SHARP_CONF_2,          0x53FC00DC,__READ_WRITE ,__sdc_sharp_conf_2_bits);

/***************************************************************************
 **
 **  AsynchDC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CONF,                  0x53FC00E0,__READ_WRITE ,__adc_conf_bits);
__IO_REG32_BIT(ADC_SYSCHA1_SA,            0x53FC00E4,__READ_WRITE ,__adc_syscha_sa_bits);
__IO_REG32_BIT(ADC_SYSCHA2_SA,            0x53FC00E8,__READ_WRITE ,__adc_syscha_sa_bits);
__IO_REG32_BIT(ADC_PRPCHAN_SA,            0x53FC00EC,__READ_WRITE ,__adc_prpchan_sa_bits);
__IO_REG32_BIT(ADC_PPCHAN_SA,             0x53FC00F0,__READ_WRITE ,__adc_ppchan_sa_bits);
__IO_REG32_BIT(ADC_DISP0_CONF,            0x53FC00F4,__READ_WRITE ,__adc_disp_conf_bits);
__IO_REG32_BIT(ADC_DISP0_RD_AP,           0x53FC00F8,__READ_WRITE ,__adc_disp_rd_ap_bits);
__IO_REG32_BIT(ADC_DISP0_RDM,             0x53FC00FC,__READ_WRITE ,__adc_disp_rdm_bits);
__IO_REG32_BIT(ADC_DISP0_SS,              0x53FC0100,__READ_WRITE ,__adc_disp_ss_bits);
__IO_REG32_BIT(ADC_DISP1_CONF,            0x53FC0104,__READ_WRITE ,__adc_disp_conf_bits);
__IO_REG32_BIT(ADC_DISP1_RD_AP,           0x53FC0108,__READ_WRITE ,__adc_disp_rd_ap_bits);
__IO_REG32_BIT(ADC_DISP1_RDM,             0x53FC010C,__READ_WRITE ,__adc_disp_rdm_bits);
__IO_REG32_BIT(ADC_DISP12_SS,             0x53FC0110,__READ_WRITE ,__adc_disp_ss_bits);
__IO_REG32_BIT(ADC_DISP2_CONF,            0x53FC0114,__READ_WRITE ,__adc_disp_conf_bits);
__IO_REG32_BIT(ADC_DISP2_RD_AP,           0x53FC0118,__READ_WRITE ,__adc_disp_rd_ap_bits);
__IO_REG32_BIT(ADC_DISP2_RDM,             0x53FC011C,__READ_WRITE ,__adc_disp_rdm_bits);
__IO_REG32_BIT(ADC_DISP_VSYNC,            0x53FC0120,__READ_WRITE ,__adc_disp_vsync_bits);

/***************************************************************************
 **
 **  DI
 **
 ***************************************************************************/
__IO_REG32_BIT(DI_DISP_IF_CONF,           0x53FC0124,__READ_WRITE ,__di_disp_if_conf_bits);
__IO_REG32_BIT(DI_DISP_SIG_POL,           0x53FC0128,__READ_WRITE ,__di_disp_sig_pol_bits);
__IO_REG32_BIT(DI_SER_DISP1_CONF,         0x53FC012C,__READ_WRITE ,__di_ser_disp_conf_bits);
__IO_REG32_BIT(DI_SER_DISP2_CONF,         0x53FC0130,__READ_WRITE ,__di_ser_disp_conf_bits);
__IO_REG32_BIT(DI_HSP_CLK_PER,            0x53FC0134,__READ_WRITE ,__di_hsp_clk_per_bits);
__IO_REG32_BIT(DI_DISP0_TIME_CONF_1,      0x53FC0138,__READ_WRITE ,__di_disp_time_conf_1_bits);
__IO_REG32_BIT(DI_DISP0_TIME_CONF_2,      0x53FC013C,__READ_WRITE ,__di_disp_time_conf_2_bits);
__IO_REG32_BIT(DI_DISP0_TIME_CONF_3,      0x53FC0140,__READ_WRITE ,__di_disp_time_conf_3_bits);
__IO_REG32_BIT(DI_DISP1_TIME_CONF_1,      0x53FC0144,__READ_WRITE ,__di_disp_time_conf_1_bits);
__IO_REG32_BIT(DI_DISP1_TIME_CONF_2,      0x53FC0148,__READ_WRITE ,__di_disp_time_conf_2_bits);
__IO_REG32_BIT(DI_DISP1_TIME_CONF_3,      0x53FC014C,__READ_WRITE ,__di_disp_time_conf_3_bits);
__IO_REG32_BIT(DI_DISP2_TIME_CONF_1,      0x53FC0150,__READ_WRITE ,__di_disp_time_conf_1_bits);
__IO_REG32_BIT(DI_DISP2_TIME_CONF_2,      0x53FC0154,__READ_WRITE ,__di_disp_time_conf_2_bits);
__IO_REG32_BIT(DI_DISP2_TIME_CONF_3,      0x53FC0158,__READ_WRITE ,__di_disp2_time_conf_3_bits);
__IO_REG32_BIT(DI_DISP3_TIME_CONF,        0x53FC015C,__READ_WRITE ,__di_disp_time_conf_1_bits);
__IO_REG32_BIT(DI_DISP0_DB0_MAP,          0x53FC0160,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP0_DB1_MAP,          0x53FC0164,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP0_DB2_MAP,          0x53FC0168,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP0_CB0_MAP,          0x53FC016C,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP0_CB1_MAP,          0x53FC0170,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP0_CB2_MAP,          0x53FC0174,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_DB0_MAP,          0x53FC0178,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_DB1_MAP,          0x53FC017C,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_DB2_MAP,          0x53FC0180,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_CB0_MAP,          0x53FC0184,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_CB1_MAP,          0x53FC0188,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP1_CB2_MAP,          0x53FC018C,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_DB0_MAP,          0x53FC0190,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_DB1_MAP,          0x53FC0194,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_DB2_MAP,          0x53FC0198,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_CB0_MAP,          0x53FC019C,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_CB1_MAP,          0x53FC01A0,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP2_CB2_MAP,          0x53FC01A4,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP3_B0_MAP,           0x53FC01A8,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP3_B1_MAP,           0x53FC01AC,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP3_B2_MAP,           0x53FC01B0,__READ_WRITE ,__di_disp_map_bits);
__IO_REG32_BIT(DI_DISP_ACC_CC,            0x53FC01B4,__READ_WRITE ,__di_disp_acc_cc_bits);
__IO_REG32_BIT(DI_DISP_LLA_CONF,          0x53FC01B8,__READ_WRITE ,__di_disp_lla_conf_bits);
__IO_REG32_BIT(DI_DISP_LLA_DATA,          0x53FC01BC,__READ_WRITE ,__di_disp_lla_data_bits);

/***************************************************************************
 **
 **  SSI1
 **
 ***************************************************************************/
__IO_REG32(    STX0_1,                    0x43FA0000,__READ_WRITE );
__IO_REG32(    STX1_1,                    0x43FA0004,__READ_WRITE );
__IO_REG32(    STR0_1,                    0x43FA0008,__READ       );
__IO_REG32(    STR1_1,                    0x43FA000C,__READ       );
__IO_REG32_BIT(SCR1,                      0x43FA0010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SISR1,                     0x43FA0014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SIER1,                     0x43FA0018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(STCR1,                     0x43FA001C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SRCR1,                     0x43FA0020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(STCCR1,                    0x43FA0024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SRCCR1,                    0x43FA0028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SFCSR1,                    0x43FA002C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SACNT1,                    0x43FA0038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SACADD1,                   0x43FA003C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SACDAT1,                   0x43FA0040,__READ_WRITE ,__sacdat_bits);
__IO_REG16(    SATAG1,                    0x43FA0044,__READ_WRITE );
__IO_REG32_BIT(STMSK1,                    0x43FA0048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SRMSK1,                    0x43FA004C,__READ_WRITE ,__srmsk_bits);

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    STX0_2,                    0x50014000,__READ_WRITE );
__IO_REG32(    STX1_2,                    0x50014004,__READ_WRITE );
__IO_REG32(    STR0_2,                    0x50014008,__READ       );
__IO_REG32(    STR1_2,                    0x5001400C,__READ       );
__IO_REG32_BIT(SCR2,                      0x50014010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SISR2,                     0x50014014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SIER2,                     0x50014018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(STCR2,                     0x5001401C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SRCR2,                     0x50014020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(STCCR2,                    0x50014024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SRCCR2,                    0x50014028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SFCSR2,                    0x5001402C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SACNT2,                    0x50014038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SACADD2,                   0x5001403C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SACDAT2,                   0x50014040,__READ_WRITE ,__sacdat_bits);
__IO_REG16(    SATAG2,                    0x50014044,__READ_WRITE );
__IO_REG32_BIT(STMSK2,                    0x50014048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SRMSK2,                    0x5001404C,__READ_WRITE ,__srmsk_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV        0x00  /* Reset                                       */
#define UNDEFV        0x04  /* Undefined instruction                       */
#define SWIV          0x08  /* Software interrupt                          */
#define PABORTV       0x0c  /* Prefetch abort                              */
#define DABORTV       0x10  /* Data abort                                  */
#define IRQV          0x18  /* Normal interrupt                            */
#define FIQV          0x1c  /* Fast interrupt                              */

/***************************************************************************
 **
 **   MCIMX31 interrupt sources
 **
 ***************************************************************************/
#define INT_I2C3               3              /* Inter-Integrated Circuit 3*/
#define INT_I2C2               4              /* Inter-Integrated Circuit 2*/
#define INT_MPEG4_ENCODER      5              /* MPEG-4 Encoder*/
#define INT_RTIC               6              /* HASH error has occurred, or the RTIC has completed hashing*/
#define INT_FIR                7              /* Fast Infrared Controller*/
#define INT_SDHC2              8              /* MultiMedia/Secure Data Host Controller 2*/
#define INT_SDHC1              9              /* MultiMedia/Secure Data Host Controller 1*/
#define INT_I2C1               10             /* Inter-Integrated Circuit 2*/
#define INT_SSI2               11             /* Synchronous Serial Interface 2*/
#define INT_SSI1               12             /* Synchronous Serial Interface 1*/
#define INT_CSPI2              13             /* Configurable Serial Peripheral Interface 2*/
#define INT_CSPI1              14             /* Configurable Serial Peripheral Interface 1*/
#define INT_ATA                15             /* Hard Drive (ATA) Controller*/
#define INT_MBX_R_S            16             /* Graphic accelerator*/
#define INT_CSPI3              17             /* Configurable Serial Peripheral Interface 3*/
#define INT_UART3              18             /* UART3*/
#define INT_IIM                19             /* IC Identification*/
#define INT_SIM1               20             /* Subscriber Identification Module*/
#define INT_SIM2               21             /* Subscriber Identification Module*/
#define INT_RNGA               22             /* Random Number Generator Accelerator*/
#define INT_EVTMON             23             /* OR of evtmon_interrupt,pmu_irq*/
#define INT_KPP                24             /* Keyboard Pad Port*/
#define INT_RTC                25             /* Real Time Clock*/
#define INT_PWM                26             /* Pulse Width Modulator*/
#define INT_EPIT2              27             /* Enhanced Periodic Timer 2*/
#define INT_EPIT1              28             /* Enhanced Periodic Timer 1*/
#define INT_GPT                29             /* General Purpose Timer*/
#define INT_POWER_FAULT        30             /* Power fault*/
#define INT_CCM_DVFS           31             /**/
#define INT_UART2              32             /* UART2*/
#define INT_NANDFC             33             /* NAND Flash Controller*/
#define INT_SDMA               34             /* Smart Direct Memory Access*/
#define INT_USB_HOST1          35             /* USB Host 1*/
#define INT_USB_HOST2          36             /* USB Host 2*/
#define INT_USB_OTG            37             /* USB OTG*/
#define INT_MSHC1              39             /* Memory Stick Host Controller 1*/
#define INT_MSHC2              40             /* Memory Stick Host Controller 2*/
#define INT_IPU_ERR            41             /* Image Processing Unit error*/
#define INT_IPU                42             /* IPU general interrupt*/
#define INT_UART1              45             /* UART1*/
#define INT_UART5              46             /* UART4*/
#define INT_UART6              47             /* UART5*/
#define INT_ECT                48             /* AND of oct_irq_b[1:0]*/
#define INT_SCM                49             /* SCM interrupt*/
#define INT_SMN                50             /* SMN interrupt*/
#define INT_GPIO2              51             /* General Purpose I/O 2*/
#define INT_GPIO1              52             /* General Purpose I/O 1*/
#define INT_CCM                53             /* Clock controller*/
#define INT_PCMCIA             54             /* PCMCIA module*/
#define INT_WDOG               55             /* Watch Dog Timer*/
#define INT_GPIO3              56             /* General Purpose I/O 3*/
#define INT_EXT_PM             58             /* External (power management)*/
#define INT_EXT_TEMP           59             /* External (Temper)*/
#define INT_EXT_SENSOR1        60             /* External (sensor)*/
#define INT_EXT_SENSOR2        61             /* External (sensor)*/
#define INT_EXT_WDT            62             /* External (WDOG)*/
#define INT_EXT_TV             63             /* External (TV)*/

#endif    /* __MCIMX31_H */
