/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MCIMX35
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIMX35_H
#define __MCIMX35_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX35 SPECIAL FUNCTION REGISTERS
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
__REG32               : 3;
__REG32 MPE           : 1;
__REG32               : 5;
__REG32 UPE           : 1;
__REG32               : 4;
__REG32 LPM           : 2;
__REG32 RAMW          : 2;
__REG32 ROMW          : 2;
__REG32 VOL_RDY_CNT   : 4;
__REG32               : 3;
__REG32 WBEN          : 1;
__REG32 VSTBY         : 1;
__REG32 STBY_EXIT_SRC : 1;
__REG32 WFI           : 1;
__REG32               : 1;
} __ccm_ccmr_bits;

/* Post Divider Register 0 (PDR0) */
typedef struct {
__REG32               :12;
__REG32 CCM_PER_AHB   : 3;
__REG32 CKIL_SEL      : 1;
__REG32 CON_MUX_DIV   : 4;
__REG32 HSP_PODF      : 2;
__REG32               : 1;
__REG32 IPU_HND_BYP   : 1;
__REG32               : 2;
__REG32 PER_SEL       : 1;
__REG32               : 5;
} __ccm_pdr0_bits;

/* Post Divider Register 1 (PDR1) */
typedef struct {
__REG32               : 7;
__REG32 MSHC_M_U      : 1;
__REG32               :14;
__REG32 MSHC_DIV      : 6;
__REG32 MSHC_DIV_PRE  : 3;
__REG32               : 1;
} __ccm_pdr1_bits;

/* Post Divider Register 2 (PDR2) */
typedef struct {
__REG32 SSI1_DIV        : 6;
__REG32 SSI_M_U         : 1;
__REG32 CSI_M_U         : 1;
__REG32 SSI2_DIV        : 6;
__REG32                 : 2;
__REG32 CSI_DIV         : 6;
__REG32                 : 2;
__REG32 SSI1_DIV_PRE    : 3;
__REG32 SSI2_DIV_PRE    : 3;
__REG32                 : 2;
} __ccm_pdr2_bits;

/* Post Divider Register 3 (PDR3) */
typedef struct {
__REG32 ESDHC1_DIV      : 6;
__REG32 ESDHC_M_U       : 1;
__REG32                 : 1;
__REG32 ESDHC2_DIV      : 6;
__REG32 UART_M_U        : 1;
__REG32                 : 1;
__REG32 ESDHC3_DIV      : 6;
__REG32 SPDIF_M_U       : 1;
__REG32 SPDIF_DIV       : 6;
__REG32 SPDIF_DIV_PRE   : 3;
} __ccm_pdr3_bits;

/* Post Divider Register 4 (PDR4) */
typedef struct {
__REG32                 : 9;
__REG32 USB_M_U         : 1;
__REG32 UART_DIV        : 6;
__REG32 PER0_DIV        : 6;
__REG32 USB_DIV         : 6;
__REG32 NFC_DIV         : 4;
} __ccm_pdr4_bits;

/* Reset Control and Source Register (RCSR) */
typedef struct {
__REG32 REST            : 4;
__REG32 GPF             : 4;
__REG32 NFC_FMS         : 1;
__REG32 NFC_4K          : 1;
__REG32 BOOT_REG        : 2;
__REG32                 : 2;
__REG32 NFC_16bit_SEL0  : 1;
__REG32                 : 7;
__REG32 BT_ECC          : 1;
__REG32 MEM_TYPE        : 2;
__REG32 MEM_CTRL        : 2;
__REG32 PAGE_SIZE       : 2;
__REG32 BUS_WIDTH       : 1;
__REG32 BT_USB_SRC      : 2;
} __ccm_rcsr_bits;

/* Core PLL (MPLL) Control Register (MPCTL) */
/* Peripheral PLL Control Register (PPCTL) */
typedef struct {
__REG32 MFN             :10;
__REG32 MFI             : 4;
__REG32                 : 2;
__REG32 MFD             :10;
__REG32 PD              : 4;
__REG32                 : 1;
__REG32 BRMO            : 1;
} __ccm_mpctl_bits;

/* Audio Clock Mux Register (ACMR) */
typedef struct {
__REG32 SSI2_AUDIO_CLK_SEL  : 4;
__REG32 SSI1_AUDIO_CLK_SEL  : 4;
__REG32 SPDIF_AUDIO_CLK_SEL : 4;
__REG32 ESAI_AUDIO_CLK_SEL  : 4;
__REG32 CKILH_PODF          :12;
__REG32                     : 4;
} __ccm_acmr_bits;

/* Clock Out Source Register (COSR) */
typedef struct {
__REG32 CLKOSEL         : 5;
__REG32 CLKOEN          : 1;
__REG32 CLKO_DIV1       : 1;
__REG32 CKILH_CLKO      : 1;
__REG32                 : 2;
__REG32 CLKO_DIV        : 6;
__REG32                 : 8;
__REG32 ASRC_AUDIO_EN   : 1;
__REG32                 : 1;
__REG32 ASRC_AUDIO_PODF : 6;
} __ccm_cosr_bits;

/* Clock Gating Registers 0 (CGR0) */
typedef struct {
__REG32 ASRC            : 2;
__REG32 ATA             : 2;
__REG32 AUDMUX          : 2;
__REG32 CAN1            : 2;
__REG32 CAN2            : 2;
__REG32 CSPI1           : 2;
__REG32 CSPI2           : 2;
__REG32 ECT             : 2;
__REG32 EDIO            : 2;
__REG32 EMI             : 2;
__REG32 EPIT1           : 2;
__REG32 EPIT2           : 2;
__REG32 ESAI            : 2;
__REG32 ESDHC1          : 2;
__REG32 ESDHC2          : 2;
__REG32 ESDHC3          : 2;
} __ccm_cgr0_bits;

/* Clock Gating Registers 1 (CGR1) */
typedef struct {
__REG32 FEC             : 2;
__REG32 GPIO1           : 2;
__REG32 GPIO2           : 2;
__REG32 GPIO3           : 2;
__REG32 GPT             : 2;
__REG32 I2C1            : 2;
__REG32 I2C2            : 2;
__REG32 I2C3            : 2;
__REG32 IOMUXC          : 2;
__REG32 IPU             : 2;
__REG32 KPP             : 2;
__REG32 MLB             : 2;
__REG32 MSHC            : 2;
__REG32 OWIRE           : 2;
__REG32 PWM             : 2;
__REG32 RNGC            : 2;
} __ccm_cgr1_bits;

/* Clock Gating Registers 2 (CGR2) */
typedef struct {
__REG32 RTC             : 2;
__REG32 RTIC            : 2;
__REG32 SCC             : 2;
__REG32 SDMA            : 2;
__REG32 SPBA            : 2;
__REG32 SPDIF           : 2;
__REG32 SSI1            : 2;
__REG32 SSI2            : 2;
__REG32 UART1           : 2;
__REG32 UART2           : 2;
__REG32 UART3           : 2;
__REG32 USBOTG          : 2;
__REG32 WDOG            : 2;
__REG32 MAX             : 2;
__REG32                 : 2;
__REG32 ADMUX           : 2;
} __ccm_cgr2_bits;

/* Clock Gating Registers 3 (CGR3) */
typedef struct {
__REG32 CSI             : 2;
__REG32 IIM             : 2;
__REG32 GPU2D           : 2;
__REG32                 :26;
} __ccm_cgr3_bits;

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
__REG32 SCR             : 1;
__REG32 DRCE0           : 1;
__REG32 DRCE1           : 1;
__REG32 DRCE2           : 1;
__REG32 DRCE3           : 1;
__REG32 WFIM            : 1;
__REG32 DPVV            : 1;
__REG32 DPVCR           : 1;
__REG32 FSVAI           : 2;
__REG32 FSVAIM          : 1;
__REG32 DVFS_START      : 1;
__REG32 PTVIS           : 1;
__REG32 LBCF            : 2;
__REG32 LBFL            : 1;
__REG32 LBMI            : 1;
__REG32 DVFIS           : 1;
__REG32 DVFEV           : 1;
__REG32 DVFS_UPD_FINISH : 1;
__REG32                 : 3;
__REG32 DVSUP           : 2;
__REG32                 : 2;
} __ccm_pmcr0_bits;

/* Power Management Control Register 1 (PMCR1) */
typedef struct {
__REG32 DVGP            : 4;
__REG32                 : 2;
__REG32 CPFA            : 1;
__REG32                 : 2;
__REG32 CPSPA           : 4;
__REG32                 : 3;
__REG32 WBCN            : 8;
__REG32 CPSPA_EMI       : 4;
__REG32 CPFA_EMI        : 1;
__REG32                 : 3;
} __ccm_pmcr1_bits;

/* Power Management Control Register 2 (PMCR2) */
typedef struct {
__REG32 DVFS_ACK        : 1;
__REG32 DVFS_REQ        : 1;
__REG32                 : 3;
__REG32 REF_COUNTER_OUT :11;
__REG32 OSC24M_DOWN     : 1;
__REG32 OSC_AUDIO_DOWN  : 1;
__REG32 IPU_GAS         : 1;
__REG32 M3_GAS          : 1;
__REG32 M4_GAS          : 1;
__REG32 M1_GAS          : 1;
__REG32                 : 1;
__REG32 OSC_RDY_CNT     : 9;
} __ccm_pmcr2_bits;

/* General Purpose Register (GPR) */
typedef struct {
__REG32 SDCTL_CSD0_SEL_B  : 1;
__REG32 SDCTL_CSD1_SEL_B  : 1;
__REG32 TAMPER_DETECT     : 1;
__REG32                   :29;
} __iomux_gpr_bits;

/* Software MUX Control Register (SW_MUX_CTL) */
typedef struct {
__REG32 MUX_MODE        : 3;
__REG32                 : 1;
__REG32 SION            : 1;
__REG32                 :27;
} __iomux_sw_mux_ctl_bits;

/* Software PAD Control Register (SW_PAD_CTL) */
typedef struct {
__REG32 SRE             : 1;
__REG32 DSE             : 2;
__REG32 ODE             : 1;
__REG32 PULL_KEEP_CTL   : 4;
__REG32 HYS             : 1;
__REG32                 : 4;
__REG32 DRIVE_VOLTAGE   : 1;
__REG32                 :18;
} __iomux_sw_pad_ctl_bits;

/* SW_PAD_CTL Generic Register Format for DDR Pins */
typedef struct {
__REG32                 : 1;
__REG32 DSE             : 2;
__REG32                 : 4;
__REG32 PKE             : 1;
__REG32                 :24;
} __iomux_sw_pad_ctl_ddr_bits;

/* Software Pad Group Control (SW_PAD_CTL_GRP) */
typedef struct {
__REG32                 :11;
__REG32 DDR_TYPE        : 2;
__REG32                 :19;
} __iomux_sw_pad_ctl_grp_bits;

/* SW_SELECT_INPUT Register */
typedef struct {
__REG32 Daisy           : 1;
__REG32                 :31;
} __iomux_sw_select_input0_bits;

/* SW_SELECT_INPUT Register */
typedef struct {
__REG32 Daisy           : 2;
__REG32                 :30;
} __iomux_sw_select_input1_bits;

/* SW_SELECT_INPUT Register */
typedef struct {
__REG32 Daisy           : 3;
__REG32                 :29;
} __iomux_sw_select_input2_bits;

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
__REG32 EMC4RST         : 1;
__REG32 EMC5RST         : 1;
__REG32                 :18;
} __emmc_bits;

/* Counter Status Register (EMCS) */
typedef struct{
__REG32 EMC0            : 1;
__REG32 EMC1            : 1;
__REG32 EMC2            : 1;
__REG32 EMC3            : 1;
__REG32 EMC4            : 1;
__REG32 EMC5            : 1;
__REG32                 :26;
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

/* M3IF WaterMark Configuration Registers (M3IFWCFG0–M3IFWCFG7) */
typedef struct {
__REG32          :10;
__REG32 WBA      :22;
} __m3ifwcfg_bits;

/* M3IF WaterMark Control and Status Register (M3IFWCSR) */
typedef struct {
__REG32 WS0      : 1;
__REG32 WS1      : 1;
__REG32 WS2      : 1;
__REG32 WS3      : 1;
__REG32 WS4      : 1;
__REG32 WS5      : 1;
__REG32 WS6      : 1;
__REG32 WS7      : 1;
__REG32          :23;
__REG32 WIE      : 1;
} __m3ifwcsr_bits;

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
__REG32 CRER            : 1;
__REG32 BCM             : 1;
__REG32                 : 5;
__REG32 AUS0            : 1;
__REG32 AUS1            : 1;
__REG32 AUS2            : 1;
__REG32 AUS3            : 1;
__REG32 AUS4            : 1;
__REG32 AUS5            : 1;
__REG32 ECP0            : 1;
__REG32 ECP1            : 1;
__REG32 ECP2            : 1;
__REG32 ECP3            : 1;
__REG32 ECP4            : 1;
__REG32 ECP5            : 1;
__REG32                 :12;
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
__REG32 MA10_SHARE      : 1;
__REG32 FRC_MSR         : 1;
__REG32 DDR_EN          : 1;
__REG32 DDR2_EN         : 1;
__REG32                 :21;
__REG32 SDRAMRDY        : 1;
} __esdmisc_bits;

/* MDDR Delay Line Configuration Debug Registers */
typedef struct {
__REG32 DLY_REG           : 8;
__REG32 DLY_ABS_OFFSET    : 8;
__REG32 DLY_OFFSET        : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG       : 1;
} __esdcdly_bits;

/* MDDR Delay Line Cycle Length Debug Register */
typedef struct{
__REG32 QTR_CYCLE_LENGTH  : 8;
__REG32                   :24;
} __esdcdlyl_bits;

/* Buffer Number for Page Data Transfer To/From Flash Memory */
typedef struct{
__REG16 RBA         : 3;
__REG16             : 1;
__REG16 ACTIVE_CS   : 2;
__REG16             :10;
} __nfc_rba_bits;

/* NFC Internal Buffer Lock Control */
typedef struct{
__REG16 BLS  : 2;
__REG16      :14;
} __nfc_iblc_bits;

/* NFC Controller Status/Result of Flash Operation */
typedef struct{
__REG16 NOSER1  : 4;
__REG16 NOSER2  : 4;
__REG16 NOSER3  : 4;
__REG16 NOSER4  : 4;
} __ecc_srr_bits;

/* NFC Controller Status/Result of Flash Operation */
typedef struct{
__REG16 NOSER5  : 4;
__REG16 NOSER6  : 4;
__REG16 NOSER7  : 4;
__REG16 NOSER8  : 4;
} __ecc_srr2_bits;

/* NFC SPare Area Size (SPAS) */
typedef struct{
__REG16 SPAS    : 8;
__REG16         : 8;
} __nfc_spas_bits;

/* NFC Nand Flash Write Protection */
typedef struct{
__REG16 WPC  : 3;
__REG16      :13;
} __nf_wr_prot_bits;

/* NFC NAND Flash Write Protection Status */
typedef struct{
__REG16 LTS0 : 1;
__REG16 LS0  : 1;
__REG16 US0  : 1;
__REG16 LTS1 : 1;
__REG16 LS1  : 1;
__REG16 US1  : 1;
__REG16 LTS2 : 1;
__REG16 LS2  : 1;
__REG16 US2  : 1;
__REG16 LTS3 : 1;
__REG16 LS3  : 1;
__REG16 US3  : 1;
__REG16      : 4;
} __nf_wr_prot_sta_bits;

/* NFC NAND Flash Operation Configuration 1 */
typedef struct{
__REG16 ECC_MODE  : 1;
__REG16 DMA_MODE  : 1;
__REG16 SP_EN     : 1;
__REG16 ECC_EN    : 1;
__REG16 INT_MASK  : 1;
__REG16 NF_BIG    : 1;
__REG16 NFC_RST   : 1;
__REG16 NF_CE     : 1;
__REG16 SYM       : 1;
__REG16 PPB       : 2;
__REG16 FP_INT    : 1;
__REG16           : 4;
} __nand_fc1_bits;

/* NFC NAND Flash Operation Configuration 2 */
typedef struct{
__REG16 FCMD  : 1;
__REG16 FADD  : 1;
__REG16 FDI   : 1;
__REG16 FDO   : 3;
__REG16       : 9;
__REG16 INT   : 1;
} __nand_fc2_bits;

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

/* O-Wire Command Register */
typedef struct{
__REG16      : 1;
__REG16 SRA  : 1;
__REG16      :14;
} __ow_command_bits;

/* O-Wire Transmit/Receive Register */
typedef struct{
__REG16 DATA : 8;
__REG16      : 8;
} __ow_rx_tx_bits;

/* O-Wire Interrupt Register */
typedef struct{
__REG16 PD   : 1;
__REG16 PDR  : 1;
__REG16 TBE  : 1;
__REG16 TSRE : 1;
__REG16 RBF  : 1;
__REG16 RSRF : 1;
__REG16      :10;
} __ow_interrupt_bits;

/* O-Wire Interrupt Enable Register */
typedef struct{
__REG16 EPD  : 1;
__REG16 IAS  : 1;
__REG16 ETBE : 1;
__REG16 ETSE : 1;
__REG16 ERBF : 1;
__REG16 ERSF : 1;
__REG16      :10;
} __ow_interrupt_en_bits;

/* ATA_CONTROL Register */
typedef struct{
__REG16 IORDY_EN            : 1;
__REG16 DMA_WRITE           : 1;
__REG16 DMA_ULTRA_SELECTED  : 1;
__REG16 DMA_PENDING         : 1;
__REG16 FIFO_RCV_EN         : 1;
__REG16 FIFO_TX_EN          : 1;
__REG16 ATA_RST_B           : 1;
__REG16 FIFO_RST_B          : 1;
__REG16 DMA_ENABLE          : 1;
__REG16 DMA_START_STOP      : 1;
__REG16 DMA_SELECT          : 2;
__REG16 DMA_SRST            : 1;
__REG16                     : 3;
} __ata_control_bits;

/* INTERRUPT_PENDING Register */
typedef struct{
__REG8                      : 1;
__REG8  DMA_TRANS_OVER      : 1;
__REG8  DMA_ERR             : 1;
__REG8  ATA_IRTRQ2          : 1;
__REG8  CONTROLLER_IDLE     : 1;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8  ATA_INTRQ1          : 1;
} __ata_intr_pend_bits;

/* INTERRUPT_ENABLE Register */
typedef struct{
__REG8                      : 1;
__REG8  DMA_TRANS_OVER      : 1;
__REG8  DMA_ERR             : 1;
__REG8  ATA_IRTRQ2          : 1;
__REG8  CONTROLLER_IDLE     : 1;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8  ATA_INTRQ1          : 1;
} __ata_intr_ena_bits;

/* INTERRUPT_CLEAR Register */
typedef struct{
__REG8                      : 1;
__REG8  DMA_TRANS_OVER      : 1;
__REG8  DMA_ERR             : 1;
__REG8                      : 2;
__REG8  FIFO_OVERFLOW       : 1;
__REG8  FIFO_UNDERFLOW      : 1;
__REG8                      : 1;
} __ata_intr_clr_bits;

/* ATA ADMA_ERR_STATUS Register */
typedef struct{
__REG8  ADMA_ERR_STATE      : 2;
__REG8  ADMA_LEN_MISMATCH   : 1;
__REG8                      : 5;
} __ata_adma_err_status_bits;

/* ATA BURST_LENGTH Register */
typedef struct{
__REG8  BURST_LENGTH        : 6;
__REG8                      : 2;
} __ata_burst_length_bits;

/* CSPI Control Register (CONREG) */
typedef struct{
__REG32 EN            : 1;
__REG32 MODE          : 1;
__REG32 XCH           : 1;
__REG32 SMC           : 1;
__REG32 POL           : 1;
__REG32 PHA           : 1;
__REG32 SSCTL         : 1;
__REG32 SSPOL         : 1;
__REG32 DRCTL         : 2;
__REG32               : 2;
__REG32 CHIP_SELECT   : 2;
__REG32               : 2;
__REG32 DATA_RATE     : 3;
__REG32               : 1;
__REG32 BURST_LENGTH  :12;
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
__REG32 TCEN     : 1;
__REG32          :24;
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
__REG32 TC      : 1;
__REG32         :24;
} __cspi_statreg_bits;

/* CSPI Sample Period Control Register (PERIODREG) */
typedef struct{
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32               :16;
} __cspi_periodreg_bits;

/* CSPI Test Control Register (TESTREG) */
typedef struct{
__REG32 TXCNT     : 4;
__REG32 RXCNT     : 4;
__REG32           : 6;
__REG32 LBC       : 1;
__REG32 SWAP      : 1;
__REG32           :16;
} __cspi_testreg_bits;

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

/* MSHC Command Register (CMDR) */
typedef struct{
__REG32 DSZ_H     : 2;
__REG32 DSL       : 1;
__REG32           : 1;
__REG32 TPC       : 4;
__REG32 DSZ_L     : 8;
__REG32 TOVW      : 8;
__REG32           : 8;
} __mshc_cmdr_bits;

/* MSHC Data Register (DATR) */
typedef struct{
__REG32 DATA3     : 8;
__REG32 DATA2     : 8;
__REG32 DATA1     : 8;
__REG32 DATA0     : 8;
} __mshc_datr_bits;

/* MSHC Status Register (STAR) */
typedef struct{
__REG32 TOE       : 1;
__REG32 CRC       : 1;
__REG32           : 2;
__REG32 RDY       : 1;
__REG32 MSINT     : 1;
__REG32 DRQ       : 1;
__REG32           : 1;
__REG32 CNK       : 1;
__REG32 BRQ       : 1;
__REG32 ERR       : 1;
__REG32 CED       : 1;
__REG32 FUL       : 1;
__REG32 EMP       : 1;
__REG32           : 3;
__REG32 HRESP_ERR : 1;
__REG32 REMP      : 1;
__REG32 WFUL      : 1;
__REG32           : 2;
__REG32 IXFR      : 1;
__REG32 IDA       : 1;
__REG32           : 8;
} __mshc_star_bits;

/* MSHC System Register (SYSR) */
typedef struct{
__REG32 FDIR            : 1;
__REG32 FCLR            : 1;
__REG32 MSIEN           : 1;
__REG32 INTCLR          : 1;
__REG32 NOCRC           : 1;
__REG32 INTEN           : 1;
__REG32 SRAC            : 1;
__REG32 RST             : 1;
__REG32 BSY             : 3;
__REG32 REO             : 1;
__REG32 REI             : 1;
__REG32 DRQSL           : 1;
__REG32 DRM             : 1;
__REG32 DAM             : 1;
__REG32 EDMA_EN         : 1;
__REG32 INTEN_HRESP_ERR : 1;
__REG32 INTEN_REMP      : 1;
__REG32 INTEN_WFUL      : 1;
__REG32                 : 2;
__REG32 INTEN_IXFR      : 1;
__REG32 INTEN_IDA       : 1;
__REG32                 : 8;
} __mshc_sysr_bits;

/* -------------------------------------------------------------------------*/
/* Enhanced Secured Digital Host Controller version 2 (eSDHCv2)             */
/* -------------------------------------------------------------------------*/
/* Block Attributes Register */
typedef struct {
__REG32 BLKSZE          :13;
__REG32                 : 3;
__REG32 BLKCNT          :16;
} __esdhc_blkattr_bits;

/* Transfer Type Register */
typedef struct {
__REG32 DMAEN           : 1;
__REG32 BCEN            : 1;
__REG32 AC12EN          : 1;
__REG32                 : 1;
__REG32 DTDSEL          : 1;
__REG32 MSBSEL          : 1;
__REG32                 :10;
__REG32 RSPTYP          : 2;
__REG32                 : 1;
__REG32 CCCEN           : 1;
__REG32 CICEN           : 1;
__REG32 DPSEL           : 1;
__REG32 CMDTYP          : 2;
__REG32 CMDINX          : 6;
__REG32                 : 2;
} __esdhc_xfertyp_bits;

/* Present State Register */
typedef struct {
__REG32 CIHB            : 1;
__REG32 CDIHB           : 1;
__REG32 DLA             : 1;
__REG32 SDSTB           : 1;
__REG32 IPGOFF          : 1;
__REG32 HCKOFF          : 1;
__REG32 PEROFF          : 1;
__REG32 SDOFF           : 1;
__REG32 WTA             : 1;
__REG32 RTA             : 1;
__REG32 BWEN            : 1;
__REG32 BREN            : 1;
__REG32                 : 4;
__REG32 CINS            : 1;
__REG32                 : 1;
__REG32 CDPL            : 1;
__REG32 WPSPL           : 1;
__REG32                 : 3;
__REG32 CLSL            : 1;
__REG32 DLSL            : 8;
} __esdhc_prsstat_bits;

/* Protocol Control Register */
typedef struct {
__REG32 LCTL            : 1;
__REG32 DTW             : 2;
__REG32 D3CD            : 1;
__REG32 EMODE           : 2;
__REG32 CDTL            : 1;
__REG32 CDSS            : 1;
__REG32 DMAS            : 2;
__REG32                 : 6;
__REG32 SABGREQ         : 1;
__REG32 CREQ            : 1;
__REG32 RWCTL           : 1;
__REG32 IABG            : 1;
__REG32                 : 4;
__REG32 WECINT          : 1;
__REG32 WECINS          : 1;
__REG32 WECRM           : 1;
__REG32                 : 5;
} __esdhc_proctl_bits;

/* System Control Register */
typedef struct {
__REG32 IPGEN           : 1;
__REG32 HCKEN           : 1;
__REG32 PEREN           : 1;
__REG32 SDCLKEN         : 1;
__REG32 DVS             : 4;
__REG32 SDCLKFS         : 8;
__REG32 DTOCV           : 4;
__REG32                 : 4;
__REG32 RSTA            : 1;
__REG32 RSTC            : 1;
__REG32 RSTD            : 1;
__REG32 INITA           : 1;
__REG32                 : 4;
} __esdhc_sysctl_bits;

/* Interrupt Status Register */
typedef struct {
__REG32 CC              : 1;
__REG32 TC              : 1;
__REG32 BGE             : 1;
__REG32 DINT            : 1;
__REG32 BWR             : 1;
__REG32 BRR             : 1;
__REG32 CINS            : 1;
__REG32 CRM             : 1;
__REG32 CINT            : 1;
__REG32                 : 7;
__REG32 CTOE            : 1;
__REG32 CCE             : 1;
__REG32 CEBE            : 1;
__REG32 CIE             : 1;
__REG32 DTOE            : 1;
__REG32 DCE             : 1;
__REG32 DEBE            : 1;
__REG32                 : 1;
__REG32 AC12E           : 1;
__REG32                 : 3;
__REG32 DMAE            : 1;
__REG32                 : 3;
} __esdhc_irqstat_bits;

/* Interrupt Status Enable Register */
typedef struct {
__REG32 CCSEN           : 1;
__REG32 TCSEN           : 1;
__REG32 BGESEN          : 1;
__REG32 DINTSEN         : 1;
__REG32 BWRSEN          : 1;
__REG32 BRRSEN          : 1;
__REG32 CINSSEN         : 1;
__REG32 CRMSEN          : 1;
__REG32 CINTSEN         : 1;
__REG32                 : 7;
__REG32 CTOESEN         : 1;
__REG32 CCESEN          : 1;
__REG32 CEBESEN         : 1;
__REG32 CIESEN          : 1;
__REG32 DTOESEN         : 1;
__REG32 DCESEN          : 1;
__REG32 DEBESEN         : 1;
__REG32                 : 1;
__REG32 AC12ESEN        : 1;
__REG32                 : 3;
__REG32 DMAESEN         : 1;
__REG32                 : 3;
} __esdhc_irqstaten_bits;

/* Interrupt Signal Enable Register */
typedef struct {
__REG32 CCIEN           : 1;
__REG32 TCIEN           : 1;
__REG32 BGEIEN          : 1;
__REG32 DINTIEN         : 1;
__REG32 BWRIEN          : 1;
__REG32 BRRIEN          : 1;
__REG32 CINSIEN         : 1;
__REG32 CRMIEN          : 1;
__REG32 CINTIEN         : 1;
__REG32                 : 7;
__REG32 CTOEIEN         : 1;
__REG32 CCEIEN          : 1;
__REG32 CEBEIEN         : 1;
__REG32 CIEIEN          : 1;
__REG32 DTOEIEN         : 1;
__REG32 DCEIEN          : 1;
__REG32 DEBEIEN         : 1;
__REG32                 : 1;
__REG32 AC12EIEN        : 1;
__REG32                 : 3;
__REG32 DMAEIEN         : 1;
__REG32                 : 3;
} __esdhc_irqsigen_bits;

/* Auto CMD12 Error Status Register */
typedef struct {
__REG32 AC12NE          : 1;
__REG32 AC12TOE         : 1;
__REG32 AC12EBE         : 1;
__REG32 AC12CE          : 1;
__REG32 AC12IE          : 1;
__REG32                 : 2;
__REG32 CNIBAC12E       : 1;
__REG32                 :24;
} __esdhc_autoc12err_bits;

/* Host Controller Capabilities */
typedef struct {
__REG32                 :16;
__REG32 MBL             : 3;
__REG32                 : 1;
__REG32 ADMAS           : 1;
__REG32 HSS             : 1;
__REG32 DMAS            : 1;
__REG32 SRS             : 1;
__REG32 VS33            : 1;
__REG32 VS30            : 1;
__REG32 VS18            : 1;
__REG32                 : 5;
} __esdhc_hostcapblt_bits;

/* Watermark Level Register (WML) */
typedef struct {
__REG32 RD_WML          : 8;
__REG32 RD_BRST_LEN     : 5;
__REG32                 : 3;
__REG32 WR_WML          : 8;
__REG32 WR_BRST_LEN     : 5;
__REG32                 : 3;
} __esdhc_wml_bits;

/* Force Event Register */
typedef struct {
__REG32 FEVTAC12NE      : 1;
__REG32 FEVTAC12TOE     : 1;
__REG32 FEVTAC12CE      : 1;
__REG32 FEVTAC12EBE     : 1;
__REG32 FEVTAC12IE      : 1;
__REG32                 : 2;
__REG32 FEVTCNIBAC12E   : 1;
__REG32                 : 8;
__REG32 FEVTCTOE        : 1;
__REG32 FEVTCCE         : 1;
__REG32 FEVTCEBE        : 1;
__REG32 FEVTCIE         : 1;
__REG32 FEVTDTOE        : 1;
__REG32 FEVTDCE         : 1;
__REG32 FEVTDEBE        : 1;
__REG32                 : 1;
__REG32 FEVTAC12E       : 1;
__REG32                 : 3;
__REG32 FEVTDMAE        : 1;
__REG32                 : 2;
__REG32 FEVTCINT        : 1;
} __esdhc_fevt_bits;

/* ADMA Error Status Register */
typedef struct {
__REG32 ADMAES          : 2;
__REG32 ADMALME         : 1;
__REG32 ADMADCE         : 1;
__REG32                 :28;
} __esdhc_admaes_bits;

/* Vendor Specific Register (VENDOR) */
typedef struct {
__REG32  EXT_DMA_EN     : 1;
__REG32  VOLT_SEL       : 1;
__REG32                 :14;
__REG32  INT_ST_VAL     : 8;
__REG32  DBG_SEL        : 4;
__REG32                 : 4;
} __esdhc_vendor_bits;

/* Host Controller Version */
typedef struct {
__REG32 SVN             : 8;
__REG32 VVN             : 8;
__REG32                 :16;
} __esdhc_hostver_bits;

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
__REG32 ONEMS :24;     
__REG32       : 8;
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
__REG32 OOCS        : 1;
__REG32 HOCS        : 1;
__REG32 OCPOL_HST   : 1;
__REG32 OCPOL_OTG   : 1;
__REG32 USBTE       : 1;
__REG32 HSTD        : 1;
__REG32 IP_PUE_DWN  : 1;
__REG32 P_PUE_UP    : 1;
__REG32 IP_PUIDP    : 1;
__REG32 XCSH        : 1;
__REG32 XCSO        : 1;
__REG32 PP_OTG      : 1;
__REG32 HLKEN       : 1;
__REG32 OLKEN       : 1;
__REG32 ID_WKEN     : 1;
__REG32 VBUS_WKEN   : 1;
__REG32 HPM         : 1;
__REG32 OV_EN       : 1;
__REG32 PP_HST      : 1;
__REG32 HWIE        : 1;
__REG32 HUIE        : 1;
__REG32 HSIC        : 2;
__REG32 HWIR        : 1;
__REG32 OPM         : 1;
__REG32 OEX_TEN     : 1;
__REG32 HEX_TEN     : 1;
__REG32 OWIE        : 1;
__REG32 OUIE        : 1;
__REG32 OSIC        : 2;
__REG32 OWIR        : 1;
} __usb_ctrl_bits;

/* OTGMIRROR—OTG Port Mirror Register */
typedef struct{
__REG32 IDDIG       : 1;
__REG32 ASESVLD     : 1;
__REG32 BSESVLD     : 1;
__REG32 VBUSVAL     : 1;
__REG32 SESEND      : 1;
__REG32             : 1;
__REG32 HULPICLK    : 1;
__REG32 OULPICLK    : 1;
__REG32 OUTMICLK    : 1;
__REG32             :23;
} __usb_otg_mirror_bits;

/* USB_PHY_CTRL_FUNC — OTG UTMI PHY Function Control reg */
typedef struct{
__REG32 VSTS        : 8;
__REG32 VCTL_DATA   : 8;
__REG32 VCTL_ADDR   : 4;
__REG32 VCTL_LD     : 1;
__REG32             : 1;
__REG32 LSFE        : 1;
__REG32 EVDO        : 1;
__REG32 USBEN       : 1;
__REG32 RESET       : 1;
__REG32 SUSP        : 1;
__REG32 CLKVLD      : 1;
__REG32             : 4;
} __usb_phy_ctrl_func_bits;

/* USB_PHY_CTRL_TEST — OTG UTMI PHY Test Control register */
typedef struct{
__REG32 FT          : 1;
__REG32 NFC         : 1;
__REG32 LB          : 1;
__REG32 TM          : 2;
__REG32             :27;
} __usb_phy_ctrl_test_bits;

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

/* USB Device Interface Version Number */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __uog_dciversion_bits;

/* HCCPARAMS—Host Control Capability Parameters */
typedef struct{
__REG32 DEN       : 5;
__REG32           : 2;
__REG32 DC        : 1;
__REG32 HC        : 1;
__REG32           :23;
} __uog_dccparams_bits;

/* SBUSFG - control for the system bus interface */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __uog_sbuscfg_bits;

/* USB General Purpose Timer #0 Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __uog_gptimerxld_bits;

/* USB General Purpose Timer #0 Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __uog_gptimerxctrl_bits;

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
__REG32           : 1;
__REG32 SUTW      : 1;
__REG32 ADTW      : 1;
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
__REG32 NAKI      : 1;
__REG32           : 7;
__REG32 TIE0      : 1;
__REG32 TIE1      : 1;
__REG32           : 6;
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
__REG32           :13;
__REG32 TIE0      : 1;
__REG32 TIE1      : 1;
__REG32           : 6;
} __uog_usbintr_bits;

/* FRINDEX—USB Frame Index */
typedef struct{
__REG32 FRINDEX   :14;
__REG32           :18;
} __uog_frindex_bits;

/* PERIODICLISTBASE—Host Controller Frame List Base Address */
/* DEVICEADDR—Device Controller USB Device Address */
typedef union {
  /* UOG_PERIODICLISTBASE */
  /* UH1_PERIODICLISTBASE */
  struct{
    __REG32           :12;
    __REG32 BASEADR   :20;
  };
  /* UOG_DEVICEADDR */
  /* UH1_DEVICEADDR */
  struct{
    __REG32           :25;
    __REG32 USBADR    : 7;
  };
} __uog_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
  /* UOG_ASYNCLISTADDR */
  /* UH1_ASYNCLISTADDR */
  struct{
    __REG32           : 5;
    __REG32 ASYBASE   :27;
  };
  /* UOG_ENDPOINTLISTADDR */
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
__REG32 TXSCHOH     : 8;
__REG32 TXSCHHEALTH : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __uog_txfilltuning_bits;

/* ULPI VIEWPORT */
typedef struct{
__REG32 ULPIDATWR : 8;
__REG32 ULPIDATRD : 8;
__REG32 ULPIADDR  : 8;
__REG32 ULPIPORT  : 3;
__REG32 ULPISS    : 1;
__REG32           : 1;
__REG32 ULPIRW    : 1;
__REG32 ULPIRUN   : 1;
__REG32 ULPIWU    : 1;
} __uog_ulpiview_bits;

/* Endpoint NAK register */
typedef struct{
__REG32 EPRN0     : 1;
__REG32 EPRN1     : 1;
__REG32 EPRN2     : 1;
__REG32 EPRN3     : 1;
__REG32 EPRN4     : 1;
__REG32 EPRN5     : 1;
__REG32 EPRN6     : 1;
__REG32 EPRN7     : 1;
__REG32           : 8;
__REG32 EPTN0     : 1;
__REG32 EPTN1     : 1;
__REG32 EPTN2     : 1;
__REG32 EPTN3     : 1;
__REG32 EPTN4     : 1;
__REG32 EPTN5     : 1;
__REG32 EPTN6     : 1;
__REG32 EPTN7     : 1;
__REG32           : 8;
} __uog_endptnak_bits;

/* Endpoint NAK register */
typedef struct{
__REG32 EPRNE0      : 1;
__REG32 EPRNE1      : 1;
__REG32 EPRNE2      : 1;
__REG32 EPRNE3      : 1;
__REG32 EPRNE4      : 1;
__REG32 EPRNE5      : 1;
__REG32 EPRNE6      : 1;
__REG32 EPRNE7      : 1;
__REG32             : 8;
__REG32 EPTNE0      : 1;
__REG32 EPTNE1      : 1;
__REG32 EPTNE2      : 1;
__REG32 EPTNE3      : 1;
__REG32 EPTNE4      : 1;
__REG32 EPTNE5      : 1;
__REG32 EPTNE6      : 1;
__REG32 EPTNE7      : 1;
__REG32             : 8;
} __uog_endptnaken_bits;

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
__REG32 PTC       : 4;
__REG32 WKCN      : 1;
__REG32 WKDS      : 1;
__REG32 WKOC      : 1;
__REG32 PHCD      : 1;
__REG32 PFSC      : 1;
__REG32           : 1;
__REG32 PSPD      : 2;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __uog_portsc_bits;

/* OTGSC—OTG Status Control */
typedef struct{
__REG32 VD        : 1;
__REG32 VC        : 1;
__REG32 HAAR      : 1;
__REG32 OT        : 1;
__REG32 DP        : 1;
__REG32 IDPU      : 1;
__REG32 HADP      : 1;
__REG32 HABA      : 1;
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
__REG32 ES        : 1;
__REG32 SLOM      : 1;
__REG32 SDIS      : 1;
__REG32           :27;
} __uog_usbmode_bits;

/* ENDPTSETUPSTAT—Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT0   : 1;
__REG32 ENDPTSETUPSTAT1   : 1;
__REG32 ENDPTSETUPSTAT2   : 1;
__REG32 ENDPTSETUPSTAT3   : 1;
__REG32 ENDPTSETUPSTAT4   : 1;
__REG32 ENDPTSETUPSTAT5   : 1;
__REG32 ENDPTSETUPSTAT6   : 1;
__REG32 ENDPTSETUPSTAT7   : 1;
__REG32 ENDPTSETUPSTAT8   : 1;
__REG32 ENDPTSETUPSTAT9   : 1;
__REG32 ENDPTSETUPSTAT10  : 1;
__REG32 ENDPTSETUPSTAT11  : 1;
__REG32 ENDPTSETUPSTAT12  : 1;
__REG32 ENDPTSETUPSTAT13  : 1;
__REG32 ENDPTSETUPSTAT14  : 1;
__REG32 ENDPTSETUPSTAT15  : 1;
__REG32                   :16;
} __uog_endptsetupstat_bits;

/* ENDPTPRIME—Endpoint Initialization */
typedef struct{
__REG32 PERB0       : 1;
__REG32 PERB1       : 1;
__REG32 PERB2       : 1;
__REG32 PERB3       : 1;
__REG32 PERB4       : 1;
__REG32 PERB5       : 1;
__REG32 PERB6       : 1;
__REG32 PERB7       : 1;
__REG32 PERB8       : 1;
__REG32 PERB9       : 1;
__REG32 PERB10      : 1;
__REG32 PERB11      : 1;
__REG32 PERB12      : 1;
__REG32 PERB13      : 1;
__REG32 PERB14      : 1;
__REG32 PERB15      : 1;
__REG32 PETB0       : 1;
__REG32 PETB1       : 1;
__REG32 PETB2       : 1;
__REG32 PETB3       : 1;
__REG32 PETB4       : 1;
__REG32 PETB5       : 1;
__REG32 PETB6       : 1;
__REG32 PETB7       : 1;
__REG32 PETB8       : 1;
__REG32 PETB9       : 1;
__REG32 PETB10      : 1;
__REG32 PETB11      : 1;
__REG32 PETB12      : 1;
__REG32 PETB13      : 1;
__REG32 PETB14      : 1;
__REG32 PETB15      : 1;
} __uog_endptprime_bits;

/* ENDPTFLUSH—Endpoint De-Initialize */
typedef struct{
__REG32 FERB0       : 1;
__REG32 FERB1       : 1;
__REG32 FERB2       : 1;
__REG32 FERB3       : 1;
__REG32 FERB4       : 1;
__REG32 FERB5       : 1;
__REG32 FERB6       : 1;
__REG32 FERB7       : 1;
__REG32 FERB8       : 1;
__REG32 FERB9       : 1;
__REG32 FERB10      : 1;
__REG32 FERB11      : 1;
__REG32 FERB12      : 1;
__REG32 FERB13      : 1;
__REG32 FERB14      : 1;
__REG32 FERB15      : 1;
__REG32 FETB0       : 1;
__REG32 FETB1       : 1;
__REG32 FETB2       : 1;
__REG32 FETB3       : 1;
__REG32 FETB4       : 1;
__REG32 FETB5       : 1;
__REG32 FETB6       : 1;
__REG32 FETB7       : 1;
__REG32 FETB8       : 1;
__REG32 FETB9       : 1;
__REG32 FETB10      : 1;
__REG32 FETB11      : 1;
__REG32 FETB12      : 1;
__REG32 FETB13      : 1;
__REG32 FETB14      : 1;
__REG32 FETB15      : 1;
} __uog_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR0       : 1;
__REG32 ERBR1       : 1;
__REG32 ERBR2       : 1;
__REG32 ERBR3       : 1;
__REG32 ERBR4       : 1;
__REG32 ERBR5       : 1;
__REG32 ERBR6       : 1;
__REG32 ERBR7       : 1;
__REG32 ERBR8       : 1;
__REG32 ERBR9       : 1;
__REG32 ERBR10      : 1;
__REG32 ERBR11      : 1;
__REG32 ERBR12      : 1;
__REG32 ERBR13      : 1;
__REG32 ERBR14      : 1;
__REG32 ERBR15      : 1;
__REG32 ETBR0       : 1;
__REG32 ETBR1       : 1;
__REG32 ETBR2       : 1;
__REG32 ETBR3       : 1;
__REG32 ETBR4       : 1;
__REG32 ETBR5       : 1;
__REG32 ETBR6       : 1;
__REG32 ETBR7       : 1;
__REG32 ETBR8       : 1;
__REG32 ETBR9       : 1;
__REG32 ETBR10      : 1;
__REG32 ETBR11      : 1;
__REG32 ETBR12      : 1;
__REG32 ETBR13      : 1;
__REG32 ETBR14      : 1;
__REG32 ETBR15      : 1;
} __uog_endptstat_bits;

/* ENDPTCOMPLETE—Endpoint Compete */
typedef struct{
__REG32 ERCE0       : 1;
__REG32 ERCE1       : 1;
__REG32 ERCE2       : 1;
__REG32 ERCE3       : 1;
__REG32 ERCE4       : 1;
__REG32 ERCE5       : 1;
__REG32 ERCE6       : 1;
__REG32 ERCE7       : 1;
__REG32 ERCE8       : 1;
__REG32 ERCE9       : 1;
__REG32 ERCE10      : 1;
__REG32 ERCE11      : 1;
__REG32 ERCE12      : 1;
__REG32 ERCE13      : 1;
__REG32 ERCE14      : 1;
__REG32 ERCE15      : 1;
__REG32 ETCE0       : 1;
__REG32 ETCE1       : 1;
__REG32 ETCE2       : 1;
__REG32 ETCE3       : 1;
__REG32 ETCE4       : 1;
__REG32 ETCE5       : 1;
__REG32 ETCE6       : 1;
__REG32 ETCE7       : 1;
__REG32 ETCE8       : 1;
__REG32 ETCE9       : 1;
__REG32 ETCE10      : 1;
__REG32 ETCE11      : 1;
__REG32 ETCE12      : 1;
__REG32 ETCE13      : 1;
__REG32 ETCE14      : 1;
__REG32 ETCE15      : 1;
} __uog_endptcomplete_bits;

/* Endpoint Control 0 Register (ENDPTCTRL0) */
typedef struct{
__REG32 RXS             : 1;
__REG32                 : 1;
__REG32 RXT             : 2;
__REG32                 : 3;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32                 : 1;
__REG32 TXT             : 2;
__REG32                 : 3;
__REG32 TXE             : 1;
__REG32                 : 8;
} __uog_endptctrl0_bits;

/* Endpoint Control x Registers (ENDPTCTRLx, x = 1…15) */
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
__REG32 WAITEN          : 1;
__REG32                 : 1;
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
__REG32                 : 1;
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
} __rtcctl_bits;

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
__REG16 WDT    : 1;     /* Bit  3       - WDOG time-out assertion*/
__REG16 SRS    : 1;     /* Bit  4       - Software Reset Signal*/
__REG16 WDA    : 1;     /* Bit  5       - Watchdog Assertion*/
__REG16        : 1;     /* Bit  6       - Reserved*/
__REG16 WDW    : 1;     /* Bit  7       - Watchdog Disable for Wait*/
__REG16 WT     : 8;     /* Bits 8 - 15  - Watchdog Time-Out Field*/
} __wcr_bits;

/* Watchdog Reset Status Register (WRSR) */
typedef struct {
__REG16 SFTW  : 1;     /* Bit  0       - Software Reset*/
__REG16 TOUT  : 1;     /* Bit  1       - Time-out*/
__REG16       :14;     /* Bits 2  - 15 - Reserved*/
} __wrsr_bits;

/* Watchdog Interrupt Control Register (WDOG-1.WICR) */
typedef struct {
__REG16 WICT  : 8;
__REG16       : 6;
__REG16 WTIS  : 1;
__REG16 WIE   : 1;
} __wicr_bits;

/* Watchdog Miscellaneous Control Register (WDOG-1.WMCR)*/
typedef struct {
__REG16 PDE   : 1;
__REG16       :15;
} __wmcr_bits;

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

/* SDMA Lock Register (SDMA_LOCK) */
typedef struct {
__REG32 LOCK              : 1;
__REG32 SRESET_LOCK_CLR   : 1;
__REG32                   :30;
} __sdma_lock_bits;

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

/* DMA Requests 2 (EVT_MIRROR2) */
typedef struct {
__REG32 EVENTS32      : 1;
__REG32 EVENTS33      : 1;
__REG32 EVENTS34      : 1;
__REG32 EVENTS35      : 1;
__REG32 EVENTS36      : 1;
__REG32 EVENTS37      : 1;
__REG32 EVENTS38      : 1;
__REG32 EVENTS39      : 1;
__REG32 EVENTS40      : 1;
__REG32 EVENTS41      : 1;
__REG32 EVENTS42      : 1;
__REG32 EVENTS43      : 1;
__REG32 EVENTS44      : 1;
__REG32 EVENTS45      : 1;
__REG32 EVENTS46      : 1;
__REG32 EVENTS47      : 1;
__REG32               :16;
} __sdma_evt_mirror2_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF1)*/
typedef struct{
__REG32 NUM0        : 6;
__REG32 CNF0        : 1;
__REG32             : 1;
__REG32 NUM1        : 6;
__REG32 CNF1        : 1;
__REG32             : 1;
__REG32 NUM2        : 6;
__REG32 CNF2        : 1;
__REG32             : 1;
__REG32 NUM3        : 6;
__REG32 CNF3        : 1;
__REG32             : 1;
} __sdma_xtrig_conf1_bits;

/* Cross-Trigger Events Configuration Register (XTRIG_CONF2)*/
typedef struct{
__REG32 NUM4        : 6;
__REG32 CNF4        : 1;
__REG32             : 1;
__REG32 NUM5        : 6;
__REG32 CNF5        : 1;
__REG32             : 1;
__REG32 NUM6        : 6;
__REG32 CNF6        : 1;
__REG32             : 1;
__REG32 NUM7        : 6;
__REG32 CNF7        : 1;
__REG32             : 1;
} __sdma_xtrig_conf2_bits;

/* Channel Priority Registers (CHNPRIn) */
typedef struct{
__REG32 CHNPRI      : 3;
__REG32             :29;
} __sdma_chnpri_bits;

/* Channel Enable RAM (CHNENBLn) */
typedef struct {
__REG32 ENBL0       : 1;
__REG32 ENBL1       : 1;
__REG32 ENBL2       : 1;
__REG32 ENBL3       : 1;
__REG32 ENBL4       : 1;
__REG32 ENBL5       : 1;
__REG32 ENBL6       : 1;
__REG32 ENBL7       : 1;
__REG32 ENBL8       : 1;
__REG32 ENBL9       : 1;
__REG32 ENBL10      : 1;
__REG32 ENBL11      : 1;
__REG32 ENBL12      : 1;
__REG32 ENBL13      : 1;
__REG32 ENBL14      : 1;
__REG32 ENBL15      : 1;
__REG32 ENBL16      : 1;
__REG32 ENBL17      : 1;
__REG32 ENBL18      : 1;
__REG32 ENBL19      : 1;
__REG32 ENBL20      : 1;
__REG32 ENBL21      : 1;
__REG32 ENBL22      : 1;
__REG32 ENBL23      : 1;
__REG32 ENBL24      : 1;
__REG32 ENBL25      : 1;
__REG32 ENBL26      : 1;
__REG32 ENBL27      : 1;
__REG32 ENBL28      : 1;
__REG32 ENBL29      : 1;
__REG32 ENBL30      : 1;
__REG32 ENBL31      : 1;
} __sdma_chnenbl_bits;

/* SPBA Peripheral Right Register */
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

/* IPU General Purpose Register (IPU_GP_REG) */
typedef struct{
__REG32 D0_BE_POL           : 1;
__REG32 D1_BE_POL           : 1;
__REG32 D2_BE_POL           : 1;
__REG32 BE_PIN_SEL          : 1;
__REG32                     :28;
} __ipu_gp_reg_bits;

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
__REG32 H_SYNC_DELAY      : 4;
__REG32                   :12;
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
__REG32 DISP_ACK_PTRN         :24;
__REG32 DISP_ACK_MAP          : 1;
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

/* -------------------------------------------------------------------------*/
/*               SSI registers                                              */
/* -------------------------------------------------------------------------*/
/* SSI Control/Status Register */
typedef struct{
__REG32 SSIEN       : 1;
__REG32 TE          : 1;
__REG32 RE          : 1;
__REG32 NET         : 1;
__REG32 SYN         : 1;
__REG32 I2S_MODE    : 2;
__REG32 SYS_CLK_EN  : 1;
__REG32 TCH_EN      : 1;
__REG32 CLK_IST     : 1;
__REG32 TFR_CLK_DIS : 1;
__REG32 RFR_CLK_DIS : 1;
__REG32 SYNC_TX_FS  : 1;
__REG32             :19;
} __scsr_bits;

/* SSI Interrupt Status Register */
typedef struct{
__REG32 TFE0   : 1;
__REG32 TFE1   : 1;
__REG32 RFF0   : 1;
__REG32 RFF1   : 1;
__REG32 RLS    : 1;
__REG32 TLS    : 1;
__REG32 RFS    : 1;
__REG32 TFS    : 1;
__REG32 TUE0   : 1;
__REG32 TUE1   : 1;
__REG32 ROE0   : 1;
__REG32 ROE1   : 1;
__REG32 TDE0   : 1;
__REG32 TDE1   : 1;
__REG32 RDR0   : 1;
__REG32 RDR1   : 1;
__REG32 RXT    : 1;
__REG32 CMDDU  : 1;
__REG32 CMDAU  : 1;
__REG32        : 4;
__REG32 TFRC   : 1;
__REG32 RFRC   : 1;
__REG32        : 7;
} __sisr_bits;

/* SSI Interrupt Enable Register */
typedef struct{
__REG32 TFE0_EN   : 1;
__REG32 TFE1_EN   : 1;
__REG32 RFF0_EN   : 1;
__REG32 RFF1_EN   : 1;
__REG32 RLS_EN    : 1;
__REG32 TLS_EN    : 1;
__REG32 RFS_EN    : 1;
__REG32 TFS_EN    : 1;
__REG32 TUE0_EN   : 1;
__REG32 TUE1_EN   : 1;
__REG32 ROE0_EN   : 1;
__REG32 ROE1_EN   : 1;
__REG32 TDE0_EN   : 1;
__REG32 TDE1_EN   : 1;
__REG32 RDR0_EN   : 1;
__REG32 RDR1_EN   : 1;
__REG32 RXT_EN    : 1;
__REG32 CMDDU_EN  : 1;
__REG32 CMDAU_EN  : 1;
__REG32 TIE       : 1;
__REG32 TDMAE     : 1;
__REG32 RIE       : 1;
__REG32 RDMAE     : 1;
__REG32 TFRC_EN   : 1;
__REG32 RFRC_EN   : 1;
__REG32           : 7;
} __sier_bits;

/* SSI Transmit Configuration Register */
typedef struct{
__REG32 TEFS    : 1;
__REG32 TFSL    : 1;
__REG32 TFSI    : 1;
__REG32 TSCKP   : 1;
__REG32 TSHFD   : 1;
__REG32 TXDIR   : 1;
__REG32 TFDIR   : 1;
__REG32 TFEN0   : 1;
__REG32 TFEN1   : 1;
__REG32 TXBIT0  : 1;
__REG32         :22;
} __stcr_bits;

/* SSI Receive Configuration Register */
typedef struct{
__REG32 REFS    : 1;
__REG32 RFSL    : 1;
__REG32 RFSI    : 1;
__REG32 RSCKP   : 1;
__REG32 RSHFD   : 1;
__REG32 RXDIR   : 1;
__REG32 RFDIR   : 1;
__REG32 RFEN0   : 1;
__REG32 RFEN1   : 1;
__REG32 RXBIT0  : 1;
__REG32 RXEXT   : 1;
__REG32         :21;
} __srcr_bits;

/* SSI Clock Control Register */
typedef struct{
__REG32 PM    : 8;
__REG32 DC    : 5;
__REG32 WL    : 4;
__REG32 PSR   : 1;
__REG32 DIV2  : 1;
__REG32       :13;
} __ssi_ccr_bits;

/* SSI FIFO Control/Status Register */
typedef struct{
__REG32 TFWM0   : 4;
__REG32 RFWM0   : 4;
__REG32 TFCNT0  : 4;
__REG32 RFCNT0  : 4;
__REG32 TFWM1   : 4;
__REG32 RFWM1   : 4;
__REG32 TFCNT1  : 4;
__REG32 RFCNT1  : 4;
} __ssi_sfcsr_bits;

/* SSI Test Register */
typedef struct{
__REG32 TXSTATE  : 5;
__REG32 TFS2RFS  : 1;
__REG32 TCK2RCK  : 1;
__REG32 TXD2RXD  : 1;
__REG32 RXSTATE  : 5;
__REG32 RFS2TFS  : 1;
__REG32 RCK2TCK  : 1;
__REG32 TEST     : 1;
__REG32          :16;
} __ssi_str_bits;

/* SSI Option Register */
typedef struct{
__REG32 SYNRST  : 1;
__REG32 WAIT    : 2;
__REG32 INIT    : 1;
__REG32 TX_CLR  : 1;
__REG32 RX_CLR  : 1;
__REG32 CLKOFF  : 1;
__REG32         :25;
} __ssi_sor_bits;

/* SSI AC97 Control Register */
typedef struct{
__REG32 A97EN  : 1;
__REG32 FV     : 1;
__REG32 TIF    : 1;
__REG32 RD     : 1;
__REG32 WR     : 1;
__REG32 FRDIV  : 6;
__REG32        :21;
} __ssi_sacnt_bits;

/* SSI AC97 Command Address Register */
typedef struct{
__REG32 SACADD  :19;
__REG32         :13;
} __ssi_sacadd_bits;

/* SSI AC97 Command Data Register */
typedef struct{
__REG32 SACDAT  :20;
__REG32         :12;
} __ssi_sacdat_bits;

/* SSI AC97 Tag Register */
typedef struct{
__REG32 SATAG  :16;
__REG32        :16;
} __ssi_satag_bits;

/* SSI AC97 Channel Status Register (SACCST) */
typedef struct{
__REG32 SACCST :10;
__REG32        :22;
} __ssi_saccst_bits;

/* SSI AC97 Channel Enable Register (SACCEN)*/
typedef struct{
__REG32 SACCEN :10;
__REG32        :22;
} __ssi_saccen_bits;

/* SSI AC97 Channel Disable Register (SACCDIS) */
typedef struct{
__REG32 SACCDIS :10;
__REG32         :22;
} __ssi_saccdis_bits;

/* General Purpose Control Register (GP_CTRL) */
typedef struct{
__REG32 GP_CTRL :11;
__REG32         :21;
} __gp_ctrl_bits;

/* Set Control Register (GP_SER) */
typedef struct{
__REG32 GP_SER  :11;
__REG32         :21;
} __gp_ser_bits;

/* Clear Control Register (GP_CER) */
typedef struct{
__REG32 GP_CER  :11;
__REG32         :21;
} __gp_cer_bits;

/* General Purpose Status Register (GP_STAT) */
typedef struct{
__REG32 GP_STAT :16;
__REG32         :16;
} __gp_stat_bits;

/* L2_MEM_VAL Register (L2_MEM_VAL) */
typedef struct{
__REG32 L2_RVAL_RST   : 4;
__REG32 L2_RVAL2_RST  : 4;
__REG32 L2_WVAL_RST   : 4;
__REG32 OFFSET_RST    : 1;
__REG32               :19;
} __l2_mem_val_bits;

/* Debug Control Register 1 (DBG_CTRL1) */
typedef struct{
__REG32               : 8;
__REG32 TOMS          : 4;
__REG32 TIMS0         : 2;
__REG32               : 2;
__REG32 TIMS1         : 4;
__REG32               : 4;
__REG32 EPS           : 1;
__REG32               : 7;
} __dbg_ctrl1_bits;

/* Platform ID Register (PLAT_ID) */
typedef struct{
__REG32               : 8;
__REG32 ECO           : 8;
__REG32 MINOR         : 8;
__REG32 IMPL          : 4;
__REG32 SPEC          : 4;
} __plat_id_bits;

/* SPDIF Configuration Register (SCR) */
typedef struct{
__REG32 USrc_Sel        : 2;
__REG32 TxSel           : 3;
__REG32 ValCtrl         : 1;
__REG32                 : 2;
__REG32 PDIR_Tx         : 1;
__REG32 PDIR_Rcv        : 1;
__REG32 TxFIFO_Ctrl     : 2;
__REG32 soft_reset      : 1;
__REG32 Low_power       : 1;
__REG32                 : 1;
__REG32 TxFIFOEmpty_Sel : 2;
__REG32 TxAutoSync      : 1;
__REG32 RcvAutoSync     : 1;
__REG32 RcvFifoFull_Sel : 2;
__REG32 RcvFifo_Rst     : 1;
__REG32 RcvFIFO_Off_On  : 1;
__REG32 RcvFIFO_Ctrl    : 1;
__REG32                 : 8;
} __spdif_scr_bits;

/* CDText Control Register (SRCD) */
typedef struct{
__REG32                 : 1;
__REG32 USyncMode       : 1;
__REG32                 :30;
} __spdif_srcd_bits;

/* PhaseConfig Register (SRPC) */
typedef struct{
__REG32                 : 3;
__REG32 GainSel         : 3;
__REG32 LOCK            : 1;
__REG32 ClkSrc_Sel      : 4;
__REG32                 :21;
} __spdif_srpc_bits;

/* InterruptEn Register (SIE) */
typedef struct{
__REG32 PdirFul         : 1;
__REG32 TxEm            : 1;
__REG32 LockLoss        : 1;
__REG32 PdirResyn       : 1;
__REG32 PdirUnOv        : 1;
__REG32 UQErr           : 1;
__REG32 UQSync          : 1;
__REG32 QRxOv           : 1;
__REG32 QRxFul          : 1;
__REG32 URxOv           : 1;
__REG32 URxFul          : 1;
__REG32                 : 3;
__REG32 BitErr          : 1;
__REG32 SymErr          : 1;
__REG32 ValNoGood       : 1;
__REG32 CNew            : 1;
__REG32 TxResyn         : 1;
__REG32 TxUnOv          : 1;
__REG32 Lock            : 1;
__REG32                 :11;
} __spdif_sie_bits;

/* Interrupt Registers */
typedef union{
  /*SPDIF_SIS*/
  struct{
  __REG32 PdirFul         : 1;
  __REG32 TxEm            : 1;
  __REG32 LockLoss        : 1;
  __REG32 PdirResyn       : 1;
  __REG32 PdirUnOv        : 1;
  __REG32 UQErr           : 1;
  __REG32 UQSync          : 1;
  __REG32 QRxOv           : 1;
  __REG32 QRxFul          : 1;
  __REG32 URxOv           : 1;
  __REG32 URxFul          : 1;
  __REG32                 : 3;
  __REG32 BitErr          : 1;
  __REG32 SymErr          : 1;
  __REG32 ValNoGood       : 1;
  __REG32 CNew            : 1;
  __REG32 TxResyn         : 1;
  __REG32 TxUnOv          : 1;
  __REG32 Lock            : 1;
  __REG32                 :11;
  };
  /*SPDIF_SIC*/
  struct{
  __REG32                 : 2;
  __REG32 LockLoss        : 1;
  __REG32 PdirResyn       : 1;
  __REG32 PdirUnOv        : 1;
  __REG32 UQErr           : 1;
  __REG32 UQSync          : 1;
  __REG32 QRxOv           : 1;
  __REG32                 : 1;
  __REG32 URxOv           : 1;
  __REG32                 : 4;
  __REG32 BitErr          : 1;
  __REG32 SymErr          : 1;
  __REG32 ValNoGood       : 1;
  __REG32 CNew            : 1;
  __REG32 TxResyn         : 1;
  __REG32 TxUnOv          : 1;
  __REG32 Lock            : 1;
  __REG32                 :11;
  } __SIC;
} __spdif_sis_bits;

/* SPDIFRcvLeft Register (SRL) */
typedef struct{
__REG32 RcvDataLeft     :24;
__REG32                 : 8;
} __spdif_srl_bits;

/* SPDIFRcvRight Register (SRR) */
typedef struct{
__REG32 RcvDataRight    :24;
__REG32                 : 8;
} __spdif_srr_bits;

/* SPDIFRcvCChannel_h Register (SRCSH) */
typedef struct{
__REG32 RxCChannel_h    :24;
__REG32                 : 8;
} __spdif_srcsh_bits;

/* SPDIFRcvCChannel_h Register (SRCSL) */
typedef struct{
__REG32 RxCChannel_l    :24;
__REG32                 : 8;
} __spdif_srcsl_bits;

/* UChannelRcv Register (SRU) */
typedef struct{
__REG32 RxUChannel      :24;
__REG32                 : 8;
} __spdif_squ_bits;

/* QChannelRcv Register (SRQ) */
typedef struct{
__REG32 RxQChannel      :24;
__REG32                 : 8;
} __spdif_srq_bits;

/* SPDIFTxLeft (STL) */
typedef struct{
__REG32 TxDataLeft      :24;
__REG32                 : 8;
} __spdif_stl_bits;

/* SPDIFTxRight (STR) */
typedef struct{
__REG32 TxDataRight     :24;
__REG32                 : 8;
} __spdif_str_bits;

/* SPDIFTxCChannelCons_h (STCSCH) */
typedef struct{
__REG32 TxCChannelCons_h  :24;
__REG32                   : 8;
} __spdif_stcsch_bits;

/* SPDIFTxCChannelCons_l (STCSCL) */
typedef struct{
__REG32 TxCChannelCons_l  :24;
__REG32                   : 8;
} __spdif_stcscl_bits;

/* FreqMeas Register */
typedef struct{
__REG32 FreqMeas        :24;
__REG32                 : 8;
} __spdif_srfm_bits;

/* SPDIFTxClk Register (STC) */
typedef struct{
__REG32 TxClk_DF        : 7;
__REG32                 : 1;
__REG32 TxClk_Source    : 3;
__REG32 SYSCLK_DF       : 9;
__REG32                 :12;
} __spdif_stc_bits;

/* ASRC Control Register (ASRCTR) */
typedef struct {
__REG32 ASRCEN    : 1;
__REG32 ASREA     : 1;
__REG32 ASREB     : 1;
__REG32 ASREC     : 1;
__REG32 SRST      : 1;
__REG32           : 8;
__REG32 IDRA      : 1;
__REG32 USRA      : 1;
__REG32 IDRB      : 1;
__REG32 USRB      : 1;
__REG32 IDRC      : 1;
__REG32 USRC      : 1;
__REG32           : 1;
__REG32 ATSA      : 1;
__REG32 ATSB      : 1;
__REG32 ATSC      : 1;
__REG32           : 9;
} __asrc_asrctr_bits;

/* Interrupt Enable Register (ASRIER) */
typedef struct {
__REG32 ADIEA     : 1;
__REG32 ADIEB     : 1;
__REG32 ADIEC     : 1;
__REG32 ADOEA     : 1;
__REG32 ADOEB     : 1;
__REG32 ADOEC     : 1;
__REG32 AOLIE     : 1;
__REG32 AFPWE     : 1;
__REG32           :24;
} __asrc_asrier_bits;

/* Channel Number Configuration Register (ASRCNCR) */
typedef struct {
__REG32 ANCA      : 3;
__REG32 ANCB      : 3;
__REG32 ANCC      : 3;
__REG32           :23;
} __asrc_asrcncr_bits;

/* Filter Configuration Status Register (ASRCFG) */
typedef struct {
__REG32           : 6;
__REG32 PREMODA   : 2;
__REG32 POSTMODA  : 2;
__REG32 PREMODB   : 2;
__REG32 POSTMODB  : 2;
__REG32 PREMODC   : 2;
__REG32 POSTMODC  : 2;
__REG32 NDPRA     : 1;
__REG32 NDPRB     : 1;
__REG32 NDPRC     : 1;
__REG32 INIRQA    : 1;
__REG32 INIRQB    : 1;
__REG32 INIRQC    : 1;
__REG32           : 8;
} __asrc_asrcfg_bits;

/* ASRC Clock Source Register (ASRCSR) */
typedef struct {
__REG32 AICSA     : 4;
__REG32 AICSB     : 4;
__REG32 AICSC     : 4;
__REG32 AOCSA     : 4;
__REG32 AOCSB     : 4;
__REG32 AOCSC     : 4;
__REG32           : 8;
} __asrc_asrcsr_bits;

/* ASRC Clock Divider Register (ASRCDR1) */
typedef struct {
__REG32 AICPA     : 3;
__REG32 AICDA     : 3;
__REG32 AICPB     : 3;
__REG32 AICDB     : 3;
__REG32 AOCPA     : 3;
__REG32 AOCDA     : 3;
__REG32 AOCPB     : 3;
__REG32 AOCDB     : 3;
__REG32           : 8;
} __asrc_asrcdr1_bits;

/* ASRC Clock Divider Register (ASRCDR2) */
typedef struct {
__REG32 AICPC     : 3;
__REG32 AICDC     : 3;
__REG32 AOCPC     : 3;
__REG32 AOCDC     : 3;
__REG32           :20;
} __asrc_asrcdr2_bits;

/* ASRC Status Register (ASRSTR) */
typedef struct {
__REG32 AIDEA     : 1;
__REG32 AIDEB     : 1;
__REG32 AIDEC     : 1;
__REG32 AODFA     : 1;
__REG32 AODFB     : 1;
__REG32 AODFC     : 1;
__REG32 AOLE      : 1;
__REG32 FPWT      : 1;
__REG32 AIDUA     : 1;
__REG32 AIDUB     : 1;
__REG32 AIDUC     : 1;
__REG32 AODOA     : 1;
__REG32 AODOB     : 1;
__REG32 AODOC     : 1;
__REG32 AIOLA     : 1;
__REG32 AIOLB     : 1;
__REG32 AIOLC     : 1;
__REG32 AOOLA     : 1;
__REG32 AOOLB     : 1;
__REG32 AOOLC     : 1;
__REG32 ATQOL     : 1;
__REG32 DSLCNT    : 1;
__REG32           :10;
} __asrc_asrstr_bits;

/* ASRC Task Queue FIFO Register(ASRTFR1) */
typedef struct {
__REG32           : 6;
__REG32 TF_BASE   : 7;
__REG32 TF_FILL   : 7;
__REG32           :12;
} __asrc_asrtfr1_bits;

/* Channel Counter Register (ASRCCR) */
typedef struct {
__REG32 ACIA      : 4;
__REG32 ACIB      : 4;
__REG32 ACIC      : 4;
__REG32 ACOA      : 4;
__REG32 ACOB      : 4;
__REG32 ACOC      : 4;
__REG32           : 8;
} __asrc_asrccr_bits;

/* Ideal Ratio Register for Pair A High (ASRIDRHA) */
typedef struct {
__REG32 IDRATIOA  : 8;
__REG32           :24;
} __asrc_asridrha_bits;

/* Ideal Ratio Register for Pair B High (ASRIDRHB) */
typedef struct {
__REG32 IDRATIOB  : 8;
__REG32           :24;
} __asrc_asridrhb_bits;

/* Ideal Ratio Register for Pair C High (ASRIDRHC) */
typedef struct {
__REG32 IDRATIOC  : 8;
__REG32           :24;
} __asrc_asridrhc_bits;

/* ASRC 76 KHz Period Register in Terms of Master Clock (ASR76K) */
typedef struct {
__REG32 ASR76K    :17;
__REG32           :15;
} __asrc_asr76k_bits;

/* ASRC 56 kHz Period Register in Terms of Master Clock (ASR56K) */
typedef struct {
__REG32 ASR56K    :17;
__REG32           :15;
} __asrc_asr56k_bits;

/* ESAI Control Register (ECR) */
typedef struct {
__REG32 ESAIEN          : 1;
__REG32 ERST            : 1;
__REG32                 :14;
__REG32 ERO             : 1;
__REG32 ERI             : 1;
__REG32 ETO             : 1;
__REG32 ETI             : 1;
__REG32                 :12;
} __esai_ecr_bits;

/* ESAI Status Register (ESR) */
typedef struct {
__REG32 RD              : 1;
__REG32 RED             : 1;
__REG32 RDE             : 1;
__REG32 RLS             : 1;
__REG32 TD              : 1;
__REG32 TED             : 1;
__REG32 TDE             : 1;
__REG32 TLS             : 1;
__REG32 TFE             : 1;
__REG32 RFF             : 1;
__REG32 TINIT           : 1;
__REG32                 :21;
} __esai_esr_bits;

/* Transmit FIFO Configuration Register (TFCR) */
typedef struct {
__REG32 TFEN            : 1;
__REG32 TFR             : 1;
__REG32 TE0             : 1;
__REG32 TE1             : 1;
__REG32 TE2             : 1;
__REG32 TE3             : 1;
__REG32 TE4             : 1;
__REG32 TE5             : 1;
__REG32 TFWM            : 8;
__REG32 TWA             : 3;
__REG32 TIEN            : 1;
__REG32                 :12;
} __esai_tfcr_bits;

/* Transmit FIFO Status Register (TFSR) */
typedef struct {
__REG32 TFCNT           : 8;
__REG32 NTFI            : 3;
__REG32                 : 1;
__REG32 NTFO            : 3;
__REG32                 :17;
} __esai_tfsr_bits;

/* Receive FIFO Configuration Register (RFCR) */
typedef struct {
__REG32 RFEN            : 1;
__REG32 RFR             : 1;
__REG32 RE0             : 1;
__REG32 RE1             : 1;
__REG32 RE2             : 1;
__REG32 RE3             : 1;
__REG32                 : 2;
__REG32 RFWM            : 8;
__REG32 RWA             : 3;
__REG32 REXT            : 1;
__REG32                 :12;
} __esai_rfcr_bits;

/* Receive FIFO Status Register (RFSR) */
typedef struct {
__REG32 RFCNT           : 8;
__REG32 NRFO            : 2;
__REG32                 : 2;
__REG32 NRFI            : 2;
__REG32                 :18;
} __esai_rfsr_bits;

/* ESAI Transmit Data Registers (TX5, TX4, TX3, TX2,TX1,TX0) */
typedef struct {
__REG32 TX              :24;
__REG32                 : 8;
} __esai_tx_bits;

/* ESAI Transmit Slot Register (TSR) */
typedef struct {
__REG32 TSR             :24;
__REG32                 : 8;
} __esai_tsr_bits;

/* ESAI Receive Data Registers (RX3, RX2, RX1, RX0) */
typedef struct {
__REG32 RX              :24;
__REG32                 : 8;
} __esai_rx_bits;

/* ESAI Status Register (SAISR) */
typedef struct {
__REG32 IF0             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32                 : 3;
__REG32 RFS             : 1;
__REG32 ROE             : 1;
__REG32 RDF             : 1;
__REG32 REDF            : 1;
__REG32 RODF            : 1;
__REG32                 : 2;
__REG32 TFS             : 1;
__REG32 TUE             : 1;
__REG32 TDE             : 1;
__REG32 TEDE            : 1;
__REG32 TODFE           : 1;
__REG32                 :14;
} __esai_saisr_bits;

/* ESAI Common Control Register (SAICR) */
typedef struct {
__REG32 OF0             : 1;
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32                 : 3;
__REG32 SYN             : 1;
__REG32 TEBE            : 1;
__REG32 ALC             : 1;
__REG32                 :23;
} __esai_saicr_bits;

/* ESAI Transmit Control Register (TCR) */
typedef struct {
__REG32 TE0             : 1;
__REG32 TE1             : 1;
__REG32 TE2             : 1;
__REG32 TE3             : 1;
__REG32 TE4             : 1;
__REG32 TE5             : 1;
__REG32 TSHFD           : 1;
__REG32 TWA             : 1;
__REG32 TMOD            : 2;
__REG32 TSWS            : 5;
__REG32 TFSL            : 1;
__REG32 TFSR            : 1;
__REG32 PADC            : 1;
__REG32                 : 1;
__REG32 TPR             : 1;
__REG32 TEIE            : 1;
__REG32 TDEIE           : 1;
__REG32 TIE             : 1;
__REG32 TLIE            : 1;
__REG32                 : 8;
} __esai_tcr_bits;

/* ESAI Transmitter Clock Control Register (TCCR) */
typedef struct {
__REG32 TPM             : 8;
__REG32 TPSR            : 1;
__REG32 TDC             : 5;
__REG32 TFP             : 4;
__REG32 TCKP            : 1;
__REG32 TFSP            : 1;
__REG32 THCKP           : 1;
__REG32 TCKD            : 1;
__REG32 TFSD            : 1;
__REG32 THCKD           : 1;
__REG32                 : 8;
} __esai_tccr_bits;

/* ESAI Receive Control Register (RCR) */
typedef struct {
__REG32 RE0               : 1;
__REG32 RE1               : 1;
__REG32 RE2               : 1;
__REG32 RE3               : 1;
__REG32                   : 2;
__REG32 RSHFD             : 1;
__REG32 RWA               : 1;
__REG32 RMOD0             : 1;
__REG32 RMOD1             : 1;
__REG32 RSWS              : 5;
__REG32 RFSL              : 1;
__REG32 RFSR              : 1;
__REG32                   : 2;
__REG32 RPR               : 1;
__REG32 REIE              : 1;
__REG32 RDEIE             : 1;
__REG32 RIE               : 1;
__REG32 RLIE              : 1;
__REG32                   : 8;
} __esai_rcr_bits;

/* ESAI Receiver Clock Control Register (RCCR) */
typedef struct {
__REG32 RPM             : 8;
__REG32 RPSR            : 1;
__REG32 RDC             : 5;
__REG32 RFP             : 4;
__REG32 RCKP            : 1;
__REG32 RFSP            : 1;
__REG32 RHCKP           : 1;
__REG32 RCKD            : 1;
__REG32 RFSD            : 1;
__REG32 RHCKD           : 1;
__REG32                 : 8;
} __esai_rccr_bits;

/* ESAI Transmit Slot Mask Register A/B */
typedef struct {
__REG32 TS              :16;
__REG32                 :16;
} __esai_tsm_bits;

/* ESAI Receive Slot Mask Register A/B */
typedef struct {
__REG32 RS              :16;
__REG32                 :16;
} __esai_rsm_bits;

/* Port C Direction Register (PRRC) */
typedef struct {
__REG32 PDC             :12;
__REG32                 :20;
} __esai_prrc_bits;

/* Port C Control Register (PCRC) */
typedef struct {
__REG32 PC              :12;
__REG32                 :20;
} __esai_pcrc_bits;

/* Ethernet Interrupt Event Register (EIR)
   Interrupt Mask Register (EIMR) */
typedef struct {
__REG32          :19;
__REG32 UN       : 1;
__REG32 RL       : 1;
__REG32 LC       : 1;
__REG32 EBERR    : 1;
__REG32 MII      : 1;
__REG32 RXB      : 1;
__REG32 RXF      : 1;
__REG32 TXB      : 1;
__REG32 TXF      : 1;
__REG32 GRA      : 1;
__REG32 BABT     : 1;
__REG32 BABR     : 1;
__REG32 HBERR    : 1;
} __fec_eir_bits;

/* Receive Descriptor Active Register (RDAR) */
typedef struct {
__REG32               :24;
__REG32 R_DES_ACTIVE  : 1;
__REG32               : 7;
} __fec_rdar_bits;

/* Transmit Descriptor Active Register (TDAR) */
typedef struct {
__REG32               :24;
__REG32 X_DES_ACTIVE  : 1;
__REG32               : 7;
} __fec_tdar_bits;

/* Ethernet Control Register (ECR) */
typedef struct {
__REG32 RESET         : 1;
__REG32 ETHER_EN      : 1;
__REG32               :30;
} __fec_ecr_bits;

/* MII Management Frame Register (MMFR) */
typedef struct {
__REG32 DATA          :16;
__REG32 TA            : 2;
__REG32 RA            : 5;
__REG32 PA            : 5;
__REG32 OP            : 2;
__REG32 ST            : 2;
} __fec_mmfr_bits;

/* MII Speed Control Register (MSCR) */
typedef struct {
__REG32               : 1;
__REG32 MII_SPEED     : 6;
__REG32 DIS_PREAMBLE  : 1;
__REG32               :24;
} __fec_mscr_bits;

/* MIB Control Register (MIBC) */
typedef struct {
__REG32               :30;
__REG32 MIB_IDLE      : 1;
__REG32 MIB_DIS       : 1;
} __fec_mibc_bits;

/* Receive Control Register (RCR) */
typedef struct {
__REG32 LOOP          : 1;
__REG32 DRT           : 1;
__REG32 MII_MODE      : 1;
__REG32 PROM          : 1;
__REG32 BC_REJ        : 1;
__REG32 FCE           : 1;
__REG32               :10;
__REG32 MAX_FL        :11;
__REG32               : 5;
} __fec_rcr_bits;

/* Transmit Control Register (TCR) */
typedef struct {
__REG32 GTS           : 1;
__REG32 HBC           : 1;
__REG32 FDEN          : 1;
__REG32 TFC_PAUSE     : 1;
__REG32 RFC_PAUSE     : 1;
__REG32               :27;
} __fec_tcr_bits;

/* Physical Address High Register (PAUR) */
typedef struct {
__REG32 TYPE          :16;
__REG32 PADDR2        :16;
} __fec_paur_bits;

/* Opcode/Pause Duration Register (OPD) */
typedef struct {
__REG32 PAUSE_DUR     :16;
__REG32 OPCODE        :16;
} __fec_opd_bits;

/* FIFO Transmit FIFO Watermark Register (TFWR) */
typedef struct {
__REG32 X_WMRK        : 2;
__REG32               :30;
} __fec_tfwr_bits;

/* FIFO Receive Bound Register (FRBR) */
typedef struct {
__REG32               : 2;
__REG32 R_BOUND       : 8;
__REG32               :22;
} __fec_frbr_bits;

/* FIFO Receive Start Register (FRSR) */
typedef struct {
__REG32               : 2;
__REG32 R_FSTART      : 8;
__REG32               :22;
} __fec_frsr_bits;

/* Receive Buffer Size Register (EMRBR) */
typedef struct {
__REG32               : 4;
__REG32 R_BUF_SIZE    : 7;
__REG32               :21;
} __fec_emrbr_bits;

/* Module Configuration Register (MCR) */
typedef struct {
__REG32 MAXMB           : 6;
__REG32                 : 2;
__REG32 IDAM            : 2;
__REG32                 : 2;
__REG32 AEN             : 1;
__REG32 LPRO_EN         : 1;
__REG32                 : 2;
__REG32 BCC             : 1;
__REG32 SRX_DIS         : 1;
__REG32 DOZE            : 1;
__REG32 WAK_SRC         : 1;
__REG32 LPM_ACK         : 1;
__REG32 WRN_EN          : 1;
__REG32 SLF_WAK         : 1;
__REG32 SUPV            : 1;
__REG32 FRZ_ACK         : 1;
__REG32 SOFT_RST        : 1;
__REG32 WAK_MSK         : 1;
__REG32 NOT_RDY         : 1;
__REG32 HALT            : 1;
__REG32 FEN             : 1;
__REG32 FRZ             : 1;
__REG32 MDIS            : 1;
} __can_mcr_bits;

/* Control Register (CTRL) */
typedef struct {
__REG32 PROPSEG         : 3;
__REG32 LOM             : 1;
__REG32 LBUF            : 1;
__REG32 TSYN            : 1;
__REG32 BOFF_REC        : 1;
__REG32 SMP             : 1;
__REG32                 : 2;
__REG32 RWRN_MSK        : 1;
__REG32 TWRN_MSK        : 1;
__REG32 LPB             : 1;
__REG32 CLK_SRC         : 1;
__REG32 ERR_MSK         : 1;
__REG32 BOFF_MSK        : 1;
__REG32 PSEG2           : 3;
__REG32 PSEG1           : 3;
__REG32 RJW             : 2;
__REG32 PRESDIV         : 8;
} __can_ctrl_bits;

/* Free Running Timer (TIMER) */
typedef struct {
__REG32 TIMER           :16;
__REG32                 :16;
} __can_timer_bits;

/* Rx Global Mask (RXGMASK) */
typedef struct {
__REG32 MI0             : 1;
__REG32 MI1             : 1;
__REG32 MI2             : 1;
__REG32 MI3             : 1;
__REG32 MI4             : 1;
__REG32 MI5             : 1;
__REG32 MI6             : 1;
__REG32 MI7             : 1;
__REG32 MI8             : 1;
__REG32 MI9             : 1;
__REG32 MI10            : 1;
__REG32 MI11            : 1;
__REG32 MI12            : 1;
__REG32 MI13            : 1;
__REG32 MI14            : 1;
__REG32 MI15            : 1;
__REG32 MI16            : 1;
__REG32 MI17            : 1;
__REG32 MI18            : 1;
__REG32 MI19            : 1;
__REG32 MI20            : 1;
__REG32 MI21            : 1;
__REG32 MI22            : 1;
__REG32 MI23            : 1;
__REG32 MI24            : 1;
__REG32 MI25            : 1;
__REG32 MI26            : 1;
__REG32 MI27            : 1;
__REG32 MI28            : 1;
__REG32 MI29            : 1;
__REG32 MI30            : 1;
__REG32 MI31            : 1;
} __can_rxgmask_bits;

/* Error Counter Register (ECR) */
typedef struct {
__REG32 Tx_Err_Counter  : 8;
__REG32 Rx_Err_Counter  : 8;
__REG32                 :16;
} __can_ecr_bits;

/* Error and Status Register (ESR) */
typedef struct {
__REG32 WAK_INT         : 1;
__REG32 ERR_INT         : 1;
__REG32 BOFF_INT        : 1;
__REG32                 : 1;
__REG32 FLT_CONF        : 2;
__REG32 TXRX            : 1;
__REG32 IDLE            : 1;
__REG32 RX_WRN          : 1;
__REG32 TX_WRN          : 1;
__REG32 STF_ERR         : 1;
__REG32 FRM_ERR         : 1;
__REG32 CRC_ERR         : 1;
__REG32 ACK_ERR         : 1;
__REG32 BIT0_ERR        : 1;
__REG32 BIT1_ERR        : 1;
__REG32 RWRN_INT        : 1;
__REG32 TWRN_INT        : 1;
__REG32                 :14;
} __can_esr_bits;

/* Interrupt Masks 2 Register (IMASK2) */
typedef struct {
__REG32 BUF32M          : 1;
__REG32 BUF33M          : 1;
__REG32 BUF34M          : 1;
__REG32 BUF35M          : 1;
__REG32 BUF36M          : 1;
__REG32 BUF37M          : 1;
__REG32 BUF38M          : 1;
__REG32 BUF39M          : 1;
__REG32 BUF40M          : 1;
__REG32 BUF41M          : 1;
__REG32 BUF42M          : 1;
__REG32 BUF43M          : 1;
__REG32 BUF44M          : 1;
__REG32 BUF45M          : 1;
__REG32 BUF46M          : 1;
__REG32 BUF47M          : 1;
__REG32 BUF48M          : 1;
__REG32 BUF49M          : 1;
__REG32 BUF50M          : 1;
__REG32 BUF51M          : 1;
__REG32 BUF52M          : 1;
__REG32 BUF53M          : 1;
__REG32 BUF54M          : 1;
__REG32 BUF55M          : 1;
__REG32 BUF56M          : 1;
__REG32 BUF57M          : 1;
__REG32 BUF58M          : 1;
__REG32 BUF59M          : 1;
__REG32 BUF60M          : 1;
__REG32 BUF61M          : 1;
__REG32 BUF62M          : 1;
__REG32 BUF63M          : 1;
} __can_imask2_bits;

/* Interrupt Masks 1 Register (IMASK1) */
typedef struct {
__REG32 BUF0M           : 1;
__REG32 BUF1M           : 1;
__REG32 BUF2M           : 1;
__REG32 BUF3M           : 1;
__REG32 BUF4M           : 1;
__REG32 BUF5M           : 1;
__REG32 BUF6M           : 1;
__REG32 BUF7M           : 1;
__REG32 BUF8M           : 1;
__REG32 BUF9M           : 1;
__REG32 BUF10M          : 1;
__REG32 BUF11M          : 1;
__REG32 BUF12M          : 1;
__REG32 BUF13M          : 1;
__REG32 BUF14M          : 1;
__REG32 BUF15M          : 1;
__REG32 BUF16M          : 1;
__REG32 BUF17M          : 1;
__REG32 BUF18M          : 1;
__REG32 BUF19M          : 1;
__REG32 BUF20M          : 1;
__REG32 BUF21M          : 1;
__REG32 BUF22M          : 1;
__REG32 BUF23M          : 1;
__REG32 BUF24M          : 1;
__REG32 BUF25M          : 1;
__REG32 BUF26M          : 1;
__REG32 BUF27M          : 1;
__REG32 BUF28M          : 1;
__REG32 BUF29M          : 1;
__REG32 BUF30M          : 1;
__REG32 BUF31M          : 1;
} __can_imask1_bits;

/* Interrupt Flags 2 Register (IFLAG2) */
typedef struct {
__REG32 BUF32I          : 1;
__REG32 BUF33I          : 1;
__REG32 BUF34I          : 1;
__REG32 BUF35I          : 1;
__REG32 BUF36I          : 1;
__REG32 BUF37I          : 1;
__REG32 BUF38I          : 1;
__REG32 BUF39I          : 1;
__REG32 BUF40I          : 1;
__REG32 BUF41I          : 1;
__REG32 BUF42I          : 1;
__REG32 BUF43I          : 1;
__REG32 BUF44I          : 1;
__REG32 BUF45I          : 1;
__REG32 BUF46I          : 1;
__REG32 BUF47I          : 1;
__REG32 BUF48I          : 1;
__REG32 BUF49I          : 1;
__REG32 BUF50I          : 1;
__REG32 BUF51I          : 1;
__REG32 BUF52I          : 1;
__REG32 BUF53I          : 1;
__REG32 BUF54I          : 1;
__REG32 BUF55I          : 1;
__REG32 BUF56I          : 1;
__REG32 BUF57I          : 1;
__REG32 BUF58I          : 1;
__REG32 BUF59I          : 1;
__REG32 BUF60I          : 1;
__REG32 BUF61I          : 1;
__REG32 BUF62I          : 1;
__REG32 BUF63I          : 1;
} __can_iflag2_bits;

/* Interrupt Flags 2 Register (IFLAG1) */
typedef struct {
__REG32 BUF0I           : 1;
__REG32 BUF1I           : 1;
__REG32 BUF2I           : 1;
__REG32 BUF3I           : 1;
__REG32 BUF4I           : 1;
__REG32 BUF5I           : 1;
__REG32 BUF6I           : 1;
__REG32 BUF7I           : 1;
__REG32 BUF8I           : 1;
__REG32 BUF9I           : 1;
__REG32 BUF10I          : 1;
__REG32 BUF11I          : 1;
__REG32 BUF12I          : 1;
__REG32 BUF13I          : 1;
__REG32 BUF14I          : 1;
__REG32 BUF15I          : 1;
__REG32 BUF16I          : 1;
__REG32 BUF17I          : 1;
__REG32 BUF18I          : 1;
__REG32 BUF19I          : 1;
__REG32 BUF20I          : 1;
__REG32 BUF21I          : 1;
__REG32 BUF22I          : 1;
__REG32 BUF23I          : 1;
__REG32 BUF24I          : 1;
__REG32 BUF25I          : 1;
__REG32 BUF26I          : 1;
__REG32 BUF27I          : 1;
__REG32 BUF28I          : 1;
__REG32 BUF29I          : 1;
__REG32 BUF30I          : 1;
__REG32 BUF31I          : 1;
} __can_iflag1_bits;

/* Device Control Configuration Register (DCCR) */
typedef struct{
__REG32 MDA           : 8;
__REG32               :15;
__REG32 MRS           : 1;
__REG32 MHRE          : 1;
__REG32               : 1;
__REG32 MLK           : 1;
__REG32               : 1;
__REG32 MCS           : 2;
__REG32 LBM           : 1;
__REG32 MDE           : 1;
} __mlb_dccr_bits;

/* System Status Configuration Register (SSCR) */
typedef struct{
__REG32 SDR           : 1;
__REG32 SDNL          : 1;
__REG32 SDNU          : 1;
__REG32 SDCS          : 1;
__REG32 SDSC          : 1;
__REG32 SDML          : 1;
__REG32 SDMU          : 1;
__REG32 SSRE          : 1;
__REG32               :24;
} __mlb_sscr_bits;

/* System Mask Configuration Register (SMCR) */
typedef struct{
__REG32 SMR           : 1;
__REG32 SMNL          : 1;
__REG32 SMNU          : 1;
__REG32 SMCS          : 1;
__REG32 SMSC          : 1;
__REG32 SMML          : 1;
__REG32 SMMU          : 1;
__REG32               :25;
} __mlb_smcr_bits;

/* Version Control Configuration Register (VCCR) */
typedef struct{
__REG32 MMI           : 8;
__REG32 MMA           : 8;
__REG32 UMI           : 8;
__REG32 UMA           : 8;
} __mlb_vccr_bits;

/* Synchronous Base Address Configuration Register (SBCR) */
typedef struct{
__REG32 STBA          :16;
__REG32 SRBA          :16;
} __mlb_sbcr_bits;

/* Asynchronous Base Address Configuration Register (ABCR) */
typedef struct{
__REG32 ATBA          :16;
__REG32 ARBA          :16;
} __mlb_abcr_bits;

/* Control Base Address Configuration Register (CBCR) */
typedef struct{
__REG32 CTBA          :16;
__REG32 CRBA          :16;
} __mlb_cbcr_bits;

/* Isochronous Base Address Configuration Register (IBCR) */
typedef struct{
__REG32 ITBA          :16;
__REG32 IRBA          :16;
} __mlb_ibcr_bits;

/* Channel Interrupt Configuration Register (CICR) */
typedef struct{
__REG32 C0SU          : 1;
__REG32 C1SU          : 1;
__REG32 C2SU          : 1;
__REG32 C3SU          : 1;
__REG32 C4SU          : 1;
__REG32 C5SU          : 1;
__REG32 C6SU          : 1;
__REG32 C7SU          : 1;
__REG32 C8SU          : 1;
__REG32 C9SU          : 1;
__REG32 C10SU         : 1;
__REG32 C11SU         : 1;
__REG32 C12SU         : 1;
__REG32 C13SU         : 1;
__REG32 C14SU         : 1;
__REG32 C15SU         : 1;
__REG32               :16;
} __mlb_cicr_bits;

/* Channel n Entry Configuration Register (CECRn) */
typedef struct{
__REG32 CA            : 8;
__REG32 FSPC_IPL      : 5;
__REG32 IPL           : 3;
__REG32 MPE           : 1;
__REG32 MDB           : 1;
__REG32 MBD           : 1;
__REG32 MBS           : 1;
__REG32 MBE           : 1;
__REG32               : 1;
__REG32 MLFS          : 1;
__REG32               : 2;
__REG32 MDS           : 2;
__REG32 FSE_FCE       : 1;
__REG32 CT            : 2;
__REG32 TR            : 1;
__REG32 CE            : 1;
} __mlb_cecr_bits;

/* Channel n Status Configuration Register (CSCRn) */
typedef struct{
__REG32 CBPE          : 1;
__REG32 CBDB          : 1;
__REG32 CBD           : 1;
__REG32 CBS           : 1;
__REG32 BE            : 1;
__REG32 ABE           : 1;
__REG32 LFS           : 1;
__REG32               : 1;
__REG32 PBPE          : 1;
__REG32 PBDB          : 1;
__REG32 PBD           : 1;
__REG32 PBS           : 1;
__REG32               : 4;
__REG32 RDY           : 1;
__REG32 GIRB_GB       : 1;
__REG32 IVB           : 2;
__REG32               :10;
__REG32 BF            : 1;
__REG32 BM            : 1;
} __mlb_cscr_bits;

/* Channel n Current Configuration Register (CCBCRn) */
typedef struct{
__REG32 BFA           :16;
__REG32 BCA           :16;
} __mlb_ccbcr_bits;

/* Channel n Next Buffer Configuration Register (CNBCRn) */
typedef struct{
__REG32 BEA           :16;
__REG32 BSA           :16;
} __mlb_cnbcr_bits;

/* Local Channel n Buffer Configuration Register (LCBCRn) */
typedef struct{
__REG32 SA            :13;
__REG32 BD            : 9;
__REG32 TH            :10;
} __mlb_lcbcr_bits;

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
__IO_REG32_BIT(CCM_PDR2,                  0x53F8000C,__READ_WRITE ,__ccm_pdr2_bits);
__IO_REG32_BIT(CCM_PDR3,                  0x53F80010,__READ_WRITE ,__ccm_pdr3_bits);
__IO_REG32_BIT(CCM_PDR4,                  0x53F80014,__READ_WRITE ,__ccm_pdr4_bits);
__IO_REG32_BIT(CCM_RCSR,                  0x53F80018,__READ_WRITE ,__ccm_rcsr_bits);
__IO_REG32_BIT(CCM_MPCTL,                 0x53F8001C,__READ_WRITE ,__ccm_mpctl_bits);
__IO_REG32_BIT(CCM_PPCTL,                 0x53F80020,__READ_WRITE ,__ccm_mpctl_bits);
__IO_REG32_BIT(CCM_ACMR,                  0x53F80024,__READ_WRITE ,__ccm_acmr_bits);
__IO_REG32_BIT(CCM_COSR,                  0x53F80028,__READ_WRITE ,__ccm_cosr_bits);
__IO_REG32_BIT(CCM_CGR0,                  0x53F8002C,__READ_WRITE ,__ccm_cgr0_bits);
__IO_REG32_BIT(CCM_CGR1,                  0x53F80030,__READ_WRITE ,__ccm_cgr1_bits);
__IO_REG32_BIT(CCM_CGR2,                  0x53F80034,__READ_WRITE ,__ccm_cgr2_bits);
__IO_REG32_BIT(CCM_CGR3,                  0x53F80038,__READ_WRITE ,__ccm_cgr3_bits);
__IO_REG32_BIT(CCM_DCVR0,                 0x53F80040,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR1,                 0x53F80044,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR2,                 0x53F80048,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_DCVR3,                 0x53F8004C,__READ_WRITE ,__ccm_dcvr_bits);
__IO_REG32_BIT(CCM_LTR0,                  0x53F80050,__READ_WRITE ,__ccm_ltr0_bits);
__IO_REG32_BIT(CCM_LTR1,                  0x53F80054,__READ_WRITE ,__ccm_ltr1_bits);
__IO_REG32_BIT(CCM_LTR2,                  0x53F80058,__READ_WRITE ,__ccm_ltr2_bits);
__IO_REG32_BIT(CCM_LTR3,                  0x53F8005C,__READ_WRITE ,__ccm_ltr3_bits);
__IO_REG32_BIT(CCM_LTBR0,                 0x53F80060,__READ       ,__ccm_ltbr0_bits);
__IO_REG32_BIT(CCM_LTBR1,                 0x53F80064,__READ       ,__ccm_ltbr1_bits);
__IO_REG32_BIT(CCM_PMCR0,                 0x53F80068,__READ_WRITE ,__ccm_pmcr0_bits);
__IO_REG32_BIT(CCM_PMCR1,                 0x53F8006C,__READ_WRITE ,__ccm_pmcr1_bits);
__IO_REG32_BIT(CCM_PMCR2,                 0x53F80070,__READ_WRITE ,__ccm_pmcr2_bits);

/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(IOMUX_GPR,                           0x43FAC000,__READ_WRITE ,__iomux_gpr_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CAPTURE,       0x43FAC004,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_COMPARE,       0x43FAC008,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_WDOG_RST,      0x43FAC00C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_0,       0x43FAC010,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_1,       0x43FAC014,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO2_0,       0x43FAC018,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO3_0,       0x43FAC01C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CLKO,          0x43FAC020,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_VSTBY,         0x43FAC024,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A0,            0x43FAC028,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A1,            0x43FAC02C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A2,            0x43FAC030,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A3,            0x43FAC034,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A4,            0x43FAC038,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A5,            0x43FAC03C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A6,            0x43FAC040,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A7,            0x43FAC044,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A8,            0x43FAC048,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A9,            0x43FAC04C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A10,           0x43FAC050,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_MA10,          0x43FAC054,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A11,           0x43FAC058,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A12,           0x43FAC05C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A13,           0x43FAC060,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A14,           0x43FAC064,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A15,           0x43FAC068,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A16,           0x43FAC06C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A17,           0x43FAC070,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A18,           0x43FAC074,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A19,           0x43FAC078,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A20,           0x43FAC07C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A21,           0x43FAC080,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A22,           0x43FAC084,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A23,           0x43FAC088,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A24,           0x43FAC08C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_A25,           0x43FAC090,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EB0,           0x43FAC094,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EB1,           0x43FAC098,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_OE,            0x43FAC09C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS0,           0x43FAC0A0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS1,           0x43FAC0A4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS2,           0x43FAC0A8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS3,           0x43FAC0AC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS4,           0x43FAC0B0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CS5,           0x43FAC0B4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NF_CE0,        0x43FAC0B8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LBA,           0x43FAC0BC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_BCLK,          0x43FAC0C0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RW,            0x43FAC0C4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFWE_B,        0x43FAC0C8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFRE_B,        0x43FAC0CC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFALE,         0x43FAC0D0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFCLE,         0x43FAC0D4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFWP_B,        0x43FAC0D8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NFRB,          0x43FAC0DC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D8,        0x43FAC0E0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D9,        0x43FAC0E4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D10,       0x43FAC0E8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D11,       0x43FAC0EC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D12,       0x43FAC0F0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D13,       0x43FAC0F4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D14,       0x43FAC0F8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_D15,       0x43FAC0FC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_MCLK,      0x43FAC100,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_VSYNC,     0x43FAC104,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_HSYNC,     0x43FAC108,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI_PIXCLK,    0x43FAC10C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_CLK,      0x43FAC110,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_DAT,      0x43FAC114,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C2_CLK,      0x43FAC118,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C2_DAT,      0x43FAC11C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_STXD4,         0x43FAC120,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SRXD4,         0x43FAC124,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SCK4,          0x43FAC128,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_STXFS4,        0x43FAC12C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_STXD5,         0x43FAC130,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SRXD5,         0x43FAC134,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SCK5,          0x43FAC138,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_STXFS5,        0x43FAC13C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SCKR,          0x43FAC140,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FSR,           0x43FAC144,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_HCKR,          0x43FAC148,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SCKT,          0x43FAC14C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FST,           0x43FAC150,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_HCKT,          0x43FAC154,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX5_RX0,       0x43FAC158,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX4_RX1,       0x43FAC15C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX3_RX2,       0x43FAC160,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX2_RX3,       0x43FAC164,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX1,           0x43FAC168,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TX0,           0x43FAC16C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MOSI,    0x43FAC170,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MISO,    0x43FAC174,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS0,     0x43FAC178,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS1,     0x43FAC17C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SCLK,    0x43FAC180,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SPI_RDY, 0x43FAC184,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RXD1,          0x43FAC188,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TXD1,          0x43FAC18C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RTS1,          0x43FAC190,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CTS1,          0x43FAC194,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RXD2,          0x43FAC198,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_TXD2,          0x43FAC19C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_RTS2,          0x43FAC1A0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CTS2,          0x43FAC1A4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBOTG_PWR,    0x43FAC1A8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBOTG_OC,     0x43FAC1AC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD0,           0x43FAC1B0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD1,           0x43FAC1B4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD2,           0x43FAC1B8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD3,           0x43FAC1BC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD4,           0x43FAC1C0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD5,           0x43FAC1C4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD6,           0x43FAC1C8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD7,           0x43FAC1CC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD8,           0x43FAC1D0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD9,           0x43FAC1D4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD10,          0x43FAC1D8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD11,          0x43FAC1DC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD12,          0x43FAC1E0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD13,          0x43FAC1E4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD14,          0x43FAC1E8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD15,          0x43FAC1EC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD16,          0x43FAC1F0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD17,          0x43FAC1F4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD18,          0x43FAC1F8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD19,          0x43FAC1FC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD20,          0x43FAC200,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD21,          0x43FAC204,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD22,          0x43FAC208,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LD23,          0x43FAC20C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_HSYNC,      0x43FAC210,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_FPSHIFT,    0x43FAC214,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_DRDY,       0x43FAC218,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CONTRAST,      0x43FAC21C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_VSYNC,      0x43FAC220,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_REV,        0x43FAC224,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_CLS,        0x43FAC228,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_D3_SPL,        0x43FAC22C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CMD,       0x43FAC230,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CLK,       0x43FAC234,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA0,     0x43FAC238,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA1,     0x43FAC23C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA2,     0x43FAC240,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA3,     0x43FAC244,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CMD,       0x43FAC248,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CLK,       0x43FAC24C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA0,     0x43FAC250,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA1,     0x43FAC254,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA2,     0x43FAC258,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA3,     0x43FAC25C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_CS0,       0x43FAC260,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_CS1,       0x43FAC264,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DIOR,      0x43FAC268,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DIOW,      0x43FAC26C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DMACK,     0x43FAC270,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_RESET_B,   0x43FAC274,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_IORDY,     0x43FAC278,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA0,     0x43FAC27C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA1,     0x43FAC280,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA2,     0x43FAC284,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA3,     0x43FAC288,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA4,     0x43FAC28C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA5,     0x43FAC290,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA6,     0x43FAC294,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA7,     0x43FAC298,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA8,     0x43FAC29C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA9,     0x43FAC2A0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA10,    0x43FAC2A4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA11,    0x43FAC2A8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA12,    0x43FAC2AC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA13,    0x43FAC2B0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA14,    0x43FAC2B4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DATA15,    0x43FAC2B8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_INTRQ,     0x43FAC2BC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_BUFF_EN,   0x43FAC2C0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DMARQ,     0x43FAC2C4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DA0,       0x43FAC2C8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DA1,       0x43FAC2CC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_ATA_DA2,       0x43FAC2D0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_MLB_CLK,       0x43FAC2D4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_MLB_DAT,       0x43FAC2D8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_MLB_SIG,       0x43FAC2DC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_CLK,    0x43FAC2E0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RX_CLK,    0x43FAC2E4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RX_DV,     0x43FAC2E8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_COL,       0x43FAC2EC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA0,    0x43FAC2F0,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA0,    0x43FAC2F4,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_EN,     0x43FAC2F8,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDC,       0x43FAC2FC,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDIO,      0x43FAC300,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_ERR,    0x43FAC304,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RX_ERR,    0x43FAC308,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_CRS,       0x43FAC30C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA1,    0x43FAC310,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA1,    0x43FAC314,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA2,    0x43FAC318,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA2,    0x43FAC31C,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RDATA3,    0x43FAC320,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TDATA3,    0x43FAC324,__READ_WRITE ,__iomux_sw_mux_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CAPTURE,       0x43FAC328,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_COMPARE,       0x43FAC32C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_WDOG_RST,      0x43FAC330,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_0,       0x43FAC334,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_1,       0x43FAC338,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO2_0,       0x43FAC33C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO3_0,       0x43FAC340,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RESET_IN_B,    0x43FAC344,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_POR_B,         0x43FAC348,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CLKO,          0x43FAC34C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE0,    0x43FAC350,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE1,    0x43FAC354,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CLK_MODE0,     0x43FAC358,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CLK_MODE1,     0x43FAC35C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_POWER_FAIL,    0x43FAC360,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_VSTBY,         0x43FAC364,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A0,            0x43FAC368,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A1,            0x43FAC36C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A2,            0x43FAC370,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A3,            0x43FAC374,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A4,            0x43FAC378,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A5,            0x43FAC37C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A6,            0x43FAC380,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A7,            0x43FAC384,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A8,            0x43FAC388,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A9,            0x43FAC38C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A10,           0x43FAC390,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_MA10,          0x43FAC394,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A11,           0x43FAC398,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A12,           0x43FAC39C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A13,           0x43FAC3A0,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A14,           0x43FAC3A4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A15,           0x43FAC3A8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A16,           0x43FAC3AC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A17,           0x43FAC3B0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A18,           0x43FAC3B4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A19,           0x43FAC3B8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A20,           0x43FAC3BC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A21,           0x43FAC3C0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A22,           0x43FAC3C4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A23,           0x43FAC3C8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A24,           0x43FAC3CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_A25,           0x43FAC3D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDBA1,         0x43FAC3D4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDBA0,         0x43FAC3D8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD0,           0x43FAC3DC,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1,           0x43FAC3E0,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2,           0x43FAC3E4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD3,           0x43FAC3E8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD4,           0x43FAC3EC,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD5,           0x43FAC3F0,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD6,           0x43FAC3F4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD7,           0x43FAC3F8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD8,           0x43FAC3FC,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD9,           0x43FAC400,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD10,          0x43FAC404,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD11,          0x43FAC408,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD12,          0x43FAC40C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD13,          0x43FAC410,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD14,          0x43FAC414,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD15,          0x43FAC418,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD16,          0x43FAC41C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD17,          0x43FAC420,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD18,          0x43FAC424,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD19,          0x43FAC428,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD20,          0x43FAC42C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD21,          0x43FAC430,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD22,          0x43FAC434,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD23,          0x43FAC438,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD24,          0x43FAC43C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD25,          0x43FAC440,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD26,          0x43FAC444,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD27,          0x43FAC448,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD28,          0x43FAC44C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD29,          0x43FAC450,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD30,          0x43FAC454,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD31,          0x43FAC458,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DQM0,          0x43FAC45C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DQM1,          0x43FAC460,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DQM2,          0x43FAC464,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DQM3,          0x43FAC468,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EB0,           0x43FAC46C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EB1,           0x43FAC470,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_OE,            0x43FAC474,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS0,           0x43FAC478,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS1,           0x43FAC47C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS2,           0x43FAC480,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS3,           0x43FAC484,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS4,           0x43FAC488,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CS5,           0x43FAC48C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NF_CE0,        0x43FAC490,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ECB,           0x43FAC494,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LBA,           0x43FAC498,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BCLK,          0x43FAC49C,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RW,            0x43FAC4A0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RAS,           0x43FAC4A4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CAS,           0x43FAC4A8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDWE,          0x43FAC4AC,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDCKE0,        0x43FAC4B0,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDCKE1,        0x43FAC4B4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDCLK,         0x43FAC4B8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDQS0,         0x43FAC4BC,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDQS1,         0x43FAC4C0,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDQS2,         0x43FAC4C4,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SDQS3,         0x43FAC4C8,__READ_WRITE ,__iomux_sw_pad_ctl_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFWE_B,        0x43FAC4CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFRE_B,        0x43FAC4D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFALE,         0x43FAC4D4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFCLE,         0x43FAC4D8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFWP_B,        0x43FAC4DC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NFRB,          0x43FAC4E0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D15,           0x43FAC4E4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D14,           0x43FAC4E8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D13,           0x43FAC4EC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D12,           0x43FAC4F0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D11,           0x43FAC4F4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D10,           0x43FAC4F8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D9,            0x43FAC4FC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D8,            0x43FAC500,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D7,            0x43FAC504,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D6,            0x43FAC508,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D5,            0x43FAC50C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D4,            0x43FAC510,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3,            0x43FAC514,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D2,            0x43FAC518,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D1,            0x43FAC51C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D0,            0x43FAC520,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D8,        0x43FAC524,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D9,        0x43FAC528,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D10,       0x43FAC52C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D11,       0x43FAC530,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D12,       0x43FAC534,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D13,       0x43FAC538,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D14,       0x43FAC53C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_D15,       0x43FAC540,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_MCLK,      0x43FAC544,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_VSYNC,     0x43FAC548,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_HSYNC,     0x43FAC54C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI_PIXCLK,    0x43FAC550,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_CLK,      0x43FAC554,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_DAT,      0x43FAC558,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C2_CLK,      0x43FAC55C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C2_DAT,      0x43FAC560,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_STXD4,         0x43FAC564,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SRXD4,         0x43FAC568,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SCK4,          0x43FAC56C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_STXFS4,        0x43FAC570,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_STXD5,         0x43FAC574,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SRXD5,         0x43FAC578,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SCK5,          0x43FAC57C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_STXFS5,        0x43FAC580,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SCKR,          0x43FAC584,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FSR,           0x43FAC588,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_HCKR,          0x43FAC58C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SCKT,          0x43FAC590,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FST,           0x43FAC594,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_HCKT,          0x43FAC598,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX5_RX0,       0x43FAC59C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX4_RX1,       0x43FAC5A0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX3_RX2,       0x43FAC5A4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX2_RX3,       0x43FAC5A8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX1,           0x43FAC5AC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TX0,           0x43FAC5B0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MOSI,    0x43FAC5B4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MISO,    0x43FAC5B8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS0,     0x43FAC5BC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS1,     0x43FAC5C0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SCLK,    0x43FAC5C4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SPI_RDY, 0x43FAC5C8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RXD1,          0x43FAC5CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TXD1,          0x43FAC5D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RTS1,          0x43FAC5D4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CTS1,          0x43FAC5D8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RXD2,          0x43FAC5DC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TXD2,          0x43FAC5E0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RTS2,          0x43FAC5E4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CTS2,          0x43FAC5E8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RTCK,          0x43FAC5EC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TCK,           0x43FAC5F0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TMS,           0x43FAC5F4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TDI,           0x43FAC5F8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TDO,           0x43FAC5FC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TRSTB,         0x43FAC600,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DE_B,          0x43FAC604,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SJC_MOD,       0x43FAC608,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBOTG_PWR,    0x43FAC60C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBOTG_OC,     0x43FAC610,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD0,           0x43FAC614,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD1,           0x43FAC618,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD2,           0x43FAC61C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD3,           0x43FAC620,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD4,           0x43FAC624,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD5,           0x43FAC628,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD6,           0x43FAC62C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD7,           0x43FAC630,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD8,           0x43FAC634,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD9,           0x43FAC638,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD10,          0x43FAC63C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD11,          0x43FAC640,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD12,          0x43FAC644,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD13,          0x43FAC648,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD14,          0x43FAC64C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD15,          0x43FAC650,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD16,          0x43FAC654,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD17,          0x43FAC658,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD18,          0x43FAC65C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD19,          0x43FAC660,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD20,          0x43FAC664,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD21,          0x43FAC668,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD22,          0x43FAC66C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_LD23,          0x43FAC670,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_HSYNC,      0x43FAC674,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_FPSHIFT,    0x43FAC678,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_DRDY,       0x43FAC67C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CONTRAST,      0x43FAC680,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_VSYNC,      0x43FAC684,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_REV,        0x43FAC688,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_CLS,        0x43FAC68C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_D3_SPL,        0x43FAC690,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CMD,       0x43FAC694,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CLK,       0x43FAC698,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA0,     0x43FAC69C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA1,     0x43FAC6A0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA2,     0x43FAC6A4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA3,     0x43FAC6A8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CMD,       0x43FAC6AC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CLK,       0x43FAC6B0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA0,     0x43FAC6B4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA1,     0x43FAC6B8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA2,     0x43FAC6BC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA3,     0x43FAC6C0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_CS0,       0x43FAC6C4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_CS1,       0x43FAC6C8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DIOR,      0x43FAC6CC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DIOW,      0x43FAC6D0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DMACK,     0x43FAC6D4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_RESET_B,   0x43FAC6D8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_IORDY,     0x43FAC6DC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA0,     0x43FAC6E0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA1,     0x43FAC6E4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA2,     0x43FAC6E8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA3,     0x43FAC6EC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA4,     0x43FAC6F0,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA5,     0x43FAC6F4,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA6,     0x43FAC6F8,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA7,     0x43FAC6FC,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA8,     0x43FAC700,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA9,     0x43FAC704,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA10,    0x43FAC708,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA11,    0x43FAC70C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA12,    0x43FAC710,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA13,    0x43FAC714,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA14,    0x43FAC718,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DATA15,    0x43FAC71C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_INTRQ,     0x43FAC720,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_BUFF_EN,   0x43FAC724,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DMARQ,     0x43FAC728,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DA0,       0x43FAC72C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DA1,       0x43FAC730,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_ATA_DA2,       0x43FAC734,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_MLB_CLK,       0x43FAC738,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_MLB_DAT,       0x43FAC73C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_MLB_SIG,       0x43FAC740,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_CLK,    0x43FAC744,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RX_CLK,    0x43FAC748,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RX_DV,     0x43FAC74C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_COL,       0x43FAC750,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA0,    0x43FAC754,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA0,    0x43FAC758,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_EN,     0x43FAC75C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDC,       0x43FAC760,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDIO,      0x43FAC764,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_ERR,    0x43FAC768,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RX_ERR,    0x43FAC76C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_CRS,       0x43FAC770,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA1,    0x43FAC774,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA1,    0x43FAC778,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA2,    0x43FAC77C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA2,    0x43FAC780,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RDATA3,    0x43FAC784,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TDATA3,    0x43FAC788,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EXT_ARMCLK,    0x43FAC78C,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TEST_MODE,     0x43FAC790,__READ_WRITE ,__iomux_sw_pad_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE_GRP4,  0x43FAC794,__READ_WRITE ,__iomux_sw_pad_ctl_grp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE_GRP5,  0x43FAC798,__READ_WRITE ,__iomux_sw_pad_ctl_grp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE_GRP1,  0x43FAC79C,__READ_WRITE ,__iomux_sw_pad_ctl_grp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE_GRP2,  0x43FAC7A0,__READ_WRITE ,__iomux_sw_pad_ctl_grp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRTYPE_GRP3,  0x43FAC7A4,__READ_WRITE ,__iomux_sw_pad_ctl_grp_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXCLK_AMX_SELECT_INPUT,   0x43FAC7A8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXFS_AMX_SELECT_INPUT,    0x43FAC7AC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_DA_AMX_SELECT_INPUT,      0x43FAC7B0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_DB_AMX_SELECT_INPUT,      0x43FAC7B4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_RXCLK_AMX_SELECT_INPUT,   0x43FAC7B8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_RXFS_AMX_SELECT_INPUT,    0x43FAC7BC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_TXCLK_AMX_SELECT_INPUT,   0x43FAC7C0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_TXFS_AMX_SELECT_INPUT,    0x43FAC7C4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_CAN1_IPP_IND_CANRX_SELECT_INPUT,          0x43FAC7C8,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CAN2_IPP_IND_CANRX_SELECT_INPUT,          0x43FAC7CC,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_32K_MUXED_IN_SELECT_INPUT,        0x43FAC7D0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_PMIC_RDY_SELECT_INPUT,            0x43FAC7D4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_CSPI1_IPP_IND_SS2_B_SELECT_INPUT,         0x43FAC7D8,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI1_IPP_IND_SS3_B_SELECT_INPUT,         0x43FAC7DC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_CSPI_CLK_IN_SELECT_INPUT,       0x43FAC7E0,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_DATAREADY_B_SELECT_INPUT,   0x43FAC7E4,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_MISO_SELECT_INPUT,          0x43FAC7E8,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_MOSI_SELECT_INPUT,          0x43FAC7EC,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS0_B_SELECT_INPUT,         0x43FAC7F0,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS1_B_SELECT_INPUT,         0x43FAC7F4,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS2_B_SELECT_INPUT,         0x43FAC7F8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_CSPI2_IPP_IND_SS3_B_SELECT_INPUT,         0x43FAC7FC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_EMI_IPP_IND_WEIM_DTACK_B_SELECT_INPUT,    0x43FAC800,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT4_IN_SELECT_INPUT,          0x43FAC804,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT5_IN_SELECT_INPUT,          0x43FAC808,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT6_IN_SELECT_INPUT,          0x43FAC80C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_DAT7_IN_SELECT_INPUT,          0x43FAC810,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_CARD_CLK_IN_SELECT_INPUT,      0x43FAC814,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_CMD_IN_SELECT_INPUT,           0x43FAC818,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT0_IN_SELECT_INPUT,          0x43FAC81C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT1_IN_SELECT_INPUT,          0x43FAC820,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT2_IN_SELECT_INPUT,          0x43FAC824,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT3_IN_SELECT_INPUT,          0x43FAC828,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_0_SELECT_INPUT,        0x43FAC82C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_10_SELECT_INPUT,       0x43FAC830,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_11_SELECT_INPUT,       0x43FAC834,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_1_SELECT_INPUT,        0x43FAC838,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_20_SELECT_INPUT,       0x43FAC83C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_21_SELECT_INPUT,       0x43FAC840,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_22_SELECT_INPUT,       0x43FAC844,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_2_SELECT_INPUT,        0x43FAC848,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_3_SELECT_INPUT,        0x43FAC84C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_4_SELECT_INPUT,        0x43FAC850,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_5_SELECT_INPUT,        0x43FAC854,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_6_SELECT_INPUT,        0x43FAC858,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_7_SELECT_INPUT,        0x43FAC85C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_8_SELECT_INPUT,        0x43FAC860,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO1_IPP_IND_G_IN_9_SELECT_INPUT,        0x43FAC864,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_0_SELECT_INPUT,        0x43FAC868,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_10_SELECT_INPUT,       0x43FAC86C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_11_SELECT_INPUT,       0x43FAC870,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_12_SELECT_INPUT,       0x43FAC874,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_13_SELECT_INPUT,       0x43FAC878,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_14_SELECT_INPUT,       0x43FAC87C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_15_SELECT_INPUT,       0x43FAC880,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_16_SELECT_INPUT,       0x43FAC884,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_17_SELECT_INPUT,       0x43FAC888,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_18_SELECT_INPUT,       0x43FAC88C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_19_SELECT_INPUT,       0x43FAC890,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_1_SELECT_INPUT,        0x43FAC894,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_20_SELECT_INPUT,       0x43FAC898,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_21_SELECT_INPUT,       0x43FAC89C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_22_SELECT_INPUT,       0x43FAC8A0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_23_SELECT_INPUT,       0x43FAC8A4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_24_SELECT_INPUT,       0x43FAC8A8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_25_SELECT_INPUT,       0x43FAC8AC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_26_SELECT_INPUT,       0x43FAC8B0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_27_SELECT_INPUT,       0x43FAC8B4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_28_SELECT_INPUT,       0x43FAC8B8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_29_SELECT_INPUT,       0x43FAC8BC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_2_SELECT_INPUT,        0x43FAC8C0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_30_SELECT_INPUT,       0x43FAC8C4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_31_SELECT_INPUT,       0x43FAC8C8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_3_SELECT_INPUT,        0x43FAC8CC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_4_SELECT_INPUT,        0x43FAC8D0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_5_SELECT_INPUT,        0x43FAC8D4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_6_SELECT_INPUT,        0x43FAC8D8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_7_SELECT_INPUT,        0x43FAC8DC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_8_SELECT_INPUT,        0x43FAC8E0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO2_IPP_IND_G_IN_9_SELECT_INPUT,        0x43FAC8E4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_0_SELECT_INPUT,        0x43FAC8E8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_10_SELECT_INPUT,       0x43FAC8EC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_11_SELECT_INPUT,       0x43FAC8F0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_12_SELECT_INPUT,       0x43FAC8F4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_13_SELECT_INPUT,       0x43FAC8F8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_14_SELECT_INPUT,       0x43FAC8FC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_15_SELECT_INPUT,       0x43FAC900,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_4_SELECT_INPUT,        0x43FAC904,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_5_SELECT_INPUT,        0x43FAC908,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_6_SELECT_INPUT,        0x43FAC90C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_7_SELECT_INPUT,        0x43FAC910,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_8_SELECT_INPUT,        0x43FAC914,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_9_SELECT_INPUT,        0x43FAC918,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SCL_IN_SELECT_INPUT,             0x43FAC91C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SDA_IN_SELECT_INPUT,             0x43FAC920,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_DISPB_D0_VSYNC_SELECT_INPUT,  0x43FAC924,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_DISPB_D12_VSYNC_SELECT_INPUT, 0x43FAC928,__READ_WRITE ,__iomux_sw_select_input2_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_DISPB_SD_D_SELECT_INPUT,      0x43FAC92C,__READ_WRITE ,__iomux_sw_select_input2_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_0_SELECT_INPUT,    0x43FAC930,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_1_SELECT_INPUT,    0x43FAC934,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_2_SELECT_INPUT,    0x43FAC938,__READ_WRITE ,__iomux_sw_select_input2_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_3_SELECT_INPUT,    0x43FAC93C,__READ_WRITE ,__iomux_sw_select_input2_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_4_SELECT_INPUT,    0x43FAC940,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_5_SELECT_INPUT,    0x43FAC944,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_6_SELECT_INPUT,    0x43FAC948,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENSB_DATA_7_SELECT_INPUT,    0x43FAC94C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_0_SELECT_INPUT,           0x43FAC950,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_1_SELECT_INPUT,           0x43FAC954,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_2_SELECT_INPUT,           0x43FAC958,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_3_SELECT_INPUT,           0x43FAC95C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_4_SELECT_INPUT,           0x43FAC960,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_5_SELECT_INPUT,           0x43FAC964,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_6_SELECT_INPUT,           0x43FAC968,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_7_SELECT_INPUT,           0x43FAC96C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_0_SELECT_INPUT,           0x43FAC970,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_1_SELECT_INPUT,           0x43FAC974,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_2_SELECT_INPUT,           0x43FAC978,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_3_SELECT_INPUT,           0x43FAC97C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_4_SELECT_INPUT,           0x43FAC980,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_5_SELECT_INPUT,           0x43FAC984,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_6_SELECT_INPUT,           0x43FAC988,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_7_SELECT_INPUT,           0x43FAC98C,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_OWIRE_BATTERY_LINE_IN_SELECT_INPUT,       0x43FAC990,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_SPDIF_HCKT_CLK2_SELECT_INPUT,             0x43FAC994,__READ_WRITE ,__iomux_sw_select_input2_bits);
__IO_REG32_BIT(IOMUXC_SPDIF_SPDIF_IN1_SELECT_INPUT,             0x43FAC998,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RTS_B_SELECT_INPUT,        0x43FAC99C,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RXD_MUX_SELECT_INPUT,      0x43FAC9A0,__READ_WRITE ,__iomux_sw_select_input1_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_0_SELECT_INPUT,  0x43FAC9A4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_1_SELECT_INPUT,  0x43FAC9A8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_2_SELECT_INPUT,  0x43FAC9AC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_3_SELECT_INPUT,  0x43FAC9B0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_4_SELECT_INPUT,  0x43FAC9B4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_5_SELECT_INPUT,  0x43FAC9B8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_6_SELECT_INPUT,  0x43FAC9BC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DATA_7_SELECT_INPUT,  0x43FAC9C0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_DIR_SELECT_INPUT,     0x43FAC9C4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_OTG_NXT_SELECT_INPUT,     0x43FAC9C8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_0_SELECT_INPUT,  0x43FAC9CC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_1_SELECT_INPUT,  0x43FAC9D0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_2_SELECT_INPUT,  0x43FAC9D4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_3_SELECT_INPUT,  0x43FAC9D8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_4_SELECT_INPUT,  0x43FAC9DC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_5_SELECT_INPUT,  0x43FAC9E0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_6_SELECT_INPUT,  0x43FAC9E4,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DATA_7_SELECT_INPUT,  0x43FAC9E8,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_DIR_SELECT_INPUT,     0x43FAC9EC,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_NXT_SELECT_INPUT,     0x43FAC9F0,__READ_WRITE ,__iomux_sw_select_input0_bits);
__IO_REG32_BIT(IOMUXC_USB_TOP_IPP_IND_UH2_USB_OC_SELECT_INPUT,  0x43FAC9F4,__READ_WRITE ,__iomux_sw_select_input1_bits);

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
__IO_REG32_BIT(GPIO1_EDGE_SEL,            0x53FCC01C,__READ_WRITE ,__BITS32);

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
__IO_REG32_BIT(GPIO2_EDGE_SEL,            0x53FD001C,__READ_WRITE ,__BITS32);

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
__IO_REG32_BIT(GPIO3_EDGE_SEL,            0x53FA401C,__READ_WRITE ,__BITS32);

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
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT( IMM_STAT,                  0x53FF0000,__READ_WRITE ,__iim_stat_bits);
__IO_REG8_BIT( IMM_STATM,                 0x53FF0004,__READ_WRITE ,__iim_statm_bits);
__IO_REG8_BIT( IMM_ERR,                   0x53FF0008,__READ_WRITE ,__iim_err_bits);
__IO_REG8_BIT( IMM_EMASK,                 0x53FF000C,__READ_WRITE ,__iim_emask_bits);
__IO_REG8_BIT( IMM_FCTL,                  0x53FF0010,__READ_WRITE ,__iim_fctl_bits);
__IO_REG8_BIT( IMM_UA,                    0x53FF0014,__READ_WRITE ,__iim_ua_bits);
__IO_REG8(     IMM_LA,                    0x53FF0018,__READ_WRITE );
__IO_REG8(     IMM_SDAT,                  0x53FF001C,__READ       );
__IO_REG8_BIT( IMM_PREV,                  0x53FF0020,__READ       ,__iim_prev_bits);
__IO_REG8(     IMM_SREV,                  0x53FF0024,__READ       );
__IO_REG8(     IMM_PREG_P,                0x53FF0028,__READ_WRITE );
__IO_REG8_BIT( IMM_SCS0,                  0x53FF002C,__READ_WRITE ,__iim_scs0_bits);
__IO_REG8_BIT( IMM_SCS1,                  0x53FF0030,__READ_WRITE ,__iim_scs_bits);
__IO_REG8_BIT( IMM_SCS2,                  0x53FF0034,__READ_WRITE ,__iim_scs_bits);
__IO_REG8_BIT( IMM_SCS3,                  0x53FF0038,__READ_WRITE ,__iim_scs_bits);

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
__IO_REG32_BIT(EMCC4,                     0x43F08018,__READ_WRITE ,__emcc_bits);
__IO_REG32_BIT(EMCC5,                     0x43F0801C,__READ_WRITE ,__emcc_bits);
__IO_REG32(    EMC0,                      0x43F08020,__READ       );
__IO_REG32(    EMC1,                      0x43F08024,__READ       );
__IO_REG32(    EMC2,                      0x43F08028,__READ       );
__IO_REG32(    EMC3,                      0x43F0802C,__READ       );
__IO_REG32(    EMC4,                      0x43F08030,__READ       );
__IO_REG32(    EMC5,                      0x43F08034,__READ       );

/***************************************************************************
 **
 **  M3IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M3IFCTL,                   0xB8003000,__READ_WRITE ,__m3ifctl_bits);
__IO_REG32_BIT(M3IFWCFG0,                 0xB8003004,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG1,                 0xB8003008,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG2,                 0xB800300C,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG3,                 0xB8003010,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG4,                 0xB8003014,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG5,                 0xB8003018,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG6,                 0xB800301C,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCFG7,                 0xB8003020,__READ_WRITE,__m3ifwcfg_bits);
__IO_REG32_BIT(M3IFWCSR,                  0xB8003024,__READ_WRITE,__m3ifwcsr_bits);
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
 **  ESDRAMC
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL0,                   0xB8001000,__READ_WRITE ,__esdctl_bits);
__IO_REG32_BIT(ESDCFG0,                   0xB8001004,__READ_WRITE ,__esdcfg_bits);
__IO_REG32_BIT(ESDCTL1,                   0xB8001008,__READ_WRITE ,__esdctl_bits);
__IO_REG32_BIT(ESDCFG1,                   0xB800100C,__READ_WRITE ,__esdcfg_bits);
__IO_REG32_BIT(ESDMISC,                   0xB8001010,__READ_WRITE ,__esdmisc_bits);
__IO_REG32_BIT(ESDCDLY1,                  0xB8001020,__READ_WRITE ,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY2,                  0xB8001024,__READ_WRITE ,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY3,                  0xB8001028,__READ_WRITE ,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY4,                  0xB800102C,__READ_WRITE ,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY5,                  0xB8001030,__READ_WRITE ,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLYL,                  0xB8001034,__READ       ,__esdcdlyl_bits);

/***************************************************************************
 **
 **  NFC
 **
 ***************************************************************************/
__IO_REG16_BIT(RAM_BUFFER_ADDRESS,        0xBB001E04,__READ_WRITE,__nfc_rba_bits);
__IO_REG16(    NAND_FLASH_ADD,            0xBB001E06,__READ_WRITE);
__IO_REG16(    NAND_FLASH_CMD,            0xBB001E08,__READ_WRITE);
__IO_REG16_BIT(NFC_CONFIGURATION,         0xBB001E0A,__READ_WRITE,__nfc_iblc_bits);
__IO_REG16_BIT(ECC_STATUS_RESULT1,        0xBB001E0C,__READ      ,__ecc_srr_bits);
__IO_REG16_BIT(ECC_STATUS_RESULT2,        0xBB001E0E,__READ      ,__ecc_srr2_bits);
__IO_REG16_BIT(NFC_SPAS,                  0xBB001E10,__READ_WRITE,__nfc_spas_bits);
__IO_REG16_BIT(NF_WR_PROT,                0xBB001E12,__READ_WRITE,__nf_wr_prot_bits);
__IO_REG16_BIT(NAND_FLASH_WR_PR_ST,       0xBB001E18,__READ_WRITE,__nf_wr_prot_sta_bits);
__IO_REG16_BIT(NAND_FLASH_CONFIG1,        0xBB001E1A,__READ_WRITE,__nand_fc1_bits);
__IO_REG16_BIT(NAND_FLASH_CONFIG2,        0xBB001E1C,__READ_WRITE,__nand_fc2_bits);
__IO_REG16(    UNLOCK_START_BLK_ADD,      0xBB001E20,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD,        0xBB001E22,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD1,     0xBB001E24,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD1,       0xBB001E26,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD2,     0xBB001E28,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD2,       0xBB001E2A,__READ_WRITE);
__IO_REG16(    UNLOCK_START_BLK_ADD3,     0xBB001E2C,__READ_WRITE);
__IO_REG16(    UNLOCK_END_BLK_ADD3,       0xBB001E2E,__READ_WRITE);

/***************************************************************************
 **
 **  One Wire
 **
 ***************************************************************************/
__IO_REG16_BIT(OW_CONTROL,                0x43F9C000,__READ_WRITE ,__ow_control_bits);
__IO_REG16_BIT(OW_TIME_DIVIDER,           0x43F9C002,__READ_WRITE ,__ow_time_divider_bits);
__IO_REG16_BIT(OW_RESET,                  0x43F9C004,__READ_WRITE ,__ow_reset_bits);
__IO_REG16_BIT(OW_COMMAND,                0x43F9C006,__READ_WRITE ,__ow_command_bits);
__IO_REG16_BIT(OW_RX_TX,                  0x43F9C008,__READ_WRITE ,__ow_rx_tx_bits);
__IO_REG16_BIT(OW_INTERRUPT,              0x43F9C00A,__READ       ,__ow_interrupt_bits);
__IO_REG16_BIT(OW_INTERRUPT_EN,           0x43F9C00C,__READ_WRITE ,__ow_interrupt_en_bits);

/***************************************************************************
 **
 **  ATA
 **
 ***************************************************************************/
__IO_REG8(     ATA_TIME_OFF,              0x50020000,__READ_WRITE );
__IO_REG8(     ATA_TIME_ON,               0x50020001,__READ_WRITE );
__IO_REG8(     ATA_TIME_1,                0x50020002,__READ_WRITE );
__IO_REG8(     ATA_TIME_2W,               0x50020003,__READ_WRITE );
__IO_REG8(     ATA_TIME_2R,               0x50020004,__READ_WRITE );
__IO_REG8(     ATA_TIME_AX,               0x50020005,__READ_WRITE );
__IO_REG8(     ATA_TIME_PIO_RDX,          0x50020006,__READ_WRITE );
__IO_REG8(     ATA_TIME_4,                0x50020007,__READ_WRITE );
__IO_REG8(     ATA_TIME_9,                0x50020008,__READ_WRITE );
__IO_REG8(     ATA_TIME_M,                0x50020009,__READ_WRITE );
__IO_REG8(     ATA_TIME_JN,               0x5002000A,__READ_WRITE );
__IO_REG8(     ATA_TIME_D,                0x5002000B,__READ_WRITE );
__IO_REG8(     ATA_TIME_K,                0x5002000C,__READ_WRITE );
__IO_REG8(     ATA_TIME_ACK,              0x5002000D,__READ_WRITE );
__IO_REG8(     ATA_TIME_ENV,              0x5002000E,__READ_WRITE );
__IO_REG8(     ATA_TIME_RPX,              0x5002000F,__READ_WRITE );
__IO_REG8(     ATA_TIME_ZAH,              0x50020010,__READ_WRITE );
__IO_REG8(     ATA_TIME_MLIX,             0x50020011,__READ_WRITE );
__IO_REG8(     ATA_TIME_DVH,              0x50020012,__READ_WRITE );
__IO_REG8(     ATA_TIME_DZFS,             0x50020013,__READ_WRITE );
__IO_REG8(     ATA_TIME_DVS,              0x50020014,__READ_WRITE );
__IO_REG8(     ATA_TIME_CVH,              0x50020015,__READ_WRITE );
__IO_REG8(     ATA_TIME_SS,               0x50020016,__READ_WRITE );
__IO_REG8(     ATA_TIME_CYC,              0x50020017,__READ_WRITE );
__IO_REG32(    ATA_FIFO_DATA_32,          0x50020018,__READ_WRITE );
__IO_REG16(    ATA_FIFO_DATA_16,          0x5002001C,__READ_WRITE );
__IO_REG8(     ATA_FIFO_FILL,             0x50020020,__READ       );
__IO_REG16_BIT(ATA_CONTROL,               0x50020024,__READ_WRITE ,__ata_control_bits);
__IO_REG8_BIT( ATA_INTR_PEND,             0x50020028,__READ       ,__ata_intr_pend_bits);
__IO_REG8_BIT( ATA_INTR_ENA,              0x5002002C,__READ_WRITE ,__ata_intr_ena_bits);
__IO_REG8_BIT( ATA_INTR_CLR,              0x50020030,__WRITE      ,__ata_intr_clr_bits);
__IO_REG8(     ATA_FIFO_ALARM,            0x50020034,__READ_WRITE );
__IO_REG8_BIT( ATA_ADMA_ERR_STATUS,       0x50020038,__READ_WRITE ,__ata_adma_err_status_bits);
__IO_REG32(    ATA_SYS_DMA_BADDR,         0x5002003C,__READ_WRITE );
__IO_REG32(    ATA_ADMA_SYS_ADDR,         0x50020040,__READ_WRITE );
__IO_REG16(    ATA_BLOCK_CNT,             0x50020048,__READ_WRITE );
__IO_REG8_BIT( ATA_BURST_LENGTH,          0x5002004C,__READ_WRITE ,__ata_burst_length_bits);
__IO_REG16(    ATA_SECTOR_SIZE,           0x50020050,__READ_WRITE );
__IO_REG16(    ATA_DRIVE_DATA,            0x500200A0,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_FEATURES,        0x500200A4,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_SECTOR_COUNT,    0x500200A8,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_SECTOR_NUM,      0x500200AC,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_CYL_LOW,         0x500200B0,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_CYL_HIGH,        0x500200B4,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_DEV_HEAD,        0x500200B8,__READ_WRITE );
__IO_REG32(    ATA_DRIVE_COMMAND,         0x500200BC,__WRITE      );
__IO_REG32(    ATA_DRIVE_STATUS,          0x500200C0,__READ       );
__IO_REG32(    ATA_DRIVE_ALT_STATUS,      0x500200C4,__READ       );
__IO_REG32(    ATA_DRIVE_CONTROL,         0x500200C8,__WRITE      );

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
__IO_REG32_BIT(CSPI1_TESTREG,             0x43FA401C,__READ_WRITE,__cspi_testreg_bits);

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
__IO_REG32_BIT(CSPI2_TESTREG,             0x5001001C,__READ_WRITE,__cspi_testreg_bits);

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
 **  MSHC
 **
 ***************************************************************************/
__IO_REG32_BIT(MSHC_CMDR,                 0x50024000,__READ_WRITE ,__mshc_cmdr_bits);
__IO_REG32_BIT(MSHC_DATR,                 0x50024008,__READ_WRITE ,__mshc_datr_bits);
__IO_REG32_BIT(MSHC_STAR,                 0x50024010,__READ_WRITE ,__mshc_star_bits);
__IO_REG32_BIT(MSHC_SYSR,                 0x50024018,__READ_WRITE ,__mshc_sysr_bits);
__IO_REG32(    MSHC_DSAR,                 0x50024020,__READ_WRITE );

/***************************************************************************
 **
 **  eSDHC1
 **
 ***************************************************************************/
__IO_REG32(    ESDHC1_DSADDR,             0x53FB4000,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_BLKATTR,            0x53FB4004,__READ_WRITE,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC1_CMDARG,             0x53FB4008,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_XFERTYP,            0x53FB400C,__READ_WRITE,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC1_CMDRSP0,            0x53FB4010,__READ      );
__IO_REG32(    ESDHC1_CMDRSP1,            0x53FB4014,__READ      );
__IO_REG32(    ESDHC1_CMDRSP2,            0x53FB4018,__READ      );
__IO_REG32(    ESDHC1_CMDRSP3,            0x53FB401C,__READ      );
__IO_REG32(    ESDHC1_DATPORT,            0x53FB4020,__READ_WRITE);
__IO_REG32_BIT(ESDHC1_PRSSTAT,            0x53FB4024,__READ      ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC1_PROCTL,             0x53FB4028,__READ_WRITE,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC1_SYSCTL,             0x53FB402C,__READ_WRITE,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC1_IRQSTAT,            0x53FB4030,__READ      ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC1_IRQSTATEN,          0x53FB4034,__READ_WRITE,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC1_IRQSIGEN,           0x53FB4038,__READ      ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC1_AUTOC12ERR,         0x53FB403C,__READ      ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC1_HOSTCAPBLT,         0x53FB4040,__READ      ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC1_WML,                0x53FB4044,__READ_WRITE,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC1_FEVT,               0x53FB4050,__WRITE     ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC1_ADMAES,             0x53FB4054,__READ      ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC1_ADSADDR,            0x53FB4058,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_VENDOR,             0x53FB40C0,__READ_WRITE,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC1_HOSTVER,            0x53FB40FC,__READ      ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  eSDHC2
 **
 ***************************************************************************/
__IO_REG32(    ESDHC2_DSADDR,             0x53FB8000,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_BLKATTR,            0x53FB8004,__READ_WRITE,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC2_CMDARG,             0x53FB8008,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_XFERTYP,            0x53FB800C,__READ_WRITE,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC2_CMDRSP0,            0x53FB8010,__READ      );
__IO_REG32(    ESDHC2_CMDRSP1,            0x53FB8014,__READ      );
__IO_REG32(    ESDHC2_CMDRSP2,            0x53FB8018,__READ      );
__IO_REG32(    ESDHC2_CMDRSP3,            0x53FB801C,__READ      );
__IO_REG32(    ESDHC2_DATPORT,            0x53FB8020,__READ_WRITE);
__IO_REG32_BIT(ESDHC2_PRSSTAT,            0x53FB8024,__READ      ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC2_PROCTL,             0x53FB8028,__READ_WRITE,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC2_SYSCTL,             0x53FB802C,__READ_WRITE,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC2_IRQSTAT,            0x53FB8030,__READ      ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC2_IRQSTATEN,          0x53FB8034,__READ_WRITE,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC2_IRQSIGEN,           0x53FB8038,__READ      ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC2_AUTOC12ERR,         0x53FB803C,__READ      ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC2_HOSTCAPBLT,         0x53FB8040,__READ      ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC2_WML,                0x53FB8044,__READ_WRITE,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC2_FEVT,               0x53FB8050,__WRITE     ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC2_ADMAES,             0x53FB8054,__READ      ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC2_ADSADDR,            0x53FB8058,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_VENDOR,             0x53FB80C0,__READ_WRITE,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC2_HOSTVER,            0x53FB80FC,__READ      ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  eSDHC2
 **
 ***************************************************************************/
__IO_REG32(    ESDHC3_DSADDR,             0x53FBC000,__READ_WRITE);
__IO_REG32_BIT(ESDHC3_BLKATTR,            0x53FBC004,__READ_WRITE,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC3_CMDARG,             0x53FBC008,__READ_WRITE);
__IO_REG32_BIT(ESDHC3_XFERTYP,            0x53FBC00C,__READ_WRITE,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC3_CMDRSP0,            0x53FBC010,__READ      );
__IO_REG32(    ESDHC3_CMDRSP1,            0x53FBC014,__READ      );
__IO_REG32(    ESDHC3_CMDRSP2,            0x53FBC018,__READ      );
__IO_REG32(    ESDHC3_CMDRSP3,            0x53FBC01C,__READ      );
__IO_REG32(    ESDHC3_DATPORT,            0x53FBC020,__READ_WRITE);
__IO_REG32_BIT(ESDHC3_PRSSTAT,            0x53FBC024,__READ      ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC3_PROCTL,             0x53FBC028,__READ_WRITE,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC3_SYSCTL,             0x53FBC02C,__READ_WRITE,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC3_IRQSTAT,            0x53FBC030,__READ      ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC3_IRQSTATEN,          0x53FBC034,__READ_WRITE,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC3_IRQSIGEN,           0x53FBC038,__READ      ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC3_AUTOC12ERR,         0x53FBC03C,__READ      ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC3_HOSTCAPBLT,         0x53FBC040,__READ      ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC3_WML,                0x53FBC044,__READ_WRITE,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC3_FEVT,               0x53FBC050,__WRITE     ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC3_ADMAES,             0x53FBC054,__READ      ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC3_ADSADDR,            0x53FBC058,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_VENDOR,             0x53FBC0C0,__READ_WRITE,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC3_HOSTVER,            0x53FBC0FC,__READ      ,__esdhc_hostver_bits);

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
__IO_REG32_BIT(UBRC_1,                    0x43F900AC,__READ        ,__ubrc_bits);
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
__IO_REG32_BIT(UBRC_2,                    0x43F940AC,__READ       ,__ubrc_bits);
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
__IO_REG32_BIT(UBRC_3,                    0x5000C0AC,__READ       ,__ubrc_bits);
__IO_REG32_BIT(ONEMS_3,                   0x5000C0B0,__READ_WRITE ,__onems_bits);
__IO_REG32_BIT(UTS_3,                     0x5000C0B4,__READ_WRITE ,__uts_bits);

/***************************************************************************
 **
 **  USBOH
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x53FF4000,__READ      ,__uog_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x53FF4004,__READ      ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x53FF4008,__READ      ,__uog_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x53FF400C,__READ      ,__uog_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x53FF4010,__READ      ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x53FF4014,__READ      ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UOG_GPTIMER0LD,            0x53FF4080,__READ_WRITE,__uog_gptimerxld_bits);
__IO_REG32_BIT(UOG_GPTIMER0CTRL,          0x53FF4084,__READ_WRITE,__uog_gptimerxctrl_bits);
__IO_REG32_BIT(UOG_GPTIMER1LD,            0x53FF4088,__READ_WRITE,__uog_gptimerxld_bits);
__IO_REG32_BIT(UOG_GPTIMER1CTRL,          0x53FF408C,__READ_WRITE,__uog_gptimerxctrl_bits);
__IO_REG32_BIT(UOG_SBUSCFG,               0x53FF4090,__READ_WRITE,__uog_sbuscfg_bits);
__IO_REG8(     UOG_CAPLENGTH,             0x53FF4100,__READ      );
__IO_REG16(    UOG_HCIVERSION,            0x53FF4102,__READ      );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x53FF4104,__READ      ,__uog_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x53FF4108,__READ      ,__uog_hccparams_bits);
__IO_REG32_BIT(UOG_DCIVERSION,            0x53FF4120,__READ      ,__uog_dciversion_bits);
__IO_REG32_BIT(UOG_DCCPARAMS,             0x53FF4124,__READ      ,__uog_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x53FF4140,__READ_WRITE,__uog_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x53FF4144,__READ_WRITE,__uog_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x53FF4148,__READ_WRITE,__uog_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x53FF414C,__READ_WRITE,__uog_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x53FF4154,__READ_WRITE,__uog_periodiclistbase_bits);
#define UOG_DEVICEADDR      UOG_PERIODICLISTBASE
#define UOG_DEVICEADDR_bit  UOG_PERIODICLISTBASE_bit
__IO_REG32_BIT(UOG_ASYNCLISTADDR,         0x53FF4158,__READ_WRITE ,__uog_asynclistaddr_bits);
#define UOG_ENDPOINTLISTADDR      UOG_ASYNCLISTADDR
#define UOG_ENDPOINTLISTADDR_bit  UOG_ASYNCLISTADDR_bit
__IO_REG32_BIT(UOG_BURSTSIZE,             0x53FF4160,__READ_WRITE,__uog_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x53FF4164,__READ_WRITE,__uog_txfilltuning_bits);
__IO_REG32_BIT(UOG_ULPIVIEW,              0x53FF4170,__READ_WRITE,__uog_ulpiview_bits);
__IO_REG32_BIT(UOG_ENDPTNAK,              0x53FF4178,__READ_WRITE,__uog_endptnak_bits);
__IO_REG32_BIT(UOG_ENDPTNAKEN,            0x53FF417C,__READ_WRITE,__uog_endptnaken_bits);
__IO_REG32(    UOG_CFGFLAG,               0x53FF4180,__READ      );
__IO_REG32_BIT(UOG_PORTSC1,               0x53FF4184,__READ_WRITE,__uog_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x53FF41A4,__READ_WRITE,__uog_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x53FF41A8,__READ_WRITE,__uog_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x53FF41AC,__READ_WRITE,__uog_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x53FF41B0,__READ_WRITE,__uog_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x53FF41B4,__READ_WRITE,__uog_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x53FF41B8,__READ      ,__uog_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x53FF41BC,__READ_WRITE,__uog_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x53FF41C0,__READ_WRITE,__uog_endptctrl0_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x53FF41C4,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x53FF41C8,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x53FF41CC,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x53FF41D0,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x53FF41D4,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x53FF41D8,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x53FF41DC,__READ_WRITE,__uog_endptctrl_bits);
__IO_REG32_BIT(UH1_ID,                    0x53FF4400,__READ      ,__uog_id_bits);
__IO_REG32_BIT(UH1_HWGENERAL,             0x53FF4404,__READ      ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH1_HWHOST,                0x53FF4408,__READ      ,__uog_hwhost_bits);
__IO_REG32_BIT(UH1_HWTXBUF,               0x53FF4410,__READ      ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH1_HWRXBUF,               0x53FF4414,__READ      ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH1_GPTIMER0LD,            0x53FF4480,__READ_WRITE,__uog_gptimerxld_bits);
__IO_REG32_BIT(UH1_GPTIMER0CTRL,          0x53FF4484,__READ_WRITE,__uog_gptimerxctrl_bits);
__IO_REG32_BIT(UH1_GPTIMER1LD,            0x53FF4488,__READ_WRITE,__uog_gptimerxld_bits);
__IO_REG32_BIT(UH1_GPTIMER1CTRL,          0x53FF448C,__READ_WRITE,__uog_gptimerxctrl_bits);
__IO_REG32_BIT(UH1_SBUSCFG,               0x53FF4490,__READ_WRITE,__uog_sbuscfg_bits);
__IO_REG16(    UH1_CAPLENGTH,             0x53FF4500,__READ      );
__IO_REG16(    UH1_HCIVERSION,            0x53FF4502,__READ      );
__IO_REG32_BIT(UH1_HCSPARAMS,             0x53FF4504,__READ      ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH1_HCCPARAMS,             0x53FF4508,__READ      ,__uog_hccparams_bits);
__IO_REG32_BIT(UH1_USBCMD,                0x53FF4540,__READ_WRITE,__uog_usbcmd_bits);
__IO_REG32_BIT(UH1_USBSTS,                0x53FF4544,__READ_WRITE,__uog_usbsts_bits);
__IO_REG32_BIT(UH1_USBINTR,               0x53FF4548,__READ_WRITE,__uog_usbintr_bits);
__IO_REG32_BIT(UH1_FRINDEX,               0x53FF454C,__READ_WRITE,__uog_frindex_bits);
__IO_REG32_BIT(UH1_PERIODICLISTBASE,      0x53FF4554,__READ_WRITE,__uog_periodiclistbase_bits);
#define UH1_DEVICEADDR      UH1_PERIODICLISTBASE
#define UH1_DEVICEADDR_bit  UH1_PERIODICLISTBASE_bit
__IO_REG32_BIT(UH1_ASYNCLISTADDR,         0x53FF4558,__READ_WRITE,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH1_BURSTSIZE,             0x53FF4560,__READ_WRITE,__uog_burstsize_bits);
__IO_REG32_BIT(UH1_TXFILLTUNING,          0x53FF4564,__READ_WRITE,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH1_ULPIVIEW,              0x53FF4570,__READ_WRITE,__uog_ulpiview_bits);
__IO_REG32_BIT(UH1_PORTSC1,               0x53FF4584,__READ_WRITE,__uog_portsc_bits);
__IO_REG32_BIT(UH1_USBMODE,               0x53FF45A8,__READ_WRITE,__uog_usbmode_bits);
__IO_REG32_BIT(USB_CTRL,                  0x53FF4600,__READ_WRITE,__usb_ctrl_bits);
__IO_REG32_BIT(USB_OTG_MIRROR,            0x53FF4604,__READ_WRITE,__usb_otg_mirror_bits);
__IO_REG32_BIT(USB_PHY_CTRL_FUNC,         0x53FF4608,__READ_WRITE,__usb_phy_ctrl_func_bits);
__IO_REG32_BIT(USB_PHY_CTRL_TEST,         0x53FF460C,__READ_WRITE,__usb_phy_ctrl_test_bits);

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT1CR,                   0x53F94000,__READ_WRITE,__epitcr_bits);
__IO_REG32_BIT(EPIT1SR,                   0x53F94004,__READ_WRITE,__epitsr_bits);
__IO_REG32(    EPIT1LR,                   0x53F94008,__READ_WRITE);
__IO_REG32(    EPIT1CMPR,                 0x53F9400C,__READ_WRITE);
__IO_REG32(    EPIT1CNR,                  0x53F94010,__READ      );

/***************************************************************************
 **
 **  EPIT2
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT2CR,                   0x53F98000,__READ_WRITE,__epitcr_bits);
__IO_REG32_BIT(EPIT2SR,                   0x53F98004,__READ_WRITE,__epitsr_bits);
__IO_REG32(    EPIT2LR,                   0x53F98008,__READ_WRITE);
__IO_REG32(    EPIT2CMPR,                 0x53F9800C,__READ_WRITE);
__IO_REG32(    EPIT2CNR,                  0x53F98010,__READ      );

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
__IO_REG32_BIT(RTCCTL,                    0x53FD8010,__READ_WRITE ,__rtcctl_bits);
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
__IO_REG16_BIT(WICR,                      0x53FDC006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WMCR,                      0x53FDC008,__READ_WRITE ,__wmcr_bits);

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
__IO_REG32_BIT(SDMA_LOCK,                 0x53FD403C,__READ_WRITE ,__sdma_lock_bits);
__IO_REG32_BIT(SDMA_ONCE_ENB,             0x53FD4040,__READ_WRITE ,__sdma_once_enb_bits);
__IO_REG32(    SDMA_ONCE_DATA,            0x53FD4044,__READ_WRITE );
__IO_REG32_BIT(SDMA_ONCE_INSTR,           0x53FD4048,__READ_WRITE ,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMA_ONCE_STAT,            0x53FD404C,__READ       ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMA_ONCE_CMD,             0x53FD4050,__READ_WRITE ,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMA_ILLINSTADDR,          0x53FD4058,__READ_WRITE ,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMA_CHN0ADDR,             0x53FD405C,__READ_WRITE ,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR,           0x53FD4060,__READ       ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR2,          0x53FD4064,__READ       ,__sdma_evt_mirror2_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF1,          0x53FD4070,__READ_WRITE ,__sdma_xtrig_conf1_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF2,          0x53FD4074,__READ_WRITE ,__sdma_xtrig_conf2_bits);
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
__IO_REG32_BIT(SDMA_CHNENBL0,             0x53FD4200,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL1,             0x53FD4204,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL2,             0x53FD4208,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL3,             0x53FD420C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL4,             0x53FD4210,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL5,             0x53FD4214,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL6,             0x53FD4218,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL7,             0x53FD421C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL8,             0x53FD4220,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL9,             0x53FD4224,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL10,            0x53FD4228,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL11,            0x53FD422C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL12,            0x53FD4230,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL13,            0x53FD4234,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL14,            0x53FD4238,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL15,            0x53FD423C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL16,            0x53FD4240,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL17,            0x53FD4244,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL18,            0x53FD4248,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL19,            0x53FD424C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL20,            0x53FD4250,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL21,            0x53FD4254,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL22,            0x53FD4258,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL23,            0x53FD425C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL24,            0x53FD4260,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL25,            0x53FD4264,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL26,            0x53FD4268,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL27,            0x53FD426C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL28,            0x53FD4270,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL29,            0x53FD4274,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL30,            0x53FD4278,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL31,            0x53FD427C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL32,            0x53FD4280,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL33,            0x53FD4284,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL34,            0x53FD4288,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL35,            0x53FD428C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL36,            0x53FD4290,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL37,            0x53FD4294,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL38,            0x53FD4298,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL39,            0x53FD429C,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL40,            0x53FD42A0,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL41,            0x53FD42A4,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL42,            0x53FD42A8,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL43,            0x53FD42AC,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL44,            0x53FD42B0,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL45,            0x53FD42B4,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL46,            0x53FD42B8,__READ_WRITE ,__sdma_chnenbl_bits);
__IO_REG32_BIT(SDMA_CHNENBL47,            0x53FD42BC,__READ_WRITE ,__sdma_chnenbl_bits);

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
__IO_REG32_BIT(IPU_GP_REG,                0x53FC0200,__READ_WRITE ,__ipu_gp_reg_bits);

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
 **  SDC
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
__IO_REG32(    SSI1_STX0,                 0x43FA0000,__READ_WRITE);
__IO_REG32(    SSI1_STX1,                 0x43FA0004,__READ_WRITE);
__IO_REG32(    SSI1_SRX0,                 0x43FA0008,__READ      );
__IO_REG32(    SSI1_SRX1,                 0x43FA000C,__READ      );
__IO_REG32_BIT(SSI1_SCR,                  0x43FA0010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI1_SISR,                 0x43FA0014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI1_SIER,                 0x43FA0018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI1_STCR,                 0x43FA001C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI1_SRCR,                 0x43FA0020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI1_STCCR,                0x43FA0024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SRCCR,                0x43FA0028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SFCSR,                0x43FA002C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI1_STR,                  0x43FA0030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI1_SOR,                  0x43FA0034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI1_SACNT,                0x43FA0038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI1_SACADD,               0x43FA003C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI1_SACDAT,               0x43FA0040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI1_SATAG,                0x43FA0044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI1_STMSK,                0x43FA0048,__READ_WRITE);
__IO_REG32(    SSI1_SRMSK,                0x43FA004C,__READ_WRITE);
__IO_REG32_BIT(SSI1_SACCST,               0x43FA0050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI1_SACCEN,               0x43FA0054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI1_SACCDIS,              0x43FA0058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    SSI2_STX0,                 0x50014000,__READ_WRITE);
__IO_REG32(    SSI2_STX1,                 0x50014004,__READ_WRITE);
__IO_REG32(    SSI2_SRX0,                 0x50014008,__READ      );
__IO_REG32(    SSI2_SRX1,                 0x5001400C,__READ      );
__IO_REG32_BIT(SSI2_SCR,                  0x50014010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI2_SISR,                 0x50014014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI2_SIER,                 0x50014018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI2_STCR,                 0x5001401C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI2_SRCR,                 0x50014020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI2_STCCR,                0x50014024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SRCCR,                0x50014028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SFCSR,                0x5001402C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI2_STR,                  0x50014030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI2_SOR,                  0x50014034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI2_SACNT,                0x50014038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI2_SACADD,               0x5001403C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI2_SACDAT,               0x50014040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI2_SATAG,                0x50014044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI2_STMSK,                0x50014048,__READ_WRITE);
__IO_REG32(    SSI2_SRMSK,                0x5001404C,__READ_WRITE);
__IO_REG32_BIT(SSI2_SACCST,               0x50014050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI2_SACCEN,               0x50014054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI2_SACCDIS,              0x50014058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  ARM1136 CLKCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(GP_CTRL,                   0x43F0C000,__READ_WRITE ,__gp_ctrl_bits);
__IO_REG32_BIT(GP_SER,                    0x43F0C004,__WRITE      ,__gp_ser_bits);
__IO_REG32_BIT(GP_CER,                    0x43F0C008,__WRITE      ,__gp_cer_bits);
__IO_REG32_BIT(GP_STAT,                   0x43F0C00C,__READ_WRITE ,__gp_stat_bits);
__IO_REG32_BIT(L2_MEM_VAL,                0x43F0C010,__READ_WRITE ,__l2_mem_val_bits);
__IO_REG32_BIT(DBG_CTRL1,                 0x43F0C014,__READ_WRITE ,__dbg_ctrl1_bits);
__IO_REG32_BIT(PLAT_ID,                   0x43F0C020,__READ       ,__plat_id_bits);

/***************************************************************************
 **
 **  SPDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(SPDIF_SCR,                 0x50028000,__READ_WRITE ,__spdif_scr_bits);
__IO_REG32_BIT(SPDIF_SRCD,                0x50028004,__READ_WRITE ,__spdif_srcd_bits);
__IO_REG32_BIT(SPDIF_SRPC,                0x50028008,__READ_WRITE ,__spdif_srpc_bits);
__IO_REG32_BIT(SPDIF_SIE,                 0x5002800C,__READ_WRITE ,__spdif_sie_bits);
__IO_REG32_BIT(SPDIF_SIS,                 0x50028010,__READ_WRITE ,__spdif_sis_bits);
#define SPDIF_SIC     SPDIF_SIS
#define SPDIF_SIC_bit SPDIF_SIS_bit.__SIC
__IO_REG32_BIT(SPDIF_SRL,                 0x50028014,__READ       ,__spdif_srl_bits);
__IO_REG32_BIT(SPDIF_SRR,                 0x50028018,__READ       ,__spdif_srr_bits);
__IO_REG32_BIT(SPDIF_SRCSH,               0x5002801C,__READ       ,__spdif_srcsh_bits);
__IO_REG32_BIT(SPDIF_SRCSL,               0x50028020,__READ       ,__spdif_srcsl_bits);
__IO_REG32_BIT(SPDIF_SQU,                 0x50028024,__READ       ,__spdif_squ_bits);
__IO_REG32_BIT(SPDIF_SRQ,                 0x50028028,__READ       ,__spdif_srq_bits);
__IO_REG32_BIT(SPDIF_STL,                 0x5002802C,__WRITE      ,__spdif_stl_bits);
__IO_REG32_BIT(SPDIF_STR,                 0x50028030,__WRITE      ,__spdif_str_bits);
__IO_REG32_BIT(SPDIF_STCSCH,              0x50028034,__READ_WRITE ,__spdif_stcsch_bits);
__IO_REG32_BIT(SPDIF_STCSCL,              0x50028038,__READ_WRITE ,__spdif_stcscl_bits);
__IO_REG32_BIT(SPDIF_SRFM,                0x50028044,__READ       ,__spdif_srfm_bits);
__IO_REG32_BIT(SPDIF_STC,                 0x50028050,__READ_WRITE ,__spdif_stc_bits);
 
/***************************************************************************
 **
 **  ASRC
 **
 ***************************************************************************/
__IO_REG32_BIT(ASRC_ASRCTR,               0x5002C000,__READ_WRITE ,__asrc_asrctr_bits);
__IO_REG32_BIT(ASRC_ASRIER,               0x5002C004,__READ_WRITE ,__asrc_asrier_bits);
__IO_REG32_BIT(ASRC_ASRCNCR,              0x5002C00C,__READ_WRITE ,__asrc_asrcncr_bits);
__IO_REG32_BIT(ASRC_ASRCFG,               0x5002C010,__READ_WRITE ,__asrc_asrcfg_bits);
__IO_REG32_BIT(ASRC_ASRCSR,               0x5002C014,__READ_WRITE ,__asrc_asrcsr_bits);
__IO_REG32_BIT(ASRC_ASRCDR1,              0x5002C018,__READ_WRITE ,__asrc_asrcdr1_bits);
__IO_REG32_BIT(ASRC_ASRCDR2,              0x5002C01C,__READ_WRITE ,__asrc_asrcdr2_bits);
__IO_REG32_BIT(ASRC_ASRSTR,               0x5002C020,__READ       ,__asrc_asrstr_bits);
__IO_REG32(    ASRC_ASRPM1,               0x5002C040,__READ_WRITE );
__IO_REG32(    ASRC_ASRPM2,               0x5002C044,__READ_WRITE );
__IO_REG32(    ASRC_ASRPM3,               0x5002C048,__READ_WRITE );
__IO_REG32(    ASRC_ASRPM4,               0x5002C04C,__READ_WRITE );
__IO_REG32(    ASRC_ASRPM5,               0x5002C050,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASRTFR1,              0x5002C054,__READ_WRITE ,__asrc_asrtfr1_bits);
__IO_REG32_BIT(ASRC_ASRCCR,               0x5002C05C,__READ_WRITE ,__asrc_asrccr_bits);
__IO_REG32_BIT(ASRC_ASRIDRHA,             0x5002C080,__READ_WRITE ,__asrc_asridrha_bits);
__IO_REG32(    ASRC_ASRIDRLA,             0x5002C084,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASRIDRHB,             0x5002C088,__READ_WRITE ,__asrc_asridrhb_bits);
__IO_REG32(    ASRC_ASRIDRLB,             0x5002C08C,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASRIDRHC,             0x5002C090,__READ_WRITE ,__asrc_asridrhc_bits);
__IO_REG32(    ASRC_ASRIDRLC,             0x5002C094,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASR76K,               0x5002C098,__READ_WRITE ,__asrc_asr76k_bits);
__IO_REG32_BIT(ASRC_ASR56K,               0x5002C09C,__READ_WRITE ,__asrc_asr56k_bits);

/***************************************************************************
 **
 **  ESAI
 **
 ***************************************************************************/
__IO_REG32(    ESAI_ETDR,                 0x50034000,__WRITE     );
__IO_REG32(    ESAI_ERDR,                 0x50034004,__READ      );
__IO_REG32_BIT(ESAI_ECR,                  0x50034008,__READ_WRITE,__esai_ecr_bits);
__IO_REG32_BIT(ESAI_ESR,                  0x5003400C,__READ      ,__esai_esr_bits);
__IO_REG32_BIT(ESAI_TFCR,                 0x50034010,__READ_WRITE,__esai_tfcr_bits);
__IO_REG32_BIT(ESAI_TFSR,                 0x50034014,__READ      ,__esai_tfsr_bits);
__IO_REG32_BIT(ESAI_RFCR,                 0x50034018,__READ_WRITE,__esai_rfcr_bits);
__IO_REG32_BIT(ESAI_RFSR,                 0x5003401C,__READ      ,__esai_rfsr_bits);
__IO_REG32_BIT(ESAI_TX0,                  0x50034080,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX1,                  0x50034084,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX2,                  0x50034088,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX3,                  0x5003408C,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX4,                  0x50034090,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TX5,                  0x50034094,__WRITE     ,__esai_tx_bits);
__IO_REG32_BIT(ESAI_TSR,                  0x50034098,__WRITE     ,__esai_tsr_bits);
__IO_REG32_BIT(ESAI_RX0,                  0x500340A0,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX1,                  0x500340A4,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX2,                  0x500340A8,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_RX3,                  0x500340AC,__READ      ,__esai_rx_bits);
__IO_REG32_BIT(ESAI_SAISR,                0x500340CC,__READ      ,__esai_saisr_bits);
__IO_REG32_BIT(ESAI_SAICR,                0x500340D0,__READ_WRITE,__esai_saicr_bits);
__IO_REG32_BIT(ESAI_TCR,                  0x500340D4,__READ_WRITE,__esai_tcr_bits);
__IO_REG32_BIT(ESAI_TCCR,                 0x500340D8,__READ_WRITE,__esai_tccr_bits);
__IO_REG32_BIT(ESAI_RCR,                  0x500340DC,__READ_WRITE,__esai_rcr_bits);
__IO_REG32_BIT(ESAI_RCCR,                 0x500340E0,__READ_WRITE,__esai_rccr_bits);
__IO_REG32_BIT(ESAI_TSMA,                 0x500340E4,__READ_WRITE,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_TSMB,                 0x500340E8,__READ_WRITE,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_RSMA,                 0x500340EC,__READ_WRITE,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_RSMB,                 0x500340F0,__READ_WRITE,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_PRRC,                 0x500340F8,__READ_WRITE,__esai_prrc_bits);
__IO_REG32_BIT(ESAI_PCRC,                 0x500340FC,__READ_WRITE,__esai_pcrc_bits);

/***************************************************************************
 **
 **  FEC
 **
 ***************************************************************************/
__IO_REG32_BIT(FEC_EIR,                   0x50038004,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_EIMR,                  0x50038008,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_RDAR,                  0x50038010,__READ_WRITE,__fec_rdar_bits);
__IO_REG32_BIT(FEC_TDAR,                  0x50038014,__READ_WRITE,__fec_tdar_bits);
__IO_REG32_BIT(FEC_ECR,                   0x50038024,__READ_WRITE,__fec_ecr_bits);
__IO_REG32_BIT(FEC_MMFR,                  0x50038040,__READ_WRITE,__fec_mmfr_bits);
__IO_REG32_BIT(FEC_MSCR,                  0x50038044,__READ_WRITE,__fec_mscr_bits);
__IO_REG32_BIT(FEC_MIBC,                  0x50038064,__READ_WRITE,__fec_mibc_bits);
__IO_REG32_BIT(FEC_RCR,                   0x50038084,__READ_WRITE,__fec_rcr_bits);
__IO_REG32_BIT(FEC_TCR,                   0x500380C4,__READ_WRITE,__fec_tcr_bits);
__IO_REG32(    FEC_PALR,                  0x500380E4,__READ_WRITE);
__IO_REG32_BIT(FEC_PAUR,                  0x500380E8,__READ_WRITE,__fec_paur_bits);
__IO_REG32_BIT(FEC_OPD,                   0x500380EC,__READ_WRITE,__fec_opd_bits);
__IO_REG32(    FEC_IAUR,                  0x50038118,__READ_WRITE);
__IO_REG32(    FEC_IALR,                  0x5003811C,__READ_WRITE);
__IO_REG32(    FEC_GAUR,                  0x50038120,__READ_WRITE);
__IO_REG32(    FEC_GALR,                  0x50038124,__READ_WRITE);
__IO_REG32_BIT(FEC_TFWR,                  0x50038144,__READ_WRITE,__fec_tfwr_bits);
__IO_REG32_BIT(FEC_FRBR,                  0x5003814C,__READ      ,__fec_frbr_bits);
__IO_REG32_BIT(FEC_FRSR,                  0x50038150,__READ_WRITE,__fec_frsr_bits);
__IO_REG32(    FEC_ERDSR,                 0x50038180,__READ_WRITE);
__IO_REG32(    FEC_ETDSR,                 0x50038184,__READ_WRITE);
__IO_REG32_BIT(FEC_EMRBR,                 0x50038188,__READ_WRITE,__fec_emrbr_bits);
__IO_REG32(    FEC_RMON_T_DROP,           0x50038200,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_PACKETS,        0x50038204,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_BC_PKT,         0x50038208,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_MC_PKT,         0x5003820C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_CRC_ALIGN,      0x50038210,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_UNDERSIZE,      0x50038214,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OVERSIZE,       0x50038218,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_FRAG,           0x5003821C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_JAB,            0x50038220,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_COL,            0x50038224,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P64,            0x50038228,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P65TO127,       0x5003822C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P128TO255,      0x50038230,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P256TO511,      0x50038234,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P512TO1023,     0x50038238,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P1024TO2047,    0x5003823C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P_GTE2048,      0x50038240,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OCTETS,         0x50038244,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DROP,           0x50038248,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FRAME_OK,       0x5003824C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_1COL,           0x50038250,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MCOL,           0x50038254,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DEF,            0x50038258,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_LCOL,           0x5003825C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_EXCOL,          0x50038260,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MACERR,         0x50038264,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_CSERR,          0x50038268,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_SQE,            0x5003826C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FDXFC,          0x50038270,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_OCTETS_OK,      0x50038274,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_PACKETS,        0x50038284,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_BC_PKT,         0x50038288,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_MC_PKT,         0x5003828C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_CRC_ALIGN,      0x50038290,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_UNDERSIZE,      0x50038294,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OVERSIZE,       0x50038298,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_FRAG,           0x5003829C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_JAB,            0x500382A0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_RESVD_0,        0x500382A4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P64,            0x500382A8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P65TO127,       0x500382AC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P128TO255,      0x500382B0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P256TO511,      0x500382B4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P512TO1023,     0x500382B8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P1024TO2047,    0x500382BC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P_GTE2048,      0x500382C0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OCTETS,         0x500382C4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_DROP,           0x500382C8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FRAME_OK,       0x500382CC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_CRC,            0x500382D0,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_ALIGN,          0x500382D4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_MACERR,         0x500382D8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FDXFC,          0x500382DC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_OCTETS_OK,      0x500382E0,__READ_WRITE);

/***************************************************************************
 **
 **  FlexCAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1_MCR,                  0x53FE4000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(CAN1_CTRL,                 0x53FE4004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(CAN1_TIMER,                0x53FE4008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(CAN1_RXGMASK,              0x53FE4010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN1_RX14MASK,             0x53FE4014,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN1_RX15MASK,             0x53FE4018,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN1_ECR,                  0x53FE401C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(CAN1_ESR,                  0x53FE4020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(CAN1_IMASK2,               0x53FE4024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(CAN1_IMASK1,               0x53FE4028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(CAN1_IFLAG2,               0x53FE402C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(CAN1_IFLAG1,               0x53FE4030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32(    CAN1_MB0_15_BASE_ADDR,     0x53FE4080,__READ_WRITE);
__IO_REG32(    CAN1_MB16_31_BASE_ADDR,    0x53FE4180,__READ_WRITE);
__IO_REG32(    CAN1_MB32_63_BASE_ADDR,    0x53FE4280,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR0_15_BASE_ADDR,  0x53FE4880,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR16_31_BASE_ADDR, 0x53FE48C0,__READ_WRITE);
__IO_REG32(    CAN1_RXIMR32_63_BASE_ADDR, 0x53FE4900,__READ_WRITE);

/***************************************************************************
 **
 **  FlexCAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN2_MCR,                  0x53FE8000,__READ_WRITE,__can_mcr_bits);
__IO_REG32_BIT(CAN2_CTRL,                 0x53FE8004,__READ_WRITE,__can_ctrl_bits);
__IO_REG32_BIT(CAN2_TIMER,                0x53FE8008,__READ_WRITE,__can_timer_bits);
__IO_REG32_BIT(CAN2_RXGMASK,              0x53FE8010,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN2_RX14MASK,             0x53FE8014,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN2_RX15MASK,             0x53FE8018,__READ_WRITE,__can_rxgmask_bits);
__IO_REG32_BIT(CAN2_ECR,                  0x53FE801C,__READ_WRITE,__can_ecr_bits);
__IO_REG32_BIT(CAN2_ESR,                  0x53FE8020,__READ_WRITE,__can_esr_bits);
__IO_REG32_BIT(CAN2_IMASK2,               0x53FE8024,__READ_WRITE,__can_imask2_bits);
__IO_REG32_BIT(CAN2_IMASK1,               0x53FE8028,__READ_WRITE,__can_imask1_bits);
__IO_REG32_BIT(CAN2_IFLAG2,               0x53FE802C,__READ_WRITE,__can_iflag2_bits);
__IO_REG32_BIT(CAN2_IFLAG1,               0x53FE8030,__READ_WRITE,__can_iflag1_bits);
__IO_REG32(    CAN2_MB0_15_BASE_ADDR,     0x53FE8080,__READ_WRITE);
__IO_REG32(    CAN2_MB16_31_BASE_ADDR,    0x53FE8180,__READ_WRITE);
__IO_REG32(    CAN2_MB32_63_BASE_ADDR,    0x53FE8280,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR0_15_BASE_ADDR,  0x53FE8880,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR16_31_BASE_ADDR, 0x53FE88C0,__READ_WRITE);
__IO_REG32(    CAN2_RXIMR32_63_BASE_ADDR, 0x53FE8900,__READ_WRITE);

/***************************************************************************
 **
 **  MLB
 **
 ***************************************************************************/
__IO_REG32_BIT(MLB_DCCR,              0x53FF8000,__READ_WRITE ,__mlb_dccr_bits);
__IO_REG32_BIT(MLB_SSCR,              0x53FF8004,__READ_WRITE ,__mlb_sscr_bits);
__IO_REG32(    MLB_SDCR,              0x53FF8008,__READ       );
__IO_REG32_BIT(MLB_SMCR,              0x53FF800C,__READ_WRITE ,__mlb_smcr_bits);
__IO_REG32_BIT(MLB_VCCR,              0x53FF801C,__READ       ,__mlb_vccr_bits);
__IO_REG32_BIT(MLB_SBCR,              0x53FF8020,__READ_WRITE ,__mlb_sbcr_bits);
__IO_REG32_BIT(MLB_ABCR,              0x53FF8024,__READ_WRITE ,__mlb_abcr_bits);
__IO_REG32_BIT(MLB_CBCR,              0x53FF8028,__READ_WRITE ,__mlb_cbcr_bits);
__IO_REG32_BIT(MLB_IBCR,              0x53FF802C,__READ_WRITE ,__mlb_ibcr_bits);
__IO_REG32_BIT(MLB_CICR,              0x53FF8030,__READ_WRITE ,__mlb_cicr_bits);
__IO_REG32_BIT(MLB_CECR0,             0x53FF8040,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR0,             0x53FF8044,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR0,            0x53FF8048,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR0,            0x53FF804C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR0,            0x53FF8280,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR1,             0x53FF8050,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR1,             0x53FF8054,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR1,            0x53FF8058,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR1,            0x53FF805C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR1,            0x53FF8284,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR2,             0x53FF8060,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR2,             0x53FF8064,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR2,            0x53FF8068,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR2,            0x53FF806C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR2,            0x53FF8288,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR3,             0x53FF8070,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR3,             0x53FF8074,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR3,            0x53FF8078,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR3,            0x53FF807C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR3,            0x53FF828C,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR4,             0x53FF8080,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR4,             0x53FF8084,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR4,            0x53FF8088,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR4,            0x53FF808C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR4,            0x53FF8290,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR5,             0x53FF8090,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR5,             0x53FF8094,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR5,            0x53FF8098,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR5,            0x53FF809C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR5,            0x53FF8294,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR6,             0x53FF80A0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR6,             0x53FF80A4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR6,            0x53FF80A8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR6,            0x53FF80AC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR6,            0x53FF8298,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR7,             0x53FF80B0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR7,             0x53FF80B4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR7,            0x53FF80B8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR7,            0x53FF80BC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR7,            0x53FF829C,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR8,             0x53FF80C0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR8,             0x53FF80C4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR8,            0x53FF80C8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR8,            0x53FF80CC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR8,            0x53FF82A0,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR9,             0x53FF80D0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR9,             0x53FF80D4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR9,            0x53FF80D8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR9,            0x53FF80DC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR9,            0x53FF82A4,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR10,            0x53FF80E0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR10,            0x53FF80E4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR10,           0x53FF80E8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR10,           0x53FF80EC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR10,           0x53FF82A8,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR11,            0x53FF80F0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR11,            0x53FF80F4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR11,           0x53FF80F8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR11,           0x53FF80FC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR11,           0x53FF82AC,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR12,            0x53FF8100,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR12,            0x53FF8104,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR12,           0x53FF8108,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR12,           0x53FF810C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR12,           0x53FF82B0,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR13,            0x53FF8110,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR13,            0x53FF8114,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR13,           0x53FF8118,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR13,           0x53FF811C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR13,           0x53FF82B4,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR14,            0x53FF8120,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR14,            0x53FF8124,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR14,           0x53FF8128,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR14,           0x53FF812C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR14,           0x53FF82B8,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR15,            0x53FF8130,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR15,            0x53FF8134,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR15,           0x53FF8138,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR15,           0x53FF813C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR15,           0x53FF82BC,__READ_WRITE ,__mlb_lcbcr_bits);

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
 **   MCIMX35 interrupt sources
 **
 ***************************************************************************/
#define INT_OWIRE              2              /* One Wire*/
#define INT_I2C3               3              /* Inter-Integrated Circuit 3*/
#define INT_I2C2               4              /* Inter-Integrated Circuit 2*/
#define INT_RTIC               6              /* HASH error has occurred, or the RTIC has completed hashing*/
#define INT_ESDHC1             7              /* Enhanced Secure Digital Host Controller 1 */
#define INT_ESDHC2             8              /* Enhanced Secure Digital Host Controller 2 */
#define INT_ESDHC3             9              /* Enhanced Secure Digital Host Controller 3 */
#define INT_I2C1               10             /* Inter-Integrated Circuit 1*/
#define INT_SSI1               11             /* Synchronous Serial Interface 1*/
#define INT_SSI2               12             /* Synchronous Serial Interface 2*/
#define INT_CSPI2              13             /* Configurable Serial Peripheral Interface 2*/
#define INT_CSPI1              14             /* Configurable Serial Peripheral Interface 1*/
#define INT_ATA                15             /* Hard Drive (ATA) Controller*/
#define INT_GPU2D              16             /* Graphic accelerator*/
#define INT_ASRC               17             /* Asynchronous Sample Rate Converter */
#define INT_UART3              18             /* UART3*/
#define INT_IIM                19             /* IC Identification*/
#define INT_RNGA               22             /* Random Number Generator Accelerator*/
#define INT_EVTMON             23             /* OR of evtmon_interrupt,pmu_irq*/
#define INT_KPP                24             /* Keyboard Pad Port*/
#define INT_RTC                25             /* Real Time Clock*/
#define INT_PWM                26             /* Pulse Width Modulator*/
#define INT_EPIT2              27             /* Enhanced Periodic Timer 2*/
#define INT_EPIT1              28             /* Enhanced Periodic Timer 1*/
#define INT_GPT                29             /* General Purpose Timer*/
#define INT_POWER_FAIL         30             /* Power fail interrupt from external PAD*/
#define INT_CCM                31             /* Clock controller*/
#define INT_UART2              32             /* UART2*/
#define INT_NANDFC             33             /* NAND Flash Controller*/
#define INT_SDMA               34             /* Smart Direct Memory Access*/
#define INT_USB_HS             35             /* USB Host*/
#define INT_USB_OTG            37             /* USB OTG*/
#define INT_MSHC               39             /* Memory Stick Host Controller*/
#define INT_ESAI               40             /* Enhanced Serial Audio Interface*/
#define INT_IPU_ERR            41             /* Image Processing Unit error*/
#define INT_IPU                42             /* IPU general interrupt*/
#define INT_CAN1               43             /* CAN1*/
#define INT_CAN2               44             /* CAN2*/
#define INT_UART1              45             /* UART1*/
#define INT_MLB                46             /* MLB*/
#define INT_SPDIF              47             /* SPDIF*/
#define INT_ECT                48             /* AND of oct_irq_b[1:0]*/
#define INT_SCM                49             /* SCM interrupt*/
#define INT_SMN                50             /* SMN interrupt*/
#define INT_GPIO2              51             /* General Purpose I/O 2*/
#define INT_GPIO1              52             /* General Purpose I/O 1*/
#define INT_WDOG               55             /* Watch Dog Timer*/
#define INT_GPIO3              56             /* General Purpose I/O 3*/
#define INT_FEC                57             /* Fast Ethernet Controller*/
#define INT_EXT_PM             58             /* External (power management)*/
#define INT_EXT_TEMP           59             /* External (Temper)*/
#define INT_EXT_SENSOR1        60             /* External (sensor)*/
#define INT_EXT_SENSOR2        61             /* External (sensor)*/
#define INT_EXT_WDT            62             /* External (WDOG)*/
#define INT_EXT_TV             63             /* External (TV)*/

#endif    /* __MCIMX35_H */
