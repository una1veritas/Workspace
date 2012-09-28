/***************************************************************************
 **
 **    This file defines the Special Function Register for
 **    Freescale MCIMX535
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 51329 $
 **
 ***************************************************************************/

#ifndef __MCIMX535_H
#define __MCIMX535_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX535 SPECIAL FUNCTION REGISTERS
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

/* Master Priority Register (AHBMAX.MPR0-3) */
typedef struct {
__REG32 MSTR_0    : 3;
__REG32           : 1;
__REG32 MSTR_1    : 3;
__REG32           : 1;
__REG32 MSTR_2    : 3;
__REG32           : 1;
__REG32 MSTR_3    : 3;
__REG32           : 1;
__REG32 MSTR_4    : 3;
__REG32           : 1;
__REG32 MSTR_5    : 3;
__REG32           : 1;
__REG32 MSTR_6    : 3;
__REG32           : 5;
} __ahbmax_mpr_bits;

/* Slave General Purpose Control Register (AHBMAX.SGPCR0-3) */
typedef struct {
__REG32 PARK      : 3;
__REG32           : 1;
__REG32 PCTL      : 2;
__REG32           : 2;
__REG32 ARB       : 2;
__REG32           :20;
__REG32 HLP       : 1;
__REG32 RO        : 1;
} __ahbmax_sgpcr_bits;

/* Master General Purpose Control Register (AHBMAX.MGPCR0-6) */
typedef struct {
__REG32 AULB      : 3;
__REG32           :29;
} __ahbmax_mgpcr_bits;

/* ASRC Control Register (ASRC.ASRCTR) */
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

/* Interrupt Enable Register(ASRC.ASRIER) */
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

/* Channel Number Configuration Register (ASRC.ASRCNCR) */
typedef struct {
__REG32 ANCA      : 4;
__REG32 ANCB      : 4;
__REG32 ANCC      : 4;
__REG32           :20;
} __asrc_asrcncr_bits;

/* Filter Configuration Status Register (ASRC.ASRCFG) */
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

/* ASRC Clock Source Register (ASRC.ASRCSR) */
typedef struct {
__REG32 AICSA     : 4;
__REG32 AICSB     : 4;
__REG32 AICSC     : 4;
__REG32 AOCSA     : 4;
__REG32 AOCSB     : 4;
__REG32 AOCSC     : 4;
__REG32           : 8;
} __asrc_asrcsr_bits;

/* ASRC Clock Divider Register (ASRC.ASRCDR1) */
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

/* ASRC Clock Divider Register (ASRC.ASRCDR2) */
typedef struct {
__REG32 AICPC     : 3;
__REG32 AICDC     : 3;
__REG32 AOCPC     : 3;
__REG32 AOCDC     : 3;
__REG32           :20;
} __asrc_asrcdr2_bits;

/* ASRC Status Register (ASRC.ASRSTR) */
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

/* ASRC Task Queue FIFO Register(ASRC.ASRTFR1) */
typedef struct {
__REG32           : 6;
__REG32 TF_BASE   : 7;
__REG32 TF_FILL   : 7;
__REG32           :12;
} __asrc_asrtfr1_bits;

/* Channel Counter Register (ASRC.ASRCCR) */
typedef struct {
__REG32 ACIA      : 4;
__REG32 ACIB      : 4;
__REG32 ACIC      : 4;
__REG32 ACOA      : 4;
__REG32 ACOB      : 4;
__REG32 ACOC      : 4;
__REG32           : 8;
} __asrc_asrccr_bits;

/* Ideal Ratio Register for Pair A High (ASRC.ASRIDRHA) */
typedef struct {
__REG32 IDRATIOA  : 8;
__REG32           :24;
} __asrc_asridrha_bits;

/* Ideal Ratio Register for Pair B High (ASRC.ASRIDRHB) */
typedef struct {
__REG32 IDRATIOB  : 8;
__REG32           :24;
} __asrc_asridrhb_bits;

/* Ideal Ratio Register for Pair C High (ASRC.ASRIDRHC) */
typedef struct {
__REG32 IDRATIOC  : 8;
__REG32           :24;
} __asrc_asridrhc_bits;

/* ASRC 76KHz Period Register in Terms of ASRC Processing Clock (ASRC.ASR76K) */
typedef struct {
__REG32 ASR76K    :17;
__REG32           :15;
} __asrc_asr76k_bits;

/* ASRC 56KHz Period Register in Terms of ASRC Processing Clock (ASRC.ASR56K) */
typedef struct {
__REG32 ASR56K    :17;
__REG32           :15;
} __asrc_asr56k_bits;

/* ASRC Misc Control Register for Pair A (ASRC.ASRMCRA) */
typedef struct {
__REG32 INFIFO_THRESHOLDA   : 6;
__REG32                     : 4;
__REG32 RSYNOFA             : 1;
__REG32 RSYNIFA             : 1;
__REG32 OUTFIFO_THRESHOLDA  : 6;
__REG32                     : 2;
__REG32 BYPASSPOLYA         : 1;
__REG32 BUFSTALLA           : 1;
__REG32 EXTTHRSHA           : 1;
__REG32 ZEROBUFA            : 1;
__REG32                     : 8;
} __asrc_asrmcra_bits;

/* ASRC FIFO Status Register for Pair A (ASRC.ASRFSTA) */
typedef struct {
__REG32 INFIFO_FILLA        : 7;
__REG32                     : 4;
__REG32 IAEA                : 1;
__REG32 OUTFIFO_FILLA       : 7;
__REG32                     : 4;
__REG32 OAFA                : 1;
__REG32                     : 8;
} __asrc_asrfsta_bits;

/* ASRC Misc Control Register for Pair B (ASRC.ASRMCRB) */
typedef struct {
__REG32 INFIFO_THRESHOLDB   : 6;
__REG32                     : 4;
__REG32 RSYNOFB             : 1;
__REG32 RSYNIFB             : 1;
__REG32 OUTFIFO_THRESHOLDB  : 6;
__REG32                     : 2;
__REG32 BYPASSPOLYB         : 1;
__REG32 BUFSTALLB           : 1;
__REG32 EXTTHRSHB           : 1;
__REG32 ZEROBUFB            : 1;
__REG32                     : 8;
} __asrc_asrmcrb_bits;

/* ASRC FIFO Status Register for Pair B (ASRC.ASRFSTB) */
typedef struct {
__REG32 INFIFO_FILLB        : 7;
__REG32                     : 4;
__REG32 IAEB                : 1;
__REG32 OUTFIFO_FILLB       : 7;
__REG32                     : 4;
__REG32 OAFB                : 1;
__REG32                     : 8;
} __asrc_asrfstb_bits;

/* ASRC FIFO Status Register for Pair B (ASRC.ASRFSTB) */
typedef struct {
__REG32 INFIFO_THRESHOLDC   : 6;
__REG32                     : 4;
__REG32 RSYNOFC             : 1;
__REG32 RSYNIFC             : 1;
__REG32 OUTFIFO_THRESHOLDC  : 6;
__REG32                     : 2;
__REG32 BYPASSPOLYC         : 1;
__REG32 BUFSTALLC           : 1;
__REG32 EXTTHRSHC           : 1;
__REG32 ZEROBUFC            : 1;
__REG32                     : 8;
} __asrc_asrmcrc_bits;

/* ASRC FIFO Status Register for Pair C (ASRC.ASRFSTC) */
typedef struct {
__REG32 INFIFO_FILLC        : 7;
__REG32                     : 4;
__REG32 IAEC                : 1;
__REG32 OUTFIFO_FILLC       : 7;
__REG32                     : 4;
__REG32 OAFC                : 1;
__REG32                     : 8;
} __asrc_asrfstc_bits;

/* ASRC Misc Control Register 1 for Pair A,B,C (ASRC.ASRMCR1x) */
typedef struct {
__REG32 OW16                : 1;
__REG32 OSGN                : 1;
__REG32 OMSB                : 1;
__REG32                     : 5;
__REG32 IMSB                : 1;
__REG32 IWD                 : 3;
__REG32                     :20;
} __asrc_asrmcr1_bits;

/* Port Timing Control Register n (AUDMUX.PTCRn) */
typedef struct {
__REG32                     :11;
__REG32 SYN                 : 1;
__REG32 RCSEL               : 4;
__REG32 RCLKDIR             : 1;
__REG32 RFSEL               : 4;
__REG32 RFSDIR              : 1;
__REG32 TCSEL               : 4;
__REG32 TCLKDIR             : 1;
__REG32 TFSEL               : 4;
__REG32 TFSDIR              : 1;
} __audmux_ptcr_bits;

/* Port Data Control Register n (AUDMUX.PDCRn) */
typedef struct {
__REG32 INMMASK             : 8;
__REG32 MODE                : 1;
__REG32                     : 3;
__REG32 TXRXEN              : 1;
__REG32 RXDSEL              : 3;
__REG32                     :16;
} __audmux_pdcr_bits;

/* CCM Control Register (CCM.CCR) */
typedef struct {
__REG32 OSCNT     : 8;
__REG32           : 1;
__REG32 CAMP1_EN  : 1;
__REG32 CAMP2_EN  : 1;
__REG32           : 1;
__REG32 COSC_EN   : 1;
__REG32 TM_EN     : 1;
__REG32 TMF_EN    : 1;
__REG32           :17;
} __ccm_ccmr_bits;

/* CCM Control Divider Register (CCM.CCDR) */
typedef struct {
__REG32                   :16;
__REG32 emi_hs_mask       : 1;
__REG32 emi_hs_fast_mask  : 1;
__REG32 emi_hs_slow_mask  : 1;
__REG32 emi_hs_int1_mask  : 1;
__REG32 emi_hs_int2_mask  : 1;
__REG32 ipu_hs_mask       : 1;
__REG32                   :10;
} __ccm_ccdr_bits;

/* CCM Status Register(CCM.CSR) */
typedef struct {
__REG32 ref_en_b        : 1;
__REG32 temp_mon_alarm  : 1;
__REG32 camp1_ready     : 1;
__REG32 camp2_ready     : 1;
__REG32 lvs_value       : 1;
__REG32 cosc_ready      : 1;
__REG32                 :26;
} __ccm_csr_bits;

/* CCM Clock Switcher Register (CCM.CCSR) */
typedef struct {
__REG32 pll3_sw_clk_sel : 1;
__REG32 pll2_sw_clk_sel : 1;
__REG32 pll1_sw_clk_sel : 1;
__REG32 pll3_div_podf   : 2;
__REG32 pll2_div_podf   : 2;
__REG32 step_sel        : 2;
__REG32 pll4_sw_clk_sel : 1;
__REG32 lp_apm          : 1;
__REG32                 :21;
} __ccm_ccsr_bits;

/* CCM Arm Clock Root Register (CCM.CACRR) */
typedef struct {
__REG32 arm_podf        : 3;
__REG32                 :29;
} __ccm_cacrr_bits;

/* CCM Bus Clock Divider Register(CCM.CBCDR) */
typedef struct {
__REG32 perclk_podf     : 3;
__REG32 perclk_pred2    : 3;
__REG32 perclk_pred1    : 2;
__REG32 ipg_podf        : 2;
__REG32 ahb_podf        : 3;
__REG32 nfc_podf        : 3;
__REG32 axi_a_podf      : 3;
__REG32 axi_b_podf      : 3;
__REG32 emi_slow_podf   : 3;
__REG32 periph_clk_sel  : 1;
__REG32 emi_clk_sel     : 1;
__REG32                 : 5;
} __ccm_cbcdr_bits;

/* CCM Bus Clock Multiplexer Register (CCM.CBCMR) */
typedef struct {
__REG32 perclk_ipg_sel    : 1;
__REG32 perclk_lp_apm_sel : 1;
__REG32 debug_apb_clk_sel : 2;
__REG32 gpu_clk_sel       : 2;
__REG32 ipu_hsp_clk_sel   : 2;
__REG32 arm_axi_clk_sel   : 2;
__REG32 ddr_clk_sel       : 2;
__REG32 periph_apm_sel    : 2;
__REG32 vpu_axi_clk_sel   : 2;
__REG32 gpu2d_clk_sel     : 2;
__REG32                   :14;
} __ccm_cbcmr_bits;

/* CCM Serial Clock Multiplexer Register 1 (CCM.CSCMR1) */
typedef struct {
__REG32 ssi_ext1_com        : 1;
__REG32 ssi_ext2_com        : 1;
__REG32 spdif_xtal_clk_sel  : 2;
__REG32 ecspi_clk_sel       : 2;
__REG32 tve_ext_clk_sel     : 1;
__REG32                     : 1;
__REG32 ssi_apm_clk_sel     : 2;
__REG32 vpu_rclk_sel        : 1;
__REG32 ssi3_clk_sel        : 1;
__REG32 ssi2_clk_sel        : 2;
__REG32 ssi1_clk_sel        : 2;
__REG32 esdhc3_clk_sel      : 2;
__REG32 esdhc4_clk_sel      : 1;
__REG32 esdhc2_clk_sel      : 1;
__REG32 esdhc1_clk_sel      : 2;
__REG32 usboh3_clk_sel      : 2;
__REG32 uart_clk_sel        : 2;
__REG32 usb_phy_clk_sel     : 1;
__REG32                     : 1;
__REG32 ssi_ext1_clk_sel    : 2;
__REG32 ssi_ext2_clk_sel    : 2;
} __ccm_cscmr1_bits;

/* CCM Serial Clock Multiplexer Register 2 (CCM.CSCMR2) */
typedef struct {
__REG32 spdif0_clk_sel      : 2;
__REG32                     : 2;
__REG32 spdif0_com          : 1;
__REG32                     : 1;
__REG32 can_clk_sel         : 2;
__REG32 ldb_di0_clk_sel     : 1;
__REG32 ldb_di1_clk_sel     : 1;
__REG32 ldb_di0_ipu_div     : 1;
__REG32 ldb_di1_ipu_div     : 1;
__REG32 firi_clk_sel        : 2;
__REG32 ieee_clk_sel        : 2;
__REG32 esai_post_sel       : 3;
__REG32 esai_pre_sel        : 2;
__REG32 asrc_clk_sel        : 1;
__REG32 csi_mclk1_clk_sel   : 2;
__REG32 csi_mclk2_clk_sel   : 2;
__REG32 di0_clk_sel         : 3;
__REG32 di1_clk_sel         : 3;
} __ccm_cscmr2_bits;

/* CCM Serial Clock Divider Register 1 (CCM.CSCDR1) */
typedef struct {
__REG32 uart_clk_podf   : 3;
__REG32 uart_clk_pred   : 3;
__REG32 usboh3_clk_podf : 2;
__REG32 usboh3_clk_pred : 3;
__REG32 esdhc1_clk_podf : 3;
__REG32 pgc_clk_podf    : 2;
__REG32 esdhc1_clk_pred : 3;
__REG32 esdhc3_clk_podf : 3;
__REG32 esdhc3_clk_pred : 3;
__REG32                 : 7;
} __ccm_cscdr1_bits;

/* CCM SSI1 Clock Divider Register(CCM.CS1CDR) */
typedef struct {
__REG32 ss1_clk_podf      : 6;
__REG32 ssi1_clk_pred     : 3;
__REG32 esai_clk_pred     : 3;
__REG32                   : 4;
__REG32 ssi_ext1_clk_podf : 6;
__REG32 ssi_ext1_clk_pred : 3;
__REG32 esai_clk_podf     : 6;
__REG32                   : 1;
} __ccm_cs1cdr_bits;

/* CCM SSI2 Clock Divider Register(CCM.CS2CDR) */
typedef struct {
__REG32 ssi2_clk_podf     : 6;
__REG32 ssi2_clk_pred     : 3;
__REG32                   : 7;
__REG32 ssi_ext2_clk_podf : 6;
__REG32 ssi_ext2_clk_pred : 3;
__REG32                   : 7;
} __ccm_cs2cdr_bits;

/* CCM DI Clock Divider Register(CCM.CDCDR) */
typedef struct {
__REG32 usb_phy_podf      : 3;
__REG32 usb_phy_pred      : 3;
__REG32 di1_clk_pred      : 3;
__REG32                   : 7;
__REG32 di_pll4_podf      : 3;
__REG32 spdif0_clk_podf   : 6;
__REG32 spdif0_clk_pred   : 3;
__REG32 tve_clk_pred      : 3;
__REG32                   : 1;
} __ccm_cdcdr_bits;

/* CCM HSC Clock Divider Register(CCM.CHSCCDR) */
typedef struct {
__REG32 ssi1_mlb_spdif_src  : 2;
__REG32 ssi2_mlb_spdif_src  : 2;
__REG32                     :28;
} __ccm_chsccdr_bits;

/* CCM Serial Clock Divider Register 2(CCM.CSCDR2) */
typedef struct {
__REG32 ieee_clk_podf       : 6;
__REG32 ieee_clk_pred       : 3;
__REG32 asrc_clk_podf       : 6;
__REG32                     : 4;
__REG32 ecspi_clk_podf      : 6;
__REG32 ecspi_clk_pred      : 3;
__REG32 asrc_clk_pred       : 3;
__REG32                     : 1;
} __ccm_cscdr2_bits;

/* CCM Serial Clock Divider Register 3(CCM.CSCDR3) */
typedef struct {
__REG32 firi_clk_podf       : 6;
__REG32 firi_clk_pred       : 3;
__REG32                     :23;
} __ccm_cscdr3_bits;

/* CCM Serial Clock Divider Register 4(CCM.CSCDR4) */
typedef struct {
__REG32 csi_mclk1_clk_podf  : 6;
__REG32 csi_mclk1_clk_pred  : 3;
__REG32 csi_mclk2_clk_podf  : 6;
__REG32                     : 1;
__REG32 csi_mclk2_clk_pred  : 3;
__REG32                     :13;
} __ccm_cscdr4_bits;

/* CCM Divider Handshake In-Process Register(CCM.CDHIPR) */
typedef struct {
__REG32 axi_a_podf_busy     : 1;
__REG32 axi_b_podf_busy     : 1;
__REG32 emi_slow_podf_busy  : 1;
__REG32 ahb_podf_busy       : 1;
__REG32 nfc_podf_busy       : 1;
__REG32 periph_clk_sel_busy : 1;
__REG32 emi_clk_sel_busy    : 1;
__REG32                     : 9;
__REG32 arm_podf_busy       : 1;
__REG32                     :15;
} __ccm_cdhipr_bits;

/* CCM DVFS Control Register(CCM.CDCR) */
typedef struct {
__REG32 periph_clk_dvfs_podf          : 2;
__REG32 arm_freq_shift_divider        : 1;
__REG32 en_frequency_req_ordering     : 1;
__REG32                               : 1;
__REG32 software_dvfs_en              : 1;
__REG32 sw_periph_clk_div_req         : 1;
__REG32 sw_periph_clk_div_req_status  : 1;
__REG32 bypass_emi_dvfs_hs            : 1;
__REG32 bypass_ipu_dvfs_hs            : 1;
__REG32                               :22;
} __ccm_cdcr_bits;

/* CCM Testing Observability Register(CCM.CTOR) */
typedef struct {
__REG32 obs_output_2_sel              : 4;
__REG32 obs_output_1_sel              : 4;
__REG32 obs_output_0_sel              : 5;
__REG32 obs_en                        : 1;
__REG32                               :18;
} __ccm_ctor_bits;

/* CCM Low Power Control Register(CCM.CLPCR) */
typedef struct {
__REG32 LPM                           : 2;
__REG32 bypass_pmic_vfunctional_ready : 1;
__REG32 lpsr_clk_sel                  : 2;
__REG32 ARM_clk_dis_on_lpm            : 1;
__REG32 SBYOS                         : 1;
__REG32 dis_ref_osc                   : 1;
__REG32 VSTBY                         : 1;
__REG32 stby_count                    : 2;
__REG32 cosc_pwrdown                  : 1;
__REG32                               : 4;
__REG32 bypass_sahara_lpm_hs          : 1;
__REG32 bypass_rtic_lpm_hs            : 1;
__REG32 bypass_ipu_lpm_hs             : 1;
__REG32 bypass_emi_lpm_hs             : 1;
__REG32 bypass_emi_fast_lpm_hs        : 1;
__REG32 bypass_emi_slow_lpm_hs        : 1;
__REG32 bypass_emi_int1_lpm_hs        : 1;
__REG32 bypass_emi_int2_lpm_hs        : 1;
__REG32 bypass_sdma_lpm_hs            : 1;
__REG32 bypass_max_lpm_hs             : 1;
__REG32 bypass_scc_lpm_hs             : 1;
__REG32 bypass_can1_lpm_hs            : 1;
__REG32 bypass_can2_lpm_hs            : 1;
__REG32                               : 3;
} __ccm_clpcr_bits;

/* CCM Interrupt Status Register(CCM.CISR) */
typedef struct {
__REG32 lrf_pll1              : 1;
__REG32 lrf_pll2              : 1;
__REG32 lrf_pll3              : 1;
__REG32 lrf_pll4              : 1;
__REG32 camp1_ready           : 1;
__REG32 camp2_ready           : 1;
__REG32 cosc_ready            : 1;
__REG32                       : 9;
__REG32 dividers_loaded       : 1;
__REG32 axi_a_podf_loaded     : 1;
__REG32 axi_b_podf_loaded     : 1;
__REG32 emi_slow_podf_loaded  : 1;
__REG32 ahb_podf_loaded       : 1;
__REG32 nfc_podf_loaded       : 1;
__REG32 periph_clk_sel_loaded : 1;
__REG32 emi_clk_sel_loaded    : 1;
__REG32                       : 1;
__REG32 temp_mon_alarm        : 1;
__REG32 arm_podf_loaded       : 1;
__REG32                       : 5;
} __ccm_cisr_bits;

/* CCM Interrupt Mask Register(CCM.CIMR) */
typedef struct {
__REG32 mask_lrf_pll1               : 1;
__REG32 mask_lrf_pll2               : 1;
__REG32 mask_lrf_pll3               : 1;
__REG32 mask_lrf_pll4               : 1;
__REG32 mask_camp1_ready            : 1;
__REG32 mask_camp2_ready            : 1;
__REG32 mask_cosc_ready             : 1;
__REG32                             : 9;
__REG32 mask_dividers_loaded        : 1;
__REG32 mask_axi_a_podf_loaded      : 1;
__REG32 mask_axi_b_podf_loaded      : 1;
__REG32 mask_emi_slow_podf_loaded   : 1;
__REG32 mask_ahb_podf_loaded        : 1;
__REG32 mask_nfc_podf_loaded        : 1;
__REG32 mask_periph_clk_sel_loaded  : 1;
__REG32 mask_emi_clk_sel_loaded     : 1;
__REG32                             : 1;
__REG32 mask_temp_mon_alarm         : 1;
__REG32 arm_podf_loaded             : 1;
__REG32                             : 5;
} __ccm_cimr_bits;

/* CCM Clock Output Source Register (CCM.CCOSR) */
typedef struct {
__REG32 cko1_sel        : 4;
__REG32 cko1_div        : 3;
__REG32 cko1_en         : 1;
__REG32                 : 8;
__REG32 cko2_sel        : 5;
__REG32 cko2_div        : 3;
__REG32 cko2_en         : 1;
__REG32                 : 7;
} __ccm_ccosr_bits;

/* CCM General Purpose Register(CCM.CGPR) */
typedef struct {
__REG32                         : 4;
__REG32 efuse_prog_supply_gate  : 1;
__REG32                         :27;
} __ccm_cgpr_bits;

/* CCM Clock Gating Register(CCM.CCGR0) */
typedef struct {
__REG32 CG0   : 2;
__REG32 CG1   : 2;
__REG32 CG2   : 2;
__REG32 CG3   : 2;
__REG32 CG4   : 2;
__REG32 CG5   : 2;
__REG32 CG6   : 2;
__REG32 CG7   : 2;
__REG32 CG8   : 2;
__REG32 CG9   : 2;
__REG32 CG10  : 2;
__REG32 CG11  : 2;
__REG32 CG12  : 2;
__REG32 CG13  : 2;
__REG32 CG14  : 2;
__REG32 CG15  : 2;
} __ccm_ccgr_bits;

/* CCM Module Enable Override Register(CCM.CMEOR) */
typedef struct {
__REG32 mod_en_ov_sahara    : 1;
__REG32                     : 1;
__REG32 mod_en_ov_owire     : 1;
__REG32 mod_en_ov_iim       : 1;
__REG32 mod_en_ov_esdhc     : 1;
__REG32 mod_en_ov_gpt       : 1;
__REG32 mod_en_ov_epit      : 1;
__REG32 mod_en_ov_gpu       : 1;
__REG32 mod_en_ov_dap       : 1;
__REG32 mod_en_ov_vpu       : 1;
__REG32 mod_en_ov_gpu2d     : 1;
__REG32                     :17;
__REG32 mod_en_ov_can2_cpi  : 1;
__REG32 mod_en_ov_can2_mbm  : 1;
__REG32 mod_en_ov_can1_cpi  : 1;
__REG32 mod_en_ov_can1_mbm  : 1;
} __ccm_cmeor_bits;

/* Control Register (CSPI.CONREG) */
typedef struct{
__REG32 EN          : 1;
__REG32 MODE        : 1;
__REG32 XCH         : 1;
__REG32 SMC         : 1;
__REG32 POL         : 1;
__REG32 PHA         : 1;
__REG32 SSCTL       : 1;
__REG32 SSPOL       : 1;
__REG32 DRCTL       : 2;
__REG32             : 2;
__REG32 CHIP_SELECT : 2;
__REG32             : 2;
__REG32 DATA_RATE   : 3;
__REG32             : 1;
__REG32 BURST_LENGTH:12;
} __cspi_conreg_bits;

/* Interrupt Control Register (CSPI.INTREG) */
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

/* DMA Control Register (CSPI.DMAREG) */
typedef struct{
__REG32 TEDEN  : 1;
__REG32 THDEN  : 1;
__REG32        : 2;
__REG32 RHDEN  : 1;
__REG32 RFDEN  : 1;
__REG32        :26;
} __cspi_dmareg_bits;

/* Status Register (CSPI.STATREG) */
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

/* Sample Period Control Register (CSPI.PERIODREG) */
typedef struct{
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32               :16;
} __cspi_periodreg_bits;

/* Test Control Register (CSPI.TESTREG) */
typedef struct{
__REG32 TXCNT     : 4;
__REG32 RXCNT     : 4;
__REG32           : 6;
__REG32 LBC       : 1;
__REG32 SWAP      : 1;
__REG32           :16;
} __cspi_testreg_bits;

/* DPLL Control Register(DPLLC.DP_CTL) */
typedef struct{
__REG32 LRF         : 1;
__REG32 BRMO        : 1;
__REG32 PLM         : 1;
__REG32 RCP         : 1;
__REG32 RST         : 1;
__REG32 UPEN        : 1;
__REG32 PRE         : 1;
__REG32 HFSM        : 1;
__REG32 ref_clk_sel : 2;
__REG32 ref_clk_div : 1;
__REG32 ADE         : 1;
__REG32 dpdck0_2_en : 1;
__REG32 mul_ctrl    : 1;
__REG32             :18;
} __dpllc_dp_ctl_bits;

/* DPLL Config Register(DPLLC.DP_CONFIG) */
typedef struct{
__REG32 LDREQ       : 1;
__REG32 AREN        : 1;
__REG32             :30;
} __dpllc_dp_config_bits;

/* DPLL Operation Register(DPLLC.DP_OP) */
typedef struct{
__REG32 PDF         : 4;
__REG32 MFI         : 4;
__REG32             :24;
} __dpllc_dp_op_bits;

/* DPLL Multiplication Factor Denominator Register(DPLLC.DP_MFD) */
typedef struct{
__REG32 MFD         :27;
__REG32             : 5;
} __dpllc_dp_mfd_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFN) */
typedef struct{
__REG32 MFN         :27;
__REG32             : 5;
} __dpllc_dp_mfn_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFNMINUS) */
typedef struct{
__REG32 MFNMINUS    :27;
__REG32             : 5;
} __dpllc_dp_mfnminus_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFNPLUS) */
typedef struct{
__REG32 MFNPLUS     :27;
__REG32             : 5;
} __dpllc_dp_mfnplus_bits;

/* DPLL High Frequency Support, Operation Register(DPLLC.DP_HFS_OP) */
typedef struct{
__REG32 HFS_PDF     : 4;
__REG32 HFS_MFI     : 4;
__REG32             :24;
} __dpllc_dp_hfs_op_bits;

/* DPLL HFS MFD Register (DPLLC.DP_HFS_MFD) */
typedef struct{
__REG32 HFS_MFD     :27;
__REG32             : 5;
} __dpllc_dp_hfs_mfd_bits;

/* DPLL HFS Multiplication Factor Numerator Register (DPLLC.DP_HFS_MFN) */
typedef struct{
__REG32 HFS_MFN     :27;
__REG32             : 5;
} __dpllc_dp_hfs_mfn_bits;

/* DPLL Multiplication Factor Numerator Toggle Control Register (DPLLC.DP_MFN_TOGC) */
typedef struct{
__REG32 TOG_MFN_CNT :16;
__REG32 TOG_EN      : 1;
__REG32 TOG_DIS     : 1;
__REG32             :14;
} __dpllc_dp_mfn_togc_bits;

/* Desense Status Register(DPLLC.DP_DESTAT) */
typedef struct{
__REG32 TOG_MFN     :27;
__REG32             : 4;
__REG32 TOG_SEL     : 1;
} __dpllc_dp_destat_bits;

/* DVFSC.DVFSTHRS Register */
typedef struct{
__REG32 PNCTHR      : 6;
__REG32             :10;
__REG32 DNTHR       : 6;
__REG32 UPTHR       : 6;
__REG32             : 4;
} __dvfsc_dvfsthrs_bits;

/* DVFSC.DVFSCOUN Register */
typedef struct{
__REG32 UPCNT       : 8;
__REG32             : 8;
__REG32 DNCNT       : 8;
__REG32             : 8;
} __dvfsc_dvfscoun_bits;

/* DVFSC.DVFSSIG1 Register */
typedef struct{
__REG32             : 2;
__REG32 WSW6        : 3;
__REG32 WSW7        : 3;
__REG32 WSW8        : 3;
__REG32 WSW9        : 3;
__REG32 WSW10       : 3;
__REG32 WSW11       : 3;
__REG32 WSW12       : 3;
__REG32 WSW13       : 3;
__REG32 WSW14       : 3;
__REG32 WSW15       : 3;
} __dvfsc_dvfssig1_bits;

/* DVFSC.DVFSSIG0 Register */
typedef struct{
__REG32 WSW0        : 6;
__REG32 WSW1        : 6;
__REG32             : 8;
__REG32 WSW2        : 3;
__REG32 WSW3        : 3;
__REG32 WSW4        : 3;
__REG32 WSW5        : 3;
} __dvfsc_dvfssig0_bits;

/* DVFSC.DVFSGPC0 Register */
typedef struct{
__REG32 GPBC0       :17;
__REG32             :13;
__REG32 C0ACT       : 1;
__REG32 C0STRT      : 1;
} __dvfsc_dvfsgpc0_bits;

/* DVFSC.DVFSGPC1 Register */
typedef struct{
__REG32 GPBC1       :17;
__REG32             :13;
__REG32 C1ACT       : 1;
__REG32 C1STRT      : 1;
} __dvfsc_dvfsgpc1_bits;

/* DVFSC.DVFSGPBT Register */
typedef struct{
__REG32 GPB0        : 1;
__REG32 GPB1        : 1;
__REG32 GPB2        : 1;
__REG32 GPB3        : 1;
__REG32 GPB4        : 1;
__REG32 GPB5        : 1;
__REG32 GPB6        : 1;
__REG32 GPB7        : 1;
__REG32 GPB8        : 1;
__REG32 GPB9        : 1;
__REG32 GPB10       : 1;
__REG32 GPB11       : 1;
__REG32 GPB12       : 1;
__REG32 GPB13       : 1;
__REG32 GPB14       : 1;
__REG32 GPB15       : 1;
__REG32             :16;
} __dvfsc_dvfsgpbt_bits;

/* DVFSC.DVFSEMAC Register */
typedef struct{
__REG32 EMAC        : 9;
__REG32             :23;
} __dvfsc_dvfsemac_bits;

/* DVFSC.DVFSCNTR Register */
typedef struct{
__REG32 DVFEN       : 1;
__REG32             : 2;
__REG32 LTBRSR      : 2;
__REG32 LTBRSH      : 1;
__REG32 PFUS        : 3;
__REG32 PFUE        : 1;
__REG32             : 1;
__REG32 DIV_RATIO   : 6;
__REG32 MINF        : 1;
__REG32 MAXF        : 1;
__REG32 WFIM        : 1;
__REG32 FSVAI       : 2;
__REG32 FSVAIM      : 1;
__REG32 PIRQS       : 1;
__REG32 DVFIS       : 1;
__REG32 LBFL0       : 1;
__REG32 LBFL1       : 1;
__REG32 LBMI        : 1;
__REG32 DVFEV       : 1;
__REG32 DIV3CK      : 3;
} __dvfsc_dvfscntr_bits;

/* DVFSC.DVFSLTR0_0 Register */
typedef struct {
__REG32 LTS0_0      : 4;
__REG32 LTS0_1      : 4;
__REG32 LTS0_2      : 4;
__REG32 LTS0_3      : 4;
__REG32 LTS0_4      : 4;
__REG32 LTS0_5      : 4;
__REG32 LTS0_6      : 4;
__REG32 LTS0_7      : 4;
} __dvfsc_dvfsltr0_0_bits;

/* DVFSC.DVFSLTR0_1 Register */
typedef struct {
__REG32 LTS0_8      : 4;
__REG32 LTS0_9      : 4;
__REG32 LTS0_10     : 4;
__REG32 LTS0_11     : 4;
__REG32 LTS0_12     : 4;
__REG32 LTS0_13     : 4;
__REG32 LTS0_14     : 4;
__REG32 LTS0_15     : 4;
} __dvfsc_dvfsltr0_1_bits;

/* DVFSC.DVFSLTR1_0 Register */
typedef struct {
__REG32 LTS1_0      : 4;
__REG32 LTS1_1      : 4;
__REG32 LTS1_2      : 4;
__REG32 LTS1_3      : 4;
__REG32 LTS1_4      : 4;
__REG32 LTS1_5      : 4;
__REG32 LTS1_6      : 4;
__REG32 LTS1_7      : 4;
} __dvfsc_dvfsltr1_0_bits;

/* DVFSC.DVFSLTR1_1 Register */
typedef struct {
__REG32 LTS1_8      : 4;
__REG32 LTS1_9      : 4;
__REG32 LTS1_10     : 4;
__REG32 LTS1_11     : 4;
__REG32 LTS1_12     : 4;
__REG32 LTS1_13     : 4;
__REG32 LTS1_14     : 4;
__REG32 LTS1_15     : 4;
} __dvfsc_dvfsltr1_1_bits;

/* DVFSC.DVFSPT0 Register */
typedef struct {
__REG32 FPTN0       :17;
__REG32 PT0A        : 1;
__REG32             :14;
} __dvfsc_dvfspt0_bits;

/* DVFSC.DVFSPT1 Register */
typedef struct {
__REG32 FPTN1       :17;
__REG32 PT1A        : 1;
__REG32             :14;
} __dvfsc_dvfspt1_bits;

/* DVFSC.DVFSPT2 Register */
typedef struct {
__REG32 FPTN2       :17;
__REG32 PT2A        : 1;
__REG32             :14;
} __dvfsc_dvfspt2_bits;

/* DVFSC.DVFSPT3 Register */
typedef struct {
__REG32 FPTN3       :17;
__REG32 PT3A        : 1;
__REG32             :14;
} __dvfsc_dvfspt3_bits;

/* LTR0 Register (DVFSP.LTR0) */
typedef struct {
__REG32 DIV3CK      : 3;
__REG32 SIGD0       : 1;
__REG32 SIGD1       : 1;
__REG32 SIGD2       : 1;
__REG32 SIGD3       : 1;
__REG32 SIGD4       : 1;
__REG32 SIGD5       : 1;
__REG32 SIGD6       : 1;
__REG32 SIGD7       : 1;
__REG32 SIGD8       : 1;
__REG32 SIGD9       : 1;
__REG32 SIGD10      : 1;
__REG32 SIGD11      : 1;
__REG32 SIGD12      : 1;
__REG32 DNTHR       : 6;
__REG32 UPTHR       : 6;
__REG32             : 1;
__REG32 SIGD13      : 1;
__REG32 SIGD14      : 1;
__REG32 SIGD15      : 1;
} __dvfsp_ltr0_bits;

/* LTR1 Register (DVFSP.LTR1) */
typedef struct {
__REG32 PNCTHR      : 6;
__REG32 UPCNT       : 8;
__REG32 DNCNT       : 8;
__REG32 LTBRSR      : 1;
__REG32 LTBRSH      : 1;
__REG32             : 2;
__REG32 DIV_RATIO   : 6;
} __dvfsp_ltr1_bits;

/* LTR2 Register (DVFSP.LTR2) */
typedef struct {
__REG32 EMAC        : 9;
__REG32             : 2;
__REG32 WSW9        : 3;
__REG32 WSW10       : 3;
__REG32 WSW11       : 3;
__REG32 WSW12       : 3;
__REG32 WSW13       : 3;
__REG32 WSW14       : 3;
__REG32 WSW15       : 3;
} __dvfsp_ltr2_bits;

/* LTR3 Register (DVFSP.LTR3) */
typedef struct {
__REG32             : 5;
__REG32 WSW0        : 3;
__REG32 WSW1        : 3;
__REG32 WSW2        : 3;
__REG32 WSW3        : 3;
__REG32 WSW4        : 3;
__REG32 WSW5        : 3;
__REG32 WSW6        : 3;
__REG32 WSW7        : 3;
__REG32 WSW8        : 3;
} __dvfsp_ltr3_bits;

/* LTBR0 Register (DVFSP.LTBR0) */
typedef struct {
__REG32 LTS0_0      : 4;
__REG32 LTS0_1      : 4;
__REG32 LTS0_2      : 4;
__REG32 LTS0_3      : 4;
__REG32 LTS0_4      : 4;
__REG32 LTS0_5      : 4;
__REG32 LTS0_6      : 4;
__REG32 LTS0_7      : 4;
} __dvfsp_ltbr0_bits;

/* LTBR1 Register (DVFSP.LTBR1) */
typedef struct {
__REG32 LTS0_8      : 4;
__REG32 LTS0_9      : 4;
__REG32 LTS0_10     : 4;
__REG32 LTS0_11     : 4;
__REG32 LTS0_12     : 4;
__REG32 LTS0_13     : 4;
__REG32 LTS0_14     : 4;
__REG32 LTS0_15     : 4;
} __dvfsp_ltbr1_bits;

/* PMCR0 Register (DVFSP.PMCR0) */
typedef struct {
__REG32             : 4;
__REG32 DVFEN       : 1;
__REG32             : 5;
__REG32 WFIM        : 1;
__REG32             : 2;
__REG32 FSVAI       : 2;
__REG32 FSVAIM      : 1;
__REG32             : 2;
__REG32 LBCF        : 2;
__REG32 LBFL        : 1;
__REG32 LBMI        : 1;
__REG32 DVFIS       : 1;
__REG32 DVFEV       : 1;
__REG32             : 3;
__REG32 UDSC        : 1;
__REG32             : 4;
} __dvfsp_pmcr0_bits;

/* PMCR0 Register (DVFSP.PMCR1) */
typedef struct {
__REG32 DVGP        :16;
__REG32 P2PM        : 1;
__REG32 P4PM        : 1;
__REG32 P1IFM       : 1;
__REG32 P1ISM       : 1;
__REG32 P1INM       : 1;
__REG32             :11;
} __dvfsp_pmcr1_bits;

/* Control Register (ECSPI.CONREG) */
typedef struct {
__REG32 EN            : 1;
__REG32 HT            : 1;
__REG32 XCH           : 1;
__REG32 SMC           : 1;
__REG32 CHANNEL_MODE  : 4;
__REG32 POST_DIVIDER  : 4;
__REG32 PRE_DIVIDER   : 4;
__REG32 DRCTL         : 2;
__REG32 CHANNEL_SELECT: 2;
__REG32 BURST_LENGTH  :12;
} __ecspi_conreg_bits;

/* Config Register (ECSPI.CONFIGREG) */
typedef struct {
__REG32 SCLK_PHA      : 4;
__REG32 SCLK_POL      : 4;
__REG32 SS_CTL        : 4;
__REG32 SS_POL        : 4;
__REG32 DATA_CTL      : 4;
__REG32 SCLK_CTL      : 4;
__REG32 HT_LENGTH     : 5;
__REG32               : 3;
} __ecspi_configreg_bits;

/* Interrupt Control Register (ECSPI.INTREG) */
typedef struct {
__REG32 TEEN          : 1;
__REG32 TDREN         : 1;
__REG32 TFEN          : 1;
__REG32 RREN          : 1;
__REG32 RDREN         : 1;
__REG32 RFEN          : 1;
__REG32 ROEN          : 1;
__REG32 TCEN          : 1;
__REG32               :24;
} __ecspi_intreg_bits;

/* DMA Control Register (ECSPI.DMAREG) */
typedef struct {
__REG32 TX_THRESHOLD  : 6;
__REG32               : 1;
__REG32 TEDEN         : 1;
__REG32               : 8;
__REG32 RX_THRESHOLD  : 6;
__REG32               : 1;
__REG32 RXDEN         : 1;
__REG32 RX_DMA_LENGTH : 6;
__REG32               : 1;
__REG32 RXTDEN        : 1;
} __ecspi_dmareg_bits;

/* Status Register (ECSPI.STATREG) */
typedef struct {
__REG32 TE            : 1;
__REG32 TDR           : 1;
__REG32 TF            : 1;
__REG32 RR            : 1;
__REG32 RDR           : 1;
__REG32 RF            : 1;
__REG32 RO            : 1;
__REG32 TC            : 1;
__REG32               :24;
} __ecspi_statreg_bits;

/* Sample Period Control Register (ECSPI.PERIODREG) */
typedef struct {
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32 CSD_CTL       : 6;
__REG32               :10;
} __ecspi_periodreg_bits;

/* Test Control Register (ECSPI.TESTREG) */
typedef struct {
__REG32 TXCNT         : 7;
__REG32               : 1;
__REG32 RXCNT         : 7;
__REG32               :16;
__REG32 LBC           : 1;
} __ecspi_testreg_bits;

/* Chip Select x General Configuration Register 1(EIM.CS<i>GCR1) */
typedef struct {
__REG32 CSEN          : 1;
__REG32 SWR           : 1;
__REG32 SRD           : 1;
__REG32 MUM           : 1;
__REG32 WFL           : 1;
__REG32 RFL           : 1;
__REG32 CRE           : 1;
__REG32 CREP          : 1;
__REG32 BL            : 3;
__REG32 WC            : 1;
__REG32 BCD           : 2;
__REG32 BCS           : 2;
__REG32 DSZ           : 3;
__REG32 SP            : 1;
__REG32 CSREC         : 3;
__REG32 AUS           : 1;
__REG32 GBC           : 3;
__REG32 WP            : 1;
__REG32 PSZ           : 4;
} __eim_csgcr1_bits;

/* Chip Select x General Configuration Register 2 (EIM.CS<i>GCR2) */
typedef struct {
__REG32 ADH               : 2;
__REG32                   : 2;
__REG32 DAPS              : 4;
__REG32 DAE               : 1;
__REG32 DAP               : 1;
__REG32                   : 2;
__REG32 MUX16_BYP_GRANT   : 1;
__REG32                   :19;
} __eim_csgcr2_bits;

/* Chip Select x Read Configuration Register 1(EIM.CS<i>RCR1) */
typedef struct {
__REG32 RCSN              : 3;
__REG32                   : 1;
__REG32 RCSA              : 3;
__REG32                   : 1;
__REG32 OEN               : 3;
__REG32                   : 1;
__REG32 OEA               : 3;
__REG32                   : 1;
__REG32 RADVN             : 3;
__REG32 RAL               : 1;
__REG32 RADVA             : 3;
__REG32                   : 1;
__REG32 RWSC              : 6;
__REG32                   : 2;
} __eim_csrcr1_bits;

/* Chip Select x Read Configuration Register 2 (EIM.CS<i>RCR2) */
typedef struct {
__REG32 RBEN              : 3;
__REG32 RBE               : 1;
__REG32 RBEA              : 3;
__REG32                   : 1;
__REG32 RL                : 2;
__REG32                   : 2;
__REG32 PAT               : 3;
__REG32 APR               : 1;
__REG32                   :16;
} __eim_csrcr2_bits;

/* Chip Select x Write Configuration Register 1 (EIM.CS<i>WCR1) */
typedef struct {
__REG32 WCSN              : 3;
__REG32 WCSA              : 3;
__REG32 WEN               : 3;
__REG32 WEA               : 3;
__REG32 WBEN              : 3;
__REG32 WBEA              : 3;
__REG32 WADVN             : 3;
__REG32 WADVA             : 3;
__REG32 WWSC              : 6;
__REG32 WBED              : 1;
__REG32 WAL               : 1;
} __eim_cswcr1_bits;

/* Chip Select x Write Configuration Register 2 (EIM.CS<i>WCR2) */
typedef struct {
__REG32 WBCDD             : 1;
__REG32                   :31;
} __eim_cswcr2_bits;

/* EIM Configuration Register (EIM.WCR) */
typedef struct {
__REG32 BCM               : 1;
__REG32 GBCD              : 2;
__REG32                   : 1;
__REG32 INTEN             : 1;
__REG32 INTPOL            : 1;
__REG32                   : 2;
__REG32 WDOG_EN           : 1;
__REG32 WDOG_LIMIT        : 2;
__REG32                   :21;
} __eim_wcr_bits;

/* EIM IP Access Register (EIM.WIAR) */
typedef struct {
__REG32 IPS_REQ           : 1;
__REG32 IPS_ACK           : 1;
__REG32 INT               : 1;
__REG32 ERRST             : 1;
__REG32 ACLK_EN           : 1;
__REG32                   :27;
} __eim_wiar_bits;

/* Error Address Register (EIM.EAR) */
typedef struct {
__REG32 IPS_REQ           : 1;
__REG32 IPS_ACK           : 1;
__REG32 INT               : 1;
__REG32 ERRST             : 1;
__REG32 ACLK_EN           : 1;
__REG32                   :27;
} __eim_ear_bits;

/*EPIT Control Register */
typedef struct {
__REG32 EN                : 1;
__REG32 ENMOD             : 1;
__REG32 OCIEN             : 1;
__REG32 RLD               : 1;
__REG32 PRESCALAR         :12;
__REG32 SWR               : 1;
__REG32 IOVW              : 1;
__REG32 DBGEN             : 1;
__REG32 WAITEN            : 1;
__REG32 RES               : 1;
__REG32 STOPEN            : 1;
__REG32 OM                : 2;
__REG32 CLKSRC            : 2;
__REG32                   : 6;
} __epit_epitcr_bits;

/* EPIT Status Register */
typedef struct {
__REG32 OCIF              : 1;
__REG32                   :31;
} __epit_epitsr_bits;

/* ESAI Control Register (ESAI.ECR)*/
typedef struct {
__REG32 ESAIEN            : 1;
__REG32 ERST              : 1;
__REG32                   :14;
__REG32 ERO               : 1;
__REG32 ERI               : 1;
__REG32 ETO               : 1;
__REG32 ETI               : 1;
__REG32                   :12;
} __esai_ecr_bits;

/* ESAI Status Register (ESAI.ESR)*/
typedef struct {
__REG32 RD                : 1;
__REG32 RED               : 1;
__REG32 RDE               : 1;
__REG32 RLS               : 1;
__REG32 TD                : 1;
__REG32 TED               : 1;
__REG32 TDE               : 1;
__REG32 TLS               : 1;
__REG32 TFE               : 1;
__REG32 RFF               : 1;
__REG32 TINIT             : 1;
__REG32                   :21;
} __esai_esr_bits;

/* Transmit FIFO Configuration Register (ESAI.TFCR) */
typedef struct {
__REG32 TFEN              : 1;
__REG32 TFR               : 1;
__REG32 TE0               : 1;
__REG32 TE1               : 1;
__REG32 TE2               : 1;
__REG32 TE3               : 1;
__REG32 TE4               : 1;
__REG32 TE5               : 1;
__REG32 TFWM              : 8;
__REG32 TWA               : 3;
__REG32 TIEN              : 1;
__REG32                   :12;
} __esai_etdr_bits;

/* Transmit FIFO Status Register (ESAI.TFSR) */
typedef struct {
__REG32 TFCNT             : 8;
__REG32 NTFI              : 3;
__REG32                   : 1;
__REG32 NTFO              : 3;
__REG32                   :17;
} __esai_tfsr_bits;

/* Receive FIFO Configuration Register (ESAI.RFCR) */
typedef struct {
__REG32 RFEN              : 1;
__REG32 RFR               : 1;
__REG32 RE0               : 1;
__REG32 RE1               : 1;
__REG32 RE2               : 1;
__REG32 RE3               : 1;
__REG32                   : 2;
__REG32 RFWM              : 8;
__REG32 RWA               : 3;
__REG32 REXT              : 1;
__REG32                   :12;
} __esai_rfcr_bits;

/* Receive FIFO Status Register (ESAI.RFSR) */
typedef struct {
__REG32 RFCNT             : 8;
__REG32 NRFO              : 2;
__REG32                   : 2;
__REG32 NRFI              : 2;
__REG32                   :18;
} __esai_rfsr_bits;

/* ESAI Transmit Data Register (ESAI.TX0) */
typedef struct {
__REG32 TX0               :24;
__REG32                   : 8;
} __esai_tx0_bits;

/* ESAI Transmit Data Register (ESAI.TX1) */
typedef struct {
__REG32 TX1               :24;
__REG32                   : 8;
} __esai_tx1_bits;

/* ESAI Transmit Data Register (ESAI.TX2) */
typedef struct {
__REG32 TX2               :24;
__REG32                   : 8;
} __esai_tx2_bits;

/* ESAI Transmit Data Register (ESAI.TX3) */
typedef struct {
__REG32 TX3               :24;
__REG32                   : 8;
} __esai_tx3_bits;

/* ESAI Transmit Data Register (ESAI.TX4) */
typedef struct {
__REG32 TX4               :24;
__REG32                   : 8;
} __esai_tx4_bits;

/* ESAI Transmit Data Register (ESAI.TX5) */
typedef struct {
__REG32 TX5               :24;
__REG32                   : 8;
} __esai_tx5_bits;

/* ESAI Transmit Slot Register (ESAI.TSR) */
typedef struct {
__REG32 TSR               :24;
__REG32                   : 8;
} __esai_tsr_bits;

/* ESAI Receive Data Register (ESAI.RX0) */
typedef struct {
__REG32 RX0               :24;
__REG32                   : 8;
} __esai_rx0_bits;

/* ESAI Receive Data Register (ESAI.RX1) */
typedef struct {
__REG32 RX1               :24;
__REG32                   : 8;
} __esai_rx1_bits;

/* ESAI Receive Data Register (ESAI.RX2) */
typedef struct {
__REG32 RX2               :24;
__REG32                   : 8;
} __esai_rx2_bits;

/* ESAI Receive Data Register (ESAI.RX3) */
typedef struct {
__REG32 RX3               :24;
__REG32                   : 8;
} __esai_rx3_bits;

/* ESAI Status Register (ESAI.SAISR) */
typedef struct {
__REG32 IF0               : 1;
__REG32 IF1               : 1;
__REG32 IF2               : 1;
__REG32                   : 3;
__REG32 RFS               : 1;
__REG32 ROE               : 1;
__REG32 RDF               : 1;
__REG32 REDF              : 1;
__REG32 RODF              : 1;
__REG32                   : 2;
__REG32 TFS               : 1;
__REG32 TUE               : 1;
__REG32 TDE               : 1;
__REG32 TEDE              : 1;
__REG32 TODFE             : 1;
__REG32                   :14;
} __esai_saisr_bits;

/* ESAI Common Control Register (ESAI.SAICR) */
typedef struct {
__REG32 OF0               : 1;
__REG32 OF1               : 1;
__REG32 OF2               : 1;
__REG32                   : 3;
__REG32 SYN               : 1;
__REG32 TEBE              : 1;
__REG32 ALC               : 1;
__REG32                   :23;
} __esai_saicr_bits;

/* ESAI Transmit Control Register (ESAI.TCR) */
typedef struct {
__REG32 TE0               : 1;
__REG32 TE1               : 1;
__REG32 TE2               : 1;
__REG32 TE3               : 1;
__REG32 TE4               : 1;
__REG32 TE5               : 1;
__REG32 TSHFD             : 1;
__REG32 TWA               : 1;
__REG32 TMOD              : 2;
__REG32 TSWS              : 5;
__REG32 TFSL              : 1;
__REG32 TFSR              : 1;
__REG32 PADC              : 1;
__REG32                   : 1;
__REG32 TPR               : 1;
__REG32 TEIE              : 1;
__REG32 TDEIE             : 1;
__REG32 TIE               : 1;
__REG32 TLIE              : 1;
__REG32                   : 8;
} __esai_tcr_bits;

/* ESAI Transmitter Clock Control Register (ESAI.TCCR) */
typedef struct {
__REG32 TPM               : 8;
__REG32 TPSR              : 1;
__REG32 TDC               : 5;
__REG32 TFP               : 4;
__REG32 TCKP              : 1;
__REG32 TFSP              : 1;
__REG32 THCKP             : 1;
__REG32 TCKD              : 1;
__REG32 TFSD              : 1;
__REG32 THCKD             : 1;
__REG32                   : 8;
} __esai_tccr_bits;

/* ESAI Receive Control Register (ESAI.RCR) */
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

/* ESAI Receiver Clock Control Register (ESAI.RCCR) */
typedef struct {
__REG32 RPM               : 8;
__REG32 RPSR              : 1;
__REG32 RDC               : 5;
__REG32 RFP               : 4;
__REG32 RCKP              : 1;
__REG32 RFSP              : 1;
__REG32 RHCKP             : 1;
__REG32 RCKD              : 1;
__REG32 RFSD              : 1;
__REG32 RHCKD             : 1;
__REG32                   : 8;
} __esai_rccr_bits;

/* ESAI Transmit Slot Mask Register A (ESAI.TSMA) */
/* ESAI Transmit Slot Mask Register B (ESAI.TSMB) */
typedef struct {
__REG32 TS                :16;
__REG32                   :16;
} __esai_tsm_bits;

/* ESAI Receive Slot Mask Register A (ESAI.RSMA) */
/* ESAI Receive Slot Mask Register B (ESAI.RSMB) */
typedef struct {
__REG32 RS                :16;
__REG32                   :16;
} __esai_rsm_bits;

/* Port C Direction Register (ESAI.PRRC) */
typedef struct {
__REG32 PDC               :12;
__REG32                   :20;
} __esai_prrc_bits;

/* Port C Control Register (ESAI.PCRC) */
typedef struct {
__REG32 PC                :12;
__REG32                   :20;
} __esai_pcrc_bits;

/* ESDCTL Control Register */
typedef struct {
__REG32                   :16;
__REG32 DSIZ              : 1;
__REG32                   : 2;
__REG32 BL                : 1;
__REG32 COL               : 3;
__REG32                   : 1;
__REG32 ROW               : 3;
__REG32                   : 3;
__REG32 SDE_1             : 1;
__REG32 SDE_0             : 1;
} __esdctl_bits;

/* ESDCTL Power Down Control Register */
typedef struct {
__REG32 tCKSRE            : 3;
__REG32 tCKSRX            : 3;
__REG32 BOTH_CS_PD        : 1;
__REG32 SLOW_PD           : 1;
__REG32 PWDT_0            : 4;
__REG32 PWDT_1            : 4;
__REG32 tCKE              : 3;
__REG32                   : 5;
__REG32 PRCT_0            : 3;
__REG32                   : 1;
__REG32 PRCT_1            : 3;
__REG32                   : 1;
} __esdctl_esdpdc_bits;

/* ESDCTL ODT Timing Control Register */
typedef struct {
__REG32                   : 4;
__REG32 tODT_off_idle     : 5;
__REG32                   : 3;
__REG32 tODTLon           : 3;
__REG32                   : 1;
__REG32 tAXPD             : 4;
__REG32 tANPD             : 4;
__REG32 tAONPD            : 3;
__REG32 tAOFPD            : 3;
__REG32                   : 2;
} __esdctl_esdotc_bits;

/* ESDCTL Timing Configuration Register 0 */
typedef struct {
__REG32 tCL               : 4;
__REG32 tFAW              : 5;
__REG32 tXPDLL            : 4;
__REG32 tXS               :11;
__REG32 tRFC              : 8;
} __esdctl_esdcfg0_bits;

/* ESDCTL Timing Configuration Register 1 */
typedef struct {
__REG32 tCWL              : 3;
__REG32                   : 2;
__REG32 tMRD              : 4;
__REG32 tWR               : 3;
__REG32                   : 3;
__REG32 tRPA              : 1;
__REG32 tRAS              : 5;
__REG32 tRC               : 5;
__REG32 tRP               : 3;
__REG32 tRCD              : 3;
} __esdctl_esdcfg1_bits;

/* ESDCTL Timing Configuration Register 2 */
typedef struct {
__REG32 tRRD              : 3;
__REG32 tWTR              : 3;
__REG32 tRTP              : 3;
__REG32                   : 7;
__REG32 tDLLK             : 9;
__REG32                   : 7;
} __esdctl_esdcfg2_bits;

/* ESDCTL Miscellaneous Register */
typedef struct {
__REG32                   : 1;
__REG32 RST               : 1;
__REG32                   : 1;
__REG32 DDR_TYPE          : 2;
__REG32 DDR_4_BANK        : 1;
__REG32 RALAT             : 3;
__REG32 MIF3_MODE         : 2;
__REG32 LPDDR2_S2         : 1;
__REG32 BI                : 1;
__REG32                   : 3;
__REG32 WALAT             : 2;
__REG32 LHD               : 1;
__REG32 ADDR_MIRROR       : 1;
__REG32 ONE_CS            : 1;
__REG32                   : 9;
__REG32 CS1_RDY           : 1;
__REG32 CS0_RDY           : 1;
} __esdmisc_bits;

/* SDRAM Special Command Register */
typedef struct {
__REG32 CMD_BA              : 3;
__REG32 CMD_CS              : 1;
__REG32 CMD                 : 3;
__REG32                     : 2;
__REG32 WL_EN               : 1;
__REG32 MRR_READ_DATA_VALID : 1;
__REG32                     : 3;
__REG32 CON_ACK             : 1;
__REG32 CON_REQ             : 1;
__REG32 MR_ADDR             : 8;
__REG32 MR_OP               : 8;
} __esdctl_esdscr_bits;

/* Refresh Control Register */
typedef struct {
__REG32 START_REF           : 1;
__REG32                     :10;
__REG32 REFR                : 3;
__REG32 REF_SEL             : 2;
__REG32 REF_CNT             :16;
} __esdctl_esdref_bits;

/* Write Command Counter Register */
typedef struct {
__REG32 WR_CMD_CNT          :30;
__REG32 WCC_RST             : 1;
__REG32 WCC_EN              : 1;
} __esdctl_esdwcc_bits;

/* Read Command Counter Register */
typedef struct {
__REG32 RD_CMD_CNT          :30;
__REG32 RCC_RST             : 1;
__REG32 RCC_EN              : 1;
} __esdctl_esdrcc_bits;

/* Read / WRITE Command Delay */
typedef struct {
__REG32 RTR_DIFF            : 3;
__REG32 RTW_DIFF            : 3;
__REG32 WTW_DIFF            : 3;
__REG32 WTR_DIFF            : 3;
__REG32 RTW_SAME            : 3;
__REG32                     : 1;
__REG32 tDAI                :13;
__REG32                     : 3;
} __esdctl_esdrwd_bits;

/* Out of Reset Delays */
typedef struct {
__REG32 RST_to_CKE          : 6;
__REG32                     : 2;
__REG32 SDE_to_RST          : 6;
__REG32                     : 2;
__REG32 tXPR                : 8;
__REG32                     : 8;
} __esdctl_esdor_bits;

/* MRR DATA REGISTER */
typedef struct {
__REG32 MRR_READ_DATA0      : 8;
__REG32 MRR_READ_DATA1      : 8;
__REG32 MRR_READ_DATA2      : 8;
__REG32 MRR_READ_DATA3      : 8;
} __esdctl_esdmrr_bits;

/* ESDCTL timing configuration 3 register (ESDCTL.ESDCFG3_LP) */
typedef struct {
__REG32 tRPab_LP            : 4;
__REG32 tRPpb_LP            : 4;
__REG32 tRCD_LP             : 4;
__REG32                     : 4;
__REG32 tRC_LP              : 6;
__REG32                     :10;
} __esdctl_esdcfg3_lp_bits;

/* ESDCTL MR4 derating register */
typedef struct {
__REG32 UPDATE_DE_REQ       : 1;
__REG32 UPDATE_DE_ACK       : 1;
__REG32                     : 2;
__REG32 tRCD_DE             : 1;
__REG32 tRC_DE              : 1;
__REG32 tRAS_DE             : 1;
__REG32 tRP_DE              : 1;
__REG32 tRRD_DE             : 1;
__REG32                     :23;
} __esdctl_esdmr4_bits;

/* ZQ HW control register */
typedef struct {
__REG32 ZQ_MODE             : 2;
__REG32 ZQ_HW_PER           : 4;
__REG32 ZQ_HW_PU_RES        : 5;
__REG32 ZQ_HW_PD_RES        : 5;
__REG32 ZQ_HW_FOR           : 1;
__REG32 TZQ_INIT            : 3;
__REG32 TZQ_OPER            : 3;
__REG32 TZQ_CS              : 3;
__REG32 ZQ_PARA_EN          : 1;
__REG32                     : 5;
} __zqhwctrl_bits;

/* ZQ SW control register */
typedef struct {
__REG32 ZQ_SW_FOR           : 1;
__REG32 ZQ_SW_VAL           : 1;
__REG32 ZQ_SW_PU_VAL        : 5;
__REG32 ZQ_SW_PD_VAL        : 5;
__REG32 ZQ_SW_PD            : 1;
__REG32 USE_ZQ_SW_VAL       : 1;
__REG32                     :18;
} __esdctl_zqswctrl_bits;

/* Write Leveling general control register */
typedef struct {
__REG32 HW_WL_EN            : 1;
__REG32 SW_WL_EN            : 1;
__REG32 SW_WL_CNT_EN        : 1;
__REG32                     : 1;
__REG32 WL_SW_RES0          : 1;
__REG32 WL_SW_RES1          : 1;
__REG32 WL_SW_RES2          : 1;
__REG32 WL_SW_RES3          : 1;
__REG32 WL_HW_ERR0          : 1;
__REG32 WL_HW_ERR1          : 1;
__REG32 WL_HW_ERR2          : 1;
__REG32 WL_HW_ERR3          : 1;
__REG32                     :20;
} __esdctl_wlgcr_bits;

/* Write Leveling delay control register0 */
typedef struct {
__REG32 WL_DL_ABS_OFFSET0   : 7;
__REG32                     : 1;
__REG32 WL_HC_DEL0          : 1;
__REG32 WL_CYC_DEL0         : 2;
__REG32                     : 5;
__REG32 WR_DL_ABS_OFFSET1   : 7;
__REG32                     : 1;
__REG32 WL_HC_DEL1          : 1;
__REG32 WL_CYC_DEL1         : 2;
__REG32                     : 5;
} __esdctl_wldectrl0_bits;

/* Write Leveling delay control register1 */
typedef struct {
__REG32 WL_DL_ABS_OFFSET2   : 7;
__REG32                     : 1;
__REG32 WL_HC_DEL2          : 1;
__REG32 WL_CYC_DEL2         : 2;
__REG32                     : 5;
__REG32 WR_DL_ABS_OFFSET3   : 7;
__REG32                     : 1;
__REG32 WL_HC_DEL3          : 1;
__REG32 WL_CYC_DEL3         : 2;
__REG32                     : 5;
} __esdctl_wldectrl1_bits;

/* Write leveling Delay Line Status Register */
typedef struct {
__REG32 WL_DL_UNIT_NUM0     : 7;
__REG32                     : 1;
__REG32 WL_DL_UNIT_NUM1     : 7;
__REG32                     : 1;
__REG32 WL_DL_UNIT_NUM2     : 7;
__REG32                     : 1;
__REG32 WL_DL_UNIT_NUM3     : 7;
__REG32                     : 1;
} __esdctl_wldlst_bits;

/* ODT control register */
typedef struct {
__REG32 ODT_WR_PAS_EN       : 1;
__REG32 ODT_WR_ACT_EN       : 1;
__REG32 ODT_RD_PAS_EN       : 1;
__REG32 ODT_RD_ACT_EN       : 1;
__REG32 ODT0_INT_RES        : 3;
__REG32                     : 1;
__REG32 ODT1_INT_RES        : 3;
__REG32                     : 1;
__REG32 ODT2_INT_RES        : 3;
__REG32                     : 1;
__REG32 ODT3_INT_RES        : 3;
__REG32                     :13;
} __esdctl_odtctrl_bits;

/* Read DQ BYTE0 Delay Register */
typedef struct {
__REG32 RD_DQ0_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ1_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ2_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ3_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ4_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ5_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ6_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ7_DEL          : 3;
__REG32                     : 1;
} __esdctl_rddqby0dl_bits;

/* Read DQ BYTE1 Delay Register */
typedef struct {
__REG32 RD_DQ8_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ9_DEL          : 3;
__REG32                     : 1;
__REG32 RD_DQ10_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ11_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ12_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ13_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ14_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ15_DEL         : 3;
__REG32                     : 1;
} __esdctl_rddqby1dl_bits;

/* Read DQ BYTE2 Delay Register */
typedef struct {
__REG32 RD_DQ16_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ17_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ18_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ19_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ20_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ21_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ22_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ23_DEL         : 3;
__REG32                     : 1;
} __esdctl_rddqby2dl_bits;

/* Read DQ BYTE3 Delay Register */
typedef struct {
__REG32 RD_DQ24_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ25_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ26_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ27_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ28_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ29_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ30_DEL         : 3;
__REG32                     : 1;
__REG32 RD_DQ31_DEL         : 3;
__REG32                     : 1;
} __esdctl_rddqby3dl_bits;

/* Write DQ BYTE0 Delay Register */
typedef struct {
__REG32 wr_dq0_del          : 2;
__REG32                     : 2;
__REG32 wr_dq1_del          : 2;
__REG32                     : 2;
__REG32 wr_dq2_del          : 2;
__REG32                     : 2;
__REG32 wr_dq3_del          : 2;
__REG32                     : 2;
__REG32 wr_dq4_del          : 2;
__REG32                     : 2;
__REG32 wr_dq5_del          : 2;
__REG32                     : 2;
__REG32 wr_dq6_del          : 2;
__REG32                     : 2;
__REG32 wr_dq7_del          : 2;
__REG32 wr_dm0_del          : 2;
} __esdctl_wrdqby0dl_bits;

/* Write DQ BYTE1 Delay Register */
typedef struct {
__REG32 wr_dq8_del          : 2;
__REG32                     : 2;
__REG32 wr_dq9_del          : 2;
__REG32                     : 2;
__REG32 wr_dq10_del         : 2;
__REG32                     : 2;
__REG32 wr_dq11_del         : 2;
__REG32                     : 2;
__REG32 wr_dq12_del         : 2;
__REG32                     : 2;
__REG32 wr_dq13_del         : 2;
__REG32                     : 2;
__REG32 wr_dq14_del         : 2;
__REG32                     : 2;
__REG32 wr_dq15_del         : 2;
__REG32 wr_dm1_del          : 2;
} __esdctl_wrdqby1dl_bits;

/* Write DQ BYTE2 Delay Register */
typedef struct {
__REG32 wr_dq16_del         : 2;
__REG32                     : 2;
__REG32 wr_dq17_del         : 2;
__REG32                     : 2;
__REG32 wr_dq18_del         : 2;
__REG32                     : 2;
__REG32 wr_dq19_del         : 2;
__REG32                     : 2;
__REG32 wr_dq20_del         : 2;
__REG32                     : 2;
__REG32 wr_dq21_del         : 2;
__REG32                     : 2;
__REG32 wr_dq22_del         : 2;
__REG32                     : 2;
__REG32 wr_dq23_del         : 2;
__REG32 wr_dm2_del          : 2;
} __esdctl_wrdqby2dl_bits;

/* Write DQ BYTE3 Delay Register */
typedef struct {
__REG32 wr_dq24_del         : 2;
__REG32                     : 2;
__REG32 wr_dq25_del         : 2;
__REG32                     : 2;
__REG32 wr_dq26_del         : 2;
__REG32                     : 2;
__REG32 wr_dq27_del         : 2;
__REG32                     : 2;
__REG32 wr_dq28_del         : 2;
__REG32                     : 2;
__REG32 wr_dq29_del         : 2;
__REG32                     : 2;
__REG32 wr_dq30_del         : 2;
__REG32                     : 2;
__REG32 wr_dq31_del         : 2;
__REG32 wr_dm3_del          : 2;
} __esdctl_wrdqby3dl_bits;

/* DQS gating control register0 */
typedef struct {
__REG32 DG_DL_ABS_OFFSET0   : 7;
__REG32                     : 1;
__REG32 DG_HC_DEL0          : 4;
__REG32 HW_DG_ERR           : 1;
__REG32                     : 3;
__REG32 DG_DL_ABS_OFFSET1   : 7;
__REG32 DG_EXT_UP           : 1;
__REG32 DG_HC_DEL1          : 4;
__REG32 HW_DG_EN            : 1;
__REG32 DG_DIS              : 1;
__REG32 DG_CMP_CYC          : 1;
__REG32 RST_RD_FIFO         : 1;
} __esdctl_dgctrl0_bits;

/* DQS Gating Control Register1 */
typedef struct {
__REG32 DG_DL_ABS_OFFSET2   : 7;
__REG32                     : 1;
__REG32 DG_HC_DEL2          : 4;
__REG32                     : 4;
__REG32 DG_DL_ABS_OFFSET3   : 7;
__REG32                     : 1;
__REG32 DG_HC_DEL3          : 4;
__REG32                     : 4;
} __esdctl_dgctrl1_bits;

/* DQS Gating Delay Line Status Register */
typedef struct {
__REG32 DG_DL_UNIT_NUM0     : 7;
__REG32                     : 1;
__REG32 DG_DL_UNIT_NUM1     : 7;
__REG32                     : 1;
__REG32 DG_DL_UNIT_NUM2     : 7;
__REG32                     : 1;
__REG32 DG_DL_UNIT_NUM3     : 7;
__REG32                     : 1;
} __esdctl_dgdlst_bits;

/* Read Delay Lines Configuration Register */
typedef struct {
__REG32 RD_DL_ABS_OFFSET0   : 7;
__REG32                     : 1;
__REG32 RD_DL_ABS_OFFSET1   : 7;
__REG32                     : 1;
__REG32 RD_DL_ABS_OFFSET2   : 7;
__REG32                     : 1;
__REG32 RD_DL_ABS_OFFSET3   : 7;
__REG32                     : 1;
} __esdctl_rddlctl_bits;

/* Read Delay Line Status Register */
typedef struct {
__REG32 RD_DL_UNIT_NUM0     : 7;
__REG32                     : 1;
__REG32 RD_DL_UNIT_NUM1     : 7;
__REG32                     : 1;
__REG32 RD_DL_UNIT_NUM2     : 7;
__REG32                     : 1;
__REG32 RD_DL_UNIT_NUM3     : 7;
__REG32                     : 1;
} __esdctl_rddlst_bits;

/* Write Delay Lines Configuration Register */
typedef struct {
__REG32 WR_DL_ABS_OFFSET0   : 7;
__REG32                     : 1;
__REG32 WR_DL_ABS_OFFSET1   : 7;
__REG32                     : 1;
__REG32 WR_DL_ABS_OFFSET2   : 7;
__REG32                     : 1;
__REG32 WR_DL_ABS_OFFSET3   : 7;
__REG32                     : 1;
} __esdctl_wrdlctl_bits;

/* Write Delay Line Status Register */
typedef struct {
__REG32 WR_DL_UNIT_NUM0     : 7;
__REG32                     : 1;
__REG32 WR_DL_UNIT_NUM1     : 7;
__REG32                     : 1;
__REG32 WR_DL_UNIT_NUM2     : 7;
__REG32                     : 1;
__REG32 WR_DL_UNIT_NUM3     : 7;
__REG32                     : 1;
} __esdctl_wrdlst_bits;

/* SDCLK Control Register */
typedef struct {
__REG32                     : 8;
__REG32 SDCLK0_DEL          : 2;
__REG32 SDCLK1_DEL          : 2;
__REG32                     :20;
} __esdctl_sdctrl_bits;

/* ZQ LPDDR2 HW CTRL REGISTER */
typedef struct {
__REG32 ZQ_LP2_HW_ZQINIT    : 9;
__REG32                     : 7;
__REG32 ZQ_LP2_HW_ZQCL      : 8;
__REG32 ZQ_LP2_HW_ZQCS      : 7;
__REG32                     : 1;
} __esdctl_zqlp2ctl_bits;

/* RD DL HW Calibration Control Register */
typedef struct {
__REG32 HW_RDL_ERR0         : 1;
__REG32 HW_RDL_ERR1         : 1;
__REG32 HW_RDL_ERR2         : 1;
__REG32 HW_RDL_ERR3         : 1;
__REG32 HW_RDL_ERR4         : 1;
__REG32 HW_RDL_CMP_CYC      : 1;
__REG32                     :26;
} __esdctl_rddlhwctl_bits;

/* WR DL HW Calibration Control Register */
typedef struct {
__REG32 HW_WDL_ERR0         : 1;
__REG32 HW_WDL_ERR1         : 1;
__REG32 HW_WDL_ERR2         : 1;
__REG32 HW_WDL_ERR3         : 1;
__REG32 HW_WDL_ERR4         : 1;
__REG32 HW_WDL_CMP_CYC      : 1;
__REG32                     :26;
} __esdctl_wrdlhwctl_bits;

/* RD DL HW Calibration Status Register0 */
typedef struct {
__REG32 HW_RD_DL_LOW0       : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_UP0        : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_LOW1       : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_UP1        : 7;
__REG32                     : 1;
} __esdctl_rddlhwst0_bits;

/* RD DL HW Calibration Status Register1 */
typedef struct {
__REG32 HW_RD_DL_LOW2       : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_UP2        : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_LOW3       : 7;
__REG32                     : 1;
__REG32 HW_RD_DL_UP3        : 7;
__REG32                     : 1;
} __esdctl_rddlhwst1_bits;

/* WR DL HW Calibration Status Register0 */
typedef struct {
__REG32 HW_WR_DL_LOW0       : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_UP0        : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_LOW1       : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_UP1        : 7;
__REG32                     : 1;
} __esdctl_wrdlhwst0_bits;

/* WR DL HW Calibration Status Register1 */
typedef struct {
__REG32 HW_WR_DL_LOW2       : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_UP2        : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_LOW3       : 7;
__REG32                     : 1;
__REG32 HW_WR_DL_UP3        : 7;
__REG32                     : 1;
} __esdctl_wrdlhwst1_bits;

/* WL HW Error Register */
typedef struct {
__REG32 HW_WL0_DQ           : 8;
__REG32 HW_WL1_DQ           : 8;
__REG32 HW_WL2_DQ           : 8;
__REG32 HW_WL3_DQ           : 8;
} __esdctl_wlhwerr_bits;

/* DG HW Status Register0 */
typedef struct {
__REG32 HW_DG_LOW0          :11;
__REG32                     : 5;
__REG32 HW_DG_UP0           :11;
__REG32                     : 5;
} __esdctl_dghwst0_bits;

/* DG HW Status Register1 */
typedef struct {
__REG32 HW_DG_LOW1          :11;
__REG32                     : 5;
__REG32 HW_DG_UP1           :11;
__REG32                     : 5;
} __esdctl_dghwst1_bits;

/* DG HW Status Register2 */
typedef struct {
__REG32 HW_DG_LOW2          :11;
__REG32                     : 5;
__REG32 HW_DG_UP2           :11;
__REG32                     : 5;
} __esdctl_dghwst2_bits;

/* DG HW Status Register3 */
typedef struct {
__REG32 HW_DG_LOW2          :11;
__REG32                     : 5;
__REG32 HW_DG_UP2           :11;
__REG32                     : 5;
} __esdctl_dghwst3_bits;

/* Pre-defined Compare Register1 */
typedef struct {
__REG32 PDV1                :16;
__REG32 PDV2                :16;
} __esdctl_pdcmpr1_bits;

/* Pre-defined Compare Register1 */
typedef struct {
__REG32 MPR_CMP             : 1;
__REG32 MPR_FULL_CMP        : 1;
__REG32 READ_LEVEL_PATTERN  : 1;
__REG32                     :13;
__REG32 CA_DL_ABS_OFFSET    : 7;
__REG32                     : 1;
__REG32 PHY_CA_DL_UNIT      : 7;
__REG32                     : 1;
} __esdctl_pdcmpr2_bits;

/* SW dummy Access Register */
typedef struct {
__REG32 SW_DUMMY_WR         : 1;
__REG32 SW_DUMMY_RD         : 1;
__REG32 SW_DUM_CMP0         : 1;
__REG32 SW_DUM_CMP1         : 1;
__REG32 SW_DUM_CMP2         : 1;
__REG32 SW_DUM_CMP3         : 1;
__REG32                     :26;
} __esdctl_swdar_bits;

/* Measure unit Register1 */
typedef struct {
__REG32 MU_BYP_VAL          :10;
__REG32 MU_BYP_EN           : 1;
__REG32 FRC_MSR             : 1;
__REG32                     : 4;
__REG32 MU_UNIT_DEL_NUM     :10;
__REG32                     : 6;
} __esdctl_mur_bits;

/* Write CA Delay Line Controller(ESDCTL.WRCADL) */
typedef struct {
__REG32 WR_CA0_DEL          : 2;
__REG32 WR_CA1_DEL          : 2;
__REG32 WR_CA2_DEL          : 2;
__REG32 WR_CA3_DEL          : 2;
__REG32 WR_CA4_DEL          : 2;
__REG32 WR_CA5_DEL          : 2;
__REG32 WR_CA6_DEL          : 2;
__REG32 WR_CA7_DEL          : 2;
__REG32 WR_CA8_DEL          : 2;
__REG32 WR_CA9_DEL          : 2;
__REG32                     :12;
} __esdctl_wrcadl_bits;

/* Block Attributes Register (ESDHCV2.BLKATTR) */
typedef struct {
__REG32 BLKSZE              :13;
__REG32                     : 3;
__REG32 BLKCNT              :16;
} __esdhc_blkattr_bits;

/* Transfer Type Register (ESDHCV2.XFERTYP) */
typedef struct {
__REG32 DMAEN               : 1;
__REG32 BCEN                : 1;
__REG32 AC12EN              : 1;
__REG32                     : 1;
__REG32 DTDSEL              : 1;
__REG32 MSBSEL              : 1;
__REG32                     :10;
__REG32 RSPTYP              : 2;
__REG32                     : 1;
__REG32 CCCEN               : 1;
__REG32 CICEN               : 1;
__REG32 DPSEL               : 1;
__REG32 CMDTYP              : 2;
__REG32 CMDINX              : 6;
__REG32                     : 2;
} __esdhc_xfertyp_bits;

/* Transfer Type Register (ESDHCV3.XFERTYP) */
typedef struct {
__REG32 DMAEN               : 1;
__REG32 BCEN                : 1;
__REG32 AC12EN              : 1;
__REG32 DDR_EN              : 1;
__REG32 DTDSEL              : 1;
__REG32 MSBSEL              : 1;
__REG32 NIBBLE_POS          : 1;
__REG32                     : 9;
__REG32 RSPTYP              : 2;
__REG32                     : 1;
__REG32 CCCEN               : 1;
__REG32 CICEN               : 1;
__REG32 DPSEL               : 1;
__REG32 CMDTYP              : 2;
__REG32 CMDINX              : 6;
__REG32                     : 2;
} __esdhcv3_xfertyp_bits;

/* Present State Register (ESDHCV2.PRSSTAT) */
typedef struct {
__REG32 CIHB                : 1;
__REG32 CDIHB               : 1;
__REG32 DLA                 : 1;
__REG32 SDSTB               : 1;
__REG32 IPGOFF              : 1;
__REG32 HCKOFF              : 1;
__REG32 PEROFF              : 1;
__REG32 SDOFF               : 1;
__REG32 WTA                 : 1;
__REG32 RTA                 : 1;
__REG32 BWEN                : 1;
__REG32 BREN                : 1;
__REG32                     : 4;
__REG32 CINS                : 1;
__REG32                     : 1;
__REG32 CDPL                : 1;
__REG32 WPSPL               : 1;
__REG32                     : 3;
__REG32 CLSL                : 1;
__REG32 DLSL                : 8;
} __esdhc_prsstat_bits;

/* Protocol Control Register (ESDHCV2.PROCTL) */
typedef struct {
__REG32 LCTL                : 1;
__REG32 DTW                 : 2;
__REG32 D3CD                : 1;
__REG32 EMODE               : 2;
__REG32 CDTL                : 1;
__REG32 CDSS                : 1;
__REG32 DMAS                : 2;
__REG32                     : 6;
__REG32 SABGREQ             : 1;
__REG32 CREQ                : 1;
__REG32 RWCTL               : 1;
__REG32 IABG                : 1;
__REG32                     : 4;
__REG32 WECINT              : 1;
__REG32 WECINS              : 1;
__REG32 WECRM               : 1;
__REG32                     : 5;
} __esdhc_proctl_bits;

/* System Control Register (ESDHCV2.SYSCTL) */
typedef struct {
__REG32 IPGEN               : 1;
__REG32 HCKEN               : 1;
__REG32 PEREN               : 1;
__REG32 SDCLKEN             : 1;
__REG32 DVS                 : 4;
__REG32 SDCLKFS             : 8;
__REG32 DTOCV               : 4;
__REG32                     : 4;
__REG32 RSTA                : 1;
__REG32 RSTC                : 1;
__REG32 RSTD                : 1;
__REG32 INITA               : 1;
__REG32                     : 4;
} __esdhc_sysctl_bits;

/* System Control Register (ESDHCV3.SYSCTL) */
typedef struct {
__REG32 IPGEN               : 1;
__REG32 HCKEN               : 1;
__REG32 PEREN               : 1;
__REG32 SDCLKEN             : 1;
__REG32 DVS                 : 4;
__REG32 SDCLKFS             : 8;
__REG32 DTOCV               : 4;
__REG32                     : 3;
__REG32 IPP_RST_N           : 1;
__REG32 RSTA                : 1;
__REG32 RSTC                : 1;
__REG32 RSTD                : 1;
__REG32 INITA               : 1;
__REG32                     : 4;
} __esdhcv3_sysctl_bits;

/* Interrupt Status Register (ESDHCV2.IRQSTAT) */
typedef struct {
__REG32 CC                  : 1;
__REG32 TC                  : 1;
__REG32 BGE                 : 1;
__REG32 DINT                : 1;
__REG32 BWR                 : 1;
__REG32 BRR                 : 1;
__REG32 CINS                : 1;
__REG32 CRM                 : 1;
__REG32 CINT                : 1;
__REG32                     : 7;
__REG32 CTOE                : 1;
__REG32 CCE                 : 1;
__REG32 CEBE                : 1;
__REG32 CIE                 : 1;
__REG32 DTOE                : 1;
__REG32 DCE                 : 1;
__REG32 DEBE                : 1;
__REG32                     : 1;
__REG32 AC12E               : 1;
__REG32                     : 3;
__REG32 DMAE                : 1;
__REG32                     : 3;
} __esdhc_irqstat_bits;

/* Interrupt Status Enable Register (ESDHCV2.IRQSTATEN) */
typedef struct {
__REG32 CCSEN               : 1;
__REG32 TCSEN               : 1;
__REG32 BGESEN              : 1;
__REG32 DINTSEN             : 1;
__REG32 BWRSEN              : 1;
__REG32 BRRSEN              : 1;
__REG32 CINSEN              : 1;
__REG32 CRMSEN              : 1;
__REG32 CINTSEN             : 1;
__REG32                     : 7;
__REG32 CTOESEN             : 1;
__REG32 CCESEN              : 1;
__REG32 CEBESEN             : 1;
__REG32 CIESEN              : 1;
__REG32 DTOESEN             : 1;
__REG32 DCESEN              : 1;
__REG32 DEBESEN             : 1;
__REG32                     : 1;
__REG32 AC12ESEN            : 1;
__REG32                     : 3;
__REG32 DMAESEN             : 1;
__REG32                     : 3;
} __esdhc_irqstaten_bits;

/* Interrupt Signal Enable Register (ESDHCV2.IRQSIGEN) */
typedef struct {
__REG32 CCIEN               : 1;
__REG32 TCIEN               : 1;
__REG32 BGEIEN              : 1;
__REG32 DINTIEN             : 1;
__REG32 BWRIEN              : 1;
__REG32 BRRIEN              : 1;
__REG32 CINIEN              : 1;
__REG32 CRMIEN              : 1;
__REG32 CINTIEN             : 1;
__REG32                     : 7;
__REG32 CTOEIEN             : 1;
__REG32 CCEIEN              : 1;
__REG32 CEBEIEN             : 1;
__REG32 CIEIEN              : 1;
__REG32 DTOEIEN             : 1;
__REG32 DCEIEN              : 1;
__REG32 DEBEIEN             : 1;
__REG32                     : 1;
__REG32 AC12EIEN            : 1;
__REG32                     : 3;
__REG32 DMAEIEN             : 1;
__REG32                     : 3;
} __esdhc_irqsigen_bits;

/* Auto CMD12 Error Status Register (ESDHCV2.AUTOC12ERR) */
typedef struct {
__REG32 AC12NE              : 1;
__REG32 AC12TOE             : 1;
__REG32 AC12EBE             : 1;
__REG32 AC12CE              : 1;
__REG32 AC12IE              : 1;
__REG32                     : 2;
__REG32 CNIBAC12E           : 1;
__REG32                     :24;
} __esdhc_autoc12err_bits;

/* Host Controller Capabilities (ESDHCV2.HOSTCAPBLT) */
typedef struct {
__REG32                     :16;
__REG32 MBL                 : 3;
__REG32                     : 1;
__REG32 ADMAS               : 1;
__REG32 HSS                 : 1;
__REG32 DMAS                : 1;
__REG32 SRS                 : 1;
__REG32 VS33                : 1;
__REG32 VS30                : 1;
__REG32 VS18                : 1;
__REG32                     : 5;
} __esdhc_hostcapblt_bits;

/* Watermark Level Register (ESDHCV2.WML) */
typedef struct {
__REG32 RD_WML              : 8;
__REG32 RD_BRST_LEN         : 5;
__REG32                     : 3;
__REG32 WR_WML              : 8;
__REG32 WR_BRST_LEN         : 5;
__REG32                     : 3;
} __esdhc_wml_bits;

/* Force Event Register(ESDHCV2.FEVT) */
typedef struct {
__REG32 FEVTAC12NE          : 1;
__REG32 FEVTAC12TOE         : 1;
__REG32 FEVTAC12CE          : 1;
__REG32 FEVTAC12EBE         : 1;
__REG32 FEVTAC12IE          : 1;
__REG32                     : 2;
__REG32 FEVTCNIBAC12E       : 1;
__REG32                     : 8;
__REG32 FEVTCTOE            : 1;
__REG32 FEVTCCE             : 1;
__REG32 FEVTCEBE            : 1;
__REG32 FEVTCIE             : 1;
__REG32 FEVTDTOE            : 1;
__REG32 FEVTDCE             : 1;
__REG32 FEVTDEBE            : 1;
__REG32                     : 1;
__REG32 FEVTAC12E           : 1;
__REG32                     : 3;
__REG32 FEVTDMAE            : 1;
__REG32                     : 2;
__REG32 FEVTCINT            : 1;
} __esdhc_fevt_bits;

/* ADMA Error Status Register (ESDHCV2.ADMAES) */
typedef struct {
__REG32 ADMAES              : 2;
__REG32 ADMALME             : 1;
__REG32 ADMADCE             : 1;
__REG32                     :28;
} __esdhc_admaes_bits;

/* DLL (Delay Line) Control Register (ESDHCV3.DLLCTRL) */
typedef struct {
__REG32 DLL_CTRL_ENABLE           : 1;
__REG32 DLL_CTRL_RESET            : 1;
__REG32 DLL_CTRL_SLV_FORCE_UPD    : 1;
__REG32 DLL_CTRL_SLV_DLY_TARGET   : 4;
__REG32 DLL_CTRL_GATE_UPDATE      : 1;
__REG32                           : 1;
__REG32 DLL_CTRL_SLV_OVERRIDE     : 1;
__REG32 DLL_CTRL_SLV_OVERRIDE_VAL : 6;
__REG32                           : 4;
__REG32 DLL_CTRL_SLV_UPDATE_INT   : 8;
__REG32 DLL_CTRL_REF_UPDATE_INT   : 4;
} __esdhcv3_dllctrl_bits;

/* DLL Status Register (ESDHCV3.DLLSTS) */
typedef struct {
__REG32 DLL_STS_SLV_LOCK          : 1;
__REG32 DLL_STS_REF_LOCK          : 1;
__REG32 DLL_STS_SLV_SEL           : 6;
__REG32 DLL_STS_REF_SEL           : 6;
__REG32                           :18;
} __esdhcv3_dllsts_bits;

/* Vendor Specific Register (ESDHCV2.VENDOR) */
typedef struct {
__REG32 EXT_DMA_EN          : 1;
__REG32 EXACT_BLK_NUM       : 1;
__REG32                     :14;
__REG32 INT_ST_VAL          : 8;
__REG32 DBG_SEL             : 4;
__REG32                     : 4;
} __esdhc_vendor_bits;

/* MMC Boot Register (ESDHCV3.MMCBOOT) */
typedef struct {
__REG32 DTOCV_ACK           : 4;
__REG32 BOOT_ACK            : 1;
__REG32 MMC_BOOT_MODE       : 1;
__REG32 BOOT_EN             : 1;
__REG32 AUTO_SABG_EN        : 1;
__REG32                     : 8;
__REG32 BOOT_BLK_CNT        :16;
} __esdhc_mmcboot_bits;

/* Host Controller Version (ESDHCV2.HOSTVER) */
typedef struct {
__REG32 SVN                 : 8;
__REG32 VVN                 : 8;
__REG32                     :16;
} __esdhc_hostver_bits;

/* IP Lock register (EXTMC.IPLCK) */
typedef struct {
__REG32 NFC_lock            : 1;
__REG32 WEIM_lock           : 1;
__REG32 ESDC_lock           : 1;
__REG32 M4IF_lock           : 1;
__REG32 Lock_all            : 1;
__REG32 XFR_ERR_EN          : 1;
__REG32                     :26;
} __extmc_iplck_bits;

/* Interrupt control & Status register (EXTMC.EICS) */
typedef struct {
__REG32 ANFC_int              : 1;
__REG32 AWEIM_int             : 1;
__REG32 AM4IF_BP_int          : 1;
__REG32 AM4IF_AP_int          : 1;
__REG32 NIS                   : 1;
__REG32 WIS                   : 1;
__REG32 MBIS                  : 1;
__REG32 MAIS                  : 1;
__REG32 BNFC_int              : 1;
__REG32 BWEIM_int             : 1;
__REG32 BM4IF_BP_int          : 1;
__REG32 BM4IF_AP_int          : 1;
__REG32 ALEN_int              : 1;
__REG32 BLEN_int              : 1;
__REG32 AUTO_PROG_DONE_ARM_int: 1;
__REG32 AUTO_PROG_DONE_BP_int : 1;
__REG32                       :14;
__REG32 EBOIE                 : 1;
__REG32 EAOIE                 : 1;
} __extmc_eics_bits;

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
__REG32 WMRK          : 2;
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

/* MIIGSK Configuration Register (MIIGSK_CFGR) */
typedef struct {
__REG32 IF_MODE       : 2;
__REG32               : 2;
__REG32 LBMODE        : 1;
__REG32               : 1;
__REG32 FRCONT        : 1;
__REG32               :25;
} __fec_miigsk_cfgr_bits;

/* MIIGSK Enable Register (MIIGSK_ENR) */
typedef struct {
__REG32               : 1;
__REG32 EN            : 1;
__REG32 READY         : 1;
__REG32               :29;
} __fec_miigsk_enr_bits;

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

/* Glitch Filter Width Register (FLEXCAN.GFWR) */
typedef struct {
__REG32 GFWR  : 8;
__REG32       :24;
} __can_gfwr_bits;

/* CNTR Register Description */
typedef struct {
__REG32 HTRI    : 4;
__REG32         : 9;
__REG32 FUPD    : 1;
__REG32 STRT    : 1;
__REG32 ADU     : 1;
__REG32 DVFS0CR : 1;
__REG32 DVFS1CR : 1;
__REG32         : 2;
__REG32 GPCIRQ  : 1;
__REG32 GPCIRQM : 1;
__REG32         : 2;
__REG32 IRQ2    : 1;
__REG32 IRQ2M   : 1;
__REG32 CSPI    : 1;
__REG32         : 5;
} __gpc_cntr_bits;

/* VCR Register Description */
typedef struct {
__REG32 VCNT    :15;
__REG32         : 1;
__REG32 VCNTU   : 1;
__REG32 VINC    : 1;
__REG32         :14;
} __gpc_vcr_bits;

/* NEON Register Description */
typedef struct {
__REG32 NEONPDR   : 1;
__REG32 NEONPUR   : 1;
__REG32           : 2;
__REG32 NEONFSMST : 2;
__REG32           :26;
} __gpc_neon_bits;

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

/* Software-Controllable Signals Register 1 (SCS1) */
typedef struct{
__REG8  SCS       : 7;
__REG8  LOCK      : 1;
} __iim_scs1_bits;

/* Software-Controllable Signals Register 2 (SCS2) */
typedef struct{
__REG8  FBRL0     : 1;
__REG8  FBRL1     : 1;
__REG8  FBRL2     : 1;
__REG8  FBRL3     : 1;
__REG8  FBRL4     : 1;
__REG8            : 2;
__REG8  LOCK      : 1;
} __iim_scs2_bits;

/* Software-Controllable Signals Register 3 (SCS3) */
typedef struct{
__REG8  FBWL0     : 1;
__REG8  FBWL1     : 1;
__REG8  FBWL2     : 1;
__REG8  FBWL3     : 1;
__REG8  FBWL4     : 1;
__REG8            : 2;
__REG8  LOCK      : 1;
} __iim_scs3_bits;

/* General Purpose Register 0 (IOMUXC.IOMUXC_GPR0) */
typedef struct{
__REG32 DMAREQ_MUX_SEL0   : 1;
__REG32 DMAREQ_MUX_SEL1   : 1;
__REG32 DMAREQ_MUX_SEL2   : 1;
__REG32 DMAREQ_MUX_SEL3   : 1;
__REG32 DMAREQ_MUX_SEL4   : 1;
__REG32 DMAREQ_MUX_SEL5   : 1;
__REG32 DMAREQ_MUX_SEL6   : 1;
__REG32 DMAREQ_MUX_SEL7   : 1;
__REG32 DMAREQ_MUX_SEL8   : 1;
__REG32 DMAREQ_MUX_SEL9   : 1;
__REG32 DMAREQ_MUX_SEL10  : 1;
__REG32 MLBCLK_IN_INV     : 1;
__REG32                   : 2;
__REG32 TX_CLK3_MUX_SEL   : 2;
__REG32 CLOCK_1_MUX_SEL   : 2;
__REG32 CLOCK_9_MUX_SEL   : 2;
__REG32 CLOCK_2_MUX_SEL   : 2;
__REG32 CLOCK_A_MUX_SEL   : 2;
__REG32 CLOCK_3_MUX_SEL   : 2;
__REG32 CLOCK_B_MUX_SEL   : 2;
__REG32 CLOCK_0_MUX_SEL   : 2;
__REG32 CLOCK_8_MUX_SEL   : 2;
} __iomuxc_gpr0_bits;

/* General Purpose Register 1 (IOMUXC.IOMUXC_GPR1) */
typedef struct{
__REG32 ACT_CS0           : 1;
__REG32 ADDRS0            : 2;
__REG32 ACT_CS1           : 1;
__REG32 ADDRS1            : 2;
__REG32 ACT_CS2           : 1;
__REG32 ADDRS2            : 2;
__REG32 ACT_CS3           : 1;
__REG32 ADDRS3            : 2;
__REG32 PLL1P2_VREG       : 4;
__REG32                   : 1;
__REG32 PLL1P8_VREG       : 5;
__REG32                   :10;
} __iomuxc_gpr1_bits;

/* General Purpose Register 2 (IOMUXC.IOMUXC_GPR2) */
typedef struct{
__REG32 CH0_MODE          : 2;
__REG32 CH1_MODE          : 2;
__REG32 SPLIT_MODE_EN     : 1;
__REG32 DATA_WIDTH_CH0    : 1;
__REG32 BIT_MAPPING_CH0   : 1;
__REG32 DATA_WIDTH_CH1    : 1;
__REG32 BIT_MAPPING_CH1   : 1;
__REG32 DI0_VS_POLARITY   : 1;
__REG32 DI1_VS_POLARITY   : 1;
__REG32                   : 4;
__REG32 BGREF_RRMODE      : 1;
__REG32 LVDS_CLK_SHIFT    : 3;
__REG32                   : 1;
__REG32 COUNTER_RESET_VAL : 2;
__REG32                   :10;
} __iomuxc_gpr2_bits;

/* OBSERVE_MUX Register n (IOMUXC.IOMUXC_OBSERVE_MUX_n)*/
typedef struct{
__REG32 OBSRV             : 6;
__REG32                   :26;
} __iomuxc_observe_mux_bits;

/* MUX_CTL Register n (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_GPIO_n)*/
typedef struct{
__REG32 MUX_MODE          : 3;
__REG32                   : 1;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_gpio_bits;

/* IOMUXC.IOMUXC.IOMUXC_SW_MUX_CTL_PAD_EIM_BCLK*/
typedef struct{
__REG32                   : 4;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_eim_bclk_bits;

/* IOMUXC.IOMUXC_SW_MUX_CTL_PAD_LVDS1_TX3_P*/
typedef struct{
__REG32 MUX_MODE          : 1;
__REG32                   :31;
} __iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits;

/* IOMUXC.IOMUXC_SW_MUX_CTL_PAD_GPIO_10 */
typedef struct{
__REG32 MUX_MODE          : 1;
__REG32                   : 3;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_gpio_10_bits;

/* IOMUXC.IOMUXC_SW_MUX_CTL_PAD_FEC_CRS_DV */
typedef struct{
__REG32 MUX_MODE          : 2;
__REG32                   : 2;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   : 2;
__REG32 DSE_TEST          : 1;
__REG32 TEST_TS           : 1;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_gpio_19_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_PAD_NVCC_KEYPAD */
typedef struct{
__REG32                   :17;
__REG32 VDOEN             : 1;
__REG32 HVEOVERWRITE      : 1;
__REG32                   :13;
} __iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM3 */
typedef struct{
__REG32                   : 4;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32 DDR_INPUT         : 1;
__REG32                   : 2;
__REG32 TEST_TS           : 1;
__REG32                   : 6;
__REG32 DSE               : 3;
__REG32 ODT               : 3;
__REG32 DDR_SEL           : 2;
__REG32                   : 5;
} __iomuxc_sw_pad_ctl_pad_dram_dqm3_bits;

/* IOMUXC_SW_PAD_CTL_PAD_PMIC_STBY_REQ */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   : 1;
__REG32 STRENGTH_MODE     : 1;
__REG32 DSE_TEST          : 1;
__REG32 TEST_TS           : 1;
__REG32                   :19;
} __iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits;

/* PAD_CTL_GRP Register */
typedef struct{
__REG32                   :19;
__REG32 DSE               : 3;
__REG32                   :10;
} __iomuxc_sw_pad_ctl_grp_addds_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDRMODE_CTL */
typedef struct{
__REG32                   : 9;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_sw_pad_ctl_grp_ddrmode_ctl_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDRPKE */
typedef struct{
__REG32                   : 7;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_grp_ddrpke_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDRPK */
typedef struct{
__REG32                   : 6;
__REG32 PKE               : 1;
__REG32                   :25;
} __iomuxc_sw_pad_ctl_grp_ddrpk_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDRHYS */
typedef struct{
__REG32                   : 8;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_grp_ddrhys_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDRMODE */
typedef struct{
__REG32                   : 9;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_sw_pad_ctl_grp_ddrmode_bits;

/* IOMUXC.IOMUXC_SW_PAD_CTL_GRP_DDR_TYPE */
typedef struct{
__REG32                   :25;
__REG32 DDR_SEL           : 2;
__REG32                   : 5;
} __iomuxc_sw_pad_ctl_grp_ddr_type_bits;

/* SELECT_INPUT Register */
typedef struct{
__REG32 DAISY             : 1;
__REG32                   :31;
} __iomuxc_audmux_p4_input_da_amx_select_input_bits;

/* IOMUXC.IOMUXC_CAN1_IPP_IND_CANRX_SELECT_INPUT */
typedef struct{
__REG32 DAISY             : 2;
__REG32                   :30;
} __iomuxc_can1_ipp_ind_canrx_select_input_bits;

/* IOMUXC.IOMUXC_UART2_IPP_UART_RXD_MUX_SELECT_INPUT */
typedef struct{
__REG32 DAISY             : 3;
__REG32                   :29;
} __iomuxc_uart2_ipp_uart_rxd_mux_select_input_bits;

/* Time Stamp Unit Parsing Definitions Register 1 (IPTP.PTP_TSPDR1) */
typedef struct{
__REG32 ETT               :16;
__REG32 IPT               : 8;
__REG32                   : 8;
} __ptp_tspdr1_bits;

/* Time Stamp Unit Parsing Definitions Register 2(IPTP.PTP_TSPDR2) */
typedef struct{
__REG32 DPNGE             :16;
__REG32 DPNEV             :16;
} __ptp_tspdr2_bits;

/* Time Stamp Unit Parsing Definitions Register 3 (IPTP.PTP_TSPDR3) */
typedef struct{
__REG32 SYCTL             : 8;
__REG32 DRCTL             : 8;
__REG32 DRPCTL            : 8;
__REG32 FUCTL             : 8;
} __ptp_tspdr3_bits;

/* Time Stamp Unit Parsing Definitions Register 4 (IPTP.PTP_TSPDR4) */
typedef struct{
__REG32 MACTL             : 8;
__REG32                   : 8;
__REG32 VLAN              :16;
} __ptp_tspdr4_bits;

/* Time Stamp Unit Parsing Definitions Register 5 (IPTP.PTP_TSPDR5) */
typedef struct{
__REG32 SYMSG             : 4;
__REG32                   : 4;
__REG32 DRMSG             : 4;
__REG32                   : 4;
__REG32 PDRMSG            : 4;
__REG32                   : 4;
__REG32 PDRPMSG           : 4;
__REG32                   : 4;
} __ptp_tspdr5_bits;

/* Time Stamp Unit Parsing Definitions Register 6 (IPTP.PTP_TSPDR6) */
typedef struct{
__REG32 FUMSG             : 4;
__REG32                   : 4;
__REG32 DRPMSG            : 4;
__REG32                   : 4;
__REG32 PDFUMSG           : 4;
__REG32                   : 4;
__REG32 ANNMSG            : 4;
__REG32                   : 4;
} __ptp_tspdr6_bits;

/* Time Stamp Unit Parsing Definitions Register 7 (IPTP.PTP_TSPDR7) */
typedef struct{
__REG32 SIGMSG            : 4;
__REG32                   : 4;
__REG32 MAMSG             : 4;
__REG32                   : 4;
__REG32 TRANSPEC          : 4;
__REG32                   :12;
} __ptp_tspdr7_bits;

/* Time Stamp Unit Parsing Offset Values (IPTP.PTP_TSPOV) */
typedef struct{
__REG32 ETTOF             : 8;
__REG32 IPTOF             : 8;
__REG32 UDOF              : 8;
__REG32 PTOF              : 8;
} __ptp_tspov_bits;

/* Time Stamp Unit Mode Register (IPTP.PTP_TSMR) */
typedef struct{
__REG32                   :12;
__REG32 OPMODE1           : 1;
__REG32 OPMODE2           : 1;
__REG32 OPMODE3           : 1;
__REG32 OPMODE4           : 1;
__REG32                   : 7;
__REG32 PTPV              : 1;
__REG32 ENUDP             : 1;
__REG32 ENTRN             : 1;
__REG32                   : 2;
__REG32 EN1               : 1;
__REG32 EN2               : 1;
__REG32 EN3               : 1;
__REG32 EN4               : 1;
} __ptp_tsmr_bits;

/* Event Register (IPTP.TMR_PEVENT) / Mask Register (IPTP.TMR_PEMASK) */
typedef struct{
__REG32 EXR               : 1;
__REG32 OVR1              : 1;
__REG32 OVT1              : 1;
__REG32 SYRE1             : 1;
__REG32 DRQRE1            : 1;
__REG32 TXE1              : 1;
__REG32 PDRQRE1           : 1;
__REG32 PDRSRE1           : 1;
__REG32 EXT               : 1;
__REG32 OVR2              : 1;
__REG32 OVT2              : 1;
__REG32 SYRE2             : 1;
__REG32 DRQRE2            : 1;
__REG32 TXE2              : 1;
__REG32 PDRQRE2           : 1;
__REG32 PDRSRE2           : 1;
__REG32                   : 1;
__REG32 OVR3              : 1;
__REG32 OVT3              : 1;
__REG32 SYRE3             : 1;
__REG32 DRQRE3            : 1;
__REG32 TXE3              : 1;
__REG32 PDRQRE3           : 1;
__REG32 PDRSRE3           : 1;
__REG32                   : 1;
__REG32 OVR4              : 1;
__REG32 OVT4              : 1;
__REG32 SYRE4             : 1;
__REG32 DRQRE4            : 1;
__REG32 TXE4              : 1;
__REG32 PDRQRE4           : 1;
__REG32 PDRSRE4           : 1;
} __ptp_tmr_pevent_bits;

/* Timer Control Register (IPTP.TMR_CTRL) */
typedef struct{
__REG32 ALM1P             : 1;
__REG32 ALM2P             : 1;
__REG32                   : 2;
__REG32 PPSW              : 1;
__REG32                   : 1;
__REG32 TCLK_PERIOD       :10;
__REG32                   : 6;
__REG32 ETEP2             : 1;
__REG32 ETEP1             : 1;
__REG32 COPH              : 1;
__REG32 CIPH              : 1;
__REG32 TMSR              : 1;
__REG32 DBG               : 1;
__REG32 BYP               : 1;
__REG32 TE                : 1;
__REG32 CKSEL             : 2;
} __tmr_ctrl_bits;

/* Timer Event Register and Timer Event Mask Register */
typedef struct{
__REG32                   : 6;
__REG32 ETS2              : 1;
__REG32 ETS1              : 1;
__REG32                   : 6;
__REG32 ALM1              : 1;
__REG32 ALM2              : 1;
__REG32 FS                : 1;
__REG32                   : 7;
__REG32 PP1               : 1;
__REG32 PP2               : 1;
__REG32 PP3               : 1;
__REG32                   : 5;
} __tmr_tevent_bits;

/* Timer Prescale Register (IPTP.TMR_PRSC) */
typedef struct{
__REG32                   :16;
__REG32 PRSC_OCK          :16;
} __tmr_prsc_bits;

/* Configuration Register (IPU.IPU_CONF) */
typedef struct{
__REG32 CSI0_EN           : 1;
__REG32 CSI1_EN           : 1;
__REG32 IC_EN             : 1;
__REG32 IRT_EN            : 1;
__REG32                   : 1;
__REG32 DP_EN             : 1;
__REG32 DI0_EN            : 1;
__REG32 DI1_EN            : 1;
__REG32 SMFC_EN           : 1;
__REG32 DC_EN             : 1;
__REG32 DMFC_EN           : 1;
__REG32 SISG_EN           : 1;
__REG32 VDI_EN            : 1;
__REG32                   : 3;
__REG32 IPU_DIAGBUS_MODE  : 5;
__REG32 IPU_DIAGBUS_ON    : 1;
__REG32 IDMAC_DISABLE     : 1;
__REG32                   : 2;
__REG32 IC_DMFC_SEL       : 1;
__REG32 IC_DMFC_SYNC      : 1;
__REG32 VDI_DMFC_SYNC     : 1;
__REG32                   : 2;
__REG32 IC_INPUT          : 1;
__REG32 CSI_SEL           : 1;
} __ipu_conf_bits;

/* SISG Control 0 Register (IPU.SISG_CTRL0) */
typedef struct{
__REG32 VSYNC_RST_CNT         : 1;
__REG32 NO_VSYNC_2_STRT_CNT   : 3;
__REG32 VAL_STOP_SISG_COUNTER :25;
__REG32 MCU_ACTV_TRIG         : 1;
__REG32 EXT_ACTV              : 1;
__REG32                       : 1;
} __ipu_sisg_ctrl0_bits;

/* SISG Control 1 Register (IPU.SISG_CTRL1) */
typedef struct{
__REG32 SISG_STROBE_CNT       : 5;
__REG32                       : 3;
__REG32 SISG_OUT_POL          : 6;
__REG32                       :18;
} __ipu_sisg_ctrl1_bits;

/* SISG Set <i> Register (IPU.SISG_SET <i>) */
typedef struct{
__REG32 SISG_SET              :25;
__REG32                       : 7;
} __ipu_sisg_set_bits;

/* SISG Clear <i> Register (IPU.SISG_CLR <i>) */
typedef struct{
__REG32 SISG_CLEAR             :25;
__REG32                       : 7;
} __ipu_sisg_clr_bits;

/* Interrupt Control Register 1 (IPU.IPU_INT_CTRL_1) */
typedef struct{
__REG32 IDMAC_EOF_EN_0        : 1;
__REG32 IDMAC_EOF_EN_1        : 1;
__REG32 IDMAC_EOF_EN_2        : 1;
__REG32 IDMAC_EOF_EN_3        : 1;
__REG32                       : 1;
__REG32 IDMAC_EOF_EN_5        : 1;
__REG32                       : 2;
__REG32 IDMAC_EOF_EN_8        : 1;
__REG32 IDMAC_EOF_EN_9        : 1;
__REG32 IDMAC_EOF_EN_10       : 1;
__REG32 IDMAC_EOF_EN_11       : 1;
__REG32 IDMAC_EOF_EN_12       : 1;
__REG32 IDMAC_EOF_EN_13       : 1;
__REG32 IDMAC_EOF_EN_14       : 1;
__REG32 IDMAC_EOF_EN_15       : 1;
__REG32                       : 1;
__REG32 IDMAC_EOF_EN_17       : 1;
__REG32 IDMAC_EOF_EN_18       : 1;
__REG32 IDMAC_EOF_EN_19       : 1;
__REG32 IDMAC_EOF_EN_20       : 1;
__REG32 IDMAC_EOF_EN_21       : 1;
__REG32 IDMAC_EOF_EN_22       : 1;
__REG32 IDMAC_EOF_EN_23       : 1;
__REG32 IDMAC_EOF_EN_24       : 1;
__REG32 IDMAC_EOF_EN_25       : 1;
__REG32 IDMAC_EOF_EN_26       : 1;
__REG32 IDMAC_EOF_EN_27       : 1;
__REG32 IDMAC_EOF_EN_28       : 1;
__REG32 IDMAC_EOF_EN_29       : 1;
__REG32                       : 1;
__REG32 IDMAC_EOF_EN_31       : 1;
} __ipu_int_ctrl_1_bits;

/* IPU Interrupt Control Register 2 (IPU.IPU_INT_CTRL_2) */
typedef struct{
__REG32                       : 1;
__REG32 IDMAC_EOF_EN_33       : 1;
__REG32                       : 6;
__REG32 IDMAC_EOF_EN_40       : 1;
__REG32 IDMAC_EOF_EN_41       : 1;
__REG32 IDMAC_EOF_EN_42       : 1;
__REG32 IDMAC_EOF_EN_43       : 1;
__REG32 IDMAC_EOF_EN_44       : 1;
__REG32 IDMAC_EOF_EN_45       : 1;
__REG32 IDMAC_EOF_EN_46       : 1;
__REG32 IDMAC_EOF_EN_47       : 1;
__REG32 IDMAC_EOF_EN_48       : 1;
__REG32 IDMAC_EOF_EN_49       : 1;
__REG32 IDMAC_EOF_EN_50       : 1;
__REG32 IDMAC_EOF_EN_51       : 1;
__REG32 IDMAC_EOF_EN_52       : 1;
__REG32                       :11;
} __ipu_int_ctrl_2_bits;

/* IPU Interrupt Control Register 3 (IPU.IPU_INT_CTRL_3) */
typedef struct{
__REG32 IDMAC_NFACK_EN_0      : 1;
__REG32 IDMAC_NFACK_EN_1      : 1;
__REG32 IDMAC_NFACK_EN_2      : 1;
__REG32 IDMAC_NFACK_EN_3      : 1;
__REG32                       : 1;
__REG32 IDMAC_NFACK_EN_5      : 1;
__REG32                       : 2;
__REG32 IDMAC_NFACK_EN_8      : 1;
__REG32 IDMAC_NFACK_EN_9      : 1;
__REG32 IDMAC_NFACK_EN_10     : 1;
__REG32 IDMAC_NFACK_EN_11     : 1;
__REG32 IDMAC_NFACK_EN_12     : 1;
__REG32 IDMAC_NFACK_EN_13     : 1;
__REG32 IDMAC_NFACK_EN_14     : 1;
__REG32 IDMAC_NFACK_EN_15     : 1;
__REG32                       : 1;
__REG32 IDMAC_NFACK_EN_17     : 1;
__REG32 IDMAC_NFACK_EN_18     : 1;
__REG32 IDMAC_NFACK_EN_19     : 1;
__REG32 IDMAC_NFACK_EN_20     : 1;
__REG32 IDMAC_NFACK_EN_21     : 1;
__REG32 IDMAC_NFACK_EN_22     : 1;
__REG32 IDMAC_NFACK_EN_23     : 1;
__REG32 IDMAC_NFACK_EN_24     : 1;
__REG32 IDMAC_NFACK_EN_25     : 1;
__REG32 IDMAC_NFACK_EN_26     : 1;
__REG32 IDMAC_NFACK_EN_27     : 1;
__REG32 IDMAC_NFACK_EN_28     : 1;
__REG32 IDMAC_NFACK_EN_29     : 1;
__REG32                       : 1;
__REG32 IDMAC_NFACK_EN_31     : 1;
} __ipu_int_ctrl_3_bits;

/* IPU Interrupt Control Register 4 (IPU.IPU_INT_CTRL_4) */
typedef struct{
__REG32                       : 1;
__REG32 IDMAC_NFACK_EN_33     : 1;
__REG32                       : 6;
__REG32 IDMAC_NFACK_EN_40     : 1;
__REG32 IDMAC_NFACK_EN_41     : 1;
__REG32 IDMAC_NFACK_EN_42     : 1;
__REG32 IDMAC_NFACK_EN_43     : 1;
__REG32 IDMAC_NFACK_EN_44     : 1;
__REG32 IDMAC_NFACK_EN_45     : 1;
__REG32 IDMAC_NFACK_EN_46     : 1;
__REG32 IDMAC_NFACK_EN_47     : 1;
__REG32 IDMAC_NFACK_EN_48     : 1;
__REG32 IDMAC_NFACK_EN_49     : 1;
__REG32 IDMAC_NFACK_EN_50     : 1;
__REG32 IDMAC_NFACK_EN_51     : 1;
__REG32 IDMAC_NFACK_EN_52     : 1;
__REG32                       :11;
} __ipu_int_ctrl_4_bits;

/* IPU Interrupt Control Register 5 (IPU.IPU_INT_CTRL_5) */
typedef struct{
__REG32 IDMAC_NFB4EOF_EN_0    : 1;
__REG32 IDMAC_NFB4EOF_EN_1    : 1;
__REG32 IDMAC_NFB4EOF_EN_2    : 1;
__REG32 IDMAC_NFB4EOF_EN_3    : 1;
__REG32                       : 1;
__REG32 IDMAC_NFB4EOF_EN_5    : 1;
__REG32                       : 2;
__REG32 IDMAC_NFB4EOF_EN_8    : 1;
__REG32 IDMAC_NFB4EOF_EN_9    : 1;
__REG32 IDMAC_NFB4EOF_EN_10   : 1;
__REG32 IDMAC_NFB4EOF_EN_11   : 1;
__REG32 IDMAC_NFB4EOF_EN_12   : 1;
__REG32 IDMAC_NFB4EOF_EN_13   : 1;
__REG32 IDMAC_NFB4EOF_EN_14   : 1;
__REG32 IDMAC_NFB4EOF_EN_15   : 1;
__REG32                       : 1;
__REG32 IDMAC_NFB4EOF_EN_17   : 1;
__REG32 IDMAC_NFB4EOF_EN_18   : 1;
__REG32 IDMAC_NFB4EOF_EN_19   : 1;
__REG32 IDMAC_NFB4EOF_EN_20   : 1;
__REG32 IDMAC_NFB4EOF_EN_21   : 1;
__REG32 IDMAC_NFB4EOF_EN_22   : 1;
__REG32 IDMAC_NFB4EOF_EN_23   : 1;
__REG32 IDMAC_NFB4EOF_EN_24   : 1;
__REG32 IDMAC_NFB4EOF_EN_25   : 1;
__REG32 IDMAC_NFB4EOF_EN_26   : 1;
__REG32 IDMAC_NFB4EOF_EN_27   : 1;
__REG32 IDMAC_NFB4EOF_EN_28   : 1;
__REG32 IDMAC_NFB4EOF_EN_29   : 1;
__REG32                       : 1;
__REG32 IDMAC_NFB4EOF_EN_31   : 1;
} __ipu_int_ctrl_5_bits;

/* IPU Interrupt Control Register 6 (IPU.IPU_INT_CTRL_6) */
typedef struct{
__REG32                       : 1;
__REG32 IDMAC_NFB4EOF_EN_33   : 1;
__REG32                       : 6;
__REG32 IDMAC_NFB4EOF_EN_40   : 1;
__REG32 IDMAC_NFB4EOF_EN_41   : 1;
__REG32 IDMAC_NFB4EOF_EN_42   : 1;
__REG32 IDMAC_NFB4EOF_EN_43   : 1;
__REG32 IDMAC_NFB4EOF_EN_44   : 1;
__REG32 IDMAC_NFB4EOF_EN_45   : 1;
__REG32 IDMAC_NFB4EOF_EN_46   : 1;
__REG32 IDMAC_NFB4EOF_EN_47   : 1;
__REG32 IDMAC_NFB4EOF_EN_48   : 1;
__REG32 IDMAC_NFB4EOF_EN_49   : 1;
__REG32 IDMAC_NFB4EOF_EN_50   : 1;
__REG32 IDMAC_NFB4EOF_EN_51   : 1;
__REG32 IDMAC_NFB4EOF_EN_52   : 1;
__REG32                       :11;
} __ipu_int_ctrl_6_bits;

/* IPU Interrupt Control Register 7 (IPU.IPU_INT_CTRL_7) */
typedef struct{
__REG32                       :19;
__REG32 IDMAC_EOS_EN_19       : 1;
__REG32                       : 3;
__REG32 IDMAC_EOS_EN_23       : 1;
__REG32 IDMAC_EOS_EN_24       : 1;
__REG32 IDMAC_EOS_EN_25       : 1;
__REG32 IDMAC_EOS_EN_26       : 1;
__REG32 IDMAC_EOS_EN_27       : 1;
__REG32 IDMAC_EOS_EN_28       : 1;
__REG32 IDMAC_EOS_EN_29       : 1;
__REG32                       : 1;
__REG32 IDMAC_EOS_EN_31       : 1;
} __ipu_int_ctrl_7_bits;

/* IPU Interrupt Control Register 8 (IPU.IPU_INT_CTRL_8) */
typedef struct{
__REG32                       : 1;
__REG32 IDMAC_EOS_EN_33       : 1;
__REG32                       : 7;
__REG32 IDMAC_EOS_EN_41       : 1;
__REG32 IDMAC_EOS_EN_42       : 1;
__REG32 IDMAC_EOS_EN_43       : 1;
__REG32 IDMAC_EOS_EN_44       : 1;
__REG32                       : 6;
__REG32 IDMAC_EOS_EN_51       : 1;
__REG32 IDMAC_EOS_EN_52       : 1;
__REG32                       :11;
} __ipu_int_ctrl_8_bits;

/* IPU Interrupt Control Register 9 (IPU.IPU_INT_CTRL_9) */
typedef struct{
__REG32 VDI_FIFO1_OVF_EN      : 1;
__REG32                       :25;
__REG32 IC_BAYER_BUF_OVF_EN   : 1;
__REG32 IC_ENC_BUF_OVF_EN     : 1;
__REG32 IC_VF_BUF_OVF_EN      : 1;
__REG32                       : 1;
__REG32 CSI0_PUPE_EN          : 1;
__REG32 CSI1_PUPE_EN          : 1;
} __ipu_int_ctrl_9_bits;

/* IPU Interrupt Control Register 10 (IPU.IPU_INT_CTRL_10) */
typedef struct{
__REG32 SMFC0_FRM_LOST_EN         : 1;
__REG32 SMFC1_FRM_LOST_EN         : 1;
__REG32 SMFC2_FRM_LOST_EN         : 1;
__REG32 SMFC3_FRM_LOST_EN         : 1;
__REG32                           :12;
__REG32 DC_TEARING_ERR_1_EN       : 1;
__REG32 DC_TEARING_ERR_2_EN       : 1;
__REG32 DC_TEARING_ERR_6_EN       : 1;
__REG32 DI0_SYNC_DISP_ERR_EN      : 1;
__REG32 DI1_SYNC_DISP_ERR_EN      : 1;
__REG32 DI0_TIME_OUT_ERR_EN       : 1;
__REG32 DI1_TIME_OUT_ERR_EN       : 1;
__REG32                           : 1;
__REG32 IC_VF_FRM_LOST_ERR_EN     : 1;
__REG32 IC_ENC_FRM_LOST_ERR_EN    : 1;
__REG32 IC_BAYER_FRM_LOST_ERR_EN  : 1;
__REG32                           : 1;
__REG32 NON_PRIVILEGED_ACC_ERR_EN : 1;
__REG32 AXIW_ERR_EN               : 1;
__REG32 AXIR_ERR_EN               : 1;
__REG32                           : 1;
} __ipu_int_ctrl_10_bits;

/* IPU Interrupt Control Register 11 (IPU.IPU_INT_CTRL_11) */
typedef struct{
__REG32 IDMAC_EOBND_EN_0          : 1;
__REG32 IDMAC_EOBND_EN_1          : 1;
__REG32 IDMAC_EOBND_EN_2          : 1;
__REG32 IDMAC_EOBND_EN_3          : 1;
__REG32                           : 1;
__REG32 IDMAC_EOBND_EN_5          : 1;
__REG32                           : 5;
__REG32 IDMAC_EOBND_EN_11         : 1;
__REG32 IDMAC_EOBND_EN_12         : 1;
__REG32                           : 7;
__REG32 IDMAC_EOBND_EN_20         : 1;
__REG32 IDMAC_EOBND_EN_21         : 1;
__REG32 IDMAC_EOBND_EN_22         : 1;
__REG32                           : 2;
__REG32 IDMAC_EOBND_EN_25         : 1;
__REG32 IDMAC_EOBND_EN_26         : 1;
__REG32                           : 5;
} __ipu_int_ctrl_11_bits;

/* IPU Interrupt Control Register 12 (IPU.IPU_INT_CTRL_12) */
typedef struct{
__REG32                           :13;
__REG32 IDMAC_EOBND_EN_45         : 1;
__REG32 IDMAC_EOBND_EN_46         : 1;
__REG32 IDMAC_EOBND_EN_47         : 1;
__REG32 IDMAC_EOBND_EN_48         : 1;
__REG32 IDMAC_EOBND_EN_49         : 1;
__REG32 IDMAC_EOBND_EN_50         : 1;
__REG32                           :13;
} __ipu_int_ctrl_12_bits;

/* IPU Interrupt Control Register 13 (IPU.IPU_INT_CTRL_13) */
typedef struct{
__REG32 IDMAC_TH_EN_0             : 1;
__REG32 IDMAC_TH_EN_1             : 1;
__REG32 IDMAC_TH_EN_2             : 1;
__REG32 IDMAC_TH_EN_3             : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_EN_5             : 1;
__REG32                           : 2;
__REG32 IDMAC_TH_EN_8             : 1;
__REG32 IDMAC_TH_EN_9             : 1;
__REG32 IDMAC_TH_EN_10            : 1;
__REG32 IDMAC_TH_EN_11            : 1;
__REG32 IDMAC_TH_EN_12            : 1;
__REG32 IDMAC_TH_EN_13            : 1;
__REG32 IDMAC_TH_EN_14            : 1;
__REG32 IDMAC_TH_EN_15            : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_EN_17            : 1;
__REG32 IDMAC_TH_EN_18            : 1;
__REG32 IDMAC_TH_EN_19            : 1;
__REG32 IDMAC_TH_EN_20            : 1;
__REG32 IDMAC_TH_EN_21            : 1;
__REG32 IDMAC_TH_EN_22            : 1;
__REG32 IDMAC_TH_EN_23            : 1;
__REG32 IDMAC_TH_EN_24            : 1;
__REG32 IDMAC_TH_EN_25            : 1;
__REG32 IDMAC_TH_EN_26            : 1;
__REG32 IDMAC_TH_EN_27            : 1;
__REG32 IDMAC_TH_EN_28            : 1;
__REG32 IDMAC_TH_EN_29            : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_EN_31            : 1;
} __ipu_int_ctrl_13_bits;

/* IPU Interrupt Control Register 14 (IPU.IPU_INT_CTRL_14) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_TH_EN_33            : 1;
__REG32                           : 6;
__REG32 IDMAC_TH_EN_40            : 1;
__REG32 IDMAC_TH_EN_41            : 1;
__REG32 IDMAC_TH_EN_42            : 1;
__REG32 IDMAC_TH_EN_43            : 1;
__REG32 IDMAC_TH_EN_44            : 1;
__REG32 IDMAC_TH_EN_45            : 1;
__REG32 IDMAC_TH_EN_46            : 1;
__REG32 IDMAC_TH_EN_47            : 1;
__REG32 IDMAC_TH_EN_48            : 1;
__REG32 IDMAC_TH_EN_49            : 1;
__REG32 IDMAC_TH_EN_50            : 1;
__REG32 IDMAC_TH_EN_51            : 1;
__REG32 IDMAC_TH_EN_52            : 1;
__REG32                           :11;
} __ipu_int_ctrl_14_bits;

/* IPU Interrupt Control Register 15 (IPU.IPU_INT_CTRL_15) */
typedef struct{
__REG32 IPU_SNOOPING1_INT_EN      : 1;
__REG32 IPU_SNOOPING2_INT_EN      : 1;
__REG32 DP_SF_START_EN            : 1;
__REG32 DP_SF_END_EN              : 1;
__REG32 DP_ASF_START_EN           : 1;
__REG32 DP_ASF_END_EN             : 1;
__REG32 DP_SF_BRAKE_EN            : 1;
__REG32 DP_ASF_BRAKE_EN           : 1;
__REG32 DC_FC_0_EN                : 1;
__REG32 DC_FC_1_EN                : 1;
__REG32 DC_FC_2_EN                : 1;
__REG32 DC_FC_3_EN                : 1;
__REG32 DC_FC_4_EN                : 1;
__REG32 DC_FC_6_EN                : 1;
__REG32 DI_VSYNC_PRE_0_EN         : 1;
__REG32 DI_VSYNC_PRE_1_EN         : 1;
__REG32 DC_DP_START_EN            : 1;
__REG32 DC_ASYNC_STOP_EN          : 1;
__REG32 DI0_DISP_CLK_EN_PRE_EN    : 1;
__REG32 DI0_CNT_EN_PRE_1_EN       : 1;
__REG32 DI0_CNT_EN_PRE_2_EN       : 1;
__REG32 DI0_CNT_EN_PRE_3_EN       : 1;
__REG32 DI0_CNT_EN_PRE_4_EN       : 1;
__REG32 DI0_CNT_EN_PRE_5_EN       : 1;
__REG32 DI0_CNT_EN_PRE_6_EN       : 1;
__REG32 DI0_CNT_EN_PRE_7_EN       : 1;
__REG32 DI0_CNT_EN_PRE_8_EN       : 1;
__REG32 DI0_CNT_EN_PRE_9_EN       : 1;
__REG32 DI0_CNT_EN_PRE_10_EN      : 1;
__REG32 DI1_DISP_CLK_EN_PRE_EN    : 1;
__REG32 DI1_CNT_EN_PRE_3_EN       : 1;
__REG32 DI1_CNT_EN_PRE_8_EN       : 1;
} __ipu_int_ctrl_15_bits;

/* IPU SDMA Event Control Register 1 (IPU.IPU_SDMA_EVENT_1) */
typedef struct{
__REG32 IDMAC_EOF_SDMA_EN_0       : 1;
__REG32 IDMAC_EOF_SDMA_EN_1       : 1;
__REG32 IDMAC_EOF_SDMA_EN_2       : 1;
__REG32 IDMAC_EOF_SDMA_EN_3       : 1;
__REG32                           : 1;
__REG32 IDMAC_EOF_SDMA_EN_5       : 1;
__REG32                           : 2;
__REG32 IDMAC_EOF_SDMA_EN_8       : 1;
__REG32 IDMAC_EOF_SDMA_EN_9       : 1;
__REG32 IDMAC_EOF_SDMA_EN_10      : 1;
__REG32 IDMAC_EOF_SDMA_EN_11      : 1;
__REG32 IDMAC_EOF_SDMA_EN_12      : 1;
__REG32 IDMAC_EOF_SDMA_EN_13      : 1;
__REG32 IDMAC_EOF_SDMA_EN_14      : 1;
__REG32 IDMAC_EOF_SDMA_EN_15      : 1;
__REG32                           : 1;
__REG32 IDMAC_EOF_SDMA_EN_17      : 1;
__REG32 IDMAC_EOF_SDMA_EN_18      : 1;
__REG32 IDMAC_EOF_SDMA_EN_19      : 1;
__REG32 IDMAC_EOF_SDMA_EN_20      : 1;
__REG32 IDMAC_EOF_SDMA_EN_21      : 1;
__REG32 IDMAC_EOF_SDMA_EN_22      : 1;
__REG32 IDMAC_EOF_SDMA_EN_23      : 1;
__REG32 IDMAC_EOF_SDMA_EN_24      : 1;
__REG32 IDMAC_EOF_SDMA_EN_25      : 1;
__REG32 IDMAC_EOF_SDMA_EN_26      : 1;
__REG32 IDMAC_EOF_SDMA_EN_27      : 1;
__REG32 IDMAC_EOF_SDMA_EN_28      : 1;
__REG32 IDMAC_EOF_SDMA_EN_29      : 1;
__REG32                           : 1;
__REG32 IDMAC_EOF_SDMA_EN_31      : 1;
} __ipu_sdma_event_1_bits;

/* IPU SDMA Event Control Register 2 (IPU.IPU_SDMA_EVENT_2) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_EOF_SDMA_EN_33      : 1;
__REG32                           : 6;
__REG32 IDMAC_EOF_SDMA_EN_40      : 1;
__REG32 IDMAC_EOF_SDMA_EN_41      : 1;
__REG32 IDMAC_EOF_SDMA_EN_42      : 1;
__REG32 IDMAC_EOF_SDMA_EN_43      : 1;
__REG32 IDMAC_EOF_SDMA_EN_44      : 1;
__REG32 IDMAC_EOF_SDMA_EN_45      : 1;
__REG32 IDMAC_EOF_SDMA_EN_46      : 1;
__REG32 IDMAC_EOF_SDMA_EN_47      : 1;
__REG32 IDMAC_EOF_SDMA_EN_48      : 1;
__REG32 IDMAC_EOF_SDMA_EN_49      : 1;
__REG32 IDMAC_EOF_SDMA_EN_50      : 1;
__REG32 IDMAC_EOF_SDMA_EN_51      : 1;
__REG32 IDMAC_EOF_SDMA_EN_52      : 1;
__REG32                           :11;
} __ipu_sdma_event_2_bits;

/* IPU SDMA Event Control Register 3 (IPU.IPU_SDMA_EVENT_3) */
typedef struct{
__REG32 IDMAC_NFACK_SDMA_EN_0     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_1     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_2     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_3     : 1;
__REG32                           : 1;
__REG32 IDMAC_NFACK_SDMA_EN_5     : 1;
__REG32                           : 2;
__REG32 IDMAC_NFACK_SDMA_EN_8     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_9     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_10    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_11    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_12    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_13    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_14    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_15    : 1;
__REG32                           : 1;
__REG32 IDMAC_NFACK_SDMA_EN_17    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_18    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_19    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_20    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_21    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_22    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_23    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_24    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_25    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_26    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_27    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_28    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_29    : 1;
__REG32                           : 1;
__REG32 IDMAC_NFACK_SDMA_EN_31    : 1;
} __ipu_sdma_event_3_bits;

/* IPU SDMA Event Control Register 4 (IPU.IPU_SDMA_EVENT_4) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_NFACK_SDMA_EN_33    : 1;
__REG32                           : 6;
__REG32 IDMAC_NFACK_SDMA_EN_40    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_41    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_42    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_43    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_44    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_45    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_46    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_47    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_48    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_49    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_50    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_51    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_52    : 1;
__REG32                           :11;
} __ipu_sdma_event_4_bits;

/* IPU SDMA Event Control Register 7 (IPU.IPU_SDMA_EVENT_7) */
typedef struct{
__REG32                           :19;
__REG32 IDMAC_EOS_SDMA_EN_19      : 1;
__REG32                           : 3;
__REG32 IDMAC_EOS_SDMA_EN_23      : 1;
__REG32 IDMAC_EOS_SDMA_EN_24      : 1;
__REG32 IDMAC_EOS_SDMA_EN_25      : 1;
__REG32 IDMAC_EOS_SDMA_EN_26      : 1;
__REG32 IDMAC_EOS_SDMA_EN_27      : 1;
__REG32 IDMAC_EOS_SDMA_EN_28      : 1;
__REG32 IDMAC_EOS_SDMA_EN_29      : 1;
__REG32                           : 1;
__REG32 IDMAC_EOS_SDMA_EN_31      : 1;
} __ipu_sdma_event_7_bits;

/* IPU SDMA Event Control Register 8 (IPU.IPU_SDMA_EVENT_8) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_EOS_SDMA_EN_33      : 1;
__REG32                           : 7;
__REG32 IDMAC_EOS_SDMA_EN_41      : 1;
__REG32 IDMAC_EOS_SDMA_EN_42      : 1;
__REG32 IDMAC_EOS_SDMA_EN_43      : 1;
__REG32 IDMAC_EOS_SDMA_EN_44      : 1;
__REG32                           : 6;
__REG32 IDMAC_EOS_SDMA_EN_51      : 1;
__REG32 IDMAC_EOS_SDMA_EN_52      : 1;
__REG32                           :11;
} __ipu_sdma_event_8_bits;

/* IPU SDMA Event Control Register 11 (IPU.IPU_SDMA_EVENT_11) */
typedef struct{
__REG32 IDMAC_EOBND_SDMA_EN_0     : 1;
__REG32 IDMAC_EOBND_SDMA_EN_1     : 1;
__REG32 IDMAC_EOBND_SDMA_EN_2     : 1;
__REG32 IDMAC_EOBND_SDMA_EN_3     : 1;
__REG32                           : 1;
__REG32 IDMAC_EOBND_SDMA_EN_5     : 1;
__REG32                           : 5;
__REG32 IDMAC_EOBND_SDMA_EN_11    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_12    : 1;
__REG32                           : 7;
__REG32 IDMAC_EOBND_SDMA_EN_20    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_21    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_22    : 1;
__REG32                           : 2;
__REG32 IDMAC_EOBND_SDMA_EN_25    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_26    : 1;
__REG32                           : 5;
} __ipu_sdma_event_11_bits;

/* IPU SDMA Event Control Register 12 (IPU.IPU_SDMA_EVENT_12) */
typedef struct{
__REG32                           :13;
__REG32 IDMAC_EOBND_SDMA_EN_45    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_46    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_47    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_48    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_49    : 1;
__REG32 IDMAC_EOBND_SDMA_EN_50    : 1;
__REG32                           :13;
} __ipu_sdma_event_12_bits;

/* IPU SDMA Event Control Register 13 (IPU.IPU_SDMA_EVENT_13) */
typedef struct{
__REG32 IDMAC_TH_SDMA_EN_0        : 1;
__REG32 IDMAC_TH_SDMA_EN_1        : 1;
__REG32 IDMAC_TH_SDMA_EN_2        : 1;
__REG32 IDMAC_TH_SDMA_EN_3        : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_SDMA_EN_5        : 1;
__REG32                           : 2;
__REG32 IDMAC_TH_SDMA_EN_8        : 1;
__REG32 IDMAC_TH_SDMA_EN_9        : 1;
__REG32 IDMAC_TH_SDMA_EN_10       : 1;
__REG32 IDMAC_TH_SDMA_EN_11       : 1;
__REG32 IDMAC_TH_SDMA_EN_12       : 1;
__REG32 IDMAC_TH_SDMA_EN_13       : 1;
__REG32 IDMAC_TH_SDMA_EN_14       : 1;
__REG32 IDMAC_TH_SDMA_EN_15       : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_SDMA_EN_17       : 1;
__REG32 IDMAC_TH_SDMA_EN_18       : 1;
__REG32 IDMAC_TH_SDMA_EN_19       : 1;
__REG32 IDMAC_TH_SDMA_EN_20       : 1;
__REG32 IDMAC_TH_SDMA_EN_21       : 1;
__REG32 IDMAC_TH_SDMA_EN_22       : 1;
__REG32 IDMAC_TH_SDMA_EN_23       : 1;
__REG32 IDMAC_TH_SDMA_EN_24       : 1;
__REG32 IDMAC_TH_SDMA_EN_25       : 1;
__REG32 IDMAC_TH_SDMA_EN_26       : 1;
__REG32 IDMAC_TH_SDMA_EN_27       : 1;
__REG32 IDMAC_TH_SDMA_EN_28       : 1;
__REG32 IDMAC_TH_SDMA_EN_29       : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_SDMA_EN_31       : 1;
} __ipu_sdma_event_13_bits;

/* IPU SDMA Event Control Register 14 (IPU.IPU_SDMA_EVENT_14) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_TH_SDMA_EN_33       : 1;
__REG32                           : 6;
__REG32 IDMAC_TH_SDMA_EN_40       : 1;
__REG32 IDMAC_TH_SDMA_EN_41       : 1;
__REG32 IDMAC_TH_SDMA_EN_42       : 1;
__REG32 IDMAC_TH_SDMA_EN_43       : 1;
__REG32 IDMAC_TH_SDMA_EN_44       : 1;
__REG32 IDMAC_TH_SDMA_EN_45       : 1;
__REG32 IDMAC_TH_SDMA_EN_46       : 1;
__REG32 IDMAC_TH_SDMA_EN_47       : 1;
__REG32 IDMAC_TH_SDMA_EN_48       : 1;
__REG32 IDMAC_TH_SDMA_EN_49       : 1;
__REG32 IDMAC_TH_SDMA_EN_50       : 1;
__REG32 IDMAC_TH_SDMA_EN_51       : 1;
__REG32 IDMAC_TH_SDMA_EN_52       : 1;
__REG32                           :11;
} __ipu_sdma_event_14_bits;

/* IPU Shadow Register Memory Priority 1 Register (IPU.IPU_SRM_PRI1) */
typedef struct{
__REG32 CSI1_SRM_PRI              : 3;
__REG32 CSI1_SRM_MODE             : 2;
__REG32                           : 3;
__REG32 CSI0_SRM_PRI              : 3;
__REG32 CSI0_SRM_MODE             : 2;
__REG32                           :19;
} __ipu_srm_pri1_bits;

/* IPU Shadow Register Memory Priority 2 Register (IPU.IPU_SRM_PRI2) */
typedef struct{
__REG32 DP_SRM_PRI                : 3;
__REG32 DP_S_SRM_MODE             : 2;
__REG32 DP_A0_SRM_MODE            : 2;
__REG32 DP_A1_SRM_MODE            : 2;
__REG32 DC_SRM_PRI                : 3;
__REG32 DC_2_SRM_MODE             : 2;
__REG32 DC_6_SRM_MODE             : 2;
__REG32 DI0_SRM_PRI               : 3;
__REG32 DI0_SRM_MODE              : 2;
__REG32                           : 3;
__REG32 DI1_SRM_PRI               : 3;
__REG32 DI1_SRM_MODE              : 2;
__REG32                           : 3;
} __ipu_srm_pri2_bits;

/* IPU FSU Processing Flow 1 Register (IPU.IPU_FS_PROC_FLOW1) */
typedef struct{
__REG32 PRPENC_ROT_SRC_SEL        : 4;
__REG32                           : 4;
__REG32 PRPVF_ROT_SRC_SEL         : 4;
__REG32 PP_SRC_SEL                : 4;
__REG32 PP_ROT_SRC_SEL            : 4;
__REG32 VDI1_SRC_SEL              : 2;
__REG32 VDI3_SRC_SEL              : 2;
__REG32 PRP_SRC_SEL               : 4;
__REG32 VDI_SRC_SEL               : 2;
__REG32 ENC_IN_VALID              : 1;
__REG32 VF_IN_VALID               : 1;
} __ipu_fs_proc_flow1_bits;

/* IPU FSU Processing Flow 2 Register (IPU.IPU_FS_PROC_FLOW2) */
typedef struct{
__REG32 PRP_ENC_DEST_SEL          : 4;
__REG32 PRPVF_DEST_SEL            : 4;
__REG32 PRPVF_ROT_DEST_SEL        : 4;
__REG32 PP_DEST_SEL               : 4;
__REG32 PP_ROT_DEST_SEL           : 4;
__REG32 PRPENC_ROT_DEST_SEL       : 4;
__REG32 PRP_DEST_SEL              : 4;
__REG32                           : 4;
} __ipu_fs_proc_flow2_bits;

/* IPU FSU Processing Flow 3 Register (IPU.IPU_FS_PROC_FLOW3) */
typedef struct{
__REG32 SMFC0_DEST_SEL            : 4;
__REG32 SMFC1_DEST_SEL            : 3;
__REG32 SMFC2_DEST_SEL            : 4;
__REG32 SMFC3_DEST_SEL            : 3;
__REG32                           : 6;
__REG32 EXT_SRC1_DEST_SEL         : 2;
__REG32 EXT_SRC2_DEST_SEL         : 2;
__REG32                           : 8;
} __ipu_fs_proc_flow3_bits;

/* IPU FSU Displaying Flow 1 Register (IPU.IPU_FS_DISP_FLOW1) */
typedef struct{
__REG32 DP_SYNC0_SRC_SEL          : 4;
__REG32 DP_SYNC1_SRC_SEL          : 4;
__REG32 DP_ASYNC0_SRC_SEL         : 4;
__REG32 DP_ASYNC1_SRC_SEL         : 4;
__REG32 DC2_SRC_SEL               : 4;
__REG32 DC1_SRC_SEL               : 4;
__REG32                           : 8;
} __ipu_fs_disp_flow1_bits;

/* IPU FSU Displaying Flow 2 Register (IPU.IPU_FS_DISP_FLOW2) */
typedef struct{
__REG32 DP_ASYNC0_ALT_SRC_SEL     : 4;
__REG32 DP_ASYNC1_ALT_SRC_SEL     : 4;
__REG32                           : 8;
__REG32 DC2_ALT_SRC_SEL           : 4;
__REG32                           :12;
} __ipu_fs_disp_flow2_bits;

/* IPU SKIP Register (IPU.IPU_SKIP) */
typedef struct{
__REG32 CSI_MAX_RATIO_SKIP_IC_ENC : 3;
__REG32 CSI_SKIP_IC_ENC           : 5;
__REG32 CSI_MAX_RATIO_SKIP_IC_VF  : 3;
__REG32 CSI_SKIP_IC_VF            : 5;
__REG32 VDI_MAX_RATIO_SKIP        : 4;
__REG32 VDI_SKIP                  :12;
} __ipu_skip_bits;

/* IPU Display General Control Register (IPU.IPU_DISP_GEN) */
typedef struct{
__REG32 DI0_DUAL_MODE             : 1;
__REG32 DI1_DUAL_MODE             : 1;
__REG32 DC2_DOUBLE_FLOW           : 1;
__REG32 DP_ASYNC_DOUBLE_FLOW      : 1;
__REG32 DP_FG_EN_ASYNC0           : 1;
__REG32 DP_FG_EN_ASYNC1           : 1;
__REG32 DP_PIPE_CLR               : 1;
__REG32                           : 9;
__REG32 MCU_DI_ID_8               : 1;
__REG32 MCU_DI_ID_9               : 1;
__REG32 MCU_T                     : 4;
__REG32 MCU_MAX_BURST_STOP        : 1;
__REG32 CSI_VSYNC_DEST            : 1;
__REG32 DI0_COUNTER_RELEASE       : 1;
__REG32 DI1_COUNTER_RELEASE       : 1;
__REG32                           : 6;
} __ipu_disp_gen_bits;

/* IPU Display Alternate Flow Control Register 1 (IPU.IPU_DISP_ALT1) */
typedef struct{
__REG32 run_value_m1_alt_0        :12;
__REG32 cnt_clr_sel_alt_0         : 3;
__REG32 cnt_auto_reload_alt_0     : 1;
__REG32 step_repeat_alt_0         :12;
__REG32 sel_alt_0                 : 4;
} __ipu_disp_alt1_bits;

/* IPU Display Alternate Flow Control Register 2 (IPU.IPU_DISP_ALT2) */
typedef struct{
__REG32 offset_value_alt_0        :12;
__REG32 offset_resolution_alt_0   : 3;
__REG32                           : 1;
__REG32 run_resolution_alt_0      : 3;
__REG32                           :13;
} __ipu_disp_alt2_bits;

/* IPU Display Alternate Flow Control Register 3 (IPU.IPU_DISP_ALT3) */
typedef struct{
__REG32 run_value_m1_alt_1        :12;
__REG32 cnt_clr_sel_alt_1         : 3;
__REG32 cnt_auto_reload_alt_1     : 1;
__REG32 step_repeat_alt_1         :12;
__REG32 sel_alt_1                 : 4;
} __ipu_disp_alt3_bits;

/* IPU Display Alternate Flow Control Register 4 (IPU.IPU_DISP_ALT4) */
typedef struct{
__REG32 offset_value_alt_1        :12;
__REG32 offset_resolution_alt_1   : 3;
__REG32                           : 1;
__REG32 run_resolution_alt_1      : 3;
__REG32                           :13;
} __ipu_disp_alt4_bits;

/* IPU Autorefresh and Snooping Control Register (IPU.IPU_SNOOP) */
typedef struct{
__REG32 AUTOREF_PER               :10;
__REG32                           : 6;
__REG32 SNOOP2_SYNC_BYP           : 1;
__REG32                           :15;
} __ipu_snoop_bits;

/* IPU Memory Reset Control Register (IPU.IPU_MEM_RST) */
typedef struct{
__REG32 RST_MEM_EN                :23;
__REG32                           : 8;
__REG32 RST_MEM_START             : 1;
} __ipu_mem_rst_bits;

/* IPU Power Modes Control Register (IPU.IPU_PM) */
typedef struct{
__REG32 DI0_CLK_PERIOD_0          : 7;
__REG32 DI0_CLK_PERIOD_1          : 7;
__REG32 DI0_SRM_CLOCK_CHANGE_MODE : 1;
__REG32 CLOCK_MODE_STAT           : 1;
__REG32 DI1_CLK_PERIOD_0          : 7;
__REG32 DI1_CLK_PERIOD_1          : 7;
__REG32 DI1_SRM_CLOCK_CHANGE_MODE : 1;
__REG32 LPSR_MODE                 : 1;
} __ipu_pm_bits;

/* IPU General Purpose Register (IPU.IPU_GPR) */
typedef struct{
__REG32 IPU_GP0                   : 1;
__REG32 IPU_GP1                   : 1;
__REG32 IPU_GP2                   : 1;
__REG32 IPU_GP3                   : 1;
__REG32 IPU_GP4                   : 1;
__REG32 IPU_GP5                   : 1;
__REG32 IPU_GP6                   : 1;
__REG32 IPU_GP7                   : 1;
__REG32 IPU_GP8                   : 1;
__REG32 IPU_GP9                   : 1;
__REG32 IPU_GP10                  : 1;
__REG32 IPU_GP11                  : 1;
__REG32 IPU_GP12                  : 1;
__REG32 IPU_GP13                  : 1;
__REG32 IPU_GP14                  : 1;
__REG32 IPU_GP15                  : 1;
__REG32 IPU_GP16                  : 1;
__REG32 IPU_GP17                  : 1;
__REG32 IPU_GP18                  : 1;
__REG32 IPU_GP19                  : 1;
__REG32 IPU_CH_BUF2_RDY0_CLR      : 1;
__REG32 IPU_CH_BUF2_RDY1_CLR      : 1;
__REG32 IPU_DI0_CLK_CHANGE_ACK_DIS: 1;
__REG32 IPU_DI1_CLK_CHANGE_ACK_DIS: 1;
__REG32 IPU_ALT_CH_BUF0_RDY0_CLR  : 1;
__REG32 IPU_ALT_CH_BUF0_RDY1_CLR  : 1;
__REG32 IPU_ALT_CH_BUF1_RDY0_CLR  : 1;
__REG32 IPU_ALT_CH_BUF1_RDY1_CLR  : 1;
__REG32 IPU_CH_BUF0_RDY0_CLR      : 1;
__REG32 IPU_CH_BUF0_RDY1_CLR      : 1;
__REG32 IPU_CH_BUF1_RDY0_CLR      : 1;
__REG32 IPU_CH_BUF1_RDY1_CLR      : 1;
} __ipu_gpr_bits;

/* IPU channel double buffer mode select 0 register (IPU.IPU_CH_DB_MODE_SEL0) */
typedef struct{
__REG32 DMA_CH_DB_MODE_SEL_0      : 1;
__REG32 DMA_CH_DB_MODE_SEL_1      : 1;
__REG32 DMA_CH_DB_MODE_SEL_2      : 1;
__REG32 DMA_CH_DB_MODE_SEL_3      : 1;
__REG32 DMA_CH_DB_MODE_SEL_4      : 1;
__REG32 DMA_CH_DB_MODE_SEL_5      : 1;
__REG32 DMA_CH_DB_MODE_SEL_6      : 1;
__REG32 DMA_CH_DB_MODE_SEL_7      : 1;
__REG32 DMA_CH_DB_MODE_SEL_8      : 1;
__REG32 DMA_CH_DB_MODE_SEL_9      : 1;
__REG32 DMA_CH_DB_MODE_SEL_10     : 1;
__REG32 DMA_CH_DB_MODE_SEL_11     : 1;
__REG32 DMA_CH_DB_MODE_SEL_12     : 1;
__REG32 DMA_CH_DB_MODE_SEL_13     : 1;
__REG32 DMA_CH_DB_MODE_SEL_14     : 1;
__REG32 DMA_CH_DB_MODE_SEL_15     : 1;
__REG32 DMA_CH_DB_MODE_SEL_16     : 1;
__REG32 DMA_CH_DB_MODE_SEL_17     : 1;
__REG32 DMA_CH_DB_MODE_SEL_18     : 1;
__REG32 DMA_CH_DB_MODE_SEL_19     : 1;
__REG32 DMA_CH_DB_MODE_SEL_20     : 1;
__REG32 DMA_CH_DB_MODE_SEL_21     : 1;
__REG32 DMA_CH_DB_MODE_SEL_22     : 1;
__REG32 DMA_CH_DB_MODE_SEL_23     : 1;
__REG32 DMA_CH_DB_MODE_SEL_24     : 1;
__REG32 DMA_CH_DB_MODE_SEL_25     : 1;
__REG32 DMA_CH_DB_MODE_SEL_26     : 1;
__REG32 DMA_CH_DB_MODE_SEL_27     : 1;
__REG32 DMA_CH_DB_MODE_SEL_28     : 1;
__REG32 DMA_CH_DB_MODE_SEL_29     : 1;
__REG32 DMA_CH_DB_MODE_SEL_30     : 1;
__REG32 DMA_CH_DB_MODE_SEL_31     : 1;
} __ipu_ch_db_mode_sel0_bits;

/* IPU channel double buffer mode select 1 register (IPU.IPU_CH_DB_MODE_SEL1) */
typedef struct{
__REG32 DMA_CH_DB_MODE_SEL_32     : 1;
__REG32 DMA_CH_DB_MODE_SEL_33     : 1;
__REG32 DMA_CH_DB_MODE_SEL_34     : 1;
__REG32 DMA_CH_DB_MODE_SEL_35     : 1;
__REG32 DMA_CH_DB_MODE_SEL_36     : 1;
__REG32 DMA_CH_DB_MODE_SEL_37     : 1;
__REG32 DMA_CH_DB_MODE_SEL_38     : 1;
__REG32 DMA_CH_DB_MODE_SEL_39     : 1;
__REG32 DMA_CH_DB_MODE_SEL_40     : 1;
__REG32 DMA_CH_DB_MODE_SEL_41     : 1;
__REG32 DMA_CH_DB_MODE_SEL_42     : 1;
__REG32 DMA_CH_DB_MODE_SEL_43     : 1;
__REG32 DMA_CH_DB_MODE_SEL_44     : 1;
__REG32 DMA_CH_DB_MODE_SEL_45     : 1;
__REG32 DMA_CH_DB_MODE_SEL_46     : 1;
__REG32 DMA_CH_DB_MODE_SEL_47     : 1;
__REG32 DMA_CH_DB_MODE_SEL_48     : 1;
__REG32 DMA_CH_DB_MODE_SEL_49     : 1;
__REG32 DMA_CH_DB_MODE_SEL_50     : 1;
__REG32 DMA_CH_DB_MODE_SEL_51     : 1;
__REG32 DMA_CH_DB_MODE_SEL_52     : 1;
__REG32 DMA_CH_DB_MODE_SEL_53     : 1;
__REG32 DMA_CH_DB_MODE_SEL_54     : 1;
__REG32 DMA_CH_DB_MODE_SEL_55     : 1;
__REG32 DMA_CH_DB_MODE_SEL_56     : 1;
__REG32 DMA_CH_DB_MODE_SEL_57     : 1;
__REG32 DMA_CH_DB_MODE_SEL_58     : 1;
__REG32 DMA_CH_DB_MODE_SEL_59     : 1;
__REG32 DMA_CH_DB_MODE_SEL_60     : 1;
__REG32 DMA_CH_DB_MODE_SEL_61     : 1;
__REG32 DMA_CH_DB_MODE_SEL_62     : 1;
__REG32 DMA_CH_DB_MODE_SEL_63     : 1;
} __ipu_ch_db_mode_sel1_bits;

/* IPU Alternate Channel Double Buffer Mode Select 0 Register (IPU.IPU_ALT_CH_DB_MODE_SEL0) */
typedef struct{
__REG32 DMA_CH_ALT_DB_MODE_SEL_0  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_1  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_2  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_3  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_4  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_5  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_6  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_7  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_8  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_9  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_10 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_11 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_12 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_13 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_14 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_15 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_16 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_17 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_18 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_19 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_20 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_21 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_22 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_23 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_24 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_25 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_26 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_27 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_28 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_29 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_30 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_31 : 1;
} __ipu_alt_ch_db_mode_sel0_bits;

/* IPU Alternate Channel Double Buffer Mode Select 1 Register (IPU.IPU_ALT_CH_DB_MODE_SEL1) */
typedef struct{
__REG32 DMA_CH_ALT_DB_MODE_SEL_32 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_33 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_34 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_35 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_36 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_37 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_38 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_39 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_40 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_41 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_42 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_43 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_44 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_45 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_46 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_47 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_48 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_49 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_50 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_51 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_52 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_53 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_54 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_55 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_56 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_57 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_58 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_59 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_60 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_61 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_62 : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_63 : 1;
} __ipu_alt_ch_db_mode_sel1_bits;

/* IPU Channel Triple Buffer Mode Select 0 Register(IPU.IPU_CH_TRB_MODE_SEL0)*/
typedef struct{
__REG32 DMA_CH_TRB_MODE_SEL_0     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_1     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_2     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_3     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_4     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_5     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_6     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_7     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_8     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_9     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_10    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_11    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_12    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_13    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_14    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_15    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_16    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_17    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_18    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_19    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_20    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_21    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_22    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_23    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_24    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_25    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_26    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_27    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_28    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_29    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_30    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_31    : 1;
} __ipu_ch_trb_mode_sel0_bits;

/* IPU Interrupt Status Register 1 (IPU.IPU_INT_STAT_1) */
typedef struct{
__REG32 IDMAC_EOF_0               : 1;
__REG32 IDMAC_EOF_1               : 1;
__REG32 IDMAC_EOF_2               : 1;
__REG32 IDMAC_EOF_3               : 1;
__REG32 IDMAC_EOF_4               : 1;
__REG32 IDMAC_EOF_5               : 1;
__REG32 IDMAC_EOF_6               : 1;
__REG32 IDMAC_EOF_7               : 1;
__REG32 IDMAC_EOF_8               : 1;
__REG32 IDMAC_EOF_9               : 1;
__REG32 IDMAC_EOF_10              : 1;
__REG32 IDMAC_EOF_11              : 1;
__REG32 IDMAC_EOF_12              : 1;
__REG32 IDMAC_EOF_13              : 1;
__REG32 IDMAC_EOF_14              : 1;
__REG32 IDMAC_EOF_15              : 1;
__REG32 IDMAC_EOF_16              : 1;
__REG32 IDMAC_EOF_17              : 1;
__REG32 IDMAC_EOF_18              : 1;
__REG32 IDMAC_EOF_19              : 1;
__REG32 IDMAC_EOF_20              : 1;
__REG32 IDMAC_EOF_21              : 1;
__REG32 IDMAC_EOF_22              : 1;
__REG32 IDMAC_EOF_23              : 1;
__REG32 IDMAC_EOF_24              : 1;
__REG32 IDMAC_EOF_25              : 1;
__REG32 IDMAC_EOF_26              : 1;
__REG32 IDMAC_EOF_27              : 1;
__REG32 IDMAC_EOF_28              : 1;
__REG32 IDMAC_EOF_29              : 1;
__REG32 IDMAC_EOF_30              : 1;
__REG32 IDMAC_EOF_31              : 1;
} __ipu_int_stat_1_bits;

/* IPU Interrupt Status Register 2 (IPU.IPU_INT_STAT_2)*/
typedef struct{
__REG32 IDMAC_EOF_32              : 1;
__REG32 IDMAC_EOF_33              : 1;
__REG32 IDMAC_EOF_34              : 1;
__REG32 IDMAC_EOF_35              : 1;
__REG32 IDMAC_EOF_36              : 1;
__REG32 IDMAC_EOF_37              : 1;
__REG32 IDMAC_EOF_38              : 1;
__REG32 IDMAC_EOF_39              : 1;
__REG32 IDMAC_EOF_40              : 1;
__REG32 IDMAC_EOF_41              : 1;
__REG32 IDMAC_EOF_42              : 1;
__REG32 IDMAC_EOF_43              : 1;
__REG32 IDMAC_EOF_44              : 1;
__REG32 IDMAC_EOF_45              : 1;
__REG32 IDMAC_EOF_46              : 1;
__REG32 IDMAC_EOF_47              : 1;
__REG32 IDMAC_EOF_48              : 1;
__REG32 IDMAC_EOF_49              : 1;
__REG32 IDMAC_EOF_50              : 1;
__REG32 IDMAC_EOF_51              : 1;
__REG32 IDMAC_EOF_52              : 1;
__REG32 IDMAC_EOF_53              : 1;
__REG32 IDMAC_EOF_54              : 1;
__REG32 IDMAC_EOF_55              : 1;
__REG32 IDMAC_EOF_56              : 1;
__REG32 IDMAC_EOF_57              : 1;
__REG32 IDMAC_EOF_58              : 1;
__REG32 IDMAC_EOF_59              : 1;
__REG32 IDMAC_EOF_60              : 1;
__REG32 IDMAC_EOF_61              : 1;
__REG32 IDMAC_EOF_62              : 1;
__REG32 IDMAC_EOF_63              : 1;
} __ipu_int_stat_2_bits;

/* IPU Interrupt Status Register 3 (IPU.IPU_INT_STAT_3)*/
typedef struct{
__REG32 IDMAC_NFACK_0             : 1;
__REG32 IDMAC_NFACK_1             : 1;
__REG32 IDMAC_NFACK_2             : 1;
__REG32 IDMAC_NFACK_3             : 1;
__REG32 IDMAC_NFACK_4             : 1;
__REG32 IDMAC_NFACK_5             : 1;
__REG32 IDMAC_NFACK_6             : 1;
__REG32 IDMAC_NFACK_7             : 1;
__REG32 IDMAC_NFACK_8             : 1;
__REG32 IDMAC_NFACK_9             : 1;
__REG32 IDMAC_NFACK_10            : 1;
__REG32 IDMAC_NFACK_11            : 1;
__REG32 IDMAC_NFACK_12            : 1;
__REG32 IDMAC_NFACK_13            : 1;
__REG32 IDMAC_NFACK_14            : 1;
__REG32 IDMAC_NFACK_15            : 1;
__REG32 IDMAC_NFACK_16            : 1;
__REG32 IDMAC_NFACK_17            : 1;
__REG32 IDMAC_NFACK_18            : 1;
__REG32 IDMAC_NFACK_19            : 1;
__REG32 IDMAC_NFACK_20            : 1;
__REG32 IDMAC_NFACK_21            : 1;
__REG32 IDMAC_NFACK_22            : 1;
__REG32 IDMAC_NFACK_23            : 1;
__REG32 IDMAC_NFACK_24            : 1;
__REG32 IDMAC_NFACK_25            : 1;
__REG32 IDMAC_NFACK_26            : 1;
__REG32 IDMAC_NFACK_27            : 1;
__REG32 IDMAC_NFACK_28            : 1;
__REG32 IDMAC_NFACK_29            : 1;
__REG32 IDMAC_NFACK_30            : 1;
__REG32 IDMAC_NFACK_31            : 1;
} __ipu_int_stat_3_bits;

/* IPU Interrupt Status Register 4 (IPU.IPU_INT_STAT_4)*/
typedef struct{
__REG32 IDMAC_NFACK_32            : 1;
__REG32 IDMAC_NFACK_33            : 1;
__REG32 IDMAC_NFACK_34            : 1;
__REG32 IDMAC_NFACK_35            : 1;
__REG32 IDMAC_NFACK_36            : 1;
__REG32 IDMAC_NFACK_37            : 1;
__REG32 IDMAC_NFACK_38            : 1;
__REG32 IDMAC_NFACK_39            : 1;
__REG32 IDMAC_NFACK_40            : 1;
__REG32 IDMAC_NFACK_41            : 1;
__REG32 IDMAC_NFACK_42            : 1;
__REG32 IDMAC_NFACK_43            : 1;
__REG32 IDMAC_NFACK_44            : 1;
__REG32 IDMAC_NFACK_45            : 1;
__REG32 IDMAC_NFACK_46            : 1;
__REG32 IDMAC_NFACK_47            : 1;
__REG32 IDMAC_NFACK_48            : 1;
__REG32 IDMAC_NFACK_49            : 1;
__REG32 IDMAC_NFACK_50            : 1;
__REG32 IDMAC_NFACK_51            : 1;
__REG32 IDMAC_NFACK_52            : 1;
__REG32 IDMAC_NFACK_53            : 1;
__REG32 IDMAC_NFACK_54            : 1;
__REG32 IDMAC_NFACK_55            : 1;
__REG32 IDMAC_NFACK_56            : 1;
__REG32 IDMAC_NFACK_57            : 1;
__REG32 IDMAC_NFACK_58            : 1;
__REG32 IDMAC_NFACK_59            : 1;
__REG32 IDMAC_NFACK_60            : 1;
__REG32 IDMAC_NFACK_61            : 1;
__REG32 IDMAC_NFACK_62            : 1;
__REG32 IDMAC_NFACK_63            : 1;
} __ipu_int_stat_4_bits;

/* IPU Interrupt Status Register 5 (IPU.IPU_INT_STAT_5)*/
typedef struct{
__REG32 IDMAC_NFB4EOF_ERR_0       : 1;
__REG32 IDMAC_NFB4EOF_ERR_1       : 1;
__REG32 IDMAC_NFB4EOF_ERR_2       : 1;
__REG32 IDMAC_NFB4EOF_ERR_3       : 1;
__REG32 IDMAC_NFB4EOF_ERR_4       : 1;
__REG32 IDMAC_NFB4EOF_ERR_5       : 1;
__REG32 IDMAC_NFB4EOF_ERR_6       : 1;
__REG32 IDMAC_NFB4EOF_ERR_7       : 1;
__REG32 IDMAC_NFB4EOF_ERR_8       : 1;
__REG32 IDMAC_NFB4EOF_ERR_9       : 1;
__REG32 IDMAC_NFB4EOF_ERR_10      : 1;
__REG32 IDMAC_NFB4EOF_ERR_11      : 1;
__REG32 IDMAC_NFB4EOF_ERR_12      : 1;
__REG32 IDMAC_NFB4EOF_ERR_13      : 1;
__REG32 IDMAC_NFB4EOF_ERR_14      : 1;
__REG32 IDMAC_NFB4EOF_ERR_15      : 1;
__REG32 IDMAC_NFB4EOF_ERR_16      : 1;
__REG32 IDMAC_NFB4EOF_ERR_17      : 1;
__REG32 IDMAC_NFB4EOF_ERR_18      : 1;
__REG32 IDMAC_NFB4EOF_ERR_19      : 1;
__REG32 IDMAC_NFB4EOF_ERR_20      : 1;
__REG32 IDMAC_NFB4EOF_ERR_21      : 1;
__REG32 IDMAC_NFB4EOF_ERR_22      : 1;
__REG32 IDMAC_NFB4EOF_ERR_23      : 1;
__REG32 IDMAC_NFB4EOF_ERR_24      : 1;
__REG32 IDMAC_NFB4EOF_ERR_25      : 1;
__REG32 IDMAC_NFB4EOF_ERR_26      : 1;
__REG32 IDMAC_NFB4EOF_ERR_27      : 1;
__REG32 IDMAC_NFB4EOF_ERR_28      : 1;
__REG32 IDMAC_NFB4EOF_ERR_29      : 1;
__REG32 IDMAC_NFB4EOF_ERR_30      : 1;
__REG32 IDMAC_NFB4EOF_ERR_31      : 1;
} __ipu_int_stat_5_bits;

/* IPU Interrupt Status Register 6 (IPU.IPU_INT_STAT_6) */
typedef struct{
__REG32 IDMAC_NFB4EOF_ERR_32      : 1;
__REG32 IDMAC_NFB4EOF_ERR_33      : 1;
__REG32 IDMAC_NFB4EOF_ERR_34      : 1;
__REG32 IDMAC_NFB4EOF_ERR_35      : 1;
__REG32 IDMAC_NFB4EOF_ERR_36      : 1;
__REG32 IDMAC_NFB4EOF_ERR_37      : 1;
__REG32 IDMAC_NFB4EOF_ERR_38      : 1;
__REG32 IDMAC_NFB4EOF_ERR_39      : 1;
__REG32 IDMAC_NFB4EOF_ERR_40      : 1;
__REG32 IDMAC_NFB4EOF_ERR_41      : 1;
__REG32 IDMAC_NFB4EOF_ERR_42      : 1;
__REG32 IDMAC_NFB4EOF_ERR_43      : 1;
__REG32 IDMAC_NFB4EOF_ERR_44      : 1;
__REG32 IDMAC_NFB4EOF_ERR_45      : 1;
__REG32 IDMAC_NFB4EOF_ERR_46      : 1;
__REG32 IDMAC_NFB4EOF_ERR_47      : 1;
__REG32 IDMAC_NFB4EOF_ERR_48      : 1;
__REG32 IDMAC_NFB4EOF_ERR_49      : 1;
__REG32 IDMAC_NFB4EOF_ERR_50      : 1;
__REG32 IDMAC_NFB4EOF_ERR_51      : 1;
__REG32 IDMAC_NFB4EOF_ERR_52      : 1;
__REG32 IDMAC_NFB4EOF_ERR_53      : 1;
__REG32 IDMAC_NFB4EOF_ERR_54      : 1;
__REG32 IDMAC_NFB4EOF_ERR_55      : 1;
__REG32 IDMAC_NFB4EOF_ERR_56      : 1;
__REG32 IDMAC_NFB4EOF_ERR_57      : 1;
__REG32 IDMAC_NFB4EOF_ERR_58      : 1;
__REG32 IDMAC_NFB4EOF_ERR_59      : 1;
__REG32 IDMAC_NFB4EOF_ERR_60      : 1;
__REG32 IDMAC_NFB4EOF_ERR_61      : 1;
__REG32 IDMAC_NFB4EOF_ERR_62      : 1;
__REG32 IDMAC_NFB4EOF_ERR_63      : 1;
} __ipu_int_stat_6_bits;

/* IPU Interrupt Status Register 7 (IPU.IPU_INT_STAT_7) */
typedef struct{
__REG32 IDMAC_EOS_0               : 1;
__REG32 IDMAC_EOS_1               : 1;
__REG32 IDMAC_EOS_2               : 1;
__REG32 IDMAC_EOS_3               : 1;
__REG32 IDMAC_EOS_4               : 1;
__REG32 IDMAC_EOS_5               : 1;
__REG32 IDMAC_EOS_6               : 1;
__REG32 IDMAC_EOS_7               : 1;
__REG32 IDMAC_EOS_8               : 1;
__REG32 IDMAC_EOS_9               : 1;
__REG32 IDMAC_EOS_10              : 1;
__REG32 IDMAC_EOS_11              : 1;
__REG32 IDMAC_EOS_12              : 1;
__REG32 IDMAC_EOS_13              : 1;
__REG32 IDMAC_EOS_14              : 1;
__REG32 IDMAC_EOS_15              : 1;
__REG32 IDMAC_EOS_16              : 1;
__REG32 IDMAC_EOS_17              : 1;
__REG32 IDMAC_EOS_18              : 1;
__REG32 IDMAC_EOS_19              : 1;
__REG32 IDMAC_EOS_20              : 1;
__REG32 IDMAC_EOS_21              : 1;
__REG32 IDMAC_EOS_22              : 1;
__REG32 IDMAC_EOS_23              : 1;
__REG32 IDMAC_EOS_24              : 1;
__REG32 IDMAC_EOS_25              : 1;
__REG32 IDMAC_EOS_26              : 1;
__REG32 IDMAC_EOS_27              : 1;
__REG32 IDMAC_EOS_28              : 1;
__REG32 IDMAC_EOS_29              : 1;
__REG32 IDMAC_EOS_30              : 1;
__REG32 IDMAC_EOS_31              : 1;
} __ipu_int_stat_7_bits;

/* IPU Interrupt Status Register 8 (IPU.IPU_INT_STAT_8) */
typedef struct{
__REG32 IDMAC_EOS_32              : 1;
__REG32 IDMAC_EOS_33              : 1;
__REG32 IDMAC_EOS_34              : 1;
__REG32 IDMAC_EOS_35              : 1;
__REG32 IDMAC_EOS_36              : 1;
__REG32 IDMAC_EOS_37              : 1;
__REG32 IDMAC_EOS_38              : 1;
__REG32 IDMAC_EOS_39              : 1;
__REG32 IDMAC_EOS_40              : 1;
__REG32 IDMAC_EOS_41              : 1;
__REG32 IDMAC_EOS_42              : 1;
__REG32 IDMAC_EOS_43              : 1;
__REG32 IDMAC_EOS_44              : 1;
__REG32 IDMAC_EOS_45              : 1;
__REG32 IDMAC_EOS_46              : 1;
__REG32 IDMAC_EOS_47              : 1;
__REG32 IDMAC_EOS_48              : 1;
__REG32 IDMAC_EOS_49              : 1;
__REG32 IDMAC_EOS_50              : 1;
__REG32 IDMAC_EOS_51              : 1;
__REG32 IDMAC_EOS_52              : 1;
__REG32 IDMAC_EOS_53              : 1;
__REG32 IDMAC_EOS_54              : 1;
__REG32 IDMAC_EOS_55              : 1;
__REG32 IDMAC_EOS_56              : 1;
__REG32 IDMAC_EOS_57              : 1;
__REG32 IDMAC_EOS_58              : 1;
__REG32 IDMAC_EOS_59              : 1;
__REG32 IDMAC_EOS_60              : 1;
__REG32 IDMAC_EOS_61              : 1;
__REG32 IDMAC_EOS_62              : 1;
__REG32 IDMAC_EOS_63              : 1;
} __ipu_int_stat_8_bits;

/* IPU Interrupt Status Register 9 (IPU.IPU_INT_STAT_9) */
typedef struct{
__REG32 VDI_FIFO1_OVF             : 1;
__REG32                           :25;
__REG32 IC_BAYER_BUF_OVF          : 1;
__REG32 IC_ENC_BUF_OVF            : 1;
__REG32 IC_VF_BUF_OVF             : 1;
__REG32                           : 1;
__REG32 CSI0_PUPE                 : 1;
__REG32 CSI1_PUPE                 : 1;
} __ipu_int_stat_9_bits;

/* IPU Interrupt Status Register 10 (IPU.IPU_INT_STAT_10) */
typedef struct{
__REG32 SMFC0_FRM_LOST            : 1;
__REG32 SMFC1_FRM_LOST            : 1;
__REG32 SMFC2_FRM_LOST            : 1;
__REG32 SMFC3_FRM_LOST            : 1;
__REG32                           :12;
__REG32 DC_TEARING_ERR_1          : 1;
__REG32 DC_TEARING_ERR_2          : 1;
__REG32 DC_TEARING_ERR_6          : 1;
__REG32 DI0_SYNC_DISP_ERR         : 1;
__REG32 DI1_SYNC_DISP_ERR         : 1;
__REG32 DI0_TIME_OUT_ERR          : 1;
__REG32 DI1_TIME_OUT_ERR          : 1;
__REG32                           : 1;
__REG32 IC_VF_FRM_LOST_ERR        : 1;
__REG32 IC_ENC_FRM_LOST_ERR       : 1;
__REG32 IC_BAYER_FRM_LOST_ERR     : 1;
__REG32                           : 1;
__REG32 NON_PRIVILEGED_ACC_ERR    : 1;
__REG32 AXIW_ERR                  : 1;
__REG32 AXIR_ERR                  : 1;
__REG32                           : 1;
} __ipu_int_stat_10_bits;

/* IPU Interrupt Status Register 11 (IPU.IPU_INT_STAT_11) */
typedef struct{
__REG32 IDMAC_EOBND_0             : 1;
__REG32 IDMAC_EOBND_1             : 1;
__REG32 IDMAC_EOBND_2             : 1;
__REG32 IDMAC_EOBND_3             : 1;
__REG32 IDMAC_EOBND_4             : 1;
__REG32 IDMAC_EOBND_5             : 1;
__REG32 IDMAC_EOBND_6             : 1;
__REG32 IDMAC_EOBND_7             : 1;
__REG32 IDMAC_EOBND_8             : 1;
__REG32 IDMAC_EOBND_9             : 1;
__REG32 IDMAC_EOBND_10            : 1;
__REG32 IDMAC_EOBND_11            : 1;
__REG32 IDMAC_EOBND_12            : 1;
__REG32 IDMAC_EOBND_13            : 1;
__REG32 IDMAC_EOBND_14            : 1;
__REG32 IDMAC_EOBND_15            : 1;
__REG32 IDMAC_EOBND_16            : 1;
__REG32 IDMAC_EOBND_17            : 1;
__REG32 IDMAC_EOBND_18            : 1;
__REG32 IDMAC_EOBND_19            : 1;
__REG32 IDMAC_EOBND_20            : 1;
__REG32 IDMAC_EOBND_21            : 1;
__REG32 IDMAC_EOBND_22            : 1;
__REG32 IDMAC_EOBND_23            : 1;
__REG32 IDMAC_EOBND_24            : 1;
__REG32 IDMAC_EOBND_25            : 1;
__REG32 IDMAC_EOBND_26            : 1;
__REG32 IDMAC_EOBND_27            : 1;
__REG32 IDMAC_EOBND_28            : 1;
__REG32 IDMAC_EOBND_29            : 1;
__REG32 IDMAC_EOBND_30            : 1;
__REG32 IDMAC_EOBND_31            : 1;
} __ipu_int_stat_11_bits;

/* IPU Interrupt Status Register 12 (IPU.IPU_INT_STAT_12) */
typedef struct{
__REG32 IDMAC_EOBND_32            : 1;
__REG32 IDMAC_EOBND_33            : 1;
__REG32 IDMAC_EOBND_34            : 1;
__REG32 IDMAC_EOBND_35            : 1;
__REG32 IDMAC_EOBND_36            : 1;
__REG32 IDMAC_EOBND_37            : 1;
__REG32 IDMAC_EOBND_38            : 1;
__REG32 IDMAC_EOBND_39            : 1;
__REG32 IDMAC_EOBND_40            : 1;
__REG32 IDMAC_EOBND_41            : 1;
__REG32 IDMAC_EOBND_42            : 1;
__REG32 IDMAC_EOBND_43            : 1;
__REG32 IDMAC_EOBND_44            : 1;
__REG32 IDMAC_EOBND_45            : 1;
__REG32 IDMAC_EOBND_46            : 1;
__REG32 IDMAC_EOBND_47            : 1;
__REG32 IDMAC_EOBND_48            : 1;
__REG32 IDMAC_EOBND_49            : 1;
__REG32 IDMAC_EOBND_50            : 1;
__REG32 IDMAC_EOBND_51            : 1;
__REG32 IDMAC_EOBND_52            : 1;
__REG32 IDMAC_EOBND_53            : 1;
__REG32 IDMAC_EOBND_54            : 1;
__REG32 IDMAC_EOBND_55            : 1;
__REG32 IDMAC_EOBND_56            : 1;
__REG32 IDMAC_EOBND_57            : 1;
__REG32 IDMAC_EOBND_58            : 1;
__REG32 IDMAC_EOBND_59            : 1;
__REG32 IDMAC_EOBND_60            : 1;
__REG32 IDMAC_EOBND_61            : 1;
__REG32 IDMAC_EOBND_62            : 1;
__REG32 IDMAC_EOBND_63            : 1;
} __ipu_int_stat_12_bits;

/* IPU Interrupt Status Register 13 (IPU.IPU_INT_STAT_13) */
typedef struct{
__REG32 IDMAC_TH_0                : 1;
__REG32 IDMAC_TH_1                : 1;
__REG32 IDMAC_TH_2                : 1;
__REG32 IDMAC_TH_3                : 1;
__REG32 IDMAC_TH_4                : 1;
__REG32 IDMAC_TH_5                : 1;
__REG32 IDMAC_TH_6                : 1;
__REG32 IDMAC_TH_7                : 1;
__REG32 IDMAC_TH_8                : 1;
__REG32 IDMAC_TH_9                : 1;
__REG32 IDMAC_TH_10               : 1;
__REG32 IDMAC_TH_11               : 1;
__REG32 IDMAC_TH_12               : 1;
__REG32 IDMAC_TH_13               : 1;
__REG32 IDMAC_TH_14               : 1;
__REG32 IDMAC_TH_15               : 1;
__REG32 IDMAC_TH_16               : 1;
__REG32 IDMAC_TH_17               : 1;
__REG32 IDMAC_TH_18               : 1;
__REG32 IDMAC_TH_19               : 1;
__REG32 IDMAC_TH_20               : 1;
__REG32 IDMAC_TH_21               : 1;
__REG32 IDMAC_TH_22               : 1;
__REG32 IDMAC_TH_23               : 1;
__REG32 IDMAC_TH_24               : 1;
__REG32 IDMAC_TH_25               : 1;
__REG32 IDMAC_TH_26               : 1;
__REG32 IDMAC_TH_27               : 1;
__REG32 IDMAC_TH_28               : 1;
__REG32 IDMAC_TH_29               : 1;
__REG32 IDMAC_TH_30               : 1;
__REG32 IDMAC_TH_31               : 1;
} __ipu_int_stat_13_bits;

/* IPU Interrupt Status Register 14 (IPU.IPU_INT_STAT_14) */
typedef struct{
__REG32 IDMAC_TH_32               : 1;
__REG32 IDMAC_TH_33               : 1;
__REG32 IDMAC_TH_34               : 1;
__REG32 IDMAC_TH_35               : 1;
__REG32 IDMAC_TH_36               : 1;
__REG32 IDMAC_TH_37               : 1;
__REG32 IDMAC_TH_38               : 1;
__REG32 IDMAC_TH_39               : 1;
__REG32 IDMAC_TH_40               : 1;
__REG32 IDMAC_TH_41               : 1;
__REG32 IDMAC_TH_42               : 1;
__REG32 IDMAC_TH_43               : 1;
__REG32 IDMAC_TH_44               : 1;
__REG32 IDMAC_TH_45               : 1;
__REG32 IDMAC_TH_46               : 1;
__REG32 IDMAC_TH_47               : 1;
__REG32 IDMAC_TH_48               : 1;
__REG32 IDMAC_TH_49               : 1;
__REG32 IDMAC_TH_50               : 1;
__REG32 IDMAC_TH_51               : 1;
__REG32 IDMAC_TH_52               : 1;
__REG32 IDMAC_TH_53               : 1;
__REG32 IDMAC_TH_54               : 1;
__REG32 IDMAC_TH_55               : 1;
__REG32 IDMAC_TH_56               : 1;
__REG32 IDMAC_TH_57               : 1;
__REG32 IDMAC_TH_58               : 1;
__REG32 IDMAC_TH_59               : 1;
__REG32 IDMAC_TH_60               : 1;
__REG32 IDMAC_TH_61               : 1;
__REG32 IDMAC_TH_62               : 1;
__REG32 IDMAC_TH_63               : 1;
} __ipu_int_stat_14_bits;

/* IPU Interrupt Status Register 15 (IPU.IPU_INT_STAT_15) */
typedef struct{
__REG32 IPU_SNOOPING1_INT         : 1;
__REG32 IPU_SNOOPING2_INT         : 1;
__REG32 DP_SF_START               : 1;
__REG32 DP_SF_END                 : 1;
__REG32 DP_ASF_START              : 1;
__REG32 DP_ASF_END                : 1;
__REG32 DP_SF_BRAKE               : 1;
__REG32 DP_ASF_BRAKE              : 1;
__REG32 DC_FC_0                   : 1;
__REG32 DC_FC_1                   : 1;
__REG32 DC_FC_2                   : 1;
__REG32 DC_FC_3                   : 1;
__REG32 DC_FC_4                   : 1;
__REG32 DC_FC_6                   : 1;
__REG32 DI_VSYNC_PRE_0            : 1;
__REG32 DI_VSYNC_PRE_1            : 1;
__REG32 DC_DP_START               : 1;
__REG32 DC_ASYNC_STOP             : 1;
__REG32 DI0_CNT_EN_PRE_0          : 1;
__REG32 DI0_CNT_EN_PRE_1          : 1;
__REG32 DI0_CNT_EN_PRE_2          : 1;
__REG32 DI0_CNT_EN_PRE_3          : 1;
__REG32 DI0_CNT_EN_PRE_4          : 1;
__REG32 DI0_CNT_EN_PRE_5          : 1;
__REG32 DI0_CNT_EN_PRE_6          : 1;
__REG32 DI0_CNT_EN_PRE_7          : 1;
__REG32 DI0_CNT_EN_PRE_8          : 1;
__REG32 DI0_CNT_EN_PRE_9          : 1;
__REG32 DI0_CNT_EN_PRE_10         : 1;
__REG32 DI1_DISP_CLK_EN_PRE       : 1;
__REG32 DI1_CNT_EN_PRE_3          : 1;
__REG32 DI1_CNT_EN_PRE_8          : 1;
} __ipu_int_stat_15_bits;

/* IPU Current Buffer Register 0 (IPU.IPU_CUR_BUF_0) */
typedef struct{
__REG32 DMA_CH_CUR_BUF_0          : 1;
__REG32 DMA_CH_CUR_BUF_1          : 1;
__REG32 DMA_CH_CUR_BUF_2          : 1;
__REG32 DMA_CH_CUR_BUF_4          : 1;
__REG32 DMA_CH_CUR_BUF_3          : 1;
__REG32 DMA_CH_CUR_BUF_5          : 1;
__REG32 DMA_CH_CUR_BUF_6          : 1;
__REG32 DMA_CH_CUR_BUF_7          : 1;
__REG32 DMA_CH_CUR_BUF_8          : 1;
__REG32 DMA_CH_CUR_BUF_9          : 1;
__REG32 DMA_CH_CUR_BUF_10         : 1;
__REG32 DMA_CH_CUR_BUF_11         : 1;
__REG32 DMA_CH_CUR_BUF_12         : 1;
__REG32 DMA_CH_CUR_BUF_13         : 1;
__REG32 DMA_CH_CUR_BUF_14         : 1;
__REG32 DMA_CH_CUR_BUF_15         : 1;
__REG32 DMA_CH_CUR_BUF_16         : 1;
__REG32 DMA_CH_CUR_BUF_17         : 1;
__REG32 DMA_CH_CUR_BUF_18         : 1;
__REG32 DMA_CH_CUR_BUF_19         : 1;
__REG32 DMA_CH_CUR_BUF_20         : 1;
__REG32 DMA_CH_CUR_BUF_21         : 1;
__REG32 DMA_CH_CUR_BUF_22         : 1;
__REG32 DMA_CH_CUR_BUF_23         : 1;
__REG32 DMA_CH_CUR_BUF_24         : 1;
__REG32 DMA_CH_CUR_BUF_25         : 1;
__REG32 DMA_CH_CUR_BUF_26         : 1;
__REG32 DMA_CH_CUR_BUF_27         : 1;
__REG32 DMA_CH_CUR_BUF_28         : 1;
__REG32 DMA_CH_CUR_BUF_29         : 1;
__REG32 DMA_CH_CUR_BUF_30         : 1;
__REG32 DMA_CH_CUR_BUF_31         : 1;
} __ipu_cur_buf_0_bits;

/* IPU Current Buffer Register 1 (IPU.IPU_CUR_BUF_1) */
typedef struct{
__REG32 DMA_CH_CUR_BUF_32         : 1;
__REG32 DMA_CH_CUR_BUF_33         : 1;
__REG32 DMA_CH_CUR_BUF_34         : 1;
__REG32 DMA_CH_CUR_BUF_35         : 1;
__REG32 DMA_CH_CUR_BUF_36         : 1;
__REG32 DMA_CH_CUR_BUF_37         : 1;
__REG32 DMA_CH_CUR_BUF_38         : 1;
__REG32 DMA_CH_CUR_BUF_39         : 1;
__REG32 DMA_CH_CUR_BUF_40         : 1;
__REG32 DMA_CH_CUR_BUF_41         : 1;
__REG32 DMA_CH_CUR_BUF_42         : 1;
__REG32 DMA_CH_CUR_BUF_43         : 1;
__REG32 DMA_CH_CUR_BUF_44         : 1;
__REG32 DMA_CH_CUR_BUF_45         : 1;
__REG32 DMA_CH_CUR_BUF_46         : 1;
__REG32 DMA_CH_CUR_BUF_47         : 1;
__REG32 DMA_CH_CUR_BUF_48         : 1;
__REG32 DMA_CH_CUR_BUF_49         : 1;
__REG32 DMA_CH_CUR_BUF_50         : 1;
__REG32 DMA_CH_CUR_BUF_51         : 1;
__REG32 DMA_CH_CUR_BUF_52         : 1;
__REG32 DMA_CH_CUR_BUF_53         : 1;
__REG32 DMA_CH_CUR_BUF_54         : 1;
__REG32 DMA_CH_CUR_BUF_55         : 1;
__REG32 DMA_CH_CUR_BUF_56         : 1;
__REG32 DMA_CH_CUR_BUF_57         : 1;
__REG32 DMA_CH_CUR_BUF_58         : 1;
__REG32 DMA_CH_CUR_BUF_59         : 1;
__REG32 DMA_CH_CUR_BUF_60         : 1;
__REG32 DMA_CH_CUR_BUF_61         : 1;
__REG32 DMA_CH_CUR_BUF_62         : 1;
__REG32 DMA_CH_CUR_BUF_63         : 1;
} __ipu_cur_buf_1_bits;

/* IPU Alternate Current Buffer Register 0 (IPU.IPU_ALT_CUR_BUF_0) */
typedef struct{
__REG32 DMA_CH_ALT_CUR_BUF_0      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_1      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_2      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_3      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_4      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_5      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_6      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_7      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_8      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_9      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_10     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_11     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_12     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_13     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_14     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_15     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_16     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_17     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_18     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_19     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_20     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_21     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_22     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_23     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_24     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_25     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_26     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_27     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_28     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_29     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_30     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_31     : 1;
} __ipu_alt_cur_buf_0_bits;

/* IPU Alternate Current Buffer Register 1 (IPU.IPU_ALT_CUR_BUF_1) */
typedef struct{
__REG32 DMA_CH_ALT_CUR_BUF_32     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_33     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_34     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_35     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_36     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_37     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_38     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_39     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_40     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_41     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_42     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_43     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_44     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_45     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_46     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_47     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_48     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_49     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_50     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_51     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_52     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_53     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_54     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_55     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_56     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_57     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_58     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_59     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_60     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_61     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_62     : 1;
__REG32 DMA_CH_ALT_CUR_BUF_63     : 1;
} __ipu_alt_cur_buf_1_bits;

/* IPU Shadow Register Memory Status Register (IPU.IPU_SRM_STAT) */
typedef struct{
__REG32 DP_S_SRM_STAT             : 1;
__REG32 DP_A0_SRM_STAT            : 1;
__REG32 DP_A1_SRM_STAT            : 1;
__REG32                           : 1;
__REG32 DC_2_SRM_STAT             : 1;
__REG32 DC_6_SRM_STAT             : 1;
__REG32 CSI0_SRM_STAT             : 1;
__REG32 CSI1_SRM_STAT             : 1;
__REG32 DI0_SRM_STAT              : 1;
__REG32 DI1_SRM_STAT              : 1;
__REG32                           :22;
} __ipu_srm_stat_bits;

/* IPU Processing Tasks Status Register (IPU.IPU_PROC_TASKS_STAT) */
typedef struct{
__REG32 ENC_TSTAT                 : 2;
__REG32 VF_TSTAT                  : 2;
__REG32 PP_TSTAT                  : 2;
__REG32 ENC_ROT_TSTAT             : 2;
__REG32 VF_ROT_TSTAT              : 2;
__REG32 PP_ROT_TSTAT              : 2;
__REG32 MEM2PRP_TSTAT             : 3;
__REG32                           :17;
} __ipu_proc_tasks_stat_bits;

/* IPU Display Tasks Status Register (IPU.IPU_DISP_TASKS_STAT) */
typedef struct{
__REG32 DP_ASYNC_STAT             : 3;
__REG32 DP_ASYNC_CUR_FLOW         : 1;
__REG32 DC_ASYNC1_STAT            : 2;
__REG32                           : 2;
__REG32 DC_ASYNC2_STAT            : 3;
__REG32 DC_ASYNC2_CUR_FLOW        : 1;
__REG32                           :20;
} __ipu_disp_tasks_stat_bits;

/* IPU Triple Current Buffer Register 0 (IPU.IPU_TRIPLE_CUR_BUF_0) */
typedef struct{
__REG32 DMA_CH_TRIPLE_CUR_BUF_0   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_1   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_2   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_3   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_4   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_5   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_6   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_7   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_8   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_9   : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_10  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_11  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_12  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_13  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_14  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_15  : 2;
} __ipu_triple_cur_buf_0_bits;

/* IPU Triple Current Buffer Register 1 (IPU.IPU_TRIPLE_CUR_BUF_1) */
typedef struct{
__REG32 DMA_CH_TRIPLE_CUR_BUF_16  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_17  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_18  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_19  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_20  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_21  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_22  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_23  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_24  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_25  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_26  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_27  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_28  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_29  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_30  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_31  : 2;
} __ipu_triple_cur_buf_1_bits;

/* IPU Triple Current Buffer Register 2 (IPU.IPU_TRIPLE_CUR_BUF_2) */
typedef struct{
__REG32 DMA_CH_TRIPLE_CUR_BUF_32  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_33  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_34  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_35  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_36  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_37  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_38  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_39  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_40  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_41  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_42  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_43  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_44  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_45  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_46  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_47  : 2;
} __ipu_triple_cur_buf_2_bits;

/* IPU Triple Current Buffer Register 3 (IPU.IPU_TRIPLE_CUR_BUF_3) */
typedef struct{
__REG32 DMA_CH_TRIPLE_CUR_BUF_48  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_49  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_50  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_51  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_52  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_53  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_54  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_55  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_56  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_57  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_58  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_59  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_60  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_61  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_62  : 2;
__REG32 DMA_CH_TRIPLE_CUR_BUF_63  : 2;
} __ipu_triple_cur_buf_3_bits;

/* IPU Channels Buffer 0 Ready 0 Register (IPU.IPU_CH_BUF0_RDY0) */
typedef struct{
__REG32 DMA_CH_BUF0_RDY_0         : 1;
__REG32 DMA_CH_BUF0_RDY_1         : 1;
__REG32 DMA_CH_BUF0_RDY_2         : 1;
__REG32 DMA_CH_BUF0_RDY_3         : 1;
__REG32 DMA_CH_BUF0_RDY_4         : 1;
__REG32 DMA_CH_BUF0_RDY_5         : 1;
__REG32 DMA_CH_BUF0_RDY_6         : 1;
__REG32 DMA_CH_BUF0_RDY_7         : 1;
__REG32 DMA_CH_BUF0_RDY_8         : 1;
__REG32 DMA_CH_BUF0_RDY_9         : 1;
__REG32 DMA_CH_BUF0_RDY_10        : 1;
__REG32 DMA_CH_BUF0_RDY_11        : 1;
__REG32 DMA_CH_BUF0_RDY_12        : 1;
__REG32 DMA_CH_BUF0_RDY_13        : 1;
__REG32 DMA_CH_BUF0_RDY_14        : 1;
__REG32 DMA_CH_BUF0_RDY_15        : 1;
__REG32 DMA_CH_BUF0_RDY_16        : 1;
__REG32 DMA_CH_BUF0_RDY_17        : 1;
__REG32 DMA_CH_BUF0_RDY_18        : 1;
__REG32 DMA_CH_BUF0_RDY_19        : 1;
__REG32 DMA_CH_BUF0_RDY_20        : 1;
__REG32 DMA_CH_BUF0_RDY_21        : 1;
__REG32 DMA_CH_BUF0_RDY_22        : 1;
__REG32 DMA_CH_BUF0_RDY_23        : 1;
__REG32 DMA_CH_BUF0_RDY_24        : 1;
__REG32 DMA_CH_BUF0_RDY_25        : 1;
__REG32 DMA_CH_BUF0_RDY_26        : 1;
__REG32 DMA_CH_BUF0_RDY_27        : 1;
__REG32 DMA_CH_BUF0_RDY_28        : 1;
__REG32 DMA_CH_BUF0_RDY_29        : 1;
__REG32 DMA_CH_BUF0_RDY_30        : 1;
__REG32 DMA_CH_BUF0_RDY_31        : 1;
} __ipu_ch_buf0_rdy0_bits;

/* IPU Channels Buffer 0 Ready 1 Register (IPU.IPU_CH_BUF0_RDY1) */
typedef struct{
__REG32 DMA_CH_BUF0_RDY_32        : 1;
__REG32 DMA_CH_BUF0_RDY_33        : 1;
__REG32 DMA_CH_BUF0_RDY_34        : 1;
__REG32 DMA_CH_BUF0_RDY_35        : 1;
__REG32 DMA_CH_BUF0_RDY_36        : 1;
__REG32 DMA_CH_BUF0_RDY_37        : 1;
__REG32 DMA_CH_BUF0_RDY_38        : 1;
__REG32 DMA_CH_BUF0_RDY_39        : 1;
__REG32 DMA_CH_BUF0_RDY_40        : 1;
__REG32 DMA_CH_BUF0_RDY_41        : 1;
__REG32 DMA_CH_BUF0_RDY_42        : 1;
__REG32 DMA_CH_BUF0_RDY_43        : 1;
__REG32 DMA_CH_BUF0_RDY_44        : 1;
__REG32 DMA_CH_BUF0_RDY_45        : 1;
__REG32 DMA_CH_BUF0_RDY_46        : 1;
__REG32 DMA_CH_BUF0_RDY_47        : 1;
__REG32 DMA_CH_BUF0_RDY_48        : 1;
__REG32 DMA_CH_BUF0_RDY_49        : 1;
__REG32 DMA_CH_BUF0_RDY_50        : 1;
__REG32 DMA_CH_BUF0_RDY_51        : 1;
__REG32 DMA_CH_BUF0_RDY_52        : 1;
__REG32 DMA_CH_BUF0_RDY_53        : 1;
__REG32 DMA_CH_BUF0_RDY_54        : 1;
__REG32 DMA_CH_BUF0_RDY_55        : 1;
__REG32 DMA_CH_BUF0_RDY_56        : 1;
__REG32 DMA_CH_BUF0_RDY_57        : 1;
__REG32 DMA_CH_BUF0_RDY_58        : 1;
__REG32 DMA_CH_BUF0_RDY_59        : 1;
__REG32 DMA_CH_BUF0_RDY_60        : 1;
__REG32 DMA_CH_BUF0_RDY_61        : 1;
__REG32 DMA_CH_BUF0_RDY_62        : 1;
__REG32 DMA_CH_BUF0_RDY_63        : 1;
} __ipu_ch_buf0_rdy1_bits;

/* IPU Channels Buffer 1 Ready 0 Register (IPU.IPU_CH_BUF1_RDY0) */
typedef struct{
__REG32 DMA_CH_BUF1_RDY_0         : 1;
__REG32 DMA_CH_BUF1_RDY_1         : 1;
__REG32 DMA_CH_BUF1_RDY_2         : 1;
__REG32 DMA_CH_BUF1_RDY_3         : 1;
__REG32 DMA_CH_BUF1_RDY_4         : 1;
__REG32 DMA_CH_BUF1_RDY_5         : 1;
__REG32 DMA_CH_BUF1_RDY_6         : 1;
__REG32 DMA_CH_BUF1_RDY_7         : 1;
__REG32 DMA_CH_BUF1_RDY_8         : 1;
__REG32 DMA_CH_BUF1_RDY_9         : 1;
__REG32 DMA_CH_BUF1_RDY_10        : 1;
__REG32 DMA_CH_BUF1_RDY_11        : 1;
__REG32 DMA_CH_BUF1_RDY_12        : 1;
__REG32 DMA_CH_BUF1_RDY_13        : 1;
__REG32 DMA_CH_BUF1_RDY_14        : 1;
__REG32 DMA_CH_BUF1_RDY_15        : 1;
__REG32 DMA_CH_BUF1_RDY_16        : 1;
__REG32 DMA_CH_BUF1_RDY_17        : 1;
__REG32 DMA_CH_BUF1_RDY_18        : 1;
__REG32 DMA_CH_BUF1_RDY_19        : 1;
__REG32 DMA_CH_BUF1_RDY_20        : 1;
__REG32 DMA_CH_BUF1_RDY_21        : 1;
__REG32 DMA_CH_BUF1_RDY_22        : 1;
__REG32 DMA_CH_BUF1_RDY_23        : 1;
__REG32 DMA_CH_BUF1_RDY_24        : 1;
__REG32 DMA_CH_BUF1_RDY_25        : 1;
__REG32 DMA_CH_BUF1_RDY_26        : 1;
__REG32 DMA_CH_BUF1_RDY_27        : 1;
__REG32 DMA_CH_BUF1_RDY_28        : 1;
__REG32 DMA_CH_BUF1_RDY_29        : 1;
__REG32 DMA_CH_BUF1_RDY_30        : 1;
__REG32 DMA_CH_BUF1_RDY_31        : 1;
} __ipu_ch_buf1_rdy0_bits;

/* IPU Channels Buffer 1 Ready 1 Register (IPU.IPU_CH_BUF1_RDY1) */
typedef struct{
__REG32 DMA_CH_BUF1_RDY_32        : 1;
__REG32 DMA_CH_BUF1_RDY_33        : 1;
__REG32 DMA_CH_BUF1_RDY_34        : 1;
__REG32 DMA_CH_BUF1_RDY_35        : 1;
__REG32 DMA_CH_BUF1_RDY_36        : 1;
__REG32 DMA_CH_BUF1_RDY_37        : 1;
__REG32 DMA_CH_BUF1_RDY_38        : 1;
__REG32 DMA_CH_BUF1_RDY_39        : 1;
__REG32 DMA_CH_BUF1_RDY_40        : 1;
__REG32 DMA_CH_BUF1_RDY_41        : 1;
__REG32 DMA_CH_BUF1_RDY_42        : 1;
__REG32 DMA_CH_BUF1_RDY_43        : 1;
__REG32 DMA_CH_BUF1_RDY_44        : 1;
__REG32 DMA_CH_BUF1_RDY_45        : 1;
__REG32 DMA_CH_BUF1_RDY_46        : 1;
__REG32 DMA_CH_BUF1_RDY_47        : 1;
__REG32 DMA_CH_BUF1_RDY_48        : 1;
__REG32 DMA_CH_BUF1_RDY_49        : 1;
__REG32 DMA_CH_BUF1_RDY_50        : 1;
__REG32 DMA_CH_BUF1_RDY_51        : 1;
__REG32 DMA_CH_BUF1_RDY_52        : 1;
__REG32 DMA_CH_BUF1_RDY_53        : 1;
__REG32 DMA_CH_BUF1_RDY_54        : 1;
__REG32 DMA_CH_BUF1_RDY_55        : 1;
__REG32 DMA_CH_BUF1_RDY_56        : 1;
__REG32 DMA_CH_BUF1_RDY_57        : 1;
__REG32 DMA_CH_BUF1_RDY_58        : 1;
__REG32 DMA_CH_BUF1_RDY_59        : 1;
__REG32 DMA_CH_BUF1_RDY_60        : 1;
__REG32 DMA_CH_BUF1_RDY_61        : 1;
__REG32 DMA_CH_BUF1_RDY_62        : 1;
__REG32 DMA_CH_BUF1_RDY_63        : 1;
} __ipu_ch_buf1_rdy1_bits;

/* IPU Alternate Channels Buffer 0 Ready 0 Register(IPU.IPU_ALT_CH_BUF0_RDY0) */
typedef struct{
__REG32 DMA_CH_ALT_BUF0_RDY_0     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_1     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_2     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_3     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_4     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_5     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_6     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_7     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_8     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_9     : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_10    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_11    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_12    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_13    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_14    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_15    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_16    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_17    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_18    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_19    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_20    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_21    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_22    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_23    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_24    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_25    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_26    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_27    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_28    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_29    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_30    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_31    : 1;
} __ipu_alt_ch_buf0_rdy0_bits;

/* IPU Alternate Channels Buffer 0 Ready 1 Register(IPU.IPU_ALT_CH_BUF0_RDY1) */
typedef struct{
__REG32 DMA_CH_ALT_BUF0_RDY_32    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_33    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_34    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_35    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_36    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_37    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_38    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_39    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_40    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_41    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_42    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_43    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_44    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_45    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_46    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_47    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_48    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_49    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_50    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_51    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_52    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_53    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_54    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_55    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_56    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_57    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_58    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_59    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_60    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_61    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_62    : 1;
__REG32 DMA_CH_ALT_BUF0_RDY_63    : 1;
} __ipu_alt_ch_buf0_rdy1_bits;

/* IPU Alternate Channels Buffer 1 Ready 0 Register(IPU.IPU_ALT_CH_BUF1_RDY0) */
typedef struct{
__REG32 DMA_CH_ALT_BUF1_RDY_0     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_1     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_2     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_3     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_4     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_5     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_6     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_7     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_8     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_9     : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_10    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_11    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_12    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_13    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_14    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_15    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_16    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_17    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_18    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_19    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_20    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_21    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_22    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_23    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_24    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_25    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_26    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_27    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_28    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_29    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_30    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_31    : 1;
} __ipu_alt_ch_buf1_rdy0_bits;

/* IPU Alternate Channels Buffer 1 Ready 1 Register(IPU.IPU_ALT_CH_BUF1_RDY1) */
typedef struct{
__REG32 DMA_CH_ALT_BUF1_RDY_32    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_33    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_34    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_35    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_36    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_37    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_38    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_39    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_40    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_41    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_42    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_43    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_44    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_45    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_46    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_47    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_48    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_49    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_50    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_51    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_52    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_53    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_54    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_55    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_56    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_57    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_58    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_59    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_60    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_61    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_62    : 1;
__REG32 DMA_CH_ALT_BUF1_RDY_63    : 1;
} __ipu_alt_ch_buf1_rdy1_bits;

/* IPU Channels Buffer 2 Ready 0 Register (IPU.IPU_CH_BUF2_RDY0) */
typedef struct{
__REG32 DMA_CH_ALT_BUF2_RDY_0     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_1     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_2     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_3     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_4     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_5     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_6     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_7     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_8     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_9     : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_10    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_11    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_12    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_13    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_14    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_15    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_16    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_17    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_18    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_19    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_20    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_21    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_22    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_23    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_24    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_25    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_26    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_27    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_28    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_29    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_30    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_31    : 1;
} __ipu_ch_buf2_rdy0_bits;

/* IPU Channels Buffer 2 Ready 1 Register (IPU.IPU_CH_BUF2_RDY1) */
typedef struct{
__REG32 DMA_CH_ALT_BUF2_RDY_32    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_33    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_34    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_35    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_36    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_37    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_38    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_39    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_40    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_41    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_42    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_43    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_44    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_45    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_46    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_47    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_48    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_49    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_50    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_51    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_52    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_53    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_54    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_55    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_56    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_57    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_58    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_59    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_60    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_61    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_62    : 1;
__REG32 DMA_CH_ALT_BUF2_RDY_63    : 1;
} __ipu_ch_buf2_rdy1_bits;

/* IDMAC Configuration Register (IPU_IDMAC_CONF) */
typedef struct{
__REG32 MAX_REQ_READ              : 3;
__REG32 WIDPT                     : 2;
__REG32 RDI                       : 1;
__REG32                           :10;
__REG32 P_ENDIAN                  : 1;
__REG32 USED_BUFS_MAX_W           : 3;
__REG32 USED_BUFS_EN_W            : 1;
__REG32 USED_BUFS_MAX_R           : 4;
__REG32 USED_BUFS_EN_R            : 1;
__REG32                           : 6;
} __ipu_idmac_conf_bits;

/* IDMAC Channel Enable 1 Register (IPU_IDMAC_CH_EN_1) */
typedef struct{
__REG32 IDMAC_CH_EN_0             : 1;
__REG32 IDMAC_CH_EN_1             : 1;
__REG32 IDMAC_CH_EN_2             : 1;
__REG32 IDMAC_CH_EN_3             : 1;
__REG32 IDMAC_CH_EN_4             : 1;
__REG32 IDMAC_CH_EN_5             : 1;
__REG32 IDMAC_CH_EN_6             : 1;
__REG32 IDMAC_CH_EN_7             : 1;
__REG32 IDMAC_CH_EN_8             : 1;
__REG32 IDMAC_CH_EN_9             : 1;
__REG32 IDMAC_CH_EN_10            : 1;
__REG32 IDMAC_CH_EN_11            : 1;
__REG32 IDMAC_CH_EN_12            : 1;
__REG32 IDMAC_CH_EN_13            : 1;
__REG32 IDMAC_CH_EN_14            : 1;
__REG32 IDMAC_CH_EN_15            : 1;
__REG32 IDMAC_CH_EN_16            : 1;
__REG32 IDMAC_CH_EN_17            : 1;
__REG32 IDMAC_CH_EN_18            : 1;
__REG32 IDMAC_CH_EN_19            : 1;
__REG32 IDMAC_CH_EN_20            : 1;
__REG32 IDMAC_CH_EN_21            : 1;
__REG32 IDMAC_CH_EN_22            : 1;
__REG32 IDMAC_CH_EN_23            : 1;
__REG32 IDMAC_CH_EN_24            : 1;
__REG32 IDMAC_CH_EN_25            : 1;
__REG32 IDMAC_CH_EN_26            : 1;
__REG32 IDMAC_CH_EN_27            : 1;
__REG32 IDMAC_CH_EN_28            : 1;
__REG32 IDMAC_CH_EN_29            : 1;
__REG32 IDMAC_CH_EN_30            : 1;
__REG32 IDMAC_CH_EN_31            : 1;
} __ipu_idmac_ch_en_1_bits;

/* IDMAC Channel Enable 2 Register (IPU_IDMAC_CH_EN_2) */
typedef struct{
__REG32 IDMAC_CH_EN_32            : 1;
__REG32 IDMAC_CH_EN_33            : 1;
__REG32 IDMAC_CH_EN_34            : 1;
__REG32 IDMAC_CH_EN_35            : 1;
__REG32 IDMAC_CH_EN_36            : 1;
__REG32 IDMAC_CH_EN_37            : 1;
__REG32 IDMAC_CH_EN_38            : 1;
__REG32 IDMAC_CH_EN_39            : 1;
__REG32 IDMAC_CH_EN_40            : 1;
__REG32 IDMAC_CH_EN_41            : 1;
__REG32 IDMAC_CH_EN_42            : 1;
__REG32 IDMAC_CH_EN_43            : 1;
__REG32 IDMAC_CH_EN_44            : 1;
__REG32 IDMAC_CH_EN_45            : 1;
__REG32 IDMAC_CH_EN_46            : 1;
__REG32 IDMAC_CH_EN_47            : 1;
__REG32 IDMAC_CH_EN_48            : 1;
__REG32 IDMAC_CH_EN_49            : 1;
__REG32 IDMAC_CH_EN_50            : 1;
__REG32 IDMAC_CH_EN_51            : 1;
__REG32 IDMAC_CH_EN_52            : 1;
__REG32 IDMAC_CH_EN_53            : 1;
__REG32 IDMAC_CH_EN_54            : 1;
__REG32 IDMAC_CH_EN_55            : 1;
__REG32 IDMAC_CH_EN_56            : 1;
__REG32 IDMAC_CH_EN_57            : 1;
__REG32 IDMAC_CH_EN_58            : 1;
__REG32 IDMAC_CH_EN_59            : 1;
__REG32 IDMAC_CH_EN_60            : 1;
__REG32 IDMAC_CH_EN_61            : 1;
__REG32 IDMAC_CH_EN_62            : 1;
__REG32 IDMAC_CH_EN_63            : 1;
} __ipu_idmac_ch_en_2_bits;

/* IDMAC Separate Alpha Indication Register (IPU_IDMAC_SEP_ALPHA) */
typedef struct{
__REG32 IDMAC_SEP_AL_0            : 1;
__REG32 IDMAC_SEP_AL_1            : 1;
__REG32 IDMAC_SEP_AL_2            : 1;
__REG32 IDMAC_SEP_AL_3            : 1;
__REG32 IDMAC_SEP_AL_4            : 1;
__REG32 IDMAC_SEP_AL_5            : 1;
__REG32 IDMAC_SEP_AL_6            : 1;
__REG32 IDMAC_SEP_AL_7            : 1;
__REG32 IDMAC_SEP_AL_8            : 1;
__REG32 IDMAC_SEP_AL_9            : 1;
__REG32 IDMAC_SEP_AL_10           : 1;
__REG32 IDMAC_SEP_AL_11           : 1;
__REG32 IDMAC_SEP_AL_12           : 1;
__REG32 IDMAC_SEP_AL_13           : 1;
__REG32 IDMAC_SEP_AL_14           : 1;
__REG32 IDMAC_SEP_AL_15           : 1;
__REG32 IDMAC_SEP_AL_16           : 1;
__REG32 IDMAC_SEP_AL_17           : 1;
__REG32 IDMAC_SEP_AL_18           : 1;
__REG32 IDMAC_SEP_AL_19           : 1;
__REG32 IDMAC_SEP_AL_20           : 1;
__REG32 IDMAC_SEP_AL_21           : 1;
__REG32 IDMAC_SEP_AL_22           : 1;
__REG32 IDMAC_SEP_AL_23           : 1;
__REG32 IDMAC_SEP_AL_24           : 1;
__REG32 IDMAC_SEP_AL_25           : 1;
__REG32 IDMAC_SEP_AL_26           : 1;
__REG32 IDMAC_SEP_AL_27           : 1;
__REG32 IDMAC_SEP_AL_28           : 1;
__REG32 IDMAC_SEP_AL_29           : 1;
__REG32 IDMAC_SEP_AL_30           : 1;
__REG32 IDMAC_SEP_AL_31           : 1;
} __ipu_idmac_sep_alpha_bits;

/* IDMAC Alternate Separate Alpha Indication Register (IPU_IDMAC_ALT_SEP_ALPHA) */
typedef struct{
__REG32 IDMAC_ALT_SEP_AL_0        : 1;
__REG32 IDMAC_ALT_SEP_AL_1        : 1;
__REG32 IDMAC_ALT_SEP_AL_2        : 1;
__REG32 IDMAC_ALT_SEP_AL_3        : 1;
__REG32 IDMAC_ALT_SEP_AL_4        : 1;
__REG32 IDMAC_ALT_SEP_AL_5        : 1;
__REG32 IDMAC_ALT_SEP_AL_6        : 1;
__REG32 IDMAC_ALT_SEP_AL_7        : 1;
__REG32 IDMAC_ALT_SEP_AL_8        : 1;
__REG32 IDMAC_ALT_SEP_AL_9        : 1;
__REG32 IDMAC_ALT_SEP_AL_10       : 1;
__REG32 IDMAC_ALT_SEP_AL_11       : 1;
__REG32 IDMAC_ALT_SEP_AL_12       : 1;
__REG32 IDMAC_ALT_SEP_AL_13       : 1;
__REG32 IDMAC_ALT_SEP_AL_14       : 1;
__REG32 IDMAC_ALT_SEP_AL_15       : 1;
__REG32 IDMAC_ALT_SEP_AL_16       : 1;
__REG32 IDMAC_ALT_SEP_AL_17       : 1;
__REG32 IDMAC_ALT_SEP_AL_18       : 1;
__REG32 IDMAC_ALT_SEP_AL_19       : 1;
__REG32 IDMAC_ALT_SEP_AL_20       : 1;
__REG32 IDMAC_ALT_SEP_AL_21       : 1;
__REG32 IDMAC_ALT_SEP_AL_22       : 1;
__REG32 IDMAC_ALT_SEP_AL_23       : 1;
__REG32 IDMAC_ALT_SEP_AL_24       : 1;
__REG32 IDMAC_ALT_SEP_AL_25       : 1;
__REG32 IDMAC_ALT_SEP_AL_26       : 1;
__REG32 IDMAC_ALT_SEP_AL_27       : 1;
__REG32 IDMAC_ALT_SEP_AL_28       : 1;
__REG32 IDMAC_ALT_SEP_AL_29       : 1;
__REG32 IDMAC_ALT_SEP_AL_30       : 1;
__REG32 IDMAC_ALT_SEP_AL_31       : 1;
} __ipu_idmac_alt_sep_alpha_bits;

/* IDMAC Channel Priority 1 Register (IPU_IDMAC_CH_PRI_1) */
typedef struct{
__REG32 IDMAC_CH_PRI_0            : 1;
__REG32 IDMAC_CH_PRI_1            : 1;
__REG32 IDMAC_CH_PRI_2            : 1;
__REG32 IDMAC_CH_PRI_3            : 1;
__REG32 IDMAC_CH_PRI_4            : 1;
__REG32 IDMAC_CH_PRI_5            : 1;
__REG32 IDMAC_CH_PRI_6            : 1;
__REG32 IDMAC_CH_PRI_7            : 1;
__REG32 IDMAC_CH_PRI_8            : 1;
__REG32 IDMAC_CH_PRI_9            : 1;
__REG32 IDMAC_CH_PRI_10           : 1;
__REG32 IDMAC_CH_PRI_11           : 1;
__REG32 IDMAC_CH_PRI_12           : 1;
__REG32 IDMAC_CH_PRI_13           : 1;
__REG32 IDMAC_CH_PRI_14           : 1;
__REG32 IDMAC_CH_PRI_15           : 1;
__REG32 IDMAC_CH_PRI_16           : 1;
__REG32 IDMAC_CH_PRI_17           : 1;
__REG32 IDMAC_CH_PRI_18           : 1;
__REG32 IDMAC_CH_PRI_19           : 1;
__REG32 IDMAC_CH_PRI_20           : 1;
__REG32 IDMAC_CH_PRI_21           : 1;
__REG32 IDMAC_CH_PRI_22           : 1;
__REG32 IDMAC_CH_PRI_23           : 1;
__REG32 IDMAC_CH_PRI_24           : 1;
__REG32 IDMAC_CH_PRI_25           : 1;
__REG32 IDMAC_CH_PRI_26           : 1;
__REG32 IDMAC_CH_PRI_27           : 1;
__REG32 IDMAC_CH_PRI_28           : 1;
__REG32 IDMAC_CH_PRI_29           : 1;
__REG32 IDMAC_CH_PRI_30           : 1;
__REG32 IDMAC_CH_PRI_31           : 1;
} __ipu_idmac_ch_pri_1_bits;

/* IDMAC Channel Priority 2 Register (IPU_IDMAC_CH_PRI_2) */
typedef struct{
__REG32 IDMAC_CH_PRI_32           : 1;
__REG32 IDMAC_CH_PRI_33           : 1;
__REG32 IDMAC_CH_PRI_34           : 1;
__REG32 IDMAC_CH_PRI_35           : 1;
__REG32 IDMAC_CH_PRI_36           : 1;
__REG32 IDMAC_CH_PRI_37           : 1;
__REG32 IDMAC_CH_PRI_38           : 1;
__REG32 IDMAC_CH_PRI_39           : 1;
__REG32 IDMAC_CH_PRI_40           : 1;
__REG32 IDMAC_CH_PRI_41           : 1;
__REG32 IDMAC_CH_PRI_42           : 1;
__REG32 IDMAC_CH_PRI_43           : 1;
__REG32 IDMAC_CH_PRI_44           : 1;
__REG32 IDMAC_CH_PRI_45           : 1;
__REG32 IDMAC_CH_PRI_46           : 1;
__REG32 IDMAC_CH_PRI_47           : 1;
__REG32 IDMAC_CH_PRI_48           : 1;
__REG32 IDMAC_CH_PRI_49           : 1;
__REG32 IDMAC_CH_PRI_50           : 1;
__REG32 IDMAC_CH_PRI_51           : 1;
__REG32 IDMAC_CH_PRI_52           : 1;
__REG32 IDMAC_CH_PRI_53           : 1;
__REG32 IDMAC_CH_PRI_54           : 1;
__REG32 IDMAC_CH_PRI_55           : 1;
__REG32 IDMAC_CH_PRI_56           : 1;
__REG32 IDMAC_CH_PRI_57           : 1;
__REG32 IDMAC_CH_PRI_58           : 1;
__REG32 IDMAC_CH_PRI_59           : 1;
__REG32 IDMAC_CH_PRI_60           : 1;
__REG32 IDMAC_CH_PRI_61           : 1;
__REG32 IDMAC_CH_PRI_62           : 1;
__REG32 IDMAC_CH_PRI_63           : 1;
} __ipu_idmac_ch_pri_2_bits;

/* IDMAC Channel Watermark Enable 1 Register (IPU_IDMAC_WM_EN_1) */
typedef struct{
__REG32 IDMAC_WM_EN_0           : 1;
__REG32 IDMAC_WM_EN_1           : 1;
__REG32 IDMAC_WM_EN_2           : 1;
__REG32 IDMAC_WM_EN_3           : 1;
__REG32 IDMAC_WM_EN_4           : 1;
__REG32 IDMAC_WM_EN_5           : 1;
__REG32 IDMAC_WM_EN_6           : 1;
__REG32 IDMAC_WM_EN_7           : 1;
__REG32 IDMAC_WM_EN_8           : 1;
__REG32 IDMAC_WM_EN_9           : 1;
__REG32 IDMAC_WM_EN_10          : 1;
__REG32 IDMAC_WM_EN_11          : 1;
__REG32 IDMAC_WM_EN_12          : 1;
__REG32 IDMAC_WM_EN_13          : 1;
__REG32 IDMAC_WM_EN_14          : 1;
__REG32 IDMAC_WM_EN_15          : 1;
__REG32 IDMAC_WM_EN_16          : 1;
__REG32 IDMAC_WM_EN_17          : 1;
__REG32 IDMAC_WM_EN_18          : 1;
__REG32 IDMAC_WM_EN_19          : 1;
__REG32 IDMAC_WM_EN_20          : 1;
__REG32 IDMAC_WM_EN_21          : 1;
__REG32 IDMAC_WM_EN_22          : 1;
__REG32 IDMAC_WM_EN_23          : 1;
__REG32 IDMAC_WM_EN_24          : 1;
__REG32 IDMAC_WM_EN_25          : 1;
__REG32 IDMAC_WM_EN_26          : 1;
__REG32 IDMAC_WM_EN_27          : 1;
__REG32 IDMAC_WM_EN_28          : 1;
__REG32 IDMAC_WM_EN_29          : 1;
__REG32 IDMAC_WM_EN_30          : 1;
__REG32 IDMAC_WM_EN_31          : 1;
} __ipu_idmac_wm_en_1_bits;

/* IDMAC Channel Watermark Enable 2 Register (IPU_IDMAC_WM_EN_2) */
typedef struct{
__REG32 IDMAC_WM_EN_32          : 1;
__REG32 IDMAC_WM_EN_33          : 1;
__REG32 IDMAC_WM_EN_34          : 1;
__REG32 IDMAC_WM_EN_35          : 1;
__REG32 IDMAC_WM_EN_36          : 1;
__REG32 IDMAC_WM_EN_37          : 1;
__REG32 IDMAC_WM_EN_38          : 1;
__REG32 IDMAC_WM_EN_39          : 1;
__REG32 IDMAC_WM_EN_40          : 1;
__REG32 IDMAC_WM_EN_41          : 1;
__REG32 IDMAC_WM_EN_42          : 1;
__REG32 IDMAC_WM_EN_43          : 1;
__REG32 IDMAC_WM_EN_44          : 1;
__REG32 IDMAC_WM_EN_45          : 1;
__REG32 IDMAC_WM_EN_46          : 1;
__REG32 IDMAC_WM_EN_47          : 1;
__REG32 IDMAC_WM_EN_48          : 1;
__REG32 IDMAC_WM_EN_49          : 1;
__REG32 IDMAC_WM_EN_50          : 1;
__REG32 IDMAC_WM_EN_51          : 1;
__REG32 IDMAC_WM_EN_52          : 1;
__REG32 IDMAC_WM_EN_53          : 1;
__REG32 IDMAC_WM_EN_54          : 1;
__REG32 IDMAC_WM_EN_55          : 1;
__REG32 IDMAC_WM_EN_56          : 1;
__REG32 IDMAC_WM_EN_57          : 1;
__REG32 IDMAC_WM_EN_58          : 1;
__REG32 IDMAC_WM_EN_59          : 1;
__REG32 IDMAC_WM_EN_60          : 1;
__REG32 IDMAC_WM_EN_61          : 1;
__REG32 IDMAC_WM_EN_62          : 1;
__REG32 IDMAC_WM_EN_63          : 1;
} __ipu_idmac_wm_en_2_bits;

/* IDMAC Channel Lock Enable 1Register (IPU_IDMAC_LOCK_EN_1) */
typedef struct{
__REG32 IDMAC_LOCK_EN_0         : 2;
__REG32 IDMAC_LOCK_EN_1         : 2;
__REG32 IDMAC_LOCK_EN_2         : 2;
__REG32 IDMAC_LOCK_EN_3         : 2;
__REG32 IDMAC_LOCK_EN_4         : 2;
__REG32 IDMAC_LOCK_EN_5         : 2;
__REG32 IDMAC_LOCK_EN_6         : 2;
__REG32 IDMAC_LOCK_EN_7         : 2;
__REG32 IDMAC_LOCK_EN_8         : 2;
__REG32 IDMAC_LOCK_EN_9         : 2;
__REG32 IDMAC_LOCK_EN_10        : 2;
__REG32                         :10;
} __ipu_idmac_lock_en_1_bits;

/* IDMAC Channel Lock Enable 2 Register (IPU_IDMAC_LOCK_EN_2) */
typedef struct{
__REG32 IDMAC_LOCK_45           : 2;
__REG32 IDMAC_LOCK_46           : 2;
__REG32 IDMAC_LOCK_47           : 2;
__REG32 IDMAC_LOCK_48           : 2;
__REG32 IDMAC_LOCK_49           : 2;
__REG32 IDMAC_LOCK_50           : 2;
__REG32                         :20;
} __ipu_idmac_lock_en_2_bits;

/* IDMAC Channel Alternate Address 1 Register (IPU_IDMAC_SUB_ADDR_1) */
typedef struct{
__REG32 IDMAC_SUB_ADDR_23       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_24       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_29       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_33       : 7;
__REG32                         : 1;
} __ipu_idmac_sub_addr_1_bits;

/* IDMAC Channel Alternate Address 2 Register (IPU_IDMAC_SUB_ADDR_2) */
typedef struct{
__REG32 IDMAC_SUB_ADDR_41       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_51       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_52       : 7;
__REG32                         : 9;
} __ipu_idmac_sub_addr_2_bits;

/* IDMAC Channel Alternate Address 3 Register (IPU_IDMAC_SUB_ADDR_3) */
typedef struct{
__REG32 IDMAC_SUB_ADDR_9        : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_10       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_13       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_27       : 7;
__REG32                         : 1;
} __ipu_idmac_sub_addr_3_bits;

/* IDMAC Channel Alternate Address 4 Register (IPU_IDMAC_SUB_ADDR_4) */
typedef struct{
__REG32 IDMAC_SUB_ADDR_28       : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_8        : 7;
__REG32                         : 1;
__REG32 IDMAC_SUB_ADDR_21       : 7;
__REG32                         : 9;
} __ipu_idmac_sub_addr_4_bits;

/* IDMAC Band Mode Enable 1 Register (IPU_IDMAC_BNDM_EN_1) */
typedef struct{
__REG32 IDMAC_BNDM_EN_0         : 1;
__REG32 IDMAC_BNDM_EN_1         : 1;
__REG32 IDMAC_BNDM_EN_2         : 1;
__REG32 IDMAC_BNDM_EN_3         : 1;
__REG32                         : 1;
__REG32 IDMAC_BNDM_EN_5         : 1;
__REG32                         : 5;
__REG32 IDMAC_BNDM_EN_11        : 1;
__REG32 IDMAC_BNDM_EN_12        : 1;
__REG32                         : 7;
__REG32 IDMAC_BNDM_EN_20        : 1;
__REG32 IDMAC_BNDM_EN_21        : 1;
__REG32 IDMAC_BNDM_EN_22        : 1;
__REG32                         : 2;
__REG32 IDMAC_BNDM_EN_25        : 1;
__REG32 IDMAC_BNDM_EN_26        : 1;
__REG32                         : 5;
} __ipu_idmac_bndm_en_1_bits;

/* IDMAC Band Mode Enable 2 Register (IPU_IDMAC_BNDM_EN_2) */
typedef struct{
__REG32                         :13;
__REG32 IDMAC_BNDM_EN_45        : 1;
__REG32 IDMAC_BNDM_EN_46        : 1;
__REG32 IDMAC_BNDM_EN_47        : 1;
__REG32 IDMAC_BNDM_EN_48        : 1;
__REG32 IDMAC_BNDM_EN_49        : 1;
__REG32 IDMAC_BNDM_EN_50        : 1;
__REG32                         :13;
} __ipu_idmac_bndm_en_2_bits;

/* IDMAC Scroll Coordinations Register (IPU_IDMAC_SC_CORD) */
typedef struct{
__REG32 SY0                     :11;
__REG32                         : 5;
__REG32 SX0                     :12;
__REG32                         : 4;
} __ipu_idmac_sc_cord_bits;

/* IDMAC Scroll Coordinations Register 1 (IPU_IDMAC_SC_CORD_1) */
typedef struct{
__REG32 SY1                     :11;
__REG32                         : 5;
__REG32 SX1                     :12;
__REG32                         : 4;
} __ipu_idmac_sc_cord_1_bits;

/* IDMAC Channel Busy 1 Register (IPU_IDMAC_CH_BUSY_1) */
typedef struct{
__REG32 IDMAC_CH_BUSY_0         : 1;
__REG32 IDMAC_CH_BUSY_1         : 1;
__REG32 IDMAC_CH_BUSY_2         : 1;
__REG32 IDMAC_CH_BUSY_3         : 1;
__REG32                         : 1;
__REG32 IDMAC_CH_BUSY_5         : 1;
__REG32                         : 2;
__REG32 IDMAC_CH_BUSY_8         : 1;
__REG32 IDMAC_CH_BUSY_9         : 1;
__REG32 IDMAC_CH_BUSY_10        : 1;
__REG32 IDMAC_CH_BUSY_11        : 1;
__REG32 IDMAC_CH_BUSY_12        : 1;
__REG32 IDMAC_CH_BUSY_13        : 1;
__REG32 IDMAC_CH_BUSY_14        : 1;
__REG32 IDMAC_CH_BUSY_15        : 1;
__REG32                         : 1;
__REG32 IDMAC_CH_BUSY_17        : 1;
__REG32 IDMAC_CH_BUSY_18        : 1;
__REG32                         : 1;
__REG32 IDMAC_CH_BUSY_20        : 1;
__REG32 IDMAC_CH_BUSY_21        : 1;
__REG32 IDMAC_CH_BUSY_22        : 1;
__REG32 IDMAC_CH_BUSY_23        : 1;
__REG32 IDMAC_CH_BUSY_24        : 1;
__REG32 IDMAC_CH_BUSY_25        : 1;
__REG32 IDMAC_CH_BUSY_26        : 1;
__REG32 IDMAC_CH_BUSY_27        : 1;
__REG32 IDMAC_CH_BUSY_28        : 1;
__REG32 IDMAC_CH_BUSY_29        : 1;
__REG32                         : 1;
__REG32 IDMAC_CH_BUSY_31        : 1;
} __ipu_idmac_ch_busy_1_bits;

/* IDMAC Channel Busy 2 Register (IPU_IDMAC_CH_BUSY_2) */
typedef struct{
__REG32                         : 1;
__REG32 IDMAC_CH_BUSY_33        : 1;
__REG32                         : 6;
__REG32 IDMAC_CH_BUSY_40        : 1;
__REG32 IDMAC_CH_BUSY_41        : 1;
__REG32 IDMAC_CH_BUSY_42        : 1;
__REG32 IDMAC_CH_BUSY_43        : 1;
__REG32 IDMAC_CH_BUSY_44        : 1;
__REG32 IDMAC_CH_BUSY_45        : 1;
__REG32 IDMAC_CH_BUSY_46        : 1;
__REG32 IDMAC_CH_BUSY_47        : 1;
__REG32 IDMAC_CH_BUSY_48        : 1;
__REG32 IDMAC_CH_BUSY_49        : 1;
__REG32 IDMAC_CH_BUSY_50        : 1;
__REG32 IDMAC_CH_BUSY_51        : 1;
__REG32 IDMAC_CH_BUSY_52        : 1;
__REG32                         :11;
} __ipu_idmac_ch_busy_2_bits;

/* DP Debug Control Register (IPU_DP_DEBUG_CNT) */
typedef struct{
__REG32 BRAKE_STATUS_EN_0       : 1;
__REG32 BRAKE_CNT_0             : 3;
__REG32 BRAKE_STATUS_EN_1       : 1;
__REG32 BRAKE_CNT_1             : 3;
__REG32                         :24;
} __ipu_dp_debug_cnt_bits;

/* DP Debug Status Register (IPU_DP_DEBUG_STAT) */
typedef struct{
__REG32 V_CNT_OLD_0             :11;
__REG32 FG_ACTIVE_0             : 1;
__REG32 COMBYP_EN_OLD_0         : 1;
__REG32 CYP_EN_OLD_0            : 1;
__REG32                         : 2;
__REG32 V_CNT_OLD_1             :11;
__REG32 FG_ACTIVE_1             : 1;
__REG32 COMBYP_EN_OLD_1         : 1;
__REG32 CYP_EN_OLD_1            : 1;
__REG32                         : 2;
} __ipu_dp_debug_stat_bits;

/* IC Configuration Register (IPU_IC_CONF) */
typedef struct{
__REG32 PRPENC_EN               : 1;
__REG32 PRPENC_CSC1             : 1;
__REG32 PRPENC_ROT_EN           : 1;
__REG32                         : 5;
__REG32 PRPVF_EN                : 1;
__REG32 PRPVF_CSC1              : 1;
__REG32 PRPVF_CSC2              : 1;
__REG32 PRPVF_CMB               : 1;
__REG32 PRPVF_ROT_EN            : 1;
__REG32                         : 3;
__REG32 PP_EN                   : 1;
__REG32 PP_CSC1                 : 1;
__REG32 PP_CSC2                 : 1;
__REG32 PP_CMB                  : 1;
__REG32 PP_ROT_EN               : 1;
__REG32                         : 7;
__REG32 IC_GLB_LOC_A            : 1;
__REG32 IC_KEY_COLOR_EN         : 1;
__REG32 RWS_EN                  : 1;
__REG32 CSI_MEM_WR_EN           : 1;
} __ipu_ic_conf_bits;

/* IC Preprocessing Encoder Resizing Coefficients Register (IPU_IC_PRP_ENC_RSC) */
typedef struct{
__REG32 PRPENC_RS_R_H           :14;
__REG32 PRPENC_DS_R_H           : 2;
__REG32 PRPENC_RS_R_V           :14;
__REG32 PRPENC_DS_R_V           : 2;
} __ipu_ic_prp_enc_rsc_bits;

/* IC Preprocessing View-Finder Resizing Coefficients Register (IPU_IC_PRP_VF_RSC) */
typedef struct{
__REG32 PRPVF_RS_R_H            :14;
__REG32 PRPVF_DS_R_H            : 2;
__REG32 PRPVF_RS_R_V            :14;
__REG32 PRPVF_DS_R_V            : 2;
} __ipu_ic_prp_vf_rsc_bits;

/* IC Postprocessing Encoder Resizing Coefficients Register (IPU_IC_PP_RSC) */
typedef struct{
__REG32 PP_RS_R_H               :14;
__REG32 PP_DS_R_H               : 2;
__REG32 PP_RS_R_V               :14;
__REG32 PP_DS_R_V               : 2;
} __ipu_ic_pp_rsc_bits;

/* IC Combining Parameters Register 1 (IPU_IC_CMBP_1) */
typedef struct{
__REG32 IC_PRPVF_ALPHA_V        : 8;
__REG32 IC_PP_ALPHA_V           : 8;
__REG32                         :16;
} __ipu_ic_cmbp_1_bits;

/* IC Combining Parameters Register 2 (IPU_IC_CMBP_2) */
typedef struct{
__REG32 IC_KEY_COLOR_B          : 8;
__REG32 IC_KEY_COLOR_G          : 8;
__REG32 IC_KEY_COLOR_R          : 8;
__REG32                         : 8;
} __ipu_ic_cmbp_2_bits;

/* IC IDMAC Parameters 1 Register (IPU_IC_IDMAC_1) */
typedef struct{
__REG32 CB0_BURST_16            : 1;
__REG32 CB1_BURST_16            : 1;
__REG32 CB2_BURST_16            : 1;
__REG32 CB3_BURST_16            : 1;
__REG32 CB4_BURST_16            : 1;
__REG32 CB5_BURST_16            : 1;
__REG32 CB6_BURST_16            : 1;
__REG32 CB7_BURST_16            : 1;
__REG32                         : 3;
__REG32 T1_ROT                  : 1;
__REG32 T1_FLIP_LR              : 1;
__REG32 T1_FLIP_UD              : 1;
__REG32 T2_ROT                  : 1;
__REG32 T2_FLIP_LR              : 1;
__REG32 T2_FLIP_UD              : 1;
__REG32 T3_ROT                  : 1;
__REG32 T3_FLIP_LR              : 1;
__REG32 T3_FLIP_UD              : 1;
__REG32 T1_FLIP_RS              : 1;
__REG32 T2_FLIP_RS              : 1;
__REG32 T3_FLIP_RS              : 1;
__REG32                         : 1;
__REG32 ALT_CB6_BURST_16        : 1;
__REG32 ALT_CB7_BURST_16        : 1;
__REG32                         : 6;
} __ipu_ic_idmac_1_bits;

/* IC IDMAC Parameters 2 Register (IPU_IC_IDMAC_2) */
typedef struct{
__REG32 T1_FR_HEIGHT            :10;
__REG32 T2_FR_HEIGHT            :10;
__REG32 T3_FR_HEIGHT            :10;
__REG32                         : 2;
} __ipu_ic_idmac_2_bits;

/* IC IDMAC Parameters 3 Register (IPU_IC_IDMAC_3) */
typedef struct{
__REG32 T1_FR_WIDTH             :10;
__REG32 T2_FR_WIDTH             :10;
__REG32 T3_FR_WIDTH             :10;
__REG32                         : 2;
} __ipu_ic_idmac_3_bits;

/* IC IDMAC Parameters 4 Register (IPU_IC_IDMAC_4) */
typedef struct{
__REG32 mpm_rw_brdg_max_rq      : 4;
__REG32 mpm_dmfc_brdg_max_rq    : 4;
__REG32 ibm_brdg_max_rq         : 4;
__REG32 rm_brdg_max_rq          : 4;
__REG32                         :16;
} __ipu_ic_idmac_4_bits;

/* CSI0 Sensor Configuration Register (IPU_CSI0_SENS_CONF) */
typedef struct{
__REG32 CSI0_VSYNC_POL          : 1;
__REG32 CSI0_HSYNC_POL          : 1;
__REG32 CSI0_DATA_POL           : 1;
__REG32 CSI0_SENS_PIX_CLK_POL   : 1;
__REG32 CSI0_SENS_PRTCL         : 3;
__REG32 CSI0_PACK_TIGHT         : 1;
__REG32 CSI0_SENS_DATA_FORMAT   : 3;
__REG32 CSI0_DATA_WIDTH         : 4;
__REG32 CSI0_EXT_VSYNC          : 1;
__REG32 CSI0_DIV_RATIO          : 8;
__REG32 CSI0_DATA_DEST          : 3;
__REG32 CSI0_JPEG8_EN           : 1;
__REG32 CSI0_JPEG_MODE          : 1;
__REG32 CSI0_FORCE_EOF          : 1;
__REG32                         : 1;
__REG32 CSI0_DATA_EN_POL        : 1;
} __ipu_csi0_sens_conf_bits;

/* CSI0 Sense Frame Size Register (IPU_CSI0_SENS_FRM_SIZE) */
typedef struct{
__REG32 CSI0_SENS_FRM_WIDTH     :13;
__REG32                         : 3;
__REG32 CSI0_SENS_FRM_HEIGHT    :12;
__REG32                         : 4;
} __ipu_csi0_sens_frm_size_bits;

/* CSI0 Actual Frame Size Register (IPU_CSI0_ACT_FRM_SIZE) */
typedef struct{
__REG32 CSI0_ACT_FRM_WIDTH      :13;
__REG32                         : 3;
__REG32 CSI0_ACT_FRM_HEIGHT     :12;
__REG32                         : 4;
} __ipu_csi0_act_frm_size_bits;

/* CSI0 Output Control Register (IPU_CSI0_OUT_FRM_CTRL) */
typedef struct{
__REG32 CSI0_VSC                :12;
__REG32                         : 4;
__REG32 CSI0_HSC                :13;
__REG32                         : 1;
__REG32 CSI0_VERT_DWNS          : 1;
__REG32 CSI0_HORZ_DWNS          : 1;
} __ipu_csi0_out_frm_ctrl_bits;

/* CSIO Test Control Register (IPU_CSI0_TST_CTRL) */
typedef struct{
__REG32 PG_R_VALUE              : 8;
__REG32 PG_G_VALUE              : 8;
__REG32 PG_B_VALUE              : 8;
__REG32 TEST_GEN_MODE           : 1;
__REG32                         : 7;
} __ipu_csi0_tst_ctrl_bits;

/* CSIO CCIR Code Register 1 (IPU_CSI0_CCIR_CODE_1) */
typedef struct{
__REG32 CSI0_END_FLD0_BLNK_1ST  : 3;
__REG32 CSI0_STRT_FLD0_BLNK_1ST : 3;
__REG32 CSI0_END_FLD0_BLNK_2ND  : 3;
__REG32 CSI0_STRT_FLD0_BLNK_2ND : 3;
__REG32                         : 4;
__REG32 CSI0_END_FLD0_ACTV      : 3;
__REG32 CSI0_STRT_FLD0_ACTV     : 3;
__REG32                         : 2;
__REG32 CSI0_CCIR_ERR_DET_EN    : 1;
__REG32                         : 7;
} __ipu_csi0_ccir_code_1_bits;

/* CSIO CCIR Code Register 2 (IPU_CSI0_CCIR_CODE_2) */
typedef struct{
__REG32 CSI0_END_FLD1_BLNK_1ST  : 3;
__REG32 CSI0_STRT_FLD1_BLNK_1ST : 3;
__REG32 CSI0_END_FLD1_BLNK_2ND  : 3;
__REG32 CSI0_STRT_FLD1_BLNK_2ND : 3;
__REG32                         : 4;
__REG32 CSI0_END_FLD1_ACTV      : 3;
__REG32 CSI0_STRT_FLD1_ACTV     : 3;
__REG32                         :10;
} __ipu_csi0_ccir_code_2_bits;

/* CSIO CCIR Code Register 3 (IPU_CSI0_CCIR_CODE_3) */
typedef struct{
__REG32 CSI0_CCIR_PRECOM        :30;
__REG32                         : 2;
} __ipu_csi0_ccir_code_3_bits;

/* CSI0 Data Identifier Register (IPU_CSI0_DI) */
typedef struct{
__REG32 CSI0_MIPI_DI0           : 8;
__REG32 CSI0_MIPI_DI1           : 8;
__REG32 CSI0_MIPI_DI2           : 8;
__REG32 CSI0_MIPI_DI3           : 8;
} __ipu_csi0_di_bits;

/* CSI0 SKIP Register (IPU_CSI0_SKIP) */
typedef struct{
__REG32 CSI0_MAX_RATIO_SKIP_SMFC  : 3;
__REG32 CSI0_SKIP_SMFC            : 5;
__REG32 CSI0_ID_2_SKIP            : 2;
__REG32                           : 6;
__REG32 CSI0_MAX_RATIO_SKIP_ISP   : 3;
__REG32 CSI0_SKIP_ISP             : 5;
__REG32                           : 8;
} __ipu_csi0_skip_bits;

/* CSI0 Compander Control Register (IPU_CSI0_CPD_CTRL) */
typedef struct{
__REG32 CSI0_GREEN_P_BEGIN        : 1;
__REG32 CSI0_RED_ROW_BEGIN        : 1;
__REG32 CSI0_CPD                  : 3;
__REG32                           :27;
} __ipu_csi0_cpd_ctrl_bits;

/* CSI0 Red Component Compander Constants Register 0 (IPU_CSI0_CPD_RC_0) */
typedef struct{
__REG32 CSI0_CPD_RC_0             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_1             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_0_bits;

/* CSI0 Red Component Compander Constants Register 1 (IPU_CSI0_CPD_RC_1) */
typedef struct{
__REG32 CSI0_CPD_RC_2             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_3             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_1_bits;

/* CSI0 Red Component Compander Constants Register 2 (IPU_CSI0_CPD_RC_2) */
typedef struct{
__REG32 CSI0_CPD_RC_4             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_5             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_2_bits;

/* CSI0 Red Component Compander Constants Register 3 (IPU_CSI0_CPD_RC_3) */
typedef struct{
__REG32 CSI0_CPD_RC_6             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_7             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_3_bits;

/* CSI0 Red Component Compander Constants Register 4 (IPU_CSI0_CPD_RC_4) */
typedef struct{
__REG32 CSI0_CPD_RC_8             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_9             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_4_bits;

/* CSI0 Red Component Compander Constants Register 5 (IPU_CSI0_CPD_RC_5) */
typedef struct{
__REG32 CSI0_CPD_RC_10            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_11            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_5_bits;

/* CSI0 Red Component Compander Constants Register 6 (IPU_CSI0_CPD_RC_6) */
typedef struct{
__REG32 CSI0_CPD_RC_12            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_13            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_6_bits;

/* CSI0 Red Component Compander Constants Register 7 (IPU_CSI0_CPD_RC_7) */
typedef struct{
__REG32 CSI0_CPD_RC_14            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_RC_15            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_rc_7_bits;

/* CSI0 Red Component Compander SLOPE Register 0 (IPU_CSI0_CPD_RS_0) */
typedef struct{
__REG32 CSI0_CPD_RS_0             : 8;
__REG32 CSI0_CPD_RS_1             : 8;
__REG32 CSI0_CPD_RS_2             : 8;
__REG32 CSI0_CPD_RS_3             : 8;
} __ipu_csi0_cpd_rs_0_bits;

/* CSI0 Red Component Compander SLOPE Register 1 (IPU_CSI0_CPD_RS_1) */
typedef struct{
__REG32 CSI0_CPD_RS_4             : 8;
__REG32 CSI0_CPD_RS_5             : 8;
__REG32 CSI0_CPD_RS_6             : 8;
__REG32 CSI0_CPD_RS_7             : 8;
} __ipu_csi0_cpd_rs_1_bits;

/* CSI0 Red Component Compander SLOPE Register 2 (IPU_CSI0_CPD_RS_2) */
typedef struct{
__REG32 CSI0_CPD_RS_8             : 8;
__REG32 CSI0_CPD_RS_9             : 8;
__REG32 CSI0_CPD_RS_10            : 8;
__REG32 CSI0_CPD_RS_11            : 8;
} __ipu_csi0_cpd_rs_2_bits;

/* CSI0 Red Component Compander SLOPE Register 3 (IPU_CSI0_CPD_RS_3) */
typedef struct{
__REG32 CSI0_CPD_RS_12            : 8;
__REG32 CSI0_CPD_RS_13            : 8;
__REG32 CSI0_CPD_RS_14            : 8;
__REG32 CSI0_CPD_RS_15            : 8;
} __ipu_csi0_cpd_rs_3_bits;

/* CSI0 GR Component Compander Constants Register 0 (IPU_CSI0_CPD_GRC_0) */
typedef struct{
__REG32 CSI0_CPD_GRC_0            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_1            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_0_bits;

/* CSI0 GR Component Compander Constants Register 1 (IPU_CSI0_CPD_GRC_1) */
typedef struct{
__REG32 CSI0_CPD_GRC_2            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_3            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_1_bits;

/* CSI0 GR Component Compander Constants Register 2 (IPU_CSI0_CPD_GRC_2) */
typedef struct{
__REG32 CSI0_CPD_GRC_4            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_5            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_2_bits;

/* CSI0 GR Component Compander Constants Register 3 (IPU_CSI0_CPD_GRC_3) */
typedef struct{
__REG32 CSI0_CPD_GRC_6            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_7            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_3_bits;

/* CSI0 GR Component Compander Constants Register 4 (IPU_CSI0_CPD_GRC_4) */
typedef struct{
__REG32 CSI0_CPD_GRC_8            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_9            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_4_bits;

/* CSI0 GR Component Compander Constants Register 5 (IPU_CSI0_CPD_GRC_5) */
typedef struct{
__REG32 CSI0_CPD_GRC_10           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_11           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_5_bits;

/* CSI0 GR Component Compander Constants Register 6 (IPU_CSI0_CPD_GRC_6) */
typedef struct{
__REG32 CSI0_CPD_GRC_12           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_13           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_6_bits;

/* CSI0 GR Component Compander Constants Register 7 (IPU_CSI0_CPD_GRC_7) */
typedef struct{
__REG32 CSI0_CPD_GRC_14           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GRC_15           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_grc_7_bits;

/* CSI0 GR Component Compander SLOPE Register 0 (IPU_CSI0_CPD_GRS_0) */
typedef struct{
__REG32 CSI0_CPD_GRS_0            : 8;
__REG32 CSI0_CPD_GRS_1            : 8;
__REG32 CSI0_CPD_GRS_2            : 8;
__REG32 CSI0_CPD_GRS_3            : 8;
} __ipu_csi0_cpd_grs_0_bits;

/* CSI0 GR Component Compander SLOPE Register 1 (IPU_CSI0_CPD_GRS_1) */
typedef struct{
__REG32 CSI0_CPD_GRS_4            : 8;
__REG32 CSI0_CPD_GRS_5            : 8;
__REG32 CSI0_CPD_GRS_6            : 8;
__REG32 CSI0_CPD_GRS_7            : 8;
} __ipu_csi0_cpd_grs_1_bits;

/* CSI0 GR Component Compander SLOPE Register 2 (IPU_CSI0_CPD_GRS_2) */
typedef struct{
__REG32 CSI0_CPD_GRS_8            : 8;
__REG32 CSI0_CPD_GRS_9            : 8;
__REG32 CSI0_CPD_GRS_10           : 8;
__REG32 CSI0_CPD_GRS_11           : 8;
} __ipu_csi0_cpd_grs_2_bits;

/* CSI0 GR Component Compander SLOPE Register 3 (IPU_CSI0_CPD_GRS_3) */
typedef struct{
__REG32 CSI0_CPD_GRS_12           : 8;
__REG32 CSI0_CPD_GRS_13           : 8;
__REG32 CSI0_CPD_GRS_14           : 8;
__REG32 CSI0_CPD_GRS_15           : 8;
} __ipu_csi0_cpd_grs_3_bits;

/* CSI0 GB Component Compander Constants Register 0 (IPU_CSI0_CPD_GBC_0) */
typedef struct{
__REG32 CSI0_CPD_GBC_0            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_1            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_0_bits;

/* CSI0 GB Component Compander Constants Register 1 (IPU_CSI0_CPD_GBC_1) */
typedef struct{
__REG32 CSI0_CPD_GBC_2            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_3            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_1_bits;

/* CSI0 GB Component Compander Constants Register 2 (IPU_CSI0_CPD_GBC_2) */
typedef struct{
__REG32 CSI0_CPD_GBC_4            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_5            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_2_bits;

/* CSI0 GB Component Compander Constants Register 3 (IPU_CSI0_CPD_GBC_3) */
typedef struct{
__REG32 CSI0_CPD_GBC_6            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_7            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_3_bits;

/* CSI0 GB Component Compander Constants Register 4 (IPU_CSI0_CPD_GBC_4) */
typedef struct{
__REG32 CSI0_CPD_GBC_8            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_9            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_4_bits;

/* CSI0 GB Component Compander Constants Register 5 (IPU_CSI0_CPD_GBC_5) */
typedef struct{
__REG32 CSI0_CPD_GBC_10           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_11           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_5_bits;

/* CSI0 GB Component Compander Constants Register 6 (IPU_CSI0_CPD_GBC_6) */
typedef struct{
__REG32 CSI0_CPD_GBC_12           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_13           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_6_bits;

/* CSI0 GB Component Compander Constants Register 7 (IPU_CSI0_CPD_GBC_7) */
typedef struct{
__REG32 CSI0_CPD_GBC_14           : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_GBC_15           : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_gbc_7_bits;

/* CSI0 GB Component Compander SLOPE Register 0 (IPU_CSI0_CPD_GBS_0) */
typedef struct{
__REG32 CSI0_CPD_GBS_0            : 8;
__REG32 CSI0_CPD_GBS_1            : 8;
__REG32 CSI0_CPD_GBS_2            : 8;
__REG32 CSI0_CPD_GBS_3            : 8;
} __ipu_csi0_cpd_gbs_0_bits;

/* CSI0 GB Component Compander SLOPE Register 1 (IPU_CSI0_CPD_GBS_1) */
typedef struct{
__REG32 CSI0_CPD_GBS_4            : 8;
__REG32 CSI0_CPD_GBS_5            : 8;
__REG32 CSI0_CPD_GBS_6            : 8;
__REG32 CSI0_CPD_GBS_7            : 8;
} __ipu_csi0_cpd_gbs_1_bits;

/* CSI0 GB Component Compander SLOPE Register 2 (IPU_CSI0_CPD_GBS_2) */
typedef struct{
__REG32 CSI0_CPD_GBS_8            : 8;
__REG32 CSI0_CPD_GBS_9            : 8;
__REG32 CSI0_CPD_GBS_10           : 8;
__REG32 CSI0_CPD_GBS_11           : 8;
} __ipu_csi0_cpd_gbs_2_bits;

/* CSI0 GB Component Compander SLOPE Register 3 (IPU_CSI0_CPD_GBS_3) */
typedef struct{
__REG32 CSI0_CPD_GBS_12           : 8;
__REG32 CSI0_CPD_GBS_13           : 8;
__REG32 CSI0_CPD_GBS_14           : 8;
__REG32 CSI0_CPD_GBS_15           : 8;
} __ipu_csi0_cpd_gbs_3_bits;

/* CSI0 Blue Component Compander Constants Register 0 (IPU_CSI0_CPD_BC_0) */
typedef struct{
__REG32 CSI0_CPD_BC_0             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_1             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_0_bits;

/* CSI0 Blue Component Compander Constants Register 1 (IPU_CSI0_CPD_BC_1) */
typedef struct{
__REG32 CSI0_CPD_BC_2             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_3             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_1_bits;

/* CSI0 Blue Component Compander Constants Register 2 (IPU_CSI0_CPD_BC_2) */
typedef struct{
__REG32 CSI0_CPD_BC_4             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_5             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_2_bits;

/* CSI0 Blue Component Compander Constants Register 3 (IPU_CSI0_CPD_BC_3) */
typedef struct{
__REG32 CSI0_CPD_BC_6             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_7             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_3_bits;

/* CSI0 Blue Component Compander Constants Register 4 (IPU_CSI0_CPD_BC_4) */
typedef struct{
__REG32 CSI0_CPD_BC_8             : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_9             : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_4_bits;

/* CSI0 Blue Component Compander Constants Register 5 (IPU_CSI0_CPD_BC_5) */
typedef struct{
__REG32 CSI0_CPD_BC_10            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_11            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_5_bits;

/* CSI0 Blue Component Compander Constants Register 6 (IPU_CSI0_CPD_BC_6) */
typedef struct{
__REG32 CSI0_CPD_BC_12            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_13            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_6_bits;

/* CSI0 Blue Component Compander Constants Register 7 (IPU_CSI0_CPD_BC_7) */
typedef struct{
__REG32 CSI0_CPD_BC_14            : 9;
__REG32                           : 7;
__REG32 CSI0_CPD_BC_15            : 9;
__REG32                           : 7;
} __ipu_csi0_cpd_bc_7_bits;

/* CSI0 Blue Component Compander SLOPE Register 0 (IPU_CSI0_CPD_BS_0) */
typedef struct{
__REG32 CSI0_CPD_BS_0             : 8;
__REG32 CSI0_CPD_BS_1             : 8;
__REG32 CSI0_CPD_BS_2             : 8;
__REG32 CSI0_CPD_BS_3             : 8;
} __ipu_csi0_cpd_bs_0_bits;

/* CSI0 Blue Component Compander SLOPE Register 1 (IPU_CSI0_CPD_BS_1) */
typedef struct{
__REG32 CSI0_CPD_BS_4             : 8;
__REG32 CSI0_CPD_BS_5             : 8;
__REG32 CSI0_CPD_BS_6             : 8;
__REG32 CSI0_CPD_BS_7             : 8;
} __ipu_csi0_cpd_bs_1_bits;

/* CSI0 Blue Component Compander SLOPE Register 2 (IPU_CSI0_CPD_BS_2) */
typedef struct{
__REG32 CSI0_CPD_BS_8             : 8;
__REG32 CSI0_CPD_BS_9             : 8;
__REG32 CSI0_CPD_BS_10            : 8;
__REG32 CSI0_CPD_BS_11            : 8;
} __ipu_csi0_cpd_bs_2_bits;

/* CSI0 Blue Component Compander SLOPE Register 3 (IPU_CSI0_CPD_BS_3) */
typedef struct{
__REG32 CSI0_CPD_BS_12            : 8;
__REG32 CSI0_CPD_BS_13            : 8;
__REG32 CSI0_CPD_BS_14            : 8;
__REG32 CSI0_CPD_BS_15            : 8;
} __ipu_csi0_cpd_bs_3_bits;

/* CSI0 Compander Offset Register 1 (IPU_CSI0_CPD_OFFSET1) */
typedef struct{
__REG32 CSI0_GR_OFFSET            :10;
__REG32 CSI0_GB_OFFSET            :10;
__REG32 CSI0_CPD_B_OFFSET         :10;
__REG32                           : 2;
} __ipu_csi0_cpd_offset1_bits;

/* CSI0 Compander Offset Register 2 (IPU_CSI0_CPD_OFFSET2) */
typedef struct{
__REG32 CSI0_CPD_R_OFFSET         :10;
__REG32                           :22;
} __ipu_csi0_cpd_offset2_bits;

/* CSI1 Sensor Configuration Register (IPU_CSI1_SENS_CONF) */
typedef struct{
__REG32 CSI1_VSYNC_POL            : 1;
__REG32 CSI1_HSYNC_POL            : 1;
__REG32 CSI1_DATA_POL             : 1;
__REG32 CSI1_SENS_PIX_CLK_POL     : 1;
__REG32 CSI1_SENS_PRTCL           : 3;
__REG32 CSI1_PACK_TIGHT           : 1;
__REG32 CSI1_SENS_DATA_FORMAT     : 3;
__REG32 CSI1_DATA_WIDTH           : 4;
__REG32 CSI1_EXT_VSYNC            : 1;
__REG32 CSI1_DIV_RATIO            : 8;
__REG32 CSI1_DATA_DEST            : 3;
__REG32 CSI1_JPEG8_EN             : 1;
__REG32 CSI1_JPEG_MODE            : 1;
__REG32 CSI1_FORCE_EOF            : 1;
__REG32                           : 1;
__REG32 CSI0_DATA_EN_POL          : 1;
} __ipu_csi1_sens_conf_bits;

/* CSI1 Sense Frame Size Register (IPU_CSI1_SENS_FRM_SIZE) */
typedef struct{
__REG32 CSI1_SENS_FRM_WIDTH       :13;
__REG32                           : 3;
__REG32 CSI1_SENS_FRM_HEIGHT      :12;
__REG32                           : 4;
} __ipu_csi1_sens_frm_size_bits;

/* CSI1 Actual Frame Size Register (IPU_CSI1_ACT_FRM_SIZE) */
typedef struct{
__REG32 CSI1_ACT_FRM_WIDTH        :13;
__REG32                           : 3;
__REG32 CSI1_ACT_FRM_HEIGHT       :12;
__REG32                           : 4;
} __ipu_csi1_act_frm_size_bits;

/* CSI1 Output Control Register (IPU_CSI1_OUT_FRM_CTRL) */
typedef struct{
__REG32 CSI1_VSC                  :12;
__REG32                           : 4;
__REG32 CSI1_HSC                  :13;
__REG32                           : 1;
__REG32 CSI1_VERT_DWNS            : 1;
__REG32 CSI1_HORZ_DWNS            : 1;
} __ipu_csi1_out_frm_ctrl_bits;

/* CSI1 Test Control Register (IPU_CSI1_TST_CTRL) */
typedef struct{
__REG32 PG_R_VALUE                : 8;
__REG32 PG_G_VALUE                : 8;
__REG32 PG_B_VALUE                : 8;
__REG32 TEST_GEN_MODE             : 1;
__REG32                           : 7;
} __ipu_csi1_tst_ctrl_bits;

/* CSI1 CCIR Code Register 1 (IPU_CSI1_CCIR_CODE_1) */
typedef struct{
__REG32 CSI1_END_FLD0_BLNK_1ST    : 3;
__REG32 CSI1_STRT_FLD0_BLNK_1ST   : 3;
__REG32 CSI1_END_FLD0_BLNK_2ND    : 3;
__REG32 CSI1_STRT_FLD0_BLNK_2ND   : 3;
__REG32                           : 4;
__REG32 CSI1_END_FLD0_ACTV        : 3;
__REG32 CSI1_STRT_FLD0_ACTV       : 3;
__REG32                           : 2;
__REG32 CSI1_CCIR_ERR_DET_EN      : 1;
__REG32                           : 7;
} __ipu_csi1_ccir_code_1_bits;

/* CSI1 CCIR Code Register 2 (IPU_CSI1_CCIR_CODE_2) */
typedef struct{
__REG32 CSI1_END_FLD1_BLNK_1ST    : 3;
__REG32 CSI1_STRT_FLD1_BLNK_1ST   : 3;
__REG32 CSI1_END_FLD1_BLNK_2ND    : 3;
__REG32 CSI1_STRT_FLD1_BLNK_2ND   : 3;
__REG32                           : 4;
__REG32 CSI1_END_FLD1_ACTV        : 3;
__REG32 CSI1_STRT_FLD1_ACTV       : 3;
__REG32                           :10;
} __ipu_csi1_ccir_code_2_bits;

/* CSI1 CCIR Code Register 3 (IPU_CSI1_CCIR_CODE_3) */
typedef struct{
__REG32 CSI1_CCIR_PRECOM          :30;
__REG32                           : 2;
} __ipu_csi1_ccir_code_3_bits;

/* CSI1 Data Identifier Register (IPU_CSI1_DI) */
typedef struct{
__REG32 CSI1_MIPI_DI0             : 8;
__REG32 CSI1_MIPI_DI1             : 8;
__REG32 CSI1_MIPI_DI2             : 8;
__REG32 CSI1_MIPI_DI3             : 8;
} __ipu_csi1_di_bits;

/* CSI1 SKIP Register (IPU_CSI1_SKIP) */
typedef struct{
__REG32 CSI1_MAX_RATIO_SKIP_SMFC  : 3;
__REG32 CSI1_SKIP_SMFC            : 5;
__REG32 CSI1_ID_2_SKIP            : 2;
__REG32                           : 6;
__REG32 CSI1_MAX_RATIO_SKIP_ISP   : 3;
__REG32 CSI1_SKIP_ISP             : 5;
__REG32                           : 8;
} __ipu_csi1_skip_bits;

/* CSI1 Red Component Compander Constants Register 0 (IPU_CSIO_CPD_RC_0) */
typedef struct{
__REG32 CSI1_CPD_RC_0             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_1             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_0_bits;

/* CSI1 Red Component Compander Constants Register 1 (IPU_CSI1_CPD_RC_1) */
typedef struct{
__REG32 CSI1_CPD_RC_2             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_3             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_1_bits;

/* CSI1 Red Component Compander Constants Register 2 (IPU_CSI1_CPD_RC_2) */
typedef struct{
__REG32 CSI1_CPD_RC_4             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_5             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_2_bits;

/* CSI1 Red Component Compander Constants Register 3 (IPU_CSI1_CPD_RC_3) */
typedef struct{
__REG32 CSI1_CPD_RC_6             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_7             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_3_bits;

/* CSI1 Red Component Compander Constants Register 4 (IPU_CSI1_CPD_RC_4) */
typedef struct{
__REG32 CSI1_CPD_RC_8             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_9             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_4_bits;

/* CSI1 Red Component Compander Constants Register 5 (IPU_CSI1_CPD_RC_5) */
typedef struct{
__REG32 CSI1_CPD_RC_10            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_11            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_5_bits;

/* CSI1 Red Component Compander Constants Register 6 (IPU_CSI1_CPD_RC_6) */
typedef struct{
__REG32 CSI1_CPD_RC_12            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_13            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_6_bits;

/* CSI1 Red Component Compander Constants Register 7 (IPU_CSI1_CPD_RC_7) */
typedef struct{
__REG32 CSI1_CPD_RC_14            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_RC_15            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_rc_7_bits;

/* CSI1 Red Component Compander SLOPE Register 0 (IPU_CSI1_CPD_RS_0) */
typedef struct{
__REG32 CSI1_CPD_RS_0             : 8;
__REG32 CSI1_CPD_RS_1             : 8;
__REG32 CSI1_CPD_RS_2             : 8;
__REG32 CSI1_CPD_RS_3             : 8;
} __ipu_csi1_cpd_rs_0_bits;

/* CSI1 Red Component Compander SLOPE Register 1 (IPU_CSI1_CPD_RS_1) */
typedef struct{
__REG32 CSI1_CPD_RS_4             : 8;
__REG32 CSI1_CPD_RS_5             : 8;
__REG32 CSI1_CPD_RS_6             : 8;
__REG32 CSI1_CPD_RS_7             : 8;
} __ipu_csi1_cpd_rs_1_bits;

/* CSI1 Red Component Compander SLOPE Register 2 (IPU_CSI1_CPD_RS_2) */
typedef struct{
__REG32 CSI1_CPD_RS_8             : 8;
__REG32 CSI1_CPD_RS_9             : 8;
__REG32 CSI1_CPD_RS_10            : 8;
__REG32 CSI1_CPD_RS_11            : 8;
} __ipu_csi1_cpd_rs_2_bits;

/* CSI1 Red Component Compander SLOPE Register 3 (IPU_CSI1_CPD_RS_3) */
typedef struct{
__REG32 CSI1_CPD_RS_12            : 8;
__REG32 CSI1_CPD_RS_13            : 8;
__REG32 CSI1_CPD_RS_14            : 8;
__REG32 CSI1_CPD_RS_15            : 8;
} __ipu_csi1_cpd_rs_3_bits;

/* CSI1 GR Component Compander Constants Register 0 (IPU_CSI1_CPD_GRC_0) */
typedef struct{
__REG32 CSI1_CPD_GRC_0            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_1            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_0_bits;

/* CSI1 GR Component Compander Constants Register 1 (IPU_CSI1_CPD_GRC_1) */
typedef struct{
__REG32 CSI1_CPD_GRC_2            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_3            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_1_bits;

/* CSI1 GR Component Compander Constants Register 2 (IPU_CSI1_CPD_GRC_2) */
typedef struct{
__REG32 CSI1_CPD_GRC_4            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_5            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_2_bits;

/* CSI1 GR Component Compander Constants Register 3 (IPU_CSI1_CPD_GRC_3) */
typedef struct{
__REG32 CSI1_CPD_GRC_6            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_7            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_3_bits;

/* CSI1 GR Component Compander Constants Register 4 (IPU_CSI1_CPD_GRC_4) */
typedef struct{
__REG32 CSI1_CPD_GRC_8            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_9            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_4_bits;

/* CSI1 GR Component Compander Constants Register 5 (IPU_CSI1_CPD_GRC_5) */
typedef struct{
__REG32 CSI1_CPD_GRC_10           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_11           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_5_bits;

/* CSI1 GR Component Compander Constants Register 6 (IPU_CSI1_CPD_GRC_6) */
typedef struct{
__REG32 CSI1_CPD_GRC_12           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_13           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_6_bits;

/* CSI1 GR Component Compander Constants Register 7 (IPU_CSI1_CPD_GRC_7) */
typedef struct{
__REG32 CSI1_CPD_GRC_14           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GRC_15           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_grc_7_bits;

/* CSI1 GR Component Compander SLOPE Register 0 (IPU_CSI1_CPD_GRS_0) */
typedef struct{
__REG32 CSI1_CPD_GRS_0            : 8;
__REG32 CSI1_CPD_GRS_1            : 8;
__REG32 CSI1_CPD_GRS_2            : 8;
__REG32 CSI1_CPD_GRS_3            : 8;
} __ipu_csi1_cpd_grs_0_bits;

/* CSI1 GR Component Compander SLOPE Register 1 (IPU_CSI1_CPD_GRS_1) */
typedef struct{
__REG32 CSI1_CPD_GRS_4            : 8;
__REG32 CSI1_CPD_GRS_5            : 8;
__REG32 CSI1_CPD_GRS_6            : 8;
__REG32 CSI1_CPD_GRS_7            : 8;
} __ipu_csi1_cpd_grs_1_bits;

/* CSI1 GR Component Compander SLOPE Register 2 (IPU_CSI1_CPD_GRS_2) */
typedef struct{
__REG32 CSI1_CPD_GRS_8            : 8;
__REG32 CSI1_CPD_GRS_9            : 8;
__REG32 CSI1_CPD_GRS_10           : 8;
__REG32 CSI1_CPD_GRS_11           : 8;
} __ipu_csi1_cpd_grs_2_bits;

/* CSI1 GR Component Compander SLOPE Register 3 (IPU_CSI1_CPD_GRS_3) */
typedef struct{
__REG32 CSI1_CPD_GRS_12           : 8;
__REG32 CSI1_CPD_GRS_13           : 8;
__REG32 CSI1_CPD_GRS_14           : 8;
__REG32 CSI1_CPD_GRS_15           : 8;
} __ipu_csi1_cpd_grs_3_bits;

/* CSI1 GB Component Compander Constants Register 0 (IPU_CSI1_CPD_GBC_0) */
typedef struct{
__REG32 CSI1_CPD_GBC_0            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_1            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_0_bits;

/* CSI1 GB Component Compander Constants Register 1 (IPU_CSI1_CPD_GBC_1) */
typedef struct{
__REG32 CSI1_CPD_GBC_2            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_3            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_1_bits;

/* CSI1 GB Component Compander Constants Register 2 (IPU_CSI1_CPD_GBC_2) */
typedef struct{
__REG32 CSI1_CPD_GBC_4            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_5            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_2_bits;

/* CSI1 GB Component Compander Constants Register 3 (IPU_CSI1_CPD_GBC_3) */
typedef struct{
__REG32 CSI1_CPD_GBC_6            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_7            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_3_bits;

/* CSI1 GB Component Compander Constants Register 4 (IPU_CSI1_CPD_GBC_4) */
typedef struct{
__REG32 CSI1_CPD_GBC_8            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_9            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_4_bits;

/* CSI1 GB Component Compander Constants Register 5 (IPU_CSI1_CPD_GBC_5) */
typedef struct{
__REG32 CSI1_CPD_GBC_10           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_11           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_5_bits;

/* CSI1 GB Component Compander Constants Register 6 (IPU_CSI1_CPD_GBC_6) */
typedef struct{
__REG32 CSI1_CPD_GBC_12           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_13           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_6_bits;

/* CSI1 GB Component Compander Constants Register 7 (IPU_CSI1_CPD_GBC_7) */
typedef struct{
__REG32 CSI1_CPD_GBC_14           : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_GBC_15           : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_gbc_7_bits;

/* CSI1 GB Component Compander SLOPE Register 0 (IPU_CSI1_CPD_GBS_0) */
typedef struct{
__REG32 CSI1_CPD_GBS_0            : 8;
__REG32 CSI1_CPD_GBS_1            : 8;
__REG32 CSI1_CPD_GBS_2            : 8;
__REG32 CSI1_CPD_GBS_3            : 8;
} __ipu_csi1_cpd_gbs_0_bits;

/* CSI1 GB Component Compander SLOPE Register 1 (IPU_CSI1_CPD_GBS_1) */
typedef struct{
__REG32 CSI1_CPD_GBS_4            : 8;
__REG32 CSI1_CPD_GBS_5            : 8;
__REG32 CSI1_CPD_GBS_6            : 8;
__REG32 CSI1_CPD_GBS_7            : 8;
} __ipu_csi1_cpd_gbs_1_bits;

/* CSI1 GB Component Compander SLOPE Register 2 (IPU_CSI1_CPD_GBS_2) */
typedef struct{
__REG32 CSI1_CPD_GBS_8            : 8;
__REG32 CSI1_CPD_GBS_9            : 8;
__REG32 CSI1_CPD_GBS_10           : 8;
__REG32 CSI1_CPD_GBS_11           : 8;
} __ipu_csi1_cpd_gbs_2_bits;

/* CSI1 GB Component Compander SLOPE Register 3 (IPU_CSI1_CPD_GBS_3) */
typedef struct{
__REG32 CSI1_CPD_GBS_12           : 8;
__REG32 CSI1_CPD_GBS_13           : 8;
__REG32 CSI1_CPD_GBS_14           : 8;
__REG32 CSI1_CPD_GBS_15           : 8;
} __ipu_csi1_cpd_gbs_3_bits;

/* CSI1 Blue Component Compander Constants Register 0 (IPU_CSI1_CPD_BC_0) */
typedef struct{
__REG32 CSI1_CPD_BC_0             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_1             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_0_bits;

/* CSI1 Blue Component Compander Constants Register 1 (IPU_CSI1_CPD_BC_1) */
typedef struct{
__REG32 CSI1_CPD_BC_2             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_3             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_1_bits;

/* CSI1 Blue Component Compander Constants Register 2 (IPU_CSI1_CPD_BC_2) */
typedef struct{
__REG32 CSI1_CPD_BC_4             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_5             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_2_bits;

/* CSI1 Blue Component Compander Constants Register 3 (IPU_CSI1_CPD_BC_3) */
typedef struct{
__REG32 CSI1_CPD_BC_6             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_7             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_3_bits;

/* CSI1 Blue Component Compander Constants Register 4 (IPU_CSI1_CPD_BC_4) */
typedef struct{
__REG32 CSI1_CPD_BC_8             : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_9             : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_4_bits;

/* CSI1 Blue Component Compander Constants Register 5 (IPU_CSI1_CPD_BC_5) */
typedef struct{
__REG32 CSI1_CPD_BC_10            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_11            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_5_bits;

/* CSI1 Blue Component Compander Constants Register 6 (IPU_CSI1_CPD_BC_6) */
typedef struct{
__REG32 CSI1_CPD_BC_12            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_13            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_6_bits;

/* CSI1 Blue Component Compander Constants Register 7 (IPU_CSI1_CPD_BC_7) */
typedef struct{
__REG32 CSI1_CPD_BC_14            : 9;
__REG32                           : 7;
__REG32 CSI1_CPD_BC_15            : 9;
__REG32                           : 7;
} __ipu_csi1_cpd_bc_7_bits;

/* CSI1 Blue Component Compander SLOPE Register 0 (IPU_CSI1_CPD_BS_0) */
typedef struct{
__REG32 CSI1_CPD_BS_0             : 8;
__REG32 CSI1_CPD_BS_1             : 8;
__REG32 CSI1_CPD_BS_2             : 8;
__REG32 CSI1_CPD_BS_3             : 8;
} __ipu_csi1_cpd_bs_0_bits;

/* CSI1 Blue Component Compander SLOPE Register 1 (IPU_CSI1_CPD_BS_1) */
typedef struct{
__REG32 CSI1_CPD_BS_4             : 8;
__REG32 CSI1_CPD_BS_5             : 8;
__REG32 CSI1_CPD_BS_6             : 8;
__REG32 CSI1_CPD_BS_7             : 8;
} __ipu_csi1_cpd_bs_1_bits;

/* CSI1 Blue Component Compander SLOPE Register 2 (IPU_CSI1_CPD_BS_2) */
typedef struct{
__REG32 CSI1_CPD_BS_8             : 8;
__REG32 CSI1_CPD_BS_9             : 8;
__REG32 CSI1_CPD_BS_10            : 8;
__REG32 CSI1_CPD_BS_11            : 8;
} __ipu_csi1_cpd_bs_2_bits;

/* CSI1 Blue Component Compander SLOPE Register 3 (IPU_CSI1_CPD_BS_3) */
typedef struct{
__REG32 CSI1_CPD_BS_12            : 8;
__REG32 CSI1_CPD_BS_13            : 8;
__REG32 CSI1_CPD_BS_14            : 8;
__REG32 CSI1_CPD_BS_15            : 8;
} __ipu_csi1_cpd_bs_3_bits;

/* CSI1 Compander Offset Register 1 (IPU_CSI1_CPD_OFFSET1) */
typedef struct{
__REG32 CSI1_GR_OFFSET            :10;
__REG32 CSI1_GB_OFFSET            :10;
__REG32 CSI1_CPD_B_OFFSET         :10;
__REG32                           : 2;
} __ipu_csi1_cpd_offset1_bits;

/* CSI1 Compander Offset Register 2 (IPU_CSI1_CPD_OFFSET2) */
typedef struct{
__REG32 CSI1_CPD_R_OFFSET         :10;
__REG32                           :22;
} __ipu_csi1_cpd_offset2_bits;

/* DI0 General Register (IPU_DI0_GENERAL) */
typedef struct{
__REG32 di0_polarity_1            : 1;
__REG32 di0_polarity_2            : 1;
__REG32 di0_polarity_3            : 1;
__REG32 di0_polarity_4            : 1;
__REG32 di0_polarity_5            : 1;
__REG32 di0_polarity_6            : 1;
__REG32 di0_polarity_7            : 1;
__REG32 di0_polarity_8            : 1;
__REG32 di0_polarity_cs0          : 1;
__REG32 di0_polarity_cs1          : 1;
__REG32 di0_erm_vsync_sel         : 1;
__REG32 di0_err_treatment         : 1;
__REG32 di0_sync_count_sel        : 4;
__REG32                           : 1;
__REG32 di0_polarity_disp_clk     : 1;
__REG32 DI0_WATCHDOG_MODE         : 2;
__REG32 di0_clk_ext               : 1;
__REG32 di0_vsync_ext             : 1;
__REG32 di0_mask_sel              : 1;
__REG32 DI0_DISP_CLOCK_INIT       : 1;
__REG32 DI0_CLOCK_STOP_MODE       : 4;
__REG32 di0_disp_y_sel            : 3;
__REG32 di0_pin8_pin15_sel        : 1;
} __ipu_di0_general_bits;

/* DI0 Base Sync Clock Gen 0 Register (IPU_DI0_BS_CLKGEN0) */
typedef struct{
__REG32 di0_disp_clk_period       :12;
__REG32                           : 4;
__REG32 di0_disp_clk_offset       : 9;
__REG32                           : 7;
} __ipu_di0_bs_clkgen0_bits;

/* DI0 Base Sync Clock Gen 1 Register (IPU_DI0_BS_CLKGEN1) */
typedef struct{
__REG32 di0_disp_clk_up           : 9;
__REG32                           : 7;
__REG32 di0_disp_clk_down         : 9;
__REG32                           : 7;
} __ipu_di0_bs_clkgen1_bits;

/* DI0 Sync Wave Gen 1 Register 0 (IPU_DI0_SW_GEN0_1) */
typedef struct{
__REG32 di0_offset_resolution_1   : 3;
__REG32 di0_offset_value_1        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_1      : 3;
__REG32 di0_run_value_m1_1        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_1_bits;

/* DI0 Sync Wave Gen 2 Register 0 (IPU_DI0_SW_GEN0_2) */
typedef struct{
__REG32 di0_offset_resolution_2   : 3;
__REG32 di0_offset_value_2        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_2      : 3;
__REG32 di0_run_value_m1_2        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_2_bits;

/* DI0 Sync Wave Gen 3 Register 0 (IPU_DI0_SW_GEN0_3) */
typedef struct{
__REG32 di0_offset_resolution_3   : 3;
__REG32 di0_offset_value_3        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_3      : 3;
__REG32 di0_run_value_m1_3        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_3_bits;

/* DI0 Sync Wave Gen 4 Register 0 (IPU_DI0_SW_GEN0_4) */
typedef struct{
__REG32 di0_offset_resolution_4   : 3;
__REG32 di0_offset_value_4        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_4      : 3;
__REG32 di0_run_value_m1_4        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_4_bits;

/* DI0 Sync Wave Gen 5 Register 0 (IPU_DI0_SW_GEN0_5) */
typedef struct{
__REG32 di0_offset_resolution_5   : 3;
__REG32 di0_offset_value_5        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_5      : 3;
__REG32 di0_run_value_m1_5        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_5_bits;

/* DI0 Sync Wave Gen 6 Register 0 (IPU_DI0_SW_GEN0_6) */
typedef struct{
__REG32 di0_offset_resolution_6   : 3;
__REG32 di0_offset_value_6        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_6      : 3;
__REG32 di0_run_value_m1_6        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_6_bits;

/* DI0 Sync Wave Gen 7 Register 0 (IPU_DI0_SW_GEN0_7) */
typedef struct{
__REG32 di0_offset_resolution_7   : 3;
__REG32 di0_offset_value_7        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_7      : 3;
__REG32 di0_run_value_m1_7        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_7_bits;

/* DI0 Sync Wave Gen 8 Register 0 (IPU_DI0_SW_GEN0_8) */
typedef struct{
__REG32 di0_offset_resolution_8   : 3;
__REG32 di0_offset_value_8        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_8      : 3;
__REG32 di0_run_value_m1_8        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_8_bits;

/* DI0 Sync Wave Gen 9 Register 0 (IPU_DI0_SW_GEN0_9) */
typedef struct{
__REG32 di0_offset_resolution_9   : 3;
__REG32 di0_offset_value_9        :12;
__REG32                           : 1;
__REG32 di0_run_resolution_9      : 3;
__REG32 di0_run_value_m1_9        :12;
__REG32                           : 1;
} __ipu_di0_sw_gen0_9_bits;

/* DI0 Sync Wave Gen 1 Register 1 (IPU_DI0_SW_GEN1_1) */
typedef struct{
__REG32 di0_cnt_up_1                    : 9;
__REG32 di0_cnt_polarity_clr_sel_1      : 3;
__REG32 di0_cnt_polarity_trigger_sel_1  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_1                  : 9;
__REG32 di0_cnt_clr_sel_1               : 3;
__REG32 di0_cnt_auto_reload_1           : 1;
__REG32 di0_cnt_polarity_gen_en_1       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_1_bits;

/* DI0 Sync Wave Gen 2 Register 1 (IPU_DI0_SW_GEN1_1) */
typedef struct{
__REG32 di0_cnt_up_2                    : 9;
__REG32 di0_cnt_polarity_clr_sel_2      : 3;
__REG32 di0_cnt_polarity_trigger_sel_2  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_2                  : 9;
__REG32 di0_cnt_clr_sel_2               : 3;
__REG32 di0_cnt_auto_reload_2           : 1;
__REG32 di0_cnt_polarity_gen_en_2       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_2_bits;

/* DI0 Sync Wave Gen 3 Register 1 (IPU_DI0_SW_GEN1_3) */
typedef struct{
__REG32 di0_cnt_up_3                    : 9;
__REG32 di0_cnt_polarity_clr_sel_3      : 3;
__REG32 di0_cnt_polarity_trigger_sel_3  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_3                  : 9;
__REG32 di0_cnt_clr_sel_3               : 3;
__REG32 di0_cnt_auto_reload_3           : 1;
__REG32 di0_cnt_polarity_gen_en_3       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_3_bits;

/* DI0 Sync Wave Gen 4 Register 1 (IPU_DI0_SW_GEN1_4) */
typedef struct{
__REG32 di0_cnt_up_4                    : 9;
__REG32 di0_cnt_polarity_clr_sel_4      : 3;
__REG32 di0_cnt_polarity_trigger_sel_4  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_4                  : 9;
__REG32 di0_cnt_clr_sel_4               : 3;
__REG32 di0_cnt_auto_reload_4           : 1;
__REG32 di0_cnt_polarity_gen_en_4       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_4_bits;

/* DI0 Sync Wave Gen 5 Register 1 (IPU_DI0_SW_GEN1_5) */
typedef struct{
__REG32 di0_cnt_up_5                    : 9;
__REG32 di0_cnt_polarity_clr_sel_5      : 3;
__REG32 di0_cnt_polarity_trigger_sel_5  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_5                  : 9;
__REG32 di0_cnt_clr_sel_5               : 3;
__REG32 di0_cnt_auto_reload_5           : 1;
__REG32 di0_cnt_polarity_gen_en_5       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_5_bits;

/* DI0 Sync Wave Gen 6 Register 1 (IPU_DI0_SW_GEN1_6) */
typedef struct{
__REG32 di0_cnt_up_6                    : 9;
__REG32 di0_cnt_polarity_clr_sel_6      : 3;
__REG32 di0_cnt_polarity_trigger_sel_6  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_6                  : 9;
__REG32 di0_cnt_clr_sel_6               : 3;
__REG32 di0_cnt_auto_reload_6           : 1;
__REG32 di0_cnt_polarity_gen_en_6       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_6_bits;

/* DI0 Sync Wave Gen 7 Register 1 (IPU_DI0_SW_GEN1_7) */
typedef struct{
__REG32 di0_cnt_up_7                    : 9;
__REG32 di0_cnt_polarity_clr_sel_7      : 3;
__REG32 di0_cnt_polarity_trigger_sel_7  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_7                  : 9;
__REG32 di0_cnt_clr_sel_7               : 3;
__REG32 di0_cnt_auto_reload_7           : 1;
__REG32 di0_cnt_polarity_gen_en_7       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_7_bits;

/* DI0 Sync Wave Gen 8 Register 1 (IPU_DI0_SW_GEN1_8) */
typedef struct{
__REG32 di0_cnt_up_8                    : 9;
__REG32 di0_cnt_polarity_clr_sel_8      : 3;
__REG32 di0_cnt_polarity_trigger_sel_8  : 3;
__REG32                                 : 1;
__REG32 di0_cnt_down_8                  : 9;
__REG32 di0_cnt_clr_sel_8               : 3;
__REG32 di0_cnt_auto_reload_8           : 1;
__REG32 di0_cnt_polarity_gen_en_8       : 2;
__REG32                                 : 1;
} __ipu_di0_sw_gen1_8_bits;

/* DI0 Sync Wave Gen 9 Register 1 (IPU_DI0_SW_GEN1_9) */
typedef struct{
__REG32 di0_cnt_up_9                    : 9;
__REG32                                 : 6;
__REG32 di0_tag_sel_9                   : 1;
__REG32 di0_cnt_down_9                  : 9;
__REG32 di0_cnt_clr_sel_9               : 3;
__REG32 di0_cnt_auto_reload_9           : 1;
__REG32 di0_cnt_polarity_gen_en_9       : 3;
} __ipu_di0_sw_gen1_9_bits;

/* DI0 Sync Assistance Gen Register (IPU_DI0_SYNC_AS_GEN) */
typedef struct{
__REG32 di0_sync_start                  :12;
__REG32                                 : 1;
__REG32 di0_vsync_sel                   : 3;
__REG32                                 :12;
__REG32 di0_sync_start_en               : 1;
__REG32                                 : 3;
} __ipu_di0_sync_as_gen_bits;

/* DI0 Data Wave Gen 0 Register (IPU_DI0_DW_GEN_0) */
typedef union {
/* IPU_DI0_DW_GEN_0 */
struct{
__REG32 di0_serial_clk_0                : 2;
__REG32 di0_serial_rs_0                 : 2;
__REG32 di0_serial_valid_bits_0         : 5;
__REG32                                 : 5;
__REG32 di0_cst_0                       : 2;
__REG32 di0_start_period_0              : 8;
__REG32 di0_serial_period_0             : 8;
};
/* IPU_DI0_DW_GEN_0_SER */
struct{
__REG32 di0_pt_0_0                      : 2;
__REG32 di0_pt_1_0                      : 2;
__REG32 di0_pt_2_0                      : 2;
__REG32 di0_pt_3_0                      : 2;
__REG32 di0_pt_4_0                      : 2;
__REG32 di0_pt_5_0                      : 2;
__REG32 di0_pt_6_0                      : 2;
__REG32 _di0_cst_0                        : 2;
__REG32 di0_componnent_size_0           : 8;
__REG32 di0_access_size_0               : 8;
};
} __ipu_di0_dw_gen_0_bits;

/* DI0 Data Wave Gen 1 Register (IPU_DI0_DW_GEN_1) */
typedef union {
/* IPU_DI0_DW_GEN_1 */
struct{
__REG32 di0_serial_clk_1                : 2;
__REG32 di0_serial_rs_1                 : 2;
__REG32 di0_serial_valid_bits_1         : 5;
__REG32                                 : 5;
__REG32 di0_cst_1                       : 2;
__REG32 di0_start_period_1              : 8;
__REG32 di0_serial_period_1             : 8;
};
/* IPU_DI0_DW_GEN_1_SER */
struct{
__REG32 di0_pt_0_1                      : 2;
__REG32 di0_pt_1_1                      : 2;
__REG32 di0_pt_2_1                      : 2;
__REG32 di0_pt_3_1                      : 2;
__REG32 di0_pt_4_1                      : 2;
__REG32 di0_pt_5_1                      : 2;
__REG32 di0_pt_6_1                      : 2;
__REG32 _di0_cst_1                      : 2;
__REG32 di0_componnent_size_1           : 8;
__REG32 di0_access_size_1               : 8;
};
} __ipu_di0_dw_gen_1_bits;

/* DI0 Data Wave Gen 2 Register (IPU_DI0_DW_GEN_2) */
typedef union {
/* IPU_DI0_DW_GEN_2 */
struct{
__REG32 di0_serial_clk_2                : 2;
__REG32 di0_serial_rs_2                 : 2;
__REG32 di0_serial_valid_bits_2         : 5;
__REG32                                 : 5;
__REG32 di0_cst_2                       : 2;
__REG32 di0_start_period_2              : 8;
__REG32 di0_serial_period_2             : 8;
};
/* IPU_DI0_DW_GEN_2_SER */
struct{
__REG32 di0_pt_0_2                      : 2;
__REG32 di0_pt_1_2                      : 2;
__REG32 di0_pt_2_2                      : 2;
__REG32 di0_pt_3_2                      : 2;
__REG32 di0_pt_4_2                      : 2;
__REG32 di0_pt_5_2                      : 2;
__REG32 di0_pt_6_2                      : 2;
__REG32 _di0_cst_2                      : 2;
__REG32 di0_componnent_size_2           : 8;
__REG32 di0_access_size_2               : 8;
};
} __ipu_di0_dw_gen_2_bits;

/* DI0 Data Wave Gen 3 Register (IPU_DI0_DW_GEN_3) */
typedef union {
/* IPU_DI0_DW_GEN_3 */
struct{
__REG32 di0_serial_clk_3                : 2;
__REG32 di0_serial_rs_3                 : 2;
__REG32 di0_serial_valid_bits_3         : 5;
__REG32                                 : 5;
__REG32 di0_cst_3                       : 2;
__REG32 di0_start_period_3              : 8;
__REG32 di0_serial_period_3             : 8;
};
/* IPU_DI0_DW_GEN_3_SER */
struct{
__REG32 di0_pt_0_3                      : 2;
__REG32 di0_pt_1_3                      : 2;
__REG32 di0_pt_2_3                      : 2;
__REG32 di0_pt_3_3                      : 2;
__REG32 di0_pt_4_3                      : 2;
__REG32 di0_pt_5_3                      : 2;
__REG32 di0_pt_6_3                      : 2;
__REG32 _di0_cst_3                      : 2;
__REG32 di0_componnent_size_3           : 8;
__REG32 di0_access_size_3               : 8;
};
} __ipu_di0_dw_gen_3_bits;

/* DI0 Data Wave Gen 4 Register (IPU_DI0_DW_GEN_4) */
typedef union {
/* IPU_DI0_DW_GEN_4 */
struct{
__REG32 di0_serial_clk_4                : 2;
__REG32 di0_serial_rs_4                 : 2;
__REG32 di0_serial_valid_bits_4         : 5;
__REG32                                 : 5;
__REG32 di0_cst_4                       : 2;
__REG32 di0_start_period_4              : 8;
__REG32 di0_serial_period_4             : 8;
};
/* IPU_DI0_DW_GEN_4_SER */
struct{
__REG32 di0_pt_0_4                      : 2;
__REG32 di0_pt_1_4                      : 2;
__REG32 di0_pt_2_4                      : 2;
__REG32 di0_pt_3_4                      : 2;
__REG32 di0_pt_4_4                      : 2;
__REG32 di0_pt_5_4                      : 2;
__REG32 di0_pt_6_4                      : 2;
__REG32 _di0_cst_4                      : 2;
__REG32 di0_componnent_size_4           : 8;
__REG32 di0_access_size_4               : 8;
};
} __ipu_di0_dw_gen_4_bits;

/* DI0 Data Wave Gen 5 Register (IPU_DI0_DW_GEN_5) */
typedef union {
/* IPU_DI0_DW_GEN_5 */
struct{
__REG32 di0_serial_clk_5                : 2;
__REG32 di0_serial_rs_5                 : 2;
__REG32 di0_serial_valid_bits_5         : 5;
__REG32                                 : 5;
__REG32 di0_cst_5                       : 2;
__REG32 di0_start_period_5              : 8;
__REG32 di0_serial_period_5             : 8;
};
/* IPU_DI0_DW_GEN_5_SER */
struct{
__REG32 di0_pt_0_5                      : 2;
__REG32 di0_pt_1_5                      : 2;
__REG32 di0_pt_2_5                      : 2;
__REG32 di0_pt_3_5                      : 2;
__REG32 di0_pt_4_5                      : 2;
__REG32 di0_pt_5_5                      : 2;
__REG32 di0_pt_6_5                      : 2;
__REG32 _di0_cst_5                      : 2;
__REG32 di0_componnent_size_5           : 8;
__REG32 di0_access_size_5               : 8;
};
} __ipu_di0_dw_gen_5_bits;

/* DI0 Data Wave Gen 6 Register (IPU_DI0_DW_GEN_6) */
typedef union {
/* IPU_DI0_DW_GEN_6 */
struct{
__REG32 di0_serial_clk_6                : 2;
__REG32 di0_serial_rs_6                 : 2;
__REG32 di0_serial_valid_bits_6         : 5;
__REG32                                 : 5;
__REG32 di0_cst_6                       : 2;
__REG32 di0_start_period_6              : 8;
__REG32 di0_serial_period_6             : 8;
};
/* IPU_DI0_DW_GEN_6_SER */
struct{
__REG32 di0_pt_0_6                      : 2;
__REG32 di0_pt_1_6                      : 2;
__REG32 di0_pt_2_6                      : 2;
__REG32 di0_pt_3_6                      : 2;
__REG32 di0_pt_4_6                      : 2;
__REG32 di0_pt_5_6                      : 2;
__REG32 di0_pt_6_6                      : 2;
__REG32 _di0_cst_6                      : 2;
__REG32 di0_componnent_size_6           : 8;
__REG32 di0_access_size_6               : 8;
};
} __ipu_di0_dw_gen_6_bits;

/* DI0 Data Wave Gen 7 Register (IPU_DI0_DW_GEN_7) */
typedef union {
/* IPU_DI0_DW_GEN_7 */
struct{
__REG32 di0_serial_clk_7                : 2;
__REG32 di0_serial_rs_7                 : 2;
__REG32 di0_serial_valid_bits_7         : 5;
__REG32                                 : 5;
__REG32 di0_cst_7                       : 2;
__REG32 di0_start_period_7              : 8;
__REG32 di0_serial_period_7             : 8;
};
/* IPU_DI0_DW_GEN_7_SER */
struct{
__REG32 di0_pt_0_7                      : 2;
__REG32 di0_pt_1_7                      : 2;
__REG32 di0_pt_2_7                      : 2;
__REG32 di0_pt_3_7                      : 2;
__REG32 di0_pt_4_7                      : 2;
__REG32 di0_pt_5_7                      : 2;
__REG32 di0_pt_6_7                      : 2;
__REG32 _di0_cst_7                      : 2;
__REG32 di0_componnent_size_7           : 8;
__REG32 di0_access_size_7               : 8;
};
} __ipu_di0_dw_gen_7_bits;

/* DI0 Data Wave Gen 8 Register (IPU_DI0_DW_GEN_8) */
typedef union {
/* IPU_DI0_DW_GEN_8 */
struct{
__REG32 di0_serial_clk_8                : 2;
__REG32 di0_serial_rs_8                 : 2;
__REG32 di0_serial_valid_bits_8         : 5;
__REG32                                 : 5;
__REG32 di0_cst_8                       : 2;
__REG32 di0_start_period_8              : 8;
__REG32 di0_serial_period_8             : 8;
};
/* IPU_DI0_DW_GEN_8_SER */
struct{
__REG32 di0_pt_0_8                      : 2;
__REG32 di0_pt_1_8                      : 2;
__REG32 di0_pt_2_8                      : 2;
__REG32 di0_pt_3_8                      : 2;
__REG32 di0_pt_4_8                      : 2;
__REG32 di0_pt_5_8                      : 2;
__REG32 di0_pt_6_8                      : 2;
__REG32 _di0_cst_8                      : 2;
__REG32 di0_componnent_size_8           : 8;
__REG32 di0_access_size_8               : 8;
};
} __ipu_di0_dw_gen_8_bits;

/* DI0 Data Wave Gen 9 Register (IPU_DI0_DW_GEN_9) */
typedef union {
/* IPU_DI0_DW_GEN_9 */
struct{
__REG32 di0_serial_clk_9                : 2;
__REG32 di0_serial_rs_9                 : 2;
__REG32 di0_serial_valid_bits_9         : 5;
__REG32                                 : 5;
__REG32 di0_cst_9                       : 2;
__REG32 di0_start_period_9              : 8;
__REG32 di0_serial_period_9             : 8;
};
/* IPU_DI0_DW_GEN_9_SER */
struct{
__REG32 di0_pt_0_9                      : 2;
__REG32 di0_pt_1_9                      : 2;
__REG32 di0_pt_2_9                      : 2;
__REG32 di0_pt_3_9                      : 2;
__REG32 di0_pt_4_9                      : 2;
__REG32 di0_pt_5_9                      : 2;
__REG32 di0_pt_6_9                      : 2;
__REG32 _di0_cst_9                      : 2;
__REG32 di0_componnent_size_9           : 8;
__REG32 di0_access_size_9               : 8;
};
} __ipu_di0_dw_gen_9_bits;

/* DI0 Data Wave Gen 10 Register (IPU_DI0_DW_GEN_10) */
typedef union {
/* IPU_DI0_DW_GEN_10 */
struct{
__REG32 di0_serial_clk_10               : 2;
__REG32 di0_serial_rs_10                : 2;
__REG32 di0_serial_valid_bits_10        : 5;
__REG32                                 : 5;
__REG32 di0_cst_10                      : 2;
__REG32 di0_start_period_10             : 8;
__REG32 di0_serial_period_10            : 8;
};
/* IPU_DI0_DW_GEN_10_SER */
struct{
__REG32 di0_pt_0_10                     : 2;
__REG32 di0_pt_1_10                     : 2;
__REG32 di0_pt_2_10                     : 2;
__REG32 di0_pt_3_10                     : 2;
__REG32 di0_pt_4_10                     : 2;
__REG32 di0_pt_5_10                     : 2;
__REG32 di0_pt_6_10                     : 2;
__REG32 _di0_cst_10                     : 2;
__REG32 di0_componnent_size_10          : 8;
__REG32 di0_access_size_10              : 8;
};
} __ipu_di0_dw_gen_10_bits;

/* DI0 Data Wave Gen 11 Register (IPU_DI0_DW_GEN_11) */
typedef union {
/* IPU_DI0_DW_GEN_11 */
struct{
__REG32 di0_serial_clk_11               : 2;
__REG32 di0_serial_rs_11                : 2;
__REG32 di0_serial_valid_bits_11        : 5;
__REG32                                 : 5;
__REG32 di0_cst_11                      : 2;
__REG32 di0_start_period_11             : 8;
__REG32 di0_serial_period_11            : 8;
};
/* IPU_DI0_DW_GEN_11_SER */
struct{
__REG32 di0_pt_0_11                     : 2;
__REG32 di0_pt_1_11                     : 2;
__REG32 di0_pt_2_11                     : 2;
__REG32 di0_pt_3_11                     : 2;
__REG32 di0_pt_4_11                     : 2;
__REG32 di0_pt_5_11                     : 2;
__REG32 di0_pt_6_11                     : 2;
__REG32 _di0_cst_11                     : 2;
__REG32 di0_componnent_size_11          : 8;
__REG32 di0_access_size_11              : 8;
};
} __ipu_di0_dw_gen_11_bits;

/* DI0 Data Wave Set 0 0 Register (IPU_DI0_DW_SET0_0) */
typedef struct{
__REG32 di0_data_cnt_up0_0              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_0            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_0_bits;

/* DI0 Data Wave Set 0 1 Register (IPU_DI0_DW_SET0_1) */
typedef struct{
__REG32 di0_data_cnt_up0_1              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_1            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_1_bits;

/* DI0 Data Wave Set 0 2 Register (IPU_DI0_DW_SET0_2) */
typedef struct{
__REG32 di0_data_cnt_up0_2              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_2            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_2_bits;

/* DI0 Data Wave Set 0 3 Register (IPU_DI0_DW_SET0_3) */
typedef struct{
__REG32 di0_data_cnt_up0_3              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_3            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_3_bits;

/* DI0 Data Wave Set 0 4 Register (IPU_DI0_DW_SET0_4) */
typedef struct{
__REG32 di0_data_cnt_up0_4              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_4            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_4_bits;

/* DI0 Data Wave Set 0 5 Register (IPU_DI0_DW_SET0_5) */
typedef struct{
__REG32 di0_data_cnt_up0_5              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_5            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_5_bits;

/* DI0 Data Wave Set 0 6 Register (IPU_DI0_DW_SET0_6) */
typedef struct{
__REG32 di0_data_cnt_up0_6              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_6            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_6_bits;

/* DI0 Data Wave Set 0 7 Register (IPU_DI0_DW_SET0_7) */
typedef struct{
__REG32 di0_data_cnt_up0_7              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_7            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_7_bits;

/* DI0 Data Wave Set 0 8 Register (IPU_DI0_DW_SET0_8) */
typedef struct{
__REG32 di0_data_cnt_up0_8              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_8            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_8_bits;

/* DI0 Data Wave Set 0 9 Register (IPU_DI0_DW_SET0_9) */
typedef struct{
__REG32 di0_data_cnt_up0_9              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_9            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_9_bits;

/* DI0 Data Wave Set 0 10 Register (IPU_DI0_DW_SET0_10) */
typedef struct{
__REG32 di0_data_cnt_up0_10             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_10           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_10_bits;

/* DI0 Data Wave Set 0 11 Register (IPU_DI0_DW_SET0_11) */
typedef struct{
__REG32 di0_data_cnt_up0_11             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down0_11           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set0_11_bits;

/* DI0 Data Wave Set 1 0 Register (IPU_DI1_DW_SET1_0) */
typedef struct{
__REG32 di0_data_cnt_up1_0              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_0            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_0_bits;

/* DI0 Data Wave Set 1 1 Register (IPU_DI0_DW_SET1_1) */
typedef struct{
__REG32 di0_data_cnt_up1_1              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_1            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_1_bits;

/* DI0 Data Wave Set 1 2 Register (IPU_DI0_DW_SET1_2) */
typedef struct{
__REG32 di0_data_cnt_up1_2              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_2            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_2_bits;

/* DI0 Data Wave Set 1 3 Register (IPU_DI0_DW_SET1_3) */
typedef struct{
__REG32 di0_data_cnt_up1_3              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_3            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_3_bits;

/* DI0 Data Wave Set 1 4 Register (IPU_DI0_DW_SET1_4) */
typedef struct{
__REG32 di0_data_cnt_up1_4              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_4            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_4_bits;

/* DI0 Data Wave Set 1 5 Register (IPU_DI0_DW_SET1_5) */
typedef struct{
__REG32 di0_data_cnt_up1_5              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_5            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_5_bits;

/* DI0 Data Wave Set 1 6 Register (IPU_DI0_DW_SET1_6) */
typedef struct{
__REG32 di0_data_cnt_up1_6              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_6            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_6_bits;

/* DI0 Data Wave Set 1 7 Register (IPU_DI0_DW_SET1_7) */
typedef struct{
__REG32 di0_data_cnt_up1_7              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_7            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_7_bits;

/* DI0 Data Wave Set 1 8 Register (IPU_DI0_DW_SET1_8) */
typedef struct{
__REG32 di0_data_cnt_up1_8              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_8            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_8_bits;

/* DI0 Data Wave Set 1 9 Register (IPU_DI0_DW_SET1_9) */
typedef struct{
__REG32 di0_data_cnt_up1_9              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_9            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_9_bits;

/* DI0 Data Wave Set 1 10 Register (IPU_DI0_DW_SET1_10) */
typedef struct{
__REG32 di0_data_cnt_up1_10             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_10           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_10_bits;

/* DI0 Data Wave Set 1 11 Register (IPU_DI0_DW_SET1_11) */
typedef struct{
__REG32 di0_data_cnt_up1_11             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down1_11           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set1_11_bits;

/* DI0 Data Wave Set 2 0 Register (IPU_DI0_DW_SET2_0) */
typedef struct{
__REG32 di0_data_cnt_up2_0              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_0            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_0_bits;

/* DI0 Data Wave Set 2 1 Register (IPU_DI0_DW_SET2_1) */
typedef struct{
__REG32 di0_data_cnt_up2_1              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_1            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_1_bits;

/* DI0 Data Wave Set 2 2 Register (IPU_DI0_DW_SET2_2) */
typedef struct{
__REG32 di0_data_cnt_up2_2              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_2            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_2_bits;

/* DI0 Data Wave Set 2 3 Register (IPU_DI0_DW_SET2_3) */
typedef struct{
__REG32 di0_data_cnt_up2_3              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_3            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_3_bits;

/* DI0 Data Wave Set 2 4 Register (IPU_DI0_DW_SET2_4) */
typedef struct{
__REG32 di0_data_cnt_up2_4              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_4            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_4_bits;

/* DI0 Data Wave Set 2 5 Register (IPU_DI0_DW_SET2_5) */
typedef struct{
__REG32 di0_data_cnt_up2_5              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_5            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_5_bits;

/* DI0 Data Wave Set 2 6 Register (IPU_DI0_DW_SET2_6) */
typedef struct{
__REG32 di0_data_cnt_up2_6              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_6            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_6_bits;

/* DI0 Data Wave Set 2 7 Register (IPU_DI0_DW_SET2_7) */
typedef struct{
__REG32 di0_data_cnt_up2_7              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_7            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_7_bits;

/* DI0 Data Wave Set 2 8 Register (IPU_DI0_DW_SET2_8) */
typedef struct{
__REG32 di0_data_cnt_up2_8              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_8            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_8_bits;

/* DI0 Data Wave Set 2 9 Register (IPU_DI0_DW_SET2_9) */
typedef struct{
__REG32 di0_data_cnt_up2_9              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_9            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_9_bits;

/* DI0 Data Wave Set 2 10 Register (IPU_DI0_DW_SET2_10) */
typedef struct{
__REG32 di0_data_cnt_up2_10             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_10           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_10_bits;

/* DI0 Data Wave Set 2 11 Register (IPU_DI0_DW_SET2_11) */
typedef struct{
__REG32 di0_data_cnt_up2_11             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down2_11           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set2_11_bits;

/* DI0 Data Wave Set 3 0 Register (IPU_DI0_DW_SET3_0) */
typedef struct{
__REG32 di0_data_cnt_up3_0              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_0            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_0_bits;

/* DI0 Data Wave Set 3 1 Register (IPU_DI0_DW_SET3_1) */
typedef struct{
__REG32 di0_data_cnt_up3_1              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_1            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_1_bits;

/* DI0 Data Wave Set 3 2 Register (IPU_DI0_DW_SET3_2) */
typedef struct{
__REG32 di0_data_cnt_up3_2              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_2            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_2_bits;

/* DI0 Data Wave Set 3 3 Register (IPU_DI0_DW_SET3_3) */
typedef struct{
__REG32 di0_data_cnt_up3_3              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_3            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_3_bits;

/* DI0 Data Wave Set 3 4 Register (IPU_DI0_DW_SET3_4) */
typedef struct{
__REG32 di0_data_cnt_up3_4              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_4            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_4_bits;

/* DI0 Data Wave Set 3 5 Register (IPU_DI0_DW_SET3_5) */
typedef struct{
__REG32 di0_data_cnt_up3_5              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_5            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_5_bits;

/* DI0 Data Wave Set 3 6 Register (IPU_DI0_DW_SET3_6) */
typedef struct{
__REG32 di0_data_cnt_up3_6              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_6            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_6_bits;

/* DI0 Data Wave Set 3 7 Register (IPU_DI0_DW_SET3_7) */
typedef struct{
__REG32 di0_data_cnt_up3_7              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_7            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_7_bits;

/* DI0 Data Wave Set 3 8 Register (IPU_DI0_DW_SET3_8) */
typedef struct{
__REG32 di0_data_cnt_up3_8              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_8            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_8_bits;

/* DI0 Data Wave Set 3 9 Register (IPU_DI0_DW_SET3_9) */
typedef struct{
__REG32 di0_data_cnt_up3_9              : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_9            : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_9_bits;

/* DI0 Data Wave Set 3 10 Register (IPU_DI0_DW_SET3_10) */
typedef struct{
__REG32 di0_data_cnt_up3_10             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_10           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_10_bits;

/* DI0 Data Wave Set 3 11 Register (IPU_DI0_DW_SET3_11) */
typedef struct{
__REG32 di0_data_cnt_up3_11             : 9;
__REG32                                 : 7;
__REG32 di0_data_cnt_down3_11           : 9;
__REG32                                 : 7;
} __ipu_di0_dw_set3_11_bits;

/* DI0 Step Repeat 1 Registers (IPU_DI0_STP_REP_1) */
typedef struct{
__REG32 di0_step_repeat_1               :12;
__REG32                                 : 4;
__REG32 di0_step_repeat_2               :12;
__REG32                                 : 4;
} __ipu_di0_stp_rep_1_bits;

/* DI0 Step Repeat 2 Registers (IPU_DI0_STP_REP_2) */
typedef struct{
__REG32 di0_step_repeat_3               :12;
__REG32                                 : 4;
__REG32 di0_step_repeat_4               :12;
__REG32                                 : 4;
} __ipu_di0_stp_rep_2_bits;

/* DI0 Step Repeat 3 Registers (IPU_DI0_STP_REP_3) */
typedef struct{
__REG32 di0_step_repeat_5               :12;
__REG32                                 : 4;
__REG32 di0_step_repeat_6               :12;
__REG32                                 : 4;
} __ipu_di0_stp_rep_3_bits;

/* DI0 Step Repeat 4 Registers (IPU_DI0_STP_REP_4) */
typedef struct{
__REG32 di0_step_repeat_7               :12;
__REG32                                 : 4;
__REG32 di0_step_repeat_8               :12;
__REG32                                 : 4;
} __ipu_di0_stp_rep_4_bits;

/* DI0 Step Repeat 9 Registers (IPU_DI0_STP_REP_9) */
typedef struct{
__REG32 di0_step_repeat_9               :12;
__REG32                                 :20;
} __ipu_di0_stp_rep_9_bits;

/* DI0 Serial Display Control Register (IPU_DI0_SER_CONF) */
typedef struct{
__REG32 DI0_WAIT4SERIAL                 : 1;
__REG32 DI0_SERIAL_CS_POLARITY          : 1;
__REG32 DI0_SERIAL_RS_POLARITY          : 1;
__REG32 DI0_SERIAL_DATA_POLARITY        : 1;
__REG32 DI0_SER_CLK_POLARITY            : 1;
__REG32 DI0_LLA_SER_ACCESS              : 1;
__REG32                                 : 2;
__REG32 DI0_SERIAL_LATCH                : 8;
__REG32 DI0_SERIAL_LLA_PNTR_RS_W_0      : 4;
__REG32 DI0_SERIAL_LLA_PNTR_RS_W_1      : 4;
__REG32 DI0_SERIAL_LLA_PNTR_RS_R_0      : 4;
__REG32 DI0_SERIAL_LLA_PNTR_RS_R_1      : 4;
} __ipu_di0_ser_conf_bits;

/* DI0 Special Signals Control Register (IPU_DI0_SSC) */
typedef struct{
__REG32 DI0_BYTE_EN_PNTR                : 3;
__REG32 DI0_BYTE_EN_RD_IN               : 1;
__REG32                                 : 1;
__REG32 DI0_WAIT_ON                     : 1;
__REG32                                 :10;
__REG32 DI0_CS_ERM                      : 1;
__REG32 DI0_PIN11_ERM                   : 1;
__REG32 DI0_PIN12_ERM                   : 1;
__REG32 DI0_PIN13_ERM                   : 1;
__REG32 DI0_PIN14_ERM                   : 1;
__REG32 DI0_PIN15_ERM                   : 1;
__REG32 DI0_PIN16_ERM                   : 1;
__REG32 DI0_PIN17_ERM                   : 1;
__REG32                                 : 8;
} __ipu_di0_ssc_bits;

/* DI0 Polarity Register (IPU_DI0_POL) */
typedef struct{
__REG32 di0_drdy_polarity_11            : 1;
__REG32 di0_drdy_polarity_12            : 1;
__REG32 di0_drdy_polarity_13            : 1;
__REG32 di0_drdy_polarity_14            : 1;
__REG32 di0_drdy_polarity_15            : 1;
__REG32 di0_drdy_polarity_16            : 1;
__REG32 di0_drdy_polarity_17            : 1;
__REG32 DI0_DRDY_DATA_POLARITY          : 1;
__REG32 di0_cs0_polarity_11             : 1;
__REG32 di0_cs0_polarity_12             : 1;
__REG32 di0_cs0_polarity_13             : 1;
__REG32 di0_cs0_polarity_14             : 1;
__REG32 di0_cs0_polarity_15             : 1;
__REG32 di0_cs0_polarity_16             : 1;
__REG32 di0_cs0_polarity_17             : 1;
__REG32 DI0_CS0_DATA_POLARITY           : 1;
__REG32 di0_cs1_polarity_11             : 1;
__REG32 di0_cs1_polarity_12             : 1;
__REG32 di0_cs1_polarity_13             : 1;
__REG32 di0_cs1_polarity_14             : 1;
__REG32 di0_cs1_polarity_15             : 1;
__REG32 di0_cs1_polarity_16             : 1;
__REG32 di0_cs1_polarity_17             : 1;
__REG32 DI0_CS1_DATA_POLARITY           : 1;
__REG32 DI0_CS0_BYTE_EN_POLARITY        : 1;
__REG32 DI0_CS1_BYTE_EN_POLARITY        : 1;
__REG32 DI0_WAIT_POLARITY               : 1;
__REG32                                 : 5;
} __ipu_di0_pol_bits;

/* DI0 Active Window 0 Register (IPU_DI0_AW0) */
typedef struct{
__REG32 DI0_AW_HSTART                   :12;
__REG32 DI0_AW_HCOUNT_SEL               : 4;
__REG32 DI0_AW_HEND                     :12;
__REG32 DI0_AW_TRIG_SEL                 : 4;
} __ipu_di0_aw0_bits;

/* DI0 Active Window 1 Register (IPU_DI0_AW1) */
typedef struct{
__REG32 DI0_AW_VSTART                   :12;
__REG32 DI0_AW_VCOUNT_SEL               : 4;
__REG32 DI0_AW_VEND                     :12;
__REG32                                 : 4;
} __ipu_di0_aw1_bits;

/* DI0 Screen Configuration Register (IPU_DI0_SCR_CONF) */
typedef struct{
__REG32 DI0_SCREEN_HEIGHT               :12;
__REG32                                 :20;
} __ipu_di0_scr_conf_bits;

/* DI0 Status Register (IPU_DI0_STAT) */
typedef struct{
__REG32 DI0_READ_FIFO_EMPTY             : 1;
__REG32 DI0_READ_FIFO_FULL              : 1;
__REG32 DI0_READ_CNTR_EMPTY             : 1;
__REG32 DI0_CNTR_FIFO_FULL              : 1;
__REG32                                 :28;
} __ipu_di0_stat_bits;

/* DI1 General Register (IPU_DI1_GENERAL) */
typedef struct{
__REG32 di1_polarity_1                  : 1;
__REG32 di1_polarity_2                  : 1;
__REG32 di1_polarity_3                  : 1;
__REG32 di1_polarity_4                  : 1;
__REG32 di1_polarity_5                  : 1;
__REG32 di1_polarity_6                  : 1;
__REG32 di1_polarity_7                  : 1;
__REG32 di1_polarity_8                  : 1;
__REG32 di1_polarity_cs0                : 1;
__REG32 di1_polarity_cs1                : 1;
__REG32 di1_erm_vsync_sel               : 1;
__REG32 di1_err_treatment               : 1;
__REG32 di1_sync_count_sel              : 4;
__REG32                                 : 1;
__REG32 di1_polarity_disp_clk           : 1;
__REG32 DI1_WATCHDOG_MODE               : 2;
__REG32 di1_clk_ext                     : 1;
__REG32 di1_vsync_ext                   : 1;
__REG32 di1_mask_sel                    : 1;
__REG32 DI1_DISP_CLOCK_INIT             : 1;
__REG32 DI1_CLOCK_STOP_MODE             : 4;
__REG32 di1_disp_y_sel                  : 3;
__REG32 di1_pin8_pin15_sel              : 1;
} __ipu_di1_general_bits;

/* DI1 Base Sync Clock Gen 0 Register (IPU_DI1_BS_CLKGEN0) */
typedef struct{
__REG32 di1_disp_clk_period             :12;
__REG32                                 : 4;
__REG32 di1_disp_clk_offset             : 9;
__REG32                                 : 7;
} __ipu_di1_bs_clkgen0_bits;

/* DI1 Base Sync Clock Gen 1 Register (IPU_DI1_BS_CLKGEN1) */
typedef struct{
__REG32 di1_disp_clk_up                 : 9;
__REG32                                 : 7;
__REG32 di1_disp_clk_down               : 9;
__REG32                                 : 7;
} __ipu_di1_bs_clkgen1_bits;

/* DI1 Sync Wave Gen 1 Register 0 (IPU_DI1_SW_GEN0_1) */
typedef struct{
__REG32 di1_offset_resolution_1         : 3;
__REG32 di1_offset_value_1              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_1            : 3;
__REG32 di1_run_value_m1_1              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_1_bits;

/* DI1 Sync Wave Gen 2 Register 0 (IPU_DI1_SW_GEN0_2) */
typedef struct{
__REG32 di1_offset_resolution_2         : 3;
__REG32 di1_offset_value_2              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_2            : 3;
__REG32 di1_run_value_m1_2              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_2_bits;

/* DI1 Sync Wave Gen 3 Register 0 (IPU_DI1_SW_GEN0_3) */
typedef struct{
__REG32 di1_offset_resolution_3         : 3;
__REG32 di1_offset_value_3              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_3            : 3;
__REG32 di1_run_value_m1_3              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_3_bits;

/* DI1 Sync Wave Gen 4 Register 0 (IPU_DI1_SW_GEN0_4) */
typedef struct{
__REG32 di1_offset_resolution_4         : 3;
__REG32 di1_offset_value_4              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_4            : 3;
__REG32 di1_run_value_m1_4              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_4_bits;

/* DI1 Sync Wave Gen 5 Register 0 (IPU_DI1_SW_GEN0_5) */
typedef struct{
__REG32 di1_offset_resolution_5         : 3;
__REG32 di1_offset_value_5              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_5            : 3;
__REG32 di1_run_value_m1_5              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_5_bits;

/* DI1 Sync Wave Gen 6 Register 0 (IPU_DI1_SW_GEN0_6) */
typedef struct{
__REG32 di1_offset_resolution_6         : 3;
__REG32 di1_offset_value_6              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_6            : 3;
__REG32 di1_run_value_m1_6              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_6_bits;

/* DI1 Sync Wave Gen 7 Register 0 (IPU_DI1_SW_GEN0_7) */
typedef struct{
__REG32 di1_offset_resolution_7         : 3;
__REG32 di1_offset_value_7              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_7            : 3;
__REG32 di1_run_value_m1_7              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_7_bits;

/* DI1 Sync Wave Gen 8 Register 0 (IPU_DI1_SW_GEN0_8) */
typedef struct{
__REG32 di1_offset_resolution_8         : 3;
__REG32 di1_offset_value_8              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_8            : 3;
__REG32 di1_run_value_m1_8              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_8_bits;

/* DI1 Sync Wave Gen 9 Register 0 (IPU_DI1_SW_GEN0_9) */
typedef struct{
__REG32 di1_offset_resolution_9         : 3;
__REG32 di1_offset_value_9              :12;
__REG32                                 : 1;
__REG32 di1_run_resolution_9            : 3;
__REG32 di1_run_value_m1_9              :12;
__REG32                                 : 1;
} __ipu_di1_sw_gen0_9_bits;

/* DI1 Sync Wave Gen 1 Register 1 (IPU_DI1_SW_GEN1_1) */
typedef struct{
__REG32 di1_cnt_up_1                    : 9;
__REG32 di1_cnt_polarity_clr_sel_1      : 3;
__REG32 di1_cnt_polarity_trigger_sel_1  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_1                  : 9;
__REG32 di1_cnt_clr_sel_1               : 3;
__REG32 di1_cnt_auto_reload_1           : 1;
__REG32 di1_cnt_polarity_gen_en_1       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_1_bits;

/* DI1 Sync Wave Gen 2 Register 1 (IPU_DI1_SW_GEN1_2) */
typedef struct{
__REG32 di1_cnt_up_2                    : 9;
__REG32 di1_cnt_polarity_clr_sel_2      : 3;
__REG32 di1_cnt_polarity_trigger_sel_2  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_2                  : 9;
__REG32 di1_cnt_clr_sel_2               : 3;
__REG32 di1_cnt_auto_reload_2           : 1;
__REG32 di1_cnt_polarity_gen_en_2       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_2_bits;

/* DI1 Sync Wave Gen 3 Register 1 (IPU_DI1_SW_GEN1_3) */
typedef struct{
__REG32 di1_cnt_up_3                    : 9;
__REG32 di1_cnt_polarity_clr_sel_3      : 3;
__REG32 di1_cnt_polarity_trigger_sel_3  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_3                  : 9;
__REG32 di1_cnt_clr_sel_3               : 3;
__REG32 di1_cnt_auto_reload_3           : 1;
__REG32 di1_cnt_polarity_gen_en_3       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_3_bits;

/* DI1 Sync Wave Gen 4 Register 1 (IPU_DI1_SW_GEN1_4) */
typedef struct{
__REG32 di1_cnt_up_4                    : 9;
__REG32 di1_cnt_polarity_clr_sel_4      : 3;
__REG32 di1_cnt_polarity_trigger_sel_4  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_4                  : 9;
__REG32 di1_cnt_clr_sel_4               : 3;
__REG32 di1_cnt_auto_reload_4           : 1;
__REG32 di1_cnt_polarity_gen_en_4       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_4_bits;

/* DI1 Sync Wave Gen 5 Register 1 (IPU_DI1_SW_GEN1_5) */
typedef struct{
__REG32 di1_cnt_up_5                    : 9;
__REG32 di1_cnt_polarity_clr_sel_5      : 3;
__REG32 di1_cnt_polarity_trigger_sel_5  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_5                  : 9;
__REG32 di1_cnt_clr_sel_5               : 3;
__REG32 di1_cnt_auto_reload_5           : 1;
__REG32 di1_cnt_polarity_gen_en_5       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_5_bits;

/* DI1 Sync Wave Gen 6 Register 1 (IPU_DI1_SW_GEN1_6) */
typedef struct{
__REG32 di1_cnt_up_6                    : 9;
__REG32 di1_cnt_polarity_clr_sel_6      : 3;
__REG32 di1_cnt_polarity_trigger_sel_6  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_6                  : 9;
__REG32 di1_cnt_clr_sel_6               : 3;
__REG32 di1_cnt_auto_reload_6           : 1;
__REG32 di1_cnt_polarity_gen_en_6       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_6_bits;

/* DI1 Sync Wave Gen 7 Register 1 (IPU_DI1_SW_GEN1_7) */
typedef struct{
__REG32 di1_cnt_up_7                    : 9;
__REG32 di1_cnt_polarity_clr_sel_7      : 3;
__REG32 di1_cnt_polarity_trigger_sel_7  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_7                  : 9;
__REG32 di1_cnt_clr_sel_7               : 3;
__REG32 di1_cnt_auto_reload_7           : 1;
__REG32 di1_cnt_polarity_gen_en_7       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_7_bits;

/* DI1 Sync Wave Gen 8 Register 1 (IPU_DI1_SW_GEN1_8) */
typedef struct{
__REG32 di1_cnt_up_8                    : 9;
__REG32 di1_cnt_polarity_clr_sel_8      : 3;
__REG32 di1_cnt_polarity_trigger_sel_8  : 3;
__REG32                                 : 1;
__REG32 di1_cnt_down_8                  : 9;
__REG32 di1_cnt_clr_sel_8               : 3;
__REG32 di1_cnt_auto_reload_8           : 1;
__REG32 di1_cnt_polarity_gen_en_8       : 2;
__REG32                                 : 1;
} __ipu_di1_sw_gen1_8_bits;

/* DI1 Sync Wave Gen 9 Register 1 (IPU_DI1_SW_GEN1_9) */
typedef struct{
__REG32 di1_cnt_up_9                    : 9;
__REG32                                 : 6;
__REG32 di1_tag_sel_9                   : 1;
__REG32 di1_cnt_down_9                  : 9;
__REG32 di1_cnt_clr_sel_9               : 3;
__REG32 di1_cnt_auto_reload_9           : 1;
__REG32 di1_gentime_sel_9               : 3;
} __ipu_di1_sw_gen1_9_bits;

/* DI1 Sync Assistance Gen Register (IPU_DI1_SYNC_AS_GEN) */
typedef struct{
__REG32 di1_sync_start                  :12;
__REG32                                 : 1;
__REG32 di1_vsync_sel                   : 3;
__REG32                                 :12;
__REG32 di1_sync_start_en               : 1;
__REG32                                 : 3;
} __ipu_di1_sync_as_gen_bits;

/* DI1 Data Wave Gen 0 Register (IPU_DI1_DW_GEN_0) */
typedef union{
/* IPU_DI1_DW_GEN_0 */
struct {
__REG32 di1_pt_0_0                      : 2;
__REG32 di1_pt_1_0                      : 2;
__REG32 di1_pt_2_0                      : 2;
__REG32 di1_pt_3_0                      : 2;
__REG32 di1_pt_4_0                      : 2;
__REG32 di1_pt_5_0                      : 2;
__REG32 di1_pt_6_0                      : 2;
__REG32 di1_cst_0                       : 2;
__REG32 componnent_size_0               : 8;
__REG32 di1_access_size_0               : 8;
};
/* IPU_DI1_DW_GEN_0_SER */
struct {
__REG32 di1_serial_clk_0                : 2;
__REG32 di1_serial_rs_0                 : 2;
__REG32 di1_serial_valid_bits_0         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_0                        : 2;
__REG32 di1_start_period_0              : 8;
__REG32 di1_serial_period_0             : 8;
};
} __ipu_di1_dw_gen_0_bits;

/* DI1 Data Wave Gen 1 Register (IPU_DI1_DW_GEN_1) */
typedef union{
/* IPU_DI1_DW_GEN_1 */
struct {
__REG32 di1_pt_0_1                      : 2;
__REG32 di1_pt_1_1                      : 2;
__REG32 di1_pt_2_1                      : 2;
__REG32 di1_pt_3_1                      : 2;
__REG32 di1_pt_4_1                      : 2;
__REG32 di1_pt_5_1                      : 2;
__REG32 di1_pt_6_1                      : 2;
__REG32 di1_cst_1                       : 2;
__REG32 componnent_size_1               : 8;
__REG32 di1_access_size_1               : 8;
};
/* IPU_DI1_DW_GEN_1_SER */
struct {
__REG32 di1_serial_clk_1                : 2;
__REG32 di1_serial_rs_1                 : 2;
__REG32 di1_serial_valid_bits_1         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_1                        : 2;
__REG32 di1_start_period_1              : 8;
__REG32 di1_serial_period_1             : 8;
};
} __ipu_di1_dw_gen_1_bits;

/* DI1 Data Wave Gen 2 Register (IPU_DI1_DW_GEN_2) */
typedef union{
/* IPU_DI1_DW_GEN_2 */
struct {
__REG32 di1_pt_0_2                      : 2;
__REG32 di1_pt_1_2                      : 2;
__REG32 di1_pt_2_2                      : 2;
__REG32 di1_pt_3_2                      : 2;
__REG32 di1_pt_4_2                      : 2;
__REG32 di1_pt_5_2                      : 2;
__REG32 di1_pt_6_2                      : 2;
__REG32 di1_cst_2                       : 2;
__REG32 componnent_size_2               : 8;
__REG32 di1_access_size_2               : 8;
};
/* IPU_DI1_DW_GEN_2_SER */
struct {
__REG32 di1_serial_clk_2                : 2;
__REG32 di1_serial_rs_2                 : 2;
__REG32 di1_serial_valid_bits_2         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_2                      : 2;
__REG32 di1_start_period_2              : 8;
__REG32 di1_serial_period_2             : 8;
};
} __ipu_di1_dw_gen_2_bits;

/* DI1 Data Wave Gen 3 Register (IPU_DI1_DW_GEN_3) */
typedef union{
/* IPU_DI1_DW_GEN_3 */
struct {
__REG32 di1_pt_0_3                      : 2;
__REG32 di1_pt_1_3                      : 2;
__REG32 di1_pt_2_3                      : 2;
__REG32 di1_pt_3_3                      : 2;
__REG32 di1_pt_4_3                      : 2;
__REG32 di1_pt_5_3                      : 2;
__REG32 di1_pt_6_3                      : 2;
__REG32 di1_cst_3                       : 2;
__REG32 componnent_size_3               : 8;
__REG32 di1_access_size_3               : 8;
};
/* IPU_DI1_DW_GEN_3_SER */
struct {
__REG32 di1_serial_clk_3                : 2;
__REG32 di1_serial_rs_3                 : 2;
__REG32 di1_serial_valid_bits_3         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_3                      : 2;
__REG32 di1_start_period_3              : 8;
__REG32 di1_serial_period_3             : 8;
};
} __ipu_di1_dw_gen_3_bits;

/* DI1 Data Wave Gen 4 Register (IPU_DI1_DW_GEN_4) */
typedef union{
/* IPU_DI1_DW_GEN_4 */
struct {
__REG32 di1_pt_0_4                      : 2;
__REG32 di1_pt_1_4                      : 2;
__REG32 di1_pt_2_4                      : 2;
__REG32 di1_pt_3_4                      : 2;
__REG32 di1_pt_4_4                      : 2;
__REG32 di1_pt_5_4                      : 2;
__REG32 di1_pt_6_4                      : 2;
__REG32 di1_cst_4                       : 2;
__REG32 componnent_size_4               : 8;
__REG32 di1_access_size_4               : 8;
};
/* IPU_DI1_DW_GEN_4_SER */
struct {
__REG32 di1_serial_clk_4                : 2;
__REG32 di1_serial_rs_4                 : 2;
__REG32 di1_serial_valid_bits_4         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_4                      : 2;
__REG32 di1_start_period_4              : 8;
__REG32 di1_serial_period_4             : 8;
};
} __ipu_di1_dw_gen_4_bits;

/* DI1 Data Wave Gen 5 Register (IPU_DI1_DW_GEN_5) */
typedef union{
/* IPU_DI1_DW_GEN_5 */
struct {
__REG32 di1_pt_0_5                      : 2;
__REG32 di1_pt_1_5                      : 2;
__REG32 di1_pt_2_5                      : 2;
__REG32 di1_pt_3_5                      : 2;
__REG32 di1_pt_4_5                      : 2;
__REG32 di1_pt_5_5                      : 2;
__REG32 di1_pt_6_5                      : 2;
__REG32 di1_cst_5                       : 2;
__REG32 componnent_size_5               : 8;
__REG32 di1_access_size_5               : 8;
};
/* IPU_DI1_DW_GEN_5_SER */
struct {
__REG32 di1_serial_clk_5                : 2;
__REG32 di1_serial_rs_5                 : 2;
__REG32 di1_serial_valid_bits_5         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_5                      : 2;
__REG32 di1_start_period_5              : 8;
__REG32 di1_serial_period_5             : 8;
};
} __ipu_di1_dw_gen_5_bits;

/* DI1 Data Wave Gen 6 Register (IPU_DI1_DW_GEN_6) */
typedef union{
/* IPU_DI1_DW_GEN_6 */
struct {
__REG32 di1_pt_0_6                      : 2;
__REG32 di1_pt_1_6                      : 2;
__REG32 di1_pt_2_6                      : 2;
__REG32 di1_pt_3_6                      : 2;
__REG32 di1_pt_4_6                      : 2;
__REG32 di1_pt_5_6                      : 2;
__REG32 di1_pt_6_6                      : 2;
__REG32 di1_cst_6                       : 2;
__REG32 componnent_size_6               : 8;
__REG32 di1_access_size_6               : 8;
};
/* IPU_DI1_DW_GEN_6_SER */
struct {
__REG32 di1_serial_clk_6                : 2;
__REG32 di1_serial_rs_6                 : 2;
__REG32 di1_serial_valid_bits_6         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_6                      : 2;
__REG32 di1_start_period_6              : 8;
__REG32 di1_serial_period_6             : 8;
};
} __ipu_di1_dw_gen_6_bits;

/* DI1 Data Wave Gen 7 Register (IPU_DI1_DW_GEN_7) */
typedef union{
/* IPU_DI1_DW_GEN_7 */
struct {
__REG32 di1_pt_0_7                      : 2;
__REG32 di1_pt_1_7                      : 2;
__REG32 di1_pt_2_7                      : 2;
__REG32 di1_pt_3_7                      : 2;
__REG32 di1_pt_4_7                      : 2;
__REG32 di1_pt_5_7                      : 2;
__REG32 di1_pt_6_7                      : 2;
__REG32 di1_cst_7                       : 2;
__REG32 componnent_size_7               : 8;
__REG32 di1_access_size_7               : 8;
};
/* IPU_DI1_DW_GEN_7_SER */
struct {
__REG32 di1_serial_clk_7                : 2;
__REG32 di1_serial_rs_7                 : 2;
__REG32 di1_serial_valid_bits_7         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_7                      : 2;
__REG32 di1_start_period_7              : 8;
__REG32 di1_serial_period_7             : 8;
};
} __ipu_di1_dw_gen_7_bits;

/* DI1 Data Wave Gen 8 Register (IPU_DI1_DW_GEN_8) */
typedef union{
/* IPU_DI1_DW_GEN_8 */
struct {
__REG32 di1_pt_0_8                      : 2;
__REG32 di1_pt_1_8                      : 2;
__REG32 di1_pt_2_8                      : 2;
__REG32 di1_pt_3_8                      : 2;
__REG32 di1_pt_4_8                      : 2;
__REG32 di1_pt_5_8                      : 2;
__REG32 di1_pt_6_8                      : 2;
__REG32 di1_cst_8                       : 2;
__REG32 componnent_size_8               : 8;
__REG32 di1_access_size_8               : 8;
};
/* IPU_DI1_DW_GEN_8_SER */
struct {
__REG32 di1_serial_clk_8                : 2;
__REG32 di1_serial_rs_8                 : 2;
__REG32 di1_serial_valid_bits_8         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_8                      : 2;
__REG32 di1_start_period_8              : 8;
__REG32 di1_serial_period_8             : 8;
};
} __ipu_di1_dw_gen_8_bits;

/* DI1 Data Wave Gen 9 Register (IPU_DI1_DW_GEN_9) */
typedef union{
/* IPU_DI1_DW_GEN_9 */
struct {
__REG32 di1_pt_0_9                      : 2;
__REG32 di1_pt_1_9                      : 2;
__REG32 di1_pt_2_9                      : 2;
__REG32 di1_pt_3_9                      : 2;
__REG32 di1_pt_4_9                      : 2;
__REG32 di1_pt_5_9                      : 2;
__REG32 di1_pt_6_9                      : 2;
__REG32 di1_cst_9                       : 2;
__REG32 componnent_size_9               : 8;
__REG32 di1_access_size_9               : 8;
};
/* IPU_DI1_DW_GEN_9_SER */
struct {
__REG32 di1_serial_clk_9                : 2;
__REG32 di1_serial_rs_9                 : 2;
__REG32 di1_serial_valid_bits_9         : 5;
__REG32                                 : 5;
__REG32 _di1_cst_9                      : 2;
__REG32 di1_start_period_9              : 8;
__REG32 di1_serial_period_9             : 8;
};
} __ipu_di1_dw_gen_9_bits;

/* DI1 Data Wave Gen 10 Register (IPU_DI1_DW_GEN_10) */
typedef union{
/* IPU_DI1_DW_GEN_10 */
struct {
__REG32 di1_pt_0_10                     : 2;
__REG32 di1_pt_1_10                     : 2;
__REG32 di1_pt_2_10                     : 2;
__REG32 di1_pt_3_10                     : 2;
__REG32 di1_pt_4_10                     : 2;
__REG32 di1_pt_5_10                     : 2;
__REG32 di1_pt_6_10                     : 2;
__REG32 di1_cst_10                      : 2;
__REG32 componnent_size_10              : 8;
__REG32 di1_access_size_10              : 8;
};
/* IPU_DI1_DW_GEN_10_SER */
struct {
__REG32 di1_serial_clk_10               : 2;
__REG32 di1_serial_rs_10                : 2;
__REG32 di1_serial_valid_bits_10        : 5;
__REG32                                 : 5;
__REG32 _di1_cst_10                     : 2;
__REG32 di1_start_period_10             : 8;
__REG32 di1_serial_period_10            : 8;
};
} __ipu_di1_dw_gen_10_bits;

/* DI1 Data Wave Gen 11 Register (IPU_DI1_DW_GEN_11) */
typedef union{
/* IPU_DI1_DW_GEN_11 */
struct {
__REG32 di1_pt_0_11                     : 2;
__REG32 di1_pt_1_11                     : 2;
__REG32 di1_pt_2_11                     : 2;
__REG32 di1_pt_3_11                     : 2;
__REG32 di1_pt_4_11                     : 2;
__REG32 di1_pt_5_11                     : 2;
__REG32 di1_pt_6_11                     : 2;
__REG32 di1_cst_11                      : 2;
__REG32 componnent_size_11              : 8;
__REG32 di1_access_size_11              : 8;
};
/* IPU_DI1_DW_GEN_11_SER */
struct {
__REG32 di1_serial_clk_11               : 2;
__REG32 di1_serial_rs_11                : 2;
__REG32 di1_serial_valid_bits_11        : 5;
__REG32                                 : 5;
__REG32 _di1_cst_11                     : 2;
__REG32 di1_start_period_11             : 8;
__REG32 di1_serial_period_11            : 8;
};
} __ipu_di1_dw_gen_11_bits;

/* DI1 Data Wave Set 0 0 Register (IPU_DI1_DW_SET0_0) */
typedef struct{
__REG32 di1_data_cnt_up0_0              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_0            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_0_bits;

/* DI1 Data Wave Set 0 1 Register (IPU_DI1_DW_SET0_1) */
typedef struct{
__REG32 di1_data_cnt_up0_1              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_1            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_1_bits;

/* DI1 Data Wave Set 0 2 Register (IPU_DI1_DW_SET0_2) */
typedef struct{
__REG32 di1_data_cnt_up0_2              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_2            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_2_bits;

/* DI1 Data Wave Set 0 3 Register (IPU_DI1_DW_SET0_3) */
typedef struct{
__REG32 di1_data_cnt_up0_3              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_3            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_3_bits;

/* DI1 Data Wave Set 0 4 Register (IPU_DI1_DW_SET0_4) */
typedef struct{
__REG32 di1_data_cnt_up0_4              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_4            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_4_bits;

/* DI1 Data Wave Set 0 5 Register (IPU_DI1_DW_SET0_5) */
typedef struct{
__REG32 di1_data_cnt_up0_5              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_5            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_5_bits;

/* DI1 Data Wave Set 0 6 Register (IPU_DI1_DW_SET0_6) */
typedef struct{
__REG32 di1_data_cnt_up0_6              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_6            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_6_bits;

/* DI1 Data Wave Set 0 7 Register (IPU_DI1_DW_SET0_7) */
typedef struct{
__REG32 di1_data_cnt_up0_7              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_7            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_7_bits;

/* DI1 Data Wave Set 0 8 Register (IPU_DI1_DW_SET0_8) */
typedef struct{
__REG32 di1_data_cnt_up0_8              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_8            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_8_bits;

/* DI1 Data Wave Set 0 9 Register (IPU_DI1_DW_SET0_9) */
typedef struct{
__REG32 di1_data_cnt_up0_9              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_9            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_9_bits;

/* DI1 Data Wave Set 0 10 Register (IPU_DI1_DW_SET0_10) */
typedef struct{
__REG32 di1_data_cnt_up0_10             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_10           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_10_bits;

/* DI1 Data Wave Set 0 11 Register (IPU_DI1_DW_SET0_11) */
typedef struct{
__REG32 di1_data_cnt_up0_11             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down0_11           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set0_11_bits;

/* DI1 Data Wave Set 1 0 Register (IPU_DI1_DW_SET1_0) */
typedef struct{
__REG32 di1_data_cnt_up1_0              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_0            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_0_bits;

/* DI1 Data Wave Set 1 1 Register (IPU_DI1_DW_SET1_1) */
typedef struct{
__REG32 di1_data_cnt_up1_1              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_1            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_1_bits;

/* DI1 Data Wave Set 1 2 Register (IPU_DI1_DW_SET1_2) */
typedef struct{
__REG32 di1_data_cnt_up1_2              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_2            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_2_bits;

/* DI1 Data Wave Set 1 3 Register (IPU_DI1_DW_SET1_3) */
typedef struct{
__REG32 di1_data_cnt_up1_3              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_3            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_3_bits;

/* DI1 Data Wave Set 1 4 Register (IPU_DI1_DW_SET1_4) */
typedef struct{
__REG32 di1_data_cnt_up1_4              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_4            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_4_bits;

/* DI1 Data Wave Set 1 5 Register (IPU_DI1_DW_SET1_5) */
typedef struct{
__REG32 di1_data_cnt_up1_5              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_5            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_5_bits;

/* DI1 Data Wave Set 1 6 Register (IPU_DI1_DW_SET1_6) */
typedef struct{
__REG32 di1_data_cnt_up1_6              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_6            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_6_bits;

/* DI1 Data Wave Set 1 7 Register (IPU_DI1_DW_SET1_7) */
typedef struct{
__REG32 di1_data_cnt_up1_7              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_7            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_7_bits;

/* DI1 Data Wave Set 1 8 Register (IPU_DI1_DW_SET1_8) */
typedef struct{
__REG32 di1_data_cnt_up1_8              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_8            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_8_bits;

/* DI1 Data Wave Set 1 9 Register (IPU_DI1_DW_SET1_9) */
typedef struct{
__REG32 di1_data_cnt_up1_9              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_9            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_9_bits;

/* DI1 Data Wave Set 1 10 Register (IPU_DI1_DW_SET1_10) */
typedef struct{
__REG32 di1_data_cnt_up1_10             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_10           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_10_bits;

/* DI1 Data Wave Set 1 11 Register (IPU_DI1_DW_SET1_11) */
typedef struct{
__REG32 di1_data_cnt_up1_11             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down1_11           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set1_11_bits;

/* DI1 Data Wave Set 2 0 Register (IPU_DI1_DW_SET2_0) */
typedef struct{
__REG32 di1_data_cnt_up2_0              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_0            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_0_bits;

/* DI1 Data Wave Set 2 1 Register (IPU_DI1_DW_SET2_1) */
typedef struct{
__REG32 di1_data_cnt_up2_1              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_1            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_1_bits;

/* DI1 Data Wave Set 2 2 Register (IPU_DI1_DW_SET2_2) */
typedef struct{
__REG32 di1_data_cnt_up2_2              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_2            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_2_bits;

/* DI1 Data Wave Set 2 3 Register (IPU_DI1_DW_SET2_3) */
typedef struct{
__REG32 di1_data_cnt_up2_3              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_3            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_3_bits;

/* DI1 Data Wave Set 2 4 Register (IPU_DI1_DW_SET2_4) */
typedef struct{
__REG32 di1_data_cnt_up2_4              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_4            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_4_bits;

/* DI1 Data Wave Set 2 5 Register (IPU_DI1_DW_SET2_5) */
typedef struct{
__REG32 di1_data_cnt_up2_5              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_5            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_5_bits;

/* DI1 Data Wave Set 2 6 Register (IPU_DI1_DW_SET2_6) */
typedef struct{
__REG32 di1_data_cnt_up2_6              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_6            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_6_bits;

/* DI1 Data Wave Set 2 7 Register (IPU_DI1_DW_SET2_7) */
typedef struct{
__REG32 di1_data_cnt_up2_7              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_7            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_7_bits;

/* DI1 Data Wave Set 2 8 Register (IPU_DI1_DW_SET2_8) */
typedef struct{
__REG32 di1_data_cnt_up2_8              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_8            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_8_bits;

/* DI1 Data Wave Set 2 9 Register (IPU_DI1_DW_SET2_9) */
typedef struct{
__REG32 di1_data_cnt_up2_9              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_9            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_9_bits;

/* DI1 Data Wave Set 2 10 Register (IPU_DI1_DW_SET2_10) */
typedef struct{
__REG32 di1_data_cnt_up2_10             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_10           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_10_bits;

/* DI1 Data Wave Set 2 11 Register (IPU_DI1_DW_SET2_11) */
typedef struct{
__REG32 di1_data_cnt_up2_11             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down2_11           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set2_11_bits;

/* DI1 Data Wave Set 3 0 Register (IPU_DI1_DW_SET3_0) */
typedef struct{
__REG32 di1_data_cnt_up3_0              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_0            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_0_bits;

/* DI1 Data Wave Set 3 1 Register (IPU_DI1_DW_SET3_1) */
typedef struct{
__REG32 di1_data_cnt_up3_1              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_1            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_1_bits;

/* DI1 Data Wave Set 3 2 Register (IPU_DI1_DW_SET3_2) */
typedef struct{
__REG32 di1_data_cnt_up3_2              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_2            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_2_bits;

/* DI1 Data Wave Set 3 3 Register (IPU_DI1_DW_SET3_3) */
typedef struct{
__REG32 di1_data_cnt_up3_3              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_3            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_3_bits;

/* DI1 Data Wave Set 3 4 Register (IPU_DI1_DW_SET3_4) */
typedef struct{
__REG32 di1_data_cnt_up3_4              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_4            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_4_bits;

/* DI1 Data Wave Set 3 5 Register (IPU_DI1_DW_SET3_5) */
typedef struct{
__REG32 di1_data_cnt_up3_5              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_5            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_5_bits;

/* DI1 Data Wave Set 3 6 Register (IPU_DI1_DW_SET3_6) */
typedef struct{
__REG32 di1_data_cnt_up3_6              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_6            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_6_bits;

/* DI1 Data Wave Set 3 7 Register (IPU_DI1_DW_SET3_7) */
typedef struct{
__REG32 di1_data_cnt_up3_7              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_7            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_7_bits;

/* DI1 Data Wave Set 3 8 Register (IPU_DI1_DW_SET3_8) */
typedef struct{
__REG32 di1_data_cnt_up3_8              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_8            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_8_bits;

/* DI1 Data Wave Set 3 9 Register (IPU_DI1_DW_SET3_9) */
typedef struct{
__REG32 di1_data_cnt_up3_9              : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_9            : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_9_bits;

/* DI1 Data Wave Set 3 10 Register (IPU_DI1_DW_SET3_10) */
typedef struct{
__REG32 di1_data_cnt_up3_10             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_10           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_10_bits;

/* DI1 Data Wave Set 3 11 Register (IPU_DI1_DW_SET3_11) */
typedef struct{
__REG32 di1_data_cnt_up3_11             : 9;
__REG32                                 : 7;
__REG32 di1_data_cnt_down3_11           : 9;
__REG32                                 : 7;
} __ipu_di1_dw_set3_11_bits;

/* DI1 Step Repeat 1 Registers (IPU_D1_STP_REP_1) */
typedef struct{
__REG32 di1_step_repeat_1               :12;
__REG32                                 : 4;
__REG32 di1_step_repeat_2               :12;
__REG32                                 : 4;
} __ipu_di1_stp_rep_1_bits;

/* DI1 Step Repeat 2 Registers (IPU_D1_STP_REP_2) */
typedef struct{
__REG32 di1_step_repeat_3               :12;
__REG32                                 : 4;
__REG32 di1_step_repeat_4               :12;
__REG32                                 : 4;
} __ipu_di1_stp_rep_2_bits;

/* DI1 Step Repeat 3 Registers (IPU_D1_STP_REP_3) */
typedef struct{
__REG32 di1_step_repeat_5               :12;
__REG32                                 : 4;
__REG32 di1_step_repeat_6               :12;
__REG32                                 : 4;
} __ipu_di1_stp_rep_3_bits;

/* DI1 Step Repeat 4 Registers (IPU_D1_STP_REP_4) */
typedef struct{
__REG32 di1_step_repeat_7               :12;
__REG32                                 : 4;
__REG32 di1_step_repeat_8               :12;
__REG32                                 : 4;
} __ipu_di1_stp_rep_4_bits;

/* DI1 Step Repeat 9 Registers (IPU_DI1_STP_REP_9) */
typedef struct{
__REG32 di1_step_repeat_9               :12;
__REG32                                 :20;
} __ipu_di1_stp_rep_9_bits;

/* DI1 Serial Display Control Register (IPU_DI1_SER_CONF) */
typedef struct{
__REG32 DI1_WAIT4SERIAL                 : 1;
__REG32 DI1_SERIAL_CS_POLARITY          : 1;
__REG32 DI1_SERIAL_RS_POLARITY          : 1;
__REG32 DI1_SERIAL_DATA_POLARITY        : 1;
__REG32 DI1_SER_CLK_POLARITY            : 1;
__REG32 DI1_LLA_SER_ACCESS              : 1;
__REG32                                 : 2;
__REG32 DI1_SERIAL_LATCH                : 8;
__REG32 DI1_SERIAL_LLA_PNTR_RS_W_0      : 4;
__REG32 DI1_SERIAL_LLA_PNTR_RS_W_1      : 4;
__REG32 DI1_SERIAL_LLA_PNTR_RS_R_0      : 4;
__REG32 DI1_SERIAL_LLA_PNTR_RS_R_1      : 4;
} __ipu_di1_ser_conf_bits;

/* DI1 Special Signals Control Register (IPU_DI1_SSC) */
typedef struct{
__REG32 DI1_BYTE_EN_PNTR                : 3;
__REG32 DI1_BYTE_EN_RD_IN               : 1;
__REG32 DI1_BYTE_EN_POLARITY            : 1;
__REG32 DI1_WAIT_ON                     : 1;
__REG32                                 :10;
__REG32 DI1_CS_ERM                      : 1;
__REG32 DI1_PIN11_ERM                   : 1;
__REG32 DI1_PIN12_ERM                   : 1;
__REG32 DI1_PIN13_ERM                   : 1;
__REG32 DI1_PIN14_ERM                   : 1;
__REG32 DI1_PIN15_ERM                   : 1;
__REG32 DI1_PIN16_ERM                   : 1;
__REG32 DI1_PIN17_ERM                   : 1;
__REG32                                 : 8;
} __ipu_di1_ssc_bits;

/* DI1 Polarity Register (IPU_DI1_POL) */
typedef struct{
__REG32 di1_drdy_polarity_11            : 1;
__REG32 di1_drdy_polarity_12            : 1;
__REG32 di1_drdy_polarity_13            : 1;
__REG32 di1_drdy_polarity_14            : 1;
__REG32 di1_drdy_polarity_15            : 1;
__REG32 di1_drdy_polarity_16            : 1;
__REG32 di1_drdy_polarity_17            : 1;
__REG32 DI1_DRDY_DATA_POLARITY          : 1;
__REG32 di1_cs0_polarity_11             : 1;
__REG32 di1_cs0_polarity_12             : 1;
__REG32 di1_cs0_polarity_13             : 1;
__REG32 di1_cs0_polarity_14             : 1;
__REG32 di1_cs0_polarity_15             : 1;
__REG32 di1_cs0_polarity_16             : 1;
__REG32 di1_cs0_polarity_17             : 1;
__REG32 DI1_CS0_DATA_POLARITY           : 1;
__REG32 di1_cs1_polarity_11             : 1;
__REG32 di1_cs1_polarity_12             : 1;
__REG32 di1_cs1_polarity_13             : 1;
__REG32 di1_cs1_polarity_14             : 1;
__REG32 di1_cs1_polarity_15             : 1;
__REG32 di1_cs1_polarity_16             : 1;
__REG32 di1_cs1_polarity_17             : 1;
__REG32 DI1_CS1_DATA_POLARITY           : 1;
__REG32 DI1_CS0_BYTE_EN_POLARITY        : 1;
__REG32 DI1_CS1_BYTE_EN_POLARITY        : 1;
__REG32 DI1_WAIT_POLARITY               : 1;
__REG32                                 : 5;
} __ipu_di1_pol_bits;

/* DI1Active Window 0 Register (IPU_DI1_AW0) */
typedef struct{
__REG32 DI1_AW_HSTART                   :12;
__REG32 DI1_AW_HCOUNT_SEL               : 4;
__REG32 DI1_AW_HEND                     :12;
__REG32 DI1_AW_TRIG_SEL                 : 4;
} __ipu_di1_aw0_bits;

/* DI1 Active Window 1 Register (IPU_DI1_AW1) */
typedef struct{
__REG32 DI1_AW_VSTART                   :12;
__REG32 DI1_AW_VCOUNT_SEL               : 4;
__REG32 DI1_AW_VEND                     :12;
__REG32                                 : 4;
} __ipu_di1_aw1_bits;

/* DI1 Screen Configuration Register (IPU_DI1_SCR_CONF) */
typedef struct{
__REG32 DI1_SCREEN_HEIGHT               :12;
__REG32                                 :20;
} __ipu_di1_scr_conf_bits;

/* DI1 Status Register (IPU_DI1_STAT) */
typedef struct{
__REG32 DI1_READ_FIFO_EMPTY             : 1;
__REG32 DI1_READ_FIFO_FULL              : 1;
__REG32 DI1_CNTR_FIFO_EMPTY             : 1;
__REG32 DI1_CNTR_FIFO_FULL              : 1;
__REG32                                 :28;
} __ipu_di1_stat_bits;

/* SMFC Mapping Register (IPU_SMFC_MAP) */
typedef struct{
__REG32 MAP_CH0                         : 3;
__REG32 MAP_CH1                         : 3;
__REG32 MAP_CH2                         : 3;
__REG32 MAP_CH3                         : 3;
__REG32                                 :20;
} __ipu_smfc_map_bits;

/* SMFC Watermark Control Register (IPU_SMFC_WMC) */
typedef struct{
__REG32 WM0_SET                         : 3;
__REG32 WM0_CLR                         : 3;
__REG32 WM1_SET                         : 3;
__REG32 WM1_CLR                         : 3;
__REG32                                 : 4;
__REG32 WM2_SET                         : 3;
__REG32 WM2_CLR                         : 3;
__REG32 WM3_SET                         : 3;
__REG32 WM3_CLR                         : 3;
__REG32                                 : 4;
} __ipu_smfc_wmc_bits;

/* SMFC Burst Size Register (IPU_SMFC_BS) */
typedef struct{
__REG32 BURST0_SIZE                     : 4;
__REG32 BURST1_SIZE                     : 4;
__REG32 BURST2_SIZE                     : 4;
__REG32 BURST3_SIZE                     : 4;
__REG32                                 :16;
} __ipu_smfc_bs_bits;

/* DC Read Channel Configuration Register (IPU_DC_READ_CH_CONF) */
typedef struct{
__REG32 RD_CHANNEL_EN                   : 1;
__REG32 PROG_DI_ID_0                    : 1;
__REG32 PROG_DISP_ID_0                  : 2;
__REG32 W_SIZE_0                        : 2;
__REG32 CHAN_MASK_DEFAULT_0             : 1;
__REG32                                 : 1;
__REG32 ID_0                            : 1;
__REG32 ID_1                            : 1;
__REG32 ID_2                            : 1;
__REG32 ID_3                            : 1;
__REG32                                 : 4;
__REG32 TIME_OUT_VALUE                  :16;
} __ipu_dc_read_ch_conf_bits;

/* DC Read Channel Start Address Register (IPU_DC_READ_SH_ADDR) */
typedef struct{
__REG32 ST_ADDR_0                       :29;
__REG32                                 : 3;
} __ipu_dc_read_sh_addr_bits;

/* DC Routine Link Register 0 Channel 0 (IPU_DC_RL0_CH_0) */
typedef struct{
__REG32 COD_NF_PRIORITY_CHAN_0          : 4;
__REG32                                 : 4;
__REG32 COD_NF_START_CHAN_0             : 8;
__REG32 COD_NL_PRIORITY_CHAN_0          : 4;
__REG32                                 : 4;
__REG32 COD_NL_START_CHAN_0             : 8;
} __ipu_dc_rl0_ch_0_bits;

/* DC Routine Link Register 1 Channel 0 (IPU_DC_RL1_CH_0) */
typedef struct{
__REG32 COD_EOF_PRIORITY_CHAN_0         : 4;
__REG32                                 : 4;
__REG32 COD_EOF_START_CHAN_0            : 8;
__REG32 COD_NFIELD_PRIORITY_CHAN_0      : 4;
__REG32                                 : 4;
__REG32 COD_NFIELD_START_CHAN_0         : 8;
} __ipu_dc_rl1_ch_0_bits;

/* DC Routine Link Registe3 Channel 0 (IPU_DC_RL3_CH_0) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_0    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_0       : 8;
__REG32 COD_NEW_CHAN_PRIORITY_CHAN_0    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_CHAN_START_CHAN_0       : 8;
} __ipu_dc_rl3_ch_0_bits;

/* DC Routine Link Register 4 Channel 0 (IPU_DC_RL4_CH_0) */
typedef struct{
__REG32 COD_NEW_DATA_PRIORITY_CHAN_0    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_DATA_START_CHAN_0       : 8;
__REG32                                 :16;
} __ipu_dc_rl4_ch_0_bits;

/* DC Write Channel 1 Configuration Register (IPU_DC_WR_CH_CONF_1) */
typedef struct{
__REG32 W_SIZE_1                        : 2;
__REG32 PROG_DI_ID_1                    : 1;
__REG32 PROG_DISP_ID_1                  : 2;
__REG32 PROG_CHAN_TYP_1                 : 3;
__REG32 CHAN_MASK_DEFAULT_1             : 1;
__REG32 FIELD_MODE_1                    : 1;
__REG32                                 : 6;
__REG32 PROG_START_TIME_1               :11;
__REG32                                 : 5;
} __ipu_dc_wr_ch_conf_1_bits;

/* DC Routine Link Register2 Channel 0 (IPU_DC_RL2_CH_0) */
typedef union {
  /* IPU_DC_RL2_CH_0 */
struct{
__REG32 COD_EOL_PRIORITY_CHAN_0         : 4;
__REG32                                 : 4;
__REG32 COD_EOL_START_CHAN_0            : 8;
__REG32 COD_EOFIELD_PRIORITY_CHAN_0     : 4;
__REG32                                 : 4;
__REG32 COD_EOFIELD_START_CHAN_0        : 8;
};
/* IPU_DC_WR_CH_ADDR_1 */
struct{
__REG32 ST_ADDR_1                       :29;
__REG32                                 : 3;
};
} __ipu_dc_rl2_ch_0_bits;

/* DC Routine Link Register 0 Channel 1 (IPU_DC_RL0_CH_1) */
typedef struct{
__REG32 COD_NF_PRIORITY_CHAN_1          : 4;
__REG32                                 : 4;
__REG32 COD_NF_START_CHAN_1             : 8;
__REG32 COD_NL_PRIORITY_CHAN_1          : 4;
__REG32                                 : 4;
__REG32 COD_NL_START_CHAN_1             : 8;
} __ipu_dc_rl0_ch_1_bits;

/* DC Routine Link Register 1 Channel 1 (IPU_DC_RL1_CH_1) */
typedef struct{
__REG32 COD_EOF_PRIORITY_CHAN_1         : 4;
__REG32                                 : 4;
__REG32 COD_EOF_START_CHAN_1            : 8;
__REG32 COD_NFIELD_PRIORITY_CHAN_1      : 4;
__REG32                                 : 4;
__REG32 COD_NFIELD_START_CHAN_1         : 8;
} __ipu_dc_rl1_ch_1_bits;

/* DC Routine Link Register 2 Channel 2 (IPU_DC_RL2_CH_2) */
typedef struct{
__REG32 COD_EOL_PRIORITY_CHAN_2         : 4;
__REG32                                 : 4;
__REG32 COD_EOL_START_CHAN_2            : 8;
__REG32 COD_EOFIELD_PRIORITY_CHAN_2     : 4;
__REG32                                 : 4;
__REG32 COD_EOFIELD_START_CHAN_2        : 8;
} __ipu_dc_rl2_ch_2_bits;

/* DC Routine Link Register 2 Channel 1 (IPU_DC_RL2_CH_1) */
typedef struct{
__REG32 COD_EOL_PRIORITY_CHAN_1         : 4;
__REG32                                 : 4;
__REG32 COD_EOL_START_CHAN_1            : 8;
__REG32 COD_EOFIELD_PRIORITY_CHAN_1     : 4;
__REG32                                 : 4;
__REG32 COD_EOFIELD_START_CHAN_1        : 8;
} __ipu_dc_rl2_ch_1_bits;

/* DC Routine Link Register 4 Channel 1 (IPU_DC_RL4_CH_1) */
typedef struct{
__REG32 COD_NEW_DATA_PRIORITY_CHAN_1    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_DATA_START_CHAN_1       : 8;
__REG32                                 :16;
} __ipu_dc_rl4_ch_1_bits;

/* DC Write Channel 2 Configuration Register (IPU_DC_WR_CH_CONF_2) */
typedef struct{
__REG32 W_SIZE_2                        : 2;
__REG32 PROG_DI_ID_2                    : 1;
__REG32 PROG_DISP_ID_2                  : 2;
__REG32 PROG_CHAN_TYP_2                 : 3;
__REG32 CHAN_MASK_DEFAULT_2             : 1;
__REG32                                 : 7;
__REG32 PROG_START_TIME_2               :11;
__REG32                                 : 5;
} __ipu_dc_wr_ch_conf_2_bits;

/* DC Write Channel 2 Address Configuration Register (IPU_DC_WR_CH_ADDR_2) */
typedef struct{
__REG32 ST_ADDR_2                       :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_addr_2_bits;

/* DC Routine Link Register 0 Channel 2 (IPU_DC_RL0_CH_2) */
typedef struct{
__REG32 COD_NF_PRIORITY_CHAN_2          : 4;
__REG32                                 : 4;
__REG32 COD_NF_START_CHAN_2             : 8;
__REG32 COD_NL_PRIORITY_CHAN_2          : 4;
__REG32                                 : 4;
__REG32 COD_NL_START_CHAN_2             : 8;
} __ipu_dc_rl0_ch_2_bits;

/* DC Routine Link Register 1 Channel 2 (IPU_DC_RL1_CH_2) */
typedef struct{
__REG32 COD_EOF_PRIORITY_CHAN_2         : 4;
__REG32                                 : 4;
__REG32 COD_EOF_START_CHAN_2            : 8;
__REG32 COD_NFIELD_PRIORITY_CHAN_2      : 4;
__REG32                                 : 4;
__REG32 COD_NFIELD_START_CHAN_2         : 8;
} __ipu_dc_rl1_ch_2_bits;

/* DC Routine Link Register 3 Channel 2 (IPU_DC_RL3_CH_2) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_2    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_2       : 8;
__REG32 COD_NEW_CHAN_PRIORITY_CHAN_2    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_CHAN_START_CHAN_2       : 8;
} __ipu_dc_rl3_ch_2_bits;

/* DC Routine Link Register 4 Channel 2 (IPU_DC_RL4_CH_2) */
typedef struct{
__REG32 COD_NEW_DATA_PRIORITY_CHAN_2    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_DATA_START_CHAN_2       : 8;
__REG32                                 :16;
} __ipu_dc_rl4_ch_2_bits;

/* DC Command Channel 3 Configuration Register (IPU_DC_CMD_CH_CONF_3) */
typedef struct{
__REG32 W_SIZE_3                        : 2;
__REG32                                 : 6;
__REG32 COD_CMND_START_CHAN_RS0_3       : 8;
__REG32                                 : 8;
__REG32 COD_CMND_START_CHAN_RS1_3       : 8;
} __ipu_dc_cmd_ch_conf_3_bits;

/* DC Command Channel 4 Configuration Register (IPU_DC_CMD_CH_CONF_4) */
typedef struct{
__REG32 W_SIZE_4                        : 2;
__REG32                                 : 6;
__REG32 COD_CMND_START_CHAN_RS0_4       : 8;
__REG32                                 : 8;
__REG32 COD_CMND_START_CHAN_RS1_4       : 8;
} __ipu_dc_cmd_ch_conf_4_bits;

/* DC Write Channel 5Configuration Register (IPU_DC_WR_CH_CONF_5) */
typedef struct{
__REG32 W_SIZE_5                        : 2;
__REG32 PROG_DI_ID_5                    : 1;
__REG32 PROG_DISP_ID_5                  : 2;
__REG32 PROG_CHAN_TYP_5                 : 3;
__REG32 CHAN_MASK_DEFAULT_5             : 1;
__REG32 FIELD_MODE_5                    : 1;
__REG32                                 : 6;
__REG32 PROG_START_TIME_5               :11;
__REG32                                 : 5;
} __ipu_dc_wr_ch_conf_5_bits;

/* DC Write Channel 5Address Configuration Register (IPU_DC_WR_CH_ADDR_5) */
typedef struct{
__REG32 ST_ADDR_5                       :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_addr_5_bits;

/* DC Routine Link Register 0 Channel 5 (IPU_DC_RL0_CH_5) */
typedef struct{
__REG32 COD_NF_PRIORITY_CHAN_5          : 4;
__REG32                                 : 4;
__REG32 COD_NF_START_CHAN_5             : 8;
__REG32 COD_NL_PRIORITY_CHAN_5          : 4;
__REG32                                 : 4;
__REG32 COD_NL_START_CHAN_5             : 8;
} __ipu_dc_rl0_ch_5_bits;

/* DC Routine Link Register 1 Channel 5 (IPU_DC_RL1_CH_5) */
typedef struct{
__REG32 COD_EOF_PRIORITY_CHAN_5         : 4;
__REG32                                 : 4;
__REG32 COD_EOF_START_CHAN_5            : 8;
__REG32 COD_NFIELD_PRIORITY_CHAN_5      : 4;
__REG32                                 : 4;
__REG32 COD_NFIELD_START_CHAN_5         : 8;
} __ipu_dc_rl1_ch_5_bits;

/* DC Routine Link Register 2 Channel 5 (IPU_DC_RL2_CH_5) */
typedef struct{
__REG32 COD_EOL_PRIORITY_CHAN_5         : 4;
__REG32                                 : 4;
__REG32 COD_EOL_START_CHAN_5            : 8;
__REG32 COD_EOFIELD_PRIORITY_CHAN_5     : 4;
__REG32                                 : 4;
__REG32 COD_EOFIELD_START_CHAN_5        : 8;
} __ipu_dc_rl2_ch_5_bits;

/* DC Routine Link Register3 Channel 5 (IPU_DC_RL3_CH_5) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_5    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_5       : 8;
__REG32 COD_NEW_CHAN_PRIORITY_CHAN_5    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_CHAN_START_CHAN_5       : 8;
} __ipu_dc_rl3_ch_5_bits;

/* DC Routine Link Register 4 Channel 5 (IPU_DC_RL4_CH_5) */
typedef struct{
__REG32 COD_NEW_DATA_PRIORITY_CHAN_5    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_DATA_START_CHAN_5       : 8;
__REG32                                 :16;
} __ipu_dc_rl4_ch_5_bits;

/* DC Write Channel 6 Configuration Register (IPU_DC_WR_CH_CONF_6) */
typedef struct{
__REG32 W_SIZE_6                        : 2;
__REG32 PROG_DI_ID_6                    : 1;
__REG32 PROG_DISP_ID_6                  : 2;
__REG32 PROG_CHAN_TYP_6                 : 3;
__REG32 CHAN_MASK_DEFAULT_6             : 1;
__REG32                                 : 7;
__REG32 PROG_START_TIME_6               :11;
__REG32                                 : 5;
} __ipu_dc_wr_ch_conf_6_bits;

/* DC Write Channel 6 Address Configuration Register (IPU_DC_WR_CH_ADDR_6) */
typedef struct{
__REG32 ST_ADDR_6                       :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_addr_6_bits;

/* DC Routine Link Register 0Channel 6 (IPU_DC_RL0_CH_6) */
typedef struct{
__REG32 COD_NF_PRIORITY_CHAN_6          : 4;
__REG32                                 : 4;
__REG32 COD_NF_START_CHAN_6             : 8;
__REG32 COD_NL_PRIORITY_CHAN_6          : 4;
__REG32                                 : 4;
__REG32 COD_NL_START_CHAN_6             : 8;
} __ipu_dc_rl0_ch_6_bits;

/* DC Routine Link Register 1 Channel 6 (IPU_DC_RL1_CH_6) */
typedef struct{
__REG32 COD_EOF_PRIORITY_CHAN_6         : 4;
__REG32                                 : 4;
__REG32 COD_EOF_START_CHAN_6            : 8;
__REG32 COD_NFIELD_PRIORITY_CHAN_6      : 4;
__REG32                                 : 4;
__REG32 COD_NFIELD_START_CHAN_6         : 8;
} __ipu_dc_rl1_ch_6_bits;

/* DC Routine Link Register 2 Channel 6 (IPU_DC_RL2_CH_6) */
typedef struct{
__REG32 COD_EOL_PRIORITY_CHAN_6         : 4;
__REG32                                 : 4;
__REG32 COD_EOL_START_CHAN_6            : 8;
__REG32 COD_EOFIELD_PRIORITY_CHAN_6     : 4;
__REG32                                 : 4;
__REG32 COD_EOFIELD_START_CHAN_6        : 8;
} __ipu_dc_rl2_ch_6_bits;

/* DC Routine Link Register 3 Channel 6 (IPU_DC_RL3_CH_6) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_6    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_6       : 8;
__REG32 COD_NEW_CHAN_PRIORITY_CHAN_6    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_CHAN_START_CHAN_6       : 8;
} __ipu_dc_rl3_ch_6_bits;

/* DC Routine Link Register 4 Channel 6 (IPU_DC_RL4_CH_6) */
typedef struct{
__REG32 COD_NEW_DATA_PRIORITY_CHAN_6    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_DATA_START_CHAN_6       : 8;
__REG32                                 :16;
} __ipu_dc_rl4_ch_6_bits;

/* DC Write Channel 8 Configuration 1 Register (IPU_DC_WR_CH_CONF1_8) */
typedef struct{
__REG32 W_SIZE_8                        : 2;
__REG32 CHAN_MASK_DEFAULT_8             : 1;
__REG32 MCU_DISP_ID_8                   : 2;
__REG32                                 :27;
} __ipu_dc_wr_ch_conf1_8_bits;

/* DC Write Channel 8 Configuration 2 Register (IPU_DC_WR_CH_CONF2_8) */
typedef struct{
__REG32 NEW_ADDR_SPACE_SA_8             :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_conf2_8_bits;

/* DC Routine Link Register 1-3 Channel 8 (IPU_DC_RL1-3_CH_8) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_8    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_W_8_0   : 8;
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_W_8_1   : 8;
} __ipu_dc_rl1_ch_8_bits;

/* DC Routine Link Register 4-6 Channel 8 (IPU_DC_RL4-6_CH_8) */
typedef struct{
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_R_8_0   : 8;
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_R_8_1   : 8;
} __ipu_dc_rl4_ch_8_bits;

/* DC Write Channel 9 Configuration 1 Register (IPU_DC_WR_CH_CONF1_9) */
typedef struct{
__REG32 W_SIZE_9                        : 2;
__REG32 CHAN_MASK_DEFAULT_9             : 1;
__REG32 MCU_DISP_ID_9                   : 2;
__REG32                                 :27;
} __ipu_dc_wr_ch_conf1_9_bits;

/* DC Write Channel 9 Configuration 2 Register (IPU_DC_WR_CH_CONF2_9) */
typedef struct{
__REG32 NEW_ADDR_SPACE_SA_9             :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_conf2_9_bits;

/* DC Routine Link Register 1-3 Channel 9 (IPU_DC_RL1-3_CH_9) */
typedef struct{
__REG32 COD_NEW_ADDR_PRIORITY_CHAN_9    : 4;
__REG32                                 : 4;
__REG32 COD_NEW_ADDR_START_CHAN_W_9_0   : 8;
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_W_9_1   : 8;
} __ipu_dc_rl1_ch_9_bits;

/* DC Routine Link Register 4-6 Channel 9 (IPU_DC_RL4-6_CH_9) */
typedef struct{
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_R_9_0   : 8;
__REG32                                 : 8;
__REG32 COD_NEW_ADDR_START_CHAN_R_9_1   : 8;
} __ipu_dc_rl4_ch_9_bits;

/* DC General Register (IPU_DC_GEN) */
typedef struct{
__REG32                                 : 1;
__REG32 SYNC_1_6                        : 2;
__REG32                                 : 1;
__REG32 MASK_EN                         : 1;
__REG32 MASK4CHAN_5                     : 1;
__REG32 SYNC_PRIORITY_5                 : 1;
__REG32 SYNC_PRIORITY_1                 : 1;
__REG32 DC_CH5_TYPE                     : 1;
__REG32                                 : 7;
__REG32 DC_BKDIV                        : 8;
__REG32 DC_BK_EN                        : 1;
__REG32                                 : 7;
} __ipu_dc_gen_bits;

/* DC Display Configuration 1 Register 0 (IPU_DC_DISP_CONF1_0) */
typedef struct{
__REG32 DISP_TYP_0                      : 2;
__REG32 ADDR_INCREMENT_0                : 2;
__REG32 ADDR_BE_L_INC_0                 : 2;
__REG32 MCU_ACC_LB_MASK_0               : 1;
__REG32 DISP_RD_VALUE_PTR_0             : 1;
__REG32                                 :24;
} __ipu_dc_disp_conf1_0_bits;

/* DC Display Configuration 1 Register 1 (IPU_DC_DISP_CONF1_1) */
typedef struct{
__REG32 DISP_TYP_1                      : 2;
__REG32 ADDR_INCREMENT_1                : 2;
__REG32 ADDR_BE_L_INC_1                 : 2;
__REG32 MCU_ACC_LB_MASK_1               : 1;
__REG32 DISP_RD_VALUE_PTR_1             : 1;
__REG32                                 :24;
} __ipu_dc_disp_conf1_1_bits;

/* DC Display Configuration 1 Register 2 (IPU_DC_DISP_CONF1_2) */
typedef struct{
__REG32 DISP_TYP_2                      : 2;
__REG32 ADDR_INCREMENT_2                : 2;
__REG32 ADDR_BE_L_INC_2                 : 2;
__REG32 MCU_ACC_LB_MASK_2               : 1;
__REG32 DISP_RD_VALUE_PTR_2             : 1;
__REG32                                 :24;
} __ipu_dc_disp_conf1_2_bits;

/* DC Display Configuration 1 Register 3 (IPU_DC_DISP_CONF1_3) */
typedef struct{
__REG32 DISP_TYP_3                      : 2;
__REG32 ADDR_INCREMENT_3                : 2;
__REG32 ADDR_BE_L_INC_3                 : 2;
__REG32 MCU_ACC_LB_MASK_3               : 1;
__REG32 DISP_RD_VALUE_PTR_3             : 1;
__REG32                                 :24;
} __ipu_dc_disp_conf1_3_bits;

/* DC Display Configuration 2 Register 0 (IPU_DC_DISP_CONF2_0) */
typedef struct{
__REG32 SL_0                            :29;
__REG32                                 : 3;
} __ipu_dc_disp_conf2_0_bits;

/* DC Display Configuration 2 Register 2 (IPU_DC_DISP_CONF2_2) */
typedef struct{
__REG32 SL_2                            :29;
__REG32                                 : 3;
} __ipu_dc_disp_conf2_2_bits;

/* DC Display Configuration 2 Register 3 (IPU_DC_DISP_CONF2_3) */
typedef struct{
__REG32 SL_2                            :29;
__REG32                                 : 3;
} __ipu_dc_disp_conf2_3_bits;

/* DC Mapping Configuration Register 0 (IPU_DC_MAP_CONF_0) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_0            : 5;
__REG32 MAPPING_PNTR_BYTE1_0            : 5;
__REG32 MAPPING_PNTR_BYTE2_0            : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_1            : 5;
__REG32 MAPPING_PNTR_BYTE1_1            : 5;
__REG32 MAPPING_PNTR_BYTE2_1            : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_0_bits;

/* DC Mapping Configuration Register 1 (IPU_DC_MAP_CONF_1) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_2            : 5;
__REG32 MAPPING_PNTR_BYTE1_2            : 5;
__REG32 MAPPING_PNTR_BYTE2_2            : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_3            : 5;
__REG32 MAPPING_PNTR_BYTE1_3            : 5;
__REG32 MAPPING_PNTR_BYTE2_3            : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_1_bits;

/* DC Mapping Configuration Register 2 (IPU_DC_MAP_CONF_2) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_4            : 5;
__REG32 MAPPING_PNTR_BYTE1_4            : 5;
__REG32 MAPPING_PNTR_BYTE2_4            : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_5            : 5;
__REG32 MAPPING_PNTR_BYTE1_5            : 5;
__REG32 MAPPING_PNTR_BYTE2_5            : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_2_bits;

/* DC Mapping Configuration Register 3 (IPU_DC_MAP_CONF_3) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_6            : 5;
__REG32 MAPPING_PNTR_BYTE1_6            : 5;
__REG32 MAPPING_PNTR_BYTE2_6            : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_7            : 5;
__REG32 MAPPING_PNTR_BYTE1_7            : 5;
__REG32 MAPPING_PNTR_BYTE2_7            : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_3_bits;

/* DC Mapping Configuration Register 4 (IPU_DC_MAP_CONF_4) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_8            : 5;
__REG32 MAPPING_PNTR_BYTE1_8            : 5;
__REG32 MAPPING_PNTR_BYTE2_8            : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_9            : 5;
__REG32 MAPPING_PNTR_BYTE1_9            : 5;
__REG32 MAPPING_PNTR_BYTE2_9            : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_4_bits;

/* DC Mapping Configuration Register 5 (IPU_DC_MAP_CONF_5) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_10           : 5;
__REG32 MAPPING_PNTR_BYTE1_10           : 5;
__REG32 MAPPING_PNTR_BYTE2_10           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_11           : 5;
__REG32 MAPPING_PNTR_BYTE1_11           : 5;
__REG32 MAPPING_PNTR_BYTE2_11           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_5_bits;

/* DC Mapping Configuration Register 6 (IPU_DC_MAP_CONF_6) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_12           : 5;
__REG32 MAPPING_PNTR_BYTE1_12           : 5;
__REG32 MAPPING_PNTR_BYTE2_12           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_13           : 5;
__REG32 MAPPING_PNTR_BYTE1_13           : 5;
__REG32 MAPPING_PNTR_BYTE2_13           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_6_bits;

/* DC Mapping Configuration Register 7 (IPU_DC_MAP_CONF_7) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_14           : 5;
__REG32 MAPPING_PNTR_BYTE1_14           : 5;
__REG32 MAPPING_PNTR_BYTE2_14           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_15           : 5;
__REG32 MAPPING_PNTR_BYTE1_15           : 5;
__REG32 MAPPING_PNTR_BYTE2_15           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_7_bits;

/* DC Mapping Configuration Register 8 (IPU_DC_MAP_CONF_8) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_16           : 5;
__REG32 MAPPING_PNTR_BYTE1_16           : 5;
__REG32 MAPPING_PNTR_BYTE2_16           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_17           : 5;
__REG32 MAPPING_PNTR_BYTE1_17           : 5;
__REG32 MAPPING_PNTR_BYTE2_17           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_8_bits;

/* DC Mapping Configuration Register 5 (IPU_DC_MAP_CONF_5) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_18           : 5;
__REG32 MAPPING_PNTR_BYTE1_18           : 5;
__REG32 MAPPING_PNTR_BYTE2_18           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_19           : 5;
__REG32 MAPPING_PNTR_BYTE1_19           : 5;
__REG32 MAPPING_PNTR_BYTE2_19           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_9_bits;

/* DC Mapping Configuration Register 10 (IPU_DC_MAP_CONF_10) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_20           : 5;
__REG32 MAPPING_PNTR_BYTE1_20           : 5;
__REG32 MAPPING_PNTR_BYTE2_20           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_21           : 5;
__REG32 MAPPING_PNTR_BYTE1_21           : 5;
__REG32 MAPPING_PNTR_BYTE2_21           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_10_bits;

/* DC Mapping Configuration Register 11 (IPU_DC_MAP_CONF_11) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_22           : 5;
__REG32 MAPPING_PNTR_BYTE1_22           : 5;
__REG32 MAPPING_PNTR_BYTE2_22           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_23           : 5;
__REG32 MAPPING_PNTR_BYTE1_23           : 5;
__REG32 MAPPING_PNTR_BYTE2_23           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_11_bits;

/* DC Mapping Configuration Register 12 (IPU_DC_MAP_CONF_12) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_24           : 5;
__REG32 MAPPING_PNTR_BYTE1_24           : 5;
__REG32 MAPPING_PNTR_BYTE2_24           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_25           : 5;
__REG32 MAPPING_PNTR_BYTE1_25           : 5;
__REG32 MAPPING_PNTR_BYTE2_25           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_12_bits;

/* DC Mapping Configuration Register 13 (IPU_DC_MAP_CONF_13) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_26           : 5;
__REG32 MAPPING_PNTR_BYTE1_26           : 5;
__REG32 MAPPING_PNTR_BYTE2_26           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_27           : 5;
__REG32 MAPPING_PNTR_BYTE1_27           : 5;
__REG32 MAPPING_PNTR_BYTE2_27           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_13_bits;

/* DC Mapping Configuration Register 14 (IPU_DC_MAP_CONF_14) */
typedef struct{
__REG32 MAPPING_PNTR_BYTE0_28           : 5;
__REG32 MAPPING_PNTR_BYTE1_28           : 5;
__REG32 MAPPING_PNTR_BYTE2_28           : 5;
__REG32                                 : 1;
__REG32 MAPPING_PNTR_BYTE0_29           : 5;
__REG32 MAPPING_PNTR_BYTE1_29           : 5;
__REG32 MAPPING_PNTR_BYTE2_29           : 5;
__REG32                                 : 1;
} __ipu_dc_map_conf_14_bits;

/* DC Mapping Configuration Register 15 (IPU_DC_MAP_CONF_15) */
typedef struct{
__REG32 MD_MASK_0                       : 8;
__REG32 MD_OFFSET_0                     : 5;
__REG32                                 : 3;
__REG32 MD_MASK_1                       : 8;
__REG32 MD_OFFSET_1                     : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_15_bits;

/* DC Mapping Configuration Register 16 (IPU_DC_MAP_CONF_16) */
typedef struct{
__REG32 MD_MASK_2                       : 8;
__REG32 MD_OFFSET_2                     : 5;
__REG32                                 : 3;
__REG32 MD_MASK_3                       : 8;
__REG32 MD_OFFSET_3                     : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_16_bits;

/* DC Mapping Configuration Register 17 (IPU_DC_MAP_CONF_17) */
typedef struct{
__REG32 MD_MASK_4                       : 8;
__REG32 MD_OFFSET_4                     : 5;
__REG32                                 : 3;
__REG32 MD_MASK_5                       : 8;
__REG32 MD_OFFSET_5                     : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_17_bits;

/* DC Mapping Configuration Register 18 (IPU_DC_MAP_CONF_18) */
typedef struct{
__REG32 MD_MASK_6                       : 8;
__REG32 MD_OFFSET_6                     : 5;
__REG32                                 : 3;
__REG32 MD_MASK_7                       : 8;
__REG32 MD_OFFSET_7                     : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_18_bits;

/* DC Mapping Configuration Register 19 (IPU_DC_MAP_CONF_19) */
typedef struct{
__REG32 MD_MASK_8                       : 8;
__REG32 MD_OFFSET_8                     : 5;
__REG32                                 : 3;
__REG32 MD_MASK_9                       : 8;
__REG32 MD_OFFSET_9                     : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_19_bits;

/* DC Mapping Configuration Register 20 (IPU_DC_MAP_CONF_20) */
typedef struct{
__REG32 MD_MASK_10                      : 8;
__REG32 MD_OFFSET_10                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_11                      : 8;
__REG32 MD_OFFSET_11                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_20_bits;

/* DC Mapping Configuration Register 20 (IPU_DC_MAP_CONF_20) */
typedef struct{
__REG32 MD_MASK_12                      : 8;
__REG32 MD_OFFSET_12                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_13                      : 8;
__REG32 MD_OFFSET_13                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_21_bits;

/* DC Mapping Configuration Register 22 (IPU_DC_MAP_CONF_22) */
typedef struct{
__REG32 MD_MASK_14                      : 8;
__REG32 MD_OFFSET_14                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_15                      : 8;
__REG32 MD_OFFSET_15                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_22_bits;

/* DC Mapping Configuration Register 23 (IPU_DC_MAP_CONF_23) */
typedef struct{
__REG32 MD_MASK_16                      : 8;
__REG32 MD_OFFSET_16                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_17                      : 8;
__REG32 MD_OFFSET_17                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_23_bits;

/* DC Mapping Configuration Register 24 (IPU_DC_MAP_CONF_24) */
typedef struct{
__REG32 MD_MASK_18                      : 8;
__REG32 MD_OFFSET_18                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_19                      : 8;
__REG32 MD_OFFSET_19                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_24_bits;

/* DC Mapping Configuration Register 25 (IPU_DC_MAP_CONF_25) */
typedef struct{
__REG32 MD_MASK_20                      : 8;
__REG32 MD_OFFSET_20                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_21                      : 8;
__REG32 MD_OFFSET_21                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_25_bits;

/* DC Mapping Configuration Register 26 (IPU_DC_MAP_CONF_26) */
typedef struct{
__REG32 MD_MASK_22                      : 8;
__REG32 MD_OFFSET_22                    : 5;
__REG32                                 : 3;
__REG32 MD_MASK_23                      : 8;
__REG32 MD_OFFSET_23                    : 5;
__REG32                                 : 3;
} __ipu_dc_map_conf_26_bits;

/* DC User General Data Event 0 Register 0 (IPU_DC_UGDE0_0) */
typedef struct{
__REG32 ID_CODED_0                      : 3;
__REG32 COD_EV_PRIORITY_0               : 4;
__REG32                                 : 1;
__REG32 COD_EV_START_0                  : 8;
__REG32 COD_ODD_START_0                 : 8;
__REG32                                 : 1;
__REG32 ODD_EN_0                        : 1;
__REG32 AUTORESTART_0                   : 1;
__REG32 NF_NL_0                         : 2;
__REG32                                 : 3;
} __ipu_dc_ugde0_0_bits;

/* DC User General Data Event 0 Register 1 (IPU_DC_UGDE0_1) */
typedef struct{
__REG32 STEP_0                          :29;
__REG32                                 : 3;
} __ipu_dc_ugde0_1_bits;

/* DC User General Data Event 0 Register 2 (IPU_DC_UGDE0_2) */
typedef struct{
__REG32 OFFSET_DT_0                     :29;
__REG32                                 : 3;
} __ipu_dc_ugde0_2_bits;

/* DC User General Data Event 0 Register 3 (IPU_DC_UGDE0_3) */
typedef struct{
__REG32 STEP_REPEAT_0                   :29;
__REG32                                 : 3;
} __ipu_dc_ugde0_3_bits;

/* DC User General Data Event 1 Register0 (IPU_DC_UGDE1_0) */
typedef struct{
__REG32 ID_CODED_1                      : 3;
__REG32 COD_EV_PRIORITY_1               : 4;
__REG32                                 : 1;
__REG32 COD_EV_START_1                  : 8;
__REG32 COD_ODD_START_1                 : 8;
__REG32                                 : 1;
__REG32 ODD_EN_1                        : 1;
__REG32 AUTORESTART_1                   : 1;
__REG32 NF_NL_1                         : 2;
__REG32                                 : 3;
} __ipu_dc_ugde1_0_bits;

/* DC User General Data Event 1 Register 1 (IPU_DC_UGDE1_1) */
typedef struct{
__REG32 STEP_1                          :29;
__REG32                                 : 3;
} __ipu_dc_ugde1_1_bits;

/* DC User General Data Event 1 Register 2 (IPU_DC_UGDE1_2) */
typedef struct{
__REG32 OFFSET_DT_1                     :29;
__REG32                                 : 3;
} __ipu_dc_ugde1_2_bits;

/* DC User General Data Event 1 Register 3 (IPU_DC_UGDE1_3) */
typedef struct{
__REG32 STEP_REPEAT_1                   :29;
__REG32                                 : 3;
} __ipu_dc_ugde1_3_bits;

/* DC User General Data Event 2 Register0 (IPU_DC_UGDE2_0) */
typedef struct{
__REG32 ID_CODED_2                      : 3;
__REG32 COD_EV_PRIORITY_2               : 4;
__REG32                                 : 1;
__REG32 COD_EV_START_2                  : 8;
__REG32 COD_ODD_START_2                 : 8;
__REG32                                 : 1;
__REG32 ODD_EN_2                        : 1;
__REG32 AUTORESTART_2                   : 1;
__REG32 NF_NL_2                         : 2;
__REG32                                 : 3;
} __ipu_dc_ugde2_0_bits;

/* DC User General Data Event 2 Register 1 (IPU_DC_UGDE2_1) */
typedef struct{
__REG32 STEP_2                          :29;
__REG32                                 : 3;
} __ipu_dc_ugde2_1_bits;

/* DC User General Data Event 2 Register 2 (IPU_DC_UGDE2_2) */
typedef struct{
__REG32 OFFSET_DT_2                     :29;
__REG32                                 : 3;
} __ipu_dc_ugde2_2_bits;

/* DC User General Data Event 2 Register 3 (IPU_DC_UGDE2_3) */
typedef struct{
__REG32 STEP_REPEAT_2                   :29;
__REG32                                 : 3;
} __ipu_dc_ugde2_3_bits;

/* DC User General Data Event 3 Register0 (IPU_DC_UGDE3_0) */
typedef struct{
__REG32 ID_CODED_3                      : 3;
__REG32 COD_EV_PRIORITY_3               : 4;
__REG32                                 : 1;
__REG32 COD_EV_START_3                  : 8;
__REG32 COD_ODD_START_3                 : 8;
__REG32                                 : 1;
__REG32 ODD_EN_3                        : 1;
__REG32 AUTORESTART_3                   : 1;
__REG32 NF_NL_3                         : 2;
__REG32                                 : 3;
} __ipu_dc_ugde3_0_bits;

/* DC User General Data Event 3 Register 1 (IPU_DC_UGDE3_1) */
typedef struct{
__REG32 STEP_3                          :29;
__REG32                                 : 3;
} __ipu_dc_ugde3_1_bits;

/* DC User General Data Event 3 Register 2 (IPU_DC_UGDE3_2) */
typedef struct{
__REG32 OFFSET_DT_3                     :29;
__REG32                                 : 3;
} __ipu_dc_ugde3_2_bits;

/* DC User General Data Event 3 Register 3 (IPU_DC_UGDE3_3) */
typedef struct{
__REG32 STEP_REPEAT_3                   :29;
__REG32                                 : 3;
} __ipu_dc_ugde3_3_bits;

/* DC Low Level Access Control Register 0 (IPU_DC_LLA0) */
typedef struct{
__REG32 MCU_RS_0_0                      : 8;
__REG32 MCU_RS_1_0                      : 8;
__REG32 MCU_RS_2_0                      : 8;
__REG32 MCU_RS_3_0                      : 8;
} __ipu_dc_lla0_bits;

/* DC Low Level Access Control Register 1 (IPU_DC_LLA1) */
typedef struct{
__REG32 MCU_RS_0_1                      : 8;
__REG32 MCU_RS_1_1                      : 8;
__REG32 MCU_RS_2_1                      : 8;
__REG32 MCU_RS_3_1                      : 8;
} __ipu_dc_lla1_bits;

/* DC Read Low Level Read Access Control Register 0 (IPU_DC_R_LLA0) */
typedef struct{
__REG32 MCU_RS_R_0_0                    : 8;
__REG32 MCU_RS_R_1_0                    : 8;
__REG32 MCU_RS_R_2_0                    : 8;
__REG32 MCU_RS_R_3_0                    : 8;
} __ipu_dc_r_lla0_bits;

/* DC Read Low Level Read Access Control Register 1 (IPU_DC_R_LLA1) */
typedef struct{
__REG32 MCU_RS_R_0_1                    : 8;
__REG32 MCU_RS_R_1_1                    : 8;
__REG32 MCU_RS_R_2_1                    : 8;
__REG32 MCU_RS_R_3_1                    : 8;
} __ipu_dc_r_lla1_bits;

/* DC Write Channel 5 Configuration Register (IPU_DC_WR_CH_ADDR_5_ALT) */
typedef struct{
__REG32 ST_ADDR_5_ALT                   :29;
__REG32                                 : 3;
} __ipu_dc_wr_ch_addr_5_alt_bits;

/* DC Status Register (IPU_DC_STAT) */
typedef struct{
__REG32 DC_TRIPLE_BUF_CNT_FULL_0        : 1;
__REG32 DC_TRIPLE_BUF_CNT_EMPTY_0       : 1;
__REG32 DC_TRIPLE_BUF_DATA_FULL_0       : 1;
__REG32 DC_TRIPLE_BUF_DATA_EMPTY_0      : 1;
__REG32 DC_TRIPLE_BUF_CNT_FULL_1        : 1;
__REG32 DC_TRIPLE_BUF_CNT_EMPTY_1       : 1;
__REG32 DC_TRIPLE_BUF_DATA_FULL_1       : 1;
__REG32 DC_TRIPLE_BUF_DATA_EMPTY_1      : 1;
__REG32                                 :24;
} __ipu_dc_stat_bits;

/* DC Display Configuration 2 Register 1 (IPU_DC_DISP_CONF2_1) */
typedef struct{
__REG32 SL_1                            :29;
__REG32                                 : 3;
} __ipu_dc_disp_conf2_1_bits;

/* DMFC Read Channel Register (IPU_DMFC_RD_CHAN) */
typedef struct{
__REG32                                 : 6;
__REG32 dmfc_burst_size_0               : 2;
__REG32                                 : 9;
__REG32 dmfc_wm_en_0                    : 1;
__REG32 dmfc_wm_set_0                   : 3;
__REG32 dmfc_wm_clr_0                   : 3;
__REG32 dmfc_ppw_c                      : 2;
__REG32                                 : 6;
} __ipu_dmfc_rd_chan_bits;

/* DMFC Write Channel Register (IPU_DMFC_WR_CHAN) */
typedef struct{
__REG32 dmfc_st_addr_1                  : 3;
__REG32 dmfc_fifo_size_1                : 3;
__REG32 dmfc_burst_size_1               : 2;
__REG32 dmfc_st_addr_2                  : 3;
__REG32 dmfc_fifo_size_2                : 3;
__REG32 dmfc_burst_size_2               : 2;
__REG32 dmfc_st_addr_1c                 : 3;
__REG32 dmfc_fifo_size_1c               : 3;
__REG32 dmfc_burst_size_1c              : 2;
__REG32 dmfc_st_addr_2c                 : 3;
__REG32 dmfc_fifo_size_2c               : 3;
__REG32 dmfc_burst_size_2c              : 2;
} __ipu_dmfc_wr_chan_bits;

/* DMFC Write Channel Definition Register (IPU_DMFC_WR_CHAN_DEF) */
typedef struct{
__REG32                                 : 1;
__REG32 dmfc_wm_en_1                    : 1;
__REG32 dmfc_wm_set_1                   : 3;
__REG32 dmfc_wm_clr_1                   : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_2                    : 1;
__REG32 dmfc_wm_set_2                   : 3;
__REG32 dmfc_wm_clr_2                   : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_1c                   : 1;
__REG32 dmfc_wm_set_1c                  : 3;
__REG32 dmfc_wm_clr_1c                  : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_2c                   : 1;
__REG32 dmfc_wm_set_2c                  : 3;
__REG32 dmfc_wm_clr_2c                  : 3;
} __ipu_dmfc_wr_chan_def_bits;

/* DMFC Display Processor Channel Register (IPU_DMFC_DP_CHAN) */
typedef struct{
__REG32 dmfc_st_addr_5b                 : 3;
__REG32 dmfc_fifo_size_5b               : 3;
__REG32 dmfc_burst_size_5b              : 2;
__REG32 dmfc_st_addr_5f                 : 3;
__REG32 dmfc_fifo_size_5f               : 3;
__REG32 dmfc_burst_size_5f              : 2;
__REG32 dmfc_st_addr_6b                 : 3;
__REG32 dmfc_fifo_size_6b               : 3;
__REG32 dmfc_burst_size_6b              : 2;
__REG32 dmfc_st_addr_6f                 : 3;
__REG32 dmfc_fifo_size_6f               : 3;
__REG32 dmfc_burst_size_6f              : 2;
} __ipu_dmfc_dp_chan_bits;

/* DMFC Display Processor Channel Definition Register (IPU_DMFC_DP_CHAN_DEF) */
typedef struct{
__REG32                                 : 1;
__REG32 dmfc_wm_en_5b                   : 1;
__REG32 dmfc_wm_set_5b                  : 3;
__REG32 dmfc_wm_clr_5b                  : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_5f                   : 1;
__REG32 dmfc_wm_set_5f                  : 3;
__REG32 dmfc_wm_clr_5f                  : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_6b                   : 1;
__REG32 dmfc_wm_set_6b                  : 3;
__REG32 dmfc_wm_clr_6b                  : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_6f                   : 1;
__REG32 dmfc_wm_set_6f                  : 3;
__REG32 dmfc_wm_clr_6f                  : 3;
} __ipu_dmfc_dp_chan_def_bits;

/* DMFC General 1 Register (IPU_DMFC_GENERAL_1) */
typedef struct{
__REG32 dmfc_dcdp_sync_pr               : 2;
__REG32                                 : 3;
__REG32 dmfc_burst_size_9               : 2;
__REG32                                 : 2;
__REG32 dmfc_wm_en_9                    : 1;
__REG32 dmfc_wm_set_9                   : 3;
__REG32 dmfc_wm_clr_9                   : 3;
__REG32 WAIT4EOT_1                      : 1;
__REG32 WAIT4EOT_2                      : 1;
__REG32 WAIT4EOT_3                      : 1;
__REG32 WAIT4EOT_4                      : 1;
__REG32 WAIT4EOT_5B                     : 1;
__REG32 WAIT4EOT_5F                     : 1;
__REG32 WAIT4EOT_6B                     : 1;
__REG32 WAIT4EOT_6F                     : 1;
__REG32 WAIT4EOT_9                      : 1;
__REG32                                 : 7;
} __ipu_dmfc_general_1_bits;

/* DMFC General 2 Register (IPU_DMFC_GENERAL_2) */
typedef struct{
__REG32 dmfc_frame_width_rd             :13;
__REG32                                 : 3;
__REG32 dmfc_frame_height_rd            :13;
__REG32                                 : 3;
} __ipu_dmfc_general_2_bits;

/* DMFC IC Interface Control Register (IPU_DMFC_IC_CTRL) */
typedef struct{
__REG32 dmfc_ic_in_port                 : 3;
__REG32                                 : 1;
__REG32 dmfc_ic_ppw_c                   : 2;
__REG32 dmfc_ic_frame_width_rd          :13;
__REG32 dmfc_ic_frame_height_rd         :13;
} __ipu_dmfc_ic_ctrl_bits;

/* DMFC Write Channel Alternate Register (IPU_DMFC_WR_CHAN_ALT) */
typedef struct{
__REG32                                 : 8;
__REG32 dmfc_st_addr_2_alt              : 3;
__REG32 dmfc_fifo_size_2_alt            : 3;
__REG32 dmfc_burst_size_2_alt           : 2;
__REG32                                 :16;
} __ipu_dmfc_wr_chan_alt_bits;

/* DMFC Write Channel Definition Alternate Register (IPU_DMFC_WR_CHAN_DEF_ALT) */
typedef struct{
__REG32                                 : 9;
__REG32 dmfc_wm_en_2_alt                : 1;
__REG32 dmfc_wm_set_2_alt               : 3;
__REG32 dmfc_wm_clr_2_alt               : 3;
__REG32                                 :16;
} __ipu_dmfc_wr_chan_def_alt_bits;

/* DMFC MFC Display Processor Channel Alternate Register (IPU_DMFC_DP_CHAN_ALT) */
typedef struct{
__REG32 dmfc_st_addr_5b_alt             : 3;
__REG32 dmfc_fifo_size_5b_alt           : 3;
__REG32 dmfc_burst_size_5b_alt          : 2;
__REG32                                 : 8;
__REG32 dmfc_st_addr_6b_alt             : 3;
__REG32 dmfc_fifo_size_6b_alt           : 3;
__REG32 dmfc_burst_size_6b_alt          : 2;
__REG32 dmfc_st_addr_6f_alt             : 3;
__REG32 dmfc_fifo_size_6f_alt           : 3;
__REG32 dmfc_burst_size_6f_alt          : 2;
} __ipu_dmfc_dp_chan_alt_bits;

/* DMFC Display Channel Definition Alternate Register (IPU_DMFC_DP_CHAN_DEF_ALT) */
typedef struct{
__REG32                                 : 1;
__REG32 dmfc_wm_en_5b_alt               : 1;
__REG32 dmfc_wm_set_5b_alt              : 3;
__REG32 dmfc_wm_clr_5b_alt              : 3;
__REG32                                 : 9;
__REG32 dmfc_wm_en_6b_alt               : 1;
__REG32 dmfc_wm_set_6b_alt              : 3;
__REG32 dmfc_wm_clr_6b_alt              : 3;
__REG32                                 : 1;
__REG32 dmfc_wm_en_6f_alt               : 1;
__REG32 dmfc_wm_set_6f_alt              : 3;
__REG32 dmfc_wm_clr_6f_alt              : 3;
} __ipu_dmfc_dp_chan_def_alt_bits;

/* DMFC General 1 Alternate Register (IPU_DMFC_GENERAL1_ALT) */
typedef struct{
__REG32                                 :17;
__REG32 WAIT4EOT_2_ALT                  : 1;
__REG32                                 : 2;
__REG32 WAIT4EOT_5B_ALT                 : 1;
__REG32                                 : 1;
__REG32 WAIT4EOT_6B_ALT                 : 1;
__REG32 WAIT4EOT_6F_ALT                 : 1;
__REG32                                 : 8;
} __ipu_dmfc_general1_alt_bits;

/* DMFC Status Register (IPU_DMFC_STAT) */
typedef struct{
__REG32 DMFC_FIFO_FULL_0                : 1;
__REG32 DMFC_FIFO_FULL_1                : 1;
__REG32 DMFC_FIFO_FULL_2                : 1;
__REG32 DMFC_FIFO_FULL_3                : 1;
__REG32 DMFC_FIFO_FULL_4                : 1;
__REG32 DMFC_FIFO_FULL_5                : 1;
__REG32 DMFC_FIFO_FULL_6                : 1;
__REG32 DMFC_FIFO_FULL_7                : 1;
__REG32 DMFC_FIFO_FULL_8                : 1;
__REG32 DMFC_FIFO_FULL_9                : 1;
__REG32 DMFC_FIFO_FULL_10               : 1;
__REG32 DMFC_FIFO_FULL_11               : 1;
__REG32 DMFC_FIFO_EMPTY_0               : 1;
__REG32 DMFC_FIFO_EMPTY_1               : 1;
__REG32 DMFC_FIFO_EMPTY_2               : 1;
__REG32 DMFC_FIFO_EMPTY_3               : 1;
__REG32 DMFC_FIFO_EMPTY_4               : 1;
__REG32 DMFC_FIFO_EMPTY_5               : 1;
__REG32 DMFC_FIFO_EMPTY_6               : 1;
__REG32 DMFC_FIFO_EMPTY_7               : 1;
__REG32 DMFC_FIFO_EMPTY_8               : 1;
__REG32 DMFC_FIFO_EMPTY_9               : 1;
__REG32 DMFC_FIFO_EMPTY_10              : 1;
__REG32 DMFC_FIFO_EMPTY_11              : 1;
__REG32 DMFC_IC_BUFFER_FULL             : 1;
__REG32 DMFC_IC_BUFFER_EMPTY            : 1;
__REG32                                 : 6;
} __ipu_dmfc_stat_bits;

/* VDI Field Size Register (IPU_VDI_FSIZE) */
typedef struct{
__REG32 VDI_FWIDTH                      :11;
__REG32                                 : 5;
__REG32 VDI_FHEIGHT                     :11;
__REG32                                 : 5;
} __ipu_vdi_fsize_bits;

/* VDI Control Register (IPU_VDI_C) */
typedef struct{
__REG32                                 : 1;
__REG32 VDI_CH_422                      : 1;
__REG32 VDI_MOT_SEL                     : 2;
__REG32 VDI_BURST_SIZE1                 : 4;
__REG32 VDI_BURST_SIZE2                 : 4;
__REG32 VDI_BURST_SIZE3                 : 4;
__REG32 VDI_VWM1_SET                    : 3;
__REG32 VDI_VWM1_CLR                    : 3;
__REG32 VDI_VWM3_SET                    : 3;
__REG32 VDI_VWM3_CLR                    : 3;
__REG32                                 : 4;
} __ipu_vdi_c_bits;

/* VDI Control Register 2 (IPU_VDI_C2) */
typedef struct{
__REG32 VDI_CMB_EN                      : 1;
__REG32 VDI_KEY_COLOR_EN                : 1;
__REG32 VDI_GLB_A_EN                    : 1;
__REG32 VDI_PLANE_1_EN                  : 1;
__REG32                                 :28;
} __ipu_vdi_c2_bits;

/* VDI Combining Parameters Register 1 (IPU_VDI_CMDP_1) */
typedef struct{
__REG32 VDI_KEY_COLOR_B                 : 8;
__REG32 VDI_KEY_COLOR_G                 : 8;
__REG32 VDI_KEY_COLOR_R                 : 8;
__REG32 VDI_ALPHA                       : 8;
} __ipu_vdi_cmdp_1_bits;

/* VDI Combining Parameters Register 2 (IPU_VDI_CMDP_2) */
typedef struct{
__REG32 VDI_KEY_COLOR_B                 : 8;
__REG32 VDI_KEY_COLOR_G                 : 8;
__REG32 VDI_KEY_COLOR_R                 : 8;
__REG32                                 : 8;
} __ipu_vdi_cmdp_2_bits;

/* VDI Plane Size Register 1 (IPU_VDI_PS_1) */
typedef struct{
__REG32 VDI_FWIDTH1                     :11;
__REG32                                 : 5;
__REG32 VDI_FHEIGHT1                    :11;
__REG32                                 : 5;
} __ipu_vdi_ps_1_bits;

/* VDI Plane Size Register 2 (IPU_VDI_PS_2) */
typedef struct{
__REG32 VDI_OFFSET_HOR1                 :11;
__REG32                                 : 5;
__REG32 VDI_OFFSET_VER1                 :11;
__REG32                                 : 5;
} __ipu_vdi_ps_2_bits;

/* VDI Plane Size Register 3 (IPU_VDI_PS_3) */
typedef struct{
__REG32 VDI_FWIDTH3                     :11;
__REG32                                 : 5;
__REG32 VDI_FHEIGHT3                    :11;
__REG32                                 : 5;
} __ipu_vdi_ps_3_bits;

/* VDI Plane Size Register 4 (IPU_VDI_PS_4) */
typedef struct{
__REG32 VDI_OFFSET_HOR3                 :11;
__REG32                                 : 5;
__REG32 VDI_OFFSET_VER3                 :11;
__REG32                                 : 5;
} __ipu_vdi_ps_4_bits;

/* DP Common Configuration Sync Flow Register (IPU_DP_COM_CONF_SYNC) */
typedef struct{
__REG32 DP_FG_EN_SYNC                   : 1;
__REG32 DP_GWSEL_SYNC                   : 1;
__REG32 DP_GWAM_SYNC                    : 1;
__REG32 DP_GWCKE_SYNC                   : 1;
__REG32 DP_COC_SYNC                     : 3;
__REG32                                 : 1;
__REG32 DP_CSC_DEF_SYNC                 : 2;
__REG32 DP_CSC_GAMUT_SAT_EN_SYNC        : 1;
__REG32 DP_CSC_YUV_SAT_MODE_SYNC        : 1;
__REG32 DP_GAMMA_EN_SYNC                : 1;
__REG32 DP_GAMMA_YUV_EN_SYNC            : 1;
__REG32                                 :18;
} __ipu_dp_com_conf_sync_bits;

/* DP Graphic Window Control Sync Flow Register (IPU_DP_Graph_Wind_CTRL_SYNC) */
typedef struct{
__REG32 DP_GWCKB_SYNC                   : 8;
__REG32 DP_GWCKG_SYNC                   : 8;
__REG32 DP_GWCKR_SYNC                   : 8;
__REG32 DP_GWAV_SYNC                    : 8;
} __ipu_dp_graph_wind_ctrl_sync_bits;

/* DP Partial Plane Window Position Sync Flow Register (IPU_DP_FG_POS_SYNC) */
typedef struct{
__REG32 DP_FGYP_SYNC                    :11;
__REG32                                 : 5;
__REG32 DP_FGXP_SYNC                    :11;
__REG32                                 : 5;
} __ipu_dp_fg_pos_sync_bits;

/* DP Cursor Position and Size Sync Flow Register (IPU_DP_CUR_POS_SYNC) */
typedef struct{
__REG32 DP_CXW_SYNC                     :11;
__REG32 DP_CXP_SYNC                     : 5;
__REG32 DP_CYH_SYNC                     :11;
__REG32 DP_CYP_SYNC                     : 5;
} __ipu_dp_cur_pos_sync_bits;

/* DP Color Cursor Mapping Sync Flow Register (IPU_DP_CUR_MAP_SYNC) */
typedef struct{
__REG32 DP_CUR_COL_R_SYNC               : 8;
__REG32 DP_CUR_COL_G_SYNC               : 8;
__REG32 DP_CUR_COL_B_SYNC               : 8;
__REG32                                 : 8;
} __ipu_dp_cur_map_sync_bits;

/* DP Gamma Constants Sync Flow Register 0 (IPU_DP_GAMMA_C_SYNC_0) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_0               : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_1               : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_0_bits;

/* DP Gamma Constants Sync Flow Register 1 (IPU_DP_GAMMA_C_SYNC_1) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_2               : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_3               : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_1_bits;

/* DP Gamma Constants Sync Flow Register 2 (IPU_DP_GAMMA_C_SYNC_2) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_4               : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_5               : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_2_bits;

/* DP Gamma Constants Sync Flow Register 3 (IPU_DP_GAMMA_C_SYNC_3) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_6               : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_7               : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_3_bits;

/* DP Gamma Constants Sync Flow Register 4 (IPU_DP_GAMMA_C_SYNC_4) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_8               : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_9               : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_4_bits;

/* DP Gamma Constants Sync Flow Register 5 (IPU_DP_GAMMA_C_SYNC_5) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_10              : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_11              : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_5_bits;

/* DP Gamma Constants Sync Flow Register 6 (IPU_DP_GAMMA_C_SYNC_6) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_12              : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_13              : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_6_bits;

/* DP Gamma Constants Sync Flow Register 7 (IPU_DP_GAMMA_C_SYNC_7) */
typedef struct{
__REG32 DP_GAMMA_C_SYNC_14              : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_SYNC_15              : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_sync_7_bits;

/* DP Gamma Correction Slope Sync Flow Register 0 (IPU_DP_GAMMA_S_SYNC_0) */
typedef struct{
__REG32 DP_GAMMA_S_SYNC_0               : 8;
__REG32 DP_GAMMA_S_SYNC_1               : 8;
__REG32 DP_GAMMA_S_SYNC_2               : 8;
__REG32 DP_GAMMA_S_SYNC_3               : 8;
} __ipu_dp_gamma_s_sync_0_bits;

/* DP Gamma Correction Slope Sync Flow Register 1 (IPU_DP_GAMMA_S_SYNC_1) */
typedef struct{
__REG32 DP_GAMMA_S_SYNC_4               : 8;
__REG32 DP_GAMMA_S_SYNC_5               : 8;
__REG32 DP_GAMMA_S_SYNC_6               : 8;
__REG32 DP_GAMMA_S_SYNC_7               : 8;
} __ipu_dp_gamma_s_sync_1_bits;

/* DP Gamma Correction Slope Sync Flow Register 2 (IPU_DP_GAMMA_S_SYNC_2) */
typedef struct{
__REG32 DP_GAMMA_S_SYNC_8               : 8;
__REG32 DP_GAMMA_S_SYNC_9               : 8;
__REG32 DP_GAMMA_S_SYNC_10              : 8;
__REG32 DP_GAMMA_S_SYNC_11              : 8;
} __ipu_dp_gamma_s_sync_2_bits;

/* DP Gamma Correction Slope Sync Flow Register 3 (IPU_DP_GAMMA_S_SYNC_3) */
typedef struct{
__REG32 DP_GAMMA_S_SYNC_12              : 8;
__REG32 DP_GAMMA_S_SYNC_13              : 8;
__REG32 DP_GAMMA_S_SYNC_14              : 8;
__REG32 DP_GAMMA_S_SYNC_15              : 8;
} __ipu_dp_gamma_s_sync_3_bits;

/* DP Color Space Conversion Control Sync Flow Registers (IPU_DP_CSCA_SYNC_0) */
typedef struct{
__REG32 DP_CSC_A_SYNC_0                 :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_SYNC_1                 :10;
__REG32                                 : 6;
} __ipu_dp_csca_sync_0_bits;

/* DP Color Space Conversion Control Sync Flow Registers (IPU_DP_CSCA_SYNC_1) */
typedef struct{
__REG32 DP_CSC_A_SYNC_2                 :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_SYNC_3                 :10;
__REG32                                 : 6;
} __ipu_dp_csca_sync_1_bits;

/* DP Color Space Conversion Control Sync Flow Registers (IPU_DP_CSCA_SYNC_2) */
typedef struct{
__REG32 DP_CSC_A_SYNC_4                 :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_SYNC_5                 :10;
__REG32                                 : 6;
} __ipu_dp_csca_sync_2_bits;

/* DP Color Space Conversion Control Sync Flow Registers (IPU_DP_CSCA_SYNC_3) */
typedef struct{
__REG32 DP_CSC_A_SYNC_6                 :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_SYNC_7                 :10;
__REG32                                 : 6;
} __ipu_dp_csca_sync_3_bits;

/* DP Color Conversion Control Sync Flow Register 0 (IPU_DP_SCS_SYNC_0) */
typedef struct{
__REG32 DP_CSC_A8_SYNC                  :10;
__REG32                                 : 6;
__REG32 DP_CSC_B0_SYNC                  :14;
__REG32 DP_CSC_S0_SYNC                  : 2;
} __ipu_dp_scs_sync_0_bits;

/* DP Color Conversion Control Sync Flow Register 1 (IPU_DP_SCS_SYNC_1) */
typedef struct{
__REG32 DP_CSC_B1_SYNC                  :14;
__REG32 DP_CSC_S1_SYNC                  : 2;
__REG32 DP_CSC_B2_SYNC                  :14;
__REG32 DP_CSC_S2_SYNC                  : 2;
} __ipu_dp_scs_sync_1_bits;

/* DP Cursor Position and Size Alternate Register (IPU_DP_CUR_POS_ALT) */
typedef struct{
__REG32 DP_CXW_SYNC_ALT                 :11;
__REG32 DP_CXP_SYNC_ALT                 : 5;
__REG32 DP_CYH_SYNC_ALT                 :11;
__REG32 DP_CYP_SYNC_ALT                 : 5;
} __ipu_dp_cur_pos_alt_bits;

/* DP Common Configuration Async 0 Flow Register (IPU_DP_COM_CONF_ASYNC0) */
typedef struct{
__REG32                                 : 1;
__REG32 DP_GWSEL_ASYNC0                 : 1;
__REG32 DP_GWAM_ASYNC0                  : 1;
__REG32 DP_GWCKE_ASYNC0                 : 1;
__REG32 DP_COC_ASYNC0                   : 3;
__REG32                                 : 1;
__REG32 DP_CSC_DEF_ASYNC0               : 2;
__REG32 DP_CSC_GAMUT_SAT_EN_ASYNC0      : 1;
__REG32 DP_CSC_YUV_SAT_MODE_ASYNC0      : 1;
__REG32 DP_GAMMA_EN_ASYNC0              : 1;
__REG32 DP_GAMMA_YUV_EN_ASYNC0          : 1;
__REG32                                 :18;
} __ipu_dp_com_conf_async0_bits;

/* DP Graphic Window Control Async 0 Flow Register (IPU_DP_GRAPH_WIND_CTRL_ASYNC0) */
typedef struct{
__REG32 DP_GWCKB_ASYNC0                 : 8;
__REG32 DP_GWCKG_ASYNC0                 : 8;
__REG32 DP_GWCKR_ASYNC0                 : 8;
__REG32 DP_GWAV_ASYNC0                  : 8;
} __ipu_dp_graph_wind_ctrl_async0_bits;

/* DP Partial Plane Window Position Async 0 Flow Register (IPU_DP_FG_POS_ASYNC0) */
typedef struct{
__REG32 DP_FGYP_ASYNC0                  :11;
__REG32                                 : 5;
__REG32 DP_FGXP_ASYNC0                  :11;
__REG32                                 : 5;
} __ipu_dp_fg_pos_async0_bits;

/* DP Cursor Position and Size Async 0 Flow Register (IPU_DP_CUR_POS_ASYNC0) */
typedef struct{
__REG32 DP_CXW_ASYNC0                   :11;
__REG32 DP_CXP_ASYNC0                   : 5;
__REG32 DP_CYH_ASYNC0                   :11;
__REG32 DP_CYP_ASYNC0                   : 5;
} __ipu_dp_cur_pos_async0_bits;

/* DP Color Cursor Mapping Async 0 Flow Register (IPU_DP_CUR_MAP_ASYNC0) */
typedef struct{
__REG32 DP_CUR_COL_R_ASYNC0             : 8;
__REG32 DP_CUR_COL_G_ASYNC0             : 8;
__REG32 DP_CUR_COL_B_ASYNC0             : 8;
__REG32                                 : 8;
} __ipu_dp_cur_map_async0_bits;

/* DP Gamma Constant Async 0 Flow Register 0 (IPU_DP_GAMMA_C_ASYNC0_0) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_0             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_1             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_0_bits;

/* DP Gamma Constant Async 0 Flow Register 1 (IPU_DP_GAMMA_C_ASYNC0_1) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_2             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_3             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_1_bits;

/* DP Gamma Constant Async 0 Flow Register 2 (IPU_DP_GAMMA_C_ASYNC0_2) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_4             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_5             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_2_bits;

/* DP Gamma Constant Async 0 Flow Register 3 (IPU_DP_GAMMA_C_ASYNC0_3) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_6             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_7             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_3_bits;

/* DP Gamma Constant Async 0 Flow Register 4 (IPU_DP_GAMMA_C_ASYNC0_4) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_8             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_9             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_4_bits;

/* DP Gamma Constant Async 0 Flow Register 5 (IPU_DP_GAMMA_C_ASYNC0_5) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_10            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_11            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_5_bits;

/* DP Gamma Constant Async 0 Flow Register 6 (IPU_DP_GAMMA_C_ASYNC0_6) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_12            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_13            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_6_bits;

/* DP Gamma Constant Async 0 Flow Register 7 (IPU_DP_GAMMA_C_ASYNC0_7) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC0_14            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC0_15            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async0_7_bits;

/* DP Gamma Correction Slope Async 0 Flow Register 0 (IPU_DP_GAMMA_S_ASYNC0_0) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC0_0             : 8;
__REG32 DP_GAMMA_S_ASYNC0_1             : 8;
__REG32 DP_GAMMA_S_ASYNC0_2             : 8;
__REG32 DP_GAMMA_S_ASYNC0_3             : 8;
} __ipu_dp_gamma_s_async0_0_bits;

/* DP Gamma Correction Slope Async 0 Flow Register 1 (IPU_DP_GAMMA_S_ASYNC0_1) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC0_4             : 8;
__REG32 DP_GAMMA_S_ASYNC0_5             : 8;
__REG32 DP_GAMMA_S_ASYNC0_6             : 8;
__REG32 DP_GAMMA_S_ASYNC0_7             : 8;
} __ipu_dp_gamma_s_async0_1_bits;

/* DP Gamma Correction Slope Async 0 Flow Register 2 (IPU_DP_GAMMA_S_ASYNC0_2) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC0_8             : 8;
__REG32 DP_GAMMA_S_ASYNC0_9             : 8;
__REG32 DP_GAMMA_S_ASYNC0_10            : 8;
__REG32 DP_GAMMA_S_ASYNC0_11            : 8;
} __ipu_dp_gamma_s_async0_2_bits;

/* DP Gamma Correction Slope Async 0 Flow Register 3 (IPU_DP_GAMMA_S_ASYNC0_3) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC0_12            : 8;
__REG32 DP_GAMMA_S_ASYNC0_13            : 8;
__REG32 DP_GAMMA_S_ASYNC0_14            : 8;
__REG32 DP_GAMMA_S_ASYNC0_15            : 8;
} __ipu_dp_gamma_s_async0_3_bits;

/* DP Color Space Conversion Control Async 0 Flow Register 0 (IPU_DP_CSCA_ASYNC0_0) */
typedef struct{
__REG32 DP_CSC_A_ASYNC0_0               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC0_1               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async0_0_bits;

/* DP Color Space Conversion Control Async 0 Flow Register 1 (IPU_DP_CSCA_ASYNC0_1) */
typedef struct{
__REG32 DP_CSC_A_ASYNC0_2               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC0_3               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async0_1_bits;

/* DP Color Space Conversion Control Async 0 Flow Register 2 (IPU_DP_CSCA_ASYNC0_2) */
typedef struct{
__REG32 DP_CSC_A_ASYNC0_4               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC0_5               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async0_2_bits;

/* DP Color Space Conversion Control Async 0 Flow Register 3 (IPU_DP_CSCA_ASYNC0_3) */
typedef struct{
__REG32 DP_CSC_A_ASYNC0_6               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC0_7               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async0_3_bits;

/* DP Color Conversion Control Async 0 Flow Register 0 (IPU_DP_CSC_ASYNC0_0) */
typedef struct{
__REG32 DP_CSC_A8_ASYNC0                :10;
__REG32                                 : 6;
__REG32 DP_CSC_B0_ASYNC0                :14;
__REG32 DP_CSC_S0_ASYNC0                : 2;
} __ipu_dp_csc_async0_0_bits;

/* DP Color Conversion Control Async 0 Flow Register 1 (IPU_DP_CSC_ASYNC0_1) */
typedef struct{
__REG32 DP_CSC_B1_ASYNC0                :14;
__REG32 DP_CSC_S1_ASYNC0                : 2;
__REG32 DP_CSC_B2_ASYNC0                :14;
__REG32 DP_CSC_S2_ASYNC0                : 2;
} __ipu_dp_csc_async0_1_bits;

/* DP Common Configuration Async 1 Flow Register (IPU_DP_COM_CONF_ASYNC1) */
typedef struct{
__REG32                                 : 1;
__REG32 DP_GWSEL_ASYNC1                 : 1;
__REG32 DP_GWAM_ASYNC1                  : 1;
__REG32 DP_GWCKE_ASYNC1                 : 1;
__REG32 DP_COC_ASYNC1                   : 3;
__REG32                                 : 1;
__REG32 DP_CSC_DEF_ASYNC1               : 2;
__REG32 DP_CSC_GAMUT_SAT_EN_ASYNC1      : 1;
__REG32 DP_CSC_YUV_SAT_MODE_ASYNC1      : 1;
__REG32 DP_GAMMA_EN_ASYNC1              : 1;
__REG32 DP_GAMMA_YUV_EN_ASYNC1          : 1;
__REG32                                 :18;
} __ipu_dp_com_conf_async1_bits;

/* DP Graphic Window Control Async 1 Flow Register (IPU_DP_GRAPH_WIND_CTRL_ASYNC1) */
typedef struct{
__REG32 DP_GWCKB_ASYNC1                 : 8;
__REG32 DP_GWCKG_ASYNC1                 : 8;
__REG32 DP_GWCKR_ASYNC1                 : 8;
__REG32 DP_GWAV_ASYNC1                  : 8;
} __ipu_dp_graph_wind_ctrl_async1_bits;

/* DP Partial Plane Window Position Async 1 Flow Register (IPU_DP_FG_POS_ASYNC1) */
typedef struct{
__REG32 DP_FGYP_ASYNC1                  :11;
__REG32                                 : 5;
__REG32 DP_FGXP_ASYNC1                  :11;
__REG32                                 : 5;
} __ipu_dp_fg_pos_async1_bits;

/* DP Cursor Postion and Size Async 1 Flow Register (IPU_DP_CUR_POS_ASYNC1) */
typedef struct{
__REG32 DP_CXW_ASYNC1                   :11;
__REG32 DP_CXP_ASYNC1                   : 5;
__REG32 DP_CYH_ASYNC1                   :11;
__REG32 DP_CYP_ASYNC1                   : 5;
} __ipu_dp_cur_pos_async1_bits;

/* DP Color Cursor Mapping Async 1 Flow Register (IPU_DP_CUR_MAP_ASYNC1) */
typedef struct{
__REG32 DP_CUR_COL_R_ASYNC1             : 8;
__REG32 DP_CUR_COL_G_ASYNC1             : 8;
__REG32 DP_CUR_COL_B_ASYNC1             : 8;
__REG32                                 : 8;
} __ipu_dp_cur_map_async1_bits;

/* DP Gamma Constants Async 1 Flow Register 0 (IPU_DP_GAMMA_C_ASYNC1_0) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_0             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_1             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_0_bits;

/* DP Gamma Constants Async 1 Flow Register 1 (IPU_DP_GAMMA_C_ASYNC1_1) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_2             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_3             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_1_bits;

/* DP Gamma Constants Async 1 Flow Register 2 (IPU_DP_GAMMA_C_ASYNC1_2) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_4             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_5             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_2_bits;

/* DP Gamma Constants Async 1 Flow Register 3 (IPU_DP_GAMMA_C_ASYNC1_3) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_6             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_7             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_3_bits;

/* DP Gamma Constants Async 1 Flow Register 4 (IPU_DP_GAMMA_C_ASYNC1_4) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_8             : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_9             : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_4_bits;

/* DP Gamma Constants Async 1 Flow Register 5 (IPU_DP_GAMMA_C_ASYNC1_5) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_10            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_11            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_5_bits;

/* DP Gamma Constants Async 1 Flow Register 6 (IPU_DP_GAMMA_C_ASYNC1_6) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_12            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_13            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_6_bits;

/* DP Gamma Constants Async 1 Flow Register 7 (IPU_DP_GAMMA_C_ASYNC1_7) */
typedef struct{
__REG32 DP_GAMMA_C_ASYNC1_14            : 9;
__REG32                                 : 7;
__REG32 DP_GAMMA_C_ASYNC1_15            : 9;
__REG32                                 : 7;
} __ipu_dp_gamma_c_async1_7_bits;

/* DP Gamma Correction Slope Async 1 Flow Register 0 (IPU_DP_GAMMA_S_ASYN1_0) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC1_0             : 8;
__REG32 DP_GAMMA_S_ASYNC1_1             : 8;
__REG32 DP_GAMMA_S_ASYNC1_2             : 8;
__REG32 DP_GAMMA_S_ASYNC1_3             : 8;
} __ipu_dp_gamma_s_asyn1_0_bits;

/* DP Gamma Correction Slope Async 1 Flow Register 1 (IPU_DP_GAMMA_S_ASYN1_1) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC1_4             : 8;
__REG32 DP_GAMMA_S_ASYNC1_5             : 8;
__REG32 DP_GAMMA_S_ASYNC1_6             : 8;
__REG32 DP_GAMMA_S_ASYNC1_7             : 8;
} __ipu_dp_gamma_s_asyn1_1_bits;

/* DP Gamma Correction Slope Async 1 Flow Register 2 (IPU_DP_GAMMA_S_ASYN1_2) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC1_8             : 8;
__REG32 DP_GAMMA_S_ASYNC1_9             : 8;
__REG32 DP_GAMMA_S_ASYNC1_10            : 8;
__REG32 DP_GAMMA_S_ASYNC1_11            : 8;
} __ipu_dp_gamma_s_asyn1_2_bits;

/* DP Gamma Correction Slope Async 1 Flow Register 3 (IPU_DP_GAMMA_S_ASYN1_3) */
typedef struct{
__REG32 DP_GAMMA_S_ASYNC1_12            : 8;
__REG32 DP_GAMMA_S_ASYNC1_13            : 8;
__REG32 DP_GAMMA_S_ASYNC1_14            : 8;
__REG32 DP_GAMMA_S_ASYNC1_15            : 8;
} __ipu_dp_gamma_s_asyn1_3_bits;

/* DP Color Space Converstion Control Async 1 Flow Register 0 (IPU_DP_CSCA_ASYNC1_0) */
typedef struct{
__REG32 DP_CSC_A_ASYNC1_0               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC1_1               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async1_0_bits;

/* DP Color Space Converstion Control Async 1 Flow Register 1 (IPU_DP_CSCA_ASYNC1_1) */
typedef struct{
__REG32 DP_CSC_A_ASYNC1_2               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC1_3               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async1_1_bits;

/* DP Color Space Converstion Control Async 1 Flow Register 2 (IPU_DP_CSCA_ASYNC1_2) */
typedef struct{
__REG32 DP_CSC_A_ASYNC1_4               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC1_5               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async1_2_bits;

/* DP Color Space Converstion Control Async 1 Flow Register 3 (IPU_DP_CSCA_ASYNC1_3) */
typedef struct{
__REG32 DP_CSC_A_ASYNC1_6               :10;
__REG32                                 : 6;
__REG32 DP_CSC_A_ASYNC1_7               :10;
__REG32                                 : 6;
} __ipu_dp_csca_async1_3_bits;

/* DP Color Conversion Control Async 1 Flow Register 0 (IPU_DP_CSC_ASYNC1_0) */
typedef struct{
__REG32 DP_CSC_A8_ASYNC1                :10;
__REG32                                 : 6;
__REG32 DP_CSC_B0_ASYNC1                :14;
__REG32 DP_CSC_S0_ASYNC1                : 2;
} __ipu_dp_csc_async1_0_bits;

/* DP Color Conversion Control Async 1 Flow Register 1 (IPU_DP_CSC_ASYNC1_1) */
typedef struct{
__REG32 DP_CSC_B1_ASYNC1                :14;
__REG32 DP_CSC_S1_ASYNC1                : 2;
__REG32 DP_CSC_B2_ASYNC1                :14;
__REG32 DP_CSC_S2_ASYNC1                : 2;
} __ipu_dp_csc_async1_1_bits;

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

/* Power Saving Masters 0 (M4IF.PSM0) */
typedef struct{
__REG32 M0_PSD  : 1;
__REG32 M0_ROC  : 3;
__REG32 M0_PSS  : 1;
__REG32 M0_RIS  : 1;
__REG32 M0_WIS  : 1;
__REG32         : 1;
__REG32 M0_PST  : 8;
__REG32 M1_PSD  : 1;
__REG32 M1_ROC  : 3;
__REG32 M1_PSS  : 1;
__REG32 M1_RIS  : 1;
__REG32 M1_WIS  : 1;
__REG32         : 1;
__REG32 M1_PST  : 8;
} __m4if_psm0_bits;

/* Power Saving Masters 1 (M4IF.PSM1) */
typedef struct{
__REG32 M2_PSD  : 1;
__REG32 M2_ROC  : 3;
__REG32 M2_PSS  : 1;
__REG32 M2_RIS  : 1;
__REG32 M2_WIS  : 1;
__REG32         : 1;
__REG32 M2_PST  : 8;
__REG32 M3_PSD  : 1;
__REG32 M3_ROC  : 3;
__REG32 M3_PSS  : 1;
__REG32 M3_RIS  : 1;
__REG32 M3_WIS  : 1;
__REG32         : 1;
__REG32 M3_PST  : 8;
} __m4if_psm1_bits;

/* M4IF Debug Status Register #6 */
typedef struct{
__REG32 REQ_ACC :21;
__REG32         :11;
} __m4if_mdsr6_bits;

/* M4IF Debug Status Register #0 */
typedef struct{
__REG32 MAX_DPR : 6;
__REG32         :10;
__REG32 MAX_TIME:10;
__REG32         : 6;
} __m4if_mdsr0_bits;

/* M4IF Debug Status Register #1 */
typedef struct{
__REG32 TOT_ACC :24;
__REG32         : 8;
} __m4if_mdsr1_bits;

/* F_Basic Priority Reg #0 */
typedef struct{
__REG32 FBPM_M0_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M0_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M1_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M1_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M2_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M2_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M3_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M3_RD  : 3;
__REG32             : 1;
} __m4if_fbpm0_bits;

/* F_Basic Priority Reg #1 */
typedef struct{
__REG32 FBPM_M4_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M4_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M5_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M5_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M6_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M6_RD  : 3;
__REG32             : 1;
__REG32 FBPM_M7_WR  : 3;
__REG32             : 1;
__REG32 FBPM_M7_RD  : 3;
__REG32             : 1;
} __m4if_fbpm1_bits;

/* MIF4 Control Register */
typedef struct{
__REG32 MIF4_GUARD  : 4;
__REG32 MIF4_DYN_MAX: 4;
__REG32 MIF4_DYN_JMP: 4;
__REG32             : 4;
__REG32 MIF4_ACC_HIT: 3;
__REG32 MIF4_PAG_HIT: 3;
__REG32             : 1;
__REG32 MIF4_IDTOO  : 1;
__REG32 MIF4_ID     : 4;
__REG32 M4IF_MAS    : 3;
__REG32 MIF4_BYP_EN : 1;
} __m4if_ctrl_bits;

/* I2_Unit_Level_Arbitration_ Register */
typedef struct{
__REG32 I2L3M01     : 3;
__REG32 I2L3M23     : 3;
__REG32 I2L3M45     : 3;
__REG32 I2L3M67     : 3;
__REG32 I2L4M03     : 3;
__REG32 I2L4M47     : 3;
__REG32 I2L5        : 3;
__REG32             :11;
} __m4if_i2ula_bits;

/* M4IF Internal 2 Control Register */
typedef struct{
__REG32 DI2PS       : 1;
__REG32 I2LPMD      : 1;
__REG32 I2DVFS      : 1;
__REG32 I2PST       : 8;
__REG32 I2PSS       : 1;
__REG32 I2LPACK     : 1;
__REG32 I2DVACK     : 1;
__REG32             :18;
} __m4if_rint2_bits;

/* Step By Step Address Controls */
typedef struct{
__REG32 SBS_END       : 1;
__REG32 SBS_MASTER    : 3;
__REG32 SBS_MASTER_ID : 4;
__REG32 SBS_AXI_ID    : 4;
__REG32 SBS_CACHE     : 4;
__REG32 SBS_LOCK      : 2;
__REG32 SBS_PROT      : 3;
__REG32 SBS_BURST     : 2;
__REG32 SBS_SIZE      : 2;
__REG32 SBS_LEN       : 3;
__REG32 SBS_TYPE      : 1;
__REG32 SBS_VLD       : 1;
__REG32               : 2;
} __m4if_sbs1_bits;

/* M4IF Control Register #0 */
typedef struct{
__REG32 DIPS          : 1;
__REG32 DSPS          : 1;
__REG32 DFPS          : 1;
__REG32               : 1;
__REG32 LPMD          : 1;
__REG32 ILPMD         : 1;
__REG32 SLPMD         : 1;
__REG32 FLMPD         : 1;
__REG32 DVFS          : 1;
__REG32 IDVFS         : 1;
__REG32 SDVFS         : 1;
__REG32 FDVFS         : 1;
__REG32 EAS0          : 1;
__REG32 EAS1          : 1;
__REG32 EAS2          : 1;
__REG32 EAS3          : 1;
__REG32 IPSS          : 1;
__REG32 SPSS          : 1;
__REG32 FPSS          : 1;
__REG32               : 1;
__REG32 LPACK         : 1;
__REG32 ILPACK        : 1;
__REG32 SLPACK        : 1;
__REG32 FLPACK        : 1;
__REG32 DVACK         : 1;
__REG32 IDVACK        : 1;
__REG32 SDVACK        : 1;
__REG32 FDVACK        : 1;
__REG32               : 3;
__REG32 SW_RST        : 1;
} __m4if_mcr0_bits;

/* M4IF Control Register #0 */
typedef struct{
__REG32 FPST          : 8;
__REG32               : 1;
__REG32 SPST          : 8;
__REG32               : 1;
__REG32 IPST          : 8;
__REG32               : 2;
__REG32 EAS4          : 1;
__REG32 EAS5          : 1;
__REG32 EAS6          : 1;
__REG32 EAS7          : 1;
} __m4if_mcr1_bits;

/* M4IF Debug Control Register */
typedef struct{
__REG32 VARB          : 2;
__REG32               : 6;
__REG32 RARB          : 2;
__REG32               : 1;
__REG32 I2SBS         : 1;
__REG32 I1SBS         : 1;
__REG32 SSBS          : 1;
__REG32 FSBS          : 1;
__REG32 FSBS_EN       : 1;
__REG32               : 1;
__REG32 DDPM          : 3;
__REG32 DDPT          : 1;
__REG32 SSBS_EN       : 1;
__REG32 I1SBS_EN      : 1;
__REG32 I2SBS_EN      : 1;
__REG32               : 3;
__REG32 DBG_RST       : 1;
__REG32               : 3;
__REG32 DBG_EN        : 1;
} __m4if_mdcr_bits;

/* Fast Arbitration Control Register */
typedef struct{
__REG32 FDPE          : 1;
__REG32               : 3;
__REG32 FDPUR         : 4;
__REG32 FRDT          : 1;
__REG32               :23;
} __m4if_facr_bits;

/* F_Priority Weighting Configuration Register */
typedef struct{
__REG32 FDPM          : 4;
__REG32               : 1;
__REG32 FAPR          : 2;
__REG32               : 1;
__REG32 FPCR          : 3;
__REG32               : 5;
__REG32 FPHR          : 4;
__REG32 FAHR          : 3;
__REG32               : 9;
} __m4if_fpwc_bits;

/* Power Saving Masters 2 */
typedef struct{
__REG32 M4_PSD        : 1;
__REG32 M4_ROC        : 3;
__REG32 M4_PSS        : 1;
__REG32 M4_RIS        : 1;
__REG32 M4_WIS        : 1;
__REG32               : 1;
__REG32 M4_PST        : 8;
__REG32 M5_PSD        : 1;
__REG32 M5_ROC        : 3;
__REG32 M5_PSS        : 1;
__REG32 M5_RIS        : 1;
__REG32 M5_WIS        : 1;
__REG32               : 1;
__REG32 M5_PST        : 8;
} __m4if_psm2_bits;

/* Power Saving Masters 3 */
typedef struct{
__REG32 M6_PSD        : 1;
__REG32 M6_ROC        : 3;
__REG32 M6_PSS        : 1;
__REG32 M6_RIS        : 1;
__REG32 M6_WIS        : 1;
__REG32               : 1;
__REG32 M6_PST        : 8;
__REG32 M7_PSD        : 1;
__REG32 M7_ROC        : 3;
__REG32 M7_PSS        : 1;
__REG32 M7_RIS        : 1;
__REG32 M7_WIS        : 1;
__REG32               : 1;
__REG32 M7_PST        : 8;
} __m4if_psm3_bits;

/* F_Unit_Level_Arbitration_ Register */
typedef struct{
__REG32 FL3M01        : 3;
__REG32 FL3M23        : 3;
__REG32 FL3M45        : 3;
__REG32 FL3M67        : 3;
__REG32 FL4M03        : 3;
__REG32 FL4M47        : 3;
__REG32 FL5           : 3;
__REG32               :11;
} __m4if_fula_bits;

/* S_Unit_Level_Arbitration_ Register */
typedef struct{
__REG32 SL3M01        : 3;
__REG32 SL3M23        : 3;
__REG32 SL3M45        : 3;
__REG32 SL3M67        : 3;
__REG32 SL4M03        : 3;
__REG32 SL4M47        : 3;
__REG32 SL5           : 3;
__REG32               :11;
} __m4if_sula_bits;

/* S_Unit_Level_Arbitration_ Register */
typedef struct{
__REG32 IL3M01        : 3;
__REG32 IL3M23        : 3;
__REG32 IL3M45        : 3;
__REG32 IL3M67        : 3;
__REG32 IL4M03        : 3;
__REG32 IL4M47        : 3;
__REG32 IL5           : 3;
__REG32               :11;
} __m4if_iula_bits;

/* Fast_Dynamic_Priority_Status Register */
typedef struct{
__REG32 FDP0          : 4;
__REG32 FDP1          : 4;
__REG32 FDP2          : 4;
__REG32 FDP3          : 4;
__REG32 FDP4          : 4;
__REG32               :12;
} __m4if_fdps_bits;

/* Fast_Dynamic_Priority_Control Register */
typedef struct{
__REG32               : 1;
__REG32 FD0_RW        : 1;
__REG32 FD0_MAS       : 3;
__REG32               : 1;
__REG32 FD1_RW        : 1;
__REG32 FD1_MAS       : 3;
__REG32               : 1;
__REG32 FD2_RW        : 1;
__REG32 FD2_MAS       : 3;
__REG32               : 1;
__REG32 FD3_RW        : 1;
__REG32 FD3_MAS       : 3;
__REG32               : 1;
__REG32 FD4_RW        : 1;
__REG32 FD4_MAS       : 3;
__REG32               : 7;
} __m4if_fdpc_bits;

/* Master Len Interrupt */
typedef struct{
__REG32 m0len         : 1;
__REG32 m1len         : 1;
__REG32 m2len         : 1;
__REG32 m3len         : 1;
__REG32 m4len         : 1;
__REG32 m5len         : 1;
__REG32 m6len         : 1;
__REG32 m7len         : 1;
__REG32               : 4;
__REG32 i2b816        : 1;
__REG32 i1b816        : 1;
__REG32 sb816         : 1;
__REG32 fb816         : 1;
__REG32               :16;
} __m4if_mlen_bits;

/* Watermark Interrupt and Status #0 Register */
typedef struct{
__REG32               : 6;
__REG32 WS0_6         : 1;
__REG32 WS0_7         : 1;
__REG32               :24;
} __m4if_wmis0_bits;

/* Watermark Start ADDR_1 Register */
typedef struct{
__REG32 WMS1          :20;
__REG32               :11;
__REG32 WE1           : 1;
} __m4if_wmsa1_bits;

/*Watermark Start ADDR_1 Register */
typedef struct{
__REG32 WME1          :20;
__REG32               :12;
} __m4if_wmea1_bits;

/* Watermark Interrupt and Status #1 Register */
typedef struct{
__REG32               : 6;
__REG32 WS0_6         : 1;
__REG32 WS0_7         : 1;
__REG32               :11;
__REG32 VMID_1        :11;
__REG32               : 1;
__REG32 WIE1          : 1;
} __m4if_wmis1_bits;

/* Device Control Configuration Register (MLB.DCCR) */
typedef struct{
__REG32 MDA           : 8;
__REG32               :15;
__REG32 MRS           : 1;
__REG32 MHRE          : 1;
__REG32 MLE           : 1;
__REG32 MLK           : 1;
__REG32               : 1;
__REG32 MCS           : 2;
__REG32 LBM           : 1;
__REG32 MDE           : 1;
} __mlb_dccr_bits;

/* System Status Configuration Register (MLB.SSCR) */
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

/* System Mask Configuration Register (MLB.SMCR) */
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

/* Version Control Configuration Register (MLB.VCCR) */
typedef struct{
__REG32 MMI           : 8;
__REG32 MMA           : 8;
__REG32 UMI           : 8;
__REG32 UMA           : 8;
} __mlb_vccr_bits;

/* Synchronous Base Address Configuration Register (MLB.SBCR) */
typedef struct{
__REG32 STBA          :16;
__REG32 SRBA          :16;
} __mlb_sbcr_bits;

/* Asynchronous Base Address Configuration Register (MLB.ABCR) */
typedef struct{
__REG32 ATBA          :16;
__REG32 ARBA          :16;
} __mlb_abcr_bits;

/* Control Base Address Configuration Register (MLB.CBCR) */
typedef struct{
__REG32 CTBA          :16;
__REG32 CRBA          :16;
} __mlb_cbcr_bits;

/* Isochronous Base Address Configuration Register (MLB.IBCR) */
typedef struct{
__REG32 ITBA          :16;
__REG32 IRBA          :16;
} __mlb_ibcr_bits;

/* Channel Interrupt Configuration Register (MLB.CICR) */
typedef struct{
__REG32 CSU           :16;
__REG32               :16;
} __mlb_cicr_bits;

/* Channel n Entry Configuration Register (MLB.CECRn) */
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

/* Channel n Status Configuration Register (MLB.CSCRn) */
typedef struct{
__REG32 CBPE          : 1;
__REG32 CBDB          : 1;
__REG32 CBD           : 1;
__REG32 CBS           : 1;
__REG32 BE            : 1;
__REG32 HBE           : 1;
__REG32 LFS           : 1;
__REG32               : 1;
__REG32 PBPE          : 1;
__REG32 PBDBc         : 1;
__REG32 PBD           : 1;
__REG32 PBS           : 1;
__REG32               : 4;
__REG32 RDY           : 1;
__REG32 GIRBGB        : 1;
__REG32 IVB           : 2;
__REG32               :10;
__REG32 BF            : 1;
__REG32 BM            : 1;
} __mlb_cscr_bits;

/* Channel n Current Configuration Register (MLB.CCCRn) */
typedef struct{
__REG32 BFA           :16;
__REG32 BCA           :16;
} __mlb_ccbcr_bits;

/* Channel n Next Buffer Configuration Register (MLB.CNBCRn) */
typedef struct{
__REG32 BEA           :16;
__REG32 BSA           :16;
} __mlb_cnbcr_bits;

/* Local Channel n Buffer Configuration Register (MLB.LCBCRn) */
typedef struct{
__REG32 SA            : 9;
__REG32               : 4;
__REG32 BD            : 7;
__REG32               :12;
} __mlb_lcbcr_bits;

/* NAND Flash Command (NFC.NAND_CMD) */
typedef struct{
__REG32 CMD0          : 8;
__REG32 CMD1          : 8;
__REG32 CMD2          : 8;
__REG32 CMD3          : 8;
} __nfc_nand_cmd_bits;

/* NFC Configuration (NFC.NFC_CONFIGURATION1) */
typedef struct{
__REG32 SP_EN             : 1;
__REG32 NF_CE             : 1;
__REG32 NFC_RST           : 1;
__REG32                   : 1;
__REG32 RBA               : 3;
__REG32                   : 1;
__REG32 NUM_OF_ITERATIONS : 4;
__REG32 ACTIVE_CS         : 3;
__REG32                   : 1;
__REG32 NF_STATUS         :16;
} __nfc_configuration1_bits;

/* ECC Status and Result of Flash Operation (NFC.ECC_STATUS_RESULT) */
typedef struct{
__REG32 NOBER0            : 4;
__REG32 NOBER1            : 4;
__REG32 NOBER2            : 4;
__REG32 NOBER3            : 4;
__REG32 NOBER4            : 4;
__REG32 NOBER5            : 4;
__REG32 NOBER6            : 4;
__REG32 NOBER7            : 4;
} __nfc_ecc_status_result_bits;

/* Status Summary (NFC.STATUS_SUM) */
typedef struct{
__REG32 NAND_STATUS_SUM   : 8;
__REG32 ECC_SUM           : 8;
__REG32                   :16;
} __nfc_status_sum_bits;

/* Initiate a NAND Transaction (NFC.LAUNCH NFC) */
typedef struct{
__REG32 FCMD              : 1;
__REG32 FADD              : 1;
__REG32 FDI               : 1;
__REG32 FDO               : 3;
__REG32 AUTO_PROG         : 1;
__REG32 AUTO_READ         : 1;
__REG32                   : 1;
__REG32 AUTO_ERASE        : 1;
__REG32 AUTO_COPY_BACK0   : 1;
__REG32 AUTO_COPY_BACK1   : 1;
__REG32 AUTO_STAT         : 1;
__REG32 ATOMIC_ST_POL     : 1;
__REG32                   :18;
} __nfc_launch_bits;

/* NAND Flash Write Protection (NFC.NFC_WR_PROT) */
typedef struct{
__REG32                   : 6;
__REG32 BLS               : 2;
__REG32                   :24;
} __nfc_wr_protect_bits;

/* NAND Flash Operation Configuration2 (NFC.NFC_CONFIGURATION2) */
typedef struct{
__REG32 PS                : 2;
__REG32 SYM               : 1;
__REG32 ECC_EN            : 1;
__REG32 NUM_CMD_PHASES    : 1;
__REG32 NUM_ADR_PHASES0   : 1;
__REG32 ECC_MODE          : 2;
__REG32 PPB               : 2;
__REG32 EDC               : 2;
__REG32 NUM_ADR_PHASES1   : 2;
__REG32 AUTO_PROG_DONE_MSK: 1;
__REG32 INT_MSK           : 1;
__REG32 SPAS              : 8;
__REG32 ST_CMD            : 8;
} __nfc_configuration2_bits;

/* NAND Flash Operation Configuration3 (NFC.NFC_CONFIGURATION3) */
typedef struct{
__REG32 ADD_OP            : 2;
__REG32 TOO               : 1;
__REG32 FW                : 1;
__REG32 SB2R              : 3;
__REG32                   : 1;
__REG32 SBB               : 3;
__REG32 DMA_MODE          : 1;
__REG32 NUM_OF_DEVICES    : 3;
__REG32 RBB_MODE          : 1;
__REG32 FMP               : 3;
__REG32 STATUS_SAMP_SEL   : 1;
__REG32 NO_SDMA           : 1;
__REG32                   :11;
} __nfc_configuration3_bits;

/* NAND Flash IP control (NFC.NFC_IPC) */
typedef struct{
__REG32 CREQ              : 1;
__REG32 CACK              : 1;
__REG32                   :24;
__REG32 DMA_STATUS        : 2;
__REG32 RB_B              : 1;
__REG32 LPS               : 1;
__REG32 AUTO_PROG_DONE    : 1;
__REG32 INT               : 1;
} __nfc_ipc_bits;

/* AXI Error Address (NFC.NFC_AXI_ERR_ADD) */
typedef struct{
__REG32 AXI_ERR_ADD       :13;
__REG32                   :19;
} __nfc_axi_err_add_bits;

/* Delay-line (NFC.NFC_DELAY_LINE) */
typedef struct{
__REG32 NFC_ABS_DEL       : 8;
__REG32 NFC_OFF_DEL       : 8;
__REG32                   :16;
} __nfc_delay_line_bits;

/* Control Register (OWIRE.CONTROL) */
typedef struct{
__REG16       : 3;
__REG16 RDST  : 1;
__REG16 WR1   : 1;
__REG16 WR0   : 1;
__REG16 PST   : 1;
__REG16 RPP   : 1;
__REG16       : 8;
} __owire_control_bits;

/* Time Divider Register (OWIRE.TIME_DIVIDER) */
typedef struct{
__REG16 DVDR  : 8;
__REG16       : 8;
} __owire_time_divider_bits;

/* Reset Register (OWIRE.RESET) */
typedef struct{
__REG16 RST  : 1;
__REG16      :15;
} __owire_reset_bits;

/* Command Register (OWIRE.COMMAND) */
typedef struct{
__REG16      : 1;
__REG16 SRA  : 1;
__REG16      :14;
} __owire_command_bits;

/* Transmit/Receive Register (OWIRE.TX/RX) */
typedef struct{
__REG16 DATA : 8;
__REG16      : 8;
} __owire_tx_rx_bits;

/* Interrupt Register (OWIRE.INTERRUPT) */
typedef struct{
__REG16 PD   : 1;
__REG16 PDR  : 1;
__REG16 TBE  : 1;
__REG16 TSRE : 1;
__REG16 RBF  : 1;
__REG16 RSRF : 1;
__REG16      :10;
} __owire_interrupt_bits;

/* Interrupt Enable Register (OWIRE.INTERRUPT_EN) */
typedef struct{
__REG16 EPD  : 1;
__REG16 IAS  : 1;
__REG16 ETBE : 1;
__REG16 ETSE : 1;
__REG16 ERBF : 1;
__REG16 ERSF : 1;
__REG16      :10;
} __owire_interrupt_en_bits;

/* ATA_CONTROL Register */
typedef struct{
__REG8  iordy_en            : 1;
__REG8  dma_write           : 1;
__REG8  dma_ultra_selected  : 1;
__REG8  dma_pending         : 1;
__REG8  fifo_rcv_en         : 1;
__REG8  fifo_tx_en          : 1;
__REG8  ata_rst_b           : 1;
__REG8  fifo_rst_b          : 1;
} __ata_control_bits;

/* INTERRUPT_PENDING Register */
typedef struct{
__REG8                      : 3;
__REG8  ata_irtrq2          : 1;
__REG8  controller_idle     : 1;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8  ata_intrq1          : 1;
} __ata_intr_pend_bits;

/* INTERRUPT_ENABLE Register */
typedef struct{
__REG8                      : 3;
__REG8  ata_irtrq2          : 1;
__REG8  controller_idle     : 1;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8  ata_intrq1          : 1;
} __ata_intr_ena_bits;

/* INTERRUPT_CLEAR Register */
typedef struct{
__REG8                      : 5;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8                      : 1;
} __ata_intr_clr_bits;

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

/* ROMC Control Register */
typedef struct{
__REG32 DATAFIX   : 8;
__REG32           :21;
__REG32 DIS       : 1;
__REG32           : 2;
} __romc_rompatchcntl_bits;

/* ROMC Enable Register */
typedef struct{
__REG32 ENABLE0   : 1;
__REG32 ENABLE1   : 1;
__REG32 ENABLE2   : 1;
__REG32 ENABLE3   : 1;
__REG32 ENABLE4   : 1;
__REG32 ENABLE5   : 1;
__REG32 ENABLE6   : 1;
__REG32 ENABLE7   : 1;
__REG32 ENABLE8   : 1;
__REG32 ENABLE9   : 1;
__REG32 ENABLE10  : 1;
__REG32 ENABLE11  : 1;
__REG32 ENABLE12  : 1;
__REG32 ENABLE13  : 1;
__REG32 ENABLE14  : 1;
__REG32 ENABLE15  : 1;
__REG32           :16;
} __romc_rompatchenl_bits;

/* ROMC Address Register */
typedef struct{
__REG32 THUMBX    : 1;
__REG32 ADDRX     :21;
__REG32           :10;
} __romc_rompatcha_bits;

/* ROMC Status Register */
typedef struct{
__REG32 SOURCE    : 6;
__REG32           :11;
__REG32 SW        : 1;
__REG32           :14;
} __romc_rompatchsr_bits;

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

/* Configuration Register (CONFIG) */
typedef struct{
__REG32 LOCK            : 1;
__REG32 SRESET_LOCK_CLR : 1;
__REG32                 :30;
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

/* DMA Requests (EVT_MIRROR) */
typedef struct{
__REG32 EVENTS32: 1;
__REG32 EVENTS33: 1;
__REG32 EVENTS34: 1;
__REG32 EVENTS35: 1;
__REG32 EVENTS36: 1;
__REG32 EVENTS37: 1;
__REG32 EVENTS38: 1;
__REG32 EVENTS39: 1;
__REG32 EVENTS40: 1;
__REG32 EVENTS41: 1;
__REG32 EVENTS42: 1;
__REG32 EVENTS43: 1;
__REG32 EVENTS44: 1;
__REG32 EVENTS45: 1;
__REG32 EVENTS46: 1;
__REG32 EVENTS47: 1;
__REG32         :16;
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
} __sdma_xtrig1_conf_bits;

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
} __sdma_xtrig2_conf_bits;

/* Channel Priority Register (CHNPRIn) */
typedef struct{
__REG32 CHNPRI      : 3;
__REG32             :29;
} __sdma_chnpri_bits;

/* Channel Priority Register (CHNPRIn) */
typedef struct{
__REG32 RAR         : 3;
__REG32             :13;
__REG32 ROI         : 2;
__REG32             :12;
__REG32 RMO         : 2;
} __spba_prr_bits;

/* SPDIF Configuration Register (SPDIF.SCR) */
typedef struct{
__REG32 USrc_Sel        : 2;
__REG32 TxSel           : 3;
__REG32 ValCtrl         : 1;
__REG32                 : 2;
__REG32 DMA_Tx_En       : 1;
__REG32 DMA_Rx_En       : 1;
__REG32 TxFIFO_Ctrl     : 2;
__REG32 soft_reset      : 1;
__REG32 Low_power       : 1;
__REG32                 : 1;
__REG32 TxFIFOEmpty_Se  : 2;
__REG32 TxAutoSync      : 1;
__REG32 RxAutoSync      : 1;
__REG32 RxFIFOFull_Sel  : 2;
__REG32 RxFIFO_Rst      : 1;
__REG32 RxFIFO_Off_On   : 1;
__REG32 RxFIFO_Ctrl     : 1;
__REG32                 : 8;
} __spdif_scr_bits;

/* CDText Control Register (SPDIF.SRCD) */
typedef struct{
__REG32                 : 1;
__REG32 USyncMode       : 1;
__REG32                 :30;
} __spdif_srcd_bits;

/* PhaseConfig Register (SPDIF.SRPC) */
typedef struct{
__REG32                 : 3;
__REG32 GainSel         : 3;
__REG32 LOCK            : 1;
__REG32 ClkSrc_Sel      : 4;
__REG32                 :21;
} __spdif_srpc_bits;

/* InterruptEn (SPDIF.SIE) Register */
typedef struct{
__REG32 RxFIFOFul       : 1;
__REG32 TxEm            : 1;
__REG32 LockLoss        : 1;
__REG32 RxFIFOResyn     : 1;
__REG32 RxFIFOUnOv      : 1;
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

/* SPDIFTxClk Register (SPDIF.STC) */
typedef struct{
__REG32 TxClk_DF        : 7;
__REG32                 : 1;
__REG32 TxClk_Source    : 3;
__REG32 SYSCLK_DF       : 9;
__REG32                 :12;
} __spdif_stc_bits;

/* SRC Control Register (SRC.SCR) */
typedef struct{
__REG32 warm_reset_enable     : 1;
__REG32 sw_gpu_rst            : 1;
__REG32 sw_vpu_rst            : 1;
__REG32 sw_ipu_rst            : 1;
__REG32 sw_open_vg_rst        : 1;
__REG32 warm_rst_bypass_count : 2;
__REG32 mask_wdog_rst         : 4;
__REG32 weim_rst              : 1;
__REG32                       :20;
} __src_scr_bits;

/* SRC Boot Mode Register (SRC.SBMR) */
typedef struct{
__REG32 BOOT_CFG1             : 8;
__REG32 BOOT_CFG2             : 8;
__REG32 BOOT_CFG3             : 8;
__REG32 BMOD                  : 2;
__REG32 BT_FUSE_SEL           : 1;
__REG32 TEST_MODE             : 3;
__REG32 BT_RESERVED           : 2;
} __src_sbmr_bits;

/* SRC Reset Status Register (SRC.SRSR) */
typedef struct{
__REG32 ipp_reset_b           : 1;
__REG32                       : 1;
__REG32 csu_reset_b           : 1;
__REG32 ipp_user_reset_b      : 1;
__REG32 wdog_rst_b            : 1;
__REG32 jtag_rst_b            : 1;
__REG32 jtag_sw_rst           : 1;
__REG32                       : 9;
__REG32 warm_boot             : 1;
__REG32                       :15;
} __src_srsr_bits;

/* SRC Interrupt Status Register (SRC.SISR) */
typedef struct{
__REG32 gpu_passed_reset      : 1;
__REG32 vpu_passed_reset      : 1;
__REG32 ipu_passed_reset      : 1;
__REG32 open_vg_passed_reset  : 1;
__REG32                       :28;
} __src_sisr_bits;

/* SRC Interrupt Mask Register (SRC.SIMR) */
typedef struct{
__REG32 mask_gpu_passed_reset     : 1;
__REG32 mask_vpu_passed_reset     : 1;
__REG32 mask_ipu_passed_reset     : 1;
__REG32 mask_open_vg_passed_reset : 1;
__REG32                           :28;
} __src_simr_bits;

/* SRTC LP Secure Counter LSB Register (SRTC.LPSCLR) */
typedef struct{
__REG32                           :17;
__REG32 LLPSC                     :15;
} __srtc_lpsclr_bits;

/* SRTC LP Control Register (SRTC.LPCR) */
typedef struct{
__REG32 SWR_LP                    : 1;
__REG32                           : 2;
__REG32 EN_LP                     : 1;
__REG32 WAE                       : 1;
__REG32 SAE                       : 1;
__REG32 SI                        : 1;
__REG32 ALP                       : 1;
__REG32 LTC                       : 1;
__REG32 LMC                       : 1;
__REG32 SV                        : 1;
__REG32 NSA                       : 1;
__REG32 NVEIE                     : 1;
__REG32 IEIE                      : 1;
__REG32 NVE                       : 1;
__REG32 IE                        : 1;
__REG32 SCALM_LP                  : 2;
__REG32 SCAL_LP                   : 5;
__REG32                           : 9;
} __srtc_lpcr_bits;

/* SRTC LP Status Register (SRTC.LPSR) */
typedef struct{
__REG32 TRI                       : 1;
__REG32 PGD                       : 1;
__REG32 CTD                       : 1;
__REG32 ALP                       : 1;
__REG32 MR                        : 1;
__REG32 TR                        : 1;
__REG32 EAD                       : 1;
__REG32 IT                        : 3;
__REG32 SM                        : 2;
__REG32 STATE_LP                  : 2;
__REG32 NVES                      : 1;
__REG32 IES                       : 1;
__REG32                           :16;
} __srtc_lpsr_bits;

/* SRTC LP General Purpose Register (SRTC.LPGR) */
typedef struct{
__REG32 LPGR                      :29;
__REG32                           : 2;
__REG32 SW_ISO                    : 1;
} __srtc_lpgr_bits;

/* SRTC LP General Purpose Register (SRTC.LPGR) */
typedef struct{
__REG32                           :17;
__REG32 LHPC                      :15;
} __srtc_hpclr_bits;

/* SRTC HP Alarm LSB Register (SRTC.HPALR) */
typedef struct{
__REG32                           :17;
__REG32 LHPA                      :15;
} __srtc_hpalr_bits;

/* SRTC HP Control Register (SRTC.HPCR) */
typedef struct{
__REG32 SWR_HP                    : 1;
__REG32                           : 2;
__REG32 EN_HP                     : 1;
__REG32 TS                        : 1;
__REG32                           :11;
__REG32 SCALM_HP                  : 1;
__REG32                           : 1;
__REG32 SCAL_HP                   : 5;
__REG32                           : 9;
} __srtc_hpcr_bits;

/* SRTC HP Interrupt Status Register (SRTC.HPISR) */
typedef struct{
__REG32 PI0                       : 1;
__REG32 PI1                       : 1;
__REG32 PI2                       : 1;
__REG32 PI3                       : 1;
__REG32 PI4                       : 1;
__REG32 PI5                       : 1;
__REG32 PI6                       : 1;
__REG32 PI7                       : 1;
__REG32 PI8                       : 1;
__REG32 PI9                       : 1;
__REG32 PI10                      : 1;
__REG32 PI11                      : 1;
__REG32 PI12                      : 1;
__REG32 PI13                      : 1;
__REG32 PI14                      : 1;
__REG32 PI15                      : 1;
__REG32 AHP                       : 1;
__REG32                           : 1;
__REG32 WDHP                      : 1;
__REG32 WDLP                      : 1;
__REG32 WPHP                      : 1;
__REG32 WPLP                      : 1;
__REG32                           :10;
} __srtc_hpisr_bits;

/* SRTC HP Interrupt Enable Register (SRTC.HPIENR) */
typedef struct{
__REG32 PI0                       : 1;
__REG32 PI1                       : 1;
__REG32 PI2                       : 1;
__REG32 PI3                       : 1;
__REG32 PI4                       : 1;
__REG32 PI5                       : 1;
__REG32 PI6                       : 1;
__REG32 PI7                       : 1;
__REG32 PI8                       : 1;
__REG32 PI9                       : 1;
__REG32 PI10                      : 1;
__REG32 PI11                      : 1;
__REG32 PI12                      : 1;
__REG32 PI13                      : 1;
__REG32 PI14                      : 1;
__REG32 PI15                      : 1;
__REG32 AHP                       : 1;
__REG32                           : 1;
__REG32 WDHP                      : 1;
__REG32 WDLP                      : 1;
__REG32                           :12;
} __srtc_hpienr_bits;

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
__REG32 TFR_CLK_DIS         : 1;
__REG32 RFR_CLK_DIS         : 1;
__REG32 SYNC_TX_FS          : 1;
__REG32                     :19;
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
__REG32                     : 4;
__REG32 TRFC                : 1;
__REG32 RFRC                : 1;
__REG32                     : 7;
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
__REG32 TFRC_EN             : 1;
__REG32 RFRC_EN             : 1;
__REG32                     : 7;
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

/* SSI Transmit and Receive Clock Control Register (STCCR and SRCCR) */
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

/* SSI AC97 Tag Register (SSI.SATAG) */
typedef struct{
__REG32 SATAG               :16;
__REG32                     :16;
} __satag_bits;

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

/* SSI AC97 Channel Status Register (SSI.SACCST) */
typedef struct{
__REG32 SACCST0             : 1;
__REG32 SACCST1             : 1;
__REG32 SACCST2             : 1;
__REG32 SACCST3             : 1;
__REG32 SACCST4             : 1;
__REG32 SACCST5             : 1;
__REG32 SACCST6             : 1;
__REG32 SACCST7             : 1;
__REG32 SACCST8             : 1;
__REG32 SACCST9             : 1;
__REG32                     :22;
} __saccst_bits;

/* SSI AC97 Channel Enable Register (SSI.SACCEN) */
typedef struct{
__REG32 SACCEN0             : 1;
__REG32 SACCEN1             : 1;
__REG32 SACCEN2             : 1;
__REG32 SACCEN3             : 1;
__REG32 SACCEN4             : 1;
__REG32 SACCEN5             : 1;
__REG32 SACCEN6             : 1;
__REG32 SACCEN7             : 1;
__REG32 SACCEN8             : 1;
__REG32 SACCEN9             : 1;
__REG32                     :22;
} __saccen_bits;

/* SSI AC97 Channel Disable Register (SSI.SACCDIS) */
typedef struct{
__REG32 SACCDIS0            : 1;
__REG32 SACCDIS1            : 1;
__REG32 SACCDIS2            : 1;
__REG32 SACCDIS3            : 1;
__REG32 SACCDIS4            : 1;
__REG32 SACCDIS5            : 1;
__REG32 SACCDIS6            : 1;
__REG32 SACCDIS7            : 1;
__REG32 SACCDIS8            : 1;
__REG32 SACCDIS9            : 1;
__REG32                     :22;
} __saccdis_bits;

/* Common Configuration Register (TVE.COM_CONF_REG) */
typedef struct{
__REG32 TVE_EN              : 1;
__REG32 TVDAC_SAMP_RATE     : 2;
__REG32 IPU_CLK_EN          : 1;
__REG32 DATA_SOURCE_SEL     : 2;
__REG32 INP_VIDEO_FORM      : 1;
__REG32 P2I_CONV_EN         : 1;
__REG32 TV_STAND            : 4;
__REG32 TV_OUT_MODE         : 3;
__REG32                     : 1;
__REG32 SD_PED_AMP_CONT     : 2;
__REG32                     : 2;
__REG32 SYNC_CH_0_EN        : 1;
__REG32 SYNC_CH_1_EN        : 1;
__REG32 SYNC_CH_2_EN        : 1;
__REG32                     : 1;
__REG32 ACT_LINE_OFFSET     : 3;
__REG32                     : 5;
} __tve_com_conf_reg_bits;

/* Luma Filter Control Register 0 (TVE.LUMA_FILT_CONT_REG_0) */
typedef struct{
__REG32 DEFLICK_EN          : 1;
__REG32 DEFLICK_MEAS_WIN    : 1;
__REG32                     : 2;
__REG32 DEFLICK_COEF        : 3;
__REG32                     : 1;
__REG32 DEFLICK_LOW_THRESH  : 8;
__REG32 DEFLICK_MID_THRESH  : 8;
__REG32 DEFLICK_HIGH_THRESH : 8;
} __tve_luma_filt_cont_reg_0_bits;

/* Luma Filter Control Register 1 (TVE.LUMA_FILT_CONT_REG_1) */
typedef struct{
__REG32 V_SHARP_EN          : 1;
__REG32                     : 3;
__REG32 V_SHARP_COEF        : 3;
__REG32                     : 1;
__REG32 V_SHARP_LOW_THRESH  : 8;
__REG32                     : 8;
__REG32 V_SHARP_HIGH_THRESH : 8;
} __tve_luma_filt_cont_reg_1_bits;

/* Luma Filter Control Register 2 (TVE.LUMA_FILT_CONT_REG_2) */
typedef struct{
__REG32 H_SHARP_EN          : 1;
__REG32                     : 3;
__REG32 H_SHARP_COEF        : 3;
__REG32                     : 1;
__REG32 H_SHARP_LOW_THRESH  : 8;
__REG32                     : 8;
__REG32 H_SHARP_HIGH_THRESH : 8;
} __tve_luma_filt_cont_reg_2_bits;

/* Luma Filter Control Register 2 (TVE.LUMA_FILT_CONT_REG_2) */
typedef struct{
__REG32 DERING_EN           : 1;
__REG32 SUPP_FILTER_TYPE    : 2;
__REG32                     : 1;
__REG32 DERING_COEF         : 3;
__REG32                     : 1;
__REG32 DERING_LOW_THRESH   : 8;
__REG32 DERING_MID_THRESH   : 8;
__REG32 DERING_HIGH_THRESH  : 8;
} __tve_luma_filt_cont_reg_3_bits;

/* Luma Statistic Analysis Control Register 0 (TVE.LUMA_SA_CONT_REG_0) */
typedef struct{
__REG32 LUMA_SA_EN          : 1;
__REG32                     : 3;
__REG32 SA_H_POINTS_NUM     : 2;
__REG32                     : 2;
__REG32 SA_V_POINTS_NUM     : 2;
__REG32                     :22;
} __tve_luma_sa_cont_reg_0_bits;

/* Luma Statistic Analysis Control Register 1 (TVE.LUMA_SA_CONT_REG_1) */
typedef struct{
__REG32 SA_WIN_WIDTH        : 8;
__REG32 SA_WIN_HEIGHT       : 8;
__REG32 SA_WIN_H_OFFSET     : 8;
__REG32 SA_WIN_V_OFFSET     : 8;
} __tve_luma_sa_cont_reg_1_bits;

/* Luma Statistic Analysis Status Register 0 (TVE.LUMA_SA_STAT_REG_0) */
typedef struct{
__REG32 DEFLICK_MEAS_MEAN   : 8;
__REG32 V_SHARP_MEAS_MEAN   : 8;
__REG32 H_SHARP_MEAS_MEAN   : 8;
__REG32 DERING_MEAS_MEAN    : 8;
} __tve_luma_sa_stat_reg_0_bits;

/* Luma Statistic Analysis Status Register 0 (TVE.LUMA_SA_STAT_REG_0) */
typedef struct{
__REG32 LUMA_MEAN           : 8;
__REG32                     :24;
} __tve_luma_sa_stat_reg_1_bits;

/* Chroma Control Register (TVE.CHROMA_CONT_REG) */
typedef struct{
__REG32 CHROMA_V_FILT_EN    : 1;
__REG32                     : 3;
__REG32 CHROMA_BW           : 3;
__REG32                     : 1;
__REG32 SCH_PHASE           : 8;
__REG32                     :16;
} __tve_chroma_cont_reg_bits;

/* TVDAC 0 Control Register (TVE.TVDAC_0_CONT_REG) */
typedef struct{
__REG32 TVDAC_0_GAIN        : 6;
__REG32                     : 2;
__REG32 TVDAC_0_OFFSET      : 8;
__REG32 BG_RDY_TIME         : 8;
__REG32                     : 8;
} __tve_tvdac_0_cont_reg_bits;

/* TVDAC 1 Control Register (TVE.TVDAC_1_CONT_REG) */
typedef struct{
__REG32 TVDAC_1_GAIN        : 6;
__REG32                     : 2;
__REG32 TVDAC_1_OFFSET      : 8;
__REG32                     :16;
} __tve_tvdac_1_cont_reg_bits;

/* TVDAC 2 Control Register (TVE.TVDAC_2_CONT_REG) */
typedef struct{
__REG32 TVDAC_2_GAIN        : 6;
__REG32                     : 2;
__REG32 TVDAC_2_OFFSET      : 8;
__REG32                     :16;
} __tve_tvdac_2_cont_reg_bits;

/* TVDAC 2 Control Register (TVE.TVDAC_2_CONT_REG) */
typedef struct{
__REG32 CD_EN               : 1;
__REG32 CD_TRIG_MODE        : 1;
__REG32                     : 2;
__REG32 CD_MON_PER          : 4;
__REG32 CD_CH_0_REF_LVL     : 1;
__REG32 CD_CH_1_REF_LVL     : 1;
__REG32 CD_CH_2_REF_LVL     : 1;
__REG32 CD_REF_MODE         : 1;
__REG32                     : 4;
__REG32 CD_CH_0_LM_EN       : 1;
__REG32 CD_CH_1_LM_EN       : 1;
__REG32 CD_CH_2_LM_EN       : 1;
__REG32                     : 1;
__REG32 CD_CH_0_SM_EN       : 1;
__REG32 CD_CH_1_SM_EN       : 1;
__REG32 CD_CH_2_SM_EN       : 1;
__REG32                     : 9;
} __tve_cd_cont_reg_bits;

/* VBI Data Control Register (TVE.VBI_DATA_CONT_REG) */
typedef struct{
__REG32 CC_SD_F1_EN         : 1;
__REG32 CC_SD_F2_EN         : 1;
__REG32 CC_SD_BOOST_EN      : 1;
__REG32                     : 1;
__REG32 CGMS_SD_F1_EN       : 1;
__REG32 CGMS_SD_F2_EN       : 1;
__REG32 CGMS_SD_SW_CRC_EN   : 1;
__REG32 WSS_SD_EN           : 1;
__REG32 CGMS_HD_A_F1_EN     : 1;
__REG32 CGMS_HD_A_F2_EN     : 1;
__REG32 CGMS_HD_A_SW_CRC_EN : 1;
__REG32                     : 1;
__REG32 CGMS_HD_B_F1_EN     : 1;
__REG32 CGMS_HD_B_F2_EN     : 1;
__REG32 CGMS_HD_B_SW_CRC_EN : 1;
__REG32                     : 1;
__REG32 CGMS_HD_B_F1_HEADER : 6;
__REG32                     : 2;
__REG32 CGMS_HD_B_F2_HEADER : 6;
__REG32                     : 2;
} __tve_vbi_data_cont_reg_bits;

/* VBI Data Register 0 (TVE.VBI_DATA_REG_0) */
typedef struct{
__REG32 CGMS_SD_HD_A_F1_DATA:20;
__REG32                     :12;
} __tve_vbi_data_reg_0_bits;

/* VBI Data Register 1 (TVE.VBI_DATA_REG_1) */
typedef struct{
__REG32 CGMS_SD_HD_A_F2_DATA:20;
__REG32                     :12;
} __tve_vbi_data_reg_1_bits;

/* Interrupt Control Register (TVE.INT_CONT_REG) */
typedef struct{
__REG32 CD_LM_IEN             : 1;
__REG32 CD_SM_IEN             : 1;
__REG32 CD_MON_END_IEN        : 1;
__REG32 CC_SD_F1_DONE_IEN     : 1;
__REG32 CC_SD_F2_DONE_IEN     : 1;
__REG32 CGMS_SD_F1_DONE_IEN   : 1;
__REG32 CGMS_SD_F2_DONE_IEN   : 1;
__REG32 WSS_SD_DONE_IEN       : 1;
__REG32 CGMS_HD_A_F1_DONE_IEN : 1;
__REG32 CGMS_HD_A_F2_DONE_IEN : 1;
__REG32 CGMS_HD_B_F1_DONE_IEN : 1;
__REG32 CGMS_HD_B_F2_DONE_IEN : 1;
__REG32 TVE_FIELD_END_IEN     : 1;
__REG32 TVE_FRAME_END_IEN     : 1;
__REG32 SA_MEAS_END_IEN       : 1;
__REG32                       :17;
} __tve_int_cont_reg_bits;

/* Status Register (TVE.STAT_REG) */
typedef struct{
__REG32 CD_LM_INT             : 1;
__REG32 CD_SM_INT             : 1;
__REG32 CD_MON_END_INT        : 1;
__REG32 CC_SD_F1_DONE_INT     : 1;
__REG32 CC_SD_F2_DONE_INT     : 1;
__REG32 CGMS_SD_F1_DONE_INT   : 1;
__REG32 CGMS_SD_F2_DONE_INT   : 1;
__REG32 WSS_SD_DONE_INT       : 1;
__REG32 CGMS_HD_A_F1_DONE_INT : 1;
__REG32 CGMS_HD_A_F2_DONE_INT : 1;
__REG32 CGMS_HD_B_F1_DONE_INT : 1;
__REG32 CGMS_HD_B_F2_DONE_INT : 1;
__REG32 TVE_FIELD_END_INT     : 1;
__REG32 TVE_FRAME_END_INT     : 1;
__REG32 SA_MEAS_END_INT       : 1;
__REG32                       : 1;
__REG32 CD_CH_0_LM_ST         : 1;
__REG32 CD_CH_1_LM_ST         : 1;
__REG32 CD_CH_2_LM_ST         : 1;
__REG32                       : 1;
__REG32 CD_CH_0_SM_ST         : 1;
__REG32 CD_CH_1_SM_ST         : 1;
__REG32 CD_CH_2_SM_ST         : 1;
__REG32                       : 1;
__REG32 CD_MAN_TRIG           : 1;
__REG32 BG_READY              : 1;
__REG32                       : 6;
} __tve_stat_reg_bits;

/* Test Mode Register (TVE.TST_MODE_REG) */
typedef struct{
__REG32 TVDAC_TEST_MODE       : 3;
__REG32                       : 1;
__REG32 TVDAC_0_DATA_FORCE    : 1;
__REG32 TVDAC_1_DATA_FORCE    : 1;
__REG32 TVDAC_2_DATA_FORCE    : 1;
__REG32                       : 1;
__REG32 TVDAC_TEST_SINE_FREQ  : 3;
__REG32                       : 1;
__REG32 TVDAC_TEST_SINE_LEVEL : 2;
__REG32                       : 2;
__REG32 COLORBAR_TYPE         : 1;
__REG32                       :15;
} __tve_tst_mode_reg_bits;

/* User Mode Control Register (TVE.USER_MODE_CONT_REG) */
typedef struct{
__REG32 H_TIMING_USR_MODE_EN        : 1;
__REG32 LUMA_FILT_USR_MODE_EN       : 1;
__REG32 SC_FREQ_USR_MODE_EN         : 1;
__REG32 CSCM_COEF_USR_MODE_EN       : 1;
__REG32 BLANK_LEVEL_USR_MODE_EN     : 1;
__REG32 VBI_DATA_USR_MODE_EN        : 1;
__REG32 TVDAC_DROP_COMP_USR_MODE_EN : 1;
__REG32                             :25;
} __tve_user_mode_cont_reg_bits;

/* SD Timing User Control Register 0 (TVE.SD_TIMING_USR_CONT_REG_0) */
typedef struct{
__REG32 SD_VBI_T0_USR               : 6;
__REG32                             : 2;
__REG32 SD_VBI_T1_USR               :10;
__REG32                             : 2;
__REG32 SD_VBI_T2_USR               :10;
__REG32                             : 2;
} __tve_sd_timing_usr_cont_reg_0_bits;

/* SD Timing User Control Register 1 (TVE.SD_TIMING_USR_CONT_REG_1) */
typedef struct{
__REG32 SD_ACT_T0_USR               : 7;
__REG32                             : 1;
__REG32 SD_ACT_T1_USR               : 5;
__REG32                             : 3;
__REG32 SD_ACT_T2_USR               : 7;
__REG32                             : 1;
__REG32 SD_ACT_T3_USR               : 7;
__REG32                             : 1;
} __tve_sd_timing_usr_cont_reg_1_bits;

/* SD Timing User Control Register 2 (TVE.SD_TIMING_USR_CONT_REG_2) */
typedef struct{
__REG32 SD_ACT_T4_USR               :11;
__REG32                             : 1;
__REG32 SD_ACT_T5_USR               :10;
__REG32                             : 2;
__REG32 SD_ACT_T6_USR               : 6;
__REG32                             : 2;
} __tve_sd_timing_usr_cont_reg_2_bits;

/* HD Timing User Control Register 0 (TVE.HD_TIMING_USR_CONT_REG_0) */
typedef struct{
__REG32 HD_VBI_ACT_T0_USR           : 7;
__REG32                             : 1;
__REG32 HD_VBI_T1_USR               : 9;
__REG32                             : 3;
__REG32 HD_VBI_T2_USR               :11;
__REG32                             : 1;
} __tve_hd_timing_usr_cont_reg_0_bits;

/* HD Timing User Control Register 1 (TVE.HD_TIMING_USR_CONT_REG_1) */
typedef struct{
__REG32 HD_VBI_ACT_T0_USR           :13;
__REG32                             : 3;
__REG32 HD_ACT_T1_USR               : 9;
__REG32                             : 7;
} __tve_hd_timing_usr_cont_reg_1_bits;

/* HD Timing User Control Register 2 (TVE.HD_TIMING_USR_CONT_REG_2)*/
typedef struct{
__REG32 HD_ACT_T2_USR               :12;
__REG32                             : 4;
__REG32 HD_ACT_T3_USR               :13;
__REG32                             : 3;
} __tve_hd_timing_usr_cont_reg_2_bits;

/* Luma User Control Register 0 (TVE.LUMA_USR_CONT_REG_0) */
typedef struct{
__REG32 DEFLICK_MASK_MATRIX_USR     :24;
__REG32                             : 8;
} __tve_luma_usr_cont_reg_0_bits;

/* Luma User Control Register 1 (TVE.LUMA_USR_CONT_REG_1) */
typedef struct{
__REG32 V_SHARP_MASK_MATRIX_USR     :24;
__REG32                             : 8;
} __tve_luma_usr_cont_reg_1_bits;

/* Luma User Control Register 2 (TVE.LUMA_USR_CONT_REG_2) */
typedef struct{
__REG32 H_SHARP_MASK_MATRIX_USR     :24;
__REG32                             : 8;
} __tve_luma_usr_cont_reg_2_bits;

/* Luma User Control Register 3 (TVE.LUMA_USR_CONT_REG_3) */
typedef struct{
__REG32 DERING_MASK_MATRIX_USR      :24;
__REG32                             : 8;
} __tve_luma_usr_cont_reg_3_bits;

/* Color Space Conversion User Control Register 0 (TVE.CSC_USR_CONT_REG_0) */
typedef struct{
__REG32 DATA_CLIP_USR       : 1;
__REG32                     : 7;
__REG32 BRIGHT_CORR_USR     : 6;
__REG32                     : 2;
__REG32 CSCM_A_COEF_USR     :11;
__REG32                     : 5;
} __tve_csc_usr_cont_reg_0_bits;

/* Color Space Conversion User Control Register 1 (TVE.CSC_USR_CONT_REG_1) */
typedef struct{
__REG32 CSCM_B_COEF_USR     :12;
__REG32                     : 4;
__REG32 CSCM_C_COEF_USR     :11;
__REG32                     : 5;
} __tve_csc_usr_cont_reg_1_bits;

/* Color Space Conversion User Control Register 2 (TVE.CSC_USR_CONT_REG_2) */
typedef struct{
__REG32 CSCM_D_COEF_USR     :12;
__REG32                     : 4;
__REG32 CSCM_E_COEF_USR     :11;
__REG32                     : 5;
} __tve_csc_usr_cont_reg_2_bits;

/* Blanking Level User Control Register (TVE.BLANK_USR_CONT_REG) */
typedef struct{
__REG32 BLANKING_CH_0_USR   :10;
__REG32 BLANKING_CH_1_USR   :10;
__REG32 BLANKING_CH_2_USR   :10;
__REG32                     : 2;
} __tve_blank_usr_cont_reg_bits;

/* SD Modulation User Control Register (TVE.SD_MOD_USR_CONT_REG) */
typedef struct{
__REG32 SC_FREQ_USR         :30;
__REG32                     : 2;
} __tve_sd_mod_usr_cont_reg_bits;

/* VBI Data User Control Register 0 (TVE.VBI_DATA_USR_CONT_REG_0) */
typedef struct{
__REG32 VBI_DATA_START_TIME_USR   :12;
__REG32                           : 4;
__REG32 VBI_DATA_STOP_TIME_USR    :12;
__REG32                           : 4;
} __tve_vbi_data_usr_cont_reg_0_bits;

/* VBI Data User Control Register 1 (TVE.VBI_DATA_USR_CONT_REG_1) */
typedef struct{
__REG32 VBI_PACKET_START_TIME_USR :12;
__REG32                           :20;
} __tve_vbi_data_usr_cont_reg_1_bits;

/* VBI Data User Control Register 2 (TVE.VBI_DATA_USR_CONT_REG_2) */
typedef struct{
__REG32 CC_SD_RUNIN_START_TIME_USR  :12;
__REG32                             : 4;
__REG32 CC_SD_RUNIN_DIV_NUM_USR     :12;
__REG32                             : 4;
} __tve_vbi_data_usr_cont_reg_2_bits;

/* VBI Data User Control Register 3 (TVE.VBI_DATA_USR_CONT_REG_3) */
typedef struct{
__REG32 CC_SD_CGMS_HD_B_DIV_NUM_USR   : 7;
__REG32                               : 9;
__REG32 CC_SD_CGMS_HD_B_DIV_DENOM_USR :13;
__REG32                               : 3;
} __tve_vbi_data_usr_cont_reg_3_bits;

/* VBI Data User Control Register 4 (TVE.VBI_DATA_USR_CONT_REG_4) */
typedef struct{
__REG32 WSS_CGMS_SD_CGMS_HD_A_DIV_NUM_USR   : 7;
__REG32                                     : 9;
__REG32 WSS_CGMS_SD_CGMS_HD_A_DIV_DENOM_USR :13;
__REG32                                     : 3;
} __tve_vbi_data_usr_cont_reg_4_bits;

/* Drop Compensation User Control Register (TVE.DROP_COMP_USR_CONT_REG) */
typedef struct{
__REG32 TVDAC_0_DROP_COMP : 4;
__REG32 TVDAC_1_DROP_COMP : 4;
__REG32 TVDAC_2_DROP_COMP : 4;
__REG32                   :20;
} __tve_drop_comp_usr_cont_reg_bits;

/* Control Register */
typedef struct{
__REG32 EN                : 1;
__REG32                   :15;
__REG32 NSEN              : 1;
__REG32                   :14;
__REG32 NSENMASK          : 1;
} __tzic_intctrl_bits;

/* Interrupt Controller Type Register */
typedef struct{
__REG32 ITLINES           : 5;
__REG32 CPUS              : 3;
__REG32                   : 2;
__REG32 DOM               : 1;
__REG32                   :21;
} __tzic_inttype_bits;

/* Priority Mask Register */
typedef struct{
__REG32 MASK              : 8;
__REG32                   :24;
} __tzic_priomask_bits;

/* Synchronizer Control */
typedef struct{
__REG32 SYNCMODE          : 2;
__REG32                   :30;
} __tzic_syncctrl_bits;

/* DSM Interrupt Holdoff Register */
typedef struct{
__REG32 DSM               : 1;
__REG32                   :31;
} __tzic_dsmint_bits;

/* Interrupt Security Register 0 */
typedef struct{
__REG32 SECURE0           : 1;
__REG32 SECURE1           : 1;
__REG32 SECURE2           : 1;
__REG32 SECURE3           : 1;
__REG32 SECURE4           : 1;
__REG32 SECURE5           : 1;
__REG32 SECURE6           : 1;
__REG32 SECURE7           : 1;
__REG32 SECURE8           : 1;
__REG32 SECURE9           : 1;
__REG32 SECURE10          : 1;
__REG32 SECURE11          : 1;
__REG32 SECURE12          : 1;
__REG32 SECURE13          : 1;
__REG32 SECURE14          : 1;
__REG32 SECURE15          : 1;
__REG32 SECURE16          : 1;
__REG32 SECURE17          : 1;
__REG32 SECURE18          : 1;
__REG32 SECURE19          : 1;
__REG32 SECURE20          : 1;
__REG32 SECURE21          : 1;
__REG32 SECURE22          : 1;
__REG32 SECURE23          : 1;
__REG32 SECURE24          : 1;
__REG32 SECURE25          : 1;
__REG32 SECURE26          : 1;
__REG32 SECURE27          : 1;
__REG32 SECURE28          : 1;
__REG32 SECURE29          : 1;
__REG32 SECURE30          : 1;
__REG32 SECURE31          : 1;
} __tzic_intsec0_bits;

/* Interrupt Security Register 1 */
typedef struct{
__REG32 SECURE32          : 1;
__REG32 SECURE33          : 1;
__REG32 SECURE34          : 1;
__REG32 SECURE35          : 1;
__REG32 SECURE36          : 1;
__REG32 SECURE37          : 1;
__REG32 SECURE38          : 1;
__REG32 SECURE39          : 1;
__REG32 SECURE40          : 1;
__REG32 SECURE41          : 1;
__REG32 SECURE42          : 1;
__REG32 SECURE43          : 1;
__REG32 SECURE44          : 1;
__REG32 SECURE45          : 1;
__REG32 SECURE46          : 1;
__REG32 SECURE47          : 1;
__REG32 SECURE48          : 1;
__REG32 SECURE49          : 1;
__REG32 SECURE50          : 1;
__REG32 SECURE51          : 1;
__REG32 SECURE52          : 1;
__REG32 SECURE53          : 1;
__REG32 SECURE54          : 1;
__REG32 SECURE55          : 1;
__REG32 SECURE56          : 1;
__REG32 SECURE57          : 1;
__REG32 SECURE58          : 1;
__REG32 SECURE59          : 1;
__REG32 SECURE60          : 1;
__REG32 SECURE61          : 1;
__REG32 SECURE62          : 1;
__REG32 SECURE63          : 1;
} __tzic_intsec1_bits;

/* Interrupt Security Register 2 */
typedef struct{
__REG32 SECURE64          : 1;
__REG32 SECURE65          : 1;
__REG32 SECURE66          : 1;
__REG32 SECURE67          : 1;
__REG32 SECURE68          : 1;
__REG32 SECURE69          : 1;
__REG32 SECURE70          : 1;
__REG32 SECURE71          : 1;
__REG32 SECURE72          : 1;
__REG32 SECURE73          : 1;
__REG32 SECURE74          : 1;
__REG32 SECURE75          : 1;
__REG32 SECURE76          : 1;
__REG32 SECURE77          : 1;
__REG32 SECURE78          : 1;
__REG32 SECURE79          : 1;
__REG32 SECURE80          : 1;
__REG32 SECURE81          : 1;
__REG32 SECURE82          : 1;
__REG32 SECURE83          : 1;
__REG32 SECURE84          : 1;
__REG32 SECURE85          : 1;
__REG32 SECURE86          : 1;
__REG32 SECURE87          : 1;
__REG32 SECURE88          : 1;
__REG32 SECURE89          : 1;
__REG32 SECURE90          : 1;
__REG32 SECURE91          : 1;
__REG32 SECURE92          : 1;
__REG32 SECURE93          : 1;
__REG32 SECURE94          : 1;
__REG32 SECURE95          : 1;
} __tzic_intsec2_bits;

/* Interrupt Security Register 3 */
typedef struct{
__REG32 SECURE96          : 1;
__REG32 SECURE97          : 1;
__REG32 SECURE98          : 1;
__REG32 SECURE99          : 1;
__REG32 SECURE100         : 1;
__REG32 SECURE101         : 1;
__REG32 SECURE102         : 1;
__REG32 SECURE103         : 1;
__REG32 SECURE104         : 1;
__REG32 SECURE105         : 1;
__REG32 SECURE106         : 1;
__REG32 SECURE107         : 1;
__REG32 SECURE108         : 1;
__REG32 SECURE109         : 1;
__REG32 SECURE110         : 1;
__REG32 SECURE111         : 1;
__REG32 SECURE112         : 1;
__REG32 SECURE113         : 1;
__REG32 SECURE114         : 1;
__REG32 SECURE115         : 1;
__REG32 SECURE116         : 1;
__REG32 SECURE117         : 1;
__REG32 SECURE118         : 1;
__REG32 SECURE119         : 1;
__REG32 SECURE120         : 1;
__REG32 SECURE121         : 1;
__REG32 SECURE122         : 1;
__REG32 SECURE123         : 1;
__REG32 SECURE124         : 1;
__REG32 SECURE125         : 1;
__REG32 SECURE126         : 1;
__REG32 SECURE127         : 1;
} __tzic_intsec3_bits;

/* Enable Set Register 0 */
typedef struct{
__REG32 INTENSET0           : 1;
__REG32 INTENSET1           : 1;
__REG32 INTENSET2           : 1;
__REG32 INTENSET3           : 1;
__REG32 INTENSET4           : 1;
__REG32 INTENSET5           : 1;
__REG32 INTENSET6           : 1;
__REG32 INTENSET7           : 1;
__REG32 INTENSET8           : 1;
__REG32 INTENSET9           : 1;
__REG32 INTENSET10          : 1;
__REG32 INTENSET11          : 1;
__REG32 INTENSET12          : 1;
__REG32 INTENSET13          : 1;
__REG32 INTENSET14          : 1;
__REG32 INTENSET15          : 1;
__REG32 INTENSET16          : 1;
__REG32 INTENSET17          : 1;
__REG32 INTENSET18          : 1;
__REG32 INTENSET19          : 1;
__REG32 INTENSET20          : 1;
__REG32 INTENSET21          : 1;
__REG32 INTENSET22          : 1;
__REG32 INTENSET23          : 1;
__REG32 INTENSET24          : 1;
__REG32 INTENSET25          : 1;
__REG32 INTENSET26          : 1;
__REG32 INTENSET27          : 1;
__REG32 INTENSET28          : 1;
__REG32 INTENSET29          : 1;
__REG32 INTENSET30          : 1;
__REG32 INTENSET31          : 1;
} __tzic_enset0_bits;

/* Enable Set Register 1 */
typedef struct{
__REG32 INTENSET32          : 1;
__REG32 INTENSET33          : 1;
__REG32 INTENSET34          : 1;
__REG32 INTENSET35          : 1;
__REG32 INTENSET36          : 1;
__REG32 INTENSET37          : 1;
__REG32 INTENSET38          : 1;
__REG32 INTENSET39          : 1;
__REG32 INTENSET40          : 1;
__REG32 INTENSET41          : 1;
__REG32 INTENSET42          : 1;
__REG32 INTENSET43          : 1;
__REG32 INTENSET44          : 1;
__REG32 INTENSET45          : 1;
__REG32 INTENSET46          : 1;
__REG32 INTENSET47          : 1;
__REG32 INTENSET48          : 1;
__REG32 INTENSET49          : 1;
__REG32 INTENSET50          : 1;
__REG32 INTENSET51          : 1;
__REG32 INTENSET52          : 1;
__REG32 INTENSET53          : 1;
__REG32 INTENSET54          : 1;
__REG32 INTENSET55          : 1;
__REG32 INTENSET56          : 1;
__REG32 INTENSET57          : 1;
__REG32 INTENSET58          : 1;
__REG32 INTENSET59          : 1;
__REG32 INTENSET60          : 1;
__REG32 INTENSET61          : 1;
__REG32 INTENSET62          : 1;
__REG32 INTENSET63          : 1;
} __tzic_enset1_bits;

/* Enable Set Register 2 */
typedef struct{
__REG32 INTENSET64          : 1;
__REG32 INTENSET65          : 1;
__REG32 INTENSET66          : 1;
__REG32 INTENSET67          : 1;
__REG32 INTENSET68          : 1;
__REG32 INTENSET69          : 1;
__REG32 INTENSET70          : 1;
__REG32 INTENSET71          : 1;
__REG32 INTENSET72          : 1;
__REG32 INTENSET73          : 1;
__REG32 INTENSET74          : 1;
__REG32 INTENSET75          : 1;
__REG32 INTENSET76          : 1;
__REG32 INTENSET77          : 1;
__REG32 INTENSET78          : 1;
__REG32 INTENSET79          : 1;
__REG32 INTENSET80          : 1;
__REG32 INTENSET81          : 1;
__REG32 INTENSET82          : 1;
__REG32 INTENSET83          : 1;
__REG32 INTENSET84          : 1;
__REG32 INTENSET85          : 1;
__REG32 INTENSET86          : 1;
__REG32 INTENSET87          : 1;
__REG32 INTENSET88          : 1;
__REG32 INTENSET89          : 1;
__REG32 INTENSET90          : 1;
__REG32 INTENSET91          : 1;
__REG32 INTENSET92          : 1;
__REG32 INTENSET93          : 1;
__REG32 INTENSET94          : 1;
__REG32 INTENSET95          : 1;
} __tzic_enset2_bits;

/* Enable Set Register 3 */
typedef struct{
__REG32 INTENSET96          : 1;
__REG32 INTENSET97          : 1;
__REG32 INTENSET98          : 1;
__REG32 INTENSET99          : 1;
__REG32 INTENSET100         : 1;
__REG32 INTENSET101         : 1;
__REG32 INTENSET102         : 1;
__REG32 INTENSET103         : 1;
__REG32 INTENSET104         : 1;
__REG32 INTENSET105         : 1;
__REG32 INTENSET106         : 1;
__REG32 INTENSET107         : 1;
__REG32 INTENSET108         : 1;
__REG32 INTENSET109         : 1;
__REG32 INTENSET110         : 1;
__REG32 INTENSET111         : 1;
__REG32 INTENSET112         : 1;
__REG32 INTENSET113         : 1;
__REG32 INTENSET114         : 1;
__REG32 INTENSET115         : 1;
__REG32 INTENSET116         : 1;
__REG32 INTENSET117         : 1;
__REG32 INTENSET118         : 1;
__REG32 INTENSET119         : 1;
__REG32 INTENSET120         : 1;
__REG32 INTENSET121         : 1;
__REG32 INTENSET122         : 1;
__REG32 INTENSET123         : 1;
__REG32 INTENSET124         : 1;
__REG32 INTENSET125         : 1;
__REG32 INTENSET126         : 1;
__REG32 INTENSET127         : 1;
} __tzic_enset3_bits;

/* Enable Clear Register 0 */
typedef struct{
__REG32 INTENCLEAR0           : 1;
__REG32 INTENCLEAR1           : 1;
__REG32 INTENCLEAR2           : 1;
__REG32 INTENCLEAR3           : 1;
__REG32 INTENCLEAR4           : 1;
__REG32 INTENCLEAR5           : 1;
__REG32 INTENCLEAR6           : 1;
__REG32 INTENCLEAR7           : 1;
__REG32 INTENCLEAR8           : 1;
__REG32 INTENCLEAR9           : 1;
__REG32 INTENCLEAR10          : 1;
__REG32 INTENCLEAR11          : 1;
__REG32 INTENCLEAR12          : 1;
__REG32 INTENCLEAR13          : 1;
__REG32 INTENCLEAR14          : 1;
__REG32 INTENCLEAR15          : 1;
__REG32 INTENCLEAR16          : 1;
__REG32 INTENCLEAR17          : 1;
__REG32 INTENCLEAR18          : 1;
__REG32 INTENCLEAR19          : 1;
__REG32 INTENCLEAR20          : 1;
__REG32 INTENCLEAR21          : 1;
__REG32 INTENCLEAR22          : 1;
__REG32 INTENCLEAR23          : 1;
__REG32 INTENCLEAR24          : 1;
__REG32 INTENCLEAR25          : 1;
__REG32 INTENCLEAR26          : 1;
__REG32 INTENCLEAR27          : 1;
__REG32 INTENCLEAR28          : 1;
__REG32 INTENCLEAR29          : 1;
__REG32 INTENCLEAR30          : 1;
__REG32 INTENCLEAR31          : 1;
} __tzic_enclear0_bits;

/* Enable Clear Register 1 */
typedef struct{
__REG32 INTENCLEAR32          : 1;
__REG32 INTENCLEAR33          : 1;
__REG32 INTENCLEAR34          : 1;
__REG32 INTENCLEAR35          : 1;
__REG32 INTENCLEAR36          : 1;
__REG32 INTENCLEAR37          : 1;
__REG32 INTENCLEAR38          : 1;
__REG32 INTENCLEAR39          : 1;
__REG32 INTENCLEAR40          : 1;
__REG32 INTENCLEAR41          : 1;
__REG32 INTENCLEAR42          : 1;
__REG32 INTENCLEAR43          : 1;
__REG32 INTENCLEAR44          : 1;
__REG32 INTENCLEAR45          : 1;
__REG32 INTENCLEAR46          : 1;
__REG32 INTENCLEAR47          : 1;
__REG32 INTENCLEAR48          : 1;
__REG32 INTENCLEAR49          : 1;
__REG32 INTENCLEAR50          : 1;
__REG32 INTENCLEAR51          : 1;
__REG32 INTENCLEAR52          : 1;
__REG32 INTENCLEAR53          : 1;
__REG32 INTENCLEAR54          : 1;
__REG32 INTENCLEAR55          : 1;
__REG32 INTENCLEAR56          : 1;
__REG32 INTENCLEAR57          : 1;
__REG32 INTENCLEAR58          : 1;
__REG32 INTENCLEAR59          : 1;
__REG32 INTENCLEAR60          : 1;
__REG32 INTENCLEAR61          : 1;
__REG32 INTENCLEAR62          : 1;
__REG32 INTENCLEAR63          : 1;
} __tzic_enclear1_bits;

/* Enable Clear Register 2 */
typedef struct{
__REG32 INTENCLEAR64          : 1;
__REG32 INTENCLEAR65          : 1;
__REG32 INTENCLEAR66          : 1;
__REG32 INTENCLEAR67          : 1;
__REG32 INTENCLEAR68          : 1;
__REG32 INTENCLEAR69          : 1;
__REG32 INTENCLEAR70          : 1;
__REG32 INTENCLEAR71          : 1;
__REG32 INTENCLEAR72          : 1;
__REG32 INTENCLEAR73          : 1;
__REG32 INTENCLEAR74          : 1;
__REG32 INTENCLEAR75          : 1;
__REG32 INTENCLEAR76          : 1;
__REG32 INTENCLEAR77          : 1;
__REG32 INTENCLEAR78          : 1;
__REG32 INTENCLEAR79          : 1;
__REG32 INTENCLEAR80          : 1;
__REG32 INTENCLEAR81          : 1;
__REG32 INTENCLEAR82          : 1;
__REG32 INTENCLEAR83          : 1;
__REG32 INTENCLEAR84          : 1;
__REG32 INTENCLEAR85          : 1;
__REG32 INTENCLEAR86          : 1;
__REG32 INTENCLEAR87          : 1;
__REG32 INTENCLEAR88          : 1;
__REG32 INTENCLEAR89          : 1;
__REG32 INTENCLEAR90          : 1;
__REG32 INTENCLEAR91          : 1;
__REG32 INTENCLEAR92          : 1;
__REG32 INTENCLEAR93          : 1;
__REG32 INTENCLEAR94          : 1;
__REG32 INTENCLEAR95          : 1;
} __tzic_enclear2_bits;

/* Enable Clear Register 3 */
typedef struct{
__REG32 INTENCLEAR96          : 1;
__REG32 INTENCLEAR97          : 1;
__REG32 INTENCLEAR98          : 1;
__REG32 INTENCLEAR99          : 1;
__REG32 INTENCLEAR100         : 1;
__REG32 INTENCLEAR101         : 1;
__REG32 INTENCLEAR102         : 1;
__REG32 INTENCLEAR103         : 1;
__REG32 INTENCLEAR104         : 1;
__REG32 INTENCLEAR105         : 1;
__REG32 INTENCLEAR106         : 1;
__REG32 INTENCLEAR107         : 1;
__REG32 INTENCLEAR108         : 1;
__REG32 INTENCLEAR109         : 1;
__REG32 INTENCLEAR110         : 1;
__REG32 INTENCLEAR111         : 1;
__REG32 INTENCLEAR112         : 1;
__REG32 INTENCLEAR113         : 1;
__REG32 INTENCLEAR114         : 1;
__REG32 INTENCLEAR115         : 1;
__REG32 INTENCLEAR116         : 1;
__REG32 INTENCLEAR117         : 1;
__REG32 INTENCLEAR118         : 1;
__REG32 INTENCLEAR119         : 1;
__REG32 INTENCLEAR120         : 1;
__REG32 INTENCLEAR121         : 1;
__REG32 INTENCLEAR122         : 1;
__REG32 INTENCLEAR123         : 1;
__REG32 INTENCLEAR124         : 1;
__REG32 INTENCLEAR125         : 1;
__REG32 INTENCLEAR126         : 1;
__REG32 INTENCLEAR127         : 1;
} __tzic_enclear3_bits;

/* Source Set Register 0 */
typedef struct{
__REG32 SRCSET0           : 1;
__REG32 SRCSET1           : 1;
__REG32 SRCSET2           : 1;
__REG32 SRCSET3           : 1;
__REG32 SRCSET4           : 1;
__REG32 SRCSET5           : 1;
__REG32 SRCSET6           : 1;
__REG32 SRCSET7           : 1;
__REG32 SRCSET8           : 1;
__REG32 SRCSET9           : 1;
__REG32 SRCSET10          : 1;
__REG32 SRCSET11          : 1;
__REG32 SRCSET12          : 1;
__REG32 SRCSET13          : 1;
__REG32 SRCSET14          : 1;
__REG32 SRCSET15          : 1;
__REG32 SRCSET16          : 1;
__REG32 SRCSET17          : 1;
__REG32 SRCSET18          : 1;
__REG32 SRCSET19          : 1;
__REG32 SRCSET20          : 1;
__REG32 SRCSET21          : 1;
__REG32 SRCSET22          : 1;
__REG32 SRCSET23          : 1;
__REG32 SRCSET24          : 1;
__REG32 SRCSET25          : 1;
__REG32 SRCSET26          : 1;
__REG32 SRCSET27          : 1;
__REG32 SRCSET28          : 1;
__REG32 SRCSET29          : 1;
__REG32 SRCSET30          : 1;
__REG32 SRCSET31          : 1;
} __tzic_srcset0_bits;

/* Source Set Register 1 */
typedef struct{
__REG32 SRCSET32          : 1;
__REG32 SRCSET33          : 1;
__REG32 SRCSET34          : 1;
__REG32 SRCSET35          : 1;
__REG32 SRCSET36          : 1;
__REG32 SRCSET37          : 1;
__REG32 SRCSET38          : 1;
__REG32 SRCSET39          : 1;
__REG32 SRCSET40          : 1;
__REG32 SRCSET41          : 1;
__REG32 SRCSET42          : 1;
__REG32 SRCSET43          : 1;
__REG32 SRCSET44          : 1;
__REG32 SRCSET45          : 1;
__REG32 SRCSET46          : 1;
__REG32 SRCSET47          : 1;
__REG32 SRCSET48          : 1;
__REG32 SRCSET49          : 1;
__REG32 SRCSET50          : 1;
__REG32 SRCSET51          : 1;
__REG32 SRCSET52          : 1;
__REG32 SRCSET53          : 1;
__REG32 SRCSET54          : 1;
__REG32 SRCSET55          : 1;
__REG32 SRCSET56          : 1;
__REG32 SRCSET57          : 1;
__REG32 SRCSET58          : 1;
__REG32 SRCSET59          : 1;
__REG32 SRCSET60          : 1;
__REG32 SRCSET61          : 1;
__REG32 SRCSET62          : 1;
__REG32 SRCSET63          : 1;
} __tzic_srcset1_bits;

/* Source Set Register 2 */
typedef struct{
__REG32 SRCSET64          : 1;
__REG32 SRCSET65          : 1;
__REG32 SRCSET66          : 1;
__REG32 SRCSET67          : 1;
__REG32 SRCSET68          : 1;
__REG32 SRCSET69          : 1;
__REG32 SRCSET70          : 1;
__REG32 SRCSET71          : 1;
__REG32 SRCSET72          : 1;
__REG32 SRCSET73          : 1;
__REG32 SRCSET74          : 1;
__REG32 SRCSET75          : 1;
__REG32 SRCSET76          : 1;
__REG32 SRCSET77          : 1;
__REG32 SRCSET78          : 1;
__REG32 SRCSET79          : 1;
__REG32 SRCSET80          : 1;
__REG32 SRCSET81          : 1;
__REG32 SRCSET82          : 1;
__REG32 SRCSET83          : 1;
__REG32 SRCSET84          : 1;
__REG32 SRCSET85          : 1;
__REG32 SRCSET86          : 1;
__REG32 SRCSET87          : 1;
__REG32 SRCSET88          : 1;
__REG32 SRCSET89          : 1;
__REG32 SRCSET90          : 1;
__REG32 SRCSET91          : 1;
__REG32 SRCSET92          : 1;
__REG32 SRCSET93          : 1;
__REG32 SRCSET94          : 1;
__REG32 SRCSET95          : 1;
} __tzic_srcset2_bits;

/* Source Set Register 3 */
typedef struct{
__REG32 SRCSET96          : 1;
__REG32 SRCSET97          : 1;
__REG32 SRCSET98          : 1;
__REG32 SRCSET99          : 1;
__REG32 SRCSET100         : 1;
__REG32 SRCSET101         : 1;
__REG32 SRCSET102         : 1;
__REG32 SRCSET103         : 1;
__REG32 SRCSET104         : 1;
__REG32 SRCSET105         : 1;
__REG32 SRCSET106         : 1;
__REG32 SRCSET107         : 1;
__REG32 SRCSET108         : 1;
__REG32 SRCSET109         : 1;
__REG32 SRCSET110         : 1;
__REG32 SRCSET111         : 1;
__REG32 SRCSET112         : 1;
__REG32 SRCSET113         : 1;
__REG32 SRCSET114         : 1;
__REG32 SRCSET115         : 1;
__REG32 SRCSET116         : 1;
__REG32 SRCSET117         : 1;
__REG32 SRCSET118         : 1;
__REG32 SRCSET119         : 1;
__REG32 SRCSET120         : 1;
__REG32 SRCSET121         : 1;
__REG32 SRCSET122         : 1;
__REG32 SRCSET123         : 1;
__REG32 SRCSET124         : 1;
__REG32 SRCSET125         : 1;
__REG32 SRCSET126         : 1;
__REG32 SRCSET127         : 1;
} __tzic_srcset3_bits;

/* Source Clear Register 0 */
typedef struct{
__REG32 SRCCLEAR0           : 1;
__REG32 SRCCLEAR1           : 1;
__REG32 SRCCLEAR2           : 1;
__REG32 SRCCLEAR3           : 1;
__REG32 SRCCLEAR4           : 1;
__REG32 SRCCLEAR5           : 1;
__REG32 SRCCLEAR6           : 1;
__REG32 SRCCLEAR7           : 1;
__REG32 SRCCLEAR8           : 1;
__REG32 SRCCLEAR9           : 1;
__REG32 SRCCLEAR10          : 1;
__REG32 SRCCLEAR11          : 1;
__REG32 SRCCLEAR12          : 1;
__REG32 SRCCLEAR13          : 1;
__REG32 SRCCLEAR14          : 1;
__REG32 SRCCLEAR15          : 1;
__REG32 SRCCLEAR16          : 1;
__REG32 SRCCLEAR17          : 1;
__REG32 SRCCLEAR18          : 1;
__REG32 SRCCLEAR19          : 1;
__REG32 SRCCLEAR20          : 1;
__REG32 SRCCLEAR21          : 1;
__REG32 SRCCLEAR22          : 1;
__REG32 SRCCLEAR23          : 1;
__REG32 SRCCLEAR24          : 1;
__REG32 SRCCLEAR25          : 1;
__REG32 SRCCLEAR26          : 1;
__REG32 SRCCLEAR27          : 1;
__REG32 SRCCLEAR28          : 1;
__REG32 SRCCLEAR29          : 1;
__REG32 SRCCLEAR30          : 1;
__REG32 SRCCLEAR31          : 1;
} __tzic_srcclear0_bits;

/* Source Clear Register 1 */
typedef struct{
__REG32 SRCCLEAR32          : 1;
__REG32 SRCCLEAR33          : 1;
__REG32 SRCCLEAR34          : 1;
__REG32 SRCCLEAR35          : 1;
__REG32 SRCCLEAR36          : 1;
__REG32 SRCCLEAR37          : 1;
__REG32 SRCCLEAR38          : 1;
__REG32 SRCCLEAR39          : 1;
__REG32 SRCCLEAR40          : 1;
__REG32 SRCCLEAR41          : 1;
__REG32 SRCCLEAR42          : 1;
__REG32 SRCCLEAR43          : 1;
__REG32 SRCCLEAR44          : 1;
__REG32 SRCCLEAR45          : 1;
__REG32 SRCCLEAR46          : 1;
__REG32 SRCCLEAR47          : 1;
__REG32 SRCCLEAR48          : 1;
__REG32 SRCCLEAR49          : 1;
__REG32 SRCCLEAR50          : 1;
__REG32 SRCCLEAR51          : 1;
__REG32 SRCCLEAR52          : 1;
__REG32 SRCCLEAR53          : 1;
__REG32 SRCCLEAR54          : 1;
__REG32 SRCCLEAR55          : 1;
__REG32 SRCCLEAR56          : 1;
__REG32 SRCCLEAR57          : 1;
__REG32 SRCCLEAR58          : 1;
__REG32 SRCCLEAR59          : 1;
__REG32 SRCCLEAR60          : 1;
__REG32 SRCCLEAR61          : 1;
__REG32 SRCCLEAR62          : 1;
__REG32 SRCCLEAR63          : 1;
} __tzic_srcclear1_bits;

/* Source Clear Register 2 */
typedef struct{
__REG32 SRCCLEAR64          : 1;
__REG32 SRCCLEAR65          : 1;
__REG32 SRCCLEAR66          : 1;
__REG32 SRCCLEAR67          : 1;
__REG32 SRCCLEAR68          : 1;
__REG32 SRCCLEAR69          : 1;
__REG32 SRCCLEAR70          : 1;
__REG32 SRCCLEAR71          : 1;
__REG32 SRCCLEAR72          : 1;
__REG32 SRCCLEAR73          : 1;
__REG32 SRCCLEAR74          : 1;
__REG32 SRCCLEAR75          : 1;
__REG32 SRCCLEAR76          : 1;
__REG32 SRCCLEAR77          : 1;
__REG32 SRCCLEAR78          : 1;
__REG32 SRCCLEAR79          : 1;
__REG32 SRCCLEAR80          : 1;
__REG32 SRCCLEAR81          : 1;
__REG32 SRCCLEAR82          : 1;
__REG32 SRCCLEAR83          : 1;
__REG32 SRCCLEAR84          : 1;
__REG32 SRCCLEAR85          : 1;
__REG32 SRCCLEAR86          : 1;
__REG32 SRCCLEAR87          : 1;
__REG32 SRCCLEAR88          : 1;
__REG32 SRCCLEAR89          : 1;
__REG32 SRCCLEAR90          : 1;
__REG32 SRCCLEAR91          : 1;
__REG32 SRCCLEAR92          : 1;
__REG32 SRCCLEAR93          : 1;
__REG32 SRCCLEAR94          : 1;
__REG32 SRCCLEAR95          : 1;
} __tzic_srcclear2_bits;

/* Source Clear Register 3 */
typedef struct{
__REG32 SRCCLEAR96          : 1;
__REG32 SRCCLEAR97          : 1;
__REG32 SRCCLEAR98          : 1;
__REG32 SRCCLEAR99          : 1;
__REG32 SRCCLEAR100         : 1;
__REG32 SRCCLEAR101         : 1;
__REG32 SRCCLEAR102         : 1;
__REG32 SRCCLEAR103         : 1;
__REG32 SRCCLEAR104         : 1;
__REG32 SRCCLEAR105         : 1;
__REG32 SRCCLEAR106         : 1;
__REG32 SRCCLEAR107         : 1;
__REG32 SRCCLEAR108         : 1;
__REG32 SRCCLEAR109         : 1;
__REG32 SRCCLEAR110         : 1;
__REG32 SRCCLEAR111         : 1;
__REG32 SRCCLEAR112         : 1;
__REG32 SRCCLEAR113         : 1;
__REG32 SRCCLEAR114         : 1;
__REG32 SRCCLEAR115         : 1;
__REG32 SRCCLEAR116         : 1;
__REG32 SRCCLEAR117         : 1;
__REG32 SRCCLEAR118         : 1;
__REG32 SRCCLEAR119         : 1;
__REG32 SRCCLEAR120         : 1;
__REG32 SRCCLEAR121         : 1;
__REG32 SRCCLEAR122         : 1;
__REG32 SRCCLEAR123         : 1;
__REG32 SRCCLEAR124         : 1;
__REG32 SRCCLEAR125         : 1;
__REG32 SRCCLEAR126         : 1;
__REG32 SRCCLEAR127         : 1;
} __tzic_srcclear3_bits;

/* Priority Register 0 */
typedef struct{
__REG32 PRIO0               : 8;
__REG32 PRIO1               : 8;
__REG32 PRIO2               : 8;
__REG32 PRIO3               : 8;
} __tzic_priority0_bits;

/* Priority Register 1 */
typedef struct{
__REG32 PRIO4               : 8;
__REG32 PRIO5               : 8;
__REG32 PRIO6               : 8;
__REG32 PRIO7               : 8;
} __tzic_priority1_bits;

/* Priority Register 2 */
typedef struct{
__REG32 PRIO8               : 8;
__REG32 PRIO9               : 8;
__REG32 PRIO10              : 8;
__REG32 PRIO11              : 8;
} __tzic_priority2_bits;

/* Priority Register 3 */
typedef struct{
__REG32 PRIO12              : 8;
__REG32 PRIO13              : 8;
__REG32 PRIO14              : 8;
__REG32 PRIO15              : 8;
} __tzic_priority3_bits;

/* Priority Register 4 */
typedef struct{
__REG32 PRIO16              : 8;
__REG32 PRIO17              : 8;
__REG32 PRIO18              : 8;
__REG32 PRIO19              : 8;
} __tzic_priority4_bits;

/* Priority Register 5 */
typedef struct{
__REG32 PRIO20              : 8;
__REG32 PRIO21              : 8;
__REG32 PRIO22              : 8;
__REG32 PRIO23              : 8;
} __tzic_priority5_bits;

/* Priority Register 6 */
typedef struct{
__REG32 PRIO24              : 8;
__REG32 PRIO25              : 8;
__REG32 PRIO26              : 8;
__REG32 PRIO27              : 8;
} __tzic_priority6_bits;

/* Priority Register 7 */
typedef struct{
__REG32 PRIO28              : 8;
__REG32 PRIO29              : 8;
__REG32 PRIO30              : 8;
__REG32 PRIO31              : 8;
} __tzic_priority7_bits;

/* Priority Register 8 */
typedef struct{
__REG32 PRIO32              : 8;
__REG32 PRIO33              : 8;
__REG32 PRIO34              : 8;
__REG32 PRIO35              : 8;
} __tzic_priority8_bits;

/* Priority Register 9 */
typedef struct{
__REG32 PRIO36              : 8;
__REG32 PRIO37              : 8;
__REG32 PRIO38              : 8;
__REG32 PRIO39              : 8;
} __tzic_priority9_bits;

/* Priority Register 10 */
typedef struct{
__REG32 PRIO40              : 8;
__REG32 PRIO41              : 8;
__REG32 PRIO42              : 8;
__REG32 PRIO43              : 8;
} __tzic_priority10_bits;

/* Priority Register 11 */
typedef struct{
__REG32 PRIO44              : 8;
__REG32 PRIO45              : 8;
__REG32 PRIO46              : 8;
__REG32 PRIO47              : 8;
} __tzic_priority11_bits;

/* Priority Register 12 */
typedef struct{
__REG32 PRIO48              : 8;
__REG32 PRIO49              : 8;
__REG32 PRIO50              : 8;
__REG32 PRIO51              : 8;
} __tzic_priority12_bits;

/* Priority Register 13 */
typedef struct{
__REG32 PRIO52              : 8;
__REG32 PRIO53              : 8;
__REG32 PRIO54              : 8;
__REG32 PRIO55              : 8;
} __tzic_priority13_bits;

/* Priority Register 14 */
typedef struct{
__REG32 PRIO56              : 8;
__REG32 PRIO57              : 8;
__REG32 PRIO58              : 8;
__REG32 PRIO59              : 8;
} __tzic_priority14_bits;

/* Priority Register 15 */
typedef struct{
__REG32 PRIO60              : 8;
__REG32 PRIO61              : 8;
__REG32 PRIO62              : 8;
__REG32 PRIO63              : 8;
} __tzic_priority15_bits;

/* Priority Register 16 */
typedef struct{
__REG32 PRIO64              : 8;
__REG32 PRIO65              : 8;
__REG32 PRIO66              : 8;
__REG32 PRIO67              : 8;
} __tzic_priority16_bits;

/* Priority Register 17 */
typedef struct{
__REG32 PRIO68              : 8;
__REG32 PRIO69              : 8;
__REG32 PRIO70              : 8;
__REG32 PRIO71              : 8;
} __tzic_priority17_bits;

/* Priority Register 18 */
typedef struct{
__REG32 PRIO72              : 8;
__REG32 PRIO73              : 8;
__REG32 PRIO74              : 8;
__REG32 PRIO75              : 8;
} __tzic_priority18_bits;

/* Priority Register 19 */
typedef struct{
__REG32 PRIO76              : 8;
__REG32 PRIO77              : 8;
__REG32 PRIO78              : 8;
__REG32 PRIO79              : 8;
} __tzic_priority19_bits;

/* Priority Register 20 */
typedef struct{
__REG32 PRIO80              : 8;
__REG32 PRIO81              : 8;
__REG32 PRIO82              : 8;
__REG32 PRIO83              : 8;
} __tzic_priority20_bits;

/* Priority Register 21 */
typedef struct{
__REG32 PRIO84              : 8;
__REG32 PRIO85              : 8;
__REG32 PRIO86              : 8;
__REG32 PRIO87              : 8;
} __tzic_priority21_bits;

/* Priority Register 22 */
typedef struct{
__REG32 PRIO88              : 8;
__REG32 PRIO89              : 8;
__REG32 PRIO90              : 8;
__REG32 PRIO91              : 8;
} __tzic_priority22_bits;

/* Priority Register 23 */
typedef struct{
__REG32 PRIO92              : 8;
__REG32 PRIO93              : 8;
__REG32 PRIO94              : 8;
__REG32 PRIO95              : 8;
} __tzic_priority23_bits;

/* Priority Register 24 */
typedef struct{
__REG32 PRIO96              : 8;
__REG32 PRIO97              : 8;
__REG32 PRIO98              : 8;
__REG32 PRIO99              : 8;
} __tzic_priority24_bits;

/* Priority Register 25 */
typedef struct{
__REG32 PRIO100             : 8;
__REG32 PRIO101             : 8;
__REG32 PRIO102             : 8;
__REG32 PRIO103             : 8;
} __tzic_priority25_bits;

/* Priority Register 26 */
typedef struct{
__REG32 PRIO104             : 8;
__REG32 PRIO105             : 8;
__REG32 PRIO106             : 8;
__REG32 PRIO107             : 8;
} __tzic_priority26_bits;

/* Priority Register 27 */
typedef struct{
__REG32 PRIO108             : 8;
__REG32 PRIO109             : 8;
__REG32 PRIO110             : 8;
__REG32 PRIO111             : 8;
} __tzic_priority27_bits;

/* Priority Register 28 */
typedef struct{
__REG32 PRIO112             : 8;
__REG32 PRIO113             : 8;
__REG32 PRIO114             : 8;
__REG32 PRIO115             : 8;
} __tzic_priority28_bits;

/* Priority Register 29 */
typedef struct{
__REG32 PRIO116             : 8;
__REG32 PRIO117             : 8;
__REG32 PRIO118             : 8;
__REG32 PRIO119             : 8;
} __tzic_priority29_bits;

/* Priority Register 30 */
typedef struct{
__REG32 PRIO120             : 8;
__REG32 PRIO121             : 8;
__REG32 PRIO122             : 8;
__REG32 PRIO123             : 8;
} __tzic_priority30_bits;

/* Priority Register 31 */
typedef struct{
__REG32 PRIO124             : 8;
__REG32 PRIO125             : 8;
__REG32 PRIO126             : 8;
__REG32 PRIO127             : 8;
} __tzic_priority31_bits;

/* Pending Register 0 */
typedef struct{
__REG32 PND0          : 1;
__REG32 PND1          : 1;
__REG32 PND2          : 1;
__REG32 PND3          : 1;
__REG32 PND4          : 1;
__REG32 PND5          : 1;
__REG32 PND6          : 1;
__REG32 PND7          : 1;
__REG32 PND8          : 1;
__REG32 PND9          : 1;
__REG32 PND10         : 1;
__REG32 PND11         : 1;
__REG32 PND12         : 1;
__REG32 PND13         : 1;
__REG32 PND14         : 1;
__REG32 PND15         : 1;
__REG32 PND16         : 1;
__REG32 PND17         : 1;
__REG32 PND18         : 1;
__REG32 PND19         : 1;
__REG32 PND20         : 1;
__REG32 PND21         : 1;
__REG32 PND22         : 1;
__REG32 PND23         : 1;
__REG32 PND24         : 1;
__REG32 PND25         : 1;
__REG32 PND26         : 1;
__REG32 PND27         : 1;
__REG32 PND28         : 1;
__REG32 PND29         : 1;
__REG32 PND30         : 1;
__REG32 PND31         : 1;
} __tzic_pnd0_bits;

/* Pending Register 1 */
typedef struct{
__REG32 PND32         : 1;
__REG32 PND33         : 1;
__REG32 PND34         : 1;
__REG32 PND35         : 1;
__REG32 PND36         : 1;
__REG32 PND37         : 1;
__REG32 PND38         : 1;
__REG32 PND39         : 1;
__REG32 PND40         : 1;
__REG32 PND41         : 1;
__REG32 PND42         : 1;
__REG32 PND43         : 1;
__REG32 PND44         : 1;
__REG32 PND45         : 1;
__REG32 PND46         : 1;
__REG32 PND47         : 1;
__REG32 PND48         : 1;
__REG32 PND49         : 1;
__REG32 PND50         : 1;
__REG32 PND51         : 1;
__REG32 PND52         : 1;
__REG32 PND53         : 1;
__REG32 PND54         : 1;
__REG32 PND55         : 1;
__REG32 PND56         : 1;
__REG32 PND57         : 1;
__REG32 PND58         : 1;
__REG32 PND59         : 1;
__REG32 PND60         : 1;
__REG32 PND61         : 1;
__REG32 PND62         : 1;
__REG32 PND63         : 1;
} __tzic_pnd1_bits;

/* Pending Register 2 */
typedef struct{
__REG32 PND64         : 1;
__REG32 PND65         : 1;
__REG32 PND66         : 1;
__REG32 PND67         : 1;
__REG32 PND68         : 1;
__REG32 PND69         : 1;
__REG32 PND70         : 1;
__REG32 PND71         : 1;
__REG32 PND72         : 1;
__REG32 PND73         : 1;
__REG32 PND74         : 1;
__REG32 PND75         : 1;
__REG32 PND76         : 1;
__REG32 PND77         : 1;
__REG32 PND78         : 1;
__REG32 PND79         : 1;
__REG32 PND80         : 1;
__REG32 PND81         : 1;
__REG32 PND82         : 1;
__REG32 PND83         : 1;
__REG32 PND84         : 1;
__REG32 PND85         : 1;
__REG32 PND86         : 1;
__REG32 PND87         : 1;
__REG32 PND88         : 1;
__REG32 PND89         : 1;
__REG32 PND90         : 1;
__REG32 PND91         : 1;
__REG32 PND92         : 1;
__REG32 PND93         : 1;
__REG32 PND94         : 1;
__REG32 PND95         : 1;
} __tzic_pnd2_bits;

/* Pending Register 3 */
typedef struct{
__REG32 PND96         : 1;
__REG32 PND97         : 1;
__REG32 PND98         : 1;
__REG32 PND99         : 1;
__REG32 PND100        : 1;
__REG32 PND101        : 1;
__REG32 PND102        : 1;
__REG32 PND103        : 1;
__REG32 PND104        : 1;
__REG32 PND105        : 1;
__REG32 PND106        : 1;
__REG32 PND107        : 1;
__REG32 PND108        : 1;
__REG32 PND109        : 1;
__REG32 PND110        : 1;
__REG32 PND111        : 1;
__REG32 PND112        : 1;
__REG32 PND113        : 1;
__REG32 PND114        : 1;
__REG32 PND115        : 1;
__REG32 PND116        : 1;
__REG32 PND117        : 1;
__REG32 PND118        : 1;
__REG32 PND119        : 1;
__REG32 PND120        : 1;
__REG32 PND121        : 1;
__REG32 PND122        : 1;
__REG32 PND123        : 1;
__REG32 PND124        : 1;
__REG32 PND125        : 1;
__REG32 PND126        : 1;
__REG32 PND127        : 1;
} __tzic_pnd3_bits;

/* High Priority Pending Register 0 */
typedef struct{
__REG32 HIPND0          : 1;
__REG32 HIPND1          : 1;
__REG32 HIPND2          : 1;
__REG32 HIPND3          : 1;
__REG32 HIPND4          : 1;
__REG32 HIPND5          : 1;
__REG32 HIPND6          : 1;
__REG32 HIPND7          : 1;
__REG32 HIPND8          : 1;
__REG32 HIPND9          : 1;
__REG32 HIPND10         : 1;
__REG32 HIPND11         : 1;
__REG32 HIPND12         : 1;
__REG32 HIPND13         : 1;
__REG32 HIPND14         : 1;
__REG32 HIPND15         : 1;
__REG32 HIPND16         : 1;
__REG32 HIPND17         : 1;
__REG32 HIPND18         : 1;
__REG32 HIPND19         : 1;
__REG32 HIPND20         : 1;
__REG32 HIPND21         : 1;
__REG32 HIPND22         : 1;
__REG32 HIPND23         : 1;
__REG32 HIPND24         : 1;
__REG32 HIPND25         : 1;
__REG32 HIPND26         : 1;
__REG32 HIPND27         : 1;
__REG32 HIPND28         : 1;
__REG32 HIPND29         : 1;
__REG32 HIPND30         : 1;
__REG32 HIPND31         : 1;
} __tzic_hipnd0_bits;

/* High Priority Pending Register 1 */
typedef struct{
__REG32 HIPND32         : 1;
__REG32 HIPND33         : 1;
__REG32 HIPND34         : 1;
__REG32 HIPND35         : 1;
__REG32 HIPND36         : 1;
__REG32 HIPND37         : 1;
__REG32 HIPND38         : 1;
__REG32 HIPND39         : 1;
__REG32 HIPND40         : 1;
__REG32 HIPND41         : 1;
__REG32 HIPND42         : 1;
__REG32 HIPND43         : 1;
__REG32 HIPND44         : 1;
__REG32 HIPND45         : 1;
__REG32 HIPND46         : 1;
__REG32 HIPND47         : 1;
__REG32 HIPND48         : 1;
__REG32 HIPND49         : 1;
__REG32 HIPND50         : 1;
__REG32 HIPND51         : 1;
__REG32 HIPND52         : 1;
__REG32 HIPND53         : 1;
__REG32 HIPND54         : 1;
__REG32 HIPND55         : 1;
__REG32 HIPND56         : 1;
__REG32 HIPND57         : 1;
__REG32 HIPND58         : 1;
__REG32 HIPND59         : 1;
__REG32 HIPND60         : 1;
__REG32 HIPND61         : 1;
__REG32 HIPND62         : 1;
__REG32 HIPND63         : 1;
} __tzic_hipnd1_bits;

/* High Priority Pending Register 2 */
typedef struct{
__REG32 HIPND64         : 1;
__REG32 HIPND65         : 1;
__REG32 HIPND66         : 1;
__REG32 HIPND67         : 1;
__REG32 HIPND68         : 1;
__REG32 HIPND69         : 1;
__REG32 HIPND70         : 1;
__REG32 HIPND71         : 1;
__REG32 HIPND72         : 1;
__REG32 HIPND73         : 1;
__REG32 HIPND74         : 1;
__REG32 HIPND75         : 1;
__REG32 HIPND76         : 1;
__REG32 HIPND77         : 1;
__REG32 HIPND78         : 1;
__REG32 HIPND79         : 1;
__REG32 HIPND80         : 1;
__REG32 HIPND81         : 1;
__REG32 HIPND82         : 1;
__REG32 HIPND83         : 1;
__REG32 HIPND84         : 1;
__REG32 HIPND85         : 1;
__REG32 HIPND86         : 1;
__REG32 HIPND87         : 1;
__REG32 HIPND88         : 1;
__REG32 HIPND89         : 1;
__REG32 HIPND90         : 1;
__REG32 HIPND91         : 1;
__REG32 HIPND92         : 1;
__REG32 HIPND93         : 1;
__REG32 HIPND94         : 1;
__REG32 HIPND95         : 1;
} __tzic_hipnd2_bits;

/* High Priority Pending Register 3 */
typedef struct{
__REG32 HIPND96         : 1;
__REG32 HIPND97         : 1;
__REG32 HIPND98         : 1;
__REG32 HIPND99         : 1;
__REG32 HIPND100        : 1;
__REG32 HIPND101        : 1;
__REG32 HIPND102        : 1;
__REG32 HIPND103        : 1;
__REG32 HIPND104        : 1;
__REG32 HIPND105        : 1;
__REG32 HIPND106        : 1;
__REG32 HIPND107        : 1;
__REG32 HIPND108        : 1;
__REG32 HIPND109        : 1;
__REG32 HIPND110        : 1;
__REG32 HIPND111        : 1;
__REG32 HIPND112        : 1;
__REG32 HIPND113        : 1;
__REG32 HIPND114        : 1;
__REG32 HIPND115        : 1;
__REG32 HIPND116        : 1;
__REG32 HIPND117        : 1;
__REG32 HIPND118        : 1;
__REG32 HIPND119        : 1;
__REG32 HIPND120        : 1;
__REG32 HIPND121        : 1;
__REG32 HIPND122        : 1;
__REG32 HIPND123        : 1;
__REG32 HIPND124        : 1;
__REG32 HIPND125        : 1;
__REG32 HIPND126        : 1;
__REG32 HIPND127        : 1;
} __tzic_hipnd3_bits;

/* Wakeup Configuration Register 0 */
typedef struct{
__REG32 WAKEUP0         : 1;
__REG32 WAKEUP1         : 1;
__REG32 WAKEUP2         : 1;
__REG32 WAKEUP3         : 1;
__REG32 WAKEUP4         : 1;
__REG32 WAKEUP5         : 1;
__REG32 WAKEUP6         : 1;
__REG32 WAKEUP7         : 1;
__REG32 WAKEUP8         : 1;
__REG32 WAKEUP9         : 1;
__REG32 WAKEUP10        : 1;
__REG32 WAKEUP11        : 1;
__REG32 WAKEUP12        : 1;
__REG32 WAKEUP13        : 1;
__REG32 WAKEUP14        : 1;
__REG32 WAKEUP15        : 1;
__REG32 WAKEUP16        : 1;
__REG32 WAKEUP17        : 1;
__REG32 WAKEUP18        : 1;
__REG32 WAKEUP19        : 1;
__REG32 WAKEUP20        : 1;
__REG32 WAKEUP21        : 1;
__REG32 WAKEUP22        : 1;
__REG32 WAKEUP23        : 1;
__REG32 WAKEUP24        : 1;
__REG32 WAKEUP25        : 1;
__REG32 WAKEUP26        : 1;
__REG32 WAKEUP27        : 1;
__REG32 WAKEUP28        : 1;
__REG32 WAKEUP29        : 1;
__REG32 WAKEUP30        : 1;
__REG32 WAKEUP31        : 1;
} __tzic_wakeup0_bits;

/* Wakeup Configuration Register 1 */
typedef struct{
__REG32 WAKEUP32        : 1;
__REG32 WAKEUP33        : 1;
__REG32 WAKEUP34        : 1;
__REG32 WAKEUP35        : 1;
__REG32 WAKEUP36        : 1;
__REG32 WAKEUP37        : 1;
__REG32 WAKEUP38        : 1;
__REG32 WAKEUP39        : 1;
__REG32 WAKEUP40        : 1;
__REG32 WAKEUP41        : 1;
__REG32 WAKEUP42        : 1;
__REG32 WAKEUP43        : 1;
__REG32 WAKEUP44        : 1;
__REG32 WAKEUP45        : 1;
__REG32 WAKEUP46        : 1;
__REG32 WAKEUP47        : 1;
__REG32 WAKEUP48        : 1;
__REG32 WAKEUP49        : 1;
__REG32 WAKEUP50        : 1;
__REG32 WAKEUP51        : 1;
__REG32 WAKEUP52        : 1;
__REG32 WAKEUP53        : 1;
__REG32 WAKEUP54        : 1;
__REG32 WAKEUP55        : 1;
__REG32 WAKEUP56        : 1;
__REG32 WAKEUP57        : 1;
__REG32 WAKEUP58        : 1;
__REG32 WAKEUP59        : 1;
__REG32 WAKEUP60        : 1;
__REG32 WAKEUP61        : 1;
__REG32 WAKEUP62        : 1;
__REG32 WAKEUP63        : 1;
} __tzic_wakeup1_bits;

/* Wakeup Configuration Register 2 */
typedef struct{
__REG32 WAKEUP64        : 1;
__REG32 WAKEUP65        : 1;
__REG32 WAKEUP66        : 1;
__REG32 WAKEUP67        : 1;
__REG32 WAKEUP68        : 1;
__REG32 WAKEUP69        : 1;
__REG32 WAKEUP70        : 1;
__REG32 WAKEUP71        : 1;
__REG32 WAKEUP72        : 1;
__REG32 WAKEUP73        : 1;
__REG32 WAKEUP74        : 1;
__REG32 WAKEUP75        : 1;
__REG32 WAKEUP76        : 1;
__REG32 WAKEUP77        : 1;
__REG32 WAKEUP78        : 1;
__REG32 WAKEUP79        : 1;
__REG32 WAKEUP80        : 1;
__REG32 WAKEUP81        : 1;
__REG32 WAKEUP82        : 1;
__REG32 WAKEUP83        : 1;
__REG32 WAKEUP84        : 1;
__REG32 WAKEUP85        : 1;
__REG32 WAKEUP86        : 1;
__REG32 WAKEUP87        : 1;
__REG32 WAKEUP88        : 1;
__REG32 WAKEUP89        : 1;
__REG32 WAKEUP90        : 1;
__REG32 WAKEUP91        : 1;
__REG32 WAKEUP92        : 1;
__REG32 WAKEUP93        : 1;
__REG32 WAKEUP94        : 1;
__REG32 WAKEUP95        : 1;
} __tzic_wakeup2_bits;

/* Wakeup Configuration Register 3 */
typedef struct{
__REG32 WAKEUP96        : 1;
__REG32 WAKEUP97        : 1;
__REG32 WAKEUP98        : 1;
__REG32 WAKEUP99        : 1;
__REG32 WAKEUP100       : 1;
__REG32 WAKEUP101       : 1;
__REG32 WAKEUP102       : 1;
__REG32 WAKEUP103       : 1;
__REG32 WAKEUP104       : 1;
__REG32 WAKEUP105       : 1;
__REG32 WAKEUP106       : 1;
__REG32 WAKEUP107       : 1;
__REG32 WAKEUP108       : 1;
__REG32 WAKEUP109       : 1;
__REG32 WAKEUP110       : 1;
__REG32 WAKEUP111       : 1;
__REG32 WAKEUP112       : 1;
__REG32 WAKEUP113       : 1;
__REG32 WAKEUP114       : 1;
__REG32 WAKEUP115       : 1;
__REG32 WAKEUP116       : 1;
__REG32 WAKEUP117       : 1;
__REG32 WAKEUP118       : 1;
__REG32 WAKEUP119       : 1;
__REG32 WAKEUP120       : 1;
__REG32 WAKEUP121       : 1;
__REG32 WAKEUP122       : 1;
__REG32 WAKEUP123       : 1;
__REG32 WAKEUP124       : 1;
__REG32 WAKEUP125       : 1;
__REG32 WAKEUP126       : 1;
__REG32 WAKEUP127       : 1;
} __tzic_wakeup3_bits;

/*Software Interrupt Register */
typedef struct{
__REG32 INTID           :10;
__REG32                 :21;
__REG32 INTNEG          : 1;
} __tzic_swint_bits;

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
__REG32            : 3;
__REG32 SAD        : 1;
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

/* UARTs One Millisecond Register */
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

/* Watchdog Control Register (WCR) */
typedef struct {
__REG16 WDZST  : 1;     /* Bit  0       - Watchdog Low Power*/
__REG16 WDBG   : 1;     /* Bits 1       - Watchdog DEBUG Enable*/
__REG16 WDE    : 1;     /* Bit  2       - Watchdog Enable*/
__REG16 WDT    : 1;     /* Bit  3       - ~WDOG or ~WDOG_RESET Enable*/
__REG16 SRS    : 1;     /* Bit  4       - Software Reset Signal*/
__REG16 WDA    : 1;     /* Bit  5       - Watchdog Assertion*/
__REG16        : 1;     /* Bit  6       - Reserved*/
__REG16 WDW    : 1;     /* Bit  7 */
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
__REG16 WICT  : 8;
__REG16       : 6;
__REG16 WTIS  : 1;
__REG16 WIE   : 1;
} __wmcr_bits;

/* VPU Code Run Register (VPU.CodeRun) */
typedef struct {
__REG32 CodeRun   : 1;
__REG32           :31;
} __vpu_coderun_bits;

/* VPU Code Run Register (VPU.CodeRun) */
typedef struct {
__REG32 CodeData  :16;
__REG32 CodeAddr  :13;
__REG32           : 3;
} __vpu_codedown_bits;

/* VPU Host Interrupt Request Register (VPU.HostIntReq) */
typedef struct {
__REG32 IntReq    : 1;
__REG32           :31;
} __vpu_hostintreq_bits;

/* VPU BIT Interrupt Clear Register (VPU.BitIntClear) */
typedef struct {
__REG32 IntClear  : 1;
__REG32           :31;
} __vpu_bitintclear_bits;

/* VPU BIT Interrupt Status Register (VPU.BitIntSts) */
typedef struct {
__REG32 IntSts    : 1;
__REG32           :31;
} __vpu_bitintsts_bits;

/* VPU BIT Code Reset Register (VPU.BitCodeReset) */
typedef struct {
__REG32 CodeReset : 1;
__REG32           :31;
} __vpu_bitcodereset_bits;

/* VPU BIT Current PC Register (VPU.BitCurPc) */
typedef struct {
__REG32 CurPc     :14;
__REG32           :18;
} __vpu_bitcurpc_bits;

/* VPU BIT Busy Register (VPU.BitCodecBusy) */
typedef struct {
__REG32 CodecBusy : 1;
__REG32           :31;
} __vpu_bitcodecbusy_bits;

/* USB Control Register 0 (USB.USB_CTRL_0) */
typedef struct {
__REG32             : 8;
__REG32 H1_PWR_POL  : 1;
__REG32             : 2;
__REG32 H1WIE       : 1;
__REG32             : 3;
__REG32 H1WIR       : 1;
__REG32             : 8;
__REG32 O_PWR_POL   : 1;
__REG32             : 2;
__REG32 OWIE        : 1;
__REG32             : 3;
__REG32 OWIR        : 1;
} __usb_ctrl_0_bits;

/* USB OTG UTMI PHY Control Register 0 (USB.USB_OTG_PHY_CTRL_0) */
typedef struct {
__REG32 CHRGDET_INT_FLG : 1;
__REG32 CHRGDET_INT_EN  : 1;
__REG32 CHRGDET         : 1;
__REG32                 : 1;
__REG32 H1_WK_PLL_EN    : 1;
__REG32 H1_OVER_CUR_DIS : 1;
__REG32 H1_OVER_CUR_POL : 1;
__REG32 OTG_WK_PLL_EN   : 1;
__REG32 OTG_OVER_CUR_DIS: 1;
__REG32 OTG_OVER_CUR_POL: 1;
__REG32 UTMI_ON_CLOCK   : 1;
__REG32                 :12;
__REG32 CHGRDETON       : 1;
__REG32 CHGRDETEN       : 1;
__REG32                 : 7;
} __usb_otg_phy_ctrl_0_bits;

/* USB OTG UTMI PHY Control Register 1 (USB.USB_OTG_PHY_CTRL_1) */
typedef struct {
__REG32 PLLDIVVALUE     : 2;
__REG32 EXTCAL          : 5;
__REG32 CALBP           : 1;
__REG32 PREEMDEPTH      : 1;
__REG32 ENPRE           : 1;
__REG32 LSRFTSEL        : 2;
__REG32 FSRFTSEL        : 2;
__REG32 ICPCTRL         : 2;
__REG32 FSTUNEVSEL      : 3;
__REG32 HSTEDVSEL       : 2;
__REG32 HSDEDVSEL       : 2;
__REG32 HSDRVSLOPE      : 4;
__REG32 HSDRVAMPLITUDE  : 2;
__REG32 HSDRVTIMINGN    : 2;
__REG32 HSDRVTIMINGP    : 1;
} __usb_otg_phy_ctrl_1_bits;

/* USB Control Register 1 (USB.USB_CTRL_1) */
typedef struct {
__REG32 H2_XCVR_SERCLK_SEL  : 2;
__REG32 H2_XCVR_CLK_SEL     : 2;
__REG32 H3_XCVR_SERCLK_SEL  : 2;
__REG32 H3_XCVR_CLK_SEL     : 2;
__REG32                     :18;
__REG32 UH2_EXT_ULPI_EN     : 1;
__REG32 UH3_EXT_ULPI_EN     : 1;
__REG32                     : 4;
} __usb_ctrl_1_bits;

/* USB Host2 Control Register (USB.USB_UH2_CTRL) */
typedef struct {
__REG32                     : 4;
__REG32 H2_PWR_POL          : 1;
__REG32 OIDWK_EN            : 1;
__REG32 OVBWK_EN            : 1;
__REG32 H2WIE               : 1;
__REG32 H2UIE               : 1;
__REG32 H2SIC               : 2;
__REG32 H2ICTPC             : 1;
__REG32                     : 1;
__REG32 ICTOIE              : 1;
__REG32                     : 3;
__REG32 H2WIR               : 1;
__REG32 ICVOL_TO            : 1;
__REG32                     :11;
__REG32 H2_OVR_DIS          : 1;
__REG32 H2_OVR_POL          : 1;
} __usb_uh2_ctrl_bits;

/* USB Host3 Control Register (USB.USB_UH3_CTRL) */
typedef struct {
__REG32                     : 4;
__REG32 H3_PWR_POL          : 1;
__REG32                     : 2;
__REG32 H3WIE               : 1;
__REG32 H3UIE               : 1;
__REG32 H3SIC               : 2;
__REG32                     : 6;
__REG32 H3WIR               : 1;
__REG32                     : 1;
__REG32 H3_SER_DRV_EN       : 1;
__REG32 H2_SER_DRV_EN       : 1;
__REG32                     : 9;
__REG32 H3_OVR_DIS          : 1;
__REG32 H3_OVR_POL          : 1;
} __usb_uh3_ctrl_bits;

/* USB Host1 UTMI PHY Control Register 0 (USB.USB_UH1_PHY_CTRL_0) */
typedef struct {
__REG32 CHRGDET_INT_FLG     : 1;
__REG32 CHRGDET_INT_EN      : 1;
__REG32 CHRGDET             : 1;
__REG32                     :20;
__REG32 CHGRDETON           : 1;
__REG32 CHGRDETEN           : 1;
__REG32                     : 7;
} __usb_uh1_phy_ctrl_0_bits;

/* USB Host1 UTMI PHY Control Register 1 (USB.USB_UH1_PHY_CTRL_1) */
typedef struct {
__REG32 PLLDIVVALUE         : 2;
__REG32 EXTCAL              : 5;
__REG32 CALBP               : 1;
__REG32 PREEMDEPTH          : 1;
__REG32 ENPRE               : 1;
__REG32 LSRFTSEL            : 2;
__REG32 FSRFTSEL            : 2;
__REG32 ICPCTRL             : 2;
__REG32 FSTUNEVSEL          : 3;
__REG32 HSTEDVSEL           : 2;
__REG32 HSDEDVSEL           : 2;
__REG32 HSDRVSLOPE          : 4;
__REG32 HSDRVAMPLITUDE      : 2;
__REG32 HSDRVTIMINGN        : 2;
__REG32 HSDRVTIMINGP        : 1;
} __usb_uh1_phy_ctrl_1_bits;

/* USB Clock ON/OFF Control Register (USB.USB_CLKONOFF_CTRL) */
typedef struct {
__REG32                     :17;
__REG32 OTG_AHBCLK_OFF      : 1;
__REG32 H1_AHBCLK_OFF       : 1;
__REG32 H2_AHBCLK_OFF       : 1;
__REG32 H3_AHBCLK_OFF       : 1;
__REG32 H2_INT60_CK_OFF     : 1;
__REG32 H3_INT60_CK_OFF     : 1;
__REG32                     : 6;
__REG32 H2_PLL_CK_OFF       : 1;
__REG32 H3_PLL_CK_OFF       : 1;
__REG32                     : 1;
} __usb_clkonoff_ctrl_bits;

/* Identification Register */
typedef struct{
__REG32 ID        : 6;
__REG32           : 2;
__REG32 NID       : 6;
__REG32           : 2;
__REG32 REVISION  : 8;
__REG32           : 8;
} __uog_id_bits;

/* General Hardware Parameters */
typedef struct{
__REG32           : 4;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 2;
__REG32           :21;
} __uog_hwgeneral_bits;

/* Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :28;
} __uog_hwhost_bits;

/* Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __uog_hwdevice_bits;

/* Buffer Hardware Parameters */
typedef struct{
__REG32 TXBURST   : 8;
__REG32           : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __uog_hwtxbuf_bits;

/* Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __uog_hwrxbuf_bits;

/* Host Control Structural Parameters */
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

/* Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __uog_hccparams_bits;

/* Device Interface Version Number */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __uh_dcivrsion_bits;

/* Device Control Capability Parameters */
typedef struct{
__REG32 DEN         : 5;
__REG32             : 2;
__REG32 DC          : 1;
__REG32 HC          : 1;
__REG32             :23;
} __uog_dccparams_bits;

/* General Purpose Timer Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __uh_gptimerld_bits;

/* General Purpose Timer Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMODE     : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __uh_gptimerctrl_bits;

/* System Bus Interface Configuration */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __uh_sbuscfg_bits;

/* USB Command Register */
typedef struct{
__REG32 RS        : 1;
__REG32 RST       : 1;
__REG32 FS0       : 1;
__REG32 FS1       : 1;
__REG32 PSE       : 1;
__REG32 ASE       : 1;
__REG32 IAA       : 1;
__REG32           : 1;
__REG32 ASP0      : 1;
__REG32 ASP1      : 1;
__REG32           : 1;
__REG32 ASPE      : 1;
__REG32           : 1;
__REG32 SUTW      : 1;
__REG32 ATDTW     : 1;
__REG32 FS2       : 1;
__REG32 ITC       : 8;
__REG32           : 8;
} __uog_usbcmd_bits;

/* USB Status Register */
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
__REG32 TI0       : 1;
__REG32 TI1       : 1;
__REG32           : 6;
} __uog_usbsts_bits;

/* USB Interrupt Enable */
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
__REG32           : 5;
__REG32 NAKE      : 1;
__REG32           : 1;
__REG32 UAIE      : 1;
__REG32 UPIE      : 1;
__REG32           : 4;
__REG32 TIE0      : 1;
__REG32 TIE1      : 1;
__REG32           : 6;
} __uog_usbintr_bits;

/* USB Frame Index */
typedef struct{
__REG32 FRINDEX   :14;
__REG32           :18;
} __uog_frindex_bits;

/* PERIODICLISTBASE—Host Controller Frame List Base Address */
/* DEVICEADDR—Device Controller USB Device Address */
typedef union {
/* UHx_PERIODICLISTBASE*/
/* UOG_PERIODICLISTBASE*/
  struct{
    __REG32           :12;
    __REG32 PERBASE   :20;
  };
/* UOG_DEVICEADDR*/
  struct{
    __REG32           :24;
    __REG32 USBADR    : 8;
  };
} __uog_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
/* UHx_ASYNCLISTADDR*/
/* UOG_ASYNCLISTADDR*/
  struct{
    __REG32           : 6;
    __REG32 ASYBASE   :26;
  };
/* UOG_ENDPOINTLISTADDR*/
  struct{
    __REG32           :11;
    __REG32 EPBASE    :21;
  };
} __uog_asynclistaddr_bits;

/* AHB Burst Size Control  */
typedef struct{
__REG32 RXPBURST  : 8;
__REG32 TXPBURST  : 9;
__REG32           :15;
} __uog_burstsize_bits;

/* USB Transmit FIFO Fill Tunning Register */
typedef struct{
__REG32 TXSCHOH   : 8;
__REG32 TXSCHEAL  : 5;
__REG32           : 3;
__REG32 TXFIFOTHR : 6;
__REG32           :10;
} __uog_txfilltuning_bits;

/* IC_USB Enable and Voltage Negotiation Register */
typedef struct{
__REG32 C_VDD1    : 3;
__REG32 IC1       : 1;
__REG32           :28;
} __uog_ic_usb_bits;

/* ULPI VIEWPORT register */
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
} __uog_viewport_bits;

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

/* Port Status Control 1 */
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
__REG32 PTS2      : 1;
__REG32 PSPD      : 2;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __uog_portsc_bits;

/* Status Control */
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

/* Device Mode */
typedef struct{
__REG32 CM        : 2;
__REG32 E         : 1;
__REG32 SL        : 1;
__REG32 S         : 1;
__REG32           :27;
} __uog_usbmode_bits;

/* Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT  :16;
__REG32                 :16;
} __uog_endptsetupstat_bits;

/* Endpoint Initialization */
typedef struct{
__REG32 PERB            : 8;
__REG32                 : 8;
__REG32 PETB            : 8;
__REG32                 : 8;
} __uog_endptprime_bits;

/* Endpoint De-Initialize */
typedef struct{
__REG32 FERB            : 8;
__REG32                 : 8;
__REG32 FETB            : 8;
__REG32                 : 8;
} __uog_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR            : 8;
__REG32                 : 8;
__REG32 ETBR            : 8;
__REG32                 : 8;
} __uog_endptstat_bits;

/* Endpoint Compete */
typedef struct{
__REG32 ERCE            : 8;
__REG32                 : 8;
__REG32 ETCE            : 8;
__REG32                 : 8;
} __uog_endptcomplete_bits;

/* Endpoint Compete */
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

/* Endpoint Compete */
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

/* Master Privilege Register 1 */
typedef struct{
__REG32 SAHARA            : 4;
__REG32                   : 4;
__REG32 PATA              : 4;
__REG32 USB               : 4;
__REG32 GPU_VPU_IPU       : 4;
__REG32 SDMA_SATA_FEC_MLB : 4;
__REG32 ARM_CORE          : 4;
__REG32                   : 4;
} __aipstz_mpr1_bits;

/* Master Privilege Register 2 */
typedef struct{
__REG32 ESDHCV2_3       : 4;
__REG32 ESDHCV2_2       : 4;
__REG32 ESDHCV2_1       : 4;
__REG32 DAP             : 4;
__REG32                 : 4;
__REG32 ESDCHV2_4       : 4;
__REG32 RTIC            : 4;
__REG32 SCC             : 4;
} __aipstz_mpr2_bits;

/* Off-Platform Peripheral Access Control Register 1 (AIPSTZ1.OPACR1) */
typedef struct{
__REG32 WDOG_2          : 4;
__REG32 WDOG_1          : 4;
__REG32 KPP             : 4;
__REG32 GPIO_4          : 4;
__REG32 GPIO_3          : 4;
__REG32 GPIO_2          : 4;
__REG32 GPIO_1          : 4;
__REG32 USB             : 4;
} __aipstz1_opacr1_bits;

/* Off-Platform Peripheral Access Control Register 2 (AIPSTZ1.OPACR2) */
typedef struct{
__REG32 UART_1          : 4;
__REG32 PWM_2           : 4;
__REG32 PWM_1           : 4;
__REG32 EPIT_2          : 4;
__REG32 EPIT_1          : 4;
__REG32 IOMUXC          : 4;
__REG32 SRTC            : 4;
__REG32 GPT             : 4;
} __aipstz1_opacr2_bits;

/* Off-Platform Peripheral Access Control Register 3 (AIPSTZ1.OPACR3) */
typedef struct{
__REG32 GPIO_5          : 4;
__REG32 GPC             : 4;
__REG32 CCM             : 4;
__REG32 SRC             : 4;
__REG32 CAN_2           : 4;
__REG32 CAN_1           : 4;
__REG32 USB             : 4;
__REG32 UART_2          : 4;
} __aipstz1_opacr3_bits;

/* Off-Platform Peripheral Access Control Register 4 (AIPSTZ1.OPACR4) */
typedef struct{
__REG32                 :12;
__REG32 UART_4          : 4;
__REG32 I2C_3           : 4;
__REG32 PATA            : 4;
__REG32 GPIO_7          : 4;
__REG32 GPIO_6          : 4;
} __aipstz1_opacr4_bits;

/* Off-Platform Peripheral Access Control Register 5 (AIPSTZ1.OPACR5) */
typedef struct{
__REG32                 :28;
__REG32 SPBA            : 4;
} __aipstz1_opacr5_bits;

/* Off-Platform Peripheral Access Control Register 1 (AIPSTZ2.OPACR1) */
typedef struct{
__REG32 CSU             : 4;
__REG32 IIM             : 4;
__REG32 AHBMAX          : 4;
__REG32 UART_5          : 4;
__REG32 DPLLIP_4        : 4;
__REG32 DPLLIP_3        : 4;
__REG32 DPLLIP_2        : 4;
__REG32 DPLLIP_1        : 4;
} __aipstz2_opacr1_bits;

/* Off-Platform Peripheral Access Control Register 2 (AIPSTZ2.OPACR2) */
typedef struct{
__REG32 RTIC            : 4;
__REG32 ROMC            : 4;
__REG32 SCC             : 4;
__REG32 SDMA            : 4;
__REG32 ECSPI_2         : 4;
__REG32 FIRI            : 4;
__REG32 OWIRE           : 4;
__REG32 ARM_CORE        : 4;
} __aipstz2_opacr2_bits;

/* Off-Platform Peripheral Access Control Register 3 (AIPSTZ2.OPACR3) */
typedef struct{
__REG32 PLARB2          : 4;
__REG32 EXTMC           : 4;
__REG32 RTC             : 4;
__REG32 AUDMUX          : 4;
__REG32 SSI_1           : 4;
__REG32 I2C_1           : 4;
__REG32 I2C_2           : 4;
__REG32 CSPI            : 4;
} __aipstz2_opacr3_bits;

/* Off-Platform Peripheral Access Control Register 4 (AIPSTZ2.OPACR4) */
typedef struct{
__REG32 PTP             : 4;
__REG32 SAHARA          : 4;
__REG32 VPU             : 4;
__REG32 TVE             : 4;
__REG32 FEC             : 4;
__REG32 SSI_3           : 4;
__REG32 MLB             : 4;
__REG32 PLARB1          : 4;
} __aipstz2_opacr4_bits;

/* Platform Version ID Register */
typedef struct{
__REG32 ECO             : 8;
__REG32 MINOR           : 8;
__REG32 IMPL            : 8;
__REG32 SPEC            : 8;
} __armpc_pvid_bits;

/* General Purpose Control Register */
typedef struct{
__REG32 GPC             :16;
__REG32 DBGEN           : 1;
__REG32 ATRDY           : 1;
__REG32 NOCLKSTP        : 1;
__REG32                 :13;
} __armpc_gpc_bits;

/* Platform Internal Control Register */
typedef struct{
__REG32 PIC             : 8;
__REG32                 :24;
} __armpc_pic_bits;

/* Low Power Control Register */
typedef struct{
__REG32 DSM             : 1;
__REG32 DBG_DSM         : 1;
__REG32                 :30;
} __armpc_lpc_bits;

/* NEON Low Power Control Register */
typedef struct{
__REG32 NEON_RST        : 1;
__REG32                 :31;
} __armpc_nlpc_bits;

/* Internal Clock Generation Control Register */
typedef struct{
__REG32 IPG_CLK_DIVR    : 3;
__REG32 IPG_PRLD        : 1;
__REG32 ACLK_DIVR       : 3;
__REG32 ACLK_PRLD       : 1;
__REG32 DT_CLK_DIVR     : 3;
__REG32 DT_PRLD         : 1;
__REG32                 :20;
} __armpc_icgc_bits;

/* ARM Memory Configuration Register */
typedef struct{
__REG32 ALP             : 3;
__REG32 ALPEN           : 1;
__REG32                 :28;
} __armpc_amc_bits;

/* NEON Monitor Control Register */
typedef struct{
__REG32                 :12;
__REG32 PL              : 8;
__REG32                 :10;
__REG32 NME             : 1;
__REG32 IE              : 1;
} __armpc_nmc_bits;

/* NEON Monitor Status Register */
typedef struct{
__REG32                 :31;
__REG32 NI              : 1;
} __armpc_nms_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  AHBMAX
 **
 ***************************************************************************/
__IO_REG32_BIT(AHBMAX_MPR0,               0x63F94000,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR0,             0x63F94010,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR1,               0x63F94100,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR1,             0x63F94110,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR2,               0x63F94200,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR2,             0x63F94210,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR3,               0x63F94300,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR3,             0x63F94310,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR0,             0x63F94800,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR1,             0x63F94900,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR2,             0x63F94A00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR3,             0x63F94B00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR4,             0x63F94C00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR5,             0x63F94D00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR6,             0x63F94E00,__READ_WRITE ,__ahbmax_mgpcr_bits);

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
__IO_REG32(    ASRC_ASRDIA,               0x5002C060,__WRITE      );
__IO_REG32(    ASRC_ASRDOA,               0x5002C064,__READ       );
__IO_REG32(    ASRC_ASRDIB,               0x5002C068,__WRITE      );
__IO_REG32(    ASRC_ASRDOB,               0x5002C06C,__READ       );
__IO_REG32(    ASRC_ASRDIC,               0x5002C070,__WRITE      );
__IO_REG32(    ASRC_ASRDOC,               0x5002C074,__READ       );
__IO_REG32_BIT(ASRC_ASRIDRHA,             0x5002C080,__READ_WRITE ,__asrc_asridrha_bits);
__IO_REG32(    ASRC_ASRIDRLA,             0x5002C084,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASRIDRHB,             0x5002C088,__READ_WRITE ,__asrc_asridrhb_bits);
__IO_REG32(    ASRC_ASRIDRLB,             0x5002C08C,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASRIDRHC,             0x5002C090,__READ_WRITE ,__asrc_asridrhc_bits);
__IO_REG32(    ASRC_ASRIDRLC,             0x5002C094,__READ_WRITE );
__IO_REG32_BIT(ASRC_ASR76K,               0x5002C098,__READ_WRITE ,__asrc_asr76k_bits);
__IO_REG32_BIT(ASRC_ASR56K,               0x5002C09C,__READ_WRITE ,__asrc_asr56k_bits);
__IO_REG32_BIT(ASRC_ASRMCRA,              0x5002C0A0,__READ_WRITE ,__asrc_asrmcra_bits);
__IO_REG32_BIT(ASRC_ASRFSTA,              0x5002C0A4,__READ       ,__asrc_asrfsta_bits);
__IO_REG32_BIT(ASRC_ASRMCRB,              0x5002C0A8,__READ_WRITE ,__asrc_asrmcrb_bits);
__IO_REG32_BIT(ASRC_ASRFSTB,              0x5002C0AC,__READ       ,__asrc_asrfstb_bits);
__IO_REG32_BIT(ASRC_ASRMCRC,              0x5002C0B0,__READ_WRITE ,__asrc_asrmcrc_bits);
__IO_REG32_BIT(ASRC_ASRFSTC,              0x5002C0B4,__READ       ,__asrc_asrfstc_bits);
__IO_REG32_BIT(ASRC_ASRMCR1A,             0x5002C0C0,__READ_WRITE ,__asrc_asrmcr1_bits);
__IO_REG32_BIT(ASRC_ASRMCR1B,             0x5002C0C4,__READ_WRITE ,__asrc_asrmcr1_bits);
__IO_REG32_BIT(ASRC_ASRMCR1C,             0x5002C0C8,__READ_WRITE ,__asrc_asrmcr1_bits);

/***************************************************************************
 **
 **  AUDMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(AUDMUX_PTCR1,              0x63FD0000,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR1,              0x63FD0004,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR2,              0x63FD0008,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR2,              0x63FD000C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR3,              0x63FD0010,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR3,              0x63FD0014,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR4,              0x63FD0018,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR4,              0x63FD001C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR5,              0x63FD0020,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR5,              0x63FD0024,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR6,              0x63FD0028,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR6,              0x63FD002C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR7,              0x63FD0030,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR7,              0x63FD0034,__READ_WRITE ,__audmux_pdcr_bits);

/***************************************************************************
 **
 **  CCM
 **
 ***************************************************************************/
__IO_REG32_BIT(CCM_CCMR,                  0x53FD4000,__READ_WRITE ,__ccm_ccmr_bits);
__IO_REG32_BIT(CCM_CCDR,                  0x53FD4004,__READ_WRITE ,__ccm_ccdr_bits);
__IO_REG32_BIT(CCM_CSR,                   0x53FD4008,__READ       ,__ccm_csr_bits);
__IO_REG32_BIT(CCM_CCSR,                  0x53FD400C,__READ_WRITE ,__ccm_ccsr_bits);
__IO_REG32_BIT(CCM_CACRR,                 0x53FD4010,__READ_WRITE ,__ccm_cacrr_bits);
__IO_REG32_BIT(CCM_CBCDR,                 0x53FD4014,__READ_WRITE ,__ccm_cbcdr_bits);
__IO_REG32_BIT(CCM_CBCMR,                 0x53FD4018,__READ_WRITE ,__ccm_cbcmr_bits);
__IO_REG32_BIT(CCM_CSCMR1,                0x53FD401C,__READ_WRITE ,__ccm_cscmr1_bits);
__IO_REG32_BIT(CCM_CSCMR2,                0x53FD4020,__READ_WRITE ,__ccm_cscmr2_bits);
__IO_REG32_BIT(CCM_CSCDR1,                0x53FD4024,__READ_WRITE ,__ccm_cscdr1_bits);
__IO_REG32_BIT(CCM_CS1CDR,                0x53FD4028,__READ_WRITE ,__ccm_cs1cdr_bits);
__IO_REG32_BIT(CCM_CS2CDR,                0x53FD402C,__READ_WRITE ,__ccm_cs2cdr_bits);
__IO_REG32_BIT(CCM_CDCDR,                 0x53FD4030,__READ_WRITE ,__ccm_cdcdr_bits);
__IO_REG32_BIT(CCM_CHSCCDR,               0x53FD4034,__READ_WRITE ,__ccm_chsccdr_bits);
__IO_REG32_BIT(CCM_CSCDR2,                0x53FD4038,__READ_WRITE ,__ccm_cscdr2_bits);
__IO_REG32_BIT(CCM_CSCDR3,                0x53FD403C,__READ_WRITE ,__ccm_cscdr3_bits);
__IO_REG32_BIT(CCM_CSCDR4,                0x53FD4040,__READ_WRITE ,__ccm_cscdr4_bits);
//__IO_REG32_BIT(CCM_CWDR,                  0x53FD4044,__READ_WRITE ,__ccm_cwdr_bits);
__IO_REG32_BIT(CCM_CDHIPR,                0x53FD4048,__READ_WRITE ,__ccm_cdhipr_bits);
__IO_REG32_BIT(CCM_CDCR,                  0x53FD404C,__READ_WRITE ,__ccm_cdcr_bits);
__IO_REG32_BIT(CCM_CTOR,                  0x53FD4050,__READ_WRITE ,__ccm_ctor_bits);
__IO_REG32_BIT(CCM_CLPCR,                 0x53FD4054,__READ_WRITE ,__ccm_clpcr_bits);
__IO_REG32_BIT(CCM_CISR,                  0x53FD4058,__READ_WRITE ,__ccm_cisr_bits);
__IO_REG32_BIT(CCM_CIMR,                  0x53FD405C,__READ_WRITE ,__ccm_cimr_bits);
__IO_REG32_BIT(CCM_CCOSR,                 0x53FD4060,__READ_WRITE ,__ccm_ccosr_bits);
__IO_REG32_BIT(CCM_CGPR,                  0x53FD4064,__READ_WRITE ,__ccm_cgpr_bits);
__IO_REG32_BIT(CCM_CCGR0,                 0x53FD4068,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR1,                 0x53FD406C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR2,                 0x53FD4070,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR3,                 0x53FD4074,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR4,                 0x53FD4078,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR5,                 0x53FD407C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR6,                 0x53FD4080,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR7,                 0x53FD4084,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CMEOR,                 0x53FD4088,__READ_WRITE ,__ccm_cmeor_bits);

/***************************************************************************
 **
 **  CSPI
 **
 ***************************************************************************/
__IO_REG32(    CSPI_RXDATA,               0x63FC0000,__READ       );
__IO_REG32(    CSPI_TXDATA,               0x63FC0004,__WRITE      );
__IO_REG32_BIT(CSPI_CONREG,               0x63FC0008,__READ_WRITE ,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI_INTREG,               0x63FC000C,__READ_WRITE ,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI_DMAREG,               0x63FC0010,__READ_WRITE ,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI_STATREG,              0x63FC0014,__READ_WRITE ,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI_PERIODREG,            0x63FC0018,__READ_WRITE ,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI_TESTREG,              0x63FC001C,__READ_WRITE ,__cspi_testreg_bits);

/***************************************************************************
 **
 **  DPLLC1
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC1_DP_CTL,             0x63F80000,__READ_WRITE ,__dpllc_dp_ctl_bits);
__IO_REG32_BIT(DPLLC1_DP_CONFIG,          0x63F80004,__READ_WRITE ,__dpllc_dp_config_bits);
__IO_REG32_BIT(DPLLC1_DP_OP,              0x63F80008,__READ_WRITE ,__dpllc_dp_op_bits);
__IO_REG32_BIT(DPLLC1_DP_MFD,             0x63F8000C,__READ_WRITE ,__dpllc_dp_mfd_bits);
__IO_REG32_BIT(DPLLC1_DP_MFN,             0x63F80010,__READ_WRITE ,__dpllc_dp_mfn_bits);
__IO_REG32_BIT(DPLLC1_DP_MFNMINUS,        0x63F80014,__READ_WRITE ,__dpllc_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLC1_DP_MFNPLUS,         0x63F80018,__READ_WRITE ,__dpllc_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLC1_DP_HFS_OP,          0x63F8001C,__READ_WRITE ,__dpllc_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLC1_DP_HFS_MFD,         0x63F80020,__READ_WRITE ,__dpllc_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC1_DP_HFS_MFN,         0x63F80024,__READ_WRITE ,__dpllc_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC1_DP_MFN_TOGC,        0x63F80028,__READ_WRITE ,__dpllc_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLC1_DP_DESTAT,          0x63F8002C,__READ_WRITE ,__dpllc_dp_destat_bits);

/***************************************************************************
 **
 **  DPLLC2
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC2_DP_CTL,             0x63F84000,__READ_WRITE ,__dpllc_dp_ctl_bits);
__IO_REG32_BIT(DPLLC2_DP_CONFIG,          0x63F84004,__READ_WRITE ,__dpllc_dp_config_bits);
__IO_REG32_BIT(DPLLC2_DP_OP,              0x63F84008,__READ_WRITE ,__dpllc_dp_op_bits);
__IO_REG32_BIT(DPLLC2_DP_MFD,             0x63F8400C,__READ_WRITE ,__dpllc_dp_mfd_bits);
__IO_REG32_BIT(DPLLC2_DP_MFN,             0x63F84010,__READ_WRITE ,__dpllc_dp_mfn_bits);
__IO_REG32_BIT(DPLLC2_DP_MFNMINUS,        0x63F84014,__READ_WRITE ,__dpllc_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLC2_DP_MFNPLUS,         0x63F84018,__READ_WRITE ,__dpllc_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLC2_DP_HFS_OP,          0x63F8401C,__READ_WRITE ,__dpllc_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLC2_DP_HFS_MFD,         0x63F84020,__READ_WRITE ,__dpllc_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC2_DP_HFS_MFN,         0x63F84024,__READ_WRITE ,__dpllc_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC2_DP_MFN_TOGC,        0x63F84028,__READ_WRITE ,__dpllc_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLC2_DP_DESTAT,          0x63F8402C,__READ_WRITE ,__dpllc_dp_destat_bits);

/***************************************************************************
 **
 **  DPLLC3
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC3_DP_CTL,             0x63F88000,__READ_WRITE ,__dpllc_dp_ctl_bits);
__IO_REG32_BIT(DPLLC3_DP_CONFIG,          0x63F88004,__READ_WRITE ,__dpllc_dp_config_bits);
__IO_REG32_BIT(DPLLC3_DP_OP,              0x63F88008,__READ_WRITE ,__dpllc_dp_op_bits);
__IO_REG32_BIT(DPLLC3_DP_MFD,             0x63F8800C,__READ_WRITE ,__dpllc_dp_mfd_bits);
__IO_REG32_BIT(DPLLC3_DP_MFN,             0x63F88010,__READ_WRITE ,__dpllc_dp_mfn_bits);
__IO_REG32_BIT(DPLLC3_DP_MFNMINUS,        0x63F88014,__READ_WRITE ,__dpllc_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLC3_DP_MFNPLUS,         0x63F88018,__READ_WRITE ,__dpllc_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLC3_DP_HFS_OP,          0x63F8801C,__READ_WRITE ,__dpllc_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLC3_DP_HFS_MFD,         0x63F88020,__READ_WRITE ,__dpllc_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC3_DP_HFS_MFN,         0x63F88024,__READ_WRITE ,__dpllc_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC3_DP_MFN_TOGC,        0x63F88028,__READ_WRITE ,__dpllc_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLC3_DP_DESTAT,          0x63F8802C,__READ_WRITE ,__dpllc_dp_destat_bits);

/***************************************************************************
 **
 **  DPLLC4
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC4_DP_CTL,             0x63F8C000,__READ_WRITE ,__dpllc_dp_ctl_bits);
__IO_REG32_BIT(DPLLC4_DP_CONFIG,          0x63F8C004,__READ_WRITE ,__dpllc_dp_config_bits);
__IO_REG32_BIT(DPLLC4_DP_OP,              0x63F8C008,__READ_WRITE ,__dpllc_dp_op_bits);
__IO_REG32_BIT(DPLLC4_DP_MFD,             0x63F8C00C,__READ_WRITE ,__dpllc_dp_mfd_bits);
__IO_REG32_BIT(DPLLC4_DP_MFN,             0x63F8C010,__READ_WRITE ,__dpllc_dp_mfn_bits);
__IO_REG32_BIT(DPLLC4_DP_MFNMINUS,        0x63F8C014,__READ_WRITE ,__dpllc_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLC4_DP_MFNPLUS,         0x63F8C018,__READ_WRITE ,__dpllc_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLC4_DP_HFS_OP,          0x63F8C01C,__READ_WRITE ,__dpllc_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLC4_DP_HFS_MFD,         0x63F8C020,__READ_WRITE ,__dpllc_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC4_DP_HFS_MFN,         0x63F8C024,__READ_WRITE ,__dpllc_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC4_DP_MFN_TOGC,        0x63F8C028,__READ_WRITE ,__dpllc_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLC4_DP_DESTAT,          0x63F8C02C,__READ_WRITE ,__dpllc_dp_destat_bits);

/***************************************************************************
 **
 **  DVFSC
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSC_DVFSTHRS,            0x53FD8180,__READ_WRITE ,__dvfsc_dvfsthrs_bits);
__IO_REG32_BIT(DVFSC_DVFSCOUN,            0x53FD8184,__READ_WRITE ,__dvfsc_dvfscoun_bits);
__IO_REG32_BIT(DVFSC_DVFSSIG1,            0x53FD8188,__READ_WRITE ,__dvfsc_dvfssig1_bits);
__IO_REG32_BIT(DVFSC_DVFSSIG0,            0x53FD818C,__READ_WRITE ,__dvfsc_dvfssig0_bits);
__IO_REG32_BIT(DVFSC_DVFSGPC0,            0x53FD8190,__READ_WRITE ,__dvfsc_dvfsgpc0_bits);
__IO_REG32_BIT(DVFSC_DVFSGPC1,            0x53FD8194,__READ_WRITE ,__dvfsc_dvfsgpc1_bits);
__IO_REG32_BIT(DVFSC_DVFSGPBT,            0x53FD8198,__READ_WRITE ,__dvfsc_dvfsgpbt_bits);
__IO_REG32_BIT(DVFSC_DVFSEMAC,            0x53FD819C,__READ_WRITE ,__dvfsc_dvfsemac_bits);
__IO_REG32_BIT(DVFSC_DVFSCNTR,            0x53FD81A0,__READ_WRITE ,__dvfsc_dvfscntr_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR0_0,          0x53FD81A4,__READ_WRITE ,__dvfsc_dvfsltr0_0_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR0_1,          0x53FD81A8,__READ_WRITE ,__dvfsc_dvfsltr0_1_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR1_0,          0x53FD81AC,__READ_WRITE ,__dvfsc_dvfsltr1_0_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR1_1,          0x53FD81B0,__READ_WRITE ,__dvfsc_dvfsltr1_1_bits);
__IO_REG32_BIT(DVFSC_DVFSPT0,             0x53FD81B4,__READ_WRITE ,__dvfsc_dvfspt0_bits);
__IO_REG32_BIT(DVFSC_DVFSPT1,             0x53FD81B8,__READ_WRITE ,__dvfsc_dvfspt1_bits);
__IO_REG32_BIT(DVFSC_DVFSPT2,             0x53FD81BC,__READ_WRITE ,__dvfsc_dvfspt2_bits);
__IO_REG32_BIT(DVFSC_DVFSPT3,             0x53FD81C0,__READ_WRITE ,__dvfsc_dvfspt3_bits);

/***************************************************************************
 **
 **  DVFSP
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSP_LTR0,                0x53FD81C4,__READ       ,__dvfsp_ltr0_bits);
__IO_REG32_BIT(DVFSP_LTR1,                0x53FD81C8,__READ       ,__dvfsp_ltr1_bits);
__IO_REG32_BIT(DVFSP_LTR2,                0x53FD81CC,__READ       ,__dvfsp_ltr2_bits);
__IO_REG32_BIT(DVFSP_LTR3,                0x53FD81D0,__READ       ,__dvfsp_ltr3_bits);
__IO_REG32_BIT(DVFSP_LTBR0,               0x53FD81D4,__READ_WRITE ,__dvfsp_ltbr0_bits);
__IO_REG32_BIT(DVFSP_LTBR1,               0x53FD81D8,__READ_WRITE ,__dvfsp_ltbr1_bits);
__IO_REG32_BIT(DVFSP_PMCR0,               0x53FD81DC,__READ_WRITE ,__dvfsp_pmcr0_bits);
__IO_REG32_BIT(DVFSP_PMCR1,               0x53FD81E0,__READ_WRITE ,__dvfsp_pmcr1_bits);

/***************************************************************************
 **
 **  ECSPI1
 **
 ***************************************************************************/
__IO_REG32(    ECSPI1_RXDATA,             0x50010000,__READ       );
__IO_REG32(    ECSPI1_TXDATA,             0x50010004,__WRITE      );
__IO_REG32_BIT(ECSPI1_CONREG,             0x50010008,__READ_WRITE ,__ecspi_conreg_bits);
__IO_REG32_BIT(ECSPI1_CONFIGREG,          0x5001000C,__READ_WRITE ,__ecspi_configreg_bits);
__IO_REG32_BIT(ECSPI1_INTREG,             0x50010010,__READ_WRITE ,__ecspi_intreg_bits);
__IO_REG32_BIT(ECSPI1_DMAREG,             0x50010014,__READ_WRITE ,__ecspi_dmareg_bits);
__IO_REG32_BIT(ECSPI1_STATREG,            0x50010018,__READ_WRITE ,__ecspi_statreg_bits);
__IO_REG32_BIT(ECSPI1_PERIODREG,          0x5001001C,__READ_WRITE ,__ecspi_periodreg_bits);
__IO_REG32_BIT(ECSPI1_TESTREG,            0x50010020,__READ_WRITE ,__ecspi_testreg_bits);
__IO_REG32(    ECSPI1_MSGDATA,            0x50010040,__WRITE      );

/***************************************************************************
 **
 **  ECSPI2
 **
 ***************************************************************************/
__IO_REG32(    ECSPI2_RXDATA,             0x63FAC000,__READ       );
__IO_REG32(    ECSPI2_TXDATA,             0x63FAC004,__WRITE      );
__IO_REG32_BIT(ECSPI2_CONREG,             0x63FAC008,__READ_WRITE ,__ecspi_conreg_bits);
__IO_REG32_BIT(ECSPI2_CONFIGREG,          0x63FAC00C,__READ_WRITE ,__ecspi_configreg_bits);
__IO_REG32_BIT(ECSPI2_INTREG,             0x63FAC010,__READ_WRITE ,__ecspi_intreg_bits);
__IO_REG32_BIT(ECSPI2_DMAREG,             0x63FAC014,__READ_WRITE ,__ecspi_dmareg_bits);
__IO_REG32_BIT(ECSPI2_STATREG,            0x63FAC018,__READ_WRITE ,__ecspi_statreg_bits);
__IO_REG32_BIT(ECSPI2_PERIODREG,          0x63FAC01C,__READ_WRITE ,__ecspi_periodreg_bits);
__IO_REG32_BIT(ECSPI2_TESTREG,            0x63FAC020,__READ_WRITE ,__ecspi_testreg_bits);
__IO_REG32(    ECSPI2_MSGDATA,            0x63FAC040,__WRITE      );

/***************************************************************************
 **
 **  EIM
 **
 ***************************************************************************/
__IO_REG32_BIT(EIM_CS0GCR1,               0x63FDA000,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS0GCR2,               0x63FDA004,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS0RCR1,               0x63FDA008,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS0RCR2,               0x63FDA00C,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS0WCR1,               0x63FDA010,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS0WCR2,               0x63FDA014,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS1GCR1,               0x63FDA018,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS1GCR2,               0x63FDA01C,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS1RCR1,               0x63FDA020,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS1RCR2,               0x63FDA024,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS1WCR1,               0x63FDA028,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS1WCR2,               0x63FDA02C,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS2GCR1,               0x63FDA030,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS2GCR2,               0x63FDA034,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS2RCR1,               0x63FDA038,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS2RCR2,               0x63FDA03C,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS2WCR1,               0x63FDA040,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS2WCR2,               0x63FDA044,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS3GCR1,               0x63FDA048,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS3GCR2,               0x63FDA04C,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS3RCR1,               0x63FDA050,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS3RCR2,               0x63FDA054,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS3WCR1,               0x63FDA058,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS3WCR2,               0x63FDA05C,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS4GCR1,               0x63FDA060,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS4GCR2,               0x63FDA064,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS4RCR1,               0x63FDA068,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS4RCR2,               0x63FDA06C,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS4WCR1,               0x63FDA070,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS4WCR2,               0x63FDA074,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS5GCR1,               0x63FDA078,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS5GCR2,               0x63FDA07C,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS5RCR1,               0x63FDA080,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS5RCR2,               0x63FDA084,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS5WCR1,               0x63FDA088,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS5WCR2,               0x63FDA08C,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_WCR,                   0x63FDA090,__READ_WRITE ,__eim_wcr_bits);
__IO_REG32_BIT(EIM_WIAR,                  0x63FDA094,__READ_WRITE ,__eim_wiar_bits);
__IO_REG32(    EIM_EAR,                   0x63FDA098,__READ_WRITE );

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT1_EPITCR,              0x53FAC000,__READ_WRITE ,__epit_epitcr_bits);
__IO_REG32_BIT(EPIT1_EPITSR,              0x53FAC004,__READ_WRITE ,__epit_epitsr_bits);
__IO_REG32(    EPIT1_EPITLR,              0x53FAC008,__READ_WRITE );
__IO_REG32(    EPIT1_EPITCMPR,            0x53FAC00C,__READ_WRITE );
__IO_REG32(    EPIT1_EPITCNR,             0x53FAC010,__READ       );

/***************************************************************************
 **
 **  EPIT2
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT2_EPITCR,              0x53FB0000,__READ_WRITE ,__epit_epitcr_bits);
__IO_REG32_BIT(EPIT2_EPITSR,              0x53FB0004,__READ_WRITE ,__epit_epitsr_bits);
__IO_REG32(    EPIT2_EPITLR,              0x53FB0008,__READ_WRITE );
__IO_REG32(    EPIT2_EPITCMPR,            0x53FB000C,__READ_WRITE );
__IO_REG32(    EPIT2_EPITCNR,             0x53FB0010,__READ       );

/***************************************************************************
 **
 **  ESAI
 **
 ***************************************************************************/
__IO_REG32(    ESAI_ETDR,                 0x50018000,__WRITE      );
__IO_REG32(    ESAI_ERDR,                 0x50018004,__READ       );
__IO_REG32_BIT(ESAI_ECR,                  0x50018008,__READ_WRITE ,__esai_ecr_bits);
__IO_REG32_BIT(ESAI_ESR,                  0x5001800C,__READ       ,__esai_esr_bits);
__IO_REG32_BIT(ESAI_TFCR,                 0x50018010,__READ_WRITE ,__esai_etdr_bits);
__IO_REG32_BIT(ESAI_TFSR,                 0x50018014,__READ       ,__esai_tfsr_bits);
__IO_REG32_BIT(ESAI_RFCR,                 0x50018018,__READ_WRITE ,__esai_rfcr_bits);
__IO_REG32_BIT(ESAI_RFSR,                 0x5001801C,__READ       ,__esai_rfsr_bits);
__IO_REG32_BIT(ESAI_TX0,                  0x50018080,__WRITE      ,__esai_tx0_bits);
__IO_REG32_BIT(ESAI_TX1,                  0x50018084,__WRITE      ,__esai_tx1_bits);
__IO_REG32_BIT(ESAI_TX2,                  0x50018088,__WRITE      ,__esai_tx2_bits);
__IO_REG32_BIT(ESAI_TX3,                  0x5001808C,__WRITE      ,__esai_tx3_bits);
__IO_REG32_BIT(ESAI_TX4,                  0x50018090,__WRITE      ,__esai_tx4_bits);
__IO_REG32_BIT(ESAI_TX5,                  0x50018094,__WRITE      ,__esai_tx5_bits);
__IO_REG32_BIT(ESAI_TSR,                  0x50018098,__WRITE      ,__esai_tsr_bits);
__IO_REG32_BIT(ESAI_RX0,                  0x500180A0,__READ       ,__esai_rx0_bits);
__IO_REG32_BIT(ESAI_RX1,                  0x500180A4,__READ       ,__esai_rx1_bits);
__IO_REG32_BIT(ESAI_RX2,                  0x500180A8,__READ       ,__esai_rx2_bits);
__IO_REG32_BIT(ESAI_RX3,                  0x500180AC,__READ       ,__esai_rx3_bits);
__IO_REG32_BIT(ESAI_SAISR,                0x500180CC,__READ       ,__esai_saisr_bits);
__IO_REG32_BIT(ESAI_SAICR,                0x500180D0,__READ_WRITE ,__esai_saicr_bits);
__IO_REG32_BIT(ESAI_TCR,                  0x500180D4,__READ_WRITE ,__esai_tcr_bits);
__IO_REG32_BIT(ESAI_TCCR,                 0x500180D8,__READ_WRITE ,__esai_tccr_bits);
__IO_REG32_BIT(ESAI_RCR,                  0x500180DC,__READ_WRITE ,__esai_rcr_bits);
__IO_REG32_BIT(ESAI_RCCR,                 0x500180E0,__READ_WRITE ,__esai_rccr_bits);
__IO_REG32_BIT(ESAI_TSMA,                 0x500180E4,__READ_WRITE ,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_TSMB,                 0x500180E8,__READ_WRITE ,__esai_tsm_bits);
__IO_REG32_BIT(ESAI_RSMA,                 0x500180EC,__READ_WRITE ,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_RSMB,                 0x500180F0,__READ_WRITE ,__esai_rsm_bits);
__IO_REG32_BIT(ESAI_PRRC,                 0x500180F8,__READ_WRITE ,__esai_prrc_bits);
__IO_REG32_BIT(ESAI_PCRC,                 0x500180FC,__READ_WRITE ,__esai_pcrc_bits);

/***************************************************************************
 **
 **  ESDCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL,                    0x63FD9000,__READ_WRITE ,__esdctl_bits);
__IO_REG32_BIT(ESDCTL_ESDPDC,             0x63FD9004,__READ_WRITE ,__esdctl_esdpdc_bits);
__IO_REG32_BIT(ESDCTL_ESDOTC,             0x63FD9008,__READ_WRITE ,__esdctl_esdotc_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG0,            0x63FD900C,__READ_WRITE ,__esdctl_esdcfg0_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG1,            0x63FD9010,__READ_WRITE ,__esdctl_esdcfg1_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG2,            0x63FD9014,__READ_WRITE ,__esdctl_esdcfg2_bits);
__IO_REG32_BIT(ESDMISC,                   0x63FD9018,__READ_WRITE ,__esdmisc_bits);
__IO_REG32_BIT(ESDCTL_ESDSCR,             0x63FD901C,__READ_WRITE ,__esdctl_esdscr_bits);
__IO_REG32_BIT(ESDCTL_ESDREF,             0x63FD9020,__READ_WRITE ,__esdctl_esdref_bits);
__IO_REG32_BIT(ESDCTL_ESDWCC,             0x63FD9024,__READ_WRITE ,__esdctl_esdwcc_bits);
__IO_REG32_BIT(ESDCTL_ESDRCC,             0x63FD9028,__READ_WRITE ,__esdctl_esdrcc_bits);
__IO_REG32_BIT(ESDCTL_ESDRWD,             0x63FD902C,__READ_WRITE ,__esdctl_esdrwd_bits);
__IO_REG32_BIT(ESDCTL_ESDOR,              0x63FD9030,__READ_WRITE ,__esdctl_esdor_bits);
__IO_REG32_BIT(ESDCTL_ESDMRR,             0x63FD9034,__READ       ,__esdctl_esdmrr_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG3_LP,         0x63FD9038,__READ_WRITE ,__esdctl_esdcfg3_lp_bits);
__IO_REG32_BIT(ESDCTL_ESDMR4,             0x63FD903C,__READ_WRITE ,__esdctl_esdmr4_bits);
__IO_REG32_BIT(ZQHWCTRL,                  0x63FD9040,__READ_WRITE ,__zqhwctrl_bits);
__IO_REG32_BIT(ESDCTL_ZQSWCTRL,           0x63FD9044,__READ_WRITE ,__esdctl_zqswctrl_bits);
__IO_REG32_BIT(ESDCTL_WLGCR,              0x63FD9048,__READ_WRITE ,__esdctl_wlgcr_bits);
__IO_REG32_BIT(ESDCTL_WLDECTRL0,          0x63FD904C,__READ_WRITE ,__esdctl_wldectrl0_bits);
__IO_REG32_BIT(ESDCTL_WLDECTRL1,          0x63FD9050,__READ_WRITE ,__esdctl_wldectrl1_bits);
__IO_REG32_BIT(ESDCTL_WLDLST,             0x63FD9054,__READ       ,__esdctl_wldlst_bits);
__IO_REG32_BIT(ESDCTL_ODTCTRL,            0x63FD9058,__READ_WRITE ,__esdctl_odtctrl_bits);
__IO_REG32_BIT(ESDCTL_RDDQBY0DL,          0x63FD905C,__READ_WRITE ,__esdctl_rddqby0dl_bits);
__IO_REG32_BIT(ESDCTL_RDDQBY1DL,          0x63FD9060,__READ_WRITE ,__esdctl_rddqby1dl_bits);
__IO_REG32_BIT(ESDCTL_RDDQBY2DL,          0x63FD9064,__READ_WRITE ,__esdctl_rddqby2dl_bits);
__IO_REG32_BIT(ESDCTL_RDDQBY3DL,          0x63FD9068,__READ_WRITE ,__esdctl_rddqby3dl_bits);
__IO_REG32_BIT(ESDCTL_WRDQBY0DL,          0x63FD906C,__READ_WRITE ,__esdctl_wrdqby0dl_bits);
__IO_REG32_BIT(ESDCTL_WRDQBY1DL,          0x63FD9070,__READ_WRITE ,__esdctl_wrdqby1dl_bits);
__IO_REG32_BIT(ESDCTL_WRDQBY2DL,          0x63FD9074,__READ_WRITE ,__esdctl_wrdqby2dl_bits);
__IO_REG32_BIT(ESDCTL_WRDQBY3DL,          0x63FD9078,__READ_WRITE ,__esdctl_wrdqby3dl_bits);
__IO_REG32_BIT(ESDCTL_DGCTRL0,            0x63FD907C,__READ_WRITE ,__esdctl_dgctrl0_bits);
__IO_REG32_BIT(ESDCTL_DGCTRL1,            0x63FD9080,__READ_WRITE ,__esdctl_dgctrl1_bits);
__IO_REG32_BIT(ESDCTL_DGDLST,             0x63FD9084,__READ       ,__esdctl_dgdlst_bits);
__IO_REG32_BIT(ESDCTL_RDDLCTL,            0x63FD9088,__READ_WRITE ,__esdctl_rddlctl_bits);
__IO_REG32_BIT(ESDCTL_RDDLST,             0x63FD908C,__READ       ,__esdctl_rddlst_bits);
__IO_REG32_BIT(ESDCTL_WRDLCTL,            0x63FD9090,__READ_WRITE ,__esdctl_wrdlctl_bits);
__IO_REG32_BIT(ESDCTL_WRDLST,             0x63FD9094,__READ       ,__esdctl_wrdlst_bits);
__IO_REG32_BIT(ESDCTL_SDCTRL,             0x63FD9098,__READ_WRITE ,__esdctl_sdctrl_bits);
__IO_REG32_BIT(ESDCTL_ZQLP2CTL,           0x63FD909C,__READ_WRITE ,__esdctl_zqlp2ctl_bits);
__IO_REG32_BIT(ESDCTL_RDDLHWCTL,          0x63FD90A0,__READ_WRITE ,__esdctl_rddlhwctl_bits);
__IO_REG32_BIT(ESDCTL_WRDLHWCTL,          0x63FD90A4,__READ_WRITE ,__esdctl_wrdlhwctl_bits);
__IO_REG32_BIT(ESDCTL_RDDLHWST0,          0x63FD90A8,__READ       ,__esdctl_rddlhwst0_bits);
__IO_REG32_BIT(ESDCTL_RDDLHWST1,          0x63FD90AC,__READ       ,__esdctl_rddlhwst1_bits);
__IO_REG32_BIT(ESDCTL_WRDLHWST0,          0x63FD90B0,__READ       ,__esdctl_wrdlhwst0_bits);
__IO_REG32_BIT(ESDCTL_WRDLHWST1,          0x63FD90B4,__READ       ,__esdctl_wrdlhwst1_bits);
__IO_REG32_BIT(ESDCTL_WLHWERR,            0x63FD90B8,__READ_WRITE ,__esdctl_wlhwerr_bits);
__IO_REG32_BIT(ESDCTL_DGHWST0,            0x63FD90BC,__READ_WRITE ,__esdctl_dghwst0_bits);
__IO_REG32_BIT(ESDCTL_DGHWST1,            0x63FD90C0,__READ       ,__esdctl_dghwst1_bits);
__IO_REG32_BIT(ESDCTL_DGHWST2,            0x63FD90C4,__READ       ,__esdctl_dghwst2_bits);
__IO_REG32_BIT(ESDCTL_DGHWST3,            0x63FD90C8,__READ       ,__esdctl_dghwst3_bits);
__IO_REG32_BIT(ESDCTL_PDCMPR1,            0x63FD90CC,__READ_WRITE ,__esdctl_pdcmpr1_bits);
__IO_REG32_BIT(ESDCTL_PDCMPR2,            0x63FD90D0,__READ_WRITE ,__esdctl_pdcmpr2_bits);
__IO_REG32_BIT(ESDCTL_SWDAR,              0x63FD90D4,__READ_WRITE ,__esdctl_swdar_bits);
__IO_REG32(    ESDCTL_SWDRDR0,            0x63FD90D8,__READ       );
__IO_REG32(    ESDCTL_SWDRDR1,            0x63FD90DC,__READ       );
__IO_REG32(    ESDCTL_SWDRDR2,            0x63FD90E0,__READ       );
__IO_REG32(    ESDCTL_SWDRDR3,            0x63FD90E4,__READ       );
__IO_REG32(    ESDCTL_SWDRDR4,            0x63FD90E8,__READ       );
__IO_REG32(    ESDCTL_SWDRDR5,            0x63FD90EC,__READ       );
__IO_REG32(    ESDCTL_SWDRDR6,            0x63FD90F0,__READ       );
__IO_REG32(    ESDCTL_SWDRDR7,            0x63FD90F4,__READ       );
__IO_REG32_BIT(ESDCTL_MUR,                0x63FD90F8,__READ_WRITE ,__esdctl_mur_bits);
__IO_REG32_BIT(ESDCTL_WRCADL,             0x63FD90FC,__READ_WRITE ,__esdctl_wrcadl_bits);

/***************************************************************************
 **
 **  ESDHC1
 **
 ***************************************************************************/
__IO_REG32(    ESDHC1_DSADDR,             0x50004000,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_BLKATTR,            0x50004004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC1_CMDARG,             0x50004008,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_XFERTYP,            0x5000400C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC1_CMDRSP0,            0x50004010,__READ       );
__IO_REG32(    ESDHC1_CMDRSP1,            0x50004014,__READ       );
__IO_REG32(    ESDHC1_CMDRSP2,            0x50004018,__READ       );
__IO_REG32(    ESDHC1_CMDRSP3,            0x5000401C,__READ       );
__IO_REG32(    ESDHC1_DATPORT,            0x50004020,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_PRSSTAT,            0x50004024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC1_PROCTL,             0x50004028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC1_SYSCTL,             0x5000402C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC1_IRQSTAT,            0x50004030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC1_IRQSTATEN,          0x50004034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC1_IRQSIGEN,           0x50004038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC1_AUTOC12ERR,         0x5000403C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC1_HOSTCAPBLT,         0x50004040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC1_WML,                0x50004044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC1_FEVT,               0x50004050,__WRITE      ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC1_ADMAES,             0x50004054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC1_ADSADDR,            0x50004058,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_VENDOR,             0x500040C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC1_MMCBOOT,            0x500040C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC1_HOSTVER,            0x500040FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC2
 **
 ***************************************************************************/
__IO_REG32(    ESDHC2_DSADDR,             0x50008000,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_BLKATTR,            0x50008004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC2_CMDARG,             0x50008008,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_XFERTYP,            0x5000800C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC2_CMDRSP0,            0x50008010,__READ       );
__IO_REG32(    ESDHC2_CMDRSP1,            0x50008014,__READ       );
__IO_REG32(    ESDHC2_CMDRSP2,            0x50008018,__READ       );
__IO_REG32(    ESDHC2_CMDRSP3,            0x5000801C,__READ       );
__IO_REG32(    ESDHC2_DATPORT,            0x50008020,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_PRSSTAT,            0x50008024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC2_PROCTL,             0x50008028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC2_SYSCTL,             0x5000802C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC2_IRQSTAT,            0x50008030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC2_IRQSTATEN,          0x50008034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC2_IRQSIGEN,           0x50008038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC2_AUTOC12ERR,         0x5000803C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC2_HOSTCAPBLT,         0x50008040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC2_WML,                0x50008044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC2_FEVT,               0x50008050,__WRITE      ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC2_ADMAES,             0x50008054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC2_ADSADDR,            0x50008058,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_VENDOR,             0x500080C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC2_MMCBOOT,            0x500080C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC2_HOSTVER,            0x500080FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC3
 **
 ***************************************************************************/
__IO_REG32(    ESDHC3_DSADDR,             0x50020000,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_BLKATTR,            0x50020004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC3_CMDARG,             0x50020008,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_XFERTYP,            0x5002000C,__READ_WRITE ,__esdhcv3_xfertyp_bits);
__IO_REG32(    ESDHC3_CMDRSP0,            0x50020010,__READ       );
__IO_REG32(    ESDHC3_CMDRSP1,            0x50020014,__READ       );
__IO_REG32(    ESDHC3_CMDRSP2,            0x50020018,__READ       );
__IO_REG32(    ESDHC3_CMDRSP3,            0x5002001C,__READ       );
__IO_REG32(    ESDHC3_DATPORT,            0x50020020,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_PRSSTAT,            0x50020024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC3_PROCTL,             0x50020028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC3_SYSCTL,             0x5002002C,__READ_WRITE ,__esdhcv3_sysctl_bits);
__IO_REG32_BIT(ESDHC3_IRQSTAT,            0x50020030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC3_IRQSTATEN,          0x50020034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC3_IRQSIGEN,           0x50020038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC3_AUTOC12ERR,         0x5002003C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC3_HOSTCAPBLT,         0x50020040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC3_WML,                0x50020044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC3_FEVT,               0x50020050,__WRITE      ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC3_ADMAES,             0x50020054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC3_ADSADDR,            0x50020058,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_DLLCTRL,            0x50020060,__READ_WRITE ,__esdhcv3_dllctrl_bits);
__IO_REG32_BIT(ESDHC3_DLLSTS,             0x50020064,__READ_WRITE ,__esdhcv3_dllsts_bits);
__IO_REG32_BIT(ESDHC3_VENDOR,             0x500200C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC3_MMCBOOT,            0x500200C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC3_HOSTVER,            0x500200FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC4
 **
 ***************************************************************************/
__IO_REG32(    ESDHC4_DSADDR,             0x50024000,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_BLKATTR,            0x50024004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC4_CMDARG,             0x50024008,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_XFERTYP,            0x5002400C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC4_CMDRSP0,            0x50024010,__READ       );
__IO_REG32(    ESDHC4_CMDRSP1,            0x50024014,__READ       );
__IO_REG32(    ESDHC4_CMDRSP2,            0x50024018,__READ       );
__IO_REG32(    ESDHC4_CMDRSP3,            0x5002401C,__READ       );
__IO_REG32(    ESDHC4_DATPORT,            0x50024020,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_PRSSTAT,            0x50024024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC4_PROCTL,             0x50024028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC4_SYSCTL,             0x5002402C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC4_IRQSTAT,            0x50024030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC4_IRQSTATEN,          0x50024034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC4_IRQSIGEN,           0x50024038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC4_AUTOC12ERR,         0x5002403C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC4_HOSTCAPBLT,         0x50024040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC4_WML,                0x50024044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32_BIT(ESDHC4_FEVT,               0x50024050,__WRITE      ,__esdhc_fevt_bits);
__IO_REG32_BIT(ESDHC4_ADMAES,             0x50024054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC4_ADSADDR,            0x50024058,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_VENDOR,             0x500240C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC4_MMCBOOT,            0x500240C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC4_HOSTVER,            0x500240FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  EXTMC
 **
 ***************************************************************************/
__IO_REG32_BIT(EXTMC_IPLCK,               0x63FDBF00,__READ_WRITE ,__extmc_iplck_bits);
__IO_REG32_BIT(EXTMC_EICS,                0x63FDBF04,__READ_WRITE ,__extmc_eics_bits);

 /***************************************************************************
 **
 **  FEC
 **
 ***************************************************************************/
__IO_REG32_BIT(FEC_EIR,                   0x63FEC004,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_EIMR,                  0x63FEC008,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_RDAR,                  0x63FEC010,__READ_WRITE,__fec_rdar_bits);
__IO_REG32_BIT(FEC_TDAR,                  0x63FEC014,__READ_WRITE,__fec_tdar_bits);
__IO_REG32_BIT(FEC_ECR,                   0x63FEC024,__READ_WRITE,__fec_ecr_bits);
__IO_REG32_BIT(FEC_MMFR,                  0x63FEC040,__READ_WRITE,__fec_mmfr_bits);
__IO_REG32_BIT(FEC_MSCR,                  0x63FEC044,__READ_WRITE,__fec_mscr_bits);
__IO_REG32_BIT(FEC_MIBC,                  0x63FEC064,__READ_WRITE,__fec_mibc_bits);
__IO_REG32_BIT(FEC_RCR,                   0x63FEC084,__READ_WRITE,__fec_rcr_bits);
__IO_REG32_BIT(FEC_TCR,                   0x63FEC0C4,__READ_WRITE,__fec_tcr_bits);
__IO_REG32(    FEC_PALR,                  0x63FEC0E4,__READ_WRITE);
__IO_REG32_BIT(FEC_PAUR,                  0x63FEC0E8,__READ_WRITE,__fec_paur_bits);
__IO_REG32_BIT(FEC_OPD,                   0x63FEC0EC,__READ_WRITE,__fec_opd_bits);
__IO_REG32(    FEC_IAUR,                  0x63FEC118,__READ_WRITE);
__IO_REG32(    FEC_IALR,                  0x63FEC11C,__READ_WRITE);
__IO_REG32(    FEC_GAUR,                  0x63FEC120,__READ_WRITE);
__IO_REG32(    FEC_GALR,                  0x63FEC124,__READ_WRITE);
__IO_REG32_BIT(FEC_TFWR,                  0x63FEC144,__READ_WRITE,__fec_tfwr_bits);
__IO_REG32_BIT(FEC_FRBR,                  0x63FEC14C,__READ      ,__fec_frbr_bits);
__IO_REG32_BIT(FEC_FRSR,                  0x63FEC150,__READ_WRITE,__fec_frsr_bits);
__IO_REG32(    FEC_ERDSR,                 0x63FEC180,__READ_WRITE);
__IO_REG32(    FEC_ETDSR,                 0x63FEC184,__READ_WRITE);
__IO_REG32_BIT(FEC_EMRBR,                 0x63FEC188,__READ_WRITE,__fec_emrbr_bits);
__IO_REG32(    FEC_RMON_T_DROP,           0x63FEC200,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_PACKETS,        0x63FEC204,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_BC_PKT,         0x63FEC208,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_MC_PKT,         0x63FEC20C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_CRC_ALIGN,      0x63FEC210,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_UNDERSIZE,      0x63FEC214,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OVERSIZE,       0x63FEC218,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_FRAG,           0x63FEC21C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_JAB,            0x63FEC220,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_COL,            0x63FEC224,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P64,            0x63FEC228,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P65TO127,       0x63FEC22C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P128TO255,      0x63FEC230,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P256TO511,      0x63FEC234,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P512TO1023,     0x63FEC238,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P1024TO2047,    0x63FEC23C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P_GTE2048,      0x63FEC240,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OCTETS,         0x63FEC244,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DROP,           0x63FEC248,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FRAME_OK,       0x63FEC24C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_1COL,           0x63FEC250,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MCOL,           0x63FEC254,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DEF,            0x63FEC258,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_LCOL,           0x63FEC25C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_EXCOL,          0x63FEC260,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MACERR,         0x63FEC264,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_CSERR,          0x63FEC268,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_SQE,            0x63FEC26C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FDXFC,          0x63FEC270,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_OCTETS_OK,      0x63FEC274,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_PACKETS,        0x63FEC284,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_BC_PKT,         0x63FEC288,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_MC_PKT,         0x63FEC28C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_CRC_ALIGN,      0x63FEC290,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_UNDERSIZE,      0x63FEC294,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OVERSIZE,       0x63FEC298,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_FRAG,           0x63FEC29C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_JAB,            0x63FEC2A0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_RESVD_0,        0x63FEC2A4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P64,            0x63FEC2A8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P65TO127,       0x63FEC2AC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P128TO255,      0x63FEC2B0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P256TO511,      0x63FEC2B4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P512TO1023,     0x63FEC2B8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P1024TO2047,    0x63FEC2BC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P_GTE2048,      0x63FEC2C0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OCTETS,         0x63FEC2C4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_DROP,           0x63FEC2C8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FRAME_OK,       0x63FEC2CC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_CRC,            0x63FEC2D0,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_ALIGN,          0x63FEC2D4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_MACERR,         0x63FEC2D8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FDXFC,          0x63FEC2DC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_OCTETS_OK,      0x63FEC2E0,__READ_WRITE);
__IO_REG32_BIT(FEC_MIIGSK_CFGR,           0x63FEC300,__READ_WRITE,__fec_miigsk_cfgr_bits);
__IO_REG32_BIT(FEC_MIIGSK_ENR,            0x63FEC308,__READ_WRITE,__fec_miigsk_enr_bits);

/***************************************************************************
 **
 ** FIRI
 **
 ***************************************************************************/
__IO_REG32_BIT(FIRITCR,                   0x63FA8000,__READ_WRITE ,__firitcr_bits);
__IO_REG32_BIT(FIRITCTR,                  0x63FA8004,__READ_WRITE ,__firitctr_bits);
__IO_REG32_BIT(FIRIRCR,                   0x63FA8008,__READ_WRITE ,__firircr_bits);
__IO_REG32_BIT(FIRITSR,                   0x63FA800C,__READ_WRITE ,__firitsr_bits);
__IO_REG32_BIT(FIRIRSR,                   0x63FA8010,__READ_WRITE ,__firirsr_bits);
__IO_REG32(    FIRITXD,                   0x63FA8014,__WRITE      );
__IO_REG32(    FIRIRXD,                   0x63FA8018,__READ       );
__IO_REG32_BIT(FIRICR,                    0x63FA801C,__READ_WRITE ,__firicr_bits);

/***************************************************************************
 **
 **  FlexCAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(FLEXCAN1_MCR,                  0x53FC8000,__READ_WRITE ,__can_mcr_bits);
__IO_REG32_BIT(FLEXCAN1_CTRL,                 0x53FC8004,__READ_WRITE ,__can_ctrl_bits);
__IO_REG32_BIT(FLEXCAN1_TIMER,                0x53FC8008,__READ_WRITE ,__can_timer_bits);
__IO_REG32_BIT(FLEXCAN1_RXGMASK,              0x53FC8010,__READ_WRITE ,__can_rxgmask_bits);
__IO_REG32(    FLEXCAN1_RX14MASK,             0x53FC8014,__READ_WRITE );
__IO_REG32(    FLEXCAN1_RX15MASK,             0x53FC8018,__READ_WRITE );
__IO_REG32_BIT(FLEXCAN1_ECR,                  0x53FC801C,__READ_WRITE ,__can_ecr_bits);
__IO_REG32_BIT(FLEXCAN1_ESR,                  0x53FC8020,__READ_WRITE ,__can_esr_bits);
__IO_REG32_BIT(FLEXCAN1_IMASK2,               0x53FC8024,__READ_WRITE ,__can_imask2_bits);
__IO_REG32_BIT(FLEXCAN1_IMASK1,               0x53FC8028,__READ_WRITE ,__can_imask1_bits);
__IO_REG32_BIT(FLEXCAN1_IFLAG2,               0x53FC802C,__READ_WRITE ,__can_iflag2_bits);
__IO_REG32_BIT(FLEXCAN1_IFLAG1,               0x53FC8030,__READ_WRITE ,__can_iflag1_bits);
__IO_REG32_BIT(FLEXCAN1_GFWR,                 0x53FC8034,__READ_WRITE ,__can_gfwr_bits);
__IO_REG32(    FLEXCAN1_MB0_15_BASE_ADDR,     0x53FC8080,__READ_WRITE );
__IO_REG32(    FLEXCAN1_MB16_31_BASE_ADDR,    0x53FC8180,__READ_WRITE );
__IO_REG32(    FLEXCAN1_MB32_63_BASE_ADDR,    0x53FC8280,__READ_WRITE );
__IO_REG32(    FLEXCAN1_RXIMR0_15_BASE_ADDR,  0x53FC8880,__READ_WRITE );
__IO_REG32(    FLEXCAN1_RXIMR16_31_BASE_ADDR, 0x53FC88C0,__READ_WRITE );
__IO_REG32(    FLEXCAN1_RXIMR32_63_BASE_ADDR, 0x53FC8900,__READ_WRITE );

/***************************************************************************
 **
 **  FlexCAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(FLEXCAN2_MCR,                  0x53FCC000,__READ_WRITE ,__can_mcr_bits);
__IO_REG32_BIT(FLEXCAN2_CTRL,                 0x53FCC004,__READ_WRITE ,__can_ctrl_bits);
__IO_REG32_BIT(FLEXCAN2_TIMER,                0x53FCC008,__READ_WRITE ,__can_timer_bits);
__IO_REG32_BIT(FLEXCAN2_RXGMASK,              0x53FCC010,__READ_WRITE ,__can_rxgmask_bits);
__IO_REG32(    FLEXCAN2_RX14MASK,             0x53FCC014,__READ_WRITE );
__IO_REG32(    FLEXCAN2_RX15MASK,             0x53FCC018,__READ_WRITE );
__IO_REG32_BIT(FLEXCAN2_ECR,                  0x53FCC01C,__READ_WRITE ,__can_ecr_bits);
__IO_REG32_BIT(FLEXCAN2_ESR,                  0x53FCC020,__READ_WRITE ,__can_esr_bits);
__IO_REG32_BIT(FLEXCAN2_IMASK2,               0x53FCC024,__READ_WRITE ,__can_imask2_bits);
__IO_REG32_BIT(FLEXCAN2_IMASK1,               0x53FCC028,__READ_WRITE ,__can_imask1_bits);
__IO_REG32_BIT(FLEXCAN2_IFLAG2,               0x53FCC02C,__READ_WRITE ,__can_iflag2_bits);
__IO_REG32_BIT(FLEXCAN2_IFLAG1,               0x53FCC030,__READ_WRITE ,__can_iflag1_bits);
__IO_REG32_BIT(FLEXCAN2_GFWR,                 0x53FCC034,__READ_WRITE ,__can_gfwr_bits);
__IO_REG32(    FLEXCAN2_MB0_15_BASE_ADDR,     0x53FCC080,__READ_WRITE );
__IO_REG32(    FLEXCAN2_MB16_31_BASE_ADDR,    0x53FCC180,__READ_WRITE );
__IO_REG32(    FLEXCAN2_MB32_63_BASE_ADDR,    0x53FCC280,__READ_WRITE );
__IO_REG32(    FLEXCAN2_RXIMR0_15_BASE_ADDR,  0x53FCC880,__READ_WRITE );
__IO_REG32(    FLEXCAN2_RXIMR16_31_BASE_ADDR, 0x53FCC8C0,__READ_WRITE );
__IO_REG32(    FLEXCAN2_RXIMR32_63_BASE_ADDR, 0x53FCC900,__READ_WRITE );

/***************************************************************************
 **
 ** GPC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPC_CNTR,                  0x53FD8000,__READ_WRITE ,__gpc_cntr_bits);
//__IO_REG32_BIT(GPC_PGR,                   0x53FD8004,__READ_WRITE ,__gpc_pgr_bits);
__IO_REG32_BIT(GPC_VCR,                   0x53FD8008,__READ_WRITE ,__gpc_vcr_bits);
__IO_REG32_BIT(GPC_NEON,                  0x53FD8010,__READ_WRITE ,__gpc_neon_bits);

/***************************************************************************
 **
 **  GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_DR,                  0x53F84000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_GDIR,                0x53F84004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_PSR,                 0x53F84008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO1_ICR1,                0x53F8400C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO1_ICR2,                0x53F84010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO1_IMR,                 0x53F84014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_ISR,                 0x53F84018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_EDGE_SEL,            0x53F8401C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_DR,                  0x53F88000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_GDIR,                0x53F88004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_PSR,                 0x53F88008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO2_ICR1,                0x53F8800C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO2_ICR2,                0x53F88010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO2_IMR,                 0x53F88014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_ISR,                 0x53F88018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_EDGE_SEL,            0x53F8801C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_DR,                  0x53F8C000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_GDIR,                0x53F8C004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_PSR,                 0x53F8C008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO3_ICR1,                0x53F8C00C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO3_ICR2,                0x53F8C010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO3_IMR,                 0x53F8C014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_ISR,                 0x53F8C018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_EDGE_SEL,            0x53F8C01C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO4_DR,                  0x53F90000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_GDIR,                0x53F90004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_PSR,                 0x53F90008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO4_ICR1,                0x53F9000C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO4_ICR2,                0x53F90010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO4_IMR,                 0x53F90014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_ISR,                 0x53F90018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_EDGE_SEL,            0x53F9001C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO5
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO5_DR,                  0x53FDC000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO5_GDIR,                0x53FDC004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO5_PSR,                 0x53FDC008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO5_ICR1,                0x53FDC00C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO5_ICR2,                0x53FDC010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO5_IMR,                 0x53FDC014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO5_ISR,                 0x53FDC018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO5_EDGE_SEL,            0x53FDC01C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO6
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO6_DR,                  0x53FE0000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO6_GDIR,                0x53FE0004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO6_PSR,                 0x53FE0008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO6_ICR1,                0x53FE000C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO6_ICR2,                0x53FE0010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO6_IMR,                 0x53FE0014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO6_ISR,                 0x53FE0018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO6_EDGE_SEL,            0x53FE001C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO7
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO7_DR,                  0x53FE4000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO7_GDIR,                0x53FE4004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO7_PSR,                 0x53FE4008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO7_ICR1,                0x53FE400C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO7_ICR2,                0x53FE4010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO7_IMR,                 0x53FE4014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO7_ISR,                 0x53FE4018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO7_EDGE_SEL,            0x53FE401C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTCR,                     0x53FA0000,__READ_WRITE ,__gptcr_bits);
__IO_REG32_BIT(GPTPR,                     0x53FA0004,__READ_WRITE ,__gptpr_bits);
__IO_REG32_BIT(GPTSR,                     0x53FA0008,__READ_WRITE ,__gptsr_bits);
__IO_REG32_BIT(GPTIR,                     0x53FA000C,__READ_WRITE ,__gptir_bits);
__IO_REG32(    GPTOCR1,                   0x53FA0010,__READ_WRITE );
__IO_REG32(    GPTOCR2,                   0x53FA0014,__READ_WRITE );
__IO_REG32(    GPTOCR3,                   0x53FA0018,__READ_WRITE );
__IO_REG32(    GPTICR1,                   0x53FA001C,__READ       );
__IO_REG32(    GPTICR2,                   0x53FA0020,__READ       );
__IO_REG32(    GPTCNT,                    0x53FA0024,__READ       );

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR1,                     0x63FC8000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR1,                     0x63FC8004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR1,                     0x63FC8008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR1,                     0x63FC800C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR1,                     0x63FC8010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR2,                     0x63FC4000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR2,                     0x63FC4004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR2,                     0x63FC4008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR2,                     0x63FC400C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR2,                     0x63FC4010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C3
 **
 ***************************************************************************/
__IO_REG16_BIT(IADR3,                     0x53FEC000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(IFDR3,                     0x53FEC004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2CR3,                     0x53FEC008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2SR3,                     0x53FEC00C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2DR3,                     0x53FEC010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT( IMM_STAT,                  0x63F98000,__READ_WRITE ,__iim_stat_bits);
__IO_REG8_BIT( IMM_STATM,                 0x63F98004,__READ_WRITE ,__iim_statm_bits);
__IO_REG8_BIT( IMM_ERR,                   0x63F98008,__READ_WRITE ,__iim_err_bits);
__IO_REG8_BIT( IMM_EMASK,                 0x63F9800C,__READ_WRITE ,__iim_emask_bits);
__IO_REG8_BIT( IMM_FCTL,                  0x63F98010,__READ_WRITE ,__iim_fctl_bits);
__IO_REG8_BIT( IMM_UA,                    0x63F98014,__READ_WRITE ,__iim_ua_bits);
__IO_REG8(     IMM_LA,                    0x63F98018,__READ_WRITE );
__IO_REG8(     IMM_SDAT,                  0x63F9801C,__READ       );
__IO_REG8_BIT( IMM_PREV,                  0x63F98020,__READ       ,__iim_prev_bits);
__IO_REG8(     IMM_SREV,                  0x63F98024,__READ       );
__IO_REG8(     IMM_PREG_P,                0x63F98028,__READ_WRITE );
__IO_REG8_BIT( IMM_SCS0,                  0x63F9802C,__READ_WRITE ,__iim_scs0_bits);
__IO_REG8_BIT( IMM_SCS1,                  0x63F98030,__READ_WRITE ,__iim_scs1_bits);
__IO_REG8_BIT( IMM_SCS2,                  0x63F98034,__READ_WRITE ,__iim_scs2_bits);
__IO_REG8_BIT( IMM_SCS3,                  0x63F98038,__READ_WRITE ,__iim_scs3_bits);

/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(IOMUXC_GPR0,                       0x53FA8000,__READ_WRITE ,__iomuxc_gpr0_bits);
__IO_REG32_BIT(IOMUXC_GPR1,                       0x53FA8004,__READ_WRITE ,__iomuxc_gpr1_bits);
__IO_REG32_BIT(IOMUXC_GPR2,                       0x53FA8008,__READ_WRITE ,__iomuxc_gpr2_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_0,              0x53FA800C,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_1,              0x53FA8010,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_2,              0x53FA8014,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_3,              0x53FA8018,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_4,              0x53FA801C,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_19,     0x53FA8020,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL0,    0x53FA8024,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW0,    0x53FA8028,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL1,    0x53FA802C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW1,    0x53FA8030,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL2,    0x53FA8034,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW2,    0x53FA8038,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL3,    0x53FA803C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW3,    0x53FA8040,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL4,    0x53FA8044,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW4,    0x53FA8048,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI0_DISP_C,  0x53FA804C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI0_PIN15,   0x53FA8050,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI0_PIN2,    0x53FA8054,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI0_PIN3,    0x53FA8058,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI0_PIN4,    0x53FA805C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT0,  0x53FA8060,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT1,  0x53FA8064,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT2,  0x53FA8068,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT3,  0x53FA806C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT4,  0x53FA8070,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT5,  0x53FA8074,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT6,  0x53FA8078,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT7,  0x53FA807C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT8,  0x53FA8080,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT9,  0x53FA8084,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT10, 0x53FA8088,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT11, 0x53FA808C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT12, 0x53FA8090,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT13, 0x53FA8094,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT14, 0x53FA8098,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT15, 0x53FA809C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT16, 0x53FA80A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT17, 0x53FA80A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT18, 0x53FA80A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT19, 0x53FA80AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT20, 0x53FA80B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT21, 0x53FA80B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT22, 0x53FA80B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP0_DAT23, 0x53FA80BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_PIXCLK, 0x53FA80C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_MCLK,   0x53FA80C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DATA_EN,0x53FA80C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_VSYNC,  0x53FA80CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT4,   0x53FA80D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT5,   0x53FA80D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT6,   0x53FA80D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT7,   0x53FA80DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT8,   0x53FA80E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT9,   0x53FA80E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT10,  0x53FA80E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT11,  0x53FA80EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT12,  0x53FA80F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT13,  0x53FA80F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT14,  0x53FA80F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT15,  0x53FA80FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT16,  0x53FA8100,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT17,  0x53FA8104,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT18,  0x53FA8108,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI0_DAT19,  0x53FA810C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A25,     0x53FA8110,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB2,     0x53FA8114,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D16,     0x53FA8118,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D17,     0x53FA811C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D18,     0x53FA8120,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D19,     0x53FA8124,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D20,     0x53FA8128,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D21,     0x53FA812C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D22,     0x53FA8130,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D23,     0x53FA8134,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB3,     0x53FA8138,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D24,     0x53FA813C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D25,     0x53FA8140,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D26,     0x53FA8144,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D27,     0x53FA8148,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D28,     0x53FA814C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D29,     0x53FA8150,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D30,     0x53FA8154,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D31,     0x53FA8158,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A24,     0x53FA815C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A23,     0x53FA8160,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A22,     0x53FA8164,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A21,     0x53FA8168,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A20,     0x53FA816C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A19,     0x53FA8170,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A18,     0x53FA8174,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A17,     0x53FA8178,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A16,     0x53FA817C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS0,     0x53FA8180,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS1,     0x53FA8184,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_OE,      0x53FA8188,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_RW,      0x53FA818C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_LBA,     0x53FA8190,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB0,     0x53FA8194,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB1,     0x53FA8198,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA0,     0x53FA819C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA1,     0x53FA81A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA2,     0x53FA81A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA3,     0x53FA81A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA4,     0x53FA81AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA5,     0x53FA81B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA6,     0x53FA81B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA7,     0x53FA81B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA8,     0x53FA81BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA9,     0x53FA81C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA10,    0x53FA81C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA11,    0x53FA81C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA12,    0x53FA81CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA13,    0x53FA81D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA14,    0x53FA81D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA15,    0x53FA81D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_WE_B,  0x53FA81DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_10_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RE_B,  0x53FA81E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_10_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_WAIT,    0x53FA81E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_BCLK,    0x53FA81E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_eim_bclk_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS1_TX3_P, 0x53FA81EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS1_TX2_P, 0x53FA81F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS1_CLK_P, 0x53FA81F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS1_TX1_P, 0x53FA81F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS1_TX0_P, 0x53FA81FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS0_TX3_P, 0x53FA8200,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS0_CLK_P, 0x53FA8204,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS0_TX2_P, 0x53FA8208,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS0_TX1_P, 0x53FA820C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_LVDS0_TX0_P, 0x53FA8210,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_lvds1_tx3_p_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_10,     0x53FA8214,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_10_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_11,     0x53FA8218,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_eim_bclk_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_12,     0x53FA821C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_eim_bclk_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_13,     0x53FA8220,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_eim_bclk_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_14,     0x53FA8224,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_eim_bclk_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CLE,   0x53FA8228,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_ALE,   0x53FA822C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_WP_B,  0x53FA8230,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RB0,   0x53FA8234,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS0,   0x53FA8238,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS1,   0x53FA823C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS2,   0x53FA8240,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS3,   0x53FA8244,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDIO,    0x53FA8248,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_REF_CLK, 0x53FA824C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RX_ER,   0x53FA8250,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_CRS_DV,  0x53FA8254,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RXD1,    0x53FA8258,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_RXD0,    0x53FA825C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TX_EN,   0x53FA8260,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_fec_crs_dv_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TXD1,    0x53FA8264,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_TXD0,    0x53FA8268,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_FEC_MDC,     0x53FA826C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DIOW,   0x53FA8270,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DMACK,  0x53FA8274,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DMARQ,  0x53FA8278,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_BUFFER_EN,0x53FA827C,__READ_WRITE,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_INTRQ,  0x53FA8280,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DIOR,   0x53FA8284,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_RESET_B,0x53FA8288,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_IORDY,  0x53FA828C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DA_0,   0x53FA8290,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DA_1,   0x53FA8294,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DA_2,   0x53FA8298,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_CS_0,   0x53FA829C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_CS_1,   0x53FA82A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA0,  0x53FA82A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA1,  0x53FA82A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA2,  0x53FA82AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA,   0x53FA82B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA4,  0x53FA82B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA5,  0x53FA82B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA6,  0x53FA82BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA7,  0x53FA82C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA8,  0x53FA82C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA9,  0x53FA82C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA10, 0x53FA82CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA11, 0x53FA82D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA12, 0x53FA82D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA13, 0x53FA82D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA14, 0x53FA82DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PATA_DATA15, 0x53FA82E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA0,   0x53FA82E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA1,   0x53FA82E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CMD,     0x53FA82EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA2,   0x53FA82F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CLK,     0x53FA82F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA3,   0x53FA82F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CLK,     0x53FA82FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CMD,     0x53FA8300,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA3,   0x53FA8304,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA2,   0x53FA8308,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA1,   0x53FA830C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA0,   0x53FA8310,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_0,      0x53FA8314,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_1,      0x53FA8318,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_9,      0x53FA831C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_3,      0x53FA8320,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_6,      0x53FA8324,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_2,      0x53FA8328,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_4,      0x53FA832C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_5,      0x53FA8330,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_7,      0x53FA8334,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_8,      0x53FA8338,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_16,     0x53FA833C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_17,     0x53FA8340,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_18,     0x53FA8344,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_gpio_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_19,     0x53FA8348,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL0,    0x53FA834C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW0,    0x53FA8350,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL1,    0x53FA8354,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW1,    0x53FA8358,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL2,    0x53FA835C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW2,    0x53FA8360,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL3,    0x53FA8364,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW3,    0x53FA8368,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL4,    0x53FA836C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW4,    0x53FA8370,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_KEYPAD, 0x53FA8374,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI0_DISP_CLK,0x53FA8378,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI0_PIN15,   0x53FA837C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI0_PIN2,    0x53FA8380,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI0_PIN3,    0x53FA8384,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI0_PIN4,    0x53FA8388,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT0,  0x53FA838C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT1,  0x53FA8390,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT2,  0x53FA8394,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT3,  0x53FA8398,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT4,  0x53FA839C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT5,  0x53FA83A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT6,  0x53FA83A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT7,  0x53FA83A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT8,  0x53FA83AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT9,  0x53FA83B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT10, 0x53FA83B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT11, 0x53FA83B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT12, 0x53FA83BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT13, 0x53FA83C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT14, 0x53FA83C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT15, 0x53FA83C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT16, 0x53FA83CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT17, 0x53FA83D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT18, 0x53FA83D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT19, 0x53FA83D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT20, 0x53FA83DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT21, 0x53FA83E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT22, 0x53FA83E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP0_DAT23, 0x53FA83E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_PIXCLK, 0x53FA83EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_MCLK,   0x53FA83F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DATA_EN,0x53FA83F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_VSYNC,  0x53FA83F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT4,   0x53FA83FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT5,   0x53FA8400,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT6,   0x53FA8404,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT7,   0x53FA8408,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT8,   0x53FA840C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT9,   0x53FA8410,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT10,  0x53FA8414,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT11,  0x53FA8418,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT12,  0x53FA841C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT13,  0x53FA8420,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT14,  0x53FA8424,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT15,  0x53FA8428,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT16,  0x53FA842C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT17,  0x53FA8430,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT18,  0x53FA8434,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI0_DAT19,  0x53FA8438,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_CSI_0,  0x53FA843C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TMS,    0x53FA8440,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_MOD,    0x53FA8444,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TRSTB,  0x53FA8448,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TDI,    0x53FA844C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TCK,    0x53FA8450,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TDO,    0x53FA8454,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A25,     0x53FA8458,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB2,     0x53FA845C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D16,     0x53FA8460,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D17,     0x53FA8464,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D18,     0x53FA8468,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D19,     0x53FA846C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D20,     0x53FA8470,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D21,     0x53FA8474,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D22,     0x53FA8478,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D23,     0x53FA847C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB3,     0x53FA8480,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D24,     0x53FA8484,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D25,     0x53FA8488,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D26,     0x53FA848C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D27,     0x53FA8490,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D28,     0x53FA8494,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D29,     0x53FA8498,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D30,     0x53FA849C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D31,     0x53FA84A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_EIM_1,  0x53FA84A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A24,     0x53FA84A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A23,     0x53FA84AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A22,     0x53FA84B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A21,     0x53FA84B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A20,     0x53FA84B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A19,     0x53FA84BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A18,     0x53FA84C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A17,     0x53FA84C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A16,     0x53FA84C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS0,     0x53FA84CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS1,     0x53FA84D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_OE,      0x53FA84D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_RW,      0x53FA84D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_LBA,     0x53FA84DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_EIM_4,  0x53FA84E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB0,     0x53FA84E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB1,     0x53FA84E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA0,     0x53FA84EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA1,     0x53FA84F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA2,     0x53FA84F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA3,     0x53FA84F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA4,     0x53FA84FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA5,     0x53FA8500,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA6,     0x53FA8504,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA7,     0x53FA8508,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA8,     0x53FA850C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA9,     0x53FA8510,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA10,    0x53FA8514,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA11,    0x53FA8518,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA12,    0x53FA851C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA13,    0x53FA8520,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA14,    0x53FA8524,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DA15,    0x53FA8528,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_WE_B,  0x53FA852C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RE_B,  0x53FA8530,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_WAIT,    0x53FA8534,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_BCLK,    0x53FA8538,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_EIM_7,  0x53FA853C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_10,     0x53FA8540,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_11,     0x53FA8544,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_12,     0x53FA8548,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_13,     0x53FA854C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_14,     0x53FA8550,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM3,   0x53FA8554,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS3,  0x53FA8558,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCKE1, 0x53FA855C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM2,   0x53FA8560,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDODT1, 0x53FA8564,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS2,  0x53FA8568,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_RESET,  0x53FA856C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCLK_1,0x53FA8570,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_CAS,    0x53FA8574,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCLK_0,0x53FA8578,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS0,  0x53FA857C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDODT0, 0x53FA8580,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM0,   0x53FA8584,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_RAS,    0x53FA8588,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCKE0, 0x53FA858C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS1,  0x53FA8590,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM1,   0x53FA8594,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dram_dqm3_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_ON_REQ, 0x53FA8598,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_STBY_REQ,0x53FA859C,__READ_WRITE,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CLE,   0x53FA85A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_ALE,   0x53FA85A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_WP_B,  0x53FA85A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RB0,   0x53FA85AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS0,   0x53FA85B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS1,   0x53FA85B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS2,   0x53FA85B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS3,   0x53FA85BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_NANDF,  0x53FA85C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDIO,    0x53FA85C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_REF_CLK, 0x53FA85C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RX_ER,   0x53FA85CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_CRS_DV,  0x53FA85D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RXD1,    0x53FA85D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_RXD0,    0x53FA85D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TX_EN,   0x53FA85DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TXD1,    0x53FA85E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_TXD0,    0x53FA85E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_FEC_MDC,     0x53FA85E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_FEC,    0x53FA85EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DIOW,   0x53FA85F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DMACK,  0x53FA85F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DMARQ,  0x53FA85F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_BUFFER_EN,0x53FA85FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_INTRQ,  0x53FA8600,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DIOR,   0x53FA8604,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_RESET_B,0x53FA8608,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_IORDY,  0x53FA860C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DA_0,   0x53FA8610,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DA_1,   0x53FA8614,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DA_2,   0x53FA8618,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_CS_0,   0x53FA861C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_CS_1,   0x53FA8620,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_PATA_2, 0x53FA8624,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA0,  0x53FA8628,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA1,  0x53FA862C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA2,  0x53FA8630,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA3,  0x53FA8634,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA4,  0x53FA8638,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA5,  0x53FA863C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA6,  0x53FA8640,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA7,  0x53FA8644,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA8,  0x53FA8648,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA9,  0x53FA864C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA10, 0x53FA8650,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA11, 0x53FA8654,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA12, 0x53FA8658,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA13, 0x53FA865C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA14, 0x53FA8660,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PATA_DATA15, 0x53FA8664,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_PATA_0, 0x53FA8668,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA0,   0x53FA866C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA1,   0x53FA8670,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CMD,     0x53FA8674,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA2,   0x53FA8678,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CLK,     0x53FA867C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA3,   0x53FA8680,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_SD1,    0x53FA8684,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CLK,     0x53FA8688,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CMD,     0x53FA868C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA3,   0x53FA8690,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA2,   0x53FA8694,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA1,   0x53FA8698,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA0,   0x53FA869C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_SD2,    0x53FA86A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_0,      0x53FA86A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_1,      0x53FA86A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_9,      0x53FA86AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_3,      0x53FA86B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_6,      0x53FA86B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_2,      0x53FA86B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_4,      0x53FA86BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_5,      0x53FA86C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_7,      0x53FA86C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_8,      0x53FA86C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_16,     0x53FA86CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_17,     0x53FA86D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_18,     0x53FA86D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_gpio_19_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NVCC_GPIO,   0x53FA86D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_nvcc_keypad_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_POR_B,       0x53FA86DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE1,  0x53FA86E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RESET_IN_B,  0x53FA86E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE0,  0x53FA86E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_TEST_MODE,   0x53FA86EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_ADDDS,       0x53FA86F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRMODE_CTL, 0x53FA86F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddrmode_ctl_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRPKE,      0x53FA86FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddrpke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRPK,       0x53FA8708,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddrpke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRHYS,      0x53FA8710,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddrhys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRMODE,     0x53FA8714,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddrmode_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_B0DS,        0x53FA8718,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_B1DS,        0x53FA871C,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_CTLDS,       0x53FA8720,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDR_TYPE,    0x53FA8724,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_ddr_type_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_B2DS,        0x53FA8728,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_B3DS,        0x53FA872C,__READ_WRITE ,__iomuxc_sw_pad_ctl_grp_addds_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DA_AMX_SELECT_INPUT,    0x53FA8730,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DB_AMX_SELECT_INPUT,    0x53FA8734,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_RXCLK_AMX_SELECT_INPUT, 0x53FA8738,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_RXFS_AMX_SELECT_INPUT,  0x53FA873C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXCLK_AMX_SELECT_INPUT, 0x53FA8740,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXFS_AMX_SELECT_INPUT,  0x53FA8744,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_DA_AMX_SELECT_INPUT,    0x53FA8748,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_DB_AMX_SELECT_INPUT,    0x53FA874C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXCLK_AMX_SELECT_INPUT, 0x53FA8750,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXFS_AMX_SELECT_INPUT,  0x53FA8754,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_TXCLK_AMX_SELECT_INPUT, 0x53FA8758,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_TXFS_AMX_SELECT_INPUT,  0x53FA875C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CAN1_IPP_IND_CANRX_SELECT_INPUT,        0x53FA8760,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CAN2_IPP_IND_CANRX_SELECT_INPUT,        0x53FA8764,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_ASRC_EXT_SELECT_INPUT,          0x53FA8768,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_DI1_CLK_SELECT_INPUT,           0x53FA876C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL1_BYPASS_CLK_SELECT_INPUT,       0x53FA8770,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL2_BYPASS_CLK_SELECT_INPUT,       0x53FA8774,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL3_BYPASS_CLK_SELECT_INPUT,       0x53FA8778,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL4_BYPASS_CLK_SELECT_INPUT,       0x53FA877C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_CSPI_CLK_IN_SELECT_INPUT,      0x53FA8780,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_MISO_SELECT_INPUT,         0x53FA8784,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_MOSI_SELECT_INPUT,         0x53FA8788,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS0_B_SELECT_INPUT,        0x53FA878C,__READ_WRITE ,__iomuxc_uart2_ipp_uart_rxd_mux_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS1_B_SELECT_INPUT,        0x53FA8790,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS2_B_SELECT_INPUT,        0x53FA8794,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS3_B_SELECT_INPUT,        0x53FA8798,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_CSPI_CLK_IN_SELECT_INPUT,    0x53FA879C,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_MISO_SELECT_INPUT,       0x53FA87A0,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_MOSI_SELECT_INPUT,       0x53FA87A4,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_SS_B_0_SELECT_INPUT,     0x53FA87A8,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_SS_B_1_SELECT_INPUT,     0x53FA87AC,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_SS_B_2_SELECT_INPUT,     0x53FA87B0,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI1_IPP_IND_SS_B_3_SELECT_INPUT,     0x53FA87B4,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_CSPI_CLK_IN_SELECT_INPUT,    0x53FA87B8,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_MISO_SELECT_INPUT,       0x53FA87BC,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_MOSI_SELECT_INPUT,       0x53FA87C0,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_SS_B_0_SELECT_INPUT,     0x53FA87C4,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_SS_B_1_SELECT_INPUT,     0x53FA87C8,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_FSR_SELECT_INPUT,         0x53FA87CC,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_FST_SELECT_INPUT,         0x53FA87D0,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_HCKR_SELECT_INPUT,        0x53FA87D4,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_HCKT_SELECT_INPUT,        0x53FA87D8,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SCKR_SELECT_INPUT,        0x53FA87DC,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SCKT_SELECT_INPUT,        0x53FA87E0,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO0_SELECT_INPUT,        0x53FA87E4,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO1_SELECT_INPUT,        0x53FA87E8,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO2_SDI3_SELECT_INPUT,   0x53FA87EC,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO3_SDI2_SELECT_INPUT,   0x53FA87F0,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO4_SDI1_SELECT_INPUT,   0x53FA87F4,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESAI1_IPP_IND_SDO5_SDI0_SELECT_INPUT,   0x53FA87F8,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_ESDHC1_IPP_WP_ON_SELECT_INPUT,          0x53FA87FC,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_COL_SELECT_INPUT,               0x53FA8800,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_MDI_SELECT_INPUT,               0x53FA8804,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_CLK_SELECT_INPUT,            0x53FA8808,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_FIRI_IPP_IND_RXD_SELECT_INPUT,          0x53FA880C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_GPC_PMIC_RDY_SELECT_INPUT,              0x53FA8810,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C1_IPP_SCL_IN_SELECT_INPUT,           0x53FA8814,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C1_IPP_SDA_IN_SELECT_INPUT,           0x53FA8818,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SCL_IN_SELECT_INPUT,           0x53FA881C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SDA_IN_SELECT_INPUT,           0x53FA8820,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SCL_IN_SELECT_INPUT,           0x53FA8824,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_I2C3_IPP_SDA_IN_SELECT_INPUT,           0x53FA8828,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_DI_0_IND_DISPB_SD_D_SELECT_INPUT,0x53FA882C,__READ_WRITE,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_DI_1_IND_DISPB_SD_D_SELECT_INPUT,0x53FA8830,__READ_WRITE,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENS1_DATA_EN_SELECT_INPUT, 0x53FA8834,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENS1_HSYNC_SELECT_INPUT,   0x53FA8838,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_IND_SENS1_VSYNC_SELECT_INPUT,   0x53FA883C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_5_SELECT_INPUT,         0x53FA8840,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_6_SELECT_INPUT,         0x53FA8844,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_7_SELECT_INPUT,         0x53FA8848,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_5_SELECT_INPUT,         0x53FA884C,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_6_SELECT_INPUT,         0x53FA8850,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_7_SELECT_INPUT,         0x53FA8854,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_MLB_MLBCLK_IN_SELECT_INPUT,             0x53FA8858,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_MLB_MLBDAT_IN_SELECT_INPUT,             0x53FA885C,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_MLB_MLBSIG_IN_SELECT_INPUT,             0x53FA8860,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_OWIRE_BATTERY_LINE_IN_SELECT_INPUT,     0x53FA8864,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_SDMA_EVENTS_14_SELECT_INPUT,            0x53FA8868,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_SDMA_EVENTS_15_SELECT_INPUT,            0x53FA886C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_SPDIF_SPDIF_IN1_SELECT_INPUT,           0x53FA8870,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART1_IPP_UART_RTS_B_SELECT_INPUT,      0x53FA8874,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART1_IPP_UART_RXD_MUX_SELECT_INPUT,    0x53FA8878,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART2_IPP_UART_RTS_B_SELECT_INPUT,      0x53FA887C,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART2_IPP_UART_RXD_MUX_SELECT_INPUT,    0x53FA8880,__READ_WRITE ,__iomuxc_uart2_ipp_uart_rxd_mux_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RTS_B_SELECT_INPUT,      0x53FA8884,__READ_WRITE ,__iomuxc_uart2_ipp_uart_rxd_mux_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RXD_MUX_SELECT_INPUT,    0x53FA8888,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART4_IPP_UART_RTS_B_SELECT_INPUT,      0x53FA888C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART4_IPP_UART_RXD_MUX_SELECT_INPUT,    0x53FA8890,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART5_IPP_UART_RTS_B_SELECT_INPUT,      0x53FA8894,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_UART5_IPP_UART_RXD_MUX_SELECT_INPUT,    0x53FA8898,__READ_WRITE ,__iomuxc_can1_ipp_ind_canrx_select_input_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_OTG_OC_SELECT_INPUT,     0x53FA889C,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH1_OC_SELECT_INPUT,     0x53FA88A0,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH2_OC_SELECT_INPUT,     0x53FA88A4,__READ_WRITE ,__iomuxc_audmux_p4_input_da_amx_select_input_bits);

/***************************************************************************
 **
 **  IPTP
 **
 ***************************************************************************/
__IO_REG32_BIT(PTP_TSPDR1,            0x63FFC000,__READ_WRITE ,__ptp_tspdr1_bits);
__IO_REG32_BIT(PTP_TSPDR2,            0x63FFC004,__READ_WRITE ,__ptp_tspdr2_bits);
__IO_REG32_BIT(PTP_TSPDR3,            0x63FFC008,__READ_WRITE ,__ptp_tspdr3_bits);
__IO_REG32_BIT(PTP_TSPDR4,            0x63FFC00C,__READ_WRITE ,__ptp_tspdr4_bits);
__IO_REG32_BIT(PTP_TSPOV,             0x63FFC010,__READ_WRITE ,__ptp_tspov_bits);
__IO_REG32_BIT(PTP_TSMR,              0x63FFC014,__READ_WRITE ,__ptp_tsmr_bits);
__IO_REG32_BIT(PTP_TMR_PEVENT,        0x63FFC018,__READ_WRITE ,__ptp_tmr_pevent_bits);
__IO_REG32_BIT(PTP_TMR_PEMASK,        0x63FFC01C,__READ_WRITE ,__ptp_tmr_pevent_bits);
__IO_REG32(    PTP_TMR_RXTS_H,        0x63FFC020,__READ       );
__IO_REG32(    PTP_TMR_RXTS_L,        0x63FFC030,__READ       );
__IO_REG32(    PTP_TMR_TXTS_H,        0x63FFC040,__READ       );
__IO_REG32(    PTP_TMR_TXTS_L,        0x63FFC050,__READ       );
__IO_REG32_BIT(PTP_TSPDR5,            0x63FFC060,__READ_WRITE ,__ptp_tspdr5_bits);
__IO_REG32_BIT(PTP_TSPDR6,            0x63FFC064,__READ_WRITE ,__ptp_tspdr6_bits);
__IO_REG32_BIT(PTP_TSPDR7,            0x63FFC068,__READ_WRITE ,__ptp_tspdr7_bits);
__IO_REG32_BIT(TMR_CTRL,              0x63FD4000,__READ_WRITE ,__tmr_ctrl_bits);
__IO_REG32_BIT(TMR_TEVENT,            0x63FD4004,__READ_WRITE ,__tmr_tevent_bits);
__IO_REG32_BIT(TMR_TEMASK,            0x63FD4008,__READ_WRITE ,__tmr_tevent_bits);
__IO_REG32(    TMR_CNT_L,             0x63FD400C,__READ_WRITE );
__IO_REG32(    TMR_CNT_H,             0x63FD4010,__READ_WRITE );
__IO_REG32(    TMR_ADD,               0x63FD4014,__READ_WRITE );
__IO_REG32(    TMR_ACC,               0x63FD4018,__READ       );
__IO_REG32_BIT(TMR_PRSC,              0x63FD401C,__READ_WRITE ,__tmr_prsc_bits);
__IO_REG32(    TMROFF_L,              0x63FD4020,__READ_WRITE );
__IO_REG32(    TMROFF_H,              0x63FD4024,__READ_WRITE );
__IO_REG32(    TMR_ALARM1_L,          0x63FD4028,__READ_WRITE );
__IO_REG32(    TMR_ALARM1_H,          0x63FD402C,__READ_WRITE );
__IO_REG32(    TMR_ALARM2_L,          0x63FD4030,__READ_WRITE );
__IO_REG32(    TMR_ALARM2_H,          0x63FD4034,__READ_WRITE );
__IO_REG32(    TMR_FIPER1,            0x63FD4038,__READ_WRITE );
__IO_REG32(    TMR_FIPER2,            0x63FD403C,__READ_WRITE );
__IO_REG32(    TMR_FIPER3,            0x63FD4040,__READ_WRITE );
__IO_REG32(    TMR_ETTS1_L,           0x63FD4044,__READ       );
__IO_REG32(    TMR_ETTS1_H,           0x63FD4048,__READ       );
__IO_REG32(    TMR_ETTS2_L,           0x63FD404C,__READ       );
__IO_REG32(    TMR_ETTS2_H,           0x63FD4050,__READ       );
__IO_REG32(    TMR_FSV_L,             0x63FD4054,__READ_WRITE );
__IO_REG32(    TMR_FSV_H,             0x63FD4058,__READ_WRITE );

/***************************************************************************
 **
 **  IPU
 **
 ***************************************************************************/
__IO_REG32_BIT(IPU_CONF,                      0x18000000,__READ_WRITE ,__ipu_conf_bits);
__IO_REG32_BIT(IPU_SISG_CTRL0,                0x18000004,__READ_WRITE ,__ipu_sisg_ctrl0_bits);
__IO_REG32_BIT(IPU_SISG_CTRL1,                0x18000008,__READ_WRITE ,__ipu_sisg_ctrl1_bits);
__IO_REG32_BIT(IPU_SISG_SET_1,                0x1800000C,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_2,                0x18000010,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_3,                0x18000014,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_4,                0x18000018,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_5,                0x1800001C,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_6,                0x18000020,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_CLR_1,                0x18000024,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_2,                0x18000028,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_3,                0x1800002C,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_4,                0x18000030,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_5,                0x18000034,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_6,                0x18000038,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_INT_CTRL_1,                0x1800003C,__READ_WRITE ,__ipu_int_ctrl_1_bits);
__IO_REG32_BIT(IPU_INT_CTRL_2,                0x18000040,__READ_WRITE ,__ipu_int_ctrl_2_bits);
__IO_REG32_BIT(IPU_INT_CTRL_3,                0x18000044,__READ_WRITE ,__ipu_int_ctrl_3_bits);
__IO_REG32_BIT(IPU_INT_CTRL_4,                0x18000048,__READ_WRITE ,__ipu_int_ctrl_4_bits);
__IO_REG32_BIT(IPU_INT_CTRL_5,                0x1800004C,__READ_WRITE ,__ipu_int_ctrl_5_bits);
__IO_REG32_BIT(IPU_INT_CTRL_6,                0x18000050,__READ_WRITE ,__ipu_int_ctrl_6_bits);
__IO_REG32_BIT(IPU_INT_CTRL_7,                0x18000054,__READ_WRITE ,__ipu_int_ctrl_7_bits);
__IO_REG32_BIT(IPU_INT_CTRL_8,                0x18000058,__READ_WRITE ,__ipu_int_ctrl_8_bits);
__IO_REG32_BIT(IPU_INT_CTRL_9,                0x1800005C,__READ_WRITE ,__ipu_int_ctrl_9_bits);
__IO_REG32_BIT(IPU_INT_CTRL_10,               0x18000060,__READ_WRITE ,__ipu_int_ctrl_10_bits);
__IO_REG32_BIT(IPU_INT_CTRL_11,               0x18000064,__READ_WRITE ,__ipu_int_ctrl_11_bits);
__IO_REG32_BIT(IPU_INT_CTRL_12,               0x18000068,__READ_WRITE ,__ipu_int_ctrl_12_bits);
__IO_REG32_BIT(IPU_INT_CTRL_13,               0x1800006C,__READ_WRITE ,__ipu_int_ctrl_13_bits);
__IO_REG32_BIT(IPU_INT_CTRL_14,               0x18000070,__READ_WRITE ,__ipu_int_ctrl_14_bits);
__IO_REG32_BIT(IPU_INT_CTRL_15,               0x18000074,__READ_WRITE ,__ipu_int_ctrl_15_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_1,              0x18000078,__READ_WRITE ,__ipu_sdma_event_1_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_2,              0x1800007C,__READ_WRITE ,__ipu_sdma_event_2_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_3,              0x18000080,__READ_WRITE ,__ipu_sdma_event_3_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_4,              0x18000084,__READ_WRITE ,__ipu_sdma_event_4_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_7,              0x18000088,__READ_WRITE ,__ipu_sdma_event_7_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_8,              0x1800008C,__READ_WRITE ,__ipu_sdma_event_8_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_11,             0x18000090,__READ_WRITE ,__ipu_sdma_event_11_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_12,             0x18000094,__READ_WRITE ,__ipu_sdma_event_12_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_13,             0x18000098,__READ_WRITE ,__ipu_sdma_event_13_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_14,             0x1800009C,__READ_WRITE ,__ipu_sdma_event_14_bits);
__IO_REG32_BIT(IPU_SRM_PRI1,                  0x180000A0,__READ_WRITE ,__ipu_srm_pri1_bits);
__IO_REG32_BIT(IPU_SRM_PRI2,                  0x180000A4,__READ_WRITE ,__ipu_srm_pri2_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW1,             0x180000A8,__READ_WRITE ,__ipu_fs_proc_flow1_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW2,             0x180000AC,__READ_WRITE ,__ipu_fs_proc_flow2_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW3,             0x180000B0,__READ_WRITE ,__ipu_fs_proc_flow3_bits);
__IO_REG32_BIT(IPU_FS_DISP_FLOW1,             0x180000B4,__READ_WRITE ,__ipu_fs_disp_flow1_bits);
__IO_REG32_BIT(IPU_FS_DISP_FLOW2,             0x180000B8,__READ_WRITE ,__ipu_fs_disp_flow2_bits);
__IO_REG32_BIT(IPU_SKIP,                      0x180000BC,__READ_WRITE ,__ipu_skip_bits);
__IO_REG32_BIT(IPU_DISP_GEN,                  0x180000C4,__READ_WRITE ,__ipu_disp_gen_bits);
__IO_REG32_BIT(IPU_DISP_ALT1,                 0x180000C8,__READ_WRITE ,__ipu_disp_alt1_bits);
__IO_REG32_BIT(IPU_DISP_ALT2,                 0x180000CC,__READ_WRITE ,__ipu_disp_alt2_bits);
__IO_REG32_BIT(IPU_DISP_ALT3,                 0x180000D0,__READ_WRITE ,__ipu_disp_alt3_bits);
__IO_REG32_BIT(IPU_DISP_ALT4,                 0x180000D4,__READ_WRITE ,__ipu_disp_alt4_bits);
__IO_REG32_BIT(IPU_SNOOP,                     0x180000D8,__READ_WRITE ,__ipu_snoop_bits);
__IO_REG32_BIT(IPU_MEM_RST,                   0x180000DC,__READ_WRITE ,__ipu_mem_rst_bits);
__IO_REG32_BIT(IPU_PM,                        0x180000E0,__READ_WRITE ,__ipu_pm_bits);
__IO_REG32_BIT(IPU_GPR,                       0x180000E4,__READ_WRITE ,__ipu_gpr_bits);
__IO_REG32_BIT(IPU_CH_DB_MODE_SEL0,           0x18000150,__READ_WRITE ,__ipu_ch_db_mode_sel0_bits);
__IO_REG32_BIT(IPU_CH_DB_MODE_SEL1,           0x18000154,__READ_WRITE ,__ipu_ch_db_mode_sel1_bits);
__IO_REG32_BIT(IPU_ALT_CH_DB_MODE_SEL0,       0x18000168,__READ_WRITE,__ipu_alt_ch_db_mode_sel0_bits);
__IO_REG32_BIT(IPU_ALT_CH_DB_MODE_SEL1,       0x1800016C,__READ_WRITE,__ipu_alt_ch_db_mode_sel1_bits);
__IO_REG32_BIT(IPU_INT_STAT_1,                0x18000200,__WRITE      ,__ipu_int_stat_1_bits);
__IO_REG32_BIT(IPU_INT_STAT_2,                0x18000204,__WRITE      ,__ipu_int_stat_2_bits);
__IO_REG32_BIT(IPU_INT_STAT_3,                0x18000208,__WRITE      ,__ipu_int_stat_3_bits);
__IO_REG32_BIT(IPU_INT_STAT_5,                0x18000210,__WRITE      ,__ipu_int_stat_5_bits);
__IO_REG32_BIT(IPU_INT_STAT_6,                0x18000214,__WRITE      ,__ipu_int_stat_6_bits);
__IO_REG32_BIT(IPU_INT_STAT_7,                0x18000218,__WRITE      ,__ipu_int_stat_7_bits);
__IO_REG32_BIT(IPU_INT_STAT_8,                0x1800021C,__WRITE      ,__ipu_int_stat_8_bits);
__IO_REG32_BIT(IPU_INT_STAT_9,                0x18000220,__WRITE      ,__ipu_int_stat_9_bits);
__IO_REG32_BIT(IPU_INT_STAT_10,               0x18000224,__WRITE      ,__ipu_int_stat_10_bits);
__IO_REG32_BIT(IPU_INT_STAT_11,               0x18000228,__WRITE      ,__ipu_int_stat_11_bits);
__IO_REG32_BIT(IPU_INT_STAT_12,               0x1800022C,__WRITE      ,__ipu_int_stat_12_bits);
__IO_REG32_BIT(IPU_INT_STAT_13,               0x18000230,__WRITE      ,__ipu_int_stat_13_bits);
__IO_REG32_BIT(IPU_INT_STAT_14,               0x18000234,__WRITE      ,__ipu_int_stat_14_bits);
__IO_REG32_BIT(IPU_INT_STAT_15,               0x18000238,__WRITE      ,__ipu_int_stat_15_bits);
__IO_REG32_BIT(IPU_CUR_BUF_0,                 0x1800023C,__READ       ,__ipu_cur_buf_0_bits);
__IO_REG32_BIT(IPU_CUR_BUF_1,                 0x18000240,__READ       ,__ipu_cur_buf_1_bits);
__IO_REG32_BIT(IPU_ALT_CUR_BUF_0,             0x18000244,__READ       ,__ipu_alt_cur_buf_0_bits);
__IO_REG32_BIT(IPU_ALT_CUR_BUF_1,             0x18000248,__READ       ,__ipu_alt_cur_buf_1_bits);
__IO_REG32_BIT(IPU_SRM_STAT,                  0x1800024C,__READ       ,__ipu_srm_stat_bits);
__IO_REG32_BIT(IPU_PROC_TASKS_STAT,           0x18000250,__READ       ,__ipu_proc_tasks_stat_bits);
__IO_REG32_BIT(IPU_DISP_TASKS_STAT,           0x18000254,__READ       ,__ipu_disp_tasks_stat_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_0,          0x18000258,__READ       ,__ipu_triple_cur_buf_0_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_1,          0x1800025C,__READ       ,__ipu_triple_cur_buf_1_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_2,          0x18000260,__READ       ,__ipu_triple_cur_buf_2_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_3,          0x18000264,__READ       ,__ipu_triple_cur_buf_3_bits);
__IO_REG32_BIT(IPU_CH_BUF0_RDY0,              0x18000268,__READ_WRITE ,__ipu_ch_buf0_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF0_RDY1,              0x1800026C,__READ_WRITE ,__ipu_ch_buf0_rdy1_bits);
__IO_REG32_BIT(IPU_CH_BUF1_RDY0,              0x18000270,__READ_WRITE ,__ipu_ch_buf1_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF1_RDY1,              0x18000274,__READ_WRITE ,__ipu_ch_buf1_rdy1_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF0_RDY0,          0x18000278,__READ_WRITE ,__ipu_alt_ch_buf0_rdy0_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF0_RDY1,          0x1800027C,__READ_WRITE ,__ipu_alt_ch_buf0_rdy1_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF1_RDY0,          0x18000280,__READ_WRITE ,__ipu_alt_ch_buf1_rdy0_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF1_RDY1,          0x18000284,__READ_WRITE ,__ipu_alt_ch_buf1_rdy1_bits);
__IO_REG32_BIT(IPU_CH_BUF2_RDY0,              0x18000288,__READ_WRITE ,__ipu_ch_buf2_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF2_RDY1,              0x1800028C,__READ_WRITE ,__ipu_ch_buf2_rdy1_bits);
__IO_REG32_BIT(IPU_INT_STAT_4,                0x180002C0,__READ_WRITE ,__ipu_int_stat_4_bits);
__IO_REG32_BIT(IPU_IDMAC_CONF,                0x18008000,__READ_WRITE ,__ipu_idmac_conf_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_EN_1,             0x18008004,__READ_WRITE ,__ipu_idmac_ch_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_EN_2,             0x18008008,__READ_WRITE ,__ipu_idmac_ch_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SEP_ALPHA,           0x1800800C,__READ_WRITE ,__ipu_idmac_sep_alpha_bits);
__IO_REG32_BIT(IPU_IDMAC_ALT_SEP_ALPHA,       0x18008010,__READ_WRITE ,__ipu_idmac_alt_sep_alpha_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_PRI_1,            0x18008014,__READ_WRITE ,__ipu_idmac_ch_pri_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_PRI_2,            0x18008018,__READ_WRITE ,__ipu_idmac_ch_pri_2_bits);
__IO_REG32_BIT(IPU_IDMAC_WM_EN_1,             0x1800801C,__READ_WRITE ,__ipu_idmac_wm_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_WM_EN_2,             0x18008020,__READ_WRITE ,__ipu_idmac_wm_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_LOCK_EN_1,           0x18008024,__READ_WRITE ,__ipu_idmac_lock_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_LOCK_EN_2,           0x18008028,__READ_WRITE ,__ipu_idmac_lock_en_2_bits);
__IO_REG32(    IPU_IDMAC_SUB_ADDR_0,          0x1800802C,__READ_WRITE );
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_1,          0x18008030,__READ_WRITE ,__ipu_idmac_sub_addr_1_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_2,          0x18008034,__READ_WRITE ,__ipu_idmac_sub_addr_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_3,          0x18008038,__READ_WRITE ,__ipu_idmac_sub_addr_3_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_4,          0x1800803C,__READ_WRITE ,__ipu_idmac_sub_addr_4_bits);
__IO_REG32_BIT(IPU_IDMAC_BNDM_EN_1,           0x18008040,__READ_WRITE ,__ipu_idmac_bndm_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_BNDM_EN_2,           0x18008044,__READ_WRITE ,__ipu_idmac_bndm_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SC_CORD,             0x18008048,__READ_WRITE ,__ipu_idmac_sc_cord_bits);
__IO_REG32_BIT(IPU_IDMAC_SC_CORD_1,           0x1800804C,__READ_WRITE ,__ipu_idmac_sc_cord_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_BUSY_1,           0x18008100,__READ       ,__ipu_idmac_ch_busy_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_BUSY_2,           0x18008104,__READ       ,__ipu_idmac_ch_busy_2_bits);
__IO_REG32_BIT(IPU_DP_DEBUG_CNT,              0x180180BC,__READ_WRITE ,__ipu_dp_debug_cnt_bits);
__IO_REG32_BIT(IPU_DP_DEBUG_STAT,             0x180180C0,__READ       ,__ipu_dp_debug_stat_bits);
__IO_REG32_BIT(IPU_IC_CONF,                   0x18020000,__READ_WRITE ,__ipu_ic_conf_bits);
__IO_REG32_BIT(IPU_IC_PRP_ENC_RSC,            0x18020004,__READ_WRITE ,__ipu_ic_prp_enc_rsc_bits);
__IO_REG32_BIT(IPU_IC_PRP_VF_RSC,             0x18020008,__READ_WRITE ,__ipu_ic_prp_vf_rsc_bits);
__IO_REG32_BIT(IPU_IC_PP_RSC,                 0x1802000C,__READ_WRITE ,__ipu_ic_pp_rsc_bits);
__IO_REG32_BIT(IPU_IC_CMBP_1,                 0x18020010,__READ_WRITE ,__ipu_ic_cmbp_1_bits);
__IO_REG32_BIT(IPU_IC_CMBP_2,                 0x18020014,__READ_WRITE ,__ipu_ic_cmbp_2_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_1,                0x18020018,__READ_WRITE ,__ipu_ic_idmac_1_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_2,                0x1802001C,__READ_WRITE ,__ipu_ic_idmac_2_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_3,                0x18020020,__READ_WRITE ,__ipu_ic_idmac_3_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_4,                0x18020024,__READ_WRITE ,__ipu_ic_idmac_4_bits);
__IO_REG32_BIT(IPU_CSI0_SENS_CONF,            0x18030000,__READ_WRITE ,__ipu_csi0_sens_conf_bits);
__IO_REG32_BIT(IPU_CSI0_SENS_FRM_SIZE,        0x18030004,__READ_WRITE ,__ipu_csi0_sens_frm_size_bits);
__IO_REG32_BIT(IPU_CSI0_ACT_FRM_SIZE,         0x18030008,__READ_WRITE ,__ipu_csi0_act_frm_size_bits);
__IO_REG32_BIT(IPU_CSI0_OUT_FRM_CTRL,         0x1803000C,__READ_WRITE ,__ipu_csi0_out_frm_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_TST_CTRL,             0x18030010,__READ_WRITE ,__ipu_csi0_tst_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_1,          0x18030014,__READ_WRITE ,__ipu_csi0_ccir_code_1_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_2,          0x18030018,__READ_WRITE ,__ipu_csi0_ccir_code_2_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_3,          0x1803001C,__READ_WRITE ,__ipu_csi0_ccir_code_3_bits);
__IO_REG32_BIT(IPU_CSI0_DI,                   0x18030020,__READ_WRITE ,__ipu_csi0_di_bits);
__IO_REG32_BIT(IPU_CSI0_SKIP,                 0x18030024,__READ_WRITE ,__ipu_csi0_skip_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_CTRL,             0x18030028,__READ_WRITE ,__ipu_csi0_cpd_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_0,             0x1803002C,__READ_WRITE ,__ipu_csi0_cpd_rc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_1,             0x18030030,__READ_WRITE ,__ipu_csi0_cpd_rc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_2,             0x18030034,__READ_WRITE ,__ipu_csi0_cpd_rc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_3,             0x18030038,__READ_WRITE ,__ipu_csi0_cpd_rc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_4,             0x1803003C,__READ_WRITE ,__ipu_csi0_cpd_rc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_5,             0x18030040,__READ_WRITE ,__ipu_csi0_cpd_rc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_6,             0x18030044,__READ_WRITE ,__ipu_csi0_cpd_rc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_7,             0x18030048,__READ_WRITE ,__ipu_csi0_cpd_rc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_0,             0x1803004C,__READ_WRITE ,__ipu_csi0_cpd_rs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_1,             0x18030050,__READ_WRITE ,__ipu_csi0_cpd_rs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_2,             0x18030054,__READ_WRITE ,__ipu_csi0_cpd_rs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_3,             0x18030058,__READ_WRITE ,__ipu_csi0_cpd_rs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_0,            0x1803005C,__READ_WRITE ,__ipu_csi0_cpd_grc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_1,            0x18030060,__READ_WRITE ,__ipu_csi0_cpd_grc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_2,            0x18030064,__READ_WRITE ,__ipu_csi0_cpd_grc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_3,            0x18030068,__READ_WRITE ,__ipu_csi0_cpd_grc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_4,            0x1803006C,__READ_WRITE ,__ipu_csi0_cpd_grc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_5,            0x18030070,__READ_WRITE ,__ipu_csi0_cpd_grc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_6,            0x18030074,__READ_WRITE ,__ipu_csi0_cpd_grc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_7,            0x18030078,__READ_WRITE ,__ipu_csi0_cpd_grc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_0,            0x1803007C,__READ_WRITE ,__ipu_csi0_cpd_grs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_1,            0x18030080,__READ_WRITE ,__ipu_csi0_cpd_grs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_2,            0x18030084,__READ_WRITE ,__ipu_csi0_cpd_grs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_3,            0x18030088,__READ_WRITE ,__ipu_csi0_cpd_grs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_0,            0x1803008C,__READ_WRITE ,__ipu_csi0_cpd_gbc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_1,            0x18030090,__READ_WRITE ,__ipu_csi0_cpd_gbc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_2,            0x18030094,__READ_WRITE ,__ipu_csi0_cpd_gbc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_3,            0x18030098,__READ_WRITE ,__ipu_csi0_cpd_gbc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_4,            0x1803009C,__READ_WRITE ,__ipu_csi0_cpd_gbc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_5,            0x180300A0,__READ_WRITE ,__ipu_csi0_cpd_gbc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_6,            0x180300A4,__READ_WRITE ,__ipu_csi0_cpd_gbc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_7,            0x180300A8,__READ_WRITE ,__ipu_csi0_cpd_gbc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_0,            0x180300AC,__READ_WRITE ,__ipu_csi0_cpd_gbs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_1,            0x180300B0,__READ_WRITE ,__ipu_csi0_cpd_gbs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_2,            0x180300B4,__READ_WRITE ,__ipu_csi0_cpd_gbs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_3,            0x180300B8,__READ_WRITE ,__ipu_csi0_cpd_gbs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_0,             0x180300BC,__READ_WRITE ,__ipu_csi0_cpd_bc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_1,             0x180300C0,__READ_WRITE ,__ipu_csi0_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_2,             0x180300C4,__READ_WRITE ,__ipu_csi0_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_3,             0x180300C8,__READ_WRITE ,__ipu_csi0_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_4,             0x180300CC,__READ_WRITE ,__ipu_csi0_cpd_bc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_5,             0x180300D0,__READ_WRITE ,__ipu_csi0_cpd_bc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_6,             0x180300D4,__READ_WRITE ,__ipu_csi0_cpd_bc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_7,             0x180300D8,__READ_WRITE ,__ipu_csi0_cpd_bc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_0,             0x180300DC,__READ_WRITE ,__ipu_csi0_cpd_bs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_1,             0x180300E0,__READ_WRITE ,__ipu_csi0_cpd_bs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_2,             0x180300E4,__READ_WRITE ,__ipu_csi0_cpd_bs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_3,             0x180300E8,__READ_WRITE ,__ipu_csi0_cpd_bs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_OFFSET1,          0x180300EC,__READ_WRITE ,__ipu_csi0_cpd_offset1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_OFFSET2,          0x180300F0,__READ_WRITE ,__ipu_csi0_cpd_offset2_bits);
__IO_REG32_BIT(IPU_CSI1_SENS_CONF,            0x18038000,__READ_WRITE ,__ipu_csi1_sens_conf_bits);
__IO_REG32_BIT(IPU_CSI1_SENS_FRM_SIZE,        0x18038004,__READ_WRITE ,__ipu_csi1_sens_frm_size_bits);
__IO_REG32_BIT(IPU_CSI1_ACT_FRM_SIZE,         0x18038008,__READ_WRITE ,__ipu_csi1_act_frm_size_bits);
__IO_REG32_BIT(IPU_CSI1_OUT_FRM_CTRL,         0x1803800C,__READ_WRITE ,__ipu_csi1_out_frm_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_TST_CTRL,             0x18038010,__READ_WRITE ,__ipu_csi1_tst_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_1,          0x18038014,__READ_WRITE ,__ipu_csi1_ccir_code_1_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_2,          0x18038018,__READ_WRITE ,__ipu_csi1_ccir_code_2_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_3,          0x1803801C,__READ_WRITE ,__ipu_csi1_ccir_code_3_bits);
__IO_REG32_BIT(IPU_CSI1_DI,                   0x18038020,__READ_WRITE ,__ipu_csi1_di_bits);
__IO_REG32_BIT(IPU_CSI1_SKIP,                 0x18038024,__READ_WRITE ,__ipu_csi1_skip_bits);
//__IO_REG32_BIT(IPU_CSI1_CPD_CTRL,             0x18038028,__READ_WRITE ,__ipu_csi1_cpd_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_0,             0x1803802C,__READ_WRITE ,__ipu_csi1_cpd_rc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_1,             0x18038030,__READ_WRITE ,__ipu_csi1_cpd_rc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_2,             0x18038034,__READ_WRITE ,__ipu_csi1_cpd_rc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_3,             0x18038038,__READ_WRITE ,__ipu_csi1_cpd_rc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_4,             0x1803803C,__READ_WRITE ,__ipu_csi1_cpd_rc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_5,             0x18038040,__READ_WRITE ,__ipu_csi1_cpd_rc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_6,             0x18038044,__READ_WRITE ,__ipu_csi1_cpd_rc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_7,             0x18038048,__READ_WRITE ,__ipu_csi1_cpd_rc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_0,             0x1803804C,__READ_WRITE ,__ipu_csi1_cpd_rs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_1,             0x18038050,__READ_WRITE ,__ipu_csi1_cpd_rs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_2,             0x18038054,__READ_WRITE ,__ipu_csi1_cpd_rs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_3,             0x18038058,__READ_WRITE ,__ipu_csi1_cpd_rs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_0,            0x1803805C,__READ_WRITE ,__ipu_csi1_cpd_grc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_1,            0x18038060,__READ_WRITE ,__ipu_csi1_cpd_grc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_2,            0x18038064,__READ_WRITE ,__ipu_csi1_cpd_grc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_3,            0x18038068,__READ_WRITE ,__ipu_csi1_cpd_grc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_4,            0x1803806C,__READ_WRITE ,__ipu_csi1_cpd_grc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_5,            0x18038070,__READ_WRITE ,__ipu_csi1_cpd_grc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_6,            0x18038074,__READ_WRITE ,__ipu_csi1_cpd_grc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_7,            0x18038078,__READ_WRITE ,__ipu_csi1_cpd_grc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_0,            0x1803807C,__READ_WRITE ,__ipu_csi1_cpd_grs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_1,            0x18038080,__READ_WRITE ,__ipu_csi1_cpd_grs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_2,            0x18038084,__READ_WRITE ,__ipu_csi1_cpd_grs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_3,            0x18038088,__READ_WRITE ,__ipu_csi1_cpd_grs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_0,            0x1803808C,__READ_WRITE ,__ipu_csi1_cpd_gbc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_1,            0x18038090,__READ_WRITE ,__ipu_csi1_cpd_gbc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_2,            0x18038094,__READ_WRITE ,__ipu_csi1_cpd_gbc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_3,            0x18038098,__READ_WRITE ,__ipu_csi1_cpd_gbc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_4,            0x1803809C,__READ_WRITE ,__ipu_csi1_cpd_gbc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_5,            0x180380A0,__READ_WRITE ,__ipu_csi1_cpd_gbc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_6,            0x180380A4,__READ_WRITE ,__ipu_csi1_cpd_gbc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_7,            0x180380A8,__READ_WRITE ,__ipu_csi1_cpd_gbc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_0,            0x180380AC,__READ_WRITE ,__ipu_csi1_cpd_gbs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_1,            0x180380B0,__READ_WRITE ,__ipu_csi1_cpd_gbs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_2,            0x180380B4,__READ_WRITE ,__ipu_csi1_cpd_gbs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_3,            0x180380B8,__READ_WRITE ,__ipu_csi1_cpd_gbs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_0,             0x180380BC,__READ_WRITE ,__ipu_csi1_cpd_bc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_1,             0x180380C0,__READ_WRITE ,__ipu_csi1_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_2,             0x180380C4,__READ_WRITE ,__ipu_csi1_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_3,             0x180380C8,__READ_WRITE ,__ipu_csi1_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_4,             0x180380CC,__READ_WRITE ,__ipu_csi1_cpd_bc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_5,             0x180380D0,__READ_WRITE ,__ipu_csi1_cpd_bc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_6,             0x180380D4,__READ_WRITE ,__ipu_csi1_cpd_bc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_7,             0x180380D8,__READ_WRITE ,__ipu_csi1_cpd_bc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_0,             0x180380DC,__READ_WRITE ,__ipu_csi1_cpd_bs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_1,             0x180380E0,__READ_WRITE ,__ipu_csi1_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_2,             0x180380E4,__READ_WRITE ,__ipu_csi1_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_3,             0x180380E8,__READ_WRITE ,__ipu_csi1_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_OFFSET1,          0x180380EC,__READ_WRITE ,__ipu_csi1_cpd_offset1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_OFFSET2,          0x180380F0,__READ_WRITE ,__ipu_csi1_cpd_offset2_bits);
__IO_REG32_BIT(IPU_DI0_GENERAL,               0x18040000,__READ_WRITE ,__ipu_di0_general_bits);
__IO_REG32_BIT(IPU_DI0_BS_CLKGEN0,            0x18040004,__READ_WRITE ,__ipu_di0_bs_clkgen0_bits);
__IO_REG32_BIT(IPU_DI0_BS_CLKGEN1,            0x18040008,__READ_WRITE ,__ipu_di0_bs_clkgen1_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_1,             0x1804000C,__READ_WRITE ,__ipu_di0_sw_gen0_1_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_2,             0x18040010,__READ_WRITE ,__ipu_di0_sw_gen0_2_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_3,             0x18040014,__READ_WRITE ,__ipu_di0_sw_gen0_3_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_4,             0x18040018,__READ_WRITE ,__ipu_di0_sw_gen0_4_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_5,             0x1804001C,__READ_WRITE ,__ipu_di0_sw_gen0_5_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_6,             0x18040020,__READ_WRITE ,__ipu_di0_sw_gen0_6_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_7,             0x18040024,__READ_WRITE ,__ipu_di0_sw_gen0_7_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_8,             0x18040028,__READ_WRITE ,__ipu_di0_sw_gen0_8_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN0_9,             0x1804002C,__READ_WRITE ,__ipu_di0_sw_gen0_9_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_1,             0x18040030,__READ_WRITE ,__ipu_di0_sw_gen1_1_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_2,             0x18040034,__READ_WRITE ,__ipu_di0_sw_gen1_2_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_3,             0x18040038,__READ_WRITE ,__ipu_di0_sw_gen1_3_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_4,             0x1804003C,__READ_WRITE ,__ipu_di0_sw_gen1_4_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_5,             0x18040040,__READ_WRITE ,__ipu_di0_sw_gen1_5_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_6,             0x18040044,__READ_WRITE ,__ipu_di0_sw_gen1_6_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_7,             0x18040048,__READ_WRITE ,__ipu_di0_sw_gen1_7_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_8,             0x1804004C,__READ_WRITE ,__ipu_di0_sw_gen1_8_bits);
__IO_REG32_BIT(IPU_DI0_SW_GEN1_9,             0x18040050,__READ_WRITE ,__ipu_di0_sw_gen1_9_bits);
__IO_REG32_BIT(IPU_DI0_SYNC_AS_GEN,           0x18040054,__READ_WRITE ,__ipu_di0_sync_as_gen_bits);
__IO_REG32_BIT(IPU_DI0_DW_GEN_0,              0x18040058,__READ_WRITE ,__ipu_di0_dw_gen_0_bits);
#define IPU_DI0_DW_GEN_0_SER        IPU_DI0_DW_GEN_0
#define IPU_DI0_DW_GEN_0_SER_bit    IPU_DI0_DW_GEN_0_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_1,              0x1804005C,__READ_WRITE ,__ipu_di0_dw_gen_1_bits);
#define IPU_DI0_DW_GEN_1_SER        IPU_DI0_DW_GEN_1
#define IPU_DI0_DW_GEN_1_SER_bit    IPU_DI0_DW_GEN_1_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_2,              0x18040060,__READ_WRITE ,__ipu_di0_dw_gen_2_bits);
#define IPU_DI0_DW_GEN_2_SER        IPU_DI0_DW_GEN_2
#define IPU_DI0_DW_GEN_2_SER_bit    IPU_DI0_DW_GEN_2_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_3,              0x18040064,__READ_WRITE ,__ipu_di0_dw_gen_3_bits);
#define IPU_DI0_DW_GEN_3_SER        IPU_DI0_DW_GEN_3
#define IPU_DI0_DW_GEN_3_SER_bit    IPU_DI0_DW_GEN_3_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_4,              0x18040068,__READ_WRITE ,__ipu_di0_dw_gen_4_bits);
#define IPU_DI0_DW_GEN_4_SER        IPU_DI0_DW_GEN_4
#define IPU_DI0_DW_GEN_4_SER_bit    IPU_DI0_DW_GEN_4_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_5,              0x1804006C,__READ_WRITE ,__ipu_di0_dw_gen_5_bits);
#define IPU_DI0_DW_GEN_5_SER        IPU_DI0_DW_GEN_5
#define IPU_DI0_DW_GEN_5_SER_bit    IPU_DI0_DW_GEN_5_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_6,              0x18040070,__READ_WRITE ,__ipu_di0_dw_gen_6_bits);
#define IPU_DI0_DW_GEN_6_SER        IPU_DI0_DW_GEN_6
#define IPU_DI0_DW_GEN_6_SER_bit    IPU_DI0_DW_GEN_6_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_7,              0x18040074,__READ_WRITE ,__ipu_di0_dw_gen_7_bits);
#define IPU_DI0_DW_GEN_7_SER        IPU_DI0_DW_GEN_7
#define IPU_DI0_DW_GEN_7_SER_bit    IPU_DI0_DW_GEN_7_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_8,              0x18040078,__READ_WRITE ,__ipu_di0_dw_gen_8_bits);
#define IPU_DI0_DW_GEN_8_SER        IPU_DI0_DW_GEN_8
#define IPU_DI0_DW_GEN_8_SER_bit    IPU_DI0_DW_GEN_8_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_9,              0x1804007C,__READ_WRITE ,__ipu_di0_dw_gen_9_bits);
#define IPU_DI0_DW_GEN_9_SER        IPU_DI0_DW_GEN_9
#define IPU_DI0_DW_GEN_9_SER_bit    IPU_DI0_DW_GEN_9_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_10,             0x18040080,__READ_WRITE ,__ipu_di0_dw_gen_10_bits);
#define IPU_DI0_DW_GEN_10_SER       IPU_DI0_DW_GEN_10
#define IPU_DI0_DW_GEN_10_SER_bit   IPU_DI0_DW_GEN_10_bit
__IO_REG32_BIT(IPU_DI0_DW_GEN_11,             0x18040084,__READ_WRITE ,__ipu_di0_dw_gen_11_bits);
#define IPU_DI0_DW_GEN_11_SER       IPU_DI0_DW_GEN_11
#define IPU_DI0_DW_GEN_11_SER_bit   IPU_DI0_DW_GEN_11_bit
__IO_REG32_BIT(IPU_DI0_DW_SET0_0,             0x18040088,__READ_WRITE ,__ipu_di0_dw_set0_0_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_1,             0x1804008C,__READ_WRITE ,__ipu_di0_dw_set0_1_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_2,             0x18040090,__READ_WRITE ,__ipu_di0_dw_set0_2_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_3,             0x18040094,__READ_WRITE ,__ipu_di0_dw_set0_3_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_4,             0x18040098,__READ_WRITE ,__ipu_di0_dw_set0_4_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_5,             0x1804009C,__READ_WRITE ,__ipu_di0_dw_set0_5_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_6,             0x180400A0,__READ_WRITE ,__ipu_di0_dw_set0_6_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_7,             0x180400A4,__READ_WRITE ,__ipu_di0_dw_set0_7_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_8,             0x180400A8,__READ_WRITE ,__ipu_di0_dw_set0_8_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_9,             0x180400AC,__READ_WRITE ,__ipu_di0_dw_set0_9_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_10,            0x180400B0,__READ_WRITE ,__ipu_di0_dw_set0_10_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET0_11,            0x180400B4,__READ_WRITE ,__ipu_di0_dw_set0_11_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_0,             0x180400B8,__READ_WRITE ,__ipu_di0_dw_set1_0_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_1,             0x180400BC,__READ_WRITE ,__ipu_di0_dw_set1_1_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_2,             0x180400C0,__READ_WRITE ,__ipu_di0_dw_set1_2_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_3,             0x180400C4,__READ_WRITE ,__ipu_di0_dw_set1_3_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_4,             0x180400C8,__READ_WRITE ,__ipu_di0_dw_set1_4_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_5,             0x180400CC,__READ_WRITE ,__ipu_di0_dw_set1_5_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_6,             0x180400D0,__READ_WRITE ,__ipu_di0_dw_set1_6_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_7,             0x180400D4,__READ_WRITE ,__ipu_di0_dw_set1_7_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_8,             0x180400D8,__READ_WRITE ,__ipu_di0_dw_set1_8_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_9,             0x180400DC,__READ_WRITE ,__ipu_di0_dw_set1_9_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_10,            0x180400E0,__READ_WRITE ,__ipu_di0_dw_set1_10_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET1_11,            0x180400E4,__READ_WRITE ,__ipu_di0_dw_set1_11_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_0,             0x180400E8,__READ_WRITE ,__ipu_di0_dw_set2_0_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_1,             0x180400EC,__READ_WRITE ,__ipu_di0_dw_set2_1_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_2,             0x180400F0,__READ_WRITE ,__ipu_di0_dw_set2_2_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_3,             0x180400F4,__READ_WRITE ,__ipu_di0_dw_set2_3_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_4,             0x180400F8,__READ_WRITE ,__ipu_di0_dw_set2_4_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_5,             0x180400FC,__READ_WRITE ,__ipu_di0_dw_set2_5_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_6,             0x18040100,__READ_WRITE ,__ipu_di0_dw_set2_6_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_7,             0x18040104,__READ_WRITE ,__ipu_di0_dw_set2_7_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_8,             0x18040108,__READ_WRITE ,__ipu_di0_dw_set2_8_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_9,             0x1804010C,__READ_WRITE ,__ipu_di0_dw_set2_9_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_10,            0x18040110,__READ_WRITE ,__ipu_di0_dw_set2_10_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET2_11,            0x18040114,__READ_WRITE ,__ipu_di0_dw_set2_11_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_0,             0x18040118,__READ_WRITE ,__ipu_di0_dw_set1_0_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_1,             0x1804011C,__READ_WRITE ,__ipu_di0_dw_set1_1_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_2,             0x18040120,__READ_WRITE ,__ipu_di0_dw_set1_2_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_3,             0x18040124,__READ_WRITE ,__ipu_di0_dw_set1_3_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_4,             0x18040128,__READ_WRITE ,__ipu_di0_dw_set1_4_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_5,             0x1804012C,__READ_WRITE ,__ipu_di0_dw_set1_5_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_6,             0x18040130,__READ_WRITE ,__ipu_di0_dw_set1_6_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_7,             0x18040134,__READ_WRITE ,__ipu_di0_dw_set1_7_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_8,             0x18040138,__READ_WRITE ,__ipu_di0_dw_set1_8_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_9,             0x1804013C,__READ_WRITE ,__ipu_di0_dw_set1_9_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_10,            0x18040140,__READ_WRITE ,__ipu_di0_dw_set1_10_bits);
__IO_REG32_BIT(IPU_DI0_DW_SET3_11,            0x18040144,__READ_WRITE ,__ipu_di0_dw_set1_11_bits);
__IO_REG32_BIT(IPU_DI0_STP_REP_1,             0x18040148,__READ_WRITE ,__ipu_di0_stp_rep_1_bits);
__IO_REG32_BIT(IPU_DI0_STP_REP_2,             0x1804014C,__READ_WRITE ,__ipu_di0_stp_rep_2_bits);
__IO_REG32_BIT(IPU_DI0_STP_REP_3,             0x18040150,__READ_WRITE ,__ipu_di0_stp_rep_3_bits);
__IO_REG32_BIT(IPU_DI0_STP_REP_4,             0x18040154,__READ_WRITE ,__ipu_di0_stp_rep_4_bits);
__IO_REG32_BIT(IPU_DI0_STP_REP_9,             0x18040158,__READ_WRITE ,__ipu_di0_stp_rep_9_bits);
__IO_REG32_BIT(IPU_DI0_SER_CONF,              0x1804015C,__READ_WRITE ,__ipu_di0_ser_conf_bits);
__IO_REG32_BIT(IPU_DI0_SSC,                   0x18040160,__READ_WRITE ,__ipu_di0_ssc_bits);
__IO_REG32_BIT(IPU_DI0_POL,                   0x18040164,__READ_WRITE ,__ipu_di0_pol_bits);
__IO_REG32_BIT(IPU_DI0_AW0,                   0x18040168,__READ_WRITE ,__ipu_di0_aw0_bits);
__IO_REG32_BIT(IPU_DI0_AW1,                   0x1804016C,__READ_WRITE ,__ipu_di0_aw1_bits);
__IO_REG32_BIT(IPU_DI0_SCR_CONF,              0x18040170,__READ_WRITE ,__ipu_di0_scr_conf_bits);
__IO_REG32_BIT(IPU_DI0_STAT,                  0x18040174,__READ       ,__ipu_di0_stat_bits);
__IO_REG32_BIT(IPU_DI1_GENERAL,               0x18048000,__READ_WRITE ,__ipu_di1_general_bits);
__IO_REG32_BIT(IPU_DI1_BS_CLKGEN0,            0x18048004,__READ_WRITE ,__ipu_di1_bs_clkgen0_bits);
__IO_REG32_BIT(IPU_DI1_BS_CLKGEN1,            0x18048008,__READ_WRITE ,__ipu_di1_bs_clkgen1_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_1,             0x1804800C,__READ_WRITE ,__ipu_di1_sw_gen0_1_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_2,             0x18048010,__READ_WRITE ,__ipu_di1_sw_gen0_2_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_3,             0x18048014,__READ_WRITE ,__ipu_di1_sw_gen0_3_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_4,             0x18048018,__READ_WRITE ,__ipu_di1_sw_gen0_4_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_5,             0x1804801C,__READ_WRITE ,__ipu_di1_sw_gen0_5_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_6,             0x18048020,__READ_WRITE ,__ipu_di1_sw_gen0_6_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_7,             0x18048024,__READ_WRITE ,__ipu_di1_sw_gen0_7_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_8,             0x18048028,__READ_WRITE ,__ipu_di1_sw_gen0_8_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN0_9,             0x1804802C,__READ_WRITE ,__ipu_di1_sw_gen0_9_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_1,             0x18048030,__READ_WRITE ,__ipu_di1_sw_gen1_1_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_2,             0x18048034,__READ_WRITE ,__ipu_di1_sw_gen1_2_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_3,             0x18048038,__READ_WRITE ,__ipu_di1_sw_gen1_3_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_4,             0x1804803C,__READ_WRITE ,__ipu_di1_sw_gen1_4_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_5,             0x18048040,__READ_WRITE ,__ipu_di1_sw_gen1_5_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_6,             0x18048044,__READ_WRITE ,__ipu_di1_sw_gen1_6_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_7,             0x18048048,__READ_WRITE ,__ipu_di1_sw_gen1_7_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_8,             0x1804804C,__READ_WRITE ,__ipu_di1_sw_gen1_8_bits);
__IO_REG32_BIT(IPU_DI1_SW_GEN1_9,             0x18048050,__READ_WRITE ,__ipu_di1_sw_gen1_9_bits);
__IO_REG32_BIT(IPU_DI1_SYNC_AS_GEN,           0x18048054,__READ_WRITE ,__ipu_di1_sync_as_gen_bits);
__IO_REG32_BIT(IPU_DI1_DW_GEN_0,              0x18048058,__READ_WRITE ,__ipu_di1_dw_gen_0_bits);
#define IPU_DI1_DW_GEN_0_SER      IPU_DI1_DW_GEN_0
#define IPU_DI1_DW_GEN_0_SER_bit  IPU_DI1_DW_GEN_0_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_1,              0x1804805C,__READ_WRITE ,__ipu_di1_dw_gen_1_bits);
#define IPU_DI1_DW_GEN_1_SER      IPU_DI1_DW_GEN_1
#define IPU_DI1_DW_GEN_1_SER_bit  IPU_DI1_DW_GEN_1_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_2,              0x18048060,__READ_WRITE ,__ipu_di1_dw_gen_2_bits);
#define IPU_DI1_DW_GEN_2_SER      IPU_DI1_DW_GEN_2
#define IPU_DI1_DW_GEN_2_SER_bit  IPU_DI1_DW_GEN_2_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_3,              0x18048064,__READ_WRITE ,__ipu_di1_dw_gen_3_bits);
#define IPU_DI1_DW_GEN_3_SER      IPU_DI1_DW_GEN_3
#define IPU_DI1_DW_GEN_3_SER_bit  IPU_DI1_DW_GEN_3_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_4,              0x18048068,__READ_WRITE ,__ipu_di1_dw_gen_4_bits);
#define IPU_DI1_DW_GEN_4_SER      IPU_DI1_DW_GEN_4
#define IPU_DI1_DW_GEN_4_SER_bit  IPU_DI1_DW_GEN_4_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_5,              0x1804806C,__READ_WRITE ,__ipu_di1_dw_gen_5_bits);
#define IPU_DI1_DW_GEN_5_SER      IPU_DI1_DW_GEN_5
#define IPU_DI1_DW_GEN_5_SER_bit  IPU_DI1_DW_GEN_5_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_6,              0x18048070,__READ_WRITE ,__ipu_di1_dw_gen_6_bits);
#define IPU_DI1_DW_GEN_6_SER      IPU_DI1_DW_GEN_6
#define IPU_DI1_DW_GEN_6_SER_bit  IPU_DI1_DW_GEN_6_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_7,              0x18048074,__READ_WRITE ,__ipu_di1_dw_gen_7_bits);
#define IPU_DI1_DW_GEN_7_SER      IPU_DI1_DW_GEN_7
#define IPU_DI1_DW_GEN_7_SER_bit  IPU_DI1_DW_GEN_7_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_8,              0x18048078,__READ_WRITE ,__ipu_di1_dw_gen_8_bits);
#define IPU_DI1_DW_GEN_8_SER      IPU_DI1_DW_GEN_8
#define IPU_DI1_DW_GEN_8_SER_bit  IPU_DI1_DW_GEN_8_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_9,              0x1804807C,__READ_WRITE ,__ipu_di1_dw_gen_9_bits);
#define IPU_DI1_DW_GEN_9_SER      IPU_DI1_DW_GEN_9
#define IPU_DI1_DW_GEN_9_SER_bit  IPU_DI1_DW_GEN_9_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_10,             0x18048080,__READ_WRITE ,__ipu_di1_dw_gen_10_bits);
#define IPU_DI1_DW_GEN_10_SER     IPU_DI1_DW_GEN_10
#define IPU_DI1_DW_GEN_10_SER_bit IPU_DI1_DW_GEN_10_bit
__IO_REG32_BIT(IPU_DI1_DW_GEN_11,             0x18048084,__READ_WRITE ,__ipu_di1_dw_gen_11_bits);
#define IPU_DI1_DW_GEN_11_SER     IPU_DI1_DW_GEN_11
#define IPU_DI1_DW_GEN_11_SER_bit IPU_DI1_DW_GEN_11_bit
__IO_REG32_BIT(IPU_DI1_DW_SET0_0,             0x18048088,__READ_WRITE ,__ipu_di1_dw_set0_0_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_1,             0x1804808C,__READ_WRITE ,__ipu_di1_dw_set0_1_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_2,             0x18048090,__READ_WRITE ,__ipu_di1_dw_set0_2_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_3,             0x18048094,__READ_WRITE ,__ipu_di1_dw_set0_3_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_4,             0x18048098,__READ_WRITE ,__ipu_di1_dw_set0_4_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_5,             0x1804809C,__READ_WRITE ,__ipu_di1_dw_set0_5_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_6,             0x180480A0,__READ_WRITE ,__ipu_di1_dw_set0_6_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_7,             0x180480A4,__READ_WRITE ,__ipu_di1_dw_set0_7_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_8,             0x180480A8,__READ_WRITE ,__ipu_di1_dw_set0_8_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_9,             0x180480AC,__READ_WRITE ,__ipu_di1_dw_set0_9_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_10,            0x180480B0,__READ_WRITE ,__ipu_di1_dw_set0_10_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET0_11,            0x180480B4,__READ_WRITE ,__ipu_di1_dw_set0_11_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_0,             0x180480B8,__READ_WRITE ,__ipu_di1_dw_set1_0_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_1,             0x180480BC,__READ_WRITE ,__ipu_di1_dw_set1_1_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_2,             0x180480C0,__READ_WRITE ,__ipu_di1_dw_set1_2_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_3,             0x180480C4,__READ_WRITE ,__ipu_di1_dw_set1_3_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_4,             0x180480C8,__READ_WRITE ,__ipu_di1_dw_set1_4_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_5,             0x180480CC,__READ_WRITE ,__ipu_di1_dw_set1_5_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_6,             0x180480D0,__READ_WRITE ,__ipu_di1_dw_set1_6_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_7,             0x180480D4,__READ_WRITE ,__ipu_di1_dw_set1_7_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_8,             0x180480D8,__READ_WRITE ,__ipu_di1_dw_set1_8_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_9,             0x180480DC,__READ_WRITE ,__ipu_di1_dw_set1_9_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_10,            0x180480E0,__READ_WRITE ,__ipu_di1_dw_set1_10_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET1_11,            0x180480E4,__READ_WRITE ,__ipu_di1_dw_set1_11_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_0,             0x180480E8,__READ_WRITE ,__ipu_di1_dw_set2_0_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_1,             0x180480EC,__READ_WRITE ,__ipu_di1_dw_set2_1_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_2,             0x180480F0,__READ_WRITE ,__ipu_di1_dw_set2_2_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_3,             0x180480F4,__READ_WRITE ,__ipu_di1_dw_set2_3_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_4,             0x180480F8,__READ_WRITE ,__ipu_di1_dw_set2_4_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_5,             0x180480FC,__READ_WRITE ,__ipu_di1_dw_set2_5_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_6,             0x18048100,__READ_WRITE ,__ipu_di1_dw_set2_6_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_7,             0x18048104,__READ_WRITE ,__ipu_di1_dw_set2_7_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_8,             0x18048108,__READ_WRITE ,__ipu_di1_dw_set2_8_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_9,             0x1804810C,__READ_WRITE ,__ipu_di1_dw_set2_9_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_10,            0x18048110,__READ_WRITE ,__ipu_di1_dw_set2_10_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET2_11,            0x18048114,__READ_WRITE ,__ipu_di1_dw_set2_11_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_0,             0x18048118,__READ_WRITE ,__ipu_di1_dw_set3_0_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_1,             0x1804811C,__READ_WRITE ,__ipu_di1_dw_set3_1_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_2,             0x18048120,__READ_WRITE ,__ipu_di1_dw_set3_2_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_3,             0x18048124,__READ_WRITE ,__ipu_di1_dw_set3_3_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_4,             0x18048128,__READ_WRITE ,__ipu_di1_dw_set3_4_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_5,             0x1804812C,__READ_WRITE ,__ipu_di1_dw_set3_5_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_6,             0x18048130,__READ_WRITE ,__ipu_di1_dw_set3_6_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_7,             0x18048134,__READ_WRITE ,__ipu_di1_dw_set3_7_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_8,             0x18048138,__READ_WRITE ,__ipu_di1_dw_set3_8_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_9,             0x1804813C,__READ_WRITE ,__ipu_di1_dw_set3_9_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_10,            0x18048140,__READ_WRITE ,__ipu_di1_dw_set3_10_bits);
__IO_REG32_BIT(IPU_DI1_DW_SET3_11,            0x18048144,__READ_WRITE ,__ipu_di1_dw_set3_11_bits);
__IO_REG32_BIT(IPU_DI1_STP_REP_1,             0x18048148,__READ_WRITE ,__ipu_di1_stp_rep_1_bits);
__IO_REG32_BIT(IPU_DI1_STP_REP_2,             0x1804814C,__READ_WRITE ,__ipu_di1_stp_rep_2_bits);
__IO_REG32_BIT(IPU_DI1_STP_REP_3,             0x18048150,__READ_WRITE ,__ipu_di1_stp_rep_3_bits);
__IO_REG32_BIT(IPU_DI1_STP_REP_4,             0x18048154,__READ_WRITE ,__ipu_di1_stp_rep_4_bits);
__IO_REG32_BIT(IPU_DI1_STP_REP_9,             0x18048158,__READ_WRITE ,__ipu_di1_stp_rep_9_bits);
__IO_REG32_BIT(IPU_DI1_SER_CONF,              0x1804815C,__READ_WRITE ,__ipu_di1_ser_conf_bits);
__IO_REG32_BIT(IPU_DI1_SSC,                   0x18048160,__READ_WRITE ,__ipu_di1_ssc_bits);
__IO_REG32_BIT(IPU_DI1_POL,                   0x18048164,__READ_WRITE ,__ipu_di1_pol_bits);
__IO_REG32_BIT(IPU_DI1_AW0,                   0x18048168,__READ_WRITE ,__ipu_di1_aw0_bits);
__IO_REG32_BIT(IPU_DI1_AW1,                   0x1804816C,__READ_WRITE ,__ipu_di1_aw1_bits);
__IO_REG32_BIT(IPU_DI1_SCR_CONF,              0x18048170,__READ_WRITE ,__ipu_di1_scr_conf_bits);
__IO_REG32_BIT(IPU_DI1_STAT,                  0x18048174,__READ       ,__ipu_di1_stat_bits);
__IO_REG32_BIT(IPU_SMFC_MAP,                  0x18050000,__READ_WRITE ,__ipu_smfc_map_bits);
__IO_REG32_BIT(IPU_SMFC_WMC,                  0x18050004,__READ_WRITE ,__ipu_smfc_wmc_bits);
__IO_REG32_BIT(IPU_SMFC_BS,                   0x18050008,__READ_WRITE ,__ipu_smfc_bs_bits);
__IO_REG32_BIT(IPU_DC_READ_CH_CONF,           0x18058000,__READ_WRITE ,__ipu_dc_read_ch_conf_bits);
__IO_REG32_BIT(IPU_DC_READ_SH_ADDR,           0x18058004,__READ_WRITE ,__ipu_dc_read_sh_addr_bits);
__IO_REG32_BIT(IPU_DC_RL0_CH_0,               0x18058008,__READ_WRITE ,__ipu_dc_rl0_ch_0_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_0,               0x1805800C,__READ_WRITE ,__ipu_dc_rl1_ch_0_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_0,               0x18058014,__READ_WRITE ,__ipu_dc_rl3_ch_0_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_0,               0x18058018,__READ_WRITE ,__ipu_dc_rl4_ch_0_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF_1,           0x1805801C,__READ_WRITE ,__ipu_dc_wr_ch_conf_1_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_0,               0x18058020,__READ_WRITE ,__ipu_dc_rl2_ch_0_bits);
#define IPU_DC_WR_CH_ADDR_1       IPU_DC_RL2_CH_0
#define IPU_DC_WR_CH_ADDR_1_bit   IPU_DC_RL2_CH_0_bit
__IO_REG32_BIT(IPU_DC_RL0_CH_1,               0x18058024,__READ_WRITE ,__ipu_dc_rl0_ch_1_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_1,               0x18058028,__READ_WRITE ,__ipu_dc_rl1_ch_1_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_2,               0x1805802C,__READ_WRITE ,__ipu_dc_rl2_ch_2_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_1,               0x18058030,__READ_WRITE ,__ipu_dc_rl2_ch_1_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_1,               0x18058034,__READ_WRITE ,__ipu_dc_rl4_ch_1_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF_2,           0x18058038,__READ_WRITE ,__ipu_dc_wr_ch_conf_2_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_ADDR_2,           0x1805803C,__READ_WRITE ,__ipu_dc_wr_ch_addr_2_bits);
__IO_REG32_BIT(IPU_DC_RL0_CH_2,               0x18058040,__READ_WRITE ,__ipu_dc_rl0_ch_2_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_2,               0x18058044,__READ_WRITE ,__ipu_dc_rl1_ch_2_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_2,               0x1805804C,__READ_WRITE ,__ipu_dc_rl3_ch_2_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_2,               0x18058050,__READ_WRITE ,__ipu_dc_rl4_ch_2_bits);
__IO_REG32_BIT(IPU_DC_CMD_CH_CONF_3,          0x18058054,__READ_WRITE ,__ipu_dc_cmd_ch_conf_3_bits);
__IO_REG32_BIT(IPU_DC_CMD_CH_CONF_4,          0x18058058,__READ_WRITE ,__ipu_dc_cmd_ch_conf_4_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF_5,           0x1805805C,__READ_WRITE ,__ipu_dc_wr_ch_conf_5_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_ADDR_5,           0x18058060,__READ_WRITE ,__ipu_dc_wr_ch_addr_5_bits);
__IO_REG32_BIT(IPU_DC_RL0_CH_5,               0x18058064,__READ_WRITE ,__ipu_dc_rl0_ch_5_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_5,               0x18058068,__READ_WRITE ,__ipu_dc_rl1_ch_5_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_5,               0x1805806C,__READ_WRITE ,__ipu_dc_rl2_ch_5_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_5,               0x18058070,__READ_WRITE ,__ipu_dc_rl3_ch_5_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_5,               0x18058074,__READ_WRITE ,__ipu_dc_rl4_ch_5_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF_6,           0x18058078,__READ_WRITE ,__ipu_dc_wr_ch_conf_6_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_ADDR_6,           0x1805807C,__READ_WRITE ,__ipu_dc_wr_ch_addr_6_bits);
__IO_REG32_BIT(IPU_DC_RL0_CH_6,               0x18058080,__READ_WRITE ,__ipu_dc_rl0_ch_6_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_6,               0x18058084,__READ_WRITE ,__ipu_dc_rl1_ch_6_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_6,               0x18058088,__READ_WRITE ,__ipu_dc_rl2_ch_6_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_6,               0x1805808C,__READ_WRITE ,__ipu_dc_rl3_ch_6_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_6,               0x18058090,__READ_WRITE ,__ipu_dc_rl4_ch_6_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF1_8,          0x18058094,__READ_WRITE ,__ipu_dc_wr_ch_conf1_8_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF2_8,          0x18058098,__READ_WRITE ,__ipu_dc_wr_ch_conf2_8_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_8,               0x1805809C,__READ_WRITE ,__ipu_dc_rl1_ch_8_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_8,               0x180580A0,__READ_WRITE ,__ipu_dc_rl1_ch_8_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_8,               0x180580A4,__READ_WRITE ,__ipu_dc_rl1_ch_8_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_8,               0x180580A8,__READ_WRITE ,__ipu_dc_rl4_ch_8_bits);
__IO_REG32_BIT(IPU_DC_RL5_CH_8,               0x180580AC,__READ_WRITE ,__ipu_dc_rl4_ch_8_bits);
__IO_REG32_BIT(IPU_DC_RL6_CH_8,               0x180580B0,__READ_WRITE ,__ipu_dc_rl4_ch_8_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF1_9,          0x180580B4,__READ_WRITE ,__ipu_dc_wr_ch_conf1_9_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_CONF2_9,          0x180580B8,__READ_WRITE ,__ipu_dc_wr_ch_conf2_9_bits);
__IO_REG32_BIT(IPU_DC_RL1_CH_9,               0x180580BC,__READ_WRITE ,__ipu_dc_rl1_ch_9_bits);
__IO_REG32_BIT(IPU_DC_RL2_CH_9,               0x180580C0,__READ_WRITE ,__ipu_dc_rl1_ch_9_bits);
__IO_REG32_BIT(IPU_DC_RL3_CH_9,               0x180580C4,__READ_WRITE ,__ipu_dc_rl1_ch_9_bits);
__IO_REG32_BIT(IPU_DC_RL4_CH_9,               0x180580C8,__READ_WRITE ,__ipu_dc_rl4_ch_9_bits);
__IO_REG32_BIT(IPU_DC_RL5_CH_9,               0x180580CC,__READ_WRITE ,__ipu_dc_rl4_ch_9_bits);
__IO_REG32_BIT(IPU_DC_RL6_CH_9,               0x180580D0,__READ_WRITE ,__ipu_dc_rl4_ch_9_bits);
__IO_REG32_BIT(IPU_DC_GEN,                    0x180580D4,__READ_WRITE ,__ipu_dc_gen_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF1_0,           0x180580D8,__READ_WRITE ,__ipu_dc_disp_conf1_0_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF1_1,           0x180580DC,__READ_WRITE ,__ipu_dc_disp_conf1_1_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF1_2,           0x180580E0,__READ_WRITE ,__ipu_dc_disp_conf1_2_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF1_3,           0x180580E4,__READ_WRITE ,__ipu_dc_disp_conf1_3_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF2_0,           0x180580E8,__READ_WRITE ,__ipu_dc_disp_conf2_0_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF2_2,           0x180580F0,__READ_WRITE ,__ipu_dc_disp_conf2_2_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF2_3,           0x180580F4,__READ_WRITE ,__ipu_dc_disp_conf2_3_bits);
__IO_REG32(    IPU_DC_DI0_CONF_1,             0x180580F8,__READ_WRITE );
__IO_REG32(    IPU_DC_DI0_CONF_2,             0x180580FC,__READ_WRITE );
__IO_REG32(    IPU_DC_DI1_CONF_1,             0x18058100,__READ_WRITE );
__IO_REG32(    IPU_DC_DI1_CONF_2,             0x18058104,__READ_WRITE );
__IO_REG32_BIT(IPU_DC_MAP_CONF_0,             0x18058108,__READ_WRITE ,__ipu_dc_map_conf_0_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_1,             0x1805810C,__READ_WRITE ,__ipu_dc_map_conf_1_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_2,             0x18058110,__READ_WRITE ,__ipu_dc_map_conf_2_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_3,             0x18058114,__READ_WRITE ,__ipu_dc_map_conf_3_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_4,             0x18058118,__READ_WRITE ,__ipu_dc_map_conf_4_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_5,             0x1805811C,__READ_WRITE ,__ipu_dc_map_conf_5_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_6,             0x18058120,__READ_WRITE ,__ipu_dc_map_conf_6_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_7,             0x18058124,__READ_WRITE ,__ipu_dc_map_conf_7_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_8,             0x18058128,__READ_WRITE ,__ipu_dc_map_conf_8_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_9,             0x1805812C,__READ_WRITE ,__ipu_dc_map_conf_9_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_10,            0x18058130,__READ_WRITE ,__ipu_dc_map_conf_10_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_11,            0x18058134,__READ_WRITE ,__ipu_dc_map_conf_11_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_12,            0x18058138,__READ_WRITE ,__ipu_dc_map_conf_12_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_13,            0x1805813C,__READ_WRITE ,__ipu_dc_map_conf_13_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_14,            0x18058140,__READ_WRITE ,__ipu_dc_map_conf_14_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_15,            0x18058144,__READ_WRITE ,__ipu_dc_map_conf_15_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_16,            0x18058148,__READ_WRITE ,__ipu_dc_map_conf_16_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_17,            0x1805814C,__READ_WRITE ,__ipu_dc_map_conf_17_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_18,            0x18058150,__READ_WRITE ,__ipu_dc_map_conf_18_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_19,            0x18058154,__READ_WRITE ,__ipu_dc_map_conf_19_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_20,            0x18058158,__READ_WRITE ,__ipu_dc_map_conf_20_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_21,            0x1805815C,__READ_WRITE ,__ipu_dc_map_conf_21_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_22,            0x18058160,__READ_WRITE ,__ipu_dc_map_conf_22_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_23,            0x18058164,__READ_WRITE ,__ipu_dc_map_conf_23_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_24,            0x18058168,__READ_WRITE ,__ipu_dc_map_conf_24_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_25,            0x1805816C,__READ_WRITE ,__ipu_dc_map_conf_25_bits);
__IO_REG32_BIT(IPU_DC_MAP_CONF_26,            0x18058170,__READ_WRITE ,__ipu_dc_map_conf_26_bits);
__IO_REG32_BIT(IPU_DC_UGDE0_0,                0x18058174,__READ_WRITE ,__ipu_dc_ugde0_0_bits);
__IO_REG32_BIT(IPU_DC_UGDE0_1,                0x18058178,__READ_WRITE ,__ipu_dc_ugde0_1_bits);
__IO_REG32_BIT(IPU_DC_UGDE0_2,                0x1805817C,__READ_WRITE ,__ipu_dc_ugde0_2_bits);
__IO_REG32_BIT(IPU_DC_UGDE0_3,                0x18058180,__READ_WRITE ,__ipu_dc_ugde0_3_bits);
__IO_REG32_BIT(IPU_DC_UGDE1_0,                0x18058184,__READ_WRITE ,__ipu_dc_ugde1_0_bits);
__IO_REG32_BIT(IPU_DC_UGDE1_1,                0x18058188,__READ_WRITE ,__ipu_dc_ugde1_1_bits);
__IO_REG32_BIT(IPU_DC_UGDE1_2,                0x1805818C,__READ_WRITE ,__ipu_dc_ugde1_2_bits);
__IO_REG32_BIT(IPU_DC_UGDE1_3,                0x18058190,__READ_WRITE ,__ipu_dc_ugde1_3_bits);
__IO_REG32_BIT(IPU_DC_UGDE2_0,                0x18058194,__READ_WRITE ,__ipu_dc_ugde2_0_bits);
__IO_REG32_BIT(IPU_DC_UGDE2_1,                0x18058198,__READ_WRITE ,__ipu_dc_ugde2_1_bits);
__IO_REG32_BIT(IPU_DC_UGDE2_2,                0x1805819C,__READ_WRITE ,__ipu_dc_ugde2_2_bits);
__IO_REG32_BIT(IPU_DC_UGDE2_3,                0x180581A0,__READ_WRITE ,__ipu_dc_ugde2_3_bits);
__IO_REG32_BIT(IPU_DC_LLA0,                   0x180581B4,__READ_WRITE ,__ipu_dc_lla0_bits);
__IO_REG32_BIT(IPU_DC_LLA1,                   0x180581B8,__READ_WRITE ,__ipu_dc_lla1_bits);
__IO_REG32_BIT(IPU_DC_R_LLA0,                 0x180581BC,__READ_WRITE ,__ipu_dc_r_lla0_bits);
__IO_REG32_BIT(IPU_DC_R_LLA1,                 0x180581C0,__READ_WRITE ,__ipu_dc_r_lla1_bits);
__IO_REG32_BIT(IPU_DC_WR_CH_ADDR_5_ALT,       0x180581C4,__READ_WRITE ,__ipu_dc_wr_ch_addr_5_alt_bits);
__IO_REG32_BIT(IPU_DC_STAT,                   0x180581C8,__READ       ,__ipu_dc_stat_bits);
__IO_REG32_BIT(IPU_DC_DISP_CONF2_1,           0x180581EC,__READ_WRITE ,__ipu_dc_disp_conf2_1_bits);
__IO_REG32_BIT(IPU_DMFC_RD_CHAN,              0x18060000,__READ_WRITE ,__ipu_dmfc_rd_chan_bits);
__IO_REG32_BIT(IPU_DMFC_WR_CHAN,              0x18060004,__READ_WRITE ,__ipu_dmfc_wr_chan_bits);
__IO_REG32_BIT(IPU_DMFC_WR_CHAN_DEF,          0x18060008,__READ_WRITE ,__ipu_dmfc_wr_chan_def_bits);
__IO_REG32_BIT(IPU_DMFC_DP_CHAN,              0x1806000C,__READ_WRITE ,__ipu_dmfc_dp_chan_bits);
__IO_REG32_BIT(IPU_DMFC_DP_CHAN_DEF,          0x18060010,__READ_WRITE ,__ipu_dmfc_dp_chan_def_bits);
__IO_REG32_BIT(IPU_DMFC_GENERAL_1,            0x18060014,__READ_WRITE ,__ipu_dmfc_general_1_bits);
__IO_REG32_BIT(IPU_DMFC_GENERAL_2,            0x18060018,__READ_WRITE ,__ipu_dmfc_general_2_bits);
__IO_REG32_BIT(IPU_DMFC_IC_CTRL,              0x1806001C,__READ_WRITE ,__ipu_dmfc_ic_ctrl_bits);
__IO_REG32_BIT(IPU_DMFC_WR_CHAN_ALT,          0x18060020,__READ_WRITE ,__ipu_dmfc_wr_chan_alt_bits);
__IO_REG32_BIT(IPU_DMFC_WR_CHAN_DEF_ALT,      0x18060024,__READ_WRITE ,__ipu_dmfc_wr_chan_def_alt_bits);
__IO_REG32_BIT(IPU_DMFC_DP_CHAN_ALT,          0x18060028,__READ_WRITE ,__ipu_dmfc_dp_chan_alt_bits);
__IO_REG32_BIT(IPU_DMFC_DP_CHAN_DEF_ALT,      0x1806002C,__READ_WRITE ,__ipu_dmfc_dp_chan_def_alt_bits);
__IO_REG32_BIT(IPU_DMFC_GENERAL1_ALT,         0x18060030,__READ_WRITE ,__ipu_dmfc_general1_alt_bits);
__IO_REG32_BIT(IPU_DMFC_STAT,                 0x18060034,__READ       ,__ipu_dmfc_stat_bits);
__IO_REG32_BIT(IPU_VDI_FSIZE,                 0x18068000,__READ_WRITE ,__ipu_vdi_fsize_bits);
__IO_REG32_BIT(IPU_VDI_C,                     0x18068004,__READ_WRITE ,__ipu_vdi_c_bits);
__IO_REG32_BIT(IPU_VDI_C2,                    0x18068008,__READ_WRITE ,__ipu_vdi_c2_bits);
__IO_REG32_BIT(IPU_VDI_CMDP_1,                0x1806800C,__READ_WRITE ,__ipu_vdi_cmdp_1_bits);
__IO_REG32_BIT(IPU_VDI_CMDP_2,                0x18068010,__READ_WRITE ,__ipu_vdi_cmdp_2_bits);
__IO_REG32_BIT(IPU_VDI_PS_1,                  0x18068014,__READ_WRITE ,__ipu_vdi_ps_1_bits);
__IO_REG32_BIT(IPU_VDI_PS_2,                  0x18068018,__READ_WRITE ,__ipu_vdi_ps_2_bits);
__IO_REG32_BIT(IPU_VDI_PS_3,                  0x1806801C,__READ_WRITE ,__ipu_vdi_ps_3_bits);
__IO_REG32_BIT(IPU_VDI_PS_4,                  0x18068020,__READ_WRITE ,__ipu_vdi_ps_4_bits);
__IO_REG32_BIT(IPU_DP_COM_CONF_SYNC,          0x19040000,__READ_WRITE ,__ipu_dp_com_conf_sync_bits);
__IO_REG32_BIT(IPU_DP_Graph_Wind_CTRL_SYNC,   0x19040004,__READ_WRITE ,__ipu_dp_graph_wind_ctrl_sync_bits);
__IO_REG32_BIT(IPU_DP_FG_POS_SYNC,            0x19040008,__READ_WRITE ,__ipu_dp_fg_pos_sync_bits);
__IO_REG32_BIT(IPU_DP_CUR_POS_SYNC,           0x1904000C,__READ_WRITE ,__ipu_dp_cur_pos_sync_bits);
__IO_REG32_BIT(IPU_DP_CUR_MAP_SYNC,           0x19040010,__READ_WRITE ,__ipu_dp_cur_map_sync_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_0,         0x19040014,__READ_WRITE ,__ipu_dp_gamma_c_sync_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_1,         0x19040018,__READ_WRITE ,__ipu_dp_gamma_c_sync_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_2,         0x1904001C,__READ_WRITE ,__ipu_dp_gamma_c_sync_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_3,         0x19040020,__READ_WRITE ,__ipu_dp_gamma_c_sync_3_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_4,         0x19040024,__READ_WRITE ,__ipu_dp_gamma_c_sync_4_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_5,         0x19040028,__READ_WRITE ,__ipu_dp_gamma_c_sync_5_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_6,         0x1904002C,__READ_WRITE ,__ipu_dp_gamma_c_sync_6_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_SYNC_7,         0x19040030,__READ_WRITE ,__ipu_dp_gamma_c_sync_7_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_SYNC_0,         0x19040034,__READ_WRITE ,__ipu_dp_gamma_s_sync_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_SYNC_1,         0x19040038,__READ_WRITE ,__ipu_dp_gamma_s_sync_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_SYNC_2,         0x1904003C,__READ_WRITE ,__ipu_dp_gamma_s_sync_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_SYNC_3,         0x19040040,__READ_WRITE ,__ipu_dp_gamma_s_sync_3_bits);
__IO_REG32_BIT(IPU_DP_CSCA_SYNC_0,            0x19040044,__READ_WRITE ,__ipu_dp_csca_sync_0_bits);
__IO_REG32_BIT(IPU_DP_CSCA_SYNC_1,            0x19040048,__READ_WRITE ,__ipu_dp_csca_sync_1_bits);
__IO_REG32_BIT(IPU_DP_CSCA_SYNC_2,            0x1904004C,__READ_WRITE ,__ipu_dp_csca_sync_2_bits);
__IO_REG32_BIT(IPU_DP_CSCA_SYNC_3,            0x19040050,__READ_WRITE ,__ipu_dp_csca_sync_3_bits);
__IO_REG32_BIT(IPU_DP_SCS_SYNC_0,             0x19040054,__READ_WRITE ,__ipu_dp_scs_sync_0_bits);
__IO_REG32_BIT(IPU_DP_SCS_SYNC_1,             0x19040058,__READ_WRITE ,__ipu_dp_scs_sync_1_bits);
__IO_REG32_BIT(IPU_DP_CUR_POS_ALT,            0x1904005C,__READ_WRITE ,__ipu_dp_cur_pos_alt_bits);
__IO_REG32_BIT(IPU_DP_COM_CONF_ASYNC0,        0x19040060,__READ_WRITE ,__ipu_dp_com_conf_async0_bits);
__IO_REG32_BIT(IPU_DP_GRAPH_WIND_CTRL_ASYNC0, 0x19040064,__READ_WRITE ,__ipu_dp_graph_wind_ctrl_async0_bits);
__IO_REG32_BIT(IPU_DP_FG_POS_ASYNC0,          0x19040068,__READ_WRITE ,__ipu_dp_fg_pos_async0_bits);
__IO_REG32_BIT(IPU_DP_CUR_POS_ASYNC0,         0x1904006C,__READ_WRITE ,__ipu_dp_cur_pos_async0_bits);
__IO_REG32_BIT(IPU_DP_CUR_MAP_ASYNC0,         0x19040070,__READ_WRITE ,__ipu_dp_cur_map_async0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_0,       0x19040074,__READ_WRITE ,__ipu_dp_gamma_c_async0_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_1,       0x19040078,__READ_WRITE ,__ipu_dp_gamma_c_async0_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_2,       0x1904007C,__READ_WRITE ,__ipu_dp_gamma_c_async0_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_3,       0x19040080,__READ_WRITE ,__ipu_dp_gamma_c_async0_3_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_4,       0x19040084,__READ_WRITE ,__ipu_dp_gamma_c_async0_4_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_5,       0x19040088,__READ_WRITE ,__ipu_dp_gamma_c_async0_5_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_6,       0x1904008C,__READ_WRITE ,__ipu_dp_gamma_c_async0_6_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC0_7,       0x19040090,__READ_WRITE ,__ipu_dp_gamma_c_async0_7_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYNC0_0,       0x19040094,__READ_WRITE ,__ipu_dp_gamma_s_async0_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYNC0_1,       0x19040098,__READ_WRITE ,__ipu_dp_gamma_s_async0_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYNC0_2,       0x1904009C,__READ_WRITE ,__ipu_dp_gamma_s_async0_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYNC0_3,       0x190400A0,__READ_WRITE ,__ipu_dp_gamma_s_async0_3_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC0_0,          0x190400A4,__READ_WRITE ,__ipu_dp_csca_async0_0_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC0_1,          0x190400A8,__READ_WRITE ,__ipu_dp_csca_async0_1_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC0_2,          0x190400AC,__READ_WRITE ,__ipu_dp_csca_async0_2_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC0_3,          0x190400B0,__READ_WRITE ,__ipu_dp_csca_async0_3_bits);
__IO_REG32_BIT(IPU_DP_CSC_ASYNC0_0,           0x190400B4,__READ_WRITE ,__ipu_dp_csc_async0_0_bits);
__IO_REG32_BIT(IPU_DP_CSC_ASYNC0_1,           0x190400B8,__READ_WRITE ,__ipu_dp_csc_async0_1_bits);
__IO_REG32_BIT(IPU_DP_COM_CONF_ASYNC1,        0x190400BC,__READ_WRITE ,__ipu_dp_com_conf_async1_bits);
__IO_REG32_BIT(IPU_DP_GRAPH_WIND_CTRL_ASYNC1, 0x190400C0,__READ_WRITE ,__ipu_dp_graph_wind_ctrl_async1_bits);
__IO_REG32_BIT(IPU_DP_FG_POS_ASYNC1,          0x190400C4,__READ_WRITE ,__ipu_dp_fg_pos_async1_bits);
__IO_REG32_BIT(IPU_DP_CUR_POS_ASYNC1,         0x190400C8,__READ_WRITE ,__ipu_dp_cur_pos_async1_bits);
__IO_REG32_BIT(IPU_DP_CUR_MAP_ASYNC1,         0x190400CC,__READ_WRITE ,__ipu_dp_cur_map_async1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_0,       0x190400D0,__READ_WRITE ,__ipu_dp_gamma_c_async1_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_1,       0x190400D4,__READ_WRITE ,__ipu_dp_gamma_c_async1_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_2,       0x190400D8,__READ_WRITE ,__ipu_dp_gamma_c_async1_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_3,       0x190400DC,__READ_WRITE ,__ipu_dp_gamma_c_async1_3_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_4,       0x190400E0,__READ_WRITE ,__ipu_dp_gamma_c_async1_4_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_5,       0x190400E4,__READ_WRITE ,__ipu_dp_gamma_c_async1_5_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_6,       0x190400E8,__READ_WRITE ,__ipu_dp_gamma_c_async1_6_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_C_ASYNC1_7,       0x190400EC,__READ_WRITE ,__ipu_dp_gamma_c_async1_7_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYN1_0,        0x190400F0,__READ_WRITE ,__ipu_dp_gamma_s_asyn1_0_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYN1_1,        0x190400F4,__READ_WRITE ,__ipu_dp_gamma_s_asyn1_1_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYN1_2,        0x190400F8,__READ_WRITE ,__ipu_dp_gamma_s_asyn1_2_bits);
__IO_REG32_BIT(IPU_DP_GAMMA_S_ASYN1_3,        0x190400FC,__READ_WRITE ,__ipu_dp_gamma_s_asyn1_3_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC1_0,          0x19040100,__READ_WRITE ,__ipu_dp_csca_async1_0_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC1_1,          0x19040104,__READ_WRITE ,__ipu_dp_csca_async1_1_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC1_2,          0x19040108,__READ_WRITE ,__ipu_dp_csca_async1_2_bits);
__IO_REG32_BIT(IPU_DP_CSCA_ASYNC1_3,          0x1904010C,__READ_WRITE ,__ipu_dp_csca_async1_3_bits);
__IO_REG32_BIT(IPU_DP_CSC_ASYNC1_0,           0x19040110,__READ_WRITE ,__ipu_dp_csc_async1_0_bits);
__IO_REG32_BIT(IPU_DP_CSC_ASYNC1_1,           0x19040114,__READ_WRITE ,__ipu_dp_csc_async1_1_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPCR,                      0x53F94000,__READ_WRITE ,__kpcr_bits);
__IO_REG16_BIT(KPSR,                      0x53F94002,__READ_WRITE ,__kpsr_bits);
__IO_REG16_BIT(KDDR,                      0x53F94004,__READ_WRITE ,__kddr_bits);
__IO_REG16_BIT(KPDR,                      0x53F94006,__READ_WRITE ,__kpdr_bits);

/***************************************************************************
 **
 **  M4IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M4IF_PSM0,             0x63FD8000,__READ_WRITE ,__m4if_psm0_bits);
__IO_REG32_BIT(M4IF_PSM1,             0x63FD8004,__READ_WRITE ,__m4if_psm1_bits);
__IO_REG32(    M4IF_GENP,             0x63FD800C,__READ_WRITE );
__IO_REG32_BIT(M4IF_MDSR6,            0x63FD8018,__READ       ,__m4if_mdsr6_bits);
__IO_REG32(    M4IF_MDSR7,            0x63FD801C,__READ       );
__IO_REG32(    M4IF_MDSR8,            0x63FD8020,__READ       );
__IO_REG32_BIT(M4IF_MDSR0,            0x63FD8024,__READ       ,__m4if_mdsr0_bits);
__IO_REG32_BIT(M4IF_MDSR1,            0x63FD8028,__READ       ,__m4if_mdsr1_bits);
__IO_REG32(    M4IF_MDSR2,            0x63FD802C,__READ       );
__IO_REG32(    M4IF_MDSR3,            0x63FD8030,__READ       );
__IO_REG32(    M4IF_MDSR4,            0x63FD8034,__READ       );
__IO_REG32(    M4IF_MDSR5,            0x63FD8038,__READ       );
__IO_REG32_BIT(M4IF_FBPM0,            0x63FD8040,__READ_WRITE ,__m4if_fbpm0_bits);
__IO_REG32_BIT(M4IF_FBPM1,            0x63FD8044,__READ_WRITE ,__m4if_fbpm1_bits);
__IO_REG32_BIT(M4IF_CTRL,             0x63FD8048,__READ_WRITE ,__m4if_ctrl_bits);
__IO_REG32_BIT(M4IF_I2ULA,            0x63FD8074,__READ_WRITE ,__m4if_i2ula_bits);
__IO_REG32_BIT(M4IF_RINT2,            0x63FD807C,__READ_WRITE ,__m4if_rint2_bits);
__IO_REG32(    M4IF_SBS0,             0x63FD8084,__READ       );
__IO_REG32_BIT(M4IF_SBS1,             0x63FD8088,__READ_WRITE ,__m4if_sbs1_bits);
__IO_REG32_BIT(M4IF_MCR0,             0x63FD808C,__READ_WRITE ,__m4if_mcr0_bits);
__IO_REG32_BIT(M4IF_MCR1,             0x63FD8090,__READ_WRITE ,__m4if_mcr1_bits);
__IO_REG32_BIT(M4IF_MDCR,             0x63FD8094,__READ_WRITE ,__m4if_mdcr_bits);
__IO_REG32_BIT(M4IF_FACR,             0x63FD8098,__READ_WRITE ,__m4if_facr_bits);
__IO_REG32_BIT(M4IF_FPWC,             0x63FD809C,__READ_WRITE ,__m4if_fpwc_bits);
__IO_REG32_BIT(M4IF_PSM2,             0x63FD80A4,__READ_WRITE ,__m4if_psm2_bits);
__IO_REG32_BIT(M4IF_PSM3,             0x63FD80AC,__READ_WRITE ,__m4if_psm3_bits);
__IO_REG32_BIT(M4IF_FULA,             0x63FD80B0,__READ_WRITE ,__m4if_fula_bits);
__IO_REG32_BIT(M4IF_SULA,             0x63FD80B4,__READ_WRITE ,__m4if_sula_bits);
__IO_REG32_BIT(M4IF_IULA,             0x63FD80B8,__READ_WRITE ,__m4if_iula_bits);
__IO_REG32_BIT(M4IF_FDPS,             0x63FD80BC,__READ       ,__m4if_fdps_bits);
__IO_REG32_BIT(M4IF_FDPC,             0x63FD80C0,__READ_WRITE ,__m4if_fdpc_bits);
__IO_REG32_BIT(M4IF_MLEN,             0x63FD80C4,__READ_WRITE ,__m4if_mlen_bits);
__IO_REG32_BIT(M4IF_WMIS0,            0x63FD8114,__READ_WRITE ,__m4if_wmis0_bits);
__IO_REG32(    M4IF_WMVA0,            0x63FD8118,__READ       );
__IO_REG32_BIT(M4IF_WMSA1_0,          0x63FD8120,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_1,          0x63FD8124,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_2,          0x63FD8128,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_3,          0x63FD812c,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_4,          0x63FD8130,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_5,          0x63FD8134,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_6,          0x63FD8138,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMSA1_7,          0x63FD813c,__READ_WRITE ,__m4if_wmsa1_bits);
__IO_REG32_BIT(M4IF_WMEA1_0,          0x63FD8140,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_1,          0x63FD8144,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_2,          0x63FD8148,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_3,          0x63FD814C,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_4,          0x63FD8150,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_5,          0x63FD8154,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_6,          0x63FD8158,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMEA1_7,          0x63FD815C,__READ_WRITE ,__m4if_wmea1_bits);
__IO_REG32_BIT(M4IF_WMIS1,            0x63FD8160,__READ_WRITE ,__m4if_wmis1_bits);
__IO_REG32(    M4IF_WMVA1,            0x63FD8164,__READ       );

/***************************************************************************
 **
 **  MLB
 **
 ***************************************************************************/
__IO_REG32_BIT(MLB_DCCR,              0x63FE4000,__READ_WRITE ,__mlb_dccr_bits);
__IO_REG32_BIT(MLB_SSCR,              0x63FE4004,__READ_WRITE ,__mlb_sscr_bits);
__IO_REG32(    MLB_SDCR,              0x63FE4008,__READ       );
__IO_REG32_BIT(MLB_SMCR,              0x63FE400C,__READ_WRITE ,__mlb_smcr_bits);
__IO_REG32_BIT(MLB_VCCR,              0x63FE401C,__READ       ,__mlb_vccr_bits);
__IO_REG32_BIT(MLB_SBCR,              0x63FE4020,__READ_WRITE ,__mlb_sbcr_bits);
__IO_REG32_BIT(MLB_ABCR,              0x63FE4024,__READ_WRITE ,__mlb_abcr_bits);
__IO_REG32_BIT(MLB_CBCR,              0x63FE4028,__READ_WRITE ,__mlb_cbcr_bits);
__IO_REG32_BIT(MLB_IBCR,              0x63FE402C,__READ_WRITE ,__mlb_ibcr_bits);
__IO_REG32_BIT(MLB_CICR,              0x63FE4030,__READ_WRITE ,__mlb_cicr_bits);
__IO_REG32_BIT(MLB_CECR0,             0x63FE4040,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR0,             0x63FE4044,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR0,            0x63FE4048,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR0,            0x63FE404C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR0,            0x63FE4280,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR1,             0x63FE4050,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR1,             0x63FE4054,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR1,            0x63FE4058,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR1,            0x63FE405C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR1,            0x63FE4284,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR2,             0x63FE4060,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR2,             0x63FE4064,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR2,            0x63FE4068,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR2,            0x63FE406C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR2,            0x63FE4288,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR3,             0x63FE4070,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR3,             0x63FE4074,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR3,            0x63FE4078,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR3,            0x63FE407C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR3,            0x63FE428C,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR4,             0x63FE4080,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR4,             0x63FE4084,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR4,            0x63FE4088,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR4,            0x63FE408C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR4,            0x63FE4290,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR5,             0x63FE4090,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR5,             0x63FE4094,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR5,            0x63FE4098,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR5,            0x63FE409C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR5,            0x63FE4294,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR6,             0x63FE40A0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR6,             0x63FE40A4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR6,            0x63FE40A8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR6,            0x63FE40AC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR6,            0x63FE4298,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR7,             0x63FE40B0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR7,             0x63FE40B4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR7,            0x63FE40B8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR7,            0x63FE40BC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR7,            0x63FE429C,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR8,             0x63FE40C0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR8,             0x63FE40C4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR8,            0x63FE40C8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR8,            0x63FE40CC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR8,            0x63FE42A0,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR9,             0x63FE40D0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR9,             0x63FE40D4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR9,            0x63FE40D8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR9,            0x63FE40DC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR9,            0x63FE42A4,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR10,            0x63FE40E0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR10,            0x63FE40E4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR10,           0x63FE40E8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR10,           0x63FE40EC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR10,           0x63FE42A8,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR11,            0x63FE40F0,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR11,            0x63FE40F4,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR11,           0x63FE40F8,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR11,           0x63FE40FC,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR11,           0x63FE42AC,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR12,            0x63FE4100,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR12,            0x63FE4104,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR12,           0x63FE4108,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR12,           0x63FE410C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR12,           0x63FE42B0,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR13,            0x63FE4110,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR13,            0x63FE4114,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR13,           0x63FE4118,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR13,           0x63FE411C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR13,           0x63FE42B4,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR14,            0x63FE4120,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR14,            0x63FE4124,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR14,           0x63FE4128,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR14,           0x63FE412C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR14,           0x63FE42B8,__READ_WRITE ,__mlb_lcbcr_bits);
__IO_REG32_BIT(MLB_CECR15,            0x63FE4130,__READ_WRITE ,__mlb_cecr_bits);
__IO_REG32_BIT(MLB_CSCR15,            0x63FE4134,__READ_WRITE ,__mlb_cscr_bits);
__IO_REG32_BIT(MLB_CCBCR15,           0x63FE4138,__READ_WRITE ,__mlb_ccbcr_bits);
__IO_REG32_BIT(MLB_CNBCR15,           0x63FE413C,__READ_WRITE ,__mlb_cnbcr_bits);
__IO_REG32_BIT(MLB_LCBCR15,           0x63FE42BC,__READ_WRITE ,__mlb_lcbcr_bits);

/***************************************************************************
 **
 **  NFC
 **
 ***************************************************************************/
__IO_REG32_BIT(NFC_NAND_CMD,          0xF7FF1E00,__READ_WRITE ,__nfc_nand_cmd_bits);
__IO_REG32(    NFC_NAND_ADD0,         0xF7FF1E04,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD1,         0xF7FF1E08,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD2,         0xF7FF1E0C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD3,         0xF7FF1E10,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD4,         0xF7FF1E14,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD5,         0xF7FF1E18,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD6,         0xF7FF1E1C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD7,         0xF7FF1E20,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD8,         0xF7FF1E24,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD9,         0xF7FF1E28,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD10,        0xF7FF1E2C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD11,        0xF7FF1E30,__READ_WRITE );
__IO_REG32_BIT(NFC_CONFIGURATION1,    0xF7FF1E34,__READ_WRITE ,__nfc_configuration1_bits);
__IO_REG32_BIT(NFC_ECC_STATUS_RESULT, 0xF7FF1E38,__READ       ,__nfc_ecc_status_result_bits);
__IO_REG32_BIT(NFC_STATUS_SUM,        0xF7FF1E3C,__READ_WRITE ,__nfc_status_sum_bits);
__IO_REG32_BIT(NFC_LAUNCH,            0xF7FF1E40,__READ_WRITE ,__nfc_launch_bits);
__IO_REG32_BIT(NFC_WR_PROTECT,        0x63FDB000,__READ_WRITE ,__nfc_wr_protect_bits);
__IO_REG32_BIT(NFC_CONFIGURATION2,    0x63FDB024,__READ_WRITE ,__nfc_configuration2_bits);
__IO_REG32_BIT(NFC_CONFIGURATION3,    0x63FDB028,__READ_WRITE ,__nfc_configuration3_bits);
__IO_REG32_BIT(NFC_IPC,               0x63FDB02C,__READ_WRITE ,__nfc_ipc_bits);
__IO_REG32_BIT(NFC_AXI_ERR_ADD,       0x63FDB030,__READ_WRITE ,__nfc_axi_err_add_bits);
__IO_REG32_BIT(NFC_DELAY_LINE,        0x63FDB034,__READ_WRITE ,__nfc_delay_line_bits);

/***************************************************************************
 **
 **  OWIRE
 **
 ***************************************************************************/
__IO_REG16_BIT(OWIRE_CONTROL,         0x63FA4000,__READ_WRITE ,__owire_control_bits);
__IO_REG16_BIT(OWIRE_TIME_DIVIDER,    0x63FA4002,__READ_WRITE ,__owire_time_divider_bits);
__IO_REG16_BIT(OWIRE_RESET,           0x63FA4004,__READ_WRITE ,__owire_reset_bits);
__IO_REG16_BIT(OWIRE_COMMAND,         0x63FA4006,__READ_WRITE ,__owire_command_bits);
__IO_REG16_BIT(OWIRE_TX_RX,           0x63FA4008,__READ_WRITE ,__owire_tx_rx_bits);
__IO_REG16_BIT(OWIRE_INTERRUPT,       0x63FA400A,__READ       ,__owire_interrupt_bits);
__IO_REG16_BIT(OWIRE_INTERRUPT_EN,    0x63FA400C,__READ_WRITE ,__owire_interrupt_en_bits);

/***************************************************************************
 **
 **  ATA
 **
 ***************************************************************************/
__IO_REG8(     PATA_TIME_OFF,           0x63FE0000,__READ_WRITE );
__IO_REG8(     PATA_TIME_ON,            0x63FE0001,__READ_WRITE );
__IO_REG8(     PATA_TIME_1,             0x63FE0002,__READ_WRITE );
__IO_REG8(     PATA_TIME_2W,            0x63FE0003,__READ_WRITE );
__IO_REG8(     PATA_TIME_2R,            0x63FE0004,__READ_WRITE );
__IO_REG8(     PATA_TIME_AX,            0x63FE0005,__READ_WRITE );
__IO_REG8(     PATA_TIME_PIO_RDX,       0x63FE0006,__READ_WRITE );
__IO_REG8(     PATA_TIME_4,             0x63FE0007,__READ_WRITE );
__IO_REG8(     PATA_TIME_9,             0x63FE0008,__READ_WRITE );
__IO_REG8(     PATA_TIME_M,             0x63FE0009,__READ_WRITE );
__IO_REG8(     PATA_TIME_JN,            0x63FE000A,__READ_WRITE );
__IO_REG8(     PATA_TIME_D,             0x63FE000B,__READ_WRITE );
__IO_REG8(     PATA_TIME_K,             0x63FE000C,__READ_WRITE );
__IO_REG8(     PATA_TIME_ACK,           0x63FE000D,__READ_WRITE );
__IO_REG8(     PATA_TIME_ENV,           0x63FE000E,__READ_WRITE );
__IO_REG8(     PATA_TIME_RPX,           0x63FE000F,__READ_WRITE );
__IO_REG8(     PATA_TIME_ZAH,           0x63FE0010,__READ_WRITE );
__IO_REG8(     PATA_TIME_MLIX,          0x63FE0011,__READ_WRITE );
__IO_REG8(     PATA_TIME_DVH,           0x63FE0012,__READ_WRITE );
__IO_REG8(     PATA_TIME_DZFS,          0x63FE0013,__READ_WRITE );
__IO_REG8(     PATA_TIME_DVS,           0x63FE0014,__READ_WRITE );
__IO_REG8(     PATA_TIME_CVH,           0x63FE0015,__READ_WRITE );
__IO_REG8(     PATA_TIME_SS,            0x63FE0016,__READ_WRITE );
__IO_REG8(     PATA_TIME_CYC,           0x63FE0017,__READ_WRITE );
__IO_REG8(     PATA_FIFO_DATA_32,       0x63FE0018,__READ_WRITE );
__IO_REG8(     PATA_FIFO_DATA_16,       0x63FE001C,__READ_WRITE );
__IO_REG8(     PATA_FIFO_FILL,          0x63FE0020,__READ       );
__IO_REG8_BIT( PATA_CONTROL,            0x63FE0024,__READ_WRITE ,__ata_control_bits);
__IO_REG8_BIT( PATA_INTR_PEND,          0x63FE0028,__READ       ,__ata_intr_pend_bits);
__IO_REG8_BIT( PATA_INTR_ENA,           0x63FE002C,__READ_WRITE ,__ata_intr_ena_bits);
__IO_REG8_BIT( PATA_INTR_CLR,           0x63FE0030,__WRITE      ,__ata_intr_clr_bits);
__IO_REG8(     PATA_FIFO_ALARM,         0x63FE0034,__READ_WRITE );
__IO_REG16(    PATA_DRIVE_DATA,         0x63FE00A0,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_FEATURES,     0x63FE00A4,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_SECTOR_COUNT, 0x63FE00A8,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_SECTOR_NUM,   0x63FE00AC,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_CYL_LOW,      0x63FE00B0,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_CYL_HIGH,     0x63FE00B4,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_DEV_HEAD,     0x63FE00B8,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_COMMAND,      0x63FE00BC,__READ_WRITE );
#define PATA_DRIVE_STATUS   PATA_DRIVE_COMMAND
__IO_REG32(    PATA_DRIVE_ALT_STATUS,   0x63FE00D8,__READ_WRITE );
#define PATA_DRIVE_CONTROL  PATA_DRIVE_ALT_STATUS

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1CR,                  0x53FB4000,__READ_WRITE ,__pwmcr_bits);
__IO_REG32_BIT(PWM1SR,                  0x53FB4004,__READ_WRITE ,__pwmsr_bits);
__IO_REG32_BIT(PWM1IR,                  0x53FB4008,__READ_WRITE ,__pwmir_bits);
__IO_REG32_BIT(PWM1SAR,                 0x53FB400C,__READ_WRITE ,__pwmsar_bits);
__IO_REG32_BIT(PWM1PR,                  0x53FB4010,__READ_WRITE ,__pwmpr_bits);
__IO_REG32_BIT(PWM1CNR,                 0x53FB4014,__READ       ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2CR,                  0x53FB8000,__READ_WRITE ,__pwmcr_bits);
__IO_REG32_BIT(PWM2SR,                  0x53FB8004,__READ_WRITE ,__pwmsr_bits);
__IO_REG32_BIT(PWM2IR,                  0x53FB8008,__READ_WRITE ,__pwmir_bits);
__IO_REG32_BIT(PWM2SAR,                 0x53FB800C,__READ_WRITE ,__pwmsar_bits);
__IO_REG32_BIT(PWM2PR,                  0x53FB8010,__READ_WRITE ,__pwmpr_bits);
__IO_REG32_BIT(PWM2CNR,                 0x53FB8014,__READ       ,__pwmcnr_bits);

/***************************************************************************
 **
 **  ROMC
 **
 ***************************************************************************/
__IO_REG32(    ROMC_ROMPATCHD7,         0x600000D4,__READ_WRITE );
__IO_REG32(    ROMC_ROMPATCHD0,         0x600000F0,__READ_WRITE );
__IO_REG32_BIT(ROMC_ROMPATCHCNTL,       0x600000F4,__READ_WRITE ,__romc_rompatchcntl_bits);
__IO_REG32_BIT(ROMC_ROMPATCHENL,        0x600000FC,__READ_WRITE ,__romc_rompatchenl_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA0,         0x60000100,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA1,         0x60000104,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA2,         0x60000108,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA3,         0x6000010C,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA4,         0x60000110,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA5,         0x60000114,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA6,         0x60000118,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA7,         0x6000011C,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA8,         0x60000120,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA9,         0x60000124,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA10,        0x60000128,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA11,        0x6000012C,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA12,        0x60000130,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA13,        0x60000134,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA14,        0x60000138,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHA15,        0x6000013C,__READ_WRITE ,__romc_rompatcha_bits);
__IO_REG32_BIT(ROMC_ROMPATCHSR,         0x60000208,__READ_WRITE ,__romc_rompatchsr_bits);

/***************************************************************************
 **
 **  SDMA
 **
 ***************************************************************************/
__IO_REG32(    SDMA_MC0PTR,               0x63FB0000,__READ_WRITE );
__IO_REG32_BIT(SDMA_INTR,                 0x63FB0004,__READ_WRITE ,__sdma_intr_bits);
__IO_REG32_BIT(SDMA_STOP_STAT,            0x63FB0008,__READ       ,__sdma_stop_stat_bits);
__IO_REG32_BIT(SDMA_HSTART,               0x63FB000C,__READ_WRITE ,__sdma_hstart_bits);
__IO_REG32_BIT(SDMA_EVTOVR,               0x63FB0010,__READ_WRITE ,__sdma_evtovr_bits);
__IO_REG32_BIT(SDMA_DSPOVR,               0x63FB0014,__READ_WRITE ,__sdma_dspovr_bits);
__IO_REG32_BIT(SDMA_HOSTOVR,              0x63FB0018,__READ_WRITE ,__sdma_hostovr_bits);
__IO_REG32_BIT(SDMA_EVTPEND,              0x63FB001C,__READ       ,__sdma_evtpend_bits);
__IO_REG32_BIT(SDMA_RESET,                0x63FB0024,__READ       ,__sdma_reset_bits);
__IO_REG32_BIT(SDMA_EVTERR,               0x63FB0028,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_INTRMASK,             0x63FB002C,__READ_WRITE ,__sdma_intrmask_bits);
__IO_REG32_BIT(SDMA_PSW,                  0x63FB0030,__READ       ,__sdma_psw_bits);
__IO_REG32_BIT(SDMA_EVTERRDBG,            0x63FB0034,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_CONFIG,               0x63FB0038,__READ_WRITE ,__sdma_config_bits);
__IO_REG32_BIT(SDMA_LOCK,                 0x63FB003C,__READ_WRITE ,__sdma_lock_bits);
__IO_REG32_BIT(SDMA_ONCE_ENB,             0x63FB0040,__READ_WRITE ,__sdma_once_enb_bits);
__IO_REG32(    SDMA_ONCE_DATA,            0x63FB0044,__READ_WRITE );
__IO_REG32_BIT(SDMA_ONCE_INSTR,           0x63FB0048,__READ_WRITE ,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMA_ONCE_STAT,            0x63FB004C,__READ       ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMA_ONCE_CMD,             0x63FB0050,__READ_WRITE ,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMA_ILLINSTADDR,          0x63FB0058,__READ_WRITE ,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMA_CHN0ADDR,             0x63FB005C,__READ_WRITE ,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR,           0x63FB0060,__READ       ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR2,          0x63FB0064,__READ       ,__sdma_evt_mirror2_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF1,          0x63FB0070,__READ_WRITE ,__sdma_xtrig1_conf_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF2,          0x63FB0074,__READ_WRITE ,__sdma_xtrig2_conf_bits);
__IO_REG32_BIT(SDMA_CHNPRI0,              0x63FB0100,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI1,              0x63FB0104,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI2,              0x63FB0108,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI3,              0x63FB010C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI4,              0x63FB0110,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI5,              0x63FB0114,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI6,              0x63FB0118,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI7,              0x63FB011C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI8,              0x63FB0120,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI9,              0x63FB0124,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI10,             0x63FB0128,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI11,             0x63FB012C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI12,             0x63FB0130,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI13,             0x63FB0134,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI14,             0x63FB0138,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI15,             0x63FB013C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI16,             0x63FB0140,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI17,             0x63FB0144,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI18,             0x63FB0148,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI19,             0x63FB014C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI20,             0x63FB0150,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI21,             0x63FB0154,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI22,             0x63FB0158,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI23,             0x63FB015C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI24,             0x63FB0160,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI25,             0x63FB0164,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI26,             0x63FB0168,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI27,             0x63FB016C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI28,             0x63FB0170,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI29,             0x63FB0174,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI30,             0x63FB0178,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI31,             0x63FB017C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32(    SDMA_CHNENBL0,             0x63FB0200,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL1,             0x63FB0204,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL2,             0x63FB0208,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL3,             0x63FB020C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL4,             0x63FB0210,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL5,             0x63FB0214,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL6,             0x63FB0218,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL7,             0x63FB021C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL8,             0x63FB0220,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL9,             0x63FB0224,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL10,            0x63FB0228,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL11,            0x63FB022C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL12,            0x63FB0230,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL13,            0x63FB0234,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL14,            0x63FB0238,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL15,            0x63FB023C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL16,            0x63FB0240,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL17,            0x63FB0244,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL18,            0x63FB0248,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL19,            0x63FB024C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL20,            0x63FB0250,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL21,            0x63FB0254,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL22,            0x63FB0258,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL23,            0x63FB025C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL24,            0x63FB0260,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL25,            0x63FB0264,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL26,            0x63FB0268,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL27,            0x63FB026C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL28,            0x63FB0270,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL29,            0x63FB0274,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL30,            0x63FB0278,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL31,            0x63FB027C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL32,            0x63FB0280,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL33,            0x63FB0284,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL34,            0x63FB0288,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL35,            0x63FB028C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL36,            0x63FB0290,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL37,            0x63FB0294,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL38,            0x63FB0298,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL39,            0x63FB029C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL40,            0x63FB02A0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL41,            0x63FB02A4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL42,            0x63FB02A8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL43,            0x63FB02AC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL44,            0x63FB02B0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL45,            0x63FB02B4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL46,            0x63FB02B8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL47,            0x63FB02BC,__READ_WRITE );

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
 **  SPDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(SPDIF_SCR,                 0x50028000,__READ_WRITE ,__spdif_scr_bits);
__IO_REG32_BIT(SPDIF_SRCD,                0x50028004,__READ_WRITE ,__spdif_srcd_bits);
__IO_REG32_BIT(SPDIF_SRPC,                0x50028008,__READ_WRITE ,__spdif_srpc_bits);
__IO_REG32_BIT(SPDIF_SIE,                 0x5002800C,__READ_WRITE ,__spdif_sie_bits);
__IO_REG32_BIT(SPDIF_SI,                  0x50028010,__READ_WRITE ,__spdif_sie_bits);
__IO_REG32(    SPDIF_SRL,                 0x50028014,__READ       );
__IO_REG32(    SPDIF_SRR,                 0x50028018,__READ       );
__IO_REG32(    SPDIF_SRCSH,               0x5002801C,__READ       );
__IO_REG32(    SPDIF_SRCSL,               0x50028020,__READ       );
__IO_REG32(    SPDIF_SQU,                 0x50028024,__READ       );
__IO_REG32(    SPDIF_SRQ,                 0x50028028,__READ       );
__IO_REG32(    SPDIF_STL,                 0x5002802C,__WRITE      );
__IO_REG32(    SPDIF_STR,                 0x50028030,__WRITE      );
__IO_REG32(    SPDIF_STCSCH,              0x50028034,__READ_WRITE );
__IO_REG32(    SPDIF_STCSCl,              0x50028038,__READ_WRITE );
__IO_REG32(    SPDIF_SRFM,                0x50028044,__READ       );
__IO_REG32_BIT(SPDIF_STC,                 0x50028050,__READ_WRITE ,__spdif_stc_bits);

/***************************************************************************
 **
 **  SRC
 **
 ***************************************************************************/
__IO_REG32_BIT(SRC_SCR,                   0x53FD0000,__READ_WRITE ,__src_scr_bits);
__IO_REG32_BIT(SRC_SBMR,                  0x53FD0004,__READ_WRITE ,__src_sbmr_bits);
__IO_REG32_BIT(SRC_SRSR,                  0x53FD0008,__READ_WRITE ,__src_srsr_bits);
__IO_REG32_BIT(SRC_SISR,                  0x53FD0014,__READ       ,__src_sisr_bits);
__IO_REG32_BIT(SRC_SIMR,                  0x53FD0018,__READ_WRITE ,__src_simr_bits);

/***************************************************************************
 **
 **  SRTC
 **
 ***************************************************************************/
__IO_REG32(    SRTC_LPSCMR,               0x53FA4000,__READ_WRITE );
__IO_REG32_BIT(SRTC_LPSCLR,               0x53FA4004,__READ_WRITE ,__srtc_lpsclr_bits);
__IO_REG32(    SRTC_LPSAR,                0x53FA4008,__READ_WRITE );
__IO_REG32(    SRTC_LPSMCR,               0x53FA400C,__READ_WRITE );
__IO_REG32_BIT(SRTC_LPCR,                 0x53FA4010,__READ_WRITE ,__srtc_lpcr_bits);
__IO_REG32_BIT(SRTC_LPSR,                 0x53FA4014,__READ_WRITE ,__srtc_lpsr_bits);
__IO_REG32(    SRTC_LPPDR,                0x53FA4018,__READ_WRITE );
__IO_REG32_BIT(SRTC_LPGR,                 0x53FA401C,__READ_WRITE ,__srtc_lpgr_bits);
__IO_REG32(    SRTC_HPCMR,                0x53FA4020,__READ_WRITE );
__IO_REG32_BIT(SRTC_HPCLR,                0x53FA4024,__READ_WRITE ,__srtc_hpclr_bits);
__IO_REG32(    SRTC_HPAMR,                0x53FA4028,__READ_WRITE );
__IO_REG32_BIT(SRTC_HPALR,                0x53FA402C,__READ_WRITE ,__srtc_hpalr_bits);
__IO_REG32_BIT(SRTC_HPCR,                 0x53FA4030,__READ_WRITE ,__srtc_hpcr_bits);
__IO_REG32_BIT(SRTC_HPISR,                0x53FA4034,__READ_WRITE ,__srtc_hpisr_bits);
__IO_REG32_BIT(SRTC_HPIENR,               0x53FA4038,__READ_WRITE ,__srtc_hpienr_bits);

/***************************************************************************
 **
 **  SSI1
 **
 ***************************************************************************/
__IO_REG32(    SSI1_STX0,                 0x63FCC000,__READ_WRITE );
__IO_REG32(    SSI1_STX1,                 0x63FCC004,__READ_WRITE );
__IO_REG32(    SSI1_STR0,                 0x63FCC008,__READ       );
__IO_REG32(    SSI1_STR1,                 0x63FCC00C,__READ       );
__IO_REG32_BIT(SSI1_SCR,                  0x63FCC010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI1_SISR,                 0x63FCC014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI1_SIER,                 0x63FCC018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI1_STCR,                 0x63FCC01C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI1_SRCR,                 0x63FCC020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI1_STCCR,                0x63FCC024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI1_SRCCR,                0x63FCC028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI1_SFCSR,                0x63FCC02C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI1_SACNT,                0x63FCC038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI1_SACADD,               0x63FCC03C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI1_SACDAT,               0x63FCC040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI1_SATAG,                0x63FCC044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI1_STMSK,                0x63FCC048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI1_SRMSK,                0x63FCC04C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI1_SACCST,               0x63FCC050,__READ_WRITE ,__saccst_bits);
__IO_REG32_BIT(SSI1_SACCEN,               0x63FCC054,__READ_WRITE ,__saccen_bits);
__IO_REG32_BIT(SSI1_SACCDIS,              0x63FCC058,__READ_WRITE ,__saccdis_bits);

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    SSI2_STX0,                 0x50014000,__READ_WRITE );
__IO_REG32(    SSI2_STX1,                 0x50014004,__READ_WRITE );
__IO_REG32(    SSI2_STR0,                 0x50014008,__READ       );
__IO_REG32(    SSI2_STR1,                 0x5001400C,__READ       );
__IO_REG32_BIT(SSI2_SCR,                  0x50014010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI2_SISR,                 0x50014014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI2_SIER,                 0x50014018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI2_STCR,                 0x5001401C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI2_SRCR,                 0x50014020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI2_STCCR,                0x50014024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI2_SRCCR,                0x50014028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI2_SFCSR,                0x5001402C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI2_SACNT,                0x50014038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI2_SACADD,               0x5001403C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI2_SACDAT,               0x50014040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI2_SATAG,                0x50014044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI2_STMSK,                0x50014048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI2_SRMSK,                0x5001404C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI2_SACCST,               0x50014050,__READ_WRITE ,__saccst_bits);
__IO_REG32_BIT(SSI2_SACCEN,               0x50014054,__READ_WRITE ,__saccen_bits);
__IO_REG32_BIT(SSI2_SACCDIS,              0x50014058,__READ_WRITE ,__saccdis_bits);

/***************************************************************************
 **
 **  SSI3
 **
 ***************************************************************************/
__IO_REG32(    SSI3_STX0,                 0x63FE8000,__READ_WRITE );
__IO_REG32(    SSI3_STX1,                 0x63FE8004,__READ_WRITE );
__IO_REG32(    SSI3_STR0,                 0x63FE8008,__READ       );
__IO_REG32(    SSI3_STR1,                 0x63FE800C,__READ       );
__IO_REG32_BIT(SSI3_SCR,                  0x63FE8010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI3_SISR,                 0x63FE8014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI3_SIER,                 0x63FE8018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI3_STCR,                 0x63FE801C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI3_SRCR,                 0x63FE8020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI3_STCCR,                0x63FE8024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI3_SRCCR,                0x63FE8028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI3_SFCSR,                0x63FE802C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI3_SACNT,                0x63FE8038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI3_SACADD,               0x63FE803C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI3_SACDAT,               0x63FE8040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI3_SATAG,                0x63FE8044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI3_STMSK,                0x63FE8048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI3_SRMSK,                0x63FE804C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI3_SACCST,               0x63FE8050,__READ_WRITE ,__saccst_bits);
__IO_REG32_BIT(SSI3_SACCEN,               0x63FE8054,__READ_WRITE ,__saccen_bits);
__IO_REG32_BIT(SSI3_SACCDIS,              0x63FE8058,__READ_WRITE ,__saccdis_bits);

/***************************************************************************
 **
 **  TVE
 **
 ***************************************************************************/
__IO_REG32_BIT(TVE_COM_CONF_REG,          0x63FF0000,__READ_WRITE ,__tve_com_conf_reg_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_0,  0x63FF0004,__READ_WRITE ,__tve_luma_filt_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_1,  0x63FF0008,__READ_WRITE ,__tve_luma_filt_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_2,  0x63FF000C,__READ_WRITE ,__tve_luma_filt_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_3,  0x63FF0010,__READ_WRITE ,__tve_luma_filt_cont_reg_3_bits);
__IO_REG32_BIT(TVE_LUMA_SA_CONT_REG_0,    0x63FF0014,__READ_WRITE ,__tve_luma_sa_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_SA_CONT_REG_1,    0x63FF0018,__READ_WRITE ,__tve_luma_sa_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_SA_STAT_REG_0,    0x63FF001C,__READ_WRITE ,__tve_luma_sa_stat_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_SA_STAT_REG_1,    0x63FF0020,__READ_WRITE ,__tve_luma_sa_stat_reg_1_bits);
__IO_REG32_BIT(TVE_CHROMA_CONT_REG,       0x63FF0024,__READ_WRITE ,__tve_chroma_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_0_CONT_REG,      0x63FF0028,__READ_WRITE ,__tve_tvdac_0_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_1_CONT_REG,      0x63FF002C,__READ_WRITE ,__tve_tvdac_1_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_2_CONT_REG,      0x63FF0030,__READ_WRITE ,__tve_tvdac_2_cont_reg_bits);
__IO_REG32_BIT(TVE_CD_CONT_REG,           0x63FF0034,__READ_WRITE ,__tve_cd_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_CONT_REG,     0x63FF0038,__READ_WRITE ,__tve_vbi_data_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_REG_0,        0x63FF003C,__READ_WRITE ,__tve_vbi_data_reg_0_bits);
__IO_REG32_BIT(TVE_VBI_DATA_REG_1,        0x63FF0040,__READ_WRITE ,__tve_vbi_data_reg_1_bits);
__IO_REG32(    TVE_VBI_DATA_REG_2,        0x63FF0044,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_3,        0x63FF0048,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_4,        0x63FF004C,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_5,        0x63FF0050,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_6,        0x63FF0054,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_7,        0x63FF0058,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_8,        0x63FF005C,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_9,        0x63FF0060,__READ_WRITE );
__IO_REG32_BIT(TVE_INT_CONT_REG,          0x63FF0064,__READ_WRITE ,__tve_int_cont_reg_bits);
__IO_REG32_BIT(TVE_STAT_REG,              0x63FF0068,__READ_WRITE ,__tve_stat_reg_bits);
__IO_REG32_BIT(TVE_TST_MODE_REG,          0x63FF006C,__READ_WRITE ,__tve_tst_mode_reg_bits);
__IO_REG32_BIT(TVE_USER_MODE_CONT_REG,    0x63FF0070,__READ_WRITE ,__tve_user_mode_cont_reg_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_0,0x63FF0074,__READ_WRITE,__tve_sd_timing_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_1,0x63FF0078,__READ_WRITE,__tve_sd_timing_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_2,0x63FF007C,__READ_WRITE,__tve_sd_timing_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_0,0x63FF0080,__READ_WRITE,__tve_hd_timing_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_1,0x63FF0084,__READ_WRITE,__tve_hd_timing_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_2,0x63FF0088,__READ_WRITE,__tve_hd_timing_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_0,   0x63FF008C,__READ_WRITE ,__tve_luma_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_1,   0x63FF0090,__READ_WRITE ,__tve_luma_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_2,   0x63FF0094,__READ_WRITE ,__tve_luma_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_3,   0x63FF0098,__READ_WRITE ,__tve_luma_usr_cont_reg_3_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_0,    0x63FF009C,__READ_WRITE ,__tve_csc_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_1,    0x63FF00A0,__READ_WRITE ,__tve_csc_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_2,    0x63FF00A4,__READ_WRITE ,__tve_csc_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_BLANK_USR_CONT_REG,    0x63FF00A8,__READ_WRITE ,__tve_blank_usr_cont_reg_bits);
__IO_REG32_BIT(TVE_SD_MOD_USR_CONT_REG,   0x63FF00AC,__READ_WRITE ,__tve_sd_mod_usr_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_0,0x63FF00B0,__READ_WRITE,__tve_vbi_data_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_1,0x63FF00B4,__READ_WRITE,__tve_vbi_data_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_2,0x63FF00B8,__READ_WRITE,__tve_vbi_data_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_3,0x63FF00BC,__READ_WRITE,__tve_vbi_data_usr_cont_reg_3_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_4,0x63FF00C0,__READ_WRITE,__tve_vbi_data_usr_cont_reg_4_bits);
__IO_REG32_BIT(TVE_DROP_COMP_USR_CONT_REG,0x63FF00C4,__READ_WRITE ,__tve_drop_comp_usr_cont_reg_bits);

/***************************************************************************
 **
 **  TZIC
 **
 ***************************************************************************/
__IO_REG32_BIT(TZIC_INTCTRL,              0x0FFFC000,__READ_WRITE ,__tzic_intctrl_bits);
__IO_REG32_BIT(TZIC_INTTYPE,              0x0FFFC004,__READ       ,__tzic_inttype_bits);
__IO_REG32_BIT(TZIC_PRIOMASK,             0x0FFFC00C,__READ_WRITE ,__tzic_priomask_bits);
__IO_REG32_BIT(TZIC_SYNCCTRL,             0x0FFFC010,__READ_WRITE ,__tzic_syncctrl_bits);
__IO_REG32_BIT(TZIC_DSMINT,               0x0FFFC014,__READ_WRITE ,__tzic_dsmint_bits);
__IO_REG32_BIT(TZIC_INTSEC0,              0x0FFFC080,__READ_WRITE ,__tzic_intsec0_bits);
__IO_REG32_BIT(TZIC_INTSEC1,              0x0FFFC084,__READ_WRITE ,__tzic_intsec1_bits);
__IO_REG32_BIT(TZIC_INTSEC2,              0x0FFFC088,__READ_WRITE ,__tzic_intsec2_bits);
__IO_REG32_BIT(TZIC_INTSEC3,              0x0FFFC08C,__READ_WRITE ,__tzic_intsec3_bits);
__IO_REG32_BIT(TZIC_ENSET0,               0x0FFFC100,__READ_WRITE ,__tzic_enset0_bits);
__IO_REG32_BIT(TZIC_ENSET1,               0x0FFFC104,__READ_WRITE ,__tzic_enset1_bits);
__IO_REG32_BIT(TZIC_ENSET2,               0x0FFFC108,__READ_WRITE ,__tzic_enset2_bits);
__IO_REG32_BIT(TZIC_ENSET3,               0x0FFFC10C,__READ_WRITE ,__tzic_enset3_bits);
__IO_REG32_BIT(TZIC_ENCLEAR0,             0x0FFFC180,__READ_WRITE ,__tzic_enclear0_bits);
__IO_REG32_BIT(TZIC_ENCLEAR1,             0x0FFFC184,__READ_WRITE ,__tzic_enclear1_bits);
__IO_REG32_BIT(TZIC_ENCLEAR2,             0x0FFFC188,__READ_WRITE ,__tzic_enclear2_bits);
__IO_REG32_BIT(TZIC_ENCLEAR3,             0x0FFFC18C,__READ_WRITE ,__tzic_enclear3_bits);
__IO_REG32_BIT(TZIC_SRCSET0,              0x0FFFC200,__READ_WRITE ,__tzic_srcset0_bits);
__IO_REG32_BIT(TZIC_SRCSET1,              0x0FFFC204,__READ_WRITE ,__tzic_srcset1_bits);
__IO_REG32_BIT(TZIC_SRCSET2,              0x0FFFC208,__READ_WRITE ,__tzic_srcset2_bits);
__IO_REG32_BIT(TZIC_SRCSET3,              0x0FFFC20C,__READ_WRITE ,__tzic_srcset3_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR0,            0x0FFFC280,__READ_WRITE ,__tzic_srcclear0_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR1,            0x0FFFC284,__READ_WRITE ,__tzic_srcclear1_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR2,            0x0FFFC288,__READ_WRITE ,__tzic_srcclear2_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR3,            0x0FFFC28C,__READ_WRITE ,__tzic_srcclear3_bits);
__IO_REG32_BIT(TZIC_PRIORITY0,            0x0FFFC400,__READ_WRITE ,__tzic_priority0_bits);
__IO_REG32_BIT(TZIC_PRIORITY1,            0x0FFFC404,__READ_WRITE ,__tzic_priority1_bits);
__IO_REG32_BIT(TZIC_PRIORITY2,            0x0FFFC408,__READ_WRITE ,__tzic_priority2_bits);
__IO_REG32_BIT(TZIC_PRIORITY3,            0x0FFFC40C,__READ_WRITE ,__tzic_priority3_bits);
__IO_REG32_BIT(TZIC_PRIORITY4,            0x0FFFC410,__READ_WRITE ,__tzic_priority4_bits);
__IO_REG32_BIT(TZIC_PRIORITY5,            0x0FFFC414,__READ_WRITE ,__tzic_priority5_bits);
__IO_REG32_BIT(TZIC_PRIORITY6,            0x0FFFC418,__READ_WRITE ,__tzic_priority6_bits);
__IO_REG32_BIT(TZIC_PRIORITY7,            0x0FFFC41C,__READ_WRITE ,__tzic_priority7_bits);
__IO_REG32_BIT(TZIC_PRIORITY8,            0x0FFFC420,__READ_WRITE ,__tzic_priority8_bits);
__IO_REG32_BIT(TZIC_PRIORITY9,            0x0FFFC424,__READ_WRITE ,__tzic_priority9_bits);
__IO_REG32_BIT(TZIC_PRIORITY10,           0x0FFFC428,__READ_WRITE ,__tzic_priority10_bits);
__IO_REG32_BIT(TZIC_PRIORITY11,           0x0FFFC42C,__READ_WRITE ,__tzic_priority11_bits);
__IO_REG32_BIT(TZIC_PRIORITY12,           0x0FFFC430,__READ_WRITE ,__tzic_priority12_bits);
__IO_REG32_BIT(TZIC_PRIORITY13,           0x0FFFC434,__READ_WRITE ,__tzic_priority13_bits);
__IO_REG32_BIT(TZIC_PRIORITY14,           0x0FFFC438,__READ_WRITE ,__tzic_priority14_bits);
__IO_REG32_BIT(TZIC_PRIORITY15,           0x0FFFC43C,__READ_WRITE ,__tzic_priority15_bits);
__IO_REG32_BIT(TZIC_PRIORITY16,           0x0FFFC440,__READ_WRITE ,__tzic_priority16_bits);
__IO_REG32_BIT(TZIC_PRIORITY17,           0x0FFFC444,__READ_WRITE ,__tzic_priority17_bits);
__IO_REG32_BIT(TZIC_PRIORITY18,           0x0FFFC448,__READ_WRITE ,__tzic_priority18_bits);
__IO_REG32_BIT(TZIC_PRIORITY19,           0x0FFFC44C,__READ_WRITE ,__tzic_priority19_bits);
__IO_REG32_BIT(TZIC_PRIORITY20,           0x0FFFC450,__READ_WRITE ,__tzic_priority20_bits);
__IO_REG32_BIT(TZIC_PRIORITY21,           0x0FFFC454,__READ_WRITE ,__tzic_priority21_bits);
__IO_REG32_BIT(TZIC_PRIORITY22,           0x0FFFC458,__READ_WRITE ,__tzic_priority22_bits);
__IO_REG32_BIT(TZIC_PRIORITY23,           0x0FFFC45C,__READ_WRITE ,__tzic_priority23_bits);
__IO_REG32_BIT(TZIC_PRIORITY24,           0x0FFFC460,__READ_WRITE ,__tzic_priority24_bits);
__IO_REG32_BIT(TZIC_PRIORITY25,           0x0FFFC464,__READ_WRITE ,__tzic_priority25_bits);
__IO_REG32_BIT(TZIC_PRIORITY26,           0x0FFFC468,__READ_WRITE ,__tzic_priority26_bits);
__IO_REG32_BIT(TZIC_PRIORITY27,           0x0FFFC46C,__READ_WRITE ,__tzic_priority27_bits);
__IO_REG32_BIT(TZIC_PRIORITY28,           0x0FFFC470,__READ_WRITE ,__tzic_priority28_bits);
__IO_REG32_BIT(TZIC_PRIORITY29,           0x0FFFC474,__READ_WRITE ,__tzic_priority29_bits);
__IO_REG32_BIT(TZIC_PRIORITY30,           0x0FFFC478,__READ_WRITE ,__tzic_priority30_bits);
__IO_REG32_BIT(TZIC_PRIORITY31,           0x0FFFC47C,__READ_WRITE ,__tzic_priority31_bits);
__IO_REG32_BIT(TZIC_PND0,                 0x0FFFCD00,__READ       ,__tzic_pnd0_bits);
__IO_REG32_BIT(TZIC_PND1,                 0x0FFFCD04,__READ       ,__tzic_pnd1_bits);
__IO_REG32_BIT(TZIC_PND2,                 0x0FFFCD08,__READ       ,__tzic_pnd2_bits);
__IO_REG32_BIT(TZIC_PND3,                 0x0FFFCD0C,__READ       ,__tzic_pnd3_bits);
__IO_REG32_BIT(TZIC_HIPND0,               0x0FFFCD80,__READ       ,__tzic_hipnd0_bits);
__IO_REG32_BIT(TZIC_HIPND1,               0x0FFFCD84,__READ       ,__tzic_hipnd1_bits);
__IO_REG32_BIT(TZIC_HIPND2,               0x0FFFCD88,__READ       ,__tzic_hipnd2_bits);
__IO_REG32_BIT(TZIC_HIPND3,               0x0FFFCD8C,__READ       ,__tzic_hipnd3_bits);
__IO_REG32_BIT(TZIC_WAKEUP0,              0x0FFFCE00,__READ       ,__tzic_wakeup0_bits);
__IO_REG32_BIT(TZIC_WAKEUP1,              0x0FFFCE04,__READ       ,__tzic_wakeup1_bits);
__IO_REG32_BIT(TZIC_WAKEUP2,              0x0FFFCE08,__READ       ,__tzic_wakeup2_bits);
__IO_REG32_BIT(TZIC_WAKEUP3,              0x0FFFCE0C,__READ       ,__tzic_wakeup3_bits);
__IO_REG32_BIT(TZIC_SWINT,                0x0FFFCF00,__WRITE      ,__tzic_swint_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1_URXD,                    0x53FBC000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART1_UTXD,                    0x53FBC040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART1_UCR1,                    0x53FBC080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART1_UCR2,                    0x53FBC084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART1_UCR3,                    0x53FBC088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART1_UCR4,                    0x53FBC08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART1_UFCR,                    0x53FBC090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART1_USR1,                    0x53FBC094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART1_USR2,                    0x53FBC098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART1_UESC,                    0x53FBC09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART1_UTIM,                    0x53FBC0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART1_UBIR,                    0x53FBC0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART1_UBMR,                    0x53FBC0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART1_UBRC,                    0x53FBC0AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART1_ONEMS,                   0x53FBC0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART1_UTS,                     0x53FBC0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2_URXD,                    0x53FC0000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART2_UTXD,                    0x53FC0040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART2_UCR1,                    0x53FC0080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART2_UCR2,                    0x53FC0084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART2_UCR3,                    0x53FC0088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART2_UCR4,                    0x53FC008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART2_UFCR,                    0x53FC0090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART2_USR1,                    0x53FC0094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART2_USR2,                    0x53FC0098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART2_UESC,                    0x53FC009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART2_UTIM,                    0x53FC00A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART2_UBIR,                    0x53FC00A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART2_UBMR,                    0x53FC00A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART2_UBRC,                    0x53FC00AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART2_ONEMS,                   0x53FC00B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART2_UTS,                     0x53FC00B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(UART3_URXD,                    0x5000C000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART3_UTXD,                    0x5000C040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART3_UCR1,                    0x5000C080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART3_UCR2,                    0x5000C084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART3_UCR3,                    0x5000C088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART3_UCR4,                    0x5000C08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART3_UFCR,                    0x5000C090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART3_USR1,                    0x5000C094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART3_USR2,                    0x5000C098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART3_UESC,                    0x5000C09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART3_UTIM,                    0x5000C0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART3_UBIR,                    0x5000C0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART3_UBMR,                    0x5000C0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART3_UBRC,                    0x5000C0AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART3_ONEMS,                   0x5000C0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART3_UTS,                     0x5000C0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(UART4_URXD,                    0x53FF0000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART4_UTXD,                    0x53FF0040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART4_UCR1,                    0x53FF0080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART4_UCR2,                    0x53FF0084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART4_UCR3,                    0x53FF0088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART4_UCR4,                    0x53FF008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART4_UFCR,                    0x53FF0090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART4_USR1,                    0x53FF0094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART4_USR2,                    0x53FF0098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART4_UESC,                    0x53FF009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART4_UTIM,                    0x53FF00A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART4_UBIR,                    0x53FF00A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART4_UBMR,                    0x53FF00A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART4_UBRC,                    0x53FF00AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART4_ONEMS,                   0x53FF00B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART4_UTS,                     0x53FF00B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(UART5_URXD,                    0x63F9000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART5_UTXD,                    0x63F9040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART5_UCR1,                    0x63F9080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART5_UCR2,                    0x63F9084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART5_UCR3,                    0x63F9088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART5_UCR4,                    0x63F908C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART5_UFCR,                    0x63F9090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART5_USR1,                    0x63F9094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART5_USR2,                    0x63F9098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART5_UESC,                    0x63F909C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART5_UTIM,                    0x63F90A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART5_UBIR,                    0x63F90A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART5_UBMR,                    0x63F90A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART5_UBRC,                    0x63F90AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART5_ONEMS,                   0x63F90B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART5_UTS,                     0x63F90B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  WDT1
 **
 ***************************************************************************/
__IO_REG16_BIT(WDT1_WCR,                  0x53F98000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WDT1_WSR,                  0x53F98002,__READ_WRITE );
__IO_REG16_BIT(WDT1_WRSR,                 0x53F98004,__READ       ,__wrsr_bits);
__IO_REG16_BIT(WDT1_WICR,                 0x53F98006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WDT1_WMCR,                 0x53F98008,__READ_WRITE ,__wmcr_bits);

/***************************************************************************
 **
 **  WDT2
 **
 ***************************************************************************/
__IO_REG16_BIT(WDT2_WCR,                  0x53F9C000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WDT2_WSR,                  0x53F9C002,__READ_WRITE );
__IO_REG16_BIT(WDT2_WRSR,                 0x53F9C004,__READ       ,__wrsr_bits);
__IO_REG16_BIT(WDT2_WICR,                 0x53F9C006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WDT2_WMCR,                 0x53F9C008,__READ_WRITE ,__wmcr_bits);

/***************************************************************************
 **
 **  VPU
 **
 ***************************************************************************/
__IO_REG32_BIT(VPU_CodeRun,               0x63FF4000,__WRITE      ,__vpu_coderun_bits);
__IO_REG32_BIT(VPU_CodeDown,              0x63FF4004,__WRITE      ,__vpu_codedown_bits);
__IO_REG32_BIT(VPU_HostIntReq,            0x63FF4008,__WRITE      ,__vpu_hostintreq_bits);
__IO_REG32_BIT(VPU_BitIntClear,           0x63FF400C,__WRITE      ,__vpu_bitintclear_bits);
__IO_REG32_BIT(VPU_BitIntSts,             0x63FF4010,__READ       ,__vpu_bitintsts_bits);
__IO_REG32_BIT(VPU_BitCodeReset,          0x63FF4014,__WRITE      ,__vpu_bitcodereset_bits);
__IO_REG32_BIT(VPU_BitCurPc,              0x63FF4018,__READ       ,__vpu_bitcurpc_bits);
__IO_REG32_BIT(VPU_BitCodecBusy,          0x63FF4020,__READ_WRITE ,__vpu_bitcodecbusy_bits);

/***************************************************************************
 **
 **  USB OTG
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x53F80000,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x53F80004,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x53F80008,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x53F8000C,__READ       ,__uog_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x53F80010,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x53F80014,__READ       ,__uog_hwrxbuf_bits);
__IO_REG8(     UOG_CAPLENGTH,             0x53F80100,__READ       );
__IO_REG16(    UOG_HCIVERSION,            0x53F80102,__READ       );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x53F80104,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x53F80108,__READ       ,__uog_hccparams_bits);
__IO_REG16(    UOG_DCIVERSION,            0x53F80120,__READ       );
__IO_REG32_BIT(UOG_DCCPARAMS,             0x53F80124,__READ       ,__uog_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x53F80140,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x53F80144,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x53F80148,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x53F8014C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x53F80154,__READ_WRITE ,__uog_periodiclistbase_bits);
#define UOG_DEVICEADDR      UOG_PERIODICLISTBASE
#define UOG_DEVICEADDR_bit  UOG_PERIODICLISTBASE_bit
__IO_REG32_BIT(UOG_ASYNCLISTADDR,         0x53F80158,__READ_WRITE ,__uog_asynclistaddr_bits);
#define UOG_ENDPOINTLISTADDR      UOG_ASYNCLISTADDR
#define UOG_ENDPOINTLISTADDR_bit  UOG_ASYNCLISTADDR_bit
__IO_REG32_BIT(UOG_BURSTSIZE,             0x53F80160,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x53F80164,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UOG_IC_USB,                0x53F8016C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UOG_VIEWPORT,              0x53F80170,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UOG_ENDPTNAK,              0x53F80178,__READ_WRITE ,__uog_endptnak_bits);
__IO_REG32_BIT(UOG_ENDPTNAKEN,            0x53F8017C,__READ_WRITE ,__uog_endptnaken_bits);
__IO_REG32(    UOG_CFGFLAG,               0x53F80180,__READ       );
__IO_REG32_BIT(UOG_PORTSC1,               0x53F80184,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x53F801A4,__READ_WRITE ,__uog_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x53F801A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x53F801AC,__READ_WRITE ,__uog_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x53F801B0,__READ_WRITE ,__uog_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x53F801B4,__READ_WRITE ,__uog_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x53F801B8,__READ       ,__uog_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x53F801BC,__READ_WRITE ,__uog_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x53F801C0,__READ_WRITE ,__uog_endptctrl0_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x53F801C4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x53F801C8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x53F801CC,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x53F801D0,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x53F801D4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x53F801D8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x53F801DC,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UH1_ID,                    0x53F80200,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH1_HWGENERAL,             0x53F80204,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH1_HWHOST,                0x53F80208,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH1_HWTXBUF,               0x53F80210,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH1_HWRXBUF,               0x53F80214,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH1_GPTIMER0LD,            0x53F80280,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH1_GPTIMER0CTRL,          0x53F80284,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH1_GPTIMER1LD,            0x53F80288,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH1_GPTIMER1CTRL,          0x53F8028C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH1_SBUSCFG,               0x53F80290,__READ_WRITE ,__uh_sbuscfg_bits);
__IO_REG8(     UH1_CAPLENGTH,             0x53F80300,__READ       );
__IO_REG16(    UH1_HCIVERSION,            0x53F80302,__READ       );
__IO_REG32_BIT(UH1_HCSPARAMS,             0x53F80304,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH1_HCCPARAMS,             0x53F80308,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH1_USBCMD,                0x53F80340,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH1_USBSTS,                0x53F80344,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH1_USBINTR,               0x53F80348,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH1_FRINDEX,               0x53F8034C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH1_PERIODICLISTBASE,      0x53F80354,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH1_ASYNCLISTADDR,         0x53F80358,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH1_BURSTSIZE,             0x53F80360,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH1_TXFILLTUNING,          0x53F80364,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH1_IC_USB,                0x53F8036C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH1_ULPIVIEW,              0x53F80370,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH1_PORTSC1,               0x53F80384,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH1_USBMODE,               0x53F803A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UH2_ID,                    0x53F80400,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH2_HWGENERAL,             0x53F80404,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH2_HWHOST,                0x53F80408,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH2_HWTXBUF,               0x53F80410,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH2_HWRXBUF,               0x53F80414,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH2_GPTIMER0LD,            0x53F80480,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH2_GPTIMER0CTRL,          0x53F80484,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH2_GPTIMER1LD,            0x53F80488,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH2_GPTIMER1CTRL,          0x53F8048C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH2_SBUSCFG,               0x53F80490,__READ_WRITE ,__uh_sbuscfg_bits);
__IO_REG8(     UH2_CAPLENGTH,             0x53F80500,__READ       );
__IO_REG16(    UH2_HCIVERSION,            0x53F80502,__READ       );
__IO_REG32_BIT(UH2_HCSPARAMS,             0x53F80504,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH2_HCCPARAMS,             0x53F80508,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH2_USBCMD,                0x53F80540,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH2_USBSTS,                0x53F80544,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH2_USBINTR,               0x53F80548,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH2_FRINDEX,               0x53F8054C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH2_PERIODICLISTBASE,      0x53F80554,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH2_ASYNCLISTADDR,         0x53F80558,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH2_BURSTSIZE,             0x53F80560,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH2_TXFILLTUNING,          0x53F80564,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH2_IC_USB,                0x53F8056C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH2_ULPIVIEW,              0x53F80570,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH2_PORTSC1,               0x53F80584,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH2_USBMODE,               0x53F805A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UH3_ID,                    0x53F80600,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH3_HWGENERAL,             0x53F80604,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH3_HWHOST,                0x53F80608,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH3_HWTXBUF,               0x53F80610,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH3_HWRXBUF,               0x53F80614,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH3_GPTIMER0LD,            0x53F80680,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH3_GPTIMER0CTRL,          0x53F80684,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH3_GPTIMER1LD,            0x53F80688,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH3_GPTIMER1CTRL,          0x53F8068C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH3_SBUSCFG,               0x53F80690,__READ_WRITE ,__uh_sbuscfg_bits);
__IO_REG8(     UH3_CAPLENGTH,             0x53F80700,__READ       );
__IO_REG16(    UH3_HCIVERSION,            0x53F80702,__READ       );
__IO_REG32_BIT(UH3_HCSPARAMS,             0x53F80704,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH3_HCCPARAMS,             0x53F80708,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH3_USBCMD,                0x53F80740,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH3_USBSTS,                0x53F80744,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH3_USBINTR,               0x53F80748,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH3_FRINDEX,               0x53F8074C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH3_PERIODICLISTBASE,      0x53F80754,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH3_ASYNCLISTADDR,         0x53F80758,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH3_BURSTSIZE,             0x53F80760,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH3_TXFILLTUNING,          0x53F80764,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH3_IC_USB,                0x53F8076C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH3_ULPIVIEW,              0x53F80770,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH3_PORTSC1,               0x53F80784,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH3_USBMODE,               0x53F807A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(USB_CTRL_0,                0x53F80800,__READ_WRITE ,__usb_ctrl_0_bits);
__IO_REG32_BIT(USB_OTG_PHY_CTRL_0,        0x53F80808,__READ_WRITE ,__usb_otg_phy_ctrl_0_bits);
__IO_REG32_BIT(USB_OTG_PHY_CTRL_1,        0x53F8080C,__READ_WRITE ,__usb_otg_phy_ctrl_1_bits);
__IO_REG32_BIT(USB_CTRL_1,                0x53F80810,__READ_WRITE ,__usb_ctrl_1_bits);
__IO_REG32_BIT(USB_UH2_CTRL,              0x53F80814,__READ_WRITE ,__usb_uh2_ctrl_bits);
__IO_REG32_BIT(USB_UH3_CTRL,              0x53F80818,__READ_WRITE ,__usb_uh3_ctrl_bits);
__IO_REG32_BIT(USB_UH1_PHY_CTRL_0,        0x53F8081C,__READ_WRITE ,__usb_uh1_phy_ctrl_0_bits);
__IO_REG32_BIT(USB_UH1_PHY_CTRL_1,        0x53F80820,__READ_WRITE ,__usb_uh1_phy_ctrl_1_bits);
__IO_REG32_BIT(USB_CLKONOFF_CTRL,         0x53F80824,__READ_WRITE ,__usb_clkonoff_ctrl_bits);

/***************************************************************************
 **
 ** AIPSTZ1
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPSTZ1_MPR1,              0x53F00000,__READ_WRITE ,__aipstz_mpr1_bits);
__IO_REG32_BIT(AIPSTZ1_MPR2,              0x53F00004,__READ_WRITE ,__aipstz_mpr2_bits);
__IO_REG32_BIT(AIPSTZ1_OPACR1,            0x53F00040,__READ_WRITE ,__aipstz1_opacr1_bits);
__IO_REG32_BIT(AIPSTZ1_OPACR2,            0x53F00044,__READ_WRITE ,__aipstz1_opacr2_bits);
__IO_REG32_BIT(AIPSTZ1_OPACR3,            0x53F00048,__READ_WRITE ,__aipstz1_opacr3_bits);
__IO_REG32_BIT(AIPSTZ1_OPACR4,            0x53F0004C,__READ_WRITE ,__aipstz1_opacr4_bits);
__IO_REG32_BIT(AIPSTZ1_OPACR5,            0x53F00050,__READ_WRITE ,__aipstz1_opacr5_bits);

/***************************************************************************
 **
 ** AIPSTZ2
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPSTZ2_MPR1,              0x63F00000,__READ_WRITE ,__aipstz_mpr1_bits);
__IO_REG32_BIT(AIPSTZ2_MPR2,              0x63F00004,__READ_WRITE ,__aipstz_mpr2_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR1,            0x63F00040,__READ_WRITE ,__aipstz2_opacr1_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR2,            0x63F00044,__READ_WRITE ,__aipstz2_opacr2_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR3,            0x63F00048,__READ_WRITE ,__aipstz2_opacr3_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR4,            0x63F0004C,__READ_WRITE ,__aipstz2_opacr4_bits);

/***************************************************************************
 **
 ** ARMPC
 **
 ***************************************************************************/
__IO_REG32_BIT(ARMPC_PVID,                0x63FA0000,__READ       ,__armpc_pvid_bits);
__IO_REG32_BIT(ARMPC_GPC,                 0x63FA0004,__READ_WRITE ,__armpc_gpc_bits);
__IO_REG32_BIT(ARMPC_PIC,                 0x63FA0008,__READ_WRITE ,__armpc_pic_bits);
__IO_REG32_BIT(ARMPC_LPC,                 0x63FA000C,__READ_WRITE ,__armpc_lpc_bits);
__IO_REG32_BIT(ARMPC_NLPC,                0x63FA0010,__READ_WRITE ,__armpc_nlpc_bits);
__IO_REG32_BIT(ARMPC_ICGC,                0x63FA0014,__READ_WRITE ,__armpc_icgc_bits);
__IO_REG32_BIT(ARMPC_AMC,                 0x63FA0018,__READ_WRITE ,__armpc_amc_bits);
__IO_REG32_BIT(ARMPC_NMC,                 0x63FA0020,__READ_WRITE ,__armpc_nmc_bits);
__IO_REG32_BIT(ARMPC_NMS,                 0x63FA0024,__READ_WRITE ,__armpc_nms_bits);

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
 **   MCIMX535 interrupt sources
 **
 ***************************************************************************/
#define INT_ESDHCV2_1          1              /* Enhanced SDHC Interrupt Request */
#define INT_ESDHCV2_2          2              /* Enhanced SDHC Interrupt Request */
#define INT_ESDHCV2_3          3              /* Enhanced SDHC Interrupt Request */
#define INT_ESDHCV2_4          4              /* Enhanced SDHC Interrupt Request */
#define INT_DAP                5              /* MPEG-4 Encoder */
#define INT_SDMA               6              /* “AND” of all 48 interrupts from all the channels */
#define INT_IOMUXC             7              /* POWER FAIL interrupt - this is a power fail indicator interrupt from on board power management IC via GPIO_16 PAD on ALT2. */
#define INT_EXTMC_NFC          8              /* NFC interrupt */
#define INT_VPU                9              /* VPU Interrupt Request */
#define INT_IPU_ERR            10             /* IPU Error Interrupt */
#define INT_IPU_SYNC           11             /* IPU Sync Interrupt */
#define INT_GPU3D              12             /* GPU Interrupt Request */
#define INT_UART_4             13             /* UART-4 ORed interrupt */
#define INT_USB_HOST1          14             /* USB Host 1 */
#define INT_EXTMC_ATA          15             /* Hard Drive (ATA) Controller */
#define INT_USB_HOST2          16             /* USB Host 2 */
#define INT_USB_HOST3          17             /* USB Host 3 */
#define INT_USB_OTG            18             /* USB OTG */
#define INT_SAHARA_HOST0       19             /* SAHARA Interrupt for Host 0 */
#define INT_SAHARA_HOST1       20             /* SAHARA Interrupt for Host 1 */
#define INT_SCC_HP             21             /* Security Monitor High Priority Interrupt Request */
#define INT_SCC_TZ             22             /* Secure (TrustZone) Interrupt Request. */
#define INT_SCC_NS             23             /* Regular (Non-Secure) Interrupt Request */
#define INT_SRTC_NTZ           24             /* SRTC Consolidated Interrupt. Non TZ. */
#define INT_SRTC_TZ            25             /* SRTC Security Interrupt. TZ. */
#define INT_RTIC               26             /* RTIC (Trust Zone) Interrupt Request. */
#define INT_CSU_1              27             /* CSU Interrupt Request 1. */
#define INT_SATA               28             /* SATA interrupt request */
#define INT_SSI_1              29             /* SSI-1 Interrupt Request */
#define INT_SSI_2              30             /* SSI-2 Interrupt Request */
#define INT_UART_1             31             /* UART-1 ORed interrupt */
#define INT_UART_2             32             /* UART-2 ORed interrupt */
#define INT_UART_3             33             /* UART-3 ORed interrupt */
#define INT_IPTP_RTC           34             /* RTC (IEEE1588) interrupt request */
#define INT_IPTP_PTP           35             /* PTP (IEEE1588) interrupt request */
#define INT_ECSPI_1            36             /* ECSPI-1 interrupt request line to the core. */
#define INT_ECSPI_2            37             /* ECSPI-2 interrupt request line to the core. */
#define INT_CSPI               38             /* CSPI interrupt request line to the core. */
#define INT_GPT                39             /* “OR” of GPT Rollover interrupt line, Input Capture 1 & 2 lines, Output Compare 1,2 &3 Interrupt lines */
#define INT_EPIT_1             40             /* EPIT-1 output compare interrupt */
#define INT_EPIT_2             41             /* EPIT-2 output compare interrupt */
#define INT_GPIO1_7            42             /* Active HIGH Interrupt from INT7 from GPIO */
#define INT_GPIO1_6            43             /* Active HIGH Interrupt from INT6 from GPIO */
#define INT_GPIO1_5            44             /* Active HIGH Interrupt from INT5 from GPIO */
#define INT_GPIO1_4            45             /* Active HIGH Interrupt from INT4 from GPIO */
#define INT_GPIO1_3            46             /* Active HIGH Interrupt from INT3 from GPIO */
#define INT_GPIO1_2            47             /* Active HIGH Interrupt from INT2 from GPIO */
#define INT_GPIO1_1            48             /* Active HIGH Interrupt from INT1 from GPIO */
#define INT_GPIO1_0            49             /* Active HIGH Interrupt from INT0 from GPIO */
#define INT_GPIO1_0_15         50             /* Combined interrupt indication for GPIO-1 signal 0 throughout 15 */
#define INT_GPIO1_16_31        51             /* Combined interrupt indication for GPIO-1 signal 16 throughout 31 */
#define INT_GPIO2_0_15         52             /* Combined interrupt indication for GPIO-2 signal 0 throughout 15 */
#define INT_GPIO2_16_31        53             /* Combined interrupt indication for GPIO-2 signal 16 throughout 31 */
#define INT_GPIO3_0_15         54             /* Combined interrupt indication for GPIO-3 signal 0 throughout 15 */
#define INT_GPIO3_16_31        55             /* Combined interrupt indication for GPIO-3 signal 16 throughout 31 */
#define INT_GPIO4_0_15         56             /* Combined interrupt indication for GPIO-4 signal 0 throughout 15 */
#define INT_GPIO4_16_31        57             /* Combined interrupt indication for GPIO-4 signal 16 throughout 31 */
#define INT_WDOG_1             58             /* Watchdog Timer reset */
#define INT_WDOG_2             59             /* TrustZone Watchdog Timer reset */
#define INT_KPP                60             /* Keypad Interrupt */
#define INT_PWM_1              61             /* CumuUART-5lative interrupt line. */
#define INT_I2C_1              62             /* I2C-1 Interrupt */
#define INT_I2C_2              63             /* I2C-2 Interrupt */
#define INT_I2C_3              64             /* I2C-3 Interrupt */
#define INT_MLB                65             /* NOR of all interrupts, mlb_cint and mlb_sint */
#define INT_ASRC               66             /* ASRC Interrupt for core 1 */
#define INT_SPDIF              67             /* SPDIF Tx interrupt OR SPDIF Rx interrupt */
#define INT_IIM                69             /* Interrupt request to the processor */
#define INT_PATA               70             /* Parallel ATA host controller interrupt request */
#define INT_CCM_1              71             /* CCM, Interrupt Request 1 */
#define INT_CCM_2              72             /* CCM, Interrupt Request 2 */
#define INT_GPC_1              73             /* GPC, Interrupt Request 1 */
#define INT_GPC_2              74             /* GPC, Interrupt Request 2 */
#define INT_SRC                75             /* SRC interrupt request */
#define INT_P_PLATFORM_NE_32K_256K_M      76  /* Neon Monitor Interrupt */
#define INT_P_PLATFORM_NE_32K_256K_PU     77  /* Performance Unit Interrupt (nPMUIRQ) */
#define INT_P_PLATFORM_NE_32K_256K_CTI    78  /* CTI IRQ */
#define INT_P_PLATFORM_NE_32K_256K_DCT11  79  /* Debug Interrupt, from Cross-Trigger 1 Interface 1 */
#define INT_P_PLATFORM_NE_32K_256K_DCT10  80  /* Debug Interrupt, from Cross-Trigger 1 Interface 0 */
#define INT_ESAI               81             /* ESAI interrupt */
#define INT_FLEXCAN_1          82             /* NOR of all interrupts */
#define INT_FLEXCAN_2          83             /* NOR of all interrupts */
#define INT_OPENVG             84             /* General Interrupt */
#define INT_OPENVG_BUSY        85             /* Busy signal (for S/W power gating feasibility) */
#define INT_UART_5             86             /* UART-5 ORed interrupt */
#define INT_FEC                87             /* Fast Interrupt Request (OR of 13 interrupt sources) */
#define INT_OWIRE              88             /* 1-Wire Interrupt Request */
#define INT_P_PLATFORM_NE_32K_256K_DCT12  89  /* Debug Interrupt, from Cross-Trigger 1 Interface 2 */
#define INT_SJC                90             /* */
#define INT_TVE                92             /* */
#define INT_FIRI               93             /* FIRI Intr (OR of all 4 interrupt sources) */
#define INT_PWM_2              94             /* Cumulative interrupt line */
#define INT_SSI_3              96             /* SSI-3 Interrupt Request */
#define INT_P_PLATFORM_NE_32K_256K_DCT13  98  /* Debug Interrupt, from Cross-Trigger 1 Interface 3 */
#define INT_VPU_IDLE           100            /* Idle interrupt from VPU (for S/W power gating) */
#define INT_EXTMC_AUTO_PROG    101            /* Indicates all pages have been transferred to NFC during an auto_prog operation */
#define INT_GPU3D_IDLE         102            /* Idle interrupt from GPU (for S/W power gating) */
#define INT_GPIO5_0_15         103            /* Combined interrupt indication for GPIO-5 signal 0 throughout 15 */
#define INT_GPIO5_16_31        104            /* Combined interrupt indication for GPIO-5 signal 16 throughout 31 */
#define INT_GPIO6_0_15         105            /* Combined interrupt indication for GPIO-6 signal 0 throughout 15 */
#define INT_GPIO6_16_31        106            /* Combined interrupt indication for GPIO-6 signal 16 throughout 31 */
#define INT_GPIO7_0_15         107            /* Combined interrupt indication for GPIO-7 signal 0 throughout 15 */
#define INT_GPIO7_16_31        108            /* Combined interrupt indication for GPIO-7 signal 16 throughout 31 */


#endif    /* __MCIMX535_H */
