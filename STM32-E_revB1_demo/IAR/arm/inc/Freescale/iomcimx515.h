/***************************************************************************
 **
 **    This file defines the Special Function Register for
 **    Freescale MCIMX515
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 52705 $
 **
 ***************************************************************************/

#ifndef __MCIMX515_H
#define __MCIMX515_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX515 SPECIAL FUNCTION REGISTERS
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
__REG32 FPM_EN    : 1;
__REG32 CAMP1_EN  : 1;
__REG32 CAMP2_EN  : 1;
__REG32 FPM_MULT  : 1;
__REG32 COSC_EN   : 1;
__REG32           : 1;
__REG32           : 1;
__REG32           :17;
} __ccm_ccr_bits;

/* CCM Control Divider Register (CCM.CCDR) */
typedef struct {
__REG32                   :16;
__REG32 emi_hs_mask       : 1;
__REG32 ipu_hs_mask       : 1;
__REG32                   :14;
} __ccm_ccdr_bits;

/* CCM Status Register(CCM.CSR) */
typedef struct {
__REG32 ref_en_b        : 1;
__REG32 fpm_ready       : 1;
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
__REG32 lp_apm          : 1;
__REG32                 :22;
} __ccm_ccsr_bits;

/* CCM Arm Clock Root Register (CCM.CACRR) */
typedef struct {
__REG32 arm_podf        : 3;
__REG32                 :29;
} __ccm_cacrr_bits;

/* CCM Bus Clock Divider Register(CCM.CBCDR) */
typedef struct {
__REG32 perclk_podf           : 3;
__REG32 perclk_pred2          : 3;
__REG32 perclk_pred1          : 2;
__REG32 ipg_podf              : 2;
__REG32 ahb_podf              : 3;
__REG32 nfc_podf              : 3;
__REG32 axi_a_podf            : 3;
__REG32 axi_b_podf            : 3;
__REG32 emi_slow_podf         : 3;
__REG32 periph_clk_sel        : 1;
__REG32 emi_clk_sel           : 1;
__REG32 ddr_clk_podf          : 3;
__REG32 ddr_high_freq_clk_sel : 1;
__REG32                       : 1;
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
__REG32 tve_clk_sel         : 1;
__REG32 ssi_apm_clk_sel     : 2;
__REG32 vpu_rclk_sel        : 1;
__REG32 ssi3_clk_sel        : 1;
__REG32 ssi2_clk_sel        : 2;
__REG32 ssi1_clk_sel        : 2;
__REG32 esdhc2_clk_sel      : 2;
__REG32 esdhc4_clk_sel      : 1;
__REG32 esdhc3_clk_sel      : 1;
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
__REG32 spdif1_clk_sel      : 2;
__REG32 spdif0_com          : 1;
__REG32 spdif1_com          : 1;
__REG32                     : 4;
__REG32 sim_clk_sel         : 2;
__REG32 firi_clk_sel        : 2;
__REG32 hsi2c_clk_sel       : 2;
__REG32                     : 6;
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
__REG32 esdhc2_clk_podf : 3;
__REG32 esdhc2_clk_pred : 3;
__REG32                 : 7;
} __ccm_cscdr1_bits;

/* CCM SSI1 Clock Divider Register(CCM.CS1CDR) */
typedef struct {
__REG32 ssi1_clk_podf     : 6;
__REG32 ssi1_clk_pred     : 3;
__REG32                   : 7;
__REG32 ssi_ext1_clk_podf : 6;
__REG32 ssi_ext1_clk_pred : 3;
__REG32                   : 7;
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
__REG32 di_clk_pred       : 3;
__REG32 spdif1_clk_podf   : 6;
__REG32                   : 1;
__REG32 spdif1_clk_pred   : 3;
__REG32 spdif0_clk_podf   : 6;
__REG32 spdif0_clk_pred   : 3;
__REG32 tve_clk_pred      : 3;
__REG32                   : 1;
} __ccm_cdcdr_bits;

/* CCM Serial Clock Divider Register 2 (CCM.CSCDR2) */
typedef struct {
__REG32                   : 9;
__REG32 sim_clk_podf      : 6;
__REG32                   : 1;
__REG32 sim_clk_pred      : 3;
__REG32 ecspi_clk_podf    : 6;
__REG32 ecspi_clk_pred    : 3;
__REG32                   : 4;
} __ccm_cscdr2_bits;

/* CCM Serial Clock Divider Register 3(CCM.CSCDR3) */
typedef struct {
__REG32 firi_clk_podf       : 6;
__REG32 firi_clk_pred       : 3;
__REG32 hsi2c_clk_podf      : 6;
__REG32                     : 1;
__REG32 hsi2c_clk_pred      : 3;
__REG32                     :13;
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

/* CCM Wakeup Detector Register (CCM.CWDR) */
typedef struct {
__REG32 gpio1_4_icr         : 2;
__REG32 gpio1_5_icr         : 2;
__REG32 gpio1_6_icr         : 2;
__REG32 gpio1_7_icr         : 2;
__REG32 gpio1_8_icr         : 2;
__REG32 gpio1_9_icr         : 2;
__REG32                     : 4;
__REG32 gpio1_4_dir         : 1;
__REG32 gpio1_5_dir         : 1;
__REG32 gpio1_6_dir         : 1;
__REG32 gpio1_7_dir         : 1;
__REG32 gpio1_8_dir         : 1;
__REG32 gpio1_9_dir         : 1;
__REG32                     :10;
} __ccm_cwdr_bits;

/* CCM Divider Handshake In-Process Register(CCM.CDHIPR) */
typedef struct {
__REG32 axi_a_podf_busy            : 1;
__REG32 axi_b_podf_busy            : 1;
__REG32 emi_slow_podf_busy         : 1;
__REG32 ahb_podf_busy              : 1;
__REG32 nfc_podf_busy              : 1;
__REG32 periph_clk_sel_busy        : 1;
__REG32 emi_clk_sel_busy           : 1;
__REG32 ddr_podf_busy              : 1;
__REG32 ddr_high_freq_clk_sel_busy : 1;
__REG32                            : 7;
__REG32 arm_podf_busy              : 1;
__REG32                            :15;
} __ccm_cdhipr_bits;

/* CCM DVFS Control Register(CCM.CDCR) */
typedef struct {
__REG32 periph_clk_DVFS_podf          : 2;
__REG32 arm_freq_shift_divider        : 1;
__REG32                               : 1;
__REG32                               : 1;
__REG32 software_DVFS_en              : 1;
__REG32 sw_periph_clk_div_req         : 1;
__REG32 sw_periph_clk_div_req_status  : 1;
__REG32                               :24;
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
__REG32 apm_sdma_clk_gate_en_bit      : 1;
__REG32                               : 3;
__REG32 bypass_sahara_lpm_hs          : 1;
__REG32 bypass_rtic_lpm_hs            : 1;
__REG32 bypass_ipu_lpm_hs             : 1;
__REG32 bypass_emi_lpm_hs             : 1;
__REG32 bypass_sdma_lpm_hs            : 1;
__REG32 bypass_max_lpm_hs             : 1;
__REG32 bypass_scc_lpm_hs             : 1;
__REG32                               : 9;
} __ccm_clpcr_bits;

/* CCM Interrupt Status Register(CCM.CISR) */
typedef struct {
__REG32 lrf_pll1                     : 1;
__REG32 lrf_pll2                     : 1;
__REG32 lrf_pll3                     : 1;
__REG32 fpm_ready                    : 1;
__REG32 CAMP1_ready                  : 1;
__REG32 CAMP2_ready                  : 1;
__REG32 cosc_ready                   : 1;
__REG32 kpp_wakeup_det               : 1;
__REG32 sdhc1_wakeup_det             : 1;
__REG32 sdhc2_wakeup_det             : 1;
__REG32 gpio1_4_wakeup_det           : 1;
__REG32 gpio1_5_wakeup_det           : 1;
__REG32 gpio1_6_wakeup_det           : 1;
__REG32 gpio1_7_wakeup_det           : 1;
__REG32 gpio1_8_wakeup_det           : 1;
__REG32 gpio1_9_wakeup_det           : 1;
__REG32 dividers_loaded              : 1;
__REG32 axi_a_podf_loaded            : 1;
__REG32 axi_b_podf_loaded            : 1;
__REG32 emi_slow_podf_loaded         : 1;
__REG32 ahb_podf_loaded              : 1;
__REG32 nfc_podf_loaded              : 1;
__REG32 periph_clk_sel_loaded        : 1;
__REG32 emi_clk_sel_loaded           : 1;
__REG32 ddr_clk_podf_loaded          : 1;
__REG32 ddr_high_freq_clk_sel_loaded : 1;
__REG32 arm_podf_loaded              : 1;
__REG32                              : 5;
} __ccm_cisr_bits;

/* CCM Interrupt Mask Register(CCM.CIMR) */
typedef struct {
__REG32 mask_lrf_pll1                     : 1;
__REG32 mask_lrf_pll2                     : 1;
__REG32 mask_lrf_pll3                     : 1;
__REG32 mask_fpm_ready                    : 1;
__REG32 mask_CAPM1_ready                  : 1;
__REG32 mask_CAMP2_ready                  : 1;
__REG32 mask_cosc_ready                   : 1;
__REG32 mask_kpp_wakeup_det               : 1;
__REG32 mask_sdhc1_wakeup_det             : 1;
__REG32 mask_sdhc2_wakeup_det             : 1;
__REG32 mask_gpio1_4_wakeup_det           : 1;
__REG32 mask_gpio1_5_wakeup_det           : 1;
__REG32 mask_gpio1_6_wakeup_det           : 1;
__REG32 mask_gpio1_7_wakeup_det           : 1;
__REG32 mask_gpio1_8_wakeup_det           : 1;
__REG32 mask_gpio1_9_wakeup_det           : 1;
__REG32 mask_dividers_loaded              : 1;
__REG32 mask_axi_a_podf_loaded            : 1;
__REG32 mask_axi_b_podf_loaded            : 1;
__REG32 mask_emi_slow_podf_loaded         : 1;
__REG32 mask_ahb_podf_loaded              : 1;
__REG32 mask_nfc_podf_loaded              : 1;
__REG32 mask_periph_clk_sel_loaded        : 1;
__REG32 mask_emi_clk_sel_loaded           : 1;
__REG32 mask_ddr_podf_loaded              : 1;
__REG32 mask_ddr_high_freq_clk_sel_loaded : 1;
__REG32 mask_arm_podf_loaded              : 1;
__REG32                                   : 5;
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
__REG32                                   : 3;
__REG32 FPM_mux_select                    : 1;
__REG32 efuse_prog_supply_gate            : 1;
__REG32 overide_apm_emi_int1_clock_gating : 1;
__REG32                                   : 9;
__REG32 arm_async_ref_sel_5               : 1;
__REG32 arm_async_ref_sel_0               : 1;
__REG32 arm_async_ref_sel_1               : 1;
__REG32 arm_async_ref_sel_2               : 1;
__REG32 arm_async_ref_sel_3               : 1;
__REG32 arm_async_ref_sel_4               : 1;
__REG32 arm_async_ref_sel_6               : 1;
__REG32 arm_async_ref_sel_7               : 1;
__REG32 arm_async_ref_en                  : 1;
__REG32                                   : 8;
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
__REG32                     : 1;
__REG32 mod_en_ov_emi_garb  : 1;
__REG32 mod_en_ov_emi_fast  : 1;
__REG32 mod_en_ov_emi_slow  : 1;
__REG32 mod_en_ov_emi_int1  : 1;
__REG32 mod_en_ov_emi_m0    : 1;
__REG32 mod_en_ov_emi_m1    : 1;
__REG32 mod_en_ov_emi_m2    : 1;
__REG32 mod_en_ov_emi_m3    : 1;
__REG32 mod_en_ov_emi_m4    : 1;
__REG32 mod_en_ov_emi_m5    : 1;
__REG32 mod_en_ov_emi_m6    : 1;
__REG32 mod_en_ov_emi_m7    : 1;
__REG32                     : 8;
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
__REG32 SMSTATUS  : 4;
__REG32           : 2;
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
__REG32 REF_CLK_SEL : 2;
__REG32 REF_CLK_DIV : 1;
__REG32 ADE         : 1;
__REG32 DPDCK0_2_EN : 1;
__REG32 MUL_CTRL    : 1;
__REG32             :18;
} __dpllip_dp_ctl_bits;

/* DPLL Config Register(DPLLC.DP_CONFIG) */
typedef struct{
__REG32 LDREQ       : 1;
__REG32 AREN        : 1;
__REG32             :30;
} __dpllip_dp_config_bits;

/* DPLL Operation Register(DPLLC.DP_OP) */
typedef struct{
__REG32 PDF         : 4;
__REG32 MFI         : 4;
__REG32             :24;
} __dpllip_dp_op_bits;

/* DPLL Multiplication Factor Denominator Register(DPLLC.DP_MFD) */
typedef struct{
__REG32 MFD         :27;
__REG32             : 5;
} __dpllip_dp_mfd_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFN) */
typedef struct{
__REG32 MFN         :27;
__REG32             : 5;
} __dpllip_dp_mfn_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFNMINUS) */
typedef struct{
__REG32 MFNMINUS    :27;
__REG32             : 5;
} __dpllip_dp_mfnminus_bits;

/* DPLL MFNxxx Register (DPLLC.DP_MFNPLUS) */
typedef struct{
__REG32 MFNPLUS     :27;
__REG32             : 5;
} __dpllip_dp_mfnplus_bits;

/* DPLL High Frequency Support, Operation Register(DPLLC.DP_HFS_OP) */
typedef struct{
__REG32 HFS_PDF     : 4;
__REG32 HFS_MFI     : 4;
__REG32             :24;
} __dpllip_dp_hfs_op_bits;

/* DPLL HFS MFD Register (DPLLC.DP_HFS_MFD) */
typedef struct{
__REG32 HFS_MFD     :27;
__REG32             : 5;
} __dpllip_dp_hfs_mfd_bits;

/* DPLL HFS Multiplication Factor Numerator Register (DPLLC.DP_HFS_MFN) */
typedef struct{
__REG32 HFS_MFN     :27;
__REG32             : 5;
} __dpllip_dp_hfs_mfn_bits;

/* DPLL Multiplication Factor Numerator Toggle Control Register (DPLLC.DP_MFN_TOGC) */
typedef struct{
__REG32 TOG_MFN_CNT :16;
__REG32 TOG_EN      : 1;
__REG32 TOG_DIS     : 1;
__REG32             :14;
} __dpllip_dp_mfn_togc_bits;

/* Desense Status Register(DPLLC.DP_DESTAT) */
typedef struct{
__REG32 TOG_MFN     :27;
__REG32             : 4;
__REG32 TOG_SEL     : 1;
} __dpllip_dp_destat_bits;

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
__REG32             : 8;
__REG32 P2THR       : 6;
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
__REG32 HW            : 1;
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
__REG32 SSB_CTL       : 4;
__REG32 SSB_POL       : 4;
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
__REG32 TX_WATERMARK  : 6;
__REG32               : 1;
__REG32 TXDEN         : 1;
__REG32               : 8;
__REG32 RX_WATERMARK  : 6;
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
__REG32               :13;
__REG32 CL            : 2;
__REG32               : 1;
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
} __weim_csgcr1_bits;

/* Chip Select x General Configuration Register 2 (EIM.CS<i>GCR2) */
typedef struct {
__REG32 ADH               : 2;
__REG32                   : 2;
__REG32 DAPS              : 4;
__REG32 DAE               : 1;
__REG32 DAP               : 1;
__REG32                   :22;
} __weim_csgcr2_bits;

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
} __weim_csrcr1_bits;

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
} __weim_csrcr2_bits;

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
} __weim_cswcr1_bits;

/* Chip Select x Write Configuration Register 2 (EIM.CS<i>WCR2) */
typedef struct {
__REG32 WBCDD             : 1;
__REG32                   :31;
} __weim_cswcr2_bits;

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
} __weim_wcr_bits;

/* EIM IP Access Register (EIM.WIAR) */
typedef struct {
__REG32 IPS_REQ           : 1;
__REG32 IPS_ACK           : 1;
__REG32 INT               : 1;
__REG32 ERRST             : 1;
__REG32 ACLK_EN           : 1;
__REG32                   :27;
} __weim_wiar_bits;

/* Error Address Register (EIM.EAR) */
typedef struct {
__REG32 IPS_REQ           : 1;
__REG32 IPS_ACK           : 1;
__REG32 INT               : 1;
__REG32 ERRST             : 1;
__REG32 ACLK_EN           : 1;
__REG32                   :27;
} __weim_ear_bits;

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

/* ESDCTLx Control Registers (ESDCTL.ESDCTL0-ESDCTL1) */
typedef struct {
__REG32                   :12;
__REG32 PWDT              : 2;
__REG32 SRT               : 2;
__REG32 DSIZ              : 2;
__REG32                   : 2;
__REG32 COL               : 2;
__REG32                   : 1;
__REG32 DBL_tRFC          : 1;
__REG32 ROW               : 3;
__REG32                   : 1;
__REG32 SREFR             : 3;
__REG32 SDE               : 1;
} __esdctl_esdctlx_bits;

/* ESDCTLx Configuration Register (ESDCTL.ESDCFG0-ESDCFG1) */
typedef struct {
__REG32 tRC               : 4;
__REG32 tRCD              : 3;
__REG32 tWR               : 1;
__REG32                   : 2;
__REG32 tRRD              : 2;
__REG32 tRAS              : 4;
__REG32 tMRD              : 2;
__REG32 tRP               : 2;
__REG32 tWTR              : 1;
__REG32 tXP               : 3;
__REG32 tXSR              : 4;
__REG32 tRFC              : 4;
} __esdctl_esdcfgx_bits;

/* ESDCTL Miscellaneous Register (ESDCTL.ESDMISC) */
typedef struct {
__REG32                   : 1;
__REG32 RST               : 1;
__REG32                   : 1;
__REG32 DDR_EN            : 1;
__REG32 DDR2_EN           : 1;
__REG32 LHD               : 1;
__REG32 DDR2_8_BANK       : 1;
__REG32 RALAT             : 2;
__REG32 MIF3_MODE         : 2;
__REG32 FRC_MSR           : 1;
__REG32 BI_ON             : 1;
__REG32 ODT_EN            : 1;
__REG32 AUTO_DLL_PAUSE    : 1;
__REG32 DIFF_DQS_EN       : 1;
__REG32 AP_bit            : 4;
__REG32 TERM_CTL0         : 2;
__REG32 TERM_CTL1         : 2;
__REG32 TERM_CTL2         : 2;
__REG32 TERM_CTL3         : 2;
__REG32 SD_CLK_EXT        : 1;
__REG32 ODT_IDLE_ON       : 1;
__REG32 CS1_RDY           : 1;
__REG32 CS0_RDY           : 1;
} __esdctl_esdmisc_bits;

/* ESDCTL Special Command Register (ESDCTL.ESDSCR) */
typedef struct {
__REG32 BA                : 2;
__REG32 CS                : 1;
__REG32 CMD               : 3;
__REG32 MAN_DLL_PAUSE     : 1;
__REG32                   : 7;
__REG32 CON_ACK           : 1;
__REG32 CON_REQ           : 1;
__REG32 PSEUDO_ADDR       :15;
__REG32                   : 1;
} __esdctl_esdscr_bits;

/* ESDCTL Delay Line 1 Configuration Debug Register (ESDCTL.ESDCDLY1) */
typedef struct {
__REG32 DLY_REG_1         : 8;
__REG32 DLY_ABS_OFFSET_1  : 8;
__REG32 DLY_OFFSET_1      : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG_1     : 1;
} __esdctl_esdcdly1_bits;

/* ESDCTL Delay Line 2 Configuration Debug Register (ESDCTL.ESDCDLY2) */
typedef struct {
__REG32 DLY_REG_2         : 8;
__REG32 DLY_ABS_OFFSET_2  : 8;
__REG32 DLY_OFFSET_2      : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG_2     : 1;
} __esdctl_esdcdly2_bits;

/* ESDCTL Delay Line 3 Configuration Debug Register (ESDCTL.ESDCDLY3) */
typedef struct {
__REG32 DLY_REG_3         : 8;
__REG32 DLY_ABS_OFFSET_3  : 8;
__REG32 DLY_OFFSET_3      : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG_3     : 1;
} __esdctl_esdcdly3_bits;

/* ESDCTL Delay Line 4 Configuration Debug Register (ESDCTL.ESDCDLY4) */
typedef struct {
__REG32 DLY_REG_4         : 8;
__REG32 DLY_ABS_OFFSET_4  : 8;
__REG32 DLY_OFFSET_4      : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG_4     : 1;
} __esdctl_esdcdly4_bits;

/* ESDCTL Delay Line 5 Configuration Debug Register (ESDCTL.ESDCDLY5) */
typedef struct {
__REG32 DLY_REG_5         : 8;
__REG32 DLY_ABS_OFFSET_5  : 8;
__REG32 DLY_OFFSET_5      : 8;
__REG32                   : 7;
__REG32 SEL_DLY_REG_5     : 1;
} __esdctl_esdcdly5_bits;

/* ESDCTL General Purpose Register (ESDCTL.ESDGPR) */
typedef struct {
__REG32 QTR_CYCLE_LENGTH    : 8;
__REG32                     : 8;
__REG32 SCT                 : 2;
__REG32                     : 1;
__REG32 DIG_OFF3            : 2;
__REG32 DIG_OFF2            : 2;
__REG32 DIG_OFF1            : 2;
__REG32 DIG_OFF0            : 2;
__REG32 DIG_QTR             : 2;
__REG32 DIG_CYC             : 2;
__REG32 DIG_EN              : 1;
} __esdctl_esdgpr_bits;

/* ESDCTL ESDPRCTx Control Register (ESDCTL.ESDPRCT0-ESDPRCT1) */
typedef struct {
__REG32 PRCT0               : 3;
__REG32 PRCT1               : 3;
__REG32 PRCT2               : 3;
__REG32 PRCT3               : 3;
__REG32 PRCT4               : 3;
__REG32 PRCT5               : 3;
__REG32 PRCT6               : 3;
__REG32 PRCT7               : 3;
__REG32                     : 8;
} __esdctl_esdprctx_bits;

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
#define ESDHC_FEVT_FEVTAC12NE    0x00000001UL
#define ESDHC_FEVT_FEVTAC12TOE   0x00000002UL
#define ESDHC_FEVT_FEVTAC12CE    0x00000004UL
#define ESDHC_FEVT_FEVTAC12EBE   0x00000008UL
#define ESDHC_FEVT_FEVTAC12IE    0x00000010UL
#define ESDHC_FEVT_FEVTCNIBAC12E 0x00000080UL
#define ESDHC_FEVT_FEVTCTOE      0x00010000UL
#define ESDHC_FEVT_FEVTCCE       0x00020000UL
#define ESDHC_FEVT_FEVTCEBE      0x00040000UL
#define ESDHC_FEVT_FEVTCIE       0x00080000UL
#define ESDHC_FEVT_FEVTDTOE      0x00100000UL
#define ESDHC_FEVT_FEVTDCE       0x00200000UL
#define ESDHC_FEVT_FEVTDEBE      0x00400000UL
#define ESDHC_FEVT_FEVTAC12E     0x01000000UL
#define ESDHC_FEVT_FEVTDMAE      0x10000000UL
#define ESDHC_FEVT_FEVTCINT      0x80000000UL

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
__REG32                     : 1;
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

/* IP Lock register (EMI.IPLCK) */
typedef struct {
__REG32 NFC_lock            : 1;
__REG32 WEIM_lock           : 1;
__REG32 ESDC_lock           : 1;
__REG32 M4IF_lock           : 1;
__REG32 Lock_all            : 1;
__REG32 XFR_ERR_EN          : 1;
__REG32                     :26;
} __emi_iplck_bits;

/* Interrupt control & Status register (EMI.EICS) */
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
__REG32                       :18;
__REG32 EBOIE                 : 1;
__REG32 EAOIE                 : 1;
} __emi_eics_bits;

/* EMPG Control Register (EMPG.EMPGCR) */
typedef struct {
__REG32 PCR                   : 1;
__REG32                       :31;
} __empgc_empgcr_bits;

/* EMPG Power-up Sequence Control Register (EMPG.PUPSCR) */
typedef struct {
__REG32 PUP                   : 8;
__REG32                       :24;
} __empgc_pupscr_bits;

/* EMPG Power-down Sequence Control Register (EMPG.PDNSCR) */
typedef struct {
__REG32 PDN                   : 8;
__REG32                       :24;
} __empgc_pdnscr_bits;

/* EMPG Status Register (EMPG.PUPSCR) */
typedef struct {
__REG32 PSR                   : 1;
__REG32                       :31;
} __empgc_empgsr_bits;

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
__REG32 MIB_DISABLE   : 1;
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
} __firi_firitcr_bits;

/* FIR Transmitter Count Register (FIRITCTR) */
typedef struct{
__REG32 TPL  :11;
__REG32      :21;
} __firi_firitctr_bits;

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
} __firi_firircr_bits;

/* FIR Transmit Status Register (FIRITSR) */
typedef struct{
__REG32 TFU   : 1;
__REG32 TPE   : 1;
__REG32 SIPE  : 1;
__REG32 TC    : 1;
__REG32       : 4;
__REG32 TFP   : 8;
__REG32       :16;
} __firi_firitsr_bits;

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
} __firi_firirsr_bits;

/* FIRI Control Register */
typedef struct{
__REG32 OSF  : 4;
__REG32      : 1;
__REG32 BL   : 7;
__REG32      :20;
} __firi_firicr_bits;


/* CNTR Register Description (GPC.CNTR) */
typedef struct {
__REG32 HTRI    : 4;
__REG32         : 9;
__REG32 FUPD    : 1;
__REG32 STRT    : 1;
__REG32 ADU     : 1;
__REG32 DVFS0CR : 1;
__REG32 DVFS1CR : 1;
__REG32 DPTC0CR : 1;
__REG32 DPTC1CR : 1;
__REG32 GPCIRQ  : 1;
__REG32 GPCIRQM : 1;
__REG32         : 2;
__REG32 IRQ2    : 1;
__REG32 IRQ2M   : 1;
__REG32 CSPI    : 1;
__REG32         : 5;
} __gpc_cntr_bits;

/* PGR Register Description (GPC.PGR) */
typedef struct {
__REG32 IPUPG   : 2;
__REG32 VPUPG   : 2;
__REG32 GPUPG   : 2;
__REG32         : 2;
__REG32 CTA8PG  : 2;
__REG32         : 6;
__REG32 IPCI    : 6;
__REG32 IPCO    : 6;
__REG32 IPCC    : 1;
__REG32 DRCIC   : 2;
__REG32         : 1;
} __gpc_pgr_bits;

/* VCR Register Description (GPC.VCR) */
typedef struct {
__REG32 VCNT    :15;
__REG32         : 1;
__REG32 VCNTU   : 1;
__REG32 VINC    : 1;
__REG32         :14;
} __gpc_vcr_bits;

/* ALL_PU Register Description (GPC.ALL_PU) */
typedef struct {
__REG32 IPUPDR      : 1;
__REG32 GPUPDR      : 1;
__REG32 VPUPDR      : 1;
__REG32             : 1;
__REG32 IPUPUR      : 1;
__REG32 GPUPUR      : 1;
__REG32 VPUPUR      : 1;
__REG32             : 1;
__REG32 IPUSWSTATUS : 1;
__REG32 GPUSWSTATUS : 1;
__REG32 VPUSWSTATUS : 1;
__REG32             :21;
} __gpc_all_pu_bits;

/* NEON Register Description (GPC.NEON) */
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

/* GPT Control Register (GPT.GPTCR) */
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
} __gpt_gptcr_bits;

/* GPT Prescaler Register (GPT.GPTPR) */
typedef struct{
__REG32 PRESCALER       :12;
__REG32                 :20;
} __gpt_gptpr_bits;

/* GPT Status Register (GPT.GPTSR) */
typedef struct{
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32 OF3             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32 ROV             : 1;
__REG32                 :26;
} __gpt_gptsr_bits;

/* GPT Interrupt Register (GPT.GPTIR) */
typedef struct{
__REG32 OF1IE           : 1;
__REG32 OF2IE           : 1;
__REG32 OF3IE           : 1;
__REG32 IF1IE           : 1;
__REG32 IF2IE           : 1;
__REG32 ROVIE           : 1;
__REG32                 :26;
} __gpt_gptir_bits;

/* I2C Address Register (I2C.IADR) */
typedef struct {
__REG16      : 1;
__REG16 ADR  : 7;
__REG16      : 8;
} __i2c_iadr_bits;

/* I2C Frequency Register (I2C.IFDR) */
typedef struct {
__REG16 IC  : 6;
__REG16     :10;
} __i2c_ifdr_bits;

/* I2C Control Register (I2C.I2CR) */
typedef struct {
__REG16       : 2;
__REG16 RSTA  : 1;
__REG16 TXAK  : 1;
__REG16 MTX   : 1;
__REG16 MSTA  : 1;
__REG16 IIEN  : 1;
__REG16 IEN   : 1;
__REG16       : 8;
} __i2c_i2cr_bits;

/* I2C Status Register (I2C.I2SR) */
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
} __i2c_i2sr_bits;

/* I2C Data Register (I2C.I2DR) */
typedef struct {
__REG16 DATA  : 8;
__REG16       : 8;
} __i2c_i2dr_bits;

/* HSI2C Master/Slave Address Registers (HSI2C.HISADR,HSI2C.HIMADR) */
typedef struct {
__REG16          : 1;
__REG16 LSB_ADR  : 7;
__REG16 MSB_ADR  : 3;
__REG16          : 5;
} __hsi2c_hixadr_bits;

/* HSI2C Control Register (HSI2C.HICR) */
typedef struct {
__REG16 HIEN        : 1;
__REG16 DMA_EN_RX   : 1;
__REG16 DMA_EN_TX   : 1;
__REG16 RSTA        : 1;
__REG16 TXAK        : 1;
__REG16 MTX         : 1;
__REG16 MSTA        : 1;
__REG16 HIIEN       : 1;
__REG16 ADDR_MODE   : 1;
__REG16 HS_MST_CODE : 3;
__REG16 HSM_EN      : 1;
__REG16 SAMC        : 2;
__REG16 AUTO_RSTA   : 1;
} __hsi2c_hicr_bits;

/* HSI2C Status Register (HSI2C.HISR) */
typedef struct {
__REG16 RDF         : 1;
__REG16 TDE         : 1;
__REG16 HIAAS       : 1;
__REG16 HIAL        : 1;
__REG16 BTD         : 1;
__REG16 RDC_ZERO    : 1;
__REG16 TDC_ZERO    : 1;
__REG16 RXAK        : 1;
__REG16 HIBB        : 1;
__REG16 SRW         : 1;
__REG16 SADDR_MODE  : 1;
__REG16 SHS_MODE    : 1;
__REG16             : 4;
} __hsi2c_hisr_bits;

/* HSI2C Control Register (HSI2C.HIIMR) */
typedef struct {
__REG16 MASK_RDF    : 1;
__REG16 MASK_TDE    : 1;
__REG16 MASK_AAS    : 1;
__REG16 MASK_AL     : 1;
__REG16 MASK_BTD    : 1;
__REG16 MASK_RDC    : 1;
__REG16 MASK_TDC    : 1;
__REG16 MASK_RXAK   : 1;
__REG16             : 8;
} __hsi2c_hiimr_bits;

/* HSI2C Rx Data Register (HSI2C.HIRDR) */
typedef struct {
__REG16 RX_DATA  : 8;
__REG16          : 8;
} __hsi2c_hirdr_bits;

/* HSI2C F/S-Mode Frequency Divide Register (HSI2C.HIFSFDR) */
typedef struct {
__REG16 FSICR    : 6;
__REG16          :10;
} __hsi2c_hifsfdr_bits;

/* HSI2C HS-Mode Frequency Divide Register (HSI2C.HIHSFDR) */
typedef struct {
__REG16 HSICR    : 6;
__REG16          :10;
} __hsi2c_hihsfdr_bits;

/* HSI2C Tx FIFO Register (HSI2C.HITFR) */
typedef struct {
__REG16 TFEN     : 1;
__REG16 TFLSH    : 1;
__REG16 TFWM     : 3;
__REG16          : 3;
__REG16 TFC      : 4;
__REG16          : 4;
} __hsi2c_hitfr_bits;

/* HSI2C Rx FIFO Register (HSI2C.HIRFR) */
typedef struct {
__REG16 RFEN     : 1;
__REG16 RFLSH    : 1;
__REG16 RFWM     : 3;
__REG16          : 3;
__REG16 RFC      : 4;
__REG16          : 4;
} __hsi2c_hirfr_bits;

/* HSI2C Tx Data Count Register (HSI2C.HITDCR) */
typedef struct {
__REG16 TDC      : 8;
__REG16 TDC_EN   : 1;
__REG16 TDC_RSTA : 1;
__REG16          : 6;
} __hsi2c_hitdcr_bits;

/* HSI2C Rx Data Count Register (HSI2C.HIRDCR) */
typedef struct {
__REG16 RDC      : 8;
__REG16 RDC_EN   : 1;
__REG16 RDC_RSTA : 1;
__REG16          : 6;
} __hsi2c_hirdcr_bits;

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
__REG8  LPB_ENABLED_STATUS : 1;
__REG8  SCS                : 6;
__REG8  LOCK               : 1;
} __iim_scs1_bits;

/* Software-Controllable Signals Register 2 (SCS2) */
typedef struct{
__REG8  FBRL0     : 1;
__REG8  FBRL1     : 1;
__REG8  FBRL2     : 1;
__REG8  FBRL3     : 1;
__REG8            : 1;
__REG8            : 2;
__REG8  LOCK      : 1;
} __iim_scs2_bits;

/* Software-Controllable Signals Register 3 (SCS3) */
typedef struct{
__REG8  FBWL0     : 1;
__REG8  FBWL1     : 1;
__REG8  FBWL2     : 1;
__REG8  FBWL3     : 1;
__REG8            : 1;
__REG8            : 2;
__REG8  LOCK      : 1;
} __iim_scs3_bits;

/* General Purpose Register 0 (IOMUXC.IOMUXC_GPR0) */
typedef struct{
__REG32 SDMA_SEL_EV47     : 1;
__REG32 SDMA_SEL_EV37     : 1;
__REG32 SDMA_SEL_EV46     : 1;
__REG32 SDMA_SEL_EV35     : 1;
__REG32 SDMA_SEL_EV25     : 1;
__REG32 SDMA_SEL_EV24     : 1;
__REG32 SDMA_SEL_EV23     : 1;
__REG32 SDMA_SEL_EV21     : 1;
__REG32 SDMA_SEL_EV20     : 1;
__REG32 TPIU_TRACE_EN     : 1;
__REG32 EMI_MUX           : 1;
__REG32                   :21;
} __iomuxc_gpr0_bits;

/* General Purpose Register 1 (IOMUXC.IOMUXC_GPR1) */
typedef struct{
__REG32 GPR0              : 1;
__REG32 GPR1              : 1;
__REG32 GPR2              : 1;
__REG32 GPR3              : 1;
__REG32 GPR4              : 1;
__REG32 GPR5              : 1;
__REG32 GPR6              : 1;
__REG32 GPR7              : 1;
__REG32 GPR8              : 1;
__REG32 GPR9              : 1;
__REG32 GPR10             : 1;
__REG32 GPR11             : 1;
__REG32 GPR12             : 1;
__REG32 GPR13             : 1;
__REG32 GPR14             : 1;
__REG32 GPR15             : 1;
__REG32 GPR16             : 1;
__REG32 GPR17             : 1;
__REG32 GPR18             : 1;
__REG32 GPR19             : 1;
__REG32 GPR20             : 1;
__REG32 GPR21             : 1;
__REG32 GPR22             : 1;
__REG32 GPR23             : 1;
__REG32 GPR24             : 1;
__REG32 GPR25             : 1;
__REG32 GPR26             : 1;
__REG32 GPR27             : 1;
__REG32 GPR28             : 1;
__REG32 GPR29             : 1;
__REG32 GPR30             : 1;
__REG32 GPR31             : 1;
} __iomuxc_gpr1_bits;

/* OBSERVE_MUX Register n (IOMUXC.IOMUXC_OBSERVE_MUX_n) */
typedef struct{
__REG32 OBSRV             : 6;
__REG32                   :26;
} __iomuxc_observe_mux_bits;

/* IOMUXC_SW_MUX_CTL_PAD_EIM_DA Register n (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_EIM_DA0-15) */
typedef struct{
__REG32 MUX_MODE          : 1;
__REG32                   :31;
} __iomuxc_sw_mux_ctl_pad_muxmode1_bits;

/* IOMUXC_SW_MUX_CTL_PAD_EIM_D Register n (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_EIM_D16-27) */
typedef struct{
__REG32 MUX_MODE          : 3;
__REG32                   : 1;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_muxmode3sion_bits;

/* IOMUXC_SW_MUX_CTL_PAD_EIM_D Register n (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_EIM_D28-31) */
typedef struct{
__REG32 MUX_MODE          : 3;
__REG32                   :29;
} __iomuxc_sw_mux_ctl_pad_muxmode3_bits;

/* IOMUXC_SW_MUX_CTL_PAD_EIM_OE Register (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_EIM_OE) */
typedef struct{
__REG32 MUX_MODE          : 1;
__REG32                   : 3;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_muxmode1sion_bits;

/* IOMUXC_SW_MUX_CTL_PAD_CSI1_D Register 8 (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_CSI1_D8) */
typedef struct{
__REG32 MUX_MODE          : 2;
__REG32                   : 2;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_muxmode2sion_bits;

/* IOMUXC_SW_MUX_CTL_PAD_CSI1_D Register 10 (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_CSI1_D10) */
typedef struct{
__REG32                   : 4;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_sw_mux_ctl_pad_sion_bits;

/* IOMUXC_SW_MUX_CTL_PAD_DISP_CLK Register n (IOMUXC.IOMUXC_SW_MUX_CTL_PAD_DISP_CLK) */
typedef struct{
__REG32 MUX_MODE          : 2;
__REG32                   :30;
} __iomuxc_sw_mux_ctl_pad_muxmode2_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hpppods_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 3;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hpp_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hppp_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hpp_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pkepue_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 7;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hp_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 4;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pke_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 3;
__REG32 PUE               : 1;
__REG32                   :25;
} __iomuxc_sw_pad_ctl_pad_pue_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32                   : 6;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pke_s_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   :29;
} __iomuxc_sw_pad_ctl_pad_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_ppp_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 3;
__REG32 ODS               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pppo_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32                   : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   : 4;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   : 4;
__REG32 PKE               : 1;
__REG32                   : 5;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_pke_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   : 4;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   : 4;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_hpppod_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 4;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_sw_pad_ctl_pad_dhppp_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_sw_pad_ctl_pad_dhppp_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   : 4;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pke_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 4;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hp_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 8;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hys_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32                   : 7;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hys_s_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32                   : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hp_pod_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32                   : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hp_pods_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32                   : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_sw_pad_ctl_pad_hpp_ods_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32                   :31;
} __iomuxc_sw_pad_ctl_pad_s_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 3;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pkepue_ds_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   : 5;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_pppod_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   : 5;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32                   :25;
} __iomuxc_sw_pad_ctl_pad_pue_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 4;
__REG32 PUS               : 2;
__REG32                   :26;
} __iomuxc_sw_pad_ctl_pad_pus_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 4;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32                   :25;
} __iomuxc_sw_pad_ctl_pad_puepus_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 7;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_sw_pad_ctl_pad_pke_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32                   :29;
} __iomuxc_sw_pad_ctl_pad_d_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   : 9;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_sw_pad_ctl_pad_ddr_bits;

/* PAD_CTL Register */
typedef struct{
__REG32                   :13;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_sw_pad_ctl_pad_hve_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 DAISY             : 1;
__REG32                   :31;
} __iomuxc_daisy_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 DAISY             : 2;
__REG32                   :30;
} __iomuxc_daisy2_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 DAISY             : 3;
__REG32                   :29;
} __iomuxc_daisy3_bits;

/* PAD_CTL Register */
typedef struct{
__REG32 DAISY             : 4;
__REG32                   :28;
} __iomuxc_daisy4_bits;

/* Configuration Register (IPU.IPU_CONF) */
typedef struct{
__REG32 CSI0_EN           : 1;
__REG32 CSI1_EN           : 1;
__REG32 IC_EN             : 1;
__REG32 IRT_EN            : 1;
__REG32 ISP_EN            : 1;
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
__REG32                   : 1;
__REG32 ISP_DOUBLE_FLOW   : 1;
__REG32 IC_DMFC_SEL       : 1;
__REG32 IC_DMFC_SYNC      : 1;
__REG32 VDI_DMFC_SYNC     : 1;
__REG32 CSI0_DATA_SOURCE  : 1;
__REG32 CSI1_DATA_SOURCE  : 1;
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
__REG32 IDMAC_EOF_EN_4        : 1;
__REG32 IDMAC_EOF_EN_5        : 1;
__REG32 IDMAC_EOF_EN_6        : 1;
__REG32 IDMAC_EOF_EN_7        : 1;
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
__REG32                       : 1;
__REG32 IDMAC_EOF_EN_20       : 1;
__REG32 IDMAC_EOF_EN_21       : 1;
__REG32 IDMAC_EOF_EN_22       : 1;
__REG32 IDMAC_EOF_EN_23       : 1;
__REG32 IDMAC_EOF_EN_24       : 1;
__REG32                       : 2;
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
__REG32 IDMAC_NFACK_EN_4      : 1;
__REG32 IDMAC_NFACK_EN_5      : 1;
__REG32 IDMAC_NFACK_EN_6      : 1;
__REG32 IDMAC_NFACK_EN_7      : 1;
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
__REG32                       : 1;
__REG32 IDMAC_NFACK_EN_20     : 1;
__REG32 IDMAC_NFACK_EN_21     : 1;
__REG32 IDMAC_NFACK_EN_22     : 1;
__REG32 IDMAC_NFACK_EN_23     : 1;
__REG32 IDMAC_NFACK_EN_24     : 1;
__REG32                       : 2;
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
__REG32 IDMAC_NFB4EOF_EN_4    : 1;
__REG32 IDMAC_NFB4EOF_EN_5    : 1;
__REG32 IDMAC_NFB4EOF_EN_6    : 1;
__REG32 IDMAC_NFB4EOF_EN_7    : 1;
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
__REG32                       : 1;
__REG32 IDMAC_NFB4EOF_EN_20   : 1;
__REG32 IDMAC_NFB4EOF_EN_21   : 1;
__REG32 IDMAC_NFB4EOF_EN_22   : 1;
__REG32 IDMAC_NFB4EOF_EN_23   : 1;
__REG32 IDMAC_NFB4EOF_EN_24   : 1;
__REG32                       : 2;
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
__REG32                       :23;
__REG32 IDMAC_EOS_EN_23       : 1;
__REG32 IDMAC_EOS_EN_24       : 1;
__REG32                       : 2;
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
__REG32 ISP_PUPE_EN           : 1;
__REG32 CSI0_PUPE_EN          : 1;
__REG32 CSI1_PUPE_EN          : 1;
} __ipu_int_ctrl_9_bits;

/* IPU Interrupt Control Register 10 (IPU.IPU_INT_CTRL_10) */
typedef struct{
__REG32 SMFC0_FRM_LOST_EN         : 1;
__REG32 SMFC1_FRM_LOST_EN         : 1;
__REG32 SMFC2_FRM_LOST_EN         : 1;
__REG32 SMFC3_FRM_LOST_EN         : 1;
__REG32 ISP_RAM_ST_OF_EN          : 1;
__REG32 ISP_RAM_HIST_OF_EN        : 1;
__REG32                           :10;
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
__REG32                           : 9;
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
__REG32 IDMAC_TH_EN_4             : 1;
__REG32 IDMAC_TH_EN_5             : 1;
__REG32 IDMAC_TH_EN_6             : 1;
__REG32 IDMAC_TH_EN_7             : 1;
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
__REG32                           : 1;
__REG32 IDMAC_TH_EN_20            : 1;
__REG32 IDMAC_TH_EN_21            : 1;
__REG32 IDMAC_TH_EN_22            : 1;
__REG32 IDMAC_TH_EN_23            : 1;
__REG32 IDMAC_TH_EN_24            : 1;
__REG32                           : 2;
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
__REG32 IDMAC_EOF_SDMA_EN_4       : 1;
__REG32 IDMAC_EOF_SDMA_EN_5       : 1;
__REG32 IDMAC_EOF_SDMA_EN_6       : 1;
__REG32 IDMAC_EOF_SDMA_EN_7       : 1;
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
__REG32                           : 1;
__REG32 IDMAC_EOF_SDMA_EN_20      : 1;
__REG32 IDMAC_EOF_SDMA_EN_21      : 1;
__REG32 IDMAC_EOF_SDMA_EN_22      : 1;
__REG32 IDMAC_EOF_SDMA_EN_23      : 1;
__REG32 IDMAC_EOF_SDMA_EN_24      : 1;
__REG32                           : 2;
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
__REG32 IDMAC_NFACK_SDMA_EN_4     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_5     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_6     : 1;
__REG32 IDMAC_NFACK_SDMA_EN_7     : 1;
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
__REG32                           : 1;
__REG32 IDMAC_NFACK_SDMA_EN_20    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_21    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_22    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_23    : 1;
__REG32 IDMAC_NFACK_SDMA_EN_24    : 1;
__REG32                           : 2;
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
__REG32                           :23;
__REG32 IDMAC_EOS_SDMA_EN_23      : 1;
__REG32 IDMAC_EOS_SDMA_EN_24      : 1;
__REG32                           : 2;
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
__REG32                           : 9;
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
__REG32 IDMAC_TH_SDMA_EN_4        : 1;
__REG32 IDMAC_TH_SDMA_EN_5        : 1;
__REG32 IDMAC_TH_SDMA_EN_6        : 1;
__REG32 IDMAC_TH_SDMA_EN_7        : 1;
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
__REG32                           : 1;
__REG32 IDMAC_TH_SDMA_EN_20       : 1;
__REG32 IDMAC_TH_SDMA_EN_21       : 1;
__REG32 IDMAC_TH_SDMA_EN_22       : 1;
__REG32 IDMAC_TH_SDMA_EN_23       : 1;
__REG32 IDMAC_TH_SDMA_EN_24       : 1;
__REG32                           : 2;
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
__REG32                           : 3;
__REG32 ISP_SRM_PRI               : 3;
__REG32 ISP_SRM_MODE              : 2;
__REG32                           :11;
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
__REG32 ALT_ISP_SRC_SEL           : 4;
__REG32 PRPVF_ROT_SRC_SEL         : 4;
__REG32 PP_SRC_SEL                : 4;
__REG32 PP_ROT_SRC_SEL            : 4;
__REG32 ISP_SRC_SEL               : 4;
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
__REG32 PRP_ALT_DEST_SEL          : 4;
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
__REG32                           : 1;
__REG32 DMA_CH_DB_MODE_SEL_17     : 1;
__REG32 DMA_CH_DB_MODE_SEL_18     : 1;
__REG32                           : 1;
__REG32 DMA_CH_DB_MODE_SEL_20     : 1;
__REG32 DMA_CH_DB_MODE_SEL_21     : 1;
__REG32 DMA_CH_DB_MODE_SEL_22     : 1;
__REG32 DMA_CH_DB_MODE_SEL_23     : 1;
__REG32 DMA_CH_DB_MODE_SEL_24     : 1;
__REG32                           : 2;
__REG32 DMA_CH_DB_MODE_SEL_27     : 1;
__REG32 DMA_CH_DB_MODE_SEL_28     : 1;
__REG32 DMA_CH_DB_MODE_SEL_29     : 1;
__REG32                           : 1;
__REG32 DMA_CH_DB_MODE_SEL_31     : 1;
} __ipu_ch_db_mode_sel0_bits;

/* IPU channel double buffer mode select 1 register (IPU.IPU_CH_DB_MODE_SEL1) */
typedef struct{
__REG32                           : 1;
__REG32 DMA_CH_DB_MODE_SEL_33     : 1;
__REG32                           : 6;
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
__REG32                           :11;
} __ipu_ch_db_mode_sel1_bits;

/* IPU Alternate Channel Double Buffer Mode Select 0 Register (IPU.IPU_ALT_CH_DB_MODE_SEL0) */
typedef struct{
__REG32                           : 4;
__REG32 DMA_CH_ALT_DB_MODE_SEL_4  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_5  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_6  : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_7  : 1;
__REG32                           :16;
__REG32 DMA_CH_ALT_DB_MODE_SEL_24 : 1;
__REG32                           : 4;
__REG32 DMA_CH_ALT_DB_MODE_SEL_29 : 1;
__REG32                           : 2;
} __ipu_alt_ch_db_mode_sel0_bits;

/* IPU Alternate Channel Double Buffer Mode Select 1 Register (IPU.IPU_ALT_CH_DB_MODE_SEL1) */
typedef struct{
__REG32                           : 1;
__REG32 DMA_CH_ALT_DB_MODE_SEL_33 : 1;
__REG32                           : 7;
__REG32 DMA_CH_ALT_DB_MODE_SEL_41 : 1;
__REG32                           :10;
__REG32 DMA_CH_ALT_DB_MODE_SEL_52 : 1;
__REG32                           :11;
} __ipu_alt_ch_db_mode_sel1_bits;

/* IPU Channel Triple Buffer Mode Select 0 Register(IPU.IPU_CH_TRB_MODE_SEL0)*/
typedef struct{
__REG32                           : 8;
__REG32 DMA_CH_TRB_MODE_SEL_8     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_9     : 1;
__REG32 DMA_CH_TRB_MODE_SEL_10    : 1;
__REG32                           : 2;
__REG32 DMA_CH_TRB_MODE_SEL_13    : 1;
__REG32                           : 7;
__REG32 DMA_CH_TRB_MODE_SEL_21    : 1;
__REG32                           : 1;
__REG32 DMA_CH_TRB_MODE_SEL_23    : 1;
__REG32                           : 3;
__REG32 DMA_CH_TRB_MODE_SEL_27    : 1;
__REG32 DMA_CH_TRB_MODE_SEL_28    : 1;
__REG32                           : 3;
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
__REG32                           : 1;
__REG32 IDMAC_EOF_17              : 1;
__REG32 IDMAC_EOF_18              : 1;
__REG32                           : 1;
__REG32 IDMAC_EOF_20              : 1;
__REG32 IDMAC_EOF_21              : 1;
__REG32 IDMAC_EOF_22              : 1;
__REG32 IDMAC_EOF_23              : 1;
__REG32 IDMAC_EOF_24              : 1;
__REG32                           : 2;
__REG32 IDMAC_EOF_27              : 1;
__REG32 IDMAC_EOF_28              : 1;
__REG32 IDMAC_EOF_29              : 1;
__REG32                           : 1;
__REG32 IDMAC_EOF_31              : 1;
} __ipu_int_stat_1_bits;

/* IPU Interrupt Status Register 2 (IPU.IPU_INT_STAT_2)*/
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_EOF_33              : 1;
__REG32                           : 6;
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
__REG32                           :11;
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
__REG32                           : 1;
__REG32 IDMAC_NFACK_17            : 1;
__REG32 IDMAC_NFACK_18            : 1;
__REG32                           : 1;
__REG32 IDMAC_NFACK_20            : 1;
__REG32 IDMAC_NFACK_21            : 1;
__REG32 IDMAC_NFACK_22            : 1;
__REG32 IDMAC_NFACK_23            : 1;
__REG32 IDMAC_NFACK_24            : 1;
__REG32                           : 2;
__REG32 IDMAC_NFACK_27            : 1;
__REG32 IDMAC_NFACK_28            : 1;
__REG32 IDMAC_NFACK_29            : 1;
__REG32                           : 1;
__REG32 IDMAC_NFACK_31            : 1;
} __ipu_int_stat_3_bits;

/* IPU Interrupt Status Register 4 (IPU.IPU_INT_STAT_4)*/
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_NFACK_33            : 1;
__REG32                           : 6;
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
__REG32                           :11;
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
__REG32                           : 1;
__REG32 IDMAC_NFB4EOF_ERR_17      : 1;
__REG32 IDMAC_NFB4EOF_ERR_18      : 1;
__REG32                           : 1;
__REG32 IDMAC_NFB4EOF_ERR_20      : 1;
__REG32 IDMAC_NFB4EOF_ERR_21      : 1;
__REG32 IDMAC_NFB4EOF_ERR_22      : 1;
__REG32 IDMAC_NFB4EOF_ERR_23      : 1;
__REG32 IDMAC_NFB4EOF_ERR_24      : 1;
__REG32                           : 2;
__REG32 IDMAC_NFB4EOF_ERR_27      : 1;
__REG32 IDMAC_NFB4EOF_ERR_28      : 1;
__REG32 IDMAC_NFB4EOF_ERR_29      : 1;
__REG32                           : 1;
__REG32 IDMAC_NFB4EOF_ERR_31      : 1;
} __ipu_int_stat_5_bits;

/* IPU Interrupt Status Register 6 (IPU.IPU_INT_STAT_6) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_NFB4EOF_ERR_33      : 1;
__REG32                           : 6;
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
__REG32                           :11;
} __ipu_int_stat_6_bits;

/* IPU Interrupt Status Register 7 (IPU.IPU_INT_STAT_7) */
typedef struct{
__REG32                           :23;
__REG32 IDMAC_EOS_23              : 1;
__REG32 IDMAC_EOS_24              : 1;
__REG32                           : 2;
__REG32 IDMAC_EOS_27              : 1;
__REG32 IDMAC_EOS_28              : 1;
__REG32 IDMAC_EOS_29              : 1;
__REG32                           : 1;
__REG32 IDMAC_EOS_31              : 1;
} __ipu_int_stat_7_bits;

/* IPU Interrupt Status Register 8 (IPU.IPU_INT_STAT_8) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_EOS_33              : 1;
__REG32                           : 7;
__REG32 IDMAC_EOS_41              : 1;
__REG32 IDMAC_EOS_42              : 1;
__REG32 IDMAC_EOS_43              : 1;
__REG32 IDMAC_EOS_44              : 1;
__REG32                           : 6;
__REG32 IDMAC_EOS_51              : 1;
__REG32 IDMAC_EOS_52              : 1;
__REG32                           :11;
} __ipu_int_stat_8_bits;

/* IPU Interrupt Status Register 9 (IPU.IPU_INT_STAT_9) */
typedef struct{
__REG32 VDI_FIFO1_OVF             : 1;
__REG32                           :25;
__REG32 IC_BAYER_BUF_OVF          : 1;
__REG32 IC_ENC_BUF_OVF            : 1;
__REG32 IC_VF_BUF_OVF             : 1;
__REG32 ISP_PUPE                  : 1;
__REG32 CSI0_PUPE                 : 1;
__REG32 CSI1_PUPE                 : 1;
} __ipu_int_stat_9_bits;

/* IPU Interrupt Status Register 10 (IPU.IPU_INT_STAT_10) */
typedef struct{
__REG32 SMFC0_FRM_LOST            : 1;
__REG32 SMFC1_FRM_LOST            : 1;
__REG32 SMFC2_FRM_LOST            : 1;
__REG32 SMFC3_FRM_LOST            : 1;
__REG32 ISP_RAM_ST_OF             : 1;
__REG32 ISP_RAM_HIST_OF           : 1;
__REG32                           :10;
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
__REG32                           : 1;
__REG32 IDMAC_EOBND_5             : 1;
__REG32                           : 5;
__REG32 IDMAC_EOBND_11            : 1;
__REG32 IDMAC_EOBND_12            : 1;
__REG32                           : 7;
__REG32 IDMAC_EOBND_20            : 1;
__REG32 IDMAC_EOBND_21            : 1;
__REG32 IDMAC_EOBND_22            : 1;
__REG32                           : 9;
} __ipu_int_stat_11_bits;

/* IPU Interrupt Status Register 12 (IPU.IPU_INT_STAT_12) */
typedef struct{
__REG32                           :13;
__REG32 IDMAC_EOBND_45            : 1;
__REG32 IDMAC_EOBND_46            : 1;
__REG32 IDMAC_EOBND_47            : 1;
__REG32 IDMAC_EOBND_48            : 1;
__REG32 IDMAC_EOBND_49            : 1;
__REG32 IDMAC_EOBND_50            : 1;
__REG32                           :13;
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
__REG32                           : 1;
__REG32 IDMAC_TH_17               : 1;
__REG32 IDMAC_TH_18               : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_20               : 1;
__REG32 IDMAC_TH_21               : 1;
__REG32 IDMAC_TH_22               : 1;
__REG32 IDMAC_TH_23               : 1;
__REG32 IDMAC_TH_24               : 1;
__REG32                           : 2;
__REG32 IDMAC_TH_27               : 1;
__REG32 IDMAC_TH_28               : 1;
__REG32 IDMAC_TH_29               : 1;
__REG32                           : 1;
__REG32 IDMAC_TH_31               : 1;
} __ipu_int_stat_13_bits;

/* IPU Interrupt Status Register 14 (IPU.IPU_INT_STAT_14) */
typedef struct{
__REG32                           : 1;
__REG32 IDMAC_TH_33               : 1;
__REG32                           : 6;
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
__REG32                           :11;
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
__REG32                           : 3;
__REG32 DMA_CH_CUR_BUF_11         : 1;
__REG32 DMA_CH_CUR_BUF_12         : 1;
__REG32                           : 1;
__REG32 DMA_CH_CUR_BUF_14         : 1;
__REG32 DMA_CH_CUR_BUF_15         : 1;
__REG32                           : 1;
__REG32 DMA_CH_CUR_BUF_17         : 1;
__REG32 DMA_CH_CUR_BUF_18         : 1;
__REG32                           : 1;
__REG32 DMA_CH_CUR_BUF_20         : 1;
__REG32 DMA_CH_CUR_BUF_21         : 1;
__REG32 DMA_CH_CUR_BUF_22         : 1;
__REG32 DMA_CH_CUR_BUF_23         : 1;
__REG32 DMA_CH_CUR_BUF_24         : 1;
__REG32                           : 2;
__REG32 DMA_CH_CUR_BUF_27         : 1;
__REG32 DMA_CH_CUR_BUF_28         : 1;
__REG32 DMA_CH_CUR_BUF_29         : 1;
__REG32                           : 1;
__REG32 DMA_CH_CUR_BUF_31         : 1;
} __ipu_cur_buf_0_bits;

/* IPU Current Buffer Register 1 (IPU.IPU_CUR_BUF_1) */
typedef struct{
__REG32                           : 1;
__REG32 DMA_CH_CUR_BUF_33         : 1;
__REG32                           : 6;
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
__REG32                           :11;
} __ipu_cur_buf_1_bits;

/* IPU Alternate Current Buffer Register 0 (IPU.IPU_ALT_CUR_BUF_0) */
typedef struct{
__REG32                           : 4;
__REG32 DMA_CH_ALT_CUR_BUF_4      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_5      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_6      : 1;
__REG32 DMA_CH_ALT_CUR_BUF_7      : 1;
__REG32                           :16;
__REG32 DMA_CH_ALT_CUR_BUF_24     : 1;
__REG32                           : 4;
__REG32 DMA_CH_ALT_CUR_BUF_29     : 1;
__REG32                           : 2;
} __ipu_alt_cur_buf_0_bits;

/* IPU Alternate Current Buffer Register 1 (IPU.IPU_ALT_CUR_BUF_1) */
typedef struct{
__REG32                           : 1;
__REG32 DMA_CH_ALT_CUR_BUF_33     : 1;
__REG32                           : 7;
__REG32 DMA_CH_ALT_CUR_BUF_41     : 1;
__REG32                           :10;
__REG32 DMA_CH_ALT_CUR_BUF_52     : 1;
__REG32                           :11;
} __ipu_alt_cur_buf_1_bits;

/* IPU Shadow Register Memory Status Register (IPU.IPU_SRM_STAT) */
typedef struct{
__REG32 DP_S_SRM_STAT             : 1;
__REG32 DP_A0_SRM_STAT            : 1;
__REG32 DP_A1_SRM_STAT            : 1;
__REG32 ISP_SRM_STAT              : 1;
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
__REG32                           : 1;
__REG32 CSI2MEM_SMFC0_TSTAT       : 1;
__REG32 CSI2MEM_SMFC1_TSTAT       : 1;
__REG32 CSI2MEM_SMFC2_TSTAT       : 1;
__REG32 CSI2MEM_SMFC3_TSTAT       : 1;
__REG32                           :12;
} __ipu_proc_tasks_stat_bits;

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
__REG32 IDMAC_WM_EN_0             : 1;
__REG32 IDMAC_WM_EN_1             : 1;
__REG32 IDMAC_WM_EN_2             : 1;
__REG32 IDMAC_WM_EN_3             : 1;
__REG32 IDMAC_WM_EN_4             : 1;
__REG32 IDMAC_WM_EN_5             : 1;
__REG32 IDMAC_WM_EN_6             : 1;
__REG32 IDMAC_WM_EN_7             : 1;
__REG32 IDMAC_WM_EN_8             : 1;
__REG32 IDMAC_WM_EN_9             : 1;
__REG32 IDMAC_WM_EN_10            : 1;
__REG32 IDMAC_WM_EN_11            : 1;
__REG32 IDMAC_WM_EN_12            : 1;
__REG32 IDMAC_WM_EN_13            : 1;
__REG32 IDMAC_WM_EN_14            : 1;
__REG32 IDMAC_WM_EN_15            : 1;
__REG32 IDMAC_WM_EN_16            : 1;
__REG32 IDMAC_WM_EN_17            : 1;
__REG32 IDMAC_WM_EN_18            : 1;
__REG32 IDMAC_WM_EN_19            : 1;
__REG32 IDMAC_WM_EN_20            : 1;
__REG32 IDMAC_WM_EN_21            : 1;
__REG32 IDMAC_WM_EN_22            : 1;
__REG32 IDMAC_WM_EN_23            : 1;
__REG32 IDMAC_WM_EN_24            : 1;
__REG32 IDMAC_WM_EN_25            : 1;
__REG32 IDMAC_WM_EN_26            : 1;
__REG32 IDMAC_WM_EN_27            : 1;
__REG32 IDMAC_WM_EN_28            : 1;
__REG32 IDMAC_WM_EN_29            : 1;
__REG32 IDMAC_WM_EN_30            : 1;
__REG32 IDMAC_WM_EN_31            : 1;
} __ipu_idmac_wm_en_1_bits;

/* IDMAC Channel Watermark Enable 2 Register (IPU_IDMAC_WM_EN_2) */
typedef struct{
__REG32 IDMAC_WM_EN_32            : 1;
__REG32 IDMAC_WM_EN_33            : 1;
__REG32 IDMAC_WM_EN_34            : 1;
__REG32 IDMAC_WM_EN_35            : 1;
__REG32 IDMAC_WM_EN_36            : 1;
__REG32 IDMAC_WM_EN_37            : 1;
__REG32 IDMAC_WM_EN_38            : 1;
__REG32 IDMAC_WM_EN_39            : 1;
__REG32 IDMAC_WM_EN_40            : 1;
__REG32 IDMAC_WM_EN_41            : 1;
__REG32 IDMAC_WM_EN_42            : 1;
__REG32 IDMAC_WM_EN_43            : 1;
__REG32 IDMAC_WM_EN_44            : 1;
__REG32 IDMAC_WM_EN_45            : 1;
__REG32 IDMAC_WM_EN_46            : 1;
__REG32 IDMAC_WM_EN_47            : 1;
__REG32 IDMAC_WM_EN_48            : 1;
__REG32 IDMAC_WM_EN_49            : 1;
__REG32 IDMAC_WM_EN_50            : 1;
__REG32 IDMAC_WM_EN_51            : 1;
__REG32 IDMAC_WM_EN_52            : 1;
__REG32 IDMAC_WM_EN_53            : 1;
__REG32 IDMAC_WM_EN_54            : 1;
__REG32 IDMAC_WM_EN_55            : 1;
__REG32 IDMAC_WM_EN_56            : 1;
__REG32 IDMAC_WM_EN_57            : 1;
__REG32 IDMAC_WM_EN_58            : 1;
__REG32 IDMAC_WM_EN_59            : 1;
__REG32 IDMAC_WM_EN_60            : 1;
__REG32 IDMAC_WM_EN_61            : 1;
__REG32 IDMAC_WM_EN_62            : 1;
__REG32 IDMAC_WM_EN_63            : 1;
} __ipu_idmac_wm_en_2_bits;

/* IDMAC Channel Lock Enable 1Register (IPU_IDMAC_LOCK_EN_1) */
typedef struct{
__REG32 IDMAC_LOCK_EN_0          : 2;
__REG32 IDMAC_LOCK_EN_1          : 2;
__REG32 IDMAC_LOCK_EN_2          : 2;
__REG32 IDMAC_LOCK_EN_3          : 2;
__REG32 IDMAC_LOCK_EN_4          : 2;
__REG32 IDMAC_LOCK_EN_5          : 2;
__REG32 IDMAC_LOCK_EN_6          : 2;
__REG32 IDMAC_LOCK_EN_7          : 2;
__REG32 IDMAC_LOCK_EN_8          : 2;
__REG32 IDMAC_LOCK_EN_9          : 2;
__REG32 IDMAC_LOCK_EN_10         : 2;
__REG32                          :10;
} __ipu_idmac_lock_en_1_bits;

/* IDMAC Channel Lock Enable 2 Register (IPU_IDMAC_LOCK_EN_2) */
typedef struct{
__REG32 IDMAC_LOCK_45            : 2;
__REG32 IDMAC_LOCK_46            : 2;
__REG32 IDMAC_LOCK_47            : 2;
__REG32 IDMAC_LOCK_48            : 2;
__REG32 IDMAC_LOCK_49            : 2;
__REG32 IDMAC_LOCK_50            : 2;
__REG32                          :20;
} __ipu_idmac_lock_en_2_bits;

/* IDMAC Channel Alternate Address 1 Register (IPU_IDMAC_SUB_ADDR_1) */
typedef struct{
__REG32 IDMAC_SUB_ADDR_23        : 7;
__REG32                          : 1;
__REG32 IDMAC_SUB_ADDR_24        : 7;
__REG32                          : 1;
__REG32 IDMAC_SUB_ADDR_29        : 7;
__REG32                          : 1;
__REG32 IDMAC_SUB_ADDR_33        : 7;
__REG32                          : 1;
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
__REG32                         : 9;
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
__REG32 IDMAC_CH_BUSY_4         : 1;
__REG32 IDMAC_CH_BUSY_5         : 1;
__REG32 IDMAC_CH_BUSY_6         : 1;
__REG32 IDMAC_CH_BUSY_7         : 1;
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
__REG32                         : 2;
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
__REG32 CSI0_PG_R_VALUE         : 8;
__REG32 CSI0_PG_G_VALUE         : 8;
__REG32 CSI0_PG_B_VALUE         : 8;
__REG32 CSI0_TEST_GEN_MODE      : 1;
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
__REG32 CSI0_CPD_GR_OFFSET        :10;
__REG32 CSI0_CPD_GB_OFFSET        :10;
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
__REG32 CSI1_PG_R_VALUE           : 8;
__REG32 CSI1_PG_G_VALUE           : 8;
__REG32 CSI1_PG_B_VALUE           : 8;
__REG32 CSI1_TEST_GEN_MODE        : 1;
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

/* CSI1 Compander Control Register (IPU_CSI1_CPD_CTRL) */
typedef struct{
__REG32 CSI1_GREEN_P_BEGIN        : 1;
__REG32 CSI1_RED_ROW_BEGIN        : 1;
__REG32 CSI1_CPD                  : 3;
__REG32                           :27;
} __ipu_csi1_cpd_ctrl_bits;

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
__REG32 CSI1_CPD_GR_OFFSET        :10;
__REG32 CSI1_CPD_GB_OFFSET        :10;
__REG32 CSI1_CPD_CPD_B_OFFSET     :10;
__REG32                           : 2;
} __ipu_csi1_cpd_offset1_bits;

/* CSI1 Compander Offset Register 2 (IPU_CSI1_CPD_OFFSET2) */
typedef struct{
__REG32 CSI1_CPD_R_OFFSET         :10;
__REG32                           :22;
} __ipu_csi1_cpd_offset2_bits;

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
__REG32                                 : 2;
__REG32 VDI_TOP_FIELD_MAN               : 1;
__REG32 VDI_TOP_FIELD_AUTO              : 1;
} __ipu_vdi_c_bits;


/* Keypad Control Register (KPCR) */
typedef struct{
__REG16 KRE  : 8;
__REG16 KCO  : 8;
} __kpp_kpcr_bits;

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
} __kpp_kpsr_bits;

/* Keypad Data Direction Register (KDDR) */
typedef struct{
__REG16 KRDD  : 8;
__REG16 KCDD  : 8;
} __kpp_kddr_bits;

/* Keypad Data Register (KPDR) */
typedef struct{
__REG16 KRD  : 8;
__REG16 KCD  : 8;
} __kpp_kpdr_bits;

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

/* M4IF Debug Status Register #2 */
typedef struct{
__REG32 M0_ACC  :16;
__REG32 M1_ACC  :16;
} __m4if_mdsr2_bits;

/* M4IF Debug Status Register #3 */
typedef struct{
__REG32 M2_ACC  :16;
__REG32 M3_ACC  :16;
} __m4if_mdsr3_bits;

/* M4IF Debug Status Register #4 */
typedef struct{
__REG32 M4_ACC  :16;
__REG32 M5_ACC  :16;
} __m4if_mdsr4_bits;

/* M4IF Debug Status Register #5 */
typedef struct{
__REG32 M6_ACC  :16;
__REG32 M7_ACC  :16;
} __m4if_mdsr5_bits;

/* F_Basic Priority Reg #0 */
typedef struct{
__REG32 FBPM_M0     : 3;
__REG32             : 5;
__REG32 FBPM_M1     : 3;
__REG32             : 5;
__REG32 FBPM_M2     : 3;
__REG32             : 5;
__REG32 FBPM_M3     : 3;
__REG32             : 5;
} __m4if_fbpm0_bits;

/* F_Basic Priority Reg #1 */
typedef struct{
__REG32 FBPM_M4     : 3;
__REG32             : 5;
__REG32 FBPM_M5     : 3;
__REG32             : 5;
__REG32 FBPM_M6     : 3;
__REG32             : 5;
__REG32 FBPM_M7     : 3;
__REG32             : 5;
} __m4if_fbpm1_bits;

/* MIF4 Control Register */
typedef struct{
__REG32 MIF4_GUARD  : 4;
__REG32 MIF4_DYN_MAX: 4;
__REG32 MIF4_DYN_JMP: 4;
__REG32             : 4;
__REG32 MIF4_ACC_HIT: 3;
__REG32 MIF4_PAG_HIT: 3;
__REG32             :10;
} __m4if_mif4_bits;

/* Snooping Base Address Register #0 */
typedef struct{
__REG32 SE0         : 1;
__REG32 SWSZ0       : 4;
__REG32 SSW0        : 3;
__REG32             : 3;
__REG32 SWBA0       :21;
} __m4if_sbar0_bits;

/* Snooping Base Address Register #1 */
typedef struct{
__REG32 SE1         : 1;
__REG32 SWSZ1       : 4;
__REG32 SSW1        : 3;
__REG32             : 3;
__REG32 SWBA1       :21;
} __m4if_sbar1_bits;

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

/* Int. 2 Memory Arbitration Control Register */
typedef struct{
__REG32             : 8;
__REG32 I2RDT       : 1;
__REG32             :23;
} __m4if_i2acr_bits;

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
__REG32               : 1;
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

/* Slow Arbitration Control Register */
typedef struct{
__REG32               : 8;
__REG32 SRDT          : 1;
__REG32               :23;
} __m4if_sacr_bits;

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

/* Int. Memory Arbitration Control Register */
typedef struct{
__REG32               : 8;
__REG32 IRDT          : 1;
__REG32               :23;
} __m4if_iacr_bits;

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

/* Watermark Start ADDR_0 Register region 0 (weim cs0) */
typedef struct{
__REG32 WMS0_0        :16;
__REG32               :15;
__REG32 WE0_0         : 1;
} __m4if_wmsa0_0_bits;

/* Watermark Start ADDR_0 Register region 1 (weim cs1) */
typedef struct{
__REG32 WMS0_1        :16;
__REG32               :15;
__REG32 WE0_1         : 1;
} __m4if_wmsa0_1_bits;

/* Watermark Start ADDR_0 Register region 2 (weim cs2) */
typedef struct{
__REG32 WMS0_2        :16;
__REG32               :15;
__REG32 WE0_2         : 1;
} __m4if_wmsa0_2_bits;

/* Watermark Start ADDR_0 Register region 3 (weim cs3) */
typedef struct{
__REG32 WMS0_3        :16;
__REG32               :15;
__REG32 WE0_3         : 1;
} __m4if_wmsa0_3_bits;

/* Watermark Start ADDR_0 Register region 4 (weim cs4) */
typedef struct{
__REG32 WMS0_4        :16;
__REG32               :15;
__REG32 WE0_4         : 1;
} __m4if_wmsa0_4_bits;

/* Watermark Start ADDR_0 Register region 5 (weim cs5) */
typedef struct{
__REG32 WMS0_5        :16;
__REG32               :15;
__REG32 WE0_5         : 1;
} __m4if_wmsa0_5_bits;

/* Watermark Start ADDR_0 Register region 6 (weim cs6) */
typedef struct{
__REG32 WMS0_6        :16;
__REG32               :15;
__REG32 WE0_6         : 1;
} __m4if_wmsa0_6_bits;

/* Watermark End ADDR_0 Register region 7 (weim cs7) */
typedef struct{
__REG32 WMS0_7        :16;
__REG32               :15;
__REG32 WE0_7         : 1;
} __m4if_wmsa0_7_bits;

/* Watermark End ADDR_0 Register region 0 (weim cs0) */
typedef struct{
__REG32 WME0_0        :16;
__REG32               :16;
} __m4if_wmea0_0_bits;

/* Watermark End ADDR_0 Register region 1 (weim cs1) */
typedef struct{
__REG32 WME0_1        :16;
__REG32               :16;
} __m4if_wmea0_1_bits;

/* Watermark End ADDR_0 Register region 2 (weim cs2) */
typedef struct{
__REG32 WME0_2        :16;
__REG32               :16;
} __m4if_wmea0_2_bits;

/* Watermark End ADDR_0 Register region 3 (weim cs3) */
typedef struct{
__REG32 WME0_3        :16;
__REG32               :16;
} __m4if_wmea0_3_bits;

/* Watermark End ADDR_0 Register region 4 (weim cs4) */
typedef struct{
__REG32 WME0_4        :16;
__REG32               :16;
} __m4if_wmea0_4_bits;

/* Watermark End ADDR_0 Register region 5 (weim cs5) */
typedef struct{
__REG32 WME0_5        :16;
__REG32               :16;
} __m4if_wmea0_5_bits;

/* Watermark End ADDR_0 Register region 6 (weim cs6) */
typedef struct{
__REG32 WME0_6        :16;
__REG32               :16;
} __m4if_wmea0_6_bits;

/* Watermark End ADDR_0 Register region 7 (weim cs7) */
typedef struct{
__REG32 WME0_7        :16;
__REG32               :16;
} __m4if_wmea0_7_bits;

/* Watermark Interrupt and Status #0 Register */
typedef struct{
__REG32 WS0_0         : 1;
__REG32 WS0_1         : 1;
__REG32 WS0_2         : 1;
__REG32 WS0_3         : 1;
__REG32 WS0_4         : 1;
__REG32 WS0_5         : 1;
__REG32 WS0_6         : 1;
__REG32 WS0_7         : 1;
__REG32 SVMID_0       :11;
__REG32 FVMID_0       :11;
__REG32               : 1;
__REG32 WIE0          : 1;
} __m4if_wmis0_bits;

/* Watermark Start ADDR_1 Register region 0 (weim cs0) */
typedef struct{
__REG32 WMS1_0        :16;
__REG32               :15;
__REG32 WE1_0         : 1;
} __m4if_wmsa1_0_bits;

/* Watermark Start ADDR_1 Register region 1 (weim cs1) */
typedef struct{
__REG32 WMS1_1        :16;
__REG32               :15;
__REG32 WE1_1         : 1;
} __m4if_wmsa1_1_bits;

/* Watermark Start ADDR_1 Register region 2 (weim cs2) */
typedef struct{
__REG32 WMS1_2        :16;
__REG32               :15;
__REG32 WE1_2         : 1;
} __m4if_wmsa1_2_bits;

/* Watermark Start ADDR_1 Register region 3 (weim cs3) */
typedef struct{
__REG32 WMS1_3        :16;
__REG32               :15;
__REG32 WE1_3         : 1;
} __m4if_wmsa1_3_bits;

/* Watermark Start ADDR_1 Register region 4 (weim cs4) */
typedef struct{
__REG32 WMS1_4        :16;
__REG32               :15;
__REG32 WE1_4         : 1;
} __m4if_wmsa1_4_bits;

/* Watermark Start ADDR_1 Register region 5 (weim cs5) */
typedef struct{
__REG32 WMS1_5        :16;
__REG32               :15;
__REG32 WE1_5         : 1;
} __m4if_wmsa1_5_bits;

/* Watermark Start ADDR_1 Register region 6 (weim cs6) */
typedef struct{
__REG32 WMS1_6        :16;
__REG32               :15;
__REG32 WE1_6         : 1;
} __m4if_wmsa1_6_bits;

/* Watermark End ADDR_1 Register region 7 (weim cs7) */
typedef struct{
__REG32 WMS1_7        :16;
__REG32               :15;
__REG32 WE1_7         : 1;
} __m4if_wmsa1_7_bits;

/* Watermark End ADDR_1 Register region 0 (weim cs0) */
typedef struct{
__REG32 WME1_0        :16;
__REG32               :16;
} __m4if_wmea1_0_bits;

/* Watermark End ADDR_1 Register region 1 (weim cs1) */
typedef struct{
__REG32 WME1_1        :16;
__REG32               :16;
} __m4if_wmea1_1_bits;

/* Watermark End ADDR_1 Register region 2 (weim cs2) */
typedef struct{
__REG32 WME1_2        :16;
__REG32               :16;
} __m4if_wmea1_2_bits;

/* Watermark End ADDR_1 Register region 3 (weim cs3) */
typedef struct{
__REG32 WME1_3        :16;
__REG32               :16;
} __m4if_wmea1_3_bits;

/* Watermark End ADDR_1 Register region 4 (weim cs4) */
typedef struct{
__REG32 WME1_4        :16;
__REG32               :16;
} __m4if_wmea1_4_bits;

/* Watermark End ADDR_1 Register region 5 (weim cs5) */
typedef struct{
__REG32 WME1_5        :16;
__REG32               :16;
} __m4if_wmea1_5_bits;

/* Watermark End ADDR_1 Register region 6 (weim cs6) */
typedef struct{
__REG32 WME1_6        :16;
__REG32               :16;
} __m4if_wmea1_6_bits;

/* Watermark End ADDR_1 Register region 7 (weim cs7) */
typedef struct{
__REG32 WME1_7        :16;
__REG32               :16;
} __m4if_wmea1_7_bits;

/* Watermark Interrupt and Status #1 Register */
typedef struct{
__REG32 WS1_0         : 1;
__REG32 WS1_1         : 1;
__REG32 WS1_2         : 1;
__REG32 WS1_3         : 1;
__REG32 WS1_4         : 1;
__REG32 WS1_5         : 1;
__REG32 WS1_6         : 1;
__REG32 WS1_7         : 1;
__REG32 SVMID_1       :11;
__REG32 FVMID_1       :11;
__REG32               : 1;
__REG32 WIE1          : 1;
} __m4if_wmis1_bits;


/* NAND Flash Command (NFC.NAND_CMD) */
typedef struct{
__REG32 NAND_CMD0         : 8;
__REG32 NAND_CMD1         : 8;
__REG32 NAND_CMD2         : 8;
__REG32 NAND_CMD3         : 8;
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
__REG32 NOBER1            : 4;
__REG32 NOBER2            : 4;
__REG32 NOBER3            : 4;
__REG32 NOBER4            : 4;
__REG32 NOBER5            : 4;
__REG32 NOBER6            : 4;
__REG32 NOBER7            : 4;
__REG32 NOBER8            : 4;
} __nfc_ecc_status_result_bits;

/* Status Summary (NFC.NFC_STATUS_SUM) */
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
__REG32                   :19;
} __nfc_launch_nfc_bits;

/* NAND Flash Write Protection (NFC.NF_WR_PROT) */
typedef struct{
__REG32 WPC               : 3;
__REG32 CS2L              : 3;
__REG32 BLS               : 2;
__REG32 LTS0              : 1;
__REG32 LS0               : 1;
__REG32 US0               : 1;
__REG32 LTS1              : 1;
__REG32 LS1               : 1;
__REG32 US1               : 1;
__REG32 LTS2              : 1;
__REG32 LS2               : 1;
__REG32 US2               : 1;
__REG32 LTS3              : 1;
__REG32 LS3               : 1;
__REG32 US3               : 1;
__REG32 LTS4              : 1;
__REG32 LS4               : 1;
__REG32 US4               : 1;
__REG32 LTS5              : 1;
__REG32 LS5               : 1;
__REG32 US5               : 1;
__REG32 LTS6              : 1;
__REG32 LS6               : 1;
__REG32 US6               : 1;
__REG32 LTS7              : 1;
__REG32 LS7               : 1;
__REG32 US7               : 1;
} __nfc_wr_protect_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD0) */
typedef struct{
__REG32 USBA0       :16;
__REG32 UEBA0       :16;
} __nfc_unlock_blk_add0_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD1) */
typedef struct{
__REG32 USBA1       :16;
__REG32 UEBA1       :16;
} __nfc_unlock_blk_add1_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD2) */
typedef struct{
__REG32 USBA2       :16;
__REG32 UEBA2       :16;
} __nfc_unlock_blk_add2_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD3) */
typedef struct{
__REG32 USBA3       :16;
__REG32 UEBA3       :16;
} __nfc_unlock_blk_add3_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD4) */
typedef struct{
__REG32 USBA4       :16;
__REG32 UEBA4       :16;
} __nfc_unlock_blk_add4_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD5) */
typedef struct{
__REG32 USBA5       :16;
__REG32 UEBA5       :16;
} __nfc_unlock_blk_add5_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD6) */
typedef struct{
__REG32 USBA6       :16;
__REG32 UEBA6       :16;
} __nfc_unlock_blk_add6_bits;

/* Address to Unlock in Write Protection Mode (NFC.UNLOCK_BLK_ADD7) */
typedef struct{
__REG32 USBA7       :16;
__REG32 UEBA7       :16;
} __nfc_unlock_blk_add7_bits;

/* NAND Flash Operation Configuration2 (NFC.NFC_CONFIGURATION2) */
typedef struct{
__REG32 PS                : 2;
__REG32 SYM               : 1;
__REG32 ECC_EN            : 1;
__REG32 NUM_CMD_PHASES    : 1;
__REG32 NUM_ADR_PHASES0   : 1;
__REG32 ECC_MODE          : 1;
__REG32 PPB               : 2;
__REG32 EDC               : 3;
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
__REG32 FMP               : 4;
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

/* AXI Error Address (NFC.AXI_ERR_ADD) */
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
} __pata_control_bits;

/* INTERRUPT_PENDING Register */
typedef struct{
__REG8                      : 3;
__REG8  ata_irtrq2          : 1;
__REG8  controller_idle     : 1;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8  ata_intrq1          : 1;
} __pata_interrupt_pending_bits;

/* INTERRUPT_ENABLE Register */
typedef struct{
__REG8                      : 3;
__REG8  ata_irtrq2          : 1;
__REG8  controller_idle     : 1;
__REG8  fifo_overflow       : 1;
__REG8  fifo_underflow      : 1;
__REG8  ata_intrq1          : 1;
} __pata_interrupt_enable_bits;

/* INTERRUPT_CLEAR Register */
#define PATA_INTERRUPT_CLEAR_fifo_overflow    0x20
#define PATA_INTERRUPT_CLEAR_fifo_underflow   0x40

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
} __pwm_pwmcr_bits;

/* PWM Status Register (PWMSR) */
typedef struct{
__REG32 FIFOAV  : 3;
__REG32 FE      : 1;
__REG32 ROV     : 1;
__REG32 CMP     : 1;
__REG32 FWE     : 1;
__REG32         :25;
} __pwm_pwmsr_bits;

/* PWM Interrupt Register (PWMIR) */
typedef struct{
__REG32 FIE     : 1;
__REG32 RIE     : 1;
__REG32 CIE     : 1;
__REG32         :29;
} __pwm_pwmir_bits;

/* PWM Sample Register (PWMSAR) */
typedef struct{
__REG32 SAMPLE :16;
__REG32        :16;
} __pwm_pwmsar_bits;

/* PWM Period Register (PWMPR) */
typedef struct{
__REG32 PERIOD :16;
__REG32        :16;
} __pwm_pwmpr_bits;

/* PWM Counter Register (PWMCNR) */
typedef struct{
__REG32 COUNT  :16;
__REG32        :16;
} __pwm_pwmcnr_bits;

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
__REG32 DSPCtrl : 1;
__REG32         :19;
} __sdma_config_bits;

/* SDMA Lock Register (SDMA_LOCK) */
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

/* OnCE Trace Buffer Register (OTB) */
typedef struct{
__REG32 CHFADDR     :14;
__REG32 TADDR       :14;
__REG32 TBF         : 1;
__REG32             : 3;
} __sdma_otb_bits;

/* Profile Counter Register (PRF_CNT_x) */
typedef struct{
__REG32 COUNTER        :22;
__REG32 OFL            : 1;
__REG32 COUNTER_CONFIG : 9;
} __sdma_prf_cnt_x_bits;

/* Profile Config/Status Register (PRF_CNT) */
typedef struct{
__REG32 EN             : 1;
__REG32 INT_EN_1       : 1;
__REG32 INT_EN_2       : 1;
__REG32 INT_EN_3       : 1;
__REG32 INT_EN_4       : 1;
__REG32 INT_EN_5       : 1;
__REG32 INT_EN_6       : 1;
__REG32 OFL1           : 1;
__REG32 OFL2           : 1;
__REG32 OFL3           : 1;
__REG32 OFL4           : 1;
__REG32 OFL5           : 1;
__REG32 OFL6           : 1;
__REG32 ISR            : 1;
__REG32                :18;
} __sdma_prf_cnt_bits;

/* Channel Priority Register (CHNPRIn) */
typedef struct{
__REG32 CHNPRI      : 3;
__REG32             :29;
} __sdma_chnpri_bits;

/* SPDIF Configuration Register (SPDIF.SCR) */
typedef struct{
__REG32                 : 2;
__REG32 TxSel           : 3;
__REG32 ValCtrl         : 1;
__REG32                 : 2;
__REG32 PDIR_Tx         : 1;
__REG32                 : 1;
__REG32 TxFifo_Ctrl     : 2;
__REG32 SW_Reset        : 1;
__REG32 Low_Power       : 1;
__REG32                 : 1;
__REG32 TxFIFOEmpty_Sel : 2;
__REG32 TxAutoSync      : 1;
__REG32                 :14;
} __spdif_scr_bits;

/* InterruptEn (SPDIF.SIE) Register */
typedef struct{
__REG32                 : 1;
__REG32 TxEm            : 1;
__REG32                 :16;
__REG32 TxResyn_En      : 1;
__REG32 TxUnOv_En       : 1;
__REG32                 :12;
} __spdif_sie_bits;

/* Interrupt Status/Clear (SPDIF.SIS/SIC) Register */
typedef struct{
__REG32                 : 1;
__REG32 TxEm            : 1;
__REG32                 :16;
__REG32 TxResyn         : 1;
__REG32 TxUnOv          : 1;
__REG32                 :12;
} __spdif_sis_sic_bits;

/* SPDIF Tx Right Channel Data Register (SPDIF.STR) */
typedef struct{
__REG32 TxDataRight     :24;
__REG32                 : 8;
} __spdif_str_bits;

/* SPDIF Tx Left Channel Data Register (SPDIF.STL) */
typedef struct{
__REG32 TxDataLeft     :24;
__REG32                : 8;
} __spdif_stl_bits;

/* SPDIFTxClk Register (SPDIF.STC) */
typedef struct{
__REG32 TxClk_DF        : 7;
__REG32                 : 1;
__REG32 TxClk_Source    : 3;
__REG32                 :21;
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
__REG32                       : 1;
__REG32                       :20;
} __src_scr_bits;

/* SRC Boot Mode Register (SRC.SBMR) */
typedef struct{
__REG32 BT_MEM_CTL            : 2;
__REG32 BT_BUS_WIDTH          : 1;
__REG32 BT_PAGE_SIZE          : 2;
__REG32                       : 1;
__REG32 BT_SPARE_SIZE         : 1;
__REG32 BT_MEM_TYPE           : 2;
__REG32                       : 1;
__REG32 BT_MLC_SEL            : 1;
__REG32                       : 1;
__REG32 BT_EEPROM_CFG         : 1;
__REG32 DIR_BT_DIS            : 1;
__REG32 BMOD                  : 2;
__REG32 BT_WEIM_MUXED         : 2;
__REG32 BT_LPB_EN             : 1;
__REG32 BT_SRC                : 2;
__REG32 BT_OSC_FREQ_SEL       : 2;
__REG32 BT_LPB                : 2;
__REG32 BT_UART_SRC           : 2;
__REG32 BT_USB_SRC            : 1;
__REG32 BT_HPN_EN             : 1;
__REG32 BT_LPB_FREQ           : 3;
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
__REG32                     :20;
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
__REG32 SACDAT              :20;
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
#define SSI_SACCEN_SACCEN0    0x00000001UL
#define SSI_SACCEN_SACCEN1    0x00000002UL
#define SSI_SACCEN_SACCEN2    0x00000004UL
#define SSI_SACCEN_SACCEN3    0x00000008UL
#define SSI_SACCEN_SACCEN4    0x00000010UL
#define SSI_SACCEN_SACCEN5    0x00000020UL
#define SSI_SACCEN_SACCEN6    0x00000040UL
#define SSI_SACCEN_SACCEN7    0x00000080UL
#define SSI_SACCEN_SACCEN8    0x00000100UL
#define SSI_SACCEN_SACCEN9    0x00000200UL

/* SSI AC97 Channel Disable Register (SSI.SACCDIS) */
#define SSI_SACCDIS_SACCDIS0    0x00000001UL
#define SSI_SACCDIS_SACCDIS1    0x00000002UL
#define SSI_SACCDIS_SACCDIS2    0x00000004UL
#define SSI_SACCDIS_SACCDIS3    0x00000008UL
#define SSI_SACCDIS_SACCDIS4    0x00000010UL
#define SSI_SACCDIS_SACCDIS5    0x00000020UL
#define SSI_SACCDIS_SACCDIS6    0x00000040UL
#define SSI_SACCDIS_SACCDIS7    0x00000080UL
#define SSI_SACCDIS_SACCDIS8    0x00000100UL
#define SSI_SACCDIS_SACCDIS9    0x00000200UL

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
#define TZIC_SWINT_INITD   0x000003FFUL
#define TZIC_SWINT_INTNEG  0x80000000UL

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
#define UART_UTXD_TX_DATA   0x000000FFUL

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
__REG32 SRST   : 1;     /* Bit  0       -Software Reset 0 = Reset the tx and rx state machines*/
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
__REG16 PDE   : 1;
__REG16       :15;
} __wmcr_bits;

/* VPU Code Run Register (VPU.CodeRun) */
#define VPU_CodeRun_CodeRun      0x00000001UL

/* VPU Code Run Register (VPU.CodeRun) */
#define VPU_CodeDown_CodeData    0x0000FFFFUL
#define VPU_CodeDown_CodeAddr    0x1FFF0000UL

/* VPU Host Interrupt Request Register (VPU.HostIntReq) */
#define VPU_HostIntReq_IntReq    0x00000001UL

/* VPU BIT Interrupt Clear Register (VPU.BitIntClear) */
#define VPU_BitIntClear_IntClear 0x00000001UL

/* VPU BIT Interrupt Status Register (VPU.BitIntSts) */
typedef struct {
__REG32 IntSts    : 1;
__REG32           :31;
} __vpu_bitintsts_bits;

/* VPU BIT Code Reset Register (VPU.BitCodeReset) */
#define VPU_BitCodeReset_CodeReset 0x00000001UL

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

/* USB Control Register (USB.USB_CTRL) */
typedef struct {
__REG32 BPE         : 1;
__REG32 OTCKOEN     : 1;
__REG32             : 2;
__REG32 H1DISFSTLL  : 1;
__REG32             : 1;
__REG32 H1HSTLL     : 1;
__REG32 OHSTLL      : 1;
__REG32 H1PM        : 1;
__REG32 H1BPVAL     : 2;
__REG32 H1WIE       : 1;
__REG32 H1UIE       : 1;
__REG32 H1SIC       : 2;
__REG32 H1WIR       : 1;
__REG32 ICTPC       : 1;
__REG32 H1TCKOEN    : 1;
__REG32 UBPCKE      : 1;
__REG32 ICTPIE      : 1;
__REG32             : 3;
__REG32 ICVOL       : 1;
__REG32 OPM         : 1;
__REG32 OBPVAL      : 2;
__REG32 OWIE        : 1;
__REG32 OUIE        : 1;
__REG32 OSIC        : 2;
__REG32 OWIR        : 1;
} __usb_ctrl_bits;

/* USB OTG Mirror Register (USB.OTG_MIRROR) */
typedef struct {
__REG32 IDDIG        : 1;
__REG32 ASESVLD      : 1;
__REG32 BSESVLD      : 1;
__REG32 VBUSVAL      : 1;
__REG32 SESEND       : 1;
__REG32              : 2;
__REG32 ULPIPHY_CLK  : 1;
__REG32 UTMIPHY_CLK  : 1;
__REG32              :23;
} __usb_otg_mirror_bits;

/* USB OTG UTMI PHY Control Register 0 (USB.USB_OTG_PHY_CTRL_0) */
typedef struct {
__REG32 CHRGDET_INT_FLG  : 1;
__REG32 CHRGDET_INT_EN   : 1;
__REG32 CHRGDET          : 1;
__REG32 PWR_POL          : 1;
__REG32 H1_XCVR_CLK_SEL  : 1;
__REG32                  : 2;
__REG32 OTG_XCVR_CLK_SEL : 1;
__REG32 OTG_OVRCUR_DIS   : 1;
__REG32 OTG_OVRCUR_POL   : 1;
__REG32 UTMI_ON_CLK      : 1;
__REG32 RESET            : 1;
__REG32 SUSPENDM         : 1;
__REG32                  : 2;
__REG32 VSTATUS          : 8;
__REG32 CHGRDETON        : 1;
__REG32 CHGRDETEN        : 1;
__REG32 CONF3            : 1;
__REG32 CONF2            : 1;
__REG32 VCONTROL         : 4;
__REG32 VLOAD            : 1;
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
__REG32                     :24;
__REG32 otg_ext_clk_en      : 1;
__REG32 uh1_ext_clk_en      : 1;
__REG32 uh2_ext_clk_en      : 1;
__REG32 uh3_ext_clk_en      : 1;
__REG32                     : 4;
} __usb_ctrl_1_bits;

/* USB Host2 Control Register (USB.USB_UH2_CTRL) */
typedef struct {
__REG32                     : 1;
__REG32 H2DISFSTLL          : 1;
__REG32                     : 1;
__REG32 H2HSTLL             : 1;
__REG32 H2PM                : 1;
__REG32 OIDWK_EN            : 1;
__REG32 OVBWK_EN            : 1;
__REG32 H2WIE               : 1;
__REG32 H2UIE               : 1;
__REG32 H2SIC               : 2;
__REG32 ICTPC               : 1;
__REG32 H2TCKOEN            : 1;
__REG32 ICTPIE              : 1;
__REG32                     : 3;
__REG32 H2WIR               : 1;
__REG32 ICVOL               : 1;
__REG32 H1_SER_DRVEN        : 1;
__REG32 OTG_SER_DRVEN       : 1;
__REG32                     :11;
} __usb_uh2_ctrl_bits;

/* USB Host3 Control Register (USB.USB_UH3_CTRL) */
typedef struct {
__REG32                     : 1;
__REG32 H3DISFSTLL          : 1;
__REG32                     : 1;
__REG32 H3HSTLL             : 1;
__REG32 H3PM                : 1;
__REG32 H3BPVAL             : 2;
__REG32 H3WIE               : 1;
__REG32 H3UIE               : 1;
__REG32 H3SIC               : 2;
__REG32 ICTPC               : 1;
__REG32 H3TCKOEN            : 1;
__REG32 ICTPIE              : 1;
__REG32                     : 3;
__REG32 H3WIR               : 1;
__REG32 ICVOL               : 1;
__REG32 H3_SER_DRVEN        : 1;
__REG32 H2_SER_DRVEN        : 1;
__REG32                     :11;
} __usb_uh3_ctrl_bits;

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
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 2;
__REG32           :21;
} __uog_hwgeneral_bits;

/* Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __uog_hwhost_bits;

/* Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __uog_hwdevice_bits;

/* Buffer Hardware Parameters */
typedef struct{
__REG32 TCBURST   : 8;
__REG32 TXADD     : 8;
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
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __uh_gptimerctrl_bits;

/* USB Command Register */
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
__REG32           : 8;
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
__REG32           :13;
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
    __REG32           :25;
    __REG32 USBADR    : 7;
  };
} __uog_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
/* UHx_ASYNCLISTADDR*/
/* UOG_ASYNCLISTADDR*/
  struct{
    __REG32           : 5;
    __REG32 ASYBASE   :27;
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
__REG32 TXSCHOH     : 8;
__REG32 TXSCHEALTH  : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __uog_txfilltuning_bits;

/* IC_USB Enable and Voltage Negotiation Register */
typedef struct{
__REG32 C_VDD1    : 3;
__REG32 IC1       : 1;
__REG32 C_VDD2    : 3;
__REG32 IC2       : 1;
__REG32 C_VDD3    : 3;
__REG32 IC3       : 1;
__REG32 C_VDD4    : 3;
__REG32 IC4       : 1;
__REG32 C_VDD5    : 3;
__REG32 IC5       : 1;
__REG32 C_VDD6    : 3;
__REG32 IC6       : 1;
__REG32 C_VDD7    : 3;
__REG32 IC7       : 1;
__REG32 C_VDD8    : 3;
__REG32 IC8       : 1;
} __uog_ic_usb_bits;

/* ULPI VIEWPORT register */
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
} __uog_viewport_bits;

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
__REG32           : 1;
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
__REG32 _1msT     : 1;
__REG32 DPS       : 1;
__REG32           : 1;
__REG32 IDIS      : 1;
__REG32 AVVIS     : 1;
__REG32 ASVIS     : 1;
__REG32 BSVIS     : 1;
__REG32 BSEIS     : 1;
__REG32 _1msS     : 1;
__REG32 DPIS      : 1;
__REG32           : 1;
__REG32 IDIE      : 1;
__REG32 AVVIE     : 1;
__REG32 ASVIE     : 1;
__REG32 BSVIE     : 1;
__REG32 BSEIE     : 1;
__REG32 _1msE     : 1;
__REG32 DPIE      : 1;
__REG32           : 1;
} __uog_otgsc_bits;

/* Device Mode */
typedef struct{
__REG32 CM        : 2;
__REG32 ES        : 1;
__REG32 SLOM      : 1;
__REG32 SDIS      : 1;
__REG32           :27;
} __uog_usbmode_bits;

/* Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT  :16;
__REG32                 :16;
} __uog_endptsetupstat_bits;

/* Endpoint Initialization */
typedef struct{
__REG32 PERB            :16;
__REG32 PETB            :16;
} __uog_endptprime_bits;

/* Endpoint De-Initialize */
typedef struct{
__REG32 FERB            :16;
__REG32 FETB            :16;
} __uog_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR            :16;
__REG32 ETBR            :16;
} __uog_endptstat_bits;

/* Endpoint Compete */
typedef struct{
__REG32 ERCE            :16;
__REG32 ETCE            :16;
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

/* DPTC Control Register (DPTC.DPTCCR) */
typedef struct{
__REG32 DEN          : 1;
__REG32 VAI          : 2;
__REG32 VAIM         : 1;
__REG32 DPVV         : 1;
__REG32 DPNVCR       : 1;
__REG32 DSMM         : 1;
__REG32              :10;
__REG32 DCR          : 2;
__REG32 DRCE0        : 1;
__REG32 DRCE1        : 1;
__REG32 DRCE2        : 1;
__REG32 DRCE3        : 1;
__REG32              : 9;
} __dptc_dptccr_bits;

/* DPTC Debug Register (DPTC.DPTCCR) */
typedef struct{
__REG32 RCCRV        : 1;
__REG32 RCCR         :11;
__REG32              : 1;
__REG32 RCCLKON      : 1;
__REG32              :18;
} __dptc_dptcdbg_bits;

/* DPTC Comparator Value Registers (DPTC.DCVR0-3) */
typedef struct{
__REG32 ELV          :10;
__REG32 LLV          :11;
__REG32 ULV          :11;
} __dptc_dcvrx_bits;

/* Platform Version ID Register */
typedef struct{
__REG32 ECO             : 8;
__REG32 MINOR           : 8;
__REG32 IMPL            : 8;
__REG32 SPEC            : 8;
} __tigerp_pvid_bits;

/* General Purpose Control Register */
typedef struct{
__REG32                 :16;
__REG32 DBGEN           : 1;
__REG32 ATRDY           : 1;
__REG32 NOCLKSTP        : 1;
__REG32                 :13;
} __tigerp_gpc_bits;

/* Platform Internal Control Register */
typedef struct{
__REG32 PIC             : 8;
__REG32                 :24;
} __tigerp_pic_bits;

/* Low Power Control Register */
typedef struct{
__REG32 DSM             : 1;
__REG32 DBG_DSM         : 1;
__REG32                 :30;
} __tigerp_lpc_bits;

/* NEON Low Power Control Register */
typedef struct{
__REG32 NEON_RST        : 1;
__REG32                 :31;
} __tigerp_nlpc_bits;

/* Internal Clock Generation Control Register */
typedef struct{
__REG32 IPG_CLK_DIVR    : 3;
__REG32 IPG_PRLD        : 1;
__REG32 ACLK_DIVR       : 3;
__REG32 ACLK_PRLD       : 1;
__REG32 DT_CLK_DIVR     : 3;
__REG32 DT_PRLD         : 1;
__REG32                 :20;
} __tigerp_icgc_bits;

/* ARM Memory Configuration Register */
typedef struct{
__REG32 ALP             : 3;
__REG32 ALPEN           : 1;
__REG32                 :28;
} __tigerp_amc_bits;

/* NEON Monitor Control Register */
typedef struct{
__REG32                 :12;
__REG32 PL              : 8;
__REG32                 :10;
__REG32 NME             : 1;
__REG32 IE              : 1;
} __tigerp_nmc_bits;

/* NEON Monitor Status Register */
typedef struct{
__REG32                 :31;
__REG32 NI              : 1;
} __tigerp_nms_bits;

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

/* SIM Port1 Detect Register (PORT1_DETECT) */
typedef struct{
__REG32 SDIM1              : 1;
__REG32 SDI1               : 1;
__REG32 SPDP1              : 1;
__REG32 SPDS1              : 1;
__REG32                   :28;
} __sim_port1_detect_bits;

/* SIM Transmit Buffer Register (XMT_BUF) */
typedef struct{
__REG32 XMT               : 8;
__REG32                   :24;
} __sim_xmt_buf_bits;

/* SIM Receive Buffer Register (RCV_BUF) */
typedef struct{
__REG32 RCV               : 8;
__REG32 PE                : 1;
__REG32 FE                : 1;
__REG32 CWT               : 1;
__REG32                   :21;
} __sim_rcv_buf_bits;

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

/* SIM Clock Prescaler Register (CLK_PRESCALER) */
typedef struct{
__REG32 CLK_PRESCALER     : 8;
__REG32                   :24;
} __sim_clk_prescaler_bits;

/* SIM Receive Threshold Register (RCV_THRESHOLD) */
typedef struct{
__REG32 RDT               : 9;
__REG32 RTH               : 4;
__REG32                   :19;
} __sim_rcv_threshold_bits;

/* SIM Enable Register (ENABLE) */
typedef struct{
__REG32 RCV_EN            : 1;
__REG32 XMT_EN            : 1;
__REG32 RXDMA_EN          : 1;
__REG32 TXDMA_EN          : 1;
__REG32                   :28;
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
__REG32 RFE               : 1;
__REG32                   : 2;
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
__REG32 RFEM              : 1;
__REG32                   :18;
} __sim_int_mask_bits;

/* SIM PORT0 Detect Register (PORT0_DETECT) */
typedef struct{
__REG32 SDIM0             : 1;
__REG32 SDI0              : 1;
__REG32 SPDP0             : 1;
__REG32 SPDS0             : 1;
__REG32                   :28;
} __sim_port0_detect_bits;

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
__REG32 SOFT_RST          : 1;
__REG32 KILL_CLK          : 1;
__REG32 DOZE              : 1;
__REG32 STOP              : 1;
__REG32 DBUG              : 1;
__REG32                   :25;
} __sim_reset_cntl_bits;

/* SIM Character Wait Timer Register (CHAR_WAIT) */
typedef struct{
__REG32 CWT               :16;
__REG32                   :16;
} __sim_char_wait_bits;

/* SIM General Purpose Counter Register (GPCNT) */
typedef struct{
__REG32 GPCNT             :16;
__REG32                   :16;
} __sim_gpcnt_bits;

/* SIM Divisor Register (DIVISOR) */
typedef struct{
__REG32 DIVISOR           : 8;
__REG32                   :24;
} __sim_divisor_bits;

/* SIM Block Wait Timer Register (BWT) */
typedef struct{
__REG32 BWT               :16;
__REG32                   :16;
} __sim_bwt_bits;

/* SIM Block Guard Timer Register (BGT) */
typedef struct{
__REG32 BGT               :16;
__REG32                   :16;
} __sim_bgt_bits;

/* SIM Block Wait Timer Register (BWT_H) */
typedef struct{
__REG32 BWT_H             :16;
__REG32                   :16;
} __sim_bwt_h_bits;

/* SIM Transmit FIFO Status Register (XMT_FIFO_STAT) */
typedef struct{
__REG32 XMT_RPTR          :4;
__REG32 XMT_WPTR          :4;
__REG32 XMT_CNT           :4;
__REG32                   :20;
} __sim_xmt_fifo_stat_bits;

/* SIM Receive FIFO Counter Register (RCV_FIFO_CNT) */
typedef struct{
__REG32 RCV_CNT           :9;
__REG32                   :23;
} __sim_rcv_fifo_cnt_bits;

/* SIM Receive FIFO Counter Register (RCV_FIFO_WPTR) */
typedef struct{
__REG32 RCV_WPTR          :9;
__REG32                   :23;
} __sim_rcv_fifo_wptr_bits;

/* SIM Receive FIFO Counter Register (RCV_FIFO_RPTR) */
typedef struct{
__REG32 RCV_RPTR          :9;
__REG32                   :23;
} __sim_rcv_fifo_rptr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/

/***************************************************************************
 **
 **  AHBMAX
 **
 ***************************************************************************/
__IO_REG32_BIT(AHBMAX_MPR0,               0x83F94000,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR0,             0x83F94010,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR1,               0x83F94100,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR1,             0x83F94110,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR2,               0x83F94200,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR2,             0x83F94210,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MPR3,               0x83F94300,__READ_WRITE ,__ahbmax_mpr_bits);
__IO_REG32_BIT(AHBMAX_SGPCR3,             0x83F94310,__READ_WRITE ,__ahbmax_sgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR0,             0x83F94800,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR1,             0x83F94900,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR2,             0x83F94A00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR3,             0x83F94B00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR4,             0x83F94C00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR5,             0x83F94D00,__READ_WRITE ,__ahbmax_mgpcr_bits);
__IO_REG32_BIT(AHBMAX_MGPCR6,             0x83F94E00,__READ_WRITE ,__ahbmax_mgpcr_bits);

/***************************************************************************
 **
 **  AUDMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(AUDMUX_PTCR1,              0x83FD0000,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR1,              0x83FD0004,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR2,              0x83FD0008,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR2,              0x83FD000C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR3,              0x83FD0010,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR3,              0x83FD0014,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR4,              0x83FD0018,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR4,              0x83FD001C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR5,              0x83FD0020,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR5,              0x83FD0024,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR6,              0x83FD0028,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR6,              0x83FD002C,__READ_WRITE ,__audmux_pdcr_bits);
__IO_REG32_BIT(AUDMUX_PTCR7,              0x83FD0030,__READ_WRITE ,__audmux_ptcr_bits);
__IO_REG32_BIT(AUDMUX_PDCR7,              0x83FD0034,__READ_WRITE ,__audmux_pdcr_bits);

/***************************************************************************
 **
 **  CCM
 **
 ***************************************************************************/
__IO_REG32_BIT(CCM_CCR,                   0x73FD4000,__READ_WRITE ,__ccm_ccr_bits);
__IO_REG32_BIT(CCM_CCDR,                  0x73FD4004,__READ_WRITE ,__ccm_ccdr_bits);
__IO_REG32_BIT(CCM_CSR,                   0x73FD4008,__READ       ,__ccm_csr_bits);
__IO_REG32_BIT(CCM_CCSR,                  0x73FD400C,__READ_WRITE ,__ccm_ccsr_bits);
__IO_REG32_BIT(CCM_CACRR,                 0x73FD4010,__READ_WRITE ,__ccm_cacrr_bits);
__IO_REG32_BIT(CCM_CBCDR,                 0x73FD4014,__READ_WRITE ,__ccm_cbcdr_bits);
__IO_REG32_BIT(CCM_CBCMR,                 0x73FD4018,__READ_WRITE ,__ccm_cbcmr_bits);
__IO_REG32_BIT(CCM_CSCMR1,                0x73FD401C,__READ_WRITE ,__ccm_cscmr1_bits);
__IO_REG32_BIT(CCM_CSCMR2,                0x73FD4020,__READ_WRITE ,__ccm_cscmr2_bits);
__IO_REG32_BIT(CCM_CSCDR1,                0x73FD4024,__READ_WRITE ,__ccm_cscdr1_bits);
__IO_REG32_BIT(CCM_CS1CDR,                0x73FD4028,__READ_WRITE ,__ccm_cs1cdr_bits);
__IO_REG32_BIT(CCM_CS2CDR,                0x73FD402C,__READ_WRITE ,__ccm_cs2cdr_bits);
__IO_REG32_BIT(CCM_CDCDR,                 0x73FD4030,__READ_WRITE ,__ccm_cdcdr_bits);
__IO_REG32_BIT(CCM_CSCDR2,                0x73FD4038,__READ_WRITE ,__ccm_cscdr2_bits);
__IO_REG32_BIT(CCM_CSCDR3,                0x73FD403C,__READ_WRITE ,__ccm_cscdr3_bits);
__IO_REG32_BIT(CCM_CSCDR4,                0x73FD4040,__READ_WRITE ,__ccm_cscdr4_bits);
__IO_REG32_BIT(CCM_CWDR,                  0x73FD4044,__READ_WRITE ,__ccm_cwdr_bits);
__IO_REG32_BIT(CCM_CDHIPR,                0x73FD4048,__READ       ,__ccm_cdhipr_bits);
__IO_REG32_BIT(CCM_CDCR,                  0x73FD404C,__READ_WRITE ,__ccm_cdcr_bits);
__IO_REG32_BIT(CCM_CTOR,                  0x73FD4050,__READ_WRITE ,__ccm_ctor_bits);
__IO_REG32_BIT(CCM_CLPCR,                 0x73FD4054,__READ_WRITE ,__ccm_clpcr_bits);
__IO_REG32_BIT(CCM_CISR,                  0x73FD4058,__READ_WRITE ,__ccm_cisr_bits);
__IO_REG32_BIT(CCM_CIMR,                  0x73FD405C,__READ_WRITE ,__ccm_cimr_bits);
__IO_REG32_BIT(CCM_CCOSR,                 0x73FD4060,__READ_WRITE ,__ccm_ccosr_bits);
__IO_REG32_BIT(CCM_CGPR,                  0x73FD4064,__READ_WRITE ,__ccm_cgpr_bits);
__IO_REG32_BIT(CCM_CCGR0,                 0x73FD4068,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR1,                 0x73FD406C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR2,                 0x73FD4070,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR3,                 0x73FD4074,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR4,                 0x73FD4078,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR5,                 0x73FD407C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR6,                 0x73FD4080,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR7,                 0x73FD4084,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CMEOR,                 0x73FD4088,__READ_WRITE ,__ccm_cmeor_bits);

/***************************************************************************
 **
 **  CSPI
 **
 ***************************************************************************/
__IO_REG32(    CSPI_RXDATA,               0x83FC0000,__READ       );
__IO_REG32(    CSPI_TXDATA,               0x83FC0004,__WRITE      );
__IO_REG32_BIT(CSPI_CONREG,               0x83FC0008,__READ_WRITE ,__cspi_conreg_bits);
__IO_REG32_BIT(CSPI_INTREG,               0x83FC000C,__READ_WRITE ,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI_DMAREG,               0x83FC0010,__READ_WRITE ,__cspi_dmareg_bits);
__IO_REG32_BIT(CSPI_STATREG,              0x83FC0014,__READ_WRITE ,__cspi_statreg_bits);
__IO_REG32_BIT(CSPI_PERIODREG,            0x83FC0018,__READ_WRITE ,__cspi_periodreg_bits);
__IO_REG32_BIT(CSPI_TESTREG,              0x83FC001C,__READ_WRITE ,__cspi_testreg_bits);

/***************************************************************************
 **
 **  DPLLIP1
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLIP1_DP_CTL,            0x83F80000,__READ_WRITE ,__dpllip_dp_ctl_bits);
__IO_REG32_BIT(DPLLIP1_DP_CONFIG,         0x83F80004,__READ_WRITE ,__dpllip_dp_config_bits);
__IO_REG32_BIT(DPLLIP1_DP_OP,             0x83F80008,__READ_WRITE ,__dpllip_dp_op_bits);
__IO_REG32_BIT(DPLLIP1_DP_MFD,            0x83F8000C,__READ_WRITE ,__dpllip_dp_mfd_bits);
__IO_REG32_BIT(DPLLIP1_DP_MFN,            0x83F80010,__READ_WRITE ,__dpllip_dp_mfn_bits);
__IO_REG32_BIT(DPLLIP1_DP_MFNMINUS,       0x83F80014,__READ_WRITE ,__dpllip_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLIP1_DP_MFNPLUS,        0x83F80018,__READ_WRITE ,__dpllip_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLIP1_DP_HFS_OP,         0x83F8001C,__READ_WRITE ,__dpllip_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLIP1_DP_HFS_MFD,        0x83F80020,__READ_WRITE ,__dpllip_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLIP1_DP_HFS_MFN,        0x83F80024,__READ_WRITE ,__dpllip_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLIP1_DP_MFN_TOGC,       0x83F80028,__READ_WRITE ,__dpllip_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLIP1_DP_DESTAT,         0x83F8002C,__READ       ,__dpllip_dp_destat_bits);

/***************************************************************************
 **
 **  DPLLIP2
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLIP2_DP_CTL,            0x83F84000,__READ_WRITE ,__dpllip_dp_ctl_bits);
__IO_REG32_BIT(DPLLIP2_DP_CONFIG,         0x83F84004,__READ_WRITE ,__dpllip_dp_config_bits);
__IO_REG32_BIT(DPLLIP2_DP_OP,             0x83F84008,__READ_WRITE ,__dpllip_dp_op_bits);
__IO_REG32_BIT(DPLLIP2_DP_MFD,            0x83F8400C,__READ_WRITE ,__dpllip_dp_mfd_bits);
__IO_REG32_BIT(DPLLIP2_DP_MFN,            0x83F84010,__READ_WRITE ,__dpllip_dp_mfn_bits);
__IO_REG32_BIT(DPLLIP2_DP_MFNMINUS,       0x83F84014,__READ_WRITE ,__dpllip_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLIP2_DP_MFNPLUS,        0x83F84018,__READ_WRITE ,__dpllip_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLIP2_DP_HFS_OP,         0x83F8401C,__READ_WRITE ,__dpllip_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLIP2_DP_HFS_MFD,        0x83F84020,__READ_WRITE ,__dpllip_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLIP2_DP_HFS_MFN,        0x83F84024,__READ_WRITE ,__dpllip_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLIP2_DP_MFN_TOGC,       0x83F84028,__READ_WRITE ,__dpllip_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLIP2_DP_DESTAT,         0x83F8402C,__READ       ,__dpllip_dp_destat_bits);

/***************************************************************************
 **
 **  DPLLIP3
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLIP3_DP_CTL,            0x83F88000,__READ_WRITE ,__dpllip_dp_ctl_bits);
__IO_REG32_BIT(DPLLIP3_DP_CONFIG,         0x83F88004,__READ_WRITE ,__dpllip_dp_config_bits);
__IO_REG32_BIT(DPLLIP3_DP_OP,             0x83F88008,__READ_WRITE ,__dpllip_dp_op_bits);
__IO_REG32_BIT(DPLLIP3_DP_MFD,            0x83F8800C,__READ_WRITE ,__dpllip_dp_mfd_bits);
__IO_REG32_BIT(DPLLIP3_DP_MFN,            0x83F88010,__READ_WRITE ,__dpllip_dp_mfn_bits);
__IO_REG32_BIT(DPLLIP3_DP_MFNMINUS,       0x83F88014,__READ_WRITE ,__dpllip_dp_mfnminus_bits);
__IO_REG32_BIT(DPLLIP3_DP_MFNPLUS,        0x83F88018,__READ_WRITE ,__dpllip_dp_mfnplus_bits);
__IO_REG32_BIT(DPLLIP3_DP_HFS_OP,         0x83F8801C,__READ_WRITE ,__dpllip_dp_hfs_op_bits);
__IO_REG32_BIT(DPLLIP3_DP_HFS_MFD,        0x83F88020,__READ_WRITE ,__dpllip_dp_hfs_mfd_bits);
__IO_REG32_BIT(DPLLIP3_DP_HFS_MFN,        0x83F88024,__READ_WRITE ,__dpllip_dp_hfs_mfn_bits);
__IO_REG32_BIT(DPLLIP3_DP_MFN_TOGC,       0x83F88028,__READ_WRITE ,__dpllip_dp_mfn_togc_bits);
__IO_REG32_BIT(DPLLIP3_DP_DESTAT,         0x83F8802C,__READ       ,__dpllip_dp_destat_bits);

/***************************************************************************
 **
 **  DVFS (DVFSC)
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSC_DVFSTHRS,            0x73FD8180,__READ_WRITE ,__dvfsc_dvfsthrs_bits);
__IO_REG32_BIT(DVFSC_DVFSCOUN,            0x73FD8184,__READ_WRITE ,__dvfsc_dvfscoun_bits);
__IO_REG32_BIT(DVFSC_DVFSSIG1,            0x73FD8188,__READ_WRITE ,__dvfsc_dvfssig1_bits);
__IO_REG32_BIT(DVFSC_DVFSSIG0,            0x73FD818C,__READ_WRITE ,__dvfsc_dvfssig0_bits);
__IO_REG32_BIT(DVFSC_DVFSGPC0,            0x73FD8190,__READ_WRITE ,__dvfsc_dvfsgpc0_bits);
__IO_REG32_BIT(DVFSC_DVFSGPC1,            0x73FD8194,__READ_WRITE ,__dvfsc_dvfsgpc1_bits);
__IO_REG32_BIT(DVFSC_DVFSGPBT,            0x73FD8198,__READ_WRITE ,__dvfsc_dvfsgpbt_bits);
__IO_REG32_BIT(DVFSC_DVFSEMAC,            0x73FD819C,__READ_WRITE ,__dvfsc_dvfsemac_bits);
__IO_REG32_BIT(DVFSC_DVFSCNTR,            0x73FD81A0,__READ_WRITE ,__dvfsc_dvfscntr_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR0_0,          0x73FD81A4,__READ       ,__dvfsc_dvfsltr0_0_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR0_1,          0x73FD81A8,__READ       ,__dvfsc_dvfsltr0_1_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR1_0,          0x73FD81AC,__READ       ,__dvfsc_dvfsltr1_0_bits);
__IO_REG32_BIT(DVFSC_DVFSLTR1_1,          0x73FD81B0,__READ       ,__dvfsc_dvfsltr1_1_bits);
__IO_REG32_BIT(DVFSC_DVFSPT0,             0x73FD81B4,__READ_WRITE ,__dvfsc_dvfspt0_bits);
__IO_REG32_BIT(DVFSC_DVFSPT1,             0x73FD81B8,__READ_WRITE ,__dvfsc_dvfspt1_bits);
__IO_REG32_BIT(DVFSC_DVFSPT2,             0x73FD81BC,__READ_WRITE ,__dvfsc_dvfspt2_bits);
__IO_REG32_BIT(DVFSC_DVFSPT3,             0x73FD81C0,__READ_WRITE ,__dvfsc_dvfspt3_bits);

/***************************************************************************
 **
 **  DVFS_PER (DVFSP)
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSP_LTR0,                0x73FD81C4,__READ       ,__dvfsp_ltr0_bits);
__IO_REG32_BIT(DVFSP_LTR1,                0x73FD81C8,__READ       ,__dvfsp_ltr1_bits);
__IO_REG32_BIT(DVFSP_LTR2,                0x73FD81CC,__READ       ,__dvfsp_ltr2_bits);
__IO_REG32_BIT(DVFSP_LTR3,                0x73FD81D0,__READ       ,__dvfsp_ltr3_bits);
__IO_REG32_BIT(DVFSP_LTBR0,               0x73FD81D4,__READ_WRITE ,__dvfsp_ltbr0_bits);
__IO_REG32_BIT(DVFSP_LTBR1,               0x73FD81D8,__READ_WRITE ,__dvfsp_ltbr1_bits);
__IO_REG32_BIT(DVFSP_PMCR0,               0x73FD81DC,__READ_WRITE ,__dvfsp_pmcr0_bits);
__IO_REG32_BIT(DVFSP_PMCR1,               0x73FD81E0,__READ_WRITE ,__dvfsp_pmcr1_bits);

/***************************************************************************
 **
 **  ECSPI1
 **
 ***************************************************************************/
__IO_REG32(    ECSPI1_RXDATA,             0x70010000,__READ       );
__IO_REG32(    ECSPI1_TXDATA,             0x70010004,__WRITE      );
__IO_REG32_BIT(ECSPI1_CONTROLREG,         0x70010008,__READ_WRITE ,__ecspi_conreg_bits);
__IO_REG32_BIT(ECSPI1_CONFIGREG,          0x7001000C,__READ_WRITE ,__ecspi_configreg_bits);
__IO_REG32_BIT(ECSPI1_INTREG,             0x70010010,__READ_WRITE ,__ecspi_intreg_bits);
__IO_REG32_BIT(ECSPI1_DMAREG,             0x70010014,__READ_WRITE ,__ecspi_dmareg_bits);
__IO_REG32_BIT(ECSPI1_STATREG,            0x70010018,__READ_WRITE ,__ecspi_statreg_bits);
__IO_REG32_BIT(ECSPI1_PERIODREG,          0x7001001C,__READ_WRITE ,__ecspi_periodreg_bits);
__IO_REG32_BIT(ECSPI1_TESTREG,            0x70010020,__READ_WRITE ,__ecspi_testreg_bits);
__IO_REG32(    ECSPI1_MSGDATA,            0x70010040,__WRITE      );

/***************************************************************************
 **
 **  ECSPI2
 **
 ***************************************************************************/
__IO_REG32(    ECSPI2_RXDATA,             0x83FAC000,__READ       );
__IO_REG32(    ECSPI2_TXDATA,             0x83FAC004,__WRITE      );
__IO_REG32_BIT(ECSPI2_CONTROLREG,         0x83FAC008,__READ_WRITE ,__ecspi_conreg_bits);
__IO_REG32_BIT(ECSPI2_CONFIGREG,          0x83FAC00C,__READ_WRITE ,__ecspi_configreg_bits);
__IO_REG32_BIT(ECSPI2_INTREG,             0x83FAC010,__READ_WRITE ,__ecspi_intreg_bits);
__IO_REG32_BIT(ECSPI2_DMAREG,             0x83FAC014,__READ_WRITE ,__ecspi_dmareg_bits);
__IO_REG32_BIT(ECSPI2_STATREG,            0x83FAC018,__READ_WRITE ,__ecspi_statreg_bits);
__IO_REG32_BIT(ECSPI2_PERIODREG,          0x83FAC01C,__READ_WRITE ,__ecspi_periodreg_bits);
__IO_REG32_BIT(ECSPI2_TESTREG,            0x83FAC020,__READ_WRITE ,__ecspi_testreg_bits);
__IO_REG32(    ECSPI2_MSGDATA,            0x83FAC040,__WRITE      );

/***************************************************************************
 **
 **  WEIM
 **
 ***************************************************************************/
__IO_REG32_BIT(WEIM_CS0GCR1,              0x83FDA000,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS0GCR2,              0x83FDA004,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS0RCR1,              0x83FDA008,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS0RCR2,              0x83FDA00C,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS0WCR1,              0x83FDA010,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS0WCR2,              0x83FDA014,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_CS1GCR1,              0x83FDA018,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS1GCR2,              0x83FDA01C,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS1RCR1,              0x83FDA020,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS1RCR2,              0x83FDA024,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS1WCR1,              0x83FDA028,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS1WCR2,              0x83FDA02C,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_CS2GCR1,              0x83FDA030,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS2GCR2,              0x83FDA034,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS2RCR1,              0x83FDA038,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS2RCR2,              0x83FDA03C,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS2WCR1,              0x83FDA040,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS2WCR2,              0x83FDA044,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_CS3GCR1,              0x83FDA048,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS3GCR2,              0x83FDA04C,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS3RCR1,              0x83FDA050,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS3RCR2,              0x83FDA054,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS3WCR1,              0x83FDA058,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS3WCR2,              0x83FDA05C,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_CS4GCR1,              0x83FDA060,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS4GCR2,              0x83FDA064,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS4RCR1,              0x83FDA068,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS4RCR2,              0x83FDA06C,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS4WCR1,              0x83FDA070,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS4WCR2,              0x83FDA074,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_CS5GCR1,              0x83FDA078,__READ_WRITE ,__weim_csgcr1_bits);
__IO_REG32_BIT(WEIM_CS5GCR2,              0x83FDA07C,__READ_WRITE ,__weim_csgcr2_bits);
__IO_REG32_BIT(WEIM_CS5RCR1,              0x83FDA080,__READ_WRITE ,__weim_csrcr1_bits);
__IO_REG32_BIT(WEIM_CS5RCR2,              0x83FDA084,__READ_WRITE ,__weim_csrcr2_bits);
__IO_REG32_BIT(WEIM_CS5WCR1,              0x83FDA088,__READ_WRITE ,__weim_cswcr1_bits);
__IO_REG32_BIT(WEIM_CS5WCR2,              0x83FDA08C,__READ_WRITE ,__weim_cswcr2_bits);
__IO_REG32_BIT(WEIM_WCR,                  0x83FDA090,__READ_WRITE ,__weim_wcr_bits);
__IO_REG32_BIT(WEIM_WIAR,                 0x83FDA094,__READ_WRITE ,__weim_wiar_bits);
__IO_REG32(    WEIM_EAR,                  0x83FDA098,__READ_WRITE );

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT1_EPITCR,              0x73FAC000,__READ_WRITE ,__epit_epitcr_bits);
__IO_REG32_BIT(EPIT1_EPITSR,              0x73FAC004,__READ_WRITE ,__epit_epitsr_bits);
__IO_REG32(    EPIT1_EPITLR,              0x73FAC008,__READ_WRITE );
__IO_REG32(    EPIT1_EPITCMPR,            0x73FAC00C,__READ_WRITE );
__IO_REG32(    EPIT1_EPITCNR,             0x73FAC010,__READ       );

/***************************************************************************
 **
 **  EPIT2
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT2_EPITCR,              0x73FB0000,__READ_WRITE ,__epit_epitcr_bits);
__IO_REG32_BIT(EPIT2_EPITSR,              0x73FB0004,__READ_WRITE ,__epit_epitsr_bits);
__IO_REG32(    EPIT2_EPITLR,              0x73FB0008,__READ_WRITE );
__IO_REG32(    EPIT2_EPITCMPR,            0x73FB000C,__READ_WRITE );
__IO_REG32(    EPIT2_EPITCNR,             0x73FB0010,__READ       );

/***************************************************************************
 **
 **  ESDCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL_ESDCTL0,            0x83FD9000,__READ_WRITE ,__esdctl_esdctlx_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG0,            0x83FD9004,__READ_WRITE ,__esdctl_esdcfgx_bits);
__IO_REG32_BIT(ESDCTL_ESDCTL1,            0x83FD9008,__READ_WRITE ,__esdctl_esdctlx_bits);
__IO_REG32_BIT(ESDCTL_ESDCFG1,            0x83FD900C,__READ_WRITE ,__esdctl_esdcfgx_bits);
__IO_REG32_BIT(ESDCTL_ESDMISC,            0x83FD9010,__READ_WRITE ,__esdctl_esdmisc_bits);
__IO_REG32_BIT(ESDCTL_ESDSCR,             0x83FD9014,__READ_WRITE ,__esdctl_esdscr_bits);
__IO_REG32(    ESDCTL_ESDDAR,             0x83FD9018,__READ_WRITE );
__IO_REG32_BIT(ESDCTL_ESDCDLY1,           0x83FD9020,__READ_WRITE ,__esdctl_esdcdly1_bits);
__IO_REG32_BIT(ESDCTL_ESDCDLY2,           0x83FD9024,__READ_WRITE ,__esdctl_esdcdly2_bits);
__IO_REG32_BIT(ESDCTL_ESDCDLY3,           0x83FD9028,__READ_WRITE ,__esdctl_esdcdly3_bits);
__IO_REG32_BIT(ESDCTL_ESDCDLY4,           0x83FD902C,__READ_WRITE ,__esdctl_esdcdly4_bits);
__IO_REG32_BIT(ESDCTL_ESDCDLY5,           0x83FD9030,__READ_WRITE ,__esdctl_esdcdly5_bits);
__IO_REG32_BIT(ESDCTL_ESDGPR,             0x83FD9034,__READ_WRITE ,__esdctl_esdgpr_bits);
__IO_REG32_BIT(ESDCTL_ESDPRCT0,           0x83FD9038,__READ_WRITE ,__esdctl_esdprctx_bits);
__IO_REG32_BIT(ESDCTL_ESDPRCT1,           0x83FD903C,__READ_WRITE ,__esdctl_esdprctx_bits);

/***************************************************************************
 **
 **  ESDHC1
 **
 ***************************************************************************/
__IO_REG32(    ESDHC1_DSADDR,             0x70004000,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_BLKATTR,            0x70004004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC1_CMDARG,             0x70004008,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_XFERTYP,            0x7000400C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC1_CMDRSP0,            0x70004010,__READ       );
__IO_REG32(    ESDHC1_CMDRSP1,            0x70004014,__READ       );
__IO_REG32(    ESDHC1_CMDRSP2,            0x70004018,__READ       );
__IO_REG32(    ESDHC1_CMDRSP3,            0x7000401C,__READ       );
__IO_REG32(    ESDHC1_DATPORT,            0x70004020,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_PRSSTAT,            0x70004024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC1_PROCTL,             0x70004028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC1_SYSCTL,             0x7000402C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC1_IRQSTAT,            0x70004030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC1_IRQSTATEN,          0x70004034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC1_IRQSIGEN,           0x70004038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC1_AUTOC12ERR,         0x7000403C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC1_HOSTCAPBLT,         0x70004040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC1_WML,                0x70004044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32(    ESDHC1_FEVT,               0x70004050,__WRITE      );
__IO_REG32_BIT(ESDHC1_ADMAES,             0x70004054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC1_ADSADDR,            0x70004058,__READ_WRITE );
__IO_REG32_BIT(ESDHC1_VENDOR,             0x700040C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC1_MMCBOOT,            0x700040C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC1_HOSTVER,            0x700040FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC2
 **
 ***************************************************************************/
__IO_REG32(    ESDHC2_DSADDR,             0x70008000,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_BLKATTR,            0x70008004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC2_CMDARG,             0x70008008,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_XFERTYP,            0x7000800C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC2_CMDRSP0,            0x70008010,__READ       );
__IO_REG32(    ESDHC2_CMDRSP1,            0x70008014,__READ       );
__IO_REG32(    ESDHC2_CMDRSP2,            0x70008018,__READ       );
__IO_REG32(    ESDHC2_CMDRSP3,            0x7000801C,__READ       );
__IO_REG32(    ESDHC2_DATPORT,            0x70008020,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_PRSSTAT,            0x70008024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC2_PROCTL,             0x70008028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC2_SYSCTL,             0x7000802C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC2_IRQSTAT,            0x70008030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC2_IRQSTATEN,          0x70008034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC2_IRQSIGEN,           0x70008038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC2_AUTOC12ERR,         0x7000803C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC2_HOSTCAPBLT,         0x70008040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC2_WML,                0x70008044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32(    ESDHC2_FEVT,               0x70008050,__WRITE      );
__IO_REG32_BIT(ESDHC2_ADMAES,             0x70008054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC2_ADSADDR,            0x70008058,__READ_WRITE );
__IO_REG32_BIT(ESDHC2_VENDOR,             0x700080C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC2_MMCBOOT,            0x700080C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC2_HOSTVER,            0x700080FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC3
 **
 ***************************************************************************/
__IO_REG32(    ESDHC3_DSADDR,             0x70020000,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_BLKATTR,            0x70020004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC3_CMDARG,             0x70020008,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_XFERTYP,            0x7002000C,__READ_WRITE ,__esdhcv3_xfertyp_bits);
__IO_REG32(    ESDHC3_CMDRSP0,            0x70020010,__READ       );
__IO_REG32(    ESDHC3_CMDRSP1,            0x70020014,__READ       );
__IO_REG32(    ESDHC3_CMDRSP2,            0x70020018,__READ       );
__IO_REG32(    ESDHC3_CMDRSP3,            0x7002001C,__READ       );
__IO_REG32(    ESDHC3_DATPORT,            0x70020020,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_PRSSTAT,            0x70020024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC3_PROCTL,             0x70020028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC3_SYSCTL,             0x7002002C,__READ_WRITE ,__esdhcv3_sysctl_bits);
__IO_REG32_BIT(ESDHC3_IRQSTAT,            0x70020030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC3_IRQSTATEN,          0x70020034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC3_IRQSIGEN,           0x70020038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC3_AUTOC12ERR,         0x7002003C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC3_HOSTCAPBLT,         0x70020040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC3_WML,                0x70020044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32(    ESDHC3_FEVT,               0x70020050,__WRITE      );
__IO_REG32_BIT(ESDHC3_ADMAES,             0x70020054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC3_ADSADDR,            0x70020058,__READ_WRITE );
__IO_REG32_BIT(ESDHC3_DLLCTRL,            0x70020060,__READ_WRITE ,__esdhcv3_dllctrl_bits);
__IO_REG32_BIT(ESDHC3_DLLSTS,             0x70020064,__READ_WRITE ,__esdhcv3_dllsts_bits);
__IO_REG32_BIT(ESDHC3_VENDOR,             0x700200C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC3_MMCBOOT,            0x700200C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC3_HOSTVER,            0x700200FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  ESDHC4
 **
 ***************************************************************************/
__IO_REG32(    ESDHC4_DSADDR,             0x70024000,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_BLKATTR,            0x70024004,__READ_WRITE ,__esdhc_blkattr_bits);
__IO_REG32(    ESDHC4_CMDARG,             0x70024008,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_XFERTYP,            0x7002400C,__READ_WRITE ,__esdhc_xfertyp_bits);
__IO_REG32(    ESDHC4_CMDRSP0,            0x70024010,__READ       );
__IO_REG32(    ESDHC4_CMDRSP1,            0x70024014,__READ       );
__IO_REG32(    ESDHC4_CMDRSP2,            0x70024018,__READ       );
__IO_REG32(    ESDHC4_CMDRSP3,            0x7002401C,__READ       );
__IO_REG32(    ESDHC4_DATPORT,            0x70024020,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_PRSSTAT,            0x70024024,__READ       ,__esdhc_prsstat_bits);
__IO_REG32_BIT(ESDHC4_PROCTL,             0x70024028,__READ_WRITE ,__esdhc_proctl_bits);
__IO_REG32_BIT(ESDHC4_SYSCTL,             0x7002402C,__READ_WRITE ,__esdhc_sysctl_bits);
__IO_REG32_BIT(ESDHC4_IRQSTAT,            0x70024030,__READ       ,__esdhc_irqstat_bits);
__IO_REG32_BIT(ESDHC4_IRQSTATEN,          0x70024034,__READ_WRITE ,__esdhc_irqstaten_bits);
__IO_REG32_BIT(ESDHC4_IRQSIGEN,           0x70024038,__READ       ,__esdhc_irqsigen_bits);
__IO_REG32_BIT(ESDHC4_AUTOC12ERR,         0x7002403C,__READ       ,__esdhc_autoc12err_bits);
__IO_REG32_BIT(ESDHC4_HOSTCAPBLT,         0x70024040,__READ       ,__esdhc_hostcapblt_bits);
__IO_REG32_BIT(ESDHC4_WML,                0x70024044,__READ_WRITE ,__esdhc_wml_bits);
__IO_REG32(    ESDHC4_FEVT,               0x70024050,__WRITE      );
__IO_REG32_BIT(ESDHC4_ADMAES,             0x70024054,__READ       ,__esdhc_admaes_bits);
__IO_REG32(    ESDHC4_ADSADDR,            0x70024058,__READ_WRITE );
__IO_REG32_BIT(ESDHC4_VENDOR,             0x700240C0,__READ_WRITE ,__esdhc_vendor_bits);
__IO_REG32_BIT(ESDHC4_MMCBOOT,            0x700240C4,__READ_WRITE ,__esdhc_mmcboot_bits);
__IO_REG32_BIT(ESDHC4_HOSTVER,            0x700240FC,__READ       ,__esdhc_hostver_bits);

/***************************************************************************
 **
 **  EMI
 **
 ***************************************************************************/
__IO_REG32_BIT(EMI_IPLCK,                 0x83FDBF00,__READ_WRITE ,__emi_iplck_bits);
__IO_REG32_BIT(EMI_EICS,                  0x83FDBF04,__READ_WRITE ,__emi_eics_bits);

 /***************************************************************************
 **
 **  FEC
 **
 ***************************************************************************/
__IO_REG32_BIT(FEC_EIR,                   0x83FEC004,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_EIMR,                  0x83FEC008,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_RDAR,                  0x83FEC010,__READ_WRITE,__fec_rdar_bits);
__IO_REG32_BIT(FEC_TDAR,                  0x83FEC014,__READ_WRITE,__fec_tdar_bits);
__IO_REG32_BIT(FEC_ECR,                   0x83FEC024,__READ_WRITE,__fec_ecr_bits);
__IO_REG32_BIT(FEC_MMFR,                  0x83FEC040,__READ_WRITE,__fec_mmfr_bits);
__IO_REG32_BIT(FEC_MSCR,                  0x83FEC044,__READ_WRITE,__fec_mscr_bits);
__IO_REG32_BIT(FEC_MIBC,                  0x83FEC064,__READ_WRITE,__fec_mibc_bits);
__IO_REG32_BIT(FEC_RCR,                   0x83FEC084,__READ_WRITE,__fec_rcr_bits);
__IO_REG32_BIT(FEC_TCR,                   0x83FEC0C4,__READ_WRITE,__fec_tcr_bits);
__IO_REG32(    FEC_PALR,                  0x83FEC0E4,__READ_WRITE);
__IO_REG32_BIT(FEC_PAUR,                  0x83FEC0E8,__READ_WRITE,__fec_paur_bits);
__IO_REG32_BIT(FEC_OPD,                   0x83FEC0EC,__READ_WRITE,__fec_opd_bits);
__IO_REG32(    FEC_IAUR,                  0x83FEC118,__READ_WRITE);
__IO_REG32(    FEC_IALR,                  0x83FEC11C,__READ_WRITE);
__IO_REG32(    FEC_GAUR,                  0x83FEC120,__READ_WRITE);
__IO_REG32(    FEC_GALR,                  0x83FEC124,__READ_WRITE);
__IO_REG32_BIT(FEC_TFWR,                  0x83FEC144,__READ_WRITE,__fec_tfwr_bits);
__IO_REG32_BIT(FEC_FRBR,                  0x83FEC14C,__READ      ,__fec_frbr_bits);
__IO_REG32_BIT(FEC_FRSR,                  0x83FEC150,__READ_WRITE,__fec_frsr_bits);
__IO_REG32(    FEC_ERDSR,                 0x83FEC180,__READ_WRITE);
__IO_REG32(    FEC_ETDSR,                 0x83FEC184,__READ_WRITE);
__IO_REG32_BIT(FEC_EMRBR,                 0x83FEC188,__READ_WRITE,__fec_emrbr_bits);
__IO_REG32(    FEC_RMON_T_DROP,           0x83FEC200,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_PACKETS,        0x83FEC204,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_BC_PKT,         0x83FEC208,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_MC_PKT,         0x83FEC20C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_CRC_ALIGN,      0x83FEC210,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_UNDERSIZE,      0x83FEC214,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OVERSIZE,       0x83FEC218,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_FRAG,           0x83FEC21C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_JAB,            0x83FEC220,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_COL,            0x83FEC224,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P64,            0x83FEC228,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P65TO127,       0x83FEC22C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P128TO255,      0x83FEC230,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P256TO511,      0x83FEC234,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P512TO1023,     0x83FEC238,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P1024TO2047,    0x83FEC23C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P_GTE2048,      0x83FEC240,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OCTETS,         0x83FEC244,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DROP,           0x83FEC248,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FRAME_OK,       0x83FEC24C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_1COL,           0x83FEC250,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MCOL,           0x83FEC254,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DEF,            0x83FEC258,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_LCOL,           0x83FEC25C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_EXCOL,          0x83FEC260,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MACERR,         0x83FEC264,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_CSERR,          0x83FEC268,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_SQE,            0x83FEC26C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FDXFC,          0x83FEC270,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_OCTETS_OK,      0x83FEC274,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_PACKETS,        0x83FEC284,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_BC_PKT,         0x83FEC288,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_MC_PKT,         0x83FEC28C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_CRC_ALIGN,      0x83FEC290,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_UNDERSIZE,      0x83FEC294,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OVERSIZE,       0x83FEC298,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_FRAG,           0x83FEC29C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_JAB,            0x83FEC2A0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_RESVD_0,        0x83FEC2A4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P64,            0x83FEC2A8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P65TO127,       0x83FEC2AC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P128TO255,      0x83FEC2B0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P256TO511,      0x83FEC2B4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P512TO1023,     0x83FEC2B8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P1024TO2047,    0x83FEC2BC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P_GTE2048,      0x83FEC2C0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OCTETS,         0x83FEC2C4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_DROP,           0x83FEC2C8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FRAME_OK,       0x83FEC2CC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_CRC,            0x83FEC2D0,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_ALIGN,          0x83FEC2D4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_MACERR,         0x83FEC2D8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FDXFC,          0x83FEC2DC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_OCTETS_OK,      0x83FEC2E0,__READ_WRITE);

/***************************************************************************
 **
 ** FIRI
 **
 ***************************************************************************/
__IO_REG32_BIT(FIRI_FIRITCR,              0x83FA8000,__READ_WRITE ,__firi_firitcr_bits);
__IO_REG32_BIT(FIRI_FIRITCTR,             0x83FA8004,__READ_WRITE ,__firi_firitctr_bits);
__IO_REG32_BIT(FIRI_FIRIRCR,              0x83FA8008,__READ_WRITE ,__firi_firircr_bits);
__IO_REG32_BIT(FIRI_FIRITSR,              0x83FA800C,__READ_WRITE ,__firi_firitsr_bits);
__IO_REG32_BIT(FIRI_FIRIRSR,              0x83FA8010,__READ_WRITE ,__firi_firirsr_bits);
__IO_REG32(    FIRI_FIRITXD,              0x83FA8014,__WRITE      );
__IO_REG32(    FIRI_FIRIRXD,              0x83FA8018,__READ       );
__IO_REG32_BIT(FIRI_FIRICR,               0x83FA801C,__READ_WRITE ,__firi_firicr_bits);

/***************************************************************************
 **
 ** EMPGC0
 **
 ***************************************************************************/
__IO_REG32_BIT(EMPGC0_EMPGCR,             0x73FD82C0,__READ_WRITE ,__empgc_empgcr_bits);
__IO_REG32_BIT(EMPGC0_PUPSCR,             0x73FD82C4,__READ_WRITE ,__empgc_pupscr_bits);
__IO_REG32_BIT(EMPGC0_PDNSCR,             0x73FD82C8,__READ_WRITE ,__empgc_pdnscr_bits);
__IO_REG32_BIT(EMPGC0_EMPGSR,             0x73FD82CC,__READ_WRITE ,__empgc_empgsr_bits);

/***************************************************************************
 **
 ** EMPGC1
 **
 ***************************************************************************/
__IO_REG32_BIT(EMPGC1_EMPGCR,             0x73FD82D0,__READ_WRITE ,__empgc_empgcr_bits);
__IO_REG32_BIT(EMPGC1_PUPSCR,             0x73FD82D4,__READ_WRITE ,__empgc_pupscr_bits);
__IO_REG32_BIT(EMPGC1_PDNSCR,             0x73FD82D8,__READ_WRITE ,__empgc_pdnscr_bits);
__IO_REG32_BIT(EMPGC1_EMPGSR,             0x73FD82DC,__READ_WRITE ,__empgc_empgsr_bits);

/***************************************************************************
 **
 ** GPC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPC_CNTR,                  0x73FD8000,__READ_WRITE ,__gpc_cntr_bits);
__IO_REG32_BIT(GPC_PGR,                   0x73FD8004,__READ_WRITE ,__gpc_pgr_bits);
__IO_REG32_BIT(GPC_VCR,                   0x73FD8008,__READ_WRITE ,__gpc_vcr_bits);
__IO_REG32_BIT(GPC_ALL_PU,                0x73FD800C,__READ_WRITE ,__gpc_all_pu_bits);
__IO_REG32_BIT(GPC_NEON,                  0x73FD8010,__READ_WRITE ,__gpc_neon_bits);

/***************************************************************************
 **
 **  GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_DR,                  0x73F84000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_GDIR,                0x73F84004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_PSR,                 0x73F84008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO1_ICR1,                0x73F8400C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO1_ICR2,                0x73F84010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO1_IMR,                 0x73F84014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_ISR,                 0x73F84018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO1_EDGE_SEL,            0x73F8401C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO2_DR,                  0x73F88000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_GDIR,                0x73F88004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_PSR,                 0x73F88008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO2_ICR1,                0x73F8800C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO2_ICR2,                0x73F88010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO2_IMR,                 0x73F88014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_ISR,                 0x73F88018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO2_EDGE_SEL,            0x73F8801C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO3_DR,                  0x73F8C000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_GDIR,                0x73F8C004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_PSR,                 0x73F8C008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO3_ICR1,                0x73F8C00C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO3_ICR2,                0x73F8C010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO3_IMR,                 0x73F8C014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_ISR,                 0x73F8C018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO3_EDGE_SEL,            0x73F8C01C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPIO4
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO4_DR,                  0x73F90000,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_GDIR,                0x73F90004,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_PSR,                 0x73F90008,__READ       ,__BITS32);
__IO_REG32_BIT(GPIO4_ICR1,                0x73F9000C,__READ_WRITE ,__icr1_bits);
__IO_REG32_BIT(GPIO4_ICR2,                0x73F90010,__READ_WRITE ,__icr2_bits);
__IO_REG32_BIT(GPIO4_IMR,                 0x73F90014,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_ISR,                 0x73F90018,__READ_WRITE ,__BITS32);
__IO_REG32_BIT(GPIO4_EDGE_SEL,            0x73F9001C,__READ_WRITE ,__BITS32);

/***************************************************************************
 **
 **  GPT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT_GPTCR,                 0x73FA0000,__READ_WRITE ,__gpt_gptcr_bits);
__IO_REG32_BIT(GPT_GPTPR,                 0x73FA0004,__READ_WRITE ,__gpt_gptpr_bits);
__IO_REG32_BIT(GPT_GPTSR,                 0x73FA0008,__READ_WRITE ,__gpt_gptsr_bits);
__IO_REG32_BIT(GPT_GPTIR,                 0x73FA000C,__READ_WRITE ,__gpt_gptir_bits);
__IO_REG32(    GPT_GPTOCR1,               0x73FA0010,__READ_WRITE );
__IO_REG32(    GPT_GPTOCR2,               0x73FA0014,__READ_WRITE );
__IO_REG32(    GPT_GPTOCR3,               0x73FA0018,__READ_WRITE );
__IO_REG32(    GPT_GPTICR1,               0x73FA001C,__READ       );
__IO_REG32(    GPT_GPTICR2,               0x73FA0020,__READ       );
__IO_REG32(    GPT_GPTCNT,                0x73FA0024,__READ       );

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C1_IADR,                 0x83FC8000,__READ_WRITE ,__i2c_iadr_bits);
__IO_REG16_BIT(I2C1_IFDR,                 0x83FC8004,__READ_WRITE ,__i2c_ifdr_bits);
__IO_REG16_BIT(I2C1_I2CR,                 0x83FC8008,__READ_WRITE ,__i2c_i2cr_bits);
__IO_REG16_BIT(I2C1_I2SR,                 0x83FC800C,__READ_WRITE ,__i2c_i2sr_bits);
__IO_REG16_BIT(I2C1_I2DR,                 0x83FC8010,__READ_WRITE ,__i2c_i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C2_IADR,                 0x83FC4000,__READ_WRITE ,__i2c_iadr_bits);
__IO_REG16_BIT(I2C2_IFDR,                 0x83FC4004,__READ_WRITE ,__i2c_ifdr_bits);
__IO_REG16_BIT(I2C2_I2CR,                 0x83FC4008,__READ_WRITE ,__i2c_i2cr_bits);
__IO_REG16_BIT(I2C2_I2SR,                 0x83FC400C,__READ_WRITE ,__i2c_i2sr_bits);
__IO_REG16_BIT(I2C2_I2DR,                 0x83FC4010,__READ_WRITE ,__i2c_i2dr_bits);

/***************************************************************************
 **
 **  HSI2C
 **
 ***************************************************************************/
__IO_REG16_BIT(HSI2C_HISADR,              0x70038000,__READ_WRITE ,__hsi2c_hixadr_bits);
__IO_REG16_BIT(HSI2C_HIMADR,              0x70038004,__READ_WRITE ,__hsi2c_hixadr_bits);
__IO_REG16_BIT(HSI2C_HICR,                0x70038008,__READ_WRITE ,__hsi2c_hicr_bits);
__IO_REG16_BIT(HSI2C_HISR,                0x7003800C,__READ_WRITE ,__hsi2c_hisr_bits);
__IO_REG16_BIT(HSI2C_HIIMR,               0x70038010,__READ_WRITE ,__hsi2c_hiimr_bits);
__IO_REG16(    HSI2C_HITDR,               0x70038014,__WRITE      );
__IO_REG16_BIT(HSI2C_HIRDR,               0x70038018,__READ       ,__hsi2c_hirdr_bits);
__IO_REG16_BIT(HSI2C_HIFSFDR,             0x7003801C,__READ_WRITE ,__hsi2c_hifsfdr_bits);
__IO_REG16_BIT(HSI2C_HIHSFDR,             0x70038020,__READ_WRITE ,__hsi2c_hihsfdr_bits);
__IO_REG16_BIT(HSI2C_HITFR,               0x70038024,__READ_WRITE ,__hsi2c_hitfr_bits);
__IO_REG16_BIT(HSI2C_HIRFR,               0x70038028,__READ_WRITE ,__hsi2c_hirfr_bits);
__IO_REG16_BIT(HSI2C_HITDCR,              0x7003802C,__READ_WRITE ,__hsi2c_hitdcr_bits);
__IO_REG16_BIT(HSI2C_HIRDCR,              0x70038030,__READ_WRITE ,__hsi2c_hirdcr_bits);


/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT(IMM_STAT,                   0x83F98000,__READ_WRITE ,__iim_stat_bits);
__IO_REG8_BIT(IMM_STATM,                  0x83F98004,__READ_WRITE ,__iim_statm_bits);
__IO_REG8_BIT(IMM_ERR,                    0x83F98008,__READ_WRITE ,__iim_err_bits);
__IO_REG8_BIT(IMM_EMASK,                  0x83F9800C,__READ_WRITE ,__iim_emask_bits);
__IO_REG8_BIT(IMM_FCTL,                   0x83F98010,__READ_WRITE ,__iim_fctl_bits);
__IO_REG8_BIT(IMM_UA,                     0x83F98014,__READ_WRITE ,__iim_ua_bits);
__IO_REG8(    IMM_LA,                     0x83F98018,__READ_WRITE );
__IO_REG8(    IMM_SDAT,                   0x83F9801C,__READ       );
__IO_REG8_BIT(IMM_PREV,                   0x83F98020,__READ       ,__iim_prev_bits);
__IO_REG8(    IMM_SREV,                   0x83F98024,__READ       );
__IO_REG8(    IMM_PREG_P,                 0x83F98028,__READ_WRITE );
__IO_REG8_BIT(IMM_SCS0,                   0x83F9802C,__READ_WRITE ,__iim_scs0_bits);
__IO_REG8_BIT(IMM_SCS1,                   0x83F98030,__READ_WRITE ,__iim_scs1_bits);
__IO_REG8_BIT(IMM_SCS2,                   0x83F98034,__READ_WRITE ,__iim_scs2_bits);
__IO_REG8_BIT(IMM_SCS3,                   0x83F98038,__READ_WRITE ,__iim_scs3_bits);
__IO_REG8(    IMM_SaharaEn0,              0x83F9803C,__READ_WRITE );
__IO_REG8(    IMM_SaharaEn1,              0x83F98040,__READ_WRITE );
__IO_REG8(    IMM_SaharaEn2,              0x83F98044,__READ_WRITE );
__IO_REG8(    IMM_SaharaEn3,              0x83F98048,__READ_WRITE );
__IO_REG8(    IMM_SaharaEn4,              0x83F9804C,__READ_WRITE );
__IO_REG8(    IMM_SaharaEn5,              0x83F98050,__READ_WRITE );


/***************************************************************************
 **
 **  IOMUXC
 **
 ***************************************************************************/
__IO_REG32_BIT(IOMUXC_GPR0,                         0x73FA8000,__READ_WRITE ,__iomuxc_gpr0_bits);
__IO_REG32_BIT(IOMUXC_GPR1,                         0x73FA8004,__READ_WRITE ,__iomuxc_gpr1_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_0,                0x73FA8008,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_1,                0x73FA800C,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_2,                0x73FA8010,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_3,                0x73FA8014,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_OBSERVE_MUX_4,                0x73FA8018,__READ_WRITE ,__iomuxc_observe_mux_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA0,       0x73FA801C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA1,       0x73FA8020,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA2,       0x73FA8024,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA3,       0x73FA8028,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA4,       0x73FA802C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA5,       0x73FA8030,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA6,       0x73FA8034,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA7,       0x73FA8038,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA8,       0x73FA803C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA9,       0x73FA8040,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA10,      0x73FA8044,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA11,      0x73FA8048,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA12,      0x73FA804C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA13,      0x73FA8050,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA14,      0x73FA8054,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DA15,      0x73FA8058,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D16,       0x73FA805C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D17,       0x73FA8060,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D18,       0x73FA8064,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D19,       0x73FA8068,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D20,       0x73FA806C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D21,       0x73FA8070,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D22,       0x73FA8074,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D23,       0x73FA8078,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D24,       0x73FA807C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D25,       0x73FA8080,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D26,       0x73FA8084,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D27,       0x73FA8088,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D28,       0x73FA808C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D29,       0x73FA8090,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D30,       0x73FA8094,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_D31,       0x73FA8098,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A16,       0x73FA809C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A17,       0x73FA80A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A18,       0x73FA80A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A19,       0x73FA80A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A20,       0x73FA80AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A21,       0x73FA80B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A22,       0x73FA80B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A23,       0x73FA80B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A24,       0x73FA80BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A25,       0x73FA80C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A26,       0x73FA80C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_A27,       0x73FA80C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB0,       0x73FA80CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB1,       0x73FA80D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB2,       0x73FA80D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_EB3,       0x73FA80D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_OE,        0x73FA80DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS0,       0x73FA80E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS1,       0x73FA80E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS2,       0x73FA80E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS3,       0x73FA80EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS4,       0x73FA80F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CS5,       0x73FA80F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_DTACK,     0x73FA80F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_LBA,       0x73FA80FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_EIM_CRE,       0x73FA8100,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DRAM_CS1,      0x73FA8104,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_WE_B,    0x73FA8108,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RE_B,    0x73FA810C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_ALE,     0x73FA8110,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CLE,     0x73FA8114,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_WP_B,    0x73FA8118,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RB0,     0x73FA811C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RB1,     0x73FA8120,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RB2,     0x73FA8124,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RB3,     0x73FA8128,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO_NAND,     0x73FA812C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS0,     0x73FA8130,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS1,     0x73FA8134,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS2,     0x73FA8138,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS3,     0x73FA813C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS4,     0x73FA8140,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS5,     0x73FA8144,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS6,     0x73FA8148,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_CS7,     0x73FA814C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_RDY_INT, 0x73FA8150,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D15,     0x73FA8154,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D14,     0x73FA8158,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D13,     0x73FA815C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D12,     0x73FA8160,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D11,     0x73FA8164,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D10,     0x73FA8168,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D9,      0x73FA816C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D8,      0x73FA8170,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D7,      0x73FA8174,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D6,      0x73FA8178,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D5,      0x73FA817C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D4,      0x73FA8180,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D3,      0x73FA8184,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D2,      0x73FA8188,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D1,      0x73FA818C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_NANDF_D0,      0x73FA8190,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D8,       0x73FA8194,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D9,       0x73FA8198,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D10,      0x73FA819C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D11,      0x73FA81A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D12,      0x73FA81A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D13,      0x73FA81A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D14,      0x73FA81AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D15,      0x73FA81B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D16,      0x73FA81B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D17,      0x73FA81B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D18,      0x73FA81BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_D19,      0x73FA81C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_VSYNC,    0x73FA81C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI1_HSYNC,    0x73FA81C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D12,      0x73FA81CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D13,      0x73FA81D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D14,      0x73FA81D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D15,      0x73FA81D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D16,      0x73FA81DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D17,      0x73FA81E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D18,      0x73FA81E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_D19,      0x73FA81E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_VSYNC,    0x73FA81EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_HSYNC,    0x73FA81F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSI2_PIXCLK,   0x73FA81F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_CLK,      0x73FA81F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_I2C1_DAT,      0x73FA81FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_AUD3_BB_TXD,   0x73FA8200,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_AUD3_BB_RXD,   0x73FA8204,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_AUD3_BB_CK,    0x73FA8208,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_AUD3_BB_FS,    0x73FA820C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MOSI,    0x73FA8210,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_MISO,    0x73FA8214,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS0,     0x73FA8218,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SS1,     0x73FA821C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_RDY,     0x73FA8220,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_CSPI1_SCLK,    0x73FA8224,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_RXD,     0x73FA8228,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_TXD,     0x73FA822C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_RTS,     0x73FA8230,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART1_CTS,     0x73FA8234,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_RXD,     0x73FA8238,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART2_TXD,     0x73FA823C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART3_RXD,     0x73FA8240,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_UART3_TXD,     0x73FA8244,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_OWIRE_LINE,    0x73FA8248,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW0,      0x73FA824C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW1,      0x73FA8250,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW2,      0x73FA8254,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_ROW3,      0x73FA8258,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL0,      0x73FA825C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL1,      0x73FA8260,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL2,      0x73FA8264,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL3,      0x73FA8268,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL4,      0x73FA826C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_KEY_COL5,      0x73FA8270,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_JTAG_DE_B,     0x73FA8274,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_CLK,     0x73FA8278,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DIR,     0x73FA827C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_STP,     0x73FA8280,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_NXT,     0x73FA8284,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA0,   0x73FA8288,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA1,   0x73FA828C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA2,   0x73FA8290,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA3,   0x73FA8294,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA4,   0x73FA8298,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA5,   0x73FA829C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA6,   0x73FA82A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_USBH1_DATA7,   0x73FA82A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_PIN11,     0x73FA82A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_PIN12,     0x73FA82AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_PIN13,     0x73FA82B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_D0_CS,     0x73FA82B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_D1_CS,     0x73FA82B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISPB2_SER_DIN,0x73FA82BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISPB2_SER_DIO,0x73FA82C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISPB2_SER_CLK,0x73FA82C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISPB2_SER_RS, 0x73FA82C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT0,    0x73FA82CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT1,    0x73FA82D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT2,    0x73FA82D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT3,    0x73FA82D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT4,    0x73FA82DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT5,    0x73FA82E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT6,    0x73FA82E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT7,    0x73FA82E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT8,    0x73FA82EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT9,    0x73FA82F0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT10,   0x73FA82F4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT11,   0x73FA82F8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT12,   0x73FA82FC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT13,   0x73FA8300,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT14,   0x73FA8304,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT15,   0x73FA8308,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT16,   0x73FA830C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT17,   0x73FA8310,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT18,   0x73FA8314,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT19,   0x73FA8318,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT20,   0x73FA831C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT21,   0x73FA8320,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT22,   0x73FA8324,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP1_DAT23,   0x73FA8328,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_PIN3,      0x73FA832C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI1_PIN2,      0x73FA8330,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI_GP1,        0x73FA8334,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI_GP2,        0x73FA8338,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI_GP3,        0x73FA833C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI2_PIN4,      0x73FA8340,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI2_PIN2,      0x73FA8344,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI2_PIN3,      0x73FA8348,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI2_DISP_CLK,  0x73FA834C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DI_GP4,        0x73FA8350,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT0,    0x73FA8354,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT1,    0x73FA8358,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT2,    0x73FA835C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT3,    0x73FA8360,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT4,    0x73FA8364,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT5,    0x73FA8368,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT6,    0x73FA836C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT7,    0x73FA8370,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT8,    0x73FA8374,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT9,    0x73FA8378,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT10,   0x73FA837C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT11,   0x73FA8380,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT12,   0x73FA8384,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT13,   0x73FA8388,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT14,   0x73FA838C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_DISP2_DAT15,   0x73FA8390,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CMD,       0x73FA8394,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_CLK,       0x73FA8398,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA0,     0x73FA839C,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA1,     0x73FA83A0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA2,     0x73FA83A4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD1_DATA3,     0x73FA83A8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_0,       0x73FA83AC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_1,       0x73FA83B0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CMD,       0x73FA83B4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_CLK,       0x73FA83B8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA0,     0x73FA83BC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA1,     0x73FA83C0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA2,     0x73FA83C4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_SD2_DATA3,     0x73FA83C8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode2sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_2,       0x73FA83CC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_3,       0x73FA83D0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_PMIC_INT_REQ,  0x73FA83D4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode1_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_4,       0x73FA83D8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_5,       0x73FA83DC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_6,       0x73FA83E0,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_7,       0x73FA83E4,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_8,       0x73FA83E8,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);
__IO_REG32_BIT(IOMUXC_SW_MUX_CTL_PAD_GPIO1_9,       0x73FA83EC,__READ_WRITE ,__iomuxc_sw_mux_ctl_pad_muxmode3sion_bits);

__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D16,       0x73FA83F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D17,       0x73FA83F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D18,       0x73FA83F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D19,       0x73FA83FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D20,       0x73FA8400,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D21,       0x73FA8404,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D22,       0x73FA8408,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D23,       0x73FA840C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D24,       0x73FA8410,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D25,       0x73FA8414,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D26,       0x73FA8418,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D27,       0x73FA841C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D28,       0x73FA8420,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D29,       0x73FA8424,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D30,       0x73FA8428,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_D31,       0x73FA842C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A16,       0x73FA8430,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A17,       0x73FA8434,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A18,       0x73FA8438,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A19,       0x73FA843C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A20,       0x73FA8440,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A21,       0x73FA8444,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A22,       0x73FA8448,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A23,       0x73FA844C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A24,       0x73FA8450,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A25,       0x73FA8454,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A26,       0x73FA8458,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_A27,       0x73FA845C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB0,       0x73FA8460,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB1,       0x73FA8464,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB2,       0x73FA8468,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_EB3,       0x73FA846C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_OE ,       0x73FA8470,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS0,       0x73FA8474,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS1,       0x73FA8478,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS2,       0x73FA847C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS3,       0x73FA8480,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS4,       0x73FA8484,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CS5,       0x73FA8488,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_DTACK,     0x73FA848C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_WAIT,      0x73FA8490,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_LBA,       0x73FA8494,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_BCLK,      0x73FA8498,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_RW,        0x73FA849C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_CRE,       0x73FA84A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_RAS,      0x73FA84A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_CAS,      0x73FA84A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDWE,     0x73FA84AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCKE0,   0x73FA84B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCKE1,   0x73FA84B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDCLK,    0x73FA84B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS0,    0x73FA84BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS1,    0x73FA84C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS2,    0x73FA84C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_SDQS3,    0x73FA84C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_CS0,      0x73FA84CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_CS1,      0x73FA84D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM0,     0x73FA84D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM1,     0x73FA84D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM2,     0x73FA84DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DRAM_DQM3,     0x73FA84E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_WE_B,    0x73FA84E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RE_B,    0x73FA84E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_ALE,     0x73FA84EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_pke_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CLE,     0x73FA84F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_pke_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_WP_B,    0x73FA84F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RB0,     0x73FA84F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RB1,     0x73FA84FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RB2,     0x73FA8500,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RB3,     0x73FA8504,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_SDBA2,     0x73FA8508,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dhppp_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_SDODT1,    0x73FA850C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dhppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_EIM_SDODT0,    0x73FA8510,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_dhppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO_NAND,     0x73FA8514,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_pke_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS0,     0x73FA8518,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS1,     0x73FA851C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS2,     0x73FA8520,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS3,     0x73FA8524,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS4,     0x73FA8528,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS5,     0x73FA852C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS6,     0x73FA8530,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_CS7,     0x73FA8534,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hp_p_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_RDY_INT, 0x73FA8538,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D15,     0x73FA853C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D14,     0x73FA8540,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D13,     0x73FA8544,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D12,     0x73FA8548,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D11,     0x73FA854C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D10,     0x73FA8550,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D9,      0x73FA8554,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D8,      0x73FA8558,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D7,      0x73FA855C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D6,      0x73FA8560,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D5,      0x73FA8564,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D4,      0x73FA8568,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D3,      0x73FA856C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D2,      0x73FA8570,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D1,      0x73FA8574,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_NANDF_D0,      0x73FA8578,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D8,       0x73FA857C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D9,       0x73FA8580,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D10,      0x73FA8584,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D11,      0x73FA8588,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D12,      0x73FA858C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D13,      0x73FA8590,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D14,      0x73FA8594,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D15,      0x73FA8598,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D16,      0x73FA859C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D17,      0x73FA85A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D18,      0x73FA85A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_D19,      0x73FA85A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_VSYNC,    0x73FA85AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_HSYNC,    0x73FA85B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_PIXCLK,   0x73FA85B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI1_MCLK,     0x73FA85B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D12,      0x73FA85BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D13,      0x73FA85C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D14,      0x73FA85C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D15,      0x73FA85C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D16,      0x73FA85CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D17,      0x73FA85D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D18,      0x73FA85D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_D19,      0x73FA85D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_VSYNC,    0x73FA85DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_HSYNC,    0x73FA85E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSI2_PIXCLK,   0x73FA85E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_CLK,      0x73FA85E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_pod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_I2C1_DAT,      0x73FA85EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_pod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_AUD3_BB_TXD,   0x73FA85F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_AUD3_BB_RXD,   0x73FA85F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_AUD3_BB_CK,    0x73FA85F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_AUD3_BB_FS,    0x73FA85FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MOSI,    0x73FA8600,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_pods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_MISO,    0x73FA8604,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS0,     0x73FA8608,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SS1,     0x73FA860C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_RDY,     0x73FA8610,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CSPI1_SCLK,    0x73FA8614,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_pods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_RXD,     0x73FA8618,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_TXD,     0x73FA861C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_RTS,     0x73FA8620,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART1_CTS,     0x73FA8624,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_RXD,     0x73FA8628,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART2_TXD,     0x73FA862C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART3_RXD,     0x73FA8630,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_UART3_TXD,     0x73FA8634,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_OWIRE_LINE,    0x73FA8638,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hp_pods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW0,      0x73FA863C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW1,      0x73FA8640,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW2,      0x73FA8644,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_ROW3,      0x73FA8648,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL0,      0x73FA864C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL1,      0x73FA8650,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL2,      0x73FA8654,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL3,      0x73FA8658,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL4,      0x73FA865C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_KEY_COL5,      0x73FA8660,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TCK,      0x73FA8664,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TMS,      0x73FA8668,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TDI,      0x73FA866C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_TRSTB,    0x73FA8670,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_JTAG_MOD,      0x73FA8674,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_CLK,     0x73FA8678,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DIR,     0x73FA867C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_STP,     0x73FA8680,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_NXT,     0x73FA8684,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA0,   0x73FA8688,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA1,   0x73FA868C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA2,   0x73FA8690,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA3,   0x73FA8694,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA4,   0x73FA8698,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA5,   0x73FA869C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA6,   0x73FA86A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_USBH1_DATA7,   0x73FA86A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN11,     0x73FA86A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN12,     0x73FA86AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN13,     0x73FA86B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_D0_CS,     0x73FA86B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_D1_CS,     0x73FA86B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISPB2_SER_DIN,0x73FA86BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISPB2_SER_DIO,0x73FA86C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISPB2_SER_CLK,0x73FA86C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISPB2_SER_RS, 0x73FA86C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT0,    0x73FA86CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT1,    0x73FA86D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT2,    0x73FA86D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT3,    0x73FA86D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT4,    0x73FA86DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT5,    0x73FA86E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT6,    0x73FA86E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT7,    0x73FA86E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT8,    0x73FA86EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT9,    0x73FA86F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT10,   0x73FA86F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT11,   0x73FA86F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT12,   0x73FA86FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT13,   0x73FA8700,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT14,   0x73FA8704,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT15,   0x73FA8708,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT16,   0x73FA870C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT17,   0x73FA8710,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT18,   0x73FA8714,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT19,   0x73FA8718,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT20,   0x73FA871C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT21,   0x73FA8720,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT22,   0x73FA8724,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP1_DAT23,   0x73FA8728,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN3,      0x73FA872C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_DISP_CLK,  0x73FA8730,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN2,      0x73FA8734,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI1_PIN15,     0x73FA8738,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI_GP1,        0x73FA873C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI_GP2,        0x73FA8740,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI_GP3,        0x73FA8744,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI2_PIN4,      0x73FA8748,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI2_PIN2,      0x73FA874C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pkepue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI2_PIN3,      0x73FA8750,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI2_DISP_CLK,  0x73FA8754,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DI_GP4,        0x73FA8758,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT0,    0x73FA875C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT1,    0x73FA8760,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT2,    0x73FA8764,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT3,    0x73FA8768,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT4,    0x73FA876C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT5,    0x73FA8770,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT6,    0x73FA8774,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT7,    0x73FA8778,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT8,    0x73FA877C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT9,    0x73FA8780,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT10,   0x73FA8784,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT11,   0x73FA8788,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT12,   0x73FA878C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT13,   0x73FA8790,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT14,   0x73FA8794,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_DISP2_DAT15,   0x73FA8798,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CMD,       0x73FA879C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_pppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_CLK,       0x73FA87A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA0,     0x73FA87A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA1,     0x73FA87A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA2,     0x73FA87AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD1_DATA3,     0x73FA87B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_0,       0x73FA87B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_1,       0x73FA87B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CMD,       0x73FA87BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_pppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_CLK,       0x73FA87C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA0,     0x73FA87C4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA1,     0x73FA87C8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA2,     0x73FA87CC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_hpppod_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_SD2_DATA3,     0x73FA87D0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_ppp_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_2,       0x73FA87D4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_3,       0x73FA87D8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_RESET_IN_B,    0x73FA87DC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_POR_B,         0x73FA87E0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE1,    0x73FA87E4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_BOOT_MODE0,    0x73FA87E8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_RDY,      0x73FA87EC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_puepus_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CKIL,          0x73FA87F0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_puepus_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_STBY_REQ, 0x73FA87F4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_ON_REQ,   0x73FA87F8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pppo_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_PMIC_INT_REQ,  0x73FA87FC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_CLK_SS,        0x73FA8800,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_puepus_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_4,       0x73FA8804,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_5,       0x73FA8808,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_6,       0x73FA880C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_7,       0x73FA8810,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_8,       0x73FA8814,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hppp_ds_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_PAD_GPIO1_9,       0x73FA8818,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hpppods_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_CSI2_PKE0,     0x73FA881C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRPKS,        0x73FA8820,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_SR1,       0x73FA8824,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DISP2_PK0,     0x73FA8828,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_B4,       0x73FA882C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_INDDR,         0x73FA8830,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_SR2,       0x73FA8834,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_PKEDDR,        0x73FA8838,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDR_A0,        0x73FA883C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EMI_PKE0,      0x73FA8840,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_SR3,       0x73FA8844,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDR_A1,        0x73FA8848,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRAPUS,       0x73FA884C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pus_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_SR4,       0x73FA8850,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EMI_SR5,       0x73FA8854,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EMI_SR6,       0x73FA8858,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_HYSDDR0,       0x73FA885C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_CSI1_PKE0,     0x73FA8860,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_HYSDDR1,       0x73FA8864,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DISP1_PKE0,    0x73FA8868,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_HYSDDR2,       0x73FA886C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_HVDDR,         0x73FA8870,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hve_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_HYSDDR3,       0x73FA8874,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_hys_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_SR_B0,    0x73FA8878,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRAPKS,       0x73FA887C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pue_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_SR_B1,    0x73FA8880,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDRPUS,        0x73FA8884,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pus_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_DS1,       0x73FA8888,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_SR_B2,    0x73FA888C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_PKEADDR,       0x73FA8890,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_pke_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_DS2,       0x73FA8894,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_DS3,       0x73FA8898,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_SR_B4,    0x73FA889C,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_INMODE1,       0x73FA88A0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_ddr_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_B0,       0x73FA88A4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EIM_DS4,       0x73FA88A8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_B1,       0x73FA88AC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDR_SR_A0,     0x73FA88B0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EMI_DS5,       0x73FA88B4,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DRAM_B2,       0x73FA88B8,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_DDR_SR_A1,     0x73FA88BC,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_s_bits);
__IO_REG32_BIT(IOMUXC_SW_PAD_CTL_GRP_EMI_DS6,       0x73FA88C0,__READ_WRITE ,__iomuxc_sw_pad_ctl_pad_d_bits);

__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DA_AMX_SELECT_INPUT,    0x73FA88C4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_DB_AMX_SELECT_INPUT,    0x73FA88C8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXCLK_AMX_SELECT_INPUT, 0x73FA88CC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P4_INPUT_TXFS_AMX_SELECT_INPUT,  0x73FA88D0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_DA_AMX_SELECT_INPUT,    0x73FA88D4,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_DB_AMX_SELECT_INPUT,    0x73FA88D8,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXCLK_AMX_SELECT_INPUT, 0x73FA88DC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_RXFS_AMX_SELECT_INPUT,  0x73FA88E0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_TXCLK_AMX_SELECT_INPUT, 0x73FA88E4,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P5_INPUT_TXFS_AMX_SELECT_INPUT,  0x73FA88E8,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_DA_AMX_SELECT_INPUT,    0x73FA88EC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_DB_AMX_SELECT_INPUT,    0x73FA88F0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_RXCLK_AMX_SELECT_INPUT, 0x73FA88F4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_RXFS_AMX_SELECT_INPUT,  0x73FA88F8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_TXCLK_AMX_SELECT_INPUT, 0x73FA88FC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_AUDMUX_P6_INPUT_TXFS_AMX_SELECT_INPUT,  0x73FA8900,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_DI0_CLK_SELECT_INPUT,           0x73FA8904,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CCM_IPP_DI1_CLK_SELECT_INPUT,           0x73FA8908,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL1_BYPASS_CLK_SELECT_INPUT,       0x73FA890C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL2_BYPASS_CLK_SELECT_INPUT,       0x73FA8910,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_CSPI_CLK_IN_SELECT_INPUT,      0x73FA8914,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_MISO_SELECT_INPUT,         0x73FA8918,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_MOSI_SELECT_INPUT,         0x73FA891C,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS1_B_SELECT_INPUT,        0x73FA8920,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS2_B_SELECT_INPUT,        0x73FA8924,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_CSPI_IPP_IND_SS3_B_SELECT_INPUT,        0x73FA8928,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_DPLLIP1_L1T_TOG_EN_SELECT_INPUT,        0x73FA892C,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_SS_B_1_SELECT_INPUT,     0x73FA8930,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_ECSPI2_IPP_IND_SS_B_3_SELECT_INPUT,     0x73FA8934,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_EMI_IPP_IND_RDY_INT_SELECT_INPUT,       0x73FA8938,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT0_IN_SELECT_INPUT,        0x73FA893C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT1_IN_SELECT_INPUT,        0x73FA8940,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT2_IN_SELECT_INPUT,        0x73FA8944,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_ESDHC3_IPP_DAT3_IN_SELECT_INPUT,        0x73FA8948,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_COL_SELECT_INPUT,               0x73FA894C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_CRS_SELECT_INPUT,               0x73FA8950,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_MDI_SELECT_INPUT,               0x73FA8954,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_0_SELECT_INPUT,           0x73FA8958,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_1_SELECT_INPUT,           0x73FA895C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_2_SELECT_INPUT,           0x73FA8960,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RDATA_3_SELECT_INPUT,           0x73FA8964,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_CLK_SELECT_INPUT,            0x73FA8968,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_DV_SELECT_INPUT,             0x73FA896C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_RX_ER_SELECT_INPUT,             0x73FA8970,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_FEC_FEC_TX_CLK_SELECT_INPUT,            0x73FA8974,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_1_SELECT_INPUT,      0x73FA8978,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_2_SELECT_INPUT,      0x73FA897C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_3_SELECT_INPUT,      0x73FA8980,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_4_SELECT_INPUT,      0x73FA8984,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_5_SELECT_INPUT,      0x73FA8988,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_6_SELECT_INPUT,      0x73FA898C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_7_SELECT_INPUT,      0x73FA8990,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_8_SELECT_INPUT,      0x73FA8994,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_GPIO3_IPP_IND_G_IN_12_SELECT_INPUT,     0x73FA8998,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_HSC_MIPI_MIX_IPP_IND_SENS1_DATA_EN_SELECT_INPUT, 0x73FA899C,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_HSC_MIPI_MIX_IPP_IND_SENS2_DATA_EN_SELECT_INPUT, 0x73FA89A0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_HSC_MIPI_MIX_PAR0_VSYNC_SELECT_INPUT,            0x73FA89A4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_HSC_MIPI_MIX_PAR1_DI_WAIT_SELECT_INPUT,          0x73FA89A8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_HSC_MIPI_MIX_PAR_SISG_TRIG_SELECT_INPUT,         0x73FA89AC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_I2C1_IPP_SCL_IN_SELECT_INPUT,          0x73FA89B0,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_I2C1_IPP_SDA_IN_SELECT_INPUT,          0x73FA89B4,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SCL_IN_SELECT_INPUT,          0x73FA89B8,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_I2C2_IPP_SDA_IN_SELECT_INPUT,          0x73FA89BC,__READ_WRITE ,__iomuxc_daisy2_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_DI_0_IND_DISPB_SD_D_SELECT_INPUT,        0x73FA89C0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_IPU_IPP_DI_1_IND_DISPB_SD_D_SELECT_INPUT,        0x73FA89C4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_6_SELECT_INPUT,        0x73FA89C8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_COL_7_SELECT_INPUT,        0x73FA89CC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_4_SELECT_INPUT,        0x73FA89D0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_5_SELECT_INPUT,        0x73FA89D4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_6_SELECT_INPUT,        0x73FA89D8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_KPP_IPP_IND_ROW_7_SELECT_INPUT,        0x73FA89DC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_UART1_IPP_UART_RTS_B_SELECT_INPUT,     0x73FA89E0,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_UART1_IPP_UART_RXD_MUX_SELECT_INPUT,   0x73FA89E4,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_UART2_IPP_UART_RTS_B_SELECT_INPUT,     0x73FA89E8,__READ_WRITE ,__iomuxc_daisy3_bits);
__IO_REG32_BIT(IOMUXC_UART2_IPP_UART_RXD_MUX_SELECT_INPUT,   0x73FA89EC,__READ_WRITE ,__iomuxc_daisy3_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RXD_B_SELECT_INPUT,     0x73FA89F0,__READ_WRITE ,__iomuxc_daisy3_bits);
__IO_REG32_BIT(IOMUXC_UART3_IPP_UART_RXD_MUX_SELECT_INPUT,   0x73FA89F4,__READ_WRITE ,__iomuxc_daisy4_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_CLK_SELECT_INPUT,   0x73FA89F8,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_0_SELECT_INPUT,0x73FA89FC,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_1_SELECT_INPUT,0x73FA8A00,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_2_SELECT_INPUT,0x73FA8A04,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_3_SELECT_INPUT,0x73FA8A08,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_4_SELECT_INPUT,0x73FA8A0C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_5_SELECT_INPUT,0x73FA8A10,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_6_SELECT_INPUT,0x73FA8A14,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DATA_7_SELECT_INPUT,0x73FA8A18,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_DIR_SELECT_INPUT,   0x73FA8A1C,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_NXT_SELECT_INPUT,   0x73FA8A20,__READ_WRITE ,__iomuxc_daisy_bits);
__IO_REG32_BIT(IOMUXC_USBOH3_IPP_IND_UH3_STP_SELECT_INPUT,   0x73FA8A24,__READ_WRITE ,__iomuxc_daisy_bits);

/***************************************************************************
 **
 **  IPU (IPUEX)
 **
 ***************************************************************************/
__IO_REG32_BIT(IPU_CONF,                      0x4E000000,__READ_WRITE ,__ipu_conf_bits);
__IO_REG32_BIT(IPU_SISG_CTRL0,                0x4E000004,__READ_WRITE ,__ipu_sisg_ctrl0_bits);
__IO_REG32_BIT(IPU_SISG_CTRL1,                0x4E000008,__READ_WRITE ,__ipu_sisg_ctrl1_bits);
__IO_REG32_BIT(IPU_SISG_SET_1,                0x4E00000C,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_2,                0x4E000010,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_3,                0x4E000014,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_4,                0x4E000018,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_5,                0x4E00001C,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_SET_6,                0x4E000020,__READ_WRITE ,__ipu_sisg_set_bits);
__IO_REG32_BIT(IPU_SISG_CLR_1,                0x4E000024,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_2,                0x4E000028,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_3,                0x4E00002C,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_4,                0x4E000030,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_5,                0x4E000034,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_SISG_CLR_6,                0x4E000038,__READ_WRITE ,__ipu_sisg_clr_bits);
__IO_REG32_BIT(IPU_INT_CTRL_1,                0x4E00003C,__READ_WRITE ,__ipu_int_ctrl_1_bits);
__IO_REG32_BIT(IPU_INT_CTRL_2,                0x4E000040,__READ_WRITE ,__ipu_int_ctrl_2_bits);
__IO_REG32_BIT(IPU_INT_CTRL_3,                0x4E000044,__READ_WRITE ,__ipu_int_ctrl_3_bits);
__IO_REG32_BIT(IPU_INT_CTRL_4,                0x4E000048,__READ_WRITE ,__ipu_int_ctrl_4_bits);
__IO_REG32_BIT(IPU_INT_CTRL_5,                0x4E00004C,__READ_WRITE ,__ipu_int_ctrl_5_bits);
__IO_REG32_BIT(IPU_INT_CTRL_6,                0x4E000050,__READ_WRITE ,__ipu_int_ctrl_6_bits);
__IO_REG32_BIT(IPU_INT_CTRL_7,                0x4E000054,__READ_WRITE ,__ipu_int_ctrl_7_bits);
__IO_REG32_BIT(IPU_INT_CTRL_8,                0x4E000058,__READ_WRITE ,__ipu_int_ctrl_8_bits);
__IO_REG32_BIT(IPU_INT_CTRL_9,                0x4E00005C,__READ_WRITE ,__ipu_int_ctrl_9_bits);
__IO_REG32_BIT(IPU_INT_CTRL_10,               0x4E000060,__READ_WRITE ,__ipu_int_ctrl_10_bits);
__IO_REG32_BIT(IPU_INT_CTRL_11,               0x4E000064,__READ_WRITE ,__ipu_int_ctrl_11_bits);
__IO_REG32_BIT(IPU_INT_CTRL_12,               0x4E000068,__READ_WRITE ,__ipu_int_ctrl_12_bits);
__IO_REG32_BIT(IPU_INT_CTRL_13,               0x4E00006C,__READ_WRITE ,__ipu_int_ctrl_13_bits);
__IO_REG32_BIT(IPU_INT_CTRL_14,               0x4E000070,__READ_WRITE ,__ipu_int_ctrl_14_bits);
__IO_REG32_BIT(IPU_INT_CTRL_15,               0x4E000074,__READ_WRITE ,__ipu_int_ctrl_15_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_1,              0x4E000078,__READ_WRITE ,__ipu_sdma_event_1_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_2,              0x4E00007C,__READ_WRITE ,__ipu_sdma_event_2_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_3,              0x4E000080,__READ_WRITE ,__ipu_sdma_event_3_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_4,              0x4E000084,__READ_WRITE ,__ipu_sdma_event_4_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_7,              0x4E000088,__READ_WRITE ,__ipu_sdma_event_7_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_8,              0x4E00008C,__READ_WRITE ,__ipu_sdma_event_8_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_11,             0x4E000090,__READ_WRITE ,__ipu_sdma_event_11_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_12,             0x4E000094,__READ_WRITE ,__ipu_sdma_event_12_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_13,             0x4E000098,__READ_WRITE ,__ipu_sdma_event_13_bits);
__IO_REG32_BIT(IPU_SDMA_EVENT_14,             0x4E00009C,__READ_WRITE ,__ipu_sdma_event_14_bits);
__IO_REG32_BIT(IPU_SRM_PRI1,                  0x4E0000A0,__READ_WRITE ,__ipu_srm_pri1_bits);
__IO_REG32_BIT(IPU_SRM_PRI2,                  0x4E0000A4,__READ_WRITE ,__ipu_srm_pri2_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW1,             0x4E0000A8,__READ_WRITE ,__ipu_fs_proc_flow1_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW2,             0x4E0000AC,__READ_WRITE ,__ipu_fs_proc_flow2_bits);
__IO_REG32_BIT(IPU_FS_PROC_FLOW3,             0x4E0000B0,__READ_WRITE ,__ipu_fs_proc_flow3_bits);
__IO_REG32_BIT(IPU_FS_DISP_FLOW1,             0x4E0000B4,__READ_WRITE ,__ipu_fs_disp_flow1_bits);
__IO_REG32_BIT(IPU_FS_DISP_FLOW2,             0x4E0000B8,__READ_WRITE ,__ipu_fs_disp_flow2_bits);
__IO_REG32_BIT(IPU_SKIP,                      0x4E0000BC,__READ_WRITE ,__ipu_skip_bits);
__IO_REG32_BIT(IPU_SNOOP,                     0x4E0000D8,__READ_WRITE ,__ipu_snoop_bits);
__IO_REG32_BIT(IPU_MEM_RST,                   0x4E0000DC,__READ_WRITE ,__ipu_mem_rst_bits);
__IO_REG32_BIT(IPU_PM,                        0x4E0000E0,__READ_WRITE ,__ipu_pm_bits);
__IO_REG32_BIT(IPU_GPR,                       0x4E0000E4,__READ_WRITE ,__ipu_gpr_bits);
__IO_REG32_BIT(IPU_CH_DB_MODE_SEL0,           0x4E000150,__READ_WRITE ,__ipu_ch_db_mode_sel0_bits);
__IO_REG32_BIT(IPU_CH_DB_MODE_SEL1,           0x4E000154,__READ_WRITE ,__ipu_ch_db_mode_sel1_bits);
__IO_REG32_BIT(IPU_ALT_CH_DB_MODE_SEL0,       0x4E000168,__READ_WRITE ,__ipu_alt_ch_db_mode_sel0_bits);
__IO_REG32_BIT(IPU_ALT_CH_DB_MODE_SEL1,       0x4E00016C,__READ_WRITE ,__ipu_alt_ch_db_mode_sel1_bits);
__IO_REG32_BIT(IPU_CH_TRB_MODE_SEL0,          0x4E000178,__READ_WRITE ,__ipu_ch_trb_mode_sel0_bits);
__IO_REG32(    IPU_CH_TRB_MODE_SEL1,          0x4E00017C,__READ_WRITE );
__IO_REG32_BIT(IPU_INT_STAT_1,                0x4E000200,__READ_WRITE ,__ipu_int_stat_1_bits);
__IO_REG32_BIT(IPU_INT_STAT_2,                0x4E000204,__READ_WRITE ,__ipu_int_stat_2_bits);
__IO_REG32_BIT(IPU_INT_STAT_3,                0x4E000208,__READ_WRITE ,__ipu_int_stat_3_bits);
__IO_REG32_BIT(IPU_INT_STAT_4,                0x4E00020C,__READ_WRITE ,__ipu_int_stat_4_bits);
__IO_REG32_BIT(IPU_INT_STAT_5,                0x4E000210,__READ_WRITE ,__ipu_int_stat_5_bits);
__IO_REG32_BIT(IPU_INT_STAT_6,                0x4E000214,__READ_WRITE ,__ipu_int_stat_6_bits);
__IO_REG32_BIT(IPU_INT_STAT_7,                0x4E000218,__READ_WRITE ,__ipu_int_stat_7_bits);
__IO_REG32_BIT(IPU_INT_STAT_8,                0x4E00021C,__READ_WRITE ,__ipu_int_stat_8_bits);
__IO_REG32_BIT(IPU_INT_STAT_9,                0x4E000220,__READ_WRITE ,__ipu_int_stat_9_bits);
__IO_REG32_BIT(IPU_INT_STAT_10,               0x4E000224,__READ_WRITE ,__ipu_int_stat_10_bits);
__IO_REG32_BIT(IPU_INT_STAT_11,               0x4E000228,__READ_WRITE ,__ipu_int_stat_11_bits);
__IO_REG32_BIT(IPU_INT_STAT_12,               0x4E00022C,__READ_WRITE ,__ipu_int_stat_12_bits);
__IO_REG32_BIT(IPU_INT_STAT_13,               0x4E000230,__READ_WRITE ,__ipu_int_stat_13_bits);
__IO_REG32_BIT(IPU_INT_STAT_14,               0x4E000234,__READ_WRITE ,__ipu_int_stat_14_bits);
__IO_REG32_BIT(IPU_INT_STAT_15,               0x4E000238,__READ_WRITE ,__ipu_int_stat_15_bits);
__IO_REG32_BIT(IPU_CUR_BUF_0,                 0x4E00023C,__READ       ,__ipu_cur_buf_0_bits);
__IO_REG32_BIT(IPU_CUR_BUF_1,                 0x4E000240,__READ       ,__ipu_cur_buf_1_bits);
__IO_REG32_BIT(IPU_ALT_CUR_BUF_0,             0x4E000244,__READ       ,__ipu_alt_cur_buf_0_bits);
__IO_REG32_BIT(IPU_ALT_CUR_BUF_1,             0x4E000248,__READ       ,__ipu_alt_cur_buf_1_bits);
__IO_REG32_BIT(IPU_SRM_STAT,                  0x4E00024C,__READ       ,__ipu_srm_stat_bits);
__IO_REG32_BIT(IPU_PROC_TASKS_STAT,           0x4E000250,__READ       ,__ipu_proc_tasks_stat_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_0,          0x4E000258,__READ       ,__ipu_triple_cur_buf_0_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_1,          0x4E00025C,__READ       ,__ipu_triple_cur_buf_1_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_2,          0x4E000260,__READ       ,__ipu_triple_cur_buf_2_bits);
__IO_REG32_BIT(IPU_TRIPLE_CUR_BUF_3,          0x4E000264,__READ       ,__ipu_triple_cur_buf_3_bits);
__IO_REG32_BIT(IPU_CH_BUF0_RDY0,              0x4E000268,__READ_WRITE ,__ipu_ch_buf0_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF0_RDY1,              0x4E00026C,__READ_WRITE ,__ipu_ch_buf0_rdy1_bits);
__IO_REG32_BIT(IPU_CH_BUF1_RDY0,              0x4E000270,__READ_WRITE ,__ipu_ch_buf1_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF1_RDY1,              0x4E000274,__READ_WRITE ,__ipu_ch_buf1_rdy1_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF0_RDY0,          0x4E000278,__READ_WRITE ,__ipu_alt_ch_buf0_rdy0_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF0_RDY1,          0x4E00027C,__READ_WRITE ,__ipu_alt_ch_buf0_rdy1_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF1_RDY0,          0x4E000280,__READ_WRITE ,__ipu_alt_ch_buf1_rdy0_bits);
__IO_REG32_BIT(IPU_ALT_CH_BUF1_RDY1,          0x4E000284,__READ_WRITE ,__ipu_alt_ch_buf1_rdy1_bits);
__IO_REG32_BIT(IPU_CH_BUF2_RDY0,              0x4E000288,__READ_WRITE ,__ipu_ch_buf2_rdy0_bits);
__IO_REG32_BIT(IPU_CH_BUF2_RDY1,              0x4E00028C,__READ_WRITE ,__ipu_ch_buf2_rdy1_bits);
__IO_REG32_BIT(IPU_IDMAC_CONF,                0x4E008000,__READ_WRITE ,__ipu_idmac_conf_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_EN_1,             0x4E008004,__READ_WRITE ,__ipu_idmac_ch_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_EN_2,             0x4E008008,__READ_WRITE ,__ipu_idmac_ch_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SEP_ALPHA,           0x4E00800C,__READ_WRITE ,__ipu_idmac_sep_alpha_bits);
__IO_REG32_BIT(IPU_IDMAC_ALT_SEP_ALPHA,       0x4E008010,__READ_WRITE ,__ipu_idmac_alt_sep_alpha_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_PRI_1,            0x4E008014,__READ_WRITE ,__ipu_idmac_ch_pri_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_PRI_2,            0x4E008018,__READ_WRITE ,__ipu_idmac_ch_pri_2_bits);
__IO_REG32_BIT(IPU_IDMAC_WM_EN_1,             0x4E00801C,__READ_WRITE ,__ipu_idmac_wm_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_WM_EN_2,             0x4E008020,__READ_WRITE ,__ipu_idmac_wm_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_LOCK_EN_1,           0x4E008024,__READ_WRITE ,__ipu_idmac_lock_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_LOCK_EN_2,           0x4E008028,__READ_WRITE ,__ipu_idmac_lock_en_2_bits);
__IO_REG32(    IPU_IDMAC_SUB_ADDR_0,          0x4E00802C,__READ_WRITE );
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_1,          0x4E008030,__READ_WRITE ,__ipu_idmac_sub_addr_1_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_2,          0x4E008034,__READ_WRITE ,__ipu_idmac_sub_addr_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_3,          0x4E008038,__READ_WRITE ,__ipu_idmac_sub_addr_3_bits);
__IO_REG32_BIT(IPU_IDMAC_SUB_ADDR_4,          0x4E00803C,__READ_WRITE ,__ipu_idmac_sub_addr_4_bits);
__IO_REG32_BIT(IPU_IDMAC_BNDM_EN_1,           0x4E008040,__READ_WRITE ,__ipu_idmac_bndm_en_1_bits);
__IO_REG32_BIT(IPU_IDMAC_BNDM_EN_2,           0x4E008044,__READ_WRITE ,__ipu_idmac_bndm_en_2_bits);
__IO_REG32_BIT(IPU_IDMAC_SC_CORD,             0x4E008048,__READ_WRITE ,__ipu_idmac_sc_cord_bits);
__IO_REG32_BIT(IPU_IDMAC_SC_CORD_1,           0x4E00804C,__READ_WRITE ,__ipu_idmac_sc_cord_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_BUSY_1,           0x4E008100,__READ       ,__ipu_idmac_ch_busy_1_bits);
__IO_REG32_BIT(IPU_IDMAC_CH_BUSY_2,           0x4E008104,__READ       ,__ipu_idmac_ch_busy_2_bits);
__IO_REG32_BIT(IPU_IC_CONF,                   0x4E020000,__READ_WRITE ,__ipu_ic_conf_bits);
__IO_REG32_BIT(IPU_IC_PRP_ENC_RSC,            0x4E020004,__READ_WRITE ,__ipu_ic_prp_enc_rsc_bits);
__IO_REG32_BIT(IPU_IC_PRP_VF_RSC,             0x4E020008,__READ_WRITE ,__ipu_ic_prp_vf_rsc_bits);
__IO_REG32_BIT(IPU_IC_PP_RSC,                 0x4E02000C,__READ_WRITE ,__ipu_ic_pp_rsc_bits);
__IO_REG32_BIT(IPU_IC_CMBP_1,                 0x4E020010,__READ_WRITE ,__ipu_ic_cmbp_1_bits);
__IO_REG32_BIT(IPU_IC_CMBP_2,                 0x4E020014,__READ_WRITE ,__ipu_ic_cmbp_2_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_1,                0x4E020018,__READ_WRITE ,__ipu_ic_idmac_1_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_2,                0x4E02001C,__READ_WRITE ,__ipu_ic_idmac_2_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_3,                0x4E020020,__READ_WRITE ,__ipu_ic_idmac_3_bits);
__IO_REG32_BIT(IPU_IC_IDMAC_4,                0x4E020024,__READ_WRITE ,__ipu_ic_idmac_4_bits);
__IO_REG32_BIT(IPU_CSI0_SENS_CONF,            0x4E030000,__READ_WRITE ,__ipu_csi0_sens_conf_bits);
__IO_REG32_BIT(IPU_CSI0_SENS_FRM_SIZE,        0x4E030004,__READ_WRITE ,__ipu_csi0_sens_frm_size_bits);
__IO_REG32_BIT(IPU_CSI0_ACT_FRM_SIZE,         0x4E030008,__READ_WRITE ,__ipu_csi0_act_frm_size_bits);
__IO_REG32_BIT(IPU_CSI0_OUT_FRM_CTRL,         0x4E03000C,__READ_WRITE ,__ipu_csi0_out_frm_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_TST_CTRL,             0x4E030010,__READ_WRITE ,__ipu_csi0_tst_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_1,          0x4E030014,__READ_WRITE ,__ipu_csi0_ccir_code_1_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_2,          0x4E030018,__READ_WRITE ,__ipu_csi0_ccir_code_2_bits);
__IO_REG32_BIT(IPU_CSI0_CCIR_CODE_3,          0x4E03001C,__READ_WRITE ,__ipu_csi0_ccir_code_3_bits);
__IO_REG32_BIT(IPU_CSI0_DI,                   0x4E030020,__READ_WRITE ,__ipu_csi0_di_bits);
__IO_REG32_BIT(IPU_CSI0_SKIP,                 0x4E030024,__READ_WRITE ,__ipu_csi0_skip_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_CTRL,             0x4E030028,__READ_WRITE ,__ipu_csi0_cpd_ctrl_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_0,             0x4E03002C,__READ_WRITE ,__ipu_csi0_cpd_rc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_1,             0x4E030030,__READ_WRITE ,__ipu_csi0_cpd_rc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_2,             0x4E030034,__READ_WRITE ,__ipu_csi0_cpd_rc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_3,             0x4E030038,__READ_WRITE ,__ipu_csi0_cpd_rc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_4,             0x4E03003C,__READ_WRITE ,__ipu_csi0_cpd_rc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_5,             0x4E030040,__READ_WRITE ,__ipu_csi0_cpd_rc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_6,             0x4E030044,__READ_WRITE ,__ipu_csi0_cpd_rc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RC_7,             0x4E030048,__READ_WRITE ,__ipu_csi0_cpd_rc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_0,             0x4E03004C,__READ_WRITE ,__ipu_csi0_cpd_rs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_1,             0x4E030050,__READ_WRITE ,__ipu_csi0_cpd_rs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_2,             0x4E030054,__READ_WRITE ,__ipu_csi0_cpd_rs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_RS_3,             0x4E030058,__READ_WRITE ,__ipu_csi0_cpd_rs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_0,            0x4E03005C,__READ_WRITE ,__ipu_csi0_cpd_grc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_1,            0x4E030060,__READ_WRITE ,__ipu_csi0_cpd_grc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_2,            0x4E030064,__READ_WRITE ,__ipu_csi0_cpd_grc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_3,            0x4E030068,__READ_WRITE ,__ipu_csi0_cpd_grc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_4,            0x4E03006C,__READ_WRITE ,__ipu_csi0_cpd_grc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_5,            0x4E030070,__READ_WRITE ,__ipu_csi0_cpd_grc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_6,            0x4E030074,__READ_WRITE ,__ipu_csi0_cpd_grc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRC_7,            0x4E030078,__READ_WRITE ,__ipu_csi0_cpd_grc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_0,            0x4E03007C,__READ_WRITE ,__ipu_csi0_cpd_grs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_1,            0x4E030080,__READ_WRITE ,__ipu_csi0_cpd_grs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_2,            0x4E030084,__READ_WRITE ,__ipu_csi0_cpd_grs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GRS_3,            0x4E030088,__READ_WRITE ,__ipu_csi0_cpd_grs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_0,            0x4E03008C,__READ_WRITE ,__ipu_csi0_cpd_gbc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_1,            0x4E030090,__READ_WRITE ,__ipu_csi0_cpd_gbc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_2,            0x4E030094,__READ_WRITE ,__ipu_csi0_cpd_gbc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_3,            0x4E030098,__READ_WRITE ,__ipu_csi0_cpd_gbc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_4,            0x4E03009C,__READ_WRITE ,__ipu_csi0_cpd_gbc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_5,            0x4E0300A0,__READ_WRITE ,__ipu_csi0_cpd_gbc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_6,            0x4E0300A4,__READ_WRITE ,__ipu_csi0_cpd_gbc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBC_7,            0x4E0300A8,__READ_WRITE ,__ipu_csi0_cpd_gbc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_0,            0x4E0300AC,__READ_WRITE ,__ipu_csi0_cpd_gbs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_1,            0x4E0300B0,__READ_WRITE ,__ipu_csi0_cpd_gbs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_2,            0x4E0300B4,__READ_WRITE ,__ipu_csi0_cpd_gbs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_GBS_3,            0x4E0300B8,__READ_WRITE ,__ipu_csi0_cpd_gbs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_0,             0x4E0300BC,__READ_WRITE ,__ipu_csi0_cpd_bc_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_1,             0x4E0300C0,__READ_WRITE ,__ipu_csi0_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_2,             0x4E0300C4,__READ_WRITE ,__ipu_csi0_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_3,             0x4E0300C8,__READ_WRITE ,__ipu_csi0_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_4,             0x4E0300CC,__READ_WRITE ,__ipu_csi0_cpd_bc_4_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_5,             0x4E0300D0,__READ_WRITE ,__ipu_csi0_cpd_bc_5_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_6,             0x4E0300D4,__READ_WRITE ,__ipu_csi0_cpd_bc_6_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BC_7,             0x4E0300D8,__READ_WRITE ,__ipu_csi0_cpd_bc_7_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_0,             0x4E0300DC,__READ_WRITE ,__ipu_csi0_cpd_bs_0_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_1,             0x4E0300E0,__READ_WRITE ,__ipu_csi0_cpd_bs_1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_2,             0x4E0300E4,__READ_WRITE ,__ipu_csi0_cpd_bs_2_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_BS_3,             0x4E0300E8,__READ_WRITE ,__ipu_csi0_cpd_bs_3_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_OFFSET1,          0x4E0300EC,__READ_WRITE ,__ipu_csi0_cpd_offset1_bits);
__IO_REG32_BIT(IPU_CSI0_CPD_OFFSET2,          0x4E0300F0,__READ_WRITE ,__ipu_csi0_cpd_offset2_bits);
__IO_REG32_BIT(IPU_CSI1_SENS_CONF,            0x4E038000,__READ_WRITE ,__ipu_csi1_sens_conf_bits);
__IO_REG32_BIT(IPU_CSI1_SENS_FRM_SIZE,        0x4E038004,__READ_WRITE ,__ipu_csi1_sens_frm_size_bits);
__IO_REG32_BIT(IPU_CSI1_ACT_FRM_SIZE,         0x4E038008,__READ_WRITE ,__ipu_csi1_act_frm_size_bits);
__IO_REG32_BIT(IPU_CSI1_OUT_FRM_CTRL,         0x4E03800C,__READ_WRITE ,__ipu_csi1_out_frm_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_TST_CTRL,             0x4E038010,__READ_WRITE ,__ipu_csi1_tst_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_1,          0x4E038014,__READ_WRITE ,__ipu_csi1_ccir_code_1_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_2,          0x4E038018,__READ_WRITE ,__ipu_csi1_ccir_code_2_bits);
__IO_REG32_BIT(IPU_CSI1_CCIR_CODE_3,          0x4E03801C,__READ_WRITE ,__ipu_csi1_ccir_code_3_bits);
__IO_REG32_BIT(IPU_CSI1_DI,                   0x4E038020,__READ_WRITE ,__ipu_csi1_di_bits);
__IO_REG32_BIT(IPU_CSI1_SKIP,                 0x4E038024,__READ_WRITE ,__ipu_csi1_skip_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_CTRL,             0x4E038028,__READ_WRITE ,__ipu_csi1_cpd_ctrl_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_0,             0x4E03802C,__READ_WRITE ,__ipu_csi1_cpd_rc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_1,             0x4E038030,__READ_WRITE ,__ipu_csi1_cpd_rc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_2,             0x4E038034,__READ_WRITE ,__ipu_csi1_cpd_rc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_3,             0x4E038038,__READ_WRITE ,__ipu_csi1_cpd_rc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_4,             0x4E03803C,__READ_WRITE ,__ipu_csi1_cpd_rc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_5,             0x4E038040,__READ_WRITE ,__ipu_csi1_cpd_rc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_6,             0x4E038044,__READ_WRITE ,__ipu_csi1_cpd_rc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RC_7,             0x4E038048,__READ_WRITE ,__ipu_csi1_cpd_rc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_0,             0x4E03804C,__READ_WRITE ,__ipu_csi1_cpd_rs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_1,             0x4E038050,__READ_WRITE ,__ipu_csi1_cpd_rs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_2,             0x4E038054,__READ_WRITE ,__ipu_csi1_cpd_rs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_RS_3,             0x4E038058,__READ_WRITE ,__ipu_csi1_cpd_rs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_0,            0x4E03805C,__READ_WRITE ,__ipu_csi1_cpd_grc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_1,            0x4E038060,__READ_WRITE ,__ipu_csi1_cpd_grc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_2,            0x4E038064,__READ_WRITE ,__ipu_csi1_cpd_grc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_3,            0x4E038068,__READ_WRITE ,__ipu_csi1_cpd_grc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_4,            0x4E03806C,__READ_WRITE ,__ipu_csi1_cpd_grc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_5,            0x4E038070,__READ_WRITE ,__ipu_csi1_cpd_grc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_6,            0x4E038074,__READ_WRITE ,__ipu_csi1_cpd_grc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRC_7,            0x4E038078,__READ_WRITE ,__ipu_csi1_cpd_grc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_0,            0x4E03807C,__READ_WRITE ,__ipu_csi1_cpd_grs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_1,            0x4E038080,__READ_WRITE ,__ipu_csi1_cpd_grs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_2,            0x4E038084,__READ_WRITE ,__ipu_csi1_cpd_grs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GRS_3,            0x4E038088,__READ_WRITE ,__ipu_csi1_cpd_grs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_0,            0x4E03808C,__READ_WRITE ,__ipu_csi1_cpd_gbc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_1,            0x4E038090,__READ_WRITE ,__ipu_csi1_cpd_gbc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_2,            0x4E038094,__READ_WRITE ,__ipu_csi1_cpd_gbc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_3,            0x4E038098,__READ_WRITE ,__ipu_csi1_cpd_gbc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_4,            0x4E03809C,__READ_WRITE ,__ipu_csi1_cpd_gbc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_5,            0x4E0380A0,__READ_WRITE ,__ipu_csi1_cpd_gbc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_6,            0x4E0380A4,__READ_WRITE ,__ipu_csi1_cpd_gbc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBC_7,            0x4E0380A8,__READ_WRITE ,__ipu_csi1_cpd_gbc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_0,            0x4E0380AC,__READ_WRITE ,__ipu_csi1_cpd_gbs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_1,            0x4E0380B0,__READ_WRITE ,__ipu_csi1_cpd_gbs_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_2,            0x4E0380B4,__READ_WRITE ,__ipu_csi1_cpd_gbs_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_GBS_3,            0x4E0380B8,__READ_WRITE ,__ipu_csi1_cpd_gbs_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_0,             0x4E0380BC,__READ_WRITE ,__ipu_csi1_cpd_bc_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_1,             0x4E0380C0,__READ_WRITE ,__ipu_csi1_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_2,             0x4E0380C4,__READ_WRITE ,__ipu_csi1_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_3,             0x4E0380C8,__READ_WRITE ,__ipu_csi1_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_4,             0x4E0380CC,__READ_WRITE ,__ipu_csi1_cpd_bc_4_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_5,             0x4E0380D0,__READ_WRITE ,__ipu_csi1_cpd_bc_5_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_6,             0x4E0380D4,__READ_WRITE ,__ipu_csi1_cpd_bc_6_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BC_7,             0x4E0380D8,__READ_WRITE ,__ipu_csi1_cpd_bc_7_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_0,             0x4E0380DC,__READ_WRITE ,__ipu_csi1_cpd_bs_0_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_1,             0x4E0380E0,__READ_WRITE ,__ipu_csi1_cpd_bc_1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_2,             0x4E0380E4,__READ_WRITE ,__ipu_csi1_cpd_bc_2_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_BS_3,             0x4E0380E8,__READ_WRITE ,__ipu_csi1_cpd_bc_3_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_OFFSET1,          0x4E0380EC,__READ_WRITE ,__ipu_csi1_cpd_offset1_bits);
__IO_REG32_BIT(IPU_CSI1_CPD_OFFSET2,          0x4E0380F0,__READ_WRITE ,__ipu_csi1_cpd_offset2_bits);
__IO_REG32_BIT(IPU_SMFC_MAP,                  0x4E050000,__READ_WRITE ,__ipu_smfc_map_bits);
__IO_REG32_BIT(IPU_SMFC_WMC,                  0x4E050004,__READ_WRITE ,__ipu_smfc_wmc_bits);
__IO_REG32_BIT(IPU_SMFC_BS,                   0x4E050008,__READ_WRITE ,__ipu_smfc_bs_bits);
__IO_REG32_BIT(IPU_VDI_FSIZE,                 0x4E068000,__READ_WRITE ,__ipu_vdi_fsize_bits);
__IO_REG32_BIT(IPU_VDI_C,                     0x4E068004,__READ_WRITE ,__ipu_vdi_c_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPP_KPCR,                  0x73F94000,__READ_WRITE ,__kpp_kpcr_bits);
__IO_REG16_BIT(KPP_KPSR,                  0x73F94002,__READ_WRITE ,__kpp_kpsr_bits);
__IO_REG16_BIT(KPP_KDDR,                  0x73F94004,__READ_WRITE ,__kpp_kddr_bits);
__IO_REG16_BIT(KPP_KPDR,                  0x73F94006,__READ_WRITE ,__kpp_kpdr_bits);

/***************************************************************************
 **
 **  M4IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M4IF_PSM0,                 0x83FD8000,__READ_WRITE ,__m4if_psm0_bits);
__IO_REG32_BIT(M4IF_PSM1,                 0x83FD8004,__READ_WRITE ,__m4if_psm1_bits);
__IO_REG32_BIT(M4IF_MDSR6,                0x83FD8018,__READ       ,__m4if_mdsr6_bits);
__IO_REG32(    M4IF_MDSR7,                0x83FD801C,__READ       );
__IO_REG32(    M4IF_MDSR8,                0x83FD8020,__READ       );
__IO_REG32_BIT(M4IF_MDSR0,                0x83FD8024,__READ       ,__m4if_mdsr0_bits);
__IO_REG32_BIT(M4IF_MDSR1,                0x83FD8028,__READ       ,__m4if_mdsr1_bits);
__IO_REG32_BIT(M4IF_MDSR2,                0x83FD802C,__READ       ,__m4if_mdsr2_bits);
__IO_REG32_BIT(M4IF_MDSR3,                0x83FD8030,__READ       ,__m4if_mdsr3_bits);
__IO_REG32_BIT(M4IF_MDSR4,                0x83FD8034,__READ       ,__m4if_mdsr4_bits);
__IO_REG32_BIT(M4IF_MDSR5,                0x83FD8038,__READ       ,__m4if_mdsr5_bits);
__IO_REG32_BIT(M4IF_FBPM0,                0x83FD8040,__READ_WRITE ,__m4if_fbpm0_bits);
__IO_REG32_BIT(M4IF_FBPM1,                0x83FD8044,__READ_WRITE ,__m4if_fbpm1_bits);
__IO_REG32_BIT(M4IF_MIF4,                 0x83FD8048,__READ_WRITE ,__m4if_mif4_bits);
__IO_REG32_BIT(M4IF_SBAR0,                0x83FD804C,__READ_WRITE ,__m4if_sbar0_bits);
__IO_REG32(    M4IF_SERL0,                0x83FD8050,__READ_WRITE );
__IO_REG32(    M4IF_SERH0,                0x83FD8054,__READ_WRITE );
__IO_REG32(    M4IF_SSRL0,                0x83FD8058,__READ_WRITE );
__IO_REG32(    M4IF_SSRH0,                0x83FD805C,__READ_WRITE );
__IO_REG32_BIT(M4IF_SBAR1,                0x83FD8060,__READ_WRITE ,__m4if_sbar1_bits);
__IO_REG32(    M4IF_SERL1,                0x83FD8064,__READ_WRITE );
__IO_REG32(    M4IF_SERH1,                0x83FD8068,__READ_WRITE );
__IO_REG32(    M4IF_SSRL1,                0x83FD806C,__READ_WRITE );
__IO_REG32(    M4IF_SSRH1,                0x83FD8070,__READ_WRITE );
__IO_REG32_BIT(M4IF_I2ULA,                0x83FD8074,__READ_WRITE ,__m4if_i2ula_bits);
__IO_REG32_BIT(M4IF_I2ACR,                0x83FD8078,__READ_WRITE ,__m4if_i2acr_bits);
__IO_REG32_BIT(M4IF_RINT2,                0x83FD807C,__READ_WRITE ,__m4if_rint2_bits);
__IO_REG32(    M4IF_SBS0,                 0x83FD8084,__READ       );
__IO_REG32_BIT(M4IF_SBS1,                 0x83FD8088,__READ_WRITE ,__m4if_sbs1_bits);
__IO_REG32_BIT(M4IF_MCR0,                 0x83FD808C,__READ_WRITE ,__m4if_mcr0_bits);
__IO_REG32_BIT(M4IF_MCR1,                 0x83FD8090,__READ_WRITE ,__m4if_mcr1_bits);
__IO_REG32_BIT(M4IF_MDCR,                 0x83FD8094,__READ_WRITE ,__m4if_mdcr_bits);
__IO_REG32_BIT(M4IF_FACR,                 0x83FD8098,__READ_WRITE ,__m4if_facr_bits);
__IO_REG32_BIT(M4IF_FPWC,                 0x83FD809C,__READ_WRITE ,__m4if_fpwc_bits);
__IO_REG32_BIT(M4IF_SACR,                 0x83FD80A0,__READ_WRITE ,__m4if_sacr_bits);
__IO_REG32_BIT(M4IF_PSM2,                 0x83FD80A4,__READ_WRITE ,__m4if_psm2_bits);
__IO_REG32_BIT(M4IF_IACR,                 0x83FD80A8,__READ_WRITE ,__m4if_iacr_bits);
__IO_REG32_BIT(M4IF_PSM3,                 0x83FD80AC,__READ_WRITE ,__m4if_psm3_bits);
__IO_REG32_BIT(M4IF_FULA,                 0x83FD80B0,__READ_WRITE ,__m4if_fula_bits);
__IO_REG32_BIT(M4IF_SULA,                 0x83FD80B4,__READ_WRITE ,__m4if_sula_bits);
__IO_REG32_BIT(M4IF_IULA,                 0x83FD80B8,__READ_WRITE ,__m4if_iula_bits);
__IO_REG32_BIT(M4IF_FDPS,                 0x83FD80BC,__READ       ,__m4if_fdps_bits);
__IO_REG32_BIT(M4IF_FDPC,                 0x83FD80C0,__READ_WRITE ,__m4if_fdpc_bits);
__IO_REG32_BIT(M4IF_MLEN,                 0x83FD80C4,__READ_WRITE ,__m4if_mlen_bits);
__IO_REG32_BIT(M4IF_WMSA0_0,              0x83FD80D4,__READ_WRITE ,__m4if_wmsa0_0_bits);
__IO_REG32_BIT(M4IF_WMSA0_1,              0x83FD80D8,__READ_WRITE ,__m4if_wmsa0_1_bits);
__IO_REG32_BIT(M4IF_WMSA0_2,              0x83FD80DC,__READ_WRITE ,__m4if_wmsa0_2_bits);
__IO_REG32_BIT(M4IF_WMSA0_3,              0x83FD80E0,__READ_WRITE ,__m4if_wmsa0_3_bits);
__IO_REG32_BIT(M4IF_WMSA0_4,              0x83FD80E4,__READ_WRITE ,__m4if_wmsa0_4_bits);
__IO_REG32_BIT(M4IF_WMSA0_5,              0x83FD80E8,__READ_WRITE ,__m4if_wmsa0_5_bits);
__IO_REG32_BIT(M4IF_WMSA0_6,              0x83FD80EC,__READ_WRITE ,__m4if_wmsa0_6_bits);
__IO_REG32_BIT(M4IF_WMSA0_7,              0x83FD80F0,__READ_WRITE ,__m4if_wmsa0_7_bits);
__IO_REG32_BIT(M4IF_WMEA0_0,              0x83FD80F4,__READ_WRITE ,__m4if_wmea0_0_bits);
__IO_REG32_BIT(M4IF_WMEA0_1,              0x83FD80F8,__READ_WRITE ,__m4if_wmea0_1_bits);
__IO_REG32_BIT(M4IF_WMEA0_2,              0x83FD80FC,__READ_WRITE ,__m4if_wmea0_2_bits);
__IO_REG32_BIT(M4IF_WMEA0_3,              0x83FD8100,__READ_WRITE ,__m4if_wmea0_3_bits);
__IO_REG32_BIT(M4IF_WMEA0_4,              0x83FD8104,__READ_WRITE ,__m4if_wmea0_4_bits);
__IO_REG32_BIT(M4IF_WMEA0_5,              0x83FD8108,__READ_WRITE ,__m4if_wmea0_5_bits);
__IO_REG32_BIT(M4IF_WMEA0_6,              0x83FD810C,__READ_WRITE ,__m4if_wmea0_6_bits);
__IO_REG32_BIT(M4IF_WMEA0_7,              0x83FD8110,__READ_WRITE ,__m4if_wmea0_7_bits);
__IO_REG32_BIT(M4IF_WMIS0,                0x83FD8114,__READ_WRITE ,__m4if_wmis0_bits);
__IO_REG32(    M4IF_FWMVA0,               0x83FD8118,__READ       );
__IO_REG32(    M4IF_SWMVA0,               0x83FD811C,__READ       );
__IO_REG32_BIT(M4IF_WMSA1_0,              0x83FD8120,__READ_WRITE ,__m4if_wmsa1_0_bits);
__IO_REG32_BIT(M4IF_WMSA1_1,              0x83FD8124,__READ_WRITE ,__m4if_wmsa1_1_bits);
__IO_REG32_BIT(M4IF_WMSA1_2,              0x83FD8128,__READ_WRITE ,__m4if_wmsa1_2_bits);
__IO_REG32_BIT(M4IF_WMSA1_3,              0x83FD812c,__READ_WRITE ,__m4if_wmsa1_3_bits);
__IO_REG32_BIT(M4IF_WMSA1_4,              0x83FD8130,__READ_WRITE ,__m4if_wmsa1_4_bits);
__IO_REG32_BIT(M4IF_WMSA1_5,              0x83FD8134,__READ_WRITE ,__m4if_wmsa1_5_bits);
__IO_REG32_BIT(M4IF_WMSA1_6,              0x83FD8138,__READ_WRITE ,__m4if_wmsa1_6_bits);
__IO_REG32_BIT(M4IF_WMSA1_7,              0x83FD813c,__READ_WRITE ,__m4if_wmsa1_7_bits);
__IO_REG32_BIT(M4IF_WMEA1_0,              0x83FD8140,__READ_WRITE ,__m4if_wmea1_0_bits);
__IO_REG32_BIT(M4IF_WMEA1_1,              0x83FD8144,__READ_WRITE ,__m4if_wmea1_1_bits);
__IO_REG32_BIT(M4IF_WMEA1_2,              0x83FD8148,__READ_WRITE ,__m4if_wmea1_2_bits);
__IO_REG32_BIT(M4IF_WMEA1_3,              0x83FD814C,__READ_WRITE ,__m4if_wmea1_3_bits);
__IO_REG32_BIT(M4IF_WMEA1_4,              0x83FD8150,__READ_WRITE ,__m4if_wmea1_4_bits);
__IO_REG32_BIT(M4IF_WMEA1_5,              0x83FD8154,__READ_WRITE ,__m4if_wmea1_5_bits);
__IO_REG32_BIT(M4IF_WMEA1_6,              0x83FD8158,__READ_WRITE ,__m4if_wmea1_6_bits);
__IO_REG32_BIT(M4IF_WMEA1_7,              0x83FD815C,__READ_WRITE ,__m4if_wmea1_7_bits);
__IO_REG32_BIT(M4IF_WMIS1,                0x83FD8160,__READ_WRITE ,__m4if_wmis1_bits);
__IO_REG32(    M4IF_FWMVA1,               0x83FD8164,__READ       );
__IO_REG32(    M4IF_SWMVA1,               0x83FD8168,__READ       );


/***************************************************************************
 **
 **  NFC
 **
 ***************************************************************************/
__IO_REG32_BIT(NFC_NAND_CMD,              0xCFFF1E00,__READ_WRITE ,__nfc_nand_cmd_bits);
__IO_REG32(    NFC_NAND_ADD0,             0xCFFF1E04,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD1,             0xCFFF1E08,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD2,             0xCFFF1E0C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD3,             0xCFFF1E10,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD4,             0xCFFF1E14,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD5,             0xCFFF1E18,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD6,             0xCFFF1E1C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD7,             0xCFFF1E20,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD8,             0xCFFF1E24,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD9,             0xCFFF1E28,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD10,            0xCFFF1E2C,__READ_WRITE );
__IO_REG32(    NFC_NAND_ADD11,            0xCFFF1E30,__READ_WRITE );
__IO_REG32_BIT(NFC_CONFIGURATION1,        0xCFFF1E34,__READ_WRITE ,__nfc_configuration1_bits);
__IO_REG32_BIT(NFC_ECC_STATUS_RESULT,     0xCFFF1E38,__READ       ,__nfc_ecc_status_result_bits);
__IO_REG32_BIT(NFC_STATUS_SUM,            0xCFFF1E3C,__READ_WRITE ,__nfc_status_sum_bits);
__IO_REG32_BIT(NFC_LAUNCH_NFC,            0xCFFF1E40,__READ_WRITE ,__nfc_launch_nfc_bits);
__IO_REG32_BIT(NFC_WR_PROTECT,            0x83FDB000,__READ_WRITE ,__nfc_wr_protect_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD0,       0x83FDB004,__READ_WRITE ,__nfc_unlock_blk_add0_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD1,       0x83FDB008,__READ_WRITE ,__nfc_unlock_blk_add1_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD2,       0x83FDB00C,__READ_WRITE ,__nfc_unlock_blk_add2_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD3,       0x83FDB010,__READ_WRITE ,__nfc_unlock_blk_add3_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD4,       0x83FDB014,__READ_WRITE ,__nfc_unlock_blk_add4_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD5,       0x83FDB018,__READ_WRITE ,__nfc_unlock_blk_add5_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD6,       0x83FDB01C,__READ_WRITE ,__nfc_unlock_blk_add6_bits);
__IO_REG32_BIT(NFC_UNLOCK_BLK_ADD7,       0x83FDB020,__READ_WRITE ,__nfc_unlock_blk_add7_bits);
__IO_REG32_BIT(NFC_CONFIGURATION2,        0x83FDB024,__READ_WRITE ,__nfc_configuration2_bits);
__IO_REG32_BIT(NFC_CONFIGURATION3,        0x83FDB028,__READ_WRITE ,__nfc_configuration3_bits);
__IO_REG32_BIT(NFC_IPC,                   0x83FDB02C,__READ_WRITE ,__nfc_ipc_bits);
__IO_REG32_BIT(NFC_AXI_ERR_ADD,           0x83FDB030,__READ_WRITE ,__nfc_axi_err_add_bits);
__IO_REG32_BIT(NFC_DELAY_LINE,            0x83FDB034,__READ_WRITE ,__nfc_delay_line_bits);

/***************************************************************************
 **
 **  OWIRE
 **
 ***************************************************************************/
__IO_REG16_BIT(OWIRE_CONTROL,             0x83FA4000,__READ_WRITE ,__owire_control_bits);
__IO_REG16_BIT(OWIRE_TIME_DIVIDER,        0x83FA4002,__READ_WRITE ,__owire_time_divider_bits);
__IO_REG16_BIT(OWIRE_RESET,               0x83FA4004,__READ_WRITE ,__owire_reset_bits);

/***************************************************************************
 **
 **  P-ATA
 **
 ***************************************************************************/
__IO_REG8(     PATA_TIME_OFF,             0x83FE0000,__READ_WRITE );
__IO_REG8(     PATA_TIME_ON,              0x83FE0001,__READ_WRITE );
__IO_REG8(     PATA_TIME_1,               0x83FE0002,__READ_WRITE );
__IO_REG8(     PATA_TIME_2W,              0x83FE0003,__READ_WRITE );
__IO_REG8(     PATA_TIME_2R,              0x83FE0004,__READ_WRITE );
__IO_REG8(     PATA_TIME_AX,              0x83FE0005,__READ_WRITE );
__IO_REG8(     PATA_TIME_PIO_RDX,         0x83FE0006,__READ_WRITE );
__IO_REG8(     PATA_TIME_4,               0x83FE0007,__READ_WRITE );
__IO_REG8(     PATA_TIME_9,               0x83FE0008,__READ_WRITE );
__IO_REG8(     PATA_TIME_M,               0x83FE0009,__READ_WRITE );
__IO_REG8(     PATA_TIME_JN,              0x83FE000A,__READ_WRITE );
__IO_REG8(     PATA_TIME_D,               0x83FE000B,__READ_WRITE );
__IO_REG8(     PATA_TIME_K,               0x83FE000C,__READ_WRITE );
__IO_REG8(     PATA_TIME_ACK,             0x83FE000D,__READ_WRITE );
__IO_REG8(     PATA_TIME_ENV,             0x83FE000E,__READ_WRITE );
__IO_REG8(     PATA_TIME_RPX,             0x83FE000F,__READ_WRITE );
__IO_REG8(     PATA_TIME_ZAH,             0x83FE0010,__READ_WRITE );
__IO_REG8(     PATA_TIME_MLIX,            0x83FE0011,__READ_WRITE );
__IO_REG8(     PATA_TIME_DVH,             0x83FE0012,__READ_WRITE );
__IO_REG8(     PATA_TIME_DZFS,            0x83FE0013,__READ_WRITE );
__IO_REG8(     PATA_TIME_DVS,             0x83FE0014,__READ_WRITE );
__IO_REG8(     PATA_TIME_CVH,             0x83FE0015,__READ_WRITE );
__IO_REG8(     PATA_TIME_SS,              0x83FE0016,__READ_WRITE );
__IO_REG8(     PATA_TIME_CYC,             0x83FE0017,__READ_WRITE );
__IO_REG8(     PATA_FIFO_DATA_32,         0x83FE0018,__READ_WRITE );
__IO_REG8(     PATA_FIFO_DATA_16,         0x83FE001C,__READ_WRITE );
__IO_REG8(     PATA_FIFO_FILL,            0x83FE0020,__READ       );
__IO_REG8_BIT( PATA_ATA_CONTROL,          0x83FE0024,__READ_WRITE ,__pata_control_bits);
__IO_REG8_BIT( PATA_INTERRUPT_PENDING,    0x83FE0028,__READ       ,__pata_interrupt_pending_bits);
__IO_REG8_BIT( PATA_INTERRUPT_ENABLE,     0x83FE002C,__READ_WRITE ,__pata_interrupt_enable_bits);
__IO_REG8(     PATA_INTERRUPT_CLEAR,      0x83FE0030,__WRITE      );
__IO_REG8(     PATA_FIFO_ALARM,           0x83FE0034,__READ_WRITE );
__IO_REG16(    PATA_DRIVE_DATA,           0x83FE00A0,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_FEATURES,       0x83FE00A4,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_SECTOR_COUNT,   0x83FE00A8,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_SECTOR_NUM,     0x83FE00AC,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_CYL_LOW,        0x83FE00B0,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_CYL_HIGH,       0x83FE00B4,__READ_WRITE );
__IO_REG32(    PATA_DRIVE_DEV_HEAD,       0x83FE00B8,__WRITE      );
__IO_REG32(    PATA_DRIVE_COMMAND,        0x83FE00BC,__READ       );
#define PATA_DRIVE_STATUS   PATA_DRIVE_COMMAND
__IO_REG32(    PATA_DRIVE_ALT_STATUS,     0x83FE00D8,__READ       );
#define PATA_DRIVE_CONTROL  PATA_DRIVE_ALT_STATUS

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_PWMCR,                0x73FB4000,__READ_WRITE ,__pwm_pwmcr_bits);
__IO_REG32_BIT(PWM1_PWMSR,                0x73FB4004,__READ_WRITE ,__pwm_pwmsr_bits);
__IO_REG32_BIT(PWM1_PWMIR,                0x73FB4008,__READ_WRITE ,__pwm_pwmir_bits);
__IO_REG32_BIT(PWM1_PWMSAR,               0x73FB400C,__READ_WRITE ,__pwm_pwmsar_bits);
__IO_REG32_BIT(PWM1_PWMPR,                0x73FB4010,__READ_WRITE ,__pwm_pwmpr_bits);
__IO_REG32_BIT(PWM1_PWMCNR,               0x73FB4014,__READ       ,__pwm_pwmcnr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_PWMCR,                0x73FB8000,__READ_WRITE ,__pwm_pwmcr_bits);
__IO_REG32_BIT(PWM2_PWMSR,                0x73FB8004,__READ_WRITE ,__pwm_pwmsr_bits);
__IO_REG32_BIT(PWM2_PWMIR,                0x73FB8008,__READ_WRITE ,__pwm_pwmir_bits);
__IO_REG32_BIT(PWM2_PWMSAR,               0x73FB800C,__READ_WRITE ,__pwm_pwmsar_bits);
__IO_REG32_BIT(PWM2_PWMPR,                0x73FB8010,__READ_WRITE ,__pwm_pwmpr_bits);
__IO_REG32_BIT(PWM2_PWMCNR,               0x73FB8014,__READ       ,__pwm_pwmcnr_bits);

/***************************************************************************
 **
 **  SDMA
 **
 ***************************************************************************/
__IO_REG32(    SDMA_MC0PTR,               0x83FB0000,__READ_WRITE );
__IO_REG32_BIT(SDMA_INTR,                 0x83FB0004,__READ_WRITE ,__sdma_intr_bits);
__IO_REG32_BIT(SDMA_STOP_STAT,            0x83FB0008,__READ       ,__sdma_stop_stat_bits);
__IO_REG32_BIT(SDMA_HSTART,               0x83FB000C,__READ_WRITE ,__sdma_hstart_bits);
__IO_REG32_BIT(SDMA_EVTOVR,               0x83FB0010,__READ_WRITE ,__sdma_evtovr_bits);
__IO_REG32_BIT(SDMA_DSPOVR,               0x83FB0014,__READ_WRITE ,__sdma_dspovr_bits);
__IO_REG32_BIT(SDMA_HOSTOVR,              0x83FB0018,__READ_WRITE ,__sdma_hostovr_bits);
__IO_REG32_BIT(SDMA_EVTPEND,              0x83FB001C,__READ       ,__sdma_evtpend_bits);
__IO_REG32_BIT(SDMA_RESET,                0x83FB0024,__READ       ,__sdma_reset_bits);
__IO_REG32_BIT(SDMA_EVTERR,               0x83FB0028,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_INTRMASK,             0x83FB002C,__READ_WRITE ,__sdma_intrmask_bits);
__IO_REG32_BIT(SDMA_PSW,                  0x83FB0030,__READ       ,__sdma_psw_bits);
__IO_REG32_BIT(SDMA_EVTERRDBG,            0x83FB0034,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMA_CONFIG,               0x83FB0038,__READ_WRITE ,__sdma_config_bits);
__IO_REG32_BIT(SDMA_LOCK,                 0x83FB003C,__READ_WRITE ,__sdma_lock_bits);
__IO_REG32_BIT(SDMA_ONCE_ENB,             0x83FB0040,__READ_WRITE ,__sdma_once_enb_bits);
__IO_REG32(    SDMA_ONCE_DATA,            0x83FB0044,__READ_WRITE );
__IO_REG32_BIT(SDMA_ONCE_INSTR,           0x83FB0048,__READ_WRITE ,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMA_ONCE_STAT,            0x83FB004C,__READ       ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMA_ONCE_CMD,             0x83FB0050,__READ_WRITE ,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMA_ILLINSTADDR,          0x83FB0058,__READ_WRITE ,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMA_CHN0ADDR,             0x83FB005C,__READ_WRITE ,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR,           0x83FB0060,__READ       ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMA_EVT_MIRROR2,          0x83FB0064,__READ       ,__sdma_evt_mirror2_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF1,          0x83FB0070,__READ_WRITE ,__sdma_xtrig1_conf_bits);
__IO_REG32_BIT(SDMA_XTRIG_CONF2,          0x83FB0074,__READ_WRITE ,__sdma_xtrig2_conf_bits);
__IO_REG32_BIT(SDMA_OTB,                  0x83FB0078,__READ       ,__sdma_otb_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_1,            0x83FB007C,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_2,            0x83FB0080,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_3,            0x83FB0084,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_4,            0x83FB0088,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_5,            0x83FB008C,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT_6,            0x83FB0090,__READ_WRITE ,__sdma_prf_cnt_x_bits);
__IO_REG32_BIT(SDMA_PRF_CNT,              0x83FB0094,__READ_WRITE ,__sdma_prf_cnt_bits);
__IO_REG32_BIT(SDMA_CHNPRI0,              0x83FB0100,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI1,              0x83FB0104,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI2,              0x83FB0108,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI3,              0x83FB010C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI4,              0x83FB0110,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI5,              0x83FB0114,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI6,              0x83FB0118,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI7,              0x83FB011C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI8,              0x83FB0120,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI9,              0x83FB0124,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI10,             0x83FB0128,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI11,             0x83FB012C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI12,             0x83FB0130,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI13,             0x83FB0134,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI14,             0x83FB0138,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI15,             0x83FB013C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI16,             0x83FB0140,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI17,             0x83FB0144,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI18,             0x83FB0148,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI19,             0x83FB014C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI20,             0x83FB0150,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI21,             0x83FB0154,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI22,             0x83FB0158,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI23,             0x83FB015C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI24,             0x83FB0160,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI25,             0x83FB0164,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI26,             0x83FB0168,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI27,             0x83FB016C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI28,             0x83FB0170,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI29,             0x83FB0174,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI30,             0x83FB0178,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMA_CHNPRI31,             0x83FB017C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32(    SDMA_CHNENBL0,             0x83FB0200,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL1,             0x83FB0204,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL2,             0x83FB0208,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL3,             0x83FB020C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL4,             0x83FB0210,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL5,             0x83FB0214,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL6,             0x83FB0218,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL7,             0x83FB021C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL8,             0x83FB0220,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL9,             0x83FB0224,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL10,            0x83FB0228,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL11,            0x83FB022C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL12,            0x83FB0230,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL13,            0x83FB0234,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL14,            0x83FB0238,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL15,            0x83FB023C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL16,            0x83FB0240,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL17,            0x83FB0244,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL18,            0x83FB0248,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL19,            0x83FB024C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL20,            0x83FB0250,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL21,            0x83FB0254,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL22,            0x83FB0258,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL23,            0x83FB025C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL24,            0x83FB0260,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL25,            0x83FB0264,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL26,            0x83FB0268,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL27,            0x83FB026C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL28,            0x83FB0270,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL29,            0x83FB0274,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL30,            0x83FB0278,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL31,            0x83FB027C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL32,            0x83FB0280,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL33,            0x83FB0284,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL34,            0x83FB0288,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL35,            0x83FB028C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL36,            0x83FB0290,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL37,            0x83FB0294,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL38,            0x83FB0298,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL39,            0x83FB029C,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL40,            0x83FB02A0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL41,            0x83FB02A4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL42,            0x83FB02A8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL43,            0x83FB02AC,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL44,            0x83FB02B0,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL45,            0x83FB02B4,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL46,            0x83FB02B8,__READ_WRITE );
__IO_REG32(    SDMA_CHNENBL47,            0x83FB02BC,__READ_WRITE );

/***************************************************************************
 **
 **  SPDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(SPDIF_SCR,                 0x70028000,__READ_WRITE ,__spdif_scr_bits);
__IO_REG32_BIT(SPDIF_SIE,                 0x7002800C,__READ_WRITE ,__spdif_sie_bits);
__IO_REG32_BIT(SPDIF_SIS,                 0x70028010,__READ_WRITE ,__spdif_sis_sic_bits);
#define SPDIF_SIC        SPDIF_SIS
#define SPDIF_SIC_bit    SPDIF_SIS_bit
__IO_REG32(    SPDIF_STL,                 0x7002802C,__WRITE      );
__IO_REG32(    SPDIF_STR,                 0x70028030,__WRITE      );
__IO_REG32(    SPDIF_STCSCH,              0x70028034,__WRITE      );
__IO_REG32(    SPDIF_STCSCL,              0x70028038,__WRITE      );
__IO_REG32_BIT(SPDIF_STC,                 0x70028050,__READ_WRITE ,__spdif_stc_bits);

/***************************************************************************
 **
 **  SRC
 **
 ***************************************************************************/
__IO_REG32_BIT(SRC_SCR,                   0x73FD0000,__READ_WRITE ,__src_scr_bits);
__IO_REG32_BIT(SRC_SBMR,                  0x73FD0004,__READ_WRITE ,__src_sbmr_bits);
__IO_REG32_BIT(SRC_SRSR,                  0x73FD0008,__READ_WRITE ,__src_srsr_bits);
__IO_REG32_BIT(SRC_SISR,                  0x73FD0014,__READ       ,__src_sisr_bits);
__IO_REG32_BIT(SRC_SIMR,                  0x73FD0018,__READ_WRITE ,__src_simr_bits);

/***************************************************************************
 **
 **  SSI1
 **
 ***************************************************************************/
__IO_REG32(    SSI1_STX0,                 0x83FCC000,__READ_WRITE );
__IO_REG32(    SSI1_STX1,                 0x83FCC004,__READ_WRITE );
__IO_REG32(    SSI1_SRX0,                 0x83FCC008,__READ       );
__IO_REG32(    SSI1_SRX1,                 0x83FCC00C,__READ       );
__IO_REG32_BIT(SSI1_SCR,                  0x83FCC010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI1_SISR,                 0x83FCC014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI1_SIER,                 0x83FCC018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI1_STCR,                 0x83FCC01C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI1_SRCR,                 0x83FCC020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI1_STCCR,                0x83FCC024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI1_SRCCR,                0x83FCC028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI1_SFCSR,                0x83FCC02C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI1_SACNT,                0x83FCC038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI1_SACADD,               0x83FCC03C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI1_SACDAT,               0x83FCC040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI1_SATAG,                0x83FCC044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI1_STMSK,                0x83FCC048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI1_SRMSK,                0x83FCC04C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI1_SACCST,               0x83FCC050,__READ       ,__saccst_bits);
__IO_REG32(    SSI1_SACCEN,               0x83FCC054,__WRITE      );
__IO_REG32(    SSI1_SACCDIS,              0x83FCC058,__WRITE      );

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    SSI2_STX0,                 0x70014000,__READ_WRITE );
__IO_REG32(    SSI2_STX1,                 0x70014004,__READ_WRITE );
__IO_REG32(    SSI2_SRX0,                 0x70014008,__READ       );
__IO_REG32(    SSI2_SRX1,                 0x7001400C,__READ       );
__IO_REG32_BIT(SSI2_SCR,                  0x70014010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI2_SISR,                 0x70014014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI2_SIER,                 0x70014018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI2_STCR,                 0x7001401C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI2_SRCR,                 0x70014020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI2_STCCR,                0x70014024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI2_SRCCR,                0x70014028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI2_SFCSR,                0x7001402C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI2_SACNT,                0x70014038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI2_SACADD,               0x7001403C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI2_SACDAT,               0x70014040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI2_SATAG,                0x70014044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI2_STMSK,                0x70014048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI2_SRMSK,                0x7001404C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI2_SACCST,               0x70014050,__READ       ,__saccst_bits);
__IO_REG32(    SSI2_SACCEN,               0x70014054,__WRITE      );
__IO_REG32(    SSI2_SACCDIS,              0x70014058,__WRITE      );

/***************************************************************************
 **
 **  SSI3
 **
 ***************************************************************************/
__IO_REG32(    SSI3_STX0,                 0x83FE8000,__READ_WRITE );
__IO_REG32(    SSI3_STX1,                 0x83FE8004,__READ_WRITE );
__IO_REG32(    SSI3_STR0,                 0x83FE8008,__READ       );
__IO_REG32(    SSI3_STR1,                 0x83FE800C,__READ       );
__IO_REG32_BIT(SSI3_SCR,                  0x83FE8010,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(SSI3_SISR,                 0x83FE8014,__READ       ,__sisr_bits);
__IO_REG32_BIT(SSI3_SIER,                 0x83FE8018,__READ_WRITE ,__sier_bits);
__IO_REG32_BIT(SSI3_STCR,                 0x83FE801C,__READ_WRITE ,__stcr_bits);
__IO_REG32_BIT(SSI3_SRCR,                 0x83FE8020,__READ_WRITE ,__srcr_bits);
__IO_REG32_BIT(SSI3_STCCR,                0x83FE8024,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI3_SRCCR,                0x83FE8028,__READ_WRITE ,__stccr_bits);
__IO_REG32_BIT(SSI3_SFCSR,                0x83FE802C,__READ_WRITE ,__sfcsr_bits);
__IO_REG32_BIT(SSI3_SACNT,                0x83FE8038,__READ_WRITE ,__sacnt_bits);
__IO_REG32_BIT(SSI3_SACADD,               0x83FE803C,__READ_WRITE ,__sacadd_bits);
__IO_REG32_BIT(SSI3_SACDAT,               0x83FE8040,__READ_WRITE ,__sacdat_bits);
__IO_REG32_BIT(SSI3_SATAG,                0x83FE8044,__READ_WRITE ,__satag_bits);
__IO_REG32_BIT(SSI3_STMSK,                0x83FE8048,__READ_WRITE ,__stmsk_bits);
__IO_REG32_BIT(SSI3_SRMSK,                0x83FE804C,__READ_WRITE ,__srmsk_bits);
__IO_REG32_BIT(SSI3_SACCST,               0x83FE8050,__READ       ,__saccst_bits);
__IO_REG32(    SSI3_SACCEN,               0x83FE8054,__WRITE      );
__IO_REG32(    SSI3_SACCDIS,              0x83FE8058,__WRITE      );

/***************************************************************************
 **
 **  TVE
 **
 ***************************************************************************/
__IO_REG32_BIT(TVE_COM_CONF_REG,          0x83FF0000,__READ_WRITE ,__tve_com_conf_reg_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_0,  0x83FF0004,__READ_WRITE ,__tve_luma_filt_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_1,  0x83FF0008,__READ_WRITE ,__tve_luma_filt_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_2,  0x83FF000C,__READ_WRITE ,__tve_luma_filt_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_FILT_CONT_REG_3,  0x83FF0010,__READ_WRITE ,__tve_luma_filt_cont_reg_3_bits);
__IO_REG32_BIT(TVE_LUMA_SA_CONT_REG_0,    0x83FF0014,__READ_WRITE ,__tve_luma_sa_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_SA_CONT_REG_1,    0x83FF0018,__READ_WRITE ,__tve_luma_sa_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_SA_STAT_REG_0,    0x83FF001C,__READ_WRITE ,__tve_luma_sa_stat_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_SA_STAT_REG_1,    0x83FF0020,__READ_WRITE ,__tve_luma_sa_stat_reg_1_bits);
__IO_REG32_BIT(TVE_CHROMA_CONT_REG,       0x83FF0024,__READ_WRITE ,__tve_chroma_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_0_CONT_REG,      0x83FF0028,__READ_WRITE ,__tve_tvdac_0_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_1_CONT_REG,      0x83FF002C,__READ_WRITE ,__tve_tvdac_1_cont_reg_bits);
__IO_REG32_BIT(TVE_TVDAC_2_CONT_REG,      0x83FF0030,__READ_WRITE ,__tve_tvdac_2_cont_reg_bits);
__IO_REG32_BIT(TVE_CD_CONT_REG,           0x83FF0034,__READ_WRITE ,__tve_cd_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_CONT_REG,     0x83FF0038,__READ_WRITE ,__tve_vbi_data_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_REG_0,        0x83FF003C,__READ_WRITE ,__tve_vbi_data_reg_0_bits);
__IO_REG32_BIT(TVE_VBI_DATA_REG_1,        0x83FF0040,__READ_WRITE ,__tve_vbi_data_reg_1_bits);
__IO_REG32(    TVE_VBI_DATA_REG_2,        0x83FF0044,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_3,        0x83FF0048,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_4,        0x83FF004C,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_5,        0x83FF0050,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_6,        0x83FF0054,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_7,        0x83FF0058,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_8,        0x83FF005C,__READ_WRITE );
__IO_REG32(    TVE_VBI_DATA_REG_9,        0x83FF0060,__READ_WRITE );
__IO_REG32_BIT(TVE_INT_CONT_REG,          0x83FF0064,__READ_WRITE ,__tve_int_cont_reg_bits);
__IO_REG32_BIT(TVE_STAT_REG,              0x83FF0068,__READ_WRITE ,__tve_stat_reg_bits);
__IO_REG32_BIT(TVE_TST_MODE_REG,          0x83FF006C,__READ_WRITE ,__tve_tst_mode_reg_bits);
__IO_REG32_BIT(TVE_USER_MODE_CONT_REG,    0x83FF0070,__READ_WRITE ,__tve_user_mode_cont_reg_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_0,0x83FF0074,__READ_WRITE,__tve_sd_timing_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_1,0x83FF0078,__READ_WRITE,__tve_sd_timing_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_SD_TIMING_USR_CONT_REG_2,0x83FF007C,__READ_WRITE,__tve_sd_timing_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_0,0x83FF0080,__READ_WRITE,__tve_hd_timing_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_1,0x83FF0084,__READ_WRITE,__tve_hd_timing_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_HD_TIMING_USR_CONT_REG_2,0x83FF0088,__READ_WRITE,__tve_hd_timing_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_0,   0x83FF008C,__READ_WRITE ,__tve_luma_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_1,   0x83FF0090,__READ_WRITE ,__tve_luma_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_2,   0x83FF0094,__READ_WRITE ,__tve_luma_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_LUMA_USR_CONT_REG_3,   0x83FF0098,__READ_WRITE ,__tve_luma_usr_cont_reg_3_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_0,    0x83FF009C,__READ_WRITE ,__tve_csc_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_1,    0x83FF00A0,__READ_WRITE ,__tve_csc_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_CSC_USR_CONT_REG_2,    0x83FF00A4,__READ_WRITE ,__tve_csc_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_BLANK_USR_CONT_REG,    0x83FF00A8,__READ_WRITE ,__tve_blank_usr_cont_reg_bits);
__IO_REG32_BIT(TVE_SD_MOD_USR_CONT_REG,   0x83FF00AC,__READ_WRITE ,__tve_sd_mod_usr_cont_reg_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_0,0x83FF00B0,__READ_WRITE,__tve_vbi_data_usr_cont_reg_0_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_1,0x83FF00B4,__READ_WRITE,__tve_vbi_data_usr_cont_reg_1_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_2,0x83FF00B8,__READ_WRITE,__tve_vbi_data_usr_cont_reg_2_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_3,0x83FF00BC,__READ_WRITE,__tve_vbi_data_usr_cont_reg_3_bits);
__IO_REG32_BIT(TVE_VBI_DATA_USR_CONT_REG_4,0x83FF00C0,__READ_WRITE,__tve_vbi_data_usr_cont_reg_4_bits);
__IO_REG32_BIT(TVE_DROP_COMP_USR_CONT_REG,0x83FF00C4,__READ_WRITE ,__tve_drop_comp_usr_cont_reg_bits);

/***************************************************************************
 **
 **  TZIC
 **
 ***************************************************************************/
__IO_REG32_BIT(TZIC_INTCTRL,              0xE0000000,__READ_WRITE ,__tzic_intctrl_bits);
__IO_REG32_BIT(TZIC_INTTYPE,              0xE0000004,__READ       ,__tzic_inttype_bits);
__IO_REG32_BIT(TZIC_PRIOMASK,             0xE000000C,__READ_WRITE ,__tzic_priomask_bits);
__IO_REG32_BIT(TZIC_SYNCCTRL,             0xE0000010,__READ_WRITE ,__tzic_syncctrl_bits);
__IO_REG32_BIT(TZIC_DSMINT,               0xE0000014,__READ_WRITE ,__tzic_dsmint_bits);
__IO_REG32_BIT(TZIC_INTSEC0,              0xE0000080,__READ_WRITE ,__tzic_intsec0_bits);
__IO_REG32_BIT(TZIC_INTSEC1,              0xE0000084,__READ_WRITE ,__tzic_intsec1_bits);
__IO_REG32_BIT(TZIC_INTSEC2,              0xE0000088,__READ_WRITE ,__tzic_intsec2_bits);
__IO_REG32_BIT(TZIC_INTSEC3,              0xE000008C,__READ_WRITE ,__tzic_intsec3_bits);
__IO_REG32_BIT(TZIC_ENSET0,               0xE0000100,__READ_WRITE ,__tzic_enset0_bits);
__IO_REG32_BIT(TZIC_ENSET1,               0xE0000104,__READ_WRITE ,__tzic_enset1_bits);
__IO_REG32_BIT(TZIC_ENSET2,               0xE0000108,__READ_WRITE ,__tzic_enset2_bits);
__IO_REG32_BIT(TZIC_ENSET3,               0xE000010C,__READ_WRITE ,__tzic_enset3_bits);
__IO_REG32_BIT(TZIC_ENCLEAR0,             0xE0000180,__READ_WRITE ,__tzic_enclear0_bits);
__IO_REG32_BIT(TZIC_ENCLEAR1,             0xE0000184,__READ_WRITE ,__tzic_enclear1_bits);
__IO_REG32_BIT(TZIC_ENCLEAR2,             0xE0000188,__READ_WRITE ,__tzic_enclear2_bits);
__IO_REG32_BIT(TZIC_ENCLEAR3,             0xE000018C,__READ_WRITE ,__tzic_enclear3_bits);
__IO_REG32_BIT(TZIC_SRCSET0,              0xE0000200,__READ_WRITE ,__tzic_srcset0_bits);
__IO_REG32_BIT(TZIC_SRCSET1,              0xE0000204,__READ_WRITE ,__tzic_srcset1_bits);
__IO_REG32_BIT(TZIC_SRCSET2,              0xE0000208,__READ_WRITE ,__tzic_srcset2_bits);
__IO_REG32_BIT(TZIC_SRCSET3,              0xE000020C,__READ_WRITE ,__tzic_srcset3_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR0,            0xE0000280,__READ_WRITE ,__tzic_srcclear0_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR1,            0xE0000284,__READ_WRITE ,__tzic_srcclear1_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR2,            0xE0000288,__READ_WRITE ,__tzic_srcclear2_bits);
__IO_REG32_BIT(TZIC_SRCCLEAR3,            0xE000028C,__READ_WRITE ,__tzic_srcclear3_bits);
__IO_REG32_BIT(TZIC_PRIORITY0,            0xE0000400,__READ_WRITE ,__tzic_priority0_bits);
__IO_REG32_BIT(TZIC_PRIORITY1,            0xE0000404,__READ_WRITE ,__tzic_priority1_bits);
__IO_REG32_BIT(TZIC_PRIORITY2,            0xE0000408,__READ_WRITE ,__tzic_priority2_bits);
__IO_REG32_BIT(TZIC_PRIORITY3,            0xE000040C,__READ_WRITE ,__tzic_priority3_bits);
__IO_REG32_BIT(TZIC_PRIORITY4,            0xE0000410,__READ_WRITE ,__tzic_priority4_bits);
__IO_REG32_BIT(TZIC_PRIORITY5,            0xE0000414,__READ_WRITE ,__tzic_priority5_bits);
__IO_REG32_BIT(TZIC_PRIORITY6,            0xE0000418,__READ_WRITE ,__tzic_priority6_bits);
__IO_REG32_BIT(TZIC_PRIORITY7,            0xE000041C,__READ_WRITE ,__tzic_priority7_bits);
__IO_REG32_BIT(TZIC_PRIORITY8,            0xE0000420,__READ_WRITE ,__tzic_priority8_bits);
__IO_REG32_BIT(TZIC_PRIORITY9,            0xE0000424,__READ_WRITE ,__tzic_priority9_bits);
__IO_REG32_BIT(TZIC_PRIORITY10,           0xE0000428,__READ_WRITE ,__tzic_priority10_bits);
__IO_REG32_BIT(TZIC_PRIORITY11,           0xE000042C,__READ_WRITE ,__tzic_priority11_bits);
__IO_REG32_BIT(TZIC_PRIORITY12,           0xE0000430,__READ_WRITE ,__tzic_priority12_bits);
__IO_REG32_BIT(TZIC_PRIORITY13,           0xE0000434,__READ_WRITE ,__tzic_priority13_bits);
__IO_REG32_BIT(TZIC_PRIORITY14,           0xE0000438,__READ_WRITE ,__tzic_priority14_bits);
__IO_REG32_BIT(TZIC_PRIORITY15,           0xE000043C,__READ_WRITE ,__tzic_priority15_bits);
__IO_REG32_BIT(TZIC_PRIORITY16,           0xE0000440,__READ_WRITE ,__tzic_priority16_bits);
__IO_REG32_BIT(TZIC_PRIORITY17,           0xE0000444,__READ_WRITE ,__tzic_priority17_bits);
__IO_REG32_BIT(TZIC_PRIORITY18,           0xE0000448,__READ_WRITE ,__tzic_priority18_bits);
__IO_REG32_BIT(TZIC_PRIORITY19,           0xE000044C,__READ_WRITE ,__tzic_priority19_bits);
__IO_REG32_BIT(TZIC_PRIORITY20,           0xE0000450,__READ_WRITE ,__tzic_priority20_bits);
__IO_REG32_BIT(TZIC_PRIORITY21,           0xE0000454,__READ_WRITE ,__tzic_priority21_bits);
__IO_REG32_BIT(TZIC_PRIORITY22,           0xE0000458,__READ_WRITE ,__tzic_priority22_bits);
__IO_REG32_BIT(TZIC_PRIORITY23,           0xE000045C,__READ_WRITE ,__tzic_priority23_bits);
__IO_REG32_BIT(TZIC_PRIORITY24,           0xE0000460,__READ_WRITE ,__tzic_priority24_bits);
__IO_REG32_BIT(TZIC_PRIORITY25,           0xE0000464,__READ_WRITE ,__tzic_priority25_bits);
__IO_REG32_BIT(TZIC_PRIORITY26,           0xE0000468,__READ_WRITE ,__tzic_priority26_bits);
__IO_REG32_BIT(TZIC_PRIORITY27,           0xE000046C,__READ_WRITE ,__tzic_priority27_bits);
__IO_REG32_BIT(TZIC_PRIORITY28,           0xE0000470,__READ_WRITE ,__tzic_priority28_bits);
__IO_REG32_BIT(TZIC_PRIORITY29,           0xE0000474,__READ_WRITE ,__tzic_priority29_bits);
__IO_REG32_BIT(TZIC_PRIORITY30,           0xE0000478,__READ_WRITE ,__tzic_priority30_bits);
__IO_REG32_BIT(TZIC_PRIORITY31,           0xE000047C,__READ_WRITE ,__tzic_priority31_bits);
__IO_REG32_BIT(TZIC_PND0,                 0xE0000D00,__READ       ,__tzic_pnd0_bits);
__IO_REG32_BIT(TZIC_PND1,                 0xE0000D04,__READ       ,__tzic_pnd1_bits);
__IO_REG32_BIT(TZIC_PND2,                 0xE0000D08,__READ       ,__tzic_pnd2_bits);
__IO_REG32_BIT(TZIC_PND3,                 0xE0000D0C,__READ       ,__tzic_pnd3_bits);
__IO_REG32_BIT(TZIC_HIPND0,               0xE0000D80,__READ       ,__tzic_hipnd0_bits);
__IO_REG32_BIT(TZIC_HIPND1,               0xE0000D84,__READ       ,__tzic_hipnd1_bits);
__IO_REG32_BIT(TZIC_HIPND2,               0xE0000D88,__READ       ,__tzic_hipnd2_bits);
__IO_REG32_BIT(TZIC_HIPND3,               0xE0000D8C,__READ       ,__tzic_hipnd3_bits);
__IO_REG32_BIT(TZIC_WAKEUP0,              0xE0000E00,__READ       ,__tzic_wakeup0_bits);
__IO_REG32_BIT(TZIC_WAKEUP1,              0xE0000E04,__READ       ,__tzic_wakeup1_bits);
__IO_REG32_BIT(TZIC_WAKEUP2,              0xE0000E08,__READ       ,__tzic_wakeup2_bits);
__IO_REG32_BIT(TZIC_WAKEUP3,              0xE0000E0C,__READ       ,__tzic_wakeup3_bits);
__IO_REG32(    TZIC_SWINT,                0xE0000F00,__WRITE      );

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1_URXD,                0x73FBC000,__READ        ,__urxd_bits);
__IO_REG32(    UART1_UTXD,                0x73FBC040,__WRITE       );
__IO_REG32_BIT(UART1_UCR1,                0x73FBC080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART1_UCR2,                0x73FBC084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART1_UCR3,                0x73FBC088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART1_UCR4,                0x73FBC08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART1_UFCR,                0x73FBC090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART1_USR1,                0x73FBC094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART1_USR2,                0x73FBC098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART1_UESC,                0x73FBC09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART1_UTIM,                0x73FBC0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART1_UBIR,                0x73FBC0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART1_UBMR,                0x73FBC0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART1_UBRC,                0x73FBC0AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART1_ONEMS,               0x73FBC0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART1_UTS,                 0x73FBC0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2_URXD,                0x73FC0000,__READ        ,__urxd_bits);
__IO_REG32(    UART2_UTXD,                0x73FC0040,__WRITE       );
__IO_REG32_BIT(UART2_UCR1,                0x73FC0080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART2_UCR2,                0x73FC0084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART2_UCR3,                0x73FC0088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART2_UCR4,                0x73FC008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART2_UFCR,                0x73FC0090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART2_USR1,                0x73FC0094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART2_USR2,                0x73FC0098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART2_UESC,                0x73FC009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART2_UTIM,                0x73FC00A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART2_UBIR,                0x73FC00A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART2_UBMR,                0x73FC00A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART2_UBRC,                0x73FC00AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART2_ONEMS,               0x73FC00B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART2_UTS,                 0x73FC00B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(UART3_URXD,                0x7000C000,__READ        ,__urxd_bits);
__IO_REG32(    UART3_UTXD,                0x7000C040,__WRITE       );
__IO_REG32_BIT(UART3_UCR1,                0x7000C080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART3_UCR2,                0x7000C084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART3_UCR3,                0x7000C088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART3_UCR4,                0x7000C08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART3_UFCR,                0x7000C090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART3_USR1,                0x7000C094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART3_USR2,                0x7000C098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART3_UESC,                0x7000C09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART3_UTIM,                0x7000C0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART3_UBIR,                0x7000C0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART3_UBMR,                0x7000C0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART3_UBRC,                0x7000C0AC,__READ_WRITE  ,__ubrc_bits);
__IO_REG32_BIT(UART3_ONEMS,               0x7000C0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART3_UTS,                 0x7000C0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  DPTC_LP
 **
 ***************************************************************************/
__IO_REG32_BIT(DPTC_LP_DPTCCR,            0x73FD8080,__READ_WRITE  ,__dptc_dptccr_bits);
__IO_REG32_BIT(DPTC_LP_DPTCDBG,           0x73FD8084,__READ_WRITE  ,__dptc_dptcdbg_bits);
__IO_REG32_BIT(DPTC_LP_DCVR0,             0x73FD8088,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_LP_DCVR1,             0x73FD808C,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_LP_DCVR2,             0x73FD8090,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_LP_DCVR3,             0x73FD8094,__READ_WRITE  ,__dptc_dcvrx_bits);

/***************************************************************************
 **
 **  DPTC_GP
 **
 ***************************************************************************/
__IO_REG32_BIT(DPTC_GP_DPTCCR,            0x73FD8100,__READ_WRITE  ,__dptc_dptccr_bits);
__IO_REG32_BIT(DPTC_GP_DPTCDBG,           0x73FD8104,__READ_WRITE  ,__dptc_dptcdbg_bits);
__IO_REG32_BIT(DPTC_GP_DCVR0,             0x73FD8108,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_GP_DCVR1,             0x73FD810C,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_GP_DCVR2,             0x73FD8110,__READ_WRITE  ,__dptc_dcvrx_bits);
__IO_REG32_BIT(DPTC_GP_DCVR3,             0x73FD8114,__READ_WRITE  ,__dptc_dcvrx_bits);

/***************************************************************************
 **
 **  WDOG1
 **
 ***************************************************************************/
__IO_REG16_BIT(WDOG1_WCR,                 0x73F98000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WDOG1_WSR,                 0x73F98002,__READ_WRITE );
__IO_REG16_BIT(WDOG1_WRSR,                0x73F98004,__READ       ,__wrsr_bits);
__IO_REG16_BIT(WDOG1_WICR,                0x73F98006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WDOG1_WMCR,                0x73F98008,__READ_WRITE ,__wmcr_bits);

/***************************************************************************
 **
 **  WDOG2
 **
 ***************************************************************************/
__IO_REG16_BIT(WDOG2_WCR,                 0x73F9C000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WDOG2_WSR,                 0x73F9C002,__READ_WRITE );
__IO_REG16_BIT(WDOG2_WRSR,                0x73F9C004,__READ       ,__wrsr_bits);
__IO_REG16_BIT(WDOG2_WICR,                0x73F9C006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WDOG2_WMCR,                0x73F9C008,__READ_WRITE ,__wmcr_bits);

/***************************************************************************
 **
 **  VPU
 **
 ***************************************************************************/
__IO_REG32(    VPU_CodeRun,               0x83FF4000,__WRITE      );
__IO_REG32(    VPU_CodeDown,              0x83FF4004,__WRITE      );
__IO_REG32(    VPU_HostIntReq,            0x83FF4008,__WRITE      );
__IO_REG32(    VPU_BitIntClear,           0x83FF400C,__WRITE      );
__IO_REG32_BIT(VPU_BitIntSts,             0x83FF4010,__READ       ,__vpu_bitintsts_bits);
__IO_REG32(    VPU_BitCodeReset,          0x83FF4014,__WRITE      );
__IO_REG32_BIT(VPU_BitCurPc,              0x83FF4018,__READ       ,__vpu_bitcurpc_bits);
__IO_REG32_BIT(VPU_BitCodecBusy,          0x83FF4020,__READ_WRITE ,__vpu_bitcodecbusy_bits);

/***************************************************************************
 **
 **  USB OTG
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x73F80000,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x73F80004,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x73F80008,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x73F8000C,__READ       ,__uog_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x73F80010,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x73F80014,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UOG_GPTIMER0LD,            0x73F80080,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UOG_GPTIMER0CTRL,          0x73F80084,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UOG_GPTIMER1LD,            0x73F80088,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UOG_GPTIMER1CTRL,          0x73F8008C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG8(     UOG_CAPLENGTH,             0x73F80100,__READ       );
__IO_REG16(    UOG_HCIVERSION,            0x73F80102,__READ       );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x73F80104,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x73F80108,__READ       ,__uog_hccparams_bits);
__IO_REG16(    UOG_DCIVERSION,            0x73F80120,__READ       );
__IO_REG32_BIT(UOG_DCCPARAMS,             0x73F80124,__READ       ,__uog_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x73F80140,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x73F80144,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x73F80148,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x73F8014C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x73F80154,__READ_WRITE ,__uog_periodiclistbase_bits);
#define UOG_DEVICEADDR      UOG_PERIODICLISTBASE
#define UOG_DEVICEADDR_bit  UOG_PERIODICLISTBASE_bit
__IO_REG32_BIT(UOG_ASYNCLISTADDR,         0x73F80158,__READ_WRITE ,__uog_asynclistaddr_bits);
#define UOG_ENDPOINTLISTADDR      UOG_ASYNCLISTADDR
#define UOG_ENDPOINTLISTADDR_bit  UOG_ASYNCLISTADDR_bit
__IO_REG32_BIT(UOG_BURSTSIZE,             0x73F80160,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x73F80164,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UOG_IC_USB,                0x73F8016C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UOG_ULPIVIEW,              0x73F80170,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32(    UOG_CFGFLAG,               0x73F80180,__READ       );
__IO_REG32_BIT(UOG_PORTSC1,               0x73F80184,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x73F801A4,__READ_WRITE ,__uog_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x73F801A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x73F801AC,__READ_WRITE ,__uog_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x73F801B0,__READ_WRITE ,__uog_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x73F801B4,__READ_WRITE ,__uog_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x73F801B8,__READ       ,__uog_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x73F801BC,__READ_WRITE ,__uog_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x73F801C0,__READ_WRITE ,__uog_endptctrl0_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x73F801C4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x73F801C8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x73F801CC,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x73F801D0,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x73F801D4,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x73F801D8,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x73F801DC,__READ_WRITE ,__uog_endptctrl_bits);
__IO_REG32_BIT(UH1_ID,                    0x73F80200,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH1_HWGENERAL,             0x73F80204,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH1_HWHOST,                0x73F80208,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH1_HWTXBUF,               0x73F80210,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH1_HWRXBUF,               0x73F80214,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH1_GPTIMER0LD,            0x73F80280,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH1_GPTIMER0CTRL,          0x73F80284,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH1_GPTIMER1LD,            0x73F80288,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH1_GPTIMER1CTRL,          0x73F8028C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG8(     UH1_CAPLENGTH,             0x73F80300,__READ       );
__IO_REG16(    UH1_HCIVERSION,            0x73F80302,__READ       );
__IO_REG32_BIT(UH1_HCSPARAMS,             0x73F80304,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH1_HCCPARAMS,             0x73F80308,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH1_USBCMD,                0x73F80340,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH1_USBSTS,                0x73F80344,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH1_USBINTR,               0x73F80348,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH1_FRINDEX,               0x73F8034C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH1_PERIODICLISTBASE,      0x73F80354,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH1_ASYNCLISTADDR,         0x73F80358,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH1_BURSTSIZE,             0x73F80360,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH1_TXFILLTUNING,          0x73F80364,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH1_IC_USB,                0x73F8036C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH1_ULPIVIEW,              0x73F80370,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH1_PORTSC1,               0x73F80384,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH1_USBMODE,               0x73F803A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UH2_ID,                    0x73F80400,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH2_HWGENERAL,             0x73F80404,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH2_HWHOST,                0x73F80408,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH2_HWTXBUF,               0x73F80410,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH2_HWRXBUF,               0x73F80414,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH2_GPTIMER0LD,            0x73F80480,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH2_GPTIMER0CTRL,          0x73F80484,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH2_GPTIMER1LD,            0x73F80488,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH2_GPTIMER1CTRL,          0x73F8048C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG8(     UH2_CAPLENGTH,             0x73F80500,__READ       );
__IO_REG16(    UH2_HCIVERSION,            0x73F80502,__READ       );
__IO_REG32_BIT(UH2_HCSPARAMS,             0x73F80504,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH2_HCCPARAMS,             0x73F80508,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH2_USBCMD,                0x73F80540,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH2_USBSTS,                0x73F80544,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH2_USBINTR,               0x73F80548,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH2_FRINDEX,               0x73F8054C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH2_PERIODICLISTBASE,      0x73F80554,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH2_ASYNCLISTADDR,         0x73F80558,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH2_BURSTSIZE,             0x73F80560,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH2_TXFILLTUNING,          0x73F80564,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH2_IC_USB,                0x73F8056C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH2_ULPIVIEW,              0x73F80570,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH2_PORTSC1,               0x73F80584,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH2_USBMODE,               0x73F805A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(UH3_ID,                    0x73F80600,__READ       ,__uog_id_bits);
__IO_REG32_BIT(UH3_HWGENERAL,             0x73F80604,__READ       ,__uog_hwgeneral_bits);
__IO_REG32_BIT(UH3_HWHOST,                0x73F80608,__READ       ,__uog_hwhost_bits);
__IO_REG32_BIT(UH3_HWTXBUF,               0x73F80610,__READ       ,__uog_hwtxbuf_bits);
__IO_REG32_BIT(UH3_HWRXBUF,               0x73F80614,__READ       ,__uog_hwrxbuf_bits);
__IO_REG32_BIT(UH3_GPTIMER0LD,            0x73F80680,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH3_GPTIMER0CTRL,          0x73F80684,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG32_BIT(UH3_GPTIMER1LD,            0x73F80688,__READ_WRITE ,__uh_gptimerld_bits);
__IO_REG32_BIT(UH3_GPTIMER1CTRL,          0x73F8068C,__READ_WRITE ,__uh_gptimerctrl_bits);
__IO_REG8(     UH3_CAPLENGTH,             0x73F80700,__READ       );
__IO_REG16(    UH3_HCIVERSION,            0x73F80702,__READ       );
__IO_REG32_BIT(UH3_HCSPARAMS,             0x73F80704,__READ       ,__uog_hcsparams_bits);
__IO_REG32_BIT(UH3_HCCPARAMS,             0x73F80708,__READ       ,__uog_hccparams_bits);
__IO_REG32_BIT(UH3_USBCMD,                0x73F80740,__READ_WRITE ,__uog_usbcmd_bits);
__IO_REG32_BIT(UH3_USBSTS,                0x73F80744,__READ_WRITE ,__uog_usbsts_bits);
__IO_REG32_BIT(UH3_USBINTR,               0x73F80748,__READ_WRITE ,__uog_usbintr_bits);
__IO_REG32_BIT(UH3_FRINDEX,               0x73F8074C,__READ_WRITE ,__uog_frindex_bits);
__IO_REG32_BIT(UH3_PERIODICLISTBASE,      0x73F80754,__READ_WRITE ,__uog_periodiclistbase_bits);
__IO_REG32_BIT(UH3_ASYNCLISTADDR,         0x73F80758,__READ_WRITE ,__uog_asynclistaddr_bits);
__IO_REG32_BIT(UH3_BURSTSIZE,             0x73F80760,__READ_WRITE ,__uog_burstsize_bits);
__IO_REG32_BIT(UH3_TXFILLTUNING,          0x73F80764,__READ_WRITE ,__uog_txfilltuning_bits);
__IO_REG32_BIT(UH3_IC_USB,                0x73F8076C,__READ_WRITE ,__uog_ic_usb_bits);
__IO_REG32_BIT(UH3_ULPIVIEW,              0x73F80770,__READ_WRITE ,__uog_viewport_bits);
__IO_REG32_BIT(UH3_PORTSC1,               0x73F80784,__READ_WRITE ,__uog_portsc_bits);
__IO_REG32_BIT(UH3_USBMODE,               0x73F807A8,__READ_WRITE ,__uog_usbmode_bits);
__IO_REG32_BIT(USB_CTRL,                  0x73F80800,__READ_WRITE ,__usb_ctrl_bits);
__IO_REG32_BIT(USB_OTG_MIRROR,            0x73F80804,__READ_WRITE ,__usb_otg_mirror_bits);
__IO_REG32_BIT(USB_PHY_CTRL_0,            0x73F80808,__READ_WRITE ,__usb_otg_phy_ctrl_0_bits);
__IO_REG32_BIT(USB_PHY_CTRL_1,            0x73F8080C,__READ_WRITE ,__usb_otg_phy_ctrl_1_bits);
__IO_REG32_BIT(USB_CTRL_1,                0x73F80810,__READ_WRITE ,__usb_ctrl_1_bits);
__IO_REG32_BIT(USB_UH2_CTRL,              0x73F80814,__READ_WRITE ,__usb_uh2_ctrl_bits);
__IO_REG32_BIT(USB_UH3_CTRL,              0x73F80818,__READ_WRITE ,__usb_uh3_ctrl_bits);

/***************************************************************************
 **
 ** TIGERP
 **
 ***************************************************************************/
__IO_REG32_BIT(TIGERP_PVID,               0x83FA0000,__READ       ,__tigerp_pvid_bits);
__IO_REG32_BIT(TIGERP_GPC,                0x83FA0004,__READ_WRITE ,__tigerp_gpc_bits);
__IO_REG32_BIT(TIGERP_PIC,                0x83FA0008,__READ_WRITE ,__tigerp_pic_bits);
__IO_REG32_BIT(TIGERP_LPC,                0x83FA000C,__READ_WRITE ,__tigerp_lpc_bits);
__IO_REG32_BIT(TIGERP_NLPC,               0x83FA0010,__READ_WRITE ,__tigerp_nlpc_bits);
__IO_REG32_BIT(TIGERP_ICGC,               0x83FA0014,__READ_WRITE ,__tigerp_icgc_bits);
__IO_REG32_BIT(TIGERP_AMC,                0x83FA0018,__READ_WRITE ,__tigerp_amc_bits);
__IO_REG32_BIT(TIGERP_NMC,                0x83FA0020,__READ_WRITE ,__tigerp_nmc_bits);
__IO_REG32_BIT(TIGERP_NMS,                0x83FA0024,__READ_WRITE ,__tigerp_nms_bits);

/***************************************************************************
 **
 **  SIM
 **
 ***************************************************************************/
__IO_REG32_BIT(SIM_PORT1_CNTL,            0x83FE4000,__READ_WRITE ,__sim_port1_cntl_bits);
__IO_REG32_BIT(SIM_SETUP,                 0x83FE4004,__READ_WRITE ,__sim_setup_bits);
__IO_REG32_BIT(SIM_PORT1_DETECT,          0x83FE4008,__READ_WRITE ,__sim_port1_detect_bits);
__IO_REG32_BIT(SIM_XMT_BUF,               0x83FE400C,__READ_WRITE ,__sim_xmt_buf_bits);
__IO_REG32_BIT(SIM_RCV_BUF,               0x83FE4010,__READ       ,__sim_rcv_buf_bits);
__IO_REG32_BIT(SIM_PORT0_CNTL,            0x83FE4014,__READ_WRITE ,__sim_port0_cntl_bits);
__IO_REG32_BIT(SIM_CNTL,                  0x83FE4018,__READ_WRITE ,__sim_cntl_bits);
__IO_REG32_BIT(SIM_CLK_PRESCALER,         0x83FE401C,__READ_WRITE ,__sim_clk_prescaler_bits);
__IO_REG32_BIT(SIM_RCV_THRESHOLD,         0x83FE4020,__READ_WRITE ,__sim_rcv_threshold_bits);
__IO_REG32_BIT(SIM_ENABLE,                0x83FE4024,__READ_WRITE ,__sim_enable_bits);
__IO_REG32_BIT(SIM_XMT_STATUS,            0x83FE4028,__READ_WRITE ,__sim_xmt_status_bits);
__IO_REG32_BIT(SIM_RCV_STATUS,            0x83FE402C,__READ_WRITE ,__sim_rcv_status_bits);
__IO_REG32_BIT(SIM_INT_MASK,              0x83FE4030,__READ_WRITE ,__sim_int_mask_bits);
__IO_REG32_BIT(SIM_PORT0_DETECT,          0x83FE403C,__READ_WRITE ,__sim_port0_detect_bits);
__IO_REG32_BIT(SIM_DATA_FORMAT,           0x83FE4040,__READ_WRITE ,__sim_data_format_bits);
__IO_REG32_BIT(SIM_XMT_THRESHOLD,         0x83FE4044,__READ_WRITE ,__sim_xmt_threshold_bits);
__IO_REG32_BIT(SIM_GUARD_CNTL,            0x83FE4048,__READ_WRITE ,__sim_guard_cntl_bits);
__IO_REG32_BIT(SIM_OD_CONFIG,             0x83FE404C,__READ_WRITE ,__sim_od_config_bits);
__IO_REG32_BIT(SIM_RESET_CNTL,            0x83FE4050,__READ_WRITE ,__sim_reset_cntl_bits);
__IO_REG16(    SIM_CHAR_WAIT,             0x83FE4054,__READ_WRITE );
__IO_REG16(    SIM_GPCNT,                 0x83FE4058,__READ_WRITE );
__IO_REG32_BIT(SIM_DIVISOR,               0x83FE405C,__READ_WRITE ,__sim_divisor_bits);
__IO_REG16(    SIM_BWT,                   0x83FE4060,__READ_WRITE );
__IO_REG16(    SIM_BGT,                   0x83FE4064,__READ_WRITE );
__IO_REG16(    SIM_BWT_H,                 0x83FE4068,__READ_WRITE );
__IO_REG32_BIT(SIM_XMT_FIFO_STAT,         0x83FE406C,__READ       ,__sim_xmt_fifo_stat_bits);
__IO_REG32_BIT(SIM_RCV_FIFO_CNT,          0x83FE4070,__READ       ,__sim_rcv_fifo_cnt_bits);
__IO_REG32_BIT(SIM_RCV_FIFO_WPTR,         0x83FE4074,__READ       ,__sim_rcv_fifo_wptr_bits);
__IO_REG32_BIT(SIM_RCV_FIFO_RPTR,         0x83FE4078,__READ       ,__sim_rcv_fifo_rptr_bits);

/***************************************************************************
 **
 **  HSC
 **
 ***************************************************************************/
__IO_REG32(    HSC_MCD,                  0x83FDC000,__READ_WRITE );
__IO_REG32(    HSC_MCCMC,                0x83FDC0D8,__READ_WRITE );
__IO_REG32(    HSC_MXT,                  0x83FDC800,__READ_WRITE );

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
 **   MCIMX515 interrupt sources
 **
 ***************************************************************************/
#define INT_ESDHC_1            1              /* Enhanced SDHC Interrupt Request */
#define INT_ESDHC_2            2              /* Enhanced SDHC Interrupt Request */
#define INT_ESDHC_3            3              /* CE-ATA Interrupt Request based on eSDHC-3 */
#define INT_ESDHC_4            4              /* Enhanced SDHC Interrupt Request */
#define INT_DAP                5              /* Power-up Request */
#define INT_SDMA               6              /* “AND” of all 48 interrupts from all the channels */
#define INT_IOMUX              7              /* POWER FAIL interrupt - this is a power fail indicator interrupt from on board power management IC via GPIO_16 PAD on ALT2. */
#define INT_EMI_NFC            8              /* NFC interrupt */
#define INT_VPU                9              /* VPU Interrupt Request */
#define INT_IPUEX_ERR          10             /* IPUEX Error Interrupt */
#define INT_IPUEX_SYNC         11             /* IPUEX Sync Interrupt */
#define INT_GPU3D              12             /* GPU Interrupt Request */
#define INT_USBOH3_HOST1       14             /* USB Host 1 */
#define INT_EMI                15             /* Consolidated EMI Interrupt */
#define INT_USBOH3_HOST2       16             /* USB Host 2 */
#define INT_USBOH3_HOST3       17             /* USB Host 3 */
#define INT_USBOH3_OTG         18             /* USB OTG */
#define INT_SAHARA_HOST0       19             /* SAHARA Interrupt for Host 0 */
#define INT_SAHARA_HOST1       20             /* SAHARA Interrupt for Host 1 */
#define INT_SCC_HP             21             /* Security Monitor High Priority Interrupt Request */
#define INT_SCC_TZ             22             /* Secure (TrustZone) Interrupt Request. */
#define INT_SCC_NS             23             /* Regular (Non-Secure) Interrupt Request */
#define INT_SRTC_NTZ           24             /* SRTC Consolidated Interrupt. Non TZ. */
#define INT_SRTC_TZ            25             /* SRTC Security Interrupt. TZ. */
#define INT_RTIC               26             /* RTIC (Trust Zone) Interrupt Request. */
#define INT_CSU_1              27             /* CSU Interrupt Request 1. */
#define INT_SSI_1              29             /* SSI-1 Interrupt Request */
#define INT_SSI_2              30             /* SSI-2 Interrupt Request */
#define INT_UART_1             31             /* UART-1 ORed interrupt */
#define INT_UART_2             32             /* UART-2 ORed interrupt */
#define INT_UART_3             33             /* UART-3 ORed interrupt */
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
#define INT_PWM_1              61             /* Cumulative interrupt line. */
#define INT_I2C_1              62             /* I2C-1 Interrupt */
#define INT_I2C_2              63             /* I2C-2 Interrupt */
#define INT_HS_I2C             64             /* HS-I2C Interrupt */
#define INT_SIM_1              67             /* SIM interrupt composed of oef, xte, sdi1, and sdi0 */
#define INT_SIM_2              68             /* SIM interrupt composed of tc, etc, tfe, and rdrf */
#define INT_IIM                69             /* Interrupt request to the processor */
#define INT_PATA               70             /* Parallel ATA host controller interrupt request */
#define INT_CCM_1              71             /* CCM, Interrupt Request 1 */
#define INT_CCM_2              72             /* CCM, Interrupt Request 2 */
#define INT_GPC_1              73             /* GPC, Interrupt Request 1 */
#define INT_GPC_2              74             /* GPC, Interrupt Request 2 */
#define INT_SRC                75             /* SRC interrupt request */
#define INT_TIGERP_PLATFORM_NE_32K_256K_M      76  /* Neon Monitor Interrupt */
#define INT_TIGERP_PLATFORM_NE_32K_256K_PU     77  /* Performance Unit Interrupt (nPMUIRQ) */
#define INT_TIGERP_PLATFORM_NE_32K_256K_CTI    78  /* CTI IRQ */
#define INT_TIGERP_PLATFORM_NE_32K_256K_DCT11  79  /* Debug Interrupt, from Cross-Trigger 1 Interface 1 */
#define INT_TIGERP_PLATFORM_NE_32K_256K_DCT10  80  /* Debug Interrupt, from Cross-Trigger 1 Interface 0 */
#define INT_GPU2D              84             /* GPU2D (OpenVG) general interrupt */
#define INT_GPU2D_BUSY         85             /* GPU2D (OpenVG) busy signal (for S/W power gating feasibility) */
#define INT_FEC                87             /* Fast Ethernet Controller Interrupt request (OR of 13 interrupt sources) */
#define INT_OWIRE              88             /* 1-Wire Interrupt Request */
#define INT_TIGERP_PLATFORM_NE_32K_256K_DCT12  89  /* Debug Interrupt, from Cross-Trigger 1 Interface 2 */
#define INT_SJC                90             /* */
#define INT_SPDIF              91             /* */
#define INT_TVE                92             /* */
#define INT_FIRI               93             /* FIRI Intr (OR of all 4 interrupt sources) */
#define INT_PWM_2              94             /* Cumulative interrupt line*/
#define INT_SSI_3              96             /* SSI-3 Interrupt Request */
#define INT_EMI_BOOT           97             /* Boot sequence completed interrupt request */
#define INT_TIGERP_PLATFORM_NE_32K_256K_DCT13  98  /* Debug Interrupt, from Cross-Trigger 1 Interface 3 */
#define INT_VPU_IDLE           100            /* Idle interrupt from VPU (for S/W power gating) */
#define INT_EMI_AUTO_PROG      101            /* Indicates all pages have been transferred to NFC during an auto_prog operation */
#define INT_GPU3D_IDLE         102            /* Idle interrupt from GPU3D (for S/W power gating) */

#endif    /* __MCIMX515_H */
