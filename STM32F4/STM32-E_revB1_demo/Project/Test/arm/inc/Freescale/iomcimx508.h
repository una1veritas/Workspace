/***************************************************************************
 **
 **    This file defines the Special Function Register for
 **    Freescale MCIMX508
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __IOMCIMX508_H
#define __IOMCIMX508_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIMX508 SPECIAL FUNCTION REGISTERS
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

/* CCM Control Register (CCM_CCR) */
typedef struct {
__REG32 OSCNT     : 8;
__REG32           : 1;
__REG32 CAMP1_EN  : 1;
__REG32           : 2;
__REG32 COSC_EN   : 1;
__REG32           :19;
} __ccm_ccr_bits;

/* CCM Control Divider Register (CCM_CCDR) */
typedef struct {
__REG32                   :16;
__REG32 EMI_HS_MASK       : 1;
__REG32 EMI_HS_FAST_MASK  : 1;
__REG32 EMI_HS_SLOW_MASK  : 1;
__REG32 EMI_HS_INT1_MASK  : 1;
__REG32 EMI_HS_INT2_MASK  : 1;
__REG32 IPU_HS_MASK       : 1;
__REG32                   :10;
} __ccm_ccdr_bits;

/* CCM Status Register(CCM_CSR) */
typedef struct {
__REG32 REF_EN_B        : 1;
__REG32 TEMP_MON_ALARM  : 1;
__REG32 CAMP1_READY     : 1;
__REG32                 : 1;
__REG32 LVS_VALUE       : 1;
__REG32 COSC_READY      : 1;
__REG32                 :26;
} __ccm_csr_bits;

/* CCM Clock Switcher Register (CCM_CCSR) */
typedef struct {
__REG32 PLL3_SW_CLK_SEL : 1;
__REG32 PLL2_SW_CLK_SEL : 1;
__REG32 PLL1_SW_CLK_SEL : 1;
__REG32 PLL3_DIV_PODF   : 2;
__REG32 PLL2_DIV_PODF   : 2;
__REG32 STEP_SEL        : 2;
__REG32                 : 1;
__REG32 LP_APM          : 1;
__REG32 PLL1_PFD_EN     : 1;
__REG32 PLL2_PFD_EN     : 1;
__REG32 PLL3_PFD_EN     : 1;
__REG32                 :18;
} __ccm_ccsr_bits;

/* CCM Arm Clock Root Register (CCM_CACRR) */
typedef struct {
__REG32 ARM_PODF        : 3;
__REG32                 :29;
} __ccm_cacrr_bits;

/* CCM Bus Clock Divider Register(CCM_CBCDR) */
typedef struct {
__REG32 PERCLK_PODF     : 3;
__REG32 PERCLK_PRED2    : 3;
__REG32 PERCLK_PRED1    : 2;
__REG32 IPG_PODF        : 2;
__REG32 AHB_PODF        : 3;
__REG32                 : 3;
__REG32 AXI_A_PODF      : 3;
__REG32 AXI_B_PODF      : 3;
__REG32 WEIM_PODF       : 3;
__REG32 PERIPH_CLK_SEL  : 2;
__REG32 WEIM_CLK_SEL    : 1;
__REG32                 : 4;
} __ccm_cbcdr_bits;

/* CCM Bus Clock Multiplexer Register (CCM_CBCMR) */
typedef struct {
__REG32 PERCLK_IPG_SEL    : 1;
__REG32 PERCLK_LP_APM_SEL : 1;
__REG32 DEBUG_APB_CLK_SEL : 2;
__REG32                   :12;
__REG32 GPU2D_CLK_SEL     : 2;
__REG32                   :14;
} __ccm_cbcmr_bits;

/* CCM Serial Clock Multiplexer Register 1 (CCM_CSCMR1) */
typedef struct {
__REG32 SSI_EXT1_COM        : 1;
__REG32 SSI_EXT2_COM        : 1;
__REG32                     : 2;
__REG32 ECSPI_CLK_SEL       : 2;
__REG32                     : 2;
__REG32 SSI_APM_CLK_SEL     : 1;
__REG32                     : 3;
__REG32 SSI2_CLK_SEL        : 2;
__REG32 SSI1_CLK_SEL        : 2;
__REG32 ESDHC3_CLK_SEL      : 3;
__REG32 ESDHC4_CLK_SEL      : 1;
__REG32 ESDHC2_CLK_SEL      : 1;
__REG32 ESDHC1_CLK_SEL      : 2;
__REG32                     : 1;
__REG32 UART_CLK_SEL        : 2;
__REG32                     : 2;
__REG32 SSI_EXT1_CLK_SEL    : 2;
__REG32 SSI_EXT2_CLK_SEL    : 2;
} __ccm_cscmr1_bits;

/* CCM Serial Clock Multiplexer Register 2 (CCM_CSCMR2) */
typedef struct {
__REG32 SPDIF0_CLK_SEL      : 2;
__REG32                     : 2;
__REG32 SPDIF0_COM          : 1;
__REG32                     : 1;
__REG32 CAN_CLK_SEL         : 2;
__REG32 LDB_DI0_CLK_SEL     : 1;
__REG32 LDB_DI1_CLK_SEL     : 1;
__REG32 LDB_DI0_IPU_DIV     : 1;
__REG32 LDB_DI1_IPU_DIV     : 1;
__REG32 FIRI_CLK_SEL        : 2;
__REG32 IEEE_CLK_SEL        : 2;
__REG32 ESAI_POST_SEL       : 3;
__REG32 ESAI_PRE_SEL        : 2;
__REG32 ASRC_CLK_SEL        : 1;
__REG32 CSI_MCLK1_CLK_SEL   : 2;
__REG32 CSI_MCLK2_CLK_SEL   : 2;
__REG32 DI0_CLK_SEL         : 3;
__REG32 DI1_CLK_SEL         : 3;
} __ccm_cscmr2_bits;

/* CCM Serial Clock Divider Register 1 (CCM_CSCDR1) */
typedef struct {
__REG32 UART_CLK_PODF   : 3;
__REG32 UART_CLK_PRED   : 3;
__REG32                 : 5;
__REG32 ESDHC1_CLK_PODF : 3;
__REG32 PGC_CLK_PODF    : 2;
__REG32 ESDHC1_CLK_PRED : 3;
__REG32 ESDHC3_CLK_PODF : 3;
__REG32 ESDHC3_CLK_PRED : 3;
__REG32                 : 7;
} __ccm_cscdr1_bits;

/* CCM SSI1 Clock Divider Register(CCM_CS1CDR) */
typedef struct {
__REG32 SSI1_CLK_PODF     : 6;
__REG32 SSI1_CLK_PRED     : 3;
__REG32                   : 7;
__REG32 SSI_EXT1_CLK_PODF : 6;
__REG32 SSI_EXT1_CLK_PRED : 3;
__REG32                   : 7;
} __ccm_cs1cdr_bits;

/* CCM SSI2 Clock Divider Register(CCM_CS2CDR) */
typedef struct {
__REG32 SSI2_CLK_PODF     : 6;
__REG32 SSI2_CLK_PRED     : 3;
__REG32                   : 7;
__REG32 SSI_EXT2_CLK_PODF : 6;
__REG32 SSI_EXT2_CLK_PRED : 3;
__REG32                   : 7;
} __ccm_cs2cdr_bits;

/* CCM DI Clock Divider Register(CCM_CDCDR) */
typedef struct {
__REG32 USB_PHY_PODF      : 3;
__REG32 USB_PHY_PRED      : 3;
__REG32 DI1_CLK_PRED      : 3;
__REG32                   : 7;
__REG32 DI_PLL4_PODF      : 3;
__REG32 SPDIF0_CLK_PODF   : 6;
__REG32 SPDIF0_CLK_PRED   : 3;
__REG32 TVE_CLK_PRED      : 3;
__REG32                   : 1;
} __ccm_cdcdr_bits;

/* CCM HSC Clock Divider Register(CCM_CHSCCDR) */
typedef struct {
__REG32 SSI1_MLB_SPDIF_SRC  : 2;
__REG32 SSI2_MLB_SPDIF_SRC  : 2;
__REG32                     :28;
} __ccm_chsccdr_bits;

/* CCM Serial Clock Divider Register 2(CCM_CSCDR2) */
typedef struct {
__REG32                     :19;
__REG32 ECSPI_CLK_PODF      : 6;
__REG32 ECSPI_CLK_PRED      : 3;
__REG32                     : 4;
} __ccm_cscdr2_bits;

/* CCM Serial Clock Divider Register 3(CCM_CSCDR3) */
typedef struct {
__REG32 FIRI_CLK_PODF       : 6;
__REG32 FIRI_CLK_PRED       : 3;
__REG32                     :23;
} __ccm_cscdr3_bits;

/* CCM Serial Clock Divider Register 4(CCM_CSCDR4) */
typedef struct {
__REG32 CSI_MCLK1_CLK_PODF  : 6;
__REG32 CSI_MCLK1_CLK_PRED  : 3;
__REG32 CSI_MCLK2_CLK_PODF  : 6;
__REG32                     : 1;
__REG32 CSI_MCLK2_CLK_PRED  : 3;
__REG32                     :13;
} __ccm_cscdr4_bits;

/* CCM Divider Handshake In-Process Register(CCM_CDHIPR) */
typedef struct {
__REG32 AXI_A_PODF_BUSY     : 1;
__REG32 AXI_B_PODF_BUSY     : 1;
__REG32 WEIM_PODF_BUSY      : 1;
__REG32 AHB_PODF_BUSY       : 1;
__REG32                     : 1;
__REG32 PERIPH_CLK_SEL_BUSY : 1;
__REG32 WEIM_CLK_SEL_BUSY   : 1;
__REG32                     : 9;
__REG32 ARM_PODF_BUSY       : 1;
__REG32                     :15;
} __ccm_cdhipr_bits;

/* CCM DVFS Control Register(CCM_CDCR) */
typedef struct {
__REG32 PERIPH_CLK_DVFS_PODF          : 2;
__REG32 ARM_FREQ_SHIFT_DIVIDER        : 1;
__REG32                               : 2;
__REG32 SOFTWARE_DVFS_EN              : 1;
__REG32 SW_PERIPH_CLK_DIV_REQ         : 1;
__REG32 SW_PERIPH_CLK_DIV_REQ_STATUS  : 1;
__REG32                               :24;
} __ccm_cdcr_bits;

/* CCM Testing Observability Register(CCM_CTOR) */
typedef struct {
__REG32 OBS_SPARE_OUTPUT_2_SEL        : 4;
__REG32 OBS_SPARE_OUTPUT_1_SEL        : 4;
__REG32 OBS_SPARE_OUTPUT_0_SEL        : 5;
__REG32 OBS_EN                        : 1;
__REG32                               :18;
} __ccm_ctor_bits;

/* CCM Low Power Control Register(CCM_CLPCR) */
typedef struct {
__REG32 LPM                           : 2;
__REG32 BYPASS_PMIC_VFUNCTIONAL_READY : 1;
__REG32                               : 2;
__REG32 ARM_CLK_DIS_ON_LPM            : 1;
__REG32 SBYOS                         : 1;
__REG32 DIS_REF_OSC                   : 1;
__REG32 VSTBY                         : 1;
__REG32 STBY_COUNT                    : 2;
__REG32 COSC_PWRDOWN                  : 1;
__REG32                               : 7;
__REG32 BYPASS_WEIM_LPM_HS            : 1;
__REG32                               : 3;
__REG32 BYPASS_RNGB_LPM_HS            : 1;
__REG32 BYPASS_SDMA_LPM_HS            : 1;
__REG32 BYPASS_MAX_LPM_HS             : 1;
__REG32                               : 6;
} __ccm_clpcr_bits;

/* CCM Interrupt Status Register(CCM_CISR) */
typedef struct {
__REG32 LRF_PLL1              : 1;
__REG32 LRF_PLL2              : 1;
__REG32 LRF_PLL3              : 1;
__REG32                       : 1;
__REG32 CAMP1_READY           : 1;
__REG32                       : 1;
__REG32 COSC_READY            : 1;
__REG32                       : 9;
__REG32 DIVIDERS_LOADED       : 1;
__REG32 AXI_A_PODF_LOADED     : 1;
__REG32 AXI_B_PODF_LOADED     : 1;
__REG32 WEIM_PODF_LOADED      : 1;
__REG32 AHB_PODF_LOADED       : 1;
__REG32                       : 1;
__REG32 PERIPH_CLK_SEL_LOADED : 1;
__REG32 WEIM_CLK_SEL_LOADED   : 1;
__REG32                       : 1;
__REG32 TEMP_MON_ALARM        : 1;
__REG32 ARM_PODF_LOADED       : 1;
__REG32                       : 5;
} __ccm_cisr_bits;

/* CCM Interrupt Mask Register(CCM_CIMR) */
typedef struct {
__REG32 MASK_LRF_PLL1               : 1;
__REG32 MASK_LRF_PLL2               : 1;
__REG32 MASK_LRF_PLL3               : 1;
__REG32                             : 1;
__REG32 MASK_CAMP1_READY            : 1;
__REG32                             : 1;
__REG32 MASK_COSC_READY             : 1;
__REG32                             : 9;
__REG32 MASK_DIVIDERS_LOADED        : 1;
__REG32 MASK_AXI_A_PODF_LOADED      : 1;
__REG32 MASK_AXI_B_PODF_LOADED      : 1;
__REG32 MASK_WEIM_PODF_LOADED       : 1;
__REG32 MASK_AHB_PODF_LOADED        : 1;
__REG32                             : 1;
__REG32 MASK_PERIPH_CLK_SEL_LOADED  : 1;
__REG32 MASK_WEIM_CLK_SEL_LOADED    : 1;
__REG32                             : 1;
__REG32 MASK_TEMP_MON_ALARM         : 1;
__REG32 MASK_ARM_PODF_LOADED        : 1;
__REG32                             : 5;
} __ccm_cimr_bits;

/* CCM Clock Output Source Register (CCM_CCOSR) */
typedef struct {
__REG32 CKO1_SEL        : 4;
__REG32 CKO1_DIV        : 3;
__REG32 CKO1_EN         : 1;
__REG32 CKO1_SLOW_SEL   : 1;
__REG32                 : 7;
__REG32 CKO2_SEL        : 5;
__REG32 CKO2_DIV        : 3;
__REG32 CKO2_EN         : 1;
__REG32                 : 7;
} __ccm_ccosr_bits;

/* CCM General Purpose Register(CCM_CGPR) */
typedef struct {
__REG32                         : 4;
__REG32 efuse_prog_supply_gate  : 1;
__REG32                         :27;
} __ccm_cgpr_bits;

/* CCM Clock Gating Register(CCM_CCGR0) */
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

/* CCM Module Enable Override Register(CCM_CMEOR) */
typedef struct {
__REG32                     : 2;
__REG32 MOD_EN_OV_OWIRE     : 1;
__REG32                     : 1;
__REG32 MOD_EN_OV_ESDHC     : 1;
__REG32 MOD_EN_OV_GPT       : 1;
__REG32 MOD_EN_OV_EPIT      : 1;
__REG32                     : 1;
__REG32 MOD_EN_OV_DAP       : 1;
__REG32                     : 1;
__REG32 MOD_EN_OV_GPU2D     : 1;
__REG32                     :21;
} __ccm_cmeor_bits;

/* CCM Control Status Register 2 (CCM_CSR2) */
typedef struct {
__REG32 SYS_CLK_REF_XTAL_BUSY : 1;
__REG32 SYS_CLK_REF_PLL_BUSY  : 1;
__REG32 DDR_CLK_REF_PLL_BUSY  : 1;
__REG32 DISPLAY_AXI_BUSY      : 1;
__REG32 EPDC_AXI_BUSY         : 1;
__REG32 GPMI_BUSY             : 1;
__REG32 BCH_BUSY              : 1;
__REG32 MSHC_XMSCKI_BUSY      : 1;
__REG32 EPDC_PIX_BUSY         : 1;
__REG32 ELCDIF_PIX_BUSY       : 1;
__REG32 SYS_CLK_XTAL_ACTIVE   : 1;
__REG32 ELCDIF_ASM_ACTIVE     : 1;
__REG32 EPXP_ASM_ACTIVE       : 1;
__REG32 EPDC_ASM_ACTIVE       : 1;
__REG32                       :18;
} __ccm_csr2_bits;

/* CCM Clock Sequence Bypass (CCM_CLKSEQ_BYPASS) */
typedef struct {
__REG32 BYPASS_SYS_CLK0         : 1;
__REG32 BYPASS_SYS_CLK1         : 1;
__REG32 BYPASS_DISPLAY_AXI_CLK  : 2;
__REG32 BYPASS_EPDC_AXI_CLK     : 2;
__REG32 BYPASS_GPMI_CLK         : 2;
__REG32 BYPASS_BCH_CLK          : 2;
__REG32 BYPASS_MSHC_XMSCKI_CLK  : 2;
__REG32 BYPASS_EPDC_PIX_CLK     : 2;
__REG32 BYPASS_ELCDIF_PIX_CLK   : 2;
__REG32                         :16;
} __ccm_clkseq_bypass_bits;

/* CCM System Clock Register (CCM_CLK_SYS) */
typedef struct {
__REG32 SYS_DIV_PLL             : 6;
__REG32 SYS_DIV_XTAL            : 4;
__REG32                         :18;
__REG32 SYS_PLL_CLKGATE         : 2;
__REG32 SYS_XTAL_CLKGATE        : 2;
} __ccm_clk_sys_bits;

/* CCM DDR Clock Register (CCM_CLK_DDR) */
typedef struct {
__REG32 DDR_DIV_PLL             : 6;
__REG32 DDR_PFD_SEL             : 1;
__REG32                         :23;
__REG32 DDR_CLKGATE             : 2;
} __ccm_clk_ddr_bits;

/* CCM ELCDIF PIX Clock Serial Divide Register(CCM_ELCDIFPIX) */
typedef struct {
__REG32 ELCDIF_PIX_CLK_PODF     :12;
__REG32 ELCDIF_CLK_PRED         : 2;
__REG32                         :16;
__REG32 ELCDIF_PIX_CLKGATE      : 2;
} __ccm_elcdifpix_bits;

/* CCM EPDC PIX Clock Serial Divide Register (CCM_EPDCPIX) */
typedef struct {
__REG32 EPDC_PIX_CLK_PODF       :12;
__REG32 EPDC_CLK_PRED           : 2;
__REG32                         :16;
__REG32 EPDC_PIX_CLKGATE        : 2;
} __ccm_epdcpix_bits;

/* CCM DISPLAY_AXI Clock Divide Register(CCM_DISPLAY_AXI) */
typedef struct {
__REG32 DISPLAY_AXI_DIV         : 6;
__REG32 ELCDIF_ASM_SLOW_DIV     : 3;
__REG32 ELCDIF_ASM_EN           : 1;
__REG32 EPXP_ASM_SLOW_DIV       : 3;
__REG32 EPXP_ASM_EN             : 1;
__REG32                         :16;
__REG32 DISPLAY_AXI_CLKGATE     : 2;
} __ccm_display_axi_bits;

/* CCM EPDC_AXI Clock Divide Register (CCM_EPDC_AXI) */
typedef struct {
__REG32 EPDC_AXI_DIV            : 6;
__REG32 EPDC_ASM_SLOW_DIV       : 3;
__REG32 EPDC_ASM_EN             : 1;
__REG32                         :20;
__REG32 EPDC_AXI_CLKGATE        : 2;
} __ccm_epdc_axi_bits;

/* CCM GPMI Clock Divide Register (CCM_GPMI) */
typedef struct {
__REG32 GPMI_DIV                : 6;
__REG32                         :24;
__REG32 GPMI_CLKGATE            : 2;
} __ccm_gpmi_bits;

/* CCM BCH Clock Divide Register (CCM_BCH) */
typedef struct {
__REG32 BCH_DIV                 : 6;
__REG32                         :24;
__REG32 BCH_CLKGATE             : 2;
} __ccm_bch_bits;

/* CCM MSHC_XMSCKI Clock Divide Register(CCM_MSHC_XMSCKI) */
typedef struct {
__REG32 MSHC_XMSCKI_DIV         : 6;
__REG32                         :24;
__REG32 MSHC_XMSCKI_CLKGATE     : 2;
} __ccm_mshc_xmscki_bits;

/* Fractional Clock Control Register 0 (CCM_ANALOG_FRAC0n) */
typedef struct {
__REG32 PFD0_FRAC               : 6;
__REG32 PFD0_STABLE             : 1;
__REG32 PFD0_CLKGATE            : 1;
__REG32 PFD1_FRAC               : 6;
__REG32 PFD1_STABLE             : 1;
__REG32 PFD1_CLKGATE            : 1;
__REG32 PFD2_FRAC               : 6;
__REG32 PFD2_STABLE             : 1;
__REG32 PFD2_CLKGATE            : 1;
__REG32 PFD3_FRAC               : 6;
__REG32 PFD3_STABLE             : 1;
__REG32 PFD3_CLKGATE            : 1;
} __ccm_analog_frac0_bits;

/* Fractional Clock Control Register 1 (CCM_ANALOG_FRAC1n) */
typedef struct {
__REG32 PFD4_FRAC               : 6;
__REG32 PFD4_STABLE             : 1;
__REG32 PFD4_CLKGATE            : 1;
__REG32 PFD5_FRAC               : 6;
__REG32 PFD5_STABLE             : 1;
__REG32 PFD5_CLKGATE            : 1;
__REG32 PFD6_FRAC               : 6;
__REG32 PFD6_STABLE             : 1;
__REG32 PFD6_CLKGATE            : 1;
__REG32 PFD7_FRAC               : 6;
__REG32 PFD7_STABLE             : 1;
__REG32 PFD7_CLKGATE            : 1;
} __ccm_analog_frac1_bits;

/* Miscellaneous Register Description (CCM_ANALOG_MISCn) */
typedef struct {
__REG32 PLL_POWERUP             : 1;
__REG32                         : 6;
__REG32 PLL_HOLD_RING_OFF       : 1;
__REG32                         : 6;
__REG32 CHGR_DETECTED           : 1;
__REG32 CHGR_FORCE_DET          : 1;
__REG32 CHGR_DET_DISABLE        : 1;
__REG32                         : 1;
__REG32 REF_PWD                 : 1;
__REG32                         : 1;
__REG32 REF_SELFBIAS_OFF        : 1;
__REG32                         :11;
} __ccm_analog_misc_bits;

/* PLL Control Register (CCM_ANALOG_PLLCTRLn) */
typedef struct {
__REG32 LOCK_COUNT              :16;
__REG32 PFD_DISABLE_MASK        : 8;
__REG32                         : 6;
__REG32 FORCE_LOCK              : 1;
__REG32 LOCK                    : 1;
} __ccm_analog_pllctrl_bits;

/* Platform Version ID (ARM_PVID) */
typedef struct {
__REG32 ECO               : 8;
__REG32 MINOR             : 8;
__REG32 IMPL              : 8;
__REG32 SPEC              : 8;
} __arm_pvid_bits;

/* General Purpose Control (ARM_GPC) */
typedef struct {
__REG32 GPC               :16;
__REG32 DBGEN             : 1;
__REG32 ATRDY             : 1;
__REG32 NOCLKSTP          : 1;
__REG32                   :12;
__REG32 DBGACTIVE         : 1;
} __arm_gpc_bits;

/* Low Power Control (ARM_LPC) */
typedef struct {
__REG32 DSM               : 1;
__REG32 DBGDSM            : 1;
__REG32                   :30;
} __arm_lpc_bits;

/* NEON Low Power Control (ARM_NLPC) */
typedef struct {
__REG32 NEONRST           : 1;
__REG32                   :31;
} __arm_nlpc_bits;

/* Internal Clock Generation Control (ARM_ICGC) */
typedef struct {
__REG32 IPG_CLK_DIVR      : 3;
__REG32 IPG_PRLD          : 1;
__REG32 ACLK_DIVR         : 3;
__REG32 ACLK_PRLD         : 1;
__REG32 DT_CLK_DIVR       : 3;
__REG32 DT_PRLD           : 1;
__REG32                   :20;
} __arm_icgc_bits;

/* ARM Memory Configuration (ARM_AMC) */
typedef struct {
__REG32 ALP               : 3;
__REG32 ALPEN             : 1;
__REG32                   :28;
} __arm_amc_bits;

/* NEON Monitor Control (ARM_NMC) */
typedef struct {
__REG32                   :12;
__REG32 PL                : 8;
__REG32                   :10;
__REG32 NME               : 1;
__REG32 IE                : 1;
} __arm_nmc_bits;

/* NEON Monitor Status (ARM_NMS) */
typedef struct {
__REG32                   :31;
__REG32 NI                : 1;
} __arm_nms_bits;

/* Master Priority Register for Slave port n (AHBMAX_MPRn) */
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

/* General Purpose Control Register for Slave port n (AHBMAX_SGPCRn) */
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

/* General Purpose Control Register for Master port n (AHBMAX_MGPCRn) */
typedef struct {
__REG32 AULB      : 3;
__REG32           :29;
} __ahbmax_mgpcr_bits;

/* AHB to APBH Bridge Control and Status Register 0(APBH_CTRL0n) */
typedef struct {
__REG32 CLKGATE_CHANNEL   :16;
__REG32                   :12;
__REG32 APB_BURST_EN      : 1;
__REG32 AHB_BURST8_EN     : 1;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __apbh_ctrl0_bits;

/* AHB to APBH Bridge Control and Status Register 1(APBH_CTRL1n) */
typedef struct {
__REG32 CH0_CMDCMPLT_IRQ      : 1;
__REG32 CH1_CMDCMPLT_IRQ      : 1;
__REG32 CH2_CMDCMPLT_IRQ      : 1;
__REG32 CH3_CMDCMPLT_IRQ      : 1;
__REG32 CH4_CMDCMPLT_IRQ      : 1;
__REG32 CH5_CMDCMPLT_IRQ      : 1;
__REG32 CH6_CMDCMPLT_IRQ      : 1;
__REG32 CH7_CMDCMPLT_IRQ      : 1;
__REG32 CH8_CMDCMPLT_IRQ      : 1;
__REG32 CH9_CMDCMPLT_IRQ      : 1;
__REG32 CH10_CMDCMPLT_IRQ     : 1;
__REG32 CH11_CMDCMPLT_IRQ     : 1;
__REG32 CH12_CMDCMPLT_IRQ     : 1;
__REG32 CH13_CMDCMPLT_IRQ     : 1;
__REG32 CH14_CMDCMPLT_IRQ     : 1;
__REG32 CH15_CMDCMPLT_IRQ     : 1;
__REG32 CH0_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH1_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH2_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH3_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH4_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH5_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH6_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH7_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH8_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH9_CMDCMPLT_IRQ_EN   : 1;
__REG32 CH10_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH11_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH12_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH13_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH14_CMDCMPLT_IRQ_EN  : 1;
__REG32 CH15_CMDCMPLT_IRQ_EN  : 1;
} __apbh_ctrl1_bits;

/* AHB to APBH Bridge Control and Status Register 2(APBH_CTRL2n) */
typedef struct {
__REG32 CH0_ERROR_IRQ       : 1;
__REG32 CH1_ERROR_IRQ       : 1;
__REG32 CH2_ERROR_IRQ       : 1;
__REG32 CH3_ERROR_IRQ       : 1;
__REG32 CH4_ERROR_IRQ       : 1;
__REG32 CH5_ERROR_IRQ       : 1;
__REG32 CH6_ERROR_IRQ       : 1;
__REG32 CH7_ERROR_IRQ       : 1;
__REG32 CH8_ERROR_IRQ       : 1;
__REG32 CH9_ERROR_IRQ       : 1;
__REG32 CH10_ERROR_IRQ      : 1;
__REG32 CH11_ERROR_IRQ      : 1;
__REG32 CH12_ERROR_IRQ      : 1;
__REG32 CH13_ERROR_IRQ      : 1;
__REG32 CH14_ERROR_IRQ      : 1;
__REG32 CH15_ERROR_IRQ      : 1;
__REG32 CH0_ERROR_STATUS    : 1;
__REG32 CH1_ERROR_STATUS    : 1;
__REG32 CH2_ERROR_STATUS    : 1;
__REG32 CH3_ERROR_STATUS    : 1;
__REG32 CH4_ERROR_STATUS    : 1;
__REG32 CH5_ERROR_STATUS    : 1;
__REG32 CH6_ERROR_STATUS    : 1;
__REG32 CH7_ERROR_STATUS    : 1;
__REG32 CH8_ERROR_STATUS    : 1;
__REG32 CH9_ERROR_STATUS    : 1;
__REG32 CH10_ERROR_STATUS   : 1;
__REG32 CH11_ERROR_STATUS   : 1;
__REG32 CH12_ERROR_STATUS   : 1;
__REG32 CH13_ERROR_STATUS   : 1;
__REG32 CH14_ERROR_STATUS   : 1;
__REG32 CH15_ERROR_STATUS   : 1;
} __apbh_ctrl2_bits;

/* AHB to APBH Bridge Channel Register(APBH_CHANNEL_CTRLn) */
typedef struct {
__REG32 FREEZE_CHANNEL    :16;
__REG32 RESET_CHANNEL     :16;
} __apbh_channel_ctrl_bits;

/* AHB to APBH DMA Device Assignment Register(APBH_DEVSEL) */
typedef struct {
__REG32 CH0               : 2;
__REG32 CH1               : 2;
__REG32 CH2               : 2;
__REG32 CH3               : 2;
__REG32 CH4               : 2;
__REG32 CH5               : 2;
__REG32 CH6               : 2;
__REG32 CH7               : 2;
__REG32 CH8               : 2;
__REG32 CH9               : 2;
__REG32 CH10              : 2;
__REG32 CH11              : 2;
__REG32 CH12              : 2;
__REG32 CH13              : 2;
__REG32 CH14              : 2;
__REG32 CH15              : 2;
} __apbh_devsel_bits;

/* AHB to APBH DMA burst size (APBH_DMA_BURST_SIZE) */
typedef struct {
__REG32 CH0               : 2;
__REG32 CH1               : 2;
__REG32 CH2               : 2;
__REG32 CH3               : 2;
__REG32 CH4               : 2;
__REG32 CH5               : 2;
__REG32 CH6               : 2;
__REG32 CH7               : 2;
__REG32 CH8               : 2;
__REG32                   :14;
} __apbh_dma_burst_size_bits;

/* AHB to APBH DMA Debug Register (APBH_DEBUG) */
typedef struct {
__REG32 GPMI_ONE_FIFO     : 1;
__REG32                   :31;
} __apbh_debug_bits;

/* APBH DMA Channel n Command Register(APBH_CHn_CMDn) */
typedef struct {
__REG32 COMMAND           : 2;
__REG32 CHAIN             : 1;
__REG32 IRQONCMPLT        : 1;
__REG32 NANDLOCK          : 1;
__REG32 NANDWAIT4READY    : 1;
__REG32 SEMAPHORE         : 1;
__REG32 WAIT4ENDCMD       : 1;
__REG32 HALTONTERMINATE   : 1;
__REG32                   : 3;
__REG32 CMDWORDS          : 4;
__REG32 XFER_COUNT        :16;
} __apbh_chn_cmd_bits;

/* APBH DMA Channel n Semaphore Register(APBH_CHn_SEMAn) */
typedef struct {
__REG32 INCREMENT_SEMA    : 8;
__REG32                   : 8;
__REG32 PHORE             : 8;
__REG32                   : 8;
} __apbh_chn_sema_bits;

/* AHB to APBH DMA Channel n Debug Information(APBH_CHn_DEBUG1n) */
typedef struct {
__REG32 STATEMACHINE      : 5;
__REG32                   :15;
__REG32 WR_FIFO_FULL      : 1;
__REG32 WR_FIFO_EMPTY     : 1;
__REG32 RD_FIFO_FULL      : 1;
__REG32 RD_FIFO_EMPTY     : 1;
__REG32 NEXTCMDADDRVALID  : 1;
__REG32 LOCK              : 1;
__REG32 READY             : 1;
__REG32 SENSE             : 1;
__REG32 END               : 1;
__REG32 KICK              : 1;
__REG32 BURST             : 1;
__REG32 REQ               : 1;
} __apbh_chn_debug1_bits;

/* AHB to APBH DMA Channel n Debug Information(APBH_CHn_DEBUG2n) */
typedef struct {
__REG32 AHB_BYTES         :16;
__REG32 APB_BYTES         :16;
} __apbh_chn_debug2_bits;

/* APBH Bridge Version Register (APBH_VERSION) */
typedef struct {
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __apbh_version_bits;

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

/* Master Privilege Register 1 */
typedef struct{
__REG32                   :16;
__REG32 SDMA              : 4;
__REG32 CAAM              : 4;
__REG32 ARM_CORE          : 4;
__REG32 IMX               : 4;
} __aipstz_mpr1_bits;

/* Off-Platform Peripheral Access Control Register 1 (AIPSTZ1_OPACR1) */
typedef struct{
__REG32                 : 4;
__REG32 WDOG1           : 4;
__REG32 KPP             : 4;
__REG32 GPIO4           : 4;
__REG32 GPIO3           : 4;
__REG32 GPIO2           : 4;
__REG32 GPIO1           : 4;
__REG32 USBOH1          : 4;
} __aipstz1_opacr1_bits;

/* Off-Platform Peripheral Access Control Register 2 (AIPSTZ1_OPACR2) */
typedef struct{
__REG32 UART1           : 4;
__REG32 PWM2            : 4;
__REG32 PWM1            : 4;
__REG32                 : 4;
__REG32 EPIT1           : 4;
__REG32 IOMUXC          : 4;
__REG32 SRTC            : 4;
__REG32 GPT             : 4;
} __aipstz1_opacr2_bits;

/* Off-Platform Peripheral Access Control Register 3 (AIPSTZ1_OPACR3) */
typedef struct{
__REG32 GPIO5           : 4;
__REG32 GPC             : 4;
__REG32 CCM             : 4;
__REG32 SRC             : 4;
__REG32                 : 8;
__REG32 USBOH1          : 4;
__REG32 UART2           : 4;
} __aipstz1_opacr3_bits;

/* Off-Platform Peripheral Access Control Register 4 (AIPSTZ1_OPACR4) */
typedef struct{
__REG32                 : 4;
__REG32 RNGB_BLOCK      : 4;
__REG32                 : 4;
__REG32 UART4           : 4;
__REG32 I2C3            : 4;
__REG32                 : 8;
__REG32 GPIO6           : 4;
} __aipstz1_opacr4_bits;

/* Off-Platform Peripheral Access Control Register 5 (AIPSTZ1_OPACR5) */
typedef struct{
__REG32                 :28;
__REG32 SPBA            : 4;
} __aipstz1_opacr5_bits;

/* Off-Platform Peripheral Access Control Register 1 (AIPSTZ2_OPACR1) */
typedef struct{
__REG32                 : 8;
__REG32 AHBMAX          : 4;
__REG32 UART5           : 4;
__REG32                 : 4;
__REG32 DPLLIP3         : 4;
__REG32 DPLLIP2         : 4;
__REG32 DPLLIP1         : 4;
} __aipstz2_opacr1_bits;

/* Off-Platform Peripheral Access Control Register 2 (AIPSTZ2_OPACR2) */
typedef struct{
__REG32                 : 4;
__REG32 ROMCP           : 4;
__REG32                 : 4;
__REG32 SDMA            : 4;
__REG32 ECSPI2          : 4;
__REG32                 : 4;
__REG32 OWIRE           : 4;
__REG32 ARM_PLATFORM    : 4;
} __aipstz2_opacr2_bits;

/* Off-Platform Peripheral Access Control Register 3 (AIPSTZ2_OPACR3) */
typedef struct{
__REG32                 : 4;
__REG32 EIM             : 4;
__REG32                 : 4;
__REG32 AUDMUX          : 4;
__REG32 SSI1            : 4;
__REG32 I2C1            : 4;
__REG32 I2C2            : 4;
__REG32 CSPI            : 4;
} __aipstz2_opacr3_bits;

/* Off-Platform Peripheral Access Control Register 4 (AIPSTZ2_OPACR4) */
typedef struct{
__REG32                 :16;
__REG32 FEC             : 4;
__REG32                 :12;
} __aipstz2_opacr4_bits;

/* Hardware BCH ECC Accelerator Control Register */
typedef struct {        
__REG32 COMPLETE_IRQ          : 1;
__REG32                       : 1;
__REG32 DEBUG_STALL_IRQ       : 1;
__REG32 BM_ERROR_IRQ          : 1;
__REG32                       : 4;
__REG32 COMPLETE_IRQ_EN       : 1;
__REG32                       : 1;
__REG32 DEBUG_STALL_IRQ_EN    : 1;
__REG32                       : 5;
__REG32 M2M_ENABLE            : 1;
__REG32 M2M_ENCODE            : 1;
__REG32 M2M_LAYOUT            : 2;
__REG32                       : 2;
__REG32 DEBUGSYNDROME         : 1;
__REG32                       : 7;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __bch_ctrl_bits;

/* Hardware BCH ECC Accelerator Status Register 0 */
typedef struct {        
__REG32                       : 2;
__REG32 UNCORRECTABLE         : 1;
__REG32 CORRECTED             : 1;
__REG32 ALLONES               : 1;
__REG32                       : 3;
__REG32 STATUS_BLK0           : 8;
__REG32 COMPLETED_CE          : 4;
__REG32 HANDLE                :12;
} __bch_status0_bits;

/* Hardware BCH ECC Accelerator Mode Register */
typedef struct {        
__REG32 ERASE_THRESHOLD       : 8;
__REG32                       :24;
} __bch_mode_bits;

/* Hardware BCH ECC Accelerator Layout Select Register */
typedef struct {        
__REG32 CS0_SELEC             : 2;
__REG32 CS1_SELEC             : 2;
__REG32 CS2_SELEC             : 2;
__REG32 CS3_SELEC             : 2;
__REG32 CS4_SELEC             : 2;
__REG32 CS5_SELEC             : 2;
__REG32 CS6_SELEC             : 2;
__REG32 CS7_SELEC             : 2;
__REG32 CS8_SELEC             : 2;
__REG32 CS9_SELEC             : 2;
__REG32 CS10_SELEC            : 2;
__REG32 CS11_SELEC            : 2;
__REG32 CS12_SELEC            : 2;
__REG32 CS13_SELEC            : 2;
__REG32 CS14_SELEC            : 2;
__REG32 CS15_SELEC            : 2;
} __bch_layoutselect_bits;

/* Hardware BCH ECC Flash 0 Layout 0 Register */
typedef struct {        
__REG32 DATA0_SIZE            :10;
__REG32 GF13_0_GF14_1         : 1;
__REG32 ECC0                  : 5;
__REG32 META_SIZE             : 8;
__REG32 NBLOCKS               : 8;
} __bch_flashxlayout0_bits;

/* Hardware BCH ECC Flash 0 Layout 1 Register */
typedef struct {        
__REG32 DATAN_SIZE            :10;
__REG32 GF13_0_GF14_1         : 1;
__REG32 ECCN                  : 5;
__REG32 PAGE_SIZE             :16;
} __bch_flashxlayout1_bits;

/* Hardware BCH ECC Debug Register 0 */
typedef struct {        
__REG32 DEBUG_REG_SELECT          : 6;
__REG32                           : 2;
__REG32 BM_KES_TEST_BYPASS        : 1;
__REG32 KES_DEBUG_STALL           : 1;
__REG32 KES_DEBUG_STEP            : 1;
__REG32 KES_STANDALONE            : 1;
__REG32 KES_DEBUG_KICK            : 1;
__REG32 KES_DEBUG_MODE4K          : 1;
__REG32 KES_DEBUG_PAYLOAD_FLAG    : 1;
__REG32 KES_DEBUG_SHIFT_SYND      : 1;
__REG32 KES_DEBUG_SYNDROME_SYMBOL : 9;
__REG32                           : 7;
} __bch_debug0_bits;

/* Hardware BCH ECC Version Register */
typedef struct {        
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __bch_version_bits;

/* Control Register (CSPI_CONREG) */
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

/* Interrupt Control Register (CSPI_INTREG) */
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

/* DMA Control Register (CSPI_DMAREG) */
typedef struct{
__REG32 TEDEN  : 1;
__REG32 THDEN  : 1;
__REG32        : 2;
__REG32 RHDEN  : 1;
__REG32 RFDEN  : 1;
__REG32        :26;
} __cspi_dmareg_bits;

/* Status Register (CSPI_STATREG) */
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

/* Sample Period Control Register (CSPI_PERIODREG) */
typedef struct{
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32               :16;
} __cspi_periodreg_bits;

/* Test Control Register (CSPI_TESTREG) */
typedef struct{
__REG32 TXCNT     : 4;
__REG32 RXCNT     : 4;
__REG32           : 6;
__REG32 LBC       : 1;
__REG32 SWAP      : 1;
__REG32           :16;
} __cspi_testreg_bits;

/* DCP Control Register 0 (DCP_CTRL) */
typedef struct {
__REG32 CHANNEL_INTERRUPT_ENABLE        : 8;
__REG32                                 :13;
__REG32 ENABLE_CONTEXT_SWITCHING        : 1;
__REG32 ENABLE_CONTEXT_CACHING          : 1;
__REG32 GATHER_RESIDUAL_WRITES          : 1;
__REG32                                 : 4;
__REG32 PRESENT_SHA                     : 1;
__REG32 PRESENT_CRYPTO                  : 1;
__REG32 CLKGATE                         : 1;
__REG32 SFTRST                          : 1;
} __dcp_ctrl_bits;

/* DCP Status Register (DCP_STAT) */
typedef struct {
__REG32 IRQ                             : 4;
__REG32                                 :12;
__REG32 READY_CHANNELS                  : 8;
__REG32 CUR_CHANNEL                     : 4;
__REG32 OTP_KEY_READY                   : 1;
__REG32                                 : 3;
} __dcp_stat_bits;

/* DCP Channel Control Register (DCP_CHANNELCTRL) */
typedef struct {
__REG32 ENABLE_CHANNEL                  : 8;
__REG32 HIGH_PRIORITY_CHANNEL           : 8;
__REG32 CH0_IRQ_MERGED                  : 1;
__REG32                                 :15;
} __dcp_channelctrl_bits;

/* DCP Capability 0 Register (DCP_CAPABILITY0) */
typedef struct {
__REG32 NUM_KEYS                        : 8;
__REG32 NUM_CHANNELS                    : 4;
__REG32                                 :17;
__REG32 DISABLE_UNIQUE_KEY              : 1;
__REG32                                 : 1;
__REG32 DISABLE_DECRYPT                 : 1;
} __dcp_capability0_bits;

/* DCP Capability 1 Register (DCP_CAPABILITY1) */
typedef struct {
__REG32 CIPHER_ALGORITHMS               :16;
__REG32 HASH_ALGORITHMS                 :16;
} __dcp_capability1_bits;

/* DCP Key Index (DCP_KEY) */
typedef struct {
__REG32 SUBWORD                         : 2;
__REG32                                 : 2;
__REG32 INDEX                           : 2;
__REG32                                 :26;
} __dcp_key_bits;

/* DCP Work Packet 1 Status Register (DCP_PACKET1) */
typedef struct {
__REG32 INTERRUPT                       : 1;
__REG32 DECR_SEMAPHORE                  : 1;
__REG32 CHAIN                           : 1;
__REG32 CHAIN_CONTIGUOUS                : 1;
__REG32 ENABLE_MEMCOPY                  : 1;
__REG32 ENABLE_CIPHER                   : 1;
__REG32 ENABLE_HASH                     : 1;
__REG32 ENABLE_BLIT                     : 1;
__REG32 CIPHER_ENCRYPT                  : 1;
__REG32 CIPHER_INIT                     : 1;
__REG32 OTP_KEY                         : 1;
__REG32 PAYLOAD_KEY                     : 1;
__REG32 HASH_INIT                       : 1;
__REG32 HASH_TERM                       : 1;
__REG32 CHECK_HASH                      : 1;
__REG32 HASH_OUTPUT                     : 1;
__REG32 CONSTANT_FILL                   : 1;
__REG32 TEST_SEMA_IRQ                   : 1;
__REG32 KEY_BYTESWAP                    : 1;
__REG32 KEY_WORDSWAP                    : 1;
__REG32 INPUT_BYTESWAP                  : 1;
__REG32 INPUT_WORDSWAP                  : 1;
__REG32 OUTPUT_BYTESWAP                 : 1;
__REG32 OUTPUT_WORDSWAP                 : 1;
__REG32 TAG                             : 8;
} __dcp_packet1_bits;

/* DCP Work Packet 2 Status Register (DCP_PACKET2) */
typedef struct {
__REG32 CIPHER_SELECT                   : 4;
__REG32 CIPHER_MODE                     : 4;
__REG32 KEY_SELECT                      : 8;
__REG32 HASH_SELECT                     : 4;
__REG32                                 : 4;
__REG32 CIPHER_CFG                      : 8;
} __dcp_packet2_bits;

/* DCP Channel n Semaphore Register (DCP_CHnSEMA) */
typedef struct {
__REG32 INCREMEN                        : 8;
__REG32                                 : 8;
__REG32 VALUE                           : 8;
__REG32                                 : 8;
} __dcp_chsema_bits;

/* DCP Channel n Status Register (DCP_CHnSTAT) */
typedef struct {
__REG32                                 : 1;
__REG32 HASH_MISMATCH                   : 1;
__REG32 ERROR_SETUP                     : 1;
__REG32 ERROR_PACKET                    : 1;
__REG32 ERROR_SRC                       : 1;
__REG32 ERROR_DST                       : 1;
__REG32 ERROR_PAGEFAULT                 : 1;
__REG32                                 : 9;
__REG32 ERROR_CODE                      : 8;
__REG32 TAG                             : 8;
} __dcp_chstat_bits;

/* DCP Channel n Options Register (DCP_CH0nPTS) */
typedef struct {
__REG32 RECOVERY_TIMER                  :16;
__REG32                                 :16;
} __dcp_chopts_bits;

/* DCP Debug Select Register (DCP_DBGSELECT) */
typedef struct {
__REG32 INDEX                           : 8;
__REG32                                 :24;
} __dcp_dbgselect_bits;

/* DCP Page Table Register (DCP_PAGETABLE) */
typedef struct {
__REG32 ENABLE                          : 1;
__REG32 FLUSH                           : 1;
__REG32 BASE                            :30;
} __dcp_pagetable_bits;

/* DCP Version Register (DCP_VERSION) */
typedef struct {
__REG32 STEP                            :16;
__REG32 MINOR                           : 8;
__REG32 MAJOR                           : 8;
} __dcp_version_bits;

/* DIGCTL Control Register (DIGCTL_CTRLn) */
typedef struct {
__REG32 ESDHC_VERSION       : 1;
__REG32                     :29;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __digctl_ctrl_bits;       

/* DIGCTL OCRAM Register (DIGCTL_OCRAMn) */
typedef struct {
__REG32 RD_DATA_WAIT_EN       : 1;
__REG32 RD_ADDR_PIPE_EN       : 1;
__REG32 WR_DATA_PIPE_EN       : 1;
__REG32 WR_ADDR_PIPE_EN       : 1;
__REG32                       :12;
__REG32 RD_DATA_WAIT_EN_STAT  : 1;
__REG32 RD_ADDR_PIPE_EN_STAT  : 1;
__REG32 WR_DATA_PIPE_EN_STAT  : 1;
__REG32 WR_ADDR_PIPE_EN_STAT  : 1;
__REG32                       :12;
} __digctl_ocram_bits;

/* Transistor Speed Control Register (DIGCTL_SPEEDCTLn) */
typedef struct {
__REG32 CTRL                  : 2;
__REG32                       : 2;
__REG32 SELECT                : 4;
__REG32                       :24;
} __digctl_speedctl_bits;

/* DPLL Control Register(DPLLC_CTL) */
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
} __dpllc_ctl_bits;

/* DPLL Config Register(DPLLC_CONFIG) */
typedef struct{
__REG32 LDREQ       : 1;
__REG32 AREN        : 1;
__REG32             :30;
} __dpllc_config_bits;

/* DPLL Operation Register(DPLLC_OP) */
typedef struct{
__REG32 PDF         : 4;
__REG32 MFI         : 4;
__REG32             :24;
} __dpllc_op_bits;

/* DPLL Multiplication Factor Denominator Register(DPLLC_MFD) */
typedef struct{
__REG32 MFD         :27;
__REG32             : 5;
} __dpllc_mfd_bits;

/* DPLL MFNxxx Register (DPLLC_MFN) */
typedef struct{
__REG32 MFN         :27;
__REG32             : 5;
} __dpllc_mfn_bits;

/* DPLL MFNxxx Register (DPLLC_MFNMINUS) */
typedef struct{
__REG32 MFNMINUS    :27;
__REG32             : 5;
} __dpllc_mfnminus_bits;

/* DPLL MFNxxx Register (DPLLC_MFNPLUS) */
typedef struct{
__REG32 MFNPLUS     :27;
__REG32             : 5;
} __dpllc_mfnplus_bits;

/* DPLL High Frequency Support, Operation Register(DPLLC_HFS_OP) */
typedef struct{
__REG32 HFS_PDF     : 4;
__REG32 HFS_MFI     : 4;
__REG32             :24;
} __dpllc_hfs_op_bits;

/* DPLL HFS MFD Register (DPLLC_HFS_MFD) */
typedef struct{
__REG32 HFS_MFD     :27;
__REG32             : 5;
} __dpllc_hfs_mfd_bits;

/* DPLL HFS Multiplication Factor Numerator Register (DPLLC_HFS_MFN) */
typedef struct{
__REG32 HFS_MFN     :27;
__REG32             : 5;
} __dpllc_hfs_mfn_bits;

/* DPLL Multiplication Factor Numerator Toggle Control Register (DPLLC_MFN_TOGC) */
typedef struct{
__REG32 TOG_MFN_CNT :16;
__REG32 TOG_EN      : 1;
__REG32 TOG_DIS     : 1;
__REG32             :14;
} __dpllc_mfn_togc_bits;

/* Desense Status Register(DPLLC_DESTAT) */
typedef struct{
__REG32 TOG_MFN     :27;
__REG32             : 4;
__REG32 TOG_SEL     : 1;
} __dpllc_destat_bits;

/* DRAM CTL Register 00 (DRAM_CTL00) */
typedef struct{
__REG32 START           : 1;
__REG32                 : 7;
__REG32 DRAM_CLASS      : 4;
__REG32                 : 4;
__REG32 VERSION         :16;
} __dram_ctl00_bits;

/* DRAM CTL Register 01 (DRAM_CTL01) */
typedef struct{
__REG32 MAX_ROW_REG     : 4;
__REG32                 : 4;
__REG32 MAX_COL_REG     : 4;
__REG32                 : 4;
__REG32 MAX_CS_REG      : 2;
__REG32                 :14;
} __dram_ctl01_bits;

/* DRAM CTL Register 02 (DRAM_CTL02) */
typedef struct{
__REG32 TINIT           :24;
__REG32                 : 8;
} __dram_ctl02_bits;

/* DRAM CTL Register 03 (DRAM_CTL03) */
typedef struct{
__REG32 TINIT3          :24;
__REG32                 : 8;
} __dram_ctl03_bits;

/* DRAM CTL Register 04 (DRAM_CTL04) */
typedef struct{
__REG32 TINIT4          :24;
__REG32                 : 8;
} __dram_ctl04_bits;

/* DRAM CTL Register 05 (DRAM_CTL05) */
typedef struct{
__REG32 TINIT4          :24;
__REG32 INITAREF        : 4;
__REG32                 : 4;
} __dram_ctl05_bits;

/* DRAM CTL Register 06 (DRAM_CTL06) */
typedef struct{
__REG32 CASLAT_LIN      : 5;
__REG32                 : 3;
__REG32 CASLAT_LIN_GATE : 5;
__REG32                 : 3;
__REG32 WRLAT           : 4;
__REG32                 : 4;
__REG32 TCCD            : 5;
__REG32                 : 3;
} __dram_ctl06_bits;

/* DRAM CTL Register 07 (DRAM_CTL07) */
typedef struct{
__REG32 TBST_INT_INTERVAL : 3;
__REG32                   : 5;
__REG32 TRRD              : 3;
__REG32                   : 5;
__REG32 TRC               : 6;
__REG32                   : 2;
__REG32 TRAS_MIN          : 8;
} __dram_ctl07_bits;

/* DRAM CTL Register 08 (DRAM_CTL08) */
typedef struct{
__REG32 TWTR            : 4;
__REG32                 : 4;
__REG32 TRP             : 5;
__REG32                 : 3;
__REG32 TRTP            : 3;
__REG32                 : 5;
__REG32 TMRD            : 5;
__REG32                 : 3;
} __dram_ctl08_bits;

/* DRAM CTL Register 09 (DRAM_CTL09) */
typedef struct{
__REG32 TMOD            : 8;
__REG32 TRAS_MAX        :16;
__REG32 WRITEINTERP     : 1;
__REG32                 : 7;
} __dram_ctl09_bits;

/* DRAM CTL Register 10 (DRAM_CTL10) */
typedef struct{
__REG32 TCKE            : 3;
__REG32                 : 5;
__REG32 TCKESR          : 5;
__REG32                 : 3;
__REG32 AP              : 1;
__REG32                 : 7;
__REG32 CONCURRENTAP    : 1;
__REG32                 : 7;
} __dram_ctl10_bits;

/* DRAM CTL Register 11 (DRAM_CTL11) */
typedef struct{
__REG32 TRAS_LOCKOUT    : 1;
__REG32                 : 7;
__REG32 TRCD_INT        : 8;
__REG32 TWR_INT         : 5;
__REG32                 : 3;
__REG32 TDAL            : 5;
__REG32                 : 3;
} __dram_ctl11_bits;

/* DRAM CTL Register 12 (DRAM_CTL12) */
typedef struct{
__REG32 TDLL            :16;
__REG32 NO_CMD_INIT     : 1;
__REG32                 : 7;
__REG32 TMRR            : 4;
__REG32                 : 4;
} __dram_ctl12_bits;

/* DRAM CTL Register 13 (DRAM_CTL13) */
typedef struct{
__REG32 BSTLEN          : 3;
__REG32                 : 5;
__REG32 TFAW            : 6;
__REG32                 : 2;
__REG32 TCPD            :16;
} __dram_ctl13_bits;

/* DRAM CTL Register 14 (DRAM_CTL14) */
typedef struct{
__REG32 TRP_AB            : 5;
__REG32                   : 3;
__REG32 REG_DIMM_ENABLE   : 1;
__REG32                   : 7;
__REG32 AREFRESH          : 1;
__REG32                   : 7;
__REG32 AUTO_REFRESH_MODE : 1;
__REG32                   : 7;
} __dram_ctl14_bits;

/* DRAM CTL Register 15 (DRAM_CTL15) */
typedef struct{
__REG32 TREF_ENABLE       : 1;
__REG32                   : 7;
__REG32 TRFC              :10;
__REG32                   :14;
} __dram_ctl15_bits;

/* DRAM CTL Register 16 (DRAM_CTL16) */
typedef struct{
__REG32 TREF              :16;
__REG32 TREF_INTERVAL     :14;
__REG32                   : 2;
} __dram_ctl16_bits;

/* DRAM CTL Register 17 (DRAM_CTL17) */
typedef struct{
__REG32 POWER_DOWN        : 1;
__REG32                   : 7;
__REG32 TPDEX             :16;
__REG32                   : 8;
} __dram_ctl17_bits;

/* DRAM CTL Register 18 (DRAM_CTL18) */
typedef struct{
__REG32 TXSR              :16;
__REG32 TXSNR             :16;
} __dram_ctl18_bits;

/* DRAM CTL Register 19 (DRAM_CTL19) */
typedef struct{
__REG32 SREFRESH              : 1;
__REG32                       : 7;
__REG32 PWRUP_SREFRESH_EXIT   : 1;
__REG32                       : 7;
__REG32 ENABLE_QUICK_SREFRESH : 1;
__REG32                       : 7;
__REG32 CKE_DELAY             : 3;
__REG32                       : 5;
} __dram_ctl19_bits;

/* DRAM CTL Register 20 (DRAM_CTL20) */
typedef struct{
__REG32 LOWPOWER_CONTROL        : 5;
__REG32                         : 3;
__REG32 LOWPOWER_POWER_DOWN_CNT :16;
__REG32                         : 8;
} __dram_ctl20_bits;

/* DRAM CTL Register 21 (DRAM_CTL21) */
typedef struct{
__REG32 LOWPOWER_SELF_REFRESH_CNT :16;
__REG32 LOWPOWER_EXTERNAL_CNT     :16;
} __dram_ctl21_bits;

/* DRAM CTL Register 22 (DRAM_CTL22) */
typedef struct{
__REG32 LOWPOWER_AUTO_ENABLE    : 5;
__REG32                         : 3;
__REG32 LOWPOWER_INTERNAL_CNT   :16;
__REG32                         : 8;
} __dram_ctl22_bits;

/* DRAM CTL Register 23 (DRAM_CTL23) */
typedef struct{
__REG32 LOWPOWER_REFRESH_HOLD   :16;
__REG32 LOWPOWER_REFRESH_ENABLE : 2;
__REG32                         : 6;
__REG32 CKSRE                   : 4;
__REG32                         : 4;
} __dram_ctl23_bits;

/* DRAM CTL Register 24 (DRAM_CTL24) */
typedef struct{
__REG32 CKSRX             : 4;
__REG32                   : 4;
__REG32 WRITE_MODEREG     : 1;
__REG32                   :23;
} __dram_ctl24_bits;

/* DRAM CTL Register 25 (DRAM_CTL25) */
typedef struct{
__REG32 READ_MODEREG      :17;
__REG32                   :15;
} __dram_ctl25_bits;

/* DRAM CTL Register 26 (DRAM_CTL26) */
typedef struct{
__REG32 PERIPHERAL_MRR_DATA :16;
__REG32 MR0_DATA_0          :15;
__REG32                     : 1;
} __dram_ctl26_bits;

/* DRAM CTL Register 27 (DRAM_CTL27) */
typedef struct{
__REG32 MR1_DATA_0          :15;
__REG32                     : 1;
__REG32 MR2_DATA_0          :15;
__REG32                     : 1;
} __dram_ctl27_bits;

/* DRAM CTL Register 28 (DRAM_CTL28) */
typedef struct{
__REG32 MR3_DATA_0          :15;
__REG32                     : 1;
__REG32 MR16_DATA_0         :15;
__REG32                     : 1;
} __dram_ctl28_bits;

/* DRAM CTL Register 29 (DRAM_CTL29) */
typedef struct{
__REG32 MR17_DATA_0         :15;
__REG32                     : 1;
__REG32 MR0_DATA_1          :15;
__REG32                     : 1;
} __dram_ctl29_bits;

/* DRAM CTL Register 30 (DRAM_CTL30) */
typedef struct{
__REG32 MR1_DATA_1          :15;
__REG32                     : 1;
__REG32 MR2_DATA_1          :15;
__REG32                     : 1;
} __dram_ctl30_bits;

/* DRAM CTL Register 31 (DRAM_CTL31) */
typedef struct{
__REG32 MR3_DATA_1          :15;
__REG32                     : 1;
__REG32 MR16_DATA_1         :15;
__REG32                     : 1;
} __dram_ctl31_bits;

/* DRAM CTL Register 32 (DRAM_CTL32) */
typedef struct{
__REG32 MR17_DATA_1         :15;
__REG32                     : 1;
__REG32 ZQINIT              :12;
__REG32                     : 4;
} __dram_ctl32_bits;

/* DRAM CTL Register 33 (DRAM_CTL33) */
typedef struct{
__REG32 ZQCS                :12;
__REG32                     : 4;
__REG32 REFRESH_PER_ZQ      : 8;
__REG32 ZQ_ON_SREF_EXIT     : 4;
__REG32                     : 4;
} __dram_ctl33_bits;

/* DRAM CTL Register 34 (DRAM_CTL34) */
typedef struct{
__REG32 ZQCS                :12;
__REG32                     : 4;
__REG32 REFRESH_PER_ZQ      : 8;
__REG32 ZQ_ON_SREF_EXIT     : 4;
__REG32                     : 4;
} __dram_ctl34_bits;

/* DRAM CTL Register 35 (DRAM_CTL35) */
typedef struct{
__REG32 ZQ_IN_PROGRESS      : 1;
__REG32                     : 7;
__REG32 ZQRESET             :12;
__REG32                     : 4;
__REG32 ZQCS_ROTATE         : 1;
__REG32                     : 7;
} __dram_ctl35_bits;

/* DRAM CTL Register 36 (DRAM_CTL36) */
typedef struct{
__REG32 EIGHT_BANK_MODE     : 1;
__REG32                     : 7;
__REG32 ADDR_PINS           : 3;
__REG32                     : 5;
__REG32 COLUMN_SIZE         : 3;
__REG32                     : 5;
__REG32 APREBIT             : 4;
__REG32                     : 4;
} __dram_ctl36_bits;

/* DRAM CTL Register 37 (DRAM_CTL37) */
typedef struct{
__REG32 AGE_COUNT           : 5;
__REG32                     : 3;
__REG32 COMMAND_AGE_COUNT   : 5;
__REG32                     : 3;
__REG32 ADDR_CMP_EN         : 1;
__REG32                     : 7;
__REG32 BANK_SPLIT_EN       : 1;
__REG32                     : 7;
} __dram_ctl37_bits;

/* DRAM CTL Register 38 (DRAM_CTL38) */
typedef struct{
__REG32 PLACEMENT_EN      : 1;
__REG32                   : 7;
__REG32 PRIORITY_EN       : 1;
__REG32                   : 7;
__REG32 RW_SAME_EN        : 1;
__REG32                   : 7;
__REG32 SWAP_EN           : 1;
__REG32                   : 7;
} __dram_ctl38_bits;

/* DRAM CTL Register 39 (DRAM_CTL39) */
typedef struct{
__REG32 DISABLE_RW_GROUP_W_BNK_CONFLICT : 2;
__REG32                                 : 6;
__REG32 SWAP_PORT_RW_SAME_EN            : 1;
__REG32                                 : 7;
__REG32 CS_MAP                          : 2;
__REG32                                 : 6;
__REG32 REDUC                           : 1;
__REG32                                 : 7;
} __dram_ctl39_bits;

/* DRAM CTL Register 40 (DRAM_CTL40) */
typedef struct{
__REG32 CMDLAT_REDUC_EN       : 1;
__REG32                       : 7;
__REG32 WRDATALAT_REDUC_EN    : 1;
__REG32                       : 7;
__REG32 LPDDR2_S4             : 1;
__REG32                       : 7;
__REG32 FAST_WRITE            : 1;
__REG32                       : 7;
} __dram_ctl40_bits;

/* DRAM CTL Register 41 (DRAM_CTL41) */
typedef struct{
__REG32 Q_FULLNESS              : 3;
__REG32                         : 5;
__REG32 RESYNC_DLL              : 1;
__REG32                         : 7;
__REG32 RESYNC_DLL_PER_AREF_EN  : 1;
__REG32                         :15;
} __dram_ctl41_bits;

/* DRAM CTL Register 42 (DRAM_CTL42) */
typedef struct{
__REG32 INT_STATUS          :11;
__REG32                     : 5;
__REG32 INT_ACK             :10;
__REG32                     : 6;
} __dram_ctl42_bits;

/* DRAM CTL Register 43 (DRAM_CTL43) */
typedef struct{
__REG32 INT_MASK            :11;
__REG32                     :21;
} __dram_ctl43_bits;

/* DRAM CTL Register 44 (DRAM_CTL44) */
typedef struct{
__REG32 OUT_OF_RANGE_ADDR   :31;
__REG32                     : 1;
} __dram_ctl44_bits;

/* DRAM CTL Register 45 (DRAM_CTL45) */
typedef struct{
__REG32 OUT_OF_RANGE_LENGTH : 7;
__REG32                     : 1;
__REG32 OUT_OF_RANGE_TYPE   : 6;
__REG32                     :18;
} __dram_ctl45_bits;

/* DRAM CTL Register 46 (DRAM_CTL46) */
typedef struct{
__REG32 OUT_OF_RANGE_SOURCE_ID :17;
__REG32                        :15;
} __dram_ctl46_bits;

/* DRAM CTL Register 47 (DRAM_CTL47) */
typedef struct{
__REG32 PORT_CMD_ERROR_ADDR    :31;
__REG32                        : 1;
} __dram_ctl47_bits;

/* DRAM CTL Register 48 (DRAM_CTL48) */
typedef struct{
__REG32 PORT_CMD_ERROR_ID    :17;
__REG32                      : 7;
__REG32 PORT_CMD_ERROR_TYPE  : 4;
__REG32                      : 4;
} __dram_ctl48_bits;

/* DRAM CTL Register 49 (DRAM_CTL49) */
typedef struct{
__REG32 PORT_DATA_ERROR_ID   :17;
__REG32                      : 7;
__REG32 PORT_DATA_ERROR_TYPE : 3;
__REG32                      : 5;
} __dram_ctl49_bits;

/* DRAM CTL Register 50 (DRAM_CTL50) */
typedef struct{
__REG32 ODT_RD_MAP_CS0   : 2;
__REG32                  : 6;
__REG32 ODT_WR_MAP_CS0   : 2;
__REG32                  : 6;
__REG32 ODT_RD_MAP_CS1   : 2;
__REG32                  : 6;
__REG32 ODT_WR_MAP_CS1   : 2;
__REG32                  : 6;
} __dram_ctl50_bits;

/* DRAM CTL Register 51 (DRAM_CTL51) */
typedef struct{
__REG32 ADD_ODT_CLK_R2W_SAMECS      : 4;
__REG32                             : 4;
__REG32 ADD_ODT_CLK_W2R_SAMECS      : 4;
__REG32                             : 4;
__REG32 ADD_ODT_CLK_DIFFTYPE_DIFFCS : 5;
__REG32                             : 3;
__REG32 ADD_ODT_CLK_SAMETYPE_DIFFCS : 4;
__REG32                             : 4;
} __dram_ctl51_bits;

/* DRAM CTL Register 52 (DRAM_CTL52) */
typedef struct{
__REG32 R2R_DIFFCS_DLY      : 3;
__REG32                     : 5;
__REG32 R2W_DIFFCS_DLY      : 3;
__REG32                     : 5;
__REG32 W2R_DIFFCS_DLY      : 3;
__REG32                     : 5;
__REG32 W2W_DIFFCS_DLY      : 3;
__REG32                     : 5;
} __dram_ctl52_bits;

/* DRAM CTL Register 53 (DRAM_CTL53) */
typedef struct{
__REG32 R2R_SAMECS_DLY      : 3;
__REG32                     : 5;
__REG32 R2W_SAMECS_DLY      : 3;
__REG32                     : 5;
__REG32 W2R_SAMECS_DLY      : 3;
__REG32                     : 5;
__REG32 W2W_SAMECS_DLY      : 3;
__REG32                     : 5;
} __dram_ctl53_bits;

/* DRAM CTL Register 54 (DRAM_CTL54) */
typedef struct{
__REG32 TDQSCK_MAX          : 2;
__REG32                     : 6;
__REG32 TDQSCK_MIN          : 2;
__REG32                     : 6;
__REG32 OCD_ADJUST_PDN_CS_0 : 5;
__REG32                     : 3;
__REG32 OCD_ADJUST_PUP_CS_0 : 5;
__REG32                     : 3;
} __dram_ctl54_bits;

/* DRAM CTL Register 55 (DRAM_CTL55) */
typedef struct{
__REG32 AXI0_EN_SIZE_LT_WIDTH_INSTR :16;
__REG32 AXI0_FIFO_TYPE_REG          : 2;
__REG32                             :14;
} __dram_ctl55_bits;

/* DRAM CTL Register 56 (DRAM_CTL56) */
typedef struct{
__REG32 AXI1_EN_SIZE_LT_WIDTH_INSTR           :16;
__REG32 AXI1_FIFO_TYPE_REG                    : 2;
__REG32                                       : 6;
__REG32 WEIGHTED_ROUND_ROBIN_LATENCY_CONTROL  : 1;
__REG32                                       : 7;
} __dram_ctl56_bits;

/* DRAM CTL Register 57 (DRAM_CTL57) */
typedef struct{
__REG32 WEIGHTED_ROUND_ROBIN_WEIGHT_SHARING   : 1;
__REG32                                       : 7;
__REG32 WRR_PARAM_VALUE_ERR                   : 4;
__REG32                                       : 4;
__REG32 AXI0_PRIORITY0_RELATIVE_PRIORITY      : 4;
__REG32                                       : 4;
__REG32 AXI0_PRIORITY1_RELATIVE_PRIORITY      : 4;
__REG32                                       : 4;
} __dram_ctl57_bits;

/* DRAM CTL Register 58 (DRAM_CTL58) */
typedef struct{
__REG32 AXI0_PRIORITY2_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI0_PRIORITY3_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI0_PRIORITY4_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI0_PRIORITY5_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
} __dram_ctl58_bits;                        
                                            
/* DRAM CTL Register 59 (DRAM_CTL59) */     
typedef struct{                             
__REG32 AXI0_PRIORITY6_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI0_PRIORITY7_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI0_PORT_ORDERING                  : 1;
__REG32                                     :15;
} __dram_ctl59_bits;

/* DRAM CTL Register 60 (DRAM_CTL60) */
typedef struct{
__REG32 AXI0_PRIORITY_RELAX                 :10;
__REG32                                     : 6;
__REG32 AXI1_PRIORITY0_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PRIORITY1_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
} __dram_ctl60_bits;

/* DRAM CTL Register 61 (DRAM_CTL61) */
typedef struct{
__REG32 AXI1_PRIORITY2_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PRIORITY3_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PRIORITY4_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PRIORITY5_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
} __dram_ctl61_bits;

/* DRAM CTL Register 62 (DRAM_CTL62) */     
typedef struct{                             
__REG32 AXI1_PRIORITY6_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PRIORITY7_RELATIVE_PRIORITY    : 4;
__REG32                                     : 4;
__REG32 AXI1_PORT_ORDERING                  : 1;
__REG32                                     :15;
} __dram_ctl62_bits;

/* DRAM CTL Register 63 (DRAM_CTL63) */     
typedef struct{                             
__REG32 AXI1_PRIORITY_RELAX   :10;
__REG32                       : 6;
__REG32 CKE_STATUS            : 1;
__REG32                       :15;
} __dram_ctl63_bits;

/* DRAM CTL Register 64 (DRAM_CTL64) */     
typedef struct{                             
__REG32 DLL_RST_DELAY     :10;
__REG32                   : 6;
__REG32 DLL_RST_ADJ_DLY   : 8;
__REG32 TDFI_PHY_WRLAT    : 4;
__REG32                   : 4;
} __dram_ctl64_bits;

/* DRAM CTL Register 65 (DRAM_CTL65) */     
typedef struct{                             
__REG32 TDFI_PHY_WRLAT_BASE   : 4;
__REG32                       : 4;
__REG32 TDFI_PHY_RDLAT        : 5;
__REG32                       : 3;
__REG32 TDFI_RDDATA_EN        : 5;
__REG32                       : 3;
__REG32 TDFI_RDDATA_EN_BASE   : 5;
__REG32                       : 3;
} __dram_ctl65_bits;

/* DRAM CTL Register 66 (DRAM_CTL66) */
typedef struct{                             
__REG32 DRAM_CLK_DISABLE      : 2;
__REG32                       : 6;
__REG32 TDFI_CTRLUPD_MIN      : 4;
__REG32                       : 4;
__REG32 TDFI_CTRLUPD_MAX      :16;
} __dram_ctl66_bits;

/* DRAM CTL Register 67 (DRAM_CTL67) */
typedef struct{                             
__REG32 TDFI_PHYUPD_TYPE0     :16;
__REG32 TDFI_PHYUPD_TYPE1     :16;
} __dram_ctl67_bits;

/* DRAM CTL Register 68 (DRAM_CTL68) */
typedef struct{                             
__REG32 TDFI_PHYUPD_TYPE2     :16;
__REG32 TDFI_PHYUPD_TYPE3     :16;
} __dram_ctl68_bits;

/* DRAM CTL Register 69 (DRAM_CTL69) */
typedef struct{                             
__REG32 TDFI_PHYUPD_RESP      :16;
__REG32 RDLAT_ADJ             : 5;
__REG32                       : 3;
__REG32 WRLAT_ADJ             : 4;
__REG32                       : 4;
} __dram_ctl69_bits;

/* DRAM CTL Register 70 (DRAM_CTL70) */
typedef struct{                             
__REG32 TDFI_CTRL_DELAY       : 4;
__REG32                       : 4;
__REG32 TDFI_DRAM_CLK_DISABLE : 3;
__REG32                       : 5;
__REG32 TDFI_DRAM_CLK_ENABLE  : 4;
__REG32                       : 4;
__REG32 ODT_ALT_EN            : 1;
__REG32                       : 7;
} __dram_ctl70_bits;

/* DRAM CTL Register 71 (DRAM_CTL71) */
typedef struct{                             
__REG32 AXI0_AWCOBUF      : 1;
__REG32                   : 7;
__REG32 MDDR_CKE_SEL      : 1;
__REG32                   : 3;
__REG32 AXI0_HIDE_BRESP   : 1;
__REG32                   :19;
} __dram_ctl71_bits;

/* DRAM CTL Register 72 (DRAM_CTL72) */
typedef struct{                             
__REG32 AXI0_SLV_ERR      : 1;
__REG32                   : 3;
__REG32 AXI0_MON_DIS      : 1;
__REG32                   : 3;
__REG32 AXI0_MON_CAPTURE  : 1;
__REG32                   :23;
} __dram_ctl72_bits;

/* DRAM CTL Register 73 (DRAM_CTL73) */
typedef struct{                             
__REG32 ZQ_HW_EN          : 1;
__REG32                   : 3;
__REG32 ZQ_HW_LOAD        : 1;
__REG32                   :11;
__REG32 ZQ_COMPARE_EN     : 1;
__REG32                   : 3;
__REG32 ZQ_SW_LOAD        : 1;
__REG32 ZQ_LOAD_SEL       : 1;
__REG32                   :10;
} __dram_ctl73_bits;

/* DRAM CTL Register 74 (DRAM_CTL74) */
typedef struct{                             
__REG32                   : 4;
__REG32 ZQ_PUPD_SEL       : 1;
__REG32                   :11;
__REG32 ZQ_PU_M1          : 5;
__REG32                   : 3;
__REG32 ZQ_PD_M1          : 4;
__REG32                   : 4;
} __dram_ctl74_bits;

/* DRAM CTL Register 75 (DRAM_CTL75) */
typedef struct{                             
__REG32 ZQ_PU             : 5;
__REG32                   : 3;
__REG32 ZQ_PD             : 4;
__REG32                   :20;
} __dram_ctl75_bits;

/* DRAM CTL Register 79 (DRAM_CTL79) */
typedef struct{                             
__REG32                   : 8;
__REG32 CTL_BUSY          : 1;
__REG32                   : 3;
__REG32 MON_AXI0_BUSY     : 1;
__REG32                   :11;
__REG32 Q_ALMOST_FULL     : 1;
__REG32                   : 7;
} __dram_ctl79_bits;

/* DRAM CTL Register 83 (DRAM_CTL83) */
typedef struct{                             
__REG32 ZQ_COM_OUT        : 1;
__REG32                   :31;
} __dram_ctl83_bits;

/* DRAM PHY Register 01 (DRAM_PHY01) */
typedef struct{                             
__REG32 PAD_ODT_VAL_B0    : 3;
__REG32                   : 1;
__REG32 PAD_ODT_VAL_B1    : 3;
__REG32                   : 1;
__REG32 PAD_ODT_VAL_B2    : 3;
__REG32                   : 1;
__REG32 PAD_ODT_VAL_B3    : 3;
__REG32                   :17;
} __dram_phy01_bits;

/* DRAM PHY Register 02 (DRAM_PHY02) */
typedef struct{                             
__REG32 DATA_OE_END       : 3;
__REG32                   : 1;
__REG32 DATA_OE_START     : 3;
__REG32                   : 1;
__REG32 DQS_OE_END        : 4;
__REG32 DQS_OE_START      : 4;
__REG32 ENABLE_HALF_CAS   : 1;
__REG32                   : 3;
__REG32 PAD_OE_POLARITY   : 1;
__REG32                   : 3;
__REG32 RD_DLY_SEL        : 3;
__REG32                   : 1;
__REG32 ECHO_GATE_EN      : 1;
__REG32 DQS_RATIO         : 1;
__REG32 DQ_TSEL_EN        : 1;
__REG32 DM_TSEL_EN        : 1;
} __dram_phy02_bits;

/* DRAM PHY Register 03 (DRAM_PHY03) */
typedef struct{                             
__REG32 GATE_CFG          : 3;
__REG32                   : 1;
__REG32 GATE_CLOSE_CFG    : 2;
__REG32 GATE_ERR_DELAY    : 3;
__REG32                   : 3;
__REG32 LPBK_ERR_DELAY    : 3;
__REG32                   : 1;
__REG32 LPBK_EN           : 1;
__REG32 LPBK_INTERNAL     : 1;
__REG32 LPBK_CTRL         : 2;
__REG32 LPBK_FAIL_SEL     : 1;
__REG32 LPBK_ERR_CHECK    : 1;
__REG32 TSEL_POLARITY     : 1;
__REG32 DQS_TSEL_EN       : 1;
__REG32 TSEL_END          : 4;
__REG32 TSEL_START        : 4;
} __dram_phy03_bits;

/* DRAM PHY Register 13 (DRAM_PHY13) */
typedef struct{                             
__REG32 DFI_RDDATA_VALID  : 4;
__REG32 LPBK_WR_EN        : 1;
__REG32 LPBK_RD_EN        : 1;
__REG32                   :10;
__REG32 DDR_SEL           : 1;
__REG32                   : 6;
__REG32 DFI_MOBILE_EN     : 1;
__REG32                   : 8;
} __dram_phy13_bits;

/* DRAM PHY Register 14 (DRAM_PHY14) */
typedef struct{                             
__REG32 DLL_START_POINT     : 8;
__REG32 DLL_RD_DELAY        : 7;
__REG32 DLL_RD_DELAY_BYPASS : 9;
__REG32                     : 4;
__REG32 DLL_BYPASS_MODE     : 1;
__REG32 PHASE_DETECT_SEL    : 3;
} __dram_phy14_bits;

/* DRAM PHY Register 15 (DRAM_PHY15) */
typedef struct{                             
__REG32 DLL_INCR            : 8;
__REG32 DLL_WR_DELAY        : 7;
__REG32 DLL_WR_DELAY_BYPASS : 9;
__REG32                     : 8;
} __dram_phy15_bits;

/* DRAM PHY Register 24 (DRAM_PHY24) */
typedef struct{                             
__REG32 LPBK_START          : 1;
__REG32 LPBK_STATUS         : 1;
__REG32                     : 2;
__REG32 LPBK_DM_DATAT       : 4;
__REG32 LPBK_DQ_DATA        :16;
__REG32 LPBK_ERR_OUT        : 1;
__REG32                     : 7;
} __dram_phy24_bits;

/* DRAM PHY Register 25 (DRAM_PHY25) */
typedef struct{                             
__REG32 dll_lock            : 1;
__REG32 dll_lock_val        :31;
} __dram_phy25_bits;

/* DRAM PHY Register 26 (DRAM_PHY26) */
typedef struct{                             
__REG32 rd_delay_val        : 8;
__REG32                     : 7;
__REG32 wr_delay_val        : 8;
__REG32                     : 9;
} __dram_phy26_bits;

/* DVFSC_THRS Register */
typedef struct{
__REG32 PNCTHR      : 6;
__REG32             :10;
__REG32 DWTHR       : 6;
__REG32 UPTHR       : 6;
__REG32             : 4;
} __dvfsc_thrs_bits;

/* DVFSC_COUN Register */
typedef struct{
__REG32 UPCNT       : 8;
__REG32             : 8;
__REG32 DNCNT       : 8;
__REG32             : 8;
} __dvfsc_coun_bits;

/* DVFSC_SIG1 Register */
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
} __dvfsc_sig1_bits;

/* DVFSC_SIG0 Register */
typedef struct{
__REG32 WSW0        : 6;
__REG32 WSW1        : 6;
__REG32             : 8;
__REG32 WSW2        : 3;
__REG32 WSW3        : 3;
__REG32 WSW4        : 3;
__REG32 WSW5        : 3;
} __dvfsc_sig0_bits;

/* DVFSC_GPC0 Register */
typedef struct{
__REG32 GPBC0       :17;
__REG32             :13;
__REG32 C0ACT       : 1;
__REG32 C0STRT      : 1;
} __dvfsc_gpc0_bits;

/* DVFSC_GPC1 Register */
typedef struct{
__REG32 GPBC1       :17;
__REG32             :13;
__REG32 C1ACT       : 1;
__REG32 C1STRT      : 1;
} __dvfsc_gpc1_bits;

/* DVFSC_GPBT Register */
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
} __dvfsc_gpbt_bits;

/* DVFSC_EMAC Register */
typedef struct{
__REG32 EMAC        : 9;
__REG32             :23;
} __dvfsc_emac_bits;

/* DVFSC_CNTR Register */
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
} __dvfsc_cntr_bits;

/* DVFSC_LTR0_0 Register */
typedef struct {
__REG32 LTS0_0      : 4;
__REG32 LTS0_1      : 4;
__REG32 LTS0_2      : 4;
__REG32 LTS0_3      : 4;
__REG32 LTS0_4      : 4;
__REG32 LTS0_5      : 4;
__REG32 LTS0_6      : 4;
__REG32 LTS0_7      : 4;
} __dvfsc_ltr0_0_bits;

/* DVFSC_LTR0_1 Register */
typedef struct {
__REG32 LTS0_8      : 4;
__REG32 LTS0_9      : 4;
__REG32 LTS0_10     : 4;
__REG32 LTS0_11     : 4;
__REG32 LTS0_12     : 4;
__REG32 LTS0_13     : 4;
__REG32 LTS0_14     : 4;
__REG32 LTS0_15     : 4;
} __dvfsc_ltr0_1_bits;

/* DVFSC_LTR1_0 Register */
typedef struct {
__REG32 LTS1_0      : 4;
__REG32 LTS1_1      : 4;
__REG32 LTS1_2      : 4;
__REG32 LTS1_3      : 4;
__REG32 LTS1_4      : 4;
__REG32 LTS1_5      : 4;
__REG32 LTS1_6      : 4;
__REG32 LTS1_7      : 4;
} __dvfsc_ltr1_0_bits;

/* DVFSC_LTR1_1 Register */
typedef struct {
__REG32 LTS1_8      : 4;
__REG32 LTS1_9      : 4;
__REG32 LTS1_10     : 4;
__REG32 LTS1_11     : 4;
__REG32 LTS1_12     : 4;
__REG32 LTS1_13     : 4;
__REG32 LTS1_14     : 4;
__REG32 LTS1_15     : 4;
} __dvfsc_ltr1_1_bits;

/* DVFSC_PT0 Register */
typedef struct {
__REG32 FPTN0       :17;
__REG32 PT0A        : 1;
__REG32             :14;
} __dvfsc_pt0_bits;

/* DVFSC_PT1 Register */
typedef struct {
__REG32 FPTN1       :17;
__REG32 PT1A        : 1;
__REG32             :14;
} __dvfsc_pt1_bits;

/* DVFSC_PT2 Register */
typedef struct {
__REG32 FPTN2       :17;
__REG32 PT2A        : 1;
__REG32             :14;
} __dvfsc_pt2_bits;

/* DVFSC_PT3 Register */
typedef struct {
__REG32 FPTN3       :17;
__REG32 PT3A        : 1;
__REG32             :14;
} __dvfsc_pt3_bits;

/* LTR0 Register (DVFSP_LTR0) */
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
__REG32 DWTHR       : 6;
__REG32 UPTHR       : 6;
__REG32             : 1;
__REG32 SIGD13      : 1;
__REG32 SIGD14      : 1;
__REG32 SIGD15      : 1;
} __dvfsp_ltr0_bits;

/* LTR1 Register (DVFSP_LTR1) */
typedef struct {
__REG32 PNCTHR      : 6;
__REG32 UPCNT       : 8;
__REG32 DNCNT       : 8;
__REG32 LTBRSR      : 1;
__REG32 LTBRSH      : 1;
__REG32             : 2;
__REG32 DIV_RATIO   : 6;
} __dvfsp_ltr1_bits;

/* LTR2 Register (DVFSP_LTR2) */
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

/* LTR3 Register (DVFSP_LTR3) */
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

/* LTBR0 Register (DVFSP_LTBR0) */
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

/* LTBR1 Register (DVFSP_LTBR1) */
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

/* PMCR0 Register (DVFSP_PMCR0) */
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

/* PMCR0 Register (DVFSP_PMCR1) */
typedef struct {
__REG32 DVGP        :16;
__REG32 P2PM        : 1;
__REG32 P4PM        : 1;
__REG32 P1IFM       : 1;
__REG32 P1ISM       : 1;
__REG32 P1INM       : 1;
__REG32             :11;
} __dvfsp_pmcr1_bits;

/* Control Register (ECSPI_CONREG) */
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

/* Config Register (ECSPI_CONFIGREG) */
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

/* Interrupt Control Register (ECSPI_INTREG) */
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

/* DMA Control Register (ECSPI_DMAREG) */
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

/* Status Register (ECSPI_STATREG) */
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

/* Sample Period Control Register (ECSPI_PERIODREG) */
typedef struct {
__REG32 SAMPLE_PERIOD :15;
__REG32 CSRC          : 1;
__REG32 CSD_CTL       : 6;
__REG32               :10;
} __ecspi_periodreg_bits;

/* Test Control Register (ECSPI_TESTREG) */
typedef struct {
__REG32 TXCNT         : 7;
__REG32               : 1;
__REG32 RXCNT         : 7;
__REG32               :16;
__REG32 LBC           : 1;
} __ecspi_testreg_bits;

/* Chip Select x General Configuration Register 1(EIM_CS<i>GCR1) */
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

/* Chip Select x General Configuration Register 2 (EIM_CS<i>GCR2) */
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

/* Chip Select x Read Configuration Register 1(EIM_CS<i>RCR1) */
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

/* Chip Select x Read Configuration Register 2 (EIM_CS<i>RCR2) */
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

/* Chip Select x Write Configuration Register 1 (EIM_CS<i>WCR1) */
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

/* Chip Select x Write Configuration Register 2 (EIM_CS<i>WCR2) */
typedef struct {
__REG32 WBCDD             : 1;
__REG32                   :31;
} __eim_cswcr2_bits;

/* EIM Configuration Register (EIM_WCR) */
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

/* EIM IP Access Register (EIM_WIAR) */
typedef struct {
__REG32 IPS_REQ           : 1;
__REG32 IPS_ACK           : 1;
__REG32 INT               : 1;
__REG32 ERRST             : 1;
__REG32 ACLK_EN           : 1;
__REG32                   :27;
} __eim_wiar_bits;

/* EPDC Control Register (EPDC_CTRLn) */
typedef struct {
__REG32 BURST_LEN_8       : 1;
__REG32                   : 3;
__REG32 LUT_DATA_SWIZZLE  : 2;
__REG32 UPD_DATA_SWIZZLE  : 2;
__REG32 SRAM_POWERDOWN    : 1;
__REG32                   :21;
__REG32 CLKGATE           : 1;
__REG32 SFTRST            : 1;
} __epdc_ctrl_bits;

/* EPDC Screen Resolution (EPDC_RES) */
typedef struct {
__REG32 HORIZONTAL        :13;
__REG32                   : 3;
__REG32 VERTICAL          :13;
__REG32                   : 3;
} __epdc_res_bits;

/* EPDC Format Control Register (EPDC_FORMATn) */
typedef struct {
__REG32 TFT_PIXEL_FORMAT  : 2; 
__REG32                   : 6; 
__REG32 BUF_PIXEL_FORMAT  : 3; 
__REG32                   : 5; 
__REG32 DEFAULT_TFT_PIXEL : 8; 
__REG32 BUF_PIXEL_SCALE   : 1; 
__REG32                   : 7;
} __epdc_format_bits;

/* EPDC FIFO Control Register (EPDC_FIFOCTRLn) */
typedef struct {
__REG32 FIFO_L_LEVEL      : 8; 
__REG32 FIFO_H_LEVEL      : 8; 
__REG32 FIFO_INIT_LEVEL   : 8; 
__REG32                   : 7; 
__REG32 ENABLE_PRIORITY   : 1; 
} __epdc_fifoctrl_bits;

/* EPDC Update Command Co-ordinate (EPDC_UPD_CORD) */
typedef struct {
__REG32 XCORD             :13; 
__REG32                   : 3; 
__REG32 YCORD             :13; 
__REG32                   : 3; 
} __epdc_upd_cord_bits;

/* EPDC Update Command Size (EPDC_UPD_SIZE) */
typedef struct {
__REG32 WIDTH             :13; 
__REG32                   : 3; 
__REG32 HEIGHT            :13; 
__REG32                   : 3; 
} __epdc_upd_size_bits;

/* EPDC Update Command Control (EPDC_UPD_CTRLn) */
typedef struct {
__REG32 UPDATE_MODE       : 1; 
__REG32                   : 7; 
__REG32 WAVEFORM_MODE     : 8; 
__REG32 LUT_SEL           : 4; 
__REG32                   :11; 
__REG32 USE_FIXED         : 1; 
} __epdc_upd_ctrl_bits;

/* EPDC Update Fixed Pixel Control (EPDC_UPD_FIXEDn) */
typedef struct {
__REG32 FIXCP             : 8; 
__REG32 FIXNP             : 8; 
__REG32                   :14; 
__REG32 FIXCP_EN          : 1; 
__REG32 FIXNP_EN          : 1; 
} __epdc_upd_fixed_bits;

/* EPDC Timing Control Engine Control Register(EPDC_TCE_CTRLn) */
typedef struct {
__REG32 PIXELS_PER_SDCLK  : 2; 
__REG32 SDDO_WIDTH        : 1; 
__REG32 DUAL_SCAN         : 1; 
__REG32 SCAN_DIR_0        : 1; 
__REG32 SCAN_DIR_1        : 1; 
__REG32 LVDS_MODE         : 1; 
__REG32 LVDS_MODE_CE      : 1; 
__REG32 DDR_MODE          : 1; 
__REG32 VCOM_MODE         : 1; 
__REG32 VCOM_VAL          : 2; 
__REG32                   : 4; 
__REG32 VSCAN_HOLDOFF     : 9; 
__REG32                   : 7; 
} __epdc_tce_ctrl_bits;

/* EPDC Timing Control Engine Source-Driver Config Register(EPDC_TCE_SDCFGn) */
typedef struct {
__REG32 PIXELS_PER_CE     :13; 
__REG32 SDDO_INVERT       : 1; 
__REG32 SDDO_REFORMAT     : 2; 
__REG32 NUM_CE            : 4; 
__REG32 SDSHR             : 1; 
__REG32 SDCLK_HOLD        : 1; 
__REG32                   :10; 
} __epdc_tce_sdcfg_bits;

/* EPDC Timing Control Engine Gate-Driver Config Register(EPDC_TCE_GDCFGn) */
typedef struct {
__REG32 GDSP_MODE         : 1; 
__REG32 GDOE_MODE         : 1; 
__REG32                   : 2; 
__REG32 GDRL              : 1; 
__REG32                   :27; 
} __epdc_tce_gdcfg_bits;

/* EPDC Timing Control Engine Horizontal Timing Register 1(EPDC_TCE_HSCAN1n) */
typedef struct {
__REG32 LINE_SYNC         :12; 
__REG32                   : 4; 
__REG32 LINE_SYNC_WIDTH   :12; 
__REG32                   : 4; 
} __epdc_tce_hscan1_bits;

/* EPDC Timing Control Engine Horizontal Timing Register 2(EPDC_TCE_HSCAN2n) */
typedef struct {
__REG32 LINE_BEGIN        :12; 
__REG32                   : 4; 
__REG32 LINE_END          :12; 
__REG32                   : 4; 
} __epdc_tce_hscan2_bits;

/* EPDC Timing Control Engine Vertical Timing Register(EPDC_TCE_VSCANn) */
typedef struct {
__REG32 FRAME_SYNC        : 8; 
__REG32 FRAME_BEGIN       : 8; 
__REG32 FRAME_END         : 8; 
__REG32                   : 8; 
} __epdc_tce_vscan_bits;

/* EPDC Timing Control Engine OE timing control Register(EPDC_TCE_OEn) */
typedef struct {
__REG32 SDOEZ_DLY         : 8; 
__REG32 SDOEZ_WIDTH       : 8; 
__REG32 SDOED_DLY         : 8; 
__REG32 SDOED_WIDTH       : 8; 
} __epdc_tce_oe_bits;

/* EPDC Timing Control Engine Driver Polarity Register(EPDC_TCE_POLARITYn) */
typedef struct {
__REG32 SDCE_POL          : 1; 
__REG32 SDLE_POL          : 1; 
__REG32 SDOE_POL          : 1; 
__REG32 GDOE_POL          : 1; 
__REG32 GDSP_POL          : 1; 
__REG32                   :27; 
} __epdc_tce_polarity_bits;

/* EPDC Timing Control Engine Timing Register 1(EPDC_TCE_TIMING1n) */
typedef struct {
__REG32 SDCLK_SHIFT       : 2; 
__REG32                   : 1; 
__REG32 SDCLK_INVERT      : 1; 
__REG32 SDLE_SHIFT        : 2; 
__REG32                   :26; 
} __epdc_tce_timing1_bits;

/* EPDC Timing Control Engine Timing Register 2(EPDC_TCE_TIMING2n) */
typedef struct {
__REG32 GDSP_OFFSET       :16; 
__REG32 GDCLK_HP          :16; 
} __epdc_tce_timing2_bits;

/* EPDC Timing Control Engine Timing Register 3(EPDC_TCE_TIMING3n) */
typedef struct {
__REG32 GDCLK_OFFSET      :16; 
__REG32 GDOE_OFFSET       :16; 
} __epdc_tce_timing3_bits;

/* EPDC IRQ Mask Register (EPDC_IRQ_MASKn) */
typedef struct {
__REG32 LUT0_CMPLT_IRQ_EN   : 1; 
__REG32 LUT1_CMPLT_IRQ_EN   : 1; 
__REG32 LUT2_CMPLT_IRQ_EN   : 1; 
__REG32 LUT3_CMPLT_IRQ_EN   : 1; 
__REG32 LUT4_CMPLT_IRQ_EN   : 1; 
__REG32 LUT5_CMPLT_IRQ_EN   : 1; 
__REG32 LUT6_CMPLT_IRQ_EN   : 1; 
__REG32 LUT7_CMPLT_IRQ_EN   : 1; 
__REG32 LUT8_CMPLT_IRQ_EN   : 1; 
__REG32 LUT9_CMPLT_IRQ_EN   : 1; 
__REG32 LUT10_CMPLT_IRQ_EN  : 1; 
__REG32 LUT11_CMPLT_IRQ_EN  : 1; 
__REG32 LUT12_CMPLT_IRQ_EN  : 1; 
__REG32 LUT13_CMPLT_IRQ_EN  : 1; 
__REG32 LUT14_CMPLT_IRQ_EN  : 1; 
__REG32 LUT15_CMPLT_IRQ_EN  : 1; 
__REG32 WB_CMPLT_IRQ_EN     : 1; 
__REG32 COL_IRQ_EN          : 1; 
__REG32 TCE_UNDERRUN_IRQ_EN : 1; 
__REG32 FRAME_END_IRQ_EN    : 1; 
__REG32 BUS_ERROR_IRQ_EN    : 1; 
__REG32 TCE_IDLE_IRQ_EN     : 1; 
__REG32                     :10; 
} __epdc_irq_mask_bits;

/* EPDC Interrupt Register (EPDC_IRQn) */
typedef struct {
__REG32 LUT0_CMPLT_IRQ    : 1; 
__REG32 LUT1_CMPLT_IRQ    : 1; 
__REG32 LUT2_CMPLT_IRQ    : 1; 
__REG32 LUT3_CMPLT_IRQ    : 1; 
__REG32 LUT4_CMPLT_IRQ    : 1; 
__REG32 LUT5_CMPLT_IRQ    : 1; 
__REG32 LUT6_CMPLT_IRQ    : 1; 
__REG32 LUT7_CMPLT_IRQ    : 1; 
__REG32 LUT8_CMPLT_IRQ    : 1; 
__REG32 LUT9_CMPLT_IRQ    : 1; 
__REG32 LUT10_CMPLT_IRQ   : 1; 
__REG32 LUT11_CMPLT_IRQ   : 1; 
__REG32 LUT12_CMPLT_IRQ   : 1; 
__REG32 LUT13_CMPLT_IRQ   : 1; 
__REG32 LUT14_CMPLT_IRQ   : 1; 
__REG32 LUT15_CMPLT_IRQ   : 1; 
__REG32 WB_CMPLT_IRQ      : 1; 
__REG32 COL_IRQ           : 1; 
__REG32 TCE_UNDERRUN_IRQ  : 1; 
__REG32 FRAME_END_IRQ     : 1; 
__REG32 BUS_ERROR_IRQ     : 1; 
__REG32 TCE_IDLE_IRQ      : 1; 
__REG32                   :10; 
} __epdc_irq_bits;

/* EPDC Status Register - LUTs (EPDC_EPDC_STATUS_LUTSn) */
typedef struct {
__REG32 LUT0_STS        : 1; 
__REG32 LUT1_STS        : 1; 
__REG32 LUT2_STS        : 1; 
__REG32 LUT3_STS        : 1; 
__REG32 LUT4_STS        : 1; 
__REG32 LUT5_STS        : 1; 
__REG32 LUT6_STS        : 1; 
__REG32 LUT7_STS        : 1; 
__REG32 LUT8_STS        : 1; 
__REG32 LUT9_STS        : 1; 
__REG32 LUT10_STS       : 1; 
__REG32 LUT11_STS       : 1; 
__REG32 LUT12_STS       : 1; 
__REG32 LUT13_STS       : 1; 
__REG32 LUT14_STS       : 1; 
__REG32 LUT15_STS       : 1; 
__REG32                 :16; 
} __epdc_status_luts_bits;

/* EPDC Status Register - Next Available LUT(EPDC_STATUS_NEXTLUT) */
typedef struct {
__REG32 NEXT_LUT        : 4; 
__REG32                 : 4; 
__REG32 NEXT_LUT_VALID  : 1; 
__REG32                 :23; 
} __epdc_status_nextlut_bits;

/* EPDC LUT Collision Status (EPDC_STATUS_COLn) */
typedef struct {
__REG32 LUT0_COL_STS    : 1; 
__REG32 LUT1_COL_STS    : 1; 
__REG32 LUT2_COL_STS    : 1; 
__REG32 LUT3_COL_STS    : 1; 
__REG32 LUT4_COL_STS    : 1; 
__REG32 LUT5_COL_STS    : 1; 
__REG32 LUT6_COL_STS    : 1; 
__REG32 LUT7_COL_STS    : 1; 
__REG32 LUT8_COL_STS    : 1; 
__REG32 LUT9_COL_STS    : 1; 
__REG32 LUT10_COL_STS   : 1; 
__REG32 LUT11_COL_STS   : 1; 
__REG32 LUT12_COL_STS   : 1; 
__REG32 LUT13_COL_STS   : 1; 
__REG32 LUT14_COL_STS   : 1; 
__REG32 LUT15_COL_STS   : 1; 
__REG32                 :16; 
} __epdc_status_col_bits;

/* EPDC General Status Register (EPDC_STATUSn) */
typedef struct {
__REG32 WB_BUSY         : 1; 
__REG32 LUTS_BUSY       : 1; 
__REG32 LUTS_UNDERRUN   : 1; 
__REG32                 :29; 
} __epdc_status_bits;

/* EPDC Debug register (EPDC_DEBUGn) */
typedef struct {
__REG32 COLLISION_OFF     : 1; 
__REG32 UNDERRUN_RECOVER  : 1; 
__REG32                   :30; 
} __epdc_debug_bits;

/* EPDC LUTn Debug Information register(EPDC_DEBUG_LUTn) */
typedef struct {
__REG32 STATEMACHINE      : 5; 
__REG32 FRAME             :10; 
__REG32                   : 1; 
__REG32 LUTADDR           :10; 
__REG32                   : 6; 
} __epdc_debug_lut_bits;

/* EPDC General Purpose I/O Debug register (EPDC_GPIOn) */
typedef struct {
__REG32 BDR               : 2; 
__REG32 PWRCTRL           : 4; 
__REG32 PWRCOM            : 1; 
__REG32                   :25; 
} __epdc_gpio_bits;

/* EPDC Version Register (EPDC_VERSION) */
typedef struct {
__REG32 STEP              :16; 
__REG32 MINOR             : 8; 
__REG32 MAJOR             : 8; 
} __epdc_version_bits;

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
__REG32                   : 1;
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

/* Block Attributes Register (ESDHCV2_BLKATTR) */
typedef struct {
__REG32 BLKSZE              :13;
__REG32                     : 3;
__REG32 BLKCNT              :16;
} __esdhc_blkattr_bits;

/* Transfer Type Register (ESDHCV2_XFERTYP) */
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

/* Transfer Type Register (ESDHCV3_XFERTYP) */
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

/* Present State Register (ESDHCV2_PRSSTAT) */
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

/* Protocol Control Register (ESDHCV2_PROCTL) */
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

/* System Control Register (ESDHCV2_SYSCTL) */
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

/* System Control Register (ESDHCV3_SYSCTL) */
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

/* Interrupt Status Register (ESDHCV2_IRQSTAT) */
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

/* Interrupt Status Enable Register (ESDHCV2_IRQSTATEN) */
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

/* Interrupt Signal Enable Register (ESDHCV2_IRQSIGEN) */
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

/* Auto CMD12 Error Status Register (ESDHCV2_AUTOC12ERR) */
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

/* Host Controller Capabilities (ESDHCV2_HOSTCAPBLT) */
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

/* Watermark Level Register (ESDHCV2_WML) */
typedef struct {
__REG32 RD_WML              : 8;
__REG32 RD_BRST_LEN         : 5;
__REG32                     : 3;
__REG32 WR_WML              : 8;
__REG32 WR_BRST_LEN         : 5;
__REG32                     : 3;
} __esdhc_wml_bits;

/* Force Event Register(ESDHCV2_FEVT) */
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

/* ADMA Error Status Register (ESDHCV2_ADMAES) */
typedef struct {
__REG32 ADMAES              : 2;
__REG32 ADMALME             : 1;
__REG32 ADMADCE             : 1;
__REG32                     :28;
} __esdhc_admaes_bits;

/* DLL (Delay Line) Control Register (ESDHCV3_DLLCTRL) */
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

/* DLL Status Register (ESDHCV3_DLLSTS) */
typedef struct {
__REG32 DLL_STS_SLV_LOCK          : 1;
__REG32 DLL_STS_REF_LOCK          : 1;
__REG32 DLL_STS_SLV_SEL           : 6;
__REG32 DLL_STS_REF_SEL           : 6;
__REG32                           :18;
} __esdhcv3_dllsts_bits;

/* Vendor Specific Register (ESDHCV2_VENDOR) */
typedef struct {
__REG32 EXT_DMA_EN          : 1;
__REG32 EXACT_BLK_NUM       : 1;
__REG32                     :14;
__REG32 INT_ST_VAL          : 8;
__REG32 DBG_SEL             : 4;
__REG32                     : 4;
} __esdhc_vendor_bits;

/* MMC Boot Register (ESDHCV2_MMCBOOT) */
typedef struct {
__REG32 DTOCV_ACK           : 4;
__REG32 BOOT_ACK            : 1;
__REG32 MMC_BOOT_MODE       : 1;
__REG32 BOOT_EN             : 1;
__REG32 AUTO_SABG_EN        : 1;
__REG32                     : 8;
__REG32 BOOT_BLK_CNT        :16;
} __esdhc_mmcboot_bits;

/* Host Controller Version (ESDHCV2_HOSTVER) */
typedef struct {
__REG32 SVN                 : 8;
__REG32 VVN                 : 8;
__REG32                     :16;
} __esdhc_hostver_bits;

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

/* Opcode/Pause Duration Register (OPDR) */
typedef struct {
__REG32 PAUSE_DUR     :16;
__REG32 OPCODE        :16;
} __fec_opdr_bits;

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

/* GPMI Control Register 0 */
typedef struct {
__REG32 XFER_COUNT          :16;
__REG32 ADDRESS_INCREMENT   : 1;
__REG32 ADDRESS             : 3;
__REG32 CS                  : 3;
__REG32 WORD_LENGTH         : 1;
__REG32 COMMAND_MODE        : 2;
__REG32 UDMA                : 1;
__REG32 LOCK_CS             : 1;
__REG32 DEV_IRQ_EN          : 1;
__REG32 RUN                 : 1;
__REG32 CLKGATE             : 1;
__REG32 SFTRST              : 1;
} __gpmi_ctrl0_bits;

/* GPMI Compare Register */
typedef struct {
__REG32 REFERENCE           :16;
__REG32 MASK                :16;
} __gpmi_compare_bits;

/* GPMI Integrated ECC Control Register */
typedef struct {
__REG32 BUFFER_MASK         : 9;
__REG32                     : 3;
__REG32 ENABLE_ECC          : 1;
__REG32 ECC_CMD             : 2;
__REG32                     : 1;
__REG32 HANDLE              :16;
} __gpmi_eccctrl_bits;

/* GPMI Integrated ECC Transfer Count Register */
typedef struct {
__REG32 COUNT               :16;
__REG32                     :16;
} __gpmi_ecccount_bits;

/* GPMI Control Register 1 */
typedef struct {
__REG32 MODE                          : 1;
__REG32 CAMERA_MODE                   : 1;
__REG32 ATA_IRQRDY_POLARITY           : 1;
__REG32 DEV_RESET                     : 1;
__REG32 ABORT_WAIT_FOR_READY_CHANNEL  : 3;
__REG32 ABORT_WAIT_REQUEST            : 1;
__REG32 BURST_EN                      : 1;
__REG32 TIMEOUT_IRQ                   : 1;
__REG32 DEV_IRQ                       : 1;
__REG32 DMA2ECC_MODE                  : 1;
__REG32 RDN_DELAY                     : 4;
__REG32 HALF_PERIOD                   : 1;
__REG32 DLL_ENABLE                    : 1;
__REG32 BCH_MODE                      : 1;
__REG32 GANGED_RDYBUSY                : 1;
__REG32 TIMEOUT_IRQ_EN                : 1;
__REG32                               : 1;
__REG32 WRN_DLY_SEL                   : 2;
__REG32 DECOUPLE_CS                   : 1;
__REG32 SSYNCMODE                     : 1;
__REG32 UPDATE_CS                     : 1;
__REG32 CLK_DIV2_EN                   : 1;
__REG32 TOGGLE_MODE                   : 1;
__REG32 WRITE_CLK_STOP                : 1;
__REG32 SSYNC_CLK_STOP                : 1;
__REG32 DEV_CLK_STOP                  : 1;
} __gpmi_ctrl1_bits;

/* GPMI Timing Register 0 */
typedef struct {
__REG32 DATA_SETUP            : 8;
__REG32 DATA_HOLD             : 8;
__REG32 ADDRESS_SETUP         : 8;
__REG32                       : 8;
} __gpmi_timing0_bits;

/* GPMI Timing Register 1 */
typedef struct {
__REG32                       :16;
__REG32 DEVICE_BUSY_TIMEOUT   :16;
} __gpmi_timing1_bits;

/* GPMI Timing Register 2 */
typedef struct {
__REG32 DATA_PAUSE                : 4;
__REG32 CMDADD_PAUSE              : 4;
__REG32 DATAPOSTAMBLE_DELAY_PAUSE : 4;
__REG32 PREAMBLE_DELAY            : 4;
__REG32 CE_DELAY                  : 5;
__REG32                           : 3;
__REG32 READ_LATENCY              : 3;
__REG32                           : 5;
} __gpmi_timing2_bits;

/* GPMI Status Register */
typedef struct {
__REG32 PRESENT               : 1;
__REG32 FIFO_FULL             : 1;
__REG32 FIFO_EMPTY            : 1;
__REG32 INVALID_BUFFER_MASK   : 1;
__REG32 ATA_IRQ               : 1;
__REG32                       : 3;
__REG32 DEV0_ERROR            : 1;
__REG32 DEV1_ERROR            : 1;
__REG32 DEV2_ERROR            : 1;
__REG32 DEV3_ERROR            : 1;
__REG32 DEV4_ERROR            : 1;
__REG32 DEV5_ERROR            : 1;
__REG32 DEV6_ERROR            : 1;
__REG32 DEV7_ERROR            : 1;
__REG32 RDY_TIMEOUT           : 8;
__REG32 READY_BUSY            : 8;
} __gpmi_stat_bits;

/* GPMI Debug Information Register */
typedef struct {
__REG32 CMD_END               : 8;
__REG32 DMAREQ                : 8;
__REG32 DMA_SENSE             : 8;
__REG32 WAIT_FOR_READY_END    : 8;
} __gpmi_debug_bits;

/* GPMI Version Register */
typedef struct {
__REG32 STEP                  :16;
__REG32 MINOR                 : 8;
__REG32 MAJOR                 : 8;
} __gpmi_version_bits;

/* GPMI Debug2 Information Register (GPMI_DEBUG2) */
typedef struct {
__REG32 RDN_TAP               : 6;
__REG32 UPDATE_WINDOW         : 1;
__REG32 VIEW_DELAYED_RDN      : 1;
__REG32 SYND2GPMI_READY       : 1;
__REG32 SYND2GPMI_VALID       : 1;
__REG32 GPMI2SYND_READY       : 1;
__REG32 GPMI2SYND_VALID       : 1;
__REG32 SYND2GPMI_BE          : 4;
__REG32 MAIN_STATE            : 4;
__REG32 PIN_STATE             : 3;
__REG32 BUSY                  : 1;
__REG32 UDMA_STATE            : 4;
__REG32                       : 4;
} __gpmi_debug2_bits;

/* GPMI Debug3 Information Register (GPMI_DEBUG3) */
typedef struct {
__REG32 DEV_WORD_CNTR         :16;
__REG32 APB_WORD_CNTR         :16;
} __gpmi_debug3_bits;

/* GPMI Double Rate Read DLL Control Register(GPMI_READ_DDR_DLL_CTRL) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 RESET                 : 1;
__REG32 SLV_FORCE_UPD         : 1;
__REG32 SLV_DLY_TARGET        : 4;
__REG32 GATE_UPDATE           : 1;
__REG32 REFCLK_ON             : 1;
__REG32 SLV_OVERRIDE          : 1;
__REG32 SLV_OVERRIDE_VAL      : 8;
__REG32                       : 2;
__REG32 SLV_UPDATE_INT        : 8;
__REG32 REF_UPDATE_INT        : 4;
} __gpmi_read_ddr_dll_ctrl_bits;

/* GPMI Double Rate Write DLL Control Register (GPMI_WRITE_DDR_DLL_CTRL) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 RESET                 : 1;
__REG32 SLV_FORCE_UPD         : 1;
__REG32 SLV_DLY_TARGET        : 4;
__REG32 GATE_UPDATE           : 1;
__REG32 REFCLK_ON             : 1;
__REG32 SLV_OVERRIDE          : 1;
__REG32 SLV_OVERRIDE_VAL      : 8;
__REG32                       : 2;
__REG32 SLV_UPDATE_INT        : 8;
__REG32 REF_UPDATE_INT        : 4;
} __gpmi_write_ddr_dll_ctrl_bits;

/* GPMI Double Rate Read DLL Status Register(GPMI_READ_DDR_DLL_STS) */
typedef struct {
__REG32 SLV_LOCK              : 1;
__REG32 SLV_SEL               : 8;
__REG32                       : 7;
__REG32 REF_LOCK              : 1;
__REG32 REF_SEL               : 8;
__REG32                       : 7;
} __gpmi_read_ddr_dll_sts_bits;

/* GPMI Double Rate Write DLL Status Register(GPMI_WRITE_DDR_DLL_STS) */
typedef struct {
__REG32 SLV_LOCK              : 1;
__REG32 SLV_SEL               : 8;
__REG32                       : 7;
__REG32 REF_LOCK              : 1;
__REG32 REF_SEL               : 8;
__REG32                       : 7;
} __gpmi_write_ddr_dll_sts_bits;

/* GPT Control Register (GPT_CR) */
typedef struct{
__REG32 EN              : 1;
__REG32 ENMOD           : 1;
__REG32 DBGEN           : 1;
__REG32 WAITEN          : 1;
__REG32 DOZEEN          : 1;
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
} __gpt_cr_bits;

/* GPT Prescaler Register (GPT_PR) */
typedef struct{
__REG32 PRESCALER       :12;
__REG32                 :20;
} __gpt_pr_bits;

/* GPT Status Register (GPT_SR) */
typedef struct{
__REG32 OF1             : 1;
__REG32 OF2             : 1;
__REG32 OF3             : 1;
__REG32 IF1             : 1;
__REG32 IF2             : 1;
__REG32 ROV             : 1;
__REG32                 :26;
} __gpt_sr_bits;

/* GPT Interrupt Register (GPT_IR) */
typedef struct{
__REG32 OF1IE           : 1;
__REG32 OF2IE           : 1;
__REG32 OF3IE           : 1;
__REG32 IF1IE           : 1;
__REG32 IF2IE           : 1;
__REG32 ROVIE           : 1;
__REG32                 :26;
} __gpt_ir_bits;

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

/* GPR0 (IOMUXC_GPR0) */
typedef struct {
__REG32 DMAREQ_MUX_SEL0   : 1;
__REG32 DMAREQ_MUX_SEL1   : 1;
__REG32 DMAREQ_MUX_SEL2   : 1;
__REG32 DMAREQ_MUX_SEL3   : 1;
__REG32                   :28;
} __iomuxc_gpr0_bits;

/* GPR1 (IOMUXC_GPR1) */
typedef struct {
__REG32 ACT_CS0           : 1;
__REG32 ADDRS0            : 2;
__REG32 ACT_CS1           : 1;
__REG32 ADDRS1            : 2;
__REG32 ACT_CS2           : 1;
__REG32 ADDRS2            : 2;
__REG32 ACT_CS3           : 1;
__REG32 ADDRS3            : 2;
__REG32                   :20;
} __iomuxc_gpr1_bits;

/* GPR2 (IOMUXC_GPR2) */
typedef struct {
__REG32 DRAM_DQ_INPUTON   : 4;
__REG32                   :28;
} __iomuxc_gpr2_bits;

/* OBSERVE_MUX_0 (IOMUXC_OBSMUX0) */
/* OBSERVE_MUX_1 (IOMUXC_OBSMUX1) */
typedef struct {
__REG32 OBSRV             : 6;
__REG32                   :26;
} __iomuxc_obsmux0_bits;

/* OBSERVE_MUX_2 (IOMUXC_OBSMUX2) */
/* OBSERVE_MUX_3 (IOMUXC_OBSMUX3) */
/* OBSERVE_MUX_4 (IOMUXC_OBSMUX4) */
typedef struct {
__REG32 OBSRV             : 5;
__REG32                   :27;
} __iomuxc_obsmux2_bits;

/* SW_MUX_CTL_PAD_x (IOMUXC_SMUXC_Px) */
typedef struct {
__REG32 MUX_MODE          : 3;
__REG32                   : 1;
__REG32 SION              : 1;
__REG32                   :27;
} __iomuxc_smuxc_pad_bits;

/* SW_PAD_CTL_PAD_x (OMUXC_SPADC_Px) */
typedef struct {
__REG32                   : 1;
__REG32 DSE               : 2;
__REG32 ODE               : 1;
__REG32 PUS               : 2;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_spadc_pad_bits;

/* SW_PAD_CTL_PAD_PMIC_ON_REQ(IOMUXC_SPADC_PPMIC_ON_REQ) */
typedef struct {
__REG32 SRE               : 1;
__REG32                   : 2;
__REG32 ODE               : 1;
__REG32                   :28;
} __iomuxc_spadc_ppmic_on_req_bits;

/* SW_PAD_CTL_PAD_PMIC_STBY_REQ(IOMUXC_SPADC_PPMIC_STBY_REQ) */
typedef struct {
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   :29;
} __iomuxc_spadc_ppmic_stby_req_bits;

/* SW_PAD_CTL_PAD_POR_B (IOMUXC_SPADC_PPOR_B) */
/* SW_PAD_CTL_PAD_BOOT_MODE1(IOMUXC_SPADC_PBOOT_MODE1) */
/* SW_PAD_CTL_PAD_BOOT_MODE0(IOMUXC_SPADC_PBOOT_MODE0) */
/* SW_PAD_CTL_PAD_TEST_MODE(IOMUXC_SPADC_PTEST_MODE) */
/* SW_PAD_CTL_PAD_JTAG_TMS(IOMUXC_SPADC_PJTAG_TMS) */
/* SW_PAD_CTL_PAD_JTAG_MOD(IOMUXC_SPADC_PJTAG_MOD) */
/* SW_PAD_CTL_PAD_JTAG_TRSTB(IOMUXC_SPADC_PJTAG_TRSTB) */
/* SW_PAD_CTL_PAD_JTAG_TDI(IOMUXC_SPADC_PJTAG_TDI) */
/* SW_PAD_CTL_PAD_JTAG_TCK(IOMUXC_SPADC_PJTAG_TCK) */
typedef struct {
__REG32                   : 7;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_spadc_ppor_b_bits;

/* SW_PAD_CTL_PAD_RESET_IN_B(IOMUXC_SPADC_PRESET_IN_B) */
typedef struct {
__REG32                   : 8;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_spadc_preset_in_b_bits;

/* SW_PAD_CTL_PAD_JTAG_TDO(IOMUXC_SPADC_PJTAG_TDO) */
typedef struct {
__REG32 SRE               : 1;
__REG32 DSE               : 2;
__REG32                   : 4;
__REG32 PKE               : 1;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_spadc_pjtag_tdo_bits;

/* SW_PAD_CTL_PAD_DRAM_OPEN(IOMUXC_SPADC_PDRAM_OPEN) */
/* SW_PAD_CTL_PAD_DRAM_OPENFB(IOMUXC_SPADC_PDRAM_OPENFB) */
typedef struct {
__REG32                   :19;
__REG32 DSE               : 3;
__REG32                   :10;
} __iomuxc_spadc_pdram_open_bits;

/* SW_PAD_CTL_PAD_DRAM_SDCLK_n(IOMUXC_SPADC_PDRAM_SDCLK_n) */
/* SW_PAD_CTL_PAD_DRAM_DQMn(IOMUXC_SPADC_PDRAM_DQMn) */
typedef struct {
__REG32                   :19;
__REG32 DSE               : 3;
__REG32                   : 5;
__REG32 DO_TRIM           : 2;
__REG32                   : 3;
} __iomuxc_spadc_pdram_sdclk_bits;

/* SW_PAD_CTL_PAD_DRAM_SDCKE(IOMUXC_SPADC_PDRAM_SDCKE) */
typedef struct {
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_spadc_pdram_sdcke_bits;

/* SW_PAD_CTL_PAD_DRAM_SDODTn(IOMUXC_SPADC_PDRAM_SDODTn) */
typedef struct {
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :11;
__REG32 DSE               : 3;
__REG32                   : 5;
__REG32 DO_TRIM           : 2;
__REG32                   : 3;
} __iomuxc_spadc_pdram_sdodt_bits;

/* SW_PAD_CTL_PAD_DRAM_Dn(IOMUXC_SPADC_PDRAM_Dn) */
typedef struct {
__REG32                   :27;
__REG32 DO_TRIM           : 2;
__REG32                   : 3;
} __iomuxc_spadc_pdram_d_bits;

/* SW_PAD_CTL_PAD_DRAM_SDQSn(IOMUXC_SPADC_PDRAM_SDQSn) */
typedef struct {
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32 PKE               : 1;
__REG32                   :11;
__REG32 DSE               : 3;
__REG32                   :10;
} __iomuxc_spadc_pdram_sdqs_bits;

/* SW_PAD_CTL_GRP_ADDDS (IOMUXC_SPAD_GADDDS) */
/* SW_PAD_CTL_GRP_BnDS (IOMUXC_SPAD_GBnDS) */
/* SW_PAD_CTL_GRP_CTLDS (IOMUXC_SPAD_GCTLDS) */
typedef struct {
__REG32                   :19;
__REG32 DSE               : 3;
__REG32                   :10;
} __iomuxc_spad_gaddds_bits;

/* SW_PAD_CTL_GRP_DDRMODE_CTL(IOMUXC_SPAD_GDDRMODE_CTL) */
/* SW_PAD_CTL_GRP_DDRMODE(IOMUXC_SPAD_GDDRMODE) */
typedef struct {
__REG32                   : 9;
__REG32 DDR_INPUT         : 1;
__REG32                   :22;
} __iomuxc_spad_gddrmode_bits;

/* SW_PAD_CTL_GRP_DDRPKE (IOMUXC_SPAD_GDDRPKE) */
typedef struct {
__REG32                   : 7;
__REG32 PKE               : 1;
__REG32                   :24;
} __iomuxc_spad_gddrpke_bits;

/* SW_PAD_CTL_GRP_EIM (IOMUXC_SPAD_GEIM) */
/* SW_PAD_CTL_GRP_EPDC (IOMUXC_SPAD_GEPDC) */
/* SW_PAD_CTL_GRP_UART (IOMUXC_SPAD_GUART) */
/* SW_PAD_CTL_GRP_KEYPAD (IOMUXC_SPAD_GKEYPAD) */
/* SW_PAD_CTL_GRP_SSI (IOMUXC_SPAD_GSSI) */
/* SW_PAD_CTL_GRP_SDn (IOMUXC_SPAD_GSDn) */
/* SW_PAD_CTL_GRP_LCD (IOMUXC_SPAD_GLCD) */
/* SW_PAD_CTL_GRP_MISC (IOMUXC_SPAD_GMISC) */
/* SW_PAD_CTL_GRP_SPI (IOMUXC_SPAD_GSPI) */
/* SW_PAD_CTL_GRP_NANDF (IOMUXC_SPAD_GNANDF) */
typedef struct {
__REG32                   :13;
__REG32 HVE               : 1;
__REG32                   :18;
} __iomuxc_spad_geim_bits;

/* SW_PAD_CTL_GRP_DDRPK (IOMUXC_SPAD_GDDRPK) */
typedef struct {
__REG32                   : 6;
__REG32 PUE               : 1;
__REG32                   :25;
} __iomuxc_spad_gddrpk_bits;

/* SW_PAD_CTL_GRP_DDRHYS (IOMUXC_SPAD_GDDRHYS) */
typedef struct {
__REG32                   : 8;
__REG32 HYS               : 1;
__REG32                   :23;
} __iomuxc_spad_gddrhys_bits;

/* SW_PAD_CTL_GRP_DDRHYS (IOMUXC_SPAD_GDDRHYS) */
typedef struct {
__REG32                   :25;
__REG32 DDR_SEL           : 2;
__REG32                   : 5;
} __iomuxc_spad_gddr_type_bits;

/* n_SELECT_INPUT (IOMUXC_n_SI) */
typedef struct {
__REG32 DAISY             : 1;
__REG32                   :31;
} __iomuxc_si1_bits;

/* n_SELECT_INPUT (IOMUXC_n_SI) */
typedef struct {
__REG32 DAISY             : 2;
__REG32                   :30;
} __iomuxc_si2_bits;

/* n_SELECT_INPUT (IOMUXC_n_SI) */
typedef struct {
__REG32 DAISY             : 3;
__REG32                   :29;
} __iomuxc_si3_bits;

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

/* eLCDIF General Control Register (LCDIF_CTRLn) */
typedef struct {
__REG32 RUN                   : 1;
__REG32 DATA_FORMAT_24_BIT    : 1;
__REG32 DATA_FORMAT_18_BIT    : 1;
__REG32 DATA_FORMAT_16_BIT    : 1;
__REG32                       : 1;
__REG32 MASTER                : 1;
__REG32 ENABLE_PXP_HANDSHAKE  : 1;
__REG32 RGB_TO_YCBCR422_CSC   : 1;
__REG32 WORD_LENGTH           : 2;
__REG32 LCD_DATABUS_WIDTH     : 2;
__REG32 CSC_DATA_SWIZZLE      : 2;
__REG32 INPUT_DATA_SWIZZLE    : 2;
__REG32 DATA_SELECT           : 1;
__REG32 DOTCLK_MODE           : 1;
__REG32 VSYNC_MODE            : 1;
__REG32 BYPASS_COUNT          : 1;
__REG32 DVI_MODE              : 1;
__REG32 SHIFT_NUM_BITS        : 5;
__REG32 DATA_SHIFT_DIR        : 1;
__REG32 WAIT_FOR_VSYNC_EDGE   : 1;
__REG32 READ_WRITEB           : 1;
__REG32 YCBCR422_INPUT        : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __lcdif_ctrl_bits;

/* eLCDIF General Control1 Register (LCDIF_CTRL1n) */
typedef struct {
__REG32 RESET                   : 1;
__REG32 MODE86                  : 1;
__REG32 BUSY_ENABLE             : 1;
__REG32                         : 5;
__REG32 VSYNC_EDGE_IRQ          : 1;
__REG32 CUR_FRAME_DONE_IRQ      : 1;
__REG32 UNDERFLOW_IRQ           : 1;
__REG32 OVERFLOW_IRQ            : 1;
__REG32 VSYNC_EDGE_IRQ_EN       : 1;
__REG32 CUR_FRAME_DONE_IRQ_EN   : 1;
__REG32 UNDERFLOW_IRQ_EN        : 1;
__REG32 OVERFLOW_IRQ_EN         : 1;
__REG32 BYTE_PACKING_FORMAT     : 4;
__REG32 IRQ_ON_ALTERNATE_FIELDS : 1;
__REG32 FIFO_CLEAR              : 1;
__REG32 START_INTERLACE_FROM_SECOND_FIELD : 1;
__REG32 INTERLACE_FIELDS        : 1;
__REG32 RECOVER_ON_UNDERFLOW    : 1;
__REG32 BM_ERROR_IRQ            : 1;
__REG32 BM_ERROR_IRQ_EN         : 1;
__REG32 COMBINE_MPU_WR_STRB     : 1;
__REG32                         : 4;
} __lcdif_ctrl1_bits;

/* eLCDIF General Control2 Register (LCDIF_CTRL2n) */
typedef struct {
__REG32                       : 1;
__REG32 INITIAL_DUMMY_READ    : 3;
__REG32 READ_MODE_NUM_PACKED_SUBWORDS   : 3;
__REG32                       : 1;
__REG32 READ_MODE_6_BIT_INPUT : 1;
__REG32 READ_MODE_OUTPUT_IN_RGB_FORMAT  : 1;
__REG32 READ_PACK_DIR         : 1;
__REG32                       : 1;
__REG32 EVEN_LINE_PATTERN     : 3;
__REG32                       : 1;
__REG32 ODD_LINE_PATTERN      : 3;
__REG32                       : 1;
__REG32 BURST_LEN_8           : 1;
__REG32 OUTSTANDING_REQS      : 3;
__REG32                       : 8;
} __lcdif_ctrl2_bits;

/* eLCDIF Horizontal and Vertical Valid Data Count Register
(LCDIF_TRANSFER_COUNT) */
typedef struct {
__REG32 H_COUNT         :16;
__REG32 V_COUNT         :16;
} __lcdif_transfer_count_bits;

/* LCD Interface Timing Register (LCDIF_TIMING) */
typedef struct {
__REG32 DATA_SETUP      : 8;
__REG32 DATA_HOLD       : 8;
__REG32 CMD_SETUP       : 8;
__REG32 CMD_HOLD        : 8;
} __lcdif_timing_bits;

/* eLCDIF VSYNC Mode and Dotclk Mode Control Register0(LCDIF_VDCTRL0n) */
typedef struct {
__REG32 VSYNC_PULSE_WIDTH       :18;
__REG32 HALF_LINE_MODE          : 1;
__REG32 HALF_LINE               : 1;
__REG32 VSYNC_PULSE_WIDTH_UNIT  : 1;
__REG32 VSYNC_PERIOD_UNIT       : 1;
__REG32                         : 2;
__REG32 ENABLE_POL              : 1;
__REG32 DOTCLK_POL              : 1;
__REG32 HSYNC_POL               : 1;
__REG32 VSYNC_POL               : 1;
__REG32 ENABLE_PRESENT          : 1;
__REG32 VSYNC_OEB               : 1;
__REG32                         : 2;
} __lcdif_vdctrl0_bits;

/* eLCDIF VSYNC Mode and Dotclk Mode Control Register1(LCDIF_VDCTRL1) */
typedef struct {
__REG32 VSYNC_PERIOD          :32;
} __lcdif_vdctrl1_bits;

/* LCDIF VSYNC Mode and Dotclk Mode Control Register2(LCDIF_VDCTRL2) */
typedef struct {
__REG32 HSYNC_PERIOD          :18;
__REG32 HSYNC_PULSE_WIDTH     :14;
} __lcdif_vdctrl2_bits;

/* eLCDIF VSYNC Mode and Dotclk Mode Control Register3(LCDIF_VDCTRL3) */
typedef struct {
__REG32 VERTICAL_WAIT_CNT     :16;
__REG32 HORIZONTAL_WAIT_CNT   :12;
__REG32 VSYNC_ONLY            : 1;
__REG32 MUX_SYNC_SIGNALS      : 1;
__REG32                       : 2;
} __lcdif_vdctrl3_bits;

/* eLCDIF VSYNC Mode and Dotclk Mode Control Register4(LCDIF_VDCTRL4) */
typedef struct {
__REG32 DOTCLK_H_VALID_DATA_CNT :18;
__REG32 SYNC_SIGNALS_ON         : 1;
__REG32                         :10;
__REG32 DOTCLK_DLY_SEL          : 3;
} __lcdif_vdctrl4_bits;

/* Digital Video Interface Control0 Register (LCDIF_DVICTRL0) */
typedef struct {
__REG32 H_BLANKING_CNT        :12;
__REG32                       : 4;
__REG32 H_ACTIVE_CNT          :12;
__REG32                       : 4;
} __lcdif_dvictrl0_bits;

/* Digital Video Interface Control1 Register (LCDIF_DVICTRL1) */
typedef struct {
__REG32 F2_START_LINE     :10;
__REG32 F1_END_LINE       :10;
__REG32 F1_START_LINE     :10;
__REG32                   : 2;
} __lcdif_dvictrl1_bits;

/* Digital Video Interface Control2 Register (LCDIF_DVICTRL2) */
typedef struct {
__REG32 V1_BLANK_END_LINE   :10;
__REG32 V1_BLANK_START_LINE :10;
__REG32 F2_END_LINE         :10;
__REG32                     : 2;
} __lcdif_dvictrl2_bits;

/* Digital Video Interface Control3 Register (LCDIF_DVICTRL3) */
typedef struct {
__REG32 V_LINES_CNT         :10;
__REG32 V2_BLANK_END_LINE   :10;
__REG32 V2_BLANK_START_LINE :10;
__REG32                     : 2;
} __lcdif_dvictrl3_bits;

/* Digital Video Interface Control4 Register (LCDIF_DVICTRL4) */
typedef struct {
__REG32 H_FILL_CNT        : 8;
__REG32 CR_FILL_VALUE     : 8;
__REG32 CB_FILL_VALUE     : 8;
__REG32 Y_FILL_VALUE      : 8;
} __lcdif_dvictrl4_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient0 Register(LCDIF_CSC_COEFF0) */
typedef struct {
__REG32 CSC_SUBSAMPLE_FILTER  : 2;
__REG32                       :14;
__REG32 C0                    :10;
__REG32                       : 6;
} __lcdif_csc_coeff0_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient1 Register(LCDIF_CSC_COEFF1) */
typedef struct {
__REG32 C1            :10;
__REG32               : 6;
__REG32 C2            :10;
__REG32               : 6;
} __lcdif_csc_coeff1_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficent2 Register(LCDIF_CSC_COEFF2) */
typedef struct {
__REG32 C3            :10;
__REG32               : 6;
__REG32 C4            :10;
__REG32               : 6;
} __lcdif_csc_coeff2_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient3 Register(LCDIF_CSC_COEFF3) */
typedef struct {
__REG32 C5            :10;
__REG32               : 6;
__REG32 C6            :10;
__REG32               : 6;
} __lcdif_csc_coeff3_bits;

/* RGB to YCbCr 4:2:2 CSC Coefficient4 Register(LCDIF_CSC_COEFF4) */
typedef struct {
__REG32 C7            :10;
__REG32               : 6;
__REG32 C8            :10;
__REG32               : 6;
} __lcdif_csc_coeff4_bits;

/* RGB to YCbCr 4:2:2 CSC Offset Register(LCDIF_CSC_OFFSET) */
typedef struct {
__REG32 Y_OFFSET      : 9;
__REG32               : 7;
__REG32 CBCR_OFFSET   : 9;
__REG32               : 7;
} __lcdif_csc_offset_bits;

/* RGB to YCbCr 4:2:2 CSC Limit Register (LCDIF_CSC_LIMIT) */
typedef struct {
__REG32 Y_MAX         : 8;
__REG32 Y_MIN         : 8;
__REG32 CBCR_MAX      : 8;
__REG32 CBCR_MIN      : 8;
} __lcdif_csc_limit_bits;

/* LCD Interface Data Register (LCDIF_DATA) */
typedef struct {
__REG32 DATA_ZERO     : 8;
__REG32 DATA_ONE      : 8;
__REG32 DATA_TWO      : 8;
__REG32 DATA_THREE    : 8;
} __lcdif_data_bits;

/* LCD Interface Status Register (LCDIF_STAT) */
typedef struct {
__REG32 LFIFO_COUNT       : 9;
__REG32                   :15;
__REG32 DVI_CURRENT_FIELD : 1;
__REG32 BUSY              : 1;
__REG32 TXFIFO_EMPTY      : 1;
__REG32 TXFIFO_FULL       : 1;
__REG32 LFIFO_EMPTY       : 1;
__REG32 LFIFO_FULL        : 1;
__REG32 DMA_REQ           : 1;
__REG32 PRESENT           : 1;
} __lcdif_stat_bits;

/* LCD Interface Version Register (LCDIF_VERSION) */
typedef struct {
__REG32 STEP              :16;
__REG32 MINOR             : 8;
__REG32 MAJOR             : 8;
} __lcdif_version_bits;

/* LCD Interface Debug0 Register (LCDIF_DEBUG0) */
typedef struct {
__REG32 MST_WORDS               : 4;
__REG32 MST_OUTSTANDING_REQS    : 5;
__REG32 MST_AVALID              : 1;
__REG32 CUR_REQ_STATE           : 2;
__REG32 PXP_B1_DONE             : 1;
__REG32 PXP_LCDIF_B1_READY      : 1;
__REG32 PXP_B0_DONE             : 1;
__REG32 PXP_LCDIF_B0_READY      : 1;
__REG32 CUR_STATE               : 7;
__REG32 EMPTY_WORD              : 1;
__REG32 CUR_FRAME_TX            : 1;
__REG32 VSYNC                   : 1;
__REG32 HSYNC                   : 1;
__REG32 ENABLE                  : 1;
__REG32 DMACMDKICK              : 1;
__REG32 SYNC_SIGNALS_ON_REG     : 1;
__REG32 WAIT_FOR_VSYNC_EDGE_OUT : 1;
__REG32 STREAMING_END_DETECTED  : 1;
} __lcdif_debug0_bits;

/* LCD Interface Debug1 Register (LCDIF_DEBUG1) */
typedef struct {
__REG32 V_DATA_COUNT        :16;
__REG32 H_DATA_COUNT        :16;
} __lcdif_debug1_bits;

/* eLCDIF Threshold Register (LCDIF_THRES) */
typedef struct {
__REG32 PANIC               : 9;
__REG32                     : 7;
__REG32 FASTCLOCK           : 9;
__REG32                     : 7;
} __lcdif_thres_bits;

/* OTP Controller Control Register (OCOTP_CTRLn) */
typedef struct {
__REG32 ADDR                : 6;
__REG32                     : 2;
__REG32 BUSY                : 1;
__REG32 ERROR               : 1;
__REG32                     : 2;
__REG32 RD_BANK_OPEN        : 1;
__REG32 RELOAD_SHADOWS      : 1;
__REG32                     : 2;
__REG32 WR_UNLOCK           :16;
} __ocotp_ctrl_bits;

/* OTP Controller Timing Register (OCOTP_TIMING) */
typedef struct {
__REG32 SCLK_COUNT          :12;
__REG32 RELAX               : 4;
__REG32 RD_BUSY             : 6;
__REG32                     :10;
} __ocotp_timing_bits;

/* LOCK Shadow Register OTP */
typedef struct {
__REG32 CFG_TESTER          : 1;
__REG32 CFG_TESTER_SHADOW   : 1;
__REG32 CFG_BOOT            : 1;
__REG32 CFG_BOOT_SHADOW     : 1;
__REG32 CFG_MISC_SHADOW     : 1;
__REG32 MEM_TRIM            : 1;
__REG32 MEM_TRIM_SHADOW     : 1;
__REG32 MEM_MISC_SHADOW     : 1;
__REG32 GP                  : 1;
__REG32 GP_SHADOW           : 1;
__REG32 SCC                 : 1;
__REG32 SCC_SHADOW          : 1;
__REG32 SRK                 : 1;
__REG32 SRK_SHADOW          : 1;
__REG32 SJC                 : 1;
__REG32 SJC_SHADOW          : 1;
__REG32 MAC                 : 1;
__REG32 MAC_SHADOW          : 1;
__REG32 HWCAP               : 1;
__REG32 HWCAP_SHADOW        : 1;
__REG32 SWCAP               : 1;
__REG32 SWCAP_SHADOW        : 1;
__REG32 DCPKEY              : 1;
__REG32 SCC_ALT             : 1;
__REG32 DCPKEY_ALT          : 1;
__REG32 PIN                 : 1;
__REG32 UNALLOCATED         : 6;
} __ocotp_lock_bits;

/* Software Controllable Signals Register (OCOTP_SCSn) */
typedef struct {
__REG32 HAB_JDE             : 1;
__REG32 SPARE               :30;
__REG32 LOCK                : 1;
} __ocotp_scs_bits;

/* OTP Controller Version Register */
typedef struct {
__REG32 STEP                :16;
__REG32 MINOR               : 8;
__REG32 MAJOR               : 8;
} __ocotp_version_bits;

/* Control Register (OWIRE_CONTROL) */
typedef struct{
__REG16       : 3;
__REG16 RDST  : 1;
__REG16 WR1   : 1;
__REG16 WR0   : 1;
__REG16 PST   : 1;
__REG16 RPP   : 1;
__REG16       : 8;
} __owire_control_bits;

/* Time Divider Register (OWIRE_TIME_DIVIDER) */
typedef struct{
__REG16 DVDR  : 8;
__REG16       : 8;
} __owire_time_divider_bits;

/* Reset Register (OWIRE_RESET) */
typedef struct{
__REG16 RST  : 1;
__REG16      :15;
} __owire_reset_bits;

/* Command Register (OWIRE_COMMAND) */
typedef struct{
__REG16      : 1;
__REG16 SRA  : 1;
__REG16      :14;
} __owire_command_bits;

/* Transmit/Receive Register (OWIRE_TX/RX) */
typedef struct{
__REG16 DATA : 8;
__REG16      : 8;
} __owire_tx_rx_bits;

/* Interrupt Register (OWIRE_INTERRUPT) */
typedef struct{
__REG16 PD   : 1;
__REG16 PDR  : 1;
__REG16 TBE  : 1;
__REG16 TSRE : 1;
__REG16 RBF  : 1;
__REG16 RSRF : 1;
__REG16      :10;
} __owire_interrupt_bits;

/* Interrupt Enable Register (OWIRE_INTERRUPT_EN) */
typedef struct{
__REG16 EPD  : 1;
__REG16 IAS  : 1;
__REG16 ETBE : 1;
__REG16 ETSE : 1;
__REG16 ERBF : 1;
__REG16 ERSF : 1;
__REG16      :10;
} __owire_interrupt_en_bits;

/* PerfMon Control Register (PERFMON_CTRL) */
typedef struct {
__REG32 RUN                         : 1;
__REG32 SNAP                        : 1;
__REG32 CLR                         : 1;
__REG32 READ_EN                     : 1;
__REG32 TRAP_ENABLE                 : 1;
__REG32 TRAP_IN_RANGE               : 1;
__REG32 LATENCY_ENABLE              : 1;
__REG32 TRAP_IRQ_EN                 : 1;
__REG32 LATENCY_IRQ_EN              : 1;
__REG32 BUS_ERR_IRQ_EN              : 1;
__REG32 TRAP_IRQ                    : 1;
__REG32 LATENCY_IRQ                 : 1;
__REG32 BUS_ERR_IRQ                 : 1;
__REG32                             : 3;
__REG32 IRQ_MID                     : 8;
__REG32                             : 6;
__REG32 CLKGATE                     : 1;
__REG32 SFTRST                      : 1;
} __perfmon_ctrl_bits;

/* PerfMon Master Enable Register (PERFMON_MASTER_EN) */
typedef struct {
__REG32 MID0                        : 1;
__REG32 MID1                        : 1;
__REG32 MID2                        : 1;
__REG32 MID3                        : 1;
__REG32 MID4                        : 1;
__REG32 MID5                        : 1;
__REG32 MID6                        : 1;
__REG32 MID7                        : 1;
__REG32 MID8                        : 1;
__REG32 MID9                        : 1;
__REG32 MID10                       : 1;
__REG32 MID11                       : 1;
__REG32 MID12                       : 1;
__REG32 MID13                       : 1;
__REG32 MID14                       : 1;
__REG32 MID15                       : 1;
__REG32                             :16;
} __perfmon_master_en_bits;

/* PerfMon Latency Threshold Register (PERFMON_LAT_THRESHOLD) */
typedef struct {
__REG32 VALUE                       :12;
__REG32                             :20;
} __perfmon_lat_threshold_bits;

/* PerfMon Maximum Latency Register (PERFMON_MAX_LATENCY) */
typedef struct {
__REG32 COUNT                       :12;
__REG32                             : 3;
__REG32 TAGID                       : 8;
__REG32 ASIZE                       : 3;
__REG32 ALEN                        : 4;
__REG32 ABURST                      : 2;
} __perfmon_max_latency_bits;

/* PerfMon Debug Register (PERFMON_DEBUG) */
typedef struct {
__REG32 ERR_MID                     : 1;
__REG32 TOTAL_CYCLE_CLR_EN          : 1;
__REG32                             :30;
} __perfmon_debug_bits;

/* PerfMon Version Register (PERFMON_VERSION) */
typedef struct {
__REG32 STEP                        :16;
__REG32 MINOR                       : 8;
__REG32 MAJOR                       : 8;
} __perfmon_version_bits;

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

/* PXP Control Register (PXP_CTRL) */
typedef struct {
__REG32 ENABLE                : 1;
__REG32 IRQ_ENABLE            : 1;
__REG32 NEXT_IRQ_ENABLE       : 1;
__REG32 ENABLE_LCD_HANDSHAKE  : 1;
__REG32 OUTPUT_RGB_FORMAT     : 4;
__REG32 ROTATE                : 2;
__REG32 HFLIP                 : 1;
__REG32 VFLIP                 : 1;
__REG32 S0_FORMAT             : 4;
__REG32 SUBSAMPLE             : 1;
__REG32 UPSAMPLE              : 1;
__REG32 SCALE                 : 1;
__REG32 CROP                  : 1;
__REG32 DELTA                 : 1;
__REG32 IN_PLACE              : 1;
__REG32 ALPHA_OUTPUT          : 1;
__REG32 BLOCK_SIZE            : 1;
__REG32 INTERLACED_INPUT      : 2;
__REG32 INTERLACED_OUTPUT     : 2;
__REG32 EN_REPEAT             : 1;
__REG32                       : 1;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __pxp_ctrl_bits;

/* PXP Status Register (PXP_STAT) */
typedef struct {
__REG32 IRQ               : 1;
__REG32 AXI_WRITE_ERROR   : 1;
__REG32 AXI_READ_ERROR    : 1;
__REG32 NEXT_IRQ          : 1;
__REG32 AXI_ERROR_ID      : 4;
__REG32                   : 8;
__REG32 BLOCKY            : 8;
__REG32 BLOCKX            : 8;
} __pxp_stat_bits;

/* PXP Output Buffer Size (PXP_OUTSIZE) */
typedef struct {
__REG32 HEIGHT            :12;
__REG32 WIDTH             :12;
__REG32 ALPHA             : 8;
} __pxp_outsize_bits;

/* PXP Source 0 (video) Buffer Parameters (PXP_S0PARAM) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __pxp_s0param_bits;

/* Source 0 Cropping Register (PXP_S0CROP) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __pxp_s0crop_bits;

/* Source 0 Scale Factor Register (PXP_S0SCALE) */
typedef struct {
__REG32 XSCALE            :15;
__REG32                   : 1;
__REG32 YSCALE            :15;
__REG32                   : 1;
} __pxp_s0scale_bits;

/* Source 0 Scale Offset Register (PXP_S0OFFSET) */
typedef struct {
__REG32 XOFFSET           :12;
__REG32                   : 4;
__REG32 YOFFSET           :12;
__REG32                   : 4;
} __pxp_s0offset_bits;

/* Color Space Conversion Coefficient Register 0 (PXP_CSCCOEFF0) */
typedef struct {
__REG32 Y_OFFSET          : 9;
__REG32 UV_OFFSET         : 9;
__REG32 C0                :11;
__REG32                   : 1;
__REG32 BYPASS            : 1;
__REG32 YCBCR_MODE        : 1;
} __pxp_csccoeff0_bits;

/* Color Space Conversion Coefficient Register 1 (PXP_CSCCOEFF1)*/
typedef struct {
__REG32 C4                :11;
__REG32                   : 5;
__REG32 C1                :11;
__REG32                   : 5;
} __pxp_csccoeff1_bits;

/* Color Space Conversion Coefficient Register 2 (PXP_CSCCOEFF2) */
typedef struct {
__REG32 C3                :11;
__REG32                   : 5;
__REG32 C2                :11;
__REG32                   : 5;
} __pxp_csccoeff2_bits;

/* PXP Next Frame Pointer (PXP_NEXT) */
typedef struct {
__REG32 ENABLED           : 1;
__REG32                   : 1;
__REG32 POINTER           :30;
} __pxp_next_bits;

/* PXP S0 Color Key Low (PXP_S0COLORKEYLOW) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __pxp_s0colorkeylow_bits;

/* PXP S0 Color Key High (PXP_S0COLORKEYHIGH) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __pxp_s0colorkeyhigh_bits;

/* PXP Overlay Color Key Low (PXP_OLCOLORKEYLOW) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __pxp_olcolorkeylow_bits;

/* PXP Overlay Color Key High (PXP_OLCOLORKEYHIGH) */
typedef struct {
__REG32 PIXEL             :24;
__REG32                   : 8;
} __pxp_olcolorkeyhigh_bits;

/* PXP Overlay 0 Size (PXP_OL0SIZE) */
typedef struct {
__REG32 HEIGHT            : 8;
__REG32 WIDTH             : 8;
__REG32 YBASE             : 8;
__REG32 XBASE             : 8;
} __pxp_olxsize_bits;

/* PXP Overlay 0 Parameters (PXP_OL0PARAM) */
typedef struct {
__REG32 ENABLE            : 1;
__REG32 ALPHA_CNTL        : 2;
__REG32 ENABLE_COLORKEY   : 1;
__REG32 FORMAT            : 4;
__REG32 ALPHA             : 8;
__REG32 ROP               : 4;
__REG32                   :12;
} __pxp_olxparam_bits;

/* Color Space Conversion Control Register(PXP_CSC2CTRL) */
typedef struct {
__REG32 BYPASS            : 1;
__REG32 CSC_MODE          : 2;
__REG32                   :29;
} __pxp_csc2ctrl_bits;

/* Color Space Conversion Coefficient Register 0(PXP_CSC2COEF0) */
typedef struct {
__REG32 A1                :11;
__REG32                   : 5;
__REG32 A2                :11;
__REG32                   : 5;
} __pxp_csc2coef0_bits;

/* Color Space Conversion Coefficient Register 1(PXP_CSC2COEF1) */
typedef struct {
__REG32 A3                :11;
__REG32                   : 5;
__REG32 B1                :11;
__REG32                   : 5;
} __pxp_csc2coef1_bits;

/* Color Space Conversion Coefficient Register 2(PXP_CSC2COEF2) */
typedef struct {
__REG32 B2                :11;
__REG32                   : 5;
__REG32 B3                :11;
__REG32                   : 5;
} __pxp_csc2coef2_bits;

/* Color Space Conversion Coefficient Register 3(PXP_CSC2COEF3) */
typedef struct {
__REG32 C1                :11;
__REG32                   : 5;
__REG32 C2                :11;
__REG32                   : 5;
} __pxp_csc2coef3_bits;

/* Color Space Conversion Coefficient Register 4(PXP_CSC2COEF4) */
typedef struct {
__REG32 C3                :11;
__REG32                   : 5;
__REG32 D1                :11;
__REG32                   : 5;
} __pxp_csc2coef4_bits;

/* Color Space Conversion Coefficient Register 5(PXP_CSC2COEF5) */
typedef struct {
__REG32 D2                :11;
__REG32                   : 5;
__REG32 D3                :11;
__REG32                   : 5;
} __pxp_csc2coef5_bits;

/* Lookup Table Control Register(PXP_LUT_CTRL) */
typedef struct {
__REG32 ADDR              : 8;
__REG32                   :23;
__REG32 BYPASS            : 1;
} __pxp_lut_ctrl_bits;

/* Lookup Table Data Register(PXP_LUT) */
typedef struct {
__REG32 DATA              : 8;
__REG32                   :24;
} __pxp_lut_bits;

/* Histogram Control Register(PXP_HIST_CTRL) */
typedef struct {
__REG32 STATUS            : 4;
__REG32 PANEL_MODE        : 2;
__REG32                   :26;
} __pxp_hist_ctrl_bits;

/* 2-level Histogram Parameter Register(PXP_HIST2_PARAM) */
typedef struct {
__REG32 VALUE0            : 5;
__REG32                   : 3;
__REG32 VALUE1            : 5;
__REG32                   :19;
} __pxp_hist2_param_bits;

/* 4-level Histogram Parameter Register(PXP_HIST4_PARAM) */
typedef struct {
__REG32 VALUE0            : 5;
__REG32                   : 3;
__REG32 VALUE1            : 5;
__REG32                   : 3;
__REG32 VALUE2            : 5;
__REG32                   : 3;
__REG32 VALUE3            : 5;
__REG32                   : 3;
} __pxp_hist4_param_bits;

/* 8-level Histogram Parameter 0 Register(PXP_HIST8_PARAM0) */
typedef struct {
__REG32 VALUE0            : 5;
__REG32                   : 3;
__REG32 VALUE1            : 5;
__REG32                   : 3;
__REG32 VALUE2            : 5;
__REG32                   : 3;
__REG32 VALUE3            : 5;
__REG32                   : 3;
} __pxp_hist8_param0_bits;

/* 8-level Histogram Parameter 1 Register(PXP_HIST8_PARAM1) */
typedef struct {
__REG32 VALUE4            : 5;
__REG32                   : 3;
__REG32 VALUE5            : 5;
__REG32                   : 3;
__REG32 VALUE6            : 5;
__REG32                   : 3;
__REG32 VALUE7            : 5;
__REG32                   : 3;
} __pxp_hist8_param1_bits;

/* 16-level Histogram Parameter 0 Register(PXP_HIST16_PARAM0) */
typedef struct {
__REG32 VALUE0            : 5;
__REG32                   : 3;
__REG32 VALUE1            : 5;
__REG32                   : 3;
__REG32 VALUE2            : 5;
__REG32                   : 3;
__REG32 VALUE3            : 5;
__REG32                   : 3;
} __pxp_hist16_param0_bits;

/* 16-level Histogram Parameter 1 Register(PXP_HIST16_PARAM1) */
typedef struct {
__REG32 VALUE4            : 5;
__REG32                   : 3;
__REG32 VALUE5            : 5;
__REG32                   : 3;
__REG32 VALUE6            : 5;
__REG32                   : 3;
__REG32 VALUE7            : 5;
__REG32                   : 3;
} __pxp_hist16_param1_bits;

/* 16-level Histogram Parameter 2 Register(PXP_HIST16_PARAM2) */
typedef struct {
__REG32 VALUE8            : 5;
__REG32                   : 3;
__REG32 VALUE9            : 5;
__REG32                   : 3;
__REG32 VALUE10           : 5;
__REG32                   : 3;
__REG32 VALUE11           : 5;
__REG32                   : 3;
} __pxp_hist16_param2_bits;

/* 16-level Histogram Parameter 3 Register(PXP_HIST16_PARAM3) */
typedef struct {
__REG32 VALUE12           : 5;
__REG32                   : 3;
__REG32 VALUE13           : 5;
__REG32                   : 3;
__REG32 VALUE14           : 5;
__REG32                   : 3;
__REG32 VALUE15           : 5;
__REG32                   : 3;
} __pxp_hist16_param3_bits;

/* QOS Control Register (QOS_CTRLn) */
typedef struct {
__REG32 EMI_PRIORITY_MODE     : 1;
__REG32 XLATE_AXI_MODE        : 1;
__REG32 EPDC_PRIORITY_BOOST   : 2;
__REG32 LCDIF_PRIORITY_BOOST  : 2;
__REG32                       :24;
__REG32 CLKGATE               : 1;
__REG32 SFTRST                : 1;
} __qos_ctrl_bits;

/* AXI QOS Register (QOS_AXI_QOS0n) */
typedef struct {
__REG32 M0_AWQOS      : 3;
__REG32               : 1;
__REG32 M0_ARQOS      : 3;
__REG32               : 1;
__REG32 M1_0_AWQOS    : 3;
__REG32               : 1;
__REG32 M1_0_ARQOS    : 3;
__REG32               : 1;
__REG32 M1_1_AWQOS    : 3;
__REG32               : 1;
__REG32 M1_1_ARQOS    : 3;
__REG32               : 1;
__REG32 M1_2_AWQOS    : 3;
__REG32               : 1;
__REG32 M1_2_ARQOS    : 3;
__REG32               : 1;
} __qos_axi_qos0_bits;

/* AXI QOS Register (QOS_AXI_QOS1n) */
typedef struct {
__REG32 M2_AWQOS      : 3;
__REG32               : 1;
__REG32 M2_ARQOS      : 3;
__REG32               : 1;
__REG32 M3_AWQOS      : 3;
__REG32               : 1;
__REG32 M3_ARQOS      : 3;
__REG32               : 1;
__REG32 M4_AWQOS      : 3;
__REG32               : 1;
__REG32 M4_ARQOS      : 3;
__REG32               : 1;
__REG32 M5_AWQOS      : 3;
__REG32               : 1;
__REG32 M5_ARQOS      : 3;
__REG32               : 1;
} __qos_axi_qos1_bits;

/* AXI QOS Register (QOS_AXI_QOS2n) */
typedef struct {
__REG32 M6_AWQOS      : 3;
__REG32               : 1;
__REG32 M6_ARQOS      : 3;
__REG32               : 1;
__REG32 M7_AWQOS      : 3;
__REG32               : 1;
__REG32 M7_ARQOS      : 3;
__REG32               : 1;
__REG32 M8_AWQOS      : 3;
__REG32               : 1;
__REG32 M8_ARQOS      : 3;
__REG32               : 1;
__REG32 M9_AWQOS      : 3;
__REG32               : 1;
__REG32 M9_ARQOS      : 3;
__REG32               : 1;
} __qos_axi_qos2_bits;

/* EMI priority Registers (QOS_EMI_PRIORITY0n) */
typedef struct {
__REG32 M0_WR         : 3;
__REG32               : 1;
__REG32 M0_RD         : 3;
__REG32               : 1;
__REG32 M1_0_WR       : 3;
__REG32               : 1;
__REG32 M1_0_RD       : 3;
__REG32               : 1;
__REG32 M1_1_WR       : 3;
__REG32               : 1;
__REG32 M1_1_RD       : 3;
__REG32               : 1;
__REG32 M1_2_WR       : 3;
__REG32               : 1;
__REG32 M1_2_RD       : 3;
__REG32               : 1;
} __qos_emi_priority0_bits;

/* EMI priority Registers (QOS_EMI_PRIORITY1n) */
typedef struct {
__REG32 M2_WR         : 3;
__REG32               : 1;
__REG32 M2_RD         : 3;
__REG32               : 1;
__REG32 M3_WR         : 3;
__REG32               : 1;
__REG32 M3_RD         : 3;
__REG32               : 1;
__REG32 M4_WR         : 3;
__REG32               : 1;
__REG32 M4_RD         : 3;
__REG32               : 1;
__REG32 M5_WR         : 3;
__REG32               : 1;
__REG32 M5_RD         : 3;
__REG32               : 1;
} __qos_emi_priority1_bits;

/* EMI priority Registers (QOS_EMI_PRIORITY2n) */
typedef struct {
__REG32 M6_WR         : 3;
__REG32               : 1;
__REG32 M6_RD         : 3;
__REG32               : 1;
__REG32 M7_WR         : 3;
__REG32               : 1;
__REG32 M7_RD         : 3;
__REG32               : 1;
__REG32 M8_WR         : 3;
__REG32               : 1;
__REG32 M8_RD         : 3;
__REG32               : 1;
__REG32 M9_WR         : 3;
__REG32               : 1;
__REG32 M9_RD         : 3;
__REG32               : 1;
} __qos_emi_priority2_bits;

/* AXI Master Disble Register (QOS_DISABLEn) */
typedef struct {
__REG32               : 1;
__REG32 M1_0_DIS      : 1;
__REG32 M1_1_DIS      : 1;
__REG32 M1_2_DIS      : 1;
__REG32 M2_DIS        : 1;
__REG32 M3_DIS        : 1;
__REG32 M4_DIS        : 1;
__REG32 M5_DIS        : 1;
__REG32 M6_DIS        : 1;
__REG32 M7_DIS        : 1;
__REG32 M8_DIS        : 1;
__REG32 M9_DIS        : 1;
__REG32               : 5;
__REG32 M1_0_DIS_STAT : 1;
__REG32 M1_1_DIS_STAT : 1;
__REG32 M1_2_DIS_STAT : 1;
__REG32 M2_DIS_STAT   : 1;
__REG32 M3_DIS_STAT   : 1;
__REG32 M4_DIS_STAT   : 1;
__REG32 M5_DIS_STAT   : 1;
__REG32 M6_DIS_STAT   : 1;
__REG32 M7_DIS_STAT   : 1;
__REG32 M8_DIS_STAT   : 1;
__REG32 M9_DIS_STAT   : 1;
__REG32               : 4;
} __qos_disable_bits;

/* QOS Version Register (QOS_VERSION) */
typedef struct {
__REG32 STEP          :16;
__REG32 MINOR         : 8;
__REG32 MAJOR         : 8;
} __qos_version_bits;

/* ROMPATCH Control Register (ROMPATCHCNTL) */
typedef struct{
__REG32 DATAFIX        : 8;
__REG32                :21;
__REG32 DIS            : 1;
__REG32                : 2;
} __rompatchcntl_bits;  

/* ROMPATCH Enable Register (ROMPATCHENL) */
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
} __rompatchenl_bits;   

/* ROMPATCH Address Registers (ROMPATCHA) */
typedef struct{
__REG32 THUMBX    : 1;
__REG32 ADDRX     :21;
__REG32           :10;
} __rompatcha_bits;   

/* ROMPATCH Status Register (ROMPATCHSR) */
typedef struct{
__REG32 SOURCE    : 6;
__REG32           :11;
__REG32 SW        : 1;
__REG32           :14;
} __rompatchsr_bits;
 
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

/* SRC Control Register (SRC_SCR) */
typedef struct{
__REG32 WARM_RESET_ENABLE     : 1;
__REG32                       : 3;
__REG32 SW_OPEN_VG_RST        : 1;
__REG32 WARM_RST_BYPASS_COUNT : 2;
__REG32 MASK_WDOG_RST         : 4;
__REG32 WEIM_RST              : 1;
__REG32                       :20;
} __src_scr_bits;

/* SRC Boot Mode Register (SRC_SBMR) */
typedef struct{
__REG32 BOOT_CFG1             : 8;
__REG32 BOOT_CFG2             : 8;
__REG32 BOOT_CFG3             : 8;
__REG32 BOOT_MODE             : 2;
__REG32 BT_FUSE_SEL           : 1;
__REG32 TEST_MODE             : 3;
__REG32                       : 2;
} __src_sbmr_bits;

/* SRC Reset Status Register (SRC_SRSR) */
typedef struct{
__REG32 IPP_RESET_B           : 1;
__REG32                       : 2;
__REG32 IPP_USER_RESET_B      : 1;
__REG32 WDOG_RST_B            : 1;
__REG32 JTAG_RST_B            : 1;
__REG32 JTAG_SW_RST           : 1;
__REG32                       : 9;
__REG32 WARM_BOOT             : 1;
__REG32                       :15;
} __src_srsr_bits;

/* SRC Interrupt Status Register (SRC_SISR) */
typedef struct{
__REG32                       : 3;
__REG32 OPEN_VG_PASSED_RESET  : 1;
__REG32                       :28;
} __src_sisr_bits;

/* SRC Interrupt Mask Register (SRC_SIMR) */
typedef struct{
__REG32 MASK_GPU_PASSED_RESET     : 1;
__REG32 MASK_VPU_PASSED_RESET     : 1;
__REG32 MASK_IPU_PASSED_RESET     : 1;
__REG32 MASK_OPEN_VG_PASSED_RESET : 1;
__REG32                           :28;
} __src_simr_bits;

/* SPRGC Control Register (SRPGC_SRPGCR) */
typedef struct{
__REG32 PCR                 : 1;
__REG32                     :31;
} __srpgc_srpgcr_bits;

/* Power-up Sequence Control Register (SRPGC_PUPSCR) */
typedef struct{
__REG32 SW                  : 6;
__REG32                     : 2;
__REG32 SW2SH               : 6;
__REG32                     : 2;
__REG32 SH2PG               : 6;
__REG32                     : 2;
__REG32 PG2ISO              : 2;
__REG32                     : 6;
} __srpgc_pupscr_bits;

/* Power-down Sequence Control Register (SRPGC_PDNSCR) */
typedef struct{
__REG32 ISO                 : 6;
__REG32                     : 2;
__REG32 ISO2PG              : 6;
__REG32                     : 2;
__REG32 PG2SH               : 2;
__REG32                     : 6;
__REG32 SH2SW               : 6;
__REG32                     : 2;
} __srpgc_pdnscr_bits;

/* SPRGC Status Register (SRPGC_SRPGSR) */
typedef struct{
__REG32 PSR                 : 1;
__REG32                     :31;
} __srpgc_srpgsr_bits;

/* SPRGC Debug Register (SRPGC_SRPGDR) */
typedef struct{
__REG32 DBG                 : 1;
__REG32                     : 7;
__REG32 ISO0                : 1;
__REG32 PG0                 : 1;
__REG32 PG0_LAFD            : 1;
__REG32 SW0                 : 1;
__REG32 SH0                 : 1;
__REG32 SH1                 : 1;
__REG32                     :18;
} __srpgc_srpgdr_bits;

/* SRTC LP Secure Counter LSB Register (SRTC_LPSCLR) */
typedef struct{
__REG32                           :17;
__REG32 LLPSC                     :15;
} __srtc_lpsclr_bits;

/* SRTC LP Control Register (SRTC_LPCR) */
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

/* SRTC LP Status Register (SRTC_LPSR) */
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

/* SRTC LP General Purpose Register (SRTC_LPGR) */
typedef struct{
__REG32 LPGR                      :29;
__REG32 LPB_STS                   : 1;
__REG32 SEC_BT                    : 1;
__REG32 SW_ISO                    : 1;
} __srtc_lpgr_bits;

/* HP Counter LSB Register (SRTC_HPCLR) */
typedef struct{
__REG32                           :17;
__REG32 LHPC                      :15;
} __srtc_hpclr_bits;

/* SRTC HP Alarm LSB Register (SRTC_HPALR) */
typedef struct{
__REG32                           :17;
__REG32 LHPA                      :15;
} __srtc_hpalr_bits;

/* SRTC HP Control Register (SRTC_HPCR) */
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

/* SRTC HP Interrupt Status Register (SRTC_HPISR) */
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

/* SRTC HP Interrupt Enable Register (SRTC_HPIENR) */
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
__REG32 TFRC                : 1;
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

/* SSI AC97 Channel Status Register (SACCST) */
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

/* SSI AC97 Channel Enable Register (SACCEN) */
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

/* SSI AC97 Channel Disable Register (ACCDIS) */
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

/* TempSensor Control Register(TEMPSENSOR_ANADIG_CTRLn) */
typedef struct{
__REG32 PWD             : 1;
__REG32 START           : 1;
__REG32 FINISHED        : 1;
__REG32                 : 5;
__REG32 TVAL1           :12;
__REG32 TVAL2           :12;
} __tempsensor_anadig_ctrl_bits;

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

/* USB Control Register (USBOH1_USB_CTRL) */
typedef struct{
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
} __usboh1_usb_ctrl_bits;

/* UTMI PHY Output Clock Valid Register (USBOH1_UTMI_CLK_VLD) */
typedef struct{
__REG32               : 6;
__REG32 H1_UTMIPHYCLK : 1;
__REG32 O_UTMIPHYCLK  : 1;
__REG32               :24;
} __usboh1_utmi_clk_vld_bits;

/* OTG UTMI PHY Control 0 Register(USBOH1_OTG_PHY_CTRL_0) */
typedef struct{
__REG32 CHRGDET_INT_FLG   : 1;
__REG32 CHRGDET_INT_EN    : 1;
__REG32 CHRGDET           : 1;
__REG32                   : 2;
__REG32 H1_Over_Cur_Dis   : 1;
__REG32 H1_Over_Cur_Pol   : 1;
__REG32                   : 1;
__REG32 OTG_Over_Cur_Dis  : 1;
__REG32 OTG_Over_Cur_Pol  : 1;
__REG32                   : 1;
__REG32 Reset             : 1;
__REG32 SUSPENDM          : 1;
__REG32                   : 2;
__REG32 VSTATUS           : 8;
__REG32 CHGRDETON         : 1;
__REG32 CHGRDETEN         : 1;
__REG32 CONF3             : 1;
__REG32 CONF2             : 1;
__REG32 VCONTROL          : 4;
__REG32 VLOAD             : 1;
} __usboh1_otg_phy_ctrl_0_bits;

/* OTG UTMI PHY Control 1 Register(USBOH1_OTG_PHY_CTRL_1) */
typedef struct{
__REG32 plldivvalue       : 2;
__REG32 extcal            : 5;
__REG32 calbp             : 1;
__REG32 preemdepth        : 1;
__REG32 enpre             : 1;
__REG32 lsrftsel          : 2;
__REG32 fsrftsel          : 2;
__REG32 icpctrl           : 2;
__REG32 fstunevsel        : 3;
__REG32 hstedvsel         : 2; 
__REG32 hsdedvsel         : 2;
__REG32 hsdrvslope        : 4;
__REG32 hsdrvamplitude    : 2;
__REG32 hsdrvtimingn      : 2;
__REG32 hsdrvtimingp      : 1;
} __usboh1_otg_phy_ctrl_1_bits;

/* USB Control Register 1 (USBOH1_USB_CTRL_1) */
typedef struct{
__REG32                   : 9;
__REG32 H1_utmi_onclk     : 1;
__REG32                   : 1;
__REG32 O_utmi_onclk      : 1;
__REG32                   :20;
} __usboh1_usb_ctrl_1_bits;

/* USB Control Register 2 (USBOH1_USB_CTRL_2) */
typedef struct{
__REG32                     : 5;
__REG32 OIDWK_EN            : 1;
__REG32 OVBWK_EN            : 1;
__REG32                     : 7;
__REG32 OPMODE_OVERRIDE     : 2;
__REG32 OPMODE_OVERRIDE_EN  : 1;
__REG32                     :15;
} __usboh1_usb_ctrl_2_bits;

/* Host1 UTMI PHY Control 0 Register(USBOH1_UH1_PHY_CTRL_0) */
typedef struct{
__REG32 CHRGDET_INT_FLG   : 1;
__REG32 CHRGDET_INT_EN    : 1;
__REG32 CHRGDET           : 1;
__REG32                   : 8;
__REG32 Reset             : 1;
__REG32 SUSPENDM          : 1;
__REG32                   : 2;
__REG32 VSTATUS           : 8;
__REG32 CHGRDETON         : 1;
__REG32 CHGRDETEN         : 1;
__REG32 CONF3             : 1;
__REG32 CONF2             : 1;
__REG32 VCONTROL          : 4;
__REG32 VLOAD             : 1;
} __usboh1_uh1_phy_ctrl_0_bits;

/* Host1 UTMI PHY Control 1 Register(USBOH1_UH1_PHY_CTRL_1) */
typedef struct{
__REG32 plldivvalue       : 2;
__REG32 extcal            : 5;
__REG32 calbp             : 1;
__REG32 preemdepth        : 1;
__REG32 enpre             : 1;
__REG32 lsrftsel          : 2;
__REG32 fsrftsel          : 2;
__REG32 icpctrl           : 2;
__REG32 fstunevsel        : 3;
__REG32 hstedvsel         : 2; 
__REG32 hsdedvsel         : 2;
__REG32 hsdrvslope        : 4;
__REG32 hsdrvamplitude    : 2;
__REG32 hsdrvtimingn      : 2;
__REG32 hsdrvtimingp      : 1;
} __usboh1_uh1_phy_ctrl_1_bits;

/* USB Clock on/off control Register(USBOH1_USB_CLKONOFF_CTRL)*/
typedef struct{
__REG32                   :17;
__REG32 OTG_AHBCLK_OFF    : 1;
__REG32 H1_AHBCLK_OFF     : 1;
__REG32                   :13;
} __usboh1_usb_clkonoff_ctrl_bits;

/* General Hardware Parameters */
typedef struct{
__REG32 HWGENERAL :12;
__REG32           :20;
} __usb_hwgeneral_bits;

/* Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __usb_hwhost_bits;

/* Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __usb_hwdevice_bits;

/* Buffer Hardware Parameters */
typedef struct{
__REG32 TXBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __usb_hwtxbuf_bits;

/* Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __usb_hwrxbuf_bits;

/* General Purpose Timer Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __usb_gptimerld_bits;

/* General Purpose Timer Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMODE     : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __usb_gptimerctrl_bits;

/* System Bus Interface Configuration */
typedef struct{
__REG32 AHBBRST     : 3;
__REG32             :29;
} __usb_sbuscfg_bits;

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
} __usb_hcsparams_bits;

/* Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __usb_hccparams_bits;

/* Device Control Capability Parameters */
typedef struct{
__REG32 DEN         : 5;
__REG32             : 2;
__REG32 DC          : 1;
__REG32 HC          : 1;
__REG32             :23;
} __usb_dccparams_bits;

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
} __usb_usbcmd_bits;

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
__REG32           : 3;
__REG32 HCH       : 1;
__REG32 RCL       : 1;
__REG32 PS        : 1;
__REG32 AS        : 1;
__REG32 NAKI      : 1;
__REG32           : 1;
__REG32 UAI       : 1;
__REG32 UPI       : 1;
__REG32           : 4;
__REG32 TI0       : 1;
__REG32 TI1       : 1;
__REG32           : 6;
} __usb_usbsts_bits;

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
__REG32           : 7;
__REG32 NAKE      : 1;
__REG32           : 1;
__REG32 UAIE      : 1;
__REG32 UPIE      : 1;
__REG32           : 4;
__REG32 TIE0      : 1;
__REG32 TIE1      : 1;
__REG32           : 6;
} __usb_usbintr_bits;

/* USB Frame Index */
typedef struct{
__REG32 FRINDEX   :14;
__REG32           :18;
} __usb_frindex_bits;

/* PERIODICLISTBASE—Host Controller Frame List Base Address */
/* DEVICEADDR—Device Controller USB Device Address */
typedef union {
/* USB_PERIODICLISTBASE*/
	struct{
		__REG32           :12;
		__REG32 PERBASE   :20;
	};
/* USB_DEVICEADDR*/
	struct{
		__REG32           :24;
		__REG32 USBADR    : 8;
	};
} __usb_periodiclistbase_bits;

/* ASYNCLISTADDR—Host Controller Next Asynchronous Address */
/* ENDPOINTLISTADDR—Device Controller Endpoint List Address */
typedef union {
/* USB_ASYNCLISTADDR*/
	struct{
		__REG32           : 6;
		__REG32 ASYBASE   :26;
	};
/* USB_ENDPOINTLISTADDR*/
	struct{
		__REG32           :11;
		__REG32 EPBASE    :21;
	};
} __usb_asynclistaddr_bits;

/* AHB Burst Size Control  */
typedef struct{
__REG32 RXPBURST  : 8;
__REG32 TXPBURST  : 9;
__REG32           :15;
} __usb_burstsize_bits;

/* USB Transmit FIFO Fill Tunning Register */
typedef struct{
__REG32 TXSCHOH     : 8;
__REG32             : 5;
__REG32 TXSCHHEALTH : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __usb_txfilltuning_bits;

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
} __usb_ic_usb_bits;

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
} __usb_ulpiview_bits;

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
__REG32 WKDC      : 1;
__REG32 WKOC      : 1;
__REG32 PHCD      : 1;
__REG32 PFSC      : 1;
__REG32 PTS2      : 1;
__REG32 PSPD      : 2;
__REG32 PTW       : 1;
__REG32 STS       : 1;
__REG32 PTS       : 2;
} __usb_portsc_bits;

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
} __usb_otgsc_bits;

/* Device Mode */
typedef struct{
__REG32 CM        : 2;
__REG32 ES        : 1;
__REG32 SLOM      : 1;
__REG32 SDIS      : 1;
__REG32           :10;
__REG32 SRT       : 1;
__REG32           :16;
} __usb_usbmode_bits;

/* Endpoint Setup Status */
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
} __usb_endptsetupstat_bits;

/* Endpoint Initialization */
typedef struct{
__REG32 PERB0           : 1;
__REG32 PERB1           : 1;
__REG32 PERB2           : 1;
__REG32 PERB3           : 1;
__REG32 PERB4           : 1;
__REG32 PERB5           : 1;
__REG32 PERB6           : 1;
__REG32 PERB7           : 1;
__REG32 PERB8           : 1;
__REG32 PERB9           : 1;
__REG32 PERB10          : 1;
__REG32 PERB11          : 1;
__REG32 PERB12          : 1;
__REG32 PERB13          : 1;
__REG32 PERB14          : 1;
__REG32 PERB15          : 1;
__REG32 PETB0           : 1;
__REG32 PETB1           : 1;
__REG32 PETB2           : 1;
__REG32 PETB3           : 1;
__REG32 PETB4           : 1;
__REG32 PETB5           : 1;
__REG32 PETB6           : 1;
__REG32 PETB7           : 1;
__REG32 PETB8           : 1;
__REG32 PETB9           : 1;
__REG32 PETB10          : 1;
__REG32 PETB11          : 1;
__REG32 PETB12          : 1;
__REG32 PETB13          : 1;
__REG32 PETB14          : 1;
__REG32 PETB15          : 1;
} __usb_endptprime_bits;

/* Endpoint De-Initialize */
typedef struct{
__REG32 FERB0           : 1;
__REG32 FERB1           : 1;
__REG32 FERB2           : 1;
__REG32 FERB3           : 1;
__REG32 FERB4           : 1;
__REG32 FERB5           : 1;
__REG32 FERB6           : 1;
__REG32 FERB7           : 1;
__REG32 FERB8           : 1;
__REG32 FERB9           : 1;
__REG32 FERB10          : 1;
__REG32 FERB11          : 1;
__REG32 FERB12          : 1;
__REG32 FERB13          : 1;
__REG32 FERB14          : 1;
__REG32 FERB15          : 1;
__REG32 FETB0           : 1;
__REG32 FETB1           : 1;
__REG32 FETB2           : 1;
__REG32 FETB3           : 1;
__REG32 FETB4           : 1;
__REG32 FETB5           : 1;
__REG32 FETB6           : 1;
__REG32 FETB7           : 1;
__REG32 FETB8           : 1;
__REG32 FETB9           : 1;
__REG32 FETB10          : 1;
__REG32 FETB11          : 1;
__REG32 FETB12          : 1;
__REG32 FETB13          : 1;
__REG32 FETB14          : 1;
__REG32 FETB15          : 1;
} __usb_endptflush_bits;

/* ENDPTSTAT—Endpoint Status */
typedef struct{
__REG32 ERBR0           : 1;
__REG32 ERBR1           : 1;
__REG32 ERBR2           : 1;
__REG32 ERBR3           : 1;
__REG32 ERBR4           : 1;
__REG32 ERBR5           : 1;
__REG32 ERBR6           : 1;
__REG32 ERBR7           : 1;
__REG32 ERBR8           : 1;
__REG32 ERBR9           : 1;
__REG32 ERBR10          : 1;
__REG32 ERBR11          : 1;
__REG32 ERBR12          : 1;
__REG32 ERBR13          : 1;
__REG32 ERBR14          : 1;
__REG32 ERBR15          : 1;
__REG32 ETBR0           : 1;
__REG32 ETBR1           : 1;
__REG32 ETBR2           : 1;
__REG32 ETBR3           : 1;
__REG32 ETBR4           : 1;
__REG32 ETBR5           : 1;
__REG32 ETBR6           : 1;
__REG32 ETBR7           : 1;
__REG32 ETBR8           : 1;
__REG32 ETBR9           : 1;
__REG32 ETBR10          : 1;
__REG32 ETBR11          : 1;
__REG32 ETBR12          : 1;
__REG32 ETBR13          : 1;
__REG32 ETBR14          : 1;
__REG32 ETBR15          : 1;
} __usb_endptstat_bits;

/* Endpoint Compete */
typedef struct{
__REG32 ERCE0           : 1;
__REG32 ERCE1           : 1;
__REG32 ERCE2           : 1;
__REG32 ERCE3           : 1;
__REG32 ERCE4           : 1;
__REG32 ERCE5           : 1;
__REG32 ERCE6           : 1;
__REG32 ERCE7           : 1;
__REG32 ERCE8           : 1;
__REG32 ERCE9           : 1;
__REG32 ERCE10          : 1;
__REG32 ERCE11          : 1;
__REG32 ERCE12          : 1;
__REG32 ERCE13          : 1;
__REG32 ERCE14          : 1;
__REG32 ERCE15          : 1;
__REG32 ETCE0           : 1;
__REG32 ETCE1           : 1;
__REG32 ETCE2           : 1;
__REG32 ETCE3           : 1;
__REG32 ETCE4           : 1;
__REG32 ETCE5           : 1;
__REG32 ETCE6           : 1;
__REG32 ETCE7           : 1;
__REG32 ETCE8           : 1;
__REG32 ETCE9           : 1;
__REG32 ETCE10          : 1;
__REG32 ETCE11          : 1;
__REG32 ETCE12          : 1;
__REG32 ETCE13          : 1;
__REG32 ETCE14          : 1;
__REG32 ETCE15          : 1;
} __usb_endptcomplete_bits;

/* Endpoint Control0 */
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
} __usb_endptctrl0_bits;

/* Endpoint Controln  */
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
} __usb_endptctrl_bits;

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
__REG16       : 2;     /* Bit  2  - 3  - Reserved*/
__REG16 POR   : 1;     /* Bit  4       - Power On Reset*/
__REG16       :11;     /* Bits 5  - 15 - Reserved*/
} __wrsr_bits;

/* Watchdog Interrupt Control Register (WDOG1_WICR) */
typedef struct {
__REG16 WICT  : 8;
__REG16       : 6;
__REG16 WTIS  : 1;
__REG16 WIE   : 1;
} __wicr_bits;

/* Watchdog Miscellaneous Control Register (WDOG1_WMCR)*/
typedef struct {
__REG16 PDE   : 1;
__REG16       :15;
} __wmcr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  CCM
 **
 ***************************************************************************/
__IO_REG32_BIT(CCM_CCR,                   0x53FD4000,__READ_WRITE ,__ccm_ccr_bits);
//__IO_REG32_BIT(CCM_CCDR,                  0x53FD4004,__READ_WRITE ,__ccm_ccdr_bits);
__IO_REG32_BIT(CCM_CSR,                   0x53FD4008,__READ       ,__ccm_csr_bits);
__IO_REG32_BIT(CCM_CCSR,                  0x53FD400C,__READ_WRITE ,__ccm_ccsr_bits);
__IO_REG32_BIT(CCM_CACRR,                 0x53FD4010,__READ_WRITE ,__ccm_cacrr_bits);
__IO_REG32_BIT(CCM_CBCDR,                 0x53FD4014,__READ_WRITE ,__ccm_cbcdr_bits);
__IO_REG32_BIT(CCM_CBCMR,                 0x53FD4018,__READ_WRITE ,__ccm_cbcmr_bits);
__IO_REG32_BIT(CCM_CSCMR1,                0x53FD401C,__READ_WRITE ,__ccm_cscmr1_bits);
//__IO_REG32_BIT(CCM_CSCMR2,                0x53FD4020,__READ_WRITE ,__ccm_cscmr2_bits);
__IO_REG32_BIT(CCM_CSCDR1,                0x53FD4024,__READ_WRITE ,__ccm_cscdr1_bits);
__IO_REG32_BIT(CCM_CS1CDR,                0x53FD4028,__READ_WRITE ,__ccm_cs1cdr_bits);
__IO_REG32_BIT(CCM_CS2CDR,                0x53FD402C,__READ_WRITE ,__ccm_cs2cdr_bits);
//__IO_REG32_BIT(CCM_CDCDR,                 0x53FD4030,__READ_WRITE ,__ccm_cdcdr_bits);
//__IO_REG32_BIT(CCM_CHSCCDR,               0x53FD4034,__READ_WRITE ,__ccm_chsccdr_bits);
__IO_REG32_BIT(CCM_CSCDR2,                0x53FD4038,__READ_WRITE ,__ccm_cscdr2_bits);
//__IO_REG32_BIT(CCM_CSCDR3,                0x53FD403C,__READ_WRITE ,__ccm_cscdr3_bits);
//__IO_REG32_BIT(CCM_CSCDR4,                0x53FD4040,__READ_WRITE ,__ccm_cscdr4_bits);
//__IO_REG32_BIT(CCM_CWDR,                  0x53FD4044,__READ_WRITE ,__ccm_cwdr_bits);
__IO_REG32_BIT(CCM_CDHIPR,                0x53FD4048,__READ       ,__ccm_cdhipr_bits);
__IO_REG32_BIT(CCM_CDCR,                  0x53FD404C,__READ_WRITE ,__ccm_cdcr_bits);
__IO_REG32_BIT(CCM_CTOR,                  0x53FD4050,__READ_WRITE ,__ccm_ctor_bits);
__IO_REG32_BIT(CCM_CLPCR,                 0x53FD4054,__READ_WRITE ,__ccm_clpcr_bits);
__IO_REG32_BIT(CCM_CISR,                  0x53FD4058,__READ_WRITE ,__ccm_cisr_bits);
__IO_REG32_BIT(CCM_CIMR,                  0x53FD405C,__READ_WRITE ,__ccm_cimr_bits);
__IO_REG32_BIT(CCM_CCOSR,                 0x53FD4060,__READ_WRITE ,__ccm_ccosr_bits);
//__IO_REG32_BIT(CCM_CGPR,                  0x53FD4064,__READ_WRITE ,__ccm_cgpr_bits);
__IO_REG32_BIT(CCM_CCGR0,                 0x53FD4068,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR1,                 0x53FD406C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR2,                 0x53FD4070,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR3,                 0x53FD4074,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR4,                 0x53FD4078,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR5,                 0x53FD407C,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR6,                 0x53FD4080,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CCGR7,                 0x53FD4084,__READ_WRITE ,__ccm_ccgr_bits);
__IO_REG32_BIT(CCM_CMEOR,                 0x53FD4088,__READ_WRITE ,__ccm_cmeor_bits);
__IO_REG32_BIT(CCM_CSR2,                  0x53FD408C,__READ_WRITE ,__ccm_csr2_bits);
__IO_REG32_BIT(CCM_CLKSEQ_BYPASS,         0x53FD4090,__READ_WRITE ,__ccm_clkseq_bypass_bits);
__IO_REG32_BIT(CCM_CLK_SYS,               0x53FD4094,__READ_WRITE ,__ccm_clk_sys_bits);
__IO_REG32_BIT(CCM_CLK_DDR,               0x53FD4098,__READ_WRITE ,__ccm_clk_ddr_bits);
__IO_REG32_BIT(CCM_ELCDIFPIX,             0x53FD409C,__READ_WRITE ,__ccm_elcdifpix_bits);
__IO_REG32_BIT(CCM_EPDCPIX,               0x53FD40A0,__READ_WRITE ,__ccm_epdcpix_bits);
__IO_REG32_BIT(CCM_DISPLAY_AXI,           0x53FD40A4,__READ_WRITE ,__ccm_display_axi_bits);
__IO_REG32_BIT(CCM_EPDC_AXI,              0x53FD40A8,__READ_WRITE ,__ccm_epdc_axi_bits);
__IO_REG32_BIT(CCM_GPMI,                  0x53FD40AC,__READ_WRITE ,__ccm_gpmi_bits);
__IO_REG32_BIT(CCM_BCH,                   0x53FD40B0,__READ_WRITE ,__ccm_bch_bits);
__IO_REG32_BIT(CCM_MSHC_XMSCKI,           0x53FD40B4,__READ_WRITE ,__ccm_mshc_xmscki_bits);

/***************************************************************************
 **
 **  CCM_ANALOG
 **
 ***************************************************************************/
__IO_REG32_BIT(CCM_ANALOG_FRAC0,          0x41018010,__READ_WRITE ,__ccm_analog_frac0_bits);
__IO_REG32(    CCM_ANALOG_FRAC0_SET,      0x41018014,__WRITE );
__IO_REG32(    CCM_ANALOG_FRAC0_CLR,      0x41018018,__WRITE );
__IO_REG32(    CCM_ANALOG_FRAC0_TOG,      0x4101801C,__WRITE );
__IO_REG32_BIT(CCM_ANALOG_FRAC1,          0x41018020,__READ_WRITE ,__ccm_analog_frac1_bits);
__IO_REG32(    CCM_ANALOG_FRAC1_SET,      0x41018024,__WRITE );
__IO_REG32(    CCM_ANALOG_FRAC1_CLR,      0x41018028,__WRITE );
__IO_REG32(    CCM_ANALOG_FRAC1_TOG,      0x4101802C,__WRITE );
__IO_REG32_BIT(CCM_ANALOG_MISC,           0x41018060,__READ_WRITE ,__ccm_analog_misc_bits);
__IO_REG32(    CCM_ANALOG_MISC_SET,       0x41018064,__WRITE );
__IO_REG32(    CCM_ANALOG_MISC_CLR,       0x41018068,__WRITE );
__IO_REG32(    CCM_ANALOG_MISC_TOG,       0x4101806C,__WRITE );
__IO_REG32_BIT(CCM_ANALOG_PLLCTRL,        0x41018070,__READ_WRITE ,__ccm_analog_pllctrl_bits);
__IO_REG32(    CCM_ANALOG_PLLCTRL_SET,    0x41018074,__WRITE );
__IO_REG32(    CCM_ANALOG_PLLCTRL_CLR,    0x41018078,__WRITE );
__IO_REG32(    CCM_ANALOG_PLLCTRL_TOG,    0x4101807C,__WRITE );

/***************************************************************************
 **
 **  ARM Platform
 **
 ***************************************************************************/
__IO_REG32_BIT(ARM_PVID,                  0x63FA0000,__READ       ,__arm_pvid_bits);
__IO_REG32_BIT(ARM_GPC,                   0x63FA0004,__READ_WRITE ,__arm_gpc_bits);
__IO_REG32_BIT(ARM_LPC,                   0x63FA000C,__READ_WRITE ,__arm_lpc_bits);
__IO_REG32_BIT(ARM_NLPC,                  0x63FA0010,__READ_WRITE ,__arm_nlpc_bits);
__IO_REG32_BIT(ARM_ICGC,                  0x63FA0014,__READ_WRITE ,__arm_icgc_bits);
__IO_REG32_BIT(ARM_AMC,                   0x63FA0018,__READ_WRITE ,__arm_amc_bits);
__IO_REG32_BIT(ARM_NMC,                   0x63FA0020,__READ_WRITE ,__arm_nmc_bits);
__IO_REG32_BIT(ARM_NMS,                   0x63FA0024,__READ_WRITE ,__arm_nms_bits);

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
 **  APBH-Bridge-DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(APBH_CTRL0,                0x41000000,__READ_WRITE ,__apbh_ctrl0_bits);
__IO_REG32(    APBH_CTRL0_SET,            0x41000004,__WRITE      );
__IO_REG32(    APBH_CTRL0_CLR,            0x41000008,__WRITE      );
__IO_REG32(    APBH_CTRL0_TOG,            0x4100000C,__WRITE      );
__IO_REG32_BIT(APBH_CTRL1,                0x41000010,__READ_WRITE ,__apbh_ctrl1_bits);
__IO_REG32(    APBH_CTRL1_SET,            0x41000014,__WRITE      );
__IO_REG32(    APBH_CTRL1_CLR,            0x41000018,__WRITE      );
__IO_REG32(    APBH_CTRL1_TOG,            0x4100001C,__WRITE      );
__IO_REG32_BIT(APBH_CTRL2,                0x41000020,__READ_WRITE ,__apbh_ctrl2_bits);
__IO_REG32(    APBH_CTRL2_SET,            0x41000024,__WRITE      );
__IO_REG32(    APBH_CTRL2_CLR,            0x41000028,__WRITE      );
__IO_REG32(    APBH_CTRL2_TOG,            0x4100002C,__WRITE      );
__IO_REG32_BIT(APBH_CHANNEL_CTRL,         0x41000030,__READ_WRITE ,__apbh_channel_ctrl_bits);
__IO_REG32(    APBH_CHANNEL_CTRL_SET,     0x41000034,__WRITE      );
__IO_REG32(    APBH_CHANNEL_CTRL_CLR,     0x41000038,__WRITE      );
__IO_REG32(    APBH_CHANNEL_CTRL_TOG,     0x4100003C,__WRITE      );
//__IO_REG32_BIT(APBH_DEVSEL,               0x41000040,__READ_WRITE ,__apbh_devsel_bits);
__IO_REG32_BIT(APBH_DMA_BURST_SIZE,       0x41000050,__READ_WRITE ,__apbh_dma_burst_size_bits);
__IO_REG32_BIT(APBH_DEBUG,                0x41000060,__READ_WRITE ,__apbh_debug_bits);
__IO_REG32(    APBH_CH0_CURCMDAR,         0x41000100,__READ       );
__IO_REG32(    APBH_CH0_NXTCMDAR,         0x41000110,__READ_WRITE );
__IO_REG32_BIT(APBH_CH0_CMD,              0x41000120,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH0_BAR,              0x41000130,__READ       );
__IO_REG32_BIT(APBH_CH0_SEMA,             0x41000140,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH0_DEBUG1,           0x41000150,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH0_DEBUG2,           0x41000160,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH1_CURCMDAR,         0x41000170,__READ       );
__IO_REG32(    APBH_CH1_NXTCMDAR,         0x41000180,__READ_WRITE );
__IO_REG32_BIT(APBH_CH1_CMD,              0x41000190,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH1_BAR,              0x410001A0,__READ       );
__IO_REG32_BIT(APBH_CH1_SEMA,             0x410001B0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH1_DEBUG1,           0x410001C0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH1_DEBUG2,           0x410001D0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH2_CURCMDAR,         0x410001E0,__READ       );
__IO_REG32(    APBH_CH2_NXTCMDAR,         0x410001F0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH2_CMD,              0x41000200,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH2_BAR,              0x41000210,__READ       );
__IO_REG32_BIT(APBH_CH2_SEMA,             0x41000220,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH2_DEBUG1,           0x41000230,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH2_DEBUG2,           0x41000240,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH3_CURCMDAR,         0x41000250,__READ       );
__IO_REG32(    APBH_CH3_NXTCMDAR,         0x41000260,__READ_WRITE );
__IO_REG32_BIT(APBH_CH3_CMD,              0x41000270,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH3_BAR,              0x41000280,__READ       );
__IO_REG32_BIT(APBH_CH3_SEMA,             0x41000290,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH3_DEBUG1,           0x410002A0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH3_DEBUG2,           0x410002B0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH4_CURCMDAR,         0x410002C0,__READ       );
__IO_REG32(    APBH_CH4_NXTCMDAR,         0x410002D0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH4_CMD,              0x410002E0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH4_BAR,              0x410002F0,__READ       );
__IO_REG32_BIT(APBH_CH4_SEMA,             0x41000300,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH4_DEBUG1,           0x41000310,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH4_DEBUG2,           0x41000320,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH5_CURCMDAR,         0x41000330,__READ       );
__IO_REG32(    APBH_CH5_NXTCMDAR,         0x41000340,__READ_WRITE );
__IO_REG32_BIT(APBH_CH5_CMD,              0x41000350,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH5_BAR,              0x41000360,__READ       );
__IO_REG32_BIT(APBH_CH5_SEMA,             0x41000370,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH5_DEBUG1,           0x41000380,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH5_DEBUG2,           0x41000390,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH6_CURCMDAR,         0x410003A0,__READ       );
__IO_REG32(    APBH_CH6_NXTCMDAR,         0x410003B0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH6_CMD,              0x410003C0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH6_BAR,              0x410003D0,__READ       );
__IO_REG32_BIT(APBH_CH6_SEMA,             0x410003E0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH6_DEBUG1,           0x410003F0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH6_DEBUG2,           0x41000400,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH7_CURCMDAR,         0x41000410,__READ       );
__IO_REG32(    APBH_CH7_NXTCMDAR,         0x41000420,__READ_WRITE );
__IO_REG32_BIT(APBH_CH7_CMD,              0x41000430,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH7_BAR,              0x41000440,__READ       );
__IO_REG32_BIT(APBH_CH7_SEMA,             0x41000450,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH7_DEBUG1,           0x41000460,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH7_DEBUG2,           0x41000470,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH8_CURCMDAR,         0x41000480,__READ       );
__IO_REG32(    APBH_CH8_NXTCMDAR,         0x41000490,__READ_WRITE );
__IO_REG32_BIT(APBH_CH8_CMD,              0x410004A0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH8_BAR,              0x410004B0,__READ       );
__IO_REG32_BIT(APBH_CH8_SEMA,             0x410004C0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH8_DEBUG1,           0x410004D0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH8_DEBUG2,           0x410004E0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH9_CURCMDAR,         0x410004F0,__READ       );
__IO_REG32(    APBH_CH9_NXTCMDAR,         0x41000500,__READ_WRITE );
__IO_REG32_BIT(APBH_CH9_CMD,              0x41000510,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH9_BAR,              0x41000520,__READ       );
__IO_REG32_BIT(APBH_CH9_SEMA,             0x41000530,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH9_DEBUG1,           0x41000540,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH9_DEBUG2,           0x41000550,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH10_CURCMDAR,        0x41000560,__READ       );
__IO_REG32(    APBH_CH10_NXTCMDAR,        0x41000570,__READ_WRITE );
__IO_REG32_BIT(APBH_CH10_CMD,             0x41000580,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH10_BAR,             0x41000590,__READ       );
__IO_REG32_BIT(APBH_CH10_SEMA,            0x410005A0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH10_DEBUG1,          0x410005B0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH10_DEBUG2,          0x410005C0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH11_CURCMDAR,        0x410005D0,__READ       );
__IO_REG32(    APBH_CH11_NXTCMDAR,        0x410005E0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH11_CMD,             0x410005F0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH11_BAR,             0x41000600,__READ       );
__IO_REG32_BIT(APBH_CH11_SEMA,            0x41000610,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH11_DEBUG1,          0x41000620,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH11_DEBUG2,          0x41000630,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH12_CURCMDAR,        0x41000640,__READ       );
__IO_REG32(    APBH_CH12_NXTCMDAR,        0x41000650,__READ_WRITE );
__IO_REG32_BIT(APBH_CH12_CMD,             0x41000660,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH12_BAR,             0x41000670,__READ       );
__IO_REG32_BIT(APBH_CH12_SEMA,            0x41000680,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH12_DEBUG1,          0x41000690,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH12_DEBUG2,          0x410006A0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH13_CURCMDAR,        0x410006B0,__READ       );
__IO_REG32(    APBH_CH13_NXTCMDAR,        0x410006C0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH13_CMD,             0x410006D0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH13_BAR,             0x410006E0,__READ       );
__IO_REG32_BIT(APBH_CH13_SEMA,            0x410006F0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH13_DEBUG1,          0x41000700,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH13_DEBUG2,          0x41000710,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH14_CURCMDAR,        0x41000720,__READ       );
__IO_REG32(    APBH_CH14_NXTCMDAR,        0x41000730,__READ_WRITE );
__IO_REG32_BIT(APBH_CH14_CMD,             0x41000740,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH14_BAR,             0x41000750,__READ       );
__IO_REG32_BIT(APBH_CH14_SEMA,            0x41000760,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH14_DEBUG1,          0x41000770,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH14_DEBUG2,          0x41000780,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32(    APBH_CH15_CURCMDAR,        0x41000790,__READ       );
__IO_REG32(    APBH_CH15_NXTCMDAR,        0x410007A0,__READ_WRITE );
__IO_REG32_BIT(APBH_CH15_CMD,             0x410007B0,__READ       ,__apbh_chn_cmd_bits);
__IO_REG32(    APBH_CH15_BAR,             0x410007C0,__READ       );
__IO_REG32_BIT(APBH_CH15_SEMA,            0x410007D0,__READ_WRITE ,__apbh_chn_sema_bits);
__IO_REG32_BIT(APBH_CH15_DEBUG1,          0x410007E0,__READ       ,__apbh_chn_debug1_bits);
__IO_REG32_BIT(APBH_CH15_DEBUG2,          0x410007F0,__READ       ,__apbh_chn_debug2_bits);
__IO_REG32_BIT(APBH_VERSION,              0x41000800,__READ       ,__apbh_version_bits);

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

/***************************************************************************
 **
 ** AIPSTZ1
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPSTZ1_MPR1,              0x53F00000,__READ_WRITE ,__aipstz_mpr1_bits);
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
__IO_REG32_BIT(AIPSTZ2_OPACR1,            0x63F00040,__READ_WRITE ,__aipstz2_opacr1_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR2,            0x63F00044,__READ_WRITE ,__aipstz2_opacr2_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR3,            0x63F00048,__READ_WRITE ,__aipstz2_opacr3_bits);
__IO_REG32_BIT(AIPSTZ2_OPACR4,            0x63F0004C,__READ_WRITE ,__aipstz2_opacr4_bits);

/***************************************************************************
 **
 ** BCH ECC
 **
 ***************************************************************************/
__IO_REG32_BIT(BCH_CTRL,                  0x41008000,__READ_WRITE ,__bch_ctrl_bits);
__IO_REG32(    BCH_CTRL_SET,              0x41008004,__WRITE      );
__IO_REG32(    BCH_CTRL_CLR,              0x41008008,__WRITE      );
__IO_REG32(    BCH_CTRL_TOG,              0x4100800C,__WRITE      );
__IO_REG32_BIT(BCH_STATUS0,               0x41008010,__READ       ,__bch_status0_bits);
__IO_REG32_BIT(BCH_MODE,                  0x41008020,__READ_WRITE ,__bch_mode_bits);
__IO_REG32(    BCH_ENCODEPTR,             0x41008030,__READ_WRITE );
__IO_REG32(    BCH_DATAPTR,               0x41008040,__READ_WRITE );
__IO_REG32(    BCH_METAPTR,               0x41008050,__READ_WRITE );
__IO_REG32_BIT(BCH_LAYOUTSELECT,          0x41008070,__READ_WRITE ,__bch_layoutselect_bits);
__IO_REG32_BIT(BCH_FLASH0LAYOUT0,         0x41008080,__READ_WRITE ,__bch_flashxlayout0_bits);
__IO_REG32_BIT(BCH_FLASH0LAYOUT1,         0x41008090,__READ_WRITE ,__bch_flashxlayout1_bits);
__IO_REG32_BIT(BCH_FLASH1LAYOUT0,         0x410080A0,__READ_WRITE ,__bch_flashxlayout0_bits);
__IO_REG32_BIT(BCH_FLASH1LAYOUT1,         0x410080B0,__READ_WRITE ,__bch_flashxlayout1_bits);
__IO_REG32_BIT(BCH_FLASH2LAYOUT0,         0x410080C0,__READ_WRITE ,__bch_flashxlayout0_bits);
__IO_REG32_BIT(BCH_FLASH2LAYOUT1,         0x410080D0,__READ_WRITE ,__bch_flashxlayout1_bits);
__IO_REG32_BIT(BCH_FLASH3LAYOUT0,         0x410080E0,__READ_WRITE ,__bch_flashxlayout0_bits);
__IO_REG32_BIT(BCH_FLASH3LAYOUT1,         0x410080F0,__READ_WRITE ,__bch_flashxlayout1_bits);
__IO_REG32_BIT(BCH_DEBUG0,                0x41008100,__READ_WRITE ,__bch_debug0_bits);
__IO_REG32(    BCH_DEBUG0_SET,            0x41008104,__WRITE      );
__IO_REG32(    BCH_DEBUG0_CLR,            0x41008108,__WRITE      );
__IO_REG32(    BCH_DEBUG0_TOG,            0x4100810C,__WRITE      );
__IO_REG32(    BCH_DBGKESREAD,            0x41008110,__READ       );
__IO_REG32(    BCH_DBGCSFEREAD,           0x41008120,__READ       );
__IO_REG32(    BCH_DBGSYNDGENREAD,        0x41008130,__READ       );
__IO_REG32(    BCH_DBGAHBMREAD,           0x41008140,__READ       );
__IO_REG32(    BCH_BLOCKNAME,             0x41008150,__READ       );
__IO_REG32_BIT(BCH_VERSION,               0x41008160,__READ       ,__bch_version_bits);

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
 **  DCP
 **
 ***************************************************************************/
__IO_REG32_BIT(DCP_CTRL,               0x4100E000,__READ_WRITE,__dcp_ctrl_bits);
__IO_REG32(    DCP_CTRL_SET,           0x4100E004,__WRITE     );
__IO_REG32(    DCP_CTRL_CLR,           0x4100E008,__WRITE     );
__IO_REG32(    DCP_CTRL_TOG,           0x4100E00C,__WRITE     );
__IO_REG32_BIT(DCP_STAT,               0x4100E010,__READ_WRITE,__dcp_stat_bits);
__IO_REG32(    DCP_STAT_SET,           0x4100E014,__WRITE     );
__IO_REG32(    DCP_STAT_CLR,           0x4100E018,__WRITE     );
__IO_REG32(    DCP_STAT_TOG,           0x4100E01C,__WRITE     );
__IO_REG32_BIT(DCP_CHANNELCTRL,        0x4100E020,__READ_WRITE,__dcp_channelctrl_bits);
__IO_REG32(    DCP_CHANNELCTRL_SET,    0x4100E024,__WRITE     );
__IO_REG32(    DCP_CHANNELCTRL_CLR,    0x4100E028,__WRITE     );
__IO_REG32(    DCP_CHANNELCTRL_TOG,    0x4100E02C,__WRITE     );
__IO_REG32_BIT(DCP_CAPABILITY0,        0x4100E030,__READ_WRITE,__dcp_capability0_bits);
__IO_REG32_BIT(DCP_CAPABILITY1,        0x4100E040,__READ      ,__dcp_capability1_bits);
__IO_REG32(    DCP_CONTEXT,            0x4100E050,__READ_WRITE);
__IO_REG32_BIT(DCP_KEY,                0x4100E060,__READ_WRITE,__dcp_key_bits);
__IO_REG32(    DCP_KEYDATA,            0x4100E070,__READ_WRITE);
__IO_REG32(    DCP_PACKET0,            0x4100E080,__READ      );
__IO_REG32_BIT(DCP_PACKET1,            0x4100E090,__READ      ,__dcp_packet1_bits);
__IO_REG32_BIT(DCP_PACKET2,            0x4100E0A0,__READ      ,__dcp_packet2_bits);
__IO_REG32(    DCP_PACKET3,            0x4100E0B0,__READ      );
__IO_REG32(    DCP_PACKET4,            0x4100E0C0,__READ      );
__IO_REG32(    DCP_PACKET5,            0x4100E0D0,__READ      );
__IO_REG32(    DCP_PACKET6,            0x4100E0E0,__READ      );
__IO_REG32(    DCP_CH0CMDPTR,          0x4100E100,__READ_WRITE);
__IO_REG32_BIT(DCP_CH0SEMA,            0x4100E110,__READ_WRITE,__dcp_chsema_bits);
__IO_REG32_BIT(DCP_CH0STAT,            0x4100E120,__READ_WRITE,__dcp_chstat_bits);
__IO_REG32(    DCP_CH0STAT_SET,        0x4100E124,__WRITE     );
__IO_REG32(    DCP_CH0STAT_CLR,        0x4100E128,__WRITE     );
__IO_REG32(    DCP_CH0STAT_TOG,        0x4100E12C,__WRITE     );
__IO_REG32_BIT(DCP_CH0OPTS,            0x4100E130,__READ_WRITE,__dcp_chopts_bits);
__IO_REG32(    DCP_CH0OPTS_SET,        0x4100E134,__WRITE     );
__IO_REG32(    DCP_CH0OPTS_CLR,        0x4100E138,__WRITE     );
__IO_REG32(    DCP_CH0OPTS_TOG,        0x4100E13C,__WRITE     );
__IO_REG32(    DCP_CH1CMDPTR,          0x4100E140,__READ_WRITE);
__IO_REG32_BIT(DCP_CH1SEMA,            0x4100E150,__READ_WRITE,__dcp_chsema_bits);
__IO_REG32_BIT(DCP_CH1STAT,            0x4100E160,__READ_WRITE,__dcp_chstat_bits);
__IO_REG32(    DCP_CH1STAT_SET,        0x4100E164,__WRITE     );
__IO_REG32(    DCP_CH1STAT_CLR,        0x4100E168,__WRITE     );
__IO_REG32(    DCP_CH1STAT_TOG,        0x4100E16C,__WRITE     );
__IO_REG32_BIT(DCP_CH1OPTS,            0x4100E170,__READ_WRITE,__dcp_chopts_bits);
__IO_REG32(    DCP_CH1OPTS_SET,        0x4100E174,__WRITE     );
__IO_REG32(    DCP_CH1OPTS_CLR,        0x4100E178,__WRITE     );
__IO_REG32(    DCP_CH1OPTS_TOG,        0x4100E17C,__WRITE     );
__IO_REG32(    DCP_CH2CMDPTR,          0x4100E180,__READ_WRITE);
__IO_REG32_BIT(DCP_CH2SEMA,            0x4100E190,__READ_WRITE,__dcp_chsema_bits);
__IO_REG32_BIT(DCP_CH2STAT,            0x4100E1A0,__READ_WRITE,__dcp_chstat_bits);
__IO_REG32(    DCP_CH2STAT_SET,        0x4100E1A4,__WRITE     );
__IO_REG32(    DCP_CH2STAT_CLR,        0x4100E1A8,__WRITE     );
__IO_REG32(    DCP_CH2STAT_TOG,        0x4100E1AC,__WRITE     );
__IO_REG32_BIT(DCP_CH2OPTS,            0x4100E1B0,__READ_WRITE,__dcp_chopts_bits);
__IO_REG32(    DCP_CH2OPTS_SET,        0x4100E1B4,__WRITE     );
__IO_REG32(    DCP_CH2OPTS_CLR,        0x4100E1B8,__WRITE     );
__IO_REG32(    DCP_CH2OPTS_TOG,        0x4100E1BC,__WRITE     );
__IO_REG32(    DCP_CH3CMDPTR,          0x4100E1C0,__READ_WRITE);
__IO_REG32_BIT(DCP_CH3SEMA,            0x4100E1D0,__READ_WRITE,__dcp_chsema_bits);
__IO_REG32_BIT(DCP_CH3STAT,            0x4100E1E0,__READ_WRITE,__dcp_chstat_bits);
__IO_REG32(    DCP_CH3STAT_SET,        0x4100E1E4,__WRITE     );
__IO_REG32(    DCP_CH3STAT_CLR,        0x4100E1E8,__WRITE     );
__IO_REG32(    DCP_CH3STAT_TOG,        0x4100E1EC,__WRITE     );
__IO_REG32_BIT(DCP_CH3OPTS,            0x4100E1F0,__READ_WRITE,__dcp_chopts_bits);
__IO_REG32(    DCP_CH3OPTS_SET,        0x4100E1F4,__WRITE     );
__IO_REG32(    DCP_CH3OPTS_CLR,        0x4100E1F8,__WRITE     );
__IO_REG32(    DCP_CH3OPTS_TOG,        0x4100E1FC,__WRITE     );
__IO_REG32_BIT(DCP_DBGSELECT,          0x4100E400,__READ_WRITE,__dcp_dbgselect_bits);
__IO_REG32(    DCP_DBGDATA,            0x4100E410,__READ      );
__IO_REG32_BIT(DCP_PAGETABLE,          0x4100E420,__READ_WRITE,__dcp_pagetable_bits);
__IO_REG32_BIT(DCP_VERSION,            0x4100E430,__READ      ,__dcp_version_bits);

/***************************************************************************
 **
 **  DIGCTL
 **
 ***************************************************************************/
__IO_REG32_BIT(DIGCTL_CTRL,             0x41004000,__READ_WRITE ,__digctl_ctrl_bits);
__IO_REG32(    DIGCTL_CTRL_SET,         0x41004004,__WRITE      );
__IO_REG32(    DIGCTL_CTRL_CLR,         0x41004008,__WRITE      );
__IO_REG32(    DIGCTL_CTRL_TOG,         0x4100400C,__WRITE      );
__IO_REG32_BIT(DIGCTL_OCRAM,            0x41004010,__READ_WRITE ,__digctl_ocram_bits);
__IO_REG32(    DIGCTL_OCRAM_SET,        0x41004014,__WRITE      );
__IO_REG32(    DIGCTL_OCRAM_CLR,        0x41004018,__WRITE      );
__IO_REG32(    DIGCTL_OCRAM_TOG,        0x4100401C,__WRITE      );
__IO_REG32_BIT(DIGCTL_SPEEDCTL,         0x41004020,__READ_WRITE ,__digctl_speedctl_bits);
__IO_REG32(    DIGCTL_SPEEDCTL_SET,     0x41004024,__WRITE      );
__IO_REG32(    DIGCTL_SPEEDCTL_CLR,     0x41004028,__WRITE      );
__IO_REG32(    DIGCTL_SPEEDCTL_TOG,     0x4100402C,__WRITE      );
__IO_REG32(    DIGCTL_DIGCTL_SPEEDSTAT, 0x41004030,__READ       );

/***************************************************************************
 **
 **  DPLLC1
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC1_CTL,             0x63F80000,__READ_WRITE ,__dpllc_ctl_bits);
__IO_REG32_BIT(DPLLC1_CONFIG,          0x63F80004,__READ_WRITE ,__dpllc_config_bits);
__IO_REG32_BIT(DPLLC1_OP,              0x63F80008,__READ_WRITE ,__dpllc_op_bits);
__IO_REG32_BIT(DPLLC1_MFD,             0x63F8000C,__READ_WRITE ,__dpllc_mfd_bits);
__IO_REG32_BIT(DPLLC1_MFN,             0x63F80010,__READ_WRITE ,__dpllc_mfn_bits);
__IO_REG32_BIT(DPLLC1_MFNMINUS,        0x63F80014,__READ_WRITE ,__dpllc_mfnminus_bits);
__IO_REG32_BIT(DPLLC1_MFNPLUS,         0x63F80018,__READ_WRITE ,__dpllc_mfnplus_bits);
__IO_REG32_BIT(DPLLC1_HFS_OP,          0x63F8001C,__READ_WRITE ,__dpllc_hfs_op_bits);
__IO_REG32_BIT(DPLLC1_HFS_MFD,         0x63F80020,__READ_WRITE ,__dpllc_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC1_HFS_MFN,         0x63F80024,__READ_WRITE ,__dpllc_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC1_MFN_TOGC,        0x63F80028,__READ_WRITE ,__dpllc_mfn_togc_bits);
__IO_REG32_BIT(DPLLC1_DESTAT,          0x63F8002C,__READ       ,__dpllc_destat_bits);

/***************************************************************************
 **
 **  DPLLC2
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC2_CTL,             0x63F84000,__READ_WRITE ,__dpllc_ctl_bits);
__IO_REG32_BIT(DPLLC2_CONFIG,          0x63F84004,__READ_WRITE ,__dpllc_config_bits);
__IO_REG32_BIT(DPLLC2_OP,              0x63F84008,__READ_WRITE ,__dpllc_op_bits);
__IO_REG32_BIT(DPLLC2_MFD,             0x63F8400C,__READ_WRITE ,__dpllc_mfd_bits);
__IO_REG32_BIT(DPLLC2_MFN,             0x63F84010,__READ_WRITE ,__dpllc_mfn_bits);
__IO_REG32_BIT(DPLLC2_MFNMINUS,        0x63F84014,__READ_WRITE ,__dpllc_mfnminus_bits);
__IO_REG32_BIT(DPLLC2_MFNPLUS,         0x63F84018,__READ_WRITE ,__dpllc_mfnplus_bits);
__IO_REG32_BIT(DPLLC2_HFS_OP,          0x63F8401C,__READ_WRITE ,__dpllc_hfs_op_bits);
__IO_REG32_BIT(DPLLC2_HFS_MFD,         0x63F84020,__READ_WRITE ,__dpllc_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC2_HFS_MFN,         0x63F84024,__READ_WRITE ,__dpllc_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC2_MFN_TOGC,        0x63F84028,__READ_WRITE ,__dpllc_mfn_togc_bits);
__IO_REG32_BIT(DPLLC2_DESTAT,          0x63F8402C,__READ       ,__dpllc_destat_bits);

/***************************************************************************
 **
 **  DPLLC3
 **
 ***************************************************************************/
__IO_REG32_BIT(DPLLC3_CTL,             0x63F88000,__READ_WRITE ,__dpllc_ctl_bits);
__IO_REG32_BIT(DPLLC3_CONFIG,          0x63F88004,__READ_WRITE ,__dpllc_config_bits);
__IO_REG32_BIT(DPLLC3_OP,              0x63F88008,__READ_WRITE ,__dpllc_op_bits);
__IO_REG32_BIT(DPLLC3_MFD,             0x63F8800C,__READ_WRITE ,__dpllc_mfd_bits);
__IO_REG32_BIT(DPLLC3_MFN,             0x63F88010,__READ_WRITE ,__dpllc_mfn_bits);
__IO_REG32_BIT(DPLLC3_MFNMINUS,        0x63F88014,__READ_WRITE ,__dpllc_mfnminus_bits);
__IO_REG32_BIT(DPLLC3_MFNPLUS,         0x63F88018,__READ_WRITE ,__dpllc_mfnplus_bits);
__IO_REG32_BIT(DPLLC3_HFS_OP,          0x63F8801C,__READ_WRITE ,__dpllc_hfs_op_bits);
__IO_REG32_BIT(DPLLC3_HFS_MFD,         0x63F88020,__READ_WRITE ,__dpllc_hfs_mfd_bits);
__IO_REG32_BIT(DPLLC3_HFS_MFN,         0x63F88024,__READ_WRITE ,__dpllc_hfs_mfn_bits);
__IO_REG32_BIT(DPLLC3_MFN_TOGC,        0x63F88028,__READ_WRITE ,__dpllc_mfn_togc_bits);
__IO_REG32_BIT(DPLLC3_DESTAT,          0x63F8802C,__READ       ,__dpllc_destat_bits);

/***************************************************************************
 **
 **  DRAM MC
 **
 ***************************************************************************/
__IO_REG32_BIT(DRAM_CTL00,             0x14000000,__READ_WRITE ,__dram_ctl00_bits);
__IO_REG32_BIT(DRAM_CTL01,             0x14000004,__READ       ,__dram_ctl01_bits);
__IO_REG32_BIT(DRAM_CTL02,             0x14000008,__READ_WRITE ,__dram_ctl02_bits);
__IO_REG32_BIT(DRAM_CTL03,             0x1400000C,__READ_WRITE ,__dram_ctl03_bits);
__IO_REG32_BIT(DRAM_CTL04,             0x14000010,__READ_WRITE ,__dram_ctl04_bits);
__IO_REG32_BIT(DRAM_CTL05,             0x14000014,__READ_WRITE ,__dram_ctl05_bits);
__IO_REG32_BIT(DRAM_CTL06,             0x14000018,__READ_WRITE ,__dram_ctl06_bits);
__IO_REG32_BIT(DRAM_CTL07,             0x1400001C,__READ_WRITE ,__dram_ctl07_bits);
__IO_REG32_BIT(DRAM_CTL08,             0x14000020,__READ_WRITE ,__dram_ctl08_bits);
__IO_REG32_BIT(DRAM_CTL09,             0x14000024,__READ_WRITE ,__dram_ctl09_bits);
__IO_REG32_BIT(DRAM_CTL10,             0x14000028,__READ_WRITE ,__dram_ctl10_bits);
__IO_REG32_BIT(DRAM_CTL11,             0x1400002C,__READ_WRITE ,__dram_ctl11_bits);
__IO_REG32_BIT(DRAM_CTL12,             0x14000030,__READ_WRITE ,__dram_ctl12_bits);
__IO_REG32_BIT(DRAM_CTL13,             0x14000034,__READ_WRITE ,__dram_ctl13_bits);
__IO_REG32_BIT(DRAM_CTL14,             0x14000038,__READ_WRITE ,__dram_ctl14_bits);
__IO_REG32_BIT(DRAM_CTL15,             0x1400003C,__READ_WRITE ,__dram_ctl15_bits);
__IO_REG32_BIT(DRAM_CTL16,             0x14000040,__READ_WRITE ,__dram_ctl16_bits);
__IO_REG32_BIT(DRAM_CTL17,             0x14000044,__READ_WRITE ,__dram_ctl17_bits);
__IO_REG32_BIT(DRAM_CTL18,             0x14000048,__READ_WRITE ,__dram_ctl18_bits);
__IO_REG32_BIT(DRAM_CTL19,             0x1400004C,__READ_WRITE ,__dram_ctl19_bits);
__IO_REG32_BIT(DRAM_CTL20,             0x14000050,__READ_WRITE ,__dram_ctl20_bits);
__IO_REG32_BIT(DRAM_CTL21,             0x14000054,__READ_WRITE ,__dram_ctl21_bits);
__IO_REG32_BIT(DRAM_CTL22,             0x14000058,__READ_WRITE ,__dram_ctl22_bits);
__IO_REG32_BIT(DRAM_CTL23,             0x1400005C,__READ_WRITE ,__dram_ctl23_bits);
__IO_REG32_BIT(DRAM_CTL24,             0x14000060,__READ_WRITE ,__dram_ctl24_bits);
__IO_REG32_BIT(DRAM_CTL25,             0x14000064,__READ_WRITE ,__dram_ctl25_bits);
__IO_REG32_BIT(DRAM_CTL26,             0x14000068,__READ_WRITE ,__dram_ctl26_bits);
__IO_REG32_BIT(DRAM_CTL27,             0x1400006C,__READ_WRITE ,__dram_ctl27_bits);
__IO_REG32_BIT(DRAM_CTL28,             0x14000070,__READ_WRITE ,__dram_ctl28_bits);
__IO_REG32_BIT(DRAM_CTL29,             0x14000074,__READ_WRITE ,__dram_ctl29_bits);
__IO_REG32_BIT(DRAM_CTL30,             0x14000078,__READ_WRITE ,__dram_ctl30_bits);
__IO_REG32_BIT(DRAM_CTL31,             0x1400007C,__READ_WRITE ,__dram_ctl31_bits);
__IO_REG32_BIT(DRAM_CTL32,             0x14000080,__READ_WRITE ,__dram_ctl32_bits);
__IO_REG32_BIT(DRAM_CTL33,             0x14000084,__READ_WRITE ,__dram_ctl33_bits);
__IO_REG32_BIT(DRAM_CTL34,             0x14000088,__READ_WRITE ,__dram_ctl34_bits);
__IO_REG32_BIT(DRAM_CTL35,             0x1400008C,__READ_WRITE ,__dram_ctl35_bits);
__IO_REG32_BIT(DRAM_CTL36,             0x14000090,__READ_WRITE ,__dram_ctl36_bits);
__IO_REG32_BIT(DRAM_CTL37,             0x14000094,__READ_WRITE ,__dram_ctl37_bits);
__IO_REG32_BIT(DRAM_CTL38,             0x14000098,__READ_WRITE ,__dram_ctl38_bits);
__IO_REG32_BIT(DRAM_CTL39,             0x1400009C,__READ_WRITE ,__dram_ctl39_bits);
__IO_REG32_BIT(DRAM_CTL40,             0x140000A0,__READ_WRITE ,__dram_ctl40_bits);
__IO_REG32_BIT(DRAM_CTL41,             0x140000A4,__READ_WRITE ,__dram_ctl41_bits);
__IO_REG32_BIT(DRAM_CTL42,             0x140000A8,__READ_WRITE ,__dram_ctl42_bits);
__IO_REG32_BIT(DRAM_CTL43,             0x140000AC,__READ_WRITE ,__dram_ctl43_bits);
__IO_REG32_BIT(DRAM_CTL44,             0x140000B0,__READ       ,__dram_ctl44_bits);
__IO_REG32_BIT(DRAM_CTL45,             0x140000B4,__READ       ,__dram_ctl45_bits);
__IO_REG32_BIT(DRAM_CTL46,             0x140000B8,__READ       ,__dram_ctl46_bits);
__IO_REG32_BIT(DRAM_CTL47,             0x140000BC,__READ       ,__dram_ctl47_bits);
__IO_REG32_BIT(DRAM_CTL48,             0x140000C0,__READ       ,__dram_ctl48_bits);
__IO_REG32_BIT(DRAM_CTL49,             0x140000C4,__READ       ,__dram_ctl49_bits);
__IO_REG32_BIT(DRAM_CTL50,             0x140000C8,__READ_WRITE ,__dram_ctl50_bits);
__IO_REG32_BIT(DRAM_CTL51,             0x140000CC,__READ_WRITE ,__dram_ctl51_bits);
__IO_REG32_BIT(DRAM_CTL52,             0x140000D0,__READ_WRITE ,__dram_ctl52_bits);
__IO_REG32_BIT(DRAM_CTL53,             0x140000D4,__READ_WRITE ,__dram_ctl53_bits);
__IO_REG32_BIT(DRAM_CTL54,             0x140000D8,__READ_WRITE ,__dram_ctl54_bits);
__IO_REG32_BIT(DRAM_CTL55,             0x140000DC,__READ_WRITE ,__dram_ctl55_bits);
__IO_REG32_BIT(DRAM_CTL56,             0x140000E0,__READ_WRITE ,__dram_ctl56_bits);
__IO_REG32_BIT(DRAM_CTL57,             0x140000E4,__READ_WRITE ,__dram_ctl57_bits);
__IO_REG32_BIT(DRAM_CTL58,             0x140000E8,__READ_WRITE ,__dram_ctl58_bits);
__IO_REG32_BIT(DRAM_CTL59,             0x140000EC,__READ_WRITE ,__dram_ctl59_bits);
__IO_REG32_BIT(DRAM_CTL60,             0x140000F0,__READ_WRITE ,__dram_ctl60_bits);
__IO_REG32_BIT(DRAM_CTL61,             0x140000F4,__READ_WRITE ,__dram_ctl61_bits);
__IO_REG32_BIT(DRAM_CTL62,             0x140000F8,__READ_WRITE ,__dram_ctl62_bits);
__IO_REG32_BIT(DRAM_CTL63,             0x140000FC,__READ_WRITE ,__dram_ctl63_bits);
__IO_REG32_BIT(DRAM_CTL64,             0x14000100,__READ_WRITE ,__dram_ctl64_bits);
__IO_REG32_BIT(DRAM_CTL65,             0x14000104,__READ_WRITE ,__dram_ctl65_bits);
__IO_REG32_BIT(DRAM_CTL66,             0x14000108,__READ_WRITE ,__dram_ctl66_bits);
__IO_REG32_BIT(DRAM_CTL67,             0x1400010C,__READ_WRITE ,__dram_ctl67_bits);
__IO_REG32_BIT(DRAM_CTL68,             0x14000110,__READ_WRITE ,__dram_ctl68_bits);
__IO_REG32_BIT(DRAM_CTL69,             0x14000114,__READ_WRITE ,__dram_ctl69_bits);
__IO_REG32_BIT(DRAM_CTL70,             0x14000118,__READ_WRITE ,__dram_ctl70_bits);
__IO_REG32_BIT(DRAM_CTL71,             0x1400011C,__READ_WRITE ,__dram_ctl71_bits);
__IO_REG32_BIT(DRAM_CTL72,             0x14000120,__READ_WRITE ,__dram_ctl72_bits);
__IO_REG32_BIT(DRAM_CTL73,             0x14000124,__READ_WRITE ,__dram_ctl73_bits);
__IO_REG32_BIT(DRAM_CTL74,             0x14000128,__READ_WRITE ,__dram_ctl74_bits);
__IO_REG32_BIT(DRAM_CTL75,             0x1400012C,__READ_WRITE ,__dram_ctl75_bits);
__IO_REG32(    DRAM_CTL76,             0x14000130,__READ_WRITE );
__IO_REG32(    DRAM_CTL77,             0x14000134,__READ_WRITE );
__IO_REG32(    DRAM_CTL78,             0x14000138,__READ_WRITE );
__IO_REG32_BIT(DRAM_CTL79,             0x1400013C,__READ       ,__dram_ctl79_bits);
__IO_REG32(    DRAM_CTL80,             0x14000140,__READ       );
__IO_REG32(    DRAM_CTL81,             0x14000144,__READ       );
__IO_REG32(    DRAM_CTL82,             0x14000148,__READ       );
__IO_REG32_BIT(DRAM_CTL83,             0x1400014C,__READ       ,__dram_ctl83_bits);
__IO_REG32(    DRAM_CTL84,             0x14000150,__READ       );
__IO_REG32(    DRAM_CTL85,             0x14000154,__READ       );
__IO_REG32(    DRAM_CTL86,             0x14000158,__READ       );
__IO_REG32_BIT(DRAM_PHY01,             0x14000204,__READ_WRITE ,__dram_phy01_bits);
__IO_REG32_BIT(DRAM_PHY02,             0x14000208,__READ_WRITE ,__dram_phy02_bits);
__IO_REG32_BIT(DRAM_PHY03,             0x1400020C,__READ_WRITE ,__dram_phy03_bits);
__IO_REG32(    DRAM_PHY04,             0x14000210,__READ_WRITE );
__IO_REG32(    DRAM_PHY05,             0x14000214,__READ_WRITE );
__IO_REG32(    DRAM_PHY06,             0x14000218,__READ_WRITE );
__IO_REG32(    DRAM_PHY07,             0x1400021C,__READ_WRITE );
__IO_REG32(    DRAM_PHY08,             0x14000220,__READ_WRITE );
__IO_REG32(    DRAM_PHY09,             0x14000224,__READ_WRITE );
__IO_REG32(    DRAM_PHY10,             0x14000228,__READ_WRITE );
__IO_REG32(    DRAM_PHY11,             0x1400022C,__READ_WRITE );
__IO_REG32_BIT(DRAM_PHY13,             0x14000234,__READ_WRITE ,__dram_phy13_bits);
__IO_REG32_BIT(DRAM_PHY14,             0x14000238,__READ_WRITE ,__dram_phy14_bits);
__IO_REG32_BIT(DRAM_PHY15,             0x1400023C,__READ_WRITE ,__dram_phy15_bits);
__IO_REG32(    DRAM_PHY16,             0x14000240,__READ_WRITE );
__IO_REG32(    DRAM_PHY17,             0x14000244,__READ_WRITE );
__IO_REG32(    DRAM_PHY18,             0x14000248,__READ_WRITE );
__IO_REG32(    DRAM_PHY19,             0x1400024C,__READ_WRITE );
__IO_REG32(    DRAM_PHY20,             0x14000250,__READ_WRITE );
__IO_REG32(    DRAM_PHY21,             0x14000254,__READ_WRITE );
__IO_REG32(    DRAM_PHY22,             0x14000258,__READ_WRITE );
__IO_REG32(    DRAM_PHY23,             0x1400025C,__READ_WRITE );
__IO_REG32_BIT(DRAM_PHY24,             0x14000260,__READ       ,__dram_phy24_bits);
__IO_REG32_BIT(DRAM_PHY25,             0x14000264,__READ       ,__dram_phy25_bits);
__IO_REG32_BIT(DRAM_PHY26,             0x14000268,__READ       ,__dram_phy26_bits);
__IO_REG32(    DRAM_PHY27,             0x1400026C,__READ       );
__IO_REG32(    DRAM_PHY28,             0x14000270,__READ       );
__IO_REG32(    DRAM_PHY29,             0x14000274,__READ       );
__IO_REG32(    DRAM_PHY30,             0x14000278,__READ       );
__IO_REG32(    DRAM_PHY31,             0x1400027C,__READ       );
__IO_REG32(    DRAM_PHY32,             0x14000280,__READ       );
__IO_REG32(    DRAM_PHY33,             0x14000284,__READ       );
__IO_REG32(    DRAM_PHY34,             0x14000288,__READ       );
__IO_REG32(    DRAM_PHY35,             0x1400028C,__READ       );
__IO_REG32(    DRAM_PHY36,             0x14000290,__READ       );
__IO_REG32(    DRAM_PHY37,             0x14000294,__READ       );
__IO_REG32(    DRAM_PHY38,             0x14000298,__READ       );

/***************************************************************************
 **
 **  DVFSC
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSC_THRS,             0x53FD8180,__READ_WRITE ,__dvfsc_thrs_bits);
__IO_REG32_BIT(DVFSC_COUN,             0x53FD8184,__READ_WRITE ,__dvfsc_coun_bits);
__IO_REG32_BIT(DVFSC_SIG1,             0x53FD8188,__READ_WRITE ,__dvfsc_sig1_bits);
__IO_REG32_BIT(DVFSC_SIG0,             0x53FD818C,__READ_WRITE ,__dvfsc_sig0_bits);
__IO_REG32_BIT(DVFSC_GPC0,             0x53FD8190,__READ_WRITE ,__dvfsc_gpc0_bits);
__IO_REG32_BIT(DVFSC_GPC1,             0x53FD8194,__READ_WRITE ,__dvfsc_gpc1_bits);
__IO_REG32_BIT(DVFSC_GPBT,             0x53FD8198,__READ_WRITE ,__dvfsc_gpbt_bits);
__IO_REG32_BIT(DVFSC_EMAC,             0x53FD819C,__READ_WRITE ,__dvfsc_emac_bits);
__IO_REG32_BIT(DVFSC_CNTR,             0x53FD81A0,__READ_WRITE ,__dvfsc_cntr_bits);
__IO_REG32_BIT(DVFSC_LTR0_0,           0x53FD81A4,__READ       ,__dvfsc_ltr0_0_bits);
__IO_REG32_BIT(DVFSC_LTR0_1,           0x53FD81A8,__READ       ,__dvfsc_ltr0_1_bits);
__IO_REG32_BIT(DVFSC_LTR1_0,           0x53FD81AC,__READ       ,__dvfsc_ltr1_0_bits);
__IO_REG32_BIT(DVFSC_LTR1_1,           0x53FD81B0,__READ       ,__dvfsc_ltr1_1_bits);
__IO_REG32_BIT(DVFSC_PT0,              0x53FD81B4,__READ_WRITE ,__dvfsc_pt0_bits);
__IO_REG32_BIT(DVFSC_PT1,              0x53FD81B8,__READ_WRITE ,__dvfsc_pt1_bits);
__IO_REG32_BIT(DVFSC_PT2,              0x53FD81BC,__READ_WRITE ,__dvfsc_pt2_bits);
__IO_REG32_BIT(DVFSC_PT3,              0x53FD81C0,__READ_WRITE ,__dvfsc_pt3_bits);

/***************************************************************************
 **
 **  DVFSP
 **
 ***************************************************************************/
__IO_REG32_BIT(DVFSP_LTR0,             0x53FD81C8,__READ       ,__dvfsp_ltr0_bits);
__IO_REG32_BIT(DVFSP_LTR1,             0x53FD81CC,__READ       ,__dvfsp_ltr1_bits);
__IO_REG32_BIT(DVFSP_LTR2,             0x53FD81D0,__READ       ,__dvfsp_ltr2_bits);
__IO_REG32_BIT(DVFSP_LTR3,             0x53FD81D4,__READ       ,__dvfsp_ltr3_bits);
__IO_REG32_BIT(DVFSP_LTBR0,            0x53FD81D8,__READ_WRITE ,__dvfsp_ltbr0_bits);
__IO_REG32_BIT(DVFSP_LTBR1,            0x53FD81DC,__READ_WRITE ,__dvfsp_ltbr1_bits);
__IO_REG32_BIT(DVFSP_PMCR0,            0x53FD81E0,__READ_WRITE ,__dvfsp_pmcr0_bits);
__IO_REG32_BIT(DVFSP_PMCR1,            0x53FD81E4,__READ_WRITE ,__dvfsp_pmcr1_bits);

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
__IO_REG32_BIT(EIM_CS0GCR1,               0x63FD8000,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS0GCR2,               0x63FD8004,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS0RCR1,               0x63FD8008,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS0RCR2,               0x63FD800C,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS0WCR1,               0x63FD8010,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS0WCR2,               0x63FD8014,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS1GCR1,               0x63FD8018,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS1GCR2,               0x63FD801C,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS1RCR1,               0x63FD8020,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS1RCR2,               0x63FD8024,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS1WCR1,               0x63FD8028,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS1WCR2,               0x63FD802C,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS2GCR1,               0x63FD8030,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS2GCR2,               0x63FD8034,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS2RCR1,               0x63FD8038,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS2RCR2,               0x63FD803C,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS2WCR1,               0x63FD8040,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS2WCR2,               0x63FD8044,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_CS3GCR1,               0x63FD8048,__READ_WRITE ,__eim_csgcr1_bits);
__IO_REG32_BIT(EIM_CS3GCR2,               0x63FD804C,__READ_WRITE ,__eim_csgcr2_bits);
__IO_REG32_BIT(EIM_CS3RCR1,               0x63FD8050,__READ_WRITE ,__eim_csrcr1_bits);
__IO_REG32_BIT(EIM_CS3RCR2,               0x63FD8054,__READ_WRITE ,__eim_csrcr2_bits);
__IO_REG32_BIT(EIM_CS3WCR1,               0x63FD8058,__READ_WRITE ,__eim_cswcr1_bits);
__IO_REG32_BIT(EIM_CS3WCR2,               0x63FD805C,__READ_WRITE ,__eim_cswcr2_bits);
__IO_REG32_BIT(EIM_WCR,                   0x63FD8090,__READ_WRITE ,__eim_wcr_bits);
__IO_REG32_BIT(EIM_WIAR,                  0x63FD8094,__READ_WRITE ,__eim_wiar_bits);
__IO_REG32(    EIM_EAR,                   0x63FD8098,__READ_WRITE );

/***************************************************************************
 **
 **  EPDC
 **
 ***************************************************************************/
__IO_REG32_BIT(EPDC_CTRL,                 0x41010000,__READ_WRITE ,__epdc_ctrl_bits);
__IO_REG32(    EPDC_CTRL_SET,             0x41010004,__WRITE      );
__IO_REG32(    EPDC_CTRL_CLR,             0x41010008,__WRITE      );
__IO_REG32(    EPDC_CTRL_TOG,             0x4101000C,__WRITE      );
__IO_REG32(    EPDC_WVADDR,               0x41010020,__READ_WRITE );
__IO_REG32(    EPDC_WB_ADDR,              0x41010030,__READ_WRITE );
__IO_REG32_BIT(EPDC_RES,                  0x41010040,__READ_WRITE ,__epdc_res_bits);
__IO_REG32_BIT(EPDC_FORMAT,               0x41010050,__READ_WRITE ,__epdc_format_bits);
__IO_REG32(    EPDC_FORMAT_SET,           0x41010054,__WRITE      );
__IO_REG32(    EPDC_FORMAT_CLR,           0x41010058,__WRITE      );
__IO_REG32(    EPDC_FORMAT_TOG,           0x4101005C,__WRITE      );
__IO_REG32_BIT(EPDC_FIFOCTRL,             0x410100A0,__READ_WRITE ,__epdc_fifoctrl_bits);
__IO_REG32(    EPDC_FIFOCTRL_SET,         0x410100A4,__WRITE      );
__IO_REG32(    EPDC_FIFOCTRL_CLR,         0x410100A8,__WRITE      );
__IO_REG32(    EPDC_FIFOCTRL_TOG,         0x410100AC,__WRITE      );
__IO_REG32(    EPDC_UPD_ADDR,             0x41010100,__READ_WRITE );
__IO_REG32_BIT(EPDC_UPD_CORD,             0x41010120,__READ_WRITE ,__epdc_upd_cord_bits);
__IO_REG32_BIT(EPDC_UPD_SIZE,             0x41010140,__READ_WRITE ,__epdc_upd_size_bits);
__IO_REG32_BIT(EPDC_UPD_CTRL,             0x41010160,__READ_WRITE ,__epdc_upd_ctrl_bits);
__IO_REG32(    EPDC_UPD_CTRL_SET,         0x41010164,__WRITE      );
__IO_REG32(    EPDC_UPD_CTRL_CLR,         0x41010168,__WRITE      );
__IO_REG32(    EPDC_UPD_CTRL_TOG,         0x4101016C,__WRITE      );
__IO_REG32_BIT(EPDC_UPD_FIXED,            0x41010180,__READ_WRITE ,__epdc_upd_fixed_bits);
__IO_REG32(    EPDC_UPD_FIXED_SET,        0x41010184,__WRITE      );
__IO_REG32(    EPDC_UPD_FIXED_CLR,        0x41010188,__WRITE      );
__IO_REG32(    EPDC_UPD_FIXED_TOG,        0x4101018C,__WRITE      );
__IO_REG32(    EPDC_TEMP,                 0x410101A0,__READ_WRITE );
__IO_REG32_BIT(EPDC_TCE_CTRL,             0x41010200,__READ_WRITE ,__epdc_tce_ctrl_bits);
__IO_REG32(    EPDC_TCE_CTRL_SET,         0x41010204,__WRITE      );
__IO_REG32(    EPDC_TCE_CTRL_CLR,         0x41010208,__WRITE      );
__IO_REG32(    EPDC_TCE_CTRL_TOG,         0x4101020C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_SDCFG,            0x41010220,__READ_WRITE ,__epdc_tce_sdcfg_bits);
__IO_REG32(    EPDC_TCE_SDCFG_SET,        0x41010224,__WRITE      );
__IO_REG32(    EPDC_TCE_SDCFG_CLR,        0x41010228,__WRITE      );
__IO_REG32(    EPDC_TCE_SDCFG_TOG,        0x4101022C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_GDCFG,            0x41010240,__READ_WRITE ,__epdc_tce_gdcfg_bits);
__IO_REG32(    EPDC_TCE_GDCFG_SET,        0x41010244,__WRITE      );
__IO_REG32(    EPDC_TCE_GDCFG_CLR,        0x41010248,__WRITE      );
__IO_REG32(    EPDC_TCE_GDCFG_TOG,        0x4101024C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_HSCAN1,           0x41010260,__READ_WRITE ,__epdc_tce_hscan1_bits);
__IO_REG32(    EPDC_TCE_HSCAN1_SET,       0x41010264,__WRITE      );
__IO_REG32(    EPDC_TCE_HSCAN1_CLR,       0x41010268,__WRITE      );
__IO_REG32(    EPDC_TCE_HSCAN1_TOG,       0x4101026C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_HSCAN2,           0x41010280,__READ_WRITE ,__epdc_tce_hscan2_bits);
__IO_REG32(    EPDC_TCE_HSCAN2_SET,       0x41010284,__WRITE      );
__IO_REG32(    EPDC_TCE_HSCAN2_CLR,       0x41010288,__WRITE      );
__IO_REG32(    EPDC_TCE_HSCAN2_TOG,       0x4101028C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_VSCAN,            0x410102A0,__READ_WRITE ,__epdc_tce_vscan_bits);
__IO_REG32(    EPDC_TCE_VSCAN_SET,        0x410102A4,__WRITE      );
__IO_REG32(    EPDC_TCE_VSCAN_CLR,        0x410102A8,__WRITE      );
__IO_REG32(    EPDC_TCE_VSCAN_TOG,        0x410102AC,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_OE,               0x410102C0,__READ_WRITE ,__epdc_tce_oe_bits);
__IO_REG32(    EPDC_TCE_OE_SET,           0x410102C4,__WRITE      );
__IO_REG32(    EPDC_TCE_OE_CLR,           0x410102C8,__WRITE      );
__IO_REG32(    EPDC_TCE_OE_TOG,           0x410102CC,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_POLARITY,         0x410102E0,__READ_WRITE ,__epdc_tce_polarity_bits);
__IO_REG32(    EPDC_TCE_POLARITY_SET,     0x410102E4,__WRITE      );
__IO_REG32(    EPDC_TCE_POLARITY_CLR,     0x410102E8,__WRITE      );
__IO_REG32(    EPDC_TCE_POLARITY_TOG,     0x410102EC,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_TIMING1,          0x41010300,__READ_WRITE ,__epdc_tce_timing1_bits);
__IO_REG32(    EPDC_TCE_TIMING1_SET,      0x41010304,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING1_CLR,      0x41010308,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING1_TOG,      0x4101030C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_TIMING2,          0x41010310,__READ_WRITE ,__epdc_tce_timing2_bits);
__IO_REG32(    EPDC_TCE_TIMING2_SET,      0x41010314,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING2_CLR,      0x41010318,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING2_TOG,      0x4101031C,__WRITE      );
__IO_REG32_BIT(EPDC_TCE_TIMING3,          0x41010320,__READ_WRITE ,__epdc_tce_timing3_bits);
__IO_REG32(    EPDC_TCE_TIMING3_SET,      0x41010324,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING3_CLR,      0x41010328,__WRITE      );
__IO_REG32(    EPDC_TCE_TIMING3_TOG,      0x4101032C,__WRITE      );
__IO_REG32_BIT(EPDC_IRQ_MASK,             0x41010400,__READ_WRITE ,__epdc_irq_mask_bits);
__IO_REG32(    EPDC_IRQ_MASK_SET,         0x41010404,__WRITE      );
__IO_REG32(    EPDC_IRQ_MASK_CLR,         0x41010408,__WRITE      );
__IO_REG32(    EPDC_IRQ_MASK_TOG,         0x4101040C,__WRITE      );
__IO_REG32_BIT(EPDC_IRQ,                  0x41010420,__READ_WRITE ,__epdc_irq_bits);
__IO_REG32(    EPDC_IRQ_SET,              0x41010424,__WRITE      );
__IO_REG32(    EPDC_IRQ_CLR,              0x41010428,__WRITE      );
__IO_REG32(    EPDC_IRQ_TOG,              0x4101042C,__WRITE      );
__IO_REG32_BIT(EPDC_STATUS_LUTS,          0x41010440,__READ       ,__epdc_status_luts_bits);
__IO_REG32_BIT(EPDC_STATUS_NEXTLUT,       0x41010460,__READ       ,__epdc_status_nextlut_bits);
__IO_REG32_BIT(EPDC_STATUS_COL,           0x41010480,__READ       ,__epdc_status_col_bits);
__IO_REG32_BIT(EPDC_STATUS,               0x410104A0,__READ       ,__epdc_status_bits);
__IO_REG32_BIT(EPDC_DEBUG,                0x41010500,__READ_WRITE ,__epdc_debug_bits);
__IO_REG32(    EPDC_DEBUG_SET,            0x41010504,__WRITE      );
__IO_REG32(    EPDC_DEBUG_CLR,            0x41010508,__WRITE      );
__IO_REG32(    EPDC_DEBUG_TOG,            0x4101050C,__WRITE      );
__IO_REG32_BIT(EPDC_DEBUG_LUT0,           0x41010540,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT1,           0x41010550,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT2,           0x41010560,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT3,           0x41010570,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT4,           0x41010580,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT5,           0x41010590,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT6,           0x410105A0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT7,           0x410105B0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT8,           0x410105C0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT9,           0x410105D0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT10,          0x410105E0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT11,          0x410106F0,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT12,          0x41010600,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT13,          0x41010610,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT14,          0x41010620,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_DEBUG_LUT15,          0x41010630,__READ       ,__epdc_debug_lut_bits);
__IO_REG32_BIT(EPDC_GPIO,                 0x41010700,__READ_WRITE ,__epdc_gpio_bits);
__IO_REG32(    EPDC_GPIO_SET,             0x41010704,__WRITE      );
__IO_REG32(    EPDC_GPIO_CLR,             0x41010708,__WRITE      );
__IO_REG32(    EPDC_GPIO_TOG,             0x4101070C,__WRITE      );
__IO_REG32_BIT(EPDC_VERSION,              0x410107F0,__READ       ,__epdc_version_bits);

/***************************************************************************
 **
 **  EPIT1
 **
 ***************************************************************************/
__IO_REG32_BIT(EPIT_EPITCR,               0x53FAC000,__READ_WRITE ,__epit_epitcr_bits);
__IO_REG32_BIT(EPIT_EPITSR,               0x53FAC004,__READ_WRITE ,__epit_epitsr_bits);
__IO_REG32(    EPIT_EPITLR,               0x53FAC008,__READ_WRITE );
__IO_REG32(    EPIT_EPITCMPR,             0x53FAC00C,__READ_WRITE );
__IO_REG32(    EPIT_EPITCNR,              0x53FAC010,__READ       );

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
__IO_REG32_BIT(FEC_OPDR,                  0x63FEC0EC,__READ_WRITE,__fec_opdr_bits);
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
 ** GPC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPC_CNTR,                  0x53FD8000,__READ_WRITE ,__gpc_cntr_bits);
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
 **  GPMI
 **
 ***************************************************************************/
__IO_REG32_BIT(GPMI_CTRL0,                0x41006000,__READ_WRITE,__gpmi_ctrl0_bits);
__IO_REG32(    GPMI_CTRL0_SET,            0x41006004,__WRITE     );
__IO_REG32(    GPMI_CTRL0_CLR,            0x41006008,__WRITE     );
__IO_REG32(    GPMI_CTRL0_TOG,            0x4100600C,__WRITE     );
__IO_REG32_BIT(GPMI_COMPARE,              0x41006010,__READ_WRITE,__gpmi_compare_bits);
__IO_REG32_BIT(GPMI_ECCCTRL,              0x41006020,__READ_WRITE,__gpmi_eccctrl_bits);
__IO_REG32(    GPMI_ECCCTRL_SET,          0x41006024,__WRITE     );
__IO_REG32(    GPMI_ECCCTRL_CLR,          0x41006028,__WRITE     );
__IO_REG32(    GPMI_ECCCTRL_TOG,          0x4100602C,__WRITE     );
__IO_REG32_BIT(GPMI_ECCCOUNT,             0x41006030,__READ_WRITE,__gpmi_ecccount_bits);
__IO_REG32(    GPMI_PAYLOAD,              0x41006040,__READ_WRITE);
__IO_REG32(    GPMI_AUXILIARY,            0x41006050,__READ_WRITE);
__IO_REG32_BIT(GPMI_CTRL1,                0x41006060,__READ_WRITE,__gpmi_ctrl1_bits);
__IO_REG32(    GPMI_CTRL1_SET,            0x41006064,__WRITE     );
__IO_REG32(    GPMI_CTRL1_CLR,            0x41006068,__WRITE     );
__IO_REG32(    GPMI_CTRL1_TOG,            0x4100606C,__WRITE     );
__IO_REG32_BIT(GPMI_TIMING0,              0x41006070,__READ_WRITE,__gpmi_timing0_bits);
__IO_REG32_BIT(GPMI_TIMING1,              0x41006080,__READ_WRITE,__gpmi_timing1_bits);
__IO_REG32_BIT(GPMI_TIMING2,              0x41006090,__READ_WRITE,__gpmi_timing2_bits);
__IO_REG32(    GPMI_DATA,                 0x410060A0,__READ_WRITE);
__IO_REG32_BIT(GPMI_STAT,                 0x410060B0,__READ      ,__gpmi_stat_bits);
__IO_REG32_BIT(GPMI_DEBUG,                0x410060C0,__READ      ,__gpmi_debug_bits);
__IO_REG32_BIT(GPMI_VERSION,              0x410060D0,__READ      ,__gpmi_version_bits);
__IO_REG32_BIT(GPMI_DEBUG2,               0x410060E0,__READ_WRITE,__gpmi_debug2_bits);
__IO_REG32_BIT(GPMI_DEBUG3,               0x410060F0,__READ      ,__gpmi_debug3_bits);
__IO_REG32_BIT(GPMI_READ_DDR_DLL_CTRL,    0x41006100,__READ_WRITE,__gpmi_read_ddr_dll_ctrl_bits);
__IO_REG32_BIT(GPMI_WRITE_DDR_DLL_CTRL,   0x41006110,__READ_WRITE,__gpmi_write_ddr_dll_ctrl_bits);
__IO_REG32_BIT(GPMI_READ_DDR_DLL_STS,     0x41006120,__READ      ,__gpmi_read_ddr_dll_sts_bits);
__IO_REG32_BIT(GPMI_WRITE_DDR_DLL_STS,    0x41006130,__READ      ,__gpmi_write_ddr_dll_sts_bits);

/***************************************************************************
 **
 **  GPT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPT_CR,                    0x53FA0000,__READ_WRITE ,__gpt_cr_bits);
__IO_REG32_BIT(GPT_PR,                    0x53FA0004,__READ_WRITE ,__gpt_pr_bits);
__IO_REG32_BIT(GPT_SR,                    0x53FA0008,__READ_WRITE ,__gpt_sr_bits);
__IO_REG32_BIT(GPT_IR,                    0x53FA000C,__READ_WRITE ,__gpt_ir_bits);
__IO_REG32(    GPT_OCR1,                  0x53FA0010,__READ_WRITE );
__IO_REG32(    GPT_OCR2,                  0x53FA0014,__READ_WRITE );
__IO_REG32(    GPT_OCR3,                  0x53FA0018,__READ_WRITE );
__IO_REG32(    GPT_ICR1,                  0x53FA001C,__READ       );
__IO_REG32(    GPT_ICR2,                  0x53FA0020,__READ       );
__IO_REG32(    GPT_CNT,                   0x53FA0024,__READ       );

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C1_IADR,                 0x63FC8000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(I2C1_IFDR,                 0x63FC8004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2C1_I2CR,                 0x63FC8008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2C1_I2SR,                 0x63FC800C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2C1_I2DR,                 0x63FC8010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C2_IADR,                 0x63FC4000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(I2C2_IFDR,                 0x63FC4004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2C2_I2CR,                 0x63FC4008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2C2_I2SR,                 0x63FC400C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2C2_I2DR,                 0x63FC4010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  I2C3
 **
 ***************************************************************************/
__IO_REG16_BIT(I2C3_IADR,                 0x53FEC000,__READ_WRITE ,__iadr_bits);
__IO_REG16_BIT(I2C3_IFDR,                 0x53FEC004,__READ_WRITE ,__ifdr_bits);
__IO_REG16_BIT(I2C3_I2CR,                 0x53FEC008,__READ_WRITE ,__i2cr_bits);
__IO_REG16_BIT(I2C3_I2SR,                 0x53FEC00C,__READ_WRITE ,__i2sr_bits);
__IO_REG16_BIT(I2C3_I2DR,                 0x53FEC010,__READ_WRITE ,__i2dr_bits);

/***************************************************************************
 **
 **  IOMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(IOMUXC_GPR0,                       0x53FA8000,__READ_WRITE ,__iomuxc_gpr0_bits);
__IO_REG32_BIT(IOMUXC_GPR1,                       0x53FA8004,__READ_WRITE ,__iomuxc_gpr1_bits);
__IO_REG32_BIT(IOMUXC_GPR2,                       0x53FA8008,__READ_WRITE ,__iomuxc_gpr2_bits);
__IO_REG32_BIT(IOMUXC_OBSMUX0,                    0x53FA800C,__READ_WRITE ,__iomuxc_obsmux0_bits);
__IO_REG32_BIT(IOMUXC_OBSMUX1,                    0x53FA8010,__READ_WRITE ,__iomuxc_obsmux0_bits);
__IO_REG32_BIT(IOMUXC_OBSMUX2,                    0x53FA8014,__READ_WRITE ,__iomuxc_obsmux2_bits);
__IO_REG32_BIT(IOMUXC_OBSMUX3,                    0x53FA8018,__READ_WRITE ,__iomuxc_obsmux2_bits);
__IO_REG32_BIT(IOMUXC_OBSMUX4,                    0x53FA801C,__READ_WRITE ,__iomuxc_obsmux2_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKC0,                 0x53FA8020,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKR0,                 0x53FA8024,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKC1,                 0x53FA8028,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKR1,                 0x53FA802C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKC2,                 0x53FA8030,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKR2,                 0x53FA8034,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKC3,                 0x53FA8038,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PKR3,                 0x53FA803C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C1_SCL,            0x53FA8040,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C1_SDA,            0x53FA8044,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C2_SCL,            0x53FA8048,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C2_SDA,            0x53FA804C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C3_SCL,            0x53FA8050,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PI2C3_SDA,            0x53FA8054,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PPWM1,                0x53FA8058,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PPWM2,                0x53FA805C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_POWIRE,               0x53FA8060,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPITO,               0x53FA8064,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PWDOG,                0x53FA8068,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_TXFS,            0x53FA806C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_TXC,             0x53FA8070,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_TXD,             0x53FA8074,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_RXD,             0x53FA8078,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_RXF,             0x53FA807C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSSI_RXC,             0x53FA8080,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART1_TXD,           0x53FA8084,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART1_RXD,           0x53FA8088,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART1_CTS,           0x53FA808C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART1_RTS,           0x53FA8090,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART2_TXD,           0x53FA8094,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART2_RXD,           0x53FA8098,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART2_CTS,           0x53FA809C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART2_RTS,           0x53FA80A0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART3_TXD,           0x53FA80A4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART3_RXD,           0x53FA80A8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART4_TXD,           0x53FA80AC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PUART4_RXD,           0x53FA80B0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PCSPI_SCLK,           0x53FA80B4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PCSPI_MOSI,           0x53FA80B8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PCSPI_MISO,           0x53FA80BC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PCSPI_SS0,            0x53FA80C0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI1_SCLK,         0x53FA80C4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI1_MOSI,         0x53FA80C8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI1_MISO,         0x53FA80CC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI1_SS0,          0x53FA80D0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI2_SCLK,         0x53FA80D4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI2_MOSI,         0x53FA80D8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI2_MISO,         0x53FA80DC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_EPCSPI2_SS0,          0x53FA80E0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_CLK,             0x53FA80E4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_CMD,             0x53FA80E8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_D0,              0x53FA80EC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_D1,              0x53FA80F0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_D2,              0x53FA80F4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD1_D3,              0x53FA80F8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_CLK,             0x53FA80FC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_CMD,             0x53FA8100,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D0,              0x53FA8104,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D1,              0x53FA8108,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D2,              0x53FA810C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D3,              0x53FA8110,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D4,              0x53FA8114,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D5,              0x53FA8118,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D6,              0x53FA811C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_D7,              0x53FA8120,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_WP,              0x53FA8124,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD2_CD,              0x53FA8128,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D0,             0x53FA812C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D1,             0x53FA8130,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D2,             0x53FA8134,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D3,             0x53FA8138,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D4,             0x53FA813C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D5,             0x53FA8140,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D6,             0x53FA8144,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D7,             0x53FA8148,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_WR,             0x53FA814C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_RD,             0x53FA8150,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_RS,             0x53FA8154,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_CS,             0x53FA8158,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_BUSY,           0x53FA815C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_RESET,          0x53FA8160,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_CMD,             0x53FA8164,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_CLK,             0x53FA8168,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D0,              0x53FA816C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D1,              0x53FA8170,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D2,              0x53FA8174,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D3,              0x53FA8178,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D4,              0x53FA817C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D5,              0x53FA8180,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D6,              0x53FA8184,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_D7,              0x53FA8188,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PSD3_WP,              0x53FA818C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D8,             0x53FA8190,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D9,             0x53FA8194,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D10,            0x53FA8198,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D11,            0x53FA819C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D12,            0x53FA81A0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D13,            0x53FA81A4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D14,            0x53FA81A8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PDISP_D15,            0x53FA81AC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D0,             0x53FA81B0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D1,             0x53FA81B4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D2,             0x53FA81B8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D3,             0x53FA81BC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D4,             0x53FA81C0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D5,             0x53FA81C4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D6,             0x53FA81C8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D7,             0x53FA81CC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D8,             0x53FA81D0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D9,             0x53FA81D4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D10,            0x53FA81D8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D11,            0x53FA81DC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D12,            0x53FA81E0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D13,            0x53FA81E4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D14,            0x53FA81E8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_D15,            0x53FA81EC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_GDCLK,          0x53FA81F0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_GDSP,           0x53FA81F4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_GDOE,           0x53FA81F8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_GDRL,           0x53FA81FC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCLK,          0x53FA8200,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDOEZ,          0x53FA8204,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDOED,          0x53FA8208,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_OMUXC_SMUXC_PEPDC_SDOE,     0x53FA820C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDLE,           0x53FA8210,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCLKN,         0x53FA8214,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDSHR,          0x53FA8218,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_PWRCOM,         0x53FA821C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_PWRSTAT,        0x53FA8220,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_PWRCTRL0,       0x53FA8224,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_OMUXC_SMUXC_PEPDC_PWRCTRL1, 0x53FA8228,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_PWRCTRL2,       0x53FA822C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_PWRCTRL3,       0x53FA8230,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_VCOM0,          0x53FA8234,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_VCOM1,          0x53FA8238,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_BDR0,           0x53FA823C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_BDR1,           0x53FA8240,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE0,          0x53FA8244,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE1,          0x53FA8248,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE2,          0x53FA824C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE3,          0x53FA8250,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE4,          0x53FA8254,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEPDC_SDCE5,          0x53FA8258,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA0,             0x53FA825C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA1,             0x53FA8260,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA2,             0x53FA8264,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA3,             0x53FA8268,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA4,             0x53FA826C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA5,             0x53FA8270,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA6,             0x53FA8274,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA7,             0x53FA8278,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA8,             0x53FA827C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA9,             0x53FA8280,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA10,            0x53FA8284,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA11,            0x53FA8288,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA12,            0x53FA828C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA13,            0x53FA8290,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA14,            0x53FA8294,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_DA15,            0x53FA8298,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_CS2,             0x53FA829C,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_CS1,             0x53FA82A0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_CS0,             0x53FA82A4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_EB0,             0x53FA82A8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_EB1,             0x53FA82AC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_WAIT,            0x53FA82B0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_BCLK,            0x53FA82B4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_RDY,             0x53FA82B8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_OE,              0x53FA82BC,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_RW,              0x53FA82C0,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_LBA,             0x53FA82C4,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SMUXC_PEIM_CRE,             0x53FA82C8,__READ_WRITE ,__iomuxc_smuxc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKC0,                 0x53FA82CC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKR0,                 0x53FA82D0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKC1,                 0x53FA82D4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKR1,                 0x53FA82D8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKC2,                 0x53FA82DC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKR2,                 0x53FA82E0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKC3,                 0x53FA82E4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PKR3,                 0x53FA82E8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C1_SCL,            0x53FA82EC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C1_SDA,            0x53FA82F0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C2_SCL,            0x53FA82F4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C2_SDA,            0x53FA82F8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C3_SCL,            0x53FA82FC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PI2C3_SDA,            0x53FA8300,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PPWM1,                0x53FA8304,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PPWM2,                0x53FA8308,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_POWIRE,               0x53FA830C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPITO,               0x53FA8310,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PWDOG,                0x53FA8314,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_TXFS,            0x53FA8318,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_TXC,             0x53FA831C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_TXD,             0x53FA8320,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_RXD,             0x53FA8324,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_RXF,             0x53FA8328,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSSI_RXC,             0x53FA832C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART1_TXD,           0x53FA8330,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART1_RXD,           0x53FA8334,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART1_CTS,           0x53FA8338,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART1_RTS,           0x53FA833C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART2_TXD,           0x53FA8340,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART2_RXD,           0x53FA8344,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART2_CTS,           0x53FA8348,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART2_RTS,           0x53FA834C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART3_TXD,           0x53FA8350,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART3_RXD,           0x53FA8354,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART4_TXD,           0x53FA8358,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PUART4_RXD,           0x53FA835C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PCSPI_SCLK,           0x53FA8360,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PCSPI_MOSI,           0x53FA8364,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PCSPI_MISO,           0x53FA8368,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PCSPI_SS0,            0x53FA836C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI1_SCLK,         0x53FA8370,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI1_MOSI,         0x53FA8374,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI1_MISO,         0x53FA8378,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI1_SS0,          0x53FA837C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI2_SCLK,         0x53FA8380,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI2_MOSI,         0x53FA8384,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI2_MISO,         0x53FA8388,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PECSPI2_SS0,          0x53FA838C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_CLK,             0x53FA8390,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_CMD,             0x53FA8394,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_D0,              0x53FA8398,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_D1,              0x53FA839C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_D2,              0x53FA83A0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD1_D3,              0x53FA83A4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_CLK,             0x53FA83A8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_CMD,             0x53FA83AC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D0,              0x53FA83B0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D1,              0x53FA83B4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D2,              0x53FA83B8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D3,              0x53FA83BC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D4,              0x53FA83C0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D5,              0x53FA83C4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D6,              0x53FA83C8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_D7,              0x53FA83CC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_WP,              0x53FA83D0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD2_CD,              0x53FA83D4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PPMIC_ON_REQ,         0x53FA83D8,__READ_WRITE ,__iomuxc_spadc_ppmic_on_req_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PPMIC_STBY_REQ,       0x53FA83DC,__READ_WRITE ,__iomuxc_spadc_ppmic_stby_req_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PPOR_B,               0x53FA83E0,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PBOOT_MODE1,          0x53FA83E4,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PRESET_IN_B,          0x53FA83E8,__READ_WRITE ,__iomuxc_spadc_preset_in_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PBOOT_MODE0,          0x53FA83EC,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PTEST_MODE,           0x53FA83F0,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_TMS,            0x53FA83F4,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_MOD,            0x53FA83F8,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_TRSTB,          0x53FA83FC,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_TDI,            0x53FA8400,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_TCK,            0x53FA8404,__READ_WRITE ,__iomuxc_spadc_ppor_b_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PJTAG_TDO,            0x53FA8408,__READ_WRITE ,__iomuxc_spadc_pjtag_tdo_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D0,             0x53FA840C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D1,             0x53FA8410,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D2,             0x53FA8414,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D3,             0x53FA8418,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D4,             0x53FA841C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D5,             0x53FA8420,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D6,             0x53FA8424,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D7,             0x53FA8428,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_WR,             0x53FA842C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_RD,             0x53FA8430,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_RS,             0x53FA8434,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_CS,             0x53FA8438,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_BUSY,           0x53FA843C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_RESET,          0x53FA8440,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_CMD,             0x53FA8444,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_CLK,             0x53FA8448,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D0,              0x53FA844C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D1,              0x53FA8450,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D2,              0x53FA8454,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D3,              0x53FA8458,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D4,              0x53FA845C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D5,              0x53FA8460,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D6,              0x53FA8464,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_D7,              0x53FA8468,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PSD3_WP,              0x53FA846C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D8,             0x53FA8470,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D9,             0x53FA8474,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D10,            0x53FA8478,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D11,            0x53FA847C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D12,            0x53FA8480,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D13,            0x53FA8484,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D14,            0x53FA8488,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDISP_D15,            0x53FA848C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_OPEN,           0x53FA8490,__READ_WRITE ,__iomuxc_spadc_pdram_open_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_OPENFB,         0x53FA8494,__READ_WRITE ,__iomuxc_spadc_pdram_open_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDCLK_1,        0x53FA8498,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDCLK_0,        0x53FA849C,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDCKE,          0x53FA84A0,__READ_WRITE ,__iomuxc_spadc_pdram_sdcke_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDODT0,         0x53FA84A4,__READ_WRITE ,__iomuxc_spadc_pdram_sdodt_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D16,            0x53FA84A8,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D17,            0x53FA84AC,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D18,            0x53FA84B0,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D19,            0x53FA84B4,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D20,            0x53FA84B8,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D21,            0x53FA84BC,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D22,            0x53FA84C0,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D23,            0x53FA84C4,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_DQM2,           0x53FA84C8,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDQS2,          0x53FA84CC,__READ_WRITE ,__iomuxc_spadc_pdram_sdqs_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D0,             0x53FA84D0,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D1,             0x53FA84D4,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D2,             0x53FA84D8,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D3,             0x53FA84DC,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D4,             0x53FA84E0,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D5,             0x53FA84E4,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D6,             0x53FA84E8,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D7,             0x53FA84EC,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_DQM0,           0x53FA84F0,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDQS0,          0x53FA84F4,__READ_WRITE ,__iomuxc_spadc_pdram_sdqs_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDODT1,         0x53FA84F8,__READ_WRITE ,__iomuxc_spadc_pdram_sdodt_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDQS1,          0x53FA84FC,__READ_WRITE ,__iomuxc_spadc_pdram_sdqs_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_DQM1,           0x53FA8500,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D8,             0x53FA8504,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D9,             0x53FA8508,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D10,            0x53FA850C,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D11,            0x53FA8510,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D12,            0x53FA8514,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D13,            0x53FA8518,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D14,            0x53FA851C,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D15,            0x53FA8520,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_SDQS3,          0x53FA8524,__READ_WRITE ,__iomuxc_spadc_pdram_sdqs_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_DQM3,           0x53FA8528,__READ_WRITE ,__iomuxc_spadc_pdram_sdclk_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D24,            0x53FA852C,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D25,            0x53FA8530,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D26,            0x53FA8534,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D27,            0x53FA8538,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D28,            0x53FA853C,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D29,            0x53FA8540,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D30,            0x53FA8544,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PDRAM_D31,            0x53FA8548,__READ_WRITE ,__iomuxc_spadc_pdram_d_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D0,             0x53FA854C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D1,             0x53FA8550,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D2,             0x53FA8554,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D3,             0x53FA8558,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D4,             0x53FA855C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D5,             0x53FA8560,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D6,             0x53FA8564,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D7,             0x53FA8568,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D8,             0x53FA856C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D9,             0x53FA8570,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D10,            0x53FA8574,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D11,            0x53FA8578,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D12,            0x53FA857C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D13,            0x53FA8580,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D14,            0x53FA8584,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_D15,            0x53FA8588,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_GDCLK,          0x53FA858C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_GDSP,           0x53FA8590,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_GDOE,           0x53FA8594,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_GDRL,           0x53FA8598,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCLK,          0x53FA859C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDOEZ,          0x53FA85A0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDOED,          0x53FA85A4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDOE,           0x53FA85A8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDLE,           0x53FA85AC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCLKN,         0x53FA85B0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDSHR,          0x53FA85B4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRCOM,         0x53FA85B8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRSTAT,        0x53FA85BC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRCTRL0,       0x53FA85C0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRCTRL1,       0x53FA85C4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRCTRL2,       0x53FA85C8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_PWRCTRL3,       0x53FA85CC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_VCOM0,          0x53FA85D0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_VCOM1,          0x53FA85D4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_BDR0,           0x53FA85D8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_BDR1,           0x53FA85DC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE0,          0x53FA85E0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE1,          0x53FA85E4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE2,          0x53FA85E8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE3,          0x53FA85EC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE4,          0x53FA85F0,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEPDC_SDCE5,          0x53FA85F4,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA0,             0x53FA85F8,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA1,             0x53FA85FC,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA2,             0x53FA8600,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA3,             0x53FA8604,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA4,             0x53FA8608,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA5,             0x53FA860C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA6,             0x53FA8610,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA7,             0x53FA8614,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA8,             0x53FA8618,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA9,             0x53FA861C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA10,            0x53FA8620,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA11,            0x53FA8624,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA12,            0x53FA8628,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA13,            0x53FA862C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA14,            0x53FA8630,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_DA15,            0x53FA8634,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_CS2,             0x53FA8638,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_CS1,             0x53FA863C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_CS0,             0x53FA8640,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_EB0,             0x53FA8644,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_EB1,             0x53FA8648,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_WAIT,            0x53FA864C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_BCLK,            0x53FA8650,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_RDY,             0x53FA8654,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_OE,              0x53FA8658,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_RW,              0x53FA865C,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_LBA,             0x53FA8660,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPADC_PEIM_CRE,             0x53FA8664,__READ_WRITE ,__iomuxc_spadc_pad_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GADDDS,                0x53FA8668,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDRMODE_CTL,          0x53FA866C,__READ_WRITE ,__iomuxc_spad_gddrmode_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDRPKE,               0x53FA8670,__READ_WRITE ,__iomuxc_spad_gddrpke_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GEIM,                  0x53FA8674,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GEPDC,                 0x53FA8678,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GUART,                 0x53FA867C,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDRPK,                0x53FA8680,__READ_WRITE ,__iomuxc_spad_gddrpk_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDRHYS,               0x53FA8684,__READ_WRITE ,__iomuxc_spad_gddrhys_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GKEYPAD,               0x53FA8688,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDRMODE,              0x53FA868C,__READ_WRITE ,__iomuxc_spad_gddrmode_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GSSI,                  0x53FA8690,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GSD1,                  0x53FA8694,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GB0DS,                 0x53FA8698,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GSD2,                  0x53FA869C,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GB1DS,                 0x53FA86A0,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GCTLDS,                0x53FA86A4,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GB2DS,                 0x53FA86A8,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GDDR_TYPE,             0x53FA86AC,__READ_WRITE ,__iomuxc_spad_gddr_type_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GLCD,                  0x53FA86B0,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GB3DS,                 0x53FA86B4,__READ_WRITE ,__iomuxc_spad_gaddds_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GMISC,                 0x53FA86B8,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GSPI,                  0x53FA86BC,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_SPAD_GNANDF,                0x53FA86C0,__READ_WRITE ,__iomuxc_spad_geim_bits);
__IO_REG32_BIT(IOMUXC_API_DA_AMX_SI,              0x53FA86C4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_API_DB_AMX_SI,              0x53FA86C8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_API_RXCLK_AMX_SI,           0x53FA86CC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_API_RXFS_AMX_SI,            0x53FA86D0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_API_TXCLK_AMX_SI,           0x53FA86D4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_API_TXFS_AMX_SI,            0x53FA86D8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL1_BYPASS_CLK_SI,     0x53FA86DC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL2_BYPASS_CLK_SI,     0x53FA86E0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CCM_PLL3_BYPASS_CLK_SI,     0x53FA86E4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CSPI_DRDY_SI,               0x53FA86E8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CSPI_SS1_SI,                0x53FA86EC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CSPI_SS2_SI,                0x53FA86F0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_CSPI_SS3_SI,                0x53FA86F4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_BUSY_SI,            0x53FA86F8,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R0_SI,              0x53FA86FC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R1_SI,              0x53FA8700,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R2_SI,              0x53FA8704,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R3_SI,              0x53FA8708,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R4_SI,              0x53FA870C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R5_SI,              0x53FA8710,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R6_SI,              0x53FA8714,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R7_SI,              0x53FA8718,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R8_SI,              0x53FA871C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R9_SI,              0x53FA8720,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R10_SI,             0x53FA8724,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R11_SI,             0x53FA8728,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R12_SI,             0x53FA872C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R13_SI,             0x53FA8730,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R14_SI,             0x53FA8734,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIFL_R15_SI,             0x53FA8738,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ELCDIF_VSYNC_I_SI,          0x53FA873C,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_CDET_SI,             0x53FA8740,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC2_WP_ON_SI,            0x53FA8744,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_CCLK_IN_SI,          0x53FA8748,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_CMD_IN_SI,           0x53FA874C,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_DAT0_IN_SI,          0x53FA8750,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_XC_ESDHC4_1_IN_SI,          0x53FA8754,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_2_IN_SI,             0x53FA8758,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_3_IN_SI,             0x53FA875C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_4_IN_SI,             0x53FA8760,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_5_IN_SI,             0x53FA8764,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_6_IN_SI,             0x53FA8768,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_ESDHC4_7_IN_SI,             0x53FA876C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_COL_SI,                 0x53FA8770,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_MDI_SI,                 0x53FA8774,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_RD0_SI,                 0x53FA8778,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_RD1_SI,                 0x53FA877C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_RX_CLK_SI,              0x53FA8780,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_RX_DV_SI,               0x53FA8784,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_RX_ER_SI,               0x53FA8788,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_FEC_TX_CLK_SI,              0x53FA878C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_KPP_FC4_SI,                 0x53FA8790,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FC5_SI,                 0x53FA8794,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FC6_SI,                 0x53FA8798,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FC7_SI,                 0x53FA879C,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FR4_SI,                 0x53FA87A0,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FR5_SI,                 0x53FA87A4,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FR6_SI,                 0x53FA87A8,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_KPP_FR7_SI,                 0x53FA87AC,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_RAWNAND_U_GPMI_INPUT_GPMI_DQS_IN_SI, 0x53FA87B0,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_RAWNAND_U_GPMI_INPUT_GPMI_RDY0_SI,   0x53FA87B4,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_SDMA_EVENTS_14_SI,          0x53FA87B8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_SDMA_EVENTS_15_SI,          0x53FA87BC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U1U_RTS_B_SI,               0x53FA87C0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U1U_RXD_MUX_SI,             0x53FA87C4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U2U_RTS_B_SI,               0x53FA87C8,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_U2U_RXD_MUX_SI,             0x53FA87CC,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_U3U_RTS_B_SI,               0x53FA87D0,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_U3U_RXD_MUX_SI,             0x53FA87D4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U4U_RTS_B_SI,               0x53FA87D8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U4U_RXD_MUX_SI,             0x53FA87DC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U5U_RTS_B_SI,               0x53FA87E0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_U5U_RXD_MUX_SI,             0x53FA87E4,__READ_WRITE ,__iomuxc_si3_bits);
__IO_REG32_BIT(IOMUXC_USBOH1_OTG_OC_SI,           0x53FA87E8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD0_SI,              0x53FA87EC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD1_SI,              0x53FA87F0,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD2_SI,              0x53FA87F4,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD3_SI,              0x53FA87F8,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD4_SI,              0x53FA87FC,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD5_SI,              0x53FA8800,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD6_SI,              0x53FA8804,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD7_SI,              0x53FA8808,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD8_SI,              0x53FA880C,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD9_SI,              0x53FA8810,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD10_SI,             0x53FA8814,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD11_SI,             0x53FA8818,__READ_WRITE ,__iomuxc_si2_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD12_SI,             0x53FA881C,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD13_SI,             0x53FA8820,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD14_SI,             0x53FA8824,__READ_WRITE ,__iomuxc_si1_bits);
__IO_REG32_BIT(IOMUXC_WEIMV2_RD15_SI,             0x53FA8828,__READ_WRITE ,__iomuxc_si1_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPP_KPCR,                  0x53F94000,__READ_WRITE ,__kpp_kpcr_bits);
__IO_REG16_BIT(KPP_KPSR,                  0x53F94002,__READ_WRITE ,__kpp_kpsr_bits);
__IO_REG16_BIT(KPP_KDDR,                  0x53F94004,__READ_WRITE ,__kpp_kddr_bits);
__IO_REG16_BIT(KPP_KPDR,                  0x53F94006,__READ_WRITE ,__kpp_kpdr_bits);

/***************************************************************************
 **
 **  eLCDIF
 **
 ***************************************************************************/
__IO_REG32_BIT(LCDIF_CTRL,                0x4100A000,__READ_WRITE ,__lcdif_ctrl_bits);
__IO_REG32(    LCDIF_CTRL_SET,            0x4100A004,__WRITE     );
__IO_REG32(    LCDIF_CTRL_CLR,            0x4100A008,__WRITE     );
__IO_REG32(    LCDIF_CTRL_TOG,            0x4100A00C,__WRITE     );
__IO_REG32_BIT(LCDIF_CTRL1,               0x4100A010,__READ_WRITE ,__lcdif_ctrl1_bits);
__IO_REG32(    LCDIF_CTRL1_SET,           0x4100A014,__WRITE     );
__IO_REG32(    LCDIF_CTRL1_CLR,           0x4100A018,__WRITE     );
__IO_REG32(    LCDIF_CTRL1_TOG,           0x4100A01C,__WRITE     );
__IO_REG32_BIT(LCDIF_CTRL2,               0x4100A020,__READ_WRITE ,__lcdif_ctrl2_bits);
__IO_REG32(    LCDIF_CTRL2_SET,           0x4100A024,__WRITE     );
__IO_REG32(    LCDIF_CTRL2_CLR,           0x4100A028,__WRITE     );
__IO_REG32(    LCDIF_CTRL2_TOG,           0x4100A02C,__WRITE     );
__IO_REG32_BIT(LCDIF_TRANSFER_COUNT,      0x4100A030,__READ_WRITE ,__lcdif_transfer_count_bits);
__IO_REG32(    LCDIF_CUR_BUF,             0x4100A040,__READ_WRITE );
__IO_REG32(    LCDIF_NEXT_BUF,            0x4100A050,__READ_WRITE );
__IO_REG32_BIT(LCDIF_TIMING,              0x4100A060,__READ_WRITE ,__lcdif_timing_bits);
__IO_REG32_BIT(LCDIF_VDCTRL0,             0x4100A070,__READ_WRITE ,__lcdif_vdctrl0_bits);
__IO_REG32(    LCDIF_VDCTRL0_SET,         0x4100A074,__WRITE     );
__IO_REG32(    LCDIF_VDCTRL0_CLR,         0x4100A078,__WRITE     );
__IO_REG32(    LCDIF_VDCTRL0_TOG,         0x4100A07C,__WRITE     );
__IO_REG32_BIT(LCDIF_VDCTRL1,             0x4100A080,__READ_WRITE ,__lcdif_vdctrl1_bits);
__IO_REG32_BIT(LCDIF_VDCTRL2,             0x4100A090,__READ_WRITE ,__lcdif_vdctrl2_bits);
__IO_REG32_BIT(LCDIF_VDCTRL3,             0x4100A0A0,__READ_WRITE ,__lcdif_vdctrl3_bits);
__IO_REG32_BIT(LCDIF_VDCTRL4,             0x4100A0B0,__READ_WRITE ,__lcdif_vdctrl4_bits);
__IO_REG32_BIT(LCDIF_DVICTRL0,            0x4100A0C0,__READ_WRITE ,__lcdif_dvictrl0_bits);
__IO_REG32_BIT(LCDIF_DVICTRL1,            0x4100A0D0,__READ_WRITE ,__lcdif_dvictrl1_bits);
__IO_REG32_BIT(LCDIF_DVICTRL2,            0x4100A0E0,__READ_WRITE ,__lcdif_dvictrl2_bits);
__IO_REG32_BIT(LCDIF_DVICTRL3,            0x4100A0F0,__READ_WRITE ,__lcdif_dvictrl3_bits);
__IO_REG32_BIT(LCDIF_DVICTRL4,            0x4100A100,__READ_WRITE ,__lcdif_dvictrl4_bits);
__IO_REG32_BIT(LCDIF_CSC_COEFF0,          0x4100A110,__READ_WRITE ,__lcdif_csc_coeff0_bits);
__IO_REG32_BIT(LCDIF_CSC_COEFF1,          0x4100A120,__READ_WRITE ,__lcdif_csc_coeff1_bits);
__IO_REG32_BIT(LCDIF_CSC_COEFF2,          0x4100A130,__READ_WRITE ,__lcdif_csc_coeff2_bits);
__IO_REG32_BIT(LCDIF_CSC_COEFF3,          0x4100A140,__READ_WRITE ,__lcdif_csc_coeff3_bits);
__IO_REG32_BIT(LCDIF_CSC_COEFF4,          0x4100A150,__READ_WRITE ,__lcdif_csc_coeff4_bits);
__IO_REG32_BIT(LCDIF_CSC_OFFSET,          0x4100A160,__READ_WRITE ,__lcdif_csc_offset_bits);
__IO_REG32_BIT(LCDIF_CSC_LIMIT,           0x4100A170,__READ_WRITE ,__lcdif_csc_limit_bits);
__IO_REG32_BIT(LCDIF_DATA,                0x4100A180,__READ_WRITE ,__lcdif_data_bits);
__IO_REG32(    LCDIF_BM_ERROR_STAT,       0x4100A190,__READ_WRITE );
__IO_REG32(    LCDIF_CRC_STAT,            0x4100A1A0,__READ_WRITE );
__IO_REG32_BIT(LCDIF_STAT,                0x4100A1B0,__READ       ,__lcdif_stat_bits);
__IO_REG32_BIT(LCDIF_VERSION,             0x4100A1C0,__READ       ,__lcdif_version_bits);
__IO_REG32_BIT(LCDIF_DEBUG0,              0x4100A1D0,__READ       ,__lcdif_debug0_bits);
__IO_REG32_BIT(LCDIF_DEBUG1,              0x4100A1E0,__READ       ,__lcdif_debug1_bits);
__IO_REG32(    LCDIF_DEBUG2,              0x4100A1F0,__READ       );
__IO_REG32_BIT(LCDIF_THRES,               0x4100A200,__READ_WRITE ,__lcdif_thres_bits);

/***************************************************************************
 **
 **  OCOTP
 **
 ***************************************************************************/
__IO_REG32_BIT(OCOTP_CTRL,                0x41002000,__READ_WRITE ,__ocotp_ctrl_bits);
__IO_REG32(    OCOTP_CTRL_SET,            0x41002004,__WRITE      );
__IO_REG32(    OCOTP_CTRL_CLR,            0x41002008,__WRITE      );
__IO_REG32(    OCOTP_CTRL_TOG,            0x4100200C,__WRITE      );
__IO_REG32_BIT(OCOTP_TIMING,              0x41002010,__READ_WRITE ,__ocotp_timing_bits);
__IO_REG32(    OCOTP_DATA,                0x41002020,__READ_WRITE );
__IO_REG32_BIT(OCOTP_LOCK,                0x41002030,__READ       ,__ocotp_lock_bits);
__IO_REG32(    OCOTP_CFG0,                0x41002040,__READ_WRITE );
__IO_REG32(    OCOTP_CFG1,                0x41002050,__READ_WRITE );
__IO_REG32(    OCOTP_CFG2,                0x41002060,__READ_WRITE );
__IO_REG32(    OCOTP_CFG3,                0x41002070,__READ_WRITE );
__IO_REG32(    OCOTP_CFG4,                0x41002080,__READ_WRITE );
__IO_REG32(    OCOTP_CFG5,                0x41002090,__READ_WRITE );
__IO_REG32(    OCOTP_CFG6,                0x410020A0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM0,                0x410020B0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM1,                0x410020C0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM2,                0x410020D0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM3,                0x410020E0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM4,                0x410020F0,__READ_WRITE );
__IO_REG32(    OCOTP_MEM5,                0x41002100,__READ_WRITE );
__IO_REG32(    OCOTP_GP0,                 0x41002110,__READ_WRITE );
__IO_REG32(    OCOTP_GP1,                 0x41002120,__READ_WRITE );
__IO_REG32(    OCOTP_SCC0,                0x41002130,__READ_WRITE );
__IO_REG32(    OCOTP_SCC1,                0x41002140,__READ_WRITE );
__IO_REG32(    OCOTP_SCC2,                0x41002150,__READ_WRITE );
__IO_REG32(    OCOTP_SCC3,                0x41002160,__READ_WRITE );
__IO_REG32(    OCOTP_SCC4,                0x41002170,__READ_WRITE );
__IO_REG32(    OCOTP_SCC5,                0x41002180,__READ_WRITE );
__IO_REG32(    OCOTP_SCC6,                0x41002190,__READ_WRITE );
__IO_REG32(    OCOTP_SCC7,                0x410021A0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK0,                0x410021B0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK1,                0x410021C0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK2,                0x410021D0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK3,                0x410021E0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK4,                0x410021F0,__READ_WRITE );
__IO_REG32(    OCOTP_SRK5,                0x41002200,__READ_WRITE );
__IO_REG32(    OCOTP_SRK6,                0x41002210,__READ_WRITE );
__IO_REG32(    OCOTP_SRK7,                0x41002220,__READ_WRITE );
__IO_REG32(    OCOTP_SJC_RESP0,           0x41002230,__READ_WRITE );
__IO_REG32(    OCOTP_SJC_RESP1,           0x41002240,__READ_WRITE );
__IO_REG32(    OCOTP_MAC0,                0x41002250,__READ_WRITE );
__IO_REG32(    OCOTP_MAC1,                0x41002260,__READ_WRITE );
__IO_REG32(    OCOTP_HWCAP0,              0x41002270,__READ_WRITE );
__IO_REG32(    OCOTP_HWCAP1,              0x41002280,__READ_WRITE );
__IO_REG32(    OCOTP_HWCAP2,              0x41002290,__READ_WRITE );
__IO_REG32(    OCOTP_SWCAP,               0x410022A0,__READ_WRITE );
__IO_REG32_BIT(OCOTP_SCS,                 0x410022B0,__READ_WRITE ,__ocotp_scs_bits);
__IO_REG32(    OCOTP_SCS_SET,             0x410022B4,__WRITE      );
__IO_REG32(    OCOTP_SCS_CLR,             0x410022B8,__WRITE      );
__IO_REG32(    OCOTP_SCS_TOG,             0x410022BC,__WRITE      );
__IO_REG32_BIT(OCOTP_VERSION,             0x410022C0,__READ       ,__ocotp_version_bits);

/***************************************************************************
 **
 **  OWIRE
 **
 ***************************************************************************/
__IO_REG16_BIT(OWIRE_CONTROL,             0x63FA4000,__READ_WRITE ,__owire_control_bits);
__IO_REG16_BIT(OWIRE_TIME_DIVIDER,        0x63FA4002,__READ_WRITE ,__owire_time_divider_bits);
__IO_REG16_BIT(OWIRE_RESET,               0x63FA4004,__READ_WRITE ,__owire_reset_bits);
__IO_REG16_BIT(OWIRE_COMMAND,             0x63FA4006,__READ_WRITE ,__owire_command_bits);
__IO_REG16_BIT(OWIRE_TX_RX,               0x63FA4008,__READ_WRITE ,__owire_tx_rx_bits);
__IO_REG16_BIT(OWIRE_INTERRUPT,           0x63FA400A,__READ       ,__owire_interrupt_bits);
__IO_REG16_BIT(OWIRE_INTERRUPT_EN,        0x63FA400C,__READ_WRITE ,__owire_interrupt_en_bits);

/***************************************************************************
 **
 **  PERFMON
 **
 ***************************************************************************/
__IO_REG32_BIT(PERFMON_CTRL,              0x41014000,__READ_WRITE ,__perfmon_ctrl_bits);
__IO_REG32(    PERFMON_CTRL_SET,          0x41014004,__WRITE      );
__IO_REG32(    PERFMON_CTRL_CLR,          0x41014008,__WRITE      );
__IO_REG32(    PERFMON_CTRL_TOG,          0x4101400C,__WRITE      );
__IO_REG32_BIT(PERFMON_MASTER_EN,         0x41014010,__READ_WRITE ,__perfmon_master_en_bits);
__IO_REG32(    PERFMON_TRAP_ADDR_LOW,     0x41014020,__READ_WRITE );
__IO_REG32(    PERFMON_TRAP_ADDR_HIGH,    0x41014030,__READ_WRITE );
__IO_REG32_BIT(PERFMON_LAT_THRESHOLD,     0x41014040,__READ_WRITE ,__perfmon_lat_threshold_bits);
__IO_REG32(    PERFMON_ACTIVE_CYCLE,      0x41014050,__READ       );
__IO_REG32(    PERFMON_TRANSFER_COUNT,    0x41014060,__READ       );
__IO_REG32(    PERFMON_TOTAL_LATENCY,     0x41014070,__READ       );
__IO_REG32(    PERFMON_DATA_COUNT,        0x41014080,__READ       );
__IO_REG32_BIT(PERFMON_MAX_LATENCY,       0x41014090,__READ       ,__perfmon_max_latency_bits);
__IO_REG32_BIT(PERFMON_DEBUG,             0x410140A0,__READ_WRITE ,__perfmon_debug_bits);
__IO_REG32_BIT(PERFMON_VERSION,           0x410140B0,__READ       ,__perfmon_version_bits);

/***************************************************************************
 **
 **  PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1_PWMCR,                0x53FB4000,__READ_WRITE ,__pwmcr_bits);
__IO_REG32_BIT(PWM1_PWMSR,                0x53FB4004,__READ_WRITE ,__pwmsr_bits);
__IO_REG32_BIT(PWM1_PWMIR,                0x53FB4008,__READ_WRITE ,__pwmir_bits);
__IO_REG32_BIT(PWM1_PWMSAR,               0x53FB400C,__READ_WRITE ,__pwmsar_bits);
__IO_REG32_BIT(PWM1_PWMPR,                0x53FB4010,__READ_WRITE ,__pwmpr_bits);
__IO_REG32_BIT(PWM1_PWMCNR,               0x53FB4014,__READ       ,__pwmcnr_bits);

/***************************************************************************
 **
 **  PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM2_PWMCR,                0x53FB8000,__READ_WRITE ,__pwmcr_bits);
__IO_REG32_BIT(PWM2_PWMSR,                0x53FB8004,__READ_WRITE ,__pwmsr_bits);
__IO_REG32_BIT(PWM2_PWMIR,                0x53FB8008,__READ_WRITE ,__pwmir_bits);
__IO_REG32_BIT(PWM2_PWMSAR,               0x53FB800C,__READ_WRITE ,__pwmsar_bits);
__IO_REG32_BIT(PWM2_PWMPR,                0x53FB8010,__READ_WRITE ,__pwmpr_bits);
__IO_REG32_BIT(PWM2_PWMCNR,               0x53FB8014,__READ       ,__pwmcnr_bits);

/***************************************************************************
 **
 **  ePXP
 **
 ***************************************************************************/
__IO_REG32_BIT(PXP_CTRL,                  0x4100C000,__READ_WRITE ,__pxp_ctrl_bits);
__IO_REG32(    PXP_CTRL_SET,              0x4100C004,__WRITE      );
__IO_REG32(    PXP_CTRL_CLR,              0x4100C008,__WRITE      );
__IO_REG32(    PXP_CTRL_TOG,              0x4100C00C,__WRITE      );
__IO_REG32_BIT(PXP_STAT,                  0x4100C010,__READ_WRITE ,__pxp_stat_bits);
__IO_REG32(    PXP_STAT_SET,              0x4100C014,__WRITE      );
__IO_REG32(    PXP_STAT_CLR,              0x4100C018,__WRITE      );
__IO_REG32(    PXP_STAT_TOG,              0x4100C01C,__WRITE      );
__IO_REG32(    PXP_OUTBUF,                0x4100C020,__READ_WRITE );
__IO_REG32(    PXP_OUTBUF2,               0x4100C030,__READ_WRITE );
__IO_REG32_BIT(PXP_OUTSIZE,               0x4100C040,__READ_WRITE ,__pxp_outsize_bits);
__IO_REG32(    PXP_S0BUF,                 0x4100C050,__READ_WRITE );
__IO_REG32(    PXP_S0UBUF,                0x4100C060,__READ_WRITE );
__IO_REG32(    PXP_S0VBUF,                0x4100C070,__READ_WRITE );
__IO_REG32_BIT(PXP_S0PARAM,               0x4100C080,__READ_WRITE ,__pxp_s0param_bits);
__IO_REG32(    PXP_S0BACKGROUND,          0x4100C090,__READ_WRITE );
__IO_REG32_BIT(PXP_S0CROP ,               0x4100C0A0,__READ_WRITE ,__pxp_s0crop_bits);
__IO_REG32_BIT(PXP_S0SCALE ,              0x4100C0B0,__READ_WRITE ,__pxp_s0scale_bits);
__IO_REG32_BIT(PXP_S0OFFSET ,             0x4100C0C0,__READ_WRITE ,__pxp_s0offset_bits);
__IO_REG32_BIT(PXP_CSCCOEFF0 ,            0x4100C0D0,__READ_WRITE ,__pxp_csccoeff0_bits);
__IO_REG32_BIT(PXP_CSCCOEFF1 ,            0x4100C0E0,__READ_WRITE ,__pxp_csccoeff1_bits);
__IO_REG32_BIT(PXP_CSCCOEFF2 ,            0x4100C0F0,__READ_WRITE ,__pxp_csccoeff2_bits);
__IO_REG32_BIT(PXP_NEXT,                  0x4100C100,__READ_WRITE ,__pxp_next_bits);
__IO_REG32(    PXP_NEXT_SET,              0x4100C104,__WRITE      );
__IO_REG32(    PXP_NEXT_CLR,              0x4100C108,__WRITE      );
__IO_REG32(    PXP_NEXT_TOG,              0x4100C10C,__WRITE      );
__IO_REG32_BIT(PXP_S0COLORKEYLOW,         0x4100C180,__READ_WRITE ,__pxp_s0colorkeylow_bits);
__IO_REG32_BIT(PXP_S0COLORKEYHIGH,        0x4100C190,__READ_WRITE ,__pxp_s0colorkeyhigh_bits);
__IO_REG32_BIT(PXP_OLCOLORKEYLOW,         0x4100C1A0,__READ_WRITE ,__pxp_olcolorkeylow_bits);
__IO_REG32_BIT(PXP_OLCOLORKEYHIGH,        0x4100C1B0,__READ_WRITE ,__pxp_olcolorkeyhigh_bits);
__IO_REG32(    PXP_DEBUG,                 0x4100C1E0,__READ       );
__IO_REG32(    PXP_OL0,                   0x4100C200,__READ_WRITE );
__IO_REG32_BIT(PXP_OL0SIZE,               0x4100C210,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL0PARAM,              0x4100C220,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL1,                   0x4100C240,__READ_WRITE );
__IO_REG32_BIT(PXP_OL1SIZE,               0x4100C250,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL1PARAM,              0x4100C260,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL2,                   0x4100C280,__READ_WRITE );
__IO_REG32_BIT(PXP_OL2SIZE,               0x4100C290,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL2PARAM,              0x4100C2A0,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL3,                   0x4100C2C0,__READ_WRITE );
__IO_REG32_BIT(PXP_OL3SIZE,               0x4100C2D0,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL3PARAM,              0x4100C2E0,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL4,                   0x4100C300,__READ_WRITE );
__IO_REG32_BIT(PXP_OL4SIZE,               0x4100C310,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL4PARAM,              0x4100C320,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL5,                   0x4100C340,__READ_WRITE );
__IO_REG32_BIT(PXP_OL5SIZE,               0x4100C350,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL5PARAM,              0x4100C360,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL6,                   0x4100C380,__READ_WRITE );
__IO_REG32_BIT(PXP_OL6SIZE,               0x4100C390,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL6PARAM,              0x4100C3A0,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32(    PXP_OL7,                   0x4100C3C0,__READ_WRITE );
__IO_REG32_BIT(PXP_OL7SIZE,               0x4100C3D0,__READ_WRITE ,__pxp_olxsize_bits);
__IO_REG32_BIT(PXP_OL7PARAM,              0x4100C3E0,__READ_WRITE ,__pxp_olxparam_bits);
__IO_REG32_BIT(PXP_CSC2CTRL,              0x4100C400,__READ_WRITE ,__pxp_csc2ctrl_bits);
__IO_REG32_BIT(PXP_CSC2COEF0,             0x4100C410,__READ_WRITE ,__pxp_csc2coef0_bits);
__IO_REG32_BIT(PXP_CSC2COEF1,             0x4100C420,__READ_WRITE ,__pxp_csc2coef1_bits);
__IO_REG32_BIT(PXP_CSC2COEF2,             0x4100C430,__READ_WRITE ,__pxp_csc2coef2_bits);
__IO_REG32_BIT(PXP_CSC2COEF3,             0x4100C440,__READ_WRITE ,__pxp_csc2coef3_bits);
__IO_REG32_BIT(PXP_CSC2COEF4,             0x4100C450,__READ_WRITE ,__pxp_csc2coef4_bits);
__IO_REG32_BIT(PXP_CSC2COEF5,             0x4100C460,__READ_WRITE ,__pxp_csc2coef5_bits);
__IO_REG32_BIT(PXP_LUT_CTRL,              0x4100C470,__READ_WRITE ,__pxp_lut_ctrl_bits);
__IO_REG32_BIT(PXP_LUT,                   0x4100C480,__READ_WRITE ,__pxp_lut_bits);
__IO_REG32_BIT(PXP_HIST_CTRL,             0x4100C490,__READ_WRITE ,__pxp_hist_ctrl_bits);
__IO_REG32_BIT(PXP_HIST2_PARAM,           0x4100C4A0,__READ_WRITE ,__pxp_hist2_param_bits);
__IO_REG32_BIT(PXP_HIST4_PARAM,           0x4100C4B0,__READ_WRITE ,__pxp_hist4_param_bits);
__IO_REG32_BIT(PXP_HIST8_PARAM0,          0x4100C4C0,__READ_WRITE ,__pxp_hist8_param0_bits);
__IO_REG32_BIT(PXP_HIST8_PARAM1,          0x4100C4D0,__READ_WRITE ,__pxp_hist8_param1_bits);
__IO_REG32_BIT(PXP_HIST16_PARAM0,         0x4100C4E0,__READ_WRITE ,__pxp_hist16_param0_bits);
__IO_REG32_BIT(PXP_HIST16_PARAM1,         0x4100C4F0,__READ_WRITE ,__pxp_hist16_param1_bits);
__IO_REG32_BIT(PXP_HIST16_PARAM2,         0x4100C500,__READ_WRITE ,__pxp_hist16_param2_bits);
__IO_REG32_BIT(PXP_HIST16_PARAM3,         0x4100C510,__READ_WRITE ,__pxp_hist16_param3_bits);

/***************************************************************************
 **
 **  QOSC
 **
 ***************************************************************************/
__IO_REG32_BIT(QOS_CTRL,                  0x41012000,__READ_WRITE ,__qos_ctrl_bits);
__IO_REG32(    QOS_CTRL_SET,              0x41012004,__WRITE      );
__IO_REG32(    QOS_CTRL_CLR,              0x41012008,__WRITE      );
__IO_REG32(    QOS_CTRL_TOG,              0x4101200C,__WRITE      );
__IO_REG32_BIT(QOS_AXI_QOS0,              0x41012010,__READ_WRITE ,__qos_axi_qos0_bits);
__IO_REG32(    QOS_AXI_QOS0_SET,          0x41012014,__WRITE      );
__IO_REG32(    QOS_AXI_QOS0_CLR,          0x41012018,__WRITE      );
__IO_REG32(    QOS_AXI_QOS0_TOG,          0x4101201C,__WRITE      );
__IO_REG32_BIT(QOS_AXI_QOS1,              0x41012020,__READ_WRITE ,__qos_axi_qos1_bits);
__IO_REG32(    QOS_AXI_QOS1_SET,          0x41012024,__WRITE      );
__IO_REG32(    QOS_AXI_QOS1_CLR,          0x41012028,__WRITE      );
__IO_REG32(    QOS_AXI_QOS1_TOG,          0x4101202C,__WRITE      );
__IO_REG32_BIT(QOS_AXI_QOS2,              0x41012030,__READ_WRITE ,__qos_axi_qos2_bits);
__IO_REG32(    QOS_AXI_QOS2_SET,          0x41012034,__WRITE      );
__IO_REG32(    QOS_AXI_QOS2_CLR,          0x41012038,__WRITE      );
__IO_REG32(    QOS_AXI_QOS2_TOG,          0x4101203C,__WRITE      );
__IO_REG32_BIT(QOS_EMI_PRIORITY0,         0x41012040,__READ_WRITE ,__qos_emi_priority0_bits);
__IO_REG32(    QOS_EMI_PRIORITY0_SET,     0x41012044,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY0_CLR,     0x41012048,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY0_TOG,     0x4101204C,__WRITE      );
__IO_REG32_BIT(QOS_EMI_PRIORITY1,         0x41012050,__READ_WRITE ,__qos_emi_priority1_bits);
__IO_REG32(    QOS_EMI_PRIORITY1_SET,     0x41012054,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY1_CLR,     0x41012058,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY1_TOG,     0x4101205C,__WRITE      );
__IO_REG32_BIT(QOS_EMI_PRIORITY2,         0x41012060,__READ_WRITE ,__qos_emi_priority2_bits);
__IO_REG32(    QOS_EMI_PRIORITY2_SET,     0x41012064,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY2_CLR,     0x41012068,__WRITE      );
__IO_REG32(    QOS_EMI_PRIORITY2_TOG,     0x4101206C,__WRITE      );
__IO_REG32_BIT(QOS_DISABLE,               0x41012070,__READ_WRITE ,__qos_disable_bits);
__IO_REG32(    QOS_DISABLE_SET,           0x41012074,__WRITE      );
__IO_REG32(    QOS_DISABLE_CLR,           0x41012078,__WRITE      );
__IO_REG32(    QOS_DISABLE_TOG,           0x4101207C,__WRITE      );
__IO_REG32_BIT(QOS_VERSION,               0x41012080,__READ       ,__qos_version_bits);

/***************************************************************************
 **
 **  ROMCP
 **
 ***************************************************************************/
__IO_REG32(    ROMCP_ROMPATCHD7,          0x63FB80D4,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD6,          0x63FB80D8,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD5,          0x63FB80DC,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD4,          0x63FB80E0,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD3,          0x63FB80E4,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD2,          0x63FB80E8,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD1,          0x63FB80EC,__READ_WRITE); 
__IO_REG32(    ROMCP_ROMPATCHD0,          0x63FB80F0,__READ_WRITE); 
__IO_REG32_BIT(ROMCP_ROMPATCHCNTL,        0x63FB80F4,__READ_WRITE,__rompatchcntl_bits);  
__IO_REG32_BIT(ROMCP_ROMPATCHENL,         0x63FB80FC,__READ_WRITE,__rompatchenl_bits);   
__IO_REG32_BIT(ROMCP_ROMPATCHA0,          0x63FB8100,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA1,          0x63FB8104,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA2,          0x63FB8108,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA3,          0x63FB810C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA4,          0x63FB8110,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA5,          0x63FB8114,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA6,          0x63FB8118,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA7,          0x63FB811C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA8,          0x63FB8120,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA9,          0x63FB8124,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA10,         0x63FB8128,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA11,         0x63FB812C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA12,         0x63FB8130,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA13,         0x63FB8134,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA14,         0x63FB8138,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHA15,         0x63FB813C,__READ_WRITE,__rompatcha_bits);    
__IO_REG32_BIT(ROMCP_ROMPATCHSR,          0x63FB8208,__READ_WRITE,__rompatchsr_bits);    

/***************************************************************************
 **
 **  SDMA
 **
 ***************************************************************************/
__IO_REG32(    SDMAARM_MC0PTR,            0x63FB0000,__READ_WRITE );
__IO_REG32_BIT(SDMAARM_INTR,              0x63FB0004,__READ_WRITE ,__sdma_intr_bits);
__IO_REG32_BIT(SDMAARM_STOP_STAT,         0x63FB0008,__READ       ,__sdma_stop_stat_bits);
__IO_REG32_BIT(SDMAARM_HSTART,            0x63FB000C,__READ_WRITE ,__sdma_hstart_bits);
__IO_REG32_BIT(SDMAARM_EVTOVR,            0x63FB0010,__READ_WRITE ,__sdma_evtovr_bits);
__IO_REG32_BIT(SDMAARM_DSPOVR,            0x63FB0014,__READ_WRITE ,__sdma_dspovr_bits);
__IO_REG32_BIT(SDMAARM_HOSTOVR,           0x63FB0018,__READ_WRITE ,__sdma_hostovr_bits);
__IO_REG32_BIT(SDMAARM_EVTPEND,           0x63FB001C,__READ       ,__sdma_evtpend_bits);
__IO_REG32_BIT(SDMAARM_RESET,             0x63FB0024,__READ       ,__sdma_reset_bits);
__IO_REG32_BIT(SDMAARM_EVTERR,            0x63FB0028,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMAARM_INTRMASK,          0x63FB002C,__READ_WRITE ,__sdma_intrmask_bits);
__IO_REG32_BIT(SDMAARM_PSW,               0x63FB0030,__READ       ,__sdma_psw_bits);
__IO_REG32_BIT(SDMAARM_EVTERRDBG,         0x63FB0034,__READ       ,__sdma_evterr_bits);
__IO_REG32_BIT(SDMAARM_CONFIG,            0x63FB0038,__READ_WRITE ,__sdma_config_bits);
__IO_REG32_BIT(SDMAARM_LOCK,              0x63FB003C,__READ_WRITE ,__sdma_lock_bits);
__IO_REG32_BIT(SDMAARM_ONCE_ENB,          0x63FB0040,__READ_WRITE ,__sdma_once_enb_bits);
__IO_REG32(    SDMAARM_ONCE_DATA,         0x63FB0044,__READ_WRITE );
__IO_REG32_BIT(SDMAARM_ONCE_INSTR,        0x63FB0048,__READ_WRITE ,__sdma_once_instr_bits);
__IO_REG32_BIT(SDMAARM_ONCE_STAT,         0x63FB004C,__READ       ,__sdma_once_stat_bits);
__IO_REG32_BIT(SDMAARM_ONCE_CMD,          0x63FB0050,__READ_WRITE ,__sdma_once_cmd_bits);
__IO_REG32_BIT(SDMAARM_ILLINSTADDR,       0x63FB0058,__READ_WRITE ,__sdma_illinstaddr_bits);
__IO_REG32_BIT(SDMAARM_CHN0ADDR,          0x63FB005C,__READ_WRITE ,__sdma_chn0addr_bits);
__IO_REG32_BIT(SDMAARM_EVT_MIRROR,        0x63FB0060,__READ       ,__sdma_evt_mirror_bits);
__IO_REG32_BIT(SDMAARM_EVT_MIRROR2,       0x63FB0064,__READ       ,__sdma_evt_mirror2_bits);
__IO_REG32_BIT(SDMAARM_XTRIG_CONF1,       0x63FB0070,__READ_WRITE ,__sdma_xtrig1_conf_bits);
__IO_REG32_BIT(SDMAARM_XTRIG_CONF2,       0x63FB0074,__READ_WRITE ,__sdma_xtrig2_conf_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI0,           0x63FB0100,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI1,           0x63FB0104,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI2,           0x63FB0108,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI3,           0x63FB010C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI4,           0x63FB0110,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI5,           0x63FB0114,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI6,           0x63FB0118,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI7,           0x63FB011C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI8,           0x63FB0120,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI9,           0x63FB0124,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI10,          0x63FB0128,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI11,          0x63FB012C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI12,          0x63FB0130,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI13,          0x63FB0134,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI14,          0x63FB0138,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI15,          0x63FB013C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI16,          0x63FB0140,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI17,          0x63FB0144,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI18,          0x63FB0148,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI19,          0x63FB014C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI20,          0x63FB0150,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI21,          0x63FB0154,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI22,          0x63FB0158,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI23,          0x63FB015C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI24,          0x63FB0160,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI25,          0x63FB0164,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI26,          0x63FB0168,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI27,          0x63FB016C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI28,          0x63FB0170,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI29,          0x63FB0174,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI30,          0x63FB0178,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32_BIT(SDMAARM_CHNPRI31,          0x63FB017C,__READ_WRITE ,__sdma_chnpri_bits);
__IO_REG32(    SDMAARM_CHNENBL0,          0x63FB0200,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL1,          0x63FB0204,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL2,          0x63FB0208,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL3,          0x63FB020C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL4,          0x63FB0210,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL5,          0x63FB0214,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL6,          0x63FB0218,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL7,          0x63FB021C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL8,          0x63FB0220,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL9,          0x63FB0224,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL10,         0x63FB0228,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL11,         0x63FB022C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL12,         0x63FB0230,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL13,         0x63FB0234,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL14,         0x63FB0238,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL15,         0x63FB023C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL16,         0x63FB0240,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL17,         0x63FB0244,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL18,         0x63FB0248,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL19,         0x63FB024C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL20,         0x63FB0250,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL21,         0x63FB0254,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL22,         0x63FB0258,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL23,         0x63FB025C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL24,         0x63FB0260,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL25,         0x63FB0264,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL26,         0x63FB0268,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL27,         0x63FB026C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL28,         0x63FB0270,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL29,         0x63FB0274,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL30,         0x63FB0278,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL31,         0x63FB027C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL32,         0x63FB0280,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL33,         0x63FB0284,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL34,         0x63FB0288,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL35,         0x63FB028C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL36,         0x63FB0290,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL37,         0x63FB0294,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL38,         0x63FB0298,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL39,         0x63FB029C,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL40,         0x63FB02A0,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL41,         0x63FB02A4,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL42,         0x63FB02A8,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL43,         0x63FB02AC,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL44,         0x63FB02B0,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL45,         0x63FB02B4,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL46,         0x63FB02B8,__READ_WRITE );
__IO_REG32(    SDMAARM_CHNENBL47,         0x63FB02BC,__READ_WRITE );

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
 **  SRC
 **
 ***************************************************************************/
__IO_REG32_BIT(SRC_SCR,                   0x53FD0000,__READ_WRITE ,__src_scr_bits);
__IO_REG32_BIT(SRC_SBMR,                  0x53FD0004,__READ       ,__src_sbmr_bits);
__IO_REG32_BIT(SRC_SRSR,                  0x53FD0008,__READ_WRITE ,__src_srsr_bits);
__IO_REG32_BIT(SRC_SISR,                  0x53FD0014,__READ       ,__src_sisr_bits);
__IO_REG32_BIT(SRC_SIMR,                  0x53FD0018,__READ_WRITE ,__src_simr_bits);

/***************************************************************************
 **
 **  SRPGC
 **
 ***************************************************************************/
__IO_REG32_BIT(SRPGC_SRPGCR,              0x53FD8280,__READ_WRITE ,__srpgc_srpgcr_bits);
__IO_REG32_BIT(SRPGC_PUPSCR,              0x53FD8284,__READ_WRITE ,__srpgc_pupscr_bits);
__IO_REG32_BIT(SRPGC_PDNSCR,              0x53FD8288,__READ_WRITE ,__srpgc_pdnscr_bits);
__IO_REG32_BIT(SRPGC_SRPGSR,              0x53FD828C,__READ_WRITE ,__srpgc_srpgsr_bits);
__IO_REG32_BIT(SRPGC_SRPGDR,              0x53FD8290,__READ_WRITE ,__srpgc_srpgdr_bits);

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
__IO_REG32_BIT(SSI1_SISR,                 0x63FCC014,__READ_WRITE ,__sisr_bits);
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
__IO_REG32_BIT(SSI1_SACCST,               0x63FCC050,__READ       ,__saccst_bits);
__IO_REG32_BIT(SSI1_SACCEN,               0x63FCC054,__WRITE      ,__saccen_bits);
__IO_REG32_BIT(SSI1_SACCDIS,              0x63FCC058,__WRITE      ,__saccdis_bits);

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
__IO_REG32_BIT(SSI2_SISR,                 0x50014014,__READ_WRITE ,__sisr_bits);
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
__IO_REG32_BIT(SSI2_SACCST,               0x50014050,__READ       ,__saccst_bits);
__IO_REG32_BIT(SSI2_SACCEN,               0x50014054,__WRITE      ,__saccen_bits);
__IO_REG32_BIT(SSI2_SACCDIS,              0x50014058,__WRITE      ,__saccdis_bits);

/***************************************************************************
 **
 ** Temp Sensor
 **
 ***************************************************************************/
__IO_REG32_BIT(TEMPSENSOR_ANADIG_CTRL,      0x41018080,__READ_WRITE ,__tempsensor_anadig_ctrl_bits);
__IO_REG32(    TEMPSENSOR_ANADIG_CTRL_SET,  0x41018084,__WRITE      );
__IO_REG32(    TEMPSENSOR_ANADIG_CTRL_CLR,  0x41018088,__WRITE      );
__IO_REG32(    TEMPSENSOR_ANADIG_CTRL_TOG,  0x4101808C,__WRITE      );

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
__IO_REG32_BIT(UART1_URXD,                0x53FBC000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART1_UTXD,                0x53FBC040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART1_UCR1,                0x53FBC080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART1_UCR2,                0x53FBC084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART1_UCR3,                0x53FBC088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART1_UCR4,                0x53FBC08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART1_UFCR,                0x53FBC090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART1_USR1,                0x53FBC094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART1_USR2,                0x53FBC098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART1_UESC,                0x53FBC09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART1_UTIM,                0x53FBC0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART1_UBIR,                0x53FBC0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART1_UBMR,                0x53FBC0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART1_UBRC,                0x53FBC0AC,__READ        ,__ubrc_bits);
__IO_REG32_BIT(UART1_ONEMS,               0x53FBC0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART1_UTS,                 0x53FBC0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2_URXD,                0x53FC0000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART2_UTXD,                0x53FC0040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART2_UCR1,                0x53FC0080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART2_UCR2,                0x53FC0084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART2_UCR3,                0x53FC0088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART2_UCR4,                0x53FC008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART2_UFCR,                0x53FC0090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART2_USR1,                0x53FC0094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART2_USR2,                0x53FC0098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART2_UESC,                0x53FC009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART2_UTIM,                0x53FC00A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART2_UBIR,                0x53FC00A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART2_UBMR,                0x53FC00A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART2_UBRC,                0x53FC00AC,__READ        ,__ubrc_bits);
__IO_REG32_BIT(UART2_ONEMS,               0x53FC00B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART2_UTS,                 0x53FC00B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(UART3_URXD,                0x5000C000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART3_UTXD,                0x5000C040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART3_UCR1,                0x5000C080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART3_UCR2,                0x5000C084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART3_UCR3,                0x5000C088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART3_UCR4,                0x5000C08C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART3_UFCR,                0x5000C090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART3_USR1,                0x5000C094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART3_USR2,                0x5000C098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART3_UESC,                0x5000C09C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART3_UTIM,                0x5000C0A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART3_UBIR,                0x5000C0A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART3_UBMR,                0x5000C0A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART3_UBRC,                0x5000C0AC,__READ        ,__ubrc_bits);
__IO_REG32_BIT(UART3_ONEMS,               0x5000C0B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART3_UTS,                 0x5000C0B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(UART4_URXD,                0x53FF0000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART4_UTXD,                0x53FF0040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART4_UCR1,                0x53FF0080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART4_UCR2,                0x53FF0084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART4_UCR3,                0x53FF0088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART4_UCR4,                0x53FF008C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART4_UFCR,                0x53FF0090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART4_USR1,                0x53FF0094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART4_USR2,                0x53FF0098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART4_UESC,                0x53FF009C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART4_UTIM,                0x53FF00A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART4_UBIR,                0x53FF00A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART4_UBMR,                0x53FF00A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART4_UBRC,                0x53FF00AC,__READ        ,__ubrc_bits);
__IO_REG32_BIT(UART4_ONEMS,               0x53FF00B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART4_UTS,                 0x53FF00B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(UART5_URXD,                0x63F9000,__READ        ,__urxd_bits);
__IO_REG32_BIT(UART5_UTXD,                0x63F9040,__WRITE       ,__utxd_bits);
__IO_REG32_BIT(UART5_UCR1,                0x63F9080,__READ_WRITE  ,__ucr1_bits);
__IO_REG32_BIT(UART5_UCR2,                0x63F9084,__READ_WRITE  ,__ucr2_bits);
__IO_REG32_BIT(UART5_UCR3,                0x63F9088,__READ_WRITE  ,__ucr3_bits);
__IO_REG32_BIT(UART5_UCR4,                0x63F908C,__READ_WRITE  ,__ucr4_bits);
__IO_REG32_BIT(UART5_UFCR,                0x63F9090,__READ_WRITE  ,__ufcr_bits);
__IO_REG32_BIT(UART5_USR1,                0x63F9094,__READ_WRITE  ,__usr1_bits);
__IO_REG32_BIT(UART5_USR2,                0x63F9098,__READ_WRITE  ,__usr2_bits);
__IO_REG32_BIT(UART5_UESC,                0x63F909C,__READ_WRITE  ,__uesc_bits);
__IO_REG32_BIT(UART5_UTIM,                0x63F90A0,__READ_WRITE  ,__utim_bits);
__IO_REG32_BIT(UART5_UBIR,                0x63F90A4,__READ_WRITE  ,__ubir_bits);
__IO_REG32_BIT(UART5_UBMR,                0x63F90A8,__READ_WRITE  ,__ubmr_bits);
__IO_REG32_BIT(UART5_UBRC,                0x63F90AC,__READ        ,__ubrc_bits);
__IO_REG32_BIT(UART5_ONEMS,               0x63F90B0,__READ_WRITE  ,__onems_bits);
__IO_REG32_BIT(UART5_UTS,                 0x63F90B4,__READ_WRITE  ,__uts_bits);

/***************************************************************************
 **
 **  USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USBOH1_USB_CTRL,           0x53FC4800,__READ_WRITE ,__usboh1_usb_ctrl_bits);
__IO_REG32_BIT(USBOH1_UTMI_CLK_VLD,       0x53FC4804,__READ       ,__usboh1_utmi_clk_vld_bits);
__IO_REG32_BIT(USBOH1_OTG_PHY_CTRL_0,     0x53FC4808,__READ_WRITE ,__usboh1_otg_phy_ctrl_1_bits);
__IO_REG32_BIT(USBOH1_OTG_PHY_CTRL_1,     0x53FC480C,__READ_WRITE ,__usboh1_otg_phy_ctrl_0_bits);
__IO_REG32_BIT(USBOH1_USB_CTRL_1,         0x53FC4810,__READ_WRITE ,__usboh1_usb_ctrl_1_bits);
__IO_REG32_BIT(USBOH1_USB_CTRL_2,         0x53FC4814,__READ_WRITE ,__usboh1_usb_ctrl_2_bits);
__IO_REG32_BIT(USBOH1_UH1_PHY_CTRL_0,     0x53FC481C,__READ_WRITE ,__usboh1_uh1_phy_ctrl_0_bits);
__IO_REG32_BIT(USBOH1_UH1_PHY_CTRL_1,     0x53FC4820,__READ_WRITE ,__usboh1_uh1_phy_ctrl_1_bits);
__IO_REG32_BIT(USBOH1_USB_CLKONOFF_CTRL,  0x53FC4824,__READ_WRITE ,__usboh1_usb_clkonoff_ctrl_bits);
__IO_REG32(    USB_ID,                    0x53F80000,__READ       );
__IO_REG32_BIT(USB_HWGENERAL,             0x53F80004,__READ       ,__usb_hwgeneral_bits);
__IO_REG32_BIT(USB_HWHOST,                0x53F80008,__READ       ,__usb_hwhost_bits);
__IO_REG32_BIT(USB_HWDEVICE,              0x53F8000C,__READ       ,__usb_hwdevice_bits);
__IO_REG32_BIT(USB_HWTXBUF,               0x53F80010,__READ       ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(USB_HWRXBUF,               0x53F80014,__READ       ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(USB_GPTIMER0LD,            0x53F80080,__READ_WRITE ,__usb_gptimerld_bits);
__IO_REG32_BIT(USB_GPTIMER0CTRL,          0x53F80084,__READ_WRITE ,__usb_gptimerctrl_bits);
__IO_REG32_BIT(USB_GPTIMER1LD,            0x53F80088,__READ_WRITE ,__usb_gptimerld_bits);
__IO_REG32_BIT(USB_GPTIMER1CTRL,          0x53F8008C,__READ_WRITE ,__usb_gptimerctrl_bits);
__IO_REG32_BIT(USB_SBUSCFG,               0x53F80090,__READ_WRITE ,__usb_sbuscfg_bits);
__IO_REG8(     USB_CAPLENGTH,             0x53F80100,__READ       );
__IO_REG16(    USB_HCIVERSION,            0x53F80102,__READ       );
__IO_REG32_BIT(USB_HCSPARAMS,             0x53F80104,__READ       ,__usb_hcsparams_bits);
__IO_REG32_BIT(USB_HCCPARAMS,             0x53F80108,__READ       ,__usb_hccparams_bits);
__IO_REG16(    USB_DCIVERSION,            0x53F80120,__READ       );
__IO_REG32_BIT(USB_DCCPARAMS,             0x53F80124,__READ       ,__usb_dccparams_bits);
__IO_REG32_BIT(USB_USBCMD,                0x53F80140,__READ_WRITE ,__usb_usbcmd_bits);
__IO_REG32_BIT(USB_USBSTS,                0x53F80144,__READ_WRITE ,__usb_usbsts_bits);
__IO_REG32_BIT(USB_USBINTR,               0x53F80148,__READ_WRITE ,__usb_usbintr_bits);
__IO_REG32_BIT(USB_FRINDEX,               0x53F8014C,__READ_WRITE ,__usb_frindex_bits);
__IO_REG32_BIT(USB_PERIODICLISTBASE,      0x53F80154,__READ_WRITE ,__usb_periodiclistbase_bits);
#define USB_DEVICEADDR      USB_PERIODICLISTBASE
#define USB_DEVICEADDR_bit  USB_PERIODICLISTBASE_bit
__IO_REG32_BIT(USB_ASYNCLISTADDR,         0x53F80158,__READ_WRITE ,__usb_asynclistaddr_bits);
#define USB_ENDPOINTLISTADDR      USB_ASYNCLISTADDR
#define USB_ENDPOINTLISTADDR_bit  USB_ASYNCLISTADDR_bit
__IO_REG32_BIT(USB_BURSTSIZE,             0x53F80160,__READ_WRITE ,__usb_burstsize_bits);
__IO_REG32_BIT(USB_TXFILLTUNING,          0x53F80164,__READ_WRITE ,__usb_txfilltuning_bits);
__IO_REG32_BIT(USB_IC_USB,                0x53F8016C,__READ_WRITE ,__usb_ic_usb_bits);
__IO_REG32_BIT(USB_ULPIVIEW,              0x53F80170,__READ_WRITE ,__usb_ulpiview_bits);
__IO_REG32_BIT(USB_PORTSC,                0x53F80184,__READ_WRITE ,__usb_portsc_bits);
__IO_REG32_BIT(USB_OTGSC,                 0x53F801A4,__READ_WRITE ,__usb_otgsc_bits);
__IO_REG32_BIT(USB_USBMODE,               0x53F801A8,__READ_WRITE ,__usb_usbmode_bits);
__IO_REG32_BIT(USB_ENDPTSETUPSTAT,        0x53F801AC,__READ_WRITE ,__usb_endptsetupstat_bits);
__IO_REG32_BIT(USB_ENDPTPRIME,            0x53F801B0,__READ_WRITE ,__usb_endptprime_bits);
__IO_REG32_BIT(USB_ENDPTFLUSH,            0x53F801B4,__READ_WRITE ,__usb_endptflush_bits);
__IO_REG32_BIT(USB_ENDPTSTAT,             0x53F801B8,__READ       ,__usb_endptstat_bits);
__IO_REG32_BIT(USB_ENDPTCOMPLETE,         0x53F801BC,__READ_WRITE ,__usb_endptcomplete_bits);
__IO_REG32_BIT(USB_ENDPTCTRL0,            0x53F801C0,__READ_WRITE ,__usb_endptctrl0_bits);
__IO_REG32_BIT(USB_ENDPTCTRL1,            0x53F801C4,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL2,            0x53F801C8,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL3,            0x53F801CC,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL4,            0x53F801D0,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL5,            0x53F801D4,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL6,            0x53F801D8,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL7,            0x53F801DC,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL8,            0x53F801E0,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL9,            0x53F801E4,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL10,           0x53F801E8,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL11,           0x53F801EC,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL12,           0x53F801F0,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL13,           0x53F801F4,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL14,           0x53F801F8,__READ_WRITE ,__usb_endptctrl_bits);
__IO_REG32_BIT(USB_ENDPTCTRL15,           0x53F801FC,__READ_WRITE ,__usb_endptctrl_bits);

/***************************************************************************
 **
 **  WDOG1
 **
 ***************************************************************************/
__IO_REG16_BIT(WDOG1_WCR,                 0x53F98000,__READ_WRITE ,__wcr_bits);
__IO_REG16(    WDOG1_WSR,                 0x53F98002,__READ_WRITE );
__IO_REG16_BIT(WDOG1_WRSR,                0x53F98004,__READ       ,__wrsr_bits);
__IO_REG16_BIT(WDOG1_WICR,                0x53F98006,__READ_WRITE ,__wicr_bits);
__IO_REG16_BIT(WDOG1_WMCR,                0x53F98008,__READ_WRITE ,__wmcr_bits);

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
 **   MCIMX508 interrupt sources
 **
 ***************************************************************************/
#define INT_ESDHC_1            1              /* Enhanced SDHC1 Interrupt Request */
#define INT_ESDHC_2            2              /* Enhanced SDHC2 Interrupt Request */
#define INT_ESDHC_3            3              /* Enhanced SDHC3 Interrupt Request */
#define INT_ESDHC_4            4              /* Enhanced SDHC4 Interrupt Request */
#define INT_DAP						     5              /* DAP Interrupt Request */
#define INT_SDMA               6              /* “AND” of all 48 interrupts from all the channels */
#define INT_IOMUX 	           7              /* External Interrupt Request, usually used as POWER FAIL interrupt */
#define INT_UART_4             13             /* UART-4 ORed interrupt */
#define INT_USB_HOST           14             /* USB Host  */
#define INT_USB_OTG            18             /* USB OTG */
#define INT_DRAM_MC            19             /* DRAM MCInterrupt Request */
#define INT_ELCDIF             20             /* eLCDIF Interrupt Request */
#define INT_EPXP               21             /* ePXP Interrupt Request */
#define INT_SRTC_NTZ           24             /* SRTC Consolidated Interrupt. Non TZ. */
#define INT_SRTC_TZ            25             /* SRTC Security Interrupt. TZ. */
#define INT_EPDC               27             /* EPDC Interrupt Request */
#define INT_NIC                28             /* Perfmon Interrupt Request */
#define INT_SSI_1              29             /* SSI-1 Interrupt Request */
#define INT_SSI_2              30             /* SSI-2 Interrupt Request */
#define INT_UART_1           	 31             /* UART-1 ORed interrupt */
#define INT_UART_2           	 32             /* UART-2 ORed interrupt */
#define INT_UART_3           	 33             /* UART-3 ORed interrupt */
#define INT_ECSPI_1         	 36             /* ECSPI-1 interrupt request line to the core. */
#define INT_ECSPI_2            37             /* ECSPI-2 interrupt request line to the core. */
#define INT_CSPI               38             /* CSPI interrupt request line to the core. */
#define INT_GPT                39             /* “OR” of GPT Rollover interrupt line, Input Capture 1 & 2 lines, Output Compare 1,2 &3 Interrupt lines */
#define INT_EPIT               40             /* EPIT-1 output compare interrupt */
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
#define INT_KPP                60             /* Keypad Interrupt */
#define INT_PWM_1        			 61             /* CumuUART-5lative interrupt line. */
#define INT_I2C_1           	 62             /* I2C-1 Interrupt */
#define INT_I2C_2           	 63             /* I2C-2 Interrupt */
#define INT_I2C_3           	 64             /* I2C-3 Interrupt */
#define INT_ANALOG_1        	 66             /* Interrupt Request to indicate DC-DC OK */
#define INT_ANALOG_2        	 67             /* Interrupt Request of thermal alarm */
#define INT_ANALOG_3        	 68             /* Interrupt Request for signal irq_ana3 */
#define INT_ANALOG_4        	 69             /* Interrupt Request for signal irq_ana4 */
#define INT_CCM_1            	 71             /* CCM, Interrupt Request 1 */
#define INT_CCM_2            	 72             /* CCM, Interrupt Request 2 */
#define INT_GPC_1            	 73             /* GPC, Interrupt Request 1 */
#define INT_GPC_2            	 74             /* GPC, Interrupt Request 2 */
#define INT_SRC            	   75             /* SRC interrupt request */
#define INT_TIGERP_PLATFORM_1  76             /* Neon Monitor Interrupt */
#define INT_TIGERP_PLATFORM_2  77             /* Performance Unit Interrupt (nPMUIRQ) */
#define INT_TIGERP_PLATFORM_3  78             /* CTI IRQ */
#define INT_TIGERP_PLATFORM_4  79             /* Debug Interrupt, from Cross-Trigger 1 Interface 1 */
#define INT_TIGERP_PLATFORM_5  80             /* Debug Interrupt, from Cross-Trigger 1 Interface 0 */
#define INT_GPU2D_1     	   	 84             /* General Interrupt */
#define INT_GPU2D_2        	   85             /* Busy signal (for S/W power gating feasibility) */
#define INT_UART_5        	   86             /* UART-5 ORed interrupt */
#define INT_FEC           	   87             /* Fast Interrupt Request (OR of 13 interrupt sources) */
#define INT_OWIRE          	   88             /* 1-Wire Interrupt Request */
#define INT_TIGERP_PLATFORM_6  89             /* Debug Interrupt, from Cross-Trigger 1 Interface 2 */
#define INT_SJC           	   90             /* SJC Interrupt Request */
#define INT_DCP_1_3      	   	 91             /* Interrupt Request for channels 1-3 */
#define INT_DCP_0         	   92             /* Interrupt Request for channel 0 */
#define INT_DCP_0_3        	   93             /* secure Interrupt Request for channels 0-3 */
#define INT_PWM_2          	   94             /* Cumulative interrupt line */
#define INT_RNGB          	   97             /* RNGB Interrupt Request */
#define INT_TIGERP_PLATFORM_7  98             /* Debug Interrupt, from Cross-Trigger 1 Interface 3 */
#define INT_RAWNAND_BCH    	   100            /* BCH Interrupt Request */
#define INT_RAWNAND_GPMI     	 102            /* GPMI Interrupt Request */
#define INT_GPIO5_0_15         103            /* Combined interrupt indication for GPIO-5 signal 0 throughout 15 */
#define INT_GPIO5_16_31        104            /* Combined interrupt indication for GPIO-5 signal 16 throughout 31 */
#define INT_GPIO6_0_15         105            /* Combined interrupt indication for GPIO-6 signal 0 throughout 15 */
#define INT_GPIO6_16_31        106            /* Combined interrupt indication for GPIO-6 signal 16 throughout 31 */
#define INT_APBHDMA_CH0        110            /* Interrupt Request for channel 0 */
#define INT_APBHDMA_CH1        111            /* Interrupt Request for channel 1 */
#define INT_APBHDMA_CH2        112            /* Interrupt Request for channel 2 */
#define INT_APBHDMA_CH3        113            /* Interrupt Request for channel 3 */
#define INT_APBHDMA_CH4        114            /* Interrupt Request for channel 4 */
#define INT_APBHDMA_CH5        115            /* Interrupt Request for channel 5 */
#define INT_APBHDMA_CH6        116            /* Interrupt Request for channel 6 */
#define INT_APBHDMA_CH7        117            /* Interrupt Request for channel 7 */

#endif    /* __IOMCIMX508_H */
