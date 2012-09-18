/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST SPEAR600
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46178 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOSPEAR600_H
#define __IOSPEAR600_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x58 = 88 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   SPEAR600 SPECIAL FUNCTION REGISTERS
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

/* VICIRQSTATUS register */
typedef struct
{
  __REG32  IRQStatus0     : 1;
  __REG32  IRQStatus1     : 1;
  __REG32  IRQStatus2     : 1;
  __REG32  IRQStatus3     : 1;
  __REG32  IRQStatus4     : 1;
  __REG32  IRQStatus5     : 1;
  __REG32  IRQStatus6     : 1;
  __REG32  IRQStatus7     : 1;
  __REG32  IRQStatus8     : 1;
  __REG32  IRQStatus9     : 1;
  __REG32  IRQStatus10    : 1;
  __REG32  IRQStatus11    : 1;
  __REG32  IRQStatus12    : 1;
  __REG32  IRQStatus13    : 1;
  __REG32  IRQStatus14    : 1;
  __REG32  IRQStatus15    : 1;
  __REG32  IRQStatus16    : 1;
  __REG32  IRQStatus17    : 1;
  __REG32  IRQStatus18    : 1;
  __REG32  IRQStatus19    : 1;
  __REG32  IRQStatus20    : 1;
  __REG32  IRQStatus21    : 1;
  __REG32  IRQStatus22    : 1;
  __REG32  IRQStatus23    : 1;
  __REG32  IRQStatus24    : 1;
  __REG32  IRQStatus25    : 1;
  __REG32  IRQStatus26    : 1;
  __REG32  IRQStatus27    : 1;
  __REG32  IRQStatus28    : 1;
  __REG32  IRQStatus29    : 1;
  __REG32  IRQStatus30    : 1;
  __REG32  IRQStatus31    : 1;
} __vicirqstatus_bits;

/* VICFIQSTATUS register */
typedef struct
{
  __REG32  FIQStatus0     : 1;
  __REG32  FIQStatus1     : 1;
  __REG32  FIQStatus2     : 1;
  __REG32  FIQStatus3     : 1;
  __REG32  FIQStatus4     : 1;
  __REG32  FIQStatus5     : 1;
  __REG32  FIQStatus6     : 1;
  __REG32  FIQStatus7     : 1;
  __REG32  FIQStatus8     : 1;
  __REG32  FIQStatus9     : 1;
  __REG32  FIQStatus10    : 1;
  __REG32  FIQStatus11    : 1;
  __REG32  FIQStatus12    : 1;
  __REG32  FIQStatus13    : 1;
  __REG32  FIQStatus14    : 1;
  __REG32  FIQStatus15    : 1;
  __REG32  FIQStatus16    : 1;
  __REG32  FIQStatus17    : 1;
  __REG32  FIQStatus18    : 1;
  __REG32  FIQStatus19    : 1;
  __REG32  FIQStatus20    : 1;
  __REG32  FIQStatus21    : 1;
  __REG32  FIQStatus22    : 1;
  __REG32  FIQStatus23    : 1;
  __REG32  FIQStatus24    : 1;
  __REG32  FIQStatus25    : 1;
  __REG32  FIQStatus26    : 1;
  __REG32  FIQStatus27    : 1;
  __REG32  FIQStatus28    : 1;
  __REG32  FIQStatus29    : 1;
  __REG32  FIQStatus30    : 1;
  __REG32  FIQStatus31    : 1;
} __vicfiqstatus_bits;

/* VICRAWINTR register */
typedef struct
{
  __REG32  RawStatus0     : 1;
  __REG32  RawStatus1     : 1;
  __REG32  RawStatus2     : 1;
  __REG32  RawStatus3     : 1;
  __REG32  RawStatus4     : 1;
  __REG32  RawStatus5     : 1;
  __REG32  RawStatus6     : 1;
  __REG32  RawStatus7     : 1;
  __REG32  RawStatus8     : 1;
  __REG32  RawStatus9     : 1;
  __REG32  RawStatus10    : 1;
  __REG32  RawStatus11    : 1;
  __REG32  RawStatus12    : 1;
  __REG32  RawStatus13    : 1;
  __REG32  RawStatus14    : 1;
  __REG32  RawStatus15    : 1;
  __REG32  RawStatus16    : 1;
  __REG32  RawStatus17    : 1;
  __REG32  RawStatus18    : 1;
  __REG32  RawStatus19    : 1;
  __REG32  RawStatus20    : 1;
  __REG32  RawStatus21    : 1;
  __REG32  RawStatus22    : 1;
  __REG32  RawStatus23    : 1;
  __REG32  RawStatus24    : 1;
  __REG32  RawStatus25    : 1;
  __REG32  RawStatus26    : 1;
  __REG32  RawStatus27    : 1;
  __REG32  RawStatus28    : 1;
  __REG32  RawStatus29    : 1;
  __REG32  RawStatus30    : 1;
  __REG32  RawStatus31    : 1;
} __vicrawintr_bits;

/* VICINTSELECT register */
typedef struct
{
  __REG32  IntSelect0     : 1;
  __REG32  IntSelect1     : 1;
  __REG32  IntSelect2     : 1;
  __REG32  IntSelect3     : 1;
  __REG32  IntSelect4     : 1;
  __REG32  IntSelect5     : 1;
  __REG32  IntSelect6     : 1;
  __REG32  IntSelect7     : 1;
  __REG32  IntSelect8     : 1;
  __REG32  IntSelect9     : 1;
  __REG32  IntSelect10    : 1;
  __REG32  IntSelect11    : 1;
  __REG32  IntSelect12    : 1;
  __REG32  IntSelect13    : 1;
  __REG32  IntSelect14    : 1;
  __REG32  IntSelect15    : 1;
  __REG32  IntSelect16    : 1;
  __REG32  IntSelect17    : 1;
  __REG32  IntSelect18    : 1;
  __REG32  IntSelect19    : 1;
  __REG32  IntSelect20    : 1;
  __REG32  IntSelect21    : 1;
  __REG32  IntSelect22    : 1;
  __REG32  IntSelect23    : 1;
  __REG32  IntSelect24    : 1;
  __REG32  IntSelect25    : 1;
  __REG32  IntSelect26    : 1;
  __REG32  IntSelect27    : 1;
  __REG32  IntSelect28    : 1;
  __REG32  IntSelect29    : 1;
  __REG32  IntSelect30    : 1;
  __REG32  IntSelect31    : 1;
} __vicintselect_bits;

/* VICINTENABLE register */
typedef struct
{
  __REG32  IntEnable0     : 1;
  __REG32  IntEnable1     : 1;
  __REG32  IntEnable2     : 1;
  __REG32  IntEnable3     : 1;
  __REG32  IntEnable4     : 1;
  __REG32  IntEnable5     : 1;
  __REG32  IntEnable6     : 1;
  __REG32  IntEnable7     : 1;
  __REG32  IntEnable8     : 1;
  __REG32  IntEnable9     : 1;
  __REG32  IntEnable10    : 1;
  __REG32  IntEnable11    : 1;
  __REG32  IntEnable12    : 1;
  __REG32  IntEnable13    : 1;
  __REG32  IntEnable14    : 1;
  __REG32  IntEnable15    : 1;
  __REG32  IntEnable16    : 1;
  __REG32  IntEnable17    : 1;
  __REG32  IntEnable18    : 1;
  __REG32  IntEnable19    : 1;
  __REG32  IntEnable20    : 1;
  __REG32  IntEnable21    : 1;
  __REG32  IntEnable22    : 1;
  __REG32  IntEnable23    : 1;
  __REG32  IntEnable24    : 1;
  __REG32  IntEnable25    : 1;
  __REG32  IntEnable26    : 1;
  __REG32  IntEnable27    : 1;
  __REG32  IntEnable28    : 1;
  __REG32  IntEnable29    : 1;
  __REG32  IntEnable30    : 1;
  __REG32  IntEnable31    : 1;
} __vicintenable_bits;

/* VICINTENCLEAR register */
typedef struct
{
  __REG32  IntEnableClear0     : 1;
  __REG32  IntEnableClear1     : 1;
  __REG32  IntEnableClear2     : 1;
  __REG32  IntEnableClear3     : 1;
  __REG32  IntEnableClear4     : 1;
  __REG32  IntEnableClear5     : 1;
  __REG32  IntEnableClear6     : 1;
  __REG32  IntEnableClear7     : 1;
  __REG32  IntEnableClear8     : 1;
  __REG32  IntEnableClear9     : 1;
  __REG32  IntEnableClear10    : 1;
  __REG32  IntEnableClear11    : 1;
  __REG32  IntEnableClear12    : 1;
  __REG32  IntEnableClear13    : 1;
  __REG32  IntEnableClear14    : 1;
  __REG32  IntEnableClear15    : 1;
  __REG32  IntEnableClear16    : 1;
  __REG32  IntEnableClear17    : 1;
  __REG32  IntEnableClear18    : 1;
  __REG32  IntEnableClear19    : 1;
  __REG32  IntEnableClear20    : 1;
  __REG32  IntEnableClear21    : 1;
  __REG32  IntEnableClear22    : 1;
  __REG32  IntEnableClear23    : 1;
  __REG32  IntEnableClear24    : 1;
  __REG32  IntEnableClear25    : 1;
  __REG32  IntEnableClear26    : 1;
  __REG32  IntEnableClear27    : 1;
  __REG32  IntEnableClear28    : 1;
  __REG32  IntEnableClear29    : 1;
  __REG32  IntEnableClear30    : 1;
  __REG32  IntEnableClear31    : 1;
} __vicintenclear_bits;

/* VICSOFTINT register */
typedef struct
{
  __REG32  SoftInt0     : 1;
  __REG32  SoftInt1     : 1;
  __REG32  SoftInt2     : 1;
  __REG32  SoftInt3     : 1;
  __REG32  SoftInt4     : 1;
  __REG32  SoftInt5     : 1;
  __REG32  SoftInt6     : 1;
  __REG32  SoftInt7     : 1;
  __REG32  SoftInt8     : 1;
  __REG32  SoftInt9     : 1;
  __REG32  SoftInt10    : 1;
  __REG32  SoftInt11    : 1;
  __REG32  SoftInt12    : 1;
  __REG32  SoftInt13    : 1;
  __REG32  SoftInt14    : 1;
  __REG32  SoftInt15    : 1;
  __REG32  SoftInt16    : 1;
  __REG32  SoftInt17    : 1;
  __REG32  SoftInt18    : 1;
  __REG32  SoftInt19    : 1;
  __REG32  SoftInt20    : 1;
  __REG32  SoftInt21    : 1;
  __REG32  SoftInt22    : 1;
  __REG32  SoftInt23    : 1;
  __REG32  SoftInt24    : 1;
  __REG32  SoftInt25    : 1;
  __REG32  SoftInt26    : 1;
  __REG32  SoftInt27    : 1;
  __REG32  SoftInt28    : 1;
  __REG32  SoftInt29    : 1;
  __REG32  SoftInt30    : 1;
  __REG32  SoftInt31    : 1;
} __vicsoftint_bits;

/* VICSOFTINTCLEAR register */
typedef struct
{
  __REG32  SoftIntClear0     : 1;
  __REG32  SoftIntClear1     : 1;
  __REG32  SoftIntClear2     : 1;
  __REG32  SoftIntClear3     : 1;
  __REG32  SoftIntClear4     : 1;
  __REG32  SoftIntClear5     : 1;
  __REG32  SoftIntClear6     : 1;
  __REG32  SoftIntClear7     : 1;
  __REG32  SoftIntClear8     : 1;
  __REG32  SoftIntClear9     : 1;
  __REG32  SoftIntClear10    : 1;
  __REG32  SoftIntClear11    : 1;
  __REG32  SoftIntClear12    : 1;
  __REG32  SoftIntClear13    : 1;
  __REG32  SoftIntClear14    : 1;
  __REG32  SoftIntClear15    : 1;
  __REG32  SoftIntClear16    : 1;
  __REG32  SoftIntClear17    : 1;
  __REG32  SoftIntClear18    : 1;
  __REG32  SoftIntClear19    : 1;
  __REG32  SoftIntClear20    : 1;
  __REG32  SoftIntClear21    : 1;
  __REG32  SoftIntClear22    : 1;
  __REG32  SoftIntClear23    : 1;
  __REG32  SoftIntClear24    : 1;
  __REG32  SoftIntClear25    : 1;
  __REG32  SoftIntClear26    : 1;
  __REG32  SoftIntClear27    : 1;
  __REG32  SoftIntClear28    : 1;
  __REG32  SoftIntClear29    : 1;
  __REG32  SoftIntClear30    : 1;
  __REG32  SoftIntClear31    : 1;
} __vicsoftintclear_bits;

/* VICPROTECTION register */
typedef struct
{
  __REG32  Protection        : 1;
  __REG32                    :31;
} __vicprotection_bits;

/* VICVECTCNTL register */
typedef struct
{
  __REG32  IntSource         : 5;
  __REG32  E                 : 1;
  __REG32                    :26;
} __vicvectcntl_bits;

/* VICPERIPHID1 register */
typedef struct
{
  __REG8   Partnumber        : 4;
  __REG8   Designer          : 4;
} __vicperiphid1_bits;

/* VICPERIPHID2 register */
typedef struct
{
  __REG8   Designer          : 4;
  __REG8   Revision          : 4;
} __vicperiphid2_bits;

/* SoC_CFG_CTR register */
typedef struct
{
  __REG32  SoC_cfg                : 6;
  __REG32  SoC_applic	            : 2;
  __REG32  expi_itf_enb           : 1;
  __REG32  expi_clk_src           : 1;
  __REG32  expi_ras_enb           : 1;
  __REG32  expi_iobrg_enb         : 1;
  __REG32                         : 1;
  __REG32  full_ras_mode          : 1;
  __REG32                         : 2;
  __REG32  nand_16b			          : 1;
  __REG32  nand_disab 	          : 1;
  __REG32  dual_core		          : 1;
  __REG32  boot_sel			          : 1;
  __REG32                         :12;
} __soc_cfg_ctr_bits;

/* DIAG_CFG_CTR register */
typedef struct
{
  __REG32  SoC_dbg	              : 3;
  __REG32                         : 8;
  __REG32  sys_error              : 1;
  __REG32                         : 2;
  __REG32  debug_freez            : 2;
  __REG32                         :16;
} __diag_cfg_ctr_bits;

/* PLL 1/2_CTR register */
typedef struct
{
  __REG32  pll_lock               : 1;
  __REG32  pll_resetn             : 1;
  __REG32  pll_enable             : 1;
  __REG32  pll_control1           : 6;
  __REG32  pll_control2           : 5;
  __REG32                         :18;
} __pll_ctr_bits;

/* PLL1/2_FRQ registers */
typedef struct
{
  __REG32  pll_prediv_N           : 8;
  __REG32  pll_postdiv_P          : 3;
  __REG32                         : 5;
  __REG32  pll_fbkdiv_M           :16;
} __pll_frq_bits;

/* PLL1/2_MOD registers */
typedef struct
{
  __REG32  pll_slope              :16;
  __REG32  pll_modperiod          :13;
  __REG32                         : 3;
} __pll_mod_bits;

/* PLL_CLK_CFG register */
typedef struct
{
  __REG32  pll1_enb_clkout        : 1;
  __REG32  pll2_enb_clkout        : 1;
  __REG32  pll3_enb_clkout        : 1;
  __REG32                         :13;
  __REG32  sys_pll1_lock          : 1;
  __REG32  sys_pll2_lock          : 1;
  __REG32  usb_pll_lock           : 1;
  __REG32  mem_dll_lock           : 1;
  __REG32  pll1_clk_sel           : 3;
  __REG32                         : 1;
  __REG32  pll2_clk_sel           : 3;
  __REG32                         : 1;
  __REG32  mctr_clk_sel           : 3;
  __REG32                         : 1;
} __pll_clk_cfg_bits;

/* CORE_CLK_CFG register */
typedef struct
{
  __REG32  pclk_ratio_arm1        : 2;
  __REG32  pclk_ratio_arm2        : 2;
  __REG32  pclk_ratio_basc        : 2;
  __REG32  pclk_ratio_appl        : 2;
  __REG32  pclk_ratio_lwsp        : 2;
  __REG32  hclk_divsel            : 2;
  __REG32  clk1_divsel            : 1;
  __REG32  clk2_divsel            : 1;
  __REG32  hclk_clk1_ratio        : 2;
  __REG32  hclk_clk2_ratio        : 2;
  __REG32  ras_synt34_clksel      : 1;
  __REG32  osci30_div_en          : 1;
  __REG32  osci30_div_ratio       : 2;
  __REG32                         :10;
} __core_clk_cfg_bits;

/* PRPH_CLK_CFG register */
typedef struct
{
  __REG32  xtaltimeen             : 1;
  __REG32  plltimeen              : 1;
  __REG32  clcd_clksel            : 2;
  __REG32  uart_clksel            : 1;
  __REG32  irda_clksel            : 2;
  __REG32  rtc_disable            : 1;
  __REG32  gptmr1_clksel          : 1;
  __REG32  gptmr2_clksel          : 1;
  __REG32  gptmr3_clksel          : 1;
  __REG32  gptmr4_clksel          : 1;
  __REG32  gptmr5_clksel	        : 1;
  __REG32  gptmr1_freez           : 1;
  __REG32  gptmr2_freez           : 1;
  __REG32  gptmr3_freez           : 1;
  __REG32  gptmr4_freez           : 1;
  __REG32  gptmr5_freez           : 1;
  __REG32                         :14;
} __prph_clk_cfg_bits;

/* PERIP1_CLK_ENB register */
typedef struct
{
  __REG32  arm_enb                : 1;
  __REG32  arm_clkenb             : 1;
  __REG32  arm2_clkenb            : 1;
  __REG32  uart1_clkenb           : 1;
  __REG32  uart2_clkenb           : 1;
  __REG32  ssp1_clkenb            : 1;
  __REG32  ssp2_clkenb            : 1;
  __REG32  i2c_clkenb             : 1;
  __REG32  jpeg_clkenb            : 1;
  __REG32  fsmc_clkenb            : 1;
  __REG32  irda_clkenb            : 1;
  __REG32  gptm4_clkenb           : 1;
  __REG32  gptm5_clkenb           : 1;
  __REG32  gpio4_clkenb           : 1;
  __REG32  ssp3_clkenb            : 1;
  __REG32  adc_clkenb             : 1;
  __REG32  gptm3_clkenb           : 1;
  __REG32  rtc_clkenb             : 1;
  __REG32  gpio3_clkenb           : 1;
  __REG32  dma_clkenb             : 1;
  __REG32  rom_clkenb             : 1;
  __REG32  smi_clkenb             : 1;
  __REG32  clcd_clkenb            : 1;
  __REG32  gmac_clkenb            : 1;
  __REG32  usbdev_clkenb          : 1;
  __REG32  usbh1_clkenb           : 1;
  __REG32  usbh2_clkenb           : 1;
  __REG32  ddr_clkenb             : 1;
  __REG32                         : 1;
  __REG32  ddrcore_clkenb         : 1;
  __REG32  ddr_enb			          : 1;
  __REG32                         : 1;
} __perip1_clk_enb_bits;

/* RAS_CLK_ENB register */
typedef struct
{
  __REG32  hclk_clkenb            : 1;
  __REG32  pll1_clkenb            : 1;
  __REG32  pclkappl_clkenb        : 1;
  __REG32  clk32K_clkenb          : 1;
  __REG32  clk30M_clkenb          : 1;
  __REG32  clk48M_clkenb          : 1;
  __REG32  clk125M_clkenb         : 1;
  __REG32  pll2_clkenb            : 1;
  __REG32  ras_synt1_clkenb       : 1;
  __REG32  ras_synt2_clkenb       : 1;
  __REG32  ras_synt3_clkenb       : 1;
  __REG32  ras_synt4_clkenb       : 1;
  __REG32  pl_gpck1_clkenb        : 1;
  __REG32  pl_gpck2_clkenb        : 1;
  __REG32  pl_gpck3_clkenb        : 1;
  __REG32  pl_gpck4_clkenb        : 1;
  __REG32                         :16;
} __ras_clk_enb_bits;

/* AMEM_CFG_CTRL register */
typedef struct
{
  __REG32  amem_clk_enb           : 1;
  __REG32  amem_clk_sel           : 3;
  __REG32  amem_synt_enb          : 1;
  __REG32                         :10;
  __REG32  amem_rst               : 1;
  __REG32  amem_ydiv              : 8;
  __REG32  amem_xdiv              : 8;
} __amem_cfg_ctrl_bits;

/* EXPI_CLK_CFG Register */
typedef struct
{
  __REG32  portctr_clk_enb        : 1;
  __REG32  expi_clk_sel           : 3;
  __REG32  expi_synt_enb          : 1;
  __REG32  expi_dma_cfg0          : 1;
  __REG32  expi_dma_cfg1          : 1;
  __REG32  expi_dma_cfg2          : 1;
  __REG32  expi_dma_cfg3          : 1;
  __REG32  expi_srst			        : 1;
  __REG32  expi_clk_enb	          : 1;
  __REG32  expi_clk_retim         : 1;
  __REG32  expi_compr_sel         : 2;
  __REG32  expi_lopbck            : 1;
  __REG32  expi_rst			          : 1;
  __REG32  amem_ydiv              : 8;
  __REG32  amem_xdiv              : 8;
} __expi_clk_cfg_bits;

/* Auxiliary clock synthesizer register */
typedef struct
{
  __REG32  synt_ydiv              :12;
  __REG32                         : 4;
  __REG32  synt_xdiv              :12;
  __REG32                         : 2;
  __REG32  synt_clkout_sel        : 1;
  __REG32  synt_clk_enb           : 1;
} __irda_clk_synt_bits;

/* PERIP1_SOF_RST register */
typedef struct
{
  __REG32  arm1_enbr              : 1;
  __REG32  arm1_swrst             : 1;
  __REG32  arm2_swrst             : 1;
  __REG32  uart1_swrst            : 1;
  __REG32  uart2_swrst            : 1;
  __REG32  ssp1_swrst             : 1;
  __REG32  ssp2_swrst             : 1;
  __REG32  i2c_swrst              : 1;
  __REG32  jpeg_swrst             : 1;
  __REG32  fsmc_swrst             : 1;
  __REG32  irda_swrst            	: 1;
  __REG32  gptm4_swrst            : 1;
  __REG32  gptm5_swrst            : 1;
  __REG32  gpio4_swrst            : 1;
  __REG32  ssp3_swrst	            : 1;
  __REG32  adc_swrst              : 1;
  __REG32  gptm3_swrst            : 1;
  __REG32  rtc_swrst              : 1;
  __REG32  gpio3_swrst            : 1;
  __REG32  dma_swrst              : 1;
  __REG32  rom_swrst              : 1;
  __REG32  smi_swrst              : 1;
  __REG32  clcd_swrst             : 1;
  __REG32  gmac_swrst             : 1;
  __REG32  usbdev_swrst           : 1;
  __REG32  usbh1_swrst		        : 1;
  __REG32  usbh2_swrst			      : 1;
  __REG32  ddrctrl_swrst          : 1;
  __REG32  ram_swrst              : 1;
  __REG32  ddrcore_swrst          : 1;
  __REG32  ddr_enbr               : 1;
  __REG32  				                : 1;
} __perip1_sof_rst_bits;

/* RAS_SOF_RST register */
typedef struct
{
  __REG32  hclk_swrst             : 1;
  __REG32  pll1_swrst             : 1;
  __REG32  pclkappl_swrst         : 1;
  __REG32  Clk32K_swrst           : 1;
  __REG32  Clk30M_swrst           : 1;
  __REG32  clk48M_swrst           : 1;
  __REG32  clk125M_swrst          : 1;
  __REG32  pll2_swrst             : 1;
  __REG32  ras_synt1_swrst        : 1;
  __REG32  ras_synt2_swrst        : 1;
  __REG32  ras_synt3_swrst        : 1;
  __REG32  ras_synt4_swrst        : 1;
  __REG32  pl_gpck1_swrst         : 1;
  __REG32  pl_gpck2_swrst         : 1;
  __REG32  pl_gpck3_swrst         : 1;
  __REG32  pl_gpck4_swrst         : 1;
  __REG32                         :16;
} __ras_sof_rst_bits;

/* PRSC1/2/3_CLK_CFG register */
typedef struct
{
  __REG32  presc_m                :12;
  __REG32  presc_n                : 4;
  __REG32                         :16;
} __prsc_clk_cfg_bits;

/* ICM 1-10_ARB_CFG register bit assignments */
typedef struct
{
  __REG32  mtx_fix_pry_lyr0       : 3;
  __REG32  mtx_fix_pry_lyr1       : 3;
  __REG32  mtx_fix_pry_lyr2       : 3;
  __REG32  mtx_fix_pry_lyr3       : 3;
  __REG32  mtx_fix_pry_lyr4       : 3;
  __REG32  mtx_fix_pry_lyr5       : 3;
  __REG32  mtx_fix_pry_lyr6       : 3;
  __REG32  mtx_fix_pry_lyr7       : 3;
  __REG32                         : 4;
  __REG32  mxt_rndrb_pry_lyr      : 3;
  __REG32  mtx_arb_type           : 1;
} __icm_arb_cfg_bits;

/* DMA_CHN_CFG register */
typedef struct
{
  __REG32  dma_cfg_chan00         : 2;
  __REG32  dma_cfg_chan01         : 2;
  __REG32  dma_cfg_chan02         : 2;
  __REG32  dma_cfg_chan03         : 2;
  __REG32  dma_cfg_chan04         : 2;
  __REG32  dma_cfg_chan05         : 2;
  __REG32  dma_cfg_chan06         : 2;
  __REG32  dma_cfg_chan07         : 2;
  __REG32  dma_cfg_chan08         : 2;
  __REG32  dma_cfg_chan09         : 2;
  __REG32  dma_cfg_chan10         : 2;
  __REG32  dma_cfg_chan11         : 2;
  __REG32  dma_cfg_chan12         : 2;
  __REG32  dma_cfg_chan13         : 2;
  __REG32  dma_cfg_chan14         : 2;
  __REG32  dma_cfg_chan15         : 2;
} __dma_chn_cfg_bits;

/* USB2_PHY_CFG register */
typedef struct
{
  __REG32  dynamic_pwdn           : 1;
  __REG32  pll_pwdn		            : 1;
  __REG32                         : 1;
  __REG32  usbh_overcur           : 1;
  __REG32                         : 4;
  __REG32  phyreset_chn1          : 1;
  __REG32  phyreset_chn2          : 1;
  __REG32  phyreset_chn3          : 1;
  __REG32                         : 1;
  __REG32  rxerror1_usbdv         : 1;
  __REG32  rxerror2_usbh1         : 1;
  __REG32  rxerror3_usbh2         : 1;
  __REG32                         :17;
} __usb2_phy_cfg_bits;

/* GMAC_CFG_CTR Register */
typedef struct
{
  __REG32  mili_reverse           : 1;
  __REG32                         : 1;
  __REG32  gmac_clk_sel           : 2;
  __REG32  gmac_synt_enb          : 1;
  __REG32                         :27;
} __gmac_cfg_ctr_bits;

/* EXPI_CFG_CTR Register */
typedef struct
{
  __REG32  icm9eh2h_rdpref        : 1;
  __REG32  icm9eh2h_sstall        : 1;
  __REG32  icm9eh2h_sflush        : 1;
  __REG32  icm9eh2h_irq		        : 1;
  __REG32  icm9eh2h_rstmst        : 1;
  __REG32  icm9eh2h_rstslv        : 1;
  __REG32  icm9eh2h_tikenb        : 1;
  __REG32  expi_fulladdr_enb      : 1;
  __REG32                         : 4;
  __REG32  icm8eh2h_rdpref        : 1;
  __REG32  icm8eh2h_sstall        : 1;
  __REG32  icm8eh2h_sflush        : 1;
  __REG32  icm8eh2h_irq           : 1;
  __REG32  icm8eh2h_rstmst        : 1;
  __REG32  icm8eh2h_rstslv        : 1;
  __REG32  icm8eh2h_tikenb        : 1;
  __REG32  expi_intout_req        : 1;
  __REG32  ic8eh2h_tiktmout       : 4;
  __REG32  ml3h2h_clksync	        : 1;
  __REG32  ml3h2h_rstmst	        : 1;
  __REG32  ml3h2h_rstslv	        : 1;
  __REG32  ml3h2h_tikenb	        : 1;
  __REG32  ml3icm9_tiktmout       : 4;
} __expi_cfg_ctr_bits;

/* PRC1-4_LOCK_CTR register */
typedef struct
{
  __REG32  lock_request           : 4;
  __REG32  lock_reset             : 4;
  __REG32                         : 9;
  __REG32  sts_loc_lock_1         : 1;
  __REG32  sts_loc_lock_2         : 1;
  __REG32  sts_loc_lock_3         : 1;
  __REG32  sts_loc_lock_4         : 1;
  __REG32  sts_loc_lock_5         : 1;
  __REG32  sts_loc_lock_6         : 1;
  __REG32  sts_loc_lock_7         : 1;
  __REG32  sts_loc_lock_8         : 1;
  __REG32  sts_loc_lock_9         : 1;
  __REG32  sts_loc_lock_10        : 1;
  __REG32  sts_loc_lock_11        : 1;
  __REG32  sts_loc_lock_12        : 1;
  __REG32  sts_loc_lock_13        : 1;
  __REG32  sts_loc_lock_14        : 1;
  __REG32  sts_loc_lock_15        : 1;
} __prc_lock_ctr_bits;

/* PRC1-4_IRQ_CTR register */
typedef struct
{
  __REG32  int2_req_prc1_1        : 1;
  __REG32  int2_req_prc1_2        : 1;
  __REG32  int3_req_prc1_1        : 1;
  __REG32  int3_req_prc1_2        : 1;
  __REG32  int4_req_prc1_1        : 1;
  __REG32  int4_req_prc1_2        : 1;
  __REG32  								        :10;
  __REG32  int1_req_prc2_1        : 1;
  __REG32  int1_req_prc2_2        : 1;
  __REG32  int1_req_prc3_1        : 1;
  __REG32  int1_req_prc3_2        : 1;
  __REG32  int1_req_prc4_1        : 1;
  __REG32  int1_req_prc4_2        : 1;
  __REG32                         :10;
} __prc_irq_ctr_bits;

/* Powerdown_CFG_CTR register */
typedef struct
{
  __REG32  wakeup_fiq_enb         : 1;
  __REG32                         :31;
} __powerdown_cfg_ctr_bits;

/* COMPSSTL_1V8_CFG/DDR_2V5_COMPENSATION register */
typedef struct
{
  __REG32  Iddq_tq                : 1;
  __REG32  en			                : 1;
  __REG32  tq			                : 1;
  __REG32  freeze                 : 1;
  __REG32  accurate               : 1;
  __REG32  sts_ok	                : 1;
  __REG32                         :10;
  __REG32  nasrc	                : 7;
  __REG32                         : 1;
  __REG32  rasrc                  : 7;
  __REG32			                    : 1;
} __compsstl_1v8_cfg_bits;

/* COMPCOR_3V3_CFG Register */
typedef struct
{
  __REG32  en			                : 1;
  __REG32  tq			                : 1;
  __REG32  freeze                 : 1;
  __REG32  accurate               : 1;
  __REG32  sts_ok	                : 1;
  __REG32                         :11;
  __REG32  nasrc	                : 7;
  __REG32                         : 1;
  __REG32  rasrc                  : 7;
  __REG32			                    : 1;
} __compcor_3v3_cfg_bits;

/* SSTLPAD_CFG_CTR Register */
typedef struct
{
  __REG32  sstl_sel								  : 1;
  __REG32  prog_b                   : 1;
  __REG32  prog_a                   : 1;
  __REG32  drive_mode_s_w           : 1;
  __REG32  pu_sel                   : 1;
  __REG32  pdn_sel                  : 1;
  __REG32  clk_pu_sel               : 1;
  __REG32  clk_pdn_sel              : 1;
  __REG32  dqs_pu_sel               : 1;
  __REG32  dqs_pdn_sel              : 1;
  __REG32  ndqs_pu_sel              : 1;
  __REG32  ndqs_pdn_sel             : 1;
  __REG32  pseudo_dif_dis           : 1;
  __REG32  				                  : 1;
  __REG32  com_ref               		: 1;
  __REG32  dram_type	              : 1;
  __REG32  swkey_ddrsel             : 4;
  __REG32  						              :11;
  __REG32  lvds_bengup_enb          : 1;
} __sstlpad_cfg_ctr_bits;

/* BIST1_CFG_CTR register */
typedef struct
{
  __REG32  rbact1_0                 : 1;
  __REG32  rbact1_1                 : 1;
  __REG32  rbact1_2                 : 1;
  __REG32  rbact1_3                 : 1;
  __REG32  rbact1_4                 : 1;
  __REG32  rbact1_5                 : 1;
  __REG32  rbact1_6                 : 1;
  __REG32  rbact1_7                 : 1;
  __REG32  rbact1_8                 : 1;
  __REG32  rbact1_9                 : 1;
  __REG32  rbact1_10                : 1;
  __REG32  rbact1_11                : 1;
  __REG32  rbact1_12                : 1;
  __REG32  rbact1_13                : 1;
  __REG32  rbact1_14                : 1;
  __REG32                           : 9;
  __REG32  bist1_iddq               : 1;
  __REG32  bist1_ret                : 1;
  __REG32  bist1_debug              : 1;
  __REG32  bist1_tm                 : 1;
  __REG32  bist1_rst                : 1;
  __REG32                           : 2;
  __REG32  bist1_res_rst            : 1;
} __bist1_cfg_ctr_bits;

/* BIST2_CFG_CTR register */
typedef struct
{
  __REG32  rbact2_0                 : 1;
  __REG32  rbact2_1                 : 1;
  __REG32  rbact2_2                 : 1;
  __REG32  rbact2_3                 : 1;
  __REG32                           :20;
  __REG32  bist2_iddq               : 1;
  __REG32  bist2_ret                : 1;
  __REG32  bist2_debug              : 1;
  __REG32  bist2_tm                 : 1;
  __REG32  bist2_rst                : 1;
  __REG32                           : 2;
  __REG32  bist2_res_rst            : 1;
} __bist2_cfg_ctr_bits;

/* BIST3_CFG_CTR register */
typedef struct
{
  __REG32  rbact3_0                 : 1;
  __REG32  rbact3_1                 : 1;
  __REG32  rbact3_2                 : 1;
  __REG32                           :21;
  __REG32  bist3_iddq               : 1;
  __REG32  bist3_ret                : 1;
  __REG32  bist3_debug              : 1;
  __REG32  bist3_tm                 : 1;
  __REG32  bist3_rst                : 1;
  __REG32                           : 2;
  __REG32  bist3_res_rst            : 1;
} __bist3_cfg_ctr_bits;

/* BIST4_CFG_CTR register */
typedef struct
{
  __REG32  rbact4_0                 : 1;
  __REG32                           :23;
  __REG32  bist4_iddq               : 1;
  __REG32  bist4_ret                : 1;
  __REG32  bist4_debug              : 1;
  __REG32  bist4_tm                 : 1;
  __REG32  bist4_rst                : 1;
  __REG32                           : 2;
  __REG32  bist4_res_rst            : 1;
} __bist4_cfg_ctr_bits;

/* BIST5_CFG_CTR Register */
typedef struct
{
  __REG32  rbact5_0                 : 1;
  __REG32                           :23;
  __REG32  bist5_iddq               : 1;
  __REG32  bist5_ret                : 1;
  __REG32  bist5_debug              : 1;
  __REG32  bist5_tm                 : 1;
  __REG32  bist5_rst                : 1;
  __REG32                           : 2;
  __REG32  bist5_res_rst            : 1;
} __bist5_cfg_ctr_bits;

/* BIST1_STS_RES register */
typedef struct
{
  __REG32  bbad1_0                  : 1;
  __REG32  bbad1_1                  : 1;
  __REG32  bbad1_2                  : 1;
  __REG32  bbad1_3                  : 1;
  __REG32  bbad1_4                  : 1;
  __REG32  bbad1_5                  : 1;
  __REG32  bbad1_6                  : 1;
  __REG32  bbad1_7                  : 1;
  __REG32  bbad1_8                  : 1;
  __REG32  bbad1_9                  : 1;
  __REG32  bbad1_10                 : 1;
  __REG32  bbad1_11                 : 1;
  __REG32  bbad1_12                 : 1;
  __REG32  bbad1_13                 : 1;
  __REG32  bbad1_14                 : 1;
  __REG32                           :16;
  __REG32  bist1_end                : 1;
} __bist1_sts_res_bits;

/* BIST2_STS_RES register */
typedef struct
{
  __REG32  bbad2_0                  : 1;
  __REG32  bbad2_1                  : 1;
  __REG32  bbad2_2                  : 1;
  __REG32  bbad2_3                  : 1;
  __REG32  bbad2_4                  : 1;
  __REG32  bbad2_5                  : 1;
  __REG32  bbad2_6                  : 1;
  __REG32  bbad2_7                  : 1;
  __REG32  bbad2_8                  : 1;
  __REG32  bbad2_9                  : 1;
  __REG32  bbad2_10                 : 1;
  __REG32  bbad2_11                 : 1;
  __REG32  bbad2_12                 : 1;
  __REG32  bbad2_13                 : 1;
  __REG32  bbad2_14                 : 1;
  __REG32  bbad2_15                 : 1;
  __REG32  bbad2_16                 : 1;
  __REG32  bbad2_17                 : 1;
  __REG32  bbad2_18                 : 1;
  __REG32                           :12;
  __REG32  bist2_end                : 1;
} __bist2_sts_res_bits;

/* BIST3_STS_RES register */
typedef struct
{
  __REG32  bbad3_0                  : 1;
  __REG32  bbad3_1                  : 1;
  __REG32  bbad3_2                  : 1;
  __REG32  bbad3_3                  : 1;
  __REG32  bbad3_4                  : 1;
  __REG32  bbad3_5                  : 1;
  __REG32  bbad3_6                  : 1;
  __REG32  bbad3_7                  : 1;
  __REG32  bbad3_8                  : 1;
  __REG32  bbad3_9                  : 1;
  __REG32  bbad3_10                 : 1;
  __REG32  bbad3_11                 : 1;
  __REG32  bbad3_12                 : 1;
  __REG32  bbad3_13                 : 1;
  __REG32  bbad3_14                 : 1;
  __REG32  bbad3_15                 : 1;
  __REG32  bbad3_16                 : 1;
  __REG32  bbad3_17                 : 1;
  __REG32  bbad3_18                 : 1;
  __REG32  bbad3_19                 : 1;
  __REG32  bbad3_20                 : 1;
  __REG32  bbad3_21                 : 1;
  __REG32  bbad3_22                 : 1;
  __REG32  bbad3_23                 : 1;
  __REG32                           : 7;
  __REG32  bist3_end                : 1;
} __bist3_sts_res_bits;

/* BIST4_STS_RES register */
typedef struct
{
  __REG32  bbad4_0                  : 1;
  __REG32  bbad4_1                  : 1;
  __REG32  bbad4_2                  : 1;
  __REG32  bbad4_3                  : 1;
  __REG32  bbad4_4                  : 1;
  __REG32  bbad4_5                  : 1;
  __REG32  bbad4_6                  : 1;
  __REG32  bbad4_7                  : 1;
  __REG32  bbad4_8                  : 1;
  __REG32  bbad4_9                  : 1;
  __REG32  bbad4_10                 : 1;
  __REG32  bbad4_11                 : 1;
  __REG32  bbad4_12                 : 1;
  __REG32  bbad4_13                 : 1;
  __REG32  bbad4_14                 : 1;
  __REG32  bbad4_15                 : 1;
  __REG32  bbad4_16                 : 1;
  __REG32  bbad4_17                 : 1;
  __REG32  bbad4_18                 : 1;
  __REG32  bbad4_19                 : 1;
  __REG32                           :11;
  __REG32  bist4_end                : 1;
} __bist4_sts_res_bits;

/* BIST5_RSLT_REG register */
typedef struct
{
  __REG32  bbad5_0                  : 1;
  __REG32  bbad5_1                  : 1;
  __REG32  bbad5_2                  : 1;
  __REG32  bbad5_3                  : 1;
  __REG32  bbad5_4                  : 1;
  __REG32  bbad5_5                  : 1;
  __REG32  bbad5_6                  : 1;
  __REG32  bbad5_7                  : 1;
  __REG32  bbad5_8                  : 1;
  __REG32  bbad5_9                  : 1;
  __REG32  bbad5_10                 : 1;
  __REG32  bbad5_11                 : 1;
  __REG32  bbad5_12                 : 1;
  __REG32  bbad5_13                 : 1;
  __REG32  bbad5_14                 : 1;
  __REG32  bbad5_15                 : 1;
  __REG32  bbad5_16                 : 1;
  __REG32  bbad5_17                 : 1;
  __REG32  bbad5_18                 : 1;
  __REG32  bbad5_19                 : 1;
  __REG32                           :11;
  __REG32  bist5_end                : 1;
} __bist5_sts_res_bits;

/* SYSERR_CFG_CTR register */
typedef struct
{
  __REG32  int_error_enb            : 1;
  __REG32  int_error_rst            : 1;
  __REG32  int_error                : 1;
  __REG32                           : 1;
  __REG32  pll_err_enb              : 1;
  __REG32                           : 1;
  __REG32  wdg_err_enb              : 1;
  __REG32  exp_err_enb              : 1;
  __REG32  usb_err_enb              : 1;
  __REG32  mem_err_enb              : 1;
  __REG32  dma_err_enb              : 1;
  __REG32                           : 1;
  __REG32  sys_pll1_err             : 1;
  __REG32  sys_pll2_err             : 1;
  __REG32  usb_pll_err              : 1;
  __REG32  mem_dll_err              : 1;
  __REG32                           : 4;
  __REG32  expi_eh2hm_err           : 1;
  __REG32  expi_eh2hs_err           : 1;
  __REG32  arm1_wdg_err             : 1;
  __REG32  arm2_wdg_err             : 1;
  __REG32  usbdv_err                : 1;
  __REG32  usbh1_err                : 1;
  __REG32  usbh2_err                : 1;
  __REG32  Mem_err                  : 1;
  __REG32  dma_err                  : 1;
  __REG32                           : 3;
} __syserr_cfg_ctr_bits;

/* SCCTRL register */
typedef struct
{
  __REG32  ModeCtrl             : 3;
  __REG32  ModeStatus           : 4;
  __REG32                       :16;
  __REG32  WDogEnOv             : 1;
  __REG32                       : 8;
} __scctrl_bits;

/* SCIMCTRL register */
typedef struct
{
  __REG32  ItMdEn               : 1;
  __REG32  ItMdCtrl             : 3;
  __REG32                       : 3;
  __REG32  InMdType             : 1;
  __REG32                       :24;
} __scimctrl_bits;

/* SCIMSTAT register */
typedef struct
{
  __REG32  ItMdStat             : 1;
  __REG32                       :31;
} __scimstat_bits;

/* SCXTALCTRL register */
typedef struct
{
  __REG32  XtalOver             : 1;
  __REG32  XtalEn               : 1;
  __REG32  XtalStat             : 1;
  __REG32  XtalTime             :16;
  __REG32                       :13;
} __scxtalctrl_bits;

/* SCPLLCTRL register */
typedef struct
{
  __REG32  PllOver              : 1;
  __REG32  PllEn                : 1;
  __REG32  PllStat              : 1;
  __REG32  PllTime              :25;
  __REG32                       : 4;
} __scpllctrl_bits;

/* WdogControl register */
typedef struct
{
  __REG32  INTEN                : 1;
  __REG32  RESEN                : 1;
  __REG32                       :30;
} __wdogcontrol_bits;

/* WdogRIS register */
typedef struct
{
  __REG32  WDOGRIS              : 1;
  __REG32                       :31;
} __wdogris_bits;

/* WdogMIS register */
typedef struct
{
  __REG32  WDOGMIS              : 1;
  __REG32                       :31;
} __wdogmis_bits;

/* Timer_control register */
typedef struct
{
  __REG16  PRESCALER            : 4;
  __REG16  MODE                 : 1;
  __REG16  ENABLE               : 1;
  __REG16  				              : 2;
  __REG16  MATCH_INT            : 1;
  __REG16                       : 7;
} __timer_control_bits;

/* TIMER_STATUS_INT_ACK register */
typedef struct
{
  __REG16  MATCH                : 1;
  __REG16                       :15;
} __timer_status_int_ack_bits;

/* MEM0_CTL register */
typedef struct
{
  __REG32  add_cmp_en        : 1;
  __REG32                    : 7;
  __REG32  ahb0_fifo_typ_reg : 2;
  __REG32                    : 6;
  __REG32  ahb1_fifo_typ_reg : 2;
  __REG32                    : 6;
  __REG32  ahb2_fifo_typ_reg : 2;
  __REG32                    : 6;
} __mem0_ctl_bits;

/* MEM1_CTL register */
typedef struct
{
  __REG32  ahb3_fifo_typ_reg : 2;
  __REG32                    : 6;
  __REG32  ahb4_fifo_typ_reg : 2;
  __REG32                    : 6;
  __REG32  ahb5_fifo_typ_reg : 2;
  __REG32                    : 6;
  __REG32  ahb6_fifo_typ_reg : 2;
  __REG32                    : 6;
} __mem1_ctl_bits;

/* MEM2_CTL register */
typedef struct
{
  __REG32  ap                : 1;
  __REG32                    : 7;
  __REG32  arefresh          : 1;
  __REG32                    : 7;
  __REG32  auto_resfreh_mod  : 1;
  __REG32                    : 7;
  __REG32  bank_split_en     : 1;
  __REG32                    : 7;
} __mem2_ctl_bits;

/* MEM3_CTL register */
typedef struct
{
  __REG32  concurrentap      : 1;
  __REG32                    : 7;
  __REG32  ddr2_ddr1_mode    : 1;
  __REG32                    : 7;
  __REG32  dll_lockreg       : 1;
  __REG32                    : 7;
  __REG32  dll_bypass_mode   : 1;
  __REG32                    : 7;
} __mem3_ctl_bits;

/* MEM4_CTL register */
typedef struct
{
  __REG32  dqs_n_en          : 1;
  __REG32                    : 7;
  __REG32  eight_bank_mode   : 1;
  __REG32                    : 7;
  __REG32  fast_write        : 1;
  __REG32                    : 7;
  __REG32  intrptapburst     : 1;
  __REG32                    : 7;
} __mem4_ctl_bits;

/* MEM5_CTL register */
typedef struct
{
  __REG32  intrptreada          : 1;
  __REG32                       : 7;
  __REG32  intrptwritea         : 1;
  __REG32                       : 7;
  __REG32  no_cmd_init          : 1;
  __REG32                       : 7;
  __REG32  odt_ad_turn_clken  	: 1;
  __REG32                       : 7;
} __mem5_ctl_bits;

/* MEM6_CTL register */
typedef struct
{
  __REG32  placement_en         : 1;
  __REG32                       : 7;
  __REG32  power_down           : 1;
  __REG32                       : 7;
  __REG32  priority_en          : 1;
  __REG32                       : 7;
  __REG32  reduce               : 1;
  __REG32                       : 7;
} __mem6_ctl_bits;

/* MEM7_CTL register */
typedef struct
{
  __REG32  reg_dimm_enable      : 1;
  __REG32                       : 7;
  __REG32  rw_same_en           : 1;
  __REG32                       : 7;
  __REG32  srefresh             : 1;
  __REG32                       : 7;
  __REG32  start                : 1;
  __REG32                       : 7;
} __mem7_ctl_bits;

/* MEM8_CTL register */
typedef struct
{
  __REG32  tras_lockout         : 1;
  __REG32                       : 7;
  __REG32  wgth_rrb_lat_ctrl    : 1;
  __REG32                       : 7;
  __REG32  writeinterp	        : 1;
  __REG32                       : 7;
  __REG32  write_modereg        : 1;
  __REG32                       : 7;
} __mem8_ctl_bits;

/* MEM9_CTL register */
typedef struct
{
  __REG32  cs_map               : 2;
  __REG32                       : 6;
  __REG32  max_cs_reg           : 2;
  __REG32                       :22;
} __mem9_ctl_bits;

/* MEM10_CTL register */
typedef struct
{
  __REG32  odt_wr_map_cs0       : 2;
  __REG32                       : 6;
  __REG32  odt_wr_map_cs1       : 2;
  __REG32                       : 6;
  __REG32  out_of_range_type    : 2;
  __REG32                       : 6;
  __REG32  rtt_0                : 2;
  __REG32                       : 6;
} __mem10_ctl_bits;

/* MEM11_CTL register */
typedef struct
{
  __REG32  rtt_pad_terminat     : 2;
  __REG32                       : 6;
  __REG32  addr_pins            : 3;
  __REG32                       : 5;
  __REG32  ahb0_prt_ordering    : 3;
  __REG32                       : 5;
  __REG32  ahb0_r_priority      : 3;
  __REG32                       : 5;
} __mem11_ctl_bits;

/* MEM12_CTL register */
typedef struct
{
  __REG32  ahb0_w_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb1_prt_ordering    : 3;
  __REG32                       : 5;
  __REG32  ahb1_r_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb1_w_priority      : 3;
  __REG32                       : 5;
} __mem12_ctl_bits;

/* MEM13_CTL register */
typedef struct
{
  __REG32  ahb2_prt_ordering    : 3;
  __REG32                       : 5;
  __REG32  ahb2_r_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb2_w_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb3_prt_ordering    : 3;
  __REG32                       : 5;
} __mem13_ctl_bits;

/* MEM14_CTL register */
typedef struct
{
  __REG32  ahb3_r_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb3_w_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb4_prt_ordering    : 3;
  __REG32                       : 5;
  __REG32  ahb4_r_priority      : 3;
  __REG32                       : 5;
} __mem14_ctl_bits;

/* MEM15_CTL register */
typedef struct
{
  __REG32  ahb4_w_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb5_prt_ordering    : 3;
  __REG32                       : 5;
  __REG32  ahb5_r_priority      : 3;
  __REG32                       : 5;
  __REG32  ahb5_w_priority      : 3;
  __REG32                       : 5;
} __mem15_ctl_bits;

/* MEM16_CTL register */
typedef struct
{
  __REG32  ahb6_prt_ordering      : 3;
  __REG32                         : 5;
  __REG32  ahb6_r_priority        : 3;
  __REG32                         : 5;
  __REG32  ahb6_w_priority 				: 3;
  __REG32                         : 5;
  __REG32  bstlen                 : 3;
  __REG32                         : 5;
} __mem16_ctl_bits;

/* MEM17_CTL register */
typedef struct
{
  __REG32  caslat						      : 3;
  __REG32                         : 5;
  __REG32  column_size		        : 3;
  __REG32                         : 5;
  __REG32  out_of_rng_src_id 		  : 3;
  __REG32                         : 5;
  __REG32  tcke                   : 3;
  __REG32                         : 5;
} __mem17_ctl_bits;

/* MEM18_CTL register */
typedef struct
{
  __REG32  temrs                  : 3;
  __REG32                         : 5;
  __REG32  tpdex                  : 3;
  __REG32                         : 5;
  __REG32  trrd                   : 3;
  __REG32                         : 5;
  __REG32  trtp                   : 3;
  __REG32                         : 5;
} __mem18_ctl_bits;

/* MEM19_CTL register */
typedef struct
{
  __REG32  twr_int                : 3;
  __REG32                         : 5;
  __REG32  twtr                   : 3;
  __REG32                         : 5;
  __REG32  wgt_rrb_wgt_shar       : 3;
  __REG32                         : 5;
  __REG32  wrlat                  : 3;
  __REG32                         : 5;
} __mem19_ctl_bits;

/* MEM20_CTL register */
typedef struct
{
  __REG32  age_count              : 6;
  __REG32                         : 2;
  __REG32  ahb0_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb0_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb0_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem20_ctl_bits;

/* MEM21_CTL register */
typedef struct
{
  __REG32  ahb0_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb0_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb0_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb0_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem21_ctl_bits;

/* MEM22_CTL register */
typedef struct
{
  __REG32  ahb0_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem22_ctl_bits;

/* MEM23_CTL register */
typedef struct
{
  __REG32  ahb1_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb1_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem23_ctl_bits;

/* MEM24_CTL register */
typedef struct
{
  __REG32  ahb1_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem24_ctl_bits;

/* MEM25_CTL register */
typedef struct
{
  __REG32  ahb2_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb2_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem25_ctl_bits;

/* MEM26_CTL register */
typedef struct
{
  __REG32  ahb2_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem26_ctl_bits;

/* MEM27_CTL register */
typedef struct
{
  __REG32  ahb3_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb3_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem27_ctl_bits;

/* MEM28_CTL register */
typedef struct
{
  __REG32  ahb3_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem28_ctl_bits;

/* MEM29_CTL register */
typedef struct
{
  __REG32  ahb4_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb4_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem29_ctl_bits;

/* MEM30_CTL register */
typedef struct
{
  __REG32  ahb4_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem30_ctl_bits;

/* MEM31_CTL register */
typedef struct
{
  __REG32  ahb5_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb5_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem31_ctl_bits;

/* MEM32_CTL register */
typedef struct
{
  __REG32  ahb5_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry0_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry1_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry2_rel_pry      : 5;
  __REG32                         : 3;
} __mem32_ctl_bits;

/* MEM33_CTL register */
typedef struct
{
  __REG32  ahb6_pry3_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry4_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry5_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  ahb6_pry6_rel_pry      : 5;
  __REG32                         : 3;
} __mem33_ctl_bits;

/* MEM34_CTL register */
typedef struct
{
  __REG32  ahb6_pry7_rel_pry      : 5;
  __REG32                         : 3;
  __REG32  aprebit                : 4;
  __REG32                         : 4;
  __REG32  caslat_lin             : 5;
  __REG32                         : 3;
  __REG32  caslat_lin_gate        : 5;
  __REG32                         : 3;
} __mem34_ctl_bits;

/* MEM35_CTL register */
typedef struct
{
  __REG32  comd_age_count		      : 6;
  __REG32                         : 2;
  __REG32  initaref               : 4;
  __REG32                         : 4;
  __REG32  max_col_reg            : 4;
  __REG32                         : 4;
  __REG32  max_row_reg            : 4;
  __REG32                         : 4;
} __mem35_ctl_bits;

/* MEM36_CTL register */
typedef struct
{
  __REG32  q_fullness             : 4;
  __REG32                         : 4;
  __REG32  tdal                   : 4;
  __REG32                         : 4;
  __REG32  trp                    : 4;
  __REG32                         : 4;
  __REG32  wrr_prm_val_err		    : 4;
  __REG32                         : 4;
} __mem36_ctl_bits;

/* MEM37_CTL register */
typedef struct
{
  __REG32  int_ack                : 6;
  __REG32                         : 2;
  __REG32  ocd_adj_pdn_cs 	      : 5;
  __REG32                         : 3;
  __REG32  ocd_adj_pup_cs			    : 5;
  __REG32                         : 3;
  __REG32  tfaw                   : 5;
  __REG32                         : 3;
} __mem37_ctl_bits;

/* MEM38_CTL register */
typedef struct
{
  __REG32  tmrd                   : 5;
  __REG32                         : 3;
  __REG32  trc                    : 5;
  __REG32                         : 3;
  __REG32  int_mask               : 7;
  __REG32                         : 1;
  __REG32  int_status             : 7;
  __REG32                         : 1;
} __mem38_ctl_bits;

/* MEM39_CTL register */
typedef struct
{
  __REG32  dll_dqs_delay_0        : 7;
  __REG32                         : 1;
  __REG32  dll_dqs_delay_1        : 7;
  __REG32                         :17;
} __mem39_ctl_bits;

/* MEM40_CTL register */
typedef struct
{
  __REG32                         :24;
  __REG32  dqs_out_shift          : 7;
  __REG32                         : 1;
} __mem40_ctl_bits;

/* MEM41_CTL register */
typedef struct
{
  __REG32                         : 8;
  __REG32  dqs_out_shift          : 7;
  __REG32                         : 1;
  __REG32  wr_dqs_shift	          : 7;
  __REG32                         : 9;
} __mem41_ctl_bits;

/* MEM42_CTL register */
typedef struct
{
  __REG32  tcpd		                : 8;
  __REG32  tras_min               : 8;
  __REG32  trcd_int               : 8;
  __REG32  trfc                   : 8;
} __mem42_ctl_bits;

/* MEM43_CTL register */
typedef struct
{
  __REG32  ahb0_pry_relax			    :11;
  __REG32                         : 5;
  __REG32  ahb1_pry_relax			    :11;
  __REG32  		                    : 5;
} __mem43_ctl_bits;

/* MEM44_CTL register */
typedef struct
{
  __REG32  ahb2_pry_relax			    :11;
  __REG32                         : 5;
  __REG32  ahb3_pry_relax			    :11;
  __REG32  			                  : 5;
} __mem44_ctl_bits;

/* MEM45_CTL register */
typedef struct
{
  __REG32  ahb4_pry_relax			    :11;
  __REG32                         : 5;
  __REG32  ahb5_pry_relax			    :11;
  __REG32                         : 5;
} __mem45_ctl_bits;

/* MEM46_CTL register */
typedef struct
{
  __REG32  ahb6_pry_relax			    :11;
  __REG32                         : 5;
  __REG32  OUT_OF_RANGE_LENGTH    :11;
  __REG32                         : 5;
} __mem46_ctl_bits;

/* MEM47_CTL register */
typedef struct
{
  __REG32  ahb0_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb0_wrcnt             :11;
  __REG32                         : 5;
} __mem47_ctl_bits;

/* MEM48_CTL register */
typedef struct
{
  __REG32  ahb1_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb1_wrcnt             :11;
  __REG32                         : 5;
} __mem48_ctl_bits;

/* MEM49_CTL register */
typedef struct
{
  __REG32  ahb2_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb2_wrcnt             :11;
  __REG32                         : 5;
} __mem49_ctl_bits;

/* MEM50_CTL register */
typedef struct
{
  __REG32  ahb3_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb3_wrcnt             :11;
  __REG32                         : 5;
} __mem50_ctl_bits;

/* MEM51_CTL register */
typedef struct
{
  __REG32  ahb4_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb4_wrcnt             :11;
  __REG32                         : 5;
} __mem51_ctl_bits;

/* MEM52_CTL register */
typedef struct
{
  __REG32  ahb5_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb5_wrcnt             :11;
  __REG32                         : 5;
} __mem52_ctl_bits;

/* MEM53_CTL register */
typedef struct
{
  __REG32  ahb6_rdcnt             :11;
  __REG32                         : 5;
  __REG32  ahb6_wrcnt             :11;
  __REG32                         : 5;
} __mem53_ctl_bits;

/* MEM54_CTL register */
typedef struct
{
  __REG32  tref			              :14;
  __REG32                         : 2;
  __REG32  emrs2_data	            :15;
  __REG32                         : 1;
} __mem54_ctl_bits;

/* MEM55_CTL register */
typedef struct
{
  __REG32  emrs3_data             :15;
  __REG32                         : 1;
  __REG32  emrs_data             	:15;
  __REG32                         : 1;
} __mem55_ctl_bits;

/* MEM56_CTL register */
typedef struct
{
  __REG32  TDLL                   :16;
  __REG32  TRAS_MAX               :16;
} __mem56_ctl_bits;

/* MEM57_CTL register */
typedef struct
{
  __REG32  txsnr                  :16;
  __REG32  txsr                   :16;
} __mem57_ctl_bits;

/* MEM58_CTL register */
typedef struct
{
  __REG32  version                :16;
  __REG32                         :16;
} __mem58_ctl_bits;

/* MEM59_CTL register */
typedef struct
{
  __REG32  tinit                  :24;
  __REG32                         : 8;
} __mem59_ctl_bits;

/* MEM61_CTL register */
typedef struct
{
  __REG32  out_rng_addr			      : 2;
  __REG32                         : 6;
  __REG32  prt_addr_prot_enb      : 1;
  __REG32                         : 7;
  __REG32  ahb0_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb0_rng_typ1		      : 2;
  __REG32                         : 6;
} __mem61_ctl_bits;

/* MEM62_CTL register */
typedef struct
{
  __REG32  ahb1_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb1_rng_typ1		      : 2;
  __REG32                         : 6;
  __REG32  ahb2_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb2_rng_typ1		      : 2;
  __REG32                         : 6;
} __mem62_ctl_bits;

/* MEM63_CTL register */
typedef struct
{
  __REG32  ahb3_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb3_rng_typ1		      : 2;
  __REG32                         : 6;
  __REG32  ahb4_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb4_rng_typ1		      : 2;
  __REG32                         : 6;
} __mem63_ctl_bits;

/* MEM64_CTL register */
typedef struct
{
  __REG32  ahb5_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb5_rng_typ1		      : 2;
  __REG32                         : 6;
  __REG32  ahb6_rng_typ0		      : 2;
  __REG32                         : 6;
  __REG32  ahb6_rng_typ1		      : 2;
  __REG32                         : 6;
} __mem64_ctl_bits;

/* MEM65_CTL register */
typedef struct
{
  __REG32  port_err_prt_num 			: 3;
  __REG32                         : 5;
  __REG32  port_err_type					: 4;
  __REG32                         : 4;
  __REG32  dll_dqs_dly_byps0			: 9;
  __REG32                         : 7;
} __mem65_ctl_bits;

/* MEM66_CTL register */
typedef struct
{
  __REG32  dll_dqs_dly_byps1		  : 9;
  __REG32                         : 7;
  __REG32  dll_increment          : 9;
  __REG32                         : 7;
} __mem66_ctl_bits;

/* MEM67_CTL register */
typedef struct
{
  __REG32  dll_lock               : 9;
  __REG32                         : 7;
  __REG32  dll_start_point        : 9;
  __REG32                         : 7;
} __mem67_ctl_bits;

/* MEM68_CTL register */
typedef struct
{
  __REG32  dqs_out_shft_byps		  : 9;
  __REG32                         : 7;
  __REG32  wr_dqs_shft_byps		    : 9;
  __REG32                         : 7;
} __mem68_ctl_bits;

/* MEM69_CTL register */
typedef struct
{
  __REG32  ahb0_end_addr0				  :22;
  __REG32                         :10;
} __mem69_ctl_bits;

/* MEM70_CTL register */
typedef struct
{
  __REG32  ahb0_end_addr1				  :22;
  __REG32                         :10;
} __mem70_ctl_bits;

/* MEM71_CTL register */
typedef struct
{
  __REG32  ahb0_str_addr0				  :22;
  __REG32                         :10;
} __mem71_ctl_bits;

/* MEM72_CTL register */
typedef struct
{
  __REG32  ahb0_str_addr1				  :22;
  __REG32                         :10;
} __mem72_ctl_bits;

/* MEM73_CTL register */
typedef struct
{
  __REG32  ahb1_end_addr0				  :22;
  __REG32                         :10;
} __mem73_ctl_bits;

/* MEM74_CTL register */
typedef struct
{
  __REG32  ahb1_end_addr1				  :22;
  __REG32                         :10;
} __mem74_ctl_bits;

/* MEM75_CTL register */
typedef struct
{
  __REG32  ahb1_str_addr0				  :22;
  __REG32                         :10;
} __mem75_ctl_bits;

/* MEM76_CTL register */
typedef struct
{
  __REG32  ahb1_str_addr1				  :22;
  __REG32                         :10;
} __mem76_ctl_bits;

/* MEM77_CTL register */
typedef struct
{
  __REG32  ahb2_end_addr0				  :22;
  __REG32                         :10;
} __mem77_ctl_bits;

/* MEM78_CTL register */
typedef struct
{
  __REG32  ahb2_end_addr1				  :22;
  __REG32                         :10;
} __mem78_ctl_bits;

/* MEM79_CTL register */
typedef struct
{
  __REG32  ahb2_str_addr0				  :22;
  __REG32                         :10;
} __mem79_ctl_bits;

/* MEM80_CTL register */
typedef struct
{
  __REG32  ahb2_str_addr1				  :22;
  __REG32                         :10;
} __mem80_ctl_bits;

/* MEM81_CTL register */
typedef struct
{
  __REG32  ahb3_end_addr0				  :22;
  __REG32                         :10;
} __mem81_ctl_bits;

/* MEM82_CTL register */
typedef struct
{
  __REG32  ahb3_end_addr1				  :22;
  __REG32                         :10;
} __mem82_ctl_bits;

/* MEM83_CTL register */
typedef struct
{
  __REG32  ahb3_str_addr0				  :22;
  __REG32                         :10;
} __mem83_ctl_bits;

/* MEM84_CTL register */
typedef struct
{
  __REG32  ahb3_str_addr1				  :22;
  __REG32                         :10;
} __mem84_ctl_bits;

/* MEM85_CTL register */
typedef struct
{
  __REG32  ahb4_end_addr0				  :22;
  __REG32                         :10;
} __mem85_ctl_bits;

/* MEM86_CTL register */
typedef struct
{
  __REG32  ahb4_end_addr1				  :22;
  __REG32                         :10;
} __mem86_ctl_bits;

/* MEM87_CTL register */
typedef struct
{
  __REG32  ahb4_str_addr0				  :22;
  __REG32                         :10;
} __mem87_ctl_bits;

/* MEM88_CTL register */
typedef struct
{
  __REG32  ahb4_str_addr1				  :22;
  __REG32                         :10;
} __mem88_ctl_bits;

/* MEM89_CTL register */
typedef struct
{
  __REG32  ahb5_end_addr0				  :22;
  __REG32                         :10;
} __mem89_ctl_bits;

/* MEM90_CTL register */
typedef struct
{
  __REG32  ahb5_end_addr1				  :22;
  __REG32                         :10;
} __mem90_ctl_bits;

/* MEM91_CTL register */
typedef struct
{
  __REG32  ahb5_str_addr0				  :22;
  __REG32                         :10;
} __mem91_ctl_bits;

/* MEM92_CTL register */
typedef struct
{
  __REG32  ahb5_str_addr1				  :22;
  __REG32                         :10;
} __mem92_ctl_bits;

/* MEM93_CTL Register */
typedef struct
{
  __REG32  ahb6_end_addr0				  :22;
  __REG32                         :10;
} __mem93_ctl_bits;

/* MEM94_CTL Register */
typedef struct
{
  __REG32  ahb6_end_addr1				  :22;
  __REG32                         :10;
} __mem94_ctl_bits;

/* MEM95_CTL Register */
typedef struct
{
  __REG32  ahb6_str_addr0				  :22;
  __REG32                         :10;
} __mem95_ctl_bits;

/* MEM96_CTL Register */
typedef struct
{
  __REG32  ahb6_str_addr1				  :22;
  __REG32                         :10;
} __mem96_ctl_bits;

/* SMI_CR1 register */
typedef struct
{
  __REG32  BE                   : 4;
  __REG32  TCS                  : 4;
  __REG32  PRESC                : 7;
  __REG32  FAST                 : 1;
  __REG32  HOLD                 : 8;
  __REG32  ADD_LENGTH           : 4;
  __REG32  SW                   : 1;
  __REG32  WBM                  : 1;
  __REG32                       : 2;
} __smi_cr1_bits;

/* SMI_CR2 register */
typedef struct
{
  __REG32  TRA_LENGTH           : 3;
  __REG32                       : 1;
  __REG32  REC_LENGTH           : 3;
  __REG32  SEND                 : 1;
  __REG32  TFIE                 : 1;
  __REG32  WCIE                 : 1;
  __REG32  RSR                  : 1;
  __REG32  WEN                  : 1;
  __REG32  BS                   : 2;
  __REG32                       :18;
} __smi_cr2_bits;

/* SMI_SR register */
typedef struct
{
  __REG32  SMSR                 : 8;
  __REG32  TFF                  : 1;
  __REG32  WCF                  : 1;
  __REG32  ERF2                 : 1;
  __REG32  ERF1                 : 1;
  __REG32  WM                   : 4;
  __REG32                       :16;
} __smi_sr_bits;

/* GenMemCtrl_PC(i) registers */
typedef struct
{
  __REG32  Reset          : 1;
  __REG32  Wait_on        : 1;
  __REG32  Enable         : 1;
  __REG32  Dev_type       : 1;
  __REG32  Dev_width      : 2;
  __REG32  Eccen          : 1;
  __REG32  Eccplen        : 1;
  __REG32                 : 1;
  __REG32  tclr           : 4;
  __REG32  tar            : 4;
  __REG32                 :15;
} __genmemctrl_pc_bits;

/* GenMemCtrl_Comm0/GenMemCtrl_Attrib0/GenMemCtrl_I/O0 */
typedef struct
{
  __REG32  Tset           : 8;
  __REG32  Twait          : 8;
  __REG32  Thold          : 8;
  __REG32  Thiz           : 8;
} __genmemctrl_comm_bits;

/* GenMemCtrl_ECCr0 registers */
typedef struct
{
  __REG32  ecc1           : 8;
  __REG32  ecc2           : 8;
  __REG32  ecc3           : 8;
  __REG32                 : 8;
} __genmemctrl_eccr_bits;

/* Bus Mode Register (Register0, DMA) */
typedef struct
{
  __REG32  SWR	          : 1;
  __REG32  DA 	          : 1;
  __REG32  DSL	          : 5;
  __REG32  			          : 1;
  __REG32  PBL	          : 6;
  __REG32  PR 	          : 2;
  __REG32  FB 	          : 1;
  __REG32                 :15;
} __gmac_dma_r0_bits;

/* Status Register (Register5, DMA) */
typedef struct
{
  __REG32  TI  	          : 1;
  __REG32  TPS 	          : 1;
  __REG32  TU 	          : 1;
  __REG32  TJT 	          : 1;
  __REG32  OVF 	          : 1;
  __REG32  UNF 	          : 1;
  __REG32  RI 	          : 1;
  __REG32  RU 	          : 1;
  __REG32  RPS 	          : 1;
  __REG32  RWT 	          : 1;
  __REG32  ETI 	          : 1;
  __REG32  		 	          : 2;
  __REG32  FBI	          : 1;
  __REG32  ERI	          : 1;
  __REG32  AIS	          : 1;
  __REG32  NIS	          : 1;
  __REG32  RS 	          : 3;
  __REG32  TS 	          : 3;
  __REG32  EB		          : 3;
  __REG32  			          : 1;
  __REG32  GMI	          : 1;
  __REG32  GPI	          : 1;
  __REG32  			          : 3;
} __gmac_dma_r5_bits;

/* Operation Mode Register (Register6, DMA) */
typedef struct
{
  __REG32  	  	          : 1;
  __REG32  SR 	          : 1;
  __REG32  OSF 	          : 1;
  __REG32  RTC 	          : 2;
  __REG32  		 	          : 1;
  __REG32  FUF 	          : 1;
  __REG32  FEF 	          : 1;
  __REG32  EFC 	          : 1;
  __REG32  RFA 	          : 2;
  __REG32  RFD 	          : 2;
  __REG32  ST 	          : 1;
  __REG32  TTC	          : 3;
  __REG32  			          : 3;
  __REG32  FTF	          : 1;
  __REG32  SF 	          : 1;
  __REG32  			          :10;
} __gmac_dma_r6_bits;

/* Interrupt Enable Register (Register7, DMA) */
typedef struct
{
  __REG32  TIE            : 1;
  __REG32  TSE            : 1;
  __REG32  TUE            : 1;
  __REG32  TJE            : 1;
  __REG32  OVE            : 1;
  __REG32  UNE 	          : 1;
  __REG32  RIE 	          : 1;
  __REG32  RUE 	          : 1;
  __REG32  RSE 	          : 1;
  __REG32  RWE 	          : 1;
  __REG32  ETE 	          : 1;
  __REG32  		 	          : 2;
  __REG32  FBE 	          : 1;
  __REG32  ERE 	          : 1;
  __REG32  AIE 	          : 1;
  __REG32  NIE	          : 1;
  __REG32  			          :15;
} __gmac_dma_r7_bits;

/* Missed Frame and Buffer Overflow Counter Register (Register8, DMA) */
typedef struct
{
  __REG32  MFC            :16;
  __REG32  OMFC           : 1;
  __REG32  MFA            :11;
  __REG32  OMFA           : 1;
  __REG32  		            : 3;
} __gmac_dma_r8_bits;

/* MAC Configuration Register (Register0, GMAC) */
typedef struct
{
  __REG32  		            : 2;
  __REG32  RE	            : 1;
  __REG32  TE	            : 1;
  __REG32  DC	            : 1;
  __REG32  BL	            : 2;
  __REG32  ACS            : 1;
  __REG32  		            : 1;
  __REG32  DR	            : 1;
  __REG32  IPC            : 1;
  __REG32  DM	            : 1;
  __REG32  LM	            : 1;
  __REG32  DO             : 1;
  __REG32  FES            : 1;
  __REG32  PS             : 1;
  __REG32  		            : 1;
  __REG32  IFG            : 3;
  __REG32  JE             : 1;
  __REG32  BE             : 1;
  __REG32  JD             : 1;
  __REG32  WD             : 1;
  __REG32  		            : 8;
} __gmac_r0_bits;

/* MAC Frame Filter Register (Register1, GMAC) */
typedef struct
{
  __REG32  PR	            : 1;
  __REG32  HUC            : 1;
  __REG32  HMC            : 1;
  __REG32  DAIF           : 1;
  __REG32  PM	            : 1;
  __REG32  DBF            : 1;
  __REG32  PCF            : 2;
  __REG32  SAIF           : 1;
  __REG32  SAF            : 1;
  __REG32  		            :21;
  __REG32  RA             : 1;
} __gmac_r1_bits;

/* GMII Address Register (Register4, GMAC) */
typedef struct
{
  __REG32  GB	            : 1;
  __REG32  GW             : 1;
  __REG32  CR	            : 3;
  __REG32  	              : 1;
  __REG32  GR	            : 5;
  __REG32  PA	            : 5;
  __REG32  		            :16;
} __gmac_r4_bits;

/* GMII Data Register (Register5, GMAC) */
typedef struct
{
  __REG32  GD	            :16;
  __REG32  		            :16;
} __gmac_r5_bits;

/* Flow Control Register (Register6, GMAC) */
typedef struct
{
  __REG32  FCB_BPA	      : 1;
  __REG32  TFE			      : 1;
  __REG32  RFE			      : 1;
  __REG32  UP				      : 1;
  __REG32  PLT			      : 2;
  __REG32  		            :10;
  __REG32  PT				      :16;
} __gmac_r6_bits;

/* VLAN Tag Register (Register7, GMAC) */
typedef struct
{
  __REG32  VL				      :16;
  __REG32  		            :16;
} __gmac_r7_bits;

/* PMT Control and Status Register (Register11, GMAC) */
typedef struct
{
  __REG32  PD				      : 1;
  __REG32  MPE			      : 1;
  __REG32  WUFE			      : 1;
  __REG32  		            : 2;
  __REG32  MPR			      : 1;
  __REG32  WUFR			      : 1;
  __REG32  		            : 2;
  __REG32  GU 			      : 1;
  __REG32  		            :21;
  __REG32  WUFFRPR	      : 1;
} __gmac_r11_bits;

/* Interrupt Status Register (Register14, GMAC) */
typedef struct
{
  __REG32  					      : 3;
  __REG32  PMTIS		      : 1;
  __REG32  MMCIS		      : 1;
  __REG32  		            :27;
} __gmac_r14_bits;

/* Interrupt Mask Register (Register 15, GMAC) */
typedef struct
{
  __REG32  					      : 3;
  __REG32  PMTIN		      : 1;
  __REG32  		            :28;
} __gmac_r15_bits;

/* MAC Address1 High Register (Register18, GMAC) */
typedef struct
{
  __REG32  A				      :16;
  __REG32  		            : 8;
  __REG32  MBC  		      : 6;
  __REG32  SA	  		      : 1;
  __REG32  AE	  		      : 1;
} __gmac_r18_bits;

/* AN Control Register (Register48, GMAC) */
typedef struct
{
  __REG32  		            : 9;
  __REG32  RAN			      : 1;
  __REG32  		            : 2;
  __REG32  ANE  		      : 1;
  __REG32  		            :19;
} __gmac_r48_bits;

/* AN Status Register (Register49, GMAC) */
typedef struct
{
  __REG32  		            : 2;
  __REG32  LS 			      : 1;
  __REG32  ANA 			      : 1;
  __REG32  		            : 1;
  __REG32  ANC  		      : 1;
  __REG32  		            :26;
} __gmac_r49_bits;

/* AN Advertisement Register (Register50, GMAC) */
typedef struct
{
  __REG32  		            : 5;
  __REG32  FD 			      : 1;
  __REG32  HD 			      : 1;
  __REG32  PSE 			      : 2;
  __REG32  		            : 3;
  __REG32  RFE  		      : 2;
  __REG32  		            : 1;
  __REG32  NP   		      : 1;
  __REG32  		            :16;
} __gmac_r50_bits;

/* AN Link Partner Ability Register (Register51, GMAC) */
typedef struct
{
  __REG32  		            : 5;
  __REG32  FD 			      : 1;
  __REG32  HD 			      : 1;
  __REG32  PSE 			      : 2;
  __REG32  		            : 3;
  __REG32  RFE  		      : 2;
  __REG32  ACK            : 1;
  __REG32  NP   		      : 1;
  __REG32  		            :16;
} __gmac_r51_bits;

/* AN Expansion Register (Register52, GMAC) */
typedef struct
{
  __REG32  		            : 1;
  __REG32  NPR			      : 1;
  __REG32  NPA 			      : 1;
  __REG32  		            :29;
} __gmac_r52_bits;

/* MMC Control Register (Register64) */
typedef struct
{
  __REG32  CR             : 1;
  __REG32  CSR			      : 1;
  __REG32  ROR 			      : 1;
  __REG32  		            :29;
} __gmac_r64_bits;

/* MMC Receive Interrupt Register (Register65) */
typedef struct
{
  __REG32  rxframecount_gb			: 1;
  __REG32  rxoctectcount_gb			: 1;
  __REG32  rxoctectcount_g			: 1;
  __REG32  rxbroadcastframes_g	: 1;
  __REG32  rxmulticastframes_g	: 1;
  __REG32  rxcrcerror						: 1;
  __REG32  rxalignmenterror			: 1;
  __REG32  rxrunterror					: 1;
  __REG32  rxjabbererror				: 1;
  __REG32  rxundersize_g				: 1;
  __REG32  rxoversize_g					: 1;
  __REG32  rx64octects_gb				: 1;
  __REG32  rx64to127octects_gb	: 1;
  __REG32  rx128to255octects_gb	: 1;
  __REG32  rx216to511octects_gb	: 1;
  __REG32  rx512to1023octects_gb: 1;
  __REG32  rx1024tomaxoctects_gb: 1;
  __REG32  rxunicastframes_gb		: 1;
  __REG32  rxlengtherror				: 1;
  __REG32  rxoutofrangetype			: 1;
  __REG32  rxpauseframes				: 1;
  __REG32  rxfifooverflow				: 1;
  __REG32  rxvlanframes_gb			: 1;
  __REG32  rxwatchdog						: 1;
  __REG32  											: 8;
} __gmac_r65_bits;

/* MMC Transmit Interrupt Register (Register66) */
typedef struct
{
  __REG32  txoctectcount_gb			: 1;
  __REG32  txframecount_gb			: 1;
  __REG32  txbroadcastframes_g	: 1;
  __REG32  txmulticastframes_g	: 1;
  __REG32  tx64octects_gb				: 1;
  __REG32  tx65to127octects_gb	: 1;
  __REG32  tx128to255octects_gb	: 1;
  __REG32  tx256to511octects_gb	: 1;
  __REG32  tx512to1023octects_gb: 1;
  __REG32  tx1024tomaxoctects_gb: 1;
  __REG32  txunicastframes_gb		: 1;
  __REG32  txmulticastframes_gb	: 1;
  __REG32  txbroadcastframes_gb	: 1;
  __REG32  txunderflowerror			: 1;
  __REG32  txsinglecol_g				: 1;
  __REG32  txmulticol_g					: 1;
  __REG32  txdeferred						: 1;
  __REG32  txlatecol						: 1;
  __REG32  txexesscol						: 1;
  __REG32  txcarriererror				: 1;
  __REG32  txoctectcount_g			: 1;
  __REG32  txframecount_g				: 1;
  __REG32  txoexcessdef					: 1;
  __REG32  txpauseframes				: 1;
  __REG32  txvlanframes_g				: 1;
  __REG32  											: 7;
} __gmac_r66_bits;

/* HCCAPBASE register */
typedef struct
{
  __REG32  CAPLENGTH            : 8;
  __REG32                       : 8;
  __REG32  HCIVERSION           :16;
} __hccapbase_bits;

/* HCSPARAMS register */
typedef struct
{
  __REG32  N_PORTS              : 4;
  __REG32  PPC                  : 1;
  __REG32                       : 2;
  __REG32  PRR                  : 1;
  __REG32  N_PCC                : 4;
  __REG32  N_CC                 : 4;
  __REG32  P_INDICATOR          : 1;
  __REG32                       : 3;
  __REG32  DPN                  : 4;
  __REG32                       : 8;
} __hcsparams_bits;

/* HCCPARAMS register */
typedef struct
{
  __REG32  _64BAC               : 1;
  __REG32  PFLF                 : 1;
  __REG32  ASPC                 : 1;
  __REG32                       : 1;
  __REG32  IST                  : 4;
  __REG32  EECP                 : 8;
  __REG32                       :16;
} __hccparams_bits;

/* USBCMD register */
typedef struct
{
  __REG32  RS                   : 1;
  __REG32  HCRESET              : 1;
  __REG32  FLS                  : 2;
  __REG32  PSE                  : 1;
  __REG32  ASE                  : 1;
  __REG32  IAAD                 : 1;
  __REG32  LHCR                 : 1;
  __REG32  ASPMC                : 2;
  __REG32                       : 1;
  __REG32  ASPME                : 1;
  __REG32                       : 4;
  __REG32  ITC                  : 8;
  __REG32                       : 8;
} __hcusbcmd_bits;

/* USBSTS register */
typedef struct
{
  __REG32  USBINT               : 1;
  __REG32  USBERRINT            : 1;
  __REG32  PCD                  : 1;
  __REG32  FLR                  : 1;
  __REG32  HSE                  : 1;
  __REG32  IAA                  : 1;
  __REG32                       : 6;
  __REG32  HH                   : 1;
  __REG32  R                    : 1;
  __REG32  PSS                  : 1;
  __REG32  ASS                  : 1;
  __REG32                       :16;
} __hcusbsts_bits;

/* USBINTR register */
typedef struct
{
  __REG32  IE                   : 1;
  __REG32  EIE                  : 1;
  __REG32  PCIE                 : 1;
  __REG32  FLRE                 : 1;
  __REG32  HSEE                 : 1;
  __REG32  IAAE                 : 1;
  __REG32                       :26;
} __hcusbintr_bits;

/* FRINDEX register */
typedef struct
{
  __REG32  FRAME                :14;
  __REG32                       :18;
} __hcfrindex_bits;

/* CONFIGFLAG register */
typedef struct
{
  __REG32  CF                   : 1;
  __REG32                       :31;
} __hcconfigflag_bits;

/* PORTSC registers */
typedef struct
{
  __REG32  CCS                  : 1;
  __REG32  CSC                  : 1;
  __REG32  PEN                  : 1;
  __REG32  PEDC                 : 1;
  __REG32  OcA                  : 1;
  __REG32  OcC                  : 1;
  __REG32  FPR                  : 1;
  __REG32  S                    : 1;
  __REG32  PR                   : 1;
  __REG32                       : 1;
  __REG32  LS                   : 2;
  __REG32  PP                   : 1;
  __REG32  PO                   : 1;
  __REG32  PIC                  : 2;
  __REG32  PTC                  : 4;
  __REG32  WKCNNT_E             : 1;
  __REG32  WKDSCNNT_E           : 1;
  __REG32  WKOC_E               : 1;
  __REG32                       : 9;
} __hcportsc_bits;

/* INSNREG00 register */
typedef struct
{
  __REG32  IN	                  : 1;
  __REG32  PMBV                 :12;
  __REG32                       :19;
} __hcinsnreg00_bits;

/* INSNREG01 register */
typedef struct
{
  __REG32  IN                   :16;
  __REG32  OUT                  :16;
} __hcinsnreg01_bits;

/* INSNREG02 register */
typedef struct
{
  __REG32  DEEP                 :12;
  __REG32                       :20;
} __hcinsnreg02_bits;

/* INSNREG03 register */
typedef struct
{
  __REG32  BMT                  : 1;
  __REG32                       :31;
} __hcinsnreg03_bits;

/* INSNREG05 register */
typedef struct
{
  __REG32  VStatus              : 8;
  __REG32  VControl             : 4;
  __REG32  VControlLoadM        : 1;
  __REG32  VPort                : 4;
  __REG32  VBusy                : 1;
  __REG32                       :14;
} __hcinsnreg05_bits;

/* HcRevision Register */
typedef struct {
  __REG32 REV               : 8;
  __REG32                   :24;
} __hcrevision_bits;

/* HcControl Register */
typedef struct {
  __REG32 CBSR              : 2;
  __REG32 PLE               : 1;
  __REG32 IE                : 1;
  __REG32 CLE               : 1;
  __REG32 BLE               : 1;
  __REG32 HCFS              : 2;
  __REG32 IR                : 1;
  __REG32 RWC               : 1;
  __REG32 RWE               : 1;
  __REG32                   :21;
} __hccontrol_bits;

/* HcCommandStatus Register */
typedef struct {
  __REG32 HCR               : 1;
  __REG32 CLF               : 1;
  __REG32 BLF               : 1;
  __REG32 OCR               : 1;
  __REG32                   :12;
  __REG32 SOC               : 2;
  __REG32                   :14;
} __hccommandstatus_bits;

/* HcInterruptStatus Register */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32                   : 1;
} __hcinterruptstatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32 MIE               : 1;
} __hcinterruptenable_bits;

/* HcHCCA Register */
typedef struct {
  __REG32                   : 8;
  __REG32 HCCA              :24;
} __hchcca_bits;

/* HcPeriodCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 PCED              :28;
} __hcperiodcurrented_bits;

/* HcControlHeadED Registerr */
typedef struct {
  __REG32                   : 4;
  __REG32 CHED              :28;
} __hccontrolheaded_bits;

/* HcControlCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 CCED              :28;
} __hccontrolcurrented_bits;

/* HcBulkHeadED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BHED              :28;
} __hcbulkheaded_bits;

/* HcBulkCurrentED Register */
typedef struct {
  __REG32                   : 4;
  __REG32 BCED              :28;
} __hcbulkcurrented_bits;

/* HcDoneHead Register */
typedef struct {
  __REG32                   : 4;
  __REG32 DH                :28;
} __hcdonehead_bits;

/* HcFmInterval Register */
typedef struct {
  __REG32 FI                :14;
  __REG32                   : 2;
  __REG32 FSMPS             :15;
  __REG32 FIT               : 1;
} __hcfminterval_bits;

/* HcFmRemaining Register */
typedef struct {
  __REG32 FR                :14;
  __REG32                   :17;
  __REG32 FRT               : 1;
} __hcfmremaining_bits;

/* HcFmNumber Register */
typedef struct {
  __REG32 FN                :16;
  __REG32                   :16;
} __hcfmnumber_bits;

/* HcPeriodicStart Register */
typedef struct {
  __REG32 PS                :14;
  __REG32                   :18;
} __hcperiodicstart_bits;

/* HcLSThreshold Register */
typedef struct {
  __REG32 LST               :12;
  __REG32                   :20;
} __hclsthreshold_bits;

/* HcRhDescriptorA Register */
typedef struct {
  __REG32 NDP               : 8;
  __REG32 PSM               : 1;
  __REG32 NPS               : 1;
  __REG32 DT                : 1;
  __REG32 OCPM              : 1;
  __REG32 NOCP              : 1;
  __REG32                   :11;
  __REG32 POTPGT            : 8;
} __hcrhdescriptora_bits;

/* HcRhDescriptorB Register */
typedef struct {
  __REG32 DR                :16;
  __REG32 PPCM              :16;
} __hcrhdescriptorb_bits;

/* HcRhStatus Register */
typedef struct {
  __REG32 LPS               : 1;
  __REG32 OCI               : 1;
  __REG32                   :13;
  __REG32 DRWE              : 1;
  __REG32 LPSC              : 1;
  __REG32 OCIC              : 1;
  __REG32                   :13;
  __REG32 CRWE              : 1;
} __hcrhstatus_bits;

/* HcRhPortStatus[1] Register */
typedef struct {
  __REG32 CCS               : 1;
  __REG32 PES               : 1;
  __REG32 PSS               : 1;
  __REG32 POCI              : 1;
  __REG32 PRS               : 1;
  __REG32                   : 3;
  __REG32 PPS               : 1;
  __REG32 LSDA              : 1;
  __REG32                   : 6;
  __REG32 CSC               : 1;
  __REG32 PESC              : 1;
  __REG32 PSSC              : 1;
  __REG32 OCIC              : 1;
  __REG32 PRSC              : 1;
  __REG32                   :11;
} __hcrhportstatus_bits;

/* Device configuration register */
typedef struct {
  __REG32 SPD               : 2;
  __REG32 RWKP              : 1;
  __REG32 SP                : 1;
  __REG32 SS                : 1;
  __REG32 PI                : 1;
  __REG32 DIR               : 1;
  __REG32 STATUS            : 1;
  __REG32 STATUS_1          : 1;
  __REG32 PHY_ERROR_DETECT  : 1;
  __REG32 FS_TIMEOUT_CALIB  : 3;
  __REG32 HS_TIMEOUT_CALIB  : 3;
  __REG32 HALT_STATUS       : 1;
  __REG32 CSR_PRG           : 1;
  __REG32 SET_DESC          : 1;
  __REG32                   :13;
} __uddcfg_bits;

/* Device configuration register */
typedef struct {
  __REG32 RES               : 1;
  __REG32                   : 1;
  __REG32 RDE               : 1;
  __REG32 TDE               : 1;
  __REG32 DU                : 1;
  __REG32 BE                : 1;
  __REG32 BF                : 1;
  __REG32 THE               : 1;
  __REG32 BREN              : 1;
  __REG32 MODE              : 1;
  __REG32 SD                : 1;
  __REG32 SCALE             : 1;
  __REG32 DEVNAK            : 1;
  __REG32 CSR_DONE          : 1;
  __REG32                   : 2;
  __REG32 BRLEN             : 8;
  __REG32 THLEN             : 8;
} __uddctrl_bits;

/* Device status register */
typedef struct {
  __REG32 CFG               : 4;
  __REG32 INTF              : 4;
  __REG32 ALT               : 4;
  __REG32 SUSP              : 1;
  __REG32 ENUM_SPD          : 2;
  __REG32 RXFIFO_EMPTY      : 1;
  __REG32 PHY_ERROR         : 1;
  __REG32                   : 1;
  __REG32 TS                :14;
} __uddstat_bits;

/* Device interrupt register */
typedef struct {
  __REG32 SC                : 1;
  __REG32 SI                : 1;
  __REG32 ES                : 1;
  __REG32 UR                : 1;
  __REG32 US                : 1;
  __REG32 SOF               : 1;
  __REG32 ENUM              : 1;
  __REG32                   :25;
} __uddintr_bits;

/* Device interrupt register */
typedef struct {
  __REG32 IN_EP0            : 1;
  __REG32 IN_EP1            : 1;
  __REG32 IN_EP2            : 1;
  __REG32 IN_EP3            : 1;
  __REG32 IN_EP4            : 1;
  __REG32 IN_EP5            : 1;
  __REG32 IN_EP6            : 1;
  __REG32 IN_EP7            : 1;
  __REG32 IN_EP8            : 1;
  __REG32 IN_EP9            : 1;
  __REG32 IN_EP10           : 1;
  __REG32 IN_EP11           : 1;
  __REG32 IN_EP12           : 1;
  __REG32 IN_EP13           : 1;
  __REG32 IN_EP14           : 1;
  __REG32 IN_EP15           : 1;
  __REG32 OUT_EP0           : 1;
  __REG32 OUT_EP1           : 1;
  __REG32 OUT_EP2           : 1;
  __REG32 OUT_EP3           : 1;
  __REG32 OUT_EP4           : 1;
  __REG32 OUT_EP5           : 1;
  __REG32 OUT_EP6           : 1;
  __REG32 OUT_EP7           : 1;
  __REG32 OUT_EP8           : 1;
  __REG32 OUT_EP9           : 1;
  __REG32 OUT_EP10          : 1;
  __REG32 OUT_EP11          : 1;
  __REG32 OUT_EP12          : 1;
  __REG32 OUT_EP13          : 1;
  __REG32 OUT_EP14          : 1;
  __REG32 OUT_EP15          : 1;
} __udeintr_bits;

/* Endpoint control register */
typedef struct {
  __REG32 S                 : 1;
  __REG32 F                 : 1;
  __REG32 SN                : 1;
  __REG32 P                 : 1;
  __REG32 ET                : 2;
  __REG32 NAK               : 1;
  __REG32 SNAK              : 1;
  __REG32 CNAK              : 1;
  __REG32 RRDY              : 1;
  __REG32                   : 1;
  __REG32 CLOSE_DESC        : 1;
  __REG32                   :20;
} __udepctrl_bits;

/* Endpoint status register */
typedef struct {
  __REG32                   : 4;
  __REG32 OUT               : 2;
  __REG32 IN                : 1;
  __REG32 BNA               : 1;
  __REG32                   : 1;
  __REG32 HE                : 1;
  __REG32 TDC               : 1;
  __REG32 RX_PKT_SIZE       :12;
  __REG32 ISOIN_DONE        : 1;
  __REG32                   : 8;
} __udepstat_bits;

/* Endpoint buffer size and received packet frame number register */
typedef struct {
  __REG32 BUFF_SIZE         :16;
  __REG32 ISO_PID           : 2;
  __REG32                   :14;
} __udepbs_bits;

/* Endpoint maximum packet size and buffer size register */
typedef struct {
  __REG32 MAX_PKT_SIZE      :16;
  __REG32 BUFF_SIZE         :16;
} __udepmps_bits;

/* UDC20 endpoint register */
typedef struct {
  __REG32 EPNumber          : 4;
  __REG32 EPDir             : 1;
  __REG32 EPType            : 2;
  __REG32 ConfNumber        : 4;
  __REG32 InterfNumber      : 4;
  __REG32 AltSetting        : 4;
  __REG32 MaxPackSize       :11;
  __REG32                   : 2;
} __udep_bits;

/* UARTDR register */
typedef struct
{
  __REG16 DATA                 : 8;
  __REG16 FE                   : 1;
  __REG16 PE                   : 1;
  __REG16 BE                   : 1;
  __REG16 OE                   : 1;
  __REG16                      : 4;
} __uartdr_bits;

/* UARTRSR register */
typedef struct
{
  __REG8  FE                   : 1;
  __REG8  PE                   : 1;
  __REG8  BE                   : 1;
  __REG8  OE                   : 1;
  __REG8                       : 4;
} __uartrsr_bits;

/* UARTFR register */
typedef struct
{
  __REG16  CTS                  : 1;
  __REG16  DSR                  : 1;
  __REG16  DCD                  : 1;
  __REG16  BUSY                 : 1;
  __REG16  RXFE                 : 1;
  __REG16  TXFF                 : 1;
  __REG16  RXFF                 : 1;
  __REG16  TXFE                 : 1;
  __REG16  RI                   : 1;
  __REG16                       : 7;
} __uartfr_bits;

/* UARTFBRD register */
typedef struct
{
  __REG8   DIVFRAC              : 6;
  __REG8                        : 2;
} __uartfbrd_bits;

/* UARTLCR_H register */
typedef struct
{
  __REG16  BRK                  : 1;
  __REG16  PEN                  : 1;
  __REG16  EPS                  : 1;
  __REG16  STP2                 : 1;
  __REG16  FEN                  : 1;
  __REG16  WLEN                 : 2;
  __REG16  SPS                  : 1;
  __REG16                       : 8;
} __uartlcr_h_bits;

/* UARTCR register */
typedef struct
{
  __REG16  UARTEN               : 1;
  __REG16                       : 6;
  __REG16  LBE                  : 1;
  __REG16  TXE                  : 1;
  __REG16  RXE                  : 1;
  __REG16  DTR                  : 1;
  __REG16  RTS                  : 1;
  __REG16  Out1                 : 1;
  __REG16  Out2                 : 1;
  __REG16  RTSEn                : 1;
  __REG16  CTSEn                : 1;
} __uartcr_bits;

/* UARTIFLS register */
typedef struct
{
  __REG16  TXIFLSEL             : 3;
  __REG16  RXIFLSEL             : 3;
  __REG16                       :10;
} __uartifls_bits;

/* UARTIMSC register */
typedef struct
{
  __REG16  RIMIM                : 1;
  __REG16  CTSMIM               : 1;
  __REG16  DCDMIM               : 1;
  __REG16  DSRMIM               : 1;
  __REG16  RXIM                 : 1;
  __REG16  TXIM                 : 1;
  __REG16  RTIM                 : 1;
  __REG16  FEIM                 : 1;
  __REG16  PEIM                 : 1;
  __REG16  BEIM                 : 1;
  __REG16  OEIM                 : 1;
  __REG16                       : 5;
} __uartimsc_bits;

/* UARTRIS register */
typedef struct
{
  __REG16  RIRMIS               : 1;
  __REG16  CTSRMIS              : 1;
  __REG16  DCDRMIS              : 1;
  __REG16  DSRRMIS              : 1;
  __REG16  RXRIS                : 1;
  __REG16  TXRIS                : 1;
  __REG16  RTRIS                : 1;
  __REG16  FERIS                : 1;
  __REG16  PERIS                : 1;
  __REG16  BERIS                : 1;
  __REG16  OERIS                : 1;
  __REG16                       : 5;
} __uartris_bits;

/* UARTMIS Register */
typedef struct
{
  __REG16  RIMMIS               : 1;
  __REG16  CTSMMIS              : 1;
  __REG16  DCDMMIS              : 1;
  __REG16  DSRMMIS              : 1;
  __REG16  RXMIS                : 1;
  __REG16  TXMIS                : 1;
  __REG16  RTMIS                : 1;
  __REG16  FEMIS                : 1;
  __REG16  PEMIS                : 1;
  __REG16  BEMIS                : 1;
  __REG16  OEMIS                : 1;
  __REG16                       : 5;
} __uartmis_bits;

/* UARTICR register */
typedef struct
{
  __REG16  RIMIC                : 1;
  __REG16  CTSMIC               : 1;
  __REG16  DCDMIC               : 1;
  __REG16  DSRMIC               : 1;
  __REG16  RXIC                 : 1;
  __REG16  TXIC                 : 1;
  __REG16  RTIC                 : 1;
  __REG16  FEIC                 : 1;
  __REG16  PEIC                 : 1;
  __REG16  BEIC                 : 1;
  __REG16  OEIC                 : 1;
  __REG16                       : 5;
} __uarticr_bits;

/* UARTDMACR register */
typedef struct
{
  __REG16  RXDMAE               : 1;
  __REG16  TXDMAE               : 1;
  __REG16  DMAONERR             : 1;
  __REG16                       :13;
} __uartdmacr_bits;

/* IrDA_CON register */
typedef struct
{
  __REG32  RUN                  : 1;
  __REG32                       :31;
} __irda_con_bits;

/* IrDA_CONF register */
typedef struct
{
  __REG32  RATV                 :13;
  __REG32                       : 3;
  __REG32  BS                   : 3;
  __REG32  POLRX                : 1;
  __REG32  POLTX                : 1;
  __REG32                       :11;
} __irda_conf_bits;

/* IrDA_PARA register */
typedef struct
{
  __REG32  MODE                 : 2;
  __REG32  ABF                  : 6;
  __REG32                       : 8;
  __REG32  MNRB                 :12;
  __REG32                       : 4;
} __irda_para_bits;

/* IrDA_DV register */
typedef struct
{
  __REG32  N                    : 8;
  __REG32  INC                  : 8;
  __REG32  DEC                  :11;
  __REG32                       : 5;
} __irda_dv_bits;

/* IrDA_STAT register */
typedef struct
{
  __REG32  RXS                  : 1;
  __REG32  TXS                  : 1;
  __REG32                       :30;
} __irda_stat_bits;

/* IrDA_TFS register */
typedef struct
{
  __REG32  TFS                  :12;
  __REG32                       :20;
} __irda_tfs_bits;

/* IrDA_RFS register */
typedef struct
{
  __REG32  RFS                  :12;
  __REG32                       :20;
} __irda_rfs_bits;

/* IrDA_IMSC register */
typedef struct
{
  __REG32  LSREQ                : 1;
  __REG32  SREQ                 : 1;
  __REG32  LBREQ                : 1;
  __REG32  BREQ                 : 1;
  __REG32  FT                   : 1;
  __REG32  SD                   : 1;
  __REG32  FI                   : 1;
  __REG32  FD                   : 1;
  __REG32                       :24;
} __irda_imsc_bits;

/* IrDA_DMA register */
typedef struct
{
  __REG32  LSREQEN              : 1;
  __REG32  SREQEN               : 1;
  __REG32  LBREQEN              : 1;
  __REG32  BREQEN               : 1;
  __REG32                       :28;
} __irda_dma_bits;

/* SSPCR0 register */
typedef struct
{
  __REG16  DSS                  : 4;
  __REG16  FRF                  : 2;
  __REG16  SPO                  : 1;
  __REG16  SPH                  : 1;
  __REG16  SCR                  : 8;
} __sspcr0_bits;

/* SSPCR1 register */
typedef struct
{
  __REG16  LBM                  : 1;
  __REG16  SSE                  : 1;
  __REG16  MS                   : 1;
  __REG16  SOD                  : 1;
  __REG16                       :12;
} __sspcr1_bits;

/* SSPSR register */
typedef struct
{
  __REG16  TFE                  : 1;
  __REG16  TNF                  : 1;
  __REG16  RNE                  : 1;
  __REG16  RFF                  : 1;
  __REG16  BSY                  : 1;
  __REG16                       :11;
} __sspsr_bits;

/* SSPCPSR register */
typedef struct
{
  __REG16  CPSDVSR              : 8;
  __REG16                       : 8;
} __sspcpsr_bits;

/* SSPIMSC register */
typedef struct
{
  __REG16  RORIM                : 1;
  __REG16  RTIM                 : 1;
  __REG16  RXIM                 : 1;
  __REG16  TXIM                 : 1;
  __REG16                       :12;
} __sspimsc_bits;

/* SSPRIS register */
typedef struct
{
  __REG16  RORRIS               : 1;
  __REG16  RTRIS                : 1;
  __REG16  RXRIS                : 1;
  __REG16  TXRIS                : 1;
  __REG16                       :12;
} __sspris_bits;

/* SSPMIS Register */
typedef struct
{
  __REG16  RORMIS               : 1;
  __REG16  RTMIS                : 1;
  __REG16  RXMIS                : 1;
  __REG16  TXMIS                : 1;
  __REG16                       :12;
} __sspmis_bits;

/* SSPICR register */
typedef struct
{
  __REG16  RORIC                : 1;
  __REG16  RTIC                 : 1;
  __REG16                       :14;
} __sspicr_bits;

/* SSPDMACR register */
typedef struct
{
  __REG16  RXDMAE               : 1;
  __REG16  TXDMAE               : 1;
  __REG16                       :14;
} __sspdmacr_bits;

/* PHERIPHID0 register */
typedef struct
{
  __REG16  PartNumber0          : 8;
  __REG16                       : 8;
} __sspperiphid0_bits;

/* PHERIPHID1 register */
typedef struct
{
  __REG16  PartNumber1          : 4;
  __REG16  Designer0            : 4;
  __REG16                       : 8;
} __sspperiphid1_bits;

/* PHERIPHID2 register */
typedef struct
{
  __REG16  Designer1            : 4;
  __REG16  Revision             : 4;
  __REG16                       : 8;
} __sspperiphid2_bits;

/* PHERIPHID3 register */
typedef struct
{
  __REG16  Configuration        : 8;
  __REG16                       : 8;
} __sspperiphid3_bits;

/* PCELLID0 register */
typedef struct
{
  __REG16  PCELLID0             : 8;
  __REG16                       : 8;
} __sspcellid0_bits;

/* PCELLID1 register */
typedef struct
{
  __REG16  PCELLID1             : 8;
  __REG16                       : 8;
} __sspcellid1_bits;

/* PCELLID2 register */
typedef struct
{
  __REG16  PCELLID2             : 8;
  __REG16                       : 8;
} __sspcellid2_bits;

/* PCELLID3 register */
typedef struct
{
  __REG16  PCELLID3             : 8;
  __REG16                       : 8;
} __sspcellid3_bits;

/* IC_CON register */
typedef struct
{
  __REG16  MASTER_MODE          : 1;
  __REG16  SPEED                : 2;
  __REG16  IC_10BITADDR_SLAVE   : 1;
  __REG16  IC_10BITADDR_MASTER  : 1;
  __REG16  IC_RESTART_EN        : 1;
  __REG16  IC_SLAVE_DISABLE     : 1;
  __REG16                       : 9;
} __ic_con_bits;

/* IC_TAR register */
typedef struct
{
  __REG16  IC_TAR               :10;
  __REG16  GC_OR_START          : 1;
  __REG16  SPECIAL              : 1;
  __REG16  IC_10BITADDR_MASTER  : 1;
  __REG16                       : 3;
} __ic_tar_bits;

/* IC_SAR register */
typedef struct
{
  __REG16  IC_SAR               :10;
  __REG16                       : 6;
} __ic_sar_bits;

/* IC_HS_MADDR register */
typedef struct
{
  __REG16  IC_HS_MAR            : 3;
  __REG16                       :13;
} __ic_hs_maddr_bits;

/* IC_DATA_CMD register */
typedef struct
{
  __REG16  DAT                  : 8;
  __REG16  CMD                  : 1;
  __REG16                       : 7;
} __ic_data_cmd_bits;

/* IC_INTR_STAT register */
typedef struct
{
  __REG16  R_RX_UNDER           : 1;
  __REG16  R_RX_OVER            : 1;
  __REG16  R_RX_FULL            : 1;
  __REG16  R_TX_OVER            : 1;
  __REG16  R_TX_EMPTY           : 1;
  __REG16  R_RD_REQ             : 1;
  __REG16  R_TX_ABRT            : 1;
  __REG16  R_RX_DONE            : 1;
  __REG16  R_ACTIVITY           : 1;
  __REG16  R_STOP_DET           : 1;
  __REG16  R_START_DET          : 1;
  __REG16  R_GEN_CALL           : 1;
  __REG16                       : 4;
} __ic_intr_stat_bits;

/* IC_INTR_MASK register */
typedef struct
{
  __REG16  M_RX_UNDER           : 1;
  __REG16  M_RX_OVER            : 1;
  __REG16  M_RX_FULL            : 1;
  __REG16  M_TX_OVER            : 1;
  __REG16  M_TX_EMPTY           : 1;
  __REG16  M_RD_REQ             : 1;
  __REG16  M_TX_ABRT            : 1;
  __REG16  M_RX_DONE            : 1;
  __REG16  M_ACTIVITY           : 1;
  __REG16  M_STOP_DET           : 1;
  __REG16  M_START_DET          : 1;
  __REG16  M_GEN_CALL           : 1;
  __REG16                       : 4;
} __ic_intr_mask_bits;

/* IC_RAW_INTR_STAT register */
typedef struct
{
  __REG16  RX_UNDER           : 1;
  __REG16  RX_OVER            : 1;
  __REG16  RX_FULL            : 1;
  __REG16  TX_OVER            : 1;
  __REG16  TX_EMPTY           : 1;
  __REG16  RD_REQ             : 1;
  __REG16  TX_ABRT            : 1;
  __REG16  RX_DONE            : 1;
  __REG16  ACTIVITY           : 1;
  __REG16  STOP_DET           : 1;
  __REG16  START_DET          : 1;
  __REG16  GEN_CALL           : 1;
  __REG16                     : 4;
} __ic_raw_intr_stat_bits;

/* IC_RX_TL register */
typedef struct
{
  __REG16  RX_TL              : 8;
  __REG16                     : 8;
} __ic_rx_tl_bits;

/* IC_TX_TL register */
typedef struct
{
  __REG16  TX_TL              : 8;
  __REG16                     : 8;
} __ic_tx_tl_bits;

/* IC_CLR_INTR register */
typedef struct
{
  __REG16  CLR_INTR           : 1;
  __REG16                     :15;
} __ic_clr_intr_bits;

/* IC_ENABLE register */
typedef struct
{
  __REG16  ENABLE             : 1;
  __REG16                     :15;
} __ic_enable_bits;

/* IC_STATUS register */
typedef struct
{
  __REG16  ACTIVITY           : 1;
  __REG16  TFNF               : 1;
  __REG16  TFE                : 1;
  __REG16  RFNE               : 1;
  __REG16  RFF                : 1;
  __REG16  MST_ACTIVITY       : 1;
  __REG16  SLV_ACTIVITY       : 1;
  __REG16                     : 9;
} __ic_status_bits;

/* IC_TXFLR register */
typedef struct
{
  __REG16  TXFLR              : 4;
  __REG16                     :12;
} __ic_txflr_bits;

/* IC_RXFLR register */
typedef struct
{
  __REG16  RXFLR              : 4;
  __REG16                     :12;
} __ic_rxflr_bits;

/* IC_TX_ABRT_SOURCE register */
typedef struct
{
  __REG16  ABRT_7B_ADDR_NOACK   : 1;
  __REG16  ABRT_10ADDR1_NOACK   : 1;
  __REG16  ABRT_10ADDR2_NOACK   : 1;
  __REG16  ABRT_TXDATA_NOACK    : 1;
  __REG16  ABRT_GCALL_NOACK     : 1;
  __REG16  ABRT_GCALL_READ      : 1;
  __REG16  ABRT_HS_ACKDET       : 1;
  __REG16  ABRT_SBYTE_ACKDET    : 1;
  __REG16  ABRT_HS_NORSTRT      : 1;
  __REG16  ABRT_SBYTE_NORSTRT   : 1;
  __REG16  ABRT_10B_RD_NORSTRT  : 1;
  __REG16  ARB_MASTER_DIS       : 1;
  __REG16  ARB_LOST             : 1;
  __REG16  ABRT_SLVFLUSH_TXFIFO : 1;
  __REG16  ABRT_SLV_ARBLOST     : 1;
  __REG16  ABRT_SLVRD_INTX      : 1;
} __ic_tx_abrt_source_bits;

/* IC_DMA_CR register */
typedef struct
{
  __REG16  RDMAE                : 1;
  __REG16  TDMAE                : 1;
  __REG16                       :14;
} __ic_dma_cr_bits;

/* IC_DMA_TDLR register */
typedef struct
{
  __REG16  DMATDL               : 3;
  __REG16                       :13;
} __ic_dma_tdlr_bits;

/* IC_DMA_RDLR register */
typedef struct
{
  __REG16  DMARDL               : 3;
  __REG16                       :13;
} __ic_dma_rdlr_bits;

/* IC_COMP_PARAM1 register */
typedef struct
{
  __REG32  APB_DATA_WIDTH       : 2;
  __REG32  MAX_SPEED_MODE       : 2;
  __REG32  HC_COUNT_VALUES      : 1;
  __REG32  INTR_IO              : 1;
  __REG32  HAS_DMA              : 1;
  __REG32  ADD_ENCODED_PARAMS   : 1;
  __REG32  RX_BUFFER_DEPTH      : 8;
  __REG32  TX_BUFFER_DEPTH      : 8;
  __REG32                       : 8;
} __ic_comp_param_1_bits;

/* DMACIntStatus register */
typedef struct
{
  __REG32  IntStatus0           : 1;
  __REG32  IntStatus1           : 1;
  __REG32  IntStatus2           : 1;
  __REG32  IntStatus3           : 1;
  __REG32  IntStatus4           : 1;
  __REG32  IntStatus5           : 1;
  __REG32  IntStatus6           : 1;
  __REG32  IntStatus7           : 1;
  __REG32                       :24;
} __dmacintstatus_bits;

/* DMACIntTCStatus register */
typedef struct
{
  __REG32  IntTCStatus0         : 1;
  __REG32  IntTCStatus1         : 1;
  __REG32  IntTCStatus2         : 1;
  __REG32  IntTCStatus3         : 1;
  __REG32  IntTCStatus4         : 1;
  __REG32  IntTCStatus5         : 1;
  __REG32  IntTCStatus6         : 1;
  __REG32  IntTCStatus7         : 1;
  __REG32                       :24;
} __dmacinttcstatus_bits;

/* DMACIntTCClear register */
typedef struct
{
  __REG32  IntTCClear0          : 1;
  __REG32  IntTCClear1          : 1;
  __REG32  IntTCClear2          : 1;
  __REG32  IntTCClear3          : 1;
  __REG32  IntTCClear4          : 1;
  __REG32  IntTCClear5          : 1;
  __REG32  IntTCClear6          : 1;
  __REG32  IntTCClear7          : 1;
  __REG32                       :24;
} __dmacinttcclear_bits;

/* DMACIntErrorStatus register */
typedef struct
{
  __REG32  IntErrorStatus0      : 1;
  __REG32  IntErrorStatus1      : 1;
  __REG32  IntErrorStatus2      : 1;
  __REG32  IntErrorStatus3      : 1;
  __REG32  IntErrorStatus4      : 1;
  __REG32  IntErrorStatus5      : 1;
  __REG32  IntErrorStatus6      : 1;
  __REG32  IntErrorStatus7      : 1;
  __REG32                       :24;
} __dmacinterrorstatus_bits;

/* DMACIntErrClr register */
typedef struct
{
  __REG32  IntErrClr0           : 1;
  __REG32  IntErrClr1           : 1;
  __REG32  IntErrClr2           : 1;
  __REG32  IntErrClr3           : 1;
  __REG32  IntErrClr4           : 1;
  __REG32  IntErrClr5           : 1;
  __REG32  IntErrClr6           : 1;
  __REG32  IntErrClr7           : 1;
  __REG32                       :24;
} __dmacinterrclr_bits;

/* DMACRawIntTCStatus register */
typedef struct
{
  __REG32  RawIntTCStatus0      : 1;
  __REG32  RawIntTCStatus1      : 1;
  __REG32  RawIntTCStatus2      : 1;
  __REG32  RawIntTCStatus3      : 1;
  __REG32  RawIntTCStatus4      : 1;
  __REG32  RawIntTCStatus5      : 1;
  __REG32  RawIntTCStatus6      : 1;
  __REG32  RawIntTCStatus7      : 1;
  __REG32                       :24;
} __dmacrawinttcstatus_bits;

/* DMACRawIntErrorStatus register */
typedef struct
{
  __REG32  RawIntErrorStatus0   : 1;
  __REG32  RawIntErrorStatus1   : 1;
  __REG32  RawIntErrorStatus2   : 1;
  __REG32  RawIntErrorStatus3   : 1;
  __REG32  RawIntErrorStatus4   : 1;
  __REG32  RawIntErrorStatus5   : 1;
  __REG32  RawIntErrorStatus6   : 1;
  __REG32  RawIntErrorStatus7   : 1;
  __REG32                       :24;
} __dmacrawinterrorstatus_bits;

/* DMACEnbldChns register */
typedef struct
{
  __REG32  EnabledChannels0     : 1;
  __REG32  EnabledChannels1     : 1;
  __REG32  EnabledChannels2     : 1;
  __REG32  EnabledChannels3     : 1;
  __REG32  EnabledChannels4     : 1;
  __REG32  EnabledChannels5     : 1;
  __REG32  EnabledChannels6     : 1;
  __REG32  EnabledChannels7     : 1;
  __REG32                       :24;
} __dmacenbldchns_bits;

/* DMACSoftBReq register */
typedef struct
{
  __REG32  SoftBReq0            : 1;
  __REG32  SoftBReq1            : 1;
  __REG32  SoftBReq2            : 1;
  __REG32  SoftBReq3            : 1;
  __REG32  SoftBReq4            : 1;
  __REG32  SoftBReq5            : 1;
  __REG32  SoftBReq6            : 1;
  __REG32  SoftBReq7            : 1;
  __REG32  SoftBReq8            : 1;
  __REG32  SoftBReq9            : 1;
  __REG32  SoftBReq10           : 1;
  __REG32  SoftBReq11           : 1;
  __REG32  SoftBReq12           : 1;
  __REG32  SoftBReq13           : 1;
  __REG32  SoftBReq14           : 1;
  __REG32  SoftBReq15           : 1;
  __REG32                       :16;
} __dmacsoftbreq_bits;

/* DMACSoftSReq register */
typedef struct
{
  __REG32  SoftSReq0            : 1;
  __REG32  SoftSReq1            : 1;
  __REG32  SoftSReq2            : 1;
  __REG32  SoftSReq3            : 1;
  __REG32  SoftSReq4            : 1;
  __REG32  SoftSReq5            : 1;
  __REG32  SoftSReq6            : 1;
  __REG32  SoftSReq7            : 1;
  __REG32  SoftSReq8            : 1;
  __REG32  SoftSReq9            : 1;
  __REG32  SoftSReq10           : 1;
  __REG32  SoftSReq11           : 1;
  __REG32  SoftSReq12           : 1;
  __REG32  SoftSReq13           : 1;
  __REG32  SoftSReq14           : 1;
  __REG32  SoftSReq15           : 1;
  __REG32                       :16;
} __dmacsoftsreq_bits;

/* DMACSoftLBReq register */
typedef struct
{
  __REG32  SoftLBReq0           : 1;
  __REG32  SoftLBReq1           : 1;
  __REG32  SoftLBReq2           : 1;
  __REG32  SoftLBReq3           : 1;
  __REG32  SoftLBReq4           : 1;
  __REG32  SoftLBReq5           : 1;
  __REG32  SoftLBReq6           : 1;
  __REG32  SoftLBReq7           : 1;
  __REG32  SoftLBReq8           : 1;
  __REG32  SoftLBReq9           : 1;
  __REG32  SoftLBReq10          : 1;
  __REG32  SoftLBReq11          : 1;
  __REG32  SoftLBReq12          : 1;
  __REG32  SoftLBReq13          : 1;
  __REG32  SoftLBReq14          : 1;
  __REG32  SoftLBReq15          : 1;
  __REG32                       :16;
} __dmacsoftlbreq_bits;

/* DMACSoftLSReq register */
typedef struct
{
  __REG32  SoftLSReq0           : 1;
  __REG32  SoftLSReq1           : 1;
  __REG32  SoftLSReq2           : 1;
  __REG32  SoftLSReq3           : 1;
  __REG32  SoftLSReq4           : 1;
  __REG32  SoftLSReq5           : 1;
  __REG32  SoftLSReq6           : 1;
  __REG32  SoftLSReq7           : 1;
  __REG32  SoftLSReq8           : 1;
  __REG32  SoftLSReq9           : 1;
  __REG32  SoftLSReq10          : 1;
  __REG32  SoftLSReq11          : 1;
  __REG32  SoftLSReq12          : 1;
  __REG32  SoftLSReq13          : 1;
  __REG32  SoftLSReq14          : 1;
  __REG32  SoftLSReq15          : 1;
  __REG32                       :16;
} __dmacsoftlsreq_bits;

/* DMAC configuration register */
typedef struct
{
  __REG32  E                    : 1;
  __REG32  M1                   : 1;
  __REG32  M2                   : 1;
  __REG32                       :29;
} __dmacconfiguration_bits;

/* DMACSoftLSReq register */
typedef struct
{
  __REG32  DMACSync0            : 1;
  __REG32  DMACSync1            : 1;
  __REG32  DMACSync2            : 1;
  __REG32  DMACSync3            : 1;
  __REG32  DMACSync4            : 1;
  __REG32  DMACSync5            : 1;
  __REG32  DMACSync6            : 1;
  __REG32  DMACSync7            : 1;
  __REG32  DMACSync8            : 1;
  __REG32  DMACSync9            : 1;
  __REG32  DMACSync10           : 1;
  __REG32  DMACSync11           : 1;
  __REG32  DMACSync12           : 1;
  __REG32  DMACSync13           : 1;
  __REG32  DMACSync14           : 1;
  __REG32  DMACSync15           : 1;
  __REG32                       :16;
} __dmacsync_bits;

/* DMACCnLLI register */
typedef struct
{
  __REG32  LM                   : 1;
  __REG32                       : 1;
  __REG32  LLI                  :30;
} __dmacclli_bits;

/* DMACCn control register */
typedef struct
{
  __REG32  TS                   :12;
  __REG32  SBSize               : 3;
  __REG32  DBSize               : 3;
  __REG32  Swidth               : 3;
  __REG32  Dwidth               : 3;
  __REG32  S                    : 1;
  __REG32  D                    : 1;
  __REG32  SI                   : 1;
  __REG32  DI                   : 1;
  __REG32  Port                 : 3;
  __REG32  I                    : 1;
} __dmacccontrol_bits;

/* DMAC Configuration register */
typedef struct
{
  __REG32  E                    : 1;
  __REG32  SrcPeripheral        : 4;
  __REG32                       : 1;
  __REG32  DestPeripheral       : 4;
  __REG32                       : 1;
  __REG32  FlowCntrl            : 3;
  __REG32  IE                   : 1;
  __REG32  ITC                  : 1;
  __REG32  L                    : 1;
  __REG32  A                    : 1;
  __REG32  H                    : 1;
  __REG32                       :13;
} __dmaccconfiguration_bits;

/* GPIODIR register */
typedef struct
{
  __REG16  GPIODIR0             : 1;
  __REG16  GPIODIR1             : 1;
  __REG16  GPIODIR2             : 1;
  __REG16  GPIODIR3             : 1;
  __REG16  GPIODIR4             : 1;
  __REG16  GPIODIR5             : 1;
  __REG16  GPIODIR6             : 1;
  __REG16  GPIODIR7             : 1;
  __REG16                       : 8;
} __gpiodir_bits;

/* GPIODATA register */
typedef struct
{
  __REG16  GPIODATA0            : 1;
  __REG16  GPIODATA1            : 1;
  __REG16  GPIODATA2            : 1;
  __REG16  GPIODATA3            : 1;
  __REG16  GPIODATA4            : 1;
  __REG16  GPIODATA5            : 1;
  __REG16  GPIODATA6            : 1;
  __REG16  GPIODATA7            : 1;
  __REG16                       : 8;
} __gpiodata_bits;

/* GPIOIS register */
typedef struct
{
  __REG16  GPIOIS0              : 1;
  __REG16  GPIOIS1              : 1;
  __REG16  GPIOIS2              : 1;
  __REG16  GPIOIS3              : 1;
  __REG16  GPIOIS4              : 1;
  __REG16  GPIOIS5              : 1;
  __REG16  GPIOIS6              : 1;
  __REG16  GPIOIS7              : 1;
  __REG16                       : 8;
} __gpiois_bits;

/* GPIOIBE register */
typedef struct
{
  __REG16  GPIOIBE0             : 1;
  __REG16  GPIOIBE1             : 1;
  __REG16  GPIOIBE2             : 1;
  __REG16  GPIOIBE3             : 1;
  __REG16  GPIOIBE4             : 1;
  __REG16  GPIOIBE5             : 1;
  __REG16  GPIOIBE6             : 1;
  __REG16  GPIOIBE7             : 1;
  __REG16                       : 8;
} __gpioibe_bits;

/* GPIOIEV register */
typedef struct
{
  __REG16  GPIOIEV0             : 1;
  __REG16  GPIOIEV1             : 1;
  __REG16  GPIOIEV2             : 1;
  __REG16  GPIOIEV3             : 1;
  __REG16  GPIOIEV4             : 1;
  __REG16  GPIOIEV5             : 1;
  __REG16  GPIOIEV6             : 1;
  __REG16  GPIOIEV7             : 1;
  __REG16                       : 8;
} __gpioiev_bits;

/* GPIOIE register */
typedef struct
{
  __REG16  GPIOIE0              : 1;
  __REG16  GPIOIE1              : 1;
  __REG16  GPIOIE2              : 1;
  __REG16  GPIOIE3              : 1;
  __REG16  GPIOIE4              : 1;
  __REG16  GPIOIE5              : 1;
  __REG16  GPIOIE6              : 1;
  __REG16  GPIOIE7              : 1;
  __REG16                       : 8;
} __gpioie_bits;

/* GPIORIS register */
typedef struct
{
  __REG16  GPIORIS0             : 1;
  __REG16  GPIORIS1             : 1;
  __REG16  GPIORIS2             : 1;
  __REG16  GPIORIS3             : 1;
  __REG16  GPIORIS4             : 1;
  __REG16  GPIORIS5             : 1;
  __REG16  GPIORIS6             : 1;
  __REG16  GPIORIS7             : 1;
  __REG16                       : 8;
} __gpioris_bits;

/* GPIOMIS register */
typedef struct
{
  __REG16  GPIOMIS0             : 1;
  __REG16  GPIOMIS1             : 1;
  __REG16  GPIOMIS2             : 1;
  __REG16  GPIOMIS3             : 1;
  __REG16  GPIOMIS4             : 1;
  __REG16  GPIOMIS5             : 1;
  __REG16  GPIOMIS6             : 1;
  __REG16  GPIOMIS7             : 1;
  __REG16                       : 8;
} __gpiomis_bits;

/* GPIOIC register */
typedef struct
{
  __REG16  GPIOIC0              : 1;
  __REG16  GPIOIC1              : 1;
  __REG16  GPIOIC2              : 1;
  __REG16  GPIOIC3              : 1;
  __REG16  GPIOIC4              : 1;
  __REG16  GPIOIC5              : 1;
  __REG16  GPIOIC6              : 1;
  __REG16  GPIOIC7              : 1;
  __REG16                       : 8;
} __gpioic_bits;

/* GPIOAFSEL Register */
typedef struct
{
  __REG16  GPIOAFSEL0           : 1;
  __REG16  GPIOAFSEL1           : 1;
  __REG16  GPIOAFSEL2           : 1;
  __REG16  GPIOAFSEL3           : 1;
  __REG16  GPIOAFSEL4           : 1;
  __REG16  GPIOAFSEL5           : 1;
  __REG16  GPIOAFSEL6           : 1;
  __REG16  GPIOAFSEL7           : 1;
  __REG16                       : 8;
} __gpioafsel_bits;

/* LCD timing 0 register */
typedef struct
{
  __REG32                : 2;
  __REG32 PPL            : 6;
  __REG32 HSW            : 8;
  __REG32 HFP            : 8;
  __REG32 HBP            : 8;
} __lcdtiming0_bits;

/* LCD timing 1 register */
typedef struct
{
  __REG32 LPP            :10;
  __REG32 VSW            : 6;
  __REG32 VFP            : 8;
  __REG32 VBP            : 8;
} __lcdtiming1_bits;

/* LCD timing 2 register */
typedef struct
{
  __REG32 PCD_LO         : 5;
  __REG32 CLKSEL         : 1;
  __REG32 ACB            : 5;
  __REG32 IVS            : 1;
  __REG32 IHS            : 1;
  __REG32 IPC            : 1;
  __REG32 IEO            : 1;
  __REG32                : 1;
  __REG32 CPL            :10;
  __REG32 BCD            : 1;
  __REG32 PCD_HI         : 5;
} __lcdtiming2_bits;

/* LCD timing 3 register */
typedef struct
{
  __REG32 LED            : 7;
  __REG32                : 9;
  __REG32 LEE            : 1;
  __REG32                :15;
} __lcdtiming3_bits;

/* LCDIMSC register */
typedef struct
{
  __REG32                : 1;
  __REG32 FUFINTRENB     : 1;
  __REG32 LNBUINTRENB    : 1;
  __REG32 VCOMPINTRENB   : 1;
  __REG32 MBERRINTRENB   : 1;
  __REG32                :27;
} __lcdmsc_bits;

/* LCD control register */
typedef struct
{
  __REG32 LCDEN          : 1;
  __REG32 LCDBPP         : 3;
  __REG32 LCDBW          : 1;
  __REG32 LCDTFT         : 1;
  __REG32 LCDMONO8       : 1;
  __REG32 LCDDUAL        : 1;
  __REG32 BGR            : 1;
  __REG32 BEBO           : 1;
  __REG32 BEPO           : 1;
  __REG32 LCDPWR         : 1;
  __REG32 LCDVCOMP       : 2;
  __REG32                : 2;
  __REG32 WATERMARK      : 1;
  __REG32                :15;
} __lcdcontrol_bits;

/* LCDRIS register */
typedef struct
{
  __REG32                : 1;
  __REG32 FUF            : 1;
  __REG32 LNBU           : 1;
  __REG32 VCOMP          : 1;
  __REG32 MBERROR        : 1;
  __REG32                :27;
} __lcdris_bits;

/* LCDMIS register */
typedef struct
{
  __REG32                : 1;
  __REG32 FUFINTR        : 1;
  __REG32 LNBUINTR       : 1;
  __REG32 VCOMPINTR      : 1;
  __REG32 MBERRORINTR    : 1;
  __REG32                :27;
} __lcdmis_bits;

/* JPGCreg0 register */
typedef struct
{
  __REG32  StartStop            : 1;
  __REG32                       :31;
} __jpgcreg0_bits;

/* JPGCreg1 register */
typedef struct
{
  __REG32  Nf                   : 2;
  __REG32  Re                   : 1;
  __REG32  De                   : 1;
  __REG32  colspctype           : 2;
  __REG32  Ns                   : 2;
  __REG32  Hdr                  : 1;
  __REG32                       : 7;
  __REG32  Ysiz                 :16;
} __jpgcreg1_bits;

/* JPGCreg2 register */
typedef struct
{
  __REG32  NMCU                 :26;
  __REG32                       : 6;
} __jpgcreg2_bits;

/* JPGCreg3 register */
typedef struct
{
  __REG32  NRST                 :16;
  __REG32  Xsiz                 :16;
} __jpgcreg3_bits;

/* JPGCreg4 register */
typedef struct
{
  __REG32  HD1                  : 1;
  __REG32  HA1                  : 1;
  __REG32  QT1                  : 2;
  __REG32  Nblock1              : 4;
  __REG32  H1                   : 4;
  __REG32  V1                   : 4;
  __REG32                       :16;
} __jpgcreg4_bits;

/* JPGCreg5 register */
typedef struct
{
  __REG32  HD2                  : 1;
  __REG32  HA2                  : 1;
  __REG32  QT2                  : 2;
  __REG32  Nblock2              : 4;
  __REG32  H2                   : 4;
  __REG32  V2                   : 4;
  __REG32                       :16;
} __jpgcreg5_bits;

/* JPGCreg6 register */
typedef struct
{
  __REG32  HD3                  : 1;
  __REG32  HA3                  : 1;
  __REG32  QT3                  : 2;
  __REG32  Nblock3              : 4;
  __REG32  H3                   : 4;
  __REG32  V3                   : 4;
  __REG32                       :16;
} __jpgcreg6_bits;

/* JPGCreg7 register */
typedef struct
{
  __REG32  HD4                  : 1;
  __REG32  HA4                  : 1;
  __REG32  QT4                  : 2;
  __REG32  Nblock4              : 4;
  __REG32  H4                   : 4;
  __REG32  V4                   : 4;
  __REG32                       :16;
} __jpgcreg7_bits;

/* JPGC control status register */
typedef struct
{
  __REG32  INT                  : 1;
  __REG32  BNV                  : 2;
  __REG32  LLI                  :15;
  __REG32                       :12;
  __REG32  SCR                  : 1;
  __REG32  EOC                  : 1;
} __jpgccs_bits;

/* JPGC bust count beforeInit register */
typedef struct
{
  __REG32  NBX                  :31;
  __REG32  ENABLE               : 1;
} __jpgcbcbi_bits;

/* ADC_STATUS_REG register */
typedef struct
{
  __REG16  E            : 1;
  __REG16  CS           : 3;
  __REG16  PD           : 1;
  __REG16  NOAS         : 3;
  __REG16  CR           : 1;
  __REG16  VRS          : 1;
  __REG16  ENM          : 1;
  __REG16  ESR          : 1;
  __REG16  DMA_EN       : 1;
  __REG16               : 3;
} __adc_status_reg_bits;

/* ADC_CLK_REG register */
typedef struct
{
  __REG16  ADC_CLK_L    : 4;
  __REG16  ADC_CLK_H    : 4;
  __REG16               : 8;
} __adc_clk_reg_bits;

/* CHx CTRL register */
typedef struct
{
  __REG16  CE           : 1;
  __REG16  A            : 3;
  __REG16               :12;
} __adc_ch_ctrl_reg_bits;

/* CHx_DATA_LSB register */
typedef struct
{
  __REG16  DATA         : 7;
  __REG16               : 9;
} __adc_ch_data_lsb_bits;

/* CHx_DATA_MSB register */
typedef struct
{
  __REG16  DATA         :10;
  __REG16  VALID_DATA   : 1;
  __REG16               : 5;
} __adc_ch_data_msb_bits;

/* AVERAGE_REG_LSB register */
typedef struct
{
  __REG16  DATA         : 7;
  __REG16               : 9;
} __adc_average_reg_lsb_bits;

/* AVERAGE_REG_MSB register */
typedef struct
{
  __REG16  DATA         :10;
  __REG16               : 6;
} __adc_average_reg_msb_bits;

/* TIME register */
typedef struct
{
  __REG32  SU                   : 4;
  __REG32  ST                   : 3;
  __REG32                       : 1;
  __REG32  MU                   : 4;
  __REG32  MT                   : 3;
  __REG32                       : 1;
  __REG32  HU                   : 4;
  __REG32  HT                   : 2;
  __REG32                       :10;
} __rtctime_bits;

/* DATE register */
typedef struct
{
  __REG32  DU                   : 4;
  __REG32  DT                   : 2;
  __REG32                       : 2;
  __REG32  MU                   : 4;
  __REG32  MT                   : 3;
  __REG32                       : 1;
  __REG32  YU                   : 4;
  __REG32  YT                   : 4;
  __REG32  YH                   : 4;
  __REG32  YM                   : 4;
} __rtcdate_bits;

/* CONTROL register */
typedef struct
{
  __REG32  MASK                 : 6;
  __REG32                       : 2;
  __REG32  PB                   : 1;
  __REG32  TB                   : 1;
  __REG32                       :21;
  __REG32  IE                   : 1;
} __rtccontrol_bits;

/* STATUS register */
typedef struct
{
  __REG32  RC                   : 1;
  __REG32                       : 1;
  __REG32  PT                   : 1;
  __REG32  PD                   : 1;
  __REG32  LT                   : 1;
  __REG32  LD                   : 1;
  __REG32                       :25;
  __REG32  I                    : 1;
} __rtcstatus_bits;

/* Control_Reg */
typedef struct
{
  __REG32                       : 1;
  __REG32  APE                  : 1;
  __REG32  ARE                  : 1;
  __REG32  AM                   : 2;
  __REG32                       : 2;
  __REG32  ROM                  : 1;
  __REG32                       : 2;
  __REG32  PCP                  : 1;
  __REG32  PDL                  : 1;
  __REG32  PWSD                 : 1;
  __REG32  PR                   : 1;
  __REG32  PWSP                 : 1;
  __REG32                       : 3;
  __REG32  RCP                  : 1;
  __REG32  RDL                  : 1;
  __REG32  RD                   : 1;
  __REG32  RR                   : 1;
  __REG32  RWSP                 : 1;
  __REG32  RWSD                 : 1;
  __REG32  RAB                  : 1;
  __REG32                       : 1;
  __REG32  RAP                  : 1;
  __REG32  RAR                  : 1;
  __REG32                       : 4;
} __i2s_control_reg_bits;

/* Irq_Reg */
typedef struct
{
  __REG32                       :14;
  __REG32  PFIFOE               : 1;
  __REG32  PCPUB                : 1;
  __REG32  PAHBE                : 1;
  __REG32  RAHBE                : 1;
  __REG32  RFIFOE               : 1;
  __REG32  RCPUB                : 1;
  __REG32  RDA                	: 1;
  __REG32  PDD                	: 1;
  __REG32                       :10;
} __i2s_irq_reg_bits;

/* Status_Reg */
typedef struct
{
  __REG32                       :26;
  __REG32  RFIFOE               : 1;
  __REG32                       : 5;
} __i2s_status_reg_bits;

/* MLEXP_PL1/2/3 Arbitration priority level register */
typedef struct
{
  __REG32  pry_lvl              : 4;
  __REG32                       :28;
} __mlexp_pl_bits;

/* MLEXP_EBTCOUNT Early burst terminations counter register */
typedef struct
{
  __REG32  ebt_cntr             :10;
  __REG32                       :22;
} __mlepx_ebtcount_bits;

/* MLEXP_EBT_EN Early burst terminations enable register */
typedef struct
{
  __REG32  ebt_enab             : 1;
  __REG32                       :31;
} __mlepx_ebt_en_bits;

/* MLEXP_EBT Early burst terminations status register */
typedef struct
{
  __REG32  early_brst_term      : 1;
  __REG32                       :31;
} __mlepx_ebt_bits;

/* MLEXP_DFT_MST Default master register */
typedef struct
{
  __REG32  def_mst				      : 4;
  __REG32                       :28;
} __mlepx_dft_mst_bits;

/* XLAT_ENT0-7 Address Segment Translation registers */
typedef struct
{
  __REG32  add_segment		      :15;
  __REG32                       :17;
} __xlat_ent_bits;

/* SE2H_EWSC Error clear register */
typedef struct
{
  __REG32                       : 1;
  __REG32  clear_mst1			      : 1;
  __REG32  clear_mst2			      : 1;
  __REG32  clear_mst3			      : 1;
  __REG32  clear_mst4			      : 1;
  __REG32                       :27;
} __se2h_ewsc_bits;

/* SE2H_COMP_PARAM1 Bridge silicon parameter1 register */
typedef struct
{
  __REG32                       : 4;
  __REG32  num_prim_mst		      : 4;
  __REG32  phy_sbig_end		      : 1;
  __REG32  phy_sadr_wdth	      : 1;
  __REG32  phy_sdat_wdth	      : 2;
  __REG32  phy_mbig_end		      : 1;
  __REG32  phy_madr_wdth	      : 1;
  __REG32  phy_mdat_wdth	      : 3;
  __REG32                       :15;
} __se2h_comp_param1_bits;

/* SE2H_COMP_PARAM2 Bridge silicon parameter2 register */
typedef struct
{
  __REG32  wd_buff_dph		      : 8;
  __REG32  rd_buff_dph		      : 8;
  __REG32  rd_prftc_dph		      : 4;
  __REG32                       :12;
} __se2h_comp_param2_bits;

/* ME2H_EWSC Error clear register */
typedef struct
{
  __REG32                       : 1;
  __REG32  clear_mst1			      : 1;
  __REG32  clear_mst2			      : 1;
  __REG32  clear_mst3			      : 1;
  __REG32  clear_mst4			      : 1;
  __REG32                       :27;
} __me2h_ewsc_bits;

/* ME2H_COMP_PARAM1 Bridge silicon parameter1 register */
typedef struct
{
  __REG32                       : 4;
  __REG32  num_prim_mst		      : 4;
  __REG32  phy_sbig_end		      : 1;
  __REG32  phy_sadr_wdth	      : 1;
  __REG32  phy_sdat_wdth	      : 2;
  __REG32  phy_mbig_end		      : 1;
  __REG32  phy_madr_wdth	      : 1;
  __REG32  phy_mdat_wdth	      : 3;
  __REG32                       :15;
} __me2h_comp_param1_bits;

/* ME2H_COMP_PARAM2 Bridge silicon parameter2 register */
typedef struct
{
  __REG32  wd_buff_dph		      : 8;
  __REG32  rd_buff_dph		      : 8;
  __REG32  rd_prftc_dph		      : 4;
  __REG32                       :12;
} __me2h_comp_param2_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  VIC0
 **
 ***************************************************************************/
__IO_REG32_BIT(VIC0IRQSTATUS,     0xF1100000,__READ       ,__vicirqstatus_bits);
__IO_REG32_BIT(VIC0FIQSTATUS,     0xF1100004,__READ       ,__vicfiqstatus_bits);
__IO_REG32_BIT(VIC0RAWINTR,       0xF1100008,__READ       ,__vicrawintr_bits);
__IO_REG32_BIT(VIC0INTSELECT,     0xF110000C,__READ_WRITE ,__vicintselect_bits);
__IO_REG32_BIT(VIC0INTENABLE,     0xF1100010,__READ_WRITE ,__vicintenable_bits);
__IO_REG32_BIT(VIC0INTENCLEAR,    0xF1100014,__WRITE      ,__vicintenclear_bits);
__IO_REG32_BIT(VIC0SOFTINT,       0xF1100018,__READ_WRITE ,__vicsoftint_bits);
__IO_REG32_BIT(VIC0SOFTINTCLEAR,  0xF110001C,__WRITE      ,__vicsoftintclear_bits);
__IO_REG32_BIT(VIC0PROTECTION,    0xF1100020,__READ_WRITE ,__vicprotection_bits);
__IO_REG32(    VIC0VECTADDR,      0xF1100030,__READ_WRITE );
__IO_REG32(    VIC0DEFVECTADDR,   0xF1100034,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR0,     0xF1100100,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR1,     0xF1100104,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR2,     0xF1100108,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR3,     0xF110010C,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR4,     0xF1100110,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR5,     0xF1100114,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR6,     0xF1100118,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR7,     0xF110011C,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR8,     0xF1100120,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR9,     0xF1100124,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR10,    0xF1100128,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR11,    0xF110012C,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR12,    0xF1100130,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR13,    0xF1100134,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR14,    0xF1100138,__READ_WRITE );
__IO_REG32(    VIC0VECTADDR15,    0xF110013C,__READ_WRITE );
__IO_REG32_BIT(VIC0VECTCNTL0,     0xF1100200,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL1,     0xF1100204,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL2,     0xF1100208,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL3,     0xF110020C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL4,     0xF1100210,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL5,     0xF1100214,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL6,     0xF1100218,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL7,     0xF110021C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL8,     0xF1100220,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL9,     0xF1100224,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL10,    0xF1100228,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL11,    0xF110022C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL12,    0xF1100230,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL13,    0xF1100234,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL14,    0xF1100238,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC0VECTCNTL15,    0xF110023C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG8(     VIC0PERIPHID0,     0xF1100FE0,__READ       );
__IO_REG8_BIT( VIC0PERIPHID1,     0xF1100FE4,__READ       ,__vicperiphid1_bits);
__IO_REG8_BIT( VIC0PERIPHID2,     0xF1100FE8,__READ       ,__vicperiphid2_bits);
__IO_REG8(     VIC0PERIPHID3,     0xF1100FEC,__READ       );
__IO_REG8(     VIC0PCELLID0,      0xF1100FF0,__READ       );
__IO_REG8(     VIC0PCELLID1,      0xF1100FF4,__READ       );
__IO_REG8(     VIC0PCELLID2,      0xF1100FF8,__READ       );
__IO_REG8(     VIC0PCELLID3,      0xF1100FFC,__READ       );

/***************************************************************************
 **
 **  VIC1
 **
 ***************************************************************************/
__IO_REG32_BIT(VIC1IRQSTATUS,     0xF1000000,__READ       ,__vicirqstatus_bits);
__IO_REG32_BIT(VIC1FIQSTATUS,     0xF1000004,__READ       ,__vicfiqstatus_bits);
__IO_REG32_BIT(VIC1RAWINTR,       0xF1000008,__READ       ,__vicrawintr_bits);
__IO_REG32_BIT(VIC1INTSELECT,     0xF100000C,__READ_WRITE ,__vicintselect_bits);
__IO_REG32_BIT(VIC1INTENABLE,     0xF1000010,__READ_WRITE ,__vicintenable_bits);
__IO_REG32_BIT(VIC1INTENCLEAR,    0xF1000014,__WRITE      ,__vicintenclear_bits);
__IO_REG32_BIT(VIC1SOFTINT,       0xF1000018,__READ_WRITE ,__vicsoftint_bits);
__IO_REG32_BIT(VIC1SOFTINTCLEAR,  0xF100001C,__WRITE      ,__vicsoftintclear_bits);
__IO_REG32_BIT(VIC1PROTECTION,    0xF1000020,__READ_WRITE ,__vicprotection_bits);
__IO_REG32(    VIC1VECTADDR,      0xF1000030,__READ_WRITE );
__IO_REG32(    VIC1DEFVECTADDR,   0xF1000034,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR0,     0xF1000100,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR1,     0xF1000104,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR2,     0xF1000108,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR3,     0xF100010C,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR4,     0xF1000110,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR5,     0xF1000114,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR6,     0xF1000118,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR7,     0xF100011C,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR8,     0xF1000120,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR9,     0xF1000124,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR10,    0xF1000128,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR11,    0xF100012C,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR12,    0xF1000130,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR13,    0xF1000134,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR14,    0xF1000138,__READ_WRITE );
__IO_REG32(    VIC1VECTADDR15,    0xF100013C,__READ_WRITE );
__IO_REG32_BIT(VIC1VECTCNTL0,     0xF1000200,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL1,     0xF1000204,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL2,     0xF1000208,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL3,     0xF100020C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL4,     0xF1000210,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL5,     0xF1000214,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL6,     0xF1000218,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL7,     0xF100021C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL8,     0xF1000220,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL9,     0xF1000224,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL10,    0xF1000228,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL11,    0xF100022C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL12,    0xF1000230,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL13,    0xF1000234,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL14,    0xF1000238,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG32_BIT(VIC1VECTCNTL15,    0xF100023C,__READ_WRITE ,__vicvectcntl_bits);
__IO_REG8(     VIC1PERIPHID0,     0xF1000FE0,__READ       );
__IO_REG8_BIT( VIC1PERIPHID1,     0xF1000FE4,__READ       ,__vicperiphid1_bits);
__IO_REG8_BIT( VIC1PERIPHID2,     0xF1000FE8,__READ       ,__vicperiphid2_bits);
__IO_REG8(     VIC1PERIPHID3,     0xF1000FEC,__READ       );
__IO_REG8(     VIC1PCELLID0,      0xF1000FF0,__READ       );
__IO_REG8(     VIC1PCELLID1,      0xF1000FF4,__READ       );
__IO_REG8(     VIC1PCELLID2,      0xF1000FF8,__READ       );
__IO_REG8(     VIC1PCELLID3,      0xF1000FFC,__READ       );

/***************************************************************************
 **
 **  Misc
 **
 ***************************************************************************/
__IO_REG32_BIT(SOC_CFG_CTR,       0xFCA80000,__READ       ,__soc_cfg_ctr_bits);
__IO_REG32_BIT(DIAG_CFG_CTR,      0xFCA80004,__READ_WRITE ,__diag_cfg_ctr_bits);
__IO_REG32_BIT(PLL1_CTR,          0xFCA80008,__READ_WRITE ,__pll_ctr_bits);
__IO_REG32_BIT(PLL1_FRQ,          0xFCA8000C,__READ_WRITE ,__pll_frq_bits);
__IO_REG32_BIT(PLL1_MOD,          0xFCA80010,__READ_WRITE ,__pll_mod_bits);
__IO_REG32_BIT(PLL2_CTR,          0xFCA80014,__READ_WRITE ,__pll_ctr_bits);
__IO_REG32_BIT(PLL2_FRQ,          0xFCA80018,__READ_WRITE ,__pll_frq_bits);
__IO_REG32_BIT(PLL2_MOD,          0xFCA8001C,__READ_WRITE ,__pll_mod_bits);
__IO_REG32_BIT(PLL_CLK_CFG,       0xFCA80020,__READ_WRITE ,__pll_clk_cfg_bits);
__IO_REG32_BIT(CORE_CLK_CFG,      0xFCA80024,__READ_WRITE ,__core_clk_cfg_bits);
__IO_REG32_BIT(PRPH_CLK_CFG,      0xFCA80028,__READ_WRITE ,__prph_clk_cfg_bits);
__IO_REG32_BIT(PERIP1_CLK_ENB,    0xFCA8002C,__READ_WRITE ,__perip1_clk_enb_bits);
__IO_REG32_BIT(RAS_CLK_ENB,       0xFCA80034,__READ_WRITE ,__ras_clk_enb_bits);
__IO_REG32_BIT(PERIP1_SOF_RST,    0xFCA80038,__READ_WRITE ,__perip1_sof_rst_bits);
__IO_REG32_BIT(RAS_SOF_RST,       0xFCA80040,__READ_WRITE ,__ras_sof_rst_bits);
__IO_REG32_BIT(PRSC1_CLK_CFG,     0xFCA80044,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(PRSC2_CLK_CFG,     0xFCA80048,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(PRSC3_CLK_CFG,     0xFCA8004C,__READ_WRITE ,__prsc_clk_cfg_bits);
__IO_REG32_BIT(AMEM_CFG_CTRL,     0xFCA80050,__READ_WRITE ,__amem_cfg_ctrl_bits);
__IO_REG32_BIT(EXPI_CLK_CFG,     	0xFCA80054,__READ_WRITE ,__expi_clk_cfg_bits);
__IO_REG32_BIT(CLCD_CLK_SYNT,     0xFCA8005C,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(IRDA_CLK_SYNT,     0xFCA80060,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(UART_CLK_SYNT,     0xFCA80064,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(MAC_CLK_SYNT,      0xFCA80068,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS1_CLK_SYNT,     0xFCA8006C,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS2_CLK_SYNT,     0xFCA80070,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS3_CLK_SYNT,     0xFCA80074,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(RAS4_CLK_SYNT,     0xFCA80078,__READ_WRITE ,__irda_clk_synt_bits);
__IO_REG32_BIT(ICM1_ARB_CFG,      0xFCA8007C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM2_ARB_CFG,      0xFCA80080,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM3_ARB_CFG,      0xFCA80084,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM4_ARB_CFG,      0xFCA80088,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM5_ARB_CFG,      0xFCA8008C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM6_ARB_CFG,      0xFCA80090,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM7_ARB_CFG,      0xFCA80094,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM8_ARB_CFG,      0xFCA80098,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(ICM9_ARB_CFG,      0xFCA8009C,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(DMA_CHN_CFG,       0xFCA800A0,__READ_WRITE ,__dma_chn_cfg_bits);
__IO_REG32_BIT(USB2_PHY_CFG,      0xFCA800A4,__READ_WRITE ,__usb2_phy_cfg_bits);
__IO_REG32_BIT(GMAC_CFG_CTR,      0xFCA800A8,__READ_WRITE ,__gmac_cfg_ctr_bits);
__IO_REG32_BIT(EXPI_CFG_CTR,      0xFCA800AC,__READ_WRITE ,__expi_cfg_ctr_bits);
__IO_REG32_BIT(ICM10_ARB_CFG,     0xFCA800B0,__READ_WRITE ,__icm_arb_cfg_bits);
__IO_REG32_BIT(PRC1_LOCK_CTR,     0xFCA800C0,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC2_LOCK_CTR,     0xFCA800C4,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC3_LOCK_CTR,     0xFCA800C8,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC4_LOCK_CTR,     0xFCA800CC,__READ_WRITE ,__prc_lock_ctr_bits);
__IO_REG32_BIT(PRC1_IRQ_CTR,      0xFCA800D0,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(PRC2_IRQ_CTR,      0xFCA800D4,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(PRC3_IRQ_CTR,      0xFCA800D8,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(PRC4_IRQ_CTR,      0xFCA800DC,__READ_WRITE ,__prc_irq_ctr_bits);
__IO_REG32_BIT(POWERDOWN_CFG_CTR, 0xFCA800E0,__READ_WRITE ,__powerdown_cfg_ctr_bits);
__IO_REG32_BIT(COMPSSTL_1V8_CFG,  0xFCA800E4,__READ_WRITE ,__compsstl_1v8_cfg_bits);
__IO_REG32_BIT(COMPSSTL_2V5_CFG,  0xFCA800E8,__READ_WRITE ,__compsstl_1v8_cfg_bits);
__IO_REG32_BIT(COMPCOR_3V3_CFG,   0xFCA800EC,__READ_WRITE ,__compcor_3v3_cfg_bits);
__IO_REG32_BIT(SSTLPAD_CFG_CTR,   0xFCA800F0,__READ_WRITE ,__sstlpad_cfg_ctr_bits);
__IO_REG32_BIT(BIST1_CFG_CTR,     0xFCA800F4,__READ_WRITE ,__bist1_cfg_ctr_bits);
__IO_REG32_BIT(BIST2_CFG_CTR,     0xFCA800F8,__READ_WRITE ,__bist2_cfg_ctr_bits);
__IO_REG32_BIT(BIST3_CFG_CTR,     0xFCA800FC,__READ_WRITE ,__bist3_cfg_ctr_bits);
__IO_REG32_BIT(BIST4_CFG_CTR,     0xFCA80100,__READ_WRITE ,__bist4_cfg_ctr_bits);
__IO_REG32_BIT(BIST5_CFG_CTR,     0xFCA80104,__READ_WRITE ,__bist5_cfg_ctr_bits);
__IO_REG32_BIT(BIST1_STS_RES,     0xFCA80108,__READ       ,__bist1_sts_res_bits);
__IO_REG32_BIT(BIST2_STS_RES,     0xFCA8010C,__READ       ,__bist2_sts_res_bits);
__IO_REG32_BIT(BIST3_STS_RES,     0xFCA80110,__READ       ,__bist3_sts_res_bits);
__IO_REG32_BIT(BIST4_STS_RES,     0xFCA80114,__READ       ,__bist4_sts_res_bits);
__IO_REG32_BIT(BIST5_STS_RES,     0xFCA80118,__READ       ,__bist5_sts_res_bits);
__IO_REG32_BIT(SYSERR_CFG_CTR,    0xFCA8011C,__READ_WRITE ,__syserr_cfg_ctr_bits);

/***************************************************************************
 **
 **  SC
 **
 ***************************************************************************/
__IO_REG32_BIT(SCCTRL,            0xFCA00000,__READ_WRITE ,__scctrl_bits);
__IO_REG32(    SCSYSSTAT,         0xFCA00004,__WRITE      );
__IO_REG32_BIT(SCIMCTRL,          0xFCA00008,__READ_WRITE ,__scimctrl_bits);
__IO_REG32_BIT(SCIMSTAT,          0xFCA0000C,__READ_WRITE ,__scimstat_bits);
__IO_REG32_BIT(SCXTALCTRL,        0xFCA00010,__READ_WRITE ,__scxtalctrl_bits);
__IO_REG32_BIT(SCPLLCTRL,         0xFCA00014,__READ_WRITE ,__scpllctrl_bits);
__IO_REG32(    SCSYSID0,          0xFCA00EE0,__READ       );
__IO_REG32(    SCSYSID1,          0xFCA00EE4,__READ       );
__IO_REG32(    SCSYSID2,          0xFCA00EE8,__READ       );
__IO_REG32(    SCSYSID3,          0xFCA00EEC,__READ       );
__IO_REG32(    SCPeriphID0,       0xFCA00FE0,__READ       );
__IO_REG32(    SCPeriphID1,       0xFCA00FE4,__READ       );
__IO_REG32(    SCPeriphID2,       0xFCA00FE8,__READ       );
__IO_REG32(    SCPeriphID3,       0xFCA00FEC,__READ       );
__IO_REG32(    SCPCellID0,        0xFCA00FF0,__READ       );
__IO_REG32(    SCPCellID1,        0xFCA00FF4,__READ       );
__IO_REG32(    SCPCellID2,        0xFCA00FF8,__READ       );
__IO_REG32(    SCPCellID3,        0xFCA00FFC,__READ       );

/***************************************************************************
 **
 **  WDT
 **
 ***************************************************************************/
__IO_REG32(    WdogLoad,          0xFC880000,__READ_WRITE );
__IO_REG32(    WdogValue,         0xFC880004,__READ       );
__IO_REG32_BIT(WdogControl,       0xFC880008,__READ_WRITE ,__wdogcontrol_bits);
__IO_REG32(    WdogIntClr,        0xFC88000C,__WRITE      );
__IO_REG32_BIT(WdogRIS,           0xFC880010,__READ       ,__wdogris_bits);
__IO_REG32_BIT(WdogMIS,           0xFC880018,__READ       ,__wdogmis_bits);
__IO_REG32(    WdogLock,          0xFC880C00,__READ_WRITE );
__IO_REG32(    WdogPeriphID0,     0xFC880FE0,__READ       );
__IO_REG32(    WdogPeriphID1,     0xFC880FE4,__READ       );
__IO_REG32(    WdogPeriphID2,     0xFC880FE8,__READ       );
__IO_REG32(    WdogPeriphID3,     0xFC880FEC,__READ       );
__IO_REG32(    WdogPCellID0,      0xFC880FF0,__READ       );
__IO_REG32(    WdogPCellID1,      0xFC880FF4,__READ       );
__IO_REG32(    WdogPCellID2,      0xFC880FF8,__READ       );
__IO_REG32(    WdogPCellID3,      0xFC880FFC,__READ       );

/***************************************************************************
 **
 **  TIMER ML1/2
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMERML_CONTROL1,       0xF0000080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMERML_STATUS_INT_ACK1,0xF0000084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMERML_COMPARE1,       0xF0000088,__READ_WRITE );
__IO_REG16(    TIMERML_COUNT1,         0xF000008C,__READ       );
__IO_REG16(    TIMERML_REDG_CAPT1,     0xF0000090,__READ       );
__IO_REG16(    TIMERML_FEDG_CAPT1,     0xF0000094,__READ       );
__IO_REG16_BIT(TIMERML_CONTROL2,       0xF0000100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMERML_STATUS_INT_ACK2,0xF0000104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMERML_COMPARE2,       0xF0000108,__READ_WRITE );
__IO_REG16(    TIMERML_COUNT2,         0xF000010C,__READ       );
__IO_REG16(    TIMERML_REDG_CAPT2,     0xF0000110,__READ       );
__IO_REG16(    TIMERML_FEDG_CAPT2,     0xF0000114,__READ       );

/***************************************************************************
 **
 **  TIMER Basic
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMERB_CONTROL1,       0xFC800080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMERB_STATUS_INT_ACK1,0xFC800084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMERB_COMPARE1,       0xFC800088,__READ_WRITE );
__IO_REG16(    TIMERB_COUNT1,         0xFC80008C,__READ       );
__IO_REG16(    TIMERB_REDG_CAPT1,     0xFC800090,__READ       );
__IO_REG16(    TIMERB_FEDG_CAPT1,     0xFC800094,__READ       );
__IO_REG16_BIT(TIMERB_CONTROL2,       0xFC800100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMERB_STATUS_INT_ACK2,0xFC800104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMERB_COMPARE2,       0xFC800108,__READ_WRITE );
__IO_REG16(    TIMERB_COUNT2,         0xFC80010C,__READ       );
__IO_REG16(    TIMERB_REDG_CAPT2,     0xFC800110,__READ       );
__IO_REG16(    TIMERB_FEDG_CAPT2,     0xFC800114,__READ       );

/***************************************************************************
 **
 **  TIMER1
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMER1_CONTROL1,       0xD8000080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER1_STATUS_INT_ACK1,0xD8000084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER1_COMPARE1,       0xD8000088,__READ_WRITE );
__IO_REG16(    TIMER1_COUNT1,         0xD800008C,__READ       );
__IO_REG16(    TIMER1_REDG_CAPT1,     0xD8000090,__READ       );
__IO_REG16(    TIMER1_FEDG_CAPT1,     0xD8000094,__READ       );
__IO_REG16_BIT(TIMER1_CONTROL2,       0xD8000100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER1_STATUS_INT_ACK2,0xD8000104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER1_COMPARE2,       0xD8000108,__READ_WRITE );
__IO_REG16(    TIMER1_COUNT2,         0xD800010C,__READ       );
__IO_REG16(    TIMER1_REDG_CAPT2,     0xD8000110,__READ       );
__IO_REG16(    TIMER1_FEDG_CAPT2,     0xD8000114,__READ       );

/***************************************************************************
 **
 **  TIMER2
 **
 ***************************************************************************/
__IO_REG16_BIT(TIMER2_CONTROL1,       0xD8080080,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER2_STATUS_INT_ACK1,0xD8080084,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER2_COMPARE1,       0xD8080088,__READ_WRITE );
__IO_REG16(    TIMER2_COUNT1,         0xD808008C,__READ       );
__IO_REG16(    TIMER2_REDG_CAPT1,     0xD8080090,__READ       );
__IO_REG16(    TIMER2_FEDG_CAPT1,     0xD8080094,__READ       );
__IO_REG16_BIT(TIMER2_CONTROL2,       0xD8080100,__READ_WRITE ,__timer_control_bits);
__IO_REG16_BIT(TIMER2_STATUS_INT_ACK2,0xD8080104,__READ_WRITE ,__timer_status_int_ack_bits);
__IO_REG16(    TIMER2_COMPARE2,       0xD8080108,__READ_WRITE );
__IO_REG16(    TIMER2_COUNT2,         0xD808010C,__READ       );
__IO_REG16(    TIMER2_REDG_CAPT2,     0xD8080110,__READ       );
__IO_REG16(    TIMER2_FEDG_CAPT2,     0xD8080114,__READ       );

/***************************************************************************
 **
 **  MPMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MEM0_CTL,          0xFC600000,__READ_WRITE ,__mem0_ctl_bits);
__IO_REG32_BIT(MEM1_CTL,          0xFC600004,__READ_WRITE ,__mem1_ctl_bits);
__IO_REG32_BIT(MEM2_CTL,          0xFC600008,__READ_WRITE ,__mem2_ctl_bits);
__IO_REG32_BIT(MEM3_CTL,          0xFC60000C,__READ_WRITE ,__mem3_ctl_bits);
__IO_REG32_BIT(MEM4_CTL,          0xFC600010,__READ_WRITE ,__mem4_ctl_bits);
__IO_REG32_BIT(MEM5_CTL,          0xFC600014,__READ_WRITE ,__mem5_ctl_bits);
__IO_REG32_BIT(MEM6_CTL,          0xFC600018,__READ_WRITE ,__mem6_ctl_bits);
__IO_REG32_BIT(MEM7_CTL,          0xFC60001C,__READ_WRITE ,__mem7_ctl_bits);
__IO_REG32_BIT(MEM8_CTL,          0xFC600020,__READ_WRITE ,__mem8_ctl_bits);
__IO_REG32_BIT(MEM9_CTL,          0xFC600024,__READ_WRITE ,__mem9_ctl_bits);
__IO_REG32_BIT(MEM10_CTL,         0xFC600028,__READ_WRITE ,__mem10_ctl_bits);
__IO_REG32_BIT(MEM11_CTL,         0xFC60002C,__READ_WRITE ,__mem11_ctl_bits);
__IO_REG32_BIT(MEM12_CTL,         0xFC600030,__READ_WRITE ,__mem12_ctl_bits);
__IO_REG32_BIT(MEM13_CTL,         0xFC600034,__READ_WRITE ,__mem13_ctl_bits);
__IO_REG32_BIT(MEM14_CTL,         0xFC600038,__READ_WRITE ,__mem14_ctl_bits);
__IO_REG32_BIT(MEM15_CTL,         0xFC60003C,__READ_WRITE ,__mem15_ctl_bits);
__IO_REG32_BIT(MEM16_CTL,         0xFC600040,__READ_WRITE ,__mem16_ctl_bits);
__IO_REG32_BIT(MEM17_CTL,         0xFC600044,__READ_WRITE ,__mem17_ctl_bits);
__IO_REG32_BIT(MEM18_CTL,         0xFC600048,__READ_WRITE ,__mem18_ctl_bits);
__IO_REG32_BIT(MEM19_CTL,         0xFC60004C,__READ_WRITE ,__mem19_ctl_bits);
__IO_REG32_BIT(MEM20_CTL,         0xFC600050,__READ_WRITE ,__mem20_ctl_bits);
__IO_REG32_BIT(MEM21_CTL,         0xFC600054,__READ_WRITE ,__mem21_ctl_bits);
__IO_REG32_BIT(MEM22_CTL,         0xFC600058,__READ_WRITE ,__mem22_ctl_bits);
__IO_REG32_BIT(MEM23_CTL,         0xFC60005C,__READ_WRITE ,__mem23_ctl_bits);
__IO_REG32_BIT(MEM24_CTL,         0xFC600060,__READ_WRITE ,__mem24_ctl_bits);
__IO_REG32_BIT(MEM25_CTL,         0xFC600064,__READ_WRITE ,__mem25_ctl_bits);
__IO_REG32_BIT(MEM26_CTL,         0xFC600068,__READ_WRITE ,__mem26_ctl_bits);
__IO_REG32_BIT(MEM27_CTL,         0xFC60006C,__READ_WRITE ,__mem27_ctl_bits);
__IO_REG32_BIT(MEM28_CTL,         0xFC600070,__READ_WRITE ,__mem28_ctl_bits);
__IO_REG32_BIT(MEM29_CTL,         0xFC600074,__READ_WRITE ,__mem29_ctl_bits);
__IO_REG32_BIT(MEM30_CTL,         0xFC600078,__READ_WRITE ,__mem30_ctl_bits);
__IO_REG32_BIT(MEM31_CTL,         0xFC60007C,__READ_WRITE ,__mem31_ctl_bits);
__IO_REG32_BIT(MEM32_CTL,         0xFC600080,__READ_WRITE ,__mem32_ctl_bits);
__IO_REG32_BIT(MEM33_CTL,         0xFC600084,__READ_WRITE ,__mem33_ctl_bits);
__IO_REG32_BIT(MEM34_CTL,         0xFC600088,__READ_WRITE ,__mem34_ctl_bits);
__IO_REG32_BIT(MEM35_CTL,         0xFC60008C,__READ_WRITE ,__mem35_ctl_bits);
__IO_REG32_BIT(MEM36_CTL,         0xFC600090,__READ_WRITE ,__mem36_ctl_bits);
__IO_REG32_BIT(MEM37_CTL,         0xFC600094,__READ_WRITE ,__mem37_ctl_bits);
__IO_REG32_BIT(MEM38_CTL,         0xFC600098,__READ_WRITE ,__mem38_ctl_bits);
__IO_REG32_BIT(MEM39_CTL,         0xFC60009C,__READ_WRITE ,__mem39_ctl_bits);
__IO_REG32_BIT(MEM40_CTL,         0xFC6000A0,__READ_WRITE ,__mem40_ctl_bits);
__IO_REG32_BIT(MEM41_CTL,         0xFC6000A4,__READ_WRITE ,__mem41_ctl_bits);
__IO_REG32_BIT(MEM42_CTL,         0xFC6000A8,__READ_WRITE ,__mem42_ctl_bits);
__IO_REG32_BIT(MEM43_CTL,         0xFC6000AC,__READ_WRITE ,__mem43_ctl_bits);
__IO_REG32_BIT(MEM44_CTL,         0xFC6000B0,__READ_WRITE ,__mem44_ctl_bits);
__IO_REG32_BIT(MEM45_CTL,         0xFC6000B4,__READ_WRITE ,__mem45_ctl_bits);
__IO_REG32_BIT(MEM46_CTL,         0xFC6000B8,__READ_WRITE ,__mem46_ctl_bits);
__IO_REG32_BIT(MEM47_CTL,         0xFC6000BC,__READ_WRITE ,__mem47_ctl_bits);
__IO_REG32_BIT(MEM48_CTL,         0xFC6000C0,__READ_WRITE ,__mem48_ctl_bits);
__IO_REG32_BIT(MEM49_CTL,         0xFC6000C4,__READ_WRITE ,__mem49_ctl_bits);
__IO_REG32_BIT(MEM50_CTL,         0xFC6000C8,__READ_WRITE ,__mem50_ctl_bits);
__IO_REG32_BIT(MEM51_CTL,         0xFC6000CC,__READ_WRITE ,__mem51_ctl_bits);
__IO_REG32_BIT(MEM52_CTL,         0xFC6000D0,__READ_WRITE ,__mem52_ctl_bits);
__IO_REG32_BIT(MEM53_CTL,         0xFC6000D4,__READ_WRITE ,__mem52_ctl_bits);
__IO_REG32_BIT(MEM54_CTL,         0xFC6000D8,__READ_WRITE ,__mem54_ctl_bits);
__IO_REG32_BIT(MEM55_CTL,         0xFC6000DC,__READ_WRITE ,__mem55_ctl_bits);
__IO_REG32_BIT(MEM56_CTL,         0xFC6000E0,__READ_WRITE ,__mem56_ctl_bits);
__IO_REG32_BIT(MEM57_CTL,         0xFC6000E4,__READ_WRITE ,__mem57_ctl_bits);
__IO_REG32_BIT(MEM58_CTL,         0xFC6000E8,__READ_WRITE ,__mem58_ctl_bits);
__IO_REG32_BIT(MEM59_CTL,         0xFC6000EC,__READ_WRITE ,__mem59_ctl_bits);
__IO_REG32(    MEM60_CTL,         0xFC6000F0,__READ_WRITE );
__IO_REG32_BIT(MEM61_CTL,         0xFC6000F4,__READ_WRITE ,__mem61_ctl_bits);
__IO_REG32_BIT(MEM62_CTL,         0xFC6000F8,__READ_WRITE ,__mem62_ctl_bits);
__IO_REG32_BIT(MEM63_CTL,         0xFC6000FC,__READ_WRITE ,__mem63_ctl_bits);
__IO_REG32_BIT(MEM64_CTL,         0xFC600100,__READ_WRITE ,__mem64_ctl_bits);
__IO_REG32_BIT(MEM65_CTL,         0xFC600104,__READ_WRITE ,__mem65_ctl_bits);
__IO_REG32_BIT(MEM66_CTL,         0xFC600108,__READ_WRITE ,__mem66_ctl_bits);
__IO_REG32_BIT(MEM67_CTL,         0xFC60010C,__READ_WRITE ,__mem67_ctl_bits);
__IO_REG32_BIT(MEM68_CTL,         0xFC600110,__READ_WRITE ,__mem68_ctl_bits);
__IO_REG32_BIT(MEM69_CTL,         0xFC600114,__READ_WRITE ,__mem69_ctl_bits);
__IO_REG32_BIT(MEM70_CTL,         0xFC600118,__READ_WRITE ,__mem70_ctl_bits);
__IO_REG32_BIT(MEM71_CTL,         0xFC60011C,__READ_WRITE ,__mem71_ctl_bits);
__IO_REG32_BIT(MEM72_CTL,         0xFC600120,__READ_WRITE ,__mem72_ctl_bits);
__IO_REG32_BIT(MEM73_CTL,         0xFC600124,__READ_WRITE ,__mem73_ctl_bits);
__IO_REG32_BIT(MEM74_CTL,         0xFC600128,__READ_WRITE ,__mem74_ctl_bits);
__IO_REG32_BIT(MEM75_CTL,         0xFC60012C,__READ_WRITE ,__mem75_ctl_bits);
__IO_REG32_BIT(MEM76_CTL,         0xFC600130,__READ_WRITE ,__mem76_ctl_bits);
__IO_REG32_BIT(MEM77_CTL,         0xFC600134,__READ_WRITE ,__mem77_ctl_bits);
__IO_REG32_BIT(MEM78_CTL,         0xFC600138,__READ_WRITE ,__mem78_ctl_bits);
__IO_REG32_BIT(MEM79_CTL,         0xFC60013C,__READ_WRITE ,__mem79_ctl_bits);
__IO_REG32_BIT(MEM80_CTL,         0xFC600140,__READ_WRITE ,__mem80_ctl_bits);
__IO_REG32_BIT(MEM81_CTL,         0xFC600144,__READ_WRITE ,__mem81_ctl_bits);
__IO_REG32_BIT(MEM82_CTL,         0xFC600148,__READ_WRITE ,__mem82_ctl_bits);
__IO_REG32_BIT(MEM83_CTL,         0xFC60014C,__READ_WRITE ,__mem83_ctl_bits);
__IO_REG32_BIT(MEM84_CTL,         0xFC600150,__READ_WRITE ,__mem84_ctl_bits);
__IO_REG32_BIT(MEM85_CTL,         0xFC600154,__READ_WRITE ,__mem85_ctl_bits);
__IO_REG32_BIT(MEM86_CTL,         0xFC600158,__READ_WRITE ,__mem86_ctl_bits);
__IO_REG32_BIT(MEM87_CTL,         0xFC60015C,__READ_WRITE ,__mem87_ctl_bits);
__IO_REG32_BIT(MEM88_CTL,         0xFC600160,__READ_WRITE ,__mem88_ctl_bits);
__IO_REG32_BIT(MEM89_CTL,         0xFC600164,__READ_WRITE ,__mem89_ctl_bits);
__IO_REG32_BIT(MEM90_CTL,         0xFC600168,__READ_WRITE ,__mem90_ctl_bits);
__IO_REG32_BIT(MEM91_CTL,         0xFC60016C,__READ_WRITE ,__mem91_ctl_bits);
__IO_REG32_BIT(MEM92_CTL,         0xFC600170,__READ_WRITE ,__mem92_ctl_bits);
__IO_REG32_BIT(MEM93_CTL,         0xFC600174,__READ_WRITE ,__mem93_ctl_bits);
__IO_REG32_BIT(MEM94_CTL,         0xFC600178,__READ_WRITE ,__mem94_ctl_bits);
__IO_REG32_BIT(MEM95_CTL,         0xFC60017C,__READ_WRITE ,__mem95_ctl_bits);
__IO_REG32_BIT(MEM96_CTL,         0xFC600180,__READ_WRITE ,__mem96_ctl_bits);
__IO_REG32(		 MEM97_CTL,         0xFC600184,__READ_WRITE );
__IO_REG32(		 MEM98_CTL,         0xFC600188,__READ_WRITE );
__IO_REG32(		 MEM99_CTL,         0xFC60018C,__READ_WRITE );

/***************************************************************************
 **
 **  SMI
 **
 ***************************************************************************/
__IO_REG32_BIT(SMI_CR1,           0xFC000000,__READ_WRITE ,__smi_cr1_bits);
__IO_REG32_BIT(SMI_CR2,           0xFC000004,__READ_WRITE ,__smi_cr2_bits);
__IO_REG32_BIT(SMI_SR,            0xFC000008,__READ_WRITE ,__smi_sr_bits);
__IO_REG32(    SMI_TR,            0xFC00000C,__READ_WRITE );
__IO_REG32(    SMI_RR,            0xFC000010,__READ_WRITE );

/***************************************************************************
 **
 **  FSMC
 **
 ***************************************************************************/
__IO_REG32_BIT(GenMemCtrl_PC0,        0xD1800040,__READ_WRITE ,__genmemctrl_pc_bits);
__IO_REG32_BIT(GenMemCtrl_Comm0,      0xD1800048,__READ_WRITE ,__genmemctrl_comm_bits);
__IO_REG32_BIT(GenMemCtrl_Attrib0,    0xD180004C,__READ_WRITE ,__genmemctrl_comm_bits);
__IO_REG32_BIT(GenMemCtrl_ECCr0,      0xD1800054,__READ       ,__genmemctrl_eccr_bits);
__IO_REG32(    GenMemCtrl_PeriphID0,  0xD1800FE0,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID1,  0xD1800FE4,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID2,  0xD1800FE8,__READ       );
__IO_REG32(    GenMemCtrl_PeriphID3,  0xD1800FEC,__READ       );
__IO_REG32(    GenMemCtrl_PCellID0,   0xD1800FF0,__READ       );
__IO_REG32(    GenMemCtrl_PCellID1,   0xD1800FF4,__READ       );
__IO_REG32(    GenMemCtrl_PCellID2,   0xD1800FF8,__READ       );
__IO_REG32(    GenMemCtrl_PCellID3,   0xD1800FFC,__READ       );

/***************************************************************************
 **
 **  GMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(GMAC_DMA_R0,        		0xE0801000,__READ_WRITE ,__gmac_dma_r0_bits);
__IO_REG32(		 GMAC_DMA_R1,        		0xE0801004,__READ_WRITE );
__IO_REG32(	   GMAC_DMA_R2,        		0xE0801008,__READ_WRITE );
__IO_REG32(		 GMAC_DMA_R3,        		0xE080100C,__READ_WRITE );
__IO_REG32(		 GMAC_DMA_R4,        		0xE0801010,__READ_WRITE );
__IO_REG32_BIT(GMAC_DMA_R5,        		0xE0801014,__READ_WRITE ,__gmac_dma_r5_bits);
__IO_REG32_BIT(GMAC_DMA_R6,        		0xE0801018,__READ_WRITE ,__gmac_dma_r6_bits);
__IO_REG32_BIT(GMAC_DMA_R7,        		0xE080101C,__READ_WRITE ,__gmac_dma_r7_bits);
__IO_REG32_BIT(GMAC_DMA_R8,        		0xE0801020,__READ_WRITE ,__gmac_dma_r8_bits);
__IO_REG32(		 GMAC_DMA_R18,        	0xE0801048,__READ_WRITE );
__IO_REG32(		 GMAC_DMA_R19,        	0xE080104C,__READ_WRITE );
__IO_REG32(		 GMAC_DMA_R20,        	0xE0801050,__READ_WRITE );
__IO_REG32(		 GMAC_DMA_R21,        	0xE0801054,__READ_WRITE );
__IO_REG32_BIT(GMAC_R0,        				0xE0800000,__READ_WRITE ,__gmac_r0_bits);
__IO_REG32_BIT(GMAC_R1,        				0xE0800004,__READ_WRITE ,__gmac_r1_bits);
__IO_REG32(		 GMAC_R2,        				0xE0800008,__READ_WRITE );
__IO_REG32(		 GMAC_R3,        				0xE080000C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R4,        				0xE0800010,__READ_WRITE ,__gmac_r4_bits);
__IO_REG32_BIT(GMAC_R5,        				0xE0800014,__READ_WRITE ,__gmac_r5_bits);
__IO_REG32_BIT(GMAC_R6,        				0xE0800018,__READ_WRITE ,__gmac_r6_bits);
__IO_REG32_BIT(GMAC_R7,        				0xE080001C,__READ_WRITE ,__gmac_r7_bits);
__IO_REG32(		 GMAC_R8,        				0xE0800020,__READ				);
__IO_REG32(		 GMAC_R10,        			0xE0800028,__READ_WRITE );
__IO_REG32_BIT(GMAC_R11,        			0xE080002C,__READ_WRITE ,__gmac_r11_bits);
__IO_REG32_BIT(GMAC_R14,        			0xE0800038,__READ_WRITE ,__gmac_r14_bits);
__IO_REG32_BIT(GMAC_R15,        			0xE080003C,__READ_WRITE ,__gmac_r15_bits);
__IO_REG16(		 GMAC_R16,        			0xE0800030,__READ_WRITE );
__IO_REG32(		 GMAC_R17,        			0xE0800034,__READ_WRITE );
__IO_REG32_BIT(GMAC_R18,        			0xE0800048,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R19,        			0xE080004C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R20,        			0xE0800050,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R21,        			0xE0800054,__READ_WRITE );
__IO_REG32_BIT(GMAC_R22,        			0xE0800058,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R23,        			0xE080005C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R24,        			0xE0800060,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R25,        			0xE0800064,__READ_WRITE );
__IO_REG32_BIT(GMAC_R26,        			0xE0800068,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R27,        			0xE080006C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R28,        			0xE0800070,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R29,        			0xE0800074,__READ_WRITE );
__IO_REG32_BIT(GMAC_R30,        			0xE0800078,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R31,        			0xE080007C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R32,        			0xE0800080,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R33,        			0xE0800084,__READ_WRITE );
__IO_REG32_BIT(GMAC_R34,        			0xE0800088,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R35,        			0xE080008C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R36,        			0xE0800090,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(	 	 GMAC_R37,        			0xE0800094,__READ_WRITE );
__IO_REG32_BIT(GMAC_R38,        			0xE0800098,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R39,        			0xE080009C,__READ_WRITE );
__IO_REG32_BIT(GMAC_R40,        			0xE08000A0,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R41,        			0xE08000A4,__READ_WRITE );
__IO_REG32_BIT(GMAC_R42,        			0xE08000A8,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R43,        			0xE08000AC,__READ_WRITE );
__IO_REG32_BIT(GMAC_R44,        			0xE08000B0,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R45,        			0xE08000B4,__READ_WRITE );
__IO_REG32_BIT(GMAC_R46,        			0xE08000B8,__READ_WRITE ,__gmac_r18_bits);
__IO_REG32(		 GMAC_R47,        			0xE08000BC,__READ_WRITE );
__IO_REG32_BIT(GMAC_R48,        			0xE08000C0,__READ_WRITE ,__gmac_r48_bits);
__IO_REG32_BIT(GMAC_R49,        			0xE08000C4,__READ_WRITE ,__gmac_r49_bits);
__IO_REG32_BIT(GMAC_R50,        			0xE08000C8,__READ_WRITE ,__gmac_r50_bits);
__IO_REG32_BIT(GMAC_R51,        			0xE08000CC,__READ_WRITE ,__gmac_r51_bits);
__IO_REG32_BIT(GMAC_R52,        			0xE08000D0,__READ_WRITE ,__gmac_r52_bits);
__IO_REG32_BIT(GMAC_R64,        			0xE0800100,__READ_WRITE ,__gmac_r64_bits);
__IO_REG32_BIT(GMAC_R65,        			0xE0800104,__READ_WRITE ,__gmac_r65_bits);
__IO_REG32_BIT(GMAC_R66,        			0xE0800108,__READ_WRITE ,__gmac_r66_bits);
__IO_REG32_BIT(GMAC_R67,        			0xE080010C,__READ_WRITE ,__gmac_r65_bits);
__IO_REG32_BIT(GMAC_R68,        			0xE0800110,__READ_WRITE ,__gmac_r66_bits);
__IO_REG32(		 GMAC_R69,        			0xE0800114,__READ_WRITE );
__IO_REG32(		 GMAC_R70,        			0xE0800118,__READ_WRITE );
__IO_REG32(		 GMAC_R71,        			0xE080011C,__READ_WRITE );
__IO_REG32(		 GMAC_R72,        			0xE0800120,__READ_WRITE );
__IO_REG32(		 GMAC_R73,        			0xE0800124,__READ_WRITE );
__IO_REG32(		 GMAC_R74,        			0xE0800128,__READ_WRITE );
__IO_REG32(		 GMAC_R75,        			0xE080012C,__READ_WRITE );
__IO_REG32(		 GMAC_R76,        			0xE0800130,__READ_WRITE );
__IO_REG32(		 GMAC_R77,        			0xE0800134,__READ_WRITE );
__IO_REG32(		 GMAC_R78,        			0xE0800138,__READ_WRITE );
__IO_REG32(		 GMAC_R79,        			0xE080013C,__READ_WRITE );
__IO_REG32(		 GMAC_R80,        			0xE0800140,__READ_WRITE );
__IO_REG32(		 GMAC_R81,        			0xE0800144,__READ_WRITE );
__IO_REG32(		 GMAC_R82,        			0xE0800148,__READ_WRITE );
__IO_REG32(		 GMAC_R83,        			0xE080014C,__READ_WRITE );
__IO_REG32(		 GMAC_R84,        			0xE0800150,__READ_WRITE );
__IO_REG32(		 GMAC_R85,        			0xE0800154,__READ_WRITE );
__IO_REG32(		 GMAC_R86,        			0xE0800158,__READ_WRITE );
__IO_REG32(		 GMAC_R87,        			0xE080015C,__READ_WRITE );
__IO_REG32(		 GMAC_R88,        			0xE0800160,__READ_WRITE );
__IO_REG32(		 GMAC_R89,        			0xE0800164,__READ_WRITE );
__IO_REG32(		 GMAC_R90,        			0xE0800168,__READ_WRITE );
__IO_REG32(		 GMAC_R91,        			0xE080016C,__READ_WRITE );
__IO_REG32(		 GMAC_R92,        			0xE0800170,__READ_WRITE );
__IO_REG32(		 GMAC_R93,        			0xE0800174,__READ_WRITE );
__IO_REG32(		 GMAC_R96,        			0xE0800180,__READ_WRITE );
__IO_REG32(		 GMAC_R97,        			0xE0800184,__READ_WRITE );
__IO_REG32(		 GMAC_R98,        			0xE0800188,__READ_WRITE );
__IO_REG32(		 GMAC_R99,        			0xE080018C,__READ_WRITE );
__IO_REG32(		 GMAC_R100,        			0xE0800190,__READ_WRITE );
__IO_REG32(		 GMAC_R101,        			0xE0800194,__READ_WRITE );
__IO_REG32(		 GMAC_R102,        			0xE0800198,__READ_WRITE );
__IO_REG32(		 GMAC_R103,        			0xE080019C,__READ_WRITE );
__IO_REG32(		 GMAC_R104,        			0xE08001A0,__READ_WRITE );
__IO_REG32(		 GMAC_R105,        			0xE08001A4,__READ_WRITE );
__IO_REG32(		 GMAC_R106,        			0xE08001A8,__READ_WRITE );
__IO_REG32(		 GMAC_R107,        			0xE08001AC,__READ_WRITE );
__IO_REG32(		 GMAC_R108,        			0xE08001B0,__READ_WRITE );
__IO_REG32(		 GMAC_R109,        			0xE08001B4,__READ_WRITE );
__IO_REG32(		 GMAC_R110,        			0xE08001B8,__READ_WRITE );
__IO_REG32(		 GMAC_R111,        			0xE08001BC,__READ_WRITE );
__IO_REG32(		 GMAC_R112,        			0xE08001C0,__READ_WRITE );
__IO_REG32(		 GMAC_R113,        			0xE08001C4,__READ_WRITE );
__IO_REG32(		 GMAC_R114,        			0xE08001C8,__READ_WRITE );
__IO_REG32(		 GMAC_R115,        			0xE08001CC,__READ_WRITE );
__IO_REG32(		 GMAC_R116,        			0xE08001D0,__READ_WRITE );
__IO_REG32(		 GMAC_R117,        			0xE08001D4,__READ_WRITE );
__IO_REG32(		 GMAC_R118,        			0xE08001D8,__READ_WRITE );
__IO_REG32(		 GMAC_R119,        			0xE08001DC,__READ_WRITE );

/***************************************************************************
 **
 **  EHCI1
 **
 ***************************************************************************/
__IO_REG32_BIT(HC1CAPBASE,            0xE1800000,__READ       ,__hccapbase_bits);
__IO_REG32_BIT(HC1SPARAMS,            0xE1800004,__READ       ,__hcsparams_bits);
__IO_REG32_BIT(HC1CPARAMS,            0xE1800008,__READ       ,__hccparams_bits);
__IO_REG32_BIT(HC1USBCMD,             0xE1800010,__READ       ,__hcusbcmd_bits);
__IO_REG32_BIT(HC1USBSTS,             0xE1800014,__READ_WRITE ,__hcusbsts_bits);
__IO_REG32_BIT(HC1USBINTR,            0xE1800018,__READ_WRITE ,__hcusbintr_bits);
__IO_REG32_BIT(HC1FRINDEX,            0xE180001C,__READ_WRITE ,__hcfrindex_bits);
__IO_REG32(    HC1PERIODICLISTBASE,   0xE1800024,__READ_WRITE );
__IO_REG32(    HC1ASYNCLISTADDR,      0xE1800028,__READ_WRITE );
__IO_REG32_BIT(HC1CONFIGFLAG,         0xE1800050,__READ_WRITE ,__hcconfigflag_bits);
__IO_REG32_BIT(HC1PORTSC1,            0xE1800054,__READ_WRITE ,__hcportsc_bits);
__IO_REG32_BIT(HC1INSNREG00,          0xE1800090,__READ_WRITE ,__hcinsnreg00_bits);
__IO_REG32_BIT(HC1INSNREG01,          0xE1800094,__READ_WRITE ,__hcinsnreg01_bits);
__IO_REG32_BIT(HC1INSNREG02,          0xE1800098,__READ_WRITE ,__hcinsnreg02_bits);
__IO_REG32_BIT(HC1INSNREG03,          0xE180009C,__READ_WRITE ,__hcinsnreg03_bits);
__IO_REG32_BIT(HC1INSNREG05,          0xE18000A4,__READ_WRITE ,__hcinsnreg05_bits);

/***************************************************************************
 **
 **  OCHI1
 **
 ***************************************************************************/
__IO_REG32_BIT(Hc1Revision,           0xE1900000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(Hc1Control,            0xE1900004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(Hc1CommandStatus,      0xE1900008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(Hc1InterruptStatus,    0xE190000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(Hc1InterruptEnable,    0xE1900010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc1InterruptDisable,   0xE1900014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc1HCCA,               0xE1900018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(Hc1PeriodCurrentED,    0xE190001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(Hc1ControlHeadED,      0xE1900020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(Hc1ControlCurrentED,   0xE1900024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(Hc1BulkHeadED,         0xE1900028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(Hc1BulkCurrentED,      0xE190002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(Hc1DoneHead,           0xE1900030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(Hc1FmInterval,         0xE1900034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(Hc1FmRemaining,        0xE1900038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(Hc1FmNumber,           0xE190003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(Hc1PeriodStart,        0xE1900040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(Hc1LSThreshold,        0xE1900044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(Hc1RhDescriptorA,      0xE1900048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(Hc1RhDescripterB,      0xE190004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(Hc1RhStatus,           0xE1900050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(Hc1RhPortStatus,       0xE1900054,__READ_WRITE ,__hcrhportstatus_bits);

/***************************************************************************
 **
 **  EHCI2
 **
 ***************************************************************************/
__IO_REG32_BIT(HC2CAPBASE,            0xE2000000,__READ       ,__hccapbase_bits);
__IO_REG32_BIT(HC2SPARAMS,            0xE2000004,__READ       ,__hcsparams_bits);
__IO_REG32_BIT(HC2CPARAMS,            0xE2000008,__READ       ,__hccparams_bits);
__IO_REG32_BIT(HC2USBCMD,             0xE2000010,__READ       ,__hcusbcmd_bits);
__IO_REG32_BIT(HC2USBSTS,             0xE2000014,__READ_WRITE ,__hcusbsts_bits);
__IO_REG32_BIT(HC2USBINTR,            0xE2000018,__READ_WRITE ,__hcusbintr_bits);
__IO_REG32_BIT(HC2FRINDEX,            0xE200001C,__READ_WRITE ,__hcfrindex_bits);
__IO_REG32(    HC2PERIODICLISTBASE,   0xE2000024,__READ_WRITE );
__IO_REG32(    HC2ASYNCLISTADDR,      0xE2000028,__READ_WRITE );
__IO_REG32_BIT(HC2CONFIGFLAG,         0xE2000050,__READ_WRITE ,__hcconfigflag_bits);
__IO_REG32_BIT(HC2PORTSC1,            0xE2000054,__READ_WRITE ,__hcportsc_bits);
__IO_REG32_BIT(HC2INSNREG00,          0xE2000090,__READ_WRITE ,__hcinsnreg00_bits);
__IO_REG32_BIT(HC2INSNREG01,          0xE2000094,__READ_WRITE ,__hcinsnreg01_bits);
__IO_REG32_BIT(HC2INSNREG02,          0xE2000098,__READ_WRITE ,__hcinsnreg02_bits);
__IO_REG32_BIT(HC2INSNREG03,          0xE200009C,__READ_WRITE ,__hcinsnreg03_bits);
__IO_REG32_BIT(HC2INSNREG05,          0xE20000A4,__READ_WRITE ,__hcinsnreg05_bits);

/***************************************************************************
 **
 **  OCHI2
 **
 ***************************************************************************/
__IO_REG32_BIT(Hc2Revision,           0xE2100000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(Hc2Control,            0xE2100004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(Hc2CommandStatus,      0xE2100008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(Hc2InterruptStatus,    0xE210000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(Hc2InterruptEnable,    0xE2100010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc2InterruptDisable,   0xE2100014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(Hc2HCCA,               0xE2100018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(Hc2PeriodCurrentED,    0xE210001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(Hc2ControlHeadED,      0xE2100020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(Hc2ControlCurrentED,   0xE2100024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(Hc2BulkHeadED,         0xE2100028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(Hc2BulkCurrentED,      0xE210002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(Hc2DoneHead,           0xE2100030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(Hc2FmInterval,         0xE2100034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(Hc2FmRemaining,        0xE2100038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(Hc2FmNumber,           0xE210003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(Hc2PeriodStart,        0xE2100040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(Hc2LSThreshold,        0xE2100044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(Hc2RhDescriptorA,      0xE2100048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(Hc2RhDescripterB,      0xE210004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(Hc2RhStatus,           0xE2100050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(Hc2RhPortStatus,       0xE2100054,__READ_WRITE ,__hcrhportstatus_bits);

/***************************************************************************
 **
 **  USBD
 **
 ***************************************************************************/
__IO_REG32_BIT(UDEP0INCTRL,           0xE1100000,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP0INSTAT,           0xE1100004,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP0INBS,             0xE1100008,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP0INMPS,            0xE110000C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP0INDDP,            0xE1100014,__READ_WRITE );
__IO_REG32(    UDEP0INWC,             0xE1100018,__WRITE      );
__IO_REG32_BIT(UDEP1INCTRL,           0xE1100020,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP1INSTAT,           0xE1100024,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP1INBS,             0xE1100028,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP1INMPS,            0xE110002C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP1INDDP,            0xE1100034,__READ_WRITE );
__IO_REG32(    UDEP1INWC,             0xE1100038,__WRITE    );
__IO_REG32_BIT(UDEP3INCTRL,           0xE1100060,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP3INSTAT,           0xE1100064,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP3INBS,             0xE1100068,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP3INMPS,            0xE110006C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP3INDDP,            0xE1100074,__READ_WRITE );
__IO_REG32(    UDEP3INWC,             0xE1100078,__WRITE      );
__IO_REG32_BIT(UDEP5INCTRL,           0xE11000A0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP5INSTAT,           0xE11000A4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP5INBS,             0xE11000A8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP5INMPS,            0xE11000AC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP5INDDP,            0xE11000B4,__READ_WRITE );
__IO_REG32(    UDEP5INWC,             0xE11000B8,__WRITE      );
__IO_REG32_BIT(UDEP7INCTRL,           0xE11000E0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP7INSTAT,           0xE11000E4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP7INBS,             0xE11000E8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP7INMPS,            0xE11000EC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP7INDDP,            0xE11000F4,__READ_WRITE );
__IO_REG32(    UDEP7INWC,             0xE11000F8,__WRITE      );
__IO_REG32_BIT(UDEP9INCTRL,           0xE1100120,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP9INSTAT,           0xE1100124,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP9INBS,             0xE1100128,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP9INMPS,            0xE110012C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP9INDDP,            0xE1100134,__READ_WRITE );
__IO_REG32(    UDEP9INWC,             0xE1100138,__WRITE      );
__IO_REG32_BIT(UDEP11INCTRL,          0xE1100160,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP11INSTAT,          0xE1100164,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP11INBS,            0xE1100168,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP11INMPS,           0xE110016C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP11INDDP,           0xE1100174,__READ_WRITE );
__IO_REG32(    UDEP11INWC,            0xE1100178,__WRITE      );
__IO_REG32_BIT(UDEP13INCTRL,          0xE11001A0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP13INSTAT,          0xE11001A4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP13INBS,            0xE11001A8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP13INMPS,           0xE11001AC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP13INDDP,           0xE11001B4,__READ_WRITE );
__IO_REG32(    UDEP13INWC,            0xE11001B8,__WRITE      );
__IO_REG32_BIT(UDEP15INCTRL,          0xE11001E0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP15INSTAT,          0xE11001E4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP15INBS,            0xE11001E8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP15INMPS,           0xE11001EC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP15INDDP,           0xE11001F4,__READ_WRITE );
__IO_REG32(    UDEP15INWC,            0xE11001F8,__WRITE      );
__IO_REG32_BIT(UDEP0OUTCTRL,          0xE1100200,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP0OUTSTAT,          0xE1100204,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP0OUTPFN,           0xE1100208,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP0OUTBS,            0xE110020C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP0OUSBP,            0xE1100210,__READ_WRITE );
__IO_REG32(    UDEP0OUTDDP,           0xE1100214,__READ_WRITE );
__IO_REG32(    UDEP0OUTRC,            0xE110021C,__READ       );
__IO_REG32_BIT(UDEP2OUTCTRL,          0xE1100240,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP2OUTSTAT,          0xE1100244,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP2OUTPFN,           0xE1100248,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP2OUTBS,            0xE110024C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP2OUSBP,            0xE1100250,__READ_WRITE );
__IO_REG32(    UDEP2OUTDDP,           0xE1100254,__READ_WRITE );
__IO_REG32(    UDEP2OUTRC,            0xE110025C,__READ       );
__IO_REG32_BIT(UDEP4OUTCTRL,          0xE1100280,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP4OUTSTAT,          0xE1100284,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP4OUTPFN,           0xE1100288,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP4OUTBS,            0xE110028C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP4OUSBP,            0xE1100290,__READ_WRITE );
__IO_REG32(    UDEP4OUTDDP,           0xE1100294,__READ_WRITE );
__IO_REG32(    UDEP4OUTRC,            0xE110029C,__READ       );
__IO_REG32_BIT(UDEP6OUTCTRL,          0xE11002C0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP6OUTSTAT,          0xE11002C4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP6OUTPFN,           0xE11002C8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP6OUTBS,            0xE11002CC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP6OUSBP,            0xE11002D0,__READ_WRITE );
__IO_REG32(    UDEP6OUTDDP,           0xE11002D4,__READ_WRITE );
__IO_REG32(    UDEP6OUTRC,            0xE11002DC,__READ       );
__IO_REG32_BIT(UDEP8OUTCTRL,          0xE1100300,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP8OUTSTAT,          0xE1100304,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP8OUTPFN,           0xE1100308,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP8OUTBS,            0xE110030C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP8OUSBP,            0xE1100310,__READ_WRITE );
__IO_REG32(    UDEP8OUTDDP,           0xE1100314,__READ_WRITE );
__IO_REG32(    UDEP8OUTRC,            0xE110031C,__READ       );
__IO_REG32_BIT(UDEP10OUTCTRL,         0xE1100340,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP10OUTSTAT,         0xE1100344,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP10OUTPFN,          0xE1100348,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP10OUTBS,           0xE110034C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP10OUSBP,           0xE1100350,__READ_WRITE );
__IO_REG32(    UDEP10OUTDDP,          0xE1100354,__READ_WRITE );
__IO_REG32(    UDEP10OUTRC,           0xE110035C,__READ       );
__IO_REG32_BIT(UDEP12OUTCTRL,         0xE1100380,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP12OUTSTAT,         0xE1100384,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP12OUTPFN,          0xE1100388,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP12OUTBS,           0xE110038C,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP12OUSBP,           0xE1100390,__READ_WRITE );
__IO_REG32(    UDEP12OUTDDP,          0xE1100394,__READ_WRITE );
__IO_REG32(    UDEP12OUTRC,           0xE110039C,__READ       );
__IO_REG32_BIT(UDEP14OUTCTRL,         0xE11003C0,__READ_WRITE ,__udepctrl_bits);
__IO_REG32_BIT(UDEP14OUTSTAT,         0xE11003C4,__READ       ,__udepstat_bits);
__IO_REG32_BIT(UDEP14OUTPFN,          0xE11003C8,__READ_WRITE ,__udepbs_bits);
__IO_REG32_BIT(UDEP14OUTBS,           0xE11003CC,__READ_WRITE ,__udepmps_bits);
__IO_REG32(    UDEP14OUSBP,           0xE11003D0,__READ_WRITE );
__IO_REG32(    UDEP14OUTDDP,          0xE11003D4,__READ_WRITE );
__IO_REG32(    UDEP14OUTRC,           0xE11003DC,__READ       );
__IO_REG32_BIT(UDDCFG,                0xE1100400,__READ_WRITE ,__uddcfg_bits);
__IO_REG32_BIT(UDDCTRL,               0xE1100404,__READ_WRITE ,__uddctrl_bits);
__IO_REG32_BIT(UDDSTAT,               0xE1100408,__READ       ,__uddstat_bits);
__IO_REG32_BIT(UDDINTR,               0xE110040C,__READ_WRITE ,__uddintr_bits);
__IO_REG32_BIT(UDDIM,                 0xE1100410,__READ_WRITE ,__uddintr_bits);
__IO_REG32_BIT(UDEINTR,               0xE1100414,__READ_WRITE ,__udeintr_bits);
__IO_REG32_BIT(UDEIM,                 0xE1100418,__READ_WRITE ,__udeintr_bits);
__IO_REG32_BIT(UDEP0,                 0xE1100504,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP1,                 0xE1100508,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP2,                 0xE110050C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP3,                 0xE1100510,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP4,                 0xE1100514,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP5,                 0xE1100518,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP6,                 0xE110051C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP7,                 0xE1100520,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP8,                 0xE1100524,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP9,                 0xE1100528,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP10,                0xE110052C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP11,                0xE1100530,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP12,                0xE1100534,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP13,                0xE1100538,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP14,                0xE110053C,__READ_WRITE ,__udep_bits);
__IO_REG32_BIT(UDEP15,                0xE1100540,__READ_WRITE ,__udep_bits);

/***************************************************************************
 **
 **  UART 1
 **
 ***************************************************************************/
__IO_REG16_BIT(UART1DR,               0xD0000000,__READ_WRITE ,__uartdr_bits);
__IO_REG8_BIT( UART1RSR,              0xD0000004,__READ_WRITE ,__uartrsr_bits);
#define UART1ECR          UART1RSR
__IO_REG16_BIT(UART1FR,               0xD0000018,__READ       ,__uartfr_bits);
__IO_REG16(    UART1IBRD,             0xD0000024,__READ_WRITE );
__IO_REG8_BIT( UART1FBRD,             0xD0000028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART1LCR_H,            0xD000002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART1CR,               0xD0000030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART1IFLS,             0xD0000034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART1IMSC,             0xD0000038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART1RIS,              0xD000003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART1MIS,              0xD0000040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART1ICR,              0xD0000044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART1DMACR,            0xD0000048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART1PeriphID0,        0xD0000FE0,__READ       );
__IO_REG32(    UART1PeriphID1,        0xD0000FE4,__READ       );
__IO_REG32(    UART1PeriphID2,        0xD0000FE8,__READ       );
__IO_REG32(    UART1PeriphID3,        0xD0000FEC,__READ       );
__IO_REG32(    UART1PCellID0,         0xD0000FF0,__READ       );
__IO_REG32(    UART1PCellID1,         0xD0000FF4,__READ       );
__IO_REG32(    UART1PCellID2,         0xD0000FF8,__READ       );
__IO_REG32(    UART1PCellID3,         0xD0000FFC,__READ       );

/***************************************************************************
 **
 **  UART 2
 **
 ***************************************************************************/
__IO_REG16_BIT(UART2DR,               0xD0080000,__READ_WRITE ,__uartdr_bits);
__IO_REG8_BIT( UART2RSR,              0xD0080004,__READ_WRITE ,__uartrsr_bits);
#define UART2ECR          UART2RSR
__IO_REG16_BIT(UART2FR,               0xD0080018,__READ       ,__uartfr_bits);
__IO_REG16(    UART2IBRD,             0xD0080024,__READ_WRITE );
__IO_REG8_BIT( UART2FBRD,             0xD0080028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG16_BIT(UART2LCR_H,            0xD008002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG16_BIT(UART2CR,               0xD0080030,__READ_WRITE ,__uartcr_bits);
__IO_REG16_BIT(UART2IFLS,             0xD0080034,__READ_WRITE ,__uartifls_bits);
__IO_REG16_BIT(UART2IMSC,             0xD0080038,__READ_WRITE ,__uartimsc_bits);
__IO_REG16_BIT(UART2RIS,              0xD008003C,__READ       ,__uartris_bits);
__IO_REG16_BIT(UART2MIS,              0xD0080040,__READ       ,__uartmis_bits);
__IO_REG16_BIT(UART2ICR,              0xD0080044,__WRITE      ,__uarticr_bits);
__IO_REG16_BIT(UART2DMACR,            0xD0080048,__READ_WRITE ,__uartdmacr_bits);
__IO_REG32(    UART2PeriphID0,        0xD0080FE0,__READ       );
__IO_REG32(    UART2PeriphID1,        0xD0080FE4,__READ       );
__IO_REG32(    UART2PeriphID2,        0xD0080FE8,__READ       );
__IO_REG32(    UART2PeriphID3,        0xD0080FEC,__READ       );
__IO_REG32(    UART2PCellID0,         0xD0080FF0,__READ       );
__IO_REG32(    UART2PCellID1,         0xD0080FF4,__READ       );
__IO_REG32(    UART2PCellID2,         0xD0080FF8,__READ       );
__IO_REG32(    UART2PCellID3,         0xD0080FFC,__READ       );

/***************************************************************************
 **
 **  IrDA
 **
 ***************************************************************************/
__IO_REG32_BIT(IrDA_CON,              0xD1000010,__READ_WRITE ,__irda_con_bits);
__IO_REG32_BIT(IrDA_CONF,             0xD1000014,__READ_WRITE ,__irda_conf_bits);
__IO_REG32_BIT(IrDA_PARA,             0xD1000018,__READ_WRITE ,__irda_para_bits);
__IO_REG32_BIT(IrDA_DV,               0xD100001C,__READ_WRITE ,__irda_dv_bits);
__IO_REG32_BIT(IrDA_STAT,             0xD1000020,__READ       ,__irda_stat_bits);
__IO_REG32_BIT(IrDA_TFS,              0xD1000024,__WRITE      ,__irda_tfs_bits);
__IO_REG32_BIT(IrDA_RFS,              0xD1000028,__READ       ,__irda_rfs_bits);
__IO_REG32(    IrDA_TXB,              0xD100002C,__WRITE      );
__IO_REG32(    IrDA_RXB,              0xD1000030,__READ       );
__IO_REG32_BIT(IrDA_IMSC,             0xD10000E8,__READ_WRITE ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_RIS,              0xD10000EC,__READ       ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_MIS,              0xD10000F0,__READ       ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_ICR,              0xD10000F4,__WRITE      ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_ISR,              0xD10000F8,__WRITE      ,__irda_imsc_bits);
__IO_REG32_BIT(IrDA_DMA,              0xD10000FC,__READ_WRITE ,__irda_dma_bits);

/***************************************************************************
 **
 **  SSP1
 **
 ***************************************************************************/
__IO_REG16_BIT(SSP1CR0,           0xD0100000,__READ_WRITE ,__sspcr0_bits);
__IO_REG16_BIT(SSP1CR1,           0xD0100004,__READ_WRITE ,__sspcr1_bits);
__IO_REG16(    SSP1DR,            0xD0100008,__READ_WRITE );
__IO_REG16_BIT(SSP1SR,            0xD010000C,__READ       ,__sspsr_bits);
__IO_REG16_BIT(SSP1CPSR,          0xD0100010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG16_BIT(SSP1IMSC,          0xD0100014,__READ_WRITE ,__sspimsc_bits);
__IO_REG16_BIT(SSP1RIS,           0xD0100018,__READ       ,__sspris_bits);
__IO_REG16_BIT(SSP1MIS,           0xD010001C,__READ       ,__sspmis_bits);
__IO_REG16_BIT(SSP1ICR,           0xD0100020,__WRITE      ,__sspicr_bits);
__IO_REG16_BIT(SSP1DMACR,         0xD0100024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG16_BIT(SSP1PeriphID0,     0xD0100FE0,__READ       ,__sspperiphid0_bits);
__IO_REG16_BIT(SSP1PeriphID1,     0xD0100FE4,__READ       ,__sspperiphid1_bits);
__IO_REG16_BIT(SSP1PeriphID2,     0xD0100FE8,__READ       ,__sspperiphid2_bits);
__IO_REG16_BIT(SSP1PeriphID3,     0xD0100FEC,__READ       ,__sspperiphid3_bits);
__IO_REG16_BIT(SSP1CellID0,       0xD0100FF0,__READ       ,__sspcellid0_bits);
__IO_REG16_BIT(SSP1CellID1,       0xD0100FF4,__READ       ,__sspcellid1_bits);
__IO_REG16_BIT(SSP1CellID2,       0xD0100FF8,__READ       ,__sspcellid2_bits);
__IO_REG16_BIT(SSP1CellID3,       0xD0100FFC,__READ       ,__sspcellid3_bits);

/***************************************************************************
 **
 **  SSP2
 **
 ***************************************************************************/
__IO_REG16_BIT(SSP2CR0,           0xD0180000,__READ_WRITE ,__sspcr0_bits);
__IO_REG16_BIT(SSP2CR1,           0xD0180004,__READ_WRITE ,__sspcr1_bits);
__IO_REG16(    SSP2DR,            0xD0180008,__READ_WRITE );
__IO_REG16_BIT(SSP2SR,            0xD018000C,__READ       ,__sspsr_bits);
__IO_REG16_BIT(SSP2CPSR,          0xD0180010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG16_BIT(SSP2IMSC,          0xD0180014,__READ_WRITE ,__sspimsc_bits);
__IO_REG16_BIT(SSP2RIS,           0xD0180018,__READ       ,__sspris_bits);
__IO_REG16_BIT(SSP2MIS,           0xD018001C,__READ       ,__sspmis_bits);
__IO_REG16_BIT(SSP2ICR,           0xD0180020,__WRITE      ,__sspicr_bits);
__IO_REG16_BIT(SSP2DMACR,         0xD0180024,__READ_WRITE ,__sspdmacr_bits);
__IO_REG16_BIT(SSP2PeriphID0,     0xD0180FE0,__READ       ,__sspperiphid0_bits);
__IO_REG16_BIT(SSP2PeriphID1,     0xD0180FE4,__READ       ,__sspperiphid1_bits);
__IO_REG16_BIT(SSP2PeriphID2,     0xD0180FE8,__READ       ,__sspperiphid2_bits);
__IO_REG16_BIT(SSP2PeriphID3,     0xD0180FEC,__READ       ,__sspperiphid3_bits);
__IO_REG16_BIT(SSP2CellID0,       0xD0180FF0,__READ       ,__sspcellid0_bits);
__IO_REG16_BIT(SSP2CellID1,       0xD0180FF4,__READ       ,__sspcellid1_bits);
__IO_REG16_BIT(SSP2CellID2,       0xD0180FF8,__READ       ,__sspcellid2_bits);
__IO_REG16_BIT(SSP2CellID3,       0xD0180FFC,__READ       ,__sspcellid3_bits);

/***************************************************************************
 **
 **  I2C
 **
 ***************************************************************************/
__IO_REG16_BIT(IC_CON,                0xD0200000,__READ_WRITE ,__ic_con_bits);
__IO_REG16_BIT(IC_TAR,                0xD0200004,__READ_WRITE ,__ic_tar_bits);
__IO_REG16_BIT(IC_SAR,                0xD0200008,__READ_WRITE ,__ic_sar_bits);
__IO_REG16_BIT(IC_HS_MADDR,           0xD020000C,__READ_WRITE ,__ic_hs_maddr_bits);
__IO_REG16_BIT(IC_DATA_CMD,           0xD0200010,__READ_WRITE ,__ic_data_cmd_bits);
__IO_REG16(    IC_SS_SCL_HCNT,        0xD0200014,__READ_WRITE );
__IO_REG16(    IC_SS_SCL_LCNT,        0xD0200018,__READ_WRITE );
__IO_REG16(    IC_FS_SCL_HCNT,        0xD020001C,__READ_WRITE );
__IO_REG16(    IC_FS_SCL_LCNT,        0xD0200020,__READ_WRITE );
__IO_REG16(    IC_HS_SCL_HCNT,        0xD0200024,__READ_WRITE );
__IO_REG16(    IC_HS_SCL_LCNT,        0xD0200028,__READ_WRITE );
__IO_REG16_BIT(IC_INTR_STAT,          0xD020002C,__READ       ,__ic_intr_stat_bits);
__IO_REG16_BIT(IC_INTR_MASK,          0xD0200030,__READ_WRITE ,__ic_intr_mask_bits);
__IO_REG16_BIT(IC_RAW_INTR_STAT,      0xD0200034,__READ       ,__ic_raw_intr_stat_bits);
__IO_REG16_BIT(IC_RX_TL,              0xD0200038,__READ_WRITE ,__ic_rx_tl_bits);
__IO_REG16_BIT(IC_TX_TL,              0xD020003C,__READ_WRITE ,__ic_tx_tl_bits);
__IO_REG16_BIT(IC_CLR_INTR,           0xD0200040,__READ       ,__ic_clr_intr_bits);
__IO_REG16(    IC_CLR_RX_UNDER,       0xD0200044,__READ       );
__IO_REG16(    IC_CLR_RX_OVER,        0xD0200048,__READ       );
__IO_REG16(    IC_CLR_TX_OVER,        0xD020004C,__READ       );
__IO_REG16(    IC_CLR_RD_REQ,         0xD0200050,__READ       );
__IO_REG16(    IC_CLR_TX_ABRT,        0xD0200054,__READ       );
__IO_REG16(    IC_CLR_RX_DONE,        0xD0200058,__READ       );
__IO_REG16(    IC_CLR_ACTIVITY,       0xD020005C,__READ       );
__IO_REG16(    IC_CLR_STOP_DET,       0xD0200060,__READ       );
__IO_REG16(    IC_CLR_START_DET,      0xD0200064,__READ       );
__IO_REG16(    IC_CLR_GEN_CALL,       0xD0200068,__READ       );
__IO_REG16_BIT(IC_ENABLE,             0xD020006C,__READ_WRITE ,__ic_enable_bits);
__IO_REG16_BIT(IC_STATUS,             0xD0200070,__READ       ,__ic_status_bits);
__IO_REG16_BIT(IC_TXFLR,              0xD0200074,__READ       ,__ic_txflr_bits);
__IO_REG16_BIT(IC_RXFLR,              0xD0200078,__READ       ,__ic_rxflr_bits);
__IO_REG16_BIT(IC_TX_ABRT_SOURCE,     0xD0200080,__READ_WRITE ,__ic_tx_abrt_source_bits);
__IO_REG16_BIT(IC_DMA_CR,             0xD0200088,__READ_WRITE ,__ic_dma_cr_bits);
__IO_REG16_BIT(IC_DMA_TDLR,           0xD020008C,__READ_WRITE ,__ic_dma_tdlr_bits);
__IO_REG16_BIT(IC_DMA_RDLR,           0xD0200090,__READ_WRITE ,__ic_dma_rdlr_bits);
__IO_REG32_BIT(IC_COMP_PARAM_1,       0xD02000F4,__READ       ,__ic_comp_param_1_bits);
__IO_REG32(    IC_COMP_VERSION,       0xD02000F8,__READ       );
__IO_REG32(    IC_COMP_TYPE,          0xD02000FC,__READ       );

/***************************************************************************
 **
 **  DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStatus,         0xFC400000,__READ       ,__dmacintstatus_bits);
__IO_REG32_BIT(DMACIntTCStatus,       0xFC400004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACIntTCClear,        0xFC400008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACIntErrorStatus,    0xFC40000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACIntErrClr,         0xFC400010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRawIntTCStatus,    0xFC400014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRawIntErrorStatus, 0xFC400018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACEnbldChns,         0xFC40001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSoftBReq,          0xFC400020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0xFC400024,__READ_WRITE ,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACSoftLBReq,         0xFC400028,__READ_WRITE ,__dmacsoftlbreq_bits);
__IO_REG32_BIT(DMACSoftLSReq,         0xFC40002C,__READ_WRITE ,__dmacsoftlsreq_bits);
__IO_REG32_BIT(DMACConfiguration,     0xFC400030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32_BIT(DMACSync,              0xFC400034,__READ_WRITE ,__dmacsync_bits);
__IO_REG32(    DMACC0SrcAddr,         0xFC400100,__READ_WRITE );
__IO_REG32(    DMACC0DestAddr,        0xFC400104,__READ_WRITE );
__IO_REG32_BIT(DMACC0LLI,             0xFC400108,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC0Control,         0xFC40010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC0Configuration,   0xFC400110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC1SrcAddr,         0xFC400120,__READ_WRITE );
__IO_REG32(    DMACC1DestAddr,        0xFC400124,__READ_WRITE );
__IO_REG32_BIT(DMACC1LLI,             0xFC400128,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC1Control,         0xFC40012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC1Configuration,   0xFC400130,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC2SrcAddr,         0xFC400140,__READ_WRITE );
__IO_REG32(    DMACC2DestAddr,        0xFC400144,__READ_WRITE );
__IO_REG32_BIT(DMACC2LLI,             0xFC400148,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC2Control,         0xFC40014C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC2Configuration,   0xFC400150,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC3SrcAddr,         0xFC400160,__READ_WRITE );
__IO_REG32(    DMACC3DestAddr,        0xFC400164,__READ_WRITE );
__IO_REG32_BIT(DMACC3LLI,             0xFC400168,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC3Control,         0xFC40016C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC3Configuration,   0xFC400170,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC4SrcAddr,         0xFC400180,__READ_WRITE );
__IO_REG32(    DMACC4DestAddr,        0xFC400184,__READ_WRITE );
__IO_REG32_BIT(DMACC4LLI,             0xFC400188,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC4Control,         0xFC40018C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC4Configuration,   0xFC400190,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC5SrcAddr,         0xFC4001A0,__READ_WRITE );
__IO_REG32(    DMACC5DestAddr,        0xFC4001A4,__READ_WRITE );
__IO_REG32_BIT(DMACC5LLI,             0xFC4001A8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC5Control,         0xFC4001AC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC5Configuration,   0xFC4001B0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC6SrcAddr,         0xFC4001C0,__READ_WRITE );
__IO_REG32(    DMACC6DestAddr,        0xFC4001C4,__READ_WRITE );
__IO_REG32_BIT(DMACC6LLI,             0xFC4001C8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC6Control,         0xFC4001CC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC6Configuration,   0xFC4001D0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC7SrcAddr,         0xFC4001E0,__READ_WRITE );
__IO_REG32(    DMACC7DestAddr,        0xFC4001E4,__READ_WRITE );
__IO_REG32_BIT(DMACC7LLI,             0xFC4001E8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC7Control,         0xFC4001EC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC7Configuration,   0xFC4001F0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(		 DMACPeriphID0,   			0xFC400FE0,__READ       );
__IO_REG32(		 DMACPeriphID1,   			0xFC400FE4,__READ       );
__IO_REG32(		 DMACPeriphID2,   			0xFC400FE8,__READ       );
__IO_REG32(		 DMACPeriphID3,   			0xFC400FEC,__READ       );
__IO_REG32(		 DMACPCellID0,   				0xFC400FF0,__READ       );
__IO_REG32(		 DMACPCellID1,   				0xFC400FF4,__READ       );
__IO_REG32(		 DMACPCellID2,   				0xFC400FF8,__READ       );
__IO_REG32(		 DMACPCellID3,   				0xFC400FFC,__READ       );

/***************************************************************************
 **
 **  GPIO Basic
 **
 ***************************************************************************/
__IO_REG16_BIT(GPIOBDATA,             0xFC9803FC,__READ_WRITE ,__gpiodata_bits);
__IO_REG16_BIT(GPIOBDIR,              0xFC980400,__READ_WRITE ,__gpiodir_bits);
__IO_REG16_BIT(GPIOBIS,               0xFC980404,__READ_WRITE ,__gpiois_bits);
__IO_REG16_BIT(GPIOBIBE,              0xFC980408,__READ_WRITE ,__gpioibe_bits);
__IO_REG16_BIT(GPIOBIEV,              0xFC98040C,__READ_WRITE ,__gpioiev_bits);
__IO_REG16_BIT(GPIOBIE,               0xFC980410,__READ_WRITE ,__gpioie_bits);
__IO_REG16_BIT(GPIOBRIS,              0xFC980414,__READ       ,__gpioris_bits);
__IO_REG16_BIT(GPIOBMIS,              0xFC980418,__READ       ,__gpiomis_bits);
__IO_REG16_BIT(GPIOBIC,               0xFC98041C,__WRITE      ,__gpioic_bits);
__IO_REG16_BIT(GPIOBAFSEL,            0xFC980420,__READ_WRITE ,__gpioafsel_bits);
__IO_REG16(    GPIOBPeriphID0,        0xFC980FE0,__READ       );
__IO_REG16(    GPIOBPeriphID1,        0xFC980FE4,__READ       );
__IO_REG16(    GPIOBPeriphID2,        0xFC980FE8,__READ       );
__IO_REG16(    GPIOBPeriphID3,        0xFC980FEC,__READ       );
__IO_REG16(    GPIOBPCellID0,         0xFC980FF0,__READ       );
__IO_REG16(    GPIOBPCellID1,         0xFC980FF4,__READ       );
__IO_REG16(    GPIOBPCellID2,         0xFC980FF8,__READ       );
__IO_REG16(    GPIOBPCellID3,         0xFC980FFC,__READ       );

/***************************************************************************
 **
 **  GPIO
 **
 ***************************************************************************/
__IO_REG16_BIT(GPIODATA,              0xD81003FC,__READ_WRITE ,__gpiodata_bits);
__IO_REG16_BIT(GPIODIR,               0xD8100400,__READ_WRITE ,__gpiodir_bits);
__IO_REG16_BIT(GPIOIS,                0xD8100404,__READ_WRITE ,__gpiois_bits);
__IO_REG16_BIT(GPIOIBE,               0xD8100408,__READ_WRITE ,__gpioibe_bits);
__IO_REG16_BIT(GPIOIEV,               0xD810040C,__READ_WRITE ,__gpioiev_bits);
__IO_REG16_BIT(GPIOIE,                0xD8100410,__READ_WRITE ,__gpioie_bits);
__IO_REG16_BIT(GPIORIS,               0xD8100414,__READ       ,__gpioris_bits);
__IO_REG16_BIT(GPIOMIS,               0xD8100418,__READ       ,__gpiomis_bits);
__IO_REG16_BIT(GPIOIC,                0xD810041C,__WRITE      ,__gpioic_bits);
__IO_REG16_BIT(GPIOAFSEL,             0xD8100420,__READ_WRITE ,__gpioafsel_bits);
__IO_REG16(    GPIOPeriphID0,         0xD8100FE0,__READ       );
__IO_REG16(    GPIOPeriphID1,         0xD8100FE4,__READ       );
__IO_REG16(    GPIOPeriphID2,         0xD8100FE8,__READ       );
__IO_REG16(    GPIOPeriphID3,         0xD8100FEC,__READ       );
__IO_REG16(    GPIOPCellID0,          0xD8100FF0,__READ       );
__IO_REG16(    GPIOPCellID1,          0xD8100FF4,__READ       );
__IO_REG16(    GPIOPCellID2,          0xD8100FF8,__READ       );
__IO_REG16(    GPIOPCellID3,          0xD8100FFC,__READ       );

/***************************************************************************
 **
 **  GPIO ML1/2
 **
 ***************************************************************************/
__IO_REG16_BIT(GPIOMLDATA,            0xF01003FC,__READ_WRITE ,__gpiodata_bits);
__IO_REG16_BIT(GPIOMLDIR,             0xF0100400,__READ_WRITE ,__gpiodir_bits);
__IO_REG16_BIT(GPIOMLIS,              0xF0100404,__READ_WRITE ,__gpiois_bits);
__IO_REG16_BIT(GPIOMLIBE,             0xF0100408,__READ_WRITE ,__gpioibe_bits);
__IO_REG16_BIT(GPIOMLIEV,             0xF010040C,__READ_WRITE ,__gpioiev_bits);
__IO_REG16_BIT(GPIOMLIE,              0xF0100410,__READ_WRITE ,__gpioie_bits);
__IO_REG16_BIT(GPIOMLRIS,             0xF0100414,__READ       ,__gpioris_bits);
__IO_REG16_BIT(GPIOMLMIS,             0xF0100418,__READ       ,__gpiomis_bits);
__IO_REG16_BIT(GPIOMLIC,              0xF010041C,__WRITE      ,__gpioic_bits);
__IO_REG16_BIT(GPIOMLAFSEL,           0xF0100420,__READ_WRITE ,__gpioafsel_bits);
__IO_REG16(    GPIOMLPeriphID0,       0xF0100FE0,__READ       );
__IO_REG16(    GPIOMLPeriphID1,       0xF0100FE4,__READ       );
__IO_REG16(    GPIOMLPeriphID2,       0xF0100FE8,__READ       );
__IO_REG16(    GPIOMLPeriphID3,       0xF0100FEC,__READ       );
__IO_REG16(    GPIOMLPCellID0,        0xF0100FF0,__READ       );
__IO_REG16(    GPIOMLPCellID1,        0xF0100FF4,__READ       );
__IO_REG16(    GPIOMLPCellID2,        0xF0100FF8,__READ       );
__IO_REG16(    GPIOMLPCellID3,        0xF0100FFC,__READ       );

/***************************************************************************
 **
 **  CLCD
 **
 ***************************************************************************/
__IO_REG32_BIT(LCDTiming0,            0xFC200000,__READ_WRITE ,__lcdtiming0_bits);
__IO_REG32_BIT(LCDTiming1,            0xFC200004,__READ_WRITE ,__lcdtiming1_bits);
__IO_REG32_BIT(LCDTiming2,            0xFC200008,__READ_WRITE ,__lcdtiming2_bits);
__IO_REG32_BIT(LCDTiming3,            0xFC20000C,__READ_WRITE ,__lcdtiming3_bits);
__IO_REG32(    LCDUPBase,             0xFC200010,__READ_WRITE );
__IO_REG32(    LCDLPBase,             0xFC200014,__READ_WRITE );
__IO_REG32_BIT(LCDMSC,                0xFC200018,__READ_WRITE ,__lcdmsc_bits);
__IO_REG32_BIT(LCDControl,            0xFC20001C,__READ_WRITE ,__lcdcontrol_bits);
__IO_REG32_BIT(LCDRIS,                0xFC200020,__READ_WRITE ,__lcdris_bits);
__IO_REG32_BIT(LCDMIS,                0xFC200024,__READ       ,__lcdmis_bits);
__IO_REG32_BIT(LCDICR,                0xFC200028,__WRITE      ,__lcdris_bits);
__IO_REG32(    LCDUPCUR,              0xFC20002C,__READ       );
__IO_REG32(    LCDLPCUR,              0xFC200030,__READ       );
__IO_REG32(    LCDPaletteBase,        0xFC200200,__READ_WRITE );
__IO_REG32(    LCDLPHERIPHID0,        0xFC200FE0,__READ       );
__IO_REG32(    LCDLPHERIPHID1,        0xFC200FE4,__READ       );
__IO_REG32(    LCDLPHERIPHID2,        0xFC200FE8,__READ       );
__IO_REG32(    LCDLPHERIPHID3,        0xFC200FEC,__READ       );
__IO_REG32(    LCDLPCELLIDID0,        0xFC200FF0,__READ       );
__IO_REG32(    LCDLPCELLIDID1,        0xFC200FF4,__READ       );
__IO_REG32(    LCDLPCELLIDID2,        0xFC200FF8,__READ       );
__IO_REG32(    LCDLPCELLIDID3,        0xFC200FFC,__READ       );

/***************************************************************************
 **
 **  JPEG
 **
 ***************************************************************************/
__IO_REG32_BIT(JPGCReg0,              0xD0800000,__WRITE      ,__jpgcreg0_bits);
__IO_REG32_BIT(JPGCReg1,              0xD0800004,__READ_WRITE ,__jpgcreg1_bits);
__IO_REG32_BIT(JPGCReg2,              0xD0800008,__READ_WRITE ,__jpgcreg2_bits);
__IO_REG32_BIT(JPGCReg3,              0xD080000C,__READ_WRITE ,__jpgcreg3_bits);
__IO_REG32_BIT(JPGCReg4,              0xD0800010,__READ_WRITE ,__jpgcreg4_bits);
__IO_REG32_BIT(JPGCReg5,              0xD0800014,__READ_WRITE ,__jpgcreg5_bits);
__IO_REG32_BIT(JPGCReg6,              0xD0800018,__READ_WRITE ,__jpgcreg6_bits);
__IO_REG32_BIT(JPGCReg7,              0xD080001C,__READ_WRITE ,__jpgcreg7_bits);
__IO_REG32_BIT(JPGCCS,                0xD0800200,__READ_WRITE ,__jpgccs_bits);
__IO_REG32(    JPGCBFIFO2C,           0xD0800204,__READ       );
__IO_REG32(    JPGCBC2FIFO,           0xD0800208,__READ       );
__IO_REG32_BIT(JPGCBCBI,              0xD080020C,__READ_WRITE ,__jpgcbcbi_bits);
__IO_REG32(    JPGCFifoIn,            0xD0800400,__READ_WRITE );
__IO_REG32(    JPGCFifoOut,           0xD0800600,__READ_WRITE );
__IO_REG32(    JPGCQMem,              0xD0800800,__READ_WRITE );
__IO_REG32(    JPGCHuffMin,           0xD0800C00,__READ_WRITE );
__IO_REG32(    JPGCHuffBase,          0xD0801000,__READ_WRITE );
__IO_REG32(    JPGCHuffSymb,          0xD0801400,__READ_WRITE );
__IO_REG32(    JPGCDHTMem,            0xD0801800,__READ_WRITE );
__IO_REG32(    JPGCHuffEnc,           0xD0801C00,__READ_WRITE );

/***************************************************************************
 **
 **  ADC
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC_STATUS_REG,        0x1200B000,__READ_WRITE ,__adc_status_reg_bits);
__IO_REG16_BIT(ADC_CLK_REG,           0x1200B00C,__READ_WRITE ,__adc_clk_reg_bits);
__IO_REG16_BIT(ADC_CH0_CTRL,      		0x1200B010,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH1_CTRL,      		0x1200B014,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH2_CTRL,      		0x1200B018,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH3_CTRL,      		0x1200B01C,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH4_CTRL,      		0x1200B020,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH5_CTRL,      		0x1200B024,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH6_CTRL,      		0x1200B028,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH7_CTRL,      		0x1200B02C,__READ_WRITE ,__adc_ch_ctrl_reg_bits);
__IO_REG16_BIT(ADC_CH0_DATA_LSB,      0x1200B030,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH0_DATA_MSB,      0x1200B034,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH1_DATA_LSB,      0x1200B038,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH1_DATA_MSB,      0x1200B03C,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH2_DATA_LSB,      0x1200B040,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH2_DATA_MSB,      0x1200B044,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH3_DATA_LSB,      0x1200B048,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH3_DATA_MSB,      0x1200B04C,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH4_DATA_LSB,      0x1200B050,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH4_DATA_MSB,      0x1200B054,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH5_DATA_LSB,      0x1200B058,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH5_DATA_MSB,      0x1200B05C,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH6_DATA_LSB,      0x1200B060,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH6_DATA_MSB,      0x1200B064,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16_BIT(ADC_CH7_DATA_LSB,      0x1200B068,__READ       ,__adc_ch_data_lsb_bits);
__IO_REG16_BIT(ADC_CH7_DATA_MSB,      0x1200B06C,__READ       ,__adc_ch_data_msb_bits);
__IO_REG16(		 ADC_SCAN_RATE_LO,      0x1200B070,__READ_WRITE );
__IO_REG16(		 ADC_SCAN_RATE_HI,      0x1200B074,__READ_WRITE );
__IO_REG16_BIT(ADC_AVERAGE_REG_LSB,   0x1200B078,__READ_WRITE ,__adc_average_reg_lsb_bits);
__IO_REG16_BIT(ADC_AVERAGE_REG_MSB,   0x1200B07C,__READ_WRITE ,__adc_average_reg_msb_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTCTIME,               0xFC900000,__READ_WRITE ,__rtctime_bits);
__IO_REG32_BIT(RTCDATE,               0xFC900004,__READ_WRITE ,__rtcdate_bits);
__IO_REG32_BIT(RTCALARMTIME,          0xFC900008,__READ_WRITE ,__rtctime_bits);
__IO_REG32_BIT(RTCALARMDATE,          0xFC90000C,__READ_WRITE ,__rtcdate_bits);
__IO_REG32_BIT(RTCCONTROL,            0xFC900010,__READ_WRITE ,__rtccontrol_bits);
__IO_REG32_BIT(RTCSTATUS,             0xFC900014,__READ_WRITE ,__rtcstatus_bits);
__IO_REG32(    RTCREG1MC,             0xFC900018,__READ_WRITE );
__IO_REG32(    RTCREG2MC,             0xFC90001C,__READ_WRITE );

/***************************************************************************
 **
 **  I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_Control_Reg,       0x40000800,__READ_WRITE ,__i2s_control_reg_bits);
__IO_REG32_BIT(I2S_Irq_Reg,           0x40000804,__READ_WRITE ,__i2s_irq_reg_bits);
__IO_REG32_BIT(I2S_Irq_Mask_Reg,      0x40000808,__READ_WRITE ,__i2s_irq_reg_bits);
__IO_REG32_BIT(I2S_Status_Reg,        0x4000080C,__READ				,__i2s_status_reg_bits);
__IO_REG32(		 I2S_Play_Desc_Reg,     0x40000818,__READ_WRITE );
__IO_REG32(		 I2S_Rec_Desc_Reg,      0x4000081C,__READ_WRITE );
__IO_REG32(		 I2S_Rec_Data_Cnt_Reg,  0x40000820,__READ_WRITE );

/***************************************************************************
 **
 **  MLEPX
 **
 ***************************************************************************/
__IO_REG32_BIT(MLEXP_PL1,       			0xCFFFE000,__READ_WRITE ,__mlexp_pl_bits);
__IO_REG32_BIT(MLEXP_PL2,       			0xCFFFE004,__READ_WRITE ,__mlexp_pl_bits);
__IO_REG32_BIT(MLEXP_PL3,       			0xCFFFE008,__READ_WRITE ,__mlexp_pl_bits);
__IO_REG32_BIT(MLEPX_EBTCOUNT,       	0xCFFFE03C,__READ_WRITE ,__mlepx_ebtcount_bits);
__IO_REG32_BIT(MLEPX_EBT_EN,       		0xCFFFE040,__READ_WRITE ,__mlepx_ebt_en_bits);
__IO_REG32_BIT(MLEPX_EBT,       			0xCFFFE044,__READ				,__mlepx_ebt_bits);
__IO_REG32_BIT(MLEPX_DFT_MST,       	0xCFFFE048,__READ_WRITE ,__mlepx_dft_mst_bits);
__IO_REG32(		 MLEXP_COMP_VERSION,    0xCFFFE090,__READ				);
__IO_REG32_BIT(XLAT_ENT0,       			0xCFFFE400,__READ				,__xlat_ent_bits);
__IO_REG32_BIT(XLAT_ENT1,       			0xCFFFE404,__READ_WRITE ,__xlat_ent_bits);
__IO_REG32_BIT(XLAT_ENT2,       			0xCFFFE408,__READ_WRITE ,__xlat_ent_bits);
__IO_REG32_BIT(XLAT_ENT3,       			0xCFFFE40C,__READ_WRITE ,__xlat_ent_bits);
__IO_REG32_BIT(XLAT_ENT4,       			0xCFFFE410,__READ_WRITE ,__xlat_ent_bits);
__IO_REG32_BIT(XLAT_ENT5,       			0xCFFFE414,__READ_WRITE ,__xlat_ent_bits);
__IO_REG32_BIT(SE2H_EWSC,       			0xCFFFF000,__WRITE 			,__se2h_ewsc_bits);
__IO_REG32_BIT(SE2H_EWS,       				0xCFFFF004,__READ				,__se2h_ewsc_bits);
__IO_REG32_BIT(SE2H_MEWS,       			0xCFFFF008,__READ				,__se2h_ewsc_bits);
__IO_REG32_BIT(SE2H_COMP_PARAM1,      0xCFFFF3F0,__READ				,__se2h_comp_param1_bits);
__IO_REG32_BIT(SE2H_COMP_PARAM2,      0xCFFFF3F4,__READ				,__se2h_comp_param2_bits);
__IO_REG32(		 SE2H_COMP_VERSION,     0xCFFFF3F8,__READ				);
__IO_REG32(		 SE2H_COMP_TYPE,       	0xCFFFF3FC,__READ				);
__IO_REG32_BIT(ME2H_EWSC,       			0xCFFFF800,__WRITE 			,__me2h_ewsc_bits);
__IO_REG32_BIT(ME2H_EWS,       				0xCFFFF804,__READ				,__me2h_ewsc_bits);
__IO_REG32_BIT(ME2H_MEWS,       			0xCFFFF808,__READ				,__me2h_ewsc_bits);
__IO_REG32_BIT(ME2H_COMP_PARAM1,      0xCFFFFBF0,__READ				,__me2h_comp_param1_bits);
__IO_REG32_BIT(ME2H_COMP_PARAM2,      0xCFFFFBF4,__READ				,__me2h_comp_param2_bits);
__IO_REG32(		 ME2H_COMP_VERSION,     0xCFFFFBF8,__READ				);
__IO_REG32(		 ME2H_COMP_TYPE,       	0xCFFFFBFC,__READ				);


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

#endif    /* __IOSPEAR600_H */
