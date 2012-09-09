/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Socle PC7130
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 30258 $
 **
 ***************************************************************************/

#ifndef __PC7130_H
#define __PC7130_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   PC7130 SPECIAL FUNCTION REGISTERS
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

/* PLL Configuration Parameter */
typedef struct {
  __REG32  CPLL_OD        : 2;
  __REG32  CPLL_DIV       : 5;
  __REG32  CPLL_MUL       : 9;
  __REG32  UPLL_OD        : 2;
  __REG32  UPLL_DIV       : 5;
  __REG32  UPLL_MUL       : 9;
} __scu_pllparam_a_bits;

/* Clock mode control register */
typedef struct {
  __REG32  PLL_LOCK_CNTR  :16;
  __REG32  AHB_DIV        : 2;
  __REG32  CPLL_PD        : 1;
  __REG32  UPLL_PD        : 1;
  __REG32                 :12;
} __scu_pllparam_b_bits;

/* Chip Configuration control register */
typedef struct {
  __REG32  FIRQ_OVRCURR_USB : 1;
  __REG32  USB_OVRCURR_POL  : 1;
  __REG32  FIRQ_POL         : 1;
  __REG32  USB_TRAN_DOWN    : 1;
  __REG32  UART_HDMA_MAP01  : 2;
  __REG32  UART_HDMA_MAP23  : 2;
  __REG32                   : 5;
  __REG32  UCFG_MODE        : 3;
  __REG32  DCFG_MODE        : 2;
  __REG32  USB_PD           : 1;
  __REG32                   :13;
} __scu_chipcfg_a_bits;

/* ADC clock control register */
typedef struct {
  __REG32  ADC_DIV        :16;
  __REG32                 :16;
} __scu_chipcfg_b_bits;

/* IP clock control register */
typedef struct {
  __REG32  SDRSTMC_CLK_EN : 1;
  __REG32  HCLK_HDMA_EN   : 1;
  __REG32  HCLK_MAC_EN    : 1;
  __REG32  HCLK_UHC_EN    : 1;
  __REG32  HCLK_UDC_EN    : 1;
  __REG32  HCLK_NFC_EN    : 1;
  __REG32  HCLK_LCD_EN    : 1;
  __REG32                 : 1;
  __REG32  PCLK_TMR_EN    : 1;
  __REG32  PCLK_RTC_EN    : 1;
  __REG32  PCLK_WDT_EN    : 1;
  __REG32  PCLK_GPIO_EN   : 1;
  __REG32  PCLK_SPI_EN    : 1;
  __REG32  PCLK_I2C_EN    : 1;
  __REG32  PCLK_I2S_EN    : 1;
  __REG32  PCLK_SDC_EN    : 1;
  __REG32  PCLK_PWM_EN    : 1;
  __REG32  PCLK_ADC_EN    : 1;
  __REG32  PCLK_UART0_EN  : 1;
  __REG32  PCLK_UART1_EN  : 1;
  __REG32  PCLK_UART2_EN  : 1;
  __REG32  PCLK_UART3_EN  : 1;
  __REG32                 : 2;
  __REG32  ADCCLK_EN      : 1;
  __REG32                 : 7;
} __scu_clkcfg_bits;

/* Chip Configuration control register */
typedef struct {
  __REG32  DEC_REMAP      : 1;
  __REG32  PLL_BYPASS     : 1;
  __REG32  ARM7EJ_CLK_DIS : 1;
  __REG32  SYS_CLK_DIS    : 1;
  __REG32  PLL_LOCK       : 1;
  __REG32  AUTO_BOOT_FAULT: 1;
  __REG32  IRQ_DIS        : 1;
  __REG32  FIQ_DIS        : 1;
  __REG32  BOOT           : 2;
  __REG32  UCFG_MODE1     : 1;
  __REG32  UCFG_MODE2     : 1;
  __REG32  UCFG_MODE3     : 1;
  __REG32  UCFG_MODE4     : 1;
  __REG32  UCFG_MODE5     : 1;
  __REG32  UCFG_MODE6     : 1;
  __REG32  RPS_MAC        : 1;
  __REG32  TPS_MAC        : 1;
  __REG32  SDRAM_BUS_SEL  : 1;
  __REG32                 :13;
} __scu_chipcfg_c_bits;

/* Cache Basic device status/info register */
typedef struct {
  __REG32                 : 4;
  __REG32  WORD_SIZE      : 1;
  __REG32  _2_WAY_ASSC    : 3;
  __REG32  CACHE_SIZE     : 2;
  __REG32                 : 6;
  __REG32  REVISION       :12;
  __REG32                 : 3;
  __REG32  CACHE_EN       : 1;
} __cachedevid_bits;

/* Control cache operations */
typedef struct {
  __REG32  OP             : 2;
  __REG32                 :30;
} __cacheop_bits;

/* Control cache operations */
typedef struct {
  __REG32  LKDN           : 2;
  __REG32                 :30;
} __cachelkdn_bits;

/* Performance counter registers */
typedef struct {
  __REG32  CNTR_EN        : 1;
  __REG32                 : 3;
  __REG32  CNTR_EVN       : 4;
  __REG32                 :24;
} __pfcntr_ctrl_bits;

/* ARB_MODE */
typedef struct {
  __REG32  FIXED          : 1;
  __REG32  TO_EN          : 1;
  __REG32  TO_CNTR        : 6;
  __REG32                 :24;
} __arb_mode_bits;

/* ARB_PRIOx (x=1 ~ 15) */
typedef struct {
  __REG32  PRIORITY       : 4;
  __REG32                 :28;
} __arb_prio_bits;

/* SDRAM CAS latency and burst length */
typedef struct {
  __REG32  BL             : 3;
  __REG32                 : 1;
  __REG32  CAS            : 3;
  __REG32                 :25;
} __mcsdr_mode_bits;

/* Memory modules address mapping */
typedef struct {
  __REG32  BANKS0         : 2;
  __REG32  BANKS1         : 2;
  __REG32  SIZE0          : 2;
  __REG32  SIZE1          : 2;
  __REG32  BA1            : 3;
  __REG32  MMAM           : 1;
  __REG32                 :20;
} __mcsdr_addmap_bits;

/* Memory modules address config */
typedef struct {
  __REG32  COL0           : 3;
  __REG32                 : 1;
  __REG32  ROW0           : 2;
  __REG32                 : 2;
  __REG32  COL1           : 3;
  __REG32                 : 1;
  __REG32  ROW1           : 2;
  __REG32                 :18;
} __mcsdr_addcfg_bits;

/* SDRAM Basic setting */
typedef struct {
  __REG32  WIDTH0         : 2;
  __REG32  WIDTH1         : 2;
  __REG32                 : 1;
  __REG32  PRI            : 3;
  __REG32                 :24;
} __mcsdr_basic_bits;

/* Average periodic refresh interval */
typedef struct {
  __REG32  T_REF          :16;
  __REG32                 :16;
} __mcsdr_t_ref_bits;

/* Auto refresh period */
typedef struct {
  __REG32  T_RFC          : 4;
  __REG32                 :28;
} __mcsdr_t_rfc_bits;

/* Command to ACTIVE or REFRESH period */
typedef struct {
  __REG32  T_MRD          : 3;
  __REG32                 :29;
} __mcsdr_t_mrd_bits;

/* Pre-charge command period */
typedef struct {
  __REG32  T_RP           : 3;
  __REG32                 :29;
} __mcsdr_t_rp_bits;

/* Active to Read or Write delay */
typedef struct {
  __REG32  T_RCD          : 3;
  __REG32                 :29;
} __mcsdr_t_rcd_bits;

/* Static memory timing control register for write CE width */
typedef struct {
  __REG32  T_CEWD         : 8;
  __REG32                 :24;
} __mcst_t_cewd_bits;

/* Static memory timing control register for low of CE to low of WE */
typedef struct {
  __REG32  T_CE2WE        : 8;
  __REG32                 :24;
} __mcst_t_ce2we_bits;

/* Static memory timing control register for WE width */
typedef struct {
  __REG32  T_WEWD         : 8;
  __REG32                 :24;
} __mcst_t_wewd_bits;

/* Static memory timing control register for high of WE to high of CE */
typedef struct {
  __REG32  T_WE2CE        : 8;
  __REG32                 :24;
} __mcst_t_we2ce_bits;

/* Static memory timing control register for read CE width */
typedef struct {
  __REG32  T_CEWDR        : 8;
  __REG32                 :24;
} __mcst_t_cewdr_bits;

/* Static memory timing control register for low of CE to low of RD */
typedef struct {
  __REG32  T_CE2RD        : 8;
  __REG32                 :24;
} __mcst_t_ce2rd_bits;

/* Static memory timing control register for RD width */
typedef struct {
  __REG32  T_RDWD         : 8;
  __REG32                 :24;
} __mcst_t_rdwd_bits;

/* Static memory timing control register for high of RD to high of CE */
typedef struct {
  __REG32  T_RD2CE        : 8;
  __REG32                 :24;
} __mcst_t_rd2ce_bits;

/* Static memory basic setting */
typedef struct {
  __REG32  WIDTH          : 2;
  __REG32                 : 3;
  __REG32  WP             : 1;
  __REG32                 :26;
} __mcst_basic_bits;

/* Source Control Registers */
typedef struct {
  __REG32  PRI            : 5;
  __REG32                 : 1;
  __REG32  SENS           : 2;
  __REG32                 :24;
} __intc_scr_bits;

/* Interrupt Status Register */
typedef struct {
  __REG32  IRQ_ID         : 5;
  __REG32                 :27;
} __intc_isr_bits;

/* Interrupt Pending Register */
typedef struct {
  __REG32  INTR0          : 1;
  __REG32  INTR1          : 1;
  __REG32  INTR2          : 1;
  __REG32  INTR3          : 1;
  __REG32  INTR4          : 1;
  __REG32  INTR5          : 1;
  __REG32  INTR6          : 1;
  __REG32  INTR7          : 1;
  __REG32  INTR8          : 1;
  __REG32  INTR9          : 1;
  __REG32  INTR10         : 1;
  __REG32  INTR11         : 1;
  __REG32  INTR12         : 1;
  __REG32  INTR13         : 1;
  __REG32  INTR14         : 1;
  __REG32  INTR15         : 1;
  __REG32  INTR16         : 1;
  __REG32  INTR17         : 1;
  __REG32  INTR18         : 1;
  __REG32  INTR19         : 1;
  __REG32  INTR20         : 1;
  __REG32  INTR21         : 1;
  __REG32  INTR22         : 1;
  __REG32  INTR23         : 1;
  __REG32  INTR24         : 1;
  __REG32  INTR25         : 1;
  __REG32  INTR26         : 1;
  __REG32  INTR27         : 1;
  __REG32  INTR28         : 1;
  __REG32  INTR29         : 1;
  __REG32  INTR30         : 1;
  __REG32  INTR31         : 1;
} __intc_ipr_bits;

/* Interrupt Status Register */
typedef struct {
  __REG32  TEST_MODE      : 1;
  __REG32                 : 3;
  __REG32  IRQ_LEVEL      : 1;
  __REG32  FIQ_LEVEL      : 1;
  __REG32                 :26;
} __intc_test_bits;

/* HDMA control register for channel x */
typedef struct {
  __REG32  HW_TRIG_EN     : 1;
  __REG32  OP             : 2;
  __REG32  DATA_SIZE      : 2;
  __REG32  DEST_FIXED     : 2;
  __REG32  SRC_FIXED      : 2;
  __REG32  EXT_HDREQ      : 4;
  __REG32  TRANS_MODE     : 3;
  __REG32                 : 2;
  __REG32  INTR_MODE      : 2;
  __REG32                 : 1;
  __REG32  HDMA_CH_EN     : 1;
  __REG32  HDMA_SLICE_EN  : 1;
  __REG32  CLER           : 1;
  __REG32                 : 8;
} __hdma_con_bits;

/* HDMA initial terminate count register for channel x */
typedef struct {
  __REG32  CNTR           :16;
  __REG32                 :16;
} __hdma_icnt_bits;

/* Interrupt status */
typedef struct {
  __REG32  ACT0           : 1;
  __REG32  ACT1           : 1;
  __REG32  PCI0           : 1;
  __REG32  PCI1           : 1;
  __REG32  PAO0           : 1;
  __REG32  PAO1           : 1;
  __REG32                 : 2;
  __REG32  ACT0M          : 1;
  __REG32  ACT1M          : 1;
  __REG32  PCI0M          : 1;
  __REG32  PCI1M          : 1;
  __REG32  PAO0M          : 1;
  __REG32  PAO1M          : 1;
  __REG32                 :18;
} __hdma_isr_bits;

/* DMA status */
typedef struct {
  __REG32  BUSY0          : 1;
  __REG32  BUSY1          : 1;
  __REG32                 :30;
} __hdma_dsr_bits;

/* HDMA initial slice count register for channel x */
typedef struct {
  __REG32  SCNT           : 8;
  __REG32                 :24;
} __hdma_iscnt_bits;

/* HDMA initial total page number count down register for channel x */
typedef struct {
  __REG32  PNCNTD         :16;
  __REG32                 :16;
} __hdma_ipncntd_bits;

/* HDMA page accumulation count register for channel x */
typedef struct {
  __REG32  PACNT          :16;
  __REG32                 :16;
} __hdma_pacnt_bits;

/* ENET CSR0 */
typedef struct {
  __REG32  SWR            : 1;
  __REG32  BAR            : 1;
  __REG32  DSL            : 5;
  __REG32  BLE            : 1;
  __REG32  PBL            : 6;
  __REG32                 : 3;
  __REG32  TAP            : 3;
  __REG32  DBO            : 1;
  __REG32                 :11;
} __enet_csr0_bits;

/* ENET CSR5 */
typedef struct {
  __REG32  TI             : 1;
  __REG32  TPS            : 1;
  __REG32  TU             : 1;
  __REG32                 : 2;
  __REG32  UNF            : 1;
  __REG32  RI             : 1;
  __REG32  RU             : 1;
  __REG32  RPS            : 1;
  __REG32                 : 1;
  __REG32  ETI            : 1;
  __REG32  GTE            : 1;
  __REG32                 : 2;
  __REG32  ERI            : 1;
  __REG32  AIS            : 1;
  __REG32  NIS            : 1;
  __REG32  RS             : 3;
  __REG32  TS             : 3;
  __REG32                 : 9;
} __enet_csr5_bits;

/* ENET CSR6 */
typedef struct {
  __REG32  HR             : 1;
  __REG32  SR             : 1;
  __REG32  HO             : 1;
  __REG32  PB             : 1;
  __REG32  IF             : 1;
  __REG32                 : 1;
  __REG32  PR             : 1;
  __REG32  PM             : 1;
  __REG32                 : 1;
  __REG32  FD             : 1;
  __REG32                 : 3;
  __REG32  ST             : 1;
  __REG32  TR             : 2;
  __REG32                 : 5;
  __REG32  SF             : 1;
  __REG32  TTM            : 1;
  __REG32                 : 7;
  __REG32  RA             : 1;
  __REG32  SS             : 1;
} __enet_csr6_bits;

/* ENET CSR7 */
typedef struct {
  __REG32  TIE            : 1;
  __REG32  TSE            : 1;
  __REG32  TUE            : 1;
  __REG32                 : 2;
  __REG32  UNE            : 1;
  __REG32  RIE            : 1;
  __REG32  RUE            : 1;
  __REG32  RSE            : 1;
  __REG32                 : 1;
  __REG32  ETE            : 1;
  __REG32  GTE            : 1;
  __REG32                 : 2;
  __REG32  ERE            : 1;
  __REG32  AIE            : 1;
  __REG32  NIE            : 1;
  __REG32                 :15;
} __enet_csr7_bits;

/* ENET CSR8 */
typedef struct {
  __REG32  MFC            :16;
  __REG32  MFO            : 1;
  __REG32  FOC            :11;
  __REG32  OCO            : 1;
  __REG32                 : 3;
} __enet_csr8_bits;

/* ENET CSR9 */
typedef struct {
  __REG32  SDS            : 1;
  __REG32  SDC            : 1;
  __REG32  SDI            : 1;
  __REG32  SDO            : 1;
  __REG32                 :12;
  __REG32  MDC            : 1;
  __REG32  MDO            : 1;
  __REG32  MII            : 1;
  __REG32  MDI            : 1;
  __REG32                 :12;
} __enet_csr9_bits;

/* ENET CSR11 */
typedef struct {
  __REG32  TIM            :16;
  __REG32  CON            : 1;
  __REG32  NRP            : 3;
  __REG32  RT             : 4;
  __REG32  NTP            : 3;
  __REG32  TT             : 4;
  __REG32  CS             : 1;
} __enet_csr11_bits;

/* PHY Test mode enable Register */
typedef struct {
  __REG32 PHY_TEST_CLK_EN   : 1;
  __REG32 PHY_VATEST_EN     : 1;
  __REG32                   :30;
} __udc_phy_test_en_bits;

/* PHY Test mode address and data input Register */
typedef struct {
  __REG32 PHY_TEST_ADDR     : 4;
  __REG32 PHY_TEST_DATA_IN  : 8;
  __REG32                   :20;
} __udc_phy_test_bits;


/* EHCI HCSPARAMS Structural Parameters */
typedef struct {
  __REG32 N_PORTS           : 4;
  __REG32 PPC               : 1;
  __REG32                   : 2;
  __REG32 PPR               : 1;
  __REG32 N_PCC             : 4;
  __REG32 N_CC              : 4;
  __REG32 PI                : 1;
  __REG32                   : 3;
  __REG32 N_DP              : 4;
  __REG32                   : 8;
} __hcsparams_bits;

/* EHCI HCCPARAMS Capability Parameters */
typedef struct {
  __REG32 _64_ADC           : 1;
  __REG32 PFL               : 1;
  __REG32 ASP               : 1;
  __REG32                   : 1;
  __REG32 IST               : 4;
  __REG32 EECP              : 8;
  __REG32                   :16;
} __hccparams_bits;

/* EHCI USBCMD */
typedef struct {
  __REG32 RS                : 1;
  __REG32 RESET             : 1;
  __REG32 FS                : 2;
  __REG32 PSE               : 1;
  __REG32 ASE               : 1;
  __REG32 IAA               : 1;
  __REG32 LR                : 1;
  __REG32 ASP               : 2;
  __REG32                   : 1;
  __REG32 ASPE              : 1;
  __REG32                   : 4;
  __REG32 ITC               : 8;
  __REG32                   : 8;
} __usbcmd_bits;

/* EHCI USBSTS - USB Status Register */
typedef struct {
  __REG32 USBINT            : 1;
  __REG32 USBERRINT         : 1;
  __REG32 PCD               : 1;
  __REG32 FLR               : 1;
  __REG32 SEI               : 1;
  __REG32 AAI               : 1;
  __REG32                   : 6;
  __REG32 HCH               : 1;
  __REG32 RCL               : 1;
  __REG32 PS                : 1;
  __REG32 AS                : 1;
  __REG32                   :16;
} __usbsts_bits;

/* EHCI USBINTR - USB Interrupt Enable Register */
typedef struct {
  __REG32 UE                : 1;
  __REG32 UEE               : 1;
  __REG32 PCE               : 1;
  __REG32 FRE               : 1;
  __REG32 SEE               : 1;
  __REG32 AAE               : 1;
  __REG32                   :26;
} __usbintr_bits;

/* EHCI FRINDEX - Frame Index Register */
typedef struct {
  __REG32 FRINDEX           :14;
  __REG32                   :18;
} __frindex_bits;

/* EHCI CONFIGFLAG - Configure Flag Register */
typedef struct {
  __REG32 CF                : 1;
  __REG32                   :31;
} __configflag_bits;

/* EHCI PORTSC - Port Status and Control Register */
typedef struct {
  __REG32 CCS               : 1;
  __REG32 CSC               : 1;
  __REG32 PE                : 1;
  __REG32 PEC               : 1;
  __REG32 OCA               : 1;
  __REG32 OCC               : 1;
  __REG32 FPR               : 1;
  __REG32 SUSPEND           : 1;
  __REG32 PR                : 1;
  __REG32                   : 1;
  __REG32 LS                : 2;
  __REG32 PP                : 1;
  __REG32 PO                : 1;
  __REG32 PIC               : 2;
  __REG32 PTC               : 4;
  __REG32 WKCNNT_E          : 1;
  __REG32 WKDSCNNT_E        : 1;
  __REG32 WKOC_E            : 1;
  __REG32                   : 9;
} __portsc_bits;

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
  __REG32 PSM               : 1;  /* ??*/
  __REG32 NPS               : 1;  /* ??*/
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
  __REG32 CCIC              : 1;
  __REG32                   :13;
  __REG32 CRWE              : 1;
} __hcrhstatus_bits;

/* HcRhPortStatus[1:2] Register */
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

/* Device Control Register */
typedef struct {
  __REG32 DEV_SPEED         : 2;
  __REG32 DEV_RMTWKP        : 1;
  __REG32 DEV_SELF_PWR      : 1;
  __REG32 DEV_SOFT_CN       : 1;
  __REG32 DEV_RESUME        : 1;
  __REG32 DEV_PHYBUS16_8    : 1;
  __REG32 SOFT_POR          : 1;
  __REG32 CSR_DONE          : 1;
  __REG32 TEST_MODE         : 1;
  __REG32                   :22;
} __udc_dev_ctl_bits;

/* Device Address Register */
typedef struct {
  __REG32 DEV_ADDR          : 7;
  __REG32 DEV_EN            : 1;
  __REG32 CFG_NUMBER        : 4;
  __REG32 NTF_NUMBER        : 4;
  __REG32 DEV_ALTINTF       : 4;
  __REG32 VBUS_SYNC         : 1;
  __REG32 DEV_SPEED         : 2;
  __REG32                   : 9;
} __udc_dev_info_bits;

/* Enable INT Status Register */
typedef struct {
  __REG32 EN_SOF_INTR       : 1;
  __REG32 EN_SETUP_INTR     : 1;
  __REG32 EN_IN0_INTR       : 1;
  __REG32 EN_OUT0_INTR      : 1;
  __REG32 EN_USBRST_INTR    : 1;
  __REG32 EN_RSUME_INTR     : 1;
  __REG32 EN_SUSP_INTR      : 1;
  __REG32                   : 1;
  __REG32 EN_BOUT1_INTR     : 1;
  __REG32 EN_BIN2_INTR      : 1;
  __REG32 EN_IIN3_INTR      : 1;
  __REG32 EN_BOUT4_INTR     : 1;
  __REG32 EN_BIN5_INTR      : 1;
  __REG32 EN_IIN6_INTR      : 1;
  __REG32 EN_BOUT7_INTR     : 1;
  __REG32 EN_BIN8_INTR      : 1;
  __REG32 EN_IIN9_INTR      : 1;
  __REG32 EN_BOUT10_INTR    : 1;
  __REG32 EN_BIN11_INTR     : 1;
  __REG32 EN_IIN12_INTR     : 1;
  __REG32 EN_BOUT13_INTR    : 1;
  __REG32 EN_BIN14_INTR     : 1;
  __REG32 EN_IIN15_INTR     : 1;
  __REG32 TEST_SE0_NAK      : 1;
  __REG32 TEST_J            : 1;
  __REG32 TEST_K            : 1;
  __REG32 TEST_PKT          : 1;
  __REG32                   : 5;
} __udc_en_int_bits;

/* AP_INT2FLAG Status Register */
typedef struct {
  __REG32 SOF_INTR          : 1;
  __REG32 SETUP_INTR        : 1;
  __REG32 IN0_INTR          : 1;
  __REG32 OUT0_INTR         : 1;
  __REG32 USBRST_INTR       : 1;
  __REG32 RSUME_INTR        : 1;
  __REG32 SUSP_INTR         : 1;
  __REG32 VBUS_INTR         : 1;
  __REG32 BOUT1_INTR        : 1;
  __REG32 BIN2_INTR         : 1;
  __REG32 IIN3_INTR         : 1;
  __REG32 BOUT4_INTR        : 1;
  __REG32 BIN5_INTR         : 1;
  __REG32 IIN6_INTR         : 1;
  __REG32 BOUT7_INTR        : 1;
  __REG32 BIN8_INTR         : 1;
  __REG32 IIN9_INTR         : 1;
  __REG32 BOUT10_INTR       : 1;
  __REG32 BIN11_INTR        : 1;
  __REG32 IIN12_INTR        : 1;
  __REG32 BOUT13_INTR       : 1;
  __REG32 BIN14_INTR        : 1;
  __REG32 IIN15_INTR        : 1;
  __REG32 TEST_SE0_NAK      : 1;
  __REG32 TEST_J            : 1;
  __REG32 TEST_K            : 1;
  __REG32 TEST_PKT          : 1;
  __REG32                   : 5;
} __udc_int2flag_bits;

/* Interrupt control Register */
typedef struct {
  __REG32 INT0EN            : 1;
  __REG32 INT0TYPE          : 1;
  __REG32 INT0MODE          : 1;
  __REG32                   :29;
} __udc_intcon_bits;

/* Standard Setup bmRequestType, bRequest, wValue Register */
typedef struct {
  __REG32 BmRequestType     : 8;
  __REG32 BRequest          : 8;
  __REG32 WValue            :16;
} __udc_setup1_bits;

/* Standard Setup wIndex, wLength Register */
typedef struct {
  __REG32 WIndex            :16;
  __REG32 WLength           :16;
} __udc_setup2_bits;

/* AHB Control Register */
typedef struct {
  __REG32 MID               : 4;
  __REG32                   :28;
} __udc_ahbcon_bits;

/* Endpoint 0 Receive status Register */
typedef struct {
  __REG32 RX0LEN            :11;
  __REG32                   : 5;
  __REG32 RX0VOID           : 1;
  __REG32 RX0ERR            : 1;
  __REG32 RX0ACK            : 1;
  __REG32                   : 5;
  __REG32 RX0FULL           : 1;
  __REG32 RX0OVF            : 1;
  __REG32                   : 6;
} __udc_rx0stat_bits;

/* Endpoint 0 Receive Control Register */
typedef struct {
  __REG32 RX0FFRC           : 1;
  __REG32 RX0CLR            : 1;
  __REG32 RX0STALL          : 1;
  __REG32 RX0NAK            : 1;
  __REG32 EP0EN             : 1;
  __REG32 RX0VOIDINTEN      : 1;
  __REG32 RX0ERRINTEN       : 1;
  __REG32 RX0ACKINTEN       : 1;
  __REG32                   :24;
} __udc_rx0con_bits;

/* DMA0 Control OUT Register */
typedef struct {
  __REG32 DMA0OUTSTA        : 1;
  __REG32                   :31;
} __udc_dma0ctlo_bits;

/* Endpoint 0 Transmit Status Register */
typedef struct {
  __REG32 TX0LEN            :11;
  __REG32                   : 5;
  __REG32 TX0VOID           : 1;
  __REG32 TX0ERR            : 1;
  __REG32 TX0ACK            : 1;
  __REG32                   :13;
} __udc_tx0stat_bits;

/* Endpoint 0 Transmit Control Register */
typedef struct {
  __REG32 TX0CLR            : 1;
  __REG32 TX0STALL          : 1;
  __REG32 TX0NAK            : 1;
  __REG32                   : 1;
  __REG32 TX0VOIDINTEN      : 1;
  __REG32 TX0ERRINTEN       : 1;
  __REG32 TX0ACKINTEN       : 1;
  __REG32                   :25;
} __udc_tx0con_bits;

/* Endpoint 0 Buffer Status Register */
typedef struct {
  __REG32 TX0FULL           : 1;
  __REG32 TX0URF            : 1;
  __REG32                   :30;
} __udc_tx0buf_bits;

/* DMA0 Control IN Register */
typedef struct {
  __REG32 DMA0INSTA         : 1;
  __REG32                   :31;
} __udc_dma0ctli_bits;

/* Receiver status Register */
typedef struct {
  __REG32 RXCNT             :11;
  __REG32                   : 5;
  __REG32 RXVOID            : 1;
  __REG32 RXERR             : 1;
  __REG32 RXACK             : 1;
  __REG32 RX_CF_INT         : 1;
  __REG32                   : 4;
  __REG32 RXFULL            : 1;
  __REG32 RXOVF             : 1;
  __REG32                   : 6;
} __udc_rxstat_bits;

/* Receiver control Register */
typedef struct {
  __REG32 RXFFRC            : 1;
  __REG32 RXCLR             : 1;
  __REG32 RXSTALL           : 1;
  __REG32 RXNAK             : 1;
  __REG32 EPEN              : 1;
  __REG32 RXVOIDINTEN       : 1;
  __REG32 RXERRINTEN        : 1;
  __REG32 RXACKINTEN        : 1;
  __REG32 RXENDP_NUM        : 4;
  __REG32 RX_CF_INTE        : 1;
  __REG32 RXSTALL_AUTOCLR   : 1;
  __REG32                   :18;
} __udc_rxcon_bits;

/* DMA Control Register */
typedef struct {
  __REG32 DMAOUTSTA         : 1;
  __REG32                   :31;
} __udc_dmactlo_bits;

/* Transmitter status Register */
typedef struct {
  __REG32 TXLEN             :11;
  __REG32                   : 5;
  __REG32 TXVOID            : 1;
  __REG32 TXERR             : 1;
  __REG32 TXACK             : 1;
  __REG32 TXDMA_DN          : 1;
  __REG32 TX_CF_INT         : 1;
  __REG32                   :11;
} __udc_txstat_bits;

/* Transmitter control Register */
typedef struct {
  __REG32 TXCLR             : 1;
  __REG32 TXSTALL           : 1;
  __REG32 TXNAK             : 1;
  __REG32 TXEPEN            : 1;
  __REG32 TXVOIDINTEN       : 1;
  __REG32 TXERRINTEN        : 1;
  __REG32 TXACKINTEN        : 1;
  __REG32 TXDMADN_EN        : 1;
  __REG32 TXENDP_NUM        : 4;
  __REG32 TX_CF_INTE        : 1;
  __REG32 TXSTALL_AUTOCLR   : 1;
  __REG32                   :18;
} __udc_txcon_bits;

/* Transmitter buffer Status Register */
typedef struct {
  __REG32 TXFULL            : 1;
  __REG32 TXURF             : 1;
  __REG32 TXDS0             : 1;
  __REG32 TXDS1             : 1;
  __REG32                   :28;
} __udc_txbuf_bits;

/* DMA Control Register */
typedef struct {
  __REG32 DMAINSTA          : 1;
  __REG32                   :31;
} __udc_dmactli_bits;

/* Transmitter Status Register */
typedef struct {
  __REG32 TXLEN             :11;
  __REG32                   : 5;
  __REG32 TXVOID            : 1;
  __REG32 TXERR             : 1;
  __REG32 TXACK             : 1;
  __REG32 TX_CF_INT         : 1;
  __REG32                   :12;
} __udc_tx3stat_bits;

/* Transmitter control Register */
typedef struct {
  __REG32 TXCLR             : 1;
  __REG32 TXSTALL           : 1;
  __REG32 TXNAK             : 1;
  __REG32 TXEPEN            : 1;
  __REG32 TXVOIDINTEN       : 1;
  __REG32 TXERRINTEN        : 1;
  __REG32 TXACKINTEN        : 1;
  __REG32                   : 1;
  __REG32 TXENDP_NUM        : 4;
  __REG32 TX_CF_INTE        : 1;
  __REG32 TXSTALL_AUTOCLR   : 1;
  __REG32                   :18;
} __udc_tx3con_bits;

/* Transmitter buffer Status Register */
typedef struct {
  __REG32 TXFULL            : 1;
  __REG32 TXURF             : 1;
  __REG32                   :30;
} __udc_tx3buf_bits;

/* LCD Control Register */
typedef struct {
  __REG32                   : 1;
  __REG32 LCDC_EN           : 1;
  __REG32 RGB_HALT          : 1;
  __REG32 DTMG_LEVEL        : 1;
  __REG32 VSYNC_LEVEL       : 1;
  __REG32 HSYNC_LEVEL       : 1;
  __REG32 LUT_EN            : 1;
  __REG32 COLOR_DEPTH       : 1;
  __REG32 PCLK_DIV          : 8;
  __REG32 SRAM_SEL          : 1;
  __REG32 LUM_CFG           : 3;
  __REG32 COLOR_SEN         : 2;
  __REG32                   :10;
} __lcd_ctrl0_bits;

/* Horizontal Timing Register */
typedef struct {
  __REG32 HACTIVE           :10;
  __REG32 HRST              : 6;
  __REG32 HBACKPORCH        : 8;
  __REG32 HFRONTPORCH       : 8;
} __lcd_ht_bits;

/* Vertical Timing Register */
typedef struct {
  __REG32 VACTIVE           :10;
  __REG32 VRST              : 6;
  __REG32 VBACKPORCH        : 8;
  __REG32 VFRONTPORCH       : 8;
} __lcd_vt_bits;

/* Interrupt Status */
typedef struct {
  __REG32 PAGE0_READ        : 1;
  __REG32 PAGE1_READ        : 1;
  __REG32 LUT_LOAD_DONE     : 1;
  __REG32 FIFO_EMPTY        : 1;
  __REG32 OUT_HALTED        : 1;
  __REG32 DMA_ERR           : 1;
  __REG32                   :26;
} __lcd_intr_sta_bits;

/* UART_IER */
typedef struct {
  __REG32 RDAIE             : 1;
  __REG32 THREIE            : 1;
  __REG32 RXLSIE            : 1;
  __REG32 RXMSIE            : 1;
  __REG32                   :28;
} __uart_ier_bits;

/* UART interrupt identification register and fifo control register */
typedef union {
  /*UARTx_IIR*/
  struct {
__REG32 IP     : 1;
__REG32 IID    : 3;
__REG32        : 2;
__REG32 IIRFE  : 2;
__REG32        :24;
  };
  /*UARTx_FCR*/
  struct {
__REG32 FCRFE  : 1;
__REG32 RFR    : 1;
__REG32 TFR    : 1;
__REG32        : 3;
__REG32 RTLS   : 2;
__REG32        :24;
  };
} __uart_iir_bits;

/* UART_LCR */
typedef struct {
  __REG32 WLS               : 2;
  __REG32 SBS               : 1;
  __REG32 PE                : 1;
  __REG32 PS                : 2;
  __REG32 BC                : 1;
  __REG32 DLAB              : 1;
  __REG32                   :24;
} __uart_lcr_bits;

/* UART_MCR */
typedef struct {
  __REG32                   : 1;
  __REG32 RTS               : 1;
  __REG32                   :30;
} __uart_mcr_bits;

/* UART_LSR */
typedef struct {
  __REG32 DR                : 1;
  __REG32 OE                : 1;
  __REG32 PE                : 1;
  __REG32 FE                : 1;
  __REG32 BI                : 1;
  __REG32 THRE              : 1;
  __REG32 TEMT              : 1;
  __REG32 RXFE              : 1;
  __REG32                   :24;
} __uart_lsr_bits;

/* UART_MSR */
typedef struct {
  __REG32                   : 4;
  __REG32 CTS               : 1;
  __REG32                   :27;
} __uart_msr_bits;

/* GPIOA */
typedef struct {
  __REG32 PA0               : 1;
  __REG32 PA1               : 1;
  __REG32 PA2               : 1;
  __REG32 PA3               : 1;
  __REG32 PA4               : 1;
  __REG32 PA5               : 1;
  __REG32 PA6               : 1;
  __REG32 PA7               : 1;
  __REG32                   :24;
} __gpio_padr_bits;

/* GPIOB */
typedef struct {
  __REG32 PB0               : 1;
  __REG32 PB1               : 1;
  __REG32 PB2               : 1;
  __REG32 PB3               : 1;
  __REG32 PB4               : 1;
  __REG32 PB5               : 1;
  __REG32 PB6               : 1;
  __REG32 PB7               : 1;
  __REG32                   :24;
} __gpio_pbdr_bits;

/* GPIOC */
typedef struct {
  __REG32 PC0               : 1;
  __REG32 PC1               : 1;
  __REG32 PC2               : 1;
  __REG32 PC3               : 1;
  __REG32 PC4               : 1;
  __REG32 PC5               : 1;
  __REG32 PC6               : 1;
  __REG32 PC7               : 1;
  __REG32                   :24;
} __gpio_pcdr_bits;

/* GPIOD */
typedef struct {
  __REG32 PD0               : 1;
  __REG32 PD1               : 1;
  __REG32 PD2               : 1;
  __REG32 PD3               : 1;
  __REG32 PD4               : 1;
  __REG32 PD5               : 1;
  __REG32 PD6               : 1;
  __REG32 PD7               : 1;
  __REG32                   :24;
} __gpio_pddr_bits;

/* GPIO function test register. */
typedef struct {
  __REG32 TEST_EN           : 1;
  __REG32 TEST_MODE         : 2;
  __REG32                   :29;
} __gpio_test_bits;

/* GPIO_ISR */
typedef struct {
  __REG32 PA0               : 1;
  __REG32 PA1               : 1;
  __REG32 PA2               : 1;
  __REG32 PA3               : 1;
  __REG32 PA4               : 1;
  __REG32 PA5               : 1;
  __REG32 PA6               : 1;
  __REG32 PA7               : 1;
  __REG32 PB0               : 1;
  __REG32 PB1               : 1;
  __REG32 PB2               : 1;
  __REG32 PB3               : 1;
  __REG32 PB4               : 1;
  __REG32 PB5               : 1;
  __REG32 PB6               : 1;
  __REG32 PB7               : 1;
  __REG32 PC0               : 1;
  __REG32 PC1               : 1;
  __REG32 PC2               : 1;
  __REG32 PC3               : 1;
  __REG32 PC4               : 1;
  __REG32 PC5               : 1;
  __REG32 PC6               : 1;
  __REG32 PC7               : 1;
  __REG32 PD0               : 1;
  __REG32 PD1               : 1;
  __REG32 PD2               : 1;
  __REG32 PD3               : 1;
  __REG32 PD4               : 1;
  __REG32 PD5               : 1;
  __REG32 PD6               : 1;
  __REG32 PD7               : 1;
} __gpio_isr_bits;

/* TMRxCON */
typedef struct {
  __REG32                   : 1;
  __REG32 LEVEL_TRIG        : 1;
  __REG32 INTR_CLR          : 1;
  __REG32 INTR_MASK         : 1;
  __REG32 PRESCALER         : 3;
  __REG32 PERIODICAL        : 1;
  __REG32 ENABLE            : 1;
  __REG32                   :23;
} __tmrcon_bits;

/* TMRMODE */
typedef struct {
  __REG32 MODE              : 2;
  __REG32                   :30;
} __tmrmode_bits;

/* WDTCON */
typedef struct {
  __REG32 PRESCALER         : 3;
  __REG32 ENABLE            : 1;
  __REG32 RESET_ENABLE      : 1;
  __REG32                   :27;
} __wdtcon_bits;

/* RTC time register */
typedef struct {
  __REG32 SOS               : 4;
  __REG32 S                 : 4;
  __REG32 TS                : 3;
  __REG32 M                 : 4;
  __REG32 TM                : 3;
  __REG32 H                 : 4;
  __REG32 TH                : 2;
  __REG32 DOW               : 3;
  __REG32                   : 5;
} __rtc_time_bits;

/* RTC date register */
typedef struct {
  __REG32 D                 : 4;
  __REG32 TD                : 2;
  __REG32 M                 : 4;
  __REG32 TM                : 1;
  __REG32 Y                 : 4;
  __REG32 TY                : 4;
  __REG32 C                 : 4;
  __REG32 TC                : 4;
  __REG32                   : 5;
} __rtc_date_bits;

/* RTC time alarm register */
typedef struct {
  __REG32 SOS               : 4;
  __REG32 S                 : 4;
  __REG32 TS                : 3;
  __REG32 M                 : 4;
  __REG32 TM                : 3;
  __REG32 H                 : 4;
  __REG32 TH                : 2;
  __REG32 DOW               : 3;
  __REG32 CSOS              : 1;
  __REG32 CS                : 1;
  __REG32 CM                : 1;
  __REG32 CH                : 1;
  __REG32 CDOW              : 1;
} __rtc_talrm_bits;

/* RTC date register */
typedef struct {
  __REG32 D                 : 4;
  __REG32 TD                : 2;
  __REG32 M                 : 4;
  __REG32 TM                : 1;
  __REG32 Y                 : 4;
  __REG32 TY                : 4;
  __REG32 C                 : 4;
  __REG32 TC                : 4;
  __REG32 CD                : 1;
  __REG32 CM                : 1;
  __REG32 CY                : 1;
  __REG32 CC                : 1;
  __REG32                   : 1;
} __rtc_dalrm_bits;

/* RTC control register */
typedef struct {
  __REG32 DIV               :27;
  __REG32 SOS               : 1;
  __REG32                   : 2;
  __REG32 ALRM              : 1;
  __REG32 EN                : 1;
} __rtc_ctrl_bits;

/* RTC control reset register */
typedef struct {
  __REG32 CRESET            : 1;
  __REG32 RTCRESET          : 1;
  __REG32                   :30;
} __rtc_reset_bits;

/* SPI_TxR, SPI_RxR */
typedef struct {
  __REG32 DATA              :16;
  __REG32                   :16;
} __spi_txr_bits;

/* SPI_IER */
typedef struct {
  __REG32 RxCIEN            : 1;
  __REG32 RxFOIEN           : 1;
  __REG32 RxFIEN            : 1;
  __REG32 TxFIEN            : 1;
  __REG32                   :28;
} __spi_ier_bits;

/* SPI_FCR */
typedef struct {
  __REG32 RxDAF             : 1;
  __REG32 TxFF              : 1;
  __REG32                   : 6;
  __REG32 TxFL              : 3;
  __REG32 RxFL              : 3;
  __REG32                   :18;
} __spi_fcr_bits;

/* SPI_FWCR */
typedef struct {
  __REG32 LBKMD             : 1;
  __REG32 BIDIREN           : 1;
  __REG32 LSBEN             : 1;
  __REG32 CPHA              : 1;
  __REG32 CPOL              : 1;
  __REG32 TxRxsten          : 1;
  __REG32 SPIDMA            : 1;
  __REG32                   : 1;
  __REG32 CKIDLEN           : 1;
  __REG32 SPIRUN            : 1;
  __REG32 SPIEN             : 1;
  __REG32 SRST_N            : 1;
  __REG32                   :20;
} __spi_fwcr_bits;

/* SPI_DLYCR */
typedef struct {
  __REG32 PBCA              : 3;
  __REG32 PBCT              : 3;
  __REG32                   : 2;
  __REG32 PBTxRx            : 3;
  __REG32                   :21;
} __spi_dlycr_bits;

/* SPI_TxC */
typedef struct {
  __REG32 SIZE              :16;
  __REG32                   :16;
} __spi_txcr_bits;

/* SPI_RxCR */
typedef struct {
  __REG32 SIZE              :16;
  __REG32                   :16;
} __spi_rxcr_bits;

/* SPI_SSCR */
typedef struct {
  __REG32 SPIDIVR           : 6;
  __REG32                   : 2;
  __REG32 SPISSR            : 3;
  __REG32 SPICHRL           : 4;
  __REG32                   :17;
} __spi_sscr_bits;

/* SPI_ISR */
typedef struct {
  __REG32 RxCIF             : 1;
  __REG32 SPIORIF           : 1;
  __REG32 RxFIF             : 1;
  __REG32 TxFIF             : 1;
  __REG32                   :28;
} __spi_isr_bits;

/* I2C_MTXR */
typedef struct {
  __REG32 DATA              : 8;
  __REG32                   :24;
} __i2c_mtxr_bits;

/* I2C_SADDR */
typedef struct {
  __REG32 SA                :10;
  __REG32                   :22;
} __i2c_saddr_bits;

/* I2C_IER */
typedef struct {
  __REG32 MRACK             : 1;
  __REG32 MPACK             : 1;
  __REG32 SRACK             : 1;
  __REG32 SPACK             : 1;
  __REG32 SAM               : 1;
  __REG32 BAM               : 1;
  __REG32 AS                : 1;
  __REG32 AL                : 1;
  __REG32                   :24;
} __i2c_ier_bits;

/* I2C_LCMR */
typedef struct {
  __REG32 START             : 1;
  __REG32 STOP              : 1;
  __REG32 RESUME            : 1;
  __REG32                   :29;
} __i2c_lcmr_bits;

/* I2C_LSR */
typedef struct {
  __REG32 BUSY              : 1;
  __REG32 NAK               : 1;
  __REG32                   :30;
} __i2c_lsr_bits;

/* I2C_CONR */
typedef struct {
  __REG32 SPE               : 1;
  __REG32 ST                : 1;
  __REG32 MPE               : 1;
  __REG32 MT                : 1;
  __REG32 NAK               : 1;
  __REG32                   :27;
} __i2c_conr_bits;

/* I2C_OPR */
typedef struct {
  __REG32 I2CCDVR           : 6;
  __REG32 EN                : 1;
  __REG32 RST               : 1;
  __REG32 AS                : 1;
  __REG32                   :23;
} __i2c_opr_bits;

/* I2S_OPR */
typedef struct {
  __REG32 RX_START          : 1;
  __REG32 TX_START          : 1;
  __REG32 LOOPBACK          : 1;
  __REG32                   :13;
  __REG32 RX_RST            : 1;
  __REG32 TX_RST            : 1;
  __REG32                   :14;
} __i2s_opr_bits;

/* I2S_TXCTL */
typedef struct {
  __REG32 MASTER            : 1;
  __REG32 BUS_MODE          : 2;
  __REG32 MONO              : 1;
  __REG32 DATA_RES          : 2;
  __REG32                   : 2;
  __REG32 DIV               : 8;
  __REG32 OVR_SAMP          : 2;
  __REG32 TRAN_DEV          : 2;
  __REG32                   :12;
} __i2s_txctl_bits;

/* I2S_RXCTL */
typedef struct {
  __REG32 MASTER            : 1;
  __REG32 BUS_MODE          : 2;
  __REG32 MONO              : 1;
  __REG32 DATA_RES          : 2;
  __REG32                   : 2;
  __REG32 DIV               : 8;
  __REG32 OVR_SAMP          : 2;
  __REG32                   : 6;
  __REG32 FIFO_RST          : 1;
  __REG32                   : 7;
} __i2s_rxctl_bits;

/* I2S_FIFOSTS */
typedef struct {
  __REG32 RX_FE             : 1;
  __REG32 RX_FF             : 1;
  __REG32 TX_FE             : 1;
  __REG32 TX_FF             : 1;
  __REG32 RX_FAE            : 1;
  __REG32 RX_FAF            : 1;
  __REG32 TX_FAE            : 1;
  __REG32 TX_FAF            : 1;
  __REG32 RX_FHF            : 1;
  __REG32 TX_FHF            : 1;
  __REG32                   : 6;
  __REG32 RX_ITL            : 2;
  __REG32 TX_ITL            : 2;
  __REG32                   :12;
} __i2s_fifosts_bits;

/* I2S_IER */
typedef struct {
  __REG32 RX_OVR            : 1;
  __REG32 RX                : 1;
  __REG32 TX                : 1;
  __REG32                   :29;
} __i2s_ier_bits;

/* MMU control register */
typedef struct {
  __REG32 DATA1_WIDTH       : 2;
  __REG32 DATA1_END         : 1;
  __REG32 DATA1_LOOP        : 1;
  __REG32 DATA2_WIDTH       : 2;
  __REG32 DATA2_END         : 1;
  __REG32 DATA2_LOOP        : 1;
  __REG32 CPU_DATA_SEL      : 1;
  __REG32 MMU_DATA_SEL      : 1;
  __REG32 MMU_DMA_WRITE     : 1;
  __REG32 MMU_DMA_START     : 1;
  __REG32 CPU_ENDIAN        : 1;
  __REG32                   :19;
} __mmu_ctrl_bits;

/* MMU pointer I setting register */
typedef struct {
  __REG32 DATA1_POINT       :11;
  __REG32                   :21;
} __mmu_pnri_bits;

/* MMU pointer II setting register */
typedef struct {
  __REG32 DATA2_POINT       :11;
  __REG32                   :21;
} __mmu_pnrii_bits;

/* MMU address setting register */
typedef struct {
  __REG32 ADDR              :24;
  __REG32                   : 8;
} __mmu_addr_bits;

/* SD/MMC Host control register */
typedef struct {
  __REG32 DIV               :11;
  __REG32 CLK_EN            : 1;
  __REG32 CARD_DET          : 1;
  __REG32 POWER_CTRL        : 1;
  __REG32                   :18;
} __sd_ctrl_bits;

/* SD/MMC Host interrupt register */
typedef struct {
  __REG32 CDIE              : 1;
  __REG32 DTIE              : 1;
  __REG32 CTIE              : 1;
  __REG32                   : 1;
  __REG32 CDIF              : 1;
  __REG32 DTIF              : 1;
  __REG32 CTIF              : 1;
  __REG32                   :25;
} __sd_int_bits;

/* SD/MMC card register */
typedef struct {
  __REG32 CD                : 1;
  __REG32 WP                : 1;
  __REG32 CB                : 1;
  __REG32                   : 1;
  __REG32 CDIE              : 1;
  __REG32 CPCE              : 1;
  __REG32 CSE               : 1;
  __REG32                   :25;
} __sd_card_bits;

/* SD/MMC command and response transfer register */
typedef struct {
  __REG32 CMD               : 6;
  __REG32                   : 2;
  __REG32 CMD_ERR           : 1;
  __REG32 RES_TYPE          : 3;
  __REG32 RES_TRAN          : 1;
  __REG32 CMD_TRAN          : 1;
  __REG32                   :18;
} __sd_cmdrest_bits;

/* SD/MMC command and response transfer status register */
typedef struct {
  __REG32 CREBE             : 1;
  __REG32 CRCRCE            : 1;
  __REG32 CRIE              : 1;
  __REG32 CRTBE             : 1;
  __REG32 CRTE              : 1;
  __REG32 CCRBCE            : 1;
  __REG32 CCRE              : 1;
  __REG32 CRT               : 1;
  __REG32 CCT               : 1;
  __REG32                   :23;
} __sd_cmdres_bits;

/* SD/MMC data transfer register */
typedef struct {
  __REG32 WDTCRCS           : 3;
  __REG32 RDTEBE            : 1;
  __REG32 RDTSBE            : 1;
  __REG32 DTCRCE            : 1;
  __REG32 DTTE              : 1;
  __REG32 DTBCE             : 1;
  __REG32 DTES              : 1;
  __REG32 DTC               : 1;
  __REG32 DTWDMAI           : 1;
  __REG32 DTBW              : 1;
  __REG32 DTD               : 1;
  __REG32 DTS               : 1;
  __REG32                   :18;
} __sd_datat_bits;

/* PWMT_CTRL */
typedef struct {
  __REG32 PWM_EN            : 1;
  __REG32                   : 2;
  __REG32 PWM_OUT_EN        : 1;
  __REG32 SINGLE_CNT_MODE   : 1;
  __REG32 PWM_INTR_EN       : 1;
  __REG32 PWM_INTR_CLR      : 1;
  __REG32 PWM_RST           : 1;
  __REG32 CAP_EN            : 1;
  __REG32 PRESCALER         : 5;
  __REG32                   :18;
} __pwmt_ctrl_bits;

/* ADC_DATA */
typedef struct {
  __REG32 DATA              :10;
  __REG32                   :22;
} __adc_data_bits;

/* ADC_STAS */
typedef struct {
  __REG32 RUN               : 1;
  __REG32                   :31;
} __adc_stas_bits;

/* ADC_CTRL */
typedef struct {
  __REG32 CH_SEL            : 3;
  __REG32 PD                : 1;
  __REG32 SOC               : 1;
  __REG32 IE                : 1;
  __REG32 IF                : 1;
  __REG32                   :25;
} __adc_ctrl_bits;

/* flsh_conf */
typedef struct {
  __REG32 CHIPS             : 2;
  __REG32 BUS_WIDTH         : 1;
  __REG32 PAGE_SIZE         : 2;
  __REG32 ADDR_CYCLES       : 2;
  __REG32 TRP               : 3;
  __REG32 TRC               : 4;
  __REG32 TWP               : 3;
  __REG32 TWC               : 4;
  __REG32                   :11;
} __flsh_conf_bits;

/* flsh_comm_1 */
typedef struct {
  __REG32 CMD               : 8;
  __REG32 CMD_VAL           : 1;
  __REG32                   :23;
} __flsh_comm_bits;

/* flsh_address_1 */
typedef struct {
  __REG32 ADDR              : 8;
  __REG32 ADDR_VAL          : 1;
  __REG32                   :23;
} __flsh_address_bits;

/* flsh_buff_staddr */
typedef struct {
  __REG32 OFFSET            :16;
  __REG32                   :16;
} __flsh_buff_staddr_bits;

/* flsh_buff_cnt */
typedef struct {
  __REG32 SIZE              :16;
  __REG32 ECC_EN            : 1;
  __REG32                   :15;
} __flsh_buff_cnt_bits;

/* flsh_buff_state */
typedef struct {
  __REG32 PAGE0_DATA        : 1;
  __REG32 PAGE1_DATA        : 1;
  __REG32 PAGE2_DATA        : 1;
  __REG32 PAGE3_DATA        : 1;
  __REG32 PAGE4_DATA        : 1;
  __REG32 PAGE5_DATA        : 1;
  __REG32 PAGE6_DATA        : 1;
  __REG32 PAGE7_DATA        : 1;
  __REG32                   :24;
} __flsh_buff_state_bits;

/* flsh_DMA_set */
typedef struct {
  __REG32 DEV0_BUSY         : 1;
  __REG32 DEV1_BUSY         : 1;
  __REG32 DEV2_BUSY         : 1;
  __REG32 DEV3_BUSY         : 1;
  __REG32 DMA_DIR           : 2;
  __REG32 DATA_PATH_SEL     : 1;
  __REG32 AUTO_STA_EN       : 1;
  __REG32 SEC_CMD           : 1;
  __REG32 BUFF_ACC_EN       : 1;
  __REG32 ECC_TEST_MODE     : 1;
  __REG32 BIG_ENDIAN        : 1;
  __REG32                   :20;
} __flsh_dma_set_bits;

/* flsh_CE_WP */
typedef struct {
  __REG32 CS0               : 1;
  __REG32 CS1               : 1;
  __REG32 CS2               : 1;
  __REG32 CS3               : 1;
  __REG32 WP0               : 1;
  __REG32 WP1               : 1;
  __REG32 WP2               : 1;
  __REG32 WP3               : 1;
  __REG32                   :24;
} __flsh_ce_wp_bits;

/* flsh_control */
typedef struct {
  __REG32 ACC_CMD_EN        : 1;
  __REG32 ACC_DATA_EN       : 1;
  __REG32 DMA_EN            : 1;
  __REG32                   :29;
} __flsh_control_bits;

/* flsh_reset */
typedef struct {
  __REG32 RST_EN            : 1;
  __REG32                   :31;
} __flsh_reset_bits;

/* flsh_state */
typedef struct {
  __REG32 BUSY0             : 1;
  __REG32 BUSY1             : 1;
  __REG32 BUSY2             : 1;
  __REG32 BUSY3             : 1;
  __REG32                   :28;
} __flsh_state_bits;

/* flsh_int_mask */
typedef struct {
  __REG32 WDE               : 1;
  __REG32 WDD               : 1;
  __REG32 TO                : 1;
  __REG32 SE                : 1;
  __REG32 RDD               : 1;
  __REG32 BEF               : 1;
  __REG32 BED               : 1;
  __REG32 BDE               : 1;
  __REG32                   :24;
} __flsh_int_mask_bits;

/* flsh_GPIO */
typedef struct {
  __REG32 GPIO              : 1;
  __REG32                   :31;
} __flsh_gpio_bits;

/* flsh_S_num */
typedef struct {
  __REG32 SER_NUM           : 8;
  __REG32                   :24;
} __flsh_s_num_bits;

/* flsh_Syndr_1 */
typedef struct {
  __REG32 SYNDR1            :10;
  __REG32 SYNDR2            :10;
  __REG32 SYNDR3            :10;
  __REG32                   : 2;
} __flsh_syndr_1_bits;

/* flsh_Syndr_2 */
typedef struct {
  __REG32 SYNDR4            :10;
  __REG32 SYNDR5            :10;
  __REG32 SYNDR6            :10;
  __REG32                   : 2;
} __flsh_syndr_2_bits;

/* flsh_Syndr_3 */
typedef struct {
  __REG32 SYNDR7            :10;
  __REG32 SYNDR8            :10;
  __REG32 SYNDR9            :10;
  __REG32                   : 2;
} __flsh_syndr_3_bits;

/* flsh_Syndr_4 */
typedef struct {
  __REG32 SYNDR10           :10;
  __REG32 SYNDR11           :10;
  __REG32 SYNDR12           :10;
  __REG32                   : 2;
} __flsh_syndr_4_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
/***************************************************************************
 **
 **  System Control Unit (SCU)
 **
 ***************************************************************************/
__IO_REG32(    SCU_P7CID,             0x1803C000,__READ       );
__IO_REG32_BIT(SCU_PLLPARAM_A,        0x1803C004,__READ_WRITE ,__scu_pllparam_a_bits);
__IO_REG32_BIT(SCU_PLLPARAM_B,        0x1803C008,__READ_WRITE ,__scu_pllparam_b_bits);
__IO_REG32_BIT(SCU_CHIPCFG_A,         0x1803C00C,__READ_WRITE ,__scu_chipcfg_a_bits);
__IO_REG32_BIT(SCU_CHIPCFG_B,         0x1803C010,__READ_WRITE ,__scu_chipcfg_b_bits);
__IO_REG32_BIT(SCU_CLKCFG,            0x1803C014,__READ_WRITE ,__scu_clkcfg_bits);
__IO_REG32_BIT(SCU_CHIPCFG_C,         0x1803C018,__WRITE      ,__scu_chipcfg_c_bits);

/***************************************************************************
 **
 **  Cache Controller (CC)
 **
 ***************************************************************************/
__IO_REG32_BIT(CacheDevID,            0xEFFF0000,__READ_WRITE ,__cachedevid_bits);
__IO_REG32_BIT(CacheOp,               0xEFFF0004,__READ_WRITE ,__cacheop_bits);
__IO_REG32_BIT(CacheLKDN,             0xEFFF0008,__READ_WRITE ,__cachelkdn_bits);
__IO_REG32(    MemMapA,               0xEFFF0010,__READ_WRITE );
__IO_REG32(    MemMapB,               0xEFFF0014,__READ_WRITE );
__IO_REG32(    MemMapC,               0xEFFF0018,__READ_WRITE );
__IO_REG32(    MemMapD,               0xEFFF001C,__READ_WRITE );
__IO_REG32_BIT(PFCNTRA_CTRL,          0xEFFF0020,__READ_WRITE ,__pfcntr_ctrl_bits);
__IO_REG32(    PFCNTRA,               0xEFFF0024,__READ_WRITE );
__IO_REG32_BIT(PFCNTRB_CTRL,          0xEFFF0028,__READ_WRITE ,__pfcntr_ctrl_bits);
__IO_REG32(    PFCNTRB,               0xEFFF002C,__READ_WRITE );

/***************************************************************************
 **
 **  AHB Bus Arbiter (AHBBA)
 **
 ***************************************************************************/
__IO_REG32_BIT(ARB_MODE,              0x1806C000,__READ_WRITE ,__arb_mode_bits);
__IO_REG32_BIT(ARB_PRIO1,             0x1806C004,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO2,             0x1806C008,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO3,             0x1806C00C,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO4,             0x1806C010,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO5,             0x1806C014,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO6,             0x1806C018,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO7,             0x1806C01C,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO8,             0x1806C020,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO9,             0x1806C024,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO10,            0x1806C028,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO11,            0x1806C02C,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO12,            0x1806C030,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO13,            0x1806C034,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO14,            0x1806C038,__READ_WRITE ,__arb_prio_bits);
__IO_REG32_BIT(ARB_PRIO15,            0x1806C03C,__READ_WRITE ,__arb_prio_bits);

/***************************************************************************
 **
 **  Static/SDRAM Memory controller (MC)
 **
 ***************************************************************************/
__IO_REG32_BIT(MCSDR_MODE,            0x18058100,__READ_WRITE ,__mcsdr_mode_bits);
__IO_REG32_BIT(MCSDR_ADDMAP,          0x18058104,__READ_WRITE ,__mcsdr_addmap_bits);
__IO_REG32_BIT(MCSDR_ADDCFG,          0x18058108,__READ_WRITE ,__mcsdr_addcfg_bits);
__IO_REG32_BIT(MCSDR_BASIC,           0x1805810C,__READ_WRITE ,__mcsdr_basic_bits);
__IO_REG32_BIT(MCSDR_T_REF,           0x18058110,__READ_WRITE ,__mcsdr_t_ref_bits);
__IO_REG32_BIT(MCSDR_T_RFC,           0x18058114,__READ_WRITE ,__mcsdr_t_rfc_bits);
__IO_REG32_BIT(MCSDR_T_MRD,           0x18058118,__READ_WRITE ,__mcsdr_t_mrd_bits);
__IO_REG32_BIT(MCSDR_T_RP,            0x18058120,__READ_WRITE ,__mcsdr_t_rp_bits);
__IO_REG32_BIT(MCSDR_T_RCD,           0x18058124,__READ_WRITE ,__mcsdr_t_rcd_bits);
__IO_REG32_BIT(MCST0_T_CEWD,          0x18058200,__READ_WRITE ,__mcst_t_cewd_bits);
__IO_REG32_BIT(MCST0_T_CE2WE,         0x18058204,__READ_WRITE ,__mcst_t_ce2we_bits);
__IO_REG32_BIT(MCST0_T_WEWD,          0x18058208,__READ_WRITE ,__mcst_t_wewd_bits);
__IO_REG32_BIT(MCST0_T_WE2CE,         0x1805820C,__READ_WRITE ,__mcst_t_we2ce_bits);
__IO_REG32_BIT(MCST0_T_CEWDR,         0x18058210,__READ_WRITE ,__mcst_t_cewdr_bits);
__IO_REG32_BIT(MCST0_T_CE2RD,         0x18058214,__READ_WRITE ,__mcst_t_ce2rd_bits);
__IO_REG32_BIT(MCST0_T_RDWD,          0x18058218,__READ_WRITE ,__mcst_t_rdwd_bits);
__IO_REG32_BIT(MCST0_T_RD2CE,         0x1805821C,__READ_WRITE ,__mcst_t_rd2ce_bits);
__IO_REG32_BIT(MCST0_BASIC,           0x18058220,__READ_WRITE ,__mcst_basic_bits);
__IO_REG32_BIT(MCST1_T_CEWD,          0x18058300,__READ_WRITE ,__mcst_t_cewd_bits);
__IO_REG32_BIT(MCST1_T_CE2WE,         0x18058304,__READ_WRITE ,__mcst_t_ce2we_bits);
__IO_REG32_BIT(MCST1_T_WEWD,          0x18058308,__READ_WRITE ,__mcst_t_wewd_bits);
__IO_REG32_BIT(MCST1_T_WE2CE,         0x1805830C,__READ_WRITE ,__mcst_t_we2ce_bits);
__IO_REG32_BIT(MCST1_T_CEWDR,         0x18058310,__READ_WRITE ,__mcst_t_cewdr_bits);
__IO_REG32_BIT(MCST1_T_CE2RD,         0x18058314,__READ_WRITE ,__mcst_t_ce2rd_bits);
__IO_REG32_BIT(MCST1_T_RDWD,          0x18058318,__READ_WRITE ,__mcst_t_rdwd_bits);
__IO_REG32_BIT(MCST1_T_RD2CE,         0x1805831C,__READ_WRITE ,__mcst_t_rd2ce_bits);
__IO_REG32_BIT(MCST1_BASIC,           0x18058320,__READ_WRITE ,__mcst_basic_bits);
__IO_REG32_BIT(MCST2_T_CEWD,          0x18058400,__READ_WRITE ,__mcst_t_cewd_bits);
__IO_REG32_BIT(MCST2_T_CE2WE,         0x18058404,__READ_WRITE ,__mcst_t_ce2we_bits);
__IO_REG32_BIT(MCST2_T_WEWD,          0x18058408,__READ_WRITE ,__mcst_t_wewd_bits);
__IO_REG32_BIT(MCST2_T_WE2CE,         0x1805840C,__READ_WRITE ,__mcst_t_we2ce_bits);
__IO_REG32_BIT(MCST2_T_CEWDR,         0x18058410,__READ_WRITE ,__mcst_t_cewdr_bits);
__IO_REG32_BIT(MCST2_T_CE2RD,         0x18058414,__READ_WRITE ,__mcst_t_ce2rd_bits);
__IO_REG32_BIT(MCST2_T_RDWD,          0x18058418,__READ_WRITE ,__mcst_t_rdwd_bits);
__IO_REG32_BIT(MCST2_T_RD2CE,         0x1805841C,__READ_WRITE ,__mcst_t_rd2ce_bits);
__IO_REG32_BIT(MCST2_BASIC,           0x18058420,__READ_WRITE ,__mcst_basic_bits);
__IO_REG32_BIT(MCST3_T_CEWD,          0x18058500,__READ_WRITE ,__mcst_t_cewd_bits);
__IO_REG32_BIT(MCST3_T_CE2WE,         0x18058504,__READ_WRITE ,__mcst_t_ce2we_bits);
__IO_REG32_BIT(MCST3_T_WEWD,          0x18058508,__READ_WRITE ,__mcst_t_wewd_bits);
__IO_REG32_BIT(MCST3_T_WE2CE,         0x1805850C,__READ_WRITE ,__mcst_t_we2ce_bits);
__IO_REG32_BIT(MCST3_T_CEWDR,         0x18058510,__READ_WRITE ,__mcst_t_cewdr_bits);
__IO_REG32_BIT(MCST3_T_CE2RD,         0x18058514,__READ_WRITE ,__mcst_t_ce2rd_bits);
__IO_REG32_BIT(MCST3_T_RDWD,          0x18058518,__READ_WRITE ,__mcst_t_rdwd_bits);
__IO_REG32_BIT(MCST3_T_RD2CE,         0x1805851C,__READ_WRITE ,__mcst_t_rd2ce_bits);
__IO_REG32_BIT(MCST3_BASIC,           0x18058520,__READ_WRITE ,__mcst_basic_bits);

/***************************************************************************
 **
 **  Interrupt Controller (IC)
 **
 ***************************************************************************/
__IO_REG32_BIT(INTC_SCR0,             0x18050000,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR1,             0x18050004,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR2,             0x18050008,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR3,             0x1805000C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR4,             0x18050010,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR5,             0x18050014,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR6,             0x18050018,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR7,             0x1805001C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR8,             0x18050020,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR9,             0x18050024,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR10,            0x18050028,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR11,            0x1805002C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR12,            0x18050030,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR13,            0x18050034,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR14,            0x18050038,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR15,            0x1805003C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR16,            0x18050040,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR17,            0x18050044,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR18,            0x18050048,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR19,            0x1805004C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR20,            0x18050050,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR21,            0x18050054,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR22,            0x18050058,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR23,            0x1805005C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR24,            0x18050060,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR25,            0x18050064,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR26,            0x18050068,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR27,            0x1805006C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR28,            0x18050070,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR29,            0x18050074,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR30,            0x18050078,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_SCR31,            0x1805007C,__READ_WRITE ,__intc_scr_bits);
__IO_REG32_BIT(INTC_ISR,              0x18050104,__READ       ,__intc_isr_bits);
__IO_REG32_BIT(INTC_IPR,              0x18050108,__READ       ,__intc_ipr_bits);
__IO_REG32_BIT(INTC_IMR,              0x1805010C,__READ_WRITE ,__intc_ipr_bits);
__IO_REG32_BIT(INTC_IECR,             0x18050114,__READ_WRITE ,__intc_ipr_bits);
__IO_REG32_BIT(INTC_ICCR,             0x18050118,__WRITE      ,__intc_ipr_bits);
__IO_REG32_BIT(INTC_ISCR,             0x1805011C,__WRITE      ,__intc_ipr_bits);
__IO_REG32_BIT(INTC_TEST,             0x18050124,__READ_WRITE ,__intc_test_bits);

/***************************************************************************
 **
 **  AHB DMA (HDMA)
 **
 ***************************************************************************/
__IO_REG32_BIT(HDMA_CON0,             0x18054000,__READ_WRITE ,__hdma_con_bits);
__IO_REG32_BIT(HDMA_CON1,             0x18054004,__READ_WRITE ,__hdma_con_bits);
__IO_REG32(    HDMA_ISRC0,            0x18054008,__READ_WRITE );
__IO_REG32(    HDMA_IDST0,            0x1805400C,__READ_WRITE );
__IO_REG32_BIT(HDMA_ICNT0,            0x18054010,__READ_WRITE ,__hdma_icnt_bits);
__IO_REG32(    HDMA_ISRC1,            0x18054014,__READ_WRITE );
__IO_REG32(    HDMA_IDST1,            0x18054018,__READ_WRITE );
__IO_REG32_BIT(HDMA_ICNT1,            0x1805401C,__READ_WRITE ,__hdma_icnt_bits);
__IO_REG32(    HDMA_CSRC0,            0x18054020,__READ       );
__IO_REG32(    HDMA_CDST0,            0x18054024,__READ       );
__IO_REG32_BIT(HDMA_CCNT0,            0x18054028,__READ       ,__hdma_icnt_bits);
__IO_REG32(    HDMA_CSRC1,            0x1805402C,__READ       );
__IO_REG32(    HDMA_CDST1,            0x18054030,__READ       );
__IO_REG32_BIT(HDMA_CCNT1,            0x18054034,__READ       ,__hdma_icnt_bits);
__IO_REG32_BIT(HDMA_ISR,              0x18054038,__READ_WRITE ,__hdma_isr_bits);
__IO_REG32_BIT(HDMA_DSR,              0x1805403C,__READ       ,__hdma_dsr_bits);
__IO_REG32_BIT(HDMA_ISCNT0,           0x18054040,__READ_WRITE ,__hdma_iscnt_bits);
__IO_REG32_BIT(HDMA_IPNCNTD0,         0x18054044,__READ_WRITE ,__hdma_ipncntd_bits);
__IO_REG32(    HDMA_IADDR_BS0,        0x18054048,__READ_WRITE );
__IO_REG32_BIT(HDMA_ISCNT1,           0x1805404C,__READ_WRITE ,__hdma_iscnt_bits);
__IO_REG32_BIT(HDMA_IPNCNTD1,         0x18054050,__READ_WRITE ,__hdma_ipncntd_bits);
__IO_REG32(    HDMA_IADDR_BS1,        0x18054054,__READ_WRITE );
__IO_REG32_BIT(HDMA_CSCNT0,           0x18054058,__READ       ,__hdma_iscnt_bits);
__IO_REG32_BIT(HDMA_CPNCNTD0,         0x1805405C,__READ       ,__hdma_ipncntd_bits);
__IO_REG32(    HDMA_CADDR_BS0,        0x18054060,__READ       );
__IO_REG32_BIT(HDMA_CSCNT1,           0x18054064,__READ       ,__hdma_iscnt_bits);
__IO_REG32_BIT(HDMA_CPNCNTD1,         0x18054068,__READ       ,__hdma_ipncntd_bits);
__IO_REG32(    HDMA_CADDR_BS1,        0x1805406C,__READ       );
__IO_REG32_BIT(HDMA_PACNT0,           0x18054070,__READ       ,__hdma_pacnt_bits);
__IO_REG32_BIT(HDMA_PACNT1,           0x18054074,__READ       ,__hdma_pacnt_bits);

/***************************************************************************
 **
 **  Ethernet 10/100 MAC
 **
 ***************************************************************************/
__IO_REG32_BIT(ENET_CSR0,             0x1805C000,__READ_WRITE ,__enet_csr0_bits);
__IO_REG32(    ENET_CSR1,             0x1805C008,__READ_WRITE );
__IO_REG32(    ENET_CSR2,             0x1805C010,__READ_WRITE );
__IO_REG32(    ENET_CSR3,             0x1805C018,__READ_WRITE );
__IO_REG32(    ENET_CSR4,             0x1805C020,__READ_WRITE );
__IO_REG32_BIT(ENET_CSR5,             0x1805C028,__READ_WRITE ,__enet_csr5_bits);
__IO_REG32_BIT(ENET_CSR6,             0x1805C030,__READ_WRITE ,__enet_csr6_bits);
__IO_REG32_BIT(ENET_CSR7,             0x1805C038,__READ_WRITE ,__enet_csr7_bits);
__IO_REG32_BIT(ENET_CSR8,             0x1805C040,__READ       ,__enet_csr8_bits);
__IO_REG32_BIT(ENET_CSR9,             0x1805C048,__READ_WRITE ,__enet_csr9_bits);
__IO_REG32_BIT(ENET_CSR11,            0x1805C058,__READ_WRITE ,__enet_csr11_bits);

/***************************************************************************
 **
 ** USB HOST OHCI CONTROLLERS
 **
 ***************************************************************************/
__IO_REG32_BIT(HCREVISION,            0x18064000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(HCCONTROL,             0x18064004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(HCCOMMANDSTATUS,       0x18064008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(HCINTERRUPTSTATUS,     0x1806400C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(HCINTERRUPTENABLE,     0x18064010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCINTERRUPTDISABLE,    0x18064014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCHCCA,                0x18064018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(HCPERIODCURRENTED,     0x1806401C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(HCCONTROLHEADED,       0x18064020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(HCCONTROLCURRENTED,    0x18064024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(HCBULKHEADED,          0x18064028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(HCBULKCURRENTED,       0x1806402C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(HCDONEHEAD,            0x18064030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(HCFMINTERVAL,          0x18064034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(HCFMREMAINING,         0x18064038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(HCFMNUMBER,            0x1806403C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(HCPERIODICSTART,       0x18064040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(HCLSTHRESHOLD,         0x18064044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(HCRHDESCRIPTORA,       0x18064048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(HCRHDESCRIPTORB,       0x1806404C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HCRHSTATUS,            0x18064050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS1,       0x18064054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS2,       0x18064058,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32(    HCRMID,                0x180640FC,__READ);

/***************************************************************************
 **
 ** USB HOST EHCI CONTROLLERS
 **
 ***************************************************************************/
__IO_REG8(     CAPLENGTH,             0x18064100,__READ       );
__IO_REG16(    HCIVERSION,            0x18064102,__READ       );
__IO_REG32_BIT(HCSPARAMS,             0x18064104,__READ       ,__hcsparams_bits);
__IO_REG32_BIT(HCCPARAMS,             0x18064108,__READ       ,__hccparams_bits);
__IO_REG32(    HCSPPORTROUTE,         0x1806410C,__READ       );
__IO_REG32_BIT(USBCMD,                0x18064120,__READ_WRITE ,__usbcmd_bits);
__IO_REG32_BIT(USBSTS,                0x18064124,__READ_WRITE ,__usbsts_bits);
__IO_REG32_BIT(USBINTR,               0x18064128,__READ_WRITE ,__usbintr_bits);
__IO_REG32_BIT(FRINDEX,               0x1806412C,__READ_WRITE ,__frindex_bits);
__IO_REG32(    CTRLDSSEGMENT,         0x18064130,__READ_WRITE );
__IO_REG32(    PERIODICLISTBASE,      0x18064134,__READ_WRITE );
__IO_REG32(    ASYNCLISTADDR,         0x18064138,__READ_WRITE );
__IO_REG32_BIT(CONFIGFLAG,            0x18064160,__READ_WRITE ,__configflag_bits);
__IO_REG32_BIT(PORTSC,                0x18064164,__READ_WRITE ,__portsc_bits);

/***************************************************************************
 **
 ** USB 2.0 Device Controller (UDC 2.0)
 **
 ***************************************************************************/
__IO_REG32_BIT(UDC_PHY_TEST_EN,       0x18060000,__READ_WRITE ,__udc_phy_test_en_bits);
__IO_REG32_BIT(UDC_PHY_TEST,          0x18060004,__READ_WRITE ,__udc_phy_test_bits);
__IO_REG32_BIT(UDC_DEV_CTL,           0x18060008,__READ_WRITE ,__udc_dev_ctl_bits);
__IO_REG32_BIT(UDC_DEV_INFO,          0x18060010,__READ       ,__udc_dev_info_bits);
__IO_REG32_BIT(UDC_EN_INT,            0x18060014,__READ_WRITE ,__udc_en_int_bits);
__IO_REG32_BIT(UDC_INT2FLAG,          0x18060018,__READ       ,__udc_int2flag_bits);
__IO_REG32_BIT(UDC_INTCON,            0x1806001C,__READ_WRITE ,__udc_intcon_bits);
__IO_REG32_BIT(UDC_SETUP1,            0x18060020,__READ       ,__udc_setup1_bits);
__IO_REG32_BIT(UDC_SETUP2,            0x18060024,__READ       ,__udc_setup2_bits);
__IO_REG32_BIT(UDC_AHBCON,            0x18060028,__READ_WRITE ,__udc_ahbcon_bits);
__IO_REG32_BIT(UDC_RX0STAT,           0x18060030,__READ       ,__udc_rx0stat_bits);
__IO_REG32_BIT(UDC_RX0CON,            0x18060034,__READ_WRITE ,__udc_rx0con_bits);
__IO_REG32_BIT(UDC_DMA0CTLO,          0x18060038,__READ_WRITE ,__udc_dma0ctlo_bits);
__IO_REG32(    UDC_DMA0LM_OADDR,      0x1806003C,__READ_WRITE );
__IO_REG32_BIT(UDC_TX0STAT,           0x18060040,__READ_WRITE ,__udc_tx0stat_bits);
__IO_REG32_BIT(UDC_TX0CON,            0x18060044,__READ_WRITE ,__udc_tx0con_bits);
__IO_REG32_BIT(UDC_TX0BUF,            0x18060048,__READ       ,__udc_tx0buf_bits);
__IO_REG32_BIT(UDC_DMA0CTLI,          0x1806004C,__READ_WRITE ,__udc_dma0ctli_bits);
__IO_REG32(    UDC_DMA0LM_IADDR,      0x18060050,__READ_WRITE );
__IO_REG32_BIT(UDC_RX1STAT,           0x18060054,__READ       ,__udc_rxstat_bits);
__IO_REG32_BIT(UDC_RX1CON,            0x18060058,__READ_WRITE ,__udc_rxcon_bits);
__IO_REG32_BIT(UDC_DMA1CTLO,          0x1806005C,__READ_WRITE ,__udc_dmactlo_bits);
__IO_REG32(    UDC_DMA1LM_OADDR,      0x18060060,__READ_WRITE );
__IO_REG32_BIT(UDC_TX2STAT,           0x18060064,__READ_WRITE ,__udc_txstat_bits);
__IO_REG32_BIT(UDC_TX2CON,            0x18060068,__READ_WRITE ,__udc_txcon_bits);
__IO_REG32_BIT(UDC_TX2BUF,            0x1806006C,__READ       ,__udc_txbuf_bits);
__IO_REG32_BIT(UDC_DMA2CTLI,          0x18060070,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA2LM_IADDR,      0x18060074,__READ_WRITE );
__IO_REG32_BIT(UDC_TX3STAT,           0x18060078,__READ_WRITE ,__udc_tx3stat_bits);
__IO_REG32_BIT(UDC_TX3CON,            0x1806007C,__READ_WRITE ,__udc_tx3con_bits);
__IO_REG32_BIT(UDC_TX3BUF,            0x18060080,__READ       ,__udc_tx3buf_bits);
__IO_REG32_BIT(UDC_DMA3CTLI,          0x18060084,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA3LM_IADDR,      0x18060088,__READ_WRITE );
__IO_REG32_BIT(UDC_RX4STAT,           0x1806008C,__READ       ,__udc_rxstat_bits);
__IO_REG32_BIT(UDC_RX4CON,            0x18060090,__READ_WRITE ,__udc_rxcon_bits);
__IO_REG32_BIT(UDC_DMA4CTLO,          0x18060094,__READ_WRITE ,__udc_dmactlo_bits);
__IO_REG32(    UDC_DMA4LM_OADDR,      0x18060098,__READ_WRITE );
__IO_REG32_BIT(UDC_TX5STAT,           0x1806009C,__READ_WRITE ,__udc_txstat_bits);
__IO_REG32_BIT(UDC_TX5CON,            0x180600A0,__READ_WRITE ,__udc_txcon_bits);
__IO_REG32_BIT(UDC_TX5BUF,            0x180600A4,__READ       ,__udc_txbuf_bits);
__IO_REG32_BIT(UDC_DMA5CTLI,          0x180600A8,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA5LM_IADDR,      0x180600AC,__READ_WRITE );
__IO_REG32_BIT(UDC_TX6STAT,           0x180600B0,__READ_WRITE ,__udc_tx3stat_bits);
__IO_REG32_BIT(UDC_TX6CON,            0x180600B4,__READ_WRITE ,__udc_tx3con_bits);
__IO_REG32_BIT(UDC_TX6BUF,            0x180600B8,__READ       ,__udc_tx3buf_bits);
__IO_REG32_BIT(UDC_DMA6CTLI,          0x180600BC,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA6LM_IADDR,      0x180600C0,__READ_WRITE );
__IO_REG32_BIT(UDC_RX7STAT,           0x180600C4,__READ       ,__udc_rxstat_bits);
__IO_REG32_BIT(UDC_RX7CON,            0x180600C8,__READ_WRITE ,__udc_rxcon_bits);
__IO_REG32_BIT(UDC_DMA7CTLO,          0x180600CC,__READ_WRITE ,__udc_dmactlo_bits);
__IO_REG32(    UDC_DMA7LM_OADDR,      0x180600D0,__READ_WRITE );
__IO_REG32_BIT(UDC_TX8STAT,           0x180600D4,__READ_WRITE ,__udc_txstat_bits);
__IO_REG32_BIT(UDC_TX8CON,            0x180600D8,__READ_WRITE ,__udc_txcon_bits);
__IO_REG32_BIT(UDC_TX8BUF,            0x180600DC,__READ       ,__udc_txbuf_bits);
__IO_REG32_BIT(UDC_DMA8CTLI,          0x180600E0,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA8LM_IADDR,      0x180600E4,__READ_WRITE );
__IO_REG32_BIT(UDC_TX9STAT,           0x180600E8,__READ_WRITE ,__udc_tx3stat_bits);
__IO_REG32_BIT(UDC_TX9CON,            0x180600EC,__READ_WRITE ,__udc_tx3con_bits);
__IO_REG32_BIT(UDC_TX9BUF,            0x180600F0,__READ       ,__udc_tx3buf_bits);
__IO_REG32_BIT(UDC_DMA9CTLI,          0x180600F4,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA9LM_IADDR,      0x180600F8,__READ_WRITE );
__IO_REG32_BIT(UDC_RX10STAT,          0x180600FC,__READ       ,__udc_rxstat_bits);
__IO_REG32_BIT(UDC_RX10CON,           0x18060100,__READ_WRITE ,__udc_rxcon_bits);
__IO_REG32_BIT(UDC_DMA10CTLO,         0x18060104,__READ_WRITE ,__udc_dmactlo_bits);
__IO_REG32(    UDC_DMA10LM_OADDR,     0x18060108,__READ_WRITE );
__IO_REG32_BIT(UDC_TX11STAT,          0x1806010C,__READ_WRITE ,__udc_txstat_bits);
__IO_REG32_BIT(UDC_TX11CON,           0x18060110,__READ_WRITE ,__udc_txcon_bits);
__IO_REG32_BIT(UDC_TX11BUF,           0x18060114,__READ       ,__udc_txbuf_bits);
__IO_REG32_BIT(UDC_DMA11CTLI,         0x18060118,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA11LM_IADDR,     0x1806011C,__READ_WRITE );
__IO_REG32_BIT(UDC_TX12STAT,          0x18060120,__READ_WRITE ,__udc_tx3stat_bits);
__IO_REG32_BIT(UDC_TX12CON,           0x18060124,__READ_WRITE ,__udc_tx3con_bits);
__IO_REG32_BIT(UDC_TX12BUF,           0x18060128,__READ       ,__udc_tx3buf_bits);
__IO_REG32_BIT(UDC_DMA12CTLI,         0x1806012C,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA12LM_IADDR,     0x18060130,__READ_WRITE );
__IO_REG32_BIT(UDC_RX13STAT,          0x18060134,__READ       ,__udc_rxstat_bits);
__IO_REG32_BIT(UDC_RX13CON,           0x18060138,__READ_WRITE ,__udc_rxcon_bits);
__IO_REG32_BIT(UDC_DMA13CTLO,         0x1806013C,__READ_WRITE ,__udc_dmactlo_bits);
__IO_REG32(    UDC_DMA13LM_OADDR,     0x18060140,__READ_WRITE );
__IO_REG32_BIT(UDC_TX14STAT,          0x18060144,__READ_WRITE ,__udc_txstat_bits);
__IO_REG32_BIT(UDC_TX14CON,           0x18060148,__READ_WRITE ,__udc_txcon_bits);
__IO_REG32_BIT(UDC_TX14BUF,           0x1806014C,__READ       ,__udc_txbuf_bits);
__IO_REG32_BIT(UDC_DMA14CTLI,         0x18060150,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA14LM_IADDR,     0x18060154,__READ_WRITE );
__IO_REG32_BIT(UDC_TX15STAT,          0x18060158,__READ_WRITE ,__udc_tx3stat_bits);
__IO_REG32_BIT(UDC_TX15CON,           0x1806015C,__READ_WRITE ,__udc_tx3con_bits);
__IO_REG32_BIT(UDC_TX15BUF,           0x18060160,__READ       ,__udc_tx3buf_bits);
__IO_REG32_BIT(UDC_DMA15CTLI,         0x18060164,__READ_WRITE ,__udc_dmactli_bits);
__IO_REG32(    UDC_DMA15LM_IADDR,     0x18060168,__READ_WRITE );

/***************************************************************************
 **
 ** LCD Controller (LCDC)
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_CTRL0,             0x18040000,__READ_WRITE ,__lcd_ctrl0_bits);
__IO_REG32_BIT(LCD_HT,                0x18040010,__READ_WRITE ,__lcd_ht_bits);
__IO_REG32_BIT(LCD_VT,                0x18040014,__READ_WRITE ,__lcd_vt_bits);
__IO_REG32(    LCD_LUT_ADDR,          0x1804001C,__READ_WRITE );
__IO_REG32(    LCD_UPAGE0_ADDR,       0x18040020,__READ_WRITE );
__IO_REG32(    LCD_UPAGE1_ADDR,       0x18040024,__READ_WRITE );
__IO_REG32(    LCD_LPAGE0_ADDR,       0x18040028,__READ_WRITE );
__IO_REG32(    LCD_LPAGE1_ADDR,       0x1804002C,__READ_WRITE );
__IO_REG32_BIT(LCD_INTR_STA,          0x18040040,__READ       ,__lcd_intr_sta_bits);
__IO_REG32_BIT(LCD_INTR_EN,           0x18040044,__WRITE      ,__lcd_intr_sta_bits);
__IO_REG32_BIT(LCD_INTR_DIS,          0x18040048,__WRITE      ,__lcd_intr_sta_bits);
__IO_REG32_BIT(LCD_INTR_MASK,         0x1804004C,__READ_WRITE ,__lcd_intr_sta_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG32(    UART0_RBR,             0x1802C000,__READ_WRITE );
#define UART0_THR     UART0_RBR
#define UART0_DLL     UART0_RBR
__IO_REG32_BIT(UART0_IER,             0x1802C004,__READ_WRITE ,__uart_ier_bits);
#define UART0_DLH     UART0_IER
__IO_REG32_BIT(UART0_IIR,             0x1802C008,__READ_WRITE ,__uart_iir_bits);
#define UART0_FCR     UART0_IIR
#define UART0_FCR_bit UART0_IIR_bit
__IO_REG32_BIT(UART0_LCR,             0x1802C00C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART0_MCR,             0x1802C010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART0_LSR,             0x1802C014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART0_MSR,             0x1802C018,__READ       ,__uart_msr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG32(    UART1_RBR,             0x18030000,__READ_WRITE );
#define UART1_THR     UART1_RBR
#define UART1_DLL     UART1_RBR
__IO_REG32_BIT(UART1_IER,             0x18030004,__READ_WRITE ,__uart_ier_bits);
#define UART1_DLH     UART1_IER
__IO_REG32_BIT(UART1_IIR,             0x18030008,__READ_WRITE ,__uart_iir_bits);
#define UART1_FCR     UART1_IIR
#define UART1_FCR_bit UART1_IIR_bit
__IO_REG32_BIT(UART1_LCR,             0x1803000C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART1_MCR,             0x18030010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART1_LSR,             0x18030014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART1_MSR,             0x18030018,__READ       ,__uart_msr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG32(    UART2_RBR,             0x18034000,__READ_WRITE );
#define UART2_THR     UART2_RBR
#define UART2_DLL     UART2_RBR
__IO_REG32_BIT(UART2_IER,             0x18034004,__READ_WRITE ,__uart_ier_bits);
#define UART2_DLH     UART2_IER
__IO_REG32_BIT(UART2_IIR,             0x18034008,__READ_WRITE ,__uart_iir_bits);
#define UART2_FCR     UART2_IIR
#define UART2_FCR_bit UART2_IIR_bit
__IO_REG32_BIT(UART2_LCR,             0x1803400C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART2_MCR,             0x18034010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART2_LSR,             0x18034014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART2_MSR,             0x18034018,__READ       ,__uart_msr_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG32(    UART3_RBR,             0x18038000,__READ_WRITE );
#define UART3_THR     UART3_RBR
#define UART3_DLL     UART3_RBR
__IO_REG32_BIT(UART3_IER,             0x18038004,__READ_WRITE ,__uart_ier_bits);
#define UART3_DLH     UART3_IER
__IO_REG32_BIT(UART3_IIR,             0x18038008,__READ_WRITE ,__uart_iir_bits);
#define UART3_FCR     UART3_IIR
#define UART3_FCR_bit UART3_IIR_bit
__IO_REG32_BIT(UART3_LCR,             0x1803800C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART3_MCR,             0x18038010,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART3_LSR,             0x18038014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART3_MSR,             0x18038018,__READ       ,__uart_msr_bits);

/***************************************************************************
 **
 ** GPIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO0_PADR,            0x1800C000,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_PACON,           0x1800C004,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_PBDR,            0x1800C008,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_PBCON,           0x1800C00C,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_PCDR,            0x1800C010,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_PCCON,           0x1800C014,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_PDDR,            0x1800C018,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_PDCON,           0x1800C01C,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_TEST,            0x1800C020,__READ_WRITE ,__gpio_test_bits);
__IO_REG32_BIT(GPIO0_IEA,             0x1800C024,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_IEB,             0x1800C028,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_IEC,             0x1800C02C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_IED,             0x1800C030,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_ISA,             0x1800C034,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_ISB,             0x1800C038,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_ISC,             0x1800C03C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_ISD,             0x1800C040,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_IBEA,            0x1800C044,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_IBEB,            0x1800C048,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_IBEC,            0x1800C04C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_IBED,            0x1800C050,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_IVEA,            0x1800C054,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_IVEB,            0x1800C058,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_IVEC,            0x1800C05C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_IVED,            0x1800C060,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_ICA,             0x1800C064,__WRITE      ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO0_ICB,             0x1800C068,__WRITE      ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO0_ICC,             0x1800C06C,__WRITE      ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO0_ICD,             0x1800C070,__WRITE      ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO0_ISR,             0x1800C074,__READ       ,__gpio_isr_bits);

/***************************************************************************
 **
 ** GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO1_PADR,            0x18010000,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_PACON,           0x18010004,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_PBDR,            0x18010008,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_PBCON,           0x1801000C,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_PCDR,            0x18010010,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_PCCON,           0x18010014,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_PDDR,            0x18010018,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_PDCON,           0x1801001C,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_TEST,            0x18010020,__READ_WRITE ,__gpio_test_bits);
__IO_REG32_BIT(GPIO1_IEA,             0x18010024,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_IEB,             0x18010028,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_IEC,             0x1801002C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_IED,             0x18010030,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_ISA,             0x18010034,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_ISB,             0x18010038,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_ISC,             0x1801003C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_ISD,             0x18010040,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_IBEA,            0x18010044,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_IBEB,            0x18010048,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_IBEC,            0x1801004C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_IBED,            0x18010050,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_IVEA,            0x18010054,__READ_WRITE ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_IVEB,            0x18010058,__READ_WRITE ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_IVEC,            0x1801005C,__READ_WRITE ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_IVED,            0x18010060,__READ_WRITE ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_ICA,             0x18010064,__WRITE      ,__gpio_padr_bits);
__IO_REG32_BIT(GPIO1_ICB,             0x18010068,__WRITE      ,__gpio_pbdr_bits);
__IO_REG32_BIT(GPIO1_ICC,             0x1801006C,__WRITE      ,__gpio_pcdr_bits);
__IO_REG32_BIT(GPIO1_ICD,             0x18010070,__WRITE      ,__gpio_pddr_bits);
__IO_REG32_BIT(GPIO1_ISR,             0x18010074,__READ       ,__gpio_isr_bits);

/***************************************************************************
 **
 ** Timers
 **
 ***************************************************************************/
__IO_REG32(    TMR0LR,                0x18000000,__WRITE      );
__IO_REG32(    TMR0CVR,               0x18000004,__READ       );
__IO_REG32_BIT(TMR0CON,               0x18000008,__READ_WRITE ,__tmrcon_bits);
__IO_REG32(    TMR1LR,                0x18000010,__WRITE      );
__IO_REG32(    TMR1CVR,               0x18000014,__READ       );
__IO_REG32_BIT(TMR1CON,               0x18000018,__READ_WRITE ,__tmrcon_bits);
__IO_REG32(    TMR2LR,                0x18000020,__WRITE      );
__IO_REG32(    TMR2CVR,               0x18000024,__READ       );
__IO_REG32_BIT(TMR2CON,               0x18000028,__READ_WRITE ,__tmrcon_bits);
__IO_REG32_BIT(TMRMODE,               0x18000030,__READ_WRITE ,__tmrmode_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32(    WDTLR,                 0x18008000,__WRITE      );
__IO_REG32(    WDTCVR,                0x18008004,__READ       );
__IO_REG32_BIT(WDTCON,                0x18008008,__READ_WRITE ,__wdtcon_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_TIME,              0x18004000,__READ_WRITE ,__rtc_time_bits);
__IO_REG32_BIT(RTC_DATE,              0x18004004,__READ_WRITE ,__rtc_date_bits);
__IO_REG32_BIT(RTC_TALRM,             0x18004008,__READ_WRITE ,__rtc_talrm_bits);
__IO_REG32_BIT(RTC_DALRM,             0x1800400C,__READ_WRITE ,__rtc_dalrm_bits);
__IO_REG32_BIT(RTC_CTRL,              0x18004010,__READ_WRITE ,__rtc_ctrl_bits);
__IO_REG32_BIT(RTC_RESET,             0x18004014,__WRITE      ,__rtc_reset_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI_TxR,               0x18014000,__READ_WRITE ,__spi_txr_bits);
#define SPI_RxR       SPI_TxR
#define SPI_RxR_bit   SPI_TxR_bit
__IO_REG32_BIT(SPI_IER,               0x18014004,__READ_WRITE ,__spi_ier_bits);
__IO_REG32_BIT(SPI_FCR,               0x18014008,__READ_WRITE ,__spi_fcr_bits);
__IO_REG32_BIT(SPI_FWCR,              0x1801400C,__READ_WRITE ,__spi_fwcr_bits);
__IO_REG32_BIT(SPI_DLYCR,             0x18014010,__READ_WRITE ,__spi_dlycr_bits);
__IO_REG32_BIT(SPI_TxCR,              0x18014014,__READ_WRITE ,__spi_txcr_bits);
__IO_REG32_BIT(SPI_RxCR,              0x18014018,__READ_WRITE ,__spi_rxcr_bits);
__IO_REG32_BIT(SPI_SSCR,              0x1801401C,__READ_WRITE ,__spi_sscr_bits);
__IO_REG32_BIT(SPI_ISR,               0x18014020,__READ       ,__spi_isr_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_MTXR,              0x18018000,__READ_WRITE ,__i2c_mtxr_bits);
__IO_REG32_BIT(I2C_MRXR,              0x18018004,__READ       ,__i2c_mtxr_bits);
__IO_REG32_BIT(I2C_STXR,              0x18018008,__READ_WRITE ,__i2c_mtxr_bits);
__IO_REG32_BIT(I2C_SRXR,              0x1801800C,__READ       ,__i2c_mtxr_bits);
__IO_REG32_BIT(I2C_SADDR,             0x18018010,__READ_WRITE ,__i2c_saddr_bits);
__IO_REG32_BIT(I2C_IER,               0x18018014,__READ_WRITE ,__i2c_ier_bits);
__IO_REG32_BIT(I2C_ISR,               0x18018018,__READ_WRITE ,__i2c_ier_bits);
__IO_REG32_BIT(I2C_LCMR,              0x1801801C,__READ_WRITE ,__i2c_lcmr_bits);
__IO_REG32_BIT(I2C_LSR,               0x18018020,__READ_WRITE ,__i2c_lsr_bits);
__IO_REG32_BIT(I2C_CONR,              0x18018024,__READ_WRITE ,__i2c_conr_bits);
__IO_REG32_BIT(I2C_OPR,               0x18018028,__READ_WRITE ,__i2c_opr_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_OPR,               0x1801C000,__READ_WRITE ,__i2s_opr_bits);
__IO_REG32(    I2S_TXR,               0x1801C004,__WRITE      );
__IO_REG32(    I2S_RXR,               0x1801C008,__READ       );
__IO_REG32_BIT(I2S_TXCTL,             0x1801C00C,__READ_WRITE ,__i2s_txctl_bits);
__IO_REG32_BIT(I2S_RXCTL,             0x1801C010,__READ_WRITE ,__i2s_rxctl_bits);
__IO_REG32_BIT(I2S_FIFOSTS,           0x1801C014,__READ_WRITE ,__i2s_fifosts_bits);
__IO_REG32_BIT(I2S_IER,               0x1801C018,__READ_WRITE ,__i2s_ier_bits);
__IO_REG32_BIT(I2S_ISR,               0x1801C020,__READ       ,__i2s_ier_bits);

/***************************************************************************
 **
 ** SD/MMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MMU_CTRL,              0x18020000,__READ_WRITE ,__mmu_ctrl_bits);
__IO_REG32_BIT(MMU_PNRI,              0x18020004,__READ_WRITE ,__mmu_pnri_bits);
__IO_REG32_BIT(CUR_PNRI,              0x18020008,__READ       ,__mmu_pnri_bits);
__IO_REG32_BIT(MMU_PNRII,             0x1802000C,__READ_WRITE ,__mmu_pnrii_bits);
__IO_REG32_BIT(CUR_PNRII,             0x18020010,__READ       ,__mmu_pnrii_bits);
__IO_REG32_BIT(MMU_ADDR,              0x18020014,__READ_WRITE ,__mmu_addr_bits);
__IO_REG32_BIT(CUR_ADDR,              0x18020018,__READ       ,__mmu_addr_bits);
__IO_REG32(    MMU_DATA,              0x1802001C,__READ_WRITE );
__IO_REG32_BIT(SD_CTRL,               0x18020020,__READ_WRITE ,__sd_ctrl_bits);
__IO_REG32_BIT(SD_INT,                0x18020024,__READ_WRITE ,__sd_int_bits);
__IO_REG32_BIT(SD_CARD,               0x18020028,__READ_WRITE ,__sd_card_bits);
__IO_REG32_BIT(SD_CMDREST,            0x18020030,__READ_WRITE ,__sd_cmdrest_bits);
__IO_REG32_BIT(SD_CMDRES,             0x18020034,__READ       ,__sd_cmdres_bits);
__IO_REG32_BIT(SD_DATAT,              0x1802003C,__READ       ,__sd_datat_bits);
__IO_REG32(    SD_CMD,                0x18020040,__READ_WRITE );
__IO_REG32(    SD_RES3,               0x18020044,__READ       );
__IO_REG32(    SD_RES2,               0x18020048,__READ       );
__IO_REG32(    SD_RES1,               0x1802004C,__READ       );
__IO_REG32(    SD_RES0,               0x18020050,__READ       );

/***************************************************************************
 **
 ** PWMT
 **
 ***************************************************************************/
__IO_REG32(    PWMT0_CNTR,            0x18024000,__READ_WRITE );
__IO_REG32(    PWMT0_HRC,             0x18024004,__READ_WRITE );
__IO_REG32(    PWMT0_LRC,             0x18024008,__READ_WRITE );
__IO_REG32_BIT(PWMT0_CTRL,            0x1802400C,__READ_WRITE ,__pwmt_ctrl_bits);
__IO_REG32(    PWMT1_CNTR,            0x18024010,__READ_WRITE );
__IO_REG32(    PWMT1_HRC,             0x18024014,__READ_WRITE );
__IO_REG32(    PWMT1_LRC,             0x18024018,__READ_WRITE );
__IO_REG32_BIT(PWMT1_CTRL,            0x1802401C,__READ_WRITE ,__pwmt_ctrl_bits);
__IO_REG32(    PWMT2_CNTR,            0x18024020,__READ_WRITE );
__IO_REG32(    PWMT2_HRC,             0x18024024,__READ_WRITE );
__IO_REG32(    PWMT2_LRC,             0x18024028,__READ_WRITE );
__IO_REG32_BIT(PWMT2_CTRL,            0x1802402C,__READ_WRITE ,__pwmt_ctrl_bits);
__IO_REG32(    PWMT3_CNTR,            0x18024030,__READ_WRITE );
__IO_REG32(    PWMT3_HRC,             0x18024034,__READ_WRITE );
__IO_REG32(    PWMT3_LRC,             0x18024038,__READ_WRITE );
__IO_REG32_BIT(PWMT3_CTRL,            0x1802403C,__READ_WRITE ,__pwmt_ctrl_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_DATA,              0x18028000,__READ_WRITE ,__adc_data_bits);
__IO_REG32_BIT(ADC_STAS,              0x18028004,__READ_WRITE ,__adc_stas_bits);
__IO_REG32_BIT(ADC_CTRL,              0x18028008,__READ_WRITE ,__adc_ctrl_bits);

/***************************************************************************
 **
 ** NAND
 **
 ***************************************************************************/
__IO_REG32(    FLSH_BUF0_BASE_ADDR,   0x18070000,__READ_WRITE );
__IO_REG32(    FLSH_BUF1_BASE_ADDR,   0x18078000,__READ_WRITE );
__IO_REG32_BIT(FLSH_CONF,             0x1807C000,__READ_WRITE ,__flsh_conf_bits);
__IO_REG32_BIT(FLSH_COMM_1,           0x1807C004,__READ_WRITE ,__flsh_comm_bits);
__IO_REG32_BIT(FLSH_COMM_2,           0x1807C008,__READ_WRITE ,__flsh_comm_bits);
__IO_REG32_BIT(FLSH_STATE_COMM,       0x1807C00C,__READ_WRITE ,__flsh_comm_bits);
__IO_REG32_BIT(FLSH_ADDRESS_1,        0x1807C010,__READ_WRITE ,__flsh_address_bits);
__IO_REG32_BIT(FLSH_ADDRESS_2,        0x1807C014,__READ_WRITE ,__flsh_address_bits);
__IO_REG32_BIT(FLSH_ADDRESS_3,        0x1807C018,__READ_WRITE ,__flsh_address_bits);
__IO_REG32_BIT(FLSH_ADDRESS_4,        0x1807C01C,__READ_WRITE ,__flsh_address_bits);
__IO_REG32_BIT(FLSH_ADDRESS_5,        0x1807C020,__READ_WRITE ,__flsh_address_bits);
__IO_REG32(    FLSH_DATA,             0x1807C024,__READ_WRITE );
__IO_REG32_BIT(FLSH_BUFF_STADDR,      0x1807C028,__READ_WRITE ,__flsh_buff_staddr_bits);
__IO_REG32_BIT(FLSH_BUFF_CNT,         0x1807C02C,__WRITE      ,__flsh_buff_cnt_bits);
__IO_REG32_BIT(FLSH_BUFF_STATE,       0x1807C030,__READ_WRITE ,__flsh_buff_state_bits);
__IO_REG32_BIT(FLSH_DMA_SET,          0x1807C034,__READ_WRITE ,__flsh_dma_set_bits);
__IO_REG32_BIT(FLSH_CE_WP,            0x1807C038,__READ_WRITE ,__flsh_ce_wp_bits);
__IO_REG32_BIT(FLSH_CONTROL,          0x1807C03C,__READ_WRITE ,__flsh_control_bits);
__IO_REG32_BIT(FLSH_RESET,            0x1807C040,__WRITE      ,__flsh_reset_bits);
__IO_REG32_BIT(FLSH_STATE,            0x1807C044,__READ       ,__flsh_state_bits);
__IO_REG32_BIT(FLSH_INT_MASK,         0x1807C048,__WRITE      ,__flsh_state_bits);
__IO_REG32_BIT(FLSH_INT_STATE,        0x1807C04C,__READ_WRITE ,__flsh_state_bits);
__IO_REG32_BIT(FLSH_GPIO,             0x1807C050,__READ_WRITE ,__flsh_gpio_bits);
__IO_REG32_BIT(FLSH_S_NUM,            0x1807C054,__READ_WRITE ,__flsh_s_num_bits);
__IO_REG32(    FLSH_1_ECC_1,          0x1807C058,__READ       );
__IO_REG32(    FLSH_1_ECC_2,          0x1807C05C,__READ       );
__IO_REG32(    FLSH_1_ECC_3,          0x1807C060,__READ       );
__IO_REG32(    FLSH_1_ECC_4,          0x1807C064,__READ       );
__IO_REG32(    FLSH_2_ECC_1,          0x1807C068,__READ       );
__IO_REG32(    FLSH_2_ECC_2,          0x1807C06C,__READ       );
__IO_REG32(    FLSH_2_ECC_3,          0x1807C070,__READ       );
__IO_REG32(    FLSH_2_ECC_4,          0x1807C074,__READ       );
__IO_REG32(    FLSH_3_ECC_1,          0x1807C078,__READ       );
__IO_REG32(    FLSH_3_ECC_2,          0x1807C07C,__READ       );
__IO_REG32(    FLSH_3_ECC_3,          0x1807C080,__READ       );
__IO_REG32(    FLSH_3_ECC_4,          0x1807C084,__READ       );
__IO_REG32(    FLSH_4_ECC_1,          0x1807C088,__READ       );
__IO_REG32(    FLSH_4_ECC_2,          0x1807C08C,__READ       );
__IO_REG32(    FLSH_4_ECC_3,          0x1807C090,__READ       );
__IO_REG32(    FLSH_4_ECC_4,          0x1807C094,__READ       );
__IO_REG32_BIT(FLSH_1_SYNDR_1,        0x1807C098,__READ       ,__flsh_syndr_1_bits);
__IO_REG32_BIT(FLSH_1_SYNDR_2,        0x1807C09C,__READ       ,__flsh_syndr_2_bits);
__IO_REG32_BIT(FLSH_1_SYNDR_3,        0x1807C0A0,__READ       ,__flsh_syndr_3_bits);
__IO_REG32_BIT(FLSH_1_SYNDR_4,        0x1807C0A4,__READ       ,__flsh_syndr_4_bits);
__IO_REG32_BIT(FLSH_2_SYNDR_1,        0x1807C0A8,__READ       ,__flsh_syndr_1_bits);
__IO_REG32_BIT(FLSH_2_SYNDR_2,        0x1807C0AC,__READ       ,__flsh_syndr_2_bits);
__IO_REG32_BIT(FLSH_2_SYNDR_3,        0x1807C0B0,__READ       ,__flsh_syndr_3_bits);
__IO_REG32_BIT(FLSH_2_SYNDR_4,        0x1807C0B4,__READ       ,__flsh_syndr_4_bits);
__IO_REG32_BIT(FLSH_3_SYNDR_1,        0x1807C0B8,__READ       ,__flsh_syndr_1_bits);
__IO_REG32_BIT(FLSH_3_SYNDR_2,        0x1807C0BC,__READ       ,__flsh_syndr_2_bits);
__IO_REG32_BIT(FLSH_3_SYNDR_3,        0x1807C0C0,__READ       ,__flsh_syndr_3_bits);
__IO_REG32_BIT(FLSH_3_SYNDR_4,        0x1807C0C4,__READ       ,__flsh_syndr_4_bits);
__IO_REG32_BIT(FLSH_4_SYNDR_1,        0x1807C0C8,__READ       ,__flsh_syndr_1_bits);
__IO_REG32_BIT(FLSH_4_SYNDR_2,        0x1807C0CC,__READ       ,__flsh_syndr_2_bits);
__IO_REG32_BIT(FLSH_4_SYNDR_3,        0x1807C0D0,__READ       ,__flsh_syndr_3_bits);
__IO_REG32_BIT(FLSH_4_SYNDR_4,        0x1807C0D4,__READ       ,__flsh_syndr_4_bits);


/* Assembler-specific declarations  ****************************************/
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
 **  PC7130 IRQ Interrupt Vector Table
 **
 ***************************************************************************/
#define SW_INT            0
#define TIMER0_INT        1
#define TIMER1_INT        2
#define TIMER2_INT        3
#define SYSTIME_NS_INT    4
#define SYSTIME_S_INT     5
#define GPIO15_INT        6
#define WATCHDOG_INT      7
#define UART0_INT         8
#define UART1_INT         9
#define UART2_INT        10
#define USB_INT          11
#define SPI_INT          12
#define I2C_INT          13
#define HIF_INT          15
#define GPIO_INT         16
#define COM0_INT         17
#define COM1_INT         18
#define COM2_INT         19
#define COM3_INT         20
#define MSYNC0_INT       21
#define MSYNC1_INT       22
#define MSYNC2_INT       23
#define MSYNC3_INT       24
#define INT_PHY_INT      25
#define ISO_AREA_INT     26
#define TIMER3_INT       29
#define TIMER4_INT       30

#endif    /* __PC7130_H */
