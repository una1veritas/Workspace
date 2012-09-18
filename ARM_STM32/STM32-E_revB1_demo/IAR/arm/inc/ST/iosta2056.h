/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST STA2056
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 30259 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOSTA2056_H
#define __IOSTA2056_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x58 = 88 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   STA2056 SPECIAL FUNCTION REGISTERS
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

/*Supplement to io_macros.h */
typedef struct
{
  __REG8 no8:1;
  __REG8 no9:1;
  __REG8 no10:1;
  __REG8 no11:1;
  __REG8 no12:1;
  __REG8 no13:1;
  __REG8 no14:1;
  __REG8 no15:1;
} __BITS8H_bits;

/* BCONx Bank n Configuration Register */
typedef struct
{
  __REG16  B_SIZE         : 2;
  __REG16  C_LENGTH       : 4;
  __REG16                 : 9;
  __REG16  BE             : 1;
} __bankcfg_bits;

/* APB Bridge Status Register (BSR) */
typedef struct {
  __REG32 ABORT           : 1;
  __REG32                 : 3;
  __REG32 OUTM            : 1;
  __REG32 APBT            : 1;
  __REG32                 :26;
} __apb_bsr_bits;

/* APB Time-out Register (TOR) */
typedef struct {
  __REG32 TOUT_CNT        : 5;
  __REG32                 : 3;
  __REG32 ABTEN           : 1;
  __REG32                 :23;
} __apb_tor_bits;

/* APB Out of Memory Register (OMR) */
typedef struct {
  __REG32                 : 7;
  __REG32 NRW             : 1;
  __REG32 PA              :16;
  __REG32                 : 8;
} __apb_omr_bits;

/* APB Time-out Error Register (TOER) */
typedef struct {
  __REG32                 : 7;
  __REG32 NRW             : 1;
  __REG32 PA              :16;
  __REG32                 : 8;
} __apb_toer_bits;

/* APB1 Peripheral bits */
typedef struct {
  __REG32                 : 1;
  __REG32 I2C             : 1;
  __REG32                 : 1;
  __REG32 UART0           : 1;
  __REG32 UART1_SC        : 1;
  __REG32                 : 3;
  __REG32 CAN             : 1;
  __REG32                 : 1;
  __REG32 SPI             : 1;
  __REG32                 :21;
} __apb1_peri_bits;

/* APB2 Peripheral bits */
typedef struct {
  __REG32 WUIMU           :1;
  __REG32                 :1;
  __REG32 IOPORT0         :1;
  __REG32 IOPORT1         :1;
  __REG32 IOPORT2         :1;
  __REG32                 :1;
  __REG32 ADC             :1;
  __REG32 CKOUT           :1;
  __REG32 TIMER0          :1;
  __REG32 TIMER1          :1;
  __REG32 TIMER2          :1;
  __REG32 TIMER3          :1;
  __REG32 RTC             :1;
  __REG32 WDT             :1;
  __REG32 EIC             :1;
  __REG32                 :17;
} __apb2_peri_bits;

/* RF Control Register (RFCTRL) */
typedef struct {
  __REG16  MUXSEL         : 3;
  __REG16  FREQ_LOW       : 1;
  __REG16  FREQ_HIGH      : 1;
  __REG16  RFOK           : 1;
  __REG16  BIAS_DIS_LP    : 1;
  __REG16  OSC16_DIS_LP   : 1;
  __REG16                 : 8;
} __rfctrl_bits;

/* RF Calibration Register (RFCAL) */
typedef struct {
  __REG16  AMP            : 3;
  __REG16  FRQ            : 3;
  __REG16  RF             : 3;
  __REG16  XTAL           : 3;
  __REG16                 : 4;
} __rfcal_bits;

/* PLL loop filter Calibration Register (LOOPCAL) */
typedef struct {
  __REG16  CHPUMP         : 3;
  __REG16  RES_LP         : 3;
  __REG16  CAP1_LP        : 3;
  __REG16  CAP2_LP        : 3;
  __REG16                 : 4;
} __rfloopcal_bits;

/* Clock Control Register (CLKCTL) */
typedef struct {
  __REG32  LOP_WFI        : 1;
  __REG32  WFI_CKSEL      : 1;
  __REG32  CKAF_SEL       : 1;
  __REG32  SRESEN         : 1;
  __REG32                 : 3;
  __REG32  EN_LOCK        : 1;
  __REG32  EN_CKAF        : 1;
  __REG32  EN_CK2_16      : 1;
  __REG32  EN_STOP        : 1;
  __REG32  EN_HALT        : 1;
  __REG32                 :20;
} __clkctl_bits;

/* Clock Flag Register (CLK_FLAG) */
typedef struct {
  __REG32  CSU_CKSEL      : 1;
  __REG32  LOCK           : 1;
  __REG32  CKAF_ST        : 1;
  __REG32  CK2_16         : 1;
  __REG32                 : 1;
  __REG32  SOFTRES        : 1;
  __REG32  WDGRES         : 1;
  __REG32  RTC_ALARM      : 1;
  __REG32  LVD_INT        : 1;
  __REG32                 : 1;
  __REG32  WKP_RES        : 1;
  __REG32  LOCK_I         : 1;
  __REG32  CKAF_I         : 1;
  __REG32  CK2_16_I       : 1;
  __REG32  STOP_I         : 1;
  __REG32  DIV2           : 1;
  __REG32                 :16;
} __clk_flag_bits;

/* PLL Configuration Register (PLLCONF) */
typedef struct {
  __REG32  DX0            : 1;
  __REG32                 : 3;
  __REG32  MX0            : 1;
  __REG32                 :10;
  __REG32  En_PLL         : 1;
  __REG32                 :16;
} __pllconf_bits;

/* Peripheral Enable Register (RCCU_PER) */
typedef struct {
  __REG32  GPS            : 1;
  __REG32  GPS_TIMEBASE   : 1;
  __REG32  EMI            : 1;
  __REG32  APB3           : 1;
  __REG32                 :28;
} __ph_clock_en_bits;

/* System Mode Register (SYSMODE) */
typedef struct {
  __REG32  WFI            : 1;
  __REG32  HALT           : 1;
  __REG32                 :30;
} __sysmode_bits;

/* CPU Clock Divider Control (CPUDIV) */
typedef struct {
  __REG16  FACT           : 2;
  __REG16                 :13;
  __REG16  BUSY           : 1;
} __cpudiv_bits;

/* Peripheral Clock Dividers Control Register (APBDIV) */
typedef struct {
  __REG16  FACT1          : 2;
  __REG16                 : 6;
  __REG16  FACT2          : 2;
  __REG16                 : 6;
} __apbdiv_bits;

/*Peripheral RESET Control Register (PH_RST)*/
typedef struct {
  __REG16  GPS            : 1;
  __REG16                 : 1;
  __REG16  EMI            : 1;
  __REG16  APB3           : 1;
  __REG16                 : 1;
  __REG16  CRC            : 1;
  __REG16                 :10;
} __ph_rst_bits;

/* Boot Configuration Register (BOOTCONF) */
typedef struct {
  __REG16  BOOT           : 2;
  __REG16                 : 1;
  __REG16                 : 1;
  __REG16  LPOWDBGEN      : 1;
  __REG16  ADCEN          : 1;
  __REG16                 : 3;
  __REG16  PKG176         : 1;
  __REG16                 : 2;
  __REG16  RF_BYPASS      : 1;
  __REG16  EN_BIAS        : 1;
  __REG16  _16F0_OSC_BYP  : 1;
  __REG16  BOOT_INIT      : 1;
} __bootconf_bits;

/* Power Control Register (PWRCTRL) */
typedef struct {
  __REG16                 : 3;
  __REG16  VRBYP          : 1;
  __REG16  LPVRWFI        : 1;
  __REG16  LPVRBYP        : 1;
  __REG16  PWRDWN         : 1;
  __REG16  OSCBYP         : 1;
  __REG16  LVDDIS         : 1;
  __REG16                 : 1;
  __REG16  LVDDIS1        : 1;
  __REG16  LVDDIS2        : 1;
  __REG16  VROK           : 1;
  __REG16  WKUPALRM       : 1;
  __REG16  BUSY           : 1;
  __REG16  WREN           : 1;
} __pwrctrl_bits;

/* RTC Control Register High (RTC_CRH) */
typedef struct {
  __REG16 SEN             : 1;
  __REG16 AEN             : 1;
  __REG16 OWEN            : 1;
  __REG16 GEN             : 1;
  __REG16                 :12;
} __rtc_crh_bits;

/* RTC Control Register LOW (RTC_CRL) */
typedef struct {
  __REG16 SIR             : 1;
  __REG16 AIR             : 1;
  __REG16 OWIR            : 1;
  __REG16 GIR             : 1;
  __REG16 CNF             : 1;
  __REG16 RTOFF           : 1;
  __REG16                 :10;
} __rtc_crl_bits;

/* RTC prescaler Load value high registers (RTC_LDH) */
typedef struct {
  __REG16 PRSL            : 4;
  __REG16                 :12;
} __rtc_ldh_bits;

/* RTC Prescaler Divider Register High (RTC_DIVH) */
typedef struct {
  __REG16 RTCDIV          : 4;
  __REG16                 :12;
} __rtc_divh_bits;

/* Wake-up Control Register (WUCTRL) */
typedef struct {
  __REG8 WKUP_INT         : 1;
  __REG8 ID1S             : 1;
  __REG8 STOP             : 1;
  __REG8                  : 5;
} __wuctrl_bits;

/* Interrupt Control Register (EIC_ICR) */
typedef struct {
  __REG32 IRQ_EN          : 1;
  __REG32 FIQ_EN          : 1;
  __REG32                 :30;
} __eic_icr_bits;

/* Current Interrupt Channel Register (EIC_CICR) */
typedef struct {
  __REG32 CIC             : 5;
  __REG32                 :27;
} __eic_cicr_bits;

/* Current Interrupt Priority Register (EIC_CIPR) */
typedef struct {
  __REG32 CIP             : 4;
  __REG32                 :28;
} __eic_cipr_bits;

/* Fast Interrupt Register (EIC_FIR) */
typedef struct {
  __REG32 FIE             : 2;
  __REG32 FIP             : 2;
  __REG32                 :28;
} __eic_fir_bits;

/* Interrupt Enable Register 0 (EIC_IER0) */
typedef struct {
  __REG32 IER_0           : 1;
  __REG32 IER_1           : 1;
  __REG32 IER_2           : 1;
  __REG32 IER_3           : 1;
  __REG32 IER_4           : 1;
  __REG32 IER_5           : 1;
  __REG32 IER_6           : 1;
  __REG32 IER_7           : 1;
  __REG32 IER_8           : 1;
  __REG32 IER_9           : 1;
  __REG32 IER_10          : 1;
  __REG32 IER_11          : 1;
  __REG32 IER_12          : 1;
  __REG32 IER_13          : 1;
  __REG32 IER_14          : 1;
  __REG32 IER_15          : 1;
  __REG32 IER_16          : 1;
  __REG32 IER_17          : 1;
  __REG32 IER_18          : 1;
  __REG32 IER_19          : 1;
  __REG32 IER_20          : 1;
  __REG32 IER_21          : 1;
  __REG32 IER_22          : 1;
  __REG32 IER_23          : 1;
  __REG32 IER_24          : 1;
  __REG32 IER_25          : 1;
  __REG32 IER_26          : 1;
  __REG32 IER_27          : 1;
  __REG32 IER_28          : 1;
  __REG32 IER_29          : 1;
  __REG32 IER_30          : 1;
  __REG32 IER_31          : 1;
} __eic_ier_bits;

/* Interrupt Pending Register 0 (EIC_IPR0) */
typedef struct {
  __REG32 IPR_0           : 1;
  __REG32 IPR_1           : 1;
  __REG32 IPR_2           : 1;
  __REG32 IPR_3           : 1;
  __REG32 IPR_4           : 1;
  __REG32 IPR_5           : 1;
  __REG32 IPR_6           : 1;
  __REG32 IPR_7           : 1;
  __REG32 IPR_8           : 1;
  __REG32 IPR_9           : 1;
  __REG32 IPR_10          : 1;
  __REG32 IPR_11          : 1;
  __REG32 IPR_12          : 1;
  __REG32 IPR_13          : 1;
  __REG32 IPR_14          : 1;
  __REG32 IPR_15          : 1;
  __REG32 IPR_16          : 1;
  __REG32 IPR_17          : 1;
  __REG32 IPR_18          : 1;
  __REG32 IPR_19          : 1;
  __REG32 IPR_20          : 1;
  __REG32 IPR_21          : 1;
  __REG32 IPR_22          : 1;
  __REG32 IPR_23          : 1;
  __REG32 IPR_24          : 1;
  __REG32 IPR_25          : 1;
  __REG32 IPR_26          : 1;
  __REG32 IPR_27          : 1;
  __REG32 IPR_28          : 1;
  __REG32 IPR_29          : 1;
  __REG32 IPR_30          : 1;
  __REG32 IPR_31          : 1;
} __eic_ipr_bits;

/* Source Interrupt Registers - Channel n (EIC_SIRn) */
typedef struct {
  __REG32 SIPL            : 4;
  __REG32                 :12;
  __REG32 SIV             :16;
} __eic_sirn_bits;

/* WDT Control Register (WDT_CR) */
typedef struct {
  __REG16 WE              : 1;
  __REG16 SC              : 1;
  __REG16 EE              : 1;
  __REG16                 :13;
} __wdt_cr_bits;

/* WDT Prescaler Register (WDT_PR) */
typedef struct {
  __REG16 PR              : 8;
  __REG16                 : 8;
} __wdt_pr_bits;

/* WDT Status Register (WDT_SR) */
typedef struct {
  __REG16 EC              : 1;
  __REG16                 :15;
} __wdt_sr_bits;

/* WDT Mask Register (WDT_MR) */
typedef struct {
  __REG16 ECM             : 1;
  __REG16                 :15;
} __wdt_mr_bits;

/* Control Register 1 (TIMn_CR1) */
typedef struct {
  __REG16 ECKEN           : 1;
  __REG16 EXEDG           : 1;
  __REG16 IEDGA           : 1;
  __REG16 IEDGB           : 1;
  __REG16 PWM             : 1;
  __REG16 OPM             : 1;
  __REG16 OCAE            : 1;
  __REG16 OCBE            : 1;
  __REG16 OLVLA           : 1;
  __REG16 OLVLB           : 1;
  __REG16 FOLVA           : 1;
  __REG16 FOLVB           : 1;
  __REG16                 : 2;
  __REG16 PWMI            : 1;
  __REG16 EN              : 1;
} __tim_cr1_bits;

/* Control Register 2 (TIMn_CR2) */
typedef struct {
  __REG16 CC              : 8;
  __REG16                 : 3;
  __REG16 OCBIE           : 1;
  __REG16 ICBIE           : 1;
  __REG16 TOIE            : 1;
  __REG16 OCAIE           : 1;
  __REG16 ICAIE           : 1;
} __tim_cr2_bits;

/* Status Register (TIMn_SR) */
typedef struct {
  __REG16                 :11;
  __REG16 OCFB            : 1;
  __REG16 ICFB            : 1;
  __REG16 TOF             : 1;
  __REG16 OCFA            : 1;
  __REG16 ICFA            : 1;
} __tim_sr_bits;

/* ADC Control/Status Register (ADC_CSR) */
typedef struct {
  __REG16 DA0             : 1;
  __REG16 DA1             : 1;
  __REG16 DA2             : 1;
  __REG16 DA3             : 1;
  __REG16 A               : 2;
  __REG16 AXT             : 1;
  __REG16                 : 1;
  __REG16 IE0             : 1;
  __REG16 IE1             : 1;
  __REG16 IE2             : 1;
  __REG16 IE3             : 1;
  __REG16                 : 1;
  __REG16 OR              : 1;
  __REG16                 : 2;
} __adc_csr_bits;

/* ADC Clock Prescaler Register (ADC_CPR) */
typedef struct {
  __REG16 PRESC           : 7;
  __REG16                 : 9;
} __adc_cpr_bits;

/* ADC Data Register n, n = 0..3 (ADC_DATA[n]) */
typedef struct {
  __REG16                 : 5;
  __REG16 DATA            :11;
} __adc_dr_bits;

/* CAN Control Register (CAN_CR) */
typedef struct {
  __REG16  INIT           : 1;
  __REG16  IE             : 1;
  __REG16  SIE            : 1;
  __REG16  EIE            : 1;
  __REG16                 : 1;
  __REG16  DAR            : 1;
  __REG16  CCE            : 1;
  __REG16  TEST           : 1;
  __REG16                 : 8;
} __can_cr_bits;

/* Status Register (CAN_SR) */
typedef struct {
  __REG16  LEC            : 3;
  __REG16  TXOK           : 1;
  __REG16  RXOK           : 1;
  __REG16  EPASS          : 1;
  __REG16  EWARN          : 1;
  __REG16  BOFF           : 1;
  __REG16                 : 8;
} __can_sr_bits;

/* Error Counter (CAN_ERR) */
typedef struct {
  __REG16  TEC            : 8;
  __REG16  REC            : 7;
  __REG16  RP             : 1;
} __can_err_bits;

/* Bit Timing Register (CAN_BTR) */
typedef struct {
  __REG16  BRP            : 6;
  __REG16  SJW            : 2;
  __REG16  TSEG1          : 4;
  __REG16  TSEG2          : 3;
  __REG16                 : 1;
} __can_btr_bits;

/* Test Register (CAN_TESTR) */
typedef struct {
  __REG16                 : 2;
  __REG16  BASIC          : 1;
  __REG16  SILENT         : 1;
  __REG16  LBACK          : 1;
  __REG16  TX             : 2;
  __REG16  RX             : 1;
  __REG16                 : 8;
} __can_testr_bits;

/* BRP Extension Register (CAN_BRPR) */
typedef struct {
  __REG16  BRPE           : 4;
  __REG16                 :12;
} __can_brpr_bits;

/* IFn Command Request Registers (CAN_IFn_CRR) */
typedef struct {
  __REG16  MSGNO          : 6;
  __REG16                 : 9;
  __REG16  BUSY           : 1;
} __can_ifn_crr_bits;

/* IFn Command Mask Registers (CAN_IFn_CMR) */
typedef struct {
  __REG16  DATAB          : 1;
  __REG16  DATAA          : 1;
  __REG16  TXRQST         : 1;
  __REG16  CLRLNTPND      : 1;
  __REG16  CONTROL        : 1;
  __REG16  ARB            : 1;
  __REG16  MASK           : 1;
  __REG16  WR_RD          : 1;
  __REG16                 : 8;
} __can_ifn_cmr_bits;

/* IFn Mask 1 Register (CAN_IFn_M1R) */
typedef struct {
  __REG16  MSK0           : 1;
  __REG16  MSK1           : 1;
  __REG16  MSK2           : 1;
  __REG16  MSK3           : 1;
  __REG16  MSK4           : 1;
  __REG16  MSK5           : 1;
  __REG16  MSK6           : 1;
  __REG16  MSK7           : 1;
  __REG16  MSK8           : 1;
  __REG16  MSK9           : 1;
  __REG16  MSK10          : 1;
  __REG16  MSK11          : 1;
  __REG16  MSK12          : 1;
  __REG16  MSK13          : 1;
  __REG16  MSK14          : 1;
  __REG16  MSK15          : 1;
} __can_ifn_m1r_bits;

/* IFn Mask 2 Register (CAN_IFn_M2R) */
typedef struct {
  __REG16  MSK16          : 1;
  __REG16  MSK17          : 1;
  __REG16  MSK18          : 1;
  __REG16  MSK19          : 1;
  __REG16  MSK20          : 1;
  __REG16  MSK21          : 1;
  __REG16  MSK22          : 1;
  __REG16  MSK23          : 1;
  __REG16  MSK24          : 1;
  __REG16  MSK25          : 1;
  __REG16  MSK26          : 1;
  __REG16  MSK27          : 1;
  __REG16  MSK28          : 1;
  __REG16                 : 1;
  __REG16  MDIR           : 1;
  __REG16  MXTD           : 1;
} __can_ifn_m2r_bits;

/* IFn Message Arbitration 1 Register (CAN_IFn_A1R) */
typedef struct {
  __REG16  ID0            : 1;
  __REG16  ID1            : 1;
  __REG16  ID2            : 1;
  __REG16  ID3            : 1;
  __REG16  ID4            : 1;
  __REG16  ID5            : 1;
  __REG16  ID6            : 1;
  __REG16  ID7            : 1;
  __REG16  ID8            : 1;
  __REG16  ID9            : 1;
  __REG16  ID10           : 1;
  __REG16  ID11           : 1;
  __REG16  ID12           : 1;
  __REG16  ID13           : 1;
  __REG16  ID14           : 1;
  __REG16  ID15           : 1;
} __can_ifn_a1r_bits;

/* IFn Message Arbitration 2 Register (CAN_IFn_A2R) */
typedef struct {
  __REG16  ID16           : 1;
  __REG16  ID17           : 1;
  __REG16  ID18           : 1;
  __REG16  ID19           : 1;
  __REG16  ID20           : 1;
  __REG16  ID21           : 1;
  __REG16  ID22           : 1;
  __REG16  ID23           : 1;
  __REG16  ID24           : 1;
  __REG16  ID25           : 1;
  __REG16  ID26           : 1;
  __REG16  ID27           : 1;
  __REG16  ID28           : 1;
  __REG16  DIR            : 1;
  __REG16  XTD            : 1;
  __REG16  MSGVAL         : 1;
} __can_ifn_a2r_bits;

/* Message Control Registers (CAN_IFn_MCR) */
typedef struct {
  __REG16  DLC            : 4;
  __REG16                 : 3;
  __REG16  EOB            : 1;
  __REG16  TXRQST         : 1;
  __REG16  RMTEN          : 1;
  __REG16  RXIE           : 1;
  __REG16  TXIE           : 1;
  __REG16  UMASK          : 1;
  __REG16  INTPND         : 1;
  __REG16  MSGLST         : 1;
  __REG16  NEWDAT         : 1;
} __can_ifn_mcr_bits;

/* IFn Data A/B Registers (CAN_IFn_DAnR and CAN_IFn_DBnR) */
typedef struct {
  __REG16  DATA0          : 8;
  __REG16  DATA1          : 8;
} __can_ifn_da1r_bits;

typedef struct {
  __REG16  DATA2          : 8;
  __REG16  DATA3          : 8;
} __can_ifn_da2r_bits;

typedef struct {
  __REG16  DATA4          : 8;
  __REG16  DATA5          : 8;
} __can_ifn_db1r_bits;

typedef struct {
  __REG16  DATA6          : 8;
  __REG16  DATA7          : 8;
} __can_ifn_db2r_bits;

/* I2C Control Register (I2Cn_CR) */
typedef struct {
  __REG8  ITE             : 1;
  __REG8  STOP            : 1;
  __REG8  ACK             : 1;
  __REG8  START           : 1;
  __REG8  ENGC            : 1;
  __REG8  PE              : 1;
  __REG8                  : 2;
} __i2cn_cr_bits;

/* I2C Status Register 1 (I2Cn_SR1) */
typedef struct {
  __REG8  SB              : 1;
  __REG8  MSL             : 1;
  __REG8  ADSL            : 1;
  __REG8  BTF             : 1;
  __REG8  BUSY            : 1;
  __REG8  TRA             : 1;
  __REG8  ADD10           : 1;
  __REG8  EVF             : 1;
} __i2cn_sr1_bits;

/* I2C Status Register 2 (I2Cn_SR2) */
typedef struct {
  __REG8  GCAL            : 1;
  __REG8  BERR            : 1;
  __REG8  ARLO            : 1;
  __REG8  STOPF           : 1;
  __REG8  AF              : 1;
  __REG8  ENDAD           : 1;
  __REG8                  : 2;
} __i2cn_sr2_bits;

/* I2C Clock Control Register (I2Cn_CCR) */
typedef struct {
  __REG8  CC0             : 1;
  __REG8  CC1             : 1;
  __REG8  CC2             : 1;
  __REG8  CC3             : 1;
  __REG8  CC4             : 1;
  __REG8  CC5             : 1;
  __REG8  CC6             : 1;
  __REG8  FM_SM           : 1;
} __i2cn_ccr_bits;

/* I2C Extended Clock Control Register (I2Cn_ECCR) */
typedef struct {
  __REG8  CC7             : 1;
  __REG8  CC8             : 1;
  __REG8  CC9             : 1;
  __REG8  CC10            : 1;
  __REG8  CC11            : 1;
  __REG8                  : 3;
} __i2cn_eccr_bits;

/* I2C Own Address Register 1 (I2Cn_OAR1) */
typedef struct {
  __REG8  ADD0            : 1;
  __REG8  ADD1            : 1;
  __REG8  ADD2            : 1;
  __REG8  ADD3            : 1;
  __REG8  ADD4            : 1;
  __REG8  ADD5            : 1;
  __REG8  ADD6            : 1;
  __REG8  ADD7            : 1;
} __i2cn_oar1_bits;

/* I2C Own Address Register 2 (I2Cn_OAR2) */
typedef struct {
  __REG8                  : 1;
  __REG8  ADD8            : 1;
  __REG8  ADD9            : 1;
  __REG8                  : 2;
  __REG8  FR0             : 1;
  __REG8  FR1             : 1;
  __REG8  FR2             : 1;
} __i2cn_oar2_bits;

/* BSPI Control/Status Register 1 (BSPIn_CSR1) */
typedef struct {
  __REG16  BSPE           : 1;
  __REG16  MSTR           : 1;
  __REG16  RIE            : 2;
  __REG16  REIE           : 1;
  __REG16                 : 2;
  __REG16  BEIE           : 1;
  __REG16  CPOL           : 1;
  __REG16  CPHA           : 1;
  __REG16  WL             : 2;
  __REG16  RFE            : 4;
} __bspin_csr1_bits;

/* BSPI Control/Status Register 2 (BSPIn_CSR2) */
typedef struct {
  __REG16  DFIFO          : 1;
  __REG16                 : 1;
  __REG16  BERR           : 1;
  __REG16  RFNE           : 1;
  __REG16  RFF            : 1;
  __REG16  ROFL           : 1;
  __REG16  TFE            : 1;
  __REG16  TUFL           : 1;
  __REG16  TFF            : 1;
  __REG16  TFNE           : 1;
  __REG16  TFEDEPTH       : 4;
  __REG16  TIE            : 2;
} __bspin_csr2_bits;

/* BSPI Transmit Register (BSPIn_TXR) */
typedef struct {
  __REG16                 : 8;
  __REG16  TX_8BITS       : 8;
} __bspin_tx_bits;

/* BSPI Receive Register (BSPIn_RXR) */
typedef struct {
  __REG16                 : 8;
  __REG16  RX_8BITS       : 8;
} __bspin_rxr_bits;

/* UART TxBuffer Register (UARTn_TxBUFR) */
typedef struct {
  __REG16  TX             : 9;
  __REG16                 : 7;
} __uartn_tx_bits;

/* UART RxBuffer Register (UARTn_RxBUFR) */
typedef struct {
  __REG16  RX             : 9;
  __REG16                 : 7;
} __uartn_rx_bits;

/* UART Control Register (UARTn_CR) */
typedef struct {
  __REG16  MODE           : 3;
  __REG16  STOPBITS       : 2;
  __REG16  PARITYODD      : 1;
  __REG16  LOOPBACK       : 1;
  __REG16  RUN            : 1;
  __REG16  RXENABLE       : 1;
  __REG16  SCENABLE       : 1;
  __REG16  FIFOENABLE     : 1;
  __REG16                 : 5;
} __uartn_cr_bits;

/* UART IntEnable Register (UARTn_IER) */
typedef struct {
  __REG16  RxBufNotEmptyIE   : 1;
  __REG16  TxEmptyIE         : 1;
  __REG16  TxHalfEmptyIE     : 1;
  __REG16  ParityErrorIE     : 1;
  __REG16  FrameErrorIE      : 1;
  __REG16  OverrunErrorIE    : 1;
  __REG16  TimeoutNotEmptyIE : 1;
  __REG16  TimeoutIdleIE     : 1;
  __REG16  RxHalfFullIE      : 1;
  __REG16                    : 7;
} __uartn_ier_bits;

/* UART Status Register (UARTn_SR) */
typedef struct {
  __REG16  RxBufNotEmpty     : 1;
  __REG16  TxEmpty           : 1;
  __REG16  TxHalfEmpty       : 1;
  __REG16  ParityError       : 1;
  __REG16  FrameError        : 1;
  __REG16  OverrunError      : 1;
  __REG16  TimeoutNotEmpty   : 1;
  __REG16  TimeoutIdle       : 1;
  __REG16  RxHalfFull        : 1;
  __REG16  TxFull            : 1;
  __REG16                    : 6;
} __uartn_sr_bits;

/* UART GuardTime Register (UARTn_GTR) */
typedef struct {
  __REG16  GUARDTIME      : 8;
  __REG16                 : 8;
} __uartn_gtr_bits;

/* UART Timeout Register (UARTn_TOR) */
typedef struct {
  __REG16  TIMEOUT        : 8;
  __REG16                 : 8;
} __uartn_tor_bits;

/* SmartCard Clock Prescaler Value (SC_CLKVAL) */
typedef struct {
  __REG16  SCCLKVAL       : 5;
  __REG16                 :11;
} __smcard_clkval_bits;

/* SmartCard Clock Control Register (SC_CLKCON) */
typedef struct {
  __REG16  EN             : 1;
  __REG16                 :15;
} __smcard_clkcon_bits;

/* Pseudo Random Noise Code register (PRNCx) */
typedef struct {
  __REG16  PRNC           : 7;
  __REG16                 : 9;
} __gps_prnc_bits;

/* PRN Phase register (PRNPx) */
typedef struct {
  __REG32  PRNP           :19;
  __REG32                 :13;
} __gps_prnp_bits;

/* NCO Frequency register (NCOFx) */
typedef struct {
  __REG32  NCOF           :18;
  __REG32                 :14;
} __gps_ncof_bits;

/* NCO Phase register (NCOPx) */
typedef struct {
  __REG16  NCOP           : 7;
  __REG16                 : 9;
} __gps_ncop_bits;

/* PRN Generator Initial value register (PRNGI) */
typedef struct {
  __REG16  PRNGI          :10;
  __REG16                 : 6;
} __gps_prngi_bits;

/* Control register (CTL) */
typedef struct {
  __REG16  SR		          : 2;
  __REG16  NCORE          : 1;
  __REG16  PRND           : 1;
  __REG16  LBKE           : 1;
  __REG16                 :11;
} __gps_ctl_bits;

/* Time Base Low register (TBL) */
typedef struct {
  __REG16   		          : 2;
  __REG16  TBL            :14;
} __gps_tbl_bits;

/* Compare Set Low register (CPSL) */
typedef struct {
  __REG16   		          : 2;
  __REG16  CPSL           :14;
} __gps_cpsl_bits;

/* Compare Reset Low register (CPRL) */
typedef struct {
  __REG16   		          : 2;
  __REG16  CPRL           :14;
} __gps_cprl_bits;

/* Time Base Buffer Low register (TBBL) */
typedef struct {
  __REG16   		          : 2;
  __REG16  TBBL           :14;
} __gps_tbbl_bits;

/* PRN Disable register (PRND) */
typedef struct {
  __REG16  PRND0          : 1;
  __REG16  PRND1          : 1;
  __REG16  PRND2          : 1;
  __REG16  PRND3          : 1;
  __REG16  PRND4          : 1;
  __REG16  PRND5          : 1;
  __REG16  PRND6          : 1;
  __REG16  PRND7          : 1;
  __REG16  PRND8          : 1;
  __REG16  PRND9          : 1;
  __REG16  PRND10         : 1;
  __REG16  PRND11         : 1;
  __REG16   		          : 4;
} __gps_prnd_bits;

/* Interrupt Status register (IS) */
typedef struct {
  __REG16  ODRP           : 1;
  __REG16  ODOV           : 1;
  __REG16   		          :14;
} __gps_is_bits;

/* Interrupt Mask register (IM) */
typedef struct {
  __REG16  ODRM           : 1;
  __REG16   		          :15;
} __gps_im_bits;

/* SyncMode register (SM) */
typedef struct {
  __REG16  SYNC           : 8;
  __REG16  MODE           : 8;
} __gps_sm_bits;

/* Time register (TIME_XY) */
typedef struct {
  __REG16  TIMEY          : 8;
  __REG16  TIMEX          : 8;
} __gps_time_xy_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  EMI
 **
 ***************************************************************************/
__IO_REG16_BIT(BCON0,       0x6C000000,__READ_WRITE,__bankcfg_bits);
__IO_REG16_BIT(BCON1,       0x6C000004,__READ_WRITE,__bankcfg_bits);
__IO_REG16_BIT(BCON2,       0x6C000008,__READ_WRITE,__bankcfg_bits);
__IO_REG16_BIT(BCON3,       0x6C00000C,__READ_WRITE,__bankcfg_bits);

/***************************************************************************
 **
 ** APB1
 **
 ***************************************************************************/
__IO_REG32_BIT(APB1_BSR,    0xC0000000,__READ_WRITE,__apb_bsr_bits);
__IO_REG32_BIT(APB1_TOR,    0xC0000004,__READ_WRITE,__apb_tor_bits);
__IO_REG32_BIT(APB1_OMR,    0xC0000008,__READ      ,__apb_omr_bits);
__IO_REG32_BIT(APB1_TOER,   0xC000000C,__READ      ,__apb_toer_bits);
__IO_REG32_BIT(APB1_CKDIS,  0xC0000010,__READ_WRITE,__apb1_peri_bits);
__IO_REG32_BIT(APB1_SWRES,  0xC0000014,__READ_WRITE,__apb1_peri_bits);

/***************************************************************************
 **
 ** APB2
 **
 ***************************************************************************/
__IO_REG32_BIT(APB2_BSR,    0xE0000000,__READ_WRITE,__apb_bsr_bits);
__IO_REG32_BIT(APB2_TOR,    0xE0000004,__READ_WRITE,__apb_tor_bits);
__IO_REG32_BIT(APB2_OMR,    0xE0000008,__READ      ,__apb_omr_bits);
__IO_REG32_BIT(APB2_TOER,   0xE000000C,__READ      ,__apb_toer_bits);
__IO_REG32_BIT(APB2_CKDIS,  0xE0000010,__READ_WRITE,__apb2_peri_bits);
__IO_REG32_BIT(APB2_SWRES,  0xE0000014,__READ_WRITE,__apb2_peri_bits);

/***************************************************************************
 **
 ** APB3
 **
 ***************************************************************************/
__IO_REG32_BIT(APB3_BSR,    0x8000F000,__READ_WRITE,__apb_bsr_bits);
__IO_REG32_BIT(APB3_TOR,    0x8000F004,__READ_WRITE,__apb_tor_bits);
__IO_REG32_BIT(APB3_OMR,    0x8000F008,__READ      ,__apb_omr_bits);
__IO_REG32_BIT(APB3_TOER,   0x8000F00C,__READ      ,__apb_toer_bits);

/***************************************************************************
 **
 **  RF
 **
 ***************************************************************************/
__IO_REG16_BIT(RFCTRL,      0xA0000058,__READ_WRITE,__rfctrl_bits);
__IO_REG16_BIT(RFCAL,       0xA000005C,__READ_WRITE,__rfcal_bits);
__IO_REG16_BIT(RFLOOPCAL,   0xA0000060,__READ_WRITE,__rfloopcal_bits);

/***************************************************************************
 **
 **  PRCCU
 **
 ***************************************************************************/
__IO_REG32_BIT(CLKCTL,      0xA0000000,__READ_WRITE,__clkctl_bits);
__IO_REG32_BIT(CLK_FLAG,    0xA0000008,__READ_WRITE,__clk_flag_bits);
__IO_REG32_BIT(PLLCONF,     0xA0000018,__READ_WRITE,__pllconf_bits);
__IO_REG32_BIT(PH_CLOCK_EN, 0xA000001C,__READ_WRITE,__ph_clock_en_bits);
__IO_REG32_BIT(SYSMODE ,    0xA0000020,__READ_WRITE,__sysmode_bits);

/***************************************************************************
 **
 **  CPM
 **
 ***************************************************************************/
__IO_REG16_BIT(CPUDIV,      0xA0000040,__READ_WRITE,__cpudiv_bits);
__IO_REG16_BIT(APBDIV,      0xA0000044,__READ_WRITE,__apbdiv_bits);
__IO_REG16_BIT(PH_RST,      0xA0000048,__READ_WRITE,__ph_rst_bits);
__IO_REG16_BIT(BOOTCONF,    0xA0000050,__READ_WRITE,__bootconf_bits);
__IO_REG16_BIT(PWRCTRL,     0xA0000054,__READ_WRITE,__pwrctrl_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG16_BIT(RTC_CRH,     0xE000D000,__READ_WRITE,__rtc_crh_bits);
__IO_REG16_BIT(RTC_CRL,     0xE000D004,__READ_WRITE,__rtc_crl_bits);
__IO_REG16_BIT(RTC_LDH,     0xE000D008,__WRITE     ,__rtc_ldh_bits);
__IO_REG16(    RTC_LDL,     0xE000D00C,__WRITE);
__IO_REG16_BIT(RTC_DIVH,    0xE000D010,__READ      ,__rtc_divh_bits);
__IO_REG16(    RTC_DIVL,    0xE000D014,__READ);
__IO_REG16(    RTC_CNTH,    0xE000D018,__READ_WRITE);
__IO_REG16(    RTC_CNTL,    0xE000D01C,__READ_WRITE);
__IO_REG16(    RTC_ALH,     0xE000D020,__WRITE);
__IO_REG16(    RTC_ALL,     0xE000D024,__WRITE);

/***************************************************************************
 **
 ** WUIMU
 **
 ***************************************************************************/
__IO_REG8_BIT(WU_SRL,       0xE000101C,__READ_WRITE,__BITS8);
__IO_REG8_BIT(WU_CR,        0xE0001024,__READ_WRITE,__wuctrl_bits);
__IO_REG8_BIT(WU_MRH,       0xE0001028,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(WU_MRL,       0xE000102C,__READ_WRITE,__BITS8);
__IO_REG8_BIT(WU_TRH,       0xE0001030,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(WU_TRL,       0xE0001034,__READ_WRITE,__BITS8);
__IO_REG8_BIT(WU_PRH,       0xE0001038,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(WU_PRL,       0xE000103C,__READ_WRITE,__BITS8);

/***************************************************************************
 **
 ** IOPORT0
 **
 ***************************************************************************/
__IO_REG16_BIT(IOPORT0_PC0, 0xE0003000,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT0_PC1, 0xE0003004,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT0_PC2, 0xE0003008,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT0_PD,  0xE000300C,__READ_WRITE,__BITS16);

/***************************************************************************
 **
 ** IOPORT1
 **
 ***************************************************************************/
__IO_REG16_BIT(IOPORT1_PC0, 0xE0004000,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT1_PC1, 0xE0004004,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT1_PC2, 0xE0004008,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT1_PD,  0xE000400C,__READ_WRITE,__BITS16);

/***************************************************************************
 **
 ** IOPORT2
 **
 ***************************************************************************/
__IO_REG16_BIT(IOPORT2_PC0, 0xE0005000,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT2_PC1, 0xE0005004,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT2_PC2, 0xE0005008,__READ_WRITE,__BITS16);
__IO_REG16_BIT(IOPORT2_PD,  0xE000500C,__READ_WRITE,__BITS16);

/***************************************************************************
 **
 ** EIC
 **
 ***************************************************************************/
__IO_REG32_BIT(EIC_ICR,     0xFFFFF800,__READ_WRITE,__eic_icr_bits);
__IO_REG32_BIT(EIC_CICR,    0xFFFFF804,__READ,      __eic_cicr_bits);
__IO_REG32_BIT(EIC_CIPR,    0xFFFFF808,__READ_WRITE,__eic_cipr_bits);
__IO_REG32(    EIC_IVR,     0xFFFFF818,__READ_WRITE);
__IO_REG32_BIT(EIC_FIR,     0xFFFFF81C,__READ_WRITE,__eic_fir_bits);
__IO_REG32_BIT(EIC_IER0,    0xFFFFF820,__READ_WRITE,__eic_ier_bits);
__IO_REG32_BIT(EIC_IPR0,    0xFFFFF840,__READ_WRITE,__eic_ipr_bits);
__IO_REG32_BIT(EIC_SIR0,    0xFFFFF860,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR1,    0xFFFFF864,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR2,    0xFFFFF868,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR3,    0xFFFFF86C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR4,    0xFFFFF870,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR5,    0xFFFFF874,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR6,    0xFFFFF878,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR7,    0xFFFFF87C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR8,    0xFFFFF880,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR9,    0xFFFFF884,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR10,   0xFFFFF888,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR11,   0xFFFFF88C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR12,   0xFFFFF890,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR13,   0xFFFFF894,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR14,   0xFFFFF898,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR15,   0xFFFFF89C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR16,   0xFFFFF8A0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR17,   0xFFFFF8A4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR18,   0xFFFFF8A8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR19,   0xFFFFF8AC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR20,   0xFFFFF8B0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR21,   0xFFFFF8B4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR22,   0xFFFFF8B8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR23,   0xFFFFF8BC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR24,   0xFFFFF8C0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR25,   0xFFFFF8C4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR26,   0xFFFFF8C8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR27,   0xFFFFF8CC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR28,   0xFFFFF8D0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR29,   0xFFFFF8D4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR30,   0xFFFFF8D8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_SIR31,   0xFFFFF8DC,__READ_WRITE,__eic_sirn_bits);

/***************************************************************************
 **
 ** EIC APB2
 **
 ***************************************************************************/
__IO_REG32_BIT(EIC_APB2_ICR,     0xFFFFFC00,__READ_WRITE,__eic_icr_bits);
__IO_REG32_BIT(EIC_APB2_CICR,    0xFFFFFC04,__READ,      __eic_cicr_bits);
__IO_REG32_BIT(EIC_APB2_CIPR,    0xFFFFFC08,__READ_WRITE,__eic_cipr_bits);
__IO_REG32(    EIC_APB2_IVR,     0xFFFFFC18,__READ_WRITE);
__IO_REG32_BIT(EIC_APB2_FIR,     0xFFFFFC1C,__READ_WRITE,__eic_fir_bits);
__IO_REG32_BIT(EIC_APB2_IER0,    0xFFFFFC20,__READ_WRITE,__eic_ier_bits);
__IO_REG32_BIT(EIC_APB2_IPR0,    0xFFFFFC40,__READ_WRITE,__eic_ipr_bits);
__IO_REG32_BIT(EIC_APB2_SIR0,    0xFFFFFC60,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR1,    0xFFFFFC64,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR2,    0xFFFFFC68,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR3,    0xFFFFFC6C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR4,    0xFFFFFC70,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR5,    0xFFFFFC74,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR6,    0xFFFFFC78,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR7,    0xFFFFFC7C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR8,    0xFFFFFC80,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR9,    0xFFFFFC84,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR10,   0xFFFFFC88,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR11,   0xFFFFFC8C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR12,   0xFFFFFC90,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR13,   0xFFFFFC94,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR14,   0xFFFFFC98,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR15,   0xFFFFFC9C,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR16,   0xFFFFFCA0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR17,   0xFFFFFCA4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR18,   0xFFFFFCA8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR19,   0xFFFFFCAC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR20,   0xFFFFFCB0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR21,   0xFFFFFCB4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR22,   0xFFFFFCB8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR23,   0xFFFFFCBC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR24,   0xFFFFFCC0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR25,   0xFFFFFCC4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR26,   0xFFFFFCC8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR27,   0xFFFFFCCC,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR28,   0xFFFFFCD0,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR29,   0xFFFFFCD4,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR30,   0xFFFFFCD8,__READ_WRITE,__eic_sirn_bits);
__IO_REG32_BIT(EIC_APB2_SIR31,   0xFFFFFCDC,__READ_WRITE,__eic_sirn_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG16_BIT(WDT_CR,      0xE000E000,__READ_WRITE,__wdt_cr_bits);
__IO_REG16_BIT(WDT_PR,      0xE000E004,__READ_WRITE,__wdt_pr_bits);
__IO_REG16(    WDT_VR,      0xE000E008,__READ_WRITE);
__IO_REG16(    WDT_CN,      0xE000E00C,__READ);
__IO_REG16_BIT(WDT_SR,      0xE000E010,__READ_WRITE,__wdt_sr_bits);
__IO_REG16_BIT(WDT_MR,      0xE000E014,__READ_WRITE,__wdt_mr_bits);
__IO_REG16(    WDTKR,       0xE000E018,__READ_WRITE);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG16(    TIM0_ICAR,   0xE0009000,__READ);
__IO_REG16(    TIM0_ICBR,   0xE0009004,__READ);
__IO_REG16(    TIM0_OCAR,   0xE0009008,__READ_WRITE);
__IO_REG16(    TIM0_OCBR,   0xE000900C,__READ_WRITE);
__IO_REG16(    TIM0_CNTR,   0xE0009010,__READ);
__IO_REG16_BIT(TIM0_CR1,    0xE0009014,__READ_WRITE,__tim_cr1_bits);
__IO_REG16_BIT(TIM0_CR2,    0xE0009018,__READ_WRITE,__tim_cr2_bits);
__IO_REG16_BIT(TIM0_SR,     0xE000901C,__READ_WRITE,__tim_sr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG16(    TIM1_ICAR,   0xE000A000,__READ);
__IO_REG16(    TIM1_ICBR,   0xE000A004,__READ);
__IO_REG16(    TIM1_OCAR,   0xE000A008,__READ_WRITE);
__IO_REG16(    TIM1_OCBR,   0xE000A00C,__READ_WRITE);
__IO_REG16(    TIM1_CNTR,   0xE000A010,__READ);
__IO_REG16_BIT(TIM1_CR1,    0xE000A014,__READ_WRITE,__tim_cr1_bits);
__IO_REG16_BIT(TIM1_CR2,    0xE000A018,__READ_WRITE,__tim_cr2_bits);
__IO_REG16_BIT(TIM1_SR,     0xE000A01C,__READ_WRITE,__tim_sr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG16(    TIM2_ICAR,   0xE000B000,__READ);
__IO_REG16(    TIM2_ICBR,   0xE000B004,__READ);
__IO_REG16(    TIM2_OCAR,   0xE000B008,__READ_WRITE);
__IO_REG16(    TIM2_OCBR,   0xE000B00C,__READ_WRITE);
__IO_REG16(    TIM2_CNTR,   0xE000B010,__READ);
__IO_REG16_BIT(TIM2_CR1,    0xE000B014,__READ_WRITE,__tim_cr1_bits);
__IO_REG16_BIT(TIM2_CR2,    0xE000B018,__READ_WRITE,__tim_cr2_bits);
__IO_REG16_BIT(TIM2_SR,     0xE000B01C,__READ_WRITE,__tim_sr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG16(    TIM3_ICAR,   0xE000C000,__READ);
__IO_REG16(    TIM3_ICBR,   0xE000C004,__READ);
__IO_REG16(    TIM3_OCAR,   0xE000C008,__READ_WRITE);
__IO_REG16(    TIM3_OCBR,   0xE000C00C,__READ_WRITE);
__IO_REG16(    TIM3_CNTR,   0xE000C010,__READ);
__IO_REG16_BIT(TIM3_CR1,    0xE000C014,__READ_WRITE,__tim_cr1_bits);
__IO_REG16_BIT(TIM3_CR2,    0xE000C018,__READ_WRITE,__tim_cr2_bits);
__IO_REG16_BIT(TIM3_SR,     0xE000C01C,__READ_WRITE,__tim_sr_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG16_BIT(ADCDATA0,   	0xE0007000,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADCDATA1,   	0xE0007008,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADCDATA2,   	0xE0007010,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADCDATA3,   	0xE0007018,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADCCSR,      0xE0007020,__READ_WRITE,__adc_csr_bits);
__IO_REG16(ADCTESTREG,     	0xE0007028,__READ_WRITE);
__IO_REG16_BIT(ADCCPR,     	0xE0007030,__WRITE,     __adc_cpr_bits);

/***************************************************************************
 **
 **  CAN
 **
 ***************************************************************************/
__IO_REG16_BIT(CAN_CR,      0xC0009000,__READ_WRITE,__can_cr_bits);
__IO_REG16_BIT(CAN_SR,      0xC0009004,__READ_WRITE,__can_sr_bits);
__IO_REG16_BIT(CAN_ERR,     0xC0009008,__READ      ,__can_err_bits);
__IO_REG16_BIT(CAN_BTR,     0xC000900C,__READ_WRITE,__can_btr_bits);
__IO_REG16(    CAN_IDR,     0xC0009010,__READ);
__IO_REG16_BIT(CAN_TESTR,   0xC0009014,__READ_WRITE,__can_testr_bits);
__IO_REG16_BIT(CAN_BRPR,    0xC0009018,__READ_WRITE,__can_brpr_bits);
__IO_REG16_BIT(CAN_IF1_CRR, 0xC0009020,__READ_WRITE,__can_ifn_crr_bits);
__IO_REG16_BIT(CAN_IF1_CMR, 0xC0009024,__READ_WRITE,__can_ifn_cmr_bits);
__IO_REG16_BIT(CAN_IF1_M1R, 0xC0009028,__READ_WRITE,__can_ifn_m1r_bits);
__IO_REG16_BIT(CAN_IF1_M2R, 0xC000902C,__READ_WRITE,__can_ifn_m2r_bits);
__IO_REG16_BIT(CAN_IF1_A1R, 0xC0009030,__READ_WRITE,__can_ifn_a1r_bits);
__IO_REG16_BIT(CAN_IF1_A2R, 0xC0009034,__READ_WRITE,__can_ifn_a2r_bits);
__IO_REG16_BIT(CAN_IF1_MCR, 0xC0009038,__READ_WRITE,__can_ifn_mcr_bits);
__IO_REG16_BIT(CAN_IF1_DA1R,0xC000903C,__READ_WRITE,__can_ifn_da1r_bits);
__IO_REG16_BIT(CAN_IF1_DA2R,0xC0009040,__READ_WRITE,__can_ifn_da2r_bits);
__IO_REG16_BIT(CAN_IF1_DB1R,0xC0009044,__READ_WRITE,__can_ifn_db1r_bits);
__IO_REG16_BIT(CAN_IF1_DB2R,0xC0009048,__READ_WRITE,__can_ifn_db2r_bits);
__IO_REG16_BIT(CAN_IF2_CRR, 0xC0009080,__READ_WRITE,__can_ifn_crr_bits);
__IO_REG16_BIT(CAN_IF2_CMR, 0xC0009084,__READ_WRITE,__can_ifn_cmr_bits);
__IO_REG16_BIT(CAN_IF2_M1R, 0xC0009088,__READ_WRITE,__can_ifn_m1r_bits);
__IO_REG16_BIT(CAN_IF2_M2R, 0xC000908C,__READ_WRITE,__can_ifn_m2r_bits);
__IO_REG16_BIT(CAN_IF2_A1R, 0xC0009090,__READ_WRITE,__can_ifn_a1r_bits);
__IO_REG16_BIT(CAN_IF2_A2R, 0xC0009094,__READ_WRITE,__can_ifn_a2r_bits);
__IO_REG16_BIT(CAN_IF2_MCR, 0xC0009098,__READ_WRITE,__can_ifn_mcr_bits);
__IO_REG16_BIT(CAN_IF2_DA1R,0xC000909c,__READ_WRITE,__can_ifn_da1r_bits);
__IO_REG16_BIT(CAN_IF2_DA2R,0xC00090A0,__READ_WRITE,__can_ifn_da2r_bits);
__IO_REG16_BIT(CAN_IF2_DB1R,0xC00090A4,__READ_WRITE,__can_ifn_db1r_bits);
__IO_REG16_BIT(CAN_IF2_DB2R,0xC00090A8,__READ_WRITE,__can_ifn_db2r_bits);
__IO_REG32(    CAN_TXR1R,   0xC0009100,__READ);
__IO_REG32(    CAN_TXR2R,   0xC0009104,__READ);
__IO_REG32(    CAN_ND1R,    0xC0009120,__READ);
__IO_REG32(    CAN_ND2R,    0xC0009124,__READ);
__IO_REG32(    CAN_IP1R,    0xC0009140,__READ);
__IO_REG32(    CAN_IP2R,    0xC0009144,__READ);
__IO_REG32(    CAN_MV1R,    0xC0009160,__READ);
__IO_REG32(    CAN_MV2R,    0xC0009164,__READ);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG8_BIT(I2C_CR,       0xC0002000,__READ_WRITE,__i2cn_cr_bits);
__IO_REG8_BIT(I2C_SR1,      0xC0002004,__READ      ,__i2cn_sr1_bits);
__IO_REG8_BIT(I2C_SR2,      0xC0002008,__READ      ,__i2cn_sr2_bits);
__IO_REG8_BIT(I2C_CCR,      0xC000200C,__READ_WRITE,__i2cn_ccr_bits);
__IO_REG8_BIT(I2C_OAR1,     0xC0002010,__READ_WRITE,__i2cn_oar1_bits);
__IO_REG8_BIT(I2C_OAR2,     0xC0002014,__READ_WRITE,__i2cn_oar2_bits);
__IO_REG8(    I2C_DR,       0xC0002018,__READ_WRITE);
__IO_REG8_BIT(I2C_ECCR,     0xC000201C,__READ_WRITE,__i2cn_eccr_bits);

/***************************************************************************
 **
 **  BSPI
 **
 ***************************************************************************/
__IO_REG16_BIT(BSPI_RXR,    0xC000B000,__READ      ,__bspin_rxr_bits);
__IO_REG16_BIT(BSPI_TXR,    0xC000B004,__WRITE     ,__bspin_tx_bits);
__IO_REG16_BIT(BSPI_CSR1,   0xC000B008,__READ_WRITE,__bspin_csr1_bits);
__IO_REG16_BIT(BSPI_CSR2,   0xC000B00C,__READ_WRITE,__bspin_csr2_bits);
__IO_REG8(     BSPI_CLK,    0xC000B010,__READ_WRITE);

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
__IO_REG16(    UART0_BR,    0xC0004000,__READ_WRITE);
__IO_REG16_BIT(UART0_TXBUFR,0xC0004004,__WRITE     ,__uartn_tx_bits);
__IO_REG16_BIT(UART0_RXBUFR,0xC0004008,__READ      ,__uartn_rx_bits);
__IO_REG16_BIT(UART0_CR,    0xC000400C,__READ_WRITE,__uartn_cr_bits);
__IO_REG16_BIT(UART0_IER,   0xC0004010,__READ_WRITE,__uartn_ier_bits);
__IO_REG16_BIT(UART0_SR,    0xC0004014,__READ      ,__uartn_sr_bits);
__IO_REG16_BIT(UART0_GTR,   0xC0004018,__READ_WRITE,__uartn_gtr_bits);
__IO_REG16_BIT(UART0_TOR,   0xC000401C,__READ_WRITE,__uartn_tor_bits);
__IO_REG16(    UART0_TXRSTR,0xC0004020,__WRITE);
__IO_REG16(    UART0_RXRSTR,0xC0004024,__WRITE);

/***************************************************************************
 **
 **  UART1 + SMARTCARD
 **
 ***************************************************************************/
__IO_REG16(    UART1_BR,    0xC0005000,__READ_WRITE);
__IO_REG16_BIT(UART1_TXBUFR,0xC0005004,__WRITE     ,__uartn_tx_bits);
__IO_REG16_BIT(UART1_RXBUFR,0xC0005008,__READ      ,__uartn_rx_bits);
__IO_REG16_BIT(UART1_CR,    0xC000500C,__READ_WRITE,__uartn_cr_bits);
__IO_REG16_BIT(UART1_IER,   0xC0005010,__READ_WRITE,__uartn_ier_bits);
__IO_REG16_BIT(UART1_SR,    0xC0005014,__READ      ,__uartn_sr_bits);
__IO_REG16_BIT(UART1_GTR,   0xC0005018,__READ_WRITE,__uartn_gtr_bits);
__IO_REG16_BIT(UART1_TOR,   0xC000501C,__READ_WRITE,__uartn_tor_bits);
__IO_REG16(    UART1_TXRSTR,0xC0005020,__WRITE);
__IO_REG16(    UART1_RXRSTR,0xC0005024,__WRITE);
__IO_REG16_BIT(SC_CLKVAL,   0xC0005040,__READ_WRITE,__smcard_clkval_bits);
__IO_REG16_BIT(SC_CLKCON,   0xC0005044,__READ_WRITE,__smcard_clkcon_bits);

/***************************************************************************
 **
 **  GPS
 **
 ***************************************************************************/
__IO_REG16_BIT(GPS_PRNC0,   0x80000000,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC1,   0x80000004,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC2,   0x80000008,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC3,   0x8000000C,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC4,   0x80000010,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC5,   0x80000014,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC6,   0x80000018,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC7,   0x8000001C,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC8,   0x80000020,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC9,   0x80000024,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC10,  0x80000028,__WRITE		 ,__gps_prnc_bits);
__IO_REG16_BIT(GPS_PRNC11,  0x8000002C,__WRITE		 ,__gps_prnc_bits);
__IO_REG32_BIT(GPS_PRNP0,   0x80000040,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP1,   0x80000044,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP2,   0x80000048,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP3,   0x8000004C,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP4,   0x80000050,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP5,   0x80000054,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP6,   0x80000058,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP7,   0x8000005C,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP8,   0x80000060,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP9,   0x80000064,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP10,  0x80000068,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_PRNP11,  0x8000006C,__WRITE		 ,__gps_prnp_bits);
__IO_REG32_BIT(GPS_NCOF0,   0x80000080,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF1,   0x80000084,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF2,   0x80000088,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF3,   0x8000008C,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF4,   0x80000090,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF5,   0x80000094,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF6,   0x80000098,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF7,   0x8000009C,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF8,   0x800000A0,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF9,   0x800000A4,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF10,  0x800000A8,__WRITE		 ,__gps_ncof_bits);
__IO_REG32_BIT(GPS_NCOF11,  0x800000AC,__WRITE		 ,__gps_ncof_bits);
__IO_REG16_BIT(GPS_NCOP0,   0x800000C0,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP1,   0x800000C4,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP2,   0x800000C8,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP3,   0x800000CC,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP4,   0x800000D0,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP5,   0x800000D4,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP6,   0x800000D8,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP7,   0x800000DC,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP8,   0x800000E0,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP9,   0x800000E4,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP10,  0x800000E8,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_NCOP11,  0x800000EC,__WRITE		 ,__gps_ncop_bits);
__IO_REG16_BIT(GPS_PRNGI0,  0x80000100,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI1,  0x80000104,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI2,  0x80000108,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI3,  0x8000010C,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI4,  0x80000110,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI5,  0x80000114,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI6,  0x80000118,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI7,  0x8000011C,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI8,  0x80000120,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI9,  0x80000124,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI10, 0x80000128,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_PRNGI11, 0x8000012C,__WRITE		 ,__gps_prngi_bits);
__IO_REG16_BIT(GPS_CTL,  		0x80000140,__READ_WRITE,__gps_ctl_bits);
__IO_REG16(	   GPS_TBH,  		0x80000144,__READ			 );
__IO_REG16_BIT(GPS_TBL,  		0x80000148,__READ			 ,__gps_tbl_bits);
__IO_REG16(	   GPS_CPSH,  	0x8000014C,__WRITE		 );
__IO_REG16_BIT(GPS_CPSL, 		0x80000150,__WRITE		 ,__gps_cpsl_bits);
__IO_REG16(	   GPS_CPRH,  	0x80000154,__WRITE		 );
__IO_REG16_BIT(GPS_CPRL, 		0x80000158,__WRITE		 ,__gps_cprl_bits);
__IO_REG16(	   GPS_TBBH,  	0x8000015C,__READ 		 );
__IO_REG16_BIT(GPS_TBBL, 		0x80000160,__READ 		 ,__gps_tbbl_bits);
__IO_REG16_BIT(GPS_PRND, 		0x80000164,__WRITE		 ,__gps_prnd_bits);
__IO_REG16_BIT(GPS_IS, 			0x80000168,__READ_WRITE,__gps_is_bits);
__IO_REG16_BIT(GPS_IM, 			0x8000016C,__READ_WRITE,__gps_im_bits);
__IO_REG16(		 GPS_NCOAP0,  0x80000180,__READ		 	 );
__IO_REG16(		 GPS_NCOAP1,  0x80000184,__READ		 	 );
__IO_REG16(		 GPS_NCOAP2,  0x80000188,__READ		 	 );
__IO_REG16(		 GPS_NCOAP3,  0x8000018C,__READ		 	 );
__IO_REG16(		 GPS_NCOAP4,  0x80000190,__READ		 	 );
__IO_REG16(		 GPS_NCOAP5,  0x80000194,__READ		 	 );
__IO_REG16(		 GPS_NCOAP6,  0x80000198,__READ		 	 );
__IO_REG16(		 GPS_NCOAP7,  0x8000019C,__READ		 	 );
__IO_REG16(		 GPS_NCOAP8,  0x800001A0,__READ		 	 );
__IO_REG16(		 GPS_NCOAP9,  0x800001A4,__READ		 	 );
__IO_REG16(		 GPS_NCOAP10, 0x800001A8,__READ		 	 );
__IO_REG16(		 GPS_NCOAP11, 0x800001AC,__READ		 	 );
__IO_REG16(		 GPS_NCOAPT,  0x800001C0,__READ		 	 );
__IO_REG16_BIT(GPS_SM, 			0x800001C4,__READ_WRITE,__gps_sm_bits);
__IO_REG16(		 GPS_IDATA0,  0x80000200,__READ		 	 );
__IO_REG16(		 GPS_IDATA1,  0x80000204,__READ		 	 );
__IO_REG16(		 GPS_IDATA2,  0x80000208,__READ		 	 );
__IO_REG16(		 GPS_IDATA3,  0x8000020C,__READ		 	 );
__IO_REG16(		 GPS_IDATA4,  0x80000210,__READ		 	 );
__IO_REG16(		 GPS_IDATA5,  0x80000214,__READ		 	 );
__IO_REG16(		 GPS_IDATA6,  0x80000218,__READ		 	 );
__IO_REG16(		 GPS_IDATA7,  0x8000021C,__READ		 	 );
__IO_REG16(		 GPS_IDATA8,  0x80000220,__READ		 	 );
__IO_REG16(		 GPS_IDATA9,  0x80000224,__READ		 	 );
__IO_REG16(		 GPS_IDATA10, 0x80000228,__READ		 	 );
__IO_REG16(		 GPS_IDATA11, 0x8000022C,__READ		 	 );
__IO_REG16(		 GPS_QDATA0,  0x80000240,__READ		 	 );
__IO_REG16(		 GPS_QDATA1,  0x80000244,__READ		 	 );
__IO_REG16(		 GPS_QDATA2,  0x80000248,__READ		 	 );
__IO_REG16(		 GPS_QDATA3,  0x8000024C,__READ		 	 );
__IO_REG16(		 GPS_QDATA4,  0x80000250,__READ		 	 );
__IO_REG16(		 GPS_QDATA5,  0x80000254,__READ		 	 );
__IO_REG16(		 GPS_QDATA6,  0x80000258,__READ		 	 );
__IO_REG16(		 GPS_QDATA7,  0x8000025C,__READ		 	 );
__IO_REG16(		 GPS_QDATA8,  0x80000260,__READ		 	 );
__IO_REG16(		 GPS_QDATA9,  0x80000264,__READ		 	 );
__IO_REG16(		 GPS_QDATA10, 0x80000268,__READ		 	 );
__IO_REG16(		 GPS_QDATA11, 0x8000026C,__READ		 	 );
__IO_REG16_BIT(TIME_XY01, 	0x80000280,__READ      ,__gps_time_xy_bits);
__IO_REG16_BIT(TIME_XY23, 	0x80000284,__READ      ,__gps_time_xy_bits);
__IO_REG16_BIT(TIME_XY45, 	0x80000288,__READ      ,__gps_time_xy_bits);
__IO_REG16_BIT(TIME_XY67, 	0x8000028C,__READ      ,__gps_time_xy_bits);
__IO_REG16_BIT(TIME_XY89, 	0x80000290,__READ      ,__gps_time_xy_bits);
__IO_REG16_BIT(TIME_XY1011, 0x80000294,__READ      ,__gps_time_xy_bits);

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
 **  IRQ Interrupt Vector Table
 **
 ***************************************************************************/
#define TIMERO_INT          0
/*#define _INT           1*/
#define PRCCU_INT           2
#define RTC_INT             3
#define WDG_INT             4
#define WUIMU_INT           5
/*#define _INT          6*/
/*#define _INT        	7*/
#define I2CTERR_INT        	8
#define UART0_INT           9
#define UART1_INT          10
/*#define _INT          11*/
/*#define _INT          12*/
/*#define _INT          13*/
#define SPI_INT            14
/*#define _INT          15*/
#define I2C_INT            16
#define CAN_INT            17
#define ADC_INT            18
#define T1_INT             19
#define T2_INT             20
#define T3_INT             21
/*#define _INT          22*/
#define FREQ_LOW_INT       23
#define FREQ_HI_INT        24
/*#define _INT          25*/
/*#define _INT          26*/
#define GPS_INT            27
/*#define _INT          28*/
#define T0OVER_INT         29
#define T0COMP1_INT        30
#define TOCOMP2_INT        31

/***************************************************************************
 **
 **  FIQ Interrupt Vector Table
 **
 ***************************************************************************/
#define T0GLB_INT           0
#define WDG_FIQ_INT         1

#endif    /* __IOSTA2056_H */
