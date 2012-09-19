/***************************************************************************
 **                        
 **    This file defines the Special Function Registers for
 **    st STR712
 **    
 **    Used with ICCARM and AARM.
 **                                
 **    (c) Copyright IAR Systems 2004
 **                                
 **    $Revision: 20586 $
 **                                
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOSTR712_H
#define __IOSTR712_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x58 = 88 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **                            
 **   IOSTR712 SPECIAL FUNCTION REGISTERS
 **                            
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

  
/* C specific declarations  ************************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
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

/* Flash Control Register 0 (FLASH_CR0)*/
typedef struct
{ 
  __REG32                 : 1;
  __REG32  BSYA0          : 1;
  __REG32  BSYA1          : 1;
  __REG32                 : 1;
  __REG32  LOCK           : 1;
  __REG32                 :10;
  __REG32  PWD            : 1;
  __REG32                 : 4;
  __REG32  INTP           : 1;
  __REG32  INTM           : 1;
  __REG32                 : 2;
  __REG32  SPR            : 1;
  __REG32                 : 2;
  __REG32  SER            : 1;
  __REG32  DWPG           : 1;
  __REG32  WPG            : 1;
  __REG32  SUSP           : 1;
  __REG32  WMS            : 1;
} __flashctrl0_bits;

/* Flash Control Register 1 (FLASH_CR1)*/
typedef struct
{ 
  __REG32  B0F0           : 1;
  __REG32  B0F1           : 1;
  __REG32  B0F2           : 1;
  __REG32  B0F3           : 1;
  __REG32  B0F4           : 1;
  __REG32  B0F5           : 1;
  __REG32  B0F6           : 1;
  __REG32  B0F7           : 1;
  __REG32                 : 8;
  __REG32  B1F0           : 1;
  __REG32  B1F1           : 1;
  __REG32                 : 6;
  __REG32  B0S            : 1;
  __REG32  B1S            : 1;
  __REG32                 : 6;
} __flashctrl1_bits;

/* Flash Address Register (FLASH_AR) */
typedef struct
{ 
  __REG32  ADD            :21;
  __REG32                 :11;
} __flashadd_bits;

/* Flash Error Register (FLASH_ER) */
typedef struct
{ 
  __REG32  ERR            : 1;
  __REG32  ERER           : 1;
  __REG32  PGER           : 1;
  __REG32  _10ER          : 1;
  __REG32                 : 2;
  __REG32  SEQER          : 1;
  __REG32  RESER          : 1;
  __REG32  WPF            : 1;
  __REG32                 :23;
} __flasherr_bits;

/* Flash Non Volatile Write Protection Register (FLASH_NVWPAR) */
typedef struct
{ 
  __REG32  W0P            : 8;
  __REG32                 : 8;
  __REG32  W1P            : 2;
  __REG32                 :14;
} __flashwrprot_bits;

/* Flash NV Access Protection Register 0 (FLASH_NVAPR0) */
typedef struct
{ 
  __REG32                 : 1;
  __REG32  DBGP           : 1;
  __REG32                 :30;
} __flashaccprot0_bits;

/* Flash NV Access Protection Register 1 (FLASH_NVAPR1) */
typedef struct
{ 
  __REG32  PDS            :16;
  __REG32  PEN            :16;
} __flashaccprot1_bits;

/* Clock Control Register (RCCU_CCR) */
typedef struct {
  __REG16  LOP_WFI        :1;
  __REG16  WFI_CKSEL      :1;
  __REG16  CKAF_SEL       :1;
  __REG16  SRESEN         :1;
  __REG16                 :3;
  __REG16  EN_LOCK        :1;
  __REG16  EN_CKAF        :1;
  __REG16  EN_CK2_16      :1;
  __REG16  EN_STOP        :1;
  __REG16  EN_HALT        :1;
  __REG16                 :4;
} __rccu_ccr_bits;

/* Clock Flag Register (RCCU_CFR) */
typedef struct {
  __REG16  CSU_CKSEL      :1;
  __REG16  LOCK           :1;
  __REG16  CKAF_ST        :1;
  __REG16  CK2_16         :1;
  __REG16                 :1;
  __REG16  SOFTRES        :1;
  __REG16  WDGRES         :1;
  __REG16  RTC_ALARM      :1;
  __REG16                 :1;
  __REG16  LVD_RES        :1;
  __REG16  WKP_RES        :1;
  __REG16  LOCK_I         :1;
  __REG16  CKAF_I         :1;
  __REG16  CK2_16_I       :1;
  __REG16  STOP_I         :1;
  __REG16  DIV2           :1;
} __rccu_cfr_bits;

/* PLL Configuration Register (RCCU_PLL1CR) */
typedef struct {
  __REG16  DX0            :1;
  __REG16  DX1            :1;
  __REG16  DX2            :1;
  __REG16                 :1;
  __REG16  MX0            :1;
  __REG16  MX1            :1;
  __REG16  FREF_RANGE     :1;
  __REG16  FREEN          :1;
  __REG16                 :8;
} __rccu_pll1cr_bits;

/* Peripheral Enable Register (RCCU_PER) */
typedef struct {
  __REG16  PH_CK0         :1;
  __REG16  PH_CK1         :1;
  __REG16  PH_CK2         :1;             /* EMI        */
  __REG16  PH_CK3         :1;
  __REG16                 :12;
} __rccu_per_bits;

/* System Mode Register (RCCU_SMR) */
typedef struct {
  __REG16  WFI            : 1;
  __REG16  HALT           : 1;
  __REG16                 :14;
} __rccu_sysmode_bits;

/* MCLK Divider Control (PCU_MDIVR) */
typedef struct {
  __REG16  FACT           : 2;
  __REG16                 :14;
} __pcu_mdivr_bits;

/* Peripheral Clock Divider Control Register (PCU_PDIVR) */
typedef struct {
  __REG16  FACT1          : 2;
  __REG16                 : 6;
  __REG16  FACT2          : 2;
  __REG16                 : 6;
} __pcu_pdivr_bits;

/* Peripheral Reset Control Register (PCU_PRSTR) */
typedef struct {
  __REG16                 : 2;
  __REG16  EMIRST         : 1;
  __REG16                 :13;
} __pcu_prstr_bits;

/* PLL2 Control Register (PCU_PLL2CR) */
typedef struct {
  __REG16  DX             : 3;
  __REG16                 : 1;
  __REG16  MX             : 2;
  __REG16  FRQRNG         : 1;
  __REG16  PLLEN          : 1;
  __REG16                 : 1;
  __REG16  IRQMASK        : 1;
  __REG16  IRQPEND        : 1;
  __REG16                 : 4;
  __REG16  LOCK           : 1;
} __pcu_pll2cr_bits;

/* Boot Configuration Register PCU_BOOTCR) */
typedef struct {
  __REG16  BOOT           : 2;
  __REG16  SPIOEN         : 1;
  __REG16                 : 1;
  __REG16  LPOWDBGEN      : 1;
  __REG16  ADCEN          : 1;
  __REG16  CAN            : 1;
  __REG16  HDLC           : 1;
  __REG16                 : 1;
  __REG16  PKG64          : 1;
  __REG16                 : 6;
} __pcu_bootcr_bits;

/* Power Control Register (PCU_PWRCR) */
typedef struct {
  __REG16                 : 3;
  __REG16  VRBYP          : 1;
  __REG16  LPVRWFI        : 1;
  __REG16  LPVRBYP        : 1;
  __REG16  PWRDWN         : 1;
  __REG16  OSCBYP         : 1;
  __REG16  LVDDIS         : 1;
  __REG16  FLASHLP        : 1;
  __REG16                 : 2;
  __REG16  VROK           : 1;
  __REG16  WKUPALRM       : 1;
  __REG16  BUSY           : 1;
  __REG16  WREN           : 1;
} __pcu_pwrcr_bits;

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

/* Interrupt Vector Register (EIC_IVR) */
typedef struct {
  __REG32 JUMPOPCODE      :16;
  __REG32 JUMADDRESS      :16;
} __eic_ivr_bits;

/* Fast Interrupt Register (EIC_FIR) */
typedef struct {
  __REG32 FIE             : 2;
  __REG32 FIP             : 2;
  __REG32                 :28;
} __eic_fir_bits;

/* Interrupt Enable Register 0 (EIC_IER0) */
typedef struct {
  __REG32 IE0_0           : 1;
  __REG32 IE0_1           : 1;
  __REG32 IE0_2           : 1;
  __REG32 IE0_3           : 1;
  __REG32 IE0_4           : 1;
  __REG32 IE0_5           : 1;
  __REG32 IE0_6           : 1;
  __REG32 IE0_7           : 1;
  __REG32 IE0_8           : 1;
  __REG32 IE0_9           : 1;
  __REG32 IE0_10          : 1;
  __REG32 IE0_11          : 1;
  __REG32 IE0_12          : 1;
  __REG32 IE0_13          : 1;
  __REG32 IE0_14          : 1;
  __REG32 IE0_15          : 1;
  __REG32 IE0_16          : 1;
  __REG32 IE0_17          : 1;
  __REG32 IE0_18          : 1;
  __REG32 IE0_19          : 1;
  __REG32 IE0_20          : 1;
  __REG32 IE0_21          : 1;
  __REG32 IE0_22          : 1;
  __REG32 IE0_23          : 1;
  __REG32 IE0_24          : 1;
  __REG32 IE0_25          : 1;
  __REG32 IE0_26          : 1;
  __REG32 IE0_27          : 1;
  __REG32 IE0_28          : 1;
  __REG32 IE0_29          : 1;
  __REG32 IE0_30          : 1;
  __REG32 IE0_31          : 1;
} __eic_ier_bits;

/* Interrupt Pending Register 0 (EIC_IPR0) */
typedef struct {
  __REG32 IP0_0           : 1;
  __REG32 IP0_1           : 1;
  __REG32 IP0_2           : 1;
  __REG32 IP0_3           : 1;
  __REG32 IP0_4           : 1;
  __REG32 IP0_5           : 1;
  __REG32 IP0_6           : 1;
  __REG32 IP0_7           : 1;
  __REG32 IP0_8           : 1;
  __REG32 IP0_9           : 1;
  __REG32 IP0_10          : 1;
  __REG32 IP0_11          : 1;
  __REG32 IP0_12          : 1;
  __REG32 IP0_13          : 1;
  __REG32 IP0_14          : 1;
  __REG32 IP0_15          : 1;
  __REG32 IP0_16          : 1;
  __REG32 IP0_17          : 1;
  __REG32 IP0_18          : 1;
  __REG32 IP0_19          : 1;
  __REG32 IP0_20          : 1;
  __REG32 IP0_21          : 1;
  __REG32 IP0_22          : 1;
  __REG32 IP0_23          : 1;
  __REG32 IP0_24          : 1;
  __REG32 IP0_25          : 1;
  __REG32 IP0_26          : 1;
  __REG32 IP0_27          : 1;
  __REG32 IP0_28          : 1;
  __REG32 IP0_29          : 1;
  __REG32 IP0_30          : 1;
  __REG32 IP0_31          : 1;
} __eic_ipr_bits;

/* Source Interrupt Registers - Channel n (EIC_SIRn) */
typedef struct {
  __REG32 SIPL            : 4;
  __REG32                 :12;
  __REG32 SIV             :16;
} __eic_sirn_bits;

/* External Interrupts (XTI) */
typedef struct {
  __REG8 WKUP_INT         : 1;
  __REG8 ID1S             : 1;
  __REG8 STOP             : 1;
  __REG8                  : 5;
} __xti_ctrl_bits;

/* RTC Control Register High (RTC_CRH) */
typedef struct {
  __REG16 SEN             : 1;
  __REG16 AEN             : 1;
  __REG16 OWEN            : 1;
  __REG16 GEN             : 1;
  __REG16                 :12;
} __rtc_crh_bits;

/* RTC Control Register Low (RTC_CRL) */
typedef struct {
  __REG16 SIR             : 1;
  __REG16 AIR             : 1;
  __REG16 OWIR            : 1;
  __REG16 GIR             : 1;
  __REG16 CNF             : 1;
  __REG16 RTOFF           : 1;
  __REG16                 :10;
} __rtc_crl_bits;

/* RTC Prescaler Load Register High (RTC_PRLH) */
typedef struct {
  __REG16 PRSL            : 4;
  __REG16                 :12;
} __rtc_prlh_bits;

/* RTC Prescaler Divider Register High (RTC_DIVH) */
typedef struct {
  __REG16 RTCDIV          : 4;
  __REG16                 :12;
} __rtc_divh_bits;

/* WDG Control Register (WDG_CR) */
typedef struct {
  __REG16 WE              : 1;
  __REG16 SC              : 1;
  __REG16                 :14;
} __wdg_cr_bits;

/* WDG Prescaler Register (WDG_PR) */
typedef struct {
  __REG16 PR              : 8;
  __REG16                 : 8;
} __wdg_pr_bits;

/* WDG Status Register (WDG_SR) */
typedef struct {
  __REG16 EC              : 1;
  __REG16                 :15;
} __wdg_sr_bits;

/* WDG Mask Register (WDG_MR) */
typedef struct {
  __REG16 ECM             : 1;
  __REG16                 :15;
} __wdg_mr_bits;

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
  __REG16 FOLVA          : 1;
  __REG16 FOLVB          : 1;
  __REG16                 : 2;
  __REG16 PWMI            : 1;
  __REG16 EN              : 1;
} __tim_cr1_bits;

/* Control Register 2 (TIMn_CR2) */
typedef struct {
  __REG16 CC0             : 1;
  __REG16 CC1             : 1;
  __REG16 CC2             : 1;
  __REG16 CC3             : 1;
  __REG16 CC4             : 1;
  __REG16 CC5             : 1;
  __REG16 CC6             : 1;
  __REG16 CC7             : 1;
  __REG16                 : 3;
  __REG16 OCBIE           : 1;
  __REG16 ICBIE           : 1;
  __REG16 TOE             : 1;
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
  __REG16  TX0            : 1;
  __REG16  TX1            : 1;
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
  __REG8  FM_SM              : 1;
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
  __REG16  RX             :10;
  __REG16                 : 6;
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

/* Private Address Register High (HDLC_PARH) */
typedef struct {
  __REG16  PAB2           : 8;
  __REG16  PAB3           : 8;
} __hdcl_parh_bits;

/* Private Address Register Low (HDLC_PARL) */
typedef struct {
  __REG16  PAB0           : 8;
  __REG16  PAB1           : 8;
} __hdcl_parl_bits;

/* Private Address Mask Register High (HDLC_PAMH) */
typedef struct {
  __REG16  PAMB2          : 8;
  __REG16  PAMB3          : 8;
} __hdcl_pamh_bits;

/* Private Address Mask Register Low (HDLC_PAML) */
typedef struct {
  __REG16  PAMB0          : 8;
  __REG16  PAMB1          : 8;
} __hdcl_paml_bits;

/* Group Address Register 1 (HDLC_GA1) */
typedef struct {
  __REG16  GA2            : 8;
  __REG16  BA             : 8;
} __hdcl_ga1_bits;

/* Group Address Register 0 (HDLC_GA0) */
typedef struct {
  __REG16  GA0            : 8;
  __REG16  GA1            : 8;
} __hdcl_ga0_bits;

/* Group Address Mask Register 1 (HDLC_GAM1) */
typedef struct {
  __REG16  GAM2           : 8;
  __REG16  BAM            : 8;
} __hdcl_gam1_bits;

/* Group Address Mask Register 0 (HDLC_GAM0) */
typedef struct {
  __REG16  GAM0           : 8;
  __REG16  GAM1           : 8;
} __hdcl_gam0_bits;

/* Transmission Control Register (HDLC_TCTL) */
typedef struct {
  __REG16  NPOSB          : 4;
  __REG16  NPREB          : 4;
  __REG16  POSE           : 1;
  __REG16  PREE           : 1;
  __REG16  TCOD           : 2;
  __REG16  TCRCI          : 1;
  __REG16  ITF            : 1;
  __REG16  SOC            : 1;
  __REG16  HTEN           : 1;
} __hdcl_tctl_bits;

/* Receive Control Register (HDLC_RCTL) */
typedef struct {
  __REG16  RCOD           : 2;
  __REG16  RCRCI          : 1;
  __REG16  SIC            : 1;
  __REG16  LBEN           : 1;
  __REG16  AEN            : 1;
  __REG16  GA0E           : 1;
  __REG16  GA1E           : 1;
  __REG16  GA2E           : 1;
  __REG16  BAE            : 1;
  __REG16  PAE            : 1;
  __REG16  DPLLE          : 1;
  __REG16  RMCE           : 1;
  __REG16                 : 3;
} __hdlc_rctl_bits;

/* Baud Rate Register (HDLC_BRR) */
typedef struct {
  __REG16  BRG            :12;
  __REG16  TCKS           : 1;
  __REG16                 : 3;
} __hdlc_brr_bits;

/* Prescaler Register (HDLC_PRSR) */
typedef struct {
  __REG16  PSR            : 8;
  __REG16  RCKS           : 2;
  __REG16                 : 6;
} __hdlc_prsr_bits;

/* Peripheral Status Register (HDLC_PSR) */
typedef struct {
  __REG16  RLS            : 2;
  __REG16  TBR            : 1;
  __REG16  RBR            : 1;
  __REG16                 :12;
} __hdlc_psr_bits;

/* Frame Status Byte Register (HDLC_FSBR) */
typedef struct {
  __REG16                 : 4;
  __REG16  RAB            : 1;
  __REG16  CRC            : 1;
  __REG16  RDO            : 1;
  __REG16  RBC            : 1;
  __REG16                 : 8;
} __hdlc_fsbr_bits;

/* Peripheral Command Register (HDLC_PCR) */
typedef struct {
  __REG16  REN            : 1;
  __REG16  TEN            : 1;
  __REG16                 :14;
} __hdlc_pcr_bits;

/* Interrupt Status Register (HDLC_ISR) */
typedef struct {
  __REG16  RME            : 1;
  __REG16  RMC            : 1;
  __REG16  RFO            : 1;
  __REG16  RBF            : 1;
  __REG16  TDU            : 1;
  __REG16  TBE            : 1;
  __REG16  TMC            : 1;
  __REG16                 : 9;
} __hdlc_isr_bits;

/* Interrupt Mask Register (HDLC_IMR) */
typedef struct {
  __REG16  RMEM           : 1;
  __REG16  RMCM           : 1;
  __REG16  RFOM           : 1;
  __REG16  RBFM           : 1;
  __REG16  TDUM           : 1;
  __REG16  TBEM           : 1;
  __REG16  TMCM           : 1;
  __REG16                 : 9;
} __hdlc_imr_bits;

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
  __REG16 PRESC           :12;
  __REG16                 : 4;
} __adc_cpr_bits;

/* ADC Data Register n, n = 0..3 (ADC_DATA[n]) */
typedef struct {
  __REG16                 : 4;
  __REG16 DATA            :12;
} __adc_dr_bits;


/* APB Clock Disable Register (APBn_CKDIS) */
typedef struct {
  __REG32 PCKDIS          :15;
  __REG32                 :17;
} __apb_ckdisr_bits;


/* APB1 Peripheral bits */
typedef struct {
  __REG32 I2C0            : 1;
  __REG32 I2C1            : 1;
  __REG32                 : 1;
  __REG32 UART0           : 1;
  __REG32 UART1PLUSCARD   : 1;
  __REG32 UART2           : 1;
  __REG32 UART3           : 1;
  __REG32                 : 1;
  __REG32 CAN             : 1;
  __REG32 BSPI0           : 1;
  __REG32 BSPI1           : 1;
  __REG32                 : 2;
  __REG32 HDLC            : 1;
  __REG32                 :18;
} __apb1_peri_bits;

/* APB2 Peripheral bits */
typedef struct {
  __REG32 XTI             :1;
  __REG32                 :1;
  __REG32 IOPORT0         :1;
  __REG32 IOPORT1         :1;
  __REG32                 :1;
  __REG32                 :1;
  __REG32 ADC             :1;
  __REG32 CKOUT           :1;
  __REG32 TIMER0          :1;
  __REG32 TIMER1          :1;
  __REG32 TIMER2          :1;
  __REG32 TIMER3          :1;
  __REG32 RTC             :1;
  __REG32 WDG             :1;
  __REG32 EIC             :1;
  __REG32                 :17;
} __apb2_peri_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/***************************************************************************/
/* Common declarations  ****************************************************/

/***************************************************************************
 **
 **  FLASH
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASH_CR0,   0x40100000,__READ_WRITE,__flashctrl0_bits);
__IO_REG32_BIT(FLASH_CR1,   0x40100004,__READ_WRITE,__flashctrl1_bits);
__IO_REG32(    FLASH_DR0,   0x40100008,__READ_WRITE);
__IO_REG32(    FLASH_DR1,   0x4010000C,__READ_WRITE);
__IO_REG32_BIT(FLASH_AR,    0x40100010,__READ_WRITE,__flashadd_bits);
__IO_REG32_BIT(FLASH_ER,    0x40100014,__READ_WRITE,__flasherr_bits);
__IO_REG32_BIT(FLASH_NVWPAR,0x4010DFB0,__READ_WRITE,__flashwrprot_bits);
__IO_REG32_BIT(FLASH_NVAPR0,0x4010DFB8,__READ_WRITE,__flashaccprot0_bits);
__IO_REG32_BIT(FLASH_NVAPR1,0x4010DFBC,__READ_WRITE,__flashaccprot1_bits);

/***************************************************************************
 **
 **  PRCCU
 **
 ***************************************************************************/
__IO_REG16_BIT(RCCU_CCR,    0xA0000000,__READ_WRITE,__rccu_ccr_bits);
__IO_REG16_BIT(RCCU_CFR,    0xA0000008,__READ_WRITE,__rccu_cfr_bits);
__IO_REG16_BIT(RCCU_PLL1CR, 0xA0000018,__READ_WRITE,__rccu_pll1cr_bits);
__IO_REG16_BIT(RCCU_PER,    0xA000001C,__READ_WRITE,__rccu_per_bits);
__IO_REG16_BIT(RCCU_SMR,    0xA0000020,__READ_WRITE,__rccu_sysmode_bits);
__IO_REG16_BIT(RCCU_MDIVR,  0xA0000040,__READ_WRITE,__pcu_mdivr_bits);
__IO_REG16_BIT(RCCU_PDIVR,  0xA0000044,__READ_WRITE,__pcu_pdivr_bits);
__IO_REG16_BIT(PCU_PRSTR,   0xA0000048,__READ_WRITE,__pcu_prstr_bits);
__IO_REG16_BIT(PCU_PLL2CR,  0xA000004C,__READ_WRITE,__pcu_pll2cr_bits);
__IO_REG16_BIT(PCU_BOOTCR,  0xA0000050,__READ_WRITE,__pcu_bootcr_bits);
__IO_REG16_BIT(PCU_PWRCR,   0xA0000054,__READ_WRITE,__pcu_pwrcr_bits);

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
 ** EIC
 **
 ***************************************************************************/
__IO_REG32_BIT(EIC_ICR,     0xFFFFF800,__READ_WRITE,__eic_icr_bits);
__IO_REG32_BIT(EIC_CICR,    0xFFFFF804,__READ,      __eic_cicr_bits);
__IO_REG32_BIT(EIC_CIPR,    0xFFFFF808,__READ_WRITE,__eic_cipr_bits);
__IO_REG32_BIT(EIC_IVR,     0xFFFFF818,__READ_WRITE,__eic_ivr_bits);
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
 ** External Interrupts (XTI)
 **
 ***************************************************************************/
__IO_REG8_BIT(XTI_SR,       0xE000101C,__READ_WRITE,__BITS8);
__IO_REG8_BIT(XTI_CTRL,     0xE0001024,__READ_WRITE,__xti_ctrl_bits);
__IO_REG8_BIT(XTI_MRH,      0xE0001028,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(XTI_MRL,      0xE000102C,__READ_WRITE,__BITS8);
__IO_REG8_BIT(XTI_TRH,      0xE0001030,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(XTI_TRL,      0xE0001034,__READ_WRITE,__BITS8);
__IO_REG8_BIT(XTI_PRH,      0xE0001038,__READ_WRITE,__BITS8H_bits);
__IO_REG8_BIT(XTI_PRL,      0xE000103C,__READ_WRITE,__BITS8);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG16_BIT(RTC_CRH,     0xE000D000,__READ_WRITE,__rtc_crh_bits);
__IO_REG16_BIT(RTC_CRL,     0xE000D004,__READ_WRITE,__rtc_crl_bits);
__IO_REG16_BIT(RTC_PRLH,    0xE000D008,__WRITE     ,__rtc_prlh_bits);
__IO_REG16(    RTC_PRLL,    0xE000D00C,__WRITE);
__IO_REG16_BIT(RTC_DIVH,    0xE000D010,__READ      ,__rtc_divh_bits);
__IO_REG16(    RTC_DIVL,    0xE000D014,__READ);
__IO_REG16(    RTC_CNTH,    0xE000D018,__READ_WRITE);
__IO_REG16(    RTC_CNTL,    0xE000D01C,__READ_WRITE);
__IO_REG16(    RTC_ALRH,    0xE000D020,__WRITE);
__IO_REG16(    RTC_ALRL,    0xE000D024,__WRITE);

/***************************************************************************
 **
 ** WDG
 **
 ***************************************************************************/
__IO_REG16_BIT(WDG_CR,      0xE000E000,__READ_WRITE,__wdg_cr_bits);
__IO_REG16_BIT(WDG_PR,      0xE000E004,__READ_WRITE,__wdg_pr_bits);
__IO_REG16(    WDG_VR,      0xE000E008,__READ_WRITE);
__IO_REG16(    WDG_CNT,     0xE000E00C,__READ);
__IO_REG16_BIT(WDG_SR,      0xE000E010,__READ_WRITE,__wdg_sr_bits);
__IO_REG16_BIT(WDG_MR,      0xE000E014,__READ_WRITE,__wdg_mr_bits);
__IO_REG16(    WDG_KR,      0xE000E018,__READ_WRITE);

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
 ** I2C0
 **
 ***************************************************************************/
__IO_REG8_BIT(I2C0_CR,      0xC0001000,__READ_WRITE,__i2cn_cr_bits);
__IO_REG8_BIT(I2C0_SR1,     0xC0001004,__READ      ,__i2cn_sr1_bits);
__IO_REG8_BIT(I2C0_SR2,     0xC0001008,__READ      ,__i2cn_sr2_bits);
__IO_REG8_BIT(I2C0_CCR,     0xC000100C,__READ_WRITE,__i2cn_ccr_bits);
__IO_REG8_BIT(I2C0_OAR1,    0xC0001010,__READ_WRITE,__i2cn_oar1_bits);
__IO_REG8_BIT(I2C0_OAR2,    0xC0001014,__READ_WRITE,__i2cn_oar2_bits);
__IO_REG8(    I2C0_DR,      0xC0001018,__READ_WRITE);
__IO_REG8_BIT(I2C0_ECCR,    0xC000101C,__READ_WRITE,__i2cn_eccr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG8_BIT(I2C1_CR,      0xC0002000,__READ_WRITE,__i2cn_cr_bits);
__IO_REG8_BIT(I2C1_SR1,     0xC0002004,__READ      ,__i2cn_sr1_bits);
__IO_REG8_BIT(I2C1_SR2,     0xC0002008,__READ      ,__i2cn_sr2_bits);
__IO_REG8_BIT(I2C1_CCR,     0xC000200C,__READ_WRITE,__i2cn_ccr_bits);
__IO_REG8_BIT(I2C1_OAR1,    0xC0002010,__READ_WRITE,__i2cn_oar1_bits);
__IO_REG8_BIT(I2C1_OAR2,    0xC0002014,__READ_WRITE,__i2cn_oar2_bits);
__IO_REG8(    I2C1_DR,      0xC0002018,__READ_WRITE);
__IO_REG8_BIT(I2C1_ECCR,    0xC000201C,__READ_WRITE,__i2cn_eccr_bits);

/***************************************************************************
 **
 **  BSPI0
 **
 ***************************************************************************/
__IO_REG16_BIT(BSPI0_RXR,   0xC000A000,__READ      ,__bspin_rxr_bits);
__IO_REG16_BIT(BSPI0_TXR,   0xC000A004,__WRITE     ,__bspin_tx_bits);
__IO_REG16_BIT(BSPI0_CSR1,  0xC000A008,__READ_WRITE,__bspin_csr1_bits);
__IO_REG16_BIT(BSPI0_CSR2,  0xC000A00C,__READ_WRITE,__bspin_csr2_bits);
__IO_REG8(     BSPI0_CLK,   0xC000A010,__READ_WRITE);

/***************************************************************************
 **
 **  BSPI1
 **
 ***************************************************************************/
__IO_REG16_BIT(BSPI1_RXR,   0xC000B000,__READ      ,__bspin_rxr_bits);
__IO_REG16_BIT(BSPI1_TXR,   0xC000B004,__WRITE     ,__bspin_tx_bits);
__IO_REG16_BIT(BSPI1_CSR1,  0xC000B008,__READ_WRITE,__bspin_csr1_bits);
__IO_REG16_BIT(BSPI1_CSR2,  0xC000B00C,__READ_WRITE,__bspin_csr2_bits);
__IO_REG8(     BSPI1_CLK,   0xC000B010,__READ_WRITE);

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
 **  UART2
 **
 ***************************************************************************/
__IO_REG16(    UART2_BR,    0xC0006000,__READ_WRITE);
__IO_REG16_BIT(UART2_TXBUFR,0xC0006004,__WRITE     ,__uartn_tx_bits);
__IO_REG16_BIT(UART2_RXBUFR,0xC0006008,__READ      ,__uartn_rx_bits);
__IO_REG16_BIT(UART2_CR,    0xC000600C,__READ_WRITE,__uartn_cr_bits);
__IO_REG16_BIT(UART2_IER,   0xC0006010,__READ_WRITE,__uartn_ier_bits);
__IO_REG16_BIT(UART2_SR,    0xC0006014,__READ      ,__uartn_sr_bits);
__IO_REG16_BIT(UART2_GTR,   0xC0006018,__READ_WRITE,__uartn_gtr_bits);
__IO_REG16_BIT(UART2_TOR,   0xC000601C,__READ_WRITE,__uartn_tor_bits);
__IO_REG16(    UART2_TXRSTR,0xC0006020,__WRITE);
__IO_REG16(    UART2_RXRSTR,0xC0006024,__WRITE);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG16(    UART3_BR,    0xC0007000,__READ_WRITE);
__IO_REG16_BIT(UART3_TXBUFR,0xC0007004,__WRITE     ,__uartn_tx_bits);
__IO_REG16_BIT(UART3_RXBUFR,0xC0007008,__READ      ,__uartn_rx_bits);
__IO_REG16_BIT(UART3_CR,    0xC000700C,__READ_WRITE,__uartn_cr_bits);
__IO_REG16_BIT(UART3_IER,   0xC0007010,__READ_WRITE,__uartn_ier_bits);
__IO_REG16_BIT(UART3_SR,    0xC0007014,__READ      ,__uartn_sr_bits);
__IO_REG16_BIT(UART3_GTR,   0xC0007018,__READ_WRITE,__uartn_gtr_bits);
__IO_REG16_BIT(UART3_TOR,   0xC000701C,__READ_WRITE,__uartn_tor_bits);
__IO_REG16(    UART3_TXRSTR,0xC0007020,__WRITE);
__IO_REG16(    UART3_RXRSTR,0xC0007024,__WRITE);

/***************************************************************************
 **
 **  HDLC
 **
 ***************************************************************************/
__IO_REG16_BIT(HDLC_PARH,   0xC000E000,__READ_WRITE,__hdcl_parh_bits);
__IO_REG16_BIT(HDLC_PARL,   0xC000E004,__READ_WRITE,__hdcl_parl_bits);
__IO_REG16_BIT(HDLC_PAMH,   0xC000E008,__READ_WRITE,__hdcl_pamh_bits);
__IO_REG16_BIT(HDLC_PAML,   0xC000E00C,__READ_WRITE,__hdcl_paml_bits);
__IO_REG16_BIT(HDLC_GA1,    0xC000E010,__READ_WRITE,__hdcl_ga1_bits);
__IO_REG16_BIT(HDLC_GA0,    0xC000E014,__READ_WRITE,__hdcl_ga0_bits);
__IO_REG16_BIT(HDLC_GAM1,   0xC000E018,__READ_WRITE,__hdcl_gam1_bits);
__IO_REG16_BIT(HDLC_GAM0,   0xC000E01C,__READ_WRITE,__hdcl_gam0_bits);
__IO_REG16(    HDLC_PRES,   0xC000E020,__READ_WRITE);
__IO_REG16(    HDLC_POSS,   0xC000E024,__READ_WRITE);
__IO_REG16_BIT(HDLC_TCTL,   0xC000E028,__READ_WRITE,__hdcl_tctl_bits);
__IO_REG16_BIT(HDLC_RCTL,   0xC000E02C,__READ_WRITE,__hdlc_rctl_bits);
__IO_REG16_BIT(HDLC_BRR,    0xC000E030,__READ_WRITE,__hdlc_brr_bits);
__IO_REG16_BIT(HDLC_PRSR,   0xC000E034,__READ_WRITE,__hdlc_prsr_bits);
__IO_REG16_BIT(HDLC_PSR,    0xC000E038,__READ,      __hdlc_psr_bits);
__IO_REG16_BIT(HDLC_FSBR,   0xC000E03C,__READ,      __hdlc_fsbr_bits);
__IO_REG16(    HDLC_TFBCR,  0xC000E040,__READ_WRITE);
__IO_REG16(    HDLC_RFBCR,  0xC000E044,__READ);
__IO_REG16_BIT(HDLC_PCR,    0xC000E048,__READ_WRITE,__hdlc_pcr_bits);
__IO_REG16_BIT(HDLC_ISR,    0xC000E04C,__READ_WRITE,__hdlc_isr_bits);
__IO_REG16_BIT(HDLC_IMR,    0xC000E050,__READ_WRITE,__hdlc_imr_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG16_BIT(ADC_DATA0,   0xE0007000,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADC_DATA1,   0xE0007008,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADC_DATA2,   0xE0007010,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADC_DATA3,   0xE0007018,__READ,      __adc_dr_bits);
__IO_REG16_BIT(ADC_CSR,     0xE0007020,__READ_WRITE,__adc_csr_bits);
__IO_REG16_BIT(ADC_CPR,     0xE0007030,__WRITE,     __adc_cpr_bits);

/***************************************************************************
 **
 ** APB1 Bridge Config 
 **
 ***************************************************************************/
__IO_REG32_BIT(APB1_CKDIS,  0xC0000010,__READ_WRITE,__apb_ckdisr_bits);
__IO_REG32_BIT(APB1_SWRES,  0xC0000014,__READ_WRITE,__apb1_peri_bits);

/***************************************************************************
 **
 ** APB2 Bridge Config 
 **
 ***************************************************************************/
__IO_REG32_BIT(APB2_CKDIS,  0xE0000010,__READ_WRITE,__apb_ckdisr_bits);
__IO_REG32_BIT(APB2_SWRES,  0xE0000014,__READ_WRITE,__apb2_peri_bits);


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
#define FLASH_INT           1  
#define PRCCU_INT           2  
#define RTC_INT             3  
#define WDG_INT             4  
#define XTI_INT             5  
#define I2C0TERR_INT        7  
#define I2C1TERR_INT        8  
#define UART0_INT           9  
#define UART1_INT          10  
#define UART2_INT          11  
#define UART3_INT          12  
#define SPI0_INT           13 
#define SPI1_INT           14 
#define I2C0_INT           15 
#define I2C1_INT           16 
#define CAN_INT            17 
#define ADC_INT            18 
#define T1_INT             19 
#define T2_INT             20 
#define T3_INT             21 
#define HDLC_INT           25 
#define USBLP_INT          26
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

#endif    /* __IOSTR712_H */
