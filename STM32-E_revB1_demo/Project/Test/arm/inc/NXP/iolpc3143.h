/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC3143
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 41683 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOLPC3143_H
#define __IOLPC3143_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC3143 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* NandIRQStatus register */
typedef struct{
__REG32 INT0S   : 1;
__REG32 INT1S   : 1;
__REG32         : 2;
__REG32 INT4S   : 1;
__REG32 INT5S   : 1;
__REG32 INT6S   : 1;
__REG32 INT7S   : 1;
__REG32 INT8S   : 1;
__REG32 INT9S   : 1;
__REG32 INT10S  : 1;
__REG32 INT11S  : 1;
__REG32 INT12S  : 1;
__REG32 INT13S  : 1;
__REG32 INT14S  : 1;
__REG32 INT15S  : 1;
__REG32 INT16S  : 1;
__REG32 INT17S  : 1;
__REG32 INT18S  : 1;
__REG32 INT19S  : 1;
__REG32 INT20S  : 1;
__REG32 INT21S  : 1;
__REG32 INT22S  : 1;
__REG32 INT23S  : 1;
__REG32 INT24S  : 1;
__REG32 INT25S  : 1;
__REG32 INT26S  : 1;
__REG32 INT27S  : 1;
__REG32 INT28S  : 1;
__REG32 INT29S  : 1;
__REG32 INT30S  : 1;
__REG32 INT31S  : 1;
} __nandirqstatus_bits;

/* NandIRQMask register */
typedef struct{
__REG32 INT0M   : 1;
__REG32 INT1M   : 1;
__REG32         : 2;
__REG32 INT4M   : 1;
__REG32 INT5M   : 1;
__REG32 INT6M   : 1;
__REG32 INT7M   : 1;
__REG32 INT8M   : 1;
__REG32 INT9M   : 1;
__REG32 INT10M  : 1;
__REG32 INT11M  : 1;
__REG32 INT12M  : 1;
__REG32 INT13M  : 1;
__REG32 INT14M  : 1;
__REG32 INT15M  : 1;
__REG32 INT16M  : 1;
__REG32 INT17M  : 1;
__REG32 INT18M  : 1;
__REG32 INT19M  : 1;
__REG32 INT20M  : 1;
__REG32 INT21M  : 1;
__REG32 INT22M  : 1;
__REG32 INT23M  : 1;
__REG32 INT24M  : 1;
__REG32 INT25M  : 1;
__REG32 INT26M  : 1;
__REG32 INT27M  : 1;
__REG32 INT28M  : 1;
__REG32 INT29M  : 1;
__REG32 INT30M  : 1;
__REG32 INT31M  : 1;
} __nandirqmask_bits;

/* NandIRQStatusRaw register */
typedef struct{
__REG32 INT0R   : 1;
__REG32 INT1R   : 1;
__REG32         : 2;
__REG32 INT4R   : 1;
__REG32 INT5R   : 1;
__REG32 INT6R   : 1;
__REG32 INT7R   : 1;
__REG32 INT8R   : 1;
__REG32 INT9R   : 1;
__REG32 INT10R  : 1;
__REG32 INT11R  : 1;
__REG32 INT12R  : 1;
__REG32 INT13R  : 1;
__REG32 INT14R  : 1;
__REG32 INT15R  : 1;
__REG32 INT16R  : 1;
__REG32 INT17R  : 1;
__REG32 INT18R  : 1;
__REG32 INT19R  : 1;
__REG32 INT20R  : 1;
__REG32 INT21R  : 1;
__REG32 INT22R  : 1;
__REG32 INT23R  : 1;
__REG32 INT24R  : 1;
__REG32 INT25R  : 1;
__REG32 INT26R  : 1;
__REG32 INT27R  : 1;
__REG32 INT28R  : 1;
__REG32 INT29R  : 1;
__REG32 INT30R  : 1;
__REG32 INT31R  : 1;
} __nandirqstatusraw_bits;

/* NandConfig register */
typedef struct{
__REG32 EC      : 1;
__REG32 WD      : 1;
__REG32 AO      : 1;
__REG32 DE      : 1;
__REG32 ES      : 1;
__REG32 LC      : 2;
__REG32 M       : 1;
__REG32 DC      : 1;
__REG32         : 1;
__REG32 TL      : 2;
__REG32 ECC_MODE: 1;
__REG32         :19;
} __nandconfig_bits;

/* NandIOConfig register */
typedef struct{
__REG32 RD      : 2;
__REG32 WD      : 2;
__REG32 AD      : 2;
__REG32 CD      : 2;
__REG32 DN      :16;
__REG32 NI      : 1;
__REG32         : 7;
} __nandioconfig_bits;

/* NandTiming1 register */
typedef struct{
__REG32 TCLH    : 3;
__REG32         : 1;
__REG32 TCLS    : 3;
__REG32         : 5;
__REG32 TALH    : 3;
__REG32         : 1;
__REG32 TALS    : 3;
__REG32         : 1;
__REG32 TSRD    : 2;
__REG32         :10;
} __nandtiming1_bits;

/* NandTiming2 register */
typedef struct{
__REG32 TWP     : 3;
__REG32         : 1;
__REG32 TWH     : 3;
__REG32         : 1;
__REG32 TRP     : 3;
__REG32         : 1;
__REG32 TREH    : 3;
__REG32         : 1;
__REG32 TCS     : 3;
__REG32         : 1;
__REG32 TCH     : 3;
__REG32         : 1;
__REG32 TEBIDEL : 3;
__REG32         : 1;
__REG32 TDRD    : 3;
__REG32         : 1;
} __nandtiming2_bits;

/* NandSetCmd register */
typedef struct{
__REG32 CV      :16;
__REG32         :16;
} __nandsetcmd_bits;

/* NandSetAddr register */
typedef struct{
__REG32 VA      :16;
__REG32         :16;
} __nandsetaddr_bits;

/* NandWriteData register */
typedef struct{
__REG32 WV      :16;
__REG32         :16;
} __nandwritedata_bits;

/* NandSetCE register */
typedef struct{
__REG32 CEV     : 4;
__REG32 WP      : 1;
__REG32         :27;
} __nandsetce_bits;

/* NandReadData register */
typedef struct{
__REG32 RV      :16;
__REG32         :16;
} __nandreaddata_bits;

/* NandCheckSTS register */
typedef struct{
__REG32 VB      : 1;
__REG32 R0      : 1;
__REG32 R1      : 1;
__REG32 R2      : 1;
__REG32 R3      : 1;
__REG32 R0R     : 1;
__REG32 R1R     : 1;
__REG32 R2R     : 1;
__REG32 R3R     : 1;
__REG32         :23;
} __nandchecksts_bits;

/* NandControlFlow register */
typedef struct{
__REG32 R0      : 1;
__REG32 R1      : 1;
__REG32         : 2;
__REG32 W0      : 1;
__REG32 W1      : 1;
__REG32         :26;
} __nandcontrolflow_bits;

/* NandGPIO1 register */
typedef struct{
__REG32 IO_DATA   :16;
__REG32 IO_DRIVE  : 1;
__REG32 CE1_n     : 1;
__REG32 CE2_n     : 1;
__REG32 CE3_n     : 1;
__REG32 CE4_n     : 1;
__REG32 WE_n      : 1;
__REG32 RE_n      : 1;
__REG32 ALE       : 1;
__REG32 CLE       : 1;
__REG32 WP_n      : 1;
__REG32 GPIO_CFG  : 1;
__REG32           : 5;
} __nandgpio1_bits;

/* NandGPIO2 register */
typedef struct{
__REG32 DATA      :16;
__REG32 RnB0      : 1;
__REG32 RnB1      : 1;
__REG32 RnB2      : 1;
__REG32 RnB3      : 1;
__REG32           :12;
} __nandgpio2_bits;

/* NandIRQStatus2 register */
typedef struct{
__REG32 INT32S    : 1;
__REG32 INT33S    : 1;
__REG32 INT34S    : 1;
__REG32 INT35S    : 1;
__REG32 INT36S    : 1;
__REG32           :27;
} __nandirqstatus2_bits;

/* NandIRQMask2 register */
typedef struct{
__REG32 INT32M    : 1;
__REG32 INT33M    : 1;
__REG32 INT34M    : 1;
__REG32 INT35M    : 1;
__REG32 INT36M    : 1;
__REG32           :27;
} __nandirqmask2_bits;

/* NandIRQStatusRaw2 register */
typedef struct{
__REG32 INT32R    : 1;
__REG32 INT33R    : 1;
__REG32 INT34R    : 1;
__REG32 INT35R    : 1;
__REG32 INT36R    : 1;
__REG32           :27;
} __nandirqstatusraw2_bits;

/* NandAESState register */
typedef struct{
__REG32 STATE     : 2;
__REG32           :30;
} __nandaesstate_bits;

/* NandECCErrStatus register */
typedef struct{
__REG32 N_ERR_0   : 4;
__REG32 N_ERR_1   : 4;
__REG32           :24;
} __nandeccerrstatus_bits;

/* AES_FROM_AHB register */
typedef struct{
__REG32 decryptRAM0   : 1;
__REG32 decryptRAM1   : 1;
__REG32               :30;
} __aes_from_ahb_bits;

/* SDRAM Controller Control Register */
typedef struct{
__REG32 E  : 1;
__REG32 M  : 1;
__REG32 L  : 1;
__REG32    :29;
} __mpmc_ctrl_bits;

/* SDRAM Controller Status Register */
typedef struct{
__REG32 B   : 1;
__REG32 S   : 1;
__REG32 SA  : 1;
__REG32     :29;
} __mpmc_status_bits;

/* SDRAM Controller Configuration Register */
typedef struct{
__REG32 N   : 1;
__REG32     : 7;
__REG32 CLK : 1;
__REG32     :23;
} __mpmc_cfg_bits;

/* Dynamic Memory Control Register */
typedef struct{
__REG32 CE     : 1;
__REG32 CS     : 1;
__REG32 SR     : 1;
__REG32        : 2;
__REG32 MMC    : 1;
__REG32        : 1;
__REG32 I      : 2;
__REG32        : 4;
__REG32 DP     : 1;
__REG32        :18;
} __mpmcd_ctrl_bits;

/* Dynamic Memory Refresh Timer Register */
typedef struct{
__REG32 REFRESH  :11;
__REG32          :21;
} __mpmcd_refresh_bits;

/* Dynamic Memory Read Configuration Register */
typedef struct{
__REG32 RD   : 2;
__REG32      :30;
} __mpmcd_read_cfg_bits;

/* Dynamic Memory Precharge Command Period Register */
typedef struct{
__REG32 tRP  : 4;
__REG32      :28;
} __mpmcd_trp_bits;

/* Dynamic Memory Active to Precharge Command Period Register */
typedef struct{
__REG32 tRAS  : 4;
__REG32       :28;
} __mpmcd_tras_bits;

/* Dynamic Memory Self-refresh Exit Time Register */
typedef struct{
__REG32 tSREX  : 4;
__REG32        :28;
} __mpmcd_tsrex_bits;

/* Dynamic Memory APR Register */
typedef struct{
__REG32 tAPR   : 4;
__REG32        :28;
} __mpmcd_apr_bits;

/* Dynamic Memory DAL Register */
typedef struct{
__REG32 tDAL   : 4;
__REG32        :28;
} __mpmcd_dal_bits;

/* Dynamic Memory Write Recovery Time Register */
typedef struct{
__REG32 tWR  : 4;
__REG32      :28;
} __mpmcd_twr_bits;

/* Dynamic Memory Active To Active Command Period Register */
typedef struct{
__REG32 tRC  : 5;
__REG32      :27;
} __mpmcd_trc_bits;

/* Dynamic Memory Auto-refresh Period Register */
typedef struct{
__REG32 tRFC  : 5;
__REG32       :27;
} __mpmcd_trfc_bits;

/* Dynamic Memory Exit Self-refresh Register */
typedef struct{
__REG32 tXSR  : 5;
__REG32       :27;
} __mpmcd_txsr_bits;

/* Dynamic Memory Active Bank A to Active Bank B Time Register */
typedef struct{
__REG32 tRRD  : 4;
__REG32       :28;
} __mpmcd_trrd_bits;

/* Dynamic Memory Load Mode Register To Active Command Time */
typedef struct{
__REG32 tMRD  : 4;
__REG32       :28;
} __mpmcd_tmrd_bits;

/* Static Memory MPMCStaticExtendedWait */
typedef struct{
__REG32 EXTENDEDWAIT  :10;
__REG32               :22;
} __mpmcs_ext_wait_bits;

/* Dynamic Memory Configuration Register */
typedef struct{
__REG32     : 3;
__REG32 MD  : 2;
__REG32     : 2;
__REG32 AM  : 8;
__REG32     : 4;
__REG32 B   : 1;
__REG32 P   : 1;
__REG32     :11;
} __mpmcd_cfg_bits;

/* Dynamic Memory RAS & CAS Delay Register */
typedef struct{
__REG32 RAS  : 2;
__REG32      : 6;
__REG32 CAS  : 2;
__REG32      :22;
} __mpmcd_ras_cas_bits;


/* MPMCStaticConfig0..3 */
typedef struct{
__REG32 MW   : 2;
__REG32      : 1;
__REG32 PM   : 1;
__REG32      : 2;
__REG32 PC   : 1;
__REG32 BLS  : 1;
__REG32 EW   : 1;
__REG32      :10;
__REG32 B    : 1;
__REG32 WP   : 1;
__REG32      :11;
} __mpmcs_cnfg_bits;

/* MPMCStaticWaitWen0/1 */
typedef struct{
__REG32 WAITWEN : 4;
__REG32         :28;
} __mpmcs_wen_bits;

/* MPMCStaticWaitOen0/1 */
typedef struct{
__REG32 WAITOEN : 4;
__REG32         :28;
} __mpmcs_oen_bits;

/* MPMCStaticWaitRd0/1 */
typedef struct{
__REG32 WAITRD  : 5;
__REG32         :27;
} __mpmcs_rd_bits;

/* MPMCStaticWaitPage0/1 */
typedef struct{
__REG32 WAITPAGE : 5;
__REG32          :27;
} __mpmcs_page_bits;

/* MPMCStaticWaitWr0/1 */
typedef struct{
__REG32 WAITWR  : 5;
__REG32         :27;
} __mpmcs_wr_bits;

/* MPMCStaticWaitTurn0/1 */
typedef struct{
__REG32 WAITTURN  : 4;
__REG32           :28;
} __mpmcs_turn_bits;

/* OTG_CAPLENGTH_REG */
typedef struct{
__REG16 CAPLENGTH       : 8;
__REG16                 : 8;
} __otg_caplength_reg_bits;

/* OTG_HCSPARAMS_REG */
typedef struct{
__REG32 N_PORTS         : 4;
__REG32 PPC             : 1;
__REG32                 : 3;
__REG32 N_PCC           : 3;
__REG32 N_CC            : 5;
__REG32 PI              : 1;
__REG32                 : 3;
__REG32 N_PTT           : 4;
__REG32 N_TT            : 4;
__REG32                 : 4;
} __otg_hcsparams_reg_bits;

/* OTG_HCCPARAMS_REG */
typedef struct{
__REG32 ADC             : 1;
__REG32 PFL             : 1;
__REG32 ASP             : 1;
__REG32                 : 1;
__REG32 IST             : 4;
__REG32 EECP            : 8;
__REG32                 :16;
} __otg_hccparams_reg_bits;

/* OTG_DCCPARAMS_REG */
typedef struct{
__REG32 DEN             : 5;
__REG32                 : 2;
__REG32 DC              : 1;
__REG32 HC              : 1;
__REG32                 :23;
} __otg_dccparams_reg_bits;

/* OTG_USBCMD_REG */
typedef struct{
__REG32 RS              : 1;
__REG32 RST             : 1;
__REG32 FS1             : 2;
__REG32 PSE             : 1;
__REG32 ASE             : 1;
__REG32 IAA             : 1;
__REG32                 : 1;
__REG32 ASP             : 2;
__REG32                 : 1;
__REG32 ASPE            : 1;
__REG32 ATDTW           : 1;
__REG32 SUTW            : 1;
__REG32                 : 1;
__REG32 FS2             : 1;
__REG32 ITC             : 8;
__REG32                 : 8;
} __otg_usbcmd_reg_bits;

/* OTG_USBSTS_REG */
typedef struct{
__REG32 UI              : 1;
__REG32 UEI             : 1;
__REG32 PCI             : 1;
__REG32 FRI             : 1;
__REG32                 : 1;
__REG32 AAI             : 1;
__REG32 URI             : 1;
__REG32 SRI             : 1;
__REG32 SLI             : 1;
__REG32                 : 3;
__REG32 HCH             : 1;
__REG32 RCL             : 1;
__REG32 PS              : 1;
__REG32 AS              : 1;
__REG32 NAKI            : 1;
__REG32                 : 1;
__REG32 UAI             : 1;
__REG32 UPI             : 1;
__REG32                 :12;
} __otg_usbsts_reg_bits;

/* OTG_USBINTR_REG */
typedef struct{
__REG32 UE              : 1;
__REG32 UEE             : 1;
__REG32 PCE             : 1;
__REG32 FRE             : 1;
__REG32                 : 1;
__REG32 AAE             : 1;
__REG32 URE             : 1;
__REG32 SRE             : 1;
__REG32 SLE             : 1;
__REG32                 : 7;
__REG32 NAKE            : 1;
__REG32                 : 1;
__REG32 UAIE            : 1;
__REG32 UPIA            : 1;
__REG32                 :12;
} __otg_usbintr_reg_bits;

/* OTG_PERIODICLISTBASE_DEVICEADDR_REG */
typedef union{
/* OTG_PERIODICLISTBASE_REG*/
  struct{
  __REG32                 :12;
  __REG32 PERBASE         :20;
  };
/* OTG_DEVICEADDR_REG*/
  struct{
  __REG32                 :24;
  __REG32 USBADRA         : 1;
  __REG32 USBADR          : 7;
  };
} __otg_periodiclistbase_reg_bits;

/* OTG_ASYNCLISTADDR_ENDPOINTLISTADDR_REG */
typedef union{
/* OTG_ASYNCLISTADDR_REG*/
  struct{
  __REG32                 : 5;
  __REG32 ASYBASE         :27;
  };
/* OTG_ENDPOINTLISTADDR_REG*/
  struct{
  __REG32                 :11;
  __REG32 EPBASE          :21;
  };
} __otg_asynclistaddr_reg_bits;

/* OTG_TTCTRL_REG */
typedef struct{
__REG32                 :24;
__REG32 TTHA            : 8;
} __otg_ttctrl_reg_bits;

/* OTG_BURSTSIZE_REG */
typedef struct{
__REG32 RXPBURST        : 8;
__REG32 TXPBURST        : 8;
__REG32                 :16;
} __otg_burstsize_reg_bits;

/* OTG_TXFILLTUNING_REG */
typedef struct{
__REG32 TXSCHOH         : 8;
__REG32 TXSCHEALTH      : 5;
__REG32                 : 3;
__REG32 TXFIFOTHRES     : 6;
__REG32                 :10;
} __otg_txfilltuning_reg_bits;

/* BINTERVAL register */
typedef struct{
__REG32 BINT            : 4;
__REG32                 :28;
} __otg_binterval_bit;

/* OTG_ENDPTNAK_REG */
typedef struct{
__REG32 EPRN            : 4;
__REG32                 :12;
__REG32 EPTN            : 4;
__REG32                 :12;
} __otg_endptnak_reg_bits;

/* OTG_ENDPTNAKEN_REG */
typedef struct{
__REG32 EPRNE           : 4;
__REG32                 :12;
__REG32 EPTNE           : 4;
__REG32                 :12;
} __otg_endptnaken_reg_bits;

/* OTG_PORTSC1_REG */
typedef struct{
__REG32 CCS             : 1;
__REG32 CSC             : 1;
__REG32 PE              : 1;
__REG32 PEC             : 1;
__REG32 OCA             : 1;
__REG32 OCC             : 1;
__REG32 FPR             : 1;
__REG32 SUSP            : 1;
__REG32 PR              : 1;
__REG32 HSP             : 1;
__REG32 LS              : 2;
__REG32 PP              : 1;
__REG32                 : 1;
__REG32 PIC             : 2;
__REG32 PTC             : 4;
__REG32 WKCN            : 1;
__REG32 WKDS            : 1;
__REG32 WKOC            : 1;
__REG32 PHCD            : 1;
__REG32 PFSC            : 1;
__REG32                 : 1;
__REG32 PSPD            : 2;
__REG32                 : 4;
} __otg_portsc1_reg_bits;

/* OTG_OTGSC_REG */
typedef struct{
__REG32 VD              : 1;
__REG32 VC              : 1;
__REG32 HAAR            : 1;
__REG32 OT              : 1;
__REG32 DP              : 1;
__REG32 IDPU            : 1;
__REG32 HADP            : 1;
__REG32 HABA            : 1;
__REG32 ID              : 1;
__REG32 AVV             : 1;
__REG32 ASV             : 1;
__REG32 BSV             : 1;
__REG32 BSE             : 1;
__REG32 _1msT           : 1;
__REG32 DPS             : 1;
__REG32                 : 1;
__REG32 IDIS            : 1;
__REG32 AVVIS           : 1;
__REG32 ASVIS           : 1;
__REG32 BSVIS           : 1;
__REG32 BSEIS           : 1;
__REG32 _1msEIS         : 1;
__REG32 DPIS            : 1;
__REG32                 : 1;
__REG32 IDIE            : 1;
__REG32 AVVIE           : 1;
__REG32 ASVIE           : 1;
__REG32 BSVIE           : 1;
__REG32 BSEIE           : 1;
__REG32 _1msE           : 1;
__REG32 DPIE            : 1;
__REG32                 : 1;
} __otg_otgsc_reg_bits;

/* OTG_USBMODE_REG */
typedef struct{
__REG32 CM              : 2;
__REG32 ES              : 1;
__REG32 SLOM            : 1;
__REG32 SDIS            : 1;
__REG32 VBPS            : 1;
__REG32                 :26;
} __otg_usbmode_reg_bits;

/* ENDPTSETUPSTAT */
typedef struct{
__REG32 ENDPTSETUPSTAT  : 4;
__REG32                 :28;
}__otg_endptsetupstat_reg_bits;

/* OTG_ENDPTPRIME_REG */
typedef struct{
__REG32 PERB            : 4;
__REG32                 :12;
__REG32 PETB            : 4;
__REG32                 :12;
} __otg_endptprime_reg_bits;

/* OTG_ENDPTFLUSH_REG */
typedef struct{
__REG32 FERB            : 4;
__REG32                 :12;
__REG32 FETB            : 4;
__REG32                 :12;
} __otg_endptflush_reg_bits;

/* OTG_ENDPTSTATUS_REG */
typedef struct{
__REG32 ERBR            : 4;
__REG32                 :12;
__REG32 ETBR            : 4;
__REG32                 :12;
} __otg_endptstatus_reg_bits;

/* OTG_ENDPTCOMPLETE_REG */
typedef struct{
__REG32 ERCE            : 4;
__REG32                 :12;
__REG32 ETCE            : 4;
__REG32                 :12;
} __otg_endptcomplete_reg_bits;

/* OTG_ENDPTCTRL0_REG */
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
} __otg_endptctrl0_reg_bits;

/* OTG_ENDPTCTRL1_REG - OTG_ENDPTCTRL3_REG */
typedef struct{
__REG32 RXS             : 1;
__REG32                 : 1;
__REG32 RXT             : 2;
__REG32                 : 1;
__REG32 RXI             : 1;
__REG32 RXR             : 1;
__REG32 RXE             : 1;
__REG32                 : 8;
__REG32 TXS             : 1;
__REG32                 : 1;
__REG32 TXT             : 2;
__REG32                 : 1;
__REG32 TXI             : 1;
__REG32 TXR             : 1;
__REG32 TXE             : 1;
__REG32                 : 8;
} __otg_endptctrl_reg_bits;

/* DMA TRANSFER_LENGTH */
typedef struct{
__REG32 TL              :21;
__REG32                 :11;
} __dma_tl_bits;

/* DMA CONFIGURATION */
typedef struct{
__REG32 WRITE_SLAVE_NR            : 5;
__REG32 READ_SLAVE_NR             : 5;
__REG32 TRANSFER_SIZE             : 2;
__REG32 INVERT_ENDIANESS          : 1;
__REG32 COMPANION_CHANNEL_NR      : 3;
__REG32                           : 1;
__REG32 COMPANION_CHANNEL_ENABLE  : 1;
__REG32 CIRCULAR_BUFFER           : 1;
__REG32                           :13;
} __dma_cfg_bits;

/* DMA ENABLE */
typedef struct{
__REG32 enable                    : 1;
__REG32                           :31;
} __dma_ena_bits;

/* DMA TRANSFER_COUNTER */
typedef struct{
__REG32 TC                        :21;
__REG32                           :11;
} __dma_tc_bits;

/* DMA ALT_ENABLE */
typedef struct{
__REG32 ALT_CH_EN0                : 1;
__REG32 ALT_CH_EN1                : 1;
__REG32 ALT_CH_EN2                : 1;
__REG32 ALT_CH_EN3                : 1;
__REG32 ALT_CH_EN4                : 1;
__REG32 ALT_CH_EN5                : 1;
__REG32 ALT_CH_EN6                : 1;
__REG32 ALT_CH_EN7                : 1;
__REG32 ALT_CH_EN8                : 1;
__REG32 ALT_CH_EN9                : 1;
__REG32 ALT_CH_EN10               : 1;
__REG32 ALT_CH_EN11               : 1;
__REG32                           :20;
}__dma_alt_enable_bits;

/* DMA IRQ_STATUS_CLR */
typedef struct{
__REG32 Finished_0                : 1;
__REG32 Half_way_0                : 1;
__REG32 Finished_1                : 1;
__REG32 Half_way_1                : 1;
__REG32 Finished_2                : 1;
__REG32 Half_way_2                : 1;
__REG32 Finished_3                : 1;
__REG32 Half_way_3                : 1;
__REG32 Finished_4                : 1;
__REG32 Half_way_4                : 1;
__REG32 Finished_5                : 1;
__REG32 Half_way_5                : 1;
__REG32 Finished_6                : 1;
__REG32 Half_way_6                : 1;
__REG32 Finished_7                : 1;
__REG32 Half_way_7                : 1;
__REG32 Finished_8                : 1;
__REG32 Half_way_8                : 1;
__REG32 Finished_9                : 1;
__REG32 Half_way_9                : 1;
__REG32 Finished_10               : 1;
__REG32 Half_way_10               : 1;
__REG32 Finished_11               : 1;
__REG32 Half_way_11               : 1;
__REG32                           : 6;
__REG32 Soft_interrupt            : 1;
__REG32 DMA_abort                 : 1;
} __dma_irq_status_clear_bits;

/* DMA IRQ_MASK */
typedef struct{
__REG32 MaskFinished_0            : 1;
__REG32 MaskHalf_way_0            : 1;
__REG32 MaskFinished_1            : 1;
__REG32 MaskHalf_way_1            : 1;
__REG32 MaskFinished_2            : 1;
__REG32 MaskHalf_way_2            : 1;
__REG32 MaskFinished_3            : 1;
__REG32 MaskHalf_way_3            : 1;
__REG32 MaskFinished_4            : 1;
__REG32 MaskHalf_way_4            : 1;
__REG32 MaskFinished_5            : 1;
__REG32 MaskHalf_way_5            : 1;
__REG32 MaskFinished_6            : 1;
__REG32 MaskHalf_way_6            : 1;
__REG32 MaskFinished_7            : 1;
__REG32 MaskHalf_way_7            : 1;
__REG32 MaskFinished_8            : 1;
__REG32 MaskHalf_way_8            : 1;
__REG32 MaskFinished_9            : 1;
__REG32 MaskHalf_way_9            : 1;
__REG32 MaskFinished_10           : 1;
__REG32 MaskHalf_way_10           : 1;
__REG32 MaskFinished_11           : 1;
__REG32 MaskHalf_way_11           : 1;
__REG32                           : 6;
__REG32 MaskSoft_interrupt        : 1;
__REG32 MaskDMA_abort             : 1;
} __dma_irq_mask_bits;

/* DMA SOFT_IN */
typedef struct{
__REG32 SOFT_INTR                 : 1;
__REG32                           :31;
} __dma_soft_int_bits;

/* INT_PRIORITYMASK_{0..1} */
typedef struct{
__REG32 PRIORITY_LIMITER          : 8;
__REG32 inter_slave_dly           :24;
} __int_prioritymask_bits;

/* INT_VECTOR_{0..1} */
typedef struct{
__REG32                           : 3;
__REG32 INDEX                     : 8;
__REG32 TABLE_ADDR                :21;
} __int_vector_bits;

/* INT_PENDING_1_31 */
typedef struct{
__REG32                           : 1;
__REG32 PENDING1                  : 1;
__REG32 PENDING2                  : 1;
__REG32 PENDING3                  : 1;
__REG32 PENDING4                  : 1;
__REG32 PENDING5                  : 1;
__REG32 PENDING6                  : 1;
__REG32 PENDING7                  : 1;
__REG32 PENDING8                  : 1;
__REG32 PENDING9                  : 1;
__REG32 PENDING10                 : 1;
__REG32 PENDING11                 : 1;
__REG32 PENDING12                 : 1;
__REG32 PENDING13                 : 1;
__REG32 PENDING14                 : 1;
__REG32 PENDING15                 : 1;
__REG32 PENDING16                 : 1;
__REG32 PENDING17                 : 1;
__REG32 PENDING18                 : 1;
__REG32 PENDING19                 : 1;
__REG32 PENDING20                 : 1;
__REG32 PENDING21                 : 1;
__REG32 PENDING22                 : 1;
__REG32 PENDING23                 : 1;
__REG32 PENDING24                 : 1;
__REG32 PENDING25                 : 1;
__REG32 PENDING26                 : 1;
__REG32 PENDING27                 : 1;
__REG32 PENDING28                 : 1;
__REG32 PENDING29                 : 1;
__REG32                           : 2;
} __int_pending_1_31_bits;

/* INT_FEATURES */
typedef struct{
__REG32 N                         : 8;
__REG32 P                         : 8;
__REG32 T                         : 6;
__REG32                           :10;
} __int_features_bits;

/* INT_REQUEST_x */
typedef struct{
__REG32 PRIORITY_LEVEL            : 8;
__REG32 TARGET                    : 6;
__REG32                           : 2;
__REG32 ENABLE                    : 1;
__REG32 ACTIVE_LOW                : 1;
__REG32                           : 7;
__REG32 WE_ACTIVE_LOW             : 1;
__REG32 WE_ENABLE                 : 1;
__REG32 WE_TARGET                 : 1;
__REG32 WE_PRIORITY_LEVEL         : 1;
__REG32 CLR_SWINT                 : 1;
__REG32 SET_SWINT                 : 1;
__REG32 PENDING                   : 1;
} __int_request_bits;

/* SCR */
typedef struct{
__REG32 ENF1                      : 1;
__REG32 ENF2                      : 1;
__REG32 RESET                     : 1;
__REG32 STOP                      : 1;
__REG32                           :28;
} __scr_bits;

/* FS */
typedef struct{
__REG32 FS                        : 3;
__REG32                           :29;
} __fs_bits;

/* SSR */
typedef struct{
__REG32 F1STAT                    : 1;
__REG32 FS2STAT                   : 1;
__REG32 FS                        : 3;
__REG32                           :27;
} __ssr_bits;

/* PCR */
typedef struct{
__REG32 RUN                       : 1;
__REG32 AUTO                      : 1;
__REG32 WAKE_EN                   : 1;
__REG32 EXTEN_EN                  : 1;
__REG32 ENOUT_EN                  : 1;
__REG32                           :27;
} __pcr_bits;

/* PSR */
typedef struct{
__REG32 ACTIVE                    : 1;
__REG32 WAKEUP                    : 1;
__REG32                           :30;
} __psr_bits;

/* ESR0 till ESR29 */
typedef struct{
__REG32 ESR_EN                    : 1;
__REG32 ESR_SEL                   : 3;
__REG32                           :28;
} __esr_bits;

/* ESR30 till ESR39 */
typedef struct{
__REG32 ESR_EN                    : 1;
__REG32 ESR_SEL                   : 1;
__REG32                           :30;
} __esr30_bits;

/* ESR50 till ESR57 */
typedef struct{
__REG32 ESR_EN                    : 1;
__REG32 ESR_SEL                   : 2;
__REG32                           :29;
} __esr50_bits;

/* ESR58 till ESR72 */
typedef struct{
__REG32 ESR_EN                    : 1;
__REG32                           :31;
} __esr58_bits;

/* BCR */
typedef struct{
__REG32 FDRUN                     : 1;
__REG32                           :31;
} __bcr_bits;

/* FDC0 till FDC23 (except FDC17) */
typedef struct{
__REG32 FDCTRL_RUN                : 1;
__REG32 FDCTRL_RESET              : 1;
__REG32 FDCTRL_STRETCH            : 1;
__REG32 MADD                      : 8;
__REG32 MSUB                      : 8;
__REG32                           :13;
} __fdc_bits;

/* FDC17 */
typedef struct{
__REG32 FDCTRL_RUN                : 1;
__REG32 FDCTRL_RESET              : 1;
__REG32 FDCTRL_STRETCH            : 1;
__REG32 MADD                      :13;
__REG32 MSUB                      :13;
__REG32                           : 3;
} __fdc17_bits;

/* DYN_FDC0 till DYN_FDC6 */
typedef struct{
__REG32 DYN_FDCTRL_RUN            : 1;
__REG32 DYN_FDC_ALLOW             : 1;
__REG32 DYN_FDCTRL_STRETCH        : 1;
__REG32 MADD                      : 8;
__REG32 MSUB                      : 8;
__REG32 STOP_AUTO_RESET           : 1;
__REG32                           :12;
} __dyn_fdc_bits;

/* DYN_SEL0 till DYN_SEL6 */
typedef struct{
__REG32 simple_dma_trans          : 1;
__REG32 simple_dma_ready          : 1;
__REG32 arm926_lp_i_trans         : 1;
__REG32 arm926_lp_i_ready         : 1;
__REG32 arm926_lp_d_trans         : 1;
__REG32 arm926_lp_d_ready         : 1;
__REG32 usb_otg_mst_trans         : 1;
__REG32 ecc_ram_busy              : 1;
__REG32 mpmc_refresh_req          : 1;
__REG32                           :23;
} __dyn_sel_bits;

/* Powermode */
typedef struct{
__REG32 Powermode                 : 2;
__REG32                           :30;
} __powermode_bits;

/* WD_BARK */
typedef struct{
__REG32 WD_BARK                   : 1;
__REG32                           :31;
} __wd_bark_bits;

/* FFAST_ON */
typedef struct{
__REG32 FFAST_ON                  : 1;
__REG32                           :31;
} __ffast_on_bits;

/* FFAST_BYPASS */
typedef struct{
__REG32 FFAST_BYPASS              : 1;
__REG32                           :31;
} __ffast_bypass_bits;

/* APB0_RESETN_SOFT */
typedef struct{
__REG32 APB0_RESETN_SOFT          : 1;
__REG32                           :31;
} __apb0_resetn_soft_bits;

/* AHB2APB0_PNRES_SOFT */
typedef struct{
__REG32 AHB2APB0_PNRES_SOFT       : 1;
__REG32                           :31;
} __ahb2apb0_pnres_soft_bits;

/* APB1_RESETN_SOFT */
typedef struct{
__REG32 APB1_RESETN_SOFT          : 1;
__REG32                           :31;
} __apb1_resetn_soft_bits;

/* AHB2APB1_PNRES_SOFT */
typedef struct{
__REG32 AHB2APB1_PNRES_SOFT       : 1;
__REG32                           :31;
} __ahb2apb1_pnres_soft_bits;

/* APB2_RESETN_SOFT */
typedef struct{
__REG32 APB2_RESETN_SOFT          : 1;
__REG32                           :31;
} __apb2_resetn_soft_bits;

/* AHB2APB2_PNRES_SOFT */
typedef struct{
__REG32 AHB2APB2_PNRES_SOFT       : 1;
__REG32                           :31;
} __ahb2apb2_pnres_soft_bits;

/* APB3_RESETN_SOFT */
typedef struct{
__REG32 APB3_RESETN_SOFT          : 1;
__REG32                           :31;
} __apb3_resetn_soft_bits;

/* AHB2APB3_PNRES_SOFT */
typedef struct{
__REG32 AHB2APB3_PNRES_SOFT       : 1;
__REG32                           :31;
} __ahb2apb3_pnres_soft_bits;

/* APB4_RESETN_SOFT */
typedef struct{
__REG32 APB4_RESETN_SOFT          : 1;
__REG32                           :31;
} __apb4_resetn_soft_bits;

/* AHB2MMIO_RESETN_SOFT */
typedef struct{
__REG32 AHB_TO_INTC_RESETN_SOFT   : 1;
__REG32                           :31;
} __ahb_to_intc_resetn_soft_bits;

/* AHB0_RESETN_SOFT */
typedef struct{
__REG32 AHB0_RESETN_SOFT          : 1;
__REG32                           :31;
} __ahb0_resetn_soft_bits;

/* EBI_RESET_N_SOFT */
typedef struct{
__REG32 EBI_RESETN_SOFT           : 1;
__REG32                           :31;
} __ebi_reset_n_soft_bits;

/* CCP_IPINT_PNRES_SOFT UNIT */
typedef struct{
__REG32 PCM_PNRES_SOFT_UNIT       : 1;
__REG32                           :31;
} __pcm_pnres_soft_bits;

/* CCP_IPINT_RESET_N_SOFT */
typedef struct{
__REG32 PCM_RESET_N_SOFT          : 1;
__REG32                           :31;
} __pcm_reset_n_soft_bits;

/* CCP_IPINT_RESET_ASYNC_N_SOFT */
typedef struct{
__REG32 PCM_RESET_ASYNC_N_SOFT        : 1;
__REG32                               :31;
} __pcm_reset_async_n_soft_bits;

/* TIMER0_PNRES_SOFT */
typedef struct{
__REG32 TIMER0_PNRES_SOFT             : 1;
__REG32                               :31;
} __timer0_pnres_soft_bits;

/* TIMER1_PNRES_SOFT */
typedef struct{
__REG32 TIMER1_PNRES_SOFT             : 1;
__REG32                               :31;
} __timer1_pnres_soft_bits;

/* TIMER2_PNRES_SOFT */
typedef struct{
__REG32 TIMER2_PNRES_SOFT             : 1;
__REG32                               :31;
} __timer2_pnres_soft_bits;

/* TIMER3_PNRES_SOFT */
typedef struct{
__REG32 TIMER3_PNRES_SOFT             : 1;
__REG32                               :31;
} __timer3_pnres_soft_bits;

/* ADC_PRESETN_SOFT */
typedef struct{
__REG32 ADC_PRESETN_SOFT              : 1;
__REG32                               :31;
} __adc_presetn_soft_bits;

/* ADC_RESETN_ADC10BITS_SOFT */
typedef struct{
__REG32 ADC_RESETN_ADC10BITS_SOFT       : 1;
__REG32                                 :31;
} __adc_resetn_adc10bits_soft_bits;

/* PWM_RESET_AN_SOFT */
typedef struct{
__REG32 PWM_RESET_AN_SOFT               : 1;
__REG32                                 :31;
} __pwm_reset_an_soft_bits;

/* UART_SYS_RST_AN_SOFT */
typedef struct{
__REG32 UART_SYS_RST_AN_SOFT            : 1;
__REG32                                 :31;
} __uart_sys_rst_an_soft_bits;

/* I2C0_PNRES_SOFT */
typedef struct{
__REG32 I2C0_PNRES_SOFT                 : 1;
__REG32                                 :31;
} __i2c0_pnres_soft_bits;

/* I2C1_PNRES_SOFT */
typedef struct{
__REG32 I2C1_PNRES_SOFT                 : 1;
__REG32                                 :31;
} __i2c1_pnres_soft_bits;

/* I2S_CFG_RST_N_SOFT */
typedef struct{
__REG32 I2S_CFG_RST_N_SOFT              : 1;
__REG32                                 :31;
} __i2s_cfg_rst_n_soft_bits;

/* I2S_NSOF_RST_N_SOFT */
typedef struct{
__REG32 I2S_NSOF_RST_N_SOFT             : 1;
__REG32                                 :31;
} __i2s_nsof_rst_n_soft_bits;

/* EDGE_DET_RST_N_SOFT */
typedef struct{
__REG32 EDGE_DET_RST_N_SOFT             : 1;
__REG32                                 :31;
} __edge_det_rst_n_soft_bits;

/* I2STX_FIFO_0_RST_N_SOFT */
typedef struct{
__REG32 I2STX_FIFO_0_RST_N_SOFT         : 1;
__REG32                                 :31;
} __i2stx_fifo_0_rst_n_soft_bits;

/* I2STX_IF_0_RST_N_SOFT */
typedef struct{
__REG32 I2STX_IF_0_RST_N_SOFT           : 1;
__REG32                                 :31;
} __i2stx_if_0_rst_n_soft_bits;

/* I2STX_FIFO_1_RST_N_SOFT */
typedef struct{
__REG32 I2STX_FIFO_1_RST_N_SOFT         : 1;
__REG32                                 :31;
} __i2stx_fifo_1_rst_n_soft_bits;

/* I2STX_IF_1_RST_N_SOFT */
typedef struct{
__REG32 I2STX_IF_1_RST_N_SOFT           : 1;
__REG32                                 :31;
} __i2stx_if_1_rst_n_soft_bits;

/* I2SRX_FIFO_0_RST_N_SOFT */
typedef struct{
__REG32 I2SRX_FIFO_0_RST_N_SOFT         : 1;
__REG32                                 :31;
} __i2srx_fifo_0_rst_n_soft_bits;

/* I2SRX_IF_0_RST_N_SOFT */
typedef struct{
__REG32 I2SRX_IF_0_RST_N_SOFT           : 1;
__REG32                                 :31;
} __i2srx_if_0_rst_n_soft_bits;

/* I2SRX_FIFO_1_RST_N_SOFT */
typedef struct{
__REG32 I2SRX_FIFO_1_RST_N_SOFT         : 1;
__REG32                                 :31;
} __i2srx_fifo_1_rst_n_soft_bits;

/* I2SRX_IF_1_RST_N_SOFT */
typedef struct{
__REG32 I2SRX_IF_1_RST_N_SOFT           : 1;
__REG32                                 :31;
} __i2srx_if_1_rst_n_soft_bits;

/*LCD_PNRES_SOFT */
typedef struct{
__REG32 LCD_PNRES_SOFT                  : 1;
__REG32                                 :31;
} __lcd_pnres_soft_bits;

/* SPI_PNRES_APB_SOFT */
typedef struct{
__REG32 SPI_PNRES_APB_SOFT              : 1;
__REG32                                 :31;
} __spi_pnres_apb_soft_bits;

/* SPI_PNRES_IP_SOFT */
typedef struct{
__REG32 SPI_PNRES_IP_SOFT               : 1;
__REG32                                 :31;
} __spi_pnres_ip_soft_bits;

/*DMA_PNRES_SOFT */
typedef struct{
__REG32 DMA_PNRES_SOFT           : 1;
__REG32                                 :31;
} __dma_pnres_soft_bits;

/* NANDFLASH_CTRL_ECC_RESET_N_SOFT */
typedef struct{
__REG32 NANDFLASH_CTRL_ECC_RESET_N_SOFT : 1;
__REG32                                 :31;
} __nandflash_ctrl_ecc_reset_n_soft_bits;

/* NANDFLASH_CTRL_AES_RESET_N_SOFT */
typedef struct{
__REG32 NANDFLASH_CTRL_AES_RESET_N_SOFT : 1;
__REG32                                 :31;
} __nandflash_ctrl_aes_reset_n_soft_bits;

/* NANDFLASH_CTRL_NAND_RESET_N_SOFT */
typedef struct{
__REG32 NANDFLASH_CTRL_NAND_RESET_N_SOFT: 1;
__REG32                                 :31;
} __nandflash_ctrl_nand_reset_n_soft_bits;

/* SD_MMC_PNRES_SOFT */
typedef struct{
__REG32 SD_MMC_PNRES_SOFT               : 1;
__REG32                                 :31;
} __sd_mmc_pnres_soft_bits;

/* SD_MMC_NRES_CCLK_IN_SOFT */
typedef struct{
__REG32 SD_MMC_NRES_CCLK_IN_SOFT        : 1;
__REG32                                 :31;
} __sd_mmc_nres_cclk_in_soft_bits;

/* USB_OTG_AHB_RST_N_SOFT */
typedef struct{
__REG32 USB_OTG_AHB_RST_N_SOFT          : 1;
__REG32                                 :31;
} __usb_otg_ahb_rst_n_soft_bits;

/* RED_CTL_RESET_N_SOFT */
typedef struct{
__REG32 RED_CTL_RESET_N_SOFT            : 1;
__REG32                                 :31;
} __red_ctl_reset_n_soft_bits;

/* AHB_MPMC_HRESETN_SOFT */
typedef struct{
__REG32 AHB_MPMC_HRESETN_SOFT           : 1;
__REG32                                 :31;
} __ahb_mpmc_hresetn_soft_bits;

/* AHB_MPMC_REFRESH_RESETN_SOFT */
typedef struct{
__REG32 AHB_MPMC_REFRESH_RESETN_SOFT        : 1;
__REG32                                     :31;
} __ahb_mpmc_refresh_resetn_soft_bits;

/* INTC_RESETN_SOFT */
typedef struct{
__REG32 INTC_RESETN_SOFT                    : 1;
__REG32                                     :31;
} __intc_resetn_soft_bits;

/* HP0 Frequency Input Select register */
typedef struct{
__REG32 HP0_FIN_SELECT                      : 4;
__REG32                                     :28;
} __hp0_fin_select_bits;

/* HP0 M-divider register */
typedef struct{
__REG32 HP0_MDEC                            :17;
__REG32                                     :15;
} __hp0_mdec_bits;

/* HP0 N-divider register */
typedef struct{
__REG32 HP0_NDEC                            :10;
__REG32                                     :22;
} __hp0_ndec_bits;

/* HP0 P-divider register */
typedef struct{
__REG32 HP0_PDEC                            : 7;
__REG32                                     :25;
} __hp0_pdec_bits;

/* HP0 Mode register */
typedef struct{
__REG32 HP0_MODE_CLKEN                      : 1;
__REG32 HP0_MODE_SKEW_EN                    : 1;
__REG32 HP0_MODE_PD                         : 1;
__REG32 HP0_MODE_DIRECTO                    : 1;
__REG32 HP0_MODE_DIRECTI                    : 1;
__REG32 HP0_MODE_FRM                        : 1;
__REG32 HP0_MODE_BANDSEL                    : 1;
__REG32 HP0_MODE_LIMUP_OFF                  : 1;
__REG32 HP0_MODE_BYPASS                     : 1;
__REG32                                     :23;
} __hp0_mode_bits;

/* HP0 Status register */
typedef struct{
__REG32 HP0_STATUS_LOCK                     : 1;
__REG32 HP0_STATUS_FR                       : 1;
__REG32                                     :30;
} __hp0_status_bits;

/* HP0 Acknowledge register */
typedef struct{
__REG32 HP0_ACK_M                          : 1;
__REG32 HP0_ACK_N                          : 1;
__REG32 HP0_ACK_P                          : 1;
__REG32                                    :29;
} __hp0_ack_bits;

/* HP0 request register */
typedef struct{
__REG32 HP0_REQ_M                          : 1;
__REG32 HP0_REQ_N                          : 1;
__REG32 HP0_REQ_P                          : 1;
__REG32                                    :29;
} __hp0_req_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_INSELR                         : 4;
__REG32                                    :28;
} __hp0_inselr_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_INSELI                         : 6;
__REG32                                    :26;
} __hp0_inseli_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_INSELP                         : 5;
__REG32                                    :27;
} __hp0_inselp_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_SELR                           : 4;
__REG32                                    :28;
} __hp0_selr_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_SELI                           : 6;
__REG32                                    :26;
} __hp0_seli_bits;

/* HP0 Bandwith Selection register */
typedef struct{
__REG32 HP0_SELP                           : 5;
__REG32                                    :27;
} __hp0_selp_bits;

/* HP1 Frequency Input Select register */
typedef struct{
__REG32 HP1_FIN_SELECT                      : 4;
__REG32                                     :28;
} __hp1_fin_select_bits;

/* HP1 M-divider register */
typedef struct{
__REG32 HP1_MDEC                            :17;
__REG32                                     :15;
} __hp1_mdec_bits;

/* HP1 N-divider register */
typedef struct{
__REG32 HP1_NDEC                            :10;
__REG32                                     :22;
} __hp1_ndec_bits;

/* HP1 P-divider register */
typedef struct{
__REG32 HP1_PDEC                            : 7;
__REG32                                     :25;
} __hp1_pdec_bits;

/* HP1 Mode register */
typedef struct{
__REG32 HP1_MODE_CLKEN                      : 1;
__REG32 HP1_MODE_SKEW_EN                    : 1;
__REG32 HP1_MODE_PD                         : 1;
__REG32 HP1_MODE_DIRECTO                    : 1;
__REG32 HP1_MODE_DIRECTI                    : 1;
__REG32 HP1_MODE_FRM                        : 1;
__REG32 HP1_MODE_BANDSEL                    : 1;
__REG32 HP1_MODE_LIMUP_OFF                  : 1;
__REG32 HP1_MODE_BYPASS                     : 1;
__REG32                                     :23;
} __hp1_mode_bits;

/* HP1 Status register */
typedef struct{
__REG32 HP1_STATUS_LOCK                     : 1;
__REG32 HP1_STATUS_FR                       : 1;
__REG32                                     :30;
} __hp1_status_bits;

/* HP1 Acknowledge register */
typedef struct{
__REG32 HP1_ACK_M                          : 1;
__REG32 HP1_ACK_N                          : 1;
__REG32 HP1_ACK_P                          : 1;
__REG32                                    :29;
} __hp1_ack_bits;

/* HP1 request register */
typedef struct{
__REG32 HP1_REQ_M                          : 1;
__REG32 HP1_REQ_N                          : 1;
__REG32 HP1_REQ_P                          : 1;
__REG32                                    :29;
} __hp1_req_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_INSELR                         : 4;
__REG32                                    :28;
} __hp1_inselr_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_INSELI                         : 6;
__REG32                                    :26;
} __hp1_inseli_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_INSELP                         : 5;
__REG32                                    :27;
} __hp1_inselp_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_SELR                           : 4;
__REG32                                    :28;
} __hp1_selr_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_SELI                           : 6;
__REG32                                    :26;
} __hp1_seli_bits;

/* HP1 Bandwith Selection register */
typedef struct{
__REG32 HP1_SELP                           : 5;
__REG32                                    :27;
} __hp1_selp_bits;

/* Interrupt Register (IR) of the Watchdog Timer */
typedef struct{
__REG32 intr_m0    : 1;
__REG32 intr_m1    : 1;
__REG32            :30;
} __wdtim_ir_bits;

/* Timer Control Register (TCR) of the Watchdog Timer */
typedef struct{
__REG32 COUNT_ENAB   : 1;
__REG32 RESET_COUNT  : 1;
__REG32              :30;
} __wdtim_tcr_bits;

/* Watchdog Timer Match Control Register */
typedef struct{
__REG32 MR0_INT       : 1;
__REG32 RESET_COUNT0  : 1;
__REG32 STOP_COUNT0   : 1;
__REG32 MR1_INT       : 1;
__REG32 RESET_COUNT1  : 1;
__REG32 STOP_COUNT1   : 1;
__REG32               :26;
} __wdtim_mcr_bits;

/* Watchdog Timer External Match Control Register */
typedef struct{
__REG32 EXT_MATCH0  : 1;
__REG32 EXT_MATCH1  : 1;
__REG32             : 2;
__REG32 MATCH_CTRL0 : 2;
__REG32 MATCH_CTRL1 : 2;
__REG32             :24;
} __wdtim_emr_bits;

/* EBI_MCI registers */
typedef struct{
__REG32 MGPIO9            : 1;
__REG32 MGPIO6            : 1;
__REG32 MLCD_DB_7         : 1;
__REG32 MLCD_DB_4         : 1;
__REG32 MLCD_DB_2         : 1;
__REG32 MNAND_RYBN0       : 1;
__REG32 MI2STX_CLK0       : 1;
__REG32 MI2STX_BCK0       : 1;
__REG32 EBI_A_1_CLE       : 1;
__REG32 EBI_NCAS_BLOUT_0  : 1;
__REG32 MLCD_DB_0         : 1;
__REG32 EBI_DQM_0_NOE     : 1;
__REG32 MLCD_CSB          : 1;
__REG32 MLCD_DB_1         : 1;
__REG32 MLCD_E_RD         : 1;
__REG32 MLCD_RS           : 1;
__REG32 MLCD_RW_WR        : 1;
__REG32 MLCD_DB_3         : 1;
__REG32 MLCD_DB_5         : 1;
__REG32 MLCD_DB_6         : 1;
__REG32 MLCD_DB_8         : 1;
__REG32 MLCD_DB_9         : 1;
__REG32 MLCD_DB_10        : 1;
__REG32 MLCD_DB_11        : 1;
__REG32 MLCD_DB_12        : 1;
__REG32 MLCD_DB_13        : 1;
__REG32 MLCD_DB_14        : 1;
__REG32 MLCD_DB_15        : 1;
__REG32 MGPIO5            : 1;
__REG32 MGPIO7            : 1;
__REG32 MGPIO8            : 1;
__REG32 MGPIO10           : 1;
} __ioconf_ebi_mci_bits;

/* EBI_I2STX_0 registers */
typedef struct{
__REG32 MNAND_RYBN1       : 1;
__REG32 MNAND_RYBN2       : 1;
__REG32 MNAND_RYBN3       : 1;
__REG32 MUART_CTS_N       : 1;
__REG32 MUART_RTS_N       : 1;
__REG32 MI2STX_DATA0      : 1;
__REG32 MI2STX_WS0        : 1;
__REG32 EBI_NRAS_BLOUT_1  : 1;
__REG32 EBI_A_0_ALE       : 1;
__REG32 EBI_NWE           : 1;
__REG32                   :22;
} __ioconf_ebi_i2stx_0_bits;

/* CGU */
typedef struct{
__REG32 CGU_SYSCLK_O      : 1;
__REG32                   :31;
} __ioconf_cgu_bits;

/* I2SRX_0 registers */
typedef struct{
__REG32 I2SRX_BCK0        : 1;
__REG32 I2SRX_DATA0       : 1;
__REG32 I2SRX_WS0         : 1;
__REG32                   :29;
} __ioconf_i2s0_rx_bits;

/* I2SRX_1 registers */
typedef struct{
__REG32 I2SRX_DATA1       : 1;
__REG32 I2SRX_BCK1        : 1;
__REG32 I2SRX_WS1         : 1;
__REG32                   :29;
} __ioconf_i2s1_rx_bits;

/* I2STX_1 registers */
typedef struct{
__REG32 I2STX_DATA1       : 1;
__REG32 I2STX_BCK1        : 1;
__REG32 I2STX_WS1         : 1;
__REG32 I2STX_256FS_O     : 1;
__REG32                   :28;
} __ioconf_i2s1_tx_bits;

/* EBI registers */
typedef struct{
__REG32 EBI_D_9  : 1;
__REG32 EBI_D_10 : 1;
__REG32 EBI_D_11 : 1;
__REG32 EBI_D_12 : 1;
__REG32 EBI_D_13 : 1;
__REG32 EBI_D_14 : 1;
__REG32 EBI_D_4  : 1;
__REG32 EBI_D_0  : 1;
__REG32 EBI_D_1  : 1;
__REG32 EBI_D_2  : 1;
__REG32 EBI_D_3  : 1;
__REG32 EBI_D_5  : 1;
__REG32 EBI_D_6  : 1;
__REG32 EBI_D_7  : 1;
__REG32 EBI_D_8  : 1;
__REG32 EBI_D_15 : 1;
__REG32          :16;
} __ioconf_ebi_bits;

/* GPIO */
typedef struct{
__REG32 GPIO_GPIO1          : 1;
__REG32 GPIO_GPIO0          : 1;
__REG32 GPIO_GPIO2          : 1;
__REG32 GPIO_GPIO3          : 1;
__REG32 GPIO_GPIO4          : 1;
__REG32 GPIO_GPIO11         : 1;
__REG32 GPIO_GPIO12         : 1;
__REG32 GPIO_GPIO13         : 1;
__REG32 GPIO_GPIO14         : 1;
__REG32 GPIO_GPIO15         : 1;
__REG32 GPIO_GPIO16         : 1;
__REG32 GPIO_GPIO17         : 1;
__REG32 GPIO_GPIO18         : 1;
__REG32 GPIO_GPIO19         : 1;
__REG32 GPIO_GPIO20         : 1;
__REG32                     :17;
} __ioconf_gpio_bits;

/* I2C1 */
typedef struct{
__REG32 I2C_SDA1       : 1;
__REG32 I2C_SCL1       : 1;
__REG32                :30;
} __ioconf_i2c1_bits;

/* SPI registers */
typedef struct{
__REG32 SPI_MISO      : 1;
__REG32 SPI_MOSI      : 1;
__REG32 SPI_CS_IN     : 1;
__REG32 SPI_SCK       : 1;
__REG32 SPI_CS_OUT0   : 1;
__REG32               :27;
} __ioconf_spi_bits;

/* NANDFLASH CTRL */
typedef struct{
__REG32 NAND_NCS_3    : 1;
__REG32 NAND_NCS_0    : 1;
__REG32 NAND_NCS_1    : 1;
__REG32 NAND_NCS_2    : 1;
__REG32               :28;
} __ioconf_nand_ctrl_bits;

/* PWM */
typedef struct{
__REG32 PWM_PWM_DATA  : 1;
__REG32               :31;
} __ioconf_pwm_bits;

/* UART */
typedef struct{
__REG32 UART_RXD      : 1;
__REG32 UART_TXD      : 1;
__REG32               :30;
} __ioconf_uart_bits;

/* ADC_Rx_REG */
typedef struct{
__REG32 VALUE           :10;
__REG32                 :22;
} __adc_r_reg_bits;

/* ADC_CON_REG */
typedef struct{
__REG32                  : 1;
__REG32 ENABLE           : 1;
__REG32 CSCAN            : 1;
__REG32 START            : 1;
__REG32 STATUS           : 1;
__REG32                  :27;
} __adc_con_reg_bits;

/* ADC_CSEL_RES_REG */
typedef struct{
__REG32 CSEL0            : 4;
__REG32 CSEL1            : 4;
__REG32 CSEL2            : 4;
__REG32 CSEL3            : 4;
__REG32                  :16;
} __adc_csel_reg_bits;

/* ADC_INT_ENABLE_REG */
typedef struct{
__REG32 INT_ENA         : 1;
__REG32                 :31;
} __adc_int_enable_reg_bits;

/* ADC_INT_STATUS_REG */
typedef struct{
__REG32 INT_STATUS      : 1;
__REG32                 :31;
} __adc_int_status_reg_bits;

/* ADC_INT_CLEAR_REG */
typedef struct{
__REG32 INT_CLEAR       : 1;
__REG32                 :31;
} __adc_int_clear_reg_bits;

/* Pend [0] register */
typedef struct{
__REG32 pcm_int               : 1;
__REG32 mLCD_DB_0             : 1;
__REG32 mLCD_DB_1             : 1;
__REG32 mLCD_DB_2             : 1;
__REG32 mLCD_DB_3             : 1;
__REG32 mLCD_DB_4             : 1;
__REG32 mLCD_DB_5             : 1;
__REG32 mLCD_DB_6             : 1;
__REG32 mLCD_DB_7             : 1;
__REG32 mLCD_DB_8             : 1;
__REG32 mLCD_DB_9             : 1;
__REG32 mLCD_DB_10            : 1;
__REG32 mLCD_DB_11            : 1;
__REG32 mLCD_DB_12            : 1;
__REG32 mLCD_DB_13            : 1;
__REG32 mLCD_DB_14            : 1;
__REG32 mLCD_DB_15            : 1;
__REG32 mLCD_RS               : 1;
__REG32 mLCD_CSB              : 1;
__REG32 mLCD_E_RD             : 1;
__REG32 mLCD_RW_WR            : 1;
__REG32 mNAND_RYBN0           : 1;
__REG32 mNAND_RYBN1           : 1;
__REG32 mNAND_RYBN2           : 1;
__REG32 mNAND_RYBN3           : 1;
__REG32 EBI_D_0               : 1;
__REG32 EBI_D_1               : 1;
__REG32 EBI_D_2               : 1;
__REG32 EBI_D_3               : 1;
__REG32 EBI_D_4               : 1;
__REG32 EBI_D_5               : 1;
__REG32 EBI_D_6               : 1;
} __er_pend0_bits;

/* Pend [1] register */
typedef struct{
__REG32 EBI_D_7               : 1;
__REG32 EBI_D_8               : 1;
__REG32 EBI_D_9               : 1;
__REG32 EBI_D_10              : 1;
__REG32 EBI_D_11              : 1;
__REG32 EBI_D_12              : 1;
__REG32 EBI_D_13              : 1;
__REG32 EBI_D_14              : 1;
__REG32 EBI_D_15              : 1;
__REG32 EBI_NWE               : 1;
__REG32 EBI_A_0_ALE           : 1;
__REG32 EBI_A_1_CLE           : 1;
__REG32 EBI_DQM_0_NOE         : 1;
__REG32 EBI_NCAS_BLOUT_0      : 1;
__REG32 EBI_NCAS_BLOUT_1      : 1;
__REG32 GPIO0                 : 1;
__REG32 GPIO1                 : 1;
__REG32 GPIO2                 : 1;
__REG32 GPIO3                 : 1;
__REG32 GPIO4                 : 1;
__REG32 mGPIO5                : 1;
__REG32 mGPIO6                : 1;
__REG32 mGPIO7                : 1;
__REG32 mGPIO8                : 1;
__REG32 mGPIO9                : 1;
__REG32 mGPIO10               : 1;
__REG32 GPIO11                : 1;
__REG32 GPIO12                : 1;
__REG32 GPIO13                : 1;
__REG32 GPIO14                : 1;
__REG32 GPIO15                : 1;
__REG32 GPIO16                : 1;
} __er_pend1_bits;

/* Pend [2] register */
typedef struct{
__REG32 GPIO17                : 1;
__REG32 GPIO18                : 1;
__REG32 NAND_NCS_0            : 1;
__REG32 NAND_NCS_1            : 1;
__REG32 NAND_NCS_2            : 1;
__REG32 NAND_NCS_3            : 1;
__REG32 SPI_MISO              : 1;
__REG32 SPI_MOSI              : 1;
__REG32 SPI_CS_IN             : 1;
__REG32 SPI_SCK               : 1;
__REG32 SPI_CS_OUT0           : 1;
__REG32 UART_RXD              : 1;
__REG32 UART_TXD              : 1;
__REG32 mUART_CTS_N           : 1;
__REG32 mUART_RTS_N           : 1;
__REG32 mI2STX_CLK0           : 1;
__REG32 mI2STX_BCK0           : 1;
__REG32 mI2STX_DATA0          : 1;
__REG32 mI2STX_WS0            : 1;
__REG32 I2SRX_BCK0            : 1;
__REG32 I2SRX_DATA0           : 1;
__REG32 I2SRX_WS0             : 1;
__REG32 I2SRX_DATA1           : 1;
__REG32 I2SRX_BCK1            : 1;
__REG32 I2SRX_WS1             : 1;
__REG32 I2STX_DATA1           : 1;
__REG32 I2STX_BCK1            : 1;
__REG32 I2STX_WS1             : 1;
__REG32 CLK_256FS_O           : 1;
__REG32 I2C_SDA1              : 1;
__REG32 I2C_SCL1              : 1;
__REG32 PWM_DATA              : 1;
} __er_pend2_bits;

/* Pend [3] register */
typedef struct{
__REG32 GPIO19                : 1;
__REG32 GPIO20                : 1;
__REG32 ssa1_timer0_intct1    : 1;
__REG32 ssa1_timer1_intct1    : 1;
__REG32 ssa1_timer2_intct1    : 1;
__REG32 ssa1_timer3_intct1    : 1;
__REG32 ssa1_adc_int          : 1;
__REG32 wdog_m0               : 1;
__REG32 uart_rxd              : 1;
__REG32 I2c0_scl_n            : 1;
__REG32 I2c1_scl_n            : 1;
__REG32 arm926_901616_lp_nfiq : 1;
__REG32 arm926_901616_lp_nirq : 1;
__REG32 MCI_DAT_0             : 1;
__REG32 MCI_DAT_1             : 1;
__REG32 MCI_DAT_2             : 1;
__REG32 MCI_DAT_3             : 1;
__REG32 MCI_DAT_4             : 1;
__REG32 MCI_DAT_5             : 1;
__REG32 MCI_DAT_6             : 1;
__REG32 MCI_DAT_7             : 1;
__REG32 MCI_CMD               : 1;
__REG32 MCI_CLK               : 1;
__REG32 USB_VBUS1             : 1;
__REG32 usb_otg_ahb_needclk   : 1;
__REG32 usb_atx_pll_lock      : 1;
__REG32 usb_otg_vbus_pwr_en   : 1;
__REG32 USB_ID                : 1;
__REG32 isram0_mrc_finished   : 1;
__REG32 isram1_mrc_finished   : 1;
__REG32                       : 2;
} __er_pend3_bits;

/* Intout register */
typedef struct{
__REG32 intout0               : 1;
__REG32 intout1               : 1;
__REG32 intout2               : 1;
__REG32 intout3               : 1;
__REG32 cgu_wakeup            : 1;
__REG32                       :27;
} __er_intout_bits;

/* POWERDOWN */
typedef struct{
__REG32 SRST                  : 1;
__REG32 FSRST                 : 1;
__REG32 PD                    : 1;
__REG32                       :29;
} __rng_powerdown_bits;

/* OTP_con register */
typedef struct{
__REG32 ADRS                  : 9;
__REG32                       : 7;
__REG32 MODE                  : 2;
__REG32                       :13;
__REG32 JTAG_EN               : 1;
} __otp_con_bits;

/* OTP_rprot register */
typedef struct{
__REG32 PROT                  :16;
__REG32                       :15;
__REG32 LOCK                  : 1;
} __otp_rprot_bits;

/* SPI Configuration */
typedef struct{
__REG32 ENA                   : 1;
__REG32 MST                   : 1;
__REG32 LB                    : 1;
__REG32 TM                    : 1;
__REG32 SDIS                  : 1;
__REG32                       : 1;
__REG32 SRST                  : 1;
__REG32 UENA                  : 1;
__REG32                       : 8;
__REG32 ISD                   :16;
} __spi_config_bits;

/* Slave Enable */
typedef struct{
__REG32 SENA                  : 6;
__REG32                       :26;
} __spi_slave_enable_bits;

/* TX_FIFO_FLUSH */
typedef struct{
__REG32 tx_fifo_flush         : 1;
__REG32                       :31;
} __spi_tx_fifo_flush_bits;

/* NHP_POP */
typedef struct{
__REG32 nhp_pop               : 1;
__REG32                       :31;
} __spi_nhp_pop_bits;

/* NHP_MODE */
typedef struct{
__REG32 nhp_mode              : 1;
__REG32                       :31;
} __spi_nhp_mode_bits;

/* DMA_SETTINGS */
typedef struct{
__REG32 rx_dma_enable         : 1;
__REG32 tx_dma_enable         : 1;
__REG32                       :30;
} __spi_dma_settings_bits;

/* STATUS */
typedef struct{
__REG32 tx_fifo_empty         : 1;
__REG32 tx_fifo_full          : 1;
__REG32 rx_fifo_empty         : 1;
__REG32 rx_fifo_full          : 1;
__REG32 spi_busy              : 1;
__REG32 sms_mode_busy         : 1;
__REG32                       :26;
} __spi_status_bits;

/* HW_INFO */
typedef struct{
__REG32 rx_fifo_depth         : 8;
__REG32 tx_fifo_depth         : 8;
__REG32 rx_fifo_width         : 5;
__REG32 tx_fifo_width         : 5;
__REG32 num_slaves            : 4;
__REG32 fifoimpl              : 1;
__REG32                       : 1;
} __spi_hw_info_bits;

/* SLVx_SETTINGS1 */
typedef struct{
__REG32 clk_divisor1          : 8;
__REG32 clk_divisor2          : 8;
__REG32 number_words          : 8;
__REG32 inter_transfer_dly    : 8;
} __spi_slv_settings1_bits;

/* SLVx_SETTINGS2 */
typedef struct{
__REG32 WS                    : 5;
__REG32 SPH                   : 1;
__REG32 SPO                   : 1;
__REG32 FORMAT                : 1;
__REG32 SC_VAL                : 1;
__REG32 CS_DLY                : 8;
__REG32                       :15;
} __spi_slv_settings2_bits;

/* INT_THRESHOLD */
typedef struct{
__REG32 rx_threshold          : 8;
__REG32 tx_threshold          : 8;
__REG32                       :16;
} __spi_int_threshold_bits;

/* INT_CLR_ENABLE */
typedef struct{
__REG32 clr_ov_int_enable     : 1;
__REG32 clr_to_int_enable     : 1;
__REG32 clr_rx_int_enable     : 1;
__REG32 clr_tx_int_enable     : 1;
__REG32 clr_sms_int_enable    : 1;
__REG32                       :27;
} __spi_int_clr_enable_bits;

/* INT_SET_ENABLE */
typedef struct{
__REG32 set_ov_int_enable     : 1;
__REG32 set_to_int_enable     : 1;
__REG32 set_rx_int_enable     : 1;
__REG32 set_tx_int_enable     : 1;
__REG32 set_sms_int_enable    : 1;
__REG32                       :27;
} __spi_int_set_enable_bits;

/* INT_STATUS */
typedef struct{
__REG32 ov_int_status         : 1;
__REG32 to_int_status         : 1;
__REG32 rx_int_status         : 1;
__REG32 tx_int_status         : 1;
__REG32 sms_int_status        : 1;
__REG32                       :27;
} __spi_int_status_bits;

/* INT_ENABLE */
typedef struct{
__REG32 ov_int_enable         : 1;
__REG32 to_int_enable         : 1;
__REG32 rx_int_enable         : 1;
__REG32 tx_int_enable         : 1;
__REG32 sms_int_enable        : 1;
__REG32                       :27;
} __spi_int_enable_bits;

/* INT_CLR_STATUS */
typedef struct{
__REG32 clr_ov_int_status     : 1;
__REG32 clr_to_int_status     : 1;
__REG32 clr_rx_int_status     : 1;
__REG32 clr_tx_int_status     : 1;
__REG32 clr_sms_int_status    : 1;
__REG32                       :27;
} __spi_int_clr_status_bits;

/* INT_SET_STATUS */
typedef struct{
__REG32 set_ov_int_status     : 1;
__REG32 set_to_int_status     : 1;
__REG32 set_rx_int_status     : 1;
__REG32 set_tx_int_status     : 1;
__REG32 set_sms_int_status    : 1;
__REG32                       :27;
} __spi_int_set_status_bits;

/* CTRL */
typedef struct{
__REG32 RST                   : 1;
__REG32 FIFO_RST              : 1;
__REG32 DMA_RST               : 1;
__REG32                       : 1;
__REG32 INT_ENA               : 1;
__REG32 DMA_ENA               : 1;
__REG32 READ_WAIT             : 1;
__REG32 SEND_RESP             : 1;
__REG32 ABORT_READ_DATA       : 1;
__REG32 SEND_CCSD             : 1;
__REG32 SEND_AUTO_STOP_CCSD   : 1;
__REG32 CE_ATA_INT            : 1;
__REG32                       :20;
} __mci_ctrl_bits;

/* CLKDIV */
typedef struct{
__REG32 CLK_DIV               : 8;
__REG32                       :24;
} __mci_clkdiv_bits;

/* CLKSRC */
typedef struct{
__REG32 CLK_SOURCE            : 2;
__REG32                       :30;
} __mci_clksrc_bits;

/* CLKENA */
typedef struct{
__REG32 CLK_ENA               : 1;
__REG32                       :15;
__REG32 CLK_LOW_PWR           : 1;
__REG32                       :15;
} __mci_clkena_bits;

/* TMOUT */
typedef struct{
__REG32 RESP_TO               : 8;
__REG32 DATA_TO               :24;
} __mci_tmout_bits;

/* CTYPE */
typedef struct{
__REG32 CARD_WIDTH_4B         : 1;
__REG32                       :15;
__REG32 CARD_WIDTH_8B         : 1;
__REG32                       :15;
} __mci_ctype_bits;

/* BLKSIZ */
typedef struct{
__REG32 Block_size            :16;
__REG32                       :16;
} __mci_blksiz_bits;

/* INTMASK */
typedef struct{
__REG32 CD                    : 1;
__REG32 RE                    : 1;
__REG32 CMDD                  : 1;
__REG32 DTO                   : 1;
__REG32 TXDR                  : 1;
__REG32 RXDR                  : 1;
__REG32 RCRC                  : 1;
__REG32 DCRC                  : 1;
__REG32 RTO                   : 1;
__REG32 DRTO                  : 1;
__REG32 HTO                   : 1;
__REG32 FRUN                  : 1;
__REG32 HLE                   : 1;
__REG32 SBE                   : 1;
__REG32 ACD                   : 1;
__REG32 EBE                   : 1;
__REG32 SDIO_INT_ENA          : 1;
__REG32                       :15;
} __mci_intmask_bits;

/* CMD */
typedef struct{
__REG32 CMD_INDX              : 6;
__REG32 RESP_EXPECT           : 1;
__REG32 RESP_LENGHT           : 1;
__REG32 RESP_CRC_ENA          : 1;
__REG32 DATA_TRAN             : 1;
__REG32 WRITE                 : 1;
__REG32 TM                    : 1;
__REG32 SEND_AUTO_STOP        : 1;
__REG32 WAIT_PRVDATA_COMP     : 1;
__REG32 STOP_ABORT_CMD        : 1;
__REG32 SEND_INIT             : 1;
__REG32                       : 5;
__REG32 UPDATE_CLK_ONLY       : 1;
__REG32 READ_CE_ATA           : 1;
__REG32 CCS_EXPECT            : 1;
__REG32                       : 7;
__REG32 START_CMD             : 1;
} __mci_cmd_bits;

/* STATUS */
typedef struct{
__REG32 fifo_rx_watermark     : 1;
__REG32 fifo_tx_watermark     : 1;
__REG32 fifo_empty            : 1;
__REG32 fifo_full             : 1;
__REG32 state                 : 4;
__REG32 data_3_status         : 1;
__REG32 data_busy             : 1;
__REG32 data_state_mc_busy    : 1;
__REG32 response_index        : 6;
__REG32 fifo_count            :13;
__REG32 dma_ack               : 1;
__REG32 Dma_req               : 1;
} __mci_status_bits;

/* FIFOTH */
typedef struct{
__REG32 TX_WMark              :12;
__REG32                       : 4;
__REG32 RX_WMark              :12;
__REG32 dma_mult_tran_size    : 3;
__REG32                       : 1;
} __mci_fifoth_bits;

/* CDETECT */
typedef struct{
__REG32 Card_detect_n         : 1;
__REG32                       :31;
} __mci_cdetect_bits;

/* WRTPRT */
typedef struct{
__REG32 Write_protect         : 1;
__REG32                       :31;
} __mci_wrtprt_bits;

/* Interrupt Enable Register (IER) */
typedef struct{
__REG32 RDAIntEn              : 1;
__REG32 THREIntEn             : 1;
__REG32 RLSIntEn              : 1;
__REG32 MSIntEn               : 1;
__REG32                       : 3;
__REG32 CTSIntEn              : 1;
__REG32                       :24;
} __uart_ier_bits;

/* Interrupt Identification Register (IIR) */
typedef union {
  /* UART_IIR*/
struct{
__REG32 IntStatus             : 1;
__REG32 IntId                 : 3;
__REG32                       : 2;
__REG32 FIFOEn                : 2;
__REG32                       :24;
};
  /* UART_FCR*/
struct{
__REG32 FIFOEnable            : 1;
__REG32 RxFIFORst             : 1;
__REG32 TxFIFORst             : 1;
__REG32 DMAMode               : 1;
__REG32                       : 2;
__REG32 RxTrigLevel           : 2;
__REG32                       :24;
};
} __uart_iir_bits;

/* Line Control Register (LCR) */
typedef struct{
__REG32 WdLenSel              : 2;
__REG32 StopBitNum            : 1;
__REG32 ParEn                 : 1;
__REG32 ParEven               : 1;
__REG32 ParStick              : 1;
__REG32 BrkCtrl               : 1;
__REG32 DLAB                  : 1;
__REG32                       :24;
} __uart_lcr_bits;

/* MCR (Modem Control Register) */
typedef struct{
__REG32                       : 1;
__REG32 RTS                   : 1;
__REG32                       : 2;
__REG32 LoopEn                : 1;
__REG32                       : 1;
__REG32 AutoRTSEn             : 1;
__REG32 AutoCTSEn             : 1;
__REG32                       :24;
} __uart_mcr_bits;

/* LSR (Line Status Register) */
typedef struct{
__REG32 DR                    : 1;
__REG32 OE                    : 1;
__REG32 PE                    : 1;
__REG32 FE                    : 1;
__REG32 BI                    : 1;
__REG32 THRE                  : 1;
__REG32 TEMT                  : 1;
__REG32 RxEr                  : 1;
__REG32                       :24;
} __uart_lsr_bits;

/* MSR (Modem Status Register) */
typedef struct{
__REG32 DCTS                  : 1;
__REG32                       : 3;
__REG32 CTS                   : 1;
__REG32                       :27;
} __uart_msr_bits;

/* ICR (IrDA Control Register) */
typedef struct{
__REG32 IrDAEn                : 1;
__REG32 IrDAInv               : 1;
__REG32 FixPulseEn            : 1;
__REG32 PulseDiv              : 3;
__REG32                       :26;
} __uart_icr_bits;

/* FDR (Fractional Divider Register) */
typedef struct{
__REG32 DivAddVal             : 4;
__REG32 MulVal                : 4;
__REG32                       :24;
} __uart_fdr_bits;

/* POP Register */
typedef struct{
__REG32 PopRBR                : 1;
__REG32                       :31;
} __uart_pop_bits;

/* Mode Selection Register */
typedef struct{
__REG32 NHP                   : 1;
__REG32                       :31;
} __uart_mode_bits;

/* INTCE (Interrupt Clear Enable Register) */
typedef struct{
__REG32 DCTSIntEnClr          : 1;
__REG32                       : 3;
__REG32 THREIntEnClr          : 1;
__REG32                       : 1;
__REG32 RxDAIntEnClr          : 1;
__REG32                       : 1;
__REG32 ABEOIntEnClr          : 1;
__REG32 ABTOIntEnClr          : 1;
__REG32                       : 2;
__REG32 BIIntEnClr            : 1;
__REG32 FEIntEnClr            : 1;
__REG32 PEIntEnClr            : 1;
__REG32 OEIntEnClr            : 1;
__REG32                       :16;
} __uart_intce_bits;

/* INTSE (Interrupt Set Enable Register) */
typedef struct{
__REG32 DCTSIntEnSet          : 1;
__REG32                       : 3;
__REG32 THREIntEnSet          : 1;
__REG32 RxTOIntEnSet          : 1;
__REG32 RxDAIntEnSet          : 1;
__REG32                       : 1;
__REG32 ABEOIntEnSet          : 1;
__REG32 ABTOIntEnSet          : 1;
__REG32                       : 2;
__REG32 BIIntEnSet            : 1;
__REG32 FEIntEnSet            : 1;
__REG32 PEIntEnSet            : 1;
__REG32 OEIntEnSet            : 1;
__REG32                       :16;
} __uart_intse_bits;

/* INTS (Interrupt Status Register) */
typedef struct{
__REG32 DCTSInt               : 1;
__REG32                       : 3;
__REG32 THREInt               : 1;
__REG32 RxTOInt               : 1;
__REG32 RxDAInt               : 1;
__REG32                       : 1;
__REG32 ABEOInt               : 1;
__REG32 ABTOInt               : 1;
__REG32                       : 2;
__REG32 BIInt                 : 1;
__REG32 FEInt                 : 1;
__REG32 PEInt                 : 1;
__REG32 OEInt                 : 1;
__REG32                       :16;
} __uart_ints_bits;

/* INTE (Interrupt Enable Register) */
typedef struct{
__REG32 DCTSIntEn             : 1;
__REG32                       : 3;
__REG32 THREIntEn             : 1;
__REG32 RxTOIntEn             : 1;
__REG32 RxDAIntEn             : 1;
__REG32                       : 1;
__REG32 ABEOIntEn             : 1;
__REG32 ABTOIntEn             : 1;
__REG32                       : 2;
__REG32 BIIntEn               : 1;
__REG32 FEIntEn               : 1;
__REG32 PEIntEn               : 1;
__REG32 OEIntEn               : 1;
__REG32                       :16;
} __uart_inte_bits;

/* INTCS (Interrupt Clear Status Register) */
typedef struct{
__REG32 DCTSIntClr            : 1;
__REG32                       : 3;
__REG32 THREIntClr            : 1;
__REG32 RxTOIntClr            : 1;
__REG32                       : 2;
__REG32 ABEOIntClr            : 1;
__REG32 ABTOIntClr            : 1;
__REG32                       : 5;
__REG32 OEIntClr              : 1;
__REG32                       :16;
} __uart_intcs_bits;

/* INTSS (Interrupt Set Status Register) */
typedef struct{
__REG32 DCTSIntSet            : 1;
__REG32                       : 3;
__REG32 THREIntSet            : 1;
__REG32 RxTOIntSet            : 1;
__REG32                       : 2;
__REG32 ABEOIntSet            : 1;
__REG32 ABTOIntSet            : 1;
__REG32                       : 5;
__REG32 OEIntSet              : 1;
__REG32                       :16;
} __uart_intss_bits;

/* LCD Interface Status Register */
typedef struct{
__REG32 LCD_INT_FIFO_EMPTY        : 1;
__REG32 LCD_INT_FIFO_HALF_EMPTY   : 1;
__REG32 LCD_INT_FIFO_OVERRUN      : 1;
__REG32 LCD_INT_READ_VALID        : 1;
__REG32 LCD_INTERFACE_BUSY        : 1;
__REG32 LCD_COUNTER               : 5;
__REG32                           :22;
} __lcd_status_bits;

/* LCD Control register */
typedef struct{
__REG32                           : 1;
__REG32 PS                        : 1;
__REG32 MI                        : 1;
__REG32 IF                        : 1;
__REG32 SERIAL_CLK_SHIFT          : 2;
__REG32 SERIAL_READ_POS           : 2;
__REG32 BUSY_FLAG_CHECK           : 1;
__REG32 BUSY_VALUE                : 1;
__REG32 BUSY_BIT_NR               : 4;
__REG32 BUSY_RS_VALUE             : 1;
__REG32 INVERT_CS                 : 1;
__REG32 INVERT_E_RD               : 1;
__REG32 MSB_FIRST                 : 1;
__REG32 LOOPBACK                  : 1;
__REG32 IF_16                     : 1;
__REG32 BYASYNC_RELCLK            : 1;
__REG32                           :11;
} __lcd_control_bits;

/* Interrupt raw register */
typedef struct{
__REG32 LCD_INT_FIFO_EMPTY_RAW      : 1;
__REG32 LCD_INT_FIFO_HALF_EMPTY_RAW : 1;
__REG32 LCD_INT_OVERRUN_RAW         : 1;
__REG32 LCD_INT_READ_VALID_RAW      : 1;
__REG32                             :28;
} __lcd_int_raw_bits;

/* Interrupt Clear register */
typedef struct{
__REG32 LCD_INT_FIFO_EMPTY_CLR      : 1;
__REG32 LCD_INT_FIFO_HALF_EMPTY_CLR : 1;
__REG32 LCD_INT_FIFO_OVERRUN_CLR    : 1;
__REG32 LCD_INT_READ_VALID_CLR      : 1;
__REG32                             :28;
} __lcd_int_clear_bits;

/* LCD Interface Interrupt Mask Register */
typedef struct{
__REG32 LCD_FIFO_EMPTY_MASK         : 1;
__REG32 LCD_FIFO_HALF_EMPTY_MASK    : 1;
__REG32 LCD_FIFO_OVERRUN_MASK       : 1;
__REG32 LCD_READ_VALID_MASK         : 1;
__REG32                             :28;
} __lcd_int_mask_bits;

/* LCD Interface Read Command Register */
typedef struct{
__REG32 LCD_READ_COMMAND            : 1;
__REG32                             :31;
} __lcd_read_cmd_bits;

/* LCD Interface Instruction Byte Register */
typedef struct{
__REG32 INSTRUCTION_BYTE            :16;
__REG32                             :16;
} __lcd_inst_byte_bits;

/* Data Byte Register */
typedef struct{
__REG32 DATA_BYTE                   :16;
__REG32                             :16;
} __lcd_data_byte_bits;

/* I2C RX/TX Data FIFO */
typedef union{
  /*I2Cx_RX*/
  struct {
__REG32 RxData  : 8;
__REG32         :24;
  };
  /*I2Cx_TX*/
  struct {
__REG32 TxData  : 8;
__REG32 START   : 1;
__REG32 STOP    : 1;
__REG32         :22;
  };
} __i2c_rx_tx_bits;

/* I2C Status Register */
typedef struct{
__REG32 TDI     : 1;
__REG32 AFI     : 1;
__REG32 NAI     : 1;
__REG32 DRMI    : 1;
__REG32 DRSI    : 1;
__REG32 ACTIVE  : 1;
__REG32 SCL     : 1;
__REG32 SDA     : 1;
__REG32 RFF     : 1;
__REG32 RFE     : 1;
__REG32 TFF     : 1;
__REG32 TFE     : 1;
__REG32 TFFS    : 1;
__REG32 TFES    : 1;
__REG32         :18;
} __i2c_sts_bits;

/* I2C Control Register */
typedef struct{
__REG32 TDIE    : 1;
__REG32 AFIE    : 1;
__REG32 NAIE    : 1;
__REG32 DRMIE   : 1;
__REG32 DRSIE   : 1;
__REG32 RFFIE   : 1;
__REG32 RFDAIE  : 1;
__REG32 TFFIE   : 1;
__REG32 RESET   : 1;
__REG32 SEVEN   : 1;
__REG32 TFFSIE  : 1;
__REG32         :21;
} __i2c_ctrl_bits;

/* I2C Clock Divider High */
typedef struct{
__REG32 CLKHI       :10;
__REG32             :22;
} __i2c_clk_hi_bits;

/* I2C Clock Divider Low */
typedef struct{
__REG32 CLKLO       :10;
__REG32             :22;
} __i2c_clk_lo_bits;

/* I2C ADR register */
typedef struct{
__REG32 ADDR        :10;
__REG32             :22;
} __i2c_adr_bits;

/* I2Cn Receive FIFO level register */
typedef struct{
__REG32 RxFL        : 2;
__REG32             :30;
} __i2c_rfl_bits;

/* I2Cn Transmit FIFO level register */
typedef struct{
__REG32 TxFL        : 2;
__REG32             :30;
} __i2c_tfl_bits;

/* I2Cn RX byte count register */
typedef struct{
__REG32 RxB         :16;
__REG32             :16;
} __i2c_rxb_bits;

/* I2Cn TX byte count register */
typedef struct{
__REG32 TxB         :16;
__REG32             :16;
} __i2c_txb_bits;

/* II2Cn Slave TX FIFO level register */
typedef struct{
__REG32 TxFL        : 2;
__REG32             :30;
} __i2c_stfl_bits;

/* Timer Control (TimerCtrl) Register */
typedef struct{
__REG32             : 2;
__REG32 PreScale    : 2;
__REG32             : 2;
__REG32 Mode        : 1;
__REG32 Enable      : 1;
__REG32             :24;
} __timerctrl_bits;

/* PWM_TMR */
typedef struct{
__REG32 MR          :12;
__REG32             :20;
} __pwm_tmr_bits;

/* PWM_CNTL */
typedef struct{
__REG32 CLK         : 2;
__REG32             : 2;
__REG32 HI          : 1;
__REG32             : 1;
__REG32 LOOP        : 1;
__REG32 PDM         : 1;
__REG32             :24;
} __pwm_cntl_bits;

/* SYSCREG_EBI_MPMC_PRIO Register */
typedef struct{
__REG32 TIMEOUTVALUE      :10;
__REG32                   :22;
} __syscreg_ebi_mpmc_prio_bits;

/* RING_OSC_CFG Register */
typedef struct{
__REG32 ccp_ring_osc_cfg_osc0_en  : 1;
__REG32 ccp_ring_osc_cfg_osc1_en  : 1;
__REG32                           :30;
} __syscreg_ring_osc_cfg_bits;

/* SYSCREG_SSA1_ADC_PD_ADC10BITS Register */
typedef struct{
__REG32 ssa1_adc_pd_adc10bits     : 1;
__REG32                           :31;
} __syscreg_adc_pd_adc10bits_bits;

/* SYSCREG_ABC_CFG Register */
typedef struct{
__REG32 simple_dma                : 3;
__REG32 arm926ejs_d               : 3;
__REG32 arm926ejs_i               : 3;
__REG32 usb_otg                   : 3;
__REG32                           :20;
} __syscreg_abc_cfg_bits;

/* SYSCREG_SD_MMC_CFG Register */
typedef struct{
__REG32 card_write_prt            : 1;
__REG32 card_detect_n             : 1;
__REG32                           :30;
} __syscreg_sd_mmc_cfg_bits;

/* SYSCREG_MCI_DELAYMODES Register */
typedef struct{
__REG32 delay_cells               : 4;
__REG32 delay_enable              : 1;
__REG32                           :27;
} __syscreg_mci_delaymodes_bits;

/* USB_ATX_PLL_PD_REG Register */
typedef struct{
__REG32 USB_ATX_PLL_PD_REG        : 1;
__REG32                           :31;
} __syscreg_usb_atx_pll_pd_reg_bits;

/* USB_OTG_CFG Register */
typedef struct{
__REG32                           : 1;
__REG32 usb_otg_host_wakeup_n     : 1;
__REG32 usb_otg_dev_wakeup_n      : 1;
__REG32 usb_otg_vbus_pwr_fault    : 1;
__REG32                           :28;
} __syscreg_usb_otg_cfg_bits;

/* USB_OTG_PORT_IND_CTL Register */
typedef struct{
__REG32 USB_OTG_PORT_IND_CTL      : 2;
__REG32                           :30;
} __syscreg_usb_otg_port_ind_ctl_bits;

/* USB_PLL_NDEC Register */
typedef struct{
__REG32 USB_PLL_NDEC              :10;
__REG32                           :22;
} __syscreg_usb_pll_ndec_bits;

/* USB_PLL_MDEC Register */
typedef struct{
__REG32 USB_PLL_MDEC              :17;
__REG32                           :15;
} __syscreg_usb_pll_mdec_bits;

/* USB_PLL_PDEC Register */
typedef struct{
__REG32 USB_PLL_PDEC              : 4;
__REG32                           :28;
} __syscreg_usb_pll_pdec_bits;

/* USB_PLL_SELR Register */
typedef struct{
__REG32 USB_PLL_SELR              : 4;
__REG32                           :28;
} __syscreg_usb_pll_selr_bits;

/* USB_PLL_SELI Register */
typedef struct{
__REG32 USB_PLL_SELI              : 4;
__REG32                           :28;
} __syscreg_usb_pll_seli_bits;

/* USB_PLL_SELP Register */
typedef struct{
__REG32 USB_PLL_SELP              : 4;
__REG32                           :28;
} __syscreg_usb_pll_selp_bits;

/* SYSCREG_ISRAM0/1_LATENCY_CFG Register */
typedef struct{
__REG32 Isram_latency_cfg         : 2;
__REG32                           :30;
} __syscreg_isram_latency_cfg_bits;

/* SYSCREG_ISROM_LATENCY_CFG Register */
typedef struct{
__REG32 Isrom_latency_cfg         : 2;
__REG32                           :30;
} __syscreg_isrom_latency_cfg_bits;

/* SYSCREG_AHB_MPMC_MISC Register */
typedef struct{
__REG32 ahb_mpmc_misc_srefreq         : 1;
__REG32                               : 2;
__REG32 ahb_mpmc_misc_stcs0pol        : 1;
__REG32 ahb_mpmc_misc_stcs1pol        : 1;
__REG32                               : 2;
__REG32 ahb_mpmc_misc_stcs1pb         : 1;
__REG32 ahb_mpmc_misc_rel1config      : 1;
__REG32                               :23;
} __syscreg_ahb_mpmc_misc_bits;

/* SYSCREG_MPMP_DELAYMODES Register */
typedef struct{
__REG32 MPMC_delaymodes0              : 6;
__REG32 MPMC_delaymodes1              : 6;
__REG32 MPMC_delaymodes2              : 6;
__REG32                               :14;
} __syscreg_mpmp_delaymodes_bits;

/* SYSCREG_MPMC_WAITREAD_DELAY0/1 Register */
typedef struct{
__REG32 STATIC_WAIT_CNTR              : 5;
__REG32 EXTRA_OE_INACT                : 1;
__REG32                               :26;
} __syscreg_mpmc_waitread_delay_bits;

/* SYSCREG_WIRE_EBI_MSIZE_INIT Register */
typedef struct{
__REG32 wire_ebi_msize_init           : 2;
__REG32                               :30;
} __syscreg_wire_ebi_msize_init_bits;

/* MPMC_TESTMODE0 Register */
typedef struct{
__REG32 ERC                           :12;
__REG32 ERE                           : 1;
__REG32                               :19;
} __syscreg_mpmc_testmode0_bits;

/* MPMC_TESTMODE1 Register */
typedef struct{
__REG32 HSECERG                       : 8;
__REG32                               :24;
} __syscreg_mpmc_testmode1_bits;

/* AHB0_EXTPRIO Register */
typedef struct{
__REG32 SDMA                          : 1;
__REG32 ARM926_I                      : 1;
__REG32 ARM926_D                      : 1;
__REG32 USB_OTG                       : 1;
__REG32                               :28;
} __syscreg_ahb0_extprio_bits;

/* SYSCREG_MUX_LCD_EBI_SEL Register */
typedef struct{
__REG32 Mux_LCD_EBI_sel               : 1;
__REG32                               :31;
} __syscreg_mux_lcd_ebi_sel_bits;

/* SYSCREG_MUX_GPIO_MCI_SEL Register */
typedef struct{
__REG32 Mux_GPIO_MCI_sel              : 1;
__REG32                               :31;
} __syscreg_mux_gpio_mci_sel_bits;

/* SYSCREG_MUX_NAND_MCI_SEL Register */
typedef struct{
__REG32 Mux_NAND_MCI_sel              : 1;
__REG32                               :31;
} __syscreg_mux_nand_mci_sel_bits;

/* SYSCREG_MUX_NAND_MCI_SEL Register */
typedef struct{
__REG32 Mux_UART_SPI_sel              : 1;
__REG32                               :31;
} __syscreg_mux_uart_spi_sel_bits;

/* SYSCREG_MUX_I2STX_PCM_SEL Register */
typedef struct{
__REG32 Mux_I2STX_0_PCM_sel             : 1;
__REG32                               :31;
} __syscreg_mux_i2stx_pcm_sel_bits;

/* SYSCREG_xxx_PCTRL Register */
typedef struct{
__REG32 P1                            : 1;
__REG32 P2                            : 1;
__REG32                               :30;
} __syscreg_pctrl_bits;

/* SYSCREG_ESHCTRL_SUP4 Register */
typedef struct{
__REG32 SYSCREG_ESHCTRL_SUP4          : 1;
__REG32                               :31;
} __syscreg_eshctrl_sup4_bits;

/* SYSCREG_ESHCTRL_SUP8 Register */
typedef struct{
__REG32 SYSCREG_ESHCTRL_SUP8          : 1;
__REG32                               :31;
} __syscreg_eshctrl_sup8_bits;

/* GLOBAL register */
typedef struct{
__REG32 ON_OFF                        : 1;
__REG32                               : 1;
__REG32 NORMAL                        : 1;
__REG32 DMATXENABLE                   : 1;
__REG32 DMARXENABLE                   : 1;
__REG32                               :27;
} __pcm_iom_global_bits;

/* CNTL0 register */
typedef struct{
__REG32                               : 3;
__REG32 CLK_SPD                       : 3;
__REG32 TYP_FRMSYNC                   : 2;
__REG32 TYP_DO_IP                     : 2;
__REG32 TYP_OD                        : 1;
__REG32 LOOPBACK                      : 1;
__REG32                               : 2;
__REG32 MASTER                        : 1;
__REG32                               :17;
} __pcm_iom_cntl0_bits;

/* CNTL1 register */
typedef struct{
__REG32 ENSLT                         :12;
__REG32                               :20;
} __pcm_iom_cntl1_bits;

/* CNTL2 register */
typedef struct{
__REG32 SLOTDIRINV                    :12;
__REG32                               :20;
} __pcm_iom_cntl2_bits;

/* I2S_FORMAT_SETTINGS register */
typedef struct{
__REG32 I2STX0_format                 : 3;
__REG32 I2STX1_format                 : 3;
__REG32 I2SRX0_format                 : 3;
__REG32 I2SRX1_format                 : 3;
__REG32                               :20;
} __i2s_format_settings_bits;

/* AUDIOMSS_MUX_SETTINGS register */
typedef struct{
__REG32                               : 1;
__REG32 I2SRX0_oe_n                   : 1;
__REG32 I2SRX1_oe_n                   : 1;
__REG32                               :29;
} __audioss_mux_settings_bits;

/* N_SOF_COUNTER register */
typedef struct{
__REG32 n_sof_counter                 : 8;
__REG32                               :24;
} __n_sof_counter_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NAND
 **
 ***************************************************************************/
__IO_REG32_BIT(NandIRQStatus1,        0x17000800,__READ_WRITE,__nandirqstatus_bits);
__IO_REG32_BIT(NandIRQMask1,          0x17000804,__READ_WRITE,__nandirqmask_bits);
__IO_REG32_BIT(NandIRQStatusRaw1,     0x17000808,__READ_WRITE,__nandirqstatusraw_bits);
__IO_REG32_BIT(NandConfig,            0x1700080C,__READ_WRITE,__nandconfig_bits);
__IO_REG32_BIT(NandIOConfig,          0x17000810,__READ_WRITE,__nandioconfig_bits);
__IO_REG32_BIT(NandTiming1,           0x17000814,__READ_WRITE,__nandtiming1_bits);
__IO_REG32_BIT(NandTiming2,           0x17000818,__READ_WRITE,__nandtiming2_bits);
__IO_REG32_BIT(NandSetCmd,            0x17000820,__READ_WRITE,__nandsetcmd_bits);
__IO_REG32_BIT(NandSetAddr,           0x17000824,__READ_WRITE,__nandsetaddr_bits);
__IO_REG32_BIT(NandWriteData,         0x17000828,__READ_WRITE,__nandwritedata_bits);
__IO_REG32_BIT(NandSetCE,             0x1700082C,__READ_WRITE,__nandsetce_bits);
__IO_REG32_BIT(NandReadData,          0x17000830,__READ      ,__nandreaddata_bits);
__IO_REG32_BIT(NandCheckSTS,          0x17000834,__READ      ,__nandchecksts_bits);
__IO_REG32_BIT(NandControlFlow,       0x17000838,__WRITE     ,__nandcontrolflow_bits);
__IO_REG32_BIT(NandGPIO1,             0x17000840,__READ_WRITE,__nandgpio1_bits);
__IO_REG32_BIT(NandGPIO2,             0x17000844,__READ      ,__nandgpio2_bits);
__IO_REG32_BIT(NandIRQStatus2,        0x17000848,__READ_WRITE,__nandirqstatus2_bits);
__IO_REG32_BIT(NandIRQMask2,          0x1700084C,__READ_WRITE,__nandirqmask2_bits);
__IO_REG32_BIT(NandIRQStatusRaw2,     0x17000850,__READ_WRITE,__nandirqstatusraw2_bits);
__IO_REG32(    NandAESKey1,           0x17000854,__WRITE     );
__IO_REG32(    NandAESKey2,           0x17000858,__WRITE     );
__IO_REG32(    NandAESKey3,           0x1700085C,__WRITE     );
__IO_REG32(    NandAESKey4,           0x17000860,__WRITE     );
__IO_REG32(    NandAESIV1,            0x17000864,__WRITE     );
__IO_REG32(    NandAESIV2,            0x17000868,__WRITE     );
__IO_REG32(    NandAESIV3,            0x1700086C,__WRITE     );
__IO_REG32(    NandAESIV4,            0x17000870,__WRITE     );
__IO_REG32_BIT(NandAESState,          0x17000874,__READ_WRITE ,__nandaesstate_bits);
__IO_REG32_BIT(NandECCErrStatus,      0x17000878,__READ       ,__nandeccerrstatus_bits);
__IO_REG32_BIT(AES_FROM_AHB,          0x1700087C,__READ_WRITE ,__aes_from_ahb_bits);

/***************************************************************************
 **
 ** MPMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MPMCControl,           0x17008000,__READ_WRITE,__mpmc_ctrl_bits);
__IO_REG32_BIT(MPMCStatus,            0x17008004,__READ      ,__mpmc_status_bits);
__IO_REG32_BIT(MPMCConfig,            0x17008008,__READ_WRITE,__mpmc_cfg_bits);
__IO_REG32_BIT(MPMCDynamicControl,    0x17008020,__READ_WRITE,__mpmcd_ctrl_bits);
__IO_REG32_BIT(MPMCDynamicRefresh,    0x17008024,__READ_WRITE,__mpmcd_refresh_bits);
__IO_REG32_BIT(MPMCDynamicReadConfig, 0x17008028,__READ_WRITE,__mpmcd_read_cfg_bits);
__IO_REG32_BIT(MPMCDynamictRP,        0x17008030,__READ_WRITE,__mpmcd_trp_bits);
__IO_REG32_BIT(MPMCDynamictRAS,       0x17008034,__READ_WRITE,__mpmcd_tras_bits);
__IO_REG32_BIT(MPMCDynamictSREX,      0x17008038,__READ_WRITE,__mpmcd_tsrex_bits);
__IO_REG32_BIT(MPMCDynamictAPR,       0x1700803C,__READ_WRITE,__mpmcd_apr_bits);
__IO_REG32_BIT(MPMCDynamictDAL,       0x17008040,__READ_WRITE,__mpmcd_dal_bits);
__IO_REG32_BIT(MPMCDynamictWR,        0x17008044,__READ_WRITE,__mpmcd_twr_bits);
__IO_REG32_BIT(MPMCDynamictRC,        0x17008048,__READ_WRITE,__mpmcd_trc_bits);
__IO_REG32_BIT(MPMCDynamictRFC,       0x1700804C,__READ_WRITE,__mpmcd_trfc_bits);
__IO_REG32_BIT(MPMCDynamictXSR,       0x17008050,__READ_WRITE,__mpmcd_txsr_bits);
__IO_REG32_BIT(MPMCDynamictRRD,       0x17008054,__READ_WRITE,__mpmcd_trrd_bits);
__IO_REG32_BIT(MPMCDynamictMRD,       0x17008058,__READ_WRITE,__mpmcd_tmrd_bits);
__IO_REG32_BIT(MPMCStaticExtendedWait,0x17008080,__READ_WRITE,__mpmcs_ext_wait_bits);
__IO_REG32_BIT(MPMCDynamicConfig0,    0x17008100,__READ_WRITE,__mpmcd_cfg_bits);
__IO_REG32_BIT(MPMCDynamicRasCas0,    0x17008104,__READ_WRITE,__mpmcd_ras_cas_bits);
__IO_REG32_BIT(MPMCStaticConfig0,     0x17008200,__READ_WRITE,__mpmcs_cnfg_bits);
__IO_REG32_BIT(MPMCStaticWaitWen0,    0x17008204,__READ_WRITE,__mpmcs_wen_bits);
__IO_REG32_BIT(MPMCStaticWaitOen0,    0x17008208,__READ_WRITE,__mpmcs_oen_bits);
__IO_REG32_BIT(MPMCStaticWaitRd0,     0x1700820C,__READ_WRITE,__mpmcs_rd_bits);
__IO_REG32_BIT(MPMCStaticWaitPage0,   0x17008210,__READ_WRITE,__mpmcs_page_bits);
__IO_REG32_BIT(MPMCStaticWaitWr0,     0x17008214,__READ_WRITE,__mpmcs_wr_bits);
__IO_REG32_BIT(MPMCStaticWaitTurn0,   0x17008218,__READ_WRITE,__mpmcs_turn_bits);
__IO_REG32_BIT(MPMCStaticConfig1,     0x17008220,__READ_WRITE,__mpmcs_cnfg_bits);
__IO_REG32_BIT(MPMCStaticWaitWen1,    0x17008224,__READ_WRITE,__mpmcs_wen_bits);
__IO_REG32_BIT(MPMCStaticWaitOen1,    0x17008228,__READ_WRITE,__mpmcs_oen_bits);
__IO_REG32_BIT(MPMCStaticWaitRd1,     0x1700822C,__READ_WRITE,__mpmcs_rd_bits);
__IO_REG32_BIT(MPMCStaticWaitPage1,   0x17008230,__READ_WRITE,__mpmcs_page_bits);
__IO_REG32_BIT(MPMCStaticWaitWr1,     0x17008234,__READ_WRITE,__mpmcs_wr_bits);
__IO_REG32_BIT(MPMCStaticWaitTurn1,   0x17008238,__READ_WRITE,__mpmcs_turn_bits);

/***************************************************************************
 **
 ** OTG USB 2.0
 **
 ***************************************************************************/
__IO_REG16_BIT(OTG_CAPLENGTH_REG,     0x19000100,__READ      ,__otg_caplength_reg_bits);
__IO_REG16(    OTG_HCIVERSION_REG,    0x19000102,__READ      );
__IO_REG32_BIT(OTG_HCSPARAMS_REG,     0x19000104,__READ      ,__otg_hcsparams_reg_bits);
__IO_REG32_BIT(OTG_HCCPARAMS_REG,     0x19000108,__READ      ,__otg_hccparams_reg_bits);
__IO_REG16(    OTG_DCIVERSION_REG,    0x19000120,__READ      );
__IO_REG32_BIT(OTG_DCCPARAMS_REG,     0x19000124,__READ      ,__otg_dccparams_reg_bits);
__IO_REG32_BIT(OTG_USBCMD_REG,        0x19000140,__READ_WRITE,__otg_usbcmd_reg_bits);
__IO_REG32_BIT(OTG_USBSTS_REG,        0x19000144,__READ_WRITE,__otg_usbsts_reg_bits);
__IO_REG32_BIT(OTG_USBINTR_REG,       0x19000148,__READ_WRITE,__otg_usbintr_reg_bits);
__IO_REG16(    OTG_FRINDEX_REG,       0x1900014C,__READ_WRITE);
__IO_REG32_BIT(OTG_PERIODICLISTBASE_REG,0x19000154,__READ_WRITE,__otg_periodiclistbase_reg_bits);
#define OTG_DEVICEADDR_REG      OTG_PERIODICLISTBASE_REG
#define OTG_DEVICEADDR_REG_bit  OTG_PERIODICLISTBASE_REG_bit
__IO_REG32_BIT(OTG_ASYNCLISTADDR_REG, 0x19000158,__READ_WRITE,__otg_asynclistaddr_reg_bits);
#define OTG_ENDPOINTLISTADDR_REG      OTG_ASYNCLISTADDR_REG
#define OTG_ENDPOINTLISTADDR_REG_bit  OTG_ASYNCLISTADDR_REG_bit
__IO_REG32_BIT(OTG_TTCTRL_REG,        0x1900015C,__READ_WRITE,__otg_ttctrl_reg_bits);
__IO_REG32_BIT(OTG_BURSTSIZE_REG,     0x19000160,__READ_WRITE,__otg_burstsize_reg_bits);
__IO_REG32_BIT(OTG_TXFILLTUNING_REG,  0x19000164,__READ_WRITE,__otg_txfilltuning_reg_bits);
__IO_REG32_BIT(OTG_BINTERVAL,         0x19000174,__READ_WRITE,__otg_binterval_bit);
__IO_REG32_BIT(OTG_ENDPTNAK_REG,      0x19000178,__READ_WRITE,__otg_endptnak_reg_bits);
__IO_REG32_BIT(OTG_ENDPTNAKEN_REG,    0x1900017C,__READ_WRITE,__otg_endptnaken_reg_bits);
__IO_REG32_BIT(OTG_PORTSC1_REG,       0x19000184,__READ_WRITE,__otg_portsc1_reg_bits);
__IO_REG32_BIT(OTG_OTGSC_REG,         0x190001A4,__READ_WRITE,__otg_otgsc_reg_bits);
__IO_REG32_BIT(OTG_USBMODE_REG,       0x190001A8,__READ_WRITE,__otg_usbmode_reg_bits);
__IO_REG32_BIT(OTG_ENDPTSETUPSTAT_REG,0x190001AC,__READ_WRITE,__otg_endptsetupstat_reg_bits);
__IO_REG32_BIT(OTG_ENDPTPRIME_REG,    0x190001B0,__READ_WRITE,__otg_endptprime_reg_bits);
__IO_REG32_BIT(OTG_ENDPTFLUSH_REG,    0x190001B4,__READ_WRITE,__otg_endptflush_reg_bits);
__IO_REG32_BIT(OTG_ENDPTSTATUS_REG,   0x190001B8,__READ      ,__otg_endptstatus_reg_bits);
__IO_REG32_BIT(OTG_ENDPTCOMPLETE_REG, 0x190001BC,__READ_WRITE,__otg_endptcomplete_reg_bits);
__IO_REG32_BIT(OTG_ENDPTCTRL0_REG,    0x190001C0,__READ_WRITE,__otg_endptctrl0_reg_bits);
__IO_REG32_BIT(OTG_ENDPTCTRL1_REG,    0x190001C4,__READ_WRITE,__otg_endptctrl_reg_bits);
__IO_REG32_BIT(OTG_ENDPTCTRL2_REG,    0x190001C8,__READ_WRITE,__otg_endptctrl_reg_bits);
__IO_REG32_BIT(OTG_ENDPTCTRL3_REG,    0x190001CC,__READ_WRITE,__otg_endptctrl_reg_bits);

/***************************************************************************
 **
 ** DMA
 **
 ***************************************************************************/
__IO_REG32(    DMA0_SA,               0x17000000,__READ_WRITE);
__IO_REG32(    DMA0_DA,               0x17000004,__READ_WRITE);
__IO_REG32_BIT(DMA0_TL,               0x17000008,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA0_CFG,              0x1700000C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA0_ENA,              0x17000010,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA0_TC,               0x1700001C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA1_SA,               0x17000020,__READ_WRITE);
__IO_REG32(    DMA1_DA,               0x17000024,__READ_WRITE);
__IO_REG32_BIT(DMA1_TL,               0x17000028,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA1_CFG,              0x1700002C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA1_ENA,              0x17000030,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA1_TC,               0x1700003C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA2_SA,               0x17000040,__READ_WRITE);
__IO_REG32(    DMA2_DA,               0x17000044,__READ_WRITE);
__IO_REG32_BIT(DMA2_TL,               0x17000048,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA2_CFG,              0x1700004C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA2_ENA,              0x17000050,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA2_TC,               0x1700005C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA3_SA,               0x17000060,__READ_WRITE);
__IO_REG32(    DMA3_DA,               0x17000064,__READ_WRITE);
__IO_REG32_BIT(DMA3_TL,               0x17000068,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA3_CFG,              0x1700006C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA3_ENA,              0x17000070,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA3_TC,               0x1700007C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA4_SA,               0x17000080,__READ_WRITE);
__IO_REG32(    DMA4_DA,               0x17000084,__READ_WRITE);
__IO_REG32_BIT(DMA4_TL,               0x17000088,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA4_CFG,              0x1700008C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA4_ENA,              0x17000090,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA4_TC,               0x1700009C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA5_SA,               0x170000A0,__READ_WRITE);
__IO_REG32(    DMA5_DA,               0x170000A4,__READ_WRITE);
__IO_REG32_BIT(DMA5_TL,               0x170000A8,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA5_CFG,              0x170000AC,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA5_ENA,              0x170000B0,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA5_TC,               0x170000BC,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA6_SA,               0x170000C0,__READ_WRITE);
__IO_REG32(    DMA6_DA,               0x170000C4,__READ_WRITE);
__IO_REG32_BIT(DMA6_TL,               0x170000C8,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA6_CFG,              0x170000CC,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA6_ENA,              0x170000D0,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA6_TC,               0x170000DC,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA7_SA,               0x170000E0,__READ_WRITE);
__IO_REG32(    DMA7_DA,               0x170000E4,__READ_WRITE);
__IO_REG32_BIT(DMA7_TL,               0x170000E8,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA7_CFG,              0x170000EC,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA7_ENA,              0x170000F0,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA7_TC,               0x170000FC,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA8_SA,               0x17000100,__READ_WRITE);
__IO_REG32(    DMA8_DA,               0x17000104,__READ_WRITE);
__IO_REG32_BIT(DMA8_TL,               0x17000108,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA8_CFG,              0x1700010C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA8_ENA,              0x17000110,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA8_TC,               0x1700011C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA9_SA,               0x17000120,__READ_WRITE);
__IO_REG32(    DMA9_DA,               0x17000124,__READ_WRITE);
__IO_REG32_BIT(DMA9_TL,               0x17000128,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA9_CFG,              0x1700012C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA9_ENA,              0x17000130,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA9_TC,               0x1700013C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA10_SA,              0x17000140,__READ_WRITE);
__IO_REG32(    DMA10_DA,              0x17000144,__READ_WRITE);
__IO_REG32_BIT(DMA10_TL,              0x17000148,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA10_CFG,             0x1700014C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA10_ENA,             0x17000150,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA10_TC,              0x1700015C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA11_SA,              0x17000160,__READ_WRITE);
__IO_REG32(    DMA11_DA,              0x17000164,__READ_WRITE);
__IO_REG32_BIT(DMA11_TL,              0x17000168,__READ_WRITE,__dma_tl_bits);
__IO_REG32_BIT(DMA11_CFG,             0x1700016C,__READ_WRITE,__dma_cfg_bits);
__IO_REG32_BIT(DMA11_ENA,             0x17000170,__READ_WRITE,__dma_ena_bits);
__IO_REG32_BIT(DMA11_TC,              0x1700017C,__READ_WRITE,__dma_tc_bits);
__IO_REG32(    DMA0_ASA,              0x17000200,__WRITE     );
__IO_REG32(    DMA0_ADA,              0x17000204,__WRITE     );
__IO_REG32_BIT(DMA0_ATL,              0x17000208,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA0_ACFG,             0x1700020C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA1_ASA,              0x17000210,__WRITE     );
__IO_REG32(    DMA1_ADA,              0x17000214,__WRITE     );
__IO_REG32_BIT(DMA1_ATL,              0x17000218,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA1_ACFG,             0x1700021C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA2_ASA,              0x17000220,__WRITE     );
__IO_REG32(    DMA2_ADA,              0x17000224,__WRITE     );
__IO_REG32_BIT(DMA2_ATL,              0x17000228,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA2_ACFG,             0x1700022C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA3_ASA,              0x17000230,__WRITE     );
__IO_REG32(    DMA3_ADA,              0x17000234,__WRITE     );
__IO_REG32_BIT(DMA3_ATL,              0x17000238,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA3_ACFG,             0x1700023C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA4_ASA,              0x17000240,__WRITE     );
__IO_REG32(    DMA4_ADA,              0x17000244,__WRITE     );
__IO_REG32_BIT(DMA4_ATL,              0x17000248,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA4_ACFG,             0x1700024C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA5_ASA,              0x17000250,__WRITE     );
__IO_REG32(    DMA5_ADA,              0x17000254,__WRITE     );
__IO_REG32_BIT(DMA5_ATL,              0x17000258,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA5_ACFG,             0x1700025C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA6_ASA,              0x17000260,__WRITE     );
__IO_REG32(    DMA6_ADA,              0x17000264,__WRITE     );
__IO_REG32_BIT(DMA6_ATL,              0x17000268,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA6_ACFG,             0x1700026C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA7_ASA,              0x17000270,__WRITE     );
__IO_REG32(    DMA7_ADA,              0x17000274,__WRITE     );
__IO_REG32_BIT(DMA7_ATL,              0x17000278,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA7_ACFG,             0x1700027C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA8_ASA,              0x17000280,__WRITE     );
__IO_REG32(    DMA8_ADA,              0x17000284,__WRITE     );
__IO_REG32_BIT(DMA8_ATL,              0x17000288,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA8_ACFG,             0x1700028C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA9_ASA,              0x17000290,__WRITE     );
__IO_REG32(    DMA9_ADA,              0x17000294,__WRITE     );
__IO_REG32_BIT(DMA9_ATL,              0x17000298,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA9_ACFG,             0x1700029C,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA10_ASA,             0x170002A0,__WRITE     );
__IO_REG32(    DMA10_ADA,             0x170002A4,__WRITE     );
__IO_REG32_BIT(DMA10_ATL,             0x170002A8,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA10_ACFG,            0x170002AC,__WRITE     ,__dma_cfg_bits);
__IO_REG32(    DMA11_ASA,             0x170002B0,__WRITE     );
__IO_REG32(    DMA11_ADA,             0x170002B4,__WRITE     );
__IO_REG32_BIT(DMA11_ATL,             0x170002B8,__WRITE     ,__dma_tl_bits);
__IO_REG32_BIT(DMA11_ACFG,            0x170002BC,__WRITE     ,__dma_cfg_bits);
__IO_REG32_BIT(DMA_ALT_ENABLE,        0x17000400,__READ_WRITE,__dma_alt_enable_bits);
__IO_REG32_BIT(DMA_IRQ_STATUS_CLEAR,  0x17000404,__READ_WRITE,__dma_irq_status_clear_bits);
__IO_REG32_BIT(DMA_IRQ_MASK,          0x17000408,__READ_WRITE,__dma_irq_mask_bits);
__IO_REG32(    DMA_TEST_FIFO_RESP_STATUS,0x1700040C,__READ   );
__IO_REG32_BIT(DMA_SOFT_INT,          0x17000410,__WRITE     ,__dma_soft_int_bits);

/***************************************************************************
 **
 ** INTC
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_PRIORITYMASK_0,    0x60000000,__READ_WRITE,__int_prioritymask_bits);
__IO_REG32_BIT(INT_PRIORITYMASK_1,    0x60000004,__READ_WRITE,__int_prioritymask_bits);
__IO_REG32_BIT(INT_VECTOR_0,          0x60000100,__READ_WRITE,__int_vector_bits);
__IO_REG32_BIT(INT_VECTOR_1,          0x60000104,__READ_WRITE,__int_vector_bits);
__IO_REG32_BIT(INT_PENDING_1_31,      0x60000200,__READ_WRITE,__int_pending_1_31_bits);
__IO_REG32_BIT(INT_FEATURES,          0x60000300,__READ_WRITE,__int_features_bits);
__IO_REG32_BIT(INT_REQUEST_1,         0x60000404,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_2,         0x60000408,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_3,         0x6000040C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_4,         0x60000410,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_5,         0x60000414,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_6,         0x60000418,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_7,         0x6000041C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_8,         0x60000420,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_9,         0x60000424,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_10,        0x60000428,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_11,        0x6000042C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_12,        0x60000430,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_13,        0x60000434,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_14,        0x60000438,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_15,        0x6000043C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_16,        0x60000440,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_17,        0x60000444,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_18,        0x60000448,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_19,        0x6000044C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_20,        0x60000450,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_21,        0x60000454,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_22,        0x60000458,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_23,        0x6000045C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_24,        0x60000460,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_25,        0x60000464,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_26,        0x60000468,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_27,        0x6000046C,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_28,        0x60000470,__READ_WRITE,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_29,        0x60000474,__READ_WRITE,__int_request_bits);

/***************************************************************************
 **
 ** CGU
 **
 ***************************************************************************/
__IO_REG32_BIT(SCR0,                  0x13004000,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR1,                  0x13004004,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR2,                  0x13004008,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR3,                  0x1300400C,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR4,                  0x13004010,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR5,                  0x13004014,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR6,                  0x13004018,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR7,                  0x1300401C,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR8,                  0x13004020,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR9,                  0x13004024,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR10,                 0x13004028,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(SCR11,                 0x1300402C,__READ_WRITE,__scr_bits);
__IO_REG32_BIT(FS1_0,                 0x13004030,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_1,                 0x13004034,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_2,                 0x13004038,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_3,                 0x1300403C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_4,                 0x13004040,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_5,                 0x13004044,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_6,                 0x13004048,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_7,                 0x1300404C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_8,                 0x13004050,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_9,                 0x13004054,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_10,                0x13004058,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS1_11,                0x1300405C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_0,                 0x13004060,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_1,                 0x13004064,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_2,                 0x13004068,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_3,                 0x1300406C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_4,                 0x13004070,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_5,                 0x13004074,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_6,                 0x13004078,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_7,                 0x1300407C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_8,                 0x13004080,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_9,                 0x13004084,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_10,                0x13004088,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(FS2_11,                0x1300408C,__READ_WRITE,__fs_bits);
__IO_REG32_BIT(SSR0,                  0x13004090,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR1,                  0x13004094,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR2,                  0x13004098,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR3,                  0x1300409C,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR4,                  0x130040A0,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR5,                  0x130040A4,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR6,                  0x130040A8,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR7,                  0x130040AC,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR8,                  0x130040B0,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR9,                  0x130040B4,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR10,                 0x130040B8,__READ      ,__ssr_bits);
__IO_REG32_BIT(SSR11,                 0x130040BC,__READ      ,__ssr_bits);
__IO_REG32_BIT(PCR0,                  0x130040C0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR1,                  0x130040C4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR2,                  0x130040C8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR3,                  0x130040CC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR4,                  0x130040D0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR5,                  0x130040D4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR6,                  0x130040D8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR7,                  0x130040DC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR8,                  0x130040E0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR9,                  0x130040E4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR10,                 0x130040E8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR11,                 0x130040EC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR12,                 0x130040F0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR13,                 0x130040F4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR14,                 0x130040F8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR15,                 0x130040FC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR16,                 0x13004100,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR17,                 0x13004104,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR18,                 0x13004108,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR19,                 0x1300410C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR20,                 0x13004110,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR21,                 0x13004114,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR22,                 0x13004118,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR23,                 0x1300411C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR24,                 0x13004120,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR25,                 0x13004124,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR26,                 0x13004128,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR27,                 0x1300412C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR28,                 0x13004130,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR29,                 0x13004134,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR30,                 0x13004138,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR31,                 0x1300413C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR32,                 0x13004140,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR33,                 0x13004144,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR34,                 0x13004148,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR35,                 0x1300414C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR36,                 0x13004150,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR37,                 0x13004154,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR40,                 0x13004160,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR41,                 0x13004164,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR42,                 0x13004168,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR43,                 0x1300416C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR44,                 0x13004170,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR45,                 0x13004174,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR46,                 0x13004178,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR47,                 0x1300417C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR48,                 0x13004180,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR49,                 0x13004184,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR50,                 0x13004188,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR51,                 0x1300418C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR52,                 0x13004190,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR53,                 0x13004194,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR54,                 0x13004198,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR55,                 0x1300419C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR56,                 0x130041A0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR57,                 0x130041A4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR58,                 0x130041A8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR59,                 0x130041AC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR60,                 0x130041B0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR61,                 0x130041B4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR62,                 0x130041B8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR63,                 0x130041BC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR64,                 0x130041C0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR65,                 0x130041C4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR66,                 0x130041C8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR67,                 0x130041CC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR68,                 0x130041D0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR69,                 0x130041D4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR70,                 0x130041D8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR71,                 0x130041DC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR72,                 0x130041E0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR73,                 0x130041E4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR74,                 0x130041E8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR75,                 0x130041EC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR76,                 0x130041F0,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR77,                 0x130041F4,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR78,                 0x130041F8,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR79,                 0x130041FC,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR80,                 0x13004200,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR81,                 0x13004204,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR82,                 0x13004208,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR83,                 0x1300420C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR84,                 0x13004210,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR85,                 0x13004214,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR86,                 0x13004218,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR87,                 0x1300421C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR88,                 0x13004220,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR89,                 0x13004224,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR90,                 0x13004228,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PCR91,                 0x1300422C,__READ_WRITE,__pcr_bits);
__IO_REG32_BIT(PSR0,                  0x13004230,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR1,                  0x13004234,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR2,                  0x13004238,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR3,                  0x1300423C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR4,                  0x13004240,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR5,                  0x13004244,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR6,                  0x13004248,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR7,                  0x1300424C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR8,                  0x13004250,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR9,                  0x13004254,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR10,                 0x13004258,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR11,                 0x1300425C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR12,                 0x13004260,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR13,                 0x13004264,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR14,                 0x13004268,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR15,                 0x1300426C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR16,                 0x13004270,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR17,                 0x13004274,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR18,                 0x13004278,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR19,                 0x1300427C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR20,                 0x13004280,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR21,                 0x13004284,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR22,                 0x13004288,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR23,                 0x1300428C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR24,                 0x13004290,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR25,                 0x13004294,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR26,                 0x13004298,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR27,                 0x1300429C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR28,                 0x130042A0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR29,                 0x130042A4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR30,                 0x130042A8,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR31,                 0x130042AC,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR32,                 0x130042B0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR33,                 0x130042B4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR34,                 0x130042B8,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR35,                 0x130042BC,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR36,                 0x130042C0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR37,                 0x130042C4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR40,                 0x130042D0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR41,                 0x130042D4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR42,                 0x130042D8,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR43,                 0x130042DC,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR44,                 0x130042E0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR45,                 0x130042E4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR46,                 0x130042E8,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR47,                 0x130042EC,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR48,                 0x130042F0,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR49,                 0x130042F4,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR50,                 0x130042F8,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR51,                 0x130042FC,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR52,                 0x13004300,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR53,                 0x13004304,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR54,                 0x13004308,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR55,                 0x1300430C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR56,                 0x13004310,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR57,                 0x13004314,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR58,                 0x13004318,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR59,                 0x1300431C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR60,                 0x13004320,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR61,                 0x13004324,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR62,                 0x13004328,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR63,                 0x1300432C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR64,                 0x13004330,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR65,                 0x13004334,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR66,                 0x13004338,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR67,                 0x1300433C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR68,                 0x13004340,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR69,                 0x13004344,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR70,                 0x13004348,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR71,                 0x1300434C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR72,                 0x13004350,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR73,                 0x13004354,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR74,                 0x13004358,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR75,                 0x1300435C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR76,                 0x13004360,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR77,                 0x13004364,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR78,                 0x13004368,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR79,                 0x1300436C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR80,                 0x13004370,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR81,                 0x13004374,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR82,                 0x13004378,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR83,                 0x1300437C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR84,                 0x13004380,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR85,                 0x13004384,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR86,                 0x13004388,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR87,                 0x1300438C,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR88,                 0x13004390,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR89,                 0x13004394,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR90,                 0x13004398,__READ      ,__psr_bits);
__IO_REG32_BIT(PSR91,                 0x1300439C,__READ      ,__psr_bits);
__IO_REG32_BIT(ESR0,                  0x130043A0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR1,                  0x130043A4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR2,                  0x130043A8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR3,                  0x130043AC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR4,                  0x130043B0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR5,                  0x130043B4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR6,                  0x130043B8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR7,                  0x130043BC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR8,                  0x130043C0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR9,                  0x130043C4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR10,                 0x130043C8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR11,                 0x130043CC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR12,                 0x130043D0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR13,                 0x130043D4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR14,                 0x130043D8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR15,                 0x130043DC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR16,                 0x130043E0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR17,                 0x130043E4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR18,                 0x130043E8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR19,                 0x130043EC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR20,                 0x130043F0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR21,                 0x130043F4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR22,                 0x130043F8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR23,                 0x130043FC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR24,                 0x13004400,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR25,                 0x13004404,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR26,                 0x13004408,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR27,                 0x1300440C,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR28,                 0x13004410,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR29,                 0x13004414,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR30,                 0x13004418,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR31,                 0x1300441C,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR32,                 0x13004420,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR33,                 0x13004424,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR34,                 0x13004428,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR35,                 0x1300442C,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR36,                 0x13004430,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR37,                 0x13004434,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR40,                 0x13004440,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR41,                 0x13004444,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR42,                 0x13004448,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR43,                 0x1300444C,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR44,                 0x13004450,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR45,                 0x13004454,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR46,                 0x13004458,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR47,                 0x1300445C,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR48,                 0x13004460,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR49,                 0x13004464,__READ_WRITE,__esr30_bits);
__IO_REG32_BIT(ESR50,                 0x13004468,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR51,                 0x1300446C,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR52,                 0x13004470,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR53,                 0x13004474,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR54,                 0x13004478,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR55,                 0x1300447C,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR56,                 0x13004480,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR57,                 0x13004484,__READ_WRITE,__esr50_bits);
__IO_REG32_BIT(ESR58,                 0x13004488,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR59,                 0x1300448C,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR60,                 0x13004490,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR61,                 0x13004494,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR62,                 0x13004498,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR63,                 0x1300449C,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR64,                 0x130044A0,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR65,                 0x130044A4,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR66,                 0x130044A8,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR67,                 0x130044AC,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR68,                 0x130044B0,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR69,                 0x130044B4,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR70,                 0x130044B8,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR71,                 0x130044BC,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR72,                 0x130044C0,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR73,                 0x130044C4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR74,                 0x130044C8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR75,                 0x130044CC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR76,                 0x130044D0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR77,                 0x130044D4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR78,                 0x130044D8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR79,                 0x130044DC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR80,                 0x130044E0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR81,                 0x130044E4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR82,                 0x130044E8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR83,                 0x130044EC,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR84,                 0x130044F0,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR85,                 0x130044F4,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR86,                 0x130044F8,__READ_WRITE,__esr_bits);
__IO_REG32_BIT(ESR87,                 0x130044FC,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(ESR88,                 0x13004500,__READ_WRITE,__esr58_bits);
__IO_REG32_BIT(BCR0,                  0x13004504,__READ_WRITE,__bcr_bits);
__IO_REG32_BIT(BCR1,                  0x13004508,__READ_WRITE,__bcr_bits);
__IO_REG32_BIT(BCR2,                  0x1300450C,__READ_WRITE,__bcr_bits);
__IO_REG32_BIT(BCR3,                  0x13004510,__READ_WRITE,__bcr_bits);
__IO_REG32_BIT(BCR7,                  0x13004514,__READ_WRITE,__bcr_bits);
__IO_REG32_BIT(FDC0,                  0x13004518,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC1,                  0x1300451C,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC2,                  0x13004520,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC3,                  0x13004524,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC4,                  0x13004528,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC5,                  0x1300452C,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC6,                  0x13004530,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC7,                  0x13004534,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC8,                  0x13004538,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC9,                  0x1300453C,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC10,                 0x13004540,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC11,                 0x13004544,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC12,                 0x13004548,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC13,                 0x1300454C,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC14,                 0x13004550,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC15,                 0x13004554,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC16,                 0x13004558,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC17,                 0x1300455C,__READ_WRITE,__fdc17_bits);
__IO_REG32_BIT(FDC18,                 0x13004560,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC19,                 0x13004564,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC20,                 0x13004568,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC21,                 0x1300456C,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC22,                 0x13004570,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(FDC23,                 0x13004574,__READ_WRITE,__fdc_bits);
__IO_REG32_BIT(DYN_FDC0,              0x13004578,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC1,              0x1300457C,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC2,              0x13004580,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC3,              0x13004584,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC4,              0x13004588,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC5,              0x1300458C,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_FDC6,              0x13004590,__READ_WRITE,__dyn_fdc_bits);
__IO_REG32_BIT(DYN_SEL0,              0x13004594,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL1,              0x13004598,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL2,              0x1300459C,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL3,              0x130045A0,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL4,              0x130045A4,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL5,              0x130045A8,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(DYN_SEL6,              0x130045AC,__READ_WRITE,__dyn_sel_bits);
__IO_REG32_BIT(POWERMODE,             0x13004C00,__READ_WRITE,__powermode_bits);
__IO_REG32_BIT(WD_BARK,               0x13004C04,__READ      ,__wd_bark_bits);
__IO_REG32_BIT(FFAST_ON,              0x13004C08,__READ_WRITE,__ffast_on_bits);
__IO_REG32_BIT(FFAST_BYPASS,          0x13004C0C,__READ_WRITE,__ffast_bypass_bits);
__IO_REG32_BIT(APB0_RESETN_SOFT,      0x13004C10,__READ_WRITE,__apb0_resetn_soft_bits);
__IO_REG32_BIT(AHB2APB0_PNRES_SOFT,   0x13004C14,__READ_WRITE,__ahb2apb0_pnres_soft_bits);
__IO_REG32_BIT(APB1_RESETN_SOFT,      0x13004C18,__READ_WRITE,__apb1_resetn_soft_bits);
__IO_REG32_BIT(AHB2APB1_PNRES_SOFT,   0x13004C1C,__READ_WRITE,__ahb2apb1_pnres_soft_bits);
__IO_REG32_BIT(APB2_RESETN_SOFT,      0x13004C20,__READ_WRITE,__apb2_resetn_soft_bits);
__IO_REG32_BIT(AHB2APB2_PNRES_SOFT,   0x13004C24,__READ_WRITE,__ahb2apb2_pnres_soft_bits);
__IO_REG32_BIT(APB3_RESETN_SOFT,      0x13004C28,__READ_WRITE,__apb3_resetn_soft_bits);
__IO_REG32_BIT(AHB2APB3_PNRES_SOFT,   0x13004C2C,__READ_WRITE,__ahb2apb3_pnres_soft_bits);
__IO_REG32_BIT(APB4_RESETN_SOFT,      0x13004C30,__READ_WRITE,__apb4_resetn_soft_bits);
__IO_REG32_BIT(AHB_TO_INTC_RESETN_SOFT,0x13004C34,__READ_WRITE,__ahb_to_intc_resetn_soft_bits);
__IO_REG32_BIT(AHB0_RESETN_SOFT,      0x13004C38,__READ_WRITE,__ahb0_resetn_soft_bits);
__IO_REG32_BIT(EBI_RESET_N_SOFT,      0x13004C3C,__READ_WRITE,__ebi_reset_n_soft_bits);
__IO_REG32_BIT(PCM_PNRES_SOFT,        0x13004C40,__READ_WRITE,__pcm_pnres_soft_bits);
__IO_REG32_BIT(PCM_RESET_N_SOFT,      0x13004C44,__READ_WRITE,__pcm_reset_n_soft_bits);
__IO_REG32_BIT(PCM_RESET_ASYNC_N_SOFT,0x13004C48,__READ_WRITE,__pcm_reset_async_n_soft_bits);
__IO_REG32_BIT(TIMER0_PNRES_SOFT,     0x13004C4C,__READ_WRITE,__timer0_pnres_soft_bits);
__IO_REG32_BIT(TIMER1_PNRES_SOFT,     0x13004C50,__READ_WRITE,__timer1_pnres_soft_bits);
__IO_REG32_BIT(TIMER2_PNRES_SOFT,     0x13004C54,__READ_WRITE,__timer2_pnres_soft_bits);
__IO_REG32_BIT(TIMER3_PNRES_SOFT,     0x13004C58,__READ_WRITE,__timer3_pnres_soft_bits);
__IO_REG32_BIT(ADC_PRESETN_SOFT,      0x13004C5C,__READ_WRITE,__adc_presetn_soft_bits);
__IO_REG32_BIT(ADC_RESETN_ADC10BITS_SOFT,0x13004C60,__READ_WRITE,__adc_resetn_adc10bits_soft_bits);
__IO_REG32_BIT(PWM_RESET_AN_SOFT,     0x13004C64,__READ_WRITE,__pwm_reset_an_soft_bits);
__IO_REG32_BIT(UART_SYS_RST_AN_SOFT,  0x13004C68,__READ_WRITE,__uart_sys_rst_an_soft_bits);
__IO_REG32_BIT(I2C0_PNRES_SOFT,       0x13004C6C,__READ_WRITE,__i2c0_pnres_soft_bits);
__IO_REG32_BIT(I2C1_PNRES_SOFT,       0x13004C70,__READ_WRITE,__i2c1_pnres_soft_bits);
__IO_REG32_BIT(I2S_CFG_RST_N_SOFT,    0x13004C74,__READ_WRITE,__i2s_cfg_rst_n_soft_bits);
__IO_REG32_BIT(I2S_NSOF_RST_N_SOFT,   0x13004C78,__READ_WRITE,__i2s_nsof_rst_n_soft_bits);
__IO_REG32_BIT(EDGE_DET_RST_N_SOFT,   0x13004C7C,__READ_WRITE,__edge_det_rst_n_soft_bits);
__IO_REG32_BIT(I2STX_FIFO_0_RST_N_SOFT,0x13004C80,__READ_WRITE,__i2stx_fifo_0_rst_n_soft_bits);
__IO_REG32_BIT(I2STX_IF_0_RST_N_SOFT, 0x13004C84,__READ_WRITE,__i2stx_if_0_rst_n_soft_bits);
__IO_REG32_BIT(I2STX_FIFO_1_RST_N_SOFT,0x13004C88,__READ_WRITE,__i2stx_fifo_1_rst_n_soft_bits);
__IO_REG32_BIT(I2STX_IF_1_RST_N_SOFT, 0x13004C8C,__READ_WRITE,__i2stx_if_1_rst_n_soft_bits);
__IO_REG32_BIT(I2SRX_FIFO_0_RST_N_SOFT,0x13004C90,__READ_WRITE,__i2srx_fifo_0_rst_n_soft_bits);
__IO_REG32_BIT(I2SRX_IF_0_RST_N_SOFT, 0x13004C94,__READ_WRITE,__i2srx_if_0_rst_n_soft_bits);
__IO_REG32_BIT(I2SRX_FIFO_1_RST_N_SOFT,0x13004C98,__READ_WRITE,__i2srx_fifo_1_rst_n_soft_bits);
__IO_REG32_BIT(I2SRX_IF_1_RST_N_SOFT, 0x13004C9C,__READ_WRITE,__i2srx_if_1_rst_n_soft_bits);
__IO_REG32_BIT(LCD_PNRES_SOFT,        0x13004CB4,__READ_WRITE,__lcd_pnres_soft_bits);
__IO_REG32_BIT(SPI_PNRES_APB_SOFT,    0x13004CB8,__READ_WRITE,__spi_pnres_apb_soft_bits);
__IO_REG32_BIT(SPI_PNRES_IP_SOFT,     0x13004CBC,__READ_WRITE,__spi_pnres_ip_soft_bits);
__IO_REG32_BIT(DMA_PNRES_SOFT,        0x13004CC0,__READ_WRITE,__dma_pnres_soft_bits);
__IO_REG32_BIT(NANDFLASH_CTRL_ECC_RESET_N_SOFT,0x13004CC4,__READ_WRITE,__nandflash_ctrl_ecc_reset_n_soft_bits);
__IO_REG32_BIT(NANDFLASH_CTRL_AES_RESET_N_SOFT,0x13004CC8,__READ_WRITE,__nandflash_ctrl_aes_reset_n_soft_bits);
__IO_REG32_BIT(NANDFLASH_CTRL_NAND_RESET_N_SOFT,0x13004CCC,__READ_WRITE,__nandflash_ctrl_nand_reset_n_soft_bits);
__IO_REG32_BIT(SD_MMC_PNRES_SOFT,     0x13004CD4,__READ_WRITE,__sd_mmc_pnres_soft_bits);
__IO_REG32_BIT(SD_MMC_NRES_CCLK_IN_SOFT,0x13004CD8,__READ_WRITE,__sd_mmc_nres_cclk_in_soft_bits);
__IO_REG32_BIT(USB_OTG_AHB_RST_N_SOFT,0x13004CDC,__READ_WRITE,__usb_otg_ahb_rst_n_soft_bits);
__IO_REG32_BIT(RED_CTL_RESET_N_SOFT,  0x13004CE0,__READ_WRITE,__red_ctl_reset_n_soft_bits);
__IO_REG32_BIT(AHB_MPMC_HRESETN_SOFT, 0x13004CE4,__READ_WRITE,__ahb_mpmc_hresetn_soft_bits);
__IO_REG32_BIT(AHB_MPMC_REFRESH_RESETN_SOFT,0x13004CE8,__READ_WRITE,__ahb_mpmc_refresh_resetn_soft_bits);
__IO_REG32_BIT(INTC_RESETN_SOFT,      0x13004CEC,__READ_WRITE,__intc_resetn_soft_bits);
__IO_REG32_BIT(HP0_FIN_SELECT,  0x13004CF0,__READ_WRITE,__hp0_fin_select_bits);
__IO_REG32_BIT(HP0_MDEC,        0x13004CF4,__READ_WRITE,__hp0_mdec_bits);
__IO_REG32_BIT(HP0_NDEC,        0x13004CF8,__READ_WRITE,__hp0_ndec_bits);
__IO_REG32_BIT(HP0_PDEC,        0x13004CFC,__READ_WRITE,__hp0_pdec_bits);
__IO_REG32_BIT(HP0_MODE,        0x13004D00,__READ_WRITE,__hp0_mode_bits);
__IO_REG32_BIT(HP0_STATUS,      0x13004D04,__READ      ,__hp0_status_bits);
__IO_REG32_BIT(HP0_ACK,         0x13004D08,__READ      ,__hp0_ack_bits);
__IO_REG32_BIT(HP0_REQ,         0x13004D0C,__READ_WRITE,__hp0_req_bits);
__IO_REG32_BIT(HP0_INSELR,      0x13004D10,__READ_WRITE,__hp0_inselr_bits);
__IO_REG32_BIT(HP0_INSELI,      0x13004D14,__READ_WRITE,__hp0_inseli_bits);
__IO_REG32_BIT(HP0_INSELP,      0x13004D18,__READ_WRITE,__hp0_inselp_bits);
__IO_REG32_BIT(HP0_SELR,        0x13004D1C,__READ_WRITE,__hp0_selr_bits);
__IO_REG32_BIT(HP0_SELI,        0x13004D20,__READ_WRITE,__hp0_seli_bits);
__IO_REG32_BIT(HP0_SELP,        0x13004D24,__READ_WRITE,__hp0_selp_bits);
__IO_REG32_BIT(HP1_FIN_SELECT,  0x13004D28,__READ_WRITE,__hp1_fin_select_bits);
__IO_REG32_BIT(HP1_MDEC,        0x13004D2C,__READ_WRITE,__hp1_mdec_bits);
__IO_REG32_BIT(HP1_NDEC,        0x13004D30,__READ_WRITE,__hp1_ndec_bits);
__IO_REG32_BIT(HP1_PDEC,        0x13004D34,__READ_WRITE,__hp1_pdec_bits);
__IO_REG32_BIT(HP1_MODE,        0x13004D38,__READ_WRITE,__hp1_mode_bits);
__IO_REG32_BIT(HP1_STATUS,      0x13004D3C,__READ      ,__hp1_status_bits);
__IO_REG32_BIT(HP1_ACK,         0x13004D40,__READ      ,__hp1_ack_bits);
__IO_REG32_BIT(HP1_REQ,         0x13004D44,__READ_WRITE,__hp1_req_bits);
__IO_REG32_BIT(HP1_INSELR,      0x13004D48,__READ_WRITE,__hp1_inselr_bits);
__IO_REG32_BIT(HP1_INSELI,      0x13004D4C,__READ_WRITE,__hp1_inseli_bits);
__IO_REG32_BIT(HP1_INSELP,      0x13004D50,__READ_WRITE,__hp1_inselp_bits);
__IO_REG32_BIT(HP1_SELR,        0x13004D54,__READ_WRITE,__hp1_selr_bits);
__IO_REG32_BIT(HP1_SELI,        0x13004D58,__READ_WRITE,__hp1_seli_bits);
__IO_REG32_BIT(HP1_SELP,        0x13004D5C,__READ_WRITE,__hp1_selp_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDTIM_IR ,             0x13002400,__READ_WRITE,__wdtim_ir_bits);
__IO_REG32_BIT(WDTIM_TCR,             0x13002404,__READ_WRITE,__wdtim_tcr_bits);
__IO_REG32(    WDTIM_TC,              0x13002408,__READ      );
__IO_REG32(    WDTIM_PR,              0x1300240C,__READ_WRITE);
__IO_REG32(    WDTIM_PC,              0x13002410,__READ      );
__IO_REG32_BIT(WDTIM_MCR,             0x13002414,__READ_WRITE,__wdtim_mcr_bits);
__IO_REG32(    WDTIM_MR0,             0x13002418,__READ_WRITE);
__IO_REG32(    WDTIM_MR1,             0x1300241C,__READ_WRITE);
__IO_REG32_BIT(WDTIM_EMR,             0x1300243C,__READ_WRITE,__wdtim_emr_bits);

/***************************************************************************
 **
 ** IOCONF
 **
 ***************************************************************************/
__IO_REG32_BIT(IOCONF_EBI_MCI_PIN,    0x13003000,__READ      ,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M0,     0x13003010,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M0_SET, 0x13003014,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M0_CLR, 0x13003018,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M1,     0x13003020,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M1_SET, 0x13003024,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_MCI_M1_CLR, 0x13003028,__READ_WRITE,__ioconf_ebi_mci_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_PIN,   0x13003040,__READ      ,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M0,    0x13003050,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M0_SET,0x13003054,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M0_CLR,0x13003058,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M1,    0x13003060,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M1_SET,0x13003064,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_EBI_I2STX_0_M1_CLR,0x13003068,__READ_WRITE,__ioconf_ebi_i2stx_0_bits);
__IO_REG32_BIT(IOCONF_CGU_PIN,        0x13003080,__READ      ,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M0,         0x13003090,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M0_SET,     0x13003094,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M0_CLR,     0x13003098,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M1,         0x130030A0,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M1_SET,     0x130030A4,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_CGU_M1_CLR,     0x130030A8,__READ_WRITE,__ioconf_cgu_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_PIN,    0x130030C0,__READ      ,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M0,     0x130030D0,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M0_SET, 0x130030D4,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M0_CLR, 0x130030D8,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M1,     0x130030E0,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M1_SET, 0x130030E4,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S0_RX_M1_CLR, 0x130030E8,__READ_WRITE,__ioconf_i2s0_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_PIN,    0x13003100,__READ      ,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M0,     0x13003110,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M0_SET, 0x13003114,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M0_CLR, 0x13003118,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M1,     0x13003120,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M1_SET, 0x13003124,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_RX_M1_CLR, 0x13003128,__READ_WRITE,__ioconf_i2s1_rx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_PIN,    0x13003140,__READ      ,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M0,     0x13003150,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M0_SET, 0x13003154,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M0_CLR, 0x13003158,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M1,     0x13003160,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M1_SET, 0x13003164,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_I2S1_TX_M1_CLR, 0x13003168,__READ_WRITE,__ioconf_i2s1_tx_bits);
__IO_REG32_BIT(IOCONF_EBI_PIN,        0x13003180,__READ      ,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M0,         0x13003190,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M0_SET,     0x13003194,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M0_CLR,     0x13003198,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M1,         0x130031A0,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M1_SET,     0x130031A4,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_EBI_M1_CLR,     0x130031A8,__READ_WRITE,__ioconf_ebi_bits);
__IO_REG32_BIT(IOCONF_GPIO_PIN,       0x130031C0,__READ      ,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M0,        0x130031D0,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M0_SET,    0x130031D4,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M0_CLR,    0x130031D8,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M1,        0x130031E0,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M1_SET,    0x130031E4,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_GPIO_M1_CLR,    0x130031E8,__READ_WRITE,__ioconf_gpio_bits);
__IO_REG32_BIT(IOCONF_I2C1_PIN,       0x13003200,__READ      ,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M0,        0x13003210,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M0_SET,    0x13003214,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M0_CLR,    0x13003218,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M1,        0x13003220,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M1_SET,    0x13003224,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_I2C1_M1_CLR,    0x13003228,__READ_WRITE,__ioconf_i2c1_bits);
__IO_REG32_BIT(IOCONF_SPI_PIN,        0x13003240,__READ      ,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M0,         0x13003250,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M0_SET,     0x13003254,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M0_CLR,     0x13003258,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M1,         0x13003260,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M1_SET,     0x13003264,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_SPI_M1_CLR,     0x13003268,__READ_WRITE,__ioconf_spi_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_PIN,   0x13003280,__READ      ,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M0,    0x13003290,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M0_SET,0x13003294,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M0_CLR,0x13003298,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M1,    0x130032A0,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M1_SET,0x130032A4,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_NANDFLASH_CTRL_M1_CLR,0x130032A8,__READ_WRITE,__ioconf_nand_ctrl_bits);
__IO_REG32_BIT(IOCONF_PWM_PIN,        0x130032C0,__READ      ,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M0,         0x130032D0,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M0_SET,     0x130032D4,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M0_CLR,     0x130032D8,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M1,         0x130032E0,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M1_SET,     0x130032E4,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_PWM_M1_CLR,     0x130032E8,__READ_WRITE,__ioconf_pwm_bits);
__IO_REG32_BIT(IOCONF_UART_PIN,       0x13003300,__READ      ,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M0,        0x13003310,__READ_WRITE,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M0_SET,    0x13003314,__READ_WRITE,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M0_CLR,    0x13003318,__READ_WRITE,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M1,        0x13003320,__READ_WRITE,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M1_SET,    0x13003324,__READ_WRITE,__ioconf_uart_bits);
__IO_REG32_BIT(IOCONF_UART_M1_CLR,    0x13003328,__READ_WRITE,__ioconf_uart_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_R0_REG,            0x13002000,__READ      ,__adc_r_reg_bits);
__IO_REG32_BIT(ADC_R1_REG,            0x13002004,__READ      ,__adc_r_reg_bits);
__IO_REG32_BIT(ADC_R2_REG,            0x13002008,__READ      ,__adc_r_reg_bits);
__IO_REG32_BIT(ADC_R3_REG,            0x1300200C,__READ      ,__adc_r_reg_bits);
__IO_REG32_BIT(ADC_CON_REG,           0x13002020,__READ_WRITE,__adc_con_reg_bits);
__IO_REG32_BIT(ADC_CSEL_REG,          0x13002024,__READ_WRITE,__adc_csel_reg_bits);
__IO_REG32_BIT(ADC_INT_ENABLE_REG,    0x13002028,__READ_WRITE,__adc_int_enable_reg_bits);
__IO_REG32_BIT(ADC_INT_STATUS_REG,    0x1300202C,__READ      ,__adc_int_status_reg_bits);
__IO_REG32_BIT(ADC_INT_CLEAR_REG,     0x13002030,__WRITE     ,__adc_int_clear_reg_bits);

/***************************************************************************
 **
 ** Event Router
 **
 ***************************************************************************/
__IO_REG32_BIT(ER_PEND0,              0x13000C00,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_PEND1,              0x13000C04,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_PEND2,              0x13000C08,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_PEND3,              0x13000C0C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INT_CLR0,           0x13000C20,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INT_CLR1,           0x13000C24,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INT_CLR2,           0x13000C28,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INT_CLR3,           0x13000C2C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INT_SET0,           0x13000C40,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INT_SET1,           0x13000C44,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INT_SET2,           0x13000C48,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INT_SET3,           0x13000C4C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_MASK0,              0x13000C60,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_MASK1,              0x13000C64,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_MASK2,              0x13000C68,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_MASK3,              0x13000C6C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_MASK_CLR0,          0x13000C80,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_MASK_CLR1,          0x13000C84,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_MASK_CLR2,          0x13000C88,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_MASK_CLR3,          0x13000C8C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_MASK_SET0,          0x13000CA0,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_MASK_SET1,          0x13000CA4,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_MASK_SET2,          0x13000CA8,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_MASK_SET3,          0x13000CAC,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_APR0,               0x13000CC0,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_APR1,               0x13000CC4,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_APR2,               0x13000CC8,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_APR3,               0x13000CCC,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_ATR0,               0x13000CE0,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_ATR1,               0x13000CE4,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_ATR2,               0x13000CE8,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_ATR3,               0x13000CEC,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_RST0,               0x13000D20,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_RST1,               0x13000D24,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_RST2,               0x13000D28,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_RST3,               0x13000D2C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUT,             0x13000D40,__READ      ,__er_intout_bits);
__IO_REG32_BIT(ER_INTOUTPEND00,       0x13001000,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTPEND01,       0x13001004,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTPEND02,       0x13001008,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTPEND03,       0x1300100C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTPEND10,       0x13001020,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTPEND11,       0x13001024,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTPEND12,       0x13001028,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTPEND13,       0x1300102C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTPEND20,       0x13001040,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTPEND21,       0x13001044,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTPEND22,       0x13001048,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTPEND23,       0x1300104C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTPEND30,       0x13001060,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTPEND31,       0x13001064,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTPEND32,       0x13001068,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTPEND33,       0x1300106C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTPEND40,       0x13001080,__READ      ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTPEND41,       0x13001084,__READ      ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTPEND42,       0x13001088,__READ      ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTPEND43,       0x1300108C,__READ      ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASK00,       0x13001400,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASK01,       0x13001404,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASK02,       0x13001408,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASK03,       0x1300140C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASK10,       0x13001420,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASK11,       0x13001424,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASK12,       0x13001428,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASK13,       0x1300142C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASK20,       0x13001440,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASK21,       0x13001444,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASK22,       0x13001448,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASK23,       0x1300144C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASK30,       0x13001460,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASK31,       0x13001464,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASK32,       0x13001468,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASK33,       0x1300146C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASK40,       0x13001480,__READ_WRITE,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASK41,       0x13001484,__READ_WRITE,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASK42,       0x13001488,__READ_WRITE,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASK43,       0x1300148C,__READ_WRITE,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR00,    0x13001800,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR01,    0x13001804,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR02,    0x13001808,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR03,    0x1300180C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR10,    0x13001820,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR11,    0x13001824,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR12,    0x13001828,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR13,    0x1300182C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR20,    0x13001840,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR21,    0x13001844,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR22,    0x13001848,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR23,    0x1300184C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR30,    0x13001860,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR31,    0x13001864,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR32,    0x13001868,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR33,    0x1300186C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR40,    0x13001880,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR41,    0x13001884,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR42,    0x13001888,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKCLR43,    0x1300188C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET00,    0x13001C00,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET01,    0x13001C04,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET02,    0x13001C08,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET03,    0x13001C0C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET10,    0x13001C20,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET11,    0x13001C24,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET12,    0x13001C28,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET13,    0x13001C2C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET20,    0x13001C40,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET21,    0x13001C44,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET22,    0x13001C48,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET23,    0x13001C4C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET30,    0x13001C60,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET31,    0x13001C64,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET32,    0x13001C68,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET33,    0x13001C6C,__WRITE     ,__er_pend3_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET40,    0x13001C80,__WRITE     ,__er_pend0_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET41,    0x13001C84,__WRITE     ,__er_pend1_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET42,    0x13001C88,__WRITE     ,__er_pend2_bits);
__IO_REG32_BIT(ER_INTOUTMASKSET43,    0x13001C8C,__WRITE     ,__er_pend3_bits);

/***************************************************************************
 **
 ** RNG
 **
 ***************************************************************************/
__IO_REG32(    RANDOM_NUMBER,         0x13006000,__READ      );
__IO_REG32_BIT(RNG_POWERDOWN,         0x13006FF4,__READ_WRITE,__rng_powerdown_bits);

/***************************************************************************
 **
 ** OTP memory
 **
 ***************************************************************************/
__IO_REG32_BIT(OTP_con,               0x13005000,__READ_WRITE,__otp_con_bits);
__IO_REG32_BIT(OTP_rprot,             0x13005004,__READ_WRITE,__otp_rprot_bits);
__IO_REG32_BIT(OTP_wprot,             0x13005008,__READ_WRITE,__otp_rprot_bits);
__IO_REG32(    OTP_data_0,            0x1300500C,__READ      );
__IO_REG32(    OTP_data_1,            0x13005010,__READ      );
__IO_REG32(    OTP_data_2,            0x13005014,__READ      );
__IO_REG32(    OTP_data_3,            0x13005018,__READ      );
__IO_REG32(    OTP_data_4,            0x1300501C,__READ      );
__IO_REG32(    OTP_data_5,            0x13005020,__READ      );
__IO_REG32(    OTP_data_6,            0x13005024,__READ      );
__IO_REG32(    OTP_data_7,            0x13005028,__READ      );
__IO_REG32(    OTP_data_8,            0x1300502C,__READ      );
__IO_REG32(    OTP_data_9,            0x13005030,__READ      );
__IO_REG32(    OTP_data_10,           0x13005034,__READ      );
__IO_REG32(    OTP_data_11,           0x13005038,__READ      );
__IO_REG32(    OTP_data_12,           0x1300503C,__READ      );
__IO_REG32(    OTP_data_13,           0x13005040,__READ      );
__IO_REG32(    OTP_data_14,           0x13005044,__READ      );
__IO_REG32(    OTP_data_15,           0x13005048,__READ      );

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI_CONFIG,            0x15002000,__READ_WRITE,__spi_config_bits);
__IO_REG32_BIT(SPI_SLAVE_ENABLE,      0x15002004,__READ_WRITE,__spi_slave_enable_bits);
__IO_REG32_BIT(SPI_TX_FIFO_FLUSH,     0x15002008,__WRITE     ,__spi_tx_fifo_flush_bits);
__IO_REG16(    SPI_FIFO_DATA,         0x1500200C,__READ_WRITE);
__IO_REG32_BIT(SPI_NHP_POP,           0x15002010,__WRITE     ,__spi_nhp_pop_bits);
__IO_REG32_BIT(SPI_NHP_MODE,          0x15002014,__READ_WRITE,__spi_nhp_mode_bits);
__IO_REG32_BIT(SPI_DMA_SETTINGS,      0x15002018,__READ_WRITE,__spi_dma_settings_bits);
__IO_REG32_BIT(SPI_STATUS,            0x1500201C,__READ      ,__spi_status_bits);
__IO_REG32_BIT(SPI_HW_INFO,           0x15002020,__READ      ,__spi_hw_info_bits);
__IO_REG32_BIT(SPI_SLV0_SETTINGS1,    0x15002024,__READ_WRITE,__spi_slv_settings1_bits);
__IO_REG32_BIT(SPI_SLV0_SETTINGS2,    0x15002028,__READ_WRITE,__spi_slv_settings2_bits);
__IO_REG32_BIT(SPI_SLV1_SETTINGS1,    0x1500202C,__READ_WRITE,__spi_slv_settings1_bits);
__IO_REG32_BIT(SPI_SLV1_SETTINGS2,    0x15002030,__READ_WRITE,__spi_slv_settings2_bits);
__IO_REG32_BIT(SPI_SLV2_SETTINGS1,    0x15002034,__READ_WRITE,__spi_slv_settings1_bits);
__IO_REG32_BIT(SPI_SLV2_SETTINGS2,    0x15002038,__READ_WRITE,__spi_slv_settings2_bits);
__IO_REG32_BIT(SPI_INT_THRESHOLD,     0x15002FD4,__READ_WRITE,__spi_int_threshold_bits);
__IO_REG32_BIT(SPI_INT_CLR_ENABLE,    0x15002FD8,__WRITE     ,__spi_int_clr_enable_bits);
__IO_REG32_BIT(SPI_INT_SET_ENABLE,    0x15002FDC,__WRITE     ,__spi_int_set_enable_bits);
__IO_REG32_BIT(SPI_INT_STATUS,        0x15002FE0,__READ      ,__spi_int_status_bits);
__IO_REG32_BIT(SPI_INT_ENABLE,        0x15002FE4,__READ      ,__spi_int_enable_bits);
__IO_REG32_BIT(SPI_INT_CLR_STATUS,    0x15002FE8,__WRITE     ,__spi_int_clr_status_bits);
__IO_REG32_BIT(SPI_INT_SET_STATUS,    0x15002FEC,__WRITE     ,__spi_int_set_status_bits);

/***************************************************************************
 **
 ** MCI
 **
 ***************************************************************************/
__IO_REG32_BIT(MCI_CTRL,              0x18000000,__READ_WRITE,__mci_ctrl_bits);
__IO_REG32(    MCI_PWREN,             0x18000004,__READ_WRITE);
__IO_REG32_BIT(MCI_CLKDIV,            0x18000008,__READ_WRITE,__mci_clkdiv_bits);
__IO_REG32_BIT(MCI_CLKSRC,            0x1800000C,__READ_WRITE,__mci_clksrc_bits);
__IO_REG32_BIT(MCI_CLKENA,            0x18000010,__READ_WRITE,__mci_clkena_bits);
__IO_REG32_BIT(MCI_TMOUT,             0x18000014,__READ_WRITE,__mci_tmout_bits);
__IO_REG32_BIT(MCI_CTYPE,             0x18000018,__READ_WRITE,__mci_ctype_bits);
__IO_REG32_BIT(MCI_BLKSIZ,            0x1800001C,__READ_WRITE,__mci_blksiz_bits);
__IO_REG32(    MCI_BYTCNT,            0x18000020,__READ_WRITE);
__IO_REG32_BIT(MCI_INTMASK,           0x18000024,__READ_WRITE,__mci_intmask_bits);
__IO_REG32(    MCI_CMDARG,            0x18000028,__READ_WRITE);
__IO_REG32_BIT(MCI_CMD,               0x1800002C,__READ_WRITE,__mci_cmd_bits);
__IO_REG32(    MCI_RESP0,             0x18000030,__READ      );
__IO_REG32(    MCI_RESP1,             0x18000034,__READ      );
__IO_REG32(    MCI_RESP2,             0x18000038,__READ      );
__IO_REG32(    MCI_RESP3,             0x1800003C,__READ      );
__IO_REG32_BIT(MCI_MINTSTS,           0x18000040,__READ      ,__mci_intmask_bits);
__IO_REG32_BIT(MCI_RINTSTS,           0x18000044,__READ_WRITE,__mci_intmask_bits);
__IO_REG32_BIT(MCI_STATUS,            0x18000048,__READ      ,__mci_status_bits);
__IO_REG32_BIT(MCI_FIFOTH,            0x1800004C,__READ_WRITE,__mci_fifoth_bits);
__IO_REG32_BIT(MCI_CDETECT,           0x18000050,__READ      ,__mci_cdetect_bits);
__IO_REG32_BIT(MCI_WRTPRT,            0x18000054,__READ      ,__mci_wrtprt_bits);
__IO_REG32(    MCI_TCBCNT,            0x1800005C,__READ      );
__IO_REG32(    MCI_TBBCNT,            0x18000060,__READ      );
__IO_REG32(    MCI_DATA,              0x18000100,__READ_WRITE);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG8(     UART_RBR,              0x15001000,__READ_WRITE);
#define UART_THR  UART_RBR
#define UART_DLL  UART_RBR
__IO_REG32_BIT(UART_IER,              0x15001004,__READ_WRITE,__uart_ier_bits);
#define UART_DLM  UART_IER
__IO_REG32_BIT(UART_IIR,              0x15001008,__READ_WRITE,__uart_iir_bits);
#define UART_FCR      UART_IIR
#define UART_FCR_bit  UART_IIR_bit
__IO_REG32_BIT(UART_LCR,              0x1500100C,__READ_WRITE,__uart_lcr_bits);
__IO_REG32_BIT(UART_MCR,              0x15001010,__READ_WRITE,__uart_mcr_bits);
__IO_REG32_BIT(UART_LSR,              0x15001014,__READ      ,__uart_lsr_bits);
__IO_REG32_BIT(UART_MSR,              0x15001018,__READ      ,__uart_msr_bits);
__IO_REG8(     UART_SCR,              0x1500101C,__READ_WRITE);
__IO_REG32_BIT(UART_ICR,              0x15001024,__READ_WRITE,__uart_icr_bits);
__IO_REG32_BIT(UART_FDR,              0x15001028,__READ_WRITE,__uart_fdr_bits);
__IO_REG32_BIT(UART_POP,              0x15001030,__WRITE     ,__uart_pop_bits);
__IO_REG32_BIT(UART_MODE,             0x15001034,__READ_WRITE,__uart_mode_bits);
__IO_REG32_BIT(UART_INTCE,            0x15001FD8,__WRITE     ,__uart_intce_bits);
__IO_REG32_BIT(UART_INTSE,            0x15001FDC,__WRITE     ,__uart_intse_bits);
__IO_REG32_BIT(UART_INTS,             0x15001FE0,__READ      ,__uart_ints_bits);
__IO_REG32_BIT(UART_INTE,             0x15001FE4,__READ      ,__uart_inte_bits);
__IO_REG32_BIT(UART_INTCS,            0x15001FE8,__WRITE     ,__uart_intcs_bits);
__IO_REG32_BIT(UART_INTSS,            0x15001FEC,__WRITE     ,__uart_intss_bits);

/***************************************************************************
 **
 ** LCD
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_STATUS,            0x15000400,__READ      ,__lcd_status_bits);
__IO_REG32_BIT(LCD_CONTROL,           0x15000404,__READ_WRITE,__lcd_control_bits);
__IO_REG32_BIT(LCD_INT_RAW,           0x15000408,__READ      ,__lcd_int_raw_bits);
__IO_REG32_BIT(LCD_INT_CLEAR,         0x1500040C,__WRITE     ,__lcd_int_clear_bits);
__IO_REG32_BIT(LCD_INT_MASK,          0x15000410,__READ_WRITE,__lcd_int_mask_bits);
__IO_REG32_BIT(LCD_READ_CMD,          0x15000414,__WRITE     ,__lcd_read_cmd_bits);
__IO_REG32_BIT(LCD_INST_BYTE,         0x15000420,__READ_WRITE,__lcd_inst_byte_bits);
__IO_REG32_BIT(LCD_DATA_BYTE,         0x15000430,__READ_WRITE,__lcd_data_byte_bits);
__IO_REG32(    LCD_INST_WORD,         0x15000440,__WRITE     );
__IO_REG32(    LCD_DATA_WORD,         0x15000480,__WRITE     );

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_TX,               0x1300A000,__READ_WRITE,__i2c_rx_tx_bits);
#define I2C0_RX     I2C0_TX
#define I2C0_RX_bit I2C0_TX_bit
__IO_REG32_BIT(I2C0_STS,              0x1300A004,__READ_WRITE,__i2c_sts_bits);
__IO_REG32_BIT(I2C0_CTL,              0x1300A008,__READ_WRITE,__i2c_ctrl_bits);
__IO_REG32_BIT(I2C0_CLK_HI,           0x1300A00C,__READ_WRITE,__i2c_clk_hi_bits);
__IO_REG32_BIT(I2C0_CLK_LO,           0x1300A010,__READ_WRITE,__i2c_clk_lo_bits);
__IO_REG32_BIT(I2C0_ADR,              0x1300A014,__READ_WRITE,__i2c_adr_bits);
__IO_REG32_BIT(I2C0_RFL,              0x1300A018,__READ      ,__i2c_rfl_bits);
__IO_REG32_BIT(I2C0_TFL,              0x1300A01C,__READ      ,__i2c_tfl_bits);
__IO_REG32_BIT(I2C0_RXB,              0x1300A020,__READ      ,__i2c_rxb_bits);
__IO_REG32_BIT(I2C0_TXB,              0x1300A024,__READ      ,__i2c_txb_bits);
__IO_REG8(     I2C0_TXS,              0x1300A028,__WRITE     );
__IO_REG32_BIT(I2C0_STFL,             0x1300A02C,__READ      ,__i2c_stfl_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_RX,               0x1300A400,__READ_WRITE,__i2c_rx_tx_bits);
#define I2C1_TX     I2C1_RX
#define I2C1_TX_bit I2C1_RX_bit
__IO_REG32_BIT(I2C1_STS,              0x1300A404,__READ_WRITE,__i2c_sts_bits);
__IO_REG32_BIT(I2C1_CTL,              0x1300A408,__READ_WRITE,__i2c_ctrl_bits);
__IO_REG32_BIT(I2C1_CLK_HI,           0x1300A40C,__READ_WRITE,__i2c_clk_hi_bits);
__IO_REG32_BIT(I2C1_CLK_LO,           0x1300A410,__READ_WRITE,__i2c_clk_lo_bits);
__IO_REG32_BIT(I2C1_ADR,              0x1300A414,__READ_WRITE,__i2c_adr_bits);
__IO_REG32_BIT(I2C1_RFL,              0x1300A418,__READ      ,__i2c_rfl_bits);
__IO_REG32_BIT(I2C1_TFL,              0x1300A41C,__READ      ,__i2c_tfl_bits);
__IO_REG32_BIT(I2C1_RXB,              0x1300A420,__READ      ,__i2c_rxb_bits);
__IO_REG32_BIT(I2C1_TXB,              0x1300A424,__READ      ,__i2c_txb_bits);
__IO_REG8(     I2C1_TXS,              0x1300A428,__WRITE     );
__IO_REG32_BIT(I2C1_STFL,             0x1300A42C,__READ      ,__i2c_stfl_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32(    Timer0Load,            0x13008000,__READ_WRITE);
__IO_REG32(    Timer0Value,           0x13008004,__READ      );
__IO_REG32_BIT(Timer0Ctrl,            0x13008008,__READ_WRITE,__timerctrl_bits);
__IO_REG32(    Timer0Clear,           0x1300800C,__WRITE     );

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32(    Timer1Load,            0x13008400,__READ_WRITE);
__IO_REG32(    Timer1Value,           0x13008404,__READ      );
__IO_REG32_BIT(Timer1Ctrl,            0x13008408,__READ_WRITE,__timerctrl_bits);
__IO_REG32(    Timer1Clear,           0x1300840C,__WRITE     );

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32(    Timer2Load,            0x13008800,__READ_WRITE);
__IO_REG32(    Timer2Value,           0x13008804,__READ      );
__IO_REG32_BIT(Timer2Ctrl,            0x13008808,__READ_WRITE,__timerctrl_bits);
__IO_REG32(    Timer2Clear,           0x1300880C,__WRITE     );

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32(    Timer3Load,            0x13008C00,__READ_WRITE);
__IO_REG32(    Timer3Value,           0x13008C04,__READ      );
__IO_REG32_BIT(Timer3Ctrl,            0x13008C08,__READ_WRITE,__timerctrl_bits);
__IO_REG32(    Timer3Clear,           0x13008C0C,__WRITE     );

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_TMR,               0x13009000,__READ_WRITE,__pwm_tmr_bits);
__IO_REG32_BIT(PWM_CNTL,              0x13009004,__READ_WRITE,__pwm_cntl_bits);

/***************************************************************************
 **
 ** SYSCTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSCREG_EBI_MPMC_PRIO,                 0x13002808,__READ_WRITE,__syscreg_ebi_mpmc_prio_bits);
__IO_REG32_BIT(SYSCREG_EBI_NANDC_PRIO,                0x1300280C,__READ_WRITE,__syscreg_ebi_mpmc_prio_bits);
__IO_REG32_BIT(SYSCREG_EBI_UNUSED_PRIO,               0x13002810,__READ_WRITE,__syscreg_ebi_mpmc_prio_bits);
__IO_REG32_BIT(SYSCREG_RING_OSC_CFG,                  0x13002814,__READ_WRITE,__syscreg_ring_osc_cfg_bits);
__IO_REG32_BIT(SYSCREG_ADC_PD_ADC10BITS,              0x13002818,__READ_WRITE,__syscreg_adc_pd_adc10bits_bits);
__IO_REG32_BIT(SYSCREG_ABC_CFG,                       0x13002824,__READ_WRITE,__syscreg_abc_cfg_bits);
__IO_REG32_BIT(SYSCREG_SD_MMC_CFG,                    0x13002828,__READ_WRITE,__syscreg_sd_mmc_cfg_bits);
__IO_REG32_BIT(SYSCREG_MCI_DELAYMODES,                0x1300282C,__READ_WRITE,__syscreg_mci_delaymodes_bits);
__IO_REG32_BIT(SYSCREG_USB_ATX_PLL_PD_REG,            0x13002830,__READ_WRITE,__syscreg_usb_atx_pll_pd_reg_bits);
__IO_REG32_BIT(SYSCREG_USB_OTG_CFG,                   0x13002834,__READ_WRITE,__syscreg_usb_otg_cfg_bits);
__IO_REG32_BIT(SYSCREG_USB_OTG_PORT_IND_CTL,          0x13002838,__READ      ,__syscreg_usb_otg_port_ind_ctl_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_NDEC,                  0x13002840,__READ_WRITE,__syscreg_usb_pll_ndec_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_MDEC,                  0x13002844,__READ_WRITE,__syscreg_usb_pll_mdec_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_PDEC,                  0x13002848,__READ_WRITE,__syscreg_usb_pll_pdec_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_SELR,                  0x1300284C,__READ_WRITE,__syscreg_usb_pll_selr_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_SELI,                  0x13002850,__READ_WRITE,__syscreg_usb_pll_seli_bits);
__IO_REG32_BIT(SYSCREG_USB_PLL_SELP,                  0x13002854,__READ_WRITE,__syscreg_usb_pll_selp_bits);
__IO_REG32_BIT(SYSCREG_ISRAM0_LATENCY_CFG,            0x13002858,__READ_WRITE,__syscreg_isram_latency_cfg_bits);
__IO_REG32_BIT(SYSCREG_ISRAM1_LATENCY_CFG,            0x1300285C,__READ_WRITE,__syscreg_isram_latency_cfg_bits);
__IO_REG32_BIT(SYSCREG_ISROM_LATENCY_CFG,             0x13002860,__READ_WRITE,__syscreg_isrom_latency_cfg_bits);
__IO_REG32_BIT(SYSCREG_AHB_MPMC_MISC,                 0x13002864,__READ_WRITE,__syscreg_ahb_mpmc_misc_bits);
__IO_REG32_BIT(SYSCREG_MPMP_DELAYMODES,               0x13002868,__READ_WRITE,__syscreg_mpmp_delaymodes_bits);
__IO_REG32_BIT(SYSCREG_MPMC_WAITREAD_DELAY0,          0x1300286C,__READ_WRITE,__syscreg_mpmc_waitread_delay_bits);
__IO_REG32_BIT(SYSCREG_MPMC_WAITREAD_DELAY1,          0x13002870,__READ_WRITE,__syscreg_mpmc_waitread_delay_bits);
__IO_REG32_BIT(SYSCREG_WIRE_EBI_MSIZE_INIT,           0x13002874,__READ_WRITE,__syscreg_wire_ebi_msize_init_bits);
__IO_REG32_BIT(SYSCREG_MPMC_TESTMODE0,                0x13002878,__READ_WRITE,__syscreg_mpmc_testmode0_bits);
__IO_REG32_BIT(SYSCREG_MPMC_TESTMODE1,                0x1300287C,__READ_WRITE,__syscreg_mpmc_testmode1_bits);
__IO_REG32_BIT(SYSCREG_AHB0_EXTPRIO,                  0x13002880,__READ_WRITE,__syscreg_ahb0_extprio_bits);
__IO_REG32(    SYSCREG_ARM926_SHADOW_POINTER,         0x13002884,__READ_WRITE);
__IO_REG32_BIT(SYSCREG_MUX_LCD_EBI_SEL,               0x13002890,__READ_WRITE,__syscreg_mux_lcd_ebi_sel_bits);
__IO_REG32_BIT(SYSCREG_MUX_GPIO_MCI_SEL,              0x13002894,__READ_WRITE,__syscreg_mux_gpio_mci_sel_bits);
__IO_REG32_BIT(SYSCREG_MUX_NAND_MCI_SEL,              0x13002898,__READ_WRITE,__syscreg_mux_nand_mci_sel_bits);
__IO_REG32_BIT(SYSCREG_MUX_UART_SPI_SEL,              0x1300289C,__READ_WRITE,__syscreg_mux_uart_spi_sel_bits);
__IO_REG32_BIT(SYSCREG_MUX_I2STX_PCM_SEL,             0x130028A0,__READ_WRITE,__syscreg_mux_i2stx_pcm_sel_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_9_PCTRL,                 0x130028A4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_10_PCTRL,                0x130028A8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_11_PCTRL,                0x130028AC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_12_PCTRL,                0x130028B0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_13_PCTRL,                0x130028B4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_14_PCTRL,                0x130028B8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_DAI_BCK0_PCTRL,                0x130028BC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO9_PCTRL,                  0x130028C0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO6_PCTRL,                  0x130028C4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_7_PCTRL,               0x130028C8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_4_PCTRL,               0x130028CC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_2_PCTRL,               0x130028D0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MNAND_RYBN0_PCTRL,             0x130028D4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO1_PCTRL,                   0x130028D8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_4_PCTRL,                 0x130028DC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MI2STX_CLK0_PCTRL,             0x130028E0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MI2STX_BCK0_PCTRL,             0x130028E4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_A_1_CLE_PCTRL,             0x130028E8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_NCAS_BLOUT_0_PCTRL,        0x130028EC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_NAND_NCS_3_PCTRL,              0x130028F0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_0_PCTRL,               0x130028F4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_DQM_0_NOE_PCTRL,           0x130028F8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_0_PCTRL,                 0x130028FC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_1_PCTRL,                 0x13002900,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_2_PCTRL,                 0x13002904,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_3_PCTRL,                 0x13002908,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_5_PCTRL,                 0x1300290C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_6_PCTRL,                 0x13002910,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_7_PCTRL,                 0x13002914,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_8_PCTRL,                 0x13002918,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_D_15_PCTRL,                0x1300291C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2STX_DATA1_PCTRL,             0x13002920,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2STX_BCK1_PCTRL,              0x13002924,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2STX_WS1_PCTRL,               0x13002928,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2SRX_DATA0_PCTRL,             0x1300292C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2SRX_WS0_PCTRL,               0x13002930,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2SRX_DATA1_PCTRL,             0x13002934,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2SRX_BCK1_PCTRL,              0x13002938,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2SRX_WS1_PCTRL,               0x1300293C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SYSCLK_O_PCTRL,                0x13002940,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_PWM_DATA_PCTRL,                0x13002944,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_UART_RXD_PCTRL,                0x13002948,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_UART_TXD_PCTRL,                0x1300294C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2C_SDA1_PCTRL,                0x13002950,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_I2C_SCL1_PCTRL,                0x13002954,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_CLK_256FS_O_PCTRL,             0x13002958,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO0_PCTRL,                   0x1300295C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO2_PCTRL,                   0x13002960,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO3_PCTRL,                   0x13002964,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO4_PCTRL,                   0x13002968,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO11_PCTRL,                  0x1300296C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO12_PCTRL,                  0x13002970,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO13_PCTRL,                  0x13002974,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO14_PCTRL,                  0x13002978,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO15_PCTRL,                  0x1300297C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO16_PCTRL,                  0x13002980,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO17_PCTRL,                  0x13002984,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO18_PCTRL,                  0x13002988,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO19_PCTRL,                  0x1300298C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_GPIO20_PCTRL,                  0x13002990,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SPI_MISO_PCTRL,                0x13002994,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SPI_MOSI_PCTRL,                0x13002998,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SPI_CS_IN_PCTRL,               0x1300299C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SPI_SCK_PCTRL,                 0x130029A0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_SPI_CS_OUT0_PCTRL,             0x130029A4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_NAND_NCS_0_PCTRL,              0x130029A8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_NAND_NCS_1_PCTRL,              0x130029AC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_NAND_NCS_2_PCTRL,              0x130029B0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_CSB_PCTRL,                0x130029B4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_1_PCTRL,               0x130029B8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_E_RD_PCTRL,               0x130029BC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_RS_PCTRL,                 0x130029C0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_RW_WR_PCTRL,              0x130029C4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_3_PCTRL,               0x130029C8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_5_PCTRL,               0x130029CC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_6_PCTRL,               0x130029D0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_8_PCTRL,               0x130029D4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_9_PCTRL,               0x130029D8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_10_PCTRL,              0x130029DC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_11_PCTRL,              0x130029E0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_12_PCTRL,              0x130029E4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_13_PCTRL,              0x130029E8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_14_PCTRL,              0x130029EC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MLCD_DB_15_PCTRL,              0x130029F0,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO5_PCTRL,                  0x130029F4,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO7_PCTRL,                  0x130029F8,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO8_PCTRL,                  0x130029FC,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MGPIO10_PCTRL,                 0x13002A00,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MNAND_RYBN1_PCTRL,             0x13002A04,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MNAND_RYBN2_PCTRL,             0x13002A08,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MNAND_RYBN3_PCTRL,             0x13002A0C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MUART_CTS_N_PCTRL,             0x13002A10,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MI2STX_DATA0_PCTRL,            0x13002A18,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_MI2STX_WS0_PCTRL,              0x13002A1C,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_NRAS_BLOUT_1_PCTRL,        0x13002A20,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_A_0_ALE_PCTRL,             0x13002A24,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_EBI_NWE_PCTRL,                 0x13002A28,__READ_WRITE,__syscreg_pctrl_bits);
__IO_REG32_BIT(SYSCREG_ESHCTRL_SUP4,                  0x13002A2C,__READ_WRITE,__syscreg_eshctrl_sup4_bits);
__IO_REG32_BIT(SYSCREG_ESHCTRL_SUP8,                  0x13002A30,__READ_WRITE,__syscreg_eshctrl_sup8_bits);

/***************************************************************************
 **
 ** PCM_IOM
 **
 ***************************************************************************/
__IO_REG32_BIT(PCM_IOM_GLOBAL,        0x15000000,__READ_WRITE,__pcm_iom_global_bits);
__IO_REG32_BIT(PCM_IOM_CNTL0,         0x15000004,__READ_WRITE,__pcm_iom_cntl0_bits);
__IO_REG32_BIT(PCM_IOM_CNTL1,         0x15000008,__READ_WRITE,__pcm_iom_cntl1_bits);
__IO_REG16(    PCM_IOM_HPOUT0,        0x1500000C,__WRITE     );
__IO_REG16(    PCM_IOM_HPOUT1,        0x15000010,__WRITE     );
__IO_REG16(    PCM_IOM_HPOUT2,        0x15000014,__WRITE     );
__IO_REG16(    PCM_IOM_HPOUT3,        0x15000018,__WRITE     );
__IO_REG16(    PCM_IOM_HPOUT4,        0x1500001C,__WRITE     );
__IO_REG16(    PCM_IOM_HPOUT5,        0x15000020,__WRITE     );
__IO_REG16(    PCM_IOM_HPIN0,         0x15000024,__READ      );
__IO_REG16(    PCM_IOM_HPIN1,         0x15000028,__READ      );
__IO_REG16(    PCM_IOM_HPIN2,         0x1500002C,__READ      );
__IO_REG16(    PCM_IOM_HPIN3,         0x15000030,__READ      );
__IO_REG16(    PCM_IOM_HPIN4,         0x15000034,__READ      );
__IO_REG16(    PCM_IOM_HPIN5,         0x15000038,__READ      );
__IO_REG32_BIT(PCM_IOM_CNTL2,         0x1500003C,__READ_WRITE,__pcm_iom_cntl2_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_FORMAT_SETTINGS,   0x16000000,__READ_WRITE,__i2s_format_settings_bits);
__IO_REG32_BIT(AUDIOSS_MUX_SETTINGS,  0x16000004,__READ_WRITE,__audioss_mux_settings_bits);
__IO_REG32_BIT(N_SOF_COUNTER,         0x16000010,__READ_WRITE,__n_sof_counter_bits);
__IO_REG32(    I2STX_0_LEFT_16BIT,    0x16000080,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_16BIT,   0x16000084,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_24BIT,    0x16000088,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_24BIT,   0x1600008C,__READ_WRITE);
__IO_REG32(    I2STX_0_INT_STATUS,    0x16000090,__READ_WRITE);
__IO_REG32(    I2STX_0_INT_MASK,      0x16000094,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_0,  0x160000A0,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_1,  0x160000A4,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_2,  0x160000A8,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_3,  0x160000AC,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_4,  0x160000B0,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_5,  0x160000B4,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_6,  0x160000B8,__READ_WRITE);
__IO_REG32(    I2STX_0_LEFT_32BIT_7,  0x160000BC,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_0, 0x160000C0,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_1, 0x160000C4,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_2, 0x160000C8,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_3, 0x160000CC,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_4, 0x160000D0,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_5, 0x160000D4,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_6, 0x160000D8,__READ_WRITE);
__IO_REG32(    I2STX_0_RIGHT_32BIT_7, 0x160000DC,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_0, 0x160000E0,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_1, 0x160000E4,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_2, 0x160000E8,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_3, 0x160000EC,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_4, 0x160000F0,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_5, 0x160000F4,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_6, 0x160000F8,__READ_WRITE);
__IO_REG32(    I2STX_0_INTERLEAVED_7, 0x160000FC,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_16BIT,    0x16000100,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_16BIT,   0x16000104,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_24BIT,    0x16000108,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_24BIT,   0x1600010C,__READ_WRITE);
__IO_REG32(    I2STX_1_INT_STATUS,    0x16000110,__READ_WRITE);
__IO_REG32(    I2STX_1_INT_MASK,      0x16000114,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_0,  0x16000120,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_1,  0x16000124,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_2,  0x16000128,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_3,  0x1600012C,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_4,  0x16000130,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_5,  0x16000134,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_6,  0x16000138,__READ_WRITE);
__IO_REG32(    I2STX_1_LEFT_32BIT_7,  0x1600013C,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_0, 0x16000140,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_1, 0x16000144,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_2, 0x16000148,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_3, 0x1600014C,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_4, 0x16000150,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_5, 0x16000154,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_6, 0x16000158,__READ_WRITE);
__IO_REG32(    I2STX_1_RIGHT_32BIT_7, 0x1600015C,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_0, 0x16000160,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_1, 0x16000164,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_2, 0x16000168,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_3, 0x1600016C,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_4, 0x16000170,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_5, 0x16000174,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_6, 0x16000178,__READ_WRITE);
__IO_REG32(    I2STX_1_INTERLEAVED_7, 0x1600017C,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_16BIT,    0x16000180,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_16BIT,   0x16000184,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_24BIT,    0x16000188,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_24BIT,   0x1600018C,__READ_WRITE);
__IO_REG32(    I2SRX_0_INT_STATUS,    0x16000190,__READ_WRITE);
__IO_REG32(    I2SRX_0_INT_MASK,      0x16000194,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_0,  0x160001A0,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_1,  0x160001A4,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_2,  0x160001A8,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_3,  0x160001AC,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_4,  0x160001B0,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_5,  0x160001B4,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_6,  0x160001B8,__READ_WRITE);
__IO_REG32(    I2SRX_0_LEFT_32BIT_7,  0x160001BC,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_0, 0x160001C0,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_1, 0x160001C4,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_2, 0x160001C8,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_3, 0x160001CC,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_4, 0x160001D0,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_5, 0x160001D4,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_6, 0x160001D8,__READ_WRITE);
__IO_REG32(    I2SRX_0_RIGHT_32BIT_7, 0x160001DC,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_0, 0x160001E0,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_1, 0x160001E4,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_2, 0x160001E8,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_3, 0x160001EC,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_4, 0x160001F0,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_5, 0x160001F4,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_6, 0x160001F8,__READ_WRITE);
__IO_REG32(    I2SRX_0_INTERLEAVED_7, 0x160001FC,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_16BIT,    0x16000200,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_16BIT,   0x16000204,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_24BIT,    0x16000208,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_24BIT,   0x1600020C,__READ_WRITE);
__IO_REG32(    I2SRX_1_INT_STATUS,    0x16000210,__READ_WRITE);
__IO_REG32(    I2SRX_1_INT_MASK,      0x16000214,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_0,  0x16000220,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_1,  0x16000224,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_2,  0x16000228,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_3,  0x1600022C,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_4,  0x16000230,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_5,  0x16000234,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_6,  0x16000238,__READ_WRITE);
__IO_REG32(    I2SRX_1_LEFT_32BIT_7,  0x1600023C,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_0, 0x16000240,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_1, 0x16000244,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_2, 0x16000248,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_3, 0x1600024C,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_4, 0x16000250,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_5, 0x16000254,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_6, 0x16000258,__READ_WRITE);
__IO_REG32(    I2SRX_1_RIGHT_32BIT_7, 0x1600025C,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_0, 0x16000260,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_1, 0x16000264,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_2, 0x16000268,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_3, 0x1600026C,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_4, 0x16000270,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_5, 0x16000274,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_6, 0x16000278,__READ_WRITE);
__IO_REG32(    I2SRX_1_INTERLEAVED_7, 0x1600027C,__READ_WRITE);

/***************************************************************************
 **  Assembler-specific declarations
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
 **  Interrupt channels
 **
 ***************************************************************************/
#define CASCADED_IRQ_0           1  /* Event Router IRQ0                       */
#define CASCADED_IRQ_1           2  /* Event Router IRQ0                       */
#define CASCADED_IRQ_2           3  /* Event Router IRQ0                       */
#define CASCADED_IRQ_3           4  /* Event Router IRQ0                       */
#define TIMER0_INTCT             5  /* Count INT Timer0                        */
#define TIMER1_INTCT             6  /* Count INT Timer0                        */
#define TIMER2_INTCT             7  /* Count INT Timer0                        */
#define TIMER3_INTCT             8  /* Count INT Timer0                        */
#define SSA1_ADC_INT             9  /* ADC INT                                 */
#define UART_INTREQ             10  /* UART INT                                */
#define I2C0_NINTR              11  /* I2C0 INT                                */
#define I2C1_NINTR              12  /* I2C1 INT                                */
#define I2STX0_IRQ              13  /* I2S0 TRANSMIT INTERRUPT                 */
#define I2STX1_IRQ              14  /* I2S1 TRANSMIT INTERRUPT                 */
#define I2SRX0_IRQ              15  /* I2S0 RECEIVE INTERRUPT                  */
#define I2SRX1_IRQ              16  /* I2S1 RECEIVE INTERRUPT                  */
#define SSA1_LCD_INTERFACE_IRQ  18  /* LCD INTR                                */
#define SPI_SMS_INT             19  /* SPI SMS INTR                            */
#define SPI_TX_INT              20  /* SPI Transmit                            */
#define SPI_RX_INT              21  /* SPI Receive                             */
#define SPI_OV_INT              22  /* SPI OV                                  */
#define SPI_INT                 23  /* SPI Interrupt                           */
#define SIMPLE_DMA_IRQ          24  /* SDMA DATA TRANSFER COMPLETE             */
#define NANDFLASH_CTRL_IR       25  /* NANDFLASH CTRL Interrupt                */
#define DW_SD_MMC_INTR          26  /* MCI Interrupt                           */
#define USB_OTG_IRQ             27  /* USB OTG Interrupt                       */
#define ISRAM0_MRC_FINISHED     28  /* ISRAM0 Interrupt                        */
#define ISRAM1_MRC_FINISHED     29  /* ISRAM1 Interrupt                        */

#endif    /* __IOLPC3143_H */
