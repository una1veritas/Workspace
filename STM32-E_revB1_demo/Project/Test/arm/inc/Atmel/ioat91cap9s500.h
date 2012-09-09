/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Atmel Semiconductors AT91CAP9S500
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2007
 **
 **    $Revision: 30244 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOAT91CAP9S500_H
#define __IOAT91CAP9S500_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    AT91CAP9S500 SPECIAL FUNCTION REGISTERS
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


/* RSTC Control Register */
typedef struct {
  __REG32   PROCRST     : 1;
  __REG32               : 1;
  __REG32   PERRST      : 1;
  __REG32   EXTRST      : 1;
  __REG32               :20;
  __REG32   KEY         : 8;
} __rstc_cr_bits;

/* RSTC Status Register */
typedef struct {
  __REG32   URSTS       : 1;
  __REG32               : 7;
  __REG32   RSTTYP      : 3;
  __REG32               : 5;
  __REG32   NRSTL       : 1;
  __REG32   SRCMP       : 1;
  __REG32               :14;
} __rstc_sr_bits;

/* RSTC Mode Register */
typedef struct {
  __REG32   URSTEN      : 1;
  __REG32               : 3;
  __REG32   URSTIEN     : 1;
  __REG32               : 3;
  __REG32   ERSTL       : 4;
  __REG32               :12;
  __REG32   KEY         : 8;
} __rstc_mr_bits;


/* RTT Mode Register */
typedef struct {
  __REG32   RTPRES      :16;
  __REG32   ALMIEN      : 1;    
  __REG32   RTTINCIEN   : 1;
  __REG32   RTTRST      : 1;
  __REG32               :13;
} __rtt_mr_bits;
             
/* RTT Status Register */
typedef struct {
  __REG32   ALMS        : 1;
  __REG32   RTTINC      : 1;
  __REG32               :30;
} __rtt_sr_bits;

/* PIT Mode Register */
typedef struct {
  __REG32   PIV         :20;
  __REG32               : 4;
  __REG32   PITEN       : 1;
  __REG32   PITIEN      : 1;
  __REG32               : 6;
} __pit_mr_bits;

/* PIT Status Register */
typedef struct {
  __REG32   PITS        : 1;
  __REG32               :31;
} __pit_sr_bits;

/* PIT Value Register */
/* PIT Image Register */
typedef struct {
  __REG32   CPIV        :20;
  __REG32   PICNT       :12;
} __pit_pivr_bits;

/* WDT Control Register */
typedef struct {
  __REG32   WDRSTT      : 1;
  __REG32               :23;
  __REG32   KEY         : 8;
} __wdt_cr_bits;

/* WDT Mode Register */
typedef struct {
  __REG32   WDV         :12;
  __REG32   WDFIEN      : 1;
  __REG32   WDRSTEN     : 1;
  __REG32   WDRPROC     : 1;
  __REG32   WDDIS       : 1;
  __REG32   WDD         :12;
  __REG32   WDDBGHLT    : 1;
  __REG32   WDIDLEHLT   : 1;
  __REG32               : 2;
} __wdt_mr_bits;

/* WDT Status Register */
typedef struct {
  __REG32   WDUNF       : 1;
  __REG32   WDERR       : 1;
  __REG32               :30;
} __wdt_sr_bits;

/* SHDWC Control Register */
typedef struct {
  __REG32   SHDW        : 1;
  __REG32               :23;
  __REG32   KEY         : 8;
} __shdw_cr_bits;

/* SHDWC Mode Register */
typedef struct {
  __REG32   WKMODE0     : 2;
  __REG32               : 2;
  __REG32   CPTWK0      : 4;
  __REG32               : 8;
  __REG32   RTTWKEN     : 1;
  __REG32               :15;
} __shdw_mr_bits;

/* SHDWC Status Register */
typedef struct {
  __REG32   WAKEUP0     : 1;
  __REG32               :15;
  __REG32   RTTWK       : 1;
  __REG32               :15;
} __shdw_sr_bits;

/* Bus Matrix Master Configuration Registers */
typedef struct {
  __REG32    ULBT       : 3;
  __REG32               :29;
} __matrix_mcfgx_bits;

/*Bus Matrix Slave Configuration Registers*/
typedef struct {
  __REG32   SLOT_CYCLE    : 8;
  __REG32                 : 8;
  __REG32   DEFMSTR_TYPE  : 2;
  __REG32   FIXED_DEFMSTR : 4;
  __REG32                 : 2;
  __REG32   ARBT          : 2;
  __REG32                 : 6;
} __matrix_scfgx_bits;

/*Bus Matrix Priority Registers A For Slaves*/
typedef struct {
  __REG32   M0PR        : 2;
  __REG32               : 2;
  __REG32   M1PR        : 2;
  __REG32               : 2;
  __REG32   M2PR        : 2;
  __REG32               : 2;
  __REG32   M3PR        : 2;
  __REG32               : 2;
  __REG32   M4PR        : 2;
  __REG32               : 2;
  __REG32   M5PR        : 2;
  __REG32               : 2;
  __REG32   M6PR        : 2;
  __REG32               : 2;
  __REG32   M7PR        : 2;
  __REG32               : 2;
} __matrix_prasx_bits;


/* Bus Matrix Priority Registers B For Slaves */
typedef struct {
  __REG32   M8PR        : 2;
  __REG32               : 2;
  __REG32   M9PR        : 2;
  __REG32               : 2;
  __REG32   M10PR       : 2;
  __REG32               : 2;  
  __REG32   M11PR       : 2;  
  __REG32               : 2;  
  __REG32               :16;
} __matrix_prbsx_bits;

/* Bus Matrix Master Remap Control Register */
typedef struct {
  __REG32   RCB0        : 1;
  __REG32   RCB1        : 1;
  __REG32   RCB2        : 1;
  __REG32   RCB3        : 1;
  __REG32   RCB4        : 1;
  __REG32   RCB5        : 1;
  __REG32   RCB6        : 1;
  __REG32   RCB7        : 1; 
  __REG32   RCB8        : 1;
  __REG32               :23;
} __matrix_mrcr_bits; 

/* EBI Chip Select Assignment Register */
typedef struct {
  __REG32               : 1;
  __REG32   EBI_CS1A    : 1;
  __REG32               : 1;
  __REG32   EBI_CS3A    : 1;
  __REG32   EBI_CS4A    : 1;
  __REG32   EBI_CS5A    : 1;
  __REG32               : 2;
  __REG32   EBI_DBPUC   : 1;
  __REG32   EBI_DQSPDC  : 1;
  __REG32               : 6;
  __REG32   VDDIOMSEL   : 1;
  __REG32               :15;
} __ebi_csa_bits;

/* SMC Setup Registers */
typedef struct {
  __REG32   NWE_SETUP     : 6;
  __REG32                 : 2;
  __REG32   NCS_WR_SETUP  : 6;
  __REG32                 : 2;
  __REG32   NRD_SETUP     : 6;
  __REG32                 : 2;
  __REG32   NCS_RD_SETUP  : 6;
  __REG32                 : 2;
} __smc_setup_csx_bits;

/* SMC Pulse Registers */
typedef struct {
  __REG32   NWE_PULSE     : 7;
  __REG32                 : 1;
  __REG32   NCS_WR_PULSE  : 7;
  __REG32                 : 1;
  __REG32   NRD_PULSE     : 7;
  __REG32                 : 1;
  __REG32   NCS_RD_PULSE  : 7;
  __REG32                 : 1;
} __smc_pulse_csx_bits;

/* SMC Cycle Registers */
typedef struct {
  __REG32   NWE_CYCLE   : 9;
  __REG32               : 7;
  __REG32   NRD_CYCLE   : 9;
  __REG32               : 7;
} __smc_cycle_csx_bits;

/* SMC MODE Registers */
typedef struct {
  __REG32   READ_MODE   : 1;
  __REG32   WRITE_MODE  : 1;
  __REG32               : 2;
  __REG32   EXNW_MODE   : 2;
  __REG32               : 2;
  __REG32   BAT         : 1;
  __REG32               : 3;
  __REG32   DBW         : 2;
  __REG32               : 2;
  __REG32   TDF_CYCLES  : 4;
  __REG32   TDF_MODE    : 1;
  __REG32               : 3;
  __REG32   PMEN        : 1;
  __REG32               : 3;
  __REG32   PS          : 2;
  __REG32               : 2;
}__smc_mode_csx_bits; 


/* DDRSDRC Mode Register */
typedef struct {
  __REG32   MODE        : 3;
  __REG32               :29;
} __ddrsdrc_mr_bits;

/* DDRSDRC Refresh Timer Register */
typedef struct {
  __REG32   COUNT       :12;
  __REG32               :20;
}__ddrsdrc_rtr_bits;

/* DDRSDRC Configuration Register */
typedef struct {
  __REG32   NC          : 2;
  __REG32   NR          : 2;
  __REG32   CAS         : 3;
  __REG32   DLL         : 1;
  __REG32   DIC_DS      : 1;
  __REG32               :23;
}__ddrsdrc_cr_bits;

/* DDRSDRC Timing 0 Parameter Register */
typedef struct {
  __REG32   TRAS        : 4;
  __REG32   TRCD        : 4;
  __REG32   TWR         : 4;
  __REG32   TRC         : 4;
  __REG32   TRP         : 4;
  __REG32   TRRD        : 4;
  __REG32   TWTR        : 1;
  __REG32               : 3;
  __REG32   TMRD        : 4;
}__ddrsdrc_t0pr_bits;

/* DDRSDRC Timing 1 Parameter Register */
typedef struct {
  __REG32   TRFC        : 5;
  __REG32               : 3;
  __REG32   TXSNR       : 8;
  __REG32   TXSRD       : 8;
  __REG32   TXP         : 4;
  __REG32               : 4;
}__ddrsdrc_t1pr_bits;

/* DDRSDRC Low-power Register */
typedef struct {
  __REG32   LPCB        : 2;
  __REG32   CLK_FR      : 1;
  __REG32               : 1;
  __REG32   PASR        : 3;
  __REG32               : 1;
  __REG32   TCR         : 2;
  __REG32   DS          : 2;
  __REG32   TIMEOUT     : 2;
  __REG32               :18;
}__ddrsdrc_lpr_bits;

/* DDRSDRC Memory Device Register */
typedef struct {
  __REG32   MD          : 2;
  __REG32               : 2;
  __REG32   DBW         : 1;
  __REG32               :27;
}__ddrsdrc_md_bits;

/* DDRSDRC DLL Information */
typedef struct {
  __REG32   MDINC       : 1;
  __REG32   MDDEC       : 1;
  __REG32   MDOVF       : 1;
  __REG32   SDCOVF      : 1;
  __REG32   SDCUDF      : 1;
  __REG32   SDERF       : 1;
  __REG32               : 2;
  __REG32   MDVAL       : 8;
  __REG32   SDVAL       : 8;
  __REG32   SDCVAL      : 8;
}__ddrsdrc_dll_bits;

/* BCRAMC Configuration Register */
typedef struct {
  __REG32   CRAM_EN       : 1;
  __REG32                 : 3;
  __REG32   LM            : 3;
  __REG32                 : 1;
  __REG32   DBW           : 1;
  __REG32                 : 3;
  __REG32   BOUNDARY_WORD : 2;
  __REG32                 : 2;
  __REG32   ADDRDATA_MUX  : 1;
  __REG32                 : 3;
  __REG32   DS            : 2;
  __REG32                 : 2;
  __REG32   VAR_FIX_LAT   : 1;
  __REG32                 : 7;
} __bcramc_cr_bits;

/* BCRAMC Timing Register */
typedef struct {
  __REG32   TCW         : 4;
  __REG32   TCRES       : 2;
  __REG32               : 2;
  __REG32   TCKA        : 4;
  __REG32               :20;
} __bcramc_tr_bits;

/* BCRAMC Low Power Register */
typedef struct {
  __REG32   PAR         : 3;
  __REG32               : 1;
  __REG32   TCR_TCSR    : 2;
  __REG32               : 2;
  __REG32   LPCB        : 2;
  __REG32               :22;
} __bcramc_lpr_bits;

/* BCRAMC Memory Device Register */
typedef struct {
  __REG32   MD          : 2;
  __REG32               :30;
} __bcramc_mdr_bits;


/* ECC Control Register */
typedef struct {
  __REG32   RST         : 1;
  __REG32               :31;
} __ecc_cr_bits;

/* ECC Mode Register */
typedef struct {
  __REG32   PAGESIZE    : 2;
  __REG32               :30;
} __ecc_mr_bits;

/* ECC Status Register */
typedef struct {
  __REG32   RECERR      : 1;
  __REG32   ECCERR      : 1;
  __REG32   MULERR      : 1;
  __REG32               :29;
} __ecc_sr_bits;

/* ECC Parity Register */
typedef struct {
  __REG32   BITADDR     : 4;
  __REG32   WORDADDR    :12;
  __REG32               :16;
} __ecc_pr_bits;


/* DMAC Global Configuration Register */
typedef struct {
  __REG32   IF0_BIGEND  : 1;
  __REG32               : 3;
  __REG32   ARB_CFG     : 1;
  __REG32               :27;
} __dmac_gcfg_bits;

/* DMAC Enable Register */
typedef struct {
  __REG32   ENABLE      : 1;
  __REG32               :31;
} __dmac_en_bits;

/* DMAC Software Single Request Register */
typedef struct {
  __REG32   SSREQ0      : 1;
  __REG32   DSREQ0      : 1;
  __REG32   SSREQ1      : 1;   
  __REG32   DSREQ1      : 1;
  __REG32   SSREQ2      : 1;
  __REG32   DSREQ2      : 1;
  __REG32   SSREQ3      : 1;
  __REG32   DSREQ3      : 1;
  __REG32               :24;
} __dmac_sreq_bits;

/* DMAC Software Chunk Transfer Request Register */
typedef struct {
  __REG32   SCREQ0      : 1;
  __REG32   DCREQ0      : 1;
  __REG32   SCREQ1      : 1;   
  __REG32   DCREQ1      : 1;
  __REG32   SCREQ2      : 1;
  __REG32   DCREQ2      : 1;
  __REG32   SCREQ3      : 1;
  __REG32   DCREQ3      : 1;
  __REG32               :24;
} __dmac_creq_bits;

/* DMAC Software Last Transfer Flag Register */
typedef struct {
  __REG32   SLAST0      : 1;
  __REG32   DLAST0      : 1;
  __REG32   SLAST1      : 1;   
  __REG32   DLAST1      : 1;
  __REG32   SLAST2      : 1;
  __REG32   DLAST2      : 1;
  __REG32   SLAST3      : 1;
  __REG32   DLAST3      : 1;
  __REG32               :24;
} __dmac_last_bits;

/* DMAC Error, Buffer Transfer and Chained Buffer Transfer Interrupt Enable Register */
/* DMAC Error, Buffer Transfer and Chained Buffer Transfer Interrupt Disable Register */
/* DMAC Error, Buffer Transfer and Chained Buffer Transfer Interrupt Mask Register */
/* DMAC Error, Buffer Transfer and Chained Buffer Transfer Status Register */
typedef struct {
  __REG32   BTC0        : 1;
  __REG32   BTC1        : 1;
  __REG32   BTC2        : 1;
  __REG32   BTC3        : 1;
  __REG32               : 4;
  __REG32   CBTC0       : 1;
  __REG32   CBTC1       : 1;
  __REG32   CBTC2       : 1;
  __REG32   CBTC3       : 1;
  __REG32               : 4;
  __REG32   ERR0        : 1;
  __REG32   ERR1        : 1;
  __REG32   ERR2        : 1;
  __REG32   ERR3        : 1;
  __REG32               : 4;
  __REG32               : 8;
} __dmac_ebcier_bits;

/* DMAC Channel Handler Enable Register */
typedef struct {
  __REG32   ENA0        : 1;
  __REG32   ENA1        : 1;
  __REG32   ENA2        : 1;
  __REG32   ENA3        : 1;
  __REG32               : 4;
  __REG32   SUSP0       : 1;
  __REG32   SUSP1       : 1;
  __REG32   SUSP2       : 1;
  __REG32   SUSP3       : 1;
  __REG32               : 4;
  __REG32               : 8;
  __REG32   KEEP0       : 1;
  __REG32   KEEP1       : 1;
  __REG32   KEEP2       : 1;
  __REG32   KEEP3       : 1;
  __REG32               : 4;
} __dmac_cher_bits;

/* DMAC Channel Handler Disable Register */
typedef struct {
  __REG32   DIS0        : 1;
  __REG32   DIS1        : 1;
  __REG32   DIS2        : 1;
  __REG32   DIS3        : 1;
  __REG32               : 4;
  __REG32   RES0        : 1;
  __REG32   RES1        : 1;
  __REG32   RES2        : 1;
  __REG32   RES3        : 1;
  __REG32               : 4;
  __REG32               :16;

} __dmac_chdr_bits;

/* DMAC Channel Handler Status Register */
typedef struct {
  __REG32   ENA0        : 1;
  __REG32   ENA1        : 1;
  __REG32   ENA2        : 1;
  __REG32   ENA3        : 1;
  __REG32               : 4;
  __REG32   SUSP0       : 1;
  __REG32   SUSP1       : 1;
  __REG32   SUSP2       : 1;
  __REG32   SUSP3       : 1;
  __REG32               : 4;
  __REG32   EMPT0       : 1;
  __REG32   EMPT1       : 1;
  __REG32   EMPT2       : 1;
  __REG32   EMPT3       : 1;
  __REG32               : 4;
  __REG32   STAL0       : 1;
  __REG32   STAL1       : 1;
  __REG32   STAL2       : 1;
  __REG32   STAL3       : 1;
  __REG32               : 4;
} __dmac_chsr_bits;

/* DMAC Channel X Control A Registers */
typedef struct {
  __REG32   BTSIZE      :16;
  __REG32   SCSIZE      : 3;
  __REG32               : 1;
  __REG32   DCSIZE      : 3;
  __REG32               : 1;
  __REG32   SRC_WIDTH   : 2;
  __REG32               : 2;
  __REG32   DST_WIDTH   : 2;
  __REG32               : 1;
  __REG32   DONE        : 1;
} __dmac_ctrlax_bits;

/* DMAC Channel x Control B Register*/
typedef struct {
  __REG32   SIF         : 2;  
  __REG32               : 2;
  __REG32   DIF         : 2;
  __REG32               : 2;
  __REG32   SRC_PIP     : 1;
  __REG32               : 3;
  __REG32   DST_PIP     : 1;
  __REG32               : 3;
  __REG32   SRC_DSCR    : 1;
  __REG32               : 3;
  __REG32   DST_DSCR    : 1;
  __REG32   FC          : 3;
  __REG32   SRC_INCR    : 2;
  __REG32               : 2;
  __REG32   DST_INCR    : 2;
  __REG32               : 1;
  __REG32   AUTO        : 1;
} __dmac_ctrlbx_bits;

/* DMAC Channel x Configuration Register */
typedef struct {
  __REG32   SRC_PER     : 4;
  __REG32   DST_PER     : 4;
  __REG32   SRC_REP     : 1;
  __REG32   SRC_H2SEL   : 1;
  __REG32               : 2;
  __REG32   DST_REP     : 1;
  __REG32   DST_H2SEL   : 1;
  __REG32               : 2;
  __REG32   SOD         : 1;
  __REG32               : 3;
  __REG32   LOCK_IF     : 1;
  __REG32   LOCK_B      : 1;
  __REG32   LOCK_IF_L   : 1;
  __REG32               : 1;
  __REG32   AHB_PROT    : 3;
  __REG32               : 5;
} __dmac_cfgx_bits;

/* DMAC Channel x Source Picture in Picture Configuration Register */
typedef struct {
  __REG32   SPIP_HOLE     :16;
  __REG32   SPIP_BOUNDARY :10;
  __REG32                 : 6;
} __dmac_spipx_bits;

/* DMAC Channel x Destination Picture in Picture Configuration Register*/
typedef struct {
  __REG32   DPIPE_HOLE    :16;
  __REG32   DPIP_BOUNDARY :10;
  __REG32                 : 6;
} __dmac_dpipx_bits;

/* DBGU, USART, SSC, SPI, MCI, ect.Transfer Control Register */
/* DBGU_PTCR*/
/* SPIx_PTCR*/
/* TWI_PTCR*/
/* USARx_PTCR*/
/* SCSx_PTCR*/
/* AC97C_CAPTCR*/
/* MCIx_PTCR*/
typedef struct {
  __REG32   RXTEN       : 1;
  __REG32   RXTDIS      : 1;
  __REG32               : 6;
  __REG32   TXTEN       : 1;
  __REG32   TXTDIS      : 1;
  __REG32               :22; 
} __pdc_ptcr_bits;

/* DBGU, USART, SSC, SPI, MCI, etc.Transfer Status Register */
/* DBGU_PTSR*/
/* SPIx_PTSR*/
/* TWI_PTSR*/
/* USARx_PTSR*/
/* SCSx_PTSR*/
/* AC97C_CAPTSR*/
/* MCIx_PTSR*/
typedef struct {
  __REG32   RXTEN       : 1;
  __REG32               : 7;
  __REG32   TXTEN       : 1;
  __REG32               :23;
} __pdc_ptsr_bits;

/* PMC System Clock Enable Register */
/* PMC System Clock Disable Register */
/* PMC System Clock Status Register */
typedef struct {
  __REG32   PCK         : 1;
  __REG32               : 5;
  __REG32   UHP         : 1;
  __REG32               : 1;
  __REG32   PCK0        : 1;
  __REG32   PCK1        : 1;
  __REG32   PCK2        : 1;
  __REG32   PCK3        : 1;
  __REG32               :20;
} __pmc_scer_bits;

/* PMC Peripheral Clock Enable Register */
/* PMC Peripheral Clock Disable Register */
/* PMC Peripheral Clock Status Register */
typedef struct {
  __REG32               : 2;
  __REG32   PIOA_D      : 1;
  __REG32   MPB0        : 1;
  __REG32   MPB1        : 1;
  __REG32   MPB2        : 1;
  __REG32   MPB3        : 1;
  __REG32   MPB4        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   US2         : 1;
  __REG32   MCI0        : 1;
  __REG32   MCI1        : 1;
  __REG32   CAN         : 1;
  __REG32   TWI         : 1;
  __REG32   SPI0        : 1;
  __REG32   SPI1        : 1;
  __REG32   SSC0        : 1;
  __REG32   SSC1        : 1;
  __REG32   AC97        : 1;
  __REG32   TC          : 1;
  __REG32   PWMC        : 1;
  __REG32   EMAC        : 1;
  __REG32               : 1;
  __REG32   ADCC        : 1;
  __REG32   ISI         : 1;
  __REG32   LCDC        : 1;
  __REG32   DMA         : 1;
  __REG32   UDPHS       : 1;
  __REG32   UHP         : 1;
  __REG32   IRQ0        : 1;
  __REG32   IRQ1        : 1;
} __pmc_pcer_bits;

/* PMC UTMI Clock Configuration Register */
typedef struct {
  __REG32               :16;
  __REG32   UPLLEN      : 1;
  __REG32               : 3;
  __REG32   PLLCOUNT    : 4;
  __REG32   BIASEN      : 1;
  __REG32               : 3;
  __REG32   BIASCOUNT   : 4;
} __ckgr_uckr_bits;

/* PMC Clock Generator Main Oscillator Register */
typedef struct {
  __REG32   MOSCEN      : 1;
  __REG32   OSCBYPASS   : 1;
  __REG32               : 6;
  __REG32   OSCOUNT     : 8;
  __REG32               :16;
} __ckgr_mor_bits;

/* PMC Clock Generator Main Clock Frequency Register */
typedef struct {
  __REG32   MAINF       :16;
  __REG32   MAINRDY     : 1;
  __REG32               :15;
} __ckgr_mcfr_bits;

/* PMC Clock Generator PLL A Register */
typedef struct {
  __REG32   DIVA        : 8;
  __REG32   PLLACOUNT   : 6;
  __REG32   OUTA        : 2;
  __REG32   MULA        :11; 
  __REG32               : 5;
} __ckgr_pllar_bits;

/* PMC Clock Generator PLL B Register */
typedef struct {
  __REG32   DIVB        : 8;
  __REG32   PLLBCOUNT   : 6;
  __REG32   OUTB        : 2;
  __REG32   MULB        :11;
  __REG32               : 1;
  __REG32   USBDIV      : 2;
  __REG32               : 2;
} __ckgr_pllbr_bits;

/* PMC Master Clock Register */
typedef struct {
  __REG32   CSS         : 2;
  __REG32   PRES        : 3;
  __REG32               : 3;
  __REG32   MDIV        : 2;
  __REG32               :22;
} __pmc_mckr_bits;

/* PMC Programmable Clock Registers */
/*PMC_PCKx*/
typedef struct {
  __REG32   CSS         : 2;
  __REG32   PRES        : 3;
  __REG32               :27;
} __pmc_pckx_bits;

/* PMC Interrupt Enable Register */
/* PMC Interrupt Disable Register */
/* PMC Status Register */
/* PMC Interrupt Mask Register*/
typedef struct {
  __REG32   MOSCS       : 1;
  __REG32   LOCKA       : 1;
  __REG32   LOCKB       : 1;
  __REG32   MCKRDY      : 1;
  __REG32               : 2;
  __REG32   LOCKU       : 1;
  __REG32               : 1;
  __REG32   PCKRDY0     : 1;
  __REG32   PCKRDY1     : 1;
  __REG32   PCKRDY2     : 1;
  __REG32   PCKRDY3     : 1;
  __REG32               :20;
} __pmc_ier_bits;

/* AIC Source Mode Register */
typedef struct {
  __REG32 PRIOR            : 3;
  __REG32                  : 2;
  __REG32 SRCTYPE          : 2;
  __REG32                  :25;
} __aicsmrx_bits;

/* AIC Interrupt Status Register */
typedef struct {
  __REG32 IRQID            : 5;
  __REG32                  :27;
} __aicisr_bits;

/* AIC Interrupt Pending Register */
/* AIC Interrupt Mask Register */
/* AIC Interrupt Enable Command Register */
/* AIC Interrupt Disable Command Register */
/* AIC Interrupt Clear Command Register */
/* AIC Interrupt Set Command Register */
typedef struct {
  __REG32   FIQ         : 1;
  __REG32   SYS         : 1;
  __REG32   PIOA_D      : 1;
  __REG32   MPB0        : 1;
  __REG32   MPB1        : 1;
  __REG32   MPB2        : 1;
  __REG32   MPB3        : 1;
  __REG32   MPB4        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   US2         : 1;
  __REG32   MCI0        : 1;
  __REG32   MCI1        : 1;
  __REG32   CAN         : 1;
  __REG32   TWI         : 1;
  __REG32   SPI0        : 1;
  __REG32   SPI1        : 1;
  __REG32   SSC0        : 1;
  __REG32   SSC1        : 1;
  __REG32   AC97        : 1;
  __REG32   TC          : 1;
  __REG32   PWMC        : 1;
  __REG32   EMAC        : 1;
  __REG32               : 1;
  __REG32   ADCC        : 1;
  __REG32   ISI         : 1;
  __REG32   LCDC        : 1;
  __REG32   DMA         : 1;
  __REG32   UDPHS       : 1;
  __REG32   UHP         : 1;
  __REG32   IRQ0        : 1;
  __REG32   IRQ1        : 1;
} __aicipr_bits;

/* AIC Core Interrupt Status Register */
typedef struct {
  __REG32   NFIQ        : 1;
  __REG32   NIRQ        : 1;
  __REG32               :30;
} __aiccisr_bits;


/* AIC Debug Control Register */
typedef struct {
  __REG32   PROT        : 1;
  __REG32   GMSK        : 1;
  __REG32               :30;
} __aicdcr_bits; 

/* AIC Fast Forcing Enable Register */
/* AIC Fast Forcing Disable Register */
/* AIC Fast Forcing Status Register */
typedef struct {
  __REG32               : 1;
  __REG32   SYS         : 1;
  __REG32   PIOA_D      : 1;
  __REG32   MPB0        : 1;
  __REG32   MPB1        : 1;
  __REG32   MPB2        : 1;
  __REG32   MPB3        : 1;
  __REG32   MPB4        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   US2         : 1;
  __REG32   MCI0        : 1;
  __REG32   MCI1        : 1;
  __REG32   CAN         : 1;
  __REG32   TWI         : 1;
  __REG32   SPI0        : 1;
  __REG32   SPI1        : 1;
  __REG32   SSC0        : 1;
  __REG32   SSC1        : 1;
  __REG32   AC97        : 1;
  __REG32   TC          : 1;
  __REG32   PWMC        : 1;
  __REG32   EMAC        : 1;
  __REG32               : 1;
  __REG32   ADCC        : 1;
  __REG32   ISI         : 1;
  __REG32   LCDC        : 1;
  __REG32   DMA         : 1;
  __REG32   UDPHS       : 1;
  __REG32   UHP         : 1;
  __REG32   IRQ0        : 1;
  __REG32   IRQ1        : 1;
} __aicffer_bits;

/*Debug Unit Control Register*/
typedef struct {
  __REG32               : 2;
  __REG32   RSTRX       : 1;
  __REG32   RSTTX       : 1;
  __REG32   RXEN        : 1;
  __REG32   RXDIS       : 1;
  __REG32   TXEN        : 1;
  __REG32   TXDIS       : 1;
  __REG32   RSTSTA      : 1;
  __REG32               :23;
} __dbgu_cr_bits; 

/* Debug Unit Mode Register */
typedef struct {
  __REG32               : 9;
  __REG32   PAR         : 3;
  __REG32               : 2;
  __REG32   CHMODE      : 2;
  __REG32               :16;
} __dbgu_mr_bits;

/* Debug Unit Interrupt Enable Register */
/* Debug Unit Interrupt Disable Register */
/* Debug Unit Interrupt Mask Register */
/* Debug Unit Status Register */
typedef struct {
  __REG32   RXRDY       : 1;
  __REG32   TXRDY       : 1;
  __REG32               : 1;
  __REG32   ENDRX       : 1;
  __REG32   ENDTX       : 1;
  __REG32   OVRE        : 1;
  __REG32   FRAME       : 1;
  __REG32   PARE        : 1;
  __REG32               : 1;
  __REG32   TXEMPTY     : 1;
  __REG32               : 1;
  __REG32   TXBUFE      : 1;
  __REG32   RXBUFF      : 1;
  __REG32               :17;
  __REG32   COMMTX      : 1;
  __REG32   COMMRX      : 1;
} __dbgu_ier_bits;

/* Debug Unit Chip ID Register */
typedef struct {
  __REG32   VERSION     : 5;
  __REG32   EPROC       : 3;
  __REG32   NVPSIZ      : 4;
  __REG32   NVPSIZ2     : 4;
  __REG32   SRAMSIZ     : 4;
  __REG32   ARCH        : 8;
  __REG32   NVPTYP      : 3;
  __REG32   EXT         : 1;
} __dbgu_cidr_bits;

/*Debug Unit Force NTRST Register*/
typedef struct {
  __REG32   FNTRST      : 1;
  __REG32               :31;
} __dbgu_fnr_bits;

/* PIO Enable Register */
/* PIO Disable Register */
/* PIO Status Register */
/* PIO Output Enable Register */
/* PIO Output Disable Register */
/* PIO Output Status Register */
/* PIO Controller Input Filter Enable Register */
/* PIO Controller Input Filter Disable Register */
/* PIO Controller Input Filter Status Register */
/* PIO Set Output Data Register */
/* PIO Clear Output Data Register */
/* PIO Output Data Status Register */
/* PIO Pin Data Status Register */
/* PIO Interrupt Enable Register */
/* PIO Interrupt Disable Register */
/* PIO Interrupt Mask Register */
/* PIO Interrupt Status Register */
/* PIO Multi-driver Enable Register */
/* PIO Multi-driver Disable Register */
/* PIO Multi-driver Status Register */
/* PIO Pull Up Disable Register */
/* PIO Pull Up Enable Register */
/* PIO Pull Up Status Register */
/* PIO Peripheral A Select Register */
/* PIO Peripheral B Select Register */
/* PIO Peripheral A B Status Register */
/* PIO Output Write Enable Register */
/* PIO Output Write Disable Register */
/* PIO Output Write Status Register */
typedef struct {
  __REG32 P0               : 1;
  __REG32 P1               : 1;
  __REG32 P2               : 1;
  __REG32 P3               : 1;
  __REG32 P4               : 1;
  __REG32 P5               : 1;
  __REG32 P6               : 1;
  __REG32 P7               : 1;
  __REG32 P8               : 1;
  __REG32 P9               : 1;
  __REG32 P10              : 1;
  __REG32 P11              : 1;
  __REG32 P12              : 1;
  __REG32 P13              : 1;
  __REG32 P14              : 1;
  __REG32 P15              : 1;
  __REG32 P16              : 1;
  __REG32 P17              : 1;
  __REG32 P18              : 1;
  __REG32 P19              : 1;
  __REG32 P20              : 1;
  __REG32 P21              : 1;
  __REG32 P22              : 1;
  __REG32 P23              : 1;
  __REG32 P24              : 1;
  __REG32 P25              : 1;
  __REG32 P26              : 1;
  __REG32 P27              : 1;
  __REG32 P28              : 1;
  __REG32 P29              : 1;
  __REG32 P30              : 1;
  __REG32 P31              : 1;
} __pioper_bits;

/* SPI Control Register */
typedef struct {
  __REG32 SPIEN            : 1;
  __REG32 SPIDIS           : 1;
  __REG32                  : 5;
  __REG32 SWRST            : 1;
  __REG32                  :16;
  __REG32 LASTXFER         : 1;
  __REG32                  : 7;
} __spicr_bits;

/* SPI Mode Register */
typedef struct {
  __REG32 MSTR             : 1;
  __REG32 PS               : 1;
  __REG32 PCSDEC           : 1;
  __REG32                  : 1;
  __REG32 MODFDIS          : 1;
  __REG32                  : 2;
  __REG32 LLB              : 1;
  __REG32                  : 8;
  __REG32 PCS              : 4;
  __REG32                  : 4;
  __REG32 DLYBCS           : 8;
} __spimr_bits;

/* SPI Receive Data Register */
typedef struct {
  __REG32 RD               :16;
  __REG32 PCS              : 4;
  __REG32                  :12;
} __spirdr_bits;

/* SPI Transmit Data Register */
typedef struct {
  __REG32 TD               :16;
  __REG32 PCS              : 4;
  __REG32                  : 4;
  __REG32 LASTXFER         : 1;
  __REG32                  : 7;
} __spitdr_bits;

/* SPI Status Register */
typedef struct {
  __REG32 RDRF             : 1;
  __REG32 TDRE             : 1;
  __REG32 MODF             : 1;
  __REG32 OVRES            : 1;
  __REG32 ENDRX            : 1;
  __REG32 ENDTX            : 1;
  __REG32 RXBUFF           : 1; 
  __REG32 TXBUFE           : 1; 
  __REG32 NSSR             : 1; 
  __REG32 TXEMPTY          : 1; 
  __REG32                  : 6;
  __REG32 SPIENS           : 1;
  __REG32                  :15;
} __spisr_bits;

/* SPI Interrupt Enable Register */
/* SPI Interrupt Disable Register */
/* SPI Interrupt Mask Register */
typedef struct {
  __REG32 RDRF             : 1;
  __REG32 TDRE             : 1;
  __REG32 MODF             : 1;
  __REG32 OVRES            : 1;
  __REG32 ENDRX            : 1;
  __REG32 ENDTX            : 1;
  __REG32 RXBUFF           : 1; 
  __REG32 TXBUFE           : 1; 
  __REG32 NSSR             : 1; 
  __REG32 TXEMPTY          : 1; 
  __REG32                  :22;
} __spiier_bits;

/* SPI Chip Select Register */
typedef struct {
  __REG32 CPOL             : 1;
  __REG32 NCPHA            : 1;
  __REG32                  : 1;
  __REG32 CSAAT            : 1;
  __REG32 BITS             : 4;
  __REG32 SCBR             : 8;
  __REG32 DLYBS            : 8;
  __REG32 DLYBCT           : 8;
} __spicsrx_bits;


/* TWI Control Register */
typedef struct {
  __REG32   START       : 1;
  __REG32   STOP        : 1;
  __REG32   MSEN        : 1;
  __REG32   MSDIS       : 1;
  __REG32   SVEN        : 1;
  __REG32   SVDIS       : 1;
  __REG32               : 1;
  __REG32   SWRST       : 1;
  __REG32               :24;
} __twi_cr_bits;  

/* TWI Master Mode Register */
typedef struct {
  __REG32               : 8;
  __REG32   IADRSZ      : 2;
  __REG32               : 2;
  __REG32   MREAD       : 1;
  __REG32               : 3;
  __REG32   DADR        : 7;
  __REG32               : 9;
} __twi_mmr_bits; 

/* TWI Slave Mode Register */
typedef struct {
  __REG32               :16;
  __REG32   SADR        : 7;
  __REG32               : 9;
} __twi_smr_bits; 

/*TWI Internal Address Register */
typedef struct {
  __REG32   IADR        :24;
  __REG32               : 8;
} __twi_iadr_bits;

/* TWI Clock Waveform Generator Register */
typedef struct {
  __REG32   CLDIV       : 8;
  __REG32   CHDIV       : 8;
  __REG32   CKDIV       : 3;
  __REG32               :13;
} __twi_cwgr_bits;

/* TWI Status Register */
typedef struct {
  __REG32   TXCOMP      : 1;
  __REG32   RXRDY       : 1;
  __REG32   TXRDY       : 1;
  __REG32   SVREAD      : 1;
  __REG32   SVACC       : 1;
  __REG32   GACC        : 1;
  __REG32   OVRE        : 1;
  __REG32               : 1;
  __REG32   NACK        : 1;
  __REG32   ARBLST      : 1;
  __REG32   SCLWS       : 1;
  __REG32   EOSACC      : 1;
  __REG32   ENDRX       : 1;
  __REG32   ENDTX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32   TXBUFE      : 1;
  __REG32               :16;
} __twi_sr_bits;

/* TWI Interrupt Enable Register*/
/* TWI Interrupt Disable Register*/
/* TWI Interrupt Mask Register*/
typedef struct {
  __REG32   TXCOMP      : 1;
  __REG32   RXRDY       : 1;
  __REG32   TXRDY       : 1;
  __REG32               : 1;
  __REG32   SVACC       : 1;
  __REG32   GACC        : 1;
  __REG32   OVRE        : 1;
  __REG32               : 1;
  __REG32   NACK        : 1;
  __REG32   ARBLST      : 1;
  __REG32   SCLWS       : 1;
  __REG32   EOSACC      : 1;
  __REG32   ENDRX       : 1;
  __REG32   ENDTX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32   TXBUFE      : 1;
  __REG32               :16;
} __twi_ier_bits; 

/* UART Control Register */
typedef struct {
  __REG32                  : 2;
  __REG32 RSTRX            : 1;
  __REG32 RSTTX            : 1;
  __REG32 RXEN             : 1;
  __REG32 RXDIS            : 1;
  __REG32 TXEN             : 1;
  __REG32 TXDIS            : 1;
  __REG32 RSTSTA           : 1;
  __REG32 STTBRK           : 1;
  __REG32 STPBRK           : 1;
  __REG32 STTTO            : 1;
  __REG32 SENDA            : 1;
  __REG32 RSTIT            : 1;
  __REG32 RSTNACK          : 1;
  __REG32 RETTO            : 1;
  __REG32                  : 2;
  __REG32 RTSEN            : 1;
  __REG32 RTSDIS           : 1;
  __REG32                  :12;
} __uscr_bits;

/* UART Mode Register */
typedef struct {
  __REG32 USART_MODE       : 4;
  __REG32 USCLKS           : 2;
  __REG32 CHRL             : 2;
  __REG32 SYNC             : 1;
  __REG32 PAR              : 3;
  __REG32 NBSTOP           : 2;
  __REG32 CHMODE           : 2;
  __REG32 MSBF             : 1;
  __REG32 MODE9            : 1;
  __REG32 CLKO             : 1;
  __REG32 OVER             : 1;
  __REG32 INACK            : 1;
  __REG32 DSNACK           : 1;
  __REG32 VAR_SYNC         : 1;
  __REG32                  : 1;
  __REG32 MAX_ITERATION    : 3;
  __REG32                  : 1;
  __REG32 FILTER           : 1;
  __REG32 MAN              : 1;
  __REG32 MODSYNC          : 1;
  __REG32 ONEBIT           : 1;
} __usmr_bits;

/* UART Interrupt Enable Register */
/* UART Interrupt Disable Register */
/* UART Channel Mask Register */
typedef struct {
  __REG32 RXRDY            : 1;
  __REG32 TXRDY            : 1;
  __REG32 RXBRK            : 1;
  __REG32 ENDRX            : 1;
  __REG32 ENDTX            : 1;
  __REG32 OVRE             : 1;
  __REG32 FRAME            : 1;
  __REG32 PARE             : 1;
  __REG32 TIMEOUT          : 1;
  __REG32 TXEMPTY          : 1;
  __REG32 ITERATION        : 1;
  __REG32 TXBUFE           : 1;
  __REG32 RXBUFF           : 1;
  __REG32 NACK             : 1;
  __REG32                  : 5;
  __REG32 CTSIC            : 1;
  __REG32 MANE             : 1;
  __REG32                  :11;
} __usier_bits;

/* UART Interrupt Satus Register */
typedef struct {
  __REG32 RXRDY            : 1;
  __REG32 TXRDY            : 1;
  __REG32 RXBRK            : 1;
  __REG32 ENDRX            : 1;
  __REG32 ENDTX            : 1;
  __REG32 OVRE             : 1;
  __REG32 FRAME            : 1;
  __REG32 PARE             : 1;
  __REG32 TIMEOUT          : 1;
  __REG32 TXEMPTY          : 1;
  __REG32 ITERATION        : 1;
  __REG32 TXBUFE           : 1;
  __REG32 RXBUFF           : 1;
  __REG32 NACK             : 1;
  __REG32                  : 5;
  __REG32 CTSIC            : 1;
  __REG32                  : 3;
  __REG32 CTS              : 1;
  __REG32 MANERR           : 1;
  __REG32                  : 7;
} __uscsr_bits;

/* UART Receiver Holding Register */
typedef struct {
  __REG32 RXCHR            : 9;
  __REG32                  : 6;
  __REG32 RXSYNH           : 1;
  __REG32                  :16;
} __usrhr_bits;

/* UART Transmitter Holding Register */
typedef struct {
  __REG32 TXCHR            : 9;
  __REG32                  : 6;
  __REG32 TXSYNH           : 1;
  __REG32                  :16;
} __usthr_bits;

/* UART Baud Rate Generator Register */
typedef struct {
  __REG32 CD               :16;
  __REG32 FP               : 3;
  __REG32                  :13;
} __usbrgr_bits;



/* USART FI DI RATIO Register */
typedef struct {
  __REG32 FI_DI_RATIO      :11;
  __REG32                  :21;
} __usfidi_bits;
             
/* USART Manchester Configuration Register */
typedef struct {
  __REG32 TX_PL            : 4;
  __REG32                  : 4;
  __REG32 TX_PP            : 2;
  __REG32                  : 2;
  __REG32 TX_MPOL          : 1;
  __REG32                  : 3;
  __REG32 RX_PL            : 4;
  __REG32                  : 4;
  __REG32 RX_PP            : 2;
  __REG32                  : 2;
  __REG32 RX_MPOL          : 1;
  __REG32                  : 1;
  __REG32 DRIFT            : 1;
  __REG32                  : 1;
} __usman_bits;

/* SSC Control Register */
typedef struct {
  __REG32   RXEN        : 1;
  __REG32   RXDIS       : 1;
  __REG32               : 6;
  __REG32   TXEN        : 1;
  __REG32   TXDIS       : 1;
  __REG32               : 5;
  __REG32   SWRST       : 1;
  __REG32               :16;
} __ssc_cr_bits;

/* SSC Clock Mode Register */
typedef struct {
  __REG32   DIV         :12;
  __REG32               :20;
} __ssc_cmr_bits;

/* SSC Receive Clock Mode Register */
typedef struct {
  __REG32   CKS         : 2;
  __REG32   CKO         : 3;
  __REG32   CKI         : 1;
  __REG32   CKG         : 2;
  __REG32   START       : 4;
  __REG32   STOP        : 1;
  __REG32               : 3;
  __REG32   STDDLY      : 8;
  __REG32   PERIOD      : 8;
} __ssc_rcmr_bits;

/* SSC Receive Frame Mode Register */
typedef struct {
  __REG32   DATLEN      : 5;
  __REG32   LOOP        : 1;
  __REG32               : 1;
  __REG32   MSBF        : 1;
  __REG32   DATNB       : 4;
  __REG32               : 4;
  __REG32   FSLEN       : 4;
  __REG32   FSOS        : 3;
  __REG32               : 1;
  __REG32   FSEDGE      : 1;
  __REG32               : 7;
} __ssc_rfmr_bits;

/* SSC Transmit Clock Mode Register */
typedef struct {
  __REG32   CKS         : 2;
  __REG32   CKO         : 3;
  __REG32   CKI         : 1;
  __REG32   CKG         : 2;
  __REG32   START       : 4;
  __REG32               : 4;
  __REG32   STTDLY      : 8;
  __REG32   PERIOD      : 8;
} __ssc_tcmr_bits;

/* SSC Transmit Frame Mode Register */
typedef struct {
  __REG32   DATLEN      : 5;
  __REG32   DATDEF      : 1;
  __REG32               : 1;
  __REG32   MSBF        : 1;
  __REG32   DATNB       : 4;
  __REG32               : 4;
  __REG32   FSLEN       : 4;
  __REG32   FSOS        : 3;
  __REG32   FSDEN       : 1;
  __REG32   FSEDGE      : 1;
  __REG32               : 7;
} __ssc_tfmr_bits;

/* SSC Status Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   ENDTX       : 1;
  __REG32   TXBUFE      : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32   ENDRX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32   CP0         : 1;
  __REG32   CP1         : 1;
  __REG32   TXSYN       : 1;
  __REG32   RXSYN       : 1;
  __REG32               : 4;
  __REG32   TXEN        : 1;
  __REG32   RXEN        : 1;
  __REG32               :14;
} __ssc_sr_bits;

/* SSC Interrupt Enable Register*/
/* SSC Interrupt Disable Register*/
/* SSC Interrupt Mask Register*/
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   ENDTX       : 1;
  __REG32   TXBUFE      : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32   ENDRX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32   CP0         : 1;
  __REG32   CP1         : 1;
  __REG32   TXSYN       : 1;
  __REG32   RXSYN       : 1;
  __REG32               :20;
} __ssc_ier_bits;

/* AC’97 Controller Mode Register */
typedef struct {
  __REG32   ENA         : 1;
  __REG32   WRST        : 1;
  __REG32   VRA         : 1;
  __REG32               :29;
} __ac97c_mr_bits;

/*AC’97 Controller Input Channel Assignment Register*/
/*AC’97 Controller Output Channel Assignment Register*/
typedef struct {
  __REG32   CHID3       : 3;
  __REG32   CHID4       : 3;
  __REG32   CHID5       : 3;
  __REG32   CHID6       : 3;
  __REG32   CHID7       : 3;
  __REG32   CHID8       : 3;
  __REG32   CHID9       : 3;
  __REG32   CHID10      : 3;
  __REG32   CHID11      : 3;
  __REG32   CHID12      : 3;
  __REG32               : 2;
} __ac97c_ca_bits;

/* AC’97 Controller Codec Channel Receive Holding Register */
typedef struct {
  __REG32   SDATA       :16;
  __REG32               :16;
} __ac97c_corhr_bits;

/* AC’97 Controller Codec Channel Transmit Holding Register */
typedef struct {
  __REG32   CDATA       :16;
  __REG32   CADDR       : 7;
  __REG32   READ        : 1;
  __REG32               : 8;
} __ac97c_cothr_bits;

/* AC’97 Controller Channel A, Channel B, Channel C Receive Holding Register */
typedef struct {
  __REG32   RDATA       :20;
  __REG32               :12;
} __ac97c_cxrhr_bits;

/* AC’97 Controller Channel A, Channel B, Channel C Transmit Holding Register */
typedef struct {
  __REG32   TDATA       :20;
  __REG32               :12;
} __ac97c_cxthr_bits;

/* AC’97 Controller Channel A Status Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               : 4;
  __REG32   ENDTX       : 1;
  __REG32   TXBUFE      : 1;
  __REG32               : 2;
  __REG32   ENDRX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32               :16;
} __ac97c_casr_bits;

/* AC’97 Controller Channel B Status Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :26;
} __ac97c_cbsr_bits;

/* AC’97 Controller Channel C Status Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :26;
} __ac97c_ccsr_bits;

/* AC’97 Controller Codec Channel Status Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :26;
} __ac97c_cosr_bits;

/* AC’97 Controller Channel A Mode Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               : 4;
  __REG32   ENDTX       : 1;
  __REG32   TXBUFE      : 1;
  __REG32               : 2;
  __REG32   ENDRX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32   SIZE        : 2;
  __REG32   CEM         : 1;
  __REG32               : 2;
  __REG32   CEN         : 1;
  __REG32   PDCEN       : 1;
  __REG32               : 9;
} __ac97c_camr_bits;

/* AC’97 Controller Channel B Mode Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :10;
  __REG32   SIZE        : 2;
  __REG32   CEM         : 1;
  __REG32               : 2;
  __REG32   CEN         : 1;
  __REG32               :10;
} __ac97c_cbmr_bits;

/*  AC’97 Controller Channel C Mode Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :10;
  __REG32   SIZE        : 2;
  __REG32   CEM         : 1;
  __REG32               : 2;
  __REG32   CEN         : 1;
  __REG32               :10;
} __ac97c_ccmr_bits;

/* AC’97 Controller Codec Channel Mode Register */
typedef struct {
  __REG32   TXRDY       : 1;
  __REG32   TXEMPTY     : 1;
  __REG32   UNRUN       : 1;
  __REG32               : 1;
  __REG32   RXRDY       : 1;
  __REG32   OVRUN       : 1;
  __REG32               :26;
} __ac97c_comr_bits;

/* AC’97 Controller Status Register */
/* AC’97 Controller Interrupt Enable Register */
/* AC’97 Controller Interrupt Disable Register */
/* AC’97 Controller Interrupt Mask Register */
typedef struct {
  __REG32   SOF         : 1;
  __REG32   WKUP        : 1;
  __REG32   COEVT       : 1;
  __REG32   CAEVT       : 1;
  __REG32   CBEVT       : 1;
  __REG32   CCEVT       : 1;
  __REG32               :26;
} __ac97c_sr_bits;


/* TC Block Control Register */
typedef struct {
  __REG32 SYNC             : 1;
  __REG32                  :31;
} __tc_bcr_bits;

/* TC Block Mode Register */
typedef struct {
  __REG32 TC0XC0S          : 2;
  __REG32 TC1XC1S          : 2;
  __REG32 TC2XC2S          : 2;
  __REG32                  :26;
} __tc_bmr_bits;

/* TC Channel Control Register */
typedef struct {
  __REG32 CLKEN            : 1;
  __REG32 CLKDIS           : 1;
  __REG32 SWTRG            : 1;
  __REG32                  :29;
} __tc_ccr_bits;

/* TC Channel Mode Register: Capture Mode */
/* TC Channel Mode Register: Waveform Mode */
typedef union {
  /*TCx_CMR*/
  struct {
    __REG32 TCCLKS           : 3;
    __REG32 CLKI             : 1;
    __REG32 BURST            : 2;
    __REG32 LDBSTOP          : 1;
    __REG32 LDBDIS           : 1;
    __REG32 ETRGEDG          : 2;
    __REG32 ABETRG           : 1;
    __REG32                  : 3;
    __REG32 CPCTRG           : 1;
    __REG32 WAVE             : 1;
    __REG32 LDRA             : 2;
    __REG32 LDRB             : 2;
    __REG32                  :12;
  };
  /*TCx_CMR*/
  struct {
    __REG32 _TCCLKS          : 3;
    __REG32 _CLKI            : 1;
    __REG32 _BURST           : 2;
    __REG32 CPCSTOP          : 1;
    __REG32 CPCDIS           : 1;
    __REG32 EEVTEDG          : 2;
    __REG32 EEVT             : 2;
    __REG32 ENETRG           : 1;
    __REG32 WAVSEL           : 2;
    __REG32 _WAVE            : 1;
    __REG32 ACPA             : 2;
    __REG32 ACPC             : 2;
    __REG32 AEEVT            : 2;
    __REG32 ASWTRG           : 2;
    __REG32 BCPB             : 2;
    __REG32 BCPC             : 2;
    __REG32 BEEVT            : 2;
    __REG32 BSWTRG           : 2;
  };
} __tc_cmr_bits;

/* TC Status Register */
typedef struct {
  __REG32 COVFS            : 1;
  __REG32 LOVRS            : 1;
  __REG32 CPAS             : 1;
  __REG32 CPBS             : 1;
  __REG32 CPCS             : 1;
  __REG32 LDRAS            : 1;
  __REG32 LDRBS            : 1;
  __REG32 ETRGS            : 1;
  __REG32                  : 8;
  __REG32 CLKSTA           : 1;
  __REG32 MTIOA            : 1;
  __REG32 MTIOB            : 1;
  __REG32                  :13;
} __tc_sr_bits;

/* TC Interrupt Enable Register */
/* TC Interrupt Disable Register */
/* TC Interrupt Mask Register */
typedef struct {
  __REG32 COVFS            : 1;
  __REG32 LOVRS            : 1;
  __REG32 CPAS             : 1;
  __REG32 CPBS             : 1;
  __REG32 CPCS             : 1;
  __REG32 LDRAS            : 1;
  __REG32 LDRBS            : 1;
  __REG32 ETRGS            : 1;
  __REG32                  : 24;
} __tc_ier_bits;

/* CAN Mode Register */
typedef struct {
  __REG32   CANEN       : 1;
  __REG32   LPM         : 1;
  __REG32   ABM         : 1;
  __REG32   OVL         : 1;
  __REG32   TEOF        : 1;
  __REG32   TTM         : 1;
  __REG32   TIMFRZ      : 1;
  __REG32   DRPT        : 1;
  __REG32               :24;
} __can_mr_bits;

/* CAN Interrupt Enable Register */
/* CAN Interrupt Disable Register */
/* CAN Interrupt Mask Register */
typedef struct {
  __REG32   MB0         : 1;
  __REG32   MB1         : 1;
  __REG32   MB2         : 1;
  __REG32   MB3         : 1;
  __REG32   MB4         : 1;
  __REG32   MB5         : 1;
  __REG32   MB6         : 1;
  __REG32   MB7         : 1;
  __REG32   MB8         : 1;
  __REG32   MB9         : 1;
  __REG32   MB10        : 1;
  __REG32   MB11        : 1;
  __REG32   MB12        : 1;
  __REG32   MB13        : 1;
  __REG32   MB14        : 1;
  __REG32   MB15        : 1;
  __REG32   ERRA        : 1;
  __REG32   WARN        : 1;
  __REG32   ERRP        : 1;
  __REG32   BOFF        : 1;
  __REG32   SLEEP       : 1;
  __REG32   WAKEUP      : 1;
  __REG32   TOVF        : 1;
  __REG32   TSTP        : 1;
  __REG32   CERR        : 1;
  __REG32   SERR        : 1;
  __REG32   AERR        : 1;
  __REG32   FERR        : 1;
  __REG32   BERR        : 1;
  __REG32               : 3;
} __can_ier_bits;

/* CAN Status Register */
typedef struct {
  __REG32   MB0         : 1;
  __REG32   MB1         : 1;
  __REG32   MB2         : 1;
  __REG32   MB3         : 1;
  __REG32   MB4         : 1;
  __REG32   MB5         : 1;
  __REG32   MB6         : 1;
  __REG32   MB7         : 1;
  __REG32   MB8         : 1;
  __REG32   MB9         : 1;
  __REG32   MB10        : 1;
  __REG32   MB11        : 1;
  __REG32   MB12        : 1;
  __REG32   MB13        : 1;
  __REG32   MB14        : 1;
  __REG32   MB15        : 1;
  __REG32   ERRA        : 1;
  __REG32   WARN        : 1;
  __REG32   ERRP        : 1;
  __REG32   BOFF        : 1;
  __REG32   SLEEP       : 1;
  __REG32   WAKEUP      : 1;
  __REG32   TOVF        : 1;
  __REG32   TSTP        : 1;
  __REG32   CERR        : 1;
  __REG32   SERR        : 1;
  __REG32   AERR        : 1;
  __REG32   FERR        : 1;
  __REG32   BERR        : 1;
  __REG32   RBSY        : 1;
  __REG32   TBSY        : 1;
  __REG32   OVLSY       : 1;
} __can_sr_bits;

/* CAN Baudrate Register */
typedef struct {
  __REG32   PHASE2      : 3;
  __REG32               : 1;
  __REG32   PHASE1      : 3;
  __REG32               : 1;
  __REG32   PROPAG      : 3;
  __REG32               : 1;
  __REG32   SJW         : 2;
  __REG32               : 2;
  __REG32   BRP         : 7;
  __REG32               : 1;
  __REG32   SMP         : 1;
  __REG32               : 7;
} __can_br_bits;

/* CAN Error Counter Register */
typedef struct {
  __REG32   REC         : 8;
  __REG32               : 8;
  __REG32   TEC         : 8;
  __REG32               : 8;
} __can_ecr_bits;

/* CAN Transfer Command Register */
typedef struct {
  __REG32   MB0         : 1;
  __REG32   MB1         : 1;
  __REG32   MB2         : 1;
  __REG32   MB3         : 1;
  __REG32   MB4         : 1;
  __REG32   MB5         : 1;
  __REG32   MB6         : 1;
  __REG32   MB7         : 1;
  __REG32   MB8         : 1;
  __REG32   MB9         : 1;
  __REG32   MB10        : 1;
  __REG32   MB11        : 1;
  __REG32   MB12        : 1;
  __REG32   MB13        : 1;
  __REG32   MB14        : 1;
  __REG32   MB15        : 1;
  __REG32               :15;
  __REG32   TIMRST      : 1;
} __can_tcr_bits;

/* CAN Abort Command Register */
typedef struct {
  __REG32   MB0         : 1;
  __REG32   MB1         : 1;
  __REG32   MB2         : 1;
  __REG32   MB3         : 1;
  __REG32   MB4         : 1;
  __REG32   MB5         : 1;
  __REG32   MB6         : 1;
  __REG32   MB7         : 1;
  __REG32   MB8         : 1;
  __REG32   MB9         : 1;
  __REG32   MB10        : 1;
  __REG32   MB11        : 1;
  __REG32   MB12        : 1;
  __REG32   MB13        : 1;
  __REG32   MB14        : 1;
  __REG32   MB15        : 1;
  __REG32               :16;
} __can_acr_bits;

/* CAN Message Mode Register */
typedef struct {
  __REG32   MTIMEMARK0  : 1;
  __REG32   MTIMEMARK1  : 1;
  __REG32   MTIMEMARK2  : 1;
  __REG32   MTIMEMARK3  : 1;
  __REG32   MTIMEMARK4  : 1;
  __REG32   MTIMEMARK5  : 1;
  __REG32   MTIMEMARK6  : 1;
  __REG32   MTIMEMARK7  : 1;
  __REG32   MTIMEMARK8  : 1;
  __REG32   MTIMEMARK9  : 1;
  __REG32   MTIMEMARK10 : 1;
  __REG32   MTIMEMARK11 : 1;
  __REG32   MTIMEMARK12 : 1;
  __REG32   MTIMEMARK13 : 1;
  __REG32   MTIMEMARK14 : 1;
  __REG32   MTIMEMARK15 : 1;
  __REG32   PRIOR       : 4;
  __REG32               : 4;
  __REG32   MOT         : 3;
  __REG32               : 5;
} __can_mmrx_bits;

/* CAN Message Acceptance Mask Register */
/* CAN Message ID Register */
typedef struct {
  __REG32   MMIDvB      :18;
  __REG32   MIDvA       :11;
  __REG32   MIDE        : 1;
  __REG32               : 2;
} __can_mamx_bits;


/* CAN Message Family ID Register */
typedef struct {
  __REG32   MFID        :29;
  __REG32               : 3;
} __can_mfidx_bits;

/* CAN Message Status Register */
typedef struct {
  __REG32   MTIMESTAMP0  : 1;
  __REG32   MTIMESTAMP1  : 1;
  __REG32   MTIMESTAMP2  : 1;
  __REG32   MTIMESTAMP3  : 1;
  __REG32   MTIMESTAMP4  : 1;
  __REG32   MTIMESTAMP5  : 1;
  __REG32   MTIMESTAMP6  : 1;
  __REG32   MTIMESTAMP7  : 1;
  __REG32   MTIMESTAMP8  : 1;
  __REG32   MTIMESTAMP9  : 1;
  __REG32   MTIMESTAMP10 : 1;
  __REG32   MTIMESTAMP11 : 1;
  __REG32   MTIMESTAMP12 : 1;
  __REG32   MTIMESTAMP13 : 1;
  __REG32   MTIMESTAMP14 : 1;
  __REG32   MTIMESTAMP15 : 1;
  __REG32   MDLC         : 4;
  __REG32   MRTR         : 1;
  __REG32                : 1;
  __REG32   MABT         : 1;
  __REG32   MRDY         : 1;
  __REG32   MMI          : 1;
  __REG32                : 7;
} __can_msrx_bits;

/* CAN Message Control Register */
typedef struct {
  __REG32               :16;
  __REG32   MDLC        : 4;
  __REG32   MRTR        : 1;
  __REG32               : 1;
  __REG32   MACR        : 1;
  __REG32   MTCR        : 1;
  __REG32               : 8;
} __can_mcrx_bits;


/* PWM Mode Register */
typedef struct {
  __REG32   DIVA        : 8;
  __REG32   PREA        : 4;
  __REG32               : 4;
  __REG32   DIVB        : 8;
  __REG32   PREB        : 4;
  __REG32               : 4;
} __pwm_mr_bits;

/* PWM Enable Register */
/* PWM Disable Register */
/* PWM Status Register */
typedef struct {
  __REG32   CHID0       : 1;
  __REG32   CHID1       : 1;
  __REG32   CHID2       : 1;
  __REG32   CHID3       : 1;
  __REG32               :28;
} __pwm_ena_bits;

/* PWM Interrupt Enable Register */
/* PWM Interrupt Disable Register */
/* PWM Interrupt Mask Register */
/* PWM Interrupt Status Register */
typedef struct {
  __REG32   CHID0       : 1;
  __REG32   CHID1       : 1;
  __REG32   CHID2       : 1;
  __REG32   CHID3       : 1;
  __REG32               :28;
} __pwm_ier_bits;

/* PWM Channel Mode Register */
typedef struct {
  __REG32   CPRE        : 4;
  __REG32               : 4;
  __REG32   CALG        : 1;
  __REG32   CPOL        : 1;
  __REG32   CPD         : 1;
  __REG32               :21;
} __pwm_cmrx_bits;


/* MCI Control Register */
typedef struct {
  __REG32   MCIEN       : 1;
  __REG32   MCIDIS      : 1;
  __REG32   PWSEN       : 1;
  __REG32   PWSDIS      : 1;
  __REG32               : 3;
  __REG32   SWRST       : 1;
  __REG32               :24;
} __mci_cr_bits;

/* MCI Mode Register */
typedef struct {
  __REG32   CLKDIV      : 8;
  __REG32   PWSDIV      : 3;
  __REG32   RDPROOF     : 1;
  __REG32   WRPROOF     : 1;
  __REG32   PDCFBYTE    : 1;
  __REG32   PDCPADV     : 1;
  __REG32   PDCMODE     : 1;
  __REG32   BLKLEN      :16;
} __mci_mr_bits;

/* MCI Data Timeout Register */
typedef struct {
  __REG32   DTOCYC      : 4;
  __REG32   DTOMUL      : 3;
  __REG32               :25;
} __mci_dtor_bits;

/* MCI SDCard/SDIO Register */
typedef struct {
  __REG32   SDCSEL      : 2;
  __REG32               : 5;
  __REG32   SDCBUS      : 1;
  __REG32               :24;
} __mci_sdcr_bits;

/* MCI Command Register */
typedef struct {
  __REG32   CMDNB       : 6;
  __REG32   RSPTYP      : 2;
  __REG32   SPCMD       : 3;
  __REG32   OPDCMD      : 1;
  __REG32   MAXLAT      : 1;
  __REG32               : 3;
  __REG32   TRCMD       : 2;
  __REG32   TRDIR       : 1;
  __REG32   TRTYP       : 3;
  __REG32               : 2;
  __REG32   IOSPCMD     : 2;
  __REG32               : 6;
} __mci_cmdr_bits;

/* MCI Block Register */
typedef struct {
  __REG32   BCNT        :16;
  __REG32   BLKLEN      :16;
} __mci_blkr_bits;

/* MCI Status Register */
/* MCI Interrupt Enable Register */
/* MCI Interrupt Disable Register */
/* MCI Interrupt Mask Register */
typedef struct {
  __REG32   CMDRDY      : 1;
  __REG32   RXRDY       : 1;
  __REG32   TXRDY       : 1;
  __REG32   BLKE        : 1;
  __REG32   DTIP        : 1;
  __REG32   NOTBUSY     : 1;
  __REG32   ENDRX       : 1;
  __REG32   ENDTX       : 1;
  __REG32   SDIOIRQA    : 1;
  __REG32               : 5;
  __REG32   RXBUFF      : 1;
  __REG32   TXBUFE      : 1;
  __REG32   RINDE       : 1;
  __REG32   RDIRE       : 1;
  __REG32   RCRCE       : 1;
  __REG32   RENDE       : 1;
  __REG32   RTOE        : 1;
  __REG32   DCRCE       : 1;
  __REG32   DTOE        : 1;
  __REG32               : 7;
  __REG32   OVRE        : 1;
  __REG32   UNRE        : 1;
} __mci_sr_bits;

/* EMAC Network Control Register */
typedef struct {
  __REG32   LB          : 1;
  __REG32   LLB         : 1;
  __REG32   RE          : 1;
  __REG32   TE          : 1;
  __REG32   MPE         : 1;
  __REG32   CLRSTAT     : 1;
  __REG32   INCSTAT     : 1;
  __REG32   WESTAT      : 1;
  __REG32   BP          : 1;
  __REG32   TSTART      : 1;
  __REG32   THALT       : 1;
  __REG32               :21;
} __emac_ncr_bits;

/* EMAC Network Configuration Register */
typedef struct {
  __REG32   SPD         : 1;
  __REG32   FD          : 1;
  __REG32               : 1;
  __REG32   JFRAME      : 1;
  __REG32   CAF         : 1;
  __REG32   NBC         : 1;
  __REG32   MTI         : 1;
  __REG32   UNI         : 1;
  __REG32   BIG         : 1;
  __REG32               : 1;
  __REG32   CLK         : 2;
  __REG32   RTY         : 1;
  __REG32   PAE         : 1;
  __REG32   RBOF        : 2;
  __REG32   RLCE        : 1;
  __REG32   DRFCS       : 1;
  __REG32   EFRHD       : 1;
  __REG32   IRXFCS      : 1;
  __REG32               :12;
} __emac_ncfg_bits;

/* EMAC Network Status Register */
typedef struct {
  __REG32               : 1;
  __REG32   MDIO        : 1;
  __REG32   IDLE        : 1;
  __REG32               :29;
} __emac_nsr_bits;

/* EMAC Transmit Status Register */
typedef struct {
  __REG32   UBR         : 1;
  __REG32   COL         : 1;
  __REG32   RLE         : 1;
  __REG32   TGO         : 1;
  __REG32   BEX         : 1;
  __REG32   COMP        : 1;
  __REG32   UND         : 1;
  __REG32               :25;
} __emac_tsr_bits;

/* EMAC Receive Status Register */
typedef struct {
  __REG32   BNA         : 1;
  __REG32   REC         : 1;
  __REG32   OVR         : 1;
  __REG32               :29;
} __emac_rsr_bits;

/* EMAC Interrupt Status Register */
/* EMAC Interrupt Enable Register */
/* EMAC Interrupt Disable Register */
/* EMAC Interrupt Mask Register */
typedef struct {
  __REG32   MFD         : 1;
  __REG32   RCOMP       : 1;
  __REG32   RXUBR       : 1;
  __REG32   TXUBR       : 1;
  __REG32   TUND        : 1;
  __REG32   RLE         : 1;
  __REG32   TXERR       : 1;
  __REG32   TCOMP       : 1;
  __REG32               : 2;
  __REG32   ROVR        : 1;
  __REG32   HRESP       : 1;
  __REG32   PFR         : 1;
  __REG32   PTZ         : 1;
  __REG32               :18;
} __emac_isr_bits;

/* EMAC PHY Maintenance Register */
typedef struct {
  __REG32   DATA        :16;
  __REG32   CODE        : 2;
  __REG32   REGA        : 5;
  __REG32   PHYA        : 5;
  __REG32   RW          : 2;
  __REG32   SOF         : 2;
} __emac_man_bits;

/* EMAC User Input/Output Register */
typedef struct {
  __REG32   RMII        : 1;
  __REG32   CLKEN       : 1;
  __REG32               :30;
} __emac_usrio_bits;

/*EMAC Frames Transmitted OK Register */
typedef struct {
  __REG32   FTOK        :24;
  __REG32               : 8;
} __emac_fto_bits;

/* EMAC Frames Received OK Register */
typedef struct {
  __REG32   FROK        :24;
  __REG32               : 8;
} __emac_fro_bits;

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


/* UDPHS Control Register */
typedef struct {
  __REG32   DEV_ADDR    : 7;
  __REG32   FADDR_EN    : 1;
  __REG32   EN_UDPHS    : 1;
  __REG32   DETACH      : 1;
  __REG32   REWAKEUP    : 1;
  __REG32   PULLD_DIS   : 1;
  __REG32               :20;
} __udphs_ctrl_bits;

/* UDPHS Frame Number Register */
typedef struct {
  __REG32   MICRO_FRAME_NUM   : 3;
  __REG32   FRAME_NUMBER      :11;
  __REG32                     :17;
  __REG32   FNUM_ERR          : 1;
} __udphs_fnum_bits;

/* UDPHS Interrupt Enable Register */
typedef struct {
  __REG32               : 1;
  __REG32   DET_SUSPD   : 1;
  __REG32   MICRO_SOF   : 1;
  __REG32   INT_SOF     : 1;
  __REG32   ENDRESET    : 1;
  __REG32   WAKE_UP     : 1;
  __REG32   ENDOFRSM    : 1;
  __REG32   UPSTR_RES   : 1;
  __REG32   EPT_0       : 1;
  __REG32   EPT_1       : 1;
  __REG32   EPT_2       : 1;
  __REG32   EPT_3       : 1;
  __REG32   EPT_4       : 1;
  __REG32   EPT_5       : 1;
  __REG32   EPT_6       : 1;
  __REG32   EPT_7       : 1;
  __REG32               : 9;
  __REG32   DMA_INT_1   : 1;
  __REG32   DMA_INT_2   : 1;
  __REG32   DMA_INT_3   : 1;
  __REG32   DMA_INT_4   : 1;
  __REG32   DMA_INT_5   : 1;
  __REG32   DMA_INT_6   : 1;
  __REG32               : 1;
} __udphs_ien_bits;

/* UDPHS Interrupt Status Register */
typedef struct {
  __REG32   SPEED       : 1;
  __REG32   DET_SUSPD   : 1;
  __REG32   MICRO_SOF   : 1;
  __REG32   INT_SOF     : 1;
  __REG32   ENDRESET    : 1;
  __REG32   WAKE_UP     : 1;
  __REG32   ENDOFRSM    : 1;
  __REG32   UPSTR_RES   : 1;
  __REG32   EPT_0       : 1;
  __REG32   EPT_1       : 1;
  __REG32   EPT_2       : 1;
  __REG32   EPT_3       : 1;
  __REG32   EPT_4       : 1;
  __REG32   EPT_5       : 1;
  __REG32   EPT_6       : 1;
  __REG32   EPT_7       : 1;
  __REG32               : 9;
  __REG32   DMA_INT_1   : 1;
  __REG32   DMA_INT_2   : 1;
  __REG32   DMA_INT_3   : 1;
  __REG32   DMA_INT_4   : 1;
  __REG32   DMA_INT_5   : 1;
  __REG32   DMA_INT_6   : 1;
  __REG32               : 1;
} __udphs_intsta_bits;

/* UDPHS Clear Interrupt Register */
typedef struct {
  __REG32               : 1;
  __REG32   DET_SUSPD   : 1;
  __REG32   MICRO_SOF   : 1;
  __REG32   INT_SOF     : 1;
  __REG32   ENDRESET    : 1;
  __REG32   WAKE_UP     : 1;
  __REG32   ENDOFRSM    : 1;
  __REG32   UPSTR_RES   : 1;
  __REG32               :24;
} __udphs_clrint_bits;

/* UDPHS Endpoints Reset Register */
typedef struct {
  __REG32   EPT_0       : 1;
  __REG32   EPT_1       : 1;
  __REG32   EPT_2       : 1;
  __REG32   EPT_3       : 1;
  __REG32   EPT_4       : 1;
  __REG32   EPT_5       : 1;
  __REG32   EPT_6       : 1;
  __REG32   EPT_7       : 1;
  __REG32               :24;
} __udphs_eptrst_bits;

/* UDPHS Test Register */
typedef struct {
  __REG32   SPEED_CFG   : 2;
  __REG32   TST_J       : 1;
  __REG32   TST_K       : 1;
  __REG32   TST_PKT     : 1;
  __REG32   OPMODE2     : 1;
  __REG32               :26;
} __udphs_tst_bits;

/* UDPHS Endpoint Configuration Register */
typedef struct {
  __REG32   EPT_SIZE    : 3;
  __REG32   EPT_DIR     : 1;
  __REG32   EPT_TYPE    : 2;
  __REG32   BK_NUMBER   : 2;
  __REG32   NB_TRANS    : 2;
  __REG32               :21;
  __REG32   EPT_MAPD    : 1;
} __udphs_eptcfgx_bits;

/* UDPHS Endpoint Control Disable Register */
typedef struct {
  __REG32   EPT_DISABL  : 1;
  __REG32   AUTO_VALID  : 1;
  __REG32               : 1;
  __REG32   INTDIS_DMA  : 1;
  __REG32   NYET_DIS    : 1;
  __REG32               : 1;
  __REG32   DATAX_RX    : 1;
  __REG32   MDATA_RX    : 1;
  __REG32   ERR_OVFLW   : 1;
  __REG32   RX_BK_RDY   : 1;
  __REG32   TX_COMPLT   : 1;
  __REG32   TX_PK_RDY   : 1;
  __REG32   RX_SETUP    : 1;
  __REG32   STALL_SNT   : 1;
  __REG32   NAK_IN      : 1;
  __REG32   NAK_OUT     : 1;
  __REG32               : 2;
  __REG32   BUSY_BANK   : 1;
  __REG32               :12;
  __REG32   SHRT_PCKT   : 1;
} __udphs_eptctldisx_bits;

/* UDPHS Endpoint Control Enable Register */
/* UDPHS Endpoint Control Register */
typedef struct {
  __REG32   EPT_ENABL   : 1;
  __REG32   AUTO_VALID  : 1;
  __REG32               : 1;
  __REG32   INTDIS_DMA  : 1;
  __REG32   NYET_DIS    : 1;
  __REG32               : 1;
  __REG32   DATAX_RX    : 1;
  __REG32   MDATA_RX    : 1;
  __REG32   ERR_OVFLW   : 1;
  __REG32   RX_BK_RDY   : 1;
  __REG32   TX_COMPLT   : 1;
  __REG32   TX_PK_RDY   : 1;
  __REG32   RX_SETUP    : 1;
  __REG32   STALL_SNT   : 1;
  __REG32   NAK_IN      : 1;
  __REG32   NAK_OUT     : 1;
  __REG32               : 2;
  __REG32   BUSY_BANK   : 1;
  __REG32               :12;
  __REG32   SHRT_PCKT   : 1;
} __udphs_eptctlx_bits;

/* UDPHS Endpoint Set Status Register */
typedef struct {
  __REG32               : 5;
  __REG32   FRCESTALL   : 1;
  __REG32               : 3;
  __REG32   KILL_BANK   : 1;
  __REG32               : 1;
  __REG32   TX_PK_RDY   : 1;
  __REG32               :20;
} __udphs_eptsetstax_bits;

/* UDPHS Endpoint Clear Status Register */
typedef struct {
  __REG32               : 5;
  __REG32   FRCESTALL   : 1;
  __REG32   TOGGLESQ    : 1;
  __REG32               : 2;
  __REG32   RX_BK_RDY   : 1;
  __REG32   TX_COMPLT   : 1;
  __REG32               : 1;
  __REG32   RX_SETUP    : 1;
  __REG32   STALL_SNT   : 1;
  __REG32   NAK_IN      : 1;
  __REG32   NAK_OUT     : 1;
  __REG32               :16;
} __udphs_eptclrstax_bits;

/* UDPHS Endpoint Status Register */
typedef struct {
  __REG32                 : 5;
  __REG32   FRCESTALL     : 1;
  __REG32   TOGGLESQ_STA  : 2;
  __REG32   ERR_OVFLW     : 1;
  __REG32   RX_BK_RDY     : 1;
  __REG32   TX_COMPLT     : 1;
  __REG32   TX_PK_RDY     : 1;
  __REG32   RX_SETUP      : 1;
  __REG32   STALL_SNT     : 1;
  __REG32   NAK_IN        : 1;
  __REG32   NAK_OUT       : 1;
  __REG32   CURRENT_BANK  : 2;
  __REG32   BUSY_BANK_STA : 2;
  __REG32   BYTE_COUNT    :11;
  __REG32   SHRT_PCKT     : 1;
} __udphs_eptstax_bits;

/* UDPHS DMA Channel Control Register */
typedef struct {
  __REG32   CHANN_ENB   : 1;
  __REG32   LDNXT_DSC   : 1;
  __REG32   END_TR_EN   : 1;
  __REG32   END_B_EN    : 1;
  __REG32   END_TR_IT   : 1;
  __REG32   END_BUFFIT  : 1;
  __REG32   DESC_LD_IT  : 1;
  __REG32   BURST_LCK   : 1;
  __REG32               : 8;
  __REG32   BUFF_LENGTH :16;
} __udphs_dmacontrolx_bits;

/* UDPHS DMA Channel Status Register */
typedef struct {
  __REG32   CHANN_ENB   : 1;
  __REG32   CHANN_ACT   : 1;
  __REG32               : 2;
  __REG32   END_TR_ST   : 1;
  __REG32   END_BF_ST   : 1;
  __REG32   DESC_LDST   : 1;
  __REG32               : 9;
  __REG32   BUFF_LENGTH :16;
} __udphs_dmastatusx_bits;


/* ISI Control 1 Register */
typedef struct {
  __REG32   ISI_RST     : 1;
  __REG32   ISI_DIS     : 1;
  __REG32   HSYNC_POL   : 1;
  __REG32   VSYNC_POL   : 1;
  __REG32   PIXCLK_POL  : 1;
  __REG32               : 1;
  __REG32   EMB_SYNC    : 1;
  __REG32   CRC_SYNC    : 1;
  __REG32   FRATE       : 3;
  __REG32               : 1;
  __REG32   FULL        : 1;
  __REG32   THMASK      : 2;
  __REG32   CODEC_ON    : 1;
  __REG32   SLD         : 8;
  __REG32   SFD         : 8;
} __isi_cr1_bits;

/* ISI Control 2 Register */
typedef struct {
  __REG32   IM_VSIZE    :11;
  __REG32   GS_MODE     : 1;
  __REG32   RGB_MODE    : 1;
  __REG32   GRAYSCALE   : 1;
  __REG32   RGB_SWAP    : 1;
  __REG32   COL_SPACE   : 1;
  __REG32   IM_HSIZE    :11;
  __REG32               : 1;
  __REG32   YCC_SWAP    : 2;
  __REG32   RGB_CFG     : 2;
} __isi_cr2_bits;

/* ISI Status Register */
typedef struct {
  __REG32   SOF         : 1;
  __REG32   DIS         : 1;
  __REG32   SOFTRST     : 1;
  __REG32   CDC_PND     : 1;
  __REG32   CRC_ERR     : 1;
  __REG32   FO_C_OVF    : 1;
  __REG32   FO_P_OVF    : 1;
  __REG32   FO_P_EMP    : 1;
  __REG32   FO_C_EMP    : 1;
  __REG32   FR_OVR      : 1;
  __REG32               :22;
} __isi_sr_bits;

/* Interrupt Enable Register */
/* Interrupt Disable Register */
/* Interrupt Mask Register */
typedef struct {
  __REG32   SOF         : 1;
  __REG32   DIS         : 1;
  __REG32   SOFTRST     : 1;
  __REG32               : 1;
  __REG32   CRC_ERR     : 1;
  __REG32   FO_C_OVF    : 1;
  __REG32   FO_P_OVF    : 1;
  __REG32   FO_P_EMP    : 1;
  __REG32   FO_C_EMP    : 1;
  __REG32   FR_OVR      : 1;
  __REG32               :22;
} __isi_ier_bits;

/* ISI Preview Register */
typedef struct {
  __REG32   PREV_VSIZE  :10;
  __REG32               : 6;
  __REG32   PREV_HSIZE  :10;
  __REG32               : 6;
} __isi_psize_bits;

/* ISI Color Space Conversion YCrCb to RGB Set 0 Register */
typedef struct {
  __REG32   C0          : 8;
  __REG32   C1          : 8;
  __REG32   C2          : 8;
  __REG32   C3          : 8;
} __isi_y2r_set0_bits;

/* ISI Color Space Conversion YCrCb to RGB Set 1 Register */
typedef struct {
  __REG32   C4          : 9;
  __REG32               : 3;
  __REG32   Yoff        : 1;
  __REG32   Croff       : 1;
  __REG32   Cboff       : 1;
  __REG32               :17;
} __isi_y2r_set1_bits;

/* ISI Color Space Conversion RGB to YCrCb Set 0 Register */
typedef struct {
  __REG32   C0          : 8;
  __REG32   C1          : 8;
  __REG32   C2          : 8;
  __REG32   Roff        : 1;
  __REG32               : 7;
} __isi_r2y_set0_bits;

/* ISI Color Space Conversion RGB to YCrCb Set 1 Register */
typedef struct {
  __REG32   C3          : 8;
  __REG32   C4          : 8;
  __REG32   C5          : 8;
  __REG32   Goff        : 1;
  __REG32               : 7;
} __isi_r2y_set1_bits;

/* ISI Color Space Conversion RGB to YCrCb Set 2 Register */
typedef struct {
  __REG32   C6          : 8;
  __REG32   C7          : 8;
  __REG32   C8          : 8;
  __REG32   Boff        : 1;
  __REG32               : 7;
} __isi_r2y_set2_bits;


/* ADC Control Register */
typedef struct {
  __REG32   SWRST       : 1;
  __REG32   START       : 1;
  __REG32               :30;
} __adc_cr_bits;

/* ADC Mode Register */
typedef struct {
  __REG32   TRGEN       : 1;
  __REG32   TRGSEL      : 3;
  __REG32   LOWRES      : 1;
  __REG32   SLEEP       : 1;
  __REG32               : 2;
  __REG32   PRESCAL     : 6;
  __REG32               : 2;
  __REG32   STARTUP     : 5;
  __REG32               : 3;
  __REG32   SHTIM       : 4;
  __REG32               : 4;
} __adc_mr_bits;

/* ADC Channel Enable Register */
/* ADC Channel Disable Register */
/* ADC Channel Status Register */
typedef struct {
  __REG32   CH0         : 1;
  __REG32   CH1         : 1;
  __REG32   CH2         : 1;
  __REG32   CH3         : 1;
  __REG32   CH4         : 1;
  __REG32   CH5         : 1;
  __REG32   CH6         : 1;
  __REG32   CH7         : 1;
  __REG32               :24;
} __adc_cher_bits;

/* ADC Status Register */
/* ADC Interrupt Enable Register */
/* ADC Interrupt Disable Register */
/* ADC Interrupt Mask Register */
typedef struct {
  __REG32   EOC0        : 1;
  __REG32   EOC1        : 1;
  __REG32   EOC2        : 1;
  __REG32   EOC3        : 1;
  __REG32   EOC4        : 1;
  __REG32   EOC5        : 1;
  __REG32   EOC6        : 1;
  __REG32   EOC7        : 1;
  __REG32   OVRE0       : 1;
  __REG32   OVRE1       : 1;
  __REG32   OVRE2       : 1;
  __REG32   OVRE3       : 1;
  __REG32   OVRE4       : 1;
  __REG32   OVRE5       : 1;
  __REG32   OVRE6       : 1;
  __REG32   OVRE7       : 1;
  __REG32   DRDY        : 1;
  __REG32   GOVRE       : 1;
  __REG32   ENDRX       : 1;
  __REG32   RXBUFF      : 1;
  __REG32               :12;
} __adc_sr_bits;

/* ADC Last Converted Data Register */
typedef struct {
  __REG32   LDATA       :10;
  __REG32               :22;
} __adc_lcdr_bits;

/* ADC Channel Data Register */
typedef struct {
  __REG32   DATA        :10;
  __REG32               :22;
} __adc_cdr_bits;



#endif	/* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/

/***************************************************************************
 **
 ** RSTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RSTC_CR,         0xFFFFFD00, __WRITE,      __rstc_cr_bits   );
__IO_REG32_BIT(RSTC_SR,         0xFFFFFD04, __READ,       __rstc_sr_bits   );
__IO_REG32_BIT(RSTC_MR,         0xFFFFFD08, __READ_WRITE, __rstc_mr_bits   );

/***************************************************************************
 **
 ** RTT
 **
 ***************************************************************************/
__IO_REG32_BIT(RTT_MR,         0xFFFFFD20, __READ_WRITE,  __rtt_mr_bits   );
__IO_REG32(    RTT_AR,         0xFFFFFD24, __READ_WRITE                   );
__IO_REG32(    RTT_VR,         0xFFFFFD28, __READ                         );
__IO_REG32_BIT(RTT_SR,         0xFFFFFD2C, __READ,        __rtt_sr_bits   );
 
/***************************************************************************
 **
 ** PIT
 **
 ***************************************************************************/
__IO_REG32_BIT(PIT_MR,         0xFFFFFD30, __READ_WRITE,  __pit_mr_bits   );
__IO_REG32_BIT(PIT_SR,         0xFFFFFD34, __READ,        __pit_sr_bits   );
__IO_REG32_BIT(PIT_PIVR,       0xFFFFFD38, __READ,        __pit_pivr_bits );
__IO_REG32_BIT(PIT_PIIR,       0xFFFFFD3C, __READ,        __pit_pivr_bits );

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_CR,         0xFFFFFD40, __WRITE,      __wdt_cr_bits   );
__IO_REG32_BIT(WDT_MR,         0xFFFFFD44, __READ_WRITE, __wdt_mr_bits   );
__IO_REG32_BIT(WDT_SR,         0xFFFFFD48, __READ,       __wdt_sr_bits   );

/***************************************************************************
 **
 ** SHDWC
 **
 ***************************************************************************/
__IO_REG32_BIT(SHDW_CR,         0xFFFFFD10, __WRITE,      __shdw_cr_bits   );
__IO_REG32_BIT(SHDW_MR,         0xFFFFFD14, __READ_WRITE, __shdw_mr_bits   );
__IO_REG32_BIT(SHDW_SR,         0xFFFFFD18, __READ,       __shdw_sr_bits   );

/***************************************************************************
 **
 ** MATRIX
 **
 ***************************************************************************/
__IO_REG32_BIT(MATRIX_MCFG0,    0xFFFFEA00, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG1,    0xFFFFEA04, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG2,    0xFFFFEA08, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG3,    0xFFFFEA0C, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG4,    0xFFFFEA10, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG5,    0xFFFFEA14, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG6,    0xFFFFEA18, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG7,    0xFFFFEA1C, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG8,    0xFFFFEA20, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG9,    0xFFFFEA24, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG10,   0xFFFFEA28, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG11,   0xFFFFEA2C, __READ_WRITE, __matrix_mcfgx_bits);

__IO_REG32_BIT(MATRIX_SCFG0,    0xFFFFEA40, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG1,    0xFFFFEA44, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG2,    0xFFFFEA48, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG3,    0xFFFFEA4C, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG4,    0xFFFFEA50, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG5,    0xFFFFEA54, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG6,    0xFFFFEA58, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG7,    0xFFFFEA5C, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG8,    0xFFFFEA60, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG9,    0xFFFFEA64, __READ_WRITE, __matrix_scfgx_bits);

__IO_REG32_BIT(MATRIX_PRAS0,    0xFFFFEA80, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS0,    0xFFFFEA84, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS1,    0xFFFFEA88, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS1,    0xFFFFEA8C, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS2,    0xFFFFEA90, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS2,    0xFFFFEA94, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS3,    0xFFFFEA98, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS3,    0xFFFFEA9C, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS4,    0xFFFFEAA0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS4,    0xFFFFEAA4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS5,    0xFFFFEAA8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS5,    0xFFFFEAAC, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS6,    0xFFFFEAB0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS6,    0xFFFFEAB4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS7,    0xFFFFEAB8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS7,    0xFFFFEABC, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS8,    0xFFFFEAC0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS8,    0xFFFFEAC4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS9,    0xFFFFEAC8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS9,    0xFFFFEACC, __READ_WRITE, __matrix_prbsx_bits);

__IO_REG32_BIT(MATRIX_MRCR,     0xFFFFEB00, __READ_WRITE, __matrix_mrcr_bits );

/***************************************************************************
 **
 ** CCFG
 **
 ***************************************************************************/
__IO_REG32(    MPBS0_SFR,     0xFFFFB14,  __READ_WRITE                  );
__IO_REG32(    MPBS1_SFR,     0xFFFFB1C,  __READ_WRITE                  );
__IO_REG32_BIT(EBI_CSA,       0xFFFFB20,  __READ_WRITE, __ebi_csa_bits  );
__IO_REG32(    MPBS2_SFR,     0xFFFFB2C,  __READ_WRITE                  );
__IO_REG32(    MPBS3_SFR,     0xFFFFB30,  __READ_WRITE                  );
__IO_REG32(    APB_SFR,       0xFFFFB34,  __READ_WRITE                  );

/***************************************************************************
 **
 ** SMC
 **
 ***************************************************************************/
__IO_REG32_BIT(SMC_SETUP_CS0,    0xFFFFE800, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS0,    0xFFFFE804, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS0,    0xFFFFE808, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS0,     0xFFFFE80C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS1,    0xFFFFE810, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS1,    0xFFFFE814, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS1,    0xFFFFE818, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS1,     0xFFFFE81C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS2,    0xFFFFE820, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS2,    0xFFFFE824, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS2,    0xFFFFE828, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS2,     0xFFFFE82C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS3,    0xFFFFE830, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS3,    0xFFFFE834, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS3,    0xFFFFE838, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS3,     0xFFFFE83C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS4,    0xFFFFE840, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS4,    0xFFFFE844, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS4,    0xFFFFE848, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS4,     0xFFFFE84C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS5,    0xFFFFE850, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS5,    0xFFFFE854, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS5,    0xFFFFE858, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS5,     0xFFFFE85C, __READ_WRITE, __smc_mode_csx_bits );

/***************************************************************************
 **
 ** DDRSDRC
 **
 ***************************************************************************/
__IO_REG32_BIT(DDRSDRC_MR,      0xFFFFE600, __READ_WRITE, __ddrsdrc_mr_bits   );
__IO_REG32_BIT(DDRSDRC_RTR,     0xFFFFE604, __READ_WRITE, __ddrsdrc_rtr_bits  );
__IO_REG32_BIT(DDRSDRC_CR,      0xFFFFE608, __READ_WRITE, __ddrsdrc_cr_bits   );
__IO_REG32_BIT(DDRSDRC_T0PR,    0xFFFFE60C, __READ_WRITE, __ddrsdrc_t0pr_bits );
__IO_REG32_BIT(DDRSDRC_T1PR,    0xFFFFE610, __READ_WRITE, __ddrsdrc_t1pr_bits );
__IO_REG32_BIT(DDRSDRC_LPR,     0xFFFFE618, __READ_WRITE, __ddrsdrc_lpr_bits  );
__IO_REG32_BIT(DDRSDRC_MD,      0xFFFFE61C, __READ_WRITE, __ddrsdrc_md_bits   );
__IO_REG32_BIT(DDRSDRC_DLL,     0xFFFFE620, __READ,       __ddrsdrc_dll_bits  );

/***************************************************************************
 **
 ** BCRAMC
 **
 ***************************************************************************/
__IO_REG32_BIT(BCRAMC_CR,       0xFFFFE400, __READ_WRITE, __bcramc_cr_bits  );
__IO_REG32_BIT(BCRAMC_TR,       0xFFFFE404, __READ_WRITE, __bcramc_tr_bits  );
__IO_REG32_BIT(BCRAMC_LPR,      0xFFFFE40C, __READ_WRITE, __bcramc_lpr_bits );
__IO_REG32_BIT(BCRAMC_MDR,      0xFFFFE410, __READ_WRITE, __bcramc_mdr_bits );
__IO_REG16(    BCRAMC_ADDRSIZE, 0xFFFFE4EC, __READ                          );
__IO_REG32(    BCRAMC_IPNAME1,  0xFFFFE4F0, __READ                          );
__IO_REG32(    BCRAMC_IPNAME2,  0xFFFFE4F4, __READ                          );
__IO_REG32(    BCRAMC_FEATURES, 0xFFFFE4F8, __READ                          );

/***************************************************************************
 **
 ** ECC
 **
 ***************************************************************************/
__IO_REG32_BIT(ECC_CR,          0xFFFFE200, __WRITE,      __ecc_cr_bits     );
__IO_REG32_BIT(ECC_MR,          0xFFFFE204, __READ_WRITE, __ecc_mr_bits     );
__IO_REG32_BIT(ECC_SR,          0xFFFFE208, __READ,       __ecc_sr_bits     );
__IO_REG32_BIT(ECC_PR,          0xFFFFE20C, __READ,       __ecc_pr_bits     );
__IO_REG16(    ECC_NPR,         0xFFFFE210, __READ                          );

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMAC_GCFG,       0xFFFFEC00, __READ_WRITE, __dmac_gcfg_bits   );
__IO_REG32_BIT(DMAC_EN,         0xFFFFEC04, __READ_WRITE, __dmac_en_bits     );
__IO_REG32_BIT(DMAC_SREQ,       0xFFFFEC08, __READ_WRITE, __dmac_sreq_bits   );
__IO_REG32_BIT(DMAC_CREQ,       0xFFFFEC0C, __READ_WRITE, __dmac_creq_bits   );
__IO_REG32_BIT(DMAC_LAST,       0xFFFFEC10, __READ_WRITE, __dmac_last_bits   );
__IO_REG32(    DMAC_SYNC,       0xFFFFEC14, __READ_WRITE                     );
__IO_REG32_BIT(DMAC_EBCIER,     0xFFFFEC18, __WRITE,      __dmac_ebcier_bits );
__IO_REG32_BIT(DMAC_EBCIDR,     0xFFFFEC1C, __WRITE,      __dmac_ebcier_bits );
__IO_REG32_BIT(DMAC_EBCIMR,     0xFFFFEC20, __READ,       __dmac_ebcier_bits );
__IO_REG32_BIT(DMAC_EBCISR,     0xFFFFEC24, __READ,       __dmac_ebcier_bits );
__IO_REG32_BIT(DMAC_CHER,       0xFFFFEC28, __WRITE,      __dmac_cher_bits   );
__IO_REG32_BIT(DMAC_CHDR,       0xFFFFEC2C, __WRITE,      __dmac_chdr_bits   );
__IO_REG32_BIT(DMAC_CHSR,       0xFFFFEC30, __READ,       __dmac_chsr_bits   );

__IO_REG32(    DMAC_SADDR0,     0xFFFFEC3C,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR0,     0xFFFFEC40,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR0,      0xFFFFEC44,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA0,     0xFFFFEC48,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB0,     0xFFFFEC4C,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG0,       0xFFFFEC50,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP0,      0xFFFFEC54,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP0,      0xFFFFEC58,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR1,     0xFFFFEC64,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR1,     0xFFFFEC68,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR1,      0xFFFFEC6C,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA1,     0xFFFFEC70,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB1,     0xFFFFEC74,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG1,       0xFFFFEC78,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP1,      0xFFFFEC7C,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP1,      0xFFFFEC80,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR2,     0xFFFFEC8C,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR2,     0xFFFFEC90,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR2,      0xFFFFEC94,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA2,     0xFFFFEC98,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB2,     0xFFFFEC9C,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG2,       0xFFFFECA0,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP2,      0xFFFFECA4,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP2,      0xFFFFECA8,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR3,     0xFFFFECB4,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR3,     0xFFFFECB8,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR3,      0xFFFFECBC,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA3,     0xFFFFECC0,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB3,     0xFFFFECC4,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG3,       0xFFFFECC8,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP3,      0xFFFFECCC,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP3,      0xFFFFECD0,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR4,     0xFFFFECDC,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR4,     0xFFFFECE0,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR4,      0xFFFFECE4,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA4,     0xFFFFECE8,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB4,     0xFFFFECEC,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG4,       0xFFFFECF0,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP4,      0xFFFFECF4,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP4,      0xFFFFECF8,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR5,     0xFFFFED04,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR5,     0xFFFFED08,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR5,      0xFFFFED0C,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA5,     0xFFFFED10,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB5,     0xFFFFED14,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG5,       0xFFFFED18,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP5,      0xFFFFED1C,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP5,      0xFFFFED20,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR6,     0xFFFFED2C,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR6,     0xFFFFED30,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR6,      0xFFFFED34,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA6,     0xFFFFED38,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB6,     0xFFFFED3C,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG6,       0xFFFFED40,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP6,      0xFFFFED44,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP6,      0xFFFFED48,  __READ_WRITE, __dmac_dpipx_bits );

__IO_REG32(    DMAC_SADDR7,     0xFFFFED54,  __READ_WRITE                    );
__IO_REG32(    DMAC_DADDR7,     0xFFFFED58,  __READ_WRITE                    );
__IO_REG32(    DMAC_DSCR7,      0xFFFFED5C,  __READ_WRITE                    );
__IO_REG32_BIT(DMAC_CTRLA7,     0xFFFFED60,  __READ_WRITE, __dmac_ctrlax_bits);
__IO_REG32_BIT(DMAC_CTRLB7,     0xFFFFED64,  __READ_WRITE, __dmac_ctrlbx_bits);
__IO_REG32_BIT(DMAC_CFG7,       0xFFFFED68,  __READ_WRITE, __dmac_cfgx_bits  );
__IO_REG32_BIT(DMAC_SPIP7,      0xFFFFED6C,  __READ_WRITE, __dmac_spipx_bits );
__IO_REG32_BIT(DMAC_DPIP7,      0xFFFFED70,  __READ_WRITE, __dmac_dpipx_bits );

/***************************************************************************
 **
 ** PMC
 **
 ***************************************************************************/
__IO_REG32_BIT(PMC_SCER,       0xFFFFFC00, __WRITE,      __pmc_scer_bits   );
__IO_REG32_BIT(PMC_SCDR,       0xFFFFFC04, __WRITE,      __pmc_scer_bits   );
__IO_REG32_BIT(PMC_SCSR,       0xFFFFFC08, __READ,       __pmc_scer_bits   );
__IO_REG32_BIT(PMC_PCER,       0xFFFFFC10, __WRITE,      __pmc_pcer_bits   );
__IO_REG32_BIT(PMC_PCDR,       0xFFFFFC14, __WRITE,      __pmc_pcer_bits   );
__IO_REG32_BIT(PMC_PCSR,       0xFFFFFC18, __READ,       __pmc_pcer_bits   );
__IO_REG32_BIT(CKGR_UCKR,      0xFFFFFC1C, __READ_WRITE, __ckgr_uckr_bits  );
__IO_REG32_BIT(CKGR_MOR,       0xFFFFFC20, __READ_WRITE, __ckgr_mor_bits   );
__IO_REG32_BIT(CKGR_MCFR,      0xFFFFFC24, __READ,       __ckgr_mcfr_bits  );
__IO_REG32_BIT(CKGR_PLLAR,     0xFFFFFC28, __READ_WRITE, __ckgr_pllar_bits );
__IO_REG32_BIT(CKGR_PLLBR,     0xFFFFFC2C, __READ_WRITE, __ckgr_pllbr_bits );
__IO_REG32_BIT(PMC_MCKR,       0xFFFFFC30, __READ_WRITE, __pmc_mckr_bits   );
__IO_REG32_BIT(PMC_PCK0,       0xFFFFFC40, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_PCK1,       0xFFFFFC44, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_IER,        0xFFFFFC60, __WRITE,      __pmc_ier_bits    );
__IO_REG32_BIT(PMC_IDR,        0xFFFFFC64, __WRITE,      __pmc_ier_bits    );
__IO_REG32_BIT(PMC_SR,         0xFFFFFC68, __READ,       __pmc_ier_bits    );
__IO_REG32_BIT(PMC_IMR,        0xFFFFFC6C, __READ,       __pmc_ier_bits    );

/***************************************************************************
 **
 ** AIC
 **
 ***************************************************************************/
__IO_REG32_BIT(AIC_SMR0,      0xFFFFF000, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR1,      0xFFFFF004, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR2,      0xFFFFF008, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR3,      0xFFFFF00C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR4,      0xFFFFF010, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR5,      0xFFFFF014, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR6,      0xFFFFF018, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR7,      0xFFFFF01C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR8,      0xFFFFF020, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR9,      0xFFFFF024, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR10,     0xFFFFF028, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR11,     0xFFFFF02C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR12,     0xFFFFF030, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR13,     0xFFFFF034, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR14,     0xFFFFF038, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR15,     0xFFFFF03C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR16,     0xFFFFF040, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR17,     0xFFFFF044, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR18,     0xFFFFF048, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR19,     0xFFFFF04C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR20,     0xFFFFF050, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR21,     0xFFFFF054, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR22,     0xFFFFF058, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR23,     0xFFFFF05C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR24,     0xFFFFF060, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR25,     0xFFFFF064, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR26,     0xFFFFF068, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR27,     0xFFFFF06C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR28,     0xFFFFF070, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR29,     0xFFFFF074, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR30,     0xFFFFF078, __READ_WRITE, __aicsmrx_bits);
__IO_REG32_BIT(AIC_SMR31,     0xFFFFF07C, __READ_WRITE, __aicsmrx_bits);
__IO_REG32(    AIC_SVR0,      0xFFFFF080, __READ_WRITE                );
__IO_REG32(    AIC_SVR1,      0xFFFFF084, __READ_WRITE                );
__IO_REG32(    AIC_SVR2,      0xFFFFF088, __READ_WRITE                );
__IO_REG32(    AIC_SVR3,      0xFFFFF08C, __READ_WRITE                );
__IO_REG32(    AIC_SVR4,      0xFFFFF090, __READ_WRITE                );
__IO_REG32(    AIC_SVR5,      0xFFFFF094, __READ_WRITE                );
__IO_REG32(    AIC_SVR6,      0xFFFFF098, __READ_WRITE                );
__IO_REG32(    AIC_SVR7,      0xFFFFF09C, __READ_WRITE                );
__IO_REG32(    AIC_SVR8,      0xFFFFF0A0, __READ_WRITE                );
__IO_REG32(    AIC_SVR9,      0xFFFFF0A4, __READ_WRITE                );
__IO_REG32(    AIC_SVR10,     0xFFFFF0A8, __READ_WRITE                );
__IO_REG32(    AIC_SVR11,     0xFFFFF0AC, __READ_WRITE                );
__IO_REG32(    AIC_SVR12,     0xFFFFF0B0, __READ_WRITE                );
__IO_REG32(    AIC_SVR13,     0xFFFFF0B4, __READ_WRITE                );
__IO_REG32(    AIC_SVR14,     0xFFFFF0B8, __READ_WRITE                );
__IO_REG32(    AIC_SVR15,     0xFFFFF0BC, __READ_WRITE                );
__IO_REG32(    AIC_SVR16,     0xFFFFF0C0, __READ_WRITE                );
__IO_REG32(    AIC_SVR17,     0xFFFFF0C4, __READ_WRITE                );
__IO_REG32(    AIC_SVR18,     0xFFFFF0C8, __READ_WRITE                );
__IO_REG32(    AIC_SVR19,     0xFFFFF0CC, __READ_WRITE                );
__IO_REG32(    AIC_SVR20,     0xFFFFF0D0, __READ_WRITE                );
__IO_REG32(    AIC_SVR21,     0xFFFFF0D4, __READ_WRITE                );
__IO_REG32(    AIC_SVR22,     0xFFFFF0D8, __READ_WRITE                );
__IO_REG32(    AIC_SVR23,     0xFFFFF0DC, __READ_WRITE                );
__IO_REG32(    AIC_SVR24,     0xFFFFF0E0, __READ_WRITE                );
__IO_REG32(    AIC_SVR25,     0xFFFFF0E4, __READ_WRITE                );
__IO_REG32(    AIC_SVR26,     0xFFFFF0E8, __READ_WRITE                );
__IO_REG32(    AIC_SVR27,     0xFFFFF0EC, __READ_WRITE                );
__IO_REG32(    AIC_SVR28,     0xFFFFF0F0, __READ_WRITE                );
__IO_REG32(    AIC_SVR29,     0xFFFFF0F4, __READ_WRITE                );
__IO_REG32(    AIC_SVR30,     0xFFFFF0F8, __READ_WRITE                );
__IO_REG32(    AIC_SVR31,     0xFFFFF0FC, __READ_WRITE                );
__IO_REG32(    AIC_IVR,       0xFFFFF100, __READ                      );
__IO_REG32(    AIC_FVR,       0xFFFFF104, __READ                      );
__IO_REG32_BIT(AIC_ISR,       0xFFFFF108, __READ      , __aicisr_bits );
__IO_REG32_BIT(AIC_IPR,       0xFFFFF10C, __READ      , __aicipr_bits );
__IO_REG32_BIT(AIC_IMR,       0xFFFFF110, __READ      , __aicipr_bits );
__IO_REG32_BIT(AIC_CISR,      0xFFFFF114, __READ      , __aiccisr_bits);
__IO_REG32_BIT(AIC_IECR,      0xFFFFF120, __WRITE     , __aicipr_bits );
__IO_REG32_BIT(AIC_IDCR,      0xFFFFF124, __WRITE     , __aicipr_bits );
__IO_REG32_BIT(AIC_ICCR,      0xFFFFF128, __WRITE     , __aicipr_bits );
__IO_REG32_BIT(AIC_ISCR,      0xFFFFF12C, __WRITE     , __aicipr_bits );
__IO_REG32(    AIC_EOICR,     0xFFFFF130, __WRITE                     );
__IO_REG32(    AIC_SPU,       0xFFFFF134, __READ_WRITE                );
__IO_REG32_BIT(AIC_DCR,       0xFFFFF138, __READ_WRITE, __aicdcr_bits );
__IO_REG32_BIT(AIC_FFER,      0xFFFFF140, __WRITE     , __aicffer_bits);
__IO_REG32_BIT(AIC_FFDR,      0xFFFFF144, __WRITE     , __aicffer_bits);
__IO_REG32_BIT(AIC_FFSR,      0xFFFFF148, __READ      , __aicffer_bits);

/***************************************************************************
 **
 ** DBGU
 **
 ***************************************************************************/
__IO_REG32_BIT(DBGU_CR,      0xFFFFEE00, __WRITE,      __dbgu_cr_bits  );
__IO_REG32_BIT(DBGU_MR,      0xFFFFEE04, __READ_WRITE, __dbgu_mr_bits  );
__IO_REG32_BIT(DBGU_IER,     0xFFFFEE08, __WRITE,      __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_IDR,     0xFFFFEE0C, __WRITE,      __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_IMR,     0xFFFFEE10, __READ,       __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_SR,      0xFFFFEE14, __READ,       __dbgu_ier_bits );
__IO_REG8(     DBGU_RHR,     0xFFFFEE18, __READ                        );
__IO_REG8(     DBGU_THR,     0xFFFFEE1C, __WRITE                       );
__IO_REG16(    DBGU_BRGR,    0xFFFFEE20, __READ_WRITE                  );
__IO_REG32_BIT(DBGU_CIDR,    0xFFFFEE40, __READ,       __dbgu_cidr_bits);
__IO_REG32(    DBGU_EXID,    0xFFFFEE44, __READ                        );
__IO_REG32_BIT(DBGU_FNR,     0xFFFFEE48, __READ_WRITE, __dbgu_fnr_bits );

__IO_REG32(    DBGU_RPR,     0xFFFFEF00, __READ_WRITE                  );
__IO_REG16(    DBGU_RCR,     0xFFFFEF04, __READ_WRITE                  );
__IO_REG32(    DBGU_TPR,     0xFFFFEF08, __READ_WRITE                  );
__IO_REG16(    DBGU_TCR,     0xFFFFEF0C, __READ_WRITE                  );
__IO_REG32(    DBGU_RNPR,    0xFFFFEF10, __READ_WRITE                  );
__IO_REG16(    DBGU_RNCR,    0xFFFFEF14, __READ_WRITE                  );
__IO_REG32(    DBGU_TNPR,    0xFFFFEF18, __READ_WRITE                  );
__IO_REG16(    DBGU_TNCR,    0xFFFFEF1C, __READ_WRITE                  );
__IO_REG32_BIT(DBGU_PTCR,    0xFFFFEF20, __WRITE,      __pdc_ptcr_bits );
__IO_REG32_BIT(DBGU_PTSR,    0xFFFFEF24, __READ,       __pdc_ptsr_bits );

/***************************************************************************
 **
 ** PIOA
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOA_PER,       0xFFFFF200, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PDR,       0xFFFFF204, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PSR,       0xFFFFF208, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_OER,       0xFFFFF210, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_ODR,       0xFFFFF214, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_OSR,       0xFFFFF218, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_IFER,      0xFFFFF220, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IFDR,      0xFFFFF224, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IFSR,      0xFFFFF228, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_SODR,      0xFFFFF230, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_CODR,      0xFFFFF234, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_ODSR,      0xFFFFF238, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_PDSR,      0xFFFFF23C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_IER,       0xFFFFF240, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IDR,       0xFFFFF244, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IMR,       0xFFFFF248, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_ISR,       0xFFFFF24C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_MDER,      0xFFFFF250, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_MDDR,      0xFFFFF254, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_MDSR,      0xFFFFF258, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_PUDR,      0xFFFFF260, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PUER,      0xFFFFF264, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PUSR,      0xFFFFF268, __READ      , __pioper_bits);


/***************************************************************************
 **
 ** PIOB
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOB_PER,       0xFFFFF400, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PDR,       0xFFFFF404, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PSR,       0xFFFFF408, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_OER,       0xFFFFF410, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_ODR,       0xFFFFF414, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_OSR,       0xFFFFF418, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_IFER,      0xFFFFF420, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IFDR,      0xFFFFF424, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IFSR,      0xFFFFF428, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_SODR,      0xFFFFF430, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_CODR,      0xFFFFF434, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_ODSR,      0xFFFFF438, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_PDSR,      0xFFFFF43C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_IER,       0xFFFFF440, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IDR,       0xFFFFF444, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IMR,       0xFFFFF448, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_ISR,       0xFFFFF44C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_MDER,      0xFFFFF450, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_MDDR,      0xFFFFF454, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_MDSR,      0xFFFFF458, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_PUDR,      0xFFFFF460, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PUER,      0xFFFFF464, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PUSR,      0xFFFFF468, __READ      , __pioper_bits);

/***************************************************************************
 **
 ** PIOC
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOC_PER,       0xFFFFF600, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_PDR,       0xFFFFF604, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_PSR,       0xFFFFF608, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_OER,       0xFFFFF610, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_ODR,       0xFFFFF614, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_OSR,       0xFFFFF618, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_IFER,      0xFFFFF620, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_IFDR,      0xFFFFF624, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_IFSR,      0xFFFFF628, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_SODR,      0xFFFFF630, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_CODR,      0xFFFFF634, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_ODSR,      0xFFFFF638, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_PDSR,      0xFFFFF63C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_IER,       0xFFFFF640, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_IDR,       0xFFFFF644, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_IMR,       0xFFFFF648, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_ISR,       0xFFFFF64C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_MDER,      0xFFFFF650, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_MDDR,      0xFFFFF654, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_MDSR,      0xFFFFF658, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOC_PUDR,      0xFFFFF660, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_PUER,      0xFFFFF664, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOC_PUSR,      0xFFFFF668, __READ      , __pioper_bits);

/***************************************************************************
 **
 ** PIOD
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOD_PER,       0xFFFFF800, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_PDR,       0xFFFFF804, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_PSR,       0xFFFFF808, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_OER,       0xFFFFF810, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_ODR,       0xFFFFF814, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_OSR,       0xFFFFF818, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_IFER,      0xFFFFF820, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_IFDR,      0xFFFFF824, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_IFSR,      0xFFFFF828, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_SODR,      0xFFFFF830, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_CODR,      0xFFFFF834, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_ODSR,      0xFFFFF838, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_PDSR,      0xFFFFF83C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_IER,       0xFFFFF840, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_IDR,       0xFFFFF844, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_IMR,       0xFFFFF848, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_ISR,       0xFFFFF84C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_MDER,      0xFFFFF850, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_MDDR,      0xFFFFF854, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_MDSR,      0xFFFFF858, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOD_PUDR,      0xFFFFF860, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_PUER,      0xFFFFF864, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOD_PUSR,      0xFFFFF868, __READ      , __pioper_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CR,         0xFFFA4000, __WRITE     , __spicr_bits  );
__IO_REG32_BIT(SPI0_MR,         0xFFFA4004, __READ_WRITE, __spimr_bits  );
__IO_REG32_BIT(SPI0_RDR,        0xFFFA4008, __READ      , __spirdr_bits );
__IO_REG32_BIT(SPI0_TDR,        0xFFFA400C, __WRITE     , __spitdr_bits );
__IO_REG32_BIT(SPI0_SR,         0xFFFA4010, __READ      , __spisr_bits  );
__IO_REG32_BIT(SPI0_IER,        0xFFFA4014, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI0_IDR,        0xFFFA4018, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI0_IMR,        0xFFFA401C, __READ      , __spiier_bits );
__IO_REG32_BIT(SPI0_CSR0,       0xFFFA4030, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR1,       0xFFFA4034, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR2,       0xFFFA4038, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR3,       0xFFFA403C, __READ_WRITE, __spicsrx_bits);

__IO_REG32(    SPI0_RPR,        0xFFFA4100, __READ_WRITE                  );
__IO_REG16(    SPI0_RCR,        0xFFFA4104, __READ_WRITE                  );
__IO_REG32(    SPI0_TPR,        0xFFFA4108, __READ_WRITE                  );
__IO_REG16(    SPI0_TCR,        0xFFFA410C, __READ_WRITE                  );
__IO_REG32(    SPI0_RNPR,       0xFFFA4110, __READ_WRITE                  );
__IO_REG16(    SPI0_RNCR,       0xFFFA4114, __READ_WRITE                  );
__IO_REG32(    SPI0_TNPR,       0xFFFA4118, __READ_WRITE                  );
__IO_REG16(    SPI0_TNCR,       0xFFFA411C, __READ_WRITE                  );
__IO_REG32_BIT(SPI0_PTCR,       0xFFFA4120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(SPI0_PTSR,       0xFFFA4124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CR,         0xFFFA8000, __WRITE     , __spicr_bits  );
__IO_REG32_BIT(SPI1_MR,         0xFFFA8004, __READ_WRITE, __spimr_bits  );
__IO_REG32_BIT(SPI1_RDR,        0xFFFA8008, __READ      , __spirdr_bits );
__IO_REG32_BIT(SPI1_TDR,        0xFFFA800C, __WRITE     , __spitdr_bits );
__IO_REG32_BIT(SPI1_SR,         0xFFFA8010, __READ      , __spisr_bits  );
__IO_REG32_BIT(SPI1_IER,        0xFFFA8014, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI1_IDR,        0xFFFA8018, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI1_IMR,        0xFFFA801C, __READ      , __spiier_bits );
__IO_REG32_BIT(SPI1_CSR0,       0xFFFA8030, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI1_CSR1,       0xFFFA8034, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI1_CSR2,       0xFFFA8038, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI1_CSR3,       0xFFFA803C, __READ_WRITE, __spicsrx_bits);

__IO_REG32(    SPI1_RPR,        0xFFFA8100, __READ_WRITE                  );
__IO_REG16(    SPI1_RCR,        0xFFFA8104, __READ_WRITE                  );
__IO_REG32(    SPI1_TPR,        0xFFFA8108, __READ_WRITE                  );
__IO_REG16(    SPI1_TCR,        0xFFFA810C, __READ_WRITE                  );
__IO_REG32(    SPI1_RNPR,       0xFFFA8110, __READ_WRITE                  );
__IO_REG16(    SPI1_RNCR,       0xFFFA8114, __READ_WRITE                  );
__IO_REG32(    SPI1_TNPR,       0xFFFA8118, __READ_WRITE                  );
__IO_REG16(    SPI1_TNCR,       0xFFFA811C, __READ_WRITE                  );
__IO_REG32_BIT(SPI1_PTCR,       0xFFFA8120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(SPI1_PTSR,       0xFFFA8124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** TWI
 **
 ***************************************************************************/
__IO_REG32_BIT(TWI_CR,          0xFFF88000, __WRITE     , __twi_cr_bits   );
__IO_REG32_BIT(TWI_MMR,         0xFFF88004, __READ_WRITE, __twi_mmr_bits  );
__IO_REG32_BIT(TWI_SMR,         0xFFF88008, __READ_WRITE, __twi_smr_bits  );
__IO_REG32_BIT(TWI_IADR,        0xFFF8800C, __READ_WRITE, __twi_iadr_bits );
__IO_REG32_BIT(TWI_CWGR,        0xFFF88010, __READ_WRITE, __twi_cwgr_bits );
__IO_REG32_BIT(TWI_SR,          0xFFF88020, __READ      , __twi_sr_bits   );
__IO_REG32_BIT(TWI_IER,         0xFFF88024, __WRITE     , __twi_ier_bits  );
__IO_REG32_BIT(TWI_IDR,         0xFFF88028, __WRITE     , __twi_ier_bits  );
__IO_REG32_BIT(TWI_IMR,         0xFFF8802C, __READ      , __twi_ier_bits  );
__IO_REG8(     TWI_RHR ,        0xFFF88030, __READ                        );
__IO_REG8(     TWI_THR ,        0xFFF88034, __WRITE                       );

__IO_REG32(    TWI_RPR,         0xFFF88100, __READ_WRITE                  );
__IO_REG16(    TWI_RCR,         0xFFF88104, __READ_WRITE                  );
__IO_REG32(    TWI_TPR,         0xFFF88108, __READ_WRITE                  );
__IO_REG16(    TWI_TCR,         0xFFF8810C, __READ_WRITE                  );
__IO_REG32(    TWI_RNPR,        0xFFF88110, __READ_WRITE                  );
__IO_REG16(    TWI_RNCR,        0xFFF88114, __READ_WRITE                  );
__IO_REG32(    TWI_TNPR,        0xFFF88118, __READ_WRITE                  );
__IO_REG16(    TWI_TNCR,        0xFFF8811C, __READ_WRITE                  );
__IO_REG32_BIT(TWI_PTCR,        0xFFF88120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(TWI_PTSR,        0xFFF88124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** USART0
 **
 ***************************************************************************/
__IO_REG32_BIT(US0_CR,          0xFFF8C000, __WRITE     , __uscr_bits  );
__IO_REG32_BIT(US0_MR,          0xFFF8C004, __READ_WRITE, __usmr_bits  );
__IO_REG32_BIT(US0_IER,         0xFFF8C008, __WRITE     , __usier_bits );
__IO_REG32_BIT(US0_IDR,         0xFFF8C00C, __WRITE     , __usier_bits );
__IO_REG32_BIT(US0_IMR,         0xFFF8C010, __READ      , __usier_bits );
__IO_REG32_BIT(US0_CSR,         0xFFF8C014, __READ      , __uscsr_bits );
__IO_REG32_BIT(US0_RHR,         0xFFF8C018, __READ      , __usrhr_bits );
__IO_REG32_BIT(US0_THR,         0xFFF8C01C, __WRITE     , __usthr_bits );
__IO_REG32_BIT(US0_BRGR,        0xFFF8C020, __READ_WRITE, __usbrgr_bits);
__IO_REG16(    US0_RTOR,        0xFFF8C024, __READ_WRITE               );
__IO_REG8(     US0_TTGR,        0xFFF8C028, __READ_WRITE               );
__IO_REG32_BIT(US0_FIDI,        0xFFF8C040, __READ_WRITE, __usfidi_bits);
__IO_REG8(     US0_NER,         0xFFF8C044, __READ                     );
__IO_REG8(     US0_IF,          0xFFF8C04C, __READ_WRITE               );
__IO_REG32_BIT(US0_MAN,         0xFFF8C050, __READ_WRITE, __usman_bits );

__IO_REG32(    US0_RPR,         0xFFF8C100, __READ_WRITE                  );
__IO_REG16(    US0_RCR,         0xFFF8C104, __READ_WRITE                  );
__IO_REG32(    US0_TPR,         0xFFF8C108, __READ_WRITE                  );
__IO_REG16(    US0_TCR,         0xFFF8C10C, __READ_WRITE                  );
__IO_REG32(    US0_RNPR,        0xFFF8C110, __READ_WRITE                  );
__IO_REG16(    US0_RNCR,        0xFFF8C114, __READ_WRITE                  );
__IO_REG32(    US0_TNPR,        0xFFF8C118, __READ_WRITE                  );
__IO_REG16(    US0_TNCR,        0xFFF8C11C, __READ_WRITE                  );
__IO_REG32_BIT(US0_PTCR,        0xFFF8C120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(US0_PTSR,        0xFFF8C124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** USART1
 **
 ***************************************************************************/
__IO_REG32_BIT(US1_CR,          0xFFF90000, __WRITE     , __uscr_bits  );
__IO_REG32_BIT(US1_MR,          0xFFF90004, __READ_WRITE, __usmr_bits  );
__IO_REG32_BIT(US1_IER,         0xFFF90008, __WRITE     , __usier_bits );
__IO_REG32_BIT(US1_IDR,         0xFFF9000C, __WRITE     , __usier_bits );
__IO_REG32_BIT(US1_IMR,         0xFFF90010, __READ      , __usier_bits );
__IO_REG32_BIT(US1_CSR,         0xFFF90014, __READ      , __uscsr_bits );
__IO_REG32_BIT(US1_RHR,         0xFFF90018, __READ      , __usrhr_bits );
__IO_REG32_BIT(US1_THR,         0xFFF9001C, __WRITE     , __usthr_bits );
__IO_REG32_BIT(US1_BRGR,        0xFFF90020, __READ_WRITE, __usbrgr_bits);
__IO_REG16(    US1_RTOR,        0xFFF90024, __READ_WRITE               );
__IO_REG8(     US1_TTGR,        0xFFF90028, __READ_WRITE               );
__IO_REG32_BIT(US1_FIDI,        0xFFF90040, __READ_WRITE, __usfidi_bits);
__IO_REG8(     US1_NER,         0xFFF90044, __READ                     );
__IO_REG8(     US1_IF,          0xFFF9004C, __READ_WRITE               );
__IO_REG32_BIT(US1_MAN,         0xFFF90050, __READ_WRITE, __usman_bits );

__IO_REG32(    US1_RPR,         0xFFF90100, __READ_WRITE                  );
__IO_REG16(    US1_RCR,         0xFFF90104, __READ_WRITE                  );
__IO_REG32(    US1_TPR,         0xFFF90108, __READ_WRITE                  );
__IO_REG16(    US1_TCR,         0xFFF9010C, __READ_WRITE                  );
__IO_REG32(    US1_RNPR,        0xFFF90110, __READ_WRITE                  );
__IO_REG16(    US1_RNCR,        0xFFF90114, __READ_WRITE                  );
__IO_REG32(    US1_TNPR,        0xFFF90118, __READ_WRITE                  );
__IO_REG16(    US1_TNCR,        0xFFF9011C, __READ_WRITE                  );
__IO_REG32_BIT(US1_PTCR,        0xFFF90120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(US1_PTSR,        0xFFF90124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** USART2
 **
 ***************************************************************************/
__IO_REG32_BIT(US2_CR,          0xFFF94000, __WRITE     , __uscr_bits  );
__IO_REG32_BIT(US2_MR,          0xFFF94004, __READ_WRITE, __usmr_bits  );
__IO_REG32_BIT(US2_IER,         0xFFF94008, __WRITE     , __usier_bits );
__IO_REG32_BIT(US2_IDR,         0xFFF9400C, __WRITE     , __usier_bits );
__IO_REG32_BIT(US2_IMR,         0xFFF94010, __READ      , __usier_bits );
__IO_REG32_BIT(US2_CSR,         0xFFF94014, __READ      , __uscsr_bits );
__IO_REG32_BIT(US2_RHR,         0xFFF94018, __READ      , __usrhr_bits );
__IO_REG32_BIT(US2_THR,         0xFFF9401C, __WRITE     , __usthr_bits );
__IO_REG32_BIT(US2_BRGR,        0xFFF94020, __READ_WRITE, __usbrgr_bits);
__IO_REG16(    US2_RTOR,        0xFFF94024, __READ_WRITE               );
__IO_REG8(     US2_TTGR,        0xFFF94028, __READ_WRITE               );
__IO_REG32_BIT(US2_FIDI,        0xFFF94040, __READ_WRITE, __usfidi_bits);
__IO_REG8(     US2_NER,         0xFFF94044, __READ                     );
__IO_REG8(     US2_IF,          0xFFF9404C, __READ_WRITE               );
__IO_REG32_BIT(US2_MAN,         0xFFF94050, __READ_WRITE, __usman_bits );

__IO_REG32(    US2_RPR,         0xFFF94100, __READ_WRITE                  );
__IO_REG16(    US2_RCR,         0xFFF94104, __READ_WRITE                  );
__IO_REG32(    US2_TPR,         0xFFF94108, __READ_WRITE                  );
__IO_REG16(    US2_TCR,         0xFFF9410C, __READ_WRITE                  );
__IO_REG32(    US2_RNPR,        0xFFF94110, __READ_WRITE                  );
__IO_REG16(    US2_RNCR,        0xFFF94114, __READ_WRITE                  );
__IO_REG32(    US2_TNPR,        0xFFF94118, __READ_WRITE                  );
__IO_REG16(    US2_TNCR,        0xFFF9411C, __READ_WRITE                  );
__IO_REG32_BIT(US2_PTCR,        0xFFF94120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(US2_PTSR,        0xFFF94124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** SSC0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSC0_CR,         0xFFF98000, __WRITE,      __ssc_cr_bits  );
__IO_REG32_BIT(SSC0_CMR,        0xFFF98004, __READ_WRITE, __ssc_cmr_bits );
__IO_REG32_BIT(SSC0_RCMR,       0xFFF98010, __READ_WRITE, __ssc_rcmr_bits);
__IO_REG32_BIT(SSC0_RFMR,       0xFFF98014, __READ_WRITE, __ssc_rfmr_bits);
__IO_REG32_BIT(SSC0_TCMR,       0xFFF98018, __READ_WRITE, __ssc_tcmr_bits);
__IO_REG32_BIT(SSC0_TFMR,       0xFFF9801C, __READ_WRITE, __ssc_tfmr_bits);
__IO_REG32(    SSC0_RHR,        0xFFF98020, __READ                       );
__IO_REG32(    SSC0_THR,        0xFFF98024, __WRITE                      );
__IO_REG16(    SSC0_RSHR,       0xFFF98030, __READ                       );
__IO_REG16(    SSC0_TSHR,       0xFFF98034, __READ_WRITE                 );
__IO_REG16(    SSC0_RC0R,       0xFFF98038, __READ_WRITE                 );
__IO_REG16(    SSC0_RC1R,       0xFFF9803C, __READ_WRITE                 );
__IO_REG32_BIT(SSC0_SR,         0xFFF98040, __READ,       __ssc_sr_bits  );
__IO_REG32_BIT(SSC0_IER,        0xFFF98044, __WRITE,      __ssc_ier_bits );
__IO_REG32_BIT(SSC0_IDR,        0xFFF98048, __WRITE,      __ssc_ier_bits );
__IO_REG32_BIT(SSC0_IMR,        0xFFF9804C, __READ,       __ssc_ier_bits );

__IO_REG32(    SSC0_RPR,        0xFFF98100, __READ_WRITE                  );
__IO_REG16(    SSC0_RCR,        0xFFF98104, __READ_WRITE                  );
__IO_REG32(    SSC0_TPR,        0xFFF98108, __READ_WRITE                  );
__IO_REG16(    SSC0_TCR,        0xFFF9810C, __READ_WRITE                  );
__IO_REG32(    SSC0_RNPR,       0xFFF98110, __READ_WRITE                  );
__IO_REG16(    SSC0_RNCR,       0xFFF98114, __READ_WRITE                  );
__IO_REG32(    SSC0_TNPR,       0xFFF98118, __READ_WRITE                  );
__IO_REG16(    SSC0_TNCR,       0xFFF9811C, __READ_WRITE                  );
__IO_REG32_BIT(SSC0_PTCR,       0xFFF98120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(SSC0_PTSR,       0xFFF98124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** SSC1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSC1_CR,         0xFFF9C000, __WRITE,      __ssc_cr_bits  );
__IO_REG32_BIT(SSC1_CMR,        0xFFF9C004, __READ_WRITE, __ssc_cmr_bits );
__IO_REG32_BIT(SSC1_RCMR,       0xFFF9C010, __READ_WRITE, __ssc_rcmr_bits);
__IO_REG32_BIT(SSC1_RFMR,       0xFFF9C014, __READ_WRITE, __ssc_rfmr_bits);
__IO_REG32_BIT(SSC1_TCMR,       0xFFF9C018, __READ_WRITE, __ssc_tcmr_bits);
__IO_REG32_BIT(SSC1_TFMR,       0xFFF9C01C, __READ_WRITE, __ssc_tfmr_bits);
__IO_REG32(    SSC1_RHR,        0xFFF9C020, __READ                       );
__IO_REG32(    SSC1_THR,        0xFFF9C024, __WRITE                      );
__IO_REG16(    SSC1_RSHR,       0xFFF9C030, __READ                       );
__IO_REG16(    SSC1_TSHR,       0xFFF9C034, __READ_WRITE                 );
__IO_REG16(    SSC1_RC0R,       0xFFF9C038, __READ_WRITE                 );
__IO_REG16(    SSC1_RC1R,       0xFFF9C03C, __READ_WRITE                 );
__IO_REG32_BIT(SSC1_SR,         0xFFF9C040, __READ,       __ssc_sr_bits  );
__IO_REG32_BIT(SSC1_IER,        0xFFF9C044, __WRITE,      __ssc_ier_bits );
__IO_REG32_BIT(SSC1_IDR,        0xFFF9C048, __WRITE,      __ssc_ier_bits );
__IO_REG32_BIT(SSC1_IMR,        0xFFF9C04C, __READ,       __ssc_ier_bits );

__IO_REG32(    SSC1_RPR,        0xFFF9C100, __READ_WRITE                  );
__IO_REG16(    SSC1_RCR,        0xFFF9C104, __READ_WRITE                  );
__IO_REG32(    SSC1_TPR,        0xFFF9C108, __READ_WRITE                  );
__IO_REG16(    SSC1_TCR,        0xFFF9C10C, __READ_WRITE                  );
__IO_REG32(    SSC1_RNPR,       0xFFF9C110, __READ_WRITE                  );
__IO_REG16(    SSC1_RNCR,       0xFFF9C114, __READ_WRITE                  );
__IO_REG32(    SSC1_TNPR,       0xFFF9C118, __READ_WRITE                  );
__IO_REG16(    SSC1_TNCR,       0xFFF9C11C, __READ_WRITE                  );
__IO_REG32_BIT(SSC1_PTCR,       0xFFF9C120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(SSC1_PTSR,       0xFFF9C124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** AC97C
 **
 ***************************************************************************/
__IO_REG32_BIT(AC97C_MR,        0xFFFA0008, __READ_WRITE, __ac97c_mr_bits   );
__IO_REG32_BIT(AC97C_ICA,       0xFFFA0010, __READ_WRITE, __ac97c_ca_bits  );
__IO_REG32_BIT(AC97C_OCA,       0xFFFA0014, __READ_WRITE, __ac97c_ca_bits  );
__IO_REG32_BIT(AC97C_CARHR,     0xFFFA0020, __READ,       __ac97c_cxrhr_bits);
__IO_REG32_BIT(AC97C_CATHR,     0xFFFA0024, __WRITE,      __ac97c_cxthr_bits);
__IO_REG32_BIT(AC97C_CASR,      0xFFFA0028, __READ,       __ac97c_casr_bits );
__IO_REG32_BIT(AC97C_CAMR,      0xFFFA002C, __READ_WRITE, __ac97c_camr_bits );
__IO_REG32_BIT(AC97C_CBRHR,     0xFFFA0030, __READ,       __ac97c_cxrhr_bits);
__IO_REG32_BIT(AC97C_CBTHR,     0xFFFA0034, __WRITE,      __ac97c_cxthr_bits);
__IO_REG32_BIT(AC97C_CBSR,      0xFFFA0038, __READ,       __ac97c_cbsr_bits );
__IO_REG32_BIT(AC97C_CBMR,      0xFFFA003C, __READ_WRITE, __ac97c_cbmr_bits );
__IO_REG32_BIT(AC97C_CORHR,     0xFFFA0040, __READ,       __ac97c_corhr_bits);
__IO_REG32_BIT(AC97C_COTHR,     0xFFFA0044, __WRITE,      __ac97c_cothr_bits);
__IO_REG32_BIT(AC97C_COSR,      0xFFFA0048, __READ,       __ac97c_cosr_bits );
__IO_REG32_BIT(AC97C_COMR,      0xFFFA004C, __READ_WRITE, __ac97c_comr_bits );
__IO_REG32_BIT(AC97C_SR,        0xFFFA0050, __READ,       __ac97c_sr_bits   );
__IO_REG32_BIT(AC97C_IER,       0xFFFA0054, __WRITE,      __ac97c_sr_bits   );
__IO_REG32_BIT(AC97C_IDR,       0xFFFA0058, __WRITE,      __ac97c_sr_bits   );
__IO_REG32_BIT(AC97C_IMR,       0xFFFA005C, __READ,       __ac97c_sr_bits   );
__IO_REG32_BIT(AC97C_CCRHR,     0xFFFA0060, __READ,       __ac97c_cxrhr_bits);
__IO_REG32_BIT(AC97C_CCTHR,     0xFFFA0064, __WRITE,      __ac97c_cxthr_bits);
__IO_REG32_BIT(AC97C_CCSR,      0xFFFA0068, __READ,       __ac97c_ccsr_bits );
__IO_REG32_BIT(AC97C_CCMR,      0xFFFA006C, __READ_WRITE, __ac97c_ccmr_bits );

__IO_REG32(    AC97C_CARPR,     0xFFFA0100, __READ_WRITE                  );
__IO_REG16(    AC97C_CARCR,     0xFFFA0104, __READ_WRITE                  );
__IO_REG32(    AC97C_CATPR,     0xFFFA0108, __READ_WRITE                  );
__IO_REG16(    AC97C_CATCR,     0xFFFA010C, __READ_WRITE                  );
__IO_REG32(    AC97C_CARNPR,    0xFFFA0110, __READ_WRITE                  );
__IO_REG16(    AC97C_CARNCR,    0xFFFA0114, __READ_WRITE                  );
__IO_REG32(    AC97C_CATNPR,    0xFFFA0118, __READ_WRITE                  );
__IO_REG16(    AC97C_CATNCR,    0xFFFA011C, __READ_WRITE                  );
__IO_REG32_BIT(AC97C_CAPTCR,    0xFFFA0120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(AC97C_CAPTSR,    0xFFFA0124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** TC
 **
 ***************************************************************************/ 
__IO_REG32_BIT(TC0_CCR,      0xFFF7C000, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC0_CMR,      0xFFF7C004, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC0_CV,       0xFFF7C010, __READ                      );
__IO_REG16(    TC0_RA,       0xFFF7C014, __READ_WRITE                );
__IO_REG16(    TC0_RB,       0xFFF7C018, __READ_WRITE                );
__IO_REG16(    TC0_RC,       0xFFF7C01C, __READ_WRITE                );
__IO_REG32_BIT(TC0_SR,       0xFFF7C020, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC0_IER,      0xFFF7C024, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC0_IDR,      0xFFF7C028, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC0_IMR,      0xFFF7C02C, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC1_CCR,      0xFFF7C040, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC1_CMR,      0xFFF7C044, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC1_CV,       0xFFF7C050, __READ                      );
__IO_REG16(    TC1_RA,       0xFFF7C054, __READ_WRITE                );
__IO_REG16(    TC1_RB,       0xFFF7C058, __READ_WRITE                );
__IO_REG16(    TC1_RC,       0xFFF7C05C, __READ_WRITE                );
__IO_REG32_BIT(TC1_SR,       0xFFF7C060, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC1_IER,      0xFFF7C064, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC1_IDR,      0xFFF7C068, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC1_IMR,      0xFFF7C06C, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC2_CCR,      0xFFF7C080, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC2_CMR,      0xFFF7C084, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC2_CV,       0xFFF7C090, __READ                      );
__IO_REG16(    TC2_RA,       0xFFF7C094, __READ_WRITE                );
__IO_REG16(    TC2_RB,       0xFFF7C098, __READ_WRITE                );
__IO_REG16(    TC2_RC,       0xFFF7C09C, __READ_WRITE                );
__IO_REG32_BIT(TC2_SR,       0xFFF7C0A0, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC2_IER,      0xFFF7C0A4, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC2_IDR,      0xFFF7C0A8, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC2_IMR,      0xFFF7C0AC, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC_BCR,       0xFFF7C0C0, __WRITE,      __tc_bcr_bits );
__IO_REG32_BIT(TC_BMR,       0xFFF7C0C4, __READ_WRITE, __tc_bmr_bits );

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN_MR,       0xFFFAC000, __READ_WRITE, __can_mr_bits   );
__IO_REG32_BIT(CAN_IER,      0xFFFAC004, __WRITE,      __can_ier_bits  );
__IO_REG32_BIT(CAN_IDR,      0xFFFAC008, __WRITE,      __can_ier_bits  );
__IO_REG32_BIT(CAN_IMR,      0xFFFAC00C, __READ,       __can_ier_bits  );
__IO_REG32_BIT(CAN_SR,       0xFFFAC010, __READ,       __can_sr_bits   );
__IO_REG32_BIT(CAN_BR,       0xFFFAC014, __READ_WRITE, __can_br_bits   );
__IO_REG16(    CAN_TIM,      0xFFFAC018, __READ                        );
__IO_REG16(    CAN_TIMESTP,  0xFFFAC01C, __READ                        );
__IO_REG32_BIT(CAN_ECR,      0xFFFAC020, __READ,       __can_ecr_bits  );
__IO_REG32_BIT(CAN_TCR,      0xFFFAC024, __WRITE,      __can_tcr_bits  );
__IO_REG32_BIT(CAN_ACR,      0xFFFAC028, __WRITE,      __can_acr_bits  );
__IO_REG32_BIT(CAN_MMR0,     0xFFFAC200, __READ_WRITE, __can_mmrx_bits );
__IO_REG32_BIT(CAN_MAM0,     0xFFFAC204, __READ_WRITE, __can_mamx_bits );
__IO_REG32_BIT(CAN_MID0,     0xFFFAC208, __READ_WRITE, __can_mamx_bits );
__IO_REG32_BIT(CAN_MFID0,    0xFFFAC20C, __READ,       __can_mfidx_bits);
__IO_REG32_BIT(CAN_MSR0,     0xFFFAC210, __READ,       __can_msrx_bits );
__IO_REG32(    CAN_MDL0,     0xFFFAC214, __READ_WRITE                  );
__IO_REG32(    CAN_MDH0,     0xFFFAC218, __READ_WRITE                  );
__IO_REG32_BIT(CAN_MCR0,     0xFFFAC21C, __WRITE,      __can_mcrx_bits );
__IO_REG32_BIT(CAN_MMR1,     0xFFFAC220, __READ_WRITE, __can_mmrx_bits );
__IO_REG32_BIT(CAN_MAM1,     0xFFFAC224, __READ_WRITE, __can_mamx_bits );
__IO_REG32_BIT(CAN_MID1,     0xFFFAC228, __READ_WRITE, __can_mamx_bits );
__IO_REG32_BIT(CAN_MFID1,    0xFFFAC22C, __READ,       __can_mfidx_bits);
__IO_REG32_BIT(CAN_MSR1,     0xFFFAC230, __READ,       __can_msrx_bits );
__IO_REG32(    CAN_MDL1,     0xFFFAC234, __READ_WRITE                  );
__IO_REG32(    CAN_MDH1,     0xFFFAC238, __READ_WRITE                  );
__IO_REG32_BIT(CAN_MCR1,     0xFFFAC23C, __WRITE,      __can_mcrx_bits );

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_MR,      0xFFFB8000, __READ_WRITE, __pwm_mr_bits  );
__IO_REG32_BIT(PWM_ENA,     0xFFFB8004, __WRITE,      __pwm_ena_bits );
__IO_REG32_BIT(PWM_DIS,     0xFFFB8008, __WRITE,      __pwm_ena_bits );
__IO_REG32_BIT(PWM_SR,      0xFFFB800C, __READ,       __pwm_ena_bits );
__IO_REG32_BIT(PWM_IER,     0xFFFB8010, __WRITE,      __pwm_ier_bits );
__IO_REG32_BIT(PWM_IDR,     0xFFFB8014, __WRITE,      __pwm_ier_bits );
__IO_REG32_BIT(PWM_IMR,     0xFFFB8018, __READ,       __pwm_ier_bits );
__IO_REG32_BIT(PWM_ISR,     0xFFFB801C, __READ,       __pwm_ier_bits );

__IO_REG32_BIT(PWM_CMR0,    0xFFFB8200, __READ_WRITE, __pwm_cmrx_bits);
__IO_REG32(    PWM_CDTY0,   0xFFFB8204, __READ_WRITE                 );
__IO_REG32(    PWM_CPRD0,   0xFFFB8208, __READ_WRITE                 );
__IO_REG32(    PWM_CCNT0,   0xFFFB820C, __READ                       );
__IO_REG32(    PWM_CUPD0,   0xFFFB8210, __WRITE                      );
__IO_REG32_BIT(PWM_CMR1,    0xFFFB8220, __READ_WRITE, __pwm_cmrx_bits);
__IO_REG32(    PWM_CDTY1,   0xFFFB8224, __READ_WRITE                 );
__IO_REG32(    PWM_CPRD1,   0xFFFB8228, __READ_WRITE                 );
__IO_REG32(    PWM_CCNT1,   0xFFFB822C, __READ                       );
__IO_REG32(    PWM_CUPD1,   0xFFFB8230, __WRITE                      );

/***************************************************************************
 **
 ** MCI0
 **
 ***************************************************************************/
__IO_REG32_BIT(MCI0_CR,      0xFFF80000, __WRITE,      __mci_cr_bits  );
__IO_REG32_BIT(MCI0_MR,      0xFFF80004, __READ_WRITE, __mci_mr_bits  );
__IO_REG32_BIT(MCI0_DTOR,    0xFFF80008, __READ_WRITE, __mci_dtor_bits);
__IO_REG32_BIT(MCI0_SDCR,    0xFFF8000C, __READ_WRITE, __mci_sdcr_bits);
__IO_REG32(    MCI0_ARGR,    0xFFF80010, __READ_WRITE                 );
__IO_REG32_BIT(MCI0_CMDR,    0xFFF80014, __WRITE,      __mci_cmdr_bits);
__IO_REG32_BIT(MCI0_BLKR,    0xFFF80018, __READ_WRITE, __mci_blkr_bits);
__IO_REG32(    MCI0_RSPR1,   0xFFF80020, __READ                       );
__IO_REG32(    MCI0_RSPR2,   0xFFF80024, __READ                       );
__IO_REG32(    MCI0_RSPR3,   0xFFF80028, __READ                       );
__IO_REG32(    MCI0_RSPR4,   0xFFF8002C, __READ                       );
__IO_REG32(    MCI0_RDR,     0xFFF80030, __READ                       );
__IO_REG32(    MCI0_TDR,     0xFFF80034, __WRITE                      );
__IO_REG32_BIT(MCI0_SR,      0xFFF80040, __READ,       __mci_sr_bits  );
__IO_REG32_BIT(MCI0_IER,     0xFFF80044, __WRITE,      __mci_sr_bits  );
__IO_REG32_BIT(MCI0_IDR,     0xFFF80048, __WRITE,      __mci_sr_bits  );
__IO_REG32_BIT(MCI0_IMR,     0xFFF8004C, __READ,       __mci_sr_bits  );

__IO_REG32(    MCI0_RPR,     0xFFF80100, __READ_WRITE                  );
__IO_REG16(    MCI0_RCR,     0xFFF80104, __READ_WRITE                  );
__IO_REG32(    MCI0_TPR,     0xFFF80108, __READ_WRITE                  );
__IO_REG16(    MCI0_TCR,     0xFFF8010C, __READ_WRITE                  );
__IO_REG32(    MCI0_RNPR,    0xFFF80110, __READ_WRITE                  );
__IO_REG16(    MCI0_RNCR,    0xFFF80114, __READ_WRITE                  );
__IO_REG32(    MCI0_TNPR,    0xFFF80118, __READ_WRITE                  );
__IO_REG16(    MCI0_TNCR,    0xFFF8011C, __READ_WRITE                  );
__IO_REG32_BIT(MCI0_PTCR,    0xFFF80120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(MCI0_PTSR,    0xFFF80124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** MCI1
 **
 ***************************************************************************/
__IO_REG32_BIT(MCI1_CR,      0xFFF84000, __WRITE,      __mci_cr_bits  );
__IO_REG32_BIT(MCI1_MR,      0xFFF84004, __READ_WRITE, __mci_mr_bits  );
__IO_REG32_BIT(MCI1_DTOR,    0xFFF84008, __READ_WRITE, __mci_dtor_bits);
__IO_REG32_BIT(MCI1_SDCR,    0xFFF8400C, __READ_WRITE, __mci_sdcr_bits);
__IO_REG32(    MCI1_ARGR,    0xFFF84010, __READ_WRITE                 );
__IO_REG32_BIT(MCI1_CMDR,    0xFFF84014, __WRITE,      __mci_cmdr_bits);
__IO_REG32_BIT(MCI1_BLKR,    0xFFF84018, __READ_WRITE, __mci_blkr_bits);
__IO_REG32(    MCI1_RSPR1,   0xFFF84020, __READ                       );
__IO_REG32(    MCI1_RSPR2,   0xFFF84024, __READ                       );
__IO_REG32(    MCI1_RSPR3,   0xFFF84028, __READ                       );
__IO_REG32(    MCI1_RSPR4,   0xFFF8402C, __READ                       );
__IO_REG32(    MCI1_RDR,     0xFFF84030, __READ                       );
__IO_REG32(    MCI1_TDR,     0xFFF84034, __WRITE                      );
__IO_REG32_BIT(MCI1_SR,      0xFFF84040, __READ,       __mci_sr_bits  );
__IO_REG32_BIT(MCI1_IER,     0xFFF84044, __WRITE,      __mci_sr_bits  );
__IO_REG32_BIT(MCI1_IDR,     0xFFF84048, __WRITE,      __mci_sr_bits  );
__IO_REG32_BIT(MCI1_IMR,     0xFFF8404C, __READ,       __mci_sr_bits  );

__IO_REG32(    MCI1_RPR,     0xFFF84100, __READ_WRITE                  );
__IO_REG16(    MCI1_RCR,     0xFFF84104, __READ_WRITE                  );
__IO_REG32(    MCI1_TPR,     0xFFF84108, __READ_WRITE                  );
__IO_REG16(    MCI1_TCR,     0xFFF8410C, __READ_WRITE                  );
__IO_REG32(    MCI1_RNPR,    0xFFF84110, __READ_WRITE                  );
__IO_REG16(    MCI1_RNCR,    0xFFF84114, __READ_WRITE                  );
__IO_REG32(    MCI1_TNPR,    0xFFF84118, __READ_WRITE                  );
__IO_REG16(    MCI1_TNCR,    0xFFF8411C, __READ_WRITE                  );
__IO_REG32_BIT(MCI1_PTCR,    0xFFF84120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(MCI1_PTSR,    0xFFF84124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** EMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(EMAC_NCR,      0xFFFBC000, __READ_WRITE, __emac_ncr_bits );
__IO_REG32_BIT(EMAC_NCFG,     0xFFFBC004, __READ_WRITE, __emac_ncfg_bits);
__IO_REG32_BIT(EMAC_NSR,      0xFFFBC008, __READ      , __emac_nsr_bits );
__IO_REG32_BIT(EMAC_TSR,      0xFFFBC014, __READ_WRITE, __emac_tsr_bits );
__IO_REG32(    EMAC_RBQP,     0xFFFBC018, __READ_WRITE                  );
__IO_REG32(    EMAC_TBQP,     0xFFFBC01C, __READ_WRITE                  );
__IO_REG32_BIT(EMAC_RSR,      0xFFFBC020, __READ_WRITE, __emac_rsr_bits );
__IO_REG32_BIT(EMAC_ISR,      0xFFFBC024, __READ_WRITE, __emac_isr_bits );
__IO_REG32_BIT(EMAC_IER,      0xFFFBC028, __WRITE     , __emac_isr_bits );
__IO_REG32_BIT(EMAC_IDR,      0xFFFBC02C, __WRITE     , __emac_isr_bits );
__IO_REG32_BIT(EMAC_IMR,      0xFFFBC030, __READ      , __emac_isr_bits );
__IO_REG32_BIT(EMAC_MAN,      0xFFFBC034, __READ_WRITE, __emac_man_bits );
__IO_REG16(    EMAC_PTR,      0xFFFBC038, __READ_WRITE                  );
__IO_REG16(    EMAC_PFR,      0xFFFBC03C, __READ_WRITE                  );
__IO_REG32_BIT(EMAC_FTO,      0xFFFBC040, __READ_WRITE, __emac_fto_bits );
__IO_REG16(    EMAC_SCF,      0xFFFBC044, __READ_WRITE                  );
__IO_REG16(    EMAC_MCF,      0xFFFBC048, __READ_WRITE                  );
__IO_REG32_BIT(EMAC_FRO,      0xFFFBC04C, __READ_WRITE, __emac_fro_bits );
__IO_REG8(     EMAC_FCSE,     0xFFFBC050, __READ_WRITE                  );
__IO_REG8(     EMAC_ALE,      0xFFFBC054, __READ_WRITE                  );
__IO_REG16(    EMAC_DTF,      0xFFFBC058, __READ_WRITE                  );
__IO_REG8(     EMAC_LCOL,     0xFFFBC05C, __READ_WRITE                  );
__IO_REG8(     EMAC_ECOL,     0xFFFBC060, __READ_WRITE                  );
__IO_REG8(     EMAC_TUND,     0xFFFBC064, __READ_WRITE                  );
__IO_REG8(     EMAC_CSE,      0xFFFBC068, __READ_WRITE                  );
__IO_REG16(    EMAC_RRE,      0xFFFBC06C, __READ_WRITE                  );
__IO_REG8(     EMAC_ROV,      0xFFFBC070, __READ_WRITE                  );
__IO_REG8(     EMAC_RSE,      0xFFFBC074, __READ_WRITE                  );
__IO_REG8(     EMAC_ELE,      0xFFFBCF78, __READ_WRITE                  );
__IO_REG8(     EMAC_RJA,      0xFFFBC07C, __READ_WRITE                  );
__IO_REG8(     EMAC_USF,      0xFFFBC080, __READ_WRITE                  );
__IO_REG8(     EMAC_STE,      0xFFFBC084, __READ_WRITE                  );
__IO_REG8(     EMAC_RLE,      0xFFFBC088, __READ_WRITE                  );
__IO_REG32(    EMAC_HRB,      0xFFFBC090, __READ_WRITE                  );
__IO_REG32(    EMAC_HRT,      0xFFFBC094, __READ_WRITE                  );
__IO_REG32(    EMAC_SA1B,     0xFFFBC098, __READ_WRITE                  );
__IO_REG16(    EMAC_SA1T,     0xFFFBC09C, __READ_WRITE                  );
__IO_REG32(    EMAC_SA2B,     0xFFFBC0A0, __READ_WRITE                  );
__IO_REG16(    EMAC_SA2T,     0xFFFBC0A4, __READ_WRITE                  );
__IO_REG32(    EMAC_SA3B,     0xFFFBC0A8, __READ_WRITE                  );
__IO_REG16(    EMAC_SA3T,     0xFFFBC0AC, __READ_WRITE                  );
__IO_REG32(    EMAC_SA4B,     0xFFFBC0B0, __READ_WRITE                  );
__IO_REG16(    EMAC_SA4T,     0xFFFBC0B4, __READ_WRITE                  );
__IO_REG16(    EMAC_TID,      0xFFFBC0B8, __READ_WRITE                  );
__IO_REG32_BIT(EMAC_USRIO,    0xFFFBC0C0, __READ_WRITE, __emac_usrio_bits);

/***************************************************************************
 **
 ** USB HOST (OHCI)
 **
 ***************************************************************************/
__IO_REG32_BIT(HCREVISION,            0x00700000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(HCCONTROL,             0x00700004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(HCCOMMANDSTATUS,       0x00700008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(HCINTERRUPTSTATUS,     0x0070000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(HCINTERRUPTENABLE,     0x00700010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCINTERRUPTDISABLE,    0x00700014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HCHCCA,                0x00700018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(HCPERIODCURRENTED,     0x0070001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(HCCONTROLHEADED,       0x00700020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(HCCONTROLCURRENTED,    0x00700024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(HCBULKHEADED,          0x00700028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(HCBULKCURRENTED,       0x0070002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(HCDONEHEAD,            0x00700030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(HCFMINTERVAL,          0x00700034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(HCFMREMAINING,         0x00700038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(HCFMNUMBER,            0x0070003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(HCPERIODICSTART,       0x00700040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(HCLSTHRESHOLD,         0x00700044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(HCRHDESCRIPTORA,       0x00700048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(HCRHDESCRIPTORB,       0x0070004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HCRHSTATUS,            0x00700050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS1,       0x00700054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(HCRHPORTSTATUS2,       0x00700058,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32(    HCRMID,                0x007000FC,__READ);

/***************************************************************************
 **
 ** UDPHS
 **
 ***************************************************************************/
__IO_REG32_BIT(UDPHS_CTRL,        0xFFF78000, __READ_WRITE, __udphs_ctrl_bits       );
__IO_REG32_BIT(UDPHS_FNUM,        0xFFF78004, __READ,       __udphs_fnum_bits       );
__IO_REG32_BIT(UDPHS_IEN,         0xFFF78010, __READ_WRITE, __udphs_ien_bits        );
__IO_REG32_BIT(UDPHS_INTSTA,      0xFFF78014, __READ,       __udphs_intsta_bits     );
__IO_REG32_BIT(UDPHS_CLRINT,      0xFFF78018, __WRITE,      __udphs_clrint_bits     );
__IO_REG32_BIT(UDPHS_EPTRST,      0xFFF7801C, __WRITE,      __udphs_eptrst_bits     );

__IO_REG32_BIT(UDPHS_TST,         0xFFF780E0, __READ_WRITE, __udphs_tst_bits        );

__IO_REG32_BIT(UDPHS_EPTCFG0,     0xFFF78100, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB0,  0xFFF78104, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS0,  0xFFF78108, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL0,     0xFFF7810C, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA0,  0xFFF78114, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA0,  0xFFF78118, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA0,     0xFFF7811C, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG1,     0xFFF78120, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB1,  0xFFF78124, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS1,  0xFFF78128, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL1,     0xFFF7812C, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA1,  0xFFF78134, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA1,  0xFFF78138, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA1,     0xFFF7813C, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG2,     0xFFF78140, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB2,  0xFFF78144, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS2,  0xFFF78148, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL2,     0xFFF7814C, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA2,  0xFFF78154, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA2,  0xFFF78158, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA2,     0xFFF7815C, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG3,     0xFFF78160, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB3,  0xFFF78164, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS3,  0xFFF78168, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL3,     0xFFF7816C, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA3,  0xFFF78174, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA3,  0xFFF78178, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA3,     0xFFF7817C, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG4,     0xFFF78180, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB4,  0xFFF78184, __WRITE,      __udphs_eptctlx_bits );
__IO_REG32_BIT(UDPHS_EPTCTLDIS4,  0xFFF78188, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL4,     0xFFF7818C, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA4,  0xFFF78194, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA4,  0xFFF78198, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA4,     0xFFF7819C, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG5,     0xFFF781A0, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB5,  0xFFF781A4, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS5,  0xFFF781A8, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL5,     0xFFF781AC, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA5,  0xFFF781B4, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA5,  0xFFF781B8, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA5,     0xFFF781BC, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG6,     0xFFF781C0, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB6,  0xFFF781C4, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS6,  0xFFF781C8, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL6,     0xFFF781CC, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA6,  0xFFF781D4, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA6,  0xFFF781D8, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA6,     0xFFF781DC, __READ,       __udphs_eptstax_bits    );

__IO_REG32_BIT(UDPHS_EPTCFG7,     0xFFF781E0, __READ_WRITE, __udphs_eptcfgx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLENB7,  0xFFF781E4, __WRITE,      __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTCTLDIS7,  0xFFF781E8, __WRITE,      __udphs_eptctldisx_bits );
__IO_REG32_BIT(UDPHS_EPTCTL7,     0xFFF781EC, __READ,       __udphs_eptctlx_bits    );
__IO_REG32_BIT(UDPHS_EPTSETSTA7,  0xFFF781F4, __WRITE,      __udphs_eptsetstax_bits );
__IO_REG32_BIT(UDPHS_EPTCLRSTA7,  0xFFF781F8, __WRITE,      __udphs_eptclrstax_bits );
__IO_REG32_BIT(UDPHS_EPTSTA7,     0xFFF781FC, __READ,       __udphs_eptstax_bits    );

__IO_REG32(    UDPHS_DMANXTDSC1,  0xFFF78310, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS1, 0xFFF78314, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL1, 0xFFF78318, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS1,  0xFFF7831C, __READ_WRITE, __udphs_dmastatusx_bits );

__IO_REG32(    UDPHS_DMANXTDSC2,  0xFFF78320, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS2, 0xFFF78324, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL2, 0xFFF78328, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS2,  0xFFF7832C, __READ_WRITE, __udphs_dmastatusx_bits );

__IO_REG32(    UDPHS_DMANXTDSC3,  0xFFF78330, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS3, 0xFFF78334, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL3, 0xFFF78338, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS3,  0xFFF7833C, __READ_WRITE, __udphs_dmastatusx_bits );

__IO_REG32(    UDPHS_DMANXTDSC4,  0xFFF78340, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS4, 0xFFF78344, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL4, 0xFFF78348, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS4,  0xFFF7834C, __READ_WRITE, __udphs_dmastatusx_bits );

__IO_REG32(    UDPHS_DMANXTDSC5,  0xFFF78350, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS5, 0xFFF78354, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL5, 0xFFF78358, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS5,  0xFFF7835C, __READ_WRITE, __udphs_dmastatusx_bits );

__IO_REG32(    UDPHS_DMANXTDSC6,  0xFFF78360, __READ_WRITE                          );
__IO_REG32(    UDPHS_DMAADDRESS6, 0xFFF78364, __READ_WRITE                          );
__IO_REG32_BIT(UDPHS_DMACONTROL6, 0xFFF78368, __READ_WRITE, __udphs_dmacontrolx_bits);
__IO_REG32_BIT(UDPHS_DMASTATUS6,  0xFFF7836C, __READ_WRITE, __udphs_dmastatusx_bits );

/***************************************************************************
 **
 ** ISI
 **
 ***************************************************************************/
__IO_REG32_BIT(ISI_CR1,           0xFFFC4000, __READ_WRITE, __isi_cr1_bits     );
__IO_REG32_BIT(ISI_CR2,           0xFFFC4004, __READ_WRITE, __isi_cr2_bits     );
__IO_REG32_BIT(ISI_SR,            0xFFFC4008, __READ,       __isi_sr_bits      );
__IO_REG32_BIT(ISI_IER,           0xFFFC400C, __READ_WRITE, __isi_ier_bits     );
__IO_REG32_BIT(ISI_IDR,           0xFFFC4010, __READ_WRITE, __isi_ier_bits     );
__IO_REG32_BIT(ISI_IMR,           0xFFFC4014, __READ_WRITE, __isi_ier_bits     );
__IO_REG32_BIT(ISI_PSIZE,         0xFFFC4020, __READ_WRITE, __isi_psize_bits   );
__IO_REG8(     ISI_PDECF,         0xFFFC4024, __READ_WRITE                     );
__IO_REG32(    ISI_PPFBD,         0xFFFC4028, __READ_WRITE                     );
__IO_REG32(    ISI_CDBA,          0xFFFC402C, __READ_WRITE                     );
__IO_REG32_BIT(ISI_Y2R_SET0,      0xFFFC4030, __READ_WRITE, __isi_y2r_set0_bits);
__IO_REG32_BIT(ISI_Y2R_SET1,      0xFFFC4034, __READ_WRITE, __isi_y2r_set1_bits);
__IO_REG32_BIT(ISI_R2Y_SET0,      0xFFFC4038, __READ_WRITE, __isi_r2y_set0_bits);
__IO_REG32_BIT(ISI_R2Y_SET1,      0xFFFC403C, __READ_WRITE, __isi_r2y_set1_bits);
__IO_REG32_BIT(ISI_R2Y_SET2,      0xFFFC4040, __READ_WRITE, __isi_r2y_set2_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CR,            0xFFFC0000, __WRITE,      __adc_cr_bits  );
__IO_REG32_BIT(ADC_MR,            0xFFFC0004, __READ_WRITE, __adc_mr_bits  );
__IO_REG32_BIT(ADC_CHER,          0xFFFC0010, __WRITE,      __adc_cher_bits);
__IO_REG32_BIT(ADC_CHDR,          0xFFFC0014, __WRITE,      __adc_cher_bits);
__IO_REG32_BIT(ADC_CHSR,          0xFFFC0018, __READ,       __adc_cher_bits);
__IO_REG32_BIT(ADC_SR,            0xFFFC001C, __READ,       __adc_sr_bits  );
__IO_REG32_BIT(ADC_LCDR,          0xFFFC0020, __READ,       __adc_lcdr_bits);
__IO_REG32_BIT(ADC_IER,           0xFFFC0024, __WRITE,      __adc_sr_bits  );
__IO_REG32_BIT(ADC_IDR,           0xFFFC0028, __WRITE,      __adc_sr_bits  );
__IO_REG32_BIT(ADC_IMR,           0xFFFC002C, __READ,       __adc_sr_bits  );
__IO_REG32_BIT(ADC_CDR0,          0xFFFC0030, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR1,          0xFFFC0034, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR2,          0xFFFC0038, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR3,          0xFFFC003C, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR4,          0xFFFC0040, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR5,          0xFFFC0044, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR6,          0xFFFC0048, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR7,          0xFFFC004C, __READ,       __adc_cdr_bits );


#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 ** AT91CAP9S500 Interrupt Codes
 **
 ***************************************************************************/
#define INT_FIQ      0
#define INT_SYS      1
#define INT_PIOA_D   2
#define INT_MPB0     3
#define INT_MPB1     4
#define INT_MPB2     5
#define INT_MPB3     6
#define INT_MPB4     7
#define INT_US0      8
#define INT_US1      9
#define INT_US2     10
#define INT_MCI0    11
#define INT_MCI1    12
#define INT_CAN     13
#define INT_TWI     14
#define INT_SPI0    15
#define INT_SPI1    16
#define INT_SSC0    17
#define INT_SSC1    18
#define INT_AC97    19
#define INT_TC      20
#define INT_PWMC    21
#define INT_EMAC    22

#define INT_ADCC    24
#define INT_ISI     25
#define INT_LCDC    26
#define INT_DMA     27
#define INT_UDPHS   28
#define INT_UHP     29
#define INT_IRQ0    30
#define INT_IRQ1    31


#endif	/* __IOAT91CAP9S500_H */
