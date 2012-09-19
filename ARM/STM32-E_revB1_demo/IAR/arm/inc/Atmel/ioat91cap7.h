/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Atmel Semiconductors AT91CAP7
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2008
 **
 **    $Revision: 33661 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOAT91CAP7_H
#define __IOAT91CAP7_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    AT91CAP7 SPECIAL FUNCTION REGISTERS
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
  __REG32   M12PR       : 2;
  __REG32               : 2;
  __REG32   M13PR       : 2;
  __REG32               : 2;
  __REG32   M14PR       : 2;
  __REG32               : 2;  
  __REG32   M15PR       : 2;  
  __REG32               : 2;
} __matrix_prbsx_bits;

/* Bus Matrix Master Remap Control Register */
typedef struct {
  __REG32   RCB0        : 1;
  __REG32   RCB1        : 1;
  __REG32   RCB2        : 1;
  __REG32   RCB3        : 1;
  __REG32   RCB4        : 1;
  __REG32   RCB5        : 1;
  __REG32               :26;
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
  __REG32               :23;
} __matrix_ebicsa_bits;

/* Matrix USB Pad Pull-up Control Register */
typedef struct {
  __REG32               :30;
  __REG32   UDP_PUP_ON  : 1;
  __REG32   PUP_IDLE    : 1;
} __matrix_usbpcr_bits;

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


/* SDRAMC Mode Register */
typedef struct {
  __REG32   MODE        : 3;
  __REG32               :29;
} __sdramc_mr_bits;

/* SDRAMC Refresh Timer Register */
typedef struct {
  __REG32   COUNT       :12;
  __REG32               :20;
}__sdramc_tr_bits;

/* SDRAMC Configuration Register */
typedef struct {
  __REG32   NC          : 2;
  __REG32   NR          : 2;
  __REG32   NB          : 1;
  __REG32   CAS         : 2;
  __REG32   DWB         : 1;
  __REG32   TWR         : 4;
  __REG32   TRC         : 4;
  __REG32   TRP         : 4;
  __REG32   TRCD        : 4;
  __REG32   TRAS        : 4;
  __REG32   TXSR        : 4;
}__sdramc_cr_bits;

/* SDRAMC High Speed Register */
typedef struct {
  __REG32   DA          : 1;
  __REG32               :31;
} __sdramc_hsr_bits;

/* SDRAMC Low-power Register */
typedef struct {
  __REG32   LPCB        : 2;
  __REG32               : 2;
  __REG32   PASR        : 3;
  __REG32               : 1;
  __REG32   TCS         : 2;
  __REG32   DS          : 2;
  __REG32   TIMEOUT     : 2;
  __REG32               :18;
}__sdramc_lpr_bits;

/* SDRAMC Interrupt Enable Register */
/* SDRAMC Interrupt Disable Register */
/* SDRAMC Interrupt Mask Register */
/* SDRAMC Interrupt Status Register */
typedef struct {
  __REG32   RES         : 1;
  __REG32               :31;
} __sdramc_ier_bits;

/* SDRAMC Memory Device Register */
typedef struct {
  __REG32   MD          : 2;
  __REG32               :30;
}__sdramc_mdr_bits;

/* DBGU, USART, SSC, SPI, MCI, ect.Transfer Control Register */
typedef struct {
  __REG32   RXTEN       : 1;
  __REG32   RXTDIS      : 1;
  __REG32               : 6;
  __REG32   TXTEN       : 1;
  __REG32   TXTDIS      : 1;
  __REG32               :22; 
} __pdc_ptcr_bits;

/* DBGU, USART, SSC, SPI, MCI, etc.Transfer Status Register */
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
  __REG32   UDP         : 1;
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
  __REG32   PIOA        : 1;
  __REG32   PIOB        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   SPI0        : 1;
  __REG32   TC0         : 1;
  __REG32   TC1         : 1;
  __REG32   TC2         : 1;
  __REG32   UDP         : 1;
  __REG32   ADC         : 1;
  __REG32   MPP0        : 1;
  __REG32   MPP1        : 1;
  __REG32   MPP2        : 1;
  __REG32   MPP3        : 1;
  __REG32   MPP4        : 1;
  __REG32   MPP5        : 1;
  __REG32   MPP6        : 1;
  __REG32   MPP7        : 1;
  __REG32   MPP8        : 1;
  __REG32   MPP9        : 1;
  __REG32   MPP10       : 1;
  __REG32   MPP11       : 1;
  __REG32   MPP12       : 1;
  __REG32   MPP13       : 1;
  __REG32   MPMA        : 1;
  __REG32   MPMB        : 1;
  __REG32   MPMC        : 1;
  __REG32   MPMD        : 1;
  __REG32               : 1;
  __REG32               : 1;
} __pmc_pcer_bits;

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
  __REG32               :27;
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
  __REG32               : 4;
  __REG32   PCKRDY0     : 1;
  __REG32   PCKRDY1     : 1;
  __REG32   PCKRDY2     : 1;
  __REG32   PCKRDY3     : 1;
  __REG32               :20;
} __pmc_ier_bits;

/* PMC Status Register */
typedef struct {
  __REG32   MOSCS       : 1;
  __REG32   LOCKA       : 1;
  __REG32   LOCKB       : 1;
  __REG32   MCKRDY      : 1;
  __REG32               : 3;
  __REG32   OSC_SEL     : 1;
  __REG32   PCKRDY0     : 1;
  __REG32   PCKRDY1     : 1;
  __REG32   PCKRDY2     : 1;
  __REG32   PCKRDY3     : 1;
  __REG32               :20;
} __pmc_isr_bits;

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
  __REG32   PIOA        : 1;
  __REG32   PIOB        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   SPI0        : 1;
  __REG32   TC0         : 1;
  __REG32   TC1         : 1;
  __REG32   TC2         : 1;
  __REG32   UDP         : 1;
  __REG32   ADC         : 1;
  __REG32   MPP0        : 1;
  __REG32   MPP1        : 1;
  __REG32   MPP2        : 1;
  __REG32   MPP3        : 1;
  __REG32   MPP4        : 1;
  __REG32   MPP5        : 1;
  __REG32   MPP6        : 1;
  __REG32   MPP7        : 1;
  __REG32   MPP8        : 1;
  __REG32   MPP9        : 1;
  __REG32   MPP10       : 1;
  __REG32   MPP11       : 1;
  __REG32   MPP12       : 1;
  __REG32   MPP13       : 1;
  __REG32   MPMA        : 1;
  __REG32   MPMB        : 1;
  __REG32   MPMC        : 1;
  __REG32   MPMD        : 1;
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
  __REG32   PIOA        : 1;
  __REG32   PIOB        : 1;
  __REG32   US0         : 1;
  __REG32   US1         : 1;
  __REG32   SPI0        : 1;
  __REG32   TC0         : 1;
  __REG32   TC1         : 1;
  __REG32   TC2         : 1;
  __REG32   UDP         : 1;
  __REG32   ADC         : 1;
  __REG32   MPP0        : 1;
  __REG32   MPP1        : 1;
  __REG32   MPP2        : 1;
  __REG32   MPP3        : 1;
  __REG32   MPP4        : 1;
  __REG32   MPP5        : 1;
  __REG32   MPP6        : 1;
  __REG32   MPP7        : 1;
  __REG32   MPP8        : 1;
  __REG32   MPP9        : 1;
  __REG32   MPP10       : 1;
  __REG32   MPP11       : 1;
  __REG32   MPP12       : 1;
  __REG32   MPP13       : 1;
  __REG32   MPMA        : 1;
  __REG32   MPMB        : 1;
  __REG32   MPMC        : 1;
  __REG32   MPMD        : 1;
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
  __REG32 FDIV             : 1;
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
  __REG32 DTREN            : 1;
  __REG32 DTRDIS           : 1;
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
  __REG32                  : 2;
  __REG32 RIIC             : 1;
  __REG32 DSRIC            : 1;
  __REG32 DCDIC            : 1;
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
  __REG32                  : 2;
  __REG32 RIIC             : 1;
  __REG32 DSRIC            : 1;
  __REG32 DCDIC            : 1;
  __REG32 CTSIC            : 1;
  __REG32 RI               : 1;
  __REG32 DSR              : 1;
  __REG32 DCD              : 1;
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

/* UDP Frame Number Register*/
typedef struct {
  __REG32 FRM_NUM          :11;
  __REG32                  : 5;
  __REG32 FRM_ERR          : 1;
  __REG32 FRM_OK           : 1;
  __REG32                  :14;
} __udp_frm_num_bits;

/* UDP Global State Register */
typedef struct {
  __REG32 FADDEN           : 1;
  __REG32 CONFG            : 1;
  __REG32                  :30;
} __udp_glb_stat_bits;

/* UDP Function Address Register */
typedef struct {
  __REG32 FADD             : 7;
  __REG32                  : 1;
  __REG32 FEN              : 1;
  __REG32                  :23;
} __udp_faddr_bits;

/* UDP Interrupt Enable Register */
/* UDP Interrupt Disable Register */
/* UDP Interrupt Mask Register */
typedef struct {
  __REG32 EP0INT           : 1;
  __REG32 EP1INT           : 1;
  __REG32 EP2INT           : 1;
  __REG32 EP3INT           : 1;
  __REG32 EP4INT           : 1;
  __REG32 EP5INT           : 1;
  __REG32                  : 2;
  __REG32 RXSUSP           : 1;
  __REG32 RXRSM            : 1;
  __REG32                  : 1;
  __REG32 SOFINT           : 1;
  __REG32                  : 1;
  __REG32 WAKEUP           : 1;
  __REG32                  :18;
} __udp_ier_bits;

/* UDP Interrupt Status Register */
typedef struct {
  __REG32 EP0INT           : 1;
  __REG32 EP1INT           : 1;
  __REG32 EP2INT           : 1;
  __REG32 EP3INT           : 1;
  __REG32 EP4INT           : 1;
  __REG32 EP5INT           : 1;
  __REG32                  : 2;
  __REG32 RXSUSP           : 1;
  __REG32 RXRSM            : 1;
  __REG32                  : 1;
  __REG32 SOFINT           : 1;
  __REG32 ENDBUSRES        : 1;
  __REG32 WAKEUP           : 1;
  __REG32                  :18;
} __udp_isr_bits;

/* UDP Interrupt Clear Register */
typedef struct {
  __REG32                  : 8;
  __REG32 RXSUSP           : 1;
  __REG32 RXRSM            : 1;
  __REG32                  : 1;
  __REG32 SOFINT           : 1;
  __REG32 ENDBUSRES        : 1;
  __REG32 WAKEUP           : 1;
  __REG32                  :18;
}__udp_icr_bits;


/* UDP Reset Endpoint Register */
typedef struct {
  __REG32 EP0              : 1;
  __REG32 EP1              : 1;
  __REG32 EP2              : 1;
  __REG32 EP3              : 1;
  __REG32 EP4              : 1;
  __REG32 EP5              : 1;
  __REG32                  :26;
} __udp_rst_ep_bits;

/* UDP Endpoint Control and Status Register*/
typedef struct {
  __REG32 TXCOMP           : 1;
  __REG32 RX_DATA_BK0      : 1;
  __REG32 RXSETUP          : 1;
  __REG32 STALLSENT        : 1;
  __REG32 TXPKTRDY         : 1;
  __REG32 FORCESTALL       : 1;
  __REG32 RX_DATA_BK1      : 1;
  __REG32 DIR              : 1;
  __REG32 EPTYPE           : 3;
  __REG32 DTGLE            : 1;
  __REG32                  : 3;
  __REG32 EPEDS            : 1;
  __REG32 RXBYTECNT        :11;
  __REG32                  : 5;
} __udp_csrx_bits;

/* UDP Endpoint Control and Status Register*/
typedef struct {
  __REG32                  : 8;
  __REG32 TXVDIS           : 1;
  __REG32                  :23;
}__udp_txvc_bits;

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
 ** RTT0
 **
 ***************************************************************************/
__IO_REG32_BIT(RTT0_MR,         0xFFFFFD20, __READ_WRITE,  __rtt_mr_bits   );
__IO_REG32(    RTT0_AR,         0xFFFFFD24, __READ_WRITE                   );
__IO_REG32(    RTT0_VR,         0xFFFFFD28, __READ                         );
__IO_REG32_BIT(RTT0_SR,         0xFFFFFD2C, __READ,        __rtt_sr_bits   );
 
/***************************************************************************
 **
 ** RTT1
 **
 ***************************************************************************/
__IO_REG32_BIT(RTT1_MR,         0xFFFFFD50, __READ_WRITE,  __rtt_mr_bits   );
__IO_REG32(    RTT1_AR,         0xFFFFFD54, __READ_WRITE                   );
__IO_REG32(    RTT1_VR,         0xFFFFFD58, __READ                         );
__IO_REG32_BIT(RTT1_SR,         0xFFFFFD5C, __READ,        __rtt_sr_bits   );

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
__IO_REG32_BIT(MATRIX_MCFG0,    0xFFFFEE00, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG1,    0xFFFFEE04, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG2,    0xFFFFEE08, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG3,    0xFFFFEE0C, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG4,    0xFFFFEE10, __READ_WRITE, __matrix_mcfgx_bits);
__IO_REG32_BIT(MATRIX_MCFG5,    0xFFFFEE14, __READ_WRITE, __matrix_mcfgx_bits);

__IO_REG32_BIT(MATRIX_SCFG0,    0xFFFFEE40, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG1,    0xFFFFEE44, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG2,    0xFFFFEE48, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG3,    0xFFFFEE4C, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG4,    0xFFFFEE50, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG5,    0xFFFFEE54, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG6,    0xFFFFEE58, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG7,    0xFFFFEE5C, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG8,    0xFFFFEE60, __READ_WRITE, __matrix_scfgx_bits);
__IO_REG32_BIT(MATRIX_SCFG9,    0xFFFFEE64, __READ_WRITE, __matrix_scfgx_bits);

__IO_REG32_BIT(MATRIX_PRAS0,    0xFFFFEE80, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS0,    0xFFFFEE84, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS1,    0xFFFFEE88, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS1,    0xFFFFEE8C, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS2,    0xFFFFEE90, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS2,    0xFFFFEE94, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS3,    0xFFFFEE98, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS3,    0xFFFFEE9C, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS4,    0xFFFFEEA0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS4,    0xFFFFEEA4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS5,    0xFFFFEEA8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS5,    0xFFFFEEAC, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS6,    0xFFFFEEB0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS6,    0xFFFFEEB4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS7,    0xFFFFEEB8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS7,    0xFFFFEEBC, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS8,    0xFFFFEEC0, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS8,    0xFFFFEEC4, __READ_WRITE, __matrix_prbsx_bits);
__IO_REG32_BIT(MATRIX_PRAS9,    0xFFFFEEC8, __READ_WRITE, __matrix_prasx_bits);
__IO_REG32_BIT(MATRIX_PRBS9,    0xFFFFEECC, __READ_WRITE, __matrix_prbsx_bits);

__IO_REG32_BIT(MATRIX_MRCR,     0xFFFFEF00, __READ_WRITE, __matrix_mrcr_bits );
__IO_REG32_BIT(MATRIX_EBICSA,   0xFFFFEF30, __READ_WRITE, __matrix_ebicsa_bits );
__IO_REG32_BIT(MATRIX_USBPCR,   0xFFFFEF34, __READ_WRITE, __matrix_usbpcr_bits );

/***************************************************************************
 **
 ** SMC
 **
 ***************************************************************************/
__IO_REG32_BIT(SMC_SETUP_CS0,    0xFFFFEC00, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS0,    0xFFFFEC04, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS0,    0xFFFFEC08, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS0,     0xFFFFEC0C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS1,    0xFFFFEC10, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS1,    0xFFFFEC14, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS1,    0xFFFFEC18, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS1,     0xFFFFEC1C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS2,    0xFFFFEC20, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS2,    0xFFFFEC24, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS2,    0xFFFFEC28, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS2,     0xFFFFEC2C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS3,    0xFFFFEC30, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS3,    0xFFFFEC34, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS3,    0xFFFFEC38, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS3,     0xFFFFEC3C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS4,    0xFFFFEC40, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS4,    0xFFFFEC44, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS4,    0xFFFFEC48, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS4,     0xFFFFEC4C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS5,    0xFFFFEC50, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS5,    0xFFFFEC54, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS5,    0xFFFFEC58, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS5,     0xFFFFEC5C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS6,    0xFFFFEC60, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS6,    0xFFFFEC64, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS6,    0xFFFFEC68, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS6,     0xFFFFEC6C, __READ_WRITE, __smc_mode_csx_bits );

__IO_REG32_BIT(SMC_SETUP_CS7,    0xFFFFEC70, __READ_WRITE, __smc_setup_csx_bits);
__IO_REG32_BIT(SMC_PULSE_CS7,    0xFFFFEC74, __READ_WRITE, __smc_pulse_csx_bits);
__IO_REG32_BIT(SMC_CYCLE_CS7,    0xFFFFEC78, __READ_WRITE, __smc_cycle_csx_bits);
__IO_REG32_BIT(SMC_MODE_CS7,     0xFFFFEC7C, __READ_WRITE, __smc_mode_csx_bits );

/***************************************************************************
 **
 ** SDRAMC
 **
 ***************************************************************************/
__IO_REG32_BIT(SDRAMC_MR,      0xFFFFEA00, __READ_WRITE, __sdramc_mr_bits  );
__IO_REG32_BIT(SDRAMC_TR,      0xFFFFEA04, __READ_WRITE, __sdramc_tr_bits  );
__IO_REG32_BIT(SDRAMC_CR,      0xFFFFEA08, __READ_WRITE, __sdramc_cr_bits  );
__IO_REG32_BIT(SDRAMC_HSR,     0xFFFFEA0C, __READ_WRITE, __sdramc_hsr_bits );
__IO_REG32_BIT(SDRAMC_LPR,     0xFFFFEA10, __READ_WRITE, __sdramc_lpr_bits );
__IO_REG32_BIT(SDRAMC_IER,     0xFFFFEA14, __WRITE     , __sdramc_ier_bits );
__IO_REG32_BIT(SDRAMC_IDR,     0xFFFFEA18, __WRITE     , __sdramc_ier_bits );
__IO_REG32_BIT(SDRAMC_IMR,     0xFFFFEA1C, __READ      , __sdramc_ier_bits );
__IO_REG32_BIT(SDRAMC_ISR,     0xFFFFEA20, __READ      , __sdramc_ier_bits );
__IO_REG32_BIT(SDRAMC_MDR,     0xFFFFEA24, __READ_WRITE, __sdramc_mdr_bits );

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
__IO_REG32_BIT(CKGR_MOR,       0xFFFFFC20, __READ_WRITE, __ckgr_mor_bits   );
__IO_REG32_BIT(CKGR_MCFR,      0xFFFFFC24, __READ,       __ckgr_mcfr_bits  );
__IO_REG32_BIT(CKGR_PLLAR,     0xFFFFFC28, __READ_WRITE, __ckgr_pllar_bits );
__IO_REG32_BIT(CKGR_PLLBR,     0xFFFFFC2C, __READ_WRITE, __ckgr_pllbr_bits );
__IO_REG32_BIT(PMC_MCKR,       0xFFFFFC30, __READ_WRITE, __pmc_mckr_bits   );
__IO_REG32_BIT(PMC_PCK0,       0xFFFFFC40, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_PCK1,       0xFFFFFC44, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_PCK2,       0xFFFFFC48, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_PCK3,       0xFFFFFC4C, __READ_WRITE, __pmc_pckx_bits   );
__IO_REG32_BIT(PMC_IER,        0xFFFFFC60, __WRITE,      __pmc_ier_bits    );
__IO_REG32_BIT(PMC_IDR,        0xFFFFFC64, __WRITE,      __pmc_ier_bits    );
__IO_REG32_BIT(PMC_SR,         0xFFFFFC68, __READ,       __pmc_isr_bits    );
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
__IO_REG32_BIT(DBGU_CR,      0xFFFFF200, __WRITE,      __dbgu_cr_bits  );
__IO_REG32_BIT(DBGU_MR,      0xFFFFF204, __READ_WRITE, __dbgu_mr_bits  );
__IO_REG32_BIT(DBGU_IER,     0xFFFFF208, __WRITE,      __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_IDR,     0xFFFFF20C, __WRITE,      __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_IMR,     0xFFFFF210, __READ,       __dbgu_ier_bits );
__IO_REG32_BIT(DBGU_SR,      0xFFFFF214, __READ,       __dbgu_ier_bits );
__IO_REG8(     DBGU_RHR,     0xFFFFF218, __READ                        );
__IO_REG8(     DBGU_THR,     0xFFFFF21C, __WRITE                       );
__IO_REG16(    DBGU_BRGR,    0xFFFFF220, __READ_WRITE                  );
__IO_REG32_BIT(DBGU_CIDR,    0xFFFFF240, __READ,       __dbgu_cidr_bits);
__IO_REG32(    DBGU_EXID,    0xFFFFF244, __READ                        );
__IO_REG32_BIT(DBGU_FNR,     0xFFFFF248, __READ_WRITE, __dbgu_fnr_bits );

__IO_REG32(    DBGU_RPR,     0xFFFFF300, __READ_WRITE                  );
__IO_REG16(    DBGU_RCR,     0xFFFFF304, __READ_WRITE                  );
__IO_REG32(    DBGU_TPR,     0xFFFFF308, __READ_WRITE                  );
__IO_REG16(    DBGU_TCR,     0xFFFFF30C, __READ_WRITE                  );
__IO_REG32(    DBGU_RNPR,    0xFFFFF310, __READ_WRITE                  );
__IO_REG16(    DBGU_RNCR,    0xFFFFF314, __READ_WRITE                  );
__IO_REG32(    DBGU_TNPR,    0xFFFFF318, __READ_WRITE                  );
__IO_REG16(    DBGU_TNCR,    0xFFFFF31C, __READ_WRITE                  );
__IO_REG32_BIT(DBGU_PTCR,    0xFFFFF320, __WRITE,      __pdc_ptcr_bits );
__IO_REG32_BIT(DBGU_PTSR,    0xFFFFF324, __READ,       __pdc_ptsr_bits );

/***************************************************************************
 **
 ** PIOA
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOA_PER,       0xFFFFF400, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PDR,       0xFFFFF404, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PSR,       0xFFFFF408, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_OER,       0xFFFFF410, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_ODR,       0xFFFFF414, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_OSR,       0xFFFFF418, __READ_WRITE, __pioper_bits);
__IO_REG32_BIT(PIOA_IFER,      0xFFFFF420, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IFDR,      0xFFFFF424, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IFSR,      0xFFFFF428, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_SODR,      0xFFFFF430, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_CODR,      0xFFFFF434, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_ODSR,      0xFFFFF438, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_PDSR,      0xFFFFF43C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_IER,       0xFFFFF440, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IDR,       0xFFFFF444, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_IMR,       0xFFFFF448, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_ISR,       0xFFFFF44C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_MDER,      0xFFFFF450, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_MDDR,      0xFFFFF454, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_MDSR,      0xFFFFF458, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_PUDR,      0xFFFFF460, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PUER,      0xFFFFF464, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_PUSR,      0xFFFFF468, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_ASR,       0xFFFFF470, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_BSR,       0xFFFFF474, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_ABSR,      0xFFFFF478, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOA_OWER,      0xFFFFF4A0, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_OWDR,      0xFFFFF4A4, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOA_OWSR,      0xFFFFF4A8, __READ      , __pioper_bits);


/***************************************************************************
 **
 ** PIOB
 **
 ***************************************************************************/
__IO_REG32_BIT(PIOB_PER,       0xFFFFF600, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PDR,       0xFFFFF604, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PSR,       0xFFFFF608, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_OER,       0xFFFFF610, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_ODR,       0xFFFFF614, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_OSR,       0xFFFFF618, __READ_WRITE, __pioper_bits);
__IO_REG32_BIT(PIOB_IFER,      0xFFFFF620, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IFDR,      0xFFFFF624, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IFSR,      0xFFFFF628, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_SODR,      0xFFFFF630, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_CODR,      0xFFFFF634, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_ODSR,      0xFFFFF638, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_PDSR,      0xFFFFF63C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_IER,       0xFFFFF640, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IDR,       0xFFFFF644, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_IMR,       0xFFFFF648, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_ISR,       0xFFFFF64C, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_MDER,      0xFFFFF650, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_MDDR,      0xFFFFF654, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_MDSR,      0xFFFFF658, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_PUDR,      0xFFFFF660, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PUER,      0xFFFFF664, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_PUSR,      0xFFFFF668, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_ASR,       0xFFFFF670, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_BSR,       0xFFFFF674, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_ABSR,      0xFFFFF678, __READ      , __pioper_bits);
__IO_REG32_BIT(PIOB_OWER,      0xFFFFF6A0, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_OWDR,      0xFFFFF6A4, __WRITE     , __pioper_bits);
__IO_REG32_BIT(PIOB_OWSR,      0xFFFFF6A8, __READ      , __pioper_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_CR,         0xFFFAC000, __WRITE     , __spicr_bits  );
__IO_REG32_BIT(SPI0_MR,         0xFFFAC004, __READ_WRITE, __spimr_bits  );
__IO_REG32_BIT(SPI0_RDR,        0xFFFAC008, __READ      , __spirdr_bits );
__IO_REG32_BIT(SPI0_TDR,        0xFFFAC00C, __WRITE     , __spitdr_bits );
__IO_REG32_BIT(SPI0_SR,         0xFFFAC010, __READ      , __spisr_bits  );
__IO_REG32_BIT(SPI0_IER,        0xFFFAC014, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI0_IDR,        0xFFFAC018, __WRITE     , __spiier_bits );
__IO_REG32_BIT(SPI0_IMR,        0xFFFAC01C, __READ      , __spiier_bits );
__IO_REG32_BIT(SPI0_CSR0,       0xFFFAC030, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR1,       0xFFFAC034, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR2,       0xFFFAC038, __READ_WRITE, __spicsrx_bits);
__IO_REG32_BIT(SPI0_CSR3,       0xFFFAC03C, __READ_WRITE, __spicsrx_bits);

__IO_REG32(    SPI0_RPR,        0xFFFAC100, __READ_WRITE                  );
__IO_REG16(    SPI0_RCR,        0xFFFAC104, __READ_WRITE                  );
__IO_REG32(    SPI0_TPR,        0xFFFAC108, __READ_WRITE                  );
__IO_REG16(    SPI0_TCR,        0xFFFAC10C, __READ_WRITE                  );
__IO_REG32(    SPI0_RNPR,       0xFFFAC110, __READ_WRITE                  );
__IO_REG16(    SPI0_RNCR,       0xFFFAC114, __READ_WRITE                  );
__IO_REG32(    SPI0_TNPR,       0xFFFAC118, __READ_WRITE                  );
__IO_REG16(    SPI0_TNCR,       0xFFFAC11C, __READ_WRITE                  );
__IO_REG32_BIT(SPI0_PTCR,       0xFFFAC120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(SPI0_PTSR,       0xFFFAC124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** USART0
 **
 ***************************************************************************/
__IO_REG32_BIT(US0_CR,          0xFFFB0000, __WRITE     , __uscr_bits  );
__IO_REG32_BIT(US0_MR,          0xFFFB0004, __READ_WRITE, __usmr_bits  );
__IO_REG32_BIT(US0_IER,         0xFFFB0008, __WRITE     , __usier_bits );
__IO_REG32_BIT(US0_IDR,         0xFFFB000C, __WRITE     , __usier_bits );
__IO_REG32_BIT(US0_IMR,         0xFFFB0010, __READ      , __usier_bits );
__IO_REG32_BIT(US0_CSR,         0xFFFB0014, __READ      , __uscsr_bits );
__IO_REG32_BIT(US0_RHR,         0xFFFB0018, __READ      , __usrhr_bits );
__IO_REG32_BIT(US0_THR,         0xFFFB001C, __WRITE     , __usthr_bits );
__IO_REG32_BIT(US0_BRGR,        0xFFFB0020, __READ_WRITE, __usbrgr_bits);
__IO_REG16(    US0_RTOR,        0xFFFB0024, __READ_WRITE               );
__IO_REG8(     US0_TTGR,        0xFFFB0028, __READ_WRITE               );
__IO_REG32_BIT(US0_FIDI,        0xFFFB0040, __READ_WRITE, __usfidi_bits);
__IO_REG8(     US0_NER,         0xFFFB0044, __READ                     );
__IO_REG8(     US0_IF,          0xFFFB004C, __READ_WRITE               );
__IO_REG32_BIT(US0_MAN,         0xFFFB0050, __READ_WRITE, __usman_bits );

__IO_REG32(    US0_RPR,         0xFFFB0100, __READ_WRITE                  );
__IO_REG16(    US0_RCR,         0xFFFB0104, __READ_WRITE                  );
__IO_REG32(    US0_TPR,         0xFFFB0108, __READ_WRITE                  );
__IO_REG16(    US0_TCR,         0xFFFB010C, __READ_WRITE                  );
__IO_REG32(    US0_RNPR,        0xFFFB0110, __READ_WRITE                  );
__IO_REG16(    US0_RNCR,        0xFFFB0114, __READ_WRITE                  );
__IO_REG32(    US0_TNPR,        0xFFFB0118, __READ_WRITE                  );
__IO_REG16(    US0_TNCR,        0xFFFB011C, __READ_WRITE                  );
__IO_REG32_BIT(US0_PTCR,        0xFFFB0120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(US0_PTSR,        0xFFFB0124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** USART1
 **
 ***************************************************************************/
__IO_REG32_BIT(US1_CR,          0xFFFB4000, __WRITE     , __uscr_bits  );
__IO_REG32_BIT(US1_MR,          0xFFFB4004, __READ_WRITE, __usmr_bits  );
__IO_REG32_BIT(US1_IER,         0xFFFB4008, __WRITE     , __usier_bits );
__IO_REG32_BIT(US1_IDR,         0xFFFB400C, __WRITE     , __usier_bits );
__IO_REG32_BIT(US1_IMR,         0xFFFB4010, __READ      , __usier_bits );
__IO_REG32_BIT(US1_CSR,         0xFFFB4014, __READ      , __uscsr_bits );
__IO_REG32_BIT(US1_RHR,         0xFFFB4018, __READ      , __usrhr_bits );
__IO_REG32_BIT(US1_THR,         0xFFFB401C, __WRITE     , __usthr_bits );
__IO_REG32_BIT(US1_BRGR,        0xFFFB4020, __READ_WRITE, __usbrgr_bits);
__IO_REG16(    US1_RTOR,        0xFFFB4024, __READ_WRITE               );
__IO_REG8(     US1_TTGR,        0xFFFB4028, __READ_WRITE               );
__IO_REG32_BIT(US1_FIDI,        0xFFFB4040, __READ_WRITE, __usfidi_bits);
__IO_REG8(     US1_NER,         0xFFFB4044, __READ                     );
__IO_REG8(     US1_IF,          0xFFFB404C, __READ_WRITE               );
__IO_REG32_BIT(US1_MAN,         0xFFFB4050, __READ_WRITE, __usman_bits );

__IO_REG32(    US1_RPR,         0xFFFB4100, __READ_WRITE                  );
__IO_REG16(    US1_RCR,         0xFFFB4104, __READ_WRITE                  );
__IO_REG32(    US1_TPR,         0xFFFB4108, __READ_WRITE                  );
__IO_REG16(    US1_TCR,         0xFFFB410C, __READ_WRITE                  );
__IO_REG32(    US1_RNPR,        0xFFFB4110, __READ_WRITE                  );
__IO_REG16(    US1_RNCR,        0xFFFB4114, __READ_WRITE                  );
__IO_REG32(    US1_TNPR,        0xFFFB4118, __READ_WRITE                  );
__IO_REG16(    US1_TNCR,        0xFFFB411C, __READ_WRITE                  );
__IO_REG32_BIT(US1_PTCR,        0xFFFB4120, __WRITE,       __pdc_ptcr_bits);
__IO_REG32_BIT(US1_PTSR,        0xFFFB4124, __READ,        __pdc_ptsr_bits);

/***************************************************************************
 **
 ** TC
 **
 ***************************************************************************/ 
__IO_REG32_BIT(TC0_CCR,      0xFFFA0000, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC0_CMR,      0xFFFA0004, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC0_CV,       0xFFFA0010, __READ                      );
__IO_REG16(    TC0_RA,       0xFFFA0014, __READ_WRITE                );
__IO_REG16(    TC0_RB,       0xFFFA0018, __READ_WRITE                );
__IO_REG16(    TC0_RC,       0xFFFA001C, __READ_WRITE                );
__IO_REG32_BIT(TC0_SR,       0xFFFA0020, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC0_IER,      0xFFFA0024, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC0_IDR,      0xFFFA0028, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC0_IMR,      0xFFFA002C, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC1_CCR,      0xFFFA0040, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC1_CMR,      0xFFFA0044, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC1_CV,       0xFFFA0050, __READ                      );
__IO_REG16(    TC1_RA,       0xFFFA0054, __READ_WRITE                );
__IO_REG16(    TC1_RB,       0xFFFA0058, __READ_WRITE                );
__IO_REG16(    TC1_RC,       0xFFFA005C, __READ_WRITE                );
__IO_REG32_BIT(TC1_SR,       0xFFFA0060, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC1_IER,      0xFFFA0064, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC1_IDR,      0xFFFA0068, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC1_IMR,      0xFFFA006C, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC2_CCR,      0xFFFA0080, __WRITE,      __tc_ccr_bits );
__IO_REG32_BIT(TC2_CMR,      0xFFFA0084, __READ_WRITE, __tc_cmr_bits );
__IO_REG16(    TC2_CV,       0xFFFA0090, __READ                      );
__IO_REG16(    TC2_RA,       0xFFFA0094, __READ_WRITE                );
__IO_REG16(    TC2_RB,       0xFFFA0098, __READ_WRITE                );
__IO_REG16(    TC2_RC,       0xFFFA009C, __READ_WRITE                );
__IO_REG32_BIT(TC2_SR,       0xFFFA00A0, __READ,       __tc_sr_bits  );
__IO_REG32_BIT(TC2_IER,      0xFFFA00A4, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC2_IDR,      0xFFFA00A8, __WRITE,      __tc_ier_bits );
__IO_REG32_BIT(TC2_IMR,      0xFFFA00AC, __READ,       __tc_ier_bits );

__IO_REG32_BIT(TC_BCR,       0xFFFA00C0, __WRITE,      __tc_bcr_bits );
__IO_REG32_BIT(TC_BMR,       0xFFFA00C4, __READ_WRITE, __tc_bmr_bits );

/***************************************************************************
 **
 ** UDP
 **
 ***************************************************************************/
__IO_REG32_BIT(UDP_FRM_NUM,  0xFFFA4000, __READ      , __udp_frm_num_bits );
__IO_REG32_BIT(UDP_GLB_STAT, 0xFFFA4004, __READ_WRITE, __udp_glb_stat_bits );
__IO_REG32_BIT(UDP_FADDR,    0xFFFA4008, __READ_WRITE, __udp_faddr_bits );
__IO_REG32_BIT(UDP_IER,      0xFFFA4010, __WRITE     , __udp_ier_bits );
__IO_REG32_BIT(UDP_IDR,      0xFFFA4014, __WRITE     , __udp_ier_bits );
__IO_REG32_BIT(UDP_IMR,      0xFFFA4018, __READ      , __udp_ier_bits );
__IO_REG32_BIT(UDP_ISR,      0xFFFA401C, __READ      , __udp_isr_bits );
__IO_REG32_BIT(UDP_ICR,      0xFFFA4020, __WRITE     , __udp_icr_bits );
__IO_REG32_BIT(UDP_RST_EP,   0xFFFA4028, __READ_WRITE, __udp_rst_ep_bits );
__IO_REG32_BIT(UDP_CSR0,     0xFFFA4030, __READ_WRITE, __udp_csrx_bits );
__IO_REG32_BIT(UDP_CSR1,     0xFFFA4034, __READ_WRITE, __udp_csrx_bits );
__IO_REG32_BIT(UDP_CSR2,     0xFFFA4038, __READ_WRITE, __udp_csrx_bits );
__IO_REG32_BIT(UDP_CSR3,     0xFFFA403C, __READ_WRITE, __udp_csrx_bits );
__IO_REG32_BIT(UDP_CSR4,     0xFFFA4040, __READ_WRITE, __udp_csrx_bits );
__IO_REG32_BIT(UDP_CSR5,     0xFFFA4044, __READ_WRITE, __udp_csrx_bits );
__IO_REG8(     UDP_FDR0,     0xFFFA4050, __READ_WRITE );
__IO_REG8(     UDP_FDR1,     0xFFFA4054, __READ_WRITE );
__IO_REG8(     UDP_FDR2,     0xFFFA4058, __READ_WRITE );
__IO_REG8(     UDP_FDR3,     0xFFFA405C, __READ_WRITE );
__IO_REG8(     UDP_FDR4,     0xFFFA4064, __READ_WRITE );
__IO_REG8(     UDP_FDR5,     0xFFFA4068, __READ_WRITE );
__IO_REG32_BIT(UDP_TXVC,     0xFFFA4074, __READ_WRITE, __udp_txvc_bits );

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CR,      0xFFFA8000, __WRITE,      __adc_cr_bits  );
__IO_REG32_BIT(ADC_MR,      0xFFFA8004, __READ_WRITE, __adc_mr_bits  );
__IO_REG32_BIT(ADC_CHER,    0xFFFA8010, __WRITE,      __adc_cher_bits);
__IO_REG32_BIT(ADC_CHDR,    0xFFFA8014, __WRITE,      __adc_cher_bits);
__IO_REG32_BIT(ADC_CHSR,    0xFFFA8018, __READ,       __adc_cher_bits);
__IO_REG32_BIT(ADC_SR,      0xFFFA801C, __READ,       __adc_sr_bits  );
__IO_REG32_BIT(ADC_LCDR,    0xFFFA8020, __READ,       __adc_lcdr_bits);
__IO_REG32_BIT(ADC_IER,     0xFFFA8024, __WRITE,      __adc_sr_bits  );
__IO_REG32_BIT(ADC_IDR,     0xFFFA8028, __WRITE,      __adc_sr_bits  );
__IO_REG32_BIT(ADC_IMR,     0xFFFA802C, __READ,       __adc_sr_bits  );
__IO_REG32_BIT(ADC_CDR0,    0xFFFA8030, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR1,    0xFFFA8034, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR2,    0xFFFA8038, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR3,    0xFFFA803C, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR4,    0xFFFA8040, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR5,    0xFFFA8044, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR6,    0xFFFA8048, __READ,       __adc_cdr_bits );
__IO_REG32_BIT(ADC_CDR7,    0xFFFA804C, __READ,       __adc_cdr_bits );

__IO_REG32(    ADC_RPR,     0xFFFA8100, __READ_WRITE                  );
__IO_REG16(    ADC_RCR,     0xFFFA8104, __READ_WRITE                  );
__IO_REG32(    ADC_RNPR,    0xFFFA8110, __READ_WRITE                  );
__IO_REG16(    ADC_RNCR,    0xFFFA8114, __READ_WRITE                  );
__IO_REG32_BIT(ADC_PTCR,    0xFFFA8120, __WRITE,      __pdc_ptcr_bits );
__IO_REG32_BIT(ADC_PTSR,    0xFFFA8124, __READ,       __pdc_ptsr_bits );

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 ** AT91CAP7 Interrupt Codes
 **
 ***************************************************************************/
#define INT_FIQ     0
#define INT_SYS     1
#define INT_PIOA    2
#define INT_PIOB    3
#define INT_US0     4
#define INT_US1     5
#define INT_SPI0    6
#define INT_TC0     7
#define INT_TC1     8
#define INT_TC2     9
#define INT_UDP    10
#define INT_ADC    11
#define INT_MPP0   12
#define INT_MPP1   13
#define INT_MPP2   14
#define INT_MPP3   15
#define INT_MPP4   16
#define INT_MPP5   17
#define INT_MPP6   18
#define INT_MPP7   19
#define INT_MPP8   20
#define INT_MPP9   21
#define INT_MPP10  22
#define INT_MPP11
#define INT_MPP12  24
#define INT_MPP13  25
#define INT_MPMA   26
#define INT_MPMB   27
#define INT_MPMC   28
#define INT_MPMD   29
#define INT_IRQ0   30
#define INT_IRQ1   31

#endif	/* __IOAT91CAP7_H */
