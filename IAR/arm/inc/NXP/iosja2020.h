/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP SJA2020
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2007
 **
 **    $Revision: 30254 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOSJA2020_H
#define __IOSJA2020_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    SJA2020 SPECIAL FUNCTION REGISTERS
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

/* Flash control register (FCTR) */
typedef struct{
__REG32 FS_CS       : 1;
__REG32 FS_WRE      : 1;
__REG32 FS_WEB      : 1;
__REG32             : 1;
__REG32 FS_DCR      : 1;
__REG32 FS_RLD      : 1;
__REG32 FS_ISS      : 1;
__REG32 FS_WPB      : 1;
__REG32             : 1;
__REG32 FS_PD       : 1;
__REG32 FS_PDL      : 1;
__REG32 FS_RLS      : 1;
__REG32 FS_PROGREQ  : 1;
__REG32 FS_CACHEBYP : 1;
__REG32 FS_CACHECLR : 1;
__REG32 FS_LOADREQ  : 1;
__REG32             :16;
} __fctr_bits;

/* Flash program time register (FPTR) */
typedef struct{
__REG32 TR          :15;
__REG32 EN_T        : 1;
__REG32             :16;
} __fptr_bits;

/* Flash bridge wait states register (FBWST) */
typedef struct{
__REG32 WST         : 8;
__REG32             : 6;
__REG32 SPECALWAYS  : 1;
__REG32 CACHE2EN    : 1;
__REG32             :16;
} __fbwst_bits;

/* Flash clock divider register (FCRA) */
typedef struct{
__REG32 FCRA        :12;
__REG32             :20;
} __fcra_bits;

/* Flash BIST control registers (FMSSTART) */
typedef struct{
__REG32 FMSSTART    :17;
__REG32             :15;
} __fmsstart_bits;

/* Flash BIST control registers (FMSSTOP) */
typedef struct{
__REG32 FMSSTOP     :17;
__REG32 MISR_START  : 1;
__REG32             :14;
} __fmsstop_bits;

/* Flash interrupt status register (INT_STATUS)
   Flash set interrupt status (INT_SET_STATUS)
   Flash clear interrupt status (INT_CLR_STATUS)
   Flash interrupt enable (INT_ENABLE)
   Flash set interrupt enable (INT_SET_ENABLE)
   Flash clear interrupt enable (INT_CLR_ENABLE)*/
typedef struct{
__REG32 END_OF_ERASE  : 1;
__REG32 END_OF_BURN   : 1;
__REG32 END_OF_MISR   : 1;
__REG32               :29;
} __fint_status_bits;

/* Bank idle cycle control registers (SMBIDCYRx) */
typedef struct{
__REG32 IDCY        : 4;
__REG32             :28;
} __smbidcyr_bits;

/* Bank wait state 1 control registers (SMBWST1Rx) */
typedef struct{
__REG32 WST1        : 5;
__REG32             :27;
} __smbwst1r_bits;

/* Bank wait state 2 control registers (SMBWST2Rx) */
typedef struct{
__REG32 WST2        : 5;
__REG32             :27;
} __smbwst2r_bits;

/* Bank output enable assertion delay control register (SMBWSTOENRx) */
typedef struct{
__REG32 WSTOEN      : 4;
__REG32             :28;
} __smbwstoenr_bits;

/* Bank write enable assertion delay control register (SMBWSTWENRx) */
typedef struct{
__REG32 WSTWEN      : 4;
__REG32             :28;
} __smbwstwenr_bits;

/* Bank configuration register (SMBCRx) */
typedef struct{
__REG32 RBLE        : 1;
__REG32             : 2;
__REG32 CSPOL       : 1;
__REG32 WP          : 1;
__REG32 BM          : 1;
__REG32 MW          : 2;
__REG32             :24;
} __smbcr_bits;

/* Bank status register (SMBSR) */
typedef struct{
__REG32               : 1;
__REG32 WRITEPROTERR  : 1;
__REG32               :30;
} __smbsr_bits;

/* Clock switch configuration register (CSC) */
typedef struct{
__REG32 ENF           : 2;
__REG32               :30;
} __csc_bits;

/* Clock frequency select registers (CFSx) */
typedef struct{
__REG32 FS            : 2;
__REG32               :30;
} __cfs_bits;

/* Clock switch status register (CSS) */
typedef struct{
__REG32 FS_SELECT     : 2;
__REG32 FSS           : 2;
__REG32               :28;
} __css_bits;

/* Clock power control registers (CPC0) */
typedef struct{
__REG32               : 1;
__REG32 WAKE          : 2;
__REG32               :29;
} __cpc0_bits;

/* Clock power control registers (CPC1) */
typedef struct{
__REG32 RUN           : 1;
__REG32 WAKE          : 2;
__REG32               :29;
} __cpc1_bits;

/* Clock power control registers (CPC2) */
typedef struct{
__REG32               : 1;
__REG32 WAKE          : 2;
__REG32               :29;
} __cpc2_bits;

/* Clock power control registers (CPC3) */
typedef struct{
__REG32 RUN           : 1;
__REG32 WAKE          : 2;
__REG32               :29;
} __cpc3_bits;

/* Clock power control registers (CPC4) */
typedef struct{
__REG32 RUN           : 1;
__REG32 WAKE          : 2;
__REG32               :29;
} __cpc4_bits;

/* Clock power status registers (CPSx) */
typedef struct{
__REG32 ACTIVE        : 1;
__REG32 WAKEUP        : 1;
__REG32               :30;
} __cps_bits;

/* Fractional clock enable register (CFCE4) */
typedef struct{
__REG32 FCE           : 1;
__REG32               :31;
} __cfce4_bits;

/* Fractional clock divider register (CFD) */
typedef struct{
__REG32 EN            : 1;
__REG32 RESET         : 1;
__REG32 STRETCH       : 1;
__REG32 MADD          :14;
__REG32 MSUB          :14;
__REG32               : 1;
} __cfd_bits;

/* Power mode register (CPM) */
typedef struct{
__REG32 PM            : 2;
__REG32               :30;
} __cpm_bits;

/* Watchdog bark register (CWDB) */
typedef struct{
__REG32 WDB           : 1;
__REG32               :31;
} __cwdb_bits;

/* Real time clock oscillator power mode register (CRTCOPM) */
typedef struct{
__REG32 RTCOPM        : 1;
__REG32               :31;
} __crtcopm_bits;

/* Oscillator power mode register (COPM) */
typedef struct{
__REG32 OPM           : 1;
__REG32               :31;
} __copm_bits;

/* Oscillator lock status register (COLS) */
typedef struct{
__REG32 OLS           : 1;
__REG32               :31;
} __cols_bits;

/* PLL clock source select register (CPCSS) */
typedef struct{
__REG32 PCSS          : 1;
__REG32               :31;
} __cpcss_bits;

/* PLL Power-down mode register (CPPDM) */
typedef struct{
__REG32 PPDM          : 1;
__REG32               :31;
} __cppdm_bits;

/* PLL lock status register (CPLS) */
typedef struct{
__REG32 PLS           : 1;
__REG32               :31;
} __cpls_bits;

/* PLL multiplication ratio register (CPMR) */
typedef struct{
__REG32 PMR           : 3;
__REG32               :29;
} __cpmr_bits;

/* PLL post divider register (CPPD) */
typedef struct{
__REG32 PPD           : 2;
__REG32               :30;
} __cppd_bits;

/* Ring oscillator power mode register (CRPM) */
typedef struct{
__REG32 RPM           : 1;
__REG32               :31;
} __crpm_bits;

/* Ring oscillator post divider register (CRPD) */
typedef struct{
__REG32 RPD           : 5;
__REG32               :27;
} __crpd_bits;

/* Ring oscillator frequency select register (CRFS) */
typedef struct{
__REG32 RFS           : 4;
__REG32               :28;
} __crfs_bits;

/* Shadow memory mapping register (SSMM) */
typedef struct{
__REG32               :10;
__REG32 SMMSA         :22;
} __ssmm_bits;

/* Port function select registers (SFSAPx) */
typedef struct{
__REG32 PFSP0         : 2;
__REG32 PFSP1         : 2;
__REG32 PFSP2         : 2;
__REG32 PFSP3         : 2;
__REG32 PFSP4         : 2;
__REG32 PFSP5         : 2;
__REG32 PFSP6         : 2;
__REG32 PFSP7         : 2;
__REG32 PFSP8         : 2;
__REG32 PFSP9         : 2;
__REG32 PFSP10        : 2;
__REG32 PFSP11        : 2;
__REG32 PFSP12        : 2;
__REG32 PFSP13        : 2;
__REG32 PFSP14        : 2;
__REG32 PFSP15        : 2;
} __sfsap_bits;

/* Port function select registers (SFSBPx) */
typedef struct{
__REG32 PFSP16        : 2;
__REG32 PFSP17        : 2;
__REG32 PFSP18        : 2;
__REG32 PFSP19        : 2;
__REG32 PFSP20        : 2;
__REG32 PFSP21        : 2;
__REG32 PFSP22        : 2;
__REG32 PFSP23        : 2;
__REG32 PFSP24        : 2;
__REG32 PFSP25        : 2;
__REG32 PFSP26        : 2;
__REG32 PFSP27        : 2;
__REG32 PFSP28        : 2;
__REG32 PFSP29        : 2;
__REG32 PFSP30        : 2;
__REG32 PFSP31        : 2;
} __sfsbp_bits;

/* Pull-up control registers (SPUCPx) */
typedef struct{
__REG32 PUC0          : 1;
__REG32 PUC1          : 1;
__REG32 PUC2          : 1;
__REG32 PUC3          : 1;
__REG32 PUC4          : 1;
__REG32 PUC5          : 1;
__REG32 PUC6          : 1;
__REG32 PUC7          : 1;
__REG32 PUC8          : 1;
__REG32 PUC9          : 1;
__REG32 PUC10         : 1;
__REG32 PUC11         : 1;
__REG32 PUC12         : 1;
__REG32 PUC13         : 1;
__REG32 PUC14         : 1;
__REG32 PUC15         : 1;
__REG32 PUC16         : 1;
__REG32 PUC17         : 1;
__REG32 PUC18         : 1;
__REG32 PUC19         : 1;
__REG32 PUC20         : 1;
__REG32 PUC21         : 1;
__REG32 PUC22         : 1;
__REG32 PUC23         : 1;
__REG32 PUC24         : 1;
__REG32 PUC25         : 1;
__REG32 PUC26         : 1;
__REG32 PUC27         : 1;
__REG32 PUC28         : 1;
__REG32 PUC29         : 1;
__REG32 PUC30         : 1;
__REG32 PUC31         : 1;
} __spucp_bits;

/* SPI control register 0 (SSPCR0) */
typedef struct{
__REG32 DSS           : 4;
__REG32 FRF           : 2;
__REG32 SPO           : 1;
__REG32 SPH           : 1;
__REG32 SCR           : 8;
__REG32               :16;
} __sspcr0_bits;

/* SPI control register 1 (SSPCR1) */
typedef struct{
__REG32 LBM           : 1;
__REG32 SSE           : 1;
__REG32 MS            : 1;
__REG32 SOD           : 1;
__REG32               :28;
} __sspcr1_bits;

/* SPI FIFO data register (SSPDR) */
typedef struct{
__REG32 DATA          :16;
__REG32               :16;
} __sspdr_bits;

/* SPI status register (SSPSR) */
typedef struct{
__REG32 TFE           : 1;
__REG32 TNF           : 1;
__REG32 RNE           : 1;
__REG32 RFF           : 1;
__REG32 BSY           : 1;
__REG32               :27;
} __sspsr_bits;

/* SPI clock prescale register (SSPCPSR) */
typedef struct{
__REG32 SPSDVSR       : 8;
__REG32               :24;
} __sspcpsr_bits;

/* SPI interrupt enable register (SSPIMSC) */
typedef struct{
__REG32 RORIM         : 1;
__REG32 RTIM          : 1;
__REG32 RXIM          : 1;
__REG32 TXIM          : 1;
__REG32               :28;
} __sspimsc_bits;

/* SPI raw interrupt status register (SSPRIS) */
typedef struct{
__REG32 RORRIS        : 1;
__REG32 RTRIS         : 1;
__REG32 RXRIS         : 1;
__REG32 TXRIS         : 1;
__REG32               :28;
} __sspris_bits;

/* SPI raw interrupt status register (SSPRIS) */
typedef struct{
__REG32 RORMIS        : 1;
__REG32 RTMIS         : 1;
__REG32 RXMIS         : 1;
__REG32 TXMIS         : 1;
__REG32               :28;
} __sspmis_bits;

/* SPI interrupt clear register (SSPICR) */
typedef struct{
__REG32 RORIC         : 1;
__REG32 RTIC          : 1;
__REG32               :30;
} __sspicr_bits;

/* Watchdog mode register (WDMOD) */
typedef struct{
__REG32 WDDB          : 1;
__REG32 WDDBLCK       : 1;
__REG32 WDCEF         : 1;
__REG32 WDTOF         : 1;
__REG32               :28;
} __wdmod_bits;

/* Watchdog trigger register (WDTRIG) */
typedef struct{
__REG32 KICKDOG       : 1;
__REG32               :31;
} __wdtrig_bits;

/* Watchdog interrupt set status (WDISS) */
typedef struct{
__REG32 INT_SET_STATUS : 1;
__REG32                :31;
} __wdiss_bits;

/* Watchdog interrupt clear status (WDICS) */
typedef struct{
__REG32 INT_CLR_STATUS : 1;
__REG32                :31;
} __wdics_bits;

/* Watchdog interrupt enable (WDIE)
   Watchdog interrupt status register (WDIS) */
typedef struct{
__REG32 OVERFLOW      : 1;
__REG32               :31;
} __wdie_bits;

/* Watchdog interrupt set enable (WDISE) */
typedef struct{
__REG32 SET_ENABLE    : 1;
__REG32               :31;
} __wdise_bits;

/* Watchdog interrupt clear enable (WDICE) */
typedef struct{
__REG32 CLR_ENABLE    : 1;
__REG32               :31;
} __wdice_bits;

/* ADC channel conversion data registers (ACDx) */
typedef struct{
__REG32 ACD           :10;
__REG32               :22;
} __acd_bits;

/* ADC control register (ACON) */
typedef struct{
__REG32               : 1;
__REG32 AEN           : 1;
__REG32 ASM           : 1;
__REG32 ASC           : 3;
__REG32 AS            : 1;
__REG32               :25;
} __acon_bits;

/* ADC channel configuration register (ACC) */
typedef struct{
__REG32 ACC0          : 4;
__REG32 ACC1          : 4;
__REG32 ACC2          : 4;
__REG32 ACC3          : 4;
__REG32 ACC4          : 4;
__REG32 ACC5          : 4;
__REG32 ACC6          : 4;
__REG32 ACC7          : 4;
} __acc_bits;

/* ADC interrupt enable register (AIE) */
typedef struct{
__REG32 ASIE          : 1;
__REG32               :31;
} __aie_bits;

/* ADC interrupt status register (AIS) */
typedef struct{
__REG32 ASI           : 1;
__REG32               :31;
} __ais_bits;

/* ADC interrupt clear register (AIC) */
typedef struct{
__REG32 ASIC          : 1;
__REG32               :31;
} __aic_bits;

/* Event status register (PEND) */
typedef struct{
__REG32 PEND0         : 1;
__REG32 PEND1         : 1;
__REG32 PEND2         : 1;
__REG32 PEND3         : 1;
__REG32 PEND4         : 1;
__REG32 PEND5         : 1;
__REG32 PEND6         : 1;
__REG32 PEND7         : 1;
__REG32 PEND8         : 1;
__REG32 PEND9         : 1;
__REG32 PEND10        : 1;
__REG32 PEND11        : 1;
__REG32 PEND12        : 1;
__REG32 PEND13        : 1;
__REG32 PEND14        : 1;
__REG32 PEND15        : 1;
__REG32 PEND16        : 1;
__REG32 PEND17        : 1;
__REG32               :14;
} __pend_bits;

/* Event status clear register (INT_CLR) */
typedef struct{
__REG32 INT_CLR0      : 1;
__REG32 INT_CLR1      : 1;
__REG32 INT_CLR2      : 1;
__REG32 INT_CLR3      : 1;
__REG32 INT_CLR4      : 1;
__REG32 INT_CLR5      : 1;
__REG32 INT_CLR6      : 1;
__REG32 INT_CLR7      : 1;
__REG32 INT_CLR8      : 1;
__REG32 INT_CLR9      : 1;
__REG32 INT_CLR10     : 1;
__REG32 INT_CLR11     : 1;
__REG32 INT_CLR12     : 1;
__REG32 INT_CLR13     : 1;
__REG32 INT_CLR14     : 1;
__REG32 INT_CLR15     : 1;
__REG32 INT_CLR16     : 1;
__REG32 INT_CLR17     : 1;
__REG32               :14;
} __int_clr_bits;

/* Event status set register (INT_SET) */
typedef struct{
__REG32 INT_SET0      : 1;
__REG32 INT_SET1      : 1;
__REG32 INT_SET2      : 1;
__REG32 INT_SET3      : 1;
__REG32 INT_SET4      : 1;
__REG32 INT_SET5      : 1;
__REG32 INT_SET6      : 1;
__REG32 INT_SET7      : 1;
__REG32 INT_SET8      : 1;
__REG32 INT_SET9      : 1;
__REG32 INT_SET10     : 1;
__REG32 INT_SET11     : 1;
__REG32 INT_SET12     : 1;
__REG32 INT_SET13     : 1;
__REG32 INT_SET14     : 1;
__REG32 INT_SET15     : 1;
__REG32 INT_SET16     : 1;
__REG32 INT_SET17     : 1;
__REG32               :14;
} __int_set_bits;

/* Event enable register (MASK) */
typedef struct{
__REG32 MASK0         : 1;
__REG32 MASK1         : 1;
__REG32 MASK2         : 1;
__REG32 MASK3         : 1;
__REG32 MASK4         : 1;
__REG32 MASK5         : 1;
__REG32 MASK6         : 1;
__REG32 MASK7         : 1;
__REG32 MASK8         : 1;
__REG32 MASK9         : 1;
__REG32 MASK10        : 1;
__REG32 MASK11        : 1;
__REG32 MASK12        : 1;
__REG32 MASK13        : 1;
__REG32 MASK14        : 1;
__REG32 MASK15        : 1;
__REG32 MASK16        : 1;
__REG32 MASK17        : 1;
__REG32               :14;
} __mask_bits;

/* Event enable clear register (MASK_CLR) */
typedef struct{
__REG32 MASK_CLR0     : 1;
__REG32 MASK_CLR1     : 1;
__REG32 MASK_CLR2     : 1;
__REG32 MASK_CLR3     : 1;
__REG32 MASK_CLR4     : 1;
__REG32 MASK_CLR5     : 1;
__REG32 MASK_CLR6     : 1;
__REG32 MASK_CLR7     : 1;
__REG32 MASK_CLR8     : 1;
__REG32 MASK_CLR9     : 1;
__REG32 MASK_CLR10    : 1;
__REG32 MASK_CLR11    : 1;
__REG32 MASK_CLR12    : 1;
__REG32 MASK_CLR13    : 1;
__REG32 MASK_CLR14    : 1;
__REG32 MASK_CLR15    : 1;
__REG32 MASK_CLR16    : 1;
__REG32 MASK_CLR17    : 1;
__REG32               :14;
} __mask_clr_bits;

/* Event enable set register (MASK_SET) */
typedef struct{
__REG32 MASK_SET0     : 1;
__REG32 MASK_SET1     : 1;
__REG32 MASK_SET2     : 1;
__REG32 MASK_SET3     : 1;
__REG32 MASK_SET4     : 1;
__REG32 MASK_SET5     : 1;
__REG32 MASK_SET6     : 1;
__REG32 MASK_SET7     : 1;
__REG32 MASK_SET8     : 1;
__REG32 MASK_SET9     : 1;
__REG32 MASK_SET10    : 1;
__REG32 MASK_SET11    : 1;
__REG32 MASK_SET12    : 1;
__REG32 MASK_SET13    : 1;
__REG32 MASK_SET14    : 1;
__REG32 MASK_SET15    : 1;
__REG32 MASK_SET16    : 1;
__REG32 MASK_SET17    : 1;
__REG32               :14;
} __mask_set_bits;

/* Activation polarity register (APR) */
typedef struct{
__REG32 APR0          : 1;
__REG32 APR1          : 1;
__REG32 APR2          : 1;
__REG32 APR3          : 1;
__REG32 APR4          : 1;
__REG32 APR5          : 1;
__REG32 APR6          : 1;
__REG32 APR7          : 1;
__REG32 APR8          : 1;
__REG32 APR9          : 1;
__REG32 APR10         : 1;
__REG32 APR11         : 1;
__REG32 APR12         : 1;
__REG32 APR13         : 1;
__REG32 APR14         : 1;
__REG32 APR15         : 1;
__REG32 APR16         : 1;
__REG32 APR17         : 1;
__REG32               :14;
} __apr_bits;

/* Activation type register (ATR) */
typedef struct{
__REG32 ATR0          : 1;
__REG32 ATR1          : 1;
__REG32 ATR2          : 1;
__REG32 ATR3          : 1;
__REG32 ATR4          : 1;
__REG32 ATR5          : 1;
__REG32 ATR6          : 1;
__REG32 ATR7          : 1;
__REG32 ATR8          : 1;
__REG32 ATR9          : 1;
__REG32 ATR10         : 1;
__REG32 ATR11         : 1;
__REG32 ATR12         : 1;
__REG32 ATR13         : 1;
__REG32 ATR14         : 1;
__REG32 ATR15         : 1;
__REG32 ATR16         : 1;
__REG32 ATR17         : 1;
__REG32               :14;
} __atr_bits;

/* Raw status register (RSR) */
typedef struct{
__REG32 RSR0          : 1;
__REG32 RSR1          : 1;
__REG32 RSR2          : 1;
__REG32 RSR3          : 1;
__REG32 RSR4          : 1;
__REG32 RSR5          : 1;
__REG32 RSR6          : 1;
__REG32 RSR7          : 1;
__REG32 RSR8          : 1;
__REG32 RSR9          : 1;
__REG32 RSR10         : 1;
__REG32 RSR11         : 1;
__REG32 RSR12         : 1;
__REG32 RSR13         : 1;
__REG32 RSR14         : 1;
__REG32 RSR15         : 1;
__REG32 RSR16         : 1;
__REG32 RSR17         : 1;
__REG32               :14;
} __rsr_bits;

/* RTC seconds fraction register (RTC_TIME_FRACTION) */
typedef struct{
__REG32 RTC_TIME_FRACTION :15;
__REG32                   :17;
} __rtc_time_fraction_bits;

/* RTC seconds fraction register (RTC_TIME_FRACTION) */
typedef struct{
__REG32                 :31;
__REG32 RTC_TICK_ENABLE : 1;
} __rtc_control_bits;

/* Timer interrupt register (IR) */
typedef struct{
__REG32 INTR_M0 : 1;
__REG32 INTR_M1 : 1;
__REG32 INTR_M2 : 1;
__REG32 INTR_M3 : 1;
__REG32 INTR_C0 : 1;
__REG32 INTR_C1 : 1;
__REG32 INTR_C2 : 1;
__REG32 INTR_C3 : 1;
__REG32         :24;
} __ir_bits;

/* Timer control register (TCR) */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tcr_bits;

/* Match control register (MCR) */
typedef struct{
__REG32 INTR_0   : 1;
__REG32 RESET_0  : 1;
__REG32 STOP_0   : 1;
__REG32 INTR_1   : 1;
__REG32 RESET_1  : 1;
__REG32 STOP_1   : 1;
__REG32 INTR_2   : 1;
__REG32 RESET_2  : 1;
__REG32 STOP_2   : 1;
__REG32 INTR_3   : 1;
__REG32 RESET_3  : 1;
__REG32 STOP_3   : 1;
__REG32          :20;
} __mcr_bits;

/* Capture control register (CCR) */
typedef struct{
__REG32 RISE_0   : 1;
__REG32 FALL_0   : 1;
__REG32 EVENT_0  : 1;
__REG32 RISE_1   : 1;
__REG32 FALL_1   : 1;
__REG32 EVENT_1  : 1;
__REG32 RISE_2   : 1;
__REG32 FALL_2   : 1;
__REG32 EVENT_2  : 1;
__REG32 RISE_3   : 1;
__REG32 FALL_3   : 1;
__REG32 EVENT_3  : 1;
__REG32          :20;
} __ccr_bits;

/* External match register (EMR) */
typedef struct{
__REG32 EMR_0   : 1;
__REG32 EMR_1   : 1;
__REG32 EMR_2   : 1;
__REG32 EMR_3   : 1;
__REG32 CTRL_0  : 2;
__REG32 CTRL_1  : 2;
__REG32 CTRL_2  : 2;
__REG32 CTRL_3  : 2;
__REG32         :20;
} __emr_bits;

/* Interrupt enable register (IER) */
typedef struct{
__REG8  RBIE      : 1;
__REG8  TBEIE     : 1;
__REG8  LSIE      : 1;
__REG8            : 5;
} __uartier_bits;

/* Line status register (LSR) */
typedef struct{
__REG8  DR    : 1;
__REG8  OE    : 1;
__REG8  PE    : 1;
__REG8  FE    : 1;
__REG8  BI    : 1;
__REG8  THRE  : 1;
__REG8  TEMT  : 1;
__REG8  RXFE  : 1;
} __uartlsr_bits;

/* Line control register (LCR) */
typedef struct{
__REG8  WLS   : 2;
__REG8  SBS   : 1;
__REG8  PEN   : 1;
__REG8  PS    : 2;
__REG8  BC    : 1;
__REG8  DLAB  : 1;
} __uartlcr_bits;

/* Interrupt ID register (IIR)
   FIFO control register (FCR) */
typedef union {
  /*UxIIR*/
  struct {
__REG8  INT_ID    : 4;
__REG8            : 2;
__REG8  FIFO_EN   : 2;
  };
  /*UxFCR*/
  struct {
__REG8  _FIFO_EN  : 1;
__REG8  RX_FIFO_R : 1;
__REG8  TX_FIFO_R : 1;
__REG8  DMA_M     : 1;
__REG8            : 2;
__REG8  REV_TRIG  : 2;
  };
} __uartfcriir_bits;

/* Port input register (IO0_PINS) */
typedef struct{
__REG32 IO0_PIN0  : 1;
__REG32 IO0_PIN1  : 1;
__REG32 IO0_PIN2  : 1;
__REG32 IO0_PIN3  : 1;
__REG32 IO0_PIN4  : 1;
__REG32 IO0_PIN5  : 1;
__REG32 IO0_PIN6  : 1;
__REG32 IO0_PIN7  : 1;
__REG32 IO0_PIN8  : 1;
__REG32 IO0_PIN9  : 1;
__REG32 IO0_PIN10 : 1;
__REG32 IO0_PIN11 : 1;
__REG32 IO0_PIN12 : 1;
__REG32 IO0_PIN13 : 1;
__REG32 IO0_PIN14 : 1;
__REG32 IO0_PIN15 : 1;
__REG32 IO0_PIN16 : 1;
__REG32 IO0_PIN17 : 1;
__REG32 IO0_PIN18 : 1;
__REG32 IO0_PIN19 : 1;
__REG32 IO0_PIN20 : 1;
__REG32 IO0_PIN21 : 1;
__REG32 IO0_PIN22 : 1;
__REG32 IO0_PIN23 : 1;
__REG32 IO0_PIN24 : 1;
__REG32 IO0_PIN25 : 1;
__REG32 IO0_PIN26 : 1;
__REG32 IO0_PIN27 : 1;
__REG32 IO0_PIN28 : 1;
__REG32 IO0_PIN29 : 1;
__REG32 IO0_PIN30 : 1;
__REG32 IO0_PIN31 : 1;
} __io0_pins_bits;

/* Port output register (IO0_OR) */
typedef struct{
__REG32 IO0_OR0  : 1;
__REG32 IO0_OR1  : 1;
__REG32 IO0_OR2  : 1;
__REG32 IO0_OR3  : 1;
__REG32 IO0_OR4  : 1;
__REG32 IO0_OR5  : 1;
__REG32 IO0_OR6  : 1;
__REG32 IO0_OR7  : 1;
__REG32 IO0_OR8  : 1;
__REG32 IO0_OR9  : 1;
__REG32 IO0_OR10 : 1;
__REG32 IO0_OR11 : 1;
__REG32 IO0_OR12 : 1;
__REG32 IO0_OR13 : 1;
__REG32 IO0_OR14 : 1;
__REG32 IO0_OR15 : 1;
__REG32 IO0_OR16 : 1;
__REG32 IO0_OR17 : 1;
__REG32 IO0_OR18 : 1;
__REG32 IO0_OR19 : 1;
__REG32 IO0_OR20 : 1;
__REG32 IO0_OR21 : 1;
__REG32 IO0_OR22 : 1;
__REG32 IO0_OR23 : 1;
__REG32 IO0_OR24 : 1;
__REG32 IO0_OR25 : 1;
__REG32 IO0_OR26 : 1;
__REG32 IO0_OR27 : 1;
__REG32 IO0_OR28 : 1;
__REG32 IO0_OR29 : 1;
__REG32 IO0_OR30 : 1;
__REG32 IO0_OR31 : 1;
} __io0_or_bits;

/* Port direction register (IO0_DR) */
typedef struct{
__REG32 IO0_DR0  : 1;
__REG32 IO0_DR1  : 1;
__REG32 IO0_DR2  : 1;
__REG32 IO0_DR3  : 1;
__REG32 IO0_DR4  : 1;
__REG32 IO0_DR5  : 1;
__REG32 IO0_DR6  : 1;
__REG32 IO0_DR7  : 1;
__REG32 IO0_DR8  : 1;
__REG32 IO0_DR9  : 1;
__REG32 IO0_DR10 : 1;
__REG32 IO0_DR11 : 1;
__REG32 IO0_DR12 : 1;
__REG32 IO0_DR13 : 1;
__REG32 IO0_DR14 : 1;
__REG32 IO0_DR15 : 1;
__REG32 IO0_DR16 : 1;
__REG32 IO0_DR17 : 1;
__REG32 IO0_DR18 : 1;
__REG32 IO0_DR19 : 1;
__REG32 IO0_DR20 : 1;
__REG32 IO0_DR21 : 1;
__REG32 IO0_DR22 : 1;
__REG32 IO0_DR23 : 1;
__REG32 IO0_DR24 : 1;
__REG32 IO0_DR25 : 1;
__REG32 IO0_DR26 : 1;
__REG32 IO0_DR27 : 1;
__REG32 IO0_DR28 : 1;
__REG32 IO0_DR29 : 1;
__REG32 IO0_DR30 : 1;
__REG32 IO0_DR31 : 1;
} __io0_dr_bits;

/* Port input register (IO1_PINS) */
typedef struct{
__REG32 IO1_PIN0  : 1;
__REG32 IO1_PIN1  : 1;
__REG32 IO1_PIN2  : 1;
__REG32 IO1_PIN3  : 1;
__REG32 IO1_PIN4  : 1;
__REG32 IO1_PIN5  : 1;
__REG32 IO1_PIN6  : 1;
__REG32 IO1_PIN7  : 1;
__REG32 IO1_PIN8  : 1;
__REG32 IO1_PIN9  : 1;
__REG32 IO1_PIN10 : 1;
__REG32 IO1_PIN11 : 1;
__REG32 IO1_PIN12 : 1;
__REG32 IO1_PIN13 : 1;
__REG32 IO1_PIN14 : 1;
__REG32 IO1_PIN15 : 1;
__REG32 IO1_PIN16 : 1;
__REG32 IO1_PIN17 : 1;
__REG32 IO1_PIN18 : 1;
__REG32 IO1_PIN19 : 1;
__REG32 IO1_PIN20 : 1;
__REG32 IO1_PIN21 : 1;
__REG32 IO1_PIN22 : 1;
__REG32 IO1_PIN23 : 1;
__REG32 IO1_PIN24 : 1;
__REG32 IO1_PIN25 : 1;
__REG32 IO1_PIN26 : 1;
__REG32 IO1_PIN27 : 1;
__REG32 IO1_PIN28 : 1;
__REG32 IO1_PIN29 : 1;
__REG32 IO1_PIN30 : 1;
__REG32 IO1_PIN31 : 1;
} __io1_pins_bits;

/* Port output register (IO1_OR) */
typedef struct{
__REG32 IO1_OR0  : 1;
__REG32 IO1_OR1  : 1;
__REG32 IO1_OR2  : 1;
__REG32 IO1_OR3  : 1;
__REG32 IO1_OR4  : 1;
__REG32 IO1_OR5  : 1;
__REG32 IO1_OR6  : 1;
__REG32 IO1_OR7  : 1;
__REG32 IO1_OR8  : 1;
__REG32 IO1_OR9  : 1;
__REG32 IO1_OR10 : 1;
__REG32 IO1_OR11 : 1;
__REG32 IO1_OR12 : 1;
__REG32 IO1_OR13 : 1;
__REG32 IO1_OR14 : 1;
__REG32 IO1_OR15 : 1;
__REG32 IO1_OR16 : 1;
__REG32 IO1_OR17 : 1;
__REG32 IO1_OR18 : 1;
__REG32 IO1_OR19 : 1;
__REG32 IO1_OR20 : 1;
__REG32 IO1_OR21 : 1;
__REG32 IO1_OR22 : 1;
__REG32 IO1_OR23 : 1;
__REG32 IO1_OR24 : 1;
__REG32 IO1_OR25 : 1;
__REG32 IO1_OR26 : 1;
__REG32 IO1_OR27 : 1;
__REG32 IO1_OR28 : 1;
__REG32 IO1_OR29 : 1;
__REG32 IO1_OR30 : 1;
__REG32 IO1_OR31 : 1;
} __io1_or_bits;

/* Port direction register (IO1_DR) */
typedef struct{
__REG32 IO1_DR0  : 1;
__REG32 IO1_DR1  : 1;
__REG32 IO1_DR2  : 1;
__REG32 IO1_DR3  : 1;
__REG32 IO1_DR4  : 1;
__REG32 IO1_DR5  : 1;
__REG32 IO1_DR6  : 1;
__REG32 IO1_DR7  : 1;
__REG32 IO1_DR8  : 1;
__REG32 IO1_DR9  : 1;
__REG32 IO1_DR10 : 1;
__REG32 IO1_DR11 : 1;
__REG32 IO1_DR12 : 1;
__REG32 IO1_DR13 : 1;
__REG32 IO1_DR14 : 1;
__REG32 IO1_DR15 : 1;
__REG32 IO1_DR16 : 1;
__REG32 IO1_DR17 : 1;
__REG32 IO1_DR18 : 1;
__REG32 IO1_DR19 : 1;
__REG32 IO1_DR20 : 1;
__REG32 IO1_DR21 : 1;
__REG32 IO1_DR22 : 1;
__REG32 IO1_DR23 : 1;
__REG32 IO1_DR24 : 1;
__REG32 IO1_DR25 : 1;
__REG32 IO1_DR26 : 1;
__REG32 IO1_DR27 : 1;
__REG32 IO1_DR28 : 1;
__REG32 IO1_DR29 : 1;
__REG32 IO1_DR30 : 1;
__REG32 IO1_DR31 : 1;
} __io1_dr_bits;

/* Port input register (IO2_PINS) */
typedef struct{
__REG32 IO2_PIN0  : 1;
__REG32 IO2_PIN1  : 1;
__REG32 IO2_PIN2  : 1;
__REG32 IO2_PIN3  : 1;
__REG32 IO2_PIN4  : 1;
__REG32 IO2_PIN5  : 1;
__REG32 IO2_PIN6  : 1;
__REG32 IO2_PIN7  : 1;
__REG32 IO2_PIN8  : 1;
__REG32 IO2_PIN9  : 1;
__REG32 IO2_PIN10 : 1;
__REG32 IO2_PIN11 : 1;
__REG32 IO2_PIN12 : 1;
__REG32 IO2_PIN13 : 1;
__REG32 IO2_PIN14 : 1;
__REG32 IO2_PIN15 : 1;
__REG32 IO2_PIN16 : 1;
__REG32 IO2_PIN17 : 1;
__REG32 IO2_PIN18 : 1;
__REG32 IO2_PIN19 : 1;
__REG32 IO2_PIN20 : 1;
__REG32 IO2_PIN21 : 1;
__REG32 IO2_PIN22 : 1;
__REG32 IO2_PIN23 : 1;
__REG32 IO2_PIN24 : 1;
__REG32 IO2_PIN25 : 1;
__REG32 IO2_PIN26 : 1;
__REG32 IO2_PIN27 : 1;
__REG32 IO2_PIN28 : 1;
__REG32 IO2_PIN29 : 1;
__REG32 IO2_PIN30 : 1;
__REG32 IO2_PIN31 : 1;
} __io2_pins_bits;

/* Port output register (IO2_OR) */
typedef struct{
__REG32 IO2_OR0  : 1;
__REG32 IO2_OR1  : 1;
__REG32 IO2_OR2  : 1;
__REG32 IO2_OR3  : 1;
__REG32 IO2_OR4  : 1;
__REG32 IO2_OR5  : 1;
__REG32 IO2_OR6  : 1;
__REG32 IO2_OR7  : 1;
__REG32 IO2_OR8  : 1;
__REG32 IO2_OR9  : 1;
__REG32 IO2_OR10 : 1;
__REG32 IO2_OR11 : 1;
__REG32 IO2_OR12 : 1;
__REG32 IO2_OR13 : 1;
__REG32 IO2_OR14 : 1;
__REG32 IO2_OR15 : 1;
__REG32 IO2_OR16 : 1;
__REG32 IO2_OR17 : 1;
__REG32 IO2_OR18 : 1;
__REG32 IO2_OR19 : 1;
__REG32 IO2_OR20 : 1;
__REG32 IO2_OR21 : 1;
__REG32 IO2_OR22 : 1;
__REG32 IO2_OR23 : 1;
__REG32 IO2_OR24 : 1;
__REG32 IO2_OR25 : 1;
__REG32 IO2_OR26 : 1;
__REG32 IO2_OR27 : 1;
__REG32 IO2_OR28 : 1;
__REG32 IO2_OR29 : 1;
__REG32 IO2_OR30 : 1;
__REG32 IO2_OR31 : 1;
} __io2_or_bits;

/* Port direction register (IO2_DR) */
typedef struct{
__REG32 IO2_DR0  : 1;
__REG32 IO2_DR1  : 1;
__REG32 IO2_DR2  : 1;
__REG32 IO2_DR3  : 1;
__REG32 IO2_DR4  : 1;
__REG32 IO2_DR5  : 1;
__REG32 IO2_DR6  : 1;
__REG32 IO2_DR7  : 1;
__REG32 IO2_DR8  : 1;
__REG32 IO2_DR9  : 1;
__REG32 IO2_DR10 : 1;
__REG32 IO2_DR11 : 1;
__REG32 IO2_DR12 : 1;
__REG32 IO2_DR13 : 1;
__REG32 IO2_DR14 : 1;
__REG32 IO2_DR15 : 1;
__REG32 IO2_DR16 : 1;
__REG32 IO2_DR17 : 1;
__REG32 IO2_DR18 : 1;
__REG32 IO2_DR19 : 1;
__REG32 IO2_DR20 : 1;
__REG32 IO2_DR21 : 1;
__REG32 IO2_DR22 : 1;
__REG32 IO2_DR23 : 1;
__REG32 IO2_DR24 : 1;
__REG32 IO2_DR25 : 1;
__REG32 IO2_DR26 : 1;
__REG32 IO2_DR27 : 1;
__REG32 IO2_DR28 : 1;
__REG32 IO2_DR29 : 1;
__REG32 IO2_DR30 : 1;
__REG32 IO2_DR31 : 1;
} __io2_dr_bits;

/* CAN controller mode register (CCMODE) */
typedef struct {
  __REG32 RM              : 1;
  __REG32 LOM             : 1;
  __REG32 STM             : 1;
  __REG32 TPM             : 1;
  __REG32                 : 1;
  __REG32 RPM             : 1;
  __REG32                 :26;
} __ccmode_bits;

/* CAN controller command register (CCCMD) */
typedef struct {
  __REG32 TR              : 1;
  __REG32 AT              : 1;
  __REG32 RRB             : 1;
  __REG32 CDO             : 1;
  __REG32 SRR             : 1;
  __REG32 STB1            : 1;
  __REG32 STB2            : 1;
  __REG32 STB3            : 1;
  __REG32                 :24;
} __cccmd_bits;

/* CAN controller global status register (CCGS) */
typedef struct {
  __REG32 RBS              : 1;
  __REG32 DOS              : 1;
  __REG32 TBS              : 1;
  __REG32 TCS              : 1;
  __REG32 RS               : 1;
  __REG32 TS               : 1;
  __REG32 ES               : 1;
  __REG32 BS               : 1;
  __REG32                  : 8;
  __REG32 RXERR            : 8;
  __REG32 TXERR            : 8;
} __ccgs_bits;

/* CAN controller interrupt and capture register (CCIC) */
typedef struct {
  __REG32 RI               : 1;
  __REG32 TI1              : 1;
  __REG32 EWI              : 1;
  __REG32 DOI              : 1;
  __REG32                  : 1;
  __REG32 EPI              : 1;
  __REG32 ALI              : 1;
  __REG32 BEI              : 1;
  __REG32 IDI              : 1;
  __REG32 TI2              : 1;
  __REG32 TI3              : 1;
  __REG32                  : 5;
  __REG32 ERRCC            : 5;
  __REG32 ERRDIR           : 1;
  __REG32 ERRT             : 2;
  __REG32 ALCBIT           : 5;
  __REG32                  : 3;
} __ccic_bits;

/* CAN controller interrupt enable register (CCIE) */
typedef struct {
  __REG32 RIE               : 1;
  __REG32 TIE1              : 1;
  __REG32 EWIE              : 1;
  __REG32 DOIE              : 1;
  __REG32                   : 1;
  __REG32 EPIE              : 1;
  __REG32 ALIE              : 1;
  __REG32 BEIE              : 1;
  __REG32 IDIE              : 1;
  __REG32 TIE2              : 1;
  __REG32 TIE3              : 1;
  __REG32                   :21;
} __ccie_bits;

/* CAN controller bus timing register (CCBT) */
typedef struct {
  __REG32 BRP                :10;
  __REG32                    : 4;
  __REG32 SJW                : 2;
  __REG32 TSEG1              : 4;
  __REG32 TSEG2              : 3;
  __REG32 SAM                : 1;
  __REG32                    : 8;
} __ccbt_bits;

/* CAN controller error warning limit register (CCEWL) */
typedef struct {
  __REG32 EWL                : 8;
  __REG32                    :24;
} __ccewl_bits;

/* CAN controller status register (CCSTAT) */
typedef struct {
  __REG32 RBS                : 1;
  __REG32 DOS                : 1;
  __REG32 TBS1               : 1;
  __REG32 TCS1               : 1;
  __REG32 RS                 : 1;
  __REG32 TS1                : 1;
  __REG32 ES                 : 1;
  __REG32 BS                 : 1;
  __REG32 /*RBS*/            : 1;
  __REG32 /*DOS*/            : 1;
  __REG32 TBS2               : 1;
  __REG32 TCS2               : 1;
  __REG32 /*RS*/             : 1;
  __REG32 TS2                : 1;
  __REG32 /*ES*/             : 1;
  __REG32 /*BS*/             : 1;
  __REG32 /*RBS*/            : 1;
  __REG32 /*DOS*/            : 1;
  __REG32 TBS3               : 1;
  __REG32 TCS3               : 1;
  __REG32 /*RS*/             : 1;
  __REG32 TS3                : 1;
  __REG32 /*ES*/             : 1;
  __REG32 /*BS*/             : 1;
  __REG32                    : 8;
} __ccstat_bits;

/* CAN controller receive buffer message info register (CCRXBMI) */
typedef struct {
  __REG32 IDI                :10;
  __REG32 BP                 : 1;
  __REG32                    : 5;
  __REG32 DLC                : 4;
  __REG32                    :10;
  __REG32 RTR                : 1;
  __REG32 FF                 : 1;
} __ccrxbmi_bits;

/* CAN controller receive buffer identifier register (CCRXBID) */
typedef struct {
 __REG32 ID                 :29;
 __REG32                    : 3;
} __ccrxbid_bits;

/* CAN controller receive buffer data A register (CCRXBDA) */
typedef struct {
  __REG32 DB1                 : 8;
  __REG32 DB2                 : 8;
  __REG32 DB3                 : 8;
  __REG32 DB4                 : 8;
} __ccrxbda_bits;

/* CAN controller receive buffer data B register (CCRXBDB) */
typedef struct {
  __REG32 DB5                 : 8;
  __REG32 DB6                 : 8;
  __REG32 DB7                 : 8;
  __REG32 DB8                 : 8;
} __ccrxbdb_bits;

/* CAN controller transmit buffer message info register (CCTXBxMI) */
typedef struct {
  __REG32 TXPRIO            : 8;
  __REG32                   : 8;
  __REG32 DLC               : 4;
  __REG32                   :10;
  __REG32 RTR               : 1;
  __REG32 FF                : 1;
} __cctxbmi_bits;

/* CAN controller transmit buffer identifier register (CCTXBxID) */
typedef struct {
__REG32 ID2                :29;
__REG32                    : 3;
} __cctxbid_bits;

/* CAN controller transmit buffer data A register (CCTXBxDA) */
typedef struct {
  __REG32 DB1                 :8;
  __REG32 DB2                 :8;
  __REG32 DB3                 :8;
  __REG32 DB4                 :8;
} __cctxbda_bits;

/* CAN controller transmit buffer data B register (CCTXBxDB) */
typedef struct {
  __REG32 DB5                 :8;
  __REG32 DB6                 :8;
  __REG32 DB7                 :8;
  __REG32 DB8                 :8;
} __cctxbdb_bits;

/* CAN acceptance filter mode register (CAMODE) */
typedef struct {
  __REG32 ACCOFF          : 1;
  __REG32 ACCBP           : 1;
  __REG32 EFCAN           : 1;
  __REG32                 :29;
} __camode_bits;

/* CAN acceptance filter standard frame explicit start address register (CASFESA) */
typedef struct {
  __REG32                     : 2;
  __REG32 SFESA               :10;
  __REG32                     :20;
} __casfesa_bits;

/* CAN acceptance filter standard frame group start address register (CASFGSA) */
typedef struct {
  __REG32                     : 2;
  __REG32 SFGSA               :10;
  __REG32                     :20;
} __casfgsa_bits;

/* CAN acceptance filter extended frame explicit start address register (CAEFESA) */
typedef struct {
  __REG32                     : 2;
  __REG32 EFESA               :10;
  __REG32                     :20;
} __caefesa_bits;

/* CAN acceptance filter extended frame group start address register (CAEFGSA) */
typedef struct {
  __REG32                     : 2;
  __REG32 EFGSA               :10;
  __REG32                     :20;
} __caefgsa_bits;

/* CAN acceptance filter end of look up table address register (CAEOTA) */
typedef struct {
  __REG32                     : 2;
  __REG32 EOTA                :10;
  __REG32                     :20;
} __caeota_bits;

/* CAN acceptance filter look-up table error address register (CALUTEA) */
typedef struct {
  __REG32                     : 2;
  __REG32 LUTEA               : 9;
  __REG32                     :21;
} __calutea_bits;

/* CAN acceptance filter look-up table error address register (CALUTEA) */
typedef struct {
  __REG32 LUTE                : 1;
  __REG32                     :31;
} __calute_bits;

/* CAN controllers central transmit status register (CCCTS) */
typedef struct {
  __REG32 TS0             : 1;
  __REG32 TS1             : 1;
  __REG32 TS2             : 1;
  __REG32 TS3             : 1;
  __REG32 TS4             : 1;
  __REG32 TS5             : 1;
  __REG32                 : 2;
  __REG32 TBS0            : 1;
  __REG32 TBS1            : 1;
  __REG32 TBS2            : 1;
  __REG32 TBS3            : 1;
  __REG32 TBS4            : 1;
  __REG32 TBS5            : 1;
  __REG32                 : 2;
  __REG32 TCS0            : 1;
  __REG32 TCS1            : 1;
  __REG32 TCS2            : 1;
  __REG32 TCS3            : 1;
  __REG32 TCS4            : 1;
  __REG32 TCS5            : 1;
  __REG32                 :10;
} __cccts_bits;

/* CAN controllers central receive status register (CCCRS) */
typedef struct {
  __REG32 RS0             : 1;
  __REG32 RS1             : 1;
  __REG32 RS2             : 1;
  __REG32 RS3             : 1;
  __REG32 RS4             : 1;
  __REG32 RS5             : 1;
  __REG32                 : 2;
  __REG32 RBS0            : 1;
  __REG32 RBS1            : 1;
  __REG32 RBS2            : 1;
  __REG32 RBS3            : 1;
  __REG32 RBS4            : 1;
  __REG32 RBS5            : 1;
  __REG32                 : 2;
  __REG32 DOS0            : 1;
  __REG32 DOS1            : 1;
  __REG32 DOS2            : 1;
  __REG32 DOS3            : 1;
  __REG32 DOS4            : 1;
  __REG32 DOS5            : 1;
  __REG32                 :10;
} __cccrs_bits;

/* CAN controllers central miscellaneous status register (CCCMS) */
typedef struct {
  __REG32 ES0             : 1;
  __REG32 ES1             : 1;
  __REG32 ES2             : 1;
  __REG32 ES3             : 1;
  __REG32 ES4             : 1;
  __REG32 ES5             : 1;
  __REG32                 : 2;
  __REG32 BS0             : 1;
  __REG32 BS1             : 1;
  __REG32 BS2             : 1;
  __REG32 BS3             : 1;
  __REG32 BS4             : 1;
  __REG32 BS5             : 1;
  __REG32                 :18;
} __cccms_bits;

/* LIN master controller mode register (LMODE) */
typedef struct {
  __REG32 LRM             : 1;
  __REG32                 : 6;
  __REG32 LM              : 1;
  __REG32                 :24;
} __lmode_bits;

/* LIN master controller configuration register (LCFG) */
typedef struct {
  __REG32 SBL             : 3;
  __REG32 IBS             : 2;
  __REG32                 : 1;
  __REG32 SWCS            : 1;
  __REG32 SWPA            : 1;
  __REG32                 :24;
} __lcfg_bits;

/* LIN master controller command register (LCMD) */
typedef struct {
  __REG32 TR              : 1;
  __REG32                 : 6;
  __REG32 SSB             : 1;
  __REG32                 :24;
} __lcmd_bits;

/* LIN master controller fractional baud rate generator register (LFBRG) */
typedef struct {
  __REG32 INT             :16;
  __REG32 FRAC            : 4;
  __REG32                 :12;
} __lfbrg_bits;

/* LIN master controller status register (LSTAT)
   Receive buffer register (RBR)
   Transmit holding register (THR) */
typedef union {
/* LxSTAT*/
  struct {
  __REG32 MR              : 1;
  __REG32 MBA             : 1;
  __REG32 HS              : 1;
  __REG32 RS              : 1;
  __REG32 TS              : 1;
  __REG32 ES              : 1;
  __REG32 IS              : 1;
  __REG32                 : 1;
  __REG32 RLL             : 1;
  __REG32 TTL             : 1;
  __REG32                 :22;
  };
/* LxRBR*/
  struct {
  __REG32 RBR             : 8;
  __REG32                 :24;
  };
/* LxTHR*/
  struct {
  __REG32 THR             : 8;
  __REG32                 :24;
  };
} __lstat_bits;

/* LIN master controller interrupt and capture register (LIC)
   Interrupt enable register (IER) */
typedef union {
/* LxIC*/
  struct {
  __REG32 RI              : 1;
  __REG32 TI              : 1;
  __REG32 BEI             : 1;
  __REG32 CSI             : 1;
  __REG32 NRI             : 1;
  __REG32 RTLCEI          : 1;
  __REG32 WPI             : 1;
  __REG32                 : 1;
  __REG32 EC              : 4;
  __REG32                 :20;
  };
/* LxIER*/
  struct {
  __REG32 RBIE            : 1;
  __REG32 TBEIE           : 1;
  __REG32 LSIE            : 1;
  __REG32                 :29;
  };
} __lic_bits;

/* LIN master controller interrupt enable register (LIE)
   Interrupt enable register (IER) */
typedef union {
/* LxIE*/
  struct {
  __REG32 RIE             : 1;
  __REG32 TIE             : 1;
  __REG32 BEIE            : 1;
  __REG32 CSIE            : 1;
  __REG32 NRIE            : 1;
  __REG32 RTLCEIE         : 1;
  __REG32 WPIE            : 1;
  __REG32                 :25;
  };
/* LxIIR*/
  struct {
  __REG32 INT_ID          : 3;
  __REG32                 :29;
  };
} __lie_bits;

/* Line control register (LCR) */
typedef struct {
__REG32 WLS             : 2;
__REG32 STB             : 1;
__REG32 PEN             : 1;
__REG32 PS              : 2;
__REG32 BC              : 1;
__REG32                 :25;
} __lcr_bits;

/* Line control register (LCR) */
typedef struct {
  __REG32 CS              : 8;
  __REG32                 :24;
} __lcs_bits;

/* LIN master controller time-out register (LTO)
   Line status register (LSR) */
typedef union {
/* LxTO*/
  struct {
  __REG32 TO              : 8;
  __REG32                 :24;
  };
/* LxLSR*/
  struct {
  __REG32 DR              : 1;
  __REG32 OE              : 1;
  __REG32 PE              : 1;
  __REG32 FE              : 1;
  __REG32 BI              : 1;
  __REG32 THRE            : 1;
  __REG32 TEMT            : 1;
  __REG32                 :25;
  };
} __lto_bits;

/* LIN master controller message buffer registers (LID)*/
typedef struct {
__REG32 ID              : 6;
__REG32 P0              : 1;
__REG32 P1              : 1;
__REG32                 : 8;
__REG32 DLC             : 5;
__REG32                 : 3;
__REG32 DD              : 1;
__REG32 CSID            : 1;
__REG32                 : 6;
} __lid_bits;

/* LIN master controller message buffer registers (LDATA)
   Scratch register (SCR) */
typedef union {
/* LxDATA*/
  struct {
  __REG32 DF1             : 8;
  __REG32 DF2             : 8;
  __REG32 DF3             : 8;
  __REG32 DF4             : 8;
  };
/* LxSCR*/
  struct {
  __REG32 SCR             : 8;
  __REG32                 :24;
  };
} __ldata_bits;

/* LIN master controller message buffer registers (LDATB)*/
typedef struct {
__REG32 DF5             : 8;
__REG32 DF6             : 8;
__REG32 DF7             : 8;
__REG32 DF8             : 8;
} __ldatb_bits;

/* LIN master controller message buffer registers (LDATC)*/
typedef struct {
__REG32 DF9             : 8;
__REG32 DF10            : 8;
__REG32 DF11            : 8;
__REG32 DF12            : 8;
} __ldatc_bits;

/* LIN master controller message buffer registers (LDATD)*/
typedef struct {
__REG32 DF13            : 8;
__REG32 DF14            : 8;
__REG32 DF15            : 8;
__REG32 DF16            : 8;
} __ldatd_bits;

/* Interrupt priority mask register (INT_PRIORITYMASK)*/
typedef struct {
__REG32 PRIORITY_LIMITER  : 4;
__REG32                   :28;
} __int_priority_mask_bits;

/* Interrupt vector register (INT_VECTOR)*/
typedef struct {
__REG32                   : 3;
__REG32 INDEX             : 5;
__REG32                   : 3;
__REG32 TABLE_ADDR        :21;
} __int_vector_bits;

/* Interrupt pending register (INT_PENDING_1_31) */
typedef struct {
__REG32                   : 1;
__REG32 PENDING1          : 1;
__REG32 PENDING2          : 1;
__REG32 PENDING3          : 1;
__REG32 PENDING4          : 1;
__REG32 PENDING5          : 1;
__REG32 PENDING6          : 1;
__REG32 PENDING7          : 1;
__REG32 PENDING8          : 1;
__REG32 PENDING9          : 1;
__REG32 PENDING10         : 1;
__REG32 PENDING11         : 1;
__REG32 PENDING12         : 1;
__REG32 PENDING13         : 1;
__REG32 PENDING14         : 1;
__REG32 PENDING15         : 1;
__REG32 PENDING16         : 1;
__REG32 PENDING17         : 1;
__REG32 PENDING18         : 1;
__REG32 PENDING19         : 1;
__REG32 PENDING20         : 1;
__REG32 PENDING21         : 1;
__REG32 PENDING22         : 1;
__REG32 PENDING23         : 1;
__REG32 PENDING24         : 1;
__REG32 PENDING25         : 1;
__REG32 PENDING26         : 1;
__REG32 PENDING27         : 1;
__REG32 PENDING28         : 1;
__REG32 PENDING29         : 1;
__REG32 PENDING30         : 1;
__REG32 PENDING31         : 1;
} __int_pending_1_31_bits;

/* Interrupt controller features register (INT_FEATURES) */
typedef struct {
__REG32 N                 : 8;
__REG32 P                 : 8;
__REG32 T                 : 6;
__REG32                   :10;
} __int_features_bits;

/* Interrupt request register (INT_REQUEST) */
typedef struct {
__REG32 PRIORITY_LEVEL    : 4;
__REG32                   : 4;
__REG32 TARGET            : 1;
__REG32                   : 7;
__REG32 ENABLE            : 1;
__REG32 ACTIVE_LOW        : 1;
__REG32                   : 7;
__REG32 WE_ACTIVE_LOW     : 1;
__REG32 WE_ENABLE         : 1;
__REG32 WE_TARGET         : 1;
__REG32 WE_PRIORITY_LEVEL : 1;
__REG32 CLR_SWINT         : 1;
__REG32 SET_SWINT         : 1;
__REG32 PENDING           : 1;
} __int_request_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** Flash controller
 **
 ***************************************************************************/
__IO_REG32_BIT(FCTR,            0x20200000,__READ_WRITE ,__fctr_bits);
__IO_REG32_BIT(FPTR,            0x20200008,__READ_WRITE ,__fptr_bits);
__IO_REG32_BIT(FBWST,           0x20200010,__READ_WRITE ,__fbwst_bits);
__IO_REG32_BIT(FCRA,            0x2020001C,__READ_WRITE ,__fcra_bits);
__IO_REG32_BIT(FMSSTART,        0x20200020,__READ_WRITE ,__fmsstart_bits);
__IO_REG32_BIT(FMSSTOP,         0x20200024,__READ_WRITE ,__fmsstop_bits);
__IO_REG32(    FMSW0,           0x2020002C,__READ       );
__IO_REG32(    FMSW1,           0x20200030,__READ       );
__IO_REG32(    FMSW2,           0x20200034,__READ       );
__IO_REG32(    FMSW3,           0x20200038,__READ       );
__IO_REG32_BIT(FINT_CLR_ENABLE, 0x20200FD8,__WRITE      ,__fint_status_bits);
__IO_REG32_BIT(FINT_SET_ENABLE, 0x20200FDC,__WRITE      ,__fint_status_bits);
__IO_REG32_BIT(FINT_STATUS,     0x20200FE0,__READ       ,__fint_status_bits);
__IO_REG32_BIT(FINT_ENABLE,     0x20200FE4,__READ       ,__fint_status_bits);
__IO_REG32_BIT(FINT_CLR_STATUS, 0x20200FE8,__WRITE      ,__fint_status_bits);
__IO_REG32_BIT(FINT_SET_STATUS, 0x20200FEC,__WRITE      ,__fint_status_bits);

/***************************************************************************
 **
 ** SMC
 **
 ***************************************************************************/
__IO_REG32_BIT(SMBIDCYR0,       0xBFFFF000,__READ_WRITE ,__smbidcyr_bits);
__IO_REG32_BIT(SMBWST1R0,       0xBFFFF004,__READ_WRITE ,__smbwst1r_bits);
__IO_REG32_BIT(SMBWST2R0,       0xBFFFF008,__READ_WRITE ,__smbwst2r_bits);
__IO_REG32_BIT(SMBWSTOENR0,     0xBFFFF00C,__READ_WRITE ,__smbwstoenr_bits);
__IO_REG32_BIT(SMBWSTWENR0,     0xBFFFF010,__READ_WRITE ,__smbwstwenr_bits);
__IO_REG32_BIT(SMBCR0,          0xBFFFF014,__READ_WRITE ,__smbcr_bits);
__IO_REG32_BIT(SMBSR0,          0xBFFFF018,__READ_WRITE ,__smbsr_bits);
__IO_REG32_BIT(SMBIDCYR1,       0xBFFFF01C,__READ_WRITE ,__smbidcyr_bits);
__IO_REG32_BIT(SMBWST1R1,       0xBFFFF020,__READ_WRITE ,__smbwst1r_bits);
__IO_REG32_BIT(SMBWST2R1,       0xBFFFF024,__READ_WRITE ,__smbwst2r_bits);
__IO_REG32_BIT(SMBWSTOENR1,     0xBFFFF028,__READ_WRITE ,__smbwstoenr_bits);
__IO_REG32_BIT(SMBWSTWENR1,     0xBFFFF02C,__READ_WRITE ,__smbwstwenr_bits);
__IO_REG32_BIT(SMBCR1,          0xBFFFF030,__READ_WRITE ,__smbcr_bits);
__IO_REG32_BIT(SMBSR1,          0xBFFFF034,__READ_WRITE ,__smbsr_bits);
__IO_REG32_BIT(SMBIDCYR2,       0xBFFFF038,__READ_WRITE ,__smbidcyr_bits);
__IO_REG32_BIT(SMBWST1R2,       0xBFFFF03C,__READ_WRITE ,__smbwst1r_bits);
__IO_REG32_BIT(SMBWST2R2,       0xBFFFF040,__READ_WRITE ,__smbwst2r_bits);
__IO_REG32_BIT(SMBWSTOENR2,     0xBFFFF044,__READ_WRITE ,__smbwstoenr_bits);
__IO_REG32_BIT(SMBWSTWENR2,     0xBFFFF048,__READ_WRITE ,__smbwstwenr_bits);
__IO_REG32_BIT(SMBCR2,          0xBFFFF04C,__READ_WRITE ,__smbcr_bits);
__IO_REG32_BIT(SMBSR2,          0xBFFFF050,__READ_WRITE ,__smbsr_bits);
__IO_REG32_BIT(SMBIDCYR3,       0xBFFFF054,__READ_WRITE ,__smbidcyr_bits);
__IO_REG32_BIT(SMBWST1R3,       0xBFFFF058,__READ_WRITE ,__smbwst1r_bits);
__IO_REG32_BIT(SMBWST2R3,       0xBFFFF05C,__READ_WRITE ,__smbwst2r_bits);
__IO_REG32_BIT(SMBWSTOENR3,     0xBFFFF060,__READ_WRITE ,__smbwstoenr_bits);
__IO_REG32_BIT(SMBWSTWENR3,     0xBFFFF064,__READ_WRITE ,__smbwstwenr_bits);
__IO_REG32_BIT(SMBCR3,          0xBFFFF068,__READ_WRITE ,__smbcr_bits);
__IO_REG32_BIT(SMBSR3,          0xBFFFF06C,__READ_WRITE ,__smbsr_bits);

/***************************************************************************
 **
 ** CGU
 **
 ***************************************************************************/
__IO_REG32_BIT(CSC,             0xE0000000,__READ_WRITE ,__csc_bits);
__IO_REG32_BIT(CFS1,            0xE0000004,__READ_WRITE ,__cfs_bits);
__IO_REG32_BIT(CFS2,            0xE0000008,__READ_WRITE ,__cfs_bits);
__IO_REG32_BIT(CSS,             0xE000000C,__READ       ,__css_bits);
__IO_REG32_BIT(CPC0,            0xE0000010,__READ_WRITE ,__cpc0_bits);
__IO_REG32_BIT(CPC1,            0xE0000014,__READ_WRITE ,__cpc1_bits);
__IO_REG32_BIT(CPC2,            0xE0000018,__READ_WRITE ,__cpc2_bits);
__IO_REG32_BIT(CPC3,            0xE000001C,__READ_WRITE ,__cpc3_bits);
__IO_REG32_BIT(CPC4,            0xE0000020,__READ_WRITE ,__cpc4_bits);
__IO_REG32_BIT(CPS0,            0xE0000028,__READ       ,__cps_bits);
__IO_REG32_BIT(CPS1,            0xE000002C,__READ       ,__cps_bits);
__IO_REG32_BIT(CPS2,            0xE0000030,__READ       ,__cps_bits);
__IO_REG32_BIT(CPS3,            0xE0000034,__READ       ,__cps_bits);
__IO_REG32_BIT(CPS4,            0xE0000038,__READ       ,__cps_bits);
__IO_REG32_BIT(CFCE4,           0xE0000050,__READ_WRITE ,__cfce4_bits);
__IO_REG32_BIT(CFD,             0xE0000058,__READ_WRITE ,__cfd_bits);
__IO_REG32_BIT(CPM,             0xE0000C00,__READ_WRITE ,__cpm_bits);
__IO_REG32_BIT(CWDB,            0xE0000C04,__READ       ,__cwdb_bits);
__IO_REG32_BIT(CRTCOPM,         0xE0000C08,__READ_WRITE ,__crtcopm_bits);
__IO_REG32_BIT(COPM,            0xE0000C10,__READ_WRITE ,__copm_bits);
__IO_REG32_BIT(COLS,            0xE0000C18,__READ       ,__cols_bits);
__IO_REG32_BIT(CPCSS,           0xE0000C40,__READ_WRITE ,__cpcss_bits);
__IO_REG32_BIT(CPPDM,           0xE0000C44,__READ_WRITE ,__cppdm_bits);
__IO_REG32_BIT(CPLS,            0xE0000C4C,__READ       ,__cpls_bits);
__IO_REG32_BIT(CPMR,            0xE0000C54,__READ_WRITE ,__cpmr_bits);
__IO_REG32_BIT(CPPD,            0xE0000C58,__READ_WRITE ,__cppd_bits);
__IO_REG32_BIT(CRPM,            0xE0000C5C,__READ_WRITE ,__crpm_bits);
__IO_REG32_BIT(CRPD,            0xE0000C60,__READ_WRITE ,__crpd_bits);
__IO_REG32_BIT(CRFS,            0xE0000C64,__READ_WRITE ,__crfs_bits);

/***************************************************************************
 **
 ** SCU
 **
 ***************************************************************************/
__IO_REG32_BIT(SSMM,            0xE0001000,__READ_WRITE ,__ssmm_bits);
__IO_REG32_BIT(SFSAP0,          0xE0001004,__READ_WRITE ,__sfsap_bits);
__IO_REG32_BIT(SFSBP0,          0xE0001008,__READ_WRITE ,__sfsbp_bits);
__IO_REG32_BIT(SFSAP1,          0xE000100C,__READ_WRITE ,__sfsap_bits);
__IO_REG32_BIT(SFSBP1,          0xE0001010,__READ_WRITE ,__sfsbp_bits);
__IO_REG32_BIT(SFSAP2,          0xE0001014,__READ_WRITE ,__sfsap_bits);
__IO_REG32_BIT(SFSBP2,          0xE0001018,__READ_WRITE ,__sfsbp_bits);
__IO_REG32_BIT(SPUCP0,          0xE000101C,__READ_WRITE ,__spucp_bits);
__IO_REG32_BIT(SPUCP1,          0xE0001020,__READ_WRITE ,__spucp_bits);
__IO_REG32_BIT(SPUCP2,          0xE0001024,__READ_WRITE ,__spucp_bits);

/***************************************************************************
 **
 ** SPI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,         0xE0002000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,         0xE0002004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,          0xE0002008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,          0xE000200C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,        0xE0002010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,        0xE0002014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,         0xE0002018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,         0xE000201C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,         0xE0002020,__WRITE      ,__sspicr_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,         0xE0003000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,         0xE0003004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,          0xE0003008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,          0xE000300C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,        0xE0003010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,        0xE0003014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,         0xE0003018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,         0xE000301C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,         0xE0003020,__WRITE      ,__sspicr_bits);

/***************************************************************************
 **
 ** SPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP2CR0,         0xE0004000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP2CR1,         0xE0004004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP2DR,          0xE0004008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP2SR,          0xE000400C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP2CPSR,        0xE0004010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP2IMSC,        0xE0004014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP2RIS,         0xE0004018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP2MIS,         0xE000401C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP2ICR,         0xE0004020,__WRITE      ,__sspicr_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,           0xE0006000,__READ_WRITE ,__wdmod_bits);
__IO_REG32(    WDRV,            0xE0006004,__READ_WRITE );
__IO_REG32(    WDCV,            0xE0006008,__READ_WRITE );
__IO_REG32_BIT(WDTRIG,          0xE000600C,__WRITE      ,__wdtrig_bits);
__IO_REG32_BIT(WDISS,           0xE0006010,__WRITE      ,__wdiss_bits);
__IO_REG32_BIT(WDICS,           0xE0006014,__WRITE      ,__wdics_bits);
__IO_REG32_BIT(WDIE,            0xE0006018,__READ       ,__wdie_bits);
__IO_REG32_BIT(WDIS,            0xE000601C,__READ       ,__wdie_bits);
__IO_REG32_BIT(WDISE,           0xE0006020,__WRITE      ,__wdise_bits);
__IO_REG32_BIT(WDICE,           0xE0006024,__WRITE      ,__wdice_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ACD0,            0xE0005000,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD1,            0xE0005004,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD2,            0xE0005008,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD3,            0xE000500C,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD4,            0xE0005010,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD5,            0xE0005014,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD6,            0xE0005018,__READ       ,__acd_bits);
__IO_REG32_BIT(ACD7,            0xE000501C,__READ       ,__acd_bits);
__IO_REG32_BIT(ACON,            0xE0005020,__READ_WRITE ,__acon_bits);
__IO_REG32_BIT(ACC,             0xE0005024,__READ_WRITE ,__acc_bits);
__IO_REG32_BIT(AIE,             0xE0005028,__READ_WRITE ,__aie_bits);
__IO_REG32_BIT(AIS,             0xE000502C,__READ       ,__ais_bits);
__IO_REG32_BIT(AIC,             0xE0005030,__WRITE      ,__aic_bits);

/***************************************************************************
 **
 ** Event Router
 **
 ***************************************************************************/
__IO_REG32_BIT(PEND,            0xE0008C00,__READ       ,__pend_bits);
__IO_REG32_BIT(INT_CLR,         0xE0008C20,__WRITE      ,__int_clr_bits);
__IO_REG32_BIT(INT_SET,         0xE0008C40,__WRITE      ,__int_set_bits);
__IO_REG32_BIT(MASK,            0xE0008C60,__READ       ,__mask_bits);
__IO_REG32_BIT(MASK_CLR,        0xE0008C80,__WRITE      ,__mask_clr_bits);
__IO_REG32_BIT(MASK_SET,        0xE0008CA0,__WRITE      ,__mask_set_bits);
__IO_REG32_BIT(APR,             0xE0008CC0,__READ_WRITE ,__apr_bits);
__IO_REG32_BIT(ATR,             0xE0008CE0,__READ_WRITE ,__atr_bits);
__IO_REG32_BIT(RSR,             0xE0008D20,__READ_WRITE ,__rsr_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTC_TIME_SECONDS,0xE000A000,__READ       );
__IO_REG32_BIT(RTC_TIME_FRACTION,0xE000A010,__READ      ,__rtc_time_fraction_bits);
__IO_REG32(    RTC_PORTIME,     0xE000A020,__READ_WRITE );
__IO_REG32_BIT(RTC_CONTROL,     0xE000AFC0,__READ_WRITE ,__rtc_control_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(T0IR,            0xE0040000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T0TCR,           0xE0040004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T0TC,            0xE0040008,__READ_WRITE );
__IO_REG32(    T0PR,            0xE004000C,__READ_WRITE );
__IO_REG32(    T0PC,            0xE0040010,__READ_WRITE );
__IO_REG32_BIT(T0MCR,           0xE0040014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T0MR0,           0xE0040018,__READ_WRITE );
__IO_REG32(    T0MR1,           0xE004001C,__READ_WRITE );
__IO_REG32(    T0MR2,           0xE0040020,__READ_WRITE );
__IO_REG32(    T0MR3,           0xE0040024,__READ_WRITE );
__IO_REG32_BIT(T0CCR,           0xE0040028,__READ_WRITE ,__ccr_bits);
__IO_REG32(    T0CR0,           0xE004002C,__READ       );
__IO_REG32(    T0CR1,           0xE0040030,__READ       );
__IO_REG32(    T0CR2,           0xE0040034,__READ       );
__IO_REG32(    T0CR3,           0xE0040038,__READ       );
__IO_REG32_BIT(T0EMR,           0xE004003C,__READ_WRITE ,__emr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(T1IR,            0xE0041000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T1TCR,           0xE0041004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T1TC,            0xE0041008,__READ_WRITE );
__IO_REG32(    T1PR,            0xE004100C,__READ_WRITE );
__IO_REG32(    T1PC,            0xE0041010,__READ_WRITE );
__IO_REG32_BIT(T1MCR,           0xE0041014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T1MR0,           0xE0041018,__READ_WRITE );
__IO_REG32(    T1MR1,           0xE004101C,__READ_WRITE );
__IO_REG32(    T1MR2,           0xE0041020,__READ_WRITE );
__IO_REG32(    T1MR3,           0xE0041024,__READ_WRITE );
__IO_REG32_BIT(T1CCR,           0xE0041028,__READ_WRITE ,__ccr_bits);
__IO_REG32(    T1CR0,           0xE004102C,__READ       );
__IO_REG32(    T1CR1,           0xE0041030,__READ       );
__IO_REG32(    T1CR2,           0xE0041034,__READ       );
__IO_REG32(    T1CR3,           0xE0041038,__READ       );
__IO_REG32_BIT(T1EMR,           0xE004103C,__READ_WRITE ,__emr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(T2IR,            0xE0042000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T2TCR,           0xE0042004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T2TC,            0xE0042008,__READ_WRITE );
__IO_REG32(    T2PR,            0xE004200C,__READ_WRITE );
__IO_REG32(    T2PC,            0xE0042010,__READ_WRITE );
__IO_REG32_BIT(T2MCR,           0xE0042014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T2MR0,           0xE0042018,__READ_WRITE );
__IO_REG32(    T2MR1,           0xE004201C,__READ_WRITE );
__IO_REG32(    T2MR2,           0xE0042020,__READ_WRITE );
__IO_REG32(    T2MR3,           0xE0042024,__READ_WRITE );
__IO_REG32_BIT(T2CCR,           0xE0042028,__READ_WRITE ,__ccr_bits);
__IO_REG32(    T2CR0,           0xE004202C,__READ       );
__IO_REG32(    T2CR1,           0xE0042030,__READ       );
__IO_REG32(    T2CR2,           0xE0042034,__READ       );
__IO_REG32(    T2CR3,           0xE0042038,__READ       );
__IO_REG32_BIT(T2EMR,           0xE004203C,__READ_WRITE ,__emr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(T3IR,            0xE0043000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T3TCR,           0xE0043004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T3TC,            0xE0043008,__READ_WRITE );
__IO_REG32(    T3PR,            0xE004300C,__READ_WRITE );
__IO_REG32(    T3PC,            0xE0043010,__READ_WRITE );
__IO_REG32_BIT(T3MCR,           0xE0043014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T3MR0,           0xE0043018,__READ_WRITE );
__IO_REG32(    T3MR1,           0xE004301C,__READ_WRITE );
__IO_REG32(    T3MR2,           0xE0043020,__READ_WRITE );
__IO_REG32(    T3MR3,           0xE0043024,__READ_WRITE );
__IO_REG32_BIT(T3CCR,           0xE0043028,__READ_WRITE ,__ccr_bits);
__IO_REG32(    T3CR0,           0xE004302C,__READ       );
__IO_REG32(    T3CR1,           0xE0043030,__READ       );
__IO_REG32(    T3CR2,           0xE0043034,__READ       );
__IO_REG32(    T3CR3,           0xE0043038,__READ       );
__IO_REG32_BIT(T3EMR,           0xE004303C,__READ_WRITE ,__emr_bits);

/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG8(     U0RBRTHR,        0xE0044000,__READ_WRITE );
#define U0DLL U0RBRTHR
#define U0RBR U0RBRTHR
#define U0THR U0RBRTHR

/* U0DLM and U0IER share the same address */
__IO_REG8_BIT( U0IER,           0xE0044004,__READ_WRITE ,__uartier_bits);
#define U0DLM U0IER

/* U0FCR and U0IIR share the same address */
__IO_REG8_BIT( U0FCR,           0xE0044008,__READ_WRITE ,__uartfcriir_bits);
#define U0IIR      U0FCR
#define U0IIR_bit  U0FCR_bit

__IO_REG8_BIT( U0LCR,           0xE004400C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT( U0LSR,           0xE0044014,__READ       ,__uartlsr_bits);
__IO_REG8(     U0SCR,           0xE004401C,__READ_WRITE );

/***************************************************************************
 **
 ** GPIO0
 **
 ***************************************************************************/
__IO_REG32_BIT(IO0_PINS,        0xE0045000,__READ       ,__io0_pins_bits);
__IO_REG32_BIT(IO0_OR,          0xE0045004,__READ_WRITE ,__io0_or_bits);
__IO_REG32_BIT(IO0_DR,          0xE0045008,__READ_WRITE ,__io0_dr_bits);

/***************************************************************************
 **
 ** GPIO1
 **
 ***************************************************************************/
__IO_REG32_BIT(IO1_PINS,        0xE0046000,__READ       ,__io1_pins_bits);
__IO_REG32_BIT(IO1_OR,          0xE0046004,__READ_WRITE ,__io1_or_bits);
__IO_REG32_BIT(IO1_DR,          0xE0046008,__READ_WRITE ,__io1_dr_bits);

/***************************************************************************
 **
 ** GPIO2
 **
 ***************************************************************************/
__IO_REG32_BIT(IO2_PINS,        0xE0047000,__READ       ,__io2_pins_bits);
__IO_REG32_BIT(IO2_OR,          0xE0047004,__READ_WRITE ,__io2_or_bits);
__IO_REG32_BIT(IO2_DR,          0xE0047008,__READ_WRITE ,__io2_dr_bits);

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(C0CMODE,         0xE0080000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C0CCMD,          0xE0080004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C0CGS,           0xE0080008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C0CIC,           0xE008000C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C0CIE,           0xE0080010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C0CBT,           0xE0080014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C0CEWL,          0xE0080018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C0CSTAT,         0xE008001C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C0CRXBMI,        0xE0080020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C0CRXBID,        0xE0080024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C0CRXBDA,        0xE0080028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C0CRXBDB,        0xE008002C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C0CTXB1MI,       0xE0080030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C0CTXB1ID,       0xE0080034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C0CTXB1DA,       0xE0080038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C0CTXB1DB,       0xE008003C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C0CTXB2MI,       0xE0080040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C0CTXB2ID,       0xE0080044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C0CTXB2DA,       0xE0080048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C0CTXB2DB,       0xE008004C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C0CTXB3MI,       0xE0080050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C0CTXB3ID,       0xE0080054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C0CTXB3DA,       0xE0080058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C0CTXB3DB,       0xE008005C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(C1CMODE,         0xE0081000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C1CCMD,          0xE0081004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C1CGS,           0xE0081008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C1CIC,           0xE008100C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C1CIE,           0xE0081010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C1CBT,           0xE0081014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C1CEWL,          0xE0081018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C1CSTAT,         0xE008101C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C1CRXBMI,        0xE0081020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C1CRXBID,        0xE0081024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C1CRXBDA,        0xE0081028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C1CRXBDB,        0xE008102C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C1CTXB1MI,       0xE0081030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C1CTXB1ID,       0xE0081034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C1CTXB1DA,       0xE0081038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C1CTXB1DB,       0xE008103C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C1CTXB2MI,       0xE0081040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C1CTXB2ID,       0xE0081044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C1CTXB2DA,       0xE0081048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C1CTXB2DB,       0xE008104C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C1CTXB3MI,       0xE0081050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C1CTXB3ID,       0xE0081054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C1CTXB3DA,       0xE0081058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C1CTXB3DB,       0xE008105C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(C2CMODE,         0xE0082000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C2CCMD,          0xE0082004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C2CGS,           0xE0082008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C2CIC,           0xE008200C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C2CIE,           0xE0082010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C2CBT,           0xE0082014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C2CEWL,          0xE0082018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C2CSTAT,         0xE008201C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C2CRXBMI,        0xE0082020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C2CRXBID,        0xE0082024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C2CRXBDA,        0xE0082028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C2CRXBDB,        0xE008202C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C2CTXB1MI,       0xE0082030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C2CTXB1ID,       0xE0082034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C2CTXB1DA,       0xE0082038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C2CTXB1DB,       0xE008203C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C2CTXB2MI,       0xE0082040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C2CTXB2ID,       0xE0082044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C2CTXB2DA,       0xE0082048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C2CTXB2DB,       0xE008204C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C2CTXB3MI,       0xE0082050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C2CTXB3ID,       0xE0082054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C2CTXB3DA,       0xE0082058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C2CTXB3DB,       0xE008205C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CAN3
 **
 ***************************************************************************/
__IO_REG32_BIT(C3CMODE,         0xE0083000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C3CCMD,          0xE0083004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C3CGS,           0xE0083008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C3CIC,           0xE008300C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C3CIE,           0xE0083010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C3CBT,           0xE0083014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C3CEWL,          0xE0083018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C3CSTAT,         0xE008301C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C3CRXBMI,        0xE0083020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C3CRXBID,        0xE0083024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C3CRXBDA,        0xE0083028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C3CRXBDB,        0xE008302C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C3CTXB1MI,       0xE0083030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C3CTXB1ID,       0xE0083034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C3CTXB1DA,       0xE0083038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C3CTXB1DB,       0xE008303C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C3CTXB2MI,       0xE0083040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C3CTXB2ID,       0xE0083044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C3CTXB2DA,       0xE0083048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C3CTXB2DB,       0xE008304C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C3CTXB3MI,       0xE0083050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C3CTXB3ID,       0xE0083054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C3CTXB3DA,       0xE0083058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C3CTXB3DB,       0xE008305C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CAN4
 **
 ***************************************************************************/
__IO_REG32_BIT(C4CMODE,         0xE0084000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C4CCMD,          0xE0084004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C4CGS,           0xE0084008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C4CIC,           0xE008400C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C4CIE,           0xE0084010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C4CBT,           0xE0084014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C4CEWL,          0xE0084018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C4CSTAT,         0xE008401C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C4CRXBMI,        0xE0084020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C4CRXBID,        0xE0084024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C4CRXBDA,        0xE0084028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C4CRXBDB,        0xE008402C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C4CTXB1MI,       0xE0084030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C4CTXB1ID,       0xE0084034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C4CTXB1DA,       0xE0084038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C4CTXB1DB,       0xE008403C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C4CTXB2MI,       0xE0084040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C4CTXB2ID,       0xE0084044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C4CTXB2DA,       0xE0084048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C4CTXB2DB,       0xE008404C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C4CTXB3MI,       0xE0084050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C4CTXB3ID,       0xE0084054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C4CTXB3DA,       0xE0084058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C4CTXB3DB,       0xE008405C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CAN5
 **
 ***************************************************************************/
__IO_REG32_BIT(C5CMODE,         0xE0085000,__READ_WRITE ,__ccmode_bits);
__IO_REG32_BIT(C5CCMD,          0xE0085004,__WRITE      ,__cccmd_bits);
__IO_REG32_BIT(C5CGS,           0xE0085008,__READ_WRITE ,__ccgs_bits);
__IO_REG32_BIT(C5CIC,           0xE008500C,__READ       ,__ccic_bits);
__IO_REG32_BIT(C5CIE,           0xE0085010,__READ_WRITE ,__ccie_bits);
__IO_REG32_BIT(C5CBT,           0xE0085014,__READ_WRITE ,__ccbt_bits);
__IO_REG32_BIT(C5CEWL,          0xE0085018,__READ_WRITE ,__ccewl_bits);
__IO_REG32_BIT(C5CSTAT,         0xE008501C,__READ       ,__ccstat_bits);
__IO_REG32_BIT(C5CRXBMI,        0xE0085020,__READ_WRITE ,__ccrxbmi_bits);
__IO_REG32_BIT(C5CRXBID,        0xE0085024,__READ_WRITE ,__ccrxbid_bits);
__IO_REG32_BIT(C5CRXBDA,        0xE0085028,__READ_WRITE ,__ccrxbda_bits);
__IO_REG32_BIT(C5CRXBDB,        0xE008502C,__READ_WRITE ,__ccrxbdb_bits);
__IO_REG32_BIT(C5CTXB1MI,       0xE0085030,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C5CTXB1ID,       0xE0085034,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C5CTXB1DA,       0xE0085038,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C5CTXB1DB,       0xE008503C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C5CTXB2MI,       0xE0085040,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C5CTXB2ID,       0xE0085044,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C5CTXB2DA,       0xE0085048,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C5CTXB2DB,       0xE008504C,__READ_WRITE ,__cctxbdb_bits);
__IO_REG32_BIT(C5CTXB3MI,       0xE0085050,__READ_WRITE ,__cctxbmi_bits);
__IO_REG32_BIT(C5CTXB3ID,       0xE0085054,__READ_WRITE ,__cctxbid_bits);
__IO_REG32_BIT(C5CTXB3DA,       0xE0085058,__READ_WRITE ,__cctxbda_bits);
__IO_REG32_BIT(C5CTXB3DB,       0xE008505C,__READ_WRITE ,__cctxbdb_bits);

/***************************************************************************
 **
 ** CANAFR
 **
 ***************************************************************************/
__IO_REG32_BIT(CAMODE,          0xE0087000,__READ_WRITE ,__camode_bits);
__IO_REG32_BIT(CASFESA,         0xE0087004,__READ_WRITE ,__casfesa_bits);
__IO_REG32_BIT(CASFGSA,         0xE0087008,__READ_WRITE ,__casfgsa_bits);
__IO_REG32_BIT(CAEFESA,         0xE008700C,__READ_WRITE ,__caefesa_bits);
__IO_REG32_BIT(CAEFGSA,         0xE0087010,__READ_WRITE ,__caefgsa_bits);
__IO_REG32_BIT(CAEOTA,          0xE0087014,__READ_WRITE ,__caeota_bits);
__IO_REG32_BIT(CALUTEA,         0xE0087018,__READ       ,__calutea_bits);
__IO_REG32_BIT(CALUTE,          0xE008701C,__READ       ,__calute_bits);

/***************************************************************************
 **
 ** CANCS
 **
 ***************************************************************************/
__IO_REG32_BIT(CCCTS,           0xE0088000,__READ       ,__cccts_bits);
__IO_REG32_BIT(CCCRS,           0xE0088004,__READ       ,__cccrs_bits);
__IO_REG32_BIT(CCCMS,           0xE0088008,__READ       ,__cccms_bits);

/***************************************************************************
 **
 ** LIN0
 **
 ***************************************************************************/
__IO_REG32_BIT(L0MODE,          0xE0089000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L0CFG,           0xE0089004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L0CMD,           0xE0089008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L0FBRG,          0xE008900C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L0STAT,          0xE0089010,__READ_WRITE ,__lstat_bits);
#define L0RBR      L0STAT
#define L0RBR_bit  L0STAT_bit
#define L0THR      L0STAT
#define L0THR_bit  L0STAT_bit
__IO_REG32_BIT(L0IC,            0xE0089014,__READ_WRITE ,__lic_bits);
#define L0IER      L0IC
#define L0IER_bit  L0IC_bit
__IO_REG32_BIT(L0IE,            0xE0089018,__READ_WRITE ,__lie_bits);
#define L0IIR      L0IE
#define L0IIR_bit  L0IE_bit
__IO_REG32_BIT(L0CR,            0xE008901C,__READ_WRITE ,__lcr_bits);
__IO_REG32_BIT(L0CS,            0xE0089020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L0TO,            0xE0089024,__READ_WRITE ,__lto_bits);
#define L0LSR      L0TO
#define L0LSR_bit  L0TO_bit
__IO_REG32_BIT(L0ID,            0xE0089028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L0DATA,          0xE008902C,__READ_WRITE ,__ldata_bits);
#define L0SCR      L0DATA
#define L0SCR_bit  L0DATA_bit
__IO_REG32_BIT(L0DATB,          0xE0089030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L0DATC,          0xE0089034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L0DATD,          0xE0089038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** LIN1
 **
 ***************************************************************************/
__IO_REG32_BIT(L1MODE,          0xE008A000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L1CFG,           0xE008A004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L1CMD,           0xE008A008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L1FBRG,          0xE008A00C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L1STAT,          0xE008A010,__READ_WRITE ,__lstat_bits);
#define L1RBR      L1STAT
#define L1RBR_bit  L1STAT_bit
#define L1THR      L1STAT
#define L1THR_bit  L1STAT_bit
__IO_REG32_BIT(L1IC,            0xE008A014,__READ_WRITE ,__lic_bits);
#define L1IER      L1IC
#define L1IER_bit  L1IC_bit
__IO_REG32_BIT(L1IE,            0xE008A018,__READ_WRITE ,__lie_bits);
#define L1IIR      L1IE
#define L1IIR_bit  L1IE_bit
__IO_REG32_BIT(L1CR,            0xE008A01C,__READ_WRITE ,__lcr_bits);
__IO_REG32_BIT(L1CS,            0xE008A020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L1TO,            0xE008A024,__READ_WRITE ,__lto_bits);
#define L1LSR      L1TO
#define L1LSR_bit  L1TO_bit
__IO_REG32_BIT(L1ID,            0xE008A028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L1DATA,          0xE008A02C,__READ_WRITE ,__ldata_bits);
#define L1SCR      L1DATA
#define L1SCR_bit  L1DATA_bit
__IO_REG32_BIT(L1DATB,          0xE008A030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L1DATC,          0xE008A034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L1DATD,          0xE008A038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** LIN2
 **
 ***************************************************************************/
__IO_REG32_BIT(L2MODE,          0xE008B000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L2CFG,           0xE008B004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L2CMD,           0xE008B008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L2FBRG,          0xE008B00C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L2STAT,          0xE008B010,__READ_WRITE ,__lstat_bits);
#define L2RBR      L2STAT
#define L2RBR_bit  L2STAT_bit
#define L2THR      L2STAT
#define L2THR_bit  L2STAT_bit
__IO_REG32_BIT(L2IC,            0xE008B014,__READ_WRITE ,__lic_bits);
#define L2IER      L2IC
#define L2IER_bit  L2IC_bit
__IO_REG32_BIT(L2IE,            0xE008B018,__READ_WRITE ,__lie_bits);
#define L2IIR      L2IE
#define L2IIR_bit  L2IE_bit
__IO_REG32_BIT(L2CR,            0xE008B01C,__READ_WRITE ,__lcr_bits);
__IO_REG32_BIT(L2CS,            0xE008B020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L2TO,            0xE008B024,__READ_WRITE ,__lto_bits);
#define L2LSR      L2TO
#define L2LSR_bit  L2TO_bit
__IO_REG32_BIT(L2ID,            0xE008B028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L2DATA,          0xE008B02C,__READ_WRITE ,__ldata_bits);
#define L2SCR      L2DATA
#define L2SCR_bit  L2DATA_bit
__IO_REG32_BIT(L2DATB,          0xE008B030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L2DATC,          0xE008B034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L2DATD,          0xE008B038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** LIN3
 **
 ***************************************************************************/
__IO_REG32_BIT(L3MODE,          0xE008C000,__READ_WRITE ,__lmode_bits);
__IO_REG32_BIT(L3CFG,           0xE008C004,__READ_WRITE ,__lcfg_bits);
__IO_REG32_BIT(L3CMD,           0xE008C008,__READ_WRITE ,__lcmd_bits);
__IO_REG32_BIT(L3FBRG,          0xE008C00C,__READ_WRITE ,__lfbrg_bits);
__IO_REG32_BIT(L3STAT,          0xE008C010,__READ_WRITE ,__lstat_bits);
#define L3RBR      L3STAT
#define L3RBR_bit  L3STAT_bit
#define L3THR      L3STAT
#define L3THR_bit  L3STAT_bit
__IO_REG32_BIT(L3IC,            0xE008C014,__READ_WRITE ,__lic_bits);
#define L3IER      L3IC
#define L3IER_bit  L3IC_bit
__IO_REG32_BIT(L3IE,            0xE008C018,__READ_WRITE ,__lie_bits);
#define L3IIR      L3IE
#define L3IIR_bit  L3IE_bit
__IO_REG32_BIT(L3CR,            0xE008C01C,__READ_WRITE ,__lcr_bits);
__IO_REG32_BIT(L3CS,            0xE008C020,__READ_WRITE ,__lcs_bits);
__IO_REG32_BIT(L3TO,            0xE008C024,__READ_WRITE ,__lto_bits);
#define L3LSR      L3TO
#define L3LSR_bit  L3TO_bit
__IO_REG32_BIT(L3ID,            0xE008C028,__READ_WRITE ,__lid_bits);
__IO_REG32_BIT(L3DATA,          0xE008C02C,__READ_WRITE ,__ldata_bits);
#define L3SCR      L3DATA
#define L3SCR_bit  L3DATA_bit
__IO_REG32_BIT(L3DATB,          0xE008C030,__READ_WRITE ,__ldatb_bits);
__IO_REG32_BIT(L3DATC,          0xE008C034,__READ_WRITE ,__ldatc_bits);
__IO_REG32_BIT(L3DATD,          0xE008C038,__READ_WRITE ,__ldatd_bits);

/***************************************************************************
 **
 ** VIC
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_PRIORITY_MASK_0, 0xFFFFF000,__READ_WRITE ,__int_priority_mask_bits);
__IO_REG32_BIT(INT_PRIORITY_MASK_1, 0xFFFFF004,__READ_WRITE ,__int_priority_mask_bits);
__IO_REG32_BIT(INT_VECTOR_0,        0xFFFFF100,__READ_WRITE ,__int_vector_bits);
__IO_REG32_BIT(INT_VECTOR_1,        0xFFFFF104,__READ_WRITE ,__int_vector_bits);
__IO_REG32_BIT(INT_PENDING_1_31,    0xFFFFF200,__READ       ,__int_pending_1_31_bits);
__IO_REG32_BIT(INT_FEATURES,        0xFFFFF300,__READ       ,__int_features_bits);
__IO_REG32_BIT(INT_REQUEST_1,       0xFFFFF404,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_2,       0xFFFFF408,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_3,       0xFFFFF40C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_4,       0xFFFFF410,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_5,       0xFFFFF414,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_6,       0xFFFFF418,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_7,       0xFFFFF41C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_8,       0xFFFFF420,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_9,       0xFFFFF424,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_10,      0xFFFFF428,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_11,      0xFFFFF42C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_12,      0xFFFFF430,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_13,      0xFFFFF434,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_14,      0xFFFFF438,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_15,      0xFFFFF43C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_16,      0xFFFFF440,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_17,      0xFFFFF444,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_18,      0xFFFFF448,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_19,      0xFFFFF44C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_20,      0xFFFFF450,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_21,      0xFFFFF454,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_22,      0xFFFFF458,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_23,      0xFFFFF45C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_24,      0xFFFFF460,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_25,      0xFFFFF464,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_26,      0xFFFFF468,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_27,      0xFFFFF46C,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_28,      0xFFFFF470,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_29,      0xFFFFF474,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_30,      0xFFFFF478,__READ_WRITE ,__int_request_bits);
__IO_REG32_BIT(INT_REQUEST_31,      0xFFFFF47C,__READ_WRITE ,__int_request_bits);

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
 **  VIC Interrupt channels
 **
 ***************************************************************************/
#define VIC_TIMER0       1  /* capture or match interrupt from timer 0              */
#define VIC_TIMER1       2  /* capture or match interrupt from timer 1              */
#define VIC_TIMER2       3  /* capture or match interrupt from timer 2              */
#define VIC_TIMER3       4  /* capture or match interrupt from timer 3              */
#define VIC_UART0        5  /* general interrupt from 16C550 UART                   */
#define VIC_SPI0         6  /* general interrupt from SPI 0                         */
#define VIC_SPI1         7  /* general interrupt from SPI 1                         */
#define VIC_SPI2         8  /* general interrupt from SPI 1                         */
#define VIC_ER           9  /* event, wake-up or real time clock tick interrupt from
                               event router                                         */
#define VIC_ADC         10  /* conversion scan completed interrupt from ADC         */
#define VIC_FLASH       11  /* Isignature, burn or erase finished interrupt from
                               flash                                                */
#define VIC_WDT         12  /* debug underflow interrupt from watchdog              */
#define VIC_DEBUGRX     13  /* communications RX for ARM debug mode                 */
#define VIC_DEBUGTX     14  /* communications TX for ARM debug mode                 */
#define VIC_LIN0        15  /* general interrupt from LIN master controller 0       */
#define VIC_LIN1        16  /* general interrupt from LIN master controller 1       */
#define VIC_LIN2        17  /* general interrupt from LIN master controller 2       */
#define VIC_LIN3        18  /* general interrupt from LIN master controller 3       */
#define VIC_ALL_CAN     19  /* combined general interrupt of all CAN controllers and
                               the CAN Look-Up table                                */
#define VIC_CAN0_RX     20  /* message received interrupt from CAN controller 0     */
#define VIC_CAN1_RX     21  /* message received interrupt from CAN controller 1     */
#define VIC_CAN2_RX     22  /* message received interrupt from CAN controller 2     */
#define VIC_CAN3_RX     23  /* message received interrupt from CAN controller 3     */
#define VIC_CAN4_RX     24  /* message received interrupt from CAN controller 4     */
#define VIC_CAN5_RX     25  /* message received interrupt from CAN controller 5     */
#define VIC_CAN0_TX     26  /* message transmitted interrupt from CAN controller 0  */
#define VIC_CAN1_TX     27  /* message transmitted interrupt from CAN controller 1  */
#define VIC_CAN2_TX     28  /* message transmitted interrupt from CAN controller 2  */
#define VIC_CAN3_TX     29  /* message transmitted interrupt from CAN controller 3  */
#define VIC_CAN4_TX     30  /* message transmitted interrupt from CAN controller 4  */
#define VIC_CAN5_TX     31  /* message transmitted interrupt from CAN controller 5  */

/***************************************************************************
 **
 **  Event router channels
 **
 ***************************************************************************/
#define ER_EXTINT0       0  /* external interrupt input 0                  */
#define ER_EXTINT1       1  /* external interrupt input 1                  */
#define ER_EXTINT2       2  /* external interrupt input 2                  */
#define ER_EXTINT3       3  /* external interrupt input 3                  */
#define ER_CAN0_RXDC     4  /* CAN0 receive data input and wake-up         */
#define ER_CAN1_RXDC     5  /* CAN1 receive data input and wake-up         */
#define ER_CAN2_RXDC     6  /* CAN2 receive data input and wake-up         */
#define ER_CAN3_RXDC     7  /* CAN3 receive data input and wake-up         */
#define ER_CAN4_RXDC     8  /* CAN4 receive data input and wake-up         */
#define ER_CAN5_RXDC     9  /* CAN5 receive data input and wake-up         */
#define ER_LIN0_RXDC    10  /* LIN0 receive data input and wake-up         */
#define ER_LIN1_RXDC    11  /* LIN1 receive data input and wake-up         */
#define ER_LIN2_RXDC    12  /* LIN2 receive data input and wake-up         */
#define ER_LIN3_RXDC    13  /* LIN3 receive data input and wake-up         */
#define ER_RTC          14  /* RTC tick event                              */
#define ER_CAN          15  /* CAN interrupt (internal)                    */
#define ER_IRQ          16  /* VIC IRQ (internal)                          */
#define ER_FIQ          17  /* VIC FIQ (internal)                          */

#endif    /* __IOSJA2020_H */
