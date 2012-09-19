/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Toshiba TMPA900CM
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 41583 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOTMPA900CM_H
#define __IOTMPA900CM_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    TMPA900CM SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

/* Remap Register */
typedef struct{
__REG32 REMAP               : 1;
__REG32                     :31;
} __remap_bits;

/* SYSCR1 (System Control Register-1) */
typedef struct{
__REG32 GEAR                : 3;
__REG32                     :29;
} __syscr1_bits;

/* SYSCR2 (System Control Register-2) */
typedef struct{
__REG32 LUPFLAG             : 1;
__REG32 FCSEL               : 1;
__REG32                     :30;
} __syscr2_bits;

/* SYSCR3 (System Control Register-3) */
typedef struct{
__REG32 ND                  : 5;
__REG32 C2S                 : 1;
__REG32                     : 1;
__REG32 PLLON               : 1;
__REG32                     :24;
} __syscr3_bits;

/* SYSCR4 (System Control Register-4) */
typedef struct{
__REG32 FS                  : 2;
__REG32 IS                  : 2;
__REG32 RS                  : 4;
__REG32                     :24;
} __syscr4_bits;

/* SYSCR5 (System Control Register-5) */
typedef struct{
__REG32 PROTECT             : 1;
__REG32                     :31;
} __syscr5_bits;

/* SYSCR6 (System Control Register-6) */
typedef struct{
__REG32 P_CODE0             : 8;
__REG32                     :24;
} __syscr6_bits;

/* SYSCR7 (System Control Register-7) */
typedef struct{
__REG32 P_CODE1             : 8;
__REG32                     :24;
} __syscr7_bits;

/* SYSCR8 (System Control Register 8) */
typedef struct{
__REG32 USBH_CLKSEL         : 3;
__REG32                     : 1;
__REG32 USBD_CLKSEL         : 2;
__REG32                     :26;
} __syscr8_bits;

/* CLKCR5 (Clock Control Register-5) */
typedef struct{
__REG32 SEL_TIM01           : 1;
__REG32 SEL_TIM23           : 1;
__REG32 SEL_TIM45           : 1;
__REG32 SEL_SMC_MCLK        : 1;
__REG32 USBH_CLKEN          : 1;
__REG32                     :27;
} __clkcr5_bits;

/* NDFMCR0 (NAND-Flash Control Register-0) */
typedef struct{
__REG32 ECCRST              : 1;
__REG32 BUSY                : 1;
__REG32 ECCE                : 1;
__REG32 CE1                 : 1;
__REG32 CE0                 : 1;
__REG32 CLE                 : 1;
__REG32 ALE                 : 1;
__REG32 WE                  : 1;
__REG32 RSECGW              : 1;
__REG32 RSESTA              : 1;
__REG32 RSEDN               : 1;
__REG32 RSECCL              : 1;
__REG32                     :20;
} __ndfmcr0_bits;

/* NDFMCR1 (NAND-Flash Control Register-1) */
typedef struct{
__REG32                     : 1;
__REG32 ECCS                : 1;
__REG32                     : 6;
__REG32 ALS                 : 1;
__REG32 SELAL               : 1;
__REG32 SERR                : 2;
__REG32 STATE               : 4;
__REG32                     :16;
} __ndfmcr1_bits;

/* NDFMCR2 (NAND-Flash Control Register-2) */
typedef struct{
__REG32 SPHR                : 3;
__REG32                     : 1;
__REG32 SPLR                : 3;
__REG32                     : 1;
__REG32 SPHW                : 3;
__REG32                     : 1;
__REG32 SPLW                : 3;
__REG32                     :17;
} __ndfmcr2_bits;

/* NDFINTC (NAND-Flash Interrupt Control Register) */
typedef struct{
__REG32 RDYIE               : 1;
__REG32 RDYRIS              : 1;
__REG32 RDYEIS              : 1;
__REG32 RDYIC               : 1;
__REG32 RSEIE               : 1;
__REG32 RSERIS              : 1;
__REG32 RSEEIS              : 1;
__REG32 RSEIC               : 1;
__REG32                     :24;
} __ndfintc_bits;

/* NDECCRD2 (NAND-Flash ECC-code Read Register-2) */
typedef struct{
__REG32 CODE2               :16;
__REG32                     :16;
} __ndeccrd2_bits;

/* NDRSCA0,1,2,3 (NAND-Flash Reed-Solomon Calculation result Address Register-0)*/
typedef struct{
__REG32 AL                  :10;
__REG32                     :22;
} __ndrsca_bits;

/* NDRSCD0,1,2,3 (NAND-Flash Reed-Solomon Calculation result Address Register-0)*/
typedef struct{
__REG32 DATA                : 8;
__REG32                     :24;
} __ndrscd_bits;

/* Timer Load Register */
typedef struct{
__REG32 TIMSD               :16;
__REG32                     :16;
} __timerload_bits;

/* Timer Data Register */
typedef struct{
__REG32 TIMCD               :16;
__REG32                     :16;
} __timervalue_bits;

/* Timer Control Register */
typedef struct{
__REG32 TIMOSCTL            : 1;
__REG32 TIMSIZE             : 1;
__REG32 TIMPRS              : 2;
__REG32                     : 1;
__REG32 TIMINTE             : 1;
__REG32 TIMMOD              : 1;
__REG32 TIMEN               : 1;
__REG32                     :24;
} __timercontrol_bits;

/* Timer Interrupt Raw Flag Register */
typedef struct{
__REG32 TIMRIF              : 1;
__REG32                     :31;
} __timerris_bits;

/* Timer Interrupt Masked Flag Register */
typedef struct{
__REG32 TIMMIF              : 1;
__REG32                     :31;
} __timermis_bits;

/* Timer Back Ground Counter Data Register */
typedef struct{
__REG32 TIMBSD              :16;
__REG32                     :16;
} __timerbgload_bits;

/* Timer mode register */
typedef struct{
__REG32                     : 4;
__REG32 PWM_Period          : 2;
__REG32 PWM_Mode            : 1;
__REG32                     :25;
} __timermode_bits;

/* Timer Compare Value */
typedef struct{
__REG32 TIMCPD              :16;
__REG32                     :16;
} __timercompare1_bits;

/* Timer Compare Enable Register */
typedef struct{
__REG32 TIMCPE              : 1;
__REG32                     :31;
} __timercmpen_bits;

/* Timer Compare raw interrupt status Register */
typedef struct{
__REG32 TIMCRIF             : 1;
__REG32                     :31;
} __timercmpris_bits;

/* Timer Compare Masked interrupt status Register */
typedef struct{
__REG32 TIMCMIF             : 1;
__REG32                     :31;
} __timercmpmis_bits;

/* Timer Back Ground Compare Register */
typedef struct{
__REG32 TIMBGCPD            :16;
__REG32                     :16;
} __timerbgcmp_bits;

/* I2STCON register */
typedef struct{
__REG32 I2STx_DELAYOFF      : 1;
__REG32 I2STx_WSINV         : 1;
__REG32 I2STx_MSBINV        : 1;
__REG32 I2STx_UNDERFLOW     : 1;
__REG32                     : 4;
__REG32 I2STx_BITCNV        : 1;
__REG32                     : 3;
__REG32 I2STx_LCH_CUT       : 1;
__REG32 I2STx_RCH_CUT       : 1;
__REG32                     :18;
} __i2stcon_bits;

/* I2SRCON register */
typedef struct{
__REG32 I2SRx_DELAYOFF      : 1;
__REG32 I2SRx_WSINV         : 1;
__REG32 I2SRx_MSBINV        : 1;
__REG32 I2SRx_UNDERFLOW     : 1;
__REG32                     : 4;
__REG32 I2SRx_BITCNV        : 1;
__REG32                     : 3;
__REG32 I2SRx_LCH_CUT       : 1;
__REG32 I2SRx_RCH_CUT       : 1;
__REG32                     :18;
} __i2srcon_bits;

/* I2STSLVON register */
typedef struct{
__REG32 I2STx_SLAVE         : 1;
__REG32                     :31;
} __i2stslvon_bits;

/* I2SRSLVON (RxI2S Slave WS/SCK Control Register) */
typedef struct{
__REG32 I2SRx_SLAVE         : 1;
__REG32                     :31;
} __i2srslvon_bits;

/* I2STFCLR register */
typedef struct{
__REG32 I2STx_FIFOCLR       : 1;
__REG32                     :31;
} __i2stfclr_bits;

/* I2SFRFCLR register */
typedef struct{
__REG32 I2SRx_FIFOCLR       : 1;
__REG32                     :31;
} __i2srfclr_bits;

/* I2STMS register */
typedef struct{
__REG32 I2STx_MASTER        : 1;
__REG32                     :31;
} __i2stms_bits;

/* I2SRMS register */
typedef struct{
__REG32 I2SRx_MASTER        : 1;
__REG32                     :31;
} __i2srms_bits;

/* I2STMCON register */
typedef struct{
__REG32 I2STx_CLKS_DIV      : 2;
__REG32 I2STx_WS_DIV        : 2;
__REG32                     :28;
} __i2stmcon_bits;

/* I2SRMCON register */
typedef struct{
__REG32 I2SRx_CLKS_DIV      : 2;
__REG32 I2SRx_WS_DIV        : 2;
__REG32                     :28;
} __i2srmcon_bits;

/* I2STMSTP register */
/* I2SRMCON register */
typedef struct{
__REG32 I2STx_MSTOP         : 1;
__REG32                     :31;
} __i2stmstp_bits;

/* I2SRMCON register */
typedef struct{
__REG32 I2SRx_MSTOP         : 1;
__REG32                     :31;
} __i2srmstp_bits;

/* I2STDMA1 register */
typedef struct{
__REG32 I2STx_DMAREADY1     : 1;
__REG32                     :31;
} __i2stdma1_bits;

/* I2SRDMA1 register */
typedef struct{
__REG32 I2SRx_DMAREADY1     : 1;
__REG32                     :31;
} __i2srdma1_bits;

/* I2SCOMMON register */
typedef struct{
__REG32 COMMON              : 1;
__REG32 LOOP                : 1;
__REG32 I2SSCLK             : 1;
__REG32 MCLKSEL1            : 1;
__REG32 MCLKSEL0            : 1;
__REG32                     :27;
} __i2scommon_bits;

/* I2STST register */
typedef struct{
__REG32 I2STx_FIFOEMPTY     : 1;
__REG32 I2STx_FIFOFULL      : 1;
__REG32 I2STx_STATUS        : 2;
__REG32                     :28;
} __i2stst_bits;

/* I2SRST register */
typedef struct{
__REG32 I2SRx_FIFOEMPTY     : 1;
__REG32 I2SRx_FIFOFULL      : 1;
__REG32 I2SRx_STATUS        : 2;
__REG32                     :28;
} __i2srst_bits;

/* I2SINT register */
typedef struct{
__REG32 I2STx_UNDERFLOW_INT : 1;
__REG32 I2STx_OVERFLOW_INT  : 1;
__REG32 I2SRx_UNDERFLOW_INT : 1;
__REG32 I2SRx_OVERFLOW_INT  : 1;
__REG32                     :28;
} __i2sint_bits;

/* I2SINTMSK register */
typedef struct{
__REG32 I2STx_UNDERFLOW_INTM  : 1;
__REG32 I2STx_OVERFLOW_INTM   : 1;
__REG32 I2SRx_UNDERFLOW_INTM  : 1;
__REG32 I2SRx_OVERFLOW_INTM   : 1;
__REG32                       :28;
} __i2sintmsk_bits;

/* I2STDAT (Transmit FIFO Window)
   I2SRDAT (Receive FIFO Window) */
typedef struct{
__REG32 PCM_Right             :16;
__REG32 PCM_Left              :16;
} __i2stdat_bits;

/* LCDTiming0 (Horizontal-direction control register) */
typedef struct{
__REG32                       : 2;
__REG32 PPL                   : 6;
__REG32 HSW                   : 8;
__REG32 HFP                   : 8;
__REG32 HBP                   : 8;
} __lcdtiming0_bits;

/* LCDTiming1 (Vertical direction control register) */
typedef struct{
__REG32 LPP                   :10;
__REG32 VSW                   : 6;
__REG32 VFP                   : 8;
__REG32 VBP                   : 8;
} __lcdtiming1_bits;

/* LCDTiming2 (Clock/signal polarity control register) */
typedef struct{
__REG32 PCD_LO                : 5;
__REG32                       : 1;
__REG32 ACB                   : 5;
__REG32 IVS                   : 1;
__REG32 IHS                   : 1;
__REG32 IPC                   : 1;
__REG32 IOE                   : 1;
__REG32                       : 1;
__REG32 CPL                   :10;
__REG32                       : 1;
__REG32 PCD_HI                : 5;
} __lcdtiming2_bits;

/* LCDTiming3 (Line end control register) */
typedef struct{
__REG32 LED                   : 7;
__REG32                       : 9;
__REG32 LEE                   : 1;
__REG32                       :15;
} __lcdtiming3_bits;

/* LCDIMSC (Interrupt Mask Set/Clear Register (Enable) register) */
typedef struct{
__REG32                       : 1;
__REG32 FUFINTRENB            : 1;
__REG32 LNBUINTRENB           : 1;
__REG32 VCOMPINTRENB          : 1;
__REG32 MBERRINTRENB          : 1;
__REG32                       :27;
} __lcdimsc_bits;

/* LCDControl (LCD Control register) */
typedef struct{
__REG32 LcdEn                 : 1;
__REG32 LcdBpp                : 3;
__REG32 LcdBW                 : 1;
__REG32 LcdTFT                : 1;
__REG32 LcdMono8              : 1;
__REG32 LcdDual               : 1;
__REG32 BGR                   : 1;
__REG32                       : 3;
__REG32 LcdVComp              : 2;
__REG32                       : 2;
__REG32 WATERMARK             : 1;
__REG32                       :15;
} __lcdcontrol_bits;

/* Raw Interrupt Status Register LCDRIS */
typedef struct{
__REG32                       : 1;
__REG32 FUF                   : 1;
__REG32 LNBU                  : 1;
__REG32 Vcomp                 : 1;
__REG32 MBERROR               : 1;
__REG32                       :27;
} __lcdris_bits;

/* LCDMIS (Masked Interrupt Status Register) */
typedef struct{
__REG32                       : 1;
__REG32 FUFINTR               : 1;
__REG32 LNBUINTR              : 1;
__REG32 VCOMPINTR             : 1;
__REG32 MBERRORINTR           : 1;
__REG32                       :27;
} __lcdmis_bits;

/* LCDICR (Interrupt Clear Register) */
typedef struct{
__REG32                       : 1;
__REG32 FUFINTRCLR            : 1;
__REG32 LNBUINTRCLR           : 1;
__REG32 VCOMPINTRCLR          : 1;
__REG32 MBERRORINTRCLR        : 1;
__REG32                       :27;
} __lcdicr_bits;

/* STN64CR (LCDC Option Control Register) */
typedef struct{
__REG32 G64_en                : 1;
__REG32 G64_8bit              : 1;
__REG32 LCP_Inv               : 1;
__REG32 CLAC_Inv              : 1;
__REG32 CLLP_Inv              : 1;
__REG32 CLFP_Inv              : 1;
__REG32                       : 1;
__REG32 NoSpikeMode           : 1;
__REG32                       :24;
} __stn64cr_bits;

/* VIC Intr */
typedef struct{
__REG32 INTR_WDT              : 1;
__REG32 INTR_RTC              : 1;
__REG32 INTR_TIMER01          : 1;
__REG32 INTR_TIMER23          : 1;
__REG32 INTR_TIMER45          : 1;
__REG32 INTR_GPIOD            : 1;
__REG32 INTR_I2S0             : 1;
__REG32 INTR_I2S1             : 1;
__REG32 INTR_ADC              : 1;
__REG32 INTR_UART2            : 1;
__REG32 INTR_UART0            : 1;
__REG32 INTR_UART1            : 1;
__REG32 INTR_SSP0             : 1;
__REG32 INTR_SSP1             : 1;
__REG32 INTR_NDFC             : 1;
__REG32 INTR_CMSIF            : 1;
__REG32 INTR_DMA_ERR          : 1;
__REG32 INTR_DMA_END          : 1;
__REG32 INTR_LCDC             : 1;
__REG32                       : 1;
__REG32 INTR_LCDDA            : 1;
__REG32 INTR_USB              : 1;
__REG32 INTR_SDHC             : 1;
__REG32 INTR_I2S              : 1;
__REG32                       : 2;
__REG32 INTR_GPIOR            : 1;
__REG32 INTR_USBH             : 1;
__REG32 INTR_GPION            : 1;
__REG32 INTR_GPIOF            : 1;
__REG32 INTR_GPIOC            : 1;
__REG32 INTR_GPIOA            : 1;
} __vic_intr_bits;

/* VICPROTECTION (Protection Enable Register) */
typedef struct{
__REG32 Protection            : 1;
__REG32                       :31;
} __vicprotection_bits;

/* VICSWPRIORITYMASK (Software Priority Mask Register) */
typedef struct{
__REG32 SWPriorityMask        :16;
__REG32                       :16;
} __vicswprioritymask_bits;

/* VICVECTPRIORITY (Vector Priority x Register) */
typedef struct{
__REG32 VectPriority          : 4;
__REG32                       :28;
} __vicvectpriority_bits;

/* DMACIntStatus (DMAC Interrupt Status Register) */
typedef struct{
__REG32 IntStatus0            : 1;
__REG32 IntStatus1            : 1;
__REG32 IntStatus2            : 1;
__REG32 IntStatus3            : 1;
__REG32 IntStatus4            : 1;
__REG32 IntStatus5            : 1;
__REG32 IntStatus6            : 1;
__REG32 IntStatus7            : 1;
__REG32                       :24;
} __dmacintstaus_bits;

/* DMACIntTCStatus (DMAC Interrupt Terminal Count Status Register) */
typedef struct{
__REG32 IntTCStatus0          : 1;
__REG32 IntTCStatus1          : 1;
__REG32 IntTCStatus2          : 1;
__REG32 IntTCStatus3          : 1;
__REG32 IntTCStatus4          : 1;
__REG32 IntTCStatus5          : 1;
__REG32 IntTCStatus6          : 1;
__REG32 IntTCStatus7          : 1;
__REG32                       :24;
} __dmacinttcstatus_bits;

/* DMACIntTCClear (DMAC Interrupt Terminal Count Clear Register) */
typedef struct{
__REG32 IntTCClear0           : 1;
__REG32 IntTCClear1           : 1;
__REG32 IntTCClear2           : 1;
__REG32 IntTCClear3           : 1;
__REG32 IntTCClear4           : 1;
__REG32 IntTCClear5           : 1;
__REG32 IntTCClear6           : 1;
__REG32 IntTCClear7           : 1;
__REG32                       :24;
} __dmacinttcclear_bits;

/* DMACIntErrorStatus (DMAC Interrupt Error Status Register) */
typedef struct{
__REG32 IntErrStatus0         : 1;
__REG32 IntErrStatus1         : 1;
__REG32 IntErrStatus2         : 1;
__REG32 IntErrStatus3         : 1;
__REG32 IntErrStatus4         : 1;
__REG32 IntErrStatus5         : 1;
__REG32 IntErrStatus6         : 1;
__REG32 IntErrStatus7         : 1;
__REG32                       :24;
} __dmacinterrorstatus_bits;

/* DMACIntErrClr (DMAC Interrupt Error Clear Register) */
typedef struct{
__REG32 IntErrClr0            : 1;
__REG32 IntErrClr1            : 1;
__REG32 IntErrClr2            : 1;
__REG32 IntErrClr3            : 1;
__REG32 IntErrClr4            : 1;
__REG32 IntErrClr5            : 1;
__REG32 IntErrClr6            : 1;
__REG32 IntErrClr7            : 1;
__REG32                       :24;
} __dmacinterrclr_bits;

/* DMACRawIntTCStatus (DMAC Raw Interrupt Terminal Count Status Register) */
typedef struct{
__REG32 RawIntTCS0            : 1;
__REG32 RawIntTCS1            : 1;
__REG32 RawIntTCS2            : 1;
__REG32 RawIntTCS3            : 1;
__REG32 RawIntTCS4            : 1;
__REG32 RawIntTCS5            : 1;
__REG32 RawIntTCS6            : 1;
__REG32 RawIntTCS7            : 1;
__REG32                       :24;
} __dmacrawinttcstatus_bits;

/* DMACRawIntErrorStatus (DMAC Raw Error Interrupt Status Register) */
typedef struct{
__REG32 RawIntErrS0           : 1;
__REG32 RawIntErrS1           : 1;
__REG32 RawIntErrS2           : 1;
__REG32 RawIntErrS3           : 1;
__REG32 RawIntErrS4           : 1;
__REG32 RawIntErrS5           : 1;
__REG32 RawIntErrS6           : 1;
__REG32 RawIntErrS7           : 1;
__REG32                       :24;
} __dmacrawinterrorstatus_bits;

/* DMACEnbldChns (DMAC Enabled Channel Register) */
typedef struct{
__REG32 EnabledCH0            : 1;
__REG32 EnabledCH1            : 1;
__REG32 EnabledCH2            : 1;
__REG32 EnabledCH3            : 1;
__REG32 EnabledCH4            : 1;
__REG32 EnabledCH5            : 1;
__REG32 EnabledCH6            : 1;
__REG32 EnabledCH7            : 1;
__REG32                       :24;
} __dmacenbldchns_bits;

/* DMACSoftSReq (DMAC Software Burst Request Register) */
typedef struct{
__REG32 SoftBReq0             : 1;
__REG32 SoftBReq1             : 1;
__REG32                       : 2;
__REG32 SoftBReq4             : 1;
__REG32 SoftBReq5             : 1;
__REG32 SoftBReq6             : 1;
__REG32 SoftBReq7             : 1;
__REG32 SoftBReq8             : 1;
__REG32 SoftBReq9             : 1;
__REG32 SoftBReq10            : 1;
__REG32 SoftBReq11            : 1;
__REG32                       : 2;
__REG32 SoftBReq14            : 1;
__REG32                       :17;
} __dmacsoftbreq_bits;

/* DMACSoftSReq (DMAC Software Single Request Register ) */
typedef struct{
__REG32 SoftSReq0             : 1;
__REG32 SoftSReq1             : 1;
__REG32 SoftSReq2             : 1;
__REG32 SoftSReq3             : 1;
__REG32                       : 2;
__REG32 SoftSReq6             : 1;
__REG32 SoftSReq7             : 1;
__REG32                       : 4;
__REG32 SoftSReq12            : 1;
__REG32 SoftSReq13            : 1;
__REG32 SoftSReq14            : 1;
__REG32                       :17;
} __dmacsoftsreq_bits;

/* DMACConfiguration (DMAC Configuration Register) */
typedef struct{
__REG32 E                     : 1;
__REG32 M1                    : 1;
__REG32 M2                    : 1;
__REG32                       :29;
} __dmacconfiguration_bits;

/* DMACCxLLI (DMAC Channel x Linked List Item Register) */
typedef struct{
__REG32 LM                    : 1;
__REG32                       : 1;
__REG32 LLI                   :30;
} __dmacclli_bits;

/* DMACCxCTL (DMAC Channel x Control Register) */
typedef struct{
__REG32 TransferSize          :12;
__REG32 SBSize                : 3;
__REG32 DBSize                : 3;
__REG32 Swidth                : 3;
__REG32 Dwidth                : 3;
__REG32 S                     : 1;
__REG32 D                     : 1;
__REG32 SI                    : 1;
__REG32 DI                    : 1;
__REG32 Prot1                 : 1;
__REG32 Prot2                 : 1;
__REG32 Prot3                 : 1;
__REG32 I                     : 1;
} __dmacccontrol_bits;

/* DMACCxCFG (DMAC Channel x Configuration Register) */
typedef struct{
__REG32 E                     : 1;
__REG32 SrcPeripheral         : 4;
__REG32                       : 1;
__REG32 DestPeripheral        : 4;
__REG32                       : 1;
__REG32 FlowCntrl             : 3;
__REG32 IE                    : 1;
__REG32 ITC                   : 1;
__REG32 Lock                  : 1;
__REG32 Active                : 1;
__REG32 Halt                  : 1;
__REG32                       :13;
} __dmaccconfiguration_bits;

/* GPIOADATA (Port Data Register) */
typedef struct{
__REG32 PA0                   : 1;
__REG32 PA1                   : 1;
__REG32 PA2                   : 1;
__REG32 PA3                   : 1;
__REG32                       :28;
} __gpioadata_bits;

/* GPIOAIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32 PA0IS                 : 1;
__REG32 PA1IS                 : 1;
__REG32 PA2IS                 : 1;
__REG32 PA3IS                 : 1;
__REG32                       :28;
} __gpioais_bits;

/* GPIOAIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32 PA0IBE                : 1;
__REG32 PA1IBE                : 1;
__REG32 PA2IBE                : 1;
__REG32 PA3IBE                : 1;
__REG32                       :28;
} __gpioaibe_bits;

/* GPIOAIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/Highlevel)) */
typedef struct{
__REG32 PA0IEV                : 1;
__REG32 PA1IEV                : 1;
__REG32 PA2IEV                : 1;
__REG32 PA3IEV                : 1;
__REG32                       :28;
} __gpioaiev_bits;

/* GPIOAIE (Port Interrupt Enable Register) */
typedef struct{
__REG32 PA0IE                 : 1;
__REG32 PA1IE                 : 1;
__REG32 PA2IE                 : 1;
__REG32 PA3IE                 : 1;
__REG32                       :28;
} __gpioaie_bits;

/* GPIOARIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32 PA0RIS                : 1;
__REG32 PA1RIS                : 1;
__REG32 PA2RIS                : 1;
__REG32 PA3RIS                : 1;
__REG32                       :28;
} __gpioaris_bits;

/* GPIOAMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32 PA0MIS                : 1;
__REG32 PA1MIS                : 1;
__REG32 PA2MIS                : 1;
__REG32 PA3MIS                : 1;
__REG32                       :28;
} __gpioamis_bits;

/* GPIOAIC (Port Interrupt Clear Register) */
typedef struct{
__REG32 PA0IC                 : 1;
__REG32 PA1IC                 : 1;
__REG32 PA2IC                 : 1;
__REG32 PA3IC                 : 1;
__REG32                       :28;
} __gpioaic_bits;

/* GPIOBDATA (Port Data Register) */
typedef struct{
__REG32 PB0                   : 1;
__REG32 PB1                   : 1;
__REG32 PB2                   : 1;
__REG32 PB3                   : 1;
__REG32                       :28;
} __gpiobdata_bits;

/* GPIOBFR1 (Port B Function Register 1) */
typedef struct{
__REG32 PB0F1                 : 1;
__REG32 PB1F1                 : 1;
__REG32 PB2F1                 : 1;
__REG32 PB3F1                 : 1;
__REG32                       :28;
} __gpiobfr1_bits;

/* GPIOBFR2 (Port B Function Register 2) */
typedef struct{
__REG32 PB0F2                 : 1;
__REG32 PB1F2                 : 1;
__REG32 PB2F2                 : 1;
__REG32 PB3F2                 : 1;
__REG32                       :28;
} __gpiobfr2_bits;

/* GPIOBODE (Port B Open-drain Output Enable Register) */
typedef struct{
__REG32 PB0ODE                : 1;
__REG32 PB1ODE                : 1;
__REG32 PB2ODE                : 1;
__REG32 PB3ODE                : 1;
__REG32                       :28;
} __gpiobode_bits;

/* GPIOCDATA (Port Data Register) */
typedef struct{
__REG32                       : 2;
__REG32 PC2                   : 1;
__REG32 PC3                   : 1;
__REG32 PC4                   : 1;
__REG32                       : 1;
__REG32 PC6                   : 1;
__REG32 PC7                   : 1;
__REG32                       :24;
} __gpiocdata_bits;

/* GPIOCDIR (Port Data Direction Register) */
typedef struct{
__REG32                       : 2;
__REG32 PC2C                  : 1;
__REG32 PC3C                  : 1;
__REG32 PC4C                  : 1;
__REG32                       : 1;
__REG32 PC6C                  : 1;
__REG32 PC7C                  : 1;
__REG32                       :24;
} __gpiocdir_bits;

/* GPIOCFR1 (Port Function Register1) */
typedef struct{
__REG32                       : 2;
__REG32 PC2F1                 : 1;
__REG32 PC3F1                 : 1;
__REG32 PC4F1                 : 1;
__REG32                       : 1;
__REG32 PC6F1                 : 1;
__REG32 PC7F1                 : 1;
__REG32                       :24;
} __gpiocfr1_bits;

/* GPIOCFR2 (Port Function Register2) */
typedef struct{
__REG32                       : 2;
__REG32 PC2F2                 : 1;
__REG32 PC3F2                 : 1;
__REG32 PC4F2                 : 1;
__REG32                       : 1;
__REG32 PC6F2                 : 1;
__REG32 PC7F2                 : 1;
__REG32                       :24;
} __gpiocfr2_bits;

/* GPIOCIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32                       : 7;
__REG32 PC7IS                 : 1;
__REG32                       :24;
} __gpiocis_bits;

/* GPIOCIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32                       : 7;
__REG32 PC7IBE                : 1;
__REG32                       :24;
} __gpiocibe_bits;

/* GPIOCIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level)) */
typedef struct{
__REG32                       : 7;
__REG32 PC7IEV                : 1;
__REG32                       :24;
} __gpiociev_bits;

/* GPIOCIE (Port Interrupt Enable Register) */
typedef struct{
__REG32                       : 7;
__REG32 PC7IE                 : 1;
__REG32                       :24;
} __gpiocie_bits;

/* GPIOCRIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32                       : 7;
__REG32 PC7RIS                : 1;
__REG32                       :24;
} __gpiocris_bits;

/* GPIOCMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32                       : 7;
__REG32 PC7MIS                : 1;
__REG32                       :24;
} __gpiocmis_bits;

/* GPIOCIC (Port Interrupt Clear Register) */
typedef struct{
__REG32                       : 7;
__REG32 PC7IC                 : 1;
__REG32                       :24;
} __gpiocic_bits;

/* GPIOCODE (Port Open-drain Output Enable Register) */
typedef struct{
__REG32                       : 2;
__REG32 PC2ODE                : 1;
__REG32 PC3ODE                : 1;
__REG32 PC4ODE                : 1;
__REG32                       : 1;
__REG32 PC6ODE                : 1;
__REG32 PC7ODE                : 1;
__REG32                       :24;
} __gpiocode_bits;

/* GPIODDATA (Port Data Register) */
typedef struct{
__REG32 PD0                   : 1;
__REG32 PD1                   : 1;
__REG32 PD2                   : 1;
__REG32 PD3                   : 1;
__REG32 PD4                   : 1;
__REG32 PD5                   : 1;
__REG32 PD6                   : 1;
__REG32 PD7                   : 1;
__REG32                       :24;
} __gpioddata_bits;

/* GPIODFR1 (Port Function Register1) */
typedef struct{
__REG32 PD0F1                 : 1;
__REG32 PD1F1                 : 1;
__REG32 PD2F1                 : 1;
__REG32 PD3F1                 : 1;
__REG32 PD4F1                 : 1;
__REG32 PD5F1                 : 1;
__REG32 PD6F1                 : 1;
__REG32 PD7F1                 : 1;
__REG32                       :24;
} __gpiodfr1_bits;

/* GPIODFR2 (Port Function Register2) */
typedef struct{
__REG32                       : 4;
__REG32 PD4F2                 : 1;
__REG32 PD5F2                 : 1;
__REG32 PD6F2                 : 1;
__REG32 PD7F2                 : 1;
__REG32                       :24;
} __gpiodfr2_bits;

/* GPIODIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32                       : 6;
__REG32 PD6IS                 : 1;
__REG32 PD7IS                 : 1;
__REG32                       :24;
} __gpiodis_bits;

/* GPIODIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32                       : 6;
__REG32 PD6IBE                : 1;
__REG32 PD7IBE                : 1;
__REG32                       :24;
} __gpiodibe_bits;

/* GPIODIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level)) */
typedef struct{
__REG32                       : 6;
__REG32 PD6IEV                : 1;
__REG32 PD7IEV                : 1;
__REG32                       :24;
} __gpiodiev_bits;

/* GPIODIE (Port Interrupt Enable Register) */
typedef struct{
__REG32                       : 6;
__REG32 PD6IE                 : 1;
__REG32 PD7IE                 : 1;
__REG32                       :24;
} __gpiodie_bits;

/* GPIODRIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32                       : 6;
__REG32 PD5RIS                : 1;
__REG32 PD7RIS                : 1;
__REG32                       :24;
} __gpiodris_bits;

/* GPIODMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32                       : 6;
__REG32 PD6MIS                : 1;
__REG32 PD7MIS                : 1;
__REG32                       :24;
} __gpiodmis_bits;

/* GPIODIC (Port Interrupt Clear Register) */
typedef struct{
__REG32                       : 6;
__REG32 PD6IC                 : 1;
__REG32 PD7IC                 : 1;
__REG32                       :24;
} __gpiodic_bits;

/* GPIOFDATA (Port Data Register) */
typedef struct{
__REG32                       : 6;
__REG32 PF6                   : 1;
__REG32 PF7                   : 1;
__REG32                       :24;
} __gpiofdata_bits;

/* GPIOFDIR (Port Data Direction Register) */
typedef struct{
__REG32                       : 6;
__REG32 PF6C                  : 1;
__REG32 PF7C                  : 1;
__REG32                       :24;
} __gpiofdir_bits;

/* GPIOFFR1 (Port Function Register1) */
typedef struct{
__REG32                       : 6;
__REG32 PF6F1                 : 1;
__REG32 PF7F1                 : 1;
__REG32                       :24;
} __gpioffr1_bits;

/* GPIOFFR2 (Port Function Register2) */
typedef struct{
__REG32                       : 6;
__REG32 PF6F2                 : 1;
__REG32 PF7F2                 : 1;
__REG32                       :24;
} __gpioffr2_bits;

/* GPIOFIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32                       : 7;
__REG32 PF7IS                 : 1;
__REG32                       :24;
} __gpiofis_bits;

/* GPIOFIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32                       : 7;
__REG32 PF7IBE                : 1;
__REG32                       :24;
} __gpiofibe_bits;

/* GPIOFIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level)) */
typedef struct{
__REG32                       : 7;
__REG32 PF7IEV                : 1;
__REG32                       :24;
} __gpiofiev_bits;

/* GPIOFIE (Port Interrupt Enable Register) */
typedef struct{
__REG32                       : 7;
__REG32 PF7IE                 : 1;
__REG32                       :24;
} __gpiofie_bits;

/* GPIOFRIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32                       : 7;
__REG32 PF7RIS                : 1;
__REG32                       :24;
} __gpiofris_bits;

/* GPIOFMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32                       : 7;
__REG32 PF7MIS                : 1;
__REG32                       :24;
} __gpiofmis_bits;

/* GPIOFIC (Port Interrupt Clear Register) */
typedef struct{
__REG32                       : 7;
__REG32 PF7IC                 : 1;
__REG32                       :24;
} __gpiofic_bits;

/* GPIOFODE (Port Open-drain Output Enable Register) */
typedef struct{
__REG32                       : 6;
__REG32 PF6ODE                : 1;
__REG32 PF7ODE                : 1;
__REG32                       :24;
} __gpiofode_bits;

/* GPIOGDATA (Port Data Register) */
typedef struct{
__REG32 PG0                   : 1;
__REG32 PG1                   : 1;
__REG32 PG2                   : 1;
__REG32 PG3                   : 1;
__REG32 PG4                   : 1;
__REG32 PG5                   : 1;
__REG32 PG6                   : 1;
__REG32 PG7                   : 1;
__REG32                       :24;
} __gpiogdata_bits;

/* GPIOGDIR (Port Data Direction Register) */
typedef struct{
__REG32 PG0C                  : 1;
__REG32 PG1C                  : 1;
__REG32 PG2C                  : 1;
__REG32 PG3C                  : 1;
__REG32 PG4C                  : 1;
__REG32 PG5C                  : 1;
__REG32 PG6C                  : 1;
__REG32 PG7C                  : 1;
__REG32                       :24;
} __gpiogdir_bits;

/* GPIOGFR1 (Port Function Register1) */
typedef struct{
__REG32 PG0F1                 : 1;
__REG32 PG1F1                 : 1;
__REG32 PG2F1                 : 1;
__REG32 PG3F1                 : 1;
__REG32 PG4F1                 : 1;
__REG32 PG5F1                 : 1;
__REG32 PG6F1                 : 1;
__REG32 PG7F1                 : 1;
__REG32                       :24;
} __gpiogfr1_bits;

/* GPIOJDATA (Port Data Register) */
typedef struct{
__REG32 PJ0                   : 1;
__REG32 PJ1                   : 1;
__REG32 PJ2                   : 1;
__REG32 PJ3                   : 1;
__REG32 PJ4                   : 1;
__REG32 PJ5                   : 1;
__REG32 PJ6                   : 1;
__REG32 PJ7                   : 1;
__REG32                       :24;
} __gpiojdata_bits;

/* GPIOJDATA (Port Data Register) */
typedef struct{
__REG32 PJ0C                  : 1;
__REG32 PJ1C                  : 1;
__REG32 PJ2C                  : 1;
__REG32 PJ3C                  : 1;
__REG32 PJ4C                  : 1;
__REG32 PJ5C                  : 1;
__REG32 PJ6C                  : 1;
__REG32 PJ7C                  : 1;
__REG32                       :24;
} __gpiojdir_bits;

/* GPIOJFR1 (Port Function Register1) */
typedef struct{
__REG32 PJ0F1                 : 1;
__REG32 PJ1F1                 : 1;
__REG32 PJ2F1                 : 1;
__REG32 PJ3F1                 : 1;
__REG32 PJ4F1                 : 1;
__REG32 PJ5F1                 : 1;
__REG32 PJ6F1                 : 1;
__REG32 PJ7F1                 : 1;
__REG32                       :24;
} __gpiojfr1_bits;

/* GPIOJFR2 (Port J Function Register2) */
typedef struct{
__REG32                       : 4;
__REG32 PJ4F2                 : 1;
__REG32 PJ5F2                 : 1;
__REG32 PJ6F2                 : 1;
__REG32 PJ7F2                 : 1;
__REG32                       :24;
} __gpiojfr2_bits;

/* GPIOKDATA (Port Data Register) */
typedef struct{
__REG32 PK0                   : 1;
__REG32 PK1                   : 1;
__REG32 PK2                   : 1;
__REG32 PK3                   : 1;
__REG32 PK4                   : 1;
__REG32 PK5                   : 1;
__REG32 PK6                   : 1;
__REG32 PK7                   : 1;
__REG32                       :24;
} __gpiokdata_bits;

/* GPIOKDIR (Port K Data Direction Register) */
typedef struct{
__REG32 PK0C                  : 1;
__REG32 PK1C                  : 1;
__REG32 PK2C                  : 1;
__REG32 PK3C                  : 1;
__REG32 PK4C                  : 1;
__REG32 PK5C                  : 1;
__REG32 PK6C                  : 1;
__REG32 PK7C                  : 1;
__REG32                       :24;
} __gpiokdir_bits;

/* GPIOKFR1 (Port Function Register1) */
typedef struct{
__REG32 PK0F1                 : 1;
__REG32 PK1F1                 : 1;
__REG32 PK2F1                 : 1;
__REG32 PK3F1                 : 1;
__REG32 PK4F1                 : 1;
__REG32 PK5F1                 : 1;
__REG32 PK6F1                 : 1;
__REG32 PK7F1                 : 1;
__REG32                       :24;
} __gpiokfr1_bits;

/* GPIOKFR2 (Port Function Register2) */
typedef struct{
__REG32 PK0F2                 : 1;
__REG32 PK1F2                 : 1;
__REG32 PK2F2                 : 1;
__REG32 PK3F2                 : 1;
__REG32 PK4F2                 : 1;
__REG32 PK5F2                 : 1;
__REG32 PK6F2                 : 1;
__REG32 PK7F2                 : 1;
__REG32                       :24;
} __gpiokfr2_bits;

/* GPIOLDATA (Port Data Register) */
typedef struct{
__REG32 PL0                   : 1;
__REG32 PL1                   : 1;
__REG32 PL2                   : 1;
__REG32 PL3                   : 1;
__REG32 PL4                   : 1;
__REG32                       :27;
} __gpioldata_bits;

/* GPIOLDIR (Port Data Direction Register) */
typedef struct{
__REG32 PL0C                  : 1;
__REG32 PL1C                  : 1;
__REG32 PL2C                  : 1;
__REG32 PL3C                  : 1;
__REG32 PL4C                  : 1;
__REG32                       :27;
} __gpioldir_bits;

/* GPIOLFR1 (Port Function Register1) */
typedef struct{
__REG32 PL0F1                 : 1;
__REG32 PL1F1                 : 1;
__REG32 PL2F1                 : 1;
__REG32 PL3F1                 : 1;
__REG32 PL4F1                 : 1;
__REG32                       :27;
} __gpiolfr1_bits;

/* GPIOLFR2 (Port Function Register1) */
typedef struct{
__REG32 PL0F2                 : 1;
__REG32 PL1F2                 : 1;
__REG32 PL2F2                 : 1;
__REG32 PL3F2                 : 1;
__REG32                       :28;
} __gpiolfr2_bits;

/* GPIOMDATA (Port Data Register) */
typedef struct{
__REG32 PM0                   : 1;
__REG32 PM1                   : 1;
__REG32 PM2                   : 1;
__REG32 PM3                   : 1;
__REG32                       :28;
} __gpiomdata_bits;

/* GPIOMDIR (Port Data Direction Register) */
typedef struct{
__REG32 PM0C                  : 1;
__REG32 PM1C                  : 1;
__REG32 PM2C                  : 1;
__REG32 PM3C                  : 1;
__REG32                       :28;
} __gpiomdir_bits;

/* GPIOMFR1 (Port Function Register1) */
typedef struct{
__REG32 PM0F1                 : 1;
__REG32 PM1F1                 : 1;
__REG32 PM2F1                 : 1;
__REG32 PM3F1                 : 1;
__REG32                       :28;
} __gpiomfr1_bits;

/* GPIONDATA (Port Data Register) */
typedef struct{
__REG32 PN0                   : 1;
__REG32 PN1                   : 1;
__REG32 PN2                   : 1;
__REG32 PN3                   : 1;
__REG32 PN4                   : 1;
__REG32 PN5                   : 1;
__REG32 PN6                   : 1;
__REG32 PN7                   : 1;
__REG32                       :24;
} __gpiondata_bits;

/* GPIONDIR (Port Data Direction Register) */
typedef struct{
__REG32 PN0C                  : 1;
__REG32 PN1C                  : 1;
__REG32 PN2C                  : 1;
__REG32 PN3C                  : 1;
__REG32 PN4C                  : 1;
__REG32 PN5C                  : 1;
__REG32 PN6C                  : 1;
__REG32 PN7C                  : 1;
__REG32                       :24;
} __gpiondir_bits;

/* GPIONFR1 (Port Function Register1) */
typedef struct{
__REG32 PN0F1                 : 1;
__REG32                       : 1;
__REG32 PN2F1                 : 1;
__REG32 PN3F1                 : 1;
__REG32 PN4F1                 : 1;
__REG32 PN5F1                 : 1;
__REG32 PN6F1                 : 1;
__REG32 PN7F1                 : 1;
__REG32                       :24;
} __gpionfr1_bits;

/* GPIONFR2 (Port Function Register1) */
typedef struct{
__REG32 PN0F2                 : 1;
__REG32 PN1F2                 : 1;
__REG32                       :30;
} __gpionfr2_bits;

/* GPIONIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32                       : 4;
__REG32 PN4IS                 : 1;
__REG32 PN5IS                 : 1;
__REG32 PN6IS                 : 1;
__REG32 PN7IS                 : 1;
__REG32                       :24;
} __gpionis_bits;

/* GPIONIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32                       : 4;
__REG32 PN4IBE                : 1;
__REG32 PN5IBE                : 1;
__REG32 PN6IBE                : 1;
__REG32 PN7IBE                : 1;
__REG32                       :24;
} __gpionibe_bits;

/* GPIONIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level)) */
typedef struct{
__REG32                       : 4;
__REG32 PN4IEV                : 1;
__REG32 PN5IEV                : 1;
__REG32 PN6IEV                : 1;
__REG32 PN7IEV                : 1;
__REG32                       :24;
} __gpioniev_bits;

/* GPIONIE (Port Interrupt Enable Register) */
typedef struct{
__REG32                       : 4;
__REG32 PN4IE                 : 1;
__REG32 PN5IE                 : 1;
__REG32 PN6IE                 : 1;
__REG32 PN7IE                 : 1;
__REG32                       :24;
} __gpionie_bits;

/* GPIONRIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32                       : 4;
__REG32 PN4RIS                : 1;
__REG32 PN5RIS                : 1;
__REG32 PN6RIS                : 1;
__REG32 PN7RIS                : 1;
__REG32                       :24;
} __gpionris_bits;

/* GPIONMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32                       : 4;
__REG32 PN4MIS                : 1;
__REG32 PN5MIS                : 1;
__REG32 PN6MIS                : 1;
__REG32 PN7MIS                : 1;
__REG32                       :24;
} __gpionmis_bits;

/* GPIONIC (Port Interrupt Clear Register) */
typedef struct{
__REG32                       : 4;
__REG32 PN4IC                 : 1;
__REG32 PN5IC                 : 1;
__REG32 PN6IC                 : 1;
__REG32 PN7IC                 : 1;
__REG32                       :24;
} __gpionic_bits;

/* GPIOPDATA (Port Data Register) */
typedef struct{
__REG32 PP0                   : 1;
__REG32 PP1                   : 1;
__REG32 PP2                   : 1;
__REG32 PP3                   : 1;
__REG32 PP4                   : 1;
__REG32 PP5                   : 1;
__REG32 PP6                   : 1;
__REG32 PP7                   : 1;
__REG32                       :24;
} __gpiopdata_bits;

/* GPIORDATA (Port Data Register) */
typedef struct{
__REG32 PR0                   : 1;
__REG32 PR1                   : 1;
__REG32 PR2                   : 1;
__REG32                       :29;
} __gpiordata_bits;

/* GPIORDIR (Port Data Direction Register) */
typedef struct{
__REG32                       : 2;
__REG32 PR2C                  : 1;
__REG32                       :29;
} __gpiordir_bits;

/* GPIORFR1 (Port Function Register1) */
typedef struct{
__REG32 PR0F1                 : 1;
__REG32 PR1F1                 : 1;
__REG32                       :30;
} __gpiorfr1_bits;

/* GPIORFR2 (Port Function Register1) */
typedef struct{
__REG32                       : 1;
__REG32 PR1F2                 : 1;
__REG32                       :30;
} __gpiorfr2_bits;

/* GPIORIS (Port Interrupt Selection Register (Level and Edge)) */
typedef struct{
__REG32                       : 2;
__REG32 PR2IS                 : 1;
__REG32                       :29;
} __gpioris_bits;

/* GPIORIBE (Port Interrupt Selection Register (Fellow edge and Both edge)) */
typedef struct{
__REG32                       : 2;
__REG32 PR2IBE                : 1;
__REG32                       :29;
} __gpioribe_bits;

/* GPIORIEV (Port Interrupt Selection Register (Fall down edge/Low level and Rising up edge/High level)) */
typedef struct{
__REG32                       : 2;
__REG32 PR2IEV                : 1;
__REG32                       :29;
} __gpioriev_bits;

/* GPIORIE (Port Interrupt Enable Register) */
typedef struct{
__REG32                       : 2;
__REG32 PR2IE                 : 1;
__REG32                       :29;
} __gpiorie_bits;

/* GPIORRIS (Port Interrupt Status Register (Raw)) */
typedef struct{
__REG32                       : 2;
__REG32 PR2RIS                : 1;
__REG32                       :29;
} __gpiorris_bits;

/* GPIORMIS (Port Interrupt Status Register (Masked)) */
typedef struct{
__REG32                       : 2;
__REG32 PR2MIS                : 1;
__REG32                       :29;
} __gpiormis_bits;

/* GPIORIC (Port Interrupt Clear Register) */
typedef struct{
__REG32                       : 2;
__REG32 PR2IC                 : 1;
__REG32                       :29;
} __gpioric_bits;

/* GPIOTDATA (Port Data Register) */
typedef struct{
__REG32 PT0                   : 1;
__REG32 PT1                   : 1;
__REG32 PT2                   : 1;
__REG32 PT3                   : 1;
__REG32 PT4                   : 1;
__REG32 PT5                   : 1;
__REG32 PT6                   : 1;
__REG32 PT7                   : 1;
__REG32                       :24;
} __gpiotdata_bits;

/* GPIOTDIR (Port Data Direction Register) */
typedef struct{
__REG32 PT0C                  : 1;
__REG32 PT1C                  : 1;
__REG32 PT2C                  : 1;
__REG32 PT3C                  : 1;
__REG32 PT4C                  : 1;
__REG32 PT5C                  : 1;
__REG32 PT6C                  : 1;
__REG32 PT7C                  : 1;
__REG32                       :24;
} __gpiotdir_bits;

/* GPIOTFR1 (Port Function Register1) */
typedef struct{
__REG32 PT0F1                 : 1;
__REG32 PT1F1                 : 1;
__REG32 PT2F1                 : 1;
__REG32 PT3F1                 : 1;
__REG32 PT4F1                 : 1;
__REG32 PT5F1                 : 1;
__REG32 PT6F1                 : 1;
__REG32 PT7F1                 : 1;
__REG32                       :24;
} __gpiotfr1_bits;

/* GPIOTFR2 (Port Function Register2) */
typedef struct{
__REG32 PT0F2                 : 1;
__REG32 PT1F2                 : 1;
__REG32 PT2F2                 : 1;
__REG32 PT3F2                 : 1;
__REG32 PT4F2                 : 1;
__REG32 PT5F2                 : 1;
__REG32 PT6F2                 : 1;
__REG32 PT7F2                 : 1;
__REG32                       :24;
} __gpiotfr2_bits;

/* GPIOUDATA (Port Data Register) */
typedef struct{
__REG32 PU0                   : 1;
__REG32 PU1                   : 1;
__REG32 PU2                   : 1;
__REG32 PU3                   : 1;
__REG32 PU4                   : 1;
__REG32 PU5                   : 1;
__REG32 PU6                   : 1;
__REG32 PU7                   : 1;
__REG32                       :24;
} __gpioudata_bits;

/* GPIOUDIR (Port Data Direction Register) */
typedef struct{
__REG32 PU0C                  : 1;
__REG32 PU1C                  : 1;
__REG32 PU2C                  : 1;
__REG32 PU3C                  : 1;
__REG32 PU4C                  : 1;
__REG32 PU5C                  : 1;
__REG32 PU6C                  : 1;
__REG32 PU7C                  : 1;
__REG32                       :24;
} __gpioudir_bits;

/* GPIOUFR1 (Port Function Register1) */
typedef struct{
__REG32 PU0F1                 : 1;
__REG32 PU1F1                 : 1;
__REG32 PU2F1                 : 1;
__REG32 PU3F1                 : 1;
__REG32 PU4F1                 : 1;
__REG32 PU5F1                 : 1;
__REG32 PU6F1                 : 1;
__REG32 PU7F1                 : 1;
__REG32                       :24;
} __gpioufr1_bits;

/* GPIOUFR2 (Port Function Register2) */
typedef struct{
__REG32 PU0F2                 : 1;
__REG32 PU1F2                 : 1;
__REG32 PU2F2                 : 1;
__REG32 PU3F2                 : 1;
__REG32 PU4F2                 : 1;
__REG32 PU5F2                 : 1;
__REG32 PU6F2                 : 1;
__REG32 PU7F2                 : 1;
__REG32                       :24;
} __gpioufr2_bits;

/* GPIOVDATA (Port Data Register) */
typedef struct{
__REG32 PV0                   : 1;
__REG32 PV1                   : 1;
__REG32 PV2                   : 1;
__REG32 PV3                   : 1;
__REG32 PV4                   : 1;
__REG32 PV5                   : 1;
__REG32 PV6                   : 1;
__REG32 PV7                   : 1;
__REG32                       :24;
} __gpiovdata_bits;

/* GPIOVDIR (Port Data Direction Register) */
typedef struct{
__REG32 PV0C                  : 1;
__REG32 PV1C                  : 1;
__REG32 PV2C                  : 1;
__REG32 PV3C                  : 1;
__REG32 PV4C                  : 1;
__REG32 PV5C                  : 1;
__REG32 PV6C                  : 1;
__REG32 PV7C                  : 1;
__REG32                       :24;
} __gpiovdir_bits;

/* GPIOVFR1 (Port Function Register1) */
typedef struct{
__REG32 PV0F1                 : 1;
__REG32 PV1F1                 : 1;
__REG32 PV2F1                 : 1;
__REG32 PV3F1                 : 1;
__REG32 PV4F1                 : 1;
__REG32 PV5F1                 : 1;
__REG32 PV6F1                 : 1;
__REG32 PV7F1                 : 1;
__REG32                       :24;
} __gpiovfr1_bits;

/* GPIOVFR2 (Port Function Register2) */
typedef struct{
__REG32 PV0F2                 : 1;
__REG32 PV1F2                 : 1;
__REG32 PV2F2                 : 1;
__REG32 PV3F2                 : 1;
__REG32 PV4F2                 : 1;
__REG32 PV5F2                 : 1;
__REG32 PV6F2                 : 1;
__REG32 PV7F2                 : 1;
__REG32                       :24;
} __gpiovfr2_bits;

/* dmc_memc_status (DMC Memory Controller Status Register) */
typedef struct{
__REG32 memc_status           : 2;
__REG32 memory_width          : 2;
__REG32 memory_ddr            : 3;
__REG32                       : 2;
__REG32 memory_banks          : 1;
__REG32                       :22;
} __dmc_memc_status_bits;

/* dmc_memc_cmd (DMC Memory Controller Command Register) */
typedef struct{
__REG32 memc_cmd              : 3;
__REG32                       :29;
} __dmc_memc_cmd_bits;

/* dmc_direct_cmd (DMC Direct Command Register) */
typedef struct{
__REG32 addr_13_to_0          :14;
__REG32                       : 2;
__REG32 bank_addr             : 2;
__REG32 memory_cmd            : 2;
__REG32 chip_nmbr             : 2;
__REG32                       :10;
} __dmc_direct_cmd_bits;

/* dmc_memory_cfg (DMC Memory Configuration Register) */
typedef struct{
__REG32 column_bits           : 3;
__REG32 row_bits              : 3;
__REG32 ap_bit                : 1;
__REG32 power_down_prd        : 6;
__REG32 auto_power_down       : 1;
__REG32 stop_mem_clock        : 1;
__REG32 memory_burst          : 3;
__REG32                       : 3;
__REG32 active_chips          : 2;
__REG32                       : 9;
} __dmc_memory_cfg_bits;

/* dmc_refresh_prd (DMC Refresh Period Register) */
typedef struct{
__REG32 refresh_prd           :15;
__REG32                       :17;
} __dmc_refresh_prd_bits;

/* dmc_cas_latency_3 (DMC CAS latency Register) */
typedef struct{
__REG32                       : 1;
__REG32 cas_latency           : 3;
__REG32                       :28;
} __dmc_cas_latency_3_bits;

/* dmc_cas_latency_5 (DMC CAS latency Register) */
typedef struct{
__REG32 cas_half_cycle        : 1;
__REG32 cas_latency           : 3;
__REG32                       :28;
} __dmc_refresh_prd_5_bits;

/* dmc_t_dqss (DMC t_dqss Register) */
typedef struct{
__REG32 t_dqss                : 2;
__REG32                       :30;
} __dmc_t_dqss_bits;

/* dmc_t_mrd (DMC t_mrd Register) */
typedef struct{
__REG32 t_mrd                 : 7;
__REG32                       :25;
} __dmc_t_mrd_bits;

/* dmc_t_ras (DMC t_ras Register) */
typedef struct{
__REG32 t_ras                 : 4;
__REG32                       :28;
} __dmc_t_ras_bits;

/* dmc_t_rc (DMC t_rc Register) */
typedef struct{
__REG32 t_rc                  : 4;
__REG32                       :28;
} __dmc_t_rc_bits;

/* dmc_t_rcd (DMC t_rcd Register) */
typedef struct{
__REG32 t_rcd                 : 3;
__REG32 schedule_rcd          : 3;
__REG32                       :26;
} __dmc_t_rcd_bits;

/* dmc_t_rfc (DMC t_rfc Register) */
typedef struct{
__REG32 t_rfc                 : 5;
__REG32 schedule_rfc          : 5;
__REG32                       :22;
} __dmc_t_rfc_bits;

/* dmc t_rp (DMC t_rp Register) */
typedef struct{
__REG32 t_rp                  : 3;
__REG32 schedule_rp           : 3;
__REG32                       :26;
} __dmc_t_rp_bits;

/* dmc t_rp (DMC t_rp Register) */
typedef struct{
__REG32 t_rrd                 : 4;
__REG32                       :28;
} __dmc_t_rrd_bits;

/* dmc_t_wr (DMC t_wr Register) */
typedef struct{
__REG32 t_wr                  : 3;
__REG32                       :29;
} __dmc_t_wr_bits;

/* dmc_t_wtr (DMC t_wtr Register) */
typedef struct{
__REG32 t_wtr                 : 3;
__REG32                       :29;
} __dmc_t_wtr_bits;

/* dmc t_xp (DMC t_xp Register) */
typedef struct{
__REG32 t_xp                  : 8;
__REG32                       :24;
} __dmc_t_xp_bits;

/* dmc t_xsr (DMC t_xsr Register) */
typedef struct{
__REG32 t_xsr                 : 8;
__REG32                       :24;
} __dmc_t_xsr_bits;

/* dmc_t_esr (DMC t_esr Register) */
typedef struct{
__REG32 t_esr                 : 8;
__REG32                       :24;
} __dmc_t_esr_bits;

/* dmc_id_<0-5>_cfg (DMC id_<0-5>_cfg Registers) */
typedef struct{
__REG32 qos_enable            : 1;
__REG32 qos_min               : 1;
__REG32 qos_max               : 8;
__REG32                       :22;
} __dmc_id_cfg_3_bits;

/* dmc_id_<0-5>_cfg (DMC id_<0-5>_cfg Registers) */
typedef struct{
__REG32 qos_enable            : 1;
__REG32                       : 1;
__REG32 qos_max               : 8;
__REG32                       :22;
} __dmc_id_cfg_5_bits;

/* dmc_chip_0_cfg (DMC chip_0_cfg Registers) */
typedef struct{
__REG32 address_mask          : 8;
__REG32 address_match         : 8;
__REG32 brc_n_rbc             : 1;
__REG32                       :15;
} __dmc_chip_0_cfg_bits;

/* dmc_user_config (DMC user_config Register) */
typedef struct{
__REG32 sdr_width             : 1;
__REG32                       : 3;
__REG32 dmclk_out1            : 3;
__REG32                       :25;
} __dmc_user_config_bits;

/* dmc_user_config_5 (DMC user_config Register) */
typedef struct{
__REG32 sdr_width             : 1;
__REG32 dmc_clk_in            : 3;
__REG32 dqs_in                : 3;
__REG32                       :25;
} __dmc_user_config_5_bits;

/* smc_memc_status (SMC Memory Controller Status Register) */
typedef struct{
__REG32 state                 : 1;
__REG32                       :31;
} __smc_memc_status_bits;

/* smc_memif_cfg (SMC Memory Interface Configuration Register) */
typedef struct{
__REG32 memory_type0          : 2;
__REG32 memory_chips0         : 2;
__REG32 memory_width0         : 2;
__REG32                       :26;
} __smc_memif_cfg_bits;

/* smc_direct_cmd (SMC Direct Command Register) */
typedef struct{
__REG32 addr                  :20;
__REG32                       : 1;
__REG32 cmd_type              : 2;
__REG32 chip_select           : 3;
__REG32                       : 6;
} __smc_direct_cmd_bits;

/* smc_set_cycles (SMC Set Cycles Register) */
typedef struct{
__REG32 Set_t0                : 4;
__REG32 Set_t1                : 4;
__REG32 Set_t2                : 3;
__REG32 Set_t3                : 3;
__REG32 Set_t4                : 3;
__REG32 Set_t5                : 3;
__REG32                       :12;
} __smc_set_cycles_bits;

/* smc_set_opmode (SMC Set Opmode Register) */
typedef struct{
__REG32 set_mw                : 2;
__REG32 set_rd_sync           : 1;
__REG32 set_rd_bl             : 3;
__REG32 set_wr_sync           : 1;
__REG32 set_wr_bl             : 3;
__REG32                       : 1;
__REG32 set_adv               : 1;
__REG32 set_bls               : 1;
__REG32 set_burst_align       : 3;
__REG32                       :16;
} __smc_set_opmode_bits;

/* smc_sram_cycles0_n (SMC SRAM Cycles Registers 0 <0..3>) */
typedef struct{
__REG32 t_rc                  : 4;
__REG32 t_wc                  : 4;
__REG32 t_ceoe                : 3;
__REG32 t_wp                  : 3;
__REG32 t_pc                  : 3;
__REG32 t_tr                  : 3;
__REG32                       :12;
} __smc_sram_cycles0_bits;

/* smc_opmode0_n (SMC Opmode Registers 0<0..3>) */
typedef struct{
__REG32 mw                    : 2;
__REG32 rd_sync               : 1;
__REG32 rd_bl                 : 3;
__REG32 wr_sync               : 1;
__REG32 wr_bl                 : 3;
__REG32                       : 1;
__REG32 adv                   : 1;
__REG32 bls                   : 1;
__REG32 burst_align           : 3;
__REG32 address_mask          : 8;
__REG32 address_match         : 8;
} __smc_opmode0_bits;

/* UARTxDR (UART x Data Register) */
typedef struct{
__REG32 DATA                  : 8;
__REG32 FE                    : 1;
__REG32 PE                    : 1;
__REG32 BE                    : 1;
__REG32 OE                    : 1;
__REG32                       :20;
} __uartdr_bits;

/* UART0RSR (UART x receive status register (read)) */
typedef struct
{
  __REG32 FE                    : 1;
  __REG32 PE                    : 1;
  __REG32 BE                    : 1;
  __REG32 OE                    : 1;
  __REG32                       :28;
} __uartrsr_bits;

/* UART0FR (UART0 flag register) */
typedef struct
{
  __REG32 CTS                   : 1;
  __REG32 DSR                   : 1;
  __REG32 DCD                   : 1;
  __REG32 BUSY                  : 1;
  __REG32 RXFE                  : 1;
  __REG32 TXFF                  : 1;
  __REG32 RXFF                  : 1;
  __REG32 TXFE                  : 1;
  __REG32 RI                    : 1;
  __REG32                       :23;
} __uart0fr_bits;

/* UART1FR (UART1 flag register) */
typedef struct
{
  __REG32 CTS                   : 1;
  __REG32                       : 2;
  __REG32 BUSY                  : 1;
  __REG32 RXFE                  : 1;
  __REG32 TXFF                  : 1;
  __REG32 RXFF                  : 1;
  __REG32 TXFE                  : 1;
  __REG32                       :24;
} __uart1fr_bits;

/* UART2FR (UART2 flag register) */
typedef struct
{
  __REG32                       : 3;
  __REG32 BUSY                  : 1;
  __REG32 RXFE                  : 1;
  __REG32 TXFF                  : 1;
  __REG32 RXFF                  : 1;
  __REG32 TXFE                  : 1;
  __REG32                       :24;
} __uart2fr_bits;

/* UARTnILPR (UARTn IrDA low-power counter register) */
typedef struct
{
  __REG32 ILPDVSR               : 8;
  __REG32                       :24;
} __uartilpr_bits;

/* UARTnIBRD (UARTn integer baud rate divisor register) */
typedef struct
{
  __REG32 BAUDDIVINT            :16;
  __REG32                       :16;
} __uartibrd_bits;

/* UARTnFBRD (UARTn fractional baud rate divisor register) */
typedef struct
{
  __REG32 BAUDDIVFRAC           : 6;
  __REG32                       :26;
} __uartfbrd_bits;

/* UARTnLCR_H (UARTn line control register) */
typedef struct
{
  __REG32 BRK                   : 1;
  __REG32 PEN                   : 1;
  __REG32 EPS                   : 1;
  __REG32 STP2                  : 1;
  __REG32 FEN                   : 1;
  __REG32 WLEN                  : 2;
  __REG32 SPS                   : 1;
  __REG32                       :24;
} __uartlcr_h_bits;

/* UART0CR (UART0 control register) */
typedef struct
{
  __REG32 UARTEN                : 1;
  __REG32 SIREN                 : 1;
  __REG32 SIRLP                 : 1;
  __REG32                       : 5;
  __REG32 TXE                   : 1;
  __REG32 RXE                   : 1;
  __REG32 DTR                   : 1;
  __REG32 RTS                   : 1;
  __REG32                       : 2;
  __REG32 RTSEn                 : 1;
  __REG32 CTSEn                 : 1;
  __REG32                       :16;
} __uart0cr_bits;

/* UART1CR (UART1 control register) */
typedef struct
{
  __REG32 UARTEN                : 1;
  __REG32                       : 7;
  __REG32 TXE                   : 1;
  __REG32 RXE                   : 1;
  __REG32                       : 5;
  __REG32 CTSEn                 : 1;
  __REG32                       :16;
} __uart1cr_bits;

/* UART2CR (UART2 control register) */
typedef struct
{
  __REG32 UARTEN                : 1;
  __REG32                       : 7;
  __REG32 TXE                   : 1;
  __REG32 RXE                   : 1;
  __REG32                       :22;
} __uart2cr_bits;

/* UARTnIFLS (UARTn interrupt FIFO level select register) */
typedef struct
{
  __REG32 TXIFLSEL              : 3;
  __REG32 RXIFLSEL              : 3;
  __REG32                       :26;
} __uartifls_bits;

/* UART0IMSC (UART0 interrupt mask set/clear register) */
typedef struct
{
  __REG32 RIMIM                 : 1;
  __REG32 CTSMIM                : 1;
  __REG32 DCDMIM                : 1;
  __REG32 DSRMIM                : 1;
  __REG32 RXIM                  : 1;
  __REG32 TXIM                  : 1;
  __REG32 RTIM                  : 1;
  __REG32 FEIM                  : 1;
  __REG32 PEIM                  : 1;
  __REG32 BEIM                  : 1;
  __REG32 OEIM                  : 1;
  __REG32                       :21;
} __uart0imsc_bits;

/* UART1IMSC (UART1 interrupt mask set/clear register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMIM                : 1;
  __REG32                       : 2;
  __REG32 RXIM                  : 1;
  __REG32 TXIM                  : 1;
  __REG32 RTIM                  : 1;
  __REG32 FEIM                  : 1;
  __REG32 PEIM                  : 1;
  __REG32 BEIM                  : 1;
  __REG32 OEIM                  : 1;
  __REG32                       :21;
} __uart1imsc_bits;

/* UART2IMSC (UART2 interrupt mask set/clear register) */
typedef struct
{
  __REG32                       : 4;
  __REG32 RXIM                  : 1;
  __REG32 TXIM                  : 1;
  __REG32 RTIM                  : 1;
  __REG32 FEIM                  : 1;
  __REG32 PEIM                  : 1;
  __REG32 BEIM                  : 1;
  __REG32 OEIM                  : 1;
  __REG32                       :21;
} __uart2imsc_bits;

/* UART0RIS (UART0 raw interrupt status register) */
typedef struct
{
  __REG32 RIRMIS                : 1;
  __REG32 CTSRMIS               : 1;
  __REG32 DCDRMIS               : 1;
  __REG32 DSRRMIS               : 1;
  __REG32 RXRIS                 : 1;
  __REG32 TXRIS                 : 1;
  __REG32 RTRIM                 : 1;
  __REG32 FERIM                 : 1;
  __REG32 PERIM                 : 1;
  __REG32 BERIM                 : 1;
  __REG32 OERIM                 : 1;
  __REG32                       :21;
} __uart0ris_bits;

/* UART1RIS (UART1 raw interrupt status register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSRMIM               : 1;
  __REG32                       : 2;
  __REG32 RXRIM                 : 1;
  __REG32 TXRIM                 : 1;
  __REG32 RTRIM                 : 1;
  __REG32 FERIM                 : 1;
  __REG32 PERIM                 : 1;
  __REG32 BERIM                 : 1;
  __REG32 OERIM                 : 1;
  __REG32                       :21;
} __uart1ris_bits;

/* UART2RIS (UART2 raw interrupt status register) */
typedef struct
{
  __REG32                       : 4;
  __REG32 RXRIM                 : 1;
  __REG32 TXRIM                 : 1;
  __REG32 RTRIM                 : 1;
  __REG32 FERIM                 : 1;
  __REG32 PERIM                 : 1;
  __REG32 BERIM                 : 1;
  __REG32 OERIM                 : 1;
  __REG32                       :21;
} __uart2ris_bits;

/* UART0MIS (UART0 masked interrupt status register) */
typedef struct
{
  __REG32 RIMMIS                : 1;
  __REG32 CTSMMIS               : 1;
  __REG32 DCDMMIS               : 1;
  __REG32 DSRMMIS               : 1;
  __REG32 RXMIS                 : 1;
  __REG32 TXMIS                 : 1;
  __REG32 RTMIM                 : 1;
  __REG32 FEMIM                 : 1;
  __REG32 PEMIM                 : 1;
  __REG32 BEMIM                 : 1;
  __REG32 OEMIM                 : 1;
  __REG32                       :21;
} __uart0mis_bits;

/* UART1MIS (UART1 masked interrupt status register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMMIM               : 1;
  __REG32                       : 2;
  __REG32 RXMIM                 : 1;
  __REG32 TXMIM                 : 1;
  __REG32 RTMIM                 : 1;
  __REG32 FEMIM                 : 1;
  __REG32 PEMIM                 : 1;
  __REG32 BEMIM                 : 1;
  __REG32 OEMIM                 : 1;
  __REG32                       :21;
} __uart1mis_bits;

/* UART2MIS (UART2 masked interrupt status register) */
typedef struct
{
  __REG32                       : 4;
  __REG32 RXMIM                 : 1;
  __REG32 TXMIM                 : 1;
  __REG32 RTMIM                 : 1;
  __REG32 FEMIM                 : 1;
  __REG32 PEMIM                 : 1;
  __REG32 BEMIM                 : 1;
  __REG32 OEMIM                 : 1;
  __REG32                       :21;
} __uart2mis_bits;

/* UART0ICR (UART0 interrupt clear register) */
typedef struct
{
  __REG32 RIMIC                 : 1;
  __REG32 CTSMIC                : 1;
  __REG32 DCDMIC                : 1;
  __REG32 DSRMIC                : 1;
  __REG32 RXIC                  : 1;
  __REG32 TXIC                  : 1;
  __REG32 RTIC                  : 1;
  __REG32 FEIC                  : 1;
  __REG32 PEIC                  : 1;
  __REG32 BEIC                  : 1;
  __REG32 OEIC                  : 1;
  __REG32                       :21;
} __uart0icr_bits;

/* UART1ICR (UART1 interrupt clear register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 CTSMIC                : 1;
  __REG32                       : 2;
  __REG32 RXIC                  : 1;
  __REG32 TXIC                  : 1;
  __REG32 RTIC                  : 1;
  __REG32 FEIC                  : 1;
  __REG32 PEIC                  : 1;
  __REG32 BEIC                  : 1;
  __REG32 OEIC                  : 1;
  __REG32                       :21;
} __uart1icr_bits;

/* UART2ICR (UART2 interrupt clear register) */
typedef struct
{
  __REG32                       : 4;
  __REG32 RXIC                  : 1;
  __REG32 TXIC                  : 1;
  __REG32 RTIC                  : 1;
  __REG32 FEIC                  : 1;
  __REG32 PEIC                  : 1;
  __REG32 BEIC                  : 1;
  __REG32 OEIC                  : 1;
  __REG32                       :21;
} __uart2icr_bits;

/* UART0DMACR (UART0 DMA control register) */
typedef struct
{
  __REG32 RXDMAE                : 1;
  __REG32 TXDMAE                : 1;
  __REG32 DMAONERR              : 1;
  __REG32                       :29;
} __uartdmacr_bits;

/* I2CnCR1 (I2Cn Control Register 1) */
typedef struct
{
  __REG32 SCK                   : 3;
  __REG32 NOACK                 : 1;
  __REG32 ACK                   : 1;
  __REG32 BC                    : 3;
  __REG32                       :24;
} __i2ccr1_bits;

/* I2CnDBR (I2Cn Data Buffer Register) */
typedef struct
{
  __REG32 DB                    : 8;
  __REG32                       :24;
} __i2cdbr_bits;

/* I2CnAR (I2Cn (Slave) Address Register) */
typedef struct
{
  __REG32 ALS                   : 1;
  __REG32 SA                    : 7;
  __REG32                       :24;
} __i2car_bits;

/* I2CnCR2 (I2Cn Control Register 2) (Write Only)
   I2CnSR (I2Cn Status Register) (Read Only) */
typedef union
{
  /* I2CxCR2*/
  struct
  {
  __REG32 SWRES                 : 2;
  __REG32                       : 1;
  __REG32 I2CM                  : 1;
  __REG32 PIN                   : 1;
  __REG32 BB                    : 1;
  __REG32 TRX                   : 1;
  __REG32 MST                   : 1;
  __REG32                       :24;
  };
  /* I2CxSR*/
  struct
  {
  __REG32 LRB                   : 1;
  __REG32 AD0                   : 1;
  __REG32 AAS                   : 1;
  __REG32 AL                    : 1;
  __REG32 _PIN                  : 1;
  __REG32 _BB                   : 1;
  __REG32 _TRX                  : 1;
  __REG32 _MST                  : 1;
  __REG32                       :24;
  };
} __i2ccr2_bits;

/* I2CnPRS (I2Cn Prescaler Clock Set Register) */
typedef struct
{
  __REG32 PRSCK                 : 5;
  __REG32                       :27;
} __i2cprs_bits;

/* I2CnIE (I2Cn Interrupt Enable Register) */
typedef struct
{
  __REG32 IE                    : 1;
  __REG32                       :31;
} __i2cie_bits;

/* I2CnIR (I2Cn Interrupt Register) */
typedef struct
{
  __REG32 IC                    : 1;
  __REG32                       :31;
} __i2cir_bits;

/* SSPnCR0 (SSPn control register 0) */
typedef struct
{
  __REG32 DSS                   : 4;
  __REG32 FRF                   : 2;
  __REG32 SPO                   : 1;
  __REG32 SPH                   : 1;
  __REG32 SCR                   : 8;
  __REG32                       :16;
} __sspcr0_bits;

/* SSPnCR1 (SSPn control register 1) */
typedef struct
{
  __REG32                       : 1;
  __REG32 SSE                   : 1;
  __REG32 MS                    : 1;
  __REG32 SOD                   : 1;
  __REG32                       :28;
} __sspcr1_bits;

/* SSPnDR (SSPn data register) */
typedef struct
{
  __REG32 DATA                  :16;
  __REG32                       :16;
} __sspdr_bits;

/* SSPnSR (SSPn status register) */
typedef struct
{
  __REG32 TFE                   : 1;
  __REG32 TNF                   : 1;
  __REG32 RNE                   : 1;
  __REG32 RFF                   : 1;
  __REG32 BSY                   : 1;
  __REG32                       :27;
} __sspsr_bits;

/* SSPnCPSR (SSPn clock prescale register) */
typedef struct
{
  __REG32 CPSDVSR               : 8;
  __REG32                       :24;
} __sspcpsr_bits;

/* SSPnCIMSC (SSPn interrupt mask set/clear register) */
typedef struct
{
  __REG32 RORIM                 : 1;
  __REG32 RTIM                  : 1;
  __REG32 RXIM                  : 1;
  __REG32 TXIM                  : 1;
  __REG32                       :28;
} __sspimsc_bits;

/* SSPnRIS (SSPn raw interrupt status register) */
typedef struct
{
  __REG32 RORRIS                : 1;
  __REG32 RTRIS                 : 1;
  __REG32 RXRIS                 : 1;
  __REG32 TXRIS                 : 1;
  __REG32                       :28;
} __sspris_bits;

/* SSPnMIS (SSPn masked interrupt status register) */
typedef struct
{
  __REG32 RORMIS                : 1;
  __REG32 RTMIS                 : 1;
  __REG32 RXMIS                 : 1;
  __REG32 TXMIS                 : 1;
  __REG32                       :28;
} __sspmis_bits;

/* SSPnICR (SSPn interrupt clear register) */
typedef struct
{
  __REG32 RORIC                 : 1;
  __REG32 RTIC                  : 1;
  __REG32                       :30;
} __sspicr_bits;

/* SSPnDMACR (SSPn DMA Control register) */
typedef struct
{
  __REG32 RXDMAE                : 1;
  __REG32 TXDMAE                : 1;
  __REG32                       :30;
} __sspdmacr_bits;

/* UDINTSTS (Interrupt Status register) */
typedef struct
{
  __REG32 int_setup             : 1;
  __REG32 int_status_nak        : 1;
  __REG32 int_status            : 1;
  __REG32 int_rx_zero           : 1;
  __REG32 int_sof               : 1;
  __REG32 int_ep0               : 1;
  __REG32 int_ep                : 1;
  __REG32 int_nak               : 1;
  __REG32 int_suspend_resume    : 1;
  __REG32 int_usb_reset         : 1;
  __REG32 int_usb_reset_end     : 1;
  __REG32                       : 6;
  __REG32 int_mw_set_add        : 1;
  __REG32 int_mw_end_add        : 1;
  __REG32 int_mw_timeout        : 1;
  __REG32 int_mw_ahberr         : 1;
  __REG32 int_mr_end_add        : 1;
  __REG32 int_mr_ep_dset        : 1;
  __REG32 int_mr_ahberr         : 1;
  __REG32 int_udc2_reg_rd       : 1;
  __REG32 int_dmac_reg_rd       : 1;
  __REG32                       : 3;
  __REG32 int_mw_rerror         : 1;
  __REG32                       : 2;
} __udintsts_bits;

/* UDINTENB (Interrupt Enable register) */
typedef struct
{
  __REG32                       : 8;
  __REG32 suspend_resume_en     : 1;
  __REG32 usb_reset_en          : 1;
  __REG32 usb_reset_end_en      : 1;
  __REG32                       : 6;
  __REG32 mw_set_add_en         : 1;
  __REG32 mw_end_add_en         : 1;
  __REG32 mw_timeout_en         : 1;
  __REG32 mw_ahberr_en          : 1;
  __REG32 mr_end_add_en         : 1;
  __REG32 mr_ep_dset_en         : 1;
  __REG32 mr_ahberr_en          : 1;
  __REG32 udc2_reg_rd_en        : 1;
  __REG32 dmac_reg_rd_en        : 1;
  __REG32                       : 3;
  __REG32 mw_rerror_en          : 1;
  __REG32                       : 2;
} __udintenb_bits;

/* UDMWTOUT (Master Write Timeout register) */
typedef struct
{
  __REG32 timeout_en            : 1;
  __REG32 timeoutset            :31;
} __udmwtout_bits;

/* UDC2STSET (UDC2 Setting register) */
typedef struct
{
  __REG32 tx0                   : 1;
  __REG32                       : 3;
  __REG32 eopb_enable           : 1;
  __REG32                       :27;
} __udc2stset_bits;

/* UDMSTSET (DMAC Setting register) */
typedef struct
{
  __REG32 mw_enable             : 1;
  __REG32 mw_abort              : 1;
  __REG32 mw_reset              : 1;
  __REG32                       : 1;
  __REG32 mr_enable             : 1;
  __REG32 mr_abort              : 1;
  __REG32 mr_reset              : 1;
  __REG32                       : 1;
  __REG32 m_burst_type          : 1;
  __REG32                       :23;
} __udmstset_bits;

/* DMACRDREQ (DMAC Read Requset register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 dmardadr              : 6;
  __REG32                       :22;
  __REG32 dmardclr              : 1;
  __REG32 dmardreq              : 1;
} __dmacrdreq_bits;

/* UDC2RDREQ (UDC2 Read Request register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 udc2rdadr             : 8;
  __REG32                       :20;
  __REG32 udc2rdclr             : 1;
  __REG32 udc2rdreq             : 1;
} __udc2rdreq_bits;

/* UDC2RDVL (UDC2 Read Value register) */
typedef struct
{
  __REG32 udc2rdata             :16;
  __REG32                       :16;
} __udc2rdvl_bits;

/* ARBTSET (Arbiter Setting register) */
typedef struct
{
  __REG32 abtpri_r0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_r1             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w0             : 2;
  __REG32                       : 2;
  __REG32 abtpri_w1             : 2;
  __REG32                       :14;
  __REG32 abtmod                : 1;
  __REG32                       : 2;
  __REG32 abt_en                : 1;
} __arbtset_bits;

/* UDPWCTL (Power Detect Control register) */
typedef struct
{
  __REG32 usb_reset             : 1;
  __REG32 pw_resetb             : 1;
  __REG32 pw_detect             : 1;
  __REG32 phy_suspend           : 1;
  __REG32 suspend_x             : 1;
  __REG32 phy_resetb            : 1;
  __REG32 phy_remote_wkup       : 1;
  __REG32 wakeup_en             : 1;
  __REG32                       :24;
} __udpwctl_bits;

/* UDMSTSTS (Master Status register) */
typedef struct
{
  __REG32 mwepdset              : 1;
  __REG32 mrepdset              : 1;
  __REG32 mwbfemp               : 1;
  __REG32 mrbfemp               : 1;
  __REG32 mrepempty             : 1;
  __REG32                       :27;
} __udmststs_bits;

/* UD2ADR (Address-State register) */
typedef struct
{
  __REG32 dev_adr               : 7;
  __REG32                       : 1;
  __REG32 Default               : 1;
  __REG32 Addressed             : 1;
  __REG32 Configured            : 1;
  __REG32 Suspend               : 1;
  __REG32 cur_speed             : 2;
  __REG32 ep_bi_mode            : 1;
  __REG32 stage_err             : 1;
  __REG32                       :16;
} __ud2adr_bits;

/* UD2FRM (Frame register) */
typedef struct
{
  __REG32 frame                 :11;
  __REG32                       : 1;
  __REG32 f_status              : 2;
  __REG32                       : 1;
  __REG32 create_sof            : 1;
  __REG32                       :16;
} __ud2frm_bits;

/* UD2TMD (USB-Testmode register) */
typedef struct
{
  __REG32 t_sel                 : 8;
  __REG32                       : 1;
  __REG32 test_j                : 1;
  __REG32 test_k                : 1;
  __REG32 se0_nak               : 1;
  __REG32 packet                : 1;
  __REG32                       :19;
} __ud2tmd_bits;

/* UD2CMD (Command register) */
typedef struct
{
  __REG32 com                   : 4;
  __REG32 ep                    : 4;
  __REG32 rx_nullpkt_ep         : 4;
  __REG32                       : 3;
  __REG32 int_toggle            : 1;
  __REG32                       :16;
} __ud2cmd_bits;

/* UD2BRQ (bRequest-bmRequestType register) */
typedef struct
{
  __REG32 recipient             : 5;
  __REG32 req_type              : 2;
  __REG32 dir                   : 1;
  __REG32 request               : 8;
  __REG32                       :16;
} __ud2brq_bits;

/* UD2WVL (wValue register) */
typedef struct
{
  __REG32 value_l               : 8;
  __REG32 value_h               : 8;
  __REG32                       :16;
} __ud2wvl_bits;

/* UD2WIDX (wIndex register) */
typedef struct
{
  __REG32 index_l               : 8;
  __REG32 index_h               : 8;
  __REG32                       :16;
} __ud2widx_bits;

/* UD2WLGTH (wLength register) */
typedef struct
{
  __REG32 length_l              : 8;
  __REG32 length_h              : 8;
  __REG32                       :16;
} __ud2wlgth_bits;

/* UD2INT (INT register)*/
typedef struct
{
  __REG32 i_setup               : 1;
  __REG32 i_status_nak          : 1;
  __REG32 i_status              : 1;
  __REG32 i_rx_data0            : 1;
  __REG32 i_sof                 : 1;
  __REG32 i_ep0                 : 1;
  __REG32 i_ep                  : 1;
  __REG32 i_nak                 : 1;
  __REG32 m_setup               : 1;
  __REG32 m_status_nak          : 1;
  __REG32 m_status              : 1;
  __REG32 m_rx_data0            : 1;
  __REG32 m_sof                 : 1;
  __REG32 m_ep0                 : 1;
  __REG32 m_ep                  : 1;
  __REG32 m_nak                 : 1;
  __REG32                       :16;
} __ud2int_bits;

/* UD2INT (INT register) 
   UD2INTNAK (INT_NAK register) */
typedef struct
{
  __REG32                       : 1;
  __REG32 i_ep1                 : 1;
  __REG32 i_ep2                 : 1;
  __REG32 i_ep3                 : 1;
  __REG32                       :28;
} __ud2intep_bits;

/* UD2INTEPMSK (INT_EP_MASK register)
   UD2INTNAKMSK (INT_NAK_MASK register)*/
typedef struct
{
  __REG32                       : 1;
  __REG32 m_ep1                 : 1;
  __REG32 m_ep2                 : 1;
  __REG32 m_ep3                 : 1;
  __REG32                       :28;
} __ud2intepmsk_bits;

/* UD2INTRX0 (INT_RX_DATA0 register) */
typedef struct
{
  __REG32 rx_d0_ep0             : 1;
  __REG32 rx_d0_ep1             : 1;
  __REG32 rx_d0_ep2             : 1;
  __REG32 rx_d0_ep3             : 1;
  __REG32                       :28;
} __ud2intrx0_bits;

/* UD2EP0MSZ (EPn_Ma0PacketSize register) */
typedef struct
{
  __REG32 max_pkt               : 7;
  __REG32                       : 5;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2ep0msz_bits;

/* UD2EPnMSZ (EPn_MaxPacketSize register) */
typedef struct
{
  __REG32 max_pkt               :11;
  __REG32                       : 1;
  __REG32 dset                  : 1;
  __REG32                       : 2;
  __REG32 tx_0data              : 1;
  __REG32                       :16;
} __ud2epmsz_bits;

/* UD2EP0STS (EP0_ Status register) */
typedef struct
{
  __REG32                       : 9;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32                       : 1;
  __REG32 ep0_mask              : 1;
  __REG32                       :16;
} __ud2ep0sts_bits;

/* UD2EP0DSZ (EP0_Datasize register) */
typedef struct
{
  __REG32 size                  : 7;
  __REG32                       :25;
} __ud2ep0dsz_bits;

/* UD2EPnFIFO (EPn_FIFO register) */
typedef struct
{
  __REG32 data                  :16;
  __REG32                       :16;
} __ud2epfifo_bits;

/* UD2EP1STS (EP1_Status register) */
typedef struct
{
  __REG32 num_mf                : 2;
  __REG32 t_type                : 2;
  __REG32                       : 3;
  __REG32 dir                   : 1;
  __REG32 disable               : 1;
  __REG32 status                : 3;
  __REG32 toggle                : 2;
  __REG32 bus_sel               : 1;
  __REG32 pkt_mode              : 1;
  __REG32                       :16;
} __ud2ep1sts_bits;

/* UD2EP1DSZ (EP1_Datasize register) */
typedef struct
{
  __REG32 size                  :11;
  __REG32                       :21;
} __ud2ep1dsz_bits;

/* LDACR0 (LCDDA Control Register 0) */
typedef struct
{
  __REG32 S1ADR                 : 8;
  __REG32 PCEN                  : 1;
  __REG32 BCENX                 : 1;
  __REG32 DTFMT                 : 1;
  __REG32 BCENYT                : 1;
  __REG32 DMAEN                 : 1;
  __REG32 DMAMD                 : 1;
  __REG32 AUTOHP                : 1;
  __REG32 BCENYB                : 1;
  __REG32 EINTM                 : 1;
  __REG32 ERRINTM               : 1;
  __REG32                       : 2;
  __REG32 EINTF                 : 1;
  __REG32 ERRINTF               : 1;
  __REG32                       :10;
} __ldacr0_bits;

/* LDADRSRC1 (LCDDA Density Ratio of Source 1 Picture) */
typedef struct
{
  __REG32 RDRSRC1               : 8;
  __REG32 GDRSRC1               : 8;
  __REG32 BDRSRC1               : 8;
  __REG32                       : 8;
} __ldadrsrc1_bits;

/* LDADRSRC0 (LCDDA Density Ratio of Source 0 Picture) */
typedef struct
{
  __REG32 RDRSRC0               : 8;
  __REG32 GDRSRC0               : 8;
  __REG32 BDRSRC0               : 8;
  __REG32                       : 8;
} __ldadrsrc0_bits;

/* LDAFCPSRC1 (LCDDA Replaced Font Area Color pallet of Source1)
   LDAEFCPSRC1 (LCDDA Replaced Except Font Area Color pallet of Source1) */
typedef struct
{
  __REG32 RFONT                 : 8;
  __REG32 GFONT                 : 8;
  __REG32 BFONT                 : 8;
  __REG32                       : 8;
} __ldafcpsrc_bits;

/* LDACR0 (LCDDA Delta Value (Read Step) address Register of Source 0) */
typedef struct
{
  __REG32 DXS0                  : 3;
  __REG32                       : 3;
  __REG32 DYS0                  :12;
  __REG32                       :12;
  __REG32 INDSAEN               : 1;
  __REG32 OVWEN                 : 1;  
} __ldadvsrc0_bits;

/* LDADVSRC1 (LCDDA Delta Value (Read Step) address Register of Source 1) */
typedef struct
{
  __REG32 DXS1                  : 3;
  __REG32                       : 3;
  __REG32 DYS1                  :12;
  __REG32                       : 6;
  __REG32 OFSETX                : 8;
} __ldadvsrc1_bits;

/* LDADXDST (LCDDA X-Delta Value (Write Step) address Register of Destination) */
typedef struct
{
  __REG32 DXDST                 :24;
  __REG32 DXDSIGN               : 1;
  __REG32                       : 3;
  __REG32 XRRATE                : 4;
} __ldadxdst_bits;

/* LDADYDST (LCDDA Y-Delta Value (Write Step) address Register of Destination) */
typedef struct
{
  __REG32 DYDST                 :24;
  __REG32 DYDSIGN               : 1;
  __REG32                       : 3;
  __REG32 YRDRATE               : 4;
} __ldadydst_bits;

/* LDASSIZE (LCDDA Source Picture Size) */
typedef struct
{
  __REG32 SXSIZE                :10;
  __REG32                       : 2;
  __REG32 SYSIZE                :10;
  __REG32                       : 2;
  __REG32 XEXRATE               : 8;
} __ldassize_bits;

/* LDADSIZE (LCDDA Destination Picture Size) */
typedef struct
{
  __REG32 DXSIZE                :10;
  __REG32                       :14;
  __REG32 YEXRATE               : 8;
} __ldadsize_bits;

/* LDACR1 (LCDDA Control Register1) */
typedef struct
{
  __REG32 S1ADR                 :24;
  __REG32 OPMODE                : 5;
  __REG32                       : 1;
  __REG32 LDASTART              : 1;
  __REG32 SYNRST                : 1;
} __ldacr1_bits;

/* LDACR2 */
typedef struct
{
  __REG32 DXS0                  : 3;
  __REG32                       : 3;
  __REG32 DYS0                  :12;
  __REG32                       :12;
  __REG32 INDSAEN               : 1;
  __REG32 OVWEN                 : 1;
} __ldacr2_bits;

/* TSICR0 (TSI Control Register0) */
typedef struct
{
  __REG32 MXEN                  : 1;
  __REG32 MYEN                  : 1;
  __REG32 PXEN                  : 1;
  __REG32 PYEN                  : 1;
  __REG32 TWIEN                 : 1;
  __REG32 PTST                  : 1;
  __REG32 INGE                  : 1;
  __REG32 TSI7                  : 1;
  __REG32                       :24;
} __tsicr0_bits;

/* TSICR1 (TSI Control Register1) */
typedef struct
{
  __REG32 DB1                   : 1;
  __REG32 DB2                   : 1;
  __REG32 DB4                   : 1;
  __REG32 DB8                   : 1;
  __REG32 DB64                  : 1;
  __REG32 DB256                 : 1;
  __REG32 DB1024                : 1;
  __REG32 DBC7                  : 1;
  __REG32                       :24;
} __tsicr1_bits;

/* CMSCR (CMOS Image Sensor Control Register) */
typedef struct
{
  __REG32 CSRST                 : 1;
  __REG32 CSIZE3                : 4;
  __REG32 CINTSEL               : 1;
  __REG32 CFPCLR                : 1;
  __REG32 CFDEF                 : 1;
  __REG32 CFOVF                 : 1;
  __REG32 CPCKPH                : 1;
  __REG32 CHBKPH                : 1;
  __REG32 CHSYPH                : 1;
  __REG32 CVSYPH                : 1;
  __REG32 CDEDLY                : 1;
  __REG32 CFINTM                : 1;
  __REG32 CSINTM                : 1;
  __REG32 CFINTF                : 1;
  __REG32 CSINTF                : 1;
  __REG32 CSFOW                 : 1;
  __REG32                       :13;
} __cmscr_bits;

/* CMSCV (CMOS Image Sensor Color Space Conversion Register) */
typedef struct
{
  __REG32                       : 2;
  __REG32 DMAEN                 : 1;
  __REG32 CCVM                  : 2;
  __REG32 CRGBM                 : 1;
  __REG32 CCVSMMS               : 1;
  __REG32 CSCVTRG               : 2;
  __REG32 CSCVST                : 1;
  __REG32                       :22;
} __cmscv_bits;

/* CMSCVP0 (CMOS Image Sensor Color Space Conversion Parameter Register0) */
typedef struct
{
  __REG32 CRYG                  : 7;
  __REG32                       : 1;
  __REG32 CRVG                  : 7;
  __REG32                       : 1;
  __REG32 CGYG                  : 7;
  __REG32                       : 1;
  __REG32 CGVG                  : 7;
  __REG32                       : 1;
} __cmscvp0_bits;

/* CMSCVP0 (CMOS Image Sensor Color Space Conversion Parameter Register0) */
typedef struct
{
  __REG32 CGUG                  : 7;
  __REG32                       : 1;
  __REG32 CBYG                  : 7;
  __REG32                       : 1;
  __REG32 CBUG                  : 7;
  __REG32                       : 1;
  __REG32 CYOFS                 : 7;
  __REG32                       : 1;
} __cmscvp1_bits;

/* CMSYD (CMOS Image Sensor Soft Conversion Y-data Register) */
typedef struct
{
  __REG32 CYD                   : 8;
  __REG32                       :24;
} __cmsyd_bits;

/* CMSUD (CMOS Image Sensor Soft Conversion U-data Register) */
typedef struct
{
  __REG32 CUD                   : 8;
  __REG32                       :24;
} __cmsud_bits;

/* CMSVD (CMOS Image Sensor Soft Conversion V-data Register) */
typedef struct
{
  __REG32 CVD                   : 8;
  __REG32                       :24;
} __cmsvd_bits;

/* CMSSCTR (CMOS Image Sensor Scaling & Trimming Control Register) */
typedef struct
{
  __REG32 CSCL                  : 2;
  __REG32                       : 2;
  __REG32 CTREN                 : 1;
  __REG32                       :27;
} __cmscstr_bits;

/* CMSSCTR (CMOS Image Sensor Scaling & Trimming Control Register) */
typedef struct
{
  __REG32 CTSH                  :11;
  __REG32                       : 5;
  __REG32 CTSV                  :10;
  __REG32                       : 6;
} __cmsts_bits;

/* CMSTE (CMOS Image Sensor Trimming Space End point Setting Register) */
typedef struct
{
  __REG32 CTEH                  :11;
  __REG32                       : 5;
  __REG32 CTEV                  :10;
  __REG32                       : 6;
} __cmste_bits;

/* CMSTE (CMOS Image Sensor Trimming Space End point Setting Register) */
typedef struct
{
  __REG32 CYD                   : 8;
  __REG32 CUD                   : 8;
  __REG32 CVD                   : 8;
  __REG32                       : 8;
} __cmsscdma_bits;

/* MLDALMINV (Melody Alarm signal Invert Register) */
typedef struct
{
  __REG32 MLALINV               : 1;
  __REG32                       :31;
} __mldalminv_bits;

/* MLDALMSEL (Melody Alarm Select Register) */
typedef struct
{
  __REG32 MLALSEL               : 2;
  __REG32                       :30;
} __mldalmsel_bits;

/* ALMCNTCR (Alarm Counter Control Register) */
typedef struct
{
  __REG32 ALMCC                 : 1;
  __REG32                       :31;
} __almcntcr_bits;

/* ALMPATTERN (Alarm Pattern Register) */
typedef struct
{
  __REG32 ALMPTSEL              : 8;
  __REG32                       :24;
} __almpattern_bits;

/* MLDCNTCR (Melody Counter Control Register) */
typedef struct
{
  __REG32 MLDCC                 : 1;
  __REG32                       :31;
} __mldcntcr_bits;

/* MLDCNTCR (Melody Counter Control Register) */
typedef struct
{
  __REG32 MLDF                  :12;
  __REG32                       :20;
} __mldfrq_bits;

/* RTCALMINTCTR (RTC ALM Interrupt Control Register) */
typedef struct
{
  __REG32 RTCINTEN              : 1;
  __REG32 AINTEN8192            : 1;
  __REG32 AINTEN512             : 1;
  __REG32 AINTEN64              : 1;
  __REG32 AINTEN2               : 1;
  __REG32 AINTEN1               : 1;
  __REG32 RTCINTCLR             : 1;
  __REG32 ALMINTCLR             : 1;
  __REG32                       :24;
} __rtcalmintctr_bits;

/* RTCALMMIS (RTC ALM Interrupt Status Registe) */
typedef struct
{
  __REG32 RTCINT                : 1;
  __REG32 ALMINT                : 1;
  __REG32                       :30;
} __rtcalmmis_bits;

/* ADMOD0 (AD mode control register 0) */
typedef struct
{
  __REG32 ADS                   : 1;
  __REG32 SCAN                  : 1;
  __REG32 REPEAT                : 1;
  __REG32 ITM                   : 1;
  __REG32                       : 2;
  __REG32 ADBFN                 : 1;
  __REG32 EOCFN                 : 1;
  __REG32                       :24;
} __admod0_bits;

/* ADMOD1 (AD mode control register 1) */
typedef struct
{
  __REG32 ADCH                  : 3;
  __REG32                       : 2;
  __REG32 ADSCN                 : 1;
  __REG32                       : 1;
  __REG32 DACON                 : 1;
  __REG32                       :24;
} __admod1_bits;

/* ADMOD2 (AD mode control register 2) */
typedef struct
{
  __REG32 HPADCH                : 3;
  __REG32                       : 2;
  __REG32 HPADCE                : 1;
  __REG32 ADBFHP                : 1;
  __REG32 EOCFHP                : 1;
  __REG32                       :24;
} __admod2_bits;

/* ADMOD3 (AD mode control register 3) */
typedef struct
{
  __REG32 ADOBSV                : 1;
  __REG32 REGS                  : 4;
  __REG32 ADOBIC                : 1;
  __REG32                       :26;
} __admod3_bits;

/* ADMOD4 (AD mode control register 4) */
typedef struct
{
  __REG32 ADRST                 : 2;
  __REG32                       :30;
} __admod4_bits;

/* ADREGnL (AD conversion result lower-order register n) */
typedef struct
{
  __REG32 ADRRF                 : 1;
  __REG32 OVR                   : 1;
  __REG32                       : 4;
  __REG32 ADR                   : 2;
  __REG32                       :24;
} __adregl_bits;

/* ADREGnH (AD conversion result higher-order register n) */
typedef struct
{
  __REG32 ADR                   : 8;
  __REG32                       :24;
} __adregh_bits;

/* ADCOMREGL (A/D conversion result comparison lower-order register) */
typedef struct
{
  __REG32                       : 6;
  __REG32 ADRCOM                : 2;
  __REG32                       :24;
} __adcomregl_bits;

/* ADCOMREGH (A/D conversion result comparison higher-order register) */
typedef struct
{
  __REG32 ADRCOM                : 8;
  __REG32                       :24;
} __adcomregh_bits;

/* ADCLK (AD conversion clock setting register) */
typedef struct
{
  __REG32 ADCLK                 : 3;
  __REG32                       :29;
} __adclk_bits;

/* ADIE (A/D interrupt enable register) */
typedef struct
{
  __REG32 NIE                   : 1;
  __REG32 HPIE                  : 1;
  __REG32 MIE                   : 1;
  __REG32                       :29;
} __adie_bits;

/* ADIS (AD interrupt status register) */
typedef struct
{
  __REG32 NIS                   : 1;
  __REG32 HPIS                  : 1;
  __REG32 MIS                   : 1;
  __REG32                       :29;
} __adis_bits;

/* ADIC (AD interrupt clear register) */
typedef struct
{
  __REG32 NIC                   : 1;
  __REG32 HPIC                  : 1;
  __REG32 MIC                   : 1;
  __REG32                       :29;
} __adic_bits;

/* WdogControl (Watchdog control register) */
typedef struct
{
  __REG32 INTEN                 : 1;
  __REG32 RESEN                 : 1;
  __REG32                       :30;
} __wdogcontrol_bits;

/* WdogRIS (Watchdog raw interrupt status) */
typedef struct
{
  __REG32 RAWWDTINT             : 1;
  __REG32                       :31;
} __wdogris_bits;

/* WdogMIS (Watchdog masked interrupt status) */
typedef struct
{
  __REG32 WDTINT                : 1;
  __REG32                       :31;
} __wdogmis_bits;

/* WdogLock (Watchdog Lock register) */
typedef struct
{
  __REG32 REGWENST              : 1;
  __REG32                       :31;
} __wdoglock_bits;

/* BPARELE register */
typedef struct
{
  __REG32 BPARELE0              : 1;
  __REG32 BPARELE1              : 1;
  __REG32 BPARELE2              : 1;
  __REG32 BPARELE3              : 1;
  __REG32 BPARELE4              : 1;
  __REG32 BPARELE5              : 1;
  __REG32 BPARELE6              : 1;
  __REG32 BPARELE7              : 1;
  __REG32                       :24;
} __bparele_bits;

/* BPDRELE register */
typedef struct
{
  __REG32                       : 6;
  __REG32 BPDRELE6              : 1;
  __REG32                       :25;
} __bpdrele_bits;

/* BPPRELE register */
typedef struct
{
  __REG32 BPPRELE0              : 1;
  __REG32 BPPRELE1              : 1;
  __REG32 BPPRELE2              : 1;
  __REG32 BPPRELE3              : 1;
  __REG32 BPPRELE4              : 1;
  __REG32 BPPRELE5              : 1;
  __REG32 BPPRELE6              : 1;
  __REG32 BPPRELE7              : 1;
  __REG32                       :24;
} __bpprele_bits;

/* BRTRELE register */
typedef struct
{
  __REG32 BRTRELE0              : 1;
  __REG32                       :31;
} __brtrele_bits;

/* BPAEDGE register */
typedef struct
{
  __REG32 BPAEDGE0              : 1;
  __REG32 BPAEDGE1              : 1;
  __REG32 BPAEDGE2              : 1;
  __REG32 BPAEDGE3              : 1;
  __REG32 BPAEDGE4              : 1;
  __REG32 BPAEDGE5              : 1;
  __REG32 BPAEDGE6              : 1;
  __REG32 BPAEDGE7              : 1;
  __REG32                       :24;
} __bpaedge_bits;

/* BPDEDGE register */
typedef struct
{
  __REG32                       : 6;
  __REG32 BPDEDGE6              : 1;
  __REG32                       :25;
} __bpdedge_bits;

/* BPPEDGE register */
typedef struct
{
  __REG32 BPPEDGE0              : 1;
  __REG32 BPPEDGE1              : 1;
  __REG32 BPPEDGE2              : 1;
  __REG32 BPPEDGE3              : 1;
  __REG32 BPPEDGE4              : 1;
  __REG32 BPPEDGE5              : 1;
  __REG32 BPPEDGE6              : 1;
  __REG32 BPPEDGE7              : 1;
  __REG32                       :24;
} __bppedge_bits;

/* BPARINT register */
typedef struct
{
  __REG32 BPARINT0              : 1;
  __REG32 BPARINT1              : 1;
  __REG32 BPARINT2              : 1;
  __REG32 BPARINT3              : 1;
  __REG32 BPARINT4              : 1;
  __REG32 BPARINT5              : 1;
  __REG32 BPARINT6              : 1;
  __REG32 BPARINT7              : 1;
  __REG32                       :24;
} __bparint_bits;

/* BPDRINT register */
typedef struct
{
  __REG32                       : 6;
  __REG32 BPDRINT6              : 1;
  __REG32                       :25;
} __bpdrint_bits;

/* BPPRINT register */
typedef struct
{
  __REG32 BPPRINT0              : 1;
  __REG32 BPPRINT1              : 1;
  __REG32 BPPRINT2              : 1;
  __REG32 BPPRINT3              : 1;
  __REG32 BPPRINT4              : 1;
  __REG32 BPPRINT5              : 1;
  __REG32 BPPRINT6              : 1;
  __REG32 BPPRINT7              : 1;
  __REG32                       :24;
} __bpprint_bits;

/* BRTRINT register */
typedef struct
{
  __REG32 BPPRINT0              : 1;
  __REG32                       :31;
} __brtrint_bits;

/* PMCDRV register */
typedef struct
{
  __REG32 DRV_MEM0              : 1;
  __REG32 DRV_MEM1              : 1;
  __REG32                       : 2;
  __REG32 DRV_LCD               : 1;
  __REG32 DRV_I2S               : 1;
  __REG32 DRV_SP0               : 1;
  __REG32                       :25;
} __pmcdrv_bits;

/* DMCCKECTL register */
typedef struct
{
  __REG32 DMCCKEHLD             : 1;
  __REG32                       :31;
} __dmcckectl_bits;

/* PMCCTL register */
typedef struct
{
  __REG32 WUTM                  : 2;
  __REG32                       : 4;
  __REG32 PMCPWE                : 1;
  __REG32 PCM_ON                : 1;
  __REG32                       :24;
} __pmcctl_bits;

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

/* CLKSCR1 (Oscillation frequency detection control register 1) */
typedef struct {
  __REG32 CLKWEN            : 8;
  __REG32                   :24;
} __clkscr1_bits;

/* CLKSCR2 (Oscillation frequency detection control register 2) */
typedef struct {
  __REG32 CLKSEN            : 8;
  __REG32                   :24;
} __clkscr2_bits;

/* CLKSCR3 (Oscillation frequency detection control register 3) */
typedef struct {
  __REG32 CLKSF             : 1;
  __REG32 RESEN             : 1;
  __REG32                   :30;
} __clkscr3_bits;

/* CLKSMN (Lower detection frequency setting register) */
typedef struct {
  __REG32 LDFS              : 9;
  __REG32                   :23;
} __clksmn_bits;

/* CLKSMX (Higher detection frequency setting register) */
typedef struct {
  __REG32 HDFS              : 9;
  __REG32                   :23;
} __clksmx_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** SC (System Control)
 **
 ***************************************************************************/
__IO_REG32_BIT(REMAP,                 0xF0000004,__READ_WRITE ,__remap_bits);
__IO_REG32(    SYSCR0,                0xF0050000,__READ_WRITE );
__IO_REG32_BIT(SYSCR1,                0xF0050004,__READ_WRITE ,__syscr1_bits);
__IO_REG32_BIT(SYSCR2,                0xF0050008,__READ_WRITE ,__syscr2_bits);
__IO_REG32_BIT(SYSCR3,                0xF005000C,__READ_WRITE ,__syscr3_bits);
__IO_REG32_BIT(SYSCR4,                0xF0050010,__READ_WRITE ,__syscr4_bits);
__IO_REG32_BIT(SYSCR5,                0xF0050014,__READ       ,__syscr5_bits);
__IO_REG32_BIT(SYSCR6,                0xF0050018,__WRITE      ,__syscr6_bits);
__IO_REG32_BIT(SYSCR7,                0xF005001C,__WRITE      ,__syscr7_bits);
__IO_REG32_BIT(SYSCR8,                0xF0050020,__READ_WRITE ,__syscr8_bits);
__IO_REG32_BIT(CLKCR5,                0xF0050054,__READ_WRITE ,__clkcr5_bits);

/***************************************************************************
 **
 ** NDFC (NAND-Flash Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(NDFMCR0,               0xF2010000,__READ_WRITE ,__ndfmcr0_bits);
__IO_REG32_BIT(NDFMCR1,               0xF2010004,__READ_WRITE ,__ndfmcr1_bits);
__IO_REG32_BIT(NDFMCR2,               0xF2010008,__READ_WRITE ,__ndfmcr2_bits);
__IO_REG32_BIT(NDFINTC,               0xF201000C,__READ_WRITE ,__ndfintc_bits);
__IO_REG32(    NDFDTR,                0xF2010010,__READ_WRITE );
__IO_REG32(    NDECCRD0,              0xF2010020,__READ       );
__IO_REG32(    NDECCRD1,              0xF2010024,__READ       );
__IO_REG32_BIT(NDECCRD2,              0xF2010028,__READ       ,__ndeccrd2_bits);
__IO_REG32_BIT(NDRSCA0,               0xF2010030,__READ       ,__ndrsca_bits);
__IO_REG32_BIT(NDRSCD0,               0xF2010034,__READ       ,__ndrscd_bits);
__IO_REG32_BIT(NDRSCA1,               0xF2010038,__READ       ,__ndrsca_bits);
__IO_REG32_BIT(NDRSCD1,               0xF201003C,__READ       ,__ndrscd_bits);
__IO_REG32_BIT(NDRSCA2,               0xF2010040,__READ       ,__ndrsca_bits);
__IO_REG32_BIT(NDRSCD2,               0xF2010044,__READ       ,__ndrscd_bits);
__IO_REG32_BIT(NDRSCA3,               0xF2010048,__READ       ,__ndrsca_bits);
__IO_REG32_BIT(NDRSCD3,               0xF201004C,__READ       ,__ndrscd_bits);

/***************************************************************************
 **
 ** TIM0
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer0Load,            0xF0040000,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer0Value,           0xF0040004,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer0Control,         0xF0040008,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer0IntClr,          0xF004000C,__WRITE      );
__IO_REG32_BIT(Timer0RIS,             0xF0040010,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer0MIS,             0xF0040014,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer0BGLoad,          0xF0040018,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer0Mode,            0xF004001C,__READ_WRITE ,__timermode_bits);
__IO_REG32_BIT(Timer0Compare1,        0xF00400A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer0CmpIntClr1,      0xF00400C0,__WRITE      );
__IO_REG32_BIT(Timer0CmpEn,           0xF00400E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer0CmpRIS,          0xF00400E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer0CmpMIS,          0xF00400E8,__READ       ,__timercmpmis_bits);
__IO_REG32_BIT(Timer0BGCmp,           0xF00400EC,__READ_WRITE ,__timerbgcmp_bits);

/***************************************************************************
 **
 ** TIM1
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer1Load,            0xF0040100,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer1Value,           0xF0040104,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer1Control,         0xF0040108,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer1IntClr,          0xF004010C,__WRITE      );
__IO_REG32_BIT(Timer1RIS,             0xF0040110,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer1MIS,             0xF0040114,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer1BGLoad,          0xF0040118,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer1Compare1,        0xF00401A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer1CmpIntClr1,      0xF00401C0,__WRITE      );
__IO_REG32_BIT(Timer1CmpEn,           0xF00401E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer1CmpRIS,          0xF00401E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer1CmpMIS,          0xF00401E8,__READ       ,__timercmpmis_bits);

/***************************************************************************
 **
 ** TIM2
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer2Load,            0xF0041000,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer2Value,           0xF0041004,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer2Control,         0xF0041008,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer2IntClr,          0xF004100C,__WRITE      );
__IO_REG32_BIT(Timer2RIS,             0xF0041010,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer2MIS,             0xF0041014,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer2BGLoad,          0xF0041018,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer2Mode,            0xF004101C,__READ_WRITE ,__timermode_bits);
__IO_REG32_BIT(Timer2Compare1,        0xF00410A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer2CmpIntClr1,      0xF00410C0,__WRITE      );
__IO_REG32_BIT(Timer2CmpEn,           0xF00410E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer2CmpRIS,          0xF00410E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer2CmpMIS,          0xF00410E8,__READ       ,__timercmpmis_bits);
__IO_REG32_BIT(Timer2BGCmp,           0xF00410EC,__READ_WRITE ,__timerbgcmp_bits);

/***************************************************************************
 **
 ** TIM3
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer3Load,            0xF0041100,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer3Value,           0xF0041104,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer3Control,         0xF0041108,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer3IntClr,          0xF004110C,__WRITE      );
__IO_REG32_BIT(Timer3RIS,             0xF0041110,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer3MIS,             0xF0041114,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer3BGLoad,          0xF0041118,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer3Compare1,        0xF00411A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer3CmpIntClr1,      0xF00411C0,__WRITE      );
__IO_REG32_BIT(Timer3CmpEn,           0xF00411E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer3CmpRIS,          0xF00411E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer3CmpMIS,          0xF00411E8,__READ       ,__timercmpmis_bits);

/***************************************************************************
 **
 ** TIM4
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer4Load,            0xF0042000,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer4Value,           0xF0042004,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer4Control,         0xF0042008,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer4IntClr,          0xF004200C,__WRITE      );
__IO_REG32_BIT(Timer4RIS,             0xF0042010,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer4MIS,             0xF0042014,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer4BGLoad,          0xF0042018,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer4Compare1,        0xF00420A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer4CmpIntClr1,      0xF00420C0,__WRITE      );
__IO_REG32_BIT(Timer4CmpEn,           0xF00420E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer4CmpRIS,          0xF00420E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer4CmpMIS,          0xF00420E8,__READ       ,__timercmpmis_bits);
__IO_REG32_BIT(Timer4BGCmp,           0xF00420EC,__READ_WRITE ,__timerbgcmp_bits);

/***************************************************************************
 **
 ** TIM5
 **
 ***************************************************************************/
__IO_REG32_BIT(Timer5Load,            0xF0042100,__READ_WRITE ,__timerload_bits);
__IO_REG32_BIT(Timer5Value,           0xF0042104,__READ       ,__timervalue_bits);
__IO_REG32_BIT(Timer5Control,         0xF0042108,__READ_WRITE ,__timercontrol_bits);
__IO_REG32(    Timer5IntClr,          0xF004210C,__WRITE      );
__IO_REG32_BIT(Timer5RIS,             0xF0042110,__READ       ,__timerris_bits);
__IO_REG32_BIT(Timer5MIS,             0xF0042114,__READ       ,__timermis_bits);
__IO_REG32_BIT(Timer5BGLoad,          0xF0042118,__READ_WRITE ,__timerbgload_bits);
__IO_REG32_BIT(Timer5Compare1,        0xF00421A0,__READ_WRITE ,__timercompare1_bits);
__IO_REG32(    Timer5CmpIntClr1,      0xF00421C0,__WRITE      );
__IO_REG32_BIT(Timer5CmpEn,           0xF00421E0,__READ_WRITE ,__timercmpen_bits);
__IO_REG32_BIT(Timer5CmpRIS,          0xF00421E4,__READ       ,__timercmpris_bits);
__IO_REG32_BIT(Timer5CmpMIS,          0xF00421E8,__READ       ,__timercmpmis_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2STCON,               0xF2040000,__READ_WRITE ,__i2stcon_bits);
__IO_REG32_BIT(I2STSLVON,             0xF2040004,__READ_WRITE ,__i2stslvon_bits);
__IO_REG32_BIT(I2STFCLR,              0xF2040008,__READ_WRITE ,__i2stfclr_bits);
__IO_REG32_BIT(I2STMS,                0xF204000C,__READ_WRITE ,__i2stms_bits);
__IO_REG32_BIT(I2STMCON,              0xF2040010,__READ_WRITE ,__i2stmcon_bits);
__IO_REG32_BIT(I2STMSTP,              0xF2040014,__READ_WRITE ,__i2stmstp_bits);
__IO_REG32_BIT(I2STDMA1,              0xF2040018,__READ_WRITE ,__i2stdma1_bits);
__IO_REG32_BIT(I2SRCON,               0xF2040020,__READ_WRITE ,__i2srcon_bits);
__IO_REG32_BIT(I2SRSLVON,             0xF2040024,__READ_WRITE ,__i2srslvon_bits);
__IO_REG32_BIT(I2SFRFCLR,             0xF2040028,__READ_WRITE ,__i2srfclr_bits);
__IO_REG32_BIT(I2SRMS,                0xF204002C,__READ_WRITE ,__i2srms_bits);
__IO_REG32_BIT(I2SRMCON,              0xF2040030,__READ_WRITE ,__i2srmcon_bits);
__IO_REG32_BIT(I2SRMSTP,              0xF2040034,__READ_WRITE ,__i2srmstp_bits);
__IO_REG32_BIT(I2SRDMA1,              0xF2040038,__READ_WRITE ,__i2srdma1_bits);
__IO_REG32_BIT(I2SCOMMON,             0xF2040044,__READ_WRITE ,__i2scommon_bits);
__IO_REG32_BIT(I2STST,                0xF2040048,__READ       ,__i2stst_bits);
__IO_REG32_BIT(I2SRST,                0xF204004C,__READ       ,__i2srst_bits);
__IO_REG32_BIT(I2SINT,                0xF2040050,__READ_WRITE ,__i2sint_bits);
__IO_REG32_BIT(I2SINTMSK,             0xF2040054,__READ_WRITE ,__i2sintmsk_bits);
__IO_REG32_BIT(I2STDAT,               0xF2041000,__READ_WRITE ,__i2stdat_bits);
__IO_REG32_BIT(I2SRDAT,               0xF2042000,__READ_WRITE ,__i2stdat_bits);

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCDTiming0,            0xF4200000,__READ_WRITE ,__lcdtiming0_bits);
__IO_REG32_BIT(LCDTiming1,            0xF4200004,__READ_WRITE ,__lcdtiming1_bits);
__IO_REG32_BIT(LCDTiming2,            0xF4200008,__READ_WRITE ,__lcdtiming2_bits);
__IO_REG32_BIT(LCDTiming3,            0xF420000C,__READ_WRITE ,__lcdtiming3_bits);
__IO_REG32(    LCDUPBASE,             0xF4200010,__READ_WRITE );
__IO_REG32(    LCDLPBASE,             0xF4200014,__READ_WRITE );
__IO_REG32_BIT(LCDIMSC,               0xF4200018,__READ_WRITE ,__lcdimsc_bits);
__IO_REG32_BIT(LCDControl,            0xF420001C,__READ_WRITE ,__lcdcontrol_bits);
__IO_REG32_BIT(LCDRIS,                0xF4200020,__READ       ,__lcdris_bits);
__IO_REG32_BIT(LCDMIS,                0xF4200024,__READ       ,__lcdmis_bits);
__IO_REG32_BIT(LCDICR,                0xF4200028,__WRITE      ,__lcdicr_bits);
__IO_REG32(    LCDUPCURR,             0xF420002C,__READ       );
__IO_REG32(    LCDLPCURR,             0xF4200030,__READ       );
__IO_REG32(    LCDPaletteBaseAddr,    0xF4200200,__READ_WRITE );

/***************************************************************************
 **
 ** LCDCOP
 **
 ***************************************************************************/
__IO_REG32_BIT(STN64CR,               0xF00B0000,__READ_WRITE ,__stn64cr_bits);

/***************************************************************************
 **
 ** VIC
 **
 ***************************************************************************/
__IO_REG32_BIT(VICIRQSTATUS,          0xF4000000,__READ       ,__vic_intr_bits);
__IO_REG32_BIT(VICFIQSTATUS,          0xF4000004,__READ       ,__vic_intr_bits);
__IO_REG32_BIT(VICRAWINTR,            0xF4000008,__READ       ,__vic_intr_bits);
__IO_REG32_BIT(VICINTSELECT,          0xF400000C,__READ_WRITE ,__vic_intr_bits);
__IO_REG32_BIT(VICINTENABLE,          0xF4000010,__READ_WRITE ,__vic_intr_bits);
__IO_REG32_BIT(VICINTENCLEAR,         0xF4000014,__WRITE      ,__vic_intr_bits);
__IO_REG32_BIT(VICSOFTINT,            0xF4000018,__READ_WRITE ,__vic_intr_bits);
__IO_REG32_BIT(VICSOFTINTCLEAR,       0xF400001C,__WRITE      ,__vic_intr_bits);
__IO_REG32_BIT(VICPROTECTION,         0xF4000020,__READ_WRITE ,__vicprotection_bits);
__IO_REG32_BIT(VICSWPRIORITYMASK,     0xF4000024,__READ_WRITE ,__vicswprioritymask_bits);
__IO_REG32(    VICVECTADDR0,          0xF4000100,__READ_WRITE );
__IO_REG32(    VICVECTADDR1,          0xF4000104,__READ_WRITE );
__IO_REG32(    VICVECTADDR2,          0xF4000108,__READ_WRITE );
__IO_REG32(    VICVECTADDR3,          0xF400010C,__READ_WRITE );
__IO_REG32(    VICVECTADDR4,          0xF4000110,__READ_WRITE );
__IO_REG32(    VICVECTADDR5,          0xF4000114,__READ_WRITE );
__IO_REG32(    VICVECTADDR6,          0xF4000118,__READ_WRITE );
__IO_REG32(    VICVECTADDR7,          0xF400011C,__READ_WRITE );
__IO_REG32(    VICVECTADDR8,          0xF4000120,__READ_WRITE );
__IO_REG32(    VICVECTADDR9,          0xF4000124,__READ_WRITE );
__IO_REG32(    VICVECTADDR10,         0xF4000128,__READ_WRITE );
__IO_REG32(    VICVECTADDR11,         0xF400012C,__READ_WRITE );
__IO_REG32(    VICVECTADDR12,         0xF4000130,__READ_WRITE );
__IO_REG32(    VICVECTADDR13,         0xF4000134,__READ_WRITE );
__IO_REG32(    VICVECTADDR14,         0xF4000138,__READ_WRITE );
__IO_REG32(    VICVECTADDR15,         0xF400013C,__READ_WRITE );
__IO_REG32(    VICVECTADDR16,         0xF4000140,__READ_WRITE );
__IO_REG32(    VICVECTADDR17,         0xF4000144,__READ_WRITE );
__IO_REG32(    VICVECTADDR18,         0xF4000148,__READ_WRITE );
__IO_REG32(    VICVECTADDR20,         0xF4000150,__READ_WRITE );
__IO_REG32(    VICVECTADDR21,         0xF4000154,__READ_WRITE );
__IO_REG32(    VICVECTADDR22,         0xF4000158,__READ_WRITE );
__IO_REG32(    VICVECTADDR23,         0xF400015C,__READ_WRITE );
__IO_REG32(    VICVECTADDR26,         0xF4000168,__READ_WRITE );
__IO_REG32(    VICVECTADDR27,         0xF400016C,__READ_WRITE );
__IO_REG32(    VICVECTADDR28,         0xF4000170,__READ_WRITE );
__IO_REG32(    VICVECTADDR29,         0xF4000174,__READ_WRITE );
__IO_REG32(    VICVECTADDR30,         0xF4000178,__READ_WRITE );
__IO_REG32(    VICVECTADDR31,         0xF400017C,__READ_WRITE );
__IO_REG32_BIT(VICVECTPRIORITY0,      0xF4000200,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY1,      0xF4000204,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY2,      0xF4000208,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY3,      0xF400020C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY4,      0xF4000210,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY5,      0xF4000214,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY6,      0xF4000218,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY7,      0xF400021C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY8,      0xF4000220,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY9,      0xF4000224,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY10,     0xF4000228,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY11,     0xF400022C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY12,     0xF4000230,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY13,     0xF4000234,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY14,     0xF4000238,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY15,     0xF400023C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY16,     0xF4000240,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY17,     0xF4000244,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY18,     0xF4000248,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY20,     0xF4000250,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY21,     0xF4000254,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY22,     0xF4000258,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY23,     0xF400025C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY26,     0xF4000268,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY27,     0xF400026C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY28,     0xF4000270,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY29,     0xF4000274,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY30,     0xF4000278,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32_BIT(VICVECTPRIORITY31,     0xF400027C,__READ_WRITE ,__vicvectpriority_bits);
__IO_REG32(    VICADDRESS,            0xF4000F00,__READ_WRITE );

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACIntStaus,          0xF4100000,__READ       ,__dmacintstaus_bits);
__IO_REG32_BIT(DMACIntTCStatus,       0xF4100004,__READ       ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACIntTCClear,        0xF4100008,__WRITE      ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACIntErrorStatus,    0xF410000C,__READ       ,__dmacinterrorstatus_bits);
__IO_REG32_BIT(DMACIntErrClr,         0xF4100010,__WRITE      ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRawIntTCStatus,    0xF4100014,__READ       ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRawIntErrorStatus, 0xF4100018,__READ       ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACEnbldChns,         0xF410001C,__READ       ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSoftBReq,          0xF4100020,__READ_WRITE ,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSoftSReq,          0xF4100024,__READ_WRITE ,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACConfiguration,     0xF4100030,__READ_WRITE ,__dmacconfiguration_bits);
__IO_REG32(    DMACC0SrcAddr,         0xF4100100,__READ_WRITE );
__IO_REG32(    DMACC0DestAddr,        0xF4100104,__READ_WRITE );
__IO_REG32_BIT(DMACC0LLI,             0xF4100108,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC0Control,         0xF410010C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC0Configuration,   0xF4100110,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC1SrcAddr,         0xF4100120,__READ_WRITE );
__IO_REG32(    DMACC1DestAddr,        0xF4100124,__READ_WRITE );
__IO_REG32_BIT(DMACC1LLI,             0xF4100128,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC1Control,         0xF410012C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC1Configuration,   0xF4100130,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC2SrcAddr,         0xF4100140,__READ_WRITE );
__IO_REG32(    DMACC2DestAddr,        0xF4100144,__READ_WRITE );
__IO_REG32_BIT(DMACC2LLI,             0xF4100148,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC2Control,         0xF410014C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC2Configuration,   0xF4100150,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC3SrcAddr,         0xF4100160,__READ_WRITE );
__IO_REG32(    DMACC3DestAddr,        0xF4100164,__READ_WRITE );
__IO_REG32_BIT(DMACC3LLI,             0xF4100168,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC3Control,         0xF410016C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC3Configuration,   0xF4100170,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC4SrcAddr,         0xF4100180,__READ_WRITE );
__IO_REG32(    DMACC4DestAddr,        0xF4100184,__READ_WRITE );
__IO_REG32_BIT(DMACC4LLI,             0xF4100188,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC4Control,         0xF410018C,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC4Configuration,   0xF4100190,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC5SrcAddr,         0xF41001A0,__READ_WRITE );
__IO_REG32(    DMACC5DestAddr,        0xF41001A4,__READ_WRITE );
__IO_REG32_BIT(DMACC5LLI,             0xF41001A8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC5Control,         0xF41001AC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC5Configuration,   0xF41001B0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC6SrcAddr,         0xF41001C0,__READ_WRITE );
__IO_REG32(    DMACC6DestAddr,        0xF41001C4,__READ_WRITE );
__IO_REG32_BIT(DMACC6LLI,             0xF41001C8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC6Control,         0xF41001CC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC6Configuration,   0xF41001D0,__READ_WRITE ,__dmaccconfiguration_bits);
__IO_REG32(    DMACC7SrcAddr,         0xF41001E0,__READ_WRITE );
__IO_REG32(    DMACC7DestAddr,        0xF41001E4,__READ_WRITE );
__IO_REG32_BIT(DMACC7LLI,             0xF41001E8,__READ_WRITE ,__dmacclli_bits);
__IO_REG32_BIT(DMACC7Control,         0xF41001EC,__READ_WRITE ,__dmacccontrol_bits);
__IO_REG32_BIT(DMACC7Configuration,   0xF41001F0,__READ_WRITE ,__dmaccconfiguration_bits);

/***************************************************************************
 **
 ** PORTA
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOADATA,             0xF08003FC,__READ_WRITE ,__gpioadata_bits);
__IO_REG32_BIT(GPIOAIS,               0xF0800804,__READ_WRITE ,__gpioais_bits);
__IO_REG32_BIT(GPIOAIBE,              0xF0800808,__READ_WRITE ,__gpioaibe_bits);
__IO_REG32_BIT(GPIOAIEV,              0xF080080C,__READ_WRITE ,__gpioaiev_bits);
__IO_REG32_BIT(GPIOAIE,               0xF0800810,__READ_WRITE ,__gpioaie_bits);
__IO_REG32_BIT(GPIOARIS,              0xF0800814,__READ       ,__gpioaris_bits);
__IO_REG32_BIT(GPIOAMIS,              0xF0800818,__READ       ,__gpioamis_bits);
__IO_REG32_BIT(GPIOAIC,               0xF080081C,__WRITE      ,__gpioaic_bits);

/***************************************************************************
 **
 ** PORTB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOBDATA,             0xF08013FC,__READ_WRITE ,__gpiobdata_bits);
__IO_REG32_BIT(GPIOBFR1,              0xF0801424,__READ_WRITE ,__gpiobfr1_bits);
__IO_REG32_BIT(GPIOBFR2,              0xF0801428,__READ_WRITE ,__gpiobfr2_bits);
__IO_REG32_BIT(GPIOBODE,              0xF0801C00,__READ_WRITE ,__gpiobode_bits);

/***************************************************************************
 **
 ** PORTC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOCDATA,             0xF08023FC,__READ_WRITE ,__gpiocdata_bits);
__IO_REG32_BIT(GPIOCDIR,              0xF0802400,__READ_WRITE ,__gpiocdir_bits);
__IO_REG32_BIT(GPIOCFR1,              0xF0802424,__READ_WRITE ,__gpiocfr1_bits);
__IO_REG32_BIT(GPIOCFR2,              0xF0802428,__READ_WRITE ,__gpiocfr2_bits);
__IO_REG32_BIT(GPIOCIS,               0xF0802804,__READ_WRITE ,__gpiocis_bits);
__IO_REG32_BIT(GPIOCIBE,              0xF0802808,__READ_WRITE ,__gpiocibe_bits);
__IO_REG32_BIT(GPIOCIEV,              0xF080280C,__READ_WRITE ,__gpiociev_bits);
__IO_REG32_BIT(GPIOCIE,               0xF0802810,__READ_WRITE ,__gpiocie_bits);
__IO_REG32_BIT(GPIOCRIS,              0xF0802814,__READ       ,__gpiocris_bits);
__IO_REG32_BIT(GPIOCMIS,              0xF0802818,__READ       ,__gpiocmis_bits);
__IO_REG32_BIT(GPIOCIC,               0xF080281C,__WRITE      ,__gpiocic_bits);
__IO_REG32_BIT(GPIOCODE,              0xF0802C00,__READ_WRITE ,__gpiocode_bits);

/***************************************************************************
 **
 ** PORTD
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIODDATA,             0xF08033FC,__READ_WRITE ,__gpioddata_bits);
__IO_REG32_BIT(GPIODFR1,              0xF0803424,__READ_WRITE ,__gpiodfr1_bits);
__IO_REG32_BIT(GPIODFR2,              0xF0803428,__READ_WRITE ,__gpiodfr2_bits);
__IO_REG32_BIT(GPIODIS,               0xF0803804,__READ_WRITE ,__gpiodis_bits);
__IO_REG32_BIT(GPIODIBE,              0xF0803808,__READ_WRITE ,__gpiodibe_bits);
__IO_REG32_BIT(GPIODIEV,              0xF080380C,__READ_WRITE ,__gpiodiev_bits);
__IO_REG32_BIT(GPIODIE,               0xF0803810,__READ_WRITE ,__gpiodie_bits);
__IO_REG32_BIT(GPIODRIS,              0xF0803814,__READ       ,__gpiodris_bits);
__IO_REG32_BIT(GPIODMIS,              0xF0803818,__READ       ,__gpiodmis_bits);
__IO_REG32_BIT(GPIODIC,               0xF080381C,__WRITE      ,__gpiodic_bits);

/***************************************************************************
 **
 ** PORTF
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOFDATA,             0xF08053FC,__READ_WRITE ,__gpiofdata_bits);
__IO_REG32_BIT(GPIOFDIR,              0xF0805400,__READ_WRITE ,__gpiofdir_bits);
__IO_REG32_BIT(GPIOFFR1,              0xF0805424,__READ_WRITE ,__gpioffr1_bits);
__IO_REG32_BIT(GPIOFFR2,              0xF0805428,__READ_WRITE ,__gpioffr2_bits);
__IO_REG32_BIT(GPIOFIS,               0xF0805804,__READ_WRITE ,__gpiofis_bits);
__IO_REG32_BIT(GPIOFIBE,              0xF0805808,__READ_WRITE ,__gpiofibe_bits);
__IO_REG32_BIT(GPIOFIEV,              0xF080580C,__READ_WRITE ,__gpiofiev_bits);
__IO_REG32_BIT(GPIOFIE,               0xF0805810,__READ_WRITE ,__gpiofie_bits);
__IO_REG32_BIT(GPIOFRIS,              0xF0805814,__READ       ,__gpiofris_bits);
__IO_REG32_BIT(GPIOFMIS,              0xF0805818,__READ       ,__gpiofmis_bits);
__IO_REG32_BIT(GPIOFIC,               0xF080581C,__WRITE      ,__gpiofic_bits);
__IO_REG32_BIT(GPIOFODE,              0xF0805C00,__READ_WRITE ,__gpiofode_bits);

/***************************************************************************
 **
 ** PORTG
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOGDATA,             0xF08063FC,__READ_WRITE ,__gpiogdata_bits);
__IO_REG32_BIT(GPIOGDIR,              0xF0806400,__READ_WRITE ,__gpiogdir_bits);
__IO_REG32_BIT(GPIOGFR1,              0xF0806424,__READ_WRITE ,__gpiogfr1_bits);

/***************************************************************************
 **
 ** PORTJ
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOJDATA,             0xF08083FC,__READ_WRITE ,__gpiojdata_bits);
__IO_REG32_BIT(GPIOJDIR,              0xF0808400,__READ_WRITE ,__gpiojdir_bits);
__IO_REG32_BIT(GPIOJFR1,              0xF0808424,__READ_WRITE ,__gpiojfr1_bits);
__IO_REG32_BIT(GPIOJFR2,              0xF0808428,__READ_WRITE ,__gpiojfr2_bits);

/***************************************************************************
 **
 ** PORTK
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOKDATA,             0xF08093FC,__READ_WRITE ,__gpiokdata_bits);
__IO_REG32_BIT(GPIOKDIR,              0xF0809400,__READ_WRITE ,__gpiokdir_bits);
__IO_REG32_BIT(GPIOKFR1,              0xF0809424,__READ_WRITE ,__gpiokfr1_bits);
__IO_REG32_BIT(GPIOKFR2,              0xF0809428,__READ_WRITE ,__gpiokfr2_bits);

/***************************************************************************
 **
 ** PORTL
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOLDATA,             0xF080A3FC,__READ_WRITE ,__gpioldata_bits);
__IO_REG32_BIT(GPIOLDIR,              0xF080A400,__READ_WRITE ,__gpioldir_bits);
__IO_REG32_BIT(GPIOLFR1,              0xF080A424,__READ_WRITE ,__gpiolfr1_bits);
__IO_REG32_BIT(GPIOLFR2,              0xF080A428,__READ_WRITE ,__gpiolfr2_bits);

/***************************************************************************
 **
 ** PORTM
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOMDATA,             0xF080B3FC,__READ_WRITE ,__gpiomdata_bits);
__IO_REG32_BIT(GPIOMDIR,              0xF080B400,__READ_WRITE ,__gpiomdir_bits);
__IO_REG32_BIT(GPIOMFR1,              0xF080B424,__READ_WRITE ,__gpiomfr1_bits);

/***************************************************************************
 **
 ** PORTN
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIONDATA,             0xF080C3FC,__READ_WRITE ,__gpiondata_bits);
__IO_REG32_BIT(GPIONDIR,              0xF080C400,__READ_WRITE ,__gpiondir_bits);
__IO_REG32_BIT(GPIONFR1,              0xF080C424,__READ_WRITE ,__gpionfr1_bits);
__IO_REG32_BIT(GPIONFR2,              0xF080C428,__READ_WRITE ,__gpionfr2_bits);
__IO_REG32_BIT(GPIONIS,               0xF080C804,__READ_WRITE ,__gpionis_bits);
__IO_REG32_BIT(GPIONIBE,              0xF080C808,__READ_WRITE ,__gpionibe_bits);
__IO_REG32_BIT(GPIONIEV,              0xF080C80C,__READ_WRITE ,__gpioniev_bits);
__IO_REG32_BIT(GPIONIE,               0xF080C810,__READ_WRITE ,__gpionie_bits);
__IO_REG32_BIT(GPIONRIS,              0xF080C814,__READ       ,__gpionris_bits);
__IO_REG32_BIT(GPIONMIS,              0xF080C818,__READ       ,__gpionmis_bits);
__IO_REG32_BIT(GPIONIC,               0xF080C81C,__WRITE      ,__gpionic_bits);

/***************************************************************************
 **
 ** PORTR
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIORDATA,             0xF080E3FC,__READ_WRITE ,__gpiordata_bits);
__IO_REG32_BIT(GPIORDIR,              0xF080E400,__READ_WRITE ,__gpiordir_bits);
__IO_REG32_BIT(GPIORFR1,              0xF080E424,__READ_WRITE ,__gpiorfr1_bits);
__IO_REG32_BIT(GPIORFR2,              0xF080E428,__READ_WRITE ,__gpiorfr2_bits);
__IO_REG32_BIT(GPIORIS,               0xF080E804,__READ_WRITE ,__gpioris_bits);
__IO_REG32_BIT(GPIORIBE,              0xF080E808,__READ_WRITE ,__gpioribe_bits);
__IO_REG32_BIT(GPIORIEV,              0xF080E80C,__READ_WRITE ,__gpioriev_bits);
__IO_REG32_BIT(GPIORIE,               0xF080E810,__READ_WRITE ,__gpiorie_bits);
__IO_REG32_BIT(GPIORRIS,              0xF080E814,__READ       ,__gpiorris_bits);
__IO_REG32_BIT(GPIORMIS,              0xF080E818,__READ       ,__gpiormis_bits);
__IO_REG32_BIT(GPIORIC,               0xF080E81C,__WRITE      ,__gpioric_bits);

/***************************************************************************
 **
 ** PORTT
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOTDATA,             0xF080F3FC,__READ_WRITE ,__gpiotdata_bits);
__IO_REG32_BIT(GPIOTDIR,              0xF080F400,__READ_WRITE ,__gpiotdir_bits);
__IO_REG32_BIT(GPIOTFR1,              0xF080F424,__READ_WRITE ,__gpiotfr1_bits);
__IO_REG32_BIT(GPIOTFR2,              0xF080F428,__READ_WRITE ,__gpiotfr2_bits);

/***************************************************************************
 **
 ** PORTU
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOUDATA,             0xF08043FC,__READ_WRITE ,__gpioudata_bits);
__IO_REG32_BIT(GPIOUDIR,              0xF0804400,__READ_WRITE ,__gpioudir_bits);
__IO_REG32_BIT(GPIOUFR1,              0xF0804424,__READ_WRITE ,__gpioufr1_bits);
__IO_REG32_BIT(GPIOUFR2,              0xF0804428,__READ_WRITE ,__gpioufr2_bits);

/***************************************************************************
 **
 ** PORTV
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOVDATA,             0xF08073FC,__READ_WRITE ,__gpiovdata_bits);
__IO_REG32_BIT(GPIOVDIR,              0xF0807400,__READ_WRITE ,__gpiovdir_bits);
__IO_REG32_BIT(GPIOVFR1,              0xF0807424,__READ_WRITE ,__gpiovfr1_bits);
__IO_REG32_BIT(GPIOVFR2,              0xF0807428,__READ_WRITE ,__gpiovfr2_bits);

/***************************************************************************
 **
 ** DMC MPMC0 (SDR)
 **
 ***************************************************************************/
__IO_REG32_BIT(dmc_memc_status_3,     0xF4300000,__READ       ,__dmc_memc_status_bits);
__IO_REG32_BIT(dmc_memc_cmd_3,        0xF4300004,__WRITE      ,__dmc_memc_cmd_bits);
__IO_REG32_BIT(dmc_direct_cmd_3,      0xF4300008,__WRITE      ,__dmc_direct_cmd_bits);
__IO_REG32_BIT(dmc_memory_cfg_3,      0xF430000C,__READ_WRITE ,__dmc_memory_cfg_bits);
__IO_REG32_BIT(dmc_refresh_prd_3,     0xF4300010,__READ_WRITE ,__dmc_refresh_prd_bits);
__IO_REG32_BIT(dmc_cas_latency_3,     0xF4300014,__READ_WRITE ,__dmc_cas_latency_3_bits);
__IO_REG32_BIT(dmc_t_dqss_3,          0xF4300018,__READ_WRITE ,__dmc_t_dqss_bits);
__IO_REG32_BIT(dmc_t_mrd_3,           0xF430001C,__READ_WRITE ,__dmc_t_mrd_bits);
__IO_REG32_BIT(dmc_t_ras_3,           0xF4300020,__READ_WRITE ,__dmc_t_ras_bits);
__IO_REG32_BIT(dmc_t_rc_3,            0xF4300024,__READ_WRITE ,__dmc_t_rc_bits);
__IO_REG32_BIT(dmc_t_rcd_3,           0xF4300028,__READ_WRITE ,__dmc_t_rcd_bits);
__IO_REG32_BIT(dmc_t_rfc_3,           0xF430002C,__READ_WRITE ,__dmc_t_rfc_bits);
__IO_REG32_BIT(dmc_t_rp_3,            0xF4300030,__READ_WRITE ,__dmc_t_rp_bits);
__IO_REG32_BIT(dmc_t_rrd_3,           0xF4300034,__READ_WRITE ,__dmc_t_rrd_bits);
__IO_REG32_BIT(dmc_t_wr_3,            0xF4300038,__READ_WRITE ,__dmc_t_wr_bits);
__IO_REG32_BIT(dmc_t_wtr_3,           0xF430003C,__READ_WRITE ,__dmc_t_wtr_bits);
__IO_REG32_BIT(dmc_t_xp_3,            0xF4300040,__READ_WRITE ,__dmc_t_xp_bits);
__IO_REG32_BIT(dmc_t_xsr_3,           0xF4300044,__READ_WRITE ,__dmc_t_xsr_bits);
__IO_REG32_BIT(dmc_t_esr_3,           0xF4300048,__READ_WRITE ,__dmc_t_esr_bits);
__IO_REG32_BIT(dmc_id_0_cfg_3,        0xF4300100,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_id_1_cfg_3,        0xF4300104,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_id_2_cfg_3,        0xF4300108,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_id_3_cfg_3,        0xF430010C,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_id_4_cfg_3,        0xF4300110,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_id_5_cfg_3,        0xF4300114,__READ_WRITE ,__dmc_id_cfg_3_bits);
__IO_REG32_BIT(dmc_chip_0_cfg_3,      0xF4300200,__READ_WRITE ,__dmc_chip_0_cfg_bits);
__IO_REG32_BIT(dmc_user_config_3,     0xF4300304,__WRITE      ,__dmc_user_config_bits);

/***************************************************************************
 **
 ** SMC MPMC0
 **
 ***************************************************************************/
__IO_REG32_BIT(smc_memc_status_3,     0xF4301000,__READ       ,__smc_memc_status_bits);
__IO_REG32_BIT(smc_memif_cfg_3,       0xF4301004,__READ       ,__smc_memif_cfg_bits);
__IO_REG32_BIT(smc_direct_cmd_3,      0xF4301010,__WRITE      ,__smc_direct_cmd_bits);
__IO_REG32_BIT(smc_set_cycles_3,      0xF4301014,__WRITE      ,__smc_set_cycles_bits);
__IO_REG32_BIT(smc_set_opmode_3,      0xF4301018,__WRITE      ,__smc_set_opmode_bits);
__IO_REG32_BIT(smc_sram_cycles0_0_3,  0xF4301100,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_1_3,  0xF4301120,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_2_3,  0xF4301140,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_3_3,  0xF4301160,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_0_3,       0xF4301104,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_1_3,       0xF4301124,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_2_3,       0xF4301144,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_3_3,       0xF4301164,__READ       ,__smc_opmode0_bits);

/***************************************************************************
 **
 ** DMC MPMC1 (DDR)
 **
 ***************************************************************************/
__IO_REG32_BIT(dmc_memc_status_5,     0xF4310000,__READ       ,__dmc_memc_status_bits);
__IO_REG32_BIT(dmc_memc_cmd_5,        0xF4310004,__WRITE      ,__dmc_memc_cmd_bits);
__IO_REG32_BIT(dmc_direct_cmd_5,      0xF4310008,__WRITE      ,__dmc_direct_cmd_bits);
__IO_REG32_BIT(dmc_memory_cfg_5,      0xF431000C,__READ_WRITE ,__dmc_memory_cfg_bits);
__IO_REG32_BIT(dmc_refresh_prd_5,     0xF4310010,__READ_WRITE ,__dmc_refresh_prd_bits);
__IO_REG32_BIT(dmc_cas_latency_5,     0xF4310014,__READ_WRITE ,__dmc_refresh_prd_5_bits);
__IO_REG32_BIT(dmc_t_dqss_5,          0xF4310018,__READ_WRITE ,__dmc_t_dqss_bits);
__IO_REG32_BIT(dmc_t_mrd_5,           0xF431001C,__READ_WRITE ,__dmc_t_mrd_bits);
__IO_REG32_BIT(dmc_t_ras_5,           0xF4310020,__READ_WRITE ,__dmc_t_ras_bits);
__IO_REG32_BIT(dmc_t_rc_5,            0xF4310024,__READ_WRITE ,__dmc_t_rc_bits);
__IO_REG32_BIT(dmc_t_rcd_5,           0xF4310028,__READ_WRITE ,__dmc_t_rcd_bits);
__IO_REG32_BIT(dmc_t_rfc_5,           0xF431002C,__READ_WRITE ,__dmc_t_rfc_bits);
__IO_REG32_BIT(dmc_t_rp_5,            0xF4310030,__READ_WRITE ,__dmc_t_rp_bits);
__IO_REG32_BIT(dmc_t_rrd_5,           0xF4310034,__READ_WRITE ,__dmc_t_rrd_bits);
__IO_REG32_BIT(dmc_t_wr_5,            0xF4310038,__READ_WRITE ,__dmc_t_wr_bits);
__IO_REG32_BIT(dmc_t_wtr_5,           0xF431003C,__READ_WRITE ,__dmc_t_wtr_bits);
__IO_REG32_BIT(dmc_t_xp_5,            0xF4310040,__READ_WRITE ,__dmc_t_xp_bits);
__IO_REG32_BIT(dmc_t_xsr_5,           0xF4310044,__READ_WRITE ,__dmc_t_xsr_bits);
__IO_REG32_BIT(dmc_t_esr_5,           0xF4310048,__READ_WRITE ,__dmc_t_esr_bits);
__IO_REG32_BIT(dmc_id_0_cfg_5,        0xF4310100,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_id_1_cfg_5,        0xF4310104,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_id_2_cfg_5,        0xF4310108,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_id_3_cfg_5,        0xF431010C,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_id_4_cfg_5,        0xF4310110,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_id_5_cfg_5,        0xF4310114,__READ_WRITE ,__dmc_id_cfg_5_bits);
__IO_REG32_BIT(dmc_chip_0_cfg_5,      0xF4310200,__READ_WRITE ,__dmc_chip_0_cfg_bits);
__IO_REG32_BIT(dmc_user_config_5,     0xF4310304,__WRITE      ,__dmc_user_config_5_bits);

/***************************************************************************
 **
 ** SMC MPMC1
 **
 ***************************************************************************/
__IO_REG32_BIT(smc_memc_status_5,     0xF4311000,__READ       ,__smc_memc_status_bits);
__IO_REG32_BIT(smc_memif_cfg_5,       0xF4311004,__READ       ,__smc_memif_cfg_bits);
__IO_REG32_BIT(smc_direct_cmd_5,      0xF4311010,__WRITE      ,__smc_direct_cmd_bits);
__IO_REG32_BIT(smc_set_cycles_5,      0xF4311014,__WRITE      ,__smc_set_cycles_bits);
__IO_REG32_BIT(smc_set_opmode_5,      0xF4311018,__WRITE      ,__smc_set_opmode_bits);
__IO_REG32_BIT(smc_sram_cycles0_0_5,  0xF4311100,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_1_5,  0xF4311120,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_2_5,  0xF4311140,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_sram_cycles0_3_5,  0xF4311160,__READ       ,__smc_sram_cycles0_bits);
__IO_REG32_BIT(smc_opmode0_0_5,       0xF4311104,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_1_5,       0xF4311124,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_2_5,       0xF4311144,__READ       ,__smc_opmode0_bits);
__IO_REG32_BIT(smc_opmode0_3_5,       0xF4311164,__READ       ,__smc_opmode0_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0DR,               0xF2000000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART0RSR,              0xF2000004,__READ_WRITE ,__uartrsr_bits);
#define UART0ECR      UART0RSR
__IO_REG32_BIT(UART0FR,               0xF2000018,__READ       ,__uart0fr_bits);
__IO_REG32_BIT(UART0ILPR,             0xF2000020,__READ_WRITE ,__uartilpr_bits);
__IO_REG32_BIT(UART0IBRD,             0xF2000024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART0FBRD,             0xF2000028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART0LCR_H,            0xF200002C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UART0CR,               0xF2000030,__READ_WRITE ,__uart0cr_bits);
__IO_REG32_BIT(UART0IFLS,             0xF2000034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART0IMSC,             0xF2000038,__READ_WRITE ,__uart0imsc_bits);
__IO_REG32_BIT(UART0RIS,              0xF200003C,__READ       ,__uart0ris_bits);
__IO_REG32_BIT(UART0MIS,              0xF2000040,__READ       ,__uart0mis_bits);
__IO_REG32_BIT(UART0ICR,              0xF2000044,__WRITE      ,__uart0icr_bits);
__IO_REG32_BIT(UART0DMACR,            0xF2000048,__READ_WRITE ,__uartdmacr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1DR,               0xF2001000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART1RSR,              0xF2001004,__READ_WRITE ,__uartrsr_bits);
#define UART1ECR      UART1RSR
__IO_REG32_BIT(UART1FR,               0xF2001018,__READ       ,__uart1fr_bits);
__IO_REG32_BIT(UART1IBRD,             0xF2001024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART1FBRD,             0xF2001028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART1LCR_H,            0xF200102C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UART1CR,               0xF2001030,__READ_WRITE ,__uart1cr_bits);
__IO_REG32_BIT(UART1IFLS,             0xF2001034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART1IMSC,             0xF2001038,__READ_WRITE ,__uart1imsc_bits);
__IO_REG32_BIT(UART1RIS,              0xF200103C,__READ       ,__uart1ris_bits);
__IO_REG32_BIT(UART1MIS,              0xF2001040,__READ       ,__uart1mis_bits);
__IO_REG32_BIT(UART1ICR,              0xF2001044,__WRITE      ,__uart1icr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2DR,               0xF2004000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART2RSR,              0xF2004004,__READ_WRITE ,__uartrsr_bits);
#define UART2ECR      UART2RSR
__IO_REG32_BIT(UART2FR,               0xF2004018,__READ       ,__uart2fr_bits);
__IO_REG32_BIT(UART2IBRD,             0xF2004024,__READ_WRITE ,__uartibrd_bits);
__IO_REG32_BIT(UART2FBRD,             0xF2004028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART2LCR_H,            0xF200402C,__READ_WRITE ,__uartlcr_h_bits);
__IO_REG32_BIT(UART2CR,               0xF2004030,__READ_WRITE ,__uart1cr_bits);
__IO_REG32_BIT(UART2IFLS,             0xF2004034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART2IMSC,             0xF2004038,__READ_WRITE ,__uart2imsc_bits);
__IO_REG32_BIT(UART2RIS,              0xF200403C,__READ       ,__uart2ris_bits);
__IO_REG32_BIT(UART2MIS,              0xF2004040,__READ       ,__uart2mis_bits);
__IO_REG32_BIT(UART2ICR,              0xF2004044,__WRITE      ,__uart2icr_bits);
__IO_REG32_BIT(UART2DMACR,            0xF2004048,__READ_WRITE ,__uartdmacr_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CR1,               0xF0070000,__READ_WRITE ,__i2ccr1_bits);
__IO_REG32_BIT(I2C0DBR,               0xF0070004,__READ_WRITE ,__i2cdbr_bits);
__IO_REG32_BIT(I2C0AR,                0xF0070008,__READ_WRITE ,__i2car_bits);
__IO_REG32_BIT(I2C0CR2,               0xF007000C,__READ_WRITE ,__i2ccr2_bits);
#define I2C0SR      I2C0CR2
#define I2C0SR_bit  I2C0CR2_bit
__IO_REG32_BIT(I2C0PRS,               0xF0070010,__READ_WRITE ,__i2cprs_bits);
__IO_REG32_BIT(I2C0IE,                0xF0070014,__READ_WRITE ,__i2cie_bits);
__IO_REG32_BIT(I2C0IR,                0xF0070018,__READ_WRITE ,__i2cir_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CR1,               0xF0071000,__READ_WRITE ,__i2ccr1_bits);
__IO_REG32_BIT(I2C1DBR,               0xF0071004,__READ_WRITE ,__i2cdbr_bits);
__IO_REG32_BIT(I2C1AR,                0xF0071008,__READ_WRITE ,__i2car_bits);
__IO_REG32_BIT(I2C1CR2,               0xF007100C,__READ_WRITE ,__i2ccr2_bits);
#define I2C1SR      I2C1CR2
#define I2C1SR_bit  I2C1CR2_bit
__IO_REG32_BIT(I2C1PRS,               0xF0071010,__READ_WRITE ,__i2cprs_bits);
__IO_REG32_BIT(I2C1IE,                0xF0071014,__READ_WRITE ,__i2cie_bits);
__IO_REG32_BIT(I2C1IR,                0xF0071018,__READ_WRITE ,__i2cir_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,               0xF2002000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,               0xF2002004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,                0xF2002008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,                0xF200200C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,              0xF2002010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,              0xF2002014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,               0xF2002018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,               0xF200201C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,               0xF2002020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,             0xF2002024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,               0xF2003000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,               0xF2003004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,                0xF2003008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,                0xF200300C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,              0xF2003010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,              0xF2003014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,               0xF2003018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,               0xF200301C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,               0xF2003020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,             0xF2003024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** UDC2AB
 **
 ***************************************************************************/
__IO_REG32_BIT(UDINTSTS,              0xF4400000,__READ_WRITE ,__udintsts_bits);
__IO_REG32_BIT(UDINTENB,              0xF4400004,__READ_WRITE ,__udintenb_bits);
__IO_REG32_BIT(UDMWTOUT,              0xF4400008,__READ_WRITE ,__udmwtout_bits);
__IO_REG32_BIT(UDC2STSET,             0xF440000C,__READ_WRITE ,__udc2stset_bits);
__IO_REG32_BIT(UDMSTSET,              0xF4400010,__READ_WRITE ,__udmstset_bits);
__IO_REG32_BIT(DMACRDREQ,             0xF4400014,__READ_WRITE ,__dmacrdreq_bits);
__IO_REG32(    DMACRDVL,              0xF4400018,__READ       );
__IO_REG32_BIT(UDC2RDREQ,             0xF440001C,__READ_WRITE ,__udc2rdreq_bits);
__IO_REG32_BIT(UDC2RDVL,              0xF4400020,__READ       ,__udc2rdvl_bits);
__IO_REG32_BIT(ARBTSET,               0xF440003C,__READ_WRITE ,__arbtset_bits);
__IO_REG32(    UDMWSADR,              0xF4400040,__READ_WRITE );
__IO_REG32(    UDMWEADR,              0xF4400044,__READ_WRITE );
__IO_REG32(    UDMWCADR,              0xF4400048,__READ       );
__IO_REG32(    UDMWAHBADR,            0xF440004C,__READ       );
__IO_REG32(    UDMRSADR,              0xF4400050,__READ_WRITE );
__IO_REG32(    UDMREADR,              0xF4400054,__READ_WRITE );
__IO_REG32(    UDMRCADR,              0xF4400058,__READ       );
__IO_REG32(    UDMRAHBADR,            0xF440005C,__READ       );
__IO_REG32_BIT(UDPWCTL,               0xF4400080,__READ_WRITE ,__udpwctl_bits);
__IO_REG32_BIT(UDMSTSTS,              0xF4400084,__READ       ,__udmststs_bits);
__IO_REG32(    UDTOUTCNT,             0xF4400088,__READ       );
__IO_REG32_BIT(UD2ADR,                0xF4400200,__READ_WRITE ,__ud2adr_bits);
__IO_REG32_BIT(UD2FRM,                0xF4400204,__READ_WRITE ,__ud2frm_bits);
__IO_REG32_BIT(UD2TMD,                0xF4400208,__READ_WRITE ,__ud2tmd_bits);
__IO_REG32_BIT(UD2CMD,                0xF440020C,__READ_WRITE ,__ud2cmd_bits);
__IO_REG32_BIT(UD2BRQ,                0xF4400210,__READ       ,__ud2brq_bits);
__IO_REG32_BIT(UD2WVL,                0xF4400214,__READ       ,__ud2wvl_bits);
__IO_REG32_BIT(UD2WIDX,               0xF4400218,__READ       ,__ud2widx_bits);
__IO_REG32_BIT(UD2WLGTH,              0xF440021C,__READ       ,__ud2wlgth_bits);
__IO_REG32_BIT(UD2INT,                0xF4400220,__READ_WRITE ,__ud2int_bits);
__IO_REG32_BIT(UD2INTEP,              0xF4400224,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UD2INTEPMSK,           0xF4400228,__READ_WRITE ,__ud2intepmsk_bits);
__IO_REG32_BIT(UD2INTRX0,             0xF440022C,__READ_WRITE ,__ud2intrx0_bits);
__IO_REG32_BIT(UD2EP0MSZ,             0xF4400230,__READ_WRITE ,__ud2ep0msz_bits);
__IO_REG32_BIT(UD2EP0STS,             0xF4400234,__READ       ,__ud2ep0sts_bits);
__IO_REG32_BIT(UD2EP0DSZ,             0xF4400238,__READ       ,__ud2ep0dsz_bits);
__IO_REG32_BIT(UD2EP0FIFO,            0xF440023C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UD2EP1MSZ,             0xF4400240,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UD2EP1STS,             0xF4400244,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UD2EP1DSZ,             0xF4400248,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UD2EP1FIFO,            0xF440024C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UD2EP2MSZ,             0xF4400250,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UD2EP2STS,             0xF4400254,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UD2EP2DSZ,             0xF4400258,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UD2EP2FIFO,            0xF440025C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UD2EP3MSZ,             0xF4400260,__READ_WRITE ,__ud2epmsz_bits);
__IO_REG32_BIT(UD2EP3STS,             0xF4400264,__READ_WRITE ,__ud2ep1sts_bits);
__IO_REG32_BIT(UD2EP3DSZ,             0xF4400268,__READ       ,__ud2ep1dsz_bits);
__IO_REG32_BIT(UD2EP3FIFO,            0xF440026C,__READ_WRITE ,__ud2epfifo_bits);
__IO_REG32_BIT(UD2INTNAK,             0xF4400330,__READ_WRITE ,__ud2intep_bits);
__IO_REG32_BIT(UD2INTNAKMSK,          0xF4400334,__READ_WRITE ,__ud2intepmsk_bits);

/***************************************************************************
 **
 ** LCDDA
 **
 ***************************************************************************/
__IO_REG32_BIT(LDACR0,                0xF2050000,__READ_WRITE ,__ldacr0_bits);
__IO_REG32_BIT(LDADRSRC1,             0xF2050004,__READ_WRITE ,__ldadrsrc1_bits);
__IO_REG32_BIT(LDADRSRC0,             0xF2050008,__READ_WRITE ,__ldadrsrc0_bits);
__IO_REG32_BIT(LDAFCPSRC1,            0xF205000C,__READ_WRITE ,__ldafcpsrc_bits);
__IO_REG32_BIT(LDAEFCPSRC1,           0xF2050010,__READ_WRITE ,__ldafcpsrc_bits);
__IO_REG32_BIT(LDADVSRC1,             0xF2050014,__READ_WRITE ,__ldadvsrc1_bits);
__IO_REG32_BIT(LDACR2,                0xF2050018,__READ_WRITE ,__ldacr2_bits);
__IO_REG32_BIT(LDADXDST,              0xF205001C,__READ_WRITE ,__ldadxdst_bits);
__IO_REG32_BIT(LDADYDST,              0xF2050020,__READ_WRITE ,__ldadydst_bits);
__IO_REG32_BIT(LDASSIZE,              0xF2050024,__READ_WRITE ,__ldassize_bits);
__IO_REG32_BIT(LDADSIZE,              0xF2050028,__READ_WRITE ,__ldadsize_bits);
__IO_REG32(    LDAS0AD,               0xF205002C,__READ_WRITE );
__IO_REG32(    LDADAD,                0xF2050030,__READ_WRITE );
__IO_REG32_BIT(LDACR1,                0xF2050034,__READ_WRITE ,__ldacr1_bits);
__IO_REG32_BIT(LDADVSRC0,             0xF2050038,__READ_WRITE ,__ldadvsrc0_bits);

/***************************************************************************
 **
 ** TSI
 **
 ***************************************************************************/
__IO_REG32_BIT(TSICR0,                0xF00601F0,__READ_WRITE ,__tsicr0_bits);
__IO_REG32_BIT(TSICR1,                0xF00601F4,__READ_WRITE ,__tsicr1_bits);

/***************************************************************************
 **
 ** CMSI
 **
 ***************************************************************************/
__IO_REG32_BIT(CMSCR,                 0xF2020000,__READ_WRITE ,__cmscr_bits);
__IO_REG32_BIT(CMSCV,                 0xF2020004,__READ_WRITE ,__cmscv_bits);
__IO_REG32_BIT(CMSCVP0,               0xF2020008,__READ_WRITE ,__cmscvp0_bits);
__IO_REG32_BIT(CMSCVP1,               0xF202000C,__READ_WRITE ,__cmscvp1_bits);
__IO_REG32_BIT(CMSYD,                 0xF2020010,__WRITE      ,__cmsyd_bits);
__IO_REG32_BIT(CMSUD,                 0xF2020014,__WRITE      ,__cmsud_bits);
__IO_REG32_BIT(CMSVD,                 0xF2020018,__WRITE      ,__cmsvd_bits);
__IO_REG32(    CMSFPT,                0xF2020020,__READ       );
__IO_REG32_BIT(CMSCSTR,               0xF2020024,__READ_WRITE ,__cmscstr_bits);
__IO_REG32_BIT(CMSTS,                 0xF2020030,__READ_WRITE ,__cmsts_bits);
__IO_REG32_BIT(CMSTE,                 0xF2020034,__READ_WRITE ,__cmste_bits);
__IO_REG32_BIT(CMSSCDMA,              0xF2020040,__WRITE      ,__cmsscdma_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTCDATA,               0xF0030000,__READ       );
__IO_REG32(    RTCCOMP,               0xF0030004,__WRITE      );
__IO_REG32(    RTCPRST,               0xF0030008,__WRITE      );
__IO_REG32_BIT(MLDALMINV,             0xF0030100,__WRITE      ,__mldalminv_bits);
__IO_REG32_BIT(MLDALMSEL,             0xF0030104,__WRITE      ,__mldalmsel_bits);
__IO_REG32_BIT(ALMCNTCR,              0xF0030108,__READ_WRITE ,__almcntcr_bits);
__IO_REG32_BIT(ALMPATTERN,            0xF003010C,__WRITE      ,__almpattern_bits);
__IO_REG32_BIT(MLDCNTCR,              0xF0030110,__READ_WRITE ,__mldcntcr_bits);
__IO_REG32_BIT(MLDFRQ,                0xF0030114,__WRITE      ,__mldfrq_bits);
__IO_REG32_BIT(RTCALMINTCTR,          0xF0030200,__READ_WRITE ,__rtcalmintctr_bits);
__IO_REG32_BIT(RTCALMMIS,             0xF0030204,__READ_WRITE ,__rtcalmmis_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADREG0L,               0xF0080000,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG0H,               0xF0080004,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG1L,               0xF0080008,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG1H,               0xF008000C,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG2L,               0xF0080010,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG2H,               0xF0080014,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG3L,               0xF0080018,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG3H,               0xF008001C,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG4L,               0xF0080020,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG4H,               0xF0080024,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG5L,               0xF0080028,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG5H,               0xF008002C,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG6L,               0xF0080030,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG6H,               0xF0080034,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREG7L,               0xF0080038,__READ       ,__adregl_bits);
__IO_REG32_BIT(ADREG7H,               0xF008003C,__READ       ,__adregh_bits);
__IO_REG32_BIT(ADREGSPL,              0xF0080040,__READ       ,__adcomregl_bits);
__IO_REG32_BIT(ADREGSPH,              0xF0080044,__READ       ,__adcomregh_bits);
__IO_REG32_BIT(ADCOMREGL,             0xF0080048,__READ_WRITE ,__adcomregl_bits);
__IO_REG32_BIT(ADCOMREGH,             0xF008004C,__READ_WRITE ,__adcomregh_bits);
__IO_REG32_BIT(ADMOD0,                0xF0080050,__READ_WRITE ,__admod0_bits);
__IO_REG32_BIT(ADMOD1,                0xF0080054,__READ_WRITE ,__admod1_bits);
__IO_REG32_BIT(ADMOD2,                0xF0080058,__READ_WRITE ,__admod2_bits);
__IO_REG32_BIT(ADMOD3,                0xF008005C,__READ_WRITE ,__admod3_bits);
__IO_REG32_BIT(ADMOD4,                0xF0080060,__READ_WRITE ,__admod4_bits);
__IO_REG32_BIT(ADCLK,                 0xF0080070,__READ_WRITE ,__adclk_bits);
__IO_REG32_BIT(ADIE,                  0xF0080074,__READ_WRITE ,__adie_bits);
__IO_REG32_BIT(ADIS,                  0xF0080078,__READ       ,__adis_bits);
__IO_REG32_BIT(ADIC,                  0xF008007C,__WRITE      ,__adic_bits);

/***************************************************************************
 **
 ** WDOG
 **
 ***************************************************************************/
__IO_REG32(    WdogLoad,              0xF0010000,__READ_WRITE );
__IO_REG32(    WdogValue,             0xF0010004,__READ       );
__IO_REG32_BIT(WdogControl,           0xF0010008,__READ_WRITE ,__wdogcontrol_bits);
__IO_REG32(    WdogIntClr,            0xF001000C,__WRITE      );
__IO_REG32_BIT(WdogRIS,               0xF0010010,__READ       ,__wdogris_bits);
__IO_REG32_BIT(WdogMIS,               0xF0010014,__READ       ,__wdogmis_bits);
__IO_REG32_BIT(WdogLock,              0xF0010C00,__READ       ,__wdoglock_bits);

/***************************************************************************
 **
 ** PMC
 **
 ***************************************************************************/
__IO_REG32_BIT(BPARELE,               0xF0020200,__READ_WRITE ,__bparele_bits);
__IO_REG32_BIT(BPDRELE,               0xF0020204,__READ_WRITE ,__bpdrele_bits);
__IO_REG32_BIT(BPPRELE,               0xF0020208,__READ_WRITE ,__bpprele_bits);
/*__IO_REG32_BIT(BPXRELE,               0xF002020C,__READ_WRITE ,__bpxrele_bits);*/
__IO_REG32_BIT(BPAEDGE,               0xF0020220,__READ_WRITE ,__bpaedge_bits);
__IO_REG32_BIT(BPDEDGE,               0xF0020224,__READ_WRITE ,__bpdedge_bits);
__IO_REG32_BIT(BPARINT,               0xF0020240,__READ_WRITE ,__bparint_bits);
__IO_REG32_BIT(BPDRINT,               0xF0020244,__READ_WRITE ,__bpdrint_bits);
__IO_REG32_BIT(BPPRINT,               0xF0020248,__READ_WRITE ,__bpprint_bits);
/*__IO_REG32_BIT(BPXRINT,               0xF002024C,__READ_WRITE ,__bpxrint_bits);*/
__IO_REG32_BIT(PMCDRV,                0xF0020260,__READ_WRITE ,__pmcdrv_bits);
__IO_REG32_BIT(DMCCKECTL,             0xF0020280,__READ_WRITE ,__dmcckectl_bits);
__IO_REG32_BIT(PMCCTL,                0xF0020300,__READ_WRITE ,__pmcctl_bits);
/*__IO_REG32_BIT(PMCWV1,                0xF0020400,__READ_WRITE ,__pmcwv1_bits);*/
/*__IO_REG32_BIT(PMCRES,                0xF002041C,__READ_WRITE ,__pmcres_bits);*/

/***************************************************************************
 **
 ** USB HOST (OHCI) CONTROLLER
 **
 ***************************************************************************/
__IO_REG32_BIT(HcRevision,            0xF4500000,__READ       ,__hcrevision_bits);
__IO_REG32_BIT(HcControl,             0xF4500004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(HcCommandStatus,       0xF4500008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(HcInterruptStatus,     0xF450000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(HcInterruptEnable,     0xF4500010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HcInterruptDisable,    0xF4500014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(HcHCCA,                0xF4500018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(HcPeriodCurrentED,     0xF450001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(HcControlHeadED,       0xF4500020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(HcControlCurrentED,    0xF4500024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(HcBulkHeadED,          0xF4500028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(HcBulkCurrentED,       0xF450002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(HcDoneHead,            0xF4500030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(HcFmInterval,          0xF4500034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(HcFmRemaining,         0xF4500038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(HcFmNumber,            0xF450003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(HcPeriodStart,         0xF4500040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(HcLSThreshold,         0xF4500044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(HcRhDescriptorA,       0xF4500048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(HcRhDescripterB,       0xF450004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(HcRhStatus,            0xF4500050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(HcRhPortStatus,        0xF4500054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32(		 HcBCR0,        				0xF4500080,__READ_WRITE );

/***************************************************************************
 **
 ** OFD (Oscillation Frequency Detector)
 **
 ***************************************************************************/
__IO_REG32_BIT(CLKSCR1,               0xF0090000,__READ_WRITE ,__clkscr1_bits);
__IO_REG32_BIT(CLKSCR2,               0xF0090004,__READ_WRITE ,__clkscr2_bits);
__IO_REG32_BIT(CLKSCR3,               0xF0090008,__READ_WRITE ,__clkscr3_bits);
__IO_REG32_BIT(CLKSMN,                0xF0090010,__READ_WRITE ,__clksmn_bits);
__IO_REG32_BIT(CLKSMX,                0xF0090020,__READ_WRITE ,__clksmx_bits);

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
 **  Interrupt sources
 **
 ***************************************************************************/
#define _INTR_WDT              0
#define _INTR_RTC              1
#define _INTR_TIMER01          2
#define _INTR_TIMER23          3
#define _INTR_TIMER45          4
#define _INTR_GPIOD            5
#define _INTR_I2C0             6
#define _INTR_I2C1             7
#define _INTR_ADC              8
#define _INTR_UART2            9
#define _INTR_UART0            10
#define _INTR_UART1            11
#define _INTR_SSP0             12
#define _INTR_SSP1             13
#define _INTR_NDFC             14
#define _INTR_CMSIF            15
#define _INTR_DMA_ERR          16
#define _INTR_DMA_END          17
#define _INTR_LCDC             18
#define _INTR_LCDDA            20
#define _INTR_USB              21
#define _INTR_SDHC             22
#define _INTR_I2S              23
#define _INTR_GPIOR            26
#define _INTR_USBH             27
#define _INTR_GPION            28
#define _INTR_GPIOF            29
#define _INTR_GPIOC            30
#define _INTR_GPIOA            31

/***************************************************************************
 **
 **  DMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define _DMA_UTART0_TX        0
#define _DMA_UTART0_RX        1
#define _DMA_SSP0_TX          2
#define _DMA_SSP0_RX          3
#define _DMA_NANDC0           4
#define _DMA_CMSI             5
#define _DMA_UTART2_TX        6
#define _DMA_UTART2_RX        7
#define _DMA_SDHC_WR_REQ      8
#define _DMA_SDHC_RD_REQ      9
#define _DMA_I2S0             10
#define _DMA_I2S1             11
#define _DMA_SSP1_TX          12
#define _DMA_SSP1_RX          13
#define _DMA_LCDDA            14

#endif    /* __IOTMPA900CM_H */
