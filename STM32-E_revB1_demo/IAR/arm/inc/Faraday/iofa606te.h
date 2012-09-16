/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Faraday FA606TE
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2008
 **
 **    $Revision: 30247 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOFA606TE_H
#define __IOFA606TE_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    FIA320DA0 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C specific declarations  ************************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* ID Number 0 Register (IDNMBR0) */
typedef struct{
__REG32 ECOID           : 4;
__REG32 VERID           : 4;
__REG32 SYSID           : 8;
__REG32 DeviceID        :16;
} __idnmbr0_bits;

/* OSC Control Register (OSCC) */
typedef struct{
__REG32 OSCHOFFLF       : 1;
__REG32 OSCLSTABLE      : 1;
__REG32 RTCLSEL         : 1;
__REG32 OSCLTRI         : 1;
__REG32                 : 4;
__REG32 OSCHOFFHF       : 1;
__REG32 OSCHSTABLE      : 1;
__REG32                 : 1;
__REG32 OSCHTRI         : 1;
__REG32                 :20;
} __oscc_bits;

/* Power Mode Register (PMODE) */
typedef struct{
__REG32 MODE            : 2;
__REG32 FCS             : 1;
__REG32                 : 1;
__REG32 DIVAHBCLK       : 3;
__REG32                 :25;
} __pmode_bits;

/* Power Manager Control Register (PMCR) */
typedef struct{
__REG32 WEGPIO          :16;
__REG32 WERTC           : 1;
__REG32 WDTCLR          : 1;
__REG32 WAITPD          : 1;
__REG32 PWRLOWMSK       : 1;
__REG32                 :12;
} __pmcr_bits;

/* Power Manager Edge Detect (PED) Register */
typedef struct{
__REG32 GPIOFE0         : 1;
__REG32 GPIOFE1         : 1;
__REG32 GPIOFE2         : 1;
__REG32 GPIOFE3         : 1;
__REG32 GPIOFE4         : 1;
__REG32 GPIOFE5         : 1;
__REG32 GPIOFE6         : 1;
__REG32 GPIOFE7         : 1;
__REG32 GPIOFE8         : 1;
__REG32 GPIOFE9         : 1;
__REG32 GPIOFE10        : 1;
__REG32 GPIOFE11        : 1;
__REG32 GPIOFE12        : 1;
__REG32 GPIOFE13        : 1;
__REG32 GPIOFE14        : 1;
__REG32 GPIOFE15        : 1;
__REG32 GPIORE0         : 1;
__REG32 GPIORE1         : 1;
__REG32 GPIORE2         : 1;
__REG32 GPIORE3         : 1;
__REG32 GPIORE4         : 1;
__REG32 GPIORE5         : 1;
__REG32 GPIORE6         : 1;
__REG32 GPIORE7         : 1;
__REG32 GPIORE8         : 1;
__REG32 GPIORE9         : 1;
__REG32 GPIORE10        : 1;
__REG32 GPIORE11        : 1;
__REG32 GPIORE12        : 1;
__REG32 GPIORE13        : 1;
__REG32 GPIORE14        : 1;
__REG32 GPIORE15        : 1;
} __ped_bits;

/* Power Manager Edge Detect Status Register (PEDSR) */
typedef struct{
__REG32 GPIOED0         : 1;
__REG32 GPIOED1         : 1;
__REG32 GPIOED2         : 1;
__REG32 GPIOED3         : 1;
__REG32 GPIOED4         : 1;
__REG32 GPIOED5         : 1;
__REG32 GPIOED6         : 1;
__REG32 GPIOED7         : 1;
__REG32 GPIOED8         : 1;
__REG32 GPIOED9         : 1;
__REG32 GPIOED10        : 1;
__REG32 GPIOED11        : 1;
__REG32 GPIOED12        : 1;
__REG32 GPIOED13        : 1;
__REG32 GPIOED14        : 1;
__REG32 GPIOED15        : 1;
__REG32                 :16;
} __pedsr_bits;

/* Power Manager Status Register (PMSR) */
typedef struct{
__REG32 CKEHLOW         : 1;
__REG32 PH              : 1;
__REG32 RDH             : 1;
__REG32                 : 5;
__REG32 HWR             : 1;
__REG32 WDT             : 1;
__REG32 SMR             : 1;
__REG32                 : 5;
__REG32 IntTurbo        : 1;
__REG32 IntFCS          : 1;
__REG32 IntPwrLow       : 1;
__REG32 PwrLowSts       : 1;
__REG32                 :12;
} __pmsr_bits;

/* Power Manager GPIO Sleep State Register (PGSR) */
typedef struct{
__REG32 SS0             : 1;
__REG32 SS1             : 1;
__REG32 SS2             : 1;
__REG32 SS3             : 1;
__REG32 SS4             : 1;
__REG32 SS5             : 1;
__REG32 SS6             : 1;
__REG32 SS7             : 1;
__REG32 SS8             : 1;
__REG32 SS9             : 1;
__REG32 SS10            : 1;
__REG32 SS11            : 1;
__REG32 SS12            : 1;
__REG32 SS13            : 1;
__REG32 SS14            : 1;
__REG32 SS15            : 1;
__REG32 SS16            : 1;
__REG32                 :15;
} __pgsr_bits;

/* Multi-Function Port Setting Register (MFPSR) */
typedef struct{
__REG32 TriAHBDbg       : 1;
__REG32 TriAHBDis       : 1;
__REG32 IrDA1WireSel    : 1;
__REG32 AC97PinSel      : 1;
__REG32 AC97ClkSel      : 1;
__REG32 I2SClkSel       : 1;
__REG32 SspClkSel       : 1;
__REG32                 : 1;
__REG32 UartClkSel      : 1;
__REG32 IrdaClkSel      : 1;
__REG32 Pwm0PinSel      : 1;
__REG32 Pwm1PinSel      : 1;
__REG32 IrdaPinSel      : 1;
__REG32 AC97ClkOutSel   : 1;
__REG32 ModemPinSel     : 1;
__REG32 Dma1PinSel      : 1;
__REG32 Dma0PinSel      : 1;
__REG32 DebugSel        : 1;
__REG32                 :14;
} __mfpsr_bits;

/* Misc Register (MISC) */
typedef struct{
__REG32 TURNDIS         : 1;
__REG32                 : 7;
__REG32 PDCNT0          : 1;
__REG32 PDCNT1          : 1;
__REG32 PDCNT2          : 1;
__REG32 PDCNT3          : 1;
__REG32 PDCNT4          : 1;
__REG32 PDCNT5          : 1;
__REG32 PDCNT6          : 1;
__REG32                 :17;
} __misc_bits;

/* PLL/DLL Control Register 0 (PDLLCR0) */
typedef struct{
__REG32 PLL1DIS         : 1;
__REG32 PLL1STABLE      : 1;
__REG32 PLL1STSEL       : 1;
__REG32 PLL1NS          : 9;
__REG32 PLL1FRANG       : 1;
__REG32                 : 3;
__REG32 DLLDIS          : 1;
__REG32 DLLSTABLE       : 1;
__REG32 DLLSTSEL        : 1;
__REG32 DLLFRANG        : 1;
__REG32 HCLKOUTDIS      : 4;
__REG32                 : 8;
} __pdllcr0_bits;

/* PLL/DLL Control Register 1 (PDLLCR1) */
typedef struct{
__REG32 PLL3DIS         : 1;
__REG32 PLL3STABLE      : 1;
__REG32 PLL3STSEL       : 1;
__REG32                 : 5;
__REG32 PLL2DIS         : 1;
__REG32 PLL2STABLE      : 1;
__REG32 PLL2STSEL       : 1;
__REG32                 : 5;
__REG32 I2SCLKDIV       : 4;
__REG32 PWMCLKDIV       : 4;
__REG32                 : 8;
} __pdllcr1_bits;

/* AHB Module Clock Off Control Register (AHBMCLKOFF) */
typedef struct{
__REG32                 : 1;
__REG32 APBBRGOFF       : 1;
__REG32                 : 1;
__REG32 SMCOFF          : 1;
__REG32 EBIOFF          : 1;
__REG32 SDRAMOFF        : 1;
__REG32                 : 1;
__REG32 DMAOFF          : 1;
__REG32                 : 4;
__REG32 MACOFF          : 1;
__REG32                 : 1;
__REG32 USBDOFF         : 1;
__REG32                 :17;
} __ahbmclkoff_bits;

/* APB Module Clock Off Control Register (APBMCLKOFF) */
typedef struct{
__REG32                 : 1;
__REG32 CFCOFF          : 1;
__REG32 SSPOFF          : 1;
__REG32 FFUART          : 1;
__REG32 BTUART          : 1;
__REG32 SDCOFF          : 1;
__REG32 I2S_AC97OFF     : 1;
__REG32                 : 1;
__REG32 STUART          : 1;
__REG32                 : 2;
__REG32 IrDAOFF         : 1;
__REG32                 : 5;
__REG32 TIMEROFF        : 1;
__REG32 WDTOFF          : 1;
__REG32 RTCOFF          : 1;
__REG32 GPIOOFF         : 1;
__REG32 INTCOFF         : 1;
__REG32 I2COFF          : 1;
__REG32 PWMOFF          : 1;
__REG32                 : 8;
} __apbmclkoff_bits;

/* Driving Capability and Slew Rate Control Register 0 (DCSRCR0) */
typedef struct{
__REG32 SRAM_DCSR       : 4;
__REG32 SDCLK_DCSR      : 4;
__REG32 DQM_DCSR        : 4;
__REG32 CKE_DCSR        : 4;
__REG32 SDRAMCTL_DCSR   : 4;
__REG32 SDRAMCS_DCSR    : 4;
__REG32 EBIDATA_DCSR    : 4;
__REG32 EBICTRL_DCSR    : 4;
} __dcsrcr0_bits;

/* Driving Capability and Slew Rate Control Register 1 (DCSRCR1). */
typedef struct{
__REG32 INTC_DCSR       : 4;
__REG32 GPIO_DCSR       : 4;
__REG32 CFC_DCSR        : 4;
__REG32 MAC_DCSR        : 4;
__REG32 I2C_DCSR        : 4;
__REG32                 : 4;
__REG32 USBDEV_DCSR     : 4;
__REG32 TRIAHB_DCSR     : 4;
} __dcsrcr1_bits;

/* Driving Capability and Slew Rate Control Register 2 (DCSRCR2). */
typedef struct{
__REG32 STUART_DCSR     : 4;
__REG32 IrDA_DCSR       : 4;
__REG32 BTUART_DCSR     : 4;
__REG32 FFUART_DCSR     : 4;
__REG32 PMU_DCSR        : 4;
__REG32 I2SAC97_DCSR    : 4;
__REG32 SSP_DCSR        : 4;
__REG32 SD_DCSR         : 4;
} __dcsrcr2_bits;

/* SDRAM Signal Hold Time Control (SDRAMHTC) */
typedef struct{
__REG32                 :12;
__REG32 SDCLK_DCSR      : 1;
__REG32 DQM_DCSR        : 1;
__REG32 CKE_DCSR        : 1;
__REG32 SDRAMCTL_DCSR   : 1;
__REG32 SDRAMCS_DCSR    : 1;
__REG32 EBIDATA_DCSR    : 1;
__REG32 EBICTRL_DCSR    : 1;
__REG32                 : 1;
__REG32 DAT_WCLK_DLY    : 4;
__REG32 CTL_WCLK_DLY    : 4;
__REG32 RCLK_DLY        : 4;
} __sdramhtc_bits;

/* AHB DMA REQ/ACK Connection Configuration Status (AHBDMAREQACK) */
typedef struct{
__REG32 CH0_REQACK_PAIR : 4;
__REG32 CH1_REQACK_PAIR : 4;
__REG32 CH2_REQACK_PAIR : 4;
__REG32 CH3_REQACK_PAIR : 4;
__REG32 CH4_REQACK_PAIR : 4;
__REG32 CH5_REQACK_PAIR : 4;
__REG32 CH6_REQACK_PAIR : 4;
__REG32 CH7_REQACK_PAIR : 4;
} __ahbdmareqack_bits;

/* Jumper Setting Status (JMPSETTSTA) */
typedef struct{
__REG32 PLLDIS          : 1;
__REG32 INI_MBW         : 2;
__REG32 INTCPUOFF       : 1;
__REG32 PLL1SETTING     : 3;
__REG32                 : 5;
__REG32 DIVAHBCLK       : 2;
__REG32                 :18;
} __jmpsettsta_bits;

/* CFC REQ/ACK Connection Configuration (CFCREQACK) 
   SSP1 REQ/ACK Connection Configuration (SSP1REQACK)
   UART1 TX REQ/ACK Connection Configuration (UART1TXREQACK) 
   UART1 RX REQ/ACK Connection Configuration (UART1RXREQACK)
   UART2 TX REQ/ACK Connection Configuration (UART1TXREQACK) 
   UART2 RX REQ/ACK Connection Configuration (UART1RXREQACK)
   SDC REQ/ACK Connection Configuration (SDCREQACK)
   AC97/I2S REQ/ACK Connection Configuration (AC97I2SREQACK)
   IrDA(SIR) TX REQ/ACK Connection Configuration (IrDATXREQACK)
   USB2.0 Device REQ/ACK Connection Configuration (USB2REQACK)
   IrDA (SIR) TX REQ/ACK Connection Configuration (SIRTXREQACK)
   IrDA (FIR) REQ/ACK Connection Configuration (FIRREQACK) 
   External Device0 REQ/ACK Connection Configuration (EXD0REQACK)
   External Device1 REQ/ACK Connection Configuration (EXD1REQACK)*/
typedef struct{
__REG32 CHANNEL         : 3;
__REG32 DMACUSED        : 1;
__REG32                 :28;
} __cfcreqack_bits;

/* AHB Slave n Base/Size Register (AHBSLAVEx) */
typedef struct{
__REG32                 :16;
__REG32 SizeAddr        : 4;
__REG32 BaseAddr        :12;
} __ahbslavex_bits;

/* AHB Priority Control Register (AHBPCR) */
typedef struct{
__REG32                 : 1;
__REG32 PLevel          : 9;
__REG32                 :22;
} __ahbpcr_bits;

/* AHB Transfer Control Register (AHBTCR) */
typedef struct{
__REG32                 : 1;
__REG32 TransCtl        : 1;
__REG32                 :30;
} __ahbtcr_bits;

/* AHB Interrupt Control Register (AHBICR) */
typedef struct{
__REG32 Remap           : 1;
__REG32                 :15;
__REG32 IntsMask        : 1;
__REG32                 : 3;
__REG32 Response        : 2;
__REG32                 : 2;
__REG32 IntrSts         : 1;
__REG32                 : 7;
} __ahbicr_bits;

/* DMAC Interrupt Status Register (INT) */
typedef struct{
__REG32 INT0            : 1;
__REG32 INT1            : 1;
__REG32 INT2            : 1;
__REG32 INT3            : 1;
__REG32                 :28;
} __int_bits;

/* DMAC Interrupt for Terminal Count Status Register (INT_TC) */
typedef struct{
__REG32 INT_TC0         : 1;
__REG32 INT_TC1         : 1;
__REG32 INT_TC2         : 1;
__REG32 INT_TC3         : 1;
__REG32                 :28;
} __int_tc_bits;

/* DMAC Interrupt for Terminal Count Clear Register (INT_TC_CLR) */
typedef struct{
__REG32 INT_TC_CLR0     : 1;
__REG32 INT_TC_CLR1     : 1;
__REG32 INT_TC_CLR2     : 1;
__REG32 INT_TC_CLR3     : 1;
__REG32                 :28;
} __int_tc_clr_bits;

/* DMAC Error/Abort Interrupt Status Register (INT_ERR) */
typedef struct{
__REG32 INT_ERR0        : 1;
__REG32 INT_ERR1        : 1;
__REG32 INT_ERR2        : 1;
__REG32 INT_ERR3        : 1;
__REG32                 :12;
__REG32 INT_ABT0        : 1;
__REG32 INT_ABT1        : 1;
__REG32 INT_ABT2        : 1;
__REG32 INT_ABT3        : 1;
__REG32                 :12;
} __int_err_bits;

/* DMAC Error/Abort Interrupt Status Clear Register (INT_ERR_CLR) */
typedef struct{
__REG32 INT_ERR_CLR0    : 1;
__REG32 INT_ERR_CLR1    : 1;
__REG32 INT_ERR_CLR2    : 1;
__REG32 INT_ERR_CLR3    : 1;
__REG32                 :12;
__REG32 INT_ABT_CLR0    : 1;
__REG32 INT_ABT_CLR1    : 1;
__REG32 INT_ABT_CLR2    : 1;
__REG32 INT_ABT_CLR3    : 1;
__REG32                 :12;
} __int_err_clr_bits;

/* DMAC Terminal Count Status Register (TC) */
typedef struct{
__REG32 TC0             : 1;
__REG32 TC1             : 1;
__REG32 TC2             : 1;
__REG32 TC3             : 1;
__REG32                 :28;
} __tc_bits;

/* DMAC Error/Abort Status Register (ERR) */
typedef struct{
__REG32 ERR0            : 1;
__REG32 ERR1            : 1;
__REG32 ERR2            : 1;
__REG32 ERR3            : 1;
__REG32                 :12;
__REG32 ABT0            : 1;
__REG32 ABT2            : 1;
__REG32 ABT3            : 1;
__REG32 ABT4            : 1;
__REG32                 :12;
} __err_bits;

/* DMAC Error/Abort Status Register (CH_EN) */
typedef struct{
__REG32 CH_EN0          : 1;
__REG32 CH_EN1          : 1;
__REG32 CH_EN2          : 1;
__REG32 CH_EN3          : 1;
__REG32                 :28;
} __ch_en_bits;

/* DMAC Channel Enable Status Register (CH_BUSY) */
typedef struct{
__REG32 CH_BUSY0        : 1;
__REG32 CH_BUSY1        : 1;
__REG32 CH_BUSY2        : 1;
__REG32 CH_BUSY3        : 1;
__REG32                 :28;
} __ch_busy_bits;

/* DMAC Main Configuration Status Register (CSR) */
typedef struct{
__REG32 DMACEN          : 1;
__REG32 M0ENDIAN        : 1;
__REG32 M1ENDIAN        : 1;
__REG32                 :29;
} __csr_bits;

/* DMAC Synchronization Register (SYNC) */
typedef struct{
__REG32 SYNC0           : 1;
__REG32 SYNC1           : 1;
__REG32 SYNC2           : 1;
__REG32 SYNC3           : 1;
__REG32                 :28;
} __sync_bits;

/* DMAC Channel 0 ~ 3 Control Register (Cn_CSR) */
typedef struct{
__REG32 CH_EN           : 1;
__REG32 DST_SEL         : 1;
__REG32 SRC_SEL         : 1;
__REG32 DSTAD_CTL       : 2;
__REG32 SRCAD_CTL       : 2;
__REG32 MODE            : 1;
__REG32 DST_WIDTH       : 3;
__REG32 SRC_WIDTH       : 3;
__REG32                 : 1;
__REG32 ABT             : 1;
__REG32 SRC_SIZE        : 3;
__REG32 PROT1           : 1;
__REG32 PROT2           : 1;
__REG32 PROT3           : 1;
__REG32 CHPRI           : 2;
__REG32                 : 7;
__REG32 TC_MSK          : 1;
} __c_csr_bits;

/* DMAC Channel 0 ~ 3 Configuration Register (Cn_CFG) */
typedef struct{
__REG32 INT_TC_MSK      : 1;
__REG32 INT_ERR_MSK     : 1;
__REG32                 : 6;
__REG32 BUSY            : 1;
__REG32                 :23;
} __c_cfg_bits;

/* DMAC Channel 0 ~ 3 Transfer Size Register (Cn_SIZE) */
typedef struct{
__REG32 TOT_SIZE        :12;
__REG32                 :20;
} __c_size_bits;

/* MAC Interrupt Status Register (MAC_ISR) */
typedef struct{
__REG32 RPKT_FINISH     : 1;
__REG32 NORXBUF         : 1;
__REG32 XPKT_FINISH     : 1;
__REG32 NOTXBUF         : 1;
__REG32 XPKT_OK         : 1;
__REG32 XPKT_LOST       : 1;
__REG32 RPKT_SAV        : 1;
__REG32 RPKT_LOST       : 1;
__REG32 AHB_ERR         : 1;
__REG32 PHYSTS_CHG      : 1;
__REG32                 :22;
} __mac_isr_bits;

/* MAC Interrupt Mask Register (MAC_ISR) */
typedef struct{
__REG32 RPKT_FINISH_M   : 1;
__REG32 NORXBUF_M       : 1;
__REG32 XPKT_FINISH_M   : 1;
__REG32 NOTXBUF_M       : 1;
__REG32 XPKT_OK_M       : 1;
__REG32 XPKT_LOST_M     : 1;
__REG32 RPKT_SAV_M      : 1;
__REG32 RPKT_LOST_M     : 1;
__REG32 AHB_ERR_M       : 1;
__REG32 PHYSTS_CHG_M    : 1;
__REG32                 :22;
} __mac_imr_bits;

/* MAC Most Significant Address Register  (MAC_MADR) */
typedef struct{
__REG32 MAC_MADR        :16;
__REG32                 :16;
} __mac_madr_bits;

/*Interrupt Timer Control Register*/
typedef struct{
__REG32 RXINT_CNT       : 4;
__REG32 RXINT_THR       : 3;
__REG32 RXINT_TIME_SEL  : 1;
__REG32 TXINT_CNT       : 4;
__REG32 TXINT_THR       : 3;
__REG32 TXINT_TIME_SEL  : 1;
__REG32                 :16;
} __mac_itc_bits;

/*Automatic Polling Timer Control Register*/
typedef struct{
__REG32 RXPOLL_CNT      : 4;
__REG32 RXPOLL_TIME_SEL : 1;
__REG32                 : 3;
__REG32 TXPOLL_CNT      : 4;
__REG32 TXPOLL_TIME_SEL : 1;
__REG32                 :19;
} __mac_aptc_bits;

/*DMA Burst Length and Arbitration Control Register*/
typedef struct{
__REG32 INCR4_EN        : 1;
__REG32 INCR8_EN        : 1;
__REG32 INCR16_EN       : 1;
__REG32 RXFIFO_LTHR     : 3;
__REG32 RXFIFO_HTHR     : 3;
__REG32 RX_THR_EN       : 1;
__REG32                 :22;
} __mac_dblac_bits;

/*MAC Control Register*/
typedef struct{
__REG32 XDMA_EN         : 1;
__REG32 RDMA_EN         : 1;
__REG32 SW_RST          : 1;
__REG32 LOOP_EN         : 1;
__REG32 CRC_DIS         : 1;
__REG32 XMT_EN          : 1;
__REG32 ENRX_IN_HALFTX  : 1;
__REG32                 : 1;
__REG32 RCV_EN          : 1;
__REG32 HT_MULTI_EN     : 1;
__REG32 RX_RUNT         : 1;
__REG32 RX_FTL          : 1;
__REG32 RCV_ALL         : 1;
__REG32                 : 1;
__REG32 CRC_APD         : 1;
__REG32 FULLDUP         : 1;
__REG32 RX_MULTIPKT     : 1;
__REG32 RX_BROADPKT     : 1;
__REG32                 :14;
} __mac_cr_bits;

/*MAC Status Register*/
typedef struct{
__REG32 MULTICAST       : 1;
__REG32 BROADCAST       : 1;
__REG32 COL             : 1;
__REG32 RPKT_SAVE       : 1;
__REG32 RPKT_LOST       : 1;
__REG32 CRC_ERR         : 1;
__REG32 FTL             : 1;
__REG32 RUNT            : 1;
__REG32 XPKT_OK         : 1;
__REG32 XPKT_LOST       : 1;
__REG32 LATE_COL        : 1;
__REG32 COL_EXCEED      : 1;
__REG32                 :20;
} __mac_sr_bits;

/*PHY Control Register*/
typedef struct{
__REG32 MIIRDATA        :16;
__REG32 PHYAD           : 5;
__REG32 REGAD           : 5;
__REG32 MIIRD           : 1;
__REG32 MIIWR           : 1;
__REG32                 : 4;
} __mac_phycr_bits;

/*PHY Write Data Register*/
typedef struct{
__REG32 MIIWDATA        :16;
__REG32                 :16;
} __mac_phywdata_bits;

/*Flow Control Register*/
typedef struct{
__REG32 FC_EN           : 1;
__REG32 TX_PAUSE        : 1;
__REG32 FCTHR_EN        : 1;
__REG32 TXPAUSED        : 1;
__REG32 RX_PAUSE        : 1;
__REG32                 : 3;
__REG32 FC_LOW          : 4;
__REG32 FC_HIGH         : 4;
__REG32 PAUSE_TIME      :16;
} __mac_fcr_bits;

/* Back Pressure Register*/
typedef struct{
__REG32 BK_EN           : 1;
__REG32 BK_MODE         : 1;
__REG32                 : 2;
__REG32 BKJAM_LEN       : 4;
__REG32 BK_LOW          : 4;
__REG32                 :20;
} __mac_bpr_bits;

/*Test Seed Register*/
typedef struct{
__REG32 TEST_SEED       :14;
__REG32                 :18;
} __mac_ts_bits;

/*DMA/FIFO State Register*/
typedef struct{
__REG32 RXDMA1_SM       : 4;
__REG32 RXDMA2_SM       : 3;
__REG32                 : 1;
__REG32 TXDMA1_SM       : 4;
__REG32 TXDMA2_SM       : 3;
__REG32                 :11;
__REG32 RXFIFO_EMPTY    : 1;
__REG32 TXFIFO_EMPTY    : 1;
__REG32 DARB_RXGNT      : 1;
__REG32 DARB_TXGNT      : 1;
__REG32 RXD_REQ         : 1;
__REG32 TXD_REQ         : 1;
} __mac_dmafifos_bits;

/*Test Mode Register*/
typedef struct{
__REG32                 : 5;
__REG32 TEST_EXCEL      : 5;
__REG32 TEST_TIME       :10;
__REG32 TEST_MODE       : 1;
__REG32 SEED_SEL        : 1;
__REG32 TEST_SEED_SEL   : 1;
__REG32                 : 1;
__REG32 ITIMER_TEST     : 1;
__REG32 PTIMER_TEST     : 1;
__REG32 SINGLE_PKT      : 1;
__REG32                 : 5;
} __mac_tm_bits;

/*TX_MCOL and TX_SCOL Counter Register*/
typedef struct{
__REG32 TX_MCOL         :16;
__REG32 TX_SCOL         :16;
} __mac_tx_mcol_tx_scol_bits;

/*RPF and AEP Counter Register*/
typedef struct{
__REG32 RPF             :16;
__REG32 AEP             :16;
} __mac_rpf_aep_bits;

/*XM and PG Counter Register*/
typedef struct{
__REG32 XM             :16;
__REG32 PG             :16;
} __mac_xm_pg_bits;

/*RUNT_CNT and TLCC Counter Register*/
typedef struct{
__REG32 RUNT_CNT        :16;
__REG32 TLCC            :16;
} __mac_runt_cnt_tlcc_bits;

/*CRCER_CNT and FTL_CNT Counter Register*/
typedef struct{
__REG32 CRCER_CNT       :16;
__REG32 FTL_CNT         :16;
} __mac_crcer_ftl_cnt_bits;

/*RLC and RCC Counter Register*/
typedef struct{
__REG32 RLC             :16;
__REG32 RCC             :16;
} __mac_rlc_rcc_bits;


/*Memory Bank 0~3 Configuration Register*/
typedef struct{
__REG32 BNK_MBW         : 2;
__REG32                 : 2;
__REG32 BNK_SIZE        : 4;
__REG32 BNK_TYP3        : 1;
__REG32 BNK_TYP2        : 1;
__REG32 BNK_TYP1        : 1;
__REG32 BNK_WPROT       : 1;
__REG32                 : 3;
__REG32 BNK_BASE        :13;
__REG32 BNK_EN          : 1;
__REG32                 : 3;
} __mbxcr_bits;

/*Memory Bank 0 ~ 3 Timing Parameter Register*/
typedef struct{
__REG32 TRNA            : 4;
__REG32 AHT             : 2;
__REG32 WTC             : 2;
__REG32 AT2             : 2;
__REG32                 : 2;
__REG32 AT1             : 4;
__REG32 CTW             : 2;
__REG32 AST             : 2;
__REG32 RBE             : 1;
__REG32                 : 3;
__REG32 EAT1            : 4;
__REG32 ETRNA           : 4;
} __mbxtpr_bits;

/*Shadow Status Register*/
typedef struct{
__REG32 SSR_BNKNUM      : 3;
__REG32                 : 1;
__REG32 SSR_REQM        : 1;
__REG32 SSR_REQ         : 1;
__REG32                 : 2;
__REG32 SSR_STS         : 1;
__REG32                 :23;
} __ssr_bits;

/*SDRAM Timing Parameter 0*/
typedef struct{
__REG32 TCL             : 2;
__REG32                 : 2;
__REG32 TWR             : 2;
__REG32                 : 2;
__REG32 TRF             : 4;
__REG32 TRCD            : 3;
__REG32                 : 1;
__REG32 TRP             : 4;
__REG32 TRAS            : 4;
__REG32                 : 8;
} __stp0_bits;

/*SDRAM Timing Parameter 1*/
typedef struct{
__REG32 REF_INTV        :16;
__REG32 INI_REFT        : 4;
__REG32 INI_PREC        : 4;
__REG32                 : 8;
} __stp1_bits;

/*SDRAM Configuration Register*/
typedef struct{
__REG32 SREF            : 1;
__REG32 PWDN            : 1;
__REG32 ISMR            : 1;
__REG32 IREF            : 1;
__REG32 IPREC           : 1;
__REG32 REFTYPE         : 1;
__REG32                 :26;
} __scr_bits;

/*External Bank Configuration Register*/
typedef struct{
__REG32 BNK_SIZE        : 4;
__REG32 BNK_MBW         : 2;
__REG32                 : 2;
__REG32 BNK_DSZ         : 2;
__REG32                 : 2;
__REG32 BNK_DDW         : 2;
__REG32                 : 2;
__REG32 BNK_BASE        :12;
__REG32 BNK_EN          : 1;
__REG32                 : 3;
} __ebxbsr_bits;

/*Arbiter Control Register*/
typedef struct{
__REG32 ARB_TOC         : 5;
__REG32                 : 3;
__REG32 ARB_TOE         : 1;
__REG32                 :23;
} __atoc_bits;


/*LCD Horizontal Timing Control*/
typedef struct{
__REG32                 : 2;
__REG32 PL              : 6;
__REG32 HW              : 8;
__REG32 HFP             : 8;
__REG32 HBP             : 8;
} __lcdtiming0_bits;

/*LCD Vertical Timing Control*/
typedef struct{
__REG32 LF              :10;
__REG32 VW              : 6;
__REG32 VFP             : 8;
__REG32 VBP             : 8;
} __lcdtiming1_bits;

/*LCD Clock and Signal Polarity Control*/
typedef struct{
__REG32 DivNo           : 6;
__REG32                 : 5;
__REG32 IVS             : 1;
__REG32 IHS             : 1;
__REG32 ICK             : 1;
__REG32 IDE             : 1;
__REG32 ADPEN           : 1;
__REG32                 :16;
} __lcdtiming2_bits;

/*LCD Panel Frame Base Address*/
typedef struct{
__REG32                 : 2;
__REG32 Frame420Size    : 4;
__REG32 LCDFrameBase    :26;
} __lcdframebase_bits;

/*LCD Interrupt Enable Mask*/
typedef struct{
__REG32                 : 1;
__REG32 IntFIFOUdnEn    : 1;
__REG32 IntNxtBaseEn    : 1;
__REG32 IntVstatusEn    : 1;
__REG32 IntBusErrEn     : 1;
__REG32                 :27;
} __lcdintenable_bits;

/*LCD Panel Pixel Parameters*/
typedef struct{
__REG32 LCDen           : 1;
__REG32 BPP             : 3;
__REG32                 : 1;
__REG32 TFT             : 1;
__REG32                 : 2;
__REG32 BGR             : 1;
__REG32 Endian          : 2;
__REG32 LCDon           : 1;
__REG32 Vcomp           : 2;
__REG32                 : 1;
__REG32 PanelType       : 1;
__REG32 FIFOThresh      : 1;
__REG32 EnYCbCr420      : 1;
__REG32 EnYCbCr         : 1;
__REG32                 :13;
} __lcdcontrol_bits;

/*LCD Interrupt Status Clear*/
typedef struct{
__REG32                 : 1;
__REG32 StatusFIFOUdn   : 1;
__REG32 StatusNxtBase   : 1;
__REG32 StatusVstatus   : 1;
__REG32 StatusBusErr    : 1;
__REG32                 :27;
} __lcdintclr_bits;

/*LCD Interrupt Status*/
typedef struct{
__REG32                 : 1;
__REG32 IntFIFOUdn      : 1;
__REG32 IntNxtBase      : 1;
__REG32 IntVstatus      : 1;
__REG32 IntBusErr       : 1;
__REG32                 :27;
} __lcdinterrupt_bits;

/*OSD Scaling and Dimension Control*/
typedef struct{
__REG32 OSDen           : 1;
__REG32 OSDVScal        : 2;
__REG32 OSDHScal        : 2;
__REG32 OSDVdim         : 5;
__REG32 OSDHdim         : 6;
__REG32                 :16;
} __osdcontrol0_bits;

/*OSD Position Control*/
typedef struct{
__REG32 OSDVPos         :10;
__REG32 OSDHdim         :10;
__REG32                 :12;
} __osdcontrol1_bits;

/*OSD Foreground Color Control*/
typedef struct{
__REG32 OSDFrPal0       : 8;
__REG32 OSDFrPal1       : 8;
__REG32 OSDFrPal2       : 8;
__REG32 OSDFrPal3       : 8;
} __osdcontrol2_bits;

/*OSD Background Color Control*/
typedef struct{
__REG32                 : 4;
__REG32 OSDTrans        : 2;
__REG32                 : 2;
__REG32 OSDBgPal1       : 8;
__REG32 OSDBgPal2       : 8;
__REG32 OSDBgPal3       : 8;
} __osdcontrol3_bits;

/*GPI/GPO Control*/
typedef struct{
__REG32 LCDGPI          : 4;
__REG32 LCDGPO          : 4;
__REG32                 :24;
} __gpiocontrol_bits;

/*Main Control Register*/
typedef struct{
__REG8  CAP_RMWKUP      : 1;
__REG8  FLUSH_HBF       : 1;
__REG8  GLINT_EN        : 1;
__REG8  GOSUSP          : 1;
__REG8  SFRST           : 1;
__REG8  CHIP_EN         : 1;
__REG8  HS_EN           : 1;
__REG8  AHB_RST         : 1;
} __main_ctl_bits;

/*Device Address Register*/
typedef struct{
__REG8  DEVADR          : 7;
__REG8  AFT_CONF        : 1;
} __dev_adr_bits;

/*Test Register*/
typedef struct{
__REG8  TST_CLRFF       : 1;
__REG8  TST_LPCX        : 1;
__REG8  TST_CLREA       : 1;
__REG8  TST_DISGENSOF   : 1;
__REG8  TST_DISCRC      : 1;
__REG8  TST_DISTOG      : 1;
__REG8  TST_MOD         : 1;
__REG8  TST_HALF_SPEED  : 1;
} __tst_ep_bits;

/*SOF Frame Number Register Byte 0*/
typedef struct{
__REG8  SOFNL           : 8;
} __frm_numb0_bits;

/*SOF Frame Number Register Byte 1*/
typedef struct{
__REG8  SOFNH           : 3;
__REG8  USOFN           : 3;
__REG8                  : 2;
} __frm_numb1_bits;

/*SOF Mask Timer Register Byte 0*/
typedef struct{
__REG8  SOFTML          : 8;
} __sof_tmskb0_bits;

/*SOF Mask Timer Register Byte 1*/
typedef struct{
__REG8  SOFTMH          : 8;
} __sof_tmskb1_bits;

/*PHY Test Mode Selector Register*/
typedef struct{
__REG8  UNPLUG          : 1;
__REG8  TST_JSTA        : 1;
__REG8  TST_KSTA        : 1;
__REG8  TST_SE0NAK      : 1;
__REG8  TST_PKT         : 1;
__REG8                  : 3;
} __phy_tms_bits;

/*Vendor Specific IO Control Register*/
typedef struct{
__REG8  VCTL            : 4;
__REG8  VCTLOAD_N       : 1;
__REG8                  : 3;
} __vnd_ctl_bits;

/*CX Configuration and Status Register*/
typedef struct{
__REG8  CX_DONE         : 1;
__REG8  TST_PKDONE      : 1;
__REG8  CX_STL          : 1;
__REG8  CX_CLR          : 1;
__REG8  CX_FUL          : 1;
__REG8  CX_EMP          : 1;
__REG8                  : 2;
} __cx_csr_bits;

/*Interrupt Group Mask Register*/
typedef struct{
__REG8  MINT_SCR0       : 1;
__REG8  MINT_SCR1       : 1;
__REG8  MINT_SCR2       : 1;
__REG8  MINT_SCR3       : 1;
__REG8  MINT_SCR4       : 1;
__REG8  MINT_SCR5       : 1;
__REG8  MINT_SCR6       : 1;
__REG8  MINT_SCR7       : 1;
} __int_mgrp_bits;

/*Interrupt Mask Register Byte 0*/
typedef struct{
__REG8  MCX_SETUP_INT   : 1;
__REG8  MCX_IN_INT      : 1;
__REG8  MCX_OUT_INT     : 1;
__REG8  MCX_COMEND_INT  : 1;
__REG8  MCX_COMFAIL_INT : 1;
__REG8  MRBUF_ERR       : 1;
__REG8                  : 1;
__REG8  MR_COM_ABORT    : 1;
} __int_mskb0_bits;

/*Interrupt Mask Register Byte 1*/
typedef struct{
__REG8  MF0_OUT_INT     : 1;
__REG8  MF0_SPK_INT     : 1;
__REG8  MF1_OUT_INT     : 1;
__REG8  MF1_SPK_INT     : 1;
__REG8  MF2_OUT_INT     : 1;
__REG8  MF2_SPK_INT     : 1;
__REG8  MF3_OUT_INT     : 1;
__REG8  MF3_SPK_INT     : 1;
} __int_mskb1_bits;

/*Interrupt Mask Register Byte 2*/
typedef struct{
__REG8  MF4_OUT_INT     : 1;
__REG8  MF4_SPK_INT     : 1;
__REG8  MF5_OUT_INT     : 1;
__REG8  MF5_SPK_INT     : 1;
__REG8  MF6_OUT_INT     : 1;
__REG8  MF6_SPK_INT     : 1;
__REG8  MF7_OUT_INT     : 1;
__REG8  MF7_SPK_INT     : 1;
} __int_mskb2_bits;

/*Interrupt Mask Register Byte 3*/
typedef struct{
__REG8  MF8_OUT_INT     : 1;
__REG8  MF8_SPK_INT     : 1;
__REG8  MF9_OUT_INT     : 1;
__REG8  MF9_SPK_INT     : 1;
__REG8  MF10_OUT_INT    : 1;
__REG8  MF10_SPK_INT    : 1;
__REG8  MF11_OUT_INT    : 1;
__REG8  MF11_SPK_INT    : 1;
} __int_mskb3_bits;

/*Interrupt Mask Register Byte 4*/
typedef struct{
__REG8  MF12_OUT_INT    : 1;
__REG8  MF12_SPK_INT    : 1;
__REG8  MF13_OUT_INT    : 1;
__REG8  MF13_SPK_INT    : 1;
__REG8  MF14_OUT_INT    : 1;
__REG8  MF14_SPK_INT    : 1;
__REG8  MF15_OUT_INT    : 1;
__REG8  MF15_SPK_INT    : 1;
} __int_mskb4_bits;

/*Interrupt Mask Register Byte 5*/
typedef struct{
__REG8  MF0_IN_INT      : 1;
__REG8  MF1_IN_INT      : 1;
__REG8  MF2_IN_INT      : 1;
__REG8  MF3_IN_INT      : 1;
__REG8  MF4_IN_INT      : 1;
__REG8  MF5_IN_INT      : 1;
__REG8  MF6_IN_INT      : 1;
__REG8  MF7_IN_INT      : 1;
} __int_mskb5_bits;

/*Interrupt Mask Register Byte 6*/
typedef struct{
__REG8  MF8_IN_INT      : 1;
__REG8  MF9_IN_INT      : 1;
__REG8  MF10_IN_INT     : 1;
__REG8  MF11_IN_INT     : 1;
__REG8  MF12_IN_INT     : 1;
__REG8  MF13_IN_INT     : 1;
__REG8  MF14_IN_INT     : 1;
__REG8  MF15_IN_INT     : 1;
} __int_mskb6_bits;

/*Interrupt Mask Register Byte 7*/
typedef struct{
__REG8  MHBF_EMPTY_INT      : 1;
__REG8  MUSBRST_INT         : 1;
__REG8  MSUSP_INT           : 1;
__REG8  MRESM_INT           : 1;
__REG8  MISO_SEQ_ERR_INT    : 1;
__REG8  MISO_SEQ_ABORT_INT  : 1;
__REG8  MTX0BYTE_INT        : 1;
__REG8  MRX0BTYE_INT        : 1;
} __int_mskb7_bits;

/*Receive Zero-length Data Packet Register Byte 0*/
typedef struct{
__REG8                      : 1;
__REG8  rx0byte_ep1         : 1;
__REG8  rx0byte_ep2         : 1;
__REG8  rx0byte_ep3         : 1;
__REG8  rx0byte_ep4         : 1;
__REG8  rx0byte_ep5         : 1;
__REG8  rx0byte_ep6         : 1;
__REG8  rx0byte_ep7         : 1;
} __rx0byte_epb0_bits;

/*Receive Zero-length Data Packet Register Byte 1*/
typedef struct{
__REG8  rx0byte_ep8         : 1;
__REG8  rx0byte_ep9         : 1;
__REG8  rx0byte_ep10        : 1;
__REG8  rx0byte_ep11        : 1;
__REG8  rx0byte_ep12        : 1;
__REG8  rx0byte_ep13        : 1;
__REG8  rx0byte_ep14        : 1;
__REG8  rx0byte_ep15        : 1;
} __rx0byte_epb1_bits;

/*FIFO Empty Byte 0*/
typedef struct{
__REG8  fempt_f0         : 1;
__REG8  fempt_f1         : 1;
__REG8  fempt_f2         : 1;
__REG8  fempt_f3         : 1;
__REG8  fempt_f4         : 1;
__REG8  fempt_f5         : 1;
__REG8  fempt_f6         : 1;
__REG8  fempt_f7         : 1;
} __fempt_b0_bits;

/*FIFO Empty Byte 0*/
typedef struct{
__REG8  fempt_f8         : 1;
__REG8  fempt_f9         : 1;
__REG8  fempt_f10        : 1;
__REG8  fempt_f11        : 1;
__REG8  fempt_f12        : 1;
__REG8  fempt_f13        : 1;
__REG8  fempt_f14        : 1;
__REG8  fempt_f15        : 1;
} __fempt_b1_bits;

/*Interrupt Group Register*/
typedef struct{
__REG8  INT_SCR0        : 1;
__REG8  INT_SCR1        : 1;
__REG8  INT_SCR2        : 1;
__REG8  INT_SCR3        : 1;
__REG8  INT_SCR4        : 1;
__REG8  INT_SCR5        : 1;
__REG8  INT_SCR6        : 1;
__REG8  INT_SCR7        : 1;
} __int_grp_bits;

/*Interrupt Source Register Byte 0*/
typedef struct{
__REG8  CX_SETUP_INT    : 1;
__REG8  CX_IN_INT       : 1;
__REG8  CX_OUT_INT      : 1;
__REG8  CX_COMEND_INT   : 1;
__REG8  CX_COMFAIL_INT  : 1;
__REG8  RBUF_ERR        : 1;
__REG8                  : 1;
__REG8  CX_COMABT_INT   : 1;
} __int_srcb0_bits;

/*Interrupt Source Register Byte 1*/
typedef struct{
__REG8  F0_OUT_INT      : 1;
__REG8  F0_SPK_INT      : 1;
__REG8  F1_OUT_INT      : 1;
__REG8  F1_SPK_INT      : 1;
__REG8  F2_OUT_INT      : 1;
__REG8  F2_SPK_INT      : 1;
__REG8  F3_OUT_INT      : 1;
__REG8  F3_SPK_INT      : 1;
} __int_srcb1_bits;

/*Interrupt Source Register Byte 2*/
typedef struct{
__REG8  F4_OUT_INT      : 1;
__REG8  F4_SPK_INT      : 1;
__REG8  F5_OUT_INT      : 1;
__REG8  F5_SPK_INT      : 1;
__REG8  F6_OUT_INT      : 1;
__REG8  F6_SPK_INT      : 1;
__REG8  F7_OUT_INT      : 1;
__REG8  F7_SPK_INT      : 1;
} __int_srcb2_bits;

/*Interrupt Source Register Byte 3*/
typedef struct{
__REG8  F8_OUT_INT      : 1;
__REG8  F8_SPK_INT      : 1;
__REG8  F9_OUT_INT      : 1;
__REG8  F9_SPK_INT      : 1;
__REG8  F10_OUT_INT     : 1;
__REG8  F10_SPK_INT     : 1;
__REG8  F11_OUT_INT     : 1;
__REG8  F11_SPK_INT     : 1;
} __int_srcb3_bits;

/*Interrupt Source Register Byte 4*/
typedef struct{
__REG8  F12_OUT_INT     : 1;
__REG8  F12_SPK_INT     : 1;
__REG8  F13_OUT_INT     : 1;
__REG8  F13_SPK_INT     : 1;
__REG8  F14_OUT_INT     : 1;
__REG8  F14_SPK_INT     : 1;
__REG8  F15_OUT_INT     : 1;
__REG8  F15_SPK_INT     : 1;
} __int_srcb4_bits;

/*Interrupt Source Register Byte 5*/
typedef struct{
__REG8  F0_IN_INT       : 1;
__REG8  F1_IN_INT       : 1;
__REG8  F2_IN_INT       : 1;
__REG8  F3_IN_INT       : 1;
__REG8  F4_IN_INT       : 1;
__REG8  F5_IN_INT       : 1;
__REG8  F6_IN_INT       : 1;
__REG8  F7_IN_INT       : 1;
} __int_srcb5_bits;

/*Interrupt Source Register Byte 6*/
typedef struct{
__REG8  F8_IN_INT       : 1;
__REG8  F9_IN_INT       : 1;
__REG8  F10_IN_INT      : 1;
__REG8  F11_IN_INT      : 1;
__REG8  F12_IN_INT      : 1;
__REG8  F13_IN_INT      : 1;
__REG8  F14_IN_INT      : 1;
__REG8  F15_IN_INT      : 1;
} __int_srcb6_bits;

/*Interrupt Source Register Byte 7*/
typedef struct{
__REG8  HBF_EMPTY_INT       : 1;
__REG8  USBRST_INT          : 1;
__REG8  SUSP_INT            : 1;
__REG8  RESM_INT            : 1;
__REG8  ISO_SEQ_ERR_INT     : 1;
__REG8  ISO_SEQ_ABORT_INT   : 1;
__REG8  TX0BYTE_INT         : 1;
__REG8  RX0BTYE_INT         : 1;
} __int_srcb7_bits;

/*Isochronous Sequential Error Register Byte 0*/
typedef struct{
__REG8                      : 1;
__REG8  iso_seq_err_ep1     : 1;
__REG8  iso_seq_err_ep2     : 1;
__REG8  iso_seq_err_ep3     : 1;
__REG8  iso_seq_err_ep4     : 1;
__REG8  iso_seq_err_ep5     : 1;
__REG8  iso_seq_err_ep6     : 1;
__REG8  iso_seq_err_ep7     : 1;
} __iso_seq_errb0_bits;

/*Isochronous Sequential Error Register Byte 1*/
typedef struct{
__REG8  iso_seq_err_ep8     : 1;
__REG8  iso_seq_err_ep9     : 1;
__REG8  iso_seq_err_ep10    : 1;
__REG8  iso_seq_err_ep11    : 1;
__REG8  iso_seq_err_ep12    : 1;
__REG8  iso_seq_err_ep13    : 1;
__REG8  iso_seq_err_ep14    : 1;
__REG8  iso_seq_err_ep15    : 1;
} __iso_seq_errb1_bits;

/*Isochronous Sequential Abort Register Byte 0*/
typedef struct{
__REG8                      : 1;
__REG8  iso_seq_abt_ep1     : 1;
__REG8  iso_seq_abt_ep2     : 1;
__REG8  iso_seq_abt_ep3     : 1;
__REG8  iso_seq_abt_ep4     : 1;
__REG8  iso_seq_abt_ep5     : 1;
__REG8  iso_seq_abt_ep6     : 1;
__REG8  iso_seq_abt_ep7     : 1;
} __iso_seq_abtb0_bits;

/*Isochronous Sequential Abort Register Byte 1*/
typedef struct{
__REG8  iso_seq_abt_ep8     : 1;
__REG8  iso_seq_abt_ep9     : 1;
__REG8  iso_seq_abt_ep10    : 1;
__REG8  iso_seq_abt_ep11    : 1;
__REG8  iso_seq_abt_ep12    : 1;
__REG8  iso_seq_abt_ep13    : 1;
__REG8  iso_seq_abt_ep14    : 1;
__REG8  iso_seq_abt_ep15    : 1;
} __iso_seq_abtb1_bits;

/*Transferred Zero-length Register Byte 0*/
typedef struct{
__REG8                      : 1;
__REG8  tx0byte_ep1         : 1;
__REG8  tx0byte_ep2         : 1;
__REG8  tx0byte_ep3         : 1;
__REG8  tx0byte_ep4         : 1;
__REG8  tx0byte_ep5         : 1;
__REG8  tx0byte_ep6         : 1;
__REG8  tx0byte_ep7         : 1;
} __tx0byteb0_bits;

/*Transferred Zero-length Register Byte 1*/
typedef struct{
__REG8  tx0byte_ep8         : 1;
__REG8  tx0byte_ep9         : 1;
__REG8  tx0byte_ep10        : 1;
__REG8  tx0byte_ep11        : 1;
__REG8  tx0byte_ep12        : 1;
__REG8  tx0byte_ep13        : 1;
__REG8  tx0byte_ep14        : 1;
__REG8  tx0byte_ep15        : 1;
} __tx0byteb1_bits;

/*Transferred Zero-length Register Byte 1*/
typedef struct{
__REG8  IDLE_CNT            : 3;
__REG8                      : 5;
} __idle_cnt_bits;

/*Endpoint x Map Register*/
typedef struct{
__REG8  FNO_IEP             : 4;
__REG8  FNO_OEP             : 4;
} __epx_map_bits;

/*HBF Data Byte Count*/
typedef struct{
__REG8  HBF_CNT             : 5;
__REG8                      : 3;
} __hbf_cnt_bits;

/*IN Endpoint x MaxPacketSize Register*/
typedef struct{
__REG16  MAXPS_IEP          :11;
__REG16  STL_IEP            : 1;
__REG16  RSTG_IEP           : 1;
__REG16  TX_NUM_HBW_IEP     : 2;
__REG16  TX0BYTE_IEP        : 1;
}__iepx_xpsz_bits;

/*OUT Endpoint x MaxPacketSize Register*/
typedef struct{
__REG16  MAXPS_OEP          :11;
__REG16  STL_OEP            : 1;
__REG16  RSTG_OEP           : 1;
__REG16                     : 3;
} __oepx_xpsz_bits;

/*DMA Mode Enable Register*/
typedef struct{
__REG16  fifo0_dma_en       : 1;
__REG16  fifo1_dma_en       : 1;
__REG16  fifo2_dma_en       : 1;
__REG16  fifo3_dma_en       : 1;
__REG16  fifo4_dma_en       : 1;
__REG16  fifo5_dma_en       : 1;
__REG16  fifo6_dma_en       : 1;
__REG16  fifo7_dma_en       : 1;
__REG16  fifo8_dma_en       : 1;
__REG16  fifo9_dma_en       : 1;
__REG16  fifo10_dma_en      : 1;
__REG16  fifo11_dma_en      : 1;
__REG16  fifo12_dma_en      : 1;
__REG16  fifo13_dma_en      : 1;
__REG16  fifo14_dma_en      : 1;
__REG16  fifo15_dma_en      : 1;
} __fifo_dma_en_bits;

/*FIFOx Map Register*/
typedef struct{
__REG8  EP_FIFO            : 4;
__REG8  Dir_F              : 1;
__REG8                     : 3;
} __fifox_map_bits;

/*FIFOx Configuration Register*/
typedef struct{
__REG8  TYP_F              : 2;
__REG8  BLKNO_F            : 2;
__REG8  BLKSZ_F            : 1;
__REG8                     : 2;
__REG8  EN_F               : 1;
} __fifox_config_bits; 

/*FIFOx Instruction Register*/
typedef struct{
__REG8  BC_FH              : 3;
__REG8  DONE_F             : 1;
__REG8  FFRST              : 1;
__REG8                     : 3;
} __fifox_inst_bits;

/*FIFOx Byte-Count Register*/
typedef struct{
__REG8  BC_FL              : 8;
} __fifox_bc_bits;

/*IRQ Source Register*/
/*IRQ Mask Register*/
/*IRQ Interrupt Clear Register*/
/*IRQ Trig Mode Register*/
/*IRQ Trig Level Register*/
/*IRQ Status Register*/
/*FIQ Source Register*/
/*FIQ Mask Register*/
/*FIQ Interrupt Clear Register*/
/*FIQ Trig Mode Register*/
/*FIQ Trig Level Register*/
/*FIQ Status Register*/
typedef struct{
__REG32  cfc_int_cd_r      : 1;
__REG32  cfc_int_dma_r     : 1;
__REG32  ssp1intr          : 1;
__REG32  isi2cv            : 1;
__REG32                    : 1;
__REG32  sdc_intr          : 1;
__REG32  ssp2intr          : 1;
__REG32  uartintr4         : 1;
__REG32  pmu_fiq           : 1;
__REG32                    : 1;
__REG32  uartintr1         : 1;
__REG32  uartintr2         : 1;
__REG32                    : 1;
__REG32  gpio_intr         : 1;
__REG32  tm2_intr          : 1;
__REG32  tm3_intr          : 1;
__REG32  wd_intr           : 1;
__REG32  rtc_alarm         : 1;
__REG32  rtc_sec           : 1;
__REG32  tm1_intr          : 1;
__REG32                    : 1;
__REG32  dmaint            : 1;
__REG32  irda_int1         : 1;
__REG32  irda_int2         : 1;
__REG32  rshint            : 1;
__REG32  mac_int           : 1;
__REG32  usb_int0          : 1;
__REG32                    : 1;
__REG32  ext_int_irq0      : 1;
__REG32  ext_int_irq1      : 1;
__REG32  ext_int_irq2      : 1;
__REG32  ext_int_irq3      : 1;
} __intsrc_bits;

/*Revision Register*/
typedef struct{
__REG32  intcrevision      :24;
__REG32                    : 8;
} __rrvision_bits;

/*Feature Register for Input Number*/
typedef struct{
__REG32  fiq_number        : 8;
__REG32  irq_number        : 8;
__REG32                    :16;
} __frin_bits;

/*GpioDataOut*/
/*GpioDataIn*/
/*PinDir*/
/*GpioDataSet*/
/*GpioDataClear*/
/*PinPullEnable*/
/*PinPullType*/
/*ntrEnable*/
/*IntrRawState*/
/*IntrMaskedState*/
/*IntrMask*/
/*IntrClear*/
/*IntrTrigger*/
/*IntrBoth*/
/*IntrRiseNeg*/
/*BounceEnable*/
typedef struct{
__REG32  P0                 : 1;
__REG32  P1                 : 1;
__REG32  P2                 : 1;
__REG32  P3                 : 1;
__REG32  P4                 : 1;
__REG32  P5                 : 1;
__REG32  P6                 : 1;
__REG32  P7                 : 1;
__REG32  P8                 : 1;
__REG32  P9                 : 1;
__REG32  P10                : 1;
__REG32  P11                : 1;
__REG32  P12                : 1;
__REG32  P13                : 1;
__REG32  P14                : 1;
__REG32  P15                : 1;
__REG32  P16                : 1;
__REG32  P17                : 1;
__REG32  P18                : 1;
__REG32  P19                : 1;
__REG32  P20                : 1;
__REG32  P21                : 1;
__REG32  P22                : 1;
__REG32  P23                : 1;
__REG32  P24                : 1;
__REG32  P25                : 1;
__REG32  P26                : 1;
__REG32  P27                : 1;
__REG32  P28                : 1;
__REG32  P29                : 1;
__REG32  P30                : 1;
__REG32  P31                : 1;
} __gpio_bits;

/*APB Slave n Base / Size Register*/
typedef struct{
__REG32                     :16;
__REG32  Size               : 4;
__REG32  Base               :10;
__REG32                     : 2;
} __apbslavexbsr_bits;

/*Cycles Register for DMA Channel A/B/C/D*/
typedef struct{
__REG32  Cyc                :24;
__REG32                     : 8;
} __xcyc_bits;

/*Command Register for DMA Channel A/B/C/D*/
typedef struct{
__REG32  EnbDis             : 1;
__REG32  FinIntSts          : 1;
__REG32  FinIntEnb          : 1;
__REG32  BurMod             : 1;
__REG32  ErrIntSts          : 1;
__REG32  ErrIntEnb          : 1;
__REG32  SrcAdrSel          : 1;
__REG32  DesAdrSel          : 1;
__REG32  SrcAdr             : 3;
__REG32                     : 1;
__REG32  DesAdrInc          : 3;
__REG32                     : 1;
__REG32  ReqSel             : 4;
__REG32  DataWidth          : 2;
__REG32                     :10;
} __xcmdr_bits;

/*PWMx CTRL Register*/
typedef struct{
__REG32  Prescale           : 6;
__REG32                     :26;
} __pwmxcr_bits;

/*PWMx DUTY Register*/
typedef struct{
__REG32  DCYCLE             :10;
__REG32  FDCYCLE            : 1;
__REG32                     :21;
} __pwmxdcr_bits;

/*PWMx PERVAL Register*/
typedef struct{
__REG32  PWM_PERVAL         :10;
__REG32                     :22;
} __pwmxpcr_bits;

/*I2C Control Register*/
typedef struct{
__REG32  I2C_RST            : 1;
__REG32  I2C_EN             : 1;
__REG32  SCL_EN             : 1;
__REG32  GC_EN              : 1;
__REG32  START              : 1;
__REG32  STOP               : 1;
__REG32  ACK_NACK           : 1;
__REG32  TB_EN              : 1;
__REG32  DTI_EN             : 1;
__REG32  DRI_EN             : 1;
__REG32  BERRI_EN           : 1;
__REG32  STOPI_EN           : 1;
__REG32  SAMI_EN            : 1;
__REG32  ALI_EN             : 1;
__REG32  STARTI_EN          : 1;
__REG32  SCL_LOW            : 1;
__REG32  SD_LOW             : 1;
__REG32  TEST_BIT           : 1;
__REG32                     :14;
} __i2c_cr_bits;

/*I2C Status Register*/
typedef struct{
__REG32  RW                 : 1;
__REG32  ACK                : 1;
__REG32  I2CB               : 1;
__REG32  BB                 : 1;
__REG32  DT                 : 1;
__REG32  DR                 : 1;
__REG32  BERR               : 1;
__REG32  STOP               : 1;
__REG32  SAM                : 1;
__REG32  GC                 : 1;
__REG32  AL                 : 1;
__REG32  START              : 1;
__REG32                     :20;
} __i2c_sr_bits;

/*I2C Clock Divider Register*/
typedef struct{
__REG32  COUNT              :10;
__REG32                     :22;
} __i2c_cdr_bits;

/*I2C Data Register*/
typedef struct{
__REG32  DR                 : 8;
__REG32                     :24;
} __i2c_dr_bits;

/*I2C Slave Address Register*/
typedef struct{
__REG32  SAR                :10;
__REG32                     :21;
__REG32  EN10               : 1;
} __i2c_sar_bits;

/*I2C Set / Hold Time & Glitch Suppression Setting Register*/
typedef struct{
__REG32  TSR                :10;
__REG32  GSR                : 3;
__REG32                     :19;
} __i2c_tgsr_bits;

/*I2C Bus Monitor Register*/
typedef struct{
__REG32  SDAin              : 1;
__REG32  SCLin              : 1;
__REG32                     :30;
} __i2c_bmr_bits;

/*WdCR*/
typedef struct{
__REG32  WdEnable           : 1;
__REG32  WdRst              : 1;
__REG32  WdIntr             : 1;
__REG32  WdExt              : 1;
__REG32  WdClock            : 1;
__REG32                     :27;
} __wdcr_bits;

/*TmCR*/
typedef struct{
__REG32  Status             : 1;
__REG32                     :31;
} __wdstatus_bits;


/*WdStatus & WdClear*/
typedef struct{
__REG32  Tm1Enable          : 1;
__REG32  Tm1Clock           : 1;
__REG32  Tm1OFEnable        : 1;
__REG32  Tm2Enable          : 1;
__REG32  Tm2Clock           : 1;
__REG32  Tm2OFEnable        : 1;
__REG32  Tm3Enable          : 1;
__REG32  Tm3Clock           : 1;
__REG32  Tm3OFEnable        : 1;
__REG32  Tm1UpDown          : 1;
__REG32  Tm2UpDown          : 1;
__REG32  Tm3UpDown          : 1;
__REG32                     :20;
} __tmcr_bits;

/*TmIntrState & TmIntrMask*/
typedef struct{
__REG32  Tm1Match1          : 1;
__REG32  Tm1Match2          : 1;
__REG32  Tm1Overflow        : 1;
__REG32  Tm2Match1          : 1;
__REG32  Tm2Match2          : 1;
__REG32  Tm2Overflow        : 1;
__REG32  Tm3Match1          : 1;
__REG32  Tm3Match2          : 1;
__REG32  Tm3Overflow        : 1;
__REG32                     :23;
} __tmintrstate_bits;

/*RtcSecond*/
typedef struct{
__REG32  RtcSecond          : 6;
__REG32                     :26;
} __rtcsecond_bits;

/*RtcMinute*/
typedef struct{
__REG32  RtcMinute          : 6;
__REG32                     :26;
} __rtcminute_bits;

/*RtcHour*/
typedef struct{
__REG32  RtcHour            : 5;
__REG32                     :27;
} __rtchour_bits;

/*RtcDay*/
typedef struct{
__REG32  RtcDay             :16;
__REG32                     :16;
} __rtcdays_bits;

/*AlarmSecond*/
typedef struct{
__REG32  AlarmSecond        : 6;
__REG32                     :26;
} __alarmsecond_bits;

/*AlarmMinute*/
typedef struct{
__REG32  AlarmMinute        : 6;
__REG32                     :26;
} __alarmminute_bits;

/*AlarmHour*/
typedef struct{
__REG32  AlarmHour          : 5;
__REG32                     :27;
} __alarmhour_bits;


/**/
typedef struct{
__REG32  RTC_enable         : 1;
__REG32  RTCintpersecond    : 1;
__REG32  RTCintperminute    : 1;
__REG32  RTCintperhour      : 1;
__REG32  RTCintperday       : 1;
__REG32                     :27;
} __rtccr_bits;

/*Command Register*/
typedef struct{
__REG32  CMD_IDX            : 6;
__REG32  NEED_RSP           : 1;
__REG32  LONG_RSP           : 1;
__REG32  APP_CMD            : 1;
__REG32  CMD_EN             : 1;
__REG32  SDC_RST            : 1;
__REG32                     :21;
} __sdcr_bits;

/*Responded Command Register*/
typedef struct{
__REG32  RSP_CMD_IDX        : 6;
__REG32  RSP_CMD_APP        : 1;
__REG32                     :25;
} __sdrcr_bits;

/*Data Control Register*/
typedef struct{
__REG32  BLK_SIZE           : 4;
__REG32  DATA_WRITE         : 1;
__REG32  DMA_EN             : 1;
__REG32  DATA_EN            : 1;
__REG32                     :25;
} __sddcr_bits;

/*Data Length Register*/
typedef struct{
__REG32  DATA_LEN           :16;
__REG32                     :16;
} __sddlr_bits;

/*Status Register*/
typedef struct{
__REG32  RSP_CRC_FAIL       : 1;
__REG32  DATA_CRC_FAIL      : 1;
__REG32  RSP_TIMEOUT        : 1;
__REG32  DATA_TIMEOUT       : 1;
__REG32  RSP_CRC_OK         : 1;
__REG32  DATA_CRC_OK        : 1;
__REG32  CMD_SENT           : 1;
__REG32  DATA_END           : 1;
__REG32  FIFO_URUN          : 1;
__REG32  FIFO_ORUN          : 1;
__REG32  CARD_CHANGE        : 1;
__REG32  CARD_DETECT        : 1;
__REG32  WRITE_PROT         : 1;
__REG32                     :19;
} __sdsr_bits;

/*Clear Register*/
typedef struct{
__REG32  RSP_FAIL       : 1;
__REG32  DATA_FAIL      : 1;
__REG32  RSP_TIMEOUT    : 1;
__REG32  DATA_TIMEOUT   : 1;
__REG32  RSP_OK         : 1;
__REG32  DATA_OK        : 1;
__REG32  CMD_SENT       : 1;
__REG32  DATA_END       : 1;
__REG32  FIFO_U_RUN     : 1;
__REG32  FIFO_O_RUN     : 1;
__REG32  CARD_CHANGE    : 1;
__REG32                 :21;
} __sdclr_bits;

/*Power Control Register*/
typedef struct{
__REG32  SD_POWER       : 4;
__REG32  SD_POWER_ON    : 1;
__REG32                 :27;
} __sdpcr_bits;

/*Clock Control Register*/
typedef struct{
__REG32  CLK_DIV        : 7;
__REG32  CLK_SD         : 1;
__REG32  CLK_DIS        : 1;
__REG32                 :23;
} __sdccr_bits;

/*Bus Width Register*/
typedef struct{
__REG32  SINGLE_BUS       : 1;
__REG32                   : 1;
__REG32  WIDE_BUS         : 1;
__REG32  WIDE_BUS_SUPPORT : 1;
__REG32                   :28;
} __sdbwr_bits;

/*CF Host Status Register*/
typedef struct{
__REG32  RDY_nIREQ          : 1;
__REG32  Card_Detect        : 1;
__REG32                     : 6;
__REG32  Buffer_Active      : 1;
__REG32  Buffer_Data_Ready  : 1;
__REG32  cfc_int_data_cmp_r : 1;
__REG32                     : 1;
__REG32  Buffer_Size        : 4;
__REG32  cfc_int_cd_r       : 1;
__REG32  cfc_io_int_r       : 1;
__REG32                     :14;
} __cfsr_bits;

/*CF Host Control Register*/
typedef struct{
__REG32                         : 4;
__REG32  Float_Control          : 1;
__REG32  Reset                  : 1;
__REG32  _8bit_Mode             : 1;
__REG32                         : 1;
__REG32  DMA_Transfer_Mode      : 1;
__REG32  card_detect_int_mask   : 1;
__REG32  cfc_int_data_cmp_mask  : 1;
__REG32  cfc_io_int_mask        : 1;
__REG32                         :20;
} __cfcr_bits;

/*Access Timing Configuration Register*/
typedef struct{
__REG32  BSA                : 4;
__REG32  BSM                : 4;
__REG32  BSIO               : 4;
__REG32  BSMOW              : 2;
__REG32  BSIORW             : 2;
__REG32                     :16;
} __cfatcr_bits;

/*Active Buffer Control Register*/
typedef struct{
__REG32  ADR                :11;
__REG32                     : 1;
__REG32  TYPE               : 2;
__REG32  INCADR             : 1;
__REG32  RW                 : 1;
__REG32  SIZE               : 4;
__REG32                     :12;
} __cfabcr_bits;

/*Multi Sector Register*/
typedef struct{
__REG32  Enable_Multi_Sector  : 1;
__REG32  Multi_Sector_Timeup  : 7;
__REG32                       :24;
} __cfmsr_bits;

/*Transfer Size Mode2 Enable Register*/
typedef struct{
__REG32  Trans_Size_Mode2_En  : 1;
__REG32                       :31;
} __tsmer_bits;

/*Transfer Size Mode2 Counter Register*/
typedef struct{
__REG32  Trans_Size_Mode2_Cnt :17;
__REG32                       :15;
} __tsmcr_bits;


/* UART Interrupt Enable Register (IER) */
typedef struct {
__REG8 Receiver_Data_Available  : 1;
__REG8 THR_Empty                : 1;
__REG8 Receiver_Line_Status     : 1;
__REG8 MODEM_Status             : 1;
__REG8                          : 4;
} __uartier_bits;

/* UART Interrupt Identification Register (IIR) */
/* UART FIFO Control Register (FCR) */
typedef union {
  /*FFUART_IIR*/
  /*BTUART_IIR*/
  /*STUART_IIR*/
  /*IrDA_IIR*/
  struct {
    __REG8 Interrupt_Pending  : 1;
    __REG8 Interrupt_Id_Code  : 2;
    __REG8 FIFO_mode_only     : 1;
    __REG8 TxFIFO_full        : 1;
    __REG8                    : 1;
    __REG8 FIFO_mode_enable   : 2;
  } ;
  /*FFUART_FCR*/
  /*BTUART_FCR*/
  /*STUART_FCR*/
  /*IrDA_FCR*/
  struct {
    __REG8 FIFO_Enable      : 1;
    __REG8 Rx_FIFO_Reset    : 1;
    __REG8 Tx_FIFO_Reset    : 1;
    __REG8 DMA_Mode         : 1;
    __REG8 TXFIFO_TRGL      : 2;
    __REG8 RXFIFO_TRGL      : 2;
  } ;
  /*FFUART_PSR*/
  /*BTUART_PSR*/
  /*STUART_PSR*/
  /*IrDA_PSR*/
  struct {
    __REG8 PSR              : 5;
    __REG8                  : 3;
  };
} __uartiir_bits;

/* UART Line Control Register (LCR) */
typedef struct {
  __REG8 WL0              : 1;
  __REG8 WL1              : 1;
  __REG8 Stop_Bits        : 1;
  __REG8 Parity_Enable    : 1;
  __REG8 Even_Parity      : 1;
  __REG8 Stick_Parity     : 1;
  __REG8 Set_Break        : 1;
  __REG8 DLAB             : 1;
} __uartlcr_bits;

/* UART Modem Control Register (MCR) */
typedef struct {
  __REG8 Delta_CTS        : 1;
  __REG8 Delta_DSR        : 1;
  __REG8 Trailing_edge_R1 : 1;
  __REG8 Delta_DCD        : 1;
  __REG8 CTS              : 1;
  __REG8 DSR              : 1;
  __REG8 RI               : 1;
  __REG8 DCD              : 1;
} __uartmcr_bits;

/* UART Line Status Register (LSR) */
typedef struct {
  __REG8 Data_Ready         : 1;
  __REG8 Overrun_Error      : 1;
  __REG8 Parity_Error       : 1;
  __REG8 Framing_Error      : 1;
  __REG8 Break_Interrupt    : 1;
  __REG8 THR_Empty          : 1;
  __REG8 Transmitter_Empty  : 1;
  __REG8 FIFO_Data_Error    : 1;
} __uartlsr_bits;

/* UART Modem Status Register (MSR) */
typedef struct {
  __REG8 Data_Ready         : 1;
  __REG8 Overrun_Error      : 1;
  __REG8 Parity_Error       : 1;
  __REG8 Framing_Error      : 1;
  __REG8 Break_Interrupt    : 1;
  __REG8 THR_Empty          : 1;
  __REG8 Transmitter_Empty  : 1;
  __REG8 FIFO_Data_Error    : 1;
} __uartmsr_bits;

/*Mode Definition Register*/
typedef struct{
  __REG8 MODE_SEL           : 2;
  __REG8 SIP_BYCPU          : 1;
  __REG8 FMEND_MD           : 1;
  __REG8 DMA_EN             : 1;
  __REG8 FIR_INV_RX         : 1;
  __REG8 IR_INV_TX          : 1;
  __REG8                    : 1;
} __mdr_bits;

/*Auxiliary Control Register*/
typedef struct{
  __REG8 TX_ENABLE          : 1;
  __REG8 RX_ENABLE          : 1;
  __REG8 SET_EOT           : 1;
  __REG8 FORCE_ABORT        : 1;
  __REG8 SEND_SIP           : 1;
  __REG8 STFF_TRGL          : 2;
  __REG8 SIR_PW             : 1;
} __acr_bits;

/*Transmit Frame-Length Register Low*/
typedef struct{
__REG8 TXLENL               : 8;
} __txlenl_bits;

/*Transmit Frame-Length Register High*/
typedef struct{
__REG8  TXLENH              : 5;
__REG8                      : 3;
} __txlenh_bits;

/*Maximum Receiver Frame-Length Low*/
typedef struct{
__REG8  MRXLENL             : 8;
} __mrxlenl_bits;

/*Maximum Receiver Frame-Length High*/
typedef struct{
__REG8  MRXLENH             : 5;
__REG8                      : 3;
} __mrxlenh_bits;

/*Preamble Length Register*/
typedef struct{
__REG8  FPL                 : 2;
__REG8                      : 6;
} __plr_bits;

/*FIR Mode Interrupt Identification Register*/
typedef union{
  /*FMIIR_PIO*/
  struct{
  __REG8  RXFIFO_TRIG       : 1;
  __REG8  TXFIFO_TRIG       : 1;
  __REG8  RXFIFO_ORUN       : 1;
  __REG8  TXFIFO_URUN       : 1;
  __REG8  EOF_DECTED        : 1;
  __REG8  FRM_SENT          : 1;
  __REG8                    : 2;
  };
  /*FMIIR_DMA*/
  struct{
  __REG8  STFIFO_TRIG       : 1;
  __REG8  STFIFO_TIME_OUT   : 1;
  __REG8  STFIFO_ORUN       : 1;
  __REG8  _TXFIFO_URUN      : 1;
  __REG8  _RXFIFO_ORUN      : 1;
  __REG8  _FRM_SENT         : 1;
  __REG8                    : 2;
  };
} __fmiir_pio_dma_bits;

/**/
typedef union{
  /*FMIIER_PIO*/
  struct{
  __REG8  RXFIFO_TRIG       : 1;
  __REG8  TXFIFO_TRIG       : 1;
  __REG8  RXFIFO_ORUN       : 1;
  __REG8  TXFIFO_URUN       : 1;
  __REG8  EOF_DECTED        : 1;
  __REG8  FRM_SENT          : 1;
  __REG8                    : 2;
  };
  /*FMIIER_DMA*/
  struct{
  __REG8  STFIFO_TRIG       : 1;
  __REG8  STFIFO_TIME_OUT   : 1;
  __REG8  STFIFO_ORUN       : 1;
  __REG8  _TXFIFO_URUN      : 1;
  __REG8                    : 1;
  __REG8  _FRM_SENT         : 1;
  __REG8                    : 2;
  };
} __fmier_pio_dma_bits;

/*Status FIFO Line Status Register*/
typedef struct{
  __REG8  RXFIFO_ORUN       : 1;
  __REG8  CRC_ERR           : 1;
  __REG8  PHY_ERR           : 1;
  __REG8  SIZE_ERR          : 1;
  __REG8  STS_VLD           : 1;
  __REG8                    : 3;
} __stff_sts_bits;

/*Status FIFO Received Frame Length Register - Low*/
typedef struct{
  __REG8  RCVLENL           : 8;
} __stff_rxlenl_bits;

/*Status FIFO Received Frame Length Register - High*/
typedef struct{
__REG8  RCVLENH             : 5;
__REG8                      : 3;
} __stff_rxlenh_bits;

/*FIR Mode Link Status Register*/
typedef struct{
  __REG8  RXFIFO_EMPTY      : 1;
  __REG8  STFIFO_EMPTY      : 1;
  __REG8  CRC_ERR           : 1;
  __REG8  PHY_ERR           : 1;
  __REG8  SIZE_ERR          : 1;
  __REG8  STFIFO_FULL       : 1;
  __REG8  TXFIFO_EMPTY      : 1;
  __REG8  FIR_IDLE          : 1;
} __fmlsr_bits;

/*Rx FIFO Count Register*/
typedef struct{
  __REG8  RXFF_CNTR         : 5;
  __REG8                    : 3;
} __rxff_cntr_bits;

/*Last Frame Length Register Low*/
typedef struct{
  __REG8  LSTFMLENL         : 8;
} __lstfmlenl_bits;

/*Status FIFO Received Frame Length Register - High*/
typedef struct{
__REG8  LSTFMLENH           : 5;
__REG8  FRM_NUM             : 3;
} __lstfmlenh_bits;

/*SSP Control Register 0*/
/*I2S_AC97 Control Register 0*/
typedef struct{
__REG32  SCLKPH             : 1;
__REG32  SCLKPO             : 1;
__REG32  OPM                : 2;
__REG32  FSJSTFY            : 1;
__REG32  FSPO               : 1;
__REG32  LSB                : 1;
__REG32  LBM                : 1;
__REG32  FSDIST             : 2;
__REG32                     : 2;
__REG32  FFMT               : 3;
__REG32                     :17;
} __sspcr0_bits;

/*SSP Control Register 1*/
/*I2S_AC97 Control Register 1*/
typedef struct{
__REG32  SCLKDIV            :16;
__REG32  SDL                : 5;
__REG32                     : 3;
__REG32  PDL                : 8;
} __sspcr1_bits;

/*SSP Control Register 2 */
/*I2S_AC97 Control Register 2 */
typedef struct{
__REG32  SSPEN              : 1;
__REG32  TXDOE              : 1;
__REG32  RXFCLR             : 1;
__REG32  TXFCLR             : 1;
__REG32  ACWRST             : 1;
__REG32  ACCRST             : 1;
__REG32  SSPRST             : 1;
__REG32                     :25;
} __sspcr2_bits;

/*SSP Status Register*/
/*I2S_AC97 Status Register*/
typedef struct{
__REG32  RFF                : 1;
__REG32  TFNF               : 1;
__REG32  BUSY               : 1;
__REG32                     : 1;
__REG32  RFVE               : 5;
__REG32                     : 3;
__REG32  TFVE               : 5;
__REG32                     :15;
} __sspsr_bits;

/*SSP Interrupt Control Register*/
/*I2S_AC97 Interrupt Control Register*/
typedef struct{
__REG32  RFORIEN            : 1;
__REG32  TFURIEN            : 1;
__REG32  RFTHIEN            : 1;
__REG32  TFTHIEN            : 1;
__REG32  RFDMAEN            : 1;
__REG32  TFDMAEN            : 1;
__REG32  AC97FCEN           : 1;
__REG32                     : 1;
__REG32  RFTHOD             : 4;
__REG32  TFTHOD             : 4;
__REG32                     :16;
} __sspicr_bits;

/*SSP Interrupt Status Register*/
/*I2S_AC97 Interrupt Status Register*/
typedef struct{
__REG32  RFORI              : 1;
__REG32  TFURI              : 1;
__REG32  RFTHI              : 1;
__REG32  TFTHI              : 1;
__REG32  AC97FCI            : 1;
__REG32                     :27;
} __sspisr_bits;

/*SSP AC-Link Slot Valid Register*/
/*I2S_AC97 AC-Link Slot Valid Register*/
typedef struct{
__REG32  CODECID            : 2;
__REG32                     : 1;
__REG32  SLOT12V            : 1;
__REG32  SLOT11V            : 1;
__REG32  SLOT10V            : 1;
__REG32  SLOT9V             : 1;
__REG32  SLOT8V             : 1;
__REG32  SLOT7V             : 1;
__REG32  SLOT6V             : 1;
__REG32  SLOT5V             : 1;
__REG32  SLOT4V             : 1;
__REG32  SLOT3V             : 1;
__REG32  SLOT2V             : 1;
__REG32  SLOT1V             : 1;
__REG32                     :17;
} __sspvr_bits;




#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/

/***************************************************************************
 **
 ** PMU
 **
 ***************************************************************************/
__IO_REG32_BIT(IDNMBR0,         0x98100000,__READ       ,__idnmbr0_bits);
__IO_REG32_BIT(OSCC,            0x98100008,__READ_WRITE ,__oscc_bits);
__IO_REG32_BIT(PMODE,           0x9810000C,__READ_WRITE ,__pmode_bits);
__IO_REG32_BIT(PMCR,            0x98100010,__READ_WRITE ,__pmcr_bits);
__IO_REG32_BIT(PED,             0x98100014,__READ_WRITE ,__ped_bits);
__IO_REG32_BIT(PEDSR,           0x98100018,__READ_WRITE ,__pedsr_bits);
__IO_REG32_BIT(PMSR,            0x98100020,__READ_WRITE ,__pmsr_bits);
__IO_REG32_BIT(PGSR,            0x98100024,__READ_WRITE ,__pgsr_bits);
__IO_REG32_BIT(MFPSR,           0x98100028,__READ_WRITE ,__mfpsr_bits);
__IO_REG32_BIT(MISC,            0x9810002C,__READ_WRITE ,__misc_bits);
__IO_REG32_BIT(PDLLCR0,         0x98100030,__READ_WRITE ,__pdllcr0_bits);
__IO_REG32_BIT(PDLLCR1,         0x98100034,__READ_WRITE ,__pdllcr1_bits);
__IO_REG32_BIT(AHBMCLKOFF,      0x98100038,__READ_WRITE ,__ahbmclkoff_bits);
__IO_REG32_BIT(APBMCLKOFF,      0x9810003C,__READ_WRITE ,__apbmclkoff_bits);
__IO_REG32_BIT(DCSRCR0,         0x98100040,__READ_WRITE ,__dcsrcr0_bits);
__IO_REG32_BIT(DCSRCR1,         0x98100044,__READ_WRITE ,__dcsrcr1_bits);
__IO_REG32_BIT(DCSRCR2,         0x98100048,__READ_WRITE ,__dcsrcr2_bits);
__IO_REG32_BIT(SDRAMHTC,        0x9810004C,__READ_WRITE ,__sdramhtc_bits);
__IO_REG32(    PSPR0,           0x98100050,__READ_WRITE );
__IO_REG32(    PSPR1,           0x98100054,__READ_WRITE );
__IO_REG32(    PSPR2,           0x98100058,__READ_WRITE );
__IO_REG32(    PSPR3,           0x9810005C,__READ_WRITE );
__IO_REG32(    PSPR4,           0x98100060,__READ_WRITE );
__IO_REG32(    PSPR5,           0x98100064,__READ_WRITE );
__IO_REG32(    PSPR6,           0x98100068,__READ_WRITE );
__IO_REG32(    PSPR7,           0x9810006C,__READ_WRITE );
__IO_REG32(    PSPR8,           0x98100070,__READ_WRITE );
__IO_REG32(    PSPR9,           0x98100074,__READ_WRITE );
__IO_REG32(    PSPR10,          0x98100078,__READ_WRITE );
__IO_REG32(    PSPR11,          0x9810007C,__READ_WRITE );
__IO_REG32(    PSPR12,          0x98100080,__READ_WRITE );
__IO_REG32(    PSPR13,          0x98100084,__READ_WRITE );
__IO_REG32(    PSPR14,          0x98100088,__READ_WRITE );
__IO_REG32(    PSPR15,          0x9810008C,__READ_WRITE );
__IO_REG32_BIT(AHBDMAREQACK,    0x98100090,__READ       ,__ahbdmareqack_bits);
__IO_REG32_BIT(JMPSETTSTA,      0x9810009C,__READ       ,__jmpsettsta_bits);
__IO_REG32_BIT(CFCREQACK,       0x981000A0,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(SSP1REQACK,      0x981000A4,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(UART1TXREQACK,   0x981000A8,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(UART1RXREQACK,   0x981000AC,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(UART2TXREQACK,   0x981000B0,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(UART2RXREQACK,   0x981000B4,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(SDCREQACK,       0x981000B8,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(AC97I2SREQACK,   0x981000BC,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(IrDATXREQACK,    0x981000C0,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(USB2REQACK,      0x981000C8,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(SIRTXREQACK,     0x981000CC,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(FIRREQACK,       0x981000D0,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(EXD0REQACK,      0x981000D4,__READ_WRITE ,__cfcreqack_bits);
__IO_REG32_BIT(EXD1REQACK,      0x981000D8,__READ_WRITE ,__cfcreqack_bits);

/***************************************************************************
 **
 ** AHB Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(AHBSLAVE0,       0x90100000,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE1,       0x90100004,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE2,       0x90100008,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE3,       0x9010000C,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE4,       0x90100010,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE5,       0x90100014,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE6,       0x90100018,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE7,       0x9010001C,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE9,       0x90100024,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE12,      0x90100030,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE13,      0x90100034,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE14,      0x90100038,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE15,      0x9010003C,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE17,      0x90100044,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE18,      0x90100048,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE19,      0x9010004C,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE21,      0x90100054,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBSLAVE22,      0x90100058,__READ_WRITE ,__ahbslavex_bits);
__IO_REG32_BIT(AHBPCR,          0x90100080,__READ_WRITE ,__ahbpcr_bits);
__IO_REG32_BIT(AHBTCR,          0x90100084,__READ_WRITE ,__ahbtcr_bits);
__IO_REG32_BIT(AHBICR,          0x90100088,__READ_WRITE ,__ahbicr_bits);

/***************************************************************************
 **
 ** DMA Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(INT,             0x90400000,__READ       ,__int_bits);
__IO_REG32_BIT(INT_TC,          0x90400004,__READ       ,__int_tc_bits);
__IO_REG32_BIT(INT_TC_CLR,      0x90400008,__WRITE      ,__int_tc_clr_bits);
__IO_REG32_BIT(INT_ERR,         0x9040000C,__READ       ,__int_err_bits);
__IO_REG32_BIT(INT_ERR_CLR,     0x90400010,__WRITE      ,__int_err_clr_bits);
__IO_REG32_BIT(TC,              0x90400014,__READ       ,__tc_bits);
__IO_REG32_BIT(ERR,             0x90400018,__READ       ,__err_bits);
__IO_REG32_BIT(CH_EN,           0x9040001C,__READ       ,__ch_en_bits);
__IO_REG32_BIT(CH_BUSY,         0x90400020,__READ       ,__ch_busy_bits);
__IO_REG32_BIT(CSR,             0x90400024,__READ_WRITE ,__csr_bits);
__IO_REG32_BIT(SYNC,            0x90400028,__READ_WRITE ,__sync_bits);
__IO_REG32_BIT(C0_CSR,          0x90400100,__READ_WRITE ,__c_csr_bits);
__IO_REG32_BIT(C0_CFG,          0x90400104,__READ_WRITE ,__c_cfg_bits);
__IO_REG32(    C0_SrcAddr,      0x90400108,__READ_WRITE );
__IO_REG32(    C0_DstAddr,      0x9040010C,__READ_WRITE );
__IO_REG32(    C0_LLP,          0x90400110,__READ_WRITE );
__IO_REG32_BIT(C0_SIZE,         0x90400114,__READ_WRITE ,__c_size_bits);
__IO_REG32_BIT(C1_CSR,          0x90400120,__READ_WRITE ,__c_csr_bits);
__IO_REG32_BIT(C1_CFG,          0x90400124,__READ_WRITE ,__c_cfg_bits);
__IO_REG32(    C1_SrcAddr,      0x90400128,__READ_WRITE );
__IO_REG32(    C1_DstAddr,      0x9040012C,__READ_WRITE );
__IO_REG32(    C1_LLP,          0x90400130,__READ_WRITE );
__IO_REG32_BIT(C1_SIZE,         0x90400134,__READ_WRITE ,__c_size_bits);
__IO_REG32_BIT(C2_CSR,          0x90400140,__READ_WRITE ,__c_csr_bits);
__IO_REG32_BIT(C2_CFG,          0x90400144,__READ_WRITE ,__c_cfg_bits);
__IO_REG32(    C2_SrcAddr,      0x90400148,__READ_WRITE );
__IO_REG32(    C2_DstAddr,      0x9040014C,__READ_WRITE );
__IO_REG32(    C2_LLP,          0x90400150,__READ_WRITE );
__IO_REG32_BIT(C2_SIZE,         0x90400154,__READ_WRITE ,__c_size_bits);
__IO_REG32_BIT(C3_CSR,          0x90400160,__READ_WRITE ,__c_csr_bits);
__IO_REG32_BIT(C3_CFG,          0x90400164,__READ_WRITE ,__c_cfg_bits);
__IO_REG32(    C3_SrcAddr,      0x90400168,__READ_WRITE );
__IO_REG32(    C3_DstAddr,      0x9040016C,__READ_WRITE );
__IO_REG32(    C3_LLP,          0x90400170,__READ_WRITE );
__IO_REG32_BIT(C3_SIZE,         0x90400174,__READ_WRITE ,__c_size_bits);

/***************************************************************************
 **
 ** MAC
 **
 ***************************************************************************/
__IO_REG32_BIT(MAC_ISR,             0x90900000,__READ       ,__mac_isr_bits);
__IO_REG32_BIT(MAC_IMR,             0x90900004,__READ_WRITE ,__mac_imr_bits);
__IO_REG32_BIT(MAC_MADR,            0x90900008,__READ_WRITE ,__mac_madr_bits);
__IO_REG32(    MAC_LADR,            0x9090000C,__READ_WRITE );
__IO_REG32(    MAC_MAHT0,           0x90900010,__READ_WRITE );
__IO_REG32(    MAC_MAHT1,           0x90900014,__READ_WRITE );
__IO_REG32(    MAC_TXPD,            0x90900018,__WRITE      );
__IO_REG32(    MAC_RXPD,            0x9090001C,__WRITE      );
__IO_REG32(    MAC_TXR_BADR,        0x90900020,__READ_WRITE );
__IO_REG32(    MAC_RXR_BADR,        0x90900024,__READ_WRITE );
__IO_REG32_BIT(MAC_ITC,             0x90900028,__READ_WRITE ,__mac_itc_bits);
__IO_REG32_BIT(MAC_APTC,            0x9090002C,__READ_WRITE ,__mac_aptc_bits);
__IO_REG32_BIT(MAC_DBLAC,           0x90900030,__READ_WRITE ,__mac_dblac_bits);
__IO_REG32_BIT(MAC_CR,              0x90900088,__READ_WRITE ,__mac_cr_bits);
__IO_REG32_BIT(MAC_SR,              0x9090008C,__READ       ,__mac_sr_bits);
__IO_REG32_BIT(MAC_PHYCR,           0x90900090,__READ_WRITE ,__mac_phycr_bits);
__IO_REG32_BIT(MAC_PHYWDATA,        0x90900094,__READ_WRITE ,__mac_phywdata_bits);
__IO_REG32_BIT(MAC_FCR,             0x90900098,__READ_WRITE ,__mac_fcr_bits);
__IO_REG32_BIT(MAC_BPR,             0x9090009C,__READ_WRITE ,__mac_bpr_bits);
__IO_REG32_BIT(MAC_TS,              0x909000C4,__READ_WRITE ,__mac_ts_bits);
__IO_REG32_BIT(MAC_DMAFIFOS,        0x909000C8,__READ       ,__mac_dmafifos_bits);
__IO_REG32_BIT(MAC_TM,              0x909000CC,__READ_WRITE ,__mac_tm_bits);
__IO_REG32_BIT(MAC_TX_MCOL_TX_SCOL, 0x909000D4,__READ       ,__mac_tx_mcol_tx_scol_bits);
__IO_REG32_BIT(MAC_RPF_AEP,         0x909000D8,__READ       ,__mac_rpf_aep_bits);
__IO_REG32_BIT(MAC_XM_PG,           0x909000DC,__READ       ,__mac_xm_pg_bits);
__IO_REG32_BIT(MAC_RUNT_CNT_TLCC,   0x909000E0,__READ       ,__mac_runt_cnt_tlcc_bits);
__IO_REG32_BIT(MAC_CRCER_FTL_CNT,   0x909000E4,__READ       ,__mac_crcer_ftl_cnt_bits);
__IO_REG32_BIT(MAC_RLC_RCC,         0x909000E8,__READ       ,__mac_rlc_rcc_bits);
__IO_REG32(    MAC_BROC,            0x909000EC,__READ       );
__IO_REG32(    MAC_MULCA,           0x909000F0,__READ       );
__IO_REG32(    MAC_RP,              0x909000F4,__READ       );
__IO_REG32(    MAC_XP,              0x909000F8,__READ       );

/***************************************************************************
 **
 ** SMC
 **
 ***************************************************************************/
__IO_REG32_BIT(MB0CR,           0x90200000,__READ_WRITE ,__mbxcr_bits);
__IO_REG32_BIT(MB0TPR,          0x90200004,__READ_WRITE ,__mbxtpr_bits);
__IO_REG32_BIT(MB1CR,           0x90200008,__READ_WRITE ,__mbxcr_bits);
__IO_REG32_BIT(MB1TPR,          0x9020000C,__READ_WRITE ,__mbxtpr_bits);
__IO_REG32_BIT(MB2CR,           0x90200010,__READ_WRITE ,__mbxcr_bits);
__IO_REG32_BIT(MB2TPR,          0x90200014,__READ_WRITE ,__mbxtpr_bits);
__IO_REG32_BIT(MB3CR,           0x90200018,__READ_WRITE ,__mbxcr_bits);
__IO_REG32_BIT(MB3TPR,          0x9020001C,__READ_WRITE ,__mbxtpr_bits);
__IO_REG32_BIT(SSR,             0x90200040,__READ_WRITE ,__ssr_bits);
  
/***************************************************************************
 **
 ** SDRAMC
 **
 ***************************************************************************/
__IO_REG32_BIT(STP0,            0x90300000,__READ_WRITE ,__stp0_bits);
__IO_REG32_BIT(STP1,            0x90300004,__READ_WRITE ,__stp1_bits);
__IO_REG32_BIT(SCR,             0x90300008,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(EB0BSR,          0x9030000C,__READ_WRITE ,__ebxbsr_bits);
__IO_REG32_BIT(EB1BSR,          0x90300010,__READ_WRITE ,__ebxbsr_bits);
__IO_REG32_BIT(EB2BSR,          0x90300014,__READ_WRITE ,__ebxbsr_bits);
__IO_REG32_BIT(EB3BSR,          0x90300018,__READ_WRITE ,__ebxbsr_bits);
__IO_REG32_BIT(ATOC,            0x90300034,__READ_WRITE ,__atoc_bits);

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCDTiming0,                0x90600000, __READ_WRITE , __lcdtiming0_bits  );
__IO_REG32_BIT(LCDTiming1,                0x90600004, __READ_WRITE , __lcdtiming1_bits  );
__IO_REG32_BIT(LCDTiming2,                0x90600008, __READ_WRITE , __lcdtiming2_bits  );
__IO_REG32_BIT(LCDFrameBase,              0x90600010, __READ_WRITE , __lcdframebase_bits);
__IO_REG32_BIT(LCDIntEnable,              0x90600018, __READ_WRITE , __lcdintenable_bits);
__IO_REG32_BIT(LCDControl,                0x9060001C, __READ_WRITE , __lcdcontrol_bits  );
__IO_REG32_BIT(LCDIntClr,                 0x90600020, __WRITE      , __lcdintclr_bits   );
__IO_REG32_BIT(LCDInterrupt,              0x90600024, __READ       , __lcdinterrupt_bits);
__IO_REG32_BIT(OSDControl0,               0x90600034, __READ_WRITE , __osdcontrol0_bits );
__IO_REG32_BIT(OSDControl1,               0x90600038, __READ_WRITE , __osdcontrol1_bits );
__IO_REG32_BIT(OSDControl2,               0x9060003C, __READ_WRITE , __osdcontrol2_bits );
__IO_REG32_BIT(OSDControl3,               0x90600040, __READ_WRITE , __osdcontrol3_bits );
__IO_REG32_BIT(GPIOControl,               0x90600044, __READ_WRITE , __gpiocontrol_bits );
__IO_REG32(    PaletteWritePortBase,      0x90600200, __WRITE );
__IO_REG32(    OSDFontWritePortBase,      0x90608000, __WRITE );
__IO_REG32(    OSDAttributeWritePortBase, 0x9060C000, __WRITE );

/***************************************************************************
 **
 ** USB 2.0
 **
 ***************************************************************************/
__IO_REG8_BIT(main_ctl,       0x90B00000, __READ_WRITE , __main_ctl_bits);
__IO_REG8_BIT(dev_adr,        0x90B00001, __READ_WRITE , __dev_adr_bits);
__IO_REG8_BIT(tst_ep,         0x90B00002, __READ_WRITE , __tst_ep_bits);
__IO_REG8_BIT(frm_numb0,      0x90B00004, __READ_WRITE , __frm_numb0_bits);
__IO_REG8_BIT(frm_numb1,      0x90B00005, __READ_WRITE , __frm_numb1_bits);
__IO_REG8_BIT(sof_tmskb0,     0x90B00006, __READ_WRITE , __sof_tmskb0_bits);
__IO_REG8_BIT(sof_tmskb1,     0x90B00007, __READ_WRITE , __sof_tmskb1_bits);
__IO_REG8_BIT(phy_tms,        0x90B00008, __READ_WRITE , __phy_tms_bits);
__IO_REG8_BIT(vnd_ctl,        0x90B00009, __READ_WRITE , __vnd_ctl_bits);
__IO_REG8(    vnd_sta,        0x90B0000A, __READ       );
__IO_REG8_BIT(cx_csr,         0x90B0000B, __READ_WRITE , __cx_csr_bits);
__IO_REG32(   ep0_dp,         0x90B0000C, __READ_WRITE );
__IO_REG8_BIT(int_mgrp,       0x90B00010, __READ_WRITE , __int_mgrp_bits);
__IO_REG8_BIT(int_mskb0,      0x90B00011, __READ_WRITE , __int_mskb0_bits);
__IO_REG8_BIT(int_mskb1,      0x90B00012, __READ_WRITE , __int_mskb1_bits);
__IO_REG8_BIT(int_mskb2,      0x90B00013, __READ_WRITE , __int_mskb2_bits);
__IO_REG8_BIT(int_mskb3,      0x90B00014, __READ_WRITE , __int_mskb3_bits);
__IO_REG8_BIT(int_mskb4,      0x90B00015, __READ_WRITE , __int_mskb4_bits);
__IO_REG8_BIT(int_mskb5,      0x90B00016, __READ_WRITE , __int_mskb5_bits);
__IO_REG8_BIT(int_mskb6,      0x90B00017, __READ_WRITE , __int_mskb6_bits);
__IO_REG8_BIT(int_mskb7,      0x90B00018, __READ_WRITE , __int_mskb7_bits);
__IO_REG8_BIT(rx0byte_epb0,   0x90B00019, __READ_WRITE , __rx0byte_epb0_bits);
__IO_REG8_BIT(rx0byte_epb1,   0x90B0001A, __READ_WRITE , __rx0byte_epb1_bits);
__IO_REG8_BIT(fempt_b0,       0x90B0001C, __READ_WRITE , __fempt_b0_bits);
__IO_REG8_BIT(fempt_b1,       0x90B0001D, __READ_WRITE , __fempt_b1_bits);
__IO_REG8_BIT(int_grp,        0x90B00020, __READ       , __int_grp_bits);
__IO_REG8_BIT(int_srcb0,      0x90B00021, __READ       , __int_srcb0_bits);
__IO_REG8_BIT(int_srcb1,      0x90B00022, __READ       , __int_srcb1_bits);
__IO_REG8_BIT(int_srcb2,      0x90B00023, __READ       , __int_srcb2_bits);
__IO_REG8_BIT(int_srcb3,      0x90B00024, __READ       , __int_srcb3_bits);
__IO_REG8_BIT(int_srcb4,      0x90B00025, __READ       , __int_srcb4_bits);
__IO_REG8_BIT(int_srcb5,      0x90B00026, __READ       , __int_srcb5_bits);
__IO_REG8_BIT(int_srcb6,      0x90B00027, __READ       , __int_srcb6_bits);
__IO_REG8_BIT(int_srcb7,      0x90B00028, __READ       , __int_srcb7_bits);
__IO_REG8_BIT(iso_seq_errb0,  0x90B00029, __READ_WRITE , __iso_seq_errb0_bits);
__IO_REG8_BIT(iso_seq_errb1,  0x90B0002A, __READ_WRITE , __iso_seq_errb1_bits);
__IO_REG8_BIT(iso_seq_abtb0,  0x90B0002B, __READ_WRITE , __iso_seq_abtb0_bits);
__IO_REG8_BIT(iso_seq_abtb1,  0x90B0002C, __READ_WRITE , __iso_seq_abtb1_bits);
__IO_REG8_BIT(tx0byteb0,      0x90B0002D, __READ_WRITE , __tx0byteb0_bits);
__IO_REG8_BIT(tx0byteb1,      0x90B0002E, __READ_WRITE , __tx0byteb1_bits);
__IO_REG8_BIT(idle_cnt,       0x90B0002F, __READ_WRITE , __idle_cnt_bits);
__IO_REG8_BIT(ep1_map,        0x90B00030, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep2_map,        0x90B00031, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep3_map,        0x90B00032, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep4_map,        0x90B00033, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep5_map,        0x90B00034, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep6_map,        0x90B00035, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep7_map,        0x90B00036, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep8_map,        0x90B00037, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep9_map,        0x90B00038, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep10_map ,      0x90B00039, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep11_map ,      0x90B0003A, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep12_map ,      0x90B0003B, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep13_map ,      0x90B0003C, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep14_map ,      0x90B0003D, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(ep15_map,       0x90B0003E, __READ_WRITE , __epx_map_bits);
__IO_REG8_BIT(hbf_cnt,        0x90B0003F, __READ       , __hbf_cnt_bits);  
__IO_REG16_BIT(iep1_xpsz,     0x90B00040, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep2_xpsz,     0x90B00042, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep3_xpsz,     0x90B00044, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep4_xpsz,     0x90B00046, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep5_xpsz,     0x90B00048, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep6_xpsz,     0x90B0004A, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep7_xpsz,     0x90B0004C, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep8_xpsz,     0x90B0004E, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep9_xpsz,     0x90B00050, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep10_xpsz,    0x90B00052, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep11_xpsz,    0x90B00054, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep12_xpsz,    0x90B00056, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep13_xpsz,    0x90B00058, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep14_xpsz,    0x90B0005A, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(iep15_xpsz,    0x90B0005C, __READ_WRITE , __iepx_xpsz_bits);
__IO_REG16_BIT(oep1_xpsz,     0x90B00060, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep2_xpsz,     0x90B00062, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep3_xpsz,     0x90B00064, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep4_xpsz,     0x90B00066, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep5_xpsz,     0x90B00068, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep6_xpsz,     0x90B0006A, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep7_xpsz,     0x90B0006C, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep8_xpsz,     0x90B0006E, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep9_xpsz,     0x90B00070, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep10_xpsz,    0x90B00072, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep11_xpsz,    0x90B00074, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep12_xpsz,    0x90B00076, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep13_xpsz,    0x90B00078, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep14_xpsz,    0x90B0007A, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(oep15_xpsz,    0x90B0007C, __READ_WRITE , __oepx_xpsz_bits);
__IO_REG16_BIT(fifo_dma_en,   0x90B0007E, __READ_WRITE , __fifo_dma_en_bits);
__IO_REG8_BIT(fifo0_map,      0x90B00080, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo1_map,      0x90B00081, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo2_map,      0x90B00082, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo3_map,      0x90B00083, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo4_map,      0x90B00084, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo5_map,      0x90B00085, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo6_map,      0x90B00086, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo7_map,      0x90B00087, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo8_map,      0x90B00088, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo9_map,      0x90B00089, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo10_map,     0x90B0008A, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo11_map,     0x90B0008B, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo12_map,     0x90B0008C, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo13_map,     0x90B0008D, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo14_map,     0x90B0008E, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo15_map,     0x90B0008F, __READ_WRITE , __fifox_map_bits);
__IO_REG8_BIT(fifo0_config,   0x90B00090, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo1_config,   0x90B00091, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo2_config,   0x90B00092, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo3_config,   0x90B00093, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo4_config,   0x90B00094, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo5_config,   0x90B00095, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo6_config,   0x90B00096, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo7_config,   0x90B00097, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo8_config,   0x90B00098, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo9_config,   0x90B00099, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo10_config,  0x90B0009A, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo11_config,  0x90B0009B, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo12_config,  0x90B0009C, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo13_config,  0x90B0009D, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo14_config,  0x90B0009E, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo15_config,  0x90B0009F, __READ_WRITE , __fifox_config_bits);
__IO_REG8_BIT(fifo0_inst,     0x90B000A0, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo1_inst,     0x90B000A1, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo2_inst,     0x90B000A2, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo3_inst,     0x90B000A3, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo4_inst,     0x90B000A4, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo5_inst,     0x90B000A5, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo6_inst,     0x90B000A6, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo7_inst,     0x90B000A7, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo8_inst,     0x90B000A8, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo9_inst,     0x90B000A9, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo10_inst,    0x90B000AA, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo11_inst,    0x90B000AB, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo12_inst,    0x90B000AC, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo13_inst,    0x90B000AD, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo14_inst,    0x90B000AE, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo15_inst,    0x90B000AF, __READ_WRITE , __fifox_inst_bits);
__IO_REG8_BIT(fifo0_bc,       0x90B000B0, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo1_bc,       0x90B000B1, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo2_bc,       0x90B000B2, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo3_bc,       0x90B000B3, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo4_bc,       0x90B000B4, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo5_bc,       0x90B000B5, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo6_bc,       0x90B000B6, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo7_bc,       0x90B000B7, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo8_bc,       0x90B000B8, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo9_bc,       0x90B000B9, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo10_bc,      0x90B000BA, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo11_bc,      0x90B000BB, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo12_bc,      0x90B000BC, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo13_bc,      0x90B000BD, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo14_bc,      0x90B000BE, __READ       , __fifox_bc_bits);
__IO_REG8_BIT(fifo15_bc,      0x90B000BF, __READ       , __fifox_bc_bits);
__IO_REG32(   fifo0_dp,       0x90B000C0, __READ_WRITE );
__IO_REG32(   fifo1_dp,       0x90B000C4, __READ_WRITE );
__IO_REG32(   fifo2_dp,       0x90B000C8, __READ_WRITE );
__IO_REG32(   fifo3_dp,       0x90B000CC, __READ_WRITE );
__IO_REG32(   fifo4_dp,       0x90B000D0, __READ_WRITE );
__IO_REG32(   fifo5_dp,       0x90B000D4, __READ_WRITE );
__IO_REG32(   fifo6_dp,       0x90B000D8, __READ_WRITE );
__IO_REG32(   fifo7_dp,       0x90B000DC, __READ_WRITE );
__IO_REG32(   fifo8_dp,       0x90B000E0, __READ_WRITE );
__IO_REG32(   fifo9_dp,       0x90B000E4, __READ_WRITE );
__IO_REG32(   fifo10_dp,      0x90B000E8, __READ_WRITE );
__IO_REG32(   fifo11_dp,      0x90B000EC, __READ_WRITE );
__IO_REG32(   fifo12_dp,      0x90B000F0, __READ_WRITE );
__IO_REG32(   fifo13_dp,      0x90B000F4, __READ_WRITE );
__IO_REG32(   fifo14_dp,      0x90B000F8, __READ_WRITE );
__IO_REG32(   fifo15_dp,      0x90B000FC, __READ_WRITE );

/***************************************************************************
 **
 ** INTC
 **
 ***************************************************************************/
__IO_REG32_BIT(IRQSR,         0x98800000, __READ       , __intsrc_bits);
__IO_REG32_BIT(IRQMR,         0x98800004, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(IRQICR,        0x98800008, __WRITE      , __intsrc_bits);
__IO_REG32_BIT(IRQTMR,        0x9880000C, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(IRQTLR,        0x98800010, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(IRQST,         0x98800014, __READ       , __intsrc_bits);
__IO_REG32_BIT(FIQSR,         0x98800020, __READ       , __intsrc_bits);
__IO_REG32_BIT(FIQMR,         0x98800024, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(FIQICR,        0x98800028, __WRITE      , __intsrc_bits);
__IO_REG32_BIT(FIQTMR,        0x9880002C, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(FIQTLR,        0x98800030, __READ_WRITE , __intsrc_bits);
__IO_REG32_BIT(FIQST,         0x98800034, __READ       , __intsrc_bits);
__IO_REG32_BIT(RRVISION,      0x98800050, __READ       , __rrvision_bits);
__IO_REG32_BIT(FRIN,          0x98800054, __READ       , __frin_bits);
__IO_REG32(    FRIDL,         0x98800058, __READ       );
__IO_REG32(    FRFDL,         0x9880005C, __READ       );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GpioDataOut,     0x98700000, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(GpioDataIn,      0x98700004, __READ       , __gpio_bits);
__IO_REG32_BIT(PinDir,          0x98700008, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(GpioDataSet,     0x98700010, __WRITE      , __gpio_bits);
__IO_REG32_BIT(GpioDataClear,   0x98700014, __WRITE      , __gpio_bits);
__IO_REG32_BIT(PinPullEnable,   0x98700018, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(PinPullType,     0x9870001C, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(IntrEnable,      0x98700020, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(IntrRawState,    0x98700024, __READ       , __gpio_bits);
__IO_REG32_BIT(IntrMaskedState, 0x98700028, __READ       , __gpio_bits);
__IO_REG32_BIT(IntrMask,        0x9870002C, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(IntrClear,       0x98700030, __WRITE      , __gpio_bits);
__IO_REG32_BIT(IntrTrigger,     0x98700034, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(IntrBoth,        0x98700038, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(IntrRiseNeg,     0x9870003C, __READ_WRITE , __gpio_bits);
__IO_REG32_BIT(BounceEnable,    0x98700040, __READ_WRITE , __gpio_bits);
__IO_REG32(    BouncePreScale,  0x98700044, __READ_WRITE );

/***************************************************************************
 **
 ** APB Bidge
 **
 ***************************************************************************/
__IO_REG32_BIT(APBSlave1BSR,    0x90500004, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave2BSR,    0x90500008, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave3BSR,    0x9050000C, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave4BSR,    0x90500010, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave5BSR,    0x90500014, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave6BSR,    0x90500018, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave8BSR,    0x90500020, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave11BSR,   0x9050002C, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave16BSR,   0x90500040, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave17BSR,   0x90500044, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave18BSR,   0x90500048, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave19BSR,   0x9050004C, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave20BSR,   0x90500050, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave21BSR,   0x90500054, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave22BSR,   0x90500058, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32_BIT(APBSlave23BSR,   0x9050005C, __READ_WRITE , __apbslavexbsr_bits);
__IO_REG32(    ASrcAddr,        0x90500080, __READ_WRITE );
__IO_REG32(    ADstAddr,        0x90500084, __READ_WRITE );
__IO_REG32_BIT(ACyc,            0x90500088, __READ_WRITE , __xcyc_bits);
__IO_REG32_BIT(ACmdR,           0x9050008C, __READ_WRITE , __xcmdr_bits);
__IO_REG32(    BSrcAddr,        0x90500090, __READ_WRITE );
__IO_REG32(    BDstAddr,        0x90500094, __READ_WRITE );
__IO_REG32_BIT(BCyc,            0x90500098, __READ_WRITE , __xcyc_bits);
__IO_REG32_BIT(BCmdR,           0x9050009C, __READ_WRITE , __xcmdr_bits);
__IO_REG32(    CSrcAddr,        0x905000A0, __READ_WRITE );
__IO_REG32(    CDstAddr,        0x905000A4, __READ_WRITE );
__IO_REG32_BIT(CCyc,            0x905000A8, __READ_WRITE , __xcyc_bits);
__IO_REG32_BIT(CCmdR,           0x905000AC, __READ_WRITE , __xcmdr_bits);
__IO_REG32(    DSrcAddr,        0x905000B0, __READ_WRITE );
__IO_REG32(    DDstAddr,        0x905000B4, __READ_WRITE );
__IO_REG32_BIT(DCyc,            0x905000B8, __READ_WRITE , __xcyc_bits);
__IO_REG32_BIT(DCmdR,           0x905000BC, __READ_WRITE , __xcmdr_bits);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0CR,          0x99100000, __READ_WRITE , __pwmxcr_bits);
__IO_REG32_BIT(PWM0DCR,         0x99100004, __READ_WRITE , __pwmxdcr_bits);
__IO_REG32_BIT(PWM0PCR,         0x99100008, __READ_WRITE , __pwmxpcr_bits);
__IO_REG32_BIT(PWM1CR,          0x99100010, __READ_WRITE , __pwmxcr_bits);
__IO_REG32_BIT(PWM1DCR,         0x99100014, __READ_WRITE , __pwmxdcr_bits);
__IO_REG32_BIT(PWM1PCR,         0x99100018, __READ_WRITE , __pwmxpcr_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_CR,          0x98A00000, __READ_WRITE , __i2c_cr_bits);
__IO_REG32_BIT(I2C_SR,          0x98A00004, __READ       , __i2c_sr_bits);
__IO_REG32_BIT(I2C_CDR,         0x98A00008, __READ_WRITE , __i2c_cdr_bits);
__IO_REG32_BIT(I2C_DR,          0x98A0000C, __READ_WRITE , __i2c_dr_bits);
__IO_REG32_BIT(I2C_SAR,         0x98A00010, __READ_WRITE , __i2c_sar_bits);
__IO_REG32_BIT(I2C_TGSR,        0x98A00014, __READ_WRITE , __i2c_tgsr_bits);
__IO_REG32_BIT(I2C_BMR,         0x98A00018, __READ       , __i2c_bmr_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32(    WdCounter,       0x98500000, __READ       );
__IO_REG32(    WdLoad,          0x98500004, __READ_WRITE );
__IO_REG16(    WdRestart,       0x98500008, __WRITE      );
__IO_REG32_BIT(WdCR,            0x9850000C, __READ_WRITE , __wdcr_bits);
__IO_REG32_BIT(WdStatus,        0x98500010, __READ       , __wdstatus_bits);
__IO_REG32_BIT(WdClear,         0x98500014, __WRITE      , __wdstatus_bits);
__IO_REG16(    WdIntrCter,      0x98500018, __READ_WRITE );

/***************************************************************************
 **
 ** Timers
 **
 ***************************************************************************/
__IO_REG32(    Tm1Counter,      0x98400000, __READ_WRITE );
__IO_REG32(    Tm1Load,         0x98400004, __READ_WRITE );
__IO_REG32(    Tm1Match1,       0x98400008, __READ_WRITE );
__IO_REG32(    Tm1Match2,       0x9840000C, __READ_WRITE );
__IO_REG32(    Tm2Counter,      0x98400010, __READ_WRITE );
__IO_REG32(    Tm2Load,         0x98400014, __READ_WRITE );
__IO_REG32(    Tm2Match1,       0x98400018, __READ_WRITE );
__IO_REG32(    Tm2Match2,       0x9840001C, __READ_WRITE );
__IO_REG32(    Tm3Counter,      0x98400020, __READ_WRITE );
__IO_REG32(    Tm3Load,         0x98400024, __READ_WRITE );
__IO_REG32(    Tm3Match1,       0x98400028, __READ_WRITE );
__IO_REG32(    Tm3Match2,       0x9840002C, __READ_WRITE );
__IO_REG32_BIT(TmCR,            0x98400030, __READ_WRITE , __tmcr_bits);
__IO_REG32_BIT(TmIntrState,     0x98400034, __READ_WRITE , __tmintrstate_bits);
__IO_REG32_BIT(TmIntrMask,      0x98400038, __READ_WRITE , __tmintrstate_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RtcSecond,       0x98600000, __READ       , __rtcsecond_bits);
__IO_REG32_BIT(RtcMinute,       0x98600004, __READ       , __rtcminute_bits);
__IO_REG32_BIT(RtcHour,         0x98600008, __READ       , __rtchour_bits);
__IO_REG32_BIT(RtcDays,         0x9860000C, __READ       , __rtcdays_bits);
__IO_REG32_BIT(AlarmSecond,     0x98600010, __READ_WRITE , __alarmsecond_bits);
__IO_REG32_BIT(AlarmMinute,     0x98600014, __READ_WRITE , __alarmminute_bits);
__IO_REG32_BIT(AlarmHour,       0x98600018, __READ_WRITE , __alarmhour_bits);
__IO_REG32(    RtcRecord,       0x9860001C, __READ_WRITE );
__IO_REG32_BIT(RtcCR,           0x98600020, __READ_WRITE , __rtccr_bits);     

/***************************************************************************
 **
 ** SDC
 **
 ***************************************************************************/
__IO_REG32_BIT(SDCR,            0x98E00000, __READ_WRITE , __sdcr_bits);
__IO_REG32(    SDAR,            0x98E00004, __READ_WRITE );
__IO_REG32(    SDRR0,           0x98E00008, __READ       );
__IO_REG32(    SDRR1,           0x98E0000C, __READ       );
__IO_REG32(    SDRR2,           0x98E00010, __READ       );
__IO_REG32(    SDRR3,           0x98E00014, __READ       );
__IO_REG32_BIT(SDRCR,           0x98E00018, __READ       , __sdrcr_bits);
__IO_REG32_BIT(SDDCR,           0x98E0001C, __READ_WRITE , __sddcr_bits);
__IO_REG32(    SDDTR,           0x98E00020, __READ_WRITE );
__IO_REG32_BIT(SDDLR,           0x98E00024, __READ_WRITE , __sddlr_bits);
__IO_REG32_BIT(SDSR,            0x98E00028, __READ       , __sdsr_bits);
__IO_REG32_BIT(SDCLR,           0x98E0002C, __WRITE      , __sdclr_bits);
__IO_REG32_BIT(SDIMR,           0x98E00030, __READ_WRITE , __sdclr_bits);
__IO_REG32_BIT(SDPCR,           0x98E00034, __READ_WRITE , __sdpcr_bits);
__IO_REG32_BIT(SDCCR,           0x98E00038, __READ_WRITE , __sdccr_bits);
__IO_REG32_BIT(SDBWR,           0x98E0003C, __READ_WRITE , __sdbwr_bits);
__IO_REG32(    SDDWR,           0x98E00040, __READ_WRITE );

/***************************************************************************
 **
 ** CFC
 **
 ***************************************************************************/
__IO_REG32_BIT(CFSR,            0x98D00000, __READ       ,__cfsr_bits);
__IO_REG32_BIT(CFCR,            0x98D00004, __READ_WRITE ,__cfcr_bits);
__IO_REG32_BIT(CFATCR,          0x98D00008, __READ_WRITE ,__cfatcr_bits);
__IO_REG32_BIT(CFABCR,          0x98D0000C, __READ_WRITE ,__cfabcr_bits);
__IO_REG32(    CFABDR,          0x98D00010, __READ_WRITE );
__IO_REG32_BIT(CFMSR,           0x98D00014, __READ_WRITE ,__cfmsr_bits);
__IO_REG32_BIT(TSMER,           0x98D00018, __READ_WRITE ,__tsmer_bits);
__IO_REG32_BIT(TSMCR,           0x98D0001C, __READ_WRITE ,__tsmcr_bits);

/***************************************************************************
 **
 **  FFUART
 **
 ***************************************************************************/
/* FFUART_DLL, FFUART_RBR and FFUART_THR share the same address */
__IO_REG8(    FFUART_RBR,              0x98200000,__READ_WRITE );
#define FFUART_THR FFUART_RBR
#define FFUART_DLL FFUART_RBR

/* FFUART_DLM and FFUART_IER share the same address */
__IO_REG8_BIT(FFUART_IER,              0x98200004,__READ_WRITE ,__uartier_bits);
#define FFUART_DLM      FFUART_IER
/* FFUART_FCR, FFUART_IIR and FFUART_PSR share the same address */
__IO_REG8_BIT(FFUART_IIR,              0x98200008,__READ_WRITE ,__uartiir_bits);
#define FFUART_FCR FFUART_IIR
#define FFUART_FCR_bit  FFUART_IIR_bit
#define FFUART_PSR FFUART_IIR
#define FFUART_PSR_bit  FFUART_IIR_bit

__IO_REG8_BIT(FFUART_LCR,              0x9820000C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT(FFUART_MCR,              0x98200010,__READ_WRITE ,__uartmcr_bits);
__IO_REG8_BIT(FFUART_LSR,              0x98200014,__READ       ,__uartlsr_bits);
__IO_REG8_BIT(FFUART_MSR,              0x98200018,__READ       ,__uartmsr_bits);
__IO_REG8(    FFUART_SPR,              0x9820001C,__READ_WRITE );

/***************************************************************************
 **
 **  BTUART
 **
 ***************************************************************************/
/* BTUART_DLL, BTUART_RBR and BTUART_THR share the same address */
__IO_REG8(    BTUART_RBR,              0x98300000,__READ_WRITE  );
#define BTUART_THR BTUART_RBR
#define BTUART_DLL BTUART_RBR

/* BTUART_DLM and BTUART_IER share the same address */
__IO_REG8_BIT(BTUART_IER,              0x98300004,__READ_WRITE ,__uartier_bits);
#define BTUART_DLM      BTUART_IER
/* BTUART_FCR, BTUART_IIR and BTUART_PSR share the same address */
__IO_REG8_BIT(BTUART_IIR,              0x98300008,__READ_WRITE ,__uartiir_bits);
#define BTUART_FCR BTUART_IIR
#define BTUART_FCR_bit  BTUART_IIR_bit
#define BTUART_PSR BTUART_IIR
#define BTUART_PSR_bit  BTUART_IIR_bit

__IO_REG8_BIT(BTUART_LCR,              0x9830000C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT(BTUART_MCR,              0x98300010,__READ_WRITE ,__uartmcr_bits);
__IO_REG8_BIT(BTUART_LSR,              0x98300014,__READ       ,__uartlsr_bits);
__IO_REG8_BIT(BTUART_MSR,              0x98300018,__READ       ,__uartmsr_bits);
__IO_REG8(    BTUART_SPR,              0x9830001C,__READ_WRITE );

/***************************************************************************
 **
 **  STUART
 **
 ***************************************************************************/
/* STUART_DLL, STUART_RBR and STUART_THR share the same address */
__IO_REG8(    STUART_RBR,              0x99600000,__READ_WRITE );
#define STUART_THR STUART_RBR
#define STUART_DLL STUART_RBR

/* STUART_DLM and STUART_IER share the same address */
__IO_REG8_BIT(STUART_IER,              0x99600004,__READ_WRITE ,__uartier_bits);
#define STUART_DLM      STUART_IER
/* STUART_FCR, STUART_IIR and STUART_PSR share the same address */
__IO_REG8_BIT(STUART_IIR,              0x99600008,__READ_WRITE ,__uartiir_bits);
#define STUART_FCR STUART_IIR
#define STUART_FCR_bit  STUART_IIR_bit
#define STUART_PSR STUART_IIR
#define STUART_PSR_bit  STUART_IIR_bit

__IO_REG8_BIT(STUART_LCR,              0x9960000C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT(STUART_LSR,              0x99600014,__READ       ,__uartlsr_bits);
__IO_REG8(    STUART_SPR,              0x9960001C,__READ_WRITE );

/***************************************************************************
 **
 **  IrDA
 **
 ***************************************************************************/
/* IrDA_DLL, IrDA_RBR and IrDA_THR share the same address */
__IO_REG8(    IrDA_RBR,              0x98900000,__READ_WRITE );
#define IrDA_THR IrDA_RBR
#define IrDA_DLL IrDA_RBR

/* IrDA_DLM and IrDA_IER share the same address */
__IO_REG8_BIT(IrDA_IER,              0x98900004,__READ_WRITE ,__uartier_bits);
#define IrDA_DLM      IrDA_IER
/* IrDA_FCR, IrDA_IIR and IrDA_PSR share the same address */
__IO_REG8_BIT(IrDA_IIR,              0x98900008,__READ_WRITE ,__uartiir_bits);
#define IrDA_FCR IrDA_IIR
#define IrDA_FCR_bit  IrDA_IIR_bit
#define IrDA_PSR IrDA_IIR
#define IrDA_PSR_bit  IrDA_IIR_bit

__IO_REG8_BIT(IrDA_LCR,              0x9890000C,__READ_WRITE ,__uartlcr_bits);
__IO_REG8_BIT(IrDA_LSR,              0x98900014,__READ       ,__uartlsr_bits);
__IO_REG8(    IrDA_SPR,              0x9890001C,__READ_WRITE );

__IO_REG8_BIT(MDR,                   0x98900020,__READ_WRITE  ,__mdr_bits);
__IO_REG8_BIT(ACR,                   0x98900024,__READ_WRITE  ,__acr_bits);
__IO_REG8_BIT(TXLENL,                0x98900028,__READ_WRITE  ,__txlenl_bits);
__IO_REG8_BIT(TXLENH,                0x9890002C,__READ_WRITE  ,__txlenh_bits);
__IO_REG8_BIT(MRXLENL,               0x98900030,__READ_WRITE  ,__mrxlenl_bits);
__IO_REG8_BIT(MRXLENH,               0x98900034,__READ_WRITE  ,__mrxlenh_bits);
__IO_REG8_BIT(PLR,                   0x98900038,__READ_WRITE  ,__plr_bits);
__IO_REG8_BIT(FMIIR_PIO,             0x9890003C,__READ        ,__fmiir_pio_dma_bits);
#define FMIIR_DMA FMIIR_PIO
#define FMIIR_DMA_bit FMIIR_PIO_bit
__IO_REG8_BIT(FMIIER_PIO,             0x98900040,__READ_WRITE  ,__fmier_pio_dma_bits);
#define FMIIER_DMA FMIIER_PIO
#define FMIIER_DMA_bit FMIIER_PIO_bit
__IO_REG8_BIT(STFF_STS,              0x98900044,__READ        ,__stff_sts_bits);
__IO_REG8_BIT(STFF_RXLENL,           0x98900048,__READ        ,__stff_rxlenl_bits);
__IO_REG8_BIT(STFF_RXLENH,           0x9890004C,__READ        ,__stff_rxlenh_bits);
__IO_REG8_BIT(FMLSR,                 0x98900050,__READ        ,__fmlsr_bits);
__IO_REG8_BIT(FMLSIER,               0x98900054,__READ_WRITE  ,__fmlsr_bits);
__IO_REG8(    RSR,                   0x98900058,__READ        );
__IO_REG8_BIT(RXFF_CNTR,             0x9890005C,__READ        ,__rxff_cntr_bits);
__IO_REG8_BIT(LSTFMLENL,             0x98900060,__READ_WRITE  ,__lstfmlenl_bits);
__IO_REG8_BIT(LSTFMLENH,             0x98900064,__READ_WRITE  ,__lstfmlenh_bits);     

/***************************************************************************
 **
 **  SSP
 **
 ***************************************************************************/
__IO_REG32_BIT(SSPCR0,           0x98B00000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(SSPCR1,           0x98B00004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(SSPCR2,           0x98B00008, __READ_WRITE , __sspcr2_bits);
__IO_REG32_BIT(SSPSR,            0x98B0000C, __READ       , __sspsr_bits);
__IO_REG32_BIT(SSPICR,           0x98B00010, __READ_WRITE , __sspicr_bits);
__IO_REG32_BIT(SSPISR,           0x98B00014, __READ       , __sspisr_bits);
__IO_REG32(    SSPDR,            0x98B00018, __READ_WRITE );
__IO_REG32_BIT(SSPVR,            0x98B00020, __READ_WRITE , __sspvr_bits);

/***************************************************************************
 **
 **  I2S/AC92
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_AC97CR0,      0x99400000, __READ_WRITE , __sspcr0_bits);
__IO_REG32_BIT(I2S_AC97CR1,      0x99400004, __READ_WRITE , __sspcr1_bits);
__IO_REG32_BIT(I2S_AC97CR2,      0x99400008, __READ_WRITE , __sspcr2_bits);
__IO_REG32_BIT(I2S_AC97SR,       0x9940000C, __READ       , __sspsr_bits);
__IO_REG32_BIT(I2S_AC97ICR,      0x99400010, __READ_WRITE , __sspicr_bits);
__IO_REG32_BIT(I2S_AC97ISR,      0x99400014, __READ       , __sspisr_bits);
__IO_REG32(    I2S_AC97DR,       0x99400018, __READ_WRITE );
__IO_REG32_BIT(I2S_AC97VR,       0x99400020, __READ_WRITE , __sspvr_bits);


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
#define RESETV          0x00  /* Reset                              */
#define UNDEFV          0x04  /* Undefined instruction              */
#define SWIV            0x08  /* Software interrupt                 */
#define PABORTV         0x0C  /* Prefetch abort                     */
#define DABORTV         0x10  /* Data abort                         */
#define IRQV            0x18  /* Normal interrupt                   */
#define FIQV            0x1C  /* Fast interrupt                     */

/***************************************************************************
 **
 ** Interrupt Routing
 **
 ***************************************************************************/
#define CFC_INT_CD_R     0    /* CFC */
#define CFC_INT_DMA_R    1    /* CFC */
#define SSP1INTR         2    /* SSP */
#define ISI2C            3    /* I2C */
#define SDC_INTR         5    /* SDC */
#define SSP2INTR         6    /* I2S/AC97 */
#define UARTINTR4        7    /* STUART */
#define PMU_FIQ          8    /* PMU */
#define UARTINTR1       10    /* FTUART */
#define UARTINTR2       11    /* BTUART */
#define GPIO_INTR       13    /* GPIO */
#define TM2_INTR        14    /*Timer 2 interrupt signal*/
#define TM3_INTR        15    /*Timer 3 interrupt signal*/
#define WD_INTR         16    /*Watch Dog Timer system interrupt signal*/
#define RTC_ALARM       17    /*RTC*/
#define RTC_SEC         18    /*RTC*/
#define TM1_INTR        19    /*Timer 1 interrupt signal*/
#define DMAINT          21    /*DMA*/
#define IRDA_INT1       22    /*IrDA*/
#define IRDA_INT2       23    /*IrDA*/
#define RSHINT          24    /*APB Bridge*/
#define MAC_INT         25    /*Ethernet MAC*/
#define USB_INT0        26    /*USB 2.0 Device*/
#define EXT_INT_IRQ0    28    /* External Interrupt */
#define EXT_INT_IRQ1    29    /* External Interrupt */
#define EXT_INT_IRQ2    30    /* External Interrupt */
#define EXT_INT_IRQ3    31    /* External Interrupt */

#endif    /* __IOFA606TE_H */
