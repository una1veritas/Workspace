/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton W90P910
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2008
 **
 **    $Revision: 39368 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IOW90P910_H
#define __IOW90P910_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   W90P910 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
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

/* Product Identifier Register (PDID) */
typedef struct{
__REG32 CHPID           :24;
__REG32 VERSION         : 8;
} __pdid_bits;

/* Power-On Setting Register (PWRON) */
typedef struct{
__REG32 PLL             : 1;
__REG32 GPIOSEL         : 3;
__REG32                 : 2;
__REG32 BOOTMODE        : 2;
__REG32                 : 1;
__REG32 USBHD           : 1;
__REG32 USBDEN          : 1;
__REG32                 :21;
} __pwron_bits;

/* Arbitration Control Register (ARBCON) */
typedef struct{
__REG32 PRTMOD0         : 1;
__REG32 PRTMOD1         : 1;
__REG32 IPEN            : 1;
__REG32 IPACT           : 1;
__REG32 DGMASK          : 1;
__REG32                 :27;
} __arbcon_bits;

/* Multiple Function Pin Select Register (MFSEL) */
typedef struct{
__REG32 G_Option        : 1;
__REG32 GPSELF          : 1;
__REG32 GPSELC          : 2;
__REG32 GPSELD          : 4;
__REG32 GPSELE          : 6;
__REG32 GPSELG          :10;
__REG32 GPSELH          : 2;
__REG32 GPSELI          : 2;
__REG32 USBPHY0         : 2;
__REG32                 : 2;
} __mfsel_bits;

/* EBI Data Pin Pull-up/down Enable Register (EBIDPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32 PPE14           : 1;
__REG32 PPE15           : 1;
__REG32 PPE16           : 1;
__REG32 PPE17           : 1;
__REG32 PPE18           : 1;
__REG32 PPE19           : 1;
__REG32 PPE20           : 1;
__REG32 PPE21           : 1;
__REG32 PPE22           : 1;
__REG32 PPE23           : 1;
__REG32 PPE24           : 1;
__REG32 PPE25           : 1;
__REG32 PPE26           : 1;
__REG32 PPE27           : 1;
__REG32 PPE28           : 1;
__REG32 PPE29           : 1;
__REG32 PPE30           : 1;
__REG32 PPE31           : 1;
} __ebidpe_bits;

/* LCD Data Pin Pull-up/down Enable Register (LCDDPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32 PPE14           : 1;
__REG32 PPE15           : 1;
__REG32 PPE16           : 1;
__REG32 PPE17           : 1;
__REG32                 :14;
} __lcddpe_bits;

/* GPIOC Pin Pull-up/down Enable Register (GPIOCPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32 PPE14           : 1;
__REG32                 :17;
} __gpiocpe_bits;

/* GPIOD Pin Pull-up/down Enable Register (GPIODPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32                 :22;
} __gpiodpe_bits;

/* GPIOE Pin Pull-up/down Enable Register (GPIOEPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32                 :18;
} __gpioepe_bits;

/* GPIOF Pin Pull-up/down Enable Register (GPIOFPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32                 :22;
} __gpiofpe_bits;

/* GPIOG Pin Pull-up/down Enable Register (GPIOGPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32 PPE14           : 1;
__REG32 PPE15           : 1;
__REG32 PPE16           : 1;
__REG32                 :15;
} __gpiogpe_bits;

/* GPIOH Pin Pull-up/down Enable Register (GPIOHPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32                 :24;
} __gpiohpe_bits;

/* GPIOI Pin Pull-up/down Enable Register (GPIOIPE) */
typedef struct{
__REG32 PPE0            : 1;
__REG32 PPE1            : 1;
__REG32 PPE2            : 1;
__REG32 PPE3            : 1;
__REG32 PPE4            : 1;
__REG32 PPE5            : 1;
__REG32 PPE6            : 1;
__REG32 PPE7            : 1;
__REG32 PPE8            : 1;
__REG32 PPE9            : 1;
__REG32 PPE10           : 1;
__REG32 PPE11           : 1;
__REG32 PPE12           : 1;
__REG32 PPE13           : 1;
__REG32 PPE14           : 1;
__REG32 PPE15           : 1;
__REG32 PPE16           : 1;
__REG32 PPE17           : 1;
__REG32 PPE18           : 1;
__REG32 PPE19           : 1;
__REG32 PPE20           : 1;
__REG32 PPE21           : 1;
__REG32 PPE22           : 1;
__REG32 PPE23           : 1;
__REG32 PPE24           : 1;
__REG32 PPE25           : 1;
__REG32 PPE26           : 1;
__REG32 PPE27           : 1;
__REG32                 : 4;
} __gpioipe_bits;

/*Clock Enable Register (CLKEN)*/
typedef struct{
__REG32 LCD             : 1;
__REG32 AUDIO           : 1;
__REG32                 : 2;
__REG32 FMI             : 1;
__REG32 DMAC            : 1;
__REG32 ATAPI           : 1;
__REG32 EMC             : 1;
__REG32 USBD            : 1;
__REG32 USBH            : 1;
__REG32 G2D             : 1;
__REG32 UART0           : 1;
__REG32 UART1           : 1;
__REG32 UART2           : 1;
__REG32 UART3           : 1;
__REG32 UART4           : 1;
__REG32 SCH0            : 1;
__REG32 SCH1            : 1;
__REG32 PWM             : 1;
__REG32 TIMER0          : 1;
__REG32 TIMER1          : 1;
__REG32 TIMER2          : 1;
__REG32 TIMER3          : 1;
__REG32 TIMER4          : 1;
__REG32 PS2             : 1;
__REG32 KPI             : 1;
__REG32 WDT             : 1;
__REG32 GDMA            : 1;
__REG32 ADC             : 1;
__REG32 USI             : 1;
__REG32 I2C0            : 1;
__REG32 I2C1            : 1;
} __clken_bits;

/*Clock Select Register (CLKSEL)*/
typedef struct{
__REG32 CPUCKSEL        : 2;
__REG32                 : 2;
__REG32 ACKSEL          : 2;
__REG32 VCKSEL          : 2;
__REG32 UART1SEL        : 2;
__REG32 ATASEL          : 2;
__REG32 MSDSEL          : 5;
__REG32                 :15;
} __clksel_bits;

/*Clock Divider Control Register (CLKDIV)*/
typedef struct{
__REG32 CPUCKDIV        : 4;
__REG32                 : 4;
__REG32 ACKDIV          : 4;
__REG32 VCKDIV          : 4;
__REG32 UART1DIV        : 4;
__REG32 ATAPIDIV        : 4;
__REG32 AHBCKDIV        : 2;
__REG32 APBCKDIV        : 2;
__REG32 ADCCKDIV        : 2;
__REG32 G2DDIV          : 1;
__REG32                 : 1;
} __clkdiv_bits;

/* PLL Control Register 0(PLLCON0) */
/* PLL Control Register 1(PLLCON1) */
typedef struct{
__REG32 INDV            : 5;
__REG32 OTDV            : 2;
__REG32 FBDV            : 9;
__REG32 PWDEN           : 1;
__REG32                 :15;
} __pllconx_bits;

/* Power Management Control Register (PMCON) */
typedef struct{
__REG32 IDLE            : 1;
__REG32 PD              : 1;
__REG32 MIDLE           : 1;
__REG32 RESET           : 1;
__REG32                 :28;
} __pmcon_bits;

/* IRQ Wakeup Control Register (IRQWAKECON) */
typedef struct{
__REG32 IRQWAKEUPEN0    : 4;
__REG32 IRQWAKEUPEN1    : 4;
__REG32 IRQWAKEUPPOL0   : 4;
__REG32 IRQWAKEUPPOL1   : 4;
__REG32                 :16;
} __irqwakecon_bits;

/* IRQ Wakeup Flag Register (IRQWAKEFLAG) */
typedef struct{
__REG32 IRQWAKEFLAG     : 8;
__REG32                 :24;
} __irqwakeflag_bits;

/* IP Software Reset Register (IPSRST) */
typedef struct{
__REG32 LCD             : 1;
__REG32 AUDIO           : 1;
__REG32                 : 1;
__REG32 GDMA            : 1;
__REG32 FMI             : 1;
__REG32 DMAC            : 1;
__REG32 ATAPI           : 1;
__REG32 EMC             : 1;
__REG32 USBD            : 1;
__REG32 USBH            : 1;
__REG32 G2D             : 1;
__REG32 UART            : 1;
__REG32                 : 4;
__REG32 SCH             : 1;
__REG32                 : 1;
__REG32 PWM             : 1;
__REG32 TIMER           : 1;
__REG32                 : 4;
__REG32 PS2             : 1;
__REG32 KPI             : 1;
__REG32                 : 2;
__REG32 ADC             : 1;
__REG32 USI             : 1;
__REG32 I2C             : 1;
__REG32                 : 1;
} __ipsrst_bits;

/* Clock Enable 1 Register (CLKEN1) */
typedef struct{
__REG32 MS              : 1;
__REG32 SD              : 1;
__REG32 RMII            : 1;
__REG32                 :29;
} __clken1_bits;

/* Clock Divider Control 1 Register (CLKDIV1) */
typedef struct{
__REG32 MS_DIV          : 8;
__REG32 SD_DIV          : 8;
__REG32                 :16;
} __clkdiv1_bits;

/* External Bus Interface Control Register (EBICON) */
typedef struct{
__REG32 LITTLE          : 1;
__REG32 WAITVT          : 2;
__REG32 REFRAT          :13;
__REG32 CLKEN           : 1;
__REG32 REFMOD          : 1;
__REG32 REFEN           : 1;
__REG32                 : 5;
__REG32 EXBE0           : 1;
__REG32 EXBE1           : 1;
__REG32 EXBE2           : 1;
__REG32 EXBE3           : 1;
__REG32 EXBE4           : 1;
__REG32                 : 3;
} __ebicon_bits;

/* ROM/Flash Control Register (ROMCON) */
typedef struct{
__REG32 PGMODE          : 2;
__REG32 BTSIZE          : 2;
__REG32 TACC            : 4;
__REG32 TPA             : 4;
__REG32                 : 3;
__REG32 SIZE            : 4;
__REG32 BASADDR         :13;
} __romcon_bits;

/* SDRAM Configuration Registers(SDCONF0/1) */
typedef struct{
__REG32 SIZE            : 3;
__REG32 COLUMN          : 2;
__REG32 DBWD            : 2;
__REG32 COMPBK          : 1;
__REG32                 : 3;
__REG32 LATENCY         : 2;
__REG32 AUTOPR          : 1;
__REG32                 : 1;
__REG32 MRSET           : 1;
__REG32                 : 3;
__REG32 BASADDR         :13;
} __sdconf_bits;

/* SDRAM Timing Control Registers (SDTIME0/1) */
typedef struct{
__REG32 TRAS            : 3;
__REG32 TRP             : 3;
__REG32 TRDL            : 2;
__REG32 TRCD            : 3;
__REG32                 :21;
} __sdtime_bits;

/* External I/O Control Registers(EXT0CON – EXT4CON) */
typedef struct{
__REG32 DBWD            : 2;
__REG32 TCOS            : 3;
__REG32 TACS            : 3;
__REG32 TCOH            : 3;
__REG32 TACC            : 4;
__REG32 ADRS            : 1;
__REG32 SIZE            : 3;
__REG32 BASADDR         :13;
} __extcon_bits;

/* Clock Skew Control Register (CKSKEW) */
typedef struct{
__REG32 MCLK_O_D        : 4;
__REG32 DLH_CLK_SKEW    : 4;
__REG32                 :24;
} __ckskew_bits;

/* CAM Command Register (CAMCMR) */
typedef struct{
__REG32 AUP             : 1;
__REG32 AMP             : 1;
__REG32 ABP             : 1;
__REG32 CCAM            : 1;
__REG32 ECMP            : 1;
__REG32 RMII            : 1;
__REG32                 :26;
} __camcmr_bits;

/* CAM Enable Register (CAMEN) */
typedef struct{
__REG32 CAM0EN          : 1;
__REG32 CAM1EN          : 1;
__REG32 CAM2EN          : 1;
__REG32 CAM3EN          : 1;
__REG32 CAM4EN          : 1;
__REG32 CAM5EN          : 1;
__REG32 CAM6EN          : 1;
__REG32 CAM7EN          : 1;
__REG32 CAM8EN          : 1;
__REG32 CAM9EN          : 1;
__REG32 CAM10EN         : 1;
__REG32 CAM11EN         : 1;
__REG32 CAM12EN         : 1;
__REG32 CAM13EN         : 1;
__REG32 CAM14EN         : 1;
__REG32 CAM15EN         : 1;
__REG32                 :16;
} __camen_bits;

/* CAM Entry Registers (CAMxM) */
typedef struct{
__REG32 MAC_ADDR2       : 8;
__REG32 MAC_ADDR3       : 8;
__REG32 MAC_ADDR4       : 8;
__REG32 MAC_ADDR5       : 8;
} __camm_bits;

/* CAM Entry Registers (CAMxL) */
typedef struct{
__REG32                 :16;
__REG32 MAC_ADDR0       : 8;
__REG32 MAC_ADDR1       : 8;
} __caml_bits;

/* CAM Entry Registers (CAM15M) */
typedef struct{
__REG32 OP_CODE         :16;
__REG32 LENGTH_TYPE     :16;
} __cam15m_bits;

/* CAM Entry Registers (CAM15L) */
typedef struct{
__REG32                 :16;
__REG32 OPERAND         :16;
} __cam15l_bits;

/* MAC Command Register (MCMDR) */
typedef struct{
__REG32 RXON            : 1;
__REG32 ALP             : 1;
__REG32 ARP             : 1;
__REG32 ACP             : 1;
__REG32 AEP             : 1;
__REG32 SPCRC           : 1;
__REG32                 : 2;
__REG32 TXON            : 1;
__REG32 NDEF            : 1;
__REG32                 : 6;
__REG32 SDPZ            : 1;
__REG32 ENSQE           : 1;
__REG32 FDUP            : 1;
__REG32 ENMDC           : 1;
__REG32 OPMOD           : 1;
__REG32 LBK             : 1;
__REG32                 : 2;
__REG32 SWR             : 1;
__REG32                 : 7;
} __mcmdr_bits;

/* MII Management Data Register (MIID) */
typedef struct{
__REG32 MIIDATA         :16;
__REG32                 :16;
} __miid_bits;

/* MII Management Control and Address Register (MIIDA) */
typedef struct{
__REG32 PHYRAD          : 5;
__REG32                 : 3;
__REG32 PHYAD           : 5;
__REG32                 : 3;
__REG32 WRITE           : 1;
__REG32 BUSY            : 1;
__REG32 PRESP           : 1;
__REG32 MDCON           : 1;
__REG32 MDCCR           : 4;
__REG32                 : 8;
} __miida_bits;

/* FIFO Threshold Control Register (FFTCR) */
typedef struct{
__REG32 RXTHD           : 2;
__REG32                 : 6;
__REG32 TXTHD           : 2;
__REG32                 :10;
__REG32 BLENGTH         : 2;
__REG32                 :10;
} __fftcr_bits;

/* Maximum Receive Frame Control Register (DMARFC) */
typedef struct{
__REG32 RXMS            :16;
__REG32                 :16;
} __dmarfc_bits;

/* MAC Interrupt Enable Register (MIEN) */
typedef struct{
__REG32 ENRXINTR        : 1;
__REG32 ENCRCE          : 1;
__REG32 ENRXOV          : 1;
__REG32 ENPTLE          : 1;
__REG32 ENRXGD          : 1;
__REG32 ENALIE          : 1;
__REG32 ENRP            : 1;
__REG32 ENMMP           : 1;
__REG32 ENDFO           : 1;
__REG32 ENDEN           : 1;
__REG32 ENRDU           : 1;
__REG32 ENRXBERR        : 1;
__REG32                 : 2;
__REG32 ENCFR           : 1;
__REG32                 : 1;
__REG32 ENTXINTR        : 1;
__REG32 ENTXEMP         : 1;
__REG32 ENTXCP          : 1;
__REG32 ENEXDEF         : 1;
__REG32 ENNCS           : 1;
__REG32 ENTXABT         : 1;
__REG32 ENLC            : 1;
__REG32 ENTDU           : 1;
__REG32 ENTXBERR        : 1;
__REG32                 : 7;
} __mien_bits;

/* MAC Interrupt Status Register (MISTA) */
typedef struct{
__REG32 RXINTR          : 1;
__REG32 CRCE            : 1;
__REG32 RXOV            : 1;
__REG32 PTLE            : 1;
__REG32 RXGD            : 1;
__REG32 ALIE            : 1;
__REG32 RP              : 1;
__REG32 MMP             : 1;
__REG32 DFOI            : 1;
__REG32 DENI            : 1;
__REG32 RDU             : 1;
__REG32 RXBERR          : 1;
__REG32                 : 2;
__REG32 CFR             : 1;
__REG32                 : 1;
__REG32 TXINTR          : 1;
__REG32 TXEMP           : 1;
__REG32 TXCP            : 1;
__REG32 EXDEF           : 1;
__REG32 NCS             : 1;
__REG32 TXABT           : 1;
__REG32 LC              : 1;
__REG32 TDU             : 1;
__REG32 TXBERR          : 1;
__REG32                 : 7;
} __mista_bits;

/* MAC General Status Register (MGSTA) */
typedef struct{
__REG32 CFR             : 1;
__REG32 RXHA            : 1;
__REG32 RFFULL          : 1;
__REG32                 : 1;
__REG32 CCNT            : 4;
__REG32 DEF             : 1;
__REG32 PAU             : 1;
__REG32 SQE             : 1;
__REG32 TXHA            : 1;
__REG32                 :20;
} __mgsta_bits;

/* Missed Packet Count Register (MPCNT) */
typedef struct{
__REG32 MPC             :16;
__REG32                 :16;
} __mpcnt_bits;

/* MAC Receive Pause Count Register (MRPC) */
typedef struct{
__REG32 MRPC            :16;
__REG32                 :16;
} __mrpc_bits;

/* MAC Receive Pause Current Count Register (MRPCC) */
typedef struct{
__REG32 MRPCC           :16;
__REG32                 :16;
} __mrpcc_bits;

/* MAC Remote Pause Count Register (MREPC) */
typedef struct{
__REG32 MREPC           :16;
__REG32                 :16;
} __mrepc_bits;

/* DMA Receive Frame Status Register (DMARFS) */
typedef struct{
__REG32 RXFLT           :16;
__REG32                 :16;
} __dmarfs_bits;

/* Channel 0/1 Control Register (GDMA_CTL0, GDMA_CTL1) */
typedef union{
  /*Non-Descriptor fetches Mode*/
  /*GDMA_CTLx*/
  struct{
    __REG32 GDMAEN          : 1;
    __REG32 BME             : 1;
    __REG32 GDMAMS          : 2;
    __REG32 DADIR           : 1;
    __REG32 SADIR           : 1;
    __REG32 DAFIX           : 1;
    __REG32 SAFIX           : 1;
    __REG32                 : 3;
    __REG32 SBMS            : 1;
    __REG32 TWS             : 2;
    __REG32                 : 1;
    __REG32 DM              : 1;
    __REG32 SOFTREQ         : 1;
    __REG32 BLOCK           : 1;
    __REG32                 : 1;
    __REG32 AUTOIEN         : 1;
    __REG32                 : 1;
    __REG32 DABNDERR        : 1;
    __REG32 SABNDERR        : 1;
    __REG32                 : 1;
    __REG32 ACK_ATV         : 1;
    __REG32 REQ_ATV         : 1;
    __REG32                 : 6;
 };
 /*Descriptor fetches Mode*/
 /*_GDMA_CTLx*/
 struct{
    __REG32 _GDMAEN         : 1;
    __REG32 _BME            : 1;
    __REG32 _GDMAMS         : 2;
    __REG32 _DADIR          : 1;
    __REG32 _SADIR          : 1;
    __REG32 _DAFIX          : 1;
    __REG32 _SAFIX          : 1;
    __REG32                 : 2;
    __REG32 _D_INTS         : 1;
    __REG32                 : 1;
    __REG32 _TWS            : 2;
    __REG32                 : 2;
    __REG32 _SOFTREQ        : 1;
    __REG32 _BLOCK          : 1;
    __REG32                 : 3;
    __REG32 _DABNDERR       : 1;
    __REG32 _SABNDERR       : 1;
    __REG32                 : 1;
    __REG32 _ACK_ATV        : 1;
    __REG32 _REQ_ATV        : 1;
    __REG32                 : 6;
 };
} __gdma_ctl_bits;

/* Channel 0/1 Transfer Count Register (GDMA_TCNT0, GDMA_TCNT1) */
typedef struct{
__REG32 TFR_CNT         :24;
__REG32                 : 8;
} __gdma_tcnt_bits;

/* Channel 0/1 Descriptor Register (GDMA_DADR0/1) */
typedef struct{
__REG32 RESET           : 1;
__REG32 ORDEN           : 1;
__REG32 NON_DSPTRMODE   : 1;
__REG32 RUN             : 1;
__REG32 DES_ADDR        :28;
} __gdma_dadr_bits;

/* Channel 0/1 GDMA Interrupt Control and Status Register (GDMA_INTCS) */
typedef struct{
__REG32 TC0EN           : 1;
__REG32 TERR0EN         : 1;
__REG32 TC1EN           : 1;
__REG32 TERR1EN         : 1;
__REG32                 : 4;
__REG32 TC0F            : 1;
__REG32 TERR0F          : 1;
__REG32 TC1F            : 1;
__REG32 TERR1F          : 1;
__REG32                 : 4;
__REG32 BUF_RD_SEL      : 2;
__REG32                 :14;
} __gdma_intcs_bits;

/* EHCI Version Number Register (EHCVNR) */
typedef struct{
__REG32 CR_LENGTH       : 8;
__REG32                 : 8;
__REG32 VERSION         :16;
} __ehcvnr_bits;

/* EHCI Structural Parameters Register (EHCSPR) */
typedef struct{
__REG32 N_PORTS         : 4;
__REG32 PPC             : 1;
__REG32                 : 3;
__REG32 N_PCC           : 4;
__REG32 N_CC            : 4;
__REG32                 :16;
} __ehcspr_bits;

/* EHCI Capability Parameters Register (EHCCPR) */
typedef struct{
__REG32 _64B            : 1;
__REG32 PFLIST          : 1;
__REG32 ASPC            : 1;
__REG32                 : 1;
__REG32 ISO_SCH_TH      : 4;
__REG32 EECP            : 8;
__REG32                 :16;
} __ehccpr_bits;

/* USB Command Register (UCMDR) */
typedef struct{
__REG32 RUNSTOP         : 1;
__REG32 HCRESET         : 1;
__REG32 FLSIZE          : 2;
__REG32 PSEN            : 1;
__REG32 ASEN            : 1;
__REG32 ASYNADB         : 1;
__REG32                 : 9;
__REG32 INT_TH_CTL      : 8;
__REG32                 : 8;
} __ucmdr_bits;

/* USB Status Register (USTSR) */
typedef struct{
__REG32 USBINT          : 1;
__REG32 UERRINT         : 1;
__REG32 PORTCHG         : 1;
__REG32 FLROVER         : 1;
__REG32 HSERR           : 1;
__REG32 INTASYNA        : 1;
__REG32                 : 6;
__REG32 HCHALTED        : 1;
__REG32 RECLA           : 1;
__REG32 PSSTS           : 1;
__REG32 ASSTS           : 1;
__REG32                 :16;
} __ustsr_bits;

/* USB Interrupt Enable Register (UIENR) */
typedef struct{
__REG32 USBIEN          : 1;
__REG32 UERREN          : 1;
__REG32 PCHGEN          : 1;
__REG32 FLREN           : 1;
__REG32 HSERREN         : 1;
__REG32 ASYNAEN         : 1;
__REG32                 :26;
} __uienr_bits;

/* USB Frame Index Register (UFINDR) */
typedef struct{
__REG32 FRAMEIND        :14;
__REG32                 :18;
} __ufindr_bits;

/* USB Periodic Frame List Base Address Register (UPFLBAR) */
typedef struct{
__REG32                 :12;
__REG32 BADDR           :20;
} __upflbar_bits;

/* USB Current Asynchronous List Address Register (UCALAR) */
typedef struct{
__REG32                 : 5;
__REG32 LPL             :27;
} __ucalar_bits;

/* USB Asynchronous Schedule Sleep Timer Register */
typedef struct{
__REG32 ASSTMR          :12;
__REG32                 :20;
} __uasstr_bits;

/* USB Configure Flag Register (UCFGR) */
typedef struct{
__REG32 CF              : 1;
__REG32                 :31;
} __ucfgr_bits;

/* USB Port 0 Status and Control Register (UPSCR0) */
/* USB Port 1 Status and Control Register (UPSCR1) */
typedef struct {
__REG32 CSTS            : 1;
__REG32 CSCHG           : 1;
__REG32 PEN             : 1;
__REG32 PENCHG          : 1;
__REG32 OCACT           : 1;
__REG32 OCCHG           : 1;
__REG32 FPRESUM         : 1;
__REG32 SUSPEND         : 1;
__REG32 PRST            : 1;
__REG32                 : 1;
__REG32 LSTATUS         : 2;
__REG32 PP              : 1;
__REG32 PO              : 1;
__REG32                 :18;
} __upscrx_bits;

/*USB PHY 0 Control Register (USBPCR0) */
typedef struct {
__REG32                 : 2;
__REG32 SIDDQ           : 1;
__REG32 XO_ON           : 1;
__REG32 CLK_SEL         : 2;
__REG32 REFCLK          : 1;
__REG32 CLK48           : 1;
__REG32 SUSPEND         : 1;
__REG32                 : 2;
__REG32 CLKVALID        : 1;
__REG32                 :20;
} __usbpcr0_bits;

/*USB PHY 1 Control Register (USBPCR1)*/
typedef struct {
__REG32                 : 2;
__REG32 SIDDQ           : 1;
__REG32 XO_ON           : 1;
__REG32 CLK_SEL         : 2;
__REG32 REFCLK          : 1;
__REG32 CLK48           : 1;
__REG32 SUSPEND         : 1;
__REG32                 : 2;
__REG32 XO_SEL          : 1;
__REG32                 :20;
} __usbpcr1_bits;

/* HcRevision Register */
typedef struct {
__REG32 REV             : 8;
__REG32                 :24;
} __HcRevision_bits;

/* HcControl Register */
typedef struct {
__REG32 CBSR            : 2;
__REG32 PLE             : 1;
__REG32 IE              : 1;
__REG32 CLE             : 1;
__REG32 BLE             : 1;
__REG32 HCFS            : 2;
__REG32 IR              : 1;
__REG32 WC              : 1;
__REG32 RWE             : 1;
__REG32                 :21;
} __HcControl_bits;

/* HcCommandStatus Register */
typedef struct {
__REG32 HCR             : 1;
__REG32 CLF             : 1;
__REG32 BLF             : 1;
__REG32 OCR             : 1;
__REG32                 :12;
__REG32 SOC             : 2;
__REG32                 :14;
} __HcCommandStatus_bits;

/* HcInterruptStatus Register */
typedef struct {
__REG32 SO              : 1;
__REG32 WDH             : 1;
__REG32 SF              : 1;
__REG32 RD              : 1;
__REG32 UE              : 1;
__REG32 FNO             : 1;
__REG32 RHSC            : 1;
__REG32                 :23;
__REG32 OC              : 1;
__REG32                 : 1;
} __HcInterruptStatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct {
__REG32 SO              : 1;
__REG32 WDH             : 1;
__REG32 SF              : 1;
__REG32 RD              : 1;
__REG32 UE              : 1;
__REG32 FNO             : 1;
__REG32 RHSC            : 1;
__REG32                 :23;
__REG32 OC              : 1;
__REG32 MIE             : 1;
} __HcInterruptEnable_bits;

/* HcHCCA Register */
typedef struct {
__REG32                 : 8;
__REG32 HCCA            :24;
} __HcHCCA_bits;

/* HcPeriodCurrentED Register */
typedef struct {
__REG32                 : 4;
__REG32 PCED            :28;
} __HcPeriodCurrentED_bits;

/* HcControlHeadED Registerr */
typedef struct {
__REG32                 : 4;
__REG32 CHED            :28;
} __HcControlHeadED_bits;

/* HcControlCurrentED Register */
typedef struct {
__REG32                 : 4;
__REG32 CCED            :28;
} __HcControlCurrentED_bits;

/* HcBulkHeadED Register */
typedef struct {
__REG32                 : 4;
__REG32 BHED            :28;
} __HcBulkHeadED_bits;

/* HcBulkCurrentED Register */
typedef struct {
__REG32                 : 4;
__REG32 BCED            :28;
} __HcBulkCurrentED_bits;

/* HcDoneHead Register */
typedef struct {
__REG32                 : 4;
__REG32 DH              :28;
} __HcDoneHead_bits;

/* HcFmInterval Register */
typedef struct {
__REG32 FI              :14;
__REG32                 : 2;
__REG32 FSMPS           :15;
__REG32 FIT             : 1;
} __HcFmInterval_bits;

/* HcFmRemaining Register */
typedef struct {
__REG32 FR              :14;
__REG32                 :17;
__REG32 FRT             : 1;
} __HcFmRemaining_bits;

/* HcPeriodicStart Register */
typedef struct {
__REG32 PS              :14;
__REG32                 :18;
} __HcPeriodicStart_bits;

/* HcLSThreshold Register */
typedef struct {
__REG32 LST             :12;
__REG32                 :20;
} __HcLSThreshold_bits;

/* HcRhDescriptorA Register */
typedef struct {
__REG32 NDP             : 8;
__REG32 PSM             : 1;
__REG32 NPS             : 1;
__REG32 DT              : 1;
__REG32 OCPM            : 1;
__REG32 NOCP            : 1;
__REG32                 :11;
__REG32 POTPGT          : 8;
} __HcRhDescriptorA_bits;

/* HcRhDescriptorB Register */
typedef struct {
__REG32 DEVRM           :16;
__REG32 PPCM            :16;
} __HcRhDescriptorB_bits;

/* HcRhStatus Register */
typedef struct {
__REG32 LPS             : 1;
__REG32 OCI             : 1;
__REG32                 :13;
__REG32 DRWE            : 1;
__REG32 LPSC            : 1;
__REG32 CCIC            : 1;
__REG32                 :13;
__REG32 CRWE            : 1;
} __HcRhStatus_bits;

/* HcRhPortStatus Register */
typedef struct {
__REG32 CCS             : 1;
__REG32 PES             : 1;
__REG32 PSS             : 1;
__REG32 POCI            : 1;
__REG32 PRS             : 1;
__REG32                 : 3;
__REG32 PPS             : 1;
__REG32 LSDA            : 1;
__REG32                 : 6;
__REG32 CSC             : 1;
__REG32 PESC            : 1;
__REG32 PSSC            : 1;
__REG32 OCIC            : 1;
__REG32 PRSC            : 1;
__REG32                 :11;
} __HcRhPortStatus_bits;

/* USB Operational Mode Enable Register */
typedef struct {
__REG32 DBREG           : 1;
__REG32 ABORT           : 1;
__REG32                 : 1;
__REG32 OVRCUR          : 1;
__REG32                 : 4;
__REG32 SIEPD           : 1;
__REG32                 :23;
} __operationalmodeenable_bits;

/* Interrupt Register (IRQ) */
typedef struct {
__REG32 USB_INT         : 1;
__REG32 CEP_INT         : 1;
__REG32 EPA_INT         : 1;
__REG32 EPB_INT         : 1;
__REG32 EPC_INT         : 1;
__REG32 EPD_INT         : 1;
__REG32 EPE_INT         : 1;
__REG32 EPF_INT         : 1;
__REG32                 :24;
} __irq_stat_bits;

/* Interrupt Enable Low Register (IRQ_ENB_L) */
typedef struct {
__REG32 USB_IE          : 1;
__REG32 CEP_IE          : 1;
__REG32 EPA_IE          : 1;
__REG32 EPB_IE          : 1;
__REG32 EPC_IE          : 1;
__REG32 EPD_IE          : 1;
__REG32 EPE_IE          : 1;
__REG32 EPF_IE          : 1;
__REG32                 :24;
} __irq_enb_l_bits;

/* USB Interrupt Status Register (USB_IRQ_STAT) */
typedef struct {
__REG32 SOF_IS          : 1;
__REG32 RST_IS          : 1;
__REG32 RUM_IS          : 1;
__REG32 SUS_IS          : 1;
__REG32 HISPD_IS        : 1;
__REG32 DMACOM_IS       : 1;
__REG32 TCLKOK_IS       : 1;
__REG32                 :25;
} __usb_irq_stat_bits;

/* USB Interrupt Enable Register (USB_IRQ_ENB) */
typedef struct {
__REG32 SOF_IE          : 1;
__REG32 RST_IE          : 1;
__REG32 RUM_IE          : 1;
__REG32 SUS_IE          : 1;
__REG32 HISPD_IE        : 1;
__REG32 DMACOM_IE       : 1;
__REG32 TCLKOK_IE       : 1;
__REG32                 :25;
} __usb_irq_enb_bits;

/* USB Operational Register (USB_OPER) */
typedef struct {
__REG32 GEN_RUM         : 1;
__REG32 SET_HISPD       : 1;
__REG32 CUR_SPD         : 1;
__REG32                 :29;
} __usb_oper_bits;

/*USB Frame Count Register (USB_FRAME_CNT)*/
typedef struct {
__REG32 MFRAME_CNT      : 3;
__REG32 FRAME_CNT       :11;
__REG32                 :18;
} __usb_frame_cnt_bits;

/*USB Address Register (USB_ADDR)*/
typedef struct {
__REG32 ADDR            : 7;
__REG32                 :25;
} __usb_addr_bits;

/*Control-ep Data Buffer (CEP_DATA_BUF)*/
typedef struct {
__REG32 DATA_BUF        :16;
__REG32                 :16;
} __cep_data_buf_bits;

/* Control-ep Control and Status (CEP_CTRL_STAT) */
typedef struct {
__REG32 NAK_CLEAR       : 1;
__REG32 STLALL          : 1;
__REG32 ZEROLEN         : 1;
__REG32 FLUSH           : 1;
__REG32                 :28;
} __cep_ctrl_stat_bits;

/*Control Endpoint Interrupt Enable (CEP_IRQ_ENABLE) */
typedef struct {
__REG32 SETUP_TK_IE     : 1;
__REG32 SETUP_PK_IE     : 1;
__REG32 OUT_TK_IE       : 1;
__REG32 IN_TK_IE        : 1;
__REG32 PING_IE         : 1;
__REG32 DATA_TXED_IE    : 1;
__REG32 DATA_RXED_IE    : 1;
__REG32 NAK_IE          : 1;
__REG32 STALL_IE        : 1;
__REG32 ERR_IE          : 1;
__REG32 STACOM_IE       : 1;
__REG32 FULL_IE         : 1;
__REG32 EMPTY_IE        : 1;
__REG32                 :19;
} __cep_irq_enb_bits;

/*Control-Endpoint Interrupt Status (CEP_IRQ_STAT) */
typedef struct {
__REG32 SETUP_TK_IS     : 1;
__REG32 SETUP_PK_IS     : 1;
__REG32 OUT_TK_IS       : 1;
__REG32 IN_TK_IS        : 1;
__REG32 PING_IS         : 1;
__REG32 DATA_TXED_IS    : 1;
__REG32 DATA_RXED_IS    : 1;
__REG32 NAK_IS          : 1;
__REG32 STALL_IS        : 1;
__REG32 ERR_IS          : 1;
__REG32 STACOM_IS       : 1;
__REG32 FULL_IS         : 1;
__REG32 EMPTY_IS        : 1;
__REG32                 :19;
} __cep_irq_stat_bits;

/*In-transfer data count (IN_TRF_CNT) */
typedef struct {
__REG32 IN_TRF_CNT      : 8;
__REG32                 :24;
} __in_trnsfr_cnt_bits;

/*Out-transfer data count (OUT_TRF_CNT)*/
typedef struct {
__REG32 OUT_TRF_CNT     :16;
__REG32                 :16;
} __out_trnsfr_cnt_bits;

/*Control- Endpoint data count (CEP_CNT)*/
typedef struct {
__REG32 CEP_CNT         :16;
__REG32                 :16;
} __cep_cnt_bits;

/*Setup1 & Setup0 bytes (SETUP1_0)*/
typedef struct {
__REG32 SETUP0          : 8;
__REG32 SETUP1          : 8;
__REG32                 :16;
} __setup1_0_bits;

/*Setup3 & Setup2 bytes (SETUP3_2)*/
typedef struct {
__REG32 SETUP2          : 8;
__REG32 SETUP3          : 8;
__REG32                 :16;
} __setup3_2_bits;

/*Setup5 & Setup4 bytes (SETUP5_4)*/
typedef struct {
__REG32 SETUP4          : 8;
__REG32 SETUP5          : 8;
__REG32                 :16;
} __setup5_4_bits;

/*Setup7 & Setup6 bytes (SETUP7_6)*/
typedef struct {
__REG32 SETUP6          : 8;
__REG32 SETUP7          : 8;
__REG32                 :16;
} __setup7_6_bits;

/*Control Endpoint RAM Start Addr Register (CEP_START_ADDR)*/
typedef struct {
__REG32 CEP_START_ADDR  :11;
__REG32                 :21;
} __cep_start_addr_bits;

/*Control Endpoint RAM End Addr Register (CEP_END_ADDR)*/
typedef struct {
__REG32 CEP_END_ADDR    :11;
__REG32                 :21;
} __cep_end_addr_bits;

/*DMA Control Status Register (DMA_CTRL_STS)*/
typedef struct {
__REG32 DMA_ADDR        : 4;
__REG32 DMA_RD          : 1;
__REG32 DMA_EN          : 1;
__REG32 SCAT_GA_EN      : 1;
__REG32 RST_DMA         : 1;
__REG32                 :24;
} __dma_ctrl_sts_bits;

/*DMA Count Register (DMA_CNT)*/
typedef struct {
__REG32 DMA_CNT         :20;
__REG32                 :12;
} __dma_cnt_bits;

/*Endpoint A~F Data Register (EPA_DATA_BUF~ EPF_DATA_BUF)*/
typedef struct {
__REG32 EP_DATA_BUF     :16;
__REG32                 :16;
} __ep_data_buf_bits;

/*Endpoint A~F Interrupt Status Register (EPA_IRQ_STAT~ EPF_IRQ_STAT)*/
typedef struct {
__REG32 FULL_IS         : 1;
__REG32 EMPTY_IS        : 1;
__REG32 SHORT_PKT_IS    : 1;
__REG32 DATA_TXED_IS    : 1;
__REG32 DATA_RXED_IS    : 1;
__REG32 OUT_TK_IS       : 1;
__REG32 IN_TK_IS        : 1;
__REG32 PING_IS         : 1;
__REG32 NAK_IS          : 1;
__REG32 STALL_IS        : 1;
__REG32 NYET_IS         : 1;
__REG32 ERR_IS          : 1;
__REG32 O_SHORT_PKT_IS  : 1;
__REG32                 :19;
} __ep_irq_stat_bits;

/*Endpoint A~F Interrupt Enable Register (EPA_IRQ_ENB~ EPF_IRQ_ENB)*/
typedef struct {
__REG32 FULL_IE         : 1;
__REG32 EMPTY_IE        : 1;
__REG32 SHORT_PKT_IE    : 1;
__REG32 DATA_TXED_IE    : 1;
__REG32 DATA_RXED_IE    : 1;
__REG32 OUT_TK_IE       : 1;
__REG32 IN_TK_IE        : 1;
__REG32 PING_IE         : 1;
__REG32 NAK_IE          : 1;
__REG32 STALL_IE        : 1;
__REG32 NYET_IE         : 1;
__REG32 ERR_IE          : 1;
__REG32 O_SHORT_PKT_IE  : 1;
__REG32                 :19;
} __ep_irq_enb_bits;

/*Endpoint A~F Data Available count register (EPA_DATA_CNT~ EPF_DATA_CNT)*/
typedef struct {
__REG32 DATA_CNT        :16;
__REG32 DMA_LOOP        :16;
} __ep_data_cnt_bits;

/*Endpoint A~F Response Set/Clear Register (EPA_RSP_SC~ EPF_RSP_SC)*/
typedef struct {
__REG32 BUF_FLUSH       : 1;
__REG32 MODE            : 2;
__REG32 TOGGLE          : 1;
__REG32 HALT            : 1;
__REG32 ZEROLEN         : 1;
__REG32 PK_END          : 1;
__REG32 DIS_BUF         : 1;
__REG32                 :24;
} __ep_rsp_sc_bits;

/*Endpoint A~F Maximum Packet Size Register (EPA_MPS~ EPF_MPS)*/
typedef struct {
__REG32 EP_MPS          :11;
__REG32                 :21;
} __ep_mps_bits;

/*Endpoint A~F Transfer Count Register (EPA_TRF_CNT~ EPF_TRF_CNT)*/
typedef struct {
__REG32 EP_TRF_CNT      :11;
__REG32                 :21;
} __ep_cnt_bits;

/*Endpoint A~F Configuration Register (EPA_CFG~ EPF_CFG)*/
typedef struct {
__REG32 EP_VALID        : 1;
__REG32 EP_TYPE         : 2;
__REG32 EP_DIR          : 1;
__REG32 EP_NUM          : 4;
__REG32 EP_MULT         : 2;
__REG32                 :22;
} __ep_cfg_bits;

/*Endpoint A~F RAM Start Address Register (EPA_START_ADDR~ EPF_START_ADDR)*/
typedef struct {
__REG32 EP_START_ADDR   :11;
__REG32                 :21;
} __ep_start_addr_bits;

/*Endpoint A~F RAM End Address Register (EPA_END_ADDR~ EPF_END_ADDR)*/
typedef struct {
__REG32 EP_END_ADDR     :11;
__REG32                 :21;
} __ep_end_addr_bits;

/*Endpoint A~F RAM End Address Register (EPA_END_ADDR~ EPF_END_ADDR)*/
typedef struct {
__REG32                 : 9;
__REG32 PHY_SUSPEND     : 1;
__REG32                 :22;
} __usb_phy_ctl_bits;

/*DMAC Control and Status Register (DMACCSR)*/
typedef struct {
__REG32 DMACEN          : 1;
__REG32 SW_RST          : 1;
__REG32 SG_EN1          : 1;
__REG32 SG_EN2          : 1;
__REG32                 : 4;
__REG32 ATA_BUSY        : 1;
__REG32 FMI_BUSY        : 1;
__REG32                 :22;
} __dmaccsr_bits;

/*DMAC Transfer Byte Count Register (DMACBCR)*/
typedef struct {
__REG32 BCNT            :26;
__REG32                 : 6;
} __dmacbcr_bits;

/*DMAC Interrupt Enable Register (DMACIER)*/
typedef struct {
__REG32 TABORT_IE       : 1;
__REG32 WEOT_IE         : 1;
__REG32                 :30;
} __dmacier_bits;

/*DMAC Interrupt Status Register (DMACISR)*/
typedef struct {
__REG32 TABORT_IF       : 1;
__REG32 WEOT_IF         : 1;
__REG32                 :30;
} __dmacisr_bits;

/*Global Control and Status Register (FMICSR)*/
typedef struct {
__REG32 SW_RST          : 1;
__REG32 SD_EN           : 1;
__REG32 MS_EN           : 1;
__REG32 SM_EN           : 1;
__REG32                 :28;
} __fmicsr_bits;

/*Global Interrupt Control Register (FMIIER)*/
typedef struct {
__REG32 DTA_IE          : 1;
__REG32                 :31;
} __fmiier_bits;

/*Global Interrupt Status Register (FMIISR)*/
typedef struct {
__REG32 DTA_IF          : 1;
__REG32                 :31;
} __fmiisr_bits;

/*SD Control and Status Register (SDCSR)*/
typedef struct {
__REG32 CO_EN           : 1;
__REG32 RI_EN           : 1;
__REG32 DI_EN           : 1;
__REG32 DO_EN           : 1;
__REG32 R2_EN           : 1;
__REG32 CLK74_OE        : 1;
__REG32 CLK8_OE         : 1;
__REG32 CLK_KEEP0       : 1;
__REG32 CMD_CODE        : 6;
__REG32 SW_RST          : 1;
__REG32 DBW             : 1;
__REG32 BLK_CNT         : 8;
__REG32 SDNWR           : 4;
__REG32                 : 1;
__REG32 SDPORT          : 2;
__REG32 CLK_KEEP1       : 1;
} __sdcsr_bits;

/*SD Interrupt Control Register (SDIER)*/
typedef struct {
__REG32 BLKD_IE         : 1;
__REG32 CRC_IE          : 1;
__REG32                 : 6;
__REG32 CD0_IE          : 1;
__REG32 CD1_IE          : 1;
__REG32 SDIO0_IE        : 1;
__REG32 SDIO1_IE        : 1;
__REG32 RITO_IE         : 1;
__REG32 DITO_IE         : 1;
__REG32 WKUP_EN         : 1;
__REG32                 :15;
__REG32 CD0SRC          : 1;
__REG32 CD1SRC          : 1;
} __sdier_bits;

/*SD Interrupt Status Register (SDISR)*/
typedef struct {
__REG32 BLKD_IF         : 1;
__REG32 CRC_IF          : 1;
__REG32 CRC_7           : 1;
__REG32 CRC_16          : 1;
__REG32 CRCSTAT         : 3;
__REG32 SDDAT0          : 1;
__REG32 CD0_IF          : 1;
__REG32 CD1_IF          : 1;
__REG32 SDIO0_IF        : 1;
__REG32 SDIO1_IF        : 1;
__REG32 RITO_IF         : 1;
__REG32 DITO_IF         : 1;
__REG32                 : 2;
__REG32 CDPS0           : 1;
__REG32 CDPS1           : 1;
__REG32 SD0DAT1         : 1;
__REG32 SD1DAT1         : 1;
__REG32                 :12;
} __sdisr_bits;

/*SD Receiving Response Token Register 0 (SDRSP0)*/
typedef struct {
__REG32 SD_RSP_TK0      :32;
} __sdrsp0_bits;

/*SD Receiving Response Token Register 1 (SDRSP1)*/
typedef struct {
__REG32 SD_RSP_TK1      : 8;
__REG32                 :24;
} __sdrsp1_bits;

/*SD Block Length Register (SDBLEN)*/
typedef struct {
__REG32 SDBLEN          : 9;
__REG32                 :23;
} __sdblen_bits;

/*SD Response/Data-in Time-out Register (SDTMOUT)*/
typedef struct {
__REG32 SDTMOUT         :24;
__REG32                 : 8;
} __sdtmout_bits;

/* Memory Stick Control and Status Register (MSCSR) */
typedef struct {
__REG32 SW_RST          : 1;
__REG32 MS_GO           : 1;
__REG32 MSPRO           : 1;
__REG32 SERIAL          : 1;
__REG32                 : 4;
__REG32 TPC             : 4;
__REG32                 : 4;
__REG32 DCNT            : 3;
__REG32 DSIZE           : 2;
__REG32 MSPORT          : 1;
__REG32                 :10;
} __mscsr_bits;

/* Memory Stick Interrupt Control Register (MSIER) */
typedef struct {
__REG32 PKT_IE          : 1;
__REG32 MSINT_IE        : 1;
__REG32 INTTO_IE        : 1;
__REG32 BSYTO_IE        : 1;
__REG32 CRC_IE          : 1;
__REG32                 :11;
__REG32 CD0_IE          : 1;
__REG32 CD1_IE          : 1;
__REG32                 :14;
} __msier_bits;

/* Memory Stick Interrupt Status Register (MSISR) */
typedef struct {
__REG32 PKT_IF          : 1;
__REG32 MSINT_IF        : 1;
__REG32 INTTO_IF        : 1;
__REG32 BSYTO_IF        : 1;
__REG32 CRC_IF          : 1;
__REG32                 : 3;
__REG32 CED             : 1;
__REG32 ERR             : 1;
__REG32 BREQ            : 1;
__REG32 CMDNK           : 1;
__REG32                 : 4;
__REG32 CD0_IF          : 1;
__REG32 CD1_IF          : 1;
__REG32                 : 6;
__REG32 CD0_            : 1;
__REG32 CD1_            : 1;
__REG32                 : 6;
} __msisr_bits;

/* NAND Flash Control and Status Register (SMCSR) */
typedef struct {
__REG32 SW_RST          : 1;
__REG32 DRD_EN          : 1;
__REG32 DWR_EN          : 1;
__REG32 PSIZE           : 1;
__REG32 DBW             : 1;
__REG32 ECC4_EN         : 1;
__REG32                 : 1;
__REG32 ECC4CHK         : 1;
__REG32 MECC4           : 4;
__REG32                 :12;
__REG32 WP_             : 1;
__REG32 SM_CS           : 2;
__REG32                 : 5;
} __smcsr_bits;

/* NAND Flash Timing Control Register (SMTCR) */
typedef struct {
__REG32 LO_WID          : 8;
__REG32 HI_WID          : 8;
__REG32 CALE_SH         : 7;
__REG32                 : 9;
} __smtcr_bits;

/* NAND Flash Interrupt Control Register (SMIER) */
typedef struct {
__REG32 DMA_IE          : 1;
__REG32 ECC_IE          : 1;
__REG32                 : 6;
__REG32 CD0_IE          : 1;
__REG32 CD1_IE          : 1;
__REG32 RB_IE           : 1;
__REG32                 :21;
} __smier_bits;

/* NAND Flash Interrupt Status Register (SMISR) */
typedef struct {
__REG32 DMA_IF          : 1;
__REG32 ECC_IF          : 1;
__REG32                 : 6;
__REG32 CD0_IF          : 1;
__REG32 CD1_IF          : 1;
__REG32 RB_IF           : 1;
__REG32                 : 5;
__REG32 CD0_            : 1;
__REG32 CD1_            : 1;
__REG32 RB_             : 1;
__REG32                 :13;
} __smisr_bits;

/* NAND Flash Command Port Register (SMCMD) */
typedef struct {
__REG32 SMCMD           : 8;
__REG32                 :24;
} __smcmd_bits;

/* NAND Flash Address Port Register (SMADDR) */
typedef struct {
__REG32 SMADDR          : 8;
__REG32                 :24;
} __smaddr_bits;

/* NAND Flash Data Port Register (SMDATA) */
typedef struct {
__REG32 SMDATA          : 8;
__REG32                 :24;
} __smdata_bits;

/* NAND Flash Error Correction Code 0 Register (SMECC0) */
typedef struct {
__REG32 SMECC0          :24;
__REG32                 : 8;
} __smecc0_bits;

/* NAND Flash Error Correction Code 1 Register (SMECC1) */
typedef struct {
__REG32 SMECC1          :24;
__REG32                 : 8;
} __smecc1_bits;

/* NAND Flash Error Correction Code 2 Register (SMECC2) */
typedef struct {
__REG32 SMECC2          :24;
__REG32                 : 8;
} __smecc2_bits;

/* NAND Flash Error Correction Code 3 Register (SMECC3) */
typedef struct {
__REG32 SMECC3          :24;
__REG32                 : 8;
} __smecc3_bits;

/* NAND Flash ECC Correction Address 0 (SMECCAD0) */
typedef struct {
__REG32 F1_ADDR         :12;
__REG32                 : 2;
__REG32 F1_STAT         : 2;
__REG32 F2_ADDR         :12;
__REG32                 : 2;
__REG32 F2_STAT         : 2;
} __smeccad0_bits;

/* NAND Flash ECC Correction Address 1 (SMECCAD1) */
typedef struct {
__REG32 F3_ADDR         :12;
__REG32                 : 2;
__REG32 F3_STAT         : 2;
__REG32 F4_ADDR         :12;
__REG32                 : 2;
__REG32 F4_STAT         : 2;
} __smeccad1_bits;

/* ECC4 Correction Status (ECC4ST) */
typedef struct {
__REG32 F1_STAT         : 2;
__REG32 F1_ECNT         : 3;
__REG32                 : 3;
__REG32 F2_STAT         : 2;
__REG32 F2_ECNT         : 3;
__REG32                 : 3;
__REG32 F3_STAT         : 2;
__REG32 F3_ECNT         : 3;
__REG32                 : 3;
__REG32 F4_STAT         : 2;
__REG32 F4_ECNT         : 3;
__REG32                 : 3;
} __ecc4st_bits;

/* ECC4 Field 1 Error Address 1 (ECC4F1A1) */
typedef struct {
__REG32 F1_ADDR1        : 9;
__REG32                 : 7;
__REG32 F1_ADDR2        : 9;
__REG32                 : 7;
} __ecc4f1a1_bits;

/* ECC4 Field 1 Error Address 2 (ECC4F1A2) */
typedef struct {
__REG32 F1_ADDR3        : 9;
__REG32                 : 7;
__REG32 F1_ADDR4        : 9;
__REG32                 : 7;
} __ecc4f1a2_bits;

/* ECC4 Field 1 Error Data (ECC4F1D) */
typedef struct {
__REG32 F1_DATA1        : 8;
__REG32 F1_DATA2        : 8;
__REG32 F1_DATA3        : 8;
__REG32 F1_DATA4        : 8;
} __ecc4f1d_bits;

/* ECC4 Field 2 Error Address 1 (ECC4F2A1) */
typedef struct {
__REG32 F2_ADDR1        : 9;
__REG32                 : 7;
__REG32 F2_ADDR2        : 9;
__REG32                 : 7;
} __ecc4f2a1_bits;

/* ECC4 Field 2 Error Address 2 (ECC4F2A2) */
typedef struct {
__REG32 F2_ADDR3        : 9;
__REG32                 : 7;
__REG32 F2_ADDR4        : 9;
__REG32                 : 7;
} __ecc4f2a2_bits;

/* ECC4 Field 2 Error Data (ECC4F2D) */
typedef struct {
__REG32 F2_DATA1        : 8;
__REG32 F2_DATA2        : 8;
__REG32 F2_DATA3        : 8;
__REG32 F2_DATA4        : 8;
} __ecc4f2d_bits;

/* ECC4 Field 3 Error Address 1 (ECC4F2A1) */
typedef struct {
__REG32 F3_ADDR1        : 9;
__REG32                 : 7;
__REG32 F3_ADDR2        : 9;
__REG32                 : 7;
} __ecc4f3a1_bits;

/* ECC4 Field 3 Error Address 2 (ECC4F2A2) */
typedef struct {
__REG32 F3_ADDR3        : 9;
__REG32                 : 7;
__REG32 F3_ADDR4        : 9;
__REG32                 : 7;
} __ecc4f3a2_bits;

/* ECC4 Field 3 Error Data (ECC4F2D) */
typedef struct {
__REG32 F3_DATA1        : 8;
__REG32 F3_DATA2        : 8;
__REG32 F3_DATA3        : 8;
__REG32 F3_DATA4        : 8;
} __ecc4f3d_bits;

/* ECC4 Field 4 Error Address 1 (ECC4F2A1) */
typedef struct {
__REG32 F4_ADDR1        : 9;
__REG32                 : 7;
__REG32 F4_ADDR2        : 9;
__REG32                 : 7;
} __ecc4f4a1_bits;

/* ECC4 Field 4 Error Address 2 (ECC4F2A2) */
typedef struct {
__REG32 F4_ADDR3        : 9;
__REG32                 : 7;
__REG32 F4_ADDR4        : 9;
__REG32                 : 7;
} __ecc4f4a2_bits;

/* ECC4 Field 4 Error Data (ECC4F2D) */
typedef struct {
__REG32 F4_DATA1        : 8;
__REG32 F4_DATA2        : 8;
__REG32 F4_DATA3        : 8;
__REG32 F4_DATA4        : 8;
} __ecc4f4d_bits;

/* Display Controller Control/Status Register (DCCS) */
typedef struct {
__REG32 ENG_RST         : 1;
__REG32 VA_EN           : 1;
__REG32 OSD_EN          : 1;
__REG32 DISP_OUT_EN     : 1;
__REG32 DISP_INT_EN     : 1;
__REG32 CMD_ON          : 1;
__REG32 FIELD_INTR      : 1;
__REG32 SINGLE          : 1;
__REG32 VA_SRC          : 3;
__REG32                 : 1;
__REG32 OSD_SRC         : 3;
__REG32 ITU_EN          : 1;
__REG32 OSD_VUP         : 2;
__REG32 OSD_HUP         : 2;
__REG32                 : 5;
__REG32 DISP_ON         : 1;
__REG32 VACT            : 1;
__REG32 HACT            : 1;
__REG32 VSYNC           : 1;
__REG32 LACE_F          : 1;
__REG32                 : 2;
} __dccs_bits;

/* Display Device Control Register (DEVICE_CTRL) */
typedef struct {
__REG32                 : 1;
__REG32 SWAP_YCBCR      : 2;
__REG32 RGB_SHIFT       : 2;
__REG32 DEVICE          : 3;
__REG32 LCD_DDA         : 8;
__REG32 YUV2CCIR        : 1;
__REG32 SEL_ODD         : 1;
__REG32 LCD_ODD         : 1;
__REG32 FAL_D           : 1;
__REG32 H_POL           : 1;
__REG32 V_POL           : 1;
__REG32 VR_LACE         : 1;
__REG32 LACE            : 1;
__REG32 RGB_SCALE       : 2;
__REG32 DBWORD          : 1;
__REG32 MCU68           : 1;
__REG32 DE_POL          : 1;
__REG32 CMD16           : 1;
__REG32 CM16T18         : 1;
__REG32 CMD_LOW         : 1;
} __device_ctrl_bits;

/* MPU-Interfaced LCD Write Command Register (MPULCD_CMD) */
typedef struct {
__REG32 MPULCD_CMD      :18;
__REG32                 :11;
__REG32 READ            : 1;
__REG32 WR_RS           : 1;
__REG32 CMD_BUSY        : 1;
} __mpulcd_cmd_bits;

/* Interrupt Control/Status Register (INT_CS) */
typedef struct {
__REG32 DISP_F_EN       : 1;
__REG32 UNDERRUN_EN     : 1;
__REG32                 :26;
__REG32 BUS_ERROR_INT   : 1;
__REG32 UNDERRUN_INT    : 1;
__REG32 DISP_F_STATUS   : 1;
__REG32 DISP_F_INT      : 1;
} __int_cs_bits;

/* CRTC Display Size Register (CRTC_SIZE) */
typedef struct {
__REG32 HTT             :11;
__REG32                 : 5;
__REG32 VTT             :11;
__REG32                 : 5;
} __crtc_size_bits;

/* CRTC Display Enable End Register (CRTC_DEND) */
typedef struct {
__REG32 HDEND           :11;
__REG32                 : 5;
__REG32 VDEND           :11;
__REG32                 : 5;
} __crtc_dend_bits;

/* CRTC Internal Horizontal Retrace Timing Register (CRTC_HR) */
typedef struct {
__REG32 HRS             :11;
__REG32                 : 5;
__REG32 HRE             :11;
__REG32                 : 5;
} __crtc_hr_bits;

/* CRTC Horizontal Sync Timing Register (CRTC_HSYNC) */
typedef struct {
__REG32 HSYNC_S         :11;
__REG32                 : 5;
__REG32 HSYNC_E         :11;
__REG32                 : 3;
__REG32 HSYNC_SHIFT     : 2;
} __crtc_hsync_bits;

/* CRTC Internal Vertical Retrace Timing Register (CRTC_VR) */
typedef struct {
__REG32 VRS             :11;
__REG32                 : 5;
__REG32 VRE             :11;
__REG32                 : 5;
} __crtc_vr_bits;

/* Image Stream Frame Buffer Control Register (VA_FBCTRL) */
typedef struct {
__REG32 VA_STRIDE       :11;
__REG32                 : 5;
__REG32 VA_FF           :11;
__REG32                 : 1;
__REG32 IO_REGION_HALF  : 1;
__REG32 FIELD_DUAL      : 1;
__REG32 START_BUF       : 1;
__REG32 DB_EN           : 1;
} __va_fbctrl_bits;

/* Image Stream Scaling Control Register (VA_SCALE) */
typedef struct {
__REG32 VA_SCALE_H      :13;
__REG32                 : 2;
__REG32 XCOPY           : 1;
__REG32 VA_SCALE_V      :13;
__REG32                 : 3;
} __va_scale_bits;

/* Image Stream Active Window Coordinates (VA_WIN) */
typedef struct {
__REG32 VA_WYE          :11;
__REG32                 : 5;
__REG32 VA_WYS          :11;
__REG32                 : 5;
} __va_win_bits;

/* Image Stream Active Window Coordinates (VA_WIN) */
typedef struct {
__REG32 VA_STUFF        :24;
__REG32                 : 8;
} __va_stuff_bits;

/* OSD Window Starting Coordinates Register (OSD_WINS) */
typedef struct {
__REG32 OSD_WXS         :11;
__REG32                 : 5;
__REG32 OSD_WYS         :11;
__REG32                 : 5;
} __osd_wins_bits;

/* OSD Window Ending Coordinates Register (OSD_WINE) */
typedef struct {
__REG32 OSD_WXE         :11;
__REG32                 : 5;
__REG32 OSD_WYE         :11;
__REG32                 : 5;
} __osd_wine_bits;

/* OSD Stream Frame Buffer Control Register (OSD_FBCTRL) */
typedef struct {
__REG32 OSD_STRIDE      :11;
__REG32                 : 5;
__REG32 OSD_FF          :11;
__REG32                 : 5;
} __osd_fbctrl_bits;

/* OSD Overlay Control Register (OSD_OVERLAY) */
typedef struct {
__REG32 OCR0            : 2;
__REG32 OCR1            : 2;
__REG32 VA_SYNW         : 3;
__REG32                 : 1;
__REG32 CKEY_ON         : 1;
__REG32 BLI_ON          : 1;
__REG32                 : 6;
__REG32 BLINK_VCNT      : 8;
__REG32                 : 8;
} __osd_overlay_bits;

/* OSD Overlay Color-Key Pattern Register */
typedef struct {
__REG32 OSD_CKEY        :24;
__REG32                 : 8;
} __osd_ckey_bits;

/* OSD Overlay Color-Key Mask Register (OSD_CMASK) */
typedef struct {
__REG32 OSD_MASK        :24;
__REG32                 : 8;
} __osd_cmask_bits;

/* OSD Window Skip1 Register (OSD_SKIP1) */
typedef struct {
__REG32 OSD_SKIP1_YE    :11;
__REG32                 : 5;
__REG32 OSD_SKIP1_YS    :11;
__REG32                 : 5;
} __osd_skip1_bits;

/* OSD Window SKIP2 Register (OSD_SKIP2) */
typedef struct {
__REG32 OSD_SKIP2_YE    :11;
__REG32                 : 5;
__REG32 OSD_SKIP2_YS    :11;
__REG32                 : 5;
} __osd_skip2_bits;

/* OSD Scaling Control Register (OSD_SCALE) */
typedef struct {
__REG32 OSD_SCALE_H     :13;
__REG32                 :19;
} __osd_scale_bits;

/* MPU Vsync Control Register (MPU_VSYNC) */
typedef struct {
__REG32 MPU_V_EN        : 1;
__REG32 MPU_FMARK       : 1;
__REG32 MPU_VSYNC_POL   : 1;
__REG32 MPU_VSYNC_WIDTH : 4;
__REG32                 :25;
} __mpu_vsync_bits;

/* Hardware Cursor Control Register (HC_CTRL) */
typedef struct {
__REG32 HC_MODE         : 3;
__REG32                 : 5;
__REG32 HC_TIP_X        : 6;
__REG32                 : 2;
__REG32 HC_TIP_Y        : 6;
__REG32                 :10;
} __hc_ctrl_bits;

/* HC POSITION Register (HC_POS) */
typedef struct {
__REG32 HC_X            :11;
__REG32                 : 5;
__REG32 HC_Y            :11;
__REG32                 : 5;
} __hc_pos_bits;

/* Hardware Cursor Window Buffer Control Register (HC_WBCTRL) */
typedef struct {
__REG32 HC_STRIDE       :11;
__REG32                 : 5;
__REG32 HC_FF           :11;
__REG32                 : 5;
} __hc_wbctrl_bits;

/* HC Color RAM 0 Register (HC_COLOR0) */
typedef struct {
__REG32 HC_COLOR0_B     : 8;
__REG32 HC_COLOR0_G     : 8;
__REG32 HC_COLOR0_R     : 8;
__REG32                 : 8;
} __hc_color0_bits;

/* HC Color RAM 1 Register (HC_COLOR1) */
typedef struct {
__REG32 HC_COLOR1_B     : 8;
__REG32 HC_COLOR1_G     : 8;
__REG32 HC_COLOR1_R     : 8;
__REG32                 : 8;
} __hc_color1_bits;

/* HC Color RAM 2 Register (HC_COLOR2) */
typedef struct {
__REG32 HC_COLOR2_B     : 8;
__REG32 HC_COLOR2_G     : 8;
__REG32 HC_COLOR2_R     : 8;
__REG32                 : 8;
} __hc_color2_bits;

/* HC Color RAM 3 Register (HC_COLOR3) */
typedef struct {
__REG32 HC_COLOR3_B     : 8;
__REG32 HC_COLOR3_G     : 8;
__REG32 HC_COLOR3_R     : 8;
__REG32                 : 8;
} __hc_color3_bits;

/* Audio controller control registers (ACTL_CON) */
typedef struct {
  __REG32                   : 1;
  __REG32 BLOCK_EN0         : 1;
  __REG32 BLOCK_EN1         : 1;
  __REG32 IRQ_DMA_DATA_Z_EN : 1;
  __REG32 IRQ_DMA_CNTR_EN   : 1;
  __REG32                   : 2;
  __REG32 FIFO_TH           : 1;
  __REG32 I2S_AC_PIN_SEL    : 1;
  __REG32                   : 2;
  __REG32 T_DMA_IRQ         : 1;
  __REG32 R_DMA_IRQ         : 1;
  __REG32                   :19;
} __actl_con_bits;

/* Sub-block reset control register (ACTL_RESET) */
typedef struct {
  __REG32 I2S_RESET         : 1;
  __REG32 AC_RESET          : 1;
  __REG32                   : 1;
  __REG32 DMA_DATA_Z_EN     : 1;
  __REG32 DMA_CNTR_EN       : 1;
  __REG32 I2S_PLAY          : 1;
  __REG32 I2S_RECORD        : 1;
  __REG32 AC_PLAY           : 1;
  __REG32 AC_RECORD         : 1;
  __REG32                   : 3;
  __REG32 PLAY_SINGLE       : 2;
  __REG32 RECORD_SINGLE     : 2;
  __REG32 ACTL_RESET        : 1;
  __REG32                   :15;
} __actl_reset_bits;

/* Audio controller record status register (ACTL_RSR) */
typedef struct {
  __REG32 R_DMA_MIDDLE_IRQ  : 1;
  __REG32 R_DMA_END_IRQ     : 1;
  __REG32 R_FIFO_FULL       : 1;
  __REG32                   :29;
} __actl_rsr_bits;

/* Audio controller playback status register (ACTL_PSR) */
typedef struct {
  __REG32 P_DMA_MIDDLE_IRQ  : 1;
  __REG32 P_DMA_END_IRQ     : 1;
  __REG32 P_FIFO_EMPTY      : 1;
  __REG32 DMA_DATA_Z_IRQ    : 1;
  __REG32 DMA_CNTR_IRQ      : 1;
  __REG32                   :27;
} __actl_psr_bits;

/* I2S control register (ACTL_I2SCON) */
typedef struct {
  __REG32                   : 3;
  __REG32 FORMAT            : 1;
  __REG32 MCLK_SEL          : 1;
  __REG32 FS_SEL            : 1;
  __REG32 BCLK_SEL          : 2;
  __REG32                   : 8;
  __REG32 PRS               : 4;
  __REG32                   :12;
} __actl_i2scon_bits;

/* AC-link Control Register (ACTL_ACCON) */
typedef struct {
  __REG32                   : 1;
  __REG32 AC_C_RES          : 1;
  __REG32 AC_W_RES          : 1;
  __REG32 AC_W_FINISH       : 1;
  __REG32 AC_R_FINISH       : 1;
  __REG32 AC_BCLK_PU_EN     : 1;
  __REG32                   :26;
} __actl_accon_bits;

/* AC-link output slot 0 (ACTL_ACOS0) */
typedef struct {
  __REG32 SLOT_VALID        : 4;
  __REG32 VALID_FRAME       : 1;
  __REG32                   :27;
} __actl_acos0_bits;

/* AC-link output slot 1 (ACTL_ACOS1) */
typedef struct {
  __REG32 R_INDEX           : 7;
  __REG32 R_WB              : 1;
  __REG32                   :24;
} __actl_acos1_bits;

/* AC-link output slot 2 (ACTL_ACOS2) */
typedef struct {
  __REG32 WD                :16;
  __REG32                   :16;
} __actl_acos2_bits;

/* AC-link input slot 0 (ACTL_ACIS0) */
typedef struct {
  __REG32 SLOT_VALID        : 4;
  __REG32 CODEC_READY       : 1;
  __REG32                   :27;
} __actl_acis0_bits;

/* AC-link input slot 1 (ACTL_ACIS1) */
typedef struct {
  __REG32 SLOT_REQ          : 2;
  __REG32 R_INDEX           : 7;
  __REG32                   :23;
} __actl_acis1_bits;

/* AC-link input slot 2 (ACTL_ACIS2) */
typedef struct {
  __REG32 RD                :16;
  __REG32                   :16;
} __actl_acis2_bits;

/* Control and Status Register (CSR) */
typedef struct {
  __REG32 SW_RST            : 1;
  __REG32 RESETN            : 1;
  __REG32 ATA_EN            : 1;
  __REG32 HI_FREQ           : 1;
  __REG32                   :28;
} __csr_bits;

/* Interrupt Control and Status Register (INTR) */
typedef struct {
  __REG32 INTRQ_IE          : 1;
  __REG32 DMARQ_IE          : 1;
  __REG32 EOS_IE            : 1;
  __REG32 DTA_IE            : 1;
  __REG32                   : 4;
  __REG32 INTRQ_IF          : 1;
  __REG32 DMARQ_IF          : 1;
  __REG32 EOS_IF            : 1;
  __REG32 DTA_IF            : 1;
  __REG32                   :20;
} __intr_bits;

/* Status of ATAPI Input Pins (PINSTAT) */
typedef struct {
  __REG32 INTRQ             : 1;
  __REG32 DMARQ             : 1;
  __REG32 IORDY             : 1;
  __REG32                   :29;
} __pinstat_bits;

/* DMA Control and Status Register (DMACSR) */
typedef struct {
  __REG32 DMAEN             : 1;
  __REG32 UDMAEN            : 1;
  __REG32 EOSEN             : 1;
  __REG32 DMADIR            : 1;
  __REG32 DMASTOP           : 1;
  __REG32                   : 3;
  __REG32 DMATIP            : 1;
  __REG32 EOSS              : 1;
  __REG32                   :22;
} __dmacsr_bits;

/* Sector Count Register for DMA Transfer (SECCNT) */
typedef struct {
  __REG32 SECCNT            :16;
  __REG32                   :16;
} __seccnt_bits;

/* Register Transfer Timing Control Register (REGTTR) */
typedef struct {
  __REG32 REGTEOC           : 8;
  __REG32 REGT4             : 8;
  __REG32 REGT2             : 8;
  __REG32 REGT1             : 8;
} __regttr_bits;

/* PIO Transfer Timing Control Register (PIOTTR) */
typedef struct {
  __REG32 PIOTEOC           : 8;
  __REG32 PIOT4             : 8;
  __REG32 PIOT2             : 8;
  __REG32 PIOT1             : 8;
} __piottr_bits;

/* DMA Transfer Timing Control Register (DMATTR) */
typedef struct {
  __REG32 DMATEOC           : 8;
  __REG32 DMATD             : 8;
  __REG32 DMATM             : 8;
  __REG32                   : 8;
} __dmattr_bits;

/* UDMA Transfer Timing Control Register (UDMATTR) */
typedef struct {
  __REG32 UDMATRP           : 8;
  __REG32 UDMATDVS          : 8;
  __REG32 UDMATCYC          : 8;
  __REG32                   : 8;
} __udmattr_bits;

/* ATA Control Registers */
typedef struct {
  __REG32 DATA              :16;
  __REG32                   :16;
} __ata_data_bits;

/* Graphic Engine Trigger Control Register */
typedef struct {
  __REG32 GO                : 1;
  __REG32                   :31;
} ___2d_getg_bits;

/* Graphic Engine XY Mode Source Memory Origin Starting Address Register */
typedef struct {
  __REG32 XYOSADDR          :28;
  __REG32                   : 4;
} ___2d_gexysorg_bits;

/* Graphic Engine Tile Width/Height Numbers and DDA V/H Scale Up/Down Factors */
typedef union  {
  /*_2D_TileXY*/
  struct {
    __REG32 TW_X              : 8;
    __REG32 TH_Y              : 8;
    __REG32                   :16;
  };
  /*_2D_VHSF*/
  struct {
    __REG32 HSF_M             : 8;
    __REG32 HSF_N             : 8;
    __REG32 VSF_M             : 8;
    __REG32 VSF_N             : 8;
  };
} ___2d_tilexy_vhsf_bits;

/* Graphic Engine Rotate Reference Point XY Register */
typedef struct {
__REG32 RR_X              :11;
__REG32                   : 5;
__REG32 RR_Y              :11;
__REG32                   : 5;
} ___2d_gerrxy_bits;

/* Graphic Engine Interrupt Status Register */
typedef struct {
__REG32 INTS              : 1;
__REG32                   :31;
} ___2d_geints_bits;

/* Graphic Engine Pattern Location Starting Address Register */
typedef struct {
__REG32 PLSADDR           :28;
__REG32                   : 4;
} ___2d_gepls_bits;

/* Graphic Engine Bresenham Error Term Stepping Constant Register */
typedef struct {
__REG32 A_ERR_INC         :14;
__REG32                   : 2;
__REG32 D_ERR_INC         :14;
__REG32                   : 2;
} ___2d_geber_bits;

/*Graphic Engine Bresenham Initial Error, Pixel Count Major –1 Register */
typedef struct {
__REG32 LPC_M1            :11;
__REG32                   : 5;
__REG32 I_ERR_TERM        :14;
__REG32                   : 2;
} ___2d_gebir_bits;

/* Graphic Engine Control Register */
typedef struct {
__REG32 DDTO              : 1;
__REG32 XY_OCTANT         : 3;
__REG32 PDT               : 1;
__REG32 SRCS              : 2;
__REG32 SDT               : 1;
__REG32 CLPC              : 1;
__REG32 CLIP_EN           : 1;
__REG32 AU                : 1;
__REG32 CTP               : 1;
__REG32 CTS               : 1;
__REG32 MTS               : 1;
__REG32 TRANSPARENCY      : 2;
__REG32 ADDR_MD           : 1;
__REG32 INT_EN            : 1;
__REG32 LSTP              : 1;
__REG32 M_D               : 1;
__REG32 LINE_STYLE        : 1;
__REG32 ALPHA_BLND        : 1;
__REG32 COMMAND           : 2;
__REG32 ROP               : 8;
} ___2d_gec_bits;

/* Graphic Engine Background Color Register */
typedef struct {
__REG32 B_COLOR           :24;
__REG32                   : 8;
} ___2d_gebc_bits;

/* Graphic Engine Foreground Color Register */
typedef struct {
__REG32 F_COLOR           :24;
__REG32                   : 8;
} ___2d_gefc_bits;

/* Graphic Engine Transparency Color Register */
typedef struct {
__REG32 T_COLOR           :24;
__REG32                   : 8;
} ___2d_getc_bits;

/* Graphic Engine Transparency Color Mask Register */
typedef struct {
__REG32 T_COLOR_MASK      :24;
__REG32                   : 8;
} ___2d_getcm_bits;

/* Graphic Engine XY Mode Display Memory Origin Starting Address Register */
typedef struct {
__REG32 XYOSADDR          :28;
__REG32                   : 4;
} ___2d_gexydorg_bits;

/* GGraphic Engine Source/Destination Pitch Register */
typedef struct {
__REG32 SOURCE_PITCH      :13;
__REG32                   : 3;
__REG32 DEST_PITCH        :13;
__REG32                   : 3;
} ___2d_gesdp_bits;

/* Graphic Engine Source Start XY/Linear Addressing Register */
typedef union {
  /*_2D_GESSXY*/
  struct {
    __REG32 SOURCE_START_X    :11;
    __REG32                   : 5;
    __REG32 SOURCE_START_Y    :11;
    __REG32                   : 5;
  };
  /*_2D_GESSL*/
  struct {
    __REG32 SRC_LINEAR_ADDR   :28;
    __REG32                   : 4;
  };
} ___2d_gessxyl_bits;

/* Graphic Engine Source Start XY/Linear Addressing Register */
typedef union {
  /*_2D_GEDSXY*/
  struct {
    __REG32 DEST_START_X      :11;
    __REG32                   : 5;
    __REG32 DEST_START_Y      :11;
    __REG32                   : 5;
  };
  /*_2D_GEDSL*/
  struct {
    __REG32 DEST_LINEAR_ADDR  :28;
    __REG32                   : 4;
  };
} ___2d_gedsxyl_bits;

/* Graphic Engine Dimension for XY/Linear Modes Register */
typedef struct {
__REG32 DIMENSION_X       :11;
__REG32                   : 5;
__REG32 DIMENSION_Y       :11;
__REG32                   : 5;
} ___2d_gedixyl_bits;

/* Graphic Engine Clipping Boundary Top/Left Register */
typedef struct {
__REG32 CLIP_BNDR_L       :11;
__REG32                   : 5;
__REG32 CLIP_BNDR_T       :11;
__REG32                   : 5;
} ___2d_gecbtl_bits;

/* Graphic Engine Clipping Boundary Top/Left Register */
typedef struct {
__REG32 CLIP_BNDR_R       :11;
__REG32                   : 5;
__REG32 CLIP_BNDR_B       :11;
__REG32                   : 5;
} ___2d_gecbbr_bits;

/* Graphic Engine Pattern Group A Register */
typedef struct {
__REG32 PATTERN0          : 8;
__REG32 PATTERN1          : 8;
__REG32 PATTERN2          : 8;
__REG32 PATTERN3          : 8;
} ___2d_geptna_bits;

/* Graphic Engine Pattern Group B Register */
typedef struct {
__REG32 PATTERN4          : 8;
__REG32 PATTERN5          : 8;
__REG32 PATTERN6          : 8;
__REG32 PATTERN7          : 8;
} ___2d_geptnb_bits;

/* Graphic Engine Write Plane Mask Register */
typedef struct {
__REG32 WP_MASK           :24;
__REG32                   : 8;
} ___2d_gewpm_bits;

/* Graphic Engine Miscellaneous Control Register */
typedef struct {
__REG32 BLT_TYPE          : 3;
__REG32 BLT_MD            : 1;
__REG32 BPP               : 2;
__REG32 RST_FIFO          : 1;
__REG32 RST_GE2D          : 1;
__REG32 BUSY              : 1;
__REG32 BITBLTSTS         : 1;
__REG32 FULL              : 1;
__REG32 EMPTY             : 1;
__REG32 FIFO_STATUS       : 4;
__REG32 ABF_KD            : 8;
__REG32 ABF_KS            : 8;
} ___2d_gemc_bits;

/* UART1 Bluetooth Control Register (UART1_UBCR) */
typedef struct {
  __REG32 UBCR              : 3;
  __REG32                   :29;
} __uart_ubcr_bits;

/* UART2 IrDA Control Register (UART2_IRCR) */
typedef struct {
  __REG32 IRDA_EN           : 1;
  __REG32 TX_SELECT         : 1;
  __REG32                   : 3;
  __REG32 INV_TX            : 1;
  __REG32 INV_RX            : 1;
  __REG32                   :25;
} __uart_ircr_bits;

/* UART Interrupt Enable Register (UARTx_IER)
   UART Divisor Latch (High Byte) Register (UARTx_DLM) */
typedef union {
  /*UARTx_IER*/
  struct {
  __REG32 RDAIE             : 1;
  __REG32 THREIE            : 1;
  __REG32 RLSIE             : 1;
  __REG32 MSIE              : 1;
  __REG32                   :28;
  };
  /* UARTx_DLM*/
  struct {
  __REG32 BRD_HI            : 8;
  __REG32                   :24;
  };
} __uart_ier_bits;

/* UART Interrupt Identification Register (UARTx_IIR)
   UART FIFO Control Register (UARTx_FCR) */
typedef union {
  /*UARTx_IIR*/
  struct {
  __REG32 NIP               : 1;
  __REG32 IID               : 3;
  __REG32                   : 1;
  __REG32 RFTLS             : 2;
  __REG32 FMES              : 1;
  __REG32                   :24;
  };
/* UARTx_FCR*/
  struct {
  __REG32 FME               : 1;
  __REG32 RFR               : 1;
  __REG32 TFR               : 1;
  __REG32                   : 1;
  __REG32 RFITL             : 4;
  __REG32                   :24;
  };
} __uart_iir_bits;

/* UART Line Control Register (UART_LCR) */
typedef struct {
  __REG32 WLS               : 2;
  __REG32 NSB               : 1;
  __REG32 PBE               : 1;
  __REG32 EPE               : 1;
  __REG32 SPE               : 1;
  __REG32 BCB               : 1;
  __REG32 DLAB              : 1;
  __REG32                   :24;
} __uart_lcr_bits;

/* UART Modem Control Register (UART_MCR) */
typedef struct {
  __REG32 DTR               : 1;
  __REG32 RTS               : 1;
  __REG32                   : 2;
  __REG32 LBME              : 1;
  __REG32                   :27;
} __uart_mcr_bits;

/* UART Line Status Control Register (UART_LSR) */
typedef struct {
  __REG32 RFDR              : 1;
  __REG32 OEI               : 1;
  __REG32 PEI               : 1;
  __REG32 FEI               : 1;
  __REG32 BII               : 1;
  __REG32 THRE              : 1;
  __REG32 TE                : 1;
  __REG32 ERR_RX            : 1;
  __REG32                   :24;
} __uart_lsr_bits;

/* UART Modem Status Register (UART_MSR) */
typedef struct {
  __REG32 DCTS              : 1;
  __REG32 DDSR              : 1;
  __REG32                   : 2;
  __REG32 CTS               : 1;
  __REG32 DSR               : 1;
  __REG32                   :26;
} __uart_msr_bits;

/* UART Time Out Register (UART_TOR) */
typedef struct {
  __REG32 TOIC              : 7;
  __REG32 TOIE              : 1;
  __REG32                   :24;
} __uart_tor_bits;

/* Timer Control and Status Register 0~4 (TCR0~TCR4)*/
typedef struct {
  __REG32 PRESCALE          : 8;
  __REG32                   :17;
  __REG32 CACT              : 1;
  __REG32 CRST              : 1;
  __REG32 MODE              : 2;
  __REG32 IE                : 1;
  __REG32 CEN               : 1;
  __REG32                   : 1;
} __tcsr_bits;

/* Timer Initial Count Register 0~4 (TICR0~TICR4) */
typedef struct {
  __REG32 TIC               :24;
  __REG32                   : 8;
} __ticr_bits;

/* Timer Data Register 0~4 (TDR0~TDR4) */
typedef struct {
  __REG32 TDR               :24;
  __REG32                   : 8;
} __tdr_bits;

/* Timer Interrupt Status Register (TISR) */
typedef struct {
  __REG32 TIF0              : 1;
  __REG32 TIF1              : 1;
  __REG32 TIF2              : 1;
  __REG32 TIF3              : 1;
  __REG32 TIF4              : 1;
  __REG32                   :27;
} __tisr_bits;

/* Watchdog Timer Control Register (WTCR) */
typedef struct {
  __REG32 WTR               : 1;
  __REG32 WTRE              : 1;
  __REG32 WTRF              : 1;
  __REG32 WTIF              : 1;
  __REG32 WTIS              : 2;
  __REG32 WTIE              : 1;
  __REG32 WTE               : 1;
  __REG32                   : 2;
  __REG32 WTCLK             : 1;
  __REG32                   :21;
} __wtcr_bits;

/* AIC Source Control Registers (AIC_SCR1 ~ AIC_SCR31) */
typedef struct {
  __REG32 PRIORITY          : 3;
  __REG32                   : 3;
  __REG32 SRCTYPE           : 2;
  __REG32                   :24;
} __aic_scr_bits;

/* External Interrupt Control Register (AIC_IRQSC) */
typedef struct {
  __REG32 NIRQ0             : 2;
  __REG32 NIRQ1             : 2;
  __REG32 NIRQ2             : 2;
  __REG32 NIRQ3             : 2;
  __REG32 NIRQ4             : 2;
  __REG32 NIRQ5             : 2;
  __REG32 NIRQ6             : 2;
  __REG32 NIRQ7             : 2;
  __REG32                   :16;
} __aic_irqsc_bits;

/* Interrupt Group Enable Control Register (AIC_GEN) */
typedef struct {
  __REG32 NIRQ0            : 1;
  __REG32 NIRQ1            : 1;
  __REG32 NIRQ2            : 1;
  __REG32 NIRQ3            : 1;
  __REG32 NIRQ4            : 1;
  __REG32 NIRQ5            : 1;
  __REG32 NIRQ6            : 1;
  __REG32 NIRQ7            : 1;
  __REG32 USBH_EHCI        : 1;
  __REG32 USBH_OHCI        : 1;
  __REG32                  : 6;
  __REG32 TMR2             : 1;
  __REG32 TMR3             : 1;
  __REG32 TMR4             : 1;
  __REG32                  : 1;
  __REG32 GDMA0            : 1;
  __REG32 GDMA1            : 1;
  __REG32 COMMRX           : 1;
  __REG32 COMMTX           : 1;
  __REG32 SC0              : 1;
  __REG32 SC1              : 1;
  __REG32 I2C0             : 1;
  __REG32 I2C1             : 1;
  __REG32 PS2P0            : 1;
  __REG32 PS2P1            : 1;
  __REG32                  : 2;
} __aic_gen_bits;

/* Interrupt Group Active Status Register (AIC_GASR) */
typedef struct {
  __REG32 NIRQ0            : 1;
  __REG32 NIRQ1            : 1;
  __REG32 NIRQ2            : 1;
  __REG32 NIRQ3            : 1;
  __REG32 NIRQ4            : 1;
  __REG32 NIRQ5            : 1;
  __REG32 NIRQ6            : 1;
  __REG32 NIRQ7            : 1;
  __REG32 USBH_EHCI        : 1;
  __REG32 USBH_OHCI        : 1;
  __REG32                  : 6;
  __REG32 TMR2             : 1;
  __REG32 TMR3             : 1;
  __REG32 TMR4             : 1;
  __REG32                  : 1;
  __REG32 GDMA0            : 1;
  __REG32 GDMA1            : 1;
  __REG32 COMMRX           : 1;
  __REG32 COMMTX           : 1;
  __REG32 SC0              : 1;
  __REG32 SC1              : 1;
  __REG32 I2C0             : 1;
  __REG32 I2C1             : 1;
  __REG32 PS2P0            : 1;
  __REG32 PS2P1            : 1;
  __REG32                  : 2;
} __aic_gasr_bits;

/* Interrupt Group Status Clear Register (AIC_GSCR) */
typedef struct {
  __REG32 IRQ0             : 1;
  __REG32 IRQ1             : 1;
  __REG32 IRQ2             : 1;
  __REG32 IRQ3             : 1;
  __REG32 IRQ4             : 1;
  __REG32 IRQ5             : 1;
  __REG32 IRQ6             : 1;
  __REG32 IRQ7             : 1;
  __REG32                  :24;
} __aic_gscr_bits;

/* AIC Interrupt Raw Status Register (AIC_IRSR) */
typedef struct {
  __REG32                   : 1;
  __REG32 IRS1              : 1;
  __REG32 IRS2              : 1;
  __REG32 IRS3              : 1;
  __REG32 IRS4              : 1;
  __REG32 IRS5              : 1;
  __REG32 IRS6              : 1;
  __REG32 IRS7              : 1;
  __REG32 IRS8              : 1;
  __REG32 IRS9              : 1;
  __REG32 IRS10             : 1;
  __REG32 IRS11             : 1;
  __REG32 IRS12             : 1;
  __REG32 IRS13             : 1;
  __REG32 IRS14             : 1;
  __REG32 IRS15             : 1;
  __REG32 IRS16             : 1;
  __REG32 IRS17             : 1;
  __REG32 IRS18             : 1;
  __REG32 IRS19             : 1;
  __REG32 IRS20             : 1;
  __REG32 IRS21             : 1;
  __REG32 IRS22             : 1;
  __REG32 IRS23             : 1;
  __REG32 IRS24             : 1;
  __REG32 IRS25             : 1;
  __REG32 IRS26             : 1;
  __REG32 IRS27             : 1;
  __REG32 IRS28             : 1;
  __REG32 IRS29             : 1;
  __REG32 IRS30             : 1;
  __REG32 IRS31             : 1;
} __aic_irsr_bits;

/* AIC Interrupt Active Status Register (AIC_IASR) */
typedef struct {
  __REG32                   : 1;
  __REG32 IAS1              : 1;
  __REG32 IAS2              : 1;
  __REG32 IAS3              : 1;
  __REG32 IAS4              : 1;
  __REG32 IAS5              : 1;
  __REG32 IAS6              : 1;
  __REG32 IAS7              : 1;
  __REG32 IAS8              : 1;
  __REG32 IAS9              : 1;
  __REG32 IAS10             : 1;
  __REG32 IAS11             : 1;
  __REG32 IAS12             : 1;
  __REG32 IAS13             : 1;
  __REG32 IAS14             : 1;
  __REG32 IAS15             : 1;
  __REG32 IAS16             : 1;
  __REG32 IAS17             : 1;
  __REG32 IAS18             : 1;
  __REG32 IAS19             : 1;
  __REG32 IAS20             : 1;
  __REG32 IAS21             : 1;
  __REG32 IAS22             : 1;
  __REG32 IAS23             : 1;
  __REG32 IAS24             : 1;
  __REG32 IAS25             : 1;
  __REG32 IAS26             : 1;
  __REG32 IAS27             : 1;
  __REG32 IAS28             : 1;
  __REG32 IAS29             : 1;
  __REG32 IAS30             : 1;
  __REG32 IAS31             : 1;
} __aic_iasr_bits;

/* AIC Interrupt Status Register (AIC_ISR) */
typedef struct {
  __REG32                   : 1;
  __REG32 IS1               : 1;
  __REG32 IS2               : 1;
  __REG32 IS3               : 1;
  __REG32 IS4               : 1;
  __REG32 IS5               : 1;
  __REG32 IS6               : 1;
  __REG32 IS7               : 1;
  __REG32 IS8               : 1;
  __REG32 IS9               : 1;
  __REG32 IS10              : 1;
  __REG32 IS11              : 1;
  __REG32 IS12              : 1;
  __REG32 IS13              : 1;
  __REG32 IS14              : 1;
  __REG32 IS15              : 1;
  __REG32 IS16              : 1;
  __REG32 IS17              : 1;
  __REG32 IS18              : 1;
  __REG32 IS19              : 1;
  __REG32 IS20              : 1;
  __REG32 IS21              : 1;
  __REG32 IS22              : 1;
  __REG32 IS23              : 1;
  __REG32 IS24              : 1;
  __REG32 IS25              : 1;
  __REG32 IS26              : 1;
  __REG32 IS27              : 1;
  __REG32 IS28              : 1;
  __REG32 IS29              : 1;
  __REG32 IS30              : 1;
  __REG32 IS31              : 1;
} __aic_isr_bits;

/* AIC IRQ Priority Encoding Register (AIC_IPER) */
typedef struct {
  __REG32                   : 2;
  __REG32 VECTOR            : 5;
  __REG32                   :25;
} __aic_iper_bits;

/* AIC Interrupt Source Number Register (AIC_ISNR) */
typedef struct {
  __REG32 IRQID             : 5;
  __REG32                   :27;
} __aic_isnr_bits;

/* AIC Interrupt Mask Register (AIC_IMR) */
typedef struct {
  __REG32                   : 1;
  __REG32 IM1               : 1;
  __REG32 IM2               : 1;
  __REG32 IM3               : 1;
  __REG32 IM4               : 1;
  __REG32 IM5               : 1;
  __REG32 IM6               : 1;
  __REG32 IM7               : 1;
  __REG32 IM8               : 1;
  __REG32 IM9               : 1;
  __REG32 IM10              : 1;
  __REG32 IM11              : 1;
  __REG32 IM12              : 1;
  __REG32 IM13              : 1;
  __REG32 IM14              : 1;
  __REG32 IM15              : 1;
  __REG32 IM16              : 1;
  __REG32 IM17              : 1;
  __REG32 IM18              : 1;
  __REG32 IM19              : 1;
  __REG32 IM20              : 1;
  __REG32 IM21              : 1;
  __REG32 IM22              : 1;
  __REG32 IM23              : 1;
  __REG32 IM24              : 1;
  __REG32 IM25              : 1;
  __REG32 IM26              : 1;
  __REG32 IM27              : 1;
  __REG32 IM28              : 1;
  __REG32 IM29              : 1;
  __REG32 IM30              : 1;
  __REG32 IM31              : 1;
} __aic_imr_bits;

/* AIC Output Interrupt Status Register (AIC_OISR) */
typedef struct {
  __REG32 FIQ               : 1;
  __REG32 IRQ               : 1;
  __REG32                   :30;
} __aic_oisr_bits;

/* AIC Mask Enable Command Register (AIC_MECR) */
typedef struct {
  __REG32                   : 1;
  __REG32 MEC1              : 1;
  __REG32 MEC2              : 1;
  __REG32 MEC3              : 1;
  __REG32 MEC4              : 1;
  __REG32 MEC5              : 1;
  __REG32 MEC6              : 1;
  __REG32 MEC7              : 1;
  __REG32 MEC8              : 1;
  __REG32 MEC9              : 1;
  __REG32 MEC10             : 1;
  __REG32 MEC11             : 1;
  __REG32 MEC12             : 1;
  __REG32 MEC13             : 1;
  __REG32 MEC14             : 1;
  __REG32 MEC15             : 1;
  __REG32 MEC16             : 1;
  __REG32 MEC17             : 1;
  __REG32 MEC18             : 1;
  __REG32 MEC19             : 1;
  __REG32 MEC20             : 1;
  __REG32 MEC21             : 1;
  __REG32 MEC22             : 1;
  __REG32 MEC23             : 1;
  __REG32 MEC24             : 1;
  __REG32 MEC25             : 1;
  __REG32 MEC26             : 1;
  __REG32 MEC27             : 1;
  __REG32 MEC28             : 1;
  __REG32 MEC29             : 1;
  __REG32 MEC30             : 1;
  __REG32 MEC31             : 1;
} __aic_mecr_bits;

/* AIC Mask Disable Command Register (AIC_MDCR) */
typedef struct {
  __REG32                   : 1;
  __REG32 MDC1              : 1;
  __REG32 MDC2              : 1;
  __REG32 MDC3              : 1;
  __REG32 MDC4              : 1;
  __REG32 MDC5              : 1;
  __REG32 MDC6              : 1;
  __REG32 MDC7              : 1;
  __REG32 MDC8              : 1;
  __REG32 MDC9              : 1;
  __REG32 MDC10             : 1;
  __REG32 MDC11             : 1;
  __REG32 MDC12             : 1;
  __REG32 MDC13             : 1;
  __REG32 MDC14             : 1;
  __REG32 MDC15             : 1;
  __REG32 MDC16             : 1;
  __REG32 MDC17             : 1;
  __REG32 MDC18             : 1;
  __REG32 MDC19             : 1;
  __REG32 MDC20             : 1;
  __REG32 MDC21             : 1;
  __REG32 MDC22             : 1;
  __REG32 MDC23             : 1;
  __REG32 MDC24             : 1;
  __REG32 MDC25             : 1;
  __REG32 MDC26             : 1;
  __REG32 MDC27             : 1;
  __REG32 MDC28             : 1;
  __REG32 MDC29             : 1;
  __REG32 MDC30             : 1;
  __REG32 MDC31             : 1;
} __aic_mdcr_bits;

/* GPIO PortC Direction Control Register (GPIOC_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32 OUTEN10             : 1;
__REG32 OUTEN11             : 1;
__REG32 OUTEN12             : 1;
__REG32 OUTEN13             : 1;
__REG32 OUTEN14             : 1;
__REG32 OUTEN15             : 1;
__REG32                     :16;
} __gpioc_dir_bits;

/* GPIO PortC Data Output Register (GPIOC_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32 DATAOUT10           : 1;
__REG32 DATAOUT11           : 1;
__REG32 DATAOUT12           : 1;
__REG32 DATAOUT13           : 1;
__REG32 DATAOUT14           : 1;
__REG32 DATAOUT15           : 1;
__REG32                     :16;
} __gpioc_dataout_bits;

/* GPIO PortC Data Input Register (GPIOC_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32 DATAIN10           : 1;
__REG32 DATAIN11           : 1;
__REG32 DATAIN12           : 1;
__REG32 DATAIN13           : 1;
__REG32 DATAIN14           : 1;
__REG32 DATAIN15           : 1;
__REG32                    :16;
} __gpioc_datain_bits;

/* GPIO PortD Direction Control Register (GPIOD_DIR)*/
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32                     :22;
} __gpiod_dir_bits;

/* GPIO PortD Data Output Register (GPIOD_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32                     :22;
} __gpiod_dataout_bits;

/* GPIO PortC Data Input Register (GPIOC_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32                    :22;
} __gpiod_datain_bits;

/* GPIO PortE Direction Control Register (GPIOE_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32 OUTEN10             : 1;
__REG32 OUTEN11             : 1;
__REG32 OUTEN12             : 1;
__REG32 OUTEN13             : 1;
__REG32                     :18;
} __gpioe_dir_bits;

/* GPIO PortE Data Output Register (GPIOE_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32 DATAOUT10           : 1;
__REG32 DATAOUT11           : 1;
__REG32 DATAOUT12           : 1;
__REG32 DATAOUT13           : 1;
__REG32                     :18;
} __gpioe_dataout_bits;

/* GPIO PortE Data Input Register (GPIOE_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32 DATAIN10           : 1;
__REG32 DATAIN11           : 1;
__REG32 DATAIN12           : 1;
__REG32 DATAIN13           : 1;
__REG32                    :18;
} __gpioe_datain_bits;

/* GPIO PortF Direction Control Register (GPIOF_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32                     :22;
} __gpiof_dir_bits;

/* GPIO PortF Data Output Register (GPIOF_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32                     :22;
} __gpiof_dataout_bits;

/* GPIO PortF Data Input Register (GPIOF_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32                    :22;
} __gpiof_datain_bits;

/* GPIO PortG Direction Control Register (GPIOG_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32 OUTEN10             : 1;
__REG32 OUTEN11             : 1;
__REG32 OUTEN12             : 1;
__REG32 OUTEN13             : 1;
__REG32 OUTEN14             : 1;
__REG32 OUTEN15             : 1;
__REG32 OUTEN16             : 1;
__REG32                     :15;
} __gpiog_dir_bits;

/* GPIO PortG Data Output Register (GPIOG_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32 DATAOUT10           : 1;
__REG32 DATAOUT11           : 1;
__REG32 DATAOUT12           : 1;
__REG32 DATAOUT13           : 1;
__REG32 DATAOUT14           : 1;
__REG32 DATAOUT15           : 1;
__REG32 DATAOUT16           : 1;
__REG32                     :15;
} __gpiog_dataout_bits;

/* GPIO PortG Data Input Register (GPIOG_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32 DATAIN10           : 1;
__REG32 DATAIN11           : 1;
__REG32 DATAIN12           : 1;
__REG32 DATAIN13           : 1;
__REG32 DATAIN14           : 1;
__REG32 DATAIN15           : 1;
__REG32 DATAIN16           : 1;
__REG32                    :15;
} __gpiog_datain_bits;

/* GPIO PortH Debounce Enable Control Register (GPIOH_DBNCE) */
typedef struct {
__REG32 DBEN0               : 1;
__REG32 DBEN1               : 1;
__REG32 DBEN2               : 1;
__REG32 DBEN3               : 1;
__REG32 DBEN4               : 1;
__REG32 DBEN5               : 1;
__REG32 DBEN6               : 1;
__REG32 DBEN7               : 1;
__REG32 DBCLKSEL            : 3;
__REG32                     :21;
} __gpioh_dbnce_bits;

/* GPIO PortH Direction Control Register (GPIOH_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32                     :24;
} __gpioh_dir_bits;

/* GPIO PortH Data Output Register (GPIOH_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32                     :24;
} __gpioh_dataout_bits;

/* GPIO PortH Data Input Register (GPIOH_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32                    :24;
} __gpioh_datain_bits;

/* GPIO PortI Direction Control Register (GPIOI_DIR) */
typedef struct {
__REG32 OUTEN0              : 1;
__REG32 OUTEN1              : 1;
__REG32 OUTEN2              : 1;
__REG32 OUTEN3              : 1;
__REG32 OUTEN4              : 1;
__REG32 OUTEN5              : 1;
__REG32 OUTEN6              : 1;
__REG32 OUTEN7              : 1;
__REG32 OUTEN8              : 1;
__REG32 OUTEN9              : 1;
__REG32 OUTEN10             : 1;
__REG32 OUTEN11             : 1;
__REG32 OUTEN12             : 1;
__REG32 OUTEN13             : 1;
__REG32 OUTEN14             : 1;
__REG32 OUTEN15             : 1;
__REG32 OUTEN16             : 1;
__REG32                     :15;
} __gpioi_dir_bits;

/* GPIO PortI Data Output Register (GPIOI_DATAOUT) */
typedef struct {
__REG32 DATAOUT0            : 1;
__REG32 DATAOUT1            : 1;
__REG32 DATAOUT2            : 1;
__REG32 DATAOUT3            : 1;
__REG32 DATAOUT4            : 1;
__REG32 DATAOUT5            : 1;
__REG32 DATAOUT6            : 1;
__REG32 DATAOUT7            : 1;
__REG32 DATAOUT8            : 1;
__REG32 DATAOUT9            : 1;
__REG32 DATAOUT10           : 1;
__REG32 DATAOUT11           : 1;
__REG32 DATAOUT12           : 1;
__REG32 DATAOUT13           : 1;
__REG32 DATAOUT14           : 1;
__REG32 DATAOUT15           : 1;
__REG32 DATAOUT16           : 1;
__REG32                     :15;
} __gpioi_dataout_bits;

/* GPIO PortI Data Input Register (GPIOI_DATAIN) */
typedef struct {
__REG32 DATAIN0            : 1;
__REG32 DATAIN1            : 1;
__REG32 DATAIN2            : 1;
__REG32 DATAIN3            : 1;
__REG32 DATAIN4            : 1;
__REG32 DATAIN5            : 1;
__REG32 DATAIN6            : 1;
__REG32 DATAIN7            : 1;
__REG32 DATAIN8            : 1;
__REG32 DATAIN9            : 1;
__REG32 DATAIN10           : 1;
__REG32 DATAIN11           : 1;
__REG32 DATAIN12           : 1;
__REG32 DATAIN13           : 1;
__REG32 DATAIN14           : 1;
__REG32 DATAIN15           : 1;
__REG32 DATAIN16           : 1;
__REG32                    :15;
} __gpioi_datain_bits;

/* RTC Access Enable Register (RTC_AER) */
typedef struct {
  __REG32 AER               :17;
  __REG32                   :15;
} __rtc_aer_bits;

/* RTC Frequency Compensation Register (RTC_FCR) */
typedef struct {
  __REG32 FCR_FRA           : 6;
  __REG32                   : 2;
  __REG32 FCR_INT           : 4;
  __REG32                   :20;
} __rtc_fcr_bits;

/* RTC Time Loading Register (RTC_TLR) */
typedef struct {
  __REG32 LO_SEC            : 4;
  __REG32 HI_SEC            : 3;
  __REG32                   : 1;
  __REG32 LO_MIN            : 4;
  __REG32 HI_MIN            : 3;
  __REG32                   : 1;
  __REG32 LO_HR             : 4;
  __REG32 HI_HR             : 2;
  __REG32                   :10;
} __rtc_tlr_bits;

/* RTC Calendar Loading Register (RTC_CLR) */
typedef struct {
  __REG32 LO_DAY            : 4;
  __REG32 HI_DAY            : 2;
  __REG32                   : 2;
  __REG32 LO_MON            : 4;
  __REG32 HI_MON            : 1;
  __REG32                   : 3;
  __REG32 LO_YEAR           : 4;
  __REG32 HI_YEAR           : 4;
  __REG32                   : 8;
} __rtc_clr_bits;

/* RTC Time Scale Selection Register (RTC_TSSR) */
typedef struct {
  __REG32 _24HR             : 1;
  __REG32                   :31;
} __rtc_tssr_bits;

/* RTC Day of the Week Register (RTC_DWR) */
typedef struct {
  __REG32 DWR               : 3;
  __REG32                   :29;
} __rtc_dwr_bits;

/* RTC Time Alarm Register (RTC_TAR) */
typedef struct {
  __REG32 LO_SEC_ALARM      : 4;
  __REG32 HI_SEC_ALARM      : 3;
  __REG32                   : 1;
  __REG32 LO_MIN_ALARM      : 4;
  __REG32 HI_MIN_ALARM      : 3;
  __REG32                   : 1;
  __REG32 LO_HR_ALARM       : 4;
  __REG32 HI_HR_ALARM       : 2;
  __REG32                   :10;
} __rtc_tar_bits;

/* RTC Calendar Alarm Register (RTC_CAR) */
typedef struct {
  __REG32 LO_DAY_ALARM      : 4;
  __REG32 HI_DAY_ALARM      : 2;
  __REG32                   : 2;
  __REG32 LO_MON_ALARM      : 4;
  __REG32 HI_MON_ALARM      : 1;
  __REG32                   : 3;
  __REG32 LO_YEAR_ALARM     : 4;
  __REG32 HI_YEAR_ALARM     : 4;
  __REG32                   : 8;
} __rtc_car_bits;

/* RTC Leap year Indication Register (RTC_LIR) */
typedef struct {
  __REG32 LIR               : 1;
  __REG32                   :31;
} __rtc_lir_bits;

/* RTC Interrupt Enable Register (RTC_RIER) */
typedef struct {
  __REG32 ALARM_INT_EN      : 1;
  __REG32 TICK_INT_EN       : 1;
  __REG32 WK_EN             : 1;
  __REG32                   :29;
} __rtc_rier_bits;

/* RTC Interrupt Indication Register (RTC_RIIR) */
typedef struct {
  __REG32 ALARM_INT_ST      : 1;
  __REG32 TICK_INT_ST       : 1;
  __REG32 WK_ST             : 1;
  __REG32                   :29;
} __rtc_riir_bits;

/* RTC Tick Time Register (RTC_TTR) */
typedef struct {
  __REG32 TTR               : 3;
  __REG32                   :29;
} __rtc_ttr_bits;

/* Interrupt Enable register (SCHI_IER) */
typedef union {
	/*SCHIx_IER*/
	struct {
  __REG32 ERDRI             : 1;
  __REG32 ETBREI            : 1;
  __REG32 ESCSRI            : 1;
  __REG32 ESCPTI            : 1;
  __REG32                   : 2;
  __REG32 A_B               : 1;
  __REG32 PWRDN             : 1;
  __REG32 ETOR0             : 1;
  __REG32 ETOR1             : 1;
  __REG32 ETOR2             : 1;
  __REG32                   :21;
  };
  /*SCHIx_BLH*/
  struct {
  __REG32 BLH	              : 8;
  __REG32                   :24;
  };
} __schi_ier_bits;

/* Interrupt Status Register (SCHI_ISR) 
   Smart Card FIFO control Register (SCHI_SCFR) */
typedef union {
	/*SCHIx_ISR*/
	struct {
  __REG32 NIP         			: 1;
  __REG32 INTS              : 3;
  __REG32 SCPTI             : 1;
  __REG32 SCPSNT	          : 1;
  __REG32                   :26;
  };
	/*SCHIx_SCFR*/
	struct {
  __REG32            				: 1;
  __REG32 RFR               : 1;
  __REG32 TFR               : 1;
  __REG32 PEC0              : 1;
  __REG32 PEC1              : 1;
  __REG32 PEC2  	          : 1;
  __REG32 RFITL  	          : 2;
  __REG32                   :24;
  };
	/*SCHIx_CID*/
	struct {
  __REG32 ID	              : 8;
  __REG32                   :24;
  };
} __schi_isr_bits;

/* Smart Card Control Register (SCHI_SCCR) */
typedef struct {
  __REG32 		              : 2;
  __REG32 CDP		            : 1;
  __REG32 PROT              : 1;
  __REG32 EPE		            : 1;
  __REG32 NSBE              : 1;
  __REG32 DIR			          : 1;
  __REG32 DLAB              : 1;
  __REG32                   :24;
} __schi_sccr_bits;

/* Smart Card Host Status Register (SCHI_CSR) */
typedef struct {
  __REG32 RDR               : 1;
  __REG32 OER		            : 1;
  __REG32 PBER              : 1;
  __REG32 NSER	            : 1;
  __REG32 SBD               : 1;
  __REG32 TBRE		          : 1;
  __REG32 TSRE              : 1;
  __REG32 SC_RESET          : 1;
  __REG32 TOF0              : 1;
  __REG32 TOF1              : 1;
  __REG32 TOF2              : 1;
  __REG32                   :21;
} __schi_csr_bits;

/* Smart Card Host Extended Control Register (SCHI_ECR) */
typedef struct {
  __REG32 		              : 2;
  __REG32 CLKSTPL           : 1;
  __REG32 CLKSTP            : 1;
  __REG32 SCKFS0            : 1;
  __REG32 SCKFS1            : 1;
  __REG32 SCKFS2	          : 1;
  __REG32 	                : 1;
  __REG32 PSCKFS0           : 1;
  __REG32 PSCKFS1           : 1;
  __REG32 PSCKFS2           : 1;
  __REG32                   :21;
} __schi_ecr_bits;

/* Smart Card Host Test Mode Register (SCHI_TEST) */
typedef struct {
  __REG32                 	: 1;
  __REG32 SCRST_L           : 1;
  __REG32                   :30;
} __schi_test_bits;

/* Smart Card Host Time-out configuration Register (SCHI_TOC) */
typedef struct {
  __REG32 TOC0              : 1;
  __REG32 TOC1	            : 1;
  __REG32 TOC2              : 1;
  __REG32                   : 1;
  __REG32 TOC3              : 1;
  __REG32 TOC4		          : 1;
  __REG32 TOC5              : 1;
  __REG32                   : 1;
  __REG32 TOC6	            : 1;
  __REG32 TOC7	            : 1;
  __REG32 TOC8	            : 1;
  __REG32                   :21;
} __schi_toc_bits;

/* Smart Card Host Time-out Initial Register 2 (SCHI_TOIR 2) */
typedef struct {
  __REG32 TOIR2             :24;
  __REG32                   : 8;
} __schi_toir2_bits;

/* Smart Card Host Time-Out Data Register 2 (SCHI_TODR2) */
typedef struct {
  __REG32 TOD2              :24;
  __REG32                   : 8;
} __schi_tod2_bits;

/* Smart Card Host Buffer Time-Out Data Register (SCHI_BTOR) */
typedef struct {
  __REG32 BTOIC_0           : 1;
  __REG32 BTOIC_1           : 1;
  __REG32 BTOIC_2           : 1;
  __REG32 BTOIC_3           : 1;
  __REG32 BTOIC_4           : 1;
  __REG32 BTOIC_5           : 1;
  __REG32 BTOIC_6           : 1;
  __REG32 BTOIE             : 1;
  __REG32                   :24;
} __schi_btor_bits;

/* I2C Control and Status Register 0/1 (I2C_CSR0/1) */
typedef struct {
  __REG32 I2C_EN            : 1;
  __REG32 IE                : 1;
  __REG32 IF                : 1;
  __REG32                   : 1;
  __REG32 TX_NUM            : 2;
  __REG32                   : 2;
  __REG32 I2C_TIP           : 1;
  __REG32 I2C_AL            : 1;
  __REG32 I2C_BUSY          : 1;
  __REG32 I2C_RXACK         : 1;
  __REG32                   :20;
} __i2c_csr_bits;

/* I2C Prescale Register 0/1 (I2C_DIVIDER 0 /1) */
typedef struct {
  __REG32 DIVIDER           :16;
  __REG32                   :16;
} __i2c_divider_bits;

/* I2C Command Register 0/1 (I2C_CMDR 0/1) */
typedef struct {
  __REG32 ACK               : 1;
  __REG32 WRITE             : 1;
  __REG32 READ              : 1;
  __REG32 STOP              : 1;
  __REG32 START             : 1;
  __REG32                   :27;
} __i2c_cmdr_bits;

/* I2C Software Mode Register 0/1(I2C_SWR 0/1) */
typedef struct {
  __REG32 SCW               : 1;
  __REG32 SDW               : 1;
  __REG32 SEW               : 1;
  __REG32 SCR               : 1;
  __REG32 SDR               : 1;
  __REG32 SER               : 1;
  __REG32                   :26;
} __i2c_swr_bits;

/* USI_Control and Status Register (USI_CNTRL) */
typedef struct {
  __REG32 GO_BUSY           : 1;
  __REG32 RX_NEG            : 1;
  __REG32 TX_NEG            : 1;
  __REG32 TX_BIT_LEN        : 5;
  __REG32 TX_NUM            : 2;
  __REG32 LSB               : 1;
  __REG32                   : 1;
  __REG32 SLEEP             : 4;
  __REG32 IF                : 1;
  __REG32 IE                : 1;
  __REG32                   :13;
  __REG32 CLK_POL           : 1;
} __usi_cntrl_bits;

/* USI Divider Register (USI_DIVIDER) */
typedef struct {
  __REG32 DIVIDER           :16;
  __REG32                   :16;
} __usi_divider_bits;

/* USI Slave Select Register (USI_SSR) */
typedef struct {
  __REG32 SSR               : 2;
  __REG32 SS_LVL            : 1;
  __REG32 ASS               : 1;
  __REG32                   :28;
} __usi_ssr_bits;

/* PWM Prescaler Register (PWM_PPR) */
typedef struct {
  __REG32 PRESCALE01        : 8;
  __REG32 PRESCALE23        : 8;
  __REG32 DZI0              : 8;
  __REG32 DZI1              : 8;
} __pwm_ppr_bits;

/* PWM Clock Select Register (PWM_CSR) */
typedef struct {
  __REG32 CH0               : 3;
  __REG32                   : 1;
  __REG32 CH1               : 3;
  __REG32                   : 1;
  __REG32 CH2               : 3;
  __REG32                   : 1;
  __REG32 CH3               : 3;
  __REG32                   :17;
} __pwm_csr_bits;

/* PWM Control Register (PWM_PCR) */
typedef struct {
  __REG32 CH0EN             : 1;
  __REG32                   : 1;
  __REG32 CH0INV            : 1;
  __REG32 CH0MOD            : 1;
  __REG32 DZ0EN             : 1;
  __REG32 DZ1EN             : 1;
  __REG32                   : 2;
  __REG32 CH1EN             : 1;
  __REG32                   : 1;
  __REG32 CH1INV            : 1;
  __REG32 CH1MOD            : 1;
  __REG32 CH2EN             : 1;
  __REG32                   : 1;
  __REG32 CH2INV            : 1;
  __REG32 CH2MOD            : 1;
  __REG32 CH3EN             : 1;
  __REG32                   : 1;
  __REG32 CH3INV            : 1;
  __REG32 CH3MOD            : 1;
  __REG32                   :12;
} __pwm_pcr_bits;

/* PWM Counter Register 0/1/2/3 (PWM_CNR0/1/2/3) */
typedef struct {
  __REG32 CNR               :16;
  __REG32                   :16;
} __pwm_cnr_bits;

/* PWM Comparator Register 0/1/2/3 (PWM_CMR0/1/2/3) */
typedef struct {
  __REG32 CMR               :16;
  __REG32                   :16;
} __pwm_cmr_bits;

/* PWM Data Register 0/1/2/3 (PWM_PDR 0/1/2/3) */
typedef struct {
  __REG32 PDR               :16;
  __REG32                   :16;
} __pwm_pdr_bits;

/* PWM Interrupt Enable Register (PWM_PIER) */
typedef struct {
  __REG32 PIER0             : 1;
  __REG32 PIER1             : 1;
  __REG32 PIER2             : 1;
  __REG32 PIER3             : 1;
  __REG32                   :28;
} __pwm_pier_bits;

/* PWM Interrupt Indication Register (PWM_PIIR) */
typedef struct {
  __REG32 PIIR0             : 1;
  __REG32 PIIR1             : 1;
  __REG32 PIIR2             : 1;
  __REG32 PIIR3             : 1;
  __REG32                   :28;
} __pwm_piir_bits;

/* Keypad Controller Configuration Register (KPI_CONF) */
typedef struct {
  __REG32 PRESCALE          : 8;
  __REG32 DBTC              : 8;
  __REG32 KSIZE             : 2;
  __REG32 ENKP              : 1;
  __REG32 KPSEL             : 1;
  __REG32                   :12;
} __kpiconf_bits;

/* Keypad Controller 3-keys Configuration Register (KPI3KCONF) */
typedef struct {
  __REG32 K30C              : 3;
  __REG32 K30R              : 4;
  __REG32                   : 1;
  __REG32 K31C              : 3;
  __REG32 K31R              : 4;
  __REG32                   : 1;
  __REG32 K32C              : 3;
  __REG32 K32R              : 4;
  __REG32                   : 1;
  __REG32 ENRST             : 1;
  __REG32 EN3KY             : 1;
  __REG32                   : 6;
} __kpi3kconf_bits;

/* KeyPad Interface Low Power Mode Configuration Register (KPILPCONF) */
typedef struct {
  __REG32 LPWR              : 4;
  __REG32                   : 4;
  __REG32 LPWCEN            : 8;
  __REG32 WAKE              : 1;
  __REG32                   :15;
} __kpilpconf_bits;

/* Key Pad Interface Status Register (KPISTATUS) */
typedef struct {
  __REG32 KEY0C             : 3;
  __REG32 KEY0R             : 4;
  __REG32                   : 1;
  __REG32 KEY1C             : 3;
  __REG32 KEY1R             : 4;
  __REG32                   : 1;
  __REG32 _1KEY             : 1;
  __REG32 _2KEY             : 1;
  __REG32 _3KEY             : 1;
  __REG32 PDWAKE            : 1;
  __REG32 _3KRST            : 1;
  __REG32 INT               : 1;
  __REG32                   :10;
} __kpistatus_bits;

/* PS2 Host Controller Command Register (PS2_CMD) */
typedef struct {
  __REG32 PS2CMD            : 8;
  __REG32 ENCMD             : 1;
  __REG32 TRAP_SHIFT        : 1;
  __REG32 RXEOFF            : 1;
  __REG32 RXROFF            : 1;
  __REG32 DWAIT             : 1;
  __REG32                   :19;
} __ps2cmd_bits;

/* PS2 Host Controller Status Register (PS2_STS) */
typedef struct {
  __REG32 RX_IRQ            : 1;
  __REG32 RX_2BYTES         : 1;
  __REG32                   : 2;
  __REG32 TX_IRQ            : 1;
  __REG32 TX_ERR            : 1;
  __REG32                   :26;
} __ps2sts_bits;

/* PS2 Host Controller RX Scan Code Register (PS2_SCANCODE) */
typedef struct {
  __REG32 RX_SCAN_CODE      : 8;
  __REG32 RX_EXTEND         : 1;
  __REG32 RX_RELEASE        : 1;
  __REG32 RX_SHIFT_KEY      : 1;
  __REG32                   : 5;
  __REG32 RX_SCAN_CODE2     : 8;
  __REG32 RX_EXTEND2        : 1;
  __REG32 RX_RELEASE2       : 1;
  __REG32 RX_SHIFT_KEY2     : 1;
  __REG32                   : 5;
} __ps2scancode_bits;

/* PS2 Host Controller Rx ASCII Code Register (PS2ASCII) */
typedef struct {
  __REG32 RX_ASCII_CODE     : 8;
  __REG32 RX_ASCII_CODE2    : 8;
  __REG32                   :16;
} __ps2ascii_bits;

/* ADC Control Register (ADC_CON) */
typedef struct {
__REG32 ADC_FINISH          : 1;
__REG32 ADC_DIV             : 8;
__REG32 ADC_MUX             : 3;
__REG32 ADC_READ_CONV       : 1;
__REG32 ADC_CONV            : 1;
__REG32 ADC_TSC_MODE        : 2;
__REG32 ADC_RST             : 1;
__REG32 ADC_EN              : 1;
__REG32 ADC_INT             : 1;
__REG32 LVD_INT             : 1;
__REG32 WT_INT              : 1;
__REG32 ADC_INT_EN          : 1;
__REG32 LVD_INT_EN          : 1;
__REG32 WT_INT_EN           : 1;
__REG32                     : 8;
} __adc_con_bits;

/* Touch screen control register (ADC_TSC) */
typedef struct {
__REG32 ADC_UD              : 1;
__REG32 ADC_TSC_TYPE        : 2;
__REG32 ADC_PU_EN           : 1;
__REG32 ADC_TSC_YM          : 1;
__REG32 ADC_TSC_YP          : 1;
__REG32 ADC_TSC_XM          : 1;
__REG32 ADC_TSC_XP          : 1;
__REG32 ADC_TSC_XY          : 1;
__REG32                     :23;
} __adc_tsc_bits;

/* ADC Delay Register (ADC_DLY) */
typedef struct {
__REG32 ADC_DELAY           :18;
__REG32 WT_DELAY            :14;
} __adc_dly_bits;

/* ADC X data buffer (ADC_XDATA) */
typedef struct {
__REG32 ADC_XDATA           :10;
__REG32                     :22;
} __adc_xdata_bits;

/* ADC Y data buffer (ADC_YDATA) */
typedef struct {
__REG32 ADC_YDATA           :10;
__REG32                     :22;
} __adc_ydata_bits;

/* Low Voltage Detector Control Register (LV_CON) */
typedef struct {
__REG32 SW_CON              : 2;
__REG32 LV_EN               : 1;
__REG32                     :29;
} __lv_con_bits;

/* Low Voltage Detector Status Register (LV_STS) */
typedef struct {
__REG32 LV_STATUS           : 1;
__REG32                     :31;
} __lv_sts_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** SMC
 **
 ***************************************************************************/
__IO_REG32_BIT(PDID,                  0xB0000000,__READ       ,__pdid_bits);
__IO_REG32_BIT(PWRON,                 0xB0000004,__READ_WRITE ,__pwron_bits);
__IO_REG32_BIT(ARBCON,                0xB0000008,__READ_WRITE ,__arbcon_bits);
__IO_REG32_BIT(MFSEL,                 0xB000000C,__READ_WRITE ,__mfsel_bits);
__IO_REG32_BIT(EBIDPE,                0xB0000010,__READ_WRITE ,__ebidpe_bits);
__IO_REG32_BIT(LCDDPE,                0xB0000014,__READ_WRITE ,__lcddpe_bits);
__IO_REG32_BIT(GPIOCPE,               0xB0000018,__READ_WRITE ,__gpiocpe_bits);
__IO_REG32_BIT(GPIODPE,               0xB000001C,__READ_WRITE ,__gpiodpe_bits);
__IO_REG32_BIT(GPIOEPE,               0xB0000020,__READ_WRITE ,__gpioepe_bits);
__IO_REG32_BIT(GPIOFPE,               0xB0000024,__READ_WRITE ,__gpiofpe_bits);
__IO_REG32_BIT(GPIOGPE,               0xB0000028,__READ_WRITE ,__gpiogpe_bits);
__IO_REG32_BIT(GPIOHPE,               0xB000002C,__READ_WRITE ,__gpiohpe_bits);
__IO_REG32_BIT(GPIOIPE,               0xB0000030,__READ_WRITE ,__gpioipe_bits);
__IO_REG32(    GTMP1,                 0xB0000034,__READ_WRITE );
__IO_REG32(    GTMP2,                 0xB0000038,__READ_WRITE );
__IO_REG32(    GTMP3,                 0xB000003C,__READ_WRITE );

/***************************************************************************
 **
 ** Clock Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(CLKEN,                 0xB0000200, __READ_WRITE, __clken_bits );
__IO_REG32_BIT(CLKSEL,                0xB0000204, __READ_WRITE, __clksel_bits );
__IO_REG32_BIT(CLKDIV,                0xB0000208, __READ_WRITE, __clkdiv_bits );
__IO_REG32_BIT(PLLCON0,               0xB000020C, __READ_WRITE, __pllconx_bits );
__IO_REG32_BIT(PLLCON1,               0xB0000210, __READ_WRITE, __pllconx_bits );
__IO_REG32_BIT(PMCON,                 0xB0000214, __READ_WRITE, __pmcon_bits );
__IO_REG32_BIT(IRQWAKECON,            0xB0000218, __READ_WRITE, __irqwakecon_bits );
__IO_REG32_BIT(IRQWAKEFLAG,           0xB000021C, __READ_WRITE, __irqwakeflag_bits );
__IO_REG32_BIT(IPSRST,                0xB0000220, __READ_WRITE, __ipsrst_bits );
__IO_REG32_BIT(CLKEN1,                0xB0000224, __READ_WRITE, __clken1_bits );
__IO_REG32_BIT(CLKDIV1,               0xB0000228, __READ_WRITE, __clkdiv1_bits );
 
/***************************************************************************
 **
 ** EBI
 **
 ***************************************************************************/
__IO_REG32_BIT(EBICON,                0xB0001000,__READ_WRITE ,__ebicon_bits);
__IO_REG32_BIT(ROMCON,                0xB0001004,__READ_WRITE ,__romcon_bits);
__IO_REG32_BIT(SDCONF0,               0xB0001008,__READ_WRITE ,__sdconf_bits);
__IO_REG32_BIT(SDCONF1,               0xB000100C,__READ_WRITE ,__sdconf_bits);
__IO_REG32_BIT(SDTIME0,               0xB0001010,__READ_WRITE ,__sdtime_bits);
__IO_REG32_BIT(SDTIME1,               0xB0001014,__READ_WRITE ,__sdtime_bits);
__IO_REG32_BIT(EXT0CON,               0xB0001018,__READ_WRITE ,__extcon_bits);
__IO_REG32_BIT(EXT1CON,               0xB000101C,__READ_WRITE ,__extcon_bits);
__IO_REG32_BIT(EXT2CON,               0xB0001020,__READ_WRITE ,__extcon_bits);
__IO_REG32_BIT(EXT3CON,               0xB0001024,__READ_WRITE ,__extcon_bits);
__IO_REG32_BIT(EXT4CON,               0xB0001028,__READ_WRITE ,__extcon_bits);
__IO_REG32_BIT(CKSKEW,                0xB000102C,__READ_WRITE ,__ckskew_bits);
 
/***************************************************************************
 **
 ** EMC
 **
 ***************************************************************************/
__IO_REG32_BIT(CAMCMR,                0xB0003000,__READ_WRITE ,__camcmr_bits);
__IO_REG32_BIT(CAMEN,                 0xB0003004,__READ_WRITE ,__camen_bits);
__IO_REG32_BIT(CAM0M,                 0xB0003008,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM0L,                 0xB000300C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM1M,                 0xB0003010,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM1L,                 0xB0003014,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM2M,                 0xB0003018,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM2L,                 0xB000301C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM3M,                 0xB0003020,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM3L,                 0xB0003024,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM4M,                 0xB0003028,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM4L,                 0xB000302C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM5M,                 0xB0003030,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM5L,                 0xB0003034,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM6M,                 0xB0003038,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM6L,                 0xB000303C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM7M,                 0xB0003040,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM7L,                 0xB0003044,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM8M,                 0xB0003048,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM8L,                 0xB000304C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM9M,                 0xB0003050,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM9L,                 0xB0003054,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM10M,                0xB0003058,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM10L,                0xB000305C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM11M,                0xB0003060,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM11L,                0xB0003064,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM12M,                0xB0003068,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM12L,                0xB000306C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM13M,                0xB0003070,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM13L,                0xB0003074,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM14M,                0xB0003078,__READ_WRITE ,__camm_bits);
__IO_REG32_BIT(CAM14L,                0xB000307C,__READ_WRITE ,__caml_bits);
__IO_REG32_BIT(CAM15M,                0xB0003080,__READ_WRITE ,__cam15m_bits);
__IO_REG32_BIT(CAM15L,                0xB0003084,__READ_WRITE ,__cam15l_bits);
__IO_REG32(    TXDLSA,                0xB0003088,__READ_WRITE );
__IO_REG32(    RXDLSA,                0xB000308C,__READ_WRITE );
__IO_REG32_BIT(MCMDR,                 0xB0003090,__READ_WRITE ,__mcmdr_bits);
__IO_REG32_BIT(MIID,                  0xB0003094,__READ_WRITE ,__miid_bits);
__IO_REG32_BIT(MIIDA,                 0xB0003098,__READ_WRITE ,__miida_bits);
__IO_REG32_BIT(FFTCR,                 0xB000309C,__READ_WRITE ,__fftcr_bits);
__IO_REG32(    TSDR,                  0xB00030A0,__WRITE      );
__IO_REG32(    RSDR,                  0xB00030A4,__WRITE      );
__IO_REG32_BIT(DMARFC,                0xB00030A8,__READ_WRITE ,__dmarfc_bits);
__IO_REG32_BIT(MIEN,                  0xB00030AC,__READ_WRITE ,__mien_bits);
__IO_REG32_BIT(MISTA,                 0xB00030B0,__READ_WRITE ,__mista_bits);
__IO_REG32_BIT(MGSTA,                 0xB00030B4,__READ_WRITE ,__mgsta_bits);
__IO_REG32_BIT(MPCNT,                 0xB00030B8,__READ_WRITE ,__mpcnt_bits);
__IO_REG32_BIT(MRPC,                  0xB00030BC,__READ       ,__mrpc_bits);
__IO_REG32_BIT(MRPCC,                 0xB00030C0,__READ       ,__mrpcc_bits);
__IO_REG32_BIT(MREPC,                 0xB00030C4,__READ       ,__mrepc_bits);
__IO_REG32_BIT(DMARFS,                0xB00030C8,__READ_WRITE ,__dmarfs_bits);
__IO_REG32(    CTXDSA,                0xB00030CC,__READ       );
__IO_REG32(    CTXBSA,                0xB00030D0,__READ       );
__IO_REG32(    CRXDSA,                0xB00030D4,__READ       );
__IO_REG32(    CRXBSA,                0xB00030D8,__READ       );

/***************************************************************************
 **
 ** GDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(GDMA_CTL0,             0xB0004000,__READ_WRITE ,__gdma_ctl_bits);
#define _GDMA_CTL0      GDMA_CTL0
#define _GDMA_CTL0_bit  GDMA_CTL0_bit
__IO_REG32(    GDMA_SRCB0,            0xB0004004,__READ_WRITE );
__IO_REG32(    GDMA_DSTB0,            0xB0004008,__READ_WRITE );
__IO_REG32_BIT(GDMA_TCNT0,            0xB000400C,__READ_WRITE ,__gdma_tcnt_bits);
__IO_REG32(    GDMA_CSRC0,            0xB0004010,__READ       );
__IO_REG32(    GDMA_CDST0,            0xB0004014,__READ       );
__IO_REG32_BIT(GDMA_CTCNT0,           0xB0004018,__READ       ,__gdma_tcnt_bits);
__IO_REG32_BIT(GDMA_DADR0,            0xB000401C,__READ_WRITE ,__gdma_dadr_bits);

__IO_REG32_BIT(GDMA_CTL1,             0xB0004020,__READ_WRITE ,__gdma_ctl_bits);
#define _GDMA_CTL1      GDMA_CTL1
#define _GDMA_CTL1_bit  GDMA_CTL1_bit
__IO_REG32(    GDMA_SRCB1,            0xB0004024,__READ_WRITE );
__IO_REG32(    GDMA_DSTB1,            0xB0004028,__READ_WRITE );
__IO_REG32_BIT(GDMA_TCNT1,            0xB000402C,__READ_WRITE ,__gdma_tcnt_bits);
__IO_REG32(    GDMA_CSRC1,            0xB0004030,__READ       );
__IO_REG32(    GDMA_CDST1,            0xB0004034,__READ       );
__IO_REG32_BIT(GDMA_CTCNT1,           0xB0004038,__READ       ,__gdma_tcnt_bits);
__IO_REG32_BIT(GDMA_DADR1,            0xB000403C,__READ_WRITE ,__gdma_dadr_bits);

__IO_REG32(    GDMA_INTBUF0,          0xB0004080,__READ       );
__IO_REG32(    GDMA_INTBUF1,          0xB0004084,__READ       );
__IO_REG32(    GDMA_INTBUF2,          0xB0004088,__READ       );
__IO_REG32(    GDMA_INTBUF3,          0xB000408C,__READ       );
__IO_REG32(    GDMA_INTBUF4,          0xB0004090,__READ       );
__IO_REG32(    GDMA_INTBUF5,          0xB0004094,__READ       );
__IO_REG32(    GDMA_INTBUF6,          0xB0004098,__READ       );
__IO_REG32(    GDMA_INTBUF7,          0xB000409C,__READ       );
__IO_REG32_BIT(GDMA_INTCS,            0xB00040A0,__READ_WRITE ,__gdma_intcs_bits);

/***************************************************************************
 **
 ** USB Host
 **
 ***************************************************************************/
__IO_REG32_BIT(EHCVNR,               0xB0005000, __READ      , __ehcvnr_bits);
__IO_REG32_BIT(EHCSPR,               0xB0005004, __READ      , __ehcspr_bits);
__IO_REG32_BIT(EHCCPR,               0xB0005008, __READ      , __ehccpr_bits);
__IO_REG32_BIT(UCMDR,                0xB0005020, __READ_WRITE, __ucmdr_bits);
__IO_REG32_BIT(USTSR,                0xB0005024, __READ_WRITE, __ustsr_bits);
__IO_REG32_BIT(UIENR,                0xB0005028, __READ_WRITE, __uienr_bits);
__IO_REG32_BIT(UFINDR,               0xB000502C, __READ_WRITE, __ufindr_bits);
__IO_REG32_BIT(UPFLBAR,              0xB0005034, __READ_WRITE, __upflbar_bits);
__IO_REG32_BIT(UCALAR,               0xB0005038, __READ_WRITE, __ucalar_bits);
__IO_REG32_BIT(UASSTR,               0xB000503C, __READ_WRITE, __uasstr_bits);
__IO_REG32_BIT(UCFGR,                0xB0005060, __READ_WRITE, __ucfgr_bits);
__IO_REG32_BIT(UPSCR0,               0xB0005064, __READ_WRITE, __upscrx_bits);
__IO_REG32_BIT(UPSCR1,               0xB0005068, __READ_WRITE, __upscrx_bits);
__IO_REG32_BIT(USBPCR0,              0xB00050C4, __READ_WRITE, __usbpcr0_bits);
__IO_REG32_BIT(USBPCR1,              0xB00050C8, __READ_WRITE, __usbpcr1_bits);
 
__IO_REG32_BIT(HcRevision,            0xB0007000,__READ       ,__HcRevision_bits);
__IO_REG32_BIT(HcControl,             0xB0007004,__READ_WRITE ,__HcControl_bits);
__IO_REG32_BIT(HcCommandStatus,       0xB0007008,__READ_WRITE ,__HcCommandStatus_bits);
__IO_REG32_BIT(HcInterruptStatus,     0xB000700C,__READ_WRITE ,__HcInterruptStatus_bits);
__IO_REG32_BIT(HcInterruptEnable,     0xB0007010,__READ_WRITE ,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcInterruptDisable,    0xB0007014,__READ_WRITE ,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcHCCA,                0xB0007018,__READ_WRITE ,__HcHCCA_bits);
__IO_REG32_BIT(HcPeriodCurrentED,     0xB000701C,__READ_WRITE ,__HcPeriodCurrentED_bits);
__IO_REG32_BIT(HcControlHeadED,       0xB0007020,__READ_WRITE ,__HcControlHeadED_bits);
__IO_REG32_BIT(HcControlCurrentED,    0xB0007024,__READ_WRITE ,__HcControlCurrentED_bits);
__IO_REG32_BIT(HcBulkHeadED,          0xB0007028,__READ_WRITE ,__HcBulkHeadED_bits);
__IO_REG32_BIT(HcBulkCurrentED,       0xB000702C,__READ_WRITE ,__HcBulkCurrentED_bits);
__IO_REG32_BIT(HcDoneHead,            0xB0007030,__READ_WRITE ,__HcDoneHead_bits);
__IO_REG32_BIT(HcFmInterval,          0xB0007034,__READ_WRITE ,__HcFmInterval_bits);
__IO_REG32_BIT(HcFmRemaining,         0xB0007038,__READ       ,__HcFmRemaining_bits);
__IO_REG16(    HcFmNumber,            0xB000703C,__READ       );
__IO_REG32_BIT(HcPeriodicStart,       0xB0007040,__READ_WRITE ,__HcPeriodicStart_bits);
__IO_REG32_BIT(HcLSThreshold,         0xB0007044,__READ_WRITE ,__HcLSThreshold_bits);
__IO_REG32_BIT(HcRhDescriptorA,       0xB0007048,__READ_WRITE ,__HcRhDescriptorA_bits);
__IO_REG32_BIT(HcRhDescriptorB,       0xB000704C,__READ_WRITE ,__HcRhDescriptorB_bits);
__IO_REG32_BIT(HcRhStatus,            0xB0007050,__READ_WRITE ,__HcRhStatus_bits);
__IO_REG32_BIT(HcRhPortStatus1,       0xB0007054,__READ_WRITE ,__HcRhPortStatus_bits);
__IO_REG32_BIT(HcRhPortStatus2,       0xB0007058,__READ_WRITE ,__HcRhPortStatus_bits);
__IO_REG32_BIT(OPERATIONALMODEENABLE, 0xB0007204,__READ_WRITE ,__operationalmodeenable_bits);

/***************************************************************************
 **
 ** USB Device
 **
 ***************************************************************************/
__IO_REG32_BIT(IRQ_STAT,              0xB0006000, __READ      , __irq_stat_bits);
__IO_REG32_BIT(IRQ_ENB_L,             0xB0006008, __READ_WRITE, __irq_enb_l_bits);
__IO_REG32_BIT(USB_IRQ_STAT,          0xB0006010, __READ_WRITE, __usb_irq_stat_bits);
__IO_REG32_BIT(USB_IRQ_ENB,           0xB0006014, __READ_WRITE, __usb_irq_enb_bits);
__IO_REG32_BIT(USB_OPER,              0xB0006018, __READ_WRITE, __usb_oper_bits);
__IO_REG32_BIT(USB_FRAME_CNT,         0xB000601C, __READ      , __usb_frame_cnt_bits);
__IO_REG32_BIT(USB_ADDR,              0xB0006020, __READ_WRITE, __usb_addr_bits);
__IO_REG32_BIT(CEP_DATA_BUF,          0xB0006028, __READ_WRITE, __cep_data_buf_bits);
__IO_REG32_BIT(CEP_CTRL_STAT,         0xB000602C, __READ_WRITE, __cep_ctrl_stat_bits);
__IO_REG32_BIT(CEP_IRQ_ENB,           0xB0006030, __READ_WRITE, __cep_irq_enb_bits);
__IO_REG32_BIT(CEP_IRQ_STAT,          0xB0006034, __READ_WRITE, __cep_irq_stat_bits);
__IO_REG32_BIT(IN_TRNSFR_CNT,         0xB0006038, __READ_WRITE, __in_trnsfr_cnt_bits);
__IO_REG32_BIT(OUT_TRNSFR_CNT,        0xB000603C, __READ      , __out_trnsfr_cnt_bits);
__IO_REG32_BIT(CEP_CNT,               0xB0006040, __READ      , __cep_cnt_bits);
__IO_REG32_BIT(SETUP1_0,              0xB0006044, __READ      , __setup1_0_bits);
__IO_REG32_BIT(SETUP3_2,              0xB0006048, __READ      , __setup3_2_bits);
__IO_REG32_BIT(SETUP5_4,              0xB000604C, __READ      , __setup5_4_bits);
__IO_REG32_BIT(SETUP7_6,              0xB0006050, __READ      , __setup7_6_bits);
__IO_REG32_BIT(CEP_START_ADDR,        0xB0006054, __READ_WRITE, __cep_start_addr_bits);
__IO_REG32_BIT(CEP_END_ADDR,          0xB0006058, __READ_WRITE, __cep_end_addr_bits);
__IO_REG32_BIT(DMA_CTRL_STS,          0xB000605C, __READ_WRITE, __dma_ctrl_sts_bits);
__IO_REG32_BIT(DMA_CNT,               0xB0006060, __READ_WRITE, __dma_cnt_bits);
__IO_REG32_BIT(EPA_DATA_BUF,          0xB0006064, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPA_IRQ_STAT,          0xB0006068, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPA_IRQ_ENB,           0xB000606C, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPA_DATA_CNT,          0xB0006070, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPA_RSP_SC,            0xB0006074, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPA_MPS,               0xB0006078, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPA_CNT,               0xB000607C, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPA_CFG,               0xB0006080, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPA_START_ADDR,        0xB0006084, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPA_END_ADDR,          0xB0006088, __READ_WRITE, __ep_end_addr_bits);
__IO_REG32_BIT(EPB_DATA_BUF,          0xB000608C, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPB_IRQ_STAT,          0xB0006090, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPB_IRQ_ENB,           0xB0006094, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPB_DATA_CNT,          0xB0006098, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPB_RSP_SC,            0xB000609C, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPB_MPS,               0xB00060A0, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPB_CNT,               0xB00060A4, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPB_CFG,               0xB00060A8, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPB_START_ADDR,        0xB00060AC, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPB_END_ADDR,          0xB00060B0, __READ_WRITE, __ep_end_addr_bits);
__IO_REG32_BIT(EPC_DATA_BUF,          0xB00060B4, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPC_IRQ_STAT,          0xB00060B8, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPC_IRQ_ENB,           0xB00060BC, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPC_DATA_CNT,          0xB00060C0, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPC_RSP_SC,            0xB00060C4, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPC_MPS,               0xB00060C8, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPC_CNT,               0xB00060CC, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPC_CFG,               0xB00060D0, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPC_START_ADDR,        0xB00060D4, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPC_END_ADDR,          0xB00060D8, __READ_WRITE, __ep_end_addr_bits);
__IO_REG32_BIT(EPD_DATA_BUF,          0xB00060DC, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPD_IRQ_STAT,          0xB00060E0, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPD_IRQ_ENB,           0xB00060E4, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPD_DATA_CNT,          0xB00060E8, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPD_RSP_SC,            0xB00060EC, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPD_MPS,               0xB00060F0, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPD_CNT,               0xB00060F4, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPD_CFG,               0xB00060F8, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPD_START_ADDR,        0xB00060FC, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPD_END_ADDR,          0xB0006100, __READ_WRITE, __ep_end_addr_bits);
__IO_REG32_BIT(EPE_DATA_BUF,          0xB0006104, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPE_IRQ_STAT,          0xB0006108, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPE_IRQ_ENB,           0xB000610C, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPE_DATA_CNT,          0xB0006110, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPE_RSP_SC,            0xB0006114, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPE_MPS,               0xB0006118, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPE_CNT,               0xB000611C, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPE_CFG,               0xB0006120, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPE_START_ADDR,        0xB0006124, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPE_END_ADDR,          0xB0006128, __READ_WRITE, __ep_end_addr_bits);
__IO_REG32_BIT(EPF_DATA_BUF,          0xB000612C, __READ_WRITE, __ep_data_buf_bits);
__IO_REG32_BIT(EPF_IRQ_STAT,          0xB0006130, __READ_WRITE, __ep_irq_stat_bits);
__IO_REG32_BIT(EPF_IRQ_ENB,           0xB0006134, __READ_WRITE, __ep_irq_enb_bits);
__IO_REG32_BIT(EPF_DATA_CNT,          0xB0006138, __READ      , __ep_data_cnt_bits);
__IO_REG32_BIT(EPF_RSP_SC,            0xB000613C, __READ_WRITE, __ep_rsp_sc_bits);
__IO_REG32_BIT(EPF_MPS,               0xB0006140, __READ_WRITE, __ep_mps_bits);
__IO_REG32_BIT(EPF_CNT,               0xB0006144, __READ_WRITE, __ep_cnt_bits);
__IO_REG32_BIT(EPF_CFG,               0xB0006148, __READ_WRITE, __ep_cfg_bits);
__IO_REG32_BIT(EPF_START_ADDR,        0xB000614C, __READ_WRITE, __ep_start_addr_bits);
__IO_REG32_BIT(EPF_END_ADDR,          0xB0006150, __READ_WRITE, __ep_end_addr_bits);

__IO_REG32(    USB_DMA_ADDR,          0xB0006700, __READ_WRITE  );
__IO_REG32_BIT(USB_PHY_CTL,           0xB0006704, __READ_WRITE, __usb_phy_ctl_bits);

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32(    DMAC_FB_BASE,          0xB000C000, __READ_WRITE );
__IO_REG32_BIT(DMACCSR,               0xB000C800, __READ_WRITE, __dmaccsr_bits);
__IO_REG32(    DMACSAR1,              0xB000C804, __READ_WRITE );
__IO_REG32(    DMACSAR2,              0xB000C808, __READ_WRITE );
__IO_REG32_BIT(DMACBCR,               0xB000C80C, __READ      , __dmacbcr_bits);
__IO_REG32_BIT(DMACIER,               0xB000C810, __READ_WRITE, __dmacier_bits);
__IO_REG32_BIT(DMACISR,               0xB000C814, __READ_WRITE, __dmacisr_bits);

/***************************************************************************
 **
 ** FMI
 **
 ***************************************************************************/
__IO_REG32_BIT(FMICSR,                0xB000D000, __READ_WRITE, __fmicsr_bits);
__IO_REG32_BIT(FMIIER,                0xB000D004, __READ_WRITE, __fmiier_bits);
__IO_REG32_BIT(FMIISR,                0xB000D008, __READ_WRITE, __fmiisr_bits);
__IO_REG32_BIT(SDCSR,                 0xB000D020, __READ_WRITE, __sdcsr_bits);
__IO_REG32(    SDARG,                 0xB000D024, __READ_WRITE );
__IO_REG32_BIT(SDIER,                 0xB000D028, __READ_WRITE, __sdier_bits);
__IO_REG32_BIT(SDISR,                 0xB000D02C, __READ_WRITE, __sdisr_bits);
__IO_REG32_BIT(SDRSP0,                0xB000D030, __READ      , __sdrsp0_bits);
__IO_REG32_BIT(SDRSP1,                0xB000D034, __READ      , __sdrsp1_bits);
__IO_REG32_BIT(SDBLEN,                0xB000D038, __READ_WRITE, __sdblen_bits);
__IO_REG32_BIT(SDTMOUT,               0xB000D03C, __READ_WRITE, __sdtmout_bits);
__IO_REG32_BIT(MSCSR,                 0xB000D060, __READ_WRITE, __mscsr_bits);
__IO_REG32_BIT(MSIER,                 0xB000D064, __READ_WRITE, __msier_bits);
__IO_REG32_BIT(MSISR,                 0xB000D068, __READ_WRITE, __msisr_bits);
__IO_REG32(    MSBUF1,                0xB000D06C, __READ_WRITE );
__IO_REG32(    MSBUF2,                0xB000D070, __READ_WRITE );
__IO_REG32_BIT(SMCSR,                 0xB000D0A0, __READ_WRITE, __smcsr_bits);
__IO_REG32_BIT(SMTCR,                 0xB000D0A4, __READ_WRITE, __smtcr_bits);
__IO_REG32_BIT(SMIER,                 0xB000D0A8, __READ_WRITE, __smier_bits);
__IO_REG32_BIT(SMISR,                 0xB000D0AC, __READ_WRITE, __smisr_bits);
__IO_REG32_BIT(SMCMD,                 0xB000D0B0, __WRITE     , __smcmd_bits);
__IO_REG32_BIT(SMADDR,                0xB000D0B4, __WRITE     , __smaddr_bits);
__IO_REG32_BIT(SMDATA,                0xB000D0B8, __READ_WRITE, __smdata_bits);
__IO_REG32_BIT(SMECC0,                0xB000D0BC, __READ      , __smecc0_bits);
__IO_REG32_BIT(SMECC1,                0xB000D0C0, __READ      , __smecc1_bits);
__IO_REG32_BIT(SMECC2,                0xB000D0C4, __READ      , __smecc2_bits);
__IO_REG32_BIT(SMECC3,                0xB000D0C8, __READ      , __smecc3_bits);
__IO_REG32(    SMRA_0,                0xB000D0CC, __READ_WRITE );
__IO_REG32(    SMRA_1,                0xB000D0D0, __READ_WRITE );
__IO_REG32(    SMRA_2,                0xB000D0D4, __READ_WRITE );
__IO_REG32(    SMRA_3,                0xB000D0D8, __READ_WRITE );
__IO_REG32(    SMRA_4,                0xB000D0DC, __READ_WRITE );
__IO_REG32(    SMRA_5,                0xB000D0E0, __READ_WRITE );
__IO_REG32(    SMRA_6,                0xB000D0E4, __READ_WRITE );
__IO_REG32(    SMRA_7,                0xB000D0E8, __READ_WRITE );
__IO_REG32(    SMRA_8,                0xB000D0EC, __READ_WRITE );
__IO_REG32(    SMRA_9,                0xB000D0F0, __READ_WRITE );
__IO_REG32(    SMRA_10,               0xB000D0F4, __READ_WRITE );
__IO_REG32(    SMRA_11,               0xB000D0F8, __READ_WRITE );
__IO_REG32(    SMRA_12,               0xB000D0FC, __READ_WRITE );
__IO_REG32(    SMRA_13,               0xB000D100, __READ_WRITE );
__IO_REG32(    SMRA_14,               0xB000D104, __READ_WRITE );
__IO_REG32(    SMRA_15,               0xB000D108, __READ_WRITE );
__IO_REG32_BIT(SMECCAD0,              0xB000D10C, __READ      , __smeccad0_bits);
__IO_REG32_BIT(SMECCAD1,              0xB000D110, __READ      , __smeccad1_bits);
__IO_REG32_BIT(ECC4ST,                0xB000D114, __READ      , __ecc4st_bits);
__IO_REG32_BIT(ECC4F1A1,              0xB000D118, __READ      , __ecc4f1a1_bits);
__IO_REG32_BIT(ECC4F1A2,              0xB000D11C, __READ      , __ecc4f1a2_bits);
__IO_REG32_BIT(ECC4F1D,               0xB000D120, __READ      , __ecc4f1d_bits);
__IO_REG32_BIT(ECC4F2A1,              0xB000D124, __READ      , __ecc4f2a1_bits);
__IO_REG32_BIT(ECC4F2A2,              0xB000D128, __READ      , __ecc4f2a2_bits);
__IO_REG32_BIT(ECC4F2D,               0xB000D12C, __READ      , __ecc4f2d_bits);
__IO_REG32_BIT(ECC4F3A1,              0xB000D130, __READ      , __ecc4f3a1_bits);
__IO_REG32_BIT(ECC4F3A2,              0xB000D134, __READ      , __ecc4f3a2_bits);
__IO_REG32_BIT(ECC4F3D,               0xB000D138, __READ      , __ecc4f3d_bits);
__IO_REG32_BIT(ECC4F4A1,              0xB000D13C, __READ      , __ecc4f4a1_bits);
__IO_REG32_BIT(ECC4F4A2,              0xB000D140, __READ      , __ecc4f4a2_bits);
__IO_REG32_BIT(ECC4F4D,               0xB000D144, __READ      , __ecc4f4d_bits);

/***************************************************************************
 **
 ** LCD
 **
 ***************************************************************************/
__IO_REG32_BIT(DCCS,                  0xB0008000, __READ_WRITE, __dccs_bits);
__IO_REG32_BIT(DEVICE_CTRL,           0xB0008004, __READ_WRITE, __device_ctrl_bits);
__IO_REG32_BIT(MPULCD_CMD,            0xB0008008, __READ_WRITE, __mpulcd_cmd_bits);
__IO_REG32_BIT(INT_CS,                0xB000800C, __READ_WRITE, __int_cs_bits);
__IO_REG32_BIT(CRTC_SIZE,             0xB0008010, __READ_WRITE, __crtc_size_bits);
__IO_REG32_BIT(CRTC_DEND,             0xB0008014, __READ_WRITE, __crtc_dend_bits);
__IO_REG32_BIT(CRTC_HR,               0xB0008018, __READ_WRITE, __crtc_hr_bits);
__IO_REG32_BIT(CRTC_HSYNC,            0xB000801C, __READ_WRITE, __crtc_hsync_bits);
__IO_REG32_BIT(CRTC_VR,               0xB0008020, __READ_WRITE, __crtc_vr_bits);
__IO_REG32(    VA_BADDR0,             0xB0008024, __READ_WRITE );
__IO_REG32(    VA_BADDR1,             0xB0008028, __READ_WRITE );
__IO_REG32_BIT(VA_FBCTRL,             0xB000802C, __READ_WRITE, __va_fbctrl_bits);
__IO_REG32_BIT(VA_SCALE,              0xB0008030, __READ_WRITE, __va_scale_bits);
__IO_REG32_BIT(VA_WIN,                0xB0008038, __READ_WRITE, __va_win_bits);
__IO_REG32_BIT(VA_STUFF,              0xB000803C, __READ_WRITE, __va_stuff_bits);
__IO_REG32_BIT(OSD_WINS,              0xB0008040, __READ_WRITE, __osd_wins_bits);
__IO_REG32_BIT(OSD_WINE,              0xB0008044, __READ_WRITE, __osd_wine_bits);
__IO_REG32(    OSD_BADDR,             0xB0008048, __READ_WRITE );
__IO_REG32_BIT(OSD_FBCTRL,            0xB000804C, __READ_WRITE, __osd_fbctrl_bits);
__IO_REG32_BIT(OSD_OVERLAY,           0xB0008050, __READ_WRITE, __osd_overlay_bits);
__IO_REG32_BIT(OSD_CKEY,              0xB0008054, __READ_WRITE, __osd_ckey_bits);
__IO_REG32_BIT(OSD_CMASK,             0xB0008058, __READ_WRITE, __osd_cmask_bits);
__IO_REG32_BIT(OSD_SKIP1,             0xB000805C, __READ_WRITE, __osd_skip1_bits);
__IO_REG32_BIT(OSD_SKIP2,             0xB0008060, __READ_WRITE, __osd_skip2_bits);
__IO_REG32_BIT(OSD_SCALE,             0xB0008064, __READ_WRITE, __osd_scale_bits);
__IO_REG32_BIT(MPU_VSYNC,             0xB0008068, __READ_WRITE, __mpu_vsync_bits);
__IO_REG32_BIT(HC_CTRL,               0xB000806C, __READ_WRITE, __hc_ctrl_bits);
__IO_REG32_BIT(HC_POS,                0xB0008070, __READ_WRITE, __hc_pos_bits);
__IO_REG32_BIT(HC_WBCTRL,             0xB0008074, __READ_WRITE, __hc_wbctrl_bits);
__IO_REG32(    HC_BADDR,              0xB0008078, __READ_WRITE );
__IO_REG32_BIT(HC_COLOR0,             0xB000807C, __READ_WRITE, __hc_color0_bits);
__IO_REG32_BIT(HC_COLOR1,             0xB0008080, __READ_WRITE, __hc_color1_bits);
__IO_REG32_BIT(HC_COLOR2,             0xB0008084, __READ_WRITE, __hc_color2_bits);
__IO_REG32_BIT(HC_COLOR3,             0xB0008088, __READ_WRITE, __hc_color3_bits);

/***************************************************************************
 **
 ** Audio Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(ACTL_CON,              0xB0009000, __READ_WRITE, __actl_con_bits);
__IO_REG32_BIT(ACTL_RESET,            0xB0009004, __READ_WRITE, __actl_reset_bits);
__IO_REG32(    ACTL_RDSTB,            0xB0009008, __READ_WRITE );
__IO_REG32(    ACTL_RDST_LENGTH,      0xB000900C, __READ_WRITE );
__IO_REG32(    ACTL_RDSTC,            0xB0009010, __READ       );
__IO_REG32_BIT(ACTL_RSR,              0xB0009014, __READ_WRITE, __actl_rsr_bits);
__IO_REG32(    ACTL_PDSTB,            0xB0009018, __READ_WRITE );
__IO_REG32(    ACTL_PDST_LENGTH,      0xB000901C, __READ_WRITE );
__IO_REG32(    ACTL_PDSTC,            0xB0009020, __READ       );
__IO_REG32_BIT(ACTL_PSR,              0xB0009024, __READ_WRITE, __actl_psr_bits);
__IO_REG32_BIT(ACTL_I2SCON,           0xB0009028, __READ_WRITE, __actl_i2scon_bits);
__IO_REG32_BIT(ACTL_ACCON,            0xB000902C, __READ_WRITE, __actl_accon_bits);
__IO_REG32_BIT(ACTL_ACOS0,            0xB0009030, __READ_WRITE, __actl_acos0_bits);
__IO_REG32_BIT(ACTL_ACOS1,            0xB0009034, __READ_WRITE, __actl_acos1_bits);
__IO_REG32_BIT(ACTL_ACOS2,            0xB0009038, __READ_WRITE, __actl_acos2_bits);
__IO_REG32_BIT(ACTL_ACIS0,            0xB000903C, __READ      , __actl_acis0_bits);
__IO_REG32_BIT(ACTL_ACIS1,            0xB0009040, __READ      , __actl_acis1_bits);
__IO_REG32_BIT(ACTL_ACIS2,            0xB0009044, __READ      , __actl_acis2_bits);
__IO_REG32(    ACTL_COUNTER,          0xB0009048, __READ_WRITE );

/***************************************************************************
 **
 ** ATAPI
 **
 ***************************************************************************/
__IO_REG32_BIT(CSR,                   0xB000A000, __READ_WRITE, __csr_bits);
__IO_REG32_BIT(INTR,                  0xB000A004, __READ_WRITE, __intr_bits);
__IO_REG32_BIT(PINSTAT,               0xB000A008, __READ      , __pinstat_bits);
__IO_REG32_BIT(DMACSR,                0xB000A00C, __READ_WRITE, __dmacsr_bits);
__IO_REG32_BIT(SECCNT,                0xB000A010, __READ_WRITE, __seccnt_bits);
__IO_REG32_BIT(REGTTR,                0xB000A020, __READ_WRITE, __regttr_bits);
__IO_REG32_BIT(PIOTTR,                0xB000A024, __READ_WRITE, __piottr_bits);
__IO_REG32_BIT(DMATTR,                0xB000A028, __READ_WRITE, __dmattr_bits);
__IO_REG32_BIT(UDMATTR,               0xB000A02C, __READ_WRITE, __udmattr_bits);
__IO_REG32_BIT(ATA_DATA,              0xB000A100, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_FEA,               0xB000A104, __READ_WRITE, __ata_data_bits);
#define ATA_ERR     ATA_FEA
#define ATA_ERR_bit ATA_FEA_bit
__IO_REG32_BIT(ATA_SEC,               0xB000A108, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_LBAL,              0xB000A10C, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_LBAM,              0xB000A110, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_LBAH,              0xB000A114, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_DEVH,              0xB000A118, __READ_WRITE, __ata_data_bits);
__IO_REG32_BIT(ATA_COMD,              0xB000A11C, __READ_WRITE, __ata_data_bits);
#define ATA_STAT      ATA_COMD
#define ATA_STAT_bit  ATA_COMD_bit
__IO_REG32_BIT(ATA_DCTRL,             0xB000A120, __READ_WRITE, __ata_data_bits);
#define ATA_ASTAT     ATA_DCTRL
#define ATA_ASTAT_bit ATA_DCTRL_bit

/***************************************************************************
 **
 ** 2D GE
 **
 ***************************************************************************/
__IO_REG32_BIT(_2D_GETG,              0xB000B000, __READ_WRITE, ___2d_getg_bits);
__IO_REG32_BIT(_2D_GEXYSORG,          0xB000B004, __READ_WRITE, ___2d_gexysorg_bits);
__IO_REG32_BIT(_2D_TileXY,            0xB000B008, __READ_WRITE, ___2d_tilexy_vhsf_bits);
#define _2D_VHSF      _2D_TileXY 
#define _2D_VHSF_bit  _2D_TileXY_bit 
__IO_REG32_BIT(_2D_GERRXY,            0xB000B00C, __READ_WRITE, ___2d_gerrxy_bits);
__IO_REG32_BIT(_2D_GEINTS,            0xB000B010, __READ_WRITE, ___2d_geints_bits);
__IO_REG32_BIT(_2D_GEPLS,             0xB000B014, __READ_WRITE, ___2d_gepls_bits);
__IO_REG32_BIT(_2D_GEBER,             0xB000B018, __READ_WRITE, ___2d_geber_bits);
__IO_REG32_BIT(_2D_GEBIR,             0xB000B01C, __READ_WRITE, ___2d_gebir_bits);
__IO_REG32_BIT(_2D_GEC,               0xB000B020, __READ_WRITE, ___2d_gec_bits);
__IO_REG32_BIT(_2D_GEBC,              0xB000B024, __READ_WRITE, ___2d_gebc_bits);
__IO_REG32_BIT(_2D_GEFC,              0xB000B028, __READ_WRITE, ___2d_gefc_bits);
__IO_REG32_BIT(_2D_GETC,              0xB000B02C, __READ_WRITE, ___2d_getc_bits);
__IO_REG32_BIT(_2D_GETCM,             0xB000B030, __READ_WRITE, ___2d_getcm_bits);
__IO_REG32_BIT(_2D_GEXYDORG,          0xB000B034, __READ_WRITE, ___2d_gexydorg_bits);
__IO_REG32_BIT(_2D_GESDP,             0xB000B038, __READ_WRITE, ___2d_gesdp_bits);
__IO_REG32_BIT(_2D_GESSXY,            0xB000B03C, __READ_WRITE, ___2d_gessxyl_bits);
#define _2D_GESSL     _2D_GESSXY
#define _2D_GESSL_bit _2D_GESSXY_bit
__IO_REG32_BIT(_2D_GEDSXY,            0xB000B040, __READ_WRITE, ___2d_gedsxyl_bits);
#define _2D_GEDSL     _2D_GEDSXY
#define _2D_GEDSL_bit _2D_GEDSXY_bit
__IO_REG32_BIT(_2D_GEDIXYL,           0xB000B044, __READ_WRITE, ___2d_gedixyl_bits);
__IO_REG32_BIT(_2D_GECBTL,            0xB000B048, __READ_WRITE, ___2d_gecbtl_bits);
__IO_REG32_BIT(_2D_GECBBR,            0xB000B04C, __READ_WRITE, ___2d_gecbbr_bits);
__IO_REG32_BIT(_2D_GEPTNA,            0xB000B050, __READ_WRITE, ___2d_geptna_bits);
__IO_REG32_BIT(_2D_GEPTNB,            0xB000B054, __READ_WRITE, ___2d_geptnb_bits);
__IO_REG32_BIT(_2D_GEWPM,             0xB000B058, __READ_WRITE, ___2d_gewpm_bits);
__IO_REG32_BIT(_2D_GEMC,              0xB000B05C, __READ_WRITE, ___2d_gemc_bits);
 
/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
__IO_REG8(     UART0_RBR,             0xB8000000,__READ_WRITE );
#define UART0_THR     UART0_RBR
#define UART0_DLL     UART0_RBR
__IO_REG32_BIT(UART0_IER,             0xB8000004,__READ_WRITE ,__uart_ier_bits);
#define UART0_DLM     UART0_IER
#define UART0_DLM_bit UART0_IER_bit
__IO_REG32_BIT(UART0_IIR,             0xB8000008,__READ_WRITE ,__uart_iir_bits);
#define UART0_FCR     UART0_IIR
#define UART0_FCR_bit UART0_IIR_bit
__IO_REG32_BIT(UART0_LCR,             0xB800000C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART0_LSR,             0xB8000014,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART0_TOR,             0xB800001C,__READ_WRITE ,__uart_tor_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
__IO_REG8(     UART1_RBR,             0xB8000100,__READ_WRITE );
#define UART1_THR     UART1_RBR
#define UART1_DLL     UART1_RBR
__IO_REG32_BIT(UART1_IER,             0xB8000104,__READ_WRITE ,__uart_ier_bits);
#define UART1_DLM     UART1_IER
#define UART1_DLM_bit UART1_IER_bit
__IO_REG32_BIT(UART1_IIR,             0xB8000108,__READ_WRITE ,__uart_iir_bits);
#define UART1_FCR     UART1_IIR
#define UART1_FCR_bit UART1_IIR_bit
__IO_REG32_BIT(UART1_LCR,             0xB800010C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART1_MCR,             0xB8000110,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART1_LSR,             0xB8000114,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART1_MSR,             0xB8000118,__READ       ,__uart_msr_bits);
__IO_REG32_BIT(UART1_TOR,             0xB800011C,__READ_WRITE ,__uart_tor_bits);
__IO_REG32_BIT(UART1_UBCR,            0xB8000120,__READ_WRITE ,__uart_ubcr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
__IO_REG8(     UART2_RBR,             0xB8000200,__READ_WRITE );
#define UART2_THR     UART2_RBR
#define UART2_DLL     UART2_RBR
__IO_REG32_BIT(UART2_IER,             0xB8000204,__READ_WRITE ,__uart_ier_bits);
#define UART2_DLM     UART2_IER
#define UART2_DLM_bit UART2_IER_bit
__IO_REG32_BIT(UART2_IIR,             0xB8000208,__READ_WRITE ,__uart_iir_bits);
#define UART2_FCR     UART2_IIR
#define UART2_FCR_bit UART2_IIR_bit
__IO_REG32_BIT(UART2_LCR,             0xB800020C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART2_LSR,             0xB8000214,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART2_TOR,             0xB800021C,__READ_WRITE ,__uart_tor_bits);
__IO_REG32_BIT(UART2_IRCR,            0xB8000220,__READ_WRITE ,__uart_ircr_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
__IO_REG8(     UART3_RBR,             0xB8000300,__READ_WRITE );
#define UART3_THR     UART3_RBR
#define UART3_DLL     UART3_RBR
__IO_REG32_BIT(UART3_IER,             0xB8000304,__READ_WRITE ,__uart_ier_bits);
#define UART3_DLM     UART3_IER
#define UART3_DLM_bit UART3_IER_bit
__IO_REG32_BIT(UART3_IIR,             0xB8000308,__READ_WRITE ,__uart_iir_bits);
#define UART3_FCR     UART3_IIR
#define UART3_FCR_bit UART3_IIR_bit
__IO_REG32_BIT(UART3_LCR,             0xB800030C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART3_MCR,             0xB8000310,__READ_WRITE ,__uart_mcr_bits);
__IO_REG32_BIT(UART3_LSR,             0xB8000314,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART3_MSR,             0xB8000318,__READ       ,__uart_msr_bits);
__IO_REG32_BIT(UART3_TOR,             0xB800031C,__READ_WRITE ,__uart_tor_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG8(     UART4_RBR,             0xB8000400,__READ_WRITE );
#define UART4_THR     UART4_RBR
#define UART4_DLL     UART4_RBR
__IO_REG32_BIT(UART4_IER,             0xB8000404,__READ_WRITE ,__uart_ier_bits);
#define UART4_DLM     UART4_IER
#define UART4_DLM_bit UART4_IER_bit
__IO_REG32_BIT(UART4_IIR,             0xB8000408,__READ_WRITE ,__uart_iir_bits);
#define UART4_FCR     UART4_IIR
#define UART4_FCR_bit UART4_IIR_bit
__IO_REG32_BIT(UART4_LCR,             0xB800040C,__READ_WRITE ,__uart_lcr_bits);
__IO_REG32_BIT(UART4_LSR,             0xB8000414,__READ       ,__uart_lsr_bits);
__IO_REG32_BIT(UART4_TOR,             0xB800041C,__READ_WRITE ,__uart_tor_bits);

/***************************************************************************
 **
 ** Timers
 **
 ***************************************************************************/
__IO_REG32_BIT(TCSR0,                 0xB8001000,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCSR1,                 0xB8001004,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TICR0,                 0xB8001008,__READ_WRITE ,__ticr_bits);
__IO_REG32_BIT(TICR1,                 0xB800100C,__READ_WRITE ,__ticr_bits);
__IO_REG32_BIT(TDR0,                  0xB8001010,__READ       ,__tdr_bits);
__IO_REG32_BIT(TDR1,                  0xB8001014,__READ       ,__tdr_bits);
__IO_REG32_BIT(TISR,                  0xB8001018,__READ_WRITE ,__tisr_bits);
__IO_REG32_BIT(TCSR2,                 0xB8001020,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TCSR3,                 0xB8001024,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TICR2,                 0xB8001028,__READ_WRITE ,__ticr_bits);
__IO_REG32_BIT(TICR3,                 0xB800102C,__READ_WRITE ,__ticr_bits);
__IO_REG32_BIT(TDR2,                  0xB8001030,__READ       ,__tdr_bits);
__IO_REG32_BIT(TDR3,                  0xB8001034,__READ       ,__tdr_bits);
__IO_REG32_BIT(TCSR4,                 0xB8001040,__READ_WRITE ,__tcsr_bits);
__IO_REG32_BIT(TICR4,                 0xB8001048,__READ_WRITE ,__ticr_bits);
__IO_REG32_BIT(TDR4,                  0xB8001050,__READ       ,__tdr_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WTCR,                  0xB800101C,__READ_WRITE ,__wtcr_bits);

/***************************************************************************
 **
 ** AIC
 **
 ***************************************************************************/
__IO_REG32_BIT(AIC_SCR1,              0xB8002004,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR2,              0xB8002008,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR3,              0xB800200C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR4,              0xB8002010,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR5,              0xB8002014,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR6,              0xB8002018,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR7,              0xB800201C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR8,              0xB8002020,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR9,              0xB8002024,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR10,             0xB8002028,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR11,             0xB800202C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR12,             0xB8002030,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR13,             0xB8002034,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR14,             0xB8002038,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR15,             0xB800203C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR16,             0xB8002040,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR17,             0xB8002044,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR18,             0xB8002048,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR19,             0xB800204C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR20,             0xB8002050,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR21,             0xB8002054,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR22,             0xB8002058,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR23,             0xB800205C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR24,             0xB8002060,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR25,             0xB8002064,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR26,             0xB8002068,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR27,             0xB800206C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR28,             0xB8002070,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR29,             0xB8002074,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR30,             0xB8002078,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_SCR31,             0xB800207C,__READ_WRITE ,__aic_scr_bits);
__IO_REG32_BIT(AIC_IRQSC,             0xB8002080,__READ_WRITE ,__aic_irqsc_bits);
__IO_REG32_BIT(AIC_GEN,               0xB8002084,__READ_WRITE ,__aic_gen_bits);
__IO_REG32_BIT(AIC_GASR,              0xB8002088,__READ       ,__aic_gasr_bits);
__IO_REG32_BIT(AIC_GSCR,              0xB800208C,__READ_WRITE ,__aic_gscr_bits);
__IO_REG32_BIT(AIC_IRSR,              0xB8002100,__READ       ,__aic_irsr_bits);
__IO_REG32_BIT(AIC_IASR,              0xB8002104,__READ       ,__aic_iasr_bits);
__IO_REG32_BIT(AIC_ISR,               0xB8002108,__READ       ,__aic_isr_bits);
__IO_REG32_BIT(AIC_IPER,              0xB800210C,__READ       ,__aic_iper_bits);
__IO_REG32_BIT(AIC_ISNR,              0xB8002110,__READ       ,__aic_isnr_bits);
__IO_REG32_BIT(AIC_IMR,               0xB8002114,__READ       ,__aic_imr_bits);
__IO_REG32_BIT(AIC_OISR,              0xB8002118,__READ       ,__aic_oisr_bits);
__IO_REG32_BIT(AIC_MECR,              0xB8002120,__WRITE      ,__aic_mecr_bits);
__IO_REG32_BIT(AIC_MDCR,              0xB8002124,__WRITE      ,__aic_mdcr_bits);
__IO_REG32(    AIC_EOSCR,             0xB8002130,__WRITE      );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOC_DIR,             0xB8003004, __READ_WRITE, __gpioc_dir_bits);
__IO_REG32_BIT(GPIOC_DATAOUT,         0xB8003008, __READ_WRITE, __gpioc_dataout_bits);
__IO_REG32_BIT(GPIOC_DATAIN,          0xB800300C, __READ      , __gpioc_datain_bits);
__IO_REG32_BIT(GPIOD_DIR,             0xB8003014, __READ_WRITE, __gpiod_dir_bits);
__IO_REG32_BIT(GPIOD_DATAOUT,         0xB8003018, __READ_WRITE, __gpiod_dataout_bits);
__IO_REG32_BIT(GPIOD_DATAIN,          0xB800301C, __READ      , __gpiod_datain_bits);
__IO_REG32_BIT(GPIOE_DIR,             0xB8003024, __READ_WRITE, __gpioe_dir_bits);
__IO_REG32_BIT(GPIOE_DATAOUT,         0xB8003028, __READ_WRITE, __gpioe_dataout_bits);
__IO_REG32_BIT(GPIOE_DATAIN,          0xB800302C, __READ      , __gpioe_datain_bits);
__IO_REG32_BIT(GPIOF_DIR,             0xB8003034, __READ_WRITE, __gpiof_dir_bits);
__IO_REG32_BIT(GPIOF_DATAOUT,         0xB8003038, __READ_WRITE, __gpiof_dataout_bits);
__IO_REG32_BIT(GPIOF_DATAIN,          0xB800303C, __READ      , __gpiof_datain_bits);
__IO_REG32_BIT(GPIOG_DIR,             0xB8003044, __READ_WRITE, __gpiog_dir_bits);
__IO_REG32_BIT(GPIOG_DATAOUT,         0xB8003048, __READ_WRITE, __gpiog_dataout_bits);
__IO_REG32_BIT(GPIOG_DATAIN,          0xB800304C, __READ      , __gpiog_datain_bits);
__IO_REG32_BIT(GPIOH_DBNCE,           0xB8003050, __READ_WRITE, __gpioh_dbnce_bits);
__IO_REG32_BIT(GPIOH_DIR,             0xB8003054, __READ_WRITE, __gpioh_dir_bits);
__IO_REG32_BIT(GPIOH_DATAOUT,         0xB8003058, __READ_WRITE, __gpioh_dataout_bits);
__IO_REG32_BIT(GPIOH_DATAIN,          0xB800305C, __READ      , __gpioh_datain_bits);
__IO_REG32_BIT(GPIOI_DIR,             0xB8003064, __READ_WRITE, __gpioi_dir_bits);
__IO_REG32_BIT(GPIOI_DATAOUT,         0xB8003068, __READ_WRITE, __gpioi_dataout_bits);
__IO_REG32_BIT(GPIOI_DATAIN,          0xB800306C, __READ      , __gpioi_datain_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RTC_INIR,              0xB8004000,__READ_WRITE );
__IO_REG32_BIT(RTC_AER,               0xB8004004,__READ_WRITE ,__rtc_aer_bits);
__IO_REG32_BIT(RTC_FCR,               0xB8004008,__READ_WRITE ,__rtc_fcr_bits);
__IO_REG32_BIT(RTC_TLR,               0xB800400C,__READ_WRITE ,__rtc_tlr_bits);
__IO_REG32_BIT(RTC_CLR,               0xB8004010,__READ_WRITE ,__rtc_clr_bits);
__IO_REG32_BIT(RTC_TSSR,              0xB8004014,__READ_WRITE ,__rtc_tssr_bits);
__IO_REG32_BIT(RTC_DWR,               0xB8004018,__READ_WRITE ,__rtc_dwr_bits);
__IO_REG32_BIT(RTC_TAR,               0xB800401C,__READ_WRITE ,__rtc_tar_bits);
__IO_REG32_BIT(RTC_CAR,               0xB8004020,__READ_WRITE ,__rtc_car_bits);
__IO_REG32_BIT(RTC_LIR,               0xB8004024,__READ       ,__rtc_lir_bits);
__IO_REG32_BIT(RTC_RIER,              0xB8004028,__READ_WRITE ,__rtc_rier_bits);
__IO_REG32_BIT(RTC_RIIR,              0xB800402C,__READ_WRITE ,__rtc_riir_bits);
__IO_REG32_BIT(RTC_TTR,               0xB8004030,__READ_WRITE ,__rtc_ttr_bits);

/***************************************************************************
 **
 ** SCHI0
 **
 ***************************************************************************/
__IO_REG8(		 SCHI0_RBR,             0xB8005000,__READ_WRITE );
#define SCHI0_TBR 			SCHI0_RBR
#define SCHI0_BLL 			SCHI0_RBR
__IO_REG32_BIT(SCHI0_IER,             0xB8005004,__READ_WRITE ,__schi_ier_bits);
#define SCHI0_BLH 			SCHI0_IER
#define SCHI0_BLH_bit 	SCHI0_IER_bit
__IO_REG32_BIT(SCHI0_ISR,             0xB8005008,__READ_WRITE ,__schi_isr_bits);
#define SCHI0_SCFR 			SCHI0_ISR
#define SCHI0_SCFR_bit	SCHI0_ISR_bit
#define SCHI0_CID				SCHI0_ISR
#define SCHI0_CID_bit		SCHI0_ISR_bit
__IO_REG32_BIT(SCHI0_SCCR,            0xB800500C,__READ_WRITE ,__schi_sccr_bits);
__IO_REG8(     SCHI0_CBR,            	0xB8005010,__READ_WRITE );
__IO_REG32_BIT(SCHI0_CSR,             0xB8005014,__READ				,__schi_csr_bits);
__IO_REG8(		 SCHI0_GTR,            	0xB8005018,__READ_WRITE );
__IO_REG32_BIT(SCHI0_ECR,            	0xB800501C,__READ_WRITE ,__schi_ecr_bits);
__IO_REG32_BIT(SCHI0_TEST,           	0xB8005020,__READ_WRITE ,__schi_test_bits);
__IO_REG32_BIT(SCHI0_TOC,            	0xB8005028,__READ_WRITE ,__schi_toc_bits);
__IO_REG8(		 SCHI0_TOIR,            0xB800502C,__READ_WRITE );
__IO_REG16(		 SCHI0_TOIR1,           0xB8005030,__READ_WRITE );
__IO_REG32_BIT(SCHI0_TOIR2,           0xB8005034,__READ_WRITE ,__schi_toir2_bits);
__IO_REG8(		 SCHI0_TOD0,          	0xB8005038,__READ				);
__IO_REG16(		 SCHI0_TOD1,          	0xB800503C,__READ				);
__IO_REG32_BIT(SCHI0_TOD2,          	0xB8005040,__READ				,__schi_tod2_bits);
__IO_REG32_BIT(SCHI0_BTOR,          	0xB8005044,__READ_WRITE ,__schi_btor_bits);

/***************************************************************************
 **
 ** SCHI1
 **
 ***************************************************************************/
__IO_REG8(		 SCHI1_RBR,             0xB8005800,__READ_WRITE );
#define SCHI1_TBR 			SCHI_RBR
#define SCHI1_BLL 			SCHI_RBR
__IO_REG32_BIT(SCHI1_IER,             0xB8005804,__READ_WRITE ,__schi_ier_bits);
#define SCHI1_BLH 			SCHI1_IER
#define SCHI1_BLH_bit		SCHI1_IER_bit
__IO_REG32_BIT(SCHI1_ISR,             0xB8005808,__READ_WRITE ,__schi_isr_bits);
#define SCHI1_SCFR 			SCHI_ISR
#define SCHI1_SCFR_bit	SCHI_ISR_bit
#define SCHI1_CID				SCHI_ISR
#define SCHI1_CID_bit		SCHI_ISR_bit
__IO_REG32_BIT(SCHI1_SCCR,            0xB800580C,__READ_WRITE ,__schi_sccr_bits);
__IO_REG8(     SCHI1_CBR,            	0xB8005810,__READ_WRITE );
__IO_REG32_BIT(SCHI1_CSR,             0xB8005814,__READ				,__schi_csr_bits);
__IO_REG8(		 SCHI1_GTR,            	0xB8005818,__READ_WRITE );
__IO_REG32_BIT(SCHI1_ECR,            	0xB800581C,__READ_WRITE ,__schi_ecr_bits);
__IO_REG32_BIT(SCHI1_TEST,          	0xB8005820,__READ_WRITE ,__schi_test_bits);
__IO_REG32_BIT(SCHI1_TOC,            	0xB8005828,__READ_WRITE ,__schi_toc_bits);
__IO_REG8(		 SCHI1_TOIR0,           0xB800582C,__READ_WRITE );
__IO_REG16(		 SCHI1_TOIR1,           0xB8005830,__READ_WRITE );
__IO_REG32_BIT(SCHI1_TOIR2,           0xB8005834,__READ_WRITE ,__schi_toir2_bits);
__IO_REG8(		 SCHI1_TOD0,          	0xB8005838,__READ				);
__IO_REG16(		 SCHI1_TOD1,          	0xB800583C,__READ				);
__IO_REG32_BIT(SCHI1_TOD2,          	0xB8005840,__READ				,__schi_tod2_bits);
__IO_REG32_BIT(SCHI1_BTOR,          	0xB8005844,__READ_WRITE ,__schi_btor_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_CSR0,              0xB8006000,__READ_WRITE ,__i2c_csr_bits);
__IO_REG32_BIT(I2C_DIVIDER0,          0xB8006004,__READ_WRITE ,__i2c_divider_bits);
__IO_REG32_BIT(I2C_CMDR0,             0xB8006008,__READ_WRITE ,__i2c_cmdr_bits);
__IO_REG32_BIT(I2C_SWR0,              0xB800600C,__READ_WRITE ,__i2c_swr_bits);
__IO_REG8(     I2C_RXR0,              0xB8006010,__READ       );
__IO_REG32(    I2C_TXR0,              0xB8006014,__READ_WRITE );

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_CSR1,              0xB8006100,__READ_WRITE ,__i2c_csr_bits);
__IO_REG32_BIT(I2C_DIVIDER1,          0xB8006104,__READ_WRITE ,__i2c_divider_bits);
__IO_REG32_BIT(I2C_CMDR1,             0xB8006108,__READ_WRITE ,__i2c_cmdr_bits);
__IO_REG32_BIT(I2C_SWR1,              0xB800610C,__READ_WRITE ,__i2c_swr_bits);
__IO_REG8(     I2C_RXR1,              0xB8006110,__READ       );
__IO_REG32(    I2C_TXR1,              0xB8006114,__READ_WRITE );

/***************************************************************************
 **
 ** USI
 **
 ***************************************************************************/
__IO_REG32_BIT(USI_CNTRL,             0xB8006200,__READ_WRITE ,__usi_cntrl_bits);
__IO_REG32_BIT(USI_DIVIDER,           0xB8006204,__READ_WRITE ,__usi_divider_bits);
__IO_REG32_BIT(USI_SSR,               0xB8006208,__READ_WRITE ,__usi_ssr_bits);
__IO_REG32(    USI_RX0,               0xB8006210,__READ_WRITE );
__IO_REG32(    USI_RX1,               0xB8006214,__READ_WRITE );
__IO_REG32(    USI_RX2,               0xB8006218,__READ_WRITE );
__IO_REG32(    USI_RX3,               0xB800621C,__READ_WRITE );

#define USI_TX0 USI_RX0
#define USI_TX1 USI_RX1
#define USI_TX2 USI_RX2
#define USI_TX3 USI_RX3

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_PPR,               0xB8007000,__READ_WRITE ,__pwm_ppr_bits);
__IO_REG32_BIT(PWM_CSR,               0xB8007004,__READ_WRITE ,__pwm_csr_bits);
__IO_REG32_BIT(PWM_PCR,               0xB8007008,__READ_WRITE ,__pwm_pcr_bits);
__IO_REG32_BIT(PWM_CNR0,              0xB800700C,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR0,              0xB8007010,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR0,              0xB8007014,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR1,              0xB8007018,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR1,              0xB800701C,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR1,              0xB8007020,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR2,              0xB8007024,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR2,              0xB8007028,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR2,              0xB800702C,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_CNR3,              0xB8007030,__READ_WRITE ,__pwm_cnr_bits);
__IO_REG32_BIT(PWM_CMR3,              0xB8007034,__READ_WRITE ,__pwm_cmr_bits);
__IO_REG32_BIT(PWM_PDR3,              0xB8007038,__READ       ,__pwm_pdr_bits);
__IO_REG32_BIT(PWM_PIER,              0xB800703C,__READ_WRITE ,__pwm_pier_bits);
__IO_REG32_BIT(PWM_PIIR,              0xB8007040,__READ_WRITE ,__pwm_piir_bits);

/***************************************************************************
 **
 ** KPI
 **
 ***************************************************************************/
__IO_REG32_BIT(KPICONF,               0xB8008000,__READ_WRITE ,__kpiconf_bits);
__IO_REG32_BIT(KPI3KCONF,             0xB8008004,__READ_WRITE ,__kpi3kconf_bits);
__IO_REG32_BIT(KPILPCONF,             0xB8008008,__READ_WRITE ,__kpilpconf_bits);
__IO_REG32_BIT(KPISTATUS,             0xB800800C,__READ       ,__kpistatus_bits);

/***************************************************************************
 **
 ** PS2 Port1
 **
 ***************************************************************************/
__IO_REG32_BIT(PS2P1CMD,              0xB8009000,__READ_WRITE ,__ps2cmd_bits);
__IO_REG32_BIT(PS2P1STS,              0xB8009004,__READ_WRITE ,__ps2sts_bits);
__IO_REG32_BIT(PS2P1SCANCODE,         0xB8009008,__READ       ,__ps2scancode_bits);
__IO_REG32_BIT(PS2P1ASCII,            0xB800900C,__READ       ,__ps2ascii_bits);

/***************************************************************************
 **
 ** PS2 Port2
 **
 ***************************************************************************/
__IO_REG32_BIT(PS2P2CMD,              0xB8009100,__READ_WRITE ,__ps2cmd_bits);
__IO_REG32_BIT(PS2P2STS,              0xB8009104,__READ_WRITE ,__ps2sts_bits);
__IO_REG32_BIT(PS2P2SCANCODE,         0xB8009108,__READ       ,__ps2scancode_bits);
__IO_REG32_BIT(PS2P2ASCII,            0xB800910C,__READ       ,__ps2ascii_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_CON,               0xB800A000, __READ_WRITE, __adc_con_bits);
__IO_REG32_BIT(ADC_TSC,               0xB800A004, __READ_WRITE, __adc_tsc_bits);
__IO_REG32_BIT(ADC_DLY,               0xB800A008, __READ_WRITE, __adc_dly_bits);
__IO_REG32_BIT(ADC_XDATA,             0xB800A00C, __READ      , __adc_xdata_bits);
__IO_REG32_BIT(ADC_YDATA,             0xB800A010, __READ      , __adc_ydata_bits);
__IO_REG32_BIT(LV_CON,                0xB800A014, __READ_WRITE, __lv_con_bits);
__IO_REG32_BIT(LV_STS,                0xB800A018, __READ      , __lv_sts_bits);

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
#define RESETV        0x00  /* Reset                                       */
#define UNDEFV        0x04  /* Undefined instruction                       */
#define SWIV          0x08  /* Software interrupt                          */
#define PABORTV       0x0c  /* Prefetch abort                              */
#define DABORTV       0x10  /* Data abort                                  */
#define IRQV          0x18  /* Normal interrupt                            */
#define FIQV          0x1c  /* Fast interrupt                              */

/***************************************************************************
 **
 **  VIC Interrupt channels
 **
 ***************************************************************************/
#define AIC_WDT          1  /* Watchdog                                    */
#define AIC_NIRQ_G0      2  /* External Interrupt Group 0                  */
#define AIC_NIRQ_G1      3  /* External Interrupt Group 1                  */
#define AIC_ACTL         4  /* Audio Controller Interrupt                  */
#define AIC_LCD          5  /* LCD Controller Interrupt                    */
#define AIC_RTC          6  /* RTC Interrupt                               */
#define AIC_UART0        7  /* UART Interrupt0                             */
#define AIC_UART1        8  /* UART Interrupt1                             */
#define AIC_UART2        9  /* UART Interrupt2                             */
#define AIC_UART3       10  /* UART Interrupt3                             */
#define AIC_UART4       11  /* UART Interrupt4                             */
#define AIC_TIMER0      12  /* Timer Interrupt 0                           */
#define AIC_TIMER1      13  /* Timer Interrupt 1                           */
#define AIC_TIMER_G     14  /* Timer Interrupt Group                       */
#define AIC_USBH_G      15  /* USB Host Interrupt Group                    */
#define AIC_EMCTX       16  /* EMC TX Interrupt                            */
#define AIC_EMCRX       17  /* EMC RX Interrupt                            */
#define AIC_GDMA_G      18  /* GDMA Channel Group                          */
#define AIC_DMAC        19  /* GDMA Channel Group                          */
#define AIC_FMI         20  /* FMI Interrupt                               */
#define AIC_USBD        21  /* USB Device Interrupt                        */
#define AIC_ATAPI       22  /* ATAPI Interrupt                             */
#define AIC_G2D         23  /* 2D Graphic Engine Interrupt                 */
#define AIC_SC_G        25  /* Smart Card Interrupt Group                  */
#define AIC_I2C_G       26  /* I2C Interrupt Group                         */
#define AIC_USI         27  /* USI Interrupt                               */
#define AIC_PWM         28  /* PWM Timer interrupt                         */
#define AIC_KPI         29  /* Keypad Interrupt                            */
#define AIC_PS2_G       30  /* PS2 Interrupt  Group                        */
#define AIC_ADC         31  /* ADC Interrupt                               */


#endif    /* __W90P910_H */
