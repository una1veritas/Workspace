/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Fujitsu MB86R01
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 39069 $
 **
 **    Note:
 ***************************************************************************/

#ifndef __IOMB86R01_H
#define __IOMB86R01_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MB86R01 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

/*PLL control register (CRPR)*/
typedef struct {
  __REG32 PLLMODE     : 5;
  __REG32 LUWMODE     : 2;
  __REG32 PLLBYPASS   : 1;
  __REG32 PLLRDY      : 1;
  __REG32             :23;
} __crpr_bits;

/*Watchdog timer control register (CRWR)*/
typedef struct {
  __REG32 WDTMODE       : 2;
  __REG32 WDTSET_WDTCLR : 1;
  __REG32 WDRST         : 1;
  __REG32 TBR           : 1;
  __REG32               : 2;
  __REG32 ERST          : 1;
  __REG32               :24;
} __crwr_bits;

/*Reset/Standby control register (CRSR)*/
typedef struct {
  __REG32 SWRMODE       : 1;
  __REG32 SWRSTREQ      : 1;
  __REG32 SWRST         : 1;
  __REG32 SRST          : 1;
  __REG32               : 3;
  __REG32 STOPEN        : 1;
  __REG32               :24;
} __crsr_bits;

/*Clock divider control register A (CRDA)*/
typedef struct {
  __REG32 HADM          : 3;
  __REG32 PADM          : 3;
  __REG32 PBDM          : 3;
  __REG32 ARMADM        : 3;
  __REG32 ARMBDM        : 3;
  __REG32               :17;
} __crda_bits;

/*Clock divider control register B (CRDB)*/
typedef struct {
  __REG32 HBDM          : 3;
  __REG32               :29;
} __crdb_bits;

/*AHB (A) bus clock gate control register (CRHA)*/
typedef struct {
  __REG32 HACLK0        : 1;
  __REG32 HACLK1        : 1;
  __REG32 HACLK2        : 1;
  __REG32 HACLK3        : 1;
  __REG32 HACLK4        : 1;
  __REG32 HACLK5        : 1;
  __REG32 HACLK6        : 1;
  __REG32 HACLK7        : 1;
  __REG32 HACLK8        : 1;
  __REG32 HACLK9        : 1;
  __REG32 HACLK10       : 1;
  __REG32 HACLK11       : 1;
  __REG32 HACLK12       : 1;
  __REG32 HACLK13       : 1;
  __REG32 HACLK14       : 1;
  __REG32 HACLK15       : 1;
  __REG32               :16;
} __crha_bits;

/*APB (A) bus clock gate control register (CRPA)*/
typedef struct {
  __REG32 PACLK0        : 1;
  __REG32 PACLK1        : 1;
  __REG32 PACLK2        : 1;
  __REG32 PACLK3        : 1;
  __REG32 PACLK4        : 1;
  __REG32 PACLK5        : 1;
  __REG32 PACLK6        : 1;
  __REG32 PACLK7        : 1;
  __REG32 PACLK8        : 1;
  __REG32 PACLK9        : 1;
  __REG32 PACLK10       : 1;
  __REG32 PACLK11       : 1;
  __REG32 PACLK12       : 1;
  __REG32 PACLK13       : 1;
  __REG32 PACLK14       : 1;
  __REG32 PACLK15       : 1;
  __REG32               :16;
} __crpa_bits;

/*APB (B) bus clock gate control register (CRPB)*/
typedef struct {
  __REG32 PBCLK0        : 1;
  __REG32 PBCLK1        : 1;
  __REG32 PBCLK2        : 1;
  __REG32 PBCLK3        : 1;
  __REG32 PBCLK4        : 1;
  __REG32 PBCLK5        : 1;
  __REG32 PBCLK6        : 1;
  __REG32 PBCLK7        : 1;
  __REG32 PBCLK8        : 1;
  __REG32 PBCLK9        : 1;
  __REG32 PBCLK10       : 1;
  __REG32 PBCLK11       : 1;
  __REG32 PBCLK12       : 1;
  __REG32 PBCLK13       : 1;
  __REG32 PBCLK14       : 1;
  __REG32 PBCLK15       : 1;
  __REG32               :16;
} __crpb_bits;

/*AHB (B) bus clock gate control register (CRHB)*/
typedef struct {
  __REG32 HBCLK0        : 1;
  __REG32 HBCLK1        : 1;
  __REG32 HBCLK2        : 1;
  __REG32 HBCLK3        : 1;
  __REG32 HBCLK4        : 1;
  __REG32 HBCLK5        : 1;
  __REG32 HBCLK6        : 1;
  __REG32 HBCLK7        : 1;
  __REG32 HBCLK8        : 1;
  __REG32 HBCLK9        : 1;
  __REG32 HBCLK10       : 1;
  __REG32 HBCLK11       : 1;
  __REG32 HBCLK12       : 1;
  __REG32 HBCLK13       : 1;
  __REG32 HBCLK14       : 1;
  __REG32 HBCLK15       : 1;
  __REG32               :16;
} __crhb_bits;

/*ARM core clock gate control register (CRAM)*/
typedef struct {
  __REG32 ARMAGATE      : 1;
  __REG32               : 3;
  __REG32 ARMBGATE      : 1;
  __REG32               :27;
} __cram_bits;

/*Remap control register (RBREMAP)*/
typedef struct {
  __REG32 REMAP         : 1;
  __REG32               :31;
} __rbremap_bits;

/*VINITHI control register A (RBVIHA)*/
typedef struct {
  __REG32 VIHA          : 1;
  __REG32               :31;
} __rbviha_bits;

/*INITRAM control register A (RBITRA)*/
typedef struct {
  __REG32 ITRA          : 1;
  __REG32               :31;
} __rbitra_bits;

/*IRQ flag register (IR0IRQF/ IR1IRQF)*/
typedef struct {
  __REG32 IRQF          : 1;
  __REG32               :31;
} __irxirqf_bits;

/*IRQ mask register (IR0IRQM/IR1IRQM)*/
typedef struct {
  __REG32 IRQM          : 1;
  __REG32               :31;
} __irxirqm_bits;

/*Interrupt level mask register (IR0ILM/IR1ILM)*/
typedef struct {
  __REG32 ILM           : 4;
  __REG32               :28;
} __irxilm_bits;

/*ICR monitoring register (IR0ICRMN/IR1ICRMN)*/
typedef struct {
  __REG32 ILM           : 4;
  __REG32               :28;
} __irxicrmn_bits;

/*Delay interrupt control register 0 (IR0DICR0)*/
typedef struct {
  __REG32 DLYI0         : 1;
  __REG32               :31;
} __ir0dicr0_bits;

/*Delay interrupt control register 1 (IR0DICR1)*/
typedef struct {
  __REG32 DLYI1         : 1;
  __REG32               :31;
} __ir0dicr1_bits;

/*Table base register (IR0TBR/IR1TBR)*/
typedef struct {
  __REG32               : 8;
  __REG32 TBR           :24;
} __irxtbr_bits;

/*Interrupt control register (IR0ICR31/IR1ICR31 – IR0ICR00/IR1ICR00)*/
typedef struct {
  __REG32 ICR           : 4;
  __REG32               :28;
} __IRxICRy_bits;

/*SRAM/Flash mode register 0-7 (MCFMODE0-7)*/
typedef struct {
  __REG32 WDTH          : 2;
  __REG32               : 3;
  __REG32 PAGE          : 1;
  __REG32 RDY           : 1;
  __REG32               :25;
} __mcfmodex_bits;

/*SRAM/Flash timing register 0-7 (MCFTIM0-7)*/
typedef struct {
  __REG32 RACC          : 4;
  __REG32 RADC          : 4;
  __REG32 FRADC         : 4;
  __REG32 RIDLC         : 4;
  __REG32 WACC          : 4;
  __REG32 WADC          : 4;
  __REG32 WWEC          : 4;
  __REG32 WIDLC         : 4;
} __mcftimx_bits;

/*SRAM/Flash area register 0-7 (MCFAREA0-7)*/
typedef struct {
  __REG32 ADDR          : 8;
  __REG32               : 8;
  __REG32 MASK          : 7;
  __REG32               : 9;
} __mcfareax_bits;

/*Memory controller error register (MCERR)*/
typedef struct {
  __REG32 SFER          : 1;
  __REG32               : 1;
  __REG32 SFION         : 1;
  __REG32               :29;
} __mcerr_bits;

/*DRAM initialization control register (DRIC)*/
typedef struct {
  __REG16 DRCMD         : 1;
  __REG16 CMDRDY        : 1;
  __REG16 DDRBSY        : 1;
  __REG16 REFBSY        : 1;
  __REG16               :10;
  __REG16 CKEN          : 1;
  __REG16 DRINI         : 1;
} __dric_bits;

/*DRAM initialization command register [1] (DRIC1)*/
typedef struct {
  __REG16 BA0           : 1;
  __REG16 BA1           : 1;
  __REG16 BA2           : 1;
  __REG16 _WE           : 1;
  __REG16 _CAS          : 1;
  __REG16 _RAS          : 1;
  __REG16 _CS           : 1;
  __REG16               : 9;
} __dric1_bits;

/*DRAM initialization command register [2] (DRIC2)*/
typedef struct {
  __REG16 A0            : 1;
  __REG16 A1            : 1;
  __REG16 A2            : 1;
  __REG16 A3            : 1;
  __REG16 A4            : 1;
  __REG16 A5            : 1;
  __REG16 A6            : 1;
  __REG16 A7            : 1;
  __REG16 A8            : 1;
  __REG16 A9            : 1;
  __REG16 A10           : 1;
  __REG16 A11           : 1;
  __REG16 A12           : 1;
  __REG16 A13           : 1;
  __REG16 A14           : 1;
  __REG16 A15           : 1;
} __dric2_bits;

/*DRAM CTRL ADD register (DRCA)*/
typedef struct {
  __REG16 ColRange      : 4;
  __REG16 RowRange      : 4;
  __REG16 BankRange     : 2;
  __REG16               : 3;
  __REG16 Bus16         : 1;
  __REG16 TYPE          : 2;
} __drca_bits;

/*DRAM control mode register (DRCM)*/
typedef struct {
  __REG16 BL            : 3;
  __REG16               : 1;
  __REG16 CL            : 3;
  __REG16               : 1;
  __REG16 AL            : 3;
  __REG16               : 1;
  __REG16 BT            : 1;
  __REG16               : 3;
} __drcm_bits;

/*DRAM CTRL SET TIME1 Register (DRCST1)*/
typedef struct {
  __REG16 TRC           : 4;
  __REG16 TRP           : 3;
  __REG16               : 1;
  __REG16 TRAS          : 3;
  __REG16               : 1;
  __REG16 TRCD          : 3;
  __REG16               : 1;
} __drcst1_bits;

/*DRAM CTRL SET TIME2 register (DRCST2)*/
typedef struct {
  __REG16 TWR           : 3;
  __REG16               : 1;
  __REG16 TRRD          : 2;
  __REG16               : 2;
  __REG16 TRFC          : 4;
  __REG16               : 4;
} __drcst2_bits;

/*DRAM CTRL REFRESH register (DRCR)*/
typedef struct {
  __REG16 REF_CNT       : 8;
  __REG16 CNTLD         : 1;
  __REG16               : 7;
} __drcr_bits;

/*DRAM CTRL FIFO register (DRCF)*/
typedef struct {
  __REG16 FIFO_CNT      : 5;
  __REG16               :10;
  __REG16 FIFO_ARB      : 1;
} __drcf_bits;

/*AXI setting register (DRASR)*/
typedef struct {
  __REG16 CACHE         : 1;
  __REG16               :15;
} __drasr_bits;

/*DRAM IF MACRO SETTING DLL register (DRIMSD)*/
typedef struct {
  __REG16 ISFT_0        : 3;
  __REG16               : 1;
  __REG16 ISFT_1        : 3;
  __REG16               : 1;
  __REG16 ISFT_2        : 3;
  __REG16               : 1;
  __REG16 ISFT_3        : 3;
  __REG16               : 1;
} __drimsd_bits;

/*DRAM ODT SETTING register (DROS)*/
typedef struct {
  __REG16 ODT0          : 1;
  __REG16               :15;
} __dros_bits;

/*IO buffer setting ODT1 (DRIBSODT1)*/
typedef struct {
  __REG16 ODTON         : 1;
  __REG16 ZSEL          : 1;
  __REG16 ODTONP        : 1;
  __REG16 ZSELP         : 1;
  __REG16 ODTONN        : 1;
  __REG16 ZSELN         : 1;
  __REG16               :10;
} __dribsodt1_bits;

/*IO buffer setting OCD (DRIBSOCD)*/
typedef struct {
  __REG16 OCDCNT        : 1;
  __REG16 DIMMCAL       : 1;
  __REG16 OCDPOL        : 1;
  __REG16 ADRV          : 1;
  __REG16 AFORCE        : 1;
  __REG16               :11;
} __dribsocd_bits;

/*IO buffer setting OCD2 (DRIBSOCD2)*/
typedef struct {
  __REG16 SSEL          : 1;
  __REG16 SUSPR         : 1;
  __REG16 SUSPD         : 1;
  __REG16               :13;
} __dribsocd2_bits;

/*ODT auto bias adjust register (DROABA)*/
typedef struct {
  __REG16 ODTBIAS       : 2;
  __REG16 IAVSET        : 2;
  __REG16               : 3;
  __REG16 OCOMPPPOL     : 1;
  __REG16 OCOMPNPOL     : 1;
  __REG16               : 7;
} __droaba_bits;

/*ODT bias select register (DROBS)*/
typedef struct {
  __REG16 AUTO          : 1;
  __REG16               :15;
} __drobs_bits;

/*IO monitor register 1 (DRIMR1)*/
typedef struct {
  __REG16 DQX0_15       :16;
} __drimr1_bits;

/*IO monitor register 2 (DRIMR2)*/
typedef struct {
  __REG16 DQX16_31      :16;
} __drimr2_bits;

/*IO monitor register 2 (DRIMR2)*/
typedef struct {
  __REG16 DQSX          : 4;
  __REG16               :12;
} __drimr3_bits;

/*IO monitor register 4 (DRIMR4)*/
typedef struct {
  __REG16 DMX           : 4;
  __REG16               :12;
} __drimr4_bits;

/*OCD impedance setting Rrgister1 (DROISR1)*/
typedef struct {
  __REG16 DRVP1         : 4;
  __REG16 DRVN1         : 4;
  __REG16 DRVP2         : 4;
  __REG16 DRVN2         : 4;
} __droisr1_bits;

/*OCD impedance setting Rrgister2 (DROISR2)*/
typedef struct {
  __REG16 DRVP3         : 4;
  __REG16 DRVN3         : 4;
  __REG16 DRVP4         : 4;
  __REG16 DRVN4         : 4;
} __droisr2_bits;

/*DMA configuration register (DMACR)*/
typedef struct {
  __REG32               :24;
  __REG32 DH            : 4;
  __REG32 PR            : 1;
  __REG32               : 1;
  __REG32 DS            : 1;
  __REG32 DE            : 1;
} __dmacr_bits;

/*DMA configuration A register (DMACAx)*/
typedef struct {
  __REG32 TC            :16;
  __REG32 BC            : 4;
  __REG32 BT            : 4;
  __REG32 IS            : 5;
  __REG32 ST            : 1;
  __REG32 PB            : 1;
  __REG32 EB            : 1;
} __dmacax_bits;

/*DMA configuration B register (DMACBx)*/
typedef struct {
  __REG32               : 8;
  __REG32 DP            : 4;
  __REG32 SP            : 4;
  __REG32 SS            : 3;
  __REG32 CI            : 1;
  __REG32 EI            : 1;
  __REG32 RD            : 1;
  __REG32 RS            : 1;
  __REG32 RC            : 1;
  __REG32 FD            : 1;
  __REG32 FS            : 1;
  __REG32 TW            : 2;
  __REG32 MS            : 2;
  __REG32 TT            : 2;
} __dmacbx_bits;

/*Port data register 0 (GPDR0)*/
typedef struct {
  __REG32 GPIO_PD0      : 1;
  __REG32 GPIO_PD1      : 1;
  __REG32 GPIO_PD2      : 1;
  __REG32 GPIO_PD3      : 1;
  __REG32 GPIO_PD4      : 1;
  __REG32 GPIO_PD5      : 1;
  __REG32 GPIO_PD6      : 1;
  __REG32 GPIO_PD7      : 1;
  __REG32               :24;
} __gpdr0_bits;

/*Port data register 1 (GPDR1)*/
typedef struct {
  __REG32 GPIO_PD8      : 1;
  __REG32 GPIO_PD9      : 1;
  __REG32 GPIO_PD10     : 1;
  __REG32 GPIO_PD11     : 1;
  __REG32 GPIO_PD12     : 1;
  __REG32 GPIO_PD13     : 1;
  __REG32 GPIO_PD14     : 1;
  __REG32 GPIO_PD15     : 1;
  __REG32               :24;
} __gpdr1_bits;

/*Port data register 2 (GPDR2)*/
typedef struct {
  __REG32 GPIO_PD16     : 1;
  __REG32 GPIO_PD17     : 1;
  __REG32 GPIO_PD18     : 1;
  __REG32 GPIO_PD19     : 1;
  __REG32 GPIO_PD20     : 1;
  __REG32 GPIO_PD21     : 1;
  __REG32 GPIO_PD22     : 1;
  __REG32 GPIO_PD23     : 1;
  __REG32               :24;
} __gpdr2_bits;

/*PWMx base clock register (PWMxBCR)*/
typedef struct {
  __REG32 BCR           :16;
  __REG32               :16;
} __PWMxBCR_bits;

/*PWMx pulse width register (PWMxTPR)*/
typedef struct {
  __REG32 TPR           :16;
  __REG32               :16;
} __PWMxTPR_bits;

/*PWMx phase register (PWMxPR)*/
typedef struct {
  __REG32 PR            :16;
  __REG32               :16;
} __PWMxPR_bits;

/*PWMx duty register (PWMxDR)*/
typedef struct {
  __REG32 DR            :16;
  __REG32               :16;
} __PWMxDR_bits;

/*PWMx status register (PWMxCR)*/
typedef struct {
  __REG32 POL           : 1;
  __REG32               : 2;
  __REG32 ONESHOT       : 1;
  __REG32               :28;
} __PWMxCR_bits;

/*PWMx start register (PWMxSR)*/
typedef struct {
  __REG32 START         : 1;
  __REG32               :31;
} __PWMxSR_bits;

/*PWMx current count register (PWMxCCR)*/
typedef struct {
  __REG32 CCR           :16;
  __REG32               :16;
} __PWMxCCR_bits;

/*PWMx interrupt register (PWMxIR)*/
typedef struct {
  __REG32 DONE          : 1;
  __REG32               : 7;
  __REG32 INTREP        : 2;
  __REG32               :22;
} __PWMxIR_bits;

/*ADCx data register (ADCxDATA)*/
typedef struct {
  __REG32 DATA          :10;
  __REG32               :22;
} __ADCxDATA_bits;

/*ADCx power down control register (ADCxXPD)*/
typedef struct {
  __REG32 XPD           : 1;
  __REG32               :31;
} __ADCxXPD_bits;

/*ADCx clock selection register (ADCxCKSEL)*/
typedef struct {
  __REG32 CKSEL         : 3;
  __REG32               :29;
} __ADCxCKSEL_bits;

/*ADCx status register (ADCxSTATUS)*/
typedef struct {
  __REG32 CMP           : 1;
  __REG32               :31;
} __ADCxSTATUS_bits;

/*I2SxCNTREG register*/
typedef struct {
  __REG32 FSPL          : 1;
  __REG32 FSLN          : 1;
  __REG32 FSPH          : 1;
  __REG32 CPOL          : 1;
  __REG32 SMPL          : 1;
  __REG32 RXDIS         : 1;
  __REG32 TXDIS         : 1;
  __REG32 MLSB          : 1;
  __REG32 FRUN          : 1;
  __REG32 BEXT          : 1;
  __REG32 ECKM          : 1;
  __REG32 RHLL          : 1;
  __REG32 SBFN          : 1;
  __REG32 MSMD          : 1;
  __REG32 MSKB          : 1;
  __REG32               : 1;
  __REG32 OVHD          :10;
  __REG32 CKRT          : 6;
} __i2sxcntreg_bits;

/*I2SxMCR0REG register*/
typedef struct {
  __REG32 S0WDL         : 5;
  __REG32 S0CHL         : 5;
  __REG32 S0CHN         : 5;
  __REG32               : 1;
  __REG32 S1WDL         : 5;
  __REG32 S1CHL         : 5;
  __REG32 S1CHN         : 5;
  __REG32               : 1;
} __i2sxmcr0reg_bits;

/*I2SxMCR1REG register*/
typedef struct {
  __REG32 S0CHN00       : 1;
  __REG32 S0CHN01       : 1;
  __REG32 S0CHN02       : 1;
  __REG32 S0CHN03       : 1;
  __REG32 S0CHN04       : 1;
  __REG32 S0CHN05       : 1;
  __REG32 S0CHN06       : 1;
  __REG32 S0CHN07       : 1;
  __REG32 S0CHN08       : 1;
  __REG32 S0CHN09       : 1;
  __REG32 S0CHN10       : 1;
  __REG32 S0CHN11       : 1;
  __REG32 S0CHN12       : 1;
  __REG32 S0CHN13       : 1;
  __REG32 S0CHN14       : 1;
  __REG32 S0CHN15       : 1;
  __REG32 S0CHN16       : 1;
  __REG32 S0CHN17       : 1;
  __REG32 S0CHN18       : 1;
  __REG32 S0CHN19       : 1;
  __REG32 S0CHN20       : 1;
  __REG32 S0CHN21       : 1;
  __REG32 S0CHN22       : 1;
  __REG32 S0CHN23       : 1;
  __REG32 S0CHN24       : 1;
  __REG32 S0CHN25       : 1;
  __REG32 S0CHN26       : 1;
  __REG32 S0CHN27       : 1;
  __REG32 S0CHN28       : 1;
  __REG32 S0CHN29       : 1;
  __REG32 S0CHN30       : 1;
  __REG32 S0CHN31       : 1;
} __i2sxmcr1reg_bits;

/*I2SxMCR2REG register*/
typedef struct {
  __REG32 S1CHN00       : 1;
  __REG32 S1CHN01       : 1;
  __REG32 S1CHN02       : 1;
  __REG32 S1CHN03       : 1;
  __REG32 S1CHN04       : 1;
  __REG32 S1CHN05       : 1;
  __REG32 S1CHN06       : 1;
  __REG32 S1CHN07       : 1;
  __REG32 S1CHN08       : 1;
  __REG32 S1CHN09       : 1;
  __REG32 S1CHN10       : 1;
  __REG32 S1CHN11       : 1;
  __REG32 S1CHN12       : 1;
  __REG32 S1CHN13       : 1;
  __REG32 S1CHN14       : 1;
  __REG32 S1CHN15       : 1;
  __REG32 S1CHN16       : 1;
  __REG32 S1CHN17       : 1;
  __REG32 S1CHN18       : 1;
  __REG32 S1CHN19       : 1;
  __REG32 S1CHN20       : 1;
  __REG32 S1CHN21       : 1;
  __REG32 S1CHN22       : 1;
  __REG32 S1CHN23       : 1;
  __REG32 S1CHN24       : 1;
  __REG32 S1CHN25       : 1;
  __REG32 S1CHN26       : 1;
  __REG32 S1CHN27       : 1;
  __REG32 S1CHN28       : 1;
  __REG32 S1CHN29       : 1;
  __REG32 S1CHN30       : 1;
  __REG32 S1CHN31       : 1;
} __i2sxmcr2reg_bits;

/*I2SxOPRREG register*/
typedef struct {
  __REG32 START         : 1;
  __REG32               :15;
  __REG32 TXENB         : 1;
  __REG32               : 7;
  __REG32 RXENB         : 1;
  __REG32               : 7;
} __i2sxoprreg_bits;

/*I2SxSRST register*/
typedef struct {
  __REG32 SRST          : 1;
  __REG32               :31;
} __i2sxsrst_bits;

/*I2SxINTCNT register*/
typedef struct {
  __REG32 RFTH          : 4;
  __REG32 RPTMR         : 2;
  __REG32               : 2;
  __REG32 TFTH          : 4;
  __REG32               : 4;
  __REG32 RXFIM         : 1;
  __REG32 RXFDM         : 1;
  __REG32 EOPM          : 1;
  __REG32 RXOVM         : 1;
  __REG32 RXUDM         : 1;
  __REG32 RBERM         : 1;
  __REG32               : 2;
  __REG32 TXFIM         : 1;
  __REG32 TXFDM         : 1;
  __REG32 TXOVM         : 1;
  __REG32 TXUD0M        : 1;
  __REG32 FERRM         : 1;
  __REG32 TBERM         : 1;
  __REG32 TXUD1M        : 1;
  __REG32               : 1;
} __i2sxintcnt_bits;

/*I2SxSTATUS register*/
typedef struct {
  __REG32 RXNUM         : 6;
  __REG32               : 2;
  __REG32 TXNUM         : 6;
  __REG32               : 2;
  __REG32 RXFI          : 1;
  __REG32 TXFI          : 1;
  __REG32 BSY           : 1;
  __REG32 EOPI          : 1;
  __REG32               : 4;
  __REG32 RXOVR         : 1;
  __REG32 RXUDR         : 1;
  __REG32 TXOVR         : 1;
  __REG32 TXUDR0        : 1;
  __REG32 TXUDR1        : 1;
  __REG32 FERR          : 1;
  __REG32 RBERR         : 1;
  __REG32 TBERR         : 1;
} __i2sxstatus_bits;

/*I2SxDMAACT register*/
typedef struct {
  __REG32 RDMACT        : 1;
  __REG32               :15;
  __REG32 TDMACT        : 1;
  __REG32               :15;
} __i2sxdmaact_bits;

/*Interrupt enable register (URTxIER)*/
typedef struct {
  __REG32 ERBFI         : 1;
  __REG32 ETBEI         : 1;
  __REG32 ELSI          : 1;
  __REG32 EDSSI         : 1;
  __REG32               :28;
} __urtxier_bits;

/*URTxIIR & URTxFCR registers*/
typedef union {
  /*URTxIIR*/
  struct {
    __REG32 NINT        : 1;
    __REG32 ID          : 3;
    __REG32             : 2;
    __REG32 FIFOST      : 2;
    __REG32             :24;
  };
  /*URTxFCR*/
  struct {
    __REG32             : 1;
    __REG32 RxF_RST     : 1;
    __REG32 TxF_RST     : 1;
    __REG32 DMA_MODE    : 1;
    __REG32             : 2;
    __REG32 RCVR        : 2;
    __REG32             :24;
  };
} __urtxiir_fcr_bits;

/*Line control register (URTxLCR)*/
typedef struct {
  __REG32 WLS           : 2;
  __REG32 STB           : 1;
  __REG32 PEN           : 1;
  __REG32 EPS           : 1;
  __REG32 SP            : 1;
  __REG32 SB            : 1;
  __REG32 DLAB          : 1;
  __REG32               :24;
} __urtxlcr_bits;

/*Modem control register (URTxMCR)*/
typedef struct {
  __REG32 DTR           : 1;
  __REG32 RTS           : 1;
  __REG32 OUT1          : 1;
  __REG32 OUT2          : 1;
  __REG32 LOOP          : 1;
  __REG32               :27;
} __urtxmcr_bits;

/*Line status register (URTxLSR)*/
typedef struct {
  __REG32 DR            : 1;
  __REG32 OE            : 1;
  __REG32 PE            : 1;
  __REG32 FE            : 1;
  __REG32 BI            : 1;
  __REG32 THRE          : 1;
  __REG32 TEMT          : 1;
  __REG32 ERRF          : 1;
  __REG32               :24;
} __urtxlsr_bits;

/*Modem status register (URTxMSR)*/
typedef struct {
  __REG32 DCTS          : 1;
  __REG32 DDSR          : 1;
  __REG32 TERI          : 1;
  __REG32 DDCD          : 1;
  __REG32 CTS           : 1;
  __REG32 DSR           : 1;
  __REG32 RI            : 1;
  __REG32 DCD           : 1;
  __REG32               :24;
} __urtxmsr_bits;

/*Bus status register (I2CxBSR)*/
typedef struct {
  __REG32 FBT           : 1;
  __REG32 GCA           : 1;
  __REG32 AAS           : 1;
  __REG32 TRX           : 1;
  __REG32 LRB           : 1;
  __REG32 AL            : 1;
  __REG32 RSC           : 1;
  __REG32 BB            : 1;
  __REG32               :24;
} __i2cxbsr_bits;

/*Bus control register (I2CxBCR)*/
typedef struct {
  __REG32 INT           : 1;
  __REG32 INTE          : 1;
  __REG32 GCAA          : 1;
  __REG32 ACK           : 1;
  __REG32 MSS           : 1;
  __REG32 SCC           : 1;
  __REG32 BEIE          : 1;
  __REG32 BER           : 1;
  __REG32               :24;
} __i2cxbcr_bits;

/*Clock control register (I2CxCCR)*/
typedef struct {
  __REG32 CS            : 5;
  __REG32 EN            : 1;
  __REG32 HSM           : 1;
  __REG32               :25;
} __i2cxccr_bits;

/*Address register (I2CxADR)*/
typedef struct {
  __REG32 A             : 7;
  __REG32               :25;
} __i2cxadr_bits;

/*Data register (I2CxDAR)*/
typedef struct {
  __REG32 D             : 8;
  __REG32               :24;
} __i2cxdar_bits;

/*Expansion CS register (I2CxECSR)*/
typedef struct {
  __REG32 CS            : 6;
  __REG32               :26;
} __i2cxecsr_bits;

/*Bus clock frequency register (I2CxBCFR)*/
typedef struct {
  __REG32 FS            : 4;
  __REG32               :28;
} __i2cxbcfr_bits;

/*Two bus control registers (I2CxBC2R)*/
typedef struct {
  __REG32 SCLL          : 1;
  __REG32 SDAL          : 1;
  __REG32               : 2;
  __REG32 SCLS          : 1;
  __REG32 SDAS          : 1;
  __REG32               :26;
} __i2cxbc2r_bits;

/*SPI control register (SPICR)*/
typedef struct {
  __REG32 CPHA          : 1;
  __REG32 CPOL          : 1;
  __REG32               : 6;
  __REG32 CDV           : 3;
  __REG32               : 5;
  __REG32 SPL0          : 1;
  __REG32               :15;
} __spicr_bits;

/*SPI slave control register (SPISCR)*/
typedef struct {
  __REG32 SSP           : 2;
  __REG32               : 2;
  __REG32 SAUT          : 1;
  __REG32 SMOD          : 1;
  __REG32               : 2;
  __REG32 DLN           : 5;
  __REG32               : 3;
  __REG32 STL           : 4;
  __REG32               : 4;
  __REG32 DRVS          : 1;
  __REG32               : 3;
  __REG32 SPE           : 1;
  __REG32               : 3;
} __spiscr_bits;

/*SPI status register (SPISR)*/
typedef struct {
  __REG32 SENB          : 1;
  __REG32 SBSY          : 1;
  __REG32 SERR          : 1;
  __REG32               : 4;
  __REG32 SIRQ          : 1;
  __REG32               :24;
} __spisr_bits;

/*Capability Register (HCCAPBASE)*/
typedef struct {
  __REG32 CAPLENGTH     : 8;
  __REG32               : 8;
  __REG32 HCIVERSION    :16;
} __hccapbase_bits;

/*CStructural Parameter Register (HCSPARAMS)*/
typedef struct {
  __REG32 N_PORTS       : 4;
  __REG32 PPC           : 1;
  __REG32               : 2;
  __REG32 PRR           : 1;
  __REG32 N_PCC         : 4;
  __REG32 N_CC          : 4;
  __REG32 PI            : 1;
  __REG32               : 3;
  __REG32 N_DP          : 4;
  __REG32               : 8;
} __hcsparams_bits;

/*Capability Parameter Register (HCCPARAMS)*/
typedef struct {
  __REG32 _64_ADC       : 1;
  __REG32 PFL           : 1;
  __REG32 ASP           : 1;
  __REG32               : 1;
  __REG32 IST           : 4;
  __REG32 EECP          : 8;
  __REG32               :16;
} __hccparams_bits;

/*USB Command Register (USBCMD)*/
typedef struct{
  __REG32 RS           : 1;
  __REG32 RESET        : 1;
  __REG32 FS           : 2;
  __REG32 PSE          : 1;
  __REG32 ASE          : 1;
  __REG32 IAA          : 1;
  __REG32 LR           : 1;
  __REG32 ASP          : 2;
  __REG32              : 1;
  __REG32 ASPE         : 1;
  __REG32              : 4;
  __REG32 ITC          : 8;
  __REG32              : 8;
} __usbcmd_bits;

/*USB Status Register (USBSTS)*/
typedef struct{
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

/*USB Interrupt Enable Register (USBINTR)*/
typedef struct{
  __REG32 UE                : 1;
  __REG32 UEE               : 1;
  __REG32 PCE               : 1;
  __REG32 FRE               : 1;
  __REG32 SEE               : 1;
  __REG32 AAE               : 1;
  __REG32                   :26;
} __usbintr_bits;

/*USB Frame Index Register (FRINDEX)*/
typedef struct{
  __REG32 FRINDEX           :14;
  __REG32                   :18;
} __frindex_bits;

/*Periodic Frame List Base Address Register (PERIODICLISTBASE)*/
typedef struct{
  __REG32                   :12;
  __REG32 BASEADDRESS       :20;
} __periodiclistbase_bits;

/*Periodic Frame List Base Address Register (PERIODICLISTBASE)*/
typedef struct{
  __REG32                   : 5;
  __REG32 LINKPOINTERLOW    :27;
} __asynclistaddr_bits;

/*Configured Flag Register (CONFIGFLAG)*/
typedef struct {
  __REG32 CF                : 1;
  __REG32                   :31;
} __configflag_bits;

/*Port Status/Control Register 1 (PORTSC_1)*/
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
} __portsc_1_bits;

/*Programmable Microframe Base Value Register (INSNREG00)*/
typedef struct {
  __REG32 EN                : 1;
  __REG32 VALUE             :13;
  __REG32                   :18;
} __insnreg00_bits;

/*Programmable Packet Buffer OUT/IN Threshold Register (INSNREG01)*/
typedef struct {
  __REG32 IN_TH             :16;
  __REG32 OUT_TH            :16;
} __insnreg01_bits;

/*Programmable Packet Buffer Depth Register (INSNREG02)*/
typedef struct {
  __REG32 DEPTH             :12;
  __REG32                   :20;
} __insnreg02_bits;

/*Time Available Offset Register (INSNREG03)*/
typedef struct {
  __REG32                   : 1;
  __REG32 TAO               : 8;
  __REG32                   :23;
} __insnreg03_bits;

/* HcRevision Register */
typedef struct{
__REG32 REV  : 8;
__REG32      :24;
} __HcRevision_bits;

/* HcControl Register */
typedef struct{
__REG32 CBSR  : 2;
__REG32 PLE   : 1;
__REG32 IE    : 1;
__REG32 CLE   : 1;
__REG32 BLE   : 1;
__REG32 HCFS  : 2;
__REG32 IR    : 1;
__REG32 RWC   : 1;
__REG32 RWE   : 1;
__REG32       :21;
} __HcControl_bits;

/* HcCommandStatus Register */
typedef struct{
__REG32 HCR  : 1;
__REG32 CLF  : 1;
__REG32 BLF  : 1;
__REG32 OCR  : 1;
__REG32      :12;
__REG32 SOC  : 2;
__REG32      :14;
} __HcCommandStatus_bits;

/* HcInterruptStatus Register */
typedef struct{
__REG32 SO    : 1;
__REG32 WDH   : 1;
__REG32 SF    : 1;
__REG32 RD    : 1;
__REG32 UE    : 1;
__REG32 FNO   : 1;
__REG32 RHSC  : 1;
__REG32       :23;
__REG32 OC    : 1;
__REG32       : 1;
} __HcInterruptStatus_bits;

/* HcInterruptEnable Register
   HcInterruptDisable Register */
typedef struct{
__REG32 SO    : 1;
__REG32 WDH   : 1;
__REG32 SF    : 1;
__REG32 RD    : 1;
__REG32 UE    : 1;
__REG32 FNO   : 1;
__REG32 RHSC  : 1;
__REG32       :23;
__REG32 OC    : 1;
__REG32 MIE   : 1;
} __HcInterruptEnable_bits;

/* HcHCCA Register */
typedef struct{
__REG32       : 8;
__REG32 HCCA  :24;
} __HcHCCA_bits;

/* HcPeriodCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 PCED  :28;
} __HcPeriodCurrentED_bits;

/* HcControlHeadED Registerr */
typedef struct{
__REG32       : 4;
__REG32 CHED  :28;
} __HcControlHeadED_bits;

/* HcControlCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 CCED  :28;
} __HcControlCurrentED_bits;

/* HcBulkHeadED Register */
typedef struct{
__REG32       : 4;
__REG32 BHED  :28;
} __HcBulkHeadED_bits;

/* HcBulkCurrentED Register */
typedef struct{
__REG32       : 4;
__REG32 BCED  :28;
} __HcBulkCurrentED_bits;

/* HcDoneHead Register */
typedef struct{
__REG32     : 4;
__REG32 DH  :28;
} __HcDoneHead_bits;

/* HcFmInterval Register */
typedef struct{
__REG32 FI     :14;
__REG32        : 2;
__REG32 FSMPS  :15;
__REG32 FIT    : 1;
} __HcFmInterval_bits;

/* HcFmRemaining Register */
typedef struct{
__REG32 FR   :14;
__REG32      :17;
__REG32 FRT  : 1;
} __HcFmRemaining_bits;

/* HcFmNumber Register */
typedef struct{
__REG32 FN  :16;
__REG32     :16;
} __HcFmNumber_bits;

/* HcPeriodicStart Register */
typedef struct{
__REG32 PS  :14;
__REG32     :18;
} __HcPeriodicStart_bits;

/* HcLSThreshold Register */
typedef struct{
__REG32 LST  :12;
__REG32      :20;
} __HcLSThreshold_bits;

/* HcRhDescriptorA Register */
typedef struct{
__REG32 NDP     : 8;
__REG32 PSM     : 1;     /* ??*/
__REG32 NPS     : 1;     /* ??*/
__REG32 DT      : 1;
__REG32 OCPM    : 1;
__REG32 NOCP    : 1;
__REG32         :11;
__REG32 POTPGT  : 8;
} __HcRhDescriptorA_bits;

/* HcRhDescriptorB Register */
typedef struct{
__REG32 DR      :16;
__REG32 PPCM    :16;
} __HcRhDescriptorB_bits;

/* HcRhStatus Register */
typedef struct{
__REG32 LPS     : 1;
__REG32 OCI     : 1;
__REG32         :13;
__REG32 DRWE    : 1;
__REG32 LPSC    : 1;
__REG32 CCIC    : 1;
__REG32         :13;
__REG32 CRWE    : 1;
} __HcRhStatus_bits;

/* HcRhPortStatus[1:2] Register */
typedef struct{
__REG32 CCS     : 1;
__REG32 PES     : 1;
__REG32 PSS     : 1;
__REG32 POCI    : 1;
__REG32 PRS     : 1;
__REG32         : 3;
__REG32 PPS     : 1;
__REG32 LSDA    : 1;
__REG32         : 6;
__REG32 CSC     : 1;
__REG32 PESC    : 1;
__REG32 PSSC    : 1;
__REG32 OCIC    : 1;
__REG32 PRSC    : 1;
__REG32         :11;
} __HcRhPortStatus_bits;

/*Link Mode Setting Register (LinkModeSetting)*/
typedef struct{
__REG32 LMF         : 2;
__REG32 LME         : 6;
__REG32 LMD         : 8;
__REG32 LMC         : 6;
__REG32 LMB         : 2;
__REG32 LMA         : 8;
} __linkmodesetting_bits;

/*PHY Mode Setting 1 Register (PHYModeSetting1)*/
typedef struct{
__REG32 PM1D        : 1;
__REG32 PM1C        :15;
__REG32 PM1B        : 6;
__REG32             : 2;
__REG32 RPDPEN      : 1;
__REG32 RPDMEN      : 1;
__REG32 PM1A        : 2;
__REG32             : 4;
} __phymodesetting1_bits;

/*PHY Mode Setting 2 Register (PHYModeSetting2)*/
typedef struct{
__REG32 LEOP        : 3;
__REG32             : 5;
__REG32 COMPDISCON  : 7;
__REG32             : 1;
__REG32 SQSEL       : 2;
__REG32             :14;
} __phymodesetting2_bits;

/*USB Function CPU Access Control Register (UFCpAC)*/
typedef struct{
  __REG32 CBW           : 2;
  __REG32 BO            : 1;
  __REG32 SR            : 1;
  __REG32               :12;
  __REG32 CFWE          : 1;
  __REG32               :15;
} __ufcpac_bits;

/*USB Function Device Control Register (UFDvC)*/
typedef struct{
  __REG32 ReqSpeed      : 2;
  __REG32 ReqResume     : 1;
  __REG32 EnRmtWkUp     : 1;
  __REG32 SelfPower     : 1;
  __REG32 DisConnect    : 1;
  __REG32               : 8;
  __REG32 PhySusp       : 1;
  __REG32               : 2;
  __REG32 LpBkPHY       : 1;
  __REG32 P_MODE        : 1;
  __REG32 L_MODE        : 1;
  __REG32               : 4;
  __REG32 MskSuspende   : 1;
  __REG32 MskSuspendb   : 1;
  __REG32 MskSof        : 1;
  __REG32 MskSetup      : 1;
  __REG32 MskUsbRste    : 1;
  __REG32 MskUsbRstb    : 1;
  __REG32 MskSetConf    : 1;
  __REG32 MskErraticErr : 1;
} __ufdvc_bits;

/*USB Function Device Status Register (UFDvS)*/
typedef struct{
  __REG32               : 8;
  __REG32 Suspend       : 1;
  __REG32 BusReset      : 1;
  __REG32               : 5;
  __REG32 PhyReset      : 1;
  __REG32 CrtSpeed      : 2;
  __REG32               : 2;
  __REG32 Conf          : 4;
  __REG32 IntSuspende   : 1;
  __REG32 IntSuspendb   : 1;
  __REG32 IntSof        : 1;
  __REG32 IntSetup      : 1;
  __REG32 IntUsbRste    : 1;
  __REG32 IntUsbRstb    : 1;
  __REG32 IntSetConf    : 1;
  __REG32 IntErraticErr : 1;
} __ufdvs_bits;

/*USB Function Endpoint Interrupt Control Register (UFEpIC)*/
typedef struct{
  __REG32 MskEp0        : 1;
  __REG32 MskEp1        : 1;
  __REG32 MskEp2        : 1;
  __REG32 MskEp3        : 1;
  __REG32               :28;
} __ufepic_bits;

/*USB Function Endpoint Interrupt Status Register (UFEpIS)*/
typedef struct{
  __REG32 IntEp0        : 1;
  __REG32 IntEp1        : 1;
  __REG32 IntEp2        : 1;
  __REG32 IntEp3        : 1;
  __REG32               :28;
} __ufepis_bits;

/*USB Function Endpoint DMA Control Register (UFEpDC)*/
typedef struct{
  __REG32               : 1;
  __REG32 MskDmaReq1    : 1;
  __REG32 MskDmaReq2    : 1;
  __REG32               :14;
  __REG32 DmaMode1      : 1;
  __REG32 DmaMode2      : 1;
  __REG32               :13;
} __ufepdc_bits;

/*USB Function Endpoint DMA Status Register (UFEpDS)*/
typedef struct{
  __REG32               : 1;
  __REG32 DmaReq1       : 1;
  __REG32 DmaReq2       : 1;
  __REG32               :29;
} __ufepds_bits;

/*USB Function Time Stamp Register (UFTSTAMP)*/
typedef struct{
  __REG32 TimStamp      :11;
  __REG32               :21;
} __uftstamp_bits;

/*UFEpTCSel Register*/
typedef struct{
  __REG32               : 1;
  __REG32 TCSelUSB1     : 1;
  __REG32 TCSelUSB2     : 1;
  __REG32               :29;
} __ufeptcsel_bits;

/*USB Function Endpoint0 Rx Size Register (UFEpRS0)*/
typedef struct{
  __REG32 Size0o        : 7;
  __REG32 SelTx0o       : 1;
  __REG32 Size0i        : 7;
  __REG32 SelTx0i       : 1;
  __REG32               :16;
} __ufeprs0_bits;

/*USB Function Endpoint1 Rx Size Register (UFEpRS1)*/
typedef struct{
  __REG32 Size1         :11;
  __REG32               : 4;
  __REG32 SelTx1        : 1;
  __REG32               :16;
} __ufeprs1_bits;

/*USB Function Endpoint2 Rx Size Register (UFEpRS2)*/
typedef struct{
  __REG32 Size2         :11;
  __REG32               : 4;
  __REG32 SelTx2        : 1;
  __REG32               :16;
} __ufeprs2_bits;

/*USB Function Endpoint3 Rx Size Register (UFEpRS3)*/
typedef struct{
  __REG32 Size3         :11;
  __REG32               : 4;
  __REG32 SelTx3        : 1;
  __REG32               :16;
} __ufeprs3_bits;

/*UFCusCnt Register*/
typedef struct{
  __REG32 ExtRPU        : 1;
  __REG32               : 7;
  __REG32 EnIniFifo     : 1;
  __REG32               : 7;
  __REG32 SetAdd        : 1;
  __REG32 SetConfig     : 1;
  __REG32 TESTP         : 1;
  __REG32 TESTJ         : 1;
  __REG32 TESTK         : 1;
  __REG32 TESTSe0Nack   : 1;
  __REG32               : 2;
  __REG32 Tadd          : 7;
  __REG32               : 1;
} __ufcuscnt_bits;

/*UFCALB Register*/
typedef struct{
  __REG32 FSCALIB       : 3;
  __REG32               : 1;
  __REG32 HSCALIB       : 3;
  __REG32               :25;
} __ufcalb_bits;

/*UFEpLpBk Register*/
typedef struct{
  __REG32 EpLpBkI0      : 4;
  __REG32 EpLpBkO0      : 4;
  __REG32               :24;
} __ufeplpbk_bits;

/*UFEpLpBk Register*/
typedef struct{
  __REG32 NumAltIntf0   : 3;
  __REG32               : 1;
  __REG32 NumAltIntf1   : 3;
  __REG32               : 1;
  __REG32 NumAltIntf2   : 3;
  __REG32               : 1;
  __REG32 NumAltIntf3   : 3;
  __REG32               : 1;
  __REG32 NumIntf       : 3;
  __REG32               :13;
} __ufintfaltnum_bits;

/*USB Function Endpoint0 Control Register (UFEpC0)*/
typedef struct{
  __REG32 Init0i        : 1;
  __REG32 Init0o        : 1;
  __REG32 ReqStall      : 1;
  __REG32               : 4;
  __REG32 TestMode0     : 1;
  __REG32               : 6;
  __REG32 IniFifo0i     : 1;
  __REG32 IniFifo0o     : 1;
  __REG32 MskReady0i    : 1;
  __REG32 MskReady0o    : 1;
  __REG32 MskPing0o     : 1;
  __REG32               : 2;
  __REG32 MskStalled    : 1;
  __REG32 MskNack       : 1;
  __REG32 MskClStall    : 1;
  __REG32               : 8;
} __ufepc0_bits;

/*USB Function Endpoint0 Control Register (UFEpC0)*/
typedef struct{
  __REG32               : 2;
  __REG32 Stalled       : 1;
  __REG32 Ready0i       : 1;
  __REG32 Ready0o       : 1;
  __REG32               :11;
  __REG32 IntReady0i    : 1;
  __REG32 IntReady0o    : 1;
  __REG32 IntPing0o     : 1;
  __REG32               : 2;
  __REG32 IntStalled    : 1;
  __REG32 IntNack       : 1;
  __REG32 IntClStall    : 1;
  __REG32               : 8;
} __ufeps0_bits;

/*USB Function Endpoint1 Control Register (UFEpC1)*/
typedef struct{
  __REG32 Init1         : 1;
  __REG32 ReqStall1     : 1;
  __REG32               : 1;
  __REG32 IniToggle1    : 1;
  __REG32 IniStall1     : 1;
  __REG32 ToggleDis1    : 1;
  __REG32 StallDis1     : 1;
  __REG32 TestMode1     : 1;
  __REG32 NullResp1     : 1;
  __REG32 NackResp1     : 1;
  __REG32 EnSPR1        : 1;
  __REG32 EnSPDD1       : 1;
  __REG32               : 1;
  __REG32 MskSPR1       : 1;
  __REG32 MskSPDD1      : 1;
  __REG32 IniFifo1      : 1;
  __REG32 MskReady1     : 1;
  __REG32 MskPing1      : 1;
  __REG32 MskAChg1      : 1;
  __REG32 MskDEnd1      : 1;
  __REG32 MskEmpty1     : 1;
  __REG32 MskStalled1   : 1;
  __REG32 MskNack1      : 1;
  __REG32 MskClStall1   : 1;
  __REG32 TestAlt1      : 4;
  __REG32               : 4;
} __ufepc1_bits;

/*USB Function Endpoint1 Status Register (UFEpS1)*/
typedef struct{
  __REG32               : 1;
  __REG32 Stalled1      : 1;
  __REG32 Ready1i       : 1;
  __REG32 Ready1o       : 1;
  __REG32               : 8;
  __REG32 Empty1        : 1;
  __REG32 IntSPR1       : 1;
  __REG32 IntSPDD1      : 1;
  __REG32               : 1;
  __REG32 IntReady1     : 1;
  __REG32 IntPing1      : 1;
  __REG32 IntAChg1      : 1;
  __REG32 IntDEnd1      : 1;
  __REG32 IntEmpty1     : 1;
  __REG32 IntStalled1   : 1;
  __REG32 IntNack1      : 1;
  __REG32 IntClStall1   : 1;
  __REG32 CrtAlt1       : 4;
  __REG32 CrtIntf1      : 4;
} __ufeps1_bits;

/*USB Function Endpoint2 Control Register (UFEpC2)*/
typedef struct{
  __REG32 Init2         : 1;
  __REG32 ReqStall2     : 1;
  __REG32               : 1;
  __REG32 IniToggle2    : 1;
  __REG32 IniStall2     : 1;
  __REG32 ToggleDis2    : 1;
  __REG32 StallDis2     : 1;
  __REG32 TestMode2     : 1;
  __REG32 NullResp2     : 1;
  __REG32 NackResp2     : 1;
  __REG32 EnSPR2        : 1;
  __REG32 EnSPDD2       : 1;
  __REG32               : 1;
  __REG32 MskSPR2       : 1;
  __REG32 MskSPDD2      : 1;
  __REG32 IniFifo2      : 1;
  __REG32 MskReady2     : 1;
  __REG32 MskPing2      : 1;
  __REG32 MskAChg2      : 1;
  __REG32 MskDEnd2      : 1;
  __REG32 MskEmpty2     : 1;
  __REG32 MskStalled2   : 1;
  __REG32 MskNack2      : 1;
  __REG32 MskClStall2   : 1;
  __REG32 TestAlt2      : 4;
  __REG32               : 4;
} __ufepc2_bits;

/*USB Function Endpoint2 Status Register (UFEpS2)*/
typedef struct{
  __REG32               : 1;
  __REG32 Stalled2      : 1;
  __REG32 Ready2i       : 1;
  __REG32 Ready2o       : 1;
  __REG32               : 8;
  __REG32 Empty2        : 1;
  __REG32 IntSPR2       : 1;
  __REG32 IntSPDD2      : 1;
  __REG32               : 1;
  __REG32 IntReady2     : 1;
  __REG32 IntPing2      : 1;
  __REG32 IntAChg2      : 1;
  __REG32 IntDEnd2      : 1;
  __REG32 IntEmpty2     : 1;
  __REG32 IntStalled2   : 1;
  __REG32 IntNack2      : 1;
  __REG32 IntClStall2   : 1;
  __REG32 CrtAlt2       : 4;
  __REG32 CrtIntf2      : 4;
} __ufeps2_bits;

/*USB Function Endpoint3 Control Register (UFEpC3)*/
typedef struct{
  __REG32 Init3         : 1;
  __REG32 ReqStall3     : 1;
  __REG32               : 1;
  __REG32 IniToggle3    : 1;
  __REG32 IniStall3     : 1;
  __REG32 ToggleDis3    : 1;
  __REG32 StallDis3     : 1;
  __REG32 TestMode3     : 1;
  __REG32 NullResp3     : 1;
  __REG32 NackResp3     : 1;
  __REG32 EnSPR3        : 1;
  __REG32 EnSPDD3       : 1;
  __REG32               : 1;
  __REG32 MskSPR3       : 1;
  __REG32 MskSPDD3      : 1;
  __REG32 IniFifo3      : 1;
  __REG32 MskReady3     : 1;
  __REG32 MskPing3      : 1;
  __REG32 MskAChg3      : 1;
  __REG32 MskDEnd3      : 1;
  __REG32 MskEmpty3     : 1;
  __REG32 MskStalled3   : 1;
  __REG32 MskNack3      : 1;
  __REG32 MskClStall3   : 1;
  __REG32 TestAlt3      : 4;
  __REG32               : 4;
} __ufepc3_bits;

/*USB Function Endpoint3 Status Register (UFEpS3)*/
typedef struct{
  __REG32               : 1;
  __REG32 Stalled23      : 1;
  __REG32 Ready3i       : 1;
  __REG32 Ready3o       : 1;
  __REG32               : 8;
  __REG32 Empty3        : 1;
  __REG32 IntSPR3       : 1;
  __REG32 IntSPDD3      : 1;
  __REG32               : 1;
  __REG32 IntReady3     : 1;
  __REG32 IntPing3      : 1;
  __REG32 IntAChg3      : 1;
  __REG32 IntDEnd3      : 1;
  __REG32 IntEmpty3     : 1;
  __REG32 IntStalled3   : 1;
  __REG32 IntNack3      : 1;
  __REG32 IntClStall3   : 1;
  __REG32 CrtAlt3       : 4;
  __REG32 CrtIntf3      : 4;
} __ufeps3_bits;

/*UFConfig Register 0*/
typedef struct{
  __REG32 MakeUp_Data   :32;
} __ufconfig0_bits;

/*UFConfig Registera 1 - 4*/
typedef struct{
  __REG32 EpNum         : 4;
  __REG32 IO            : 1;
  __REG32 Type          : 2;
  __REG32 Conf          : 4;
  __REG32 Intf          : 4;
  __REG32 Alt           : 4;
  __REG32 Size          :11;
  __REG32 NumTr         : 2;
} __ufconfigx_bits;

/*USB Function Endpoint1 DMA Control/Status Register (UFEpDC1)*/
typedef struct{
  __REG32 EpDE1         : 1;
  __REG32 EpDI1         : 1;
  __REG32 EpDM1         : 1;
  __REG32 EpDF1         : 1;
  __REG32 EpIO1         : 1;
  __REG32 EpAI1         : 1;
  __REG32 EpNE1         : 1;
  __REG32 EpNF1         : 1;
  __REG32               :24;
} __ufepdc1_bits;

/*USB Function Endpoint2 DMA Control/Status Register (UFEpDC2)*/
typedef struct{
  __REG32 EpDE2         : 1;
  __REG32 EpDI2         : 1;
  __REG32 EpDM2         : 1;
  __REG32 EpDF2         : 1;
  __REG32 EpIO2         : 1;
  __REG32 EpAI2         : 1;
  __REG32 EpNE2         : 1;
  __REG32 EpNF2         : 1;
  __REG32               :24;
} __ufepdc2_bits;

/*CS0 data register (CS0DAT)*/
typedef struct{
  __REG32 DATA          :16;
  __REG32               :16;
} __cs0dat_bits;

/*CS0 error register (CS0ER) and 
CS0 features register (CS0FT)*/
typedef union {
  /*CS0ER*/
  struct {
    __REG32 MED         : 1;
    __REG32 NM          : 1;
    __REG32 ABRT        : 1;
    __REG32 MCR         : 1;
    __REG32 IDNF        : 1;
    __REG32 MC          : 1;
    __REG32 WP          : 1;
    __REG32             :25;
  };
  /*CS0FT*/
  struct {
    __REG32 FEATURES    : 8;
    __REG32             :24;
  };
} __cs0er_cs0ft_bits;

/*CS0 sector count register (CS0SC)*/
typedef struct{
  __REG32 SECTOR_COUNT  : 8;
  __REG32               :24;
} __cs0sc_bits;

/*CS0 sector number register (CS0SN)*/
typedef struct{
  __REG32 SECTOR_NUMBER : 8;
  __REG32               :24;
} __cs0sn_bits;

/*CS0 cylinder low register (CS0CL)*/
typedef struct{
  __REG32 CYLINDER_LOW  : 8;
  __REG32               :24;
} __cs0cl_bits;

/*CS0 cylinder High register (CS0CH)*/
typedef struct{
  __REG32 CYLINDER_HIGH : 8;
  __REG32               :24;
} __cs0ch_bits;

/*CS0 device/head register (CS0DH)*/
typedef struct{
  __REG32 HEAD          : 4;
  __REG32 DEV           : 1;
  __REG32               : 1;
  __REG32 L             : 1;
  __REG32               :25;
} __cs0dh_bits;

/*CS0 status register (CS0ST) and 
CS0 command register (CS0CMD)*/
typedef union {
  /*CS0ST*/
  struct {
    __REG32 ERR         : 1;
    __REG32             : 2;
    __REG32 DRQ         : 1;
    __REG32 DSC         : 1;
    __REG32 DF          : 1;
    __REG32 DRDY        : 1;
    __REG32 BSY         : 1;
    __REG32             :24;
  };
  /*CS0CMD*/
  struct {
    __REG32 COMMAND_CODE : 8;
    __REG32              :24;
  };
} __cs0st_cs0cmd_bits;

/*CS1 alternate status register (CS1AS) and 
CS1 device control register (CS1DC)*/
typedef union {
  /*CS1AS*/
  struct {
    __REG32 ERR         : 1;
    __REG32             : 2;
    __REG32 DRQ         : 1;
    __REG32 DSC         : 1;
    __REG32 DF          : 1;
    __REG32 DRDY        : 1;
    __REG32 BSY         : 1;
    __REG32             :24;
  };
  /*CS1DC*/
  struct {
    __REG32             : 1;
    __REG32 XIEN        : 1;
    __REG32 SRST        : 1;
    __REG32             :29;
  };
} __cs1as_cs1dc_bits;

/*Data register (IDEDAT)*/
typedef struct{
  __REG32 DATA          :16;
  __REG32               :16;
} __idedata_bits;

/*PIO timing control register (IDEPTCR)*/
typedef struct{
  __REG32 RECOVERY_COUNT  : 4;
  __REG32 ACTIVE_COUNT    : 4;
  __REG32                 :24;
} __ideptcr_bits;

/*PIO address setup register (IDEPASR)*/
typedef struct{
  __REG32 ADDRESS_SETUP   : 3;
  __REG32                 :29;
} __idepasr_bits;

/*IDE command register*/
typedef struct{
  __REG32 INTEN           : 1;
  __REG32 DMAIFEN         : 1;
  __REG32 DMAD            : 1;
  __REG32 INTCLR          : 1;
  __REG32                 : 1;
  __REG32 RST             : 1;
  __REG32 DRESET          : 1;
  __REG32 CSEL            : 1;
  __REG32                 :24;
} __ideicmr_bits;

/*IDE status register (IDEISTR)*/
typedef struct{
  __REG32 INTRQ           : 1;
  __REG32 XDASP           : 1;
  __REG32 XIOCS16         : 1;
  __REG32 XCBLID          : 1;
  __REG32                 :28;
} __ideistr_bits;

/*Interrupt enable register (IDEINER)*/
typedef struct{
  __REG32                 : 1;
  __REG32 TXFIFO_EMPTY    : 1;
  __REG32 RXFIFO_EMPTY    : 1;
  __REG32                 :29;
} __ideiner_bits;

/*Interrupt status register (IDEINSR)*/
typedef struct{
  __REG32 INTRQ           : 1;
  __REG32 TXFIFO_EMPTY    : 1;
  __REG32 RXFIFO_EMPTY    : 1;
  __REG32                 :29;
} __ideinsr_bits;

/*FIFO command register (IDEFCMR)*/
typedef struct{
  __REG32 RXFIFO_ENABLE   : 1;
  __REG32 TXFIFO_ENABLE   : 1;
  __REG32 RXFIFO_CLEAR    : 1;
  __REG32 TXFIFO_CLEAR    : 1;
  __REG32                 :28;
} __idefcmr_bits;

/*FIFO status register (IDEFSTR)*/
typedef struct{
  __REG32 RXFIFO_EMPTY    : 1;
  __REG32 TXFIFO_EMPTY    : 1;
  __REG32 RXFIFO_FULL     : 1;
  __REG32 TXFIFO_FULL     : 1;
  __REG32                 :28;
} __idefstr_bits;

/*Transmission FIFO count register (IDETFCR)*/
typedef struct{
  __REG32 XCNT            : 8;
  __REG32                 :24;
} __idetfcr_bits;

/*Reception FIFO count register (IDERFCR)*/
typedef struct{
  __REG32 RCNT            : 8;
  __REG32                 :24;
} __iderfcr_bits;

/*UDMA timing control register (IDEUTCR)*/
typedef struct{
  __REG32 RECOVERY_COUNT  : 4;
  __REG32 ACTIVE_COUNT    : 4;
  __REG32                 :24;
} __ideutcr_bits;

/*UDMA command register (IDEUCMR)*/
typedef struct{
  __REG32                 : 1;
  __REG32 UDMA_ENABLE     : 1;
  __REG32 UDMA_DIRECTION  : 1;
  __REG32 PAUSE_TERM      : 1;
  __REG32                 :28;
} __ideucmr_bits;

/*UDMA status register (IDEUSTR)*/
typedef struct{
  __REG32 UDMA_DIB        : 1;
  __REG32 UDMA_DOB        : 1;
  __REG32                 :30;
} __ideustr_bits;

/*RxFIFO rest count compare value (IDERRCC)*/
typedef struct{
  __REG32 RRCC            : 8;
  __REG32                 :24;
} __iderrcc_bits;

/*Ultra DMA timing control 1 (IDEUTC1)*/
typedef struct{
  __REG32 TENV            : 4;
  __REG32 TACK            : 4;
  __REG32                 :24;
} __ideutc1_bits;

/*Ultra DMA timing control 2 (IDEUTC2)*/
typedef struct{
  __REG32 TUI             : 4;
  __REG32 TLI             : 4;
  __REG32                 :24;
} __ideutc2_bits;

/*Ultra DMA timing control 3 (IDEUTC3)*/
typedef struct{
  __REG32 TMLI            : 4;
  __REG32 TSS             : 4;
  __REG32                 :24;
} __ideutc3_bits;

/*DMA status register (IDESTATUS)*/
typedef struct{
  __REG32 DMABSY          : 1;
  __REG32                 :31;
} __idestatus_bits;

/*Interrupt register (IDEINT)*/
typedef struct{
  __REG32 DMAEND          : 1;
  __REG32 AHBMWERR        : 1;
  __REG32 DRVACSERR       : 1;
  __REG32                 :29;
} __ideint_bits;

/*Interrupt register (IDEINT)*/
typedef struct{
  __REG32 DMAENDM         : 1;
  __REG32 AHBMWERRM       : 1;
  __REG32 DRVACSERRM      : 1;
  __REG32                 : 1;
  __REG32 FIDE66MCRINTM   : 1;
  __REG32                 :27;
} __ideintmsk_bits;

/*PIO access control register (IDEPIOCTL)*/
typedef struct{
  __REG32 PIOCTRL         : 1;
  __REG32                 :31;
} __idepioctl_bits;

/*DMA control register (IDEDMACTL)*/
typedef struct{
  __REG32 TRANS_MODE      : 1;
  __REG32                 :30;
  __REG32 DMA_START       : 1;
} __idedmactl_bits;

/*DMA transfer control register (IDEDMATC)*/
typedef struct{
  __REG32 TYPE            : 3;
  __REG32 INCR            : 1;
  __REG32 DIV             : 2;
  __REG32                 : 2;
  __REG32 SC              : 8;
  __REG32 DTC             : 9;
  __REG32                 : 7;
} __idedmatc_bits;

/*CHIP ID register (CCID)*/
typedef struct{
  __REG32 VERSION         : 8;
  __REG32 CHIPNAME        : 8;
  __REG32 YEAR            :16;
} __ccid_bits;

/*Software reset register (CSRST)*/
typedef struct{
  __REG32 SFTRST          : 1;
  __REG32                 :31;
} __csrst_bits;

/*Interrupt status register (CIST)*/
typedef struct{
  __REG32                 : 5;
  __REG32 INT5            : 1;
  __REG32                 :18;
  __REG32 INT24           : 1;
  __REG32                 : 1;
  __REG32 INT26           : 1;
  __REG32 INT27           : 1;
  __REG32 INT28           : 1;
  __REG32                 : 2;
  __REG32 INT31           : 1;
} __cist_bits;

/*Interrupt status mask register(CISTM)*/
typedef struct{
  __REG32 INT0_MASK       : 1;
  __REG32 INT1_MASK       : 1;
  __REG32                 : 3;
  __REG32 INT5_MASK       : 1;
  __REG32                 :18;
  __REG32 INT24_MASK      : 1;
  __REG32                 : 1;
  __REG32 INT26_MASK      : 1;
  __REG32 INT27_MASK      : 1;
  __REG32 INT28_MASK      : 1;
  __REG32                 : 2;
  __REG32 INT31_MASK      : 1;
} __cistm_bits;

/*GPIO interrupt status register (CGPIO_IST)*/
typedef struct{
  __REG32 GPIO0_INT_ST    : 1;
  __REG32 GPIO1_INT_ST    : 1;
  __REG32 GPIO2_INT_ST    : 1;
  __REG32 GPIO3_INT_ST    : 1;
  __REG32 GPIO4_INT_ST    : 1;
  __REG32 GPIO5_INT_ST    : 1;
  __REG32 GPIO6_INT_ST    : 1;
  __REG32 GPIO7_INT_ST    : 1;
  __REG32 GPIO8_INT_ST    : 1;
  __REG32 GPIO9_INT_ST    : 1;
  __REG32 GPIO10_INT_ST   : 1;
  __REG32 GPIO11_INT_ST   : 1;
  __REG32 GPIO12_INT_ST   : 1;
  __REG32 GPIO13_INT_ST   : 1;
  __REG32 GPIO14_INT_ST   : 1;
  __REG32 GPIO15_INT_ST   : 1;
  __REG32 GPIO16_INT_ST   : 1;
  __REG32 GPIO17_INT_ST   : 1;
  __REG32 GPIO18_INT_ST   : 1;
  __REG32 GPIO19_INT_ST   : 1;
  __REG32 GPIO20_INT_ST   : 1;
  __REG32 GPIO21_INT_ST   : 1;
  __REG32 GPIO22_INT_ST   : 1;
  __REG32 GPIO23_INT_ST   : 1;
  __REG32                 : 8;
} __cgpio_ist_bits;

/*GPIO interrupt status mask register (CGPIO_ISTM)*/
typedef struct{
  __REG32 GPIO0_INT_EN    : 1;
  __REG32 GPIO1_INT_EN    : 1;
  __REG32 GPIO2_INT_EN    : 1;
  __REG32 GPIO3_INT_EN    : 1;
  __REG32 GPIO4_INT_EN    : 1;
  __REG32 GPIO5_INT_EN    : 1;
  __REG32 GPIO6_INT_EN    : 1;
  __REG32 GPIO7_INT_EN    : 1;
  __REG32 GPIO8_INT_EN    : 1;
  __REG32 GPIO9_INT_EN    : 1;
  __REG32 GPIO10_INT_EN   : 1;
  __REG32 GPIO11_INT_EN   : 1;
  __REG32 GPIO12_INT_EN   : 1;
  __REG32 GPIO13_INT_EN   : 1;
  __REG32 GPIO14_INT_EN   : 1;
  __REG32 GPIO15_INT_EN   : 1;
  __REG32 GPIO16_INT_EN   : 1;
  __REG32 GPIO17_INT_EN   : 1;
  __REG32 GPIO18_INT_EN   : 1;
  __REG32 GPIO19_INT_EN   : 1;
  __REG32 GPIO20_INT_EN   : 1;
  __REG32 GPIO21_INT_EN   : 1;
  __REG32 GPIO22_INT_EN   : 1;
  __REG32 GPIO23_INT_EN   : 1;
  __REG32                 : 8;
} __cgpio_istm_bits;

/*GPIO interrupt polarity setting register (CGPIO_IP)*/
typedef struct{
  __REG32 GPIO0_INT_PL    : 1;
  __REG32 GPIO1_INT_PL    : 1;
  __REG32 GPIO2_INT_PL    : 1;
  __REG32 GPIO3_INT_PL    : 1;
  __REG32 GPIO4_INT_PL    : 1;
  __REG32 GPIO5_INT_PL    : 1;
  __REG32 GPIO6_INT_PL    : 1;
  __REG32 GPIO7_INT_PL    : 1;
  __REG32 GPIO8_INT_PL    : 1;
  __REG32 GPIO9_INT_PL    : 1;
  __REG32 GPIO10_INT_PL   : 1;
  __REG32 GPIO11_INT_PL   : 1;
  __REG32 GPIO12_INT_PL   : 1;
  __REG32 GPIO13_INT_PL   : 1;
  __REG32 GPIO14_INT_PL   : 1;
  __REG32 GPIO15_INT_PL   : 1;
  __REG32 GPIO16_INT_PL   : 1;
  __REG32 GPIO17_INT_PL   : 1;
  __REG32 GPIO18_INT_PL   : 1;
  __REG32 GPIO19_INT_PL   : 1;
  __REG32 GPIO20_INT_PL   : 1;
  __REG32 GPIO21_INT_PL   : 1;
  __REG32 GPIO22_INT_PL   : 1;
  __REG32 GPIO23_INT_PL   : 1;
  __REG32                 : 8;
} __cgpio_ip_bits;

/*GPIO interrupt mode setting register (CGPIO_IM)*/
typedef struct{
  __REG32 GPIO0_INT_M     : 1;
  __REG32 GPIO1_INT_M     : 1;
  __REG32 GPIO2_INT_M     : 1;
  __REG32 GPIO3_INT_M     : 1;
  __REG32 GPIO4_INT_M     : 1;
  __REG32 GPIO5_INT_M     : 1;
  __REG32 GPIO6_INT_M     : 1;
  __REG32 GPIO7_INT_M     : 1;
  __REG32 GPIO8_INT_M     : 1;
  __REG32 GPIO9_INT_M     : 1;
  __REG32 GPIO10_INT_M    : 1;
  __REG32 GPIO11_INT_M    : 1;
  __REG32 GPIO12_INT_M    : 1;
  __REG32 GPIO13_INT_M    : 1;
  __REG32 GPIO14_INT_M    : 1;
  __REG32 GPIO15_INT_M    : 1;
  __REG32 GPIO16_INT_M    : 1;
  __REG32 GPIO17_INT_M    : 1;
  __REG32 GPIO18_INT_M    : 1;
  __REG32 GPIO19_INT_M    : 1;
  __REG32 GPIO20_INT_M    : 1;
  __REG32 GPIO21_INT_M    : 1;
  __REG32 GPIO22_INT_M    : 1;
  __REG32 GPIO23_INT_M    : 1;
  __REG32                 : 8;
} __cgpio_im_bits;

/*AXI bus wait cycle setting register (CAXI_BW)*/
typedef struct{
  __REG32 PAHB_WW         : 4;
  __REG32 PAHB_RW         : 4;
  __REG32                 : 8;
  __REG32 DRAW_WW         : 4;
  __REG32 DRAW_RW         : 4;
  __REG32 DISP_WW         : 4;
  __REG32 DISP_RW         : 4;
} __caxi_bw_bits;

/*AXI polarity setting register (CAXI_PS)*/
typedef struct{
  __REG32 P_SEL0          : 3;
  __REG32                 : 1;
  __REG32 P_SEL1          : 3;
  __REG32                 : 1;
  __REG32 P_SEL2          : 3;
  __REG32                 : 1;
  __REG32 P_SEL3          : 3;
  __REG32                 : 1;
  __REG32 P_SEL4          : 3;
  __REG32                 :13;
} __caxi_ps_bits;

/*Multiplex mode setting register (CMUX_MD)*/
typedef struct{
  __REG32 MPX_MODE_2      : 3;
  __REG32                 : 1;
  __REG32 MPX_MODE_4      : 2;
  __REG32                 :26;
} __cmux_md_bits;

/*External pin status register (CEX_PIN_ST)*/
typedef struct{
  __REG32 MPX_MODE_1      : 2;
  __REG32 MPX_MODE_5      : 2;
  __REG32                 : 4;
  __REG32 CRIPM           : 4;
  __REG32 USB_MODE        : 1;
  __REG32                 :19;
} __cex_pin_st_bits;

/*MediaLB setting register (CMLB)*/
typedef struct{
  __REG32 SEL_SPREAD      : 1;
  __REG32                 :31;
} __cmlb_bits;

/*USB set register (CUSB)*/
typedef struct{
  __REG32 SYS_INTERRUPPT  : 1;
  __REG32                 : 3;
  __REG32 APP_PRT_OVRCUR  : 1;
  __REG32                 :27;
} __cusb_bits;

/*Byte swap switching register (CBSC)*/
typedef struct{
  __REG32                       : 4;
  __REG32 USB_HOST_ENDIAN       : 3;
  __REG32                       : 1;
  __REG32 I2S2_ENDIAN           : 3;
  __REG32                       : 1;
  __REG32 I2S1_ENDIAN           : 3;
  __REG32                       : 1;
  __REG32 I2S0_ENDIAN           : 3;
  __REG32                       : 1;
  __REG32 SDMC_ENDIAN           : 3;
  __REG32                       : 1;
  __REG32 IDE_SLV_PIO_ENDIAN    : 3;
  __REG32                       : 1;
  __REG32 IDE_MSTR_DMA_ENDIAN   : 3;
  __REG32                       : 1;
} __cbsc_bits;

/*DDR2 controller reset control register (CDCRC)*/
typedef struct{
  __REG32 IDLLRST         : 1;
  __REG32 IRESET_IUSRRST  : 1;
  __REG32                 :30;
} __cdcrc_bits;

/*Software reset register 0 for macro (CMSR0)*/
typedef struct{
  __REG32 SRST0_0         : 1;
  __REG32 SRST0_1         : 1;
  __REG32 SRST0_2         : 1;
  __REG32 SRST0_3         : 1;
  __REG32 SRST0_4         : 1;
  __REG32                 : 2;
  __REG32 SRST0_7         : 1;
  __REG32                 : 8;
  __REG32 SRST0_16        : 1;
  __REG32                 : 7;
  __REG32 SRST0_24        : 1;
  __REG32 SRST0_25        : 1;
  __REG32                 : 6;
} __cmsr0_bits;

/*Software reset register 1 for macro (CMSR1)*/
typedef struct{
  __REG32 SRST1_0         : 1;
  __REG32 SRST1_1         : 1;
  __REG32 SRST1_2         : 1;
  __REG32 SRST1_3         : 1;
  __REG32 SRST1_4         : 1;
  __REG32 SRST1_5         : 1;
  __REG32 SRST1_6         : 1;
  __REG32 SRST1_7         : 1;
  __REG32 SRST1_8         : 1;
  __REG32 SRST1_9         : 1;
  __REG32 SRST1_10        : 1;
  __REG32 SRST1_11        : 1;
  __REG32 SRST1_12        : 1;
  __REG32 SRST1_13        : 1;
  __REG32 SRST1_14        : 1;
  __REG32 SRST1_15        : 1;
  __REG32 SRST1_16        : 1;
  __REG32 SRST1_17        : 1;
  __REG32 SRST1_18        : 1;
  __REG32 SRST1_19        : 1;
  __REG32 SRST1_20        : 1;
  __REG32 SRST1_21        : 1;
  __REG32 SRST1_22        : 1;
  __REG32 SRST1_23        : 1;
  __REG32 SRST1_24        : 1;
  __REG32 SRST1_25        : 1;
  __REG32 SRST1_26        : 1;
  __REG32 SRST1_27        : 1;
  __REG32 SRST1_28        : 1;
  __REG32 SRST1_29        : 1;
  __REG32                 : 2;
} __cmsr1_bits;

/*External interrupt enable register (EIENB)*/
typedef struct{
  __REG32 ENB0            : 1;
  __REG32 ENB1            : 1;
  __REG32 ENB2            : 1;
  __REG32 ENB3            : 1;
  __REG32                 :28;
} __eienb_bits;

/*External interrupt request register (EIREQ)*/
typedef struct{
  __REG32 REQ0            : 1;
  __REG32 REQ1            : 1;
  __REG32 REQ2            : 1;
  __REG32 REQ3            : 1;
  __REG32                 :28;
} __eireq_bits;

/*External interrupt level register (EILVL)*/
typedef struct{
  __REG32 LVL0            : 2;
  __REG32 LVL1            : 2;
  __REG32 LVL2            : 2;
  __REG32 LVL3            : 2;
  __REG32                 :24;
} __eilvl_bits;

/*VCCC (Vdisp/Capture common control )*/
typedef struct{
  __REG32 V0sr            : 1;
  __REG32 V1sr            : 1;
  __REG32 C0sr            : 1;
  __REG32 C1sr            : 1;
  __REG32                 : 8;
  __REG32 C0sel           : 1;
  __REG32 C1sel           : 1;
  __REG32                 : 3;
  __REG32 hmon            : 1;
  __REG32                 : 2;
  __REG32 dis2s           : 1;
  __REG32                 :11;
} __vccc_bits;

/* DCM0 (Display Control Mode 0) */
typedef struct{
  __REG32 SYNC            : 2;
  __REG32 ESY             : 1;
  __REG32 SF              : 1;
  __REG32                 : 2;
  __REG32 ODE             : 1;
  __REG32 EEQ             : 1;
  __REG32 SC              : 5;
  __REG32                 : 2;
  __REG32 CKS             : 1;
  __REG32 L0E             : 1;
  __REG32 L1E             : 1;
  __REG32 L23E            : 1;
  __REG32 L45E            : 1;
  __REG32                 :10;
  __REG32 STOP            : 1;
  __REG32 DEN             : 1;
} __dcxdcm0_bits;

/* DCM1 (Display Control Mode 1) */
typedef struct{
  __REG32 SYNC            : 2;
  __REG32 ESY             : 1;
  __REG32 SF              : 1;
  __REG32                 : 2;
  __REG32 ODE             : 1;
  __REG32 EEQ             : 1;
  __REG32 SC              : 6;
  __REG32                 : 1;
  __REG32 CKS             : 1;
  __REG32 L0E             : 1;
  __REG32 L1E             : 1;
  __REG32 L2E             : 1;
  __REG32 L3E             : 1;
  __REG32 L4E             : 1;
  __REG32 L5E             : 1;
  __REG32                 : 8;
  __REG32 STOP            : 1;
  __REG32 DEN             : 1;
} __dcxdcm1_bits;

/* DCM2 (Display Control Mode 2) */
typedef struct{
  __REG32 RUM             : 1;
  __REG32 RUF             : 1;
  __REG32                 :30;
} __dcxdcm2_bits;

/* DCM3 (Display Control Mode 3) */
typedef struct{
  __REG32 DCKD            : 5;
  __REG32                 : 3;
  __REG32 DCKinv          : 1;
  __REG32 DCKed           : 1;
  __REG32 POM             : 1;
  __REG32                 : 5;
  __REG32 MBST            : 1;
  __REG32                 : 3;
  __REG32 RGBsh           : 1;
  __REG32 RGBrv           : 1;
  __REG32 CSY0            : 1;
  __REG32 Gswap           : 1;
  __REG32 VPWMs           : 1;
  __REG32 GVD             : 1;
  __REG32                 : 6;
} __dcxdcm3_bits;

/*HTP (Horizontal Total Pixels)*/
typedef struct{
  __REG16 HTP             :12;
  __REG16                 : 4;
} __dcxhtp_bits;

/*HDP (Horizontal Display Period)*/
typedef struct{
  __REG16 HDP             :12;
  __REG16                 : 4;
} __dcxhdp_bits;

/*HDB (Horizontal Display Boundary)*/
typedef struct{
  __REG16 HDB             :12;
  __REG16                 : 4;
} __dcxhdb_bits;

/*HSP (Horizontal Synchronize pulse Position)*/
typedef struct{
  __REG16 HSP             :12;
  __REG16                 : 4;
} __dcxhsp_bits;

/*HSW (Horizontal Synchronize pulse Width)*/
typedef struct{
  __REG8  HSW             : 8;
} __dcxhsw_bits;

/*VSW (Vertical Synchronize pulse Width)*/
typedef struct{
  __REG8  VSW             : 6;
  __REG8                  : 2;
} __dcxvsw_bits;

/*VTR (Vertical Total Rasters)*/
typedef struct{
  __REG16 VTR             :12;
  __REG16                 : 4;
} __dcxvtr_bits;

/*VSP (Vertical Synchronize pulse Position)*/
typedef struct{
  __REG16 VSP             :12;
  __REG16                 : 4;
} __dcxvsp_bits;

/*VDP (Vertical Display Period)*/
typedef struct{
  __REG16 VDP             :12;
  __REG16                 : 4;
} __dcxvdp_bits;

/*L0M (L0 layer Mode)*/
typedef struct{
  __REG32 L0H             :12;
  __REG32                 : 4;
  __REG32 L0W             : 8;
  __REG32                 : 7;
  __REG32 L0C             : 1;
} __dcxl0m_bits;

/*L0EM (L0-layer Extended Mode)*/
typedef struct{
  __REG32 L0WP            : 1;
  __REG32                 :19;
  __REG32 L0PB            : 4;
  __REG32                 : 6;
  __REG32 L0EC            : 2;
} __dcxl0em_bits;

/*L0DX (L0-layer Display position X)*/
typedef struct{
  __REG16 L0DX            :12;
  __REG16                 : 4;
} __dcxl0dx_bits;

/*L0DY (L0-layer Display position Y)*/
typedef struct{
  __REG16 L0DY            :12;
  __REG16                 : 4;
} __dcxl0dy_bits;

/*L0WX (L0 layer Window position X)*/
typedef struct{
  __REG16 L0WX            :12;
  __REG16                 : 4;
} __dcxl0wx_bits;

/*L0WY (L0 layer Window position Y)*/
typedef struct{
  __REG16 L0WY            :12;
  __REG16                 : 4;
} __dcxl0wy_bits;

/*L0WW (L0 layer Window Width)*/
typedef struct{
  __REG16 L0WW            :12;
  __REG16                 : 4;
} __dcxl0ww_bits;

/*L0WH (L0 layer Window Height)*/
typedef struct{
  __REG16 L0WH            :12;
  __REG16                 : 4;
} __dcxl0wh_bits;

/*L1M (L1 layer Mode)*/
typedef struct{
  __REG32                 :16;
  __REG32 L1W             : 8;
  __REG32                 : 4;
  __REG32 L1IM            : 1;
  __REG32 L1CS            : 1;
  __REG32 L1YC            : 1;
  __REG32 L1C             : 1;
} __dcxl1m_bits;

/*L1EM (L1-layer Extended Mode)*/
typedef struct{
  __REG32                 :20;
  __REG32 L1PB            : 4;
  __REG32 L1DM            : 2;
  __REG32                 : 4;
  __REG32 L1EC            : 2;
} __dcxl1em_bits;

/*L1WX (L1 layer Window position X)*/
typedef struct{
  __REG16 L1WX            :12;
  __REG16                 : 4;
} __dcxl1wx_bits;

/*L1WY (L1 layer Window position Y)*/
typedef struct{
  __REG16 L1WY            :12;
  __REG16                 : 4;
} __dcxl1wy_bits;

/*L1WW (L1 layer Window Width)*/
typedef struct{
  __REG16 L1WW            :12;
  __REG16                 : 4;
} __dcxl1ww_bits;

/*L1WH (L1 layer Window Height)*/
typedef struct{
  __REG16 L1WH            :12;
  __REG16                 : 4;
} __dcxl1wh_bits;

/*L2M (L2 layer Mode)*/
typedef struct{
  __REG32 L2H             :12;
  __REG32                 : 4;
  __REG32 L2W             : 8;
  __REG32                 : 5;
  __REG32 L2FLP           : 2;
  __REG32 L2C             : 1;
} __dcxl2m_bits;

/*L2EM (L2-layer Extended Mode)*/
typedef struct{
  __REG32 L2WP            : 1;
  __REG32 L2OM            : 1;
  __REG32                 :18;
  __REG32 L2PB            : 4;
  __REG32                 : 6;
  __REG32 L2EC            : 2;
} __dcxl2em_bits;

/*L2DX (L2-layer Display position X)*/
typedef struct{
  __REG16 L2DX            :12;
  __REG16                 : 4;
} __dcxl2dx_bits;

/*L2DY (L2-layer Display position Y)*/
typedef struct{
  __REG16 L2DY            :12;
  __REG16                 : 4;
} __dcxl2dy_bits;

/*L2WX (L2 layer Window position X)*/
typedef struct{
  __REG16 L2WX            :12;
  __REG16                 : 4;
} __dcxl2wx_bits;

/*L2WY (L2 layer Window position Y)*/
typedef struct{
  __REG16 L2WY            :12;
  __REG16                 : 4;
} __dcxl2wy_bits;

/*L2WW (L2 layer Window Width)*/
typedef struct{
  __REG16 L2WW            :12;
  __REG16                 : 4;
} __dcxl2ww_bits;

/*L2WH (L2 layer Window Height)*/
typedef struct{
  __REG16 L2WH            :12;
  __REG16                 : 4;
} __dcxl2wh_bits;

/*L3M (L3 layer Mode)*/
typedef struct{
  __REG32 L3H             :12;
  __REG32                 : 4;
  __REG32 L3W             : 8;
  __REG32                 : 5;
  __REG32 L3FLP           : 2;
  __REG32 L3C             : 1;
} __dcxl3m_bits;

/*L3EM (L3-layer Extended Mode)*/
typedef struct{
  __REG32 L3WP            : 1;
  __REG32 L3OM            : 1;
  __REG32                 :18;
  __REG32 L3PB            : 4;
  __REG32                 : 6;
  __REG32 L3EC            : 2;
} __dcxl3em_bits;

/*L3DX (L3-layer Display position X)*/
typedef struct{
  __REG16 L3DX            :12;
  __REG16                 : 4;
} __dcxl3dx_bits;

/*L3DY (L3-layer Display position Y)*/
typedef struct{
  __REG16 L3DY            :12;
  __REG16                 : 4;
} __dcxl3dy_bits;

/*L3WX (L3 layer Window position X)*/
typedef struct{
  __REG16 L3WX            :12;
  __REG16                 : 4;
} __dcxl3wx_bits;

/*L3WY (L3 layer Window position Y)*/
typedef struct{
  __REG16 L3WY            :12;
  __REG16                 : 4;
} __dcxl3wy_bits;

/*L3WW (L3 layer Window Width)*/
typedef struct{
  __REG16 L3WW            :12;
  __REG16                 : 4;
} __dcxl3ww_bits;

/*L3WH (L3 layer Window Height)*/
typedef struct{
  __REG16 L3WH            :12;
  __REG16                 : 4;
} __dcxl3wh_bits;

/*L4M (L4 layer Mode)*/
typedef struct{
  __REG32 L4H             :12;
  __REG32                 : 4;
  __REG32 L4W             : 8;
  __REG32                 : 5;
  __REG32 L4FLP           : 2;
  __REG32 L4C             : 1;
} __dcxl4m_bits;

/*L4EM (L4-layer Extended Mode)*/
typedef struct{
  __REG32 L4WP            : 1;
  __REG32 L4OM            : 1;
  __REG32                 :18;
  __REG32 L4PB            : 4;
  __REG32                 : 6;
  __REG32 L4EC            : 2;
} __dcxl4em_bits;

/*L4DX (L4-layer Display position X)*/
typedef struct{
  __REG16 L4DX            :12;
  __REG16                 : 4;
} __dcxl4dx_bits;

/*L4DY (L4-layer Display position Y)*/
typedef struct{
  __REG16 L4DY            :12;
  __REG16                 : 4;
} __dcxl4dy_bits;

/*L4WX (L4 layer Window position X)*/
typedef struct{
  __REG16 L4WX            :12;
  __REG16                 : 4;
} __dcxl4wx_bits;

/*L4WY (L4 layer Window position Y)*/
typedef struct{
  __REG16 L4WY            :12;
  __REG16                 : 4;
} __dcxl4wy_bits;

/*L4WW (L4 layer Window Width)*/
typedef struct{
  __REG16 L4WW            :12;
  __REG16                 : 4;
} __dcxl4ww_bits;

/*L4WH (L4 layer Window Height)*/
typedef struct{
  __REG16 L4WH            :12;
  __REG16                 : 4;
} __dcxl4wh_bits;

/*L5M (L5 layer Mode)*/
typedef struct{
  __REG32 L5H             :12;
  __REG32                 : 4;
  __REG32 L5W             : 8;
  __REG32                 : 5;
  __REG32 L5FLP           : 2;
  __REG32 L5C             : 1;
} __dcxl5m_bits;

/*L5EM (L5-layer Extended Mode)*/
typedef struct{
  __REG32 L5WP            : 1;
  __REG32 L5OM            : 1;
  __REG32                 :18;
  __REG32 L5PB            : 4;
  __REG32                 : 6;
  __REG32 L5EC            : 2;
} __dcxl5em_bits;

/*L5DX (L5-layer Display position X)*/
typedef struct{
  __REG16 L5DX            :12;
  __REG16                 : 4;
} __dcxl5dx_bits;

/*L5DY (L5-layer Display position Y)*/
typedef struct{
  __REG16 L5DY            :12;
  __REG16                 : 4;
} __dcxl5dy_bits;

/*L5WX (L5 layer Window position X)*/
typedef struct{
  __REG16 L5WX            :12;
  __REG16                 : 4;
} __dcxl5wx_bits;

/*L5WY (L5 layer Window position Y)*/
typedef struct{
  __REG16 L5WY            :12;
  __REG16                 : 4;
} __dcxl5wy_bits;

/*L5WW (L5 layer Window Width)*/
typedef struct{
  __REG16 L5WW            :12;
  __REG16                 : 4;
} __dcxl5ww_bits;

/*L5WH (L5 layer Window Height)*/
typedef struct{
  __REG16 L5WH            :12;
  __REG16                 : 4;
} __dcxl5wh_bits;

/*CUTC (Cursor Transparent Control)*/
typedef struct{
  __REG16 CUTC            : 8;
  __REG16 CUZT            : 1;
  __REG16                 : 7;
} __dcxcutc_bits;

/*CPM (Cursor Priority Mode)*/
typedef struct{
  __REG8  CUO0            : 1;
  __REG8  CUO1            : 1;
  __REG8                  : 2;
  __REG8  CEN0            : 1;
  __REG8  CEN1            : 1;
  __REG8                  : 2;
} __dcxcpm_bits;

/*CUX0 (Cursor-0 X position)*/
typedef struct{
  __REG16 CUX0            :12;
  __REG16                 : 4;
} __dcxcux0_bits;

/*CUY0 (Cursor-0 Y position)*/
typedef struct{
  __REG16 CUY0            :12;
  __REG16                 : 4;
} __dcxcuy0_bits;

/*CUX1 (Cursor-1 X position)*/
typedef struct{
  __REG16 CUX1            :12;
  __REG16                 : 4;
} __dcxcux1_bits;

/*CUY1 (Cursor-1 Y position)*/
typedef struct{
  __REG16 CUY1            :12;
  __REG16                 : 4;
} __dcxcuy1_bits;

/*MDC (Multi Display Control)*/
typedef struct{
  __REG32 SC0en0          : 1;
  __REG32 SC0en1          : 1;
  __REG32 SC0en2          : 1;
  __REG32 SC0en3          : 1;
  __REG32 SC0en4          : 1;
  __REG32 SC0en5          : 1;
  __REG32 SC0en6          : 1;
  __REG32 SC0en7          : 1;
  __REG32 SC1en0          : 1;
  __REG32 SC1en1          : 1;
  __REG32 SC1en2          : 1;
  __REG32 SC1en3          : 1;
  __REG32 SC1en4          : 1;
  __REG32 SC1en5          : 1;
  __REG32 SC1en6          : 1;
  __REG32 SC1en7          : 1;
  __REG32                 :15;
  __REG32 MDen            : 1;
} __dcxmdc_bits;

/*DLS (Display Layer Select)*/
typedef struct{
  __REG32 DSL0            : 4;
  __REG32 DSL1            : 4;
  __REG32 DSL2            : 4;
  __REG32 DSL3            : 4;
  __REG32 DSL4            : 4;
  __REG32 DSL5            : 4;
  __REG32                 : 8;
} __dcxdls_bits;

/*DBGC (Display Background Color)*/
typedef struct{
  __REG32 DBGB            : 8;
  __REG32 DBGG            : 8;
  __REG32 DBGR            : 8;
  __REG32                 : 8;
} __dcxdbgc_bits;

/*L0BLD (L0 Blend)*/
typedef struct{
  __REG32 L0BR            : 8;
  __REG32                 : 5;
  __REG32 L0BP            : 1;
  __REG32 L0BI            : 1;
  __REG32 L0BS            : 1;
  __REG32 L0BE            : 1;
  __REG32                 :15;
} __dcxl0bld_bits;

/*L1BLD (L1 Blend)*/
typedef struct{
  __REG32 L1BR            : 8;
  __REG32                 : 5;
  __REG32 L1BP            : 1;
  __REG32 L1BI            : 1;
  __REG32 L1BS            : 1;
  __REG32 L1BE            : 1;
  __REG32                 :15;
} __dcxl1bld_bits;

/*L2BLD (L2 Blend)*/
typedef struct{
  __REG32 L2BR            : 8;
  __REG32                 : 5;
  __REG32 L2BP            : 1;
  __REG32 L2BI            : 1;
  __REG32 L2BS            : 1;
  __REG32 L2BE            : 1;
  __REG32                 :15;
} __dcxl2bld_bits;

/*L3BLD (L3 Blend)*/
typedef struct{
  __REG32 L3BR            : 8;
  __REG32                 : 5;
  __REG32 L3BP            : 1;
  __REG32 L3BI            : 1;
  __REG32 L3BS            : 1;
  __REG32 L3BE            : 1;
  __REG32                 :15;
} __dcxl3bld_bits;

/*L4BLD (L4 Blend)*/
typedef struct{
  __REG32 L4BR            : 8;
  __REG32                 : 5;
  __REG32 L4BP            : 1;
  __REG32 L4BI            : 1;
  __REG32 L4BS            : 1;
  __REG32 L4BE            : 1;
  __REG32                 :15;
} __dcxl4bld_bits;

/*L5BLD (L5 Blend)*/
typedef struct{
  __REG32 L5BR            : 8;
  __REG32                 : 5;
  __REG32 L5BP            : 1;
  __REG32 L5BI            : 1;
  __REG32 L5BS            : 1;
  __REG32 L5BE            : 1;
  __REG32                 :15;
} __dcxl5bld_bits;

/*L0TC (L0 layer Transparency Control)*/
typedef struct{
  __REG16 L0TC            :15;
  __REG16 L0ZT            : 1;
} __dcxl0tc_bits;

/*L2TC (L2 layer Transparency Control)*/
typedef struct{
  __REG16 L2TC            :15;
  __REG16 L2ZT            : 1;
} __dcxl2tc_bits;

/*L3TC (L3 layer Transparency Control)*/
typedef struct{
  __REG16 L3TC            :15;
  __REG16 L3ZT            : 1;
} __dcxl3tc_bits;

/*L0ETC (L0 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L0TEC           :24;
  __REG32                 : 7;
  __REG32 L0ETZ           : 1;
} __dcxl0etc_bits;

/*L1ETC (L1 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L1TEC           :24;
  __REG32                 : 7;
  __REG32 L1ETZ           : 1;
} __dcxl1etc_bits;

/*L2ETC (L2 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L2TEC           :24;
  __REG32                 : 7;
  __REG32 L2ETZ           : 1;
} __dcxl2etc_bits;

/*L3ETC (L3 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L3TEC           :24;
  __REG32                 : 7;
  __REG32 L3ETZ           : 1;
} __dcxl3etc_bits;

/*L4ETC (L4 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L4TEC           :24;
  __REG32                 : 7;
  __REG32 L4ETZ           : 1;
} __dcxl4etc_bits;

/*L5ETC (L5 layer Extend Transparency Control)*/
typedef struct{
  __REG32 L5TEC           :24;
  __REG32                 : 7;
  __REG32 L5ETZ           : 1;
} __dcxl5etc_bits;

/*L1YCR0 (L1 layer YC to Red coefficient 0)*/
typedef struct{
  __REG32 a11             :11;
  __REG32                 : 5;
  __REG32 a12             :11;
  __REG32                 : 5;
} __dcxl1ycr0_bits;

/*L1YCR1 (L1 layer YC to Red coefficient 1)*/
typedef struct{
  __REG32 a13             :11;
  __REG32                 : 5;
  __REG32 b1              : 9;
  __REG32                 : 7;
} __dcxl1ycr1_bits;

/*L1YCG0 (L1 layer YC to Green coefficient 0)*/
typedef struct{
  __REG32 a21             :11;
  __REG32                 : 5;
  __REG32 a22             :11;
  __REG32                 : 5;
} __dcxl1ycg0_bits;

/*L1YCR1 (L1 layer YC to Red coefficient 1)*/
typedef struct{
  __REG32 a23             :11;
  __REG32                 : 5;
  __REG32 b2              : 9;
  __REG32                 : 7;
} __dcxl1ycg1_bits;

/*L1YCB0 (L1 layer YC to Blue coefficient 0)*/
typedef struct{
  __REG32 a31             :11;
  __REG32                 : 5;
  __REG32 a32             :11;
  __REG32                 : 5;
} __dcxl1ycb0_bits;

/*L1YCB1 (L1 layer YC to Blue coefficient 1)*/
typedef struct{
  __REG32 a33             :11;
  __REG32                 : 5;
  __REG32 b3              : 9;
  __REG32                 : 7;
} __dcxl1ycb1_bits;

/*VCM (Video Capture Mode)*/
typedef struct{
  __REG32                 : 1;
  __REG32 VS              : 1;
  __REG32 NRGB            : 1;
  __REG32                 :17;
  __REG32 VI              : 1;
  __REG32                 : 3;
  __REG32 CM              : 2;
  __REG32                 : 2;
  __REG32 VICE            : 1;
  __REG32                 : 1;
  __REG32 VIS             : 1;
  __REG32 VIE             : 1;
} __vcxvcm_bits;

/*CSC (Capture SCale)*/
typedef struct{
  __REG32 HSCF            :11;
  __REG32 HSCI            : 5;
  __REG32 VSCF            :11;
  __REG32 VSCI            : 5;
} __vcxcsc_bits;

/*VCS (Video Capture Status)*/
typedef struct{
  __REG32 CE              : 5;
  __REG32                 :27;
} __vcxvcs_bits;

/*CBM (video Capture Buffer Mode)*/
typedef struct{
  __REG32 CBST            : 1;
  __REG32                 : 3;
  __REG32 HRV             : 1;
  __REG32 SSM             : 3;
  __REG32 SSS             : 3;
  __REG32                 : 1;
  __REG32 CSW             : 1;
  __REG32 BED             : 1;
  __REG32 C24             : 1;
  __REG32                 : 1;
  __REG32 CBW             : 8;
  __REG32                 : 4;
  __REG32 PAU             : 1;
  __REG32 CRGB            : 1;
  __REG32 SBUF            : 1;
  __REG32 OO              : 1;
} __vcxcbm_bits;

/*CIHSTR (Capture Image Horizontal STaRt)*/
typedef struct{
  __REG16 CIHSTR          :12;
  __REG16                 : 4;
} __vcxcihstr_bits;

/*CIVSTR (Capture Image Vertical STaRt)*/
typedef struct{
  __REG16 CIVSTR          :12;
  __REG16                 : 4;
} __vcxcivstr_bits;

/*CIHEND (Capture Image Horizontal END)*/
typedef struct{
  __REG16 CIHEND          :12;
  __REG16                 : 4;
} __vcxcihend_bits;

/*CIVEND (Capture Image Vertical END)*/
typedef struct{
  __REG16 CIVEND          :12;
  __REG16                 : 4;
} __vcxcivend_bits;

/*CHP (Capture Horizontal Pixel)*/
typedef struct{
  __REG32 CHP             :10;
  __REG32                 :22;
} __vcxchp_bits;

/*CVP (Capture Vertical Pixel)*/
typedef struct{
  __REG32 CVPN            :10;
  __REG32                 : 6;
  __REG32 CVPP            :10;
  __REG32                 : 6;
} __vcxcvp_bits;

/*CLPF (Capture Low Pass Filter)*/
typedef struct{
  __REG32                 :16;
  __REG32 CHLPF           : 4;
  __REG32                 : 4;
  __REG32 CVLPF           : 4;
  __REG32                 : 4;
} __vcxclpf_bits;

/*CMSS (Capture Magnify Source Size)*/
typedef struct{
  __REG32 CMSVL           :12;
  __REG32                 : 4;
  __REG32 CMSHP           :12;
  __REG32                 : 4;
} __vcxcmss_bits;

/*CMDS (Capture Magnify Display Size)*/
typedef struct{
  __REG32 CMDVL           :12;
  __REG32                 : 4;
  __REG32 CMDHP           :12;
  __REG32                 : 4;
} __vcxcmds_bits;

/*RGBHC(RGB input Hsync Cycle)*/
typedef struct{
  __REG32 RGBHC           :14;
  __REG32                 :18;
} __vcxrgbhc_bits;

/*CRGBHEN(RGB input Horizontal Enable area)*/
typedef struct{
  __REG32 RGBHEN          :13;
  __REG32                 : 3;
  __REG32 RGBHST          :12;
  __REG32                 : 4;
} __vcxrgbhen_bits;

/*RGBVEN(RGB input Vertical Enable area)*/
typedef struct{
  __REG32 RGBVEN          :13;
  __REG32                 : 3;
  __REG32 RGBVST          : 9;
  __REG32                 : 7;
} __vcxrgbven_bits;

/*RGBS (RGB input Sync)*/
typedef struct{
  __REG32 VP              : 1;
  __REG32 HP              : 1;
  __REG32                 :14;
  __REG32 RM              : 1;
  __REG32                 :15;
} __vcxrgbs_bits;

/*RGBCMY (RGB Color convert Matrix Y coefficient)*/
typedef struct{
  __REG32 a13             :10;
  __REG32                 : 1;
  __REG32 a12             :10;
  __REG32                 : 1;
  __REG32 a11             :10;
} __vcxrgbcmy_bits;

/*RGBCMCb (RGB Color convert Matrix Cb coefficient)*/
typedef struct{
  __REG32 a23             :10;
  __REG32                 : 1;
  __REG32 a22             :10;
  __REG32                 : 1;
  __REG32 a21             :10;
} __vcxrgbcmcb_bits;

/*RGBCMCr (RGB Color convert Matrix Cr coefficient)*/
typedef struct{
  __REG32 a33             :10;
  __REG32                 : 1;
  __REG32 a32             :10;
  __REG32                 : 1;
  __REG32 a31             :10;
} __vcxrgbcmcr_bits;

/*RGBCMb (RGB Color convert Matrix b coefficient)*/
typedef struct{
  __REG32 b3              : 9;
  __REG32                 : 2;
  __REG32 b2              : 9;
  __REG32                 : 2;
  __REG32 b1              : 9;
  __REG32                 : 1;
} __vcxrgbcmb_bits;

/*CVCNT (Capture Vertical Count)*/
typedef struct{
  __REG16 CVCNT           :12;
  __REG16                 : 4;
} __vcxcvcnt_bits;

/*CDCN (Capture Data Count for NTSC)*/
typedef struct{
  __REG32 VDCN            :13;
  __REG32                 : 3;
  __REG32 BDCN            :13;
  __REG32                 : 3;
} __vcxcdcn_bits;

/*CDCP (Capture Data Count for PAL)*/
typedef struct{
  __REG32 VDCP            :13;
  __REG32                 : 3;
  __REG32 BDCP            :13;
  __REG32                 : 3;
} __vcxcdcp_bits;

/*CAN Control Register*/
typedef struct{
  __REG32 Init            : 1;
  __REG32 IE              : 1;
  __REG32 SIE             : 1;
  __REG32 EIE             : 1;
  __REG32                 : 1;
  __REG32 DAR             : 1;
  __REG32 CCE             : 1;
  __REG32 Test            : 1;
  __REG32                 :24;
} __canxcr_bits;

/*CAN Status Register*/
typedef struct{
  __REG32 LEC             : 3;
  __REG32 TxOk            : 1;
  __REG32 RxOk            : 1;
  __REG32 EPass           : 1;
  __REG32 EWarn           : 1;
  __REG32 BOff            : 1;
  __REG32                 :24;
} __canxsr_bits;

/*CAN Error Counter*/
typedef struct{
  __REG32 TEC             : 8;
  __REG32 REC             : 7;
  __REG32 RP              : 1;
  __REG32                 :16;
} __canxec_bits;

/*CAN Bit Timing Register*/
typedef struct{
  __REG32 BRP             : 6;
  __REG32 SJW             : 2;
  __REG32 TSeg1           : 4;
  __REG32 TSeg2           : 3;
  __REG32                 :17;
} __canxbtr_bits;

/*CAN Interrupt Register*/
typedef struct{
  __REG32 IntID           :16;
  __REG32                 :16;
} __canxir_bits;

/*CAN Test Register*/
typedef struct{
  __REG32                 : 2;
  __REG32 Basic           : 1;
  __REG32 Silent          : 1;
  __REG32 LBack           : 1;
  __REG32 Tx              : 2;
  __REG32 Rx              : 1;
  __REG32                 :24;
} __canxtr_bits;

/*CAN BRP Extension Register*/
typedef struct{
  __REG32 BRPE            : 4;
  __REG32                 :28;
} __canxbrper_bits;

/*CAN IFx Command Request Registers*/
typedef struct{
  __REG32 MssgNum         : 6;
  __REG32                 : 9;
  __REG32 Busy            : 1;
  __REG32                 :16;
} __canxifxcr_bits;

/*CAN IFx Command Mask Registers*/
typedef struct{
  __REG32 DataB           : 1;
  __REG32 DataA           : 1;
  __REG32 TxRqst_NewDat   : 1;
  __REG32 ClrIntPnd       : 1;
  __REG32 Control         : 1;
  __REG32 Arb             : 1;
  __REG32 Mask            : 1;
  __REG32 WR_RD           : 1;
  __REG32                 :24;
} __canxifxcm_bits;

/*CAN IFx Command Mask 1 Registers*/
typedef struct{
  __REG32 Msk             :16;
  __REG32                 :16;
} __canxifxm1_bits;

/*CAN IFx Command Mask 2 Registers*/
typedef struct{
  __REG32 Msk             :13;
  __REG32                 : 1;
  __REG32 MDir            : 1;
  __REG32 MXtd            : 1;
  __REG32                 :16;
} __canxifxm2_bits;

/*CAN IFx Arbitration  1 Registers*/
typedef struct{
  __REG32 ID              :16;
  __REG32                 :16;
} __canxifxa1_bits;

/*CAN IFx Arbitration 2 Registers*/
typedef struct{
  __REG32 ID              :13;
  __REG32 Dir             : 1;
  __REG32 Xtd             : 1;
  __REG32 MsgVal          : 1;
  __REG32                 :16;
} __canxifxa2_bits;

/*CAN IFx Message Control Registers*/
typedef struct{
  __REG32 DLC             : 4;
  __REG32                 : 3;
  __REG32 EoB             : 1;
  __REG32 TxRqst          : 1;
  __REG32 RmtEn           : 1;
  __REG32 RxIE            : 1;
  __REG32 TxIE            : 1;
  __REG32 UMask           : 1;
  __REG32 IntPnd          : 1;
  __REG32 MsgLst          : 1;
  __REG32 NewDat          : 1;
  __REG32                 :16;
} __canxifxmc_bits;

/*CAN IFx  Message Data A1*/
typedef struct{
  __REG32 DATA0           : 8;
  __REG32 DATA1           : 8;
  __REG32                 :16;
} __canxifxda1_bits;

/*CAN IFx  Message Data A2*/
typedef struct{
  __REG32 DATA2           : 8;
  __REG32 DATA3           : 8;
  __REG32                 :16;
} __canxifxda2_bits;

/*CAN IFx  Message Data B1*/
typedef struct{
  __REG32 DATA4           : 8;
  __REG32 DATA5           : 8;
  __REG32                 :16;
} __canxifxdb1_bits;

/*CAN IFx  Message Data B2*/
typedef struct{
  __REG32 DATA6           : 8;
  __REG32 DATA7           : 8;
  __REG32                 :16;
} __canxifxdb2_bits;

/*CAN Transmission Request 1 Register*/
typedef struct{
  __REG32 TxRqst1         : 1;
  __REG32 TxRqst2         : 1;
  __REG32 TxRqst3         : 1;
  __REG32 TxRqst4         : 1;
  __REG32 TxRqst5         : 1;
  __REG32 TxRqst6         : 1;
  __REG32 TxRqst7         : 1;
  __REG32 TxRqst8         : 1;
  __REG32 TxRqst9         : 1;
  __REG32 TxRqst10        : 1;
  __REG32 TxRqst11        : 1;
  __REG32 TxRqst12        : 1;
  __REG32 TxRqst13        : 1;
  __REG32 TxRqst14        : 1;
  __REG32 TxRqst15        : 1;
  __REG32 TxRqst16        : 1;
  __REG32                 :16;
} __canxtr1_bits;

/*CAN Transmission Request 2 Register*/
typedef struct{
  __REG32 TxRqst17        : 1;
  __REG32 TxRqst18        : 1;
  __REG32 TxRqst19        : 1;
  __REG32 TxRqst20        : 1;
  __REG32 TxRqst21        : 1;
  __REG32 TxRqst22        : 1;
  __REG32 TxRqst23        : 1;
  __REG32 TxRqst24        : 1;
  __REG32 TxRqst25        : 1;
  __REG32 TxRqst26        : 1;
  __REG32 TxRqst27        : 1;
  __REG32 TxRqst28        : 1;
  __REG32 TxRqst29        : 1;
  __REG32 TxRqst30        : 1;
  __REG32 TxRqst31        : 1;
  __REG32 TxRqst32        : 1;
  __REG32                 :16;
} __canxtr2_bits;

/*CAN New Data 1 Register*/
typedef struct{
  __REG32 NewDat1         : 1;
  __REG32 NewDat2         : 1;
  __REG32 NewDat3         : 1;
  __REG32 NewDat4         : 1;
  __REG32 NewDat5         : 1;
  __REG32 NewDat6         : 1;
  __REG32 NewDat7         : 1;
  __REG32 NewDat8         : 1;
  __REG32 NewDat9         : 1;
  __REG32 NewDat10        : 1;
  __REG32 NewDat11        : 1;
  __REG32 NewDat12        : 1;
  __REG32 NewDat13        : 1;
  __REG32 NewDat14        : 1;
  __REG32 NewDat15        : 1;
  __REG32 NewDat16        : 1;
  __REG32                 :16;
} __canxnd1_bits;

/*CAN New Data 2 Register*/
typedef struct{
  __REG32 NewDat17        : 1;
  __REG32 NewDat18        : 1;
  __REG32 NewDat19        : 1;
  __REG32 NewDat20        : 1;
  __REG32 NewDat21        : 1;
  __REG32 NewDat22        : 1;
  __REG32 NewDat23        : 1;
  __REG32 NewDat24        : 1;
  __REG32 NewDat25        : 1;
  __REG32 NewDat26        : 1;
  __REG32 NewDat27        : 1;
  __REG32 NewDat28        : 1;
  __REG32 NewDat29        : 1;
  __REG32 NewDat30        : 1;
  __REG32 NewDat31        : 1;
  __REG32 NewDat32        : 1;
  __REG32                 :16;
} __canxnd2_bits ;

/*CAN Interrupt Pending 1 Register*/
typedef struct{
  __REG32 IntPnd1         : 1;
  __REG32 IntPnd2         : 1;
  __REG32 IntPnd3         : 1;
  __REG32 IntPnd4         : 1;
  __REG32 IntPnd5         : 1;
  __REG32 IntPnd6         : 1;
  __REG32 IntPnd7         : 1;
  __REG32 IntPnd8         : 1;
  __REG32 IntPnd9         : 1;
  __REG32 IntPnd10        : 1;
  __REG32 IntPnd11        : 1;
  __REG32 IntPnd12        : 1;
  __REG32 IntPnd13        : 1;
  __REG32 IntPnd14        : 1;
  __REG32 IntPnd15        : 1;
  __REG32 IntPnd16        : 1;
  __REG32                 :16;
} __canxip1_bits;

/*CAN Interrupt Pending 2 Register*/
typedef struct{
  __REG32 IntPnd17        : 1;
  __REG32 IntPnd18        : 1;
  __REG32 IntPnd19        : 1;
  __REG32 IntPnd20        : 1;
  __REG32 IntPnd21        : 1;
  __REG32 IntPnd22        : 1;
  __REG32 IntPnd23        : 1;
  __REG32 IntPnd24        : 1;
  __REG32 IntPnd25        : 1;
  __REG32 IntPnd26        : 1;
  __REG32 IntPnd27        : 1;
  __REG32 IntPnd28        : 1;
  __REG32 IntPnd29        : 1;
  __REG32 IntPnd30        : 1;
  __REG32 IntPnd31        : 1;
  __REG32 IntPnd32        : 1;
  __REG32                 :16;
} __canxip2_bits;

/*CAN Message Valid 1 Register*/
typedef struct{
  __REG32 MsgVal1         : 1;
  __REG32 MsgVal2         : 1;
  __REG32 MsgVal3         : 1;
  __REG32 MsgVal4         : 1;
  __REG32 MsgVal5         : 1;
  __REG32 MsgVal6         : 1;
  __REG32 MsgVal7         : 1;
  __REG32 MsgVal8         : 1;
  __REG32 MsgVal9         : 1;
  __REG32 MsgVal10        : 1;
  __REG32 MsgVal11        : 1;
  __REG32 MsgVal12        : 1;
  __REG32 MsgVal13        : 1;
  __REG32 MsgVal14        : 1;
  __REG32 MsgVal15        : 1;
  __REG32 MsgVal16        : 1;
  __REG32                 :16;
} __canxmv1_bits;

/*CAN Message Valid 2 Register*/
typedef struct{
  __REG32 MsgVal17        : 1;
  __REG32 MsgVal18        : 1;
  __REG32 MsgVal19        : 1;
  __REG32 MsgVal20        : 1;
  __REG32 MsgVal21        : 1;
  __REG32 MsgVal22        : 1;
  __REG32 MsgVal23        : 1;
  __REG32 MsgVal24        : 1;
  __REG32 MsgVal25        : 1;
  __REG32 MsgVal26        : 1;
  __REG32 MsgVal27        : 1;
  __REG32 MsgVal28        : 1;
  __REG32 MsgVal29        : 1;
  __REG32 MsgVal30        : 1;
  __REG32 MsgVal31        : 1;
  __REG32 MsgVal32        : 1;
  __REG32                 :16;
} __canxmv2_bits;

/*Timer Control Register*/
typedef struct{
  __REG32 OSC             : 1;
  __REG32 TS              : 1;
  __REG32 TP              : 2;
  __REG32                 : 1;
  __REG32 IE              : 1;
  __REG32 TM              : 1;
  __REG32 TE              : 1;
  __REG32                 :24;
} __timercontrol_bits;

/*Raw Interrupt Status Register*/
typedef struct{
  __REG32 RTI             : 1;
  __REG32                 :31;
} __timerris_bits;

/*Interrupt Status Register*/
typedef struct{
  __REG32 TI              : 1;
  __REG32                 :31;
} __timermis_bits;

/*Integration Test Control Register*/
typedef struct{
  __REG32 ITME            : 1;
  __REG32                 :31;
} __timeritcr_bits;

/*Integration Test Output Set Register*/
typedef struct{
  __REG32 TIMINT1         : 1;
  __REG32 TIMINT2         : 1;
  __REG32                 :30;
} __timeritop_bits;

/*Peripheral Identification Register 0*/
typedef struct{
  __REG32 PARTNUMBER0     : 8;
  __REG32                 :24;
} __timerperiphid0_bits;

/*Peripheral Identification Register 1*/
typedef struct{
  __REG32 PARTNUMBER1     : 4;
  __REG32 DESIGNER0       : 4;
  __REG32                 :24;
} __timerperiphid1_bits;

/*Peripheral Identification Register 2*/
typedef struct{
  __REG32 DESIGNER1       : 4;
  __REG32 REVISION        : 4;
  __REG32                 :24;
} __timerperiphid2_bits;

/*Peripheral Identification Register */
typedef struct{
  __REG32 CONFIGURATION   : 8;
  __REG32                 :24;
} __timerperiphid3_bits;

/*PrimeCell Identification Register 0*/
typedef struct{
  __REG32 TIMERPCELLID0   : 8;
  __REG32                 :24;
} __timerpcellid0_bits;

/*PrimeCell Identification Register 1*/
typedef struct{
  __REG32 TIMERPCELLID1   : 8;
  __REG32                 :24;
} __timerpcellid1_bits;

/*PrimeCell Identification Register 2*/
typedef struct{
  __REG32 TIMERPCELLID2   : 8;
  __REG32                 :24;
} __timerpcellid2_bits;

/*PrimeCell Identification Register 3*/
typedef struct{
  __REG32 TIMERPCELLID3   : 8;
  __REG32                 :24;
} __timerpcellid3_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** CRG
 **
 ***************************************************************************/
__IO_REG32_BIT(CRPR,               0xFFFE7000 , __READ_WRITE , __crpr_bits);
__IO_REG32_BIT(CRWR,               0xFFFE7008 , __READ_WRITE , __crwr_bits);
__IO_REG32_BIT(CRSR,               0xFFFE700C , __READ_WRITE , __crsr_bits);
__IO_REG32_BIT(CRDA,               0xFFFE7010 , __READ_WRITE , __crda_bits);
__IO_REG32_BIT(CRDB,               0xFFFE7014 , __READ_WRITE , __crdb_bits);
__IO_REG32_BIT(CRHA,               0xFFFE7018 , __READ_WRITE , __crha_bits);
__IO_REG32_BIT(CRPA,               0xFFFE701C , __READ_WRITE , __crpa_bits);
__IO_REG32_BIT(CRPB,               0xFFFE7020 , __READ_WRITE , __crpb_bits);
__IO_REG32_BIT(CRHB,               0xFFFE7024 , __READ_WRITE , __crhb_bits);
__IO_REG32_BIT(CRAM,               0xFFFE7028 , __READ_WRITE , __cram_bits);

/***************************************************************************
 **
 ** RBC
 **
 ***************************************************************************/
__IO_REG32_BIT(RBREMAP,            0xFFFE6004 , __READ_WRITE , __rbremap_bits);
__IO_REG32_BIT(RBVIHA,             0xFFFE6008 , __READ_WRITE , __rbviha_bits);
__IO_REG32_BIT(RBITRA,             0xFFFE600C , __READ_WRITE , __rbitra_bits);

/***************************************************************************
 **
 ** IRQ0
 **
 ***************************************************************************/
__IO_REG32_BIT(IR0IRQF,            0xFFFE8000 , __READ_WRITE , __irxirqf_bits);
__IO_REG32_BIT(IR0IRQM,            0xFFFE8004 , __READ_WRITE , __irxirqm_bits);
__IO_REG32_BIT(IR0ILM,             0xFFFE8008 , __READ_WRITE , __irxilm_bits);
__IO_REG32_BIT(IR0ICRMN,           0xFFFE800C , __READ_WRITE , __irxicrmn_bits);
__IO_REG32_BIT(IR0DICR0,           0xFFFE8014 , __READ_WRITE , __ir0dicr0_bits);
__IO_REG32_BIT(IR0DICR1,           0xFFFE8018 , __READ_WRITE , __ir0dicr1_bits);
__IO_REG32_BIT(IR0TBR,             0xFFFE801C , __READ_WRITE , __irxtbr_bits);
__IO_REG32(    IR0VCT,             0xFFFE8020 , __READ       );
__IO_REG32_BIT(IR0ICR06,           0xFFFE8048 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR07,           0xFFFE804C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR08,           0xFFFE8050 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR09,           0xFFFE8054 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR10,           0xFFFE8058 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR11,           0xFFFE805C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR12,           0xFFFE8060 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR13,           0xFFFE8064 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR14,           0xFFFE8068 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR15,           0xFFFE806C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR16,           0xFFFE8070 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR17,           0xFFFE8074 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR18,           0xFFFE8078 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR19,           0xFFFE807C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR20,           0xFFFE8080 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR21,           0xFFFE8084 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR22,           0xFFFE8088 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR23,           0xFFFE808C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR24,           0xFFFE8090 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR25,           0xFFFE8094 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR28,           0xFFFE80A0 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR29,           0xFFFE80A4 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR0ICR30,           0xFFFE80A8 , __READ_WRITE , __IRxICRy_bits);

/***************************************************************************
 **
 ** IRQ1
 **
 ***************************************************************************/
__IO_REG32_BIT(IR1IRQF,            0xFFFB0000 , __READ_WRITE , __irxirqf_bits);
__IO_REG32_BIT(IR1IRQM,            0xFFFB0004 , __READ_WRITE , __irxirqm_bits);
__IO_REG32_BIT(IR1ILM,             0xFFFB0008 , __READ_WRITE , __irxilm_bits);
__IO_REG32_BIT(IR1ICRMN,           0xFFFB000C , __READ_WRITE , __irxicrmn_bits);
__IO_REG32_BIT(IR1TBR,             0xFFFB001C , __READ_WRITE , __irxtbr_bits);
__IO_REG32(    IR1VCT,             0xFFFB0020 , __READ       );
__IO_REG32_BIT(IR1ICR00,           0xFFFB0030 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR02,           0xFFFB0038 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR03,           0xFFFB003C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR04,           0xFFFB0040 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR05,           0xFFFB0044 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR06,           0xFFFB0048 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR07,           0xFFFB004C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR08,           0xFFFB0050 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR09,           0xFFFB0054 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR10,           0xFFFB0058 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR11,           0xFFFB005C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR12,           0xFFFB0060 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR13,           0xFFFB0064 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR14,           0xFFFB0068 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR15,           0xFFFB006C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR16,           0xFFFB0070 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR17,           0xFFFB0074 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR18,           0xFFFB0078 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR19,           0xFFFB007C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR20,           0xFFFB0080 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR21,           0xFFFB0084 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR22,           0xFFFB0088 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR23,           0xFFFB008C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR24,           0xFFFB0090 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR26,           0xFFFB0098 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR27,           0xFFFB009C , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR28,           0xFFFB00A0 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR29,           0xFFFB00A4 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR30,           0xFFFB00A8 , __READ_WRITE , __IRxICRy_bits);
__IO_REG32_BIT(IR1ICR31,           0xFFFB00AC , __READ_WRITE , __IRxICRy_bits);

/***************************************************************************
 **
 ** External Bus I/F
 **
 ***************************************************************************/
__IO_REG32_BIT(MCFMODE0,           0xFFFC0000 , __READ_WRITE , __mcfmodex_bits);
__IO_REG32_BIT(MCFMODE2,           0xFFFC0008 , __READ_WRITE , __mcfmodex_bits);
__IO_REG32_BIT(MCFMODE4,           0xFFFC0010 , __READ_WRITE , __mcfmodex_bits);
__IO_REG32_BIT(MCFTIM0,            0xFFFC0020 , __READ_WRITE , __mcftimx_bits);
__IO_REG32_BIT(MCFTIM2,            0xFFFC0028 , __READ_WRITE , __mcftimx_bits);
__IO_REG32_BIT(MCFTIM4,            0xFFFC0030 , __READ_WRITE , __mcftimx_bits);
__IO_REG32_BIT(MCFAREA0,           0xFFFC0040 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA1,           0xFFFC0044 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA2,           0xFFFC0048 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA3,           0xFFFC004C , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA4,           0xFFFC0050 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA5,           0xFFFC0054 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA6,           0xFFFC0058 , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCFAREA7,           0xFFFC005C , __READ_WRITE , __mcfareax_bits);
__IO_REG32_BIT(MCERR,              0xFFFC0200 , __READ_WRITE , __mcerr_bits);

/***************************************************************************
 **
 ** DDR2C
 **
 ***************************************************************************/
__IO_REG16_BIT(DRIC,               0xF3000000 , __READ_WRITE , __dric_bits);
__IO_REG16_BIT(DRIC1,              0xF3000002 , __READ_WRITE , __dric1_bits);
__IO_REG16_BIT(DRIC2,              0xF3000004 , __READ_WRITE , __dric2_bits);
__IO_REG16_BIT(DRCA,               0xF3000006 , __READ_WRITE , __drca_bits);
__IO_REG16_BIT(DRCM,               0xF3000008 , __READ_WRITE , __drcm_bits);
__IO_REG16_BIT(DRCST1,             0xF300000A , __READ_WRITE , __drcst1_bits);
__IO_REG16_BIT(DRCST2,             0xF300000C , __READ_WRITE , __drcst2_bits);
__IO_REG16_BIT(DRCR,               0xF300000E , __READ_WRITE , __drcr_bits);
__IO_REG16_BIT(DRCF,               0xF3000020 , __READ_WRITE , __drcf_bits);
__IO_REG16_BIT(DRASR,              0xF3000030 , __READ_WRITE , __drasr_bits);
__IO_REG16_BIT(DRIMSD,             0xF3000050 , __READ_WRITE , __drimsd_bits);
__IO_REG16_BIT(DROS,               0xF3000060 , __READ_WRITE , __dros_bits);
__IO_REG16_BIT(DRIBSODT1,          0xF3000064 , __READ_WRITE , __dribsodt1_bits);
__IO_REG16_BIT(DRIBSOCD,           0xF3000066 , __READ_WRITE , __dribsocd_bits);
__IO_REG16_BIT(DRIBSOCD2,          0xF3000068 , __READ_WRITE , __dribsocd2_bits);
__IO_REG16_BIT(DROABA,             0xF3000070 , __READ_WRITE , __droaba_bits);
__IO_REG16_BIT(DROBS,              0xF3000084 , __READ_WRITE , __drobs_bits);
__IO_REG16_BIT(DRIMR1,             0xF3000090 , __READ       , __drimr1_bits);
__IO_REG16_BIT(DRIMR2,             0xF3000092 , __READ       , __drimr2_bits);
__IO_REG16_BIT(DRIMR3,             0xF3000094 , __READ       , __drimr3_bits);
__IO_REG16_BIT(DRIMR4,             0xF3000096 , __READ       , __drimr4_bits);
__IO_REG16_BIT(DROISR1,            0xF3000098 , __READ_WRITE , __droisr1_bits);
__IO_REG16_BIT(DROISR2,            0xF300009A , __READ_WRITE , __droisr2_bits);

/***************************************************************************
 **
 ** DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACR,              0xFFFD0000 , __READ_WRITE , __dmacr_bits);
__IO_REG32_BIT(DMACA0,             0xFFFD0010 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB0,             0xFFFD0014 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA0,            0xFFFD0018 , __READ_WRITE );
__IO_REG32(    DMACDA0,            0xFFFD001C , __READ_WRITE );
__IO_REG32_BIT(DMACA1,             0xFFFD0020 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB1,             0xFFFD0024 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA1,            0xFFFD0028 , __READ_WRITE );
__IO_REG32(    DMACDA1,            0xFFFD002C , __READ_WRITE );
__IO_REG32_BIT(DMACA2,             0xFFFD0030 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB2,             0xFFFD0034 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA2,            0xFFFD0038 , __READ_WRITE );
__IO_REG32(    DMACDA2,            0xFFFD003C , __READ_WRITE );
__IO_REG32_BIT(DMACA3,             0xFFFD0040 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB3,             0xFFFD0044 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA3,            0xFFFD0048 , __READ_WRITE );
__IO_REG32(    DMACDA3,            0xFFFD004C , __READ_WRITE );
__IO_REG32_BIT(DMACA4,             0xFFFD0050 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB4,             0xFFFD0054 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA4,            0xFFFD0058 , __READ_WRITE );
__IO_REG32(    DMACDA4,            0xFFFD005C , __READ_WRITE );
__IO_REG32_BIT(DMACA5,             0xFFFD0060 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB5,             0xFFFD0064 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA5,            0xFFFD0068 , __READ_WRITE );
__IO_REG32(    DMACDA5,            0xFFFD006C , __READ_WRITE );
__IO_REG32_BIT(DMACA6,             0xFFFD0070 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB6,             0xFFFD0074 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA6,            0xFFFD0078 , __READ_WRITE );
__IO_REG32(    DMACDA6,            0xFFFD007C , __READ_WRITE );
__IO_REG32_BIT(DMACA7,             0xFFFD0080 , __READ_WRITE , __dmacax_bits);
__IO_REG32_BIT(DMACB7,             0xFFFD0084 , __READ_WRITE , __dmacbx_bits);
__IO_REG32(    DMACSA7,            0xFFFD0088 , __READ_WRITE );
__IO_REG32(    DMACDA7,            0xFFFD008C , __READ_WRITE );

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPDR0,              0xFFFE9000 , __READ_WRITE , __gpdr0_bits);
__IO_REG32_BIT(GPDR1,              0xFFFE9004 , __READ_WRITE , __gpdr1_bits);
__IO_REG32_BIT(GPDR2,              0xFFFE9008 , __READ_WRITE , __gpdr2_bits);
__IO_REG32_BIT(GPDDR0,             0xFFFE9010 , __READ_WRITE , __gpdr0_bits);
__IO_REG32_BIT(GPDDR1,             0xFFFE9014 , __READ_WRITE , __gpdr1_bits);
__IO_REG32_BIT(GPDDR2,             0xFFFE9018 , __READ_WRITE , __gpdr2_bits);

/***************************************************************************
 **
 ** PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0BCR,            0xFFF41000 , __READ_WRITE , __PWMxBCR_bits);
__IO_REG32_BIT(PWM0TPR,            0xFFF41004 , __READ_WRITE , __PWMxTPR_bits);
__IO_REG32_BIT(PWM0PR,             0xFFF41008 , __READ_WRITE , __PWMxPR_bits);
__IO_REG32_BIT(PWM0DR,             0xFFF4100C , __READ_WRITE , __PWMxDR_bits);
__IO_REG32_BIT(PWM0CR,             0xFFF41010 , __READ_WRITE , __PWMxCR_bits);
__IO_REG32_BIT(PWM0SR,             0xFFF41014 , __READ_WRITE , __PWMxSR_bits);
__IO_REG32_BIT(PWM0CCR,            0xFFF41018 , __READ       , __PWMxCCR_bits);
__IO_REG32_BIT(PWM0IR,             0xFFF4101C , __READ_WRITE , __PWMxIR_bits);

/***************************************************************************
 **
 ** PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM1BCR,            0xFFF41100 , __READ_WRITE , __PWMxBCR_bits);
__IO_REG32_BIT(PWM1TPR,            0xFFF41104 , __READ_WRITE , __PWMxTPR_bits);
__IO_REG32_BIT(PWM1PR,             0xFFF41108 , __READ_WRITE , __PWMxPR_bits);
__IO_REG32_BIT(PWM1DR,             0xFFF4110C , __READ_WRITE , __PWMxDR_bits);
__IO_REG32_BIT(PWM1CR,             0xFFF41110 , __READ_WRITE , __PWMxCR_bits);
__IO_REG32_BIT(PWM1SR,             0xFFF41114 , __READ_WRITE , __PWMxSR_bits);
__IO_REG32_BIT(PWM1CCR,            0xFFF41118 , __READ       , __PWMxCCR_bits);
__IO_REG32_BIT(PWM1IR,             0xFFF4111C , __READ_WRITE , __PWMxIR_bits);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0DATA,           0xFFF52000 , __READ       , __ADCxDATA_bits);
__IO_REG32_BIT(ADC0XPD,            0xFFF52008 , __READ_WRITE , __ADCxXPD_bits);
__IO_REG32_BIT(ADC0CKSEL,          0xFFF52010 , __READ_WRITE , __ADCxCKSEL_bits);
__IO_REG32_BIT(ADC0STATUS,         0xFFF52014 , __READ_WRITE , __ADCxSTATUS_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1DATA,           0xFFF53000 , __READ       , __ADCxDATA_bits);
__IO_REG32_BIT(ADC1XPD,            0xFFF53008 , __READ_WRITE , __ADCxXPD_bits);
__IO_REG32_BIT(ADC1CKSEL,          0xFFF53010 , __READ_WRITE , __ADCxCKSEL_bits);
__IO_REG32_BIT(ADC1STATUS,         0xFFF53014 , __READ_WRITE , __ADCxSTATUS_bits);

/***************************************************************************
 **
 ** I2S0
 **
 ***************************************************************************/
__IO_REG32(    I2S0RXFDAT,         0xFFEE0000 , __READ       );
__IO_REG32(    I2S0TXFDAT,         0xFFEE0004 , __WRITE      );
__IO_REG32_BIT(I2S0CNTREG,         0xFFEE0008 , __READ_WRITE , __i2sxcntreg_bits);
__IO_REG32_BIT(I2S0MCR0REG,        0xFFEE000C , __READ_WRITE , __i2sxmcr0reg_bits);
__IO_REG32_BIT(I2S0MCR1REG,        0xFFEE0010 , __READ_WRITE , __i2sxmcr1reg_bits);
__IO_REG32_BIT(I2S0MCR2REG,        0xFFEE0014 , __READ_WRITE , __i2sxmcr2reg_bits);
__IO_REG32_BIT(I2S0OPRREG,         0xFFEE0018 , __READ_WRITE , __i2sxoprreg_bits);
__IO_REG32_BIT(I2S0SRST,           0xFFEE001C , __READ_WRITE , __i2sxsrst_bits);
__IO_REG32_BIT(I2S0INTCNT,         0xFFEE0020 , __READ_WRITE , __i2sxintcnt_bits);
__IO_REG32_BIT(I2S0STATUS,         0xFFEE0024 , __READ_WRITE , __i2sxstatus_bits);
__IO_REG32_BIT(I2S0DMAACT,         0xFFEE0028 , __READ_WRITE , __i2sxdmaact_bits);

/***************************************************************************
 **
 ** I2S1
 **
 ***************************************************************************/
__IO_REG32(    I2S1RXFDAT,         0xFFEF0000 , __READ       );
__IO_REG32(    I2S1TXFDAT,         0xFFEF0004 , __WRITE      );
__IO_REG32_BIT(I2S1CNTREG,         0xFFEF0008 , __READ_WRITE , __i2sxcntreg_bits);
__IO_REG32_BIT(I2S1MCR0REG,        0xFFEF000C , __READ_WRITE , __i2sxmcr0reg_bits);
__IO_REG32_BIT(I2S1MCR1REG,        0xFFEF0010 , __READ_WRITE , __i2sxmcr1reg_bits);
__IO_REG32_BIT(I2S1MCR2REG,        0xFFEF0014 , __READ_WRITE , __i2sxmcr2reg_bits);
__IO_REG32_BIT(I2S1OPRREG,         0xFFEF0018 , __READ_WRITE , __i2sxoprreg_bits);
__IO_REG32_BIT(I2S1SRST,           0xFFEF001C , __READ_WRITE , __i2sxsrst_bits);
__IO_REG32_BIT(I2S1INTCNT,         0xFFEF0020 , __READ_WRITE , __i2sxintcnt_bits);
__IO_REG32_BIT(I2S1STATUS,         0xFFEF0024 , __READ_WRITE , __i2sxstatus_bits);
__IO_REG32_BIT(I2S1DMAACT,         0xFFEF0028 , __READ_WRITE , __i2sxdmaact_bits);

/***************************************************************************
 **
 ** I2S2
 **
 ***************************************************************************/
__IO_REG32(    I2S2RXFDAT,         0xFFF00000 , __READ       );
__IO_REG32(    I2S2TXFDAT,         0xFFF00004 , __WRITE      );
__IO_REG32_BIT(I2S2CNTREG,         0xFFF00008 , __READ_WRITE , __i2sxcntreg_bits);
__IO_REG32_BIT(I2S2MCR0REG,        0xFFF0000C , __READ_WRITE , __i2sxmcr0reg_bits);
__IO_REG32_BIT(I2S2MCR1REG,        0xFFF00010 , __READ_WRITE , __i2sxmcr1reg_bits);
__IO_REG32_BIT(I2S2MCR2REG,        0xFFF00014 , __READ_WRITE , __i2sxmcr2reg_bits);
__IO_REG32_BIT(I2S2OPRREG,         0xFFF00018 , __READ_WRITE , __i2sxoprreg_bits);
__IO_REG32_BIT(I2S2SRST,           0xFFF0001C , __READ_WRITE , __i2sxsrst_bits);
__IO_REG32_BIT(I2S2INTCNT,         0xFFF00020 , __READ_WRITE , __i2sxintcnt_bits);
__IO_REG32_BIT(I2S2STATUS,         0xFFF00024 , __READ_WRITE , __i2sxstatus_bits);
__IO_REG32_BIT(I2S2DMAACT,         0xFFF00028 , __READ_WRITE , __i2sxdmaact_bits);

/***************************************************************************
 **
 ** UART0
 **
 ***************************************************************************/
/* URT0RFR, URT0TFR and URT0DLL share the same address */
__IO_REG8(     URT0RFR,            0xFFFE1000 , __READ_WRITE);
#define URT0TFR URT0RFR
#define URT0DLL URT0RFR

/* URT0IER and URT0DLM share the same address */
__IO_REG32_BIT(URT0IER,            0xFFFE1004 , __READ_WRITE , __urtxier_bits);
#define URT0DLM URT0IER

/* URT0IIR and URT0FCR share the same address */
__IO_REG32_BIT(URT0IIR,            0xFFFE1008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT0FCR     URT0IIR
#define URT0FCR_bit URT0IIR_bit

__IO_REG32_BIT(URT0LCR,            0xFFFE100C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT0MCR,            0xFFFE1010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT0LSR,            0xFFFE1014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT0MSR,            0xFFFE1018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** UART1
 **
 ***************************************************************************/
/* URT1RFR, URT1TFR and URT1DLL share the same address */
__IO_REG8(     URT1RFR,            0xFFFE2000 , __READ_WRITE);
#define URT1TFR URT1RFR
#define URT1DLL URT1RFR

/* URT1IER and URT1DLM share the same address */
__IO_REG32_BIT(URT1IER,            0xFFFE2004 , __READ_WRITE , __urtxier_bits);
#define URT1DLM URT1IER

/* URT1IIR and URT1FCR share the same address */
__IO_REG32_BIT(URT1IIR,            0xFFFE2008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT1FCR     URT1IIR
#define URT1FCR_bit URT1IIR_bit

__IO_REG32_BIT(URT1LCR,            0xFFFE200C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT1MCR,            0xFFFE2010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT1LSR,            0xFFFE2014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT1MSR,            0xFFFE2018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** UART2
 **
 ***************************************************************************/
/* URT2RFR, URT2TFR and URT2DLL share the same address */
__IO_REG8(     URT2RFR,            0xFFF50000 , __READ_WRITE);
#define URT2TFR URT2RFR
#define URT2DLL URT2RFR

/* URT2IER and URT2DLM share the same address */
__IO_REG32_BIT(URT2IER,            0xFFF50004 , __READ_WRITE , __urtxier_bits);
#define URT2DLM URT2IER

/* URT2IIR and URT2FCR share the same address */
__IO_REG32_BIT(URT2IIR,            0xFFF50008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT2FCR     URT2IIR
#define URT2FCR_bit URT2IIR_bit

__IO_REG32_BIT(URT2LCR,            0xFFF5000C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT2MCR,            0xFFF50010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT2LSR,            0xFFF50014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT2MSR,            0xFFF50018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** UART3
 **
 ***************************************************************************/
/* URT3RFR, URT3TFR and URT3DLL share the same address */
__IO_REG8(     URT3RFR,            0xFFF51000 , __READ_WRITE);
#define URT3TFR URT3RFR
#define URT3DLL URT3RFR

/* URT3IER and URT3DLM share the same address */
__IO_REG32_BIT(URT3IER,            0xFFF51004 , __READ_WRITE , __urtxier_bits);
#define URT3DLM URT3IER

/* URT3IIR and URT3FCR share the same address */
__IO_REG32_BIT(URT3IIR,            0xFFF51008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT3FCR     URT3IIR
#define URT3FCR_bit URT3IIR_bit

__IO_REG32_BIT(URT3LCR,            0xFFF5100C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT3MCR,            0xFFF51010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT3LSR,            0xFFF51014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT3MSR,            0xFFF51018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
/* URT4RFR, URT4TFR and URT4DLL share the same address */
__IO_REG8(     URT4RFR,            0xFFF43000 , __READ_WRITE);
#define URT4TFR URT4RFR
#define URT4DLL URT4RFR

/* URT4IER and URT4DLM share the same address */
__IO_REG32_BIT(URT4IER,            0xFFF43004 , __READ_WRITE , __urtxier_bits);
#define URT4DLM URT4IER

/* URT4IIR and URT4FCR share the same address */
__IO_REG32_BIT(URT4IIR,            0xFFF43008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT4FCR     URT4IIR
#define URT4FCR_bit URT4IIR_bit

__IO_REG32_BIT(URT4LCR,            0xFFF4300C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT4MCR,            0xFFF43010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT4LSR,            0xFFF43014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT4MSR,            0xFFF43018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** UART5
 **
 ***************************************************************************/
/* URT5RFR, URT5TFR and URT5DLL share the same address */
__IO_REG8(     URT5RFR,            0xFFF44000 , __READ_WRITE);
#define URT5TFR URT5RFR
#define URT5DLL URT5RFR

/* URT5IER and URT5DLM share the same address */
__IO_REG32_BIT(URT5IER,            0xFFF44004 , __READ_WRITE , __urtxier_bits);
#define URT5DLM URT5IER

/* URT5IIR and URT5FCR share the same address */
__IO_REG32_BIT(URT5IIR,            0xFFF44008 , __READ_WRITE , __urtxiir_fcr_bits);
#define URT5FCR     URT5IIR
#define URT5FCR_bit URT5IIR_bit

__IO_REG32_BIT(URT5LCR,            0xFFF4400C , __READ_WRITE , __urtxlcr_bits);
__IO_REG32_BIT(URT5MCR,            0xFFF44010 , __READ_WRITE , __urtxmcr_bits);
__IO_REG32_BIT(URT5LSR,            0xFFF44014 , __READ       , __urtxlsr_bits);
__IO_REG32_BIT(URT5MSR,            0xFFF44018 , __READ       , __urtxmsr_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0BSR,            0xFFF56000 , __READ       , __i2cxbsr_bits);
__IO_REG32_BIT(I2C0BCR,            0xFFF56004 , __READ_WRITE , __i2cxbcr_bits);
__IO_REG32_BIT(I2C0CCR,            0xFFF56008 , __READ_WRITE , __i2cxccr_bits);
__IO_REG32_BIT(I2C0ADR,            0xFFF5600C , __READ_WRITE , __i2cxadr_bits);
__IO_REG32_BIT(I2C0DAR,            0xFFF56010 , __READ_WRITE , __i2cxdar_bits);
__IO_REG32_BIT(I2C0ECSR,           0xFFF56014 , __READ_WRITE , __i2cxecsr_bits);
__IO_REG32_BIT(I2C0BCFR,           0xFFF56018 , __READ_WRITE , __i2cxbcfr_bits);
__IO_REG32_BIT(I2C0BC2R,           0xFFF5601C , __READ_WRITE , __i2cxbc2r_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1BSR,            0xFFF57000 , __READ       , __i2cxbsr_bits);
__IO_REG32_BIT(I2C1BCR,            0xFFF57004 , __READ_WRITE , __i2cxbcr_bits);
__IO_REG32_BIT(I2C1CCR,            0xFFF57008 , __READ_WRITE , __i2cxccr_bits);
__IO_REG32_BIT(I2C1ADR,            0xFFF5700C , __READ_WRITE , __i2cxadr_bits);
__IO_REG32_BIT(I2C1DAR,            0xFFF57010 , __READ_WRITE , __i2cxdar_bits);
__IO_REG32_BIT(I2C1ECSR,           0xFFF57014 , __READ_WRITE , __i2cxecsr_bits);
__IO_REG32_BIT(I2C1BCFR,           0xFFF57018 , __READ_WRITE , __i2cxbcfr_bits);
__IO_REG32_BIT(I2C1BC2R,           0xFFF5701C , __READ_WRITE , __i2cxbc2r_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPICR,              0xFFF40000 , __READ_WRITE , __spicr_bits);
__IO_REG32_BIT(SPISCR,             0xFFF40004 , __READ_WRITE , __spiscr_bits);
__IO_REG32(    SPIDR,              0xFFF40008 , __READ_WRITE );
__IO_REG32_BIT(SPISR,              0xFFF4000C , __READ       , __spisr_bits);

/***************************************************************************
 **
 ** USB Host controller
 **
 ***************************************************************************/
__IO_REG32_BIT(HCCAPBASE,             0xFFF80000,__READ      ,__hccapbase_bits);
__IO_REG32_BIT(HCSPARAMS,             0xFFF80004,__READ      ,__hcsparams_bits);
__IO_REG32_BIT(HCCPARAMS,             0xFFF80008,__READ      ,__hccparams_bits);
__IO_REG32_BIT(USBCMD,                0xFFF80010,__READ_WRITE,__usbcmd_bits);
__IO_REG32_BIT(USBSTS,                0xFFF80014,__READ_WRITE,__usbsts_bits);
__IO_REG32_BIT(USBINTR,               0xFFF80018,__READ_WRITE,__usbintr_bits);
__IO_REG32_BIT(FRINDEX,               0xFFF8001C,__READ_WRITE,__frindex_bits);
__IO_REG32_BIT(PERIODICLISTBASE,      0xFFF80024,__READ_WRITE,__periodiclistbase_bits);
__IO_REG32_BIT(ASYNCLISTADDR,         0xFFF80028,__READ_WRITE,__asynclistaddr_bits);
__IO_REG32_BIT(CONFIGFLAG,            0xFFF80050,__READ_WRITE,__configflag_bits);
__IO_REG32_BIT(PORTSC_1,              0xFFF80054,__READ_WRITE,__portsc_1_bits);
__IO_REG32_BIT(INSNREG00,             0xFFF80090,__READ_WRITE,__insnreg00_bits);
__IO_REG32_BIT(INSNREG01,             0xFFF80094,__READ_WRITE,__insnreg01_bits);
__IO_REG32_BIT(INSNREG02,             0xFFF80098,__READ_WRITE,__insnreg02_bits);
__IO_REG32_BIT(INSNREG03,             0xFFF8009C,__READ_WRITE,__insnreg03_bits);
__IO_REG32_BIT(HcRevision,            0xFFF81000,__READ      ,__HcRevision_bits);
__IO_REG32_BIT(HcControl,             0xFFF81004,__READ_WRITE,__HcControl_bits);
__IO_REG32_BIT(HcCommandStatus,       0xFFF81008,__READ_WRITE,__HcCommandStatus_bits);
__IO_REG32_BIT(HcInterruptStatus,     0xFFF8100C,__READ_WRITE,__HcInterruptStatus_bits);
__IO_REG32_BIT(HcInterruptEnable,     0xFFF81010,__READ_WRITE,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcInterruptDisable,    0xFFF81014,__READ_WRITE,__HcInterruptEnable_bits);
__IO_REG32_BIT(HcHCCA,                0xFFF81018,__READ_WRITE,__HcHCCA_bits);
__IO_REG32_BIT(HcPeriodCurrentED,     0xFFF8101C,__READ      ,__HcPeriodCurrentED_bits);
__IO_REG32_BIT(HcControlHeadED,       0xFFF81020,__READ_WRITE,__HcControlHeadED_bits);
__IO_REG32_BIT(HcControlCurrentED,    0xFFF81024,__READ_WRITE,__HcControlCurrentED_bits);
__IO_REG32_BIT(HcBulkHeadED,          0xFFF81028,__READ_WRITE,__HcBulkHeadED_bits);
__IO_REG32_BIT(HcBulkCurrentED,       0xFFF8102C,__READ_WRITE,__HcBulkCurrentED_bits);
__IO_REG32_BIT(HcDoneHead,            0xFFF81030,__READ      ,__HcDoneHead_bits);
__IO_REG32_BIT(HcFmInterval,          0xFFF81034,__READ_WRITE,__HcFmInterval_bits);
__IO_REG32_BIT(HcFmRemaining,         0xFFF81038,__READ      ,__HcFmRemaining_bits);
__IO_REG32_BIT(HcFmNumber,            0xFFF8103C,__READ      ,__HcFmNumber_bits);
__IO_REG32_BIT(HcPeriodicStart,       0xFFF81040,__READ_WRITE,__HcPeriodicStart_bits);
__IO_REG32_BIT(HcLSThreshold,         0xFFF81044,__READ_WRITE,__HcLSThreshold_bits);
__IO_REG32_BIT(HcRhDescriptorA,       0xFFF81048,__READ_WRITE,__HcRhDescriptorA_bits);
__IO_REG32_BIT(HcRhDescriptorB,       0xFFF8104C,__READ_WRITE,__HcRhDescriptorB_bits);
__IO_REG32_BIT(HcRhStatus,            0xFFF81050,__READ_WRITE,__HcRhStatus_bits);
__IO_REG32_BIT(HcRhPortStatus1,       0xFFF81054,__READ_WRITE,__HcRhPortStatus_bits);
__IO_REG32_BIT(LinkModeSetting,       0xFFF82000,__READ_WRITE,__linkmodesetting_bits);
__IO_REG32_BIT(PHYModeSetting1,       0xFFF82004,__READ_WRITE,__phymodesetting1_bits);
__IO_REG32_BIT(PHYModeSetting2,       0xFFF82008,__READ_WRITE,__phymodesetting2_bits);

/***************************************************************************
 **
 ** USB function controller
 **
 ***************************************************************************/
__IO_REG32_BIT(UFCpAC,             0xFFF70000 , __READ_WRITE , __ufcpac_bits);
__IO_REG32_BIT(UFDvC,              0xFFF70004 , __READ_WRITE , __ufdvc_bits);
__IO_REG32_BIT(UFDvS,              0xFFF70008 , __READ       , __ufdvs_bits);
__IO_REG32_BIT(UFEpIC,             0xFFF7000C , __READ_WRITE , __ufepic_bits);
__IO_REG32_BIT(UFEpIS,             0xFFF70010 , __READ       , __ufepis_bits);
__IO_REG32_BIT(UFEpDC,             0xFFF70014 , __READ_WRITE , __ufepdc_bits);
__IO_REG32_BIT(UFEpDS,             0xFFF70018 , __READ       , __ufepds_bits);
__IO_REG32_BIT(UFTSTAMP,           0xFFF7001C , __READ       , __uftstamp_bits);
__IO_REG32_BIT(UFEpTCSel,          0xFFF70020 , __READ       , __ufeptcsel_bits);
__IO_REG32(    UFEpTC1,            0xFFF70024 , __READ_WRITE );
__IO_REG32(    UFEpTC2,            0xFFF70028 , __READ_WRITE );
__IO_REG32_BIT(UFEpRS0,            0xFFF70070 , __READ_WRITE , __ufeprs0_bits);
__IO_REG32_BIT(UFEpRS1,            0xFFF70078 , __READ_WRITE , __ufeprs1_bits);
__IO_REG32_BIT(UFEpRS2,            0xFFF70080 , __READ_WRITE , __ufeprs2_bits);
__IO_REG32_BIT(UFEpRS3,            0xFFF70088 , __READ_WRITE , __ufeprs3_bits);
__IO_REG32_BIT(UFCusCnt,           0xFFF700F0 , __READ_WRITE , __ufcuscnt_bits);
__IO_REG32_BIT(UFCALB,             0xFFF700F4 , __READ_WRITE , __ufcalb_bits);
__IO_REG32_BIT(UFEpLpBk,           0xFFF700F8 , __READ_WRITE , __ufeplpbk_bits);
__IO_REG32_BIT(UFIntfAltNum,       0xFFF700FC , __READ_WRITE , __ufintfaltnum_bits);
__IO_REG32_BIT(UFEpC0,             0xFFF70100 , __READ_WRITE , __ufepc0_bits);
__IO_REG32_BIT(UFEpS0,             0xFFF70104 , __READ       , __ufeps0_bits);
__IO_REG32_BIT(UFEpC1,             0xFFF70108 , __READ_WRITE , __ufepc1_bits);
__IO_REG32_BIT(UFEpS1,             0xFFF7010C , __READ       , __ufeps1_bits);
__IO_REG32_BIT(UFEpC2,             0xFFF70110 , __READ_WRITE , __ufepc2_bits);
__IO_REG32_BIT(UFEpS2,             0xFFF70114 , __READ       , __ufeps2_bits);
__IO_REG32_BIT(UFEpC3,             0xFFF70118 , __READ_WRITE , __ufepc3_bits);
__IO_REG32_BIT(UFEpS3,             0xFFF7011C , __READ       , __ufeps3_bits);
__IO_REG32(    UFEpIB0,            0xFFF70180 , __WRITE      );
__IO_REG32(    UFEpIB1,            0xFFF70184 , __WRITE      );
__IO_REG32(    UFEpIB2,            0xFFF70188 , __WRITE      );
__IO_REG32(    UFEpIB3,            0xFFF7018C , __WRITE      );
__IO_REG32(    UFEpOB0,            0xFFF701C0 , __READ       );
__IO_REG32(    UFEpOB1,            0xFFF701C4 , __READ       );
__IO_REG32(    UFEpOB2,            0xFFF701C8 , __READ       );
__IO_REG32_BIT(UFConfig0,          0xFFF70200 , __READ       , __ufconfig0_bits);
__IO_REG32_BIT(UFConfig1,          0xFFF70204 , __READ       , __ufconfigx_bits);
__IO_REG32_BIT(UFConfig2,          0xFFF70208 , __READ       , __ufconfigx_bits);
__IO_REG32_BIT(UFConfig3,          0xFFF7020C , __READ       , __ufconfigx_bits);
__IO_REG32_BIT(UFConfig4,          0xFFF70210 , __READ       , __ufconfigx_bits);
__IO_REG32_BIT(UFEpDC1,            0xFFF70404 , __READ_WRITE , __ufepdc1_bits);
__IO_REG32_BIT(UFEpDC2,            0xFFF70408 , __READ_WRITE , __ufepdc2_bits);
__IO_REG32(    UFEpDA1,            0xFFF70414 , __READ_WRITE );
__IO_REG32(    UFEpDA2,            0xFFF70418 , __READ_WRITE );
__IO_REG32(    UFEpDS1,            0xFFF70424 , __READ_WRITE );
__IO_REG32(    UFEpDS2,            0xFFF70428 , __READ_WRITE );

/***************************************************************************
 **
 ** IDE66
 **
 ***************************************************************************/
__IO_REG32_BIT(CS0DAT,             0xFFF20000 , __READ_WRITE , __cs0dat_bits);
/*CS0FT and CS0ER share same address*/
__IO_REG32_BIT(CS0ER,              0xFFF20004 , __READ_WRITE , __cs0er_cs0ft_bits);
#define CS0FT     CS0ER
#define CS0FT_bit CS0ER_bit
__IO_REG32_BIT(CS0SC,              0xFFF20008 , __READ_WRITE , __cs0sc_bits);
__IO_REG32_BIT(CS0SN,              0xFFF2000C , __READ_WRITE , __cs0sn_bits);
__IO_REG32_BIT(CS0CL,              0xFFF20010 , __READ_WRITE , __cs0cl_bits);
__IO_REG32_BIT(CS0CH,              0xFFF20014 , __READ_WRITE , __cs0ch_bits);
__IO_REG32_BIT(CS0DH,              0xFFF20018 , __READ_WRITE , __cs0dh_bits);
/*CS0ST and CS0CMD share same address*/
__IO_REG32_BIT(CS0ST,              0xFFF2001C , __READ_WRITE , __cs0st_cs0cmd_bits);
#define CS0CMD      CS0ST
#define CS0CMD_bit  CS0ST_bit
/*CS1AS and CS1DC share same address*/
__IO_REG32_BIT(CS1AS,              0xFFF20038 , __READ_WRITE , __cs1as_cs1dc_bits);
#define CS1DC     CS1AS
#define CS1DC_bit CS1AS_bit
__IO_REG32_BIT(IDEDATA,            0xFFF20040 , __READ_WRITE , __idedata_bits);
__IO_REG32_BIT(IDEPTCR,            0xFFF20048 , __READ_WRITE , __ideptcr_bits);
__IO_REG32_BIT(IDEPASR,            0xFFF2004C , __READ_WRITE , __idepasr_bits);
__IO_REG32_BIT(IDEICMR,            0xFFF20050 , __READ_WRITE , __ideicmr_bits);
__IO_REG32_BIT(IDEISTR,            0xFFF20054 , __READ       , __ideistr_bits);
__IO_REG32_BIT(IDEINER,            0xFFF20058 , __READ_WRITE , __ideiner_bits);
__IO_REG32_BIT(IDEINSR,            0xFFF2005C , __READ       , __ideinsr_bits);
__IO_REG32_BIT(IDEFCMR,            0xFFF20060 , __READ_WRITE , __idefcmr_bits);
__IO_REG32_BIT(IDEFSTR,            0xFFF20064 , __READ       , __idefstr_bits);
__IO_REG32_BIT(IDETFCR,            0xFFF20068 , __READ       , __idetfcr_bits);
__IO_REG32_BIT(IDERFCR,            0xFFF20070 , __READ       , __iderfcr_bits);
__IO_REG32_BIT(IDEUTCR,            0xFFF200C8 , __READ_WRITE , __ideutcr_bits);
__IO_REG32_BIT(IDEUCMR,            0xFFF200D0 , __READ_WRITE , __ideucmr_bits);
__IO_REG32_BIT(IDEUSTR,            0xFFF200D4 , __READ       , __ideustr_bits);
__IO_REG32_BIT(IDERRCC,            0xFFF20150 , __READ_WRITE , __iderrcc_bits);
__IO_REG32_BIT(IDEUTC1,            0xFFF20154 , __READ_WRITE , __ideutc1_bits);
__IO_REG32_BIT(IDEUTC2,            0xFFF20158 , __READ_WRITE , __ideutc2_bits);
__IO_REG32_BIT(IDEUTC3,            0xFFF2015C , __READ_WRITE , __ideutc3_bits);
__IO_REG32_BIT(IDESTATUS,          0xFFF20200 , __READ       , __idestatus_bits);
__IO_REG32_BIT(IDEINT,             0xFFF20204 , __READ_WRITE , __ideint_bits);
__IO_REG32_BIT(IDEINTMSK,          0xFFF20208 , __READ_WRITE , __ideintmsk_bits);
__IO_REG32_BIT(IDEPIOCTL,          0xFFF2020C , __READ_WRITE , __idepioctl_bits);
__IO_REG32_BIT(IDEDMACTL,          0xFFF20210 , __READ_WRITE , __idedmactl_bits);
__IO_REG32_BIT(IDEDMATC,           0xFFF20214 , __READ_WRITE , __idedmatc_bits);
__IO_REG32(    IDEDMASAD,          0xFFF20218 , __READ_WRITE );
__IO_REG32(    IDEDMADAD,          0xFFF2021C , __READ_WRITE );

/***************************************************************************
 **
 ** CCNT
 **
 ***************************************************************************/
__IO_REG32_BIT(CCID,              0xFFF42000 , __READ       , __ccid_bits);
__IO_REG32_BIT(CSRST,             0xFFF42004 , __READ_WRITE , __csrst_bits);
__IO_REG32_BIT(CIST,              0xFFF42010 , __READ_WRITE , __cist_bits);
__IO_REG32_BIT(CISTM,             0xFFF42014 , __READ_WRITE , __cistm_bits);
__IO_REG32_BIT(CGPIO_IST,         0xFFF42018 , __READ_WRITE , __cgpio_ist_bits);
__IO_REG32_BIT(CGPIO_ISTM,        0xFFF4201C , __READ_WRITE , __cgpio_istm_bits);
__IO_REG32_BIT(CGPIO_IP,          0xFFF42020 , __READ_WRITE , __cgpio_ip_bits);
__IO_REG32_BIT(CGPIO_IM,          0xFFF42024 , __READ_WRITE , __cgpio_im_bits);
__IO_REG32_BIT(CAXI_BW,           0xFFF42028 , __READ_WRITE , __caxi_bw_bits);
__IO_REG32_BIT(CAXI_PS,           0xFFF4202C , __READ_WRITE , __caxi_ps_bits);
__IO_REG32_BIT(CMUX_MD,           0xFFF42030 , __READ_WRITE , __cmux_md_bits);
__IO_REG32_BIT(CEX_PIN_ST,        0xFFF42034 , __READ_WRITE , __cex_pin_st_bits);
__IO_REG32_BIT(CMLB,              0xFFF42038 , __READ_WRITE , __cmlb_bits);
__IO_REG32_BIT(CUSB,              0xFFF42040 , __READ_WRITE , __cusb_bits);
__IO_REG32_BIT(CBSC,              0xFFF420E8 , __READ_WRITE , __cbsc_bits);
__IO_REG32_BIT(CDCRC,             0xFFF420EC , __READ_WRITE , __cdcrc_bits);
__IO_REG32_BIT(CMSR0,             0xFFF420F0 , __READ_WRITE , __cmsr0_bits);
__IO_REG32_BIT(CMSR1,             0xFFF420F4 , __READ_WRITE , __cmsr1_bits);

/***************************************************************************
 **
 ** EXIRC
 **
 ***************************************************************************/
__IO_REG32_BIT(EIENB,             0xFFFE4000 , __READ_WRITE , __eienb_bits);
__IO_REG32_BIT(EIREQ,             0xFFFE4004 , __READ_WRITE , __eireq_bits);
__IO_REG32_BIT(EILVL,             0xFFFE4008 , __READ_WRITE , __eilvl_bits);

/***************************************************************************
 **
 ** GDC Common
 **
 ***************************************************************************/
__IO_REG32_BIT(VCCC,              0xF1FD7FF8 , __READ_WRITE , __vccc_bits);
__IO_REG32(    VCSR,              0xF1FD7FFC , __READ_WRITE );

/***************************************************************************
 **
 ** GDC Display 0
 **
 ***************************************************************************/
__IO_REG32_BIT(DC0DCM0,           0xF1FD0000 , __READ_WRITE , __dcxdcm0_bits);
__IO_REG32_BIT(DC0DCM1,           0xF1FD0100 , __READ_WRITE , __dcxdcm1_bits);
__IO_REG32_BIT(DC0DCM2,           0xF1FD0104 , __READ_WRITE , __dcxdcm2_bits);
__IO_REG32_BIT(DC0DCM3,           0xF1FD0108 , __READ_WRITE , __dcxdcm3_bits);
__IO_REG16_BIT(DC0HTP,            0xF1FD0006 , __READ_WRITE , __dcxhtp_bits);
__IO_REG16_BIT(DC0HDP,            0xF1FD0008 , __READ_WRITE , __dcxhdp_bits);
__IO_REG16_BIT(DC0HDB,            0xF1FD000A , __READ_WRITE , __dcxhdb_bits);
__IO_REG16_BIT(DC0HSP,            0xF1FD000C , __READ_WRITE , __dcxhsp_bits);
__IO_REG8_BIT( DC0HSW,            0xF1FD000E , __READ_WRITE , __dcxhsw_bits);
__IO_REG8_BIT( DC0VSW,            0xF1FD000F , __READ_WRITE , __dcxvsw_bits);
__IO_REG16_BIT(DC0VTR,            0xF1FD0012 , __READ_WRITE , __dcxvtr_bits);
__IO_REG16_BIT(DC0VSP,            0xF1FD0014 , __READ_WRITE , __dcxvsp_bits);
__IO_REG16_BIT(DC0VDP,            0xF1FD0016 , __READ_WRITE , __dcxvdp_bits);
__IO_REG32_BIT(DC0L0M,            0xF1FD0020 , __READ_WRITE , __dcxl0m_bits);
__IO_REG32_BIT(DC0L0EM,           0xF1FD0110 , __READ_WRITE , __dcxl0em_bits);
__IO_REG32(    DC0L0OA,           0xF1FD0024 , __READ_WRITE );
__IO_REG32(    DC0L0DA,           0xF1FD0028 , __READ_WRITE );
__IO_REG16_BIT(DC0L0DX,           0xF1FD002C , __READ_WRITE , __dcxl0dx_bits);
__IO_REG16_BIT(DC0L0DY,           0xF1FD002E , __READ_WRITE , __dcxl0dy_bits);
__IO_REG16_BIT(DC0L0WX,           0xF1FD0114 , __READ_WRITE , __dcxl0wx_bits);
__IO_REG16_BIT(DC0L0WY,           0xF1FD0116 , __READ_WRITE , __dcxl0wy_bits);
__IO_REG16_BIT(DC0L0WW,           0xF1FD0118 , __READ_WRITE , __dcxl0ww_bits);
__IO_REG16_BIT(DC0L0WH,           0xF1FD011A , __READ_WRITE , __dcxl0wh_bits);
__IO_REG32_BIT(DC0L1M,            0xF1FD0030 , __READ_WRITE , __dcxl1m_bits);
__IO_REG32_BIT(DC0L1EM,           0xF1FD0120 , __READ_WRITE , __dcxl1em_bits);
__IO_REG32(    DC0L1DA,           0xF1FD0034 , __READ_WRITE );
__IO_REG16_BIT(DC0L1WX,           0xF1FD0124 , __READ_WRITE , __dcxl1wx_bits);
__IO_REG16_BIT(DC0L1WY,           0xF1FD0126 , __READ_WRITE , __dcxl1wy_bits);
__IO_REG16_BIT(DC0L1WW,           0xF1FD0128 , __READ_WRITE , __dcxl1ww_bits);
__IO_REG16_BIT(DC0L1WH,           0xF1FD012A , __READ_WRITE , __dcxl1wh_bits);
__IO_REG32_BIT(DC0L2M,            0xF1FD0040 , __READ_WRITE , __dcxl2m_bits);
__IO_REG32_BIT(DC0L2EM,           0xF1FD0130 , __READ_WRITE , __dcxl2em_bits);
__IO_REG32(    DC0L2OA0,          0xF1FD0044 , __READ_WRITE );
__IO_REG32(    DC0L2DA0,          0xF1FD0048 , __READ_WRITE );
__IO_REG32(    DC0L2OA1,          0xF1FD004C , __READ_WRITE );
__IO_REG32(    DC0L2DA1,          0xF1FD0050 , __READ_WRITE );
__IO_REG16_BIT(DC0L2DX,           0xF1FD0054 , __READ_WRITE , __dcxl2dx_bits);
__IO_REG16_BIT(DC0L2DY,           0xF1FD0056 , __READ_WRITE , __dcxl2dy_bits);
__IO_REG16_BIT(DC0L2WX,           0xF1FD0134 , __READ_WRITE , __dcxl2wx_bits);
__IO_REG16_BIT(DC0L2WY,           0xF1FD0136 , __READ_WRITE , __dcxl2wy_bits);
__IO_REG16_BIT(DC0L2WW,           0xF1FD0138 , __READ_WRITE , __dcxl2ww_bits);
__IO_REG16_BIT(DC0L2WH,           0xF1FD013A , __READ_WRITE , __dcxl2wh_bits);
__IO_REG32_BIT(DC0L3M,            0xF1FD0058 , __READ_WRITE , __dcxl3m_bits);
__IO_REG32_BIT(DC0L3EM,           0xF1FD0140 , __READ_WRITE , __dcxl3em_bits);
__IO_REG32(    DC0L3OA0,          0xF1FD005C , __READ_WRITE );
__IO_REG32(    DC0L3DA0,          0xF1FD0060 , __READ_WRITE );
__IO_REG32(    DC0L3OA1,          0xF1FD0064 , __READ_WRITE );
__IO_REG32(    DC0L3DA1,          0xF1FD0068 , __READ_WRITE );
__IO_REG16_BIT(DC0L3DX,           0xF1FD006C , __READ_WRITE , __dcxl3dx_bits);
__IO_REG16_BIT(DC0L3DY,           0xF1FD006E , __READ_WRITE , __dcxl3dy_bits);
__IO_REG16_BIT(DC0L3WX,           0xF1FD0144 , __READ_WRITE , __dcxl3wx_bits);
__IO_REG16_BIT(DC0L3WY,           0xF1FD0146 , __READ_WRITE , __dcxl3wy_bits);
__IO_REG16_BIT(DC0L3WW,           0xF1FD0148 , __READ_WRITE , __dcxl3ww_bits);
__IO_REG16_BIT(DC0L3WH,           0xF1FD014A , __READ_WRITE , __dcxl3wh_bits);
__IO_REG32_BIT(DC0L4M,            0xF1FD0070 , __READ_WRITE , __dcxl4m_bits);
__IO_REG32_BIT(DC0L4EM,           0xF1FD0150 , __READ_WRITE , __dcxl4em_bits);
__IO_REG32(    DC0L4OA0,          0xF1FD0074 , __READ_WRITE );
__IO_REG32(    DC0L4DA0,          0xF1FD0078 , __READ_WRITE );
__IO_REG32(    DC0L4OA1,          0xF1FD007C , __READ_WRITE );
__IO_REG32(    DC0L4DA1,          0xF1FD0080 , __READ_WRITE );
__IO_REG16_BIT(DC0L4DX,           0xF1FD0084 , __READ_WRITE , __dcxl4dx_bits);
__IO_REG16_BIT(DC0L4DY,           0xF1FD0086 , __READ_WRITE , __dcxl4dy_bits);
__IO_REG16_BIT(DC0L4WX,           0xF1FD0154 , __READ_WRITE , __dcxl4wx_bits);
__IO_REG16_BIT(DC0L4WY,           0xF1FD0156 , __READ_WRITE , __dcxl4wy_bits);
__IO_REG16_BIT(DC0L4WW,           0xF1FD0158 , __READ_WRITE , __dcxl4ww_bits);
__IO_REG16_BIT(DC0L4WH,           0xF1FD015A , __READ_WRITE , __dcxl4wh_bits);
__IO_REG32_BIT(DC0L5M,            0xF1FD0088 , __READ_WRITE , __dcxl5m_bits);
__IO_REG32_BIT(DC0L5EM,           0xF1FD0160 , __READ_WRITE , __dcxl5em_bits);
__IO_REG32(    DC0L5OA0,          0xF1FD008C , __READ_WRITE );
__IO_REG32(    DC0L5DA0,          0xF1FD0090 , __READ_WRITE );
__IO_REG32(    DC0L5OA1,          0xF1FD0094 , __READ_WRITE );
__IO_REG32(    DC0L5DA1,          0xF1FD0098 , __READ_WRITE );
__IO_REG16_BIT(DC0L5DX,           0xF1FD009C , __READ_WRITE , __dcxl5dx_bits);
__IO_REG16_BIT(DC0L5DY,           0xF1FD009E , __READ_WRITE , __dcxl5dy_bits);
__IO_REG16_BIT(DC0L5WX,           0xF1FD0164 , __READ_WRITE , __dcxl5wx_bits);
__IO_REG16_BIT(DC0L5WY,           0xF1FD0166 , __READ_WRITE , __dcxl5wy_bits);
__IO_REG16_BIT(DC0L5WW,           0xF1FD0168 , __READ_WRITE , __dcxl5ww_bits);
__IO_REG16_BIT(DC0L5WH,           0xF1FD016A , __READ_WRITE , __dcxl5wh_bits);
__IO_REG16_BIT(DC0CUTC,           0xF1FD00A0 , __READ_WRITE , __dcxcutc_bits);
__IO_REG8_BIT( DC0CPM,            0xF1FD00A2 , __READ_WRITE , __dcxcpm_bits);
__IO_REG32(    DC0LCUOA0,         0xF1FD00A4 , __READ_WRITE );
__IO_REG16_BIT(DC0CUX0,           0xF1FD00A8 , __READ_WRITE , __dcxcux0_bits);
__IO_REG16_BIT(DC0CUY0,           0xF1FD00AA , __READ_WRITE , __dcxcuy0_bits);
__IO_REG32(    DC0LCUOA1,         0xF1FD00AC , __READ_WRITE );
__IO_REG16_BIT(DC0CUX1,           0xF1FD00B0 , __READ_WRITE , __dcxcux1_bits);
__IO_REG16_BIT(DC0CUY1,           0xF1FD00B2 , __READ_WRITE , __dcxcuy1_bits);
__IO_REG32_BIT(DC0MDC,            0xF1FD0170 , __READ_WRITE , __dcxmdc_bits);
__IO_REG32_BIT(DC0DLS,            0xF1FD0180 , __READ_WRITE , __dcxdls_bits);
__IO_REG32_BIT(DC0DBGC,           0xF1FD0184 , __READ_WRITE , __dcxdbgc_bits);
__IO_REG32_BIT(DC0L0BLD,          0xF1FD00B4 , __READ_WRITE , __dcxl0bld_bits);
__IO_REG32_BIT(DC0L1BLD,          0xF1FD0188 , __READ_WRITE , __dcxl1bld_bits);
__IO_REG32_BIT(DC0L2BLD,          0xF1FD018C , __READ_WRITE , __dcxl2bld_bits);
__IO_REG32_BIT(DC0L3BLD,          0xF1FD0190 , __READ_WRITE , __dcxl3bld_bits);
__IO_REG32_BIT(DC0L4BLD,          0xF1FD0194 , __READ_WRITE , __dcxl4bld_bits);
__IO_REG32_BIT(DC0L5BLD,          0xF1FD0198 , __READ_WRITE , __dcxl5bld_bits);
__IO_REG16_BIT(DC0L0TC,           0xF1FD00BC , __READ_WRITE , __dcxl0tc_bits);
__IO_REG16_BIT(DC0L2TC,           0xF1FD00C2 , __READ_WRITE , __dcxl2tc_bits);
__IO_REG16_BIT(DC0L3TC,           0xF1FD00C0 , __READ_WRITE , __dcxl3tc_bits);
__IO_REG32_BIT(DC0L0ETC,          0xF1FD01A0 , __READ_WRITE , __dcxl0etc_bits);
__IO_REG32_BIT(DC0L1ETC,          0xF1FD01A4 , __READ_WRITE , __dcxl1etc_bits);
__IO_REG32_BIT(DC0L2ETC,          0xF1FD01A8 , __READ_WRITE , __dcxl2etc_bits);
__IO_REG32_BIT(DC0L3ETC,          0xF1FD01AC , __READ_WRITE , __dcxl3etc_bits);
__IO_REG32_BIT(DC0L4ETC,          0xF1FD01B0 , __READ_WRITE , __dcxl4etc_bits);
__IO_REG32_BIT(DC0L5ETC,          0xF1FD01B4 , __READ_WRITE , __dcxl5etc_bits);
__IO_REG32_BIT(DC0L1YCR0,         0xF1FD01E0 , __READ_WRITE , __dcxl1ycr0_bits);
__IO_REG32_BIT(DC0L1YCR1,         0xF1FD01E4 , __READ_WRITE , __dcxl1ycr1_bits);
__IO_REG32_BIT(DC0L1YCG0,         0xF1FD01E8 , __READ_WRITE , __dcxl1ycg0_bits);
__IO_REG32_BIT(DC0L1YCG1,         0xF1FD01EC , __READ_WRITE , __dcxl1ycg1_bits);
__IO_REG32_BIT(DC0L1YCB0,         0xF1FD01F0 , __READ_WRITE , __dcxl1ycb0_bits);
__IO_REG32_BIT(DC0L1YCB1,         0xF1FD01F4 , __READ_WRITE , __dcxl1ycb1_bits);

/***************************************************************************
 **
 ** GDC Display 1
 **
 ***************************************************************************/
__IO_REG32_BIT(DC1DCM0,           0xF1FD2000 , __READ_WRITE , __dcxdcm0_bits);
__IO_REG32_BIT(DC1DCM1,           0xF1FD2100 , __READ_WRITE , __dcxdcm1_bits);
__IO_REG32_BIT(DC1DCM2,           0xF1FD2104 , __READ_WRITE , __dcxdcm2_bits);
__IO_REG32_BIT(DC1DCM3,           0xF1FD2108 , __READ_WRITE , __dcxdcm3_bits);
__IO_REG16_BIT(DC1HTP,            0xF1FD2006 , __READ_WRITE , __dcxhtp_bits);
__IO_REG16_BIT(DC1HDP,            0xF1FD2008 , __READ_WRITE , __dcxhdp_bits);
__IO_REG16_BIT(DC1HDB,            0xF1FD200A , __READ_WRITE , __dcxhdb_bits);
__IO_REG16_BIT(DC1HSP,            0xF1FD200C , __READ_WRITE , __dcxhsp_bits);
__IO_REG8_BIT( DC1HSW,            0xF1FD200E , __READ_WRITE , __dcxhsw_bits);
__IO_REG8_BIT( DC1VSW,            0xF1FD200F , __READ_WRITE , __dcxvsw_bits);
__IO_REG16_BIT(DC1VTR,            0xF1FD2012 , __READ_WRITE , __dcxvtr_bits);
__IO_REG16_BIT(DC1VSP,            0xF1FD2014 , __READ_WRITE , __dcxvsp_bits);
__IO_REG16_BIT(DC1VDP,            0xF1FD2016 , __READ_WRITE , __dcxvdp_bits);
__IO_REG32_BIT(DC1L0M,            0xF1FD2020 , __READ_WRITE , __dcxl0m_bits);
__IO_REG32_BIT(DC1L0EM,           0xF1FD2110 , __READ_WRITE , __dcxl0em_bits);
__IO_REG32(    DC1L0OA,           0xF1FD2024 , __READ_WRITE );
__IO_REG32(    DC1L0DA,           0xF1FD2028 , __READ_WRITE );
__IO_REG16_BIT(DC1L0DX,           0xF1FD202C , __READ_WRITE , __dcxl0dx_bits);
__IO_REG16_BIT(DC1L0DY,           0xF1FD202E , __READ_WRITE , __dcxl0dy_bits);
__IO_REG16_BIT(DC1L0WX,           0xF1FD2114 , __READ_WRITE , __dcxl0wx_bits);
__IO_REG16_BIT(DC1L0WY,           0xF1FD2116 , __READ_WRITE , __dcxl0wy_bits);
__IO_REG16_BIT(DC1L0WW,           0xF1FD2118 , __READ_WRITE , __dcxl0ww_bits);
__IO_REG16_BIT(DC1L0WH,           0xF1FD211A , __READ_WRITE , __dcxl0wh_bits);
__IO_REG32_BIT(DC1L1M,            0xF1FD2030 , __READ_WRITE , __dcxl1m_bits);
__IO_REG32_BIT(DC1L1EM,           0xF1FD2120 , __READ_WRITE , __dcxl1em_bits);
__IO_REG32(    DC1L1DA,           0xF1FD2034 , __READ_WRITE );
__IO_REG16_BIT(DC1L1WX,           0xF1FD2124 , __READ_WRITE , __dcxl1wx_bits);
__IO_REG16_BIT(DC1L1WY,           0xF1FD2126 , __READ_WRITE , __dcxl1wy_bits);
__IO_REG16_BIT(DC1L1WW,           0xF1FD2128 , __READ_WRITE , __dcxl1ww_bits);
__IO_REG16_BIT(DC1L1WH,           0xF1FD212A , __READ_WRITE , __dcxl1wh_bits);
__IO_REG32_BIT(DC1L2M,            0xF1FD2040 , __READ_WRITE , __dcxl2m_bits);
__IO_REG32_BIT(DC1L2EM,           0xF1FD2130 , __READ_WRITE , __dcxl2em_bits);
__IO_REG32(    DC1L2OA0,          0xF1FD2044 , __READ_WRITE );
__IO_REG32(    DC1L2DA0,          0xF1FD2048 , __READ_WRITE );
__IO_REG32(    DC1L2OA1,          0xF1FD204C , __READ_WRITE );
__IO_REG32(    DC1L2DA1,          0xF1FD2050 , __READ_WRITE );
__IO_REG16_BIT(DC1L2DX,           0xF1FD2054 , __READ_WRITE , __dcxl2dx_bits);
__IO_REG16_BIT(DC1L2DY,           0xF1FD2056 , __READ_WRITE , __dcxl2dy_bits);
__IO_REG16_BIT(DC1L2WX,           0xF1FD2134 , __READ_WRITE , __dcxl2wx_bits);
__IO_REG16_BIT(DC1L2WY,           0xF1FD2136 , __READ_WRITE , __dcxl2wy_bits);
__IO_REG16_BIT(DC1L2WW,           0xF1FD2138 , __READ_WRITE , __dcxl2ww_bits);
__IO_REG16_BIT(DC1L2WH,           0xF1FD213A , __READ_WRITE , __dcxl2wh_bits);
__IO_REG32_BIT(DC1L3M,            0xF1FD2058 , __READ_WRITE , __dcxl3m_bits);
__IO_REG32_BIT(DC1L3EM,           0xF1FD2140 , __READ_WRITE , __dcxl3em_bits);
__IO_REG32(    DC1L3OA0,          0xF1FD205C , __READ_WRITE );
__IO_REG32(    DC1L3DA0,          0xF1FD2060 , __READ_WRITE );
__IO_REG32(    DC1L3OA1,          0xF1FD2064 , __READ_WRITE );
__IO_REG32(    DC1L3DA1,          0xF1FD2068 , __READ_WRITE );
__IO_REG16_BIT(DC1L3DX,           0xF1FD206C , __READ_WRITE , __dcxl3dx_bits);
__IO_REG16_BIT(DC1L3DY,           0xF1FD206E , __READ_WRITE , __dcxl3dy_bits);
__IO_REG16_BIT(DC1L3WX,           0xF1FD2144 , __READ_WRITE , __dcxl3wx_bits);
__IO_REG16_BIT(DC1L3WY,           0xF1FD2146 , __READ_WRITE , __dcxl3wy_bits);
__IO_REG16_BIT(DC1L3WW,           0xF1FD2148 , __READ_WRITE , __dcxl3ww_bits);
__IO_REG16_BIT(DC1L3WH,           0xF1FD214A , __READ_WRITE , __dcxl3wh_bits);
__IO_REG32_BIT(DC1L4M,            0xF1FD2070 , __READ_WRITE , __dcxl4m_bits);
__IO_REG32_BIT(DC1L4EM,           0xF1FD2150 , __READ_WRITE , __dcxl4em_bits);
__IO_REG32(    DC1L4OA0,          0xF1FD2074 , __READ_WRITE );
__IO_REG32(    DC1L4DA0,          0xF1FD2078 , __READ_WRITE );
__IO_REG32(    DC1L4OA1,          0xF1FD207C , __READ_WRITE );
__IO_REG32(    DC1L4DA1,          0xF1FD2080 , __READ_WRITE );
__IO_REG16_BIT(DC1L4DX,           0xF1FD2084 , __READ_WRITE , __dcxl4dx_bits);
__IO_REG16_BIT(DC1L4DY,           0xF1FD2086 , __READ_WRITE , __dcxl4dy_bits);
__IO_REG16_BIT(DC1L4WX,           0xF1FD2154 , __READ_WRITE , __dcxl4wx_bits);
__IO_REG16_BIT(DC1L4WY,           0xF1FD2156 , __READ_WRITE , __dcxl4wy_bits);
__IO_REG16_BIT(DC1L4WW,           0xF1FD2158 , __READ_WRITE , __dcxl4ww_bits);
__IO_REG16_BIT(DC1L4WH,           0xF1FD215A , __READ_WRITE , __dcxl4wh_bits);
__IO_REG32_BIT(DC1L5M,            0xF1FD2088 , __READ_WRITE , __dcxl5m_bits);
__IO_REG32_BIT(DC1L5EM,           0xF1FD2160 , __READ_WRITE , __dcxl5em_bits);
__IO_REG32(    DC1L5OA0,          0xF1FD208C , __READ_WRITE );
__IO_REG32(    DC1L5DA0,          0xF1FD2090 , __READ_WRITE );
__IO_REG32(    DC1L5OA1,          0xF1FD2094 , __READ_WRITE );
__IO_REG32(    DC1L5DA1,          0xF1FD2098 , __READ_WRITE );
__IO_REG16_BIT(DC1L5DX,           0xF1FD209C , __READ_WRITE , __dcxl5dx_bits);
__IO_REG16_BIT(DC1L5DY,           0xF1FD209E , __READ_WRITE , __dcxl5dy_bits);
__IO_REG16_BIT(DC1L5WX,           0xF1FD2164 , __READ_WRITE , __dcxl5wx_bits);
__IO_REG16_BIT(DC1L5WY,           0xF1FD2166 , __READ_WRITE , __dcxl5wy_bits);
__IO_REG16_BIT(DC1L5WW,           0xF1FD2168 , __READ_WRITE , __dcxl5ww_bits);
__IO_REG16_BIT(DC1L5WH,           0xF1FD216A , __READ_WRITE , __dcxl5wh_bits);
__IO_REG16_BIT(DC1CUTC,           0xF1FD20A0 , __READ_WRITE , __dcxcutc_bits);
__IO_REG8_BIT( DC1CPM,            0xF1FD20A2 , __READ_WRITE , __dcxcpm_bits);
__IO_REG32(    DC1LCUOA0,         0xF1FD20A4 , __READ_WRITE );
__IO_REG16_BIT(DC1CUX0,           0xF1FD20A8 , __READ_WRITE , __dcxcux0_bits);
__IO_REG16_BIT(DC1CUY0,           0xF1FD20AA , __READ_WRITE , __dcxcuy0_bits);
__IO_REG32(    DC1LCUOA1,         0xF1FD20AC , __READ_WRITE );
__IO_REG16_BIT(DC1CUX1,           0xF1FD20B0 , __READ_WRITE , __dcxcux1_bits);
__IO_REG16_BIT(DC1CUY1,           0xF1FD20B2 , __READ_WRITE , __dcxcuy1_bits);
__IO_REG32_BIT(DC1MDC,            0xF1FD2170 , __READ_WRITE , __dcxmdc_bits);
__IO_REG32_BIT(DC1DLS,            0xF1FD2180 , __READ_WRITE , __dcxdls_bits);
__IO_REG32_BIT(DC1DBGC,           0xF1FD2184 , __READ_WRITE , __dcxdbgc_bits);
__IO_REG32_BIT(DC1L0BLD,          0xF1FD20B4 , __READ_WRITE , __dcxl0bld_bits);
__IO_REG32_BIT(DC1L1BLD,          0xF1FD2188 , __READ_WRITE , __dcxl1bld_bits);
__IO_REG32_BIT(DC1L2BLD,          0xF1FD218C , __READ_WRITE , __dcxl2bld_bits);
__IO_REG32_BIT(DC1L3BLD,          0xF1FD2190 , __READ_WRITE , __dcxl3bld_bits);
__IO_REG32_BIT(DC1L4BLD,          0xF1FD2194 , __READ_WRITE , __dcxl4bld_bits);
__IO_REG32_BIT(DC1L5BLD,          0xF1FD2198 , __READ_WRITE , __dcxl5bld_bits);
__IO_REG16_BIT(DC1L0TC,           0xF1FD20BC , __READ_WRITE , __dcxl0tc_bits);
__IO_REG16_BIT(DC1L2TC,           0xF1FD20C2 , __READ_WRITE , __dcxl2tc_bits);
__IO_REG16_BIT(DC1L3TC,           0xF1FD20C0 , __READ_WRITE , __dcxl3tc_bits);
__IO_REG32_BIT(DC1L0ETC,          0xF1FD21A0 , __READ_WRITE , __dcxl0etc_bits);
__IO_REG32_BIT(DC1L1ETC,          0xF1FD21A4 , __READ_WRITE , __dcxl1etc_bits);
__IO_REG32_BIT(DC1L2ETC,          0xF1FD21A8 , __READ_WRITE , __dcxl2etc_bits);
__IO_REG32_BIT(DC1L3ETC,          0xF1FD21AC , __READ_WRITE , __dcxl3etc_bits);
__IO_REG32_BIT(DC1L4ETC,          0xF1FD21B0 , __READ_WRITE , __dcxl4etc_bits);
__IO_REG32_BIT(DC1L5ETC,          0xF1FD21B4 , __READ_WRITE , __dcxl5etc_bits);
__IO_REG32_BIT(DC1L1YCR0,         0xF1FD21E0 , __READ_WRITE , __dcxl1ycr0_bits);
__IO_REG32_BIT(DC1L1YCR1,         0xF1FD21E4 , __READ_WRITE , __dcxl1ycr1_bits);
__IO_REG32_BIT(DC1L1YCG0,         0xF1FD21E8 , __READ_WRITE , __dcxl1ycg0_bits);
__IO_REG32_BIT(DC1L1YCG1,         0xF1FD21EC , __READ_WRITE , __dcxl1ycg1_bits);
__IO_REG32_BIT(DC1L1YCB0,         0xF1FD21F0 , __READ_WRITE , __dcxl1ycb0_bits);
__IO_REG32_BIT(DC1L1YCB1,         0xF1FD21F4 , __READ_WRITE , __dcxl1ycb1_bits);

/***************************************************************************
 **
 ** GDC Video capture 0
 **
 ***************************************************************************/
__IO_REG32_BIT(VC0VCM,            0xF1FD8000 , __READ_WRITE , __vcxvcm_bits);
__IO_REG32_BIT(VC0CSC,            0xF1FD8004 , __READ_WRITE , __vcxcsc_bits);
__IO_REG32_BIT(VC0VCS,            0xF1FD8008 , __READ_WRITE , __vcxvcs_bits);
__IO_REG32_BIT(VC0CBM,            0xF1FD8010 , __READ_WRITE , __vcxcbm_bits);
__IO_REG32(    VC0CBOA,           0xF1FD8014 , __READ_WRITE );
__IO_REG32(    VC0CBLA,           0xF1FD8018 , __READ_WRITE );
__IO_REG16_BIT(VC0CIHSTR,         0xF1FD801C , __READ_WRITE , __vcxcihstr_bits);
__IO_REG16_BIT(VC0CIVSTR,         0xF1FD801E , __READ_WRITE , __vcxcivstr_bits);
__IO_REG16_BIT(VC0CIHEND,         0xF1FD8020 , __READ_WRITE , __vcxcihend_bits);
__IO_REG16_BIT(VC0CIVEND,         0xF1FD8022 , __READ_WRITE , __vcxcivend_bits);
__IO_REG32_BIT(VC0CHP,            0xF1FD8028 , __READ_WRITE , __vcxchp_bits);
__IO_REG32_BIT(VC0CVP,            0xF1FD802C , __READ_WRITE , __vcxcvp_bits);
__IO_REG32_BIT(VC0CLPF,           0xF1FD8040 , __READ_WRITE , __vcxclpf_bits);
__IO_REG32_BIT(VC0CMSS,           0xF1FD8048 , __READ_WRITE , __vcxcmss_bits);
__IO_REG32_BIT(VC0CMDS,           0xF1FD804C , __READ_WRITE , __vcxcmds_bits);
__IO_REG32_BIT(VC0RGBHC,          0xF1FD8080 , __READ_WRITE , __vcxrgbhc_bits);
__IO_REG32_BIT(VC0RGBHEN,         0xF1FD8084 , __READ_WRITE , __vcxrgbhen_bits);
__IO_REG32_BIT(VC0RGBVEN,         0xF1FD8088 , __READ_WRITE , __vcxrgbven_bits);
__IO_REG32_BIT(VC0RGBS,           0xF1FD8090 , __READ_WRITE , __vcxrgbs_bits);
__IO_REG32_BIT(VC0RGBCMY,         0xF1FD80C0 , __READ_WRITE , __vcxrgbcmy_bits);
__IO_REG32_BIT(VC0RGBCMCb,        0xF1FD80C4 , __READ_WRITE , __vcxrgbcmcb_bits);
__IO_REG32_BIT(VC0RGBCMCr,        0xF1FD80C8 , __READ_WRITE , __vcxrgbcmcr_bits);
__IO_REG32_BIT(VC0RGBCMb,         0xF1FD80CC , __READ_WRITE , __vcxrgbcmb_bits);
__IO_REG16_BIT(VC0CVCNT,          0xF1FD8300 , __READ       , __vcxcvcnt_bits);
__IO_REG32_BIT(VC0CDCN,           0xF1FDC000 , __READ_WRITE , __vcxcdcn_bits);
__IO_REG32_BIT(VC0CDCP,           0xF1FDC004 , __READ_WRITE , __vcxcdcp_bits);

/***************************************************************************
 **
 ** GDC Video capture 1
 **
 ***************************************************************************/
__IO_REG32_BIT(VC1VCM,            0xF1FDA000 , __READ_WRITE , __vcxvcm_bits);
__IO_REG32_BIT(VC1CSC,            0xF1FDA004 , __READ_WRITE , __vcxcsc_bits);
__IO_REG32_BIT(VC1VCS,            0xF1FDA008 , __READ_WRITE , __vcxvcs_bits);
__IO_REG32_BIT(VC1CBM,            0xF1FDA010 , __READ_WRITE , __vcxcbm_bits);
__IO_REG32(    VC1CBOA,           0xF1FDA014 , __READ_WRITE );
__IO_REG32(    VC1CBLA,           0xF1FDA018 , __READ_WRITE );
__IO_REG16_BIT(VC1CIHSTR,         0xF1FDA01C , __READ_WRITE , __vcxcihstr_bits);
__IO_REG16_BIT(VC1CIVSTR,         0xF1FDA01E , __READ_WRITE , __vcxcivstr_bits);
__IO_REG16_BIT(VC1CIHEND,         0xF1FDA020 , __READ_WRITE , __vcxcihend_bits);
__IO_REG16_BIT(VC1CIVEND,         0xF1FDA022 , __READ_WRITE , __vcxcivend_bits);
__IO_REG32_BIT(VC1CHP,            0xF1FDA028 , __READ_WRITE , __vcxchp_bits);
__IO_REG32_BIT(VC1CVP,            0xF1FDA02C , __READ_WRITE , __vcxcvp_bits);
__IO_REG32_BIT(VC1CLPF,           0xF1FDA040 , __READ_WRITE , __vcxclpf_bits);
__IO_REG32_BIT(VC1CMSS,           0xF1FDA048 , __READ_WRITE , __vcxcmss_bits);
__IO_REG32_BIT(VC1CMDS,           0xF1FDA04C , __READ_WRITE , __vcxcmds_bits);
__IO_REG32_BIT(VC1RGBHC,          0xF1FDA080 , __READ_WRITE , __vcxrgbhc_bits);
__IO_REG32_BIT(VC1RGBHEN,         0xF1FDA084 , __READ_WRITE , __vcxrgbhen_bits);
__IO_REG32_BIT(VC1RGBVEN,         0xF1FDA088 , __READ_WRITE , __vcxrgbven_bits);
__IO_REG32_BIT(VC1RGBS,           0xF1FDA090 , __READ_WRITE , __vcxrgbs_bits);
__IO_REG32_BIT(VC1RGBCMY,         0xF1FDA0C0 , __READ_WRITE , __vcxrgbcmy_bits);
__IO_REG32_BIT(VC1RGBCMCb,        0xF1FDA0C4 , __READ_WRITE , __vcxrgbcmcb_bits);
__IO_REG32_BIT(VC1RGBCMCr,        0xF1FDA0C8 , __READ_WRITE , __vcxrgbcmcr_bits);
__IO_REG32_BIT(VC1RGBCMb,         0xF1FDA0CC , __READ_WRITE , __vcxrgbcmb_bits);
__IO_REG16_BIT(VC1CVCNT,          0xF1FDA300 , __READ       , __vcxcvcnt_bits);
__IO_REG32_BIT(VC1CDCN,           0xF1FDE000 , __READ_WRITE , __vcxcdcn_bits);
__IO_REG32_BIT(VC1CDCP,           0xF1FDE004 , __READ_WRITE , __vcxcdcp_bits);

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0CR,            0xFFF54000 , __READ_WRITE , __canxcr_bits);
__IO_REG32_BIT(CAN0SR,            0xFFF54004 , __READ_WRITE , __canxsr_bits);
__IO_REG32_BIT(CAN0EC,            0xFFF54008 , __READ       , __canxec_bits);
__IO_REG32_BIT(CAN0BTR,           0xFFF5400C , __READ_WRITE , __canxbtr_bits);
__IO_REG32_BIT(CAN0IR,            0xFFF54010 , __READ       , __canxir_bits);
__IO_REG32_BIT(CAN0TR,            0xFFF54014 , __READ_WRITE , __canxtr_bits);
__IO_REG32_BIT(CAN0BRPER,         0xFFF54018 , __READ_WRITE , __canxbrper_bits);
__IO_REG32_BIT(CAN0IF1CR,         0xFFF54020 , __READ_WRITE , __canxifxcr_bits);
__IO_REG32_BIT(CAN0IF1CM,         0xFFF54024 , __READ_WRITE , __canxifxcm_bits);
__IO_REG32_BIT(CAN0IF1M1,         0xFFF54028 , __READ_WRITE , __canxifxm1_bits);
__IO_REG32_BIT(CAN0IF1M2,         0xFFF5402C , __READ_WRITE , __canxifxm2_bits);
__IO_REG32_BIT(CAN0IF1A1,         0xFFF54030 , __READ_WRITE , __canxifxa1_bits);
__IO_REG32_BIT(CAN0IF1A2,         0xFFF54034 , __READ_WRITE , __canxifxa2_bits);
__IO_REG32_BIT(CAN0IF1MC,         0xFFF54038 , __READ_WRITE , __canxifxmc_bits);
__IO_REG32_BIT(CAN0IF1DA1,        0xFFF5403C , __READ_WRITE , __canxifxda1_bits);
__IO_REG32_BIT(CAN0IF1DA2,        0xFFF54040 , __READ_WRITE , __canxifxda2_bits);
__IO_REG32_BIT(CAN0IF1DB1,        0xFFF54044 , __READ_WRITE , __canxifxdb1_bits);
__IO_REG32_BIT(CAN0IF1DB2,        0xFFF54048 , __READ_WRITE , __canxifxdb2_bits);
__IO_REG32_BIT(CAN0IF2CR,         0xFFF54080 , __READ_WRITE , __canxifxcr_bits);
__IO_REG32_BIT(CAN0IF2CM,         0xFFF54084 , __READ_WRITE , __canxifxcm_bits);
__IO_REG32_BIT(CAN0IF2M1,         0xFFF54088 , __READ_WRITE , __canxifxm1_bits);
__IO_REG32_BIT(CAN0IF2M2,         0xFFF5408C , __READ_WRITE , __canxifxm2_bits);
__IO_REG32_BIT(CAN0IF2A1,         0xFFF54090 , __READ_WRITE , __canxifxa1_bits);
__IO_REG32_BIT(CAN0IF2A2,         0xFFF54094 , __READ_WRITE , __canxifxa2_bits);
__IO_REG32_BIT(CAN0IF2MC,         0xFFF54098 , __READ_WRITE , __canxifxmc_bits);
__IO_REG32_BIT(CAN0IF2DA1,        0xFFF5409C , __READ_WRITE , __canxifxda1_bits);
__IO_REG32_BIT(CAN0IF2DA2,        0xFFF540A0 , __READ_WRITE , __canxifxda2_bits);
__IO_REG32_BIT(CAN0IF2DB1,        0xFFF540A4 , __READ_WRITE , __canxifxdb1_bits);
__IO_REG32_BIT(CAN0IF2DB2,        0xFFF540A8 , __READ_WRITE , __canxifxdb2_bits);
__IO_REG32_BIT(CAN0TR1,           0xFFF54100 , __READ       , __canxtr1_bits);
__IO_REG32_BIT(CAN0TR2,           0xFFF54104 , __READ       , __canxtr2_bits);
__IO_REG32_BIT(CAN0ND1,           0xFFF54120 , __READ       , __canxnd1_bits);
__IO_REG32_BIT(CAN0ND2,           0xFFF54124 , __READ       , __canxnd2_bits);
__IO_REG32_BIT(CAN0IP1,           0xFFF54140 , __READ       , __canxip1_bits);
__IO_REG32_BIT(CAN0IP2,           0xFFF54144 , __READ       , __canxip2_bits);
__IO_REG32_BIT(CAN0MV1,           0xFFF54160 , __READ       , __canxmv1_bits);
__IO_REG32_BIT(CAN0MV2,           0xFFF54164 , __READ       , __canxmv2_bits);

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1CR,            0xFFF55000 , __READ_WRITE , __canxcr_bits);
__IO_REG32_BIT(CAN1SR,            0xFFF55004 , __READ_WRITE , __canxsr_bits);
__IO_REG32_BIT(CAN1EC,            0xFFF55008 , __READ       , __canxec_bits);
__IO_REG32_BIT(CAN1BTR,           0xFFF5500C , __READ_WRITE , __canxbtr_bits);
__IO_REG32_BIT(CAN1IR,            0xFFF55010 , __READ       , __canxir_bits);
__IO_REG32_BIT(CAN1TR,            0xFFF55014 , __READ_WRITE , __canxtr_bits);
__IO_REG32_BIT(CAN1BRPER,         0xFFF55018 , __READ_WRITE , __canxbrper_bits);
__IO_REG32_BIT(CAN1IF1CR,         0xFFF55020 , __READ_WRITE , __canxifxcr_bits);
__IO_REG32_BIT(CAN1IF1CM,         0xFFF55024 , __READ_WRITE , __canxifxcm_bits);
__IO_REG32_BIT(CAN1IF1M1,         0xFFF55028 , __READ_WRITE , __canxifxm1_bits);
__IO_REG32_BIT(CAN1IF1M2,         0xFFF5502C , __READ_WRITE , __canxifxm2_bits);
__IO_REG32_BIT(CAN1IF1A1,         0xFFF55030 , __READ_WRITE , __canxifxa1_bits);
__IO_REG32_BIT(CAN1IF1A2,         0xFFF55034 , __READ_WRITE , __canxifxa2_bits);
__IO_REG32_BIT(CAN1IF1MC,         0xFFF55038 , __READ_WRITE , __canxifxmc_bits);
__IO_REG32_BIT(CAN1IF1DA1,        0xFFF5503C , __READ_WRITE , __canxifxda1_bits);
__IO_REG32_BIT(CAN1IF1DA2,        0xFFF55040 , __READ_WRITE , __canxifxda2_bits);
__IO_REG32_BIT(CAN1IF1DB1,        0xFFF55044 , __READ_WRITE , __canxifxdb1_bits);
__IO_REG32_BIT(CAN1IF1DB2,        0xFFF55048 , __READ_WRITE , __canxifxdb2_bits);
__IO_REG32_BIT(CAN1IF2CR,         0xFFF55080 , __READ_WRITE , __canxifxcr_bits);
__IO_REG32_BIT(CAN1IF2CM,         0xFFF55084 , __READ_WRITE , __canxifxcm_bits);
__IO_REG32_BIT(CAN1IF2M1,         0xFFF55088 , __READ_WRITE , __canxifxm1_bits);
__IO_REG32_BIT(CAN1IF2M2,         0xFFF5508C , __READ_WRITE , __canxifxm2_bits);
__IO_REG32_BIT(CAN1IF2A1,         0xFFF55090 , __READ_WRITE , __canxifxa1_bits);
__IO_REG32_BIT(CAN1IF2A2,         0xFFF55094 , __READ_WRITE , __canxifxa2_bits);
__IO_REG32_BIT(CAN1IF2MC,         0xFFF55098 , __READ_WRITE , __canxifxmc_bits);
__IO_REG32_BIT(CAN1IF2DA1,        0xFFF5509C , __READ_WRITE , __canxifxda1_bits);
__IO_REG32_BIT(CAN1IF2DA2,        0xFFF550A0 , __READ_WRITE , __canxifxda2_bits);
__IO_REG32_BIT(CAN1IF2DB1,        0xFFF550A4 , __READ_WRITE , __canxifxdb1_bits);
__IO_REG32_BIT(CAN1IF2DB2,        0xFFF550A8 , __READ_WRITE , __canxifxdb2_bits);
__IO_REG32_BIT(CAN1TR1,           0xFFF55100 , __READ       , __canxtr1_bits);
__IO_REG32_BIT(CAN1TR2,           0xFFF55104 , __READ       , __canxtr2_bits);
__IO_REG32_BIT(CAN1ND1,           0xFFF55120 , __READ       , __canxnd1_bits);
__IO_REG32_BIT(CAN1ND2,           0xFFF55124 , __READ       , __canxnd2_bits);
__IO_REG32_BIT(CAN1IP1,           0xFFF55140 , __READ       , __canxip1_bits);
__IO_REG32_BIT(CAN1IP2,           0xFFF55144 , __READ       , __canxip2_bits);
__IO_REG32_BIT(CAN1MV1,           0xFFF55160 , __READ       , __canxmv1_bits);
__IO_REG32_BIT(CAN1MV2,           0xFFF55164 , __READ       , __canxmv2_bits);

/***************************************************************************
 **
 ** TIMER
 **
 ***************************************************************************/
__IO_REG32(    TIMER1LOAD,        0xFFFE0000 , __READ_WRITE );
__IO_REG32(    TIMER1VALUE,       0xFFFE0004 , __READ       );
__IO_REG32_BIT(TIMER1CONTROL,     0xFFFE0008 , __READ_WRITE , __timercontrol_bits);
__IO_REG32(    TIMER1INTCLR,      0xFFFE000C , __WRITE      );
__IO_REG32_BIT(TIMER1RIS,         0xFFFE0010 , __READ       , __timerris_bits);
__IO_REG32_BIT(TIMER1MIS,         0xFFFE0014 , __READ       , __timermis_bits);
__IO_REG32(    TIMER1BGLOAD,      0xFFFE0018 , __READ_WRITE );
__IO_REG32(    TIMER2LOAD,        0xFFFE0020 , __READ_WRITE );
__IO_REG32(    TIMER2VALUE,       0xFFFE0024 , __READ       );
__IO_REG32_BIT(TIMER2CONTROL,     0xFFFE0028 , __READ_WRITE , __timercontrol_bits);
__IO_REG32(    TIMER2INTCLR,      0xFFFE002C , __WRITE      );
__IO_REG32_BIT(TIMER2RIS,         0xFFFE0030 , __READ       , __timerris_bits);
__IO_REG32_BIT(TIMER2MIS,         0xFFFE0034 , __READ       , __timermis_bits);
__IO_REG32(    TIMER2BGLOAD,      0xFFFE0038 , __READ_WRITE );
__IO_REG32_BIT(TIMERITCR,         0xFFFE0F00 , __READ_WRITE , __timeritcr_bits);
__IO_REG32_BIT(TIMERITOP,         0xFFFE0F04 , __WRITE      , __timeritop_bits);
__IO_REG32_BIT(TIMERPERIPHID0,    0xFFFE0FE0 , __READ       , __timerperiphid0_bits);
__IO_REG32_BIT(TIMERPERIPHID1,    0xFFFE0FE4 , __READ       , __timerperiphid1_bits);
__IO_REG32_BIT(TIMERPERIPHID2,    0xFFFE0FE8 , __READ       , __timerperiphid2_bits);
__IO_REG32_BIT(TIMERPERIPHID3,    0xFFFE0FEC , __READ       , __timerperiphid3_bits);
__IO_REG32_BIT(TIMERPCELLID0,     0xFFFE0FF0 , __READ       , __timerpcellid0_bits);
__IO_REG32_BIT(TIMERPCELLID1,     0xFFFE0FF4 , __READ       , __timerpcellid1_bits);
__IO_REG32_BIT(TIMERPCELLID2,     0xFFFE0FF8 , __READ       , __timerpcellid2_bits);
__IO_REG32_BIT(TIMERPCELLID3,     0xFFFE0FFC , __READ       , __timerpcellid3_bits);

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
 **  IRQ 0 sources
 **
 ***************************************************************************/
#define INT_IRC1             6
#define INT_GPIO             7
#define INT_ADC_CH0          8
#define INT_ADC_CH1          9
#define INT_EXT0            10
#define INT_EXT1            11
#define INT_EXT2            12
#define INT_EXT3            13
#define INT_TIMER_CH0       14
#define INT_TIMER_CH1       15
#define INT_DMAC_CH0        16
#define INT_DMAC_CH1        17
#define INT_DMAC_CH2        18
#define INT_DMAC_CH3        19
#define INT_DMAC_CH4        20
#define INT_DMAC_CH5        21
#define INT_DMAC_CH6        22
#define INT_DMAC_CH7        23
#define INT_UART_CH0        24
#define INT_UART_CH1        25
#define INT_COMMRX          28
#define INT_COMMTX          29
#define INT_DELAY0          30

/***************************************************************************
 **
 **  IRQ 1 sources
 **
 ***************************************************************************/
#define INT_GDC              0
#define INT_CAN_CH0          2
#define INT_CAN_CH1          3
#define INT_SD_IF            4
#define INT_MBUS2AXI_CAP     5
#define INT_I2S_CH0          6
#define INT_I2S_CH1          7
#define INT_I2S_CH2          8
#define INT_SPI              9
#define INT_IDE66           10
#define INT_I2C_CH0         11
#define INT_I2C_CH1         12
#define INT_PWM_CH0         13
#define INT_PWM_CH1         14
#define INT_UART_CH2        15
#define INT_UART_CH3        16
#define INT_UART_CH4        17
#define INT_UART_CH5        18
#define INT_USB2_0_HOSTPHY  19
#define INT_USB2_0_EHCI     20
#define INT_USB1_1_OHCI     21
#define INT_USB2_0_FUN      22
#define INT_USB2_0_FUNDAMC  23
#define INT_AHB2_AXI        24
#define INT_MBUS2AXI_DISP   26
#define INT_MBUS2AXI_DRAW   27
#define INT_HBUS2AXI        28
#define INT_MLB_CINT        29
#define INT_MLB_SINT        30
#define INT_MLB_DINT        31

#endif    /* __IOMB86R01_H */
