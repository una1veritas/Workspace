/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton Nano100 Devices
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 52497 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IONANO100_H
#define __IONANO100_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   NANO100 SPECIAL FUNCTION REGISTERS
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

/* System Reset Source Register */
typedef struct {
  __REG32 RSTS_POR        : 1;
  __REG32 RSTS_PAD        : 1;
  __REG32 RSTS_WDT        : 1;
  __REG32                 : 1;
  __REG32 RSTS_BOD        : 1;
  __REG32 RSTS_SYS        : 1;
  __REG32                 : 1;
  __REG32 RSTS_CPU        : 1;
  __REG32                 : 24;
} __gcr_rst_src_bits;

/* IP Reset Control Resister1 */
typedef struct {
  __REG32 CHIP_RST        : 1;
  __REG32 CPU_RST         : 1;
  __REG32 DMA_RST         : 1;
  __REG32 EBI_RST         : 1;
  __REG32                 : 28;
} __gcr_iprst_ctl1_bits;

/* IP Reset Control Resister2 */
typedef struct {
  __REG32                 : 1;
  __REG32 GPIO_RST        : 1;
  __REG32 TMR0_RST        : 1;
  __REG32 TMR1_RST        : 1;
  __REG32 TMR2_RST        : 1;
  __REG32 TMR3_RST        : 1;
  __REG32                 : 2;
  __REG32 I2C0_RST        : 1;
  __REG32 I2C1_RST        : 1;
  __REG32                 : 2;
  __REG32 SPI0_RST        : 1;
  __REG32 SPI1_RST        : 1;
  __REG32 SPI2_RST        : 1;
  __REG32                 : 1;
  __REG32 UART0_RST       : 1;
  __REG32 UART1_RST       : 1;
  __REG32                 : 2;
  __REG32 PWM0_RST        : 1;
  __REG32 PWM1_RST        : 1;
  __REG32                 : 2;
  __REG32 TK_RST          : 1;
  __REG32 DAC_RST         : 1;
  __REG32                 : 2;
  __REG32 ADC_RST         : 1;
  __REG32 I2S_RST         : 1;
  __REG32 SC0_RST         : 1;
  __REG32 SC1_RST         : 1;
} __gcr_iprst_ctl2_bits;

/* Chip Performance Register */
typedef struct {
  __REG32 HPE             : 1;
  __REG32                 : 31;
} __gcr_cpr_bits;

/* Internal Test Controller Register */
typedef struct {
  __REG32 SELF_TEST       : 1;
  __REG32 DELAY_TEST      : 1;
  __REG32                 : 30;
} __gcr_itestcr_bits;

/* Temperature Sensor Control Register */
typedef struct {
  __REG32 VTEMP_EN        : 1;
  __REG32                 : 15;
  __REG32                 : 16;
} __gcr_tempctl_bits;

/* Port A low byte multiple function control register */
typedef struct {
  __REG32 PA0_MFP         : 1;
  __REG32                 : 2;
  __REG32                 : 1;
  __REG32 PA1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA5_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA6_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA7_MFP         : 3;
  __REG32                 : 1;
} __gcr_pa_l_mfp_bits;

/* Port A high byte multiple function control register */
typedef struct {
  __REG32 PA8_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA9_MFP         : 3;
  __REG32                 : 1;
  __REG32 PA10_MFP        : 3;
  __REG32                 : 1;
  __REG32 PA11_MFP        : 3;
  __REG32                 : 1;
  __REG32 PA12_MFP        : 3;
  __REG32                 : 1;
  __REG32 PA13_MFP        : 3;
  __REG32                 : 1;
  __REG32 PA14_MFP        : 3;
  __REG32                 : 1;
  __REG32 PA15_MFP        : 3;
  __REG32                 : 1;
} __gcr_pa_h_mfp_bits;

/* Port B low byte multiple function control register */
typedef struct {
  __REG32 PB0_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB5_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB6_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB7_MFP         : 3;
  __REG32                 : 1;
} __gcr_pb_l_mfp_bits;

/* Port B high byte multiple function control register */
typedef struct {
  __REG32 PB8_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB9_MFP         : 3;
  __REG32                 : 1;
  __REG32 PB10_MFP        : 3;
  __REG32                 : 1;
  __REG32 PB11_MFP        : 3;
  __REG32                 : 1;
  __REG32 PB12_MFP        : 3;
  __REG32                 : 1;
  __REG32 PB13_MFP        : 3;
  __REG32                 : 1;
  __REG32 PB14_MFP        : 3;
  __REG32                 : 1;
  __REG32 PB15_MFP        : 3;
  __REG32                 : 1;
} __gcr_pb_h_mfp_bits;

/* Port C low byte multiple function control register */
typedef struct {
  __REG32 PC0_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC5_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC6_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC7_MFP         : 3;
  __REG32                 : 1;
} __gcr_pc_l_mfp_bits;

/* Port C high byte multiple function control register */
typedef struct {
  __REG32 PC8_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC9_MFP         : 3;
  __REG32                 : 1;
  __REG32 PC10_MFP        : 3;
  __REG32                 : 1;
  __REG32 PC11_MFP        : 3;
  __REG32                 : 1;
  __REG32 PC12_MFP        : 3;
  __REG32                 : 1;
  __REG32 PC13_MFP        : 3;
  __REG32                 : 1;
  __REG32 PC14_MFP        : 3;
  __REG32                 : 1;
  __REG32 PC15_MFP        : 3;
  __REG32                 : 1;
} __gcr_pc_h_mfp_bits;

/* Port D low byte multiple function control register */
typedef struct {
  __REG32 PD0_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD5_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD6_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD7_MFP         : 3;
  __REG32                 : 1;
} __gcr_pd_l_mfp_bits;

/* Port D high byte multiple function control register */
typedef struct {
  __REG32 PD8_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD9_MFP         : 3;
  __REG32                 : 1;
  __REG32 PD10_MFP        : 3;
  __REG32                 : 1;
  __REG32 PD11_MFP        : 3;
  __REG32                 : 1;
  __REG32 PD12_MFP        : 3;
  __REG32                 : 1;
  __REG32 PD13_MFP        : 3;
  __REG32                 : 1;
  __REG32 PD14_MFP        : 3;
  __REG32                 : 1;
  __REG32 PD15_MFP        : 3;
  __REG32                 : 1;
} __gcr_pd_h_mfp_bits;

/* Port E low byte multiple function control register */
typedef struct {
  __REG32 PE0_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE5_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE6_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE7_MFP         : 3;
  __REG32                 : 1;
} __gcr_pe_l_mfp_bits;

/* Port E high byte multiple function control register */
typedef struct {
  __REG32 PE8_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE9_MFP         : 3;
  __REG32                 : 1;
  __REG32 PE10_MFP        : 3;
  __REG32                 : 1;
  __REG32 PE11_MFP        : 3;
  __REG32                 : 1;
  __REG32 PE12_MFP        : 3;
  __REG32                 : 1;
  __REG32 PE13_MFP        : 3;
  __REG32                 : 1;
  __REG32 PE14_MFP        : 3;
  __REG32                 : 1;
  __REG32 PE15_MFP        : 3;
  __REG32                 : 1;
} __gcr_pe_h_mfp_bits;

/* Port F low byte multiple function control register */
typedef struct {
  __REG32 PF0_MFP         : 3;
  __REG32                 : 1;
  __REG32 PF1_MFP         : 3;
  __REG32                 : 1;
  __REG32 PF2_MFP         : 3;
  __REG32                 : 1;
  __REG32 PF3_MFP         : 3;
  __REG32                 : 1;
  __REG32 PF4_MFP         : 3;
  __REG32                 : 1;
  __REG32 PF5_MFP         : 3;
  __REG32                 : 9;
} __gcr_pf_l_mfp_bits;

/* Power-On-Reset Controller Register */
typedef struct {
  __REG32 POR_DIS_CODE    : 16;
  __REG32                 : 16;
} __gcr_porctl_bits;

/* Brown Out Detector Control Register */
typedef struct {
  __REG32 BOD17_EN        : 1;
  __REG32 BOD20_EN        : 1;
  __REG32 BOD25_EN        : 1;
  __REG32                 : 1;
  __REG32 BOD17_RST_EN    : 1;
  __REG32 BOD20_RST_EN    : 1;
  __REG32 BOD25_RST_EN    : 1;
  __REG32                 : 1;
  __REG32 BOD17_INT_EN    : 1;
  __REG32 BOD20_INT_EN    : 1;
  __REG32 BOD25_INT_EN    : 1;
  __REG32                 : 1;
  __REG32 BOD17_TRIM      : 4;
  __REG32 BOD20_TRIM      : 4;
  __REG32 BOD25_TRIM      : 4;
  __REG32                 : 8;
} __gcr_bodctl_bits;

/* Brown Out Detector Status Register */
typedef struct {
  __REG32 BOD_INT         : 1;
  __REG32 BOD17_OUT       : 1;
  __REG32 BOD20_OUT       : 1;
  __REG32 BOD25_OUT       : 1;
  __REG32                 : 28;
} __gcr_bodsts_bits;

/* Voltage reference Control register */
typedef struct {
  __REG32 BGP_EN          : 1;
  __REG32 REG_EN          : 1;
  __REG32 SEL25           : 1;
  __REG32 EXT_MODE        : 1;
  __REG32 BGP_MODE        : 1;
  __REG32 BGP_TEST        : 1;
  __REG32                 : 2;
  __REG32 BGP_TRIM        : 4;
  __REG32                 : 20;
} __gcr_vrefctl_bits;

/* LDO Control Register */
typedef struct {
  __REG32 ADD_3UA         : 1;
  __REG32 ADD_15UA        : 1;
  __REG32 LDO_LEVEL       : 2;
  __REG32                 : 28;
} __gcr_ldoctl_bits;

/* Voltage Detector Control Register */
typedef struct {
  __REG32 VD_10K_OFF      : 1;
  __REG32 VD_HYS_EN       : 1;
  __REG32 SCAN_EN         : 1;
  __REG32                 : 29;
} __gcr_vdctl_bits;

/* Low Power LDO Control Register */
typedef struct {
  __REG32 LPLDO_EN        : 1;
  __REG32 LPLDO_MODE      : 1;
  __REG32                 : 2;
  __REG32 LPLDO_REGN      : 4;
  __REG32 LPLDO_REGP      : 4;
  __REG32                 : 20;
} __gcr_lpldoctl_bits;

/* HIRC Trim Control Register */
typedef struct {
  __REG32 TRIM_SEL        : 2;
  __REG32                 : 2;
  __REG32 TRIM_LOOP       : 2;
  __REG32 TRIM_RETRY_CNT  : 2;
  __REG32                 : 24;
} __gcr_irctrimctl_bits;

/* HIRC Trim Interrupt Enable Register */
typedef struct {
  __REG32                 : 1;
  __REG32 TRIM_FAIL_IEN   : 1;
  __REG32 _32K_ERR_IEN    : 1;
  __REG32                 : 29;
} __gcr_irctrimien_bits;

/* HIRC Trim Interrupt Status Register */
typedef struct {
  __REG32 FREQ_LOCK       : 1;
  __REG32 TRIM_FAIL_INT   : 1;
  __REG32 _32K_ERR_INT    : 1;
  __REG32                 : 29;
} __gcr_irctrimint_bits;

/* Register Lock Key address */
typedef struct {
  __REG32 RegUnLock       : 1;
  __REG32                 : 31;
} __gcr_reglockaddr_bits;

/* RC Adjustment Value */
typedef struct {
  __REG32 RCADJ           : 8;
  __REG32                 : 24;
} __gcr_rcadj_bits;

/* SysTick Control and Status */
typedef struct {
  __REG32 ENABLE          : 1;
  __REG32 TICKINT         : 1;
  __REG32 CLKSRC          : 1;
  __REG32                 : 13;
  __REG32 COUNTFLAG       : 1;
  __REG32                 : 15;
} __scs_syst_ctl_bits;

/* SysTick Reload value */
typedef struct {
  __REG32 RELOAD          : 24;
  __REG32                 : 8;
} __scs_syst_rvr_bits;

/* SysTick Current value */
typedef struct {
  __REG32 CURRENT         : 24;
  __REG32                 : 8;
} __scs_syst_cvr_bits;

/* IRQ0 ~ IRQ3 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_0           : 2;
  __REG32                 : 6;
  __REG32 PRI_1           : 2;
  __REG32                 : 6;
  __REG32 PRI_2           : 2;
  __REG32                 : 6;
  __REG32 PRI_3           : 2;
} __scs_nvic_ipr0_bits;

/* IRQ4 ~ IRQ7 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_4           : 2;
  __REG32                 : 6;
  __REG32 PRI_5           : 2;
  __REG32                 : 6;
  __REG32 PRI_6           : 2;
  __REG32                 : 6;
  __REG32 PRI_7           : 2;
} __scs_nvic_ipr1_bits;

/* IRQ8 ~ IRQ11 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_8           : 2;
  __REG32                 : 6;
  __REG32 PRI_9           : 2;
  __REG32                 : 6;
  __REG32 PRI_10          : 2;
  __REG32                 : 6;
  __REG32 PRI_11          : 2;
} __scs_nvic_ipr2_bits;

/* IRQ12 ~ IRQ15 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_12          : 2;
  __REG32                 : 6;
  __REG32 PRI_13          : 2;
  __REG32                 : 6;
  __REG32 PRI_14          : 2;
  __REG32                 : 6;
  __REG32 PRI_15          : 2;
} __scs_nvic_ipr3_bits;

/* IRQ16 ~ IRQ19 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_16          : 2;
  __REG32                 : 6;
  __REG32 PRI_17          : 2;
  __REG32                 : 6;
  __REG32 PRI_18          : 2;
  __REG32                 : 6;
  __REG32 PRI_19          : 2;
} __scs_nvic_ipr4_bits;

/* IRQ20 ~ IRQ23 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_20          : 2;
  __REG32                 : 6;
  __REG32 PRI_21          : 2;
  __REG32                 : 6;
  __REG32 PRI_22          : 2;
  __REG32                 : 6;
  __REG32 PRI_23          : 2;
} __scs_nvic_ipr5_bits;

/* IRQ24 ~ IRQ27 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_24          : 2;
  __REG32                 : 6;
  __REG32 PRI_25          : 2;
  __REG32                 : 6;
  __REG32 PRI_26          : 2;
  __REG32                 : 6;
  __REG32 PRI_27          : 2;
} __scs_nvic_ipr6_bits;

/* IRQ28 ~ IRQ31 Priority Control Register */
typedef struct {
  __REG32                 : 6;
  __REG32 PRI_28          : 2;
  __REG32                 : 6;
  __REG32 PRI_29          : 2;
  __REG32                 : 6;
  __REG32 PRI_30          : 2;
  __REG32                 : 6;
  __REG32 PRI_31          : 2;
} __scs_nvic_ipr7_bits;

/* CPUID Base Register */
typedef struct {
  __REG32 REVISION        : 4;
  __REG32 PARTNO          : 12;
  __REG32 PART            : 4;
  __REG32                 : 4;
  __REG32 IMPLEMENTER     : 8;
} __scs_cpuid_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32 VECTACTIVE      : 9;
  __REG32                 : 3;
  __REG32 VECTPENDING     : 9;
  __REG32                 : 1;
  __REG32 ISRPENDING      : 1;
  __REG32 ISRPREEMPT      : 1;
  __REG32                 : 1;
  __REG32 PENDSTCLR       : 1;
  __REG32 PENDSTSET       : 1;
  __REG32 PENDSVCLR       : 1;
  __REG32 PENDSVSET       : 1;
  __REG32                 : 2;
  __REG32 NMIPENDSET      : 1;
} __scs_icsr_bits;

/* System Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 SLEEPONEXIT     : 1;
  __REG32 SLEEPDEEP       : 1;
  __REG32                 : 1;
  __REG32 SEVONPEND       : 1;
  __REG32                 : 27;
} __scs_scr_bits;

/* System Handler Priority Register 2 */
typedef struct {
  __REG32                 : 30;
  __REG32 PRI_11          : 2;
} __scs_shpr2_bits;

/* System Handler Priority Register 3 */
typedef struct {
  __REG32                 : 22;
  __REG32 PRI_14          : 2;
  __REG32                 : 6;
  __REG32 PRI_15          : 2;
} __scs_shpr3_bits;

/* MCU IRQ0 (BOD) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq0_src_bits;

/* MCU IRQ1 (WDT) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq1_src_bits;

/* MCU IRQ2 ((EINT0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq2_src_bits;

/* MCU IRQ3 (EINT1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq3_src_bits;

/* MCU IRQ4 (GPA/B) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq4_src_bits;

/* MCU IRQ5 (GPC/D/E) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq5_src_bits;

/* MCU IRQ6 (PWM0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 4;
  __REG32                 : 28;
} __int_irq6_src_bits;

/* MCU IRQ7 (PWM1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 4;
  __REG32                 : 28;
} __int_irq7_src_bits;

/* MCU IRQ8 (TMR0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq8_src_bits;

/* MCU IRQ9 (TMR1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq9_src_bits;

/* MCU IRQ10 (TMR2) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq10_src_bits;

/* MCU IRQ11 (TMR3) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq11_src_bits;

/* MCU IRQ12 (URT0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq12_src_bits;

/* MCU IRQ13 (URT1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq13_src_bits;

/* MCU IRQ14 (SPI0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq14_src_bits;

/* MCU IRQ15 (SPI1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq15_src_bits;

/* MCU IRQ18 (I2C0) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq18_src_bits;

/* MCU IRQ19 (I2C1) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq19_src_bits;

/* MCU IRQ20 (Reserved) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq20_src_bits;

/* MCU IRQ23 (USBD) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq23_src_bits;

/* MCU IRQ24 (Touch key) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq24_src_bits;

/* MCU IRQ25 (LCD) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq25_src_bits;

/* MCU IRQ26 (DMA) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq26_src_bits;

/* MCU IRQ27 (I2S) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq27_src_bits;

/* MCU IRQ28 (PWRWU) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq28_src_bits;

/* MCU IRQ29 (ADC) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq29_src_bits;

/* MCU IRQ30 (DAC) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq30_src_bits;

/* MCU IRQ31 (RTC) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irq31_src_bits;

/* NMI source interrupt select control register */
typedef struct {
  __REG32 NMI_SEL         : 5;
  __REG32                 : 2;
  __REG32 IRQTEST         : 1;
  __REG32                 : 24;
} __int_nmi_sel_bits;

/* System Power Down Control Register */
typedef struct {
  __REG32 HXT_EN          : 1;
  __REG32 LXT_EN          : 1;
  __REG32 HIRC_EN         : 1;
  __REG32 LIRC_EN         : 1;
  __REG32 WK_DLY          : 1;
  __REG32 PD_WK_IE        : 1;
  __REG32 PD_EN           : 1;
  __REG32 DPD_EN          : 1;
  __REG32 HXT_SELXT       : 1;
  __REG32 HXT_GAIN        : 1;
  __REG32 LXT_SCNT        : 1;
  __REG32                 : 21;
} __clk_pwrctl_bits;

/* AHB Devices Clock Enable Control Register */
typedef struct {
  __REG32 GPIO_EN         : 1;
  __REG32 DMA_EN          : 1;
  __REG32 ISP_EN          : 1;
  __REG32 EBI_EN          : 1;
  __REG32 SRAM_EN         : 1;
  __REG32 TICK_EN         : 1;
  __REG32                 : 26;
} __clk_ahbclk_bits;

/* APB Devices Clock Enable Control Register */
typedef struct {
  __REG32 WDT_EN          : 1;
  __REG32 RTC_EN          : 1;
  __REG32 TMR0_EN         : 1;
  __REG32 TMR1_EN         : 1;
  __REG32 TMR2_EN         : 1;
  __REG32 TMR3_EN         : 1;
  __REG32 FDIV_EN         : 1;
  __REG32                 : 1;
  __REG32 I2C0_EN         : 1;
  __REG32 I2C1_EN         : 1;
  __REG32                 : 2;
  __REG32 SPI0_EN         : 1;
  __REG32 SPI1_EN         : 1;
  __REG32 SPI2_EN         : 1;
  __REG32                 : 1;
  __REG32 UART0_EN        : 1;
  __REG32 UART1_EN        : 1;
  __REG32                 : 2;
  __REG32 PWM0_CH01_EN    : 1;
  __REG32 PWM0_CH23_EN    : 1;
  __REG32 PWM1_CH01_EN    : 1;
  __REG32 PWM1_CH23_EN    : 1;
  __REG32 TK_EN           : 1;
  __REG32 DAC_EN          : 1;
  __REG32                 : 2;
  __REG32 ADC_EN          : 1;
  __REG32 I2S_EN          : 1;
  __REG32 SC0_EN          : 1;
  __REG32 SC1_EN          : 1;
} __clk_apbclk_bits;

/* Clock status monitor Register */
typedef struct {
  __REG32 HXT_STB         : 1;
  __REG32 LXT_STB         : 1;
  __REG32 PLL_STB         : 1;
  __REG32 LIRC_STB        : 1;
  __REG32 HIRC_STB        : 1;
  __REG32                 : 2;
  __REG32 CLK_SW_FAIL     : 1;
  __REG32                 : 24;
} __clk_clkstatus_bits;

/* Clock Source Select Control Register 0 */
typedef struct {
  __REG32 HCLK_S          : 3;
  __REG32                 : 29;
} __clk_clksel0_bits;

/* Clock Source Select Control Register 1 */
typedef struct {
  __REG32 UART_S          : 2;
  __REG32 ADC_S           : 2;
  __REG32 PWM0_CH01_S     : 2;
  __REG32 PWM0_CH23_S     : 2;
  __REG32 TMR0_S          : 3;
  __REG32                 : 1;
  __REG32 TMR1_S          : 3;
  __REG32                 : 1;
  __REG32 TK_S            : 2;
  __REG32                 : 14;
} __clk_clksel1_bits;

/* Clock Source Select Control Register 2 */
typedef struct {
  __REG32                 : 2;
  __REG32 FRQDIV_S        : 2;
  __REG32 PWM1_CH01_S     : 2;
  __REG32 PWM1_CH23_S     : 2;
  __REG32 TMR2_S          : 3;
  __REG32                 : 1;
  __REG32 TMR3_S          : 3;
  __REG32                 : 1;
  __REG32 I2S_S           : 2;
  __REG32 SC_S            : 2;
  __REG32                 : 12;
} __clk_clksel2_bits;

/* Clock Divider Number Register 0 */
typedef struct {
  __REG32 HCLK_N          : 4;
  __REG32                 : 4;
  __REG32 UART_N          : 4;
  __REG32 I2S_N           : 4;
  __REG32 ADC_N           : 8;
  __REG32 TK_N            : 4;
  __REG32 SC0_N           : 4;
} __clk_clkdiv0_bits;

/* Clock Divider Number Register 1 */
typedef struct {
  __REG32 SC1_N           : 4;
  __REG32                 : 28;
} __clk_clkdiv1_bits;

/* PLL Control Register */
typedef struct {
  __REG32 FB_DV           : 6;
  __REG32                 : 2;
  __REG32 IN_DV           : 2;
  __REG32                 : 2;
  __REG32 OUT_DV          : 1;
  __REG32                 : 3;
  __REG32 PD              : 1;
  __REG32 PLL_SRC         : 1;
  __REG32                 : 14;
} __clk_pllctl_bits;

/* Frequency Divider Control Register */
typedef struct {
  __REG32 FSEL            : 4;
  __REG32 FDIV_EN         : 1;
  __REG32                 : 27;
} __clk_frqdiv_bits;

/* Test clock source Select Control Register */
typedef struct {
  __REG32 TCLK_SEL        : 6;
  __REG32                 : 1;
  __REG32 TEST_EN         : 1;
  __REG32                 : 24;
} __clk_testclk_bits;

/* Power down wake-up interrupt status register */
typedef struct {
  __REG32 PD_WK_IS        : 1;
  __REG32                 : 31;
} __clk_pd_wk_is_bits;

/* GPIO Port A Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpioa_pmd_bits;

/* GPIO Port A Pin OFF Digital Enable Register */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 16;
} __gpio_gpioa_offd_bits;

/* GPIO Port A Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32 DOUT8           : 1;
  __REG32 DOUT9           : 1;
  __REG32 DOUT10          : 1;
  __REG32 DOUT11          : 1;
  __REG32 DOUT12          : 1;
  __REG32 DOUT13          : 1;
  __REG32 DOUT14          : 1;
  __REG32 DOUT15          : 1;
  __REG32                 : 16;
} __gpio_gpioa_dout_bits;

/* GPIO Port A Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32 DMASK8          : 1;
  __REG32 DMASK9          : 1;
  __REG32 DMASK10         : 1;
  __REG32 DMASK11         : 1;
  __REG32 DMASK12         : 1;
  __REG32 DMASK13         : 1;
  __REG32 DMASK14         : 1;
  __REG32 DMASK15         : 1;
  __REG32                 : 16;
} __gpio_gpioa_dmask_bits;

/* GPIO Port A Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32 PIN8            : 1;
  __REG32 PIN9            : 1;
  __REG32 PIN10           : 1;
  __REG32 PIN11           : 1;
  __REG32 PIN12           : 1;
  __REG32 PIN13           : 1;
  __REG32 PIN14           : 1;
  __REG32 PIN15           : 1;
  __REG32                 : 16;
} __gpio_gpioa_pin_bits;

/* GPIO Port A De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32 DBEN8           : 1;
  __REG32 DBEN9           : 1;
  __REG32 DBEN10          : 1;
  __REG32 DBEN11          : 1;
  __REG32 DBEN12          : 1;
  __REG32 DBEN13          : 1;
  __REG32 DBEN14          : 1;
  __REG32 DBEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioa_dben_bits;

/* GPIO Port A Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32 IMD8            : 1;
  __REG32 IMD9            : 1;
  __REG32 IMD10           : 1;
  __REG32 IMD11           : 1;
  __REG32 IMD12           : 1;
  __REG32 IMD13           : 1;
  __REG32 IMD14           : 1;
  __REG32 IMD15           : 1;
  __REG32                 : 16;
} __gpio_gpioa_imd_bits;

/* GPIO Port A Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32 FIER6           : 1;
  __REG32 FIER7           : 1;
  __REG32 FIER8           : 1;
  __REG32 FIER9           : 1;
  __REG32 FIER10          : 1;
  __REG32 FIER11          : 1;
  __REG32 FIER12          : 1;
  __REG32 FIER13          : 1;
  __REG32 FIER14          : 1;
  __REG32 FIER15          : 1;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32 RIER6           : 1;
  __REG32 RIER7           : 1;
  __REG32 RIER8           : 1;
  __REG32 RIER9           : 1;
  __REG32 RIER10          : 1;
  __REG32 RIER11          : 1;
  __REG32 RIER12          : 1;
  __REG32 RIER13          : 1;
  __REG32 RIER14          : 1;
  __REG32 RIER15          : 1;
} __gpio_gpioa_ier_bits;

/* GPIO Port A Interrupt Trigger Source Indicator Status Register */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32 ISRC8           : 1;
  __REG32 ISRC9           : 1;
  __REG32 ISRC10          : 1;
  __REG32 ISRC11          : 1;
  __REG32 ISRC12          : 1;
  __REG32 ISRC13          : 1;
  __REG32 ISRC14          : 1;
  __REG32 ISRC15          : 1;
  __REG32                 : 16;
} __gpio_gpioa_isrc_bits;

/* GPIO Port A Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32 PUEN6           : 1;
  __REG32 PUEN7           : 1;
  __REG32 PUEN8           : 1;
  __REG32 PUEN9           : 1;
  __REG32 PUEN10          : 1;
  __REG32 PUEN11          : 1;
  __REG32 PUEN12          : 1;
  __REG32 PUEN13          : 1;
  __REG32 PUEN14          : 1;
  __REG32 PUEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioa_puen_bits;

/* GPIO Port B Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpiob_pmd_bits;

/* GPIO Port B Pin OFF Digital Enable RegisterGPIO Port B Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 16;
} __gpio_gpiob_offd_bits;

/* GPIO Port B Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32 DOUT8           : 1;
  __REG32 DOUT9           : 1;
  __REG32 DOUT10          : 1;
  __REG32 DOUT11          : 1;
  __REG32 DOUT12          : 1;
  __REG32 DOUT13          : 1;
  __REG32 DOUT14          : 1;
  __REG32 DOUT15          : 1;
  __REG32                 : 16;
} __gpio_gpiob_dout_bits;

/* GPIO Port B Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32 DMASK8          : 1;
  __REG32 DMASK9          : 1;
  __REG32 DMASK10         : 1;
  __REG32 DMASK11         : 1;
  __REG32 DMASK12         : 1;
  __REG32 DMASK13         : 1;
  __REG32 DMASK14         : 1;
  __REG32 DMASK15         : 1;
  __REG32                 : 16;
} __gpio_gpiob_dmask_bits;

/* GPIO Port B Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32 PIN8            : 1;
  __REG32 PIN9            : 1;
  __REG32 PIN10           : 1;
  __REG32 PIN11           : 1;
  __REG32 PIN12           : 1;
  __REG32 PIN13           : 1;
  __REG32 PIN14           : 1;
  __REG32 PIN15           : 1;
  __REG32                 : 16;
} __gpio_gpiob_pin_bits;

/* GPIO Port B De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32 DBEN8           : 1;
  __REG32 DBEN9           : 1;
  __REG32 DBEN10          : 1;
  __REG32 DBEN11          : 1;
  __REG32 DBEN12          : 1;
  __REG32 DBEN13          : 1;
  __REG32 DBEN14          : 1;
  __REG32 DBEN15          : 1;
  __REG32                 : 16;
} __gpio_gpiob_dben_bits;

/* GPIO Port B Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32 IMD8            : 1;
  __REG32 IMD9            : 1;
  __REG32 IMD10           : 1;
  __REG32 IMD11           : 1;
  __REG32 IMD12           : 1;
  __REG32 IMD13           : 1;
  __REG32 IMD14           : 1;
  __REG32 IMD15           : 1;
  __REG32                 : 16;
} __gpio_gpiob_imd_bits;

/* GPIO Port B Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32 FIER6           : 1;
  __REG32 FIER7           : 1;
  __REG32 FIER8           : 1;
  __REG32 FIER9           : 1;
  __REG32 FIER10          : 1;
  __REG32 FIER11          : 1;
  __REG32 FIER12          : 1;
  __REG32 FIER13          : 1;
  __REG32 FIER14          : 1;
  __REG32 FIER15          : 1;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32 RIER6           : 1;
  __REG32 RIER7           : 1;
  __REG32 RIER8           : 1;
  __REG32 RIER9           : 1;
  __REG32 RIER10          : 1;
  __REG32 RIER11          : 1;
  __REG32 RIER12          : 1;
  __REG32 RIER13          : 1;
  __REG32 RIER14          : 1;
  __REG32 RIER15          : 1;
} __gpio_gpiob_ier_bits;

/* GPIO Port B Interrupt Trigger Source Status RegisterIndicator */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32 ISRC8           : 1;
  __REG32 ISRC9           : 1;
  __REG32 ISRC10          : 1;
  __REG32 ISRC11          : 1;
  __REG32 ISRC12          : 1;
  __REG32 ISRC13          : 1;
  __REG32 ISRC14          : 1;
  __REG32 ISRC15          : 1;
  __REG32                 : 16;
} __gpio_gpiob_isrc_bits;

/* GPIO Port B Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32 PUEN6           : 1;
  __REG32 PUEN7           : 1;
  __REG32 PUEN8           : 1;
  __REG32 PUEN9           : 1;
  __REG32 PUEN10          : 1;
  __REG32 PUEN11          : 1;
  __REG32 PUEN12          : 1;
  __REG32 PUEN13          : 1;
  __REG32 PUEN14          : 1;
  __REG32 PUEN15          : 1;
  __REG32                 : 16;
} __gpio_gpiob_puen_bits;

/* GPIO Port C Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpioc_pmd_bits;

/* GPIO Port C Pin OFF Digital Enable RegisterGPIO Port C Bit OFF digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 16;
} __gpio_gpioc_offd_bits;

/* GPIO Port C Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32 DOUT8           : 1;
  __REG32 DOUT9           : 1;
  __REG32 DOUT10          : 1;
  __REG32 DOUT11          : 1;
  __REG32 DOUT12          : 1;
  __REG32 DOUT13          : 1;
  __REG32 DOUT14          : 1;
  __REG32 DOUT15          : 1;
  __REG32                 : 16;
} __gpio_gpioc_dout_bits;

/* GPIO Port C Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32 DMASK8          : 1;
  __REG32 DMASK9          : 1;
  __REG32 DMASK10         : 1;
  __REG32 DMASK11         : 1;
  __REG32 DMASK12         : 1;
  __REG32 DMASK13         : 1;
  __REG32 DMASK14         : 1;
  __REG32 DMASK15         : 1;
  __REG32                 : 16;
} __gpio_gpioc_dmask_bits;

/* GPIO Port C Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32 PIN8            : 1;
  __REG32 PIN9            : 1;
  __REG32 PIN10           : 1;
  __REG32 PIN11           : 1;
  __REG32 PIN12           : 1;
  __REG32 PIN13           : 1;
  __REG32 PIN14           : 1;
  __REG32 PIN15           : 1;
  __REG32                 : 16;
} __gpio_gpioc_pin_bits;

/* GPIO Port C De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32 DBEN8           : 1;
  __REG32 DBEN9           : 1;
  __REG32 DBEN10          : 1;
  __REG32 DBEN11          : 1;
  __REG32 DBEN12          : 1;
  __REG32 DBEN13          : 1;
  __REG32 DBEN14          : 1;
  __REG32 DBEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioc_dben_bits;

/* GPIO Port C Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32 IMD8            : 1;
  __REG32 IMD9            : 1;
  __REG32 IMD10           : 1;
  __REG32 IMD11           : 1;
  __REG32 IMD12           : 1;
  __REG32 IMD13           : 1;
  __REG32 IMD14           : 1;
  __REG32 IMD15           : 1;
  __REG32                 : 16;
} __gpio_gpioc_imd_bits;

/* GPIO Port C Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32 FIER6           : 1;
  __REG32 FIER7           : 1;
  __REG32 FIER8           : 1;
  __REG32 FIER9           : 1;
  __REG32 FIER10          : 1;
  __REG32 FIER11          : 1;
  __REG32 FIER12          : 1;
  __REG32 FIER13          : 1;
  __REG32 FIER14          : 1;
  __REG32 FIER15          : 1;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32 RIER6           : 1;
  __REG32 RIER7           : 1;
  __REG32 RIER8           : 1;
  __REG32 RIER9           : 1;
  __REG32 RIER10          : 1;
  __REG32 RIER11          : 1;
  __REG32 RIER12          : 1;
  __REG32 RIER13          : 1;
  __REG32 RIER14          : 1;
  __REG32 RIER15          : 1;
} __gpio_gpioc_ier_bits;

/* GPIO Port C Interrupt Trigger Source Status RegisterIndicator */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32 ISRC8           : 1;
  __REG32 ISRC9           : 1;
  __REG32 ISRC10          : 1;
  __REG32 ISRC11          : 1;
  __REG32 ISRC12          : 1;
  __REG32 ISRC13          : 1;
  __REG32 ISRC14          : 1;
  __REG32 ISRC15          : 1;
  __REG32                 : 16;
} __gpio_gpioc_isrc_bits;

/* GPIO Port C Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32 PUEN6           : 1;
  __REG32 PUEN7           : 1;
  __REG32 PUEN8           : 1;
  __REG32 PUEN9           : 1;
  __REG32 PUEN10          : 1;
  __REG32 PUEN11          : 1;
  __REG32 PUEN12          : 1;
  __REG32 PUEN13          : 1;
  __REG32 PUEN14          : 1;
  __REG32 PUEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioc_puen_bits;

/* GPIO Port D Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpiod_pmd_bits;

/* GPIO Port D Pin OFF Digital Enable RegisterGPIO Port D Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 16;
} __gpio_gpiod_offd_bits;

/* GPIO Port D Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32 DOUT8           : 1;
  __REG32 DOUT9           : 1;
  __REG32 DOUT10          : 1;
  __REG32 DOUT11          : 1;
  __REG32 DOUT12          : 1;
  __REG32 DOUT13          : 1;
  __REG32 DOUT14          : 1;
  __REG32 DOUT15          : 1;
  __REG32                 : 16;
} __gpio_gpiod_dout_bits;

/* GPIO Port D Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32 DMASK8          : 1;
  __REG32 DMASK9          : 1;
  __REG32 DMASK10         : 1;
  __REG32 DMASK11         : 1;
  __REG32 DMASK12         : 1;
  __REG32 DMASK13         : 1;
  __REG32 DMASK14         : 1;
  __REG32 DMASK15         : 1;
  __REG32                 : 16;
} __gpio_gpiod_dmask_bits;

/* GPIO Port D Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32 PIN8            : 1;
  __REG32 PIN9            : 1;
  __REG32 PIN10           : 1;
  __REG32 PIN11           : 1;
  __REG32 PIN12           : 1;
  __REG32 PIN13           : 1;
  __REG32 PIN14           : 1;
  __REG32 PIN15           : 1;
  __REG32                 : 16;
} __gpio_gpiod_pin_bits;

/* GPIO Port D De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32 DBEN8           : 1;
  __REG32 DBEN9           : 1;
  __REG32 DBEN10          : 1;
  __REG32 DBEN11          : 1;
  __REG32 DBEN12          : 1;
  __REG32 DBEN13          : 1;
  __REG32 DBEN14          : 1;
  __REG32 DBEN15          : 1;
  __REG32                 : 16;
} __gpio_gpiod_dben_bits;

/* GPIO Port D Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32 IMD8            : 1;
  __REG32 IMD9            : 1;
  __REG32 IMD10           : 1;
  __REG32 IMD11           : 1;
  __REG32 IMD12           : 1;
  __REG32 IMD13           : 1;
  __REG32 IMD14           : 1;
  __REG32 IMD15           : 1;
  __REG32                 : 16;
} __gpio_gpiod_imd_bits;

/* GPIO Port D Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32 FIER6           : 1;
  __REG32 FIER7           : 1;
  __REG32 FIER8           : 1;
  __REG32 FIER9           : 1;
  __REG32 FIER10          : 1;
  __REG32 FIER11          : 1;
  __REG32 FIER12          : 1;
  __REG32 FIER13          : 1;
  __REG32 FIER14          : 1;
  __REG32 FIER15          : 1;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32 RIER6           : 1;
  __REG32 RIER7           : 1;
  __REG32 RIER8           : 1;
  __REG32 RIER9           : 1;
  __REG32 RIER10          : 1;
  __REG32 RIER11          : 1;
  __REG32 RIER12          : 1;
  __REG32 RIER13          : 1;
  __REG32 RIER14          : 1;
  __REG32 RIER15          : 1;
} __gpio_gpiod_ier_bits;

/* GPIO Port D Interrupt Trigger Source Status RegisterIndicator */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32 ISRC8           : 1;
  __REG32 ISRC9           : 1;
  __REG32 ISRC10          : 1;
  __REG32 ISRC11          : 1;
  __REG32 ISRC12          : 1;
  __REG32 ISRC13          : 1;
  __REG32 ISRC14          : 1;
  __REG32 ISRC15          : 1;
  __REG32                 : 16;
} __gpio_gpiod_isrc_bits;

/* GPIO Port D Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32 PUEN6           : 1;
  __REG32 PUEN7           : 1;
  __REG32 PUEN8           : 1;
  __REG32 PUEN9           : 1;
  __REG32 PUEN10          : 1;
  __REG32 PUEN11          : 1;
  __REG32 PUEN12          : 1;
  __REG32 PUEN13          : 1;
  __REG32 PUEN14          : 1;
  __REG32 PUEN15          : 1;
  __REG32                 : 16;
} __gpio_gpiod_puen_bits;

/* GPIO Port E Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpioe_pmd_bits;

/* GPIO Port E Pin OFF Digital Enable RegisterGPIO Port E Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 16;
} __gpio_gpioe_offd_bits;

/* GPIO Port E Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32 DOUT8           : 1;
  __REG32 DOUT9           : 1;
  __REG32 DOUT10          : 1;
  __REG32 DOUT11          : 1;
  __REG32 DOUT12          : 1;
  __REG32 DOUT13          : 1;
  __REG32 DOUT14          : 1;
  __REG32 DOUT15          : 1;
  __REG32                 : 16;
} __gpio_gpioe_dout_bits;

/* GPIO Port E Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32 DMASK8          : 1;
  __REG32 DMASK9          : 1;
  __REG32 DMASK10         : 1;
  __REG32 DMASK11         : 1;
  __REG32 DMASK12         : 1;
  __REG32 DMASK13         : 1;
  __REG32 DMASK14         : 1;
  __REG32 DMASK15         : 1;
  __REG32                 : 16;
} __gpio_gpioe_dmask_bits;

/* GPIO Port E Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32 PIN8            : 1;
  __REG32 PIN9            : 1;
  __REG32 PIN10           : 1;
  __REG32 PIN11           : 1;
  __REG32 PIN12           : 1;
  __REG32 PIN13           : 1;
  __REG32 PIN14           : 1;
  __REG32 PIN15           : 1;
  __REG32                 : 16;
} __gpio_gpioe_pin_bits;

/* GPIO Port E De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32 DBEN8           : 1;
  __REG32 DBEN9           : 1;
  __REG32 DBEN10          : 1;
  __REG32 DBEN11          : 1;
  __REG32 DBEN12          : 1;
  __REG32 DBEN13          : 1;
  __REG32 DBEN14          : 1;
  __REG32 DBEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioe_dben_bits;

/* GPIO Port E Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32 IMD8            : 1;
  __REG32 IMD9            : 1;
  __REG32 IMD10           : 1;
  __REG32 IMD11           : 1;
  __REG32 IMD12           : 1;
  __REG32 IMD13           : 1;
  __REG32 IMD14           : 1;
  __REG32 IMD15           : 1;
  __REG32                 : 16;
} __gpio_gpioe_imd_bits;

/* GPIO Port E Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32 FIER6           : 1;
  __REG32 FIER7           : 1;
  __REG32 FIER8           : 1;
  __REG32 FIER9           : 1;
  __REG32 FIER10          : 1;
  __REG32 FIER11          : 1;
  __REG32 FIER12          : 1;
  __REG32 FIER13          : 1;
  __REG32 FIER14          : 1;
  __REG32 FIER15          : 1;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32 RIER6           : 1;
  __REG32 RIER7           : 1;
  __REG32 RIER8           : 1;
  __REG32 RIER9           : 1;
  __REG32 RIER10          : 1;
  __REG32 RIER11          : 1;
  __REG32 RIER12          : 1;
  __REG32 RIER13          : 1;
  __REG32 RIER14          : 1;
  __REG32 RIER15          : 1;
} __gpio_gpioe_ier_bits;

/* GPIO Port E Interrupt Trigger Source Status RegisterIndicator */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32 ISRC8           : 1;
  __REG32 ISRC9           : 1;
  __REG32 ISRC10          : 1;
  __REG32 ISRC11          : 1;
  __REG32 ISRC12          : 1;
  __REG32 ISRC13          : 1;
  __REG32 ISRC14          : 1;
  __REG32 ISRC15          : 1;
  __REG32                 : 16;
} __gpio_gpioe_isrc_bits;

/* GPIO Port E Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32 PUEN6           : 1;
  __REG32 PUEN7           : 1;
  __REG32 PUEN8           : 1;
  __REG32 PUEN9           : 1;
  __REG32 PUEN10          : 1;
  __REG32 PUEN11          : 1;
  __REG32 PUEN12          : 1;
  __REG32 PUEN13          : 1;
  __REG32 PUEN14          : 1;
  __REG32 PUEN15          : 1;
  __REG32                 : 16;
} __gpio_gpioe_puen_bits;

/* GPIO Port F Pin I/O Mode Control Register */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32 PMD8            : 2;
  __REG32 PMD9            : 2;
  __REG32 PMD10           : 2;
  __REG32 PMD11           : 2;
  __REG32 PMD12           : 2;
  __REG32 PMD13           : 2;
  __REG32 PMD14           : 2;
  __REG32 PMD15           : 2;
} __gpio_gpiof_pmd_bits;

/* GPIO Port F Pin OFF Digital Enable RegisterGPIO Port F Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 15;
  __REG32                 : 1;
  __REG32 OFFD            : 16;
} __gpio_gpiof_offd_bits;

/* GPIO Port F Data Output Value Register */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32                 : 26;
} __gpio_gpiof_dout_bits;

/* GPIO Port F Data Output Write Mask Register */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32                 : 26;
} __gpio_gpiof_dmask_bits;

/* GPIO Port F Pin Value Register */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32                 : 26;
} __gpio_gpiof_pin_bits;

/* GPIO Port F De-bounce Enable Register */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32                 : 26;
} __gpio_gpiof_dben_bits;

/* GPIO Port F Interrupt Mode Control Register */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32                 : 26;
} __gpio_gpiof_imd_bits;

/* GPIO Port F Interrupt Enable Register */
typedef struct {
  __REG32 FIER0           : 1;
  __REG32 FIER1           : 1;
  __REG32 FIER2           : 1;
  __REG32 FIER3           : 1;
  __REG32 FIER4           : 1;
  __REG32 FIER5           : 1;
  __REG32                 : 10;
  __REG32 RIER0           : 1;
  __REG32 RIER1           : 1;
  __REG32 RIER2           : 1;
  __REG32 RIER3           : 1;
  __REG32 RIER4           : 1;
  __REG32 RIER5           : 1;
  __REG32                 : 10;
} __gpio_gpiof_ier_bits;

/* GPIO Port F Interrupt Trigger Source Status Register */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32                 : 26;
} __gpio_gpiof_isrc_bits;

/* GPIO Port F Pull-Up Enable Register */
typedef struct {
  __REG32 PUEN0           : 1;
  __REG32 PUEN1           : 1;
  __REG32 PUEN2           : 1;
  __REG32 PUEN3           : 1;
  __REG32 PUEN4           : 1;
  __REG32 PUEN5           : 1;
  __REG32                 : 26;
} __gpio_gpiof_puen_bits;

/* De-bounce Cycle Control Register */
typedef struct {
  __REG32 DBCLKSEL        : 4;
  __REG32 DBCLKSRC        : 1;
  __REG32 DBCLK_ON        : 1;
  __REG32                 : 26;
} __gpio_dbncecon_bits;

/* GPIO Port X Bit Y Data Register */
typedef struct {
  __REG32 GPIOxx          : 1;
  __REG32                 : 31;
} __gpio_gpioxy_bits;

/* I2C Control Register */
typedef struct {
  __REG32 IPEN            : 1;
  __REG32 ACK             : 1;
  __REG32 STOP            : 1;
  __REG32 START           : 1;
  __REG32 I2C_STS         : 1;
  __REG32                 : 2;
  __REG32 INTEN           : 1;
  __REG32                 : 24;
} __i2cx_i2ccon_bits;

/* I2C Control Flag Register */
typedef struct {
  __REG32 STAINTSTS       : 1;
  __REG32 TOUTSTS         : 1;
  __REG32                 : 30;
} __i2cx_i2cintsts_bits;

/* I2C Status Register */
typedef struct {
  __REG32 STATUS          : 8;
  __REG32                 : 24;
} __i2cx_i2cstatus_bits;

/* I2C clock divided Register */
typedef struct {
  __REG32 CLK_DIV         : 8;
  __REG32                 : 24;
} __i2cx_i2cdiv_bits;

/* I2C Time out control Register */
typedef struct {
  __REG32 TOUTEN          : 1;
  __REG32 DIV4            : 1;
  __REG32                 : 30;
} __i2cx_i2ctout_bits;

/* I2C DATA Register */
typedef struct {
  __REG32 DATA            : 8;
  __REG32                 : 24;
} __i2cx_i2cdata_bits;

/* Slave address Register Y */
typedef struct {
  __REG32 GCALL           : 1;
  __REG32 SADDR           : 7;
  __REG32                 : 24;
} __i2cx_i2csaddry_bits;

/* Slave address Mask Register Y */
typedef struct {
  __REG32                 : 1;
  __REG32 SAMASK          : 7;
  __REG32                 : 24;
} __i2cx_i2csamasky_bits;

/* PWM Prescaler Register */
typedef struct {
  __REG32 CP01            : 8;
  __REG32 CP23            : 8;
  __REG32 DZ01            : 8;
  __REG32 DZ23            : 8;
} __pwm0_pwmx_pres_bits;

/* PWM Clock Select Register */
typedef struct {
  __REG32 CLKSEL0         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL1         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL2         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL3         : 3;
  __REG32                 : 17;
} __pwm0_pwmx_clksel_bits;

/* PWM Control Register */
typedef struct {
  __REG32 CH0EN           : 1;
  __REG32                 : 1;
  __REG32 CH0INV          : 1;
  __REG32 CH0MOD          : 1;
  __REG32 DZEN01          : 1;
  __REG32 DZEN23          : 1;
  __REG32                 : 2;
  __REG32 CH1EN           : 1;
  __REG32                 : 1;
  __REG32 CH1INV          : 1;
  __REG32 CH1MOD          : 1;
  __REG32                 : 4;
  __REG32 CH2EN           : 1;
  __REG32                 : 1;
  __REG32 CH2INV          : 1;
  __REG32 CH2MOD          : 1;
  __REG32                 : 4;
  __REG32 CH3EN           : 1;
  __REG32                 : 1;
  __REG32 CH3INV          : 1;
  __REG32 CH3MOD          : 1;
  __REG32                 : 4;
} __pwm0_pwmx_ctl_bits;

/* PWM Interrupt Enable Register */
typedef struct {
  __REG32 TMIE0           : 1;
  __REG32 TMIE1           : 1;
  __REG32 TMIE2           : 1;
  __REG32 TMIE3           : 1;
  __REG32                 : 28;
} __pwm0_pwmx_inten_bits;

/* PWM Interrupt Indication Register */
typedef struct {
  __REG32 TMINT0          : 1;
  __REG32 TMINT1          : 1;
  __REG32 TMINT2          : 1;
  __REG32 TMINT3          : 1;
  __REG32 Duty0Syncflag   : 1;
  __REG32 Duty1Syncflag   : 1;
  __REG32 Duty2Syncflag   : 1;
  __REG32 Duty3Syncflag   : 1;
  __REG32 PresSyncFlag    : 1;
  __REG32                 : 23;
} __pwm0_pwmx_intsts_bits;

/* PWM Output Enable for PWM0~PWM3 */
typedef struct {
  __REG32 CH0_OE          : 1;
  __REG32 CH1_OE          : 1;
  __REG32 CH2_OE          : 1;
  __REG32 CH3_OE          : 1;
  __REG32                 : 28;
} __pwm0_pwmx_oe_bits;

/* PWM Counter/Comparator Register 0 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm0_pwmx_duty0_bits;

/* PWM Data Register 0 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm0_pwmx_data0_bits;

/* PWM Counter/Comparator Register 1 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm0_pwmx_duty1_bits;

/* PWM Data Register 1 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm0_pwmx_data1_bits;

/* PWM Counter/Comparator Register 2 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm0_pwmx_duty2_bits;

/* PWM Data Register 2 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm0_pwmx_data2_bits;

/* PWM Counter/Comparator Register 3 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm0_pwmx_duty3_bits;

/* PWM Data Register 3 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm0_pwmx_data3_bits;

/* Capture Control Register */
typedef struct {
  __REG32 INV0            : 1;
  __REG32 CAPCH0EN        : 1;
  __REG32 CAPCH0PADEN     : 1;
  __REG32 CH0PDMAEN       : 1;
  __REG32 PDMACAPMOD0     : 2;
  __REG32 CAPRELOADREN0   : 1;
  __REG32 CAPRELOADFEN0   : 1;
  __REG32 INV1            : 1;
  __REG32 CAPCH1EN        : 1;
  __REG32 CAPCH1PADEN     : 1;
  __REG32                 : 1;
  __REG32 CH0RFORDER      : 1;
  __REG32 CH01CASK        : 1;
  __REG32 CAPRELOADREN1   : 1;
  __REG32 CAPRELOADFEN1   : 1;
  __REG32 INV2            : 1;
  __REG32 CAPCH2EN        : 1;
  __REG32 CAPCH2PADEN     : 1;
  __REG32 CH2PDMAEN       : 1;
  __REG32 PDMACAPMOD2     : 2;
  __REG32 CAPRELOADREN2   : 1;
  __REG32 CAPRELOADFEN2   : 1;
  __REG32 INV3            : 1;
  __REG32 CAPCH3EN        : 1;
  __REG32 CAPCH3PADEN     : 1;
  __REG32                 : 1;
  __REG32 CH2RFORDER      : 1;
  __REG32 CH23CASK        : 1;
  __REG32 CAPRELOADREN3   : 1;
  __REG32 CAPRELOADFEN3   : 1;
} __pwm0_pwmx_capctl_bits;

/* Capture interrupt enable Register */
typedef struct {
  __REG32 CRL_IE0         : 1;
  __REG32 CFL_IE0         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE1         : 1;
  __REG32 CFL_IE1         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE2         : 1;
  __REG32 CFL_IE2         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE3         : 1;
  __REG32 CFL_IE3         : 1;
  __REG32                 : 6;
} __pwm0_pwmx_capinten_bits;

/* Capture Interrupt Indication Register */
typedef struct {
  __REG32 CAPIF0          : 1;
  __REG32 CRLI0           : 1;
  __REG32 CFLRI0          : 1;
  __REG32 CAPOVR0         : 1;
  __REG32 CAPOVF0         : 1;
  __REG32                 : 3;
  __REG32 CAPIF1          : 1;
  __REG32 CRLI1           : 1;
  __REG32 CFLI1           : 1;
  __REG32 CAPOVR1         : 1;
  __REG32 CAPOVF1         : 1;
  __REG32                 : 3;
  __REG32 CAPIF2          : 1;
  __REG32 CRLI2           : 1;
  __REG32 CFLI2           : 1;
  __REG32 CAPOVR2         : 1;
  __REG32 CAPOVF2         : 1;
  __REG32                 : 3;
  __REG32 CAPIF3          : 1;
  __REG32 CRLI3           : 1;
  __REG32 CFLI3           : 1;
  __REG32 CAPOVR3         : 1;
  __REG32 CAPOVF3         : 1;
  __REG32                 : 3;
} __pwm0_pwmx_capintsts_bits;

/* Capture Rising Latch Register (Channel 0) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm0_pwmx_crl0_bits;

/* Capture Falling Latch Register (Channel 0) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm0_pwmx_cfl0_bits;

/* Capture Rising Latch Register (Channel 1) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm0_pwmx_crl1_bits;

/* Capture Falling Latch Register (Channel 1) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm0_pwmx_cfl1_bits;

/* Capture Rising Latch Register (Channel 2) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm0_pwmx_crl2_bits;

/* Capture Falling Latch Register (Channel 2) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm0_pwmx_cfl2_bits;

/* Capture Rising Latch Register (Channel 3) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm0_pwmx_crl3_bits;

/* Capture Falling Latch Register (Channel 3) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm0_pwmx_cfl3_bits;

/* PDMA channel 0 captured data */
typedef struct {
  __REG32 CapturedData_0_8: 8;
  __REG32 CapturedData_8_16: 8;
  __REG32 CapturedData_16_24: 8;
  __REG32 CapturedData_24_32: 8;
} __pwm0_pwmx_pdmach0_bits;

/* PDMA channel 2 captured data */
typedef struct {
  __REG32 CapturedData_0_8: 8;
  __REG32 CapturedData_8_16: 8;
  __REG32 CapturedData_16_24: 8;
  __REG32 CapturedData_24_32: 8;
} __pwm0_pwmx_pdmach2_bits;

/* PWM Prescaler Register */
typedef struct {
  __REG32 CP01            : 8;
  __REG32 CP23            : 8;
  __REG32 DZ01            : 8;
  __REG32 DZ23            : 8;
} __pwm1_pwmx_pres_bits;

/* PWM Clock Select Register */
typedef struct {
  __REG32 CLKSEL0         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL1         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL2         : 3;
  __REG32                 : 1;
  __REG32 CLKSEL3         : 3;
  __REG32                 : 17;
} __pwm1_pwmx_clksel_bits;

/* PWM Control Register */
typedef struct {
  __REG32 CH0EN           : 1;
  __REG32                 : 1;
  __REG32 CH0INV          : 1;
  __REG32 CH0MOD          : 1;
  __REG32 DZEN01          : 1;
  __REG32 DZEN23          : 1;
  __REG32                 : 2;
  __REG32 CH1EN           : 1;
  __REG32                 : 1;
  __REG32 CH1INV          : 1;
  __REG32 CH1MOD          : 1;
  __REG32                 : 4;
  __REG32 CH2EN           : 1;
  __REG32                 : 1;
  __REG32 CH2INV          : 1;
  __REG32 CH2MOD          : 1;
  __REG32                 : 4;
  __REG32 CH3EN           : 1;
  __REG32                 : 1;
  __REG32 CH3INV          : 1;
  __REG32 CH3MOD          : 1;
  __REG32                 : 4;
} __pwm1_pwmx_ctl_bits;

/* PWM Interrupt Enable Register */
typedef struct {
  __REG32 TMIE0           : 1;
  __REG32 TMIE1           : 1;
  __REG32 TMIE2           : 1;
  __REG32 TMIE3           : 1;
  __REG32                 : 28;
} __pwm1_pwmx_inten_bits;

/* PWM Interrupt Indication Register */
typedef struct {
  __REG32 TMINT0          : 1;
  __REG32 TMINT1          : 1;
  __REG32 TMINT2          : 1;
  __REG32 TMINT3          : 1;
  __REG32 Duty0Syncflag   : 1;
  __REG32 Duty1Syncflag   : 1;
  __REG32 Duty2Syncflag   : 1;
  __REG32 Duty3Syncflag   : 1;
  __REG32 PresSyncFlag    : 1;
  __REG32                 : 23;
} __pwm1_pwmx_intsts_bits;

/* PWM Output Enable for PWM0~PWM3 */
typedef struct {
  __REG32 CH0_OE          : 1;
  __REG32 CH1_OE          : 1;
  __REG32 CH2_OE          : 1;
  __REG32 CH3_OE          : 1;
  __REG32                 : 28;
} __pwm1_pwmx_oe_bits;

/* PWM Counter/Comparator Register 0 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm1_pwmx_duty0_bits;

/* PWM Data Register 0 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm1_pwmx_data0_bits;

/* PWM Counter/Comparator Register 1 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm1_pwmx_duty1_bits;

/* PWM Data Register 1 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm1_pwmx_data1_bits;

/* PWM Counter/Comparator Register 2 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm1_pwmx_duty2_bits;

/* PWM Data Register 2 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm1_pwmx_data2_bits;

/* PWM Counter/Comparator Register 3 */
typedef struct {
  __REG32 CN              : 16;
  __REG32 CM              : 16;
} __pwm1_pwmx_duty3_bits;

/* PWM Data Register 3 */
typedef struct {
  __REG32 PWMx_DATAy_x01_y03: 16;
  __REG32 PWMx_DATAy_x01_y02: 8;
  __REG32                 : 7;
  __REG32 sync            : 1;
} __pwm1_pwmx_data3_bits;

/* Capture Control Register */
typedef struct {
  __REG32 INV0            : 1;
  __REG32 CAPCH0EN        : 1;
  __REG32 CAPCH0PADEN     : 1;
  __REG32 CH0PDMAEN       : 1;
  __REG32 PDMACAPMOD0     : 2;
  __REG32 CAPRELOADREN0   : 1;
  __REG32 CAPRELOADFEN0   : 1;
  __REG32 INV1            : 1;
  __REG32 CAPCH1EN        : 1;
  __REG32 CAPCH1PADEN     : 1;
  __REG32                 : 1;
  __REG32 CH0RFORDER      : 1;
  __REG32 CH01CASK        : 1;
  __REG32 CAPRELOADREN1   : 1;
  __REG32 CAPRELOADFEN1   : 1;
  __REG32 INV2            : 1;
  __REG32 CAPCH2EN        : 1;
  __REG32 CAPCH2PADEN     : 1;
  __REG32 CH2PDMAEN       : 1;
  __REG32 PDMACAPMOD2     : 2;
  __REG32 CAPRELOADREN2   : 1;
  __REG32 CAPRELOADFEN2   : 1;
  __REG32 INV3            : 1;
  __REG32 CAPCH3EN        : 1;
  __REG32 CAPCH3PADEN     : 1;
  __REG32                 : 1;
  __REG32 CH2RFORDER      : 1;
  __REG32 CH23CASK        : 1;
  __REG32 CAPRELOADREN3   : 1;
  __REG32 CAPRELOADFEN3   : 1;
} __pwm1_pwmx_capctl_bits;

/* Capture interrupt enable Register */
typedef struct {
  __REG32 CRL_IE0         : 1;
  __REG32 CFL_IE0         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE1         : 1;
  __REG32 CFL_IE1         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE2         : 1;
  __REG32 CFL_IE2         : 1;
  __REG32                 : 6;
  __REG32 CRL_IE3         : 1;
  __REG32 CFL_IE3         : 1;
  __REG32                 : 6;
} __pwm1_pwmx_capinten_bits;

/* Capture Interrupt Indication Register */
typedef struct {
  __REG32 CAPIF0          : 1;
  __REG32 CRLI0           : 1;
  __REG32 CFLRI0          : 1;
  __REG32 CAPOVR0         : 1;
  __REG32 CAPOVF0         : 1;
  __REG32                 : 3;
  __REG32 CAPIF1          : 1;
  __REG32 CRLI1           : 1;
  __REG32 CFLI1           : 1;
  __REG32 CAPOVR1         : 1;
  __REG32 CAPOVF1         : 1;
  __REG32                 : 3;
  __REG32 CAPIF2          : 1;
  __REG32 CRLI2           : 1;
  __REG32 CFLI2           : 1;
  __REG32 CAPOVR2         : 1;
  __REG32 CAPOVF2         : 1;
  __REG32                 : 3;
  __REG32 CAPIF3          : 1;
  __REG32 CRLI3           : 1;
  __REG32 CFLI3           : 1;
  __REG32 CAPOVR3         : 1;
  __REG32 CAPOVF3         : 1;
  __REG32                 : 3;
} __pwm1_pwmx_capintsts_bits;

/* Capture Rising Latch Register (Channel 0) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm1_pwmx_crl0_bits;

/* Capture Falling Latch Register (Channel 0) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm1_pwmx_cfl0_bits;

/* Capture Rising Latch Register (Channel 1) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm1_pwmx_crl1_bits;

/* Capture Falling Latch Register (Channel 1) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm1_pwmx_cfl1_bits;

/* Capture Rising Latch Register (Channel 2) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm1_pwmx_crl2_bits;

/* Capture Falling Latch Register (Channel 2) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm1_pwmx_cfl2_bits;

/* Capture Rising Latch Register (Channel 3) */
typedef struct {
  __REG32 CRL_0_15        : 16;
  __REG32 CRL_16_31       : 16;
} __pwm1_pwmx_crl3_bits;

/* Capture Falling Latch Register (Channel 3) */
typedef struct {
  __REG32 CFL             : 16;
  __REG32 CFL_16_31       : 16;
} __pwm1_pwmx_cfl3_bits;

/* PDMA channel 0 captured data */
typedef struct {
  __REG32 CapturedData_0_8: 8;
  __REG32 CapturedData_8_16: 8;
  __REG32 CapturedData_16_24: 8;
  __REG32 CapturedData_24_32: 8;
} __pwm1_pwmx_pdmach0_bits;

/* PDMA channel 2 captured data */
typedef struct {
  __REG32 CapturedData_0_8: 8;
  __REG32 CapturedData_8_16: 8;
  __REG32 CapturedData_16_24: 8;
  __REG32 CapturedData_24_32: 8;
} __pwm1_pwmx_pdmach2_bits;

/* RTC Initiation Register */
typedef struct {
  __REG32 INIR0_OR_ACTIVE : 1;
  __REG32 INIR            : 31;
} __rtc_rtc_inir_bits;

/* RTC Access Enable Register */
typedef struct {
  __REG32 AER             : 16;
  __REG32 ENF             : 1;
  __REG32                 : 15;
} __rtc_rtc_aer_bits;

/* RTC Frequency Compensation Register */
typedef struct {
  __REG32 FRACTION        : 6;
  __REG32                 : 2;
  __REG32 INTEGER         : 4;
  __REG32                 : 20;
} __rtc_rtc_fcr_bits;

/* Time Loading Register */
typedef struct {
  __REG32 _1SEC           : 4;
  __REG32 _10SEC          : 3;
  __REG32                 : 1;
  __REG32 _1MIN           : 4;
  __REG32 _10MIN          : 3;
  __REG32                 : 1;
  __REG32 _1HR            : 4;
  __REG32 _10HR           : 2;
  __REG32                 : 10;
} __rtc_rtc_tlr_bits;

/* Calendar Loading Register */
typedef struct {
  __REG32 _1DAY           : 4;
  __REG32 _10DAY          : 2;
  __REG32                 : 2;
  __REG32 _1MON           : 4;
  __REG32 _10MON          : 1;
  __REG32                 : 3;
  __REG32 _1YEAR          : 4;
  __REG32 _10YEAR         : 4;
  __REG32                 : 8;
} __rtc_rtc_clr_bits;

/* Time Scale Selection Register */
typedef struct {
  __REG32 _24H_12H        : 1;
  __REG32                 : 31;
} __rtc_rtc_tssr_bits;

/* Day of the Week Register */
typedef struct {
  __REG32 DWR             : 3;
  __REG32                 : 29;
} __rtc_rtc_dwr_bits;

/* Time Alarm Register */
typedef struct {
  __REG32 _1SEC           : 4;
  __REG32 _10SEC          : 3;
  __REG32                 : 1;
  __REG32 _1MIN           : 4;
  __REG32 _10MIN          : 3;
  __REG32                 : 1;
  __REG32 _1HR            : 4;
  __REG32 _10HR           : 2;
  __REG32                 : 10;
} __rtc_rtc_tar_bits;

/* Calendar Alarm Register */
typedef struct {
  __REG32 _1DAY           : 4;
  __REG32 _10DAY          : 2;
  __REG32                 : 2;
  __REG32 _1MON           : 4;
  __REG32 _10MON          : 1;
  __REG32                 : 3;
  __REG32 _1YEAR          : 4;
  __REG32 _10YEAR         : 4;
  __REG32                 : 8;
} __rtc_rtc_car_bits;

/* RTC Leap year Indicator Register */
typedef struct {
  __REG32 LIR             : 1;
  __REG32                 : 31;
} __rtc_rtc_lir_bits;

/* RTC Interrupt Enable Register */
typedef struct {
  __REG32 AIER            : 1;
  __REG32 TIER            : 1;
  __REG32 SNOOPIER        : 1;
  __REG32                 : 29;
} __rtc_rtc_rier_bits;

/* RTC Interrupt Indicator Register */
typedef struct {
  __REG32 AIS             : 1;
  __REG32 TIS             : 1;
  __REG32 SNOOPIS         : 1;
  __REG32                 : 29;
} __rtc_rtc_riir_bits;

/* RTC Time Tick Register */
typedef struct {
  __REG32 TTR             : 3;
  __REG32 TWKE            : 1;
  __REG32                 : 28;
} __rtc_rtc_ttr_bits;

/* RTC Spare Functional Control Register */
typedef struct {
  __REG32 SNOOPEN         : 1;
  __REG32 SNOOPEDGE       : 1;
  __REG32                 : 5;
  __REG32 SPRRDY          : 1;
  __REG32                 : 24;
} __rtc_rtc_sprctl_bits;

/* SPI Control Register */
typedef struct {
  __REG32 GO_BUSY         : 1;
  __REG32 RX_NEG          : 1;
  __REG32 TX_NEG          : 1;
  __REG32 TX_BIT_LEN      : 5;
  __REG32 TX_NUM          : 2;
  __REG32 LSB             : 1;
  __REG32 CLKP            : 1;
  __REG32 SP_CYCLE        : 4;
  __REG32                 : 1;
  __REG32 INTEN           : 1;
  __REG32 SLAVE           : 1;
  __REG32 REORDER         : 2;
  __REG32 FIFOM           : 1;
  __REG32 TWOB            : 1;
  __REG32 VARCLK_EN       : 1;
  __REG32 SCLK_DLY        : 3;
  __REG32                 : 4;
  __REG32 WKEUP_EN        : 1;
} __spix_spi_ctl_bits;

/* SPI Status Register */
typedef struct {
  __REG32 RX_EMPTY        : 1;
  __REG32 RX_FULL         : 1;
  __REG32 TX_EMPTY        : 1;
  __REG32 TX_FULL         : 1;
  __REG32 LTRIG_FLAG      : 1;
  __REG32                 : 1;
  __REG32 SLV_START_INTSTS: 1;
  __REG32 INTSTS          : 1;
  __REG32                 : 16;
  __REG32                 : 8;
} __spix_spi_status_bits;

/* Serial Clock Divider Register */
typedef struct {
  __REG32 DIVIDER1        : 16;
  __REG32 DIVIDER2        : 16;
} __spix_spi_clkdiv_bits;

/* Slave Select Register */
typedef struct {
  __REG32 SSR             : 2;
  __REG32 SS_LVL          : 1;
  __REG32 AUTOSS          : 1;
  __REG32 SS_LTRIG        : 1;
  __REG32 NOSLVSEL        : 1;
  __REG32                 : 2;
  __REG32 SLV_ABORT       : 1;
  __REG32 SSTA_INTEN      : 1;
  __REG32                 : 22;
} __spix_spi_ssr_bits;

/* SPI PDMA Control Register */
typedef struct {
  __REG32 TX_DMA_EN       : 1;
  __REG32 RX_DMA_EN       : 1;
  __REG32 PDMA_RST        : 1;
  __REG32                 : 29;
} __spix_spi_pdma_bits;

/* SPI FIFO Counter Clear Control Register */
typedef struct {
  __REG32 RX_CLR          : 1;
  __REG32 TX_CLR          : 1;
  __REG32                 : 30;
} __spix_spi_ffclr_bits;

/* Timer x Channel 0 Control Register */
typedef struct {
  __REG32 TMR_EN          : 1;
  __REG32 SW_RST          : 1;
  __REG32 WAKE_EN         : 1;
  __REG32 DBGACK_EN       : 1;
  __REG32 MODE_SEL        : 2;
  __REG32                 : 1;
  __REG32 TMR_ACT         : 1;
  __REG32 ADC_TEEN        : 1;
  __REG32 DAC_TEEN        : 1;
  __REG32 PDMA_TEEN       : 1;
  __REG32 CAP_TRG_EN      : 1;
  __REG32 EVENT_EN        : 1;
  __REG32 EVENT_EDGE      : 1;
  __REG32 EVNT_DEB_EN     : 1;
  __REG32                 : 1;
  __REG32 TCAP_EN         : 1;
  __REG32 TCAP_MODE       : 1;
  __REG32 TCAP_EDGE       : 2;
  __REG32 CAP_CNT_MOD     : 1;
  __REG32                 : 1;
  __REG32 CAP_DEB_EN      : 1;
  __REG32                 : 1;
  __REG32 INTR_TRG_EN     : 1;
  __REG32                 : 7;
} __tmr0_tmrx_ctl0_bits;

/* Timer x Channel 0 Pre-Scale Counter Register */
typedef struct {
  __REG32 PRESCALE_CNT    : 8;
  __REG32                 : 24;
} __tmr0_tmrx_precnt0_bits;

/* Timer x Channel 0 Compare Register */
typedef struct {
  __REG32 TMR_CMP         : 25;
  __REG32                 : 7;
} __tmr0_tmrx_cmpr0_bits;

/* Timer x Channel 0 Interrupt Enable Register */
typedef struct {
  __REG32 TMR_IE          : 1;
  __REG32 TCAP_IE         : 1;
  __REG32                 : 30;
} __tmr0_tmrx_ier0_bits;

/* Timer x Channel 0 Interrupt Status Register */
typedef struct {
  __REG32 TMR_IS          : 1;
  __REG32 TCAP_IS         : 1;
  __REG32                 : 2;
  __REG32 TMR_Wake_STS    : 1;
  __REG32 NCAP_DET_STS    : 1;
  __REG32                 : 26;
} __tmr0_tmrx_isr0_bits;

/* Timer x Channel 0 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr0_tmrx_dr0_bits;

/* Timer x Channel 0 Capture Data Register */
typedef struct {
  __REG32 CAP             : 24;
  __REG32                 : 8;
} __tmr0_tmrx_tcap0_bits;

/* Timer x Channel 1 Control Register */
typedef struct {
  __REG32 TMR_EN          : 1;
  __REG32 SW_RST          : 1;
  __REG32 WAKE_EN         : 1;
  __REG32 DBGACK_EN       : 1;
  __REG32 MODE_SEL        : 2;
  __REG32                 : 1;
  __REG32 TMR_ACT         : 1;
  __REG32 ADC_TEEN        : 1;
  __REG32 DAC_TEEN        : 1;
  __REG32 PDMA_TEEN       : 1;
  __REG32 CAP_TRG_EN      : 1;
  __REG32 EVENT_EN        : 1;
  __REG32 EVENT_EDGE      : 1;
  __REG32 EVNT_DEB_EN     : 1;
  __REG32                 : 1;
  __REG32 TCAP_EN         : 1;
  __REG32 TCAP_MODE       : 1;
  __REG32 TCAP_EDGE       : 2;
  __REG32 CAP_CNT_MOD     : 1;
  __REG32                 : 1;
  __REG32 CAP_DEB_EN      : 1;
  __REG32                 : 1;
  __REG32 INTR_TRG_EN     : 1;
  __REG32                 : 7;
} __tmr0_tmrx_ctl1_bits;

/* Timer x Channel 1 Pre-Scale Counter Register */
typedef struct {
  __REG32 PRESCALE_CNT    : 8;
  __REG32                 : 24;
} __tmr0_tmrx_precnt1_bits;

/* Timer x Channel 1 Compare Register */
typedef struct {
  __REG32 TMR_CMP         : 25;
  __REG32                 : 7;
} __tmr0_tmrx_cmpr1_bits;

/* Timer x Channel 1 Interrupt Enable Register */
typedef struct {
  __REG32 TMR_IE          : 1;
  __REG32 TCAP_IE         : 1;
  __REG32                 : 30;
} __tmr0_tnrx_ier1_bits;

/* Timer x Channel 1 Interrupt Status Register */
typedef struct {
  __REG32 TMR_IS          : 1;
  __REG32 TCAP_IS         : 1;
  __REG32                 : 2;
  __REG32 TMR_Wake_STS    : 1;
  __REG32 NCAP_DET_STS    : 1;
  __REG32                 : 26;
} __tmr0_tmrx_isr1_bits;

/* Timer x Channel 1 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr0_tmrx_dr1_bits;

/* Timer x Channel 1 Capture Data Register */
typedef struct {
  __REG32 CAP             : 24;
  __REG32                 : 8;
} __tmr0_tmrx_tcap1_bits;

/* GPIO Port A Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpa_shadow_bits;

/* GPIO Port B Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpb_shadow_bits;

/* GPIO Port C Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpc_shadow_bits;

/* GPIO Port D Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpd_shadow_bits;

/* GPIO Port E Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpe_shadow_bits;

/* GPIO Port F Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr0_gpf_shadow_bits;

/* Timer x Channel 0 Control Register */
typedef struct {
  __REG32 TMR_EN          : 1;
  __REG32 SW_RST          : 1;
  __REG32 WAKE_EN         : 1;
  __REG32 DBGACK_EN       : 1;
  __REG32 MODE_SEL        : 2;
  __REG32                 : 1;
  __REG32 TMR_ACT         : 1;
  __REG32 ADC_TEEN        : 1;
  __REG32 DAC_TEEN        : 1;
  __REG32 PDMA_TEEN       : 1;
  __REG32 CAP_TRG_EN      : 1;
  __REG32 EVENT_EN        : 1;
  __REG32 EVENT_EDGE      : 1;
  __REG32 EVNT_DEB_EN     : 1;
  __REG32                 : 1;
  __REG32 TCAP_EN         : 1;
  __REG32 TCAP_MODE       : 1;
  __REG32 TCAP_EDGE       : 2;
  __REG32 CAP_CNT_MOD     : 1;
  __REG32                 : 1;
  __REG32 CAP_DEB_EN      : 1;
  __REG32                 : 1;
  __REG32 INTR_TRG_EN     : 1;
  __REG32                 : 7;
} __tmr1_tmrx_ctl0_bits;

/* Timer x Channel 0 Pre-Scale Counter Register */
typedef struct {
  __REG32 PRESCALE_CNT    : 8;
  __REG32                 : 24;
} __tmr1_tmrx_precnt0_bits;

/* Timer x Channel 0 Compare Register */
typedef struct {
  __REG32 TMR_CMP         : 25;
  __REG32                 : 7;
} __tmr1_tmrx_cmpr0_bits;

/* Timer x Channel 0 Interrupt Enable Register */
typedef struct {
  __REG32 TMR_IE          : 1;
  __REG32 TCAP_IE         : 1;
  __REG32                 : 30;
} __tmr1_tmrx_ier0_bits;

/* Timer x Channel 0 Interrupt Status Register */
typedef struct {
  __REG32 TMR_IS          : 1;
  __REG32 TCAP_IS         : 1;
  __REG32                 : 2;
  __REG32 TMR_Wake_STS    : 1;
  __REG32 NCAP_DET_STS    : 1;
  __REG32                 : 26;
} __tmr1_tmrx_isr0_bits;

/* Timer x Channel 0 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr1_tmrx_dr0_bits;

/* Timer x Channel 0 Capture Data Register */
typedef struct {
  __REG32 CAP             : 24;
  __REG32                 : 8;
} __tmr1_tmrx_tcap0_bits;

/* Timer x Channel 1 Control Register */
typedef struct {
  __REG32 TMR_EN          : 1;
  __REG32 SW_RST          : 1;
  __REG32 WAKE_EN         : 1;
  __REG32 DBGACK_EN       : 1;
  __REG32 MODE_SEL        : 2;
  __REG32                 : 1;
  __REG32 TMR_ACT         : 1;
  __REG32 ADC_TEEN        : 1;
  __REG32 DAC_TEEN        : 1;
  __REG32 PDMA_TEEN       : 1;
  __REG32 CAP_TRG_EN      : 1;
  __REG32 EVENT_EN        : 1;
  __REG32 EVENT_EDGE      : 1;
  __REG32 EVNT_DEB_EN     : 1;
  __REG32                 : 1;
  __REG32 TCAP_EN         : 1;
  __REG32 TCAP_MODE       : 1;
  __REG32 TCAP_EDGE       : 2;
  __REG32 CAP_CNT_MOD     : 1;
  __REG32                 : 1;
  __REG32 CAP_DEB_EN      : 1;
  __REG32                 : 1;
  __REG32 INTR_TRG_EN     : 1;
  __REG32                 : 7;
} __tmr1_tmrx_ctl1_bits;

/* Timer x Channel 1 Pre-Scale Counter Register */
typedef struct {
  __REG32 PRESCALE_CNT    : 8;
  __REG32                 : 24;
} __tmr1_tmrx_precnt1_bits;

/* Timer x Channel 1 Compare Register */
typedef struct {
  __REG32 TMR_CMP         : 25;
  __REG32                 : 7;
} __tmr1_tmrx_cmpr1_bits;

/* Timer x Channel 1 Interrupt Enable Register */
typedef struct {
  __REG32 TMR_IE          : 1;
  __REG32 TCAP_IE         : 1;
  __REG32                 : 30;
} __tmr1_tnrx_ier1_bits;

/* Timer x Channel 1 Interrupt Status Register */
typedef struct {
  __REG32 TMR_IS          : 1;
  __REG32 TCAP_IS         : 1;
  __REG32                 : 2;
  __REG32 TMR_Wake_STS    : 1;
  __REG32 NCAP_DET_STS    : 1;
  __REG32                 : 26;
} __tmr1_tmrx_isr1_bits;

/* Timer x Channel 1 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr1_tmrx_dr1_bits;

/* Timer x Channel 1 Capture Data Register */
typedef struct {
  __REG32 CAP             : 24;
  __REG32                 : 8;
} __tmr1_tmrx_tcap1_bits;

/* GPIO Port A Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpa_shadow_bits;

/* GPIO Port B Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpb_shadow_bits;

/* GPIO Port C Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpc_shadow_bits;

/* GPIO Port D Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpd_shadow_bits;

/* GPIO Port E Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpe_shadow_bits;

/* GPIO Port F Pin Value Shadow Register */
typedef struct {
  __REG32 PIN_0_15        : 16;
  __REG32                 : 16;
} __tmr1_gpf_shadow_bits;

/* Watchdog Timer Control Register */
typedef struct {
  __REG32 WTR             : 1;
  __REG32 WTRE            : 1;
  __REG32                 : 2;
  __REG32 WTWKE           : 1;
  __REG32                 : 2;
  __REG32 WTE             : 1;
  __REG32                 : 24;
} __wdt_wdt_ctl_bits;

/* Watchdog Timer Interrupt Enable Register */
typedef struct {
  __REG32 WDT_IE          : 1;
  __REG32                 : 31;
} __wdt_wdt_ie_bits;

/* Watchdog Timer Interrupt Status Register */
typedef struct {
  __REG32 WDT_IS          : 1;
  __REG32 WDT_RST_IS      : 1;
  __REG32 WDT_WAKE_IS     : 1;
  __REG32                 : 29;
} __wdt_wdt_isr_bits;

/* Receive Buffer Register (UART0/1_UARTx_RBR) */
/* Transmit Holding Register (UART0/1_UARTx_THR) */
typedef union {
  /* UART0_UARTx_RBR */
  /* UART1_UARTx_RBR */
  struct {
    __REG32 RBR               : 8;
    __REG32                   :24;
  };
  /* UART0_UARTx_THR */
  /* UART1_UARTx_THR */
  struct {
    __REG32 THR               : 8;
    __REG32                   :24;  
  };
  
  struct {
    __REG32 DATA              : 8;
    __REG32                   :24;  
  };
} __uart_uartx_rbr_thr_bits; 

/* UART Control State Register. */
typedef struct {
  __REG32 RX_RST          : 1;
  __REG32 TX_RST          : 1;
  __REG32 RX_DIS          : 1;
  __REG32 TX_DIS          : 1;
  __REG32 AUTO_RTS_EN     : 1;
  __REG32 AUTO_CTS_EN     : 1;
  __REG32 DMA_RX_EN       : 1;
  __REG32 DMA_TX_EN       : 1;
  __REG32 WAKE_CTS_EN     : 1;
  __REG32 WAKE_DATA_EN    : 1;
  __REG32                 : 2;
  __REG32 ABAUD_EN        : 1;
  __REG32                 : 19;
} __uart_uartx_ctl_bits;

/* UART Transfer Line Control Register. */
typedef struct {
  __REG32 DATA_LEN        : 2;
  __REG32 NSB             : 1;
  __REG32 PBE             : 1;
  __REG32 EPE             : 1;
  __REG32 SPE             : 1;
  __REG32 BCB             : 1;
  __REG32                 : 1;
  __REG32 RFITL           : 2;
  __REG32                 : 2;
  __REG32 RTS_TRI_LEV     : 2;
  __REG32                 : 18;
} __uart_uartx_tlctl_bits;

/* UART Interrupt Enable Register. */
typedef struct {
  __REG32 RDA_IE          : 1;
  __REG32 THRE_IE         : 1;
  __REG32 RLS_IE          : 1;
  __REG32 MODEM_IE        : 1;
  __REG32 RTO_IE          : 1;
  __REG32 BUF_ERR_IE      : 1;
  __REG32 WAKE_IE         : 1;
  __REG32 ABAUD_IE        : 1;
  __REG32 LIN_IE          : 1;
  __REG32                 : 23;
} __uart_uartx_ier_bits;

/* UART Interrupt Status Register. */
typedef struct {
  __REG32 RDA_IS          : 1;
  __REG32 THRE_IS         : 1;
  __REG32 RLS_IS          : 1;
  __REG32 MODEM_IS        : 1;
  __REG32 RTO_IS          : 1;
  __REG32 BUF_ERR_IS      : 1;
  __REG32 WAKE_IS         : 1;
  __REG32 ABAUD_IS        : 1;
  __REG32 LIN_IS          : 1;
  __REG32                 : 23;
} __uart_uartx_isr_bits;

/* UART Transfer State Status Register. */
typedef struct {
  __REG32 RS_485_ADDET_F  : 1;
  __REG32 ABAUD_F         : 1;
  __REG32 ABAUD_TOUT_F    : 1;
  __REG32 LIN_TX_F        : 1;
  __REG32 LIN_RX_F        : 1;
  __REG32 BIT_ERR_F       : 1;
  __REG32                 : 2;
  __REG32 LIN_RX_SYNC_ERR_F: 1;
  __REG32                 : 23;
} __uart_uartx_trsr_bits;

/* UART FIFO State Status Register. */
typedef struct {
  __REG32 RX_OVER_F       : 1;
  __REG32 RX_EMPTY_F      : 1;
  __REG32 RX_FULL_F       : 1;
  __REG32                 : 1;
  __REG32 PE_F            : 1;
  __REG32 FE_F            : 1;
  __REG32 BI_F            : 1;
  __REG32                 : 1;
  __REG32 TX_OVER_F       : 1;
  __REG32 TX_EMPTY_F      : 1;
  __REG32 TX_FULL_F       : 1;
  __REG32 TE_F            : 1;
  __REG32                 : 4;
  __REG32 RX_POINTER_F    : 4;
  __REG32                 : 4;
  __REG32 TX_POINTER_F    : 4;
  __REG32                 : 4;
} __uart_uartx_fsr_bits;

/* UART Modem State Status Register. */
typedef struct {
  __REG32 LEV_RTS         : 1;
  __REG32 RTS_ST          : 1;
  __REG32                 : 6;
  __REG32 LEV_CTS         : 1;
  __REG32                 : 8;
  __REG32 CTS_ST          : 1;
  __REG32 DCT_F           : 1;
  __REG32                 : 13;
} __uart_uartx_mcsr_bits;

/* UART Time- Out Control State Register. */
typedef struct {
  __REG32 TOIC            : 9;
  __REG32                 : 7;
  __REG32 DLY             : 8;
  __REG32                 : 8;
} __uart_uartx_tmctl_bits;

/* UART Baud Rate Divisor Register */
typedef struct {
  __REG32 BRD             : 16;
  __REG32                 : 13;
  __REG32 DIV_16_EN       : 1;
  __REG32                 : 2;
} __uart_uartx_baud_bits;

/* UART Wake- Up Register */
typedef struct {
  __REG32 WAKE_CNT        : 15;
  __REG32                 : 15;
  __REG32 PE_OLD          : 1;
  __REG32 UNLOCK          : 1;
} __uart_uartx_wake_bits;

/* UART IrDA Control Register. */
typedef struct {
  __REG32                 : 1;
  __REG32 TX_SELECT       : 1;
  __REG32 LB              : 1;
  __REG32                 : 2;
  __REG32 INV_TX          : 1;
  __REG32 INV_RX          : 1;
  __REG32                 : 25;
} __uart_uartx_ircr_bits;

/* UART Alternate Control State Register. */
typedef struct {
  __REG32 LIN_TX_BCNT     : 3;
  __REG32                 : 1;
  __REG32 LIN_HEAD_SEL    : 2;
  __REG32 LIN_RX_EN       : 1;
  __REG32 LIN_TX_EN       : 1;
  __REG32 Bit_ERR_EN      : 1;
  __REG32                 : 7;
  __REG32 RS_485_NMM      : 1;
  __REG32 RS_485_AAD      : 1;
  __REG32 RS_485_AUD      : 1;
  __REG32 RS_485_ADD_EN   : 1;
  __REG32                 : 4;
  __REG32 ADDR_PID_MATCH  : 8;
} __uart_uartx_alt_csr_bits;

/* UART Function Select Register. */
typedef struct {
  __REG32 FUN_SEL         : 2;
  __REG32                 : 30;
} __uart_uartx_fun_sel_bits;

/* I2S Control Register */
typedef struct {
  __REG32 I2SEN           : 1;
  __REG32 TXEN            : 1;
  __REG32 RXEN            : 1;
  __REG32 MUTE            : 1;
  __REG32 WORDWIDTH_0_1   : 2;
  __REG32 MONO            : 1;
  __REG32 FORMAT          : 1;
  __REG32 SLAVE           : 1;
  __REG32 TXTH_0_2        : 3;
  __REG32 RXTH_0_2        : 3;
  __REG32 MCLKEN          : 1;
  __REG32 RCHZCEN         : 1;
  __REG32 LCHZCEN         : 1;
  __REG32 CLR_TXFIFO      : 1;
  __REG32 CLR_RXFIFO      : 1;
  __REG32 TXDMA           : 1;
  __REG32 RXDMA           : 1;
  __REG32                 : 10;
} __i2s_i2s_ctrl_bits;

/* I2S Clock Divider Register */
typedef struct {
  __REG32 MCLK_DIV        : 3;
  __REG32                 : 5;
  __REG32 BCLK_DIV        : 8;
  __REG32                 : 16;
} __i2s_i2s_clkdiv_bits;

/* I2S Interrupt Enable Register */
typedef struct {
  __REG32 RXUDFIE         : 1;
  __REG32 RXOVFIE         : 1;
  __REG32 RXTHIE          : 1;
  __REG32                 : 5;
  __REG32 TXUDFIE         : 1;
  __REG32 TXOVFIE         : 1;
  __REG32 TXTHIE          : 1;
  __REG32 RZCIE           : 1;
  __REG32 LZCIE           : 1;
  __REG32                 : 19;
} __i2s_i2s_inten_bits;

/* I2S Status Register */
typedef struct {
  __REG32 I2SINT          : 1;
  __REG32 I2SRXINT        : 1;
  __REG32 I2STXINT        : 1;
  __REG32 RIGHT           : 1;
  __REG32                 : 4;
  __REG32 RXUDF           : 1;
  __REG32 RXOVF           : 1;
  __REG32 RXTHF           : 1;
  __REG32 RXFULL          : 1;
  __REG32 RXEMPTY         : 1;
  __REG32                 : 3;
  __REG32 TXUDF           : 1;
  __REG32 TXOVF           : 1;
  __REG32 TXTHF           : 1;
  __REG32 TXFULL          : 1;
  __REG32 TXEMPTY         : 1;
  __REG32 TXBUSY          : 1;
  __REG32 RZCF            : 1;
  __REG32 LZCF            : 1;
  __REG32 RX_LEVEL        : 4;
  __REG32 TX_LEVEL        : 4;
} __i2s_i2s_status_bits;

/* A/D Data Register X */
typedef struct {
  __REG32 RSLT            : 12;
  __REG32                 : 20;
} __adc_adc_resultx_bits;

/* A/D Control Register */
typedef struct {
  __REG32 ADEN            : 1;
  __REG32 ADIE            : 1;
  __REG32 ADMD            : 2;
  __REG32 TRGS            : 2;
  __REG32 TRGCOND         : 2;
  __REG32 TRGE            : 1;
  __REG32 PTEN            : 1;
  __REG32                 : 1;
  __REG32 ADST            : 1;
  __REG32 TMSEL           : 2;
  __REG32                 : 1;
  __REG32 TMTRGMOD        : 1;
  __REG32 REFSEL          : 2;
  __REG32                 : 14;
} __adc_adcr_bits;

/* A/D Channel Enable Register */
typedef struct {
  __REG32 CHEN0           : 1;
  __REG32 CHEN1           : 1;
  __REG32 CHEN2           : 1;
  __REG32 CHEN3           : 1;
  __REG32 CHEN4           : 1;
  __REG32 CHEN5           : 1;
  __REG32 CHEN6           : 1;
  __REG32 CHEN7           : 1;
  __REG32 CHEN8           : 1;
  __REG32 CHEN9           : 1;
  __REG32 CHEN10          : 1;
  __REG32 CH10SEL         : 2;
  __REG32                 : 19;
} __adc_adcher_bits;

/* A/D Compare Register 0 */
typedef struct {
  __REG32 CMPEN           : 1;
  __REG32 CMPIE           : 1;
  __REG32 CMPCOND         : 1;
  __REG32 CMPCH           : 4;
  __REG32                 : 1;
  __REG32 CMPMATCNT       : 4;
  __REG32                 : 4;
  __REG32 CMPD            : 12;
  __REG32                 : 4;
} __adc_adcmpr0_bits;

/* A/D Compare Register 1 */
typedef struct {
  __REG32 CMPEN           : 1;
  __REG32 CMPIE           : 1;
  __REG32 CMPCOND         : 1;
  __REG32 CMPCH           : 4;
  __REG32                 : 1;
  __REG32 CMPMATCNT       : 4;
  __REG32                 : 4;
  __REG32 CMPD            : 12;
  __REG32                 : 4;
} __adc_adcmpr1_bits;

/* A/D Status Register */
typedef struct {
  __REG32 ADF             : 1;
  __REG32 CMPF0           : 1;
  __REG32 CMPF1           : 1;
  __REG32 BUSY            : 1;
  __REG32 CHANNEL         : 4;
  __REG32 VALID           : 9;
  __REG32                 : 3;
  __REG32 OVERRUN         : 9;
  __REG32                 : 3;
} __adc_adsr_bits;

/* A/D FPGA Control Register+ */
typedef struct {
  __REG32 FCR_0_4         : 5;
  __REG32 Chsel_0_2       : 3;
  __REG32 FCR_8_11        : 4;
  __REG32                 : 20;
} __adc_adfcr_bits;

/* ADC PDMA current transfer data */
typedef struct {
  __REG32 AD_PDMA         : 12;
  __REG32                 : 20;
} __adc_adpdma_bits;

/* PDMA counter for delay time and PDMA transfer count and ADC start hold counter */
typedef struct {
  __REG32 En2StDelay      : 8;
  __REG32 TMPDMACNT       : 8;
  __REG32 ADCSTHOLDCNT    : 8;
  __REG32                 : 8;
} __adc_adcdelsel_bits;

/* VDMA Control Register */
typedef struct {
  __REG32 VDMACEN         : 1;
  __REG32 SW_RST          : 1;
  __REG32                 : 8;
  __REG32 Stride_EN       : 1;
  __REG32 DIR_SEL         : 1;
  __REG32                 : 11;
  __REG32 TRIG_EN         : 1;
  __REG32                 : 8;
} __vdma_vdma_csr_bits;

/* VDMA Transfer Byte Count Register */
typedef struct {
  __REG32 VDMA_BCR        : 16;
  __REG32                 : 16;
} __vdma_vdma_bcr_bits;

/* VDMA Current Transfer Byte Count Register */
typedef struct {
  __REG32 VDMA_CBCR       : 16;
  __REG32                 : 16;
} __vdma_vdma_cbcr_bits;

/* VDMA Interrupt Enable Register */
typedef struct {
  __REG32 TABORT_IE       : 1;
  __REG32 TD_IE           : 1;
  __REG32                 : 30;
} __vdma_vdma_ier_bits;

/* VDMA Interrupt Status Register */
typedef struct {
  __REG32 TABORT_IS       : 1;
  __REG32 TD_IS           : 1;
  __REG32                 : 13;
  __REG32 Busy            : 1;
  __REG32                 : 15;
  __REG32 INTR            : 1;
} __vdma_vdma_isr_bits;

/* VDMA Source Address Stride Offset Register */
typedef struct {
  __REG32 SASTOBL         : 16;
  __REG32 STBC            : 16;
} __vdma_vdma_sasocr_bits;

/* VDMA Destination Address Stride Offset Register */
typedef struct {
  __REG32 DASTOBL         : 16;
  __REG32                 : 16;
} __vdma_vdma_dasocr_bits;

/* PMAC Control and Status Register */
typedef struct {
  __REG32 PDMACEN         : 1;
  __REG32 SW_RST          : 1;
  __REG32 MODE_SEL        : 2;
  __REG32 SAD_SEL         : 2;
  __REG32 DAD_SEL         : 2;
  __REG32                 : 4;
  __REG32 TO_EN           : 1;
  __REG32                 : 6;
  __REG32 APB_TWS         : 2;
  __REG32                 : 2;
  __REG32 TRIG_EN         : 1;
  __REG32                 : 8;
} __pdma1_pdma_csrx_bits;

/* PDMA Transfer Byte Count Register */
typedef struct {
  __REG32 PDMA_BCR        : 16;
  __REG32                 : 8;
  __REG32                 : 8;
} __pdma1_pdma_bcrx_bits;

/* PDMA Current Byte Count Register */
typedef struct {
  __REG32 PDMA_CBCR       : 24;
  __REG32                 : 8;
} __pdma1_pdma_cbcrx_bits;

/* PDMA Interrupt Enable Control Register */
typedef struct {
  __REG32 TABORT_IE       : 1;
  __REG32 TD_IE           : 1;
  __REG32 WRA_BCR_IE      : 4;
  __REG32 TO_IE           : 1;
  __REG32                 : 25;
} __pdma1_pdma_ierx_bits;

/* PDMA Interrupt Status Register */
typedef struct {
  __REG32 TABORT_IS       : 1;
  __REG32 TD_IS           : 1;
  __REG32 WRA_BCR_IS      : 4;
  __REG32 TO_IS           : 1;
  __REG32                 : 8;
  __REG32 Busy            : 1;
  __REG32                 : 15;
  __REG32 INTR            : 1;
} __pdma1_pdma_isrx_bits;

/* PDMA Timer Count Setting Register */
typedef struct {
  __REG32 PDMA_TCR        : 16;
  __REG32                 : 16;
} __pdma1_pdma_tcrx_bits;

/* PMAC Control and Status Register */
typedef struct {
  __REG32 PDMACEN         : 1;
  __REG32 SW_RST          : 1;
  __REG32 MODE_SEL        : 2;
  __REG32 SAD_SEL         : 2;
  __REG32 DAD_SEL         : 2;
  __REG32                 : 4;
  __REG32 TO_EN           : 1;
  __REG32                 : 6;
  __REG32 APB_TWS         : 2;
  __REG32                 : 2;
  __REG32 TRIG_EN         : 1;
  __REG32                 : 8;
} __pdma2_pdma_csrx_bits;

/* PDMA Transfer Byte Count Register */
typedef struct {
  __REG32 PDMA_BCR        : 16;
  __REG32                 : 8;
  __REG32                 : 8;
} __pdma2_pdma_bcrx_bits;

/* PDMA Current Byte Count Register */
typedef struct {
  __REG32 PDMA_CBCR       : 24;
  __REG32                 : 8;
} __pdma2_pdma_cbcrx_bits;

/* PDMA Interrupt Enable Control Register */
typedef struct {
  __REG32 TABORT_IE       : 1;
  __REG32 TD_IE           : 1;
  __REG32 WRA_BCR_IE      : 4;
  __REG32 TO_IE           : 1;
  __REG32                 : 25;
} __pdma2_pdma_ierx_bits;

/* PDMA Interrupt Status Register */
typedef struct {
  __REG32 TABORT_IS       : 1;
  __REG32 TD_IS           : 1;
  __REG32 WRA_BCR_IS      : 4;
  __REG32 TO_IS           : 1;
  __REG32                 : 8;
  __REG32 Busy            : 1;
  __REG32                 : 15;
  __REG32 INTR            : 1;
} __pdma2_pdma_isrx_bits;

/* PDMA Timer Count Setting Register */
typedef struct {
  __REG32 PDMA_TCR        : 16;
  __REG32                 : 16;
} __pdma2_pdma_tcrx_bits;

/* PMAC Control and Status Register */
typedef struct {
  __REG32 PDMACEN         : 1;
  __REG32 SW_RST          : 1;
  __REG32 MODE_SEL        : 2;
  __REG32 SAD_SEL         : 2;
  __REG32 DAD_SEL         : 2;
  __REG32                 : 4;
  __REG32 TO_EN           : 1;
  __REG32                 : 6;
  __REG32 APB_TWS         : 2;
  __REG32                 : 2;
  __REG32 TRIG_EN         : 1;
  __REG32                 : 8;
} __pdma3_pdma_csrx_bits;

/* PDMA Transfer Byte Count Register */
typedef struct {
  __REG32 PDMA_BCR        : 16;
  __REG32                 : 8;
  __REG32                 : 8;
} __pdma3_pdma_bcrx_bits;

/* PDMA Current Byte Count Register */
typedef struct {
  __REG32 PDMA_CBCR       : 24;
  __REG32                 : 8;
} __pdma3_pdma_cbcrx_bits;

/* PDMA Interrupt Enable Control Register */
typedef struct {
  __REG32 TABORT_IE       : 1;
  __REG32 TD_IE           : 1;
  __REG32 WRA_BCR_IE      : 4;
  __REG32 TO_IE           : 1;
  __REG32                 : 25;
} __pdma3_pdma_ierx_bits;

/* PDMA Interrupt Status Register */
typedef struct {
  __REG32 TABORT_IS       : 1;
  __REG32 TD_IS           : 1;
  __REG32 WRA_BCR_IS      : 4;
  __REG32 TO_IS           : 1;
  __REG32                 : 8;
  __REG32 Busy            : 1;
  __REG32                 : 15;
  __REG32 INTR            : 1;
} __pdma3_pdma_isrx_bits;

/* PDMA Timer Count Setting Register */
typedef struct {
  __REG32 PDMA_TCR        : 16;
  __REG32                 : 16;
} __pdma3_pdma_tcrx_bits;

/* PMAC Control and Status Register */
typedef struct {
  __REG32 PDMACEN         : 1;
  __REG32 SW_RST          : 1;
  __REG32 MODE_SEL        : 2;
  __REG32 SAD_SEL         : 2;
  __REG32 DAD_SEL         : 2;
  __REG32                 : 4;
  __REG32 TO_EN           : 1;
  __REG32                 : 6;
  __REG32 APB_TWS         : 2;
  __REG32                 : 2;
  __REG32 TRIG_EN         : 1;
  __REG32                 : 8;
} __pdma4_pdma_csrx_bits;

/* PDMA Transfer Byte Count Register */
typedef struct {
  __REG32 PDMA_BCR        : 16;
  __REG32                 : 8;
  __REG32                 : 8;
} __pdma4_pdma_bcrx_bits;

/* PDMA Current Byte Count Register */
typedef struct {
  __REG32 PDMA_CBCR       : 24;
  __REG32                 : 8;
} __pdma4_pdma_cbcrx_bits;

/* PDMA Interrupt Enable Control Register */
typedef struct {
  __REG32 TABORT_IE       : 1;
  __REG32 TD_IE           : 1;
  __REG32 WRA_BCR_IE      : 4;
  __REG32 TO_IE           : 1;
  __REG32                 : 25;
} __pdma4_pdma_ierx_bits;

/* PDMA Interrupt Status Register */
typedef struct {
  __REG32 TABORT_IS       : 1;
  __REG32 TD_IS           : 1;
  __REG32 WRA_BCR_IS      : 4;
  __REG32 TO_IS           : 1;
  __REG32                 : 8;
  __REG32 Busy            : 1;
  __REG32                 : 15;
  __REG32 INTR            : 1;
} __pdma4_pdma_isrx_bits;

/* PDMA Timer Count Setting Register */
typedef struct {
  __REG32 PDMA_TCR        : 16;
  __REG32                 : 16;
} __pdma4_pdma_tcrx_bits;

/* PDMA Global Control Register */
typedef struct {
  __REG32 PDMA_RST        : 1;
  __REG32                 : 7;
  __REG32 CLK0_EN         : 1;
  __REG32 CLK1_EN         : 1;
  __REG32 CLK2_EN         : 1;
  __REG32 CLK3_EN         : 1;
  __REG32 CLK4_EN         : 1;
  __REG32                 : 19;
} __pdma_gcr_pdma_gcrcsr_bits;

/* PDMA Service Selection Control Register 0 */
typedef struct {
  __REG32                 : 8;
  __REG32 CH1_SEL         : 5;
  __REG32                 : 3;
  __REG32 CH2_SEL         : 5;
  __REG32                 : 3;
  __REG32 CH3_SEL         : 5;
  __REG32                 : 3;
} __pdma_gcr_pdssr0_bits;

/* PDMA Service Selection Control Register 1 */
typedef struct {
  __REG32 CH4_SEL         : 5;
  __REG32                 : 27;
} __pdma_gcr_pdssr1_bits;

/* PDMA Global Interrupt Register */
typedef struct {
  __REG32 INTR0           : 1;
  __REG32 INTR1           : 1;
  __REG32 INTR2           : 1;
  __REG32 INTR3           : 1;
  __REG32 INTR4           : 1;
  __REG32                 : 26;
  __REG32 INTR            : 1;
} __pdma_gcr_pdma_gcrisr_bits;

/* External Bus Interface General Control Register */
typedef struct {
  __REG32 ExtEN           : 1;
  __REG32 ExtBW16         : 1;
  __REG32                 : 6;
  __REG32 MCLKDIV         : 3;
  __REG32 MCLKEN          : 1;
  __REG32                 : 4;
  __REG32 ExttALE         : 3;
  __REG32                 : 13;
} __ebi_ebicon_bits;

/* External Bus Interface Timing Control Register */
typedef struct {
  __REG32 ExttACC         : 5;
  __REG32                 : 3;
  __REG32 ExttAHD         : 3;
  __REG32                 : 1;
  __REG32 ExtIW2X         : 4;
  __REG32 ExtIR2W         : 4;
  __REG32                 : 4;
  __REG32 ExtIR2R         : 4;
  __REG32                 : 4;
} __ebi_extime_bits;

/* Touch-Key Control Register */
typedef struct {
  __REG32 TK_MUX          : 4;
  __REG32 TK_FREQ         : 2;
  __REG32 EXT_CAP_EN      : 1;
  __REG32                 : 1;
  __REG32 TK_CUR_CTRL     : 4;
  __REG32 TK_START        : 1;
  __REG32 TK_SEN_SEL      : 2;
  __REG32 TK_EN           : 1;
  __REG32                 : 16;
} __tk_tk_ctl_bits;

/* Touch-Key Status Register */
typedef struct {
  __REG32 TK_BUSY         : 1;
  __REG32 TK_DAT_RDY      : 1;
  __REG32 TK_SEN_FAIL     : 1;
  __REG32                 : 1;
  __REG32 SEN_MATCH_LEVEL : 4;
  __REG32                 : 24;
} __tk_tk_stat_bits;

/* Touch-Key Data Register */
typedef struct {
  __REG32 TK_DATA         : 16;
  __REG32                 : 16;
} __tk_tk_data_bits;

/* Touch Key Interrupt Enable Register */
typedef struct {
  __REG32                 : 1;
  __REG32 TK_DAT_RDY_IE   : 1;
  __REG32 TK_SEN_FAIL_IE  : 1;
  __REG32                 : 29;
} __tk_tk_inten_bits;

/* DAC0 control register */
typedef struct {
  __REG32                 : 1;
  __REG32 DACEN           : 1;
  __REG32 DACIE           : 1;
  __REG32                 : 1;
  __REG32 DACLSEL         : 3;
  __REG32                 : 1;
  __REG32 PWONDACTMSEL    : 2;
  __REG32                 : 6;
  __REG32 PWONDACTRANSCNT : 8;
  __REG32 DACPWONSTBCNT   : 8;
} __dac_dac0_ctl_bits;

/* DAC0 data register */
typedef struct {
  __REG32 DAC_Data        : 12;
  __REG32                 : 20;
} __dac_dac0_data_bits;

/* DAC0 status register */
typedef struct {
  __REG32 DACIFG          : 1;
  __REG32 DACSTFG         : 1;
  __REG32 BUSY            : 1;
  __REG32                 : 29;
} __dac_dac0_sts_bits;

/* DAC1 control register */
typedef struct {
  __REG32                 : 1;
  __REG32 DACEN           : 1;
  __REG32 DACIE           : 1;
  __REG32                 : 1;
  __REG32 DACLSEL         : 3;
  __REG32                 : 1;
  __REG32 PWONDACTMSEL    : 2;
  __REG32                 : 6;
  __REG32 PWONDACTRANSCNT : 8;
  __REG32 DACPWONSTBCNT   : 8;
} __dac_dac1_ctl_bits;

/* DAC1 data register */
typedef struct {
  __REG32 DAC_Data        : 12;
  __REG32                 : 20;
} __dac_dac1_data_bits;

/* DAC1 status register */
typedef struct {
  __REG32 DACIFG          : 1;
  __REG32 DACSTFG         : 1;
  __REG32 BUSY            : 1;
  __REG32                 : 29;
} __dac_dac1_sts_bits;

/* DAC01 common control register */
typedef struct {
  __REG32 WAITDACCONV     : 8;
  __REG32 DAC01GRP        : 1;
  __REG32 REFSEL          : 2;
  __REG32                 : 21;
} __dac_dac01_comctl_bits;

/* DAC0 FPGA read DAC0 DATA */
typedef struct {
  __REG32 DACxFPGAData    : 12;
  __REG32                 : 20;
} __dac_dac0_fpga_dat_bits;

/* DAC1 FPGA read DAC1 DATA */
typedef struct {
  __REG32 DACxFPGAData    : 12;
  __REG32                 : 20;
} __dac_dac1_fpga_dat_bits;

/* Receive Buffer Register (SCx_SC_RBR) */
/* Transmit Holding Register (SCx_SC_THR) */
typedef union {
  /* SC0_SC_RBR */
  /* SC1_SC_RBR */
  struct {
    __REG32 RBR               : 8;
    __REG32                   :24;
  };
  /* SC0_SC_THR */
  /* SC1_SC_THR */
  struct {
    __REG32 THR               : 8;
    __REG32                   :24;  
  };
  
  struct {
    __REG32 DATA              : 8;
    __REG32                   :24;  
  };
} __scx_sc_rbr_thr_bits; 

/* SC Control Register */
typedef struct {
  __REG32 SC_CEN          : 1;
  __REG32 DIS_RX          : 1;
  __REG32 DIS_TX          : 1;
  __REG32 AUTO_CON_EN     : 1;
  __REG32 CON_SEL         : 2;
  __REG32 RX_FTRI_LEV     : 2;
  __REG32 BGT             : 5;
  __REG32 TMR_SEL         : 2;
  __REG32 SLEN            : 1;
  __REG32 RX_ERETRY       : 3;
  __REG32 RX_ERETRY_EN    : 1;
  __REG32 TX_ERETRY       : 3;
  __REG32 TX_ERETRY_EN    : 1;
  __REG32 CD_DEB_SEL      : 2;
  __REG32                 : 5;
  __REG32 NDBGACK_EN      : 1;
} __scx_sc_ctl_bits;

/* SC Alternate Control State Register */
typedef struct {
  __REG32 TX_RST          : 1;
  __REG32 RX_RST          : 1;
  __REG32 DACT_EN         : 1;
  __REG32 ACT_EN          : 1;
  __REG32 WARST_EN        : 1;
  __REG32 TMR0_SEN        : 1;
  __REG32 TMR1_SEN        : 1;
  __REG32 TMR2_SEN        : 1;
  __REG32 INIT_SEL        : 2;
  __REG32                 : 2;
  __REG32 RX_BGT_EN       : 1;
  __REG32 TMR0_ATV        : 1;
  __REG32 TMR1_ATV        : 1;
  __REG32 TMR2_ATV        : 1;
  __REG32                 : 16;
} __scx_sc_altctl_bits;

/* SC Extend Guard Time Register. */
typedef struct {
  __REG32 EGT             : 8;
  __REG32                 : 24;
} __scx_sc_egtr_bits;

/* SC Receiver buffer Time- Out Register. */
typedef struct {
  __REG32 RFTM            : 9;
  __REG32                 : 23;
} __scx_sc_rftmr_bits;

/* SC ETU Control Register */
typedef struct {
  __REG32 ETU_RDIV        : 12;
  __REG32                 : 3;
  __REG32 COMPEN_EN       : 1;
  __REG32                 : 16;
} __scx_sc_etucr_bits;

/* SC Interrupt Enable Register */
typedef struct {
  __REG32 RDA_IE          : 1;
  __REG32 TBE_IE          : 1;
  __REG32 TERR_IE         : 1;
  __REG32 TMR0_IE         : 1;
  __REG32 TMR1_IE         : 1;
  __REG32 TMR2_IE         : 1;
  __REG32 BGT_IE          : 1;
  __REG32 CD_IE           : 1;
  __REG32 INIT_IE         : 1;
  __REG32 RTMR_IE         : 1;
  __REG32 ACON_ERR_IE     : 1;
  __REG32                 : 21;
} __scx_sc_ier_bits;

/* SC Interrupt Status Register (Read Only) */
typedef struct {
  __REG32 RDA_IS          : 1;
  __REG32 TBE_IS          : 1;
  __REG32 TERR_IS         : 1;
  __REG32 TMR0_IS         : 1;
  __REG32 TMR1_IS         : 1;
  __REG32 TMR2_IS         : 1;
  __REG32 BGT_IS          : 1;
  __REG32 CD_IS           : 1;
  __REG32 INIT_IS         : 1;
  __REG32 RTMR_IS         : 1;
  __REG32 ACON_ERR_IS     : 1;
  __REG32                 : 21;
} __scx_sc_isr_bits;

/* SC Transfer Status Register (Read Only) */
typedef struct {
  __REG32 RX_OVER_F       : 1;
  __REG32 RX_EMPTY_F      : 1;
  __REG32 RX_FULL_F       : 1;
  __REG32                 : 1;
  __REG32 RX_EPA_F        : 1;
  __REG32 RX_EFR_F        : 1;
  __REG32 RX_EBR_F        : 1;
  __REG32                 : 1;
  __REG32 TX_OVER_F       : 1;
  __REG32 TX_EMPTY_F      : 1;
  __REG32 TX_FULL_F       : 1;
  __REG32                 : 5;
  __REG32 RX_POINT_F      : 2;
  __REG32                 : 3;
  __REG32 RX_REERR        : 1;
  __REG32 RX_OVER_REERR   : 1;
  __REG32 RX_ATV          : 1;
  __REG32 TX_POINT_F      : 2;
  __REG32                 : 3;
  __REG32 TX_REERR        : 1;
  __REG32 TX_OVER_REERR   : 1;
  __REG32 TX_ATV          : 1;
} __scx_sc_trsr_bits;

/* SC Pin Control State Register */
typedef struct {
  __REG32 POW_EN          : 1;
  __REG32 SC_RST          : 1;
  __REG32 CD_REM_F        : 1;
  __REG32 CD_INS_F        : 1;
  __REG32 CD_PIN_ST       : 1;
  __REG32 CLK_STOP_LEV    : 1;
  __REG32 CLK_KEEP        : 1;
  __REG32 ADAC_CD_EN      : 1;
  __REG32 SC_OEN_ST       : 1;
  __REG32 SC_DATA_O       : 1;
  __REG32 CD_LEV          : 1;
  __REG32                 : 5;
  __REG32 SC_DATA_I_ST    : 1;
  __REG32                 : 14;
  __REG32 LOOP_BACK       : 1;
} __scx_sc_pincsr_bits;

/* SC Internal Timer Control Register 0 */
typedef struct {
  __REG32 CNT             : 24;
  __REG32 MODE            : 4;
  __REG32                 : 4;
} __scx_sc_tmr0_bits;

/* SC Internal Timer Control Register 1 */
typedef struct {
  __REG32 CNT             : 24;
  __REG32 MODE            : 4;
  __REG32                 : 4;
} __scx_sc_tmr1_bits;

/* SC Internal Timer Control Register 2 */
typedef struct {
  __REG32 CNT             : 24;
  __REG32 MODE            : 4;
  __REG32                 : 4;
} __scx_sc_tmr2_bits;

/* SC Timer Current Data Register A */
typedef struct {
  __REG32 TDR0            : 24;
  __REG32                 : 8;
} __scx_sc_tdra_bits;

/* SC Timer Current Data Register B */
typedef struct {
  __REG32 TDR1            : 8;
  __REG32 TDR2            : 8;
  __REG32                 : 16;
} __scx_sc_tdrb_bits;

/* ISP Control Register */
typedef struct {
  __REG32 ISPEN           : 1;
  __REG32 BS              : 1;
  __REG32                 : 1;
  __REG32 APUEN           : 1;
  __REG32 CFGUEN          : 1;
  __REG32 LDUEN           : 1;
  __REG32 ISPFF           : 1;
  __REG32 SWRST           : 1;
  __REG32 PT_0_2          : 3;
  __REG32                 : 1;
  __REG32 ET_0_2          : 3;
  __REG32                 : 17;
} __fmc_ispcon_bits;

/* ISP Command Register */
typedef struct {
  __REG32 FOEN_FCEN_FCTRL : 6;
  __REG32                 : 26;
} __fmc_ispcmd_bits;

/* ISP Trigger Register */
typedef struct {
  __REG32 ISPGO           : 1;
  __REG32                 : 31;
} __fmc_isptrg_bits;

/* Flash Access Window Control Register */
typedef struct {
  __REG32                 : 4;
  __REG32 LSPEED          : 1;
  __REG32 FIDLE           : 1;
  __REG32 MSPEED          : 1;
  __REG32 FBEN            : 1;
  __REG32                 : 24;
} __fmc_fatcon_bits;

/* ICP Control Register */
typedef struct {
  __REG32 ICPEN           : 1;
  __REG32                 : 31;
} __fmc_icpcon_bits;

/* ROMMAP Control Register */
typedef struct {
  __REG32 ICPRMP          : 1;
  __REG32                 : 31;
} __fmc_rmpcon_bits;

/* FPGA ICE Download Control Register */
typedef struct {
  __REG32 TargetAP        : 1;
  __REG32 TargetLD        : 1;
  __REG32 TargetCFG       : 1;
  __REG32 TargetMAP       : 1;
  __REG32                 : 28;
} __fmc_icecon_bits;

/* ISP Status Register */
typedef struct {
  __REG32 ISPBUSY         : 1;
  __REG32 CBS             : 2;
  __REG32                 : 3;
  __REG32 ISPFF           : 1;
  __REG32                 : 2;
  __REG32 VECMAP          : 12;
  __REG32                 : 11;
} __fmc_ispsta_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** GCR
 **
 ***************************************************************************/
__IO_REG32(    GCR_PDID,              0x50000000,__READ       );
__IO_REG32_BIT(GCR_RST_SRC,           0x50000004,__READ_WRITE ,__gcr_rst_src_bits);
__IO_REG32_BIT(GCR_IPRST_CTL1,        0x50000008,__READ_WRITE ,__gcr_iprst_ctl1_bits);
__IO_REG32_BIT(GCR_IPRST_CTL2,        0x5000000C,__READ_WRITE ,__gcr_iprst_ctl2_bits);
__IO_REG32_BIT(GCR_CPR,               0x50000010,__READ_WRITE ,__gcr_cpr_bits);
__IO_REG32_BIT(GCR_ITESTCR,           0x50000014,__READ_WRITE ,__gcr_itestcr_bits);
__IO_REG32_BIT(GCR_TEMPCTL,           0x50000020,__READ_WRITE ,__gcr_tempctl_bits);
__IO_REG32_BIT(GCR_PA_L_MFP,          0x50000030,__READ_WRITE ,__gcr_pa_l_mfp_bits);
__IO_REG32_BIT(GCR_PA_H_MFP,          0x50000034,__READ_WRITE ,__gcr_pa_h_mfp_bits);
__IO_REG32_BIT(GCR_PB_L_MFP,          0x50000038,__READ_WRITE ,__gcr_pb_l_mfp_bits);
__IO_REG32_BIT(GCR_PB_H_MFP,          0x5000003C,__READ_WRITE ,__gcr_pb_h_mfp_bits);
__IO_REG32_BIT(GCR_PC_L_MFP,          0x50000040,__READ_WRITE ,__gcr_pc_l_mfp_bits);
__IO_REG32_BIT(GCR_PC_H_MFP,          0x50000044,__READ_WRITE ,__gcr_pc_h_mfp_bits);
__IO_REG32_BIT(GCR_PD_L_MFP,          0x50000048,__READ_WRITE ,__gcr_pd_l_mfp_bits);
__IO_REG32_BIT(GCR_PD_H_MFP,          0x5000004C,__READ_WRITE ,__gcr_pd_h_mfp_bits);
__IO_REG32_BIT(GCR_PE_L_MFP,          0x50000050,__READ_WRITE ,__gcr_pe_l_mfp_bits);
__IO_REG32_BIT(GCR_PE_H_MFP,          0x50000054,__READ_WRITE ,__gcr_pe_h_mfp_bits);
__IO_REG32_BIT(GCR_PF_L_MFP,          0x50000058,__READ_WRITE ,__gcr_pf_l_mfp_bits);
__IO_REG32_BIT(GCR_PORCTL,            0x50000060,__READ_WRITE ,__gcr_porctl_bits);
__IO_REG32_BIT(GCR_BODCTL,            0x50000064,__READ_WRITE ,__gcr_bodctl_bits);
__IO_REG32_BIT(GCR_BODSTS,            0x50000068,__READ       ,__gcr_bodsts_bits);
__IO_REG32_BIT(GCR_VREFCTL,           0x5000006C,__READ_WRITE ,__gcr_vrefctl_bits);
__IO_REG32_BIT(GCR_LDOCTL,            0x50000070,__READ_WRITE ,__gcr_ldoctl_bits);
__IO_REG32_BIT(GCR_VDCTL,             0x50000074,__READ_WRITE ,__gcr_vdctl_bits);
__IO_REG32_BIT(GCR_LPLDOCTL,          0x50000078,__READ_WRITE ,__gcr_lpldoctl_bits);
__IO_REG32_BIT(GCR_IRCTRIMCTL,        0x50000080,__READ_WRITE ,__gcr_irctrimctl_bits);
__IO_REG32_BIT(GCR_IRCTRIMIEN,        0x50000084,__READ_WRITE ,__gcr_irctrimien_bits);
__IO_REG32_BIT(GCR_IRCTRIMINT,        0x50000088,__READ_WRITE ,__gcr_irctrimint_bits);
__IO_REG32_BIT(GCR_RegLockAddr,       0x50000100,__READ_WRITE ,__gcr_reglockaddr_bits);
__IO_REG32_BIT(GCR_RCADJ,             0x50000110,__READ_WRITE ,__gcr_rcadj_bits);

/***************************************************************************
 **
 ** SCS
 **
 ***************************************************************************/
__IO_REG32_BIT(SCS_SYST_CTL,          0xE000E010,__READ_WRITE ,__scs_syst_ctl_bits);
__IO_REG32_BIT(SCS_SYST_RVR,          0xE000E014,__READ_WRITE ,__scs_syst_rvr_bits);
__IO_REG32_BIT(SCS_SYST_CVR,          0xE000E018,__READ_WRITE ,__scs_syst_cvr_bits);
__IO_REG32(    SCS_NVIC_ISER,         0xE000E100,__READ_WRITE );
__IO_REG32(    SCS_NVIC_ICER,         0xE000E180,__READ_WRITE );
__IO_REG32(    SCS_NVIC_ISPR,         0xE000E200,__READ_WRITE );
__IO_REG32(    SCS_NVIC_ICPR,         0xE000E280,__READ_WRITE );
__IO_REG32_BIT(SCS_NVIC_IPR0,         0xE000E400,__READ_WRITE ,__scs_nvic_ipr0_bits);
__IO_REG32_BIT(SCS_NVIC_IPR1,         0xE000E404,__READ_WRITE ,__scs_nvic_ipr1_bits);
__IO_REG32_BIT(SCS_NVIC_IPR2,         0xE000E408,__READ_WRITE ,__scs_nvic_ipr2_bits);
__IO_REG32_BIT(SCS_NVIC_IPR3,         0xE000E40C,__READ_WRITE ,__scs_nvic_ipr3_bits);
__IO_REG32_BIT(SCS_NVIC_IPR4,         0xE000E410,__READ_WRITE ,__scs_nvic_ipr4_bits);
__IO_REG32_BIT(SCS_NVIC_IPR5,         0xE000E414,__READ_WRITE ,__scs_nvic_ipr5_bits);
__IO_REG32_BIT(SCS_NVIC_IPR6,         0xE000E418,__READ_WRITE ,__scs_nvic_ipr6_bits);
__IO_REG32_BIT(SCS_NVIC_IPR7,         0xE000E41C,__READ_WRITE ,__scs_nvic_ipr7_bits);
__IO_REG32_BIT(SCS_CPUID,             0xE000ED00,__READ       ,__scs_cpuid_bits);
__IO_REG32_BIT(SCS_ICSR,              0xE000ED04,__READ_WRITE ,__scs_icsr_bits);
__IO_REG32_BIT(SCS_SCR,               0xE000ED10,__READ_WRITE ,__scs_scr_bits);
__IO_REG32_BIT(SCS_SHPR2,             0xE000ED1C,__READ_WRITE ,__scs_shpr2_bits);
__IO_REG32_BIT(SCS_SHPR3,             0xE000ED20,__READ_WRITE ,__scs_shpr3_bits);

/***************************************************************************
 **
 ** INT
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_IRQ0_SRC,          0x50000300,__READ       ,__int_irq0_src_bits);
__IO_REG32_BIT(INT_IRQ1_SRC,          0x50000304,__READ       ,__int_irq1_src_bits);
__IO_REG32_BIT(INT_IRQ2_SRC,          0x50000308,__READ       ,__int_irq2_src_bits);
__IO_REG32_BIT(INT_IRQ3_SRC,          0x5000030C,__READ       ,__int_irq3_src_bits);
__IO_REG32_BIT(INT_IRQ4_SRC,          0x50000310,__READ       ,__int_irq4_src_bits);
__IO_REG32_BIT(INT_IRQ5_SRC,          0x50000314,__READ       ,__int_irq5_src_bits);
__IO_REG32_BIT(INT_IRQ6_SRC,          0x50000318,__READ       ,__int_irq6_src_bits);
__IO_REG32_BIT(INT_IRQ7_SRC,          0x5000031C,__READ       ,__int_irq7_src_bits);
__IO_REG32_BIT(INT_IRQ8_SRC,          0x50000320,__READ       ,__int_irq8_src_bits);
__IO_REG32_BIT(INT_IRQ9_SRC,          0x50000324,__READ       ,__int_irq9_src_bits);
__IO_REG32_BIT(INT_IRQ10_SRC,         0x50000328,__READ       ,__int_irq10_src_bits);
__IO_REG32_BIT(INT_IRQ11_SRC,         0x5000032C,__READ       ,__int_irq11_src_bits);
__IO_REG32_BIT(INT_IRQ12_SRC,         0x50000330,__READ       ,__int_irq12_src_bits);
__IO_REG32_BIT(INT_IRQ13_SRC,         0x50000334,__READ       ,__int_irq13_src_bits);
__IO_REG32_BIT(INT_IRQ14_SRC,         0x50000338,__READ       ,__int_irq14_src_bits);
__IO_REG32_BIT(INT_IRQ15_SRC,         0x5000033C,__READ       ,__int_irq15_src_bits);
__IO_REG32_BIT(INT_IRQ18_SRC,         0x50000348,__READ       ,__int_irq18_src_bits);
__IO_REG32_BIT(INT_IRQ19_SRC,         0x5000034C,__READ       ,__int_irq19_src_bits);
__IO_REG32_BIT(INT_IRQ20_SRC,         0x50000350,__READ       ,__int_irq20_src_bits);
__IO_REG32_BIT(INT_IRQ23_SRC,         0x5000035C,__READ       ,__int_irq23_src_bits);
__IO_REG32_BIT(INT_IRQ24_SRC,         0x50000360,__READ       ,__int_irq24_src_bits);
__IO_REG32_BIT(INT_IRQ25_SRC,         0x50000364,__READ       ,__int_irq25_src_bits);
__IO_REG32_BIT(INT_IRQ26_SRC,         0x50000368,__READ       ,__int_irq26_src_bits);
__IO_REG32_BIT(INT_IRQ27_SRC,         0x5000036C,__READ       ,__int_irq27_src_bits);
__IO_REG32_BIT(INT_IRQ28_SRC,         0x50000370,__READ       ,__int_irq28_src_bits);
__IO_REG32_BIT(INT_IRQ29_SRC,         0x50000374,__READ       ,__int_irq29_src_bits);
__IO_REG32_BIT(INT_IRQ30_SRC,         0x50000378,__READ       ,__int_irq30_src_bits);
__IO_REG32_BIT(INT_IRQ31_SRC,         0x5000037C,__READ       ,__int_irq31_src_bits);
__IO_REG32_BIT(INT_NMI_SEL,           0x50000380,__READ_WRITE ,__int_nmi_sel_bits);
__IO_REG32(    INT_MCU_IRQ,           0x50000384,__READ_WRITE );

/***************************************************************************
 **
 ** CLK
 **
 ***************************************************************************/
__IO_REG32_BIT(CLK_PWRCTL,            0x50000200,__READ_WRITE ,__clk_pwrctl_bits);
__IO_REG32_BIT(CLK_AHBCLK,            0x50000204,__READ_WRITE ,__clk_ahbclk_bits);
__IO_REG32_BIT(CLK_APBCLK,            0x50000208,__READ_WRITE ,__clk_apbclk_bits);
__IO_REG32_BIT(CLK_CLKSTATUS,         0x5000020C,__READ       ,__clk_clkstatus_bits);
__IO_REG32_BIT(CLK_CLKSEL0,           0x50000210,__READ_WRITE ,__clk_clksel0_bits);
__IO_REG32_BIT(CLK_CLKSEL1,           0x50000214,__READ_WRITE ,__clk_clksel1_bits);
__IO_REG32_BIT(CLK_CLKSEL2,           0x50000218,__READ_WRITE ,__clk_clksel2_bits);
__IO_REG32_BIT(CLK_CLKDIV0,           0x5000021C,__READ_WRITE ,__clk_clkdiv0_bits);
__IO_REG32_BIT(CLK_CLKDIV1,           0x50000220,__READ_WRITE ,__clk_clkdiv1_bits);
__IO_REG32_BIT(CLK_PLLCTL,            0x50000224,__READ_WRITE ,__clk_pllctl_bits);
__IO_REG32_BIT(CLK_FRQDIV,            0x50000228,__READ_WRITE ,__clk_frqdiv_bits);
__IO_REG32_BIT(CLK_TESTCLK,           0x5000022C,__READ_WRITE ,__clk_testclk_bits);
__IO_REG32_BIT(CLK_PD_WK_IS,          0x50000230,__READ       ,__clk_pd_wk_is_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_GPIOA_PMD,        0x50004000,__READ_WRITE ,__gpio_gpioa_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOA_OFFD,       0x50004004,__READ_WRITE ,__gpio_gpioa_offd_bits);
__IO_REG32_BIT(GPIO_GPIOA_DOUT,       0x50004008,__READ_WRITE ,__gpio_gpioa_dout_bits);
__IO_REG32_BIT(GPIO_GPIOA_DMASK,      0x5000400C,__READ_WRITE ,__gpio_gpioa_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOA_PIN,        0x50004010,__READ       ,__gpio_gpioa_pin_bits);
__IO_REG32_BIT(GPIO_GPIOA_DBEN,       0x50004014,__READ_WRITE ,__gpio_gpioa_dben_bits);
__IO_REG32_BIT(GPIO_GPIOA_IMD,        0x50004018,__READ_WRITE ,__gpio_gpioa_imd_bits);
__IO_REG32_BIT(GPIO_GPIOA_IER,        0x5000401C,__READ_WRITE ,__gpio_gpioa_ier_bits);
__IO_REG32_BIT(GPIO_GPIOA_ISRC,       0x50004020,__READ_WRITE ,__gpio_gpioa_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOA_PUEN,       0x50004024,__READ_WRITE ,__gpio_gpioa_puen_bits);
__IO_REG32_BIT(GPIO_GPIOB_PMD,        0x50004040,__READ_WRITE ,__gpio_gpiob_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOB_OFFD,       0x50004044,__READ_WRITE ,__gpio_gpiob_offd_bits);
__IO_REG32_BIT(GPIO_GPIOB_DOUT,       0x50004048,__READ_WRITE ,__gpio_gpiob_dout_bits);
__IO_REG32_BIT(GPIO_GPIOB_DMASK,      0x5000404C,__READ_WRITE ,__gpio_gpiob_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOB_PIN,        0x50004050,__READ       ,__gpio_gpiob_pin_bits);
__IO_REG32_BIT(GPIO_GPIOB_DBEN,       0x50004054,__READ_WRITE ,__gpio_gpiob_dben_bits);
__IO_REG32_BIT(GPIO_GPIOB_IMD,        0x50004058,__READ_WRITE ,__gpio_gpiob_imd_bits);
__IO_REG32_BIT(GPIO_GPIOB_IER,        0x5000405C,__READ_WRITE ,__gpio_gpiob_ier_bits);
__IO_REG32_BIT(GPIO_GPIOB_ISRC,       0x50004060,__READ_WRITE ,__gpio_gpiob_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOB_PUEN,       0x50004064,__READ_WRITE ,__gpio_gpiob_puen_bits);
__IO_REG32_BIT(GPIO_GPIOC_PMD,        0x50004080,__READ_WRITE ,__gpio_gpioc_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOC_OFFD,       0x50004084,__READ_WRITE ,__gpio_gpioc_offd_bits);
__IO_REG32_BIT(GPIO_GPIOC_DOUT,       0x50004088,__READ_WRITE ,__gpio_gpioc_dout_bits);
__IO_REG32_BIT(GPIO_GPIOC_DMASK,      0x5000408C,__READ_WRITE ,__gpio_gpioc_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOC_PIN,        0x50004090,__READ       ,__gpio_gpioc_pin_bits);
__IO_REG32_BIT(GPIO_GPIOC_DBEN,       0x50004094,__READ_WRITE ,__gpio_gpioc_dben_bits);
__IO_REG32_BIT(GPIO_GPIOC_IMD,        0x50004098,__READ_WRITE ,__gpio_gpioc_imd_bits);
__IO_REG32_BIT(GPIO_GPIOC_IER,        0x5000409C,__READ_WRITE ,__gpio_gpioc_ier_bits);
__IO_REG32_BIT(GPIO_GPIOC_ISRC,       0x500040A0,__READ_WRITE ,__gpio_gpioc_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOC_PUEN,       0x500040A4,__READ_WRITE ,__gpio_gpioc_puen_bits);
__IO_REG32_BIT(GPIO_GPIOD_PMD,        0x500040C0,__READ_WRITE ,__gpio_gpiod_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOD_OFFD,       0x500040C4,__READ_WRITE ,__gpio_gpiod_offd_bits);
__IO_REG32_BIT(GPIO_GPIOD_DOUT,       0x500040C8,__READ_WRITE ,__gpio_gpiod_dout_bits);
__IO_REG32_BIT(GPIO_GPIOD_DMASK,      0x500040CC,__READ_WRITE ,__gpio_gpiod_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOD_PIN,        0x500040D0,__READ       ,__gpio_gpiod_pin_bits);
__IO_REG32_BIT(GPIO_GPIOD_DBEN,       0x500040D4,__READ_WRITE ,__gpio_gpiod_dben_bits);
__IO_REG32_BIT(GPIO_GPIOD_IMD,        0x500040D8,__READ_WRITE ,__gpio_gpiod_imd_bits);
__IO_REG32_BIT(GPIO_GPIOD_IER,        0x500040DC,__READ_WRITE ,__gpio_gpiod_ier_bits);
__IO_REG32_BIT(GPIO_GPIOD_ISRC,       0x500040E0,__READ_WRITE ,__gpio_gpiod_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOD_PUEN,       0x500040E4,__READ_WRITE ,__gpio_gpiod_puen_bits);
__IO_REG32_BIT(GPIO_GPIOE_PMD,        0x50004100,__READ_WRITE ,__gpio_gpioe_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOE_OFFD,       0x50004104,__READ_WRITE ,__gpio_gpioe_offd_bits);
__IO_REG32_BIT(GPIO_GPIOE_DOUT,       0x50004108,__READ_WRITE ,__gpio_gpioe_dout_bits);
__IO_REG32_BIT(GPIO_GPIOE_DMASK,      0x5000410C,__READ_WRITE ,__gpio_gpioe_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOE_PIN,        0x50004110,__READ       ,__gpio_gpioe_pin_bits);
__IO_REG32_BIT(GPIO_GPIOE_DBEN,       0x50004114,__READ_WRITE ,__gpio_gpioe_dben_bits);
__IO_REG32_BIT(GPIO_GPIOE_IMD,        0x50004118,__READ_WRITE ,__gpio_gpioe_imd_bits);
__IO_REG32_BIT(GPIO_GPIOE_IER,        0x5000411C,__READ_WRITE ,__gpio_gpioe_ier_bits);
__IO_REG32_BIT(GPIO_GPIOE_ISRC,       0x50004120,__READ_WRITE ,__gpio_gpioe_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOE_PUEN,       0x50004124,__READ_WRITE ,__gpio_gpioe_puen_bits);
__IO_REG32_BIT(GPIO_GPIOF_PMD,        0x50004140,__READ_WRITE ,__gpio_gpiof_pmd_bits);
__IO_REG32_BIT(GPIO_GPIOF_OFFD,       0x50004144,__READ_WRITE ,__gpio_gpiof_offd_bits);
__IO_REG32_BIT(GPIO_GPIOF_DOUT,       0x50004148,__READ_WRITE ,__gpio_gpiof_dout_bits);
__IO_REG32_BIT(GPIO_GPIOF_DMASK,      0x5000414C,__READ_WRITE ,__gpio_gpiof_dmask_bits);
__IO_REG32_BIT(GPIO_GPIOF_PIN,        0x50004150,__READ       ,__gpio_gpiof_pin_bits);
__IO_REG32_BIT(GPIO_GPIOF_DBEN,       0x50004154,__READ_WRITE ,__gpio_gpiof_dben_bits);
__IO_REG32_BIT(GPIO_GPIOF_IMD,        0x50004158,__READ_WRITE ,__gpio_gpiof_imd_bits);
__IO_REG32_BIT(GPIO_GPIOF_IER,        0x5000415C,__READ_WRITE ,__gpio_gpiof_ier_bits);
__IO_REG32_BIT(GPIO_GPIOF_ISRC,       0x50004160,__READ_WRITE ,__gpio_gpiof_isrc_bits);
__IO_REG32_BIT(GPIO_GPIOF_PUEN,       0x50004164,__READ_WRITE ,__gpio_gpiof_puen_bits);
__IO_REG32_BIT(GPIO_DBNCECON,         0x50004180,__READ_WRITE ,__gpio_dbncecon_bits);
__IO_REG32_BIT(GPIO_GPIOA0,           0x50004200,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA1,           0x50004204,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA2,           0x50004208,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA3,           0x5000420C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA4,           0x50004210,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA5,           0x50004214,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA6,           0x50004218,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA7,           0x5000421C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA8,           0x50004220,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA9,           0x50004224,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA10,          0x50004228,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA11,          0x5000422C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA12,          0x50004230,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA13,          0x50004234,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA14,          0x50004238,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOA15,          0x5000423C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB0,           0x50004240,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB1,           0x50004244,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB2,           0x50004248,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB3,           0x5000424C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB4,           0x50004250,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB5,           0x50004254,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB6,           0x50004258,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB7,           0x5000425C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB8,           0x50004260,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB9,           0x50004264,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB10,          0x50004268,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB11,          0x5000426C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB12,          0x50004270,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB13,          0x50004274,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB14,          0x50004278,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOB15,          0x5000427C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC0,           0x50004280,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC1,           0x50004284,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC2,           0x50004288,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC3,           0x5000428C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC4,           0x50004290,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC5,           0x50004294,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC6,           0x50004298,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC7,           0x5000429C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC8,           0x500042A0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC9,           0x500042A4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC10,          0x500042A8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC11,          0x500042AC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC12,          0x500042B0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC13,          0x500042B4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC14,          0x500042B8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOC15,          0x500042BC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD0,           0x500042C0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD1,           0x500042C4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD2,           0x500042C8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD3,           0x500042CC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD4,           0x500042D0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD5,           0x500042D4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD6,           0x500042D8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD7,           0x500042DC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD8,           0x500042E0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD9,           0x500042E4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD10,          0x500042E8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD11,          0x500042EC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD12,          0x500042F0,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD13,          0x500042F4,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD14,          0x500042F8,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOD15,          0x500042FC,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE0,           0x50004300,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE1,           0x50004304,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE2,           0x50004308,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE3,           0x5000430C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE4,           0x50004310,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE5,           0x50004314,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE6,           0x50004318,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE7,           0x5000431C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE8,           0x50004320,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE9,           0x50004324,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE10,          0x50004328,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE11,          0x5000432C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE12,          0x50004330,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE13,          0x50004334,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE14,          0x50004338,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOE15,          0x5000433C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF0,           0x50004340,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF1,           0x50004344,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF2,           0x50004348,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF3,           0x5000434C,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF4,           0x50004350,__READ_WRITE ,__gpio_gpioxy_bits);
__IO_REG32_BIT(GPIO_GPIOF5,           0x50004354,__READ_WRITE ,__gpio_gpioxy_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0_I2CCON,           0x40020000,__READ_WRITE ,__i2cx_i2ccon_bits);
__IO_REG32_BIT(I2C0_I2CINTSTS,        0x40020004,__READ_WRITE ,__i2cx_i2cintsts_bits);
__IO_REG32_BIT(I2C0_I2CSTATUS,        0x40020008,__READ       ,__i2cx_i2cstatus_bits);
__IO_REG32_BIT(I2C0_I2CDIV,           0x4002000C,__READ_WRITE ,__i2cx_i2cdiv_bits);
__IO_REG32_BIT(I2C0_I2CTOUT,          0x40020010,__READ_WRITE ,__i2cx_i2ctout_bits);
__IO_REG32_BIT(I2C0_I2CDATA,          0x40020014,__READ_WRITE ,__i2cx_i2cdata_bits);
__IO_REG32_BIT(I2C0_I2CSADDR0,        0x40020018,__READ_WRITE ,__i2cx_i2csaddry_bits);
__IO_REG32_BIT(I2C0_I2CSADDR1,        0x4002001C,__READ_WRITE ,__i2cx_i2csaddry_bits);
__IO_REG32_BIT(I2C0_I2CSAMASK0,       0x40020028,__READ_WRITE ,__i2cx_i2csamasky_bits);
__IO_REG32_BIT(I2C0_I2CSAMASK1,       0x4002002C,__READ_WRITE ,__i2cx_i2csamasky_bits);
__IO_REG32_BIT(I2C1_I2CCON,           0x40120000,__READ_WRITE ,__i2cx_i2ccon_bits);
__IO_REG32_BIT(I2C1_I2CINTSTS,        0x40120004,__READ_WRITE ,__i2cx_i2cintsts_bits);
__IO_REG32_BIT(I2C1_I2CSTATUS,        0x40120008,__READ       ,__i2cx_i2cstatus_bits);
__IO_REG32_BIT(I2C1_I2CDIV,           0x4012000C,__READ_WRITE ,__i2cx_i2cdiv_bits);
__IO_REG32_BIT(I2C1_I2CTOUT,          0x40120010,__READ_WRITE ,__i2cx_i2ctout_bits);
__IO_REG32_BIT(I2C1_I2CDATA,          0x40120014,__READ_WRITE ,__i2cx_i2cdata_bits);
__IO_REG32_BIT(I2C1_I2CSADDR0,        0x40120018,__READ_WRITE ,__i2cx_i2csaddry_bits);
__IO_REG32_BIT(I2C1_I2CSADDR1,        0x4012001C,__READ_WRITE ,__i2cx_i2csaddry_bits);
__IO_REG32_BIT(I2C1_I2CSAMASK0,       0x40120028,__READ_WRITE ,__i2cx_i2csamasky_bits);
__IO_REG32_BIT(I2C1_I2CSAMASK1,       0x4012002C,__READ_WRITE ,__i2cx_i2csamasky_bits);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM0_PWMx_PRES,        0x40040000,__READ_WRITE ,__pwm0_pwmx_pres_bits);
__IO_REG32_BIT(PWM0_PWMx_CLKSEL,      0x40040004,__READ_WRITE ,__pwm0_pwmx_clksel_bits);
__IO_REG32_BIT(PWM0_PWMx_CTL,         0x40040008,__READ_WRITE ,__pwm0_pwmx_ctl_bits);
__IO_REG32_BIT(PWM0_PWMx_INTEN,       0x4004000C,__READ_WRITE ,__pwm0_pwmx_inten_bits);
__IO_REG32_BIT(PWM0_PWMx_INTSTS,      0x40040010,__READ_WRITE ,__pwm0_pwmx_intsts_bits);
__IO_REG32_BIT(PWM0_PWMx_OE,          0x40040014,__READ_WRITE ,__pwm0_pwmx_oe_bits);
__IO_REG32_BIT(PWM0_PWMx_DUTY0,       0x4004001C,__READ_WRITE ,__pwm0_pwmx_duty0_bits);
__IO_REG32_BIT(PWM0_PWMx_DATA0,       0x40040020,__READ       ,__pwm0_pwmx_data0_bits);
__IO_REG32_BIT(PWM0_PWMx_DUTY1,       0x40040028,__READ_WRITE ,__pwm0_pwmx_duty1_bits);
__IO_REG32_BIT(PWM0_PWMx_DATA1,       0x4004002C,__READ       ,__pwm0_pwmx_data1_bits);
__IO_REG32_BIT(PWM0_PWMx_DUTY2,       0x40040034,__READ_WRITE ,__pwm0_pwmx_duty2_bits);
__IO_REG32_BIT(PWM0_PWMx_DATA2,       0x40040038,__READ       ,__pwm0_pwmx_data2_bits);
__IO_REG32_BIT(PWM0_PWMx_DUTY3,       0x40040040,__READ_WRITE ,__pwm0_pwmx_duty3_bits);
__IO_REG32_BIT(PWM0_PWMx_DATA3,       0x40040044,__READ       ,__pwm0_pwmx_data3_bits);
__IO_REG32_BIT(PWM0_PWMx_CAPCTL,      0x40040054,__READ_WRITE ,__pwm0_pwmx_capctl_bits);
__IO_REG32_BIT(PWM0_PWMx_CAPINTEN,    0x40040058,__READ_WRITE ,__pwm0_pwmx_capinten_bits);
__IO_REG32_BIT(PWM0_PWMx_CAPINTSTS,   0x4004005C,__READ_WRITE ,__pwm0_pwmx_capintsts_bits);
__IO_REG32_BIT(PWM0_PWMx_CRL0,        0x40040060,__READ       ,__pwm0_pwmx_crl0_bits);
__IO_REG32_BIT(PWM0_PWMx_CFL0,        0x40040064,__READ       ,__pwm0_pwmx_cfl0_bits);
__IO_REG32_BIT(PWM0_PWMx_CRL1,        0x40040068,__READ       ,__pwm0_pwmx_crl1_bits);
__IO_REG32_BIT(PWM0_PWMx_CFL1,        0x4004006C,__READ       ,__pwm0_pwmx_cfl1_bits);
__IO_REG32_BIT(PWM0_PWMx_CRL2,        0x40040070,__READ       ,__pwm0_pwmx_crl2_bits);
__IO_REG32_BIT(PWM0_PWMx_CFL2,        0x40040074,__READ       ,__pwm0_pwmx_cfl2_bits);
__IO_REG32_BIT(PWM0_PWMx_CRL3,        0x40040078,__READ       ,__pwm0_pwmx_crl3_bits);
__IO_REG32_BIT(PWM0_PWMx_CFL3,        0x4004007C,__READ       ,__pwm0_pwmx_cfl3_bits);
__IO_REG32_BIT(PWM0_PWMx_PDMACH0,     0x40040080,__READ       ,__pwm0_pwmx_pdmach0_bits);
__IO_REG32_BIT(PWM0_PWMx_PDMACH2,     0x40040084,__READ       ,__pwm0_pwmx_pdmach2_bits);
__IO_REG32_BIT(PWM1_PWMx_PRES,        0x40140000,__READ_WRITE ,__pwm1_pwmx_pres_bits);
__IO_REG32_BIT(PWM1_PWMx_CLKSEL,      0x40140004,__READ_WRITE ,__pwm1_pwmx_clksel_bits);
__IO_REG32_BIT(PWM1_PWMx_CTL,         0x40140008,__READ_WRITE ,__pwm1_pwmx_ctl_bits);
__IO_REG32_BIT(PWM1_PWMx_INTEN,       0x4014000C,__READ_WRITE ,__pwm1_pwmx_inten_bits);
__IO_REG32_BIT(PWM1_PWMx_INTSTS,      0x40140010,__READ_WRITE ,__pwm1_pwmx_intsts_bits);
__IO_REG32_BIT(PWM1_PWMx_OE,          0x40140014,__READ_WRITE ,__pwm1_pwmx_oe_bits);
__IO_REG32_BIT(PWM1_PWMx_DUTY0,       0x4014001C,__READ_WRITE ,__pwm1_pwmx_duty0_bits);
__IO_REG32_BIT(PWM1_PWMx_DATA0,       0x40140020,__READ       ,__pwm1_pwmx_data0_bits);
__IO_REG32_BIT(PWM1_PWMx_DUTY1,       0x40140028,__READ_WRITE ,__pwm1_pwmx_duty1_bits);
__IO_REG32_BIT(PWM1_PWMx_DATA1,       0x4014002C,__READ       ,__pwm1_pwmx_data1_bits);
__IO_REG32_BIT(PWM1_PWMx_DUTY2,       0x40140034,__READ_WRITE ,__pwm1_pwmx_duty2_bits);
__IO_REG32_BIT(PWM1_PWMx_DATA2,       0x40140038,__READ       ,__pwm1_pwmx_data2_bits);
__IO_REG32_BIT(PWM1_PWMx_DUTY3,       0x40140040,__READ_WRITE ,__pwm1_pwmx_duty3_bits);
__IO_REG32_BIT(PWM1_PWMx_DATA3,       0x40140044,__READ       ,__pwm1_pwmx_data3_bits);
__IO_REG32_BIT(PWM1_PWMx_CAPCTL,      0x40140054,__READ_WRITE ,__pwm1_pwmx_capctl_bits);
__IO_REG32_BIT(PWM1_PWMx_CAPINTEN,    0x40140058,__READ_WRITE ,__pwm1_pwmx_capinten_bits);
__IO_REG32_BIT(PWM1_PWMx_CAPINTSTS,   0x4014005C,__READ_WRITE ,__pwm1_pwmx_capintsts_bits);
__IO_REG32_BIT(PWM1_PWMx_CRL0,        0x40140060,__READ       ,__pwm1_pwmx_crl0_bits);
__IO_REG32_BIT(PWM1_PWMx_CFL0,        0x40140064,__READ       ,__pwm1_pwmx_cfl0_bits);
__IO_REG32_BIT(PWM1_PWMx_CRL1,        0x40140068,__READ       ,__pwm1_pwmx_crl1_bits);
__IO_REG32_BIT(PWM1_PWMx_CFL1,        0x4014006C,__READ       ,__pwm1_pwmx_cfl1_bits);
__IO_REG32_BIT(PWM1_PWMx_CRL2,        0x40140070,__READ       ,__pwm1_pwmx_crl2_bits);
__IO_REG32_BIT(PWM1_PWMx_CFL2,        0x40140074,__READ       ,__pwm1_pwmx_cfl2_bits);
__IO_REG32_BIT(PWM1_PWMx_CRL3,        0x40140078,__READ       ,__pwm1_pwmx_crl3_bits);
__IO_REG32_BIT(PWM1_PWMx_CFL3,        0x4014007C,__READ       ,__pwm1_pwmx_cfl3_bits);
__IO_REG32_BIT(PWM1_PWMx_PDMACH0,     0x40140080,__READ       ,__pwm1_pwmx_pdmach0_bits);
__IO_REG32_BIT(PWM1_PWMx_PDMACH2,     0x40140084,__READ       ,__pwm1_pwmx_pdmach2_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_INIR,          0x40008000,__READ_WRITE ,__rtc_rtc_inir_bits);
__IO_REG32_BIT(RTC_AER,           0x40008004,__READ_WRITE ,__rtc_rtc_aer_bits);
__IO_REG32_BIT(RTC_FCR,           0x40008008,__READ_WRITE ,__rtc_rtc_fcr_bits);
__IO_REG32_BIT(RTC_TLR,           0x4000800C,__READ_WRITE ,__rtc_rtc_tlr_bits);
__IO_REG32_BIT(RTC_CLR,           0x40008010,__READ_WRITE ,__rtc_rtc_clr_bits);
__IO_REG32_BIT(RTC_TSSR,          0x40008014,__READ_WRITE ,__rtc_rtc_tssr_bits);
__IO_REG32_BIT(RTC_DWR,           0x40008018,__READ_WRITE ,__rtc_rtc_dwr_bits);
__IO_REG32_BIT(RTC_TAR,           0x4000801C,__READ_WRITE ,__rtc_rtc_tar_bits);
__IO_REG32_BIT(RTC_CAR,           0x40008020,__READ_WRITE ,__rtc_rtc_car_bits);
__IO_REG32_BIT(RTC_LIR,           0x40008024,__READ       ,__rtc_rtc_lir_bits);
__IO_REG32_BIT(RTC_RIER,          0x40008028,__READ_WRITE ,__rtc_rtc_rier_bits);
__IO_REG32_BIT(RTC_RIIR,          0x4000802C,__READ       ,__rtc_rtc_riir_bits);
__IO_REG32_BIT(RTC_TTR,           0x40008030,__READ_WRITE ,__rtc_rtc_ttr_bits);
__IO_REG32_BIT(RTC_SPRCTL,        0x4000803C,__READ_WRITE ,__rtc_rtc_sprctl_bits);
__IO_REG32(    RTC_SPR0,          0x40008040,__READ_WRITE );
__IO_REG32(    RTC_SPR1,          0x40008044,__READ_WRITE );
__IO_REG32(    RTC_SPR2,          0x40008048,__READ_WRITE );
__IO_REG32(    RTC_SPR3,          0x4000804C,__READ_WRITE );
__IO_REG32(    RTC_SPR4,          0x40008050,__READ_WRITE );
__IO_REG32(    RTC_SPR5,          0x40008054,__READ_WRITE );
__IO_REG32(    RTC_SPR6,          0x40008058,__READ_WRITE );
__IO_REG32(    RTC_SPR7,          0x4000805C,__READ_WRITE );
__IO_REG32(    RTC_SPR8,          0x40008060,__READ_WRITE );
__IO_REG32(    RTC_SPR9,          0x40008064,__READ_WRITE );
__IO_REG32(    RTC_SPR10,         0x40008068,__READ_WRITE );
__IO_REG32(    RTC_SPR11,         0x4000806C,__READ_WRITE );
__IO_REG32(    RTC_SPR12,         0x40008070,__READ_WRITE );
__IO_REG32(    RTC_SPR13,         0x40008074,__READ_WRITE );
__IO_REG32(    RTC_SPR14,         0x40008078,__READ_WRITE );
__IO_REG32(    RTC_SPR15,         0x4000807C,__READ_WRITE );
__IO_REG32(    RTC_SPR16,         0x40008080,__READ_WRITE );
__IO_REG32(    RTC_SPR17,         0x40008084,__READ_WRITE );
__IO_REG32(    RTC_SPR18,         0x40008088,__READ_WRITE );
__IO_REG32(    RTC_SPR19,         0x4000808C,__READ_WRITE );

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_SPI_CTL,          0x40030000,__READ_WRITE ,__spix_spi_ctl_bits);
__IO_REG32_BIT(SPI0_SPI_STATUS,       0x40030004,__READ_WRITE ,__spix_spi_status_bits);
__IO_REG32_BIT(SPI0_SPI_CLKDIV,       0x40030008,__READ_WRITE ,__spix_spi_clkdiv_bits);
__IO_REG32_BIT(SPI0_SPI_SSR,          0x4003000C,__READ_WRITE ,__spix_spi_ssr_bits);
__IO_REG32(    SPI0_SPI_RX0,          0x40030010,__READ       );
__IO_REG32(    SPI0_SPI_RX1,          0x40030014,__READ       );
__IO_REG32(    SPI0_SPI_TX0,          0x40030020,__WRITE      );
__IO_REG32(    SPI0_SPI_TX1,          0x40030024,__WRITE      );
__IO_REG32(    SPI0_SPI_VARCLK,       0x40030034,__READ_WRITE );
__IO_REG32_BIT(SPI0_SPI_PDMA,         0x40030038,__READ_WRITE ,__spix_spi_pdma_bits);
__IO_REG32_BIT(SPI0_SPI_FFCLR,        0x4003003C,__READ_WRITE ,__spix_spi_ffclr_bits);
__IO_REG32_BIT(SPI1_SPI_CTL,          0x40130000,__READ_WRITE ,__spix_spi_ctl_bits);
__IO_REG32_BIT(SPI1_SPI_STATUS,       0x40130004,__READ_WRITE ,__spix_spi_status_bits);
__IO_REG32_BIT(SPI1_SPI_CLKDIV,       0x40130008,__READ_WRITE ,__spix_spi_clkdiv_bits);
__IO_REG32_BIT(SPI1_SPI_SSR,          0x4013000C,__READ_WRITE ,__spix_spi_ssr_bits);
__IO_REG32(    SPI1_SPI_RX0,          0x40130010,__READ       );
__IO_REG32(    SPI1_SPI_RX1,          0x40130014,__READ       );
__IO_REG32(    SPI1_SPI_TX0,          0x40130020,__WRITE      );
__IO_REG32(    SPI1_SPI_TX1,          0x40130024,__WRITE      );
__IO_REG32(    SPI1_SPI_VARCLK,       0x40130034,__READ_WRITE );
__IO_REG32_BIT(SPI1_SPI_PDMA,         0x40130038,__READ_WRITE ,__spix_spi_pdma_bits);
__IO_REG32_BIT(SPI1_SPI_FFCLR,        0x4013003C,__READ_WRITE ,__spix_spi_ffclr_bits);
__IO_REG32_BIT(SPI2_SPI_CTL,          0x400D0000,__READ_WRITE ,__spix_spi_ctl_bits);
__IO_REG32_BIT(SPI2_SPI_STATUS,       0x400D0004,__READ_WRITE ,__spix_spi_status_bits);
__IO_REG32_BIT(SPI2_SPI_CLKDIV,       0x400D0008,__READ_WRITE ,__spix_spi_clkdiv_bits);
__IO_REG32_BIT(SPI2_SPI_SSR,          0x400D000C,__READ_WRITE ,__spix_spi_ssr_bits);
__IO_REG32(    SPI2_SPI_RX0,          0x400D0010,__READ       );
__IO_REG32(    SPI2_SPI_RX1,          0x400D0014,__READ       );
__IO_REG32(    SPI2_SPI_TX0,          0x400D0020,__WRITE      );
__IO_REG32(    SPI2_SPI_TX1,          0x400D0024,__WRITE      );
__IO_REG32(    SPI2_SPI_VARCLK,       0x400D0034,__READ_WRITE );
__IO_REG32_BIT(SPI2_SPI_PDMA,         0x400D0038,__READ_WRITE ,__spix_spi_pdma_bits);
__IO_REG32_BIT(SPI2_SPI_FFCLR,        0x400D003C,__READ_WRITE ,__spix_spi_ffclr_bits);

/***************************************************************************
 **
 ** TMR
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR0_TMRx_CTL0,        0x40010000,__READ_WRITE ,__tmr0_tmrx_ctl0_bits);
__IO_REG32_BIT(TMR0_TMRx_PRECNT0,     0x40010004,__READ_WRITE ,__tmr0_tmrx_precnt0_bits);
__IO_REG32_BIT(TMR0_TMRx_CMPR0,       0x40010008,__READ_WRITE ,__tmr0_tmrx_cmpr0_bits);
__IO_REG32_BIT(TMR0_TMRx_IER0,        0x4001000C,__READ_WRITE ,__tmr0_tmrx_ier0_bits);
__IO_REG32_BIT(TMR0_TMRx_ISR0,        0x40010010,__READ_WRITE ,__tmr0_tmrx_isr0_bits);
__IO_REG32_BIT(TMR0_TMRx_DR0,         0x40010014,__READ       ,__tmr0_tmrx_dr0_bits);
__IO_REG32_BIT(TMR0_TMRx_TCAP0,       0x40010018,__READ       ,__tmr0_tmrx_tcap0_bits);
__IO_REG32_BIT(TMR0_TMRx_CTL1,        0x40010100,__READ_WRITE ,__tmr0_tmrx_ctl1_bits);
__IO_REG32_BIT(TMR0_TMRx_PRECNT1,     0x40010104,__READ_WRITE ,__tmr0_tmrx_precnt1_bits);
__IO_REG32_BIT(TMR0_TMRx_CMPR1,       0x40010108,__READ_WRITE ,__tmr0_tmrx_cmpr1_bits);
__IO_REG32_BIT(TMR0_TNRx_IER1,        0x4001010C,__READ_WRITE ,__tmr0_tnrx_ier1_bits);
__IO_REG32_BIT(TMR0_TMRx_ISR1,        0x40010110,__READ_WRITE ,__tmr0_tmrx_isr1_bits);
__IO_REG32_BIT(TMR0_TMRx_DR1,         0x40010114,__READ       ,__tmr0_tmrx_dr1_bits);
__IO_REG32_BIT(TMR0_TMRx_TCAP1,       0x40010118,__READ       ,__tmr0_tmrx_tcap1_bits);
__IO_REG32_BIT(TMR0_GPA_SHADOW,       0x40010200,__READ       ,__tmr0_gpa_shadow_bits);
__IO_REG32_BIT(TMR0_GPB_SHADOW,       0x40010204,__READ       ,__tmr0_gpb_shadow_bits);
__IO_REG32_BIT(TMR0_GPC_SHADOW,       0x40010208,__READ       ,__tmr0_gpc_shadow_bits);
__IO_REG32_BIT(TMR0_GPD_SHADOW,       0x4001020C,__READ       ,__tmr0_gpd_shadow_bits);
__IO_REG32_BIT(TMR0_GPE_SHADOW,       0x40010210,__READ       ,__tmr0_gpe_shadow_bits);
__IO_REG32_BIT(TMR0_GPF_SHADOW,       0x40010214,__READ       ,__tmr0_gpf_shadow_bits);
__IO_REG32_BIT(TMR1_TMRx_CTL0,        0x40110000,__READ_WRITE ,__tmr1_tmrx_ctl0_bits);
__IO_REG32_BIT(TMR1_TMRx_PRECNT0,     0x40110004,__READ_WRITE ,__tmr1_tmrx_precnt0_bits);
__IO_REG32_BIT(TMR1_TMRx_CMPR0,       0x40110008,__READ_WRITE ,__tmr1_tmrx_cmpr0_bits);
__IO_REG32_BIT(TMR1_TMRx_IER0,        0x4011000C,__READ_WRITE ,__tmr1_tmrx_ier0_bits);
__IO_REG32_BIT(TMR1_TMRx_ISR0,        0x40110010,__READ_WRITE ,__tmr1_tmrx_isr0_bits);
__IO_REG32_BIT(TMR1_TMRx_DR0,         0x40110014,__READ       ,__tmr1_tmrx_dr0_bits);
__IO_REG32_BIT(TMR1_TMRx_TCAP0,       0x40110018,__READ       ,__tmr1_tmrx_tcap0_bits);
__IO_REG32_BIT(TMR1_TMRx_CTL1,        0x40110100,__READ_WRITE ,__tmr1_tmrx_ctl1_bits);
__IO_REG32_BIT(TMR1_TMRx_PRECNT1,     0x40110104,__READ_WRITE ,__tmr1_tmrx_precnt1_bits);
__IO_REG32_BIT(TMR1_TMRx_CMPR1,       0x40110108,__READ_WRITE ,__tmr1_tmrx_cmpr1_bits);
__IO_REG32_BIT(TMR1_TNRx_IER1,        0x4011010C,__READ_WRITE ,__tmr1_tnrx_ier1_bits);
__IO_REG32_BIT(TMR1_TMRx_ISR1,        0x40110110,__READ_WRITE ,__tmr1_tmrx_isr1_bits);
__IO_REG32_BIT(TMR1_TMRx_DR1,         0x40110114,__READ       ,__tmr1_tmrx_dr1_bits);
__IO_REG32_BIT(TMR1_TMRx_TCAP1,       0x40110118,__READ       ,__tmr1_tmrx_tcap1_bits);
__IO_REG32_BIT(TMR1_GPA_SHADOW,       0x40110200,__READ       ,__tmr1_gpa_shadow_bits);
__IO_REG32_BIT(TMR1_GPB_SHADOW,       0x40110204,__READ       ,__tmr1_gpb_shadow_bits);
__IO_REG32_BIT(TMR1_GPC_SHADOW,       0x40110208,__READ       ,__tmr1_gpc_shadow_bits);
__IO_REG32_BIT(TMR1_GPD_SHADOW,       0x4011020C,__READ       ,__tmr1_gpd_shadow_bits);
__IO_REG32_BIT(TMR1_GPE_SHADOW,       0x40110210,__READ       ,__tmr1_gpe_shadow_bits);
__IO_REG32_BIT(TMR1_GPF_SHADOW,       0x40110214,__READ       ,__tmr1_gpf_shadow_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_CTL,           0x40004000,__READ_WRITE ,__wdt_wdt_ctl_bits);
__IO_REG32_BIT(WDT_IE,            0x40004004,__READ_WRITE ,__wdt_wdt_ie_bits);
__IO_REG32_BIT(WDT_ISR,           0x40004008,__READ       ,__wdt_wdt_isr_bits);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0_UARTx_RBR,          0x40050000,__READ_WRITE ,__uart_uartx_rbr_thr_bits);
#define UART0_UARTx_THR     UART0_UARTx_RBR
#define UART0_UARTx_THR_bit UART0_UARTx_RBR_bit
__IO_REG32_BIT(UART0_UARTx_CTL,       0x40050004,__READ_WRITE ,__uart_uartx_ctl_bits);
__IO_REG32_BIT(UART0_UARTx_TLCTL,     0x40050008,__READ_WRITE ,__uart_uartx_tlctl_bits);
__IO_REG32_BIT(UART0_UARTx_IER,       0x4005000C,__READ_WRITE ,__uart_uartx_ier_bits);
__IO_REG32_BIT(UART0_UARTx_ISR,       0x40050010,__READ_WRITE ,__uart_uartx_isr_bits);
__IO_REG32_BIT(UART0_UARTx_TRSR,      0x40050014,__READ_WRITE ,__uart_uartx_trsr_bits);
__IO_REG32_BIT(UART0_UARTx_FSR,       0x40050018,__READ_WRITE ,__uart_uartx_fsr_bits);
__IO_REG32_BIT(UART0_UARTx_MCSR,      0x4005001C,__READ_WRITE ,__uart_uartx_mcsr_bits);
__IO_REG32_BIT(UART0_UARTx_TMCTL,     0x40050020,__READ_WRITE ,__uart_uartx_tmctl_bits);
__IO_REG32_BIT(UART0_UARTx_BAUD,      0x40050024,__READ_WRITE ,__uart_uartx_baud_bits);
__IO_REG32_BIT(UART0_UARTx_WAKE,      0x40050028,__READ_WRITE ,__uart_uartx_wake_bits);
__IO_REG32_BIT(UART0_UARTx_IRCR,      0x40050030,__READ_WRITE ,__uart_uartx_ircr_bits);
__IO_REG32_BIT(UART0_UARTx_ALT_CSR,   0x40050034,__READ_WRITE ,__uart_uartx_alt_csr_bits);
__IO_REG32_BIT(UART0_UARTx_FUN_SEL,   0x40050038,__READ_WRITE ,__uart_uartx_fun_sel_bits);
__IO_REG32_BIT(UART1_UARTx_RBR,          0x40150000,__READ_WRITE ,__uart_uartx_rbr_thr_bits);
#define UART1_UARTx_THR     UART0_UARTx_RBR
#define UART1_UARTx_THR_bit UART0_UARTx_RBR_bit
__IO_REG32_BIT(UART1_UARTx_CTL,       0x40150004,__READ_WRITE ,__uart_uartx_ctl_bits);
__IO_REG32_BIT(UART1_UARTx_TLCTL,     0x40150008,__READ_WRITE ,__uart_uartx_tlctl_bits);
__IO_REG32_BIT(UART1_UARTx_IER,       0x4015000C,__READ_WRITE ,__uart_uartx_ier_bits);
__IO_REG32_BIT(UART1_UARTx_ISR,       0x40150010,__READ_WRITE ,__uart_uartx_isr_bits);
__IO_REG32_BIT(UART1_UARTx_TRSR,      0x40150014,__READ_WRITE ,__uart_uartx_trsr_bits);
__IO_REG32_BIT(UART1_UARTx_FSR,       0x40150018,__READ_WRITE ,__uart_uartx_fsr_bits);
__IO_REG32_BIT(UART1_UARTx_MCSR,      0x4015001C,__READ_WRITE ,__uart_uartx_mcsr_bits);
__IO_REG32_BIT(UART1_UARTx_TMCTL,     0x40150020,__READ_WRITE ,__uart_uartx_tmctl_bits);
__IO_REG32_BIT(UART1_UARTx_BAUD,      0x40150024,__READ_WRITE ,__uart_uartx_baud_bits);
__IO_REG32_BIT(UART1_UARTx_WAKE,      0x40150028,__READ_WRITE ,__uart_uartx_wake_bits);
__IO_REG32_BIT(UART1_UARTx_IRCR,      0x40150030,__READ_WRITE ,__uart_uartx_ircr_bits);
__IO_REG32_BIT(UART1_UARTx_ALT_CSR,   0x40150034,__READ_WRITE ,__uart_uartx_alt_csr_bits);
__IO_REG32_BIT(UART1_UARTx_FUN_SEL,   0x40150038,__READ_WRITE ,__uart_uartx_fun_sel_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2S_CTRL,          0x401A0000,__READ_WRITE ,__i2s_i2s_ctrl_bits);
__IO_REG32_BIT(I2S_CLKDIV,        0x401A0004,__READ_WRITE ,__i2s_i2s_clkdiv_bits);
__IO_REG32_BIT(I2S_INTEN,         0x401A0008,__READ_WRITE ,__i2s_i2s_inten_bits);
__IO_REG32_BIT(I2S_STATUS,        0x401A000C,__READ_WRITE ,__i2s_i2s_status_bits);
__IO_REG32(    I2S_TXFIFO,        0x401A0010,__WRITE      );
__IO_REG32(    I2S_RXFIFO,        0x401A0014,__READ       );

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ADC_RESULT0,       0x400E0000,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT1,       0x400E0004,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT2,       0x400E0008,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT3,       0x400E000C,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT4,       0x400E0010,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT5,       0x400E0014,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT6,       0x400E0018,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT7,       0x400E001C,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT8,       0x400E0020,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT9,       0x400E0024,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADC_RESULT10,      0x400E0028,__READ       ,__adc_adc_resultx_bits);
__IO_REG32_BIT(ADC_ADCR,              0x400E0030,__READ_WRITE ,__adc_adcr_bits);
__IO_REG32_BIT(ADC_ADCHER,            0x400E0034,__READ_WRITE ,__adc_adcher_bits);
__IO_REG32_BIT(ADC_ADCMPR0,           0x400E0038,__READ_WRITE ,__adc_adcmpr0_bits);
__IO_REG32_BIT(ADC_ADCMPR1,           0x400E003C,__READ_WRITE ,__adc_adcmpr1_bits);
__IO_REG32_BIT(ADC_ADSR,              0x400E0040,__READ_WRITE ,__adc_adsr_bits);
__IO_REG32_BIT(ADC_ADFCR,             0x400E0054,__READ_WRITE ,__adc_adfcr_bits);
__IO_REG32_BIT(ADC_ADPDMA,            0x400E0060,__READ       ,__adc_adpdma_bits);
__IO_REG32_BIT(ADC_ADCDELSEL,         0x400E0064,__READ_WRITE ,__adc_adcdelsel_bits);

/***************************************************************************
 **
 ** VDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(VDMA_CSR,         0x50008000,__READ_WRITE ,__vdma_vdma_csr_bits);
__IO_REG32(    VDMA_SAR,         0x50008004,__READ_WRITE );
__IO_REG32(    VDMA_DAR,         0x50008008,__READ_WRITE );
__IO_REG32_BIT(VDMA_BCR,         0x5000800C,__READ_WRITE ,__vdma_vdma_bcr_bits);
__IO_REG32(    VDMA_CSAR,        0x50008014,__READ       );
__IO_REG32(    VDMA_CDAR,        0x50008018,__READ       );
__IO_REG32_BIT(VDMA_CBCR,        0x5000801C,__READ       ,__vdma_vdma_cbcr_bits);
__IO_REG32_BIT(VDMA_IER,         0x50008020,__READ_WRITE ,__vdma_vdma_ier_bits);
__IO_REG32_BIT(VDMA_ISR,         0x50008024,__READ_WRITE ,__vdma_vdma_isr_bits);
__IO_REG32_BIT(VDMA_SASOCR,      0x5000802C,__READ_WRITE ,__vdma_vdma_sasocr_bits);
__IO_REG32_BIT(VDMA_DASOCR,      0x50008030,__READ_WRITE ,__vdma_vdma_dasocr_bits);
__IO_REG32(    VDMA_BUF0,        0x50008080,__READ       );
__IO_REG32(    VDMA_BUF1,        0x50008084,__READ       );

/***************************************************************************
 **
 ** PDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(PDMA1_PDMA_CSRx,       0x50008100,__READ_WRITE ,__pdma1_pdma_csrx_bits);
__IO_REG32(    PDMA1_PDMA_SARx,       0x50008104,__READ_WRITE );
__IO_REG32(    PDMA1_PDMA_DARx,       0x50008108,__READ_WRITE );
__IO_REG32_BIT(PDMA1_PDMA_BCRx,       0x5000810C,__READ_WRITE ,__pdma1_pdma_bcrx_bits);
__IO_REG32(    PDMA1_PDMA_CSARx,      0x50008114,__READ       );
__IO_REG32(    PDMA1_PDMA_CDARx,      0x50008118,__READ       );
__IO_REG32_BIT(PDMA1_PDMA_CBCRx,      0x5000811C,__READ       ,__pdma1_pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA1_PDMA_IERx,       0x50008120,__READ_WRITE ,__pdma1_pdma_ierx_bits);
__IO_REG32_BIT(PDMA1_PDMA_ISRx,       0x50008124,__READ_WRITE ,__pdma1_pdma_isrx_bits);
__IO_REG32_BIT(PDMA1_PDMA_TCRx,       0x50008128,__READ_WRITE ,__pdma1_pdma_tcrx_bits);
__IO_REG32(    PDMA1_PDMA_BUF,        0x50008180,__READ       );
__IO_REG32_BIT(PDMA2_PDMA_CSRx,       0x50008200,__READ_WRITE ,__pdma2_pdma_csrx_bits);
__IO_REG32(    PDMA2_PDMA_SARx,       0x50008204,__READ_WRITE );
__IO_REG32(    PDMA2_PDMA_DARx,       0x50008208,__READ_WRITE );
__IO_REG32_BIT(PDMA2_PDMA_BCRx,       0x5000820C,__READ_WRITE ,__pdma2_pdma_bcrx_bits);
__IO_REG32(    PDMA2_PDMA_CSARx,      0x50008214,__READ       );
__IO_REG32(    PDMA2_PDMA_CDARx,      0x50008218,__READ       );
__IO_REG32_BIT(PDMA2_PDMA_CBCRx,      0x5000821C,__READ       ,__pdma2_pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA2_PDMA_IERx,       0x50008220,__READ_WRITE ,__pdma2_pdma_ierx_bits);
__IO_REG32_BIT(PDMA2_PDMA_ISRx,       0x50008224,__READ_WRITE ,__pdma2_pdma_isrx_bits);
__IO_REG32_BIT(PDMA2_PDMA_TCRx,       0x50008228,__READ_WRITE ,__pdma2_pdma_tcrx_bits);
__IO_REG32(    PDMA2_PDMA_BUF,        0x50008280,__READ       );
__IO_REG32_BIT(PDMA3_PDMA_CSRx,       0x50008300,__READ_WRITE ,__pdma3_pdma_csrx_bits);
__IO_REG32(    PDMA3_PDMA_SARx,       0x50008304,__READ_WRITE );
__IO_REG32(    PDMA3_PDMA_DARx,       0x50008308,__READ_WRITE );
__IO_REG32_BIT(PDMA3_PDMA_BCRx,       0x5000830C,__READ_WRITE ,__pdma3_pdma_bcrx_bits);
__IO_REG32(    PDMA3_PDMA_CSARx,      0x50008314,__READ       );
__IO_REG32(    PDMA3_PDMA_CDARx,      0x50008318,__READ       );
__IO_REG32_BIT(PDMA3_PDMA_CBCRx,      0x5000831C,__READ       ,__pdma3_pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA3_PDMA_IERx,       0x50008320,__READ_WRITE ,__pdma3_pdma_ierx_bits);
__IO_REG32_BIT(PDMA3_PDMA_ISRx,       0x50008324,__READ_WRITE ,__pdma3_pdma_isrx_bits);
__IO_REG32_BIT(PDMA3_PDMA_TCRx,       0x50008328,__READ_WRITE ,__pdma3_pdma_tcrx_bits);
__IO_REG32(    PDMA3_PDMA_BUF,        0x50008380,__READ       );
__IO_REG32_BIT(PDMA4_PDMA_CSRx,       0x50008400,__READ_WRITE ,__pdma4_pdma_csrx_bits);
__IO_REG32(    PDMA4_PDMA_SARx,       0x50008404,__READ_WRITE );
__IO_REG32(    PDMA4_PDMA_DARx,       0x50008408,__READ_WRITE );
__IO_REG32_BIT(PDMA4_PDMA_BCRx,       0x5000840C,__READ_WRITE ,__pdma4_pdma_bcrx_bits);
__IO_REG32(    PDMA4_PDMA_CSARx,      0x50008414,__READ       );
__IO_REG32(    PDMA4_PDMA_CDARx,      0x50008418,__READ       );
__IO_REG32_BIT(PDMA4_PDMA_CBCRx,      0x5000841C,__READ       ,__pdma4_pdma_cbcrx_bits);
__IO_REG32_BIT(PDMA4_PDMA_IERx,       0x50008420,__READ_WRITE ,__pdma4_pdma_ierx_bits);
__IO_REG32_BIT(PDMA4_PDMA_ISRx,       0x50008424,__READ_WRITE ,__pdma4_pdma_isrx_bits);
__IO_REG32_BIT(PDMA4_PDMA_TCRx,       0x50008428,__READ_WRITE ,__pdma4_pdma_tcrx_bits);
__IO_REG32(    PDMA4_PDMA_BUF,        0x50008480,__READ       );

/***************************************************************************
 **
 ** PDMA_GCR
 **
 ***************************************************************************/
__IO_REG32_BIT(PDMA_GCR_PDMA_GCRCSR,  0x50008F00,__READ_WRITE ,__pdma_gcr_pdma_gcrcsr_bits);
__IO_REG32_BIT(PDMA_GCR_PDSSR0,       0x50008F04,__READ_WRITE ,__pdma_gcr_pdssr0_bits);
__IO_REG32_BIT(PDMA_GCR_PDSSR1,       0x50008F08,__READ_WRITE ,__pdma_gcr_pdssr1_bits);
__IO_REG32_BIT(PDMA_GCR_PDMA_GCRISR,  0x50008F0C,__READ       ,__pdma_gcr_pdma_gcrisr_bits);

/***************************************************************************
 **
 ** EBI
 **
 ***************************************************************************/
__IO_REG32_BIT(EBI_EBICON,            0x50010000,__READ_WRITE ,__ebi_ebicon_bits);
__IO_REG32_BIT(EBI_EXTIME,            0x50010004,__READ_WRITE ,__ebi_extime_bits);

/***************************************************************************
 **
 ** TK
 **
 ***************************************************************************/
__IO_REG32_BIT(TK_CTL,             0x400C0000,__READ_WRITE ,__tk_tk_ctl_bits);
__IO_REG32_BIT(TK_STAT,            0x400C0004,__READ       ,__tk_tk_stat_bits);
__IO_REG32_BIT(TK_DATA,            0x400C0008,__READ       ,__tk_tk_data_bits);
__IO_REG32_BIT(TK_INTEN,           0x400C000C,__READ_WRITE ,__tk_tk_inten_bits);

/***************************************************************************
 **
 ** DAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DAC_DAC0_CTL,          0x400A0000,__READ_WRITE ,__dac_dac0_ctl_bits);
__IO_REG32_BIT(DAC_DAC0_DATA,         0x400A0004,__READ_WRITE ,__dac_dac0_data_bits);
__IO_REG32_BIT(DAC_DAC0_STS,          0x400A0008,__READ_WRITE ,__dac_dac0_sts_bits);
__IO_REG32_BIT(DAC_DAC1_CTL,          0x400A0010,__READ_WRITE ,__dac_dac1_ctl_bits);
__IO_REG32_BIT(DAC_DAC1_DATA,         0x400A0014,__READ_WRITE ,__dac_dac1_data_bits);
__IO_REG32_BIT(DAC_DAC1_STS,          0x400A0018,__READ_WRITE ,__dac_dac1_sts_bits);
__IO_REG32_BIT(DAC_DAC01_COMCTL,      0x400A0020,__READ_WRITE ,__dac_dac01_comctl_bits);
__IO_REG32_BIT(DAC_DAC0_FPGA_DAT,     0x400A0030,__READ       ,__dac_dac0_fpga_dat_bits);
__IO_REG32_BIT(DAC_DAC1_FPGA_DAT,     0x400A0034,__READ       ,__dac_dac1_fpga_dat_bits);

/***************************************************************************
 **
 ** SC
 **
 ***************************************************************************/
__IO_REG32_BIT(SC0_SC_RBR,            0x40190000,__READ       ,__scx_sc_rbr_thr_bits);
#define SC0_SC_THR          SC0_SC_RBR
#define SC0_SC_THR_bit      SC0_SC_RBR_bit
__IO_REG32_BIT(SC0_SC_CTL,            0x40190004,__READ_WRITE ,__scx_sc_ctl_bits);
__IO_REG32_BIT(SC0_SC_ALTCTL,         0x40190008,__READ_WRITE ,__scx_sc_altctl_bits);
__IO_REG32_BIT(SC0_SC_EGTR,           0x4019000C,__READ_WRITE ,__scx_sc_egtr_bits);
__IO_REG32_BIT(SC0_SC_RFTMR,          0x40190010,__READ_WRITE ,__scx_sc_rftmr_bits);
__IO_REG32_BIT(SC0_SC_ETUCR,          0x40190014,__READ_WRITE ,__scx_sc_etucr_bits);
__IO_REG32_BIT(SC0_SC_IER,            0x40190018,__READ_WRITE ,__scx_sc_ier_bits);
__IO_REG32_BIT(SC0_SC_ISR,            0x4019001C,__READ_WRITE ,__scx_sc_isr_bits);
__IO_REG32_BIT(SC0_SC_TRSR,           0x40190020,__READ_WRITE ,__scx_sc_trsr_bits);
__IO_REG32_BIT(SC0_SC_PINCSR,         0x40190024,__READ_WRITE ,__scx_sc_pincsr_bits);
__IO_REG32_BIT(SC0_SC_TMR0,           0x40190028,__READ_WRITE ,__scx_sc_tmr0_bits);
__IO_REG32_BIT(SC0_SC_TMR1,           0x4019002C,__READ_WRITE ,__scx_sc_tmr1_bits);
__IO_REG32_BIT(SC0_SC_TMR2,           0x40190030,__READ_WRITE ,__scx_sc_tmr2_bits);
__IO_REG32_BIT(SC0_SC_TDRA,           0x40190038,__READ       ,__scx_sc_tdra_bits);
__IO_REG32_BIT(SC0_SC_TDRB,           0x4019003C,__READ       ,__scx_sc_tdrb_bits);
__IO_REG32_BIT(SC1_SC_RBR,            0x401B0000,__READ       ,__scx_sc_rbr_thr_bits);
#define SC1_SC_THR          SC1_SC_RBR
#define SC1_SC_THR_bit      SC1_SC_RBR_bit
__IO_REG32_BIT(SC1_SC_CTL,            0x401B0004,__READ_WRITE ,__scx_sc_ctl_bits);
__IO_REG32_BIT(SC1_SC_ALTCTL,         0x401B0008,__READ_WRITE ,__scx_sc_altctl_bits);
__IO_REG32_BIT(SC1_SC_EGTR,           0x401B000C,__READ_WRITE ,__scx_sc_egtr_bits);
__IO_REG32_BIT(SC1_SC_RFTMR,          0x401B0010,__READ_WRITE ,__scx_sc_rftmr_bits);
__IO_REG32_BIT(SC1_SC_ETUCR,          0x401B0014,__READ_WRITE ,__scx_sc_etucr_bits);
__IO_REG32_BIT(SC1_SC_IER,            0x401B0018,__READ_WRITE ,__scx_sc_ier_bits);
__IO_REG32_BIT(SC1_SC_ISR,            0x401B001C,__READ_WRITE ,__scx_sc_isr_bits);
__IO_REG32_BIT(SC1_SC_TRSR,           0x401B0020,__READ_WRITE ,__scx_sc_trsr_bits);
__IO_REG32_BIT(SC1_SC_PINCSR,         0x401B0024,__READ_WRITE ,__scx_sc_pincsr_bits);
__IO_REG32_BIT(SC1_SC_TMR0,           0x401B0028,__READ_WRITE ,__scx_sc_tmr0_bits);
__IO_REG32_BIT(SC1_SC_TMR1,           0x401B002C,__READ_WRITE ,__scx_sc_tmr1_bits);
__IO_REG32_BIT(SC1_SC_TMR2,           0x401B0030,__READ_WRITE ,__scx_sc_tmr2_bits);
__IO_REG32_BIT(SC1_SC_TDRA,           0x401B0038,__READ       ,__scx_sc_tdra_bits);
__IO_REG32_BIT(SC1_SC_TDRB,           0x401B003C,__READ       ,__scx_sc_tdrb_bits);

/***************************************************************************
 **
 ** FMC
 **
 ***************************************************************************/
__IO_REG32_BIT(FMC_ISPCON,            0x5000C000,__READ_WRITE ,__fmc_ispcon_bits);
__IO_REG32(    FMC_ISPADR,            0x5000C004,__READ_WRITE );
__IO_REG32(    FMC_ISPDAT,            0x5000C008,__READ_WRITE );
__IO_REG32_BIT(FMC_ISPCMD,            0x5000C00C,__READ_WRITE ,__fmc_ispcmd_bits);
__IO_REG32_BIT(FMC_ISPTRG,            0x5000C010,__READ_WRITE ,__fmc_isptrg_bits);
__IO_REG32(    FMC_DFBADR,            0x5000C014,__READ       );
__IO_REG32_BIT(FMC_FATCON,            0x5000C018,__READ_WRITE ,__fmc_fatcon_bits);
__IO_REG32_BIT(FMC_ICPCON,            0x5000C01C,__READ_WRITE ,__fmc_icpcon_bits);
__IO_REG32_BIT(FMC_RMPCON,            0x5000C020,__READ_WRITE ,__fmc_rmpcon_bits);
__IO_REG32_BIT(FMC_ICECON,            0x5000C024,__READ_WRITE ,__fmc_icecon_bits);
__IO_REG32_BIT(FMC_ISPSTA,            0x5000C040,__READ_WRITE ,__fmc_ispsta_bits);



/***************************************************************************
 **  Assembler specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  NVIC Interrupt channels
 **
 ***************************************************************************/
#define MAIN_STACK             0  /* Main Stack                                             */
#define RESETI                 1  /* Reset                                                  */
#define NMII                   2  /* Non-maskable Interrupt                                 */
#define HFI                    3  /* Hard Fault                                             */
#define SVCI                  11  /* SVCall                                                 */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_BOD_OUT          16  /* Brownout low voltage detected interrupt                */
#define NVIC_WDT_INT          17  /* Watch Dog Timer interrupt                              */
#define NVIC_EINT0            18  /* External signal interrupt from P3.2 pin                */
#define NVIC_EINT1            19  /* External signal interrupt from P3.3 pin                */
#define NVIC_GPABC_INT        20  /* External signal interrupt from P0[7:0] / P1[7:0]       */
#define NVIC_GPDEF_INT        21  /* External interrupt from P2[7:0]/P3[7:0]/P4[7:0] except P32 & P33 */
#define NVIC_PWM0_INT         22  /* PWM0,PWM1,PWM2 and PWM3 interrupt                      */
#define NVIC_PWM1_INT         22  /* PWM4,PWM5,PWM6 and PWM7 interrupt                      */
#define NVIC_TMR0_INT         24  /* Timer 0 interrupt                                      */
#define NVIC_TMR1_INT         25  /* Timer 1 interrupt                                      */
#define NVIC_TMR2_INT         26  /* Timer 2 interrupt                                      */
#define NVIC_TMR3_INT         27  /* Timer 3 interrupt                                      */
#define NVIC_UART0_INT        28  /* UART0 interrupt                                        */
#define NVIC_UART1_INT        29  /* UART1 interrupt                                        */
#define NVIC_SPI0_INT         30  /* SPI0 interrupt                                         */
#define NVIC_SPI1_INT         31  /* SPI0 interrupt                                         */
#define NVIC_SPI2_INT         32  /* SPI0 interrupt                                         */
#define NVIC_HFIRC_TRIM_INT   33  /* HFIRC_TRIM interrupt                                   */
#define NVIC_I2C0_INT         34  /* I2C interrupt                                          */
#define NVIC_I2C1_INT         35  /* I2C interrupt                                          */
#define NVIC_SC0_INT          37  /* SC0 interrupt                                          */
#define NVIC_SC1_INT          38  /* SC0 interrupt                                          */
#define NVIC_TK_INT           40  /* TK interrupt                                           */
#define NVIC_DMA_INT          42  /* DMA interrupt                                          */
#define NVIC_I2S_INT          43  /* I2S interrupt                                          */
#define NVIC_PD_WU_INT        44  /* Clock controller interrupt for chip wake up from power-down state */
#define NVIC_ADC_INT          45  /* ADC interrupt                                          */
#define NVIC_DAC_INT          46  /* DAC interrupt                                          */
#define NVIC_RTC_INT          47  /* RTC interrupt                                          */

#endif    /* __IONANO100_REGISTERS_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = SVC            0x2C
Interrupt3   = PendSV         0x38
Interrupt4   = SysTick        0x3C
Interrupt5   = BOD_OUT        0x40
Interrupt6   = WDT_INT        0x44
Interrupt7   = EINT0          0x48
Interrupt8   = EINT1          0x4C
Interrupt9   = GPABC_INT      0x50
Interrupt10  = GPDEF_INT      0x54
Interrupt11  = PWM0_INT       0x58
Interrupt12  = PWM1_INT       0x5C
Interrupt13  = TMR0_INT       0x60
Interrupt14  = TMR1_INT       0x64
Interrupt15  = TMR2_INT       0x68
Interrupt16  = TMR3_INT       0x6C
Interrupt17  = UART0_INT      0x70
Interrupt18  = UART1_INT      0x74
Interrupt19  = SPI0_INT       0x78
Interrupt20  = SPI1_INT       0x7C
Interrupt21  = SPI2_INT       0x80
Interrupt22  = IRC_INT        0x84
Interrupt23  = I2C0_INT       0x88
Interrupt24  = I2C1_INT       0x8C
Interrupt25  = SC0_INT        0x94
Interrupt26  = SC1_INT        0x98
Interrupt27  = TK_INT         0xA0
Interrupt28  = DMA_INT        0xA8
Interrupt29  = I2S_INT        0xAC
Interrupt30  = PD_WU_INT      0xB0
Interrupt31  = ADC_INT        0xB4
Interrupt32  = DAC_INT        0xB8
Interrupt33  = RTC_INT        0xBC
###DDF-INTERRUPT-END###*/