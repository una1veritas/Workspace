/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton Mini51 Devices
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46179 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOMINI51_H
#define __IOMINI51_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   MINI51 SPECIAL FUNCTION REGISTERS
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
  __REG32 RSTS_RESET      : 1;
  __REG32 RSTS_WDT        : 1;
  __REG32                 : 1;
  __REG32 RSTS_BOD        : 1;
  __REG32 RSTS_MCU        : 1;
  __REG32                 : 1;
  __REG32 RSTS_CPU        : 1;
  __REG32                 : 24;
} __gcr_rstsrc_bits;

/* IP Reset Control Resister1 */
typedef struct {
  __REG32 CHIP_RST        : 1;
  __REG32 CPU_RST         : 1;
  __REG32                 : 30;
} __gcr_iprstc1_bits;

/* IP Reset Control Resister 2 */
typedef struct {
  __REG32                 : 1;
  __REG32 GPIO_RST        : 1;
  __REG32 TMR0_RST        : 1;
  __REG32 TMR1_RST        : 1;
  __REG32                 : 4;
  __REG32 I2C_RST         : 1;
  __REG32                 : 3;
  __REG32 SPI_RST         : 1;
  __REG32                 : 3;
  __REG32 UART_RST        : 1;
  __REG32                 : 3;
  __REG32 PWM_RST         : 1;
  __REG32                 : 1;
  __REG32 ACMP_RST        : 1;
  __REG32                 : 5;
  __REG32 ADC_RST         : 1;
  __REG32                 : 3;
} __gcr_iprstc2_bits;

/* Internal Test Controller Register */
typedef struct {
  __REG32 SELF_TEST       : 1;
  __REG32 DELAY_TEST      : 1;
  __REG32                 : 6;
  __REG32                 : 24;
} __gcr_itestcr_bits;

/* Brown-Out Detector Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 BOD_VL          : 2;
  __REG32 BOD_RSTEN       : 1;
  __REG32 BOD_INTF        : 1;
  __REG32                 : 1;
  __REG32 BOD_OUT         : 1;
  __REG32                 : 9;
  __REG32 BOD27_TRIM_0_3  : 4;
  __REG32 BOD38_TRIM_0_3  : 4;
  __REG32                 : 8;
} __gcr_bodcr_bits;

/* LDO control Register */
typedef struct {
  __REG32 LDO_TRIM_0_4    : 5;
  __REG32                 : 27;
} __gcr_ldocr_bits;

/* Power-On-Reset Controller Register */
typedef struct {
  __REG32 POR_DIS_CODE    : 16;
  __REG32                 : 16;
} __gcr_porcr_bits;

/* P0 multiple function and input type control register */
typedef struct {
  __REG32 P0_MFP_0_7      : 8;
  __REG32 P0_ALT_0        : 1;
  __REG32 P0_ALT_1        : 1;
  __REG32                 : 2;
  __REG32 P0_ALT_4        : 1;
  __REG32 P0_ALT_5        : 1;
  __REG32 P0_ALT_6        : 1;
  __REG32 P0_ALT_7        : 1;
  __REG32 P0_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p0_mfp_bits;

/* P1 multiple function and input type control register */
typedef struct {
  __REG32 P1_MFP_0_7      : 8;
  __REG32 P1_ALT_0        : 1;
  __REG32                 : 1;
  __REG32 P1_ALT_2        : 1;
  __REG32 P1_ALT_3        : 1;
  __REG32 P1_ALT_4        : 1;
  __REG32 P1_ALT_5        : 1;
  __REG32                 : 2;
  __REG32 P1_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p1_mfp_bits;

/* P2 multiple function and input type control register */
typedef struct {
  __REG32 P2_MFP_0_7      : 8;
  __REG32                 : 2;
  __REG32 P2_ALT_2        : 1;
  __REG32 P2_ALT_3        : 1;
  __REG32 P2_ALT_4        : 1;
  __REG32 P2_ALT_5        : 1;
  __REG32 P2_ALT_6        : 1;
  __REG32                 : 1;
  __REG32 P2_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p2_mfp_bits;

/* P3 multiple function and input type control register */
typedef struct {
  __REG32 P3_MFP_0_7      : 8;
  __REG32 P3_ALT_0        : 1;
  __REG32 P3_ALT_1        : 1;
  __REG32 P3_ALT_2        : 1;
  __REG32                 : 1;
  __REG32 P3_ALT_4        : 1;
  __REG32 P3_ALT_5        : 1;
  __REG32 P3_ALT_6        : 1;
  __REG32                 : 1;
  __REG32 P3_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p3_mfp_bits;

/* P4 multiple function and input type control register */
typedef struct {
  __REG32 P4_MFP_0_7      : 8;
  __REG32                 : 6;
  __REG32 P4_ALT_6        : 1;
  __REG32 P4_ALT_7        : 1;
  __REG32 P4_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p4_mfp_bits;

/* P5 multiple function and input type control register */
typedef struct {
  __REG32 P5_MFP_0_7      : 8;
  __REG32 P5_ALT_0        : 1;
  __REG32 P5_ALT_1        : 1;
  __REG32 P5_ALT_2        : 1;
  __REG32 P5_ALT_3        : 1;
  __REG32 P5_ALT_4        : 1;
  __REG32 P5_ALT_5        : 1;
  __REG32                 : 2;
  __REG32 P5_TYPEn        : 8;
  __REG32                 : 8;
} __gcr_p5_mfp_bits;

/* HFIRC Trim Control Register */
typedef struct {
  __REG32 TRIM_SEL        : 2;
  __REG32                 : 2;
  __REG32 TRIM_LOOP       : 2;
  __REG32 TRIM_RETRY_CNT  : 2;
  __REG32                 : 24;
} __gcr_irctrimctl_bits;

/* HFIRC Trim Interrupt Enable Register */
typedef struct {
  __REG32                 : 1;
  __REG32 TRIM_FAIL_IEN   : 1;
  __REG32 _32K_ERR_IEN    : 1;
  __REG32                 : 29;
} __gcr_irctrimien_bits;

/* HFIRC Trim Interrupt Status Register */
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
  __REG32 RCADJ           : 7;
  __REG32                 : 25;
} __gcr_rcadj_bits;

/* SysTick Control and Status */
typedef struct {
  __REG32 ENABLE          : 1;
  __REG32 TICKINT         : 1;
  __REG32 CLKSRC          : 1;
  __REG32                 : 13;
  __REG32 COUNTFLAG       : 1;
  __REG32                 : 15;
} __scs_syst_csr_bits;

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

/* Application Interrupt and Reset Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 VECTCLRACTIVE   : 1;
  __REG32 SYSRESETREQ     : 1;
  __REG32                 : 13;
  __REG32 VECTORKEY       : 16;
} __scs_aircr_bits;

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
/* MCU IRQ1 (WDT) interruptinterrupt source identifyidentify */
/* MCU IRQ2 ((EINT0) interrupt source identify */
/* MCU IRQ3 (EINT1) interrupt source identify */
/* MCU IRQ4 (P0/1) interrupt source identify */
/* MCU IRQ5 (P2/3/4) interrupt source identify */
/* MCU IRQ6 (PWM) interrupt source identify */
/* MCU IRQ7(BRAKE) interrupt source identify */
/* MCU IRQ8 (TMR0) interrupt source identify */
/* MCU IRQ9 (TMR1) interrupt source identify */
/* MCU IRQ12 (UART) interrupt source identify */
/* MCU IRQ14 (SPI) interrupt source identify */
/* MCU IRQ16 (P5) interrupt source identify */
/* MCU IRQ17(IRQ17 (HFIRC trim) interrupt source identify */
/* MCU IRQ18 (I2C) interrupt source identify */
/* MCU IRQ25(ACMP) interrupt source identify- */
/* MCU IRQ28 (PWRWU) interrupt source identify */
/* MCU IRQ29 (ADC) interrupt source identify */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irqx_src_bits;

/* NMI source interrupt select control register */
typedef struct {
  __REG32 NMI_SEL         : 5;
  __REG32                 : 2;
  __REG32 INT_TEST        : 1;
  __REG32 NMI_SEL_EN      : 1;
  __REG32                 : 23;
} __int_nmi_sel_bits;

/* System Power Down Control Register */
typedef struct {
  __REG32 XTLCLK_EN       : 2;
  __REG32 OSC22M_EN       : 1;
  __REG32 OSC10K_EN       : 1;
  __REG32 WU_DLY          : 1;
  __REG32 PD_WU_INT_EN    : 1;
  __REG32 PD_WU_STS       : 1;
  __REG32 PWR_DOWN_EN     : 1;
  __REG32                 : 1;
  __REG32 PD_32K          : 1;
  __REG32                 : 22;
} __clk_pwrcon_bits;

/* AHB Devices Clock Enable Control Register */
typedef struct {
  __REG32 CPU_EN          : 1;
  __REG32                 : 1;
  __REG32 ISP_EN          : 1;
  __REG32                 : 1;
  __REG32                 : 28;
} __clk_ahbclk_bits;

/* APB Devices Clock Enable Control Register */
typedef struct {
  __REG32 WDCLK_EN        : 1;
  __REG32                 : 1;
  __REG32 TMR0_EN         : 1;
  __REG32 TMR1_EN         : 1;
  __REG32                 : 2;
  __REG32 FDIV_EN         : 1;
  __REG32                 : 1;
  __REG32 I2C_EN          : 1;
  __REG32                 : 3;
  __REG32 SPI_EN          : 1;
  __REG32                 : 3;
  __REG32 UART_EN         : 1;
  __REG32                 : 3;
  __REG32 PWM01_EN        : 1;
  __REG32 PWM23_EN        : 1;
  __REG32 PWM45_EN        : 1;
  __REG32                 : 5;
  __REG32 ADC_EN          : 1;
  __REG32                 : 1;
  __REG32 CMP_EN          : 1;
  __REG32                 : 1;
} __clk_apbclk_bits;

/* Clock status monitor Register */
typedef struct {
  __REG32 XTL_STB         : 1;
  __REG32                 : 2;
  __REG32 OSC10K_STB      : 1;
  __REG32 OSC22M_STB      : 1;
  __REG32                 : 2;
  __REG32 CLK_SW_FAIL     : 1;
  __REG32                 : 24;
} __clk_clkstatus_bits;

/* Clock Source Select Control Register 0 */
typedef struct {
  __REG32 HCLK_S          : 3;
  __REG32 STCLK_S         : 3;
  __REG32                 : 26;
} __clk_clksel0_bits;

/* Clock Source Select Control Register 1 */
typedef struct {
  __REG32 WDT_S           : 2;
  __REG32 ADC_S           : 2;
  __REG32                 : 4;
  __REG32 TMR0_S          : 3;
  __REG32                 : 1;
  __REG32 TMR1_S          : 3;
  __REG32                 : 9;
  __REG32 UART_S          : 2;
  __REG32                 : 2;
  __REG32 PWM01_S         : 2;
  __REG32 PWM23_S         : 2;
} __clk_clksel1_bits;

/* Clock Source Select Control Register 2 */
typedef struct {
  __REG32                 : 2;
  __REG32 FRQDIV_S        : 2;
  __REG32 PWM45_S         : 2;
  __REG32                 : 26;
} __clk_clksel2_bits;

/* Clock Divider Number Register */
typedef struct {
  __REG32 HCLK_N          : 4;
  __REG32                 : 4;
  __REG32 UART_N          : 4;
  __REG32                 : 4;
  __REG32 ADC_N           : 8;
  __REG32                 : 8;
} __clk_clkdiv_bits;

/* Frequency Divider Control Register */
typedef struct {
  __REG32 FSEL            : 4;
  __REG32 DIVIDER_EN      : 1;
  __REG32                 : 27;
} __clk_frqdiv_bits;

/* P0 Bit Mode Control */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p0_pmd_bits;

/* P0 Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p0_offd_bits;

/* P0 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p0_dout_bits;

/* P0 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p0_dmask_bits;

/* P0 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p0_pin_bits;

/* P0 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p0_dben_bits;

/* P0 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p0_imd_bits;

/* P0 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p0_ien_bits;

/* P0 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p0_isrc_bits;

/* P1 Bit Mode Enable */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p1_pmd_bits;

/* P1 Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p1_offd_bits;

/* P1 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p1_dout_bits;

/* P1 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p1_dmask_bits;

/* P1 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p1_pin_bits;

/* P1 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p1_dben_bits;

/* P1 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p1_imd_bits;

/* P1 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p1_ien_bits;

/* P1 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p1_isrc_bits;

/* P2 Bit Mode Enable */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p2_pmd_bits;

/* P2 Bit OFF digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p2_offd_bits;

/* P2 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p2_dout_bits;

/* P2 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p2_dmask_bits;

/* P2 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p2_pin_bits;

/* P2 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p2_dben_bits;

/* P2 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p2_imd_bits;

/* P2 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p2_ien_bits;

/* P2 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p2_isrc_bits;

/* P3 Bit Mode Enable */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p3_pmd_bits;

/* P3 Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p3_offd_bits;

/* P3 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p3_dout_bits;

/* P3 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p3_dmask_bits;

/* P3 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p3_pin_bits;

/* P3 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p3_dben_bits;

/* P3 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p3_imd_bits;

/* P3 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p3_ien_bits;

/* P3 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p3_isrc_bits;

/* P4 Bit Mode Enable */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p4_pmd_bits;

/* P4 Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p4_offd_bits;

/* P4 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p4_dout_bits;

/* P4 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p4_dmask_bits;

/* P4 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p4_pin_bits;

/* P4 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p4_dben_bits;

/* P4 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p4_imd_bits;

/* P4 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p4_ien_bits;

/* P4 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p4_isrc_bits;

/* P5 Bit Mode Control */
typedef struct {
  __REG32 PMD0            : 2;
  __REG32 PMD1            : 2;
  __REG32 PMD2            : 2;
  __REG32 PMD3            : 2;
  __REG32 PMD4            : 2;
  __REG32 PMD5            : 2;
  __REG32 PMD6            : 2;
  __REG32 PMD7            : 2;
  __REG32                 : 16;
} __gpio_p5_pmd_bits;

/* P5 Bit OFF Digital Enable */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_p5_offd_bits;

/* P5 Data Output Value */
typedef struct {
  __REG32 DOUT0           : 1;
  __REG32 DOUT1           : 1;
  __REG32 DOUT2           : 1;
  __REG32 DOUT3           : 1;
  __REG32 DOUT4           : 1;
  __REG32 DOUT5           : 1;
  __REG32 DOUT6           : 1;
  __REG32 DOUT7           : 1;
  __REG32                 : 24;
} __gpio_p5_dout_bits;

/* P5 Data Output Write Mask */
typedef struct {
  __REG32 DMASK0          : 1;
  __REG32 DMASK1          : 1;
  __REG32 DMASK2          : 1;
  __REG32 DMASK3          : 1;
  __REG32 DMASK4          : 1;
  __REG32 DMASK5          : 1;
  __REG32 DMASK6          : 1;
  __REG32 DMASK7          : 1;
  __REG32                 : 24;
} __gpio_p5_dmask_bits;

/* P5 Pin Value */
typedef struct {
  __REG32 PIN0            : 1;
  __REG32 PIN1            : 1;
  __REG32 PIN2            : 1;
  __REG32 PIN3            : 1;
  __REG32 PIN4            : 1;
  __REG32 PIN5            : 1;
  __REG32 PIN6            : 1;
  __REG32 PIN7            : 1;
  __REG32                 : 24;
} __gpio_p5_pin_bits;

/* P5 De-bounce Enable */
typedef struct {
  __REG32 DBEN0           : 1;
  __REG32 DBEN1           : 1;
  __REG32 DBEN2           : 1;
  __REG32 DBEN3           : 1;
  __REG32 DBEN4           : 1;
  __REG32 DBEN5           : 1;
  __REG32 DBEN6           : 1;
  __REG32 DBEN7           : 1;
  __REG32                 : 24;
} __gpio_p5_dben_bits;

/* P5 Interrupt Mode Control */
typedef struct {
  __REG32 IMD0            : 1;
  __REG32 IMD1            : 1;
  __REG32 IMD2            : 1;
  __REG32 IMD3            : 1;
  __REG32 IMD4            : 1;
  __REG32 IMD5            : 1;
  __REG32 IMD6            : 1;
  __REG32 IMD7            : 1;
  __REG32                 : 24;
} __gpio_p5_imd_bits;

/* P5 Interrupt Enable */
typedef struct {
  __REG32 IF_EN0          : 1;
  __REG32 IF_EN1          : 1;
  __REG32 IF_EN2          : 1;
  __REG32 IF_EN3          : 1;
  __REG32 IF_EN4          : 1;
  __REG32 IF_EN5          : 1;
  __REG32 IF_EN6          : 1;
  __REG32 IF_EN7          : 1;
  __REG32                 : 8;
  __REG32 IR_EN0          : 1;
  __REG32 IR_EN1          : 1;
  __REG32 IR_EN2          : 1;
  __REG32 IR_EN3          : 1;
  __REG32 IR_EN4          : 1;
  __REG32 IR_EN5          : 1;
  __REG32 IR_EN6          : 1;
  __REG32 IR_EN7          : 1;
  __REG32                 : 8;
} __gpio_p5_ien_bits;

/* P5 Interrupt Source Flag */
typedef struct {
  __REG32 ISRC0           : 1;
  __REG32 ISRC1           : 1;
  __REG32 ISRC2           : 1;
  __REG32 ISRC3           : 1;
  __REG32 ISRC4           : 1;
  __REG32 ISRC5           : 1;
  __REG32 ISRC6           : 1;
  __REG32 ISRC7           : 1;
  __REG32                 : 24;
} __gpio_p5_isrc_bits;

/* De-bounce Cycle Control */
typedef struct {
  __REG32 DBCLKSEL        : 4;
  __REG32 DBCLKSRC        : 1;
  __REG32 ICLK_ON         : 1;
  __REG32                 : 26;
} __gpio_dbncecon_bits;

/* PY Pin I/O Bit Output/Input Control.
For PX, X=0,1,4,5,6,7 */
typedef struct {
  __REG32 DOUT            : 1;
  __REG32                 : 31;
} __gpio_pxy_dout_bits;

/* I2C Control Register */
typedef struct {
  __REG32                 : 2;
  __REG32 AA              : 1;
  __REG32 SI              : 1;
  __REG32 STO             : 1;
  __REG32 STA             : 1;
  __REG32 ENSI            : 1;
  __REG32 EI              : 1;
  __REG32                 : 24;
} __i2c_i2con_bits;

/* I2C DATA Register */
typedef struct {
  __REG32 I2CDAT          : 8;
  __REG32                 : 24;
} __i2c_i2cdat_bits;

/* I2C Status Register */
typedef struct {
  __REG32 I2CSTATUS       : 8;
  __REG32                 : 24;
} __i2c_i2cstatus_bits;

/* I2C clock divided Register */
typedef struct {
  __REG32 I2CLK           : 8;
  __REG32                 : 24;
} __i2c_i2clk_bits;

/* I2C Time out control Register */
typedef struct {
  __REG32 TIF             : 1;
  __REG32 DIV4            : 1;
  __REG32 ENTI            : 1;
  __REG32                 : 29;
} __i2c_i2ctoc_bits;

/* I2C slave Address Register0 */
typedef struct {
  __REG32 GC              : 1;
  __REG32 I2CADDR         : 7;
  __REG32                 : 24;
} __i2c_i2caddr0_bits;

/* Slave address Register X */
typedef struct {
  __REG32 GC              : 1;
  __REG32 I2CADDR         : 7;
  __REG32                 : 24;
} __i2c_i2caddrx_bits;

/* Slave address Mask Register X */
typedef struct {
  __REG32                 : 1;
  __REG32 I2CADMx         : 7;
  __REG32                 : 24;
} __i2c_i2cadmx_bits;

/* PWM Pre-scale Register */
typedef struct {
  __REG32 CP01            : 8;
  __REG32 CP23            : 8;
  __REG32 CP45            : 8;
  __REG32                 : 8;
} __pwm_ppr_bits;

/* PWM Clock Select Register */
typedef struct {
  __REG32 CSR0            : 3;
  __REG32                 : 1;
  __REG32 CSR1            : 3;
  __REG32                 : 1;
  __REG32 CSR2            : 3;
  __REG32                 : 1;
  __REG32 CSR3            : 3;
  __REG32                 : 1;
  __REG32 CSR4            : 3;
  __REG32                 : 1;
  __REG32 CSR5            : 3;
  __REG32                 : 9;
} __pwm_csr_bits;

/* PWM Control Register */
typedef struct {
  __REG32 CH0EN           : 1;
  __REG32 DB_MODE         : 1;
  __REG32 CH0INV          : 1;
  __REG32 CH0MOD          : 1;
  __REG32 CH1EN           : 1;
  __REG32                 : 1;
  __REG32 CH1INV          : 1;
  __REG32 CH1MOD          : 1;
  __REG32 CH2EN           : 1;
  __REG32                 : 1;
  __REG32 CH2INV          : 1;
  __REG32 CH2MOD          : 1;
  __REG32 CH3EN           : 1;
  __REG32                 : 1;
  __REG32 CH3INV          : 1;
  __REG32 CH3MOD          : 1;
  __REG32 CH4EN           : 1;
  __REG32                 : 1;
  __REG32 CH4INV          : 1;
  __REG32 CH4MOD          : 1;
  __REG32 CH5EN           : 1;
  __REG32                 : 1;
  __REG32 CH5INV          : 1;
  __REG32 CH5MOD          : 1;
  __REG32 DZEN01          : 1;
  __REG32 DZEN23          : 1;
  __REG32 DZEN45          : 1;
  __REG32 CLRPWM          : 1;
  __REG32 PWMMOD_0_1      : 2;
  __REG32 GRP             : 1;
  __REG32 PWMTYPE         : 1;
} __pwm_pcr_bits;

/* PWM Counter Register 0 */
typedef struct {
  __REG32 CNR0            : 16;
  __REG32                 : 16;
} __pwm_cnr0_bits;

/* PWM Counter Register 1 */
typedef struct {
  __REG32 CNR1            : 16;
  __REG32                 : 16;
} __pwm_cnr1_bits;

/* PWM Counter Register 2 */
typedef struct {
  __REG32 CNR2            : 16;
  __REG32                 : 16;
} __pwm_cnr2_bits;

/* PWM Counter Register 3 */
typedef struct {
  __REG32 CNR3            : 16;
  __REG32                 : 16;
} __pwm_cnr3_bits;

/* PWM Counter Register 4 */
typedef struct {
  __REG32 CNR4            : 16;
  __REG32                 : 16;
} __pwm_cnr4_bits;

/* PWM Counter Register 5 */
typedef struct {
  __REG32 CNR5            : 16;
  __REG32                 : 16;
} __pwm_cnr5_bits;

/* PWM Comparator Register 0 */
typedef struct {
  __REG32 CMR0            : 16;
  __REG32                 : 16;
} __pwm_cmr0_bits;

/* PWM Comparator Register 1 */
typedef struct {
  __REG32 CMR1            : 16;
  __REG32                 : 16;
} __pwm_cmr1_bits;

/* PWM Comparator Register 2 */
typedef struct {
  __REG32 CMR2            : 16;
  __REG32                 : 16;
} __pwm_cmr2_bits;

/* PWM Comparator Register 3 */
typedef struct {
  __REG32 CMR3            : 16;
  __REG32                 : 16;
} __pwm_cmr3_bits;

/* PWM Comparator Register 4 */
typedef struct {
  __REG32 CMR4            : 16;
  __REG32                 : 16;
} __pwm_cmr4_bits;

/* PWM Comparator Register 5 */
typedef struct {
  __REG32 CMR5            : 16;
  __REG32                 : 16;
} __pwm_cmr5_bits;

/* PWM Data Register 0 */
typedef struct {
  __REG32 PDR0            : 16;
  __REG32                 : 16;
} __pwm_pdr0_bits;

/* PWM Data Register 1 */
typedef struct {
  __REG32 PDR1            : 16;
  __REG32                 : 16;
} __pwm_pdr1_bits;

/* PWM Data Register 2 */
typedef struct {
  __REG32 PDR2            : 16;
  __REG32                 : 16;
} __pwm_pdr2_bits;

/* PWM Data Register 3 */
typedef struct {
  __REG32 PDR3            : 16;
  __REG32                 : 16;
} __pwm_pdr3_bits;

/* PWM Data Register 4 */
typedef struct {
  __REG32 PDR4            : 16;
  __REG32                 : 16;
} __pwm_pdr4_bits;

/* PWM Data Register 5 */
typedef struct {
  __REG32 PDR5            : 16;
  __REG32                 : 16;
} __pwm_pdr5_bits;

/* PWM Interrupt Enable Register */
typedef struct {
  __REG32 PWMPIE0         : 1;
  __REG32 PWMPIE1         : 1;
  __REG32 PWMPIE2         : 1;
  __REG32 PWMPIE3         : 1;
  __REG32 PWMPIE4         : 1;
  __REG32 PWMPIE5         : 1;
  __REG32                 : 2;
  __REG32 PWMDIE0         : 1;
  __REG32 PWMDIE1         : 1;
  __REG32 PWMDIE2         : 1;
  __REG32 PWMDIE3         : 1;
  __REG32 PWMDIE4         : 1;
  __REG32 PWMDIE5         : 1;
  __REG32                 : 2;
  __REG32 BRKIE           : 1;
  __REG32 INT_TYPE        : 1;
  __REG32                 : 14;
} __pwm_pier_bits;

/* PWM Interrupt Indication Register */
typedef struct {
  __REG32 PWMPIF0         : 1;
  __REG32 PWMPIF1         : 1;
  __REG32 PWMPIF2         : 1;
  __REG32 PWMPIF3         : 1;
  __REG32 PWMPIF4         : 1;
  __REG32 PWMPIF5         : 1;
  __REG32                 : 2;
  __REG32 PWMDIF0         : 1;
  __REG32 PWMDIF1         : 1;
  __REG32 PWMDIF2         : 1;
  __REG32 PWMDIF3         : 1;
  __REG32 PWMDIF4         : 1;
  __REG32 PWMDIF5         : 1;
  __REG32                 : 2;
  __REG32 BKF0            : 1;
  __REG32 BKF1            : 1;
  __REG32                 : 14;
} __pwm_piir_bits;

/* PWM Output Control Register for channel 0~5 */
typedef struct {
  __REG32 PWM0            : 1;
  __REG32 PWM1            : 1;
  __REG32 PWM2            : 1;
  __REG32 PWM3            : 1;
  __REG32 PWM4            : 1;
  __REG32 PWM5            : 1;
  __REG32                 : 26;
} __pwm_poe_bits;

/* PWM Fault Brake control Register */
typedef struct {
  __REG32 BKEN0           : 1;
  __REG32 BKEN1           : 1;
  __REG32 CPO0BKEN        : 1;
  __REG32                 : 4;
  __REG32 BKF             : 1;
  __REG32                 : 16;
  __REG32 PWMBKO0         : 1;
  __REG32 PWMBKO1         : 1;
  __REG32 PWMBKO2         : 1;
  __REG32 PWMBKO3         : 1;
  __REG32 PWMBKO4         : 1;
  __REG32 PWMBKO5         : 1;
  __REG32                 : 2;
} __pwm_pfbcon_bits;

/* PWM dead-zone interval register */
typedef struct {
  __REG32 DZI01           : 8;
  __REG32 DZI23           : 8;
  __REG32 DZI45           : 8;
  __REG32                 : 8;
} __pwm_pdzir_bits;

/* Control and Status Register */
typedef struct {
  __REG32 GO_BUSY         : 1;
  __REG32 Rx_NEG          : 1;
  __REG32 Tx_NEG          : 1;
  __REG32 Tx_BIT_LEN      : 5;
  __REG32 Tx_NUM          : 2;
  __REG32 LSB             : 1;
  __REG32 CLKP            : 1;
  __REG32 SP_CYCLE        : 4;
  __REG32 IF              : 1;
  __REG32 IE              : 1;
  __REG32 SLAVE           : 1;
  __REG32 REORDER         : 2;
  __REG32 FIFO            : 1;
  __REG32                 : 1;
  __REG32 VARCLK_EN       : 1;
  __REG32 RX_EMPTY        : 1;
  __REG32 RX_FULL         : 1;
  __REG32 TX_EMPTY        : 1;
  __REG32 TX_FULL         : 1;
  __REG32                 : 4;
} __spi_spi_cntrl_bits;

/* Clock Divider Register */
typedef struct {
  __REG32 DIVIDER         : 16;
  __REG32 DIVIDER2        : 16;
} __spi_spi_divider_bits;

/* Slave Select Register */
typedef struct {
  __REG32 SSR             : 1;
  __REG32                 : 1;
  __REG32 SS_LVL          : 1;
  __REG32 AUTOSS          : 1;
  __REG32 SS_LTRIG        : 1;
  __REG32 LTRIG_FLAG      : 1;
  __REG32                 : 26;
} __spi_spi_ssr_bits;

/* The second Control and Status Register 2 */
typedef struct {
  __REG32 DIV_ONE         : 1;
  __REG32                 : 7;
  __REG32 NOSLVSEL        : 1;
  __REG32 SLV_ABORT       : 1;
  __REG32 SSTA_INTEN      : 1;
  __REG32 SLV_START_INTSTS: 1;
  __REG32                 : 12;
  __REG32                 : 8;
} __spi_spi_cntrl2_bits;

/* SPI Control Register */
typedef struct {
  __REG32 RX_CLR          : 1;
  __REG32 TX_CLR          : 1;
  __REG32                 : 30;
} __spi_spi_fifo_ctl_bits;

/* Timer0 Control and Status Register */
typedef struct {
  __REG32 PRESCALE        : 8;
  __REG32                 : 8;
  __REG32 TDR_EN          : 1;
  __REG32                 : 6;
  __REG32 WAKE_EN         : 1;
  __REG32 CTB             : 1;
  __REG32 CACT            : 1;
  __REG32 CRST            : 1;
  __REG32 MODE            : 2;
  __REG32 IE              : 1;
  __REG32 CEN             : 1;
  __REG32 DBGACK_TMR      : 1;
} __tmr01_tcsr0_bits;

/* Timer0 Compare Register */
typedef struct {
  __REG32 TCMP            : 24;
  __REG32                 : 8;
} __tmr01_tcmpr0_bits;

/* Timer0 Interrupt Status Register */
typedef struct {
  __REG32 TIF             : 1;
  __REG32 TWF             : 1;
  __REG32                 : 30;
} __tmr01_tisr0_bits;

/* Timer0 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr01_tdr0_bits;

/* Timer0 Capture Data Register */
typedef struct {
  __REG32 TCAP            : 24;
  __REG32                 : 8;
} __tmr01_tcap0_bits;

/* Timer0 External Control Register */
typedef struct {
  __REG32 TX_PHASE        : 1;
  __REG32 TEX_EDGE        : 2;
  __REG32 TEXEN           : 1;
  __REG32 RSTCAPn         : 1;
  __REG32 TEXIEN          : 1;
  __REG32 TEXDB           : 1;
  __REG32 TCDB            : 1;
  __REG32 CAP_MODE        : 1;
  __REG32                 : 23;
} __tmr01_texcon0_bits;

/* Timer0 External Interrupt Status Register */
typedef struct {
  __REG32 TEXIF           : 1;
  __REG32                 : 31;
} __tmr01_texisr0_bits;

/* Timer1 Control and Status Register */
typedef struct {
  __REG32 PRESCALE        : 8;
  __REG32                 : 8;
  __REG32 TDR_EN          : 1;
  __REG32                 : 6;
  __REG32 WAKE_EN         : 1;
  __REG32 CTB             : 1;
  __REG32 CACT            : 1;
  __REG32 CRST            : 1;
  __REG32 MODE            : 2;
  __REG32 IE              : 1;
  __REG32 CEN             : 1;
  __REG32 DBGACK_TMR      : 1;
} __tmr01_tcsr1_bits;

/* Timer1 Compare Register */
typedef struct {
  __REG32 TCMP            : 24;
  __REG32                 : 8;
} __tmr01_tcmpr1_bits;

/* Timer1 Interrupt Status Register */
typedef struct {
  __REG32 TIF             : 1;
  __REG32 TWF             : 1;
  __REG32                 : 30;
} __tmr01_tisr1_bits;

/* Timer1 Data Register */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr01_tdr1_bits;

/* Timer1 Capture Data Register */
typedef struct {
  __REG32 TCAP            : 24;
  __REG32                 : 8;
} __tmr01_tcap1_bits;

/* Timer1 External Control Register */
typedef struct {
  __REG32 TX_PHASE        : 1;
  __REG32 TEX_EDGE        : 2;
  __REG32 TEXEN           : 1;
  __REG32 RSTCAPn         : 1;
  __REG32 TEXIEN          : 1;
  __REG32 TEXDB           : 1;
  __REG32 TCDB            : 1;
  __REG32 CAP_MODE        : 1;
  __REG32                 : 23;
} __tmr01_texcon1_bits;

/* Timer1 External Interrupt Status Register */
typedef struct {
  __REG32 TEXIF           : 1;
  __REG32                 : 31;
} __tmr01_texisr1_bits;

/* Watchdog Timer Control Register */
typedef struct {
  __REG32 WTR             : 1;
  __REG32 WTRE            : 1;
  __REG32 WTRF            : 1;
  __REG32 WTIF            : 1;
  __REG32 WTWKE           : 1;
  __REG32 WTWKF           : 1;
  __REG32 WTIE            : 1;
  __REG32 WTE             : 1;
  __REG32 WTIS            : 3;
  __REG32                 : 20;
  __REG32 DBGACK_WDT      : 1;
} __wdt_wtcr_bits;

/* Receive Buffer Register (UART_UA_RBR) */
/* Transmit Holding Register (UART_UA_THR) */
typedef union {
  /* UART_UA_RBR */
  struct {
    __REG32 RBR               : 8;
    __REG32                   :24;
  };
  /* UART_UA_THR */
  struct {
    __REG32 THR               : 8;
    __REG32                   :24;  
  };
  
  struct {
    __REG32 DATA              : 8;
    __REG32                   :24;  
  };
} __uart_ua_rbr_thr_bits; 

/* UART Interrupt Enable Register */
typedef struct {
  __REG32 RDA_IEN         : 1;
  __REG32 THRE_IEN        : 1;
  __REG32 RLS_IEN         : 1;
  __REG32 MODEM_IEN       : 1;
  __REG32 RTO_IEN         : 1;
  __REG32 BUF_ERR_IEN     : 1;
  __REG32 WAKE_EN         : 1;
  __REG32                 : 4;
  __REG32 TIME_OUT_EN     : 1;
  __REG32 AUTO_RTS_EN     : 1;
  __REG32 AUTO_CTS_EN     : 1;
  __REG32                 : 18;
} __uart_ua_ier_bits;

/* UART FIFO Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 RFR             : 1;
  __REG32 TFR             : 1;
  __REG32                 : 1;
  __REG32 RFITL           : 4;
  __REG32 RX_DIS          : 1;
  __REG32                 : 7;
  __REG32 RTS_TRI_LEV     : 4;
  __REG32                 : 12;
} __uart_ua_fcr_bits;

/* UART Line Control Register */
typedef struct {
  __REG32 WLS             : 2;
  __REG32 NSB             : 1;
  __REG32 PBE             : 1;
  __REG32 EPE             : 1;
  __REG32 SPE             : 1;
  __REG32 BCB             : 1;
  __REG32                 : 25;
} __uart_ua_lcr_bits;

/* UART Modem Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 RTSn            : 1;
  __REG32                 : 7;
  __REG32 LEV_RTS         : 1;
  __REG32                 : 3;
  __REG32 RTS_ST          : 1;
  __REG32                 : 18;
} __uart_ua_mcr_bits;

/* UART Modem Status Register */
typedef struct {
  __REG32 DCTSF           : 1;
  __REG32                 : 3;
  __REG32 CTS_ST          : 1;
  __REG32                 : 3;
  __REG32 LEV_CTS         : 1;
  __REG32                 : 23;
} __uart_ua_msr_bits;

/* UART FIFO Status Register */
typedef struct {
  __REG32 RX_OVER_IF      : 1;
  __REG32                 : 2;
  __REG32 RS_485_ADD_DETF : 1;
  __REG32 PEF             : 1;
  __REG32 FEF             : 1;
  __REG32 BIF             : 1;
  __REG32                 : 1;
  __REG32 RX_POINTER      : 6;
  __REG32 RX_EMPTY        : 1;
  __REG32 RX_FULL         : 1;
  __REG32 TX_POINTER      : 6;
  __REG32 TX_EMPTY        : 1;
  __REG32 TX_FULL         : 1;
  __REG32 TX_OVER_IF      : 1;
  __REG32                 : 3;
  __REG32 TE_FLAG         : 1;
  __REG32                 : 3;
} __uart_ua_fsr_bits;

/* UART Interrupt Status Register */
typedef struct {
  __REG32 RDA_IF          : 1;
  __REG32 THRE_IF         : 1;
  __REG32 RLS_IF          : 1;
  __REG32 MODEM_IF        : 1;
  __REG32 TOUT_IF         : 1;
  __REG32 BUF_ERR_IF      : 1;
  __REG32                 : 2;
  __REG32 RDA_INT         : 1;
  __REG32 THRE_INT        : 1;
  __REG32 RLS_INT         : 1;
  __REG32 MODEM_INT       : 1;
  __REG32 TOUT_INT        : 1;
  __REG32 BUF_ERR_INT     : 1;
  __REG32                 : 18;
} __uart_ua_isr_bits;

/* UART Time Out Register */
typedef struct {
  __REG32 TOIC            : 7;
  __REG32                 : 1;
  __REG32 DLY             : 8;
  __REG32                 : 16;
} __uart_ua_tor_bits;

/* UART Baud Rate Divisor Register */
typedef struct {
  __REG32 BRD             : 16;
  __REG32                 : 8;
  __REG32 DIVIDER_X       : 4;
  __REG32 DIV_X_ONE       : 1;
  __REG32 DIV_X_EN        : 1;
  __REG32                 : 2;
} __uart_ua_baud_bits;

/* UART IrDA Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32 TX_SELECT       : 1;
  __REG32                 : 3;
  __REG32 INV_TX          : 1;
  __REG32 INV_RX          : 1;
  __REG32                 : 25;
} __uart_ua_ircr_bits;

/* UART Alternate Control/Status Register */
typedef struct {
  __REG32                 : 8;
  __REG32 RS_485_NMM      : 1;
  __REG32 RS_485_AAD      : 1;
  __REG32 RS_485_AUD      : 1;
  __REG32                 : 4;
  __REG32 RS_485_ADD_EN   : 1;
  __REG32                 : 8;
  __REG32 ADDR_MATCH      : 8;
} __uart_ua_alt_csr_bits;

/* UART Function Select Register */
typedef struct {
  __REG32 FUN_SEL         : 2;
  __REG32                 : 30;
} __uart_ua_fun_sel_bits;

/* A/D Data Register */
typedef struct {
  __REG32 RSLT            : 10;
  __REG32                 : 6;
  __REG32 OVERRUN         : 1;
  __REG32 VALID           : 1;
  __REG32                 : 14;
} __adc_addr_bits;

/* A/D Control Register */
typedef struct {
  __REG32 ADEN            : 1;
  __REG32 ADIE            : 1;
  __REG32                 : 4;
  __REG32 TRGCOND         : 1;
  __REG32                 : 1;
  __REG32 TRGEN           : 1;
  __REG32                 : 2;
  __REG32 ADST            : 1;
  __REG32                 : 20;
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
  __REG32 PRESEL          : 1;
  __REG32                 : 23;
} __adc_adcher_bits;

/* A/D Compare Register 0 */
typedef struct {
  __REG32 CMPEN           : 1;
  __REG32 CMPIE           : 1;
  __REG32 CMPCOND         : 1;
  __REG32 CMPCH           : 3;
  __REG32                 : 2;
  __REG32 CMPMATCNT       : 4;
  __REG32                 : 4;
  __REG32 CMPD            : 10;
  __REG32                 : 6;
} __adc_adcmpr0_bits;

/* A/D Compare Register 1 */
typedef struct {
  __REG32 CMPEN           : 1;
  __REG32 CMPIE           : 1;
  __REG32 CMPCOND         : 1;
  __REG32 CMPCH           : 3;
  __REG32                 : 2;
  __REG32 CMPMATCNT       : 4;
  __REG32                 : 4;
  __REG32 CMPD            : 10;
  __REG32                 : 6;
} __adc_adcmpr1_bits;

/* A/D Status Register */
typedef struct {
  __REG32 ADF             : 1;
  __REG32 CMPF0           : 1;
  __REG32 CMPF1           : 1;
  __REG32 BUSY            : 1;
  __REG32 CHANNEL         : 3;
  __REG32                 : 1;
  __REG32 VALID           : 1;
  __REG32                 : 7;
  __REG32 OVERRUN         : 1;
  __REG32                 : 15;
} __adc_adsr_bits;

/* ISP Control Register */
typedef struct {
  __REG32 ISPEN           : 1;
  __REG32 BS              : 1;
  __REG32                 : 2;
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

/* Comparator0 Control Register */
typedef struct {
  __REG32 CMP0EN          : 1;
  __REG32 CMP0IE          : 1;
  __REG32 CMP0_HYSEN      : 1;
  __REG32                 : 1;
  __REG32 CN0             : 1;
  __REG32                 : 27;
} __cmp_cmp0cr_bits;

/* Comparator1 Control Register */
typedef struct {
  __REG32 CMP1EN          : 1;
  __REG32 CMP1IE          : 1;
  __REG32 CMP1_HYSEN      : 1;
  __REG32                 : 1;
  __REG32 CN1             : 1;
  __REG32                 : 27;
} __cmp_cmp1cr_bits;

/* Comparator Status Register */
typedef struct {
  __REG32 CMPF0           : 1;
  __REG32 CMPF1           : 1;
  __REG32 CO0             : 1;
  __REG32 CO1             : 1;
  __REG32                 : 28;
} __cmp_cmpsr_bits;

/* Comparator reference Voltage Control Register */
typedef struct {
  __REG32 CRVS            : 4;
  __REG32                 : 3;
  __REG32 OUT_SEL         : 1;
  __REG32                 : 24;
} __cmp_cmprvcr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */
/***************************************************************************/
/* Common declarations  ****************************************************/
/***************************************************************************
 **
 ** GCR
 **
 ***************************************************************************/
__IO_REG32(    GCR_PDID,              0x50000000,__READ       );
__IO_REG32_BIT(GCR_RSTSRC,            0x50000004,__READ_WRITE ,__gcr_rstsrc_bits);
__IO_REG32_BIT(GCR_IPRSTC1,           0x50000008,__READ_WRITE ,__gcr_iprstc1_bits);
__IO_REG32_BIT(GCR_IPRSTC2,           0x5000000C,__READ_WRITE ,__gcr_iprstc2_bits);
__IO_REG32_BIT(GCR_ITESTCR,           0x50000014,__READ_WRITE ,__gcr_itestcr_bits);
__IO_REG32_BIT(GCR_BODCR,             0x50000018,__READ_WRITE ,__gcr_bodcr_bits);
__IO_REG32_BIT(GCR_LDOCR,             0x5000001C,__READ_WRITE ,__gcr_ldocr_bits);
__IO_REG32_BIT(GCR_PORCR,             0x50000024,__READ_WRITE ,__gcr_porcr_bits);
__IO_REG32_BIT(GCR_P0_MFP,            0x50000030,__READ_WRITE ,__gcr_p0_mfp_bits);
__IO_REG32_BIT(GCR_P1_MFP,            0x50000034,__READ_WRITE ,__gcr_p1_mfp_bits);
__IO_REG32_BIT(GCR_P2_MFP,            0x50000038,__READ_WRITE ,__gcr_p2_mfp_bits);
__IO_REG32_BIT(GCR_P3_MFP,            0x5000003C,__READ_WRITE ,__gcr_p3_mfp_bits);
__IO_REG32_BIT(GCR_P4_MFP,            0x50000040,__READ_WRITE ,__gcr_p4_mfp_bits);
__IO_REG32_BIT(GCR_P5_MFP,            0x50000044,__READ_WRITE ,__gcr_p5_mfp_bits);
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
__IO_REG32_BIT(SCS_SYST_CSR,          0xE000E010,__READ_WRITE ,__scs_syst_csr_bits);
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
__IO_REG32_BIT(SCS_AIRCR,             0xE000ED0C,__READ_WRITE ,__scs_aircr_bits);
__IO_REG32_BIT(SCS_SCR,               0xE000ED10,__READ_WRITE ,__scs_scr_bits);
__IO_REG32_BIT(SCS_SHPR2,             0xE000ED1C,__READ_WRITE ,__scs_shpr2_bits);
__IO_REG32_BIT(SCS_SHPR3,             0xE000ED20,__READ_WRITE ,__scs_shpr3_bits);

/***************************************************************************
 **
 ** INT
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_IRQ0_SRC,          0x50000300,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ1_SRC,          0x50000304,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ2_SRC,          0x50000308,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ3_SRC,          0x5000030C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ4_SRC,          0x50000310,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ5_SRC,          0x50000314,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ6_SRC,          0x50000318,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ7_SRC,          0x5000031C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ8_SRC,          0x50000320,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ9_SRC,          0x50000324,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ12_SRC,         0x50000330,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ14_SRC,         0x50000338,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ16_SRC,         0x50000340,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ17_SRC,         0x50000344,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ18_SRC,         0x50000348,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ25_SRC,         0x50000364,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ28_SRC,         0x50000370,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ29_SRC,         0x50000374,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_NMI_SEL,           0x50000380,__READ_WRITE ,__int_nmi_sel_bits);
__IO_REG32(    INT_MCU_IRQ,           0x50000384,__READ_WRITE );

/***************************************************************************
 **
 ** CLK
 **
 ***************************************************************************/
__IO_REG32_BIT(CLK_PWRCON,            0x50000200,__READ_WRITE ,__clk_pwrcon_bits);
__IO_REG32_BIT(CLK_AHBCLK,            0x50000204,__READ_WRITE ,__clk_ahbclk_bits);
__IO_REG32_BIT(CLK_APBCLK,            0x50000208,__READ_WRITE ,__clk_apbclk_bits);
__IO_REG32_BIT(CLK_CLKSTATUS,         0x5000020C,__READ_WRITE ,__clk_clkstatus_bits);
__IO_REG32_BIT(CLK_CLKSEL0,           0x50000210,__READ_WRITE ,__clk_clksel0_bits);
__IO_REG32_BIT(CLK_CLKSEL1,           0x50000214,__READ_WRITE ,__clk_clksel1_bits);
__IO_REG32_BIT(CLK_CLKSEL2,           0x5000021C,__READ_WRITE ,__clk_clksel2_bits);
__IO_REG32_BIT(CLK_CLKDIV,            0x50000218,__READ_WRITE ,__clk_clkdiv_bits);
__IO_REG32_BIT(CLK_FRQDIV,            0x50000224,__READ_WRITE ,__clk_frqdiv_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_P0_PMD,           0x50004000,__READ_WRITE ,__gpio_p0_pmd_bits);
__IO_REG32_BIT(GPIO_P0_OFFD,          0x50004004,__READ_WRITE ,__gpio_p0_offd_bits);
__IO_REG32_BIT(GPIO_P0_DOUT,          0x50004008,__READ_WRITE ,__gpio_p0_dout_bits);
__IO_REG32_BIT(GPIO_P0_DMASK,         0x5000400C,__READ_WRITE ,__gpio_p0_dmask_bits);
__IO_REG32_BIT(GPIO_P0_PIN,           0x50004010,__READ       ,__gpio_p0_pin_bits);
__IO_REG32_BIT(GPIO_P0_DBEN,          0x50004014,__READ_WRITE ,__gpio_p0_dben_bits);
__IO_REG32_BIT(GPIO_P0_IMD,           0x50004018,__READ_WRITE ,__gpio_p0_imd_bits);
__IO_REG32_BIT(GPIO_P0_IEN,           0x5000401C,__READ_WRITE ,__gpio_p0_ien_bits);
__IO_REG32_BIT(GPIO_P0_ISRC,          0x50004020,__READ_WRITE ,__gpio_p0_isrc_bits);
__IO_REG32_BIT(GPIO_P1_PMD,           0x50004040,__READ_WRITE ,__gpio_p1_pmd_bits);
__IO_REG32_BIT(GPIO_P1_OFFD,          0x50004044,__READ_WRITE ,__gpio_p1_offd_bits);
__IO_REG32_BIT(GPIO_P1_DOUT,          0x50004048,__READ_WRITE ,__gpio_p1_dout_bits);
__IO_REG32_BIT(GPIO_P1_DMASK,         0x5000404C,__READ_WRITE ,__gpio_p1_dmask_bits);
__IO_REG32_BIT(GPIO_P1_PIN,           0x50004050,__READ       ,__gpio_p1_pin_bits);
__IO_REG32_BIT(GPIO_P1_DBEN,          0x50004054,__READ_WRITE ,__gpio_p1_dben_bits);
__IO_REG32_BIT(GPIO_P1_IMD,           0x50004058,__READ_WRITE ,__gpio_p1_imd_bits);
__IO_REG32_BIT(GPIO_P1_IEN,           0x5000405C,__READ_WRITE ,__gpio_p1_ien_bits);
__IO_REG32_BIT(GPIO_P1_ISRC,          0x50004060,__READ_WRITE ,__gpio_p1_isrc_bits);
__IO_REG32_BIT(GPIO_P2_PMD,           0x50004080,__READ_WRITE ,__gpio_p2_pmd_bits);
__IO_REG32_BIT(GPIO_P2_OFFD,          0x50004084,__READ_WRITE ,__gpio_p2_offd_bits);
__IO_REG32_BIT(GPIO_P2_DOUT,          0x50004088,__READ_WRITE ,__gpio_p2_dout_bits);
__IO_REG32_BIT(GPIO_P2_DMASK,         0x5000408C,__READ_WRITE ,__gpio_p2_dmask_bits);
__IO_REG32_BIT(GPIO_P2_PIN,           0x50004090,__READ       ,__gpio_p2_pin_bits);
__IO_REG32_BIT(GPIO_P2_DBEN,          0x50004094,__READ_WRITE ,__gpio_p2_dben_bits);
__IO_REG32_BIT(GPIO_P2_IMD,           0x50004098,__READ_WRITE ,__gpio_p2_imd_bits);
__IO_REG32_BIT(GPIO_P2_IEN,           0x5000409C,__READ_WRITE ,__gpio_p2_ien_bits);
__IO_REG32_BIT(GPIO_P2_ISRC,          0x500040A0,__READ_WRITE ,__gpio_p2_isrc_bits);
__IO_REG32_BIT(GPIO_P3_PMD,           0x500040C0,__READ_WRITE ,__gpio_p3_pmd_bits);
__IO_REG32_BIT(GPIO_P3_OFFD,          0x500040C4,__READ_WRITE ,__gpio_p3_offd_bits);
__IO_REG32_BIT(GPIO_P3_DOUT,          0x500040C8,__READ_WRITE ,__gpio_p3_dout_bits);
__IO_REG32_BIT(GPIO_P3_DMASK,         0x500040CC,__READ_WRITE ,__gpio_p3_dmask_bits);
__IO_REG32_BIT(GPIO_P3_PIN,           0x500040D0,__READ       ,__gpio_p3_pin_bits);
__IO_REG32_BIT(GPIO_P3_DBEN,          0x500040D4,__READ_WRITE ,__gpio_p3_dben_bits);
__IO_REG32_BIT(GPIO_P3_IMD,           0x500040D8,__READ_WRITE ,__gpio_p3_imd_bits);
__IO_REG32_BIT(GPIO_P3_IEN,           0x500040DC,__READ_WRITE ,__gpio_p3_ien_bits);
__IO_REG32_BIT(GPIO_P3_ISRC,          0x500040E0,__READ_WRITE ,__gpio_p3_isrc_bits);
__IO_REG32_BIT(GPIO_P4_PMD,           0x50004100,__READ_WRITE ,__gpio_p4_pmd_bits);
__IO_REG32_BIT(GPIO_P4_OFFD,          0x50004104,__READ_WRITE ,__gpio_p4_offd_bits);
__IO_REG32_BIT(GPIO_P4_DOUT,          0x50004108,__READ_WRITE ,__gpio_p4_dout_bits);
__IO_REG32_BIT(GPIO_P4_DMASK,         0x5000410C,__READ_WRITE ,__gpio_p4_dmask_bits);
__IO_REG32_BIT(GPIO_P4_PIN,           0x50004110,__READ       ,__gpio_p4_pin_bits);
__IO_REG32_BIT(GPIO_P4_DBEN,          0x50004114,__READ_WRITE ,__gpio_p4_dben_bits);
__IO_REG32_BIT(GPIO_P4_IMD,           0x50004118,__READ_WRITE ,__gpio_p4_imd_bits);
__IO_REG32_BIT(GPIO_P4_IEN,           0x5000411C,__READ_WRITE ,__gpio_p4_ien_bits);
__IO_REG32_BIT(GPIO_P4_ISRC,          0x50004120,__READ_WRITE ,__gpio_p4_isrc_bits);
__IO_REG32_BIT(GPIO_P5_PMD,           0x50004140,__READ_WRITE ,__gpio_p5_pmd_bits);
__IO_REG32_BIT(GPIO_P5_OFFD,          0x50004144,__READ_WRITE ,__gpio_p5_offd_bits);
__IO_REG32_BIT(GPIO_P5_DOUT,          0x50004148,__READ_WRITE ,__gpio_p5_dout_bits);
__IO_REG32_BIT(GPIO_P5_DMASK,         0x5000414C,__READ_WRITE ,__gpio_p5_dmask_bits);
__IO_REG32_BIT(GPIO_P5_PIN,           0x50004150,__READ       ,__gpio_p5_pin_bits);
__IO_REG32_BIT(GPIO_P5_DBEN,          0x50004154,__READ_WRITE ,__gpio_p5_dben_bits);
__IO_REG32_BIT(GPIO_P5_IMD,           0x50004158,__READ_WRITE ,__gpio_p5_imd_bits);
__IO_REG32_BIT(GPIO_P5_IEN,           0x5000415C,__READ_WRITE ,__gpio_p5_ien_bits);
__IO_REG32_BIT(GPIO_P5_ISRC,          0x50004160,__READ_WRITE ,__gpio_p5_isrc_bits);
__IO_REG32_BIT(GPIO_DBNCECON,         0x50004180,__READ_WRITE ,__gpio_dbncecon_bits);
__IO_REG32_BIT(GPIO_P00_DOUT,         0x50004200,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P01_DOUT,         0x50004204,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P04_DOUT,         0x50004210,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P05_DOUT,         0x50004214,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P06_DOUT,         0x50004218,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P07_DOUT,         0x5000421C,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P10_DOUT,         0x50004220,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P12_DOUT,         0x50004228,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P13_DOUT,         0x5000422C,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P14_DOUT,         0x50004230,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P15_DOUT,         0x50004234,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P22_DOUT,         0x50004248,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P23_DOUT,         0x5000424C,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P24_DOUT,         0x50004250,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P25_DOUT,         0x50004254,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P26_DOUT,         0x50004258,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P30_DOUT,         0x50004260,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P31_DOUT,         0x50004264,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P32_DOUT,         0x50004268,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P34_DOUT,         0x50004270,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P35_DOUT,         0x50004274,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P36_DOUT,         0x50004278,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P46_DOUT,         0x50004298,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P47_DOUT,         0x5000429C,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P50_DOUT,         0x500042A0,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P51_DOUT,         0x500042A4,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P52_DOUT,         0x500042A8,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P53_DOUT,         0x500042AC,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P54_DOUT,         0x500042B0,__READ_WRITE ,__gpio_pxy_dout_bits);
__IO_REG32_BIT(GPIO_P55_DOUT,         0x500042B4,__READ_WRITE ,__gpio_pxy_dout_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C_I2CON,             0x40020000,__READ_WRITE ,__i2c_i2con_bits);
__IO_REG32_BIT(I2C_I2CDAT,            0x40020008,__READ_WRITE ,__i2c_i2cdat_bits);
__IO_REG32_BIT(I2C_I2CSTATUS,         0x4002000C,__READ       ,__i2c_i2cstatus_bits);
__IO_REG32_BIT(I2C_I2CLK,             0x40020010,__READ_WRITE ,__i2c_i2clk_bits);
__IO_REG32_BIT(I2C_I2CTOC,            0x40020014,__READ_WRITE ,__i2c_i2ctoc_bits);
__IO_REG32_BIT(I2C_I2CADDR0,          0x40020004,__READ_WRITE ,__i2c_i2caddrx_bits);
__IO_REG32_BIT(I2C_I2CADDR1,          0x40020018,__READ_WRITE ,__i2c_i2caddrx_bits);
__IO_REG32_BIT(I2C_I2CADDR2,          0x4002001C,__READ_WRITE ,__i2c_i2caddrx_bits);
__IO_REG32_BIT(I2C_I2CADDR3,          0x40020020,__READ_WRITE ,__i2c_i2caddrx_bits);
__IO_REG32_BIT(I2C_I2CADM0,           0x40020024,__READ_WRITE ,__i2c_i2cadmx_bits);
__IO_REG32_BIT(I2C_I2CADM1,           0x40020028,__READ_WRITE ,__i2c_i2cadmx_bits);
__IO_REG32_BIT(I2C_I2CADM2,           0x4002002C,__READ_WRITE ,__i2c_i2cadmx_bits);
__IO_REG32_BIT(I2C_I2CADM3,           0x40020030,__READ_WRITE ,__i2c_i2cadmx_bits);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMA_PPR,               0x40040000,__READ_WRITE ,__pwm_ppr_bits);
__IO_REG32_BIT(PWMA_CSR,               0x40040004,__READ_WRITE ,__pwm_csr_bits);
__IO_REG32_BIT(PWMA_PCR,               0x40040008,__READ_WRITE ,__pwm_pcr_bits);
__IO_REG32_BIT(PWMA_CNR0,              0x4004000C,__READ_WRITE ,__pwm_cnr0_bits);
__IO_REG32_BIT(PWMA_CNR1,              0x40040010,__READ_WRITE ,__pwm_cnr1_bits);
__IO_REG32_BIT(PWMA_CNR2,              0x40040014,__READ_WRITE ,__pwm_cnr2_bits);
__IO_REG32_BIT(PWMA_CNR3,              0x40040018,__READ_WRITE ,__pwm_cnr3_bits);
__IO_REG32_BIT(PWMA_CNR4,              0x4004001C,__READ_WRITE ,__pwm_cnr4_bits);
__IO_REG32_BIT(PWMA_CNR5,              0x40040020,__READ_WRITE ,__pwm_cnr5_bits);
__IO_REG32_BIT(PWMA_CMR0,              0x40040024,__READ_WRITE ,__pwm_cmr0_bits);
__IO_REG32_BIT(PWMA_CMR1,              0x40040028,__READ_WRITE ,__pwm_cmr1_bits);
__IO_REG32_BIT(PWMA_CMR2,              0x4004002C,__READ_WRITE ,__pwm_cmr2_bits);
__IO_REG32_BIT(PWMA_CMR3,              0x40040030,__READ_WRITE ,__pwm_cmr3_bits);
__IO_REG32_BIT(PWMA_CMR4,              0x40040034,__READ_WRITE ,__pwm_cmr4_bits);
__IO_REG32_BIT(PWMA_CMR5,              0x40040038,__READ_WRITE ,__pwm_cmr5_bits);
__IO_REG32_BIT(PWMA_PDR0,              0x4004003C,__READ_WRITE ,__pwm_pdr0_bits);
__IO_REG32_BIT(PWMA_PDR1,              0x40040040,__READ_WRITE ,__pwm_pdr1_bits);
__IO_REG32_BIT(PWMA_PDR2,              0x40040044,__READ_WRITE ,__pwm_pdr2_bits);
__IO_REG32_BIT(PWMA_PDR3,              0x40040048,__READ_WRITE ,__pwm_pdr3_bits);
__IO_REG32_BIT(PWMA_PDR4,              0x4004004C,__READ_WRITE ,__pwm_pdr4_bits);
__IO_REG32_BIT(PWMA_PDR5,              0x40040050,__READ_WRITE ,__pwm_pdr5_bits);
__IO_REG32_BIT(PWMA_PIER,              0x40040054,__READ_WRITE ,__pwm_pier_bits);
__IO_REG32_BIT(PWMA_PIIR,              0x40040058,__READ_WRITE ,__pwm_piir_bits);
__IO_REG32_BIT(PWMA_POE,               0x4004005C,__READ_WRITE ,__pwm_poe_bits);
__IO_REG32_BIT(PWMA_PFBCON,            0x40040060,__READ_WRITE ,__pwm_pfbcon_bits);
__IO_REG32_BIT(PWMA_PDZIR,             0x40040064,__READ_WRITE ,__pwm_pdzir_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_SPI_CNTRL,         0x40030000,__READ_WRITE ,__spi_spi_cntrl_bits);
__IO_REG32_BIT(SPI0_SPI_DIVIDER,       0x40030004,__READ_WRITE ,__spi_spi_divider_bits);
__IO_REG32_BIT(SPI0_SPI_SSR,           0x40030008,__READ_WRITE ,__spi_spi_ssr_bits);
__IO_REG32(    SPI0_SPI_Rx0,           0x40030010,__READ       );
__IO_REG32(    SPI0_SPI_Rx1,           0x40030014,__READ       );
__IO_REG32(    SPI0_SPI_Tx0,           0x40030020,__WRITE      );
__IO_REG32(    SPI0_SPI_Tx1,           0x40030024,__WRITE      );
__IO_REG32(    SPI0_SPI_VARCLK,        0x40030034,__READ_WRITE );
__IO_REG32_BIT(SPI0_SPI_CNTRL2,        0x4003003C,__READ_WRITE ,__spi_spi_cntrl2_bits);
__IO_REG32_BIT(SPI0_SPI_FIFO_CTL,      0x40030040,__READ_WRITE ,__spi_spi_fifo_ctl_bits);

/***************************************************************************
 **
 ** TMR01
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR01_TCSR0,           0x40010000,__READ_WRITE ,__tmr01_tcsr0_bits);
__IO_REG32_BIT(TMR01_TCMPR0,          0x40010004,__READ_WRITE ,__tmr01_tcmpr0_bits);
__IO_REG32_BIT(TMR01_TISR0,           0x40010008,__READ_WRITE ,__tmr01_tisr0_bits);
__IO_REG32_BIT(TMR01_TDR0,            0x4001000C,__READ       ,__tmr01_tdr0_bits);
__IO_REG32_BIT(TMR01_TCAP0,           0x40010010,__READ       ,__tmr01_tcap0_bits);
__IO_REG32_BIT(TMR01_TEXCON0,         0x40010014,__READ_WRITE ,__tmr01_texcon0_bits);
__IO_REG32_BIT(TMR01_TEXISR0,         0x40010018,__READ_WRITE ,__tmr01_texisr0_bits);
__IO_REG32_BIT(TMR01_TCSR1,           0x40010020,__READ_WRITE ,__tmr01_tcsr1_bits);
__IO_REG32_BIT(TMR01_TCMPR1,          0x40010024,__READ_WRITE ,__tmr01_tcmpr1_bits);
__IO_REG32_BIT(TMR01_TISR1,           0x40010028,__READ_WRITE ,__tmr01_tisr1_bits);
__IO_REG32_BIT(TMR01_TDR1,            0x4001002C,__READ       ,__tmr01_tdr1_bits);
__IO_REG32_BIT(TMR01_TCAP1,           0x40010030,__READ       ,__tmr01_tcap1_bits);
__IO_REG32_BIT(TMR01_TEXCON1,         0x40010034,__READ_WRITE ,__tmr01_texcon1_bits);
__IO_REG32_BIT(TMR01_TEXISR1,         0x40010038,__READ_WRITE ,__tmr01_texisr1_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_WTCR,              0x40004000,__READ_WRITE ,__wdt_wtcr_bits);

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG32_BIT(UART_UA_RBR,          0x40050000,__READ_WRITE ,__uart_ua_rbr_thr_bits);
#define UART_UA_THR     UART_UA_RBR
#define UART_UA_THR_bit UART_UA_RBR_bit
__IO_REG32_BIT(UART_UA_IER,           0x40050004,__READ_WRITE ,__uart_ua_ier_bits);
__IO_REG32_BIT(UART_UA_FCR,           0x40050008,__READ_WRITE ,__uart_ua_fcr_bits);
__IO_REG32_BIT(UART_UA_LCR,           0x4005000C,__READ_WRITE ,__uart_ua_lcr_bits);
__IO_REG32_BIT(UART_UA_MCR,           0x40050010,__READ_WRITE ,__uart_ua_mcr_bits);
__IO_REG32_BIT(UART_UA_MSR,           0x40050014,__READ_WRITE ,__uart_ua_msr_bits);
__IO_REG32_BIT(UART_UA_FSR,           0x40050018,__READ_WRITE ,__uart_ua_fsr_bits);
__IO_REG32_BIT(UART_UA_ISR,           0x4005001C,__READ_WRITE ,__uart_ua_isr_bits);
__IO_REG32_BIT(UART_UA_TOR,           0x40050020,__READ_WRITE ,__uart_ua_tor_bits);
__IO_REG32_BIT(UART_UA_BAUD,          0x40050024,__READ_WRITE ,__uart_ua_baud_bits);
__IO_REG32_BIT(UART_UA_IRCR,          0x40050028,__READ_WRITE ,__uart_ua_ircr_bits);
__IO_REG32_BIT(UART_UA_ALT_CSR,       0x4005002C,__READ_WRITE ,__uart_ua_alt_csr_bits);
__IO_REG32_BIT(UART_UA_FUN_SEL,       0x40050030,__READ_WRITE ,__uart_ua_fun_sel_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ADDR,              0x400E0000,__READ       ,__adc_addr_bits);
__IO_REG32_BIT(ADC_ADCR,              0x400E0020,__READ_WRITE ,__adc_adcr_bits);
__IO_REG32_BIT(ADC_ADCHER,            0x400E0024,__READ_WRITE ,__adc_adcher_bits);
__IO_REG32_BIT(ADC_ADCMPR0,           0x400E0028,__READ_WRITE ,__adc_adcmpr0_bits);
__IO_REG32_BIT(ADC_ADCMPR1,           0x400E002C,__READ_WRITE ,__adc_adcmpr1_bits);
__IO_REG32_BIT(ADC_ADSR,              0x400E0030,__READ_WRITE ,__adc_adsr_bits);

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
__IO_REG32(    FMC_DFBA,              0x5000C014,__READ       );
__IO_REG32_BIT(FMC_ICPCON,            0x5000C01C,__READ_WRITE ,__fmc_icpcon_bits);
__IO_REG32_BIT(FMC_RMPCON,            0x5000C020,__READ_WRITE ,__fmc_rmpcon_bits);
__IO_REG32_BIT(FMC_ICECON,            0x5000C024,__READ_WRITE ,__fmc_icecon_bits);

/***************************************************************************
 **
 ** CMP
 **
 ***************************************************************************/
__IO_REG32_BIT(CMP_CMP0CR,            0x400D0000,__READ_WRITE ,__cmp_cmp0cr_bits);
__IO_REG32_BIT(CMP_CMP1CR,            0x400D0004,__READ_WRITE ,__cmp_cmp1cr_bits);
__IO_REG32_BIT(CMP_CMPSR,             0x400D0008,__READ_WRITE ,__cmp_cmpsr_bits);
__IO_REG32_BIT(CMP_CMPRVCR,           0x400D000C,__READ_WRITE ,__cmp_cmprvcr_bits);



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
#define NVIC_GP01_INT         20  /* External signal interrupt from P0[7:0] / P1[7:0]       */
#define NVIC_GP234_INT        21  /* External interrupt from P2[7:0]/P3[7:0]/P4[7:0] except P32 & P33 */
#define NVIC_PWMA_INT         22  /* PWM0,PWM1,PWM2 and PWM3 interrupt                      */
#define NVIC_TMR0_INT         24  /* Timer 0 interrupt                                      */
#define NVIC_TMR1_INT         25  /* Timer 1 interrupt                                      */
#define NVIC_TMR2_INT         26  /* Timer 2 interrupt                                      */
#define NVIC_TMR3_INT         27  /* Timer 3 interrupt                                      */
#define NVIC_UART0_INT        28  /* UART0 interrupt                                        */
#define NVIC_SPI0_INT         30  /* SPI0 interrupt                                         */
#define NVIC_GP5_INT          32  /* GP5 interrupt                                          */
#define NVIC_HFIRC_TRIM_INT   33  /* HFIRC_TRIM interrupt                                   */
#define NVIC_I2C_INT          34  /* I2C interrupt                                          */
#define NVIC_ACMP_INT         41  /* ACMP interrupt                                         */
#define NVIC_PWRWU_INT        44  /* Clock controller interrupt for chip wake up from power-down state */
#define NVIC_ADC_INT          45  /* ADC interrupt                                          */

#endif    /* __IOMINI51_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = SVC            0x2C
Interrupt3   = PendSV         0x38
Interrupt4   = SysTick        0x3C
Interrupt5   = BOD_OUT        0x40
Interrupt6   = WDT_INT        0x44
Interrupt7   = INT0           0x48
Interrupt8   = INT1           0x4C
Interrupt9   = GP01_INT       0x50
Interrupt10  = GP234_INT      0x54
Interrupt11  = PWMA_INT       0x58
Interrupt12  = BRAKE_INT      0x5C
Interrupt13  = TMR0_INT       0x60
Interrupt14  = TMR1_INT       0x64
Interrupt15  = UART0_INT      0x70
Interrupt16  = SPI0_INT       0x78
Interrupt17  = GP5_INT        0x80
Interrupt18  = HFIRC_TRIM_INT 0x84
Interrupt19  = I2C_INT        0x88
Interrupt20  = ACMP_INT       0xA4
Interrupt21  = PWRWU_INT      0xB0
Interrupt22  = ADC_INT        0xB4
###DDF-INTERRUPT-END###*/

