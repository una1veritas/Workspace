/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Nuvoton M05x Devices
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 46179 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/
#ifndef __IOM05X_H
#define __IOM05X_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   M05X SPECIAL FUNCTION REGISTERS
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

/* A/D Data Register (ADC_ADDR0,ADC_ADDR1,ADC_ADDR2,ADC_ADDR3,ADC_ADDR4,ADC_ADDR5,ADC_ADDR6,ADC_ADDR7) */
typedef struct {
  __REG32 RSLT            : 12;
  __REG32                 : 4;
  __REG32 OVERRUN         : 1;
  __REG32 VALID           : 1;
  __REG32                 : 14;
} __adc_addrx_bits;

/* A/D Control Register (ADC_ADCR) */
typedef struct {
  __REG32 ADEN            : 1;
  __REG32 ADIE            : 1;
  __REG32 ADMD            : 2;
  __REG32 TRGS            : 2;
  __REG32 TRGCOND         : 2;
  __REG32 TRGEN           : 1;
  __REG32                 : 1;
  __REG32 DIFFEN          : 1;
  __REG32 ADST            : 1;
  __REG32                 : 20;
} __adc_adcr_bits;

/* A/D Channel Enable Register (ADC_ADCHER) */
typedef struct {
  __REG32 CHEN0           : 1;
  __REG32 CHEN1           : 1;
  __REG32 CHEN2           : 1;
  __REG32 CHEN3           : 1;
  __REG32 CHEN4           : 1;
  __REG32 CHEN5           : 1;
  __REG32 CHEN6           : 1;
  __REG32 CHEN7           : 1;
  __REG32 PRESEL          : 2;
  __REG32                 : 22;
} __adc_adcher_bits;

/* A/D Compare Register (ADC_ADCMPR0,ADC_ADCMPR1) */
typedef struct {
  __REG32 CMPEN           : 1;
  __REG32 CMPIE           : 1;
  __REG32 CMPCOND         : 1;
  __REG32 CMPCH           : 3;
  __REG32                 : 2;
  __REG32 CMPMATCNT       : 4;
  __REG32                 : 4;
  __REG32 CMPD            : 12;
  __REG32                 : 4;
} __adc_adcmprx_bits;

/* A/D Status Register (ADC_ADSR) */
typedef struct {
  __REG32 ADF             : 1;
  __REG32 CMPF0           : 1;
  __REG32 CMPF1           : 1;
  __REG32 BUSY            : 1;
  __REG32 CHANNEL         : 3;
  __REG32                 : 1;
  __REG32 VALID           : 8;
  __REG32 OVERRUN         : 8;
  __REG32                 : 8;
} __adc_adsr_bits;

/* A/D Calibration Register (ADC_ADCALR) */
typedef struct {
  __REG32 CALEN           : 1;
  __REG32 CALDONE         : 1;
  __REG32                 : 30;
} __adc_adcalr_bits;

/* System Power Down Control Register (CLK_PWRCON) */
typedef struct {
  __REG32 XTL12M_EN       : 1;
  __REG32                 : 1;
  __REG32 OSC22M_EN       : 1;
  __REG32 OSC10K_EN       : 1;
  __REG32 PD_WU_DLY       : 1;
  __REG32 PD_WU_INT_EN    : 1;
  __REG32 PD_WU_STS       : 1;
  __REG32 PWR_DOWN_EN     : 1;
  __REG32 PD_WAIT_CPU     : 1;
  __REG32                 : 23;
} __clk_pwrcon_bits;

/* AHB Devices Clock Enable Control Register (CLK_AHBCLK) */
typedef struct {
  __REG32                 : 2;
  __REG32 ISP_EN          : 1;
  __REG32 EBI_EN          : 1;
  __REG32                 : 28;
} __clk_ahbclk_bits;

/* APB Devices Clock Enable Control Register (CLK_APBCLK) */
typedef struct {
  __REG32 WDT_EN          : 1;
  __REG32                 : 1;
  __REG32 TMR0_EN         : 1;
  __REG32 TMR1_EN         : 1;
  __REG32 TMR2_EN         : 1;
  __REG32 TMR3_EN         : 1;
  __REG32 FDIV_EN         : 1;
  __REG32                 : 1;
  __REG32 I2C_EN          : 1;
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 SPI0_EN         : 1;
  __REG32 SPI1_EN         : 1;
  __REG32                 : 2;
  __REG32 UART0_EN        : 1;
  __REG32 UART1_EN        : 1;
  __REG32                 : 2;
  __REG32 PWM01_EN        : 1;
  __REG32 PWM23_EN        : 1;
  __REG32 PWM45_EN        : 1;
  __REG32 PWM67_EN        : 1;
  __REG32                 : 4;
  __REG32 ADC_EN          : 1;
  __REG32                 : 3;
} __clk_apbclk_bits;

/* Clock status monitor Register (CLK_CLKSTATUS) */
typedef struct {
  __REG32 XTL12M_STB      : 1;
  __REG32                 : 1;
  __REG32 PLL_STB         : 1;
  __REG32 OSC10K_STB      : 1;
  __REG32 OSC22M_STB      : 1;
  __REG32                 : 2;
  __REG32 CLK_SW_FAIL     : 1;
  __REG32                 : 24;
} __clk_clkstatus_bits;

/* Clock Source Select Control Register 0 (CLK_CLKSEL0) */
typedef struct {
  __REG32 HCLK_S          : 3;
  __REG32 STCLK_S         : 3;
  __REG32                 : 26;
} __clk_clksel0_bits;

/* Clock Source Select Control Register 1 (CLK_CLKSEL1) */
typedef struct {
  __REG32 WDT_S           : 2;
  __REG32 ADC_S           : 2;
  __REG32                 : 4;
  __REG32 TMR0_S          : 3;
  __REG32                 : 1;
  __REG32 TMR1_S          : 3;
  __REG32                 : 1;
  __REG32 TMR2_S          : 3;
  __REG32                 : 1;
  __REG32 TMR3_S          : 3;
  __REG32                 : 1;
  __REG32 UART_S          : 2;
  __REG32                 : 2;
  __REG32 PWM01_S         : 2;
  __REG32 PWM23_S         : 2;
} __clk_clksel1_bits;

/* Clock Source Select Control Register 2 (CLK_CLKSEL2) */
typedef struct {
  __REG32                 : 2;
  __REG32 FRQDIV_S        : 2;
  __REG32 PWM45_S         : 2;
  __REG32 PWM67_S         : 2;
  __REG32                 : 24;
} __clk_clksel2_bits;

/* Clock Divider Number Register (CLK_CLKDIV) */
typedef struct {
  __REG32 HCLK_N          : 4;
  __REG32                 : 4;
  __REG32 UART_N          : 4;
  __REG32                 : 4;
  __REG32 ADC_N           : 8;
  __REG32                 : 8;
} __clk_clkdiv_bits;

/* PLL Control Register (CLK_PLLCON) */
typedef struct {
  __REG32 FB_DV           : 9;
  __REG32 IN_DV           : 5;
  __REG32 OUT_DV          : 2;
  __REG32 PD              : 1;
  __REG32 BP              : 1;
  __REG32 OE              : 1;
  __REG32 PLL_SRC         : 1;
  __REG32                 : 12;
} __clk_pllcon_bits;

/* Frequency Divider Control Register (CLK_FRQDIV) */
typedef struct {
  __REG32 FSEL            : 4;
  __REG32 DIVIDER_EN      : 1;
  __REG32                 : 27;
} __clk_frqdiv_bits;

/* External Bus Interface General Control Register (EBI_EBICON) */
typedef struct {
  __REG32 ExtEN           : 1;
  __REG32 ExtBW16         : 1;
  __REG32                 : 6;
  __REG32 MCLKDIV         : 3;
  __REG32                 : 5;
  __REG32 ExttALE         : 3;
  __REG32                 : 13;
} __ebi_ctl_ebicon_bits;

/* External Bus Interface Timing Control Register (EBI_EXTIME) */
typedef struct {
  __REG32                 : 3;
  __REG32 ExttACC         : 5;
  __REG32 ExttAHD         : 3;
  __REG32                 : 1;
  __REG32 ExtIW2X         : 4;
  __REG32                 : 8;
  __REG32 ExtIR2R         : 4;
  __REG32                 : 4;
} __ebi_ctl_extime_bits;

/* ISP Control Register (FMC_ISPCON) */
typedef struct {
  __REG32 ISPEN           : 1;
  __REG32 BS              : 1;
  __REG32                 : 2;
  __REG32 CFGUEN          : 1;
  __REG32 LDUEN           : 1;
  __REG32 ISPFF           : 1;
  __REG32 SWRST           : 1;
  __REG32 PT              : 3;
  __REG32                 : 1;
  __REG32 ET              : 3;
  __REG32                 : 17;
} __fmc_ispcon_bits;

/* ISP Command Register (FMC_ISPCMD) */
typedef struct {
  __REG32 FCTRL           : 4;
  __REG32 FCEN            : 1;
  __REG32 FOEN            : 1;
  __REG32                 : 26;
} __fmc_ispcmd_bits;

/* ISP Trigger Control Register (FMC_ISPTRG) */
typedef struct {
  __REG32 ISPGO           : 1;
  __REG32                 : 31;
} __fmc_isptrg_bits;

/* Flash Access Time Control Register (FMC_FATCON) */
typedef struct {
  __REG32 FPSEN           : 1;
  __REG32 FATS            : 3;
  __REG32 LFOM            : 1;
  __REG32                 : 27;
} __fmc_fatcon_bits;

/* System Reset Source Register (GCR_RSTSRC) */
typedef struct {
  __REG32 RSTS_POR        : 1;
  __REG32 RSTS_RESET      : 1;
  __REG32 RSTS_WDT        : 1;
  __REG32 RSTS_LVR        : 1;
  __REG32 RSTS_BOD        : 1;
  __REG32 RSTS_MCU        : 1;
  __REG32                 : 1;
  __REG32 RSTS_CPU        : 1;
  __REG32                 : 24;
} __gcr_rstsrc_bits;

/* Peripheral Reset Control Resister 1 (GCR_IPRSTC1) */
typedef struct {
  __REG32 CHIP_RST        : 1;
  __REG32 CPU_RST         : 1;
  __REG32                 : 1;
  __REG32 EBI_RST         : 1;
  __REG32                 : 28;
} __gcr_iprstc1_bits;

/* Peripheral Reset Control Resister 2 (GCR_IPRSTC2) */
typedef struct {
  __REG32                 : 1;
  __REG32 GPIO_RST        : 1;
  __REG32 TMR0_RST        : 1;
  __REG32 TMR1_RST        : 1;
  __REG32 TMR2_RST        : 1;
  __REG32 TMR3_RST        : 1;
  __REG32                 : 2;
  __REG32 I2C_RST         : 1;
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 SPI0_RST        : 1;
  __REG32 SPI1_RST        : 1;
  __REG32                 : 2;
  __REG32 UART0_RST       : 1;
  __REG32 UART1_RST       : 1;
  __REG32                 : 2;
  __REG32 PWM03_RST       : 1;
  __REG32 PWM47_RST       : 1;
  __REG32                 : 6;
  __REG32 ADC_RST         : 1;
  __REG32                 : 3;
} __gcr_iprstc2_bits;

/* Brown-Out Detector Control Register (GCR_BODCR) */
typedef struct {
  __REG32 BOD_EN          : 1;
  __REG32 BOD_VL          : 2;
  __REG32 BOD_RSTEN       : 1;
  __REG32 BOD_INTF        : 1;
  __REG32 BOD_LPM         : 1;
  __REG32 BOD_OUT         : 1;
  __REG32 LVR_EN          : 1;
  __REG32                 : 24;
} __gcr_bodcr_bits;

/* Power-On-Reset Control Register (GCR_PORCR */
typedef struct {
  __REG32 POR_DIS_CODE    : 16;
  __REG32                 : 16;
} __gcr_porcr_bits;

/* Multiple function Port0 Control register (GCR_P0_MFP) */
typedef struct {
  __REG32 P0_MFP0         : 1;
  __REG32 P0_MFP1         : 1;
  __REG32 P0_MFP2         : 1;
  __REG32 P0_MFP3         : 1;
  __REG32 P0_MFP4         : 1;
  __REG32 P0_MFP5         : 1;
  __REG32 P0_MFP6         : 1;
  __REG32 P0_MFP7         : 1;
  __REG32 P0_ALT0         : 1;
  __REG32 P0_ALT1         : 1;
  __REG32 P0_ALT2         : 1;
  __REG32 P0_ALT3         : 1;
  __REG32 P0_ALT4         : 1;
  __REG32 P0_ALT5         : 1;
  __REG32 P0_ALT6         : 1;
  __REG32 P0_ALT7         : 1;
  __REG32 P0_TYPE0        : 1;
  __REG32 P0_TYPE1        : 1;
  __REG32 P0_TYPE2        : 1;
  __REG32 P0_TYPE3        : 1;
  __REG32 P0_TYPE4        : 1;
  __REG32 P0_TYPE5        : 1;
  __REG32 P0_TYPE6        : 1;
  __REG32 P0_TYPE7        : 1;
  __REG32                 : 8;
} __gcr_p0_mfp_bits;

/* Multiple function Port1 Control register (GCR_P1_MFP) */
typedef struct {
  __REG32 P1_MFP0         : 1;
  __REG32 P1_MFP1         : 1;
  __REG32 P1_MFP2         : 1;
  __REG32 P1_MFP3         : 1;
  __REG32 P1_MFP4         : 1;
  __REG32 P1_MFP5         : 1;
  __REG32 P1_MFP6         : 1;
  __REG32 P1_MFP7         : 1;
  __REG32 P1_ALT0         : 1;
  __REG32 P1_ALT1         : 1;
  __REG32 P1_ALT2         : 1;
  __REG32 P1_ALT3         : 1;
  __REG32 P1_ALT4         : 1;
  __REG32 P1_ALT5         : 1;
  __REG32 P1_ALT6         : 1;
  __REG32 P1_ALT7         : 1;
  __REG32 P1_TYPE0        : 1;
  __REG32 P1_TYPE1        : 1;
  __REG32 P1_TYPE2        : 1;
  __REG32 P1_TYPE3        : 1;
  __REG32 P1_TYPE4        : 1;
  __REG32 P1_TYPE5        : 1;
  __REG32 P1_TYPE6        : 1;
  __REG32 P1_TYPE7        : 1;
  __REG32                 : 8;
} __gcr_p1_mfp_bits;

/* Multiple function Port2 Control register (GCR_P2_MFP) */
typedef struct {
  __REG32 P2_MFP0         : 1;
  __REG32 P2_MFP1         : 1;
  __REG32 P2_MFP2         : 1;
  __REG32 P2_MFP3         : 1;
  __REG32 P2_MFP4         : 1;
  __REG32 P2_MFP5         : 1;
  __REG32 P2_MFP6         : 1;
  __REG32 P2_MFP7         : 1;
  __REG32 P2_ALT0         : 1;
  __REG32 P2_ALT1         : 1;
  __REG32 P2_ALT2         : 1;
  __REG32 P2_ALT3         : 1;
  __REG32 P2_ALT4         : 1;
  __REG32 P2_ALT5         : 1;
  __REG32 P2_ALT6         : 1;
  __REG32 P2_ALT7         : 1;
  __REG32 P2_TYPE0        : 1;
  __REG32 P2_TYPE1        : 1;
  __REG32 P2_TYPE2        : 1;
  __REG32 P2_TYPE3        : 1;
  __REG32 P2_TYPE4        : 1;
  __REG32 P2_TYPE5        : 1;
  __REG32 P2_TYPE6        : 1;
  __REG32 P2_TYPE7        : 1;
  __REG32                 : 8;
} __gcr_p2_mfp_bits;

/* Multiple function Port3 Control register (GCR_P3_MFP) */
typedef struct {
  __REG32 P3_MFP0         : 1;
  __REG32 P3_MFP1         : 1;
  __REG32 P3_MFP2         : 1;
  __REG32 P3_MFP3         : 1;
  __REG32 P3_MFP4         : 1;
  __REG32 P3_MFP5         : 1;
  __REG32 P3_MFP6         : 1;
  __REG32 P3_MFP7         : 1;
  __REG32 P3_ALT0         : 1;
  __REG32 P3_ALT1         : 1;
  __REG32 P3_ALT2         : 1;
  __REG32 P3_ALT3         : 1;
  __REG32 P3_ALT4         : 1;
  __REG32 P3_ALT5         : 1;
  __REG32 P3_ALT6         : 1;
  __REG32 P3_ALT7         : 1;
  __REG32 P3_TYPE0        : 1;
  __REG32 P3_TYPE1        : 1;
  __REG32 P3_TYPE2        : 1;
  __REG32 P3_TYPE3        : 1;
  __REG32 P3_TYPE4        : 1;
  __REG32 P3_TYPE5        : 1;
  __REG32 P3_TYPE6        : 1;
  __REG32 P3_TYPE7        : 1;
  __REG32                 : 8;
} __gcr_p3_mfp_bits;

/* Multiple function Port4 Control register (GCR_P4_MFP) */
typedef struct {
  __REG32 P4_MFP0         : 1;
  __REG32 P4_MFP1         : 1;
  __REG32 P4_MFP2         : 1;
  __REG32 P4_MFP3         : 1;
  __REG32 P4_MFP4         : 1;
  __REG32 P4_MFP5         : 1;
  __REG32 P4_MFP6         : 1;
  __REG32 P4_MFP7         : 1;
  __REG32 P4_ALT0         : 1;
  __REG32 P4_ALT1         : 1;
  __REG32 P4_ALT2         : 1;
  __REG32 P4_ALT3         : 1;
  __REG32 P4_ALT4         : 1;
  __REG32 P4_ALT5         : 1;
  __REG32 P4_ALT6         : 1;
  __REG32 P4_ALT7         : 1;
  __REG32 P4_TYPE0        : 1;
  __REG32 P4_TYPE1        : 1;
  __REG32 P4_TYPE2        : 1;
  __REG32 P4_TYPE3        : 1;
  __REG32 P4_TYPE4        : 1;
  __REG32 P4_TYPE5        : 1;
  __REG32 P4_TYPE6        : 1;
  __REG32 P4_TYPE7        : 1;
  __REG32                 : 8;
} __gcr_p4_mfp_bits;

/* Register Write-Protection Control Register (GCR_REGWRPROT) */
typedef union {
  /* GCR_REGWRPROT */
  struct
  {
    __REG32 REGWRPROT     : 8;
    __REG32               : 24;
    
  };
  /* GCR_REGPROTDIS */
  struct
  {
    __REG32 REGPROTDIS    : 1;
    __REG32               : 31;
  };
  
} __gcr_regwrprot_bits;

/* Px Bit Mode Control (GPIO_P0_PMD,GPIO_P1_PMD,GPIO_P2_PMD,GPIO_P3_PMD,GPIO_P4_PMD) */
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
} __gpio_px_pmd_bits;

/* Px Bit OFF Digital Enable (GPIO_P0_OFFD,GPIO_P1_OFFD,GPIO_P2_OFFD,GPIO_P3_OFFD,GPIO_P4_OFFD) */
typedef struct {
  __REG32                 : 16;
  __REG32 OFFD            : 8;
  __REG32                 : 8;
} __gpio_px_offd_bits;

/* Px Data Output Value (GPIO_P0_DOUT,GPIO_P1_DOUT,GPIO_P2_DOUT,GPIO_P3_DOUT,GPIO_P4_DOUT) */
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
} __gpio_px_dout_bits;

/* Px Data Output Write Mask (GPIO_P0_DMASK,GPIO_P1_DMASK,GPIO_P2_DMASK,GPIO_P3_DMASK,GPIO_P4_DMASK) */
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
} __gpio_px_dmask_bits;

/* Px Pin Value (GPIO_P0_PIN,GPIO_P1_PIN,GPIO_P2_PIN,GPIO_P3_PIN,GPIO_P4_PIN) */
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
} __gpio_px_pin_bits;

/* Px De-bounce Enable (GPIO_P0_DBEN,GPIO_P1_DBEN,GPIO_P2_DBEN,GPIO_P3_DBEN,GPIO_P4_DBEN) */
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
} __gpio_px_dben_bits;

/* Px Interrupt Mode Control (GPIO_P0_IMD,GPIO_P1_IMD,GPIO_P2_IMD,GPIO_P3_IMD,GPIO_P4_IMD) */
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
} __gpio_px_imd_bits;

/* Px Interrupt Enable (GPIO_P0_IEN,GPIO_P1_IEN,GPIO_P2_IEN,GPIO_P3_IEN,GPIO_P4_IEN) */
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
} __gpio_px_ien_bits;

/* Px Interrupt Trigger Source (GPIO_P0_ISRC,GPIO_P1_ISRC,GPIO_P2_ISRC,GPIO_P3_ISRC,GPIO_P4_ISRC) */
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
} __gpio_px_isrc_bits;

/* Interrupt De-bounce Cycle Control (GPIO_DBNCEON) */
typedef struct {
  __REG32 DBCLKSEL        : 4;
  __REG32 DBCLKSRC        : 1;
  __REG32 ICLK_ON         : 1;
  __REG32                 : 26;
} __gpio_dbncecon_bits;

/* P0.0 Data Output Value (GPIO_P00_DOUT) */
typedef struct {
  __REG32 P00_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p00_dout_bits;

/* P0.1 Data Output Value (GPIO_P01_DOUT) */
typedef struct {
  __REG32 P01_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p01_dout_bits;

/* P0.2 Data Output Value (GPIO_P02_DOUT) */
typedef struct {
  __REG32 P02_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p02_dout_bits;

/* P0.3 Data Output Value (GPIO_P03_DOUT) */
typedef struct {
  __REG32 P03_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p03_dout_bits;

/* P0.4 Data Output Value (GPIO_P04_DOUT) */
typedef struct {
  __REG32 P04_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p04_dout_bits;

/* P0.5 Data Output Value (GPIO_P05_DOUT) */
typedef struct {
  __REG32 P05_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p05_dout_bits;

/* P0.6 Data Output Value (GPIO_P06_DOUT) */
typedef struct {
  __REG32 P06_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p06_dout_bits;

/* P0.7 Data Output Value (GPIO_P07_DOUT) */
typedef struct {
  __REG32 P07_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p07_dout_bits;

/* P1.0 Data Output Value (GPIO_P10_DOUT) */
typedef struct {
  __REG32 P10_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p10_dout_bits;

/* P1.1 Data Output Value (GPIO_P11_DOUT) */
typedef struct {
  __REG32 P11_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p11_dout_bits;

/* P1.2 Data Output Value (GPIO_P12_DOUT) */
typedef struct {
  __REG32 P12_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p12_dout_bits;

/* P1.3 Data Output Value (GPIO_P13_DOUT) */
typedef struct {
  __REG32 P13_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p13_dout_bits;

/* P1.4 Data Output Value (GPIO_P14_DOUT) */
typedef struct {
  __REG32 P14_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p14_dout_bits;

/* P1.5 Data Output Value (GPIO_P15_DOUT) */
typedef struct {
  __REG32 P15_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p15_dout_bits;

/* P1.6 Data Output Value (GPIO_P16_DOUT) */
typedef struct {
  __REG32 P16_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p16_dout_bits;

/* P1.7 Data Output Value (GPIO_P17_DOUT) */
typedef struct {
  __REG32 P17_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p17_dout_bits;

/* P2.0 Data Output Value (GPIO_P20_DOUT) */
typedef struct {
  __REG32 P20_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p20_dout_bits;

/* P2.1 Data Output Value (GPIO_P21_DOUT) */
typedef struct {
  __REG32 P21_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p21_dout_bits;

/* P2.2 Data Output Value (GPIO_P22_DOUT) */
typedef struct {
  __REG32 P22_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p22_dout_bits;

/* P2.3 Data Output Value (GPIO_P23_DOUT) */
typedef struct {
  __REG32 P23_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p23_dout_bits;

/* P2.4 Data Output Value (GPIO_P24_DOUT) */
typedef struct {
  __REG32 P24_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p24_dout_bits;

/* P2.5 Data Output Value (GPIO_P25_DOUT) */
typedef struct {
  __REG32 P25_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p25_dout_bits;

/* P2.6 Data Output Value (GPIO_P26_DOUT) */
typedef struct {
  __REG32 P26_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p26_dout_bits;

/* P2.7 Data Output Value (GPIO_P27_DOUT) */
typedef struct {
  __REG32 P27_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p27_dout_bits;

/* P3.0 Data Output Value (GPIO_P30_DOUT) */
typedef struct {
  __REG32 P30_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p30_dout_bits;

/* P3.1 Data Output Value (GPIO_P31_DOUT) */
typedef struct {
  __REG32 P31_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p31_dout_bits;

/* P3.2 Data Output Value (GPIO_P32_DOUT) */
typedef struct {
  __REG32 P32_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p32_dout_bits;

/* P3.3 Data Output Value (GPIO_P33_DOUT) */
typedef struct {
  __REG32 P33_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p33_dout_bits;

/* P3.4 Data Output Value (GPIO_P34_DOUT) */
typedef struct {
  __REG32 P34_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p34_dout_bits;

/* P3.5 Data Output Value (GPIO_P35_DOUT) */
typedef struct {
  __REG32 P35_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p35_dout_bits;

/* P3.6 Data Output Value (GPIO_P36_DOUT) */
typedef struct {
  __REG32 P36_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p36_dout_bits;

/* P3.7 Data Output Value (GPIO_P37_DOUT) */
typedef struct {
  __REG32 P37_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p37_dout_bits;

/* P4.0 Data Output Value (GPIO_P40_DOUT) */
typedef struct {
  __REG32 P40_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p40_dout_bits;

/* P4.1 Data Output Value (GPIO_P41_DOUT) */
typedef struct {
  __REG32 P41_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p41_dout_bits;

/* P4.2 Data Output Value (GPIO_P42_DOUT) */
typedef struct {
  __REG32 P42_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p42_dout_bits;

/* P4.3 Data Output Value (GPIO_P43_DOUT) */
typedef struct {
  __REG32 P43_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p43_dout_bits;

/* P4.4 Data Output Value (GPIO_P44_DOUT) */
typedef struct {
  __REG32 P44_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p44_dout_bits;

/* P4.5 Data Output Value (GPIO_P45_DOUT) */
typedef struct {
  __REG32 P45_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p45_dout_bits;

/* P4.6 Data Output Value (GPIO_P46_DOUT) */
typedef struct {
  __REG32 P46_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p46_dout_bits;

/* P4.7 Data Output Value (GPIO_P47_DOUT) */
typedef struct {
  __REG32 P47_DOUT        : 1;
  __REG32                 : 31;
} __gpio_p47_dout_bits;

/* I2C Control Register (I2C_I2CON) */
typedef struct {
  __REG32                 : 2;
  __REG32 AA              : 1;
  __REG32 SI              : 1;
  __REG32 STO             : 1;
  __REG32 STA             : 1;
  __REG32 ENS1            : 1;
  __REG32 EI              : 1;
  __REG32                 : 24;
} __i2c_i2con_bits;

/* I2C DATA Register (I2C_I2CDAT) */
typedef struct {
  __REG32 I2CDAT          : 8;
  __REG32                 : 24;
} __i2c_i2cdat_bits;

/* I2C Status Register (I2C_I2CSTATUS) */
typedef struct {
  __REG32 I2CSTATUS       : 8;
  __REG32                 : 24;
} __i2c_i2cstatus_bits;

/* I2C clock divided Register (I2C_I2CLK) */
typedef struct {
  __REG32 I2CLK           : 8;
  __REG32                 : 24;
} __i2c_i2clk_bits;

/* I2C Time out control Register (I2C_I2CTOC) */
typedef struct {
  __REG32 TIF             : 1;
  __REG32 DIV4            : 1;
  __REG32 ENTI            : 1;
  __REG32                 : 29;
} __i2c_i2ctoc_bits;

/* I2C slave Address Register (I2C_I2CADDR0,I2C_I2CADDR1,I2C_I2CADDR2,I2C_I2CADDR3) */
typedef struct {
  __REG32 GC              : 1;
  __REG32 I2CADDR         : 7;
  __REG32                 : 24;
} __i2c_i2caddrx_bits;

/* I2C Slave address Mask Registerx (I2C_I2CADM0,I2C_I2CADM1,I2C_I2CADM2,I2C_I2CADM3) */
typedef struct {
  __REG32                 : 1;
  __REG32 I2ADMx          : 7;
  __REG32                 : 24;
} __i2c_i2cadmx_bits;

/* MCU IRQ0 (BOD) interrupt source identify (INT_IRQ0_SRC) */
/* MCU IRQ1 (WDT) interrupt source identify (INT_IRQ1_SRC) */
/* MCU IRQ2 ((EINT0) interrupt source identify (INT_IRQ2_SRC) */
/* MCU IRQ3 (EINT1) interrupt source identify (INT_IRQ3_SRC) */
/* MCU IRQ4 (P0/1) interrupt source identify (INT_IRQ4_SRC) */
/* MCU IRQ5 (P2/3/4) interrupt source identify (INT_IRQ5_SRC) */
/* MCU IRQ8 (TMR0) interrupt source identify (INT_IRQ8_SRC) */
/* MCU IRQ9 (TMR1) interrupt source identify (INT_IRQ9_SRC)*/
/* MCU IRQ10 (TMR2) interrupt source identify (INT_IRQ10_SRC) */
/* MCU IRQ11 (TMR3) interrupt source identify (INT_IRQ11_SRC) */
/* MCU IRQ12 (URT0) interrupt source identify (INT_IRQ12_SRC) */
/* MCU IRQ13 (URT1) interrupt source identify (INT_IRQ13_SRC) */
/* MCU IRQ14 (SPI0) interrupt source identify (INT_IRQ14_SRC) */
/* MCU IRQ15 (SPI1) interrupt source identify (INT_IRQ15_SRC) */
/* MCU IRQ18 (I2C) interrupt source identify (INT_IRQ18_SRC) */
/* MCU IRQ28 (PWRWU) interrupt source identify (INT_IRQ28_SRC) */
/* MCU IRQ29 (ADC) interrupt source identify (INT_IRQ29_SRC) */
typedef struct {
  __REG32 INT_SRC         : 3;
  __REG32                 : 29;
} __int_irqx_src_bits;

/* MCU IRQ6 (PWMA) interrupt source identify (INT_IRQ6_SRC) */
/* MCU IRQ7 (PWMB) interrupt source identify (INT_IRQ7_SRC) */
typedef struct {
  __REG32 INT_SRC         : 4;
  __REG32                 : 28;
} __int_irq67_src_bits;

/* NMI source interrupt select control register (INT_NMI_SEL)*/
typedef struct {
  __REG32 NMI_SEL         : 5;
  __REG32                 : 27;
} __int_nmi_sel_bits;

/* PWM Pre-scalar Register (PWMx_PPR) */
typedef struct {
  __REG32 CP01            : 8;
  __REG32 CP23            : 8;
  __REG32 DZI01           : 8;
  __REG32 DZI23           : 8;
} __pwm_ppr_bits;

/* PWM Clock Select Register (PWMx_CSR) */
typedef struct {
  __REG32 CSR0            : 3;
  __REG32                 : 1;
  __REG32 CSR1            : 3;
  __REG32                 : 1;
  __REG32 CSR2            : 3;
  __REG32                 : 1;
  __REG32 CSR3            : 3;
  __REG32                 : 17;
} __pwm_csr_bits;

/* PWM Control Register (PWMx_PCR) */
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
} __pwm_pcr_bits;

/* PWM Counter Register (PWMx_CNR0,PWMx_CNR1,PWMx_CNR2,PWMx_CNR3) */
typedef struct {
  __REG32 CNR             : 16;
  __REG32                 : 16;
} __pwm_cnrx_bits;

/* PWM Comparator Register (PWMx_CMR0,PWMx_CMR1,PWMx_CMR2,PWMx_CMR3) */
typedef struct {
  __REG32 CMR             : 16;
  __REG32                 : 16;
} __pwm_cmrx_bits;

/* PWM Data Register (PWMx_PDR0,PWMx_PDR1,PWMx_PDR2,PWMx_PDR3) */
typedef struct {
  __REG32 PDR             : 16;
  __REG32                 : 16;
} __pwm_pdrx_bits;

/* PWM Interrupt Enable Register (PWMx_PIER) */
typedef struct {
  __REG32 PWMIE0          : 1;
  __REG32 PWMIE1          : 1;
  __REG32 PWMIE2          : 1;
  __REG32 PWMIE3          : 1;
  __REG32                 : 28;
} __pwm_pier_bits;

/* PWM Interrupt Indication Register (PWMx_PIIR) */
typedef struct {
  __REG32 PWMIF0          : 1;
  __REG32 PWMIF1          : 1;
  __REG32 PWMIF2          : 1;
  __REG32 PWMIF3          : 1;
  __REG32                 : 28;
} __pwm_piir_bits;

/* Capture Control Register (PWMx_CCR0) */
typedef struct {
  __REG32 INV0            : 1;
  __REG32 CRL_IE0         : 1;
  __REG32 CFL_IE0         : 1;
  __REG32 CAPCH0EN        : 1;
  __REG32 CAPIF0          : 1;
  __REG32                 : 1;
  __REG32 CRLRI0          : 1;
  __REG32 CFLRI0          : 1;
  __REG32                 : 8;
  __REG32 INV1            : 1;
  __REG32 CRL_IE1         : 1;
  __REG32 CFL_IE1         : 1;
  __REG32 CAPCH1EN        : 1;
  __REG32 CAPIF1          : 1;
  __REG32                 : 1;
  __REG32 CRLRI1          : 1;
  __REG32 CFLRI1          : 1;
  __REG32                 : 8;
} __pwm_ccr0_bits;

/* Capture Control Register (PWMx_CCR2) */
typedef struct {
  __REG32 INV2            : 1;
  __REG32 CRL_IE2         : 1;
  __REG32 CFL_IE2         : 1;
  __REG32 CAPCH2EN        : 1;
  __REG32 CAPIF2          : 1;
  __REG32                 : 1;
  __REG32 CRLRI2          : 1;
  __REG32 CFLRI2          : 1;
  __REG32                 : 8;
  __REG32 INV3            : 1;
  __REG32 CRL_IE3         : 1;
  __REG32 CFL_IE3         : 1;
  __REG32 CAPCH3EN        : 1;
  __REG32 CAPIF3          : 1;
  __REG32                 : 1;
  __REG32 CRLRI3          : 1;
  __REG32 CFLRI3          : 1;
  __REG32                 : 8;
} __pwm_ccr2_bits;

/* Capture Rising Latch Register (PWMx_CRLR0,PWMx_CRLR1,PWMx_CRLR2,PWMx_CRLR3) */
typedef struct {
  __REG32 CRLR            : 16;
  __REG32                 : 16;
} __pwm_crlrx_bits;

/* Capture Falling Latch Register (PWMx_CFLR0,PWMx_CFLR1,PWMx_CFLR2,PWMx_CFLR3) */
typedef struct {
  __REG32 CFLR            : 16;
  __REG32                 : 16;
} __pwm_cflrx_bits;

/* Capture Input Enable Register (PWMx_CAPENR) */
typedef struct {
  __REG32 CAPENR          : 4;
  __REG32                 : 28;
} __pwm_capenr_bits;

/* PWM Output Enable (PWMx_POE) */
typedef struct {
  __REG32 PWM0            : 1;
  __REG32 PWM1            : 1;
  __REG32 PWM2            : 1;
  __REG32 PWM3            : 1;
  __REG32                 : 28;
} __pwm_poe_bits;

/* SysTick Control and Status Register (SCS_SYST_CSR) */
typedef struct {
  __REG32 ENABLE          : 1;
  __REG32 TICKINT         : 1;
  __REG32 CLKSRC          : 1;
  __REG32                 : 13;
  __REG32 COUNTFLAG       : 1;
  __REG32                 : 15;
} __scs_syst_csr_bits;

/* SysTick Reload value Register (SCS_SYST_RVR) */
typedef struct {
  __REG32 RELOAD          : 24;
  __REG32                 : 8;
} __scs_syst_rvr_bits;

/* SysTick Current value Register (SCS_SYST_CVR) */
typedef struct {
  __REG32 CURRENT         : 24;
  __REG32                 : 8;
} __scs_syst_cvr_bits;

/* IRQ0 ~ IRQ3 Priority Control Register (SCS_NVIC_IPR0) */
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

/* IRQ4 ~ IRQ7 Priority Control Register (SCS_NVIC_IPR1) */
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

/* IRQ8 ~ IRQ11 Priority Control Register (SCS_NVIC_IPR2) */
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

/* IRQ12 ~ IRQ15 Priority Control Register (SCS_NVIC_IPR3) */
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

/* IRQ16 ~ IRQ19 Priority Control Register (SCS_NVIC_IPR4) */
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

/* IRQ20 ~ IRQ23 Priority Control Register (SCS_NVIC_IPR5) */
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

/* IRQ24 ~ IRQ27 Priority Control Register (SCS_NVIC_IPR6) */
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

/* IRQ28 ~ IRQ31 Priority Control Register (SCS_NVIC_IPR7) */
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

/* CPUID Base Register (SCS_CPUID) */
typedef struct {
  __REG32 REVISION        : 4;
  __REG32 PARTNO          : 12;
  __REG32 PART            : 4;
  __REG32                 : 4;
  __REG32 IMPLEMENTER     : 8;
} __scs_cpuid_bits;

/* Interrupt Control State Register (SCS_ICSR) */
typedef struct {
  __REG32 VECTACTIVE      : 6;
  __REG32                 : 6;
  __REG32 VECTPENDING     : 6;
  __REG32                 : 4;
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

/* Application Interrupt and Reset Control Register (SCS_AIRCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 VECTCLRACTIVE   : 1;
  __REG32 SYSRESETREQ     : 1;
  __REG32                 : 13;
  __REG32 VECTORKEY       : 16;
} __scs_aircr_bits;

/* System Control Register (SCS_SCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 SLEEPONEXIT     : 1;
  __REG32 SLEEPDEEP       : 1;
  __REG32                 : 1;
  __REG32 SEVONPEND       : 1;
  __REG32                 : 27;
} __scs_scr_bits;

/* System Handler Priority Register 2 (SCS_SHPR2) */
typedef struct {
  __REG32                 : 30;
  __REG32 PRI_11          : 2;
} __scs_shpr2_bits;

/* System Handler Priority Register 3 (SCS_SHPR3) */
typedef struct {
  __REG32                 : 22;
  __REG32 PRI_14          : 2;
  __REG32                 : 6;
  __REG32 PRI_15          : 2;
} __scs_shpr3_bits;

/* Timer Control and Status Register (TMRx_TCSR0,TMRx_TCSR1,TMRx_TCSR2,TMRx_TCSR3) */
typedef struct {
  __REG32 PRESCALE        : 8;
  __REG32                 : 8;
  __REG32 TDR_EN          : 1;
  __REG32                 : 8;
  __REG32 CACT            : 1;
  __REG32 CRST            : 1;
  __REG32 MODE            : 2;
  __REG32 IE              : 1;
  __REG32 CEN             : 1;
  __REG32 DBGACK_TMR      : 1;
} __tmr_tcsrx_bits;

/* Timer Compare Register (TMRx_TCMPR0,TMRx_TCMPR1,TMRx_TCMPR2,TMRx_TCMPR3) */
typedef struct {
  __REG32 TCMP            : 24;
  __REG32                 : 8;
} __tmr_tcmprx_bits;

/* Timer Interrupt Status Register (TMRx_TISR0,TMRx_TISR1,TMRx_TISR2,TMRx_TISR3) */
typedef struct {
  __REG32 TIF             : 1;
  __REG32                 : 31;
} __tmr_tisrx_bits;

/* Timer Data Register (TMRx_TDR0,TMRx_TDR1,TMRx_TDR2,TMRx_TDR3) */
typedef struct {
  __REG32 TDR             : 24;
  __REG32                 : 8;
} __tmr_tdrx_bits;

/* Watchdog Timer Control Register (WDT_WTCR) */
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

/* Control and Status Register (SPIx_SPI_CNTRL) */
typedef struct {
  __REG32 GO_BUSY         : 1;
  __REG32 RX_NEG          : 1;
  __REG32 TX_NEG          : 1;
  __REG32 TX_BIT_LEN      : 5;
  __REG32 TX_NUM          : 2;
  __REG32 LSB             : 1;
  __REG32 CLKP            : 1;
  __REG32 SP_CYCLE        : 4;
  __REG32 IF              : 1;
  __REG32 IE              : 1;
  __REG32 SLAVE           : 1;
  __REG32 REORDER         : 2;
  __REG32                 : 2;
  __REG32 VARCLK_EN       : 1;
  __REG32                 : 8;
} __spi_spi_cntrl_bits;

/* Clock Divider Register (SPIx_SPI_DIVIDER) */
typedef struct {
  __REG32 DIVIDER         : 16;
  __REG32 DIVIDER2        : 16;
} __spi_spi_divider_bits;

/* Slave Select Register (SPIx_SPI_SSR) */
typedef struct {
  __REG32 SSR             : 1;
  __REG32                 : 1;
  __REG32 SS_LVL          : 1;
  __REG32 AUTOSS          : 1;
  __REG32 SS_LTRIG        : 1;
  __REG32 LTRIG_FLAG      : 1;
  __REG32                 : 26;
} __spi_spi_ssr_bits;

/* Receive Buffer Register (UARTx_UA_RBR) */
/* Transmit Holding Register (UARTx_UA_THR) */
typedef union {
  /* UART0_UA_RBR */
  /* UART1_UA_RBR */
  struct {
    __REG32 RBR               : 8;
    __REG32                   :24;
  };
  /* UART0_UA_THR */
  /* UART1_UA_THR */
  struct {
    __REG32 THR               : 8;
    __REG32                   :24;  
  };
  
  struct {
    __REG32 DATA              : 8;
    __REG32                   :24;  
  };
} __uartx_ua_rbr_thr_bits;

/* UART Interrupt Enable Register (UARTx_UA_IER) */
typedef struct {
  __REG32 RDA_IEN         : 1;
  __REG32 THRE_IEN        : 1;
  __REG32 RLS_IEN         : 1;
  __REG32 MODEM_IEN       : 1;
  __REG32 RTO_IEN         : 1;
  __REG32                 : 1;
  __REG32 WAKE_EN         : 1;
  __REG32                 : 1;
  __REG32                 : 3;
  __REG32 TIME_OUT_EN     : 1;
  __REG32 AUTO_RTS_EN     : 1;
  __REG32 AUTO_CTS_EN     : 1;
  __REG32                 : 18;
} __uartx_ua_ier_bits;

/* UART FIFO Control Register (UARTx_UA_FCR) */
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
} __uartx_ua_fcr_bits;

/* UART Line Control Register (UARTx_UA_LCR) */
typedef struct {
  __REG32 WLS             : 2;
  __REG32 NSB             : 1;
  __REG32 PBE             : 1;
  __REG32 EPE             : 1;
  __REG32 SPE             : 1;
  __REG32 BCB             : 1;
  __REG32                 : 25;
} __uartx_ua_lcr_bits;

/* UART Modem Control Register (UARTx_UA_MCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 RTS             : 1;
  __REG32                 : 7;
  __REG32 LEV_RTS         : 1;
  __REG32                 : 3;
  __REG32 RTS_ST          : 1;
  __REG32                 : 18;
} __uartx_ua_mcr_bits;

/* UART Modem Status Register (UARTx_UA_MSR) */
typedef struct {
  __REG32 DCTSF           : 1;
  __REG32                 : 3;
  __REG32 CTS_ST          : 1;
  __REG32                 : 3;
  __REG32 LEV_CTS         : 1;
  __REG32                 : 23;
} __uartx_ua_msr_bits;

/* UART FIFO Status Register (UARTx_UA_FSR) */
typedef struct {
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 RS_485_ADD_DETF : 1;
  __REG32 PEF             : 1;
  __REG32 FEF             : 1;
  __REG32 BIF             : 1;
  __REG32                 : 1;
  __REG32 RX_POINTER      : 6;
  __REG32 RX_EMPTY        : 1;
  __REG32 RX_OVER         : 1;
  __REG32 TX_POINTER      : 6;
  __REG32 TX_EMPTY        : 1;
  __REG32 TX_OVER         : 1;
  __REG32                 : 1;
  __REG32                 : 3;
  __REG32 TE_FLAG         : 1;
  __REG32                 : 3;
} __uartx_ua_fsr_bits;

/* UART Interrupt Status Register (UARTx_UA_ISR) */
typedef struct {
  __REG32 RDA_IF          : 1;
  __REG32 THRE_IF         : 1;
  __REG32 RLS_IF          : 1;
  __REG32 MODEM_IF        : 1;
  __REG32 TOUT_IF         : 1;
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 RDA_INT         : 1;
  __REG32 THRE_INT        : 1;
  __REG32 RLS_INT         : 1;
  __REG32 MODEM_INT       : 1;
  __REG32 TOUT_INT        : 1;
  __REG32                 : 1;
  __REG32                 : 18;
} __uartx_ua_isr_bits;

/* UART Time Out Register (UARTx_UA_TOR) */
typedef struct {
  __REG32 TOIC            : 7;
  __REG32                 : 1;
  __REG32 DLY             : 8;
  __REG32                 : 16;
} __uartx_ua_tor_bits;

/* UART Baud Rate Divisor Register (UARTx_UA_BAUD) */
typedef struct {
  __REG32 BRD             :16;
  __REG32                 : 8;
  __REG32 DIVIDER_X       : 4;
  __REG32 DIV_X_ONE       : 1;
  __REG32 DIV_X_EN        : 1;
  __REG32                 : 2;
} __uartx_ua_baud_bits;

/* UART IrDA Control Register (UARTx_UA_IRCR) */
typedef struct {
  __REG32                 : 1;
  __REG32 TX_SELECT       : 1;
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 INV_TX          : 1;
  __REG32 INV_RX          : 1;
  __REG32                 : 25;
} __uartx_ua_ircr_bits;

/* UART RS485 Control State Register (UARTx_UA_ALT_CSR) */
typedef struct {
  __REG32                 : 8;
  __REG32 RS_485_NMM      : 1;
  __REG32 RS_485_AAD      : 1;
  __REG32 RS_485_AUD      : 1;
  __REG32                 : 4;
  __REG32 RS_485_ADD_EN   : 1;
  __REG32                 : 1;
  __REG32                 : 7;
  __REG32 ADDR_MATCH      : 8;
} __uartx_ua_alt_csr_bits;

/* UART Function Select Register (UARTx_UA_FUN_SEL) */
typedef struct {
  __REG32 FUN_SEL         : 2;
  __REG32                 : 30;
} __uartx_ua_fun_sel_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/***************************************************************************/
/* Common declarations  ****************************************************/

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ADDR0,             0x400E0000,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR1,             0x400E0004,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR2,             0x400E0008,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR3,             0x400E000C,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR4,             0x400E0010,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR5,             0x400E0014,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR6,             0x400E0018,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADDR7,             0x400E001C,__READ       ,__adc_addrx_bits);
__IO_REG32_BIT(ADC_ADCR,              0x400E0020,__READ_WRITE ,__adc_adcr_bits);
__IO_REG32_BIT(ADC_ADCHER,            0x400E0024,__READ_WRITE ,__adc_adcher_bits);
__IO_REG32_BIT(ADC_ADCMPR0,           0x400E0028,__READ_WRITE ,__adc_adcmprx_bits);
__IO_REG32_BIT(ADC_ADCMPR1,           0x400E002C,__READ_WRITE ,__adc_adcmprx_bits);
__IO_REG32_BIT(ADC_ADSR,              0x400E0030,__READ_WRITE ,__adc_adsr_bits);
__IO_REG32_BIT(ADC_ADCALR,            0x400E0034,__READ_WRITE ,__adc_adcalr_bits);

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
__IO_REG32_BIT(CLK_PLLCON,            0x50000220,__READ_WRITE ,__clk_pllcon_bits);
__IO_REG32_BIT(CLK_FRQDIV,            0x50000224,__READ_WRITE ,__clk_frqdiv_bits);

/***************************************************************************
 **
 ** EBI_CTL
 **
 ***************************************************************************/
__IO_REG32_BIT(EBI_CTL_EBICON,        0x50010000,__READ_WRITE ,__ebi_ctl_ebicon_bits);
__IO_REG32_BIT(EBI_CTL_EXTIME,        0x50010004,__READ_WRITE ,__ebi_ctl_extime_bits);

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

/***************************************************************************
 **
 ** GCR
 **
 ***************************************************************************/
__IO_REG32(    GCR_PDID,              0x50000000,__READ       );
__IO_REG32_BIT(GCR_RSTSRC,            0x50000004,__READ_WRITE ,__gcr_rstsrc_bits);
__IO_REG32_BIT(GCR_IPRSTC1,           0x50000008,__READ_WRITE ,__gcr_iprstc1_bits);
__IO_REG32_BIT(GCR_IPRSTC2,           0x5000000C,__READ_WRITE ,__gcr_iprstc2_bits);
__IO_REG32_BIT(GCR_BODCR,             0x50000018,__READ_WRITE ,__gcr_bodcr_bits);
__IO_REG32_BIT(GCR_PORCR,             0x50000024,__READ_WRITE ,__gcr_porcr_bits);
__IO_REG32_BIT(GCR_P0_MFP,            0x50000030,__READ_WRITE ,__gcr_p0_mfp_bits);
__IO_REG32_BIT(GCR_P1_MFP,            0x50000034,__READ_WRITE ,__gcr_p1_mfp_bits);
__IO_REG32_BIT(GCR_P2_MFP,            0x50000038,__READ_WRITE ,__gcr_p2_mfp_bits);
__IO_REG32_BIT(GCR_P3_MFP,            0x5000003C,__READ_WRITE ,__gcr_p3_mfp_bits);
__IO_REG32_BIT(GCR_P4_MFP,            0x50000040,__READ_WRITE ,__gcr_p4_mfp_bits);
__IO_REG32_BIT(GCR_REGWRPROT,         0x50000100,__READ_WRITE ,__gcr_regwrprot_bits);
#define GCR_REGPROTDIS        GCR_REGWRPROT
#define GCR_REGPROTDIS_bit    GCR_REGWRPROT_bit

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIO_P0_PMD,           0x50004000,__READ_WRITE ,__gpio_px_pmd_bits);
__IO_REG32_BIT(GPIO_P0_OFFD,          0x50004004,__READ_WRITE ,__gpio_px_offd_bits);
__IO_REG32_BIT(GPIO_P0_DOUT,          0x50004008,__READ_WRITE ,__gpio_px_dout_bits);
__IO_REG32_BIT(GPIO_P0_DMASK,         0x5000400C,__READ_WRITE ,__gpio_px_dmask_bits);
__IO_REG32_BIT(GPIO_P0_PIN,           0x50004010,__READ       ,__gpio_px_pin_bits);
__IO_REG32_BIT(GPIO_P0_DBEN,          0x50004014,__READ_WRITE ,__gpio_px_dben_bits);
__IO_REG32_BIT(GPIO_P0_IMD,           0x50004018,__READ_WRITE ,__gpio_px_imd_bits);
__IO_REG32_BIT(GPIO_P0_IEN,           0x5000401C,__READ_WRITE ,__gpio_px_ien_bits);
__IO_REG32_BIT(GPIO_P0_ISRC,          0x50004020,__READ_WRITE ,__gpio_px_isrc_bits);
__IO_REG32_BIT(GPIO_P1_PMD,           0x50004040,__READ_WRITE ,__gpio_px_pmd_bits);
__IO_REG32_BIT(GPIO_P1_OFFD,          0x50004044,__READ_WRITE ,__gpio_px_offd_bits);
__IO_REG32_BIT(GPIO_P1_DOUT,          0x50004048,__READ_WRITE ,__gpio_px_dout_bits);
__IO_REG32_BIT(GPIO_P1_DMASK,         0x5000404C,__READ_WRITE ,__gpio_px_dmask_bits);
__IO_REG32_BIT(GPIO_P1_PIN,           0x50004050,__READ       ,__gpio_px_pin_bits);
__IO_REG32_BIT(GPIO_P1_DBEN,          0x50004054,__READ_WRITE ,__gpio_px_dben_bits);
__IO_REG32_BIT(GPIO_P1_IMD,           0x50004058,__READ_WRITE ,__gpio_px_imd_bits);
__IO_REG32_BIT(GPIO_P1_IEN,           0x5000405C,__READ_WRITE ,__gpio_px_ien_bits);
__IO_REG32_BIT(GPIO_P1_ISRC,          0x50004060,__READ_WRITE ,__gpio_px_isrc_bits);
__IO_REG32_BIT(GPIO_P2_PMD,           0x50004080,__READ_WRITE ,__gpio_px_pmd_bits);
__IO_REG32_BIT(GPIO_P2_OFFD,          0x50004084,__READ_WRITE ,__gpio_px_offd_bits);
__IO_REG32_BIT(GPIO_P2_DOUT,          0x50004088,__READ_WRITE ,__gpio_px_dout_bits);
__IO_REG32_BIT(GPIO_P2_DMASK,         0x5000408C,__READ_WRITE ,__gpio_px_dmask_bits);
__IO_REG32_BIT(GPIO_P2_PIN,           0x50004090,__READ       ,__gpio_px_pin_bits);
__IO_REG32_BIT(GPIO_P2_DBEN,          0x50004094,__READ_WRITE ,__gpio_px_dben_bits);
__IO_REG32_BIT(GPIO_P2_IMD,           0x50004098,__READ_WRITE ,__gpio_px_imd_bits);
__IO_REG32_BIT(GPIO_P2_IEN,           0x5000409C,__READ_WRITE ,__gpio_px_ien_bits);
__IO_REG32_BIT(GPIO_P2_ISRC,          0x500040A0,__READ_WRITE ,__gpio_px_isrc_bits);
__IO_REG32_BIT(GPIO_P3_PMD,           0x500040C0,__READ_WRITE ,__gpio_px_pmd_bits);
__IO_REG32_BIT(GPIO_P3_OFFD,          0x500040C4,__READ_WRITE ,__gpio_px_offd_bits);
__IO_REG32_BIT(GPIO_P3_DOUT,          0x500040C8,__READ_WRITE ,__gpio_px_dout_bits);
__IO_REG32_BIT(GPIO_P3_DMASK,         0x500040CC,__READ_WRITE ,__gpio_px_dmask_bits);
__IO_REG32_BIT(GPIO_P3_PIN,           0x500040D0,__READ       ,__gpio_px_pin_bits);
__IO_REG32_BIT(GPIO_P3_DBEN,          0x500040D4,__READ_WRITE ,__gpio_px_dben_bits);
__IO_REG32_BIT(GPIO_P3_IMD,           0x500040D8,__READ_WRITE ,__gpio_px_imd_bits);
__IO_REG32_BIT(GPIO_P3_IEN,           0x500040DC,__READ_WRITE ,__gpio_px_ien_bits);
__IO_REG32_BIT(GPIO_P3_ISRC,          0x500040E0,__READ_WRITE ,__gpio_px_isrc_bits);
__IO_REG32_BIT(GPIO_P4_PMD,           0x50004100,__READ_WRITE ,__gpio_px_pmd_bits);
__IO_REG32_BIT(GPIO_P4_OFFD,          0x50004104,__READ_WRITE ,__gpio_px_offd_bits);
__IO_REG32_BIT(GPIO_P4_DOUT,          0x50004108,__READ_WRITE ,__gpio_px_dout_bits);
__IO_REG32_BIT(GPIO_P4_DMASK,         0x5000410C,__READ_WRITE ,__gpio_px_dmask_bits);
__IO_REG32_BIT(GPIO_P4_PIN,           0x50004110,__READ       ,__gpio_px_pin_bits);
__IO_REG32_BIT(GPIO_P4_DBEN,          0x50004114,__READ_WRITE ,__gpio_px_dben_bits);
__IO_REG32_BIT(GPIO_P4_IMD,           0x50004118,__READ_WRITE ,__gpio_px_imd_bits);
__IO_REG32_BIT(GPIO_P4_IEN,           0x5000411C,__READ_WRITE ,__gpio_px_ien_bits);
__IO_REG32_BIT(GPIO_P4_ISRC,          0x50004120,__READ_WRITE ,__gpio_px_isrc_bits);
__IO_REG32_BIT(GPIO_DBNCECON,         0x50004180,__READ_WRITE ,__gpio_dbncecon_bits);
__IO_REG32_BIT(GPIO_P00_DOUT,         0x50004200,__READ_WRITE ,__gpio_p00_dout_bits);
__IO_REG32_BIT(GPIO_P01_DOUT,         0x50004204,__READ_WRITE ,__gpio_p01_dout_bits);
__IO_REG32_BIT(GPIO_P02_DOUT,         0x50004208,__READ_WRITE ,__gpio_p02_dout_bits);
__IO_REG32_BIT(GPIO_P03_DOUT,         0x5000420C,__READ_WRITE ,__gpio_p03_dout_bits);
__IO_REG32_BIT(GPIO_P04_DOUT,         0x50004210,__READ_WRITE ,__gpio_p04_dout_bits);
__IO_REG32_BIT(GPIO_P05_DOUT,         0x50004214,__READ_WRITE ,__gpio_p05_dout_bits);
__IO_REG32_BIT(GPIO_P06_DOUT,         0x50004218,__READ_WRITE ,__gpio_p06_dout_bits);
__IO_REG32_BIT(GPIO_P07_DOUT,         0x5000421C,__READ_WRITE ,__gpio_p07_dout_bits);
__IO_REG32_BIT(GPIO_P10_DOUT,         0x50004220,__READ_WRITE ,__gpio_p10_dout_bits);
__IO_REG32_BIT(GPIO_P11_DOUT,         0x50004224,__READ_WRITE ,__gpio_p11_dout_bits);
__IO_REG32_BIT(GPIO_P12_DOUT,         0x50004228,__READ_WRITE ,__gpio_p12_dout_bits);
__IO_REG32_BIT(GPIO_P13_DOUT,         0x5000422C,__READ_WRITE ,__gpio_p13_dout_bits);
__IO_REG32_BIT(GPIO_P14_DOUT,         0x50004230,__READ_WRITE ,__gpio_p14_dout_bits);
__IO_REG32_BIT(GPIO_P15_DOUT,         0x50004234,__READ_WRITE ,__gpio_p15_dout_bits);
__IO_REG32_BIT(GPIO_P16_DOUT,         0x50004238,__READ_WRITE ,__gpio_p16_dout_bits);
__IO_REG32_BIT(GPIO_P17_DOUT,         0x5000423C,__READ_WRITE ,__gpio_p17_dout_bits);
__IO_REG32_BIT(GPIO_P20_DOUT,         0x50004240,__READ_WRITE ,__gpio_p20_dout_bits);
__IO_REG32_BIT(GPIO_P21_DOUT,         0x50004244,__READ_WRITE ,__gpio_p21_dout_bits);
__IO_REG32_BIT(GPIO_P22_DOUT,         0x50004248,__READ_WRITE ,__gpio_p22_dout_bits);
__IO_REG32_BIT(GPIO_P23_DOUT,         0x5000424C,__READ_WRITE ,__gpio_p23_dout_bits);
__IO_REG32_BIT(GPIO_P24_DOUT,         0x50004250,__READ_WRITE ,__gpio_p24_dout_bits);
__IO_REG32_BIT(GPIO_P25_DOUT,         0x50004254,__READ_WRITE ,__gpio_p25_dout_bits);
__IO_REG32_BIT(GPIO_P26_DOUT,         0x50004258,__READ_WRITE ,__gpio_p26_dout_bits);
__IO_REG32_BIT(GPIO_P27_DOUT,         0x5000425C,__READ_WRITE ,__gpio_p27_dout_bits);
__IO_REG32_BIT(GPIO_P30_DOUT,         0x50004260,__READ_WRITE ,__gpio_p30_dout_bits);
__IO_REG32_BIT(GPIO_P31_DOUT,         0x50004264,__READ_WRITE ,__gpio_p31_dout_bits);
__IO_REG32_BIT(GPIO_P32_DOUT,         0x50004268,__READ_WRITE ,__gpio_p32_dout_bits);
__IO_REG32_BIT(GPIO_P33_DOUT,         0x5000426C,__READ_WRITE ,__gpio_p33_dout_bits);
__IO_REG32_BIT(GPIO_P34_DOUT,         0x50004270,__READ_WRITE ,__gpio_p34_dout_bits);
__IO_REG32_BIT(GPIO_P35_DOUT,         0x50004274,__READ_WRITE ,__gpio_p35_dout_bits);
__IO_REG32_BIT(GPIO_P36_DOUT,         0x50004278,__READ_WRITE ,__gpio_p36_dout_bits);
__IO_REG32_BIT(GPIO_P37_DOUT,         0x5000427C,__READ_WRITE ,__gpio_p37_dout_bits);
__IO_REG32_BIT(GPIO_P40_DOUT,         0x50004280,__READ_WRITE ,__gpio_p40_dout_bits);
__IO_REG32_BIT(GPIO_P41_DOUT,         0x50004284,__READ_WRITE ,__gpio_p41_dout_bits);
__IO_REG32_BIT(GPIO_P42_DOUT,         0x50004288,__READ_WRITE ,__gpio_p42_dout_bits);
__IO_REG32_BIT(GPIO_P43_DOUT,         0x5000428C,__READ_WRITE ,__gpio_p43_dout_bits);
__IO_REG32_BIT(GPIO_P44_DOUT,         0x50004290,__READ_WRITE ,__gpio_p44_dout_bits);
__IO_REG32_BIT(GPIO_P45_DOUT,         0x50004294,__READ_WRITE ,__gpio_p45_dout_bits);
__IO_REG32_BIT(GPIO_P46_DOUT,         0x50004298,__READ_WRITE ,__gpio_p46_dout_bits);
__IO_REG32_BIT(GPIO_P47_DOUT,         0x5000429C,__READ_WRITE ,__gpio_p47_dout_bits);

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
 ** INT
 **
 ***************************************************************************/
__IO_REG32_BIT(INT_IRQ0_SRC,          0x50000300,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ1_SRC,          0x50000304,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ2_SRC,          0x50000308,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ3_SRC,          0x5000030C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ4_SRC,          0x50000310,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ5_SRC,          0x50000314,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ6_SRC,          0x50000318,__READ       ,__int_irq67_src_bits);
__IO_REG32_BIT(INT_IRQ7_SRC,          0x5000031C,__READ       ,__int_irq67_src_bits);
__IO_REG32_BIT(INT_IRQ8_SRC,          0x50000320,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ9_SRC,          0x50000324,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ10_SRC,         0x50000328,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ11_SRC,         0x5000032C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ12_SRC,         0x50000330,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ13_SRC,         0x50000334,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ14_SRC,         0x50000338,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ15_SRC,         0x5000033C,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ18_SRC,         0x50000348,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ21_SRC,         0x50000354,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ22_SRC,         0x50000358,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ28_SRC,         0x50000370,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ29_SRC,         0x50000374,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_IRQ30_SRC,         0x50000378,__READ       ,__int_irqx_src_bits);
__IO_REG32_BIT(INT_NMI_SEL,           0x50000380,__READ_WRITE ,__int_nmi_sel_bits);
__IO_REG32(    INT_MCU_IRQ,           0x50000384,__READ_WRITE );

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMA_PPR,              0x40040000,__READ_WRITE ,__pwm_ppr_bits);
__IO_REG32_BIT(PWMA_CSR,              0x40040004,__READ_WRITE ,__pwm_csr_bits);
__IO_REG32_BIT(PWMA_PCR,              0x40040008,__READ_WRITE ,__pwm_pcr_bits);
__IO_REG32_BIT(PWMA_CNR0,             0x4004000C,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMA_CMR0,             0x40040010,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMA_PDR0,             0x40040014,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMA_CNR1,             0x40040018,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMA_CMR1,             0x4004001C,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMA_PDR1,             0x40040020,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMA_CNR2,             0x40040024,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMA_CMR2,             0x40040028,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMA_PDR2,             0x4004002C,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMA_CNR3,             0x40040030,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMA_CMR3,             0x40040034,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMA_PDR3,             0x40040038,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMA_PIER,             0x40040040,__READ_WRITE ,__pwm_pier_bits);
__IO_REG32_BIT(PWMA_PIIR,             0x40040044,__READ_WRITE ,__pwm_piir_bits);
__IO_REG32_BIT(PWMA_CCR0,             0x40040050,__READ_WRITE ,__pwm_ccr0_bits);
__IO_REG32_BIT(PWMA_CCR2,             0x40040054,__READ_WRITE ,__pwm_ccr2_bits);
__IO_REG32_BIT(PWMA_CRLR0,            0x40040058,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMA_CFLR0,            0x4004005C,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMA_CRLR1,            0x40040060,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMA_CFLR1,            0x40040064,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMA_CRLR2,            0x40040068,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMA_CFLR2,            0x4004006C,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMA_CRLR3,            0x40040070,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMA_CFLR3,            0x40040074,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMA_CAPENR,           0x40040078,__READ_WRITE ,__pwm_capenr_bits);
__IO_REG32_BIT(PWMA_POE,              0x4004007C,__READ_WRITE ,__pwm_poe_bits);
__IO_REG32_BIT(PWMB_PPR,              0x40140000,__READ_WRITE ,__pwm_ppr_bits);
__IO_REG32_BIT(PWMB_CSR,              0x40140004,__READ_WRITE ,__pwm_csr_bits);
__IO_REG32_BIT(PWMB_PCR,              0x40140008,__READ_WRITE ,__pwm_pcr_bits);
__IO_REG32_BIT(PWMB_CNR0,             0x4014000C,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMB_CMR0,             0x40140010,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMB_PDR0,             0x40140014,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMB_CNR1,             0x40140018,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMB_CMR1,             0x4014001C,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMB_PDR1,             0x40140020,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMB_CNR2,             0x40140024,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMB_CMR2,             0x40140028,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMB_PDR2,             0x4014002C,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMB_CNR3,             0x40140030,__READ_WRITE ,__pwm_cnrx_bits);
__IO_REG32_BIT(PWMB_CMR3,             0x40140034,__READ_WRITE ,__pwm_cmrx_bits);
__IO_REG32_BIT(PWMB_PDR3,             0x40140038,__READ       ,__pwm_pdrx_bits);
__IO_REG32_BIT(PWMB_PIER,             0x40140040,__READ_WRITE ,__pwm_pier_bits);
__IO_REG32_BIT(PWMB_PIIR,             0x40140044,__READ_WRITE ,__pwm_piir_bits);
__IO_REG32_BIT(PWMB_CCR0,             0x40140050,__READ_WRITE ,__pwm_ccr0_bits);
__IO_REG32_BIT(PWMB_CCR2,             0x40140054,__READ_WRITE ,__pwm_ccr2_bits);
__IO_REG32_BIT(PWMB_CRLR0,            0x40140058,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMB_CFLR0,            0x4014005C,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMB_CRLR1,            0x40140060,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMB_CFLR1,            0x40140064,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMB_CRLR2,            0x40140068,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMB_CFLR2,            0x4014006C,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMB_CRLR3,            0x40140070,__READ       ,__pwm_crlrx_bits);
__IO_REG32_BIT(PWMB_CFLR3,            0x40140074,__READ       ,__pwm_cflrx_bits);
__IO_REG32_BIT(PWMB_CAPENR,           0x40140078,__READ_WRITE ,__pwm_capenr_bits);
__IO_REG32_BIT(PWMB_POE,              0x4014007C,__READ_WRITE ,__pwm_poe_bits);

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
 ** TMR01
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR01_TCSR0,           0x40010000,__READ_WRITE ,__tmr_tcsrx_bits);
__IO_REG32_BIT(TMR01_TCMPR0,          0x40010004,__READ_WRITE ,__tmr_tcmprx_bits);
__IO_REG32_BIT(TMR01_TISR0,           0x40010008,__READ_WRITE ,__tmr_tisrx_bits);
__IO_REG32_BIT(TMR01_TDR0,            0x4001000C,__READ       ,__tmr_tdrx_bits);
__IO_REG32_BIT(TMR01_TCSR1,           0x40010020,__READ_WRITE ,__tmr_tcsrx_bits);
__IO_REG32_BIT(TMR01_TCMPR1,          0x40010024,__READ_WRITE ,__tmr_tcmprx_bits);
__IO_REG32_BIT(TMR01_TISR1,           0x40010028,__READ_WRITE ,__tmr_tisrx_bits);
__IO_REG32_BIT(TMR01_TDR1,            0x4001002C,__READ       ,__tmr_tdrx_bits);

/***************************************************************************
 **
 ** TMR23
 **
 ***************************************************************************/
__IO_REG32_BIT(TMR23_TCSR2,           0x40110000,__READ_WRITE ,__tmr_tcsrx_bits);
__IO_REG32_BIT(TMR23_TCMPR2,          0x40110004,__READ_WRITE ,__tmr_tcmprx_bits);
__IO_REG32_BIT(TMR23_TISR2,           0x40110008,__READ_WRITE ,__tmr_tisrx_bits);
__IO_REG32_BIT(TMR23_TDR2,            0x4011000C,__READ       ,__tmr_tdrx_bits);
__IO_REG32_BIT(TMR23_TCSR3,           0x40110020,__READ_WRITE ,__tmr_tcsrx_bits);
__IO_REG32_BIT(TMR23_TCMPR3,          0x40110024,__READ_WRITE ,__tmr_tcmprx_bits);
__IO_REG32_BIT(TMR23_TISR3,           0x40110028,__READ_WRITE ,__tmr_tisrx_bits);
__IO_REG32_BIT(TMR23_TDR3,            0x4011002C,__READ       ,__tmr_tdrx_bits);

/***************************************************************************
 **
 ** WDT
 **
 ***************************************************************************/
__IO_REG32_BIT(WDT_WTCR,              0x40004000,__READ_WRITE ,__wdt_wtcr_bits);

/***************************************************************************
 **
 ** SPI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI0_SPI_CNTRL,        0x40030000,__READ_WRITE ,__spi_spi_cntrl_bits);
__IO_REG32_BIT(SPI0_SPI_DIVIDER,      0x40030004,__READ_WRITE ,__spi_spi_divider_bits);
__IO_REG32_BIT(SPI0_SPI_SSR,          0x40030008,__READ_WRITE ,__spi_spi_ssr_bits);
__IO_REG32(    SPI0_SPI_RX0,          0x40030010,__READ       );
__IO_REG32(    SPI0_SPI_RX1,          0x40030014,__READ       );
__IO_REG32(    SPI0_SPI_TX0,          0x40030020,__WRITE      );
__IO_REG32(    SPI0_SPI_TX1,          0x40030024,__WRITE      );
__IO_REG32(    SPI0_SPI_VARCLK,       0x40030034,__READ_WRITE );
__IO_REG32_BIT(SPI1_SPI_CNTRL,        0x40034000,__READ_WRITE ,__spi_spi_cntrl_bits);
__IO_REG32_BIT(SPI1_SPI_DIVIDER,      0x40034004,__READ_WRITE ,__spi_spi_divider_bits);
__IO_REG32_BIT(SPI1_SPI_SSR,          0x40034008,__READ_WRITE ,__spi_spi_ssr_bits);
__IO_REG32(    SPI1_SPI_RX0,          0x40034010,__READ       );
__IO_REG32(    SPI1_SPI_RX1,          0x40034014,__READ       );
__IO_REG32(    SPI1_SPI_TX0,          0x40034020,__WRITE      );
__IO_REG32(    SPI1_SPI_TX1,          0x40034024,__WRITE      );
__IO_REG32(    SPI1_SPI_VARCLK,       0x40034034,__READ_WRITE );

/***************************************************************************
 **
 ** UART
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0_UA_RBR,          0x40050000,__READ_WRITE ,__uartx_ua_rbr_thr_bits);
#define UART0_UA_THR     UART0_UA_RBR
#define UART0_UA_THR_bit UART0_UA_RBR_bit
__IO_REG32_BIT(UART0_UA_IER,          0x40050004,__READ_WRITE ,__uartx_ua_ier_bits);
__IO_REG32_BIT(UART0_UA_FCR,          0x40050008,__READ_WRITE ,__uartx_ua_fcr_bits);
__IO_REG32_BIT(UART0_UA_LCR,          0x4005000C,__READ_WRITE ,__uartx_ua_lcr_bits);
__IO_REG32_BIT(UART0_UA_MCR,          0x40050010,__READ_WRITE ,__uartx_ua_mcr_bits);
__IO_REG32_BIT(UART0_UA_MSR,          0x40050014,__READ_WRITE ,__uartx_ua_msr_bits);
__IO_REG32_BIT(UART0_UA_FSR,          0x40050018,__READ_WRITE ,__uartx_ua_fsr_bits);
__IO_REG32_BIT(UART0_UA_ISR,          0x4005001C,__READ_WRITE ,__uartx_ua_isr_bits);
__IO_REG32_BIT(UART0_UA_TOR,          0x40050020,__READ_WRITE ,__uartx_ua_tor_bits);
__IO_REG32_BIT(UART0_UA_BAUD,         0x40050024,__READ_WRITE ,__uartx_ua_baud_bits);
__IO_REG32_BIT(UART0_UA_IRCR,         0x40050028,__READ_WRITE ,__uartx_ua_ircr_bits);
__IO_REG32_BIT(UART0_UA_ACT_CSR,      0x4005002C,__READ_WRITE ,__uartx_ua_alt_csr_bits);
__IO_REG32_BIT(UART0_UA_FUN_SEL,      0x40050030,__READ_WRITE ,__uartx_ua_fun_sel_bits);

__IO_REG32_BIT(UART1_UA_RBR,          0x40150000,__READ_WRITE ,__uartx_ua_rbr_thr_bits);
#define UART1_UA_THR     UART1_UA_RBR
#define UART1_UA_THR_bit UART1_UA_RBR_bit
__IO_REG32_BIT(UART1_UA_IER,          0x40150004,__READ_WRITE ,__uartx_ua_ier_bits);
__IO_REG32_BIT(UART1_UA_FCR,          0x40150008,__READ_WRITE ,__uartx_ua_fcr_bits);
__IO_REG32_BIT(UART1_UA_LCR,          0x4015000C,__READ_WRITE ,__uartx_ua_lcr_bits);
__IO_REG32_BIT(UART1_UA_MCR,          0x40150010,__READ_WRITE ,__uartx_ua_mcr_bits);
__IO_REG32_BIT(UART1_UA_MSR,          0x40150014,__READ_WRITE ,__uartx_ua_msr_bits);
__IO_REG32_BIT(UART1_UA_FSR,          0x40150018,__READ_WRITE ,__uartx_ua_fsr_bits);
__IO_REG32_BIT(UART1_UA_ISR,          0x4015001C,__READ_WRITE ,__uartx_ua_isr_bits);
__IO_REG32_BIT(UART1_UA_TOR,          0x40150020,__READ_WRITE ,__uartx_ua_tor_bits);
__IO_REG32_BIT(UART1_UA_BAUD,         0x40150024,__READ_WRITE ,__uartx_ua_baud_bits);
__IO_REG32_BIT(UART1_UA_IRCR,         0x40150028,__READ_WRITE ,__uartx_ua_ircr_bits);
__IO_REG32_BIT(UART1_UA_ACT_CSR,      0x4015002C,__READ_WRITE ,__uartx_ua_alt_csr_bits);
__IO_REG32_BIT(UART1_UA_FUN_SEL,      0x40150030,__READ_WRITE ,__uartx_ua_fun_sel_bits);

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
#define NVIC_PWMB_INT         23  /* PWM4,PWM5,PWM6 and PWM7 interrupt                      */
#define NVIC_TMR0_INT         24  /* Timer 0 interrupt                                      */
#define NVIC_TMR1_INT         25  /* Timer 1 interrupt                                      */
#define NVIC_TMR2_INT         26  /* Timer 2 interrupt                                      */
#define NVIC_TMR3_INT         27  /* Timer 3 interrupt                                      */
#define NVIC_UART0_INT        28  /* UART0 interrupt                                        */
#define NVIC_UART1_INT        29  /* UART1 interrupt                                        */
#define NVIC_SPI0_INT         30  /* SPI0 interrupt                                         */
#define NVIC_SPI1_INT         31  /* SPI1 interrupt                                         */
#define NVIC_I2C_INT          34  /* I2C interrupt                                          */
#define NVIC_PWRWU_INT        44  /* Clock controller interrupt for chip wake up from power-down state */
#define NVIC_ADC_INT          45  /* ADC interrupt                                          */

#endif    /* __IOM05X_H */

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
Interrupt9   = GP01_INT       0x50
Interrupt10  = GP234_INT      0x54
Interrupt11  = PWMA_INT       0x58
Interrupt12  = PWMB_INT       0x5C
Interrupt13  = TMR0_INT       0x60
Interrupt14  = TMR1_INT       0x64
Interrupt15  = TMR2_INT       0x68
Interrupt16  = TMR3_INT       0x6C
Interrupt17  = UART0_INT      0x70
Interrupt18  = UART1_INT      0x74
Interrupt19  = SPI0_INT       0x78
Interrupt20  = SPI1_INT       0x7C
Interrupt21  = I2C_INT        0x88
Interrupt22  = PWRWU_INT      0xB0
Interrupt23  = ADC_INT        0xB4
###DDF-INTERRUPT-END###*/

