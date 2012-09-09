/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST STM32F051x8
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2012
 **
 **    $Revision: 52044 $
 **
 ***************************************************************************/

#ifndef __IOSTM32F051x8_H
#define __IOSTM32F051x8_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   STM32F051x8 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/


/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* Power control register (PWR_CR) */
typedef struct {
  __REG32  LPDS           : 1;
  __REG32  PDDS           : 1;
  __REG32  CWUF           : 1;
  __REG32  CSBF           : 1;
  __REG32  PVDE           : 1;
  __REG32  PLS            : 3;
  __REG32  DBP            : 1;
  __REG32                 :23;
} __pwr_cr_bits;

/* Power control/status register (PWR_CSR) */
typedef struct {
  __REG32  WUF            : 1;
  __REG32  SBF            : 1;
  __REG32  PVDO           : 1;
  __REG32                 : 5;
  __REG32  EWUP1          : 1;
  __REG32  EWUP2          : 1;
  __REG32                 :22;
} __pwr_csr_bits;

/* Clock control register (RCC_CR) */
typedef struct {
  __REG32  HSION          : 1;
  __REG32  HSIRDY         : 1;
  __REG32                 : 1;
  __REG32  HSITRIM        : 5;
  __REG32  HSICAL         : 8;
  __REG32  HSEON          : 1;
  __REG32  HSERDY         : 1;
  __REG32  HSEBYP         : 1;
  __REG32  CSSON          : 1;
  __REG32                 : 4;
  __REG32  PLLON          : 1;
  __REG32  PLLRDY         : 1;
  __REG32                 : 6;
} __rcc_cr_bits;

/* Clock configuration register (RCC_CFGR) */
typedef struct {
  __REG32  SW             : 2;
  __REG32  SWS            : 2;
  __REG32  HPRE           : 4;
  __REG32  PPRE           : 3;
  __REG32                 : 3;
  __REG32  ADCPRE         : 1;
  __REG32                 : 1;
  __REG32  PLLSRC         : 1;
  __REG32  PLLXTPRE       : 1;
  __REG32  PLLMUL         : 4;
  __REG32                 : 2;
  __REG32  MCO            : 3;
  __REG32                 : 5;
} __rcc_cfgr_bits;

/* Clock interrupt register (RCC_CIR) */
typedef struct {
  __REG32  LSIRDYF        : 1;
  __REG32  LSERDYF        : 1;
  __REG32  HSIRDYF        : 1;
  __REG32  HSERDYF        : 1;
  __REG32  PLLRDYF        : 1;
  __REG32  HSI14RDYF      : 1;
  __REG32                 : 1;
  __REG32  CSSF           : 1;
  __REG32  LSIRDYIE       : 1;
  __REG32  LSERDYIE       : 1;
  __REG32  HSIRDYIE       : 1;
  __REG32  HSERDYIE       : 1;
  __REG32  PLLRDYIE       : 1;
  __REG32  HSI14RDYIE     : 1;
  __REG32                 : 2;
  __REG32  LSIRDYC        : 1;
  __REG32  LSERDYC        : 1;
  __REG32  HSIRDYC        : 1;
  __REG32  HSERDYC        : 1;
  __REG32  PLLRDYC        : 1;
  __REG32  HSI14RDYC      : 1;
  __REG32                 : 1;
  __REG32  CSSC           : 1;
  __REG32                 : 8;
} __rcc_cir_bits;

/* APB2 Peripheral reset register (RCC_APB2RSTR) */
typedef struct {
  __REG32  SYSCFGRST      : 1;
  __REG32                 : 8;
  __REG32  ADCRST         : 1;
  __REG32                 : 1;
  __REG32  TIM1RST        : 1;
  __REG32  SPI1RST        : 1;
  __REG32                 : 1;
  __REG32  USART1RST      : 1;
  __REG32                 : 1;
  __REG32  TIM15RST       : 1;
  __REG32  TIM16RST       : 1;
  __REG32  TIM17RST       : 1;
  __REG32                 : 3;
  __REG32  DBGMCURST      : 1;
  __REG32                 : 9;
} __rcc_apb2rstr_bits;

/* APB1 Peripheral reset register (RCC_APB1RSTR) */
typedef struct {
  __REG32  TIM2RST        : 1;
  __REG32  TIM3RST        : 1;
  __REG32                 : 2;
  __REG32  TIM6RST        : 1;
  __REG32                 : 3;
  __REG32  TIM14RST       : 1;
  __REG32                 : 2;
  __REG32  WWDGRST        : 1;
  __REG32                 : 2;
  __REG32  SPI2RST        : 1;
  __REG32                 : 2;
  __REG32  USART2RST      : 1;
  __REG32                 : 3;
  __REG32  I2C1RST        : 1;
  __REG32  I2C2RST        : 1;
  __REG32                 : 5;
  __REG32  PWRRST         : 1;
  __REG32  DACRST         : 1;
  __REG32  CECCRST        : 1;
  __REG32                 : 1;
} __rcc_apb1rstr_bits;

/* AHB Peripheral Clock enable register (RCC_AHBENR) */
typedef struct {
  __REG32  DMA1E          : 1;
  __REG32                 : 1;
  __REG32  SRAMEN         : 1;
  __REG32                 : 1;
  __REG32  FLITFEN        : 1;
  __REG32                 : 1;
  __REG32  CRCEN          : 1;
  __REG32                 :10;
  __REG32  IOPAEN         : 1;
  __REG32  IOPBEN         : 1;
  __REG32  IOPCEN         : 1;
  __REG32  IOPDEN         : 1;
  __REG32                 : 1;
  __REG32  IOPFEN         : 1;
  __REG32                 : 1;
  __REG32  TSCEN          : 1;
  __REG32                 : 7;
} __rcc_ahbenr_bits;

/* APB2 Peripheral Clock enable register (RCC_APB2ENR) */
typedef struct {
  __REG32  SYSCFGEN      : 1;
  __REG32                : 8;
  __REG32  ADC1EN        : 1;
  __REG32                : 1;
  __REG32  TIM1EN        : 1;
  __REG32  SPI1EN        : 1;
  __REG32                : 1;
  __REG32  USART1EN      : 1;
  __REG32                : 1;
  __REG32  TIM15EN       : 1;
  __REG32  TIM16EN       : 1;
  __REG32  TIM17EN       : 1;
  __REG32                : 3;
  __REG32  DBGMCUEN      : 1;
  __REG32                : 9;
} __rcc_apb2enr_bits;

/* APB1 Peripheral Clock enable register (RCC_APB1ENR) */
typedef struct {
  __REG32  TIM2EN        : 1;
  __REG32  TIM3EN        : 1;
  __REG32                : 2;
  __REG32  TIM6EN        : 1;
  __REG32                : 3;
  __REG32  TIM14EN       : 1;
  __REG32                : 2;
  __REG32  WWDGEN        : 1;
  __REG32                : 2;
  __REG32  SPI2EN        : 1;
  __REG32                : 2;
  __REG32  USART2EN      : 1;
  __REG32                : 3;
  __REG32  I2C1EN        : 1;
  __REG32  I2C2EN        : 1;
  __REG32                : 5;
  __REG32  PWREN         : 1;
  __REG32  DACEN         : 1;
  __REG32  CECEN         : 1;
  __REG32                : 1;
} __rcc_apb1enr_bits;

/* Backup domain control register (RCC_BDCR) */
typedef struct {
  __REG32  LSEON          : 1;
  __REG32  LSERDY         : 1;
  __REG32  LSEBYP         : 1;
  __REG32  LSEDRV         : 2;
  __REG32                 : 3;
  __REG32  RTCSEL         : 2;
  __REG32                 : 5;
  __REG32  RTCEN          : 1;
  __REG32  BDRST          : 1;
  __REG32                 :15;
} __rcc_bdcr_bits;

/* Control/status register (RCC_CSR) */
typedef struct {
  __REG32  LSION          : 1;
  __REG32  LSIRDY         : 1;
  __REG32                 :22;
  __REG32  RMVF           : 1;
  __REG32  OBLRSTF        : 1;
  __REG32  PINRSTF        : 1;
  __REG32  PORRSTF        : 1;
  __REG32  SFTRSTF        : 1;
  __REG32  IWDGRSTF       : 1;
  __REG32  WWDGRSTF       : 1;
  __REG32  LPWRRSTF       : 1;
} __rcc_csr_bits;

/* AHB peripheral reset register (RCC_AHBRSTR) */
typedef struct {
  __REG32                 :17;
  __REG32  IOPARST        : 1;
  __REG32  IOPBRST        : 1;
  __REG32  IOPCRST        : 1;
  __REG32  IOPDRST        : 1;
  __REG32                 : 1;
  __REG32  IOPFRST        : 1;
  __REG32                 : 1;
  __REG32  TSCRST         : 1;
  __REG32                 : 7;
} __rcc_ahbrstr_bits;

/* Clock configuration register2 (RCC_CFGR2) */
typedef struct {
  __REG32  PREDIV         : 4;
  __REG32                 :28;
} __rcc_cfgr2_bits;

/* Clock configuration register 3 (RCC_CFGR3) */
typedef struct {
  __REG32  USART1SW       : 2;
  __REG32                 : 2;
  __REG32  I2C1SW         : 1;
  __REG32                 : 1;
  __REG32  CECSW          : 1;
  __REG32                 : 1;
  __REG32  ADCSW          : 1;
  __REG32                 :23;
} __rcc_cfgr3_bits;

/* Clock control register 2 (RCC_CR2) */
typedef struct {
  __REG32  HSI14ON        : 1;
  __REG32  HSI14RDY       : 1;
  __REG32  HSI14DIS       : 1;
  __REG32  HSI14TRIM      : 5;
  __REG32  HSI14CAL       : 8;
  __REG32                 :16;
} __rcc_cr2_bits;

/* GPIO port mode register (GPIOx_MODER) (x = A..D,F) */
typedef struct {
  __REG32  MODER0         : 2;
  __REG32  MODER1         : 2;
  __REG32  MODER2         : 2;
  __REG32  MODER3         : 2;
  __REG32  MODER4         : 2;
  __REG32  MODER5         : 2;
  __REG32  MODER6         : 2;
  __REG32  MODER7         : 2;
  __REG32  MODER8         : 2;
  __REG32  MODER9         : 2;
  __REG32  MODER10        : 2;
  __REG32  MODER11        : 2;
  __REG32  MODER12        : 2;
  __REG32  MODER13        : 2;
  __REG32  MODER14        : 2;
  __REG32  MODER15        : 2;
} __gpio_moder_bits;

/* GPIO port output type register (GPIOx_OTYPER) (x = A..D,F) */
typedef struct {
  __REG32  OT0            : 1;
  __REG32  OT1            : 1;
  __REG32  OT2            : 1;
  __REG32  OT3            : 1;
  __REG32  OT4            : 1;
  __REG32  OT5            : 1;
  __REG32  OT6            : 1;
  __REG32  OT7            : 1;
  __REG32  OT8            : 1;
  __REG32  OT9            : 1;
  __REG32  OT10           : 1;
  __REG32  OT11           : 1;
  __REG32  OT12           : 1;
  __REG32  OT13           : 1;
  __REG32  OT14           : 1;
  __REG32  OT15           : 1;
  __REG32                 :16;
} __gpio_otyper_bits;

/* GPIO port output speed register (GPIOx_OSPEEDR) (x = A..D,F)*/
typedef struct {
  __REG32  OSPEEDR0       : 2;
  __REG32  OSPEEDR1       : 2;
  __REG32  OSPEEDR2       : 2;
  __REG32  OSPEEDR3       : 2;
  __REG32  OSPEEDR4       : 2;
  __REG32  OSPEEDR5       : 2;
  __REG32  OSPEEDR6       : 2;
  __REG32  OSPEEDR7       : 2;
  __REG32  OSPEEDR8       : 2;
  __REG32  OSPEEDR9       : 2;
  __REG32  OSPEEDR10      : 2;
  __REG32  OSPEEDR11      : 2;
  __REG32  OSPEEDR12      : 2;
  __REG32  OSPEEDR13      : 2;
  __REG32  OSPEEDR14      : 2;
  __REG32  OSPEEDR15      : 2;
} __gpio_ospeedr_bits;

/* GPIO port pull-up/pull-down register (GPIOx_PUPDR) (x = A..D,F) */
typedef struct {
  __REG32  PUPDR0         : 2;
  __REG32  PUPDR1         : 2;
  __REG32  PUPDR2         : 2;
  __REG32  PUPDR3         : 2;
  __REG32  PUPDR4         : 2;
  __REG32  PUPDR5         : 2;
  __REG32  PUPDR6         : 2;
  __REG32  PUPDR7         : 2;
  __REG32  PUPDR8         : 2;
  __REG32  PUPDR9         : 2;
  __REG32  PUPDR10        : 2;
  __REG32  PUPDR11        : 2;
  __REG32  PUPDR12        : 2;
  __REG32  PUPDR13        : 2;
  __REG32  PUPDR14        : 2;
  __REG32  PUPDR15        : 2;
} __gpio_pupdr_bits;

/* GPIO port input data register (GPIOx_IDR) (x = A..D,F) */
typedef struct {
  __REG32  IDR0           : 1;
  __REG32  IDR1           : 1;
  __REG32  IDR2           : 1;
  __REG32  IDR3           : 1;
  __REG32  IDR4           : 1;
  __REG32  IDR5           : 1;
  __REG32  IDR6           : 1;
  __REG32  IDR7           : 1;
  __REG32  IDR8           : 1;
  __REG32  IDR9           : 1;
  __REG32  IDR10          : 1;
  __REG32  IDR11          : 1;
  __REG32  IDR12          : 1;
  __REG32  IDR13          : 1;
  __REG32  IDR14          : 1;
  __REG32  IDR15          : 1;
  __REG32                 :16;
} __gpio_idr_bits;

/* GPIO port output data register (GPIOx_ODR) (x = A..D,F) */
typedef struct {
  __REG32  ODR0           : 1;
  __REG32  ODR1           : 1;
  __REG32  ODR2           : 1;
  __REG32  ODR3           : 1;
  __REG32  ODR4           : 1;
  __REG32  ODR5           : 1;
  __REG32  ODR6           : 1;
  __REG32  ODR7           : 1;
  __REG32  ODR8           : 1;
  __REG32  ODR9           : 1;
  __REG32  ODR10          : 1;
  __REG32  ODR11          : 1;
  __REG32  ODR12          : 1;
  __REG32  ODR13          : 1;
  __REG32  ODR14          : 1;
  __REG32  ODR15          : 1;
  __REG32                 :16;
} __gpio_odr_bits;

/* GPIO port bit set/reset register (GPIOx_BSRR) (x = A..D,F) */
typedef struct {
  __REG32  BS0            : 1;
  __REG32  BS1            : 1;
  __REG32  BS2            : 1;
  __REG32  BS3            : 1;
  __REG32  BS4            : 1;
  __REG32  BS5            : 1;
  __REG32  BS6            : 1;
  __REG32  BS7            : 1;
  __REG32  BS8            : 1;
  __REG32  BS9            : 1;
  __REG32  BS10           : 1;
  __REG32  BS11           : 1;
  __REG32  BS12           : 1;
  __REG32  BS13           : 1;
  __REG32  BS14           : 1;
  __REG32  BS15           : 1;
  __REG32  BR0            : 1;
  __REG32  BR1            : 1;
  __REG32  BR2            : 1;
  __REG32  BR3            : 1;
  __REG32  BR4            : 1;
  __REG32  BR5            : 1;
  __REG32  BR6            : 1;
  __REG32  BR7            : 1;
  __REG32  BR8            : 1;
  __REG32  BR9            : 1;
  __REG32  BR10           : 1;
  __REG32  BR11           : 1;
  __REG32  BR12           : 1;
  __REG32  BR13           : 1;
  __REG32  BR14           : 1;
  __REG32  BR15           : 1;
} __gpio_bsrr_bits;

/* GPIO port configuration lock register (GPIOx_LCKR) (x = A..B) */
typedef struct {
  __REG32  LCK0           : 1;
  __REG32  LCK1           : 1;
  __REG32  LCK2           : 1;
  __REG32  LCK3           : 1;
  __REG32  LCK4           : 1;
  __REG32  LCK5           : 1;
  __REG32  LCK6           : 1;
  __REG32  LCK7           : 1;
  __REG32  LCK8           : 1;
  __REG32  LCK9           : 1;
  __REG32  LCK10          : 1;
  __REG32  LCK11          : 1;
  __REG32  LCK12          : 1;
  __REG32  LCK13          : 1;
  __REG32  LCK14          : 1;
  __REG32  LCK15          : 1;
  __REG32  LCKK           : 1;
  __REG32                 :15;
} __gpio_lckr_bits;

/* GPIO alternate function low register (GPIOx_AFRL) (x = A..B) */
typedef struct {
  __REG32  AFRL0          : 4;
  __REG32  AFRL1          : 4;
  __REG32  AFRL2          : 4;
  __REG32  AFRL3          : 4;
  __REG32  AFRL4          : 4;
  __REG32  AFRL5          : 4;
  __REG32  AFRL6          : 4;
  __REG32  AFRL7          : 4;
} __gpio_afrl_bits;

/* GPIO alternate function high register (GPIOx_AFRH) (x = A..B) */
typedef struct {
  __REG32  AFRH0          : 4;
  __REG32  AFRH1          : 4;
  __REG32  AFRH2          : 4;
  __REG32  AFRH3          : 4;
  __REG32  AFRH4          : 4;
  __REG32  AFRH5          : 4;
  __REG32  AFRH6          : 4;
  __REG32  AFRH7          : 4;
} __gpio_afrh_bits;

/* Port bit reset register (GPIOx_BRR) (x=A..D,F) */
typedef struct {
  __REG32  AFRH0          : 4;
  __REG32  AFRH1          : 4;
  __REG32  AFRH2          : 4;
  __REG32  AFRH3          : 4;
  __REG32  AFRH4          : 4;
  __REG32  AFRH5          : 4;
  __REG32  AFRH6          : 4;
  __REG32  AFRH7          : 4;
} __gpio_brr_bits;

/* SYSCFG configuration register 1 (SYSCFG_CFGR1) */
typedef struct {
  __REG32  MEM_MODE           : 2;
  __REG32                     : 6;
  __REG32  ADC_DMA_RMP        : 1;
  __REG32  USART1_TX_DMA_RMP  : 1;
  __REG32  USART1_RX_DMA_RMP  : 1;
  __REG32  TIM16_DMA_RMP      : 1;
  __REG32  TIM17_DMA_RMP      : 1;
  __REG32                     : 3;
  __REG32  I2C_PB6_FM         : 1;
  __REG32  I2C_PB7_FM         : 1;
  __REG32  I2C_PB8_FM         : 1;
  __REG32  I2C_PB9_FM         : 1;
  __REG32                     :12;
} __syscfg_cfgr1_bits;

/* SYSCFG external interrupt configuration register 1 (SYSCFG_EXTICR1) */
typedef struct {
  __REG32  EXTI0              : 4;
  __REG32  EXTI1              : 4;
  __REG32  EXTI2              : 4;
  __REG32  EXTI3              : 4;
  __REG32                     :16;
} __syscfg_exticr1_bits;

/* SYSCFG external interrupt configuration register 2 (SYSCFG_EXTICR2) */
typedef struct {
  __REG32  EXTI4              : 4;
  __REG32  EXTI5              : 4;
  __REG32  EXTI6              : 4;
  __REG32  EXTI7              : 4;
  __REG32                     :16;
} __syscfg_exticr2_bits;

/* SYSCFG external interrupt configuration register 3 (SYSCFG_EXTICR3) */
typedef struct {
  __REG32  EXTI8              : 4;
  __REG32  EXTI9              : 4;
  __REG32  EXTI10             : 4;
  __REG32  EXTI11             : 4;
  __REG32                     :16;
} __syscfg_exticr3_bits;

/* SYSCFG external interrupt configuration register 4 (SYSCFG_EXTICR4) */
typedef struct {
  __REG32  EXTI12             : 4;
  __REG32  EXTI13             : 4;
  __REG32  EXTI14             : 4;
  __REG32  EXTI15             : 4;
  __REG32                     :16;
} __syscfg_exticr4_bits;

/* SYSCFG configuration register 2 (SYSCFG_CFGR2) */
typedef struct {
  __REG32  LOCUP_LOCK         : 1;
  __REG32  SRAM_PARITY_LOCK   : 1;
  __REG32  PVD_LOCK           : 1;
  __REG32                     : 5;
  __REG32  SRAM_PEF           : 1;
  __REG32                     :23;
} __syscfg_cfgr2_bits;

/* Interrupt mask register (EXTI_IMR) */
typedef struct {
  __REG32  MR0            : 1;
  __REG32  MR1            : 1;
  __REG32  MR2            : 1;
  __REG32  MR3            : 1;
  __REG32  MR4            : 1;
  __REG32  MR5            : 1;
  __REG32  MR6            : 1;
  __REG32  MR7            : 1;
  __REG32  MR8            : 1;
  __REG32  MR9            : 1;
  __REG32  MR10           : 1;
  __REG32  MR11           : 1;
  __REG32  MR12           : 1;
  __REG32  MR13           : 1;
  __REG32  MR14           : 1;
  __REG32  MR15           : 1;
  __REG32  MR16           : 1;
  __REG32  MR17           : 1;
  __REG32  MR18           : 1;
  __REG32  MR19           : 1;
  __REG32  MR20           : 1;
  __REG32  MR21           : 1;
  __REG32  MR22           : 1;
  __REG32  MR23           : 1;
  __REG32  MR24           : 1;
  __REG32  MR25           : 1;
  __REG32  MR26           : 1;
  __REG32  MR27           : 1;
  __REG32                 : 4;
} __exti_imr_bits;

/* Event mask register (EXTI_EMR) */
typedef struct {
  __REG32  MR0            : 1;
  __REG32  MR1            : 1;
  __REG32  MR2            : 1;
  __REG32  MR3            : 1;
  __REG32  MR4            : 1;
  __REG32  MR5            : 1;
  __REG32  MR6            : 1;
  __REG32  MR7            : 1;
  __REG32  MR8            : 1;
  __REG32  MR9            : 1;
  __REG32  MR10           : 1;
  __REG32  MR11           : 1;
  __REG32  MR12           : 1;
  __REG32  MR13           : 1;
  __REG32  MR14           : 1;
  __REG32  MR15           : 1;
  __REG32  MR16           : 1;
  __REG32  MR17           : 1;
  __REG32  MR18           : 1;
  __REG32  MR19           : 1;
  __REG32  MR20           : 1;
  __REG32  MR21           : 1;
  __REG32  MR22           : 1;
  __REG32  MR23           : 1;
  __REG32  MR24           : 1;
  __REG32  MR25           : 1;
  __REG32  MR26           : 1;
  __REG32  MR27           : 1;
  __REG32                 : 4;
} __exti_emr_bits;

/* Rising Trigger selection register (EXTI_RTSR) */
typedef struct {
  __REG32  TR0            : 1;
  __REG32  TR1            : 1;
  __REG32  TR2            : 1;
  __REG32  TR3            : 1;
  __REG32  TR4            : 1;
  __REG32  TR5            : 1;
  __REG32  TR6            : 1;
  __REG32  TR7            : 1;
  __REG32  TR8            : 1;
  __REG32  TR9            : 1;
  __REG32  TR10           : 1;
  __REG32  TR11           : 1;
  __REG32  TR12           : 1;
  __REG32  TR13           : 1;
  __REG32  TR14           : 1;
  __REG32  TR15           : 1;
  __REG32  TR16           : 1;
  __REG32  TR17           : 1;
  __REG32                 : 1;
  __REG32  TR19           : 1;
  __REG32                 :12;
} __exti_rtsr_bits;

/* Falling Trigger selection register (EXTI_FTSR) */
typedef struct {
  __REG32  TR0            : 1;
  __REG32  TR1            : 1;
  __REG32  TR2            : 1;
  __REG32  TR3            : 1;
  __REG32  TR4            : 1;
  __REG32  TR5            : 1;
  __REG32  TR6            : 1;
  __REG32  TR7            : 1;
  __REG32  TR8            : 1;
  __REG32  TR9            : 1;
  __REG32  TR10           : 1;
  __REG32  TR11           : 1;
  __REG32  TR12           : 1;
  __REG32  TR13           : 1;
  __REG32  TR14           : 1;
  __REG32  TR15           : 1;
  __REG32  TR16           : 1;
  __REG32  TR17           : 1;
  __REG32                 : 1;
  __REG32  TR19           : 1;
  __REG32                 :12;
} __exti_ftsr_bits;

/* Software interrupt event register (EXTI_SWIER) */
typedef struct {
  __REG32  SWIER0         : 1;
  __REG32  SWIER1         : 1;
  __REG32  SWIER2         : 1;
  __REG32  SWIER3         : 1;
  __REG32  SWIER4         : 1;
  __REG32  SWIER5         : 1;
  __REG32  SWIER6         : 1;
  __REG32  SWIER7         : 1;
  __REG32  SWIER8         : 1;
  __REG32  SWIER9         : 1;
  __REG32  SWIER10        : 1;
  __REG32  SWIER11        : 1;
  __REG32  SWIER12        : 1;
  __REG32  SWIER13        : 1;
  __REG32  SWIER14        : 1;
  __REG32  SWIER15        : 1;
  __REG32  SWIER16        : 1;
  __REG32  SWIER17        : 1;
  __REG32                 : 1;
  __REG32  SWIER19        : 1;
  __REG32                 :12;
} __exti_swier_bits;

/* Pending register (EXTI_PR) */
typedef struct {
  __REG32  PR0            : 1;
  __REG32  PR1            : 1;
  __REG32  PR2            : 1;
  __REG32  PR3            : 1;
  __REG32  PR4            : 1;
  __REG32  PR5            : 1;
  __REG32  PR6            : 1;
  __REG32  PR7            : 1;
  __REG32  PR8            : 1;
  __REG32  PR9            : 1;
  __REG32  PR10           : 1;
  __REG32  PR11           : 1;
  __REG32  PR12           : 1;
  __REG32  PR13           : 1;
  __REG32  PR14           : 1;
  __REG32  PR15           : 1;
  __REG32  PR16           : 1;
  __REG32  PR17           : 1;
  __REG32                 : 1;
  __REG32  PR19           : 1;
  __REG32                 :12;
} __exti_pr_bits;

/* DMA interrupt status register (DMA_ISR) */
typedef struct {
  __REG32  GIF1           : 1;
  __REG32  TCIF1          : 1;
  __REG32  HTIF1          : 1;
  __REG32  TEIF1          : 1;
  __REG32  GIF2           : 1;
  __REG32  TCIF2          : 1;
  __REG32  HTIF2          : 1;
  __REG32  TEIF2          : 1;
  __REG32  GIF3           : 1;
  __REG32  TCIF3          : 1;
  __REG32  HTIF3          : 1;
  __REG32  TEIF3          : 1;
  __REG32  GIF4           : 1;
  __REG32  TCIF4          : 1;
  __REG32  HTIF4          : 1;
  __REG32  TEIF4          : 1;
  __REG32  GIF5           : 1;
  __REG32  TCIF5          : 1;
  __REG32  HTIF5          : 1;
  __REG32  TEIF5          : 1;
  __REG32                 :12;
} __dma_isr_bits;

/* DMA interrupt flag clear register (DMA_IFCR) */
typedef struct {
  __REG32  CGIF1          : 1;
  __REG32  CTCIF1         : 1;
  __REG32  CHTIF1         : 1;
  __REG32  CTEIF1         : 1;
  __REG32  CGIF2          : 1;
  __REG32  CTCIF2         : 1;
  __REG32  CHTIF2         : 1;
  __REG32  CTEIF2         : 1;
  __REG32  CGIF3          : 1;
  __REG32  CTCIF3         : 1;
  __REG32  CHTIF3         : 1;
  __REG32  CTEIF3         : 1;
  __REG32  CGIF4          : 1;
  __REG32  CTCIF4         : 1;
  __REG32  CHTIF4         : 1;
  __REG32  CTEIF4         : 1;
  __REG32  CGIF5          : 1;
  __REG32  CTCIF5         : 1;
  __REG32  CHTIF5         : 1;
  __REG32  CTEIF5         : 1;
  __REG32                 :12;
} __dma_ifcr_bits;

/* DMA channel x configuration register (DMA_CCRx) (x = 1 ..5) */
typedef struct {
  __REG32  EN             : 1;
  __REG32  TCIE           : 1;
  __REG32  HTIE           : 1;
  __REG32  TEIE           : 1;
  __REG32  DIR            : 1;
  __REG32  CIRC           : 1;
  __REG32  PINC           : 1;
  __REG32  MINC           : 1;
  __REG32  PSIZE          : 2;
  __REG32  MSIZE          : 2;
  __REG32  PL             : 2;
  __REG32  MEM2MEM        : 1;
  __REG32                 :17;
} __dma_ccr_bits;

/* DMA channel x number of data register (DMA_CNDTRx) (x = 1 ..5) */
typedef struct {
  __REG32  NDT            :16;
  __REG32                 :16;
} __dma_cndtr_bits;

/* RTC time register (RTC_TR) */
typedef struct {
  __REG32 SU	            : 4;
  __REG32 ST	            : 3;
  __REG32                 : 1;
  __REG32 MNU	            : 4;
  __REG32 MNT	            : 3;
  __REG32                 : 1;
  __REG32 HU		          : 4;
  __REG32 HT	            : 2;
  __REG32 PM	            : 1;
  __REG32                 : 9;
} __rtc_tr_bits;

/* RTC date register (RTC_DR) */
typedef struct {
  __REG32 DU	            : 4;
  __REG32 DT	            : 2;
  __REG32                 : 2;
  __REG32 MU	            : 4;
  __REG32 MT	            : 1;
  __REG32 WDU	            : 3;
  __REG32 YU		          : 4;
  __REG32 YT	            : 4;
  __REG32                 : 8;
} __rtc_dr_bits;

/* RTC control register (RTC_CR) */
typedef struct {
  __REG32 		            : 3;
  __REG32 TSEDGE          : 1;
  __REG32 REFCKON         : 1;
  __REG32 BYPSHAD	        : 1;
  __REG32 FMT	            : 1;
  __REG32                 : 1;
  __REG32 ALRAE           : 1;
  __REG32                 : 2;
  __REG32 TSE	            : 1;
  __REG32 ALRAIE          : 1;
  __REG32                 : 2;
  __REG32 TSIE	          : 1;
  __REG32 ADD1H           : 1;
  __REG32 SUB1H           : 1;
  __REG32 BKP	            : 1;
  __REG32 COSEL           : 1;
  __REG32 POL             : 1;
  __REG32 OSEL	          : 2;
  __REG32 COE             : 1;
  __REG32                 : 8;
} __rtc_cr_bits;

/* RTC initialization and status register (RTC_ISR) */
typedef struct {
  __REG32 ALRAWF          : 1;
  __REG32 		            : 2;
  __REG32 SHPF	          : 1;
  __REG32 INITS		        : 1;
  __REG32 RSF	            : 1;
  __REG32 INITF           : 1;
  __REG32 INIT            : 1;
  __REG32 ALRAF           : 1;
  __REG32                 : 2;
  __REG32 TSF	            : 1;
  __REG32 TSOVF	          : 1;
  __REG32 TAMP1F          : 1;
  __REG32 TAMP2F          : 1;
  __REG32                 : 1;
  __REG32 RECALPF	        : 1;
  __REG32                 :15;
} __rtc_isr_bits;

/* RTC prescaler register (RTC_PRER) */
typedef struct {
  __REG32 PREDIV_S        :15;
  __REG32 		            : 1;
  __REG32 PREDIV_A        : 7;
  __REG32                 : 9;
} __rtc_prer_bits;

/* RTC alarm A register (RTC_ALRMAR) */
typedef struct {
  __REG32 SU	            : 4;
  __REG32 ST	            : 3;
  __REG32 MSK1            : 1;
  __REG32 MNU	            : 4;
  __REG32 MNT	            : 3;
  __REG32 MSK2            : 1;
  __REG32 HU		          : 4;
  __REG32 HT	            : 2;
  __REG32 PM	            : 1;
  __REG32 MSK3            : 1;
  __REG32 DU	            : 4;
  __REG32 DT	            : 2;
  __REG32 WDSEL           : 1;
  __REG32 MSK4	          : 1;
} __rtc_alrmar_bits;

/* RTC sub second register (RTC_SSR) */
typedef struct {
  __REG32 SS	            :16;
  __REG32   	            :16;
} __rtc_ssr_bits;

/* RTC shift control register (RTC_SHIFTR) */
typedef struct {
  __REG32 SUBFS           :15;
  __REG32   	            :16;
  __REG32 ADD1S           : 1;
} __rtc_shiftr_bits;

/* RTC write protection register (RTC_WPR) */
typedef struct {
  __REG32 KEY	            : 8;
  __REG32   	            :24;
} __rtc_wpr_bits;

/* RTC timestamp date register (RTC_TSDR) */
typedef struct {
  __REG32 DU	            : 4;
  __REG32 DT	            : 2;
  __REG32                 : 2;
  __REG32 MU	            : 4;
  __REG32 MT	            : 1;
  __REG32 WDU	            : 3;
  __REG32                 :16;
} __rtc_tsdr_bits;

/* RTC calibration register (RTC_CALR) */
typedef struct {
  __REG32 CALM            : 9;
  __REG32                 : 4;
  __REG32 CALW16	        : 1;
  __REG32 CALW8 	        : 1;
  __REG32 CALP		        : 1;
  __REG32                 :16;
} __rtc_calr_bits;

/* RTC tamper and alternate function configuration register (RTC_TAFCR) */
typedef struct {
  __REG32 TAMP1E          : 1;
  __REG32 TAMP1TRG        : 1;
  __REG32 TAMPIE          : 1;
  __REG32 TAMP2E          : 1;
  __REG32 TAMP2TRG        : 1;
  __REG32                 : 2;
  __REG32 TAMPTS	        : 1;
  __REG32 TAMPFREQ        : 3;
  __REG32 TAMPFLT         : 2;
  __REG32 TAMPPRCH        : 2;
  __REG32 TAMPPUDIS       : 1;
  __REG32                 : 2;
  __REG32 PC13VALUE       : 1;
  __REG32 PC13MODE       	: 1;
  __REG32 PC14VALUE       : 1;
  __REG32 PC14MODE       	: 1;
  __REG32 PC15VALUE       : 1;
  __REG32 PC15MODE       	: 1;
  __REG32                 : 8;
} __rtc_tafcr_bits;

/* RTC alarm A sub second register (RTC_ALRMASSR) */
typedef struct {
  __REG32 SS              :15;
  __REG32                 : 9;
  __REG32 MASKSS	        : 4;
  __REG32                 : 4;
} __rtc_alrmassr_bits;

/* Prescaler register (IWWDG_PR) */
typedef struct {
  __REG32 PR              : 3;
  __REG32                 :29;
} __iwwdg_pr_bits;

/* Reload register (IWWDG_RLR) */
typedef struct {
  __REG32 RL              :12;
  __REG32                 :20;
} __iwwdg_rlr_bits;

/* Status register (IWWDG_SR) */
typedef struct {
  __REG32 PVU             : 1;
  __REG32 RVU             : 1;
  __REG32 WVU             : 1;
  __REG32                 :29;
} __iwwdg_sr_bits;

/* Window register (IWWDG_WINR) */
typedef struct {
  __REG32 WIN             :12;
  __REG32                 :20;
} __iwwdg_winr_bits;

/* Control Register (WWDG_CR) */
typedef struct {
  __REG32 T               : 7;
  __REG32 WDGA            : 1;
  __REG32                 :24;
} __wwdg_cr_bits;

/* Configuration register (WWDG_CFR) */
typedef struct {
  __REG32 W               : 7;
  __REG32 WDGTB           : 2;
  __REG32 EWI             : 1;
  __REG32                 :22;
} __wwdg_cfr_bits;

/* Status register (WWDG_SR) */
typedef struct {
  __REG32 EWIF            : 1;
  __REG32                 :31;
} __wwdg_sr_bits;

/* Control register 1 (TIM1_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32 DIR             : 1;
  __REG32 CMS             : 2;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim1_cr1_bits;

/* Control register 2 (TIM1_CR2) */
typedef struct {
  __REG32 CCPC            : 1;
  __REG32                 : 1;
  __REG32 CCUS            : 1;
  __REG32 CCDS            : 1;
  __REG32 MMS             : 3;
  __REG32 TI1S            : 1;
  __REG32 OIS1            : 1;
  __REG32 OIS1N           : 1;
  __REG32 OIS2            : 1;
  __REG32 OIS2N           : 1;
  __REG32 OIS3            : 1;
  __REG32 OIS3N           : 1;
  __REG32 OIS4            : 1;
  __REG32                 :17;
} __tim1_cr2_bits;

/* Slave mode control register (TIM1_SMCR) */
typedef struct {
  __REG32 SMS             : 3;
  __REG32                 : 1;
  __REG32 TS              : 3;
  __REG32 MSM             : 1;
  __REG32 ETF             : 4;
  __REG32 ETPS            : 2;
  __REG32 ECE             : 1;
  __REG32 ETP             : 1;
  __REG32                 :16;
} __tim1_smcr_bits;

/* DMA/Interrupt enable register (TIM1_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32 CC2IE           : 1;
  __REG32 CC3IE           : 1;
  __REG32 CC4IE           : 1;
  __REG32 COMIE           : 1;
  __REG32 TIE             : 1;
  __REG32 BIE             : 1;
  __REG32 UDE             : 1;
  __REG32 CC1DE           : 1;
  __REG32 CC2DE           : 1;
  __REG32 CC3DE           : 1;
  __REG32 CC4DE           : 1;
  __REG32 COMDE           : 1;
  __REG32 TDE             : 1;
  __REG32                 :17;
} __tim1_dier_bits;

/* Status register (TIM1_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32 CC2IF           : 1;
  __REG32 CC3IF           : 1;
  __REG32 CC4IF           : 1;
  __REG32 COMIF           : 1;
  __REG32 TIF             : 1;
  __REG32 BIF             : 1;
  __REG32                 : 1;
  __REG32 CC1OF           : 1;
  __REG32 CC2OF           : 1;
  __REG32 CC3OF           : 1;
  __REG32 CC4OF           : 1;
  __REG32                 :19;
} __tim1_sr_bits;

/* Event generation register (TIM1_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32 CC2G            : 1;
  __REG32 CC3G            : 1;
  __REG32 CC4G            : 1;
  __REG32 COMG            : 1;
  __REG32 TG              : 1;
  __REG32 BG              : 1;
  __REG32                 :24;
} __tim1_egr_bits;

/* Capture/compare mode register 1 (TIM1_CCMR1) */
typedef union {
  /* TIM1_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32 IC2S            : 2;
  __REG32 IC2PSC          : 2;
  __REG32 IC2F            : 4;
  __REG32                 :16;
  };
  /* TIM1_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32 OC1CE           : 1;
  __REG32 OC2S            : 2;
  __REG32 OC2FE           : 1;
  __REG32 OC2PE           : 1;
  __REG32 OC2M            : 3;
  __REG32 OC2CE           : 1;
  __REG32                 :16;
  };
} __tim1_ccmr1_bits;

/* Capture/compare mode register 2 (TIM1_CCMR2) */
typedef union {
  /* TIM1_CCMR2*/
  struct {
  __REG32 IC3S            : 2;
  __REG32 IC3PSC          : 2;
  __REG32 IC3F            : 4;
  __REG32 IC4S            : 2;
  __REG32 IC4PSC          : 2;
  __REG32 IC4F            : 4;
  __REG32                 :16;
  };
  /* TIM1_OCMR2*/
  struct {
  __REG32 OC3S            : 2;
  __REG32 OC3FE           : 1;
  __REG32 OC3PE           : 1;
  __REG32 OC3M            : 3;
  __REG32 OC3CE           : 1;
  __REG32 OC4S            : 2;
  __REG32 OC4FE           : 1;
  __REG32 OC4PE           : 1;
  __REG32 OC4M            : 3;
  __REG32 OC4CE           : 1;
  __REG32                 :16;
  };
} __tim1_ccmr2_bits;

/* Capture/compare enable register (TIM1_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32 CC1NE           : 1;
  __REG32 CC1NP           : 1;
  __REG32 CC2E            : 1;
  __REG32 CC2P            : 1;
  __REG32 CC2NE           : 1;
  __REG32 CC2NP           : 1;
  __REG32 CC3E            : 1;
  __REG32 CC3P            : 1;
  __REG32 CC3NE           : 1;
  __REG32 CC3NP           : 1;
  __REG32 CC4E            : 1;
  __REG32 CC4P            : 1;
  __REG32                 :18;
} __tim1_ccer_bits;

/* Counter (TIM1_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim1_cnt_bits;

/* Prescaler (TIM1_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim1_psc_bits;

/* Auto-reload register (TIM1_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim1_arr_bits;

/* Repetition counter register (TIM1_RCR) */
typedef struct {
  __REG32 REP             : 8;
  __REG32                 :24;
} __tim1_rcr_bits;

/* Capture/compare register (TIM1_CCRx) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim1_ccr_bits;

/* Break and dead-time register (TIM1_BDTR) */
typedef struct {
  __REG32 DTG             : 8;
  __REG32 LOCK            : 2;
  __REG32 OSSI            : 1;
  __REG32 OSSR            : 1;
  __REG32 BKE             : 1;
  __REG32 BKP             : 1;
  __REG32 AOE             : 1;
  __REG32 MOE             : 1;
  __REG32                 :16;
} __tim1_bdtr_bits;

/* DMA control register (TIM1_DCR) */
typedef struct {
  __REG32 DBA             : 5;
  __REG32                 : 3;
  __REG32 DBL             : 5;
  __REG32                 :19;
} __tim1_dcr_bits;

/* DMA address for burst mode (TIM1_DMAR) */
typedef struct {
  __REG32 DMAB            :16;
  __REG32                 :16;
} __tim1_dmar_bits;

/* Control register 1 (TIMx_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32 DIR             : 1;
  __REG32 CMS             : 2;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim_cr1_bits;

/* Control register 2 (TIMx_CR2) */
typedef struct {
  __REG32                 : 3;
  __REG32 CCDS            : 1;
  __REG32 MMS             : 3;
  __REG32 TI1S            : 1;
  __REG32                 :24;
} __tim_cr2_bits;

/* Slave mode control register (TIMx_SMCR) */
typedef struct {
  __REG32 SMS             : 3;
  __REG32 		            : 1;
  __REG32 TS              : 3;
  __REG32 MSM             : 1;
  __REG32 ETF             : 4;
  __REG32 ETPS            : 2;
  __REG32 ECE             : 1;
  __REG32 ETP             : 1;
  __REG32                 :16;
} __tim_smcr_bits;

/* DMA/Interrupt enable register (TIMx_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32 CC2IE           : 1;
  __REG32 CC3IE           : 1;
  __REG32 CC4IE           : 1;
  __REG32                 : 1;
  __REG32 TIE             : 1;
  __REG32                 : 1;
  __REG32 UDE             : 1;
  __REG32 CC1DE           : 1;
  __REG32 CC2DE           : 1;
  __REG32 CC3DE           : 1;
  __REG32 CC4DE           : 1;
  __REG32                 : 1;
  __REG32 TDE             : 1;
  __REG32                 :17;
} __tim_dier_bits;

/* Status register (TIMx_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32 CC2IF           : 1;
  __REG32 CC3IF           : 1;
  __REG32 CC4IF           : 1;
  __REG32                 : 1;
  __REG32 TIF             : 1;
  __REG32                 : 2;
  __REG32 CC1OF           : 1;
  __REG32 CC2OF           : 1;
  __REG32 CC3OF           : 1;
  __REG32 CC4OF           : 1;
  __REG32                 :19;
} __tim_sr_bits;

/* Event generation register (TIMx_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32 CC2G            : 1;
  __REG32 CC3G            : 1;
  __REG32 CC4G            : 1;
  __REG32                 : 1;
  __REG32 TG              : 1;
  __REG32                 :25;
} __tim_egr_bits;

/* Capture/compare mode register 1 (TIMx_CCMR1) */
typedef union {
  /* TIMx_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32 IC2S            : 2;
  __REG32 IC2PSC          : 2;
  __REG32 IC2F            : 4;
  __REG32                 :16;
  };
  /* TIMx_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32 OC1CE           : 1;
  __REG32 OC2S            : 2;
  __REG32 OC2FE           : 1;
  __REG32 OC2PE           : 1;
  __REG32 OC2M            : 3;
  __REG32 OC2CE           : 1;
  __REG32                 :16;
  };
} __tim_ccmr1_bits;

/* Capture/compare mode register 2 (TIMx_CCMR2) */
typedef union {
  /* TIMx_CCMR2*/
  struct {
  __REG32 IC3S            : 2;
  __REG32 IC3PSC          : 2;
  __REG32 IC3F            : 4;
  __REG32 IC4S            : 2;
  __REG32 IC4PSC          : 2;
  __REG32 IC4F            : 4;
  __REG32                 :16;
  };
  /* TIMx_OCMR2*/
  struct {
  __REG32 OC3S            : 2;
  __REG32 OC3FE           : 1;
  __REG32 OC3PE           : 1;
  __REG32 OC3M            : 3;
  __REG32 OC3CE           : 1;
  __REG32 OC4S            : 2;
  __REG32 OC4FE           : 1;
  __REG32 OC4PE           : 1;
  __REG32 OC4M            : 3;
  __REG32 OC4CE           : 1;
  __REG32                 :16;
  };
} __tim_ccmr2_bits;

/* Capture/compare enable register (TIMx_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32                 : 1;
  __REG32 CC1NP           : 1;
  __REG32 CC2E            : 1;
  __REG32 CC2P            : 1;
  __REG32                 : 1;
  __REG32 CC2NP           : 1;
  __REG32 CC3E            : 1;
  __REG32 CC3P            : 1;
  __REG32                 : 1;
  __REG32 CC3NP           : 1;
  __REG32 CC4E            : 1;
  __REG32 CC4P            : 1;
  __REG32                 : 1;
  __REG32 CC4NP           : 1;
  __REG32                 :16;
} __tim_ccer_bits;

/* Counter (TIMx_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim_cnt_bits;

/* Prescaler (TIMx_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim_psc_bits;

/* Auto-reload register (TIMx_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim_arr_bits;

/* Capture/compare register (TIMx_CCR) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim_ccr_bits;

/* DMA control register (TIMx_DCR) */
typedef struct {
  __REG32 DBA             : 5;
  __REG32                 : 3;
  __REG32 DBL             : 5;
  __REG32                 :19;
} __tim_dcr_bits;

/* DMA address for burst mode (TIMx_DMAR) */
typedef struct {
  __REG32 DMAB            :16;
  __REG32                 :16;
} __tim_dmar_bits;

/* Control register 1 (TIM14_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32                 : 4;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim14_cr1_bits;

/* Slave mode control register (TIM14_SMCR) */
typedef struct {
  __REG32 SMS             : 3;
  __REG32                 : 1;
  __REG32 TS              : 3;
  __REG32 MSM             : 1;
  __REG32                 :24;
} __tim14_smcr_bits;

/* DMA/Interrupt enable register (TIM14_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32                 :30;
} __tim14_dier_bits;

/* Status register (TIM14_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32                 : 7;
  __REG32 CC1OF           : 1;
  __REG32                 :22;
} __tim14_sr_bits;

/* Event generation register (TIM14_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32                 :30;
} __tim14_egr_bits;

/* Capture/compare mode register 1 (TIM14_CCMR1) */
typedef union {
  /* TIM14_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32                 :24;
  };
  /* TIM14_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32                 :25;
  };
} __tim14_ccmr1_bits;

/* Capture/compare enable register (TIM14_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32 		            : 1;
  __REG32 CC1NP           : 1;
  __REG32                 :28;
} __tim14_ccer_bits;

/* Counter (TIM14_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim14_cnt_bits;

/* Prescaler (TIM14_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim14_psc_bits;

/* Auto-reload register (TIM14_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim14_arr_bits;

/* Capture/compare register (TIM14_CCRx) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim14_ccr_bits;

/* TIM14 option register (TIM14_OR) */
typedef struct {
  __REG32 TI1_RMP         : 1;
  __REG32                 :31;
} __tim14_or_bits;

/* Control register 1 (TIM15_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32                 : 3;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim15_cr1_bits;

/* Control register 2 (TIM15_CR2) */
typedef struct {
  __REG32 CCPC            : 1;
  __REG32                 : 1;
  __REG32 CCUS            : 1;
  __REG32 CCDS            : 1;
  __REG32 MMS             : 3;
  __REG32                 : 1;
  __REG32 OIS1            : 1;
  __REG32 OIS1N           : 1;
  __REG32 OIS2            : 1;
  __REG32                 :21;
} __tim15_cr2_bits;

/* Slave mode control register (TIM15_SMCR) */
typedef struct {
  __REG32 SMS             : 3;
  __REG32                 : 1;
  __REG32 TS              : 3;
  __REG32 MSM             : 1;
  __REG32                 :24;
} __tim15_smcr_bits;

/* DMA/Interrupt enable register (TIM15_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32 CC2IE           : 1;
  __REG32                 : 2;
  __REG32 COMIE           : 1;
  __REG32 TIE             : 1;
  __REG32 BIE             : 1;
  __REG32 UDE             : 1;
  __REG32 CC1DE           : 1;
  __REG32 CC2DE           : 1;
  __REG32                 : 3;
  __REG32 TDE             : 1;
  __REG32                 :17;
} __tim15_dier_bits;

/* Status register (TIM15_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32 CC2IF           : 1;
  __REG32                 : 2;
  __REG32 COMIF           : 1;
  __REG32 TIF             : 1;
  __REG32 BIF             : 1;
  __REG32                 : 1;
  __REG32 CC1OF           : 1;
  __REG32 CC2OF           : 1;
  __REG32                 :21;
} __tim15_sr_bits;

/* Event generation register (TIM15_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32 CC2G            : 1;
  __REG32                 : 2;
  __REG32 COMG            : 1;
  __REG32 TG              : 1;
  __REG32 BG              : 1;
  __REG32                 :24;
} __tim15_egr_bits;

/* Capture/compare mode register 1 (TIM15_CCMR1) */
typedef union {
  /* TIM15_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32 IC2S            : 2;
  __REG32 IC2PSC          : 2;
  __REG32 IC2F            : 4;
  __REG32                 :16;
  };
  /* TIM15_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32 			          : 1;
  __REG32 OC2S            : 2;
  __REG32 OC2FE           : 1;
  __REG32 OC2PE           : 1;
  __REG32 OC2M            : 3;
  __REG32                 :17;
  };
} __tim15_ccmr1_bits;

/* Capture/compare enable register (TIM15_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32 CC1NE           : 1;
  __REG32 CC1NP           : 1;
  __REG32 CC2E            : 1;
  __REG32 CC2P            : 1;
  __REG32                 : 1;
  __REG32 CC2NP           : 1;
  __REG32                 :24;
} __tim15_ccer_bits;

/* Counter (TIM15_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim15_cnt_bits;

/* Prescaler (TIM15_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim15_psc_bits;

/* Auto-reload register (TIM15_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim15_arr_bits;

/* Repetition counter register (TIM15_RCR) */
typedef struct {
  __REG32 REP             : 8;
  __REG32                 :24;
} __tim15_rcr_bits;

/* Capture/compare register (TIM15_CCRx) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim15_ccr_bits;

/* Break and dead-time register (TIM15_BDTR) */
typedef struct {
  __REG32 DTG             : 8;
  __REG32 LOCK            : 2;
  __REG32 OSSI            : 1;
  __REG32 OSSR            : 1;
  __REG32 BKE             : 1;
  __REG32 BKP             : 1;
  __REG32 AOE             : 1;
  __REG32 MOE             : 1;
  __REG32                 :16;
} __tim15_bdtr_bits;

/* DMA control register (TIM15_DCR) */
typedef struct {
  __REG32 DBA             : 5;
  __REG32                 : 3;
  __REG32 DBL             : 5;
  __REG32                 :19;
} __tim15_dcr_bits;

/* DMA address for burst mode (TIM15_DMAR) */
typedef struct {
  __REG32 DMAB            :16;
  __REG32                 :16;
} __tim15_dmar_bits;

/* Control register 1 (TIM16_CR1) */
/* Control register 1 (TIM17_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32                 : 3;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim16_cr1_bits;

/* Control register 2 (TIM16_CR2) */
/* Control register 2 (TIM17_CR2) */
typedef struct {
  __REG32 CCPC            : 1;
  __REG32                 : 1;
  __REG32 CCUS            : 1;
  __REG32 CCDS            : 1;
  __REG32                 : 4;
  __REG32 OIS1            : 1;
  __REG32 OIS1N           : 1;
  __REG32                 :22;
} __tim16_cr2_bits;

/* DMA/Interrupt enable register (TIM16_DIER) */
/* DMA/Interrupt enable register (TIM17_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32                 : 3;
  __REG32 COMIE           : 1;
  __REG32 TIE             : 1;
  __REG32 BIE             : 1;
  __REG32 UDE             : 1;
  __REG32 CC1DE           : 1;
  __REG32                 : 4;
  __REG32 TDE             : 1;
  __REG32                 :17;
} __tim16_dier_bits;

/* Status register (TIM16_SR) */
/* Status register (TIM17_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32                 : 3;
  __REG32 COMIF           : 1;
  __REG32 TIF             : 1;
  __REG32 BIF             : 1;
  __REG32                 : 1;
  __REG32 CC1OF           : 1;
  __REG32                 :22;
} __tim16_sr_bits;

/* Event generation register (TIM16_EGR) */
/* Event generation register (TIM17_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32                 : 3;
  __REG32 COMG            : 1;
  __REG32 TG              : 1;
  __REG32 BG              : 1;
  __REG32                 :24;
} __tim16_egr_bits;

/* Capture/compare mode register 1 (TIM16_CCMR1) */
/* Capture/compare mode register 1 (TIM17_CCMR1) */
typedef union {
  /* TIM16_CCMR1*/
  /* TIM17_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32                 :24;
  };
  /* TIM16_OCMR1*/
  /* TIM17_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32                 :25;
  };
} __tim16_ccmr1_bits;

/* Capture/compare enable register (TIM16_CCER) */
/* Capture/compare enable register (TIM17_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32 CC1NE           : 1;
  __REG32 CC1NP           : 1;
  __REG32                 :28;
} __tim16_ccer_bits;

/* Counter (TIM16_CNT) */
/* Counter (TIM17_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim16_cnt_bits;

/* Prescaler (TIM16_PSC) */
/* Prescaler (TIM17_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim16_psc_bits;

/* Auto-reload register (TIM16_ARR) */
/* Auto-reload register (TIM17_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim16_arr_bits;

/* Repetition counter register (TIM16_RCR) */
/* Repetition counter register (TIM17_RCR) */
typedef struct {
  __REG32 REP             : 8;
  __REG32                 :24;
} __tim16_rcr_bits;

/* Capture/compare register (TIM16_CCRx) */
/* Capture/compare register (TIM17_CCRx) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim16_ccr_bits;

/* Break and dead-time register (TIM16_BDTR) */
/* Break and dead-time register (TIM17_BDTR) */
typedef struct {
  __REG32 DTG             : 8;
  __REG32 LOCK            : 2;
  __REG32 OSSI            : 1;
  __REG32 OSSR            : 1;
  __REG32 BKE             : 1;
  __REG32 BKP             : 1;
  __REG32 AOE             : 1;
  __REG32 MOE             : 1;
  __REG32                 :16;
} __tim16_bdtr_bits;

/* DMA control register (TIM16_DCR) */
/* DMA control register (TIM17_DCR) */
typedef struct {
  __REG32 DBA             : 5;
  __REG32                 : 3;
  __REG32 DBL             : 5;
  __REG32                 :19;
} __tim16_dcr_bits;

/* DMA address for burst mode (TIM16_DMAR) */
/* DMA address for burst mode (TIM17_DMAR) */
typedef struct {
  __REG32 DMAB            :16;
  __REG32                 :16;
} __tim16_dmar_bits;

/* Control register 1 (TIM6_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32                 : 3;
  __REG32 ARPE            : 1;
  __REG32                 :24;
} __tim6_cr1_bits;

/* Control register 2 (TIM6_CR2) */
typedef struct {
  __REG32                 : 4;
  __REG32 MMS             : 3;
  __REG32                 :25;
} __tim6_cr2_bits;

/* DMA/Interrupt enable register (TIM6_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32                 : 7;
  __REG32 UDE             : 1;
  __REG32                 :23;
} __tim6_dier_bits;

/* Status register (TIM6_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32                 :31;
} __tim6_sr_bits;

/* Event generation register (TIM6_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32                 :31;
} __tim6_egr_bits;

/* Counter (TIM6_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim6_cnt_bits;

/* Prescaler (TIM6_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim6_psc_bits;

/* Auto-reload register (TIM6_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim6_arr_bits;

/* Control register 1 (I2Cx_CR1) */
typedef struct {
  __REG32 PE              : 1;
  __REG32 TXIE            : 1;
  __REG32 RXIE            : 1;
  __REG32 ADDRIE         	: 1;
  __REG32 NACKIE          : 1;
  __REG32 STOPIE          : 1;
  __REG32 TCIE            : 1;
  __REG32 ERRIE		        : 1;
  __REG32 DNF	            : 4;
  __REG32 ANFOFF          : 1;
  __REG32 SWRST           : 1;
  __REG32 TXDMAEN         : 1;
  __REG32 RXDMAEN         : 1;
  __REG32 SBC             : 1;
  __REG32 NOSTRETCH       : 1;
  __REG32 WUPEN           : 1;
  __REG32 GCEN            : 1;
  __REG32 SMBHEN          : 1;
  __REG32 SMBDEN          : 1;
  __REG32 ALERTEN         : 1;
  __REG32 PECEN           : 1;
  __REG32                 : 8;
} __i2c_cr1_bits;

/* Control register 2 (I2C_CR2) */
typedef struct {
  __REG32 SADD            :10;
  __REG32 RD_WRN          : 1;
  __REG32 ADD10           : 1;
  __REG32 HEAD10R         : 1;
  __REG32 START		        : 1;
  __REG32 STOP            : 1;
  __REG32 NACK            : 1;
  __REG32 NBYTES          : 8;
  __REG32 RELOAD          : 1;
  __REG32 AUTOEND         : 1;
  __REG32 PECBYTE         : 1;
  __REG32                 : 5;
} __i2c_cr2_bits;

/* Own address register 1 (I2C_OAR1) */
typedef struct {
  __REG32 OA1             :10;
  __REG32 OA1MODE         : 1;
  __REG32                 : 4;
  __REG32 OA1EN	          : 1;
  __REG32                 :16;
} __i2c_oar1_bits;

/* Own address register 2 (I2C_OAR2) */
typedef struct {
  __REG32                 : 1;
  __REG32 OA2		          : 7;
  __REG32 OA2MSK          : 3;
  __REG32                 : 4;
  __REG32 OA2EN           : 1;
  __REG32                 :16;
} __i2c_oar2_bits;

/* Timing register (I2Cx_TIMINGR) */
typedef struct {
  __REG32 SCLL            : 8;
  __REG32 SCLH	          : 8;
  __REG32 SDADEL          : 4;
  __REG32 SCLDEL          : 4;
  __REG32                 : 4;
  __REG32 PRESC           : 4;
} __i2c_timingr_bits;

/* Timeout register (I2Cx_TIMEOUTR) */
typedef struct {
  __REG32 TIMEOUTA        :12;
  __REG32 TIDLE	          : 1;
  __REG32                 : 2;
  __REG32 TIMOUTEN        : 1;
  __REG32 TIMEOUTB        :12;
  __REG32                 : 3;
  __REG32 TEXTEN	        : 1;
} __i2c_timeoutr_bits;

/* Interrupt and Status register (I2Cx_ISR) */
typedef struct {
  __REG32 TXE			        : 1;
  __REG32 TXIS		        : 1;
  __REG32 RXNE		        : 1;
  __REG32 ADDR		        : 1;
  __REG32 NACKF		        : 1;
  __REG32 STOPF		        : 1;
  __REG32 TC 			        : 1;
  __REG32 TCR			        : 1;
  __REG32 BERR		        : 1;
  __REG32 ARLO		        : 1;
  __REG32 OVR			        : 1;
  __REG32 PECERR	        : 1;
  __REG32 TIMEOUT	        : 1;
  __REG32 ALERT 	        : 1;
  __REG32                 : 1;
  __REG32 BUSY		        : 1;
  __REG32 DIR  		        : 1;
  __REG32 ADDCODE         : 7;
  __REG32                 : 8;
} __i2c_isr_bits;

/* Interrupt clear register (I2Cx_ICR) */
typedef struct {
  __REG32 				        : 3;
  __REG32 ADDRCF		      : 1;
  __REG32 NACKCF		      : 1;
  __REG32 STOPCF		      : 1;
  __REG32 				        : 2;
  __REG32 BERRCF		      : 1;
  __REG32 ARLOCF 			    : 1;
  __REG32 OVRCF 	        : 1;
  __REG32 PECCF		        : 1;
  __REG32 TIMOUTCF		    : 1;
  __REG32 ALERTCF			    : 1;
  __REG32                 :18;
} __i2c_icr_bits;

/* PEC register (I2Cx_PECR) */
typedef struct {
  __REG32 PEC   		      : 8;
  __REG32                 :24;
} __i2c_pecr_bits;

/* Receive data register (I2Cx_RXDR) */
typedef struct {
  __REG32 RXDATA 		      : 8;
  __REG32                 :24;
} __i2c_rxdr_bits;

/* Transmit data register (I2Cx_TXDR) */
typedef struct {
  __REG32 TXDATA          : 8;
  __REG32                 :24;
} __i2c_txdr_bits;

/* Status register 1 (I2C_SR1) */
typedef struct {
  __REG32 SB              : 1;
  __REG32 ADDR            : 1;
  __REG32 BTF             : 1;
  __REG32 ADD10           : 1;
  __REG32 STOPF           : 1;
  __REG32                 : 1;
  __REG32 RxNE            : 1;
  __REG32 TxE             : 1;
  __REG32 BERR            : 1;
  __REG32 ARLO            : 1;
  __REG32 AF              : 1;
  __REG32 OVR             : 1;
  __REG32 PECERR          : 1;
  __REG32                 : 1;
  __REG32 TIMEOUT         : 1;
  __REG32 SMBALERT        : 1;
  __REG32                 :16;
} __i2c_sr1_bits;

/* Status register 2 (I2C_SR2) */
typedef struct {
  __REG32 MSL             : 1;
  __REG32 BUSY            : 1;
  __REG32 TRA             : 1;
  __REG32                 : 1;
  __REG32 GENCALL         : 1;
  __REG32 SMBDEFAULT      : 1;
  __REG32 SMBHOST         : 1;
  __REG32 DUALF           : 1;
  __REG32 PEC             : 8;
  __REG32                 :16;
} __i2c_sr2_bits;

/* Clock control register (I2C_CCR) */
typedef struct {
  __REG32 CCR             :12;
  __REG32                 : 2;
  __REG32 DUTY            : 1;
  __REG32 F_S             : 1;
  __REG32                 :16;
} __i2c_ccr_bits;

/* TRISE Register (I2C_TRISE)*/
typedef struct {
  __REG32 TRISE           : 6;
  __REG32                 :26;
} __i2c_trise_bits;

/* SPI control register 1 (SPI_CR1)*/
typedef struct {
  __REG32 CPHA            : 1;
  __REG32 CPOL            : 1;
  __REG32 MSTR            : 1;
  __REG32 BR              : 3;
  __REG32 SPE             : 1;
  __REG32 LSBFIRST        : 1;
  __REG32 SSI             : 1;
  __REG32 SSM             : 1;
  __REG32 RXONLY          : 1;
  __REG32 CRCL            : 1;
  __REG32 CRCNEXT         : 1;
  __REG32 CRCEN           : 1;
  __REG32 BIDIOE          : 1;
  __REG32 BIDIMODE        : 1;
  __REG32                 :16;
} __spi_cr1_bits;

/* SPI control register 2 (SPI_CR2)*/
typedef struct {
  __REG32 RXDMAEN         : 1;
  __REG32 TXDMAEN         : 1;
  __REG32 SSOE            : 1;
  __REG32 NSSP            : 1;
  __REG32 FRF             : 1;
  __REG32 ERRIE           : 1;
  __REG32 RXNEIE          : 1;
  __REG32 TXEIE           : 1;
  __REG32 DS              : 4;
  __REG32 FRXTH           : 1;
  __REG32 LDMA_RX         : 1;
  __REG32 LDMA_TX         : 1;
  __REG32                 :17;
} __spi_cr2_bits;

/* SPI status register (SPI_SR)*/
typedef struct {
  __REG32 RXNE            : 1;
  __REG32 TXE             : 1;
  __REG32 CHSIDE          : 1;
  __REG32 UDR		          : 1;
  __REG32 CRCERR          : 1;
  __REG32 MODF            : 1;
  __REG32 OVR             : 1;
  __REG32 BSY             : 1;
  __REG32 FRE             : 1;
  __REG32 FRLVL           : 2;
  __REG32 FTLVL           : 2;
  __REG32                 :19;
} __spi_sr_bits;

/* SPI data register (SPI_DR) */
typedef struct {
  __REG32 DR              :16;
  __REG32                 :16;
} __spi_dr_bits;

/* SPI CRC polynomial register (SPI_CRCPR) */
typedef struct {
  __REG32 CRCPOLY         :16;
  __REG32                 :16;
} __spi_crcpr_bits;

/* SPI Rx CRC register (SPI_RXCRCR) */
typedef struct {
  __REG32 RxCRC           :16;
  __REG32                 :16;
} __spi_rxcrcr_bits;

/* SPI Tx CRC register (SPI_TXCRCR) */
typedef struct {
  __REG32 TxCRC           :16;
  __REG32                 :16;
} __spi_txcrcr_bits;

/* SPI_I2S configuration register (SPI_I2SCFGR) */
typedef struct {
  __REG32 CHLEN           : 1;
  __REG32 DATLEN          : 2;
  __REG32 CKPOL           : 1;
  __REG32 I2SSTD          : 2;
  __REG32                 : 1;
  __REG32 PCMSYNC         : 1;
  __REG32 I2SCFG          : 2;
  __REG32 I2SE            : 1;
  __REG32 I2SMOD          : 1;
  __REG32                 :20;
} __spi_i2scfgr_bits;

/* SPI_I2S Prescaler register (SPI_I2SPR) */
typedef struct {
  __REG32 I2SDIV          : 8;
  __REG32 ODD             : 1;
  __REG32 MCKOE           : 1;
  __REG32                 :22;
} __spi_i2spr_bits;

/* Control register 1 (USART_CR1) */
typedef struct {
  __REG32 UE              : 1;
  __REG32 UESM            : 1;
  __REG32 RE              : 1;
  __REG32 TE              : 1;
  __REG32 IDLEIE          : 1;
  __REG32 RXNEIE          : 1;
  __REG32 TCIE            : 1;
  __REG32 TXEIE           : 1;
  __REG32 PEIE            : 1;
  __REG32 PS              : 1;
  __REG32 PCE             : 1;
  __REG32 WAKE            : 1;
  __REG32 M               : 1;
  __REG32 MME             : 1;
  __REG32 CMIE            : 1;
  __REG32 OVER8           : 1;
  __REG32 DEDT            : 5;
  __REG32 DEAT            : 5;
  __REG32 RTOIE           : 1;
  __REG32 EOBIE           : 1;
  __REG32                 : 4;
} __usart_cr1_bits;

/* Control register 2 (USART_CR2) */
typedef struct {
  __REG32                 : 4;
  __REG32 ADDM7           : 1;
  __REG32 LBDL            : 1;
  __REG32 LBDIE           : 1;
  __REG32 SSM	            : 1;
  __REG32 LBCL            : 1;
  __REG32 CPHA            : 1;
  __REG32 CPOL            : 1;
  __REG32 CLKEN           : 1;
  __REG32 STOP            : 2;
  __REG32 LINEN           : 1;
  __REG32 SWAP            : 1;
  __REG32 RXINV           : 1;
  __REG32 TXINV           : 1;
  __REG32 DATAINV         : 1;
  __REG32 MSBFIRST        : 1;
  __REG32 ABREN         	: 1;
  __REG32 ABRMOD         	: 2;
  __REG32 RTOEN         	: 1;
  __REG32 ADD           	: 8;
} __usart_cr2_bits;

/* Control register 3 (USART_CR3) */
typedef struct {
  __REG32 EIE             : 1;
  __REG32 IREN            : 1;
  __REG32 IRLP            : 1;
  __REG32 HDSEL           : 1;
  __REG32 NACK            : 1;
  __REG32 SCEN            : 1;
  __REG32 DMAR            : 1;
  __REG32 DMAT            : 1;
  __REG32 RTSE            : 1;
  __REG32 CTSE            : 1;
  __REG32 CTSIE           : 1;
  __REG32 ONEBIT          : 1;
  __REG32 OVRDIS          : 1;
  __REG32 DDRE            : 1;
  __REG32 DEM	            : 1;
  __REG32 DEP             : 1;
  __REG32 		            : 1;
  __REG32 SCARCNT         : 3;
  __REG32 WUS             : 2;
  __REG32 WUFIE		        : 1;
  __REG32 		           	: 9;
} __usart_cr3_bits;

/* Baud rate register (USART_BRR) */
typedef struct {
  __REG32 DIV_Fraction    : 4;
  __REG32 DIV_Mantissa    :12;
  __REG32                 :16;
} __usart_brr_bits;

/* Guard time and prescaler register (USART_GTPR) */
typedef struct {
  __REG32 PSC					    : 8;
  __REG32 GT					    : 8;
  __REG32                 :16;
} __usart_gtpr_bits;

/* Receiver timeout register (USART_RTOR) */
typedef struct {
  __REG32 RTO             :24;
  __REG32 BLEN            : 8;
} __usart_rtor_bits;

/* Request register (USART_RQR) */
typedef struct {
  __REG32 ABRRQ           : 1;
  __REG32 SBKRQ           : 1;
  __REG32 MMRQ            : 1;
  __REG32 RXFRQ           : 1;
  __REG32 TXFRQ           : 1;
  __REG32                 :27;
} __usart_rqr_bits;

/* Interrupt & status register (USART_ISR) */
typedef struct {
  __REG32 PE	            : 1;
  __REG32 FE	            : 1;
  __REG32 NF	            : 1;
  __REG32 ORE	            : 1;
  __REG32 IDLE            : 1;
  __REG32 RXNE            : 1;
  __REG32 TC	            : 1;
  __REG32 TXE	            : 1;
  __REG32 LBDF            : 1;
  __REG32 CTSIF           : 1;
  __REG32 CTS	            : 1;
  __REG32 RTOF            : 1;
  __REG32 EOBF            : 1;
  __REG32                 : 1;
  __REG32 ABRE            : 1;
  __REG32 ABRF            : 1;
  __REG32 BUSY            : 1;
  __REG32 CMF             : 1;
  __REG32 SBKF            : 1;
  __REG32 RWU             : 1;
  __REG32 WUF             : 1;
  __REG32 TEACK           : 1;
  __REG32 REACK           : 1;
  __REG32                 : 9;
} __usart_isr_bits;

/* Interrupt flag clear register (USART_ICR) */
typedef struct {
  __REG32 PECF            : 1;
  __REG32 FECF            : 1;
  __REG32 NCF	            : 1;
  __REG32 ORECF           : 1;
  __REG32 IDLECF          : 1;
  __REG32 		            : 1;
  __REG32 TCCF            : 1;
  __REG32 		            : 1;
  __REG32 LBDCF           : 1;
  __REG32 CTSCF           : 1;
  __REG32    	            : 1;
  __REG32 RTOCF           : 1;
  __REG32 EOBCF           : 1;
  __REG32 		            : 4;
  __REG32 CMCF            : 1;
  __REG32 	              : 2;
  __REG32 WUCF            : 1;
  __REG32                 :11;
} __usart_icr_bits;

/* Receive data register (USART_RDR) */
typedef struct {
  __REG32 RDR             : 9;
  __REG32                 :23;
} __usart_rdr_bits;

/* Transmit data register (USART_TDR) */
typedef struct {
  __REG32 RDR             : 9;
  __REG32                 :23;
} __usart_tdr_bits;

/* ADC interrupt and status register (ADC_ISR) */
typedef struct {
  __REG32 ADRDY           : 1;
  __REG32 EOSMP           : 1;
  __REG32 EOC             : 1;
  __REG32 EOS	            : 1;
  __REG32 OVR	            : 1;
  __REG32                 : 2;
  __REG32 AWD	            : 1;
  __REG32                 :24;
} __adc_isr_bits;

/* ADC interrupt enable register (ADC_IER) */
typedef struct {
  __REG32 ADRDYIE         : 1;
  __REG32 EOSMPIE         : 1;
  __REG32 EOCIE           : 1;
  __REG32 EOSIE	          : 1;
  __REG32 OVRIE           : 1;
  __REG32 			          : 2;
  __REG32 AWDIE           : 1;
  __REG32                 :24;
} __adc_ier_bits;

/* ADC control register (ADC_CR) */
typedef struct {
  __REG32 ADEN            : 1;
  __REG32 ADDIS           : 1;
  __REG32 ADSTART         : 1;
  __REG32                 : 1;
  __REG32 ADSTP           : 1;
  __REG32                 :26;
  __REG32 ADCAL           : 1;
} __adc_cr_bits;

/* ADC configuration register 1 (ADC_CFGR1) */
typedef struct {
  __REG32 DMAEN           : 1;
  __REG32 DMACFG          : 1;
  __REG32 SCANDIR         : 1;
  __REG32 RES		          : 2;
  __REG32 ALIGN           : 1;
  __REG32 EXTSEL          : 3;
  __REG32                 : 1;
  __REG32 EXTEN           : 2;
  __REG32 OVRMOD          : 1;
  __REG32 CONT	          : 1;
  __REG32 AUTDLY	        : 1;
  __REG32 AUTOFF          : 1;
  __REG32 DISCEN          : 1;
  __REG32                 : 5;
  __REG32 AWDSGL          : 1;
  __REG32 AWDEN           : 1;
  __REG32                 : 2;
  __REG32 AWDCH           : 5;
  __REG32                 : 1;
} __adc_cfgr1_bits;

/* ADC configuration register 2 (ADC_CFGR2) */
typedef struct {
  __REG32                 :30;
  __REG32 JITOFF_D2       : 1;
  __REG32 JITOFF_D4       : 1;
} __adc_cfgr2_bits;

/* ADC sampling time register (ADC_SMPR) */
typedef struct {
  __REG32 SMP		          : 3;
  __REG32                 :29;
} __adc_smpr_bits;

/* ADC watchdog threshold register (ADC_TR) */
typedef struct {
  __REG32 LT              :12;
  __REG32                 : 4;
  __REG32 HT              :12;
  __REG32                 : 4;
} __adc_tr_bits;

/* ADC channel selection register (ADC_CHSELR) */
typedef struct {
  __REG32 CHSEL0          : 1;
  __REG32 CHSEL1          : 1;
  __REG32 CHSEL2          : 1;
  __REG32 CHSEL3          : 1;
  __REG32 CHSEL4          : 1;
  __REG32 CHSEL5          : 1;
  __REG32 CHSEL6          : 1;
  __REG32 CHSEL7          : 1;
  __REG32 CHSEL8          : 1;
  __REG32 CHSEL9          : 1;
  __REG32 CHSEL10         : 1;
  __REG32 CHSEL11         : 1;
  __REG32 CHSEL12         : 1;
  __REG32 CHSEL13         : 1;
  __REG32 CHSEL14         : 1;
  __REG32 CHSEL15         : 1;
  __REG32 CHSEL16         : 1;
  __REG32 CHSEL17         : 1;
  __REG32                 :14;
} __adc_chselr_bits;

/* ADC regular data register (ADC_DR) */
typedef struct {
  __REG32 DATA            :16;
  __REG32                 :16;
} __adc_dr_bits;

/* ADC common configuration register (ADC_CCR) */
typedef struct {
  __REG32                 :22;
  __REG32 VREFEN          : 1;
  __REG32 TSEN          	: 1;
  __REG32 VBATEN          : 1;
  __REG32                 : 7;
} __adc_ccr_bits;

/* DAC control register (DAC_CR) */
typedef struct {
  __REG32 EN1             : 1;
  __REG32 BOFF1           : 1;
  __REG32 TEN1            : 1;
  __REG32 TSEL1           : 3;
  __REG32 		            : 6;
  __REG32 DMAEN1          : 1;
  __REG32 DMAUDRIE1       : 1;
  __REG32                 :18;
} __dac_cr_bits;

/* DAC Software Trigger Register (DAC_SWTRIGR) */
typedef struct {
  __REG32 SWTRIG1         : 1;
  __REG32                 :31;
} __dac_swtrigr_bits;

/* DAC channel1 12-bit Right-aligned Data Holding Register (DAC_DHR12R1) */
typedef struct {
  __REG32 DACC1DHR        :12;
  __REG32                 :20;
} __dac_dhr12r1_bits;

/* DAC channel1 12-bit Left aligned Data Holding Register (DAC_DHR12L1) */
typedef struct {
  __REG32                 : 4;
  __REG32 DACC1DHR        :12;
  __REG32                 :16;
} __dac_dhr12l1_bits;

/* DAC channel1 8-bit Right aligned Data Holding Register (DAC_DHR8R1) */
typedef struct {
  __REG32 DACC1DHR        : 8;
  __REG32                 :24;
} __dac_dhr8r1_bits;

/* DUAL DAC 8-bit Right aligned Data Holding Register (DAC_DHR8RD) */
typedef struct {
  __REG32 DACC1DHR        : 8;
  __REG32                 :24;
} __dac_dhr8rd_bits;

/* DAC channel1 Data Output Register (DAC_DOR1) */
typedef struct {
  __REG32 DACC1DOR        :12;
  __REG32                 :20;
} __dac_dor1_bits;

/* DAC status register (DAC_SR) */
typedef struct {
  __REG32                 :13;
  __REG32 DMAUDR1         : 1;
  __REG32                 :18;
} __dac_sr_bits;

/* COMP control and status register (COMP_CSR) */
typedef struct {
  __REG32 COMP1EN         : 1;
  __REG32 COMP1_INP_DAC   : 1;
  __REG32 COMP1MODE       : 2;
  __REG32 COMP1INSEL      : 3;
  __REG32                 : 1;
  __REG32 COMP1OUTSEL			: 3;
  __REG32 COMP1POL      	: 1;
  __REG32 COMP1HYST      	: 2;
  __REG32 COMP1OUT      	: 1;
  __REG32 COMP1LOCK      	: 1;
  __REG32 COMP2EN      		: 1;
  __REG32                 : 1;
  __REG32 COMP2MODE				: 2;
  __REG32 COMP2INSEL			: 3;
  __REG32 WNDWEN					: 1;
  __REG32 COMP2OUTSEL			: 3;
  __REG32 COMP2POL				: 1;
  __REG32 COMP2HYST				: 2;
  __REG32 COMP2OUT				: 1;
  __REG32 COMP2LOCK				: 1;
} __comp_csr_bits;

/* SysTick Control and Status Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  TICKINT        : 1;
  __REG32  CLKSOURCE      : 1;
  __REG32                 :13;
  __REG32  COUNTFLAG      : 1;
  __REG32                 :15;
} __systickcsr_bits;

/* SysTick Reload Value Register */
typedef struct {
  __REG32  RELOAD         :24;
  __REG32                 : 8;
} __systickrvr_bits;

/* SysTick Current Value Register */
typedef struct {
  __REG32  CURRENT        :24;
  __REG32                 : 8;
} __systickcvr_bits;

/* SysTick Calibration Value Register */
typedef struct {
  __REG32  TENMS          :24;
  __REG32                 : 6;
  __REG32  SKEW           : 1;
  __REG32  NOREF          : 1;
} __systickcalvr_bits;

/* Interrupt Set-Enable Registers 0-31 */
typedef struct {
  __REG32  SETENA0        : 1;
  __REG32  SETENA1        : 1;
  __REG32  SETENA2        : 1;
  __REG32  SETENA3        : 1;
  __REG32  SETENA4        : 1;
  __REG32  SETENA5        : 1;
  __REG32  SETENA6        : 1;
  __REG32  SETENA7        : 1;
  __REG32  SETENA8        : 1;
  __REG32  SETENA9        : 1;
  __REG32  SETENA10       : 1;
  __REG32  SETENA11       : 1;
  __REG32  SETENA12       : 1;
  __REG32  SETENA13       : 1;
  __REG32  SETENA14       : 1;
  __REG32  SETENA15       : 1;
  __REG32  SETENA16       : 1;
  __REG32  SETENA17       : 1;
  __REG32  SETENA18       : 1;
  __REG32  SETENA19       : 1;
  __REG32  SETENA20       : 1;
  __REG32  SETENA21       : 1;
  __REG32  SETENA22       : 1;
  __REG32  SETENA23       : 1;
  __REG32  SETENA24       : 1;
  __REG32  SETENA25       : 1;
  __REG32  SETENA26       : 1;
  __REG32  SETENA27       : 1;
  __REG32  SETENA28       : 1;
  __REG32  SETENA29       : 1;
  __REG32  SETENA30       : 1;
  __REG32  SETENA31       : 1;
} __setena0_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  CLRENA0        : 1;
  __REG32  CLRENA1        : 1;
  __REG32  CLRENA2        : 1;
  __REG32  CLRENA3        : 1;
  __REG32  CLRENA4        : 1;
  __REG32  CLRENA5        : 1;
  __REG32  CLRENA6        : 1;
  __REG32  CLRENA7        : 1;
  __REG32  CLRENA8        : 1;
  __REG32  CLRENA9        : 1;
  __REG32  CLRENA10       : 1;
  __REG32  CLRENA11       : 1;
  __REG32  CLRENA12       : 1;
  __REG32  CLRENA13       : 1;
  __REG32  CLRENA14       : 1;
  __REG32  CLRENA15       : 1;
  __REG32  CLRENA16       : 1;
  __REG32  CLRENA17       : 1;
  __REG32  CLRENA18       : 1;
  __REG32  CLRENA19       : 1;
  __REG32  CLRENA20       : 1;
  __REG32  CLRENA21       : 1;
  __REG32  CLRENA22       : 1;
  __REG32  CLRENA23       : 1;
  __REG32  CLRENA24       : 1;
  __REG32  CLRENA25       : 1;
  __REG32  CLRENA26       : 1;
  __REG32  CLRENA27       : 1;
  __REG32  CLRENA28       : 1;
  __REG32  CLRENA29       : 1;
  __REG32  CLRENA30       : 1;
  __REG32  CLRENA31       : 1;
} __clrena0_bits;

/* Interrupt Set-Pending Register 0-31 */
typedef struct {
  __REG32  SETPEND0       : 1;
  __REG32  SETPEND1       : 1;
  __REG32  SETPEND2       : 1;
  __REG32  SETPEND3       : 1;
  __REG32  SETPEND4       : 1;
  __REG32  SETPEND5       : 1;
  __REG32  SETPEND6       : 1;
  __REG32  SETPEND7       : 1;
  __REG32  SETPEND8       : 1;
  __REG32  SETPEND9       : 1;
  __REG32  SETPEND10      : 1;
  __REG32  SETPEND11      : 1;
  __REG32  SETPEND12      : 1;
  __REG32  SETPEND13      : 1;
  __REG32  SETPEND14      : 1;
  __REG32  SETPEND15      : 1;
  __REG32  SETPEND16      : 1;
  __REG32  SETPEND17      : 1;
  __REG32  SETPEND18      : 1;
  __REG32  SETPEND19      : 1;
  __REG32  SETPEND20      : 1;
  __REG32  SETPEND21      : 1;
  __REG32  SETPEND22      : 1;
  __REG32  SETPEND23      : 1;
  __REG32  SETPEND24      : 1;
  __REG32  SETPEND25      : 1;
  __REG32  SETPEND26      : 1;
  __REG32  SETPEND27      : 1;
  __REG32  SETPEND28      : 1;
  __REG32  SETPEND29      : 1;
  __REG32  SETPEND30      : 1;
  __REG32  SETPEND31      : 1;
} __setpend0_bits;

/* Interrupt Clear-Pending Register 0-31 */
typedef struct {
  __REG32  CLRPEND0       : 1;
  __REG32  CLRPEND1       : 1;
  __REG32  CLRPEND2       : 1;
  __REG32  CLRPEND3       : 1;
  __REG32  CLRPEND4       : 1;
  __REG32  CLRPEND5       : 1;
  __REG32  CLRPEND6       : 1;
  __REG32  CLRPEND7       : 1;
  __REG32  CLRPEND8       : 1;
  __REG32  CLRPEND9       : 1;
  __REG32  CLRPEND10      : 1;
  __REG32  CLRPEND11      : 1;
  __REG32  CLRPEND12      : 1;
  __REG32  CLRPEND13      : 1;
  __REG32  CLRPEND14      : 1;
  __REG32  CLRPEND15      : 1;
  __REG32  CLRPEND16      : 1;
  __REG32  CLRPEND17      : 1;
  __REG32  CLRPEND18      : 1;
  __REG32  CLRPEND19      : 1;
  __REG32  CLRPEND20      : 1;
  __REG32  CLRPEND21      : 1;
  __REG32  CLRPEND22      : 1;
  __REG32  CLRPEND23      : 1;
  __REG32  CLRPEND24      : 1;
  __REG32  CLRPEND25      : 1;
  __REG32  CLRPEND26      : 1;
  __REG32  CLRPEND27      : 1;
  __REG32  CLRPEND28      : 1;
  __REG32  CLRPEND29      : 1;
  __REG32  CLRPEND30      : 1;
  __REG32  CLRPEND31      : 1;
} __clrpend0_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32  PRI_0          : 8;
  __REG32  PRI_1          : 8;
  __REG32  PRI_2          : 8;
  __REG32  PRI_3          : 8;
} __pri0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32  PRI_4          : 8;
  __REG32  PRI_5          : 8;
  __REG32  PRI_6          : 8;
  __REG32  PRI_7          : 8;
} __pri1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32  PRI_8          : 8;
  __REG32  PRI_9          : 8;
  __REG32  PRI_10         : 8;
  __REG32  PRI_11         : 8;
} __pri2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32  PRI_12         : 8;
  __REG32  PRI_13         : 8;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __pri3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32  PRI_16         : 8;
  __REG32  PRI_17         : 8;
  __REG32  PRI_18         : 8;
  __REG32  PRI_19         : 8;
} __pri4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32  PRI_20         : 8;
  __REG32  PRI_21         : 8;
  __REG32  PRI_22         : 8;
  __REG32  PRI_23         : 8;
} __pri5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32  PRI_24         : 8;
  __REG32  PRI_25         : 8;
  __REG32  PRI_26         : 8;
  __REG32  PRI_27         : 8;
} __pri6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32  PRI_28         : 8;
  __REG32  PRI_29         : 8;
  __REG32  PRI_30         : 8;
  __REG32  PRI_31         : 8;
} __pri7_bits;

/* CPU ID Base Register */
typedef struct {
  __REG32  REVISION       : 4;
  __REG32  PARTNO         :12;
  __REG32  CONSTANT       : 4;
  __REG32  VARIANT        : 4;
  __REG32  IMPLEMENTER    : 8;
} __cpuidbr_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32  VECTACTIVE     : 6;
  __REG32                 : 6;
  __REG32  VECTPENDING    : 6;
  __REG32                 : 4;
  __REG32  ISRPENDING     : 1;
  __REG32                 : 2;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __icsr_bits;

/* Application Interrupt and Reset Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32  VECTCLRACTIVE  : 1;
  __REG32  SYSRESETREQ    : 1;
  __REG32                 :12;
  __REG32  ENDIANESS      : 1;
  __REG32  VECTKEY        :16;
} __aircr_bits;

/* System Control Register  */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scr_bits;

/* Configuration and Control Register */
typedef struct {
  __REG32                 : 3;
  __REG32  UNALIGN_TRP    : 1;
  __REG32                 : 4;
  __REG32  STKALIGN       : 1;
  __REG32                 :23;
} __ccr_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32                 :24;
  __REG32  PRI_11         : 8;
} __shpr2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32                 :16;
  __REG32  PRI_14         : 8;
  __REG32  PRI_15         : 8;
} __shpr3_bits;

/* Flash Access Control Register (FLASH_ACR) */
typedef struct {
  __REG32  LATENCY        : 3;
  __REG32                 : 1;
  __REG32  PRFTBE         : 1;
  __REG32  PRFTBS         : 1;
  __REG32                 :26;
} __flash_acr_bits;

/* Flash Status Register (FLASH_SR) */
typedef struct {
  __REG32  BSY            : 1;
  __REG32                 : 1;
  __REG32  PGERR          : 1;
  __REG32                 : 1;
  __REG32  WRPRTERR       : 1;
  __REG32  EOP            : 1;
  __REG32                 :26;
} __flash_sr_bits;

/* Flash Control Register (FLASH_CR) */
typedef struct {
  __REG32  PG             : 1;
  __REG32  PER            : 1;
  __REG32  MER            : 1;
  __REG32                 : 1;
  __REG32  OPTPG          : 1;
  __REG32  OPTER          : 1;
  __REG32  STRT           : 1;
  __REG32  LOCK           : 1;
  __REG32                 : 1;
  __REG32  OPTWRE         : 1;
  __REG32  ERRIE          : 1;
  __REG32                 : 1;
  __REG32  EOPIE          : 1;
  __REG32  FORCE_OPTLOAD  : 1;
  __REG32                 :18;
} __flash_cr_bits;

/* Option Byte Register (FLASH_OBR) */
typedef struct {
  __REG32  OPTERR         : 1;
  __REG32  LEVEL1_PROT    : 1;
  __REG32  LEVEL2_PROT    : 1;
  __REG32                 : 5;
  __REG32  WDG_SW         : 1;
  __REG32  nRST_STOP      : 1;
  __REG32  nRST_STDBY     : 1;
  __REG32                 : 1;
  __REG32  BOOT1          : 1;
  __REG32  VDDA_MONITOR   : 1;
  __REG32                 : 2;
  __REG32  Data0          : 8;
  __REG32  Data1          : 8;
} __flash_obr_bits;

/* Write Protection Register (FLASH_WRPR) */
typedef struct {
  __REG32  WRP0           : 1;
  __REG32  WRP1           : 1;
  __REG32  WRP2           : 1;
  __REG32  WRP3           : 1;
  __REG32  WRP4           : 1;
  __REG32  WRP5           : 1;
  __REG32  WRP6           : 1;
  __REG32  WRP7           : 1;
  __REG32  WRP8           : 1;
  __REG32  WRP9           : 1;
  __REG32  WRP10          : 1;
  __REG32  WRP11          : 1;
  __REG32  WRP12          : 1;
  __REG32  WRP13          : 1;
  __REG32  WRP14          : 1;
  __REG32  WRP15          : 1;
  __REG32                 :16;
} __flash_wrpr_bits;

/* Independent data register (CRC_IDR) */
typedef struct {
  __REG32  IDR            : 8;
  __REG32                 :24;
} __crc_idr_bits;

/* Control register (CRC_CR) */
typedef struct {
  __REG32  RESET          : 1;
  __REG32                 : 4;
  __REG32  REV_IN         : 2;
  __REG32  REV_OUT        : 1;
  __REG32                 :24;
} __crc_cr_bits;

/* TSC control register (TSC_CR) */
typedef struct {
  __REG32 TSCE	          : 1;
  __REG32 START	          : 1;
  __REG32 AM		          : 1;
  __REG32 SYNCPOL	        : 1;
  __REG32 IODEF	          : 1;
  __REG32 MCV 	          : 3;
  __REG32 			          : 4;
  __REG32 PGPSC	          : 3;
  __REG32 SSPSC           : 1;
  __REG32 SSE	            : 1;
  __REG32 SSD	            : 7;
  __REG32 CTPL	          : 4;
  __REG32 CTPH            : 4;
} __tsc_cr_bits;

/* TSC interrupt enable register (TSC_IER) */
typedef struct {
  __REG32 EOAIE	          : 1;
  __REG32 MCEIE	          : 1;
  __REG32 			          :30;
} __tsc_ier_bits;

/* TSC interrupt clear register (TSC_ICR) */
typedef struct {
  __REG32 EOAIC	          : 1;
  __REG32 MCEIC	          : 1;
  __REG32 			          :30;
} __tsc_icr_bits;

/* TSC interrupt status register (TSC_ISR) */
typedef struct {
  __REG32 EOAF 	          : 1;
  __REG32 MCEF	          : 1;
  __REG32 			          :30;
} __tsc_isr_bits;

/* TSC I/O hysteresis control register (TSC_IOHCR) */
/* TSC I/O analog switch control register (TSC_IOASCR) */
/* TSC I/O sampling control register (TSC_IOSCR) */
typedef struct {
  __REG32 G1_IO1          : 1;
  __REG32 G1_IO2          : 1;
  __REG32 G1_IO3          : 1;
  __REG32 G1_IO4          : 1;
  __REG32 G2_IO1          : 1;
  __REG32 G2_IO2          : 1;
  __REG32 G2_IO3          : 1;
  __REG32 G2_IO4          : 1;
  __REG32 G3_IO1          : 1;
  __REG32 G3_IO2          : 1;
  __REG32 G3_IO3          : 1;
  __REG32 G3_IO4          : 1;
  __REG32 G4_IO1          : 1;
  __REG32 G4_IO2          : 1;
  __REG32 G4_IO3          : 1;
  __REG32 G4_IO4          : 1;
  __REG32 G5_IO1          : 1;
  __REG32 G5_IO2          : 1;
  __REG32 G5_IO3          : 1;
  __REG32 G5_IO4          : 1;
  __REG32 G6_IO1          : 1;
  __REG32 G6_IO2          : 1;
  __REG32 G6_IO3          : 1;
  __REG32 G6_IO4          : 1;
  __REG32 			          : 8;
} __tsc_iohcr_bits;

/* TSC I/O group control status register (TSC_IOGCSR) */
typedef struct {
  __REG32 G1E		          : 1;
  __REG32 G2E		          : 1;
  __REG32 G3E		          : 1;
  __REG32 G4E		          : 1;
  __REG32 G5E		          : 1;
  __REG32 G6E		          : 1;
  __REG32   		          :10;
  __REG32 G1S		          : 1;
  __REG32 G2S		          : 1;
  __REG32 G3S		          : 1;
  __REG32 G4S		          : 1;
  __REG32 G5S		          : 1;
  __REG32 G6S		          : 1;
  __REG32 			          :10;
} __tsc_iogcsr_bits;

/* TSC I/O group x counter register (TSC_IOGxCR) (x=1..6) */
typedef struct {
  __REG32 CNT		          :14;
  __REG32   		          :18;
} __tsc_iogcr_bits;

/* CEC control register (CEC_CR) */
typedef struct {
  __REG32 CECON		        : 1;
  __REG32 TXSOM		        : 1;
  __REG32 TXEOM		        : 1;
  __REG32   		          :29;
} __cec_cr_bits;

/* CEC configuration register (CEC_CFGR) */
typedef struct {
  __REG32 OAR 		        : 4;
  __REG32 LSTN		        : 1;
  __REG32 SFT			        : 3;
  __REG32 RXTOL		        : 1;
  __REG32 BRESTP		      : 1;
  __REG32 BREGEN		      : 1;
  __REG32 LBPEGEN		      : 1;
  __REG32   		          :20;
} __cec_cfgr_bits;

/* CEC Tx data register (CEC_TXDR) */
typedef struct {
  __REG32 TXD 		        : 8;
  __REG32   		          :24;
} __cec_txdr_bits;

/* CEC Rx Data Register (CEC_RXDR) */
typedef struct {
  __REG32 RXD 		        : 8;
  __REG32   		          :24;
} __cec_rxdr_bits;

/* CEC Interrupt and Status Register (CEC_ISR) */
typedef struct {
  __REG32 RXBR 		        : 1;
  __REG32 RXEND 		      : 1;
  __REG32 RXOVR 		      : 1;
  __REG32 BRE  		        : 1;
  __REG32 SBPE 		        : 1;
  __REG32 LBPE 		        : 1;
  __REG32 RXACKE 		      : 1;
  __REG32 ARBLST 		      : 1;
  __REG32 TXBR 		        : 1;
  __REG32 TXEND 		      : 1;
  __REG32 TXUDR 		      : 1;
  __REG32 TXERR 		      : 1;
  __REG32 TXACKE 		      : 1;
  __REG32   		          :19;
} __cec_isr_bits;

/* CEC interrupt enable register (CEC_IER) */
typedef struct {
  __REG32 RXBRIE	        : 1;
  __REG32 RXENDIE		      : 1;
  __REG32 RXOVRIE		      : 1;
  __REG32 BREIE		        : 1;
  __REG32 SBPEIE	        : 1;
  __REG32 LBPEIE	        : 1;
  __REG32 RXACKEIE	      : 1;
  __REG32 ARBLSTIE	      : 1;
  __REG32 TXBRIE	        : 1;
  __REG32 TXENDIE		      : 1;
  __REG32 TXUDRIE		      : 1;
  __REG32 TXERRIE		      : 1;
  __REG32 TXACKEIE	      : 1;
  __REG32   		          :19;
} __cec_ier_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,               0xE000E100,__READ_WRITE ,__setena0_bits);
#define ISER        SETENA0
#define ISER_bit    SETENA0_bit
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
#define ICER        CLRENA0
#define ICER_bit    CLRENA0_bit
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
#define ISPR        SETPEND0
#define ISPR_bit    SETPEND0_bit
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
#define ICPR        CLRPEND0
#define ICPR_bit    CLRPEND0_bit
__IO_REG32_BIT(IP0,                   0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                   0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                   0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                   0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                   0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                   0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                   0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                   0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
#define CPUID         CPUIDBR
#define CPUID_bit     CPUIDBR_bit
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED1C,__READ_WRITE ,__shpr2_bits);
__IO_REG32_BIT(SHPR3,                 0xE000ED20,__READ_WRITE ,__shpr3_bits);

/***************************************************************************
 **
 ** PWR
 **
 ***************************************************************************/
__IO_REG32_BIT(PWR_CR,            0x40007000,__READ_WRITE ,__pwr_cr_bits);
__IO_REG32_BIT(PWR_CSR,           0x40007004,__READ_WRITE ,__pwr_csr_bits);

/***************************************************************************
 **
 ** RCC
 **
 ***************************************************************************/
__IO_REG32_BIT(RCC_CR,            0x40021000,__READ_WRITE ,__rcc_cr_bits);
__IO_REG32_BIT(RCC_CFGR,          0x40021004,__READ_WRITE ,__rcc_cfgr_bits);
__IO_REG32_BIT(RCC_CIR,           0x40021008,__READ_WRITE ,__rcc_cir_bits);
__IO_REG32_BIT(RCC_APB2RSTR,      0x4002100C,__READ_WRITE ,__rcc_apb2rstr_bits);
__IO_REG32_BIT(RCC_APB1RSTR,      0x40021010,__READ_WRITE ,__rcc_apb1rstr_bits);
__IO_REG32_BIT(RCC_AHBENR,        0x40021014,__READ_WRITE ,__rcc_ahbenr_bits);
__IO_REG32_BIT(RCC_APB2ENR,       0x40021018,__READ_WRITE ,__rcc_apb2enr_bits);
__IO_REG32_BIT(RCC_APB1ENR,       0x4002101C,__READ_WRITE ,__rcc_apb1enr_bits);
__IO_REG32_BIT(RCC_BDCR,          0x40021020,__READ_WRITE ,__rcc_bdcr_bits);
__IO_REG32_BIT(RCC_CSR,           0x40021024,__READ_WRITE ,__rcc_csr_bits);
__IO_REG32_BIT(RCC_AHBRSTR,       0x40021028,__READ_WRITE ,__rcc_ahbrstr_bits);
__IO_REG32_BIT(RCC_CFGR2,         0x4002102C,__READ_WRITE ,__rcc_cfgr2_bits);
__IO_REG32_BIT(RCC_CFGR3,         0x40021030,__READ_WRITE ,__rcc_cfgr3_bits);
__IO_REG32_BIT(RCC_CR2,           0x40021034,__READ_WRITE ,__rcc_cr2_bits);

/***************************************************************************
 **
 ** GPIOA
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_MODER,       0x48000000,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOA_OTYPER,      0x48000004,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOA_OSPEEDR,     0x48000008,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOA_PUPDR,       0x4800000C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOA_IDR,         0x48000010,__READ_WRITE ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOA_ODR,         0x48000014,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOA_BSRR,        0x48000018,__READ_WRITE ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOA_LCKR,        0x4800001C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOA_AFRL,        0x48000020,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOA_AFRH,        0x48000024,__READ_WRITE ,__gpio_afrh_bits);
__IO_REG32_BIT(GPIOA_BRR,         0x48000028,__READ_WRITE ,__gpio_brr_bits);

/***************************************************************************
 **
 ** GPIOB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOB_MODER,       0x48000400,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOB_OTYPER,      0x48000404,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOB_OSPEEDR,     0x48000408,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOB_PUPDR,       0x4800040C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOB_IDR,         0x48000410,__READ_WRITE ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOB_ODR,         0x48000414,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOB_BSRR,        0x48000418,__READ_WRITE ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOB_LCKR,        0x4800041C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOB_AFRL,        0x48000420,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOB_AFRH,        0x48000424,__READ_WRITE ,__gpio_afrh_bits);
__IO_REG32_BIT(GPIOB_BRR,         0x48000428,__READ_WRITE ,__gpio_brr_bits);

/***************************************************************************
 **
 ** GPIOC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOC_MODER,       0x48000800,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOC_OTYPER,      0x48000804,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOC_OSPEEDR,     0x48000808,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOC_PUPDR,       0x4800080C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOC_IDR,         0x48000810,__READ_WRITE ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOC_ODR,         0x48000814,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOC_BSRR,        0x48000818,__READ_WRITE ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOC_LCKR,        0x4800081C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOC_AFRL,        0x48000820,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOC_AFRH,        0x48000824,__READ_WRITE ,__gpio_afrh_bits);
__IO_REG32_BIT(GPIOC_BRR,         0x48000828,__READ_WRITE ,__gpio_brr_bits);

/***************************************************************************
 **
 ** GPIOD
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOD_MODER,       0x48000C00,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOD_OTYPER,      0x48000C04,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOD_OSPEEDR,     0x48000C08,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOD_PUPDR,       0x48000C0C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOD_IDR,         0x48000C10,__READ_WRITE ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOD_ODR,         0x48000C14,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOD_BSRR,        0x48000C18,__READ_WRITE ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOD_LCKR,        0x48000C1C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOD_AFRL,        0x48000C20,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOD_AFRH,        0x48000C24,__READ_WRITE ,__gpio_afrh_bits);
__IO_REG32_BIT(GPIOD_BRR,         0x48000C28,__READ_WRITE ,__gpio_brr_bits);

/***************************************************************************
 **
 ** GPIOF
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOF_MODER,       0x48001400,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOF_OTYPER,      0x48001404,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOF_OSPEEDR,     0x48001408,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOF_PUPDR,       0x4800140C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOF_IDR,         0x48001410,__READ_WRITE ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOF_ODR,         0x48001414,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOF_BSRR,        0x48001418,__READ_WRITE ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOF_LCKR,        0x4800141C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOF_AFRL,        0x48001420,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOF_AFRH,        0x48001424,__READ_WRITE ,__gpio_afrh_bits);
__IO_REG32_BIT(GPIOF_BRR,         0x48001428,__READ_WRITE ,__gpio_brr_bits);

/***************************************************************************
 **
 ** SYSCFG
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSCFG_CFGR1,      0x40010000,__READ_WRITE ,__syscfg_cfgr1_bits);
__IO_REG32_BIT(SYSCFG_EXTICR1,    0x40010008,__READ_WRITE ,__syscfg_exticr1_bits);
__IO_REG32_BIT(SYSCFG_EXTICR2,    0x4001000C,__READ_WRITE ,__syscfg_exticr2_bits);
__IO_REG32_BIT(SYSCFG_EXTICR3,    0x40010010,__READ_WRITE ,__syscfg_exticr3_bits);
__IO_REG32_BIT(SYSCFG_EXTICR4,    0x40010014,__READ_WRITE ,__syscfg_exticr4_bits);
__IO_REG32_BIT(SYSCFG_CFGR2,      0x40010018,__READ_WRITE ,__syscfg_cfgr2_bits);

/***************************************************************************
 **
 ** EXTI
 **
 ***************************************************************************/
__IO_REG32_BIT(EXTI_IMR,          0x40010400,__READ_WRITE ,__exti_imr_bits);
__IO_REG32_BIT(EXTI_EMR,          0x40010404,__READ_WRITE ,__exti_emr_bits);
__IO_REG32_BIT(EXTI_RTSR,         0x40010408,__READ_WRITE ,__exti_rtsr_bits);
__IO_REG32_BIT(EXTI_FTSR,         0x4001040C,__READ_WRITE ,__exti_ftsr_bits);
__IO_REG32_BIT(EXTI_SWIER,        0x40010410,__READ_WRITE ,__exti_swier_bits);
__IO_REG32_BIT(EXTI_PR,           0x40010414,__READ_WRITE ,__exti_pr_bits);

/***************************************************************************
 **
 ** DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA_ISR,           0x40020000,__READ       ,__dma_isr_bits);
__IO_REG32_BIT(DMA_IFCR,          0x40020004,__WRITE      ,__dma_ifcr_bits);
__IO_REG32_BIT(DMA_CCR1,          0x40020008,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR1,        0x4002000C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR1,         0x40020010,__READ_WRITE );
__IO_REG32(    DMA_CMAR1,         0x40020014,__READ_WRITE );
__IO_REG32_BIT(DMA_CCR2,          0x4002001C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR2,        0x40020020,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR2,         0x40020024,__READ_WRITE );
__IO_REG32(    DMA_CMAR2,         0x40020028,__READ_WRITE );
__IO_REG32_BIT(DMA_CCR3,          0x40020030,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR3,        0x40020034,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR3,         0x40020038,__READ_WRITE );
__IO_REG32(    DMA_CMAR3,         0x4002003C,__READ_WRITE );
__IO_REG32_BIT(DMA_CCR4,          0x40020044,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR4,        0x40020048,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR4,         0x4002004C,__READ_WRITE );
__IO_REG32(    DMA_CMAR4,         0x40020050,__READ_WRITE );
__IO_REG32_BIT(DMA_CCR5,          0x40020058,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR5,        0x4002005C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR5,         0x40020060,__READ_WRITE );
__IO_REG32(    DMA_CMAR5,         0x40020064,__READ_WRITE );

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_TR,           	0x40002800,__READ_WRITE ,__rtc_tr_bits);
__IO_REG32_BIT(RTC_DR,           	0x40002804,__READ_WRITE ,__rtc_dr_bits);
__IO_REG32_BIT(RTC_CR,           	0x40002808,__READ_WRITE ,__rtc_cr_bits);
__IO_REG32_BIT(RTC_ISR,           0x4000280C,__READ_WRITE ,__rtc_isr_bits);
__IO_REG32_BIT(RTC_PRER,          0x40002810,__READ_WRITE ,__rtc_prer_bits);
__IO_REG32_BIT(RTC_ALRMAR,        0x4000281C,__READ_WRITE ,__rtc_alrmar_bits);
__IO_REG32_BIT(RTC_WPR,        		0x40002824,__WRITE 		  ,__rtc_wpr_bits);
__IO_REG32_BIT(RTC_SSR,        		0x40002828,__READ				,__rtc_ssr_bits);
__IO_REG32_BIT(RTC_SHIFTR,        0x4000282C,__WRITE 		  ,__rtc_shiftr_bits);
__IO_REG32_BIT(RTC_TSTR,        	0x40002830,__READ				,__rtc_tr_bits);
__IO_REG32_BIT(RTC_TSDR,        	0x40002834,__READ				,__rtc_tsdr_bits);
__IO_REG32_BIT(RTC_TSSSR,        	0x40002838,__READ				,__rtc_ssr_bits);
__IO_REG32_BIT(RTC_CALR,        	0x4000283C,__READ_WRITE ,__rtc_calr_bits);
__IO_REG32_BIT(RTC_TAFCR,        	0x40002840,__READ_WRITE ,__rtc_tafcr_bits);
__IO_REG32_BIT(RTC_ALRMASSR,      0x40002844,__READ_WRITE ,__rtc_alrmassr_bits);
__IO_REG32(		 RTC_BKP0R,      		0x40002850,__READ_WRITE );
__IO_REG32(		 RTC_BKP1R,      		0x40002854,__READ_WRITE );
__IO_REG32(		 RTC_BKP2R,      		0x40002858,__READ_WRITE );
__IO_REG32(		 RTC_BKP3R,      		0x4000285C,__READ_WRITE );
__IO_REG32(		 RTC_BKP4R,      		0x40002860,__READ_WRITE );

/***************************************************************************
 **
 ** IWWDG
 **
 ***************************************************************************/
__IO_REG32(    IWWDG_KR,          0x40003000,__WRITE      );
__IO_REG32_BIT(IWWDG_PR,          0x40003004,__READ_WRITE ,__iwwdg_pr_bits);
__IO_REG32_BIT(IWWDG_RLR,         0x40003008,__READ_WRITE ,__iwwdg_rlr_bits);
__IO_REG32_BIT(IWWDG_SR,          0x4000300C,__READ       ,__iwwdg_sr_bits);
__IO_REG32_BIT(IWWDG_WINR,        0x40003010,__READ_WRITE	,__iwwdg_winr_bits);

/***************************************************************************
 **
 ** WWDG
 **
 ***************************************************************************/
__IO_REG32_BIT(WWDG_CR,           0x40002C00,__READ_WRITE ,__wwdg_cr_bits);
__IO_REG32_BIT(WWDG_CFR,          0x40002C04,__READ_WRITE ,__wwdg_cfr_bits);
__IO_REG32_BIT(WWDG_SR,           0x40002C08,__READ_WRITE ,__wwdg_sr_bits);

/***************************************************************************
 **
 ** TIM1
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM1_CR1,          0x40012C00,__READ_WRITE ,__tim1_cr1_bits);
__IO_REG32_BIT(TIM1_CR2,          0x40012C04,__READ_WRITE ,__tim1_cr2_bits);
__IO_REG32_BIT(TIM1_SMCR,         0x40012C08,__READ_WRITE ,__tim1_smcr_bits);
__IO_REG32_BIT(TIM1_DIER,         0x40012C0C,__READ_WRITE ,__tim1_dier_bits);
__IO_REG32_BIT(TIM1_SR,           0x40012C10,__READ_WRITE ,__tim1_sr_bits);
__IO_REG32_BIT(TIM1_EGR,          0x40012C14,__WRITE 			,__tim1_egr_bits);
__IO_REG32_BIT(TIM1_CCMR1,        0x40012C18,__READ_WRITE ,__tim1_ccmr1_bits);
#define TIM1_OCMR1      TIM1_CCMR1
#define TIM1_OCMR1_bit  TIM1_CCMR1_bit
__IO_REG32_BIT(TIM1_CCMR2,        0x40012C1C,__READ_WRITE ,__tim1_ccmr2_bits);
#define TIM1_OCMR2      TIM1_CCMR2
#define TIM1_OCMR2_bit  TIM1_CCMR2_bit
__IO_REG32_BIT(TIM1_CCER,         0x40012C20,__READ_WRITE ,__tim1_ccer_bits);
__IO_REG32_BIT(TIM1_CNT,          0x40012C24,__READ_WRITE ,__tim1_cnt_bits);
__IO_REG32_BIT(TIM1_PSC,          0x40012C28,__READ_WRITE ,__tim1_psc_bits);
__IO_REG32_BIT(TIM1_ARR,          0x40012C2C,__READ_WRITE ,__tim1_arr_bits);
__IO_REG32_BIT(TIM1_RCR,          0x40012C30,__READ_WRITE ,__tim1_rcr_bits);
__IO_REG32_BIT(TIM1_CCR1,         0x40012C34,__READ_WRITE ,__tim1_ccr_bits);
__IO_REG32_BIT(TIM1_CCR2,         0x40012C38,__READ_WRITE ,__tim1_ccr_bits);
__IO_REG32_BIT(TIM1_CCR3,         0x40012C3C,__READ_WRITE ,__tim1_ccr_bits);
__IO_REG32_BIT(TIM1_CCR4,         0x40012C40,__READ_WRITE ,__tim1_ccr_bits);
__IO_REG32_BIT(TIM1_BDTR,         0x40012C44,__READ_WRITE ,__tim1_bdtr_bits);
__IO_REG32_BIT(TIM1_DCR,          0x40012C48,__READ_WRITE ,__tim1_dcr_bits);
__IO_REG32_BIT(TIM1_DMAR,         0x40012C4C,__READ_WRITE ,__tim1_dmar_bits);

/***************************************************************************
 **
 ** TIM2
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM2_CR1,          0x40000000,__READ_WRITE ,__tim_cr1_bits);
__IO_REG32_BIT(TIM2_CR2,          0x40000004,__READ_WRITE ,__tim_cr2_bits);
__IO_REG32_BIT(TIM2_SMCR,         0x40000008,__READ_WRITE ,__tim_smcr_bits);
__IO_REG32_BIT(TIM2_DIER,         0x4000000C,__READ_WRITE ,__tim_dier_bits);
__IO_REG32_BIT(TIM2_SR,           0x40000010,__READ_WRITE ,__tim_sr_bits);
__IO_REG32_BIT(TIM2_EGR,          0x40000014,__WRITE 			,__tim_egr_bits);
__IO_REG32_BIT(TIM2_CCMR1,        0x40000018,__READ_WRITE ,__tim_ccmr1_bits);
#define TIM2_OCMR1      TIM2_CCMR1
#define TIM2_OCMR1_bit  TIM2_CCMR1_bit
__IO_REG32_BIT(TIM2_CCMR2,        0x4000001C,__READ_WRITE ,__tim_ccmr2_bits);
#define TIM2_OCMR2      TIM2_CCMR2
#define TIM2_OCMR2_bit  TIM2_CCMR2_bit
__IO_REG32_BIT(TIM2_CCER,         0x40000020,__READ_WRITE ,__tim_ccer_bits);
__IO_REG32(    TIM2_CNT,          0x40000024,__READ_WRITE );
__IO_REG32_BIT(TIM2_PSC,          0x40000028,__READ_WRITE ,__tim_psc_bits);
__IO_REG32(    TIM2_ARR,          0x4000002C,__READ_WRITE );
__IO_REG32(    TIM2_CCR1,         0x40000034,__READ_WRITE );
__IO_REG32(    TIM2_CCR2,         0x40000038,__READ_WRITE );
__IO_REG32(    TIM2_CCR3,         0x4000003C,__READ_WRITE );
__IO_REG32(    TIM2_CCR4,         0x40000040,__READ_WRITE );
__IO_REG32_BIT(TIM2_DCR,          0x40000048,__READ_WRITE ,__tim_dcr_bits);
__IO_REG32_BIT(TIM2_DMAR,         0x4000004C,__READ_WRITE ,__tim_dmar_bits);

/***************************************************************************
 **
 ** TIM3
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM3_CR1,          0x40000400,__READ_WRITE ,__tim_cr1_bits);
__IO_REG32_BIT(TIM3_CR2,          0x40000404,__READ_WRITE ,__tim_cr2_bits);
__IO_REG32_BIT(TIM3_SMCR,         0x40000408,__READ_WRITE ,__tim_smcr_bits);
__IO_REG32_BIT(TIM3_DIER,         0x4000040C,__READ_WRITE ,__tim_dier_bits);
__IO_REG32_BIT(TIM3_SR,           0x40000410,__READ_WRITE ,__tim_sr_bits);
__IO_REG32_BIT(TIM3_EGR,          0x40000414,__WRITE 			,__tim_egr_bits);
__IO_REG32_BIT(TIM3_CCMR1,        0x40000418,__READ_WRITE ,__tim_ccmr1_bits);
#define TIM3_OCMR1      TIM3_CCMR1
#define TIM3_OCMR1_bit  TIM3_CCMR1_bit
__IO_REG32_BIT(TIM3_CCMR2,        0x4000041C,__READ_WRITE ,__tim_ccmr2_bits);
#define TIM3_OCMR2      TIM3_CCMR2
#define TIM3_OCMR2_bit  TIM3_CCMR2_bit
__IO_REG32_BIT(TIM3_CCER,         0x40000420,__READ_WRITE ,__tim_ccer_bits);
__IO_REG32_BIT(TIM3_CNT,          0x40000424,__READ_WRITE ,__tim_cnt_bits);
__IO_REG32_BIT(TIM3_PSC,          0x40000428,__READ_WRITE ,__tim_psc_bits);
__IO_REG32_BIT(TIM3_ARR,          0x4000042C,__READ_WRITE ,__tim_arr_bits);
__IO_REG32_BIT(TIM3_CCR1,         0x40000434,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM3_CCR2,         0x40000438,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM3_CCR3,         0x4000043C,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM3_CCR4,         0x40000440,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM3_DCR,          0x40000448,__READ_WRITE ,__tim_dcr_bits);
__IO_REG32_BIT(TIM3_DMAR,         0x4000044C,__READ_WRITE ,__tim_dmar_bits);

/***************************************************************************
 **
 ** TIM14
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM14_CR1,         0x40002000,__READ_WRITE ,__tim14_cr1_bits);
__IO_REG32_BIT(TIM14_SMCR,        0x40002008,__READ_WRITE ,__tim14_smcr_bits);
__IO_REG32_BIT(TIM14_DIER,        0x4000200C,__READ_WRITE ,__tim14_dier_bits);
__IO_REG32_BIT(TIM14_SR,          0x40002010,__READ_WRITE ,__tim14_sr_bits);
__IO_REG32_BIT(TIM14_EGR,         0x40002014,__WRITE 			,__tim14_egr_bits);
__IO_REG32_BIT(TIM14_CCMR1,       0x40002018,__READ_WRITE ,__tim14_ccmr1_bits);
#define TIM14_OCMR1      TIM14_CCMR1
#define TIM14_OCMR1_bit  TIM14_CCMR1_bit
__IO_REG32_BIT(TIM14_CCER,        0x40002020,__READ_WRITE ,__tim14_ccer_bits);
__IO_REG32_BIT(TIM14_CNT,         0x40002024,__READ_WRITE ,__tim14_cnt_bits);
__IO_REG32_BIT(TIM14_PSC,         0x40002028,__READ_WRITE ,__tim14_psc_bits);
__IO_REG32_BIT(TIM14_ARR,         0x4000202C,__READ_WRITE ,__tim14_arr_bits);
__IO_REG32_BIT(TIM14_CCR1,        0x40002034,__READ_WRITE ,__tim14_ccr_bits);
__IO_REG32_BIT(TIM14_OR,        	0x40002050,__READ_WRITE ,__tim14_or_bits);

/***************************************************************************
 **
 ** TIM15
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM15_CR1,         0x40014000,__READ_WRITE ,__tim15_cr1_bits);
__IO_REG32_BIT(TIM15_CR2,         0x40014004,__READ_WRITE ,__tim15_cr2_bits);
__IO_REG32_BIT(TIM15_SMCR,        0x40014008,__READ_WRITE ,__tim15_smcr_bits);
__IO_REG32_BIT(TIM15_DIER,        0x4001400C,__READ_WRITE ,__tim15_dier_bits);
__IO_REG32_BIT(TIM15_SR,          0x40014010,__READ_WRITE ,__tim15_sr_bits);
__IO_REG32_BIT(TIM15_EGR,         0x40014014,__WRITE 			,__tim15_egr_bits);
__IO_REG32_BIT(TIM15_CCMR1,       0x40014018,__READ_WRITE ,__tim15_ccmr1_bits);
#define TIM15_OCMR1      TIM15_CCMR1
#define TIM15_OCMR1_bit  TIM15_CCMR1_bit
__IO_REG32_BIT(TIM15_CCER,        0x40014020,__READ_WRITE ,__tim15_ccer_bits);
__IO_REG32_BIT(TIM15_CNT,         0x40014024,__READ_WRITE ,__tim15_cnt_bits);
__IO_REG32_BIT(TIM15_PSC,         0x40014028,__READ_WRITE ,__tim15_psc_bits);
__IO_REG32_BIT(TIM15_ARR,         0x4001402C,__READ_WRITE ,__tim15_arr_bits);
__IO_REG32_BIT(TIM15_RCR,         0x40014030,__READ_WRITE ,__tim15_rcr_bits);
__IO_REG32_BIT(TIM15_CCR1,        0x40014034,__READ_WRITE ,__tim15_ccr_bits);
__IO_REG32_BIT(TIM15_CCR2,        0x40014038,__READ_WRITE ,__tim15_ccr_bits);
__IO_REG32_BIT(TIM15_BDTR,        0x40014044,__READ_WRITE ,__tim15_bdtr_bits);
__IO_REG32_BIT(TIM15_DCR,         0x40014048,__READ_WRITE ,__tim15_dcr_bits);
__IO_REG32_BIT(TIM15_DMAR,        0x4001404C,__READ_WRITE ,__tim15_dmar_bits);

/***************************************************************************
 **
 ** TIM16
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM16_CR1,         0x40014400,__READ_WRITE ,__tim16_cr1_bits);
__IO_REG32_BIT(TIM16_CR2,         0x40014404,__READ_WRITE ,__tim16_cr2_bits);
__IO_REG32_BIT(TIM16_DIER,        0x4001440C,__READ_WRITE ,__tim16_dier_bits);
__IO_REG32_BIT(TIM16_SR,          0x40014410,__READ_WRITE ,__tim16_sr_bits);
__IO_REG32_BIT(TIM16_EGR,         0x40014414,__WRITE 			,__tim16_egr_bits);
__IO_REG32_BIT(TIM16_CCMR1,       0x40014418,__READ_WRITE ,__tim16_ccmr1_bits);
#define TIM16_OCMR1      TIM16_CCMR1
#define TIM16_OCMR1_bit  TIM16_CCMR1_bit
__IO_REG32_BIT(TIM16_CCER,        0x40014420,__READ_WRITE ,__tim16_ccer_bits);
__IO_REG32_BIT(TIM16_CNT,         0x40014424,__READ_WRITE ,__tim16_cnt_bits);
__IO_REG32_BIT(TIM16_PSC,         0x40014428,__READ_WRITE ,__tim16_psc_bits);
__IO_REG32_BIT(TIM16_ARR,         0x4001442C,__READ_WRITE ,__tim16_arr_bits);
__IO_REG32_BIT(TIM16_RCR,         0x40014430,__READ_WRITE ,__tim16_rcr_bits);
__IO_REG32_BIT(TIM16_CCR1,        0x40014434,__READ_WRITE ,__tim16_ccr_bits);
__IO_REG32_BIT(TIM16_BDTR,        0x40014444,__READ_WRITE ,__tim16_bdtr_bits);
__IO_REG32_BIT(TIM16_DCR,         0x40014448,__READ_WRITE ,__tim16_dcr_bits);
__IO_REG32_BIT(TIM16_DMAR,        0x4001444C,__READ_WRITE ,__tim16_dmar_bits);

/***************************************************************************
 **
 ** TIM17
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM17_CR1,         0x40014800,__READ_WRITE ,__tim16_cr1_bits);
__IO_REG32_BIT(TIM17_CR2,         0x40014804,__READ_WRITE ,__tim16_cr2_bits);
__IO_REG32_BIT(TIM17_DIER,        0x4001480C,__READ_WRITE ,__tim16_dier_bits);
__IO_REG32_BIT(TIM17_SR,          0x40014810,__READ_WRITE ,__tim16_sr_bits);
__IO_REG32_BIT(TIM17_EGR,         0x40014814,__WRITE 			,__tim16_egr_bits);
__IO_REG32_BIT(TIM17_CCMR1,       0x40014818,__READ_WRITE ,__tim16_ccmr1_bits);
#define TIM17_OCMR1      TIM17_CCMR1
#define TIM17_OCMR1_bit  TIM17_CCMR1_bit
__IO_REG32_BIT(TIM17_CCER,        0x40014820,__READ_WRITE ,__tim16_ccer_bits);
__IO_REG32_BIT(TIM17_CNT,         0x40014824,__READ_WRITE ,__tim16_cnt_bits);
__IO_REG32_BIT(TIM17_PSC,         0x40014828,__READ_WRITE ,__tim16_psc_bits);
__IO_REG32_BIT(TIM17_ARR,         0x4001482C,__READ_WRITE ,__tim16_arr_bits);
__IO_REG32_BIT(TIM17_RCR,         0x40014830,__READ_WRITE ,__tim16_rcr_bits);
__IO_REG32_BIT(TIM17_CCR1,        0x40014834,__READ_WRITE ,__tim16_ccr_bits);
__IO_REG32_BIT(TIM17_BDTR,        0x40014844,__READ_WRITE ,__tim16_bdtr_bits);
__IO_REG32_BIT(TIM17_DCR,         0x40014848,__READ_WRITE ,__tim16_dcr_bits);
__IO_REG32_BIT(TIM17_DMAR,        0x4001484C,__READ_WRITE ,__tim16_dmar_bits);

/***************************************************************************
 **
 ** TIM6
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM6_CR1,          0x40001000,__READ_WRITE ,__tim6_cr1_bits);
__IO_REG32_BIT(TIM6_CR2,          0x40001004,__READ_WRITE ,__tim6_cr2_bits);
__IO_REG32_BIT(TIM6_DIER,         0x4000100C,__READ_WRITE ,__tim6_dier_bits);
__IO_REG32_BIT(TIM6_SR,           0x40001010,__READ_WRITE ,__tim6_sr_bits);
__IO_REG32_BIT(TIM6_EGR,          0x40001014,__WRITE 			,__tim6_egr_bits);
__IO_REG32_BIT(TIM6_CNT,          0x40001024,__READ_WRITE ,__tim6_cnt_bits);
__IO_REG32_BIT(TIM6_PSC,          0x40001028,__READ_WRITE ,__tim6_psc_bits);
__IO_REG32_BIT(TIM6_ARR,          0x4000102C,__READ_WRITE ,__tim6_arr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_CR1,          0x40005400,__READ_WRITE ,__i2c_cr1_bits);
__IO_REG32_BIT(I2C1_CR2,          0x40005404,__READ_WRITE ,__i2c_cr2_bits);
__IO_REG32_BIT(I2C1_OAR1,         0x40005408,__READ_WRITE ,__i2c_oar1_bits);
__IO_REG32_BIT(I2C1_OAR2,         0x4000540C,__READ_WRITE ,__i2c_oar2_bits);
__IO_REG32_BIT(I2C1_TIMINGR,      0x40005410,__READ_WRITE ,__i2c_timingr_bits);
__IO_REG32_BIT(I2C1_TIMEOUTR,     0x40005414,__READ_WRITE ,__i2c_timeoutr_bits);
__IO_REG32_BIT(I2C1_ISR,     			0x40005418,__READ_WRITE ,__i2c_isr_bits);
__IO_REG32_BIT(I2C1_ICR,     			0x4000541C,__WRITE 			,__i2c_icr_bits);
__IO_REG32_BIT(I2C1_PECR,     		0x40005420,__READ 			,__i2c_pecr_bits);
__IO_REG32_BIT(I2C1_RXDR,     		0x40005424,__READ 			,__i2c_rxdr_bits);
__IO_REG32_BIT(I2C1_TXDR,         0x40005428,__READ_WRITE ,__i2c_txdr_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C2_CR1,          0x40005800,__READ_WRITE ,__i2c_cr1_bits);
__IO_REG32_BIT(I2C2_CR2,          0x40005804,__READ_WRITE ,__i2c_cr2_bits);
__IO_REG32_BIT(I2C2_OAR1,         0x40005808,__READ_WRITE ,__i2c_oar1_bits);
__IO_REG32_BIT(I2C2_OAR2,         0x4000580C,__READ_WRITE ,__i2c_oar2_bits);
__IO_REG32_BIT(I2C2_TIMINGR,      0x40005810,__READ_WRITE ,__i2c_timingr_bits);
__IO_REG32_BIT(I2C2_TIMEOUTR,     0x40005814,__READ_WRITE ,__i2c_timeoutr_bits);
__IO_REG32_BIT(I2C2_ISR,     			0x40005818,__READ_WRITE ,__i2c_isr_bits);
__IO_REG32_BIT(I2C2_ICR,     			0x4000581C,__WRITE 			,__i2c_icr_bits);
__IO_REG32_BIT(I2C2_PECR,     		0x40005820,__READ 			,__i2c_pecr_bits);
__IO_REG32_BIT(I2C2_RXDR,     		0x40005824,__READ 			,__i2c_rxdr_bits);
__IO_REG32_BIT(I2C2_TXDR,         0x40005828,__READ_WRITE ,__i2c_txdr_bits);

/***************************************************************************
 **
 ** SPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI1_CR1,          0x40013000,__READ_WRITE ,__spi_cr1_bits);
__IO_REG32_BIT(SPI1_CR2,          0x40013004,__READ_WRITE ,__spi_cr2_bits);
__IO_REG32_BIT(SPI1_SR,           0x40013008,__READ_WRITE ,__spi_sr_bits);
__IO_REG32_BIT(SPI1_DR,           0x4001300C,__READ_WRITE ,__spi_dr_bits);
__IO_REG32_BIT(SPI1_CRCPR,        0x40013010,__READ_WRITE ,__spi_crcpr_bits);
__IO_REG32_BIT(SPI1_RXCRCR,       0x40013014,__READ       ,__spi_rxcrcr_bits);
__IO_REG32_BIT(SPI1_TXCRCR,       0x40013018,__READ       ,__spi_txcrcr_bits);
__IO_REG32_BIT(SPI1_I2SCFGR,      0x4001301C,__READ_WRITE	,__spi_i2scfgr_bits);
__IO_REG32_BIT(SPI1_I2SPR,      	0x40013020,__READ_WRITE	,__spi_i2spr_bits);

/***************************************************************************
 **
 ** SPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI2_CR1,          0x40003800,__READ_WRITE ,__spi_cr1_bits);
__IO_REG32_BIT(SPI2_CR2,          0x40003804,__READ_WRITE ,__spi_cr2_bits);
__IO_REG32_BIT(SPI2_SR,           0x40003808,__READ_WRITE ,__spi_sr_bits);
__IO_REG32_BIT(SPI2_DR,           0x4000380C,__READ_WRITE ,__spi_dr_bits);
__IO_REG32_BIT(SPI2_CRCPR,        0x40003810,__READ_WRITE ,__spi_crcpr_bits);
__IO_REG32_BIT(SPI2_RXCRCR,       0x40003814,__READ       ,__spi_rxcrcr_bits);
__IO_REG32_BIT(SPI2_TXCRCR,       0x40003818,__READ       ,__spi_txcrcr_bits);

/***************************************************************************
 **
 ** USART1
 **
 ***************************************************************************/
__IO_REG32_BIT(USART1_CR1,        0x40013800,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART1_CR2,        0x40013804,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART1_CR3,        0x40013808,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART1_BRR,        0x4001380C,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART1_GTPR,       0x40013810,__READ_WRITE ,__usart_gtpr_bits);
__IO_REG32_BIT(USART1_RTOR,       0x40013814,__READ_WRITE ,__usart_rtor_bits);
__IO_REG32_BIT(USART1_RQR,       	0x40013818,__READ_WRITE ,__usart_rqr_bits);
__IO_REG32_BIT(USART1_ISR,       	0x4001381C,__READ				,__usart_isr_bits);
__IO_REG32_BIT(USART1_ICR,       	0x40013820,__READ_WRITE ,__usart_icr_bits);
__IO_REG32_BIT(USART1_RDR,       	0x40013824,__READ				,__usart_rdr_bits);
__IO_REG32_BIT(USART1_TDR,       	0x40013828,__READ_WRITE ,__usart_tdr_bits);

/***************************************************************************
 **
 ** USART2
 **
 ***************************************************************************/
__IO_REG32_BIT(USART2_CR1,        0x40004400,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART2_CR2,        0x40004404,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART2_CR3,        0x40004408,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART2_BRR,        0x4000440C,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART2_GTPR,       0x40004410,__READ_WRITE ,__usart_gtpr_bits);
__IO_REG32_BIT(USART2_RTOR,       0x40004414,__READ_WRITE ,__usart_rtor_bits);
__IO_REG32_BIT(USART2_RQR,       	0x40004418,__READ_WRITE ,__usart_rqr_bits);
__IO_REG32_BIT(USART2_ISR,       	0x4000441C,__READ				,__usart_isr_bits);
__IO_REG32_BIT(USART2_ICR,       	0x40004420,__READ_WRITE ,__usart_icr_bits);
__IO_REG32_BIT(USART2_RDR,       	0x40004424,__READ				,__usart_rdr_bits);
__IO_REG32_BIT(USART2_TDR,       	0x40004428,__READ_WRITE ,__usart_tdr_bits);

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_ISR,           0x40012400,__READ_WRITE ,__adc_isr_bits);
__IO_REG32_BIT(ADC_IER,          	0x40012404,__READ_WRITE ,__adc_ier_bits);
__IO_REG32_BIT(ADC_CR,          	0x40012408,__READ_WRITE ,__adc_cr_bits);
__IO_REG32_BIT(ADC_CFGR1,        	0x4001240C,__READ_WRITE ,__adc_cfgr1_bits);
__IO_REG32_BIT(ADC_CFGR2,        	0x40012410,__READ_WRITE ,__adc_cfgr2_bits);
__IO_REG32_BIT(ADC_SMPR,        	0x40012414,__READ_WRITE ,__adc_smpr_bits);
__IO_REG32_BIT(ADC_TR,          	0x40012420,__READ_WRITE ,__adc_tr_bits);
__IO_REG32_BIT(ADC_CHSELR,        0x40012428,__READ_WRITE ,__adc_chselr_bits);
__IO_REG32_BIT(ADC_DR,           	0x40012440,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC_CCR,           0x40012708,__READ_WRITE ,__adc_ccr_bits);

/***************************************************************************
 **
 ** DAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DAC_CR,            0x40007400,__READ_WRITE ,__dac_cr_bits     );
__IO_REG32_BIT(DAC_SWTRIGR,       0x40007404,__WRITE      ,__dac_swtrigr_bits);
__IO_REG32_BIT(DAC_DHR12R1,       0x40007408,__READ_WRITE ,__dac_dhr12r1_bits);
__IO_REG32_BIT(DAC_DHR12L1,       0x4000740C,__READ_WRITE ,__dac_dhr12l1_bits);
__IO_REG32_BIT(DAC_DHR8R1,        0x40007410,__READ_WRITE ,__dac_dhr8r1_bits );
__IO_REG32_BIT(DAC_DHR8RD,        0x40007428,__READ_WRITE ,__dac_dhr8rd_bits );
__IO_REG32_BIT(DAC_DOR1,          0x4000742C,__READ       ,__dac_dor1_bits   );
__IO_REG32_BIT(DAC_SR,            0x40007434,__READ_WRITE ,__dac_sr_bits     );

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASH_ACR,         0x40022000,__READ_WRITE ,__flash_acr_bits);
__IO_REG32(    FLASH_KEYR,        0x40022004,__WRITE      );
__IO_REG32(    FLASH_OPTKEYR,     0x40022008,__WRITE      );
__IO_REG32_BIT(FLASH_SR,          0x4002200C,__READ_WRITE ,__flash_sr_bits);
__IO_REG32_BIT(FLASH_CR,          0x40022010,__READ_WRITE ,__flash_cr_bits);
__IO_REG32(    FLASH_AR,          0x40022014,__WRITE      );
__IO_REG32_BIT(FLASH_OBR,         0x4002201C,__READ       ,__flash_obr_bits);
__IO_REG32_BIT(FLASH_WRPR,        0x40022020,__READ       ,__flash_wrpr_bits);

/***************************************************************************
 **
 ** CRC
 **
 ***************************************************************************/
__IO_REG32(    CRC_DR,            0x40023000,__READ_WRITE );
__IO_REG32_BIT(CRC_IDR,           0x40023004,__READ_WRITE ,__crc_idr_bits);
__IO_REG32_BIT(CRC_CR,            0x40023008,__WRITE      ,__crc_cr_bits);
__IO_REG32(    CRC_INIT,          0x40023010,__READ_WRITE );

/***************************************************************************
 **
 ** COMP
 **
 ***************************************************************************/
__IO_REG32_BIT(COMP_CSR,     			0x4001001C,__READ_WRITE	,__comp_csr_bits);

/***************************************************************************
 **
 ** TSC
 **
 ***************************************************************************/
__IO_REG32_BIT(TSC_CR,     				0x40024000,__READ_WRITE	,__tsc_cr_bits);
__IO_REG32_BIT(TSC_IER,     			0x40024004,__READ_WRITE	,__tsc_ier_bits);
__IO_REG32_BIT(TSC_ICR,     			0x40024008,__READ_WRITE	,__tsc_icr_bits);
__IO_REG32_BIT(TSC_ISR,     			0x4002400C,__READ_WRITE	,__tsc_isr_bits);
__IO_REG32_BIT(TSC_IOHCR,     		0x40024010,__READ_WRITE	,__tsc_iohcr_bits);
__IO_REG32_BIT(TSC_IOASCR,     		0x40024018,__READ_WRITE	,__tsc_iohcr_bits);
__IO_REG32_BIT(TSC_IOSCR,     		0x40024020,__READ_WRITE	,__tsc_iohcr_bits);
__IO_REG32_BIT(TSC_IOCCR,     		0x40024028,__READ_WRITE	,__tsc_iohcr_bits);
__IO_REG32_BIT(TSC_IOGCSR,     		0x40024030,__READ_WRITE	,__tsc_iogcsr_bits);
__IO_REG32_BIT(TSC_IOG1CR,     		0x40024034,__READ_WRITE	,__tsc_iogcr_bits);
__IO_REG32_BIT(TSC_IOG2CR,     		0x40024038,__READ_WRITE	,__tsc_iogcr_bits);
__IO_REG32_BIT(TSC_IOG3CR,     		0x4002403C,__READ_WRITE	,__tsc_iogcr_bits);
__IO_REG32_BIT(TSC_IOG4CR,     		0x40024040,__READ_WRITE	,__tsc_iogcr_bits);
__IO_REG32_BIT(TSC_IOG5CR,     		0x40024044,__READ_WRITE	,__tsc_iogcr_bits);
__IO_REG32_BIT(TSC_IOG6CR,     		0x40024048,__READ_WRITE	,__tsc_iogcr_bits);

/***************************************************************************
 **
 ** CEC
 **
 ***************************************************************************/
__IO_REG32_BIT(CEC_CR,     				0x40007800,__READ_WRITE	,__cec_cr_bits);
__IO_REG32_BIT(CEC_CFGR,     			0x40007804,__READ_WRITE	,__cec_cfgr_bits);
__IO_REG32_BIT(CEC_TXDR,     			0x40007808,__WRITE			,__cec_txdr_bits);
__IO_REG32_BIT(CEC_RXDR,     			0x4000780C,__READ 			,__cec_rxdr_bits);
__IO_REG32_BIT(CEC_ISR,     			0x40007810,__READ_WRITE	,__cec_isr_bits);
__IO_REG32_BIT(CEC_IER,     			0x40007814,__READ_WRITE	,__cec_ier_bits);

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  STM32F051x8 Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             			  0          /* Main Stack                   */
#define RESETI                 			  1          /* Reset                        */
#define NMII                   			  2          /* Non-maskable Interrupt       */
#define HFI                    			  3          /* Hard Fault                   */
#define MMI                    			  4          /* Memory Management            */
#define BFI                    			  5          /* Bus Fault                    */
#define UFI                    			  6          /* Usage Fault                  */
#define SVCI                   			 11          /* SVCall                       */
#define DMI                    			 12          /* Debug Monitor                */
#define PSI                    			 14          /* PendSV                       */
#define STI                    			 15          /* SysTick                      */
#define NVIC_WWDG              			(16+0)       /* Window Watchdog interrupt    */
#define NVIC_PVD               			(16+1)       /* PVD through EXTI Line detection interrupt*/
#define NVIC_RTC               			(16+2)       /* RTC global interrupt         */
#define NVIC_FLASH             			(16+3)       /* Flash global interrupt       */
#define NVIC_RCC               			(16+4)       /* RCC global interrupt         */
#define NVIC_EXTI0_1           			(16+5)       /* EXTI Line0-1 interrupt       */
#define NVIC_EXTI2_3           			(16+6)       /* EXTI Line2-3 interrupt       */
#define NVIC_EXTI4_15          			(16+7)       /* EXTI Line4-15 interrupt      */
#define NVIC_TSC               			(16+8)       /* Touch sensing interrupt      */
#define NVIC_DMA_CH1           			(16+9)       /* DMA Channel1 global interrupt*/
#define NVIC_DMA_CH2_3         			(16+10)      /* DMA Channel2-3 global interrupt*/
#define NVIC_DMA_CH4_5         			(16+11)      /* DMA Channel4-5 global interrupt*/
#define NVIC_ADC_COM           			(16+12)      /* ADC and comparator 1 and 2 interrupts */
#define NVIC_TIM1_BRK_UP_TRG_COM    (16+13)      /* TIM1 Break, update, trigger and commutation interrupt */
#define NVIC_TIM1_CC     						(16+14)      /* TIM1 Capture Compare interrupt */
#define NVIC_TIM2		     						(16+15)      /* TIM2 global interrupt */
#define NVIC_TIM3		     						(16+16)      /* TIM3 global interrupt */
#define NVIC_TIM6_DAC				       	(16+17)      /* TIM6 global interrupt and DAC underrun interrupt */
#define NVIC_TIM14									(16+19)      /* TIM14 global interrupt */
#define NVIC_TIM15									(16+20)      /* TIM15 global interrupt */
#define NVIC_TIM16									(16+21)      /* TIM16 global interrupt */
#define NVIC_TIM17									(16+22)      /* TIM17 global interrupt */
#define NVIC_I2C1				            (16+23)      /* I2C1 global interrupt */
#define NVIC_I2C2				            (16+24)      /* I2C2 global interrupt */
#define NVIC_SPI1               		(16+25)      /* SPI1 global interrupt */
#define NVIC_SPI2               		(16+26)      /* SPI2 global interrupt */
#define NVIC_USART1             		(16+27)      /* USART1 global interrupt */
#define NVIC_USART2             		(16+28)      /* USART2 global interrupt */
#define NVIC_CEC                		(16+30)      /* CEC global interrupt */

#endif    /* __IOSTM32F051x8_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI                  				0x08
Interrupt1   = HardFault            				0x0C
Interrupt2   = MemManage            				0x10
Interrupt3   = BusFault             				0x14
Interrupt4   = UsageFault           				0x18
Interrupt5   = SVC                  				0x2C
Interrupt6   = DebugMon             				0x30
Interrupt7   = PendSV               				0x38
Interrupt8   = SysTick              				0x3C
Interrupt9   = NVIC_WWDG              			0x40
Interrupt10  = NVIC_PVD               			0x44
Interrupt11  = NVIC_RTC               			0x48
Interrupt12  = NVIC_FLASH             			0x4C
Interrupt13  = NVIC_RCC               			0x50
Interrupt14  = NVIC_EXTI0_1           			0x54
Interrupt15  = NVIC_EXTI2_3           			0x58
Interrupt16  = NVIC_EXTI4_15          			0x5C
Interrupt17  = NVIC_TSC               			0x60
Interrupt18  = NVIC_DMA_CH1           			0x64
Interrupt19  = NVIC_DMA_CH2_3         			0x68
Interrupt20  = NVIC_DMA_CH4_5         			0x6C
Interrupt21  = NVIC_ADC_COM           			0x70
Interrupt22  = NVIC_TIM1_BRK_UP_TRG_COM     0x74
Interrupt23  = NVIC_TIM1_CC     						0x78
Interrupt24  = NVIC_TIM2		     						0x7C
Interrupt25  = NVIC_TIM3		     						0x80
Interrupt26  = NVIC_TIM6_DAC				       	0x84
Interrupt27  = NVIC_TIM14									  0x8C
Interrupt28  = NVIC_TIM15									  0x90
Interrupt29  = NVIC_TIM16									  0x94
Interrupt30  = NVIC_TIM17									  0x98
Interrupt31  = NVIC_I2C1				            0x9C
Interrupt32  = NVIC_I2C2				            0xA0
Interrupt33  = NVIC_SPI1               		  0xA4
Interrupt34  = NVIC_SPI2               		  0xA8
Interrupt35  = NVIC_USART1             		  0xAC
Interrupt36  = NVIC_USART2             		  0xB0
Interrupt37  = NVIC_CEC                		  0xB8

###DDF-INTERRUPT-END###*/
