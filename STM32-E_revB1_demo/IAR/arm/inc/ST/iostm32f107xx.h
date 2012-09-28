/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST STM32F107xx
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2009
 **
 **    $Revision: 52044 $
 **
 ***************************************************************************/

#ifndef __IOSTM32F107xx_H
#define __IOSTM32F107xx_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   STM32F107xx SPECIAL FUNCTION REGISTERS
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
  __REG32  EWUP           : 1;
  __REG32                 :23;
} __pwr_csr_bits;

/* Clock control register (RCC_CR) */
typedef struct {
  __REG32  HSION          : 1;
  __REG32  HSIRDY         : 1;
  __REG32                 : 1;
  __REG32  HSI_TRIM       : 5;
  __REG32  HSI_CAL        : 8;
  __REG32  HSEON          : 1;
  __REG32  HSERDY         : 1;
  __REG32  HSEBYP         : 1;
  __REG32  CSSON          : 1;
  __REG32                 : 4;
  __REG32  PLLON          : 1;
  __REG32  PLLRDY         : 1;
  __REG32  PLL2ON         : 1;
  __REG32  PLL2RDY        : 1;
  __REG32  PLL3ON         : 1;
  __REG32  PLL3RDY        : 1;
  __REG32                 : 2;
} __rcc_cr_bits;

/* Clock configuration register (RCC_CFGR) */
typedef struct {
  __REG32  SW             : 2;
  __REG32  SWS            : 2;
  __REG32  HPRE           : 4;
  __REG32  PPRE1          : 3;
  __REG32  PPRE2          : 3;
  __REG32  ADC_PRE        : 2;
  __REG32  PLLSRC         : 1;
  __REG32  PLLXTPRE       : 1;
  __REG32  PLLMUL         : 4;
  __REG32  OTGFSPRE       : 1;
  __REG32                 : 1;
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
  __REG32  PLL2RDYF       : 1;
  __REG32  PLL3RDYF       : 1;
  __REG32  CSSF           : 1;
  __REG32  LSIRDYIE       : 1;
  __REG32  LSERDYIE       : 1;
  __REG32  HSIRDYIE       : 1;
  __REG32  HSERDYIE       : 1;
  __REG32  PLLRDYIE       : 1;
  __REG32  PLL2RDYIE      : 1;
  __REG32  PLL3RDYIE      : 1;
  __REG32                 : 1;
  __REG32  LSIRDYC        : 1;
  __REG32  LSERDYC        : 1;
  __REG32  HSIRDYC        : 1;
  __REG32  HSERDYC        : 1;
  __REG32  PLLRDYC        : 1;
  __REG32  PLL2RDYC       : 1;
  __REG32  PLL3RDYC       : 1;
  __REG32  CSSC           : 1;
  __REG32                 : 8;
} __rcc_cir_bits;

/* APB2 Peripheral reset register (RCC_APB2RSTR) */
typedef struct {
  __REG32  AFIORST        : 1;
  __REG32                 : 1;
  __REG32  IOPARST        : 1;
  __REG32  IOPBRST        : 1;
  __REG32  IOPCRST        : 1;
  __REG32  IOPDRST        : 1;
  __REG32  IOPERST        : 1;
  __REG32                 : 2;
  __REG32  ADC1RST        : 1;
  __REG32  ADC2RST        : 1;
  __REG32  TIM1RST        : 1;
  __REG32  SPI1RST        : 1;
  __REG32                 : 1;
  __REG32  USART1RST      : 1;
  __REG32                 :17;
} __rcc_apb2rstr_bits;

/* APB1 Peripheral reset register (RCC_APB1RSTR) */
typedef struct {
  __REG32  TIM2RST        : 1;
  __REG32  TIM3RST        : 1;
  __REG32  TIM4RST        : 1;
  __REG32  TIM5RST        : 1;
  __REG32  TIM6RST        : 1;
  __REG32  TIM7RST        : 1;
  __REG32                 : 5;
  __REG32  WWDGRST        : 1;
  __REG32                 : 2;
  __REG32  SPI2RST        : 1;
  __REG32  SPI3RST        : 1;
  __REG32                 : 1;
  __REG32  USART2RST      : 1;
  __REG32  USART3RST      : 1;
  __REG32  UART4RST       : 1;
  __REG32  UART5RST       : 1;
  __REG32  I2C1RST        : 1;
  __REG32  I2C2RST        : 1;
  __REG32                 : 2;
  __REG32  CAN1RST        : 1;
  __REG32  CAN2RST        : 1;
  __REG32  BKPRST         : 1;
  __REG32  PWRRST         : 1;
  __REG32  DACRST         : 1;
  __REG32                 : 2;
} __rcc_apb1rstr_bits;

/* AHB Peripheral Clock enable register (RCC_AHBENR) */
typedef struct {
  __REG32  DMA1E          : 1;
  __REG32  DMA2E          : 1;
  __REG32  SRAMEN         : 1;
  __REG32                 : 1;
  __REG32  FLITFEN        : 1;
  __REG32                 : 1;
  __REG32  CRCEN          : 1;
  __REG32                 : 5;
  __REG32  OTGFSEN        : 1;
  __REG32                 : 1;
  __REG32  ETHMACEN       : 1;
  __REG32  ETHMACTXEN     : 1;
  __REG32  ETHMACRXEN     : 1;
  __REG32                 :15;
} __rcc_ahbenr_bits;

/* APB2 Peripheral Clock enable register (RCC_APB2ENR) */
typedef struct {
  __REG32  AFIOEN        : 1;
  __REG32                : 1;
  __REG32  IOPAEN        : 1;
  __REG32  IOPBEN        : 1;
  __REG32  IOPCEN        : 1;
  __REG32  IOPDEN        : 1;
  __REG32  IOPEEN        : 1;
  __REG32                : 2;
  __REG32  ADC1EN        : 1;
  __REG32  ADC2EN        : 1;
  __REG32  TIM1EN        : 1;
  __REG32  SPI1EN        : 1;
  __REG32                : 1;
  __REG32  USART1EN      : 1;
  __REG32                :17;
} __rcc_apb2enr_bits;

/* APB1 Peripheral Clock enable register (RCC_APB1ENR) */
typedef struct {
  __REG32  TIM2EN        : 1;
  __REG32  TIM3EN        : 1;
  __REG32  TIM4EN        : 1;
  __REG32  TIM5EN        : 1;
  __REG32  TIM6EN        : 1;
  __REG32  TIM7EN        : 1;
  __REG32                : 5;
  __REG32  WWDGEN        : 1;
  __REG32                : 2;
  __REG32  SPI2EN        : 1;
  __REG32  SPI3EN        : 1;
  __REG32                : 1;
  __REG32  USART2EN      : 1;
  __REG32  USART3EN      : 1;
  __REG32  UART4EN       : 1;
  __REG32  UART5EN       : 1;
  __REG32  I2C1EN        : 1;
  __REG32  I2C2EN        : 1;
  __REG32                : 2;
  __REG32  CAN1EN        : 1;
  __REG32  CAN2EN        : 1;
  __REG32  BKPEN         : 1;
  __REG32  PWREN         : 1;
  __REG32  DACEN         : 1;
  __REG32                : 2;
} __rcc_apb1enr_bits;

/* Backup domain control register (RCC_BDCR) */
typedef struct {
  __REG32  LSEON          : 1;
  __REG32  LSERDY         : 1;
  __REG32  LSEBYP         : 1;
  __REG32                 : 5;
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
  __REG32                 : 1;
  __REG32  PINRSTF        : 1;
  __REG32  PORRSTF        : 1;
  __REG32  SFTRSTF        : 1;
  __REG32  IWDGRSTF       : 1;
  __REG32  WWDGRSTF       : 1;
  __REG32  LPWRRSTF       : 1;
} __rcc_csr_bits;

/* AHB Peripheral Clock reset register (RCC_AHBRSTR) */
typedef struct {
  __REG32                 :12;
  __REG32  OTGFSRST       : 1;
  __REG32                 : 1;
  __REG32  ETHMACRST      : 1;
  __REG32                 :17;
} __rcc_ahbrstr_bits;

/* Clock configuration register2 (RCC_CFGR2) */
typedef struct {
  __REG32  PREDIV1        : 4;
  __REG32  PREDIV2        : 4;
  __REG32  PLL2MUL        : 4;
  __REG32  PLL3MUL        : 4;
  __REG32  PREDIV1SRC     : 1;
  __REG32  I2S2SRC        : 1;
  __REG32  I2S3SRC        : 1;
  __REG32                 :13;
} __rcc_cfgr2_bits;

/* Port configuration register low (GPIOx_CRL) (x=A..G) */
typedef struct {
  __REG32  MODE0          : 2;
  __REG32  CNF0           : 2;
  __REG32  MODE1          : 2;
  __REG32  CNF1           : 2;
  __REG32  MODE2          : 2;
  __REG32  CNF2           : 2;
  __REG32  MODE3          : 2;
  __REG32  CNF3           : 2;
  __REG32  MODE4          : 2;
  __REG32  CNF4           : 2;
  __REG32  MODE5          : 2;
  __REG32  CNF5           : 2;
  __REG32  MODE6          : 2;
  __REG32  CNF6           : 2;
  __REG32  MODE7          : 2;
  __REG32  CNF7           : 2;
} __gpio_crl_bits;

/* Port configuration register high (GPIOx_CRH) (x=A..G) */
typedef struct {
  __REG32  MODE8          : 2;
  __REG32  CNF8           : 2;
  __REG32  MODE9          : 2;
  __REG32  CNF9           : 2;
  __REG32  MODE10         : 2;
  __REG32  CNF10          : 2;
  __REG32  MODE11         : 2;
  __REG32  CNF11          : 2;
  __REG32  MODE12         : 2;
  __REG32  CNF12          : 2;
  __REG32  MODE13         : 2;
  __REG32  CNF13          : 2;
  __REG32  MODE14         : 2;
  __REG32  CNF14          : 2;
  __REG32  MODE15         : 2;
  __REG32  CNF15          : 2;
} __gpio_crh_bits;

/* Port input data register (GPIOx_IDR) (x=A..G) */
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

/* Port output data register (GPIOx_ODR) (x=A..G) */
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

/* Port bit set/reset register (GPIOx_BSRR) (x=A..G) */
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

/* Port bit reset register (GPIOx_BRR) (x=A..G) */
typedef struct {
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
  __REG32                 :16;
} __gpio_brr_bits;

/* Port configuration lock register (GPIOx_LCKR) (x=A..G) */
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

/* Event control register (AFIO_EVCR) */
typedef struct {
  __REG32  PIN            : 4;
  __REG32  PORT           : 3;
  __REG32  EVOE           : 1;
  __REG32                 :24;
} __afio_evcr_bits;

/* AF remap and debug I/O configuration register (AFIO_MAPR) */
typedef struct {
  __REG32  SPI1_REMAP         : 1;
  __REG32  I2C1_REMAP         : 1;
  __REG32  USART1_REMAP       : 1;
  __REG32  USART2_REMAP       : 1;
  __REG32  USART3_REMAP       : 2;
  __REG32  TIM1_REMAP         : 2;
  __REG32  TIM2_REMAP         : 2;
  __REG32  TIM3_REMAP         : 2;
  __REG32  TIM4_REMAP         : 1;
  __REG32  CAN1_REMAP         : 2;
  __REG32  PD01_REMAP         : 1;
  __REG32  TIM5CH4_IREMAP     : 1;
  __REG32                     : 4;
  __REG32  ETH_REMAP          : 1;
  __REG32  CAN2_REMAP         : 1;
  __REG32  MII_RMII_SEL       : 1;
  __REG32  SWJ_CFG            : 3;
  __REG32                     : 1;
  __REG32  SPI3_REMAP         : 1;
  __REG32  TIM2ITR1_IREMAP    : 1;
  __REG32  PTP_PPS_REMAP      : 1;
  __REG32                     : 1;
} __afio_mapr_bits;

/* External interrupt configuration register 1 (AFIO_EXTICR1) */
typedef struct {
  __REG32  EXTI0          : 4;
  __REG32  EXTI1          : 4;
  __REG32  EXTI2          : 4;
  __REG32  EXTI3          : 4;
  __REG32                 :16;
} __afio_exticr1_bits;

/* External interrupt configuration register 2 (AFIO_EXTICR2) */
typedef struct {
  __REG32  EXTI4          : 4;
  __REG32  EXTI5          : 4;
  __REG32  EXTI6          : 4;
  __REG32  EXTI7          : 4;
  __REG32                 :16;
} __afio_exticr2_bits;

/* External interrupt configuration register 3 (AFIO_EXTICR3) */
typedef struct {
  __REG32  EXTI8          : 4;
  __REG32  EXTI9          : 4;
  __REG32  EXTI10         : 4;
  __REG32  EXTI11         : 4;
  __REG32                 :16;
} __afio_exticr3_bits;

/* External interrupt configuration register 4 (AFIO_EXTICR4) */
typedef struct {
  __REG32  EXTI12         : 4;
  __REG32  EXTI13         : 4;
  __REG32  EXTI14         : 4;
  __REG32  EXTI15         : 4;
  __REG32                 :16;
} __afio_exticr4_bits;

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
  __REG32                 :12;
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
  __REG32                 :12;
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
  __REG32  TR18           : 1;
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
  __REG32  TR18           : 1;
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
  __REG32  SWIER18        : 1;
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
  __REG32  PR18           : 1;
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
  __REG32  GIF6           : 1;
  __REG32  TCIF6          : 1;
  __REG32  HTIF6          : 1;
  __REG32  TEIF6          : 1;
  __REG32  GIF7           : 1;
  __REG32  TCIF7          : 1;
  __REG32  HTIF7          : 1;
  __REG32  TEIF7          : 1;
  __REG32                 : 4;
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
  __REG32  CGIF6          : 1;
  __REG32  CTCIF6         : 1;
  __REG32  CHTIF6         : 1;
  __REG32  CTEIF6         : 1;
  __REG32  CGIF7          : 1;
  __REG32  CTCIF7         : 1;
  __REG32  CHTIF7         : 1;
  __REG32  CTEIF7         : 1;
  __REG32                 : 4;
} __dma_ifcr_bits;
/* DMA2 interrupt status register (DMA_ISR) */
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
} __dma2_isr_bits;

/* DMA2 interrupt flag clear register (DMA_IFCR) */
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
} __dma2_ifcr_bits;

/* DMA channel x configuration register (DMA_CCRx) (x = 1 ..7) */
/* DMA2 channel x configuration register (DMA_CCRx) (x = 1 ..5) */
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

/* DMA channel x number of data register (DMA_CNDTRx) (x = 1 ..7) */
/* DMA2 channel x number of data register (DMA_CNDTRx) (x = 1 ..5) */
typedef struct {
  __REG32  NDT            :16;
  __REG32                 :16;
} __dma_cndtr_bits;

/* RTC control register High (RTC_CRH) */
typedef struct {
  __REG16 SECIE           : 1;
  __REG16 ALRIE           : 1;
  __REG16 OWIE            : 1;
  __REG16                 :13;
} __rtccrh_bits;

/* RTC Control Register Low (RTC_CRL) */
typedef struct {
  __REG16 SECF            : 1;
  __REG16 ALRF            : 1;
  __REG16 OWF             : 1;
  __REG16 RSF             : 1;
  __REG16 CNF             : 1;
  __REG16 RTOFF           : 1;
  __REG16                 :10;
} __rtccrl_bits;

/* RTC Prescaler Load Register High (RTC_PRLH) */
typedef struct {
  __REG16 PRL             : 4;
  __REG16                 :12;
} __rtcprlh_bits;

/* RTC Prescaler Divider Register High (RTC_DIVH) */
typedef struct {
  __REG16 RTC_DIV         : 4;
  __REG16                 :12;
} __rtcdivh_bits;

/* RTC clock calibration register (BKP_RTCCR) */
typedef struct {
  __REG16 CAL             : 7;
  __REG16 CCO             : 1;
  __REG16 ASOE            : 1;
  __REG16 ASOS            : 1;
  __REG16                 : 6;
} __bkp_rtccr_bits;

/* Backup control register (BKP_CR) */
typedef struct {
  __REG16 TPE             : 1;
  __REG16 TPAL            : 1;
  __REG16                 :14;
} __bkp_cr_bits;

/* Backup control/status register (BKP_CSR) */
typedef struct {
  __REG16 CTE             : 1;
  __REG16 CTI             : 1;
  __REG16 TPIE            : 1;
  __REG16                 : 5;
  __REG16 TEF             : 1;
  __REG16 TIF             : 1;
  __REG16                 : 6;
} __bkp_csr_bits;

/* Prescaler register (IWDG_PR) */
typedef struct {
  __REG16 PR              : 3;
  __REG16                 :13;
} __iwdg_pr_bits;

/* Reload register (IWDG_RLR) */
typedef struct {
  __REG16 RL              :12;
  __REG16                 : 4;
} __iwdg_rlr_bits;

/* Status register (IWDG_SR) */
typedef struct {
  __REG16 PVU             : 1;
  __REG16 RVU             : 1;
  __REG16                 :14;
} __iwdg_sr_bits;

/* Control Register (WWDG_CR) */
typedef struct {
  __REG16 T               : 7;
  __REG16 WDGA            : 1;
  __REG16                 : 8;
} __wwdg_cr_bits;

/* Configuration register (WWDG_CFR) */
typedef struct {
  __REG16 W               : 7;
  __REG16 WDGTB           : 2;
  __REG16 EWI             : 1;
  __REG16                 : 6;
} __wwdg_cfr_bits;

/* Status register (WWDG_SR) */
typedef struct {
  __REG16 EWIF            : 1;
  __REG16                 :15;
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

/* Capture/compare register (TIM1_CCR) */
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
  __REG32                 : 1;
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
  __REG32                 : 2;
  __REG32 CC2E            : 1;
  __REG32 CC2P            : 1;
  __REG32                 : 2;
  __REG32 CC3E            : 1;
  __REG32 CC3P            : 1;
  __REG32                 : 2;
  __REG32 CC4E            : 1;
  __REG32 CC4P            : 1;
  __REG32                 :18;
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

/* Control register 1 (TIM6_CR1) */
/* Control register 1 (TIM7_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32                 : 1;
  __REG32                 : 2;
  __REG32 ARPE            : 1;
  __REG32                 : 2;
  __REG32                 :22;
} __tim6_cr1_bits;

/* Control register 2 (TIM6_CR2) */
/* Control register 2 (TIM7_CR2) */
typedef struct {
  __REG32                 : 3;
  __REG32                 : 1;
  __REG32 MMS             : 3;
  __REG32                 : 1;
  __REG32                 :24;
} __tim6_cr2_bits;

/* DMA/Interrupt enable register (TIM6_DIER) */
/* DMA/Interrupt enable register (TIM7_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32                 : 7;
  __REG32 UDE             : 1;
  __REG32                 :23;
} __tim6_dier_bits;

/* Status register (TIM6_SR) */
/* Status register (TIM7_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32                 :31;
} __tim6_sr_bits;

/* Event generation register (TIM6_EGR) */
/* Event generation register (TIM7_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32                 :31;
} __tim6_egr_bits;

/* Counter (TIM6_CNT) */
/* Counter (TIM7_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim6_cnt_bits;

/* Prescaler (TIM6_PSC) */
/* Prescaler (TIM7_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim6_psc_bits;

/* Auto-reload register (TIM6_ARR) */
/* Auto-reload register (TIM7_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim6_arr_bits;


/* CAN master control register (CAN_MCR) */
typedef struct {
  __REG32 INRQ            : 1;
  __REG32 SLEEP           : 1;
  __REG32 TXFP            : 1;
  __REG32 RFLM            : 1;
  __REG32 NART            : 1;
  __REG32 AWUM            : 1;
  __REG32 ABOM            : 1;
  __REG32 TTCM            : 1;
  __REG32                 : 7;
  __REG32 RESET           : 1;
  __REG32                 :16;
} __can_mcr_bits;

/* CAN master status register (CAN_MSR) */
typedef struct {
  __REG32 INAK            : 1;
  __REG32 SLAK            : 1;
  __REG32 ERRI            : 1;
  __REG32 WKUI            : 1;
  __REG32 SLAKI           : 1;
  __REG32                 : 3;
  __REG32 TXM             : 1;
  __REG32 RXM             : 1;
  __REG32 SAMP            : 1;
  __REG32 RX              : 1;
  __REG32                 :20;
} __can_msr_bits;

/* CAN transmit status register (CAN_TSR) */
typedef struct {
  __REG32 RQCP0           : 1;
  __REG32 TXOK0           : 1;
  __REG32 ALST0           : 1;
  __REG32 TERR0           : 1;
  __REG32                 : 3;
  __REG32 ABRQ0           : 1;
  __REG32 RQCP1           : 1;
  __REG32 TXOK1           : 1;
  __REG32 ALST1           : 1;
  __REG32 TERR1           : 1;
  __REG32                 : 3;
  __REG32 ABRQ1           : 1;
  __REG32 RQCP2           : 1;
  __REG32 TXOK2           : 1;
  __REG32 ALST2           : 1;
  __REG32 TERR2           : 1;
  __REG32                 : 3;
  __REG32 ABRQ2           : 1;
  __REG32 CODE            : 2;
  __REG32 TME0            : 1;
  __REG32 TME1            : 1;
  __REG32 TME2            : 1;
  __REG32 LOW0            : 1;
  __REG32 LOW1            : 1;
  __REG32 LOW2            : 1;
} __can_tsr_bits;

/* CAN receive FIFO 0/1 register (CAN_RF0R) */
typedef struct {
  __REG32 FMP             : 2;
  __REG32                 : 1;
  __REG32 FULL            : 1;
  __REG32 FOVR            : 1;
  __REG32 RFOM            : 1;
  __REG32                 :26;
} __can_rfr_bits;

/* CAN interrupt enable register (CAN_IER) */
typedef struct {
  __REG32 TMEIE           : 1;
  __REG32 FMPIE0          : 1;
  __REG32 FFIE0           : 1;
  __REG32 FOVIE0          : 1;
  __REG32 FMPIE1          : 1;
  __REG32 FFIE1           : 1;
  __REG32 FOVIE1          : 1;
  __REG32                 : 1;
  __REG32 EWGIE           : 1;
  __REG32 EPVIE           : 1;
  __REG32 BOFIE           : 1;
  __REG32 LECIE           : 1;
  __REG32                 : 3;
  __REG32 ERRIE           : 1;
  __REG32 WKUIE           : 1;
  __REG32 SLKIE           : 1;
  __REG32                 :14;
} __can_ier_bits;

/* CAN error status register (CAN_ESR) */
typedef struct {
  __REG32 EWGF            : 1;
  __REG32 EPVF            : 1;
  __REG32 BOFF            : 1;
  __REG32                 : 1;
  __REG32 LEC             : 3;
  __REG32                 : 9;
  __REG32 TEC             : 8;
  __REG32 REC             : 8;
} __can_esr_bits;

/* CAN bit timing register (CAN_BTR) */
typedef struct {
  __REG32 BRP             :10;
  __REG32                 : 6;
  __REG32 TS1             : 4;
  __REG32 TS2             : 3;
  __REG32                 : 1;
  __REG32 SJW             : 2;
  __REG32                 : 4;
  __REG32 LBKM            : 1;
  __REG32 SILM            : 1;
} __can_btr_bits;

/* TX mailbox identifier register (CAN_TIxR) (x=0..2) */
typedef union {
  /*CANx_TIyR*/
  struct{
    __REG32 TXRQ            : 1;
    __REG32 RTR             : 1;
    __REG32 IDE             : 1;
    __REG32                 :18;
    __REG32 STID            :11;
  };
  /*CANx_TIyR  */
  struct{
    __REG32 _TXRQ            : 1;
    __REG32 _RTR             : 1;
    __REG32 _IDE             : 1;
    __REG32 _EXID            :29;
  };
} __can_tir_bits;

/* Mailbox data length control and time stamp register (CAN_TDTxR) (x=0..2) */
typedef struct {
  __REG32 DLC             : 4;
  __REG32                 : 4;
  __REG32 TGT             : 1;
  __REG32                 : 7;
  __REG32 TIME            :16;
} __can_tdtr_bits;

/* Mailbox data low register (CAN_TDLxR) (x=0..2) */
typedef struct {
  __REG32 DATA0           : 8;
  __REG32 DATA1           : 8;
  __REG32 DATA2           : 8;
  __REG32 DATA3           : 8;
} __can_tdlr_bits;

/* Mailbox data high register (CAN_TDHxR) (x=0..2) */
typedef struct {
  __REG32 DATA4           : 8;
  __REG32 DATA5           : 8;
  __REG32 DATA6           : 8;
  __REG32 DATA7           : 8;
} __can_tdhr_bits;

/* Rx FIFO mailbox identifier register (CAN_RIxR) (x=0..1) */
typedef union {
  /*CANx_RIyR*/
  struct{
    __REG32                 : 1;
    __REG32 RTR             : 1;
    __REG32 IDE             : 1;
    __REG32                 :18;
    __REG32 STID            :11;
  };
  /*CANx_RIyR*/
  struct{
    __REG32                 : 1;
    __REG32 _RTR            : 1;
    __REG32 _IDE            : 1;
    __REG32 _EXID           :29;
  }; 
} __can_rir_bits;

/* Receive FIFO mailbox data length control and time stamp register (CAN_RDTxR) (x=0..1) */
typedef struct {
  __REG32 DLC             : 4;
  __REG32                 : 4;
  __REG32 FMI             : 8;
  __REG32 TIME            :16;
} __can_rdtr_bits;

/* Receive FIFO mailbox data low register (CAN_RDLxR) (x=0..1) */
typedef struct {
  __REG32 DATA0           : 8;
  __REG32 DATA1           : 8;
  __REG32 DATA2           : 8;
  __REG32 DATA3           : 8;
} __can_rdlr_bits;

/* Receive FIFO mailbox data high register (CAN_RDHxR) (x=0..1) */
typedef struct {
  __REG32 DATA4           : 8;
  __REG32 DATA5           : 8;
  __REG32 DATA6           : 8;
  __REG32 DATA7           : 8;
} __can_rdhr_bits;

/* CAN filter master register (CAN_FMR) */
typedef struct {
  __REG32 FINIT           : 1;
  __REG32                 : 7;
  __REG32 CAN2SB          : 6;
  __REG32                 :18;
} __can_fmr_bits;

/* CAN filter mode register (CAN_FM1R) */
typedef struct {
  __REG32 FBM0            : 1;
  __REG32 FBM1            : 1;
  __REG32 FBM2            : 1;
  __REG32 FBM3            : 1;
  __REG32 FBM4            : 1;
  __REG32 FBM5            : 1;
  __REG32 FBM6            : 1;
  __REG32 FBM7            : 1;
  __REG32 FBM8            : 1;
  __REG32 FBM9            : 1;
  __REG32 FBM10           : 1;
  __REG32 FBM11           : 1;
  __REG32 FBM12           : 1;
  __REG32 FBM13           : 1;
  __REG32 FBM14           : 1;
  __REG32 FBM15           : 1;
  __REG32 FBM16           : 1;
  __REG32 FBM17           : 1;
  __REG32 FBM18           : 1;
  __REG32 FBM19           : 1;
  __REG32 FBM20           : 1;
  __REG32 FBM21           : 1;
  __REG32 FBM22           : 1;
  __REG32 FBM23           : 1;
  __REG32 FBM24           : 1;
  __REG32 FBM25           : 1;
  __REG32 FBM26           : 1;
  __REG32 FBM27           : 1;
  __REG32                 : 4;
} __can_fm1r_bits;

/* CAN filter scale register (CAN_FS1R) */
typedef struct {
  __REG32 FSC0            : 1;
  __REG32 FSC1            : 1;
  __REG32 FSC2            : 1;
  __REG32 FSC3            : 1;
  __REG32 FSC4            : 1;
  __REG32 FSC5            : 1;
  __REG32 FSC6            : 1;
  __REG32 FSC7            : 1;
  __REG32 FSC8            : 1;
  __REG32 FSC9            : 1;
  __REG32 FSC10           : 1;
  __REG32 FSC11           : 1;
  __REG32 FSC12           : 1;
  __REG32 FSC13           : 1;
  __REG32 FSC14           : 1;
  __REG32 FSC15           : 1;
  __REG32 FSC16           : 1;
  __REG32 FSC17           : 1;
  __REG32 FSC18           : 1;
  __REG32 FSC19           : 1;
  __REG32 FSC20           : 1;
  __REG32 FSC21           : 1;
  __REG32 FSC22           : 1;
  __REG32 FSC23           : 1;
  __REG32 FSC24           : 1;
  __REG32 FSC25           : 1;
  __REG32 FSC26           : 1;
  __REG32 FSC27           : 1;
  __REG32                 : 4;
} __can_fs1r_bits;

/* CAN filter FIFO assignment register (CAN_FFA1R) */
typedef struct {
  __REG32 FFA0            : 1;
  __REG32 FFA1            : 1;
  __REG32 FFA2            : 1;
  __REG32 FFA3            : 1;
  __REG32 FFA4            : 1;
  __REG32 FFA5            : 1;
  __REG32 FFA6            : 1;
  __REG32 FFA7            : 1;
  __REG32 FFA8            : 1;
  __REG32 FFA9            : 1;
  __REG32 FFA10           : 1;
  __REG32 FFA11           : 1;
  __REG32 FFA12           : 1;
  __REG32 FFA13           : 1;
  __REG32 FFA14           : 1;
  __REG32 FFA15           : 1;
  __REG32 FFA16           : 1;
  __REG32 FFA17           : 1;
  __REG32 FFA18           : 1;
  __REG32 FFA19           : 1;
  __REG32 FFA20           : 1;
  __REG32 FFA21           : 1;
  __REG32 FFA22           : 1;
  __REG32 FFA23           : 1;
  __REG32 FFA24           : 1;
  __REG32 FFA25           : 1;
  __REG32 FFA26           : 1;
  __REG32 FFA27           : 1;
  __REG32                 : 4;
} __can_ffa1r_bits;

/* CAN filter activation register (CAN_FA1R) */
typedef struct {
  __REG32 FACT0           : 1;
  __REG32 FACT1           : 1;
  __REG32 FACT2           : 1;
  __REG32 FACT3           : 1;
  __REG32 FACT4           : 1;
  __REG32 FACT5           : 1;
  __REG32 FACT6           : 1;
  __REG32 FACT7           : 1;
  __REG32 FACT8           : 1;
  __REG32 FACT9           : 1;
  __REG32 FACT10          : 1;
  __REG32 FACT11          : 1;
  __REG32 FACT12          : 1;
  __REG32 FACT13          : 1;
  __REG32 FACT14          : 1;
  __REG32 FACT15          : 1;
  __REG32 FACT16          : 1;
  __REG32 FACT17          : 1;
  __REG32 FACT18          : 1;
  __REG32 FACT19          : 1;
  __REG32 FACT20          : 1;
  __REG32 FACT21          : 1;
  __REG32 FACT22          : 1;
  __REG32 FACT23          : 1;
  __REG32 FACT24          : 1;
  __REG32 FACT25          : 1;
  __REG32 FACT26          : 1;
  __REG32 FACT27          : 1;
  __REG32                 : 4;
} __can_fa1r_bits;

/* Filter bank x registers (CAN_FxR[1:0]) (x=0..27) */
typedef struct {
  __REG32 FB0             : 1;
  __REG32 FB1             : 1;
  __REG32 FB2             : 1;
  __REG32 FB3             : 1;
  __REG32 FB4             : 1;
  __REG32 FB5             : 1;
  __REG32 FB6             : 1;
  __REG32 FB7             : 1;
  __REG32 FB8             : 1;
  __REG32 FB9             : 1;
  __REG32 FB10            : 1;
  __REG32 FB11            : 1;
  __REG32 FB12            : 1;
  __REG32 FB13            : 1;
  __REG32 FB14            : 1;
  __REG32 FB15            : 1;
  __REG32 FB16            : 1;
  __REG32 FB17            : 1;
  __REG32 FB18            : 1;
  __REG32 FB19            : 1;
  __REG32 FB20            : 1;
  __REG32 FB21            : 1;
  __REG32 FB22            : 1;
  __REG32 FB23            : 1;
  __REG32 FB24            : 1;
  __REG32 FB25            : 1;
  __REG32 FB26            : 1;
  __REG32 FB27            : 1;
  __REG32 FB28            : 1;
  __REG32 FB29            : 1;
  __REG32 FB30            : 1;
  __REG32 FB31            : 1;
} __can_fr_bits;

/* Control register 1(I2C_CR1) */
typedef struct {
  __REG32 PE              : 1;
  __REG32 SMBUS           : 1;
  __REG32                 : 1;
  __REG32 SMBTYPE         : 1;
  __REG32 ENARP           : 1;
  __REG32 ENPEC           : 1;
  __REG32 ENGC            : 1;
  __REG32 NOSTRETCH       : 1;
  __REG32 START           : 1;
  __REG32 STOP            : 1;
  __REG32 ACK             : 1;
  __REG32 POS             : 1;
  __REG32 PEC             : 1;
  __REG32 ALERT           : 1;
  __REG32                 : 1;
  __REG32 SWRST           : 1;
  __REG32                 :16;
} __i2c_cr1_bits;

/* Control register 2 (I2C_CR2) */
typedef struct {
  __REG32 FREQ            : 6;
  __REG32                 : 2;
  __REG32 ITERREN         : 1;
  __REG32 ITEVTEN         : 1;
  __REG32 ITBUFEN         : 1;
  __REG32 DMAEN           : 1;
  __REG32 LAST            : 1;
  __REG32                 :19;
} __i2c_cr2_bits;

/* Own address register 1 (I2C_OAR1) */
typedef struct {
  __REG32 ADD0            : 1;
  __REG32 ADD             : 9;
  __REG32                 : 5;
  __REG32 ADDMODE         : 1;
  __REG32                 :16;
} __i2c_oar1_bits;

/* Own address register 2 (I2C_OAR2) */
typedef struct {
  __REG32 ENDUAL          : 1;
  __REG32 ADD2            : 7;
  __REG32                 :24;
} __i2c_oar2_bits;

/* Data register (I2C_DR) */
typedef struct {
  __REG32 DR              : 8;
  __REG32                 :24;
} __i2c_dr_bits;

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
  __REG32 DFF             : 1;
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
  __REG32                 : 2;
  __REG32 ERRIE           : 1;
  __REG32 RXNEIE          : 1;
  __REG32 TXEIE           : 1;
  __REG32                 :24;
} __spi_cr2_bits;

/* SPI status register (SPI_SR)*/
typedef struct {
  __REG32 RXNE            : 1;
  __REG32 TXE             : 1;
  __REG32 CHSIDE          : 1;
  __REG32 UDR             : 1;
  __REG32 CRCERR          : 1;
  __REG32 MODF            : 1;
  __REG32 OVR             : 1;
  __REG32 BSY             : 1;
  __REG32                 :24;
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

/* Status register (USART_SR) */
typedef struct {
  __REG32 PE              : 1;
  __REG32 FE              : 1;
  __REG32 NE              : 1;
  __REG32 ORE             : 1;
  __REG32 IDLE            : 1;
  __REG32 RXNE            : 1;
  __REG32 TC              : 1;
  __REG32 TXE             : 1;
  __REG32 LBD             : 1;
  __REG32 CTS             : 1;
  __REG32                 :22;
} __usart_sr_bits;

/* Data register (USART_DR) */
typedef struct {
  __REG32 DR              : 9;
  __REG32                 :23;
} __usart_dr_bits;

/* Baud rate register (USART_BRR) */
typedef struct {
  __REG32 DIV_Fraction    : 4;
  __REG32 DIV_Mantissa    :12;
  __REG32                 :16;
} __usart_brr_bits;

/* Control register 1 (USART_CR1) */
typedef struct {
  __REG32 SBK             : 1;
  __REG32 RWU             : 1;
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
  __REG32 UE              : 1;
  __REG32                 :18;
} __usart_cr1_bits;

/* Control register 2 (USART_CR2) */
typedef struct {
  __REG32 ADD             : 4;
  __REG32                 : 1;
  __REG32 LBDL            : 1;
  __REG32 LBDIE           : 1;
  __REG32                 : 1;
  __REG32 LBCL            : 1;
  __REG32 CPHA            : 1;
  __REG32 CPOL            : 1;
  __REG32 CLKEN           : 1;
  __REG32 STOP            : 2;
  __REG32 LINEN           : 1;
  __REG32                 :17;
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
  __REG32                 :21;
} __usart_cr3_bits;

/* Guard time and prescaler register (USART_GTPR) */
typedef struct {
  __REG32 PSC             : 8;
  __REG32 GT              : 8;
  __REG32                 :16;
} __usart_gtpr_bits;

/* Status register (UART_SR) */
typedef struct {
  __REG32 PE              : 1;
  __REG32 FE              : 1;
  __REG32 NE              : 1;
  __REG32 ORE             : 1;
  __REG32 IDLE            : 1;
  __REG32 RXNE            : 1;
  __REG32 TC              : 1;
  __REG32 TXE             : 1;
  __REG32 LBD             : 1;
  __REG32                 :23;
} __uart_sr_bits;

/* Data register (UART_DR) */
typedef struct {
  __REG32 DR              : 9;
  __REG32                 :23;
} __uart_dr_bits;

/* Baud rate register (UART_BRR) */
typedef struct {
  __REG32 DIV_Fraction    : 4;
  __REG32 DIV_Mantissa    :12;
  __REG32                 :16;
} __uart_brr_bits;

/* Control register 1 (UART_CR1) */
typedef struct {
  __REG32 SBK             : 1;
  __REG32 RWU             : 1;
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
  __REG32 UE              : 1;
  __REG32                 :18;
} __uart_cr1_bits;

/* Control register 2 (UART_CR2) */
typedef struct {
  __REG32 ADD             : 4;
  __REG32                 : 1;
  __REG32 LBDL            : 1;
  __REG32 LBDIE           : 1;
  __REG32                 : 5;
  __REG32 STOP            : 2;
  __REG32 LINEN           : 1;
  __REG32                 :17;
} __uart_cr2_bits;

/* Control register 3 (UART4_CR3) */
typedef struct {
  __REG32 EIE             : 1;
  __REG32 IREN            : 1;
  __REG32 IRLP            : 1;
  __REG32 HDSEL           : 1;
  __REG32                 : 2;
  __REG32 DMAR            : 1;
  __REG32 DMAT            : 1;
  __REG32                 :24;
} __uart4_cr3_bits;

/* Control register 3 (UART5_CR3) */
typedef struct {
  __REG32 EIE             : 1;
  __REG32 IREN            : 1;
  __REG32 IRLP            : 1;
  __REG32 HDSEL           : 1;
  __REG32                 : 2;
  __REG32                 : 1;
  __REG32                 : 1;
  __REG32                 :24;
} __uart5_cr3_bits;

/* ADC status register (ADC_SR) */
typedef struct {
  __REG32 AWD             : 1;
  __REG32 EOC             : 1;
  __REG32 JEOC            : 1;
  __REG32 JSTRT           : 1;
  __REG32 STRT            : 1;
  __REG32                 :27;
} __adc_sr_bits;

/* ADC control register 1 (ADC_CR1) */
typedef struct {
  __REG32 AWDCH           : 5;
  __REG32 EOCIE           : 1;
  __REG32 AWDIE           : 1;
  __REG32 JEOCIE          : 1;
  __REG32 SCAN            : 1;
  __REG32 AWDSGL          : 1;
  __REG32 JAUTO           : 1;
  __REG32 DISCEN          : 1;
  __REG32 JDISCEN         : 1;
  __REG32 DISCNUM         : 3;
  __REG32 DUALMOD         : 4;
  __REG32                 : 2;
  __REG32 JAWDEN          : 1;
  __REG32 AWDEN           : 1;
  __REG32                 : 8;
} __adc_cr1_bits;

/* ADC control register 2 (ADC_CR2) */
typedef struct {
  __REG32 ADON            : 1;
  __REG32 CONT            : 1;
  __REG32 CAL             : 1;
  __REG32 RSTCAL          : 1;
  __REG32                 : 4;
  __REG32 DMA             : 1;
  __REG32                 : 2;
  __REG32 ALIGN           : 1;
  __REG32 JEXTSEL         : 3;
  __REG32 JEXTTRIG        : 1;
  __REG32                 : 1;
  __REG32 EXTSEL          : 3;
  __REG32 EXTTRIG         : 1;
  __REG32 JSWSTART        : 1;
  __REG32 SWSTART         : 1;
  __REG32 TSVREFE         : 1;
  __REG32                 : 8;
} __adc_cr2_bits;

/* ADC sample time register 1 (ADC_SMPR1) */
typedef struct {
  __REG32 SMP10           : 3;
  __REG32 SMP11           : 3;
  __REG32 SMP12           : 3;
  __REG32 SMP13           : 3;
  __REG32 SMP14           : 3;
  __REG32 SMP15           : 3;
  __REG32 SMP16           : 3;
  __REG32 SMP17           : 3;
  __REG32                 : 8;
} __adc_smpr1_bits;

/* ADC sample time register 2 (ADC_SMPR2) */
typedef struct {
  __REG32 SMP0            : 3;
  __REG32 SMP1            : 3;
  __REG32 SMP2            : 3;
  __REG32 SMP3            : 3;
  __REG32 SMP4            : 3;
  __REG32 SMP5            : 3;
  __REG32 SMP6            : 3;
  __REG32 SMP7            : 3;
  __REG32 SMP8            : 3;
  __REG32 SMP9            : 3;
  __REG32                 : 2;
} __adc_smpr2_bits;

/* ADC injected channel data offset register x (ADC_JOFRx)(x=1..4) */
typedef struct {
  __REG32 JOFFSET         :12;
  __REG32                 :20;
} __adc_jofr_bits;

/* ADC watchdog high threshold register (ADC_HTR) */
typedef struct {
  __REG32 HT              :12;
  __REG32                 :20;
} __adc_htr_bits;

/* ADC watchdog low threshold register (ADC_LTR) */
typedef struct {
  __REG32 LT              :12;
  __REG32                 :20;
} __adc_ltr_bits;

/* ADC regular sequence register 1 (ADC_SQR1) */
typedef struct {
  __REG32 SQ13            : 5;
  __REG32 SQ14            : 5;
  __REG32 SQ15            : 5;
  __REG32 SQ16            : 5;
  __REG32 L               : 4;
  __REG32                 : 8;
} __adc_sqr1_bits;

/* ADC regular sequence register 2 (ADC_SQR2) */
typedef struct {
  __REG32 SQ7             : 5;
  __REG32 SQ8             : 5;
  __REG32 SQ9             : 5;
  __REG32 SQ10            : 5;
  __REG32 SQ11            : 5;
  __REG32 SQ12            : 5;
  __REG32                 : 2;
} __adc_sqr2_bits;

/* ADC regular sequence register 3 (ADC_SQR3) */
typedef struct {
  __REG32 SQ1             : 5;
  __REG32 SQ2             : 5;
  __REG32 SQ3             : 5;
  __REG32 SQ4             : 5;
  __REG32 SQ5             : 5;
  __REG32 SQ6             : 5;
  __REG32                 : 2;
} __adc_sqr3_bits;

/* ADC injected sequence register (ADC_JSQR) */
typedef struct {
  __REG32 JSQ1            : 5;
  __REG32 JSQ2            : 5;
  __REG32 JSQ3            : 5;
  __REG32 JSQ4            : 5;
  __REG32 JL              : 2;
  __REG32                 :10;
} __adc_jsqr_bits;

/* ADC injected data register x (ADC_JDRx) (x= 1..4) */
typedef struct {
  __REG32 JDATA           :16;
  __REG32                 :16;
} __adc_jdr_bits;

/* ADC regular data register (ADC_DR) */
typedef struct {
  __REG32 DATA            :16;
  __REG32 ADC2DATA        :16;
} __adc_dr_bits;

/* DAC control register (DAC_CR) */
typedef struct {
  __REG32 EN1             : 1;
  __REG32 BOFF1           : 1;
  __REG32 TEN1            : 1;
  __REG32 TSEL1           : 3;
  __REG32 WAVE1           : 2;
  __REG32 MAMP1           : 4;
  __REG32 DMAEN1          : 1;
  __REG32                 : 3;
  __REG32 EN2             : 1;
  __REG32 BOFF2           : 1;
  __REG32 TEN2            : 1;
  __REG32 TSEL2           : 3;
  __REG32 WAVE2           : 2;
  __REG32 MAMP2           : 4;
  __REG32 DMAEN2          : 1;
  __REG32                 : 3;
} __dac_cr_bits;

/* DAC Software Trigger Register (DAC_SWTRIGR) */
typedef struct {
  __REG32 SWTRIG1         : 1;
  __REG32 SWTRIG2         : 1;
  __REG32                 :30;
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

/* DAC channel2 12-bit Right aligned Data Holding Register (DAC_DHR12R2) */
typedef struct {
  __REG32 DACC2DHR        :12;
  __REG32                 :20;
} __dac_dhr12r2_bits;

/* DAC channel2 12-bit Left aligned Data Holding Register (DAC_DHR12L2) */
typedef struct {
  __REG32                 : 4;
  __REG32 DACC2DHR        :12;
  __REG32                 :16;
} __dac_dhr12l2_bits;

/* DAC channel2 8-bit Right-aligned Data Holding Register (DAC_DHR8R2) */
typedef struct {
  __REG32 DACC2DHR        : 8;
  __REG32                 :24;
} __dac_dhr8r2_bits;

/* Dual DAC 12-bit Right-aligned Data Holding Register (DAC_DHR12RD) */
typedef struct {
  __REG32 DACC1DHR        :12;
  __REG32                 : 4;
  __REG32 DACC2DHR        :12;
  __REG32                 : 4;
} __dac_dhr12rd_bits;

/* DUAL DAC 12-bit Left aligned Data Holding Register (DAC_DHR12LD) */
typedef struct {
  __REG32                 : 4;
  __REG32 DACC1DHR        :12;
  __REG32                 : 4;
  __REG32 DACC2DHR        :12;
} __dac_dhr12ld_bits;

/* DUAL DAC 8-bit Right aligned Data Holding Register (DAC_DHR8RD) */
typedef struct {
  __REG32 DACC1DHR        : 8;
  __REG32 DACC2DHR        : 8;
  __REG32                 :16;  
} __dac_dhr8rd_bits;

/* DAC channel1 Data Output Register (DAC_DOR1) */
typedef struct {
  __REG32 DACC1DOR        :12;
  __REG32                 :20;  
} __dac_dor1_bits;

/* DAC channel2 Data Output Register (DAC_DOR2) */
typedef struct {
  __REG32 DACC2DOR        :12;
  __REG32                 :20;  
} __dac_dor2_bits;


/* MCU device ID code */
typedef struct {
  __REG32 DEV_ID          :12;
  __REG32                 : 4;
  __REG32 REV_ID          :16;
} __dbgmcu_idcode_bits;

/* Debug MCU configuration register  */
typedef struct {
  __REG32 DBG_SLEEP               : 1;
  __REG32 DBG_STOP                : 1;
  __REG32 DBG_STANDBY             : 1;
  __REG32                         : 2;
  __REG32 TRACE_IOEN              : 1;
  __REG32 TRACE_MODE              : 2;
  __REG32 DBG_IWDG_STOP           : 1;
  __REG32 DBG_WWDG_STOP           : 1;
  __REG32 DBG_TIM1_STOP           : 1;
  __REG32 DBG_TIM2_STOP           : 1;
  __REG32 DBG_TIM3_STOP           : 1;
  __REG32 DBG_TIM4_STOP           : 1;
  __REG32 DBG_CAN1_STOP           : 1;
  __REG32 DBG_I2C1_SMBUS_TIMEOUT  : 1;
  __REG32 DBG_I2C2_SMBUS_TIMEOUT  : 1;
  __REG32 DBG_TIM8_STOP           : 1;
  __REG32 DBG_TIM5_STOP           : 1;
  __REG32 DBG_TIM6_STOP           : 1;
  __REG32 DBG_TIM7_STOP           : 1;
  __REG32 DBG_CAN2_STOP           : 1;
  __REG32                         :10;
} __dbgmcu_cr_bits;

/* Interrupt Controller Type Register */
typedef struct {
  __REG32  INTLINESNUM    : 5;
  __REG32                 :27;
} __nvic_bits;

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

/* Interrupt Set-Enable Registers 32-63 */
typedef struct {
  __REG32  SETENA32       : 1;
  __REG32  SETENA33       : 1;
  __REG32  SETENA34       : 1;
  __REG32  SETENA35       : 1;
  __REG32  SETENA36       : 1;
  __REG32  SETENA37       : 1;
  __REG32  SETENA38       : 1;
  __REG32  SETENA39       : 1;
  __REG32  SETENA40       : 1;
  __REG32  SETENA41       : 1;
  __REG32  SETENA42       : 1;
  __REG32  SETENA43       : 1;
  __REG32  SETENA44       : 1;
  __REG32  SETENA45       : 1;
  __REG32  SETENA46       : 1;
  __REG32  SETENA47       : 1;
  __REG32  SETENA48       : 1;
  __REG32  SETENA49       : 1;
  __REG32  SETENA50       : 1;
  __REG32  SETENA51       : 1;
  __REG32  SETENA52       : 1;
  __REG32  SETENA53       : 1;
  __REG32  SETENA54       : 1;
  __REG32  SETENA55       : 1;
  __REG32  SETENA56       : 1;
  __REG32  SETENA57       : 1;
  __REG32  SETENA58       : 1;
  __REG32  SETENA59       : 1;
  __REG32  SETENA60       : 1;
  __REG32  SETENA61       : 1;
  __REG32  SETENA62       : 1;
  __REG32  SETENA63       : 1;
} __setena1_bits;

/* Interrupt Set-Enable Registers 64-95 */
typedef struct {
  __REG32  SETENA64       : 1;
  __REG32  SETENA65       : 1;
  __REG32  SETENA66       : 1;
  __REG32  SETENA67       : 1;
  __REG32  SETENA68       : 1;
  __REG32  SETENA69       : 1;
  __REG32  SETENA70       : 1;
  __REG32  SETENA71       : 1;
  __REG32  SETENA72       : 1;
  __REG32  SETENA73       : 1;
  __REG32  SETENA74       : 1;
  __REG32  SETENA75       : 1;
  __REG32  SETENA76       : 1;
  __REG32  SETENA77       : 1;
  __REG32  SETENA78       : 1;
  __REG32  SETENA79       : 1;
  __REG32  SETENA80       : 1;
  __REG32  SETENA81       : 1;
  __REG32  SETENA82       : 1;
  __REG32  SETENA83       : 1;
  __REG32  SETENA84       : 1;
  __REG32  SETENA85       : 1;
  __REG32  SETENA86       : 1;
  __REG32  SETENA87       : 1;
  __REG32  SETENA88       : 1;
  __REG32  SETENA89       : 1;
  __REG32  SETENA90       : 1;
  __REG32  SETENA91       : 1;
  __REG32  SETENA92       : 1;
  __REG32  SETENA93       : 1;
  __REG32  SETENA94       : 1;
  __REG32  SETENA95       : 1;
} __setena2_bits;

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

/* Interrupt Clear-Enable Registers 32-63 */
typedef struct {
  __REG32  CLRENA32       : 1;
  __REG32  CLRENA33       : 1;
  __REG32  CLRENA34       : 1;
  __REG32  CLRENA35       : 1;
  __REG32  CLRENA36       : 1;
  __REG32  CLRENA37       : 1;
  __REG32  CLRENA38       : 1;
  __REG32  CLRENA39       : 1;
  __REG32  CLRENA40       : 1;
  __REG32  CLRENA41       : 1;
  __REG32  CLRENA42       : 1;
  __REG32  CLRENA43       : 1;
  __REG32  CLRENA44       : 1;
  __REG32  CLRENA45       : 1;
  __REG32  CLRENA46       : 1;
  __REG32  CLRENA47       : 1;
  __REG32  CLRENA48       : 1;
  __REG32  CLRENA49       : 1;
  __REG32  CLRENA50       : 1;
  __REG32  CLRENA51       : 1;
  __REG32  CLRENA52       : 1;
  __REG32  CLRENA53       : 1;
  __REG32  CLRENA54       : 1;
  __REG32  CLRENA55       : 1;
  __REG32  CLRENA56       : 1;
  __REG32  CLRENA57       : 1;
  __REG32  CLRENA58       : 1;
  __REG32  CLRENA59       : 1;
  __REG32  CLRENA60       : 1;
  __REG32  CLRENA61       : 1;
  __REG32  CLRENA62       : 1;
  __REG32  CLRENA63       : 1;
} __clrena1_bits;

/* Interrupt Clear-Enable Registers 64-95 */
typedef struct {
  __REG32  CLRENA64       : 1;
  __REG32  CLRENA65       : 1;
  __REG32  CLRENA66       : 1;
  __REG32  CLRENA67       : 1;
  __REG32  CLRENA68       : 1;
  __REG32  CLRENA69       : 1;
  __REG32  CLRENA70       : 1;
  __REG32  CLRENA71       : 1;
  __REG32  CLRENA72       : 1;
  __REG32  CLRENA73       : 1;
  __REG32  CLRENA74       : 1;
  __REG32  CLRENA75       : 1;
  __REG32  CLRENA76       : 1;
  __REG32  CLRENA77       : 1;
  __REG32  CLRENA78       : 1;
  __REG32  CLRENA79       : 1;
  __REG32  CLRENA80       : 1;
  __REG32  CLRENA81       : 1;
  __REG32  CLRENA82       : 1;
  __REG32  CLRENA83       : 1;
  __REG32  CLRENA84       : 1;
  __REG32  CLRENA85       : 1;
  __REG32  CLRENA86       : 1;
  __REG32  CLRENA87       : 1;
  __REG32  CLRENA88       : 1;
  __REG32  CLRENA89       : 1;
  __REG32  CLRENA90       : 1;
  __REG32  CLRENA91       : 1;
  __REG32  CLRENA92       : 1;
  __REG32  CLRENA93       : 1;
  __REG32  CLRENA94       : 1;
  __REG32  CLRENA95       : 1;
} __clrena2_bits;

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

/* Interrupt Set-Pending Register 32-63 */
typedef struct {
  __REG32  SETPEND32      : 1;
  __REG32  SETPEND33      : 1;
  __REG32  SETPEND34      : 1;
  __REG32  SETPEND35      : 1;
  __REG32  SETPEND36      : 1;
  __REG32  SETPEND37      : 1;
  __REG32  SETPEND38      : 1;
  __REG32  SETPEND39      : 1;
  __REG32  SETPEND40      : 1;
  __REG32  SETPEND41      : 1;
  __REG32  SETPEND42      : 1;
  __REG32  SETPEND43      : 1;
  __REG32  SETPEND44      : 1;
  __REG32  SETPEND45      : 1;
  __REG32  SETPEND46      : 1;
  __REG32  SETPEND47      : 1;
  __REG32  SETPEND48      : 1;
  __REG32  SETPEND49      : 1;
  __REG32  SETPEND50      : 1;
  __REG32  SETPEND51      : 1;
  __REG32  SETPEND52      : 1;
  __REG32  SETPEND53      : 1;
  __REG32  SETPEND54      : 1;
  __REG32  SETPEND55      : 1;
  __REG32  SETPEND56      : 1;
  __REG32  SETPEND57      : 1;
  __REG32  SETPEND58      : 1;
  __REG32  SETPEND59      : 1;
  __REG32  SETPEND60      : 1;
  __REG32  SETPEND61      : 1;
  __REG32  SETPEND62      : 1;
  __REG32  SETPEND63      : 1;
} __setpend1_bits;

/* Interrupt Set-Pending Register 64-95 */
typedef struct {
  __REG32  SETPEND64      : 1;
  __REG32  SETPEND65      : 1;
  __REG32  SETPEND66      : 1;
  __REG32  SETPEND67      : 1;
  __REG32  SETPEND68      : 1;
  __REG32  SETPEND69      : 1;
  __REG32  SETPEND70      : 1;
  __REG32  SETPEND71      : 1;
  __REG32  SETPEND72      : 1;
  __REG32  SETPEND73      : 1;
  __REG32  SETPEND74      : 1;
  __REG32  SETPEND75      : 1;
  __REG32  SETPEND76      : 1;
  __REG32  SETPEND77      : 1;
  __REG32  SETPEND78      : 1;
  __REG32  SETPEND79      : 1;
  __REG32  SETPEND80      : 1;
  __REG32  SETPEND81      : 1;
  __REG32  SETPEND82      : 1;
  __REG32  SETPEND83      : 1;
  __REG32  SETPEND84      : 1;
  __REG32  SETPEND85      : 1;
  __REG32  SETPEND86      : 1;
  __REG32  SETPEND87      : 1;
  __REG32  SETPEND88      : 1;
  __REG32  SETPEND89      : 1;
  __REG32  SETPEND90      : 1;
  __REG32  SETPEND91      : 1;
  __REG32  SETPEND92      : 1;
  __REG32  SETPEND93      : 1;
  __REG32  SETPEND94      : 1;
  __REG32  SETPEND95      : 1;
} __setpend2_bits;

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

/* Interrupt Clear-Pending Register 32-63 */
typedef struct {
  __REG32  CLRPEND32      : 1;
  __REG32  CLRPEND33      : 1;
  __REG32  CLRPEND34      : 1;
  __REG32  CLRPEND35      : 1;
  __REG32  CLRPEND36      : 1;
  __REG32  CLRPEND37      : 1;
  __REG32  CLRPEND38      : 1;
  __REG32  CLRPEND39      : 1;
  __REG32  CLRPEND40      : 1;
  __REG32  CLRPEND41      : 1;
  __REG32  CLRPEND42      : 1;
  __REG32  CLRPEND43      : 1;
  __REG32  CLRPEND44      : 1;
  __REG32  CLRPEND45      : 1;
  __REG32  CLRPEND46      : 1;
  __REG32  CLRPEND47      : 1;
  __REG32  CLRPEND48      : 1;
  __REG32  CLRPEND49      : 1;
  __REG32  CLRPEND50      : 1;
  __REG32  CLRPEND51      : 1;
  __REG32  CLRPEND52      : 1;
  __REG32  CLRPEND53      : 1;
  __REG32  CLRPEND54      : 1;
  __REG32  CLRPEND55      : 1;
  __REG32  CLRPEND56      : 1;
  __REG32  CLRPEND57      : 1;
  __REG32  CLRPEND58      : 1;
  __REG32  CLRPEND59      : 1;
  __REG32  CLRPEND60      : 1;
  __REG32  CLRPEND61      : 1;
  __REG32  CLRPEND62      : 1;
  __REG32  CLRPEND63      : 1;
} __clrpend1_bits;

/* Interrupt Clear-Pending Register 64-95 */
typedef struct {
  __REG32  CLRPEND64      : 1;
  __REG32  CLRPEND65      : 1;
  __REG32  CLRPEND66      : 1;
  __REG32  CLRPEND67      : 1;
  __REG32  CLRPEND68      : 1;
  __REG32  CLRPEND69      : 1;
  __REG32  CLRPEND70      : 1;
  __REG32  CLRPEND71      : 1;
  __REG32  CLRPEND72      : 1;
  __REG32  CLRPEND73      : 1;
  __REG32  CLRPEND74      : 1;
  __REG32  CLRPEND75      : 1;
  __REG32  CLRPEND76      : 1;
  __REG32  CLRPEND77      : 1;
  __REG32  CLRPEND78      : 1;
  __REG32  CLRPEND79      : 1;
  __REG32  CLRPEND80      : 1;
  __REG32  CLRPEND81      : 1;
  __REG32  CLRPEND82      : 1;
  __REG32  CLRPEND83      : 1;
  __REG32  CLRPEND84      : 1;
  __REG32  CLRPEND85      : 1;
  __REG32  CLRPEND86      : 1;
  __REG32  CLRPEND87      : 1;
  __REG32  CLRPEND88      : 1;
  __REG32  CLRPEND89      : 1;
  __REG32  CLRPEND90      : 1;
  __REG32  CLRPEND91      : 1;
  __REG32  CLRPEND92      : 1;
  __REG32  CLRPEND93      : 1;
  __REG32  CLRPEND94      : 1;
  __REG32  CLRPEND95      : 1;
} __clrpend2_bits;

/* Active Bit Register 0-31 */
typedef struct {
  __REG32  ACTIVE0        : 1;
  __REG32  ACTIVE1        : 1;
  __REG32  ACTIVE2        : 1;
  __REG32  ACTIVE3        : 1;
  __REG32  ACTIVE4        : 1;
  __REG32  ACTIVE5        : 1;
  __REG32  ACTIVE6        : 1;
  __REG32  ACTIVE7        : 1;
  __REG32  ACTIVE8        : 1;
  __REG32  ACTIVE9        : 1;
  __REG32  ACTIVE10       : 1;
  __REG32  ACTIVE11       : 1;
  __REG32  ACTIVE12       : 1;
  __REG32  ACTIVE13       : 1;
  __REG32  ACTIVE14       : 1;
  __REG32  ACTIVE15       : 1;
  __REG32  ACTIVE16       : 1;
  __REG32  ACTIVE17       : 1;
  __REG32  ACTIVE18       : 1;
  __REG32  ACTIVE19       : 1;
  __REG32  ACTIVE20       : 1;
  __REG32  ACTIVE21       : 1;
  __REG32  ACTIVE22       : 1;
  __REG32  ACTIVE23       : 1;
  __REG32  ACTIVE24       : 1;
  __REG32  ACTIVE25       : 1;
  __REG32  ACTIVE26       : 1;
  __REG32  ACTIVE27       : 1;
  __REG32  ACTIVE28       : 1;
  __REG32  ACTIVE29       : 1;
  __REG32  ACTIVE30       : 1;
  __REG32  ACTIVE31       : 1;
} __active0_bits;

/* Active Bit Register 32-63 */
typedef struct {
  __REG32  ACTIVE32       : 1;
  __REG32  ACTIVE33       : 1;
  __REG32  ACTIVE34       : 1;
  __REG32  ACTIVE35       : 1;
  __REG32  ACTIVE36       : 1;
  __REG32  ACTIVE37       : 1;
  __REG32  ACTIVE38       : 1;
  __REG32  ACTIVE39       : 1;
  __REG32  ACTIVE40       : 1;
  __REG32  ACTIVE41       : 1;
  __REG32  ACTIVE42       : 1;
  __REG32  ACTIVE43       : 1;
  __REG32  ACTIVE44       : 1;
  __REG32  ACTIVE45       : 1;
  __REG32  ACTIVE46       : 1;
  __REG32  ACTIVE47       : 1;
  __REG32  ACTIVE48       : 1;
  __REG32  ACTIVE49       : 1;
  __REG32  ACTIVE50       : 1;
  __REG32  ACTIVE51       : 1;
  __REG32  ACTIVE52       : 1;
  __REG32  ACTIVE53       : 1;
  __REG32  ACTIVE54       : 1;
  __REG32  ACTIVE55       : 1;
  __REG32  ACTIVE56       : 1;
  __REG32  ACTIVE57       : 1;
  __REG32  ACTIVE58       : 1;
  __REG32  ACTIVE59       : 1;
  __REG32  ACTIVE60       : 1;
  __REG32  ACTIVE61       : 1;
  __REG32  ACTIVE62       : 1;
  __REG32  ACTIVE63       : 1;
} __active1_bits;

/* Active Bit Register 64-95 */
typedef struct {
  __REG32  ACTIVE64       : 1;
  __REG32  ACTIVE65       : 1;
  __REG32  ACTIVE66       : 1;
  __REG32  ACTIVE67       : 1;
  __REG32  ACTIVE68       : 1;
  __REG32  ACTIVE69       : 1;
  __REG32  ACTIVE70       : 1;
  __REG32  ACTIVE71       : 1;
  __REG32  ACTIVE72       : 1;
  __REG32  ACTIVE73       : 1;
  __REG32  ACTIVE74       : 1;
  __REG32  ACTIVE75       : 1;
  __REG32  ACTIVE76       : 1;
  __REG32  ACTIVE77       : 1;
  __REG32  ACTIVE78       : 1;
  __REG32  ACTIVE79       : 1;
  __REG32  ACTIVE80       : 1;
  __REG32  ACTIVE81       : 1;
  __REG32  ACTIVE82       : 1;
  __REG32  ACTIVE83       : 1;
  __REG32  ACTIVE84       : 1;
  __REG32  ACTIVE85       : 1;
  __REG32  ACTIVE86       : 1;
  __REG32  ACTIVE87       : 1;
  __REG32  ACTIVE88       : 1;
  __REG32  ACTIVE89       : 1;
  __REG32  ACTIVE90       : 1;
  __REG32  ACTIVE91       : 1;
  __REG32  ACTIVE92       : 1;
  __REG32  ACTIVE93       : 1;
  __REG32  ACTIVE94       : 1;
  __REG32  ACTIVE95       : 1;
} __active2_bits;

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

/* Interrupt Priority Registers 32-35 */
typedef struct {
  __REG32  PRI_32         : 8;
  __REG32  PRI_33         : 8;
  __REG32  PRI_34         : 8;
  __REG32  PRI_35         : 8;
} __pri8_bits;

/* Interrupt Priority Registers 36-39 */
typedef struct {
  __REG32  PRI_36         : 8;
  __REG32  PRI_37         : 8;
  __REG32  PRI_38         : 8;
  __REG32  PRI_39         : 8;
} __pri9_bits;

/* Interrupt Priority Registers 40-43 */
typedef struct {
  __REG32  PRI_40         : 8;
  __REG32  PRI_41         : 8;
  __REG32  PRI_42         : 8;
  __REG32  PRI_43         : 8;
} __pri10_bits;

/* Interrupt Priority Registers 44-47 */
typedef struct {
  __REG32  PRI_44         : 8;
  __REG32  PRI_45         : 8;
  __REG32  PRI_46         : 8;
  __REG32  PRI_47         : 8;
} __pri11_bits;

/* Interrupt Priority Registers 48-51 */
typedef struct {
  __REG32  PRI_48         : 8;
  __REG32  PRI_49         : 8;
  __REG32  PRI_50         : 8;
  __REG32  PRI_51         : 8;
} __pri12_bits;

/* Interrupt Priority Registers 52-55 */
typedef struct {
  __REG32  PRI_52         : 8;
  __REG32  PRI_53         : 8;
  __REG32  PRI_54         : 8;
  __REG32  PRI_55         : 8;
} __pri13_bits;

/* Interrupt Priority Registers 56-59 */
typedef struct {
  __REG32  PRI_56         : 8;
  __REG32  PRI_57         : 8;
  __REG32  PRI_58         : 8;
  __REG32  PRI_59         : 8;
} __pri14_bits;

/* Interrupt Priority Registers 60-63 */
typedef struct {
  __REG32  PRI_60         : 8;
  __REG32  PRI_61         : 8;
  __REG32  PRI_62         : 8;
  __REG32  PRI_63         : 8;
} __pri15_bits;

/* Interrupt Priority Registers 61-67 */
typedef struct {
  __REG32  PRI_64         : 8;
  __REG32  PRI_65         : 8;
  __REG32  PRI_66         : 8;
  __REG32  PRI_67         : 8;
} __pri16_bits;

/* CPU ID Base Register */
typedef struct {
  __REG32  REVISION       : 4;
  __REG32  PARTNO         :12;
  __REG32                 : 4;
  __REG32  VARIANT        : 4;
  __REG32  IMPLEMENTER    : 8;
} __cpuidbr_bits;

/* Interrupt Control State Register */
typedef struct {
  __REG32  VECTACTIVE     :10;
  __REG32                 : 1;
  __REG32  RETTOBASE      : 1;
  __REG32  VECTPENDING    :10;
  __REG32  ISRPENDING     : 1;
  __REG32  ISRPREEMPT     : 1;
  __REG32                 : 1;
  __REG32  PENDSTCLR      : 1;
  __REG32  PENDSTSET      : 1;
  __REG32  PENDSVCLR      : 1;
  __REG32  PENDSVSET      : 1;
  __REG32                 : 2;
  __REG32  NMIPENDSET     : 1;
} __icsr_bits;

/* Vector Table Offset Register */
typedef struct {
  __REG32                 : 7;
  __REG32  TBLOFF         :22;
  __REG32  TBLBASE        : 1;
  __REG32                 : 2;
} __vtor_bits;

/* Application Interrupt and Reset Control Register */
typedef struct {
  __REG32  VECTRESET      : 1;
  __REG32  VECTCLRACTIVE  : 1;
  __REG32  SYSRESETREQ    : 1;
  __REG32                 : 5;
  __REG32  PRIGROUP       : 3;
  __REG32                 : 4;
  __REG32  ENDIANESS      : 1;
  __REG32  VECTKEY        :16;
} __aircr_bits;

/* System Control Register */
typedef struct {
  __REG32                 : 1;
  __REG32  SLEEPONEXIT    : 1;
  __REG32  SLEEPDEEP      : 1;
  __REG32                 : 1;
  __REG32  SEVONPEND      : 1;
  __REG32                 :27;
} __scr_bits;

/* Configuration Control Register */
typedef struct {
  __REG32  NONEBASETHRDENA: 1;
  __REG32  USERSETMPEND   : 1;
  __REG32                 : 1;
  __REG32  UNALIGN_TRP    : 1;
  __REG32  DIV_0_TRP      : 1;
  __REG32                 : 3;
  __REG32  BFHFNMIGN      : 1;
  __REG32  STKALIGN       : 1;
  __REG32                 :22;
} __ccr_bits;

/* System Handler Control and State Register */
typedef struct {
  __REG32  MEMFAULTACT    : 1;
  __REG32  BUSFAULTACT    : 1;
  __REG32                 : 1;
  __REG32  USGFAULTACT    : 1;
  __REG32                 : 3;
  __REG32  SVCALLACT      : 1;
  __REG32  MONITORACT     : 1;
  __REG32                 : 1;
  __REG32  PENDSVACT      : 1;
  __REG32  SYSTICKACT     : 1;
  __REG32                 : 1;
  __REG32  MEMFAULTPENDED : 1;
  __REG32  BUSFAULTPENDED : 1;
  __REG32  SVCALLPENDED   : 1;
  __REG32  MEMFAULTENA    : 1;
  __REG32  BUSFAULTENA    : 1;
  __REG32  USGFAULTENA    : 1;
  __REG32                 :13;
} __shcsr_bits;

/* Configurable Fault Status Registers */
typedef struct {
  __REG32  IACCVIOL       : 1;
  __REG32  DACCVIOL       : 1;
  __REG32                 : 1;
  __REG32  MUNSTKERR      : 1;
  __REG32  MSTKERR        : 1;
  __REG32                 : 2;
  __REG32  MMARVALID      : 1;
  __REG32  IBUSERR        : 1;
  __REG32  PRECISERR      : 1;
  __REG32  IMPRECISERR    : 1;
  __REG32  UNSTKERR       : 1;
  __REG32  STKERR         : 1;
  __REG32                 : 2;
  __REG32  BFARVALID      : 1;
  __REG32  UNDEFINSTR     : 1;
  __REG32  INVSTATE       : 1;
  __REG32  INVPC          : 1;
  __REG32  NOCP           : 1;
  __REG32                 : 4;
  __REG32  UNALIGNED      : 1;
  __REG32  DIVBYZERO      : 1;
  __REG32                 : 6;
} __cfsr_bits;

/* Hard Fault Status Register */
typedef struct {
  __REG32                 : 1;
  __REG32  VECTTBL        : 1;
  __REG32                 :28;
  __REG32  FORCED         : 1;
  __REG32  DEBUGEVT       : 1;
} __hfsr_bits;

/* Debug Fault Status Register */
typedef struct {
  __REG32  HALTED         : 1;
  __REG32  BKPT           : 1;
  __REG32  DWTTRAP        : 1;
  __REG32  VCATCH         : 1;
  __REG32  EXTERNAL       : 1;
  __REG32                 :27;
} __dfsr_bits;

/* Software Trigger Interrupt Register */
typedef struct {
  __REG32  INTID          : 9;
  __REG32                 :23;
} __stir_bits;

/* Flash Access Control Register (FLASH_ACR) */
typedef struct {
  __REG32  LATENCY        : 3;
  __REG32  HLFCYA         : 1;
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
  __REG32                 :19;
} __flash_cr_bits;

/* Option Byte Register (FLASH_OBR) */
typedef struct {
  __REG32  OPTERR         : 1;
  __REG32  RDPRT          : 1;
  __REG32  WDG_SW         : 1;
  __REG32  nRST_STOP      : 1;
  __REG32  nRST_STDBY     : 1;
  __REG32                 :27;
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
  __REG32  WRP16          : 1;
  __REG32  WRP17          : 1;
  __REG32  WRP18          : 1;
  __REG32  WRP19          : 1;
  __REG32  WRP20          : 1;
  __REG32  WRP21          : 1;
  __REG32  WRP22          : 1;
  __REG32  WRP23          : 1;
  __REG32  WRP24          : 1;
  __REG32  WRP25          : 1;
  __REG32  WRP26          : 1;
  __REG32  WRP27          : 1;
  __REG32  WRP28          : 1;
  __REG32  WRP29          : 1;
  __REG32  WRP30          : 1;
  __REG32  WRP31          : 1;
} __flash_wrpr_bits;

/* Independent data register (CRC_IDR) */
typedef struct {
  __REG32  IDR            : 8;
  __REG32                 :24;
} __crc_idr_bits;

/* Control register (CRC_CR) */
typedef struct {
  __REG32  RESET          : 1;
  __REG32                 :31;
} __crc_cr_bits;

/* OTG_FS control and status register (OTG_FS_GOTGCTL) */
typedef struct {
  __REG32  SRQSCS         : 1;
  __REG32  SRQ            : 1;
  __REG32                 : 6;
  __REG32  HNGSCS         : 1;
  __REG32  HNPRQ          : 1;
  __REG32  HSHNPEN        : 1;
  __REG32  DHNPEN         : 1;
  __REG32                 : 4;
  __REG32  CIDSTS         : 1;
  __REG32  DBCT           : 1;
  __REG32  ASVLD          : 1;
  __REG32  BSVLD          : 1;
  __REG32                 :12;
} __otg_fs_gotgctl_bits;

/* OTG_FS interrupt register (OTG_FS_GOTGINT) */
typedef struct {
  __REG32                 : 2;
  __REG32  SEDET          : 1;
  __REG32                 : 5;
  __REG32  SRSSCHG        : 1;
  __REG32  HNSSCHG        : 1;
  __REG32                 : 7;
  __REG32  HNGDET         : 1;
  __REG32  ADTOCHG        : 1;
  __REG32  DBCDNE         : 1;
  __REG32                 :12;
} __otg_fs_gotgint_bits;

/* OTG_FS AHB configuration register (OTG_FS_GAHBCFG) */
typedef struct {
  __REG32  GINT           : 1;
  __REG32                 : 6;
  __REG32  TXFELVL        : 1;
  __REG32  PTXFELVL       : 1;
  __REG32                 :23;
} __otg_fs_gahbcfg_bits;

/* OTG_FS USB configuration register (OTG_FS_GUSBCFG) */
typedef struct {
  __REG32  TOCAL          : 3;
  __REG32                 : 5;
  __REG32  SRPCAP         : 1;
  __REG32  HNPCAP         : 1;
  __REG32  TRDT           : 4;
  __REG32  NPTXRWEN       : 1;
  __REG32                 :14;
  __REG32  FHMOD          : 1;
  __REG32  FDMOD          : 1;
  __REG32  CTXPKT         : 1;
} __otg_fs_gusbcfg_bits;

/* OTG_FS reset register (OTG_FS_GRSTCTL) */
typedef struct {
  __REG32  CSRST          : 1;
  __REG32  HSRST          : 1;
  __REG32  FCRST          : 1;
  __REG32                 : 1;
  __REG32  RXFFLSH        : 1;
  __REG32  TXFFLSH        : 1;
  __REG32  TXFNUM         : 5;
  __REG32                 :20;
  __REG32  AHBIDL         : 1;
} __otg_fs_grstctl_bits;

/* OTG_FS core interrupt register (OTG_FS_GINTSTS) */
typedef struct {
  __REG32  CMOD           : 1;
  __REG32  MMIS           : 1;
  __REG32  OTGINT         : 1;
  __REG32  SOF            : 1;
  __REG32  RXFLVL         : 1;
  __REG32                 : 1;
  __REG32  GINAKEFF       : 1;
  __REG32  GONAKEFF       : 1;
  __REG32                 : 2;
  __REG32  ESUSP          : 1;
  __REG32  USBSUSP        : 1;
  __REG32  USBRST         : 1;
  __REG32  ENUMDNE        : 1;
  __REG32  ISOODRP        : 1;
  __REG32  EOPF           : 1;
  __REG32                 : 2;
  __REG32  IEPINT         : 1;
  __REG32  OEPINT         : 1;
  __REG32  IISOIXFR       : 1;
  __REG32  IPXFR          : 1;
  __REG32                 : 2;
  __REG32  HPRTINT        : 1;
  __REG32  HCINT          : 1;
  __REG32  PTXFE          : 1;
  __REG32                 : 1;
  __REG32  CIDSCHG        : 1;  
  __REG32  DISCINT        : 1;  
  __REG32  SRQINT         : 1;  
  __REG32  WKUINT         : 1;  
} __otg_fs_gintsts_bits;

/* OTG_FS interrupt mask register (OTG_FS_GINTMSK) */
typedef struct {
  __REG32                 : 1;
  __REG32  MMISM          : 1;
  __REG32  OTGINT         : 1;
  __REG32  SOFM           : 1;
  __REG32  RXFLVLM        : 1;
  __REG32  NPTXFEM        : 1;
  __REG32  GINAKEFFM      : 1;
  __REG32  GONAKEFFM      : 1;
  __REG32                 : 2;
  __REG32  ESUSPM         : 1;
  __REG32  USBSUSPM       : 1;
  __REG32  USBRST         : 1;
  __REG32  ENUMDNEM       : 1;
  __REG32  ISOODRPM       : 1;
  __REG32  EOPFM          : 1;
  __REG32                 : 1;
  __REG32  EPMISM         : 1;
  __REG32  IEPINT         : 1;
  __REG32  OEPINT         : 1;
  __REG32  IISOIXFRM      : 1;
  __REG32  IPXFRM         : 1;
  __REG32  FSUSPM         : 1;
  __REG32                 : 1;
  __REG32  PRTIM          : 1;
  __REG32  HCIM           : 1;
  __REG32  PTXFEM         : 1;
  __REG32                 : 1;
  __REG32  CIDSCHGM       : 1;  
  __REG32  DISCINT        : 1;  
  __REG32  SRQIM          : 1;  
  __REG32  WUIM           : 1;  
} __otg_fs_gintmsk_bits;

/* OTG_FS Receive status debug read/OTG status read and pop registers */
/* (OTG_FS_GRXSTSR/OTG_FS_GRXSTSP)                                    */
typedef union {
  /*OTG_FS_GRXSTSR*/
  /*OTG_FS_GRXSTSP*/
  struct {
    __REG32  CHNUM        : 4;  
    __REG32  BCNT         :11;  
    __REG32  DPID         : 2;  
    __REG32  PKTSTS       : 4;  
    __REG32               :11;  
  };
  /*OTG_FS_GRXSTSR_DEV*/
  /*OTG_FS_GRXSTSP_DEV  */
  struct {
    __REG32  _EPNUM       : 4;  
    __REG32  _BCNT        :11;  
    __REG32  _DPID        : 2;  
    __REG32  _PKTSTS      : 4;  
    __REG32  _FRMNUM      : 4;  
    __REG32               : 7;  
  };
} __otg_fs_grxstsr_bits;

/* OTG_FS Receive FIFO size register (OTG_FS_GRXFSIZ) */
typedef struct {
  __REG32  RXFD           :16;
  __REG32                 :16;  
} __otg_fs_grxfsiz_bits;

/* OTG_FS non-periodic transmit FIFO size register (OTG_FS_GNPTXFSIZ) */
typedef struct {
  __REG32  NPTXFSA        :16;
  __REG32  NPTXFD         :16;  
} __otg_fs_gnptxfsiz_bits;

/* OTG_FS non-periodic transmit FIFO/queue status register */
/* (OTG_FS_GNPTXSTS) */
typedef struct {
  __REG32  NPTXFSAV       :16;
  __REG32  NPTQXSAV       : 8;  
  __REG32  NPTXQTOP       : 7;  
  __REG32                 : 1;  
} __otg_fs_gnptxsts_bits;

/* OTG_FS general core configuration register (OTG_FS_GCCFG) */
typedef struct {
  __REG32                 :16;
  __REG32  PWRDWN         : 1;  
  __REG32                 : 1;  
  __REG32  VBUSASEN       : 1;  
  __REG32  VBUSBSEN       : 1;  
  __REG32  SOFOUTEN       : 1;  
  __REG32                 :11;  
} __otg_fs_gccfg_bits;

/* OTG_FS Host periodic transmit FIFO size register (OTG_FS_HPTXFSIZ) */
typedef struct {
  __REG32  PTXSA          :16;
  __REG32  PTXFSIZ        :16;  
} __otg_fs_hptxfsiz_bits;

/* OTG_FS device IN endpoint transmit FIFO size register (OTG_FS_DIEPTXFx) */
typedef struct {
  __REG32  INEPTXSA       :16;
  __REG32  INEPTXFD       :16;  
} __otg_fs_dieptxfx_bits;

/* OTG_FS host configuration register (OTG_FS_HCFG) */
typedef struct {
  __REG32  FSLSPCS        : 2;
  __REG32  FSLSS          : 1;
  __REG32                 :29;  
} __otg_fs_hcfg_bits;

/* OTG_FS Host frame interval register (OTG_FS_HFIR) */
typedef struct {
  __REG32  FRIVL          :16;
  __REG32                 :16;  
} __otg_fs_hfir_bits;

/* OTG_FS host frame number/frame time remaining register (OTG_FS_HFNUM) */
typedef struct {
  __REG32  FRNUM          :16;
  __REG32  FTREM          :16;  
} __otg_fs_hfnum_bits;

/* OTG_FS_Host periodic transmit FIFO/queue status register (OTG_FS_HPTXSTS)*/
typedef struct {
  __REG32  PTXFSAVL       :16;
  __REG32  PTXQSAV        : 8;  
  __REG32  PTXQTOP        : 8;  
} __otg_fs_hptxsts_bits;

/* OTG_FS Host all channels interrupt register (OTG_FS_HAINT)*/
typedef struct {
  __REG32  CHN0           : 1;
  __REG32  CHN1           : 1;
  __REG32  CHN2           : 1;
  __REG32  CHN3           : 1;
  __REG32  CHN4           : 1;
  __REG32  CHN5           : 1;
  __REG32  CHN6           : 1;
  __REG32  CHN7           : 1;
  __REG32  CHN8           : 1;
  __REG32  CHN9           : 1;
  __REG32  CHN10          : 1;
  __REG32  CHN11          : 1;
  __REG32  CHN12          : 1;
  __REG32  CHN13          : 1;
  __REG32  CHN14          : 1;
  __REG32  CHN15          : 1;
  __REG32                 :16;  
} __otg_fs_haint_bits;

/* OTG_FS host all channels interrupt mask register (OTG_FS_HAINTMSK)*/
typedef struct {
  __REG32  CHN0M          : 1;
  __REG32  CHN1M          : 1;
  __REG32  CHN2M          : 1;
  __REG32  CHN3M          : 1;
  __REG32  CHN4M          : 1;
  __REG32  CHN5M          : 1;
  __REG32  CHN6M          : 1;
  __REG32  CHN7M          : 1;
  __REG32  CHN8M          : 1;
  __REG32  CHN9M          : 1;
  __REG32  CHN10M         : 1;
  __REG32  CHN11M         : 1;
  __REG32  CHN12M         : 1;
  __REG32  CHN13M         : 1;
  __REG32  CHN14M         : 1;
  __REG32  CHN15M         : 1;
  __REG32                 :16;  
} __otg_fs_haintmsk_bits;

/* OTG_FS Host port control and status register (OTG_FS_HPRT) */
typedef struct {
  __REG32  PCSTS          : 1;
  __REG32  PCDET          : 1;
  __REG32  PENA           : 1;
  __REG32  PENCHNG        : 1;
  __REG32  POCA           : 1;
  __REG32  POCCHNG        : 1;
  __REG32  PRES           : 1;
  __REG32  PSUSP          : 1;
  __REG32  PRST           : 1;
  __REG32                 : 1;
  __REG32  PLSTS          : 2;
  __REG32  PPWR           : 1;
  __REG32  PTCTL          : 4;
  __REG32  PSPD           : 2;
  __REG32                 :13;  
} __otg_fs_hprt_bits;

/* OTG_FS host channel-x characteristics register (OTG_FS_HCCHARx) */
typedef struct {  
  __REG32  MPSIZ          :11;
  __REG32  EPNUM          : 4;
  __REG32  EPDIR          : 1;
  __REG32                 : 1;
  __REG32  LSDEV          : 1;
  __REG32  EPTYP          : 2;
  __REG32                 : 2;
  __REG32  DAD            : 7;
  __REG32  ODDFRM         : 1;
  __REG32  CHDIS          : 1;
  __REG32  CHENA          : 1;
} __otg_fs_hccharx_bits;

/* OTG_FS Host channel-x interrupt register (OTG_FS_HCINTx) */
typedef struct {  
  __REG32  XFRC           : 1;
  __REG32  CHH            : 1;
  __REG32                 : 1;
  __REG32  STALL          : 1;
  __REG32  NAK            : 1;
  __REG32  ACK            : 1;
  __REG32                 : 1;
  __REG32  TXERR          : 1;
  __REG32  BBERR          : 1;
  __REG32  FRMOR          : 1;
  __REG32  DTERR          : 1;
  __REG32                 :21;
} __otg_fs_hcintx_bits;

/* OTG_FS host channel-x interrupt mask register (OTG_FS_HCINTMSKx) */
typedef struct {  
  __REG32  XFRCM          : 1;
  __REG32  CHHM           : 1;
  __REG32                 : 1;
  __REG32  STALLM         : 1;
  __REG32  NAKM           : 1;
  __REG32  ACKM           : 1;
  __REG32  NYET           : 1;
  __REG32  TXERRM         : 1;
  __REG32  BBERRM         : 1;
  __REG32  FRMORM         : 1;
  __REG32  DTERRM         : 1;
  __REG32                 :21;
} __otg_fs_hcintmskx_bits;

/* OTG_FS host channel-x transfer size register (OTG_FS_HCTSIZx) */
typedef struct {  
  __REG32  XFRSIZ         :19;
  __REG32  PKTCNT         :10;
  __REG32  DPID           : 2;
  __REG32                 : 1;
} __otg_fs_hctsizx_bits;

/* OTG_FS Device configuration register (OTG_FS_DCFG) */
typedef struct {  
  __REG32  DSPD           : 2;
  __REG32  NZLSOHSK       : 1;
  __REG32                 : 1;
  __REG32  DAD            : 7;
  __REG32  PFIVL          : 2;
  __REG32                 :19;
} __otg_fs_dcfg_bits;

/* OTG_FS Device control register (OTG_FS_DCTL) */
typedef struct {  
  __REG32  RWUSIG         : 1;
  __REG32  SDIS           : 1;
  __REG32  GINSTS         : 1;
  __REG32  GONSTS         : 1;
  __REG32  TCTL           : 3;
  __REG32  SGINAK         : 1;
  __REG32  CGINAK         : 1;
  __REG32  SGONAK         : 1;
  __REG32  CGONAK         : 1;
  __REG32  POPRGDNE       : 1;
  __REG32                 :20;
} __otg_fs_dctl_bits;

/* OTG_FS Device status register (OTG_FS_DSTS) */
typedef struct {  
  __REG32  SUSPSTS        : 1;
  __REG32  ENUMSPD        : 2;
  __REG32  EERR           : 1;
  __REG32                 : 4;
  __REG32  FNSOF          :14;
  __REG32                 :10;
} __otg_fs_dsts_bits;

/* OTG_FS device IN endpoint common interrupt mask register */
/* (OTG_FS_DIEPMSK) */
typedef struct {  
  __REG32  XFRCM          : 1;
  __REG32  EPDM           : 1;
  __REG32                 : 1;
  __REG32  TOM            : 1;
  __REG32  ITTXFEMSK      : 1;
  __REG32  INEPNMM        : 1;
  __REG32  INEPNEM        : 1;
  __REG32                 : 1;
  __REG32  TXFURM         : 1;
  __REG32  BIM            : 1;
  __REG32                 :22;
} __otg_fs_diepmsk_bits;

/* OTG_FS Device OUT endpoint common interrupt mask register */
/* (OTG_FS_DOEPMSK) */
typedef struct {  
  __REG32  XFRCM          : 1;
  __REG32  EPDM           : 1;
  __REG32                 : 1;
  __REG32  STUPM          : 1;
  __REG32  OTEPDM         : 1;
  __REG32                 : 1;
  __REG32  B2BSTUP        : 1;
  __REG32                 : 1;
  __REG32  OPEM           : 1;
  __REG32  BOIM           : 1;
  __REG32                 :22;
} __otg_fs_doepmsk_bits;

/* OTG_FS Device all endpoints interrupt register (OTG_FS_DAINT)*/
typedef struct {
  __REG32  IEP0           : 1;
  __REG32  IEP1           : 1;
  __REG32  IEP2           : 1;
  __REG32  IEP3           : 1;
  __REG32  IEP4           : 1;
  __REG32  IEP5           : 1;
  __REG32  IEP6           : 1;
  __REG32  IEP7           : 1;
  __REG32  IEP8           : 1;
  __REG32  IEP9           : 1;
  __REG32  IEP10          : 1;
  __REG32  IEP11          : 1;
  __REG32  IEP12          : 1;
  __REG32  IEP13          : 1;
  __REG32  IEP14          : 1;
  __REG32  IEP15          : 1;
  __REG32  OEP0           : 1;
  __REG32  OEP1           : 1;
  __REG32  OEP2           : 1;
  __REG32  OEP3           : 1;
  __REG32  OEP4           : 1;
  __REG32  OEP5           : 1;
  __REG32  OEP6           : 1;
  __REG32  OEP7           : 1;
  __REG32  OEP8           : 1;
  __REG32  OEP9           : 1;
  __REG32  OEP10          : 1;
  __REG32  OEP11          : 1;
  __REG32  OEP12          : 1;
  __REG32  OEP13          : 1;
  __REG32  OEP14          : 1;
  __REG32  OEP15          : 1;
} __otg_fs_daint_bits;

/* OTG_FS All endpoints interrupt mask register (OTG_FS_DAINTMSK)*/
typedef struct {
  __REG32  IEP0M          : 1;
  __REG32  IEP1M          : 1;
  __REG32  IEP2M          : 1;
  __REG32  IEP3M          : 1;
  __REG32  IEP4M          : 1;
  __REG32  IEP5M          : 1;
  __REG32  IEP6M          : 1;
  __REG32  IEP7M          : 1;
  __REG32  IEP8M          : 1;
  __REG32  IEP9M          : 1;
  __REG32  IEP10M         : 1;
  __REG32  IEP11M         : 1;
  __REG32  IEP12M         : 1;
  __REG32  IEP13M         : 1;
  __REG32  IEP14M         : 1;
  __REG32  IEP15M         : 1;
  __REG32  OEP0M          : 1;
  __REG32  OEP1M          : 1;
  __REG32  OEP2M          : 1;
  __REG32  OEP3M          : 1;
  __REG32  OEP4M          : 1;
  __REG32  OEP5M          : 1;
  __REG32  OEP6M          : 1;
  __REG32  OEP7M          : 1;
  __REG32  OEP8M          : 1;
  __REG32  OEP9M          : 1;
  __REG32  OEP10M         : 1;
  __REG32  OEP11M         : 1;
  __REG32  OEP12M         : 1;
  __REG32  OEP13M         : 1;
  __REG32  OEP14M         : 1;
  __REG32  OEP15M         : 1;
} __otg_fs_daintmsk_bits;

/* OTG_FS Device VBUS discharge time register (OTG_FS_DVBUSDIS) */
typedef struct {  
  __REG32  VBUSDT         :16;
  __REG32                 :16;
} __otg_fs_dvbusdis_bits;

/* OTG_FS Device VBUS pulsing time register (OTG_FS_DVBUSPULSE) */
typedef struct {  
  __REG32  DVBUSP         :12;
  __REG32                 :20;
} __otg_fs_dvbuspulse_bits;

/* OTG_FS Device IN endpoint FIFO empty interrupt mask register */
/* (OTG_FS_DIEPEMPMSK) */
typedef struct {  
  __REG32  INEPTXFEM      :16;
  __REG32                 :16;
} __otg_fs_diepempmsk_bits;

/* OTG_FS device control IN endpoint 0 control register (OTG_FS_DIEPCTL0) */
typedef struct {  
  __REG32  MPSIZ          : 2;
  __REG32                 :13;
  __REG32  USBAEP         : 1;
  __REG32                 : 1;
  __REG32  NAKSTS         : 1;
  __REG32  EPTYP          : 2;
  __REG32                 : 1;
  __REG32  STALL          : 1;
  __REG32  TXFNUM         : 4;
  __REG32  CNAK           : 1;
  __REG32  SNAK           : 1;
  __REG32                 : 2;
  __REG32  EPDIS          : 1;
  __REG32  EPENA          : 1;
} __otg_fs_diepctl0_bits;

/* OTG Device endpoint-x control register (OTG_FS_DIEPCTLx) */
typedef struct {  
  __REG32  MPSIZ          :11;
  __REG32                 : 4;
  __REG32  USBAEP         : 1;
  __REG32  EONUM_DPID     : 1;
  __REG32  NAKSTS         : 1;
  __REG32  EPTYP          : 2;
  __REG32                 : 1;
  __REG32  STALL          : 1;
  __REG32  TXFNUM         : 4;
  __REG32  CNAK           : 1;
  __REG32  SNAK           : 1;
  __REG32  SD0PID_SEVNFRM : 1;
  __REG32  SODDFRM        : 1;
  __REG32  EPDIS          : 1;
  __REG32  EPENA          : 1;
} __otg_fs_diepctlx_bits;

/* OTG_FS Device endpoint-x interrupt register (OTG_FS_DIEPINTx) */
typedef struct {  
  __REG32  XFRC           : 1;
  __REG32  EPDISD         : 1;
  __REG32                 : 1;
  __REG32  TOC            : 1;
  __REG32  ITTXFE         : 1;
  __REG32                 : 1;
  __REG32  INEPNE         : 1;
  __REG32  TXFE           : 1;
  __REG32                 :24;
} __otg_fs_diepintx_bits;

/* OTG_FS device IN endpoint 0 transfer size register (OTG_FS_DIEPTSIZ0) */
typedef struct {  
  __REG32  XFRSIZ         : 7;
  __REG32                 :12;
  __REG32  PKTCNT         : 2;
  __REG32                 :11;
} __otg_fs_dieptsiz0_bits;

/* OTG_FS Device endpoint-x transfer size register (OTG_FS_DIEPTSIZx) */
typedef struct {  
  __REG32  XFRSIZ         :19;
  __REG32  PKTCNT         :10;
  __REG32  MCNT           : 2;
  __REG32                 : 1;
} __otg_fs_dieptsizx_bits;

/* OTG_FS Device IN endpoint transmit FIFO status register (OTG_FS_DTXFSTSx) */
typedef struct {  
  __REG32  INEPTFSAV      :16;
  __REG32                 :16;
} __otg_fs_dtxfstsx_bits;

/* OTG_FS device control OUT endpoint 0 control register (OTG_FS_DOEPCTL0) */
typedef struct {  
  __REG32  MPSIZ          : 2;
  __REG32                 :13;
  __REG32  USBAEP         : 1;
  __REG32                 : 1;
  __REG32  NAKSTS         : 1;
  __REG32  EPTYP          : 2;
  __REG32  SNPM           : 1;
  __REG32  STALL          : 1;
  __REG32                 : 4;
  __REG32  CNAK           : 1;
  __REG32  SNAK           : 1;
  __REG32                 : 2;
  __REG32  EPDIS          : 1;
  __REG32  EPENA          : 1;
} __otg_fs_doepctl0_bits;

/* OTG_FS Device endpoint-x control register (OTG_FS_DOEPCTLx) */
typedef struct {  
  __REG32  MPSIZ          :11;
  __REG32                 : 4;
  __REG32  USBAEP         : 1;
  __REG32  EONUM_DPID     : 1;
  __REG32  NAKSTS         : 1;
  __REG32  EPTYP          : 2;
  __REG32  SNPM           : 1;
  __REG32  STALL          : 1;
  __REG32                 : 4;
  __REG32  CNAK           : 1;
  __REG32  SNAK           : 1;
  __REG32  SD0PID_SEVNFRM : 1;
  __REG32  SODDFRM        : 1;
  __REG32  EPDIS          : 1;
  __REG32  EPENA          : 1;
} __otg_fs_doepctlx_bits;

/* OTG_FS Device endpoint-x interrupt register (OTG_FS_DOEPINTx) */
typedef struct {  
  __REG32  XFRC           : 1;
  __REG32  EPDISD         : 1;
  __REG32                 : 1;
  __REG32  STUP           : 1;
  __REG32  OTEPDIS        : 1;
  __REG32                 : 1;
  __REG32  B2BSTUP        : 1;
  __REG32                 :25;
} __otg_fs_doepintx_bits;

/* OTG_FS device OUT endpoint 0 transfer size register (OTG_FS_DOEPTSIZ0) */
typedef struct {  
  __REG32  XFRSIZ         : 7;
  __REG32                 :12;
  __REG32  PKTCNT         : 1;
  __REG32                 : 9;
  __REG32  STUPCNT        : 2;
  __REG32                 : 1;
} __otg_fs_doeptsiz0_bits;

/* OTG_FS Device endpoint-x transfer size register (OTG_FS_DOEPTSIZx) */
typedef struct {  
  __REG32  XFRSIZ         :19;
  __REG32  PKTCNT         :10;
  __REG32  RXDPID_STUPCNT : 2;
  __REG32                 : 1;
} __otg_fs_doeptsizx_bits;

/* OTG_FS power and clock gating control register (OTG_FS_PCGCCTL) */
typedef struct {  
  __REG32  STPPCLK        : 1;
  __REG32  GATEHCLK       : 1;
  __REG32                 : 2;
  __REG32  PHYSUSP        : 1;
  __REG32                 :27;
} __otg_fs_pcgcctl_bits;

/* Ethernet MAC configuration register (ETH_MACCR) */
typedef struct {  
  __REG32                 : 2;
  __REG32  RE             : 1;
  __REG32  TE             : 1;
  __REG32  DC             : 1;
  __REG32  BL             : 2;
  __REG32  APCS           : 1;
  __REG32                 : 1;
  __REG32  RD             : 1;
  __REG32  IPCO           : 1;
  __REG32  DM             : 1;
  __REG32  LM             : 1;
  __REG32  ROD            : 1;
  __REG32  FES            : 1;
  __REG32                 : 1;
  __REG32  CSD            : 1;
  __REG32  IFG            : 3;
  __REG32  JFE            : 1;
  __REG32                 : 1;
  __REG32  JD             : 1;
  __REG32  WD             : 1;
  __REG32                 : 8;
} __eth_maccr_bits;

/* Ethernet MAC frame filter register (ETH_MACFFR) */
typedef struct {  
  __REG32  PM             : 1;
  __REG32  HU             : 1;
  __REG32  HM             : 1;
  __REG32  DAIF           : 1;
  __REG32  PAM            : 1;
  __REG32  BFD            : 1;
  __REG32  PCF            : 2;
  __REG32  SAIF           : 1;
  __REG32  SAF            : 1;
  __REG32  HPF            : 1;
  __REG32                 :20;
  __REG32  RA             : 1;
} __eth_macffr_bits;

/* Ethernet MAC MII address register (ETH_MACMIIAR) */
typedef struct {  
  __REG32  MB             : 1;
  __REG32  MW             : 1;
  __REG32  CR             : 3;
  __REG32                 : 1;
  __REG32  MR             : 5;
  __REG32  PA             : 5;
  __REG32                 :16;
} __eth_macmiiar_bits;

/* Ethernet MAC MII data register (ETH_MACMIIDR) */
typedef struct {  
  __REG32  MD             :16;
  __REG32                 :16;
} __eth_macmiidr_bits;

/* Ethernet MAC flow control register (ETH_MACFCR) */
typedef struct {  
  __REG32  FCB_BPA        : 1;
  __REG32  TFCE           : 1;
  __REG32  RFCE           : 1;
  __REG32  UPFD           : 1;
  __REG32  PLT            : 2;
  __REG32                 : 1;
  __REG32  ZQPD           : 1;
  __REG32                 : 8;
  __REG32  PT             :16;
} __eth_macfcr_bits;

/* Ethernet MAC VLAN tag register (ETH_MACVLANTR) */
typedef struct {  
  __REG32  VLANTI         :16;
  __REG32  VLANTC         : 1;
  __REG32                 :15;
} __eth_macvlantr_bits;

/* Ethernet MAC PMT control and status register (ETH_MACPMTCSR) */
typedef struct {  
  __REG32  PD             : 1;
  __REG32  MPE            : 1;
  __REG32  WFE            : 1;
  __REG32                 : 2;
  __REG32  MPR            : 1;
  __REG32  WFR            : 1;
  __REG32                 : 2;
  __REG32  GU             : 1;
  __REG32                 :21;
  __REG32  WFFRPR         : 1;
} __eth_macpmtcsr_bits;

/* Ethernet MAC interrupt status register (ETH_MACSR) */
typedef struct {  
  __REG32                 : 3;
  __REG32  PMTS           : 1;
  __REG32  MMCS           : 1;
  __REG32  MMCRS          : 1;
  __REG32  MMCTS          : 1;
  __REG32                 : 2;
  __REG32  TSTS           : 1;
  __REG32                 :22;
} __eth_macsr_bits;

/* Ethernet MAC interrupt mask register (ETH_MACIMR) */
typedef struct {  
  __REG32                 : 3;
  __REG32  PMTIM          : 1;
  __REG32                 : 5;
  __REG32  TSTIM          : 1;
  __REG32                 :22;
} __eth_macimr_bits;

/* Ethernet MAC address 0 high register (ETH_MACA0HR) */
typedef struct {  
  __REG32  MACA0H         :16;
  __REG32                 :15;
  __REG32  MO             : 1;
} __eth_maca0hr_bits;

/* Ethernet MAC address 0 low register (ETH_MACA0LR) */
typedef struct {  
  __REG32  MACA0L         :32;
} __eth_maca0lr_bits;

/* Ethernet MAC address 1 high register (ETH_MACA1HR) */
typedef struct {  
  __REG32  MACA1H         :16;
  __REG32                 : 8;
  __REG32  MBC            : 6;
  __REG32  SA             : 1;
  __REG32  AE             : 1;
} __eth_maca1hr_bits;

/* Ethernet MAC address1 low register (ETH_MACA1LR) */
typedef struct {  
  __REG32  MACA1L         :32;
} __eth_maca1lr_bits;

/* Ethernet MAC address 2 high register (ETH_MACA2HR) */
typedef struct {  
  __REG32  MACA2H         :16;
  __REG32                 : 8;
  __REG32  MBC            : 6;
  __REG32  SA             : 1;
  __REG32  AE             : 1;
} __eth_maca2hr_bits;

/* Ethernet MAC address 2 low register (ETH_MACA2LR) */
typedef struct {  
  __REG32  MACA2L         :32;
} __eth_maca2lr_bits;

/* Ethernet MAC address 3 high register (ETH_MACA3HR) */
typedef struct {  
  __REG32  MACA3H         :16;
  __REG32                 : 8;
  __REG32  MBC            : 6;
  __REG32  SA             : 1;
  __REG32  AE             : 1;
} __eth_maca3hr_bits;

/* Ethernet MAC address 3 low register (ETH_MACA3LR) */
typedef struct {  
  __REG32  MACA3L         :32;
} __eth_maca3lr_bits;

/* Ethernet MMC control register (ETH_MMCCR) */
typedef struct {  
  __REG32  CR             : 1;
  __REG32  CSR            : 1;
  __REG32  ROR            : 1;
  __REG32  MCF            : 1;
  __REG32                 :28;
} __eth_mmccr_bits;

/* Ethernet MMC receive interrupt register (ETH_MMCRIR) */
typedef struct {  
  __REG32                 : 5;
  __REG32  RFCES          : 1;
  __REG32  RFAES          : 1;
  __REG32                 :10;
  __REG32  RGUFS          : 1;
  __REG32                 :14;
} __eth_mmcrir_bits;

/* Ethernet MMC transmit interrupt register (ETH_MMCTIR) */
typedef struct {  
  __REG32                 :14;
  __REG32  TGFSCS         : 1;
  __REG32  TGFMSCS        : 1;
  __REG32                 : 5;
  __REG32  TGFS           : 1;
  __REG32                 :10;
} __eth_mmctir_bits;

/* Ethernet MMC receive interrupt mask register (ETH_MMCRIMR) */
typedef struct {  
  __REG32                 : 5;
  __REG32  RFCEM          : 1;
  __REG32  RFAEM          : 1;
  __REG32                 :10;
  __REG32  RGUFM          : 1;
  __REG32                 :14;
} __eth_mmcrimr_bits;

/* Ethernet MMC transmit interrupt mask register (ETH_MMCTIMR) */
typedef struct {  
  __REG32                 :14;
  __REG32  TGFSCM         : 1;
  __REG32  TGFMSCM        : 1;
  __REG32                 : 5;
  __REG32  TGFM           : 1;
  __REG32                 :10;
} __eth_mmctimr_bits;

/* Ethernet PTP time stamp control register (ETH_PTPTSCR) */
typedef struct {  
  __REG32  TSE            : 1;
  __REG32  TSFCU          : 1;
  __REG32  TSSTI          : 1;
  __REG32  TSSTU          : 1;
  __REG32  TSITE          : 1;
  __REG32  TTSARU         : 1;
  __REG32                 :26;
} __eth_ptptscr_bits;

/* Ethernet PTP subsecond increment register (ETH_PTPSSIR) */
typedef struct {  
  __REG32  STSSI          : 8;
  __REG32                 :24;
} __eth_ptpssir_bits;

/* Ethernet PTP time stamp high register (ETH_PTPTSHR) */
typedef struct {  
  __REG32  STS            :32;
} __eth_ptptshr_bits;

/* Ethernet PTP time stamp low register (ETH_PTPTSLR) */
typedef struct {  
  __REG32  STSS           :31;
  __REG32  STPNS          : 1;
} __eth_ptptslr_bits;

/* Ethernet PTP time stamp high update register (ETH_PTPTSHUR) */
typedef struct {  
  __REG32  TSUS           :32;
} __eth_ptptshur_bits;

/* Ethernet PTP time stamp low update register (ETH_PTPTSLUR) */
typedef struct {  
  __REG32  TSUSS          :31;
  __REG32  TSUPNS         : 1;
} __eth_ptptslur_bits;

/* Ethernet DMA bus mode register (ETH_DMABMR) */
typedef struct {  
  __REG32  SR             : 1;
  __REG32  DA             : 1;
  __REG32  DSL            : 5;
  __REG32                 : 1;
  __REG32  PBL            : 6;
  __REG32  RTPR           : 2;
  __REG32  FB             : 1;
  __REG32  RDP            : 6;
  __REG32  USP            : 1;
  __REG32  FPM            : 1;
  __REG32  AAB            : 1;
  __REG32                 : 6;
} __eth_dmabmr_bits;

/* Ethernet DMA status register (ETH_DMASR) */
typedef struct {  
  __REG32  TS             : 1;
  __REG32  TPSS           : 1;
  __REG32  TBUS           : 1;
  __REG32  TJTS           : 1;
  __REG32  ROS            : 1;
  __REG32  TUS            : 1;
  __REG32  RS             : 1;
  __REG32  RBUS           : 1;
  __REG32  RPSS           : 1;
  __REG32  RWTS           : 1;
  __REG32  ETS            : 1;
  __REG32                 : 2;
  __REG32  FBES           : 1;
  __REG32  ERS            : 1;
  __REG32  AIS            : 1;
  __REG32  NIS            : 1;
  __REG32  RPS            : 3;
  __REG32  TPS            : 3;
  __REG32  EBS            : 3;
  __REG32                 : 1;
  __REG32  MMCS           : 1;
  __REG32  PMTS           : 1;
  __REG32  TSTS           : 1;
  __REG32                 : 2;
} __eth_dmasr_bits;

/* Ethernet DMA operation mode register (ETH_DMAOMR) */
typedef struct {  
  __REG32                 : 1;
  __REG32  SR             : 1;
  __REG32  OSF            : 1;
  __REG32  RTC            : 2;
  __REG32                 : 1;
  __REG32  FUGF           : 1;
  __REG32  FEF            : 1;
  __REG32                 : 5;
  __REG32  ST             : 1;
  __REG32  TTC            : 3;
  __REG32                 : 3;
  __REG32  FTF            : 1;
  __REG32  TSF            : 1;
  __REG32                 : 2;
  __REG32  DFRF           : 1;
  __REG32  RSF            : 1;
  __REG32  DTCEFD         : 1;
  __REG32                 : 5;
} __eth_dmaomr_bits;

/* Ethernet DMA interrupt enable register (ETH_DMAIER) */
typedef struct {  
  __REG32  TIE            : 1;
  __REG32  TPSIE          : 1;
  __REG32  TBUIE          : 1;
  __REG32  TJTIE          : 1;
  __REG32  ROIE           : 1;
  __REG32  TUIE           : 1;
  __REG32  RIE            : 1;
  __REG32  RBUIE          : 1;
  __REG32  RPSIE          : 1;
  __REG32  RWTIE          : 1;
  __REG32  ETIE           : 1;
  __REG32                 : 2;
  __REG32  FBEIE          : 1;
  __REG32  ERIE           : 1;
  __REG32  AISE           : 1;
  __REG32  NISE           : 1;
  __REG32                 :15;
} __eth_dmaier_bits;

/* Ethernet DMA missed frame and buffer overflow counter register */
/* (ETH_DMAMFBOCR) */
typedef struct {  
  __REG32  MFC            :16;
  __REG32  OMFC           : 1;
  __REG32  MFA            :11;
  __REG32  OFOC           : 1;
  __REG32                 : 3;
} __eth_dmamfbocr_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler  **************************/
/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,              0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,        0xE000E010,__READ_WRITE ,__systickcsr_bits);
__IO_REG32_BIT(SYSTICKRVR,        0xE000E014,__READ_WRITE ,__systickrvr_bits);
__IO_REG32_BIT(SYSTICKCVR,        0xE000E018,__READ_WRITE ,__systickcvr_bits);
__IO_REG32_BIT(SYSTICKCALVR,      0xE000E01C,__READ       ,__systickcalvr_bits);
__IO_REG32_BIT(SETENA0,           0xE000E100,__READ_WRITE ,__setena0_bits);
__IO_REG32_BIT(SETENA1,           0xE000E104,__READ_WRITE ,__setena1_bits);
__IO_REG32_BIT(SETENA2,           0xE000E108,__READ_WRITE ,__setena2_bits);
__IO_REG32_BIT(CLRENA0,           0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,           0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(CLRENA2,           0xE000E188,__READ_WRITE ,__clrena2_bits);
__IO_REG32_BIT(SETPEND0,          0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,          0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(SETPEND2,          0xE000E208,__READ_WRITE ,__setpend2_bits);
__IO_REG32_BIT(CLRPEND0,          0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,          0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(CLRPEND2,          0xE000E288,__READ_WRITE ,__clrpend2_bits);
__IO_REG32_BIT(ACTIVE0,           0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,           0xE000E304,__READ       ,__active1_bits);
__IO_REG32_BIT(ACTIVE2,           0xE000E308,__READ       ,__active2_bits);
__IO_REG32_BIT(IP0,               0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,               0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,               0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,               0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,               0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,               0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,               0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,               0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(IP8,               0xE000E420,__READ_WRITE ,__pri8_bits);
__IO_REG32_BIT(IP9,               0xE000E424,__READ_WRITE ,__pri9_bits);
__IO_REG32_BIT(IP10,              0xE000E428,__READ_WRITE ,__pri10_bits);
__IO_REG32_BIT(IP11,              0xE000E42C,__READ_WRITE ,__pri11_bits);
__IO_REG32_BIT(IP12,              0xE000E430,__READ_WRITE ,__pri12_bits);
__IO_REG32_BIT(IP13,              0xE000E434,__READ_WRITE ,__pri13_bits);
__IO_REG32_BIT(IP14,              0xE000E438,__READ_WRITE ,__pri14_bits);
__IO_REG32_BIT(IP15,              0xE000E43C,__READ_WRITE ,__pri15_bits);
__IO_REG32_BIT(IP16,              0xE000E440,__READ_WRITE ,__pri16_bits);
__IO_REG32_BIT(CPUIDBR,           0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,              0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,              0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,             0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,               0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,               0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,             0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,             0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,             0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,             0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,              0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,              0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,              0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,             0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,              0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,              0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** DBG
 **
 ***************************************************************************/
__IO_REG32_BIT(DBGMCU_IDCODE,     0xE0042000,__READ       ,__dbgmcu_idcode_bits);
__IO_REG32_BIT(DBGMCU_CR,         0xE0042004,__READ_WRITE ,__dbgmcu_cr_bits);

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

/***************************************************************************
 **
 ** GPIOA
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_CRL,         0x40010800,__READ_WRITE ,__gpio_crl_bits);
__IO_REG32_BIT(GPIOA_CRH,         0x40010804,__READ_WRITE ,__gpio_crh_bits);
__IO_REG32_BIT(GPIOA_IDR,         0x40010808,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOA_ODR,         0x4001080C,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOA_BSRR,        0x40010810,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOA_BRR,         0x40010814,__WRITE      ,__gpio_brr_bits);
__IO_REG32_BIT(GPIOA_LCKR,        0x40010818,__READ_WRITE ,__gpio_lckr_bits);

/***************************************************************************
 **
 ** GPIOB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOB_CRL,         0x40010C00,__READ_WRITE ,__gpio_crl_bits);
__IO_REG32_BIT(GPIOB_CRH,         0x40010C04,__READ_WRITE ,__gpio_crh_bits);
__IO_REG32_BIT(GPIOB_IDR,         0x40010C08,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOB_ODR,         0x40010C0C,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOB_BSRR,        0x40010C10,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOB_BRR,         0x40010C14,__WRITE      ,__gpio_brr_bits);
__IO_REG32_BIT(GPIOB_LCKR,        0x40010C18,__READ_WRITE ,__gpio_lckr_bits);

/***************************************************************************
 **
 ** GPIOC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOC_CRL,         0x40011000,__READ_WRITE ,__gpio_crl_bits);
__IO_REG32_BIT(GPIOC_CRH,         0x40011004,__READ_WRITE ,__gpio_crh_bits);
__IO_REG32_BIT(GPIOC_IDR,         0x40011008,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOC_ODR,         0x4001100C,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOC_BSRR,        0x40011010,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOC_BRR,         0x40011014,__WRITE      ,__gpio_brr_bits);
__IO_REG32_BIT(GPIOC_LCKR,        0x40011018,__READ_WRITE ,__gpio_lckr_bits);

/***************************************************************************
 **
 ** GPIOD
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOD_CRL,         0x40011400,__READ_WRITE ,__gpio_crl_bits);
__IO_REG32_BIT(GPIOD_CRH,         0x40011404,__READ_WRITE ,__gpio_crh_bits);
__IO_REG32_BIT(GPIOD_IDR,         0x40011408,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOD_ODR,         0x4001140C,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOD_BSRR,        0x40011410,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOD_BRR,         0x40011414,__WRITE      ,__gpio_brr_bits);
__IO_REG32_BIT(GPIOD_LCKR,        0x40011418,__READ_WRITE ,__gpio_lckr_bits);

/***************************************************************************
 **
 ** GPIOE
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOE_CRL,         0x40011800,__READ_WRITE ,__gpio_crl_bits);
__IO_REG32_BIT(GPIOE_CRH,         0x40011804,__READ_WRITE ,__gpio_crh_bits);
__IO_REG32_BIT(GPIOE_IDR,         0x40011808,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOE_ODR,         0x4001180C,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOE_BSRR,        0x40011810,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOE_BRR,         0x40011814,__WRITE      ,__gpio_brr_bits);
__IO_REG32_BIT(GPIOE_LCKR,        0x40011818,__READ_WRITE ,__gpio_lckr_bits);

/***************************************************************************
 **
 ** AFIO
 **
 ***************************************************************************/
__IO_REG32_BIT(AFIO_EVCR,         0x40010000,__READ_WRITE ,__afio_evcr_bits);
__IO_REG32_BIT(AFIO_MAPR,         0x40010004,__READ_WRITE ,__afio_mapr_bits);
__IO_REG32_BIT(AFIO_EXTICR1,      0x40010008,__READ_WRITE ,__afio_exticr1_bits);
__IO_REG32_BIT(AFIO_EXTICR2,      0x4001000C,__READ_WRITE ,__afio_exticr2_bits);
__IO_REG32_BIT(AFIO_EXTICR3,      0x40010010,__READ_WRITE ,__afio_exticr3_bits);
__IO_REG32_BIT(AFIO_EXTICR4,      0x40010014,__READ_WRITE ,__afio_exticr4_bits);

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
__IO_REG32_BIT(DMA_CCR6,          0x4002006C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR6,        0x40020070,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR6,         0x40020074,__READ_WRITE );
__IO_REG32(    DMA_CMAR6,         0x40020078,__READ_WRITE );
__IO_REG32_BIT(DMA_CCR7,          0x40020080,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA_CNDTR7,        0x40020084,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA_CPAR7,         0x40020088,__READ_WRITE );
__IO_REG32(    DMA_CMAR7,         0x4002008C,__READ_WRITE );

/***************************************************************************
 **
 ** DMA2
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA2_ISR,           0x40020400,__READ       ,__dma2_isr_bits);
__IO_REG32_BIT(DMA2_IFCR,          0x40020404,__READ_WRITE ,__dma2_ifcr_bits);
__IO_REG32_BIT(DMA2_CCR1,          0x40020408,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR1,        0x4002040C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR1,         0x40020410,__READ_WRITE );
__IO_REG32(    DMA2_CMAR1,         0x40020414,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR2,          0x4002041C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR2,        0x40020420,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR2,         0x40020424,__READ_WRITE );
__IO_REG32(    DMA2_CMAR2,         0x40020428,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR3,          0x40020430,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR3,        0x40020434,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR3,         0x40020438,__READ_WRITE );
__IO_REG32(    DMA2_CMAR3,         0x4002043C,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR4,          0x40020444,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR4,        0x40020448,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR4,         0x4002044C,__READ_WRITE );
__IO_REG32(    DMA2_CMAR4,         0x40020450,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR5,          0x40020458,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR5,        0x4002045C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR5,         0x40020460,__READ_WRITE );
__IO_REG32(    DMA2_CMAR5,         0x40020464,__READ_WRITE );

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG16_BIT(RTC_CRH,           0x40002800,__READ_WRITE ,__rtccrh_bits);
__IO_REG16_BIT(RTC_CRL,           0x40002804,__READ_WRITE ,__rtccrl_bits);
__IO_REG16_BIT(RTC_PRLH,          0x40002808,__WRITE      ,__rtcprlh_bits);
__IO_REG16(    RTC_PRLL,          0x4000280C,__WRITE      );
__IO_REG16_BIT(RTC_DIVH,          0x40002810,__READ       ,__rtcdivh_bits);
__IO_REG16(    RTC_DIVL,          0x40002814,__READ       );
__IO_REG16(    RTC_CNTH,          0x40002818,__READ_WRITE );
__IO_REG16(    RTC_CNTL,          0x4000281C,__READ_WRITE );
__IO_REG16(    RTC_ALRH,          0x40002820,__WRITE      );
__IO_REG16(    RTC_ALRL,          0x40002824,__WRITE      );

/***************************************************************************
 **
 ** BKP
 **
 ***************************************************************************/
__IO_REG16(    BKP_DR1,           0x40006C04,__READ_WRITE );
__IO_REG16(    BKP_DR2,           0x40006C08,__READ_WRITE );
__IO_REG16(    BKP_DR3,           0x40006C0C,__READ_WRITE );
__IO_REG16(    BKP_DR4,           0x40006C10,__READ_WRITE );
__IO_REG16(    BKP_DR5,           0x40006C14,__READ_WRITE );
__IO_REG16(    BKP_DR6,           0x40006C18,__READ_WRITE );
__IO_REG16(    BKP_DR7,           0x40006C1C,__READ_WRITE );
__IO_REG16(    BKP_DR8,           0x40006C20,__READ_WRITE );
__IO_REG16(    BKP_DR9,           0x40006C24,__READ_WRITE );
__IO_REG16(    BKP_DR10,          0x40006C28,__READ_WRITE );
__IO_REG16_BIT(BKP_RTCCR,         0x40006C2C,__READ_WRITE ,__bkp_rtccr_bits);
__IO_REG16_BIT(BKP_CR,            0x40006C30,__READ_WRITE ,__bkp_cr_bits);
__IO_REG16_BIT(BKP_CSR,           0x40006C34,__READ_WRITE ,__bkp_csr_bits);
__IO_REG16(    BKP_DR11,          0x40006C40,__READ_WRITE );
__IO_REG16(    BKP_DR12,          0x40006C44,__READ_WRITE );
__IO_REG16(    BKP_DR13,          0x40006C48,__READ_WRITE );
__IO_REG16(    BKP_DR14,          0x40006C4C,__READ_WRITE );
__IO_REG16(    BKP_DR15,          0x40006C50,__READ_WRITE );
__IO_REG16(    BKP_DR16,          0x40006C54,__READ_WRITE );
__IO_REG16(    BKP_DR17,          0x40006C58,__READ_WRITE );
__IO_REG16(    BKP_DR18,          0x40006C5C,__READ_WRITE );
__IO_REG16(    BKP_DR19,          0x40006C60,__READ_WRITE );
__IO_REG16(    BKP_DR20,          0x40006C64,__READ_WRITE );
__IO_REG16(    BKP_DR21,          0x40006C68,__READ_WRITE );
__IO_REG16(    BKP_DR22,          0x40006C6C,__READ_WRITE );
__IO_REG16(    BKP_DR23,          0x40006C70,__READ_WRITE );
__IO_REG16(    BKP_DR24,          0x40006C74,__READ_WRITE );
__IO_REG16(    BKP_DR25,          0x40006C78,__READ_WRITE );
__IO_REG16(    BKP_DR26,          0x40006C7C,__READ_WRITE );
__IO_REG16(    BKP_DR27,          0x40006C80,__READ_WRITE );
__IO_REG16(    BKP_DR28,          0x40006C84,__READ_WRITE );
__IO_REG16(    BKP_DR29,          0x40006C88,__READ_WRITE );
__IO_REG16(    BKP_DR30,          0x40006C8C,__READ_WRITE );
__IO_REG16(    BKP_DR31,          0x40006C90,__READ_WRITE );
__IO_REG16(    BKP_DR32,          0x40006C94,__READ_WRITE );
__IO_REG16(    BKP_DR33,          0x40006C98,__READ_WRITE );
__IO_REG16(    BKP_DR34,          0x40006C9C,__READ_WRITE );
__IO_REG16(    BKP_DR35,          0x40006CA0,__READ_WRITE );
__IO_REG16(    BKP_DR36,          0x40006CA4,__READ_WRITE );
__IO_REG16(    BKP_DR37,          0x40006CA8,__READ_WRITE );
__IO_REG16(    BKP_DR38,          0x40006CAC,__READ_WRITE );
__IO_REG16(    BKP_DR39,          0x40006CB0,__READ_WRITE );
__IO_REG16(    BKP_DR40,          0x40006CB4,__READ_WRITE );
__IO_REG16(    BKP_DR41,          0x40006CB8,__READ_WRITE );
__IO_REG16(    BKP_DR42,          0x40006CBC,__READ_WRITE );

/***************************************************************************
 **
 ** IWDG
 **
 ***************************************************************************/
__IO_REG16(    IWDG_KR,           0x40003000,__WRITE      );
__IO_REG16_BIT(IWDG_PR,           0x40003004,__READ_WRITE ,__iwdg_pr_bits);
__IO_REG16_BIT(IWDG_RLR,          0x40003008,__READ_WRITE ,__iwdg_rlr_bits);
__IO_REG16_BIT(IWDG_SR,           0x4000300C,__READ       ,__iwdg_sr_bits);

/***************************************************************************
 **
 ** WWDG
 **
 ***************************************************************************/
__IO_REG16_BIT(WWDG_CR,           0x40002C00,__READ_WRITE ,__wwdg_cr_bits);
__IO_REG16_BIT(WWDG_CFR,          0x40002C04,__READ_WRITE ,__wwdg_cfr_bits);
__IO_REG16_BIT(WWDG_SR,           0x40002C08,__READ_WRITE ,__wwdg_sr_bits);

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
__IO_REG32_BIT(TIM1_EGR,          0x40012C14,__READ_WRITE ,__tim1_egr_bits);
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
__IO_REG32_BIT(TIM2_EGR,          0x40000014,__READ_WRITE ,__tim_egr_bits);
__IO_REG32_BIT(TIM2_CCMR1,        0x40000018,__READ_WRITE ,__tim_ccmr1_bits);
#define TIM2_OCMR1      TIM2_CCMR1
#define TIM2_OCMR1_bit  TIM2_CCMR1_bit
__IO_REG32_BIT(TIM2_CCMR2,        0x4000001C,__READ_WRITE ,__tim_ccmr2_bits);
#define TIM2_OCMR2      TIM2_CCMR2
#define TIM2_OCMR2_bit  TIM2_CCMR2_bit
__IO_REG32_BIT(TIM2_CCER,         0x40000020,__READ_WRITE ,__tim_ccer_bits);
__IO_REG32_BIT(TIM2_CNT,          0x40000024,__READ_WRITE ,__tim_cnt_bits);
__IO_REG32_BIT(TIM2_PSC,          0x40000028,__READ_WRITE ,__tim_psc_bits);
__IO_REG32_BIT(TIM2_ARR,          0x4000002C,__READ_WRITE ,__tim_arr_bits);
__IO_REG32_BIT(TIM2_CCR1,         0x40000034,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM2_CCR2,         0x40000038,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM2_CCR3,         0x4000003C,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM2_CCR4,         0x40000040,__READ_WRITE ,__tim_ccr_bits);
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
__IO_REG32_BIT(TIM3_EGR,          0x40000414,__READ_WRITE ,__tim_egr_bits);
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
 ** TIM4
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM4_CR1,          0x40000800,__READ_WRITE ,__tim_cr1_bits);
__IO_REG32_BIT(TIM4_CR2,          0x40000804,__READ_WRITE ,__tim_cr2_bits);
__IO_REG32_BIT(TIM4_SMCR,         0x40000808,__READ_WRITE ,__tim_smcr_bits);
__IO_REG32_BIT(TIM4_DIER,         0x4000080C,__READ_WRITE ,__tim_dier_bits);
__IO_REG32_BIT(TIM4_SR,           0x40000810,__READ_WRITE ,__tim_sr_bits);
__IO_REG32_BIT(TIM4_EGR,          0x40000814,__READ_WRITE ,__tim_egr_bits);
__IO_REG32_BIT(TIM4_CCMR1,        0x40000818,__READ_WRITE ,__tim_ccmr1_bits);
#define TIM4_OCMR1      TIM4_CCMR1
#define TIM4_OCMR1_bit  TIM4_CCMR1_bit
__IO_REG32_BIT(TIM4_CCMR2,        0x4000081C,__READ_WRITE ,__tim_ccmr2_bits);
#define TIM4_OCMR2      TIM4_CCMR2
#define TIM4_OCMR2_bit  TIM4_CCMR2_bit
__IO_REG32_BIT(TIM4_CCER,         0x40000820,__READ_WRITE ,__tim_ccer_bits);
__IO_REG32_BIT(TIM4_CNT,          0x40000824,__READ_WRITE ,__tim_cnt_bits);
__IO_REG32_BIT(TIM4_PSC,          0x40000828,__READ_WRITE ,__tim_psc_bits);
__IO_REG32_BIT(TIM4_ARR,          0x4000082C,__READ_WRITE ,__tim_arr_bits);
__IO_REG32_BIT(TIM4_CCR1,         0x40000834,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM4_CCR2,         0x40000838,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM4_CCR3,         0x4000083C,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM4_CCR4,         0x40000840,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM4_DCR,          0x40000848,__READ_WRITE ,__tim_dcr_bits);
__IO_REG32_BIT(TIM4_DMAR,         0x4000084C,__READ_WRITE ,__tim_dmar_bits);

/***************************************************************************
 **
 ** TIM5
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM5_CR1,          0x40000C00,__READ_WRITE ,__tim_cr1_bits);
__IO_REG32_BIT(TIM5_CR2,          0x40000C04,__READ_WRITE ,__tim_cr2_bits);
__IO_REG32_BIT(TIM5_SMCR,         0x40000C08,__READ_WRITE ,__tim_smcr_bits);
__IO_REG32_BIT(TIM5_DIER,         0x40000C0C,__READ_WRITE ,__tim_dier_bits);
__IO_REG32_BIT(TIM5_SR,           0x40000C10,__READ_WRITE ,__tim_sr_bits);
__IO_REG32_BIT(TIM5_EGR,          0x40000C14,__READ_WRITE ,__tim_egr_bits);
__IO_REG32_BIT(TIM5_CCMR1,        0x40000C18,__READ_WRITE ,__tim_ccmr1_bits);
#define TIM5_OCMR1      TIM5_CCMR1
#define TIM5_OCMR1_bit  TIM5_CCMR1_bit
__IO_REG32_BIT(TIM5_CCMR2,        0x40000C1C,__READ_WRITE ,__tim_ccmr2_bits);
#define TIM5_OCMR2      TIM5_CCMR2
#define TIM5_OCMR2_bit  TIM5_CCMR2_bit
__IO_REG32_BIT(TIM5_CCER,         0x40000C20,__READ_WRITE ,__tim_ccer_bits);
__IO_REG32_BIT(TIM5_CNT,          0x40000C24,__READ_WRITE ,__tim_cnt_bits);
__IO_REG32_BIT(TIM5_PSC,          0x40000C28,__READ_WRITE ,__tim_psc_bits);
__IO_REG32_BIT(TIM5_ARR,          0x40000C2C,__READ_WRITE ,__tim_arr_bits);
__IO_REG32_BIT(TIM5_CCR1,         0x40000C34,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM5_CCR2,         0x40000C38,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM5_CCR3,         0x40000C3C,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM5_CCR4,         0x40000C40,__READ_WRITE ,__tim_ccr_bits);
__IO_REG32_BIT(TIM5_DCR,          0x40000C48,__READ_WRITE ,__tim_dcr_bits);
__IO_REG32_BIT(TIM5_DMAR,         0x40000C4C,__READ_WRITE ,__tim_dmar_bits);

/***************************************************************************
 **
 ** TIM6
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM6_CR1,          0x40001000,__READ_WRITE ,__tim6_cr1_bits);
__IO_REG32_BIT(TIM6_CR2,          0x40001004,__READ_WRITE ,__tim6_cr2_bits);
__IO_REG32_BIT(TIM6_DIER,         0x4000100C,__READ_WRITE ,__tim6_dier_bits);
__IO_REG32_BIT(TIM6_SR,           0x40001010,__READ_WRITE ,__tim6_sr_bits);
__IO_REG32_BIT(TIM6_EGR,          0x40001014,__READ_WRITE ,__tim6_egr_bits);
__IO_REG32_BIT(TIM6_CNT,          0x40001024,__READ_WRITE ,__tim6_cnt_bits);
__IO_REG32_BIT(TIM6_PSC,          0x40001028,__READ_WRITE ,__tim6_psc_bits);
__IO_REG32_BIT(TIM6_ARR,          0x4000102C,__READ_WRITE ,__tim6_arr_bits);

/***************************************************************************
 **
 ** TIM7
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM7_CR1,          0x40001400,__READ_WRITE ,__tim6_cr1_bits);
__IO_REG32_BIT(TIM7_CR2,          0x40001404,__READ_WRITE ,__tim6_cr2_bits);
__IO_REG32_BIT(TIM7_DIER,         0x4000140C,__READ_WRITE ,__tim6_dier_bits);
__IO_REG32_BIT(TIM7_SR,           0x40001410,__READ_WRITE ,__tim6_sr_bits);
__IO_REG32_BIT(TIM7_EGR,          0x40001414,__READ_WRITE ,__tim6_egr_bits);
__IO_REG32_BIT(TIM7_CNT,          0x40001424,__READ_WRITE ,__tim6_cnt_bits);
__IO_REG32_BIT(TIM7_PSC,          0x40001428,__READ_WRITE ,__tim6_psc_bits);
__IO_REG32_BIT(TIM7_ARR,          0x4000142C,__READ_WRITE ,__tim6_arr_bits);

/***************************************************************************
 **
 ** bxCAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1_MCR,          0x40006400,__READ_WRITE ,__can_mcr_bits);
__IO_REG32_BIT(CAN1_MSR,          0x40006404,__READ_WRITE ,__can_msr_bits);
__IO_REG32_BIT(CAN1_TSR,          0x40006408,__READ_WRITE ,__can_tsr_bits);
__IO_REG32_BIT(CAN1_RF0R,         0x4000640C,__READ_WRITE ,__can_rfr_bits);
__IO_REG32_BIT(CAN1_RF1R,         0x40006410,__READ_WRITE ,__can_rfr_bits);
__IO_REG32_BIT(CAN1_IER,          0x40006414,__READ_WRITE ,__can_ier_bits);
__IO_REG32_BIT(CAN1_ESR,          0x40006418,__READ_WRITE ,__can_esr_bits);
__IO_REG32_BIT(CAN1_BTR,          0x4000641C,__READ_WRITE ,__can_btr_bits);
__IO_REG32_BIT(CAN1_TI0R,         0x40006580,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN1_TDT0R,        0x40006584,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN1_TDL0R,        0x40006588,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN1_TDH0R,        0x4000658c,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN1_TI1R,         0x40006590,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN1_TDT1R,        0x40006594,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN1_TDL1R,        0x40006598,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN1_TDH1R,        0x4000659C,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN1_TI2R,         0x400065A0,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN1_TDT2R,        0x400065A4,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN1_TDL2R,        0x400065A8,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN1_TDH2R,        0x400065AC,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN1_RI0R,         0x400065B0,__READ       ,__can_rir_bits);
__IO_REG32_BIT(CAN1_RDT0R,        0x400065B4,__READ       ,__can_rdtr_bits);
__IO_REG32_BIT(CAN1_RDL0R,        0x400065B8,__READ       ,__can_rdlr_bits);
__IO_REG32_BIT(CAN1_RDH0R,        0x400065BC,__READ       ,__can_rdhr_bits);
__IO_REG32_BIT(CAN1_RI1R,         0x400065C0,__READ       ,__can_rir_bits);
__IO_REG32_BIT(CAN1_RDT1R,        0x400065C4,__READ       ,__can_rdtr_bits);
__IO_REG32_BIT(CAN1_RDL1R,        0x400065C8,__READ       ,__can_rdlr_bits);
__IO_REG32_BIT(CAN1_RDH1R,        0x400065CC,__READ       ,__can_rdhr_bits);
__IO_REG32_BIT(CAN2_MCR,          0x40006800,__READ_WRITE ,__can_mcr_bits);
__IO_REG32_BIT(CAN2_MSR,          0x40006804,__READ_WRITE ,__can_msr_bits);
__IO_REG32_BIT(CAN2_TSR,          0x40006808,__READ_WRITE ,__can_tsr_bits);
__IO_REG32_BIT(CAN2_RF0R,         0x4000680C,__READ_WRITE ,__can_rfr_bits);
__IO_REG32_BIT(CAN2_RF1R,         0x40006810,__READ_WRITE ,__can_rfr_bits);
__IO_REG32_BIT(CAN2_IER,          0x40006814,__READ_WRITE ,__can_ier_bits);
__IO_REG32_BIT(CAN2_ESR,          0x40006818,__READ_WRITE ,__can_esr_bits);
__IO_REG32_BIT(CAN2_BTR,          0x4000681C,__READ_WRITE ,__can_btr_bits);
__IO_REG32_BIT(CAN2_TI0R,         0x40006980,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN2_TDT0R,        0x40006984,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN2_TDL0R,        0x40006988,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN2_TDH0R,        0x4000698c,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN2_TI1R,         0x40006990,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN2_TDT1R,        0x40006994,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN2_TDL1R,        0x40006998,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN2_TDH1R,        0x4000699C,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN2_TI2R,         0x400069A0,__READ_WRITE ,__can_tir_bits);
__IO_REG32_BIT(CAN2_TDT2R,        0x400069A4,__READ_WRITE ,__can_tdtr_bits);
__IO_REG32_BIT(CAN2_TDL2R,        0x400069A8,__READ_WRITE ,__can_tdlr_bits);
__IO_REG32_BIT(CAN2_TDH2R,        0x400069AC,__READ_WRITE ,__can_tdhr_bits);
__IO_REG32_BIT(CAN2_RI0R,         0x400069B0,__READ       ,__can_rir_bits);
__IO_REG32_BIT(CAN2_RDT0R,        0x400069B4,__READ       ,__can_rdtr_bits);
__IO_REG32_BIT(CAN2_RDL0R,        0x400069B8,__READ       ,__can_rdlr_bits);
__IO_REG32_BIT(CAN2_RDH0R,        0x400069BC,__READ       ,__can_rdhr_bits);
__IO_REG32_BIT(CAN2_RI1R,         0x400069C0,__READ       ,__can_rir_bits);
__IO_REG32_BIT(CAN2_RDT1R,        0x400069C4,__READ       ,__can_rdtr_bits);
__IO_REG32_BIT(CAN2_RDL1R,        0x400069C8,__READ       ,__can_rdlr_bits);
__IO_REG32_BIT(CAN2_RDH1R,        0x400069CC,__READ       ,__can_rdhr_bits);
__IO_REG32_BIT(CAN_FMR,           0x40006600,__READ_WRITE ,__can_fmr_bits);
__IO_REG32_BIT(CAN_FM1R,          0x40006604,__READ_WRITE ,__can_fm1r_bits);
__IO_REG32_BIT(CAN_FS1R,          0x4000660C,__READ_WRITE ,__can_fs1r_bits);
__IO_REG32_BIT(CAN_FFA1R,         0x40006614,__READ_WRITE ,__can_ffa1r_bits);
__IO_REG32_BIT(CAN_FA1R,          0x4000661C,__READ_WRITE ,__can_fa1r_bits);
__IO_REG32_BIT(CAN_F0R1,          0x40006640,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F0R2,          0x40006644,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F1R1,          0x40006648,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F1R2,          0x4000664C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F2R1,          0x40006650,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F2R2,          0x40006654,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F3R1,          0x40006658,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F3R2,          0x4000665C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F4R1,          0x40006660,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F4R2,          0x40006664,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F5R1,          0x40006668,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F5R2,          0x4000666C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F6R1,          0x40006670,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F6R2,          0x40006674,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F7R1,          0x40006678,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F7R2,          0x4000667C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F8R1,          0x40006680,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F8R2,          0x40006684,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F9R1,          0x40006688,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F9R2,          0x4000668C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F10R1,         0x40006690,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F10R2,         0x40006694,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F11R1,         0x40006698,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F11R2,         0x4000669C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F12R1,         0x400066A0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F12R2,         0x400066A4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F13R1,         0x400066A8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F13R2,         0x400066AC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F14R1,         0x400066B0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F14R2,         0x400066B4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F15R1,         0x400066B8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F15R2,         0x400066BC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F16R1,         0x400066C0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F16R2,         0x400066C4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F17R1,         0x400066c8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F17R2,         0x400066CC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F18R1,         0x400066D0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F18R2,         0x400066D4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F19R1,         0x400066D8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F19R2,         0x400066DC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F20R1,         0x400066E0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F20R2,         0x400066E4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F21R1,         0x400066E8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F21R2,         0x400066EC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F22R1,         0x400066F0,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F22R2,         0x400066F4,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F23R1,         0x400066F8,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F23R2,         0x400066FC,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F24R1,         0x40006700,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F24R2,         0x40006704,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F25R1,         0x40006708,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F25R2,         0x4000670C,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F26R1,         0x40006710,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F26R2,         0x40006714,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F27R1,         0x40006718,__READ_WRITE ,__can_fr_bits);
__IO_REG32_BIT(CAN_F27R2,         0x4000671C,__READ_WRITE ,__can_fr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1_CR1,          0x40005400,__READ_WRITE ,__i2c_cr1_bits);
__IO_REG32_BIT(I2C1_CR2,          0x40005404,__READ_WRITE ,__i2c_cr2_bits);
__IO_REG32_BIT(I2C1_OAR1,         0x40005408,__READ_WRITE ,__i2c_oar1_bits);
__IO_REG32_BIT(I2C1_OAR2,         0x4000540C,__READ_WRITE ,__i2c_oar2_bits);
__IO_REG32_BIT(I2C1_DR,           0x40005410,__READ_WRITE ,__i2c_dr_bits);
__IO_REG32_BIT(I2C1_SR1,          0x40005414,__READ_WRITE ,__i2c_sr1_bits);
__IO_REG32_BIT(I2C1_SR2,          0x40005418,__READ       ,__i2c_sr2_bits);
__IO_REG32_BIT(I2C1_CCR,          0x4000541C,__READ_WRITE ,__i2c_ccr_bits);
__IO_REG32_BIT(I2C1_TRISE,        0x40005420,__READ_WRITE ,__i2c_trise_bits);

/***************************************************************************
 **
 ** I2C2
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C2_CR1,          0x40005800,__READ_WRITE ,__i2c_cr1_bits);
__IO_REG32_BIT(I2C2_CR2,          0x40005804,__READ_WRITE ,__i2c_cr2_bits);
__IO_REG32_BIT(I2C2_OAR1,         0x40005808,__READ_WRITE ,__i2c_oar1_bits);
__IO_REG32_BIT(I2C2_OAR2,         0x4000580C,__READ_WRITE ,__i2c_oar2_bits);
__IO_REG32_BIT(I2C2_DR,           0x40005810,__READ_WRITE ,__i2c_dr_bits);
__IO_REG32_BIT(I2C2_SR1,          0x40005814,__READ_WRITE ,__i2c_sr1_bits);
__IO_REG32_BIT(I2C2_SR2,          0x40005818,__READ_WRITE ,__i2c_sr2_bits);
__IO_REG32_BIT(I2C2_CCR,          0x4000581C,__READ_WRITE ,__i2c_ccr_bits);
__IO_REG32_BIT(I2C2_TRISE,        0x40005820,__READ_WRITE ,__i2c_trise_bits);

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
__IO_REG32_BIT(SPI2_I2SCFGR,      0x4000381C,__READ_WRITE ,__spi_i2scfgr_bits);
__IO_REG32_BIT(SPI2_I2SPR,        0x40003820,__READ_WRITE ,__spi_i2spr_bits);

/***************************************************************************
 **
 ** SPI3
 **
 ***************************************************************************/
__IO_REG32_BIT(SPI3_CR1,          0x40003C00,__READ_WRITE ,__spi_cr1_bits);
__IO_REG32_BIT(SPI3_CR2,          0x40003C04,__READ_WRITE ,__spi_cr2_bits);
__IO_REG32_BIT(SPI3_SR,           0x40003C08,__READ_WRITE ,__spi_sr_bits);
__IO_REG32_BIT(SPI3_DR,           0x40003C0C,__READ_WRITE ,__spi_dr_bits);
__IO_REG32_BIT(SPI3_CRCPR,        0x40003C10,__READ_WRITE ,__spi_crcpr_bits);
__IO_REG32_BIT(SPI3_RXCRCR,       0x40003C14,__READ       ,__spi_rxcrcr_bits);
__IO_REG32_BIT(SPI3_TXCRCR,       0x40003C18,__READ       ,__spi_txcrcr_bits);
__IO_REG32_BIT(SPI3_I2SCFGR,      0x40003C1C,__READ_WRITE ,__spi_i2scfgr_bits);
__IO_REG32_BIT(SPI3_I2SPR,        0x40003C20,__READ_WRITE ,__spi_i2spr_bits);

/***************************************************************************
 **
 ** USART1
 **
 ***************************************************************************/
__IO_REG32_BIT(USART1_SR,         0x40013800,__READ_WRITE ,__usart_sr_bits);
__IO_REG32_BIT(USART1_DR,         0x40013804,__READ_WRITE ,__usart_dr_bits);
__IO_REG32_BIT(USART1_BRR,        0x40013808,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART1_CR1,        0x4001380C,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART1_CR2,        0x40013810,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART1_CR3,        0x40013814,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART1_GTPR,       0x40013818,__READ_WRITE ,__usart_gtpr_bits);

/***************************************************************************
 **
 ** USART2
 **
 ***************************************************************************/
__IO_REG32_BIT(USART2_SR,         0x40004400,__READ_WRITE ,__usart_sr_bits);
__IO_REG32_BIT(USART2_DR,         0x40004404,__READ_WRITE ,__usart_dr_bits);
__IO_REG32_BIT(USART2_BRR,        0x40004408,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART2_CR1,        0x4000440C,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART2_CR2,        0x40004410,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART2_CR3,        0x40004414,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART2_GTPR,       0x40004418,__READ_WRITE ,__usart_gtpr_bits);

/***************************************************************************
 **
 ** USART3
 **
 ***************************************************************************/
__IO_REG32_BIT(USART3_SR,         0x40004800,__READ_WRITE ,__usart_sr_bits);
__IO_REG32_BIT(USART3_DR,         0x40004804,__READ_WRITE ,__usart_dr_bits);
__IO_REG32_BIT(USART3_BRR,        0x40004808,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART3_CR1,        0x4000480C,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART3_CR2,        0x40004810,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART3_CR3,        0x40004814,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART3_GTPR,       0x40004818,__READ_WRITE ,__usart_gtpr_bits);

/***************************************************************************
 **
 ** UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(UART4_SR,         0x40004C00,__READ_WRITE ,__uart_sr_bits);
__IO_REG32_BIT(UART4_DR,         0x40004C04,__READ_WRITE ,__uart_dr_bits);
__IO_REG32_BIT(UART4_BRR,        0x40004C08,__READ_WRITE ,__uart_brr_bits);
__IO_REG32_BIT(UART4_CR1,        0x40004C0C,__READ_WRITE ,__uart_cr1_bits);
__IO_REG32_BIT(UART4_CR2,        0x40004C10,__READ_WRITE ,__uart_cr2_bits);
__IO_REG32_BIT(UART4_CR3,        0x40004C14,__READ_WRITE ,__uart4_cr3_bits);

/***************************************************************************
 **
 ** UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(UART5_SR,         0x40005000,__READ_WRITE ,__uart_sr_bits);
__IO_REG32_BIT(UART5_DR,         0x40005004,__READ_WRITE ,__uart_dr_bits);
__IO_REG32_BIT(UART5_BRR,        0x40005008,__READ_WRITE ,__uart_brr_bits);
__IO_REG32_BIT(UART5_CR1,        0x4000500C,__READ_WRITE ,__uart_cr1_bits);
__IO_REG32_BIT(UART5_CR2,        0x40005010,__READ_WRITE ,__uart_cr2_bits);
__IO_REG32_BIT(UART5_CR3,        0x40005014,__READ_WRITE ,__uart5_cr3_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1_SR,           0x40012400,__READ_WRITE ,__adc_sr_bits);
__IO_REG32_BIT(ADC1_CR1,          0x40012404,__READ_WRITE ,__adc_cr1_bits);
__IO_REG32_BIT(ADC1_CR2,          0x40012408,__READ_WRITE ,__adc_cr2_bits);
__IO_REG32_BIT(ADC1_SMPR1,        0x4001240C,__READ_WRITE ,__adc_smpr1_bits);
__IO_REG32_BIT(ADC1_SMPR2,        0x40012410,__READ_WRITE ,__adc_smpr2_bits);
__IO_REG32_BIT(ADC1_JOFR1,        0x40012414,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC1_JOFR2,        0x40012418,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC1_JOFR3,        0x4001241C,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC1_JOFR4,        0x40012420,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC1_HTR,          0x40012424,__READ_WRITE ,__adc_htr_bits);
__IO_REG32_BIT(ADC1_LTR,          0x40012428,__READ_WRITE ,__adc_ltr_bits);
__IO_REG32_BIT(ADC1_SQR1,         0x4001242C,__READ_WRITE ,__adc_sqr1_bits);
__IO_REG32_BIT(ADC1_SQR2,         0x40012430,__READ_WRITE ,__adc_sqr2_bits);
__IO_REG32_BIT(ADC1_SQR3,         0x40012434,__READ_WRITE ,__adc_sqr3_bits);
__IO_REG32_BIT(ADC1_JSQR,         0x40012438,__READ_WRITE ,__adc_jsqr_bits);
__IO_REG32_BIT(ADC1_JDR1,         0x4001243C,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC1_JDR2,         0x40012440,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC1_JDR3,         0x40012444,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC1_JDR4,         0x40012448,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC1_DR,           0x4001244C,__READ       ,__adc_dr_bits);

/***************************************************************************
 **
 ** ADC2
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC2_SR,           0x40012800,__READ_WRITE ,__adc_sr_bits);
__IO_REG32_BIT(ADC2_CR1,          0x40012804,__READ_WRITE ,__adc_cr1_bits);
__IO_REG32_BIT(ADC2_CR2,          0x40012808,__READ_WRITE ,__adc_cr2_bits);
__IO_REG32_BIT(ADC2_SMPR1,        0x4001280C,__READ_WRITE ,__adc_smpr1_bits);
__IO_REG32_BIT(ADC2_SMPR2,        0x40012810,__READ_WRITE ,__adc_smpr2_bits);
__IO_REG32_BIT(ADC2_JOFR1,        0x40012814,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC2_JOFR2,        0x40012818,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC2_JOFR3,        0x4001281C,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC2_JOFR4,        0x40012820,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC2_HTR,          0x40012824,__READ_WRITE ,__adc_htr_bits);
__IO_REG32_BIT(ADC2_LTR,          0x40012828,__READ_WRITE ,__adc_ltr_bits);
__IO_REG32_BIT(ADC2_SQR1,         0x4001282C,__READ_WRITE ,__adc_sqr1_bits);
__IO_REG32_BIT(ADC2_SQR2,         0x40012830,__READ_WRITE ,__adc_sqr2_bits);
__IO_REG32_BIT(ADC2_SQR3,         0x40012834,__READ_WRITE ,__adc_sqr3_bits);
__IO_REG32_BIT(ADC2_JSQR,         0x40012838,__READ_WRITE ,__adc_jsqr_bits);
__IO_REG32_BIT(ADC2_JDR1,         0x4001283C,__READ_WRITE ,__adc_jdr_bits);
__IO_REG32_BIT(ADC2_JDR2,         0x40012840,__READ_WRITE ,__adc_jdr_bits);
__IO_REG32_BIT(ADC2_JDR3,         0x40012844,__READ_WRITE ,__adc_jdr_bits);
__IO_REG32_BIT(ADC2_JDR4,         0x40012848,__READ_WRITE ,__adc_jdr_bits);
__IO_REG32_BIT(ADC2_DR,           0x4001284C,__READ_WRITE ,__adc_dr_bits);

/***************************************************************************
 **
 ** DAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DAC_CR,            0x40007400,__READ_WRITE ,__dac_cr_bits     );
__IO_REG32_BIT(DAC_SWTRIGR,       0x40007404,__READ_WRITE ,__dac_swtrigr_bits);
__IO_REG32_BIT(DAC_DHR12R1,       0x40007408,__READ_WRITE ,__dac_dhr12r1_bits);
__IO_REG32_BIT(DAC_DHR12L1,       0x4000740C,__READ_WRITE ,__dac_dhr12l1_bits);
__IO_REG32_BIT(DAC_DHR8R1,        0x40007410,__READ_WRITE ,__dac_dhr8r1_bits );
__IO_REG32_BIT(DAC_DHR12R2,       0x40007414,__READ_WRITE ,__dac_dhr12r2_bits);
__IO_REG32_BIT(DAC_DHR12L2,       0x40007418,__READ_WRITE ,__dac_dhr12l2_bits);
__IO_REG32_BIT(DAC_DHR8R2,        0x4000741C,__READ_WRITE ,__dac_dhr8r2_bits );
__IO_REG32_BIT(DAC_DHR12RD,       0x40007420,__READ_WRITE ,__dac_dhr12rd_bits);
__IO_REG32_BIT(DAC_DHR12LD,       0x40007424,__READ_WRITE ,__dac_dhr12ld_bits);
__IO_REG32_BIT(DAC_DHR8RD,        0x40007428,__READ_WRITE ,__dac_dhr8rd_bits );
__IO_REG32_BIT(DAC_DOR1,          0x4000742C,__READ_WRITE ,__dac_dor1_bits   );
__IO_REG32_BIT(DAC_DOR2,          0x40007430,__READ_WRITE ,__dac_dor2_bits   );

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

/***************************************************************************
 **
 ** OTG_FS
 **
 ***************************************************************************/
__IO_REG32_BIT(OTG_FS_GOTGCTL,    0x50000000,__READ_WRITE ,__otg_fs_gotgctl_bits);
__IO_REG32_BIT(OTG_FS_GOTGINT,    0x50000004,__READ_WRITE ,__otg_fs_gotgint_bits);
__IO_REG32_BIT(OTG_FS_GAHBCFG,    0x50000008,__READ_WRITE ,__otg_fs_gahbcfg_bits);
__IO_REG32_BIT(OTG_FS_GUSBCFG,    0x5000000C,__READ_WRITE ,__otg_fs_gusbcfg_bits);
__IO_REG32_BIT(OTG_FS_GRSTCTL,    0x50000010,__READ_WRITE ,__otg_fs_grstctl_bits);
__IO_REG32_BIT(OTG_FS_GINTSTS,    0x50000014,__READ_WRITE ,__otg_fs_gintsts_bits);
__IO_REG32_BIT(OTG_FS_GINTMSK,    0x50000018,__READ_WRITE ,__otg_fs_gintmsk_bits);
__IO_REG32_BIT(OTG_FS_GRXSTSR,    0x5000001C,__READ       ,__otg_fs_grxstsr_bits);
#define OTG_FS_GRXSTSR_DEV        OTG_FS_GRXSTSR
#define OTG_FS_GRXSTSR_DEV_bit    OTG_FS_GRXSTSR_bit
__IO_REG32_BIT(OTG_FS_GRXSTSP,    0x50000020,__READ       ,__otg_fs_grxstsr_bits);
#define OTG_FS_GRXSTSP_DEV        OTG_FS_GRXSTSP
#define OTG_FS_GRXSTSP_DEV_bit    OTG_FS_GRXSTSP_bit
__IO_REG32_BIT(OTG_FS_GRXFSIZ,    0x50000024,__READ_WRITE ,__otg_fs_grxfsiz_bits);
__IO_REG32_BIT(OTG_FS_GNPTXFSIZ,  0x50000028,__READ_WRITE ,__otg_fs_gnptxfsiz_bits);
__IO_REG32_BIT(OTG_FS_GNPTXSTS,   0x5000002C,__READ       ,__otg_fs_gnptxsts_bits);
__IO_REG32_BIT(OTG_FS_GCCFG,      0x50000038,__READ_WRITE ,__otg_fs_gccfg_bits);
__IO_REG32(    OTG_FS_CID,        0x5000003C,__READ_WRITE );
__IO_REG32_BIT(OTG_FS_HPTXFSIZ,   0x50000100,__READ_WRITE ,__otg_fs_hptxfsiz_bits);
__IO_REG32_BIT(OTG_FS_DIEPTXF1,   0x50000104,__READ_WRITE ,__otg_fs_dieptxfx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTXF2,   0x50000108,__READ_WRITE ,__otg_fs_dieptxfx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTXF3,   0x5000010C,__READ_WRITE ,__otg_fs_dieptxfx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTXF4,   0x50000110,__READ_WRITE ,__otg_fs_dieptxfx_bits);
__IO_REG32_BIT(OTG_FS_HCFG,       0x50000400,__READ_WRITE ,__otg_fs_hcfg_bits);
__IO_REG32_BIT(OTG_FS_HFIR,       0x50000404,__READ_WRITE ,__otg_fs_hfir_bits);
__IO_REG32_BIT(OTG_FS_HFNUM,      0x50000408,__READ_WRITE ,__otg_fs_hfnum_bits);
__IO_REG32_BIT(OTG_FS_HPTXSTS,    0x50000410,__READ_WRITE ,__otg_fs_hptxsts_bits);
__IO_REG32_BIT(OTG_FS_HAINT,      0x50000414,__READ       ,__otg_fs_haint_bits);
__IO_REG32_BIT(OTG_FS_HAINTMSK,   0x50000418,__READ_WRITE ,__otg_fs_haintmsk_bits);
__IO_REG32_BIT(OTG_FS_HPRT,       0x50000440,__READ_WRITE ,__otg_fs_hprt_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR0,    0x50000500,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT0,     0x50000508,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK0,  0x5000050C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ0,    0x50000510,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR1,    0x50000520,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT1,     0x50000528,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK1,  0x5000052C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ1,    0x50000530,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR2,    0x50000540,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT2,     0x50000548,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK2,  0x5000054C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ2,    0x50000550,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR3,    0x50000560,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT3,     0x50000568,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK3,  0x5000056C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ3,    0x50000570,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR4,    0x50000580,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT4,     0x50000588,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK4,  0x5000058C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ4,    0x50000590,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR5,    0x500005A0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT5,     0x500005A8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK5,  0x500005AC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ5,    0x500005B0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR6,    0x500005C0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT6,     0x500005C8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK6,  0x500005CC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ6,    0x500005D0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR7,    0x500005E0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT7,     0x500005E8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK7,  0x500005EC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ7,    0x500005F0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR8,    0x50000600,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT8,     0x50000608,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK8,  0x5000060C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ8,    0x50000610,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR9,    0x50000620,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT9,     0x50000628,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK9,  0x5000062C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ9,    0x50000630,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR10,   0x50000640,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT10,    0x50000648,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK10, 0x5000064C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ10,   0x50000650,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR11,   0x50000660,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT11,    0x50000668,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK11, 0x5000066C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ11,   0x50000670,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR12,   0x50000680,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT12,    0x50000688,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK12, 0x5000068C,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ12,   0x50000690,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR13,   0x500006A0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT13,    0x500006A8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK13, 0x500006AC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ13,   0x500006B0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR14,   0x500006C0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT14,    0x500006C8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK14, 0x500006CC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ14,   0x500006D0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_HCCHAR15,   0x500006E0,__READ_WRITE ,__otg_fs_hccharx_bits);
__IO_REG32_BIT(OTG_FS_HCINT15,    0x500006E8,__READ_WRITE ,__otg_fs_hcintx_bits);
__IO_REG32_BIT(OTG_FS_HCINTMSK15, 0x500006EC,__READ_WRITE ,__otg_fs_hcintmskx_bits);
__IO_REG32_BIT(OTG_FS_HCTSIZ15,   0x500006F0,__READ_WRITE ,__otg_fs_hctsizx_bits);
__IO_REG32_BIT(OTG_FS_DCFG,       0x50000800,__READ_WRITE ,__otg_fs_dcfg_bits);
__IO_REG32_BIT(OTG_FS_DCTL,       0x50000804,__READ_WRITE ,__otg_fs_dctl_bits);
__IO_REG32_BIT(OTG_FS_DSTS,       0x50000808,__READ       ,__otg_fs_dsts_bits);
__IO_REG32_BIT(OTG_FS_DIEPMSK,    0x50000810,__READ_WRITE ,__otg_fs_diepmsk_bits);
__IO_REG32_BIT(OTG_FS_DOEPMSK,    0x50000814,__READ_WRITE ,__otg_fs_doepmsk_bits);
__IO_REG32_BIT(OTG_FS_DAINT,      0x50000818,__READ       ,__otg_fs_daint_bits);
__IO_REG32_BIT(OTG_FS_DAINTMSK,   0x5000081C,__READ_WRITE ,__otg_fs_daintmsk_bits);
__IO_REG32_BIT(OTG_FS_DVBUSDIS,   0x50000828,__READ_WRITE ,__otg_fs_dvbusdis_bits);
__IO_REG32_BIT(OTG_FS_DVBUSPULSE, 0x5000082C,__READ_WRITE ,__otg_fs_dvbuspulse_bits);
__IO_REG32_BIT(OTG_FS_DIEPEMPMSK, 0x50000834,__READ_WRITE ,__otg_fs_diepempmsk_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL0,   0x50000900,__READ_WRITE ,__otg_fs_diepctl0_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT0,   0x50000908,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ0,  0x50000910,__READ_WRITE ,__otg_fs_dieptsiz0_bits);
__IO_REG32_BIT(OTG_FS_DTXFSTS0,   0x50000918,__READ       ,__otg_fs_dtxfstsx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL1,   0x50000920,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT1,   0x50000928,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ1,  0x50000930,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DTXFSTS1,   0x50000938,__READ       ,__otg_fs_dtxfstsx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL2,   0x50000940,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT2,   0x50000948,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ2,  0x50000950,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DTXFSTS2,   0x50000958,__READ       ,__otg_fs_dtxfstsx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL3,   0x50000960,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT3,   0x50000968,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ3,  0x50000970,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DTXFSTS3,   0x50000978,__READ       ,__otg_fs_dtxfstsx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL4,   0x50000980,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT4,   0x50000988,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ4,  0x50000990,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DTXFSTS4,   0x50000998,__READ       ,__otg_fs_dtxfstsx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL5,   0x500009A0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT5,   0x500009A8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ5,  0x500009B0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL6,   0x500009C0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT6,   0x500009C8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ6,  0x500009D0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL7,   0x500009E0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT7,   0x500009E8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ7,  0x500009F0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL8,   0x50000A00,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT8,   0x50000A08,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ8,  0x50000A10,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL9,   0x50000A20,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT9,   0x50000A28,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ9,  0x50000A30,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL10,  0x50000A40,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT10,  0x50000A48,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ10, 0x50000A50,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL11,  0x50000A60,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT11,  0x50000A68,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ11, 0x50000A70,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL12,  0x50000A80,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT12,  0x50000A88,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ12, 0x50000A90,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL13,  0x50000AA0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT13,  0x50000AA8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ13, 0x50000AB0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL14,  0x50000AC0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT14,  0x50000AC8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ14, 0x50000AD0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DIEPCTL15,  0x50000AE0,__READ_WRITE ,__otg_fs_diepctlx_bits);
__IO_REG32_BIT(OTG_FS_DIEPINT15,  0x50000AE8,__READ_WRITE ,__otg_fs_diepintx_bits);
__IO_REG32_BIT(OTG_FS_DIEPTSIZ15, 0x50000AF0,__READ_WRITE ,__otg_fs_dieptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL0,   0x50000B00,__READ_WRITE ,__otg_fs_doepctl0_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT0,   0x50000B08,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ0,  0x50000B10,__READ_WRITE ,__otg_fs_doeptsiz0_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL1,   0x50000B20,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT1,   0x50000B28,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ1,  0x50000B30,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL2,   0x50000B40,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT2,   0x50000B48,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ2,  0x50000B50,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL3,   0x50000B60,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT3,   0x50000B68,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ3,  0x50000B70,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL4,   0x50000B80,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT4,   0x50000B88,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ4,  0x50000B90,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL5,   0x50000BA0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT5,   0x50000BA8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ5,  0x50000BB0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL6,   0x50000BC0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT6,   0x50000BC8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ6,  0x50000BD0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL7,   0x50000BE0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT7,   0x50000BE8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ7,  0x50000BF0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL8,   0x50000C00,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT8,   0x50000C08,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ8,  0x50000C10,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL9,   0x50000C20,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT9,   0x50000C28,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ9,  0x50000C30,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL10,  0x50000C40,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT10,  0x50000C48,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ10, 0x50000C50,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL11,  0x50000C60,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT11,  0x50000C68,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ11, 0x50000C70,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL12,  0x50000C80,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT12,  0x50000C88,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ12, 0x50000C90,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL13,  0x50000CA0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT13,  0x50000CA8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ13, 0x50000CB0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL14,  0x50000CC0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT14,  0x50000CC8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ14, 0x50000CD0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_DOEPCTL15,  0x50000CE0,__READ_WRITE ,__otg_fs_doepctlx_bits);
__IO_REG32_BIT(OTG_FS_DOEPINT15,  0x50000CE8,__READ_WRITE ,__otg_fs_doepintx_bits);
__IO_REG32_BIT(OTG_FS_DOEPTSIZ15, 0x50000CF0,__READ_WRITE ,__otg_fs_doeptsizx_bits);
__IO_REG32_BIT(OTG_FS_PCGCCTL,    0x50000E00,__READ_WRITE ,__otg_fs_pcgcctl_bits);

/***************************************************************************
 **
 ** Ethernet
 **
 ***************************************************************************/
__IO_REG32_BIT(ETH_MACCR,         0x40028000,__READ_WRITE ,__eth_maccr_bits);
__IO_REG32_BIT(ETH_MACFFR,        0x40028004,__READ_WRITE ,__eth_macffr_bits);
__IO_REG32(    ETH_MACHTHR,       0x40028008,__READ_WRITE );
__IO_REG32(    ETH_MACHTLR,       0x4002800C,__READ_WRITE );
__IO_REG32_BIT(ETH_MACMIIAR,      0x40028010,__READ_WRITE ,__eth_macmiiar_bits);
__IO_REG32_BIT(ETH_MACMIIDR,      0x40028014,__READ_WRITE ,__eth_macmiidr_bits);
__IO_REG32_BIT(ETH_MACFCR,        0x40028018,__READ_WRITE ,__eth_macfcr_bits);
__IO_REG32_BIT(ETH_MACVLANTR,     0x4002801C,__READ_WRITE ,__eth_macvlantr_bits);
__IO_REG32(    ETH_MACRWUFFR,     0x40028028,__READ_WRITE );
__IO_REG32_BIT(ETH_MACPMTCSR,     0x4002802C,__READ_WRITE ,__eth_macpmtcsr_bits);
__IO_REG32_BIT(ETH_MACSR,         0x40028038,__READ       ,__eth_macsr_bits);
__IO_REG32_BIT(ETH_MACIMR,        0x4002803C,__READ_WRITE ,__eth_macimr_bits);
__IO_REG32_BIT(ETH_MACA0HR,       0x40028040,__READ_WRITE ,__eth_maca0hr_bits);
__IO_REG32_BIT(ETH_MACA0LR,       0x40028044,__READ_WRITE ,__eth_maca0lr_bits);
__IO_REG32_BIT(ETH_MACA1HR,       0x40028048,__READ_WRITE ,__eth_maca1hr_bits);
__IO_REG32_BIT(ETH_MACA1LR,       0x4002804C,__READ_WRITE ,__eth_maca1lr_bits);
__IO_REG32_BIT(ETH_MACA2HR,       0x40028050,__READ_WRITE ,__eth_maca2hr_bits);
__IO_REG32_BIT(ETH_MACA2LR,       0x40028054,__READ_WRITE ,__eth_maca2lr_bits);
__IO_REG32_BIT(ETH_MACA3HR,       0x40028058,__READ_WRITE ,__eth_maca3hr_bits);
__IO_REG32_BIT(ETH_MACA3LR,       0x4002805C,__READ_WRITE ,__eth_maca3lr_bits);
__IO_REG32_BIT(ETH_MMCCR,         0x40028100,__READ_WRITE ,__eth_mmccr_bits);
__IO_REG32_BIT(ETH_MMCRIR,        0x40028104,__READ       ,__eth_mmcrir_bits);
__IO_REG32_BIT(ETH_MMCTIR,        0x40028108,__READ       ,__eth_mmctir_bits);
__IO_REG32_BIT(ETH_MMCRIMR,       0x4002810C,__READ_WRITE ,__eth_mmcrimr_bits);
__IO_REG32_BIT(ETH_MMCTIMR,       0x40028110,__READ_WRITE ,__eth_mmctimr_bits);
__IO_REG32(    ETH_MMCTGFSCCR,    0x4002814C,__READ       );
__IO_REG32(    ETH_MMCTGFMSCCR,   0x40028150,__READ       );
__IO_REG32(    ETH_MMCTGFCR,      0x40028168,__READ       );
__IO_REG32(    ETH_MMCRFCECR,     0x40028194,__READ       );
__IO_REG32(    ETH_MMCRFAECR,     0x40028198,__READ       );
__IO_REG32(    ETH_MMCRGUFCR,     0x400281C4,__READ       );
__IO_REG32_BIT(ETH_PTPTSCR,       0x40028700,__READ_WRITE ,__eth_ptptscr_bits);
__IO_REG32_BIT(ETH_PTPSSIR,       0x40028704,__READ_WRITE ,__eth_ptpssir_bits);
__IO_REG32_BIT(ETH_PTPTSHR,       0x40028708,__READ       ,__eth_ptptshr_bits);
__IO_REG32_BIT(ETH_PTPTSLR,       0x4002870C,__READ       ,__eth_ptptslr_bits);
__IO_REG32_BIT(ETH_PTPTSHUR,      0x40028710,__READ_WRITE ,__eth_ptptshur_bits);
__IO_REG32_BIT(ETH_PTPTSLUR,      0x40028714,__READ_WRITE ,__eth_ptptslur_bits);
__IO_REG32(    ETH_PTPTSAR,       0x40028718,__READ_WRITE );
__IO_REG32(    ETH_PTPTTHR,       0x4002871C,__READ_WRITE );
__IO_REG32(    ETH_PTPTTLR,       0x40028720,__READ_WRITE );
__IO_REG32_BIT(ETH_DMABMR,        0x40029000,__READ_WRITE ,__eth_dmabmr_bits);
__IO_REG32(    ETH_DMATPDR,       0x40029004,__READ_WRITE );
__IO_REG32(    ETH_DMARPDR,       0x40029008,__READ_WRITE );
__IO_REG32(    ETH_DMARDLAR,      0x4002900C,__READ_WRITE );
__IO_REG32(    ETH_DMATDLAR,      0x40029010,__READ_WRITE );
__IO_REG32_BIT(ETH_DMASR,         0x40029014,__READ_WRITE ,__eth_dmasr_bits);
__IO_REG32_BIT(ETH_DMAOMR,        0x40029018,__READ_WRITE ,__eth_dmaomr_bits);
__IO_REG32_BIT(ETH_DMAIER,        0x4002901C,__READ_WRITE ,__eth_dmaier_bits);
__IO_REG32_BIT(ETH_DMAMFBOCR,     0x40029020,__READ_WRITE ,__eth_dmamfbocr_bits);
__IO_REG32(    ETH_DMACHTDR,      0x40029048,__READ       );
__IO_REG32(    ETH_DMACHRDR,      0x4002904C,__READ       );
__IO_REG32(    ETH_DMACHTBAR,     0x40029050,__READ       );
__IO_REG32(    ETH_DMACHRBAR,     0x40029054,__READ       );


/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  STM32F107xx Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK             0          /* Main Stack                   */
#define RESETI                 1          /* Reset                        */
#define NMII                   2          /* Non-maskable Interrupt       */
#define HFI                    3          /* Hard Fault                   */
#define MMI                    4          /* Memory Management            */
#define BFI                    5          /* Bus Fault                    */
#define UFI                    6          /* Usage Fault                  */
#define SVCI                  11          /* SVCall                       */
#define DMI                   12          /* Debug Monitor                */
#define PSI                   14          /* PendSV                       */
#define STI                   15          /* SysTick                      */
#define WWDG                  16          /* Window Watchdog interrupt    */
#define NVIC_PVD              17          /* PVD through EXTI Line detection interrupt*/
#define NVIC_TAMPER           18          /* Tamper interrupt             */
#define NVIC_RTC              19          /* RTC global interrupt         */
#define NVIC_FLASH            20          /* Flash global interrupt       */
#define NVIC_RCC              21          /* RCC global interrupt         */
#define NVIC_EXTI0            22          /* EXTI Line0 interrupt         */
#define NVIC_EXTI1            23          /* EXTI Line1 interrupt         */
#define NVIC_EXTI2            24          /* EXTI Line2 interrupt         */
#define NVIC_EXTI3            25          /* EXTI Line3 interrupt         */
#define NVIC_EXTI4            26          /* EXTI Line4 interrupt         */
#define NVIC_DMA_CH1          27          /* DMA Channel1 global interrupt*/
#define NVIC_DMA_CH2          28          /* DMA Channel2 global interrupt*/
#define NVIC_DMA_CH3          29          /* DMA Channel3 global interrupt*/
#define NVIC_DMA_CH4          30          /* DMA Channel4 global interrupt*/
#define NVIC_DMA_CH5          31          /* DMA Channel5 global interrupt*/
#define NVIC_DMA_CH6          32          /* DMA Channel6 global interrupt*/
#define NVIC_DMA_CH7          33          /* DMA Channel7 global interrupt*/
#define NVIC_ADC1_2           34          /* ADC global interrupt         */
#define NVIC_CAN1_TX          35          /* CAN1 TX interrupt */
#define NVIC_CAN1_RX0         36          /* CAN1 RX0 interrupt */
#define NVIC_CAN1_RX1         37          /* CAN1 RX1 interrupt            */
#define NVIC_CAN1_SCE         38          /* CAN1 SCE interrupt            */
#define NVIC_EXTI9_5          39          /* EXTI Line[9:5] interrupts    */
#define NVIC_TIM1_BRK         40          /* TIM1 Break interrupt         */
#define NVIC_TIM1_UP          41          /* TIM1 Update interrupt        */
#define NVIC_TIM1_TRG_COM     42          /* TIM1 Trigger and Commutation interrupts */
#define NVIC_TIM1_CC          43          /* TIM1 Capture Compare interrupt */
#define NVIC_TIM2             44          /* TIM2 global interrupt        */
#define NVIC_TIM3             45          /* TIM3 global interrupt        */
#define NVIC_TIM4             46          /* TIM4 global interrupt        */
#define NVIC_I2C1_EV          47          /* I2C1 event interrupt         */
#define NVIC_I2C1_ER          48          /* I2C1 error interrupt         */
#define NVIC_I2C2_EV          49          /* I2C2 event interrupt         */
#define NVIC_I2C2_ER          50          /* I2C2 error interrupt         */
#define NVIC_SPI1             51          /* SPI1 global interrupt        */
#define NVIC_SPI2             52          /* SPI2 global interrupt        */
#define NVIC_USART1           53          /* USART1 global interrupt      */
#define NVIC_USART2           54          /* USART2 global interrupt      */
#define NVIC_USART3           55          /* USART3 global interrupt      */
#define NVIC_EXTI15_10        56          /* EXTI Line[15:10] interrupts  */
#define NVIC_RTC_ALARM        57          /* RTC alarm through EXTI line interrupt */
#define NVIC_OTG_FS_WKUP      58          /* USB On-The-Go FS Wakeup through EXTI line interrupt */
#define NVIC_TIM5             66          /* TIM5 global interrupt        */
#define NVIC_SPI3             67          /* SPI3 global interrupt        */
#define NVIC_UART4            68          /* UART4 global interrupt       */
#define NVIC_UART5            69          /* UART5 global interrupt       */
#define NVIC_TIM6             70          /* TIM6 global interrupt        */
#define NVIC_TIM7             71          /* TIM7 global interrupt        */
#define NVIC_DMA2_CH1         72          /* DMA2 Channel1 global interrupt*/
#define NVIC_DMA2_CH2         73          /* DMA2 Channel2 global interrupt*/
#define NVIC_DMA2_CH3         74          /* DMA2 Channel3 global interrupt*/
#define NVIC_DMA2_CH4         75          /* DMA2 Channel4 global interrupt*/
#define NVIC_DMA2_CH5         76          /* DMA2 Channel5 global interrupt*/
#define NVIC_ETH              77          /* Ethernet global interrupt     */
#define NVIC_ETH_WKUP         78          /* Ethernet Wakeup through EXTI line interrupt*/
#define NVIC_CAN2_TX          79          /* CAN2 TX interrupt */
#define NVIC_CAN2_RX0         80          /* CAN2 RX0 interrupt */
#define NVIC_CAN2_RX1         81          /* CAN2 RX1 interrupt            */
#define NVIC_CAN2_SCE         82          /* CAN2 SCE interrupt            */
#define NVIC_OTG_FS           83          /* USB On The Go FS global interrupt */

#endif    /* __IOSTM32F107xx_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI            0x08
Interrupt1   = HardFault      0x0C
Interrupt2   = MemManage      0x10
Interrupt3   = BusFault       0x14
Interrupt4   = UsageFault     0x18
Interrupt5   = SVC            0x2C
Interrupt6   = DebugMon       0x30
Interrupt7   = PendSV         0x38
Interrupt8   = SysTick        0x3C
Interrupt9   = WWDG           0x40
Interrupt10  = PVD            0x44
Interrupt11  = TAMPER         0x48
Interrupt12  = RTC            0x4C
Interrupt13  = FLASH          0x50
Interrupt14  = RCC            0x54
Interrupt15  = EXTI0          0x58
Interrupt16  = EXTI1          0x5C
Interrupt17  = EXTI2          0x60
Interrupt18  = EXTI3          0x64
Interrupt19  = EXTI4          0x68
Interrupt20  = DMA1Ch1        0x6C
Interrupt21  = DMA1Ch2        0x70
Interrupt22  = DMA1Ch3        0x74
Interrupt23  = DMA1Ch4        0x78
Interrupt24  = DMA1Ch5        0x7C
Interrupt25  = DMA1Ch6        0x80
Interrupt26  = DMA1Ch7        0x84
Interrupt27  = ADC1_2         0x88
Interrupt28  = CAN1_TX        0x8C
Interrupt29  = CAN1_RX0       0x90
Interrupt30  = CAN1_RX1       0x94
Interrupt31  = CAN1_SCE       0x98
Interrupt32  = EXTI9_5        0x9C
Interrupt33  = TIM1_BRK       0xA0
Interrupt34  = TIM1_UP        0xA4
Interrupt35  = TIM1_TRG_COM   0xA8
Interrupt36  = TIM1_CC        0xAC
Interrupt37  = TIM2           0xB0
Interrupt38  = TIM3           0xB4
Interrupt39  = TIM4           0xB8
Interrupt40  = I2C1_EV        0xBC
Interrupt41  = I2C1_ER        0xC0
Interrupt42  = I2C2_EV        0xC4
Interrupt43  = I2C2_ER        0xC8
Interrupt44  = SPI1           0xCC
Interrupt45  = SPI2           0xD0
Interrupt46  = USART1         0xD4
Interrupt47  = USART2         0xD8
Interrupt48  = USART3         0xDC
Interrupt49  = EXTI15_10      0xE0
Interrupt50  = RTCAlarm       0xE4
Interrupt51  = OTG_FSWakeup   0xE8
Interrupt52  = TIM5           0x108
Interrupt53  = SPI3           0x10C
Interrupt54  = UART4          0x110
Interrupt55  = UART5          0x114
Interrupt56  = TIM6           0x118
Interrupt57  = TIM7           0x11C
Interrupt58  = DMA2Ch1        0x120
Interrupt59  = DMA2Ch2        0x124
Interrupt60  = DMA2Ch3        0x128
Interrupt61  = DMA2Ch4        0x12C
Interrupt62  = DMA2Ch5        0x130
Interrupt63  = ETH            0x134
Interrupt64  = ETHWakeup      0x138
Interrupt65  = CAN2_TX        0x13C
Interrupt66  = CAN2_RX0       0x140
Interrupt67  = CAN2_RX1       0x144
Interrupt68  = CAN2_SCE       0x148
Interrupt69  = OTG_FS         0x14C
 
###DDF-INTERRUPT-END###*/
