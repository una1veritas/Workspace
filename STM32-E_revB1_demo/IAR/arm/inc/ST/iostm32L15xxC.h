/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    ST STM32L15xxC
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2011
 **
 **    $Revision: 52044 $
 **
 ***************************************************************************/

#ifndef __IOSTM32L15xxC_H
#define __IOSTM32L15xxC_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   STM32L15xxC SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/


/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* Flash Access Control Register (FLASH_ACR) */
typedef struct {
  __REG32  LATENCY        : 1;
  __REG32  PRFTEN         : 1;
  __REG32  ACC64          : 1;
  __REG32  SLEEP_PD       : 1;
  __REG32  RUN_PD         : 1;
  __REG32                 :27;
} __flash_acr_bits;

/* Flash program erase control register (FLASH_PECR) */
typedef struct {
  __REG32  PELOCK         : 1;
  __REG32  PRGLOCK        : 1;
  __REG32  OPTLOCK        : 1;
  __REG32  PROG           : 1;
  __REG32  DATA           : 1;
  __REG32                 : 3;
  __REG32  FTDW           : 1;
  __REG32  ERASE          : 1;
  __REG32  FPRG           : 1;
  __REG32                 : 5;
  __REG32  EOPIE          : 1;
  __REG32  ERRIE          : 1;
  __REG32  OBL_LAUNCH     : 1;
  __REG32                 :13;
} __flash_pecr_bits;

/* Flash status register (FLASH_SR) */
typedef struct {
  __REG32  BSY            : 1;
  __REG32  EOP            : 1;
  __REG32  ENDHV          : 1;
  __REG32  READY          : 1;
  __REG32                 : 4;
  __REG32  WRPERR         : 1;
  __REG32  PGAERR         : 1;
  __REG32  SIZERR         : 1;
  __REG32  OPTVERR        : 1;
  __REG32                 :20;
} __flash_sr_bits;

/* Option byte register (FLASH_OBR) */
typedef struct {
  __REG32  RDPRT          : 8;
  __REG32                 : 8;
  __REG32  BOR_LEV        : 4;
  __REG32  IWDG_SW        : 1;
  __REG32  nRST_STOP      : 1;
  __REG32  nRST_STDBY     : 1;
  __REG32                 : 9;
} __flash_obr_bits;

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

/* Power control register (PWR_CR) */
typedef struct {
  __REG32  LPSDSR         : 1;
  __REG32  PDDS           : 1;
  __REG32  CWUF           : 1;
  __REG32  CSBF           : 1;
  __REG32  PVDE           : 1;
  __REG32  PLS            : 3;
  __REG32  DBP            : 1;
  __REG32  ULP            : 1;
  __REG32  FWU            : 1;
  __REG32  VOS            : 2;
  __REG32                 : 1;
  __REG32  LPRUN          : 1;
  __REG32                 :17;
} __pwr_cr_bits;

/* Power control/status register (PWR_CSR) */
typedef struct {
  __REG32  WUF            : 1;
  __REG32  SBF            : 1;
  __REG32  PVDO           : 1;
  __REG32  VREFINTRDYF    : 1;
  __REG32  VOSF           : 1;
  __REG32  REGLPF         : 1;
  __REG32                 : 2;
  __REG32  EWUP1          : 1;
  __REG32  EWUP2          : 1;
  __REG32  EWUP3          : 1;
  __REG32                 :21;
} __pwr_csr_bits;

/* Clock control register (RCC_CR) */
typedef struct {
  __REG32  HSION          : 1;
  __REG32  HSIRDY         : 1;
  __REG32                 : 6;
  __REG32  MSION          : 1;
  __REG32  MSIRDY         : 1;
  __REG32                 : 6;
  __REG32  HSEON          : 1;
  __REG32  HSERDY         : 1;
  __REG32  HSEBYP         : 1;
  __REG32                 : 5;
  __REG32  PLLON          : 1;
  __REG32  PLLRDY         : 1;
  __REG32                 : 2;
  __REG32  CSSON          : 1;
  __REG32  RTCPRE         : 2;
  __REG32                 : 1;
} __rcc_cr_bits;

/* Internal Clock Sources Calibration Register (RCC_ICSCR) */
typedef struct {
  __REG32  HSICAL         : 8;
  __REG32  HSITRIM        : 5;
  __REG32  MSIRANGE       : 3;
  __REG32  MSICAL         : 8;
  __REG32  MSITRIM        : 8;
} __rcc_icscr_bits;

/* Clock configuration register (RCC_CFGR) */
typedef struct {
  __REG32  SW             : 2;
  __REG32  SWS            : 2;
  __REG32  HPRE           : 4;
  __REG32  PPRE1          : 3;
  __REG32  PPRE2          : 3;
  __REG32                 : 2;
  __REG32  PLLSRC         : 1;
  __REG32                 : 1;
  __REG32  PLLMUL         : 4;
  __REG32  PLLDIV         : 2;
  __REG32  MCOSEL         : 3;
  __REG32                 : 1;
  __REG32  MCOPRE         : 3;
  __REG32                 : 1;
} __rcc_cfgr_bits;

/* Clock interrupt register (RCC_CIR) */
typedef struct {
  __REG32  LSIRDYF        : 1;
  __REG32  LSERDYF        : 1;
  __REG32  HSIRDYF        : 1;
  __REG32  HSERDYF        : 1;
  __REG32  PLLRDYF        : 1;
  __REG32  MSIRDYF        : 1;
  __REG32                 : 1;
  __REG32  CSSF           : 1;
  __REG32  LSIRDYIE       : 1;
  __REG32  LSERDYIE       : 1;
  __REG32  HSIRDYIE       : 1;
  __REG32  HSERDYIE       : 1;
  __REG32  PLLRDYIE       : 1;
  __REG32  MSIRDYIE       : 1;
  __REG32                 : 2;
  __REG32  LSIRDYC        : 1;
  __REG32  LSERDYC        : 1;
  __REG32  HSIRDYC        : 1;
  __REG32  HSERDYC        : 1;
  __REG32  PLLRDYC        : 1;
  __REG32  MSIRDYC        : 1;
  __REG32                 : 1;
  __REG32  CSSC           : 1;
  __REG32                 : 8;
} __rcc_cir_bits;

/* AHB Peripheral reset register (RCC_AHBRSTR)  */
typedef struct {
  __REG32  GPIOARST       : 1;
  __REG32  GPIOBRST       : 1;
  __REG32  GPIOCRST       : 1;
  __REG32  GPIODRST       : 1;
  __REG32  GPIOERST       : 1;
  __REG32  GPIOHRST       : 1;
  __REG32  GPIOFRST       : 1;
  __REG32  GPIOGRST       : 1;
  __REG32                 : 4;
  __REG32  CRCRST         : 1;
  __REG32                 : 2;
  __REG32  FLITFRST       : 1;
  __REG32                 : 8;
  __REG32  DMA1RST        : 1;
  __REG32  DMA2RST        : 1;
  __REG32                 : 4;
  __REG32  FSMCRST        : 1;
  __REG32                 : 1;
} __rcc_ahbrstr_bits;

/* APB2 Peripheral reset register (RCC_APB2RSTR) */
typedef struct {
  __REG32  SYSCFGRST      : 1;
  __REG32                 : 1;
  __REG32  TIM9RST        : 1;
  __REG32  TIM10RST       : 1;
  __REG32  TIM11RST       : 1;
  __REG32                 : 4;
  __REG32  ADC1RST        : 1;
  __REG32                 : 2;
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
  __REG32                 : 3;
  __REG32  LCDRST         : 1;
  __REG32                 : 1;
  __REG32  WWDGRST        : 1;
  __REG32                 : 2;
  __REG32  SPI2RST        : 1;
  __REG32  SPI3RST        : 1;
  __REG32                 : 1;
  __REG32  USART2RST      : 1;
  __REG32  USART3RST      : 1;
  __REG32  USART4RST      : 1;
  __REG32  USART5RST      : 1;
  __REG32  I2C1RST        : 1;
  __REG32  I2C2RST        : 1;
  __REG32  USBRST         : 1;
  __REG32                 : 4;
  __REG32  PWRRST         : 1;
  __REG32  DACRST         : 1;
  __REG32                 : 1;
  __REG32  COMPRST        : 1;
} __rcc_apb1rstr_bits;

/* AHB Peripheral Clock enable register (RCC_AHBENR) */
typedef struct {
  __REG32  GPIOAEN        : 1;
  __REG32  GPIOBEN        : 1;
  __REG32  GPIOCEN        : 1;
  __REG32  GPIODEN        : 1;
  __REG32  GPIOEEN        : 1;
  __REG32  GPIOHEN        : 1;
  __REG32  GPIOFEN        : 1;
  __REG32  GPIOGEN        : 1;
  __REG32                 : 4;
  __REG32  CRCEN          : 1;
  __REG32                 : 2;
  __REG32  FLITFEN        : 1;
  __REG32                 : 8;
  __REG32  DMA1EN         : 1;
  __REG32  DMA2EN         : 1;
  __REG32                 : 6;
} __rcc_ahbenr_bits;

/* APB2 Peripheral Clock enable register (RCC_APB2ENR) */
typedef struct {
  __REG32  SYSCFGEN       : 1;
  __REG32                 : 1;
  __REG32  TIM9EN         : 1;
  __REG32  TIM10EN        : 1;
  __REG32  TIM11EN        : 1;
  __REG32                 : 4;
  __REG32  ADC1EN         : 1;
  __REG32                 : 1;
  __REG32  SDIOEN         : 1;
  __REG32  SPI1EN         : 1;
  __REG32                 : 1;
  __REG32  USART1EN       : 1;
  __REG32                 :17;
} __rcc_apb2enr_bits;

/* APB1 Peripheral Clock enable register (RCC_APB1ENR) */
typedef struct {
  __REG32  TIM2EN         : 1;
  __REG32  TIM3EN         : 1;
  __REG32  TIM4EN         : 1;
  __REG32  TIM5EN         : 1;
  __REG32  TIM6EN         : 1;
  __REG32  TIM7EN         : 1;
  __REG32                 : 3;
  __REG32  LCDEN          : 1;
  __REG32                 : 1;
  __REG32  WWDGEN         : 1;
  __REG32                 : 2;
  __REG32  SPI2EN         : 1;
  __REG32  SPI3EN         : 1;
  __REG32                 : 1;
  __REG32  USART2EN       : 1;
  __REG32  USART3EN       : 1;
  __REG32  USART4EN       : 1;
  __REG32  USART5EN       : 1;
  __REG32  I2C1EN         : 1;
  __REG32  I2C2EN         : 1;
  __REG32  USBEN          : 1;
  __REG32                 : 4;
  __REG32  PWREN          : 1;
  __REG32  DACEN          : 1;
  __REG32                 : 1;
  __REG32  COMPEN         : 1;
} __rcc_apb1enr_bits;

/* AHB peripheral clock enable in low power mode register (RCC_AHBLPENR) */
typedef struct {
  __REG32  GPIOALPEN      : 1;
  __REG32  GPIOBLPEN      : 1;
  __REG32  GPIOCLPEN      : 1;
  __REG32  GPIODLPEN      : 1;
  __REG32  GPIOELPEN      : 1;
  __REG32  GPIOHLPEN      : 1;
  __REG32  GPIOFLPEN      : 1;
  __REG32  GPIOGLPEN      : 1;
  __REG32                 : 4;
  __REG32  CRCLPEN        : 1;
  __REG32                 : 2;
  __REG32  FLITFLPEN      : 1;
  __REG32  SRAMLPEN       : 1;
  __REG32                 : 7;
  __REG32  DMA1LPEN       : 1;
  __REG32  DMA2LPEN       : 1;
  __REG32                 : 6;
} __rcc_ahblpenr_bits;

/* APB2 peripheral clock enable in low power mode register (RCC_APB2LPENR) */
typedef struct {
  __REG32  SYSCFGLPEN     : 1;
  __REG32                 : 1;
  __REG32  TIM9LPEN       : 1;
  __REG32  TIM10LPEN      : 1;
  __REG32  TIM11LPEN      : 1;
  __REG32                 : 4;
  __REG32  ADC1LPEN       : 1;
  __REG32                 : 2;
  __REG32  SPI1LPEN       : 1;
  __REG32                 : 1;
  __REG32  USART1LPEN     : 1;
  __REG32                 :17;
} __rcc_apb2lpenr_bits;

/* APB1 peripheral clock enable in low power mode register (RCC_APB1LPENR) */
typedef struct {
  __REG32  TIM2LPEN       : 1;
  __REG32  TIM3LPEN       : 1;
  __REG32  TIM4LPEN       : 1;
  __REG32  TIM5LPEN       : 1;
  __REG32  TIM6LPEN       : 1;
  __REG32  TIM7LPEN       : 1;
  __REG32                 : 3;
  __REG32  LCDLPEN        : 1;
  __REG32                 : 1;
  __REG32  WWDLPEN        : 1;
  __REG32                 : 2;
  __REG32  SPI2LPEN       : 1;
  __REG32  SPI3LPEN       : 1;
  __REG32                 : 1;
  __REG32  USART2LPEN     : 1;
  __REG32  USART3LPEN     : 1;
  __REG32  USART4LPEN     : 1;
  __REG32  USART5LPEN     : 1;
  __REG32  I2C1LPEN       : 1;
  __REG32  I2C2LPEN       : 1;
  __REG32  USBLPEN        : 1;
  __REG32                 : 4;
  __REG32  PWRLPEN        : 1;
  __REG32  DACLPEN        : 1;
  __REG32                 : 1;
  __REG32  COMPLPEN       : 1;
} __rcc_apb1lpenr_bits;

/* Control/status register (RCC_CSR) */
typedef struct {
  __REG32  LSION          : 1;
  __REG32  LSIRDY         : 1;
  __REG32                 : 6;
  __REG32  LSEON          : 1;
  __REG32  LSERDY         : 1;
  __REG32  LSEBYP         : 1;
  __REG32                 : 5;
  __REG32  RTCSEL         : 2;
  __REG32                 : 4;
  __REG32  RTCEN          : 1;
  __REG32  RTCRST         : 1;
  __REG32  RMVF           : 1;
  __REG32  OBLRSTF        : 1;
  __REG32  PINRSTF        : 1;
  __REG32  PORRSTF        : 1;
  __REG32  SFTRSTF        : 1;
  __REG32  IWDGRSTF       : 1;
  __REG32  WWDGRSTF       : 1;
  __REG32  LPWRRSTF       : 1;
} __rcc_csr_bits;

/* GPIO port mode register (GPIOx_MODER) (x = A..G) */
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

/* GPIO port output type register (GPIOx_OTYPER) (x = A..G) */
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

/* GPIO port output speed register (GPIOx_OSPEEDR) (x = A..G) */
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

/* GPIO port pull-up/pull-down register (GPIOx_PUPDR) (x = A..G) */
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

/* GPIO port input data register (GPIOx_IDR) (x = A..G) */
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

/* GPIO port output data register (GPIOx_ODR) (x = A..G) */
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

/* GPIO port bit set/reset register (GPIOx_BSRR) (x = A..G) */
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

/* GPIO port configuration lock register (GPIOx_LCKR) (x = A..G) */
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

/* GPIO alternate function low register (GPIOx_AFRL) (x = A..G) */
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

/* GPIO alternate function high register (GPIOx_AFRH) (x = A..G) */
typedef struct {
  __REG32  AFRH8          : 4;
  __REG32  AFRH9          : 4;
  __REG32  AFRH10         : 4;
  __REG32  AFRH11         : 4;
  __REG32  AFRH12         : 4;
  __REG32  AFRH13         : 4;
  __REG32  AFRH14         : 4;
  __REG32  AFRH15         : 4;
} __gpio_afrh_bits;

/* RI input capture register (RI_ICR) */
typedef struct {
  __REG32  IC1IOS         : 4;
  __REG32  IC2IOS         : 4;
  __REG32  IC3IOS         : 4;
  __REG32  IC4IOS         : 4;
  __REG32  TIM            : 2;
  __REG32  IC1            : 1;
  __REG32  IC2            : 1;
  __REG32  IC3            : 1;
  __REG32  IC4            : 1;
  __REG32                 :10;
} __ri_icr_bits;

/* RI analog switches control register (RI_ASCR1) */
typedef struct {
  __REG32  CH0            : 1;
  __REG32  CH1            : 1;
  __REG32  CH2            : 1;
  __REG32  CH3            : 1;
  __REG32  CH4            : 1;
  __REG32  COMP1_SW1      : 1;
  __REG32  CH6            : 1;
  __REG32  CH7            : 1;
  __REG32  CH8            : 1;
  __REG32  CH9            : 1;
  __REG32  CH10           : 1;
  __REG32  CH11           : 1;
  __REG32  CH12           : 1;
  __REG32  CH13           : 1;
  __REG32  CH14           : 1;
  __REG32  CH15           : 1;
  __REG32  CH16           : 1;
  __REG32                 : 1;
  __REG32  CH18           : 1;
  __REG32  CH19           : 1;
  __REG32  CH20           : 1;
  __REG32  CH21           : 1;
  __REG32  CH22           : 1;
  __REG32  CH23           : 1;
  __REG32  CH24           : 1;
  __REG32  CH25           : 1;
  __REG32  VCOMP          : 1;
  __REG32  CH27           : 1;
  __REG32  CH28           : 1;
  __REG32  CH29           : 1;
  __REG32  CH30           : 1;
  __REG32  SCM            : 1;
} __ri_ascr1_bits;

/* RI analog switch control register 2 (RI_ASCR2) */
typedef struct {
  __REG32  GR10_1         : 1;
  __REG32  GR10_2         : 1;
  __REG32  GR10_3         : 1;
  __REG32  GR10_4         : 1;
  __REG32  GR6_1          : 1;
  __REG32  GR6_2          : 1;
  __REG32  GR5_1          : 1;
  __REG32  GR5_2          : 1;
  __REG32  GR5_3          : 1;
  __REG32  GR4_1          : 1;
  __REG32  GR4_2          : 1;
  __REG32  GR4_3          : 1;
  __REG32                 : 4;
  __REG32  CH0b_GR3_3     : 1;
  __REG32  CH1b_GR3_4     : 1;
  __REG32  CH2b_GR3_5     : 1;
  __REG32  CH3b_GR9_3     : 1;
  __REG32  CH6b_GR9_4     : 1;
  __REG32  CH7b_GR2_3     : 1;
  __REG32  CH8b_GR2_4     : 1;
  __REG32  CH9b_GR2_5     : 1;
  __REG32  CH10b_GR7_5    : 1;
  __REG32  CH11b_GR7_6    : 1;
  __REG32  CH12b_GR7_7    : 1;
  __REG32  GR6_3			    : 1;
  __REG32  GR6_4			    : 1;
  __REG32  GR5_4			    : 1;
  __REG32                 : 2;
} __ri_ascr2_bits;

/* RI hysteresis control register (RI_HYSCR1) */
typedef struct {
  __REG32  PA0            : 1;
  __REG32  PA1            : 1;
  __REG32  PA2            : 1;
  __REG32  PA3            : 1;
  __REG32  PA4            : 1;
  __REG32  PA5            : 1;
  __REG32  PA6            : 1;
  __REG32  PA7            : 1;
  __REG32  PA8            : 1;
  __REG32  PA9            : 1;
  __REG32  PA10           : 1;
  __REG32  PA11           : 1;
  __REG32  PA12           : 1;
  __REG32  PA13           : 1;
  __REG32  PA14           : 1;
  __REG32  PA15           : 1;
  __REG32  PB0            : 1;
  __REG32  PB1            : 1;
  __REG32  PB2            : 1;
  __REG32  PB3            : 1;
  __REG32  PB4            : 1;
  __REG32  PB5            : 1;
  __REG32  PB6            : 1;
  __REG32  PB7            : 1;
  __REG32  PB8            : 1;
  __REG32  PB9            : 1;
  __REG32  PB10           : 1;
  __REG32  PB11           : 1;
  __REG32  PB12           : 1;
  __REG32  PB13           : 1;
  __REG32  PB14           : 1;
  __REG32  PB15           : 1;
} __ri_hyscr1_bits;

/* RI hysteresis control register (RI_HYSCR2) */
typedef struct {
  __REG32  PC0            : 1;
  __REG32  PC1            : 1;
  __REG32  PC2            : 1;
  __REG32  PC3            : 1;
  __REG32  PC4            : 1;
  __REG32  PC5            : 1;
  __REG32  PC6            : 1;
  __REG32  PC7            : 1;
  __REG32  PC8            : 1;
  __REG32  PC9            : 1;
  __REG32  PC10           : 1;
  __REG32  PC11           : 1;
  __REG32  PC12           : 1;
  __REG32  PC13           : 1;
  __REG32  PC14           : 1;
  __REG32  PC15           : 1;
  __REG32  PD0            : 1;
  __REG32  PD1            : 1;
  __REG32  PD2            : 1;
  __REG32  PD3            : 1;
  __REG32  PD4            : 1;
  __REG32  PD5            : 1;
  __REG32  PD6            : 1;
  __REG32  PD7            : 1;
  __REG32  PD8            : 1;
  __REG32  PD9            : 1;
  __REG32  PD10           : 1;
  __REG32  PD11           : 1;
  __REG32  PD12           : 1;
  __REG32  PD13           : 1;
  __REG32  PD14           : 1;
  __REG32  PD15           : 1;
} __ri_hyscr2_bits;

/* RI hysteresis control register (RI_HYSCR3) */
typedef struct {
  __REG32  PE0            : 1;
  __REG32  PE1            : 1;
  __REG32  PE2            : 1;
  __REG32  PE3            : 1;
  __REG32  PE4            : 1;
  __REG32  PE5            : 1;
  __REG32  PE6            : 1;
  __REG32  PE7            : 1;
  __REG32  PE8            : 1;
  __REG32  PE9            : 1;
  __REG32  PE10           : 1;
  __REG32  PE11           : 1;
  __REG32  PE12           : 1;
  __REG32  PE13           : 1;
  __REG32  PE14           : 1;
  __REG32  PE15           : 1;
  __REG32  PF0            : 1;
  __REG32  PF1            : 1;
  __REG32  PF2            : 1;
  __REG32  PF3            : 1;
  __REG32  PF4            : 1;
  __REG32  PF5            : 1;
  __REG32  PF6            : 1;
  __REG32  PF7            : 1;
  __REG32  PF8            : 1;
  __REG32  PF9            : 1;
  __REG32  PF10           : 1;
  __REG32  PF11           : 1;
  __REG32  PF12           : 1;
  __REG32  PF13           : 1;
  __REG32  PF14           : 1;
  __REG32  PF15           : 1;
} __ri_hyscr3_bits;

/* RI hysteresis control register (RI_HYSCR4) */
typedef struct {
  __REG32  PG0            : 1;
  __REG32  PG1            : 1;
  __REG32  PG2            : 1;
  __REG32  PG3            : 1;
  __REG32  PG4            : 1;
  __REG32  PG5            : 1;
  __REG32  PG6            : 1;
  __REG32  PG7            : 1;
  __REG32  PG8            : 1;
  __REG32  PG9            : 1;
  __REG32  PG10           : 1;
  __REG32  PG11           : 1;
  __REG32  PG12           : 1;
  __REG32  PG13           : 1;
  __REG32  PG14           : 1;
  __REG32  PG15           : 1;
  __REG32                 :16;
} __ri_hyscr4_bits;

/* Analog switch mode register (RI_ASMR1) */
typedef struct {
  __REG32  PA0            : 1;
  __REG32  PA1            : 1;
  __REG32  PA2            : 1;
  __REG32  PA3            : 1;
  __REG32  PA4            : 1;
  __REG32  PA5            : 1;
  __REG32  PA6            : 1;
  __REG32  PA7            : 1;
  __REG32  PA8            : 1;
  __REG32  PA9            : 1;
  __REG32  PA10           : 1;
  __REG32  PA11           : 1;
  __REG32  PA12           : 1;
  __REG32  PA13           : 1;
  __REG32  PA14           : 1;
  __REG32  PA15           : 1;
  __REG32                 :16;
} __ri_asmr1_bits;

/* Analog switch mode register (RI_ASMR2) */
typedef struct {
  __REG32  PB0            : 1;
  __REG32  PB1            : 1;
  __REG32  PB2            : 1;
  __REG32  PB3            : 1;
  __REG32  PB4            : 1;
  __REG32  PB5            : 1;
  __REG32  PB6            : 1;
  __REG32  PB7            : 1;
  __REG32  PB8            : 1;
  __REG32  PB9            : 1;
  __REG32  PB10           : 1;
  __REG32  PB11           : 1;
  __REG32  PB12           : 1;
  __REG32  PB13           : 1;
  __REG32  PB14           : 1;
  __REG32  PB15           : 1;
  __REG32                 :16;
} __ri_asmr2_bits;

/* Analog switch mode register (RI_ASMR3) */
typedef struct {
  __REG32  PC0            : 1;
  __REG32  PC1            : 1;
  __REG32  PC2            : 1;
  __REG32  PC3            : 1;
  __REG32  PC4            : 1;
  __REG32  PC5            : 1;
  __REG32  PC6            : 1;
  __REG32  PC7            : 1;
  __REG32  PC8            : 1;
  __REG32  PC9            : 1;
  __REG32  PC10           : 1;
  __REG32  PC11           : 1;
  __REG32  PC12           : 1;
  __REG32  PC13           : 1;
  __REG32  PC14           : 1;
  __REG32  PC15           : 1;
  __REG32                 :16;
} __ri_asmr3_bits;

/* Analog switch mode register (RI_ASMR4) */
typedef struct {
  __REG32  PF0            : 1;
  __REG32  PF1            : 1;
  __REG32  PF2            : 1;
  __REG32  PF3            : 1;
  __REG32  PF4            : 1;
  __REG32  PF5            : 1;
  __REG32  PF6            : 1;
  __REG32  PF7            : 1;
  __REG32  PF8            : 1;
  __REG32  PF9            : 1;
  __REG32  PF10           : 1;
  __REG32  PF11           : 1;
  __REG32  PF12           : 1;
  __REG32  PF13           : 1;
  __REG32  PF14           : 1;
  __REG32  PF15           : 1;
  __REG32                 :16;
} __ri_asmr4_bits;

/* Analog switch mode register (RI_ASMR5) */
typedef struct {
  __REG32  PG0            : 1;
  __REG32  PG1            : 1;
  __REG32  PG2            : 1;
  __REG32  PG3            : 1;
  __REG32  PG4            : 1;
  __REG32  PG5            : 1;
  __REG32  PG6            : 1;
  __REG32  PG7            : 1;
  __REG32  PG8            : 1;
  __REG32  PG9            : 1;
  __REG32  PG10           : 1;
  __REG32  PG11           : 1;
  __REG32  PG12           : 1;
  __REG32  PG13           : 1;
  __REG32  PG14           : 1;
  __REG32  PG15           : 1;
  __REG32                 :16;
} __ri_asmr5_bits;

/* SYSCFG memory remap register (SYSCFG_MEMRMP) */
typedef struct {
  __REG32  MEM_MODE       : 2;
  __REG32                 : 6;
  __REG32  BOOT_MODE      : 2;
  __REG32                 :22;
} __syscfg_memrmp_bits;

/* USB internal pull-up management register (SYSCFG_PMC) */
typedef struct {
  __REG32  USB_PU			    : 1;
  __REG32                 :31;
} __syscfg_pmc_bits;

/* SYSCFG external interrupt configuration register 1 (SYSCFG_EXTICR1) */
typedef struct {
  __REG32  EXTI0          : 4;
  __REG32  EXTI1          : 4;
  __REG32  EXTI2          : 4;
  __REG32  EXTI3          : 4;
  __REG32                 :16;
} __syscfg_exticr1_bits;

/* SYSCFG external interrupt configuration register 2 (SYSCFG_EXTICR2) */
typedef struct {
  __REG32  EXTI4          : 4;
  __REG32  EXTI5          : 4;
  __REG32  EXTI6          : 4;
  __REG32  EXTI7          : 4;
  __REG32                 :16;
} __syscfg_exticr2_bits;

/* SYSCFG external interrupt configuration register 3 (SYSCFG_EXTICR3) */
typedef struct {
  __REG32  EXTI8          : 4;
  __REG32  EXTI9          : 4;
  __REG32  EXTI10         : 4;
  __REG32  EXTI11         : 4;
  __REG32                 :16;
} __syscfg_exticr3_bits;

/* SYSCFG external interrupt configuration register 4 (SYSCFG_EXTICR4) */
typedef struct {
  __REG32  EXTI12         : 4;
  __REG32  EXTI13         : 4;
  __REG32  EXTI14         : 4;
  __REG32  EXTI15         : 4;
  __REG32                 :16;
} __syscfg_exticr4_bits;

/* Interrupt mask register (EXTI_IMR) */
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
  __REG32                 : 9;
} __exti_imr_bits;

/* Rising Trigger selection register (EXTI_RTSR) */
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
  __REG32  TR20           : 1;
  __REG32  TR21           : 1;
  __REG32  TR22           : 1;
  __REG32                 : 9;
} __exti_rtsr_bits;

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
  __REG32  SWIER20        : 1;
  __REG32  SWIER21        : 1;
  __REG32  SWIER22        : 1;
  __REG32                 : 9;
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
  __REG32  PR20           : 1;
  __REG32  PR21           : 1;
  __REG32  PR22           : 1;
  __REG32                 : 9;
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

/* DMA channel x configuration register (DMA_CCRx) (x = 1 ..7) */
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
typedef struct {
  __REG32  NDT            :16;
  __REG32                 :16;
} __dma_cndtr_bits;

/* ADC status register (ADC_SR) */
typedef struct {
  __REG32 AWD             : 1;
  __REG32 EOC             : 1;
  __REG32 JEOC            : 1;
  __REG32 JSTRT           : 1;
  __REG32 STRT            : 1;
  __REG32 OVR             : 1;
  __REG32 ADONS           : 1;
  __REG32                 : 1;
  __REG32 RCNR            : 1;
  __REG32 JCNR            : 1;
  __REG32                 :22;
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
  __REG32 PDD             : 1;
  __REG32 PDI             : 1;
  __REG32                 : 4;
  __REG32 JAWDEN          : 1;
  __REG32 AWDEN           : 1;
  __REG32 RES             : 2;
  __REG32 OVRIE           : 1;
  __REG32                 : 5;
} __adc_cr1_bits;

/* ADC control register 2 (ADC_CR2) */
typedef struct {
  __REG32 ADON            : 1;
  __REG32 CONT            : 1;
  __REG32 ADC_CFG         : 1;
  __REG32                 : 1;
  __REG32 DELS            : 3;
  __REG32                 : 1;
  __REG32 DMA             : 1;
  __REG32 DDS             : 1;
  __REG32 EOCS            : 1;
  __REG32 ALIGN           : 1;
  __REG32                 : 4;
  __REG32 JEXTSEL         : 4;
  __REG32 JEXTEN          : 2;
  __REG32 JSWSTART        : 1;
  __REG32                 : 1;
  __REG32 EXTSEL          : 4;
  __REG32 EXTEN           : 2;
  __REG32 SWSTART         : 1;
  __REG32                 : 1;
} __adc_cr2_bits;

/* ADC sample time register 1 (ADC_SMPR1) */
typedef struct {
  __REG32 SMP20           : 3;
  __REG32 SMP21           : 3;
  __REG32 SMP22           : 3;
  __REG32 SMP23           : 3;
  __REG32 SMP24           : 3;
  __REG32 SMP25           : 3;
  __REG32 SMP26           : 3;
  __REG32 SMP27           : 3;
  __REG32 SMP28           : 3;
  __REG32 SMP29           : 3;
  __REG32                 : 2;
} __adc_smpr1_bits;

/* ADC sample time register 2 (ADC_SMPR2) */
typedef struct {
  __REG32 SMP10           : 3;
  __REG32 SMP11           : 3;
  __REG32 SMP12           : 3;
  __REG32 SMP13           : 3;
  __REG32 SMP14           : 3;
  __REG32 SMP15           : 3;
  __REG32 SMP16           : 3;
  __REG32 SMP17           : 3;
  __REG32 SMP18           : 3;
  __REG32 SMP19           : 3;
  __REG32                 : 2;
} __adc_smpr2_bits;

/* ADC sample time register 3 (ADC_SMPR3) */
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
} __adc_smpr3_bits;

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
  __REG32 SQ25            : 5;
  __REG32 SQ26            : 5;
  __REG32 SQ27            : 5;
  __REG32 SQ28            : 5;
  __REG32 L               : 5;
  __REG32                 : 7;
} __adc_sqr1_bits;

/* ADC regular sequence register 2 (ADC_SQR2) */
typedef struct {
  __REG32 SQ19            : 5;
  __REG32 SQ20            : 5;
  __REG32 SQ21            : 5;
  __REG32 SQ22            : 5;
  __REG32 SQ23            : 5;
  __REG32 SQ24            : 5;
  __REG32                 : 2;
} __adc_sqr2_bits;

/* ADC regular sequence register 3 (ADC_SQR3) */
typedef struct {
  __REG32 SQ13            : 5;
  __REG32 SQ14            : 5;
  __REG32 SQ15            : 5;
  __REG32 SQ16            : 5;
  __REG32 SQ17            : 5;
  __REG32 SQ18            : 5;
  __REG32                 : 2;
} __adc_sqr3_bits;

/* ADC regular sequence register 4 (ADC_SQR4) */
typedef struct {
  __REG32 SQ7             : 5;
  __REG32 SQ8             : 5;
  __REG32 SQ9             : 5;
  __REG32 SQ10            : 5;
  __REG32 SQ11            : 5;
  __REG32 SQ12            : 5;
  __REG32                 : 2;
} __adc_sqr4_bits;

/* ADC regular sequence register 5 (ADC_SQR5) */
typedef struct {
  __REG32 SQ1             : 5;
  __REG32 SQ2             : 5;
  __REG32 SQ3             : 5;
  __REG32 SQ4             : 5;
  __REG32 SQ5             : 5;
  __REG32 SQ6             : 5;
  __REG32                 : 2;
} __adc_sqr5_bits;

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
  __REG32                 :16;
} __adc_dr_bits;

/* ADC sample time register 0 (ADC_SMPR0) */
typedef struct {
  __REG32 SMP30           : 3;
  __REG32 SMP31           : 3;
  __REG32                 :26;
} __adc_smpr0_bits;

/* ADC common status register (ADC_CSR) */
typedef struct {
  __REG32 AWD1            : 1;
  __REG32 EOC1            : 1;
  __REG32 JEOC1           : 1;
  __REG32 JSTRT1          : 1;
  __REG32 STRT1           : 1;
  __REG32 OVR1            : 1;
  __REG32 ADONS1          : 1;
  __REG32                 :25;
} __adc_csr_bits;

/* ADC common control register (ADC_CCR) */
typedef struct {
  __REG32                 :16;
  __REG32 ADCPRE          : 2;
  __REG32                 : 5;
  __REG32 TSVREFE         : 1;
  __REG32                 : 8;
} __adc_ccr_bits;

/* DAC control register (DAC_CR) */
typedef struct {
  __REG32 EN1             : 1;
  __REG32 BOFF1           : 1;
  __REG32 TEN1            : 1;
  __REG32 TSEL1           : 3;
  __REG32 WAVE1           : 2;
  __REG32 MAMP1           : 4;
  __REG32 DMAEN1          : 1;
  __REG32 DMAUDRIE1       : 1;
  __REG32                 : 2;
  __REG32 EN2             : 1;
  __REG32 BOFF2           : 1;
  __REG32 TEN2            : 1;
  __REG32 TSEL2           : 3;
  __REG32 WAVE2           : 2;
  __REG32 MAMP2           : 4;
  __REG32 DMAEN2          : 1;
  __REG32 DMAUDRIE2       : 1;
  __REG32                 : 2;
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

/* DAC status register (DAC_SR) */
typedef struct {
  __REG32                 :13;
  __REG32 DMAUDR1         : 1;
  __REG32                 :15;
  __REG32 DMAUDR2         : 1;
  __REG32                 : 2;
} __dac_sr_bits;

/* COMP comparator control and status register (COMP_CSR) */
typedef struct {
  __REG32 _10KPU          : 1;
  __REG32 _400KPU         : 1;
  __REG32 _10KPD          : 1;
  __REG32 _400KPD         : 1;
  __REG32 CMP1EN          : 1;
  __REG32 SW1	            : 1;
  __REG32                 : 1;
  __REG32 CMP1OUT         : 1;
  __REG32                 : 4;
  __REG32 SPEED           : 1;
  __REG32 CMP2OUT         : 1;
  __REG32                 : 2;
  __REG32 VREFOUTEN       : 1;
  __REG32 WNDWE           : 1;
  __REG32 INSEL           : 3;
  __REG32 OUTSEL          : 3;
  __REG32                 : 2;
  __REG32 FCH3	          : 1;
  __REG32 FCH8	          : 1;
  __REG32 RCH13	          : 1;
  __REG32 CAIE	          : 1;
  __REG32 CAIF	          : 1;
  __REG32 TSUSP	          : 1;
} __comp_csr_bits;

/* OPAMP control/status register (OPAMP_CSR) */
typedef struct {
  __REG32 OPA1PD          : 1;
  __REG32 S3SEL1         	: 1;
  __REG32 S4SEL1          : 1;
  __REG32 S5SEL1	        : 1;
  __REG32 S6SEL1          : 1;
  __REG32 OPA1CAL_L	      : 1;
  __REG32 OPA1CAL_H	      : 1;
  __REG32 OPA1LPM	      	: 1;
  __REG32 OPA2PD	      	: 1;
  __REG32 S3SEL2	      	: 1;
  __REG32 S4SEL2	      	: 1;
  __REG32 S5SEL2	      	: 1;
  __REG32 S6SEL2	      	: 1;
  __REG32 OPA2CAL_L	     	: 1;
  __REG32 OPA2CAL_H	     	: 1;
  __REG32 OPA2LPM	     		: 1;
  __REG32 OPA3PD	     		: 1;
  __REG32 S3SEL3	     		: 1;
  __REG32 S4SEL3	     		: 1;
  __REG32 S5SEL3	     		: 1;
  __REG32 S6SEL3	     		: 1;
  __REG32 OPA3CAL_L	     	: 1;
  __REG32 OPA3CAL_H	     	: 1;
  __REG32 OPA3LPM	     		: 1;
  __REG32 ANAWSEL1	     	: 1;
  __REG32 ANAWSEL2	     	: 1;
  __REG32 ANAWSEL3	     	: 1;
  __REG32 S7SEL2	     		: 1;
  __REG32 AOP_RANGE	     	: 1;
  __REG32 OPA1CALOUT	    : 1;
  __REG32 OPA2CALOUT	    : 1;
  __REG32 OPA3CALOUT	    : 1;
} __opamp_csr_bits;

/* OPAMP offset trimming register for normal mode (OPAMP_OTR) */
/* OPAMP offset trimming register for low power mode (OPAMP_LPOTR) */
typedef struct {
  __REG32 AO1_OPT_OFFSET_TRIM	:10;
  __REG32 AO2_OPT_OFFSET_TRIM	:10;
  __REG32 AO3_OPT_OFFSET_TRIM	:10;
  __REG32 					         	: 1;
  __REG32 OT_USER          		: 1;
} __opamp_otr_bits;

/* LCD control register (LCD_CR) */
typedef struct {
  __REG32 LCDEN           : 1;
  __REG32 VSEL            : 1;
  __REG32 DUTY            : 3;
  __REG32 BIAS            : 2;
  __REG32 MUX_SEG         : 1;
  __REG32                 :24;
} __lcd_cr_bits;

/* LCD frame control register (LCD_FCR) */
typedef struct {
  __REG32 HD              : 1;
  __REG32 SOFIE           : 1;
  __REG32                 : 1;
  __REG32 UDDIE           : 1;
  __REG32 PON             : 3;
  __REG32 DEAD            : 3;
  __REG32 CC              : 3;
  __REG32 BLINKF          : 3;
  __REG32 BLINK           : 2;
  __REG32 DIV             : 4;
  __REG32 PS              : 4;
  __REG32                 : 6;
} __lcd_fcr_bits;

/* LCD status register (LCD_SR) */
typedef struct {
  __REG32 ENS             : 1;
  __REG32 SOF             : 1;
  __REG32 UDR             : 1;
  __REG32 UDD             : 1;
  __REG32 RDY             : 1;
  __REG32 FCRSF           : 1;
  __REG32                 :26;
} __lcd_sr_bits;

/* LCD clear register (LCD_CLR) */
typedef struct {
  __REG32                 : 1;
  __REG32 SOFC            : 1;
  __REG32                 : 1;
  __REG32 UDDC            : 1;
  __REG32                 :28;
} __lcd_clr_bits;

/* LCD display memory (LCD_RAM0) */
typedef struct {
  __REG32 S00             : 1;
  __REG32 S01             : 1;
  __REG32 S02             : 1;
  __REG32 S03             : 1;
  __REG32 S04             : 1;
  __REG32 S05             : 1;
  __REG32 S06             : 1;
  __REG32 S07             : 1;
  __REG32 S08             : 1;
  __REG32 S09             : 1;
  __REG32 S10             : 1;
  __REG32 S11             : 1;
  __REG32 S12             : 1;
  __REG32 S13             : 1;
  __REG32 S14             : 1;
  __REG32 S15             : 1;
  __REG32 S16             : 1;
  __REG32 S17             : 1;
  __REG32 S18             : 1;
  __REG32 S19             : 1;
  __REG32 S20             : 1;
  __REG32 S21             : 1;
  __REG32 S22             : 1;
  __REG32 S23             : 1;
  __REG32 S24             : 1;
  __REG32 S25             : 1;
  __REG32 S26             : 1;
  __REG32 S27             : 1;
  __REG32 S28             : 1;
  __REG32 S29             : 1;
  __REG32 S30             : 1;
  __REG32 S31             : 1;
} __lcd_ram0_bits;

/* LCD display memory (LCD_RAM1) */
typedef struct {
  __REG32 S32             : 1;
  __REG32 S33             : 1;
  __REG32 S34             : 1;
  __REG32 S35             : 1;
  __REG32 S36             : 1;
  __REG32 S37             : 1;
  __REG32 S38             : 1;
  __REG32 S39             : 1;
  __REG32 S40             : 1;
  __REG32 S41             : 1;
  __REG32 S42             : 1;
  __REG32 S43             : 1;
  __REG32                 :20;
} __lcd_ram1_bits;

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
  __REG32 OCCS            : 1;
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

/* Control register 1 (TIM6_CR1) */
/* Control register 1 (TIM7_CR1) */
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
/* Control register 2 (TIM7_CR2) */
typedef struct {
  __REG32                 : 4;
  __REG32 MMS             : 3;
  __REG32                 :25;
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

/* Control register 1 (TIM9_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32 OPM             : 1;
  __REG32                 : 3;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim9_cr1_bits;

/* Control register 1 (TIM10_CR1) */
/* Control register 1 (TIM11_CR1) */
typedef struct {
  __REG32 CEN             : 1;
  __REG32 UDIS            : 1;
  __REG32 URS             : 1;
  __REG32                 : 4;
  __REG32 ARPE            : 1;
  __REG32 CKD             : 2;
  __REG32                 :22;
} __tim10_cr1_bits;

/* Control register 2 (TIM9_CR2) */
typedef struct {
  __REG32                 : 4;
  __REG32 MMS             : 3;
  __REG32                 :25;
} __tim9_cr2_bits;

/* Slave mode control register (TIM9_SMCR) */
typedef struct {
  __REG32 SMS             : 3;
  __REG32                 : 1;
  __REG32 TS              : 3;
  __REG32 MSM             : 1;
  __REG32                 :24;
} __tim9_smcr_bits;

/* Interrupt enable register (TIM9_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32 CC2IE           : 1;
  __REG32                 : 3;
  __REG32 TIE             : 1;
  __REG32                 :25;
} __tim9_dier_bits;

/* Interrupt enable register (TIM10_DIER) */
/* Interrupt enable register (TIM11_DIER) */
typedef struct {
  __REG32 UIE             : 1;
  __REG32 CC1IE           : 1;
  __REG32                 :30;
} __tim10_dier_bits;

/* Status register (TIM9_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32 CC2IF           : 1;
  __REG32                 : 3;
  __REG32 TIF             : 1;
  __REG32                 : 2;
  __REG32 CC1OF           : 1;
  __REG32 CC2OF           : 1;
  __REG32                 :21;
} __tim9_sr_bits;

/* Status register (TIM10_SR) */
/* Status register (TIM11_SR) */
typedef struct {
  __REG32 UIF             : 1;
  __REG32 CC1IF           : 1;
  __REG32                 : 7;
  __REG32 CC1OF           : 1;
  __REG32                 :22;
} __tim10_sr_bits;

/* Event generation register (TIM9_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32 CC2G            : 1;
  __REG32                 : 3;
  __REG32 TG              : 1;
  __REG32                 :25;
} __tim9_egr_bits;

/* Event generation register (TIM10_EGR) */
/* Event generation register (TIM11_EGR) */
typedef struct {
  __REG32 UG              : 1;
  __REG32 CC1G            : 1;
  __REG32                 :30;
} __tim10_egr_bits;

/* Capture/compare mode register 1 (TIM9_CCMR1) */
typedef union {
  /* TIM9_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32 IC2S            : 2;
  __REG32 IC2PSC          : 2;
  __REG32 IC2F            : 4;
  __REG32                 :16;
  };
  /* TIM9_OCMR1*/
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
} __tim9_ccmr1_bits;

/* Capture/compare mode register 1 (TIM10_CCMR1) */
/* Capture/compare mode register 1 (TIM11_CCMR1) */
typedef union {
  /* TIM10_CCMR1*/
  /* TIM11_CCMR1*/
  struct {
  __REG32 IC1S            : 2;
  __REG32 IC1PSC          : 2;
  __REG32 IC1F            : 4;
  __REG32                 :24;
  };
  /* TIM10_OCMR1*/
  /* TIM11_OCMR1*/
  struct {
  __REG32 OC1S            : 2;
  __REG32 OC1FE           : 1;
  __REG32 OC1PE           : 1;
  __REG32 OC1M            : 3;
  __REG32                 :25;
  };
} __tim10_ccmr1_bits;

/* Capture/compare enable register (TIM9_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32                 : 1;
  __REG32 CC1NP           : 1;
  __REG32 CC2E            : 1;
  __REG32 CC2P            : 1;
  __REG32                 : 1;
  __REG32 CC2NP           : 1;
  __REG32                 :24;
} __tim9_ccer_bits;

/* Capture/compare enable register (TIM10_CCER) */
/* Capture/compare enable register (TIM11_CCER) */
typedef struct {
  __REG32 CC1E            : 1;
  __REG32 CC1P            : 1;
  __REG32                 : 1;
  __REG32 CC1NP           : 1;
  __REG32                 :28;
} __tim10_ccer_bits;

/* Counter (TIM9_CNT) */
/* Counter (TIM10_CNT) */
/* Counter (TIM11_CNT) */
typedef struct {
  __REG32 CNT             :16;
  __REG32                 :16;
} __tim9_cnt_bits;

/* Prescaler (TIM9_PSC) */
/* Prescaler (TIM10_PSC) */
/* Prescaler (TIM11_PSC) */
typedef struct {
  __REG32 PSC             :16;
  __REG32                 :16;
} __tim9_psc_bits;

/* Auto-reload register (TIM9_ARR) */
/* Auto-reload register (TIM10_ARR) */
/* Auto-reload register (TIM11_ARR) */
typedef struct {
  __REG32 ARR             :16;
  __REG32                 :16;
} __tim9_arr_bits;

/* Capture/compare register (TIM9_CCRx) */
/* Capture/compare register (TIM10_CCR) */
/* Capture/compare register (TIM11_CCR) */
typedef struct {
  __REG32 CCR             :16;
  __REG32                 :16;
} __tim9_ccr_bits;

/* TIM10 slave mode control register (TIM10_SMCR) */
/* TIM11 slave mode control register (TIM11_SMCR) */
typedef struct {
  __REG32                 : 8;
  __REG32 ETF             : 4;
  __REG32 ETPS            : 2;
  __REG32 ECE             : 1;
  __REG32 ETP             : 1;
  __REG32                 :16;
} __tim10_smcr_bits;

/* TIM9  option register 1 (TIM9_OR) */
/* TIM10 option register 1 (TIM10_OR) */
/* TIM11 option register 1 (TIM11_OR) */
typedef struct {
  __REG32 TI1_RMP         : 2;
  __REG32                 :30;
} __tim9_or_bits;

/* RTC time register (RTC_TR) */
typedef struct {
  __REG32 SU              : 4;
  __REG32 ST              : 3;
  __REG32                 : 1;
  __REG32 MNU             : 4;
  __REG32 MNT             : 3;
  __REG32                 : 1;
  __REG32 HU              : 4;
  __REG32 HT              : 2;
  __REG32 PM              : 1;
  __REG32                 : 9;
} __rtc_tr_bits;

/* RTC date register (RTC_DR) */
typedef struct {
  __REG32 DU              : 4;
  __REG32 DT              : 2;
  __REG32                 : 2;
  __REG32 MU              : 4;
  __REG32 MT              : 1;
  __REG32 WDU             : 3;
  __REG32 YU              : 4;
  __REG32 YT              : 4;
  __REG32                 : 8;
} __rtc_dr_bits;

/* RTC control register (RTC_CR) */
typedef struct {
  __REG32 WUCKSEL         : 3;
  __REG32 TSEDGE          : 1;
  __REG32 REFCKON         : 1;
  __REG32 BYPSHAD         : 1;
  __REG32 FMT             : 1;
  __REG32 DCE             : 1;
  __REG32 ALRAE           : 1;
  __REG32 ALRBE           : 1;
  __REG32 WUTE            : 1;
  __REG32 TSE             : 1;
  __REG32 ALRAIE          : 1;
  __REG32 ALRBIE          : 1;
  __REG32 WUTIE           : 1;
  __REG32 TSIE            : 1;
  __REG32 ADD1H           : 1;
  __REG32 SUB1H           : 1;
  __REG32 BKP             : 1;
  __REG32 COSEL           : 1;
  __REG32 POL             : 1;
  __REG32 OSEL            : 2;
  __REG32 COE             : 1;
  __REG32                 : 8;
} __rtc_cr_bits;

/* RTC initialization and status register (RTC_ISR) */
typedef struct {
  __REG32 ALRAWF          : 1;
  __REG32 ALRBWF          : 1;
  __REG32 WUTWF           : 1;
  __REG32 SHPF            : 1;
  __REG32 INITS           : 1;
  __REG32 RSF             : 1;
  __REG32 INITF           : 1;
  __REG32 INIT            : 1;
  __REG32 ALRAF           : 1;
  __REG32 ALRBF           : 1;
  __REG32 WUTF            : 1;
  __REG32 TSF             : 1;
  __REG32 TSOVF           : 1;
  __REG32 TAMP1F          : 1;
  __REG32 TAMP2F          : 1;
  __REG32 TAMP3F          : 1;
  __REG32 RECALPF					: 1;
  __REG32                 :15;
} __rtc_isr_bits;

/* RTC prescaler register (RTC_PRER) */
typedef struct {
  __REG32 PREDIV_S        :15;
  __REG32                 : 1;
  __REG32 PREDIV_A        : 7;
  __REG32                 : 9;
} __rtc_prer_bits;

/* RTC wakeup timer register (RTC_WUTR) */
typedef struct {
  __REG32 WUT             :16;
  __REG32                 :16;
} __rtc_wutr_bits;

/* RTC calibration register (RTC_CALIBR) */
typedef struct {
  __REG32 DC              : 5;
  __REG32                 : 2;
  __REG32 DCS             : 1;
  __REG32                 :24;
} __rtc_calibr_bits;

/* RTC alarm A register (RTC_ALRMAR) */
/* RTC alarm B register (RTC_ALRMBR) */
typedef struct {
  __REG32 SU              : 4;
  __REG32 ST              : 3;
  __REG32 MSK1            : 1;
  __REG32 MNU             : 4;
  __REG32 MNT             : 3;
  __REG32 MSK2            : 1;
  __REG32 HU              : 4;
  __REG32 HT              : 2;
  __REG32 PM              : 1;
  __REG32 MSK3            : 1;
  __REG32 DU              : 4;
  __REG32 DT              : 2;
  __REG32 WDSEL           : 1;
  __REG32 MSK4            : 1;
} __rtc_alrmar_bits;

/* RTC sub second register (RTC_SSR) */
typedef struct {
  __REG32 SS              :16;
  __REG32                 :16;
} __rtc_ssr_bits;

/* RTC shift control register (RTC_SHIFTR) */
typedef struct {
  __REG32 SUBFS           :15;
  __REG32                 :16;
  __REG32 ADD1S           : 1;
} __rtc_shiftr_bits;

/* RTC write protection register (RTC_WPR) */
typedef struct {
  __REG32 KEY             : 8;
  __REG32                 :24;
} __rtc_wpr_bits;

/* RTC time stamp time register (RTC_TSTR) */
typedef struct {
  __REG32 SU              : 4;
  __REG32 ST              : 3;
  __REG32                 : 1;
  __REG32 MNU             : 4;
  __REG32 MNT             : 3;
  __REG32                 : 1;
  __REG32 HU              : 4;
  __REG32 HT              : 2;
  __REG32 PM              : 1;
  __REG32                 : 9;
} __rtc_tstr_bits;

/* RTC time-stamp date register (RTC_TSDR) */
typedef struct {
  __REG32 DU              : 4;
  __REG32 DT              : 2;
  __REG32                 : 2;
  __REG32 MU              : 4;
  __REG32 MT              : 1;
  __REG32 WDU             : 3;
  __REG32                 :16;
} __rtc_tsdr_bits;

/* RTC tamper and alternate function configuration register (RTC_TAFCR) */
typedef struct {
  __REG32 TAMP1E          : 1;
  __REG32 TAMP1TRG        : 1;
  __REG32 TAMPIE          : 1;
  __REG32 TAMP2E          : 1;
  __REG32 TAMP2TRG        : 1;
  __REG32 TAMP3E          : 1;
  __REG32 TAMP3TRG        : 1;
  __REG32 TAMPTS        	: 1;
  __REG32 TAMPFREQ        : 3;
  __REG32 TAMPFLT        	: 2;
  __REG32 TAMPPRCH        : 2;
  __REG32 TAMPPUDIS       : 1;
  __REG32                 : 2;
  __REG32 ALARMOUTTYPE    : 1;
  __REG32                 :13;
} __rtc_tafcr_bits;

/* RTC calibration register (RTC_CALR) */
typedef struct {
  __REG32 CALM            : 9;
  __REG32                 : 4;
  __REG32 CALW16	        : 1;
  __REG32 CALW8           : 1;
  __REG32 CALP            : 1;
  __REG32                 :16;
} __rtc_calr_bits;

/* RTC alarm A sub second register (RTC_ALRMASSR) */
/* RTC alarm B sub second register (RTC_ALRMBSSR) */
typedef struct {
  __REG32 SS	            :15;
  __REG32                 : 9;
  __REG32 MASKSS	        : 4;
  __REG32                 : 4;
} __rtc_alrmassr_bits;

/* Prescaler register (IWDG_PR) */
typedef struct {
  __REG32 PR              : 3;
  __REG32                 :29;
} __iwdg_pr_bits;

/* Reload register (IWDG_RLR) */
typedef struct {
  __REG32 RL              :12;
  __REG32                 :20;
} __iwdg_rlr_bits;

/* Status register (IWDG_SR) */
typedef struct {
  __REG32 PVU             : 1;
  __REG32 RVU             : 1;
  __REG32                 :30;
} __iwdg_sr_bits;

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

/* USB endpoint n register, n=[0..7] */
typedef struct {
  __REG32 EA              : 4;
  __REG32 STAT_TX         : 2;
  __REG32 DTOG_TX         : 1;
  __REG32 CTR_TX          : 1;
  __REG32 EP_KIND         : 1;
  __REG32 EP_TYPE         : 2;
  __REG32 SETUP           : 1;
  __REG32 STAT_RX         : 2;
  __REG32 DTOG_RX         : 1;
  __REG32 CTR_RX          : 1;
  __REG32                 :16;
} __usb_epr_bits;

/* USB control register (USB_CNTR) */
typedef struct {
  __REG32 FRES            : 1;
  __REG32 PDWN            : 1;
  __REG32 LP_MODE         : 1;
  __REG32 FSUSP           : 1;
  __REG32 RESUME          : 1;
  __REG32                 : 3;
  __REG32 ESOFM           : 1;
  __REG32 SOFM            : 1;
  __REG32 RESETM          : 1;
  __REG32 SUSPM           : 1;
  __REG32 WKUPM           : 1;
  __REG32 ERRM            : 1;
  __REG32 PMAOVRM         : 1;
  __REG32 CTRM            : 1;
  __REG32                 :16;
} __usb_cntr_bits;

/* USB interrupt status register (USB_ISTR) */
typedef struct {
  __REG32 EP_ID           : 4;
  __REG32 DIR             : 1;
  __REG32                 : 3;
  __REG32 ESOF            : 1;
  __REG32 SOF             : 1;
  __REG32 RESET           : 1;
  __REG32 SUSP            : 1;
  __REG32 WKUP            : 1;
  __REG32 ERR             : 1;
  __REG32 PMAOVR          : 1;
  __REG32 CTR             : 1;
  __REG32                 :16;
} __usb_istr_bits;

/* USB frame number register (USB_FNR) */
typedef struct {
  __REG32 FN              :11;
  __REG32 LSOF            : 2;
  __REG32 LCK             : 1;
  __REG32 RXDM            : 1;
  __REG32 RXDP            : 1;
  __REG32                 :16;
} __usb_fnr_bits;

/* USB device address (USB_DADDR) */
typedef struct {
  __REG32 ADD             : 7;
  __REG32 EF              : 1;
  __REG32                 :24;
} __usb_daddr_bits;

/* Buffer table address (USB_BTABLE) */
typedef struct {
  __REG32 BTABLE          :16;
  __REG32                 :16;
} __usb_btable_bits;

/* SRAM/NOR-Flash chip-select control registers 1..4 (FSMC_BCR1..4) */
typedef struct {
  __REG32 MBKEN           : 1;
  __REG32 MUXEN           : 1;
  __REG32 MTYP           	: 2;
  __REG32 MWID            : 2;
  __REG32 FACCEN          : 1;
  __REG32 		            : 1;
  __REG32 BURSTEN         : 1;
  __REG32 WAITPOL         : 1;
  __REG32 WRAPMOD         : 1;
  __REG32 WAITCFG         : 1;
  __REG32 WREN            : 1;
  __REG32 WAITEN          : 1;
  __REG32 EXTMOD          : 1;
  __REG32 ASCYCWAIT       : 1;
  __REG32                 : 3;
  __REG32 CBURSTRW        : 1;
  __REG32                 :12;
} __fsmc_bcr_bits;

/* SRAM/NOR-Flash chip-select timing registers 1..4 (FSMC_BTR1..4) */
typedef struct {
  __REG32 ADDSET          : 4;
  __REG32 ADDHLD          : 4;
  __REG32 DATAST          : 8;
  __REG32 BUSTURN         : 4;
  __REG32 CLKDIV          : 4;
  __REG32 DATLAT          : 4;
  __REG32 ACCMOD          : 2;
  __REG32 		            : 2;
} __fsmc_btr_bits;

/* SRAM/NOR-Flash write timing registers 1..4 (FSMC_BWTR1..4) */
typedef struct {
  __REG32 ADDSET          : 4;
  __REG32 ADDHLD          : 4;
  __REG32 DATAST          : 8;
  __REG32 			          : 4;
  __REG32 CLKDIV          : 4;
  __REG32 DATLAT          : 4;
  __REG32 ACCMOD          : 2;
  __REG32 		            : 2;
} __fsmc_bwtr_bits;

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
  __REG32                 : 1;
  __REG32 OVER8           : 1;
  __REG32                 :16;
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
  __REG32 ONEBITE         : 1;
  __REG32                 :20;
} __usart_cr3_bits;

/* Guard time and prescaler register (USART_GTPR) */
typedef struct {
  __REG32 PSC             : 8;
  __REG32 GT              : 8;
  __REG32                 :16;
} __usart_gtpr_bits;

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
  __REG32                 : 1;
  __REG32 FRF             : 1;
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
  __REG32 FRE             : 1;
  __REG32                 :23;
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
  __REG32 RXCRC           :16;
  __REG32                 :16;
} __spi_rxcrcr_bits;

/* SPI Tx CRC register (SPI_TXCRCR) */
typedef struct {
  __REG32 TXCRC           :16;
  __REG32                 :16;
} __spi_txcrcr_bits;

/* SPI_I2S configuration register (SPI_I2SCFGR) */
typedef struct {
  __REG32 CHLEN           : 1;
  __REG32 DATLEN          : 2;
  __REG32 CKPOL           : 1;
  __REG32 I2SSTD          : 2;
  __REG32 		            : 1;
  __REG32 PCMSYNC         : 1;
  __REG32 I2SCFG          : 2;
  __REG32 I2SE            : 1;
  __REG32 I2SMOD          : 1;
  __REG32                 :20;
} __spi_i2scfgr_bits;

/* SPI_I2S prescaler register (SPI_I2SPR) */
typedef struct {
  __REG32 I2SDIV          : 8;
  __REG32 ODD             : 1;
  __REG32 MCKOE           : 1;
  __REG32 		            :22;
} __spi_i2spr_bits;

/* SDIO Power Control Register (SDIO_POWER) */
typedef struct {
  __REG32 PWRCTRL         : 2;
  __REG32                 :30;
} __sdio_power_bits;

/* SDIO Clock Control Register (SDIO_CLKCR) */
typedef struct {
  __REG32 CLKDIV          : 8;
  __REG32 CLKEN           : 1;
  __REG32 PWRSAV          : 1;
  __REG32 BYPASS          : 1;
  __REG32 WIDBUS          : 2;
  __REG32 NEGEDGE         : 1;
  __REG32 HWFC_EN         : 1;
  __REG32                 :17;
} __sdio_clkcr_bits;

/* SDIO Command Register (SDIO_CMD) */
typedef struct {
  __REG32 CMDINDEX        : 6;
  __REG32 WAITRESP        : 2;
  __REG32 WAITINT         : 1;
  __REG32 WAITPEND        : 1;
  __REG32 CPSMEN          : 1;
  __REG32 SDIOSuspend     : 1;
  __REG32 ENCMDcompl      : 1;
  __REG32 nIEN            : 1;
  __REG32 ATACMD          : 1;
  __REG32                 :17;
} __sdio_cmd_bits;

/* SDIO Command Response Register (SDIO_RESPCMD) */
typedef struct {
  __REG32 RESPCMD         : 6;
  __REG32                 :26;
} __sdio_respcmd_bits;

/* SDIO Data Length Register (SDIO_DLEN) */
typedef struct {
  __REG32 DATALENGTH      :25;
  __REG32                 : 7;
} __sdio_dlen_bits;

/* SDIO Data Control Register (SDIO_DCTRL) */
typedef struct {
  __REG32 DTEN            : 1;
  __REG32 DTDIR           : 1;
  __REG32 DTMODE          : 1;
  __REG32 DMAEN           : 1;
  __REG32 DBLOCKSIZE      : 4;
  __REG32 RWSTART         : 1;
  __REG32 RWSTOP          : 1;
  __REG32 RWMOD           : 1;
  __REG32 SDIOEN          : 1;
  __REG32                 :20;
} __sdio_dctrl_bits;

/* SDIO Data Counter Register (SDIO_DCOUNT) */
typedef struct {
  __REG32 DATACOUNT       :25;
  __REG32                 : 7;
} __sdio_dcount_bits;

/* SDIO Status Register (SDIO_STA) */
typedef struct {
  __REG32 CCRCFAIL        : 1;
  __REG32 DCRCFAIL        : 1;
  __REG32 CTIMEOUT        : 1;
  __REG32 DTIMEOUT        : 1;
  __REG32 TXUNDERR        : 1;
  __REG32 RXOVERR         : 1;
  __REG32 CMDREND         : 1;
  __REG32 CMDSENT         : 1;
  __REG32 DATAEND         : 1;
  __REG32 STBITERR        : 1;
  __REG32 DBCKEND         : 1;
  __REG32 CMDACT          : 1;
  __REG32 TXACT           : 1;
  __REG32 RXACT           : 1;
  __REG32 TXFIFOHE        : 1;
  __REG32 RXFIFOHF        : 1;
  __REG32 TXFIFOF         : 1;
  __REG32 RXFIFOF         : 1;
  __REG32 TXFIFOE         : 1;
  __REG32 RXFIFOE         : 1;
  __REG32 TXDAVL          : 1;
  __REG32 RXDAVL          : 1;
  __REG32 SDIOIT          : 1;
  __REG32 CEATAEND        : 1;
  __REG32                 : 8;
} __sdio_sta_bits;

/* SDIO Interrupt Clear Register (SDIO_ICR) */
typedef struct {
  __REG32 CCRCFAILC       : 1;
  __REG32 DCRCFAILC       : 1;
  __REG32 CTIMEOUTC       : 1;
  __REG32 DTIMEOUTC       : 1;
  __REG32 TXUNDERRC       : 1;
  __REG32 RXOVERRC        : 1;
  __REG32 CMDRENDC        : 1;
  __REG32 CMDSENTC        : 1;
  __REG32 DATAENDC        : 1;
  __REG32 STBITERRC       : 1;
  __REG32 DBCKENDC        : 1;
  __REG32                 :11;
  __REG32 SDIOITC         : 1;
  __REG32 CEATAENDC       : 1;
  __REG32                 : 8;
} __sdio_icr_bits;

/* SDIO Mask Register (SDIO_MASK) */
typedef struct {
  __REG32 CCRCFAILIE      : 1;
  __REG32 DCRCFAILIE      : 1;
  __REG32 CTIMEOUTIE      : 1;
  __REG32 DTIMEOUTIE      : 1;
  __REG32 TXUNDERRIE      : 1;
  __REG32 RXOVERRIE       : 1;
  __REG32 CMDRENDIE       : 1;
  __REG32 CMDSENTIE       : 1;
  __REG32 DATAENDIE       : 1;
  __REG32 STBITERRIE      : 1;
  __REG32 DBCKENDIE       : 1;
  __REG32 CMDACTIE        : 1;
  __REG32 TXACTIE         : 1;
  __REG32 RXACTIE         : 1;
  __REG32 TXFIFOHEIE      : 1;
  __REG32 RXFIFOHFIE      : 1;
  __REG32 TXFIFOFIE       : 1;
  __REG32 RXFIFOFIE       : 1;
  __REG32 TXFIFOEIE       : 1;
  __REG32 RXFIFOEIE       : 1;
  __REG32 TXDAVLIE        : 1;
  __REG32 RXDAVLIE        : 1;
  __REG32 SDIOITIE        : 1;
  __REG32 CEATAENDIE      : 1;
  __REG32                 : 8;
} __sdio_mask_bits;

/* SDIO FIFO Counter Register (SDIO_FIFOCNT) */
typedef struct {
  __REG32 FIFOCOUNT       :24;
  __REG32                 : 8;
} __sdio_fifocnt_bits;

/* MCU device ID code */
typedef struct {
  __REG32 DEV_ID          :12;
  __REG32                 : 4;
  __REG32 REV_ID          :16;
} __dbgmcu_idcode_bits;

/* Debug MCU configuration register */
typedef struct {
  __REG32 DBG_SLEEP               : 1;
  __REG32 DBG_STOP                : 1;
  __REG32 DBG_STANDBY             : 1;
  __REG32                         : 2;
  __REG32 TRACE_IOEN              : 1;
  __REG32 TRACE_MODE              : 2;
  __REG32                         :24;
} __dbgmcu_cr_bits;

/* Debug MCU APB1 freeze register (DBGMCU_APB1_FZ) */
typedef struct {
  __REG32 DBG_TIM2_STOP           : 1;
  __REG32 DBG_TIM3_STOP           : 1;
  __REG32 DBG_TIM4_STOP           : 1;
  __REG32                         : 1;
  __REG32 DBG_TIM6_STOP           : 1;
  __REG32 DBG_TIM7_STOP           : 1;
  __REG32                         : 5;
  __REG32 DBG_WWDG_STOP           : 1;
  __REG32 DBG_IWDG_STOP           : 1;
  __REG32                         : 8;
  __REG32 DBG_I2C1_SMBUS_TIMEOUT  : 1;
  __REG32 DBG_I2C2_SMBUS_TIMEOUT  : 1;
  __REG32                         : 9;
} __dbgmcu_apb1_fz_bits;

/* Debug MCU APB2 freeze register (DBGMCU_APB2_FZ) */
typedef struct {
  __REG32                         : 2;
  __REG32 DBG_TIM9_STOP           : 1;
  __REG32 DBG_TIM10_STOP          : 1;
  __REG32 DBG_TIM11_STOP          : 1;
  __REG32                         :27;
} __dbgmcu_apb2_fz_bits;

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
__IO_REG32_BIT(CLRENA0,           0xE000E180,__READ_WRITE ,__clrena0_bits);
__IO_REG32_BIT(CLRENA1,           0xE000E184,__READ_WRITE ,__clrena1_bits);
__IO_REG32_BIT(SETPEND0,          0xE000E200,__READ_WRITE ,__setpend0_bits);
__IO_REG32_BIT(SETPEND1,          0xE000E204,__READ_WRITE ,__setpend1_bits);
__IO_REG32_BIT(CLRPEND0,          0xE000E280,__READ_WRITE ,__clrpend0_bits);
__IO_REG32_BIT(CLRPEND1,          0xE000E284,__READ_WRITE ,__clrpend1_bits);
__IO_REG32_BIT(ACTIVE0,           0xE000E300,__READ       ,__active0_bits);
__IO_REG32_BIT(ACTIVE1,           0xE000E304,__READ       ,__active1_bits);
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
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(FLASH_ACR,         0x40023C00,__READ_WRITE ,__flash_acr_bits);
__IO_REG32_BIT(FLASH_PECR,        0x40023C04,__READ_WRITE ,__flash_pecr_bits);
__IO_REG32(    FLASH_PDKEYR,      0x40023C08,__WRITE      );
__IO_REG32(    FLASH_PEKEYR,      0x40023C0C,__WRITE      );
__IO_REG32(    FLASH_PRGKEYR,     0x40023C10,__WRITE      );
__IO_REG32(    FLASH_OPTKEYR,     0x40023C14,__WRITE      );
__IO_REG32_BIT(FLASH_SR,          0x40023C18,__READ_WRITE ,__flash_sr_bits);
__IO_REG32_BIT(FLASH_OBR,         0x40023C1C,__READ       ,__flash_obr_bits);
__IO_REG32(    FLASH_WRPR,        0x40023C20,__READ       );

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
__IO_REG32_BIT(RCC_CR,            0x40023800,__READ_WRITE ,__rcc_cr_bits);
__IO_REG32_BIT(RCC_ICSCR,         0x40023804,__READ_WRITE ,__rcc_icscr_bits);
__IO_REG32_BIT(RCC_CFGR,          0x40023808,__READ_WRITE ,__rcc_cfgr_bits);
__IO_REG32_BIT(RCC_CIR,           0x4002380C,__READ_WRITE ,__rcc_cir_bits);
__IO_REG32_BIT(RCC_AHBRSTR,       0x40023810,__READ_WRITE ,__rcc_ahbrstr_bits);
__IO_REG32_BIT(RCC_APB2RSTR,      0x40023814,__READ_WRITE ,__rcc_apb2rstr_bits);
__IO_REG32_BIT(RCC_APB1RSTR,      0x40023818,__READ_WRITE ,__rcc_apb1rstr_bits);
__IO_REG32_BIT(RCC_AHBENR,        0x4002381C,__READ_WRITE ,__rcc_ahbenr_bits);
__IO_REG32_BIT(RCC_APB2ENR,       0x40023820,__READ_WRITE ,__rcc_apb2enr_bits);
__IO_REG32_BIT(RCC_APB1ENR,       0x40023824,__READ_WRITE ,__rcc_apb1enr_bits);
__IO_REG32_BIT(RCC_AHBLPENR,      0x40023828,__READ_WRITE ,__rcc_ahblpenr_bits);
__IO_REG32_BIT(RCC_APB2LPENR,     0x4002382C,__READ_WRITE ,__rcc_apb2lpenr_bits);
__IO_REG32_BIT(RCC_APB1LPENR,     0x40023830,__READ_WRITE ,__rcc_apb1lpenr_bits);
__IO_REG32_BIT(RCC_CSR,           0x40023834,__READ_WRITE ,__rcc_csr_bits);

/***************************************************************************
 **
 ** GPIOA
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_MODER,       0x40020000,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOA_OTYPER,      0x40020004,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOA_OSPEEDR,     0x40020008,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOA_PUPDR,       0x4002000C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOA_IDR,         0x40020010,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOA_ODR,         0x40020014,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOA_BSRR,        0x40020018,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOA_LCKR,        0x4002001C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOA_AFRL,        0x40020020,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOA_AFRH,        0x40020024,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOB_MODER,       0x40020400,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOB_OTYPER,      0x40020404,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOB_OSPEEDR,     0x40020408,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOB_PUPDR,       0x4002040C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOB_IDR,         0x40020410,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOB_ODR,         0x40020414,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOB_BSRR,        0x40020418,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOB_LCKR,        0x4002041C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOB_AFRL,        0x40020420,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOB_AFRH,        0x40020424,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOC
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOC_MODER,       0x40020800,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOC_OTYPER,      0x40020804,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOC_OSPEEDR,     0x40020808,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOC_PUPDR,       0x4002080C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOC_IDR,         0x40020810,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOC_ODR,         0x40020814,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOC_BSRR,        0x40020818,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOC_LCKR,        0x4002081C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOC_AFRL,        0x40020820,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOC_AFRH,        0x40020824,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOD
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOD_MODER,       0x40020C00,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOD_OTYPER,      0x40020C04,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOD_OSPEEDR,     0x40020C08,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOD_PUPDR,       0x40020C0C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOD_IDR,         0x40020C10,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOD_ODR,         0x40020C14,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOD_BSRR,        0x40020C18,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOD_LCKR,        0x40020C1C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOD_AFRL,        0x40020C20,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOD_AFRH,        0x40020C24,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOE
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOE_MODER,       0x40021000,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOE_OTYPER,      0x40021004,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOE_OSPEEDR,     0x40021008,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOE_PUPDR,       0x4002100C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOE_IDR,         0x40021010,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOE_ODR,         0x40021014,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOE_BSRR,        0x40021018,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOE_LCKR,        0x4002101C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOE_AFRL,        0x40021020,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOE_AFRH,        0x40021024,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOH
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOH_MODER,       0x40021400,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOH_OTYPER,      0x40021404,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOH_OSPEEDR,     0x40021408,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOH_PUPDR,       0x4002140C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOH_IDR,         0x40021410,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOH_ODR,         0x40021414,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOH_BSRR,        0x40021418,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOH_LCKR,        0x4002141C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOH_AFRL,        0x40021420,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOH_AFRH,        0x40021424,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOF
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOF_MODER,       0x40021800,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOF_OTYPER,      0x40021804,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOF_OSPEEDR,     0x40021808,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOF_PUPDR,       0x4002180C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOF_IDR,         0x40021810,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOF_ODR,         0x40021814,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOF_BSRR,        0x40021818,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOF_LCKR,        0x4002181C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOF_AFRL,        0x40021820,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOF_AFRH,        0x40021824,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** GPIOG
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOG_MODER,       0x40021C00,__READ_WRITE ,__gpio_moder_bits);
__IO_REG32_BIT(GPIOG_OTYPER,      0x40021C04,__READ_WRITE ,__gpio_otyper_bits);
__IO_REG32_BIT(GPIOG_OSPEEDR,     0x40021C08,__READ_WRITE ,__gpio_ospeedr_bits);
__IO_REG32_BIT(GPIOG_PUPDR,       0x40021C0C,__READ_WRITE ,__gpio_pupdr_bits);
__IO_REG32_BIT(GPIOG_IDR,         0x40021C10,__READ       ,__gpio_idr_bits);
__IO_REG32_BIT(GPIOG_ODR,         0x40021C14,__READ_WRITE ,__gpio_odr_bits);
__IO_REG32_BIT(GPIOG_BSRR,        0x40021C18,__WRITE      ,__gpio_bsrr_bits);
__IO_REG32_BIT(GPIOG_LCKR,        0x40021C1C,__READ_WRITE ,__gpio_lckr_bits);
__IO_REG32_BIT(GPIOG_AFRL,        0x40021C20,__READ_WRITE ,__gpio_afrl_bits);
__IO_REG32_BIT(GPIOG_AFRH,        0x40021C24,__READ_WRITE ,__gpio_afrh_bits);

/***************************************************************************
 **
 ** RI
 **
 ***************************************************************************/
__IO_REG32_BIT(RI_ICR,            0x40007C04,__READ_WRITE ,__ri_icr_bits);
__IO_REG32_BIT(RI_ASCR1,          0x40007C08,__READ_WRITE ,__ri_ascr1_bits);
__IO_REG32_BIT(RI_ASCR2,          0x40007C0C,__READ_WRITE ,__ri_ascr2_bits);
__IO_REG32_BIT(RI_HYSCR1,         0x40007C10,__READ_WRITE ,__ri_hyscr1_bits);
__IO_REG32_BIT(RI_HYSCR2,         0x40007C14,__READ_WRITE ,__ri_hyscr2_bits);
__IO_REG32_BIT(RI_HYSCR3,         0x40007C18,__READ_WRITE ,__ri_hyscr3_bits);
__IO_REG32_BIT(RI_HYSCR4,         0x40007C1C,__READ_WRITE ,__ri_hyscr4_bits);
__IO_REG32_BIT(RI_ASMR1,         	0x40007C20,__READ_WRITE ,__ri_asmr1_bits);
__IO_REG32_BIT(RI_CMR1,         	0x40007C24,__READ_WRITE ,__ri_asmr1_bits);
__IO_REG32_BIT(RI_CICR1,         	0x40007C28,__READ_WRITE ,__ri_asmr1_bits);
__IO_REG32_BIT(RI_ASMR2,         	0x40007C2C,__READ_WRITE ,__ri_asmr2_bits);
__IO_REG32_BIT(RI_CMR2,         	0x40007C30,__READ_WRITE ,__ri_asmr2_bits);
__IO_REG32_BIT(RI_CICR2,         	0x40007C34,__READ_WRITE ,__ri_asmr2_bits);
__IO_REG32_BIT(RI_ASMR3,         	0x40007C38,__READ_WRITE ,__ri_asmr3_bits);
__IO_REG32_BIT(RI_CMR3,         	0x40007C3C,__READ_WRITE ,__ri_asmr3_bits);
__IO_REG32_BIT(RI_CICR3,         	0x40007C40,__READ_WRITE ,__ri_asmr3_bits);
__IO_REG32_BIT(RI_ASMR4,         	0x40007C44,__READ_WRITE ,__ri_asmr4_bits);
__IO_REG32_BIT(RI_CMR4,         	0x40007C48,__READ_WRITE ,__ri_asmr4_bits);
__IO_REG32_BIT(RI_CICR4,         	0x40007C4C,__READ_WRITE ,__ri_asmr4_bits);
__IO_REG32_BIT(RI_ASMR5,         	0x40007C50,__READ_WRITE ,__ri_asmr5_bits);
__IO_REG32_BIT(RI_CMR5,         	0x40007C54,__READ_WRITE ,__ri_asmr5_bits);
__IO_REG32_BIT(RI_CICR5,         	0x40007C58,__READ_WRITE ,__ri_asmr5_bits);

/***************************************************************************
 **
 ** SYSCFG
 **
 ***************************************************************************/
__IO_REG32_BIT(SYSCFG_MEMRMP,     0x40010000,__READ_WRITE ,__syscfg_memrmp_bits);
__IO_REG32_BIT(SYSCFG_PMC,        0x40010004,__READ_WRITE ,__syscfg_pmc_bits);
__IO_REG32_BIT(SYSCFG_EXTICR1,    0x40010008,__READ_WRITE ,__syscfg_exticr1_bits);
__IO_REG32_BIT(SYSCFG_EXTICR2,    0x4001000C,__READ_WRITE ,__syscfg_exticr2_bits);
__IO_REG32_BIT(SYSCFG_EXTICR3,    0x40010010,__READ_WRITE ,__syscfg_exticr3_bits);
__IO_REG32_BIT(SYSCFG_EXTICR4,    0x40010014,__READ_WRITE ,__syscfg_exticr4_bits);

/***************************************************************************
 **
 ** EXTI
 **
 ***************************************************************************/
__IO_REG32_BIT(EXTI_IMR,          0x40010400,__READ_WRITE ,__exti_imr_bits);
__IO_REG32_BIT(EXTI_EMR,          0x40010404,__READ_WRITE ,__exti_imr_bits);
__IO_REG32_BIT(EXTI_RTSR,         0x40010408,__READ_WRITE ,__exti_rtsr_bits);
__IO_REG32_BIT(EXTI_FTSR,         0x4001040C,__READ_WRITE ,__exti_rtsr_bits);
__IO_REG32_BIT(EXTI_SWIER,        0x40010410,__READ_WRITE ,__exti_swier_bits);
__IO_REG32_BIT(EXTI_PR,           0x40010414,__READ_WRITE ,__exti_pr_bits);

/***************************************************************************
 **
 ** DMA1
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA1_ISR,          0x40026000,__READ       ,__dma_isr_bits);
__IO_REG32_BIT(DMA1_IFCR,         0x40026004,__WRITE      ,__dma_ifcr_bits);
__IO_REG32_BIT(DMA1_CCR1,         0x40026008,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR1,       0x4002600C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR1,        0x40026010,__READ_WRITE );
__IO_REG32(    DMA1_CMAR1,        0x40026014,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR2,         0x4002601C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR2,       0x40026020,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR2,        0x40026024,__READ_WRITE );
__IO_REG32(    DMA1_CMAR2,        0x40026028,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR3,         0x40026030,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR3,       0x40026034,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR3,        0x40026038,__READ_WRITE );
__IO_REG32(    DMA1_CMAR3,        0x4002603C,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR4,         0x40026044,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR4,       0x40026048,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR4,        0x4002604C,__READ_WRITE );
__IO_REG32(    DMA1_CMAR4,        0x40026050,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR5,         0x40026058,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR5,       0x4002605C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR5,        0x40026060,__READ_WRITE );
__IO_REG32(    DMA1_CMAR5,        0x40026064,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR6,         0x4002606C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR6,       0x40026070,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR6,        0x40026074,__READ_WRITE );
__IO_REG32(    DMA1_CMAR6,        0x40026078,__READ_WRITE );
__IO_REG32_BIT(DMA1_CCR7,         0x40026080,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA1_CNDTR7,       0x40026084,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA1_CPAR7,        0x40026088,__READ_WRITE );
__IO_REG32(    DMA1_CMAR7,        0x4002608C,__READ_WRITE );

/***************************************************************************
 **
 ** DMA2
 **
 ***************************************************************************/
__IO_REG32_BIT(DMA2_ISR,          0x40026400,__READ       ,__dma_isr_bits);
__IO_REG32_BIT(DMA2_IFCR,         0x40026404,__WRITE      ,__dma_ifcr_bits);
__IO_REG32_BIT(DMA2_CCR1,         0x40026408,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR1,       0x4002640C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR1,        0x40026410,__READ_WRITE );
__IO_REG32(    DMA2_CMAR1,        0x40026414,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR2,         0x4002641C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR2,       0x40026420,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR2,        0x40026424,__READ_WRITE );
__IO_REG32(    DMA2_CMAR2,        0x40026428,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR3,         0x40026430,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR3,       0x40026434,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR3,        0x40026438,__READ_WRITE );
__IO_REG32(    DMA2_CMAR3,        0x4002643C,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR4,         0x40026444,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR4,       0x40026448,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR4,        0x4002644C,__READ_WRITE );
__IO_REG32(    DMA2_CMAR4,        0x40026450,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR5,         0x40026458,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR5,       0x4002645C,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR5,        0x40026460,__READ_WRITE );
__IO_REG32(    DMA2_CMAR5,        0x40026464,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR6,         0x4002646C,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR6,       0x40026470,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR6,        0x40026474,__READ_WRITE );
__IO_REG32(    DMA2_CMAR6,        0x40026478,__READ_WRITE );
__IO_REG32_BIT(DMA2_CCR7,         0x40026480,__READ_WRITE ,__dma_ccr_bits);
__IO_REG32_BIT(DMA2_CNDTR7,       0x40026484,__READ_WRITE ,__dma_cndtr_bits);
__IO_REG32(    DMA2_CPAR7,        0x40026488,__READ_WRITE );
__IO_REG32(    DMA2_CMAR7,        0x4002648C,__READ_WRITE );

/***************************************************************************
 **
 ** ADC
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC_SR,            0x40012400,__READ_WRITE ,__adc_sr_bits);
__IO_REG32_BIT(ADC_CR1,           0x40012404,__READ_WRITE ,__adc_cr1_bits);
__IO_REG32_BIT(ADC_CR2,           0x40012408,__READ_WRITE ,__adc_cr2_bits);
__IO_REG32_BIT(ADC_SMPR1,         0x4001240C,__READ_WRITE ,__adc_smpr1_bits);
__IO_REG32_BIT(ADC_SMPR2,         0x40012410,__READ_WRITE ,__adc_smpr2_bits);
__IO_REG32_BIT(ADC_SMPR3,         0x40012414,__READ_WRITE ,__adc_smpr3_bits);
__IO_REG32_BIT(ADC_JOFR1,         0x40012418,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC_JOFR2,         0x4001241C,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC_JOFR3,         0x40012420,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC_JOFR4,         0x40012424,__READ_WRITE ,__adc_jofr_bits);
__IO_REG32_BIT(ADC_HTR,           0x40012428,__READ_WRITE ,__adc_htr_bits);
__IO_REG32_BIT(ADC_LTR,           0x4001242C,__READ_WRITE ,__adc_ltr_bits);
__IO_REG32_BIT(ADC_SQR1,          0x40012430,__READ_WRITE ,__adc_sqr1_bits);
__IO_REG32_BIT(ADC_SQR2,          0x40012434,__READ_WRITE ,__adc_sqr2_bits);
__IO_REG32_BIT(ADC_SQR3,          0x40012438,__READ_WRITE ,__adc_sqr3_bits);
__IO_REG32_BIT(ADC_SQR4,          0x4001243C,__READ_WRITE ,__adc_sqr4_bits);
__IO_REG32_BIT(ADC_SQR5,          0x40012440,__READ_WRITE ,__adc_sqr5_bits);
__IO_REG32_BIT(ADC_JSQR,          0x40012444,__READ_WRITE ,__adc_jsqr_bits);
__IO_REG32_BIT(ADC_JDR1,          0x40012448,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC_JDR2,          0x4001244C,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC_JDR3,          0x40012450,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC_JDR4,          0x40012454,__READ       ,__adc_jdr_bits);
__IO_REG32_BIT(ADC_DR,            0x40012458,__READ       ,__adc_dr_bits);
__IO_REG32_BIT(ADC_SMPR0,         0x4001245C,__READ_WRITE	,__adc_smpr0_bits);
__IO_REG32_BIT(ADC_CSR,           0x40012700,__READ       ,__adc_csr_bits);
__IO_REG32_BIT(ADC_CCR,           0x40012704,__READ_WRITE ,__adc_ccr_bits);

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
__IO_REG32_BIT(DAC_DOR1,          0x4000742C,__READ       ,__dac_dor1_bits   );
__IO_REG32_BIT(DAC_DOR2,          0x40007430,__READ       ,__dac_dor2_bits   );
__IO_REG32_BIT(DAC_SR,            0x40007434,__READ_WRITE ,__dac_sr_bits     );

/***************************************************************************
 **
 ** COMP
 **
 ***************************************************************************/
__IO_REG32_BIT(COMP_CSR,          0x40007C00,__READ_WRITE ,__comp_csr_bits);

/***************************************************************************
 **
 ** OPAMP
 **
 ***************************************************************************/
__IO_REG32_BIT(OPAMP_CSR,         0x40007C5C,__READ_WRITE ,__opamp_csr_bits);
__IO_REG32_BIT(OPAMP_OTR,         0x40007C60,__READ_WRITE ,__opamp_otr_bits);
__IO_REG32_BIT(OPAMP_LPOTR,       0x40007C64,__READ_WRITE ,__opamp_otr_bits);

/***************************************************************************
 **
 ** LCD
 **
 ***************************************************************************/
__IO_REG32_BIT(LCD_CR,            0x40002400,__READ_WRITE ,__lcd_cr_bits);
__IO_REG32_BIT(LCD_FCR,           0x40002404,__READ_WRITE ,__lcd_fcr_bits);
__IO_REG32_BIT(LCD_SR,            0x40002408,__READ       ,__lcd_sr_bits);
__IO_REG32_BIT(LCD_CLR,           0x4000240C,__WRITE      ,__lcd_sr_bits);
__IO_REG32_BIT(LCD_RAM_COM00,     0x40002414,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM01,     0x40002418,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM10,     0x4000241C,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM11,     0x40002420,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM20,     0x40002424,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM21,     0x40002428,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM30,     0x4000242C,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM31,     0x40002430,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM40,     0x40002434,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM41,     0x40002438,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM50,     0x4000243C,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM51,     0x40002440,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM60,     0x40002444,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM61,     0x40002448,__READ_WRITE ,__lcd_ram1_bits);
__IO_REG32_BIT(LCD_RAM_COM70,     0x4000244C,__READ_WRITE ,__lcd_ram0_bits);
__IO_REG32_BIT(LCD_RAM_COM71,     0x40002450,__READ_WRITE ,__lcd_ram1_bits);

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
__IO_REG32(		 TIM5_CNT,          0x40000C24,__READ_WRITE );
__IO_REG32_BIT(TIM5_PSC,          0x40000C28,__READ_WRITE ,__tim_psc_bits);
__IO_REG32(		 TIM5_ARR,          0x40000C2C,__READ_WRITE );
__IO_REG32(		 TIM5_CCR1,         0x40000C34,__READ_WRITE );
__IO_REG32(		 TIM5_CCR2,         0x40000C38,__READ_WRITE );
__IO_REG32(		 TIM5_CCR3,         0x40000C3C,__READ_WRITE );
__IO_REG32(		 TIM5_CCR4,         0x40000C40,__READ_WRITE );
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
 ** TIM9
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM9_CR1,          0x40010800,__READ_WRITE ,__tim9_cr1_bits);
__IO_REG32_BIT(TIM9_CR2,          0x40010804,__READ_WRITE ,__tim9_cr2_bits);
__IO_REG32_BIT(TIM9_SMCR,         0x40010808,__READ_WRITE ,__tim9_smcr_bits);
__IO_REG32_BIT(TIM9_DIER,         0x4001080C,__READ_WRITE ,__tim9_dier_bits);
__IO_REG32_BIT(TIM9_SR,           0x40010810,__READ_WRITE ,__tim9_sr_bits);
__IO_REG32_BIT(TIM9_EGR,          0x40010814,__READ_WRITE ,__tim9_egr_bits);
__IO_REG32_BIT(TIM9_CCMR1,        0x40010818,__READ_WRITE ,__tim9_ccmr1_bits);
#define TIM9_OCMR1      TIM9_CCMR1
#define TIM9_OCMR1_bit  TIM9_CCMR1_bit
__IO_REG32_BIT(TIM9_CCER,         0x40010820,__READ_WRITE ,__tim9_ccer_bits);
__IO_REG32_BIT(TIM9_CNT,          0x40010824,__READ_WRITE ,__tim9_cnt_bits);
__IO_REG32_BIT(TIM9_PSC,          0x40010828,__READ_WRITE ,__tim9_psc_bits);
__IO_REG32_BIT(TIM9_ARR,          0x4001082C,__READ_WRITE ,__tim9_arr_bits);
__IO_REG32_BIT(TIM9_CCR1,         0x40010834,__READ_WRITE ,__tim9_ccr_bits);
__IO_REG32_BIT(TIM9_CCR2,         0x40010838,__READ_WRITE ,__tim9_ccr_bits);
__IO_REG32_BIT(TIM9_OR,           0x40010850,__READ_WRITE ,__tim9_or_bits);

/***************************************************************************
 **
 ** TIM10
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM10_CR1,         0x40010C00,__READ_WRITE ,__tim10_cr1_bits);
__IO_REG32_BIT(TIM10_SMCR,        0x40010C08,__READ_WRITE ,__tim10_smcr_bits);
__IO_REG32_BIT(TIM10_DIER,        0x40010C0C,__READ_WRITE ,__tim10_dier_bits);
__IO_REG32_BIT(TIM10_SR,          0x40010C10,__READ_WRITE ,__tim10_sr_bits);
__IO_REG32_BIT(TIM10_EGR,         0x40010C14,__READ_WRITE ,__tim10_egr_bits);
__IO_REG32_BIT(TIM10_CCMR1,       0x40010C18,__READ_WRITE ,__tim10_ccmr1_bits);
#define TIM10_OCMR1      TIM10_CCMR1
#define TIM10_OCMR1_bit  TIM10_CCMR1_bit
__IO_REG32_BIT(TIM10_CCER,        0x40010C20,__READ_WRITE ,__tim10_ccer_bits);
__IO_REG32_BIT(TIM10_CNT,         0x40010C24,__READ_WRITE ,__tim9_cnt_bits);
__IO_REG32_BIT(TIM10_PSC,         0x40010C28,__READ_WRITE ,__tim9_psc_bits);
__IO_REG32_BIT(TIM10_ARR,         0x40010C2C,__READ_WRITE ,__tim9_arr_bits);
__IO_REG32_BIT(TIM10_CCR1,        0x40010C34,__READ_WRITE ,__tim9_ccr_bits);
__IO_REG32_BIT(TIM10_OR,          0x40010C50,__READ_WRITE ,__tim9_or_bits);

/***************************************************************************
 **
 ** TIM11
 **
 ***************************************************************************/
__IO_REG32_BIT(TIM11_CR1,         0x40011000,__READ_WRITE ,__tim10_cr1_bits);
__IO_REG32_BIT(TIM11_SMCR,        0x40011008,__READ_WRITE ,__tim10_smcr_bits);
__IO_REG32_BIT(TIM11_DIER,        0x4001100C,__READ_WRITE ,__tim10_dier_bits);
__IO_REG32_BIT(TIM11_SR,          0x40011010,__READ_WRITE ,__tim10_sr_bits);
__IO_REG32_BIT(TIM11_EGR,         0x40011014,__READ_WRITE ,__tim10_egr_bits);
__IO_REG32_BIT(TIM11_CCMR1,       0x40011018,__READ_WRITE ,__tim10_ccmr1_bits);
#define TIM11_OCMR1      TIM11_CCMR1
#define TIM11_OCMR1_bit  TIM11_CCMR1_bit
__IO_REG32_BIT(TIM11_CCER,        0x40011020,__READ_WRITE ,__tim10_ccer_bits);
__IO_REG32_BIT(TIM11_CNT,         0x40011024,__READ_WRITE ,__tim9_cnt_bits);
__IO_REG32_BIT(TIM11_PSC,         0x40011028,__READ_WRITE ,__tim9_psc_bits);
__IO_REG32_BIT(TIM11_ARR,         0x4001102C,__READ_WRITE ,__tim9_arr_bits);
__IO_REG32_BIT(TIM11_CCR1,        0x40011034,__READ_WRITE ,__tim9_ccr_bits);
__IO_REG32_BIT(TIM11_OR,          0x40011050,__READ_WRITE ,__tim9_or_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_TR,            0x40002800,__READ_WRITE ,__rtc_tr_bits);
__IO_REG32_BIT(RTC_DR,            0x40002804,__READ_WRITE ,__rtc_dr_bits);
__IO_REG32_BIT(RTC_CR,            0x40002808,__READ_WRITE ,__rtc_cr_bits);
__IO_REG32_BIT(RTC_ISR,           0x4000280C,__READ_WRITE ,__rtc_isr_bits);
__IO_REG32_BIT(RTC_PRER,          0x40002810,__READ_WRITE ,__rtc_prer_bits);
__IO_REG32_BIT(RTC_WUTR,          0x40002814,__READ_WRITE ,__rtc_wutr_bits);
__IO_REG32_BIT(RTC_CALIBR,        0x40002818,__READ_WRITE ,__rtc_calibr_bits);
__IO_REG32_BIT(RTC_ALRMAR,        0x4000281C,__READ_WRITE ,__rtc_alrmar_bits);
__IO_REG32_BIT(RTC_ALRMBR,        0x40002820,__READ_WRITE ,__rtc_alrmar_bits);
__IO_REG32_BIT(RTC_WPR,           0x40002824,__WRITE      ,__rtc_wpr_bits);
__IO_REG32_BIT(RTC_SSR,           0x40002828,__READ       ,__rtc_ssr_bits);
__IO_REG32_BIT(RTC_SHIFTR,        0x4000282C,__WRITE      ,__rtc_shiftr_bits);
__IO_REG32_BIT(RTC_TSTR,          0x40002830,__READ       ,__rtc_tstr_bits);
__IO_REG32_BIT(RTC_TSDR,          0x40002834,__READ       ,__rtc_tsdr_bits);
__IO_REG32_BIT(RTC_TSSSR,         0x40002838,__READ       ,__rtc_ssr_bits);
__IO_REG32_BIT(RTC_CALR,          0x4000283C,__READ_WRITE ,__rtc_calr_bits);
__IO_REG32_BIT(RTC_TAFCR,         0x40002840,__READ_WRITE ,__rtc_tafcr_bits);
__IO_REG32_BIT(RTC_ALRMASSR,      0x40002844,__READ_WRITE ,__rtc_alrmassr_bits);
__IO_REG32_BIT(RTC_ALRMBSSR,      0x40002848,__READ_WRITE ,__rtc_alrmassr_bits);
__IO_REG32(    RTC_BK0R,          0x40002850,__READ_WRITE );
__IO_REG32(    RTC_BK1R,          0x40002854,__READ_WRITE );
__IO_REG32(    RTC_BK2R,          0x40002858,__READ_WRITE );
__IO_REG32(    RTC_BK3R,          0x4000285C,__READ_WRITE );
__IO_REG32(    RTC_BK4R,          0x40002860,__READ_WRITE );
__IO_REG32(    RTC_BK5R,          0x40002864,__READ_WRITE );
__IO_REG32(    RTC_BK6R,          0x40002868,__READ_WRITE );
__IO_REG32(    RTC_BK7R,          0x4000286C,__READ_WRITE );
__IO_REG32(    RTC_BK8R,          0x40002870,__READ_WRITE );
__IO_REG32(    RTC_BK9R,          0x40002874,__READ_WRITE );
__IO_REG32(    RTC_BK10R,         0x40002878,__READ_WRITE );
__IO_REG32(    RTC_BK11R,         0x4000287C,__READ_WRITE );
__IO_REG32(    RTC_BK12R,         0x40002880,__READ_WRITE );
__IO_REG32(    RTC_BK13R,         0x40002884,__READ_WRITE );
__IO_REG32(    RTC_BK14R,         0x40002888,__READ_WRITE );
__IO_REG32(    RTC_BK15R,         0x4000288C,__READ_WRITE );
__IO_REG32(    RTC_BK16R,         0x40002890,__READ_WRITE );
__IO_REG32(    RTC_BK17R,         0x40002894,__READ_WRITE );
__IO_REG32(    RTC_BK18R,         0x40002898,__READ_WRITE );
__IO_REG32(    RTC_BK19R,         0x4000289C,__READ_WRITE );
__IO_REG32(    RTC_BK20R,         0x400028A0,__READ_WRITE );
__IO_REG32(    RTC_BK21R,         0x400028A4,__READ_WRITE );
__IO_REG32(    RTC_BK22R,         0x400028A8,__READ_WRITE );
__IO_REG32(    RTC_BK23R,         0x400028AC,__READ_WRITE );
__IO_REG32(    RTC_BK24R,         0x400028B0,__READ_WRITE );
__IO_REG32(    RTC_BK25R,         0x400028B4,__READ_WRITE );
__IO_REG32(    RTC_BK26R,         0x400028B8,__READ_WRITE );
__IO_REG32(    RTC_BK27R,         0x400028BC,__READ_WRITE );
__IO_REG32(    RTC_BK28R,         0x400028C0,__READ_WRITE );
__IO_REG32(    RTC_BK29R,         0x400028C4,__READ_WRITE );
__IO_REG32(    RTC_BK30R,         0x400028C8,__READ_WRITE );
__IO_REG32(    RTC_BK31R,         0x400028CC,__READ_WRITE );

/***************************************************************************
 **
 ** IWDG
 **
 ***************************************************************************/
__IO_REG32(    IWDG_KR,           0x40003000,__WRITE      );
__IO_REG32_BIT(IWDG_PR,           0x40003004,__READ_WRITE ,__iwdg_pr_bits);
__IO_REG32_BIT(IWDG_RLR,          0x40003008,__READ_WRITE ,__iwdg_rlr_bits);
__IO_REG32_BIT(IWDG_SR,           0x4000300C,__READ       ,__iwdg_sr_bits);

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
 ** USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_EP0R,          0x40005C00,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP1R,          0x40005C04,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP2R,          0x40005C08,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP3R,          0x40005C0C,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP4R,          0x40005C10,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP5R,          0x40005C14,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP6R,          0x40005C18,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_EP7R,          0x40005C1C,__READ_WRITE ,__usb_epr_bits);
__IO_REG32_BIT(USB_CNTR,          0x40005C40,__READ_WRITE ,__usb_cntr_bits);
__IO_REG32_BIT(USB_ISTR,          0x40005C44,__READ_WRITE ,__usb_istr_bits);
__IO_REG32_BIT(USB_FNR,           0x40005C48,__READ       ,__usb_fnr_bits);
__IO_REG32_BIT(USB_DADDR,         0x40005C4C,__READ_WRITE ,__usb_daddr_bits);
__IO_REG32_BIT(USB_BTABLE,        0x40005C50,__READ_WRITE ,__usb_btable_bits);

/***************************************************************************
 **
 ** FSMC
 **
 ***************************************************************************/
__IO_REG32_BIT(FSMC_BCR1,         0xA0000000,__READ_WRITE ,__fsmc_bcr_bits);
__IO_REG32_BIT(FSMC_BTR1,         0xA0000004,__READ_WRITE ,__fsmc_btr_bits);
__IO_REG32_BIT(FSMC_BCR2,         0xA0000008,__READ_WRITE ,__fsmc_bcr_bits);
__IO_REG32_BIT(FSMC_BTR2,         0xA000000C,__READ_WRITE ,__fsmc_btr_bits);
__IO_REG32_BIT(FSMC_BCR3,         0xA0000010,__READ_WRITE ,__fsmc_bcr_bits);
__IO_REG32_BIT(FSMC_BTR3,         0xA0000014,__READ_WRITE ,__fsmc_btr_bits);
__IO_REG32_BIT(FSMC_BCR4,         0xA0000018,__READ_WRITE ,__fsmc_bcr_bits);
__IO_REG32_BIT(FSMC_BTR4,         0xA000001C,__READ_WRITE ,__fsmc_btr_bits);
__IO_REG32_BIT(FSMC_BWTR1,        0xA0000104,__READ_WRITE ,__fsmc_bwtr_bits);
__IO_REG32_BIT(FSMC_BWTR2,        0xA000010C,__READ_WRITE ,__fsmc_bwtr_bits);
__IO_REG32_BIT(FSMC_BWTR3,        0xA0000114,__READ_WRITE ,__fsmc_bwtr_bits);
__IO_REG32_BIT(FSMC_BWTR4,        0xA000011C,__READ_WRITE ,__fsmc_bwtr_bits);

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
__IO_REG32_BIT(I2C2_SR2,          0x40005818,__READ       ,__i2c_sr2_bits);
__IO_REG32_BIT(I2C2_CCR,          0x4000581C,__READ_WRITE ,__i2c_ccr_bits);
__IO_REG32_BIT(I2C2_TRISE,        0x40005820,__READ_WRITE ,__i2c_trise_bits);

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
 ** USART4
 **
 ***************************************************************************/
__IO_REG32_BIT(USART4_SR,         0x40004C00,__READ_WRITE ,__usart_sr_bits);
__IO_REG32_BIT(USART4_DR,         0x40004C04,__READ_WRITE ,__usart_dr_bits);
__IO_REG32_BIT(USART4_BRR,        0x40004C08,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART4_CR1,        0x40004C0C,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART4_CR2,        0x40004C10,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART4_CR3,        0x40004C14,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART4_GTPR,       0x40004C18,__READ_WRITE ,__usart_gtpr_bits);

/***************************************************************************
 **
 ** USART5
 **
 ***************************************************************************/
__IO_REG32_BIT(USART5_SR,         0x40005000,__READ_WRITE ,__usart_sr_bits);
__IO_REG32_BIT(USART5_DR,         0x40005004,__READ_WRITE ,__usart_dr_bits);
__IO_REG32_BIT(USART5_BRR,        0x40005008,__READ_WRITE ,__usart_brr_bits);
__IO_REG32_BIT(USART5_CR1,        0x4000500C,__READ_WRITE ,__usart_cr1_bits);
__IO_REG32_BIT(USART5_CR2,        0x40005010,__READ_WRITE ,__usart_cr2_bits);
__IO_REG32_BIT(USART5_CR3,        0x40005014,__READ_WRITE ,__usart_cr3_bits);
__IO_REG32_BIT(USART5_GTPR,       0x40005018,__READ_WRITE ,__usart_gtpr_bits);

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
__IO_REG32_BIT(SPI1_I2SCFGR,      0x4001301C,__READ_WRITE ,__spi_i2scfgr_bits);
__IO_REG32_BIT(SPI1_I2SPR,        0x40013020,__READ_WRITE ,__spi_i2spr_bits);

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
 ** SDIO
 **
 ***************************************************************************/
__IO_REG32_BIT(SDIO_POWER,        0x40012C00,__READ_WRITE ,__sdio_power_bits  );
__IO_REG32_BIT(SDIO_CLKCR,        0x40012C04,__READ_WRITE ,__sdio_clkcr_bits  );
__IO_REG32(    SDIO_ARG,          0x40012C08,__READ_WRITE                     );
__IO_REG32_BIT(SDIO_CMD,          0x40012C0C,__READ_WRITE ,__sdio_cmd_bits    );
__IO_REG32_BIT(SDIO_RESPCMD,      0x40012C10,__READ       ,__sdio_respcmd_bits);
__IO_REG32(    SDIO_RESP1,        0x40012C14,__READ                           );
__IO_REG32(    SDIO_RESP2,        0x40012C18,__READ                           );
__IO_REG32(    SDIO_RESP3,        0x40012C1C,__READ                           );
__IO_REG32(    SDIO_RESP4,        0x40012C20,__READ                           );
__IO_REG32(    SDIO_DTIMER,       0x40012C24,__READ_WRITE                     );
__IO_REG32_BIT(SDIO_DLEN,         0x40012C28,__READ_WRITE ,__sdio_dlen_bits   );
__IO_REG32_BIT(SDIO_DCTRL,        0x40012C2C,__READ_WRITE ,__sdio_dctrl_bits  );
__IO_REG32_BIT(SDIO_DCOUNT,       0x40012C30,__READ       ,__sdio_dcount_bits );
__IO_REG32_BIT(SDIO_STA,          0x40012C34,__READ       ,__sdio_sta_bits    );
__IO_REG32_BIT(SDIO_ICR,          0x40012C38,__READ_WRITE ,__sdio_icr_bits    );
__IO_REG32_BIT(SDIO_MASK,         0x40012C3C,__READ_WRITE ,__sdio_mask_bits   );
__IO_REG32_BIT(SDIO_FIFOCNT,      0x40012C48,__READ       ,__sdio_fifocnt_bits);
__IO_REG32(    SDIO_FIFO,         0x40012C80,__READ_WRITE                     );

/***************************************************************************
 **
 ** DBG
 **
 ***************************************************************************/
__IO_REG32_BIT(DBGMCU_IDCODE,     0xE0042000,__READ       ,__dbgmcu_idcode_bits);
__IO_REG32_BIT(DBGMCU_CR,         0xE0042004,__READ_WRITE ,__dbgmcu_cr_bits);
__IO_REG32_BIT(DBGMCU_APB1_FZ,    0xE0042008,__READ_WRITE ,__dbgmcu_apb1_fz_bits);
__IO_REG32_BIT(DBGMCU_APB2_FZ,    0xE004200C,__READ_WRITE ,__dbgmcu_apb2_fz_bits);

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  STM32L15xxC Interrupt Lines
 **
 ***************************************************************************/
#define MAIN_STACK               0          /* Main Stack                   */
#define RESETI                   1          /* Reset                        */
#define NMII                     2          /* Non-maskable Interrupt       */
#define HFI                      3          /* Hard Fault                   */
#define MMI                      4          /* Memory Management            */
#define BFI                      5          /* Bus Fault                    */
#define UFI                      6          /* Usage Fault                  */
#define SVCI                    11          /* SVCall                       */
#define DMI                     12          /* Debug Monitor                */
#define PSI                     14          /* PendSV                       */
#define STI                     15          /* SysTick                      */
#define WWDG                    16          /* Window Watchdog interrupt    */
#define NVIC_PVD                17          /* PVD through EXTI Line detection interrupt*/
#define NVIC_TAMPER             18          /* Tamper interrupt             */
#define NVIC_RTC                19          /* RTC global interrupt         */
#define NVIC_FLASH              20          /* Flash global interrupt       */
#define NVIC_RCC                21          /* RCC global interrupt         */
#define NVIC_EXTI0              22          /* EXTI Line0 interrupt         */
#define NVIC_EXTI1              23          /* EXTI Line1 interrupt         */
#define NVIC_EXTI2              24          /* EXTI Line2 interrupt         */
#define NVIC_EXTI3              25          /* EXTI Line3 interrupt         */
#define NVIC_EXTI4              26          /* EXTI Line4 interrupt         */
#define NVIC_DMA_CH1            27          /* DMA Channel1 global interrupt*/
#define NVIC_DMA_CH2            28          /* DMA Channel2 global interrupt*/
#define NVIC_DMA_CH3            29          /* DMA Channel3 global interrupt*/
#define NVIC_DMA_CH4            30          /* DMA Channel4 global interrupt*/
#define NVIC_DMA_CH5            31          /* DMA Channel5 global interrupt*/
#define NVIC_DMA_CH6            32          /* DMA Channel6 global interrupt*/
#define NVIC_DMA_CH7            33          /* DMA Channel7 global interrupt*/
#define NVIC_ADC1               34          /* ADC global interrupt         */
#define NVIC_USB_HP             35          /* USB High Priority interrupt  */
#define NVIC_USB_LP             36          /* USB Low Priority interrupt   */
#define NVIC_DAC                37          /* DAC interrupt                */
#define NVIC_COMP               38          /* Comparator wakeup through EXTI line (21 and 22) interrupt */
#define NVIC_EXTI9_5            39          /* EXTI Line[9:5] interrupts    */
#define NVIC_LCD                40          /* LCD global interrupt         */
#define NVIC_TIM9               41          /* TIM9 global interrupt        */
#define NVIC_TIM10              42          /* TIM10 global interrupt       */
#define NVIC_TIM11              43          /* TIM11 global interrupt       */
#define NVIC_TIM2               44          /* TIM2 global interrupt        */
#define NVIC_TIM3               45          /* TIM3 global interrupt        */
#define NVIC_TIM4               46          /* TIM4 global interrupt        */
#define NVIC_I2C1_EV            47          /* I2C1 event interrupt         */
#define NVIC_I2C1_ER            48          /* I2C1 error interrupt         */
#define NVIC_I2C2_EV            49          /* I2C2 event interrupt         */
#define NVIC_I2C2_ER            50          /* I2C2 error interrupt         */
#define NVIC_SPI1               51          /* SPI1 global interrupt        */
#define NVIC_SPI2               52          /* SPI2 global interrupt        */
#define NVIC_USART1             53          /* USART1 global interrupt      */
#define NVIC_USART2             54          /* USART2 global interrupt      */
#define NVIC_USART3             55          /* USART3 global interrupt      */
#define NVIC_EXTI15_10          56          /* EXTI Line[15:10] interrupts  */
#define NVIC_RTC_ALARM          57          /* RTC alarm through EXTI line interrupt */
#define NVIC_USB_WAKE_UP        58          /* USB wakeup from suspend through EXTI line interrupt */
#define NVIC_TIM6               59          /* TIM6 global interrupt        */
#define NVIC_TIM7               60          /* TIM7 global interrupt        */
#define NVIC_SDIO               61          /* SDIO Global interrupt	      */
#define NVIC_TIM5               62          /* TIM5 global interrupt        */
#define NVIC_SPI3               63          /* SPI3 global interrupt        */
#define NVIC_USART4             64          /* USART4 Global interrupt      */
#define NVIC_USART5             65          /* USART5 Global interrupt      */
#define NVIC_DMA2_CH1           66          /* DMA2 Channel 1 interrupt     */
#define NVIC_DMA2_CH2           67          /* DMA2 Channel 2 interrupt     */
#define NVIC_DMA2_CH3           68          /* DMA2 Channel 3 interrupt     */
#define NVIC_DMA2_CH4           69          /* DMA2 Channel 4 interrupt     */
#define NVIC_DMA2_CH5           70          /* DMA2 Channel 5 interrupt     */

#endif    /* __IOSTM32L15xxC_H */

/*###DDF-INTERRUPT-BEGIN###
Interrupt0   = NMI                0x08
Interrupt1   = HardFault          0x0C
Interrupt2   = MemManage          0x10
Interrupt3   = BusFault           0x14
Interrupt4   = UsageFault         0x18
Interrupt5   = SVC                0x2C
Interrupt6   = DebugMon           0x30
Interrupt7   = PendSV             0x38
Interrupt8   = SysTick            0x3C
Interrupt9   = WWDG               0x40
Interrupt10  = PVD                0x44
Interrupt11  = TAMPER             0x48
Interrupt12  = RTC                0x4C
Interrupt13  = FLASH              0x50
Interrupt14  = RCC                0x54
Interrupt15  = EXTI0              0x58
Interrupt16  = EXTI1              0x5C
Interrupt17  = EXTI2              0x60
Interrupt18  = EXTI3              0x64
Interrupt19  = EXTI4              0x68
Interrupt20  = DMA_CH1            0x6C
Interrupt21  = DMA_CH2            0x70
Interrupt22  = DMA_CH3            0x74
Interrupt23  = DMA_CH4            0x78
Interrupt24  = DMA_CH5            0x7C
Interrupt25  = DMA_CH6            0x80
Interrupt26  = DMA_CH7            0x84
Interrupt27  = ADC1               0x88
Interrupt28  = USB_HP_TX          0x8C
Interrupt29  = USB_LP_RX          0x90
Interrupt30  = DAC                0x94
Interrupt31  = COMP               0x98
Interrupt32  = EXTI9_5            0x9C
Interrupt33  = LCD                0xA0
Interrupt34  = TIM9               0xA4
Interrupt35  = TIM10              0xA8
Interrupt36  = TIM11              0xAC
Interrupt37  = TIM2               0xB0
Interrupt38  = TIM3               0xB4
Interrupt39  = TIM4               0xB8
Interrupt40  = I2C1_EV            0xBC
Interrupt41  = I2C1_ER            0xC0
Interrupt42  = I2C2_EV            0xC4
Interrupt43  = I2C2_ER            0xC8
Interrupt44  = SPI1               0xCC
Interrupt45  = SPI2               0xD0
Interrupt46  = USART1             0xD4
Interrupt47  = USART2             0xD8
Interrupt48  = USART3             0xDC
Interrupt49  = EXTI15_10          0xE0
Interrupt50  = RTC_ALARM          0xE4
Interrupt51  = USB_WAKE_UP        0xE8
Interrupt52  = TIM6               0xEC
Interrupt53  = TIM7               0xF0
Interrupt54  = NVIC_SDIO          0xF4
Interrupt55  = NVIC_TIM5          0xF8
Interrupt56  = NVIC_SPI3          0xFC
Interrupt57  = NVIC_USART4        0x100
Interrupt58  = NVIC_USART5        0x104
Interrupt59  = NVIC_DMA2_CH1      0x108
Interrupt60  = NVIC_DMA2_CH2      0x10C
Interrupt61  = NVIC_DMA2_CH3      0x110
Interrupt62  = NVIC_DMA2_CH4      0x114
Interrupt63  = NVIC_DMA2_CH5      0x118
###DDF-INTERRUPT-END###*/
