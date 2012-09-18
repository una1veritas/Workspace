/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    NXP LPC1810 rev "-"
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2010
 **
 **    $Revision: 48961 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOLPC1810_REV_H
#define __IOLPC1810_REV_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    LPC1810 SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/

/* C-compiler specific declarations  ***************************************/
#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
#pragma system_include
#endif

#if __LITTLE_ENDIAN__ == 0
#error This file should only be compiled in little endian mode
#endif

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
  __REG32  ISE_DAC        : 1;
  __REG32  ISE_WIC        : 1;
  __REG32  ISE_DMA        : 1;
  __REG32                 : 3;
  __REG32  ISE_SDIO       : 1;
  __REG32                 : 3;
  __REG32  ISE_SCT        : 1;
  __REG32  ISE_RIT        : 1;
  __REG32  ISE_TIMER0     : 1;
  __REG32  ISE_TIMER1     : 1;
  __REG32  ISE_TIMER2     : 1;
  __REG32  ISE_TIMER3     : 1;
  __REG32  ISE_MOTOCONPWM : 1;
  __REG32  ISE_ADC0       : 1;
  __REG32  ISE_I2C0       : 1;
  __REG32  ISE_I2C1       : 1;
  __REG32                 : 1;
  __REG32  ISE_ADC1       : 1;
  __REG32  ISE_SSP0       : 1;
  __REG32  ISE_SSP1       : 1;
  __REG32  ISE_USART0     : 1;
  __REG32  ISE_UART1      : 1;
  __REG32  ISE_USART2     : 1;
  __REG32  ISE_USART3     : 1;
  __REG32  ISE_I2S        : 1;
  __REG32  ISE_AES        : 1;
  __REG32  ISE_SPIFI      : 1;
  __REG32                 : 1;
} __setena0_bits;

/* Interrupt Clear-Enable Registers 0-31 */
typedef struct {
  __REG32  ICE_DAC        : 1;
  __REG32  ICE_WIC        : 1;
  __REG32  ICE_DMA        : 1;
  __REG32                 : 3;
  __REG32  ICE_SDIO       : 1;
  __REG32                 : 3;
  __REG32  ICE_SCT        : 1;
  __REG32  ICE_RIT        : 1;
  __REG32  ICE_TIMER0     : 1;
  __REG32  ICE_TIMER1     : 1;
  __REG32  ICE_TIMER2     : 1;
  __REG32  ICE_TIMER3     : 1;
  __REG32  ICE_MOTOCONPWM : 1;
  __REG32  ICE_ADC0       : 1;
  __REG32  ICE_I2C0       : 1;
  __REG32  ICE_I2C1       : 1;
  __REG32                 : 1;
  __REG32  ICE_ADC1       : 1;
  __REG32  ICE_SSP0       : 1;
  __REG32  ICE_SSP1       : 1;
  __REG32  ICE_USART0     : 1;
  __REG32  ICE_UART1      : 1;
  __REG32  ICE_USART2     : 1;
  __REG32  ICE_USART3     : 1;
  __REG32  ICE_I2S        : 1;
  __REG32  ICE_AES        : 1;
  __REG32  ICE_SPIFI      : 1;
  __REG32                 : 1;
} __clrena0_bits;

/* Interrupt Set-Pending Register 0-31 */
typedef struct {
  __REG32  ISP_DAC        : 1;
  __REG32  ISP_WIC        : 1;
  __REG32  ISP_DMA        : 1;
  __REG32                 : 3;
  __REG32  ISP_SDIO       : 1;
  __REG32                 : 3;
  __REG32  ISP_SCT        : 1;
  __REG32  ISP_RIT        : 1;
  __REG32  ISP_TIMER0     : 1;
  __REG32  ISP_TIMER1     : 1;
  __REG32  ISP_TIMER2     : 1;
  __REG32  ISP_TIMER3     : 1;
  __REG32  ISP_MOTOCONPWM : 1;
  __REG32  ISP_ADC0       : 1;
  __REG32  ISP_I2C0       : 1;
  __REG32  ISP_I2C1       : 1;
  __REG32                 : 1;
  __REG32  ISP_ADC1       : 1;
  __REG32  ISP_SSP0       : 1;
  __REG32  ISP_SSP1       : 1;
  __REG32  ISP_USART0     : 1;
  __REG32  ISP_UART1      : 1;
  __REG32  ISP_USART2     : 1;
  __REG32  ISP_USART3     : 1;
  __REG32  ISP_I2S        : 1;
  __REG32  ISP_AES        : 1;
  __REG32  ISP_SPIFI      : 1;
  __REG32                 : 1;
} __setpend0_bits;

/* Interrupt Clear-Pending Register 0-31 */
typedef struct {
  __REG32  ICP_DAC        : 1;
  __REG32  ICP_WIC        : 1;
  __REG32  ICP_DMA        : 1;
  __REG32                 : 3;
  __REG32  ICP_SDIO       : 1;
  __REG32                 : 3;
  __REG32  ICP_SCT        : 1;
  __REG32  ICP_RIT        : 1;
  __REG32  ICP_TIMER0     : 1;
  __REG32  ICP_TIMER1     : 1;
  __REG32  ICP_TIMER2     : 1;
  __REG32  ICP_TIMER3     : 1;
  __REG32  ICP_MOTOCONPWM : 1;
  __REG32  ICP_ADC0       : 1;
  __REG32  ICP_I2C0       : 1;
  __REG32  ICP_I2C1       : 1;
  __REG32                 : 1;
  __REG32  ICP_ADC1       : 1;
  __REG32  ICP_SSP0       : 1;
  __REG32  ICP_SSP1       : 1;
  __REG32  ICP_USART0     : 1;
  __REG32  ICP_UART1      : 1;
  __REG32  ICP_USART2     : 1;
  __REG32  ICP_USART3     : 1;
  __REG32  ICP_I2S        : 1;
  __REG32  ICP_AES        : 1;
  __REG32  ICP_SPIFI      : 1;
  __REG32                 : 1;
} __clrpend0_bits;

/* Active Bit Register 0-31 */
typedef struct {
  __REG32  IABP_DAC       : 1;
  __REG32  IABP_WIC       : 1;
  __REG32  IABP_DMA       : 1;
  __REG32                 : 3;
  __REG32  IAB_SDIO       : 1;
  __REG32                 : 3;
  __REG32  IAB_SCT        : 1;
  __REG32  IAB_RIT        : 1;
  __REG32  IAB_TIMER0     : 1;
  __REG32  IAB_TIMER1     : 1;
  __REG32  IAB_TIMER2     : 1;
  __REG32  IAB_TIMER3     : 1;
  __REG32  IAB_MOTOCONPWM : 1;
  __REG32  IAB_ADC0       : 1;
  __REG32  IAB_I2C0       : 1;
  __REG32  IAB_I2C1       : 1;
  __REG32                 : 1;
  __REG32  IAB_ADC1       : 1;
  __REG32  IAB_SSP0       : 1;
  __REG32  IAB_SSP1       : 1;
  __REG32  IAB_USART0     : 1;
  __REG32  IAB_UART1      : 1;
  __REG32  IAB_USART2     : 1;
  __REG32  IAB_USART3     : 1;
  __REG32  IAB_I2S        : 1;
  __REG32  IAB_AES        : 1;
  __REG32  IAB_SPIFI      : 1;
  __REG32                 : 1;
} __active0_bits;

/* Interrupt Priority Registers 0-3 */
typedef struct {
  __REG32                 : 3;
  __REG32  IP_DAC         : 5;
  __REG32                 : 3;
  __REG32  IP_WIC         : 5;
  __REG32                 : 3;
  __REG32  IP_DMA         : 5;
  __REG32                 : 8;
} __pri0_bits;

/* Interrupt Priority Registers 4-7 */
typedef struct {
  __REG32                 :19;
  __REG32  IP_SDIO        : 5;
  __REG32                 : 8;
} __pri1_bits;

/* Interrupt Priority Registers 8-11 */
typedef struct {
  __REG32                 :19;
  __REG32  IP_SCT         : 5;
  __REG32                 : 3;
  __REG32  IP_RIT         : 5;
} __pri2_bits;

/* Interrupt Priority Registers 12-15 */
typedef struct {
  __REG32                 : 3;
  __REG32  IP_TIMER0      : 5;
  __REG32                 : 3;
  __REG32  IP_TIMER1      : 5;
  __REG32                 : 3;
  __REG32  IP_TIMER2      : 5;
  __REG32                 : 3;
  __REG32  IP_TIMER3      : 5;
} __pri3_bits;

/* Interrupt Priority Registers 16-19 */
typedef struct {
  __REG32                 : 3;
  __REG32  IP_MOTOCONPWM  : 5;
  __REG32                 : 3;
  __REG32  IP_ADC0        : 5;
  __REG32                 : 3;
  __REG32  IP_I2C0        : 5;
  __REG32                 : 3;
  __REG32  IP_I2C1        : 5;
} __pri4_bits;

/* Interrupt Priority Registers 20-23 */
typedef struct {
  __REG32                 :11;
  __REG32  IP_ADC1        : 5;
  __REG32                 : 3;
  __REG32  IP_SSP0        : 5;
  __REG32                 : 3;
  __REG32  IP_SSP1        : 5;
} __pri5_bits;

/* Interrupt Priority Registers 24-27 */
typedef struct {
  __REG32                 : 3;
  __REG32  IP_USART0      : 5;
  __REG32                 : 3;
  __REG32  IP_UART1       : 5;
  __REG32                 : 3;
  __REG32  IP_USART2      : 5;
  __REG32                 : 3;
  __REG32  IP_USART3      : 5;
} __pri6_bits;

/* Interrupt Priority Registers 28-31 */
typedef struct {
  __REG32                 : 3;
  __REG32  IP_I2S         : 5;
  __REG32                 : 3;
  __REG32  IP_AES         : 5;
  __REG32                 : 3;
  __REG32  IP_SPIFI       : 5;
  __REG32                 : 8;
} __pri7_bits;

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

/* Level configuration register */
typedef struct {
  __REG32  WAKEUP0_L      : 1;
  __REG32  WAKEUP1_L      : 1;
  __REG32  WAKEUP2_L      : 1;
  __REG32  WAKEUP3_L      : 1;
  __REG32  ATIMER_L       : 1;
  __REG32  RTC_L          : 1;
  __REG32  BOD_L          : 1;
  __REG32  WWDT_L         : 1;
  __REG32  			          : 4;
  __REG32  CAN_L          : 1;
  __REG32  TIM2_L         : 1;
  __REG32  TIM6_L         : 1;
  __REG32  QEI_L          : 1;
  __REG32  TIM14_L        : 1;
  __REG32                 : 2;
  __REG32  RESET_L        : 1;
  __REG32                 :12;
} __er_hilo_bits;

/* Edge configuration register */
typedef struct {
  __REG32  WAKEUP0_E      : 1;
  __REG32  WAKEUP1_E      : 1;
  __REG32  WAKEUP2_E      : 1;
  __REG32  WAKEUP3_E      : 1;
  __REG32  ATIMER_E       : 1;
  __REG32  RTC_E          : 1;
  __REG32  BOD_E          : 1;
  __REG32  WWDT_E         : 1;
  __REG32  			          : 4;
  __REG32  CAN_E          : 1;
  __REG32  TIM2_E         : 1;
  __REG32  TIM6_E         : 1;
  __REG32  QEI_E          : 1;
  __REG32  TIM14_E        : 1;
  __REG32                 : 2;
  __REG32  RESET_E        : 1;
  __REG32                 :12;
} __er_edge_bits;

/* Interrupt clear enable register */
typedef struct {
  __REG32  WAKEUP0_CLREN  : 1;
  __REG32  WAKEUP1_CLREN  : 1;
  __REG32  WAKEUP2_CLREN  : 1;
  __REG32  WAKEUP3_CLREN  : 1;
  __REG32  ATIMER_CLREN   : 1;
  __REG32  RTC_CLREN      : 1;
  __REG32  BOD_CLREN      : 1;
  __REG32  WWDT_CLREN     : 1;
  __REG32  					      : 4;
  __REG32  CAN_CLREN      : 1;
  __REG32  TIM2_CLREN     : 1;
  __REG32  TIM6_CLREN     : 1;
  __REG32  QEI_CLREN      : 1;
  __REG32  TIM14_CLREN    : 1;
  __REG32                 : 2;
  __REG32  RESET_CLREN    : 1;
  __REG32                 :12;
} __er_clr_en_bits;

/* Interrupt set enable register */
typedef struct {
  __REG32  WAKEUP0_SETEN  : 1;
  __REG32  WAKEUP1_SETEN  : 1;
  __REG32  WAKEUP2_SETEN  : 1;
  __REG32  WAKEUP3_SETEN  : 1;
  __REG32  ATIMER_SETEN   : 1;
  __REG32  RTC_SETEN      : 1;
  __REG32  BOD_SETEN      : 1;
  __REG32  WWDT_SETEN     : 1;
  __REG32  					      : 4;
  __REG32  CAN_SETEN      : 1;
  __REG32  TIM2_SETEN     : 1;
  __REG32  TIM6_SETEN     : 1;
  __REG32  QEI_SETEN      : 1;
  __REG32  TIM14_SETEN    : 1;
  __REG32                 : 2;
  __REG32  RESET_SETEN    : 1;
  __REG32                 :12;
} __er_set_en_bits;

/* Interrupt status register */
typedef struct {
  __REG32  WAKEUP0_ST     : 1;
  __REG32  WAKEUP1_ST     : 1;
  __REG32  WAKEUP2_ST     : 1;
  __REG32  WAKEUP3_ST     : 1;
  __REG32  ATIMER_ST      : 1;
  __REG32  RTC_ST         : 1;
  __REG32  BOD_ST         : 1;
  __REG32  WWDT_ST        : 1;
  __REG32  				        : 4;
  __REG32  CAN_ST         : 1;
  __REG32  TIM2_ST        : 1;
  __REG32  TIM6_ST        : 1;
  __REG32  QEI_ST         : 1;
  __REG32  TIM14_ST       : 1;
  __REG32                 : 2;
  __REG32  RESET_ST       : 1;
  __REG32                 :12;
} __er_status_bits;

/* Event enable register */
typedef struct {
  __REG32  WAKEUP0_EN     : 1;
  __REG32  WAKEUP1_EN     : 1;
  __REG32  WAKEUP2_EN     : 1;
  __REG32  WAKEUP3_EN     : 1;
  __REG32  ATIMER_EN      : 1;
  __REG32  RTC_EN         : 1;
  __REG32  BOD_EN         : 1;
  __REG32  WWDT_EN        : 1;
  __REG32  			          : 4;
  __REG32  CAN_EN         : 1;
  __REG32  TIM2_EN        : 1;
  __REG32  TIM6_EN        : 1;
  __REG32  QEI_EN         : 1;
  __REG32  TIM14_EN       : 1;
  __REG32                 : 2;
  __REG32  RESET_EN       : 1;
  __REG32                 :12;
} __er_enable_bits;

/* Interrupt clear status register */
typedef struct {
  __REG32  WAKEUP0_CLRST  : 1;
  __REG32  WAKEUP1_CLRST  : 1;
  __REG32  WAKEUP2_CLRST  : 1;
  __REG32  WAKEUP3_CLRST  : 1;
  __REG32  ATIMER_CLRST   : 1;
  __REG32  RTC_CLRST      : 1;
  __REG32  BOD_CLRST      : 1;
  __REG32  WWDT_CLRST     : 1;
  __REG32  				        : 4;
  __REG32  CAN_CLRST      : 1;
  __REG32  TIM2_CLRST     : 1;
  __REG32  TIM6_CLRST     : 1;
  __REG32  QEI_CLRST      : 1;
  __REG32  TIM14_CLRST    : 1;
  __REG32                 : 2;
  __REG32  RESET_CLRST    : 1;
  __REG32                 :12;
} __er_clr_stat_bits;

/* Interrupt set status register */
typedef struct {
  __REG32  WAKEUP0_SETST  : 1;
  __REG32  WAKEUP1_SETST  : 1;
  __REG32  WAKEUP2_SETST  : 1;
  __REG32  WAKEUP3_SETST  : 1;
  __REG32  ATIMER_SETST   : 1;
  __REG32  RTC_SETST      : 1;
  __REG32  BOD_SETST      : 1;
  __REG32  WWDT_SETST     : 1;
  __REG32  					      : 4;
  __REG32  CAN_SETST      : 1;
  __REG32  TIM2_SETST     : 1;
  __REG32  TIM6_SETST     : 1;
  __REG32  QEI_SETST      : 1;
  __REG32  TIM14_SETST    : 1;
  __REG32                 : 2;
  __REG32  RESET_SETST    : 1;
  __REG32                 :12;
} __er_set_stat_bits;

/* IRC trim register */
typedef struct {
  __REG32  TRM            :12;
  __REG32                 :20;
} __irctrm_bits;

/* CREG0 control register */
typedef struct {
  __REG32  EN1KHZ         : 1;
  __REG32  EN32KHZ        : 1;
  __REG32  RESET32KHZ     : 1;
  __REG32  _32KHZPD       : 1;
  __REG32                 : 4;
  __REG32  BODLVL1        : 2;
  __REG32  BODLVL2        : 2;
  __REG32                 :20;
} __creg0_bits;

/* Power mode control register */
typedef struct {
  __REG32  PMUCON         : 2;
  __REG32                 :30;
} __pmucon_bits;

/* CREG5 control register */
typedef struct {
  __REG32                 : 6;
  __REG32  M3TAPSEL       : 1;
  __REG32                 : 1;
  __REG32  OTPJTAG        : 1;
  __REG32                 :23;
} __creg5_bits;

/* DMA muxing register */
typedef struct {
  __REG32  DMAMUXCH0      : 2;
  __REG32  DMAMUXCH1      : 2;
  __REG32  DMAMUXCH2      : 2;
  __REG32  DMAMUXCH3      : 2;
  __REG32  DMAMUXCH4      : 2;
  __REG32  DMAMUXCH5      : 2;
  __REG32  DMAMUXCH6      : 2;
  __REG32  DMAMUXCH7      : 2;
  __REG32  DMAMUXCH8      : 2;
  __REG32  DMAMUXCH9      : 2;
  __REG32  DMAMUXCH10     : 2;
  __REG32  DMAMUXCH11     : 2;
  __REG32  DMAMUXCH12     : 2;
  __REG32  DMAMUXCH13     : 2;
  __REG32  DMAMUXCH14     : 2;
  __REG32  DMAMUXCH15     : 2;
} __dmamux_bits;

/* ETB SRAM control register */
typedef struct {
  __REG32  ETB            : 1;
  __REG32                 :31;  
} __etbcfg_bits;

/* CREG6 control register */
typedef struct {
  __REG32  				        : 3;
  __REG32                 : 1;
  __REG32  TIMIN7CTRL     : 2;
  __REG32  TIM1INCTRL     : 1;
  __REG32  TIM2INCTRL     : 1;
  __REG32  TIM3INCTRL     : 1;
  __REG32                 :23;
} __creg6_bits;

/* Hardware sleep event enable register */
typedef struct {
  __REG32  ENA_EVENT0     : 1;
  __REG32                 :31;
} __pd0_sleep0_hw_ena_bits;

/* Sleep power mode register */
typedef struct {
  __REG32  ENA_EVENT0     : 1;
  __REG32                 :31;
} __pd0_sleep0_mode_bits;

/* FREQ_MON register */
typedef struct {
  __REG32  RCNT           : 9;
  __REG32  FCNT           :14;
  __REG32  MEAS           : 1;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_freq_mon_bits;

/* XTAL_OSC_CTRL register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  BYPASS         : 1;
  __REG32  HF             : 1;
  __REG32                 :29;
} __cgu_xtal_osc_ctrl_bits;

/* PLL0_STAT register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32  FR             : 1;
  __REG32                 :30;
} __cgu_pll0_stat_bits;

/* PLL0 control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32  BYPASS         : 1;
  __REG32  DIRECTI        : 1;
  __REG32  DIRECTO        : 1;
  __REG32  CLKEN          : 1;
  __REG32                 : 1;
  __REG32  FRM            : 1;
  __REG32                 : 4;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_pll0_ctrl_bits;

/* PLL0_MDIV register */
typedef struct {
  __REG32  MDEC           :17;
  __REG32  SELP           : 5;
  __REG32  SELI           : 6;
  __REG32  SELR           : 4;
} __cgu_pll0_mdiv_bits;

/* PLL0 NP-divider register */
typedef struct {
  __REG32  PDEC           : 7;
  __REG32                 : 5;
  __REG32  NDEC           :10;
  __REG32                 :10;
} __cgu_pll0_np_div_bits;

/* PLL1_STAT register */
typedef struct {
  __REG32  LOCK           : 1;
  __REG32                 :31;
} __cgu_pll1_stat_bits;

/* PPLL1_CTRL register */
typedef struct {
  __REG32  PD             : 1;
  __REG32  BYPASS         : 1;
  __REG32                 : 4;
  __REG32  FBSEL          : 1;
  __REG32  DIRECT         : 1;
  __REG32  PSEL           : 2;
  __REG32                 : 1;
  __REG32  AUTOBLOCK      : 1;
  __REG32  NSEL           : 2;
  __REG32                 : 2;
  __REG32  MSEL           : 8;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_pll1_ctrl_bits;

/* IDIVA control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 2;
  __REG32                 : 7;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_idiva_ctrl_bits;

/* IDIVB/C/D control registers */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 4;
  __REG32                 : 5;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_idivx_ctrl_bits;

/* IDIVE control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 : 1;
  __REG32  IDIV           : 8;
  __REG32                 : 1;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_idive_ctrl_bits;

/* Output stage X control register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 :10;
  __REG32  AUTOBLOCK      : 1;
  __REG32                 :12;
  __REG32  CLK_SEL        : 4;
  __REG32                 : 4;
} __cgu_outclk_ctrl_bits;

/* CCU1/2 power mode register */
typedef struct {
  __REG32  PD             : 1;
  __REG32                 :31;
} __ccu_pm_bits;

/* CCU1 base clock status register */
typedef struct {
  __REG32  BASE_APB3_CLK_IND  : 1;
  __REG32  BASE_APB1_CLK_IND  : 1;
  __REG32  BASE_SPIFI_CLK_IND : 1;
  __REG32  BASE_M3_CLK_IND    : 1;
  __REG32                     : 3;
  __REG32                     : 2;
  __REG32                     :23;
} __ccu1_base_stat_bits;

/* CCU2 base clock status register */
typedef struct {
  __REG32                     : 1;
  __REG32  BASE_UART3_CLK     : 1;
  __REG32  BASE_UART2_CLK     : 1;
  __REG32  BASE_UART1_CLK     : 1;
  __REG32  BASE_UART0_CLK     : 1;
  __REG32  BASE_SSP1_CLK      : 1;
  __REG32  BASE_SSP0_CLK      : 1;
  __REG32                     :25;
} __ccu2_base_stat_bits;

/* CCU1/2 branch clock configuration register */
/* CCU1/2 branch clock status register */
typedef struct {
  __REG32  RUN                : 1;
  __REG32  AUTO               : 1;
  __REG32  WAKEUP             : 1;
  __REG32                     :29;
} __ccu_clk_cfg_bits;

/* Reset control register 0 */
/* Reset active status register 0 */
typedef struct {
  __REG32  CORE_RST           : 1;
  __REG32  PERIPH_RST         : 1;
  __REG32  MASTER_RST         : 1;
  __REG32                     : 1;
  __REG32  WWDT_RST           : 1;
  __REG32  CREG_RST           : 1;
  __REG32                     : 2;
  __REG32  BUS_RST            : 1;
  __REG32  SCU_RST            : 1;
  __REG32  PINMUX_RST         : 1;
  __REG32                     : 2;
  __REG32  M3_RST             : 1;
  __REG32                     : 2;
  __REG32                     : 3;
  __REG32  DMA_RST            : 1;
  __REG32  SDIO_RST           : 1;
  __REG32  EMC_RST            : 1;
  __REG32                     : 1;
  __REG32  AES_RST            : 1;
  __REG32                     : 4;
  __REG32  GPIO_RST           : 1;
  __REG32                     : 3;
} __rgu_reset_ctrl0_bits;

/* Reset control register 1 */
/* Reset active status register 1 */
typedef struct {
  __REG32  TIMER0_RST         : 1;
  __REG32  TIMER1_RST         : 1;
  __REG32  TIMER2_RST         : 1;
  __REG32  TIMER3_RST         : 1;
  __REG32  RITIMER_RST        : 1;
  __REG32  SCT_RST            : 1;
  __REG32  MOTOCONPWM_RST     : 1;
  __REG32  QEI_RST            : 1;
  __REG32  ADC0_RST           : 1;
  __REG32  ADC1_RST           : 1;
  __REG32  DAC_RST            : 1;
  __REG32                     : 1;
  __REG32  UART0_RST          : 1;
  __REG32  UART1_RST          : 1;
  __REG32  UART2_RST          : 1;
  __REG32  UART3_RST          : 1;
  __REG32  I2C0_RST           : 1;
  __REG32  I2C1_RST           : 1;
  __REG32  SSP0_RST           : 1;
  __REG32  SSP1_RST           : 1;
  __REG32  I2S_RST            : 1;
  __REG32  SPIFI_RST          : 1;
  __REG32  CAN1_RST           : 1;
  __REG32  CAN0_RST           : 1;
  __REG32                     : 8;
} __rgu_reset_ctrl1_bits;

/* Reset status register 0 */
typedef struct {
  __REG32  CORE_RST           : 2;
  __REG32  PERIPH_RST         : 2;
  __REG32  MASTER_RST         : 2;
  __REG32                     : 2;
  __REG32  WWDT_RST           : 2;
  __REG32  CREG_RST           : 2;
  __REG32                     : 4;
  __REG32  BUS_RST            : 2;
  __REG32  SCU_RST            : 2;
  __REG32                     : 6;
  __REG32  M3_RST             : 2;
  __REG32                     : 4;
} __rgu_reset_status0_bits;

/* Reset status register 1 */
typedef struct {
  __REG32                     : 6;
  __REG32  DMA_RST            : 2;
  __REG32  SDIO_RST           : 2;
  __REG32  EMC_RST            : 2;
  __REG32                     : 2;
  __REG32  AES_RST            : 2;
  __REG32                     : 8;
  __REG32  GPIO_RST           : 2;
  __REG32                     : 6;
} __rgu_reset_status1_bits;

/* Reset status register 2 */
typedef struct {
  __REG32  TIMER0_RST         : 2;
  __REG32  TIMER1_RST         : 2;
  __REG32  TIMER2_RST         : 2;
  __REG32  TIMER3_RST         : 2;
  __REG32  RITIMER_RST        : 2;
  __REG32  SCT_RST            : 2;
  __REG32  MOTOCONPWM_RST     : 2;
  __REG32  QEI_RST            : 2;
  __REG32  ADC0_RST           : 2;
  __REG32  ADC1_RST           : 2;
  __REG32  DAC_RST            : 2;
  __REG32                     : 2;
  __REG32  UART0_RST          : 2;
  __REG32  UART1_RST          : 2;
  __REG32  UART2_RST          : 2;
  __REG32  UART3_RST          : 2;
} __rgu_reset_status2_bits;

/* Reset status register 3 */
typedef struct {
  __REG32  I2C0_RST           : 2;
  __REG32  I2C1_RST           : 2;
  __REG32  SSP0_RST           : 2;
  __REG32  SSP1_RST           : 2;
  __REG32  I2S_RST            : 2;
  __REG32  SPIFI_RST          : 2;
  __REG32  CAN1_RST           : 2;
  __REG32  CAN0_RST           : 2;
  __REG32                     :16;
} __rgu_reset_status3_bits;

/* Reset external status register 0 */
typedef struct {
  __REG32  EXT_RESET          : 1;
  __REG32                     : 3;
  __REG32  BOD_RESET          : 1;
  __REG32  WWDT_RESET         : 1;
  __REG32                     :26;
} __rgu_reset_ext_stat0_bits;

/* Reset external status register 1 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat1_bits;

/* Reset external status register 2 */
typedef struct {
  __REG32                     : 2;
  __REG32  PERIPHERAL_RESET   : 1;
  __REG32                     :29;
} __rgu_reset_ext_stat2_bits;

/* Reset external status register 4 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat4_bits;

/* Reset external status register 5 */
typedef struct {
  __REG32                     : 1;
  __REG32  CORE_RESET         : 1;
  __REG32                     :30;
} __rgu_reset_ext_stat5_bits;

/* Reset external status registers for PERIPHERAL_RESET */
typedef struct {
  __REG32                     : 2;
  __REG32  PERIPHERAL_RESET   : 1;
  __REG32                     :29;
} __rgu_peripheral_reset_bits;

/* Reset external status registers for MASTER_RESET */
typedef struct {
  __REG32                     : 3;
  __REG32  MASTER_RESET       : 1;
  __REG32                     :28;
} __rgu_master_reset_bits;

/* Pin configuration registers for normal drive and high speed pins */
typedef struct {
  __REG32  MODE               : 3;
  __REG32  EPD                : 1;
  __REG32  EPUN               : 1;
  __REG32  EHS                : 1;
  __REG32  EZI                : 1;
  __REG32  ZIF                : 1;
  __REG32                     :24;
} __sfspx_normdrv_hispd_bits;

/* Pin configuration registers for high drive pins */
typedef struct {
  __REG32  MODE               : 3;
  __REG32  EPD                : 1;
  __REG32  EPUN               : 1;
  __REG32                     : 1;
  __REG32  EZI                : 1;
  __REG32  ZIF                : 1;
  __REG32  EHD                : 2;
  __REG32                     :22;
} __sfspx_hidrv_bits;

/* Pin configuration register for open-drain I2C-bus pins */
typedef struct {
  __REG32  SDA_EHS            : 1;
  __REG32  SCL_EHS            : 1;
  __REG32  SCL_ECS            : 1;
  __REG32                     :29;
} __sfsi2c0_bits;

/* EMC clock delay register */
typedef struct {
  __REG32  CLK0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CLK1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CLK2_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CLK3_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CKE0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CKE1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CKE2_DELAY         : 3;
  __REG32                     : 1;
  __REG32  CKE3_DELAY         : 3;
  __REG32                     : 1;
} __emcclkdelay_bits;

/* EMC control delay register*/
typedef struct {
  __REG32  RAS_DELAY          : 3;
  __REG32                     : 1;
  __REG32  CAS_DELAY          : 3;
  __REG32                     : 1;
  __REG32  OE_DELAY           : 3;
  __REG32                     : 1;
  __REG32  WE_DELAY           : 3;
  __REG32                     : 1;
  __REG32  BLS0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  BLS1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  BLS2_DELAY         : 3;
  __REG32                     : 1;
  __REG32  BLS3_DELAY         : 3;
  __REG32                     : 1;
} __emcctrldelay_bits;

/* EMC chip select delay register */
typedef struct {
  __REG32  DYCS0_DELAY        : 3;
  __REG32                     : 1;
  __REG32  DYCS1_DELAY        : 3;
  __REG32                     : 1;
  __REG32  DYCS2_DELAY        : 3;
  __REG32                     : 1;
  __REG32  DYCS3_DELAY        : 3;
  __REG32                     : 1;
  __REG32  CS0_DELAY          : 3;
  __REG32                     : 1;
  __REG32  CS1_DELAY          : 3;
  __REG32                     : 1;
  __REG32  CS2_DELAY          : 3;
  __REG32                     : 1;
  __REG32  CS3_DELAY          : 3;
  __REG32                     : 1;
} __emccsdelay_bits;

/* EMC data out delay register */
typedef struct {
  __REG32  DQM0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DQM1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DQM2_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DQM3_DELAY         : 3;
  __REG32                     : 1;
  __REG32  D0_DELAY           : 3;
  __REG32                     : 1;
  __REG32  D1_DELAY           : 3;
  __REG32                     : 1;
  __REG32  D2_DELAY           : 3;
  __REG32                     : 1;
  __REG32  D3_DELAY           : 3;
  __REG32                     : 1;
} __emcdoutdelay_bits;

/* EMC DQM delay register */
typedef struct {
  __REG32  FBCLK0_DELAY       : 3;
  __REG32                     : 1;
  __REG32  FBCLK1_DELAY       : 3;
  __REG32                     : 1;
  __REG32  FBCLK2_DELAY       : 3;
  __REG32                     : 1;
  __REG32  FBCLK3_DELAY       : 3;
  __REG32                     : 1;
  __REG32  CCLK_DELAY         : 3;
  __REG32                     :13;
} __emcfbclkdelay_bits;

/* EMC  delay register 0 */
typedef struct {
  __REG32  ADDR0_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR1_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR2_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR3_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR4_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR5_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR6_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR7_DELAY        : 3;
  __REG32                     : 1;
} __emcaddrdelay0_bits;

/* EMC  delay register 1 */
typedef struct {
  __REG32  ADDR8_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR9_DELAY        : 3;
  __REG32                     : 1;
  __REG32  ADDR10_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR11_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR12_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR13_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR14_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR15_DELAY       : 3;
  __REG32                     : 1;
} __emcaddrdelay1_bits;

/* EMC  delay register 2 */
typedef struct {
  __REG32  ADDR16_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR17_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR18_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR19_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR20_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR21_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR22_DELAY       : 3;
  __REG32                     : 1;
  __REG32  ADDR23_DELAY       : 3;
  __REG32                     : 1;
} __emcaddrdelay2_bits;

/* EMC data in delay register */
typedef struct {
  __REG32  DIN0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DIN1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DIN2_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DIN3_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DEN0_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DEN1_DELAY         : 3;
  __REG32                     : 1;
  __REG32  DEN2_DELAY         : 3;
  __REG32                     : 5;
} __emcdindelay_bits;

/* FGPIO 0 Registers*/
typedef union{
  /*FIO0DIR*/
  /*FIO0MASK*/
  /*FIO0PIN*/
  /*FIO0SET*/
  /*FIO0CLR*/
  struct {
    __REG32 P0_0   : 1;
    __REG32 P0_1   : 1;
    __REG32 P0_2   : 1;
    __REG32 P0_3   : 1;
    __REG32 P0_4   : 1;
    __REG32 P0_5   : 1;
    __REG32 P0_6   : 1;
    __REG32 P0_7   : 1;
    __REG32 P0_8   : 1;
    __REG32 P0_9   : 1;
    __REG32 P0_10  : 1;
    __REG32 P0_11  : 1;
    __REG32 P0_12  : 1;
    __REG32 P0_13  : 1;
    __REG32 P0_14  : 1;
    __REG32 P0_15  : 1;
    __REG32        :16;
  };

  struct
  {
    union
    {
    /*FIO0DIR0*/
    /*FIO0MASK0*/
    /*FIO0PIN0*/
    /*FIO0SET0*/
    /*FIO0CLR0*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
    /*FIO0DIR1*/
    /*FIO0MASK1*/
    /*FIO0PIN1*/
    /*FIO0SET1*/
    /*FIO0CLR1*/
      struct{
        __REG8  P0_0   : 1;
        __REG8  P0_1   : 1;
        __REG8  P0_2   : 1;
        __REG8  P0_3   : 1;
        __REG8  P0_4   : 1;
        __REG8  P0_5   : 1;
        __REG8  P0_6   : 1;
        __REG8  P0_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8             : 8;
    __REG8             : 8;
  };

  struct
  {
    union
    {
      /*FIO0DIRL*/
      /*FIO0MASKL*/
      /*FIO0PINL*/
      /*FIO0SETL*/
      /*FIO0CLRL*/
      struct{
        __REG16 P0_0   : 1;
        __REG16 P0_1   : 1;
        __REG16 P0_2   : 1;
        __REG16 P0_3   : 1;
        __REG16 P0_4   : 1;
        __REG16 P0_5   : 1;
        __REG16 P0_6   : 1;
        __REG16 P0_7   : 1;
        __REG16 P0_8   : 1;
        __REG16 P0_9   : 1;
        __REG16 P0_10  : 1;
        __REG16 P0_11  : 1;
        __REG16 P0_12  : 1;
        __REG16 P0_13  : 1;
        __REG16 P0_14  : 1;
        __REG16 P0_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16            :16;
  };
} __fgpio0_bits;

/* FGPIO 1 Registers*/
typedef union{
  /*FIO1DIR*/
  /*FIO1MASK*/
  /*FIO1PIN*/
  /*FIO1SET*/
  /*FIO1CLR*/
  struct {
    __REG32 P1_0   : 1;
    __REG32 P1_1   : 1;
    __REG32 P1_2   : 1;
    __REG32 P1_3   : 1;
    __REG32 P1_4   : 1;
    __REG32 P1_5   : 1;
    __REG32 P1_6   : 1;
    __REG32 P1_7   : 1;
    __REG32 P1_8   : 1;
    __REG32 P1_9   : 1;
    __REG32 P1_10  : 1;
    __REG32 P1_11  : 1;
    __REG32 P1_12  : 1;
    __REG32 P1_13  : 1;
    __REG32 P1_14  : 1;
    __REG32 P1_15  : 1;
    __REG32        :16;
  };

  struct
  {
    union
    {
      /*FIO1DIR0*/
      /*FIO1MASK0*/
      /*FIO1PIN0*/
      /*FIO1SET0*/
      /*FIO1CLR0*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO1DIR1*/
      /*FIO1MASK1*/
      /*FIO1PIN1*/
      /*FIO1SET1*/
      /*FIO1CLR1*/
      struct{
        __REG8  P1_0   : 1;
        __REG8  P1_1   : 1;
        __REG8  P1_2   : 1;
        __REG8  P1_3   : 1;
        __REG8  P1_4   : 1;
        __REG8  P1_5   : 1;
        __REG8  P1_6   : 1;
        __REG8  P1_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8             : 8;
    __REG8             : 8;
  };

  struct
  {
    union
    {
      /*FIO1DIRL*/
      /*FIO1MASKL*/
      /*FIO1PINL*/
      /*FIO1SETL*/
      /*FIO1CLRL*/
      struct{
        __REG16 P1_0   : 1;
        __REG16 P1_1   : 1;
        __REG16 P1_2   : 1;
        __REG16 P1_3   : 1;
        __REG16 P1_4   : 1;
        __REG16 P1_5   : 1;
        __REG16 P1_6   : 1;
        __REG16 P1_7   : 1;
        __REG16 P1_8   : 1;
        __REG16 P1_9   : 1;
        __REG16 P1_10  : 1;
        __REG16 P1_11  : 1;
        __REG16 P1_12  : 1;
        __REG16 P1_13  : 1;
        __REG16 P1_14  : 1;
        __REG16 P1_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16            :16;
  };
} __fgpio1_bits;

/* FGPIO 2 Registers*/
typedef union{
  /*FIO2DIR*/
  /*FIO2MASK*/
  /*FIO2PIN*/
  /*FIO2SET*/
  /*FIO2CLR*/
  struct {
    __REG32 P2_0   : 1;
    __REG32 P2_1   : 1;
    __REG32 P2_2   : 1;
    __REG32 P2_3   : 1;
    __REG32 P2_4   : 1;
    __REG32 P2_5   : 1;
    __REG32 P2_6   : 1;
    __REG32 P2_7   : 1;
    __REG32 P2_8   : 1;
    __REG32 P2_9   : 1;
    __REG32 P2_10  : 1;
    __REG32 P2_11  : 1;
    __REG32 P2_12  : 1;
    __REG32 P2_13  : 1;
    __REG32 P2_14  : 1;
    __REG32 P2_15  : 1;
    __REG32        :16;
  };

  struct
  {
    union
    {
      /*FIO2DIR0*/
      /*FIO2MASK0*/
      /*FIO2PIN0*/
      /*FIO2SET0*/
      /*FIO2CLR0*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO2DIR1*/
      /*FIO2MASK1*/
      /*FIO2PIN1*/
      /*FIO2SET1*/
      /*FIO2CLR1*/
      struct{
        __REG8  P2_0   : 1;
        __REG8  P2_1   : 1;
        __REG8  P2_2   : 1;
        __REG8  P2_3   : 1;
        __REG8  P2_4   : 1;
        __REG8  P2_5   : 1;
        __REG8  P2_6   : 1;
        __REG8  P2_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8             : 8;
    __REG8             : 8;
  };

  struct
  {
    union
    {
      /*FIO2DIRL*/
      /*FIO2MASKL*/
      /*FIO2PINL*/
      /*FIO2SETL*/
      /*FIO2CLRL*/
      struct{
        __REG16 P2_0   : 1;
        __REG16 P2_1   : 1;
        __REG16 P2_2   : 1;
        __REG16 P2_3   : 1;
        __REG16 P2_4   : 1;
        __REG16 P2_5   : 1;
        __REG16 P2_6   : 1;
        __REG16 P2_7   : 1;
        __REG16 P2_8   : 1;
        __REG16 P2_9   : 1;
        __REG16 P2_10  : 1;
        __REG16 P2_11  : 1;
        __REG16 P2_12  : 1;
        __REG16 P2_13  : 1;
        __REG16 P2_14  : 1;
        __REG16 P2_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16            :16;
  };
} __fgpio2_bits;

/* FGPIO 3 Registers*/
typedef union{
  /*FIO3DIR*/
  /*FIO3MASK*/
  /*FIO3PIN*/
  /*FIO3SET*/
  /*FIO3CLR*/
  struct {
    __REG32 P3_0   : 1;
    __REG32 P3_1   : 1;
    __REG32 P3_2   : 1;
    __REG32 P3_3   : 1;
    __REG32 P3_4   : 1;
    __REG32 P3_5   : 1;
    __REG32 P3_6   : 1;
    __REG32 P3_7   : 1;
    __REG32 P3_8   : 1;
    __REG32 P3_9   : 1;
    __REG32 P3_10  : 1;
    __REG32 P3_11  : 1;
    __REG32 P3_12  : 1;
    __REG32 P3_13  : 1;
    __REG32 P3_14  : 1;
    __REG32 P3_15  : 1;
    __REG32        :16;
  };

  struct
  {
    union
    {
      /*FIO3DIR0*/
      /*FIO3MASK0*/
      /*FIO3PIN0*/
      /*FIO3SET0*/
      /*FIO3CLR0*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO3DIR1*/
      /*FIO3MASK1*/
      /*FIO3PIN1*/
      /*FIO3SET1*/
      /*FIO3CLR1*/
      struct{
        __REG8  P3_0   : 1;
        __REG8  P3_1   : 1;
        __REG8  P3_2   : 1;
        __REG8  P3_3   : 1;
        __REG8  P3_4   : 1;
        __REG8  P3_5   : 1;
        __REG8  P3_6   : 1;
        __REG8  P3_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8             : 8;
    __REG8             : 8;
  };

  struct
  {
    union
    {
      /*FIO3DIRL*/
      /*FIO3MASKL*/
      /*FIO3PINL*/
      /*FIO3SETL*/
      /*FIO3CLRL*/
      struct{
        __REG16 P3_0   : 1;
        __REG16 P3_1   : 1;
        __REG16 P3_2   : 1;
        __REG16 P3_3   : 1;
        __REG16 P3_4   : 1;
        __REG16 P3_5   : 1;
        __REG16 P3_6   : 1;
        __REG16 P3_7   : 1;
        __REG16 P3_8   : 1;
        __REG16 P3_9   : 1;
        __REG16 P3_10  : 1;
        __REG16 P3_11  : 1;
        __REG16 P3_12  : 1;
        __REG16 P3_13  : 1;
        __REG16 P3_14  : 1;
        __REG16 P3_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16            :16;
  };
} __fgpio3_bits;

/* FGPIO 4 Registers*/
typedef union{
  /*FIO4DIR*/
  /*FIO4MASK*/
  /*FIO4PIN*/
  /*FIO4SET*/
  /*FIO4CLR*/
  struct {
    __REG32 P4_0   : 1;
    __REG32 P4_1   : 1;
    __REG32 P4_2   : 1;
    __REG32 P4_3   : 1;
    __REG32 P4_4   : 1;
    __REG32 P4_5   : 1;
    __REG32 P4_6   : 1;
    __REG32 P4_7   : 1;
    __REG32 P4_8   : 1;
    __REG32 P4_9   : 1;
    __REG32 P4_10  : 1;
    __REG32 P4_11  : 1;
    __REG32 P4_12  : 1;
    __REG32 P4_13  : 1;
    __REG32 P4_14  : 1;
    __REG32 P4_15  : 1;
    __REG32        :16;
  };

  struct
  {
    union
    {
      /*FIO4DIR0*/
      /*FIO4MASK0*/
      /*FIO4PIN0*/
      /*FIO4SET0*/
      /*FIO4CLR0*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*FIO4DIR1*/
      /*FIO4MASK1*/
      /*FIO4PIN1*/
      /*FIO4SET1*/
      /*FIO4CLR1*/
      struct{
        __REG8  P4_0   : 1;
        __REG8  P4_1   : 1;
        __REG8  P4_2   : 1;
        __REG8  P4_3   : 1;
        __REG8  P4_4   : 1;
        __REG8  P4_5   : 1;
        __REG8  P4_6   : 1;
        __REG8  P4_7   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    __REG8             : 8;
    __REG8             : 8;
  };

  struct
  {
    union
    {
      /*FIO4DIRL*/
      /*FIO4MASKL*/
      /*FIO4PINL*/
      /*FIO4SETL*/
      /*FIO4CLRL*/
      struct{
        __REG16 P4_0   : 1;
        __REG16 P4_1   : 1;
        __REG16 P4_2   : 1;
        __REG16 P4_3   : 1;
        __REG16 P4_4   : 1;
        __REG16 P4_5   : 1;
        __REG16 P4_6   : 1;
        __REG16 P4_7   : 1;
        __REG16 P4_8   : 1;
        __REG16 P4_9   : 1;
        __REG16 P4_10  : 1;
        __REG16 P4_11  : 1;
        __REG16 P4_12  : 1;
        __REG16 P4_13  : 1;
        __REG16 P4_14  : 1;
        __REG16 P4_15  : 1;
      } __shortl_bit;
      __REG16 __shortl;
    };
    __REG16            :16;
  };
} __fgpio4_bits;

/* DMA Interrupt Status Register */
typedef struct{
__REG32 INTSTATUS0  : 1;
__REG32 INTSTATUS1  : 1;
__REG32 INTSTATUS2  : 1;
__REG32 INTSTATUS3  : 1;
__REG32 INTSTATUS4  : 1;
__REG32 INTSTATUS5  : 1;
__REG32 INTSTATUS6  : 1;
__REG32 INTSTATUS7  : 1;
__REG32             :24;
} __dmacintstatus_bits;

/* DMA Interrupt Terminal Count Request Status Register */
typedef struct{
__REG32 INTTCSTATUS0  : 1;
__REG32 INTTCSTATUS1  : 1;
__REG32 INTTCSTATUS2  : 1;
__REG32 INTTCSTATUS3  : 1;
__REG32 INTTCSTATUS4  : 1;
__REG32 INTTCSTATUS5  : 1;
__REG32 INTTCSTATUS6  : 1;
__REG32 INTTCSTATUS7  : 1;
__REG32               :24;
} __dmacinttcstatus_bits;

/* DMA Interrupt Terminal Count Request Clear Register */
typedef struct{
__REG32 INTTCCLEAR0   : 1;
__REG32 INTTCCLEAR1   : 1;
__REG32 INTTCCLEAR2   : 1;
__REG32 INTTCCLEAR3   : 1;
__REG32 INTTCCLEAR4   : 1;
__REG32 INTTCCLEAR5   : 1;
__REG32 INTTCCLEAR6   : 1;
__REG32 INTTCCLEAR7   : 1;
__REG32               :24;
} __dmacinttcclear_bits;

/* DMA Interrupt Error Status Register */
typedef struct{
__REG32 INTERRSTAT0 : 1;
__REG32 INTERRSTAT1 : 1;
__REG32 INTERRSTAT2 : 1;
__REG32 INTERRSTAT3 : 1;
__REG32 INTERRSTAT4 : 1;
__REG32 INTERRSTAT5 : 1;
__REG32 INTERRSTAT6 : 1;
__REG32 INTERRSTAT7 : 1;
__REG32             :24;
} __dmacinterrstat_bits;

/* DMA Interrupt Error Clear Register */
typedef struct{
__REG32 INTERRCLR0      : 1;
__REG32 INTERRCLR1      : 1;
__REG32 INTERRCLR2      : 1;
__REG32 INTERRCLR3      : 1;
__REG32 INTERRCLR4      : 1;
__REG32 INTERRCLR5      : 1;
__REG32 INTERRCLR6      : 1;
__REG32 INTERRCLR7      : 1;
__REG32                 :24;
} __dmacinterrclr_bits;

/* DMA Raw Interrupt Terminal Count Status Register */
typedef struct{
__REG32 RAWINTTCSTATUS0 : 1;
__REG32 RAWINTTCSTATUS1 : 1;
__REG32 RAWINTTCSTATUS2 : 1;
__REG32 RAWINTTCSTATUS3 : 1;
__REG32 RAWINTTCSTATUS4 : 1;
__REG32 RAWINTTCSTATUS5 : 1;
__REG32 RAWINTTCSTATUS6 : 1;
__REG32 RAWINTTCSTATUS7 : 1;
__REG32                 :24;
} __dmacrawinttcstatus_bits;

/* DMA Raw Error Interrupt Status Register */
typedef struct{
__REG32 RAWINTERRSTAT0  : 1;
__REG32 RAWINTERRSTAT1  : 1;
__REG32 RAWINTERRSTAT2  : 1;
__REG32 RAWINTERRSTAT3  : 1;
__REG32 RAWINTERRSTAT4  : 1;
__REG32 RAWINTERRSTAT5  : 1;
__REG32 RAWINTERRSTAT6  : 1;
__REG32 RAWINTERRSTAT7  : 1;
__REG32                 :24;
} __dmacrawinterrorstatus_bits;

/* DMA Enabled Channel Register */
typedef struct{
__REG32 ENABLEDCHANNELS0  : 1;
__REG32 ENABLEDCHANNELS1  : 1;
__REG32 ENABLEDCHANNELS2  : 1;
__REG32 ENABLEDCHANNELS3  : 1;
__REG32 ENABLEDCHANNELS4  : 1;
__REG32 ENABLEDCHANNELS5  : 1;
__REG32 ENABLEDCHANNELS6  : 1;
__REG32 ENABLEDCHANNELS7  : 1;
__REG32                   :24;
} __dmacenbldchns_bits;

/* DMA Software Burst Request Register */
typedef struct{
__REG32 SOFTBREQ0         : 1;
__REG32 SOFTBREQ1         : 1;
__REG32 SOFTBREQ2         : 1;
__REG32 SOFTBREQ3         : 1;
__REG32 SOFTBREQ4         : 1;
__REG32 SOFTBREQ5         : 1;
__REG32 SOFTBREQ6         : 1;
__REG32 SOFTBREQ7         : 1;
__REG32 SOFTBREQ8         : 1;
__REG32 SOFTBREQ9         : 1;
__REG32 SOFTBREQ10        : 1;
__REG32 SOFTBREQ11        : 1;
__REG32 SOFTBREQ12        : 1;
__REG32 SOFTBREQ13        : 1;
__REG32 SOFTBREQ14        : 1;
__REG32 SOFTBREQ15        : 1;
__REG32                   :16;
} __dmacsoftbreq_bits;

/* DMA Software Single Request Register */
typedef struct{
__REG32 SOFTSREQ0         : 1;
__REG32 SOFTSREQ1         : 1;
__REG32 SOFTSREQ2         : 1;
__REG32 SOFTSREQ3         : 1;
__REG32 SOFTSREQ4         : 1;
__REG32 SOFTSREQ5         : 1;
__REG32 SOFTSREQ6         : 1;
__REG32 SOFTSREQ7         : 1;
__REG32 SOFTSREQ8         : 1;
__REG32 SOFTSREQ9         : 1;
__REG32 SOFTSREQ10        : 1;
__REG32 SOFTSREQ11        : 1;
__REG32 SOFTSREQ12        : 1;
__REG32 SOFTSREQ13        : 1;
__REG32 SOFTSREQ14        : 1;
__REG32 SOFTSREQ15        : 1;
__REG32                   :16;
} __dmacsoftsreq_bits;

/* DMA Software Last Burst Request Register */
typedef struct{
__REG32 SOFTLBREQ0         : 1;
__REG32 SOFTLBREQ1         : 1;
__REG32 SOFTLBREQ2         : 1;
__REG32 SOFTLBREQ3         : 1;
__REG32 SOFTLBREQ4         : 1;
__REG32 SOFTLBREQ5         : 1;
__REG32 SOFTLBREQ6         : 1;
__REG32 SOFTLBREQ7         : 1;
__REG32 SOFTLBREQ8         : 1;
__REG32 SOFTLBREQ9         : 1;
__REG32 SOFTLBREQ10        : 1;
__REG32 SOFTLBREQ11        : 1;
__REG32 SOFTLBREQ12        : 1;
__REG32 SOFTLBREQ13        : 1;
__REG32 SOFTLBREQ14        : 1;
__REG32 SOFTLBREQ15        : 1;
__REG32                    :16;
} __dmacsoftlbreq_bits;

/* DMA Software Last Single Request Register */
typedef struct{
__REG32 SOFTLSREQ0          : 1;
__REG32 SOFTLSREQ1          : 1;
__REG32 SOFTLSREQ2          : 1;
__REG32 SOFTLSREQ3          : 1;
__REG32 SOFTLSREQ4          : 1;
__REG32 SOFTLSREQ5          : 1;
__REG32 SOFTLSREQ6          : 1;
__REG32 SOFTLSREQ7          : 1;
__REG32 SOFTLSREQ8          : 1;
__REG32 SOFTLSREQ9          : 1;
__REG32 SOFTLSREQ10         : 1;
__REG32 SOFTLSREQ11         : 1;
__REG32 SOFTLSREQ12         : 1;
__REG32 SOFTLSREQ13         : 1;
__REG32 SOFTLSREQ14         : 1;
__REG32 SOFTLSREQ15         : 1;
__REG32                     :16;
} __dmacsoftlsreq_bits;

/* DMA Synchronization Register */
typedef struct{
__REG32 DMACSYNC0   : 1;
__REG32 DMACSYNC1   : 1;
__REG32 DMACSYNC2   : 1;
__REG32 DMACSYNC3   : 1;
__REG32 DMACSYNC4   : 1;
__REG32 DMACSYNC5   : 1;
__REG32 DMACSYNC6   : 1;
__REG32 DMACSYNC7   : 1;
__REG32 DMACSYNC8   : 1;
__REG32 DMACSYNC9   : 1;
__REG32 DMACSYNC10  : 1;
__REG32 DMACSYNC11  : 1;
__REG32 DMACSYNC12  : 1;
__REG32 DMACSYNC13  : 1;
__REG32 DMACSYNC14  : 1;
__REG32 DMACSYNC15  : 1;
__REG32             :16;
} __dmacsync_bits;

/* DMA Configuration Register */
typedef struct{
__REG32 E   : 1;
__REG32 M0  : 1;
__REG32 M1  : 1;
__REG32     :29;
} __dmacconfig_bits;

/* DMA Channel Linked List Item registers */
typedef struct{
__REG32 LM  : 1;
__REG32     : 1;
__REG32 LLI :30;
} __dma_lli_bits;

/* DMA Channel Control Registers */
typedef struct{
__REG32 TRANSFERSIZE  :12;
__REG32 SBSIZE        : 3;
__REG32 DBSIZE        : 3;
__REG32 SWIDTH        : 3;
__REG32 DWIDTH        : 3;
__REG32 S             : 1;
__REG32 D             : 1;
__REG32 SI            : 1;
__REG32 DI            : 1;
__REG32 PROT1         : 1;
__REG32 PROT2         : 1;
__REG32 PROT3         : 1;
__REG32 I             : 1;
} __dma_ctrl_bits;

/* DMA Channel Configuration Registers */
typedef struct{
__REG32 E               : 1;
__REG32 SRCPERIPHERAL   : 5;
__REG32 DESTPERIPHERAL  : 5;
__REG32 FLOWCNTRL       : 3;
__REG32 IE              : 1;
__REG32 ITC             : 1;
__REG32 L               : 1;
__REG32 A               : 1;
__REG32 H               : 1;
__REG32                 :13;
} __dma_cfg_bits;

/* SPIFI control register */
typedef struct{
__REG32 AMSB            : 5;
__REG32 TO              :16;
__REG32 CLRID           : 1;
__REG32 INTEN           : 1;
__REG32 MODE3           : 1;
__REG32 CSHI            : 3;
__REG32 FSCKI           : 1;
__REG32                 : 4;
} __spifictrl_bits;

/* SPIFI command register */
typedef struct{
__REG32 DATALEN         :14;
__REG32 POLLRS          : 1;
__REG32 DOUT            : 1;
__REG32 INTLEN          : 3;
__REG32 P_S             : 2;
__REG32 FRAMEFORM       : 3;
__REG32 OPCODE          : 8;
} __spificmd_bits;

/* SPIFI memory command register */
typedef struct{
__REG32 DATALEN         :14;
__REG32 POLLRS          : 1;
__REG32 DOUT            : 1;
__REG32                 : 5;
__REG32 FRAMEFORM       : 3;
__REG32                 : 8;
} __spifimemcmd_bits;

/* SPIFI status register */
typedef struct{
__REG32 MCINIT          : 1;
__REG32 CMD             : 1;
__REG32 MCMD            : 1;
__REG32 CMDI            : 1;
__REG32                 : 4;
__REG32 FIFOBYTES       : 5;
__REG32                 :19;
} __spifistat_bits;

/* SDIO Block register */
typedef struct{
__REG32 TXFBLOCKSIZE    :12;
__REG32 BUFFSIZE        : 3;
__REG32                 : 1;
__REG32 BLCKCNT         :16;
} __sdio_blk_bits;

/* SDIO Transfer mode and command register */
typedef struct{
__REG32 DMAEN           : 1;
__REG32 BLCKCNTEN       : 1;
__REG32 AUTOCMD12EN     : 1;
__REG32                 : 1;
__REG32 TXFDIR          : 1;
__REG32 BLKSEL          : 1;
__REG32 CMDCOMPATA      : 1;
__REG32 SPIMODE         : 1;
__REG32 BOOTEN          : 1;
__REG32                 : 7;
__REG32 RESPTYPE        : 2;
__REG32                 : 1;
__REG32 CMDCRCEN        : 1;
__REG32 CMDINDXEN       : 1;
__REG32 DATAPRE         : 1;
__REG32 CMDTYPE         : 2;
__REG32 CMDINDX         : 6;
__REG32                 : 2;
} __sdio_cmd_txfmode_bits;

/* SDIO Present state register */
typedef struct{
__REG32 CMDINHCMD       : 1;
__REG32 CMDINHDAT       : 1;
__REG32 DATACT          : 1;
__REG32                 : 5;
__REG32 WRITETXFACT     : 1;
__REG32 READTXFACT      : 1;
__REG32 WRITEBUFEN      : 1;
__REG32 READBUFEN       : 1;
__REG32                 : 4;
__REG32 CARDIN          : 1;
__REG32 CARDSTABLE      : 1;
__REG32 CARDDETPIN      : 1;
__REG32 CARDWP          : 1;
__REG32 DATASTAT3_0     : 4;
__REG32 CMDSTAT         : 1;
__REG32 DATASTAT7_4     : 4;
__REG32                 : 3;
} __sdio_presentstate_bits;

/* SDIO Present state register */
typedef struct{
__REG32 LEDCTRL         : 1;
__REG32 TXFWIDTH        : 1;
__REG32 HSEN            : 1;
__REG32 DMASEL          : 2;
__REG32 SD8MODE         : 1;
__REG32 CARDDETLV       : 1;
__REG32 CARDDET         : 1;
__REG32 BUSPWR          : 1;
__REG32 BUSPWRSEL       : 3;
__REG32                 : 4;
__REG32 WINTEN          : 1;
__REG32 WINSEN          : 1;
__REG32 WREMEN          : 1;
__REG32                 : 5;
__REG32 STOP            : 1;
__REG32 CONT            : 1;
__REG32 READWAITCTRL    : 1;
__REG32 INT             : 1;
__REG32                 : 4;
} __sdio_wake_hostctrl_bits;

/* SDIO Clock control register */
typedef struct{
__REG32 ICLKEN          : 1;
__REG32 ICLKSTABLE      : 1;
__REG32 SDCLKEN         : 1;
__REG32                 : 5;
__REG32 SDCLKDIV        : 8;
__REG32 DTIMOUTVAL      : 4;
__REG32                 : 4;
__REG32 SWRESETALL      : 1;
__REG32 SWRESETCMD      : 1;
__REG32 SWRESETDAT      : 1;
__REG32                 : 5;
} __sdio_clkctrl_bits;

/* SDIO Normal interrupt status and error interrupt status register */
typedef struct{
__REG32 CMDCOMPLETE     : 1;
__REG32 TXFCOMPLETE     : 1;
__REG32 BLOCKGAPEV      : 1;
__REG32 DMAINT          : 1;
__REG32 BUFWRDY         : 1;
__REG32 BUFRRDY         : 1;
__REG32 CARDIN          : 1;
__REG32 CARDREM         : 1;
__REG32 CARDINT         : 1;
__REG32 BOOTACKRCVD     : 1;
__REG32 BOOTTERM        : 1;
__REG32                 : 4;
__REG32 ERRINT          : 1;
__REG32 CMDTIMEOUTERR   : 1;
__REG32 CMDCRCERR       : 1;
__REG32 CMDENDBITERR    : 1;
__REG32 CMDIDXERR       : 1;
__REG32 DATTIMEOUTERR   : 1;
__REG32 DATCRCERR       : 1;
__REG32 DATBITENDERR    : 1;
__REG32 CURRLIMERR      : 1;
__REG32 AUTOCMD12ERR    : 1;
__REG32 ADMAERR         : 1;
__REG32                 : 2;
__REG32 TARGRESERR      : 1;
__REG32 CEATAERR        : 1;
__REG32                 : 2;
} __sdio_intstat_bits;

/* SDIO Normal interrupt status and error interrupt status enable register */
typedef struct{
__REG32 CMDCOMPLETESTEN   : 1;
__REG32 TXFCOMPLETESTENQ  : 1;
__REG32 BLOCKGAPEVSTEN    : 1;
__REG32 DMAINTSTEN        : 1;
__REG32 BUFWRDYSTEN       : 1;
__REG32 BUDRDRDYSTEN      : 1;
__REG32 CARDINSTEN        : 1;
__REG32 CARDREMSTEN       : 1;
__REG32 CARDINTSTEN       : 1;
__REG32 BOOTACKSTEN       : 1;
__REG32 BOOTTERMINTSTEN   : 1;
__REG32                   : 5;
__REG32 CMDTIMEOUTERRSTEN : 1;
__REG32 CMDCRCERRSTEN     : 1;
__REG32 CMDENDBITERRSTEN  : 1;
__REG32 CMDIDXERRSTEN     : 1;
__REG32 DATTIMEOUTERRSTEN : 1;
__REG32 DATCRCERRSTEN     : 1;
__REG32 DATBITENDERRSTEN  : 1;
__REG32 CURRLIMERRSTEN    : 1;
__REG32 AUTOCMD12ERRSTEN  : 1;
__REG32 ADMAERRSTEN       : 1;
__REG32                   : 2;
__REG32 TARGRESERRSTEN    : 1;
__REG32 CEATAERRSTEN      : 1;
__REG32                   : 2;
} __sdio_intstaten_bits;

/* SDIO Normal interrupt signal and error signal status enable register */
typedef struct{
__REG32 CMDCOMPLETESIGEN    : 1;
__REG32 TXFCOMPLETESIGENQ   : 1;
__REG32 BLOCKGAPEVSIGEN     : 1;
__REG32 DMAINTSIGEN         : 1;
__REG32 BUFWRDYSIGEN        : 1;
__REG32 BUDRDRDYSIGEN       : 1;
__REG32 CARDINSIGEN         : 1;
__REG32 CARDREMSIGEN        : 1;
__REG32 CARDINTSIGEN        : 1;
__REG32 BOOTACKSIGEN        : 1;
__REG32 BOOTTERMINTSIGEN    : 1;
__REG32                     : 5;
__REG32 CMDTIMEOUTERRSIGEN  : 1;
__REG32 CMDCRCERRSIGEN      : 1;
__REG32 CMDENDBITERRSIGEN   : 1;
__REG32 CMDIDXERRSIGEN      : 1;
__REG32 DATTIMEOUTERRSIGEN  : 1;
__REG32 DATCRCERRSIGEN      : 1;
__REG32 DATBITENDERRSIGEN   : 1;
__REG32 CURRLIMERRSIGEN     : 1;
__REG32 AUTOCMD12ERRSIGEN   : 1;
__REG32 ADMAERRSIGEN        : 1;
__REG32                     : 2;
__REG32 TARGRESERRSIGEN     : 1;
__REG32 CEATAERRSIGEN       : 1;
__REG32                     : 2;
} __sdio_nintsigen_bits;

/* SDIO Auto CMD12 error status register */
typedef struct{
__REG32 AUTOCMD12NOEX       : 1;
__REG32 AUTOCMD12TIMEOUTERR : 1;
__REG32 AUTOCMD12CRCERR     : 1;
__REG32 AUTOCMD12ENDBITERR  : 1;
__REG32 AUTOCMD12IDXERR     : 1;
__REG32                     : 2;
__REG32 NOCMD               : 1;
__REG32                     :24;
} __sdio_autocmd12errstat_bits;

/* SDIO capabilities register */
typedef struct{
__REG32 TIMEOUTFREQ         : 6;
__REG32                     : 1;
__REG32 TIMEOUTUNIT         : 1;
__REG32 SDCLOCK             : 6;
__REG32                     : 2;
__REG32 MAXBLKLEN           : 2;
__REG32 EXTMED              : 1;
__REG32 ADMA2               : 1;
__REG32                     : 1;
__REG32 HIGHSPEED           : 1;
__REG32 SDMA                : 1;
__REG32 SUSRES              : 1;
__REG32 VS3V3               : 1;
__REG32 VS3V0               : 1;
__REG32 VS1V8               : 1;
__REG32 INT                 : 1;
__REG32 BIT64               : 1;
__REG32 SPI                 : 1;
__REG32 SPIBLK              : 1;
__REG32                     : 1;
} __sdio_capb_bits;

/* SDIO Maximum current capabilities register */
typedef struct{
__REG32 MAXCURR3V3          : 8;
__REG32 MAXCURR3V0          : 8;
__REG32 MAXCURR1V8          : 8;
__REG32                     : 8;
} __sdio_maxcurrcapb_bits;

/* SDIO Force event for Auto CMD12 error and error interrupt status register */
typedef struct{
__REG32 FE_AUTOCMD12NOEX        : 1;
__REG32 FE_AUTOCMD12TIMEOUTERR  : 1;
__REG32 FE_AUTOCMD12CRCERR      : 1;
__REG32 FE_AUTOCMD12ENDBITERR   : 1;
__REG32 FE_AUTOCMD12IDXERR      : 1;
__REG32                         : 2;
__REG32 FE_NOCMD                : 1;
__REG32                         : 8;
__REG32 FE_CMDTIMEOUTERREN      : 1;
__REG32 FE_CMDCRCERREN          : 1;
__REG32 FE_CMDENDBITERREN       : 1;
__REG32 FE_CMDIDXERREN          : 1;
__REG32 FE_DATTIMEOUTERREN      : 1;
__REG32 FE_DATCRCERREN          : 1;
__REG32 FE_DATBITENDERREN       : 1;
__REG32 FE_CURRLIMERREN         : 1;
__REG32 FE_AUTOCMD12ERREN       : 1;
__REG32 FE_ADMAERREN            : 1;
__REG32                         : 2;
__REG32 FE_TARGRESERREN         : 1;
__REG32 FE_CEATAERREN           : 1;
__REG32                         : 2;
} __sdio_force_evt_bits;

/* SDIO ADMA error status register */
typedef struct{
__REG32 ADMAERRSTATE    : 2;
__REG32 ADMALENERR      : 1;
__REG32                 :29;
} __sdio_adma_err_bits;

/* SDIO SPI interrupt support register */
typedef struct{
__REG32 SPIINT          : 8;
__REG32                 :24;
} __sdio_spiint_bits;

/* EMC Control Register */
typedef struct {
  __REG32 E         : 1;
  __REG32 M         : 1;
  __REG32 L         : 1;
  __REG32           :29;
} __emc_ctrl_bits;

/* EMC Status Register */
typedef struct {
  __REG32 B         : 1;
  __REG32 S         : 1;
  __REG32 SA        : 1;
  __REG32           :29;
} __emc_st_bits;

/* EMC Configuration Register */
typedef struct {
  __REG32 EM        : 1;
  __REG32           : 7;
  __REG32 CR        : 1;
  __REG32           :23;
} __emc_cfg_bits;

/* Dynamic Memory Control Register */
typedef struct {
  __REG32 CE        : 1;
  __REG32 CS        : 1;
  __REG32 SR        : 1;
  __REG32           : 2;
  __REG32 MMC       : 1;
  __REG32           : 1;
  __REG32 I         : 2;
  __REG32           : 4;
  __REG32 DP        : 1;
  __REG32           :18;
} __emc_dctrl_bits;

/* Dynamic Memory Refresh Timer Register */
typedef struct {
  __REG32 REFRESH   :11;
  __REG32           :21;
} __emc_drfr_bits;

/* Dynamic Memory Read Configuration Register */
typedef struct {
  __REG32 RD        : 2;
  __REG32           :30;
} __emc_drdcfg_bits;

/* Dynamic Memory Percentage Command Period Register */
typedef struct {
  __REG32 tRP       : 4;
  __REG32           :28;
} __emc_drp_bits;

/* Dynamic Memory Active to Precharge Command Period Register */
typedef struct {
  __REG32 tRAS      : 4;
  __REG32           :28;
} __emc_dras_bits;

/* Dynamic Memory Self-refresh Exit Time Register */
typedef struct {
  __REG32 tSREX     : 4;
  __REG32           :28;
} __emc_dsrex_bits;

/* Dynamic Memory Last Data Out to Active Time Register */
typedef struct {
  __REG32 tAPR      : 4;
  __REG32           :28;
} __emc_dapr_bits;

/* Dynamic Memory Data-in to Active Command Time Register */
typedef struct {
  __REG32 tDAL      : 4;
  __REG32           :28;
} __emc_ddal_bits;

/* Dynamic Memory Write Recovery Time Register */
typedef struct {
  __REG32 tWR       : 4;
  __REG32           :28;
} __emc_dwr_bits;

/* Dynamic Memory Active to Active Command Period Register */
typedef struct {
  __REG32 tRC       : 5;
  __REG32           :27;
} __emc_drc_bits;

/* Dynamic Memory Auto-refresh Period Register */
typedef struct {
  __REG32 tRFC      : 5;
  __REG32           :27;
} __emc_drfc_bits;

/* Dynamic Memory Exit Self-refresh Register */
typedef struct {
  __REG32 tXSR      : 5;
  __REG32           :27;
} __emc_dxsr_bits;

/* Dynamic Memory Active Bank A to Active Bank B Time Register */
typedef struct {
  __REG32 tRRD      : 4;
  __REG32           :28;
} __emc_drrd_bits;

/* Dynamic Memory Load Mode Register to Active Command Time */
typedef struct {
  __REG32 tMRD      : 4;
  __REG32           :28;
} __emc_dmrd_bits;

/* Static Memory Extended Wait Register */
typedef struct {
  __REG32 EXTENDEDWAIT  :10;
  __REG32               :22;
} __emc_s_ext_wait_bits;

/* Dynamic Memory Configuration Registers */
typedef struct {
  __REG32           : 3;
  __REG32 MD        : 2;
  __REG32           : 2;
  __REG32 AML       : 6;
  __REG32           : 1;
  __REG32 AMH       : 1;
  __REG32           : 4;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_d_config_bits;

/* Dynamic Memory RAS & CAS Delay Registers */
typedef struct {
  __REG32 RAS       : 2;
  __REG32           : 6;
  __REG32 CAS       : 2;
  __REG32           :22;
} __emc_d_ras_cas_bits;

/* Static Memory Configuration Registers */
typedef struct {
  __REG32 MW        : 2;
  __REG32           : 1;
  __REG32 PM        : 1;
  __REG32           : 2;
  __REG32 PC        : 1;
  __REG32 PB        : 1;
  __REG32 EW        : 1;
  __REG32           :10;
  __REG32 B         : 1;
  __REG32 P         : 1;
  __REG32           :11;
} __emc_s_config_bits;

/* Static Memory Write Enable Delay Registers */
typedef struct {
  __REG32 WAITWEN   : 4;
  __REG32           :28;
} __emc_s_wait_wen_bits;

/* Static Memory Output Enable Delay Registers */
typedef struct {
  __REG32 WAITOEN   : 4;
  __REG32           :28;
} __emc_s_wait_oen_bits;

/* Static Memory Read Delay Registers */
typedef struct {
  __REG32 WAITRD    : 5;
  __REG32           :27;
} __emc_s_wait_rd_bits;

/* Static Memory Page Mode Read Delay Registers */
typedef struct {
  __REG32 WAITPAGE  : 5;
  __REG32           :27;
} __emc_s_wait_pg_bits;

/* Static Memory Write Delay Registers */
typedef struct {
  __REG32 WAITWR    : 5;
  __REG32           :27;
} __emc_s_wait_wr_bits;

/* Static Memory Turn Round Delay Registers */
typedef struct {
  __REG32 WAITTURN  : 4;
  __REG32           :28;
} __emc_s_wait_turn_bits;

/* SCT configuration register */
typedef struct{
__REG32 UNIFY                 : 1;
__REG32 CLKMODE               : 2;
__REG32 CLKSEL                : 4;
__REG32 NORELAODL_NORELOADU   : 1;
__REG32 NORELOADH             : 1;
__REG32 INSYNC0               : 1;
__REG32 INSYNC1               : 1;
__REG32 INSYNC2               : 1;
__REG32 INSYNC3               : 1;
__REG32 INSYNC4               : 1;
__REG32 INSYNC5               : 1;
__REG32 INSYNC6               : 1;
__REG32 INSYNC7               : 1;
__REG32                       :15;
} __sctconfig_bits;

/* SCT control register */
typedef struct{
__REG32 DOWN_L                : 1;
__REG32 STOP_L                : 1;
__REG32 HALT_L                : 1;
__REG32 CLRCTR_L              : 1;
__REG32 BIDIR_L               : 1;
__REG32 PRE_L                 : 8;
__REG32                       : 3;
__REG32 DOWN_H                : 1;
__REG32 STOP_H                : 1;
__REG32 HALT_H                : 1;
__REG32 CLRCTR_H              : 1;
__REG32 BIDIR_H               : 1;
__REG32 PRE_H                 : 8;
__REG32                       : 3;
} __sctctrl_bits;

/* SCT limit register */
typedef struct{
__REG32 LIMMSK0_L             : 1;
__REG32 LIMMSK1_L             : 1;
__REG32 LIMMSK2_L             : 1;
__REG32 LIMMSK3_L             : 1;
__REG32 LIMMSK4_L             : 1;
__REG32 LIMMSK5_L             : 1;
__REG32 LIMMSK6_L             : 1;
__REG32 LIMMSK7_L             : 1;
__REG32 LIMMSK8_L             : 1;
__REG32 LIMMSK9_L             : 1;
__REG32 LIMMSK10_L            : 1;
__REG32 LIMMSK11_L            : 1;
__REG32 LIMMSK12_L            : 1;
__REG32 LIMMSK13_L            : 1;
__REG32 LIMMSK14_L            : 1;
__REG32 LIMMSK15_L            : 1;
__REG32 LIMMSK0_H             : 1;
__REG32 LIMMSK1_H             : 1;
__REG32 LIMMSK2_H             : 1;
__REG32 LIMMSK3_H             : 1;
__REG32 LIMMSK4_H             : 1;
__REG32 LIMMSK5_H             : 1;
__REG32 LIMMSK6_H             : 1;
__REG32 LIMMSK7_H             : 1;
__REG32 LIMMSK8_H             : 1;
__REG32 LIMMSK9_H             : 1;
__REG32 LIMMSK10_H            : 1;
__REG32 LIMMSK11_H            : 1;
__REG32 LIMMSK12_H            : 1;
__REG32 LIMMSK13_H            : 1;
__REG32 LIMMSK14_H            : 1;
__REG32 LIMMSK15_H            : 1;
} __sctlimit_bits;

/* SCT halt condition register */
typedef struct{
__REG32 HALTMSK0_L            : 1;
__REG32 HALTMSK1_L            : 1;
__REG32 HALTMSK2_L            : 1;
__REG32 HALTMSK3_L            : 1;
__REG32 HALTMSK4_L            : 1;
__REG32 HALTMSK5_L            : 1;
__REG32 HALTMSK6_L            : 1;
__REG32 HALTMSK7_L            : 1;
__REG32 HALTMSK8_L            : 1;
__REG32 HALTMSK9_L            : 1;
__REG32 HALTMSK10_L           : 1;
__REG32 HALTMSK11_L           : 1;
__REG32 HALTMSK12_L           : 1;
__REG32 HALTMSK13_L           : 1;
__REG32 HALTMSK14_L           : 1;
__REG32 HALTMSK15_L           : 1;
__REG32 HALTMSK0_H            : 1;
__REG32 HALTMSK1_H            : 1;
__REG32 HALTMSK2_H            : 1;
__REG32 HALTMSK3_H            : 1;
__REG32 HALTMSK4_H            : 1;
__REG32 HALTMSK5_H            : 1;
__REG32 HALTMSK6_H            : 1;
__REG32 HALTMSK7_H            : 1;
__REG32 HALTMSK8_H            : 1;
__REG32 HALTMSK9_H            : 1;
__REG32 HALTMSK10_H           : 1;
__REG32 HALTMSK11_H           : 1;
__REG32 HALTMSK12_H           : 1;
__REG32 HALTMSK13_H           : 1;
__REG32 HALTMSK14_H           : 1;
__REG32 HALTMSK15_H           : 1;
} __scthalt_bits;

/* UT stop condition register */
typedef struct{
__REG32 STOPMSK0_L            : 1;
__REG32 STOPMSK1_L            : 1;
__REG32 STOPMSK2_L            : 1;
__REG32 STOPMSK3_L            : 1;
__REG32 STOPMSK4_L            : 1;
__REG32 STOPMSK5_L            : 1;
__REG32 STOPMSK6_L            : 1;
__REG32 STOPMSK7_L            : 1;
__REG32 STOPMSK8_L            : 1;
__REG32 STOPMSK9_L            : 1;
__REG32 STOPMSK10_L           : 1;
__REG32 STOPMSK11_L           : 1;
__REG32 STOPMSK12_L           : 1;
__REG32 STOPMSK13_L           : 1;
__REG32 STOPMSK14_L           : 1;
__REG32 STOPMSK15_L           : 1;
__REG32 STOPMSK0_H            : 1;
__REG32 STOPMSK1_H            : 1;
__REG32 STOPMSK2_H            : 1;
__REG32 STOPMSK3_H            : 1;
__REG32 STOPMSK4_H            : 1;
__REG32 STOPMSK5_H            : 1;
__REG32 STOPMSK6_H            : 1;
__REG32 STOPMSK7_H            : 1;
__REG32 STOPMSK8_H            : 1;
__REG32 STOPMSK9_H            : 1;
__REG32 STOPMSK10_H           : 1;
__REG32 STOPMSK11_H           : 1;
__REG32 STOPMSK12_H           : 1;
__REG32 STOPMSK13_H           : 1;
__REG32 STOPMSK14_H           : 1;
__REG32 STOPMSK15_H           : 1;
} __sctstop_bits;

/* SCT start condition register */
typedef struct{
__REG32 STARTMSK0_L           : 1;
__REG32 STARTMSK1_L           : 1;
__REG32 STARTMSK2_L           : 1;
__REG32 STARTMSK3_L           : 1;
__REG32 STARTMSK4_L           : 1;
__REG32 STARTMSK5_L           : 1;
__REG32 STARTMSK6_L           : 1;
__REG32 STARTMSK7_L           : 1;
__REG32 STARTMSK8_L           : 1;
__REG32 STARTMSK9_L           : 1;
__REG32 STARTMSK10_L          : 1;
__REG32 STARTMSK11_L          : 1;
__REG32 STARTMSK12_L          : 1;
__REG32 STARTMSK13_L          : 1;
__REG32 STARTMSK14_L          : 1;
__REG32 STARTMSK15_L          : 1;
__REG32 STARTMSK0_H           : 1;
__REG32 STARTMSK1_H           : 1;
__REG32 STARTMSK2_H           : 1;
__REG32 STARTMSK3_H           : 1;
__REG32 STARTMSK4_H           : 1;
__REG32 STARTMSK5_H           : 1;
__REG32 STARTMSK6_H           : 1;
__REG32 STARTMSK7_H           : 1;
__REG32 STARTMSK8_H           : 1;
__REG32 STARTMSK9_H           : 1;
__REG32 STARTMSK10_H          : 1;
__REG32 STARTMSK11_H          : 1;
__REG32 STARTMSK12_H          : 1;
__REG32 STARTMSK13_H          : 1;
__REG32 STARTMSK14_H          : 1;
__REG32 STARTMSK15_H          : 1;
} __sctstart_bits;

/* SCT counter register */
typedef struct{
__REG32 CTR_L                 :16;
__REG32 CTR_H                 :16;
} __sctcount_bits;

/* UT state register */
typedef struct{
__REG32 STATE_L               : 5;
__REG32                       :11;
__REG32 STATE_H               : 5;
__REG32                       :11;
} __sctstate_bits;

/* SCT input register */
typedef struct{
__REG32 AIN0                  : 1;
__REG32 AIN1                  : 1;
__REG32 AIN2                  : 1;
__REG32 AIN3                  : 1;
__REG32 AIN4                  : 1;
__REG32 AIN5                  : 1;
__REG32 AIN6                  : 1;
__REG32 AIN7                  : 1;
__REG32                       : 8;
__REG32 SIN0                  : 1;
__REG32 SIN1                  : 1;
__REG32 SIN2                  : 1;
__REG32 SIN3                  : 1;
__REG32 SIN4                  : 1;
__REG32 SIN5                  : 1;
__REG32 SIN6                  : 1;
__REG32 SIN7                  : 1;
__REG32                       : 8;
} __sctinput_bits;

/* SCT match/capture registers mode register */
typedef struct{
__REG32 REGMOD0_L             : 1;
__REG32 REGMOD1_L             : 1;
__REG32 REGMOD2_L             : 1;
__REG32 REGMOD3_L             : 1;
__REG32 REGMOD4_L             : 1;
__REG32 REGMOD5_L             : 1;
__REG32 REGMOD6_L             : 1;
__REG32 REGMOD7_L             : 1;
__REG32 REGMOD8_L             : 1;
__REG32 REGMOD9_L             : 1;
__REG32 REGMOD10_L            : 1;
__REG32 REGMOD11_L            : 1;
__REG32 REGMOD12_L            : 1;
__REG32 REGMOD13_L            : 1;
__REG32 REGMOD14_L            : 1;
__REG32 REGMOD15_L            : 1;
__REG32 REGMOD0_H             : 1;
__REG32 REGMOD1_H             : 1;
__REG32 REGMOD2_H             : 1;
__REG32 REGMOD3_H             : 1;
__REG32 REGMOD4_H             : 1;
__REG32 REGMOD5_H             : 1;
__REG32 REGMOD6_H             : 1;
__REG32 REGMOD7_H             : 1;
__REG32 REGMOD8_H             : 1;
__REG32 REGMOD9_H             : 1;
__REG32 REGMOD10_H            : 1;
__REG32 REGMOD11_H            : 1;
__REG32 REGMOD12_H            : 1;
__REG32 REGMOD13_H            : 1;
__REG32 REGMOD14_H            : 1;
__REG32 REGMOD15_H            : 1;
} __sctregmode_bits;

/* SCT output register */
typedef struct{
__REG32 OUT0                  : 1;
__REG32 OUT1                  : 1;
__REG32 OUT2                  : 1;
__REG32 OUT3                  : 1;
__REG32 OUT4                  : 1;
__REG32 OUT5                  : 1;
__REG32 OUT6                  : 1;
__REG32 OUT7                  : 1;
__REG32 OUT8                  : 1;
__REG32 OUT9                  : 1;
__REG32 OUT10                 : 1;
__REG32 OUT11                 : 1;
__REG32 OUT12                 : 1;
__REG32 OUT13                 : 1;
__REG32 OUT14                 : 1;
__REG32 OUT15                 : 1;
__REG32                       :16;
} __sctoutput_bits;

/* SCT bidirectional output control register */
typedef struct{
__REG32 SETCLR0               : 2;
__REG32 SETCLR1               : 2;
__REG32 SETCLR2               : 2;
__REG32 SETCLR3               : 2;
__REG32 SETCLR4               : 2;
__REG32 SETCLR5               : 2;
__REG32 SETCLR6               : 2;
__REG32 SETCLR7               : 2;
__REG32 SETCLR8               : 2;
__REG32 SETCLR9               : 2;
__REG32 SETCLR10              : 2;
__REG32 SETCLR11              : 2;
__REG32 SETCLR12              : 2;
__REG32 SETCLR13              : 2;
__REG32 SETCLR14              : 2;
__REG32 SETCLR15              : 2;
} __sctoutputdirctrl_bits;

/* SCT conflict resolution register */
typedef struct{
__REG32 O0RES                 : 2;
__REG32 O1RES                 : 2;
__REG32 O2RES                 : 2;
__REG32 O3RES                 : 2;
__REG32 O4RES                 : 2;
__REG32 O5RES                 : 2;
__REG32 O6RES                 : 2;
__REG32 O7RES                 : 2;
__REG32 O8RES                 : 2;
__REG32 O9RES                 : 2;
__REG32 O10RES                : 2;
__REG32 O11RES                : 2;
__REG32 O12RES                : 2;
__REG32 O13RES                : 2;
__REG32 O14RES                : 2;
__REG32 O15RES                : 2;
} __sctres_bits;

/* SCT DMA request 0 */
typedef struct{
__REG32 DEV0_0                : 1;
__REG32 DEV1_0                : 1;
__REG32 DEV2_0                : 1;
__REG32 DEV3_0                : 1;
__REG32 DEV4_0                : 1;
__REG32 DEV5_0                : 1;
__REG32 DEV6_0                : 1;
__REG32 DEV7_0                : 1;
__REG32 DEV8_0                : 1;
__REG32 DEV9_0                : 1;
__REG32 DEV10_0               : 1;
__REG32 DEV11_0               : 1;
__REG32 DEV12_0               : 1;
__REG32 DEV13_0               : 1;
__REG32 DEV14_0               : 1;
__REG32 DEV15_0               : 1;
__REG32                       :14;
__REG32 DRL0                  : 1;
__REG32 DRQ0                  : 1;
} __sctdmareq0_bits;

/* SCT DMA request 1 */
typedef struct{
__REG32 DEV0_1                : 1;
__REG32 DEV1_1                : 1;
__REG32 DEV2_1                : 1;
__REG32 DEV3_1                : 1;
__REG32 DEV4_1                : 1;
__REG32 DEV5_1                : 1;
__REG32 DEV6_1                : 1;
__REG32 DEV7_1                : 1;
__REG32 DEV8_1                : 1;
__REG32 DEV9_1                : 1;
__REG32 DEV10_1               : 1;
__REG32 DEV11_1               : 1;
__REG32 DEV12_1               : 1;
__REG32 DEV13_1               : 1;
__REG32 DEV14_1               : 1;
__REG32 DEV15_1               : 1;
__REG32                       :14;
__REG32 DRL1                  : 1;
__REG32 DRQ1                  : 1;
} __sctdmareq1_bits;

/* SCT flag enable register */
typedef struct{
__REG32 IEN0                  : 1;
__REG32 IEN1                  : 1;
__REG32 IEN2                  : 1;
__REG32 IEN3                  : 1;
__REG32 IEN4                  : 1;
__REG32 IEN5                  : 1;
__REG32 IEN6                  : 1;
__REG32 IEN7                  : 1;
__REG32 IEN8                  : 1;
__REG32 IEN9                  : 1;
__REG32 IEN10                 : 1;
__REG32 IEN11                 : 1;
__REG32 IEN12                 : 1;
__REG32 IEN13                 : 1;
__REG32 IEN14                 : 1;
__REG32 IEN15                 : 1;
__REG32                       :16;
} __scteven_bits;

/* SCT event flag register */
typedef struct{
__REG32 FLAG0                 : 1;
__REG32 FLAG1                 : 1;
__REG32 FLAG2                 : 1;
__REG32 FLAG3                 : 1;
__REG32 FLAG4                 : 1;
__REG32 FLAG5                 : 1;
__REG32 FLAG6                 : 1;
__REG32 FLAG7                 : 1;
__REG32 FLAG8                 : 1;
__REG32 FLAG9                 : 1;
__REG32 FLAG10                : 1;
__REG32 FLAG11                : 1;
__REG32 FLAG12                : 1;
__REG32 FLAG13                : 1;
__REG32 FLAG14                : 1;
__REG32 FLAG15                : 1;
__REG32                       :16;
} __sctevflag_bits;

/* SCT conflict enable register */
typedef struct{
__REG32 NCEN0                 : 1;
__REG32 NCEN1                 : 1;
__REG32 NCEN2                 : 1;
__REG32 NCEN3                 : 1;
__REG32 NCEN4                 : 1;
__REG32 NCEN5                 : 1;
__REG32 NCEN6                 : 1;
__REG32 NCEN7                 : 1;
__REG32 NCEN8                 : 1;
__REG32 NCEN9                 : 1;
__REG32 NCEN10                : 1;
__REG32 NCEN11                : 1;
__REG32 NCEN12                : 1;
__REG32 NCEN13                : 1;
__REG32 NCEN14                : 1;
__REG32 NCEN15                : 1;
__REG32                       :16;
} __sctconen_bits;

/* SCT conflict flag register */
typedef struct{
__REG32 NCFLAG0               : 1;
__REG32 NCFLAG1               : 1;
__REG32 NCFLAG2               : 1;
__REG32 NCFLAG3               : 1;
__REG32 NCFLAG4               : 1;
__REG32 NCFLAG5               : 1;
__REG32 NCFLAG6               : 1;
__REG32 NCFLAG7               : 1;
__REG32 NCFLAG8               : 1;
__REG32 NCFLAG9               : 1;
__REG32 NCFLAG10              : 1;
__REG32 NCFLAG11              : 1;
__REG32 NCFLAG12              : 1;
__REG32 NCFLAG13              : 1;
__REG32 NCFLAG14              : 1;
__REG32 NCFLAG15              : 1;
__REG32                       :14;
__REG32 BUSERRL               : 1;
__REG32 BUSERRH               : 1;
} __sctconflag_bits;

/* SCT match and capture registers */
typedef union{
  /* SCTMATCHx */
  struct {
    __REG32 MATCH_L             :16;
    __REG32 MATCH_H             :16;
  };
  /* SCTCAPx */
  struct {
    __REG32 CAP_L               :16;
    __REG32 CAP_H               :16;
  };
} __sctmatch_cap_bits;   

/* SCT match reload and capture control registers */
typedef union{
  /* SCTMATCHRELx */
  struct {
    __REG32 RELOAD_L            :16;
    __REG32 RELOAD_H            :16;
  };
  /* SCTCAPCTRLx */
  struct {
    __REG32 CAPCON0_L           : 1;
    __REG32 CAPCON1_L           : 1;
    __REG32 CAPCON2_L           : 1;
    __REG32 CAPCON3_L           : 1;
    __REG32 CAPCON4_L           : 1;
    __REG32 CAPCON5_L           : 1;
    __REG32 CAPCON6_L           : 1;
    __REG32 CAPCON7_L           : 1;
    __REG32 CAPCON8_L           : 1;
    __REG32 CAPCON9_L           : 1;
    __REG32 CAPCON10_L          : 1;
    __REG32 CAPCON11_L          : 1;
    __REG32 CAPCON12_L          : 1;
    __REG32 CAPCON13_L          : 1;
    __REG32 CAPCON14_L          : 1;
    __REG32 CAPCON15_L          : 1;
    __REG32 CAPCON0_H           : 1;
    __REG32 CAPCON1_H           : 1;
    __REG32 CAPCON2_H           : 1;
    __REG32 CAPCON3_H           : 1;
    __REG32 CAPCON4_H           : 1;
    __REG32 CAPCON5_H           : 1;
    __REG32 CAPCON6_H           : 1;
    __REG32 CAPCON7_H           : 1;
    __REG32 CAPCON8_H           : 1;
    __REG32 CAPCON9_H           : 1;
    __REG32 CAPCON10_H          : 1;
    __REG32 CAPCON11_H          : 1;
    __REG32 CAPCON12_H          : 1;
    __REG32 CAPCON13_H          : 1;
    __REG32 CAPCON14_H          : 1;
    __REG32 CAPCON15_H          : 1;
  };
} __sctmatchrel_capctrl_bits;

/* SCT event state mask registers */
typedef struct{
__REG32 STATEMSK0             : 1;
__REG32 STATEMSK1             : 1;
__REG32 STATEMSK2             : 1;
__REG32 STATEMSK3             : 1;
__REG32 STATEMSK4             : 1;
__REG32 STATEMSK5             : 1;
__REG32 STATEMSK6             : 1;
__REG32 STATEMSK7             : 1;
__REG32 STATEMSK8             : 1;
__REG32 STATEMSK9             : 1;
__REG32 STATEMSK10            : 1;
__REG32 STATEMSK11            : 1;
__REG32 STATEMSK12            : 1;
__REG32 STATEMSK13            : 1;
__REG32 STATEMSK14            : 1;
__REG32 STATEMSK15            : 1;
__REG32 STATEMSK16            : 1;
__REG32 STATEMSK17            : 1;
__REG32 STATEMSK18            : 1;
__REG32 STATEMSK19            : 1;
__REG32 STATEMSK20            : 1;
__REG32 STATEMSK21            : 1;
__REG32 STATEMSK22            : 1;
__REG32 STATEMSK23            : 1;
__REG32 STATEMSK24            : 1;
__REG32 STATEMSK25            : 1;
__REG32 STATEMSK26            : 1;
__REG32 STATEMSK27            : 1;
__REG32 STATEMSK28            : 1;
__REG32 STATEMSK29            : 1;
__REG32 STATEMSK30            : 1;
__REG32 STATEMSK31            : 1;
} __sctevstatemsk_bits;

/* SCT event control registers */
typedef struct{
__REG32 MATCHSEL              : 4;
__REG32 HEVENT                : 1;
__REG32 OUTSEL                : 1;
__REG32 IOSEL                 : 4;
__REG32 IOCOND                : 2;
__REG32 COMBMODE              : 2;
__REG32 STATELD               : 1;
__REG32 STATEV                : 5;
__REG32                       :12;
} __sctevctrl_bits;

/* SCT output set registers */
typedef struct{
__REG32 SET0                  : 1;
__REG32 SET1                  : 1;
__REG32 SET2                  : 1;
__REG32 SET3                  : 1;
__REG32 SET4                  : 1;
__REG32 SET5                  : 1;
__REG32 SET6                  : 1;
__REG32 SET7                  : 1;
__REG32 SET8                  : 1;
__REG32 SET9                  : 1;
__REG32 SET10                 : 1;
__REG32 SET11                 : 1;
__REG32 SET12                 : 1;
__REG32 SET13                 : 1;
__REG32 SET14                 : 1;
__REG32 SET15                 : 1;
__REG32                       :16;
} __sctoutputset_bits;

/* SCT output clear registers */
typedef struct{
__REG32 CLR0                  : 1;
__REG32 CLR1                  : 1;
__REG32 CLR2                  : 1;
__REG32 CLR3                  : 1;
__REG32 CLR4                  : 1;
__REG32 CLR5                  : 1;
__REG32 CLR6                  : 1;
__REG32 CLR7                  : 1;
__REG32 CLR8                  : 1;
__REG32 CLR9                  : 1;
__REG32 CLR10                 : 1;
__REG32 CLR11                 : 1;
__REG32 CLR12                 : 1;
__REG32 CLR13                 : 1;
__REG32 CLR14                 : 1;
__REG32 CLR15                 : 1;
__REG32                       :16;
} __sctoutputcl_bits;

/* TIMER interrupt register */
typedef struct{
__REG32 MR0INT  : 1;
__REG32 MR1INT  : 1;
__REG32 MR2INT  : 1;
__REG32 MR3INT  : 1;
__REG32 CR0INT  : 1;
__REG32 CR1INT  : 1;
__REG32 CR2INT  : 1;
__REG32 CR3INT  : 1;
__REG32         :24;
} __ir_bits;

/* TIMER control register */
typedef struct{
__REG32 CE  : 1;
__REG32 CR  : 1;
__REG32     :30;
} __tcr_bits;

/* TIMER count control register */
typedef struct{
__REG32 CTM : 2;     /*Counter/Timer Mode*/
__REG32 CIS : 2;     /*Count Input Select*/
__REG32     :28;
} __ctcr_bits;

/* TIMER match control register */
typedef struct{
__REG32 MR0I     : 1;
__REG32 MR0R     : 1;
__REG32 MR0S     : 1;
__REG32 MR1I     : 1;
__REG32 MR1R     : 1;
__REG32 MR1S     : 1;
__REG32 MR2I     : 1;
__REG32 MR2R     : 1;
__REG32 MR2S     : 1;
__REG32 MR3I     : 1;
__REG32 MR3R     : 1;
__REG32 MR3S     : 1;
__REG32          :20;
} __mcr_bits;

/* TIMER capture control register */
typedef struct{
__REG32 CAP0RE   : 1;
__REG32 CAP0FE   : 1;
__REG32 CAP0I    : 1;
__REG32 CAP1RE   : 1;
__REG32 CAP1FE   : 1;
__REG32 CAP1I    : 1;
__REG32 CAP2RE   : 1;
__REG32 CAP2FE   : 1;
__REG32 CAP2I    : 1;
__REG32 CAP3RE   : 1;
__REG32 CAP3FE   : 1;
__REG32 CAP3I    : 1;
__REG32          :20;
} __tccr_bits;

/* TIMER external match register */
typedef struct{
__REG32 EM0   : 1;
__REG32 EM1   : 1;
__REG32 EM2   : 1;
__REG32 EM3   : 1;
__REG32 EMC0  : 2;
__REG32 EMC1  : 2;
__REG32 EMC2  : 2;
__REG32 EMC3  : 2;
__REG32       :20;
} __emr_bits;

/* MCPWM Control read  */
typedef struct{
__REG32 RUN0    : 1;
__REG32 CENTER0 : 1;
__REG32 POLA0   : 1;
__REG32 DTE0    : 1;
__REG32 DISUP0  : 1;
__REG32         : 3;
__REG32 RUN1    : 1;
__REG32 CENTER1 : 1;
__REG32 POLA1   : 1;
__REG32 DTE1    : 1;
__REG32 DISUP1  : 1;
__REG32         : 3;
__REG32 RUN2    : 1;
__REG32 CENTER2 : 1;
__REG32 POLA2   : 1;
__REG32 DTE2    : 1;
__REG32 DISUP2  : 1;
__REG32         : 8;
__REG32 INVBDC  : 1;
__REG32 ACMODE  : 1;
__REG32 DCMODE  : 1;
} __mccon_bits;

/*MCPWM Control set  */
typedef struct{
__REG32 RUN0_SET    : 1;
__REG32 CENTER0_SET : 1;
__REG32 POLA0_SET   : 1;
__REG32 DTE0_SET    : 1;
__REG32 DISUP0_SET  : 1;
__REG32             : 3;
__REG32 RUN1_SET    : 1;
__REG32 CENTER1_SET : 1;
__REG32 POLA1_SET   : 1;
__REG32 DTE1_SET    : 1;
__REG32 DISUP1_SET  : 1;
__REG32             : 3;
__REG32 RUN2_SET    : 1;
__REG32 CENTER2_SET : 1;
__REG32 POLA2_SET   : 1;
__REG32 DTE2_SET    : 1;
__REG32 DISUP2_SET  : 1;
__REG32             : 8;
__REG32 INVBDC_SET  : 1;
__REG32 ACMODE_SET  : 1;
__REG32 DCMODE_SET  : 1;
} __mccon_set_bits;

/*MCPWM Control clear  */
typedef struct{
__REG32 RUN0_CLR    : 1;
__REG32 CENTER0_CLR : 1;
__REG32 POLA0_CLR   : 1;
__REG32 DTE0_CLR    : 1;
__REG32 DISUP0_CLR  : 1;
__REG32             : 3;
__REG32 RUN1_CLR    : 1;
__REG32 CENTER1_CLR : 1;
__REG32 POLA1_CLR   : 1;
__REG32 DTE1_CLR    : 1;
__REG32 DISUP1_CLR  : 1;
__REG32             : 3;
__REG32 RUN2_CLR    : 1;
__REG32 CENTER2_CLR : 1;
__REG32 POLA2_CLR   : 1;
__REG32 DTE2_CLR    : 1;
__REG32 DISUP2_CLR  : 1;
__REG32             : 8;
__REG32 INVBDC_CLR  : 1;
__REG32 ACMODE_CLR  : 1;
__REG32 DCMODE_CLR  : 1;
} __mccon_clr_bits;

/* MCPWM Capture control register */
typedef struct{
__REG32 CAP0MCI0_RE : 1;
__REG32 CAP0MCI0_FE : 1;
__REG32 CAP0MCI1_RE : 1;
__REG32 CAP0MCI1_FE : 1;
__REG32 CAP0MCI2_RE : 1;
__REG32 CAP0MCI2_FE : 1;
__REG32 CAP1MCI0_RE : 1;
__REG32 CAP1MCI0_FE : 1;
__REG32 CAP1MCI1_RE : 1;
__REG32 CAP1MCI1_FE : 1;
__REG32 CAP1MCI2_RE : 1;
__REG32 CAP1MCI2_FE : 1;
__REG32 CAP2MCI0_RE : 1;
__REG32 CAP2MCI0_FE : 1;
__REG32 CAP2MCI1_RE : 1;
__REG32 CAP2MCI1_FE : 1;
__REG32 CAP2MCI2_RE : 1;
__REG32 CAP2MCI2_FE : 1;
__REG32 RT0         : 1;
__REG32 RT1         : 1;
__REG32 RT2         : 1;
__REG32 HNFCAP0     : 1;
__REG32 HNFCAP1     : 1;
__REG32 HNFCAP2     : 1;
__REG32             : 8;
} __mccapcon_bits;

/* MCPWM Capture Control set  */
typedef struct{
__REG32 CAP0MCI0_RE_SET : 1;
__REG32 CAP0MCI0_FE_SET : 1;
__REG32 CAP0MCI1_RE_SET : 1;
__REG32 CAP0MCI1_FE_SET : 1;
__REG32 CAP0MCI2_RE_SET : 1;
__REG32 CAP0MCI2_FE_SET : 1;
__REG32 CAP1MCI0_RE_SET : 1;
__REG32 CAP1MCI0_FE_SET : 1;
__REG32 CAP1MCI1_RE_SET : 1;
__REG32 CAP1MCI1_FE_SET : 1;
__REG32 CAP1MCI2_RE_SET : 1;
__REG32 CAP1MCI2_FE_SET : 1;
__REG32 CAP2MCI0_RE_SET : 1;
__REG32 CAP2MCI0_FE_SET : 1;
__REG32 CAP2MCI1_RE_SET : 1;
__REG32 CAP2MCI1_FE_SET : 1;
__REG32 CAP2MCI2_RE_SET : 1;
__REG32 CAP2MCI2_FE_SET : 1;
__REG32 RT0_SET         : 1;
__REG32 RT1_SET         : 1;
__REG32 RT2_SET         : 1;
__REG32 HNFCAP0_SET     : 1;
__REG32 HNFCAP1_SET     : 1;
__REG32 HNFCAP2_SET     : 1;
__REG32                 : 8;
} __mccapcon_set_bits;

/* MCPWM Capture control clear  */
typedef struct{
__REG32 CAP0MCI0_RE_CLR : 1;
__REG32 CAP0MCI0_FE_CLR : 1;
__REG32 CAP0MCI1_RE_CLR : 1;
__REG32 CAP0MCI1_FE_CLR : 1;
__REG32 CAP0MCI2_RE_CLR : 1;
__REG32 CAP0MCI2_FE_CLR : 1;
__REG32 CAP1MCI0_RE_CLR : 1;
__REG32 CAP1MCI0_FE_CLR : 1;
__REG32 CAP1MCI1_RE_CLR : 1;
__REG32 CAP1MCI1_FE_CLR : 1;
__REG32 CAP1MCI2_RE_CLR : 1;
__REG32 CAP1MCI2_FE_CLR : 1;
__REG32 CAP2MCI0_RE_CLR : 1;
__REG32 CAP2MCI0_FE_CLR : 1;
__REG32 CAP2MCI1_RE_CLR : 1;
__REG32 CAP2MCI1_FE_CLR : 1;
__REG32 CAP2MCI2_RE_CLR : 1;
__REG32 CAP2MCI2_FE_CLR : 1;
__REG32 RT0_CLR         : 1;
__REG32 RT1_CLR         : 1;
__REG32 RT2_CLR         : 1;
__REG32 HNFCAP0_CLR     : 1;
__REG32 HNFCAP1_CLR     : 1;
__REG32 HNFCAP2_CLR     : 1;
__REG32                 : 8;
} __mccapcon_clr_bits;

/* MCPWM interrupt enable register */
typedef struct{
__REG32 ILIM0       : 1;
__REG32 IMAT0       : 1;
__REG32 ICAP0       : 1;
__REG32             : 1;
__REG32 ILIM1       : 1;
__REG32 IMAT1       : 1;
__REG32 ICAP1       : 1;
__REG32             : 1;
__REG32 ILIM2       : 1;
__REG32 IMAT2       : 1;
__REG32 ICAP2       : 1;
__REG32             : 4;
__REG32 ABORT       : 1;
__REG32             :16;
} __mcinten_bits;

/* MCPWM Interrupt Enable set */
typedef struct{
__REG32 ILIM0_SET       : 1;
__REG32 IMAT0_SET       : 1;
__REG32 ICAP0_SET       : 1;
__REG32                 : 1;
__REG32 ILIM1_SET       : 1;
__REG32 IMAT1_SET       : 1;
__REG32 ICAP1_SET       : 1;
__REG32                 : 1;
__REG32 ILIM2_SET       : 1;
__REG32 IMAT2_SET       : 1;
__REG32 ICAP2_SET       : 1;
__REG32                 : 4;
__REG32 ABORT_SET       : 1;
__REG32                 :16;
} __mcinten_set_bits;

/* MCPWM Interrupt Enable clear */
typedef struct{
__REG32 ILIM0_CLR       : 1;
__REG32 IMAT0_CLR       : 1;
__REG32 ICAP0_CLR       : 1;
__REG32                 : 1;
__REG32 ILIM1_CLR       : 1;
__REG32 IMAT1_CLR       : 1;
__REG32 ICAP1_CLR       : 1;
__REG32                 : 1;
__REG32 ILIM2_CLR       : 1;
__REG32 IMAT2_CLR       : 1;
__REG32 ICAP2_CLR       : 1;
__REG32                 : 4;
__REG32 ABORT_CLR       : 1;
__REG32                 :16;
} __mcinten_clr_bits;

/* MCPWM Interrupt Flags */
typedef struct{
__REG32 ILIM0_F       : 1;
__REG32 IMAT0_F       : 1;
__REG32 ICAP0_F       : 1;
__REG32               : 1;
__REG32 ILIM1_F       : 1;
__REG32 IMAT1_F       : 1;
__REG32 ICAP1_F       : 1;
__REG32               : 1;
__REG32 ILIM2_F       : 1;
__REG32 IMAT2_F       : 1;
__REG32 ICAP2_F       : 1;
__REG32               : 4;
__REG32 ABORT_F       : 1;
__REG32               :16;
} __mcintf_bits;

/* MCPWM Interrupt Flags set */
typedef struct{
__REG32 ILIM0_F_SET       : 1;
__REG32 IMAT0_F_SET       : 1;
__REG32 ICAP0_F_SET       : 1;
__REG32                   : 1;
__REG32 ILIM1_F_SET       : 1;
__REG32 IMAT1_F_SET       : 1;
__REG32 ICAP1_F_SET       : 1;
__REG32                   : 1;
__REG32 ILIM2_F_SET       : 1;
__REG32 IMAT2_F_SET       : 1;
__REG32 ICAP2_F_SET       : 1;
__REG32                   : 4;
__REG32 ABORT_F_SET       : 1;
__REG32                   :16;
} __mcintf_set_bits;

/* MCPWM Interrupt Flags clear */
typedef struct{
__REG32 ILIM0_F_CLR       : 1;
__REG32 IMAT0_F_CLR       : 1;
__REG32 ICAP0_F_CLR       : 1;
__REG32                   : 1;
__REG32 ILIM1_F_CLR       : 1;
__REG32 IMAT1_F_CLR       : 1;
__REG32 ICAP1_F_CLR       : 1;
__REG32                   : 1;
__REG32 ILIM2_F_CLR       : 1;
__REG32 IMAT2_F_CLR       : 1;
__REG32 ICAP2_F_CLR       : 1;
__REG32                   : 4;
__REG32 ABORT_F_CLR       : 1;
__REG32                   :16;
} __mcintf_clr_bits;

/* Count control register */
typedef struct{
__REG32 TC0MCI0_RE  : 1;
__REG32 TC0MCI0_FE  : 1;
__REG32 TC0MCI1_RE  : 1;
__REG32 TC0MCI1_FE  : 1;
__REG32 TC0MCI2_RE  : 1;
__REG32 TC0MCI2_FE  : 1;
__REG32 TC1MCI0_RE  : 1;
__REG32 TC1MCI0_FE  : 1;
__REG32 TC1MCI1_RE  : 1;
__REG32 TC1MCI1_FE  : 1;
__REG32 TC1MCI2_RE  : 1;
__REG32 TC1MCI2_FE  : 1;
__REG32 TC2MCI0_RE  : 1;
__REG32 TC2MCI0_FE  : 1;
__REG32 TC2MCI1_RE  : 1;
__REG32 TC2MCI1_FE  : 1;
__REG32 TC2MCI2_RE  : 1;
__REG32 TC2MCI2_FE  : 1;
__REG32             :11;
__REG32 CNTR0       : 1;
__REG32 CNTR1       : 1;
__REG32 CNTR2       : 1;
} __mccntcon_bits;

/* Count control register set */
typedef struct{
__REG32 TC0MCI0_RE_SET  : 1;
__REG32 TC0MCI0_FE_SET  : 1;
__REG32 TC0MCI1_RE_SET  : 1;
__REG32 TC0MCI1_FE_SET  : 1;
__REG32 TC0MCI2_RE_SET  : 1;
__REG32 TC0MCI2_FE_SET  : 1;
__REG32 TC1MCI0_RE_SET  : 1;
__REG32 TC1MCI0_FE_SET  : 1;
__REG32 TC1MCI1_RE_SET  : 1;
__REG32 TC1MCI1_FE_SET  : 1;
__REG32 TC1MCI2_RE_SET  : 1;
__REG32 TC1MCI2_FE_SET  : 1;
__REG32 TC2MCI0_RE_SET  : 1;
__REG32 TC2MCI0_FE_SET  : 1;
__REG32 TC2MCI1_RE_SET  : 1;
__REG32 TC2MCI1_FE_SET  : 1;
__REG32 TC2MCI2_RE_SET  : 1;
__REG32 TC2MCI2_FE_SET  : 1;
__REG32                 :11;
__REG32 CNTR0_SET       : 1;
__REG32 CNTR1_SET       : 1;
__REG32 CNTR2_SET       : 1;
} __mccntcon_set_bits;

/* Count control register */
typedef struct{
__REG32 TC0MCI0_RE_CLR  : 1;
__REG32 TC0MCI0_FE_CLR  : 1;
__REG32 TC0MCI1_RE_CLR  : 1;
__REG32 TC0MCI1_FE_CLR  : 1;
__REG32 TC0MCI2_RE_CLR  : 1;
__REG32 TC0MCI2_FE_CLR  : 1;
__REG32 TC1MCI0_RE_CLR  : 1;
__REG32 TC1MCI0_FE_CLR  : 1;
__REG32 TC1MCI1_RE_CLR  : 1;
__REG32 TC1MCI1_FE_CLR  : 1;
__REG32 TC1MCI2_RE_CLR  : 1;
__REG32 TC1MCI2_FE_CLR  : 1;
__REG32 TC2MCI0_RE_CLR  : 1;
__REG32 TC2MCI0_FE_CLR  : 1;
__REG32 TC2MCI1_RE_CLR  : 1;
__REG32 TC2MCI1_FE_CLR  : 1;
__REG32 TC2MCI2_RE_CLR  : 1;
__REG32 TC2MCI2_FE_CLR  : 1;
__REG32                 :11;
__REG32 CNTR0_CLR       : 1;
__REG32 CNTR1_CLR       : 1;
__REG32 CNTR2_CLR       : 1;
} __mccntcon_clr_bits;

/* Dead-time register */
typedef struct{
__REG32 DT0         :10;
__REG32 DT1         :10;
__REG32 DT2         :10;
__REG32             : 2;
} __mcdt_bits;

/* Current communication pattern register */
typedef struct{
__REG32 CCPA0       : 1;
__REG32 CCPB0       : 1;
__REG32 CCPA1       : 1;
__REG32 CCPB1       : 1;
__REG32 CCPA2       : 1;
__REG32 CCPB2       : 1;
__REG32             :26;
} __mcccp_bits;

/* Capture clear register */
typedef struct{
__REG32 CAP_CLR0    : 1;
__REG32 CAP_CLR1    : 1;
__REG32 CAP_CLR2    : 1;
__REG32             :29;
} __mccap_clr_bits;

/* QEI Control register */
typedef struct{
__REG32 RESP        : 1;
__REG32 RESPI       : 1;
__REG32 RESV        : 1;
__REG32 RESI        : 1;
__REG32             :28;
} __qeicon_bits;

/* QEI Configuration register */
typedef struct{
__REG32 DIRINV      : 1;
__REG32 SIGMODE     : 1;
__REG32 CAPMODE     : 1;
__REG32 INVINX      : 1;
__REG32 CRESPI      : 1;
__REG32             :11;
__REG32 INXGATE     : 4;
__REG32             :12;
} __qeiconf_bits;

/* QEI Status register */
typedef struct{
__REG32 DIR         : 1;
__REG32             :31;
} __qeistat_bits;

/* QEI Interrupt Set register */
/* QEI Interrupt Clear register */
typedef struct{
__REG32 INX_EN      : 1;
__REG32 TIM_EN      : 1;
__REG32 VELC_EN     : 1;
__REG32 DIR_EN      : 1;
__REG32 ERR_EN      : 1;
__REG32 ENCLK_EN    : 1;
__REG32 POS0_INT    : 1;
__REG32 POS1_INT    : 1;
__REG32 POS2_INT    : 1;
__REG32 REV0_INT    : 1;
__REG32 POS0REV_INT : 1;
__REG32 POS1REV_INT : 1;
__REG32 POS2REV_INT : 1;
__REG32 REV1_INT    : 1;
__REG32 REV2_INT    : 1;
__REG32 MAXPOS_INT  : 1;
__REG32             :16;
} __qeiiec_bits;

/* QEI Interrupt Status register */
typedef struct{
__REG32 INX_INT     : 1;
__REG32 TIM_INT     : 1;
__REG32 VELC_INT    : 1;
__REG32 DIR_INT     : 1;
__REG32 ERR_INT     : 1;
__REG32 ENCLK_INT   : 1;
__REG32 POS0_INT    : 1;
__REG32 POS1_INT    : 1;
__REG32 POS2_INT    : 1;
__REG32 REV0_INT    : 1;
__REG32 POS0REV_INT : 1;
__REG32 POS1REV_INT : 1;
__REG32 POS2REV_INT : 1;
__REG32 REV1_INT    : 1;
__REG32 REV2_INT    : 1;
__REG32 MAXPOS_INT  : 1;
__REG32             :16;
} __qeiintstat_bits;

/* RI Control register */
typedef struct{
__REG32 RITINT   : 1;
__REG32 RITENCLR : 1;
__REG32 RITENBR  : 1;
__REG32 RITEN    : 1;
__REG32          :28;
} __rictrl_bits;

/* Interrupt clear enable register */
typedef struct{
__REG32 CLR_EN   : 1;
__REG32          :31;
} __atclr_en_bits;

/* Interrupt set enable register */
typedef struct{
__REG32 SET_EN   : 1;
__REG32          :31;
} __atset_en_bits;

/* Interrupt status register */
typedef struct{
__REG32 STAT     : 1;
__REG32          :31;
} __atstatus_bits;

/* Interrupt enable register */
typedef struct{
__REG32 EN       : 1;
__REG32          :31;
} __atenable_bits;

/* Clear status register */
typedef struct{
__REG32 CSTAT    : 1;
__REG32          :31;
} __atclr_stat_bits;

/* Set status register */
typedef struct{
__REG32 SSTAT    : 1;
__REG32          :31;
} __atset_stat_bits;

/* Watchdog mode register */
typedef struct{
__REG32 WDEN     : 1;
__REG32 WDRESET  : 1;
__REG32 WDTOF    : 1;
__REG32 WDINT    : 1;
__REG32 WDPROTECT: 1;
__REG32          :27;
} __wdmod_bits;

/* Watchdog Timer Constant register */
typedef struct{
__REG32 Count    :24;
__REG32          : 8;
} __wdtc_bits;

/* Watchdog feed register */
typedef struct{
__REG32 Feed  : 8;
__REG32       :24;
} __wdfeed_bits;

/* Watchdog Timer Warning Interrupt register */
typedef struct{
__REG32 WDWARNINT :10;
__REG32           :22;
} __wdwarnint_bits;

/* Watchdog Timer Window register */
typedef struct{
__REG32 WDWINDOW  :24;
__REG32           : 8;
} __wdwindow_bits;

/* RTC interrupt location register */
typedef struct{
__REG32 RTCCIF  : 1;
__REG32 RTCALF  : 1;
__REG32         :30;
} __ilr_bits;

/* RTC clock control register */
typedef struct{
__REG32 CLKEN   : 1;
__REG32 CTCRST  : 1;
__REG32         : 2;
__REG32 CCALEN  : 1;
__REG32         :27;
} __rtcccr_bits;

/* RTC counter increment interrupt register */
typedef struct{
__REG32 IMSEC   : 1;
__REG32 IMMIN   : 1;
__REG32 IMHOUR  : 1;
__REG32 IMDOM   : 1;
__REG32 IMDOW   : 1;
__REG32 IMDOY   : 1;
__REG32 IMMON   : 1;
__REG32 IMYEAR  : 1;
__REG32         :24;
} __ciir_bits;

/* RTC Counter Increment Select Mask Register */
typedef struct{
__REG32 CALVAL    :17;
__REG32 CALDIR    : 1;
__REG32           :14;
} __calibration_bits;

/* RTC alarm mask register */
typedef struct{
__REG32 AMRSEC   : 1;
__REG32 AMRMIN   : 1;
__REG32 AMRHOUR  : 1;
__REG32 AMRDOM   : 1;
__REG32 AMRDOW   : 1;
__REG32 AMRDOY   : 1;
__REG32 AMRMON   : 1;
__REG32 AMRYEAR  : 1;
__REG32          :24;
} __amr_bits;

/* RTC consolidated time register 0 */
typedef struct{
__REG32 SEC   : 6;
__REG32       : 2;
__REG32 MIN   : 6;
__REG32       : 2;
__REG32 HOUR  : 5;
__REG32       : 3;
__REG32 DOW   : 3;
__REG32       : 5;
} __ctime0_bits;

/* RTC consolidated time register 1 */
typedef struct{
__REG32 DOM   : 5;
__REG32       : 3;
__REG32 MON   : 4;
__REG32       : 4;
__REG32 YEAR  :12;
__REG32       : 4;
} __ctime1_bits;

/* RTC consolidated time register 2 */
typedef struct{
__REG32 DOY  :12;
__REG32      :20;
} __ctime2_bits;

/* RTC second register */
typedef struct{
__REG32 SECONDS : 6;
__REG32         :26;
} __sec_bits;

/* RTC minute register */
typedef struct{
__REG32 MINUTES : 6;
__REG32         :26;
} __min_bits;

/* RTC hour register */
typedef struct{
__REG32 HOURS : 5;
__REG32       :27;
} __hour_bits;

/* RTC day of month register */
typedef struct{
__REG32 DOM  : 5;
__REG32      :27;
} __dom_bits;

/* RTC day of week register */
typedef struct{
__REG32 DOW  : 3;
__REG32      :29;
} __dow_bits;

/* RTC day of year register */
typedef struct{
__REG32 DOY  : 9;
__REG32      :23;
} __doy_bits;

/* RTC month register */
typedef struct{
__REG32 MONTH : 4;
__REG32       :28;
} __month_bits;

/* RTC year register */
typedef struct{
__REG32 YEAR :12;
__REG32      :20;
} __year_bits;

/* UART Receive Buffer Register (RBR) */
/* UART Transmit Holding Register (THR) */
/* UART Divisor Latch Register  Low (DLL) */
typedef union {
  /*UxRBR*/
  struct {
    __REG32 RBR           : 8;
    __REG32               :24;
  } ;
  /*UxTHR*/
  struct {
    __REG32 THR           : 8;
    __REG32               :24;
  } ;
  /*UxDLL*/
  struct {
    __REG32 DLL           : 8;
    __REG32               :24;
  } ;
} __uartrbr_bits;

/* UART interrupt enable register */
typedef struct{
__REG32 RBRIE     : 1;
__REG32 THREIE    : 1;
__REG32 RXIE      : 1;
__REG32           : 5;
__REG32 ABEOINTEN : 1;
__REG32 ABTOINTEN : 1;
__REG32           :22;
} __uartier0_bits;

/* UART1 interrupt enable register */
typedef struct{
__REG32 RDAIE     : 1;
__REG32 THREIE    : 1;
__REG32 RXLSIE    : 1;
__REG32 RXMSIE    : 1;
__REG32           : 3;
__REG32 CTSIE     : 1;
__REG32 ABEOINTEN : 1;
__REG32 ABTOINTEN : 1;
__REG32           :22;
} __uartier1_bits;

/* UART Transmit Enable Register */
typedef struct{
  __REG32 TXEN            : 1;
  __REG32                 :31;
} __uartter_bits;

/* UART1 Transmit Enable Register */
typedef struct{
  __REG32                 : 7;
  __REG32 TXEN            : 1;
  __REG32                 :24;
} __uart1ter_bits;

/* UART Line Status Register (LSR) */
typedef struct {
__REG32 DR              : 1;
__REG32 OE              : 1;
__REG32 PE              : 1;
__REG32 FE              : 1;
__REG32 BI              : 1;
__REG32 THRE            : 1;
__REG32 TEMT            : 1;
__REG32 RXFE            : 1;
__REG32 TXERR           : 1;
__REG32                 :23;
} __uartlsr_bits;

/* UART1 Line Status Register (LSR) */
typedef struct {
__REG32 RDR             : 1;
__REG32 OE              : 1;
__REG32 PE              : 1;
__REG32 FE              : 1;
__REG32 BI              : 1;
__REG32 THRE            : 1;
__REG32 TEMT            : 1;
__REG32 RXFE            : 1;
__REG32 TXERR           : 1;
__REG32                 :23;
} __uart1lsr_bits;

/* UART Line Control Register (LCR) */
typedef struct {
__REG32 WLS   : 2;
__REG32 SBS   : 1;
__REG32 PE    : 1;
__REG32 PS    : 2;
__REG32 BC    : 1;
__REG32 DLAB  : 1;
__REG32       :24;
} __uartlcr_bits;

/* UART interrupt identification register and fifo control register */
typedef union {
  /*UxIIR*/
  struct {
__REG32 INTSTATUS : 1;
__REG32 INTID     : 3;
__REG32           : 2;
__REG32 FIFOENABLE: 2;
__REG32 ABEOINT   : 1;
__REG32 ABTOINT   : 1;
__REG32           :22;
  };
  /*UxFCR*/
  struct {
__REG32 FIFOEN      : 1;
__REG32 RXFIFORES   : 1;
__REG32 TXFIFORES   : 1;
__REG32 DMAMODE     : 1;
__REG32             : 2;
__REG32 RXTRIGLVL   : 2;
__REG32             :24;
  };
} __uartfcriir_bits;

/* UART modem control register */
typedef struct{
__REG32 DTR   : 1;
__REG32 RTS   : 1;
__REG32       : 2;
__REG32 LMS   : 1;
__REG32       : 1;
__REG32 RTSEN : 1;
__REG32 CTSEN : 1;
__REG32       :24;
} __uartmcr_bits;

/* UART modem status register */
typedef struct {
__REG32 DCTS  : 1;
__REG32 DDSR  : 1;
__REG32 TERI  : 1;
__REG32 DDCD  : 1;
__REG32 CTS   : 1;
__REG32 DSR   : 1;
__REG32 RI    : 1;
__REG32 DCD   : 1;
__REG32       :24;
} __uartmsr_bits;

/* UART Auto-baud Control Register */
typedef struct{
__REG32 START        : 1;
__REG32 MODE         : 1;
__REG32 AUTORESTART  : 1;
__REG32              : 5;
__REG32 ABEOINTCLR   : 1;
__REG32 ABTOINTCLR   : 1;
__REG32              :22;
} __uartacr_bits;

/* IrDA Control Register */
typedef struct{
__REG32 IRDAEN       : 1;
__REG32 IRDAINV      : 1;
__REG32 FIXPULSEEN   : 1;
__REG32 PULSEDIV     : 3;
__REG32              :26;
} __uarticr_bits;

/* UART Fractional Divider Register */
typedef struct{
__REG32 DIVADDVAL  : 4;
__REG32 MULVAL     : 4;
__REG32            :24;
} __uartfdr_bits;

/* UART RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32            : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uars485ctrl_bits;

/* UART1 RS485 Control register */
typedef struct{
__REG32 NMMEN      : 1;
__REG32 RXDIS      : 1;
__REG32 AADEN      : 1;
__REG32 SEL        : 1;
__REG32 DCTRL      : 1;
__REG32 OINV       : 1;
__REG32            :26;
} __uars1485ctrl_bits;

/* UART Half-duplex enable register */
typedef struct{
__REG32 HDEN       : 1;
__REG32            :31;
} __uarthden_bits;

/* UART Smart card interface control register */
typedef struct{
__REG32 SCIEN      : 1;
__REG32 NACKDIS    : 1;
__REG32 PROTSEL    : 1;
__REG32            : 2;
__REG32 TXRETRY    : 3;
__REG32 GUARDTIME  : 8;
__REG32            :16;
} __uartscictrl_bits;

/* UART Smart card interface control register */
typedef struct{
__REG32 SYNC              : 1;
__REG32 CSRC              : 1;
__REG32 FES               : 1;
__REG32 TSBYPASS          : 1;
__REG32 CSCEN             : 1;
__REG32 SSSDIS            : 1;
__REG32 CCCLR             : 1;
__REG32                   :25;
} __uartsyncctrl_bits;

/* UART FIFO Level register */
typedef struct{
__REG32 RXFIFILVL  : 4;
__REG32            : 4;
__REG32 TXFIFOLVL  : 4;
__REG32            :20;
} __uartfifolvl_bits;

/* SSP Control Register 0 */
typedef struct{
__REG32 DSS  : 4;
__REG32 FRF  : 2;
__REG32 CPOL : 1;
__REG32 CPHA : 1;
__REG32 SCR  : 8;
__REG32      :16;
} __sspcr0_bits;

/* SSP Control Register 1 */
typedef struct{
__REG32 LBM  : 1;
__REG32 SSE  : 1;
__REG32 MS   : 1;
__REG32 SOD  : 1;
__REG32      :28;
} __sspcr1_bits;

/* SSP Data Register */
typedef struct{
__REG32 DATA :16;
__REG32      :16;
} __sspdr_bits;

/* SSP Status Register */
typedef struct{
__REG32 TFE  : 1;
__REG32 TNF  : 1;
__REG32 RNE  : 1;
__REG32 RFF  : 1;
__REG32 BSY  : 1;
__REG32      :27;
} __sspsr_bits;

/* SSP Clock Prescale Register */
typedef struct{
__REG32 CPSDVSR : 8;
__REG32         :24;
} __sspcpsr_bits;

/* SSP Interrupt Mask Set/Clear Register */
typedef struct{
__REG32 RORIM  : 1;
__REG32 RTIM   : 1;
__REG32 RXIM   : 1;
__REG32 TXIM   : 1;
__REG32        :28;
} __sspimsc_bits;

/* SSP Raw Interrupt Status Register */
typedef struct{
__REG32 RORRIS  : 1;
__REG32 RTRIS   : 1;
__REG32 RXRIS   : 1;
__REG32 TXRIS   : 1;
__REG32         :28;
} __sspris_bits;

/* SSP Masked Interrupt Status Register */
typedef struct{
__REG32 RORMIS  : 1;
__REG32 RTMIS   : 1;
__REG32 RXMIS   : 1;
__REG32 TXMIS   : 1;
__REG32         :28;
} __sspmis_bits;

/* SSP Interrupt Clear Register */
typedef struct{
__REG32 RORIC  : 1;
__REG32 RTIC   : 1;
__REG32        :30;
} __sspicr_bits;

/* SSP DMA Control Register */
typedef struct{
__REG32 RXDMAE : 1;
__REG32 TXDMAE : 1;
__REG32        :30;
} __sspdmacr_bits;

/* CAN control register */
typedef struct{
__REG32 INIT      : 1;
__REG32 IE        : 1;
__REG32 SIE       : 1;
__REG32 EIE       : 1;
__REG32           : 1;
__REG32 DAR       : 1;
__REG32 CCE       : 1;
__REG32 TEST      : 1;
__REG32           :24;
} __cancntl_bits;

/* CAN status register */
typedef struct{
__REG32 LEC       : 3;
__REG32 TXOK      : 1;
__REG32 RXOK      : 1;
__REG32 EPASS     : 1;
__REG32 EWARN     : 1;
__REG32 BOFF      : 1;
__REG32           :24;
} __canstat_bits;

/* CAN error counter */
typedef struct{
__REG32 TEC       : 8;
__REG32 REC       : 7;
__REG32 RP        : 1;
__REG32           :16;
} __canec_bits;

/* CAN bit timing register */
typedef struct{
__REG32 BRP       : 6;
__REG32 SJW       : 2;
__REG32 TSEG1     : 4;
__REG32 TSEG2     : 3;
__REG32           :17;
} __canbt_bits;

/* CAN interrupt register */
typedef struct{
__REG32 INTID     :16;
__REG32           :16;
} __canint_bits;

/* CAN test register */
typedef struct{
__REG32           : 2;
__REG32 BASIC     : 1;
__REG32 SILENT    : 1;
__REG32 LBACK     : 1;
__REG32 TX        : 2;
__REG32 RX        : 1;
__REG32           :24;
} __cantest_bits;

/* CAN baud rate prescaler extension register */
typedef struct{
__REG32 BRPE      : 4;
__REG32           :28;
} __canbrpe_bits;

/* CAN message interface command request registers */
typedef struct{
__REG32 Message_Number : 6;
__REG32                : 9;
__REG32 BUSY           : 1;
__REG32                :16;
} __canifx_cmdreq_bits;

/*CAN message interface command mask registers */
typedef struct{
__REG32 DATA_B         : 1;
__REG32 DATA_A         : 1;
__REG32 TXRQST_NEWDAT  : 1;
__REG32 CLRINTPND      : 1;
__REG32 CTRL           : 1;
__REG32 ARB            : 1;
__REG32 MASK           : 1;
__REG32 WR_RD          : 1;
__REG32                :24;
} __canifx_cmdmsk_bits;

/* CAN message interface command mask 1 registers */
typedef struct{
__REG32 MSK0           : 1;
__REG32 MSK1           : 1;
__REG32 MSK2           : 1;
__REG32 MSK3           : 1;
__REG32 MSK4           : 1;
__REG32 MSK5           : 1;
__REG32 MSK6           : 1;
__REG32 MSK7           : 1;
__REG32 MSK8           : 1;
__REG32 MSK9           : 1;
__REG32 MSK10          : 1;
__REG32 MSK11          : 1;
__REG32 MSK12          : 1;
__REG32 MSK13          : 1;
__REG32 MSK14          : 1;
__REG32 MSK15          : 1;
__REG32                :16;
} __canifx_msk1_bits;

/* CAN message interface command mask 1 registers */
typedef struct{
__REG32 MSK16          : 1;
__REG32 MSK17          : 1;
__REG32 MSK18          : 1;
__REG32 MSK19          : 1;
__REG32 MSK20          : 1;
__REG32 MSK21          : 1;
__REG32 MSK22          : 1;
__REG32 MSK23          : 1;
__REG32 MSK24          : 1;
__REG32 MSK25          : 1;
__REG32 MSK26          : 1;
__REG32 MSK27          : 1;
__REG32 MSK28          : 1;
__REG32                : 1;
__REG32 MDIR           : 1;
__REG32 MXTD           : 1;
__REG32                :16;
} __canifx_msk2_bits;

/* CAN message interface command arbitration 1 registers */
typedef struct{
__REG32  ID             :16;
__REG32                 :16;
} __canifx_arb1_bits;

/* CAN message interface command arbitration 2 registers */
typedef struct{
__REG32  ID             :13;
__REG32  DIR            : 1;
__REG32  XTD            : 1;
__REG32  MSGVAL         : 1;
__REG32                 :16;
} __canifx_arb2_bits;

/* CAN message interface message control registers */
typedef struct{
__REG32  DLC            : 4;
__REG32                 : 3;
__REG32  EOB            : 1;
__REG32  TXRQST         : 1;
__REG32  RMTEN          : 1;
__REG32  RXIE           : 1;
__REG32  TXIE           : 1;
__REG32  UMASK          : 1;
__REG32  INTPND         : 1;
__REG32  MSGLST         : 1;
__REG32  NEWDAT         : 1;
__REG32                 :16;
} __canifx_mctrl_bits;

/* CAN message interface data A1 registers */
typedef struct{
__REG32  DATA0          : 8;
__REG32  DATA1          : 8;
__REG32                 :16;
} __canifx_da1_bits;

/* CAN message interface data A2 registers */
typedef struct{
__REG32  DATA2          : 8;
__REG32  DATA3          : 8;
__REG32                 :16;
} __canifx_da2_bits;

/* CAN message interface data B1 registers */
typedef struct{
__REG32  DATA4          : 8;
__REG32  DATA5          : 8;
__REG32                 :16;
} __canifx_db1_bits;

/* CAN message interface data B2 registers */
typedef struct{
__REG32  DATA6          : 8;
__REG32  DATA7          : 8;
__REG32                 :16;
} __canifx_db2_bits;

/* CAN transmission request 1 register */
typedef struct{
__REG32  TXRQST1        : 1;
__REG32  TXRQST2        : 1;
__REG32  TXRQST3        : 1;
__REG32  TXRQST4        : 1;
__REG32  TXRQST5        : 1;
__REG32  TXRQST6        : 1;
__REG32  TXRQST7        : 1;
__REG32  TXRQST8        : 1;
__REG32  TXRQST9        : 1;
__REG32  TXRQST10       : 1;
__REG32  TXRQST11       : 1;
__REG32  TXRQST12       : 1;
__REG32  TXRQST13       : 1;
__REG32  TXRQST14       : 1;
__REG32  TXRQST15       : 1;
__REG32  TXRQST16       : 1;
__REG32                 :16;
} __cantxreq1_bits;

/* CAN transmission request 2 register */
typedef struct{
__REG32  TXRQST17       : 1;
__REG32  TXRQST18       : 1;
__REG32  TXRQST19       : 1;
__REG32  TXRQST20       : 1;
__REG32  TXRQST21       : 1;
__REG32  TXRQST22       : 1;
__REG32  TXRQST23       : 1;
__REG32  TXRQST24       : 1;
__REG32  TXRQST25       : 1;
__REG32  TXRQST26       : 1;
__REG32  TXRQST27       : 1;
__REG32  TXRQST28       : 1;
__REG32  TXRQST29       : 1;
__REG32  TXRQST30       : 1;
__REG32  TXRQST31       : 1;
__REG32  TXRQST32       : 1;
__REG32                 :16;
} __cantxreq2_bits;

/* CAN new data 1 register */
typedef struct{
__REG32  NEWDAT1        : 1;
__REG32  NEWDAT2        : 1;
__REG32  NEWDAT3        : 1;
__REG32  NEWDAT4        : 1;
__REG32  NEWDAT5        : 1;
__REG32  NEWDAT6        : 1;
__REG32  NEWDAT7        : 1;
__REG32  NEWDAT8        : 1;
__REG32  NEWDAT9        : 1;
__REG32  NEWDAT10       : 1;
__REG32  NEWDAT11       : 1;
__REG32  NEWDAT12       : 1;
__REG32  NEWDAT13       : 1;
__REG32  NEWDAT14       : 1;
__REG32  NEWDAT15       : 1;
__REG32  NEWDAT16       : 1;
__REG32                 :16;
} __cannd1_bits;

/* CAN new data 2 register */
typedef struct{
__REG32  NEWDAT17       : 1;
__REG32  NEWDAT18       : 1;
__REG32  NEWDAT19       : 1;
__REG32  NEWDAT20       : 1;
__REG32  NEWDAT21       : 1;
__REG32  NEWDAT22       : 1;
__REG32  NEWDAT23       : 1;
__REG32  NEWDAT24       : 1;
__REG32  NEWDAT25       : 1;
__REG32  NEWDAT26       : 1;
__REG32  NEWDAT27       : 1;
__REG32  NEWDAT28       : 1;
__REG32  NEWDAT29       : 1;
__REG32  NEWDAT30       : 1;
__REG32  NEWDAT31       : 1;
__REG32  NEWDAT32       : 1;
__REG32                 :16;
} __cannd2_bits;

/* CAN interrupt pending 1 register */
typedef struct{
__REG32  INTPND1        : 1;
__REG32  INTPND2        : 1;
__REG32  INTPND3        : 1;
__REG32  INTPND4        : 1;
__REG32  INTPND5        : 1;
__REG32  INTPND6        : 1;
__REG32  INTPND7        : 1;
__REG32  INTPND8        : 1;
__REG32  INTPND9        : 1;
__REG32  INTPND10       : 1;
__REG32  INTPND11       : 1;
__REG32  INTPND12       : 1;
__REG32  INTPND13       : 1;
__REG32  INTPND14       : 1;
__REG32  INTPND15       : 1;
__REG32  INTPND16       : 1;
__REG32                 :16;
} __canir1_bits;

/* CAN interrupt pending 2 register */
typedef struct{
__REG32  INTPND17       : 1;
__REG32  INTPND18       : 1;
__REG32  INTPND19       : 1;
__REG32  INTPND20       : 1;
__REG32  INTPND21       : 1;
__REG32  INTPND22       : 1;
__REG32  INTPND23       : 1;
__REG32  INTPND24       : 1;
__REG32  INTPND25       : 1;
__REG32  INTPND26       : 1;
__REG32  INTPND27       : 1;
__REG32  INTPND28       : 1;
__REG32  INTPND29       : 1;
__REG32  INTPND30       : 1;
__REG32  INTPND31       : 1;
__REG32  INTPND32       : 1;
__REG32                 :16;
} __canir2_bits;

/* CAN message valid 1 register */
typedef struct{
__REG32  MSGVAL1        : 1;
__REG32  MSGVAL2        : 1;
__REG32  MSGVAL3        : 1;
__REG32  MSGVAL4        : 1;
__REG32  MSGVAL5        : 1;
__REG32  MSGVAL6        : 1;
__REG32  MSGVAL7        : 1;
__REG32  MSGVAL8        : 1;
__REG32  MSGVAL9        : 1;
__REG32  MSGVAL10       : 1;
__REG32  MSGVAL11       : 1;
__REG32  MSGVAL12       : 1;
__REG32  MSGVAL13       : 1;
__REG32  MSGVAL14       : 1;
__REG32  MSGVAL15       : 1;
__REG32  MSGVAL16       : 1;
__REG32                 :16;
} __canmsgv1_bits;

/* CAN message valid 2 register */
typedef struct{
__REG32  MSGVAL17       : 1;
__REG32  MSGVAL18       : 1;
__REG32  MSGVAL19       : 1;
__REG32  MSGVAL20       : 1;
__REG32  MSGVAL21       : 1;
__REG32  MSGVAL22       : 1;
__REG32  MSGVAL23       : 1;
__REG32  MSGVAL24       : 1;
__REG32  MSGVAL25       : 1;
__REG32  MSGVAL26       : 1;
__REG32  MSGVAL27       : 1;
__REG32  MSGVAL28       : 1;
__REG32  MSGVAL29       : 1;
__REG32  MSGVAL30       : 1;
__REG32  MSGVAL31       : 1;
__REG32  MSGVAL32       : 1;
__REG32                 :16;
} __canmsgv2_bits;

/* CAN clock divider register */
typedef struct{
__REG32  CLKDIVVAL      : 3;
__REG32                 :29;
} __canclkdiv_bits;  

/* I2S Digital Audio Output Registes */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32 MUTE          : 1;
__REG32               :16;
} __i2sdao_bits;

/* I2S Digital Audio Input Register */
typedef struct{
__REG32 WORDWIDTH     : 2;
__REG32 MONO          : 1;
__REG32 STOP          : 1;
__REG32 RESET         : 1;
__REG32 WS_SEL        : 1;
__REG32 WS_HALFPERIOD : 9;
__REG32               :17;
} __i2sdai_bits;

/* I2S Status Feedback Register */
typedef struct{
__REG32 IRQ           : 1;
__REG32 DMAREQ1       : 1;
__REG32 DMAREQ2       : 1;
__REG32               : 5;
__REG32 RX_LEVEL      : 5;
__REG32               : 3;
__REG32 TX_LEVEL      : 5;
__REG32               :11;
} __i2sstate_bits;

/* I2S DMA Configuration Register */
typedef struct{
__REG32 RX_DMA_EN     : 1;
__REG32 TX_DMA_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_DMA  : 5;
__REG32               : 3;
__REG32 TX_DEPTH_DMA  : 5;
__REG32               :11;
} __i2sdma_bits;

/* I2S Interrupt Request Control register */
typedef struct{
__REG32 RX_IRQ_EN     : 1;
__REG32 TX_IRQ_EN     : 1;
__REG32               : 6;
__REG32 RX_DEPTH_IRQ  : 5;
__REG32               : 3;
__REG32 TX_DEPTH_IRQ  : 5;
__REG32               :11;
} __i2sirq_bits;

/* I2S Transmit Clock Rate Register */
typedef struct{
__REG32 Y_DIVIDER     : 8;
__REG32 X_DIVIDER     : 8;
__REG32               :16;
} __i2stxrate_bits;

/* Transmit Clock Rate register */
typedef struct{
__REG32 TX_BITRATE    : 6;
__REG32               :26;
} __i2stxbitrate_bits;

/* Receive Clock Rate register */
typedef struct{
__REG32 RX_BITRATE    : 6;
__REG32               :26;
} __i2srxbitrate_bits;

/* Transmit Mode Control register */
typedef struct{
__REG32 TXCLKSEL      : 2;
__REG32 TX4PIN        : 1;
__REG32 TXMCENA       : 1;
__REG32               :28;
} __i2stxmode_bits;

/* Receive Mode Control register */
typedef struct{
__REG32 RXCLKSEL      : 2;
__REG32 RX4PIN        : 1;
__REG32 RXMCENA       : 1;
__REG32               :28;
} __i2srxmode_bits;

/* I2C control set register */
typedef struct{
__REG32       : 2;
__REG32 AA    : 1;
__REG32 SI    : 1;
__REG32 STO   : 1;
__REG32 STA   : 1;
__REG32 I2EN  : 1;
__REG32       :25;
} __i2conset_bits;

/* I2C control clear register */
typedef struct{
__REG32        : 2;
__REG32 AAC    : 1;
__REG32 SIC    : 1;
__REG32        : 1;
__REG32 STAC   : 1;
__REG32 I2ENC  : 1;
__REG32        :25;
} __i2conclr_bits;

/* I2C status register */
typedef struct{
__REG32         : 2;
__REG32 STATUS  : 6;
__REG32         :24;
} __i2stat_bits;

/* I2C data register */
typedef struct{
__REG32 DATA  : 8;
__REG32       :24;
} __i2dat_bits;

/* I2C Monitor mode control register */
typedef struct{
__REG32 MM_ENA    : 1;
__REG32 ENA_SCL   : 1;
__REG32 MATCH_ALL : 1;
__REG32           :29;
} __i2cmmctrl_bits;

/* I2C slave  register */
typedef struct{
__REG32 GC    : 1;
__REG32 ADDR  : 7;
__REG32       :24;
} __i2adr_bits;

/* I2C Mask registers */
typedef struct{
__REG32       : 1;
__REG32 MASK  : 7;
__REG32       :24;
} __i2cmask_bits;

/* I2C SCL High Duty Cycle register */
typedef struct{
__REG32 SCLH   :16;
__REG32        :16;
} __i2sch_bits;

/* I2C scl duty cycle register */
typedef struct{
__REG32 SCLL   :16;
__REG32        :16;
} __i2scl_bits;

/* A/D Control Register */
typedef struct{
__REG32 SEL     : 8;
__REG32 CLKDIV  : 8;
__REG32 BURST   : 1;
__REG32 CLKS    : 3;
__REG32         : 1;
__REG32 PDN     : 1;
__REG32         : 2;
__REG32 START   : 3;
__REG32 EDGE    : 1;
__REG32         : 4;
} __adcr_bits;

/* A/D Global Data Register */
typedef struct{
__REG32         : 6;
__REG32 RESULT  :10;
__REG32         : 8;
__REG32 CHN     : 3;
__REG32         : 3;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __adgdr_bits;

/* A/D Status Register */
typedef struct{
__REG32 DONE0     : 1;
__REG32 DONE1     : 1;
__REG32 DONE2     : 1;
__REG32 DONE3     : 1;
__REG32 DONE4     : 1;
__REG32 DONE5     : 1;
__REG32 DONE6     : 1;
__REG32 DONE7     : 1;
__REG32 OVERRUN0  : 1;
__REG32 OVERRUN1  : 1;
__REG32 OVERRUN2  : 1;
__REG32 OVERRUN3  : 1;
__REG32 OVERRUN4  : 1;
__REG32 OVERRUN5  : 1;
__REG32 OVERRUN6  : 1;
__REG32 OVERRUN7  : 1;
__REG32 ADINT     : 1;
__REG32           :15;
} __adstat_bits;

/* A/D Intrrupt Enable Register */
typedef struct{
__REG32 ADINTEN0  : 1;
__REG32 ADINTEN1  : 1;
__REG32 ADINTEN2  : 1;
__REG32 ADINTEN3  : 1;
__REG32 ADINTEN4  : 1;
__REG32 ADINTEN5  : 1;
__REG32 ADINTEN6  : 1;
__REG32 ADINTEN7  : 1;
__REG32 ADGINTEN  : 1;
__REG32           :23;
} __adinten_bits;

/* A/D Data Register */
typedef struct{
__REG32         : 6;
__REG32 RESULT  :10;
__REG32         :14;
__REG32 OVERUN  : 1;
__REG32 DONE    : 1;
} __addr_bits;

/* D/A Converter Register */
typedef struct{
__REG32        : 6;
__REG32 VALUE  :10;
__REG32 BIAS   : 1;
__REG32        :15;
} __dacr_bits;

/* D/A Converter Control register */
typedef struct{
__REG32 INT_DMA_REQ : 1;
__REG32 DBLBUF_ENA  : 1;
__REG32 CNT_ENA     : 1;
__REG32 DMA_ENA     : 1;
__REG32             :28;
} __dacctrl_bits;

/* D/A Converter Counter Value register */
typedef struct{
__REG32 VALUE       :16;
__REG32             :16;
} __daccntval_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** NVIC
 **
 ***************************************************************************/
__IO_REG32_BIT(NVIC,                  0xE000E004,__READ       ,__nvic_bits);
__IO_REG32_BIT(SYSTICKCSR,            0xE000E010,__READ_WRITE ,__systickcsr_bits);
#define STCTRL      SYSTICKCSR
#define STCTRL_bit  SYSTICKCSR_bit
__IO_REG32_BIT(SYSTICKRVR,            0xE000E014,__READ_WRITE ,__systickrvr_bits);
#define STRELOAD      SYSTICKRVR
#define STRELOAD_bit  SYSTICKRVR_bit
__IO_REG32_BIT(SYSTICKCVR,            0xE000E018,__READ_WRITE ,__systickcvr_bits);
#define STCURR      SYSTICKCVR
#define STCURR_bit  SYSTICKCVR_bit
__IO_REG32_BIT(SYSTICKCALVR,          0xE000E01C,__READ       ,__systickcalvr_bits);
#define STCALIB      SYSTICKCALVR
#define STCALIB_bit  SYSTICKCALVR_bit
__IO_REG32_BIT(SETENA0,               0xE000E100,__READ_WRITE ,__setena0_bits);
#define ISER0      SETENA0
#define ISER0_bit  SETENA0_bit
__IO_REG32_BIT(CLRENA0,               0xE000E180,__READ_WRITE ,__clrena0_bits);
#define ICER0      CLRENA0
#define ICER0_bit  CLRENA0_bit
__IO_REG32_BIT(SETPEND0,              0xE000E200,__READ_WRITE ,__setpend0_bits);
#define ISPR0      SETPEND0
#define ISPR0_bit  SETPEND0_bit
__IO_REG32_BIT(CLRPEND0,              0xE000E280,__READ_WRITE ,__clrpend0_bits);
#define ICPR0      CLRPEND0
#define ICPR0_bit  CLRPEND0_bit
__IO_REG32_BIT(ACTIVE0,               0xE000E300,__READ       ,__active0_bits);
#define IABR0      ACTIVE0
#define IABR0_bit  ACTIVE0_bit
__IO_REG32_BIT(IP0,                   0xE000E400,__READ_WRITE ,__pri0_bits);
__IO_REG32_BIT(IP1,                   0xE000E404,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(IP2,                   0xE000E408,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(IP3,                   0xE000E40C,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(IP4,                   0xE000E410,__READ_WRITE ,__pri4_bits);
__IO_REG32_BIT(IP5,                   0xE000E414,__READ_WRITE ,__pri5_bits);
__IO_REG32_BIT(IP6,                   0xE000E418,__READ_WRITE ,__pri6_bits);
__IO_REG32_BIT(IP7,                   0xE000E41C,__READ_WRITE ,__pri7_bits);
__IO_REG32_BIT(CPUIDBR,               0xE000ED00,__READ       ,__cpuidbr_bits);
__IO_REG32_BIT(ICSR,                  0xE000ED04,__READ_WRITE ,__icsr_bits);
__IO_REG32_BIT(VTOR,                  0xE000ED08,__READ_WRITE ,__vtor_bits);
__IO_REG32_BIT(AIRCR,                 0xE000ED0C,__READ_WRITE ,__aircr_bits);
__IO_REG32_BIT(SCR,                   0xE000ED10,__READ_WRITE ,__scr_bits);
__IO_REG32_BIT(CCR,                   0xE000ED14,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(SHPR0,                 0xE000ED18,__READ_WRITE ,__pri1_bits);
__IO_REG32_BIT(SHPR1,                 0xE000ED1C,__READ_WRITE ,__pri2_bits);
__IO_REG32_BIT(SHPR2,                 0xE000ED20,__READ_WRITE ,__pri3_bits);
__IO_REG32_BIT(SHCSR,                 0xE000ED24,__READ_WRITE ,__shcsr_bits);
__IO_REG32_BIT(CFSR,                  0xE000ED28,__READ_WRITE ,__cfsr_bits);
__IO_REG32_BIT(HFSR,                  0xE000ED2C,__READ_WRITE ,__hfsr_bits);
__IO_REG32_BIT(DFSR,                  0xE000ED30,__READ_WRITE ,__dfsr_bits);
__IO_REG32(    MMFAR,                 0xE000ED34,__READ_WRITE);
__IO_REG32(    BFAR,                  0xE000ED38,__READ_WRITE);
__IO_REG32_BIT(STIR,                  0xE000EF00,__WRITE      ,__stir_bits);

/***************************************************************************
 **
 ** ER (Event router)
 **
 ***************************************************************************/
__IO_REG32_BIT(ER_HILO,               0x40044000,__READ_WRITE ,__er_hilo_bits);
__IO_REG32_BIT(ER_EDGE,               0x40044004,__READ_WRITE ,__er_edge_bits);
__IO_REG32_BIT(ER_CLR_EN,             0x40044FD8,__WRITE      ,__er_clr_en_bits);
__IO_REG32_BIT(ER_SET_EN,             0x40044FDC,__WRITE      ,__er_set_en_bits);
__IO_REG32_BIT(ER_STATUS,             0x40044FE0,__READ       ,__er_status_bits);
__IO_REG32_BIT(ER_ENABLE,             0x40044FE4,__READ       ,__er_enable_bits);
__IO_REG32_BIT(ER_CLR_STAT,           0x40044FE8,__WRITE      ,__er_clr_stat_bits);
__IO_REG32_BIT(ER_SET_STAT,           0x40044FEC,__WRITE      ,__er_set_stat_bits);

/***************************************************************************
 **
 ** CREG (Configuration registers)
 **
 ***************************************************************************/
__IO_REG32_BIT(IRCTRM,                0x40043000,__READ       ,__irctrm_bits);
__IO_REG32_BIT(CREG0,                 0x40043004,__READ_WRITE ,__creg0_bits);
__IO_REG32_BIT(PMUCON,                0x40043008,__READ_WRITE ,__pmucon_bits);
__IO_REG32(    M3MEMMAP,              0x40043100,__READ_WRITE );
//__IO_REG32_BIT(CREG1,                 0x40043108,__READ       ,__creg1_bits);
//__IO_REG32_BIT(CREG2,                 0x4004310C,__READ       ,__creg2_bits);
//__IO_REG32_BIT(CREG3,                 0x40043110,__READ       ,__creg3_bits);
//__IO_REG32_BIT(CREG4,                 0x40043114,__READ       ,__creg4_bits);
__IO_REG32_BIT(CREG5,                 0x40043118,__READ_WRITE ,__creg5_bits);
__IO_REG32_BIT(DMAMUX,                0x4004311C,__READ_WRITE ,__dmamux_bits);
__IO_REG32_BIT(ETBCFG,                0x40043128,__READ_WRITE ,__etbcfg_bits);
__IO_REG32_BIT(CREG6,                 0x4004312C,__READ_WRITE ,__creg6_bits);
__IO_REG32(    CHIPID,                0x40043200,__READ       );
//__IO_REG32_BIT(LOCKREG,               0x40043F00,__READ       ,__lockreg_bits);

/***************************************************************************
 **
 ** PMC (Power Management Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(PD0_SLEEP0_HW_ENA,     0x40042000,__READ_WRITE ,__pd0_sleep0_hw_ena_bits);
__IO_REG32(    PD0_SLEEP0_MODE,       0x4004201C,__READ_WRITE );

/***************************************************************************
 **
 ** CGU
 **
 ***************************************************************************/
__IO_REG32_BIT(CGU_FREQ_MON,          0x40050014,__READ_WRITE ,__cgu_freq_mon_bits);
__IO_REG32_BIT(CGU_XTAL_OSC_CTRL,     0x40050018,__READ_WRITE ,__cgu_xtal_osc_ctrl_bits);
__IO_REG32_BIT(CGU_PLL0_STAT,         0x4005001C,__READ       ,__cgu_pll0_stat_bits);
__IO_REG32_BIT(CGU_PLL0_CTRL,         0x40050020,__READ_WRITE ,__cgu_pll0_ctrl_bits);
__IO_REG32_BIT(CGU_PLL0_MDIV,         0x40050024,__READ_WRITE ,__cgu_pll0_mdiv_bits);
__IO_REG32_BIT(CGU_PLL0_NP_DIV,       0x40050028,__READ_WRITE ,__cgu_pll0_np_div_bits);
__IO_REG32_BIT(CGU_PLL1_STAT,         0x4005002C,__READ       ,__cgu_pll1_stat_bits);
__IO_REG32_BIT(CGU_PLL1_CTRL,         0x40050030,__READ_WRITE ,__cgu_pll1_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVA_CTRL,        0x40050034,__READ_WRITE ,__cgu_idiva_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVB_CTRL,        0x40050038,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVC_CTRL,        0x4005003C,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVD_CTRL,        0x40050040,__READ_WRITE ,__cgu_idivx_ctrl_bits);
__IO_REG32_BIT(CGU_IDIVE_CTRL,        0x40050044,__READ_WRITE ,__cgu_idive_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_0_CTRL,     0x40050048,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_1_CTRL,     0x4005004C,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_2_CTRL,     0x40050050,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_3_CTRL,     0x40050054,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_4_CTRL,     0x40050058,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_5_CTRL,     0x4005005C,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_7_CTRL,     0x40050064,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_8_CTRL,     0x40050068,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_9_CTRL,     0x4005006C,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_10_CTRL,    0x40050070,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_11_CTRL,    0x40050074,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_13_CTRL,    0x4005007C,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_14_CTRL,    0x40050080,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_15_CTRL,    0x40050084,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_16_CTRL,    0x40050088,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_17_CTRL,    0x4005008C,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_18_CTRL,    0x40050090,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_19_CTRL,    0x40050094,__READ_WRITE ,__cgu_outclk_ctrl_bits);
__IO_REG32_BIT(CGU_OUTCLK_20_CTRL,    0x40050098,__READ_WRITE ,__cgu_outclk_ctrl_bits);

/***************************************************************************
 **
 ** CCU1
 **
 ***************************************************************************/
__IO_REG32_BIT(CCU1_PM,                       0x40051000,__READ_WRITE ,__ccu_pm_bits);
__IO_REG32_BIT(CCU1_BASE_STAT,                0x40051004,__READ       ,__ccu1_base_stat_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_BUS_CFG,         0x40051100,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_BUS_STAT,        0x40051104,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_I2C1_CFG,        0x40051108,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_I2C1_STAT,       0x4005110C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_DAC_CFG,         0x40051110,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_DAC_STAT,        0x40051114,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC0_CFG,        0x40051118,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC0_STAT,       0x4005111C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC1_CFG,        0x40051120,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_ADC1_STAT,       0x40051124,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_CAN_CFG,         0x40051128,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB3_CAN_STAT,        0x4005112C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_BUS_CFG,         0x40051200,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_BUS_STAT,        0x40051204,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_MOTOCON_CFG,     0x40051208,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_MOTOCON_STAT,    0x4005120C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2C0_CFG,        0x40051210,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2C0_STAT,       0x40051214,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2S_CFG,         0x40051218,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_APB1_I2S_STAT,        0x4005121C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPIFI_CFG,            0x40051300,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPIFI_STAT,           0x40051304,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_BUS_CFG,           0x40051400,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_BUS_STAT,          0x40051404,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SPIFI_CFG,         0x40051408,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SPIFI_STAT,        0x4005140C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_GPIO_CFG,          0x40051410,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_GPIO_STAT,         0x40051414,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_WWDT_CFG,          0x40051418,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_WWDT_STAT,         0x4005141C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART0_CFG,         0x40051420,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART0_STAT,        0x40051424,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART1_CFG,         0x40051428,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART1_STAT,        0x4005142C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SSP0_CFG,          0x40051430,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SSP0_STAT,         0x40051434,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER0_CFG,        0x40051438,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER0_STAT,       0x4005143C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER1_CFG,        0x40051440,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER1_STAT,       0x40051444,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SCU_CFG,           0x40051448,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SCU_STAT,          0x4005144C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_CREG_CFG,          0x40051450,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_CREG_STAT,         0x40051454,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_OSTIMER_CFG,       0x40051458,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_OSTIMER_STAT,      0x4005145C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART2_CFG,         0x40051460,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART2_STAT,        0x40051464,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART3_CFG,         0x40051468,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UART3_STAT,        0x4005146C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER2_CFG,        0x40051470,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER2_STAT,       0x40051474,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER3_CFG,        0x40051478,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_TIMER3_STAT,       0x4005147C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SSP1_CFG,          0x40051480,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SSP1_STAT,         0x40051484,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SCI_CFG,           0x40051488,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SCI_STAT,          0x4005148C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_QEI_CFG,           0x40051490,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_QEI_STAT,          0x40051494,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_EMC_CFG,           0x400514B0,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_EMC_STAT,          0x400514B4,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SDIO_CFG,          0x400514B8,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_SDIO_STAT,         0x400514BC,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_DMA_CFG,           0x400514C0,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_DMA_STAT,          0x400514C4,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_M3CORE_CFG,        0x400514C8,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_M3CORE_STAT,       0x400514CC,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_EVENTHANDLER_CFG,  0x400514D8,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_EVENTHANDLER_STAT, 0x400514DC,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_AES_CFG,           0x400514E0,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_AES_STAT,          0x400514E4,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UTIMER_CFG,        0x400514E8,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M3_UTIMER_STAT,       0x400514EC,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M0_SGPIO_CFG,         0x40051518,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_M0_SGPIO_STAT,        0x4005151C,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPI_CFG,              0x40051800,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU1_CLK_SPI_STAT,             0x40051804,__READ       ,__ccu_clk_cfg_bits);

/***************************************************************************
 **
 ** CCU2
 **
 ***************************************************************************/
__IO_REG32_BIT(CCU2_PM,                       0x40052000,__READ_WRITE ,__ccu_pm_bits);
__IO_REG32_BIT(CCU2_BASE_STAT,                0x40052004,__READ       ,__ccu2_base_stat_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART3_CFG,       0x40052200,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART3_STAT,      0x40052204,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART2_CFG,       0x40052300,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART2_STAT,      0x40052304,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART1_CFG,       0x40052400,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART1_STAT,      0x40052404,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART0_CFG,       0x40052500,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_UART0_STAT,      0x40052504,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP1_CFG,        0x40052600,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP1_STAT,       0x40052604,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP0_CFG,        0x40052700,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_APB2_SSP0_STAT,       0x40052704,__READ       ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_SDIO_CFG,             0x40052800,__READ_WRITE ,__ccu_clk_cfg_bits);
__IO_REG32_BIT(CCU2_CLK_SDIO_STAT,            0x40052804,__READ       ,__ccu_clk_cfg_bits);

/***************************************************************************
 **
 ** RGU
 **
 ***************************************************************************/
__IO_REG32_BIT(RGU_RESET_CTRL0,               0x40053100,__WRITE      ,__rgu_reset_ctrl0_bits);
__IO_REG32_BIT(RGU_RESET_CTRL1,               0x40053104,__WRITE      ,__rgu_reset_ctrl1_bits);
__IO_REG32_BIT(RGU_RESET_STATUS0,             0x40053110,__READ_WRITE ,__rgu_reset_status0_bits);
__IO_REG32_BIT(RGU_RESET_STATUS1,             0x40053114,__READ_WRITE ,__rgu_reset_status1_bits);
__IO_REG32_BIT(RGU_RESET_STATUS2,             0x40053118,__READ_WRITE ,__rgu_reset_status2_bits);
__IO_REG32_BIT(RGU_RESET_STATUS3,             0x4005311C,__READ_WRITE ,__rgu_reset_status3_bits);
__IO_REG32_BIT(RGU_RESET_ACTIVE_STATUS0,      0x40053150,__READ       ,__rgu_reset_ctrl0_bits);
__IO_REG32_BIT(RGU_RESET_ACTIVE_STATUS1,      0x40053154,__READ       ,__rgu_reset_ctrl1_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT0,           0x40053400,__READ_WRITE ,__rgu_reset_ext_stat0_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT1,           0x40053404,__READ_WRITE ,__rgu_reset_ext_stat1_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT2,           0x40053408,__READ_WRITE ,__rgu_reset_ext_stat2_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT4,           0x40053410,__READ_WRITE ,__rgu_reset_ext_stat4_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT5,           0x40053414,__READ_WRITE ,__rgu_reset_ext_stat5_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT8,           0x40053420,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT9,           0x40053424,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT13,          0x40053434,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT16,          0x40053440,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT17,          0x40053444,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT18,          0x40053448,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT19,          0x4005344C,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT20,          0x40053450,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT21,          0x40053454,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT22,          0x40053458,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT23,          0x4005345C,__READ_WRITE ,__rgu_master_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT28,          0x40053470,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT32,          0x40053480,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT33,          0x40053484,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT34,          0x40053488,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT35,          0x4005348C,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT36,          0x40053490,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT37,          0x40053494,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT38,          0x40053498,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT39,          0x4005349C,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT40,          0x400534A0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT41,          0x400534A4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT42,          0x400534A8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT44,          0x400534B0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT45,          0x400534B4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT46,          0x400534B8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT47,          0x400534BC,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT48,          0x400534C0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT49,          0x400534C4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT50,          0x400534C8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT51,          0x400534CC,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT52,          0x400534D0,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT53,          0x400534D4,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT54,          0x400534D8,__READ_WRITE ,__rgu_peripheral_reset_bits);
__IO_REG32_BIT(RGU_RESET_EXT_STAT55,          0x400534DC,__READ_WRITE ,__rgu_peripheral_reset_bits);

/***************************************************************************
 **
 ** System control unit
 **
 ***************************************************************************/
__IO_REG32_BIT(SFSP0_0,                       0x40086000,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP0_1,                       0x40086004,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_0,                       0x40086080,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_1,                       0x40086084,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_2,                       0x40086088,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_3,                       0x4008608C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_4,                       0x40086090,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_5,                       0x40086094,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_6,                       0x40086098,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_7,                       0x4008609C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_8,                       0x400860A0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_9,                       0x400860A4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_10,                      0x400860A8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_11,                      0x400860AC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_12,                      0x400860B0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_13,                      0x400860B4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_14,                      0x400860B8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_15,                      0x400860BC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_16,                      0x400860C0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_17,                      0x400860C4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_18,                      0x400860C8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_19,                      0x400860CC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP1_20,                      0x400860D0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_0,                       0x40086100,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_1,                       0x40086104,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_2,                       0x40086108,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_3,                       0x4008610C,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP2_4,                       0x40086110,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP2_5,                       0x40086114,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP2_6,                       0x40086118,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_7,                       0x4008611C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_8,                       0x40086120,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_9,                       0x40086124,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_10,                      0x40086128,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_11,                      0x4008612C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_12,                      0x40086130,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP2_13,                      0x40086134,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_0,                       0x40086180,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_1,                       0x40086184,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_2,                       0x40086188,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_3,                       0x4008618C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_4,                       0x40086190,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_5,                       0x40086194,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_6,                       0x40086198,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_7,                       0x4008619C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP3_8,                       0x400861A0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_0,                       0x40086200,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_1,                       0x40086204,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_2,                       0x40086208,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_3,                       0x4008620C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_4,                       0x40086210,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_5,                       0x40086214,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_6,                       0x40086218,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_7,                       0x4008621C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_8,                       0x40086220,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_9,                       0x40086224,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP4_10,                      0x40086228,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_0,                       0x40086280,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_1,                       0x40086284,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_2,                       0x40086288,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_3,                       0x4008628C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_4,                       0x40086290,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_5,                       0x40086294,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_6,                       0x40086298,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP5_7,                       0x4008629C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_0,                       0x40086300,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_1,                       0x40086304,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_2,                       0x40086308,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_3,                       0x4008630C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_4,                       0x40086310,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_5,                       0x40086314,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_6,                       0x40086318,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_7,                       0x4008631C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_8,                       0x40086320,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_9,                       0x40086324,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_10,                      0x40086328,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_11,                      0x4008632C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP6_12,                      0x40086330,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_0,                       0x40086380,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_1,                       0x40086384,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_2,                       0x40086388,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_3,                       0x4008638C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_4,                       0x40086390,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_5,                       0x40086394,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_6,                       0x40086398,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP7_7,                       0x4008639C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_0,                       0x40086400,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP8_1,                       0x40086404,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP8_2,                       0x40086408,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSP8_3,                       0x4008640C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_4,                       0x40086410,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_5,                       0x40086414,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_6,                       0x40086418,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_7,                       0x4008641C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP8_8,                       0x40086420,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_0,                       0x40086480,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_1,                       0x40086484,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_2,                       0x40086488,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_3,                       0x4008648C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_4,                       0x40086490,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_5,                       0x40086494,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSP9_6,                       0x40086498,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPA_1,                       0x40086504,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSPA_2,                       0x40086508,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSPA_3,                       0x4008650C,__READ_WRITE ,__sfspx_hidrv_bits);
__IO_REG32_BIT(SFSPA_4,                       0x40086510,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_0,                       0x40086580,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_1,                       0x40086584,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_2,                       0x40086588,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_3,                       0x4008658C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_4,                       0x40086590,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_5,                       0x40086594,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPB_6,                       0x40086598,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_0,                       0x40086600,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_1,                       0x40086604,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_2,                       0x40086608,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_3,                       0x4008660C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_4,                       0x40086610,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_5,                       0x40086614,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_6,                       0x40086618,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_7,                       0x4008661C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_8,                       0x40086620,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_9,                       0x40086624,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_10,                      0x40086628,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_11,                      0x4008662C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_12,                      0x40086630,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_13,                      0x40086634,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPC_14,                      0x40086638,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_0,                       0x40086680,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_1,                       0x40086684,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_2,                       0x40086688,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_3,                       0x4008668C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_4,                       0x40086690,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_5,                       0x40086694,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_6,                       0x40086698,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_7,                       0x4008669C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_8,                       0x400866A0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_9,                       0x400866A4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_10,                      0x400866A8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_11,                      0x400866AC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_12,                      0x400866B0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_13,                      0x400866B4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_14,                      0x400866B8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_15,                      0x400866BC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPD_16,                      0x400866C0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_0,                       0x40086700,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_1,                       0x40086704,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_2,                       0x40086708,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_3,                       0x4008670C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_4,                       0x40086710,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_5,                       0x40086714,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_6,                       0x40086718,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_7,                       0x4008671C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_8,                       0x40086720,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_9,                       0x40086724,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_10,                      0x40086728,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_11,                      0x4008672C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_12,                      0x40086730,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_13,                      0x40086734,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_14,                      0x40086738,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPE_15,                      0x4008673C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_0,                       0x40086780,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_1,                       0x40086784,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_2,                       0x40086788,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_3,                       0x4008678C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_4,                       0x40086790,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_5,                       0x40086794,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_6,                       0x40086798,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_7,                       0x4008679C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_8,                       0x400867A0,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_9,                       0x400867A4,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_10,                      0x400867A8,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSPF_11,                      0x400867AC,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSCLK0,                       0x40086C00,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSCLK1,                       0x40086C04,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSCLK2,                       0x40086C08,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSCLK3,                       0x40086C0C,__READ_WRITE ,__sfspx_normdrv_hispd_bits);
__IO_REG32_BIT(SFSI2C0,                       0x40086C84,__READ_WRITE ,__sfsi2c0_bits);
__IO_REG32_BIT(EMCCLKDELAY,                   0x40086D00,__READ_WRITE ,__emcclkdelay_bits);
__IO_REG32_BIT(EMCCTRLDELAY,                  0x40086D04,__READ_WRITE ,__emcctrldelay_bits);
__IO_REG32_BIT(EMCCSDELAY,                    0x40086D08,__READ_WRITE ,__emccsdelay_bits);
__IO_REG32_BIT(EMCDOUTDELAY,                  0x40086D0C,__READ_WRITE ,__emcdoutdelay_bits);
__IO_REG32_BIT(EMCFBCLKDELAY,                 0x40086D10,__READ_WRITE ,__emcfbclkdelay_bits);
__IO_REG32_BIT(EMCADDRDELAY0,                 0x40086D14,__READ_WRITE ,__emcaddrdelay0_bits);
__IO_REG32_BIT(EMCADDRDELAY1,                 0x40086D18,__READ_WRITE ,__emcaddrdelay1_bits);
__IO_REG32_BIT(EMCADDRDELAY2,                 0x40086D1C,__READ_WRITE ,__emcaddrdelay2_bits);
__IO_REG32_BIT(EMCDINDELAY,                   0x40086D24,__READ_WRITE ,__emcdindelay_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(FIO0DIR,         0x400F0000,__READ_WRITE,__fgpio0_bits);
#define FIO0DIR0          FIO0DIR_bit.__byte0
#define FIO0DIR0_bit      FIO0DIR_bit.__byte0_bit
#define FIO0DIR1          FIO0DIR_bit.__byte1
#define FIO0DIR1_bit      FIO0DIR_bit.__byte1_bit
#define FIO0DIRL          FIO0DIR_bit.__shortl
#define FIO0DIRL_bit      FIO0DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO0MASK,        0x400F0010,__READ_WRITE,__fgpio0_bits);
#define FIO0MASK0         FIO0MASK_bit.__byte0
#define FIO0MASK0_bit     FIO0MASK_bit.__byte0_bit
#define FIO0MASK1         FIO0MASK_bit.__byte1
#define FIO0MASK1_bit     FIO0MASK_bit.__byte1_bit
#define FIO0MASKL         FIO0MASK_bit.__shortl
#define FIO0MASKL_bit     FIO0MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO0PIN,         0x400F0014,__READ_WRITE,__fgpio0_bits);
#define FIO0PIN0          FIO0PIN_bit.__byte0
#define FIO0PIN0_bit      FIO0PIN_bit.__byte0_bit
#define FIO0PIN1          FIO0PIN_bit.__byte1
#define FIO0PIN1_bit      FIO0PIN_bit.__byte1_bit
#define FIO0PINL          FIO0PIN_bit.__shortl
#define FIO0PINL_bit      FIO0PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO0SET,         0x400F0018,__READ_WRITE,__fgpio0_bits);
#define FIO0SET0          FIO0SET_bit.__byte0
#define FIO0SET0_bit      FIO0SET_bit.__byte0_bit
#define FIO0SET1          FIO0SET_bit.__byte1
#define FIO0SET1_bit      FIO0SET_bit.__byte1_bit
#define FIO0SETL          FIO0SET_bit.__shortl
#define FIO0SETL_bit      FIO0SET_bit.__shortl_bit
__IO_REG32_BIT(FIO0CLR,         0x400F001C,__WRITE     ,__fgpio0_bits);
#define FIO0CLR0          FIO0CLR_bit.__byte0
#define FIO0CLR0_bit      FIO0CLR_bit.__byte0_bit
#define FIO0CLR1          FIO0CLR_bit.__byte1
#define FIO0CLR1_bit      FIO0CLR_bit.__byte1_bit
#define FIO0CLRL          FIO0CLR_bit.__shortl
#define FIO0CLRL_bit      FIO0CLR_bit.__shortl_bit
__IO_REG32_BIT(FIO1DIR,         0x400F0020,__READ_WRITE,__fgpio1_bits);
#define FIO1DIR0          FIO1DIR_bit.__byte0
#define FIO1DIR0_bit      FIO1DIR_bit.__byte0_bit
#define FIO1DIR1          FIO1DIR_bit.__byte1
#define FIO1DIR1_bit      FIO1DIR_bit.__byte1_bit
#define FIO1DIRL          FIO1DIR_bit.__shortl
#define FIO1DIRL_bit      FIO1DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO1MASK,        0x400F0030,__READ_WRITE,__fgpio1_bits);
#define FIO1MASK0         FIO1MASK_bit.__byte0
#define FIO1MASK0_bit     FIO1MASK_bit.__byte0_bit
#define FIO1MASK1         FIO1MASK_bit.__byte1
#define FIO1MASK1_bit     FIO1MASK_bit.__byte1_bit
#define FIO1MASKL         FIO1MASK_bit.__shortl
#define FIO1MASKL_bit     FIO1MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO1PIN,         0x400F0034,__READ_WRITE,__fgpio1_bits);
#define FIO1PIN0          FIO1PIN_bit.__byte0
#define FIO1PIN0_bit      FIO1PIN_bit.__byte0_bit
#define FIO1PIN1          FIO1PIN_bit.__byte1
#define FIO1PIN1_bit      FIO1PIN_bit.__byte1_bit
#define FIO1PINL          FIO1PIN_bit.__shortl
#define FIO1PINL_bit      FIO1PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO1SET,         0x400F0038,__READ_WRITE,__fgpio1_bits);
#define FIO1SET0          FIO1SET_bit.__byte0
#define FIO1SET0_bit      FIO1SET_bit.__byte0_bit
#define FIO1SET1          FIO1SET_bit.__byte1
#define FIO1SET1_bit      FIO1SET_bit.__byte1_bit
#define FIO1SETL          FIO1SET_bit.__shortl
#define FIO1SETL_bit      FIO1SET_bit.__shortl_bit
__IO_REG32_BIT(FIO1CLR,         0x400F003C,__WRITE     ,__fgpio1_bits);
#define FIO1CLR0          FIO1CLR_bit.__byte0
#define FIO1CLR0_bit      FIO1CLR_bit.__byte0_bit
#define FIO1CLR1          FIO1CLR_bit.__byte1
#define FIO1CLR1_bit      FIO1CLR_bit.__byte1_bit
#define FIO1CLRL          FIO1CLR_bit.__shortl
#define FIO1CLRL_bit      FIO1CLR_bit.__shortl_bit
__IO_REG32_BIT(FIO2DIR,         0x400F0040,__READ_WRITE,__fgpio2_bits);
#define FIO2DIR0          FIO2DIR_bit.__byte0
#define FIO2DIR0_bit      FIO2DIR_bit.__byte0_bit
#define FIO2DIR1          FIO2DIR_bit.__byte1
#define FIO2DIR1_bit      FIO2DIR_bit.__byte1_bit
#define FIO2DIRL          FIO2DIR_bit.__shortl
#define FIO2DIRL_bit      FIO2DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO2MASK,        0x400F0050,__READ_WRITE,__fgpio2_bits);
#define FIO2MASK0         FIO2MASK_bit.__byte0
#define FIO2MASK0_bit     FIO2MASK_bit.__byte0_bit
#define FIO2MASK1         FIO2MASK_bit.__byte1
#define FIO2MASK1_bit     FIO2MASK_bit.__byte1_bit
#define FIO2MASKL         FIO2MASK_bit.__shortl
#define FIO2MASKL_bit     FIO2MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO2PIN,         0x400F0054,__READ_WRITE,__fgpio2_bits);
#define FIO2PIN0          FIO2PIN_bit.__byte0
#define FIO2PIN0_bit      FIO2PIN_bit.__byte0_bit
#define FIO2PIN1          FIO2PIN_bit.__byte1
#define FIO2PIN1_bit      FIO2PIN_bit.__byte1_bit
#define FIO2PINL          FIO2PIN_bit.__shortl
#define FIO2PINL_bit      FIO2PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO2SET,         0x400F0058,__READ_WRITE,__fgpio2_bits);
#define FIO2SET0          FIO2SET_bit.__byte0
#define FIO2SET0_bit      FIO2SET_bit.__byte0_bit
#define FIO2SET1          FIO2SET_bit.__byte1
#define FIO2SET1_bit      FIO2SET_bit.__byte1_bit
#define FIO2SETL          FIO2SET_bit.__shortl
#define FIO2SETL_bit      FIO2SET_bit.__shortl_bit
__IO_REG32_BIT(FIO2CLR,         0x400F005C,__WRITE     ,__fgpio2_bits);
#define FIO2CLR0          FIO2CLR_bit.__byte0
#define FIO2CLR0_bit      FIO2CLR_bit.__byte0_bit
#define FIO2CLR1          FIO2CLR_bit.__byte1
#define FIO2CLR1_bit      FIO2CLR_bit.__byte1_bit
#define FIO2CLRL          FIO2CLR_bit.__shortl
#define FIO2CLRL_bit      FIO2CLR_bit.__shortl_bit
__IO_REG32_BIT(FIO3DIR,         0x400F0060,__READ_WRITE,__fgpio3_bits);
#define FIO3DIR0          FIO3DIR_bit.__byte0
#define FIO3DIR0_bit      FIO3DIR_bit.__byte0_bit
#define FIO3DIR1          FIO3DIR_bit.__byte1
#define FIO3DIR1_bit      FIO3DIR_bit.__byte1_bit
#define FIO3DIRL          FIO3DIR_bit.__shortl
#define FIO3DIRL_bit      FIO3DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO3MASK,        0x400F0070,__READ_WRITE,__fgpio3_bits);
#define FIO3MASK0         FIO3MASK_bit.__byte0
#define FIO3MASK0_bit     FIO3MASK_bit.__byte0_bit
#define FIO3MASK1         FIO3MASK_bit.__byte1
#define FIO3MASK1_bit     FIO3MASK_bit.__byte1_bit
#define FIO3MASKL         FIO3MASK_bit.__shortl
#define FIO3MASKL_bit     FIO3MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO3PIN,         0x400F0074,__READ_WRITE,__fgpio3_bits);
#define FIO3PIN0          FIO3PIN_bit.__byte0
#define FIO3PIN0_bit      FIO3PIN_bit.__byte0_bit
#define FIO3PIN1          FIO3PIN_bit.__byte1
#define FIO3PIN1_bit      FIO3PIN_bit.__byte1_bit
#define FIO3PINL          FIO3PIN_bit.__shortl
#define FIO3PINL_bit      FIO3PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO3SET,         0x400F0078,__READ_WRITE,__fgpio3_bits);
#define FIO3SET0          FIO3SET_bit.__byte0
#define FIO3SET0_bit      FIO3SET_bit.__byte0_bit
#define FIO3SET1          FIO3SET_bit.__byte1
#define FIO3SET1_bit      FIO3SET_bit.__byte1_bit
#define FIO3SETL          FIO3SET_bit.__shortl
#define FIO3SETL_bit      FIO3SET_bit.__shortl_bit
__IO_REG32_BIT(FIO3CLR,         0x400F007C,__WRITE     ,__fgpio3_bits);
#define FIO3CLR0          FIO3CLR_bit.__byte0
#define FIO3CLR0_bit      FIO3CLR_bit.__byte0_bit
#define FIO3CLR1          FIO3CLR_bit.__byte1
#define FIO3CLR1_bit      FIO3CLR_bit.__byte1_bit
#define FIO3CLRL          FIO3CLR_bit.__shortl
#define FIO3CLRL_bit      FIO3CLR_bit.__shortl_bit
__IO_REG32_BIT(FIO4DIR,         0x400F0080,__READ_WRITE,__fgpio4_bits);
#define FIO4DIR0          FIO4DIR_bit.__byte0
#define FIO4DIR0_bit      FIO4DIR_bit.__byte0_bit
#define FIO4DIR1          FIO4DIR_bit.__byte1
#define FIO4DIR1_bit      FIO4DIR_bit.__byte1_bit
#define FIO4DIRL          FIO4DIR_bit.__shortl
#define FIO4DIRL_bit      FIO4DIR_bit.__shortl_bit
__IO_REG32_BIT(FIO4MASK,        0x400F0090,__READ_WRITE,__fgpio4_bits);
#define FIO4MASK0         FIO4MASK_bit.__byte0
#define FIO4MASK0_bit     FIO4MASK_bit.__byte0_bit
#define FIO4MASK1         FIO4MASK_bit.__byte1
#define FIO4MASK1_bit     FIO4MASK_bit.__byte1_bit
#define FIO4MASKL         FIO4MASK_bit.__shortl
#define FIO4MASKL_bit     FIO4MASK_bit.__shortl_bit
__IO_REG32_BIT(FIO4PIN,         0x400F0094,__READ_WRITE,__fgpio4_bits);
#define FIO4PIN0          FIO4PIN_bit.__byte0
#define FIO4PIN0_bit      FIO4PIN_bit.__byte0_bit
#define FIO4PIN1          FIO4PIN_bit.__byte1
#define FIO4PIN1_bit      FIO4PIN_bit.__byte1_bit
#define FIO4PINL          FIO4PIN_bit.__shortl
#define FIO4PINL_bit      FIO4PIN_bit.__shortl_bit
__IO_REG32_BIT(FIO4SET,         0x400F0098,__READ_WRITE,__fgpio4_bits);
#define FIO4SET0          FIO4SET_bit.__byte0
#define FIO4SET0_bit      FIO4SET_bit.__byte0_bit
#define FIO4SET1          FIO4SET_bit.__byte1
#define FIO4SET1_bit      FIO4SET_bit.__byte1_bit
#define FIO4SETL          FIO4SET_bit.__shortl
#define FIO4SETL_bit      FIO4SET_bit.__shortl_bit
__IO_REG32_BIT(FIO4CLR,         0x400F009C,__WRITE     ,__fgpio4_bits);
#define FIO4CLR0          FIO4CLR_bit.__byte0
#define FIO4CLR0_bit      FIO4CLR_bit.__byte0_bit
#define FIO4CLR1          FIO4CLR_bit.__byte1
#define FIO4CLR1_bit      FIO4CLR_bit.__byte1_bit
#define FIO4CLRL          FIO4CLR_bit.__shortl
#define FIO4CLRL_bit      FIO4CLR_bit.__shortl_bit

/***************************************************************************
 **
 ** GPDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMACINTSTATUS,         0x40002000,__READ      ,__dmacintstatus_bits);
__IO_REG32_BIT(DMACINTTCSTATUS,       0x40002004,__READ      ,__dmacinttcstatus_bits);
__IO_REG32_BIT(DMACINTTCCLEAR,        0x40002008,__WRITE     ,__dmacinttcclear_bits);
__IO_REG32_BIT(DMACINTERRSTAT,        0x4000200C,__READ      ,__dmacinterrstat_bits);
__IO_REG32_BIT(DMACINTERRCLR,         0x40002010,__WRITE     ,__dmacinterrclr_bits);
__IO_REG32_BIT(DMACRAWINTTCSTATUS,    0x40002014,__READ      ,__dmacrawinttcstatus_bits);
__IO_REG32_BIT(DMACRAWINTERRORSTATUS, 0x40002018,__READ      ,__dmacrawinterrorstatus_bits);
__IO_REG32_BIT(DMACENBLDCHNS,         0x4000201C,__READ      ,__dmacenbldchns_bits);
__IO_REG32_BIT(DMACSOFTBREQ,          0x40002020,__READ_WRITE,__dmacsoftbreq_bits);
__IO_REG32_BIT(DMACSOFTSREQ,          0x40002024,__READ_WRITE,__dmacsoftsreq_bits);
__IO_REG32_BIT(DMACSOFTLBREQ,         0x40002028,__READ_WRITE,__dmacsoftlbreq_bits);
__IO_REG32_BIT(DMACSOFTLSREQ,         0x4000202C,__READ_WRITE,__dmacsoftlsreq_bits);
__IO_REG32_BIT(DMACCONFIGURATION,     0x40002030,__READ_WRITE,__dmacconfig_bits);
__IO_REG32_BIT(DMACSYNC,              0x40002034,__READ_WRITE,__dmacsync_bits);
__IO_REG32(    DMACC0SRCADDR,         0x40002100,__READ_WRITE);
__IO_REG32(    DMACC0DESTADDR,        0x40002104,__READ_WRITE);
__IO_REG32_BIT(DMACC0LLI,             0x40002108,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC0CONTROL,         0x4000210C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC0CONFIGURATION,   0x40002110,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC1SRCADDR,         0x40002120,__READ_WRITE);
__IO_REG32(    DMACC1DESTADDR,        0x40002124,__READ_WRITE);
__IO_REG32_BIT(DMACC1LLI,             0x40002128,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC1CONTROL,         0x4000212C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC1CONFIGURATION,   0x40002130,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC2SRCADDR,         0x40002140,__READ_WRITE);
__IO_REG32(    DMACC2DESTADDR,        0x40002144,__READ_WRITE);
__IO_REG32_BIT(DMACC2LLI,             0x40002148,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC2CONTROL,         0x4000214C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC2CONFIGURATION,   0x40002150,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC3SRCADDR,         0x40002160,__READ_WRITE);
__IO_REG32(    DMACC3DESTADDR,        0x40002164,__READ_WRITE);
__IO_REG32_BIT(DMACC3LLI,             0x40002168,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC3CONTROL,         0x4000216C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC3CONFIGURATION,   0x40002170,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC4SRCADDR,         0x40002180,__READ_WRITE);
__IO_REG32(    DMACC4DESTADDR,        0x40002184,__READ_WRITE);
__IO_REG32_BIT(DMACC4LLI,             0x40002188,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC4CONTROL,         0x4000218C,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC4CONFIGURATION,   0x40002190,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC5SRCADDR,         0x400021A0,__READ_WRITE);
__IO_REG32(    DMACC5DESTADDR,        0x400021A4,__READ_WRITE);
__IO_REG32_BIT(DMACC5LLI,             0x400021A8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC5CONTROL,         0x400021AC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC5CONFIGURATION,   0x400021B0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC6SRCADDR,         0x400021C0,__READ_WRITE);
__IO_REG32(    DMACC6DESTADDR,        0x400021C4,__READ_WRITE);
__IO_REG32_BIT(DMACC6LLI,             0x400021C8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC6CONTROL,         0x400021CC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC6CONFIGURATION,   0x400021D0,__READ_WRITE,__dma_cfg_bits);
__IO_REG32(    DMACC7SRCADDR,         0x400021E0,__READ_WRITE);
__IO_REG32(    DMACC7DESTADDR,        0x400021E4,__READ_WRITE);
__IO_REG32_BIT(DMACC7LLI,             0x400021E8,__READ_WRITE,__dma_lli_bits);
__IO_REG32_BIT(DMACC7CONTROL,         0x400021EC,__READ_WRITE,__dma_ctrl_bits);
__IO_REG32_BIT(DMACC7CONFIGURATION,   0x400021F0,__READ_WRITE,__dma_cfg_bits);

/***************************************************************************
 **
 ** SPIFI
 **
 ***************************************************************************/
__IO_REG32_BIT(SPIFICTRL,             0x40003000,__READ_WRITE,__spifictrl_bits);
__IO_REG32_BIT(SPIFICMD,              0x40003004,__READ_WRITE,__spificmd_bits);
__IO_REG32(    SPIFIADDR,             0x40003008,__READ_WRITE);
__IO_REG32(    SPIFIDATINTM,          0x4000300C,__READ_WRITE);
__IO_REG32(    SPIFIADDRINTM,         0x40003010,__READ_WRITE);
__IO_REG32(    SPIFIDAT,              0x40003014,__READ_WRITE);
__IO_REG32_BIT(SPIFIMEMCMD,           0x40003018,__READ_WRITE,__spifimemcmd_bits);
__IO_REG32_BIT(SPIFISTAT,             0x4000301C,__READ      ,__spifistat_bits);

/***************************************************************************
 **
 ** SDIO
 **
 ***************************************************************************/
__IO_REG32(    SDIO_SDMASYSADDR,      0x40004000,__READ_WRITE );
__IO_REG32_BIT(SDIO_BLK,              0x40004004,__READ_WRITE ,__sdio_blk_bits);
__IO_REG32(    SDIO_ARG,              0x40004008,__READ_WRITE );
__IO_REG32_BIT(SDIO_CMD_TXFMODE,      0x4000400C,__READ_WRITE ,__sdio_cmd_txfmode_bits);
__IO_REG32(    SDIO_RESP01,           0x40004010,__READ       );
__IO_REG32(    SDIO_RESP23,           0x40004014,__READ       );
__IO_REG32(    SDIO_RESP45,           0x40004018,__READ       );
__IO_REG32(    SDIO_RESP67,           0x4000401C,__READ       );
__IO_REG32(    SDIO_BUFDATA,          0x40004020,__READ_WRITE );
__IO_REG32_BIT(SDIO_PRESENTSTATE,     0x40004024,__READ_WRITE ,__sdio_presentstate_bits);
__IO_REG32_BIT(SDIO_WAKE_HOSTCTRL,    0x40004028,__READ_WRITE ,__sdio_wake_hostctrl_bits);
__IO_REG32_BIT(SDIO_CLKCTRL,          0x4000402C,__READ_WRITE ,__sdio_clkctrl_bits);
__IO_REG32_BIT(SDIO_INTSTAT,          0x40004030,__READ_WRITE ,__sdio_intstat_bits);
__IO_REG32_BIT(SDIO_INTSTATEN,        0x40004034,__READ_WRITE ,__sdio_intstaten_bits);
__IO_REG32_BIT(SDIO_NINTSIGEN,        0x40004038,__READ_WRITE ,__sdio_nintsigen_bits);
__IO_REG32_BIT(SDIO_AUTOCMD12ERRSTAT, 0x4000403C,__READ_WRITE ,__sdio_autocmd12errstat_bits);
__IO_REG32_BIT(SDIO_CAPB,             0x40004040,__READ_WRITE ,__sdio_capb_bits);
__IO_REG32_BIT(SDIO_MAXCURRCAPB,      0x40004048,__READ       ,__sdio_maxcurrcapb_bits);
__IO_REG32_BIT(SDIO_FORCE_EVT,        0x40004050,__READ_WRITE ,__sdio_force_evt_bits);
__IO_REG32_BIT(SDIO_ADMA_ERR,         0x40004054,__READ_WRITE ,__sdio_adma_err_bits);
__IO_REG32(    SDIO_ADMA_ADDRL,       0x40004058,__READ_WRITE );
__IO_REG32(    SDIO_ADMA_ADDRH,       0x4000405C,__READ_WRITE );
__IO_REG32(    SDIO_BOOT_TO,          0x40004060,__READ_WRITE );
__IO_REG32_BIT(SDIO_SPIINT,           0x400040F0,__READ_WRITE ,__sdio_spiint_bits);
__IO_REG32(    SDIO_HCVERS,           0x400040FC,__READ       );       

/***************************************************************************
 **
 ** EMC
 **
 ***************************************************************************/
__IO_REG32_BIT(EMCControl,            0x40005000,__READ_WRITE ,__emc_ctrl_bits);
__IO_REG32_BIT(EMCStatus,             0x40005004,__READ       ,__emc_st_bits);
__IO_REG32_BIT(EMCConfig,             0x40005008,__READ_WRITE ,__emc_cfg_bits);
__IO_REG32_BIT(EMCDynamicControl,     0x40005020,__READ_WRITE ,__emc_dctrl_bits);
__IO_REG32_BIT(EMCDynamicRefresh,     0x40005024,__READ_WRITE ,__emc_drfr_bits);
__IO_REG32_BIT(EMCDynamicReadConfig,  0x40005028,__READ_WRITE ,__emc_drdcfg_bits);
__IO_REG32_BIT(EMCDynamictRP,         0x40005030,__READ_WRITE ,__emc_drp_bits);
__IO_REG32_BIT(EMCDynamictRAS,        0x40005034,__READ_WRITE ,__emc_dras_bits);
__IO_REG32_BIT(EMCDynamictSREX,       0x40005038,__READ_WRITE ,__emc_dsrex_bits);
__IO_REG32_BIT(EMCDynamictAPR,        0x4000503C,__READ_WRITE ,__emc_dapr_bits);
__IO_REG32_BIT(EMCDynamictDAL,        0x40005040,__READ_WRITE ,__emc_ddal_bits);
__IO_REG32_BIT(EMCDynamictWR,         0x40005044,__READ_WRITE ,__emc_dwr_bits);
__IO_REG32_BIT(EMCDynamictRC,         0x40005048,__READ_WRITE ,__emc_drc_bits);
__IO_REG32_BIT(EMCDynamictRFC,        0x4000504C,__READ_WRITE ,__emc_drfc_bits);
__IO_REG32_BIT(EMCDynamictXSR,        0x40005050,__READ_WRITE ,__emc_dxsr_bits);
__IO_REG32_BIT(EMCDynamictRRD,        0x40005054,__READ_WRITE ,__emc_drrd_bits);
__IO_REG32_BIT(EMCDynamictMRD,        0x40005058,__READ_WRITE ,__emc_dmrd_bits);
__IO_REG32_BIT(EMCStaticExtendedWait, 0x40005080,__READ_WRITE ,__emc_s_ext_wait_bits);
__IO_REG32_BIT(EMCDynamicConfig0,     0x40005100,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas0,     0x40005104,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig1,     0x40005120,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas1,     0x40005124,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig2,     0x40005140,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas2,     0x40005144,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCDynamicConfig3,     0x40005160,__READ_WRITE ,__emc_d_config_bits);
__IO_REG32_BIT(EMCDynamicRasCas3,     0x40005164,__READ_WRITE ,__emc_d_ras_cas_bits);
__IO_REG32_BIT(EMCStaticConfig0,      0x40005200,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen0,     0x40005204,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen0,     0x40005208,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd0,      0x4000520C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage0,    0x40005210,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr0,      0x40005214,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn0,    0x40005218,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig1,      0x40005220,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen1,     0x40005224,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen1,     0x40005228,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd1,      0x4000522C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage1,    0x40005230,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr1,      0x40005234,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn1,    0x40005238,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig2,      0x40005240,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen2,     0x40005244,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen2,     0x40005248,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd2,      0x4000524C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage2,    0x40005250,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr2,      0x40005254,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn2,    0x40005258,__READ_WRITE ,__emc_s_wait_turn_bits);
__IO_REG32_BIT(EMCStaticConfig3,      0x40005260,__READ_WRITE ,__emc_s_config_bits);
__IO_REG32_BIT(EMCStaticWaitWen3,     0x40005264,__READ_WRITE ,__emc_s_wait_wen_bits);
__IO_REG32_BIT(EMCStaticWaitOen3,     0x40005268,__READ_WRITE ,__emc_s_wait_oen_bits);
__IO_REG32_BIT(EMCStaticWaitRd3,      0x4000526C,__READ_WRITE ,__emc_s_wait_rd_bits);
__IO_REG32_BIT(EMCStaticWaitPage3,    0x40005270,__READ_WRITE ,__emc_s_wait_pg_bits);
__IO_REG32_BIT(EMCStaticWaitWr3,      0x40005274,__READ_WRITE ,__emc_s_wait_wr_bits);
__IO_REG32_BIT(EMCStaticWaitTurn3,    0x40005278,__READ_WRITE ,__emc_s_wait_turn_bits);

/***************************************************************************
 **
 ** SCT
 **
 ***************************************************************************/
__IO_REG32_BIT(SCTCONFIG,             0x40000000,__READ_WRITE ,__sctconfig_bits);
__IO_REG32_BIT(SCTCTRL,               0x40000004,__READ_WRITE ,__sctctrl_bits);
__IO_REG32_BIT(SCTLIMIT,              0x4000000C,__READ_WRITE ,__sctlimit_bits);
__IO_REG32_BIT(SCTHALT,               0x40000010,__READ_WRITE ,__scthalt_bits);
__IO_REG32_BIT(SCTSTOP,               0x40000014,__READ_WRITE ,__sctstop_bits);
__IO_REG32_BIT(SCTSTART,              0x40000018,__READ_WRITE ,__sctstart_bits);
__IO_REG32_BIT(SCTCOUNT,              0x40000040,__READ_WRITE ,__sctcount_bits);
__IO_REG32_BIT(SCTSTATE,              0x40000044,__READ_WRITE ,__sctstate_bits);
__IO_REG32_BIT(SCTINPUT,              0x40000048,__READ       ,__sctinput_bits);
__IO_REG32_BIT(SCTREGMODE,            0x4000004C,__READ_WRITE ,__sctregmode_bits);
__IO_REG32_BIT(SCTOUTPUT,             0x40000050,__READ_WRITE ,__sctoutput_bits);
__IO_REG32_BIT(SCTOUTPUTDIRCTRL,      0x40000054,__READ_WRITE ,__sctoutputdirctrl_bits);
__IO_REG32_BIT(SCTRES,                0x40000058,__READ_WRITE ,__sctres_bits);
__IO_REG32_BIT(SCTDMAREQ0,            0x4000005C,__READ_WRITE ,__sctdmareq0_bits);
__IO_REG32_BIT(SCTDMAREQ1,            0x40000060,__READ_WRITE ,__sctdmareq1_bits);
__IO_REG32_BIT(SCTEVEN,               0x400000F0,__READ_WRITE ,__scteven_bits);
__IO_REG32_BIT(SCTEVFLAG,             0x400000F4,__READ_WRITE ,__sctevflag_bits);
__IO_REG32_BIT(SCTCONEN,              0x400000F8,__READ_WRITE ,__sctconen_bits);
__IO_REG32_BIT(SCTCONFLAG,            0x400000FC,__READ_WRITE ,__sctconflag_bits);
__IO_REG32_BIT(SCTMATCH0,             0x40000100,__READ_WRITE ,__sctmatch_cap_bits);   
#define SCTCAP0             SCTMATCH0
#define SCTCAP0_bit         SCTMATCH0_bit
__IO_REG32_BIT(SCTMATCH1,             0x40000104,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP1             SCTMATCH1
#define SCTCAP1_bit         SCTMATCH1_bit
__IO_REG32_BIT(SCTMATCH2,             0x40000108,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP2             SCTMATCH2
#define SCTCAP2_bit         SCTMATCH2_bit
__IO_REG32_BIT(SCTMATCH3,             0x4000010C,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP3             SCTMATCH3
#define SCTCAP3_bit         SCTMATCH3_bit
__IO_REG32_BIT(SCTMATCH4,             0x40000110,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP4             SCTMATCH4
#define SCTCAP4_bit         SCTMATCH4_bit
__IO_REG32_BIT(SCTMATCH5,             0x40000114,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP5             SCTMATCH5
#define SCTCAP5_bit         SCTMATCH5_bit
__IO_REG32_BIT(SCTMATCH6,             0x40000118,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP6             SCTMATCH6
#define SCTCAP6_bit         SCTMATCH6_bit
__IO_REG32_BIT(SCTMATCH7,             0x4000011C,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP7             SCTMATCH7
#define SCTCAP7_bit         SCTMATCH7_bit
__IO_REG32_BIT(SCTMATCH8,             0x40000120,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP8             SCTMATCH8
#define SCTCAP8_bit         SCTMATCH8_bit
__IO_REG32_BIT(SCTMATCH9,             0x40000124,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP9             SCTMATCH9
#define SCTCAP9_bit         SCTMATCH9_bit
__IO_REG32_BIT(SCTMATCH10,            0x40000128,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP10            SCTMATCH10
#define SCTCAP10_bit        SCTMATCH10_bit
__IO_REG32_BIT(SCTMATCH11,            0x4000012C,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP11            SCTMATCH11
#define SCTCAP11_bit        SCTMATCH11_bit
__IO_REG32_BIT(SCTMATCH12,            0x40000130,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP12            SCTMATCH12
#define SCTCAP12_bit        SCTMATCH12_bit
__IO_REG32_BIT(SCTMATCH13,            0x40000134,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP13            SCTMATCH13
#define SCTCAP13_bit        SCTMATCH13_bit
__IO_REG32_BIT(SCTMATCH14,            0x40000138,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP14            SCTMATCH14
#define SCTCAP14_bit        SCTMATCH14_bit
__IO_REG32_BIT(SCTMATCH15,            0x4000013C,__READ_WRITE ,__sctmatch_cap_bits);
#define SCTCAP15            SCTMATCH15
#define SCTCAP15_bit        SCTMATCH15_bit
__IO_REG32_BIT(SCTMATCHREL0,          0x40000200,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL0         SCTMATCHREL0
#define SCTCAPCTRL0_bit     SCTMATCHREL0_bit
__IO_REG32_BIT(SCTMATCHREL1,          0x40000204,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL1         SCTMATCHREL1
#define SCTCAPCTRL1_bit     SCTMATCHREL1_bit
__IO_REG32_BIT(SCTMATCHREL2,          0x40000208,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL2         SCTMATCHREL2
#define SCTCAPCTRL2_bit     SCTMATCHREL2_bit
__IO_REG32_BIT(SCTMATCHREL3,          0x4000020C,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL3         SCTMATCHREL3
#define SCTCAPCTRL3_bit     SCTMATCHREL3_bit
__IO_REG32_BIT(SCTMATCHREL4,          0x40000210,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL4         SCTMATCHREL4
#define SCTCAPCTRL4_bit     SCTMATCHREL4_bit
__IO_REG32_BIT(SCTMATCHREL5,          0x40000214,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL5         SCTMATCHREL5
#define SCTCAPCTRL5_bit     SCTMATCHREL5_bit
__IO_REG32_BIT(SCTMATCHREL6,          0x40000218,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL6         SCTMATCHREL6
#define SCTCAPCTRL6_bit     SCTMATCHREL6_bit
__IO_REG32_BIT(SCTMATCHREL7,          0x4000021C,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL7         SCTMATCHREL7
#define SCTCAPCTRL7_bit     SCTMATCHREL7_bit
__IO_REG32_BIT(SCTMATCHREL8,          0x40000220,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL8         SCTMATCHREL8
#define SCTCAPCTRL8_bit     SCTMATCHREL8_bit
__IO_REG32_BIT(SCTMATCHREL9,          0x40000224,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL9         SCTMATCHREL9
#define SCTCAPCTRL9_bit     SCTMATCHREL9_bit
__IO_REG32_BIT(SCTMATCHREL10,         0x40000228,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL10        SCTMATCHREL10
#define SCTCAPCTRL10_bit    SCTMATCHREL10_bit
__IO_REG32_BIT(SCTMATCHREL11,         0x4000022C,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL11        SCTMATCHREL11
#define SCTCAPCTRL11_bit    SCTMATCHREL11_bit
__IO_REG32_BIT(SCTMATCHREL12,         0x40000230,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL12        SCTMATCHREL12
#define SCTCAPCTRL12_bit    SCTMATCHREL12_bit
__IO_REG32_BIT(SCTMATCHREL13,         0x40000234,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL13        SCTMATCHREL13
#define SCTCAPCTRL13_bit    SCTMATCHREL13_bit
__IO_REG32_BIT(SCTMATCHREL14,         0x40000238,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL14        SCTMATCHREL14
#define SCTCAPCTRL14_bit    SCTMATCHREL14_bit
__IO_REG32_BIT(SCTMATCHREL15,         0x4000023C,__READ_WRITE ,__sctmatchrel_capctrl_bits);
#define SCTCAPCTRL15        SCTMATCHREL15
#define SCTCAPCTRL15_bit    SCTMATCHREL15_bit
__IO_REG32_BIT(SCTEVSTATEMSK0,        0x40000300,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL0,            0x40000304,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK1,        0x40000308,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL1,            0x4000030C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK2,        0x40000310,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL2,            0x40000314,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK3,        0x40000318,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL3,            0x4000031C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK4,        0x40000320,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL4,            0x40000324,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK5,        0x40000328,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL5,            0x4000032C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK6,        0x40000330,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL6,            0x40000334,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK7,        0x40000338,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL7,            0x4000033C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK8,        0x40000340,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL8,            0x40000344,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK9,        0x40000348,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL9,            0x4000034C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK10,       0x40000350,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL10,           0x40000354,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK11,       0x40000358,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL11,           0x4000035C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK12,       0x40000360,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL12,           0x40000364,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK13,       0x40000368,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL13,           0x4000036C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK14,       0x40000370,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL14,           0x40000374,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTEVSTATEMSK15,       0x40000378,__READ_WRITE ,__sctevstatemsk_bits);
__IO_REG32_BIT(SCTEVCTRL15,           0x4000037C,__READ_WRITE ,__sctevctrl_bits);
__IO_REG32_BIT(SCTOUTPUTSET0,         0x40000500,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL0,          0x40000504,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET1,         0x40000508,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL1,          0x4000050C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET2,         0x40000510,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL2,          0x40000514,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET3,         0x40000518,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL3,          0x4000051C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET4,         0x40000520,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL4,          0x40000524,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET5,         0x40000528,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL5,          0x4000052C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET6,         0x40000530,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL6,          0x40000534,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET7,         0x40000538,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL7,          0x4000053C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET8,         0x40000540,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL8,          0x40000544,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET9,         0x40000548,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL9,          0x4000054C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET10,        0x40000550,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL10,         0x40000554,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET11,        0x40000558,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL11,         0x4000055C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET12,        0x40000560,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL12,         0x40000564,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET13,        0x40000568,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL13,         0x4000056C,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET14,        0x40000570,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL14,         0x40000574,__READ_WRITE ,__sctoutputcl_bits);
__IO_REG32_BIT(SCTOUTPUTSET15,        0x40000578,__READ_WRITE ,__sctoutputset_bits);
__IO_REG32_BIT(SCTOUTPUTCL15,         0x4000057C,__READ_WRITE ,__sctoutputcl_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(T0IR,                  0x40084000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T0TCR,                 0x40084004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T0TC,                  0x40084008,__READ_WRITE);
__IO_REG32(    T0PR,                  0x4008400C,__READ_WRITE);
__IO_REG32(    T0PC,                  0x40084010,__READ_WRITE);
__IO_REG32_BIT(T0MCR,                 0x40084014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T0MR0,                 0x40084018,__READ_WRITE);
__IO_REG32(    T0MR1,                 0x4008401C,__READ_WRITE);
__IO_REG32(    T0MR2,                 0x40084020,__READ_WRITE);
__IO_REG32(    T0MR3,                 0x40084024,__READ_WRITE);
__IO_REG32_BIT(T0CCR,                 0x40084028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T0CR0,                 0x4008402C,__READ);
__IO_REG32(    T0CR1,                 0x40084030,__READ);
__IO_REG32(    T0CR2,                 0x40084034,__READ);
__IO_REG32(    T0CR3,                 0x40084038,__READ);
__IO_REG32_BIT(T0EMR,                 0x4008403C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T0CTCR,                0x40084070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(T1IR,                  0x40085000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T1TCR,                 0x40085004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T1TC,                  0x40085008,__READ_WRITE);
__IO_REG32(    T1PR,                  0x4008500C,__READ_WRITE);
__IO_REG32(    T1PC,                  0x40085010,__READ_WRITE);
__IO_REG32_BIT(T1MCR,                 0x40085014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T1MR0,                 0x40085018,__READ_WRITE);
__IO_REG32(    T1MR1,                 0x4008501C,__READ_WRITE);
__IO_REG32(    T1MR2,                 0x40085020,__READ_WRITE);
__IO_REG32(    T1MR3,                 0x40085024,__READ_WRITE);
__IO_REG32_BIT(T1CCR,                 0x40085028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T1CR0,                 0x4008502C,__READ);
__IO_REG32(    T1CR1,                 0x40085030,__READ);
__IO_REG32(    T1CR2,                 0x40085034,__READ);
__IO_REG32(    T1CR3,                 0x40085038,__READ);
__IO_REG32_BIT(T1EMR,                 0x4008503C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T1CTCR,                0x40085070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(T2IR,                  0x400C3000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T2TCR,                 0x400C3004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T2TC,                  0x400C3008,__READ_WRITE);
__IO_REG32(    T2PR,                  0x400C300C,__READ_WRITE);
__IO_REG32(    T2PC,                  0x400C3010,__READ_WRITE);
__IO_REG32_BIT(T2MCR,                 0x400C3014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T2MR0,                 0x400C3018,__READ_WRITE);
__IO_REG32(    T2MR1,                 0x400C301C,__READ_WRITE);
__IO_REG32(    T2MR2,                 0x400C3020,__READ_WRITE);
__IO_REG32(    T2MR3,                 0x400C3024,__READ_WRITE);
__IO_REG32_BIT(T2CCR,                 0x400C3028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T2CR0,                 0x400C302C,__READ);
__IO_REG32(    T2CR1,                 0x400C3030,__READ);
__IO_REG32(    T2CR2,                 0x400C3034,__READ);
__IO_REG32(    T2CR3,                 0x400C3038,__READ);
__IO_REG32_BIT(T2EMR,                 0x400C303C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T2CTCR,                0x400C3070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(T3IR,                  0x400C4000,__READ_WRITE ,__ir_bits);
__IO_REG32_BIT(T3TCR,                 0x400C4004,__READ_WRITE ,__tcr_bits);
__IO_REG32(    T3TC,                  0x400C4008,__READ_WRITE);
__IO_REG32(    T3PR,                  0x400C400C,__READ_WRITE);
__IO_REG32(    T3PC,                  0x400C4010,__READ_WRITE);
__IO_REG32_BIT(T3MCR,                 0x400C4014,__READ_WRITE ,__mcr_bits);
__IO_REG32(    T3MR0,                 0x400C4018,__READ_WRITE);
__IO_REG32(    T3MR1,                 0x400C401C,__READ_WRITE);
__IO_REG32(    T3MR2,                 0x400C4020,__READ_WRITE);
__IO_REG32(    T3MR3,                 0x400C4024,__READ_WRITE);
__IO_REG32_BIT(T3CCR,                 0x400C4028,__READ_WRITE ,__tccr_bits);
__IO_REG32(    T3CR0,                 0x400C402C,__READ);
__IO_REG32(    T3CR1,                 0x400C4030,__READ);
__IO_REG32(    T3CR2,                 0x400C4034,__READ);
__IO_REG32(    T3CR3,                 0x400C4038,__READ);
__IO_REG32_BIT(T3EMR,                 0x400C403C,__READ_WRITE ,__emr_bits);
__IO_REG32_BIT(T3CTCR,                0x400C4070,__READ_WRITE ,__ctcr_bits);

/***************************************************************************
 **
 ** Motor control PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(MCCON,                 0x400A0000,__READ       ,__mccon_bits);
__IO_REG32_BIT(MCCON_SET,             0x400A0004,__WRITE      ,__mccon_set_bits);
__IO_REG32_BIT(MCCON_CLR,             0x400A0008,__WRITE      ,__mccon_clr_bits);
__IO_REG32_BIT(MCCAPCON,              0x400A000C,__READ       ,__mccapcon_bits);
__IO_REG32_BIT(MCCAPCON_SET,          0x400A0010,__WRITE      ,__mccapcon_set_bits);
__IO_REG32_BIT(MCCAPCON_CLR,          0x400A0014,__WRITE      ,__mccapcon_clr_bits);
__IO_REG32(    MCTC0,                 0x400A0018,__READ_WRITE );
__IO_REG32(    MCTC1,                 0x400A001C,__READ_WRITE );
__IO_REG32(    MCTC2,                 0x400A0020,__READ_WRITE );
__IO_REG32(    MCLIM0,                0x400A0024,__READ_WRITE );
__IO_REG32(    MCLIM1,                0x400A0028,__READ_WRITE );
__IO_REG32(    MCLIM2,                0x400A002C,__READ_WRITE );
__IO_REG32(    MCMAT0,                0x400A0030,__READ_WRITE );
__IO_REG32(    MCMAT1,                0x400A0034,__READ_WRITE );
__IO_REG32(    MCMAT2,                0x400A0038,__READ_WRITE );
__IO_REG32_BIT(MCDT,                  0x400A003C,__READ_WRITE ,__mcdt_bits);
__IO_REG32_BIT(MCCP,                  0x400A0040,__READ_WRITE ,__mcccp_bits);
__IO_REG32(    MCCAP0,                0x400A0044,__READ       );
__IO_REG32(    MCCAP1,                0x400A0048,__READ       );
__IO_REG32(    MCCAP2,                0x400A004C,__READ       );
__IO_REG32_BIT(MCINTEN,               0x400A0050,__READ       ,__mcinten_bits);
__IO_REG32_BIT(MCINTEN_SET,           0x400A0054,__WRITE      ,__mcinten_set_bits);
__IO_REG32_BIT(MCINTEN_CLR,           0x400A0058,__WRITE      ,__mcinten_clr_bits);
__IO_REG32_BIT(MCCNTCON,              0x400A005C,__READ       ,__mccntcon_bits);
__IO_REG32_BIT(MCCNTCON_SET,          0x400A0060,__WRITE      ,__mccntcon_set_bits);
__IO_REG32_BIT(MCCNTCON_CLR,          0x400A0064,__WRITE      ,__mccntcon_clr_bits);
__IO_REG32_BIT(MCINTF,                0x400A0068,__READ       ,__mcintf_bits);
__IO_REG32_BIT(MCINTF_SET,            0x400A006C,__WRITE      ,__mcintf_set_bits);
__IO_REG32_BIT(MCINTF_CLR,            0x400A0070,__WRITE      ,__mcintf_clr_bits);
__IO_REG32_BIT(MCCAP_CLR,             0x400A0074,__WRITE      ,__mccap_clr_bits);

/***************************************************************************
 **
 ** Quadrature Encoder Interface
 **
 ***************************************************************************/
__IO_REG32_BIT(QEICON,                0x400C6000,__WRITE      ,__qeicon_bits);
__IO_REG32_BIT(QEISTAT,               0x400C6004,__READ       ,__qeistat_bits);
__IO_REG32_BIT(QEICONF,               0x400C6008,__READ_WRITE ,__qeiconf_bits);
__IO_REG32(    QEIPOS,                0x400C600C,__READ       );
__IO_REG32(    QEIMAXPSOS,            0x400C6010,__READ_WRITE );
__IO_REG32(    CMPOS0,                0x400C6014,__READ_WRITE );
__IO_REG32(    CMPOS1,                0x400C6018,__READ_WRITE );
__IO_REG32(    CMPOS2,                0x400C601C,__READ_WRITE );
__IO_REG32(    INXCNT,                0x400C6020,__READ       );
__IO_REG32(    INXCMP0,               0x400C6024,__READ_WRITE );
__IO_REG32(    QEILOAD,               0x400C6028,__READ_WRITE );
__IO_REG32(    QEITIME,               0x400C602C,__READ       );
__IO_REG32(    QEIVEL,                0x400C6030,__READ       );
__IO_REG32(    QEICAP,                0x400C6034,__READ       );
__IO_REG32(    VELCOMP,               0x400C6038,__READ_WRITE );
__IO_REG32(    FILTERPHA,             0x400C603C,__READ_WRITE );
__IO_REG32(    FILTERPHB,             0x400C6040,__READ_WRITE );
__IO_REG32(    FILTERINX,             0x400C6044,__READ_WRITE );
__IO_REG32(    WINDOW,                0x400C6048,__READ_WRITE );
__IO_REG32(    INXCMP1,               0x400C604C,__READ_WRITE );
__IO_REG32(    INXCMP2,               0x400C6050,__READ_WRITE );
__IO_REG32_BIT(QEIIEC,                0x400C6FD8,__WRITE      ,__qeiiec_bits);
__IO_REG32_BIT(QEIIES,                0x400C6FDC,__WRITE      ,__qeiiec_bits);
__IO_REG32_BIT(QEIINTSTAT,            0x400C6FE0,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEIIE,                 0x400C6FE4,__READ       ,__qeiintstat_bits);
__IO_REG32_BIT(QEICLR,                0x400C6FE8,__WRITE      ,__qeiintstat_bits);
__IO_REG32_BIT(QEISET,                0x400C6FEC,__WRITE      ,__qeiintstat_bits);

/***************************************************************************
 **
 ** Repetitive Interrupt Timer
 **
 ***************************************************************************/
__IO_REG32(    RICOMPVAL,             0x400C0000,__READ_WRITE );
__IO_REG32(    RIMASK,                0x400C0004,__READ_WRITE );
__IO_REG32_BIT(RICTRL,                0x400C0008,__READ_WRITE ,__rictrl_bits);
__IO_REG32(    RICOUNTER,             0x400C000C,__READ_WRITE );

/***************************************************************************
 **
 ** Alarm timer
 **
 ***************************************************************************/
__IO_REG16(    ATDOWNCOUNTER,         0x40040000,__READ_WRITE );
__IO_REG16(    ATPRESET,              0x40040004,__READ_WRITE );
__IO_REG32_BIT(ATCLR_EN,              0x40040FD8,__WRITE      ,__atclr_en_bits);
__IO_REG32_BIT(ATSET_EN,              0x40040FDC,__WRITE      ,__atset_en_bits);
__IO_REG32_BIT(ATSTATUS,              0x40040FE0,__READ       ,__atstatus_bits);
__IO_REG32_BIT(ATENABLE,              0x40040FE4,__READ       ,__atenable_bits);
__IO_REG32_BIT(ATCLR_STAT,            0x40040FE8,__WRITE      ,__atclr_stat_bits);
__IO_REG32_BIT(ATSET_STAT,            0x40040FEC,__WRITE      ,__atset_stat_bits);

/***************************************************************************
 **
 ** Watchdog
 **
 ***************************************************************************/
__IO_REG32_BIT(WDMOD,                 0x40080000,__READ_WRITE ,__wdmod_bits);
__IO_REG32_BIT(WDTC,                  0x40080004,__READ_WRITE ,__wdtc_bits);
__IO_REG32_BIT(WDFEED,                0x40080008,__WRITE      ,__wdfeed_bits);
__IO_REG32_BIT(WDTV,                  0x4008000C,__READ       ,__wdtc_bits);
__IO_REG32_BIT(WDWARNINT,             0x40080014,__READ_WRITE ,__wdwarnint_bits);
__IO_REG32_BIT(WDWINDOW,              0x40080018,__READ_WRITE ,__wdwindow_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTCILR,                0x40046000,__READ_WRITE ,__ilr_bits);
__IO_REG32_BIT(RTCCCR,                0x40046008,__READ_WRITE ,__rtcccr_bits);
__IO_REG32_BIT(RTCCIIR,               0x4004600C,__READ_WRITE ,__ciir_bits);
__IO_REG32_BIT(RTCAMR,                0x40046010,__READ_WRITE ,__amr_bits);
__IO_REG32_BIT(RTCCTIME0,             0x40046014,__READ       ,__ctime0_bits);
__IO_REG32_BIT(RTCCTIME1,             0x40046018,__READ       ,__ctime1_bits);
__IO_REG32_BIT(RTCCTIME2,             0x4004601C,__READ       ,__ctime2_bits);
__IO_REG32_BIT(RTCSEC,                0x40046020,__READ_WRITE ,__sec_bits);
__IO_REG32_BIT(RTCMIN,                0x40046024,__READ_WRITE ,__min_bits);
__IO_REG32_BIT(RTCHOUR,               0x40046028,__READ_WRITE ,__hour_bits);
__IO_REG32_BIT(RTCDOM,                0x4004602C,__READ_WRITE ,__dom_bits);
__IO_REG32_BIT(RTCDOW,                0x40046030,__READ_WRITE ,__dow_bits);
__IO_REG32_BIT(RTCDOY,                0x40046034,__READ_WRITE ,__doy_bits);
__IO_REG32_BIT(RTCMONTH,              0x40046038,__READ_WRITE ,__month_bits);
__IO_REG32_BIT(RTCYEAR,               0x4004603C,__READ_WRITE ,__year_bits);
__IO_REG32_BIT(RTCCALIBRATION,        0x40046040,__READ_WRITE ,__calibration_bits);
__IO_REG32_BIT(RTCALSEC,              0x40046060,__READ_WRITE ,__sec_bits);
__IO_REG32_BIT(RTCALMIN,              0x40046064,__READ_WRITE ,__min_bits);
__IO_REG32_BIT(RTCALHOUR,             0x40046068,__READ_WRITE ,__hour_bits);
__IO_REG32_BIT(RTCALDOM,              0x4004606C,__READ_WRITE ,__dom_bits);
__IO_REG32_BIT(RTCALDOW,              0x40046070,__READ_WRITE ,__dow_bits);
__IO_REG32_BIT(RTCALDOY,              0x40046074,__READ_WRITE ,__doy_bits);
__IO_REG32_BIT(RTCALMON,              0x40046078,__READ_WRITE ,__month_bits);
__IO_REG32_BIT(RTCALYEAR,             0x4004607C,__READ_WRITE ,__year_bits);
__IO_REG32(    REGFILE0,              0x40041000,__READ_WRITE );
__IO_REG32(    REGFILE1,              0x40041004,__READ_WRITE );
__IO_REG32(    REGFILE2,              0x40041008,__READ_WRITE );
__IO_REG32(    REGFILE3,              0x4004100C,__READ_WRITE );
__IO_REG32(    REGFILE4,              0x40041010,__READ_WRITE );
__IO_REG32(    REGFILE5,              0x40041014,__READ_WRITE );
__IO_REG32(    REGFILE6,              0x40041018,__READ_WRITE );
__IO_REG32(    REGFILE7,              0x4004101C,__READ_WRITE );
__IO_REG32(    REGFILE8,              0x40041020,__READ_WRITE );
__IO_REG32(    REGFILE9,              0x40041024,__READ_WRITE );
__IO_REG32(    REGFILE10,             0x40041028,__READ_WRITE );
__IO_REG32(    REGFILE11,             0x4004102C,__READ_WRITE );
__IO_REG32(    REGFILE12,             0x40041030,__READ_WRITE );
__IO_REG32(    REGFILE13,             0x40041034,__READ_WRITE );
__IO_REG32(    REGFILE14,             0x40041038,__READ_WRITE );
__IO_REG32(    REGFILE15,             0x4004103C,__READ_WRITE );
__IO_REG32(    REGFILE16,             0x40041040,__READ_WRITE );
__IO_REG32(    REGFILE17,             0x40041044,__READ_WRITE );
__IO_REG32(    REGFILE18,             0x40041048,__READ_WRITE );
__IO_REG32(    REGFILE19,             0x4004104C,__READ_WRITE );
__IO_REG32(    REGFILE20,             0x40041050,__READ_WRITE );
__IO_REG32(    REGFILE21,             0x40041054,__READ_WRITE );
__IO_REG32(    REGFILE22,             0x40041058,__READ_WRITE );
__IO_REG32(    REGFILE23,             0x4004105C,__READ_WRITE );
__IO_REG32(    REGFILE24,             0x40041060,__READ_WRITE );
__IO_REG32(    REGFILE25,             0x40041064,__READ_WRITE );
__IO_REG32(    REGFILE26,             0x40041068,__READ_WRITE );
__IO_REG32(    REGFILE27,             0x4004106C,__READ_WRITE );
__IO_REG32(    REGFILE28,             0x40041070,__READ_WRITE );
__IO_REG32(    REGFILE29,             0x40041074,__READ_WRITE );
__IO_REG32(    REGFILE30,             0x40041078,__READ_WRITE );
__IO_REG32(    REGFILE31,             0x4004107C,__READ_WRITE );
__IO_REG32(    REGFILE32,             0x40041080,__READ_WRITE );
__IO_REG32(    REGFILE33,             0x40041084,__READ_WRITE );
__IO_REG32(    REGFILE34,             0x40041088,__READ_WRITE );
__IO_REG32(    REGFILE35,             0x4004108C,__READ_WRITE );
__IO_REG32(    REGFILE36,             0x40041090,__READ_WRITE );
__IO_REG32(    REGFILE37,             0x40041094,__READ_WRITE );
__IO_REG32(    REGFILE38,             0x40041098,__READ_WRITE );
__IO_REG32(    REGFILE39,             0x4004109C,__READ_WRITE );
__IO_REG32(    REGFILE40,             0x400410A0,__READ_WRITE );
__IO_REG32(    REGFILE41,             0x400410A4,__READ_WRITE );
__IO_REG32(    REGFILE42,             0x400410A8,__READ_WRITE );
__IO_REG32(    REGFILE43,             0x400410AC,__READ_WRITE );
__IO_REG32(    REGFILE44,             0x400410B0,__READ_WRITE );
__IO_REG32(    REGFILE45,             0x400410B4,__READ_WRITE );
__IO_REG32(    REGFILE46,             0x400410B8,__READ_WRITE );
__IO_REG32(    REGFILE47,             0x400410BC,__READ_WRITE );
__IO_REG32(    REGFILE48,             0x400410C0,__READ_WRITE );
__IO_REG32(    REGFILE49,             0x400410C4,__READ_WRITE );
__IO_REG32(    REGFILE50,             0x400410C8,__READ_WRITE );
__IO_REG32(    REGFILE51,             0x400410CC,__READ_WRITE );
__IO_REG32(    REGFILE52,             0x400410D0,__READ_WRITE );
__IO_REG32(    REGFILE53,             0x400410D4,__READ_WRITE );
__IO_REG32(    REGFILE54,             0x400410D8,__READ_WRITE );
__IO_REG32(    REGFILE55,             0x400410DC,__READ_WRITE );
__IO_REG32(    REGFILE56,             0x400410E0,__READ_WRITE );
__IO_REG32(    REGFILE57,             0x400410E4,__READ_WRITE );
__IO_REG32(    REGFILE58,             0x400410E8,__READ_WRITE );
__IO_REG32(    REGFILE59,             0x400410EC,__READ_WRITE );
__IO_REG32(    REGFILE60,             0x400410F0,__READ_WRITE );
__IO_REG32(    REGFILE61,             0x400410F4,__READ_WRITE );
__IO_REG32(    REGFILE62,             0x400410F8,__READ_WRITE );
__IO_REG32(    REGFILE63,             0x400410FC,__READ_WRITE );
 
/***************************************************************************
 **
 **  UART0
 **
 ***************************************************************************/
/* U0DLL, U0RBR and U0THR share the same address */
__IO_REG32_BIT(U0RBR,                 0x40081000,__READ_WRITE  ,__uartrbr_bits);
#define U0THR       U0RBR
#define U0THR_bit   U0RBR_bit
#define U0DLL       U0RBR
#define U0DLL_bit   U0RBR_bit
/* U0DLM and U0IER share the same address */
__IO_REG32_BIT(U0IER,                 0x40081004,__READ_WRITE ,__uartier0_bits);
#define U0DLM      U0IER

/* U0FCR and U0IIR share the same address */
__IO_REG32_BIT(U0FCR,                 0x40081008,__READ_WRITE ,__uartfcriir_bits);
#define U0IIR      U0FCR
#define U0IIR_bit  U0FCR_bit

__IO_REG32_BIT(U0LCR,                 0x4008100C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U0LSR,                 0x40081014,__READ       ,__uartlsr_bits);
__IO_REG8(     U0SCR,                 0x4008101C,__READ_WRITE);
__IO_REG32_BIT(U0ACR,                 0x40081020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U0FDR,                 0x40081028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U0HDEN,                0x40081040,__READ_WRITE ,__uarthden_bits);
__IO_REG32_BIT(U0SCICTRL,             0x40081048,__READ_WRITE ,__uartscictrl_bits);
__IO_REG32_BIT(U0RS485CTRL,           0x4008104C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U0ADRMATCH,            0x40081050,__READ_WRITE );
__IO_REG8(     U0RS485DLY,            0x40081054,__READ_WRITE );
__IO_REG32_BIT(U0SYNCCTRL,            0x40081058,__READ_WRITE ,__uartsyncctrl_bits);
__IO_REG32_BIT(U0TER,                 0x4008105C,__READ_WRITE ,__uartter_bits);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
/* U1DLL, U1RBR and U1THR share the same address */
__IO_REG32_BIT(U1RBR,                 0x40082000,__READ_WRITE  ,__uartrbr_bits);
#define U1THR         U1RBR
#define U1THR_bit     U1RBR_bit
#define U1DLL         U1RBR
#define U1DLL_bit     U1RBR_bit

/* U1DLM and U1IER share the same address */
__IO_REG32_BIT(U1IER,                 0x40082004,__READ_WRITE ,__uartier1_bits);
#define U1DLM      U1IER

/* U1FCR and U1IIR share the same address */
__IO_REG32_BIT(U1FCR,                 0x40082008,__READ_WRITE ,__uartfcriir_bits);
#define U1IIR      U1FCR
#define U1IIR_bit  U1FCR_bit

__IO_REG32_BIT(U1LCR,                 0x4008200C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U1MCR,                 0x40082010,__READ_WRITE ,__uartmcr_bits);
__IO_REG32_BIT(U1LSR,                 0x40082014,__READ       ,__uart1lsr_bits);
__IO_REG32_BIT(U1MSR,                 0x40082018,__READ       ,__uartmsr_bits);
__IO_REG8(     U1SCR,                 0x4008201C,__READ_WRITE);
__IO_REG32_BIT(U1ACR,                 0x40082020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U1FDR,                 0x40082028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U1TER,                 0x40082030,__READ_WRITE ,__uart1ter_bits);
__IO_REG32_BIT(U1RS485CTRL,           0x4008204C,__READ_WRITE ,__uars1485ctrl_bits);
__IO_REG8(     U1ADRMATCH,            0x40082050,__READ_WRITE );
__IO_REG8(     U1RS485DLY,            0x40082054,__READ_WRITE );
__IO_REG32_BIT(U1FIFOLVL,             0x40082058,__READ_WRITE ,__uartfifolvl_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
/* U2DLL, U2RBR and U2THR share the same address */
__IO_REG32_BIT(U2RBR,                 0x400C1000,__READ_WRITE  ,__uartrbr_bits);
#define U2THR         U2RBR
#define U2THR_bit     U2RBR_bit
#define U2DLL         U2RBR
#define U2DLL_bit     U2RBR_bit

/* U2DLM and U2IER share the same address */
__IO_REG32_BIT(U2IER,                 0x400C1004,__READ_WRITE ,__uartier0_bits);
#define U2DLM      U2IER

/* U2FCR and U2IIR share the same address */
__IO_REG32_BIT(U2FCR,                 0x400C1008,__READ_WRITE ,__uartfcriir_bits);
#define U2IIR      U2FCR
#define U2IIR_bit  U2FCR_bit

__IO_REG32_BIT(U2LCR,                 0x400C100C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U2LSR,                 0x400C1014,__READ       ,__uartlsr_bits);
__IO_REG8(     U2SCR,                 0x400C101C,__READ_WRITE);
__IO_REG32_BIT(U2ACR,                 0x400C1020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U2FDR,                 0x400C1028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U2HDEN,                0x400C1040,__READ_WRITE ,__uarthden_bits);
__IO_REG32_BIT(U2SCICTRL,             0x400C1048,__READ_WRITE ,__uartscictrl_bits);
__IO_REG32_BIT(U2RS485CTRL,           0x400C104C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U2ADRMATCH,            0x400C1050,__READ_WRITE );
__IO_REG8(     U2RS485DLY,            0x400C1054,__READ_WRITE );
__IO_REG32_BIT(U2SYNCCTRL,            0x400C1058,__READ_WRITE ,__uartsyncctrl_bits);
__IO_REG32_BIT(U2TER,                 0x400C105C,__READ_WRITE ,__uartter_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
/* U3DLL, U3RBR and U3THR share the same address */
__IO_REG32_BIT(U3RBR,                 0x400C2000,__READ_WRITE  ,__uartrbr_bits);
#define U3THR         U3RBR
#define U3THR_bit     U3RBR_bit
#define U3DLL         U3RBR
#define U3DLL_bit     U3RBR_bit

/* U3DLM and U3IER share the same address */
__IO_REG32_BIT(U3IER,                 0x400C2004,__READ_WRITE ,__uartier0_bits);
#define U3DLM      U3IER

/* U3FCR and U3IIR share the same address */
__IO_REG32_BIT(U3FCR,                 0x400C2008,__READ_WRITE ,__uartfcriir_bits);
#define U3IIR      U3FCR
#define U3IIR_bit  U3FCR_bit

__IO_REG32_BIT(U3LCR,                 0x400C200C,__READ_WRITE ,__uartlcr_bits);
__IO_REG32_BIT(U3LSR,                 0x400C2014,__READ       ,__uartlsr_bits);
__IO_REG8(     U3SCR,                 0x400C201C,__READ_WRITE);
__IO_REG32_BIT(U3ACR,                 0x400C2020,__READ_WRITE ,__uartacr_bits);
__IO_REG32_BIT(U3ICR,                 0x400C2024,__READ_WRITE ,__uarticr_bits);
__IO_REG32_BIT(U3FDR,                 0x400C2028,__READ_WRITE ,__uartfdr_bits);
__IO_REG32_BIT(U3HDEN,                0x400C2040,__READ_WRITE ,__uarthden_bits);
__IO_REG32_BIT(U3SCICTRL,             0x400C2048,__READ_WRITE ,__uartscictrl_bits);
__IO_REG32_BIT(U3RS485CTRL,           0x400C204C,__READ_WRITE ,__uars485ctrl_bits);
__IO_REG8(     U3ADRMATCH,            0x400C2050,__READ_WRITE );
__IO_REG8(     U3RS485DLY,            0x400C2054,__READ_WRITE );
__IO_REG32_BIT(U3SYNCCTRL,            0x400C2058,__READ_WRITE ,__uartsyncctrl_bits);
__IO_REG32_BIT(U3TER,                 0x400C205C,__READ_WRITE ,__uartter_bits);

/***************************************************************************
 **
 ** SSP0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP0CR0,               0x40083000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP0CR1,               0x40083004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP0DR,                0x40083008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP0SR,                0x4008300C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP0CPSR,              0x40083010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP0IMSC,              0x40083014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP0RIS,               0x40083018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP0MIS,               0x4008301C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP0ICR,               0x40083020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP0DMACR,             0x40083024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSP1CR0,               0x400C5000,__READ_WRITE ,__sspcr0_bits);
__IO_REG32_BIT(SSP1CR1,               0x400C5004,__READ_WRITE ,__sspcr1_bits);
__IO_REG32_BIT(SSP1DR,                0x400C5008,__READ_WRITE ,__sspdr_bits);
__IO_REG32_BIT(SSP1SR,                0x400C500C,__READ       ,__sspsr_bits);
__IO_REG32_BIT(SSP1CPSR,              0x400C5010,__READ_WRITE ,__sspcpsr_bits);
__IO_REG32_BIT(SSP1IMSC,              0x400C5014,__READ_WRITE ,__sspimsc_bits);
__IO_REG32_BIT(SSP1RIS,               0x400C5018,__READ       ,__sspris_bits);
__IO_REG32_BIT(SSP1MIS,               0x400C501C,__READ       ,__sspmis_bits);
__IO_REG32_BIT(SSP1ICR,               0x400C5020,__WRITE      ,__sspicr_bits);
__IO_REG32_BIT(SSP1DMACR,             0x400C5024,__READ_WRITE ,__sspdmacr_bits);

/***************************************************************************
 **
 ** CAN
 **
 ***************************************************************************/
__IO_REG32_BIT(CANCNTL,               0x400E2000,__READ_WRITE ,__cancntl_bits);
__IO_REG32_BIT(CANSTAT,               0x400E2004,__READ_WRITE ,__canstat_bits);
__IO_REG32_BIT(CANEC,                 0x400E2008,__READ       ,__canec_bits);
__IO_REG32_BIT(CANBT,                 0x400E200C,__READ_WRITE ,__canbt_bits);
__IO_REG32_BIT(CANINT,                0x400E2010,__READ       ,__canint_bits);
__IO_REG32_BIT(CANTEST,               0x400E2014,__READ_WRITE ,__cantest_bits);
__IO_REG32_BIT(CANBRPE,               0x400E2018,__READ_WRITE ,__canbrpe_bits);
__IO_REG32_BIT(CANIF1_CMDREQ,         0x400E2020,__READ_WRITE ,__canifx_cmdreq_bits);
__IO_REG32_BIT(CANIF1_CMDMSK,         0x400E2024,__READ_WRITE ,__canifx_cmdmsk_bits);
__IO_REG32_BIT(CANIF1_MSK1,           0x400E2028,__READ_WRITE ,__canifx_msk1_bits);
__IO_REG32_BIT(CANIF1_MSK2,           0x400E202C,__READ_WRITE ,__canifx_msk2_bits);
__IO_REG32_BIT(CANIF1_ARB1,           0x400E2030,__READ_WRITE ,__canifx_arb1_bits);
__IO_REG32_BIT(CANIF1_ARB2,           0x400E2034,__READ_WRITE ,__canifx_arb2_bits);
__IO_REG32_BIT(CANIF1_MCTRL,          0x400E2038,__READ_WRITE ,__canifx_mctrl_bits);
__IO_REG32_BIT(CANIF1_DA1,            0x400E203C,__READ_WRITE ,__canifx_da1_bits);
__IO_REG32_BIT(CANIF1_DA2,            0x400E2040,__READ_WRITE ,__canifx_da2_bits);
__IO_REG32_BIT(CANIF1_DB1,            0x400E2044,__READ_WRITE ,__canifx_db1_bits);
__IO_REG32_BIT(CANIF1_DB2,            0x400E2048,__READ_WRITE ,__canifx_db2_bits);
__IO_REG32_BIT(CANIF2_CMDREQ,         0x400E2080,__READ_WRITE ,__canifx_cmdreq_bits);
__IO_REG32_BIT(CANIF2_CMDMSK,         0x400E2084,__READ_WRITE ,__canifx_cmdmsk_bits);
__IO_REG32_BIT(CANIF2_MSK1,           0x400E2088,__READ_WRITE ,__canifx_msk1_bits);
__IO_REG32_BIT(CANIF2_MSK2,           0x400E208C,__READ_WRITE ,__canifx_msk2_bits);
__IO_REG32_BIT(CANIF2_ARB1,           0x400E2090,__READ_WRITE ,__canifx_arb1_bits);
__IO_REG32_BIT(CANIF2_ARB2,           0x400E2094,__READ_WRITE ,__canifx_arb2_bits);
__IO_REG32_BIT(CANIF2_MCTRL,          0x400E2098,__READ_WRITE ,__canifx_mctrl_bits);
__IO_REG32_BIT(CANIF2_DA1,            0x400E209C,__READ_WRITE ,__canifx_da1_bits);
__IO_REG32_BIT(CANIF2_DA2,            0x400E20A0,__READ_WRITE ,__canifx_da2_bits);
__IO_REG32_BIT(CANIF2_DB1,            0x400E20A4,__READ_WRITE ,__canifx_db1_bits);
__IO_REG32_BIT(CANIF2_DB2,            0x400E20A8,__READ_WRITE ,__canifx_db2_bits);
__IO_REG32_BIT(CANTXREQ1,             0x400E2100,__READ       ,__cantxreq1_bits);
__IO_REG32_BIT(CANTXREQ2,             0x400E2104,__READ       ,__cantxreq2_bits);
__IO_REG32_BIT(CANND1,                0x400E2120,__READ       ,__cannd1_bits);
__IO_REG32_BIT(CANND2,                0x400E2124,__READ       ,__cannd2_bits);
__IO_REG32_BIT(CANIR1,                0x400E2140,__READ       ,__canir1_bits);
__IO_REG32_BIT(CANIR2,                0x400E2144,__READ       ,__canir2_bits);
__IO_REG32_BIT(CANMSGV1,              0x400E2160,__READ       ,__canmsgv1_bits);
__IO_REG32_BIT(CANMSGV2,              0x400E2164,__READ       ,__canmsgv2_bits);
__IO_REG32_BIT(CANCLKDIV,             0x400E2180,__READ_WRITE ,__canclkdiv_bits);  

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(I2SDAO,                0x400A2000,__READ_WRITE ,__i2sdao_bits);
__IO_REG32_BIT(I2SDAI,                0x400A2004,__READ_WRITE ,__i2sdai_bits);
__IO_REG32(    I2STXFIFO,             0x400A2008,__WRITE);
__IO_REG32(    I2SRXFIFO,             0x400A200C,__READ);
__IO_REG32_BIT(I2SSTATE,              0x400A2010,__READ       ,__i2sstate_bits);
__IO_REG32_BIT(I2SDMA1,               0x400A2014,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2SDMA2,               0x400A2018,__READ_WRITE ,__i2sdma_bits);
__IO_REG32_BIT(I2SIRQ,                0x400A201C,__READ_WRITE ,__i2sirq_bits);
__IO_REG32_BIT(I2STXRATE,             0x400A2020,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2SRXRATE,             0x400A2024,__READ_WRITE ,__i2stxrate_bits);
__IO_REG32_BIT(I2STXBITRATE,          0x400A2028,__READ_WRITE ,__i2stxbitrate_bits);
__IO_REG32_BIT(I2SRXBITRATE,          0x400A202C,__READ_WRITE ,__i2srxbitrate_bits);
__IO_REG32_BIT(I2STXMODE,             0x400A2030,__READ_WRITE ,__i2stxmode_bits);
__IO_REG32_BIT(I2SRXMODE,             0x400A2034,__READ_WRITE ,__i2srxmode_bits);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0CONSET,            0x400A1000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C0STAT,              0x400A1004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C0DAT,               0x400A1008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C0ADR0,              0x400A100C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0SCLH,              0x400A1010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C0SCLL,              0x400A1014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C0CONCLR,            0x400A1018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C0MMCTRL,            0x400A101C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C0ADR1,              0x400A1020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR2,              0x400A1024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0ADR3,              0x400A1028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C0DATA_BUFFER,       0x400A102C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C0MASK0,             0x400A1030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK1,             0x400A1034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK2,             0x400A1038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C0MASK3,             0x400A103C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1CONSET,            0x400E0000,__READ_WRITE ,__i2conset_bits);
__IO_REG32_BIT(I2C1STAT,              0x400E0004,__READ       ,__i2stat_bits);
__IO_REG32_BIT(I2C1DAT,               0x400E0008,__READ_WRITE ,__i2dat_bits);
__IO_REG32_BIT(I2C1ADR0,              0x400E000C,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1SCLH,              0x400E0010,__READ_WRITE ,__i2sch_bits);
__IO_REG32_BIT(I2C1SCLL,              0x400E0014,__READ_WRITE ,__i2scl_bits);
__IO_REG32_BIT(I2C1CONCLR,            0x400E0018,__WRITE      ,__i2conclr_bits);
__IO_REG32_BIT(I2C1MMCTRL,            0x400E001C,__READ_WRITE ,__i2cmmctrl_bits);
__IO_REG32_BIT(I2C1ADR1,              0x400E0020,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR2,              0x400E0024,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1ADR3,              0x400E0028,__READ_WRITE ,__i2adr_bits);
__IO_REG32_BIT(I2C1DATA_BUFFER,        0x400E002C,__READ       ,__i2dat_bits);
__IO_REG32_BIT(I2C1MASK0,             0x400E0030,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK1,             0x400E0034,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK2,             0x400E0038,__READ_WRITE ,__i2cmask_bits);
__IO_REG32_BIT(I2C1MASK3,             0x400E003C,__READ_WRITE ,__i2cmask_bits);

/***************************************************************************
 **
 ** ADC0 
 **
 ***************************************************************************/
__IO_REG32_BIT(AD0CR,                 0x400E3000,__READ_WRITE ,__adcr_bits);
__IO_REG32_BIT(AD0GDR,                0x400E3004,__READ_WRITE ,__adgdr_bits);
__IO_REG32_BIT(AD0INTEN,              0x400E300C,__READ_WRITE ,__adinten_bits);
__IO_REG32_BIT(AD0DR0,                0x400E3010,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR1,                0x400E3014,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR2,                0x400E3018,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR3,                0x400E301C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR4,                0x400E3020,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR5,                0x400E3024,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR6,                0x400E3028,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0DR7,                0x400E302C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD0STAT,               0x400E3030,__READ       ,__adstat_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(AD1CR,                 0x400E4000,__READ_WRITE ,__adcr_bits);
__IO_REG32_BIT(AD1GDR,                0x400E4004,__READ_WRITE ,__adgdr_bits);
__IO_REG32_BIT(AD1INTEN,              0x400E400C,__READ_WRITE ,__adinten_bits);
__IO_REG32_BIT(AD1DR0,                0x400E4010,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR1,                0x400E4014,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR2,                0x400E4018,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR3,                0x400E401C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR4,                0x400E4020,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR5,                0x400E4024,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR6,                0x400E4028,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1DR7,                0x400E402C,__READ       ,__addr_bits);
__IO_REG32_BIT(AD1STAT,               0x400E4030,__READ       ,__adstat_bits);

/***************************************************************************
 **
 ** D/A Converter
 **
 ***************************************************************************/
__IO_REG32_BIT(DACR,                  0x400E1000,__READ_WRITE ,__dacr_bits);
__IO_REG32_BIT(DACCTRL,               0x400E1004,__READ_WRITE ,__dacctrl_bits);
__IO_REG32_BIT(DACCNTVAL,             0x400E1008,__READ_WRITE ,__daccntval_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/

#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */


/***************************************************************************
 **
 **  GPDMA Controller peripheral devices lines
 **
 ***************************************************************************/
#define GPDMA_SPIFI             0   /* SPIFI                                   */
#define GPDMA_T0MATCH0          1   /* Timer 0 match 0                         */
#define GPDMA_UART0TX           1   /* UART0 transmit                          */
#define GPDMA_T0MATCH1          2   /* Timer 0 match 1                         */
#define GPDMA_UART0RX           2   /* UART0 receive                           */
#define GPDMA_T1MATCH0          3   /* Timer 1 match 0                         */
#define GPDMA_UART1TX           3   /* UART1 transmit                          */
#define GPDMA_T1MATCH1          4   /* Timer 1 match 1                         */
#define GPDMA_UART1RX           4   /* UART1 receive                           */
#define GPDMA_T2MATCH0          5   /* Timer 2 match 0                         */
#define GPDMA_UART2TX           5   /* UART2 transmit                          */
#define GPDMA_T2MATCH1          6   /* Timer 2 match 1                         */
#define GPDMA_UART2RX           6   /* UART2 receive                           */
#define GPDMA_T3MATCH0          7   /* Timer 3 match 0                         */
#define GPDMA_UART3TX           7   /* UART3 transmit                          */
#define GPDMA_SCT0              7   /* SCT DMA request 0                        */
#define GPDMA_T3MATCH1          8   /* Timer 3 match 1                         */
#define GPDMA_UART3RX           8   /* UART3 receive                           */
#define GPDMA_SCT1              8   /* SCT DMA request 0                        */
#define GPDMA_SSP0RX            9   /* SSP0 receive                            */
#define GPDMA_I2SCH0            9   /* I2S channel 0                           */
#define GPDMA_SSP0TX           10   /* SSP0 transmit                           */
#define GPDMA_I2SCH1           10   /* I2S channel 1                           */
#define GPDMA_SSP1RX           11   /* SSP1 receive                            */
#define GPDMA_SSP1TX           12   /* SSP0 transmit                           */
#define GPDMA_ADC0             13   /* ADC0                                    */
#define GPDMA_ADC1             14   /* ADC1                                    */
#define GPDMA_DAC              15   /* DAC                                     */

/***************************************************************************
 **
 **  NVIC Interrupt channels
 **
 ***************************************************************************/
#define MAIN_STACK             0  /* Main Stack                                             */
#define RESETI                 1  /* Reset                                                  */
#define NMII                   2  /* Non-maskable Interrupt                                 */
#define HFI                    3  /* Hard Fault                                             */
#define MMI                    4  /* Memory Management                                      */
#define BFI                    5  /* Bus Fault                                              */
#define UFI                    6  /* Usage Fault                                            */
#define SVCI                  11  /* SVCall                                                 */
#define DMI                   12  /* Debug Monitor                                          */
#define PSI                   14  /* PendSV                                                 */
#define STI                   15  /* SysTick                                                */
#define NVIC_DAC              16  
#define NVIC_EVNR             17  
#define NVIC_DMA              18
#define NVIC_SDIO             22
#define NVIC_SCT              26
#define NVIC_RITIMER          27
#define NVIC_TIMER0           28
#define NVIC_TIMER1           29
#define NVIC_TIMER2           30
#define NVIC_TIMER3           31
#define NVIC_MC               32
#define NVIC_ADC0             33
#define NVIC_I2C0             34
#define NVIC_I2C1             35
#define NVIC_ADC1             37
#define NVIC_SSP0             38
#define NVIC_SSP2             39
#define NVIC_UART0            40
#define NVIC_UART1            41
#define NVIC_UART2            42
#define NVIC_UART3            43
#define NVIC_I2S              44
#define NVIC_AES              45
#define NVIC_SPIFI            46


/***************************************************************************
 **
 **  EVR Interrupt channels
 **
 ***************************************************************************/
#define EVR_WAKEUP0           0
#define EVR_WAKEUP1           1
#define EVR_WAKEUP2           2
#define EVR_WAKEUP3           3
#define EVR_ATIMER            4
#define EVR_RTC               5
#define EVR_BOD               6
#define EVR_WWDT              7
#define EVR_CCAN              12
#define EVR_TIMER2            13
#define EVR_TIMER6            14
#define EVR_QEI               15
#define EVR_TIMER14           16
#define EVR_RESET             19  

#endif    /* __IOLPC1810_REV_H */

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
Interrupt9   = DAC            0x40
Interrupt10  = ER             0x44
Interrupt11  = DMA            0x48
Interrupt12  = SDIO           0x58
Interrupt13  = SCT            0x68
Interrupt14  = RITIMER        0x6C
Interrupt15  = TIMER0         0x70
Interrupt16  = TIMER1         0x74
Interrupt17  = TIMER2         0x78
Interrupt18  = TIMER3         0x7C
Interrupt19  = MC             0x80
Interrupt20  = ADC0           0x84
Interrupt21  = I2C0           0x88
Interrupt22  = I2C1           0x8C
Interrupt23  = ADC1           0x94
Interrupt24  = SSP0           0x98
Interrupt25  = SSP2           0x9C
Interrupt26  = UART0          0xA0
Interrupt27  = UART1          0xA4
Interrupt28  = UART2          0xA8
Interrupt29  = UART3          0xAC
Interrupt30  = I2S            0xB0
Interrupt31  = SPIFI          0xB8

###DDF-INTERRUPT-END###*/
