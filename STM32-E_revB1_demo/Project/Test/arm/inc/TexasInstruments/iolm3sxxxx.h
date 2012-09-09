/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Luminary LM3Sxxxx (Fury class) devices 
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler
 **
 **    (c) Copyright IAR Systems 2006
 **
 **    $Revision: 48145 $
 **
 ***************************************************************************/

#ifndef __IOLM3SXXXX_H
#define __IOLM3SXXXX_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   LM3SXXXX SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
 ***************************************************************************/


/* C-compiler specific declarations  ***************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
  #pragma system_include
#endif

/* Device Identification 0 (DID0) */
typedef struct {
  __REG32  MINOR          : 8;
  __REG32  MAJOR          : 8;
  __REG32  CLASS          : 8;
  __REG32                 : 4;
  __REG32  VER            : 3;
  __REG32                 : 1;
} __did0_bits;

/* Device Identification 1 (DID1) */
typedef struct {
  __REG32  QUAL           : 2;
  __REG32  ROHS           : 1;
  __REG32  PKG            : 2;
  __REG32  TEMP           : 3;
  __REG32                 : 5;
  __REG32  PINCOUNT       : 3;
  __REG32  PARTNO         : 8;
  __REG32  FAM            : 4;
  __REG32  VER            : 4;
} __did1_bits;

/* Device Capabilities 0 (DC0) */
typedef struct {
  __REG32  FLASHSZ        :16;
  __REG32  SRAMSZ         :16;
} __dc0_bits;

/* Device Capabilities 1 (DC1) */
typedef struct {
  __REG32  JTAG           : 1;
  __REG32  SWD            : 1;
  __REG32  SWO            : 1;
  __REG32  WDT0           : 1;
  __REG32  PLL            : 1;
  __REG32  TEMPSNS        : 1;
  __REG32  HIB            : 1;
  __REG32  MPU            : 1;
  __REG32  MAXADC0SPD     : 2;
  __REG32  MAXADC1SPD     : 2;
  __REG32  MINSYSDIV      : 4;
  __REG32  ADC0           : 1;
  __REG32  ADC1           : 1;
  __REG32                 : 2;
  __REG32  PWM            : 1;
  __REG32                 : 3;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32  CAN2           : 1;
  __REG32                 : 1;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __dc1_bits;

/* Device Capabilities 2 (DC2) */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32                 : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32                 : 2;
  __REG32  QEI0           : 1;
  __REG32  QEI1           : 1;
  __REG32                 : 2;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 : 4;
  __REG32  COMP0          : 1;
  __REG32  COMP1          : 1;
  __REG32  COMP2          : 1;
  __REG32                 : 1;
  __REG32  I2S0           : 1;
  __REG32                 : 1;
  __REG32  EPI0           : 1;
  __REG32                 : 1;
} __dc2_bits;

/* Device Capabilities 3 (DC3) */
typedef struct {
  __REG32  PWM0           : 1;
  __REG32  PWM1           : 1;
  __REG32  PWM2           : 1;
  __REG32  PWM3           : 1;
  __REG32  PWM4           : 1;
  __REG32  PWM5           : 1;
  __REG32  C0MINUS        : 1;
  __REG32  C0PLUS         : 1; 
  __REG32  C0O            : 1;   
  __REG32  C1MINUS        : 1;
  __REG32  C1PLUS         : 1; 
  __REG32  C1O            : 1;
  __REG32  C2MINUS        : 1;
  __REG32  C2PLUS         : 1; 
  __REG32  C2O            : 1; 
  __REG32  PWMFAULT       : 1;
  __REG32  ADC0AIN0       : 1;
  __REG32  ADC0AIN1       : 1;
  __REG32  ADC0AIN2       : 1;
  __REG32  ADC0AIN3       : 1;
  __REG32  ADC0AIN4       : 1;
  __REG32  ADC0AIN5       : 1;
  __REG32  ADC0AIN6       : 1;
  __REG32  ADC0AIN7       : 1;
  __REG32  CCP0           : 1;
  __REG32  CCP1           : 1;
  __REG32  CCP2           : 1;
  __REG32  CCP3           : 1;
  __REG32  CCP4           : 1;
  __REG32  CCP5           : 1;
  __REG32                 : 1;
  __REG32  _32KHZ         : 1;  
} __dc3_bits;

/* Device Capabilities 4 (DC4) */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;   
  __REG32  GPIOH          : 1;       
  __REG32  GPIOJ          : 1;
  __REG32                 : 3;
  __REG32  ROM            : 1;
  __REG32  UDMA           : 1;
  __REG32  CCP6           : 1;
  __REG32  CCP7           : 1;
  __REG32                 : 2;
  __REG32  PICAL          : 1;
  __REG32                 : 5;
  __REG32  E1588          : 1;
  __REG32                 : 3;
  __REG32  EMAC0          : 1;
  __REG32                 : 1;
  __REG32  EPHY0          : 1;
  __REG32                 : 1;
} __dc4_bits;

/* Device Capabilities 5 (DC5) */
typedef struct {
  __REG32  PWM0           : 1;
  __REG32  PWM1           : 1;
  __REG32  PWM2           : 1;
  __REG32  PWM3           : 1;
  __REG32  PWM4           : 1;
  __REG32  PWM5           : 1;
  __REG32  PWM6           : 1;   
  __REG32  PWM7           : 1;       
  __REG32                 :12;
  __REG32  PWMESYNC       : 1;
  __REG32  PWMEFLT        : 1;
  __REG32                 : 2;
  __REG32  PWMFAULT0      : 1;
  __REG32  PWMFAULT1      : 1;
  __REG32  PWMFAULT2      : 1;
  __REG32  PWMFAULT3      : 1;
  __REG32                 : 4;
} __dc5_bits;

/* Device Capabilities 6 (DC6) */
typedef struct {
  __REG32  USB0           : 2;
  __REG32                 : 2;
  __REG32  USB0PHY        : 1;
  __REG32                 :27;
} __dc6_bits;

/* Device Capabilities 7 (DC7) */
typedef struct {
  __REG32  DMACH0         : 1;
  __REG32  DMACH1         : 1;
  __REG32  DMACH2         : 1;
  __REG32  DMACH3         : 1;
  __REG32  DMACH4         : 1;
  __REG32  DMACH5         : 1;
  __REG32  DMACH6         : 1;
  __REG32  DMACH7         : 1;
  __REG32  DMACH8         : 1;
  __REG32  DMACH9         : 1;
  __REG32  DMACH10        : 1;
  __REG32  DMACH11        : 1;
  __REG32  DMACH12        : 1;
  __REG32  DMACH13        : 1;
  __REG32  DMACH14        : 1;
  __REG32  DMACH15        : 1;
  __REG32  DMACH16        : 1;
  __REG32  DMACH17        : 1;
  __REG32  DMACH18        : 1;
  __REG32  DMACH19        : 1;
  __REG32  DMACH20        : 1;
  __REG32  DMACH21        : 1;
  __REG32  DMACH22        : 1;
  __REG32  DMACH23        : 1;
  __REG32  DMACH24        : 1;
  __REG32  DMACH25        : 1;
  __REG32  DMACH26        : 1;
  __REG32  DMACH27        : 1;
  __REG32  DMACH28        : 1;
  __REG32  DMACH29        : 1;
  __REG32  DMACH30        : 1;
  __REG32                 : 1;
} __dc7_bits;

/* Device Capabilities 8 (DC8) */
typedef struct {
  __REG32  ADC0AIN0       : 1;
  __REG32  ADC0AIN1       : 1;
  __REG32  ADC0AIN2       : 1;
  __REG32  ADC0AIN3       : 1;
  __REG32  ADC0AIN4       : 1;
  __REG32  ADC0AIN5       : 1;
  __REG32  ADC0AIN6       : 1;
  __REG32  ADC0AIN7       : 1;
  __REG32  ADC0AIN8       : 1;
  __REG32  ADC0AIN9       : 1;
  __REG32  ADC0AIN10      : 1;
  __REG32  ADC0AIN11      : 1;
  __REG32  ADC0AIN12      : 1;
  __REG32  ADC0AIN13      : 1;
  __REG32  ADC0AIN14      : 1;
  __REG32  ADC0AIN15      : 1;
  __REG32  ADC1AIN0       : 1;
  __REG32  ADC1AIN1       : 1;
  __REG32  ADC1AIN2       : 1;
  __REG32  ADC1AIN3       : 1;
  __REG32  ADC1AIN4       : 1;
  __REG32  ADC1AIN5       : 1;
  __REG32  ADC1AIN6       : 1;
  __REG32  ADC1AIN7       : 1;
  __REG32  ADC1AIN8       : 1;
  __REG32  ADC1AIN9       : 1;
  __REG32  ADC1AIN10      : 1;
  __REG32  ADC1AIN11      : 1;
  __REG32  ADC1AIN12      : 1;
  __REG32  ADC1AIN13      : 1;
  __REG32  ADC1AIN14      : 1;
  __REG32  ADC1AIN15      : 1;
} __dc8_bits;

/* Device Capabilities 9 (DC9) */
typedef struct {
  __REG32  ADC0DC0        : 1;
  __REG32  ADC0DC1        : 1;
  __REG32  ADC0DC2        : 1;
  __REG32  ADC0DC3        : 1;
  __REG32  ADC0DC4        : 1;
  __REG32  ADC0DC5        : 1;
  __REG32  ADC0DC6        : 1;
  __REG32  ADC0DC7        : 1;
  __REG32                 : 8;
  __REG32  ADC1DC0        : 1;
  __REG32  ADC1DC1        : 1;
  __REG32  ADC1DC2        : 1;
  __REG32  ADC1DC3        : 1;
  __REG32  ADC1DC4        : 1;
  __REG32  ADC1DC5        : 1;
  __REG32  ADC1DC6        : 1;
  __REG32  ADC1DC7        : 1;
  __REG32                 : 8;
} __dc9_bits;

/* Non-Volatile Memory Information (NVMSTAT) */
typedef struct {
  __REG32  FWB            : 1;
  __REG32                 : 3;
  __REG32  TPSW           : 1;
  __REG32                 :27;
} __nvmstat_bits;

/* Power-On and Brown-Out Reset Control (PBORCTL) */
typedef struct {
  __REG32                 : 1;
  __REG32  BORIOR         : 1;
  __REG32                 :30;
} __pborctl_bits;

/* LDO Power Control (LDOPCTL) */
typedef struct {
  __REG32  VADJ           : 6;
  __REG32                 :26;
} __ldopctl_bits;

/* Software Reset Control 0 (SRCR0) */
typedef struct {
  __REG32                 : 3;
  __REG32  WDT0           : 1;
  __REG32                 : 2;
  __REG32  HIB            : 1;
  __REG32                 : 9;
  __REG32  ADC0           : 1;
  __REG32  ADC1           : 1;
  __REG32                 : 2;
  __REG32  PWM            : 1;
  __REG32                 : 3;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32  CAN2           : 1;
  __REG32                 : 1;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __srcr0_bits;

/* Software Reset Control 1 (SRCR1) */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32                 : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32                 : 2;
  __REG32  QEI0           : 1;
  __REG32  QEI1           : 1;
  __REG32                 : 2;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 : 4;
  __REG32  COMP0          : 1;
  __REG32  COMP1          : 1;
  __REG32  COMP2          : 1;
  __REG32                 : 1;
  __REG32  I2S0           : 1;
  __REG32                 : 1;
  __REG32  EPI0           : 1;
  __REG32                 : 1;
} __srcr1_bits;

/* Software Reset Control 2 (SRCR2) */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;   
  __REG32  GPIOH          : 1;       
  __REG32  GPIOJ          : 1;       
  __REG32                 : 4;
  __REG32  UDMA           : 1;
  __REG32                 : 2;
  __REG32  USB0           : 1;
  __REG32                 :11;
  __REG32  EMAC0          : 1;       
  __REG32                 : 1;
  __REG32  EPHY0          : 1;       
  __REG32                 : 1;
} __srcr2_bits;

/* Raw Interrupt Status (RIS) */
typedef struct {
  __REG32                 : 1;
  __REG32  BORRIS         : 1;
  __REG32                 : 4;
  __REG32  PLLLRIS        : 1;
  __REG32  USBPLLLRIS     : 1;
  __REG32  MOSCPUPRIS     : 1;
  __REG32                 :23;
} __ris_bits;

/* Interrupt Mask Control (IMC) */
typedef struct {
  __REG32                 : 1;
  __REG32  BORIM          : 1;
  __REG32                 : 4;
  __REG32  PLLLIM         : 1;
  __REG32  USBPLLLIM      : 1;
  __REG32  MOSCPUPIM      : 1;
  __REG32                 :23;
} __imc_bits;

/* Masked Interrupt Status and Clear (MISC) */
typedef struct {
  __REG32                 : 1;
  __REG32  BORMIS         : 1;
  __REG32                 : 4;
  __REG32  PLLLMIS        : 1;
  __REG32  USBPLLLMIS     : 1;
  __REG32  MOSCPUPMIS     : 1;
  __REG32                 :23;
} __misc_bits;

/* Reset Cause (RESC) */
typedef struct {
  __REG32  EXT            : 1;
  __REG32  POR            : 1;
  __REG32  BOR            : 1;
  __REG32  WDT0           : 1;
  __REG32  SW             : 1;
  __REG32  WDT1           : 1;
  __REG32                 :10;
  __REG32  MOSCFAIL       : 1;
  __REG32                 :15;
} __resc_bits;

/* GPIO High-Performance Bus Control (GPIOHBCTL) */
typedef struct {
  __REG32  PORTA          : 1;
  __REG32  PORTB          : 1;
  __REG32  PORTC          : 1;
  __REG32  PORTD          : 1;
  __REG32  PORTE          : 1;
  __REG32  PORTF          : 1;
  __REG32  PORTG          : 1;
  __REG32  PORTH          : 1;
  __REG32  PORTJ          : 1;
  __REG32                 :23;
} __gpiohbctl_bits;

/* Run-Mode Clock Configuration (RCC) */
typedef struct {
  __REG32  MOSCDIS        : 1;
  __REG32  IOSCDIS        : 1;
  __REG32                 : 2;
  __REG32  OSCSRC         : 2;
  __REG32  XTAL           : 5;
  __REG32  BYPASS         : 1;
  __REG32                 : 1;
  __REG32  PWRDN          : 1;
  __REG32                 : 3;
  __REG32  PWMDIV         : 3;
  __REG32  USEPWMDIV      : 1;
  __REG32                 : 1;
  __REG32  USESYSDIV      : 1;
  __REG32  SYSDIV         : 4;
  __REG32  ACG            : 1;
  __REG32                 : 4;
} __rcc_bits;

/* XTAL to PLL Translation (PLLCFG) */
typedef struct {
  __REG32  R              : 5;
  __REG32  F              : 9;
  __REG32                 :18;
} __pllcfg_bits;

/* Run-Mode Clock Configuration 2 (RCC2) */
typedef struct {
  __REG32                 : 4;
  __REG32  OSCSRC2        : 3;
  __REG32                 : 4;
  __REG32  BYPASS2        : 1;
  __REG32                 : 1;
  __REG32  PWRDN2         : 1; 
  __REG32  USBPWRDN       : 1; 
  __REG32                 : 7;
  __REG32  SYSDIV2LSB     : 1;
  __REG32  SYSDIV2        : 6;
  __REG32                 : 1;
  __REG32  DIV400         : 1;
  __REG32  USERCC2        : 1;
} __rcc2_bits;

/* Main Oscillator Control (MOSCCTL) */
typedef struct {
  __REG32  CVAL           : 1;
  __REG32                 :31;
} __moscctl_bits;

/* Deep Sleep Clock Configuration (DSLPCLKCFG) */
typedef struct {
  __REG32                 : 4;
  __REG32  DSOSCSRC       : 3;
  __REG32                 :16;
  __REG32  DSDIVORIDE     : 6;
  __REG32                 : 3;
} __dslpclkcfg_bits;

/* Precision Internal Oscillator Calibration (PIOSCCAL) */
typedef struct {
  __REG32  UT             : 7;
  __REG32                 : 1;
  __REG32  UPDATE         : 1;
  __REG32  CAL            : 1;
  __REG32                 :21;
  __REG32  UTEN           : 1;
} __piosccal_bits;

/* Precision Internal Oscillator Statistics (PIOSCSTAT) */
typedef struct {
  __REG32  CT             : 7;
  __REG32                 : 1;
  __REG32  RESULT         : 2;
  __REG32                 : 6;
  __REG32  DT             : 7;
  __REG32                 : 9;
} __pioscstat_bits;

/* I2S MCLK Configuration (I2SMCLKCFG) */
typedef struct {
  __REG32  TXF            : 4;
  __REG32  TXI            :10;
  __REG32                 : 1;
  __REG32  TXEN           : 1;
  __REG32  RXF            : 4;
  __REG32  RXI            :10;
  __REG32                 : 1;
  __REG32  RXEN           : 1;
} __i2smclkcfg_bits;

/* Run-Mode Clock Gating Control 0 (RCGC0) 
  Sleep-Mode Clock Gating Control 0 (SCGC0) */
typedef struct {
  __REG32                 : 3;
  __REG32  WDT0           : 1;
  __REG32                 : 2;
  __REG32  HIB            : 1;
  __REG32                 : 1;
  __REG32  MAXADC0SPD     : 2;
  __REG32  MAXADC1SPD     : 2;
  __REG32                 : 4;
  __REG32  ADC0           : 1;
  __REG32  ADC1           : 1;
  __REG32                 : 2;
  __REG32  PWM            : 1;
  __REG32                 : 3;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32  CAN2           : 1;
  __REG32                 : 1;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __rcgc0_bits;

/* Deep-Sleep-Mode Clock Gating Control 0 (DCGC0)  */
typedef struct {
  __REG32                 : 3;
  __REG32  WDT0           : 1;
  __REG32                 : 2;
  __REG32  HIB            : 1;
  __REG32                 : 9;
  __REG32  ADC0           : 1;
  __REG32  ADC1           : 1;
  __REG32                 : 2;
  __REG32  PWM            : 1;
  __REG32                 : 3;
  __REG32  CAN0           : 1;
  __REG32  CAN1           : 1;
  __REG32  CAN2           : 1;
  __REG32                 : 1;
  __REG32  WDT1           : 1;
  __REG32                 : 3;
} __dcgc0_bits;

/* Run-Mode Clock Gating Control 1 (RCGC1)
   Sleep-Mode Clock Gating Control 1 (SCGC1)
   Deep-Sleep-Mode Clock Gating Control 1 (DCGC1) */
typedef struct {
  __REG32  UART0          : 1;
  __REG32  UART1          : 1;
  __REG32  UART2          : 1;
  __REG32                 : 1;
  __REG32  SSI0           : 1;
  __REG32  SSI1           : 1;
  __REG32                 : 2;
  __REG32  QEI0           : 1;
  __REG32  QEI1           : 1;
  __REG32                 : 2;
  __REG32  I2C0           : 1;
  __REG32                 : 1;
  __REG32  I2C1           : 1;
  __REG32                 : 1;
  __REG32  TIMER0         : 1;
  __REG32  TIMER1         : 1;
  __REG32  TIMER2         : 1;
  __REG32  TIMER3         : 1;
  __REG32                 : 4;
  __REG32  COMP0          : 1;
  __REG32  COMP1          : 1;
  __REG32  COMP2          : 1;
  __REG32                 : 1;
  __REG32  I2S0           : 1;
  __REG32                 : 1;
  __REG32  EPI0           : 1;
  __REG32                 : 1;
} __rcgc1_bits;

/* Run-Mode Clock Gating Control 2 (RCGC2)
   Sleep-Mode Clock Gating Control 2 (SCGC2)
   Deep-Sleep-Mode Clock Gating Control 2 (DCGC2) */
typedef struct {
  __REG32  GPIOA          : 1;
  __REG32  GPIOB          : 1;
  __REG32  GPIOC          : 1;
  __REG32  GPIOD          : 1;
  __REG32  GPIOE          : 1;
  __REG32  GPIOF          : 1;
  __REG32  GPIOG          : 1;   
  __REG32  GPIOH          : 1;       
  __REG32  GPIOJ          : 1;       
  __REG32                 : 4;
  __REG32  UDMA           : 1;       
  __REG32                 : 2;
  __REG32  USB0           : 1;       
  __REG32                 :11;
  __REG32  EMAC0          : 1;       
  __REG32                 : 1;
  __REG32  EPHY0          : 1;       
  __REG32                 : 1;
} __rcgc2_bits;

/* Hibernation Module RTC Counter (HIBRTCC) */
typedef struct {
  __REG32  RTCC           :32;  
} __hibrtcc_bits;

/* Hibernation Module RTC Match 0 (HIBRTCM0) */
typedef struct {
  __REG32  RTCM0          :32;  
} __hibrtcm0_bits;

/* Hibernation Module RTC Match 1 (HIBRTCM1) */
typedef struct {
  __REG32  RTCM1          :32;  
} __hibrtcm1_bits;

/* Hibernation Module RTC Load (HIBRTCLD) */
typedef struct {
  __REG32  RTCLD          :32;  
} __hibrtcld_bits;

/* Hibernation Module Control (HIBCTL) */
typedef struct {
  __REG32  RTCEN          : 1; 
  __REG32  HIBREQ         : 1;   
  __REG32  CLKSEL         : 1;  
  __REG32  RTCWEN         : 1;    
  __REG32  PINWEN         : 1;
  __REG32  LOWBATEN       : 1;
  __REG32  CLK32EN        : 1;    
  __REG32  VABORT         : 1;  
  __REG32  VDD3ON         : 1;  
  __REG32                 :22;            
  __REG32  WRC            : 1;  
} __hibctl_bits;

/* Hibernation Module Interrupt Mask (HIBIM) */
typedef struct {
  __REG32  RTCALT0        : 1;
  __REG32  RTCALT1        : 1;    
  __REG32  LOWBAT         : 1;     
  __REG32  EXTW           : 1;
  __REG32                 :28;         
} __hibim_bits;

/* Hibernation Module RTC Trim (HIBRTCT) */
typedef struct {
  __REG32  TRIM           :16;  
  __REG32                 :16;   
} __hibrtct_bits;

/* Hibernation Module Data (HIBDATA) */
typedef struct {
  __REG32  RTD            :32;  
} __hibdata_bits;

/* Flash Memory Protection Read Enable (FMPRE)
   Flash Memory Protection Program Enable (FMPPE) */
typedef struct {
  __REG32  BLOCK0         : 1;
  __REG32  BLOCK1         : 1;
  __REG32  BLOCK2         : 1;
  __REG32  BLOCK3         : 1;
  __REG32  BLOCK4         : 1;
  __REG32  BLOCK5         : 1;
  __REG32  BLOCK6         : 1;
  __REG32  BLOCK7         : 1;
  __REG32  BLOCK8         : 1;
  __REG32  BLOCK9         : 1;
  __REG32  BLOCK10        : 1;
  __REG32  BLOCK11        : 1;
  __REG32  BLOCK12        : 1;
  __REG32  BLOCK13        : 1;
  __REG32  BLOCK14        : 1;
  __REG32  BLOCK15        : 1;
  __REG32  BLOCK16        : 1;
  __REG32  BLOCK17        : 1;
  __REG32  BLOCK18        : 1;
  __REG32  BLOCK19        : 1;
  __REG32  BLOCK20        : 1;
  __REG32  BLOCK21        : 1;
  __REG32  BLOCK22        : 1;
  __REG32  BLOCK23        : 1;
  __REG32  BLOCK24        : 1;
  __REG32  BLOCK25        : 1;
  __REG32  BLOCK26        : 1;
  __REG32  BLOCK27        : 1;
  __REG32  BLOCK28        : 1;
  __REG32  BLOCK29        : 1;
  __REG32  BLOCK30        : 1;
  __REG32  BLOCK31        : 1;
} __fmpre_bits;

/* Flash Memory Address (FMA) */
typedef struct {
  __REG32  OFFSET         :19;
  __REG32                 :13;
} __fma_bits;

/* Flash Memory Control (FMC) */
typedef struct {
  __REG32  WRITE          : 1;
  __REG32  ERASE          : 1;
  __REG32  MERASE         : 1;
  __REG32  COMT           : 1;
  __REG32                 :12;
  __REG32  WRKEY          :16;
} __fmc_bits;

/* Flash Controller Raw Interrupt Status (FCRIS) */
typedef struct {
  __REG32  ARIS           : 1;
  __REG32  PRIS           : 1;
  __REG32                 :30;
} __fcris_bits;

/* Flash Controller Interrupt Mask (FCIM) */
typedef struct {
  __REG32  AMASK          : 1;
  __REG32  PMASK          : 1;
  __REG32                 :30;
} __fcim_bits;

/* Flash Controller Masked Interrupt Status and Clear (FCMISC) */
typedef struct {
  __REG32  AMISC          : 1;
  __REG32  PMISC          : 1;
  __REG32                 :30;
} __fcmisc_bits;

/* Flash Memory Control 2 (FMC2) */
typedef struct {
  __REG32  WRBUF          : 1;
  __REG32                 :15;
  __REG32  WRKEY          :16;
} __fmc2_bits;

/* Flash Control (FCTL) */
typedef struct {
  __REG32  USDREQ         : 1;
  __REG32  USDACK         : 1;
  __REG32                 :30;
} __fctl_bits;

/* ROM Control (RMCTL) */
typedef struct {
  __REG32  BA             : 1;
  __REG32                 :31;
} __rmctl_bits;

/* ROM Version Register (RMVER) */
typedef struct {
  __REG32  REV            : 8;
  __REG32  VER            : 8;
  __REG32  SIZE           : 8;
  __REG32  CONT           : 8;
} __rmver_bits;

/* USec Reload (USECRL) */
typedef struct {
  __REG32  USEC           : 8;
  __REG32                 :24;
} __usecrl_bits;

/* User Debug (USER_DBG) */
typedef union {
  /*USER_DBG*/
  struct {
    __REG32  DBG0           : 1;
    __REG32  DBG1           : 1;
    __REG32  DATA           :29;
    __REG32  NW             : 1;
  };
  /*BOOTCFG*/
  struct {
    __REG32  DBG0           : 1;
    __REG32  DBG1           : 1;
    __REG32                 : 6;
    __REG32  EN             : 1;
    __REG32  POL            : 1;
    __REG32  PIN            : 3;
    __REG32  PORT           : 3;
    __REG32                 :15;
    __REG32  NW             : 1;
  } __bootcfg;
} __user_dbg_bits;

/* User Register 0 (USER_REG0) */
typedef struct {
  __REG32  DATA           :31;
  __REG32  NW             : 1;
} __user_reg_bits;

/* DMA Status (DMASTAT) */
typedef struct {
  __REG32  MASTEN         : 1;
  __REG32                 : 3;
  __REG32  STATE          : 4;
  __REG32                 : 8;
  __REG32  DMACHANS       : 5;
  __REG32                 :11;
} __dmastat_bits;

/* DMA Configuration (DMACFG) */
typedef struct {
  __REG32  MASTEN         : 1;
  __REG32                 :31;
} __dmacfg_bits;

/* DMA Channel Wait-on-Request Status (DMAWAITSTAT) */
typedef struct {
  __REG32  WAITREQ0       : 1;
  __REG32  WAITREQ1       : 1;
  __REG32  WAITREQ2       : 1;
  __REG32  WAITREQ3       : 1;
  __REG32  WAITREQ4       : 1;
  __REG32  WAITREQ5       : 1;
  __REG32  WAITREQ6       : 1;
  __REG32  WAITREQ7       : 1;
  __REG32  WAITREQ8       : 1;
  __REG32  WAITREQ9       : 1;
  __REG32  WAITREQ10      : 1;
  __REG32  WAITREQ11      : 1;
  __REG32  WAITREQ12      : 1;
  __REG32  WAITREQ13      : 1;
  __REG32  WAITREQ14      : 1;
  __REG32  WAITREQ15      : 1;
  __REG32  WAITREQ16      : 1;
  __REG32  WAITREQ17      : 1;
  __REG32  WAITREQ18      : 1;
  __REG32  WAITREQ19      : 1;
  __REG32  WAITREQ20      : 1;
  __REG32  WAITREQ21      : 1;
  __REG32  WAITREQ22      : 1;
  __REG32  WAITREQ23      : 1;
  __REG32  WAITREQ24      : 1;
  __REG32  WAITREQ25      : 1;
  __REG32  WAITREQ26      : 1;
  __REG32  WAITREQ27      : 1;
  __REG32  WAITREQ28      : 1;
  __REG32  WAITREQ29      : 1;
  __REG32  WAITREQ30      : 1;
  __REG32  WAITREQ31      : 1;
} __dmawaitstat_bits;

/* DMA Channel Software Request (DMASWREQ) */
typedef struct {
  __REG32  SWREQ0         : 1;
  __REG32  SWREQ1         : 1;
  __REG32  SWREQ2         : 1;
  __REG32  SWREQ3         : 1;
  __REG32  SWREQ4         : 1;
  __REG32  SWREQ5         : 1;
  __REG32  SWREQ6         : 1;
  __REG32  SWREQ7         : 1;
  __REG32  SWREQ8         : 1;
  __REG32  SWREQ9         : 1;
  __REG32  SWREQ10        : 1;
  __REG32  SWREQ11        : 1;
  __REG32  SWREQ12        : 1;
  __REG32  SWREQ13        : 1;
  __REG32  SWREQ14        : 1;
  __REG32  SWREQ15        : 1;
  __REG32  SWREQ16        : 1;
  __REG32  SWREQ17        : 1;
  __REG32  SWREQ18        : 1;
  __REG32  SWREQ19        : 1;
  __REG32  SWREQ20        : 1;
  __REG32  SWREQ21        : 1;
  __REG32  SWREQ22        : 1;
  __REG32  SWREQ23        : 1;
  __REG32  SWREQ24        : 1;
  __REG32  SWREQ25        : 1;
  __REG32  SWREQ26        : 1;
  __REG32  SWREQ27        : 1;
  __REG32  SWREQ28        : 1;
  __REG32  SWREQ29        : 1;
  __REG32  SWREQ30        : 1;
  __REG32  SWREQ31        : 1;
} __dmaswreq_bits;

/* DMA Channel Useburst Set (DMAUSEBURSTSET) 
   DMA Channel Request Mask Set (DMAREQMASKSET)
   DMA Channel Enable Set (DMAENASET)
   DMA Channel Primary Alternate Set (DMAALTSET)
   DMA Channel Priority Set (DMAPRIOSET) */
typedef struct {
  __REG32  SET0         : 1;
  __REG32  SET1         : 1;
  __REG32  SET2         : 1;
  __REG32  SET3         : 1;
  __REG32  SET4         : 1;
  __REG32  SET5         : 1;
  __REG32  SET6         : 1;
  __REG32  SET7         : 1;
  __REG32  SET8         : 1;
  __REG32  SET9         : 1;
  __REG32  SET10        : 1;
  __REG32  SET11        : 1;
  __REG32  SET12        : 1;
  __REG32  SET13        : 1;
  __REG32  SET14        : 1;
  __REG32  SET15        : 1;
  __REG32  SET16        : 1;
  __REG32  SET17        : 1;
  __REG32  SET18        : 1;
  __REG32  SET19        : 1;
  __REG32  SET20        : 1;
  __REG32  SET21        : 1;
  __REG32  SET22        : 1;
  __REG32  SET23        : 1;
  __REG32  SET24        : 1;
  __REG32  SET25        : 1;
  __REG32  SET26        : 1;
  __REG32  SET27        : 1;
  __REG32  SET28        : 1;
  __REG32  SET29        : 1;
  __REG32  SET30        : 1;
  __REG32  SET31        : 1;
} __dmauseburstset_bits;

/* DMA Channel Useburst Clear (DMAUSEBURSTCLR)
   DMA Channel Request Mask Clear (DMAREQMASKCLR)
   DMA Channel Enable Clear (DMAENACLR)
   DMA Channel Primary Alternate Clear (DMAALTCLR)
   DMA Channel Priority Clear (DMAPRIOCLR) */
typedef struct {
  __REG32  CLR0         : 1;
  __REG32  CLR1         : 1;
  __REG32  CLR2         : 1;
  __REG32  CLR3         : 1;
  __REG32  CLR4         : 1;
  __REG32  CLR5         : 1;
  __REG32  CLR6         : 1;
  __REG32  CLR7         : 1;
  __REG32  CLR8         : 1;
  __REG32  CLR9         : 1;
  __REG32  CLR10        : 1;
  __REG32  CLR11        : 1;
  __REG32  CLR12        : 1;
  __REG32  CLR13        : 1;
  __REG32  CLR14        : 1;
  __REG32  CLR15        : 1;
  __REG32  CLR16        : 1;
  __REG32  CLR17        : 1;
  __REG32  CLR18        : 1;
  __REG32  CLR19        : 1;
  __REG32  CLR20        : 1;
  __REG32  CLR21        : 1;
  __REG32  CLR22        : 1;
  __REG32  CLR23        : 1;
  __REG32  CLR24        : 1;
  __REG32  CLR25        : 1;
  __REG32  CLR26        : 1;
  __REG32  CLR27        : 1;
  __REG32  CLR28        : 1;
  __REG32  CLR29        : 1;
  __REG32  CLR30        : 1;
  __REG32  CLR31        : 1;
} __dmauseburstclr_bits;

/* DMA Bus Error Clear (DMAERRCLR) */
typedef struct {
  __REG32  ERRCLR       : 1;
  __REG32               :31;
} __dmaerrclr_bits;

/* DMA Channel Alternate Select (DMACHALT) */
typedef union {
    /*DMACHALT*/
    struct {
    __REG32  CHALT0         : 1;
    __REG32  CHALT1         : 1;
    __REG32  CHALT2         : 1;
    __REG32  CHALT3         : 1;
    __REG32  CHALT4         : 1;
    __REG32  CHALT5         : 1;
    __REG32  CHALT6         : 1;
    __REG32  CHALT7         : 1;
    __REG32  CHALT8         : 1;
    __REG32  CHALT9         : 1;
    __REG32  CHALT10        : 1;
    __REG32  CHALT11        : 1;
    __REG32  CHALT12        : 1;
    __REG32  CHALT13        : 1;
    __REG32  CHALT14        : 1;
    __REG32  CHALT15        : 1;
    __REG32  CHALT16        : 1;
    __REG32  CHALT17        : 1;
    __REG32  CHALT18        : 1;
    __REG32  CHALT19        : 1;
    __REG32  CHALT20        : 1;
    __REG32  CHALT21        : 1;
    __REG32  CHALT22        : 1;
    __REG32  CHALT23        : 1;
    __REG32  CHALT24        : 1;
    __REG32  CHALT25        : 1;
    __REG32  CHALT26        : 1;
    __REG32  CHALT27        : 1;
    __REG32  CHALT28        : 1;
    __REG32  CHALT29        : 1;
    __REG32  CHALT30        : 1;
    __REG32  CHALT31        : 1;
  };
  /*DMACHASGN*/
  struct {
    __REG32  CHASGN0        : 1;
    __REG32  CHASGN1        : 1;
    __REG32  CHASGN2        : 1;
    __REG32  CHASGN3        : 1;
    __REG32  CHASGN4        : 1;
    __REG32  CHASGN5        : 1;
    __REG32  CHASGN6        : 1;
    __REG32  CHASGN7        : 1;
    __REG32  CHASGN8        : 1;
    __REG32  CHASGN9        : 1;
    __REG32  CHASGN10       : 1;
    __REG32  CHASGN11       : 1;
    __REG32  CHASGN12       : 1;
    __REG32  CHASGN13       : 1;
    __REG32  CHASGN14       : 1;
    __REG32  CHASGN15       : 1;
    __REG32  CHASGN16       : 1;
    __REG32  CHASGN17       : 1;
    __REG32  CHASGN18       : 1;
    __REG32  CHASGN19       : 1;
    __REG32  CHASGN20       : 1;
    __REG32  CHASGN21       : 1;
    __REG32  CHASGN22       : 1;
    __REG32  CHASGN23       : 1;
    __REG32  CHASGN24       : 1;
    __REG32  CHASGN25       : 1;
    __REG32  CHASGN26       : 1;
    __REG32  CHASGN27       : 1;
    __REG32  CHASGN28       : 1;
    __REG32  CHASGN29       : 1;
    __REG32  CHASGN30       : 1;
    __REG32  CHASGN31       : 1;
  };
} __dmachalt_bits;

/* DMA Bus Error Clear (DMACHIS) */
typedef struct {
  __REG32  CHIS0            : 1;
  __REG32  CHIS1            : 1;
  __REG32  CHIS2            : 1;
  __REG32  CHIS3            : 1;
  __REG32  CHIS4            : 1;
  __REG32  CHIS5            : 1;
  __REG32  CHIS6            : 1;
  __REG32  CHIS7            : 1;
  __REG32  CHIS8            : 1;
  __REG32  CHIS9            : 1;
  __REG32  CHIS10           : 1;
  __REG32  CHIS11           : 1;
  __REG32  CHIS12           : 1;
  __REG32  CHIS13           : 1;
  __REG32  CHIS14           : 1;
  __REG32  CHIS15           : 1;
  __REG32  CHIS16           : 1;
  __REG32  CHIS17           : 1;
  __REG32  CHIS18           : 1;
  __REG32  CHIS19           : 1;
  __REG32  CHIS20           : 1;
  __REG32  CHIS21           : 1;
  __REG32  CHIS22           : 1;
  __REG32  CHIS23           : 1;
  __REG32  CHIS24           : 1;
  __REG32  CHIS25           : 1;
  __REG32  CHIS26           : 1;
  __REG32  CHIS27           : 1;
  __REG32  CHIS28           : 1;
  __REG32  CHIS29           : 1;
  __REG32  CHIS30           : 1;
  __REG32  CHIS31           : 1;
} __dmachis_bits;

/* GPIO registers */
typedef struct {
  __REG32  no0            : 1;
  __REG32  no1            : 1;
  __REG32  no2            : 1;
  __REG32  no3            : 1;
  __REG32  no4            : 1;
  __REG32  no5            : 1;
  __REG32  no6            : 1;
  __REG32  no7            : 1;
  __REG32                 :24;
} __gpio_bits;

/* GPIO Port Control (GPIOPCTL) */
typedef struct {
  __REG32  PMC0           : 4;
  __REG32  PMC1           : 4;
  __REG32  PMC2           : 4;
  __REG32  PMC3           : 4;
  __REG32  PMC4           : 4;
  __REG32  PMC5           : 4;
  __REG32  PMC6           : 4;
  __REG32  PMC7           : 4;
} __gpiopctl_bits;

/* EPI Configuration (EPICFG) */
typedef struct {
  __REG32  MODE           : 4;
  __REG32  BLKEN          : 1;
  __REG32                 :27;
} __epicfg_bits;

/* EPI Main Baud Rate (EPIBAUD) */
typedef struct {
  __REG32  COUNT0         :16;
  __REG32  COUNT1         :16;
} __epibaud_bits;

/* EPI SDRAM Configuration (EPISDRAMCFG) */
typedef union{
  /* EPISDRAMCFG */
  struct {
  __REG32  SIZE           : 2;
  __REG32                 : 7;
  __REG32  SLEEP          : 1;
  __REG32                 : 6;
  __REG32  RFSH           :11;
  __REG32                 : 3;
  __REG32  FREQ           : 2;
  };
  /* EPIHB8CFG */
  struct {
  __REG32  MODE           : 2;
  __REG32                 : 2;
  __REG32  RDWS           : 2;
  __REG32  WRWS           : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 4;
  __REG32  RDHIGH         : 1;
  __REG32  WRHIGH         : 1;
  __REG32  XFEEN          : 1;
  __REG32  XFFEN          : 1;
  __REG32                 : 8;
  } __epihb8cfg;
    
  /* EPIHB16CFG */
  struct {
  __REG32  MODE           : 2;
  __REG32  BSEL           : 1;
  __REG32                 : 1;
  __REG32  RDWS           : 2;
  __REG32  WRWS           : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 4;
  __REG32  RDHIGH         : 1;
  __REG32  WRHIGH         : 1;
  __REG32  XFEEN          : 1;
  __REG32  XFFEN          : 1;
  __REG32                 : 8;
  } __epihb16cfg;
  
  /* EPIGPCFG */
  struct {
  __REG32  DSIZE          : 2;
  __REG32                 : 2;
  __REG32  ASIZE          : 2;
  __REG32                 : 2;
  __REG32  MAXWAIT        : 8;
  __REG32                 : 2;
  __REG32  RD2CYC         : 1;
  __REG32  WR2CYC         : 1;
  __REG32                 : 1;
  __REG32  RW             : 1;
  __REG32  FRMCNT         : 4;
  __REG32  FRM50          : 1;
  __REG32  FRMPIN         : 1;
  __REG32  RDYEN          : 1;
  __REG32                 : 1;
  __REG32  CLKGATE        : 1;
  __REG32  CLKPIN         : 1;
  } __epigpcfg;
} __episdramcfg_bits;

/* EPI Host-Bus 8 Configuration 2 (EPIHB8CFG2) */
typedef union {
/* EPIHB8CFG2 */
/* EPIHB16CFG2 */
  struct {
  __REG32                 : 4;
  __REG32  RDWS           : 2;
  __REG32  WRWS           : 2;
  __REG32                 :12;
  __REG32  RDHIGH         : 1;
  __REG32  WRHIGH         : 1;
  __REG32                 : 2;
  __REG32  CSCFG          : 2;
  __REG32  CSBAUD         : 1;
  __REG32                 : 4;
  __REG32  WORD           : 1;
  };
  /* EPIGPCFG2 */
  struct {
  __REG32                 :31;
  __REG32   WORD          : 1;
  } __epigpcfg2;
} __epihb8cfg2_bits;

/* EPI Address Map (EPIADDRMAP) */
typedef struct {
  __REG32  ERADR          : 2;
  __REG32  ERSZ           : 2;
  __REG32  EPADR          : 2;
  __REG32  EPSZ           : 2;
  __REG32                 :24;
} __epiaddrmap_bits;

/* EPI Read Address 0 (EPIRADDR0) 
   EPI Read Address 1 (EPIRADDR1) */
typedef struct {
  __REG32  ADDR           :29;
  __REG32                 : 3;
} __epiraddr_bits;

/* EPI Read Size 0 (EPIRSIZE0) 
   EPI Read Size 1 (EPIRSIZE1) */
typedef struct {
  __REG32  SIZE           : 2;
  __REG32                 :30;
} __epirsize_bits;

/* EPI Non-Blocking Read Data 0 (EPIRPSTD0)
   EPI Non-Blocking Read Data 1 (EPIRPSTD1) */
typedef struct {
  __REG32  POSTCNT        :13;
  __REG32                 :19;
} __epirpstd_bits;

/* EPI Status (EPISTAT) */
typedef struct {
  __REG32  ACTIVE         : 1;
  __REG32                 : 3;
  __REG32  NBRBUSY        : 1;
  __REG32  WBUSY          : 1;
  __REG32  INITSEQ        : 1;
  __REG32  XFEMPTY        : 1;
  __REG32  XFFULL         : 1;
  __REG32  CELOW          : 1;
  __REG32                 :22;
} __epistat_bits;

/* EPI Read FIFO Count (EPIRFIFOCNT) */
typedef struct {
  __REG32  COUNT          : 3;
  __REG32                 :29;
} __epirfifocnt_bits;

/* EPI FIFO Level Selects (EPIFIFOLVL) */
typedef struct {
  __REG32  RDFIFO         : 3;
  __REG32                 : 1;
  __REG32  WRFIFO         : 3;
  __REG32                 : 9;
  __REG32  RSERR          : 1;
  __REG32  WFERR          : 1;
  __REG32                 :14;
} __epififolvl_bits;

/* EPI Write FIFO Count (EPIWFIFOCNT) */
typedef struct {
  __REG32  WTAV           : 3;
  __REG32                 :29;
} __epiwfifocnt_bits;

/* EPI Interrupt Mask (EPIIM) */
typedef struct {
  __REG32  ERRIM          : 1;
  __REG32  RDIM           : 1;
  __REG32  WRIM           : 1;
  __REG32                 :29;
} __epiim_bits;

/* EPI Raw Interrupt Status (EPIRIS) */
typedef struct {
  __REG32  ERRRIS         : 1;
  __REG32  RDRIS          : 1;
  __REG32  WRRIS          : 1;
  __REG32                 :29;
} __epiris_bits;

/* EPI Masked Interrupt Status (EPIMIS) */
typedef struct {
  __REG32  ERRMIS         : 1;
  __REG32  RDMIS          : 1;
  __REG32  WRMIS          : 1;
  __REG32                 :29;
} __epimis_bits;

/* EPI Error Interrupt Status and Clear (EPIEISC) */
typedef struct {
  __REG32  TOUT           : 1;
  __REG32  RSTALL         : 1;
  __REG32  WTFULL         : 1;
  __REG32                 :29;
} __epieisc_bits;

/* GPTM Configuration (GPTMCFG) */
typedef struct {
  __REG32  GPTMCFG        : 3;
  __REG32                 :29;
} __gptmcfg_bits;

/* GPTM TimerA Mode (GPTMTAMR) */
typedef struct {
  __REG32  TAMR           : 2;
  __REG32  TACMR          : 1;
  __REG32  TAAMS          : 1;
  __REG32  TACDIR         : 1;
  __REG32  TAMIE          : 1;
  __REG32  TAWOT          : 1;
  __REG32  TASNAPS        : 1;
  __REG32                 :24;
} __gptmtamr_bits;

/* GPTM TimerB Mode (GPTMTBMR) */
typedef struct {
  __REG32  TBMR           : 2;
  __REG32  TBCMR          : 1;
  __REG32  TBAMS          : 1;
  __REG32  TBCDIR         : 1;
  __REG32  TBMIE          : 1;
  __REG32  TBWOT          : 1;
  __REG32  TBSNAPS        : 1;
  __REG32                 :24;
} __gptmtbmr_bits;

/* GPTM Control (GPTMCTL) */
typedef struct {
  __REG32  TAEN           : 1;
  __REG32  TASTALL        : 1;
  __REG32  TAEVENT        : 2;
  __REG32  RTCEN          : 1;
  __REG32  TAOTE          : 1;
  __REG32  TAPWML         : 1;
  __REG32                 : 1;
  __REG32  TBEN           : 1;
  __REG32  TBSTALL        : 1;
  __REG32  TBEVENT        : 2;
  __REG32                 : 1;
  __REG32  TBOTE          : 1;
  __REG32  TBPWML         : 1;
  __REG32                 :17;
} __gptmctl_bits;

/* GPTM Interrupt Mask (GPTMIMR) */
typedef struct {
  __REG32  TATOIM         : 1;
  __REG32  CAMIM          : 1;
  __REG32  CAEIM          : 1;
  __REG32  RTCIM          : 1;
  __REG32  TAMIM          : 1;
  __REG32                 : 3;
  __REG32  TBTOIM         : 1;
  __REG32  CBMIM          : 1;
  __REG32  CBEIM          : 1;
  __REG32  TBMIM          : 1;
  __REG32                 :20;
} __gptmimr_bits;

/* GPTM Raw Interrupt Status (GPTMRIS) */
typedef struct {
  __REG32  TATORIS        : 1;
  __REG32  CAMRIS         : 1;
  __REG32  CAERIS         : 1;
  __REG32  RTCRIS         : 1;
  __REG32  TAMRIS         : 1;
  __REG32                 : 3;
  __REG32  TBTORIS        : 1;
  __REG32  CBMRIS         : 1;
  __REG32  CBERIS         : 1;
  __REG32  TBMRIS         : 1;
  __REG32                 :20;
} __gptmris_bits;

/* GPTM Masked Interrupt Status (GPTMMIS) */
typedef struct {
  __REG32  TATOMIS        : 1;
  __REG32  CAMMIS         : 1;
  __REG32  CAEMIS         : 1;
  __REG32  RTCMIS         : 1;
  __REG32  TAMMIS         : 1;
  __REG32                 : 3;
  __REG32  TBTOMIS        : 1;
  __REG32  CBMMIS         : 1;
  __REG32  CBEMIS         : 1;
  __REG32  TBMMIS         : 1;
  __REG32                 :20;
} __gptmmis_bits;

/* GPTM Interrupt Clear (GPTMICR) */
typedef struct {
  __REG32  TATOCINT       : 1;
  __REG32  CAMCINT        : 1;
  __REG32  CAECINT        : 1;
  __REG32  RTCCINT        : 1;
  __REG32  TAMCINT        : 1;
  __REG32                 : 3;
  __REG32  TBTOCINT       : 1;
  __REG32  CBMCINT        : 1;
  __REG32  CBECINT        : 1;
  __REG32  TBMCINT        : 1;
  __REG32                 :20;
} __gptmicr_bits;

/* GPTM TimerA Interval Load (GPTMTAILR) */
typedef struct {
  __REG32  TAILRL         :16;
  __REG32  TAILRH         :16;
} __gptmtailr_bits;

/* GPTM TimerB Interval Load (GPTMTBILR) */
typedef struct {
  __REG32  TBILRL         :16;
  __REG32                 :16;
} __gptmtbilr_bits;

/* GPTM TimerA Match (GPTMTAMATCHR) */
typedef struct {
  __REG32  TAMRL          :16;
  __REG32  TAMRH          :16;
} __gptmtamatchr_bits;

/* GPTM TimerB Match (GPTMTBMATCHR) */
typedef struct {
  __REG32  TBMRL          :16;
  __REG32                 :16;
} __gptmtbmatchr_bits;

/* GPTM Timer A Prescale (GPTMTAPR) */
typedef struct {
  __REG32  TAPSR          : 8;
  __REG32                 :24;
} __gptmtapr_bits;

/* GPTM Timer B Prescale (GPTMTBPR) */
typedef struct {
  __REG32  TBPSR          : 8;
  __REG32                 :24;
} __gptmtbpr_bits;

/* GPTM TimerA Prescale Match (GPTMTAPMR) */
typedef struct {
  __REG32  TAPSMR         : 8;
  __REG32                 :24;
} __gptmtapmr_bits;

/* GPTM TimerB Prescale Match (GPTMTBPMR) */
typedef struct {
  __REG32  TBPSMR         : 8;
  __REG32                 :24;
} __gptmtbpmr_bits;

/* GPTM TimerA (GPTMTAR) */
typedef struct {
  __REG32  TARL           :16;
  __REG32  TARH           :16;
} __gptmtar_bits;

/* GPTM TimerB (GPTMTBR) */
typedef struct {
  __REG32  TBRL           :16;
  __REG32  TBRH           : 8;
  __REG32                 : 8;
} __gptmtbr_bits;

/* GPTM Timer A Value (GPTMTAV) */
typedef struct {
  __REG32  TAVL           :16;
  __REG32  TAVH           :16;
} __gptmtav_bits;

/* GPTM Timer B Value (GPTMTBV) */
typedef struct {
  __REG32  TBVL           :16;
  __REG32  TBVH           : 8;
  __REG32                 : 8;
} __gptmtbv_bits;

/* Watchdog 0 Control (WDTCTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :30;
} __wdt0ctl_bits;

/* Watchdog 1 Control (WDTCTL) */
typedef struct {
  __REG32  INTEN          : 1;
  __REG32  RESEN          : 1;
  __REG32                 :29;
  __REG32  WRC            : 1;
} __wdt1ctl_bits;

/* Watchdog Raw Interrupt Status (WDTRIS) */
typedef struct {
  __REG32  WDTRIS         : 1;
  __REG32                 :31;
} __wdtris_bits;

/* Watchdog Masked Interrupt Status (WDTMIS) */
typedef struct {
  __REG32  WDTMIS         : 1;
  __REG32                 :31;
} __wdtmis_bits;

/* Watchdog Test (WDTTEST) */
typedef struct {
  __REG32                 : 8;
  __REG32  STALL          : 1;
  __REG32                 :23;
} __wdttest_bits;

/* Analog-to-Digital Converter Active Sample Sequencer (ADCACTSS) */
typedef struct {
  __REG32  ASEN0          : 1;
  __REG32  ASEN1          : 1;
  __REG32  ASEN2          : 1;
  __REG32  ASEN3          : 1;
  __REG32                 :28;
} __adcactss_bits;

/* Analog-to-Digital Converter Raw Interrupt Status (ADCRIS) */
typedef struct {
  __REG32  INR0           : 1;
  __REG32  INR1           : 1;
  __REG32  INR2           : 1;
  __REG32  INR3           : 1;
  __REG32                 :12;
  __REG32  INRDC          : 1;
  __REG32                 :15;
} __adcris_bits;

/* Analog-to-Digital Converter Interrupt Mask (ADCIM) */
typedef struct {
  __REG32  MASK0          : 1;
  __REG32  MASK1          : 1;
  __REG32  MASK2          : 1;
  __REG32  MASK3          : 1;
  __REG32                 :12;
  __REG32  DCONSS0        : 1;
  __REG32  DCONSS1        : 1;
  __REG32  DCONSS2        : 1;
  __REG32  DCONSS3        : 1;
  __REG32                 :12;
} __adcim_bits;

/* Analog-to-Digital Converter Interrupt Status and Clear (ADCISC) */
typedef struct {
  __REG32  IN0            : 1;
  __REG32  IN1            : 1;
  __REG32  IN2            : 1;
  __REG32  IN3            : 1;
  __REG32                 :12;
  __REG32  DCINSS0        : 1;
  __REG32  DCINSS1        : 1;
  __REG32  DCINSS2        : 1;
  __REG32  DCINSS3        : 1;
  __REG32                 :12;
} __adcisc_bits;

/* Analog-to-Digital Converter Overflow Status (ADCOSTAT) */
typedef struct {
  __REG32  OV0            : 1;
  __REG32  OV1            : 1;
  __REG32  OV2            : 1;
  __REG32  OV3            : 1;
  __REG32                 :28;
} __adcostat_bits;

/* Analog-to-Digital Converter Event Multiplexer Select (ADCEMUX) */
typedef struct {
  __REG32  EM0            : 4;
  __REG32  EM1            : 4;
  __REG32  EM2            : 4;
  __REG32  EM3            : 4;
  __REG32                 :16;
} __adcemux_bits;

/* Analog-to-Digital Converter Underflow Status (ADCUSTAT) */
typedef struct {
  __REG32  UV0            : 1;
  __REG32  UV1            : 1;
  __REG32  UV2            : 1;
  __REG32  UV3            : 1;
  __REG32                 :28;
} __adcustat_bits;

/* Analog-to-Digital Converter Sample Sequencer Priority (ADCSSPRI) */
typedef struct {
  __REG32  SS0            : 2;
  __REG32                 : 2;
  __REG32  SS1            : 2;
  __REG32                 : 2;
  __REG32  SS2            : 2;
  __REG32                 : 2;
  __REG32  SS3            : 2;
  __REG32                 :18;
} __adcsspri_bits;

/* ADC Sample Phase Control (ADCSPC) */
typedef struct {
  __REG32  PHASE          : 4;
  __REG32                 :28;
} __adcspc_bits;

/* Analog-to-Digital Converter Processor Sample Sequence Initiate (ADCPSSI) */
typedef struct {
  __REG32  SS0            : 1;
  __REG32  SS1            : 1;
  __REG32  SS2            : 1;
  __REG32  SS3            : 1;
  __REG32                 :23;
  __REG32  SYNCWAIT       : 1;
  __REG32                 : 3;
  __REG32  GSYNC          : 1;
} __adcpssi_bits;

/* Analog-to-Digital Converter Sample Averaging Control (ADCSAC) */
typedef struct {
  __REG32  AVG            : 3;
  __REG32                 :29;
} __adcsac_bits;

/* ADC Digital Comparator Interrupt Status and Clear (ADCDCISC) */
typedef struct {
  __REG32  DCINT0         : 1;
  __REG32  DCINT1         : 1;
  __REG32  DCINT2         : 1;
  __REG32  DCINT3         : 1;
  __REG32  DCINT4         : 1;
  __REG32  DCINT5         : 1;
  __REG32  DCINT6         : 1;
  __REG32  DCINT7         : 1;
  __REG32                 :24;
} __adcdcisc_bits;

/* ADC Control (ADCCTL) */
typedef struct {
  __REG32  VREF           : 2;
  __REG32                 : 2;
  __REG32  RES            : 1;
  __REG32                 :27;
} __adcctl_bits;

/* Analog-to-Digital Converter Sample Sequence Input Mux Select 0 (ADCSSMUX0) */
typedef struct {
  __REG32  MUX0           : 4;
  __REG32  MUX1           : 4;
  __REG32  MUX2           : 4;
  __REG32  MUX3           : 4;
  __REG32  MUX4           : 4;
  __REG32  MUX5           : 4;
  __REG32  MUX6           : 4;
  __REG32  MUX7           : 4;
} __adcssmux0_bits;

/* Analog-to-Digital Converter Sample Sequence Control 0 (ADCSSCTL0) */
typedef struct {
  __REG32  D0             : 1;
  __REG32  END0           : 1;
  __REG32  IE0            : 1;
  __REG32  TS0            : 1;
  __REG32  D1             : 1;
  __REG32  END1           : 1;
  __REG32  IE1            : 1;
  __REG32  TS1            : 1;
  __REG32  D2             : 1;
  __REG32  END2           : 1;
  __REG32  IE2            : 1;
  __REG32  TS2            : 1;
  __REG32  D3             : 1;
  __REG32  END3           : 1;
  __REG32  IE3            : 1;
  __REG32  TS3            : 1;
  __REG32  D4             : 1;
  __REG32  END4           : 1;
  __REG32  IE4            : 1;
  __REG32  TS4            : 1;
  __REG32  D5             : 1;
  __REG32  END5           : 1;
  __REG32  IE5            : 1;
  __REG32  TS5            : 1;
  __REG32  D6             : 1;
  __REG32  END6           : 1;
  __REG32  IE6            : 1;
  __REG32  TS6            : 1;
  __REG32  D7             : 1;
  __REG32  END7           : 1;
  __REG32  IE7            : 1;
  __REG32  TS7            : 1;
} __adcssctl0_bits;

/* Analog-to-Digital Converter Sample Sequence FIFO 0 (ADCSSFIFO0) */
typedef struct {
  __REG32  DATA           :12;
  __REG32                 :20;
} __adcssfifo0_bits;

/* Analog-to-Digital Converter Sample Sequence FIFO 0 Status (ADCSSFSTAT0) */
typedef struct {
  __REG32  TPTR           : 4;
  __REG32  HPTR           : 4;
  __REG32  EMPTY          : 1;
  __REG32                 : 3;
  __REG32  FULL           : 1;
  __REG32                 :19;
} __adcssfstat0_bits;

/* ADC Sample Sequence 0 Operation (ADCSSOP0) */
typedef struct {
  __REG32  S0DCOP         : 1;
  __REG32                 : 3;
  __REG32  S1DCOP         : 1;
  __REG32                 : 3;
  __REG32  S2DCOP         : 1;
  __REG32                 : 3;
  __REG32  S3DCOP         : 1;
  __REG32                 : 3;
  __REG32  S4DCOP         : 1;
  __REG32                 : 3;
  __REG32  S5DCOP         : 1;
  __REG32                 : 3;
  __REG32  S6DCOP         : 1;
  __REG32                 : 3;
  __REG32  S7DCOP         : 1;
  __REG32                 : 3;
} __adcssop_bits;

/* ADC Sample Sequence 0 Digital Comparator Select (ADCSSDC0) */
typedef struct {
  __REG32  S0DCSEL        : 4;
  __REG32  S1DCSEL        : 4;
  __REG32  S2DCSEL        : 4;
  __REG32  S3DCSEL        : 4;
  __REG32  S4DCSEL        : 4;
  __REG32  S5DCSEL        : 4;
  __REG32  S6DCSEL        : 4;
  __REG32  S7DCSEL        : 4;
} __adcssdc_bits;

/* Analog-to-Digital Converter Sample Sequence Input Mux Select 1 (ADCSSMUX1) */
typedef struct {
  __REG32  MUX0           : 4;
  __REG32  MUX1           : 4;
  __REG32  MUX2           : 4;
  __REG32  MUX3           : 4;
  __REG32                 :16;
} __adcssmux1_bits;

/* Analog-to-Digital Converter Sample Sequence Control 1 (ADCSSCTL1) */
typedef struct {
  __REG32  D0             : 1;
  __REG32  END0           : 1;
  __REG32  IE0            : 1;
  __REG32  TS0            : 1;
  __REG32  D1             : 1;
  __REG32  END1           : 1;
  __REG32  IE1            : 1;
  __REG32  TS1            : 1;
  __REG32  D2             : 1;
  __REG32  END2           : 1;
  __REG32  IE2            : 1;
  __REG32  TS2            : 1;
  __REG32  D3             : 1;
  __REG32  END3           : 1;
  __REG32  IE3            : 1;
  __REG32  TS3            : 1;
  __REG32                 :16;
} __adcssctl1_bits;

/* ADC Sample Sequence 1 Operation (ADCSSOP1) */
typedef struct {
  __REG32  S0DCOP         : 1;
  __REG32                 : 3;
  __REG32  S1DCOP         : 1;
  __REG32                 : 3;
  __REG32  S2DCOP         : 1;
  __REG32                 : 3;
  __REG32  S3DCOP         : 1;
  __REG32                 :19;
} __adcssop1_bits;

/* ADC Sample Sequence 1 Digital Comparator Select (ADCSSDC1) */
typedef struct {
  __REG32  S0DCSEL        : 4;
  __REG32  S1DCSEL        : 4;
  __REG32  S2DCSEL        : 4;
  __REG32  S3DCSEL        : 4;
  __REG32                 :16;
} __adcssdc1_bits;

/* Analog-to-Digital Converter Sample Sequence Input Mux Select 3 (ADCSSMUX3) */
typedef struct {
  __REG32  MUX0           : 3;
  __REG32                 :29;
} __adcssmux3_bits;

/* Analog-to-Digital Converter Sample Sequence Control 3 (ADCSSCTL3) */
typedef struct {
  __REG32  D0             : 1;
  __REG32  END0           : 1;
  __REG32  IE0            : 1;
  __REG32  TS0            : 1;
  __REG32                 :28;
} __adcssctl3_bits;

/* ADC Sample Sequence 3 Operation (ADCSSOP3) */
typedef struct {
  __REG32  S0DCOP         : 1;
  __REG32                 :31;
} __adcssop3_bits;

/* ADC Sample Sequence 3 Digital Comparator Select (ADCSSDC3) */
typedef struct {
  __REG32  S0DCSEL        : 4;
  __REG32                 :28;
} __adcssdc3_bits;

/* ADC Digital Comparator Reset Initial Conditions (ADCDCRIC) */
typedef struct {
  __REG32  DCINT0         : 1;
  __REG32  DCINT1         : 1;
  __REG32  DCINT2         : 1;
  __REG32  DCINT3         : 1;
  __REG32  DCINT4         : 1;
  __REG32  DCINT5         : 1;
  __REG32  DCINT6         : 1;
  __REG32  DCINT7         : 1;
  __REG32                 : 8;
  __REG32  DCTRIG0        : 1;
  __REG32  DCTRIG1        : 1;
  __REG32  DCTRIG2        : 1;
  __REG32  DCTRIG3        : 1;
  __REG32  DCTRIG4        : 1;
  __REG32  DCTRIG5        : 1;
  __REG32  DCTRIG6        : 1;
  __REG32  DCTRIG7        : 1;
  __REG32                 : 8;
} __adcdcric_bits;

/* ADC Digital Comparator Control x (ADCDCCTLx) */
typedef struct {
  __REG32  CIM            : 2;
  __REG32  CIC            : 2;
  __REG32  CIE            : 1;
  __REG32                 : 3;
  __REG32  CTM            : 2;
  __REG32  CTC            : 2;
  __REG32  CTE            : 1;
  __REG32                 :19;
} __adcdcctl_bits;

/* ADC Digital Comparator Range x (ADCDCCMPx) */
typedef struct {
  __REG32  COMP0          :12;
  __REG32                 : 4;
  __REG32  COMP1          :12;
  __REG32                 : 4;
} __adcdccmp_bits;

/* UART Data (UARTDR) */
typedef struct {
  __REG32  DATA           : 8;
  __REG32  FE             : 1;
  __REG32  PE             : 1;
  __REG32  BE             : 1;
  __REG32  OE             : 1;
  __REG32                 :20;
} __uartdr_bits;

/* UART Receive Status/Error Clear (UARTRSR/UARTECR) */
typedef union {
  /* UARTxRSR */
  struct {
    __REG32  FE           : 1;
    __REG32  PE           : 1;
    __REG32  BE           : 1;
    __REG32  OE           : 1;
    __REG32               :28;
  };
  /* UARTxECR */
  struct {
    __REG32  DATA         : 8;
    __REG32               :24;
  };
} __uartrsr_bits;

/* UART Flag (UARTFR) */
typedef struct {
  __REG32  CTS            : 1;
  __REG32  DSR            : 1;
  __REG32  DCD            : 1;
  __REG32  BUSY           : 1;
  __REG32  RXFE           : 1;
  __REG32  TXFF           : 1;
  __REG32  RXFF           : 1;
  __REG32  TXFE           : 1;
  __REG32  RI             : 1;
  __REG32                 :23;
} __uartfr_bits;

/* UART Fractional Baud-Rate Divisor (UARTFBRD) */
typedef struct {
  __REG32  DIVFRAC        : 6;
  __REG32                 :26;
} __uartfbrd_bits;

/* UART Line Control (UARTLCRH) */
typedef struct {
  __REG32  BRK            : 1;
  __REG32  PEN            : 1;
  __REG32  EPS            : 1;
  __REG32  STP2           : 1;
  __REG32  FEN            : 1;
  __REG32  WLEN           : 2;
  __REG32  SPS            : 1;
  __REG32                 :24;
} __uartlcrh_bits;

/* UART Control (UARTCTL) */
typedef struct {
  __REG32  UARTEN         : 1;
  __REG32  SIREN          : 1;
  __REG32  SIRLP          : 1;
  __REG32  SMART          : 1;
  __REG32  EOT            : 1;
  __REG32  HSE            : 1;
  __REG32  LIN            : 1;   
  __REG32  LBE            : 1;
  __REG32  TXE            : 1;
  __REG32  RXE            : 1;
  __REG32  DTR            : 1;
  __REG32  RTS            : 1;
  __REG32                 : 2;
  __REG32  RTSEN          : 1;
  __REG32  CTSEN          : 1;
  __REG32                 :16;
} __uartctl_bits;

/* UART Interrupt FIFO Level Select (UARTIFLS) */
typedef struct {
  __REG32  TXIFLSEL       : 3;
  __REG32  RXIFLSEL       : 3;
  __REG32                 :26;
} __uartifls_bits;

/* UART Interrupt Mask (UARTIM) */
typedef struct {
  __REG32  RIIM           : 1;
  __REG32  CTSIM          : 1;
  __REG32  DCDIM          : 1;
  __REG32  DSRIM          : 1;
  __REG32  RXIM           : 1;
  __REG32  TXIM           : 1;
  __REG32  RTIM           : 1;
  __REG32  FEIM           : 1;
  __REG32  PEIM           : 1;
  __REG32  BEIM           : 1;
  __REG32  OEIM           : 1;
  __REG32                 : 2;
  __REG32  LMSBIM         : 1;
  __REG32  LME1IM         : 1;
  __REG32  LME5IM         : 1;
  __REG32                 :16;
} __uartim_bits;

/* UART Raw Interrupt Status (UARTRIS) */
typedef struct {
  __REG32  RIRIS          : 1;
  __REG32  CTSRIS         : 1;
  __REG32  DCDRIS         : 1;
  __REG32  DSRRIS         : 1;
  __REG32  RXRIS          : 1;
  __REG32  TXRIS          : 1;
  __REG32  RTRIS          : 1;
  __REG32  FERIS          : 1;
  __REG32  PERIS          : 1;
  __REG32  BERIS          : 1;
  __REG32  OERIS          : 1;
  __REG32                 : 2;
  __REG32  LMSBRIS        : 1;
  __REG32  LME1RIS        : 1;
  __REG32  LME5RIS        : 1;
  __REG32                 :16;
} __uartris_bits;

/* UART Masked Interrupt Status (UARTMIS) */
typedef struct {
  __REG32  RIMIS          : 1;
  __REG32  CTSMIS         : 1;
  __REG32  DCDMIS         : 1;
  __REG32  DSRMIS         : 1;
  __REG32  RXMIS          : 1;
  __REG32  TXMIS          : 1;
  __REG32  RTMIS          : 1;
  __REG32  FEMIS          : 1;
  __REG32  PEMIS          : 1;
  __REG32  BEMIS          : 1;
  __REG32  OEMIS          : 1;
  __REG32                 : 2;
  __REG32  LMSBMIS        : 1;
  __REG32  LME1MIS        : 1;
  __REG32  LME5MIS        : 1;
  __REG32                 :16;
} __uartmis_bits;

/* UART Interrupt Clear (UARTICR) */
typedef struct {
  __REG32  RIMIC          : 1;
  __REG32  CTSMIC         : 1;
  __REG32  DCDMIC         : 1;
  __REG32  DSRMIC         : 1;
  __REG32  RXIC           : 1;
  __REG32  TXIC           : 1;
  __REG32  RTIC           : 1;
  __REG32  FEIC           : 1;
  __REG32  PEIC           : 1;
  __REG32  BEIC           : 1;
  __REG32  OEIC           : 1;
  __REG32                 : 2;
  __REG32  LMSBMIC        : 1;
  __REG32  LME1MIC        : 1;
  __REG32  LME5MIC        : 1;
  __REG32                 :16;
} __uarticr_bits;

/* UART DMA Control (UARTDMACTL) */
typedef struct {
  __REG32  RXDMAE         : 1;
  __REG32  TXDMAE         : 1;
  __REG32  DMAERR         : 1;
  __REG32                 :29;
} __uartdmactl_bits;

/* UART LIN Control (UARTLCTL) */
typedef struct {
  __REG32  MASTER         : 1;
  __REG32                 : 3;
  __REG32  BLEN           : 2;
  __REG32                 :26;
} __uartlctl_bits;

/* SSI Control 0 (SSICR0) */
typedef struct {
  __REG32  DSS            : 4;
  __REG32  FRF            : 2;
  __REG32  SPO            : 1;
  __REG32  SPH            : 1;
  __REG32  SCR            : 8;
  __REG32                 :16;
} __ssicr0_bits;

/* SSI Control 1 (SSICR1) */
typedef struct {
  __REG32  LBM            : 1;
  __REG32  SSE            : 1;
  __REG32  MS             : 1;
  __REG32  SOD            : 1;
  __REG32  EOT            : 1;
  __REG32                 :27;
} __ssicr1_bits;

/* SSI Status (SSISR) */
typedef struct {
  __REG32  TFE            : 1;
  __REG32  TNF            : 1;
  __REG32  RNE            : 1;
  __REG32  RFF            : 1;
  __REG32  BSY            : 1;
  __REG32                 :27;
} __ssisr_bits;

/* SSI Interrupt Mask (SSIIM) */
typedef struct {
  __REG32  RORIM          : 1;
  __REG32  RTIM           : 1;
  __REG32  RXIM           : 1;
  __REG32  TXIM           : 1;
  __REG32                 :28;
} __ssiim_bits;

/* SSI Raw Interrupt Status (SSIRIS) */
typedef struct {
  __REG32  RORRIS         : 1;
  __REG32  RTRIS          : 1;
  __REG32  RXRIS          : 1;
  __REG32  TXRIS          : 1;
  __REG32                 :28;
} __ssiris_bits;

/* SSI Masked Interrupt Status (SSIMIS) */
typedef struct {
  __REG32  RORMIS         : 1;
  __REG32  RTMIS          : 1;
  __REG32  RXMIS          : 1;
  __REG32  TXMIS          : 1;
  __REG32                 :28;
} __ssimis_bits;

/* SSI Interrupt Clear (SSIICR) */
typedef struct {
  __REG32  RORIC          : 1;
  __REG32  RTIC           : 1;
  __REG32                 :30;
} __ssiicr_bits;

/* SSI DMA Control (SSIDMACTL) */
typedef struct {
  __REG32  RXDMAE         : 1;
  __REG32  TXDMAE         : 1;
  __REG32                 :30;
} __ssidmactl_bits;

/* I2C Master Slave Address (I2CMSA) */
typedef struct {
  __REG32  R_S            : 1;
  __REG32  SA             : 7;
  __REG32                 :24;
} __i2cmsa_bits;

/* I2C Master Control/Status (I2CMCS) */
typedef union {
  /* I2CxMS */
  struct {
    __REG32  BUSY         : 1;
    __REG32  ERROR        : 1;
    __REG32  ADRACK       : 1;
    __REG32  DATACK       : 1;
    __REG32  ARBLST       : 1;
    __REG32  IDLE         : 1;
    __REG32  BUSBSY       : 1;
    __REG32               :25;
  };
  /* I2CxMC */
  struct {
    __REG32  RUN          : 1;
    __REG32  START        : 1;
    __REG32  STOP         : 1;
    __REG32  ACK          : 1;
    __REG32               :28;
  };
} __i2cmcs_bits;

/* I2C Master Interrupt Mask (I2CMIMR) */
typedef struct {
  __REG32  IM             : 1;
  __REG32                 :31;
} __i2cmimr_bits;

/* I2C Master Raw Interrupt Status (I2CMRIS) */
typedef struct {
  __REG32  RIS            : 1;
  __REG32                 :31;
} __i2cmris_bits;

/* I2C Master Masked Interrupt Status (I2CMMIS) */
typedef struct {
  __REG32  MIS            : 1;
  __REG32                 :31;
} __i2cmmis_bits;

/* I2C Master Interrupt Clear (I2CMICR) */
typedef struct {
  __REG32  IC             : 1;
  __REG32                 :31;
} __i2cmicr_bits;

/* I2C Master Configuration (I2CMCR) */
typedef struct {
  __REG32  LPBK           : 1;
  __REG32                 : 3;
  __REG32  MFE            : 1;
  __REG32  SFE            : 1;
  __REG32                 :26;
} __i2cmcr_bits;

/* I2C Slave Own Address (I2CSOAR) */
typedef struct {
  __REG32  OAR            : 7;
  __REG32                 :25;
} __i2csoar_bits;

/* I2C Slave Control/Status (I2CSCSR) */
typedef union {
  /* I2CxSSR */
  struct {
    __REG32  RREQ         : 1;
    __REG32  TREQ         : 1;
    __REG32  FBR          : 1;
    __REG32               :29;
  };
  /* I2CxSCR */
  struct {
    __REG32  DA           : 1;
    __REG32               :31;
  };
} __i2cscsr_bits;

/* I2C Slave Interrupt Mask (I2CSIMR) */
typedef struct {
  __REG32  DATAIM         : 1;
  __REG32  STARTIM        : 1;
  __REG32  STOPIM         : 1;
  __REG32                 :29;
} __i2csimr_bits;

/* I2C Slave Raw Interrupt Status (I2CSRIS) */
typedef struct {
  __REG32  DATARIS        : 1;
  __REG32  STARTRIS       : 1;
  __REG32  STOPRIS        : 1;
  __REG32                 :29;
} __i2csris_bits;

/* I2C Slave Masked Interrupt Status (I2CSMIS) */
typedef struct {
  __REG32  DATAMIS        : 1;
  __REG32  STARTMIS       : 1;
  __REG32  STOPMIS        : 1;
  __REG32                 :29;
} __i2csmis_bits;

/* I2C Slave Interrupt Clear (I2CSICR) */
typedef struct {
  __REG32  DATAIC         : 1;
  __REG32  STARTIC        : 1;
  __REG32  STOPIC         : 1;
  __REG32                 :29;
} __i2csicr_bits;

/* I2S Transmit FIFO Configuration (I2STXFIFOCFG) */
typedef struct {
  __REG32  LRS            : 1;
  __REG32  CSS            : 1;
  __REG32                 :30;
} __i2stxfifocfg_bits;

/* I2S Transmit Module Configuration (I2STXCFG) */
typedef struct {
  __REG32                 : 4;
  __REG32  SDSZ           : 6;
  __REG32  SSZ            : 6;
  __REG32                 : 6;
  __REG32  MSL            : 1;
  __REG32  FMT            : 1;
  __REG32  WM             : 2;
  __REG32  LRP            : 1;
  __REG32  SCP            : 1;
  __REG32  DLY            : 1;
  __REG32  JST            : 1;
  __REG32                 : 2;
} __i2stxcfg_bits;

/* I2S Transmit FIFO Limit (I2STXLIMIT) */
typedef struct {
  __REG32  LIMIT          : 5;
  __REG32                 :27;
} __i2stxlimit_bits;

/* I2S Transmit Interrupt Status and Mask (I2STXISM) */
typedef struct {
  __REG32  FFM            : 1;
  __REG32                 :15;
  __REG32  FFI            : 1;
  __REG32                 :15;
} __i2stxism_bits;

/* I2S Transmit FIFO Level (I2STXLEV) */
typedef struct {
  __REG32  LEVEL          : 5;
  __REG32                 :27;
} __i2stxlev_bits;

/* I2S Receive FIFO Configuration (I2SRXFIFOCFG) */
typedef struct {
  __REG32  LRS            : 1;
  __REG32  CSS            : 1;
  __REG32  FMM            : 1;
  __REG32                 :29;
} __i2srxfifocfg_bits;

/* I2S Receive Module Configuration (I2SRXCFG) */
typedef struct {
  __REG32                 : 4;
  __REG32  SDSZ           : 6;
  __REG32  SSZ            : 6;
  __REG32                 : 6;
  __REG32  MSL            : 1;
  __REG32                 : 1;
  __REG32  RM             : 1;
  __REG32                 : 1;
  __REG32  LRP            : 1;
  __REG32  SCP            : 1;
  __REG32  DLY            : 1;
  __REG32  JST            : 1;
  __REG32                 : 2;
} __i2srxcfg_bits;

/* I2S Receive FIFO Limit (I2SRXLIMIT) */
typedef struct {
  __REG32  LIMIT          : 5;
  __REG32                 :27;
} __i2srxlimit_bits;

/* I2S Receive Interrupt Status and Mask (I2SRXISM) */
typedef struct {
  __REG32  FFM            : 1;
  __REG32                 :15;
  __REG32  FFI            : 1;
  __REG32                 :15;
} __i2srxism_bits;

/* I2S Receive FIFO Level (I2SRXLEV) */
typedef struct {
  __REG32  LEVEL          : 5;
  __REG32                 :27;
} __i2srxlev_bits;

/* I2S Module Configuration (I2SCFG) */
typedef struct {
  __REG32  TXEN           : 1;
  __REG32  RXEN           : 1;
  __REG32                 : 2;
  __REG32  TXSLV          : 1;
  __REG32  RXSLV          : 1;
  __REG32                 :26;
} __i2scfg_bits;

/* I2S Interrupt Mask (I2SIM) */
typedef struct {
  __REG32  TXSRIM         : 1;
  __REG32  TXWEIM         : 1;
  __REG32                 : 2;
  __REG32  RXSRIM         : 1;
  __REG32  RXREIM         : 1;
  __REG32                 :26;
} __i2sim_bits;

/* I2S Raw Interrupt Status (I2SRIS) */
typedef struct {
  __REG32  TXSRRIS        : 1;
  __REG32  TXWERIS        : 1;
  __REG32                 : 2;
  __REG32  RXSRRIS        : 1;
  __REG32  RXRERIS        : 1;
  __REG32                 :26;
} __i2sris_bits;

/* I2S Masked Interrupt Status (I2SMIS) */
typedef struct {
  __REG32  TXSRMIS        : 1;
  __REG32  TXWEMIS        : 1;
  __REG32                 : 2;
  __REG32  RXSRMIS        : 1;
  __REG32  RXREMIS        : 1;
  __REG32                 :26;
} __i2smis_bits;

/* I2S Interrupt Clear (I2SIC) */
typedef struct {
  __REG32                 : 1;
  __REG32  TXWEIC         : 1;
  __REG32                 : 3;
  __REG32  RXREIC         : 1;
  __REG32                 :26;
} __i2sic_bits;

/* CAN Control (CANCTL) */
typedef struct {
  __REG32  INIT           : 1;
  __REG32  IE             : 1;
  __REG32  SIE            : 1;
  __REG32  EIE            : 1;
  __REG32                 : 1;
  __REG32  DAR            : 1;
  __REG32  CCE            : 1;
  __REG32  TEST           : 1; 
  __REG32                 :24;
} __canctl_bits;

/* CAN Status (CANSTS) */
typedef struct {
  __REG32  LEC            : 3;
  __REG32  TXOK           : 1;  
  __REG32  RXOK           : 1;  
  __REG32  EPASS          : 1;  
  __REG32  EWARN          : 1;
  __REG32  BOFF           : 1;  
  __REG32                 :24;
} __cansts_bits;

/* CAN Error Counter (CANERR) */
typedef struct {
  __REG32  TEC            : 8;
  __REG32  REC            : 7;
  __REG32  RP             : 1;    
  __REG32                 :16;
} __canerr_bits;

/* CAN Bit Timing (CANBIT) */
typedef struct {
  __REG32  BRP            : 6;
  __REG32  SJW            : 2;
  __REG32  TSEG1          : 4;  
  __REG32  TSEG2          : 3;    
  __REG32                 :17;
} __canbit_bits;

/* CAN Interrupt (CANINT) */
typedef struct {
  __REG32  INTID          :16; 
  __REG32                 :16;  
} __canint_bits;

/* CAN Test  (CANTST) */
typedef struct {
  __REG32                 : 2;   
  __REG32  BASIC          : 1;
  __REG32  SILENT         : 1;
  __REG32  LBACK          : 1;  
  __REG32  TX             : 2;
  __REG32  RX             : 1;          
  __REG32                 :24;
} __cantst_bits;

/* CAN Baud Rate Prescaler Extension (CANBRPE) */
typedef struct {
  __REG32  BRPE           : 4;
  __REG32                 :28;     
} __canbrpe_bits;

/* CAN IFn Command Request (CANIFnCRQ) */
typedef struct {
  __REG32  MNUM           : 6;
  __REG32                 : 9;
  __REG32  BUSY           : 1;        
  __REG32                 :16;
} __canifcrq_bits;

/* CAN IFn Command Mask (CANIFnCMSK) */
typedef struct {
  __REG32  DATAB          : 1;
  __REG32  DATAA          : 1;   
  __REG32  TXRQST         : 1;   
  __REG32  CLRINTPND      : 1;  
  __REG32  CONTROL        : 1;
  __REG32  ARB            : 1;    
  __REG32  MASK           : 1;    
  __REG32  WRNRD          : 1;   
  __REG32                 :24;
} __canifcmsk_bits;

/* CAN IFn Mask1 (CANIFnMSK1) */
typedef struct {
  __REG32  MSK            :16;
  __REG32                 :16;   
} __canifmsk1_bits;

/* CAN IFn Mask2 (CANIFnMSK2) */
typedef struct {
  __REG32  MSK            :13;
  __REG32                 : 1;
  __REG32  MDIR           : 1;
  __REG32  MXTD           : 1;
  __REG32                 :16;
} __canifmsk2_bits;

/* CAN IFn Arbitration 1 (CANIFnARB1) */
typedef struct {
  __REG32  ID             :16;
  __REG32                 :16;
} __canifarb1_bits;

/* CAN IFn Arbitration 2 (CANIFnARB2) */
typedef struct {
  __REG32  ID             :13;
  __REG32  DIR            : 1; 
  __REG32  XTD            : 1;     
  __REG32  MSGVAL         : 1;
  __REG32                 :16;
} __canifarb2_bits;

/* CAN IFn Message Control (CANIFnMCTL) */
typedef struct {
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
} __canifmctl_bits;

/* CAN IFn Data A1 (CANIFnDA1) */
/* CAN IFn Data A2 (CANIFnDA2) */
/* CAN IFn Data B1 (CANIFnDB1) */
/* CAN IFn Data B2 (CANIFnDB2) */
typedef struct {
  __REG32  DATA           :16;
  __REG32                 :16;
} __canifdxy_bits;

/* CAN Transmission Request 1 (CANTXRQ1) */
/* CAN Transmission Request 2 (CANTXRQ2) */
typedef struct {
  __REG32  TXRQST         :16;
  __REG32                 :16;
} __cantxrqx_bits;

/* CAN New Data 1 (CANNWDA1) */
/* CAN New Data 2 (CANNWDA2) */
typedef struct {
  __REG32  NEWDAT         :16;
  __REG32                 :16;
} __cannwdax_bits;

/* CAN Message 1 Interrupt Pending (CANMSG1INT) */
/* CAN Message 2 Interrupt Pending (CANMSG2INT) */
typedef struct {
  __REG32  INTPND         :16;
  __REG32                 :16;
} __canmsgxint_bits;

/* CAN Message 1 Valid (CANMSG1VAL) */
/* CAN Message 2 Valid (CANMSG1VAL) */
typedef struct {
  __REG32  MSGVAL         :16;
  __REG32                 :16;
} __canmsgxval_bits;

/* Ethernet MAC Raw Interrupt Status (MACRIS) */
/* Ethernet MAC Interrupt Acknowledge (MACIACK) */
typedef struct {
  __REG32  RXINT          : 1;
  __REG32  TXER           : 1;     
  __REG32  TXEMP          : 1; 
  __REG32  FOV            : 1;   
  __REG32  RXER           : 1; 
  __REG32  MDINT          : 1;     
  __REG32  PHYINT         : 1;     
  __REG32                 :25;    
} __macris_bits;

/* Ethernet MAC Interrupt Mask (MACIM) */
typedef struct {
  __REG32  RXINTM         : 1;
  __REG32  TXERM          : 1;     
  __REG32  TXEMPM         : 1; 
  __REG32  FOVM           : 1;   
  __REG32  RXERM          : 1; 
  __REG32  MDINTM         : 1;     
  __REG32  PHYINTM        : 1;     
  __REG32                 :25;   
} __macim_bits;

/* Ethernet MAC Receive Control (MACRCTL) */
typedef struct {
  __REG32  RXEN           : 1; 
  __REG32  AMUL           : 1; 
  __REG32  PRMS           : 1;   
  __REG32  BADCRC         : 1;     
  __REG32  RSTFIFO        : 1;     
  __REG32                 :27;    
} __macrctl_bits;

/* Ethernet MAC Transmit Control (MACTCTL) */
typedef struct {
  __REG32  TXEN           : 1;   
  __REG32  PADEN          : 1; 
  __REG32  CRC            : 1;   
  __REG32                 : 1; 
  __REG32  DUPLEX         : 1;   
  __REG32                 : 27;   
} __mactctl_bits;

/* Ethernet MAC Data (MACDATA) */
typedef struct {
  __REG32  DATA           :32;   
} __macdata_bits;

/* Ethernet MAC Individual Address 0 (MACIA0) */
typedef struct {
  __REG32  MACOCT1        : 8;
  __REG32  MACOCT2        : 8;
  __REG32  MACOCT3        : 8;
  __REG32  MACOCT4        : 8;   
} __macia0_bits;

/* Ethernet MAC Individual Address 1 (MACIA1) */
typedef struct {
  __REG32  MACOCT5        : 8;
  __REG32  MACOCT6        : 8;    
  __REG32                 :16;   
} __macia1_bits;

/* Ethernet MAC Threshold (MACTHR) */
typedef struct {
  __REG32  THRESH         : 6;  
  __REG32                 :26;   
} __macthr_bits;

/* Ethernet MAC Management Control (MACMCTL) */
typedef struct {
  __REG32  START          : 1; 
  __REG32  WRITE          : 1;   
  __REG32                 : 1; 
  __REG32  REGADR         : 5;   
  __REG32                 :24;   
} __macmctl_bits;

/* Ethernet MAC Management Divider (MACMDV) */
typedef struct {
  __REG32  DIV            : 8;
  __REG32                 :24;      
} __macmdv_bits;

/* Ethernet MAC Management Address (MACMADD) */
typedef struct {
  __REG32  PHYADR         : 5;
  __REG32                 :27;      
} __macmadd_bits;

/* Ethernet MAC Management Transmit Data (MACMTXD) */
typedef struct {
  __REG32  MDTX           :16;
  __REG32                 :16;      
} __macmtxd_bits;

/* Ethernet MAC Management Receive Data (MACMRXD) */
typedef struct {
  __REG32  MDRX           :16;   
  __REG32                 :16;   
} __macmrxd_bits;

/* Ethernet MAC Number of Packets (MACNP) */
typedef struct {
  __REG32  NPR            : 6;   
  __REG32                 :26;   
} __macnp_bits;

/* Ethernet MAC Transmission Request (MACTR) */
typedef struct {
  __REG32  NEWTX          : 1;   
  __REG32                 :31;   
} __mactr_bits;

/* Ethernet MAC Timer Support (MACTS) */
typedef struct {
  __REG32  TSEN           : 1;
  __REG32                 :31;      
} __macts_bits;

/* Ethernet MAC LED Encoding (MACLED) */
typedef struct {
  __REG32  LED0           : 4;
  __REG32                 : 4;
  __REG32  LED1           : 4;
  __REG32                 :20;      
} __macled_bits;

/* Ethernet PHY MDIX (MDIX) */
typedef struct {
  __REG32  EN             : 1;
  __REG32                 :31;      
} __mdix_bits;

/* USB Device Functional Address (USBFADDR) */
typedef struct {
  __REG8   FUNCADDR       : 7;
  __REG8                  : 1;
} __usbfaddr_bits;

/* USB Power (USBPOWER) */
typedef struct {
  __REG8   PWRDNPHY       : 1;
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   RESET          : 1;
  __REG8                  : 2;
  __REG8   SOFTCONN       : 1;
  __REG8   ISOUP          : 1;
} __usbpower_bits;

/* USB Transmit Interrupt Status (USBTXIS) */
typedef struct {
  __REG16  EP0            : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbtxis_bits;

/* USB Receive Interrupt Status (USBRXIS) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxis_bits;

/* USB Transmit Interrupt Enable (USBTXIE) */
typedef struct {
  __REG16  EP0            : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbtxie_bits;

/* USB Receive Interrupt Enable (USBRXIE) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxie_bits;

/* USB General Interrupt Status (USBIS) */
typedef struct {
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   BABBLE_RESET   : 1;
  __REG8   SOF            : 1;
  __REG8   CONN           : 1;
  __REG8   DISCON         : 1;
  __REG8   SESREQ         : 1;
  __REG8   VBUSERR        : 1;
} __usbis_bits;

/* USB Interrupt Enable (USBIE) */
typedef struct {
  __REG8   SUSPEND        : 1;
  __REG8   RESUME         : 1;
  __REG8   BABBLE_RESET   : 1;
  __REG8   SOF            : 1;
  __REG8   CONN           : 1;
  __REG8   DISCON         : 1;
  __REG8   SESREQ         : 1;
  __REG8   VBUSERR        : 1;
} __usbie_bits;

/* USB Frame Value (USBFRAME) */
typedef struct {
  __REG16  FRAME          :11;
  __REG16                 : 5;
} __usbframe_bits;

/* USB Endpoint Index (USBEPIDX) */
typedef struct {
  __REG8   EPIDX          : 4;
  __REG8                  : 4;
} __usbepidx_bits;

/* USB Test Mode (USBTEST) */
typedef struct {
  __REG8                  : 5;
  __REG8   FORCEFS        : 1;
  __REG8   FIFOACC        : 1;
  __REG8   FORCEH         : 1;
} __usbtest_bits;

/* USB Device Control (USBDEVCTL) */
typedef struct {
  __REG8   SESSION        : 1;
  __REG8   HOSTREQ        : 1;
  __REG8   HOST           : 1;
  __REG8   VBUS           : 2;
  __REG8   LSDEV          : 1;
  __REG8   FSDEV          : 1;
  __REG8   DEV            : 1;
} __usbdevctl_bits;

/* USB Transmit Dynamic FIFO Sizing (USBTXFIFOSZ)
   USB Receive Dynamic FIFO Sizing (USBRXFIFOSZ) */
typedef struct {
  __REG8   SIZE           : 4;
  __REG8   DPB            : 1;
  __REG8                  : 3;
} __usbtxfifosz_bits;

/* USB Transmit FIFO Start Address (USBTXFIFOADD)
   USB Receive FIFO Start Address (USBRXFIFOADD) */
typedef struct {
  __REG16  ADDR           : 9;
  __REG16                 : 7;
} __usbtxfifoadd_bits;

/* USB Connect Timing (USBCONTIM) */
typedef struct {
  __REG8   WTID           : 4;
  __REG8   WTCON          : 4;
} __usbcontim_bits;

/* USB Transmit Functional Address Endpoint x (USBTXFUNCADDRx) */
/* USB Receive Functional Address Endpoint x (USBRXFUNCADDRx) */
typedef struct {
  __REG8   ADDR           : 7;
  __REG8                  : 1;
} __usbtxfuncaddr_bits;

/* USB Transmit Hub Address Endpoint x (USBTXHUBADDRx) */
/* USB Receive Hub Address Endpoint x (USBRXHUBADDRx) */
typedef struct {
  __REG8   ADDR           : 7;
  __REG8   MULTTRAN       : 1;
} __usbtxhubaddr_bits;

/* USB Transmit Hub Port Endpoint x (USBTXHUBPORTx) */
/* USB Receive Hub Port Endpoint x (USBRXHUBPORTx) */
typedef struct {
  __REG8   PORT           : 7;
  __REG8                  : 1;
} __usbtxhubport_bits;

/* USB Maximum Transmit Data Endpoint x (USBTXMAXPx) */
typedef struct {
  __REG16  MAXLOAD        :11;
  __REG16                 : 5;
} __usbtxmaxp_bits;

/* USB Control and Status Endpoint 0 Low (USBCSRL0) */
typedef union {
  /* USBCSRL0 */
  struct {
    __REG8   RXRDY          : 1;
    __REG8   TXRDY          : 1;
    __REG8   STALLED        : 1;
    __REG8   SETUP          : 1;
    __REG8   ERROR          : 1;
    __REG8   REQPKT         : 1;
    __REG8   STATUS         : 1;
    __REG8   NAKTO          : 1;
  };
  /* USBDCSRL0 */
  struct {
    __REG8   RXRDY          : 1;
    __REG8   TXRDY          : 1;
    __REG8   STALLED        : 1;
    __REG8   DATAEND        : 1;
    __REG8   SETEND         : 1;
    __REG8   STALL          : 1;
    __REG8   RXRDYC         : 1;
    __REG8   SETENDC        : 1;
  }__usbdcsrl;
} __usbcsrl0_bits;

/* USB Control and Status Endpoint 0 High (USBCSRH0) */
typedef struct {
  __REG8   FLUSH          : 1;
  __REG8   DT             : 1;
  __REG8   DTWE           : 1;
  __REG8                  : 5;
} __usbcsrh0_bits;

/* USB Receive Byte Count Endpoint 0 (USBCOUNT0) */
typedef struct {
  __REG8   COUNT          : 7;
  __REG8                  : 1;
} __usbcount0_bits;

/* USB Type Endpoint 0 (USBTYPE0) */
typedef struct {
  __REG8                  : 6;
  __REG8   SPEED          : 2;
} __usbtype0_bits;

/* USB NAK Limit (USBNAKLMT) */
typedef struct {
  __REG8   NAKLMT         : 5;
  __REG8                  : 3;
} __usbnaklmt_bits;

/* USB Transmit Control and Status Endpoint x Low (USBTXCSRLx) */
typedef union {
  /* USBTXCSRLx */
  struct {
    __REG8   TXRDY          : 1;
    __REG8   FIFONE         : 1;
    __REG8   ERROR          : 1;
    __REG8   FLUSH          : 1;
    __REG8   SETUP          : 1;
    __REG8   STALLED        : 1;
    __REG8   CLRDT          : 1;
    __REG8   NAKTO          : 1;
  };
  /* USBDTXCSRLx */
  struct {
    __REG8   TXRDY          : 1;
    __REG8   FIFONE         : 1;
    __REG8   UNDRN          : 1;
    __REG8   FLUSH          : 1;
    __REG8   STALL          : 1;
    __REG8   STALLED        : 1;
    __REG8   CLRDT          : 1;
    __REG8                  : 1;
  }__usbdtxcsrl;
} __usbtxcsrl_bits;

/* USB Transmit Control and Status Endpoint x High (USBTXCSRHx) */
typedef struct {
  __REG8   DT             : 1;
  __REG8   DTWE           : 1;
  __REG8   DMAMOD         : 1;
  __REG8   FDT            : 1;
  __REG8   DMAEN          : 1;
  __REG8   MODE           : 1;
  __REG8   ISO            : 1;
  __REG8   AUTOSET        : 1;
} __usbtxcsrh_bits;

/* USB Maximum Receive Data Endpoint x (USBRXMAXPx) */
typedef struct {
  __REG16  MAXLOAD        :11;
  __REG16                 : 5;
} __usbrxmaxp_bits;

/* USB Receive Control and Status Endpoint x Low (USBRXCSRLx) */
typedef union {
  /* USBRXCSRLx */
  struct
  {
    __REG8   RXRDY          : 1;
    __REG8   FULL           : 1;
    __REG8   ERROR          : 1;
    __REG8   DATAERR_NAKTO  : 1;
    __REG8   FLUSH          : 1;
    __REG8   REQPKT         : 1;
    __REG8   STALLED        : 1;
    __REG8   CLRDT          : 1;
  };
  /* USBDRXCSRLx */
  struct
  {
    __REG8   RXRDY          : 1;
    __REG8   FULL           : 1;
    __REG8   OVER           : 1;
    __REG8   DATAERR        : 1;
    __REG8   FLUSH          : 1;
    __REG8   STALL          : 1;
    __REG8   STALLED        : 1;
    __REG8   CLRDT          : 1;
  }__usbdrxcsrl;
} __usbrxcsrl_bits;

/* USB Receive Control and Status Endpoint x High (USBRXCSRHx) */
typedef union {
  /* USBRXCSRHx */
  struct {
    __REG8                  : 1;
    __REG8   DT             : 1;
    __REG8   DTWE           : 1;
    __REG8   DMAMOD         : 1;
    __REG8   PIDERR         : 1;
    __REG8   DMAEN          : 1;
    __REG8   AUTORQ         : 1;
    __REG8   AUTOCL         : 1;
  };
  /* USBDRXCSRHx */
  struct {
    __REG8                  : 3;
    __REG8   DMAMOD         : 1;
    __REG8   DISNYET_PIDERR : 1;
    __REG8   DMAEN          : 1;
    __REG8   ISO            : 1;
    __REG8   AUTOCL         : 1;
  }__usbdrxcsrh;
} __usbrxcsrh_bits;

/* USB Receive Byte Count Endpoint x (USBRXCOUNTx) */
typedef struct {
  __REG16  COUNT          :13;
  __REG16                 : 3;
} __usbrxcount_bits;

/* USB Host Transmit Configure Type Endpoint x (USBTXTYPEx) */
/* USB Host Configure Receive Type Endpoint x (USBRXTYPEx) */
typedef struct {
  __REG8   TEP            : 4;
  __REG8   PROTO          : 2;
  __REG8   SPEED          : 2;
} __usbtxtype_bits;

/* USB Receive Double Packet Buffer Disable (USBRXDPKTBUFDIS) 
   USB Transmit Double Packet Buffer Disable (USBTXDPKTBUFDIS) */
typedef struct {
  __REG16                 : 1;
  __REG16  EP1            : 1;
  __REG16  EP2            : 1;
  __REG16  EP3            : 1;
  __REG16  EP4            : 1;
  __REG16  EP5            : 1;
  __REG16  EP6            : 1;
  __REG16  EP7            : 1;
  __REG16  EP8            : 1;
  __REG16  EP9            : 1;
  __REG16  EP10           : 1;
  __REG16  EP11           : 1;
  __REG16  EP12           : 1;
  __REG16  EP13           : 1;
  __REG16  EP14           : 1;
  __REG16  EP15           : 1;
} __usbrxdpktbufdis_bits;

/* USB External Power Control (USBEPC) */
typedef struct {
  __REG32  EPEN           : 2;  
  __REG32  EPENDE         : 1;  
  __REG32                 : 1;  
  __REG32  PFLTEN         : 1;  
  __REG32  PFLTSEN        : 1;  
  __REG32  PFLTAEN        : 1;  
  __REG32                 : 1;  
  __REG32  PFLTACT        : 2;  
  __REG32                 :22;  
} __usbepc_bits;

/* USB External Power Control Raw Interrupt Status (USBEPCRIS) 
   USB External Power Control Interrupt Mask (USBEPCIM) 
   USB External Power Control Interrupt Status and Clear (USBEPCISC) */
typedef struct {
  __REG32  PF             : 1;  
  __REG32                 :31;  
} __usbepcris_bits;

/* USB Device RESUME Raw Interrupt Status (USBDRRIS)
   USB Device RESUME Interrupt Mask (USBDRIM)
   USB Device RESUME Interrupt Status and Clear (USBDRISC) */
typedef struct {
  __REG32  RESUME         : 1;  
  __REG32                 :31;  
} __usbdrris_bits;

/* USB General-Purpose Control and Status (USBGPCS) */
typedef struct {
  __REG32  DEVMOD         : 1;  
  __REG32  DEVMODOTG      : 1;  
  __REG32                 :30;  
} __usbgpcs_bits;

/* USB VBUS Droop Control (USBVDC) */
typedef struct {
  __REG32  VBDEN          : 1;  
  __REG32                 :31;  
} __usbvdc_bits;

/* USB VBUS Droop Control Raw Interrupt Status (USBVDCRIS)
   USB VBUS Droop Control Interrupt Mask (USBVDCIM)
   USB VBUS Droop Control Interrupt Status and Clear (USBVDCISC) */
typedef struct {
  __REG32  VD             : 1;  
  __REG32                 :31;  
} __usbvdcris_bits;

/* USB ID Valid Detect Raw Interrupt Status (USBIDVRIS)
   USB ID Valid Detect Interrupt Mask (USBIDVIM)
   USB ID Valid Detect Interrupt Status and Clear (USBIDVISC) */
typedef struct {
  __REG32  ID             : 1;  
  __REG32                 :31;  
} __usbidvris_bits;

/* USB DMA Select (USBDMASEL) */
typedef struct {
  __REG32  DMAARX         : 4;  
  __REG32  DMAATX         : 4;  
  __REG32  DMABRX         : 4;  
  __REG32  DMABTX         : 4;  
  __REG32  DMACRX         : 4;  
  __REG32  DMACTX         : 4;  
  __REG32                 : 8;  
} __usbdmasel_bits;

/* Analog Comparator Masked Interrupt Status (ACMIS)
   Analog Comparator Raw Interrupt Status (ACRIS)
   Analog Comparator Interrupt Enable (ACINTEN) */
typedef struct {
  __REG32  IN0            : 1;
  __REG32  IN1            : 1;
  __REG32  IN2            : 1;
  __REG32                 :29;
} __acmis_bits;

/* Analog Comparator Reference Voltage Control (ACREFCTL) */
typedef struct {
  __REG32  VREF           : 4;
  __REG32                 : 4;
  __REG32  RNG            : 1;
  __REG32  EN             : 1;
  __REG32                 :22;
} __acrefctl_bits;

/* Analog Comparator Status (ACSTAT) */
typedef struct {
  __REG32                 : 1;
  __REG32  OVAL           : 1;
  __REG32                 :30;
} __acstat_bits;

/* Analog Comparator Control (ACCTL) */
typedef struct {
  __REG32                 : 1;
  __REG32  CINV           : 1;
  __REG32  ISEN           : 2;
  __REG32  ISLVAL         : 1;
  __REG32  TSEN         	: 2;
  __REG32  TSLVAL         : 1;
  __REG32                 : 1;
  __REG32  ASRCP          : 2;
  __REG32  TOEN           : 1;
  __REG32                 :20;
} __acctl_bits;

/* Pulse Width Modulator Master Control (PWMCTL) */
typedef struct {
  __REG32  GLOBALSYNC0    : 1;
  __REG32  GLOBALSYNC1    : 1;
  __REG32  GLOBALSYNC2    : 1;
  __REG32  GLOBALSYNC3    : 1;
  __REG32                 :28;
} __pwmctl_bits;

/* Pulse Width Modulator Time Base Sync (PWMSYNC) */
typedef struct {
  __REG32  SYNC0          : 1;
  __REG32  SYNC1          : 1;
  __REG32  SYNC2          : 1;
  __REG32  SYNC3          : 1;
  __REG32                 :28;
} __pwmsync_bits;

/* Pulse Width Modulator Output Enable (PWMENABLE) */
typedef struct {
  __REG32  PWM0EN         : 1;
  __REG32  PWM1EN         : 1;
  __REG32  PWM2EN         : 1;
  __REG32  PWM3EN         : 1;    
  __REG32  PWM4EN         : 1;    
  __REG32  PWM5EN         : 1;    
  __REG32  PWM6EN         : 1;    
  __REG32  PWM7EN         : 1;    
  __REG32                 :24;
} __pwmenable_bits;

/* Pulse Width Modulator Output Inversion (PWMINVERT) */
typedef struct {
  __REG32  PWM0INV        : 1;
  __REG32  PWM1INV        : 1;
  __REG32  PWM2INV        : 1;
  __REG32  PWM3INV        : 1;    
  __REG32  PWM4INV        : 1;    
  __REG32  PWM5INV        : 1;    
  __REG32  PWM6INV        : 1;    
  __REG32  PWM7INV        : 1;    
  __REG32                 :24;
} __pwminvert_bits;

/* Pulse Width Modulator Output Fault (PWMFAULT) */
typedef struct {
  __REG32  FAULT0         : 1;
  __REG32  FAULT1         : 1;
  __REG32  FAULT2         : 1;
  __REG32  FAULT3         : 1;    
  __REG32  FAULT4         : 1;    
  __REG32  FAULT5         : 1;    
  __REG32  FAULT6         : 1;    
  __REG32  FAULT7         : 1;    
  __REG32                 :24;
} __pwmfault_bits;

/* Pulse Width Modulator Interrupt Enable (PWMINTEN) */
typedef struct {
  __REG32  INTPWM0        : 1;
  __REG32  INTPWM1        : 1;
  __REG32  INTPWM2        : 1;
  __REG32  INTPWM3        : 1;
  __REG32                 :12;
  __REG32  INTFAULT0      : 1;    
  __REG32  INTFAULT1      : 1;    
  __REG32  INTFAULT2      : 1;    
  __REG32  INTFAULT3      : 1;    
  __REG32                 :12;
} __pwminten_bits;

/* Pulse Width Modulator Interrupt Status (PWMRIS) */
typedef struct {
  __REG32  INTPWM0        : 1;
  __REG32  INTPWM1        : 1;
  __REG32  INTPWM2        : 1;
  __REG32  INTPWM3        : 1;
  __REG32                 :12;
  __REG32  INTFAULT0      : 1;    
  __REG32  INTFAULT1      : 1;    
  __REG32  INTFAULT2      : 1;    
  __REG32  INTFAULT3      : 1;    
  __REG32                 :12;
} __pwmris_bits;

/* Pulse Width Modulator Interrupt Status and Clear (PWMISC) */
typedef struct {
  __REG32  INTPWM0        : 1;
  __REG32  INTPWM1        : 1;
  __REG32  INTPWM2        : 1;
  __REG32  INTPWM3        : 1;
  __REG32                 :12;
  __REG32  INTFAULT0      : 1;    
  __REG32  INTFAULT1      : 1;    
  __REG32  INTFAULT2      : 1;    
  __REG32  INTFAULT3      : 1;    
  __REG32                 :12;
} __pwmisc_bits;

/* Pulse Width Modulator Status (PWMSTATUS) */
typedef struct {
  __REG32  FAULT0         : 1;   
  __REG32  FAULT1         : 1;   
  __REG32  FAULT2         : 1;   
  __REG32  FAULT3         : 1;   
  __REG32                 :28;
} __pwmstatus_bits;

/* PWM Fault Condition Value (PWMFAULTVAL) */
typedef struct {
  __REG32  PWM0           : 1;   
  __REG32  PWM1           : 1;   
  __REG32  PWM2           : 1;   
  __REG32  PWM3           : 1;   
  __REG32  PWM4           : 1;   
  __REG32  PWM5           : 1;   
  __REG32  PWM6           : 1;   
  __REG32  PWM7           : 1;   
  __REG32                 :24;
} __pwmfaultval_bits;

/* PWM Enable Update (PWMENUPD) */
typedef struct {
  __REG32  ENUPD0         : 2;   
  __REG32  ENUPD1         : 2;   
  __REG32  ENUPD2         : 2;   
  __REG32  ENUPD3         : 2;   
  __REG32  ENUPD4         : 2;   
  __REG32  ENUPD5         : 2;   
  __REG32  ENUPD6         : 2;   
  __REG32  ENUPD7         : 2;   
  __REG32                 :16;
} __pwmenupd_bits;

/* Pulse Width Modulator 0 Control (PWM0CTL) */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  MODE           : 1;
  __REG32  DEBUG          : 1;
  __REG32  LOADUPD        : 1;    
  __REG32  CMPAUPD        : 1;
  __REG32  CMPBUPD        : 1;
  __REG32  GENAUPD        : 2;
  __REG32  GENBUPD        : 2;
  __REG32  DBCTLUPD       : 2;
  __REG32  DBRISEUPD      : 2;
  __REG32  DBFALLUPD      : 2;
  __REG32  FLTSRC         : 1;
  __REG32  MINFLTPER      : 1;
  __REG32  LATCH          : 1;
  __REG32                 :13;                  
} __pwm0ctl_bits;

/* Pulse Width Modulator 0 Interrupt/Trigger Enable (PWM0INTEN) */
typedef struct {
  __REG32  INTCNTZERO     : 1;
  __REG32  INTCNTLOAD     : 1;
  __REG32  INTCMPAU       : 1;
  __REG32  INTCMPAD       : 1;    
  __REG32  INTCMPBU       : 1;
  __REG32  INTCMPBD       : 1;
  __REG32                 : 2;   
  __REG32  TRCNTZERO      : 1;
  __REG32  TRCNTLOAD      : 1;
  __REG32  TRCMPAU        : 1;
  __REG32  TRCMPAD        : 1;    
  __REG32  TRCMPBU        : 1;
  __REG32  TRCMPBD        : 1;  
  __REG32                 :18;                     
} __pwm0inten_bits;

/* Pulse Width Modulator 0 Raw Interrupt Status (PWM0RIS) */
typedef struct {
  __REG32  INTCNTZERO     : 1;
  __REG32  INTCNTLOAD     : 1;
  __REG32  INTCMPAU       : 1;
  __REG32  INTCMPAD       : 1;    
  __REG32  INTCMPBU       : 1;
  __REG32  INTCMPBD       : 1;    
  __REG32                 :26;                     
} __pwm0ris_bits;

/* Pulse Width Modulator 0 Interrupt Status and Clear (PWM0ISC) */
typedef struct {
  __REG32  INTCNTZERO     : 1;
  __REG32  INTCNTLOAD     : 1;
  __REG32  INTCMPAU       : 1;
  __REG32  INTCMPAD       : 1;    
  __REG32  INTCMPBU       : 1;
  __REG32  INTCMPBD       : 1;    
  __REG32                 :26;                     
} __pwm0isc_bits;

/* Pulse Width Modulator 0 Load (PWM0LOAD) */
typedef struct {
  __REG32  LOAD           :16;    
  __REG32                 :16;                     
} __pwm0load_bits;

/* Pulse Width Modulator 0 Counter (PWM0COUNT) */
typedef struct {
  __REG32  COUNT          :16;    
  __REG32                 :16;                     
} __pwm0count_bits;

/* Pulse Width Modulator 0 Compare A (PWM0CMPA) */
typedef struct {
  __REG32  COMPA          :16;    
  __REG32                 :16;                     
} __pwm0cmpa_bits;

/* Pulse Width Modulator 0 Compare B (PWM0CMPB) */
typedef struct {
  __REG32  COMPB          :16;    
  __REG32                 :16;                     
} __pwm0cmpb_bits;

/* Pulse Width Modulator 0 Generator A Control (PWM0GENA) */
typedef struct {
  __REG32  ACTZERO        : 2;
  __REG32  ACTLOAD        : 2;
  __REG32  ACTCMPAU       : 2;
  __REG32  ACTCMPAD       : 2;    
  __REG32  ACTCMPBU       : 2;
  __REG32  ACTCMPBD       : 2;    
  __REG32                 :20;                     
} __pwm0gena_bits;

/* Pulse Width Modulator 0 Generator B Control (PWM0GENB) */
typedef struct {
  __REG32  ACTZERO        : 2;
  __REG32  ACTLOAD        : 2;
  __REG32  ACTCMPAU       : 2;
  __REG32  ACTCMPAD       : 2;    
  __REG32  ACTCMPBU       : 2;
  __REG32  ACTCMPBD       : 2;    
  __REG32                 :20;                     
} __pwm0genb_bits;

/* Pulse Width Modulator 0 Dead-Band Control (PWM0DBCTL) */
typedef struct {
  __REG32  ENABLE         : 1;    
  __REG32                 :31;                     
} __pwm0dbctl_bits;

/* Pulse Width Modulator 0 Dead-Band Rising-Edge Delay (PWM0DBRISE) */
typedef struct {
  __REG32  RISEDELAY      :12;    
  __REG32                 :20;                     
} __pwm0dbrise_bits;

/* Pulse Width Modulator 0 Dead-Band Falling-Edge Delay (PWM0DBFALL) */
typedef struct {
  __REG32  FALLDELAY      :12;    
  __REG32                 :20;                     
} __pwm0dbfall_bits;

/* PWMx Fault Source 0 (PWM0FLTSRC0) */
typedef struct {
  __REG32  FAULT0      		: 1;    
  __REG32  FAULT1      		: 1;    
  __REG32  FAULT2      		: 1;    
  __REG32  FAULT3      		: 1;    
  __REG32                 :28;                     
} __pwm0fltstat_bits;

/* PWM0 Fault Source x (PWM0FLTSRCx) */
typedef struct {
  __REG32  DCMP0       		: 1;    
  __REG32  DCMP1      		: 1;    
  __REG32  DCMP2      		: 1;    
  __REG32  DCMP3      		: 1;    
  __REG32  DCMP4      		: 1;    
  __REG32  DCMP5      		: 1;    
  __REG32  DCMP6      		: 1;    
  __REG32  DCMP7      		: 1;    
  __REG32                 :24;                     
} __pwm0fltsrc_bits;

/* PWM0 Minimum Fault Period (PWM0MINFLTPER) */
typedef struct {
  __REG32  MFP       			:16;    
  __REG32                 :16;                     
} __pwm0minfltper_bits;

/* QEI Control (QEICTL) */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  SWAP           : 1;
  __REG32  SIGMODE        : 1;
  __REG32  CAPMODE        : 1;
  __REG32  RESMODE        : 1;
  __REG32  VELEN          : 1;
  __REG32  VELDIV         : 3;
  __REG32  INVA           : 1;
  __REG32  INVB           : 1;
  __REG32  INVI           : 1;
  __REG32  STALLEN        : 1;
  __REG32  FILTEN         : 1;
  __REG32                 : 2;
  __REG32  FILTCNT        : 4;
  __REG32                 :12;
} __qeictl_bits;

/* QEI Status (QEISTAT) */
typedef struct {
  __REG32  ERROR          : 1;
  __REG32  DIRECTION      : 1;
  __REG32                 :30;
} __qeistat_bits;

/* QEI Interrupt Enable (QEIINTEN) */
typedef struct {
  __REG32  INTINDEX       : 1;
  __REG32  INTTIMER       : 1;
  __REG32  INTDIR         : 1;
  __REG32  INTERROR       : 1;
  __REG32                 :28;
} __qeiinten_bits;

/* QEI Raw Interrupt Status (QEIRIS) */
typedef struct {
  __REG32  INTINDEX       : 1;
  __REG32  INTTIMER       : 1;
  __REG32  INTDIR         : 1;
  __REG32  INTERROR       : 1;
  __REG32                 :28;
} __qeiris_bits;

/* QEI Interrupt Status and Clear (QEIISC) */
typedef struct {
  __REG32  INTINDEX       : 1;
  __REG32  INTTIMER       : 1;
  __REG32  INTDIR         : 1;
  __REG32  INTERROR       : 1;
  __REG32                 :28;
} __qeiisc_bits;

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

/* Debug Halting Control and Status Register */
typedef union {
  /* DHSR */
  struct {
    __REG32  C_DEBUGEN    : 1;
    __REG32  C_HALT       : 1;
    __REG32  C_STEP       : 1;
    __REG32  C_MASKINTS   : 1;
    __REG32               : 1;
    __REG32  C_SNAPSTALL  : 1;
    __REG32               :10;
    __REG32  S_REGRDY     : 1;
    __REG32  S_HALT       : 1;
    __REG32  S_SLEEP      : 1;
    __REG32  S_LOCKUP     : 1;
    __REG32               : 4;
    __REG32  S_RETIRE_ST  : 1;
    __REG32  S_RESET_ST   : 1;
    __REG32               : 6;
  };
  /* DHCR */
  struct {
    __REG32               :16;
    __REG32  DBGKEY       :16;
  };
} __dhsr_bits;

/* Debug Core Selector Register */
typedef struct {
  __REG32  REGSEL         : 5;
  __REG32                 :11;
  __REG32  REGWnR         : 1;
  __REG32                 :15;
} __dcrsr_bits;

/* Debug Exception and Monitor Control Register */
typedef struct {
  __REG32  VC_CORERESET   : 1;
  __REG32                 : 3;
  __REG32  VC_MMERR       : 1;
  __REG32  VC_NOCPERR     : 1;
  __REG32  VC_CHKERR      : 1;
  __REG32  VC_STATERR     : 1;
  __REG32  VC_BUSERR      : 1;
  __REG32  VC_INTERR      : 1;
  __REG32  VC_HARDERR     : 1;
  __REG32                 : 5;
  __REG32  MON_EN         : 1;
  __REG32  MON_PEND       : 1;
  __REG32  MON_STEP       : 1;
  __REG32  MON_REQ        : 1;
  __REG32                 : 4;
  __REG32  TRCENA         : 1;
  __REG32                 : 7;
} __demcr_bits;

/* Flash Patch Control Register */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32  KEY            : 1;
  __REG32                 : 2;
  __REG32  NUM_CODE       : 4;
  __REG32  NUM_LIT        : 4;
  __REG32                 :20;
} __fp_ctrl_bits;

/* Flash Patch Remap Register */
typedef struct {
  __REG32                 : 5;
  __REG32  REMAP          :24;
  __REG32                 : 3;
} __fp_remap_bits;

/* Flash Patch Comparator Registers */
typedef struct {
  __REG32  ENABLE         : 1;
  __REG32                 : 1;
  __REG32  REMAP          :27;
  __REG32                 : 1;
  __REG32  REPLACE        : 2;
} __fp_comp_bits;

/* DWT Control Register */
typedef struct {
  __REG32  CYCCNTENA      : 1;
  __REG32  POSTPRESET     : 4;
  __REG32  POSTCNT        : 4;
  __REG32  CYCTAP         : 1;
  __REG32  SYNCTAP        : 2;
  __REG32  PCSAMPLEENA    : 1;
  __REG32                 : 3;
  __REG32  EXCTRCENA      : 1;
  __REG32  CPIEVTENA      : 1;
  __REG32  EXCEVTENA      : 1;
  __REG32  SLEEPEVTENA    : 1;
  __REG32  LSUEVTENA      : 1;
  __REG32  FOLDEVTENA     : 1;
  __REG32  CYCEVTEN       : 1;
  __REG32                 : 5;
  __REG32  NUMCOMP        : 4;
} __dwt_ctrl_bits;

/* DWT Mask Registers */
typedef struct {
  __REG32  MASK           : 4;
  __REG32                 :28;
} __dwt_mask_bits;

/* DWT Function Registers */
typedef struct {
  __REG32  FUNCTION       : 4;
  __REG32                 : 1;
  __REG32  EMITRANGE      : 1;
  __REG32                 : 1;
  __REG32  CYCMATCH       : 1;
  __REG32                 :24;
} __dwt_function_bits;

/* ITM Trace Enable Register */
typedef struct {
  __REG32  STIMULUS_MASK0 : 1;
  __REG32  STIMULUS_MASK1 : 1;
  __REG32  STIMULUS_MASK2 : 1;
  __REG32  STIMULUS_MASK3 : 1;
  __REG32  STIMULUS_MASK4 : 1;
  __REG32  STIMULUS_MASK5 : 1;
  __REG32  STIMULUS_MASK6 : 1;
  __REG32  STIMULUS_MASK7 : 1;
  __REG32  STIMULUS_MASK8 : 1;
  __REG32  STIMULUS_MASK9 : 1;
  __REG32  STIMULUS_MASK10: 1;
  __REG32  STIMULUS_MASK11: 1;
  __REG32  STIMULUS_MASK12: 1;
  __REG32  STIMULUS_MASK13: 1;
  __REG32  STIMULUS_MASK14: 1;
  __REG32  STIMULUS_MASK15: 1;
  __REG32  STIMULUS_MASK16: 1;
  __REG32  STIMULUS_MASK17: 1;
  __REG32  STIMULUS_MASK18: 1;
  __REG32  STIMULUS_MASK19: 1;
  __REG32  STIMULUS_MASK20: 1;
  __REG32  STIMULUS_MASK21: 1;
  __REG32  STIMULUS_MASK22: 1;
  __REG32  STIMULUS_MASK23: 1;
  __REG32  STIMULUS_MASK24: 1;
  __REG32  STIMULUS_MASK25: 1;
  __REG32  STIMULUS_MASK26: 1;
  __REG32  STIMULUS_MASK27: 1;
  __REG32  STIMULUS_MASK28: 1;
  __REG32  STIMULUS_MASK29: 1;
  __REG32  STIMULUS_MASK30: 1;
  __REG32  STIMULUS_MASK31: 1;
} __itm_te_bits;

/* ITM Trace Privilege Register */
typedef struct {
  __REG32  PRIVILEGE_MASK : 4;
  __REG32                 :28;
} __itm_tp_bits;

/* ITM Control Register */
typedef struct {
  __REG32  ITMEN          : 1;
  __REG32  TSENA          : 1;
  __REG32  SYNCEN         : 1;
  __REG32  DWTEN          : 1;
  __REG32                 : 4;
  __REG32  TSPRESCALE     : 2;
  __REG32                 : 6;
  __REG32  ATBID          : 7;
  __REG32                 : 9;
} __itm_cr_bits;

/* ITM Integration Write Register */
typedef struct {
  __REG32  ATVALIDM       : 1;
  __REG32                 :31;
} __itm_iw_bits;

/* ITM Integration Read Register */
typedef struct {
  __REG32  ATREADYM       : 1;
  __REG32                 :31;
} __itm_ir_bits;

/* ITM Integration Mode Control Register */
typedef struct {
  __REG32  INTEGRATION    : 1;
  __REG32                 :31;
} __itm_imc_bits;

/* ITM Lock Status Register */
typedef struct {
  __REG32  PRESENT        : 1;
  __REG32  ACCESS         : 1;
  __REG32  BYTEACC        : 1;
  __REG32                 :29;
} __itm_lsr_bits;

/* Supported Port Sizes Register / Current */
typedef struct {
  __REG32  MAXPORTSIZE1   : 1;
  __REG32  MAXPORTSIZE2   : 1;
  __REG32  MAXPORTSIZE3   : 1;
  __REG32  MAXPORTSIZE4   : 1;
  __REG32                 :28;
} __tpiu_spsr_bits;

/* Current Output Speed Divisors Register */
typedef struct {
  __REG32  PRESCALER      :13;
  __REG32                 :19;
} __tpiu_cosdr_bits;

/* Selected Pin Protocol Register */
typedef struct {
  __REG32  PROTOCOL       : 2;
  __REG32                 :30;
} __tpiu_sppr_bits;

/* Formatter and Flush Status Register */
typedef struct {
  __REG32  FL_IN_PROG     : 1;
  __REG32  FT_STOPPED     : 1;
  __REG32  TC_PRESENT     : 1;
  __REG32  FT_NON_STOP    : 1;
  __REG32                 :28;
} __tpiu_ffsr_bits;

/* Integration Test Register-ITATBCTR2 */
typedef struct {
  __REG32  ATREADY1_2     : 1;
  __REG32                 :31;
} __tpiu_itatbctr2_bits;

/* Integration Test Register-ITATBCTR0 */
typedef struct {
  __REG32  ATVALID1_2     : 1;
  __REG32                 :31;
} __tpiu_itatbctr0_bits;

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
__IO_REG32_BIT(AITCR,             0xE000ED0C,__READ_WRITE ,__aircr_bits);
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
 ** Device Identification
 **
 ***************************************************************************/
__IO_REG32_BIT(DID0,              0x400FE000,__READ       ,__did0_bits);
__IO_REG32_BIT(DID1,              0x400FE004,__READ       ,__did1_bits);
__IO_REG32_BIT(DC0,               0x400FE008,__READ       ,__dc0_bits);
__IO_REG32_BIT(DC1,               0x400FE010,__READ       ,__dc1_bits);
__IO_REG32_BIT(DC2,               0x400FE014,__READ       ,__dc2_bits);
__IO_REG32_BIT(DC3,               0x400FE018,__READ       ,__dc3_bits);
__IO_REG32_BIT(DC4,               0x400FE01C,__READ       ,__dc4_bits);
__IO_REG32_BIT(DC5,               0x400FE020,__READ       ,__dc5_bits);
__IO_REG32_BIT(DC6,               0x400FE024,__READ       ,__dc6_bits);
__IO_REG32_BIT(DC7,               0x400FE028,__READ       ,__dc7_bits);
__IO_REG32_BIT(DC8,               0x400FE02C,__READ       ,__dc8_bits);

/***************************************************************************
 **
 ** Local Control
 **
 ***************************************************************************/
__IO_REG32_BIT(PBORCTL,           0x400FE030,__READ_WRITE ,__pborctl_bits);
__IO_REG32_BIT(LDOPCTL,           0x400FE034,__READ_WRITE ,__ldopctl_bits);
__IO_REG32_BIT(SRCR0,             0x400FE040,__READ_WRITE ,__srcr0_bits);
__IO_REG32_BIT(SRCR1,             0x400FE044,__READ_WRITE ,__srcr1_bits);
__IO_REG32_BIT(SRCR2,             0x400FE048,__READ_WRITE ,__srcr2_bits);
__IO_REG32_BIT(RIS,               0x400FE050,__READ       ,__ris_bits);
__IO_REG32_BIT(IMC,               0x400FE054,__READ_WRITE ,__imc_bits);
__IO_REG32_BIT(MISC,              0x400FE058,__READ_WRITE ,__misc_bits);
__IO_REG32_BIT(RESC,              0x400FE05C,__READ_WRITE ,__resc_bits);
__IO_REG32_BIT(RCC,               0x400FE060,__READ_WRITE ,__rcc_bits);
__IO_REG32_BIT(PLLCFG,            0x400FE064,__READ       ,__pllcfg_bits);
__IO_REG32_BIT(GPIOHBCTL,         0x400FE06C,__READ       ,__gpiohbctl_bits);
__IO_REG32_BIT(RCC2,              0x400FE070,__READ_WRITE ,__rcc2_bits);
__IO_REG32_BIT(MOSCCTL,           0x400FE07C,__READ_WRITE ,__moscctl_bits);

/***************************************************************************
 **
 ** System Control
 **
 ***************************************************************************/
__IO_REG32_BIT(RCGC0,             0x400FE100,__READ_WRITE ,__rcgc0_bits);
__IO_REG32_BIT(RCGC1,             0x400FE104,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(RCGC2,             0x400FE108,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(SCGC0,             0x400FE110,__READ_WRITE ,__rcgc0_bits);
__IO_REG32_BIT(SCGC1,             0x400FE114,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(SCGC2,             0x400FE118,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(DCGC0,             0x400FE120,__READ_WRITE ,__dcgc0_bits);
__IO_REG32_BIT(DCGC1,             0x400FE124,__READ_WRITE ,__rcgc1_bits);
__IO_REG32_BIT(DCGC2,             0x400FE128,__READ_WRITE ,__rcgc2_bits);
__IO_REG32_BIT(DSLPCLKCFG,        0x400FE144,__READ_WRITE ,__dslpclkcfg_bits);
__IO_REG32_BIT(PIOSCCAL,          0x400FE150,__READ_WRITE ,__piosccal_bits);
__IO_REG32_BIT(PIOSCSTAT,         0x400FE154,__READ       ,__pioscstat_bits);
__IO_REG32_BIT(I2SMCLKCFG,        0x400FE170,__READ_WRITE ,__i2smclkcfg_bits);
__IO_REG32_BIT(DC9,               0x400FE190,__READ       ,__dc9_bits);
__IO_REG32_BIT(NVMSTAT,           0x400FE1A0,__READ       ,__nvmstat_bits);

/***************************************************************************
 **
 ** Hibernation Module
 **
 ***************************************************************************/
__IO_REG32_BIT(HIBRTCC,           0x400FC000,__READ        ,__hibrtcc_bits);
__IO_REG32_BIT(HIBRTCM0,          0x400FC004,__READ_WRITE  ,__hibrtcm0_bits);
__IO_REG32_BIT(HIBRTCM1,          0x400FC008,__READ_WRITE  ,__hibrtcm1_bits);
__IO_REG32_BIT(HIBRTCLD,          0x400FC00C,__READ_WRITE  ,__hibrtcld_bits);
__IO_REG32_BIT(HIBCTL,            0x400FC010,__READ_WRITE  ,__hibctl_bits);
__IO_REG32_BIT(HIBIM,             0x400FC014,__READ_WRITE  ,__hibim_bits);
__IO_REG32_BIT(HIBRIS,            0x400FC018,__READ        ,__hibim_bits);
__IO_REG32_BIT(HIBMIS,            0x400FC01C,__READ        ,__hibim_bits);
__IO_REG32_BIT(HIBIC,             0x400FC020,__READ_WRITE  ,__hibim_bits);
__IO_REG32_BIT(HIBRTCT,           0x400FC024,__READ_WRITE  ,__hibrtct_bits);
__IO_REG32_BIT(HIBDATA0,          0x400FC030,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA1,          0x400FC034,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA2,          0x400FC038,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA3,          0x400FC03C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA4,          0x400FC040,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA5,          0x400FC044,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA6,          0x400FC048,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA7,          0x400FC04C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA8,          0x400FC050,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA9,          0x400FC054,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA10,         0x400FC058,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA11,         0x400FC05C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA12,         0x400FC060,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA13,         0x400FC064,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA14,         0x400FC068,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA15,         0x400FC06C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA16,         0x400FC070,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA17,         0x400FC074,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA18,         0x400FC078,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA19,         0x400FC07C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA20,         0x400FC080,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA21,         0x400FC084,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA22,         0x400FC088,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA23,         0x400FC08C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA24,         0x400FC090,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA25,         0x400FC094,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA26,         0x400FC098,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA27,         0x400FC09C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA28,         0x400FC0A0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA29,         0x400FC0A4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA30,         0x400FC0A8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA31,         0x400FC0AC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA32,         0x400FC0B0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA33,         0x400FC0B4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA34,         0x400FC0B8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA35,         0x400FC0BC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA36,         0x400FC0C0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA37,         0x400FC0C4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA38,         0x400FC0C8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA39,         0x400FC0CC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA40,         0x400FC0D0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA41,         0x400FC0D4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA42,         0x400FC0D8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA43,         0x400FC0DC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA44,         0x400FC0E0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA45,         0x400FC0E4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA46,         0x400FC0E8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA47,         0x400FC0EC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA48,         0x400FC0F0,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA49,         0x400FC0F4,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA50,         0x400FC0F8,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA51,         0x400FC0FC,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA52,         0x400FC100,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA53,         0x400FC104,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA54,         0x400FC108,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA55,         0x400FC10C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA56,         0x400FC110,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA57,         0x400FC114,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA58,         0x400FC118,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA59,         0x400FC11C,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA60,         0x400FC120,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA61,         0x400FC124,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA62,         0x400FC128,__READ_WRITE  ,__hibdata_bits);
__IO_REG32_BIT(HIBDATA63,         0x400FC12C,__READ_WRITE  ,__hibdata_bits);

/***************************************************************************
 **
 ** Flash
 **
 ***************************************************************************/
__IO_REG32_BIT(RMCTL,             0x400FE0F0,__READ_WRITE ,__rmctl_bits);
__IO_REG32_BIT(RMVER,             0x400FE0F4,__READ_WRITE ,__rmver_bits);
__IO_REG32_BIT(USECRL,            0x400FE140,__READ_WRITE ,__usecrl_bits );
__IO_REG32_BIT(USER_DBG,          0x400FE1D0,__READ_WRITE ,__user_dbg_bits);
#define BOOTCFG       USER_DBG
#define BOOTCFG_bit   USER_DBG_bit.__bootcfg
__IO_REG32_BIT(USER_REG0,         0x400FE1E0,__READ_WRITE ,__user_reg_bits);
__IO_REG32_BIT(USER_REG1,         0x400FE1E4,__READ_WRITE ,__user_reg_bits);
__IO_REG32_BIT(USER_REG2,         0x400FE1E8,__READ_WRITE ,__user_reg_bits);
__IO_REG32_BIT(USER_REG3,         0x400FE1EC,__READ_WRITE ,__user_reg_bits);
__IO_REG32_BIT(FMPRE0,            0x400FE200,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE1,            0x400FE204,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE2,            0x400FE208,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE3,            0x400FE20C,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE4,            0x400FE210,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE5,            0x400FE214,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE6,            0x400FE218,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPRE7,            0x400FE21C,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE0,            0x400FE400,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE1,            0x400FE404,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE2,            0x400FE408,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE3,            0x400FE40C,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE4,            0x400FE410,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE5,            0x400FE414,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE6,            0x400FE41C,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMPPE7,            0x400FE418,__READ_WRITE ,__fmpre_bits);
__IO_REG32_BIT(FMA,               0x400FD000,__READ_WRITE ,__fma_bits);
__IO_REG32(    FMD,               0x400FD004,__READ_WRITE);
__IO_REG32_BIT(FMC,               0x400FD008,__READ_WRITE ,__fmc_bits);
__IO_REG32_BIT(FCRIS,             0x400FD00C,__READ       ,__fcris_bits);
__IO_REG32_BIT(FCIM,              0x400FD010,__READ_WRITE ,__fcim_bits);
__IO_REG32_BIT(FCMISC,            0x400FD014,__READ_WRITE ,__fcmisc_bits);
__IO_REG32_BIT(FMC2,              0x400FD020,__READ_WRITE ,__fmc2_bits);
__IO_REG32(    FWBVAL,            0x400FD030,__READ_WRITE );
__IO_REG32_BIT(FCTL,              0x400FD0F8,__READ_WRITE ,__fctl_bits);
__IO_REG32(    FWB0,              0x400FD100,__READ_WRITE );
__IO_REG32(    FWB1,              0x400FD104,__READ_WRITE );
__IO_REG32(    FWB2,              0x400FD108,__READ_WRITE );
__IO_REG32(    FWB3,              0x400FD10C,__READ_WRITE );
__IO_REG32(    FWB4,              0x400FD110,__READ_WRITE );
__IO_REG32(    FWB5,              0x400FD114,__READ_WRITE );
__IO_REG32(    FWB6,              0x400FD118,__READ_WRITE );
__IO_REG32(    FWB7,              0x400FD11C,__READ_WRITE );
__IO_REG32(    FWB8,              0x400FD120,__READ_WRITE );
__IO_REG32(    FWB9,              0x400FD124,__READ_WRITE );
__IO_REG32(    FWB10,             0x400FD128,__READ_WRITE );
__IO_REG32(    FWB11,             0x400FD12C,__READ_WRITE );
__IO_REG32(    FWB12,             0x400FD130,__READ_WRITE );
__IO_REG32(    FWB13,             0x400FD134,__READ_WRITE );
__IO_REG32(    FWB14,             0x400FD138,__READ_WRITE );
__IO_REG32(    FWB15,             0x400FD13C,__READ_WRITE );
__IO_REG32(    FWB16,             0x400FD140,__READ_WRITE );
__IO_REG32(    FWB17,             0x400FD144,__READ_WRITE );
__IO_REG32(    FWB18,             0x400FD148,__READ_WRITE );
__IO_REG32(    FWB19,             0x400FD14C,__READ_WRITE );
__IO_REG32(    FWB20,             0x400FD150,__READ_WRITE );
__IO_REG32(    FWB21,             0x400FD154,__READ_WRITE );
__IO_REG32(    FWB22,             0x400FD158,__READ_WRITE );
__IO_REG32(    FWB23,             0x400FD15C,__READ_WRITE );
__IO_REG32(    FWB24,             0x400FD160,__READ_WRITE );
__IO_REG32(    FWB25,             0x400FD164,__READ_WRITE );
__IO_REG32(    FWB26,             0x400FD168,__READ_WRITE );
__IO_REG32(    FWB27,             0x400FD16C,__READ_WRITE );
__IO_REG32(    FWB28,             0x400FD170,__READ_WRITE );
__IO_REG32(    FWB29,             0x400FD174,__READ_WRITE );
__IO_REG32(    FWB30,             0x400FD178,__READ_WRITE );
__IO_REG32(    FWB31,             0x400FD17C,__READ_WRITE );

/***************************************************************************
 **
 ** uDMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DMASTAT,           0x400FF000,__READ       ,__dmastat_bits);
__IO_REG32_BIT(DMACFG,            0x400FF004,__WRITE      ,__dmacfg_bits);
__IO_REG32(    DMACTLBASE,        0x400FF008,__READ_WRITE );
__IO_REG32(    DMAALTBASE,        0x400FF00C,__READ       );
__IO_REG32_BIT(DMAWAITSTAT,       0x400FF010,__READ       ,__dmawaitstat_bits);
__IO_REG32_BIT(DMASWREQ,          0x400FF014,__WRITE      ,__dmaswreq_bits);
__IO_REG32_BIT(DMAUSEBURSTSET,    0x400FF018,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32_BIT(DMAUSEBURSTCLR,    0x400FF01C,__WRITE      ,__dmauseburstclr_bits);
__IO_REG32_BIT(DMAREQMASKSET,     0x400FF020,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32_BIT(DMAREQMASKCLR,     0x400FF024,__WRITE      ,__dmauseburstclr_bits);
__IO_REG32_BIT(DMAENASET,         0x400FF028,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32_BIT(DMAENACLR,         0x400FF02C,__WRITE      ,__dmauseburstclr_bits);
__IO_REG32_BIT(DMAALTSET,         0x400FF030,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32_BIT(DMAALTCLR,         0x400FF034,__WRITE      ,__dmauseburstclr_bits);
__IO_REG32_BIT(DMAPRIOSET,        0x400FF038,__READ_WRITE ,__dmauseburstset_bits);
__IO_REG32_BIT(DMAPRIOCLR,        0x400FF03C,__WRITE      ,__dmauseburstclr_bits);
__IO_REG32_BIT(DMAERRCLR,         0x400FF04C,__READ_WRITE ,__dmaerrclr_bits);
__IO_REG32_BIT(DMACHALT,          0x400FF500,__READ_WRITE ,__dmachalt_bits);
#define DMACHASGN       DMACHALT
#define DMACHASGN_bit   DMACHALT_bit
__IO_REG32_BIT(DMACHIS,           0x400FF504,__READ_WRITE ,__dmachis_bits);
__IO_REG8(     DMAPeriphID4,      0x400FFFD0,__READ       );
__IO_REG8(     DMAPeriphID0,      0x400FFFE0,__READ       );
__IO_REG8(     DMAPeriphID1,      0x400FFFE4,__READ       );
__IO_REG8(     DMAPeriphID2,      0x400FFFE8,__READ       );
__IO_REG8(     DMAPeriphID3,      0x400FFFEC,__READ       );
__IO_REG8(     DMAPCellID0,       0x400FFFF0,__READ       );
__IO_REG8(     DMAPCellID1,       0x400FFFF4,__READ       );
__IO_REG8(     DMAPCellID2,       0x400FFFF8,__READ       );
__IO_REG8(     DMAPCellID3,       0x400FFFFC,__READ       );

/***************************************************************************
 **
 ** GPIOA APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOADATAMASK,     0x40004000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADATA,         0x400043FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADIR,          0x40004400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAIS,           0x40004404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAIBE,          0x40004408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAIEV,          0x4000440C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAIM,           0x40004410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOARIS,          0x40004414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOAMIS,          0x40004418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOAICR,          0x4000441C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOAAFSEL,        0x40004420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADR2R,         0x40004500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADR4R,         0x40004504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADR8R,         0x40004508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAODR,          0x4000450C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAPUR,          0x40004510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAPDR,          0x40004514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOASLR,          0x40004518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOADEN,          0x4000451C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOALOCK,         0x40004520,__READ_WRITE );
__IO_REG32_BIT(GPIOACR,           0x40004524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAAMSEL,        0x40004528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOAPCTL,         0x4000452C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOAPERIPHID4,    0x40004FD0,__READ);
__IO_REG8(     GPIOAPERIPHID5,    0x40004FD4,__READ);
__IO_REG8(     GPIOAPERIPHID6,    0x40004FD8,__READ);
__IO_REG8(     GPIOAPERIPHID7,    0x40004FDC,__READ);
__IO_REG8(     GPIOAPERIPHID0,    0x40004FE0,__READ);
__IO_REG8(     GPIOAPERIPHID1,    0x40004FE4,__READ);
__IO_REG8(     GPIOAPERIPHID2,    0x40004FE8,__READ);
__IO_REG8(     GPIOAPERIPHID3,    0x40004FEC,__READ);
__IO_REG8(     GPIOAPCELLID0,     0x40004FF0,__READ);
__IO_REG8(     GPIOAPCELLID1,     0x40004FF4,__READ);
__IO_REG8(     GPIOAPCELLID2,     0x40004FF8,__READ);
__IO_REG8(     GPIOAPCELLID3,     0x40004FFC,__READ);

/***************************************************************************
 **
 ** GPIOA AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOA_AHB_DATAMASK, 0x40058000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DATA,     0x400583FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DIR,      0x40058400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_IS,       0x40058404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_IBE,      0x40058408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_IEV,      0x4005840C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_IM,       0x40058410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_RIS,      0x40058414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_MIS,      0x40058418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_ICR,      0x4005841C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_AFSEL,    0x40058420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DR2R,     0x40058500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DR4R,     0x40058504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DR8R,     0x40058508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_ODR,      0x4005850C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_PUR,      0x40058510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_PDR,      0x40058514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_SLR,      0x40058518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_DEN,      0x4005851C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOA_AHB_LOCK,     0x40058520,__READ_WRITE );
__IO_REG32_BIT(GPIOA_AHB_CR,       0x40058524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_AMSEL,    0x40058528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOA_AHB_PCTL,     0x4005852C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOA_AHB_PERIPHID4,0x40058FD0,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID5,0x40058FD4,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID6,0x40058FD8,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID7,0x40058FDC,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID0,0x40058FE0,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID1,0x40058FE4,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID2,0x40058FE8,__READ);
__IO_REG8(     GPIOA_AHB_PERIPHID3,0x40058FEC,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID0, 0x40058FF0,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID1, 0x40058FF4,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID2, 0x40058FF8,__READ);
__IO_REG8(     GPIOA_AHB_PCELLID3, 0x40058FFC,__READ);

/***************************************************************************
 **
 ** GPIOB APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOBDATAMASK,     0x40005000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDATA,         0x400053FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDIR,          0x40005400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBIS,           0x40005404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBIBE,          0x40005408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBIEV,          0x4000540C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBIM,           0x40005410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBRIS,          0x40005414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOBMIS,          0x40005418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOBICR,          0x4000541C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOBAFSEL,        0x40005420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDR2R,         0x40005500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDR4R,         0x40005504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDR8R,         0x40005508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBODR,          0x4000550C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBPUR,          0x40005510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBPDR,          0x40005514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBSLR,          0x40005518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBDEN,          0x4000551C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOBLOCK,         0x40005520,__READ_WRITE );
__IO_REG32_BIT(GPIOBCR,           0x40005524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBAMSEL,        0x40005528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOBPCTL,         0x4000552C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOBPERIPHID4,    0x40005FD0,__READ);
__IO_REG8(     GPIOBPERIPHID5,    0x40005FD4,__READ);
__IO_REG8(     GPIOBPERIPHID6,    0x40005FD8,__READ);
__IO_REG8(     GPIOBPERIPHID7,    0x40005FDC,__READ);
__IO_REG8(     GPIOBPERIPHID0,    0x40005FE0,__READ);
__IO_REG8(     GPIOBPERIPHID1,    0x40005FE4,__READ);
__IO_REG8(     GPIOBPERIPHID2,    0x40005FE8,__READ);
__IO_REG8(     GPIOBPERIPHID3,    0x40005FEC,__READ);
__IO_REG8(     GPIOBPCELLID0,     0x40005FF0,__READ);
__IO_REG8(     GPIOBPCELLID1,     0x40005FF4,__READ);
__IO_REG8(     GPIOBPCELLID2,     0x40005FF8,__READ);
__IO_REG8(     GPIOBPCELLID3,     0x40005FFC,__READ);

/***************************************************************************
 **
 ** GPIOB AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOB_AHB_DATAMASK, 0x40059000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DATA,     0x400593FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DIR,      0x40059400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_IS,       0x40059404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_IBE,      0x40059408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_IEV,      0x4005940C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_IM,       0x40059410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_RIS,      0x40059414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_MIS,      0x40059418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_ICR,      0x4005941C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_AFSEL,    0x40059420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DR2R,     0x40059500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DR4R,     0x40059504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DR8R,     0x40059508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_ODR,      0x4005950C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_PUR,      0x40059510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_PDR,      0x40059514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_SLR,      0x40059518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_DEN,      0x4005951C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOB_AHB_LOCK,     0x40059520,__READ_WRITE );
__IO_REG32_BIT(GPIOB_AHB_CR,       0x40059524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_AMSEL,    0x40059528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOB_AHB_PCTL,     0x4005952C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOB_AHB_PERIPHID4,0x40059FD0,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID5,0x40059FD4,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID6,0x40059FD8,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID7,0x40059FDC,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID0,0x40059FE0,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID1,0x40059FE4,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID2,0x40059FE8,__READ);
__IO_REG8(     GPIOB_AHB_PERIPHID3,0x40059FEC,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID0, 0x40059FF0,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID1, 0x40059FF4,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID2, 0x40059FF8,__READ);
__IO_REG8(     GPIOB_AHB_PCELLID3, 0x40059FFC,__READ);

/***************************************************************************
 **
 ** GPIOC APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOCDATAMASK,     0x40006000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDATA,         0x400063FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDIR,          0x40006400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCIS,           0x40006404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCIBE,          0x40006408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCIEV,          0x4000640C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCIM,           0x40006410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCRIS,          0x40006414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOCMIS,          0x40006418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOCICR,          0x4000641C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOCAFSEL,        0x40006420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDR2R,         0x40006500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDR4R,         0x40006504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDR8R,         0x40006508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCODR,          0x4000650C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCPUR,          0x40006510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCPDR,          0x40006514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCSLR,          0x40006518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCDEN,          0x4000651C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOCLOCK,         0x40006520,__READ_WRITE );
__IO_REG32_BIT(GPIOCCR,           0x40006524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCAMSEL,        0x40006528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOCPCTL,         0x4000652C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOCPERIPHID4,    0x40006FD0,__READ);
__IO_REG8(     GPIOCPERIPHID5,    0x40006FD4,__READ);
__IO_REG8(     GPIOCPERIPHID6,    0x40006FD8,__READ);
__IO_REG8(     GPIOCPERIPHID7,    0x40006FDC,__READ);
__IO_REG8(     GPIOCPERIPHID0,    0x40006FE0,__READ);
__IO_REG8(     GPIOCPERIPHID1,    0x40006FE4,__READ);
__IO_REG8(     GPIOCPERIPHID2,    0x40006FE8,__READ);
__IO_REG8(     GPIOCPERIPHID3,    0x40006FEC,__READ);
__IO_REG8(     GPIOCPCELLID0,     0x40006FF0,__READ);
__IO_REG8(     GPIOCPCELLID1,     0x40006FF4,__READ);
__IO_REG8(     GPIOCPCELLID2,     0x40006FF8,__READ);
__IO_REG8(     GPIOCPCELLID3,     0x40006FFC,__READ);

/***************************************************************************
 **
 ** GPIOC AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOC_AHB_DATAMASK, 0x4005A000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DATA,     0x4005A3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DIR,      0x4005A400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_IS,       0x4005A404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_IBE,      0x4005A408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_IEV,      0x4005A40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_IM,       0x4005A410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_RIS,      0x4005A414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_MIS,      0x4005A418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_ICR,      0x4005A41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_AFSEL,    0x4005A420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DR2R,     0x4005A500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DR4R,     0x4005A504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DR8R,     0x4005A508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_ODR,      0x4005A50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_PUR,      0x4005A510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_PDR,      0x4005A514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_SLR,      0x4005A518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_DEN,      0x4005A51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOC_AHB_LOCK,     0x4005A520,__READ_WRITE );
__IO_REG32_BIT(GPIOC_AHB_CR,       0x4005A524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_AMSEL,    0x4005A528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOC_AHB_PCTL,     0x4005A52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOC_AHB_PERIPHID4,0x4005AFD0,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID5,0x4005AFD4,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID6,0x4005AFD8,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID7,0x4005AFDC,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID0,0x4005AFE0,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID1,0x4005AFE4,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID2,0x4005AFE8,__READ);
__IO_REG8(     GPIOC_AHB_PERIPHID3,0x4005AFEC,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID0, 0x4005AFF0,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID1, 0x4005AFF4,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID2, 0x4005AFF8,__READ);
__IO_REG8(     GPIOC_AHB_PCELLID3, 0x4005AFFC,__READ);

/***************************************************************************
 **
 ** GPIOD APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIODDATAMASK,     0x40007000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDATA,         0x400073FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDIR,          0x40007400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODIS,           0x40007404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODIBE,          0x40007408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODIEV,          0x4000740C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODIM,           0x40007410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODRIS,          0x40007414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIODMIS,          0x40007418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIODICR,          0x4000741C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIODAFSEL,        0x40007420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDR2R,         0x40007500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDR4R,         0x40007504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDR8R,         0x40007508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODODR,          0x4000750C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODPUR,          0x40007510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODPDR,          0x40007514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODSLR,          0x40007518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODDEN,          0x4000751C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIODLOCK,         0x40007520,__READ_WRITE );
__IO_REG32_BIT(GPIODCR,           0x40007524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODAMSEL,        0x40007528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIODPCTL,         0x4000752C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIODPERIPHID4,    0x40007FD0,__READ);
__IO_REG8(     GPIODPERIPHID5,    0x40007FD4,__READ);
__IO_REG8(     GPIODPERIPHID6,    0x40007FD8,__READ);
__IO_REG8(     GPIODPERIPHID7,    0x40007FDC,__READ);
__IO_REG8(     GPIODPERIPHID0,    0x40007FE0,__READ);
__IO_REG8(     GPIODPERIPHID1,    0x40007FE4,__READ);
__IO_REG8(     GPIODPERIPHID2,    0x40007FE8,__READ);
__IO_REG8(     GPIODPERIPHID3,    0x40007FEC,__READ);
__IO_REG8(     GPIODPCELLID0,     0x40007FF0,__READ);
__IO_REG8(     GPIODPCELLID1,     0x40007FF4,__READ);
__IO_REG8(     GPIODPCELLID2,     0x40007FF8,__READ);
__IO_REG8(     GPIODPCELLID3,     0x40007FFC,__READ);

/***************************************************************************
 **
 ** GPIOD AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOD_AHB_DATAMASK, 0x4005B000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DATA,     0x4005B3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DIR,      0x4005B400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_IS,       0x4005B404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_IBE,      0x4005B408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_IEV,      0x4005B40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_IM,       0x4005B410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_RIS,      0x4005B414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_MIS,      0x4005B418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_ICR,      0x4005B41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_AFSEL,    0x4005B420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DR2R,     0x4005B500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DR4R,     0x4005B504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DR8R,     0x4005B508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_ODR,      0x4005B50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_PUR,      0x4005B510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_PDR,      0x4005B514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_SLR,      0x4005B518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_DEN,      0x4005B51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOD_AHB_LOCK,     0x4005B520,__READ_WRITE );
__IO_REG32_BIT(GPIOD_AHB_CR,       0x4005B524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_AMSEL,    0x4005B528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOD_AHB_PCTL,     0x4005B52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOD_AHB_PERIPHID4,0x4005BFD0,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID5,0x4005BFD4,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID6,0x4005BFD8,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID7,0x4005BFDC,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID0,0x4005BFE0,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID1,0x4005BFE4,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID2,0x4005BFE8,__READ);
__IO_REG8(     GPIOD_AHB_PERIPHID3,0x4005BFEC,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID0, 0x4005BFF0,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID1, 0x4005BFF4,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID2, 0x4005BFF8,__READ);
__IO_REG8(     GPIOD_AHB_PCELLID3, 0x4005BFFC,__READ);

/***************************************************************************
 **
 ** GPIOE APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOEDATAMASK,     0x40024000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDATA,         0x400243FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDIR,          0x40024400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEIS,           0x40024404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEIBE,          0x40024408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEIEV,          0x4002440C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEIM,           0x40024410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOERIS,          0x40024414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOEMIS,          0x40024418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOEICR,          0x4002441C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOEAFSEL,        0x40024420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDR2R,         0x40024500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDR4R,         0x40024504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDR8R,         0x40024508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEODR,          0x4002450C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEPUR,          0x40024510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEPDR,          0x40024514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOESLR,          0x40024518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEDEN,          0x4002451C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOELOCK,         0x40024520,__READ_WRITE );
__IO_REG32_BIT(GPIOECR,           0x40024524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEAMSEL,        0x40024528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOEPCTL,         0x4002452C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOEPERIPHID4,    0x40024FD0,__READ);
__IO_REG8(     GPIOEPERIPHID5,    0x40024FD4,__READ);
__IO_REG8(     GPIOEPERIPHID6,    0x40024FD8,__READ);
__IO_REG8(     GPIOEPERIPHID7,    0x40024FDC,__READ);
__IO_REG8(     GPIOEPERIPHID0,    0x40024FE0,__READ);
__IO_REG8(     GPIOEPERIPHID1,    0x40024FE4,__READ);
__IO_REG8(     GPIOEPERIPHID2,    0x40024FE8,__READ);
__IO_REG8(     GPIOEPERIPHID3,    0x40024FEC,__READ);
__IO_REG8(     GPIOEPCELLID0,     0x40024FF0,__READ);
__IO_REG8(     GPIOEPCELLID1,     0x40024FF4,__READ);
__IO_REG8(     GPIOEPCELLID2,     0x40024FF8,__READ);
__IO_REG8(     GPIOEPCELLID3,     0x40024FFC,__READ);

/***************************************************************************
 **
 ** GPIOE AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOE_AHB_DATAMASK, 0x4005C000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DATA,     0x4005C3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DIR,      0x4005C400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_IS,       0x4005C404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_IBE,      0x4005C408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_IEV,      0x4005C40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_IM,       0x4005C410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_RIS,      0x4005C414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_MIS,      0x4005C418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_ICR,      0x4005C41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_AFSEL,    0x4005C420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DR2R,     0x4005C500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DR4R,     0x4005C504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DR8R,     0x4005C508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_ODR,      0x4005C50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_PUR,      0x4005C510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_PDR,      0x4005C514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_SLR,      0x4005C518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_DEN,      0x4005C51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOE_AHB_LOCK,     0x4005C520,__READ_WRITE );
__IO_REG32_BIT(GPIOE_AHB_CR,       0x4005C524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_AMSEL,    0x4005C528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOE_AHB_PCTL,     0x4005C52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOE_AHB_PERIPHID4,0x4005CFD0,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID5,0x4005CFD4,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID6,0x4005CFD8,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID7,0x4005CFDC,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID0,0x4005CFE0,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID1,0x4005CFE4,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID2,0x4005CFE8,__READ);
__IO_REG8(     GPIOE_AHB_PERIPHID3,0x4005CFEC,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID0, 0x4005CFF0,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID1, 0x4005CFF4,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID2, 0x4005CFF8,__READ);
__IO_REG8(     GPIOE_AHB_PCELLID3, 0x4005CFFC,__READ);

/***************************************************************************
 **
 ** GPIOF APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOFDATAMASK,     0x40025000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDATA,         0x400253FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDIR,          0x40025400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFIS,           0x40025404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFIBE,          0x40025408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFIEV,          0x4002540C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFIM,           0x40025410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFRIS,          0x40025414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOFMIS,          0x40025418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOFICR,          0x4002541C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOFAFSEL,        0x40025420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDR2R,         0x40025500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDR4R,         0x40025504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDR8R,         0x40025508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFODR,          0x4002550C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFPUR,          0x40025510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFPDR,          0x40025514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFSLR,          0x40025518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFDEN,          0x4002551C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOFLOCK,         0x40025520,__READ_WRITE );
__IO_REG32_BIT(GPIOFCR,           0x40025524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFAMSEL,        0x40025528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOFPCTL,         0x4002552C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOFPERIPHID4,    0x40025FD0,__READ);
__IO_REG8(     GPIOFPERIPHID5,    0x40025FD4,__READ);
__IO_REG8(     GPIOFPERIPHID6,    0x40025FD8,__READ);
__IO_REG8(     GPIOFPERIPHID7,    0x40025FDC,__READ);
__IO_REG8(     GPIOFPERIPHID0,    0x40025FE0,__READ);
__IO_REG8(     GPIOFPERIPHID1,    0x40025FE4,__READ);
__IO_REG8(     GPIOFPERIPHID2,    0x40025FE8,__READ);
__IO_REG8(     GPIOFPERIPHID3,    0x40025FEC,__READ);
__IO_REG8(     GPIOFPCELLID0,     0x40025FF0,__READ);
__IO_REG8(     GPIOFPCELLID1,     0x40025FF4,__READ);
__IO_REG8(     GPIOFPCELLID2,     0x40025FF8,__READ);
__IO_REG8(     GPIOFPCELLID3,     0x40025FFC,__READ);

/***************************************************************************
 **
 ** GPIOF AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOF_AHB_DATAMASK, 0x4005D000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DATA,     0x4005D3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DIR,      0x4005D400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_IS,       0x4005D404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_IBE,      0x4005D408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_IEV,      0x4005D40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_IM,       0x4005D410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_RIS,      0x4005D414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_MIS,      0x4005D418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_ICR,      0x4005D41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_AFSEL,    0x4005D420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DR2R,     0x4005D500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DR4R,     0x4005D504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DR8R,     0x4005D508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_ODR,      0x4005D50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_PUR,      0x4005D510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_PDR,      0x4005D514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_SLR,      0x4005D518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_DEN,      0x4005D51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOF_AHB_LOCK,     0x4005D520,__READ_WRITE );
__IO_REG32_BIT(GPIOF_AHB_CR,       0x4005D524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_AMSEL,    0x4005D528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOF_AHB_PCTL,     0x4005D52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOF_AHB_PERIPHID4,0x4005DFD0,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID5,0x4005DFD4,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID6,0x4005DFD8,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID7,0x4005DFDC,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID0,0x4005DFE0,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID1,0x4005DFE4,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID2,0x4005DFE8,__READ);
__IO_REG8(     GPIOF_AHB_PERIPHID3,0x4005DFEC,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID0, 0x4005DFF0,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID1, 0x4005DFF4,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID2, 0x4005DFF8,__READ);
__IO_REG8(     GPIOF_AHB_PCELLID3, 0x4005DFFC,__READ);

/***************************************************************************
 **
 ** GPIOG APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOGDATAMASK,     0x40026000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDATA,         0x400263FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDIR,          0x40026400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGIS,           0x40026404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGIBE,          0x40026408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGIEV,          0x4002640C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGIM,           0x40026410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGRIS,          0x40026414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOGMIS,          0x40026418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOGICR,          0x4002641C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOGAFSEL,        0x40026420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDR2R,         0x40026500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDR4R,         0x40026504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDR8R,         0x40026508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGODR,          0x4002650C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGPUR,          0x40026510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGPDR,          0x40026514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGSLR,          0x40026518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGDEN,          0x4002651C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOGLOCK,         0x40026520,__READ_WRITE );
__IO_REG32_BIT(GPIOGCR,           0x40026524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGAMSEL,        0x40026528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOGPCTL,         0x4002652C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOGPERIPHID4,    0x40026FD0,__READ);
__IO_REG8(     GPIOGPERIPHID5,    0x40026FD4,__READ);
__IO_REG8(     GPIOGPERIPHID6,    0x40026FD8,__READ);
__IO_REG8(     GPIOGPERIPHID7,    0x40026FDC,__READ);
__IO_REG8(     GPIOGPERIPHID0,    0x40026FE0,__READ);
__IO_REG8(     GPIOGPERIPHID1,    0x40026FE4,__READ);
__IO_REG8(     GPIOGPERIPHID2,    0x40026FE8,__READ);
__IO_REG8(     GPIOGPERIPHID3,    0x40026FEC,__READ);
__IO_REG8(     GPIOGPCELLID0,     0x40026FF0,__READ);
__IO_REG8(     GPIOGPCELLID1,     0x40026FF4,__READ);
__IO_REG8(     GPIOGPCELLID2,     0x40026FF8,__READ);
__IO_REG8(     GPIOGPCELLID3,     0x40026FFC,__READ);

/***************************************************************************
 **
 ** GPIOG AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOG_AHB_DATAMASK, 0x4005E000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DATA,     0x4005E3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DIR,      0x4005E400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_IS,       0x4005E404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_IBE,      0x4005E408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_IEV,      0x4005E40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_IM,       0x4005E410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_RIS,      0x4005E414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_MIS,      0x4005E418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_ICR,      0x4005E41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_AFSEL,    0x4005E420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DR2R,     0x4005E500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DR4R,     0x4005E504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DR8R,     0x4005E508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_ODR,      0x4005E50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_PUR,      0x4005E510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_PDR,      0x4005E514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_SLR,      0x4005E518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_DEN,      0x4005E51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOG_AHB_LOCK,     0x4005E520,__READ_WRITE );
__IO_REG32_BIT(GPIOG_AHB_CR,       0x4005E524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_AMSEL,    0x4005E528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOG_AHB_PCTL,     0x4005E52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOG_AHB_PERIPHID4,0x4005EFD0,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID5,0x4005EFD4,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID6,0x4005EFD8,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID7,0x4005EFDC,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID0,0x4005EFE0,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID1,0x4005EFE4,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID2,0x4005EFE8,__READ);
__IO_REG8(     GPIOG_AHB_PERIPHID3,0x4005EFEC,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID0, 0x4005EFF0,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID1, 0x4005EFF4,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID2, 0x4005EFF8,__READ);
__IO_REG8(     GPIOG_AHB_PCELLID3, 0x4005EFFC,__READ);

/***************************************************************************
 **
 ** GPIOH APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOHDATAMASK,     0x40027000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDATA,         0x400273FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDIR,          0x40027400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHIS,           0x40027404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHIBE,          0x40027408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHIEV,          0x4002740C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHIM,           0x40027410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHRIS,          0x40027414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOHMIS,          0x40027418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOHICR,          0x4002741C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOHAFSEL,        0x40027420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDR2R,         0x40027500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDR4R,         0x40027504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDR8R,         0x40027508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHODR,          0x4002750C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHPUR,          0x40027510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHPDR,          0x40027514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHSLR,          0x40027518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHDEN,          0x4002751C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOHLOCK,         0x40027520,__READ_WRITE );
__IO_REG32_BIT(GPIOHCR,           0x40027524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHAMSEL,        0x40027528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOHPCTL,         0x4002752C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOHPERIPHID4,    0x40027FD0,__READ);
__IO_REG8(     GPIOHPERIPHID5,    0x40027FD4,__READ);
__IO_REG8(     GPIOHPERIPHID6,    0x40027FD8,__READ);
__IO_REG8(     GPIOHPERIPHID7,    0x40027FDC,__READ);
__IO_REG8(     GPIOHPERIPHID0,    0x40027FE0,__READ);
__IO_REG8(     GPIOHPERIPHID1,    0x40027FE4,__READ);
__IO_REG8(     GPIOHPERIPHID2,    0x40027FE8,__READ);
__IO_REG8(     GPIOHPERIPHID3,    0x40027FEC,__READ);
__IO_REG8(     GPIOHPCELLID0,     0x40027FF0,__READ);
__IO_REG8(     GPIOHPCELLID1,     0x40027FF4,__READ);
__IO_REG8(     GPIOHPCELLID2,     0x40027FF8,__READ);
__IO_REG8(     GPIOHPCELLID3,     0x40027FFC,__READ);

/***************************************************************************
 **
 ** GPIOH AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOH_AHB_DATAMASK, 0x4005F000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DATA,     0x4005F3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DIR,      0x4005F400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_IS,       0x4005F404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_IBE,      0x4005F408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_IEV,      0x4005F40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_IM,       0x4005F410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_RIS,      0x4005F414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_MIS,      0x4005F418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_ICR,      0x4005F41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_AFSEL,    0x4005F420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DR2R,     0x4005F500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DR4R,     0x4005F504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DR8R,     0x4005F508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_ODR,      0x4005F50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_PUR,      0x4005F510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_PDR,      0x4005F514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_SLR,      0x4005F518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_DEN,      0x4005F51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOH_AHB_LOCK,     0x4005F520,__READ_WRITE );
__IO_REG32_BIT(GPIOH_AHB_CR,       0x4005F524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_AMSEL,    0x4005F528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOH_AHB_PCTL,     0x4005F52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOH_AHB_PERIPHID4,0x4005FFD0,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID5,0x4005FFD4,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID6,0x4005FFD8,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID7,0x4005FFDC,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID0,0x4005FFE0,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID1,0x4005FFE4,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID2,0x4005FFE8,__READ);
__IO_REG8(     GPIOH_AHB_PERIPHID3,0x4005FFEC,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID0, 0x4005FFF0,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID1, 0x4005FFF4,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID2, 0x4005FFF8,__READ);
__IO_REG8(     GPIOH_AHB_PCELLID3, 0x4005FFFC,__READ);

/***************************************************************************
 **
 ** GPIOJ APB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOJDATAMASK,     0x4003D000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDATA,         0x4003D3FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDIR,          0x4003D400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJIS,           0x4003D404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJIBE,          0x4003D408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJIEV,          0x4003D40C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJIM,           0x4003D410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJRIS,          0x4003D414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOJMIS,          0x4003D418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOJICR,          0x4003D41C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOJAFSEL,        0x4003D420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDR2R,         0x4003D500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDR4R,         0x4003D504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDR8R,         0x4003D508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJODR,          0x4003D50C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJPUR,          0x4003D510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJPDR,          0x4003D514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJSLR,          0x4003D518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJDEN,          0x4003D51C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOJLOCK,         0x4003D520,__READ_WRITE );
__IO_REG32_BIT(GPIOJCR,           0x4003D524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJAMSEL,        0x4003D528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJPCTL,         0x4003D52C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOJPERIPHID4,    0x4003DFD0,__READ);
__IO_REG8(     GPIOJPERIPHID5,    0x4003DFD4,__READ);
__IO_REG8(     GPIOJPERIPHID6,    0x4003DFD8,__READ);
__IO_REG8(     GPIOJPERIPHID7,    0x4003DFDC,__READ);
__IO_REG8(     GPIOJPERIPHID0,    0x4003DFE0,__READ);
__IO_REG8(     GPIOJPERIPHID1,    0x4003DFE4,__READ);
__IO_REG8(     GPIOJPERIPHID2,    0x4003DFE8,__READ);
__IO_REG8(     GPIOJPERIPHID3,    0x4003DFEC,__READ);
__IO_REG8(     GPIOJPCELLID0,     0x4003DFF0,__READ);
__IO_REG8(     GPIOJPCELLID1,     0x4003DFF4,__READ);
__IO_REG8(     GPIOJPCELLID2,     0x4003DFF8,__READ);
__IO_REG8(     GPIOJPCELLID3,     0x4003DFFC,__READ);

/***************************************************************************
 **
 ** GPIOJ AHB
 **
 ***************************************************************************/
__IO_REG32_BIT(GPIOJ_AHB_DATAMASK, 0x40060000,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DATA,     0x400603FC,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DIR,      0x40060400,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_IS,       0x40060404,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_IBE,      0x40060408,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_IEV,      0x4006040C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_IM,       0x40060410,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_RIS,      0x40060414,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_MIS,      0x40060418,__READ       ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_ICR,      0x4006041C,__WRITE      ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_AFSEL,    0x40060420,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DR2R,     0x40060500,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DR4R,     0x40060504,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DR8R,     0x40060508,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_ODR,      0x4006050C,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_PUR,      0x40060510,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_PDR,      0x40060514,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_SLR,      0x40060518,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_DEN,      0x4006051C,__READ_WRITE ,__gpio_bits);
__IO_REG32(    GPIOJ_AHB_LOCK,     0x40060520,__READ_WRITE );
__IO_REG32_BIT(GPIOJ_AHB_CR,       0x40060524,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_AMSEL,    0x40060528,__READ_WRITE ,__gpio_bits);
__IO_REG32_BIT(GPIOJ_AHB_PCTL,     0x4006052C,__READ_WRITE ,__gpiopctl_bits);
__IO_REG8(     GPIOJ_AHB_PERIPHID4,0x40060FD0,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID5,0x40060FD4,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID6,0x40060FD8,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID7,0x40060FDC,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID0,0x40060FE0,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID1,0x40060FE4,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID2,0x40060FE8,__READ);
__IO_REG8(     GPIOJ_AHB_PERIPHID3,0x40060FEC,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID0, 0x40060FF0,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID1, 0x40060FF4,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID2, 0x40060FF8,__READ);
__IO_REG8(     GPIOJ_AHB_PCELLID3, 0x40060FFC,__READ);

/***************************************************************************
 **
 ** EPI
 **
 ***************************************************************************/
__IO_REG32_BIT(EPICFG,            0x400D0000,__READ_WRITE ,__epicfg_bits);
__IO_REG32_BIT(EPIBAUD,           0x400D0004,__READ_WRITE ,__epibaud_bits);
__IO_REG32_BIT(EPISDRAMCFG,       0x400D0010,__READ_WRITE ,__episdramcfg_bits);
#define EPIHB8CFG       EPISDRAMCFG
#define EPIHB8CFG_bit   EPISDRAMCFG_bit.__epihb8cfg
#define EPIHB16CFG      EPISDRAMCFG
#define EPIHB16CFG_bit  EPISDRAMCFG_bit.__epihb16cfg
#define EPIGPCFG        EPISDRAMCFG
#define EPIGPCFG_bit    EPISDRAMCFG_bit.__epigpcfg
__IO_REG32_BIT(EPIHB8CFG2,        0x400D0014,__READ_WRITE ,__epihb8cfg2_bits);
#define EPIHB16CFG2     EPIHB8CFG2
#define EPIHB16CFG2_bit EPIHB8CFG2_bit
#define EPIGPCFG2       EPIHB8CFG2
#define EPIGPCFG2_bit   EPIHB8CFG2_bit.__epigpcfg2
__IO_REG32_BIT(EPIADDRMAP,        0x400D001C,__READ_WRITE ,__epiaddrmap_bits);
__IO_REG32_BIT(EPIRSIZE0,         0x400D0020,__READ_WRITE ,__epirsize_bits);
__IO_REG32_BIT(EPIRADDR0,         0x400D0024,__READ_WRITE ,__epiraddr_bits);
__IO_REG32_BIT(EPIRPSTD0,         0x400D0028,__READ_WRITE ,__epirpstd_bits);
__IO_REG32_BIT(EPIRSIZE1,         0x400D0030,__READ_WRITE ,__epirsize_bits);
__IO_REG32_BIT(EPIRADDR1,         0x400D0034,__READ_WRITE ,__epiraddr_bits);
__IO_REG32_BIT(EPIRPSTD1,         0x400D0038,__READ_WRITE ,__epirpstd_bits);
__IO_REG32_BIT(EPISTAT,           0x400D0060,__READ       ,__epistat_bits);
__IO_REG32_BIT(EPIRFIFOCNT,       0x400D006C,__READ       ,__epirfifocnt_bits);
__IO_REG32(    EPIREADFIFO,       0x400D0070,__READ       );
__IO_REG32(    EPIREADFIFO1,      0x400D0074,__READ       );
__IO_REG32(    EPIREADFIFO2,      0x400D0078,__READ       );
__IO_REG32(    EPIREADFIFO3,      0x400D007C,__READ       );
__IO_REG32(    EPIREADFIFO4,      0x400D0080,__READ       );
__IO_REG32(    EPIREADFIFO5,      0x400D0084,__READ       );
__IO_REG32(    EPIREADFIFO6,      0x400D0088,__READ       );
__IO_REG32(    EPIREADFIFO7,      0x400D008C,__READ       );
__IO_REG32_BIT(EPIFIFOLVL,        0x400D0200,__READ_WRITE ,__epififolvl_bits);
__IO_REG32_BIT(EPIWFIFOCNT,       0x400D0204,__READ       ,__epiwfifocnt_bits);
__IO_REG32_BIT(EPIIM,             0x400D0210,__READ_WRITE ,__epiim_bits);
__IO_REG32_BIT(EPIRIS,            0x400D0214,__READ       ,__epiris_bits);
__IO_REG32_BIT(EPIMIS,            0x400D0218,__READ       ,__epimis_bits);
__IO_REG32_BIT(EPIEISC,           0x400D021C,__READ_WRITE ,__epieisc_bits);

/***************************************************************************
 **
 ** TIMER0
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM0CFG,          0x40030000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM0TAMR,         0x40030004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM0TBMR,         0x40030008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM0CTL,          0x4003000C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM0IMR,          0x40030018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM0RIS,          0x4003001C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM0MIS,          0x40030020,__READ       ,__gptmmis_bits);
__IO_REG32_BIT(GPTM0ICR,          0x40030024,__WRITE      ,__gptmicr_bits);
__IO_REG32_BIT(GPTM0TAILR,        0x40030028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM0TBILR,        0x4003002C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM0TAMATCHR,     0x40030030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM0TBMATCHR,     0x40030034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM0TAPR,         0x40030038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM0TBPR,         0x4003003C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM0TAPMR,        0x40030040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM0TBPMR,        0x40030044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM0TAR,          0x40030048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM0TBR,          0x4003004C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM0TAV,          0x40030050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM0TBV,          0x40030054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER1
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM1CFG,          0x40031000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM1TAMR,         0x40031004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM1TBMR,         0x40031008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM1CTL,          0x4003100C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM1IMR,          0x40031018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM1RIS,          0x4003101C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM1MIS,          0x40031020,__READ       ,__gptmmis_bits);
__IO_REG32_BIT(GPTM1ICR,          0x40031024,__WRITE      ,__gptmicr_bits);
__IO_REG32_BIT(GPTM1TAILR,        0x40031028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM1TBILR,        0x4003102C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM1TAMATCHR,     0x40031030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM1TBMATCHR,     0x40031034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM1TAPR,         0x40031038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM1TBPR,         0x4003103C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM1TAPMR,        0x40031040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM1TBPMR,        0x40031044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM1TAR,          0x40031048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM1TBR,          0x4003104C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM1TAV,          0x40031050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM1TBV,          0x40031054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER2
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM2CFG,          0x40032000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM2TAMR,         0x40032004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM2TBMR,         0x40032008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM2CTL,          0x4003200C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM2IMR,          0x40032018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM2RIS,          0x4003201C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM2MIS,          0x40032020,__READ       ,__gptmmis_bits);
__IO_REG32_BIT(GPTM2ICR,          0x40032024,__WRITE      ,__gptmicr_bits);
__IO_REG32_BIT(GPTM2TAILR,        0x40032028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM2TBILR,        0x4003202C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM2TAMATCHR,     0x40032030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM2TBMATCHR,     0x40032034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM2TAPR,         0x40032038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM2TBPR,         0x4003203C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM2TAPMR,        0x40032040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM2TBPMR,        0x40032044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM2TAR,          0x40032048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM2TBR,          0x4003204C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM2TAV,          0x40032050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM2TBV,          0x40032054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** TIMER3
 **
 ***************************************************************************/
__IO_REG32_BIT(GPTM3CFG,          0x40033000,__READ_WRITE ,__gptmcfg_bits);
__IO_REG32_BIT(GPTM3TAMR,         0x40033004,__READ_WRITE ,__gptmtamr_bits);
__IO_REG32_BIT(GPTM3TBMR,         0x40033008,__READ_WRITE ,__gptmtbmr_bits);
__IO_REG32_BIT(GPTM3CTL,          0x4003300C,__READ_WRITE ,__gptmctl_bits);
__IO_REG32_BIT(GPTM3IMR,          0x40033018,__READ_WRITE ,__gptmimr_bits);
__IO_REG32_BIT(GPTM3RIS,          0x4003301C,__READ       ,__gptmris_bits);
__IO_REG32_BIT(GPTM3MIS,          0x40033020,__READ       ,__gptmmis_bits);
__IO_REG32_BIT(GPTM3ICR,          0x40033024,__WRITE      ,__gptmicr_bits);
__IO_REG32_BIT(GPTM3TAILR,        0x40033028,__READ_WRITE ,__gptmtailr_bits);
__IO_REG32_BIT(GPTM3TBILR,        0x4003302C,__READ_WRITE ,__gptmtbilr_bits);
__IO_REG32_BIT(GPTM3TAMATCHR,     0x40033030,__READ_WRITE ,__gptmtamatchr_bits);
__IO_REG32_BIT(GPTM3TBMATCHR,     0x40033034,__READ_WRITE ,__gptmtbmatchr_bits);
__IO_REG32_BIT(GPTM3TAPR,         0x40033038,__READ_WRITE ,__gptmtapr_bits);
__IO_REG32_BIT(GPTM3TBPR,         0x4003303C,__READ_WRITE ,__gptmtbpr_bits);
__IO_REG32_BIT(GPTM3TAPMR,        0x40033040,__READ_WRITE ,__gptmtapmr_bits);
__IO_REG32_BIT(GPTM3TBPMR,        0x40033044,__READ_WRITE ,__gptmtbpmr_bits);
__IO_REG32_BIT(GPTM3TAR,          0x40033048,__READ       ,__gptmtar_bits);
__IO_REG32_BIT(GPTM3TBR,          0x4003304C,__READ       ,__gptmtbr_bits);
__IO_REG32_BIT(GPTM3TAV,          0x40033050,__READ       ,__gptmtav_bits);
__IO_REG32_BIT(GPTM3TBV,          0x40033054,__READ       ,__gptmtbv_bits);

/***************************************************************************
 **
 ** WDT0
 **
 ***************************************************************************/
__IO_REG32(    WDT0LOAD,           0x40000000,__READ_WRITE);
__IO_REG32(    WDT0VALUE,          0x40000004,__READ);
__IO_REG32_BIT(WDT0CTL,            0x40000008,__READ_WRITE ,__wdt0ctl_bits);
__IO_REG32(    WDT0ICR,            0x4000000C,__WRITE);
__IO_REG32_BIT(WDT0RIS,            0x40000010,__READ       ,__wdtris_bits);
__IO_REG32_BIT(WDT0MIS,            0x40000014,__READ       ,__wdtmis_bits);
__IO_REG32_BIT(WDT0TEST,           0x40000418,__READ_WRITE ,__wdttest_bits);
__IO_REG32(    WDT0LOCK,           0x40000C00,__READ_WRITE);
__IO_REG8(     WDT0PERIPHID4,      0x40000FD0,__READ);
__IO_REG8(     WDT0PERIPHID5,      0x40000FD4,__READ);
__IO_REG8(     WDT0PERIPHID6,      0x40000FD8,__READ);
__IO_REG8(     WDT0PERIPHID7,      0x40000FDC,__READ);
__IO_REG8(     WDT0PERIPHID0,      0x40000FE0,__READ);
__IO_REG8(     WDT0PERIPHID1,      0x40000FE4,__READ);
__IO_REG8(     WDT0PERIPHID2,      0x40000FE8,__READ);
__IO_REG8(     WDT0PERIPHID3,      0x40000FEC,__READ);
__IO_REG8(     WDT0PCELLID0,       0x40000FF0,__READ);
__IO_REG8(     WDT0PCELLID1,       0x40000FF4,__READ);
__IO_REG8(     WDT0PCELLID2,       0x40000FF8,__READ);
__IO_REG8(     WDT0PCELLID3,       0x40000FFC,__READ);

/***************************************************************************
 **
 ** WDT1
 **
 ***************************************************************************/
__IO_REG32(    WDT1LOAD,           0x40001000,__READ_WRITE);
__IO_REG32(    WDT1VALUE,          0x40001004,__READ);
__IO_REG32_BIT(WDT1CTL,            0x40001008,__READ_WRITE ,__wdt1ctl_bits);
__IO_REG32(    WDT1ICR,            0x4000100C,__WRITE);
__IO_REG32_BIT(WDT1RIS,            0x40001010,__READ       ,__wdtris_bits);
__IO_REG32_BIT(WDT1MIS,            0x40001014,__READ       ,__wdtmis_bits);
__IO_REG32_BIT(WDT1TEST,           0x40001418,__READ_WRITE ,__wdttest_bits);
__IO_REG32(    WDT1LOCK,           0x40001C00,__READ_WRITE);
__IO_REG8(     WDT1PERIPHID4,      0x40001FD0,__READ);
__IO_REG8(     WDT1PERIPHID5,      0x40001FD4,__READ);
__IO_REG8(     WDT1PERIPHID6,      0x40001FD8,__READ);
__IO_REG8(     WDT1PERIPHID7,      0x40001FDC,__READ);
__IO_REG8(     WDT1PERIPHID0,      0x40001FE0,__READ);
__IO_REG8(     WDT1PERIPHID1,      0x40001FE4,__READ);
__IO_REG8(     WDT1PERIPHID2,      0x40001FE8,__READ);
__IO_REG8(     WDT1PERIPHID3,      0x40001FEC,__READ);
__IO_REG8(     WDT1PCELLID0,       0x40001FF0,__READ);
__IO_REG8(     WDT1PCELLID1,       0x40001FF4,__READ);
__IO_REG8(     WDT1PCELLID2,       0x40001FF8,__READ);
__IO_REG8(     WDT1PCELLID3,       0x40001FFC,__READ);

/***************************************************************************
 **
 ** ADC0
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC0ACTSS,         0x40038000,__READ_WRITE ,__adcactss_bits);
__IO_REG32_BIT(ADC0RIS,           0x40038004,__READ       ,__adcris_bits);
__IO_REG32_BIT(ADC0IM,            0x40038008,__READ_WRITE ,__adcim_bits);
__IO_REG32_BIT(ADC0ISC,           0x4003800C,__READ_WRITE ,__adcisc_bits);
__IO_REG32_BIT(ADC0OSTAT,         0x40038010,__READ_WRITE ,__adcostat_bits);
__IO_REG32_BIT(ADC0EMUX,          0x40038014,__READ_WRITE ,__adcemux_bits);
__IO_REG32_BIT(ADC0USTAT,         0x40038018,__READ_WRITE ,__adcustat_bits);
__IO_REG32_BIT(ADC0SSPRI,         0x40038020,__READ_WRITE ,__adcsspri_bits);
__IO_REG32_BIT(ADC0SPC,           0x40038024,__READ_WRITE ,__adcspc_bits);
__IO_REG32_BIT(ADC0PSSI,          0x40038028,__WRITE      ,__adcpssi_bits);
__IO_REG32_BIT(ADC0SAC,           0x40038030,__READ_WRITE ,__adcsac_bits);
__IO_REG32_BIT(ADC0DCISC,         0x40038034,__READ_WRITE ,__adcdcisc_bits);
__IO_REG32_BIT(ADC0CTL,           0x40038038,__READ_WRITE ,__adcctl_bits);
__IO_REG32_BIT(ADC0SSMUX0,        0x40038040,__READ_WRITE ,__adcssmux0_bits);
__IO_REG32_BIT(ADC0SSCTL0,        0x40038044,__READ_WRITE ,__adcssctl0_bits);
__IO_REG32_BIT(ADC0SSFIFO0,       0x40038048,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC0SSFSTAT0,      0x4003804C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC0SSOP0,         0x40038050,__READ_WRITE ,__adcssop_bits);
__IO_REG32_BIT(ADC0SSDC0,         0x40038054,__READ_WRITE ,__adcssdc_bits);
__IO_REG32_BIT(ADC0SSMUX1,        0x40038060,__READ_WRITE ,__adcssmux1_bits);
__IO_REG32_BIT(ADC0SSCTL1,        0x40038064,__READ_WRITE ,__adcssctl1_bits);
__IO_REG32_BIT(ADC0SSFIFO1,       0x40038068,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC0SSFSTAT1,      0x4003806C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC0SSOP1,         0x40038070,__READ_WRITE ,__adcssop1_bits);
__IO_REG32_BIT(ADC0SSDC1,         0x40038074,__READ_WRITE ,__adcssdc1_bits);
__IO_REG32_BIT(ADC0SSMUX2,        0x40038080,__READ_WRITE ,__adcssmux1_bits);
__IO_REG32_BIT(ADC0SSCTL2,        0x40038084,__READ_WRITE ,__adcssctl1_bits);
__IO_REG32_BIT(ADC0SSFIFO2,       0x40038088,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC0SSFSTAT2,      0x4003808C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC0SSOP2,         0x40038090,__READ_WRITE ,__adcssop1_bits);
__IO_REG32_BIT(ADC0SSDC2,         0x40038094,__READ_WRITE ,__adcssdc1_bits);
__IO_REG32_BIT(ADC0SSMUX3,        0x400380A0,__READ_WRITE ,__adcssmux3_bits);
__IO_REG32_BIT(ADC0SSCTL3,        0x400380A4,__READ_WRITE ,__adcssctl3_bits);
__IO_REG32_BIT(ADC0SSFIFO3,       0x400380A8,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC0SSFSTAT3,      0x400380AC,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC0SSOP3,         0x400380B0,__READ_WRITE ,__adcssop3_bits);
__IO_REG32_BIT(ADC0SSDC3,         0x400380B4,__READ_WRITE ,__adcssdc3_bits);
__IO_REG32_BIT(ADC0DCRIC,         0x40038D00,__READ_WRITE ,__adcdcric_bits);
__IO_REG32_BIT(ADC0DCCTL0,        0x40038E00,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL1,        0x40038E04,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL2,        0x40038E08,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL3,        0x40038E0C,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL4,        0x40038E10,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL5,        0x40038E14,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL6,        0x40038E18,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCTL7,        0x40038E1C,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC0DCCMP0,        0x40038E40,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP1,        0x40038E44,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP2,        0x40038E48,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP3,        0x40038E4C,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP4,        0x40038E50,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP5,        0x40038E54,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP6,        0x40038E58,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC0DCCMP7,        0x40038E5C,__READ_WRITE ,__adcdccmp_bits);

/***************************************************************************
 **
 ** ADC1
 **
 ***************************************************************************/
__IO_REG32_BIT(ADC1ACTSS,         0x40039000,__READ_WRITE ,__adcactss_bits);
__IO_REG32_BIT(ADC1RIS,           0x40039004,__READ       ,__adcris_bits);
__IO_REG32_BIT(ADC1IM,            0x40039008,__READ_WRITE ,__adcim_bits);
__IO_REG32_BIT(ADC1ISC,           0x4003900C,__READ_WRITE ,__adcisc_bits);
__IO_REG32_BIT(ADC1OSTAT,         0x40039010,__READ_WRITE ,__adcostat_bits);
__IO_REG32_BIT(ADC1EMUX,          0x40039014,__READ_WRITE ,__adcemux_bits);
__IO_REG32_BIT(ADC1USTAT,         0x40039018,__READ_WRITE ,__adcustat_bits);
__IO_REG32_BIT(ADC1SSPRI,         0x40039020,__READ_WRITE ,__adcsspri_bits);
__IO_REG32_BIT(ADC1SPC,           0x40039024,__READ_WRITE ,__adcspc_bits);
__IO_REG32_BIT(ADC1PSSI,          0x40039028,__WRITE      ,__adcpssi_bits);
__IO_REG32_BIT(ADC1SAC,           0x40039030,__READ_WRITE ,__adcsac_bits);
__IO_REG32_BIT(ADC1DCISC,         0x40039034,__READ_WRITE ,__adcdcisc_bits);
__IO_REG32_BIT(ADC1CTL,           0x40039038,__READ_WRITE ,__adcctl_bits);
__IO_REG32_BIT(ADC1SSMUX0,        0x40039040,__READ_WRITE ,__adcssmux0_bits);
__IO_REG32_BIT(ADC1SSCTL0,        0x40039044,__READ_WRITE ,__adcssctl0_bits);
__IO_REG32_BIT(ADC1SSFIFO0,       0x40039048,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC1SSFSTAT0,      0x4003904C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC1SSOP0,         0x40039050,__READ_WRITE ,__adcssop_bits);
__IO_REG32_BIT(ADC1SSDC0,         0x40039054,__READ_WRITE ,__adcssdc_bits);
__IO_REG32_BIT(ADC1SSMUX1,        0x40039060,__READ_WRITE ,__adcssmux1_bits);
__IO_REG32_BIT(ADC1SSCTL1,        0x40039064,__READ_WRITE ,__adcssctl1_bits);
__IO_REG32_BIT(ADC1SSFIFO1,       0x40039068,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC1SSFSTAT1,      0x4003906C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC1SSOP1,         0x40039070,__READ_WRITE ,__adcssop1_bits);
__IO_REG32_BIT(ADC1SSDC1,         0x40039074,__READ_WRITE ,__adcssdc1_bits);
__IO_REG32_BIT(ADC1SSMUX2,        0x40039080,__READ_WRITE ,__adcssmux1_bits);
__IO_REG32_BIT(ADC1SSCTL2,        0x40039084,__READ_WRITE ,__adcssctl1_bits);
__IO_REG32_BIT(ADC1SSFIFO2,       0x40039088,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC1SSFSTAT2,      0x4003908C,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC1SSOP2,         0x40039090,__READ_WRITE ,__adcssop1_bits);
__IO_REG32_BIT(ADC1SSDC2,         0x40039094,__READ_WRITE ,__adcssdc1_bits);
__IO_REG32_BIT(ADC1SSMUX3,        0x400390A0,__READ_WRITE ,__adcssmux3_bits);
__IO_REG32_BIT(ADC1SSCTL3,        0x400390A4,__READ_WRITE ,__adcssctl3_bits);
__IO_REG32_BIT(ADC1SSFIFO3,       0x400390A8,__READ       ,__adcssfifo0_bits);
__IO_REG32_BIT(ADC1SSFSTAT3,      0x400390AC,__READ       ,__adcssfstat0_bits);
__IO_REG32_BIT(ADC1SSOP3,         0x400390B0,__READ_WRITE ,__adcssop3_bits);
__IO_REG32_BIT(ADC1SSDC3,         0x400390B4,__READ_WRITE ,__adcssdc3_bits);
__IO_REG32_BIT(ADC1DCRIC,         0x40039D00,__READ_WRITE ,__adcdcric_bits);
__IO_REG32_BIT(ADC1DCCTL0,        0x40039E00,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL1,        0x40039E04,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL2,        0x40039E08,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL3,        0x40039E0C,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL4,        0x40039E10,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL5,        0x40039E14,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL6,        0x40039E18,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCTL7,        0x40039E1C,__READ_WRITE ,__adcdcctl_bits);
__IO_REG32_BIT(ADC1DCCMP0,        0x40039E40,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP1,        0x40039E44,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP2,        0x40039E48,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP3,        0x40039E4C,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP4,        0x40039E50,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP5,        0x40039E54,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP6,        0x40039E58,__READ_WRITE ,__adcdccmp_bits);
__IO_REG32_BIT(ADC1DCCMP7,        0x40039E5C,__READ_WRITE ,__adcdccmp_bits);

/***************************************************************************
 **
 ** UART 0
 **
 ***************************************************************************/
__IO_REG32_BIT(UART0DR,            0x4000C000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART0RSR,           0x4000C004,__READ_WRITE ,__uartrsr_bits);
#define UART0ECR         UART0RSR
#define UART0ECR_bit     UART0RSR_bit
__IO_REG32_BIT(UART0FR,            0x4000C018,__READ       ,__uartfr_bits);
__IO_REG8(     UART0ILPR,          0x4000C020,__READ_WRITE );
__IO_REG16(    UART0IBRD,          0x4000C024,__READ_WRITE);
__IO_REG32_BIT(UART0FBRD,          0x4000C028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART0LCRH,          0x4000C02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART0CTL,           0x4000C030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART0IFLS,          0x4000C034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART0IM,            0x4000C038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART0RIS,           0x4000C03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART0MIS,           0x4000C040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UART0ICR,           0x4000C044,__WRITE      ,__uarticr_bits);
__IO_REG32_BIT(UART0DMACTL,        0x4000C048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART0LCTL,          0x4000C090,__READ_WRITE ,__uartlctl_bits);
__IO_REG16(    UART0LSS,           0x4000C094,__READ);
__IO_REG16(    UART0LTIM,          0x4000C098,__READ);
__IO_REG8(     UART0PERIPHID4,     0x4000CFD0,__READ);
__IO_REG8(     UART0PERIPHID5,     0x4000CFD4,__READ);
__IO_REG8(     UART0PERIPHID6,     0x4000CFD8,__READ);
__IO_REG8(     UART0PERIPHID7,     0x4000CFDC,__READ);
__IO_REG8(     UART0PERIPHID0,     0x4000CFE0,__READ);
__IO_REG8(     UART0PERIPHID1,     0x4000CFE4,__READ);
__IO_REG8(     UART0PERIPHID2,     0x4000CFE8,__READ);
__IO_REG8(     UART0PERIPHID3,     0x4000CFEC,__READ);
__IO_REG8(     UART0PCELLID0,      0x4000CFF0,__READ);
__IO_REG8(     UART0PCELLID1,      0x4000CFF4,__READ);
__IO_REG8(     UART0PCELLID2,      0x4000CFF8,__READ);
__IO_REG8(     UART0PCELLID3,      0x4000CFFC,__READ);

/***************************************************************************
 **
 ** UART 1
 **
 ***************************************************************************/
__IO_REG32_BIT(UART1DR,            0x4000D000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART1RSR,           0x4000D004,__READ_WRITE ,__uartrsr_bits);
#define UART1ECR         UART1RSR
#define UART1ECR_bit     UART1RSR_bit
__IO_REG32_BIT(UART1FR,            0x4000D018,__READ       ,__uartfr_bits);
__IO_REG8(     UART1ILPR,          0x4000D020,__READ_WRITE );
__IO_REG16(    UART1IBRD,          0x4000D024,__READ_WRITE);
__IO_REG32_BIT(UART1FBRD,          0x4000D028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART1LCRH,          0x4000D02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART1CTL,           0x4000D030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART1IFLS,          0x4000D034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART1IM,            0x4000D038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART1RIS,           0x4000D03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART1MIS,           0x4000D040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UART1ICR,           0x4000D044,__WRITE       ,__uarticr_bits);
__IO_REG32_BIT(UART1DMACTL,        0x4000D048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART1LCTL,          0x4000D090,__READ_WRITE ,__uartlctl_bits);
__IO_REG16(    UART1LSS,           0x4000D094,__READ);
__IO_REG16(    UART1LTIM,          0x4000D098,__READ);
__IO_REG8(     UART1PERIPHID4,     0x4000DFD0,__READ);
__IO_REG8(     UART1PERIPHID5,     0x4000DFD4,__READ);
__IO_REG8(     UART1PERIPHID6,     0x4000DFD8,__READ);
__IO_REG8(     UART1PERIPHID7,     0x4000DFDC,__READ);
__IO_REG8(     UART1PERIPHID0,     0x4000DFE0,__READ);
__IO_REG8(     UART1PERIPHID1,     0x4000DFE4,__READ);
__IO_REG8(     UART1PERIPHID2,     0x4000DFE8,__READ);
__IO_REG8(     UART1PERIPHID3,     0x4000DFEC,__READ);
__IO_REG8(     UART1PCELLID0,      0x4000DFF0,__READ);
__IO_REG8(     UART1PCELLID1,      0x4000DFF4,__READ);
__IO_REG8(     UART1PCELLID2,      0x4000DFF8,__READ);
__IO_REG8(     UART1PCELLID3,      0x4000DFFC,__READ);

/***************************************************************************
 **
 ** UART 2
 **
 ***************************************************************************/
__IO_REG32_BIT(UART2DR,            0x4000E000,__READ_WRITE ,__uartdr_bits);
__IO_REG32_BIT(UART2RSR,           0x4000E004,__READ_WRITE ,__uartrsr_bits);
#define UART2ECR         UART2RSR
#define UART2ECR_bit     UART2RSR_bit
__IO_REG32_BIT(UART2FR,            0x4000E018,__READ       ,__uartfr_bits);
__IO_REG8(     UART2ILPR,          0x4000E020,__READ_WRITE );
__IO_REG16(    UART2IBRD,          0x4000E024,__READ_WRITE);
__IO_REG32_BIT(UART2FBRD,          0x4000E028,__READ_WRITE ,__uartfbrd_bits);
__IO_REG32_BIT(UART2LCRH,          0x4000E02C,__READ_WRITE ,__uartlcrh_bits);
__IO_REG32_BIT(UART2CTL,           0x4000E030,__READ_WRITE ,__uartctl_bits);
__IO_REG32_BIT(UART2IFLS,          0x4000E034,__READ_WRITE ,__uartifls_bits);
__IO_REG32_BIT(UART2IM,            0x4000E038,__READ_WRITE ,__uartim_bits);
__IO_REG32_BIT(UART2RIS,           0x4000E03C,__READ       ,__uartris_bits);
__IO_REG32_BIT(UART2MIS,           0x4000E040,__READ       ,__uartmis_bits);
__IO_REG32_BIT(UART2ICR,           0x4000E044,__WRITE       ,__uarticr_bits);
__IO_REG32_BIT(UART2DMACTL,        0x4000E048,__READ_WRITE ,__uartdmactl_bits);
__IO_REG32_BIT(UART2LCTL,          0x4000E090,__READ_WRITE ,__uartlctl_bits);
__IO_REG16(    UART2LSS,           0x4000E094,__READ);
__IO_REG16(    UART2LTIM,          0x4000E098,__READ);
__IO_REG8(     UART2PERIPHID4,     0x4000EFD0,__READ);
__IO_REG8(     UART2PERIPHID5,     0x4000EFD4,__READ);
__IO_REG8(     UART2PERIPHID6,     0x4000EFD8,__READ);
__IO_REG8(     UART2PERIPHID7,     0x4000EFDC,__READ);
__IO_REG8(     UART2PERIPHID0,     0x4000EFE0,__READ);
__IO_REG8(     UART2PERIPHID1,     0x4000EFE4,__READ);
__IO_REG8(     UART2PERIPHID2,     0x4000EFE8,__READ);
__IO_REG8(     UART2PERIPHID3,     0x4000EFEC,__READ);
__IO_REG8(     UART2PCELLID0,      0x4000EFF0,__READ);
__IO_REG8(     UART2PCELLID1,      0x4000EFF4,__READ);
__IO_REG8(     UART2PCELLID2,      0x4000EFF8,__READ);
__IO_REG8(     UART2PCELLID3,      0x4000EFFC,__READ);

/***************************************************************************
 **
 ** SSI0
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI0CR0,            0x40008000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI0CR1,            0x40008004,__READ_WRITE ,__ssicr1_bits);
__IO_REG16(    SSI0DR,             0x40008008,__READ_WRITE);
__IO_REG32_BIT(SSI0SR,             0x4000800C,__READ       ,__ssisr_bits);
__IO_REG8(     SSI0CPSR,           0x40008010,__READ_WRITE);
__IO_REG32_BIT(SSI0IM,             0x40008014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI0RIS,            0x40008018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI0MIS,            0x4000801C,__READ       ,__ssimis_bits);
__IO_REG32_BIT(SSI0ICR,            0x40008020,__WRITE      ,__ssiicr_bits);
__IO_REG32_BIT(SSI0DMACTL,         0x40008024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI0PERIPHID4,      0x40008FD0,__READ);
__IO_REG8(     SSI0PERIPHID5,      0x40008FD4,__READ);
__IO_REG8(     SSI0PERIPHID6,      0x40008FD8,__READ);
__IO_REG8(     SSI0PERIPHID7,      0x40008FDC,__READ);
__IO_REG8(     SSI0PERIPHID0,      0x40008FE0,__READ);
__IO_REG8(     SSI0PERIPHID1,      0x40008FE4,__READ);
__IO_REG8(     SSI0PERIPHID2,      0x40008FE8,__READ);
__IO_REG8(     SSI0PERIPHID3,      0x40008FEC,__READ);
__IO_REG8(     SSI0PCELLID0,       0x40008FF0,__READ);
__IO_REG8(     SSI0PCELLID1,       0x40008FF4,__READ);
__IO_REG8(     SSI0PCELLID2,       0x40008FF8,__READ);
__IO_REG8(     SSI0PCELLID3,       0x40008FFC,__READ);

/***************************************************************************
 **
 ** SSI1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSI1CR0,            0x40009000,__READ_WRITE ,__ssicr0_bits);
__IO_REG32_BIT(SSI1CR1,            0x40009004,__READ_WRITE ,__ssicr1_bits);
__IO_REG16(    SSI1DR,             0x40009008,__READ_WRITE);
__IO_REG32_BIT(SSI1SR,             0x4000900C,__READ       ,__ssisr_bits);
__IO_REG8(     SSI1CPSR,           0x40009010,__READ_WRITE);
__IO_REG32_BIT(SSI1IM,             0x40009014,__READ_WRITE ,__ssiim_bits);
__IO_REG32_BIT(SSI1RIS,            0x40009018,__READ       ,__ssiris_bits);
__IO_REG32_BIT(SSI1MIS,            0x4000901C,__READ       ,__ssimis_bits);
__IO_REG32_BIT(SSI1ICR,            0x40009020,__WRITE      ,__ssiicr_bits);
__IO_REG32_BIT(SSI1DMACTL,         0x40009024,__READ_WRITE ,__ssidmactl_bits);
__IO_REG8(     SSI1PERIPHID4,      0x40009FD0,__READ);
__IO_REG8(     SSI1PERIPHID5,      0x40009FD4,__READ);
__IO_REG8(     SSI1PERIPHID6,      0x40009FD8,__READ);
__IO_REG8(     SSI1PERIPHID7,      0x40009FDC,__READ);
__IO_REG8(     SSI1PERIPHID0,      0x40009FE0,__READ);
__IO_REG8(     SSI1PERIPHID1,      0x40009FE4,__READ);
__IO_REG8(     SSI1PERIPHID2,      0x40009FE8,__READ);
__IO_REG8(     SSI1PERIPHID3,      0x40009FEC,__READ);
__IO_REG8(     SSI1PCELLID0,       0x40009FF0,__READ);
__IO_REG8(     SSI1PCELLID1,       0x40009FF4,__READ);
__IO_REG8(     SSI1PCELLID2,       0x40009FF8,__READ);
__IO_REG8(     SSI1PCELLID3,       0x40009FFC,__READ);

/***************************************************************************
 **
 ** I2C0
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C0MSA,            0x40020000,__READ_WRITE ,__i2cmsa_bits);
__IO_REG32_BIT(I2C0MS,             0x40020004,__READ_WRITE ,__i2cmcs_bits);
#define I2C0MC           I2C0MS
#define I2C0MC_bit       I2C0MS_bit
__IO_REG8(     I2C0MDR,            0x40020008,__READ_WRITE);
__IO_REG8(     I2C0MTPR,           0x4002000C,__READ_WRITE);
__IO_REG32_BIT(I2C0MIMR,           0x40020010,__READ_WRITE ,__i2cmimr_bits);
__IO_REG32_BIT(I2C0MRIS,           0x40020014,__READ       ,__i2cmris_bits);
__IO_REG32_BIT(I2C0MMIS,           0x40020018,__READ       ,__i2cmmis_bits);
__IO_REG32_BIT(I2C0MICR,           0x4002001C,__WRITE      ,__i2cmicr_bits);
__IO_REG32_BIT(I2C0MCR,            0x40020020,__READ_WRITE ,__i2cmcr_bits);
__IO_REG32_BIT(I2C0SOAR,           0x40020800,__READ_WRITE ,__i2csoar_bits);
__IO_REG32_BIT(I2C0SSR,            0x40020804,__READ_WRITE ,__i2cscsr_bits);
#define I2C0SCR          I2C0SSR
#define I2C0SCR_bit      I2C0SSR_bit
__IO_REG8(     I2C0SDR,            0x40020808,__READ_WRITE);
__IO_REG32_BIT(I2C0SIMR,           0x4002080C,__READ_WRITE ,__i2csimr_bits);
__IO_REG32_BIT(I2C0SRIS,           0x40020810,__READ       ,__i2csris_bits);
__IO_REG32_BIT(I2C0SMIS,           0x40020814,__READ       ,__i2csmis_bits);
__IO_REG32_BIT(I2C0SICR,           0x40020818,__WRITE      ,__i2csicr_bits);

/***************************************************************************
 **
 ** I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(I2C1MSA,            0x40021000,__READ_WRITE ,__i2cmsa_bits);
__IO_REG32_BIT(I2C1MS,             0x40021004,__READ_WRITE ,__i2cmcs_bits);
#define I2C1MC           I2C1MS
#define I2C1MC_bit       I2C1MS_bit
__IO_REG8(     I2C1MDR,            0x40021008,__READ_WRITE);
__IO_REG8(     I2C1MTPR,           0x4002100C,__READ_WRITE);
__IO_REG32_BIT(I2C1MIMR,           0x40021010,__READ_WRITE ,__i2cmimr_bits);
__IO_REG32_BIT(I2C1MRIS,           0x40021014,__READ       ,__i2cmris_bits);
__IO_REG32_BIT(I2C1MMIS,           0x40021018,__READ       ,__i2cmmis_bits);
__IO_REG32_BIT(I2C1MICR,           0x4002101C,__WRITE      ,__i2cmicr_bits);
__IO_REG32_BIT(I2C1MCR,            0x40021020,__READ_WRITE ,__i2cmcr_bits);
__IO_REG32_BIT(I2C1SOAR,           0x40021800,__READ_WRITE ,__i2csoar_bits);
__IO_REG32_BIT(I2C1SSR,            0x40021804,__READ_WRITE ,__i2cscsr_bits);
#define I2C1SCR          I2C1SSR
#define I2C1SCR_bit      I2C1SSR_bit
__IO_REG8(     I2C1SDR,            0x40021808,__READ_WRITE);
__IO_REG32_BIT(I2C1SIMR,           0x4002180C,__READ_WRITE ,__i2csimr_bits);
__IO_REG32_BIT(I2C1SRIS,           0x40021810,__READ       ,__i2csris_bits);
__IO_REG32_BIT(I2C1SMIS,           0x40021814,__READ       ,__i2csmis_bits);
__IO_REG32_BIT(I2C1SICR,           0x40021818,__WRITE      ,__i2csicr_bits);

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32(    I2STXFIFO,         0x40054000,__WRITE      );
__IO_REG32_BIT(I2STXFIFOCFG,      0x40054004,__READ_WRITE ,__i2stxfifocfg_bits);
__IO_REG32_BIT(I2STXCFG,          0x40054008,__READ_WRITE ,__i2stxcfg_bits);
__IO_REG32_BIT(I2STXLIMIT,        0x4005400C,__READ_WRITE ,__i2stxlimit_bits);
__IO_REG32_BIT(I2STXISM,          0x40054010,__READ_WRITE ,__i2stxism_bits);
__IO_REG32_BIT(I2STXLEV,          0x40054018,__READ       ,__i2stxlev_bits);
__IO_REG32(    I2SRXFIFO,         0x40054800,__READ       );
__IO_REG32_BIT(I2SRXFIFOCFG,      0x40054804,__READ_WRITE ,__i2srxfifocfg_bits);
__IO_REG32_BIT(I2SRXCFG,          0x40054808,__READ_WRITE ,__i2srxcfg_bits);
__IO_REG32_BIT(I2SRXLIMIT,        0x4005480C,__READ_WRITE ,__i2srxlimit_bits);
__IO_REG32_BIT(I2SRXISM,          0x40054810,__READ_WRITE ,__i2srxism_bits);
__IO_REG32_BIT(I2SRXLEV,          0x40054818,__READ       ,__i2srxlev_bits);
__IO_REG32_BIT(I2SCFG,            0x40054C00,__READ_WRITE ,__i2scfg_bits);
__IO_REG32_BIT(I2SIM,             0x40054C10,__READ_WRITE ,__i2sim_bits);
__IO_REG32_BIT(I2SRIS,            0x40054C14,__READ_WRITE ,__i2sris_bits);
__IO_REG32_BIT(I2SMIS,            0x40054C18,__READ_WRITE ,__i2smis_bits);
__IO_REG32_BIT(I2SIC,             0x40054C1C,__READ_WRITE ,__i2sic_bits);

/***************************************************************************
 **
 ** CAN0
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN0CTL,           0x40040000,__READ_WRITE  ,__canctl_bits);
__IO_REG32_BIT(CAN0STS,           0x40040004,__READ_WRITE  ,__cansts_bits);
__IO_REG32_BIT(CAN0ERR,           0x40040008,__READ        ,__canerr_bits);
__IO_REG32_BIT(CAN0BIT,           0x4004000C,__READ_WRITE  ,__canbit_bits);
__IO_REG32_BIT(CAN0INT,           0x40040010,__READ        ,__canint_bits);
__IO_REG32_BIT(CAN0TST,           0x40040014,__READ_WRITE  ,__cantst_bits);
__IO_REG32_BIT(CAN0BRPE,          0x40040018,__READ_WRITE  ,__canbrpe_bits);
__IO_REG32_BIT(CAN0IF1CRQ,        0x40040020,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN0IF1CMSK,       0x40040024,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN0IF1MSK1,       0x40040028,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN0IF1MSK2,       0x4004002C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN0IF1ARB1,       0x40040030,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN0IF1ARB2,       0x40040034,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN0IF1MCTL,       0x40040038,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN0IF1DA1,        0x4004003C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF1DA2,        0x40040040,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF1DB1,        0x40040044,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF1DB2,        0x40040048,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF2CRQ,        0x40040080,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN0IF2CMSK,       0x40040084,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN0IF2MSK1,       0x40040088,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN0IF2MSK2,       0x4004008C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN0IF2ARB1,       0x40040090,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN0IF2ARB2,       0x40040094,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN0IF2MCTL,       0x40040098,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN0IF2DA1,        0x4004009C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF2DA2,        0x400400A0,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF2DB1,        0x400400A4,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0IF2DB2,        0x400400A8,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN0TXRQ1,         0x40040100,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN0TXRQ2,         0x40040104,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN0NWDA1,         0x40040120,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN0NWDA2,         0x40040124,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN0MSG1INT,       0x40040140,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN0MSG2INT,       0x40040144,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN0MSG1VAL,       0x40040160,__READ        ,__canmsgxval_bits);
__IO_REG32_BIT(CAN0MSG2VAL,       0x40040164,__READ        ,__canmsgxval_bits);

/***************************************************************************
 **
 ** CAN1
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN1CTL,           0x40041000,__READ_WRITE  ,__canctl_bits);
__IO_REG32_BIT(CAN1STS,           0x40041004,__READ_WRITE  ,__cansts_bits);
__IO_REG32_BIT(CAN1ERR,           0x40041008,__READ        ,__canerr_bits);
__IO_REG32_BIT(CAN1BIT,           0x4004100C,__READ_WRITE  ,__canbit_bits);
__IO_REG32_BIT(CAN1INT,           0x40041010,__READ        ,__canint_bits);
__IO_REG32_BIT(CAN1TST,           0x40041014,__READ_WRITE  ,__cantst_bits);
__IO_REG32_BIT(CAN1BRPE,          0x40041018,__READ_WRITE  ,__canbrpe_bits);
__IO_REG32_BIT(CAN1IF1CRQ,        0x40041020,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN1IF1CMSK,       0x40041024,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN1IF1MSK1,       0x40041028,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN1IF1MSK2,       0x4004102C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN1IF1ARB1,       0x40041030,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN1IF1ARB2,       0x40041034,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN1IF1MCTL,       0x40041038,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN1IF1DA1,        0x4004103C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF1DA2,        0x40041040,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF1DB1,        0x40041044,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF1DB2,        0x40041048,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF2CRQ,        0x40041080,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN1IF2CMSK,       0x40041084,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN1IF2MSK1,       0x40041088,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN1IF2MSK2,       0x4004108C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN1IF2ARB1,       0x40041090,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN1IF2ARB2,       0x40041094,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN1IF2MCTL,       0x40041098,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN1IF2DA1,        0x4004109C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF2DA2,        0x400410A0,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF2DB1,        0x400410A4,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1IF2DB2,        0x400410A8,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN1TXRQ1,         0x40041100,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN1TXRQ2,         0x40041104,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN1NWDA1,         0x40041120,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN1NWDA2,         0x40041124,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN1MSG1INT,       0x40041140,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN1MSG2INT,       0x40041144,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN1MSG1VAL,       0x40041160,__READ        ,__canmsgxval_bits);
__IO_REG32_BIT(CAN1MSG2VAL,       0x40041164,__READ        ,__canmsgxval_bits);

/***************************************************************************
 **
 ** CAN2
 **
 ***************************************************************************/
__IO_REG32_BIT(CAN2CTL,           0x40042000,__READ_WRITE  ,__canctl_bits);
__IO_REG32_BIT(CAN2STS,           0x40042004,__READ_WRITE  ,__cansts_bits);
__IO_REG32_BIT(CAN2ERR,           0x40042008,__READ        ,__canerr_bits);
__IO_REG32_BIT(CAN2BIT,           0x4004200C,__READ_WRITE  ,__canbit_bits);
__IO_REG32_BIT(CAN2INT,           0x40042010,__READ        ,__canint_bits);
__IO_REG32_BIT(CAN2TST,           0x40042014,__READ_WRITE  ,__cantst_bits);
__IO_REG32_BIT(CAN2BRPE,          0x40042018,__READ_WRITE  ,__canbrpe_bits);
__IO_REG32_BIT(CAN2IF1CRQ,        0x40042020,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN2IF1CMSK,       0x40042024,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN2IF1MSK1,       0x40042028,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN2IF1MSK2,       0x4004202C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN2IF1ARB1,       0x40042030,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN2IF1ARB2,       0x40042034,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN2IF1MCTL,       0x40042038,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN2IF1DA1,        0x4004203C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF1DA2,        0x40042040,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF1DB1,        0x40042044,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF1DB2,        0x40042048,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF2CRQ,        0x40042080,__READ_WRITE  ,__canifcrq_bits);
__IO_REG32_BIT(CAN2IF2CMSK,       0x40042084,__READ_WRITE  ,__canifcmsk_bits);
__IO_REG32_BIT(CAN2IF2MSK1,       0x40042088,__READ_WRITE  ,__canifmsk1_bits);
__IO_REG32_BIT(CAN2IF2MSK2,       0x4004208C,__READ_WRITE  ,__canifmsk2_bits);
__IO_REG32_BIT(CAN2IF2ARB1,       0x40042090,__READ_WRITE  ,__canifarb1_bits);
__IO_REG32_BIT(CAN2IF2ARB2,       0x40042094,__READ_WRITE  ,__canifarb2_bits);
__IO_REG32_BIT(CAN2IF2MCTL,       0x40042098,__READ_WRITE  ,__canifmctl_bits);
__IO_REG32_BIT(CAN2IF2DA1,        0x4004209C,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF2DA2,        0x400420A0,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF2DB1,        0x400420A4,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2IF2DB2,        0x400420A8,__READ_WRITE  ,__canifdxy_bits);
__IO_REG32_BIT(CAN2TXRQ1,         0x40042100,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN2TXRQ2,         0x40042104,__READ        ,__cantxrqx_bits);
__IO_REG32_BIT(CAN2NWDA1,         0x40042120,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN2NWDA2,         0x40042124,__READ        ,__cannwdax_bits);
__IO_REG32_BIT(CAN2MSG1INT,       0x40042140,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN2MSG2INT,       0x40042144,__READ        ,__canmsgxint_bits);
__IO_REG32_BIT(CAN2MSG1VAL,       0x40042160,__READ        ,__canmsgxval_bits);
__IO_REG32_BIT(CAN2MSG2VAL,       0x40042164,__READ        ,__canmsgxval_bits);

/***************************************************************************
 **
 ** Ethernet MAC
 **
 ***************************************************************************/
__IO_REG32_BIT(MACRIS,            0x40048000,__READ_WRITE  ,__macris_bits);
#define MACIACK MACRIS
#define MACIACK_bit MACRIS_bit
__IO_REG32_BIT(MACIM,             0x40048004,__READ_WRITE  ,__macim_bits);
__IO_REG32_BIT(MACRCTL,           0x40048008,__READ_WRITE  ,__macrctl_bits);
__IO_REG32_BIT(MACTCTL,           0x4004800C,__READ_WRITE  ,__mactctl_bits);
__IO_REG32_BIT(MACDATA,           0x40048010,__READ_WRITE  ,__macdata_bits);
__IO_REG32_BIT(MACIA0,            0x40048014,__READ_WRITE  ,__macia0_bits);
__IO_REG32_BIT(MACIA1,            0x40048018,__READ_WRITE  ,__macia1_bits);
__IO_REG32_BIT(MACTHR,            0x4004801C,__READ_WRITE  ,__macthr_bits);
__IO_REG32_BIT(MACMCTL,           0x40048020,__READ_WRITE  ,__macmctl_bits);
__IO_REG32_BIT(MACMDV,            0x40048024,__READ_WRITE  ,__macmdv_bits);
__IO_REG32_BIT(MACMADD,           0x40048028,__READ        ,__macmadd_bits);
__IO_REG32_BIT(MACMTXD,           0x4004802C,__READ_WRITE  ,__macmtxd_bits);
__IO_REG32_BIT(MACMRXD,           0x40048030,__READ_WRITE  ,__macmrxd_bits);
__IO_REG32_BIT(MACNP,             0x40048034,__READ        ,__macnp_bits);
__IO_REG32_BIT(MACTR,             0x40048038,__READ_WRITE  ,__mactr_bits);
__IO_REG32_BIT(MACTS,             0x4004803C,__READ_WRITE  ,__macts_bits);
__IO_REG32_BIT(MACLED,            0x40048040,__READ_WRITE  ,__macled_bits);
__IO_REG32_BIT(MDIX,              0x40048044,__READ_WRITE  ,__mdix_bits);

/***************************************************************************
 **
 ** USB
 **
 ***************************************************************************/
__IO_REG8_BIT( USBFADDR,          0x40050000,__READ_WRITE ,__usbfaddr_bits);
__IO_REG8_BIT( USBPOWER,          0x40050001,__READ_WRITE ,__usbpower_bits);
__IO_REG16_BIT(USBTXIS,           0x40050002,__READ       ,__usbtxis_bits);
__IO_REG16_BIT(USBRXIS,           0x40050004,__READ       ,__usbrxis_bits);
__IO_REG16_BIT(USBTXIE,           0x40050006,__READ_WRITE ,__usbtxie_bits);
__IO_REG16_BIT(USBRXIE,           0x40050008,__READ_WRITE ,__usbrxie_bits);
__IO_REG8_BIT( USBIS,             0x4005000A,__READ				,__usbis_bits);
__IO_REG8_BIT( USBIE,             0x4005000B,__READ_WRITE ,__usbie_bits);
__IO_REG16_BIT(USBFRAME,          0x4005000C,__READ				,__usbframe_bits);
__IO_REG8_BIT( USBEPIDX,          0x4005000E,__READ_WRITE ,__usbepidx_bits);
__IO_REG8_BIT( USBTEST,           0x4005000F,__READ_WRITE ,__usbtest_bits);
__IO_REG32(    USBFIFO0,          0x40050020,__READ_WRITE );
__IO_REG32(    USBFIFO1,          0x40050024,__READ_WRITE );
__IO_REG32(    USBFIFO2,          0x40050028,__READ_WRITE );
__IO_REG32(    USBFIFO3,          0x4005002C,__READ_WRITE );
__IO_REG32(    USBFIFO4,          0x40050030,__READ_WRITE );
__IO_REG32(    USBFIFO5,          0x40050034,__READ_WRITE );
__IO_REG32(    USBFIFO6,          0x40050038,__READ_WRITE );
__IO_REG32(    USBFIFO7,          0x4005003C,__READ_WRITE );
__IO_REG32(    USBFIFO8,          0x40050040,__READ_WRITE );
__IO_REG32(    USBFIFO9,          0x40050044,__READ_WRITE );
__IO_REG32(    USBFIFO10,         0x40050048,__READ_WRITE );
__IO_REG32(    USBFIFO11,         0x4005004C,__READ_WRITE );
__IO_REG32(    USBFIFO12,         0x40050050,__READ_WRITE );
__IO_REG32(    USBFIFO13,         0x40050054,__READ_WRITE );
__IO_REG32(    USBFIFO14,         0x40050058,__READ_WRITE );
__IO_REG32(    USBFIFO15,         0x4005005C,__READ_WRITE );
__IO_REG8_BIT( USBDEVCTL,         0x40050060,__READ       ,__usbdevctl_bits);
__IO_REG8_BIT( USBTXFIFOSZ,       0x40050062,__READ_WRITE ,__usbtxfifosz_bits);
__IO_REG8_BIT( USBRXFIFOSZ,       0x40050063,__READ_WRITE ,__usbtxfifosz_bits);
__IO_REG16_BIT(USBTXFIFOADD,      0x40050064,__READ_WRITE ,__usbtxfifoadd_bits);
__IO_REG16_BIT(USBRXFIFOADD,      0x40050066,__READ_WRITE ,__usbtxfifoadd_bits);
__IO_REG8_BIT( USBCONTIM,         0x4005007A,__READ_WRITE ,__usbcontim_bits);
__IO_REG8(     USBVPLEN,          0x4005007B,__READ_WRITE );
__IO_REG8(     USBFSEOF,          0x4005007D,__READ_WRITE );
__IO_REG8(     USBLSEOF,          0x4005007E,__READ_WRITE );
__IO_REG8_BIT( USBTXFUNCADDR0,    0x40050080,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR0,     0x40050082,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT0,     0x40050083,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR1,    0x40050088,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR1,     0x4005008A,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT1,     0x4005008B,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR1,    0x4005008C,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR1,     0x4005008E,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT1,     0x4005008F,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR2,    0x40050090,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR2,     0x40050092,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT2,     0x40050093,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR2,    0x40050094,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR2,     0x40050096,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT2,     0x40050097,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR3,    0x40050098,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR3,     0x4005009A,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT3,     0x4005009B,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR3,    0x4005009C,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR3,     0x4005009E,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT3,     0x4005009F,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR4,    0x400500A0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR4,     0x400500A2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT4,     0x400500A3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR4,    0x400500A4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR4,     0x400500A6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT4,     0x400500A7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR5,    0x400500A8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR5,     0x400500AA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT5,     0x400500AB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR5,    0x400500AC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR5,     0x400500AE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT5,     0x400500AF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR6,    0x400500B0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR6,     0x400500B2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT6,     0x400500B3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR6,    0x400500B4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR6,     0x400500B6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT6,     0x400500B7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR7,    0x400500B8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR7,     0x400500BA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT7,     0x400500BB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR7,    0x400500BC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR7,     0x400500BE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT7,     0x400500BF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR8,    0x400500C0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR8,     0x400500C2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT8,     0x400500C3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR8,    0x400500C4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR8,     0x400500C6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT8,     0x400500C7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR9,    0x400500C8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR9,     0x400500CA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT9,     0x400500CB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR9,    0x400500CC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR9,     0x400500CE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT9,     0x400500CF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR10,   0x400500D0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR10,    0x400500D2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT10,    0x400500D3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR10,   0x400500D4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR10,    0x400500D6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT10,    0x400500D7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR11,   0x400500D8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR11,    0x400500DA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT11,    0x400500DB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR11,   0x400500DC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR11,    0x400500DE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT11,    0x400500DF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR12,   0x400500E0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR12,    0x400500E2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT12,    0x400500E3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR12,   0x400500E4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR12,    0x400500E6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT12,    0x400500E7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR13,   0x400500E8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR13,    0x400500EA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT13,    0x400500EB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR13,   0x400500EC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR13,    0x400500EE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT13,    0x400500EF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR14,   0x400500F0,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR14,    0x400500F2,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT14,    0x400500F3,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR14,   0x400500F4,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR14,    0x400500F6,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT14,    0x400500F7,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBTXFUNCADDR15,   0x400500F8,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBTXHUBADDR15,    0x400500FA,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBTXHUBPORT15,    0x400500FB,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBRXFUNCADDR15,   0x400500FC,__READ_WRITE ,__usbtxfuncaddr_bits);
__IO_REG8_BIT( USBRXHUBADDR15,    0x400500FE,__READ_WRITE ,__usbtxhubaddr_bits);
__IO_REG8_BIT( USBRXHUBPORT15,    0x400500FF,__READ_WRITE ,__usbtxhubport_bits);
__IO_REG8_BIT( USBCSRL0,          0x40050102,__READ_WRITE ,__usbcsrl0_bits);
#define       USBDCSRL0           USBCSRL0
#define       USBDCSRL0_bit       USBCSRL0_bit.__usbdcsrl
__IO_REG8_BIT( USBCSRH0,          0x40050103,__READ_WRITE ,__usbcsrh0_bits);
__IO_REG8_BIT( USBCOUNT0,         0x40050108,__READ_WRITE ,__usbcount0_bits);
__IO_REG8_BIT( USBTYPE0,          0x4005010A,__READ_WRITE ,__usbtype0_bits);
__IO_REG8_BIT( USBNAKLMT,         0x4005010B,__READ_WRITE ,__usbnaklmt_bits);
__IO_REG16_BIT(USBTXMAXP1,        0x40050110,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL1,        0x40050112,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL1         USBTXCSRL1
#define       USBDTXCSRL1_bit     USBTXCSRL1_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH1,        0x40050113,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP1,        0x40050114,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL1,        0x40050116,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL1         USBRXCSRL1
#define       USBDRXCSRL1_bit     USBRXCSRL1_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH1,       	0x40050117,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH1         USBRXCSRH1
#define       USBDRXCSRH1_bit     USBRXCSRH1_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT1,       0x40050118,__READ_WRITE ,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE1,        0x4005011A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL1,    0x4005011B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE1,        0x4005011C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL1,    0x4005011D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP2,        0x40050120,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL2,        0x40050122,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL2         USBTXCSRL2
#define       USBDTXCSRL2_bit     USBTXCSRL2_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH2,        0x40050123,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP2,        0x40050124,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL2,        0x40050126,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL2         USBRXCSRL2
#define       USBDRXCSRL2_bit     USBRXCSRL2_bit.__usbdrxcsrl  
__IO_REG8_BIT( USBRXCSRH2,        0x40050127,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH2         USBRXCSRH2
#define       USBDRXCSRH2_bit     USBRXCSRH2_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT2,       0x40050128,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE2,        0x4005012A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL2,    0x4005012B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE2,        0x4005012C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL2,    0x4005012D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP3,        0x40050130,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL3,        0x40050132,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL3         USBTXCSRL3
#define       USBDTXCSRL3_bit     USBTXCSRL3_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH3,        0x40050133,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP3,        0x40050134,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL3,        0x40050136,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL3         USBRXCSRL3
#define       USBDRXCSRL3_bit     USBRXCSRL3_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH3,        0x40050137,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH3         USBRXCSRH3
#define       USBDRXCSRH3_bit     USBRXCSRH3_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT3,       0x40050138,__READ		 		,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE3,        0x4005013A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL3,    0x4005013B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE3,        0x4005013C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL3,    0x4005013D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP4,        0x40050140,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL4,        0x40050142,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL4         USBTXCSRL4
#define       USBDTXCSRL4_bit     USBTXCSRL4_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH4,        0x40050143,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP4,        0x40050144,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL4,        0x40050146,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL4         USBRXCSRL4
#define       USBDRXCSRL4_bit     USBRXCSRL4_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH4,        0x40050147,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH4         USBRXCSRH4
#define       USBDRXCSRH4_bit     USBRXCSRH4_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT4,       0x40050148,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE4,        0x4005014A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL4,    0x4005014B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE4,        0x4005014C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL4,    0x4005014D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP5,        0x40050150,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL5,        0x40050152,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL5         USBTXCSRL5
#define       USBDTXCSRL5_bit     USBTXCSRL5_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH5,        0x40050153,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP5,        0x40050154,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL5,        0x40050156,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL5         USBRXCSRL5
#define       USBDRXCSRL5_bit     USBRXCSRL5_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH5,        0x40050157,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH5         USBRXCSRH5
#define       USBDRXCSRH5_bit     USBRXCSRH5_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT5,       0x40050158,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE5,        0x4005015A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL5,    0x4005015B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE5,        0x4005015C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL5,    0x4005015D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP6,        0x40050160,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL6,        0x40050162,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL6         USBTXCSRL6
#define       USBDTXCSRL6_bit     USBTXCSRL6_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH6,        0x40050163,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP6,        0x40050164,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL6,        0x40050166,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL6         USBRXCSRL6
#define       USBDRXCSRL6_bit     USBRXCSRL6_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH6,        0x40050167,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH6         USBRXCSRH6
#define       USBDRXCSRH6_bit     USBRXCSRH6_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT6,       0x40050168,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE6,        0x4005016A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL6,    0x4005016B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE6,        0x4005016C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL6,    0x4005016D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP7,        0x40050170,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL7,        0x40050172,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL7         USBTXCSRL7
#define       USBDTXCSRL7_bit     USBTXCSRL7_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH7,        0x40050173,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP7,        0x40050174,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL7,        0x40050176,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL7         USBRXCSRL7
#define       USBDRXCSRL7_bit     USBRXCSRL7_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH7,        0x40050177,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH7         USBRXCSRH7
#define       USBDRXCSRH7_bit     USBRXCSRH7_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT7,       0x40050178,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE7,        0x4005017A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL7,    0x4005017B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE7,        0x4005017C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL7,    0x4005017D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP8,        0x40050180,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL8,        0x40050182,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL8         USBTXCSRL8
#define       USBDTXCSRL8_bit     USBTXCSRL8_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH8,        0x40050183,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP8,        0x40050184,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL8,        0x40050186,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL8         USBRXCSRL8
#define       USBDRXCSRL8_bit     USBRXCSRL8_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH8,        0x40050187,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH8         USBRXCSRH8
#define       USBDRXCSRH8_bit     USBRXCSRH8_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT8,       0x40050188,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE8,        0x4005018A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL8,    0x4005018B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE8,        0x4005018C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL8,    0x4005018D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP9,        0x40050190,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL9,        0x40050192,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL9         USBTXCSRL9
#define       USBDTXCSRL9_bit     USBTXCSRL9_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH9,        0x40050193,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP9,        0x40050194,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL9,        0x40050196,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL9         USBRXCSRL9
#define       USBDRXCSRL9_bit     USBRXCSRL9_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH9,        0x40050197,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH9         USBRXCSRH9
#define       USBDRXCSRH9_bit     USBRXCSRH9_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT9,       0x40050198,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE9,        0x4005019A,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL9,    0x4005019B,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE9,        0x4005019C,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL9,    0x4005019D,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP10,       0x400501A0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL10,       0x400501A2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL10        USBTXCSRL10
#define       USBDTXCSRL10_bit    USBTXCSRL10_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH10,       0x400501A3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP10,       0x400501A4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL10,       0x400501A6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL10        USBRXCSRL10
#define       USBDRXCSRL10_bit    USBRXCSRL10_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH10,       0x400501A7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH10        USBRXCSRH10
#define       USBDRXCSRH10_bit    USBRXCSRH10_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT10,      0x400501A8,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE10,       0x400501AA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL10,   0x400501AB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE10,       0x400501AC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL10,   0x400501AD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP11,       0x400501B0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL11,       0x400501B2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL11        USBTXCSRL11
#define       USBDTXCSRL11_bit    USBTXCSRL11_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH11,       0x400501B3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP11,       0x400501B4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL11,       0x400501B6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL11        USBRXCSRL11
#define       USBDRXCSRL11_bit    USBRXCSRL11_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH11,       0x400501B7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH11        USBRXCSRH11
#define       USBDRXCSRH11_bit    USBRXCSRH11_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT11,      0x400501B8,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE11,       0x400501BA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL11,   0x400501BB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE11,       0x400501BC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL11,   0x400501BD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP12,       0x400501C0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL12,       0x400501C2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL12        USBTXCSRL12
#define       USBDTXCSRL12_bit    USBTXCSRL12_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH12,       0x400501C3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP12,       0x400501C4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL12,       0x400501C6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL12        USBRXCSRL12
#define       USBDRXCSRL12_bit    USBRXCSRL12_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH12,       0x400501C7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH12        USBRXCSRH12
#define       USBDRXCSRH12_bit    USBRXCSRH12_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT12,      0x400501C8,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE12,       0x400501CA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL12,   0x400501CB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE12,       0x400501CC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL12,   0x400501CD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP13,       0x400501D0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL13,       0x400501D2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL13        USBTXCSRL13
#define       USBDTXCSRL13_bit    USBTXCSRL13_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH13,       0x400501D3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP13,       0x400501D4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL13,       0x400501D6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL13        USBRXCSRL13
#define       USBDRXCSRL13_bit    USBRXCSRL13_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH13,       0x400501D7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH13        USBRXCSRH13
#define       USBDRXCSRH13_bit    USBRXCSRH13_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT13,      0x400501D8,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE13,       0x400501DA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL13,   0x400501DB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE13,       0x400501DC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL13,   0x400501DD,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP14,       0x400501E0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL14,       0x400501E2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL14        USBTXCSRL14
#define       USBDTXCSRL14_bit    USBTXCSRL14_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH14,       0x400501E3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP14,       0x400501E4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL14,       0x400501E6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL14        USBRXCSRL14
#define       USBDRXCSRL14_bit    USBRXCSRL14_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH14,       0x400501E7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH14        USBRXCSRH14
#define       USBDRXCSRH14_bit    USBRXCSRH14_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT14,      0x400501E8,__READ				,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE14,       0x400501EA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL14,   0x400501EB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE14,       0x400501EC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8( 		 USBRXINTERVAL14,   0x400501ED,__READ_WRITE );
__IO_REG16_BIT(USBTXMAXP15,       0x400501F0,__READ_WRITE ,__usbtxmaxp_bits);
__IO_REG8_BIT( USBTXCSRL15,       0x400501F2,__READ_WRITE ,__usbtxcsrl_bits);
#define       USBDTXCSRL15        USBTXCSRL15
#define       USBDTXCSRL15_bit    USBTXCSRL15_bit.__usbdtxcsrl
__IO_REG8_BIT( USBTXCSRH15,       0x400501F3,__READ_WRITE ,__usbtxcsrh_bits);
__IO_REG16_BIT(USBRXMAXP15,       0x400501F4,__READ_WRITE ,__usbrxmaxp_bits);
__IO_REG8_BIT( USBRXCSRL15,       0x400501F6,__READ_WRITE ,__usbrxcsrl_bits);
#define       USBDRXCSRL15        USBRXCSRL15
#define       USBDRXCSRL15_bit    USBRXCSRL15_bit.__usbdrxcsrl
__IO_REG8_BIT( USBRXCSRH15,       0x400501F7,__READ_WRITE ,__usbrxcsrh_bits);
#define       USBDRXCSRH15        USBRXCSRH15
#define       USBDRXCSRH15_bit    USBRXCSRH15_bit.__usbdrxcsrh
__IO_REG16_BIT(USBRXCOUNT15,      0x400501F8,__READ 			,__usbrxcount_bits);
__IO_REG8_BIT( USBTXTYPE15,       0x400501FA,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBTXINTERVAL15,   0x400501FB,__READ_WRITE );
__IO_REG8_BIT( USBRXTYPE15,       0x400501FC,__READ_WRITE ,__usbtxtype_bits);
__IO_REG8(     USBRXINTERVAL15,   0x400501FD,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT1,    0x40050304,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT2,    0x40050308,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT3,    0x4005030C,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT4,    0x40050310,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT5,    0x40050314,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT6,    0x40050318,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT7,    0x4005031C,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT8,    0x40050320,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT9,    0x40050324,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT10,   0x40050328,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT11,   0x4005032C,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT12,   0x40050330,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT13,   0x40050334,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT14,   0x40050338,__READ_WRITE );
__IO_REG16(    USBRQPKTCOUNT15,   0x4005033C,__READ_WRITE );
__IO_REG16_BIT(USBRXDPKTBUFDIS,   0x40050340,__READ_WRITE ,__usbrxdpktbufdis_bits);
__IO_REG16_BIT(USBTXDPKTBUFDIS,   0x40050342,__READ_WRITE ,__usbrxdpktbufdis_bits);
__IO_REG32_BIT(USBEPC,            0x40050400,__READ_WRITE ,__usbepc_bits);
__IO_REG32_BIT(USBEPCRIS,         0x40050404,__READ       ,__usbepcris_bits);
__IO_REG32_BIT(USBEPCIM,          0x40050408,__READ_WRITE ,__usbepcris_bits);
__IO_REG32_BIT(USBEPCISC,         0x4005040C,__READ_WRITE ,__usbepcris_bits);
__IO_REG32_BIT(USBDRRIS,          0x40050410,__READ				,__usbdrris_bits);
__IO_REG32_BIT(USBDRIM,           0x40050414,__READ_WRITE ,__usbdrris_bits);
__IO_REG32_BIT(USBDRISC,          0x40050418,__WRITE 			,__usbdrris_bits);
__IO_REG32_BIT(USBGPCS,           0x4005041C,__READ_WRITE ,__usbgpcs_bits);
__IO_REG32_BIT(USBVDC,            0x40050430,__READ_WRITE ,__usbvdc_bits);
__IO_REG32_BIT(USBVDCRIS,         0x40050434,__READ				,__usbvdcris_bits);
__IO_REG32_BIT(USBVDCIM,         	0x40050438,__READ_WRITE ,__usbvdcris_bits);
__IO_REG32_BIT(USBVDCISC,         0x4005043C,__READ_WRITE ,__usbvdcris_bits);
__IO_REG32_BIT(USBIDVRIS,         0x40050444,__READ				,__usbidvris_bits);
__IO_REG32_BIT(USBIDVIM,          0x40050448,__READ_WRITE	,__usbidvris_bits);
__IO_REG32_BIT(USBIDVISC,         0x4005044C,__READ_WRITE	,__usbidvris_bits);
__IO_REG32_BIT(USBDMASEL,         0x40050450,__READ_WRITE	,__usbdmasel_bits);

/***************************************************************************
 **
 ** Analog Comparators
 **
 ***************************************************************************/
__IO_REG32_BIT(ACMIS,             0x4003C000,__READ_WRITE	,__acmis_bits);
__IO_REG32_BIT(ACRIS,             0x4003C004,__READ       ,__acmis_bits);
__IO_REG32_BIT(ACINTEN,           0x4003C008,__READ_WRITE ,__acmis_bits);
__IO_REG32_BIT(ACREFCTL,          0x4003C010,__READ_WRITE ,__acrefctl_bits);
__IO_REG32_BIT(ACSTAT0,           0x4003C020,__READ       ,__acstat_bits);
__IO_REG32_BIT(ACCTL0,            0x4003C024,__READ_WRITE ,__acctl_bits);
__IO_REG32_BIT(ACSTAT1,           0x4003C040,__READ       ,__acstat_bits);
__IO_REG32_BIT(ACCTL1,            0x4003C044,__READ_WRITE ,__acctl_bits);
__IO_REG32_BIT(ACSTAT2,           0x4003C060,__READ       ,__acstat_bits);
__IO_REG32_BIT(ACCTL2,            0x4003C064,__READ_WRITE ,__acctl_bits);

/***************************************************************************
 **
 ** PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCTL,            0x40028000,__READ_WRITE  ,__pwmctl_bits);
__IO_REG32_BIT(PWMSYNC,           0x40028004,__READ_WRITE  ,__pwmsync_bits);
__IO_REG32_BIT(PWMENABLE,         0x40028008,__READ_WRITE  ,__pwmenable_bits);
__IO_REG32_BIT(PWMINVERT,         0x4002800C,__READ_WRITE  ,__pwminvert_bits);
__IO_REG32_BIT(PWMFAULT,          0x40028010,__READ_WRITE  ,__pwmfault_bits);
__IO_REG32_BIT(PWMINTEN,          0x40028014,__READ_WRITE  ,__pwminten_bits);
__IO_REG32_BIT(PWMRIS,            0x40028018,__READ        ,__pwmris_bits);
__IO_REG32_BIT(PWMISC,            0x4002801C,__READ_WRITE  ,__pwmisc_bits);
__IO_REG32_BIT(PWMSTATUS,         0x40028020,__READ        ,__pwmstatus_bits);
__IO_REG32_BIT(PWMFAULTVAL,       0x40028024,__READ_WRITE  ,__pwmfaultval_bits);
__IO_REG32_BIT(PWMENUPD,       		0x40028028,__READ_WRITE  ,__pwmenupd_bits);
__IO_REG32_BIT(PWM0CTL,           0x40028040,__READ_WRITE  ,__pwm0ctl_bits);
__IO_REG32_BIT(PWM0INTEN,         0x40028044,__READ_WRITE  ,__pwm0inten_bits);
__IO_REG32_BIT(PWM0RIS,           0x40028048,__READ        ,__pwm0ris_bits);
__IO_REG32_BIT(PWM0ISC,           0x4002804C,__READ_WRITE  ,__pwm0isc_bits);
__IO_REG32_BIT(PWM0LOAD,          0x40028050,__READ_WRITE  ,__pwm0load_bits);
__IO_REG32_BIT(PWM0COUNT,         0x40028054,__READ        ,__pwm0count_bits);
__IO_REG32_BIT(PWM0CMPA,          0x40028058,__READ_WRITE  ,__pwm0cmpa_bits);
__IO_REG32_BIT(PWM0CMPB,          0x4002805C,__READ_WRITE  ,__pwm0cmpb_bits);
__IO_REG32_BIT(PWM0GENA,          0x40028060,__READ_WRITE  ,__pwm0gena_bits);
__IO_REG32_BIT(PWM0GENB,          0x40028064,__READ_WRITE  ,__pwm0genb_bits);
__IO_REG32_BIT(PWM0DBCTL,         0x40028068,__READ_WRITE  ,__pwm0dbctl_bits);
__IO_REG32_BIT(PWM0DBRISE,        0x4002806C,__READ_WRITE  ,__pwm0dbrise_bits);
__IO_REG32_BIT(PWM0DBFALL,        0x40028070,__READ_WRITE  ,__pwm0dbfall_bits);
__IO_REG32_BIT(PWM0FLTSRC0,       0x40028074,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM0FLTSRC1,       0x40028078,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM0MINFLTPER,     0x4002807C,__READ_WRITE  ,__pwm0minfltper_bits);
__IO_REG32_BIT(PWM1CTL,           0x40028080,__READ_WRITE  ,__pwm0ctl_bits);
__IO_REG32_BIT(PWM1INTEN,         0x40028084,__READ_WRITE  ,__pwm0inten_bits);
__IO_REG32_BIT(PWM1RIS,           0x40028088,__READ        ,__pwm0ris_bits);
__IO_REG32_BIT(PWM1ISC,           0x4002808C,__READ_WRITE  ,__pwm0isc_bits);
__IO_REG32_BIT(PWM1LOAD,          0x40028090,__READ_WRITE  ,__pwm0load_bits);
__IO_REG32_BIT(PWM1COUNT,         0x40028094,__READ        ,__pwm0count_bits);
__IO_REG32_BIT(PWM1CMPA,          0x40028098,__READ_WRITE  ,__pwm0cmpa_bits);
__IO_REG32_BIT(PWM1CMPB,          0x4002809C,__READ_WRITE  ,__pwm0cmpb_bits);
__IO_REG32_BIT(PWM1GENA,          0x400280A0,__READ_WRITE  ,__pwm0gena_bits);
__IO_REG32_BIT(PWM1GENB,          0x400280A4,__READ_WRITE  ,__pwm0genb_bits);
__IO_REG32_BIT(PWM1DBCTL,         0x400280A8,__READ_WRITE  ,__pwm0dbctl_bits);
__IO_REG32_BIT(PWM1DBRISE,        0x400280AC,__READ_WRITE  ,__pwm0dbrise_bits);
__IO_REG32_BIT(PWM1DBFALL,        0x400280B0,__READ_WRITE  ,__pwm0dbfall_bits);
__IO_REG32_BIT(PWM1FLTSRC0,       0x400280B4,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM1FLTSRC1,       0x400280B8,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM1MINFLTPER,     0x400280BC,__READ_WRITE  ,__pwm0minfltper_bits);
__IO_REG32_BIT(PWM2CTL,           0x400280C0,__READ_WRITE  ,__pwm0ctl_bits);
__IO_REG32_BIT(PWM2INTEN,         0x400280C4,__READ_WRITE  ,__pwm0inten_bits);
__IO_REG32_BIT(PWM2RIS,           0x400280C8,__READ        ,__pwm0ris_bits);
__IO_REG32_BIT(PWM2ISC,           0x400280CC,__READ_WRITE  ,__pwm0isc_bits);
__IO_REG32_BIT(PWM2LOAD,          0x400280D0,__READ_WRITE  ,__pwm0load_bits);
__IO_REG32_BIT(PWM2COUNT,         0x400280D4,__READ        ,__pwm0count_bits);
__IO_REG32_BIT(PWM2CMPA,          0x400280D8,__READ_WRITE  ,__pwm0cmpa_bits);
__IO_REG32_BIT(PWM2CMPB,          0x400280DC,__READ_WRITE  ,__pwm0cmpb_bits);
__IO_REG32_BIT(PWM2GENA,          0x400280E0,__READ_WRITE  ,__pwm0gena_bits);
__IO_REG32_BIT(PWM2GENB,          0x400280E4,__READ_WRITE  ,__pwm0genb_bits);
__IO_REG32_BIT(PWM2DBCTL,         0x400280E8,__READ_WRITE  ,__pwm0dbctl_bits);
__IO_REG32_BIT(PWM2DBRISE,        0x400280EC,__READ_WRITE  ,__pwm0dbrise_bits);
__IO_REG32_BIT(PWM2DBFALL,        0x400280F0,__READ_WRITE  ,__pwm0dbfall_bits);
__IO_REG32_BIT(PWM2FLTSRC0,       0x400280F4,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM2FLTSRC1,       0x400280F8,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM2MINFLTPER,     0x400280FC,__READ_WRITE  ,__pwm0minfltper_bits);
__IO_REG32_BIT(PWM3CTL,           0x40028100,__READ_WRITE  ,__pwm0ctl_bits);
__IO_REG32_BIT(PWM3INTEN,         0x40028104,__READ_WRITE  ,__pwm0inten_bits);
__IO_REG32_BIT(PWM3RIS,           0x40028108,__READ        ,__pwm0ris_bits);
__IO_REG32_BIT(PWM3ISC,           0x4002810C,__READ_WRITE  ,__pwm0isc_bits);
__IO_REG32_BIT(PWM3LOAD,          0x40028110,__READ_WRITE  ,__pwm0load_bits);
__IO_REG32_BIT(PWM3COUNT,         0x40028114,__READ        ,__pwm0count_bits);
__IO_REG32_BIT(PWM3CMPA,          0x40028118,__READ_WRITE  ,__pwm0cmpa_bits);
__IO_REG32_BIT(PWM3CMPB,          0x4002811C,__READ_WRITE  ,__pwm0cmpb_bits);
__IO_REG32_BIT(PWM3GENA,          0x40028120,__READ_WRITE  ,__pwm0gena_bits);
__IO_REG32_BIT(PWM3GENB,          0x40028124,__READ_WRITE  ,__pwm0genb_bits);
__IO_REG32_BIT(PWM3DBCTL,         0x40028128,__READ_WRITE  ,__pwm0dbctl_bits);
__IO_REG32_BIT(PWM3DBRISE,        0x4002812C,__READ_WRITE  ,__pwm0dbrise_bits);
__IO_REG32_BIT(PWM3DBFALL,        0x40028130,__READ_WRITE  ,__pwm0dbfall_bits);
__IO_REG32_BIT(PWM3FLTSRC0,       0x40028134,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM3FLTSRC1,       0x40028138,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM3MINFLTPER,     0x4002813C,__READ_WRITE  ,__pwm0minfltper_bits);
__IO_REG32_BIT(PWM0FLTSEN,     		0x40028800,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM0FLTSTAT0,     	0x40028804,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM0FLTSTAT1,     	0x40028808,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM1FLTSEN,     		0x40028880,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM1FLTSTAT0,     	0x40028884,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM1FLTSTAT1,     	0x40028888,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM2FLTSEN,     		0x40028900,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM2FLTSTAT0,     	0x40028904,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM2FLTSTAT1,     	0x40028908,__READ_WRITE  ,__pwm0fltsrc_bits);
__IO_REG32_BIT(PWM3FLTSEN,     		0x40028980,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM3FLTSTAT0,     	0x40028984,__READ_WRITE  ,__pwm0fltstat_bits);
__IO_REG32_BIT(PWM3FLTSTAT1,     	0x40028988,__READ_WRITE  ,__pwm0fltsrc_bits);

/***************************************************************************
 **
 ** QEI0
 **
 ***************************************************************************/
__IO_REG32_BIT(QEI0CTL,           0x4002C000,__READ_WRITE  ,__qeictl_bits);
__IO_REG32_BIT(QEI0STAT,          0x4002C004,__READ        ,__qeistat_bits);
__IO_REG32(		 QEI0POS,           0x4002C008,__READ_WRITE  );
__IO_REG32(		 QEI0MAXPOS,        0x4002C00C,__READ_WRITE  );
__IO_REG32(		 QEI0LOAD,          0x4002C010,__READ_WRITE  );
__IO_REG32(		 QEI0TIME,          0x4002C014,__READ        );
__IO_REG32(		 QEI0COUNT,         0x4002C018,__READ        );
__IO_REG32(		 QEI0SPEED,         0x4002C01C,__READ        );
__IO_REG32_BIT(QEI0INTEN,         0x4002C020,__READ_WRITE  ,__qeiinten_bits);
__IO_REG32_BIT(QEI0RIS,           0x4002C024,__READ        ,__qeiris_bits);
__IO_REG32_BIT(QEI0ISC,           0x4002C028,__READ_WRITE  ,__qeiisc_bits);

/***************************************************************************
 **
 ** QEI1
 **
 ***************************************************************************/
__IO_REG32_BIT(QEI1CTL,           0x4002D000,__READ_WRITE  ,__qeictl_bits);
__IO_REG32_BIT(QEI1STAT,          0x4002D004,__READ        ,__qeistat_bits);
__IO_REG32(		 QEI1POS,           0x4002D008,__READ_WRITE  );
__IO_REG32(		 QEI1MAXPOS,        0x4002D00C,__READ_WRITE  );
__IO_REG32(		 QEI1LOAD,          0x4002D010,__READ_WRITE  );
__IO_REG32(		 QEI1TIME,          0x4002D014,__READ        );
__IO_REG32(		 QEI1COUNT,         0x4002D018,__READ        );
__IO_REG32(		 QEI1SPEED,         0x4002D01C,__READ        );
__IO_REG32_BIT(QEI1INTEN,         0x4002D020,__READ_WRITE  ,__qeiinten_bits);
__IO_REG32_BIT(QEI1RIS,           0x4002D024,__READ        ,__qeiris_bits);
__IO_REG32_BIT(QEI1ISC,           0x4002D028,__READ_WRITE  ,__qeiisc_bits);

/***************************************************************************
 **
 ** Core debug
 **
 ***************************************************************************/
__IO_REG32_BIT(DHSR,              0xE000EDF0,__READ_WRITE ,__dhsr_bits);
#define DHCR        DHSR
#define DHCR_bit    DHSR_bit
__IO_REG32_BIT(DCRSR,             0xE000EDF4,__WRITE      ,__dcrsr_bits);
__IO_REG32(    DCRDR,             0xE000EDF8,__READ_WRITE);
__IO_REG32_BIT(DEMCR,             0xE000EDFC,__READ_WRITE ,__demcr_bits);

/***************************************************************************
 **
 ** FPB
 **
 ***************************************************************************/
__IO_REG32_BIT(FP_CTRL,           0xE0002000,__READ_WRITE ,__fp_ctrl_bits);
__IO_REG32_BIT(FP_REMAP,          0xE0002004,__READ_WRITE ,__fp_remap_bits);
__IO_REG32_BIT(FP_COMP0,          0xE0002008,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP1,          0xE000200C,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP2,          0xE0002010,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP3,          0xE0002014,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP4,          0xE0002018,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP5,          0xE000201C,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP6,          0xE0002020,__READ_WRITE ,__fp_comp_bits);
__IO_REG32_BIT(FP_COMP7,          0xE0002024,__READ_WRITE ,__fp_comp_bits);
__IO_REG8(     FP_PERIPID4,       0xE0002FD0,__READ);
__IO_REG8(     FP_PERIPID5,       0xE0002FD4,__READ);
__IO_REG8(     FP_PERIPID6,       0xE0002FD8,__READ);
__IO_REG8(     FP_PERIPID7,       0xE0002FDC,__READ);
__IO_REG8(     FP_PERIPID0,       0xE0002FE0,__READ);
__IO_REG8(     FP_PERIPID1,       0xE0002FE4,__READ);
__IO_REG8(     FP_PERIPID2,       0xE0002FE8,__READ);
__IO_REG8(     FP_PERIPID3,       0xE0002FEC,__READ);
__IO_REG8(     FP_PCELLID0,       0xE0002FF0,__READ);
__IO_REG8(     FP_PCELLID1,       0xE0002FF4,__READ);
__IO_REG8(     FP_PCELLID2,       0xE0002FF8,__READ);
__IO_REG8(     FP_PCELLID3,       0xE0002FFC,__READ);

/***************************************************************************
 **
 ** DWT
 **
 ***************************************************************************/
__IO_REG32_BIT(DWT_CTRL,          0xE0001000,__READ_WRITE ,__dwt_ctrl_bits);
__IO_REG32(    DWT_CYCCNT,        0xE0001004,__READ);
__IO_REG8(     DWT_CPICNT,        0xE0001008,__READ_WRITE);
__IO_REG8(     DWT_EXCCNT,        0xE000100C,__READ_WRITE);
__IO_REG8(     DWT_SLEEPCNT,      0xE0001010,__READ_WRITE);
__IO_REG8(     DWT_LSUCNT,        0xE0001014,__READ_WRITE);
__IO_REG8(     DWT_FOLDCNT,       0xE0001018,__READ_WRITE);
__IO_REG32(    DWT_COMP0,         0xE0001020,__READ_WRITE);
__IO_REG32_BIT(DWT_MASK0,         0xE0001024,__READ_WRITE ,__dwt_mask_bits);
__IO_REG32_BIT(DWT_FUNCTION0,     0xE0001028,__READ_WRITE ,__dwt_function_bits);
__IO_REG32(    DWT_COMP1,         0xE0001030,__READ_WRITE);
__IO_REG32_BIT(DWT_MASK1,         0xE0001034,__READ_WRITE ,__dwt_mask_bits);
__IO_REG32_BIT(DWT_FUNCTION1,     0xE0001038,__READ_WRITE ,__dwt_function_bits);
__IO_REG32(    DWT_COMP2,         0xE0001040,__READ_WRITE);
__IO_REG32_BIT(DWT_MASK2,         0xE0001044,__READ_WRITE ,__dwt_mask_bits);
__IO_REG32_BIT(DWT_FUNCTION2,     0xE0001048,__READ_WRITE ,__dwt_function_bits);
__IO_REG32(    DWT_COMP3,         0xE0001050,__READ_WRITE);
__IO_REG32_BIT(DWT_MASK3,         0xE0001054,__READ_WRITE ,__dwt_mask_bits);
__IO_REG32_BIT(DWT_FUNCTION3,     0xE0001058,__READ_WRITE ,__dwt_function_bits);
__IO_REG8(     DWT_PERIPID4,      0xE0001FD0,__READ);
__IO_REG8(     DWT_PERIPID5,      0xE0001FD4,__READ);
__IO_REG8(     DWT_PERIPID6,      0xE0001FD8,__READ);
__IO_REG8(     DWT_PERIPID7,      0xE0001FDC,__READ);
__IO_REG8(     DWT_PERIPID0,      0xE0001FE0,__READ);
__IO_REG8(     DWT_PERIPID1,      0xE0001FE4,__READ);
__IO_REG8(     DWT_PERIPID2,      0xE0001FE8,__READ);
__IO_REG8(     DWT_PERIPID3,      0xE0001FEC,__READ);
__IO_REG8(     DWT_PCELLID0,      0xE0001FF0,__READ);
__IO_REG8(     DWT_PCELLID1,      0xE0001FF4,__READ);
__IO_REG8(     DWT_PCELLID2,      0xE0001FF8,__READ);
__IO_REG8(     DWT_PCELLID3,      0xE0001FFC,__READ);

/***************************************************************************
 **
 ** ITM
 **
 ***************************************************************************/
__IO_REG32(    ITM_SP0,           0xE0000000,__READ_WRITE);
__IO_REG32(    ITM_SP1,           0xE0000004,__READ_WRITE);
__IO_REG32(    ITM_SP2,           0xE0000008,__READ_WRITE);
__IO_REG32(    ITM_SP3,           0xE000000C,__READ_WRITE);
__IO_REG32(    ITM_SP4,           0xE0000010,__READ_WRITE);
__IO_REG32(    ITM_SP5,           0xE0000014,__READ_WRITE);
__IO_REG32(    ITM_SP6,           0xE0000018,__READ_WRITE);
__IO_REG32(    ITM_SP7,           0xE000001C,__READ_WRITE);
__IO_REG32(    ITM_SP8,           0xE0000020,__READ_WRITE);
__IO_REG32(    ITM_SP9,           0xE0000024,__READ_WRITE);
__IO_REG32(    ITM_SP10,          0xE0000028,__READ_WRITE);
__IO_REG32(    ITM_SP11,          0xE000002C,__READ_WRITE);
__IO_REG32(    ITM_SP12,          0xE0000030,__READ_WRITE);
__IO_REG32(    ITM_SP13,          0xE0000034,__READ_WRITE);
__IO_REG32(    ITM_SP14,          0xE0000038,__READ_WRITE);
__IO_REG32(    ITM_SP15,          0xE000003C,__READ_WRITE);
__IO_REG32(    ITM_SP16,          0xE0000040,__READ_WRITE);
__IO_REG32(    ITM_SP17,          0xE0000044,__READ_WRITE);
__IO_REG32(    ITM_SP18,          0xE0000048,__READ_WRITE);
__IO_REG32(    ITM_SP19,          0xE000004C,__READ_WRITE);
__IO_REG32(    ITM_SP20,          0xE0000050,__READ_WRITE);
__IO_REG32(    ITM_SP21,          0xE0000054,__READ_WRITE);
__IO_REG32(    ITM_SP22,          0xE0000058,__READ_WRITE);
__IO_REG32(    ITM_SP23,          0xE000005C,__READ_WRITE);
__IO_REG32(    ITM_SP24,          0xE0000060,__READ_WRITE);
__IO_REG32(    ITM_SP25,          0xE0000064,__READ_WRITE);
__IO_REG32(    ITM_SP26,          0xE0000068,__READ_WRITE);
__IO_REG32(    ITM_SP27,          0xE000006C,__READ_WRITE);
__IO_REG32(    ITM_SP28,          0xE0000070,__READ_WRITE);
__IO_REG32(    ITM_SP29,          0xE0000074,__READ_WRITE);
__IO_REG32(    ITM_SP30,          0xE0000078,__READ_WRITE);
__IO_REG32(    ITM_SP31,          0xE000007C,__READ_WRITE);
__IO_REG32_BIT(ITM_TE,            0xE0000E00,__READ_WRITE ,__itm_te_bits);
__IO_REG32_BIT(ITM_TP,            0xE0000E40,__READ_WRITE ,__itm_tp_bits);
__IO_REG32_BIT(ITM_CR,            0xE0000E80,__READ_WRITE ,__itm_cr_bits);
__IO_REG32_BIT(ITM_IW,            0xE0000EF8,__WRITE      ,__itm_iw_bits);
__IO_REG32_BIT(ITM_IR,            0xE0000EFC,__READ       ,__itm_ir_bits);
__IO_REG32_BIT(ITM_IMC,           0xE0000F00,__READ_WRITE ,__itm_imc_bits);
__IO_REG32(    ITM_LAR,           0xE0000FB0,__WRITE);
__IO_REG32_BIT(ITM_LSR,           0xE0000FB4,__READ       ,__itm_lsr_bits);
__IO_REG8(     ITM_PERIPID4,      0xE0000FD0,__READ);
__IO_REG8(     ITM_PERIPID5,      0xE0000FD4,__READ);
__IO_REG8(     ITM_PERIPID6,      0xE0000FD8,__READ);
__IO_REG8(     ITM_PERIPID7,      0xE0000FDC,__READ);
__IO_REG8(     ITM_PERIPID0,      0xE0000FE0,__READ);
__IO_REG8(     ITM_PERIPID1,      0xE0000FE4,__READ);
__IO_REG8(     ITM_PERIPID2,      0xE0000FE8,__READ);
__IO_REG8(     ITM_PERIPID3,      0xE0000FEC,__READ);
__IO_REG8(     ITM_PCELLID0,      0xE0000FF0,__READ);
__IO_REG8(     ITM_PCELLID1,      0xE0000FF4,__READ);
__IO_REG8(     ITM_PCELLID2,      0xE0000FF8,__READ);
__IO_REG8(     ITM_PCELLID3,      0xE0000FFC,__READ);

/***************************************************************************
 **
 ** TPIU
 **
 ***************************************************************************/
__IO_REG32_BIT(TPIU_SPSR,         0xE0040000,__READ       ,__tpiu_spsr_bits);
__IO_REG32_BIT(TPIU_CPSR,         0xE0040004,__READ_WRITE ,__tpiu_spsr_bits);
__IO_REG32_BIT(TPIU_COSDR,        0xE0040010,__READ_WRITE ,__tpiu_cosdr_bits);
__IO_REG32_BIT(TPIU_SPPR,         0xE00400F0,__READ_WRITE ,__tpiu_sppr_bits);
__IO_REG32_BIT(TPIU_FFSR,         0xE0040300,__READ       ,__tpiu_ffsr_bits);
__IO_REG32_BIT(TPIU_FFCR,         0xE0040304,__READ       ,__tpiu_ffsr_bits);
__IO_REG32(    TPIU_FSCR,         0xE0040308,__READ);
__IO_REG32_BIT(TPIU_ITATBCTR2,    0xE0040EF0,__READ       ,__tpiu_itatbctr2_bits);
__IO_REG32_BIT(TPIU_ITATBCTR0,    0xE0040EF8,__READ       ,__tpiu_itatbctr0_bits);

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  LM3SXXXX Interrupt Lines
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
#define EII                   16          /* External Interrupt           */
#define NVIC_GPIOA           ( 0 + EII)   /* PORTA                        */
#define NVIC_GPIOB           ( 1 + EII)   /* PORTB                        */
#define NVIC_GPIOC           ( 2 + EII)   /* PORTC                        */
#define NVIC_GPIOD           ( 3 + EII)   /* PORTD                        */
#define NVIC_GPIOE           ( 4 + EII)   /* PORTE                        */
#define NVIC_UART0           ( 5 + EII)   /* UART 0                       */
#define NVIC_UART1           ( 6 + EII)   /* UART 1                       */
#define NVIC_SSI0            ( 7 + EII)   /* SSI0                         */
#define NVIC_I2C0            ( 8 + EII)   /* I2C0                         */
#define NVIC_PWM_FAULT       ( 9 + EII)   /* PWM Fault                    */
#define NVIC_PWM_GEN0        (10 + EII)   /* PWM Generator 0              */
#define NVIC_PWM_GEN1        (11 + EII)   /* PWM Generator 1              */
#define NVIC_PWM_GEN2        (12 + EII)   /* PWM Generator 2              */
#define NVIC_QEI0            (13 + EII)   /* QEI0                         */
#define NVIC_ADC0_SS0        (14 + EII)   /* ADC Sample Sequencer 0       */
#define NVIC_ADC0_SS1        (15 + EII)   /* ADC Sample Sequencer 1       */
#define NVIC_ADC0_SS2        (16 + EII)   /* ADC Sample Sequencer 2       */
#define NVIC_ADC0_SS3        (17 + EII)   /* ADC Sample Sequencer 3       */
#define NVIC_WDT             (18 + EII)   /* WDT                          */
#define NVIC_TIMER0A         (19 + EII)   /* Timer 0 Channel A            */
#define NVIC_TIMER0B         (20 + EII)   /* Timer 0 Channel B            */
#define NVIC_TIMER1A         (21 + EII)   /* Timer 1 Channel A            */
#define NVIC_TIMER1B         (22 + EII)   /* Timer 1 Channel B            */
#define NVIC_TIMER2A         (23 + EII)   /* Timer 2 Channel A            */
#define NVIC_TIMER2B         (24 + EII)   /* Timer 2 Channel B            */
#define NVIC_ACOMP0          (25 + EII)   /* Analog Comparator 0          */
#define NVIC_ACOMP1          (26 + EII)   /* Analog Comparator 1          */
#define NVIC_ACOMP2          (27 + EII)   /* Analog Comparator 2          */
#define NVIC_SYS_CTRL        (28 + EII)   /* System control               */
#define NVIC_FLASH_CTRL      (29 + EII)   /* Flash controller             */
#define NVIC_GPIOF           (30 + EII)   /* PORTF                        */
#define NVIC_GPIOG           (31 + EII)   /* PORTG                        */
#define NVIC_GPIOH           (32 + EII)   /* PORTH                        */
#define NVIC_UART2           (33 + EII)   /* UART2                        */
#define NVIC_SSI1            (34 + EII)   /* SSI1                         */
#define NVIC_TIMER3A         (35 + EII)   /* Timer 3 Channel A            */
#define NVIC_TIMER3B         (36 + EII)   /* Timer 3 Channel B            */
#define NVIC_I2C1            (37 + EII)   /* I2C1                         */
#define NVIC_QEI1            (38 + EII)   /* QEI1                         */
#define NVIC_CAN0            (39 + EII)   /* CAN0                         */
#define NVIC_CAN1            (40 + EII)   /* CAN1                         */
#define NVIC_CAN2            (41 + EII)   /* CAN2                         */
#define NVIC_ENET            (42 + EII)   /* Ethernet MAC                 */
#define NVIC_HIBERNATION     (43 + EII)   /* Hibernation Module           */
#define NVIC_USB             (44 + EII)   /* USB Module                   */
#define NVIC_PWM_GEN3        (45 + EII)   /* PWM Generator 3              */
#define NVIC_UDMA_SOFT       (46 + EII)   /* uDMA Software                */
#define NVIC_UDMA_ERR        (47 + EII)   /* uDMA Error                   */
#define NVIC_ADC1_SS0        (48 + EII)   /* ADC1 Sample Sequencer 0      */
#define NVIC_ADC1_SS1        (49 + EII)   /* ADC1 Sample Sequencer 1      */
#define NVIC_ADC1_SS2        (50 + EII)   /* ADC1 Sample Sequencer 2      */
#define NVIC_ADC1_SS3        (51 + EII)   /* ADC1 Sample Sequencer 3      */
#define NVIC_I2S0            (52 + EII)   /* I2S0                         */
#define NVIC_EPI             (53 + EII)   /* EPI                          */
#define NVIC_GPIOJ           (54 + EII)   /* PORTJ                        */

#endif    /* __IOLM3SXXXX_H */

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
Interrupt9   = GPIOA          0x40
Interrupt10  = GPIOB          0x44
Interrupt11  = GPIOC          0x48
Interrupt12  = GPIOD          0x4C
Interrupt13  = GPIOE          0x50
Interrupt14  = UART0          0x54
Interrupt15  = UART1          0x58
Interrupt16  = SSI0           0x5C
Interrupt17  = I2C0           0x60
Interrupt18  = PWM_FAULT      0x64
Interrupt19  = PWM0           0x68
Interrupt20  = PWM1           0x6C
Interrupt21  = PWM2           0x70
Interrupt22  = QEI0           0x74
Interrupt23  = ADC00          0x78
Interrupt24  = ADC01          0x7C
Interrupt25  = ADC02          0x80
Interrupt26  = ADC03          0x84
Interrupt27  = WDT            0x88
Interrupt28  = TIMER0A        0x8C
Interrupt29  = TIMER0B        0x90
Interrupt30  = TIMER1A        0x94
Interrupt31  = TIMER1B        0x98
Interrupt32  = TIMER2A        0x9C
Interrupt33  = TIMER2B        0xA0
Interrupt34  = ACOMP0         0xA4
Interrupt35  = ACOMP1         0xA8
Interrupt36  = ACOMP2         0xAC
Interrupt37  = SysCtrl        0xB0
Interrupt38  = FlashCtrl      0xB4
Interrupt39  = GPIOF          0xB8
Interrupt40  = GPIOG          0xBC
Interrupt41  = GPIOH          0xC0
Interrupt42  = UART2          0xC4
Interrupt43  = SSI1           0xC8
Interrupt44  = TIMER3A        0xCC
Interrupt45  = TIMER3B        0xD0
Interrupt46  = I2C1           0xD4
Interrupt47  = QEI0           0xD8
Interrupt48  = CAN0           0xDC
Interrupt49  = CAN1           0xE0
Interrupt50  = CAN2           0xE4
Interrupt51  = ENET           0xE8
Interrupt52  = HIBERNATION    0xEC
Interrupt53  = USB            0xF0
Interrupt54  = PWM3           0xF4
Interrupt55  = uDMA_Software  0xF8
Interrupt56  = uDMA_Error     0xFC
Interrupt57  = ADC10          0x100
Interrupt58  = ADC11          0x104
Interrupt59  = ADC12          0x108
Interrupt60  = ADC13          0x10C
Interrupt61  = I2S0           0x110
Interrupt62  = EPI            0x114
Interrupt63  = GPIOJ          0x118

###DDF-INTERRUPT-END###*/
