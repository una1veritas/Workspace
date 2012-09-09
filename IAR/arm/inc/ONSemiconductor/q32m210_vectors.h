/* ----------------------------------------------------------------------------
 * Copyright (c) 2011 - 2012 Semiconductor Components Industries, LLC (d/b/a ON
 * Semiconductor), All Rights Reserved
 *
 * This code is the property of ON Semiconductor and may not be redistributed
 * in any form without prior written permission from ON Semiconductor.
 * The terms of use and warranty for this code are covered by contractual
 * agreements between ON Semiconductor and the licensee.
 * ----------------------------------------------------------------------------
 * q32m210_vectors.h
 * - Interrupt vector locations
 * 
 * IMPORTANT: Do not include this file directly. Please use <q32.h> or 
 *            <q32m210.h> instead.
 * ----------------------------------------------------------------------------
 * $Revision: 53234 $
 * $Date: 2012-05-15 12:18:28 +0200 (ti, 15 maj 2012) $
 * ------------------------------------------------------------------------- */

#ifndef Q32M210_VECTORS_H
#define Q32M210_VECTORS_H

/* ----------------------------------------------------------------------------
 * Interrupt Vector Table Numbers
 *  - Equivalent to the offset from the base of the external interrupt list
 *  - Can be converted into an index into the vector table by adding 16
 * ------------------------------------------------------------------------- */

typedef enum IRQn
{
  Reset_IRQn                    = -15,      /*  1 Reset Vector */
  NonMaskableInt_IRQn           = -14,      /*  2 Non Maskable Interrupt */
  HardFault_IRQn                = -13,      /*  3 Hard Fault Interrupt */
  MemoryManagement_IRQn         = -12,      /*  4 Memory Management Interrupt */
  BusFault_IRQn                 = -11,      /*  5 Bus Fault Interrupt */
  UsageFault_IRQn               = -10,      /*  6 Usage Fault Interrupt */
  SVCall_IRQn                   = -5,       /* 11 SV Call Interrupt */
  DebugMonitor_IRQn             = -4,       /* 12 Debug Monitor Interrupt */
  PendSV_IRQn                   = -2,       /* 14 Pend SV Interrupt */
  SysTick_IRQn                  = -1,       /* 15 System Tick Interrupt */
  WAKEUP_IRQn                   = 0,        /* 16 Wakeup Interrupt */
  WAKEUP_IF5_PIN0_IRQn          = 1,        /* 17 IF5, Pin 0 Specific Wakeup 
                                             *    Interrupt */
  RTC_ALARM_IRQn                = 2,        /* 18 Real-Time Clock (RTC) Alarm 
                                             *    Interrupt */
  RTC_CLOCK_IRQn                = 3,        /* 19 Real-Time Clock (RTC) Clock 
                                             *    Interrupt */
  THRESHOLD_COMPARE_IRQn        = 4,        /* 20 Threshold Compare Interrupt */
  NO_EXTCLK_DETECT_IRQn         = 5,        /* 21 No EXTCLK Detected 
                                             *    Interrupt */
  GPIO_GP0_IRQn                 = 6,        /* 22 GPIO 0 Interrupt */
  GPIO_GP1_IRQn                 = 7,        /* 23 GPIO 1 Interrupt */
  WATCHDOG_IRQn                 = 8,        /* 24 Watchdog Interrupt */
  TIMER0_IRQn                   = 9,        /* 25 Timer 0 Interrupt */
  TIMER1_IRQn                   = 10,       /* 26 Timer 1 Interrupt */
  TIMER2_IRQn                   = 11,       /* 27 Timer 1 Interrupt */
  TIMER3_IRQn                   = 12,       /* 28 Timer 1 Interrupt */
  UART0_RX_IRQn                 = 13,       /* 29 UART 0 Receive Interrupt */
  UART0_TX_IRQn                 = 14,       /* 30 UART 0 Transmit Interrupt */
  UART0_ERROR_IRQn              = 15,       /* 31 UART 0 Error Interrupt */
  UART1_RX_IRQn                 = 16,       /* 32 UART 1 Receive Interrupt */
  UART1_TX_IRQn                 = 17,       /* 33 UART 1 Transmit Interrupt */
  UART1_ERROR_IRQn              = 18,       /* 34 UART 1 Error Interrupt */
  I2C_IRQn                      = 19,       /* 35 I2C Interrupt */
  SPI0_IRQn                     = 20,       /* 36 SPI 0 Interrupt */
  SPI0_ERROR_IRQn               = 21,       /* 37 SPI 0 Error Interrupt */
  SPI1_IRQn                     = 22,       /* 38 SPI 1 Interrupt */
  SPI1_ERROR_IRQn               = 23,       /* 39 SPI 1 Error Interrupt */
  PCM_RX_IRQn                   = 24,       /* 40 PCM Receive Interrupt */
  PCM_TX_IRQn                   = 25,       /* 41 PCM Transmit Interrupt */
  PCM_ERROR_IRQn                = 26,       /* 42 PCM Error Interrupt */
  ADC_IRQn                      = 27,       /* 43 ADC Interrupt */
  DMA0_IRQn                     = 28,       /* 44 DMA Channel 0 Interrupt */
  DMA1_IRQn                     = 29,       /* 45 DMA Channel 1 Interrupt */
  DMA2_IRQn                     = 30,       /* 46 DMA Channel 2 Interrupt */
  DMA3_IRQn                     = 31,       /* 47 DMA Channel 3 Interrupt */
  USB_WAKEUP_IRQn               = 32,       /* 48 USB Wakeup Interrupt */
  USB_SDAV_IRQn                 = 33,       /* 49 USB Setup Data Available 
                                             *    Interrupt */
  USB_SOF_IRQn                  = 34,       /* 50 USB Start Of Frame 
                                             *    Interrupt */
  USB_SUTOK_IRQn                = 35,       /* 51 USB Setup Data Loading 
                                             *    Interrupt */
  USB_SUSPEND_IRQn              = 36,       /* 52 USB Suspend Interrupt */
  USB_RESET_IRQn                = 37,       /* 53 USB Reset Interrupt */
  USB_EP0IN_IRQn                = 38,       /* 54 USB EP0IN Interrupt */
  USB_EP0OUT_IRQn               = 39,       /* 55 USB EP0OUT Interrupt */
  USB_EP2IN_IRQn                = 40,       /* 56 USB EP2IN Interrupt */
  USB_EP3IN_IRQn                = 42,       /* 58 USB EP3IN Interrupt */
  USB_EP4OUT_IRQn               = 45,       /* 61 USB EP4OUT Interrupt */
  USB_EP5OUT_IRQn               = 47,       /* 63 USB EP5OUT Interrupt */
  ECC_CORRECTED_IRQn            = 48,       /* 64 ECC Error Corrected 
                                             *    Interrupt */
  GPIO_GP2_IRQn                 = 49,       /* 65 GPIO Interrupt 2 */
  GPIO_GP3_IRQn                 = 50,       /* 66 GPIO Interrupt 3 */
  GPIO_GP4_IRQn                 = 51,       /* 67 GPIO Interrupt 4 */
  GPIO_GP5_IRQn                 = 52,       /* 68 GPIO Interrupt 5 */
  GPIO_GP6_IRQn                 = 53,       /* 69 GPIO Interrupt 6 */
  GPIO_GP7_IRQn                 = 54,       /* 70 GPIO Interrupt 7 */
} IRQn_Type;

#endif /* Q32M210_VECTORS_H */

