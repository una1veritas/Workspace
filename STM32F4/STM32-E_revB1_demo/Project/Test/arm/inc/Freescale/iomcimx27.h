/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Freescale MCIX27
 **
 **    Used with ICCARM and AARM.
 **
 **    (c) Copyright IAR Systems 2003
 **
 **    $Revision: 50408 $
 **
 ***************************************************************************/

#ifndef __MCIX27_H
#define __MCIX27_H


#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ICCARM/AARM
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    MCIX27 SPECIAL FUNCTION REGISTERS
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

/* -------------------------------------------------------------------------*/
/*      PLL and Clock Controller Module Registers                           */
/* -------------------------------------------------------------------------*/
typedef struct {        /* Clock Source Control Register (0x10027000) */
__REG32 MPEN           : 1;     /* Bit  0       - MPLL Enable*/
__REG32 SPEN           : 1;     /* Bit  1       - Serial Peripheral PLL Enable*/
__REG32 FPM_EN         : 1;     /* Bit  2       - Frequency Premultiplier Enable*/
__REG32 OSC26M_DIS     : 1;     /* Bit  3       - Oscillator Disable*/
__REG32 OSC26M_DIV1P5  : 1;     /* Bit  4       - Oscillator 26M Divide*/
__REG32                : 3;     /* Bits 5  - 7  - Reserved*/
__REG32 AHB_DIV        : 2;     /* Bit  8  - 9  - Divider value for AHB clk*/
__REG32                : 2;     /* Bits 10 - 11 - Reserved*/
__REG32 ARM_DIV        : 2;     /* Bits 12 - 13 - Divider value for arm clk*/
__REG32                : 1;     /* Bits 14      - Reserved*/
__REG32 ARMSRC         : 1;     /* Bits 15      - selects the ARM clock source.*/
__REG32 MCU_SEL        : 1;     /* Bit  16      - MPLL Select*/
__REG32 SP_SEL         : 1;     /* Bit  17      - SPLL Select*/
__REG32 MPLL_RESTART   : 1;     /* Bit  18      - MPLL Restart—Restarts the MPLL at a new assigned frequency*/
__REG32 SPLL_RESTART   : 1;     /* Bit  19      - SPLL Restart—Restarts the SPLL at new assigned frequency.*/
__REG32 MSHC_SEL       : 1;     /* Bit  20      - MSHC CCLK Source Select—Selects the clock source to the MSHC divider (MSHC_DIV).*/
__REG32 H264_SEL       : 1;     /* Bit  21      - H264 CCLK Source Select—Selects the clock source to the H264 divider (H264_DIV).*/
__REG32 SSI1_SEL       : 1;     /* Bit  22      - SSI1 Baud Source Select—Selects the clock source to the SSI1 fractional divider (SSI1_DIV).*/
__REG32 SSI2_SEL       : 1;     /* Bit  23      - SSI2 Baud Source Select—Selects the clock source to the SSI2 fractional divider (SSI2_DIV).*/
__REG32 SD_CNT         : 2;     /* Bits 24 - 25 - Shut-Down Control*/
__REG32                : 2;     /* Bits 26 - 27 - Reserved*/
__REG32 USB_DIV        : 3;     /* Bits 28 - 30 - USB Clock Divider*/
__REG32                : 1;     /* Bits 31      - Reserved*/
} __cscr_bits;

typedef struct { /* MCU & System PLL Control Register 0 (0x10027004) */
                 /* Serial Peripheral PLL Control Register 0 (0x1002700C) reset (0x807F2065) */
__REG32 MFN   :10;     /* Bits 0  - 9  - Multiplication Factor (Numerator)*/
__REG32 MFI   : 4;     /* Bits 10 - 13 - Multiplication Factor (Integer)*/
__REG32       : 2;     /* Bits 14 - 15 - Reserved*/
__REG32 MFD   :10;     /* Bits 16 - 25 - Multiplication Factor (Denominator Part)*/
__REG32 PD    : 4;     /* Bits 26 - 29 - Predivider Factor applied to the PLL input frequency.(0-15)*/
__REG32       : 1;     /* Bit  30      - Reserved*/
__REG32 CPLM  : 1;     /* Bit  31      - Phase Lock Mode*/
} __mpctl0_bits;

typedef struct { /* MCU & System PLL Control Register 1 (0x10027008) */
__REG32       : 5;     /* Bits 0  - 4  - Reserved*/
__REG32 BRMO  : 2;     /* Bit  5  - 6  - Controls the BRM order.*/
__REG32       : 8;     /* Bits 7 - 14  - Reserved*/
__REG32 LF    : 1;     /* Bit  15      - Lock Flag - Indicates whether the System PLL is locked.*/
__REG32       :16;     /* Bits 16 - 31 - Reserved*/
} __mpctl1_bits;

typedef struct { /* Serial Peripheral PLL Control Register 1 (0x10027010) reset (0x00008000) */
__REG32       : 6;     /* Bits 0  - 5  - Reserved*/
__REG32 BRMO  : 1;     /* Bit  6       - Controls the BRM order.*/
__REG32       : 8;     /* Bits 7 - 14  - Reserved*/
__REG32 LF    : 1;     /* Bit  15      - Lock Flag - Indicates whether the System PLL is locked.*/
__REG32       :16;     /* Bits 16 - 31 - Reserved*/
} __spctl1_bits;

typedef struct { /* Oscillator 26M Control Register (0x10027014) */
__REG32              : 8;     /* Bits 0  - 7  - Reserved*/
__REG32 AGC          : 6;     /* Bits 8  - 13 - Automatic Gain Control*/
__REG32              : 2;     /* Bits 14 - 15 - Reserved*/
__REG32 OSC26M_PEAK  : 2;     /* Bits 16 - 17 - OSC26M_PEAK - the oscillator's amplitude*/
__REG32              :14;     /* Bits 16 - 31   - Reserved*/
} __osc26mctl_bits;

typedef struct { /* Peripheral Clock Divider Register 0 (0x10027018) */
__REG32 MSHCDIV      : 6;     /* Bits 0  - 5  - MSHC Clock Divider — Contains the 6-bit divider that produces the clock for the CCLK clock signal of the MSHC.*/
__REG32 NFCDIV       : 4;     /* Bits 6  - 9  - Nand Flash Controller Clock Divider*/
__REG32 H264DIV      : 6;     /* Bits 10 - 15 - H264 Clock Divider — Contains the 6-bit divider that produces the clock for the CCLK clock signal of the H264.*/
__REG32 SSI1DIV      : 6;     /* Bits 16 - 21 - SSI1 Baud Clock Divider*/
__REG32 CLKO_DIV     : 3;     /* Bits 22 - 24 - Clock Out Divider — Contains the 3-bit divider that divides output clocks to CLKO pin.*/
__REG32 CLKO_EN      : 1;     /* Bits 25      - Clock Out Divider — Contains the 3-bit divider that divides output clocks to CLKO pin.*/
__REG32 SSI2DIV      : 6;     /* Bits 26 - 31 - SSI2 Baud Clock Divider*/
} __pcdr0_bits;

typedef struct { /* Peripheral Clock Divider Register 1 (0x1002701C) */
__REG32 PERDIV1    : 6;     /* Bits 0  - 5  - Peripheral Clock Divider 1*/
__REG32            : 2;     /* Bits 6  - 7  - Reserved*/
__REG32 PERDIV2    : 6;     /* Bits 8  - 13 - Peripheral Clock Divider 2*/
__REG32            : 2;     /* Bits 14 - 15 - Reserved*/
__REG32 PERDIV3    : 6;     /* Bits 16 - 21 - Peripheral Clock Divider 3*/
__REG32            : 2;     /* Bits 22 - 23 - Reserved*/
__REG32 PERDIV4    : 6;     /* Bits 24 - 29 - Peripheral Clock Divider 4*/
__REG32            : 2;     /* Bits 30 - 31 - Reserved*/
} __pcdr1_bits;

typedef struct { /* Peripheral Clock Control Register 0 (0x10027020) */
__REG32 SSI2_EN         : 1;     /* Bit  0       - SSI2 IPG Clock Enable*/
__REG32 SSI1_EN         : 1;     /* Bit  1       - SSI1 IPG Clock Enable*/
__REG32 SLCDC_EN        : 1;     /* Bit  2       - SLCDC IPG Clock Enable*/
__REG32 SDHC3_EN        : 1;     /* Bit  3       - SDHC3 IPG Clock Enable*/
__REG32 SDHC2_EN        : 1;     /* Bit  4       - SDHC2 IPG Clock Enable*/
__REG32 SDHC1_EN        : 1;     /* Bit  5       - SDHC1 IPG Clock Enable*/
__REG32 SCC_EN          : 1;     /* Bit  6       - SCC IPG Clock Enable*/
__REG32 SAHARA_EN       : 1;     /* Bit  7       - SAHARA IPG Clock Enable*/
__REG32 RTIC_EN         : 1;     /* Bit  8       - RTIC IPG Clock Enable*/
__REG32 RTC_EN          : 1;     /* Bit  9       - RTC IPG Clock Enable*/
__REG32                 : 1;     /* Bits 10      - Reserved*/
__REG32 PWM_EN          : 1;     /* Bit  11      - PWM IPG Clock Enable*/
__REG32 OWIRE_EN        : 1;     /* Bit  12      - OWIRE IPG Clock Enable*/
__REG32 MSHC_EN         : 1;     /* Bit  13      - MSHC IPG Clock Enable*/
__REG32 LCDC_EN         : 1;     /* Bit  14      - LCDC IPG Clock Enable*/
__REG32 KPP_EN          : 1;     /* Bit  15      - KPP IPG Clock Enable*/
__REG32 IIM_EN          : 1;     /* Bit  16      - IIM IPG Clock Enable*/
__REG32 I2C2_EN         : 1;     /* Bit  17      - I2C2 IPG Clock Enable*/
__REG32 I2C1_EN         : 1;     /* Bit  18      - I2C1 IPG Clock Enable*/
__REG32 GPT6_EN         : 1;     /* Bit  19      - GPT6 IPG Clock Enable*/
__REG32 GPT5_EN         : 1;     /* Bit  20      - GPT5 IPG Clock Enable*/
__REG32 GPT4_EN         : 1;     /* Bit  21      - GPT4 IPG Clock Enable*/
__REG32 GPT3_EN         : 1;     /* Bit  22      - GPT3 IPG Clock Enable*/
__REG32 GPT2_EN         : 1;     /* Bit  23      - GPT2 IPG Clock Enable*/
__REG32 GPT1_EN         : 1;     /* Bit  24      - GPT1 IPG Clock Enable*/
__REG32 GPIO_EN         : 1;     /* Bit  25      - GPIO IPG Clock Enable*/
__REG32 FEC_EN          : 1;     /* Bit  26      - FEC IPG Clock Enable*/
__REG32 EMMA_EN         : 1;     /* Bit  27      - EMMA IPG Clock Enable*/
__REG32 DMA_EN          : 1;     /* Bit  28      - DMA IPG Clock Enable*/
__REG32 CSPI3_EN        : 1;     /* Bits 29      - CSPI3 IPG Clock Enable*/
__REG32 CSPI2_EN        : 1;     /* Bit  30      - CSPI2 IPG Clock Enable*/
__REG32 CSPI1_EN        : 1;     /* Bit  31      - CSPI1 IPG Clock Enable*/
} __pccr0_bits;

typedef struct { /* Peripheral Clock Control Register 1 (0x10027024) */
__REG32               : 2;     /* Bits 0  - 1  - Reserved*/
__REG32 MSHC_BAUDEN   : 1;     /* Bit  2       - MSHC BAUD Clock Enable*/
__REG32 NFC_BAUDEN    : 1;     /* Bit  3       - NFC BAUD Clock Enable*/
__REG32 SSI2_BAUDEN   : 1;     /* Bit  4       - SSI2 BAUD Clock Enable*/
__REG32 SSI1_BAUDEN   : 1;     /* Bit  5       - SSI1 BAUD Clock Enable*/
__REG32 H264_BAUDEN   : 1;     /* Bit  6       - H264 BAUD Clock Enable*/
__REG32 PERCLK4_EN    : 1;     /* Bit  7       - PERCLK4 Clock Enable*/
__REG32 PERCLK3_EN    : 1;     /* Bit  8       - PERCLK3 Clock Enable*/
__REG32 PERCLK2_EN    : 1;     /* Bit  9       - PERCLK2 Clock Enable*/
__REG32 PERCLK1_EN    : 1;     /* Bit 10       - PERCLK1 Clock Enable*/
__REG32 HCLK_USB      : 1;     /* Bit 11       - USB AHB Clock Enable*/
__REG32 HCLK_SLCDC    : 1;     /* Bit 12       - SLCDC AHB Clock Enable*/
__REG32 HCLK_SAHARA   : 1;     /* Bit 13       - SAHARA AHB Clock Enable*/
__REG32 HCLK_RTIC     : 1;     /* Bit 14       - RTIC AHB Clock Enable*/
__REG32 HCLK_LCDC     : 1;     /* Bit 15       - LCDC AHB Clock Enable*/
__REG32 HCLK_H264     : 1;     /* Bit 16       - H264 AHB Clock Enable*/
__REG32 HCLK_FEC      : 1;     /* Bit 17       - FEC AHB Clock Enable*/
__REG32 HCLK_EMMA     : 1;     /* Bit 18       - EMMA AHB Clock Enable*/
__REG32 HCLK_EMI      : 1;     /* Bit 19       - EMI AHB Clock Enable*/
__REG32 HCLK_DMA      : 1;     /* Bit 20       - DMA AHB Clock Enable*/
__REG32 HCLK_CSI      : 1;     /* Bit 21       - CSI AHB Clock Enable*/
__REG32 HCLK_BROM     : 1;     /* Bit 22       - BROM AHB Clock Enable*/
__REG32 HCLK_ATA      : 1;     /* Bit 23       - ATA AHB Clock Enable*/
__REG32 WDT_EN        : 1;     /* Bit 24       - WDT Clock Enable*/
__REG32 USB_EN        : 1;     /* Bit 25       - USB IPG Clock Enable*/
__REG32 UART6_EN      : 1;     /* Bit 26       - UART6 IPG Clock Enable*/
__REG32 UART5_EN      : 1;     /* Bit 27       - UART5 IPG Clock Enable*/
__REG32 UART4_EN      : 1;     /* Bit 28       - UART4 IPG Clock Enable*/
__REG32 UART3_EN      : 1;     /* Bit 29       - UART3 IPG Clock Enable*/
__REG32 UART2_EN      : 1;     /* Bit 30       - UART2 IPG Clock Enable*/
__REG32 UART1_EN      : 1;     /* Bit 31       - UART1 IPG Clock Enable*/
} __pccr1_bits;

typedef struct { /* Clock Control Status Register (0x10027028) */
__REG32 CLKO_SEL  : 5;     /* Bits 1  - 4  - CLKO Select*/
__REG32           : 3;     /* Bits 5  - 7  - Reserved*/
__REG32 CLKMODE   : 2;     /* Bit  8  - 9  - Determines the configuration of FPM, OSC26M and DPLL on the chip.*/
__REG32           : 5;     /* Bits 10 - 14 - Reserved*/
__REG32 _32K_SR   : 1;     /* Bit  15      - 32K Status Register*/
__REG32           :16;     /* Bits 16 - 31 - Reserved*/
} __ccsr_bits;

typedef struct { /* Wakeup Guard Mode Control Register (0x10027034) */
__REG32 WKGD_EN  : 1;     /* Bit  0       - Wakeup Guard Mode Enable*/
__REG32          :31;     /* Bits 1  - 31 - Reserved*/
} __wkgdctl_bits;

/* -------------------------------------------------------------------------*/
/*               System control registers                                   */
/* -------------------------------------------------------------------------*/
/* Chip ID Register */
typedef struct {
__REG32 MANUFACTURER_ID :12;
__REG32 PART_NUMBER     :16;
__REG32 VERSION_ID      : 4;
} __cid_bits;

typedef struct { /* Function Multiplexing Control Register (0x10027814) */
__REG32 SDCS0_SEL      : 1;     /* Bit  0       - SDRAM/SyncFlash Chip Select CS2/CSD0*/
__REG32 SDCS1_SEL      : 1;     /* Bit  1       - SDRAM/SyncFlash Chip Select CS3/CS1*/
__REG32 SLCDC_SEL      : 1;     /* Bit  2       - SLCDC Select*/
__REG32                : 1;     /* Bit  3       - Reserved*/
__REG32 NF_16BIT_SEL   : 1;     /* Bit  4       - Nand Flash 16-bit Select*/
__REG32 NF_FMS         : 1;     /* Bit  5       - Flash Memory Select*/
__REG32                : 2;     /* Bit  6  - 7  - Reserved*/
__REG32 IOIS16_CTL     : 1;     /* Bit  8       - IOIS16 Control*/
__REG32 PC_BVD2_CTL    : 1;     /* Bit  9       - PC_BVD2 Control*/
__REG32 PC_BVD1_CTL    : 1;     /* Bit  10      - PC_BVD1 Control*/
__REG32 PC_VS2_CTL     : 1;     /* Bit  11      - PC_VS2 Control*/
__REG32 PC_VS1_CTL     : 1;     /* Bit  12      - PC_VS1 Control*/
__REG32 PC_READY_CTL   : 1;     /* Bit  13      - PC_READY Control*/
__REG32 PC_WAIT_B_CTL  : 1;     /* Bit  14      - PC_WAIT_B Control*/
__REG32                : 1;     /* Bit  15      - Reserved*/
__REG32 KP_ROW6_CTL    : 1;     /* Bit  16      - Keypad Row 6 Control*/
__REG32 KP_ROW7_CTL    : 1;     /* Bit  17      - Keypad Row 7 Control*/
__REG32 KP_COL6_CTL    : 1;     /* Bit  18      - Keypad Column 6 Control*/
__REG32                : 5;     /* Bit  19 - 23 - Reserved*/
__REG32 UART4_RTS_CTL  : 1;     /* Bit  24      - UART4 RTS Control*/
__REG32 UART4_RXD_CTL  : 1;     /* Bit  25      - UART4 RXD Control*/
__REG32                : 6;     /* Bits 26 - 31 - Reserved*/
} __fmcr_bits;

typedef struct { /* Global Peripheral Control Register (0x10027818) */
__REG32 DDR_INPUT        : 1;     /* Bits 0        - Used to force input mode of DDR pads to CMOS input mode.*/
__REG32 CLK_DDR_MODE     : 1;     /* Bits 1        - CLK DDR MODE*/
__REG32 DDR_MODE         : 1;     /* Bits 2        - DDR Drive Strength Control*/
__REG32 CLOCK_GATING_EN  : 1;     /* Bit  3        - Clock Gating Enable*/
__REG32                  : 4;     /* Bits 4  - 7   - Reserved*/
__REG32 DMA_Burst_Override : 1;   /* Bits 8        - When this bit is set, the burst type of DMA will be forced to INCR4 or INCR8.*/
__REG32 PP_Burst_Override  : 1;   /* Bits 9        - When this bit is set, the burst type of EMMA PP will be forced to INCR4 or INCR8.*/
__REG32 USB_Burst_Override : 1;   /* Bits 10       - When this bit is set, the burst type of USB will be forced to INCR8. 0 Bypass. The burst type will not be forced.*/
__REG32 ETM9_PAD_EN      : 1;     /* Bits 11       - When this bit is set, pads for ETM9 are enabled.*/
__REG32                  : 4;     /* Bits 12 - 15  - Reserved*/
__REG32 BOOT             : 4;     /* Bits 16 - 19  - Boot Mode*/
__REG32                  :12;     /* Bits 20 - 31  - Reserved*/
} __gpcr_bits;

typedef struct { /* Well Bias Control Register (0x1002781C) */
__REG32 CRM_WBM       : 2;
__REG32 CRM_WBFA      : 2;
__REG32               : 4;
__REG32 CRM_SPA       : 4;
__REG32               : 4;
__REG32 CRM_WBM_EMI   : 2;
__REG32 CRM_WBFA_EMI  : 2;
__REG32               : 4;
__REG32 CRM_SPA_EMI   : 4;
__REG32               : 4;
} __wbcr_bits;

typedef struct { /* Driving Strength Control Register 1 (0x10027820) */
__REG32 DS_SLOW1  : 2;     /* Bits 0  - 1   - Driving Strength Slow I/O*/
__REG32 DS_SLOW2  : 2;     /* Bits 2  - 3   - Driving Strength Slow I/O*/
__REG32 DS_SLOW3  : 2;     /* Bits 4  - 5   - Driving Strength Slow I/O*/
__REG32 DS_SLOW4  : 2;     /* Bits 6  - 7   - Driving Strength Slow I/O*/
__REG32 DS_SLOW5  : 2;     /* Bits 8  - 9   - Driving Strength Slow I/O*/
__REG32 DS_SLOW6  : 2;     /* Bits 10 - 11  - Driving Strength Slow I/O*/
__REG32 DS_SLOW7  : 2;     /* Bits 12 - 13  - Driving Strength Slow I/O*/
__REG32 DS_SLOW8  : 2;     /* Bits 14 - 15  - Driving Strength Slow I/O*/
__REG32 DS_SLOW9  : 2;     /* Bits 16 - 17  - Driving Strength Slow I/O*/
__REG32 DS_SLOW10 : 2;     /* Bits 18 - 19  - Driving Strength Slow I/O*/
__REG32 DS_SLOW11 : 2;     /* Bits 20 - 21  - Driving Strength Slow I/O*/
__REG32           :10;     /* Bits 22 - 31  - Reserved*/
} __dscr1_bits;

typedef struct { /* Driving Strength Control Register 2 (0x10027824) */
__REG32 DS_FAST1   : 2;
__REG32 DS_FAST2   : 2;
__REG32 DS_FAST3   : 2;
__REG32 DS_FAST4   : 2;
__REG32 DS_FAST5   : 2;
__REG32 DS_FAST6   : 2;
__REG32 DS_FAST7   : 2;
__REG32 DS_FAST8   : 2;
__REG32 DS_FAST9   : 2;
__REG32 DS_FAST10  : 2;
__REG32 DS_FAST11  : 2;
__REG32 DS_FAST12  : 2;
__REG32 DS_FAST13  : 2;
__REG32 DS_FAST14  : 2;
__REG32 DS_FAST15  : 2;
__REG32 DS_FAST16  : 2;
} __dscr2_bits;

typedef struct { /* Driving Strength Control Register 3 (0x10027828)*/
__REG32 DS_FAST17  : 2;
__REG32 DS_FAST18  : 2;
__REG32 DS_FAST19  : 2;
__REG32 DS_FAST20  : 2;
__REG32 DS_FAST21  : 2;
__REG32 DS_FAST22  : 2;
__REG32 DS_FAST23  : 2;
__REG32 DS_FAST24  : 2;
__REG32 DS_FAST25  : 2;
__REG32 DS_FAST26  : 2;
__REG32 DS_FAST27  : 2;
__REG32 DS_FAST28  : 2;
__REG32 DS_FAST29  : 2;
__REG32 DS_FAST30  : 2;
__REG32 DS_FAST31  : 2;
__REG32 DS_FAST32  : 2;
} __dscr3_bits;

typedef struct { /* Driving Strength Control Register 4 (0x1002782C) */
__REG32 DS_FAST33  : 2;
__REG32 DS_FAST34  : 2;
__REG32 DS_FAST35  : 2;
__REG32 DS_FAST36  : 2;
__REG32 DS_FAST37  : 2;
__REG32 DS_FAST38  : 2;
__REG32 DS_FAST39  : 2;
__REG32 DS_FAST40  : 2;
__REG32 DS_FAST41  : 2;
__REG32 DS_FAST42  : 2;
__REG32            :12;
} __dscr4_bits;

typedef struct { /* Driving Strength Control Register 5 (0x10027830) */
__REG32 DS_FAST49      : 2;
__REG32 DS_FAST50      : 2;
__REG32 DS_FAST51      : 2;
__REG32 DS_FAST52      : 2;
__REG32 DS_FAST53      : 2;
__REG32 DS_FAST54      : 2;
__REG32 DS_FAST55      : 2;
__REG32 DS_FAST56      : 2;
__REG32 DS_FAST57      : 2;
__REG32 DS_FAST58      : 2;
__REG32 DS_FAST59      : 2;
__REG32 DS_FAST60      : 2;
__REG32 DS_FAST61      : 2;
__REG32 DS_FAST62      : 2;
__REG32 DS_FAST63      : 2;
__REG32 DS_FAST64      : 2;
} __dscr5_bits;

typedef struct { /* Driving Strength Control Register 6 (0x10027834) */
__REG32 DS_FAST65      : 2;
__REG32 DS_FAST66      : 2;
__REG32 DS_FAST67      : 2;
__REG32 DS_FAST68      : 2;
__REG32 DS_FAST69      : 2;
__REG32 DS_FAST70      : 2;
__REG32 DS_FAST71      : 2;
__REG32 DS_FAST72      : 2;
__REG32 DS_FAST73      : 2;
__REG32 DS_FAST74      : 2;
__REG32 DS_FAST75      : 2;
__REG32 DS_FAST76      : 2;
__REG32 DS_FAST77      : 2;
__REG32 DS_FAST78      : 2;
__REG32 DS_FAST79      : 2;
__REG32 DS_FAST80      : 2;
} __dscr6_bits;

typedef struct { /* Driving Strength Control Register 7 (0x10027838) */
__REG32 DS_FAST81 : 2;
__REG32 DS_FAST82 : 2;
__REG32 DS_FAST83 : 2;
__REG32 DS_FAST84 : 2;
__REG32 DS_FAST85 : 2;
__REG32 DS_FAST86 : 2;
__REG32 DS_FAST87 : 2;
__REG32 DS_FAST88 : 2;
__REG32 DS_FAST89 : 2;
__REG32 DS_FAST90 : 2;
__REG32 DS_FAST91 : 2;
__REG32 DS_FAST92 : 2;
__REG32 DS_FAST93 : 2;
__REG32 DS_FAST94 : 2;
__REG32 DS_FAST95 : 2;
__REG32           : 2;
} __dscr7_bits;

typedef struct { /* Driving Strength Control Register 8 (0x1002783C) Reset (0x00000000) */
__REG32 DS_FAST97  : 2;
__REG32 DS_FAST98  : 2;
__REG32 DS_FAST99  : 2;
__REG32 DS_FAST100 : 2;
__REG32 DS_FAST101 : 2;
__REG32 DS_FAST102 : 2;
__REG32 DS_FAST103 : 2;
__REG32 DS_FAST104 : 2;
__REG32 DS_FAST105 : 2;
__REG32 DS_FAST106 : 2;
__REG32 DS_FAST107 : 2;
__REG32 DS_FAST108 : 2;
__REG32 DS_FAST109 : 2;
__REG32 DS_FAST110 : 2;
__REG32 DS_FAST111 : 2;
__REG32            : 2;
} __dscr8_bits;

typedef struct { /* Driving Strength Control Register 9 (0x10027840) */
__REG32 DS_FAST113  : 2;
__REG32 DS_FAST114  : 2;
__REG32 DS_FAST115  : 2;
__REG32 DS_FAST116  : 2;
__REG32 DS_FAST117  : 2;
__REG32 DS_FAST118  : 2;
__REG32 DS_FAST119  : 2;
__REG32 DS_FAST120  : 2;
__REG32 DS_FAST121  : 2;
__REG32 DS_FAST122  : 2;
__REG32 DS_FAST123  : 2;
__REG32 DS_FAST124  : 2;
__REG32 DS_FAST125  : 2;
__REG32 DS_FAST126  : 2;
__REG32 DS_FAST127  : 2;
__REG32             : 2;
} __dscr9_bits;

typedef struct { /* Driving Strength Control Register 10 (0x10027844)*/
__REG32 DS_FAST129 : 2;
__REG32 DS_FAST130 : 2;
__REG32 DS_FAST131 : 2;
__REG32 DS_FAST132 : 2;
__REG32 DS_FAST133 : 2;
__REG32 DS_FAST134 : 2;
__REG32 DS_FAST135 : 2;
__REG32 DS_FAST136 : 2;
__REG32 DS_FAST137 : 2;
__REG32 DS_FAST138 : 2;
__REG32 DS_FAST139 : 2;
__REG32 DS_FAST140 : 2;
__REG32 DS_FAST141 : 2;
__REG32 DS_FAST142 : 2;
__REG32 DS_FAST143 : 2;
__REG32            : 2;
} __dscr10_bits;

typedef struct { /* Driving Strength Control Register 11 (0x10027848) */
__REG32 DS_FAST145  : 2;
__REG32 DS_FAST146  : 2;
__REG32 DS_FAST147  : 2;
__REG32 DS_FAST148  : 2;
__REG32 DS_FAST149  : 2;
__REG32 DS_FAST150  : 2;
__REG32 DS_FAST151  : 2;
__REG32 DS_FAST152  : 2;
__REG32 DS_FAST153  : 2;
__REG32 DS_FAST154  : 2;
__REG32 DS_FAST155  : 2;
__REG32 DS_FAST156  : 2;
__REG32 DS_FAST157  : 2;
__REG32 DS_FAST158  : 2;
__REG32 DS_FAST159  : 2;
__REG32 DS_FAST160  : 2;
} __dscr11_bits;

typedef struct { /* Driving Strength Control Register 12 (0x1002784C)*/
__REG32 DS_FAST161    : 2;
__REG32 DS_FAST162    : 2;
__REG32 DS_FAST163    : 2;
__REG32 DS_FAST164    : 2;
__REG32 DS_FAST165    : 2;
__REG32 DS_FAST166    : 2;
__REG32 DS_FAST167    : 2;
__REG32 DS_FAST168    : 2;
__REG32 DS_FAST169    : 2;
__REG32 DS_FAST170    : 2;
__REG32 DS_FAST171    : 2;
__REG32 DS_FAST172    : 2;
__REG32               : 8;
} __dscr12_bits;

/* Driving Strength Control Register 13 */
typedef struct {
__REG32 DS_FAST177   : 2;
__REG32 DS_FAST178   : 2;
__REG32 DS_FAST179   : 2;
__REG32 DS_FAST180   : 2;
__REG32 DS_FAST181   : 2;
__REG32 DS_FAST182   : 2;
__REG32 DS_FAST183   : 2;
__REG32 DS_FAST184   : 2;
__REG32 DS_FAST185   : 2;
__REG32 DS_FAST186   : 2;
__REG32 DS_FAST187   : 2;
__REG32 DS_FAST188   : 2;
__REG32              : 8;
} __dscr13_bits;

/* Pull Strength Control Register Description */
typedef struct {
__REG32 PUENCR0      : 2;
__REG32 PUENCR1      : 2;
__REG32 PUENCR2      : 2;
__REG32 PUENCR3      : 2;
__REG32 PUENCR4      : 2;
__REG32 PUENCR5      : 2;
__REG32 PUENCR6      : 2;
__REG32 PUENCR7      : 2;
__REG32              :16;
} __pscr_bits;

/* Priority Control and Select Register */
typedef struct {
__REG32 M0_HIGH_PRIORITY  : 1;
__REG32 M1_HIGH_PRIORITY  : 1;
__REG32 M2_HIGH_PRIORITY  : 1;
__REG32 M3_HIGH_PRIORITY  : 1;
__REG32 M4_HIGH_PRIORITY  : 1;
__REG32 M5_HIGH_PRIORITY  : 1;
__REG32                   :10;
__REG32 S0_AMPR_SEL       : 1;
__REG32 S1_AMPR_SEL       : 1;
__REG32 S2_AMPR_SEL       : 1;
__REG32 S3_AMPR_SEL       : 1;
__REG32                   :12;
} __pcsr_bits;

/* Power Management Control Regitser */
typedef struct {
__REG32 DPTEN       : 1;
__REG32 DIE         : 1;
__REG32 DIM         : 2;
__REG32 DRCE0       : 1;
__REG32 DRCE1       : 1;
__REG32 DRCE2       : 1;
__REG32 DRCE3       : 1;
__REG32 RCLKON      : 1;
__REG32 DCR         : 1;
__REG32             : 2;
__REG32 VSTBY       : 2;
__REG32             : 1;
__REG32 RVEN        : 1;
__REG32 REFCOUNTER  :11;
__REG32             : 1;
__REG32 LO          : 1;
__REG32 UP          : 1;
__REG32 EM          : 1;
__REG32 MC          : 1;
} __pmcr_bits;

/* DPTC Comparator Value Register x */
typedef struct {
__REG32 ELV         :10;
__REG32 LLV         :11;
__REG32 ULV         :11;
} __dcvr_bits;

/* PMIC Pad Control Register */
typedef struct {
__REG32 OE0         :1;
__REG32 DSE0        :2;
__REG32 PUE0        :1;
__REG32 PUS0        :2;
__REG32             :2;
__REG32 OE1         :1;
__REG32 DSE1        :2;
__REG32 PUE1        :1;
__REG32 PUS1        :2;
__REG32             :18;
} __ppcr_bits;

/* -------------------------------------------------------------------------*/
/*      GPIO Registers                                                      */
/* -------------------------------------------------------------------------*/
typedef struct { /* Structure for GPIO Register Type 1. */
__REG32 PIN0   : 2;     /* Bits 1-0*/
__REG32 PIN1   : 2;     /* Bits 3-2*/
__REG32 PIN2   : 2;     /* Bits 5-4*/
__REG32 PIN3   : 2;     /* Bits 7-6*/
__REG32 PIN4   : 2;     /* Bits 9-8*/
__REG32 PIN5   : 2;     /* Bits 11-10*/
__REG32 PIN6   : 2;     /* Bits 13-12*/
__REG32 PIN7   : 2;     /* Bits 15-14*/
__REG32 PIN8   : 2;     /* Bits 17-16*/
__REG32 PIN9   : 2;     /* Bits 19-18*/
__REG32 PIN10  : 2;     /* Bits 21-20*/
__REG32 PIN11  : 2;     /* Bits 23-22*/
__REG32 PIN12  : 2;     /* Bits 25-24*/
__REG32 PIN13  : 2;     /* Bits 27-26*/
__REG32 PIN14  : 2;     /* Bits 29-28*/
__REG32 PIN15  : 2;     /* Bits 31-30*/
} __port_reg_15_0_bits;

typedef struct { /* Structure for GPIO Register Type 2. */
__REG32 PIN16  : 2;     /* Bits 1-0*/
__REG32 PIN17  : 2;     /* Bits 3-2*/
__REG32 PIN18  : 2;     /* Bits 5-4*/
__REG32 PIN19  : 2;     /* Bits 7-6*/
__REG32 PIN20  : 2;     /* Bits 9-8*/
__REG32 PIN21  : 2;     /* Bits 11-10*/
__REG32 PIN22  : 2;     /* Bits 13-12*/
__REG32 PIN23  : 2;     /* Bits 15-14*/
__REG32 PIN24  : 2;     /* Bits 17-16*/
__REG32 PIN25  : 2;     /* Bits 19-18*/
__REG32 PIN26  : 2;     /* Bits 21-20*/
__REG32 PIN27  : 2;     /* Bits 23-22*/
__REG32 PIN28  : 2;     /* Bits 25-24*/
__REG32 PIN29  : 2;     /* Bits 27-26*/
__REG32 PIN30  : 2;     /* Bits 29-28*/
__REG32 PIN31  : 2;     /* Bits 31-30           - See table below*/
} __port_reg_31_16_bits;

typedef struct { /* Structure for GPIO Register Type 3.                                         */
__REG32 PIN0   : 1;     /* Bit 0*/
__REG32 PIN1   : 1;     /* Bit 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32 PIN24  : 1;     /* Bit 24*/
__REG32 PIN25  : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __port_reg_31_0_bits;

/* GPIO In Use Register A */
typedef struct {
__REG32 PIN0   : 1;     /* Bit 0*/
__REG32 PIN1   : 1;     /* Bit 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32 PIN24  : 1;     /* Bit 24*/
__REG32 PIN25  : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __pta_gius_bits;

typedef struct { /* GPIO In Use Register B */
__REG32        : 2;     /* Bit 0  - 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32 PIN24  : 1;     /* Bit 24*/
__REG32 PIN25  : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __ptb_gius_bits;

typedef struct { /* GPIO In Use Register C */
__REG32        : 5;     /* Bit 0  - 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32 PIN24  : 1;     /* Bit 24*/
__REG32 PIN25  : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __ptc_gius_bits;

typedef struct { /* GPIO In Use Register D */
__REG32 PIN0   : 1;     /* Bit 0*/
__REG32 PIN1   : 1;     /* Bit 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32 PIN24  : 1;     /* Bit 24*/
__REG32 PIN25  : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __ptd_gius_bits;

typedef struct { /* GPIO In Use Register E */
__REG32 PIN0   : 1;     /* Bit 0*/
__REG32 PIN1   : 1;     /* Bit 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32 PIN23  : 1;     /* Bit 23*/
__REG32        : 1;     /* Bit 24*/
__REG32        : 1;     /* Bit 25*/
__REG32 PIN26  : 1;     /* Bit 26*/
__REG32 PIN27  : 1;     /* Bit 27*/
__REG32 PIN28  : 1;     /* Bit 28*/
__REG32 PIN29  : 1;     /* Bit 29*/
__REG32 PIN30  : 1;     /* Bit 30*/
__REG32 PIN31  : 1;     /* Bit 31*/
} __pte_gius_bits;

typedef struct { /* GPIO In Use Register F */
__REG32 PIN0   : 1;     /* Bit 0*/
__REG32 PIN1   : 1;     /* Bit 1*/
__REG32 PIN2   : 1;     /* Bit 2*/
__REG32 PIN3   : 1;     /* Bit 3*/
__REG32 PIN4   : 1;     /* Bit 4*/
__REG32 PIN5   : 1;     /* Bit 5*/
__REG32 PIN6   : 1;     /* Bit 6*/
__REG32 PIN7   : 1;     /* Bit 7*/
__REG32 PIN8   : 1;     /* Bit 8*/
__REG32 PIN9   : 1;     /* Bit 9*/
__REG32 PIN10  : 1;     /* Bit 10*/
__REG32 PIN11  : 1;     /* Bit 11*/
__REG32 PIN12  : 1;     /* Bit 12*/
__REG32 PIN13  : 1;     /* Bit 13*/
__REG32 PIN14  : 1;     /* Bit 14*/
__REG32 PIN15  : 1;     /* Bit 15*/
__REG32 PIN16  : 1;     /* Bit 16*/
__REG32 PIN17  : 1;     /* Bit 17*/
__REG32 PIN18  : 1;     /* Bit 18*/
__REG32 PIN19  : 1;     /* Bit 19*/
__REG32 PIN20  : 1;     /* Bit 20*/
__REG32 PIN21  : 1;     /* Bit 21*/
__REG32 PIN22  : 1;     /* Bit 22*/
__REG32        : 9;     /* Bit 23 - 31*/
} __ptf_gius_bits;

/* Software Reset Register (SWR) */
typedef struct{
__REG32 SWR  : 1;     /* Bit 0        -   Software Reset (0 = No effect, 1 = GPIO circuitry for Port X Reset)*/
__REG32      :31;     /* Bits 31-1    -   Reserved, should read 0*/
} __swr_bits;

/* Port Interrupt Mask Register */
typedef struct{
__REG32 PTA  : 1;     /* Bit 0        -  Port A—The bit helps in masking the Port A interrupt. The bit clears during software reset of Port A.*/
__REG32 PTB  : 1;     /* Bit 1        -  Port B—The bit helps in masking the Port B interrupt. The bit clears during software reset of Port B.*/
__REG32 PTC  : 1;     /* Bit 2        -  Port C—The bit helps in masking the Port C interrupt. The bit clears during software reset of Port C.*/
__REG32 PTD  : 1;     /* Bit 3        -  Port D—The bit helps in masking the Port D interrupt. The bit clears during software reset of Port D.*/
__REG32 PTE  : 1;     /* Bit 4        -  Port E—The bit helps in masking the Port E interrupt. The bit clears during software reset of Port E.*/
__REG32 PTF  : 1;     /* Bit 5        -  Port F—The bit helps in masking the Port F interrupt. The bit clears during software reset of Port F.*/
__REG32      :26;     /* Bits 31 - 6  -  Reserved*/
} __pmask_bits;

/* -------------------------------------------------------------------------*/
/*               AITC registers                                             */
/* -------------------------------------------------------------------------*/
typedef struct {        /* Interrupt Control Register (0x10040000) */
__REG32          : 2;     /* Bits 0-1     - Reserved*/
__REG32 POINTER  :10;     /* Bits 2-11    - Interrupt Vector Table Pointer*/
__REG32          : 4;     /* Bits 12-15   - Reserved*/
__REG32 MD       : 1;     /* Bit 16       - Interrupt Vector Table Mode*/
__REG32          : 2;     /* Bits 17-18   - Reserved*/
__REG32 FIAD     : 1;     /* Bit 19       - Fast Interrupt Arbiter Disable*/
__REG32 NIAD     : 1;     /* Bit 20       - Normal Interrupt Arbiter Disable*/
__REG32 FIDIS    : 1;     /* Bit 21       - Fast Interrupt Disable*/
__REG32 NIDIS    : 1;     /* Bit 22       - Normal Interrupt Disable*/
__REG32          : 9;     /* Bits 23-31   - Reserved*/
} __intcntl_bits;

typedef struct {        /* Normal Interrupt Mask Register (0x10040004) Reset (0x0000001F)                       */
__REG32 NIMASK  : 5;     /* Bits 0-4     - Normal Interrupt Mask (0 = Disable priority level 0 ints, 1 = disable priority 1 and lower... 16+ = disable all interrupts)*/
__REG32         :27;     /* Bits 5-31    - Reserved*/
} __nimask_bits;

typedef struct {        /* Interrupt Enable Number Register (0x10040008) Reset (0x00000000)                     */
__REG32 ENNUM  : 6;     /* Bits 0-5     - Interrupt Enable Number - Enables/Disables the interrupt source associated with this value.*/
__REG32        :26;     /* Bits 6-31    - Reserved*/
} __intennum_bits;

typedef struct {        /* Interrupt Disable Number Register (0x1004000C) Reset (0x00000000)                    */
__REG32 DISNUM  : 6;     /* Bits 0-5     - Interrupt Disable Number - Enables/Disables the interrupt source associated with this value.*/
__REG32         :26;     /* Bits 6-31    - Reserved*/
} __intdisnum_bits;

typedef struct {        /* Interrupt Enable Register High (0x10040010) Reset (0x00000000)                       */
__REG32 INTENABLE32  : 1;     /* Bit  0           - Interrupt Enable*/
__REG32 INTENABLE33  : 1;     /* Bit  1           - Interrupt Enable*/
__REG32 INTENABLE34  : 1;     /* Bit  2           - Interrupt Enable*/
__REG32 INTENABLE35  : 1;     /* Bit  3           - Interrupt Enable*/
__REG32 INTENABLE36  : 1;     /* Bit  4           - Interrupt Enable*/
__REG32 INTENABLE37  : 1;     /* Bit  5           - Interrupt Enable*/
__REG32 INTENABLE38  : 1;     /* Bit  6           - Interrupt Enable*/
__REG32 INTENABLE39  : 1;     /* Bit  7           - Interrupt Enable*/
__REG32 INTENABLE40  : 1;     /* Bit  8           - Interrupt Enable*/
__REG32 INTENABLE41  : 1;     /* Bit  9           - Interrupt Enable*/
__REG32 INTENABLE42  : 1;     /* Bit  10          - Interrupt Enable*/
__REG32 INTENABLE43  : 1;     /* Bit  11          - Interrupt Enable*/
__REG32 INTENABLE44  : 1;     /* Bit  12          - Interrupt Enable*/
__REG32 INTENABLE45  : 1;     /* Bit  13          - Interrupt Enable*/
__REG32 INTENABLE46  : 1;     /* Bit  14          - Interrupt Enable*/
__REG32 INTENABLE47  : 1;     /* Bit  15          - Interrupt Enable*/
__REG32 INTENABLE48  : 1;     /* Bit  16          - Interrupt Enable*/
__REG32 INTENABLE49  : 1;     /* Bit  17          - Interrupt Enable*/
__REG32 INTENABLE50  : 1;     /* Bit  18          - Interrupt Enable*/
__REG32 INTENABLE51  : 1;     /* Bit  19          - Interrupt Enable*/
__REG32 INTENABLE52  : 1;     /* Bit  20          - Interrupt Enable*/
__REG32 INTENABLE53  : 1;     /* Bit  21          - Interrupt Enable*/
__REG32 INTENABLE54  : 1;     /* Bit  22          - Interrupt Enable*/
__REG32 INTENABLE55  : 1;     /* Bit  23          - Interrupt Enable*/
__REG32 INTENABLE56  : 1;     /* Bit  24          - Interrupt Enable*/
__REG32 INTENABLE57  : 1;     /* Bit  25          - Interrupt Enable*/
__REG32 INTENABLE58  : 1;     /* Bit  26          - Interrupt Enable*/
__REG32 INTENABLE59  : 1;     /* Bit  27          - Interrupt Enable*/
__REG32 INTENABLE60  : 1;     /* Bit  28          - Interrupt Enable*/
__REG32 INTENABLE61  : 1;     /* Bit  29          - Interrupt Enable*/
__REG32 INTENABLE62  : 1;     /* Bit  30          - Interrupt Enable*/
__REG32 INTENABLE63  : 1;     /* Bit  31          - Interrupt Enable*/
} __intenableh_bits;

typedef struct {        /* Interrupt Enable Register Low (0x10040014) Reset (0x00000000)                        */
__REG32 INTENABLE0   : 1;     /* Bit  0           - Interrupt Enable*/
__REG32 INTENABLE1   : 1;     /* Bit  1           - Interrupt Enable*/
__REG32 INTENABLE2   : 1;     /* Bit  2           - Interrupt Enable*/
__REG32 INTENABLE3   : 1;     /* Bit  3           - Interrupt Enable*/
__REG32 INTENABLE4   : 1;     /* Bit  4           - Interrupt Enable*/
__REG32 INTENABLE5   : 1;     /* Bit  5           - Interrupt Enable*/
__REG32 INTENABLE6   : 1;     /* Bit  6           - Interrupt Enable*/
__REG32 INTENABLE7   : 1;     /* Bit  7           - Interrupt Enable*/
__REG32 INTENABLE8   : 1;     /* Bit  8           - Interrupt Enable*/
__REG32 INTENABLE9   : 1;     /* Bit  9           - Interrupt Enable*/
__REG32 INTENABLE10  : 1;     /* Bit  10          - Interrupt Enable*/
__REG32 INTENABLE11  : 1;     /* Bit  11          - Interrupt Enable*/
__REG32 INTENABLE12  : 1;     /* Bit  12          - Interrupt Enable*/
__REG32 INTENABLE13  : 1;     /* Bit  13          - Interrupt Enable*/
__REG32 INTENABLE14  : 1;     /* Bit  14          - Interrupt Enable*/
__REG32 INTENABLE15  : 1;     /* Bit  15          - Interrupt Enable*/
__REG32 INTENABLE16  : 1;     /* Bit  16          - Interrupt Enable*/
__REG32 INTENABLE17  : 1;     /* Bit  17          - Interrupt Enable*/
__REG32 INTENABLE18  : 1;     /* Bit  18          - Interrupt Enable*/
__REG32 INTENABLE19  : 1;     /* Bit  19          - Interrupt Enable*/
__REG32 INTENABLE20  : 1;     /* Bit  20          - Interrupt Enable*/
__REG32 INTENABLE21  : 1;     /* Bit  21          - Interrupt Enable*/
__REG32 INTENABLE22  : 1;     /* Bit  22          - Interrupt Enable*/
__REG32 INTENABLE23  : 1;     /* Bit  23          - Interrupt Enable*/
__REG32 INTENABLE24  : 1;     /* Bit  24          - Interrupt Enable*/
__REG32 INTENABLE25  : 1;     /* Bit  25          - Interrupt Enable*/
__REG32 INTENABLE26  : 1;     /* Bit  26          - Interrupt Enable*/
__REG32 INTENABLE27  : 1;     /* Bit  27          - Interrupt Enable*/
__REG32 INTENABLE28  : 1;     /* Bit  28          - Interrupt Enable*/
__REG32 INTENABLE29  : 1;     /* Bit  29          - Interrupt Enable*/
__REG32 INTENABLE30  : 1;     /* Bit  30          - Interrupt Enable*/
__REG32 INTENABLE31  : 1;     /* Bit  31          - Interrupt Enable*/
} __intenablel_bits;

typedef struct {        /* Interrupt Type Register High (0x10040018) Reset (0x00000000)                 */
__REG32 INTTYPE32  : 1;     /* Bit  0         - Interrupt Enable*/
__REG32 INTTYPE33  : 1;     /* Bit  1         - Interrupt Enable*/
__REG32 INTTYPE34  : 1;     /* Bit  2         - Interrupt Enable*/
__REG32 INTTYPE35  : 1;     /* Bit  3         - Interrupt Enable*/
__REG32 INTTYPE36  : 1;     /* Bit  4         - Interrupt Enable*/
__REG32 INTTYPE37  : 1;     /* Bit  5         - Interrupt Enable*/
__REG32 INTTYPE38  : 1;     /* Bit  6         - Interrupt Enable*/
__REG32 INTTYPE39  : 1;     /* Bit  7         - Interrupt Enable*/
__REG32 INTTYPE40  : 1;     /* Bit  8         - Interrupt Enable*/
__REG32 INTTYPE41  : 1;     /* Bit  9         - Interrupt Enable*/
__REG32 INTTYPE42  : 1;     /* Bit  10        - Interrupt Enable*/
__REG32 INTTYPE43  : 1;     /* Bit  11        - Interrupt Enable*/
__REG32 INTTYPE44  : 1;     /* Bit  12        - Interrupt Enable*/
__REG32 INTTYPE45  : 1;     /* Bit  13        - Interrupt Enable*/
__REG32 INTTYPE46  : 1;     /* Bit  14        - Interrupt Enable*/
__REG32 INTTYPE47  : 1;     /* Bit  15        - Interrupt Enable*/
__REG32 INTTYPE48  : 1;     /* Bit  16        - Interrupt Enable*/
__REG32 INTTYPE49  : 1;     /* Bit  17        - Interrupt Enable*/
__REG32 INTTYPE50  : 1;     /* Bit  18        - Interrupt Enable*/
__REG32 INTTYPE51  : 1;     /* Bit  19        - Interrupt Enable*/
__REG32 INTTYPE52  : 1;     /* Bit  20        - Interrupt Enable*/
__REG32 INTTYPE53  : 1;     /* Bit  21        - Interrupt Enable*/
__REG32 INTTYPE54  : 1;     /* Bit  22        - Interrupt Enable*/
__REG32 INTTYPE55  : 1;     /* Bit  23        - Interrupt Enable*/
__REG32 INTTYPE56  : 1;     /* Bit  24        - Interrupt Enable*/
__REG32 INTTYPE57  : 1;     /* Bit  25        - Interrupt Enable*/
__REG32 INTTYPE58  : 1;     /* Bit  26        - Interrupt Enable*/
__REG32 INTTYPE59  : 1;     /* Bit  27        - Interrupt Enable*/
__REG32 INTTYPE60  : 1;     /* Bit  28        - Interrupt Enable*/
__REG32 INTTYPE61  : 1;     /* Bit  29        - Interrupt Enable*/
__REG32 INTTYPE62  : 1;     /* Bit  30        - Interrupt Enable*/
__REG32 INTTYPE63  : 1;     /* Bit  31        - Interrupt Enable*/
} __inttypeh_bits;

typedef struct {        /* Interrupt Enable Register Low (0x1004001C) Reset (0x00000000)                        */
__REG32 INTTYPE0   : 1;     /* Bit  0         - Interrupt Enable*/
__REG32 INTTYPE1   : 1;     /* Bit  1         - Interrupt Enable*/
__REG32 INTTYPE2   : 1;     /* Bit  2         - Interrupt Enable*/
__REG32 INTTYPE3   : 1;     /* Bit  3         - Interrupt Enable*/
__REG32 INTTYPE4   : 1;     /* Bit  4         - Interrupt Enable*/
__REG32 INTTYPE5   : 1;     /* Bit  5         - Interrupt Enable*/
__REG32 INTTYPE6   : 1;     /* Bit  6         - Interrupt Enable*/
__REG32 INTTYPE7   : 1;     /* Bit  7         - Interrupt Enable*/
__REG32 INTTYPE8   : 1;     /* Bit  8         - Interrupt Enable*/
__REG32 INTTYPE9   : 1;     /* Bit  9         - Interrupt Enable*/
__REG32 INTTYPE10  : 1;     /* Bit  10        - Interrupt Enable*/
__REG32 INTTYPE11  : 1;     /* Bit  11        - Interrupt Enable*/
__REG32 INTTYPE12  : 1;     /* Bit  12        - Interrupt Enable*/
__REG32 INTTYPE13  : 1;     /* Bit  13        - Interrupt Enable*/
__REG32 INTTYPE14  : 1;     /* Bit  14        - Interrupt Enable*/
__REG32 INTTYPE15  : 1;     /* Bit  15        - Interrupt Enable*/
__REG32 INTTYPE16  : 1;     /* Bit  16        - Interrupt Enable*/
__REG32 INTTYPE17  : 1;     /* Bit  17        - Interrupt Enable*/
__REG32 INTTYPE18  : 1;     /* Bit  18        - Interrupt Enable*/
__REG32 INTTYPE19  : 1;     /* Bit  19        - Interrupt Enable*/
__REG32 INTTYPE20  : 1;     /* Bit  20        - Interrupt Enable*/
__REG32 INTTYPE21  : 1;     /* Bit  21        - Interrupt Enable*/
__REG32 INTTYPE22  : 1;     /* Bit  22        - Interrupt Enable*/
__REG32 INTTYPE23  : 1;     /* Bit  23        - Interrupt Enable*/
__REG32 INTTYPE24  : 1;     /* Bit  24        - Interrupt Enable*/
__REG32 INTTYPE25  : 1;     /* Bit  25        - Interrupt Enable*/
__REG32 INTTYPE26  : 1;     /* Bit  26        - Interrupt Enable*/
__REG32 INTTYPE27  : 1;     /* Bit  27        - Interrupt Enable*/
__REG32 INTTYPE28  : 1;     /* Bit  28        - Interrupt Enable*/
__REG32 INTTYPE29  : 1;     /* Bit  29        - Interrupt Enable*/
__REG32 INTTYPE30  : 1;     /* Bit  30        - Interrupt Enable*/
__REG32 INTTYPE31  : 1;     /* Bit  31        - Interrupt Enable*/
} __inttypel_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 7 (0x10040020) Reset (0x00000000)                   */
__REG32 NIPR56  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR57  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR58  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR59  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR60  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR61  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR62  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR63  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority7_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 6 (0x10040024) Reset (0x00000000)                   */
__REG32 NIPR48  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR49  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR50  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR51  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR52  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR53  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR54  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR55  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority6_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 5 (0x10040028) Reset (0x00000000)                   */
__REG32 NIPR40  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR41  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR42  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR43  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR44  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR45  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR46  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR47  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority5_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 4 (0x1004002C) Reset (0x00000000)                   */
__REG32 NIPR32  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR33  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR34  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR35  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR36  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR37  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR38  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR39  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority4_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 3 (0x10040030) Reset (0x00000000)                   */
__REG32 NIPR24  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR25  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR26  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR27  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR28  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR29  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR30  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR31  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority3_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 2 (0x10040034) Reset (0x00000000)                   */
__REG32 NIPR16  : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR17  : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR18  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR19  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR20  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR21  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR22  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR23  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority2_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 1 (0x10040038) Reset (0x00000000)                   */
__REG32 NIPR8   : 4;     /* Bits 0-3     - Normal Interrupt Priority Level*/
__REG32 NIPR9   : 4;     /* Bits 4-7     - Normal Interrupt Priority Level*/
__REG32 NIPR10  : 4;     /* Bits 8-11    - Normal Interrupt Priority Level*/
__REG32 NIPR11  : 4;     /* Bits 12-15   - Normal Interrupt Priority Level*/
__REG32 NIPR12  : 4;     /* Bits 16-19   - Normal Interrupt Priority Level*/
__REG32 NIPR13  : 4;     /* Bits 20-23   - Normal Interrupt Priority Level*/
__REG32 NIPR14  : 4;     /* Bits 24-27   - Normal Interrupt Priority Level*/
__REG32 NIPR15  : 4;     /* Bits 28-31   - Normal Interrupt Priority Level*/
} __nipriority1_bits;

typedef struct {        /* Normal Interrupt Priority Level Register 0 (0x1004003C) Reset (0x00000000)                   */
__REG32 NIPR0  : 4;     /* Bits 0  - 3  - Normal Interrupt Priority Level*/
__REG32 NIPR1  : 4;     /* Bits 4  - 7  - Normal Interrupt Priority Level*/
__REG32 NIPR2  : 4;     /* Bits 8  - 11 - Normal Interrupt Priority Level*/
__REG32 NIPR3  : 4;     /* Bits 12 - 15 - Normal Interrupt Priority Level*/
__REG32 NIPR4  : 4;     /* Bits 16 - 19 - Normal Interrupt Priority Level*/
__REG32 NIPR5  : 4;     /* Bits 20 - 23 - Normal Interrupt Priority Level*/
__REG32 NIPR6  : 4;     /* Bits 24 - 27 - Normal Interrupt Priority Level*/
__REG32 NIPR7  : 4;     /* Bits 28 - 31 - Normal Interrupt Priority Level*/
} __nipriority0_bits;

typedef struct {        /* Normal Interrupt Vector and Status Register (0x10040040) Reset (0xFFFFFFFF)  */
__REG32 NIPRILVL  :16;     /* Bits 0  - 15 - Normal Interrupt Priority Level - Indicates the priority level of the highest priority normal interrupt.*/
__REG32 NIVECTOR  :16;     /* Bits 16 - 31 - Normal Interrupt Vector - Indicates vector index for the highest pending normal interrupt.*/
} __nivecsr_bits;

typedef struct {        /* Interrupt Source Register High (0x10040048) Reset (0x00000000)       */
__REG32 INTIN32  : 1;     /* Bit  0       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN33  : 1;     /* Bit  1       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN34  : 1;     /* Bit  2       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN35  : 1;     /* Bit  3       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN36  : 1;     /* Bit  4       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN37  : 1;     /* Bit  5       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN38  : 1;     /* Bit  6       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN39  : 1;     /* Bit  7       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN40  : 1;     /* Bit  8       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN41  : 1;     /* Bit  9       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN42  : 1;     /* Bit  10      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN43  : 1;     /* Bit  11      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN44  : 1;     /* Bit  12      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN45  : 1;     /* Bit  13      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN46  : 1;     /* Bit  14      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN47  : 1;     /* Bit  15      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN48  : 1;     /* Bit  16      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN49  : 1;     /* Bit  17      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN50  : 1;     /* Bit  18      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN51  : 1;     /* Bit  19      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN52  : 1;     /* Bit  20      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN53  : 1;     /* Bit  21      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN54  : 1;     /* Bit  22      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN55  : 1;     /* Bit  23      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN56  : 1;     /* Bit  24      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN57  : 1;     /* Bit  25      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN58  : 1;     /* Bit  26      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN59  : 1;     /* Bit  27      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN60  : 1;     /* Bit  28      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN61  : 1;     /* Bit  29      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN62  : 1;     /* Bit  30      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN63  : 1;     /* Bit  31      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
} __intsrch_bits;

typedef struct {        /* Interrupt Source Register Low (0x1004004C) Reset (0x00000000)        */
__REG32 INTIN0   : 1;     /* Bit  0       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN1   : 1;     /* Bit  1       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN2   : 1;     /* Bit  2       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN3   : 1;     /* Bit  3       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN4   : 1;     /* Bit  4       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN5   : 1;     /* Bit  5       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN6   : 1;     /* Bit  6       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN7   : 1;     /* Bit  7       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN8   : 1;     /* Bit  8       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN9   : 1;     /* Bit  9       - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN10  : 1;     /* Bit  10      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN11  : 1;     /* Bit  11      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN12  : 1;     /* Bit  12      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN13  : 1;     /* Bit  13      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN14  : 1;     /* Bit  14      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN15  : 1;     /* Bit  15      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN16  : 1;     /* Bit  16      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN17  : 1;     /* Bit  17      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN18  : 1;     /* Bit  18      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN19  : 1;     /* Bit  19      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN20  : 1;     /* Bit  20      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN21  : 1;     /* Bit  21      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN22  : 1;     /* Bit  22      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN23  : 1;     /* Bit  23      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN24  : 1;     /* Bit  24      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN25  : 1;     /* Bit  25      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN26  : 1;     /* Bit  26      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN27  : 1;     /* Bit  27      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN28  : 1;     /* Bit  28      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN29  : 1;     /* Bit  29      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN30  : 1;     /* Bit  30      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
__REG32 INTIN31  : 1;     /* Bit  31      - Interrupt Source (0 = Interrupt source negated, 1 = Interrupt source asserted)*/
} __intsrcl_bits;

typedef struct {        /* Interrupt Force Register High (0x10040050) Reset (0x00000000)        */
__REG32 FORCE32  : 1;     /* Bit  0       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE33  : 1;     /* Bit  1       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE34  : 1;     /* Bit  2       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE35  : 1;     /* Bit  3       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE36  : 1;     /* Bit  4       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE37  : 1;     /* Bit  5       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE38  : 1;     /* Bit  6       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE39  : 1;     /* Bit  7       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE40  : 1;     /* Bit  8       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE41  : 1;     /* Bit  9       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE42  : 1;     /* Bit  10      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE43  : 1;     /* Bit  11      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE44  : 1;     /* Bit  12      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE45  : 1;     /* Bit  13      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE46  : 1;     /* Bit  14      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE47  : 1;     /* Bit  15      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE48  : 1;     /* Bit  16      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE49  : 1;     /* Bit  17      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE50  : 1;     /* Bit  18      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE51  : 1;     /* Bit  19      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE52  : 1;     /* Bit  20      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE53  : 1;     /* Bit  21      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE54  : 1;     /* Bit  22      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE55  : 1;     /* Bit  23      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE56  : 1;     /* Bit  24      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE57  : 1;     /* Bit  25      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE58  : 1;     /* Bit  26      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE59  : 1;     /* Bit  27      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE60  : 1;     /* Bit  28      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE61  : 1;     /* Bit  29      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE62  : 1;     /* Bit  30      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE63  : 1;     /* Bit  31      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
} __intfrch_bits;

typedef struct {        /* Interrupt Force Register Low (0x10040054) Reset (0x00000000) */
__REG32 FORCE0   : 1;     /* Bit  0       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE1   : 1;     /* Bit  1       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE2   : 1;     /* Bit  2       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE3   : 1;     /* Bit  3       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE4   : 1;     /* Bit  4       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE5   : 1;     /* Bit  5       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE6   : 1;     /* Bit  6       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE7   : 1;     /* Bit  7       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE8   : 1;     /* Bit  8       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE9   : 1;     /* Bit  9       - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE10  : 1;     /* Bit  10      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE11  : 1;     /* Bit  11      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE12  : 1;     /* Bit  12      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE13  : 1;     /* Bit  13      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE14  : 1;     /* Bit  14      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE15  : 1;     /* Bit  15      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE16  : 1;     /* Bit  16      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE17  : 1;     /* Bit  17      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE18  : 1;     /* Bit  18      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE19  : 1;     /* Bit  19      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE20  : 1;     /* Bit  20      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE21  : 1;     /* Bit  21      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE22  : 1;     /* Bit  22      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE23  : 1;     /* Bit  23      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE24  : 1;     /* Bit  24      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE25  : 1;     /* Bit  25      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE26  : 1;     /* Bit  26      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE27  : 1;     /* Bit  27      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE28  : 1;     /* Bit  28      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE29  : 1;     /* Bit  29      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE30  : 1;     /* Bit  30      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
__REG32 FORCE31  : 1;     /* Bit  31      - Interrupt Source Force Request - Writing a 1 Forces a request for the corresponding interrupt source.*/
} __intfrcl_bits;

typedef struct {        /* Normal Interrupt Pending Register High (0x10040058) Reset (0x00000000)       */
__REG32 NIPEND32  : 1;     /* Bit  0       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND33  : 1;     /* Bit  1       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND34  : 1;     /* Bit  2       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND35  : 1;     /* Bit  3       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND36  : 1;     /* Bit  4       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND37  : 1;     /* Bit  5       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND38  : 1;     /* Bit  6       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND39  : 1;     /* Bit  7       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND40  : 1;     /* Bit  8       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND41  : 1;     /* Bit  9       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND42  : 1;     /* Bit  10      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND43  : 1;     /* Bit  11      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND44  : 1;     /* Bit  12      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND45  : 1;     /* Bit  13      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND46  : 1;     /* Bit  14      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND47  : 1;     /* Bit  15      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND48  : 1;     /* Bit  16      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND49  : 1;     /* Bit  17      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND50  : 1;     /* Bit  18      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND51  : 1;     /* Bit  19      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND52  : 1;     /* Bit  20      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND53  : 1;     /* Bit  21      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND54  : 1;     /* Bit  22      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND55  : 1;     /* Bit  23      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND56  : 1;     /* Bit  24      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND57  : 1;     /* Bit  25      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND58  : 1;     /* Bit  26      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND59  : 1;     /* Bit  27      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND60  : 1;     /* Bit  28      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND61  : 1;     /* Bit  29      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND62  : 1;     /* Bit  30      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND63  : 1;     /* Bit  31      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
} __nipndh_bits;

typedef struct {        /* Normal Interrupt Pending Register Low (0x1004005C) Reset (0x00000000)        */
__REG32 NIPEND0   : 1;     /* Bit  0       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND1   : 1;     /* Bit  1       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND2   : 1;     /* Bit  2       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND3   : 1;     /* Bit  3       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND4   : 1;     /* Bit  4       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND5   : 1;     /* Bit  5       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND6   : 1;     /* Bit  6       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND7   : 1;     /* Bit  7       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND8   : 1;     /* Bit  8       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND9   : 1;     /* Bit  9       - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND10  : 1;     /* Bit  10      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND11  : 1;     /* Bit  11      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND12  : 1;     /* Bit  12      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND13  : 1;     /* Bit  13      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND14  : 1;     /* Bit  14      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND15  : 1;     /* Bit  15      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND16  : 1;     /* Bit  16      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND17  : 1;     /* Bit  17      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND18  : 1;     /* Bit  18      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND19  : 1;     /* Bit  19      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND20  : 1;     /* Bit  20      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND21  : 1;     /* Bit  21      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND22  : 1;     /* Bit  22      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND23  : 1;     /* Bit  23      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND24  : 1;     /* Bit  24      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND25  : 1;     /* Bit  25      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND26  : 1;     /* Bit  26      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND27  : 1;     /* Bit  27      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND28  : 1;     /* Bit  28      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND29  : 1;     /* Bit  29      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND30  : 1;     /* Bit  30      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
__REG32 NIPEND31  : 1;     /* Bit  31      - Normal Interrupt Pending Bit - (0=No int pending, 1 = Normal Interrupt request pending)*/
} __nipndl_bits;

typedef struct {        /* Fast Interrupt Pending Register High (0x10040060) Reset (0x00000000) */
__REG32 FIPEND32  : 1;     /* Bit  0       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND33  : 1;     /* Bit  1       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND34  : 1;     /* Bit  2       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND35  : 1;     /* Bit  3       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND36  : 1;     /* Bit  4       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND37  : 1;     /* Bit  5       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND38  : 1;     /* Bit  6       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND39  : 1;     /* Bit  7       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND40  : 1;     /* Bit  8       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND41  : 1;     /* Bit  9       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND42  : 1;     /* Bit  10      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND43  : 1;     /* Bit  11      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND44  : 1;     /* Bit  12      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND45  : 1;     /* Bit  13      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND46  : 1;     /* Bit  14      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND47  : 1;     /* Bit  15      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND48  : 1;     /* Bit  16      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND49  : 1;     /* Bit  17      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND50  : 1;     /* Bit  18      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND51  : 1;     /* Bit  19      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND52  : 1;     /* Bit  20      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND53  : 1;     /* Bit  21      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND54  : 1;     /* Bit  22      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND55  : 1;     /* Bit  23      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND56  : 1;     /* Bit  24      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND57  : 1;     /* Bit  25      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND58  : 1;     /* Bit  26      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND59  : 1;     /* Bit  27      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND60  : 1;     /* Bit  28      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND61  : 1;     /* Bit  29      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND62  : 1;     /* Bit  30      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND63  : 1;     /* Bit  31      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
} __fipndh_bits;

typedef struct {        /* Fast Interrupt Pending Register Low (0x10040064) Reset (0x00000000)  */
__REG32 FIPEND0   : 1;     /* Bit  0       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND1   : 1;     /* Bit  1       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND2   : 1;     /* Bit  2       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND3   : 1;     /* Bit  3       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND4   : 1;     /* Bit  4       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND5   : 1;     /* Bit  5       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND6   : 1;     /* Bit  6       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND7   : 1;     /* Bit  7       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND8   : 1;     /* Bit  8       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND9   : 1;     /* Bit  9       - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND10  : 1;     /* Bit  10      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND11  : 1;     /* Bit  11      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND12  : 1;     /* Bit  12      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND13  : 1;     /* Bit  13      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND14  : 1;     /* Bit  14      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND15  : 1;     /* Bit  15      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND16  : 1;     /* Bit  16      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND17  : 1;     /* Bit  17      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND18  : 1;     /* Bit  18      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND19  : 1;     /* Bit  19      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND20  : 1;     /* Bit  20      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND21  : 1;     /* Bit  21      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND22  : 1;     /* Bit  22      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND23  : 1;     /* Bit  23      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND24  : 1;     /* Bit  24      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND25  : 1;     /* Bit  25      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND26  : 1;     /* Bit  26      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND27  : 1;     /* Bit  27      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND28  : 1;     /* Bit  28      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND29  : 1;     /* Bit  29      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND30  : 1;     /* Bit  30      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
__REG32 FIPEND31  : 1;     /* Bit  31      - Fast Interrupt Pending Bit - (0=No int pending, 1 = Fast Interrupt request pending)*/
} __fipndl_bits;

/* -------------------------------------------------------------------------*/
/*      Security Controller                                                 */
/* -------------------------------------------------------------------------*/
/* SCM Red Start Address Register */
typedef struct{
__REG32 RED_START  : 7;
__REG32            :25;
} __scm_red_start_bits;

/* SCM Black Start Address Register */
typedef struct{
__REG32 BLACK_START  : 7;
__REG32              :25;
} __scm_black_start_bits;

/* SCM Length Register */
typedef struct{
__REG32 LENGTH  : 7;
__REG32         :25;
} __scm_length_bits;

/* SCM Control Register */
typedef struct{
__REG32 E_D    : 1;
__REG32 E_C    : 1;
__REG32 START  : 1;
__REG32        :29;
} __scm_control_bits;

/* SCM Status Register */
typedef struct{
__REG32 BUSY              : 1;
__REG32 CLR_MEM           : 1;
__REG32 ENCR              : 1;
__REG32 BLOCK_ACCESS      : 1;
__REG32 CLR_FAIL          : 1;
__REG32 BAD_KEY           : 1;
__REG32 INTERN_ERROR      : 1;
__REG32 SECRET_KEY        : 1;
__REG32 INT_STATUS        : 1;
__REG32 CLR_COMP          : 1;
__REG32 ENCR_COMP         : 1;
__REG32 BLOCK_ACCESS_REM  : 1;
__REG32 LENGTH_ERROR      : 1;
__REG32                   :19;
} __scm_status_bits;

/* SCM Error Status Register */
typedef struct{
__REG32 BUSY              : 1;
__REG32 CLR_MEM           : 1;
__REG32 ENCR              : 1;
__REG32 BLOCK_ACCESS      : 1;
__REG32 CLR_FAIL          : 1;
__REG32 BAD_KEY           : 1;
__REG32 INTERN_ERROR      : 1;
__REG32 DEFAULT_KEY       : 1;
__REG32 USER_ACCESS       : 1;
__REG32 ILLEGAL_ADDRESS   : 1;
__REG32 BYTEA_CCESS       : 1;
__REG32 UNALIGN_ACCESS    : 1;
__REG32 ILLEGAL_MASTER    : 1;
__REG32 CACHEABLE_ACCESS  : 1;
__REG32                   :18;
} __scm_error_bits;

/* SCM Interrupt Control Register */
typedef struct{
__REG32 INTR_MASK : 1;
__REG32 CLR_INT   : 1;
__REG32 CLR_MEM   : 1;
__REG32           :29;
} __scm_int_ctrl_bits;

/* SCM Configuration Register */
typedef struct{
__REG32 BLOCK_SIZE  : 7;
__REG32 RED_SIZE    :10;
__REG32 BLACK_SIZE  :10;
__REG32 VERSION_ID  : 5;
} __scm_cfg_bits;

/* SMN Status Register */
typedef struct{
__REG32 STATE     : 5;
__REG32 BI        : 1;
__REG32 ZF        : 1;
__REG32 DA        : 1;
__REG32 SPE       : 1;
__REG32 ASCE      : 1;
__REG32 BBE       : 1;
__REG32 PCE       : 1;
__REG32 TE        : 1;
__REG32 SA        : 1;
__REG32 SMNI      : 1;
__REG32 SCME      : 1;
__REG32 IA        : 1;
__REG32 BK        : 1;
__REG32 DK        : 1;
__REG32 UA        : 1;
__REG32 IAD       : 1;
__REG32 BA        : 1;
__REG32 UAA       : 1;
__REG32 SE        : 1;
__REG32 IM        : 1;
__REG32 CA        : 1;
__REG32 VID       : 6;
} __smn_status_bits;

/* SMN Command Register (R/W, Supervisor) */
typedef struct{
__REG32 SA  : 1;
__REG32 EI  : 1;
__REG32 CBB : 1;
__REG32 CI  : 1;
__REG32     :28;
} __smn_command_bits;

/* SMN Sequence Start Register */
typedef struct{
__REG32 START_VALUE  :16;
__REG32              :16;
} __smn_ssr_bits;

/* SMN Sequence End Register */
typedef struct{
__REG32 END_VALUE  :16;
__REG32            :16;
} __smn_ser_bits;

/* SMN Sequence Check Register */
typedef struct{
__REG32 CHECK_VALUE  :16;
__REG32              :16;
} __smn_scr_bits;

/* SMN Bit Count Register */
typedef struct{
__REG32 BITCNT  :11;
__REG32         :21;
} __smn_bcr_bits;

/* SMN Bit Bank Increment Size Register */
typedef struct{
__REG32 INC_SIZE  :11;
__REG32           :21;
} __smn_bbisr_bits;

/* SMN Bit Bank Decrement Register */
typedef struct{
__REG32 DEC_AMNT :11;
__REG32          :21;
} __smn_bbdr_bits;

/* SMN Compare Size Register */
typedef struct{
__REG32 SIZE  : 6;
__REG32       :26;
} __smn_csr_bits;

/* SMN Timer Control Register */
typedef struct{
__REG32 ST  : 1;
__REG32 LD  : 1;
__REG32     :30;
} __smn_tcr_bits;

/* SMN Debug Detector Register */
typedef struct{
__REG32 D1  : 1;
__REG32 D2  : 1;
__REG32 D3  : 1;
__REG32 D4  : 1;
__REG32 D5  : 1;
__REG32 D6  : 1;
__REG32 D7  : 1;
__REG32 D8  : 1;
__REG32 D9  : 1;
__REG32 D10 : 1;
__REG32 D11 : 1;
__REG32 D12 : 1;
__REG32     :20;
} __smn_ddr_bits;

/* -------------------------------------------------------------------------*/
/*      SAHARA                                                            */
/* -------------------------------------------------------------------------*/
/* Control Register */
typedef struct{
__REG32 MESS_BYTE_SWAP      : 1;
__REG32 MESS_HALFWORD_SWAP  : 1;
__REG32 EXT_MEM_CFG         : 2;
__REG32 INT_ENA             : 1;
__REG32 HIGH_ASSURE         : 1;
__REG32 HIGH_ASSURE_DIS     : 1;
__REG32 RING_AUTO_RESEED    : 1;
__REG32                     : 8;
__REG32 MAX_BURST           : 8;
__REG32 THROTTLE            : 8;
} __sahara_ctrl_bits;

/* Command Register */
typedef struct{
__REG32 SW_RESET            : 8;
__REG32 CLR_INT             : 1;
__REG32 CLR_ERROR           : 1;
__REG32 SINGLE_STEP         : 1;
__REG32                     : 5;
__REG32 CH_MODE             : 3;
__REG32                     :13;
} __sahara_cmd_bits;

/* Status Register */
typedef struct{
__REG32 STATE               : 3;
__REG32 DAR_FULL            : 1;
__REG32 ERROR               : 1;
__REG32 SECURE              : 1;
__REG32 FAIL                : 1;
__REG32 INIT                : 1;
__REG32 RNG_RESEED_REQ      : 1;
__REG32 ACTIVE              : 3;
__REG32                     : 4;
__REG32 MODE                : 3;
__REG32                     : 5;
__REG32 INTERNAL_STATE      : 8;
} __sahara_sta_bits;

/* Error Status Register */
typedef struct{
__REG32 ERROR_SRC           : 4;
__REG32                     : 4;
__REG32 DMA_ERR_DIRECT      : 1;
__REG32 DMA_SIZE_ERR        : 2;
__REG32                     : 1;
__REG32 DMA_ERR_SRC         : 4;
__REG32 CHA_ERR_SRC         :12;
__REG32 CHA_ERR             : 2;
__REG32                     : 2;
} __sahara_err_sta_bits;

/* Buffer Level Register */
typedef struct{
__REG32 IN_LEVEL            : 8;
__REG32 OUT_LEVEL           : 8;
__REG32                     :16;
} __sahara_buff_level_bits;

/* Internal Flow Control Register */
typedef struct{
__REG32 OUT_BUF_MUX         : 1;
__REG32                     :31;
} __sahara_int_flow_ctrl_bits;

/* SKHA - Symmetric Key Hardware Accelerator Modes */
typedef struct{
__REG32 ALGORITHM           : 2;
__REG32 ENC_DEC             : 1;
__REG32 CIPHER_MODE         : 2;
__REG32 NO_PERM             : 1;
__REG32                     : 2;
__REG32 DIS_KEY_PARITY      : 1;
__REG32 CRT_MOD             : 4;
__REG32                     :19;
} __skha_mode_dbg_bits;

/* MDHA - Message Digest Hardware Accelerator Modes */
typedef struct{
__REG32 ALG                 : 2;
__REG32 PDATA               : 1;
__REG32 HMAC                : 1;
__REG32                     : 1;
__REG32 INIT                : 1;
__REG32 IPAD                : 1;
__REG32 OPAD                : 1;
__REG32 SWAP                : 1;
__REG32 MAC_FULL            : 1;
__REG32 SSL                 : 1;
__REG32                     :21;
} __mdha_mode_dbg_bits;

/* RNG - Random Number Generator Modes */
typedef struct{
__REG32 EXC_MODE            :16;
__REG32                     :16;
} __rng_mode_dbg_bits;

/* -------------------------------------------------------------------------*/
/*      Run-Time Integrity Checker (RTIC)                                   */
/* -------------------------------------------------------------------------*/
/* RTIC Status Register */
typedef struct{
__REG32 BUSY            : 1;
__REG32 HASH_DONE       : 1;
__REG32 HASH_ERR        : 1;
__REG32 WD_ERR          : 1;
__REG32 MEMORY_INT_STA  : 4;
__REG32 ADDRESS_ERROR   : 4;
__REG32 LENGTH_ERROR    : 4;
__REG32 DBG_STA         :16;
} __rticsr_bits;

/* RTIC Command Register */
typedef struct{
__REG32 CLR_IRQ       : 1;
__REG32 SW_RST        : 1;
__REG32 HASH_ONCE     : 1;
__REG32 RUN_TIME_CHK  : 1;
__REG32               :28;
} __rticmd_bits;

/* RTIC Control Register */
typedef struct{
__REG32 IRQ_EN            : 1;
__REG32 DMA_BURST_SIZ     : 3;
__REG32 HASH_ONCE_MEM_EN  : 4;
__REG32 RUNTIME_MEM_EN    : 4;
__REG32                   :20;
} __rticcntlr_bits;

/* RTIC DMA Throttle/Bus Duty Cycle Register */
typedef struct{
__REG32 DELAY  :21;
__REG32        :11;
} __rtictr_bits;

/* RTIC Watchdog Timeout Register */
typedef struct{
__REG32 DMA_WDOG  :21;
__REG32           :11;
} __rticwr_bits;

/* -------------------------------------------------------------------------*/
/*      IC Identification (IIM)                                                           */
/* -------------------------------------------------------------------------*/
/* Status Register (STAT) */
typedef struct {
__REG8  SNSD  : 1;
__REG8  PRGD  : 1;
__REG8        : 5;
__REG8  BUSY  : 1;
} __iim_stat_bits;

/* Status IRQ Mask (STATM) */
typedef struct {
__REG8  SNSD_M  : 1;
__REG8  PRGD_M  : 1;
__REG8          : 6;
} __iim_statm_bits;

/* Module Errors Register (ERR) */
typedef struct {
__REG8            : 1;
__REG8  PARITYE   : 1;
__REG8  SNSE      : 1;
__REG8  WLRE      : 1;
__REG8  RPE       : 1;
__REG8  OPE       : 1;
__REG8  WPE       : 1;
__REG8  PRGE      : 1;
} __iim_err_bits;

/* Error IRQ Mask Register (EMASK) */
typedef struct {
__REG8            : 1;
__REG8  PARITYE_M : 1;
__REG8  SNSE_M    : 1;
__REG8  WLRE_M    : 1;
__REG8  RPE_M     : 1;
__REG8  OPE_M     : 1;
__REG8  WPE_M     : 1;
__REG8  PRGE_M    : 1;
} __iim_emask_bits;

/* Fuse Control Register (FCTL) */
typedef struct {
__REG8  PRG         : 1;
__REG8  ESNS_1      : 1;
__REG8  ESNS_0      : 1;
__REG8  ESNS_N      : 1;
__REG8  PRG_LENGTH  : 3;
__REG8  DPC         : 1;
} __iim_fctl_bits;

/* Upper Address (UA) */
typedef struct {
__REG8  A           : 6;
__REG8              : 2;
} __iim_ua_bits;

/* Product Revision (PREV) */
typedef struct {
__REG8  PROD_VT     : 3;
__REG8  PROD_REV    : 5;
} __iim_prev_bits;

/* Software-Controllable Signals Register 0 (SCS0) */
typedef struct {
__REG8  SCS         : 7;
__REG8  LOCK        : 1;
} __iim_scs_bits;

/* Fuse Bank 0 Access Protection Register — (FBAC0) */
typedef struct {
__REG8  SAHARA_LOCK : 1;
__REG8              : 2;
__REG8  FBESP       : 1;
__REG8  FBSP        : 1;
__REG8  FBRP        : 1;
__REG8  FBOP        : 1;
__REG8  FBWP        : 1;
} __iim_fbac0_bits;

/* Word1 of Fusebank0 */
typedef struct {
__REG8  HAB_CUS       : 5;
__REG8  SCC_EN        : 1;
__REG8  JTAG_DISABLE  : 1;
__REG8  BOOT_INT      : 1;
} __iim_fb0_word1_bits;

/* Word2 of Fusebank0 */
typedef struct {
__REG8  HAB_TYPE      : 4;
__REG8  HAB_SRS       : 3;
__REG8  SHW_EN        : 1;
} __iim_fb0_word2_bits;

/* Word3 of Fusebank0 */
typedef struct {
__REG8  MSHC_DIS      : 1;
__REG8  CPFA          : 1;
__REG8  CPSPA         : 4;
__REG8  SAHARA_EN     : 1;
__REG8  C_M_DISABLE   : 1;
} __iim_fb0_word3_bits;

/* Word4 of Fusebank0 */
typedef struct {
__REG8  CLK_A926_CTRL : 1;
__REG8  MEM_A926_CTRL : 1;
__REG8  CLK_SCC_CTRL  : 1;
__REG8  MEM_SCC_CTRL  : 1;
__REG8  VER_ID        : 4;
} __iim_fb0_word4_bits;

/* Fuse Bank 1Access Protection Register — (FBAC1) */
typedef struct {
__REG8  MAC_ADDR_LOCK : 1;
__REG8                : 2;
__REG8  FBESP         : 1;
__REG8  FBSP          : 1;
__REG8  FBRP          : 1;
__REG8  FBOP          : 1;
__REG8  FBWP          : 1;
} __iim_fbac1_bits;

/* -------------------------------------------------------------------------*/
/*      M3IF                                                                */
/* -------------------------------------------------------------------------*/
/* M3IF Control Register (M3IFCTL) */
typedef struct {
__REG32 MRRP     : 8;
__REG32 MLSD     : 3;
__REG32 MLSD_EN  : 1;
__REG32          :19;
__REG32 SDA      : 1;
} __m3ifctl_bits;

/* M3IF Snooping Configuration Register 0 (M3IFSCFG0) */
typedef struct {
__REG32 SE       : 1;
__REG32 SWSZ     : 4;
__REG32          : 6;
__REG32 SWBA     :21;
} __m3ifscfg0_bits;

/* M3IF Snooping Configuration Register 1 (M3IFSCFG1) */
typedef struct {
__REG32 SSE0_0   : 1;
__REG32 SSE0_1   : 1;
__REG32 SSE0_2   : 1;
__REG32 SSE0_3   : 1;
__REG32 SSE0_4   : 1;
__REG32 SSE0_5   : 1;
__REG32 SSE0_6   : 1;
__REG32 SSE0_7   : 1;
__REG32 SSE0_8   : 1;
__REG32 SSE0_9   : 1;
__REG32 SSE0_10  : 1;
__REG32 SSE0_11  : 1;
__REG32 SSE0_12  : 1;
__REG32 SSE0_13  : 1;
__REG32 SSE0_14  : 1;
__REG32 SSE0_15  : 1;
__REG32 SSE0_16  : 1;
__REG32 SSE0_17  : 1;
__REG32 SSE0_18  : 1;
__REG32 SSE0_19  : 1;
__REG32 SSE0_20  : 1;
__REG32 SSE0_21  : 1;
__REG32 SSE0_22  : 1;
__REG32 SSE0_23  : 1;
__REG32 SSE0_24  : 1;
__REG32 SSE0_25  : 1;
__REG32 SSE0_26  : 1;
__REG32 SSE0_27  : 1;
__REG32 SSE0_28  : 1;
__REG32 SSE0_29  : 1;
__REG32 SSE0_30  : 1;
__REG32 SSE0_31  : 1;
} __m3ifscfg1_bits;

/* M3IF Snooping Configuration Register 2 (M3IFSCFG2) */
typedef struct {
__REG32 SSE1_0   : 1;
__REG32 SSE1_1   : 1;
__REG32 SSE1_2   : 1;
__REG32 SSE1_3   : 1;
__REG32 SSE1_4   : 1;
__REG32 SSE1_5   : 1;
__REG32 SSE1_6   : 1;
__REG32 SSE1_7   : 1;
__REG32 SSE1_8   : 1;
__REG32 SSE1_9   : 1;
__REG32 SSE1_10  : 1;
__REG32 SSE1_11  : 1;
__REG32 SSE1_12  : 1;
__REG32 SSE1_13  : 1;
__REG32 SSE1_14  : 1;
__REG32 SSE1_15  : 1;
__REG32 SSE1_16  : 1;
__REG32 SSE1_17  : 1;
__REG32 SSE1_18  : 1;
__REG32 SSE1_19  : 1;
__REG32 SSE1_20  : 1;
__REG32 SSE1_21  : 1;
__REG32 SSE1_22  : 1;
__REG32 SSE1_23  : 1;
__REG32 SSE1_24  : 1;
__REG32 SSE1_25  : 1;
__REG32 SSE1_26  : 1;
__REG32 SSE1_27  : 1;
__REG32 SSE1_28  : 1;
__REG32 SSE1_29  : 1;
__REG32 SSE1_30  : 1;
__REG32 SSE1_31  : 1;
} __m3ifscfg2_bits;

/* M3IF Snooping Status Register 0 (M3IFSSR0) */
typedef struct {
__REG32 SSS0_0   : 1;
__REG32 SSS0_1   : 1;
__REG32 SSS0_2   : 1;
__REG32 SSS0_3   : 1;
__REG32 SSS0_4   : 1;
__REG32 SSS0_5   : 1;
__REG32 SSS0_6   : 1;
__REG32 SSS0_7   : 1;
__REG32 SSS0_8   : 1;
__REG32 SSS0_9   : 1;
__REG32 SSS0_10  : 1;
__REG32 SSS0_11  : 1;
__REG32 SSS0_12  : 1;
__REG32 SSS0_13  : 1;
__REG32 SSS0_14  : 1;
__REG32 SSS0_15  : 1;
__REG32 SSS0_16  : 1;
__REG32 SSS0_17  : 1;
__REG32 SSS0_18  : 1;
__REG32 SSS0_19  : 1;
__REG32 SSS0_20  : 1;
__REG32 SSS0_21  : 1;
__REG32 SSS0_22  : 1;
__REG32 SSS0_23  : 1;
__REG32 SSS0_24  : 1;
__REG32 SSS0_25  : 1;
__REG32 SSS0_26  : 1;
__REG32 SSS0_27  : 1;
__REG32 SSS0_28  : 1;
__REG32 SSS0_29  : 1;
__REG32 SSS0_30  : 1;
__REG32 SSS0_31  : 1;
} __m3ifssr0_bits;

/* M3IF Snooping Status Register 1 (M3IFSSR1) */
typedef struct {
__REG32 SSS1_0   : 1;
__REG32 SSS1_1   : 1;
__REG32 SSS1_2   : 1;
__REG32 SSS1_3   : 1;
__REG32 SSS1_4   : 1;
__REG32 SSS1_5   : 1;
__REG32 SSS1_6   : 1;
__REG32 SSS1_7   : 1;
__REG32 SSS1_8   : 1;
__REG32 SSS1_9   : 1;
__REG32 SSS1_10  : 1;
__REG32 SSS1_11  : 1;
__REG32 SSS1_12  : 1;
__REG32 SSS1_13  : 1;
__REG32 SSS1_14  : 1;
__REG32 SSS1_15  : 1;
__REG32 SSS1_16  : 1;
__REG32 SSS1_17  : 1;
__REG32 SSS1_18  : 1;
__REG32 SSS1_19  : 1;
__REG32 SSS1_20  : 1;
__REG32 SSS1_21  : 1;
__REG32 SSS1_22  : 1;
__REG32 SSS1_23  : 1;
__REG32 SSS1_24  : 1;
__REG32 SSS1_25  : 1;
__REG32 SSS1_26  : 1;
__REG32 SSS1_27  : 1;
__REG32 SSS1_28  : 1;
__REG32 SSS1_29  : 1;
__REG32 SSS1_30  : 1;
__REG32 SSS1_31  : 1;
} __m3ifssr1_bits;

/* M3IF Master Lock WEIM CSx Register (M3IFMLWEx) */
typedef struct {
__REG32 MLGE     : 3;
__REG32 MLGE_EN  : 1;
__REG32          :27;
__REG32 WEMA     : 1;
} __m3ifmlwe_bits;

/* -------------------------------------------------------------------------*/
/*      WEIM                                                            */
/* -------------------------------------------------------------------------*/
/* Chip Select x Upper Control Register (CSCRxU) */
typedef struct {
__REG32 EDC      : 4;
__REG32 WWS      : 3;
__REG32 EW       : 1;
__REG32 WSC      : 6;
__REG32 CNC      : 2;
__REG32 DOL      : 4;
__REG32 SYNC     : 1;
__REG32 PME      : 1;
__REG32 PSZ      : 2;
__REG32 BCS      : 4;
__REG32 BCD      : 2;
__REG32 WP       : 1;
__REG32 SP       : 1;
} __cscru_bits;

/* Chip Select x Lower Control Register (CSCRxL) */
typedef struct {
__REG32 CSEN     : 1;
__REG32 WRAP     : 1;
__REG32 CRE      : 1;
__REG32 PSR      : 1;
__REG32 CSN      : 4;
__REG32 DSZ      : 3;
__REG32 EBC      : 1;
__REG32 CSA      : 4;
__REG32 EBWN     : 4;
__REG32 EBWA     : 4;
__REG32 OEN      : 4;
__REG32 OEA      : 4;
} __cscrl_bits;

/* Chip Select x Additional Control Register (CSCRxA) */
typedef struct {
__REG32 FCE      : 1;
__REG32 CNC2     : 1;
__REG32 AGE      : 1;
__REG32 WWU      : 1;
__REG32 DCT      : 2;
__REG32 DWW      : 2;
__REG32 LBA      : 2;
__REG32 LBN      : 3;
__REG32 LAH      : 2;
__REG32 MUM      : 1;
__REG32 RWN      : 4;
__REG32 RWA      : 4;
__REG32 EBRN     : 4;
__REG32 EBRA     : 4;
} __cscra_bits;

/* WEIM Configuration Register (WCR) */
typedef struct {
__REG32 MAS      : 1;
__REG32          : 1;
__REG32 BCM      : 1;
__REG32          : 5;
__REG32 AUS0     : 1;
__REG32 AUS1     : 1;
__REG32 AUS2     : 1;
__REG32 AUS3     : 1;
__REG32 AUS4     : 1;
__REG32 AUS5     : 1;
__REG32 ECP0     : 1;
__REG32 ECP1     : 1;
__REG32 ECP2     : 1;
__REG32 ECP3     : 1;
__REG32 ECP4     : 1;
__REG32 ECP5     : 1;
__REG32          :12;
} __weim_wcr_bits;

/* -------------------------------------------------------------------------*/
/*      ESDTCL                                                              */
/* -------------------------------------------------------------------------*/
/* Enhanced SDRAM Control Register (ESDCTL0/1) */
typedef struct {
__REG32 PRCT     : 6;
__REG32          : 1;
__REG32 BL       : 1;
__REG32 FP       : 1;
__REG32          : 1;
__REG32 PWDT     : 2;
__REG32          : 1;
__REG32 SREFR    : 3;
__REG32 DSIZ     : 2;
__REG32          : 2;
__REG32 COL      : 2;
__REG32          : 2;
__REG32 ROW      : 3;
__REG32 SP       : 1;
__REG32 SMODE    : 3;
__REG32 SDE      : 1;
} __esdctl_bits;

/* Enhanced SDRAM Configurations Registers (ESDCFG0 /ESDCFG1) */
typedef struct {
__REG32 TRC      : 4;
__REG32 TRCD     : 3;
__REG32          : 1;
__REG32 TCAS     : 2;
__REG32 TRRD     : 2;
__REG32 TRAS     : 3;
__REG32 TWR      : 1;
__REG32 TMRD     : 2;
__REG32 TRP      : 2;
__REG32 TWTR     : 1;
__REG32 TXP      : 2;
__REG32          : 9;
} __esdcfg_bits;

/* Enhanced SDRAM Miscellaneous Register (ESDMISC) */
typedef struct {
__REG32             : 1;
__REG32 RST         : 1;
__REG32 MDDREN      : 1;
__REG32 MDDR_DL_RST : 1;
__REG32 MDDR_MDIS   : 1;
__REG32 LHD         : 1;
__REG32 MA10_SHARE  : 1;
__REG32             :24;
__REG32 SDRAMRDY    : 1;
} __esdmisc_bits;

/* MDDR Delay Line 1–5 Configuration Debug Register */
typedef struct {
__REG32 DLY_REG     :11;
__REG32             : 5;
__REG32 DLY_CORR    :11;
__REG32             : 4;
__REG32 SEL_DLY_REG : 1;
} __esdcdly_bits;

/* MDDR Delay Line Cycle Length Debug Register */
typedef struct {
__REG32 DLY_CYCLE_LENGTH  :11;
__REG32                   :21;
} __esdcdlyl_bits;

/* -------------------------------------------------------------------------*/
/*               NAND Flash Controller (NFC)                                */
/* -------------------------------------------------------------------------*/
/* NAND FC Buffer Size Register */
typedef struct{
__REG16 BUFSIZE  : 4;
__REG16          :12;
} __nfc_bufsiz_bits;

/* Buffer Number for Page Data Transfer To/From Flash Memory */
typedef struct{
__REG16 RBA  : 4;
__REG16      :12;
} __nfc_rba_bits;

/* NFC Internal Buffer Lock Control */
typedef struct{
__REG16 BLS  : 2;
__REG16      :14;
} __nfc_iblc_bits;

/* NFC Controller Status/Result of Flash Operation */
typedef struct{
__REG16 ERS  : 2;
__REG16 ERM  : 2;
__REG16      :12;
} __ecc_srr_bits;

/* NFC ECC Error Position of Main Area Data Error */
typedef union{
  /*ECC_RSLT_MA_8*/
  struct {
__REG16 ECC8_RESULT2   : 3;
__REG16 ECC8_RESULT1   : 9;
__REG16                : 4;
  };
  /*ECC_RSLT_MA_16*/
  struct {
__REG16 ECC16_RESULT2  : 4;
__REG16 ECC16_RESULT1  : 8;
__REG16                : 4;
  };
} __ecc_rslt_ma_bits;

/* NFC ECC Error Position of Spare Area Data Error */
typedef union{
  /*ECC_RSLT_SA_8*/
  struct {
__REG16 ECC8_RESULT3   : 3;
__REG16 ECC8_RESULT4   : 2;
__REG16                :11;
  };
  /*ECC_RSLT_SA */
  /*ECC_RSLT_SA_16*/
  struct {
__REG16 ECC16_RESULT3  : 4;
__REG16 ECC16_RESULT4  : 1;
__REG16                :11;
  };
} __ecc_rslt_sa_bits;

/* NFC Nand Flash Write Protection */
typedef struct{
__REG16 WPC  : 3;
__REG16      :13;
} __nf_wr_prot_bits;

/* NFC NAND Flash Write Protection Status */
typedef struct{
__REG16 LTS  : 1;
__REG16 LS   : 1;
__REG16 US   : 1;
__REG16      :13;
} __nf_wr_prot_sta_bits;

/* NFC NAND Flash Operation Configuration 1 */
typedef struct{
__REG16           : 2;
__REG16 SP_EN     : 1;
__REG16 ECC_EN    : 1;
__REG16 INT_MASK  : 1;
__REG16 NF_BIG    : 1;
__REG16 NFC_RST   : 1;
__REG16 NF_CE     : 1;
__REG16           : 8;
} __nand_fc1_bits;

/* NFC NAND Flash Operation Configuration 2 */
typedef struct{
__REG16 FCMD  : 1;
__REG16 FADD  : 1;
__REG16 FDI   : 1;
__REG16 FDO   : 3;
__REG16       : 9;
__REG16 INT   : 1;
} __nand_fc2_bits;

/* -------------------------------------------------------------------------*/
/*      PCMCIA Registers                                                       */
/* -------------------------------------------------------------------------*/
/* PCMCIA Input Pins Register */
typedef struct{
__REG32 VS       : 2;
__REG32 WP       : 1;
__REG32 CD       : 2;
__REG32 BVD1     : 1;
__REG32 BVD2     : 1;
__REG32 RDY      : 1;
__REG32 POWERON  : 1;
__REG32          :23;
} __pcmcia_pipr_bits;

/* PCMCIA Status Change Register */
typedef struct{
__REG32 VSC1   : 1;
__REG32 VSC2   : 1;
__REG32 WPC    : 1;
__REG32 CDC1   : 1;
__REG32 CDC2   : 1;
__REG32 BVDC1  : 1;
__REG32 BVDC2  : 1;
__REG32 RDYL   : 1;
__REG32 RDYH   : 1;
__REG32 RDYF   : 1;
__REG32 RDYR   : 1;
__REG32 POWC   : 1;
__REG32        :20;
} __pcmcia_pscr_bits;

/* PCMCIA Enable Register */
typedef struct{
__REG32 VSE1       : 1;
__REG32 VSE2       : 1;
__REG32 WPE        : 1;
__REG32 CDE1       : 1;
__REG32 CDE2       : 1;
__REG32 BVDE1      : 1;
__REG32 BVDE2      : 1;
__REG32 RDYLE      : 1;
__REG32 RDYHE      : 1;
__REG32 RDYFE      : 1;
__REG32 RDYRE      : 1;
__REG32 POWERONEN  : 1;
__REG32 ERRINTEN   : 1;
__REG32            :19;
} __pcmcia_per_bits;

/* PCMCIA Base Registers */
typedef struct{
__REG32 PBA  :26;
__REG32      : 6;
} __pcmcia_pbr_bits;

/* PCMCIA Option Registers */
typedef struct{
__REG32 BSIZE  : 5;
__REG32 PSHT   : 6;
__REG32 PSST   : 6;
__REG32 PSL    : 7;
__REG32 PPS    : 1;
__REG32 PRS    : 2;
__REG32 WP     : 1;
__REG32 WPEN   : 1;
__REG32 PV     : 1;
__REG32        : 2;
} __pcmcia_por_bits;

/* PCMCIA Offset Registers */
typedef struct{
__REG32 POFA  :26;
__REG32       : 6;
} __pcmcia_pofr_bits;

/* PCMCIA General Control Register */
typedef struct{
__REG32 RESET   : 1;
__REG32 POE     : 1;
__REG32 SPKREN  : 1;
__REG32 LPMEN   : 1;
__REG32         :28;
} __pcmcia_pgcr_bits;

/* PCMCIA General Control Register */
typedef struct{
__REG32 WPE    : 1;
__REG32 CDE    : 1;
__REG32 SE     : 1;
__REG32 LPE    : 1;
__REG32 NWINE  : 1;
__REG32        :27;
} __pcmcia_pgsr_bits;

/* -------------------------------------------------------------------------*/
/*      One Wire                                                            */
/* -------------------------------------------------------------------------*/
/* O-Wire Control Register */
typedef struct{
__REG16       : 3;
__REG16 RDST  : 1;
__REG16 WR1   : 1;
__REG16 WR0   : 1;
__REG16 PST   : 1;
__REG16 RPP   : 1;
__REG16       : 8;
} __ow_control_bits;

/* O-Wire Time Divider Register */
typedef struct{
__REG16 DVDR  : 8;
__REG16       : 8;
} __ow_time_divider_bits;


/* O-Wire Reset Register */
typedef struct{
__REG16 RST  : 1;
__REG16      :15;
} __ow_reset_bits;

/* -------------------------------------------------------------------------*/
/*      ATA                                                                 */
/* -------------------------------------------------------------------------*/
/* ATA TIME_CONFIG0 Register */
typedef struct{
__REG32 TIME_OFF  : 8;
__REG32 TIME_ON   : 8;
__REG32 TIME_1    : 8;
__REG32 TIME_2W   : 8;
} __ata_time_config0_bits;

/* ATA TIME_CONFIG1 Register */
typedef struct{
__REG32 TIME_2R       : 8;
__REG32 TIME_AX       : 8;
__REG32 TIME_PIO_RDX  : 8;
__REG32 TIME_4        : 8;
} __ata_time_config1_bits;

/* ATA TIME_CONFIG2 Register */
typedef struct{
__REG32 TIME_9        : 8;
__REG32 TIME_M        : 8;
__REG32 TIME_JN       : 8;
__REG32 TIME_D        : 8;
} __ata_time_config2_bits;

/* ATA TIME_CONFIG3 Register */
typedef struct{
__REG32 TIME_K        : 8;
__REG32 TIME_ACK      : 8;
__REG32 TIME_ENV      : 8;
__REG32 TIME_RPX      : 8;
} __ata_time_config3_bits;

/* ATA TIME_CONFIG4 Register */
typedef struct{
__REG32 TIME_ZAH      : 8;
__REG32 TIME_MLIX     : 8;
__REG32 TIME_DVH      : 8;
__REG32 TIME_DZFS     : 8;
} __ata_time_config4_bits;

/* ATA TIME_CONFIG5 Register */
typedef struct{
__REG32 TIME_DVS      : 8;
__REG32 TIME_CVH      : 8;
__REG32 TIME_SS       : 8;
__REG32 TIME_CYC      : 8;
} __ata_time_config5_bits;

/* ATA FIFO_FILL Register */
typedef struct{
__REG32 FIFO_FILL     : 8;
__REG32               :24;
} __ata_fifo_fill_bits;

/* ATA_CONTROL Register */
typedef struct{
__REG32 IORDY_EN          : 1;
__REG32 DMA_WRITE         : 1;
__REG32 DMA_ULTRA_SELETED : 1;
__REG32 DMA_PENDING       : 1;
__REG32 FIFO_RCV_EN       : 1;
__REG32 FIFO_TX_EN        : 1;
__REG32 ATA_RST_B         : 1;
__REG32 FIFO_RST_B        : 1;
__REG32                   :24;
} __ata_control_bits;

/* ATA_INT_PENDING, INT_ENABLE, INT_CLEAR Registers */
typedef struct{
__REG32                   : 3;
__REG32 ATA_INTRQ2        : 1;
__REG32 CONTROLLER_IDLE   : 1;
__REG32 FIFO_OVERFLOW     : 1;
__REG32 FIFO_UNDERFLOW    : 1;
__REG32 ATA_INTRQ1        : 1;
__REG32                   :24;
} __ata_int_pending_bits;

/* ATA FIFO_ALARM Register */
typedef struct{
__REG32 FIFO_ALARM        : 8;
__REG32                   :24;
} __ata_fifo_alarm_bits;

/* -------------------------------------------------------------------------*/
/*      CSPI                                                                */
/* -------------------------------------------------------------------------*/
/* Control Registers */
typedef struct{
__REG32 BIT_COUNT   : 5;
__REG32 POL         : 1;
__REG32 PHA         : 1;
__REG32 SSCTL       : 1;
__REG32 SSPOL       : 1;
__REG32 XCH         : 1;
__REG32 SPIEN       : 1;
__REG32 MODE        : 1;
__REG32 DR_CTL      : 2;
__REG32 DATARATE    : 5;
__REG32 CS          : 2;
__REG32 SWAP        : 1;
__REG32 SDHC_SPIEN  : 1;
__REG32 BURST       : 1;
__REG32             : 8;
} __cspi_controlreg_bits;

/* Interrupt Control and Status Register */
typedef struct{
__REG32 TE          : 1;
__REG32 TH          : 1;
__REG32 TF          : 1;
__REG32 TSHFE       : 1;
__REG32 RR          : 1;
__REG32 RH          : 1;
__REG32 RF          : 1;
__REG32 RO          : 1;
__REG32 BO          : 1;
__REG32 TEEN        : 1;
__REG32 THEN        : 1;
__REG32 TFEN        : 1;
__REG32 TSHFEEN     : 1;
__REG32 RREN        : 1;
__REG32 RHEN        : 1;
__REG32 RFEN        : 1;
__REG32 ROEN        : 1;
__REG32 BOEN        : 1;
__REG32             :14;
} __cspi_intreg_bits;

/* Test Register */
typedef struct{
__REG32 TXCNT       : 4;
__REG32 RXCNT       : 4;
__REG32 SSTATUS     : 4;
__REG32             : 2;
__REG32 LBC         : 1;
__REG32             :17;
} __cspi_test_bits;

/* Sample Period Control Register */
typedef struct{
__REG32 WAIT          :15;
__REG32 CSRC          : 1;
__REG32               :16;
} __cspi_period_bits;

/* DMA Control Register */
typedef struct{
__REG32             : 4;
__REG32 RHDMA       : 1;
__REG32 RFDMA       : 1;
__REG32 TEDMA       : 1;
__REG32 THDMA       : 1;
__REG32             : 4;
__REG32 RHDEN       : 1;
__REG32 RFDEN       : 1;
__REG32 TEDEN       : 1;
__REG32 THDEN       : 1;
__REG32             :16;
} __cspi_dma_bits;

/* Soft Reset Register */
typedef struct{
__REG32 START       : 1;
__REG32             :31;
} __cspi_reset_bits;

/* -------------------------------------------------------------------------*/
/*               I2C registers                                              */
/* -------------------------------------------------------------------------*/
typedef struct {        /* I2C Address Register  */
__REG32      : 1;      /* Bit  0       - reserved*/
__REG32 ADR  : 7;      /* Bits 1  - 7  - Slave Address - Contains the specific slave address to be used by the I2C module.*/
__REG32      :24;      /* Bits 31 - 8  - Reserved*/
} __iadr_bits;

typedef struct {        /* I2C Frequency Divider Register (IFDR) */
__REG32 IC  : 6;       /* Bits 0  - 5   - I2C Clock Rate Divider - Prescales the clock for bit-rate selection.*/
__REG32     :26;       /* Bits 6  - 31  - Reserved*/
} __ifdr_bits;

typedef struct {        /* I2C Control Register (I2CR) */
__REG32       : 2;     /* Bits 0  - 1  - Reserved*/
__REG32 RSTA  : 1;     /* Bit  2       - Repeated START - Generates a repeated START condition*/
__REG32 TXAK  : 1;     /* Bit  3       - Transmit Acknowledge Enable (0 = Send ACK, 1 = Dont send ACK)*/
__REG32 MTX   : 1;     /* Bit  4       - Transmit/Receive Mode Select (0 = Rx, 1 = Tx)*/
__REG32 MSTA  : 1;     /* Bit  5       - Master/Slave Mode Select (0 = Slave, 1 = Master)*/
__REG32 IIEN  : 1;     /* Bit  6       - I2C Interrupt Enable*/
__REG32 IEN   : 1;     /* Bit  7       - I2C Enable*/
__REG32       :24;     /* Bits 8 - 31  - Reserved*/
} __i2cr_bits;

typedef struct {        /* I2C Status Register (I2SR) */
__REG32 RXAK  : 1;     /* Bit  0       - Received Acknowledge (0 = ACK received, 1 = No ACK received)*/
__REG32 IIF   : 1;     /* Bit  1       - I2C interrupt - (0 = No Int. pending, 1 = Interrupt pending )*/
__REG32 SRW   : 1;     /* Bit  2       - Slave Read/Write - Indicates the value of the R/W command bit*/
__REG32       : 1;     /* Bit  3       - Reserved*/
__REG32 IAL   : 1;     /* Bit  4       - Arbitration Lost*/
__REG32 IBB   : 1;     /* Bit  5       - I2C Bus Busy*/
__REG32 IAAS  : 1;     /* Bit  6       - I2C Addressed As a Slave*/
__REG32 ICF   : 1;     /* Bit  7       - Data Transfer (0=In Progress, 1 = Complete)*/
__REG32       :24;     /* Bits 8  - 31 - Reserved*/
} __i2sr_bits;

typedef struct {        /* I2C Data I/O Register (I2DR) */
__REG32 DATA  : 8;     /* Bits 0  - 7  - I2C Data to be transmitted / last byte received*/
__REG32       :24;     /* Bits 8 - 31  - Reserved*/
} __i2dr_bits;

/* -------------------------------------------------------------------------*/
/*      Keypad Port (KPP)                                                   */
/* -------------------------------------------------------------------------*/
/* Keypad Control Register */
typedef struct{
__REG16 KRE  : 8;
__REG16 KCO  : 8;
} __kpcr_bits;

/* Keypad Status Register */
typedef struct{
__REG16 KPKD    : 1;
__REG16 KPKR    : 1;
__REG16 KDSC    : 1;
__REG16 KRSS    : 1;
__REG16         : 4;
__REG16 KDIE    : 1;
__REG16 KRIE    : 1;
__REG16 KPP_EN  : 1;
__REG16         : 5;
} __kpsr_bits;

/* Keypad Data Direction Register */
typedef struct{
__REG16 KRDD  : 8;
__REG16 KCDD  : 8;
} __kddr_bits;

/* Keypad Data Direction Register */
typedef struct{
__REG16 KRD  : 8;
__REG16 KCD  : 8;
} __kpdr_bits;

/* -------------------------------------------------------------------------*/
/*      MMC/SDHC Registers                                                  */
/* -------------------------------------------------------------------------*/
/* MMC/SD Clock Control Register */
typedef struct{
__REG32 STOP_CLK     : 1;
__REG32 START_CLK    : 1;
__REG32              : 1;
__REG32 SDHC_RESET   : 1;
__REG32              :28;
} __str_stp_clk_bits;

/* MMC/SD Status Register */
typedef struct{
__REG32 TIME_OUT_READ      : 1;
__REG32 TIME_OUT_RESP      : 1;
__REG32 WRITE_CRC_ERR      : 1;
__REG32 READ_CRC_ERR       : 1;
__REG32                    : 1;
__REG32 RESP_CRC_ERR       : 1;
__REG32 BUF_WRITE_READY    : 1;
__REG32 BUF_READ_READY     : 1;
__REG32 CARD_BUS_CLK_RUN   : 1;
__REG32 WR_CRC_ERROR_CODE  : 2;
__REG32 READ_OP_DONE       : 1;
__REG32 WRITE_OP_DONE      : 1;
__REG32 END_CMD_RESP       : 1;
__REG32 SDIO_INT_ACTIVE    : 1;
__REG32                    : 9;
__REG32 BUF_OVFL           : 1;
__REG32 BUF_UND_RUN        : 1;
__REG32 XBUF_FULL          : 1;
__REG32 YBUF_FULL          : 1;
__REG32 XBUF_EMPTY         : 1;
__REG32 YBUF_EMPTY         : 1;
__REG32 CARD_REMOVAL       : 1;
__REG32 CARD_INSERTION     : 1;
} __sd_status_bits;

/* MMC/SD clock rate register */
typedef struct{
__REG32 CLK_DIVIDER    : 4;
__REG32 CLK_PRESCALER  :12;
__REG32                :16;
} __clk_rate_bits;

/* MMC/SD command and data control register */
typedef struct{
__REG32 FORMAT_OF_RESPONSE  : 3;
__REG32 DATA_ENABLE         : 1;
__REG32 WRITE_READ          : 1;
__REG32                     : 2;
__REG32 INIT                : 1;
__REG32 BUS_WIDTH           : 2;
__REG32 START_READ_WAIT     : 1;
__REG32 STOP_READ_WAIT      : 1;
__REG32 CMD_RESP_LONG_OFF   : 1;
__REG32                     : 2;
__REG32 CMD_RESUME          : 1;
__REG32                     :16;
} __cmd_dat_cont_bits;

/* MMC/SD response time out register */
typedef struct{
__REG32 RESPONSE_TIME_OUT  : 8;
__REG32                    :24;
} __res_to_bits;

/* MMC/SD read time out register */
typedef struct{
__REG32 DATA_READ_TIME_OUT  :16;
__REG32                     :16;
} __read_to_bits;

/* MMC/SD block length register */
typedef struct{
__REG32 BLOCK_LENGTH  :12;
__REG32               :20;
} __blk_len_bits;

/* MMC/SD number of blocks register */
typedef struct{
__REG32 NOB  :16;
__REG32      :16;
} __nob_bits;

/* MMC/SD revision number register */
typedef struct{
__REG32 REVISION_NUMBER  :16;
__REG32                  :16;
} __rev_no_bits;

/* SDHC Interrupt Control Register (INT_CNTR) */
typedef struct{
__REG32 READ_OP_DONE          : 1;
__REG32 WRITE_OP_DONE         : 1;
__REG32 END_CMD_RES           : 1;
__REG32 BUF_WRITE_EN          : 1;
__REG32 BUF_READ_EN           : 1;
__REG32                       : 7;
__REG32 DAT0_EN               : 1;
__REG32 SDIO_INT_EN           : 1;
__REG32 CARD_REMOVAL_EN       : 1;
__REG32 CARD_INSERTION_EN     : 1;
__REG32 CARD_REMOVAL_WKP_EN   : 1;
__REG32 CARD_INSERTION_WKP_EN : 1;
__REG32 SDIO_INT_WKP_EN       : 1;
__REG32                       :13;
} __mmcsd_int_ctrl_bits;

/* MMC/SD command number register */
typedef struct{
__REG32 COMMAND_NUMBER  : 6;
__REG32                 :26;
} __cmd_bits;

/* MMC/SD response FIFO register */
typedef struct{
__REG32 RESPONSE_CONTENT  :16;
__REG32                   :16;
} __res_fifo_bits;

/* -------------------------------------------------------------------------*/
/*      UARTs                                                               */
/* -------------------------------------------------------------------------*/
/* UARTs Receiver Register */
typedef struct{
__REG32 RX_DATA  : 8;     /* Bits 0-7             - Recieve Data*/
__REG32          : 2;     /* Bits 8-9             - Reserved*/
__REG32 PRERR    : 1;     /* Bit  10              - Receive Parity Error 1=error*/
__REG32 BRK      : 1;     /* Bit  11              - Receive break Caracter detected 1 = detected*/
__REG32 FRMERR   : 1;     /* Bit  12              - Receive Framing Error 1=error*/
__REG32 OVRRUN   : 1;     /* Bit  13              - Receive Over run Error 1=error*/
__REG32 ERR      : 1;     /* Bit  14              - Receive Error Detect (OVR,FRM,BRK,PR 0=error*/
__REG32          :17;
} __urxd_bits;

/* UARTs Transmitter Register */
typedef struct{
__REG32 TX_DATA  : 8;     /* Bits 7-0             - Transmit Data*/
__REG32          :24;
} __utxd_bits;

/* UARTs Control Register 1 */
typedef struct{
__REG32 UARTEN    : 1;     /* Bit  0       - UART Enable 1 = Enable the UART*/
__REG32 DOZE      : 1;     /* Bit  1       - DOZE 1 = The UART is disabled when in DOZE state*/
__REG32           : 1;     /* Bit  2*/
__REG32 TDMAEN    : 1;     /* Bit  3       - Transmitter Ready DMA Enable 1 = enable*/
__REG32 SNDBRK    : 1;     /* Bit  4       - Send BREAK 1 = send break char continuous*/
__REG32 RTSDEN    : 1;     /* Bit  5       - RTS Delta Interrupt Enable 1 = enable*/
__REG32 TXMPTYEN  : 1;     /* Bit  6       - Transmitter Empty Interrupt Enable 1 = enable*/
__REG32 IREN      : 1;     /* Bit  7       - Infrared Interface Enable 1 = enable*/
__REG32 RXDMAEN   : 1;     /* Bit  8       - Receive Ready DMA Enable 1 = enable*/
__REG32 RRDYEN    : 1;     /* Bit  9       - Receiver Ready Interrupt Enable 1 = Enable*/
__REG32 ICD       : 2;     /* Bit  10-11   - Idle Condition Detect*/
                           /*              - 00 = Idle for more than 4 frames*/
                           /*              - 01 = Idle for more than 8 frames*/
                           /*              - 10 = Idle for more than 16 frames*/
                           /*              - 11 = Idle for more than 32 frames*/
__REG32 IDEN      : 1;     /* Bit  12      - Idle Condition Detected Interrupt en 1=en*/
__REG32 TRDYEN    : 1;     /* Bit  13      - Transmitter Ready Interrupt Enable 1=en*/
__REG32 ADBR      : 1;     /* Bit  14      - AutoBaud Rate Detection enable 1=en*/
__REG32 ADEN      : 1;     /* Bit  15      - AutoBaud Rate Detection Interrupt en 1=en*/
__REG32           :16;
} __ucr1_bits;

/* UARTs Control Register 2 */
typedef struct{
__REG32 SRST   : 1;     /* Bit  0       -Software Reset 0 = Reset the tx and rx state machines*/
__REG32 RXEN   : 1;     /* Bit  1       -Receiver Enable 1 = Enable*/
__REG32 TXEN   : 1;     /* Bit  2       -Transmitter Enable 1= enable*/
__REG32 ATEN   : 1;     /* Bit  3       -Aging Timer Enable—This bit is used to mask the aging timer interrupt (triggered with AGTIM)*/
__REG32 RTSEN  : 1;     /* Bit  4       -Request to Send Interrupt Enable 1=enable*/
__REG32 WS     : 1;     /* Bit  5       -Word Size 0 = 7bit, 1= 8 bit*/
__REG32 STPB   : 1;     /* Bit  6       -Stop 0= 1 stop bits, 1= 2 stop bits*/
__REG32 PROE   : 1;     /* Bit  7       -Parity Odd/Even 1=Odd*/
__REG32 PREN   : 1;     /* Bit  8       -Parity Enable 1=enable parity generator*/
__REG32 RTEC   : 2;     /* Bits 9-10    -Request to Send Edge Control*/
                        /*              - 00 = Trigger interrupt on a rising edge*/
                        /*              - 01 = Trigger interrupt on a falling edge*/
                        /*              - 1X = Trigger interrupt on any edge*/
__REG32 ESCEN  : 1;     /* Bit  11      -Escape Enable 1 = Enable escape sequence detection*/
__REG32 CTS    : 1;     /* Bit  12      -Clear to Send 1 = The UARTx_CTS pin is low (active)*/
__REG32 CTSC   : 1;     /* Bit  13      -UARTx_CTS Pin controlled by 1= receiver 0= CTS bit*/
__REG32 IRTS   : 1;     /* Bit  14      -Ignore UARTx_RTS Pin 1=ignore*/
__REG32 ESCI   : 1;     /* Bit  15      -Escape Sequence Interrupt En 1=enable*/
__REG32        :16;
} __ucr2_bits;

/* UARTs Control Register 3 */
typedef struct{
__REG32 ACIEN      : 1;
__REG32 INVT       : 1;
__REG32 RXDMUXSEL  : 1;
__REG32            : 1;
__REG32 AWAKEN     : 1;
__REG32 AIRINTEN   : 1;
__REG32 RXDSEN     : 1;
__REG32 ADNIMP     : 1;
__REG32            : 3;
__REG32 FRAERREN   : 1;
__REG32 PARERREN   : 1;
__REG32            : 3;
__REG32            :16;
} __ucr3_bits;

/* UARTs Control Register 4 */
typedef struct{
__REG32 DREN   : 1;     /* Bit  0       -Receive Data Ready Interrupt Enable 1= enable*/
__REG32 OREN   : 1;     /* Bit  1       -Receiver Overrun Interrupt Enable 1= enable*/
__REG32 BKEN   : 1;     /* Bit  2       -BREAK Condition Detected Interrupt en 1= enable*/
__REG32 TCEN   : 1;     /* Bit  3       -Transmit Complete Interrupt Enable1 = Enable*/
__REG32 LPBYP  : 1;     /* Bit  4       -Low Power Bypass—Allows to bypass the low power new features in UART for . To use during debug phase.*/
__REG32 IRSC   : 1;     /* Bit  5       -IR Special Case vote logic uses 1= uart ref clk*/
__REG32        : 1;     /* Bit  6       -*/
__REG32 WKEN   : 1;     /* Bit  7       -WAKE Interrupt Enable 1= enable*/
__REG32 ENIRI  : 1;     /* Bit  8       -Serial Infrared Interrupt Enable 1= enable*/
__REG32 INVR   : 1;     /* Bit  9       -Inverted Infrared Reception 1= active high*/
__REG32 CTSTL  : 6;     /* Bits 10-15   -CTS Trigger Level*/
                        /*              000000 = 0 characters received*/
                        /*              000001 = 1 characters in the RxFIFO*/
                        /*              ...*/
                        /*              100000 = 32 characters in the RxFIFO (maximum)*/
                        /*              All Other Settings Reserved*/
__REG32        :16;
} __ucr4_bits;

/* UARTs FIFO Control Register */
typedef struct{
__REG32 RXTL    : 6;     /* Bits 0-5     -Receiver Trigger Level*/
                         /*              000000 = 0 characters received*/
                         /*              000001 = RxFIFO has 1 character*/
                         /*              ...*/
                         /*              011111 = RxFIFO has 31 characters*/
                         /*              100000 = RxFIFO has 32 characters (maximum)*/
                         /*              All Other Settings Reserved*/
__REG32         : 1;     /* Bit  6       - Reserved*/
__REG32 RFDIV   : 3;     /* Bits 7-9     -Reference Frequency Divider*/
                         /*              000 = Divide input clock by 6*/
                         /*              001 = Divide input clock by 5*/
                         /*              010 = Divide input clock by 4*/
                         /*              011 = Divide input clock by 3*/
                         /*              100 = Divide input clock by 2*/
                         /*              101 = Divide input clock by 1*/
                         /*              110 = Divide input clock by 7*/
__REG32 TXTL    : 6;     /* Bits 10-15   -Transmitter Trigger Level*/
                         /*              000000 = Reserved*/
                         /*              000001 = Reserved*/
                         /*              000010 = TxFIFO has 2 or fewer characters*/
                         /*              ...*/
                         /*              011111 = TxFIFO has 31 or fewer characters*/
                         /*              100000 = TxFIFO has 32 characters (maximum)*/
                         /*              All Other Settings Reserved*/
__REG32        :16;
} __ufcr_bits;

/* UARTs Status Register 1 */
typedef struct{
__REG32            : 4;
__REG32 AWAKE      : 1;
__REG32 AIRINT     : 1;
__REG32 RXDS       : 1;
__REG32            : 1;
__REG32 AGTIM      : 1;
__REG32 RRDY       : 1;
__REG32 FRAMERR    : 1;
__REG32 ESCF       : 1;
__REG32 RTSD       : 1;
__REG32 TRDY       : 1;
__REG32 RTSS       : 1;
__REG32 PARITYERR  : 1;
__REG32            :16;
} __usr1_bits;

/* UARTs Status Register 2 */
typedef struct{
__REG32 RDR      : 1;
__REG32 ORE      : 1;
__REG32 BRCD     : 1;
__REG32 TXDC     : 1;
__REG32 RTSF     : 1;
__REG32          : 2;
__REG32 WAKE     : 1;
__REG32 IRINT    : 1;
__REG32          : 2;
__REG32 ACST     : 1;
__REG32 IDLE     : 1;
__REG32          : 1;
__REG32 TXFE     : 1;
__REG32 ADET     : 1;
__REG32          :16;
} __usr2_bits;

/* UARTs Escape Character Register */
typedef struct{
__REG32 ESC_CHAR  : 8;     /* Bits 0-7     -UART Escape Character*/
__REG32           :24;
} __uesc_bits;

/* UARTs Escape Timer Register */
typedef struct{
__REG32 TIM  :12;     /* Bits 0-11    -UART Escape Timer*/
__REG32      :20;
} __utim_bits;

/* UARTS Test Register 1 */
typedef struct{
__REG32 SOFTRST  : 1;
__REG32          : 2;
__REG32 RXFULL   : 1;
__REG32 TXFULL   : 1;
__REG32 RXEMPTY  : 1;
__REG32 TXEMPTY  : 1;
__REG32          : 2;
__REG32 RXDBG    : 1;
__REG32 LOOPIR   : 1;
__REG32 DBGEN    : 1;
__REG32 LOOP     : 1;
__REG32 FRCPERR  : 1;
__REG32          :18;
} __uts_bits;

/* -------------------------------------------------------------------------*/
/*      FEC                                                                 */
/* -------------------------------------------------------------------------*/
/* Ethernet Interrupt Event Register (EIR)
   Interrupt Mask Register (EIMR) */
typedef struct {
__REG32          :19;
__REG32 UN       : 1;
__REG32 RL       : 1;
__REG32 LC       : 1;
__REG32 EBERR    : 1;
__REG32 MII      : 1;
__REG32 RXB      : 1;
__REG32 RXF      : 1;
__REG32 TXB      : 1;
__REG32 TXF      : 1;
__REG32 GRA      : 1;
__REG32 BABT     : 1;
__REG32 BABR     : 1;
__REG32 HBERR    : 1;
} __fec_eir_bits;

/* Receive Descriptor Active Register (RDAR) */
typedef struct {
__REG32               :24;
__REG32 RDAR          : 1;
__REG32               : 7;
} __fec_rdar_bits;

/* Transmit Descriptor Active Register (TDAR) */
typedef struct {
__REG32               :24;
__REG32 TDAR          : 1;
__REG32               : 7;
} __fec_tdar_bits;

/* Ethernet Control Register (ECR) */
typedef struct {
__REG32 RESET         : 1;
__REG32 ETHER_EN      : 1;
__REG32               :30;
} __fec_ecr_bits;

/* MII Management Frame Register (MMFR) */
typedef struct {
__REG32 DATA          :16;
__REG32 TA            : 2;
__REG32 RA            : 5;
__REG32 PA            : 5;
__REG32 OP            : 2;
__REG32 ST            : 2;
} __fec_mmfr_bits;

/* MII Speed Control Register (MSCR) */
typedef struct {
__REG32               : 1;
__REG32 MII_SPEED     : 6;
__REG32 DIS_PREAMBLE  : 1;
__REG32               :24;
} __fec_mscr_bits;

/* MIB Control Register (MIBC) */
typedef struct {
__REG32               :30;
__REG32 MIB_IDLE      : 1;
__REG32 MIB_DIS       : 1;
} __fec_mibc_bits;

/* Receive Control Register (RCR) */
typedef struct {
__REG32 LOOP          : 1;
__REG32 DRT           : 1;
__REG32 MII_MODE      : 1;
__REG32 PROM          : 1;
__REG32 BC_REJ        : 1;
__REG32 FCE           : 1;
__REG32               :10;
__REG32 MAX_FL        :11;
__REG32               : 5;
} __fec_rcr_bits;

/* Transmit Control Register (TCR) */
typedef struct {
__REG32 GTS           : 1;
__REG32 HBC           : 1;
__REG32 FDEN          : 1;
__REG32 TFC_PAUSE     : 1;
__REG32 RFC_PAUSE     : 1;
__REG32               :27;
} __fec_tcr_bits;

/* Physical Address High Register (PAUR) */
typedef struct {
__REG32 TYPE          :16;
__REG32 PADDR2        :16;
} __fec_paur_bits;

/* Opcode/Pause Duration Register (OPD) */
typedef struct {
__REG32 PAUSE_DUR     :16;
__REG32 OPCODE        :16;
} __fec_opd_bits;

/* FIFO Transmit FIFO Watermark Register (TFWR) */
typedef struct {
__REG32 WMRK          : 2;
__REG32               :30;
} __fec_tfwr_bits;

/* FIFO Receive Bound Register (FRBR) */
typedef struct {
__REG32               : 2;
__REG32 R_BOUND       : 8;
__REG32               :22;
} __fec_frbr_bits;

/* FIFO Receive Start Register (FRSR) */
typedef struct {
__REG32               : 2;
__REG32 R_FSTART      : 8;
__REG32               :22;
} __fec_frsr_bits;

/* Receive Buffer Size Register (EMRBR) */
typedef struct {
__REG32               : 4;
__REG32 R_BUF_SIZE    : 7;
__REG32               :21;
} __fec_emrbr_bits;

/* -------------------------------------------------------------------------*/
/*      USB Registers                                                       */
/* -------------------------------------------------------------------------*/
/* USB Control Register */
typedef struct{
__REG32 BPE       : 1;
__REG32           : 3;
__REG32 H1DT      : 1;
__REG32 H2DT      : 1;
__REG32           : 2;
__REG32 H1PM      : 1;
__REG32 H1BPVAL   : 2;
__REG32 H1WIE     : 1;
__REG32           : 1;
__REG32 H1SIC     : 2;
__REG32 H1WIR     : 1;
__REG32 H2PM      : 1;
__REG32           : 2;
__REG32 H2WIE     : 1;
__REG32 H2UIE     : 1;
__REG32 H2SIC     : 2;
__REG32 H2WIR     : 1;
__REG32 OPM       : 1;
__REG32 OBPVAL    : 2;
__REG32 OWIE      : 1;
__REG32 OUIE      : 1;
__REG32 OSIC      : 2;
__REG32 OWIR      : 1;
} __usb_ctrl_bits;

/* OTG Mirror Register (OTGMIRROR) */
typedef struct{
__REG32 IDDIG     : 1;
__REG32 ASESVLD   : 1;
__REG32 BSESVLD   : 1;
__REG32 VBUSVAL   : 1;
__REG32 SESEND    : 1;
__REG32           :27;
} __usb_otg_mirror_bits;

/* -------------------------------------------------------------------------*/
/*      USB OTG/HOST Registers                                              */
/* -------------------------------------------------------------------------*/
/* USB Identification Register */
typedef struct{
__REG32 ID        : 6;
__REG32           : 2;
__REG32 NID       : 6;
__REG32           : 2;
__REG32 REVISION  : 8;
__REG32           : 8;
} __usb_id_bits;

/* USB General Hardware Parameters */
typedef struct{
__REG32 RT        : 1;
__REG32 CLKC      : 2;
__REG32 BWT       : 1;
__REG32 PHYW      : 2;
__REG32 PHYM      : 3;
__REG32 SM        : 1;
__REG32           :22;
} __usb_hwgeneral_bits;

/* USB Host Hardware Parameters */
typedef struct{
__REG32 HC        : 1;
__REG32 NPORT     : 3;
__REG32           :12;
__REG32 TTASY     : 8;
__REG32 TTPER     : 8;
} __usb_hwhost_bits;

/* USB Device Hardware Parameters */
typedef struct{
__REG32 DC        : 1;
__REG32 DEVEP     : 5;
__REG32           :26;
} __usb_hwdevice_bits;

/* USB TX Buffer Hardware Parameters */
typedef struct{
__REG32 TCBURST   : 8;
__REG32 TXADD     : 8;
__REG32 TXCHANADD : 8;
__REG32           : 7;
__REG32 TXLCR     : 1;
} __usb_hwtxbuf_bits;

/* USB RX Buffer Hardware Parameters */
typedef struct{
__REG32 RXBURST   : 8;
__REG32 RXADD     : 8;
__REG32           :16;
} __usb_hwrxbuf_bits;

/* USB Host Control Structural Parameters */
typedef struct{
__REG32 N_PORTS   : 4;
__REG32 PPC       : 1;
__REG32           : 3;
__REG32 N_PCC     : 4;
__REG32 N_CC      : 4;
__REG32 PI        : 1;
__REG32           : 3;
__REG32 N_PTT     : 4;
__REG32 N_TT      : 4;
__REG32           : 4;
} __usb_hcsparams_bits;

/* USB Host Control Capability Parameters */
typedef struct{
__REG32 ADC       : 1;
__REG32 PFL       : 1;
__REG32 ASP       : 1;
__REG32           : 1;
__REG32 IST       : 4;
__REG32 EECP      : 8;
__REG32           :16;
} __usb_hccparams_bits;

/* USB Device Interface Version Number */
typedef struct{
__REG32 DCIVERSION  :16;
__REG32             :16;
} __usb_dciversion_bits;

/* USB Device Interface Version Number */
typedef struct{
__REG32 DEN         : 5;
__REG32             : 2;
__REG32 DC          : 1;
__REG32 HC          : 1;
__REG32             :23;
} __usb_dccparams_bits;

/* USB General Purpose Timer #0 Load Register */
typedef struct{
__REG32 GPTLD       :24;
__REG32             : 8;
} __usb_gptimer0ld_bits;

/* USB General Purpose Timer #0 Controller */
typedef struct{
__REG32 GPTCNT      :24;
__REG32 GPTMOD      : 1;
__REG32             : 5;
__REG32 GPTRST      : 1;
__REG32 GPTRUN      : 1;
} __usb_gptimer0ctrl_bits;

/* USB Command Register */
typedef struct{
__REG32 RS          : 1;
__REG32 RST         : 1;
__REG32 FS0         : 1;
__REG32 FS1         : 1;
__REG32 PSE         : 1;
__REG32 ASE         : 1;
__REG32 IAA         : 1;
__REG32 LR          : 1;
__REG32 ASP0        : 1;
__REG32 ASP1        : 1;
__REG32             : 1;
__REG32 ASPE        : 1;
__REG32 ATDTW       : 1;
__REG32 SUTW        : 1;
__REG32             : 1;
__REG32 FS2         : 1;
__REG32 ITC         : 8;
__REG32             : 8;
} __usb_usbcmd_bits;

/* USB Status */
typedef struct{
__REG32 UI          : 1;
__REG32 UEI         : 1;
__REG32 PCI         : 1;
__REG32 FRI         : 1;
__REG32 SEI         : 1;
__REG32 AAI         : 1;
__REG32 URI         : 1;
__REG32 SRI         : 1;
__REG32 SLI         : 1;
__REG32             : 1;
__REG32 ULPII       : 1;
__REG32             : 1;
__REG32 HCH         : 1;
__REG32 RCL         : 1;
__REG32 PS          : 1;
__REG32 AS          : 1;
__REG32             : 8;
__REG32 TI0         : 1;
__REG32 TI1         : 1;
__REG32             : 6;
} __usb_usbsts_bits;

/* USB Interrupt Enable */
typedef struct{
__REG32 UE          : 1;
__REG32 UEE         : 1;
__REG32 PCE         : 1;
__REG32 FRE         : 1;
__REG32 SEE         : 1;
__REG32 AAE         : 1;
__REG32 URE         : 1;
__REG32 SRE         : 1;
__REG32 SLE         : 1;
__REG32             : 1;
__REG32 ULPIE       : 1;
__REG32             :13;
__REG32 TIE0        : 1;
__REG32 TIE1        : 1;
__REG32             : 6;
} __usb_usbintr_bits;

/* USB Frame Index */
typedef struct{
__REG32 FRINDEX     :14;
__REG32             :18;
} __usb_frindex_bits;

/* USB OTG Host Controller Frame List Base Address
   Device Controller USB Device Address */
typedef union {
  /* UOG_PERIODICLISTBASE*/
  struct {
   __REG32 PERBASE     :32;
  };
  /* UOG_DEVICEADDR*/
  struct {
  __REG32             :25;
  __REG32 USBADR      : 7;
  };
} __usb_periodiclistbase_bits;

typedef union {
  /* UHx_PERIODICLISTBASE*/
  struct {
   __REG32 PERBASE     :32;
  };
  /* UHx_DEVICEADDR*/
  struct {
  __REG32             :25;
  __REG32 USBADR      : 7;
  };
} __uh_periodiclistbase_bits;

/* USB Host Controller Embedded TT Async. Buffer Status */
typedef struct{
__REG32 RXPBURST    : 8;
__REG32 TXPBURST    : 9;
__REG32             :15;
} __usb_burstsize_bits;

/* USB TXFILLTUNING */
typedef struct{
__REG32 TXSCHOH     : 8;
__REG32 TXSCHEALTH  : 5;
__REG32             : 3;
__REG32 TXFIFOTHRES : 6;
__REG32             :10;
} __usb_txfilltuning_bits;

/* USB ULPI VIEWPORT */
typedef struct{
__REG32 ULPIDATWR   : 8;
__REG32 ULPIDATRD   : 8;
__REG32 ULPIADDR    : 8;
__REG32 ULPIPORT    : 3;
__REG32 ULPISS      : 1;
__REG32             : 1;
__REG32 ULPIRW      : 1;
__REG32 ULPIRUN     : 1;
__REG32 ULPIWU      : 1;
} __usb_ulpiview_bits;

/* USB Port Status Control[1:8] */
typedef struct{
__REG32 CCS         : 1;
__REG32 CSC         : 1;
__REG32 PE          : 1;
__REG32 PEC         : 1;
__REG32 OCA         : 1;
__REG32 OCC         : 1;
__REG32 FPR         : 1;
__REG32 SUSP        : 1;
__REG32 PR          : 1;
__REG32 HSP         : 1;
__REG32 LS          : 2;
__REG32 PP          : 1;
__REG32 PO          : 1;
__REG32 PIC         : 2;
__REG32 PTC         : 4;
__REG32 WKCN        : 1;
__REG32 WKDS        : 1;
__REG32 WKOC        : 1;
__REG32 PHCD        : 1;
__REG32 PFSC        : 1;
__REG32             : 1;
__REG32 PSPD        : 2;
__REG32 PTW         : 1;
__REG32 STS         : 1;
__REG32 PTS         : 2;
} __usb_portsc_bits;

/* USB Status Control */
typedef struct{
__REG32 VD          : 1;
__REG32 VC          : 1;
__REG32             : 1;
__REG32 OT          : 1;
__REG32 DP          : 1;
__REG32 IDPU        : 1;
__REG32             : 2;
__REG32 ID          : 1;
__REG32 AVV         : 1;
__REG32 ASV         : 1;
__REG32 BSV         : 1;
__REG32 BSE         : 1;
__REG32 _1MST       : 1;
__REG32 DPS         : 1;
__REG32             : 1;
__REG32 IDIS        : 1;
__REG32 AVVIS       : 1;
__REG32 ASVIS       : 1;
__REG32 BSVIS       : 1;
__REG32 BSEIS       : 1;
__REG32 _1MSS       : 1;
__REG32 DPIS        : 1;
__REG32             : 1;
__REG32 IDIE        : 1;
__REG32 AVVIE       : 1;
__REG32 ASVIE       : 1;
__REG32 BSVIE       : 1;
__REG32 BSEIE       : 1;
__REG32 _1MSE       : 1;
__REG32 DPIE        : 1;
__REG32             : 1;
} __usb_otgsc_bits;

/* USB Device Mode */
typedef struct{
__REG32 CM          : 2;
__REG32 ES          : 1;
__REG32 SLOM        : 1;
__REG32 SDIS        : 1;
__REG32             :27;
} __usb_usbmode_bits;

/* USB Endpoint Setup Status */
typedef struct{
__REG32 ENDPTSETUPSTAT0   : 1;
__REG32 ENDPTSETUPSTAT1   : 1;
__REG32 ENDPTSETUPSTAT2   : 1;
__REG32 ENDPTSETUPSTAT3   : 1;
__REG32 ENDPTSETUPSTAT4   : 1;
__REG32 ENDPTSETUPSTAT5   : 1;
__REG32 ENDPTSETUPSTAT6   : 1;
__REG32 ENDPTSETUPSTAT7   : 1;
__REG32 ENDPTSETUPSTAT8   : 1;
__REG32 ENDPTSETUPSTAT9   : 1;
__REG32 ENDPTSETUPSTAT10  : 1;
__REG32 ENDPTSETUPSTAT11  : 1;
__REG32 ENDPTSETUPSTAT12  : 1;
__REG32 ENDPTSETUPSTAT13  : 1;
__REG32 ENDPTSETUPSTAT14  : 1;
__REG32 ENDPTSETUPSTAT15  : 1;
__REG32                   :16;
} __usb_endptsetupstat_bits;

/* USB Endpoint Initialization */
typedef struct{
__REG32 PERB0       : 1;
__REG32 PERB1       : 1;
__REG32 PERB2       : 1;
__REG32 PERB3       : 1;
__REG32 PERB4       : 1;
__REG32 PERB5       : 1;
__REG32 PERB6       : 1;
__REG32 PERB7       : 1;
__REG32 PERB8       : 1;
__REG32 PERB9       : 1;
__REG32 PERB10      : 1;
__REG32 PERB11      : 1;
__REG32 PERB12      : 1;
__REG32 PERB13      : 1;
__REG32 PERB14      : 1;
__REG32 PERB15      : 1;
__REG32 PETB0       : 1;
__REG32 PETB1       : 1;
__REG32 PETB2       : 1;
__REG32 PETB3       : 1;
__REG32 PETB4       : 1;
__REG32 PETB5       : 1;
__REG32 PETB6       : 1;
__REG32 PETB7       : 1;
__REG32 PETB8       : 1;
__REG32 PETB9       : 1;
__REG32 PETB10      : 1;
__REG32 PETB11      : 1;
__REG32 PETB12      : 1;
__REG32 PETB13      : 1;
__REG32 PETB14      : 1;
__REG32 PETB15      : 1;
} __usb_endptprime_bits;

/* USB Endpoint De-Initialize */
typedef struct{
__REG32 FERB0       : 1;
__REG32 FERB1       : 1;
__REG32 FERB2       : 1;
__REG32 FERB3       : 1;
__REG32 FERB4       : 1;
__REG32 FERB5       : 1;
__REG32 FERB6       : 1;
__REG32 FERB7       : 1;
__REG32 FERB8       : 1;
__REG32 FERB9       : 1;
__REG32 FERB10      : 1;
__REG32 FERB11      : 1;
__REG32 FERB12      : 1;
__REG32 FERB13      : 1;
__REG32 FERB14      : 1;
__REG32 FERB15      : 1;
__REG32 FETB0       : 1;
__REG32 FETB1       : 1;
__REG32 FETB2       : 1;
__REG32 FETB3       : 1;
__REG32 FETB4       : 1;
__REG32 FETB5       : 1;
__REG32 FETB6       : 1;
__REG32 FETB7       : 1;
__REG32 FETB8       : 1;
__REG32 FETB9       : 1;
__REG32 FETB10      : 1;
__REG32 FETB11      : 1;
__REG32 FETB12      : 1;
__REG32 FETB13      : 1;
__REG32 FETB14      : 1;
__REG32 FETB15      : 1;
} __usb_endptflush_bits;

/* USB Endpoint Status */
typedef struct{
__REG32 ERBR0       : 1;
__REG32 ERBR1       : 1;
__REG32 ERBR2       : 1;
__REG32 ERBR3       : 1;
__REG32 ERBR4       : 1;
__REG32 ERBR5       : 1;
__REG32 ERBR6       : 1;
__REG32 ERBR7       : 1;
__REG32 ERBR8       : 1;
__REG32 ERBR9       : 1;
__REG32 ERBR10      : 1;
__REG32 ERBR11      : 1;
__REG32 ERBR12      : 1;
__REG32 ERBR13      : 1;
__REG32 ERBR14      : 1;
__REG32 ERBR15      : 1;
__REG32 ETBR0       : 1;
__REG32 ETBR1       : 1;
__REG32 ETBR2       : 1;
__REG32 ETBR3       : 1;
__REG32 ETBR4       : 1;
__REG32 ETBR5       : 1;
__REG32 ETBR6       : 1;
__REG32 ETBR7       : 1;
__REG32 ETBR8       : 1;
__REG32 ETBR9       : 1;
__REG32 ETBR10      : 1;
__REG32 ETBR11      : 1;
__REG32 ETBR12      : 1;
__REG32 ETBR13      : 1;
__REG32 ETBR14      : 1;
__REG32 ETBR15      : 1;
} __usb_endptstat_bits;

/* USB Endpoint Compete */
typedef struct{
__REG32 ERCE0       : 1;
__REG32 ERCE1       : 1;
__REG32 ERCE2       : 1;
__REG32 ERCE3       : 1;
__REG32 ERCE4       : 1;
__REG32 ERCE5       : 1;
__REG32 ERCE6       : 1;
__REG32 ERCE7       : 1;
__REG32 ERCE8       : 1;
__REG32 ERCE9       : 1;
__REG32 ERCE10      : 1;
__REG32 ERCE11      : 1;
__REG32 ERCE12      : 1;
__REG32 ERCE13      : 1;
__REG32 ERCE14      : 1;
__REG32 ERCE15      : 1;
__REG32 ETCE0       : 1;
__REG32 ETCE1       : 1;
__REG32 ETCE2       : 1;
__REG32 ETCE3       : 1;
__REG32 ETCE4       : 1;
__REG32 ETCE5       : 1;
__REG32 ETCE6       : 1;
__REG32 ETCE7       : 1;
__REG32 ETCE8       : 1;
__REG32 ETCE9       : 1;
__REG32 ETCE10      : 1;
__REG32 ETCE11      : 1;
__REG32 ETCE12      : 1;
__REG32 ETCE13      : 1;
__REG32 ETCE14      : 1;
__REG32 ETCE15      : 1;
} __usb_endptcomplete_bits;

/* USB Endpoint Control 0 */
typedef struct{
__REG32 RXS         : 1;
__REG32             : 1;
__REG32 RXT         : 2;
__REG32             : 3;
__REG32 RXE         : 1;
__REG32             : 8;
__REG32 TXS         : 1;
__REG32             : 1;
__REG32 TXT         : 2;
__REG32             : 3;
__REG32 TXE         : 1;
__REG32             : 8;
} __usb_endptctrl0_bits;

/* USB Endpoint Control 1-15 */
typedef struct{
__REG32 RXS         : 1;
__REG32 RXD         : 1;
__REG32 RXT         : 2;
__REG32             : 1;
__REG32 RXI         : 1;
__REG32 RXR         : 1;
__REG32 RXE         : 1;
__REG32             : 8;
__REG32 TXS         : 1;
__REG32 TXD         : 1;
__REG32 TXT         : 2;
__REG32             : 1;
__REG32 TXI         : 1;
__REG32 TXR         : 1;
__REG32 TXE         : 1;
__REG32             : 8;
} __usb_endptctrl_bits;

/* -------------------------------------------------------------------------*/
/*      General-Purpose Timer                                               */
/* -------------------------------------------------------------------------*/
/* GPT Control Register */
typedef struct{
__REG32 TEN        : 1;     /* Bit  0       - Timer Enable1 = Timer is enabled 0 = Timer is disabled (counter reset to 0)*/
__REG32 CLKSOURCE  : 3;     /* Bits 1  - 3  - Clock Source*/
                            /*              - 000 = Stop count (clock disabled)*/
                            /*              - 001 = PERCLK1 to prescaler*/
                            /*              - 010 = PERCLK1 ÷16 to prescaler*/
                            /*              - 011 = TIN to prescaler*/
                            /*              - 1xx = 32 kHz clock to prescaler*/
__REG32 COMP_EN    : 1;     /* Bit  4       - Compare Interrupt Enable—This bit enables the compares interrupt*/
__REG32 CAPT_EN    : 1;     /* Bit  5       - Capture Interrupt Enable—This bit enables the capture interrupt*/
__REG32 CAP        : 2;     /* Bits 6  - 7  - Capture Edge*/
                            /*              - 00 = Disable the capture function*/
                            /*              - 01 = Capture on the rising edge and generate an interrupt*/
                            /*              - 10 = Capture on the falling edge and generate an interrupt*/
                            /*              - 11 = Capture on the rising and falling edges and generate an interrupt*/
__REG32 FRR        : 1;     /* Bit  8       - Free-Run/Restart0 = Restart mode 1 = Free-run mode*/
__REG32 OM         : 1;     /* Bit  9       - Output Mode—This bit controls the output mode of the timer after compare event occurs.*/
__REG32 CC         : 1;     /* Bit  10      - Counter Clear—This bit determines whether the counter is to be cleared when TEN=0 (timer disabled).*/
__REG32            : 4;     /* Bits 11 - 14 - Reserved*/
__REG32 SWR        : 1;     /* Bit  15      - Software Reset0 = No software reset sent*/
__REG32            :16;     /* Bits 16 - 31 - Reserved*/
} __tctl_bits;

/* GPT Prescaler Register */
typedef struct{
__REG32 PRESCALER  :11;     /* Bits 0  - 10 -*/
                            /*              - 0x00 = Divide by 1*/
                            /*              - 0x01 = Divide by 2*/
                            /*              - ...*/
                            /*              - 0x3FF = Divide by 2048*/
__REG32            :21;
} __tprer_bits;

/* GPT Status Register 1 */
typedef struct{
__REG32 COMP  : 1;     /* bit  0  - Compare Event0 = No compare event occurred*/
__REG32 CAPT  : 1;     /* bit  1  - Capture Event0 = No capture event occurred*/
__REG32       :30;
} __tstat_bits;

/* -------------------------------------------------------------------------*/
/*      PWM Registers                                                       */
/* -------------------------------------------------------------------------*/
/* PWM control register */
typedef struct{
__REG32 EN         : 1;
__REG32 REPEAT     : 2;
__REG32 SWR        : 1;
__REG32 PRESCALER  :12;
__REG32 CLKSRC     : 2;
__REG32 POUTC      : 2;
__REG32 HCTR       : 1;
__REG32 BCTR       : 1;
__REG32 DBGEN      : 1;
__REG32 WAITEN     : 1;
__REG32 DOZEN      : 1;
__REG32 STOPEN     : 1;
__REG32 FWM        : 2;
__REG32            : 4;
} __pwmcr_bits;

/* PWM Status Register */
typedef struct{
__REG32 FIFOAV     : 3;
__REG32 FE         : 1;
__REG32 ROV        : 1;
__REG32 CMP        : 1;
__REG32 FWE        : 1;
__REG32            :25;
} __pwmsr_bits;

/* PWM Interrupt Register */
typedef struct{
__REG32 FIE     : 1;
__REG32 RIE     : 1;
__REG32 CIE     : 1;
__REG32         :29;
} __pwmir_bits;

/* PWM Sample Register */
typedef struct{
__REG32 SAMPLE  :16;
__REG32         :16;
} __pwmsar_bits;

/* PWM Period Register */
typedef struct{
__REG32 PERIOD  :16;
__REG32         :16;
} __pwmpr_bits;

/* PWM Counter Register */
typedef struct{
__REG32 COUNT   :16;
__REG32         :16;
} __pwmcnr_bits;

/* -------------------------------------------------------------------------*/
/*      RTC Registers                                                       */
/* -------------------------------------------------------------------------*/
/* RTC hours and minutes counter register */
typedef struct{
__REG32 MINUTES  : 6;
__REG32          : 2;
__REG32 HOURS    : 5;
__REG32          :19;
} __rtc_hourmin_bits;

/* RTC seconds counter register */
typedef struct{
__REG32 SECONDS  : 6;
__REG32          :26;
} __rtc_seconds_bits;

/* RTC control register */
typedef struct{
__REG32 SWR  : 1;
__REG32 GEN  : 1;
__REG32      : 3;
__REG32 XTL  : 2;
__REG32 EN   : 1;
__REG32      :24;
} __rtcctl_bits;

/* RTC interrupt status register */
typedef struct{
__REG32 SW    : 1;
__REG32 MIN   : 1;
__REG32 ALM   : 1;
__REG32 DAY   : 1;
__REG32 _1HZ  : 1;
__REG32 HR    : 1;
__REG32       : 1;
__REG32 _2HZ  : 1;
__REG32 SAM0  : 1;
__REG32 SAM1  : 1;
__REG32 SAM2  : 1;
__REG32 SAM3  : 1;
__REG32 SAM4  : 1;
__REG32 SAM5  : 1;
__REG32 SAM6  : 1;
__REG32 SAM7  : 1;
__REG32       :16;
} __rtcisr_bits;

/* RTC stopwatch minutes register */
typedef struct{
__REG32 CNT  : 6;
__REG32      :26;
} __rtc_stpwch_bits;

/* RTC Days counter register */
typedef struct{
__REG32 DAYS  :16;
__REG32       :16;
} __rtc_dayr_bits;

/* RTC Day alarm register */
typedef struct{
__REG32 DAYSAL  :16;
__REG32         :16;
} __rtc_dayalarm_bits;

/* -------------------------------------------------------------------------*/
/*      Watchdog Registers                                                  */
/* -------------------------------------------------------------------------*/
typedef struct {        /* Watchdog Control Register (0x10002000) Reset  (0x00000000)*/
__REG16 WDZST  : 1;     /* Bit  0       - Watchdog Low Power*/
__REG16 WDBG   : 1;     /* Bits 1       - Watchdog DEBUG Enable*/
__REG16 WDE    : 1;     /* Bit  2       - Watchdog Enable*/
__REG16 WRE    : 1;     /* Bit  3       - ~WDOG or ~WDOG_RESET Enable*/
__REG16 SRS    : 1;     /* Bit  4       - ~Software Reset Signal*/
__REG16 WDA    : 1;     /* Bit  5       - ~Watchdog Assertion*/
__REG16        : 2;     /* Bits 6 - 7   - Reserved*/
__REG16 WT     : 8;     /* Bits 8 - 15  - Watchdog Time-Out Field*/
} __wcr_bits;

typedef struct {        /* Watchdog Reset Status Register (0x10002004) Reset (*)*/
__REG16 SFTW  : 1;     /* Bit  0       - Software Reset*/
__REG16 TOUT  : 1;     /* Bit  1       - Time-out*/
__REG16       : 1;     /* Bit  2       - Reserved*/
__REG16 EXT   : 1;     /* Bit  3       - External Reset*/
__REG16 PWR   : 1;     /* Bit  4       - Power-On Reset*/
__REG16       :11;     /* Bits 5  - 15 - Reserved*/
} __wrsr_bits;

/* -------------------------------------------------------------------------*/
/*      AIPI Registers                                                      */
/* -------------------------------------------------------------------------*/
typedef struct { /* Peripheral Size Register 0 (0x10000000) Reset (0x00040304)*/
                 /* Peripheral Size Register 1 (0x10000004) Reset (0xFFFBFCFB)*/
                 /* Peripheral Access Register (0x10000008) Reset (0xFFFFFFFF)*/
__REG32 AIPI1_CONTROL  : 1;     /* Bit  0*/
__REG32 DMA            : 1;     /* Bit  1*/
__REG32 WDOG           : 1;     /* Bit  2*/
__REG32 GPT1           : 1;     /* Bit  3*/
__REG32 GPT2           : 1;     /* Bit  4*/
__REG32 GPT3           : 1;     /* Bit  5*/
__REG32 PWM            : 1;     /* Bit  6*/
__REG32 RTC            : 1;     /* Bit  7*/
__REG32 KPP            : 1;     /* Bit  8*/
__REG32 OWIRE          : 1;     /* Bit  9*/
__REG32 UART1          : 1;     /* Bit  10*/
__REG32 UART2          : 1;     /* Bit  11*/
__REG32 UART3          : 1;     /* Bit  12*/
__REG32 UART4          : 1;     /* Bit  13*/
__REG32 CSPI1          : 1;     /* Bit  14*/
__REG32 CSPI2          : 1;     /* Bit  15*/
__REG32 SSI1           : 1;     /* Bit  16*/
__REG32 SSI2           : 1;     /* Bit  17*/
__REG32 I2C1           : 1;     /* Bit  18*/
__REG32 SDHC1          : 1;     /* Bit  19*/
__REG32 SDHC2          : 1;     /* Bit  20*/
__REG32 GPIO           : 1;     /* Bit  21*/
__REG32 AUDMUX         : 1;     /* Bit  22*/
__REG32 CSPI3          : 1;     /* Bit  23*/
__REG32 MSHC           : 1;     /* Bit  24*/
__REG32 GPT5           : 1;     /* Bit  25*/
__REG32 GPT4           : 1;     /* Bit  26*/
__REG32 UART5          : 1;     /* Bit  27*/
__REG32 UART6          : 1;     /* Bit  28*/
__REG32 I2C2           : 1;     /* Bit  29*/
__REG32 SDHC3          : 1;     /* Bit  30*/
__REG32 GPT6           : 1;     /* Bit  31*/
} __aipi1_bits;

typedef struct { /* Peripheral Size Register 0 (0x10020000) Reset (0x3FFC0000)*/
                 /* Peripheral Size Register 1 (0x10020004) Reset (0xFFFFFFFF)*/
                 /* Peripheral Access Register (0x10020008) Reset (0xFFFFFFFF)*/
__REG32 AIPI2_CONTROL  : 1;     /* Bit  0*/
__REG32 LCDC           : 1;     /* Bit  1*/
__REG32 SLCDC          : 1;     /* Bit  2*/
__REG32 H264           : 1;     /* Bit  3*/
__REG32 USB2           : 1;     /* Bit  4*/
__REG32 SAHARA         : 1;     /* Bit  5*/
__REG32 EMMA           : 1;     /* Bit  6*/
__REG32 CRM            : 1;     /* Bit  7*/
__REG32 IIM            : 1;     /* Bit  8*/
__REG32                : 1;     /* Bit  9*/
__REG32 RTIC           : 1;     /* Bit  10*/
__REG32 FEC            : 1;     /* Bit  11*/
__REG32 SCCL           : 1;     /* Bit  12*/
__REG32 SCCH           : 1;     /* Bit  13*/
__REG32                :18;     /* Bits 14 - 31*/
} __aipi2_bits;

/***************************************************************************
 **
 **  Multi-layer AHB Crossbar Switch (MAX)
 **
 ***************************************************************************/
typedef struct { /*           Master Priority Register for Slave Port 0 (0x1003F000) Reset (0x76543210) */
                 /*           Master Priority Register for Slave Port 1 (0x1003F100) Reset (0x76543210) */
                 /*           Master Priority Register for Slave Port 2 (0x1003F200) Reset (0x76543210) */
                 /*           Master Priority Register for Slave Port 3 (0x1003F300) Reset (0x76543210) */
                 /* Alternate Master Priority Register for Slave Port 0 (0x1003F004) Reset (0x76543210) */
                 /* Alternate Master Priority Register for Slave Port 1 (0x1003F104) Reset (0x76543210) */
                 /* Alternate Master Priority Register for Slave Port 2 (0x1003F204) Reset (0x76543210) */
                 /* Alternate Master Priority Register for Slave Port 3 (0x1003F304) Reset (0x76543210) */
__REG32 MSTR_0  : 3;     /* Bits 0  - 2  - Master 0 Priority*/
__REG32         : 1;     /* Bit  3       - Reserved*/
__REG32 MSTR_1  : 3;     /* Bits 4  - 6  - Master 1 Priority*/
__REG32         : 1;     /* Bit  7       - Reserved*/
__REG32 MSTR_2  : 3;     /* Bits 8  - 10 - Master 2 Priority*/
__REG32         : 1;     /* Bit  11      - Reserved*/
__REG32 MSTR_3  : 3;     /* Bits 12 - 14 - Master 3 Priority*/
__REG32         : 1;     /* Bit  15      - Reserved*/
__REG32 MSTR_4  : 3;     /* Bits 16 - 18 - Master 4 Priority*/
__REG32         : 1;     /* Bit  19      - Reserved*/
__REG32 MSTR_5  : 3;     /* Bits 20 - 22 - Master 5 Priority*/
__REG32         : 9;     /* Bits 23 - 31 - Reserved*/
} __mpr_bits;

typedef struct { /* Slave General Purpose Control Register for Slave Port 0 (0x1003F010) Reset (0x00000000) */
                 /* Slave General Purpose Control Register for Slave Port 1 (0x1003F110) Reset (0x00000000) */
                 /* Slave General Purpose Control Register for Slave Port 2 (0x1003F210) Reset (0x00000000) */
                 /* Slave General Purpose Control Register for Slave Port 3 (0x1003F310) Reset (0x00000000) */
__REG32 PARK  : 3;     /* Bits 0  - 2  - PARK*/
__REG32       : 1;     /* Bit  3       - Reserved*/
__REG32 PCTL  : 2;     /* Bits 4  - 5  - Parking Control*/
__REG32       : 2;     /* Bits 6  - 7  - Reserved*/
__REG32 ARB   : 2;     /* Bits 8  - 9  - Arbitration Mode*/
__REG32       :20;     /* Bits 10 - 29 - Reserved*/
__REG32 HLP   : 1;     /* Bit  30      - Halt Low Priority*/
__REG32 RO    : 1;     /* Bit  31      - Read Only*/
} __sgpcr_bits;

typedef struct { /* Alternate Slave General Purpose Control Register for Slave Port 0 (0x1003F014) Reset (0x00000000) */
                 /* Alternate Slave General Purpose Control Register for Slave Port 1 (0x1003F114) Reset (0x00000000) */
                 /* Alternate Slave General Purpose Control Register for Slave Port 2 (0x1003F214) Reset (0x00000000) */
                 /* Alternate Slave General Purpose Control Register for Slave Port 3 (0x1003F314) Reset (0x00000000) */
__REG32 PARK  : 3;     /* Bits 0  - 2  - PARK*/
__REG32       : 1;     /* Bit  3       - Reserved*/
__REG32 PCTL  : 2;     /* Bits 4  - 5  - Parking Control*/
__REG32       : 2;     /* Bits 6  - 7  - Reserved*/
__REG32 ARB   : 2;     /* Bits 8  - 9  - Arbitration Mode*/
__REG32       :20;     /* Bits 10 - 29 - Reserved*/
__REG32 HLP   : 1;     /* Bit  30      - Halt Low Priority*/
__REG32       : 1;     /* Bit  31      - Reserved*/
} __asgpcr_bits;

typedef struct { /* Master General Purpose Control Register for Master Port 0 (0x1003F800) Reset (0x00000000) */
                 /* Master General Purpose Control Register for Master Port 1 (0x1003F900) Reset (0x00000000) */
                 /* Master General Purpose Control Register for Master Port 2 (0x1003FA00) Reset (0x00000000) */
                 /* Master General Purpose Control Register for Master Port 3 (0x1003FB00) Reset (0x00000000) */
                 /* Master General Purpose Control Register for Master Port 4 (0x1003FC00) Reset (0x00000000) */
                 /* Master General Purpose Control Register for Master Port 5 (0x1003FD00) Reset (0x00000000) */
__REG32 AULB  : 3;     /* Bits 0  - 2  - Arbitrate on Undefined Length Bursts*/
__REG32       :29;     /* Bits 3  - 31 - Reserved*/
} __mgpcr_bits;

/* -------------------------------------------------------------------------*/
/*               DMA Registers                                              */
/* -------------------------------------------------------------------------*/
typedef struct {        /* DMA Control Register (0x10001000) Reset (0x0000)                     */
__REG32 DEN   : 1;     /* Bit  0       - DMA Enable - Enables/Disables the system clock to the DMA module.*/
__REG32 DRST  : 1;     /* Bit  1       - DMA Reset - Writing "1" Generates a 3-cycle reset pulse that resets the entire DMA module*/
__REG32 DAM   : 1;     /* Bit  2       - DMA Access Mode—Specifies user or privileged access to be performed by DMA.*/
__REG32       :29;     /* Bits 31 - 3  - Reserved*/
} __dcr_bits;

typedef struct {        /* DMA Register Type                                                                            */
__REG32 CH0   : 1;     /* Bit 0        - Channel 0*/
__REG32 CH1   : 1;     /* Bit 1        - Channel 1*/
__REG32 CH2   : 1;     /* Bit 2        - Channel 2*/
__REG32 CH3   : 1;     /* Bit 3        - Channel 3*/
__REG32 CH4   : 1;     /* Bit 4        - Channel 4*/
__REG32 CH5   : 1;     /* Bit 5        - Channel 5*/
__REG32 CH6   : 1;     /* Bit 6        - Channel 6*/
__REG32 CH7   : 1;     /* Bit 7        - Channel 7*/
__REG32 CH8   : 1;     /* Bit 8        - Channel 8*/
__REG32 CH9   : 1;     /* Bit 9        - Channel 9*/
__REG32 CH10  : 1;     /* Bit 10       - Channel 10*/
__REG32 CH11  : 1;     /* Bit 11       - Channel 11*/
__REG32 CH12  : 1;     /* Bit 12       - Channel 12*/
__REG32 CH13  : 1;     /* Bit 13       - Channel 13*/
__REG32 CH14  : 1;     /* Bit 14       - Channel 14*/
__REG32 CH15  : 1;     /* Bit 15       - Channel 15*/
__REG32       :16;     /* Bits 31-16   - Reserved*/
} __disr_bits;

typedef struct {        /* DMA Burst Time-Out Control Register (0x1000101C) Reset (0x00000000) */
__REG32 CNT  :15;     /* Bits 14 - 0  - Count - Contains the time-out count down value.*/
__REG32 EN   : 1;     /* Bit  15      - Enable - (0 = burst time-out Disabled, 1 = burst time-out Enabled)*/
__REG32      :16;     /* Bits 31 - 16  - Reserved*/
} __dbtocr_bits;

typedef struct {        /* W-Size Register */
__REG32 WS  :16;     /* Bits 15-0    - W-Size - Contains the number of bytes that make up the display width.*/
__REG32     :16;     /* Bits 31-16   - Reserved*/
} __wsr_bits;

typedef struct {        /* X-Size Registers */
__REG32 XS  :16;     /* Bits 15-0    - X-Size - Contains the number of bytes per row that define the X-Size of the 2D memory.*/
__REG32     :16;     /* Bits 31-16   - Reserved*/
} __xsr_bits;

typedef struct {        /* Y-Size Registers */
__REG32 YS  :16;     /* Bits 15-0    - Y-Size - Contains the number of rows that make up the 2D memory window.*/
__REG32     :16;     /* Bits 31-16   - Reserved*/
} __ysr_bits;

typedef struct {        /* DMA Channel Count Register */
__REG32 CNT  :24;     /* Bits 23 - 0  - Count - Contains the number of bytes of data to be transferred during a DMA cycle.*/
__REG32      : 8;     /* Bits 31 - 24 - Reserved*/
} __cntr_bits;

typedef struct {        /* DMA Channel Control Register */
__REG32 CEN    : 1;     /* Bit  0       - DMA Channel Enable (0 = Disables, 1 = Enable)*/
__REG32 FRC    : 1;     /* Bit  1       - Forces a DMA Cycle to occur (0=No Effect, 1=Force DMA cycle)*/
__REG32 RPT    : 1;     /* Bit  2       - Repeat - Enables(1)/Disables(0) the data transfer repeat function.*/
__REG32 REN    : 1;     /* Bit  3       - Request Enable - Enables(1)/Disables(0) the DMA request signal. When REN is set, DMA is started by DMA_REQ[x] signal. When REN is cleared, DMA transfer is initiated by CEN*/
__REG32 SSIZ   : 2;     /* Bits 5  - 4  - Source Size - Selects the source size of data transfer. (00=32 Bit port, 01=8-bit port, 10=16-bit port, 11=reserved)*/
__REG32 DSIZ   : 2;     /* Bits 7  - 6  - Destination Size - Selects the destination size of a data (00=32 Bit port, 01=8-bit port, 10=16-bit port, 11=reserved)*/
__REG32 MSEL   : 1;     /* Bit  8       - Memory Select - Selects the 2D memory register set when either source and/or destination is programmed to 2D memory mode. (0=2D memory register set A, 1=2D mem reg set B)*/
__REG32 MDIR   : 1;     /* Bit  9       - Memory Direction - Selects the memory address direction. (0 = Memory address increment, 1 = Memory address decrement)*/
__REG32 SMOD   : 2;     /* Bits 11 - 10 - Source Mode - Selects the source transfer mode (00=Linear Memory, 01=2D Memory, 10=FIFO, 11=End-of-burst enable FIFO)*/
__REG32 DMOD   : 2;     /* Bits 13 - 12 - Destination Mode - Selects the destination transfer mode. (00=Linear Memory, 01=2D Memory, 10=FIFO, 11=End-of-burst enable FIFO)*/
__REG32 ACRPT  : 1;     /* Bits 14      - Auto Clear RPT—This bit is to be sampled at the end of the transfer along with the RPT bit (0 = Do not modify RPT, 1 = Reset RPT at end of current transfer.)*/
__REG32        :17;     /* Bits 31 - 15 - Reserved*/
} __ccr_bits;

typedef struct {        /* DMA Channel Request Source Select Register */
__REG32 RSS  : 6;     /* Bits 5  - 0  - Request Source Select (0=DMA_REQ[0]....31=DMA_REQ[31])*/
__REG32      :26;     /* Bits 31 - 6  - Reserved*/
} __rssr_bits;

typedef struct {        /* DMA Channel Burst Length Register */
__REG32 BL  : 6;     /* Bits 6  - 0  - Burst Length - Contains the number of data bytes transferred in a DMA burst. (0=64 bytes, 1=1 byte...63 = 63 bytes)*/
__REG32     :26;     /* Bits 31 - 6  - Reserved*/
} __blr_bits;

typedef union{
  /*RTORx*/
  struct{
__REG32 CNT     :13;     /* Bits 12 - 0  - Request Time-Out Count - Contains the time-out count down value for the internal counter.*/
__REG32 PSC     : 1;     /* Bit  13      - Prescaler Count (0=/1, 1=/256)*/
__REG32 CLK     : 1;     /* Bit  14      - Clock Source - Selects the counter of input clock source. (0 = HCLK, 1 = 32.768 kHz)*/
__REG32 EN      : 1;     /* Bit  15      - Enable - Enables/Disables the DMA request time-out.*/
__REG32         :16;     /* Bits 31 - 16 - Reserved*/
  };
  /*BUCRx*/
  struct {
__REG32 BU_CNT  :16;     /* Bits 15 - 0  - Clock Count - Sets the number of system clocks that must occur before the memory channel releases the AHB, before the next DMA request for the channel.*/
__REG32         :16;     /* Bits 31 - 16 - Reserved*/
  };
} __rtor_bits;

typedef struct {        /* DMA Channel Counter Register */
__REG32 CCNR  :24;     /* Bits 23 - 0  - Channel Counter—Indicates the number of bytes transferred for the channel.*/
__REG32       : 8;     /* Bits 31 - 24 - Reserved*/
} __ccnr_bits;

/* -------------------------------------------------------------------------*/
/*      Digital Audio Mux (AUDMUX)                                          */
/* -------------------------------------------------------------------------*/
/* AUDMUX Host Port Configuration Register */
typedef struct{
__REG32 INMMASK  : 8;
__REG32 INMEN    : 1;
__REG32          : 1;
__REG32 TXRXEN   : 1;
__REG32          : 1;
__REG32 SYN      : 1;
__REG32 RXDSEL   : 3;
__REG32          : 4;
__REG32 RFCSEL   : 4;
__REG32 RCLKDIR  : 1;
__REG32 RFSDIR   : 1;
__REG32 TFCSEL   : 4;
__REG32 TCLKDIR  : 1;
__REG32 TFSDIR   : 1;
} __hpcr_bits;

/* AUDMUX Peripheral Port Configuration Register */
typedef struct{
__REG32          :10;
__REG32 TXRXEN   : 1;
__REG32          : 1;
__REG32 SYN      : 1;
__REG32 RXDSEL   : 3;
__REG32          : 4;
__REG32 RFCSEL   : 4;
__REG32 RCLKDIR  : 1;
__REG32 RFSDIR   : 1;
__REG32 TFCSEL   : 4;
__REG32 TCLKDIR  : 1;
__REG32 TFSDIR   : 1;
} __ppcr123_bits;

/* -------------------------------------------------------------------------*/
/*               CSI  registers                                             */
/* -------------------------------------------------------------------------*/
/* CSI Control Register 1 */
typedef struct{
__REG32               : 1;
__REG32 REDGE         : 1;
__REG32 INV_PCLK      : 1;
__REG32 INV_DATA      : 1;
__REG32 GCLK_MODE     : 1;
__REG32 CLR_RXFIFO    : 1;
__REG32 CLR_STATFIFO  : 1;
__REG32 PACK_DIR      : 1;
__REG32 FCC           : 1;
__REG32 MCLKEN        : 1;
__REG32 CCIR_EN       : 1;
__REG32 HSYNC_POL     : 1;
__REG32 MCLKDIV       : 4;
__REG32 SOF_INTEN     : 1;
__REG32 SOF_POL       : 1;
__REG32 RXFF_INTEN    : 1;
__REG32 RXFF_LEVEL    : 2;
__REG32 STATFF_INTEN  : 1;
__REG32 STATFF_LEVEL  : 2;
__REG32 RF_OR_INTEN   : 1;
__REG32 SF_OR_INTEN   : 1;
__REG32 COF_INT_E     : 1;
__REG32 CCIR_MODE     : 1;
__REG32 PRP_IF_EN     : 1;
__REG32 EOF_INT_EN    : 1;
__REG32 EXT_VSYNC     : 1;
__REG32 SWAP16_EN     : 1;
} __csicr1_bits;

/* CSI Control Register 2 */
typedef struct{
__REG32 HSC   : 8;
__REG32 VSC   : 8;
__REG32 LVRM  : 3;
__REG32 BTS   : 2;
__REG32       : 2;
__REG32 SCE   : 1;
__REG32 AFS   : 2;
__REG32 DRM   : 1;
__REG32       : 5;
} __csicr2_bits;

/* CSI Control Register 3 */
typedef struct{
__REG32 ECC_AUTO_EN   : 1;
__REG32 ECC_INT_EN    : 1;
__REG32 ZERO_PACK_EN  : 1;
__REG32 CSI_SVR       : 1;
__REG32               :11;
__REG32 FRMCNT_RST    : 1;
__REG32 FRMCNT        :16;
} __csicr3_bits;

/* CSI Status Register */
typedef struct{
__REG32 DRDY        : 1;
__REG32 ECC_INT     : 1;
__REG32             :11;
__REG32 COF_INT     : 1;
__REG32 F1_INT      : 1;
__REG32 F2_INT      : 1;
__REG32 SOF_INT     : 1;
__REG32 EOF_INT     : 1;
__REG32 RXFF_INT    : 1;
__REG32             : 2;
__REG32 STATFF_INT  : 1;
__REG32             : 2;
__REG32 RFF_OR_INT  : 1;
__REG32 SFF_OR_INT  : 1;
__REG32             : 6;
} __csisr_bits;

/* CSI RX Count Register */
typedef struct{
__REG32 RXCNT  :22;
__REG32        :10;
} __csirxcnt_bits;

/* -------------------------------------------------------------------------*/
/*      Video Codec                                                         */
/* -------------------------------------------------------------------------*/
/* Video Codec BIT run start */
typedef struct{
__REG32 CodeRun       : 1;
__REG32               :31;
} __vcCodeRun_bits;

/* Video Codec Code Download Data Register */
typedef struct{
__REG32 CodeData      :16;
__REG32 CodeAddr      :13;
__REG32               : 3;
} __vcCodeDownLoad_bits;

/* Video Codec Host Interrupt Request to BI */
typedef struct{
__REG32 IntReq        : 1;
__REG32               :31;
} __vcHostIntReq_bits;

/* Video Codec BIT Interrupt Clear */
typedef struct{
__REG32 IntClear      : 1;
__REG32               :31;
} __vcBitIntClear_bits;

/* Video Codec BIT Interrupt Status */
typedef struct{
__REG32 IntSts        : 1;
__REG32               :31;
} __vcBitIntSts_bits;

/* Video Codec BIT Code Reset */
typedef struct{
__REG32 CodeReset     : 1;
__REG32               :31;
} __vcBitCodeReset_bits;

/* Video Codec BIT Current PC */
typedef struct{
__REG32 CurPc         :14;
__REG32               :18;
} __vcBitCurPc_bits;

/* -------------------------------------------------------------------------*/
/*      enhanced Multimedia Accelerator (eMMA) Post-Processor               */
/* -------------------------------------------------------------------------*/
/* PP Control Register */
typedef struct{
__REG32 PP_EN          : 1;
__REG32 DEBLOCKEN      : 1;
__REG32 DERINGEN       : 1;
__REG32                : 1;
__REG32 CSCEN          : 1;
__REG32 CSC_TABLE_SEL  : 2;
__REG32                : 1;
__REG32 SWRST          : 1;
__REG32 MB_MODE        : 1;
__REG32 CSC_OUT        : 2;
__REG32 BSDI           : 1;
__REG32                :19;
} __pp_cntl_bits;

/* PP Interrupt Control Register */
typedef struct{
__REG32 FRAME_COMP_INTR_EN  : 1;
__REG32                     : 1;
__REG32 ERR_INTR_EN         : 1;
__REG32                     :29;
} __pp_intrcntl_bits;

/* PP Interrupt Status Register */
typedef struct{
__REG32 FRAME_COMP_INTR : 1;
__REG32                 : 1;
__REG32 ERR_INTR        : 1;
__REG32                 :29;
} __pp_intrstatus_bits;

/* PP Process Frame Parameter Register */
typedef struct{
__REG32 PROCESS_FRAME_HEIGHT  :10;
__REG32                       : 6;
__REG32 PROCESS_FRAME_WIDTH   :10;
__REG32                       : 6;
} __pp_process_para_bits;

/* PP Source Frame Width Register */
typedef struct{
__REG32 Y_INPUT_LINE_STRIDE    :12;
__REG32                        : 4;
__REG32 QUANTIZER_FRAME_WIDTH  : 8;
__REG32                        : 8;
} __pp_frame_width_bits;

/* PP Destination Display Width Register */
typedef struct{
__REG32 OUTPUT_LINE_STRIDE  :13;
__REG32                     :19;
} __pp_display_width_bits;

/* PP Destination Image Size Register */
typedef struct{
__REG32 OUT_IMAGE_HEIGHT  :12;
__REG32                   : 4;
__REG32 OUT_IMAGE_WIDTH   :12;
__REG32                   : 4;
} __pp_image_size_bits;

/* PP Destination Frame Format Control Register */
typedef struct{
__REG32 BLUE_WIDTH    : 4;
__REG32 GREEN_WIDTH   : 4;
__REG32 RED_WIDTH     : 4;
__REG32               : 4;
__REG32 BLUE_OFFSET   : 5;
__REG32 GREEN_OFFSET  : 5;
__REG32 RED_OFFSET    : 5;
__REG32               : 1;
} __pp_dest_frame_format_cntl_bits;

/* PP Resize Table Index Register */
typedef struct{
__REG32 VERT_TBL_END_INDEX    : 6;
__REG32                       : 2;
__REG32 VERT_TBL_START_INDEX  : 6;
__REG32                       : 2;
__REG32 HORI_TBL_END_INDEX    : 6;
__REG32                       : 2;
__REG32 HORI_TBL_START_INDEX  : 6;
__REG32                       : 2;
} __pp_resize_index_bits;

/* PP CSC Coefficient_123 Register */
typedef struct{
__REG32 C3  : 8;
__REG32 C2  : 8;
__REG32 C1  : 8;
__REG32 C0  : 8;
} __pp_csc_coef_123_bits;

/* PP CSC Coefficient_4 Register */
typedef struct{
__REG32 C4  : 9;
__REG32 X0  : 1;
__REG32     :22;
} __pp_csc_coef_4_bits;

/* PP Resize Coefficient Table */
typedef struct{
__REG32 OP  : 1;
__REG32 N   : 2;
__REG32 W   : 5;
__REG32     :24;
} __pp_resize_coef_tbl_bits;

/* -------------------------------------------------------------------------*/
/*      enhanced Multimedia Accelerator (eMMA) Pre-Processor                */
/* -------------------------------------------------------------------------*/
/* PrP Control Register */
typedef struct{
__REG32 CH1EN             : 1;
__REG32 CH2EN             : 1;
__REG32 CSIEN             : 1;
__REG32 DATA_IN_MODE      : 2;
__REG32 CH1_OUT_MODE      : 2;
__REG32 CH2_OUT_MODE      : 2;
__REG32 CH1_LEN           : 1;
__REG32 CH2_LEN           : 1;
__REG32 SKIP_FRAME        : 1;
__REG32 SWRST             : 1;
__REG32 CLKEN             : 1;
__REG32 WEN               : 1;
__REG32 CH1BYP            : 1;
__REG32 IN_TSKIP          : 3;
__REG32 CH1_TSKIP         : 3;
__REG32 CH2_TSKIP         : 3;
__REG32 INPUT_FIFO_LEVEL  : 2;
__REG32 RZ_FIFO_LEVEL     : 2;
__REG32 CH2B1EN           : 1;
__REG32 CH2B2EN           : 1;
__REG32 CH2FEN            : 1;
} __prp_cntl_bits;

/* PrP Interrupt Control Register */
typedef struct{
__REG32 RDERRIE    : 1;
__REG32 CH1WERRIE  : 1;
__REG32 CH2WERRIE  : 1;
__REG32 CH1FCIE    : 1;
__REG32            : 1;
__REG32 CH2FCIE    : 1;
__REG32            : 1;
__REG32 LBOVFIE    : 1;
__REG32 CH2OVFIE   : 1;
__REG32            :23;
} __prp_intr_cntl_bits;

/* PrP Interrupt Status Register */
typedef struct{
__REG32 READERR   : 1;
__REG32 CH1WRERR  : 1;
__REG32 CH2WRERR  : 1;
__REG32 CH2B2CI   : 1;
__REG32 CH2B1CI   : 1;
__REG32 CH1B2CI   : 1;
__REG32 CH1B1CI   : 1;
__REG32 LBOVF     : 1;
__REG32 CH2OVF    : 1;
__REG32           :23;
} __prp_intrstatus_bits;

/* PrP Source Frame Size Register */
typedef struct{
__REG32 PICTURE_Y_SIZE  :11;
__REG32                 : 5;
__REG32 PICTURE_X_SIZE  :11;
__REG32                 : 5;
} __prp_src_frame_size_bits;

/* PrP Destination Channel-1 Line Stride Register */
typedef struct{
__REG32 CH1_OUT_LINE_STRIDE  :12;
__REG32                      :20;
} __prp_dest_ch1_line_stride_bits;

/* PrP Source Pixel Format Control Register */
typedef struct{
__REG32 BLUE_WIDTH         : 4;
__REG32 GREEN_WIDTH        : 4;
__REG32 RED_WIDTH          : 4;
__REG32                    : 4;
__REG32 BLUE_V_CR_OFFSET   : 5;
__REG32 GREEN_U_CB_OFFSET  : 5;
__REG32 RED_Y_OFFSET       : 5;
__REG32                    : 1;
} __prp_src_pixel_format_cntl_bits;

/* PrP Channel-1 Pixel Format Control Register */
typedef struct{
__REG32 BLUE_WIDTH    : 4;
__REG32 GREEN_WIDTH   : 4;
__REG32 RED_WIDTH     : 4;
__REG32               : 4;
__REG32 BLUE_OFFSET   : 5;
__REG32 GREEN_OFFSET  : 5;
__REG32 RED_OFFSET    : 5;
__REG32               : 1;
} __prp_ch1_pixel_format_cntl_bits;

/* PrP Destination Channel-1 Output Image Size Register */
typedef struct{
__REG32 CH1_OUT_IMAGE_HEIGHT  :11;
__REG32                       : 5;
__REG32 CH1_OUT_IMAGE_WIDTH   :11;
__REG32                       : 5;
} __prp_ch1_out_image_size_bits;

/* PrP Destination Channel-2 Output Image Size Register */
typedef struct{
__REG32 CH2_OUT_IMAGE_HEIGHT  :11;
__REG32                       : 5;
__REG32 CH2_OUT_IMAGE_WIDTH   :11;
__REG32                       : 5;
} __prp_ch2_out_image_size_bits;

/* PrP Source Line Stride Register */
typedef struct{
__REG32 SOURCE_LINE_STRIDE  :13;
__REG32                     : 3;
__REG32 CSI_LINE_SKIP       :13;
__REG32                     : 3;
} __prp_src_line_stride_bits;

/* PrP CSC Coefficient 012 */
typedef struct{
__REG32 C2  : 8;
__REG32     : 3;
__REG32 C1  : 8;
__REG32     : 2;
__REG32 C0  : 8;
__REG32     : 3;
} __prp_csc_coef_012_bits;

/* PrP CSC coefficient 345 */
typedef struct{
__REG32 C5  : 7;
__REG32     : 4;
__REG32 C4  : 9;
__REG32     : 1;
__REG32 C3  : 8;
__REG32     : 3;
} __prp_csc_coef_345_bits;

/* PrP CSC Coefficient 678 */
typedef struct{
__REG32 C8  : 7;
__REG32     : 4;
__REG32 C7  : 7;
__REG32     : 3;
__REG32 C6  : 7;
__REG32     : 3;
__REG32 X0  : 1;
} __prp_csc_coef_678_bits;

/* PrP Horizontal Resize Coefficient-1 */
typedef struct{
__REG32 HC0  : 3;
__REG32 HC1  : 3;
__REG32 HC2  : 3;
__REG32 HC3  : 3;
__REG32 HC4  : 3;
__REG32      : 1;
__REG32 HC5  : 3;
__REG32 HC6  : 3;
__REG32 HC7  : 3;
__REG32 HC8  : 3;
__REG32 HC9  : 3;
__REG32      : 1;
} __prp_rz_hori_coef1_bits;

/* PrP Horizontal Resize Coefficient-2 */
typedef struct{
__REG32 HC10  : 3;
__REG32 HC11  : 3;
__REG32 HC12  : 3;
__REG32 HC13  : 3;
__REG32 HC14  : 3;
__REG32       : 1;
__REG32 HC15  : 3;
__REG32 HC16  : 3;
__REG32 HC17  : 3;
__REG32 HC18  : 3;
__REG32 HC19  : 3;
__REG32       : 1;
} __prp_rz_hori_coef2_bits;

/* PrP Horizontal Resize Valid */
typedef struct{
__REG32 HOV           :20;
__REG32               : 4;
__REG32 HORI_TBL_LEN  : 5;
__REG32               : 2;
__REG32 AVG_BIL       : 1;
} __prp_rz_hori_valid_bits;

/* PrP Vertical Resize Coefficient-1 */
typedef struct{
__REG32 VC0  : 3;
__REG32 VC1  : 3;
__REG32 VC2  : 3;
__REG32 VC3  : 3;
__REG32 VC4  : 3;
__REG32      : 1;
__REG32 VC5  : 3;
__REG32 VC6  : 3;
__REG32 VC7  : 3;
__REG32 VC8  : 3;
__REG32 VC9  : 3;
__REG32      : 1;
} __prp_rz_vert_coef1_bits;

/* PrP Vertical Resize Coefficient-2 */
typedef struct{
__REG32 VC10  : 3;
__REG32 VC11  : 3;
__REG32 VC12  : 3;
__REG32 VC13  : 3;
__REG32 VC14  : 3;
__REG32       : 1;
__REG32 VC15  : 3;
__REG32 VC16  : 3;
__REG32 VC17  : 3;
__REG32 VC18  : 3;
__REG32 VC19  : 3;
__REG32       : 1;
} __prp_rz_vert_coef2_bits;

/* PrP Channel 1 Vertical Resize Valid */
typedef struct{
__REG32 VOV           :20;
__REG32               : 4;
__REG32 VERT_TBL_LEN  : 5;
__REG32               : 2;
__REG32 AVG_BIL       : 1;
} __prp_rz_vert_valid_bits;

/* -------------------------------------------------------------------------*/
/*               SSI registers                                              */
/* -------------------------------------------------------------------------*/
/* SSI Control/Status Register */
typedef struct{
__REG32 SSIEN       : 1;
__REG32 TE          : 1;
__REG32 RE          : 1;
__REG32 NET         : 1;
__REG32 SYN         : 1;
__REG32 I2S_MODE    : 2;
__REG32 SYS_CLK_EN  : 1;
__REG32 TCH_EN      : 1;
__REG32 CLK_IST     : 1;
__REG32             :22;
} __scsr_bits;

/* SSI Interrupt Status Register */
typedef struct{
__REG32 TFE0   : 1;
__REG32 TFE1   : 1;
__REG32 RFF0   : 1;
__REG32 RFF1   : 1;
__REG32 RLS    : 1;
__REG32 TLS    : 1;
__REG32 RFS    : 1;
__REG32 TFS    : 1;
__REG32 TUE0   : 1;
__REG32 TUE1   : 1;
__REG32 ROE0   : 1;
__REG32 ROE1   : 1;
__REG32 TDE0   : 1;
__REG32 TDE1   : 1;
__REG32 RDR0   : 1;
__REG32 RDR1   : 1;
__REG32 RXT    : 1;
__REG32 CMDDU  : 1;
__REG32 CMDAU  : 1;
__REG32        :13;
} __sisr_bits;

/* SSI Interrupt Enable Register */
typedef struct{
__REG32 TFE0_EN   : 1;
__REG32 TFE1_EN   : 1;
__REG32 RFF0_EN   : 1;
__REG32 RFF1_EN   : 1;
__REG32 RLS_EN    : 1;
__REG32 TLS_EN    : 1;
__REG32 RFS_EN    : 1;
__REG32 TFS_EN    : 1;
__REG32 TUE0_EN   : 1;
__REG32 TUE1_EN   : 1;
__REG32 ROE0_EN   : 1;
__REG32 ROE1_EN   : 1;
__REG32 TDE0_EN   : 1;
__REG32 TDE1_EN   : 1;
__REG32 RDR0_EN   : 1;
__REG32 RDR1_EN   : 1;
__REG32 RXT_EN    : 1;
__REG32 CMDDU_EN  : 1;
__REG32 CMDAU_EN  : 1;
__REG32 TIE       : 1;
__REG32 TDMAE     : 1;
__REG32 RIE       : 1;
__REG32 RDMAE     : 1;
__REG32           : 9;
} __sier_bits;

/* SSI Transmit Configuration Register */
typedef struct{
__REG32 TEFS    : 1;
__REG32 TFSL    : 1;
__REG32 TFSI    : 1;
__REG32 TSCKP   : 1;
__REG32 TSHFD   : 1;
__REG32 TXDIR   : 1;
__REG32 TFDIR   : 1;
__REG32 TFEN0   : 1;
__REG32 TFEN1   : 1;
__REG32 TXBIT0  : 1;
__REG32         :22;
} __stcr_bits;

/* SSI Receive Configuration Register */
typedef struct{
__REG32 REFS    : 1;
__REG32 RFSL    : 1;
__REG32 RFSI    : 1;
__REG32 RSCKP   : 1;
__REG32 RSHFD   : 1;
__REG32 RXDIR   : 1;
__REG32 RFDIR   : 1;
__REG32 RFEN0   : 1;
__REG32 RFEN1   : 1;
__REG32 RXBIT0  : 1;
__REG32 RXEXT   : 1;
__REG32         :21;
} __srcr_bits;

/* SSI Clock Control Register */
typedef struct{
__REG32 PM    : 8;
__REG32 DC    : 5;
__REG32 WL    : 4;
__REG32 PSR   : 1;
__REG32 DIV2  : 1;
__REG32       :13;
} __ssi_ccr_bits;

/* SSI FIFO Control/Status Register */
typedef struct{
__REG32 TFWM0   : 4;
__REG32 RFWM0   : 4;
__REG32 TFCNT0  : 4;
__REG32 RFCNT0  : 4;
__REG32 TFWM1   : 4;
__REG32 RFWM1   : 4;
__REG32 TFCNT1  : 4;
__REG32 RFCNT1  : 4;
} __ssi_sfcsr_bits;

/* SSI Test Register */
typedef struct{
__REG32 TXSTATE  : 5;
__REG32 TFS2RFS  : 1;
__REG32 TCK2RCK  : 1;
__REG32 TXD2RXD  : 1;
__REG32 RXSTATE  : 5;
__REG32 RFS2TFS  : 1;
__REG32 RCK2TCK  : 1;
__REG32 TEST     : 1;
__REG32          :16;
} __ssi_str_bits;

/* SSI Option Register */
typedef struct{
__REG32 SYNRST  : 1;
__REG32 WAIT    : 2;
__REG32 INIT    : 1;
__REG32 TX_CLR  : 1;
__REG32 RX_CLR  : 1;
__REG32 CLKOFF  : 1;
__REG32         :25;
} __ssi_sor_bits;

/* SSI AC97 Control Register */
typedef struct{
__REG32 A97EN  : 1;
__REG32 FV     : 1;
__REG32 TIF    : 1;
__REG32 RD     : 1;
__REG32 WR     : 1;
__REG32 FRDIV  : 6;
__REG32        :21;
} __ssi_sacnt_bits;

/* SSI AC97 Command Address Register */
typedef struct{
__REG32 SACADD  :19;
__REG32         :13;
} __ssi_sacadd_bits;

/* SSI AC97 Command Data Register */
typedef struct{
__REG32 SACDAT  :19;
__REG32         :13;
} __ssi_sacdat_bits;

/* SSI AC97 Tag Register */
typedef struct{
__REG32 SATAG  :16;
__REG32        :16;
} __ssi_satag_bits;

/* SSI AC97 Channel Status Register (SACCST) */
typedef struct{
__REG32 SACCST :10;
__REG32        :22;
} __ssi_saccst_bits;

/* SSI AC97 Channel Enable Register (SACCEN)*/
typedef struct{
__REG32 SACCEN :10;
__REG32        :22;
} __ssi_saccen_bits;

/* SSI AC97 Channel Disable Register (SACCDIS) */
typedef struct{
__REG32 SACCDIS :10;
__REG32         :22;
} __ssi_saccdis_bits;

/* -------------------------------------------------------------------------*/
/*      LCDC Registers                                                       */
/* -------------------------------------------------------------------------*/
/* LCDC Size Register */
typedef struct{
__REG32 YMAX  :10;
__REG32       :10;
__REG32 XMAX  : 6;
__REG32       : 2;
__REG32 BS    : 1;
__REG32       : 3;
} __lsr_bits;

/* LCDC Virtual Page Width Register */
typedef struct{
__REG32 VPW  :10;
__REG32      :22;
} __lvpwr_bits;

/* LCDC Panel Configuration Register */
typedef struct{
__REG32 PCD        : 6;
__REG32 SHARP      : 1;
__REG32 SCLK_SEL   : 1;
__REG32 ACD        : 7;
__REG32 ACD_SEL    : 1;
__REG32 REV_VS     : 1;
__REG32 SWAP_SEL   : 1;
__REG32 END_SEL    : 1;
__REG32 SCLK_IDLE  : 1;
__REG32 OE_POL     : 1;
__REG32 CLK_POL    : 1;
__REG32 LP_POL     : 1;
__REG32 FLM_POL    : 1;
__REG32 PIX_POL    : 1;
__REG32 BPIX       : 3;
__REG32 PBSIZ      : 2;
__REG32 COLOR      : 1;
__REG32 TFT        : 1;
} __lpcr_bits;

/* LCDC Horizontal Configuration Register */
typedef struct{
__REG32 H_WAIT_2  : 8;
__REG32 H_WAIT_1  : 8;
__REG32           :10;
__REG32 H_WIDTH   : 6;
} __lhcr_bits;

/* LCDC Vertical Configuration Register */
typedef struct{
__REG32 V_WAIT_2  : 8;
__REG32 V_WAIT_1  : 8;
__REG32           :10;
__REG32 V_WIDTH   : 6;
} __lvcr_bits;

/* LCDC Panning Offset Register */
typedef struct{
__REG32 POS  : 5;
__REG32      :27;
} __lpor_bits;

/* LCDC Cursor Position Register */
typedef struct{
__REG32 CYP  :10;
__REG32      : 6;
__REG32 CXP  :10;
__REG32      : 2;
__REG32 OP   : 1;
__REG32      : 1;
__REG32 CC   : 2;
} __lcpr_bits;

/* LCDC Cursor Width Height and Blink Register */
typedef struct{
__REG32 BD     : 8;
__REG32        : 8;
__REG32 CH     : 5;
__REG32        : 3;
__REG32 CW     : 5;
__REG32        : 2;
__REG32 BK_EN  : 1;
} __lcwhb_bits;

/* LCDC Color Cursor Mapping Register */
typedef struct{
__REG32 CUR_COL_B  : 6;
__REG32 CUR_COL_G  : 6;
__REG32 CUR_COL_R  : 6;
__REG32            :14;
} __lccmr_bits;

/* LCDC Sharp Configuration Register */
typedef struct{
__REG32 GRAY1             : 4;
__REG32 GRAY2             : 4;
__REG32 REV_TOGGLE_DELAY  : 4;
__REG32                   : 4;
__REG32 CLS_RISE_DELAY    : 8;
__REG32                   : 2;
__REG32 PS_RISE_DELAY     : 6;
} __lscr_bits;

/* LCDC PWM Contrast Control Register */
typedef struct{
__REG32 PW            : 8;
__REG32 CC_EN         : 1;
__REG32 SCR           : 2;
__REG32               : 4;
__REG32 LDMSK         : 1;
__REG32 CLS_HI_WIDTH  : 9;
__REG32               : 7;
} __lpccr_bits;

/* LCDC Refresh Mode Control Register */
typedef struct{
__REG32 SELF_REF  : 1;
__REG32           :31;
} __lrmcr_bits;

/* LCDC Graphic Window DMA Control Register */
typedef struct{
__REG32 TM     : 5;
__REG32        :11;
__REG32 HM     : 5;
__REG32        :10;
__REG32 BURST  : 1;
} __ldcr_bits;

/* LCDC Interrupt Configuration Register */
typedef struct{
__REG32 INTCON      : 1;
__REG32             : 1;
__REG32 INTSYN      : 1;
__REG32             : 1;
__REG32 GW_INT_CON  : 1;
__REG32             :27;
} __licr_bits;

/* LCDC Interrupt Enable Register */
typedef struct{
__REG32 BOF_EN         : 1;
__REG32 EOF_EN         : 1;
__REG32 ERR_RES_EN     : 1;
__REG32 UDR_ERR_EN     : 1;
__REG32 GW_BOF_EN      : 1;
__REG32 GW_EOF_EN      : 1;
__REG32 GW_ERR_RES_EN  : 1;
__REG32 GW_UDR_ERR_EN  : 1;
__REG32                :24;
} __lier_bits;

/* LCDC Interrupt Status Register */
typedef struct{
__REG32 BOF         : 1;
__REG32 EOFR        : 1;
__REG32 ERR_RES     : 1;
__REG32 UDR_ERR     : 1;
__REG32 GW_BOF      : 1;
__REG32 GW_EOF      : 1;
__REG32 GW_ERR_RES  : 1;
__REG32 GW_UDR_ERR  : 1;
__REG32             :24;
} __lisr_bits;

/* LCDC Graphic Window Size Register */
typedef struct{
__REG32 GWH  :10;
__REG32      :10;
__REG32 GWW  : 6;
__REG32      : 6;
} __lgwsr_bits;

/* LCDC Graphic Window Virtual Page Width Register */
typedef struct{
__REG32 GWVPW  :10;
__REG32        :22;
} __lgwvpwr_bits;

/* LCDC Graphic Window Panning Offset Register */
typedef struct{
__REG32 GWPO  : 5;
__REG32       :27;
} __lgwpor_bits;

/* LCDC Graphic Window Position Register */
typedef struct{
__REG32 GWYP  :10;
__REG32       : 6;
__REG32 GWXP  :10;
__REG32       : 6;
} __lgwpr_bits;

/* LCDC Graphic Window Control Register */
typedef struct{
__REG32 GWCKB   : 6;
__REG32 GWCKG   : 6;
__REG32 GWCKR   : 6;
__REG32         : 3;
__REG32 GW_RVS  : 1;
__REG32 GWE     : 1;
__REG32 GWCKE   : 1;
__REG32 GWAV    : 8;
} __lgwcr_bits;

/* LCDC Graphic Window Graphic Window DMA Control Register */
typedef struct{
__REG32 GWTM  : 7;
__REG32       : 9;
__REG32 GWHM  : 7;
__REG32       : 8;
__REG32 GWBT  : 1;
} __lgwdcr_bits;

/* LCDC AUS Mode Control Register */
typedef struct{
__REG32 AGWCKB    : 8;
__REG32 AGWCKG    : 8;
__REG32 AGWCKR    : 8;
__REG32           : 7;
__REG32 AUS_MODE  : 1;
} __lauscr_bits;

/* LCDC AUS Mode Cursor Control Register */
typedef struct{
__REG32 ACUR_COL_B  : 8;
__REG32 ACUR_COL_G  : 8;
__REG32 ACUR_COL_R  : 8;
__REG32             : 8;
} __lausccr_bits;

/* -------------------------------------------------------------------------*/
/*      Smart Liquid Crystal Display Controller (SLCDC)                     */
/* -------------------------------------------------------------------------*/
/* SLCDC Data Buffer Size Register */
typedef struct{
__REG32 DATABUFSIZE  :17;
__REG32              :15;
} __data_buff_size_bits;

/* SLCDC Command Buffer Size Register */
typedef struct{
__REG32 COMBUFSIZE  :17;
__REG32             :15;
} __cmd_buff_size_bits;

/* SLCDC Command String Size Register */
typedef struct{
__REG32 COMSTRINGSIZ  : 8;
__REG32               :24;
} __string_size_bits;

/* SLCDC FIFO Configuration Register */
typedef struct{
__REG32 BURST  : 3;
__REG32        :29;
} __fifo_config_bits;

/* SLCDC LCD Controller Configuration Register */
typedef struct{
__REG32 WORDPPAGE  :13;
__REG32            :19;
} __lcd_config_bits;

/* SLCDC LCD Transfer Configuration Register */
typedef struct{
__REG32 SKCPOL        : 1;
__REG32 CSPOL         : 1;
__REG32 XFRMODE       : 1;
__REG32 WORDDEFCOM    : 1;
__REG32 WORDDEFDAT    : 1;
__REG32 WORDDEFWRITE  : 1;
__REG32               :10;
__REG32 IMGEND        : 2;
__REG32               :14;
} __lcdtransconfig_bits;

/* SLCDC Control/Status Register */
typedef struct{
__REG32 GO        : 1;
__REG32 ABORT     : 1;
__REG32 BUSY      : 1;
__REG32           : 1;
__REG32 TEA       : 1;
__REG32 UNDRFLOW  : 1;
__REG32 IRQ       : 1;
__REG32 IRQEN     : 1;
__REG32 PROT1     : 1;
__REG32           : 2;
__REG32 AUTOMODE  : 2;
__REG32           :19;
} __dma_ctrl_stat_bits;

/* SLCDC LCD Clock Configuration Register */
typedef struct{
__REG32 DIVIDE  : 6;
__REG32         :26;
} __lcd_clk_config_bits;

/* SLCDC LCD Write Data Register */
typedef struct{
__REG32 LCDDAT  :16;
__REG32 RS      : 1;
__REG32         :15;
} __lcd_write_data_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Common declarations  ****************************************************/
/***************************************************************************
 **
 **  PLLCLK
 **
 ***************************************************************************/
__IO_REG32_BIT(CSCR,                      0x10027000,__READ_WRITE,__cscr_bits);
__IO_REG32_BIT(MPCTL0,                    0x10027004,__READ_WRITE,__mpctl0_bits);
__IO_REG32_BIT(MPCTL1,                    0x10027008,__READ_WRITE,__mpctl1_bits);
__IO_REG32_BIT(SPCTL0,                    0x1002700C,__READ_WRITE,__mpctl0_bits);
__IO_REG32_BIT(SPCTL1,                    0x10027010,__READ_WRITE,__spctl1_bits);
__IO_REG32_BIT(OSC26MCTL,                 0x10027014,__READ_WRITE,__osc26mctl_bits);
__IO_REG32_BIT(PCDR0,                     0x10027018,__READ_WRITE,__pcdr0_bits);
__IO_REG32_BIT(PCDR1,                     0x1002701C,__READ_WRITE,__pcdr1_bits);
__IO_REG32_BIT(PCCR0,                     0x10027020,__READ_WRITE,__pccr0_bits);
__IO_REG32_BIT(PCCR1,                     0x10027024,__READ_WRITE,__pccr1_bits);
__IO_REG32_BIT(CCSR,                      0x10027028,__READ_WRITE,__ccsr_bits);
__IO_REG32_BIT(WKGDCTL,                   0x10027034,__READ_WRITE,__wkgdctl_bits);

/***************************************************************************
 **
 **  SYS CTRL
 **
 ***************************************************************************/
__IO_REG32_BIT(CID,                       0x10027800,__READ      ,__cid_bits);
__IO_REG32_BIT(FMCR,                      0x10027814,__READ_WRITE,__fmcr_bits);
__IO_REG32_BIT(GPCR,                      0x10027818,__READ_WRITE,__gpcr_bits);
__IO_REG32_BIT(WBCR,                      0x1002781C,__READ_WRITE,__wbcr_bits);
__IO_REG32_BIT(DSCR1,                     0x10027820,__READ_WRITE,__dscr1_bits);
__IO_REG32_BIT(DSCR2,                     0x10027824,__READ_WRITE,__dscr2_bits);
__IO_REG32_BIT(DSCR3,                     0x10027828,__READ_WRITE,__dscr3_bits);
__IO_REG32_BIT(DSCR4,                     0x1002782C,__READ_WRITE,__dscr4_bits);
__IO_REG32_BIT(DSCR5,                     0x10027830,__READ_WRITE,__dscr5_bits);
__IO_REG32_BIT(DSCR6,                     0x10027834,__READ_WRITE,__dscr6_bits);
__IO_REG32_BIT(DSCR7,                     0x10027838,__READ_WRITE,__dscr7_bits);
__IO_REG32_BIT(DSCR8,                     0x1002783C,__READ_WRITE,__dscr8_bits);
__IO_REG32_BIT(DSCR9,                     0x10027840,__READ_WRITE,__dscr9_bits);
__IO_REG32_BIT(DSCR10,                    0x10027844,__READ_WRITE,__dscr10_bits);
__IO_REG32_BIT(DSCR11,                    0x10027848,__READ_WRITE,__dscr11_bits);
__IO_REG32_BIT(DSCR12,                    0x1002784C,__READ_WRITE,__dscr12_bits);
__IO_REG32_BIT(DSCR13,                    0x10027850,__READ_WRITE,__dscr13_bits);
__IO_REG32_BIT(PSCR,                      0x10027854,__READ_WRITE,__pscr_bits);
__IO_REG32_BIT(PCSR,                      0x10027858,__READ_WRITE,__pcsr_bits);
__IO_REG32_BIT(PMCR,                      0x10027860,__READ_WRITE,__pmcr_bits);
__IO_REG32_BIT(DCVR0,                     0x10027864,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR1,                     0x10027868,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR2,                     0x1002786C,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(DCVR3,                     0x10027870,__READ_WRITE,__dcvr_bits);
__IO_REG32_BIT(PPCR,                      0x10027874,__READ_WRITE,__ppcr_bits);

/***************************************************************************
 **
 **  GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(PMASK,                     0x10015600,__READ_WRITE,__pmask_bits);

/***************************************************************************
 **
 **  GPIO PORTA
 **
 ***************************************************************************/
__IO_REG32_BIT(PTA_DDIR,                  0x10015000,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_OCR1,                  0x10015004,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTA_OCR2,                  0x10015008,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTA_ICONFA1,               0x1001500C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTA_ICONFA2,               0x10015010,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTA_ICONFB1,               0x10015014,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTA_ICONFB2,               0x10015018,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTA_DR,                    0x1001501C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_GIUS,                  0x10015020,__READ_WRITE,__pta_gius_bits);
__IO_REG32_BIT(PTA_SSR,                   0x10015024,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_ICR1,                  0x10015028,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTA_ICR2,                  0x1001502C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTA_IMR,                   0x10015030,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_ISR,                   0x10015034,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_GPR,                   0x10015038,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTA_SWR,                   0x1001503C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTA_PUEN,                  0x10015040,__READ_WRITE,__port_reg_31_0_bits);

/***************************************************************************
 **
 **  GPIO PORTB
 **
 ***************************************************************************/
__IO_REG32_BIT(PTB_DDIR,                  0x10015100,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_OCR1,                  0x10015104,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTB_OCR2,                  0x10015108,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTB_ICONFA1,               0x1001510C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTB_ICONFA2,               0x10015110,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTB_ICONFB1,               0x10015114,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTB_ICONFB2,               0x10015118,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTB_DR,                    0x1001511C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_GIUS,                  0x10015120,__READ_WRITE,__ptb_gius_bits);
__IO_REG32_BIT(PTB_SSR,                   0x10015124,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_ICR1,                  0x10015128,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTB_ICR2,                  0x1001512C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTB_IMR,                   0x10015130,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_ISR,                   0x10015134,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_GPR,                   0x10015138,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTB_SWR,                   0x1001513C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTB_PUEN,                  0x10015140,__READ_WRITE,__port_reg_31_0_bits);

/***************************************************************************
 **
 **  GPIO PORTC
 **
 ***************************************************************************/
__IO_REG32_BIT(PTC_DDIR,                  0x10015200,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_OCR1,                  0x10015204,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTC_OCR2,                  0x10015208,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTC_ICONFA1,               0x1001520C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTC_ICONFA2,               0x10015210,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTC_ICONFB1,               0x10015214,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTC_ICONFB2,               0x10015218,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTC_DR,                    0x1001521C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_GIUS,                  0x10015220,__READ_WRITE,__ptc_gius_bits);
__IO_REG32_BIT(PTC_SSR,                   0x10015224,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_ICR1,                  0x10015228,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTC_ICR2,                  0x1001522C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTC_IMR,                   0x10015230,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_ISR,                   0x10015234,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_GPR,                   0x10015238,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTC_SWR,                   0x1001523C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTC_PUEN,                  0x10015240,__READ_WRITE,__port_reg_31_0_bits);

/***************************************************************************
 **
 **  GPIO PORTD
 **
 ***************************************************************************/
__IO_REG32_BIT(PTD_DDIR,                  0x10015300,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_OCR1,                  0x10015304,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTD_OCR2,                  0x10015308,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTD_ICONFA1,               0x1001530C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTD_ICONFA2,               0x10015310,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTD_ICONFB1,               0x10015314,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTD_ICONFB2,               0x10015318,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTD_DR,                    0x1001531C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_GIUS,                  0x10015320,__READ_WRITE,__ptd_gius_bits);
__IO_REG32_BIT(PTD_SSR,                   0x10015324,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_ICR1,                  0x10015328,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTD_ICR2,                  0x1001532C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTD_IMR,                   0x10015330,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_ISR,                   0x10015334,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_GPR,                   0x10015338,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTD_SWR,                   0x1001533C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTD_PUEN,                  0x10015340,__READ_WRITE,__port_reg_31_0_bits);

/***************************************************************************
 **
 **  GPIO PORTE
 **
 ***************************************************************************/
__IO_REG32_BIT(PTE_DDIR,                  0x10015400,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_OCR1,                  0x10015404,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTE_OCR2,                  0x10015408,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTE_ICONFA1,               0x1001540C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTE_ICONFA2,               0x10015410,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTE_ICONFB1,               0x10015414,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTE_ICONFB2,               0x10015418,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTE_DR,                    0x1001541C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_GIUS,                  0x10015420,__READ_WRITE,__pte_gius_bits);
__IO_REG32_BIT(PTE_SSR,                   0x10015424,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_ICR1,                  0x10015428,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTE_ICR2,                  0x1001542C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTE_IMR,                   0x10015430,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_ISR,                   0x10015434,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_GPR,                   0x10015438,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTE_SWR,                   0x1001543C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTE_PUEN,                  0x10015440,__READ_WRITE,__port_reg_31_0_bits);

/***************************************************************************
 **
 **  GPIO PORTF
 **
 ***************************************************************************/
__IO_REG32_BIT(PTF_DDIR,                  0x10015500,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_OCR1,                  0x10015504,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTF_OCR2,                  0x10015508,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTF_ICONFA1,               0x1001550C,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTF_ICONFA2,               0x10015510,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTF_ICONFB1,               0x10015514,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTF_ICONFB2,               0x10015518,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTF_DR,                    0x1001551C,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_GIUS,                  0x10015520,__READ_WRITE,__ptf_gius_bits);
__IO_REG32_BIT(PTF_SSR,                   0x10015524,__READ      ,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_ICR1,                  0x10015528,__READ_WRITE,__port_reg_15_0_bits);
__IO_REG32_BIT(PTF_ICR2,                  0x1001552C,__READ_WRITE,__port_reg_31_16_bits);
__IO_REG32_BIT(PTF_IMR,                   0x10015530,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_ISR,                   0x10015534,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_GPR,                   0x10015538,__READ_WRITE,__port_reg_31_0_bits);
__IO_REG32_BIT(PTF_SWR,                   0x1001553C,__READ_WRITE,__swr_bits);
__IO_REG32_BIT(PTF_PUEN,                  0x10015540,__READ_WRITE,__port_reg_31_0_bits);


/***************************************************************************
 **
 **  AITC
 **
 ***************************************************************************/
__IO_REG32_BIT(INTCNTL,                   0x10040000,__READ_WRITE,__intcntl_bits);
__IO_REG32_BIT(NIMASK,                    0x10040004,__READ_WRITE,__nimask_bits);
__IO_REG32_BIT(INTENNUM,                  0x10040008,__READ_WRITE,__intennum_bits);
__IO_REG32_BIT(INTDISNUM,                 0x1004000C,__READ_WRITE,__intdisnum_bits);
__IO_REG32_BIT(INTENABLEH,                0x10040010,__READ_WRITE,__intenableh_bits);
__IO_REG32_BIT(INTENABLEL,                0x10040014,__READ_WRITE,__intenablel_bits);
__IO_REG32_BIT(INTTYPEH,                  0x10040018,__READ_WRITE,__inttypeh_bits);
__IO_REG32_BIT(INTTYPEL,                  0x1004001C,__READ_WRITE,__inttypel_bits);
__IO_REG32_BIT(NIPRIORITY7,               0x10040020,__READ_WRITE,__nipriority7_bits);
__IO_REG32_BIT(NIPRIORITY6,               0x10040024,__READ_WRITE,__nipriority6_bits);
__IO_REG32_BIT(NIPRIORITY5,               0x10040028,__READ_WRITE,__nipriority5_bits);
__IO_REG32_BIT(NIPRIORITY4,               0x1004002C,__READ_WRITE,__nipriority4_bits);
__IO_REG32_BIT(NIPRIORITY3,               0x10040030,__READ_WRITE,__nipriority3_bits);
__IO_REG32_BIT(NIPRIORITY2,               0x10040034,__READ_WRITE,__nipriority2_bits);
__IO_REG32_BIT(NIPRIORITY1,               0x10040038,__READ_WRITE,__nipriority1_bits);
__IO_REG32_BIT(NIPRIORITY0,               0x1004003C,__READ_WRITE,__nipriority0_bits);
__IO_REG32_BIT(NIVECSR,                   0x10040040,__READ      ,__nivecsr_bits);
__IO_REG32(    FIVECSR,                   0x10040044,__READ      );
__IO_REG32_BIT(INTSRCH,                   0x10040048,__READ      ,__intsrch_bits);
__IO_REG32_BIT(INTSRCL,                   0x1004004C,__READ      ,__intsrcl_bits);
__IO_REG32_BIT(INTFRCH,                   0x10040050,__READ_WRITE,__intfrch_bits);
__IO_REG32_BIT(INTFRCL,                   0x10040054,__READ_WRITE,__intfrcl_bits);
__IO_REG32_BIT(NIPNDH,                    0x10040058,__READ      ,__nipndh_bits);
__IO_REG32_BIT(NIPNDL,                    0x1004005C,__READ      ,__nipndl_bits);
__IO_REG32_BIT(FIPNDH,                    0x10040060,__READ      ,__fipndh_bits);
__IO_REG32_BIT(FIPNDL,                    0x10040064,__READ      ,__fipndl_bits);

/***************************************************************************
 **
 **  SCC
 **
 ***************************************************************************/
__IO_REG32_BIT(SCM_RED_START,             0x1002C000,__READ_WRITE,__scm_red_start_bits);
__IO_REG32_BIT(SCM_BLACK_START,           0x1002C004,__READ_WRITE,__scm_black_start_bits);
__IO_REG32_BIT(SCM_LENGTH,                0x1002C008,__READ_WRITE,__scm_length_bits);
__IO_REG32_BIT(SCM_CONTROL,               0x1002C00C,__READ_WRITE,__scm_control_bits);
__IO_REG32_BIT(SCM_STATUS,                0x1002C010,__READ      ,__scm_status_bits);
__IO_REG32_BIT(SCM_ERROR,                 0x1002C014,__READ_WRITE,__scm_error_bits);
__IO_REG32_BIT(SCM_INT_CTRL,              0x1002C018,__READ_WRITE,__scm_int_ctrl_bits);
__IO_REG32_BIT(SCM_CFG,                   0x1002C01C,__READ      ,__scm_cfg_bits);
__IO_REG32(    SCM_VEC0,                  0x1002C020,__READ_WRITE);
__IO_REG32(    SCM_VEC1,                  0x1002C024,__READ_WRITE);
__IO_REG32_BIT(SMN_STATUS,                0x1002D000,__READ_WRITE,__smn_status_bits);
__IO_REG32_BIT(SMN_COMMAND,               0x1002D004,__READ_WRITE,__smn_command_bits);
__IO_REG32_BIT(SMN_SSR,                   0x1002D008,__READ_WRITE,__smn_ssr_bits);
__IO_REG32_BIT(SMN_SER,                   0x1002D00C,__READ_WRITE,__smn_ser_bits);
__IO_REG32_BIT(SMN_SCR,                   0x1002D010,__READ_WRITE,__smn_scr_bits);
__IO_REG32_BIT(SMN_BCR,                   0x1002D014,__READ      ,__smn_bcr_bits);
__IO_REG32_BIT(SMN_BBISR,                 0x1002D018,__READ_WRITE,__smn_bbisr_bits);
__IO_REG32_BIT(SMN_BBDR,                  0x1002D01C,__WRITE     ,__smn_bbdr_bits);
__IO_REG32_BIT(SMN_CSR,                   0x1002D020,__READ_WRITE,__smn_csr_bits);
__IO_REG32(    SMN_PLAINTEXT,             0x1002D024,__READ_WRITE);
__IO_REG32(    SMN_CIPHERTEXT,            0x1002D028,__READ_WRITE);
__IO_REG32(    SMN_IVR,                   0x1002D02C,__READ_WRITE);
__IO_REG32_BIT(SMN_TCR,                   0x1002D030,__READ_WRITE,__smn_tcr_bits);
__IO_REG32_BIT(SMN_DDR,                   0x1002D034,__READ_WRITE,__smn_ddr_bits);
__IO_REG32(    SMN_TR,                    0x1002D038,__READ      );

/***************************************************************************
 **
 **  SAHARA
 **
 ***************************************************************************/
__IO_REG32(    SAHARA_VER_ID,             0x10025000,__READ      );
__IO_REG32(    SAHARA_DSCR_ADDR,          0x10025004,__READ_WRITE);
__IO_REG32_BIT(SAHARA_CTRL,               0x10025008,__READ_WRITE,__sahara_ctrl_bits);
__IO_REG32_BIT(SAHARA_CMD,                0x1002500C,__READ_WRITE,__sahara_cmd_bits);
__IO_REG32_BIT(SAHARA_STA,                0x10025010,__READ      ,__sahara_sta_bits);
__IO_REG32_BIT(SAHARA_ERR_STA,            0x10025014,__READ      ,__sahara_err_sta_bits);
__IO_REG32(    SAHARA_FAULT_ADDR,         0x10025018,__READ      );
__IO_REG32(    SAHARA_CURR_DSCR,          0x1002501C,__READ      );
__IO_REG32(    SAHARA_INIT_DSCR,          0x10025020,__READ      );
__IO_REG32_BIT(SAHARA_BUFF_LEVEL,         0x10025024,__READ      ,__sahara_buff_level_bits);
__IO_REG32_BIT(SAHARA_INT_FLOW_CTRL,      0x100250C0,__READ_WRITE,__sahara_int_flow_ctrl_bits);
__IO_REG32(    SAHARA_TRAP_ADDR,          0x100250FC,__READ_WRITE);
__IO_REG32_BIT(SKHA_MODE_DBG,             0x10025100,__READ_WRITE,__skha_mode_dbg_bits);
__IO_REG32(    SKHA_KEY_SIZE_DBG,         0x10025104,__READ_WRITE);
__IO_REG32(    SKHA_DATA_SIZE_DBG,        0x10025108,__READ_WRITE);
__IO_REG32(    SKHA_STA_DBG,              0x1002510C,__READ      );
__IO_REG32(    SKHA_ERR_STA_DBG,          0x10025110,__READ      );
__IO_REG32(    SKHA_EOM_DBG,              0x10025114,__READ_WRITE);
__IO_REG32_BIT(MDHA_MODE_DBG,             0x10025200,__READ_WRITE,__mdha_mode_dbg_bits);
__IO_REG32(    MDHA_KEY_SIZE_DBG,         0x10025204,__READ_WRITE);
__IO_REG32(    MDHA_DATA_SIZE_DBG,        0x10025208,__READ_WRITE);
__IO_REG32(    MDHA_STA_DBG,              0x1002520C,__READ      );
__IO_REG32(    MDHA_ERR_STA_DBG,          0x10025210,__READ      );
__IO_REG32(    MDHA_EOM_DBG,              0x10025214,__READ_WRITE);
__IO_REG32_BIT(RNG_MODE_DBG,              0x10025300,__READ_WRITE,__rng_mode_dbg_bits);
__IO_REG32(    RNG_DATA_SIZE_DBG,         0x10025308,__READ_WRITE);
__IO_REG32(    RNG_STA_DBG,               0x1002530C,__READ      );
__IO_REG32(    RNG_ERR_STA_DBG,           0x10025310,__READ      );
__IO_REG32(    RNG_EOM_DBG,               0x10025314,__READ_WRITE);
__IO_REG32(    RNG_ENTROPY_DBG,           0x10025380,__WRITE     );

/***************************************************************************
 **
 **  RTIC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTICSR,                    0x1002A000,__READ      ,__rticsr_bits);
__IO_REG32_BIT(RTICMD,                    0x1002A004,__WRITE     ,__rticmd_bits);
__IO_REG32_BIT(RTICCNTLR,                 0x1002A008,__READ_WRITE,__rticcntlr_bits);
__IO_REG32_BIT(RTICTR,                    0x1002A00C,__READ_WRITE,__rtictr_bits);
__IO_REG32(    RTICAMSAR1,                0x1002A010,__READ_WRITE);
__IO_REG32(    RTICAMLR1,                 0x1002A014,__READ_WRITE);
__IO_REG32(    RTICAMSAR2,                0x1002A018,__READ_WRITE);
__IO_REG32(    RTICAMLR2,                 0x1002A01C,__READ_WRITE);
__IO_REG32(    RTICBMSAR1,                0x1002A030,__READ_WRITE);
__IO_REG32(    RTICBMLR1,                 0x1002A034,__READ_WRITE);
__IO_REG32(    RTICBMSAR2,                0x1002A038,__READ_WRITE);
__IO_REG32(    RTICBMLR2,                 0x1002A03C,__READ_WRITE);
__IO_REG32(    RTICCMSAR1,                0x1002A050,__READ_WRITE);
__IO_REG32(    RTICCMLR1,                 0x1002A054,__READ_WRITE);
__IO_REG32(    RTICCMSAR2,                0x1002A058,__READ_WRITE);
__IO_REG32(    RTICCMLR2,                 0x1002A05C,__READ_WRITE);
__IO_REG32(    RTICDMSAR1,                0x1002A070,__READ_WRITE);
__IO_REG32(    RTICDMLR1,                 0x1002A074,__READ_WRITE);
__IO_REG32(    RTICDMSAR2,                0x1002A078,__READ_WRITE);
__IO_REG32(    RTICDMLR2,                 0x1002A07C,__READ_WRITE);
__IO_REG32(    RTICFAR,                   0x1002A090,__READ      );
__IO_REG32_BIT(RTICWR,                    0x1002A094,__READ_WRITE,__rticwr_bits);
__IO_REG32(    RTICAMHR1,                 0x1002A0A0,__READ      );
__IO_REG32(    RTICAMHR2,                 0x1002A0A4,__READ      );
__IO_REG32(    RTICAMHR3,                 0x1002A0A8,__READ      );
__IO_REG32(    RTICAMHR4,                 0x1002A0AC,__READ      );
__IO_REG32(    RTICAMHR5,                 0x1002A0B0,__READ      );
__IO_REG32(    RTICBMHR1,                 0x1002A0C0,__READ      );
__IO_REG32(    RTICBMHR2,                 0x1002A0C4,__READ      );
__IO_REG32(    RTICBMHR3,                 0x1002A0C8,__READ      );
__IO_REG32(    RTICBMHR4,                 0x1002A0CC,__READ      );
__IO_REG32(    RTICBMHR5,                 0x1002A0D0,__READ      );
__IO_REG32(    RTICCMHR1,                 0x1002A0E0,__READ      );
__IO_REG32(    RTICCMHR2,                 0x1002A0E4,__READ      );
__IO_REG32(    RTICCMHR3,                 0x1002A0E8,__READ      );
__IO_REG32(    RTICCMHR4,                 0x1002A0EC,__READ      );
__IO_REG32(    RTICCMHR5,                 0x1002A0F0,__READ      );
__IO_REG32(    RTICDMHR1,                 0x1002A100,__READ      );
__IO_REG32(    RTICDMHR2,                 0x1002A104,__READ      );
__IO_REG32(    RTICDMHR3,                 0x1002A108,__READ      );
__IO_REG32(    RTICDMHR4,                 0x1002A10C,__READ      );
__IO_REG32(    RTICDMHR5,                 0x1002A110,__READ      );

/***************************************************************************
 **
 **  IIM
 **
 ***************************************************************************/
__IO_REG8_BIT( IIM_STAT,                  0x10028000,__READ_WRITE,__iim_stat_bits);
__IO_REG8_BIT( IIM_STATM,                 0x10028004,__READ_WRITE,__iim_statm_bits);
__IO_REG8_BIT( IIM_ERR,                   0x10028008,__READ_WRITE,__iim_err_bits);
__IO_REG8_BIT( IIM_EMASK,                 0x1002800C,__READ_WRITE,__iim_emask_bits);
__IO_REG8_BIT( IIM_FCTL,                  0x10028010,__READ_WRITE,__iim_fctl_bits);
__IO_REG8_BIT( IIM_UA,                    0x10028014,__READ_WRITE,__iim_ua_bits);
__IO_REG8(     IIM_LA,                    0x10028018,__READ_WRITE);
__IO_REG8(     IIM_SDAT,                  0x1002801C,__READ      );
__IO_REG8_BIT( IIM_PREV,                  0x10028020,__READ      ,__iim_prev_bits);
__IO_REG8(     IIM_SREV,                  0x10028024,__READ      );
__IO_REG8(     IIM_PREG_P,                0x10028028,__READ_WRITE);
__IO_REG8_BIT( IIM_SCS0,                  0x1002802C,__READ_WRITE,__iim_scs_bits);
__IO_REG8_BIT( IIM_SCS1,                  0x10028030,__READ_WRITE,__iim_scs_bits);
__IO_REG8_BIT( IIM_SCS2,                  0x10028034,__READ_WRITE,__iim_scs_bits);
__IO_REG8_BIT( IIM_SCS3,                  0x10028038,__READ_WRITE,__iim_scs_bits);
__IO_REG8_BIT( IIM_FBAC0,                 0x10028800,__READ_WRITE,__iim_fbac0_bits);
__IO_REG8_BIT( IIM_FB0_WORD1,             0x10028804,__READ_WRITE,__iim_fb0_word1_bits);
__IO_REG8_BIT( IIM_FB0_WORD2,             0x10028808,__READ_WRITE,__iim_fb0_word2_bits);
__IO_REG8_BIT( IIM_FB0_WORD3,             0x1002880C,__READ_WRITE,__iim_fb0_word3_bits);
__IO_REG8_BIT( IIM_FB0_WORD4,             0x10028810,__READ_WRITE,__iim_fb0_word4_bits);
__IO_REG8(     IIM_SI_ID0,                0x10028814,__READ_WRITE);
__IO_REG8(     IIM_SI_ID1,                0x10028818,__READ_WRITE);
__IO_REG8(     IIM_SI_ID2,                0x1002881C,__READ_WRITE);
__IO_REG8(     IIM_SI_ID3,                0x10028820,__READ_WRITE);
__IO_REG8(     IIM_SI_ID4,                0x10028824,__READ_WRITE);
__IO_REG8(     IIM_SI_ID5,                0x10028828,__READ_WRITE);
__IO_REG8(     IIM_SCC_KEY0,              0x1002882C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY1,              0x10028830,__WRITE     );
__IO_REG8(     IIM_SCC_KEY2,              0x10028834,__WRITE     );
__IO_REG8(     IIM_SCC_KEY3,              0x10028838,__WRITE     );
__IO_REG8(     IIM_SCC_KEY4,              0x1002883C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY5,              0x10028840,__WRITE     );
__IO_REG8(     IIM_SCC_KEY6,              0x10028844,__WRITE     );
__IO_REG8(     IIM_SCC_KEY7,              0x10028848,__WRITE     );
__IO_REG8(     IIM_SCC_KEY8,              0x1002884C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY9,              0x10028850,__WRITE     );
__IO_REG8(     IIM_SCC_KEY10,             0x10028854,__WRITE     );
__IO_REG8(     IIM_SCC_KEY11,             0x10028858,__WRITE     );
__IO_REG8(     IIM_SCC_KEY12,             0x1002885C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY13,             0x10028860,__WRITE     );
__IO_REG8(     IIM_SCC_KEY14,             0x10028864,__WRITE     );
__IO_REG8(     IIM_SCC_KEY15,             0x10028868,__WRITE     );
__IO_REG8(     IIM_SCC_KEY16,             0x1002886C,__WRITE     );
__IO_REG8(     IIM_SCC_KEY17,             0x10028870,__WRITE     );
__IO_REG8(     IIM_SCC_KEY18,             0x10028874,__WRITE     );
__IO_REG8(     IIM_SCC_KEY19,             0x10028878,__WRITE     );
__IO_REG8(     IIM_SCC_KEY20,             0x1002887C,__WRITE     );
__IO_REG8_BIT( IIM_FBAC1,                 0x10028C00,__READ_WRITE,__iim_fbac1_bits);
__IO_REG8(     IIM_MAC_ADDR0,             0x10028C04,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR1,             0x10028C08,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR2,             0x10028C0C,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR3,             0x10028C10,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR4,             0x10028C14,__READ_WRITE);
__IO_REG8(     IIM_MAC_ADDR5,             0x10028C18,__READ_WRITE);

/***************************************************************************
 **
 **  M3IF
 **
 ***************************************************************************/
__IO_REG32_BIT(M3IFCTL,                   0xD8003000,__READ_WRITE,__m3ifctl_bits);
__IO_REG32_BIT(M3IFSCFG0,                 0xD8003028,__READ_WRITE,__m3ifscfg0_bits);
__IO_REG32_BIT(M3IFSCFG1,                 0xD800302C,__READ_WRITE,__m3ifscfg1_bits);
__IO_REG32_BIT(M3IFSCFG2,                 0xD8003030,__READ_WRITE,__m3ifscfg2_bits);
__IO_REG32_BIT(M3IFSSR0,                  0xD8003034,__READ_WRITE,__m3ifssr0_bits);
__IO_REG32_BIT(M3IFSSR1,                  0xD8003038,__READ_WRITE,__m3ifssr1_bits);
__IO_REG32_BIT(M3IFMLWE0,                 0xD8003040,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE1,                 0xD8003044,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE2,                 0xD8003048,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE3,                 0xD800304C,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE4,                 0xD8003050,__READ_WRITE,__m3ifmlwe_bits);
__IO_REG32_BIT(M3IFMLWE5,                 0xD8003054,__READ_WRITE,__m3ifmlwe_bits);

/***************************************************************************
 **
 **  WEIM
 **
 ***************************************************************************/
__IO_REG32_BIT(CSCR0U,                    0xD8002000,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR0L,                    0xD8002004,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR0A,                    0xD8002008,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR1U,                    0xD8002010,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR1L,                    0xD8002014,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR1A,                    0xD8002018,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR2U,                    0xD8002020,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR2L,                    0xD8002024,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR2A,                    0xD8002028,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR3U,                    0xD8002030,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR3L,                    0xD8002034,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR3A,                    0xD8002038,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR4U,                    0xD8002040,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR4L,                    0xD8002044,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR4A,                    0xD8002048,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(CSCR5U,                    0xD8002050,__READ_WRITE,__cscru_bits);
__IO_REG32_BIT(CSCR5L,                    0xD8002054,__READ_WRITE,__cscrl_bits);
__IO_REG32_BIT(CSCR5A,                    0xD8002058,__READ_WRITE,__cscra_bits);
__IO_REG32_BIT(WEIM_WCR,                  0xD8002060,__READ_WRITE,__weim_wcr_bits);

/***************************************************************************
 **
 **  ESDTCL
 **
 ***************************************************************************/
__IO_REG32_BIT(ESDCTL0,                   0xD8001000,__READ_WRITE,__esdctl_bits);
__IO_REG32_BIT(ESDCFG0,                   0xD8001004,__READ_WRITE,__esdcfg_bits);
__IO_REG32_BIT(ESDCTL1,                   0xD8001008,__READ_WRITE,__esdctl_bits);
__IO_REG32_BIT(ESDCFG1,                   0xD800100C,__READ_WRITE,__esdcfg_bits);
__IO_REG32_BIT(ESDMISC,                   0xD8001010,__READ_WRITE,__esdmisc_bits);
__IO_REG32_BIT(ESDCDLY1,                  0xD8001020,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY2,                  0xD8001024,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY3,                  0xD8001028,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY4,                  0xD800102C,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLY5,                  0xD8001030,__READ_WRITE,__esdcdly_bits);
__IO_REG32_BIT(ESDCDLYL,                  0xD8001034,__READ      ,__esdcdlyl_bits);

/***************************************************************************
 **
 **  NFC
 **
 ***************************************************************************/
__IO_REG16_BIT(NFC_BUFSIZ,                0xD8000E00,__READ      ,__nfc_bufsiz_bits);
__IO_REG16_BIT(NFC_RBA,                   0xD8000E04,__READ_WRITE,__nfc_rba_bits);
#define RAM_BUFFER_ADDRESS NFC_RBA
__IO_REG16(    NAND_FLASH_ADD,            0xD8000E06,__READ_WRITE);
__IO_REG16(    NAND_FLASH_CMD,            0xD8000E08,__READ_WRITE);
__IO_REG16_BIT(NFC_IBLC,                  0xD8000E0A,__READ_WRITE,__nfc_iblc_bits);
#define NFC_CONFIGURATION   NFC_IBLC
__IO_REG16_BIT(ECC_SRR,                   0xD8000E0C,__READ      ,__ecc_srr_bits);
#define ECC_STATUS_RESULT   ECC_SRR
__IO_REG16_BIT(ECC_RSLT_MA_8,             0xD8000E0E,__READ      ,__ecc_rslt_ma_bits);
#define ECC_RSLT_MAIN_AREA  ECC_RSLT_MA_8
#define ECC_RSLT_MA_16      ECC_RSLT_MA_8
#define ECC_RSLT_MA_16_bit  ECC_RSLT_MA_8_bit
__IO_REG16_BIT(ECC_RSLT_SA_8,             0xD8000E10,__READ      ,__ecc_rslt_sa_bits);
#define ECC_RSLT_SPARE_AREA ECC_RSLT_SA_8
#define ECC_RSLT_SA_16      ECC_RSLT_SA_8
#define ECC_RSLT_SA_16_bit  ECC_RSLT_SA_8_bit
__IO_REG16_BIT(NF_WR_PROT,                0xD8000E12,__READ_WRITE,__nf_wr_prot_bits);
__IO_REG16(    NFC_USBA,                  0xD8000E14,__READ_WRITE);
#define UNLOCK_START_BLK_ADD  NFC_USBA
__IO_REG16(    NFC_UEBA,                  0xD8000E16,__READ_WRITE);
#define UNLOCK_END_BLK_ADD    NFC_UEBA
__IO_REG16_BIT(NF_WR_PROT_STA,            0xD8000E18,__READ_WRITE,__nf_wr_prot_sta_bits);
#define NAND_FLASH_WR_PR_ST   NF_WR_PROT_STA
__IO_REG16_BIT(NAND_FC1,                  0xD8000E1A,__READ_WRITE,__nand_fc1_bits);
#define NAND_FLASH_CONFIG1    NAND_FC1
__IO_REG16_BIT(NAND_FC2,                  0xD8000E1C,__READ_WRITE,__nand_fc2_bits);
#define NAND_FLASH_CONFIG2    NAND_FC2

/***************************************************************************
 **
 **  PCMCIA
 **
 ***************************************************************************/
__IO_REG32_BIT(PCMCIA_PIPR,               0xD8004000,__READ      ,__pcmcia_pipr_bits);
__IO_REG32_BIT(PCMCIA_PSCR,               0xD8004004,__READ_WRITE,__pcmcia_pscr_bits);
__IO_REG32_BIT(PCMCIA_PER,                0xD8004008,__READ_WRITE,__pcmcia_per_bits);
__IO_REG32_BIT(PCMCIA_PBR0,               0xD800400C,__READ_WRITE,__pcmcia_pbr_bits);
__IO_REG32_BIT(PCMCIA_PBR1,               0xD8004010,__READ_WRITE,__pcmcia_pbr_bits);
__IO_REG32_BIT(PCMCIA_PBR2,               0xD8004014,__READ_WRITE,__pcmcia_pbr_bits);
__IO_REG32_BIT(PCMCIA_PBR3,               0xD8004018,__READ_WRITE,__pcmcia_pbr_bits);
__IO_REG32_BIT(PCMCIA_PBR4,               0xD800401C,__READ_WRITE,__pcmcia_pbr_bits);
__IO_REG32_BIT(PCMCIA_POR0,               0xD8004028,__READ_WRITE,__pcmcia_por_bits);
__IO_REG32_BIT(PCMCIA_POR1,               0xD800402C,__READ_WRITE,__pcmcia_por_bits);
__IO_REG32_BIT(PCMCIA_POR2,               0xD8004030,__READ_WRITE,__pcmcia_por_bits);
__IO_REG32_BIT(PCMCIA_POR3,               0xD8004034,__READ_WRITE,__pcmcia_por_bits);
__IO_REG32_BIT(PCMCIA_POR4,               0xD8004038,__READ_WRITE,__pcmcia_por_bits);
__IO_REG32_BIT(PCMCIA_POFR0,              0xD8004044,__READ_WRITE,__pcmcia_pofr_bits);
__IO_REG32_BIT(PCMCIA_POFR1,              0xD8004048,__READ_WRITE,__pcmcia_pofr_bits);
__IO_REG32_BIT(PCMCIA_POFR2,              0xD800404C,__READ_WRITE,__pcmcia_pofr_bits);
__IO_REG32_BIT(PCMCIA_POFR3,              0xD8004050,__READ_WRITE,__pcmcia_pofr_bits);
__IO_REG32_BIT(PCMCIA_POFR4,              0xD8004054,__READ_WRITE,__pcmcia_pofr_bits);
__IO_REG32_BIT(PCMCIA_PGCR,               0xD8004060,__READ_WRITE,__pcmcia_pgcr_bits);
__IO_REG32_BIT(PCMCIA_PGSR,               0xD8004064,__READ_WRITE,__pcmcia_pgsr_bits);

/***************************************************************************
 **
 **  1-Wire
 **
 ***************************************************************************/
__IO_REG16_BIT(OW_CONTROL,                0x10009000,__READ_WRITE,__ow_control_bits);
__IO_REG16_BIT(OW_TIME_DIVIDER,           0x10009002,__READ_WRITE,__ow_time_divider_bits);
__IO_REG16_BIT(OW_RESET,                  0x10009004,__READ_WRITE,__ow_reset_bits);

/***************************************************************************
 **
 **  ATA
 **
 ***************************************************************************/
__IO_REG32_BIT(ATA_TIME_CONFIG0,          0x80001000,__READ_WRITE,__ata_time_config0_bits);
__IO_REG32_BIT(ATA_TIME_CONFIG1,          0x80001004,__READ_WRITE,__ata_time_config1_bits);
__IO_REG32_BIT(ATA_TIME_CONFIG2,          0x80001008,__READ_WRITE,__ata_time_config2_bits);
__IO_REG32_BIT(ATA_TIME_CONFIG3,          0x8000100C,__READ_WRITE,__ata_time_config3_bits);
__IO_REG32_BIT(ATA_TIME_CONFIG4,          0x80001010,__READ_WRITE,__ata_time_config4_bits);
__IO_REG32_BIT(ATA_TIME_CONFIG5,          0x80001014,__READ_WRITE,__ata_time_config5_bits);
__IO_REG32(    ATA_FIFO_DATA_32,          0x80001018,__READ_WRITE);
__IO_REG32(    ATA_FIFO_DATA_16,          0x8000101C,__READ_WRITE);
__IO_REG32_BIT(ATA_FIFO_FILL,             0x80001020,__READ      ,__ata_fifo_fill_bits);
__IO_REG32_BIT(ATA_CONTROL,               0x80001024,__READ_WRITE,__ata_control_bits);
__IO_REG32_BIT(ATA_INT_PENDING,           0x80001028,__READ      ,__ata_int_pending_bits);
__IO_REG32_BIT(ATA_INT_ENABLE,            0x8000102C,__READ_WRITE,__ata_int_pending_bits);
__IO_REG32_BIT(ATA_INT_CLEAR,             0x80001030,__WRITE     ,__ata_int_pending_bits);
__IO_REG32_BIT(ATA_FIFO_ALARM,            0x80001034,__READ_WRITE,__ata_fifo_alarm_bits);
__IO_REG8(     ATA_DDTR,                  0x800010A0,__READ_WRITE);
__IO_REG8(     ATA_DFTR,                  0x800010A4,__READ_WRITE);
__IO_REG8(     ATA_DSCR,                  0x800010A8,__READ_WRITE);
__IO_REG8(     ATA_DSNR,                  0x800010AC,__READ_WRITE);
__IO_REG8(     ATA_DCLR,                  0x800010B0,__READ_WRITE);
__IO_REG8(     ATA_DCHR,                  0x800010B4,__READ_WRITE);
__IO_REG8(     ATA_DDHR,                  0x800010B8,__READ_WRITE);
__IO_REG8(     ATA_DCDR,                  0x800010BC,__READ_WRITE);
#define ATA_DSR       ATA_DCDR
__IO_REG16(    ATA_DCTR,                  0x800010D8,__READ_WRITE);
#define ATA_DASR      ATA_DCTR

/***************************************************************************
 **
 **  CSPI1
 **
 ***************************************************************************/
__IO_REG32(    CSPI1_RXDATA,              0x1000E000,__READ      );
__IO_REG32(    CSPI1_TXDATA,              0x1000E004,__WRITE     );
__IO_REG32_BIT(CSPI1_CONREG,              0x1000E008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI1_INTREG,              0x1000E00C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI1_TESTREG,             0x1000E010,__READ_WRITE,__cspi_test_bits);
__IO_REG32_BIT(CSPI1_PERIODREG,           0x1000E014,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI1_DMAREG,              0x1000E018,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI1_RESETREG,            0x1000E01C,__READ_WRITE,__cspi_reset_bits);

/***************************************************************************
 **
 **  CSPI2
 **
 ***************************************************************************/
__IO_REG32(    CSPI2_RXDATA,              0x1000F000,__READ      );
__IO_REG32(    CSPI2_TXDATA,              0x1000F004,__WRITE     );
__IO_REG32_BIT(CSPI2_CONREG,              0x1000F008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI2_INTREG,              0x1000F00C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI2_TESTREG,             0x1000F010,__READ_WRITE,__cspi_test_bits);
__IO_REG32_BIT(CSPI2_PERIODREG,           0x1000F014,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI2_DMAREG,              0x1000F018,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI2_RESETREG,            0x1000F01C,__READ_WRITE,__cspi_reset_bits);

/***************************************************************************
 **
 **  CSPI3
 **
 ***************************************************************************/
__IO_REG32(    CSPI3_RXDATA,              0x10017000,__READ      );
__IO_REG32(    CSPI3_TXDATA,              0x10017004,__WRITE     );
__IO_REG32_BIT(CSPI3_CONREG,              0x10017008,__READ_WRITE,__cspi_controlreg_bits);
__IO_REG32_BIT(CSPI3_INTREG,              0x1001700C,__READ_WRITE,__cspi_intreg_bits);
__IO_REG32_BIT(CSPI3_TESTREG,             0x10017010,__READ_WRITE,__cspi_test_bits);
__IO_REG32_BIT(CSPI3_PERIODREG,           0x10017014,__READ_WRITE,__cspi_period_bits);
__IO_REG32_BIT(CSPI3_DMAREG,              0x10017018,__READ_WRITE,__cspi_dma_bits);
__IO_REG32_BIT(CSPI3_RESETREG,            0x1001701C,__READ_WRITE,__cspi_reset_bits);

/***************************************************************************
 **
 **  I2C1
 **
 ***************************************************************************/
__IO_REG32_BIT(IADR1,                     0x10012000,__READ_WRITE,__iadr_bits);
__IO_REG32_BIT(IFDR1,                     0x10012004,__READ_WRITE,__ifdr_bits);
__IO_REG32_BIT(I2CR1,                     0x10012008,__READ_WRITE,__i2cr_bits);
__IO_REG32_BIT(I2SR1,                     0x1001200C,__READ_WRITE,__i2sr_bits);
__IO_REG32_BIT(I2DR1,                     0x10012010,__READ_WRITE,__i2dr_bits);

/***************************************************************************
 **
 **  I2C2
 **
 ***************************************************************************/
__IO_REG32_BIT(IADR2,                     0x1001D000,__READ_WRITE,__iadr_bits);
__IO_REG32_BIT(IFDR2,                     0x1001D004,__READ_WRITE,__ifdr_bits);
__IO_REG32_BIT(I2CR2,                     0x1001D008,__READ_WRITE,__i2cr_bits);
__IO_REG32_BIT(I2SR2,                     0x1001D00C,__READ_WRITE,__i2sr_bits);
__IO_REG32_BIT(I2DR2,                     0x1001D010,__READ_WRITE,__i2dr_bits);

/***************************************************************************
 **
 **  KPP
 **
 ***************************************************************************/
__IO_REG16_BIT(KPCR,                      0x10008000,__READ_WRITE,__kpcr_bits);
__IO_REG16_BIT(KPSR,                      0x10008002,__READ_WRITE,__kpsr_bits);
__IO_REG16_BIT(KDDR,                      0x10008004,__READ_WRITE,__kddr_bits);
__IO_REG16_BIT(KPDR,                      0x10008006,__READ_WRITE,__kpdr_bits);

 /***************************************************************************
 **
 **  MMC/SDHC1
 **
 ***************************************************************************/
__IO_REG32_BIT(SDHC1_STR_STP_CLK,         0x10013000,__READ_WRITE,__str_stp_clk_bits);
__IO_REG32_BIT(SDHC1_STATUS,              0x10013004,__READ_WRITE,__sd_status_bits);
__IO_REG32_BIT(SDHC1_CLK_RATE,            0x10013008,__READ_WRITE,__clk_rate_bits);
__IO_REG32_BIT(SDHC1_CMD_DAT_CONT,        0x1001300C,__READ_WRITE,__cmd_dat_cont_bits);
__IO_REG32_BIT(SDHC1_RES_TO,              0x10013010,__READ_WRITE,__res_to_bits);
__IO_REG32_BIT(SDHC1_READ_TO,             0x10013014,__READ_WRITE,__read_to_bits);
__IO_REG32_BIT(SDHC1_BLK_LEN,             0x10013018,__READ_WRITE,__blk_len_bits);
__IO_REG32_BIT(SDHC1_NOB,                 0x1001301C,__READ_WRITE,__nob_bits);
__IO_REG32_BIT(SDHC1_REV_NO,              0x10013020,__READ      ,__rev_no_bits);
__IO_REG32_BIT(SDHC1_INT_CTRL,            0x10013024,__READ_WRITE,__mmcsd_int_ctrl_bits);
__IO_REG32_BIT(SDHC1_CMD,                 0x10013028,__READ_WRITE,__cmd_bits);
__IO_REG32(    SDHC1_ARG,                 0x1001302C,__READ_WRITE);
__IO_REG32_BIT(SDHC1_RES_FIFO,            0x10013034,__READ      ,__res_fifo_bits);
__IO_REG32(    SDHC1_BUFFER_ACCESS,       0x10013038,__READ_WRITE);

 /***************************************************************************
 **
 **  MMC/SDHC2
 **
 ***************************************************************************/
__IO_REG32_BIT(SDHC2_STR_STP_CLK,         0x10014000,__READ_WRITE,__str_stp_clk_bits);
__IO_REG32_BIT(SDHC2_STATUS,              0x10014004,__READ_WRITE,__sd_status_bits);
__IO_REG32_BIT(SDHC2_CLK_RATE,            0x10014008,__READ_WRITE,__clk_rate_bits);
__IO_REG32_BIT(SDHC2_CMD_DAT_CONT,        0x1001400C,__READ_WRITE,__cmd_dat_cont_bits);
__IO_REG32_BIT(SDHC2_RES_TO,              0x10014010,__READ_WRITE,__res_to_bits);
__IO_REG32_BIT(SDHC2_READ_TO,             0x10014014,__READ_WRITE,__read_to_bits);
__IO_REG32_BIT(SDHC2_BLK_LEN,             0x10014018,__READ_WRITE,__blk_len_bits);
__IO_REG32_BIT(SDHC2_NOB,                 0x1001401C,__READ_WRITE,__nob_bits);
__IO_REG32_BIT(SDHC2_REV_NO,              0x10014020,__READ      ,__rev_no_bits);
__IO_REG32_BIT(SDHC2_INT_CTRL,            0x10014024,__READ_WRITE,__mmcsd_int_ctrl_bits);
__IO_REG32_BIT(SDHC2_CMD,                 0x10014028,__READ_WRITE,__cmd_bits);
__IO_REG32(    SDHC2_ARG,                 0x1001402C,__READ_WRITE);
__IO_REG32_BIT(SDHC2_RES_FIFO,            0x10014034,__READ      ,__res_fifo_bits);
__IO_REG32(    SDHC2_BUFFER_ACCESS,       0x10014038,__READ_WRITE);

 /***************************************************************************
 **
 **  MMC/SDHC3
 **
 ***************************************************************************/
__IO_REG32_BIT(SDHC3_STR_STP_CLK,         0x1001E000,__READ_WRITE,__str_stp_clk_bits);
__IO_REG32_BIT(SDHC3_STATUS,              0x1001E004,__READ_WRITE,__sd_status_bits);
__IO_REG32_BIT(SDHC3_CLK_RATE,            0x1001E008,__READ_WRITE,__clk_rate_bits);
__IO_REG32_BIT(SDHC3_CMD_DAT_CONT,        0x1001E00C,__READ_WRITE,__cmd_dat_cont_bits);
__IO_REG32_BIT(SDHC3_RES_TO,              0x1001E010,__READ_WRITE,__res_to_bits);
__IO_REG32_BIT(SDHC3_READ_TO,             0x1001E014,__READ_WRITE,__read_to_bits);
__IO_REG32_BIT(SDHC3_BLK_LEN,             0x1001E018,__READ_WRITE,__blk_len_bits);
__IO_REG32_BIT(SDHC3_NOB,                 0x1001E01C,__READ_WRITE,__nob_bits);
__IO_REG32_BIT(SDHC3_REV_NO,              0x1001E020,__READ      ,__rev_no_bits);
__IO_REG32_BIT(SDHC3_INT_CTRL,            0x1001E024,__READ_WRITE,__mmcsd_int_ctrl_bits);
__IO_REG32_BIT(SDHC3_CMD,                 0x1001E028,__READ_WRITE,__cmd_bits);
__IO_REG32(    SDHC3_ARG,                 0x1001E02C,__READ_WRITE);
__IO_REG32_BIT(SDHC3_RES_FIFO,            0x1001E034,__READ      ,__res_fifo_bits);
__IO_REG32(    SDHC3_BUFFER_ACCESS,       0x1001E038,__READ_WRITE);

/***************************************************************************
 **
 **  UART1
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_1,                    0x1000A000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_1,                    0x1000A040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_1,                    0x1000A080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_1,                    0x1000A084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_1,                    0x1000A088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_1,                    0x1000A08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_1,                    0x1000A090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_1,                    0x1000A094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_1,                    0x1000A098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_1,                    0x1000A09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_1,                    0x1000A0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_1,                    0x1000A0A4,__READ_WRITE);
__IO_REG32(    UBMR_1,                    0x1000A0A8,__READ_WRITE);
__IO_REG32(    UBRC_1,                    0x1000A0AC,__READ_WRITE);
__IO_REG32(    ONEMS_1,                   0x1000A0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_1,                     0x1000A0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART2
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_2,                    0x1000B000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_2,                    0x1000B040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_2,                    0x1000B080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_2,                    0x1000B084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_2,                    0x1000B088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_2,                    0x1000B08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_2,                    0x1000B090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_2,                    0x1000B094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_2,                    0x1000B098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_2,                    0x1000B09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_2,                    0x1000B0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_2,                    0x1000B0A4,__READ_WRITE);
__IO_REG32(    UBMR_2,                    0x1000B0A8,__READ_WRITE);
__IO_REG32(    UBRC_2,                    0x1000B0AC,__READ_WRITE);
__IO_REG32(    ONEMS_2,                   0x1000B0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_2,                     0x1000B0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART3
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_3,                    0x1000C000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_3,                    0x1000C040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_3,                    0x1000C080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_3,                    0x1000C084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_3,                    0x1000C088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_3,                    0x1000C08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_3,                    0x1000C090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_3,                    0x1000C094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_3,                    0x1000C098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_3,                    0x1000C09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_3,                    0x1000C0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_3,                    0x1000C0A4,__READ_WRITE);
__IO_REG32(    UBMR_3,                    0x1000C0A8,__READ_WRITE);
__IO_REG32(    UBRC_3,                    0x1000C0AC,__READ_WRITE);
__IO_REG32(    ONEMS_3,                   0x1000C0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_3,                     0x1000C0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART4
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_4,                    0x1000D000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_4,                    0x1000D040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_4,                    0x1000D080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_4,                    0x1000D084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_4,                    0x1000D088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_4,                    0x1000D08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_4,                    0x1000D090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_4,                    0x1000D094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_4,                    0x1000D098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_4,                    0x1000D09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_4,                    0x1000D0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_4,                    0x1000D0A4,__READ_WRITE);
__IO_REG32(    UBMR_4,                    0x1000D0A8,__READ_WRITE);
__IO_REG32(    UBRC_4,                    0x1000D0AC,__READ_WRITE);
__IO_REG32(    ONEMS_4,                   0x1000D0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_4,                     0x1000D0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART5
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_5,                    0x1001B000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_5,                    0x1001B040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_5,                    0x1001B080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_5,                    0x1001B084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_5,                    0x1001B088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_5,                    0x1001B08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_5,                    0x1001B090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_5,                    0x1001B094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_5,                    0x1001B098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_5,                    0x1001B09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_5,                    0x1001B0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_5,                    0x1001B0A4,__READ_WRITE);
__IO_REG32(    UBMR_5,                    0x1001B0A8,__READ_WRITE);
__IO_REG32(    UBRC_5,                    0x1001B0AC,__READ_WRITE);
__IO_REG32(    ONEMS_5,                   0x1001B0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_5,                     0x1001B0B4,__READ_WRITE,__uts_bits);

/***************************************************************************
 **
 **  UART6
 **
 ***************************************************************************/
__IO_REG32_BIT(URXD_6,                    0x1001C000,__READ      ,__urxd_bits);
__IO_REG32_BIT(UTXD_6,                    0x1001C040,__WRITE     ,__utxd_bits);
__IO_REG32_BIT(UCR1_6,                    0x1001C080,__READ_WRITE,__ucr1_bits);
__IO_REG32_BIT(UCR2_6,                    0x1001C084,__READ_WRITE,__ucr2_bits);
__IO_REG32_BIT(UCR3_6,                    0x1001C088,__READ_WRITE,__ucr3_bits);
__IO_REG32_BIT(UCR4_6,                    0x1001C08C,__READ_WRITE,__ucr4_bits);
__IO_REG32_BIT(UFCR_6,                    0x1001C090,__READ_WRITE,__ufcr_bits);
__IO_REG32_BIT(USR1_6,                    0x1001C094,__READ_WRITE,__usr1_bits);
__IO_REG32_BIT(USR2_6,                    0x1001C098,__READ_WRITE,__usr2_bits);
__IO_REG32_BIT(UESC_6,                    0x1001C09C,__READ_WRITE,__uesc_bits);
__IO_REG32_BIT(UTIM_6,                    0x1001C0A0,__READ_WRITE,__utim_bits);
__IO_REG32(    UBIR_6,                    0x1001C0A4,__READ_WRITE);
__IO_REG32(    UBMR_6,                    0x1001C0A8,__READ_WRITE);
__IO_REG32(    UBRC_6,                    0x1001C0AC,__READ_WRITE);
__IO_REG32(    ONEMS_6,                   0x1001C0B0,__READ_WRITE);
__IO_REG32_BIT(UTS_6,                     0x1001C0B4,__READ_WRITE,__uts_bits);

 /***************************************************************************
 **
 **  FEC
 **
 ***************************************************************************/
__IO_REG32_BIT(FEC_EIR,                   0x1002B004,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_EIMR,                  0x1002B008,__READ_WRITE,__fec_eir_bits);
__IO_REG32_BIT(FEC_RDAR,                  0x1002B010,__READ_WRITE,__fec_rdar_bits);
__IO_REG32_BIT(FEC_TDAR,                  0x1002B014,__READ_WRITE,__fec_tdar_bits);
__IO_REG32_BIT(FEC_ECR,                   0x1002B024,__READ_WRITE,__fec_ecr_bits);
__IO_REG32_BIT(FEC_MMFR,                  0x1002B040,__READ_WRITE,__fec_mmfr_bits);
__IO_REG32_BIT(FEC_MSCR,                  0x1002B044,__READ_WRITE,__fec_mscr_bits);
__IO_REG32_BIT(FEC_MIBC,                  0x1002B064,__READ_WRITE,__fec_mibc_bits);
__IO_REG32_BIT(FEC_RCR,                   0x1002B084,__READ_WRITE,__fec_rcr_bits);
__IO_REG32_BIT(FEC_TCR,                   0x1002B0C4,__READ_WRITE,__fec_tcr_bits);
__IO_REG32(    FEC_PALR,                  0x1002B0E4,__READ_WRITE);
__IO_REG32_BIT(FEC_PAUR,                  0x1002B0E8,__READ_WRITE,__fec_paur_bits);
__IO_REG32_BIT(FEC_OPD,                   0x1002B0EC,__READ_WRITE,__fec_opd_bits);
__IO_REG32(    FEC_IAUR,                  0x1002B118,__READ_WRITE);
__IO_REG32(    FEC_IALR,                  0x1002B11C,__READ_WRITE);
__IO_REG32(    FEC_GAUR,                  0x1002B120,__READ_WRITE);
__IO_REG32(    FEC_GALR,                  0x1002B124,__READ_WRITE);
__IO_REG32_BIT(FEC_TFWR,                  0x1002B144,__READ_WRITE,__fec_tfwr_bits);
__IO_REG32_BIT(FEC_FRBR,                  0x1002B14C,__READ      ,__fec_frbr_bits);
__IO_REG32_BIT(FEC_FRSR,                  0x1002B150,__READ_WRITE,__fec_frsr_bits);
__IO_REG32(    FEC_ERDSR,                 0x1002B180,__READ_WRITE);
__IO_REG32(    FEC_ETDSR,                 0x1002B184,__READ_WRITE);
__IO_REG32_BIT(FEC_EMRBR,                 0x1002B188,__READ_WRITE,__fec_emrbr_bits);
__IO_REG32(    FEC_RMON_T_DROP,           0x1002B200,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_PACKETS,        0x1002B204,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_BC_PKT,         0x1002B208,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_MC_PKT,         0x1002B20C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_CRC_ALIGN,      0x1002B210,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_UNDERSIZE,      0x1002B214,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OVERSIZE,       0x1002B218,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_FRAG,           0x1002B21C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_JAB,            0x1002B220,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_COL,            0x1002B224,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P64,            0x1002B228,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P65TO127,       0x1002B22C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P128TO255,      0x1002B230,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P256TO511,      0x1002B234,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P512TO1023,     0x1002B238,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P1024TO2047,    0x1002B23C,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_P_GTE2048,      0x1002B240,__READ_WRITE);
__IO_REG32(    FEC_RMON_T_OCTETS,         0x1002B244,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DROP,           0x1002B248,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FRAME_OK,       0x1002B24C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_1COL,           0x1002B250,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MCOL,           0x1002B254,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_DEF,            0x1002B258,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_LCOL,           0x1002B25C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_EXCOL,          0x1002B260,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_MACERR,         0x1002B264,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_CSERR,          0x1002B268,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_SQE,            0x1002B26C,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_FDXFC,          0x1002B270,__READ_WRITE);
__IO_REG32(    FEC_IEEE_T_OCTETS_OK,      0x1002B274,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_PACKETS,        0x1002B284,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_BC_PKT,         0x1002B288,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_MC_PKT,         0x1002B28C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_CRC_ALIGN,      0x1002B290,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_UNDERSIZE,      0x1002B294,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OVERSIZE,       0x1002B298,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_FRAG,           0x1002B29C,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_JAB,            0x1002B2A0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_RESVD_0,        0x1002B2A4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P64,            0x1002B2A8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P65TO127,       0x1002B2AC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P128TO255,      0x1002B2B0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P256TO511,      0x1002B2B4,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P512TO1023,     0x1002B2B8,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P1024TO2047,    0x1002B2BC,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_P_GTE2048,      0x1002B2C0,__READ_WRITE);
__IO_REG32(    FEC_RMON_R_OCTETS,         0x1002B2C4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_DROP,           0x1002B2C8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FRAME_OK,       0x1002B2CC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_CRC,            0x1002B2D0,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_ALIGN,          0x1002B2D4,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_MACERR,         0x1002B2D8,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_FDXFC,          0x1002B2DC,__READ_WRITE);
__IO_REG32(    FEC_IEEE_R_OCTETS_OK,      0x1002B2E0,__READ_WRITE);

/***************************************************************************
 **
 **  USB OTG
 **
 ***************************************************************************/
__IO_REG32_BIT(UOG_ID,                    0x10024000,__READ      ,__usb_id_bits);
__IO_REG32_BIT(UOG_HWGENERAL,             0x10024004,__READ      ,__usb_hwgeneral_bits);
__IO_REG32_BIT(UOG_HWHOST,                0x10024008,__READ      ,__usb_hwhost_bits);
__IO_REG32_BIT(UOG_HWDEVICE,              0x1002400C,__READ      ,__usb_hwdevice_bits);
__IO_REG32_BIT(UOG_HWTXBUF,               0x10024010,__READ      ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(UOG_HWRXBUF,               0x10024014,__READ      ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(UOG_GPTIMER0LD,            0x10024080,__READ_WRITE,__usb_gptimer0ld_bits);
__IO_REG32_BIT(UOG_GPTIMER0CTRL,          0x10024084,__READ_WRITE,__usb_gptimer0ctrl_bits);
__IO_REG32(    UOG_GPTIMER1LD,            0x10024088,__READ_WRITE);
__IO_REG32(    UOG_GPTIMER1CTRL,          0x1002408C,__READ_WRITE);
__IO_REG16(    UOG_CAPLENGTH,             0x10024100,__READ      );
__IO_REG16(    UOG_HCIVERSION,            0x10024102,__READ      );
__IO_REG32_BIT(UOG_HCSPARAMS,             0x10024104,__READ      ,__usb_hcsparams_bits);
__IO_REG32_BIT(UOG_HCCPARAMS,             0x10024108,__READ      ,__usb_hccparams_bits);
__IO_REG32_BIT(UOG_DCIVERSION,            0x10024120,__READ      ,__usb_dciversion_bits);
__IO_REG32_BIT(UOG_DCCPARAMS,             0x10024124,__READ      ,__usb_dccparams_bits);
__IO_REG32_BIT(UOG_USBCMD,                0x10024140,__READ_WRITE,__usb_usbcmd_bits);
__IO_REG32_BIT(UOG_USBSTS,                0x10024144,__READ_WRITE,__usb_usbsts_bits);
__IO_REG32_BIT(UOG_USBINTR,               0x10024148,__READ_WRITE,__usb_usbintr_bits);
__IO_REG32_BIT(UOG_FRINDEX,               0x1002414C,__READ_WRITE,__usb_frindex_bits);
__IO_REG32_BIT(UOG_PERIODICLISTBASE,      0x10024154,__READ_WRITE,__usb_periodiclistbase_bits);
#define UOG_DEVICEADDR      UOG_PERIODICLISTBASE
#define UOG_DEVICEADDR_bit  UOG_PERIODICLISTBASE_bit
__IO_REG32(    UOG_ASYNCLISTADDR,         0x10024158,__READ_WRITE);
#define UOG_ENDPOINTLISTADDR  UOG_ASYNCLISTADDR
__IO_REG32_BIT(UOG_BURSTSIZE,             0x10024160,__READ_WRITE,__usb_burstsize_bits);
__IO_REG32_BIT(UOG_TXFILLTUNING,          0x10024164,__READ_WRITE,__usb_txfilltuning_bits);
__IO_REG32_BIT(UOG_ULPIVIEW,              0x10024170,__READ_WRITE,__usb_ulpiview_bits);
__IO_REG32(    UOG_CFGFLAG,               0x10024180,__READ      );
__IO_REG32_BIT(UOG_PORTSC1,               0x10024184,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC2,               0x10024188,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC3,               0x1002418C,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC4,               0x10024190,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC5,               0x10024194,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC6,               0x10024198,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC7,               0x1002419C,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_PORTSC8,               0x100241A0,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UOG_OTGSC,                 0x100241A4,__READ_WRITE,__usb_otgsc_bits);
__IO_REG32_BIT(UOG_USBMODE,               0x100241A8,__READ_WRITE,__usb_usbmode_bits);
__IO_REG32_BIT(UOG_ENDPTSETUPSTAT,        0x100241AC,__READ_WRITE,__usb_endptsetupstat_bits);
__IO_REG32_BIT(UOG_ENDPTPRIME,            0x100241B0,__READ_WRITE,__usb_endptprime_bits);
__IO_REG32_BIT(UOG_ENDPTFLUSH,            0x100241B4,__READ_WRITE,__usb_endptflush_bits);
__IO_REG32_BIT(UOG_ENDPTSTAT,             0x100241B8,__READ      ,__usb_endptstat_bits);
__IO_REG32_BIT(UOG_ENDPTCOMPLETE,         0x100241BC,__READ_WRITE,__usb_endptcomplete_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL0,            0x100241C0,__READ_WRITE,__usb_endptctrl0_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL1,            0x100241C4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL2,            0x100241C8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL3,            0x100241CC,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL4,            0x100241D0,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL5,            0x100241D4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL6,            0x100241D8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL7,            0x100241DC,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL8,            0x100241E0,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL9,            0x100241E4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL10,           0x100241E8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL11,           0x100241EC,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL12,           0x100241F0,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL13,           0x100241F4,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL14,           0x100241F8,__READ_WRITE,__usb_endptctrl_bits);
__IO_REG32_BIT(UOG_ENDPTCTRL15,           0x100241FC,__READ_WRITE,__usb_endptctrl_bits);

/***************************************************************************
 **
 **  USB HOST1
 **
 ***************************************************************************/
__IO_REG32_BIT(UH1_ID,                    0x10024200,__READ      ,__usb_id_bits);
__IO_REG32_BIT(UH1_HWGENERAL,             0x10024204,__READ      ,__usb_hwgeneral_bits);
__IO_REG32_BIT(UH1_HWHOST,                0x10024208,__READ      ,__usb_hwhost_bits);
__IO_REG32_BIT(UH1_HWTXBUF,               0x10024210,__READ      ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(UH1_HWRXBUF,               0x10024214,__READ      ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(UH1_GPTIMER0LD,            0x10024280,__READ_WRITE,__usb_gptimer0ld_bits);
__IO_REG32_BIT(UH1_GPTIMER0CTRL,          0x10024284,__READ_WRITE,__usb_gptimer0ctrl_bits);
__IO_REG32(    UH1_GPTIMER1LD,            0x10024288,__READ_WRITE);
__IO_REG32(    UH1_GPTIMER1CTRL,          0x1002428C,__READ_WRITE);
__IO_REG16(    UH1_CAPLENGTH,             0x10024300,__READ      );
__IO_REG16(    UH1_HCIVERSION,            0x10024302,__READ      );
__IO_REG32_BIT(UH1_HCSPARAMS,             0x10024304,__READ      ,__usb_hcsparams_bits);
__IO_REG32_BIT(UH1_HCCPARAMS,             0x10024308,__READ      ,__usb_hccparams_bits);
__IO_REG32_BIT(UH1_USBCMD,                0x10024340,__READ_WRITE,__usb_usbcmd_bits);
__IO_REG32_BIT(UH1_USBSTS,                0x10024344,__READ_WRITE,__usb_usbsts_bits);
__IO_REG32_BIT(UH1_USBINTR,               0x10024348,__READ_WRITE,__usb_usbintr_bits);
__IO_REG32_BIT(UH1_FRINDEX,               0x1002434C,__READ_WRITE,__usb_frindex_bits);
__IO_REG32_BIT(UH1_PERIODICLISTBASE,      0x10024354,__READ_WRITE,__uh_periodiclistbase_bits);
#define UH1_DEVICEADDR      UH1_PERIODICLISTBASE
#define UH1_DEVICEADDR_bit  UH1_PERIODICLISTBASE_bit
__IO_REG32(    UH1_ASYNCLISTADDR,         0x10024358,__READ_WRITE);
__IO_REG32_BIT(UH1_BURSTSIZE,             0x10024360,__READ_WRITE,__usb_burstsize_bits);
__IO_REG32_BIT(UH1_TXFILLTUNING,          0x10024364,__READ_WRITE,__usb_txfilltuning_bits);
__IO_REG32_BIT(UH1_PORTSC1,               0x10024384,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UH1_USBMODE,               0x100243A8,__READ_WRITE,__usb_usbmode_bits);

/***************************************************************************
 **
 **  USB HOST2
 **
 ***************************************************************************/
__IO_REG32_BIT(UH2_ID,                    0x10024400,__READ      ,__usb_id_bits);
__IO_REG32_BIT(UH2_HWGENERAL,             0x10024404,__READ      ,__usb_hwgeneral_bits);
__IO_REG32_BIT(UH2_HWHOST,                0x10024408,__READ      ,__usb_hwhost_bits);
__IO_REG32_BIT(UH2_HWTXBUF,               0x10024410,__READ      ,__usb_hwtxbuf_bits);
__IO_REG32_BIT(UH2_HWRXBUF,               0x10024414,__READ      ,__usb_hwrxbuf_bits);
__IO_REG32_BIT(UH2_GPTIMER0LD,            0x10024480,__READ_WRITE,__usb_gptimer0ld_bits);
__IO_REG32_BIT(UH2_GPTIMER0CTRL,          0x10024484,__READ_WRITE,__usb_gptimer0ctrl_bits);
__IO_REG32(    UH2_GPTIMER1LD,            0x10024488,__READ_WRITE);
__IO_REG32(    UH2_GPTIMER1CTRL,          0x1002448C,__READ_WRITE);
__IO_REG16(    UH2_CAPLENGTH,             0x10024500,__READ      );
__IO_REG16(    UH2_HCIVERSION,            0x10024502,__READ      );
__IO_REG32_BIT(UH2_HCSPARAMS,             0x10024504,__READ      ,__usb_hcsparams_bits);
__IO_REG32_BIT(UH2_HCCPARAMS,             0x10024508,__READ      ,__usb_hccparams_bits);
__IO_REG32_BIT(UH2_USBCMD,                0x10024540,__READ_WRITE,__usb_usbcmd_bits);
__IO_REG32_BIT(UH2_USBSTS,                0x10024544,__READ_WRITE,__usb_usbsts_bits);
__IO_REG32_BIT(UH2_USBINTR,               0x10024548,__READ_WRITE,__usb_usbintr_bits);
__IO_REG32_BIT(UH2_FRINDEX,               0x1002454C,__READ_WRITE,__usb_frindex_bits);
__IO_REG32_BIT(UH2_PERIODICLISTBASE,      0x10024554,__READ_WRITE,__uh_periodiclistbase_bits);
#define UH2_DEVICEADDR      UH2_PERIODICLISTBASE
#define UH2_DEVICEADDR_bit  UH2_PERIODICLISTBASE_bit
__IO_REG32(    UH2_ASYNCLISTADDR,         0x10024558,__READ_WRITE);
__IO_REG32_BIT(UH2_BURSTSIZE,             0x10024560,__READ_WRITE,__usb_burstsize_bits);
__IO_REG32_BIT(UH2_TXFILLTUNING,          0x10024564,__READ_WRITE,__usb_txfilltuning_bits);
__IO_REG32_BIT(UH2_ULPIVIEW,              0x10024570,__READ_WRITE,__usb_ulpiview_bits);
__IO_REG32_BIT(UH2_PORTSC1,               0x10024584,__READ_WRITE,__usb_portsc_bits);
__IO_REG32_BIT(UH2_USBMODE,               0x100245A8,__READ_WRITE,__usb_usbmode_bits);

/***************************************************************************
 **
 **  USB
 **
 ***************************************************************************/
__IO_REG32_BIT(USB_CTRL,                  0x10024600,__READ_WRITE,__usb_ctrl_bits);
__IO_REG32_BIT(USB_OTG_MIRROR,            0x10024604,__READ_WRITE,__usb_otg_mirror_bits);

/***************************************************************************
 **
 **  GPT1
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL1,                     0x10003000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER1,                    0x10003004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP1,                     0x10003008,__READ_WRITE);
__IO_REG32(    TCR1,                      0x1000300C,__READ      );
__IO_REG32(    TCN1,                      0x10003010,__READ      );
__IO_REG32_BIT(TSTAT1,                    0x10003014,__READ_WRITE,__tstat_bits);


/***************************************************************************
 **
 **  GPT2
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL2,                     0x10004000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER2,                    0x10004004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP2,                     0x10004008,__READ_WRITE);
__IO_REG32(    TCR2,                      0x1000400C,__READ      );
__IO_REG32(    TCN2,                      0x10004010,__READ      );
__IO_REG32_BIT(TSTAT2,                    0x10004014,__READ_WRITE,__tstat_bits);

/***************************************************************************
 **
 **  GPT3
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL3,                     0x10005000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER3,                    0x10005004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP3,                     0x10005008,__READ_WRITE);
__IO_REG32(    TCR3,                      0x1000500C,__READ      );
__IO_REG32(    TCN3,                      0x10005010,__READ      );
__IO_REG32_BIT(TSTAT3,                    0x10005014,__READ_WRITE,__tstat_bits);

/***************************************************************************
 **
 **  GPT4
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL4,                     0x10019000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER4,                    0x10019004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP4,                     0x10019008,__READ_WRITE);
__IO_REG32(    TCR4,                      0x1001900C,__READ      );
__IO_REG32(    TCN4,                      0x10019010,__READ      );
__IO_REG32_BIT(TSTAT4,                    0x10019014,__READ_WRITE,__tstat_bits);

/***************************************************************************
 **
 **  GPT5
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL5,                     0x1001A000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER5,                    0x1001A004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP5,                     0x1001A008,__READ_WRITE);
__IO_REG32(    TCR5,                      0x1001A00C,__READ      );
__IO_REG32(    TCN5,                      0x1001A010,__READ      );
__IO_REG32_BIT(TSTAT5,                    0x1001A014,__READ_WRITE,__tstat_bits);

/***************************************************************************
 **
 **  GPT6
 **
 ***************************************************************************/
__IO_REG32_BIT(TCTL6,                     0x1001F000,__READ_WRITE,__tctl_bits);
__IO_REG32_BIT(TPRER6,                    0x1001F004,__READ_WRITE,__tprer_bits);
__IO_REG32(    TCMP6,                     0x1001F008,__READ_WRITE);
__IO_REG32(    TCR6,                      0x1001F00C,__READ      );
__IO_REG32(    TCN6,                      0x1001F010,__READ      );
__IO_REG32_BIT(TSTAT6,                    0x1001F014,__READ_WRITE,__tstat_bits);

/***************************************************************************
 **
 **  PWM
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR,                     0x10006000,__READ_WRITE,__pwmcr_bits);
__IO_REG32_BIT(PWMSR,                     0x10006004,__READ_WRITE,__pwmsr_bits);
__IO_REG32_BIT(PWMIR,                     0x10006008,__READ_WRITE,__pwmir_bits);
__IO_REG32_BIT(PWMSAR,                    0x1000600C,__READ_WRITE,__pwmsar_bits);
__IO_REG32_BIT(PWMPR,                     0x10006010,__READ_WRITE,__pwmpr_bits);
__IO_REG32_BIT(PWMCNR,                    0x10006014,__READ      ,__pwmcnr_bits);

/***************************************************************************
 **
 **  RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTC_HOURMIN,               0x10007000,__READ_WRITE,__rtc_hourmin_bits);
__IO_REG32_BIT(RTC_SECONDS,               0x10007004,__READ_WRITE,__rtc_seconds_bits);
__IO_REG32_BIT(RTC_ALRM_HM,               0x10007008,__READ_WRITE,__rtc_hourmin_bits);
__IO_REG32_BIT(RTC_ALRM_SEC,              0x1000700C,__READ_WRITE,__rtc_seconds_bits);
__IO_REG32_BIT(RTCCTL,                    0x10007010,__READ_WRITE,__rtcctl_bits);
__IO_REG32_BIT(RTCISR,                    0x10007014,__READ_WRITE,__rtcisr_bits);
__IO_REG32_BIT(RTCIENR,                   0x10007018,__READ_WRITE,__rtcisr_bits);
__IO_REG32_BIT(RTC_STPWCH,                0x1000701C,__READ_WRITE,__rtc_stpwch_bits);
__IO_REG32_BIT(RTC_DAYR,                  0x10007020,__READ_WRITE,__rtc_dayr_bits);
__IO_REG32_BIT(RTC_DAYALARM,              0x10007024,__READ_WRITE,__rtc_dayalarm_bits);

/***************************************************************************
 **
 **  WDOG
 **
 ***************************************************************************/
__IO_REG16_BIT(WCR,                       0x10002000,__READ_WRITE,__wcr_bits);
__IO_REG16(    WSR,                       0x10002002,__READ_WRITE);
__IO_REG16_BIT(WRSR,                      0x10002004,__READ      ,__wrsr_bits);

/***************************************************************************
 **
 **  AIPI1
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPI1_PSR0,                0x10000000,__READ_WRITE,__aipi1_bits);
__IO_REG32_BIT(AIPI1_PSR1,                0x10000004,__READ_WRITE,__aipi1_bits);
__IO_REG32_BIT(AIPI1_PAR,                 0x10000008,__READ_WRITE,__aipi1_bits);

/***************************************************************************
 **
 **  AIPI2
 **
 ***************************************************************************/
__IO_REG32_BIT(AIPI2_PSR0,                0x10020000,__READ_WRITE,__aipi2_bits);
__IO_REG32_BIT(AIPI2_PSR1,                0x10020004,__READ_WRITE,__aipi2_bits);
__IO_REG32_BIT(AIPI2_PAR,                 0x10020008,__READ_WRITE,__aipi2_bits);

/***************************************************************************
 **
 **  MAX
 **
 ***************************************************************************/
__IO_REG32_BIT(MPR0,                      0x1003F000,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(AMPR0,                     0x1003F004,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR0,                    0x1003F010,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(ASGPCR0,                   0x1003F014,__READ_WRITE,__asgpcr_bits);
__IO_REG32_BIT(MPR1,                      0x1003F100,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(AMPR1,                     0x1003F104,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR1,                    0x1003F110,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(ASGPCR1,                   0x1003F114,__READ_WRITE,__asgpcr_bits);
__IO_REG32_BIT(MPR2,                      0x1003F200,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(AMPR2,                     0x1003F204,__READ_WRITE,__mpr_bits);
__IO_REG32_BIT(SGPCR2,                    0x1003F210,__READ_WRITE,__sgpcr_bits);
__IO_REG32_BIT(ASGPCR2,                   0x1003F214,__READ_WRITE,__asgpcr_bits);
__IO_REG32_BIT(MGPCR0,                    0x1003F800,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR1,                    0x1003F900,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR2,                    0x1003FA00,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR3,                    0x1003FB00,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR4,                    0x1003FC00,__READ_WRITE,__mgpcr_bits);
__IO_REG32_BIT(MGPCR5,                    0x1003FD00,__READ_WRITE,__mgpcr_bits);

/***************************************************************************
 **
 **  DMAC
 **
 ***************************************************************************/
__IO_REG32_BIT(DCR,                       0x10001000,__READ_WRITE,__dcr_bits);
__IO_REG32_BIT(DISR,                      0x10001004,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DIMR,                      0x10001008,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DBTOSR,                    0x1000100C,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DRTOSR,                    0x10001010,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DSESR,                     0x10001014,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DBOSR,                     0x10001018,__READ_WRITE,__disr_bits);
__IO_REG32_BIT(DBTOCR,                    0x1000101C,__READ_WRITE,__dbtocr_bits);
__IO_REG32_BIT(WSRA,                      0x10001040,__READ_WRITE,__wsr_bits);
__IO_REG32_BIT(XSRA,                      0x10001044,__READ_WRITE,__xsr_bits);
__IO_REG32_BIT(YSRA,                      0x10001048,__READ_WRITE,__ysr_bits);
__IO_REG32_BIT(WSRB,                      0x1000104C,__READ_WRITE,__wsr_bits);
__IO_REG32_BIT(XSRB,                      0x10001050,__READ_WRITE,__xsr_bits);
__IO_REG32_BIT(YSRB,                      0x10001054,__READ_WRITE,__ysr_bits);
__IO_REG32(    SAR0,                      0x10001080,__READ_WRITE);
__IO_REG32(    DAR0,                      0x10001084,__READ_WRITE);
__IO_REG32_BIT(CNTR0,                     0x10001088,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR0,                      0x1000108C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR0,                     0x10001090,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR0,                      0x10001094,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR0,                     0x10001098,__READ_WRITE,__rtor_bits);
#define BUCR0_bit     RTOR0_bit
#define BUCR0         RTOR0
__IO_REG32_BIT(CCNR0,                     0x1000109C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR1,                      0x100010C0,__READ_WRITE);
__IO_REG32(    DAR1,                      0x100010C4,__READ_WRITE);
__IO_REG32_BIT(CNTR1,                     0x100010C8,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR1,                      0x100010CC,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR1,                     0x100010D0,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR1,                      0x100010D4,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR1,                     0x100010D8,__READ_WRITE,__rtor_bits);
#define BUCR1_bit     RTOR1_bit
#define BUCR1         RTOR1
__IO_REG32_BIT(CCNR1,                     0x100010DC,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR2,                      0x10001100,__READ_WRITE);
__IO_REG32(    DAR2,                      0x10001104,__READ_WRITE);
__IO_REG32_BIT(CNTR2,                     0x10001108,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR2,                      0x1000110C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR2,                     0x10001110,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR2,                      0x10001114,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR2,                     0x10001118,__READ_WRITE,__rtor_bits);
#define BUCR2_bit     RTOR2_bit
#define BUCR2         RTOR2
__IO_REG32_BIT(CCNR2,                     0x1000111C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR3,                      0x10001140,__READ_WRITE);
__IO_REG32(    DAR3,                      0x10001144,__READ_WRITE);
__IO_REG32_BIT(CNTR3,                     0x10001148,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR3,                      0x1000114C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR3,                     0x10001150,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR3,                      0x10001154,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR3,                     0x10001158,__READ_WRITE,__rtor_bits);
#define BUCR3_bit     RTOR3_bit
#define BUCR3         RTOR3
__IO_REG32_BIT(CCNR3,                     0x1000115C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR4,                      0x10001180,__READ_WRITE);
__IO_REG32(    DAR4,                      0x10001184,__READ_WRITE);
__IO_REG32_BIT(CNTR4,                     0x10001188,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR4,                      0x1000118C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR4,                     0x10001190,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR4,                      0x10001194,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR4,                     0x10001198,__READ_WRITE,__rtor_bits);
#define BUCR4_bit     RTOR4_bit
#define BUCR4         RTOR4
__IO_REG32_BIT(CCNR4,                     0x1000119C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR5,                      0x100011C0,__READ_WRITE);
__IO_REG32(    DAR5,                      0x100011C4,__READ_WRITE);
__IO_REG32_BIT(CNTR5,                     0x100011C8,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR5,                      0x100011CC,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR5,                     0x100011D0,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR5,                      0x100011D4,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR5,                     0x100011D8,__READ_WRITE,__rtor_bits);
#define BUCR5_bit     RTOR5_bit
#define BUCR5         RTOR5
__IO_REG32_BIT(CCNR5,                     0x100011DC,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR6,                      0x10001200,__READ_WRITE);
__IO_REG32(    DAR6,                      0x10001204,__READ_WRITE);
__IO_REG32_BIT(CNTR6,                     0x10001208,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR6,                      0x1000120C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR6,                     0x10001210,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR6,                      0x10001214,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR6,                     0x10001218,__READ_WRITE,__rtor_bits);
#define BUCR6_bit     RTOR6_bit
#define BUCR6         RTOR6
__IO_REG32_BIT(CCNR6,                     0x1000121C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR7,                      0x10001240,__READ_WRITE);
__IO_REG32(    DAR7,                      0x10001244,__READ_WRITE);
__IO_REG32_BIT(CNTR7,                     0x10001248,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR7,                      0x1000124C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR7,                     0x10001250,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR7,                      0x10001254,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR7,                     0x10001258,__READ_WRITE,__rtor_bits);
#define BUCR7_bit     RTOR7_bit
#define BUCR7         RTOR7
__IO_REG32_BIT(CCNR7,                     0x1000125C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR8,                      0x10001280,__READ_WRITE);
__IO_REG32(    DAR8,                      0x10001284,__READ_WRITE);
__IO_REG32_BIT(CNTR8,                     0x10001288,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR8,                      0x1000128C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR8,                     0x10001290,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR8,                      0x10001294,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR8,                     0x10001298,__READ_WRITE,__rtor_bits);
#define BUCR8_bit     RTOR8_bit
#define BUCR8         RTOR8
__IO_REG32_BIT(CCNR8,                     0x1000129C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR9,                      0x100012C0,__READ_WRITE);
__IO_REG32(    DAR9,                      0x100012C4,__READ_WRITE);
__IO_REG32_BIT(CNTR9,                     0x100012C8,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR9,                      0x100012CC,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR9,                     0x100012D0,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR9,                      0x100012D4,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR9,                     0x100012D8,__READ_WRITE,__rtor_bits);
#define BUCR9_bit     RTOR9_bit
#define BUCR9         RTOR9
__IO_REG32_BIT(CCNR9,                     0x100012DC,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR10,                     0x10001300,__READ_WRITE);
__IO_REG32(    DAR10,                     0x10001304,__READ_WRITE);
__IO_REG32_BIT(CNTR10,                    0x10001308,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR10,                     0x1000130C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR10,                    0x10001310,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR10,                     0x10001314,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR10,                    0x10001318,__READ_WRITE,__rtor_bits);
#define BUCR10_bit    RTOR10_bit
#define BUCR10        RTOR10
__IO_REG32_BIT(CCNR10,                    0x1000131C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR11,                     0x10001340,__READ_WRITE);
__IO_REG32(    DAR11,                     0x10001344,__READ_WRITE);
__IO_REG32_BIT(CNTR11,                    0x10001348,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR11,                     0x1000134C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR11,                    0x10001350,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR11,                     0x10001354,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR11,                    0x10001358,__READ_WRITE,__rtor_bits);
#define BUCR11_bit    RTOR11_bit
#define BUCR11        RTOR11
__IO_REG32_BIT(CCNR11,                    0x1000135C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR12,                     0x10001380,__READ_WRITE);
__IO_REG32(    DAR12,                     0x10001384,__READ_WRITE);
__IO_REG32_BIT(CNTR12,                    0x10001388,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR12,                     0x1000138C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR12,                    0x10001390,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR12,                     0x10001394,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR12,                    0x10001398,__READ_WRITE,__rtor_bits);
#define BUCR12_bit    RTOR12_bit
#define BUCR12        RTOR12
__IO_REG32_BIT(CCNR12,                    0x1000139C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR13,                     0x100013C0,__READ_WRITE);
__IO_REG32(    DAR13,                     0x100013C4,__READ_WRITE);
__IO_REG32_BIT(CNTR13,                    0x100013C8,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR13,                     0x100013CC,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR13,                    0x100013D0,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR13,                     0x100013D4,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR13,                    0x100013D8,__READ_WRITE,__rtor_bits);
#define BUCR13_bit    RTOR13_bit
#define BUCR13        RTOR13
__IO_REG32_BIT(CCNR13,                    0x100013DC,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR14,                     0x10001400,__READ_WRITE);
__IO_REG32(    DAR14,                     0x10001404,__READ_WRITE);
__IO_REG32_BIT(CNTR14,                    0x10001408,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR14,                     0x1000140C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR14,                    0x10001410,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR14,                     0x10001414,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR14,                    0x10001418,__READ_WRITE,__rtor_bits);
#define BUCR14_bit    RTOR14_bit
#define BUCR14        RTOR14
__IO_REG32_BIT(CCNR14,                    0x1000141C,__READ_WRITE,__ccnr_bits);
__IO_REG32(    SAR15,                     0x10001440,__READ_WRITE);
__IO_REG32(    DAR15,                     0x10001444,__READ_WRITE);
__IO_REG32_BIT(CNTR15,                    0x10001448,__READ_WRITE,__cntr_bits);
__IO_REG32_BIT(CCR15,                     0x1000144C,__READ_WRITE,__ccr_bits);
__IO_REG32_BIT(RSSR15,                    0x10001450,__READ_WRITE,__rssr_bits);
__IO_REG32_BIT(BLR15,                     0x10001454,__READ_WRITE,__blr_bits);
__IO_REG32_BIT(RTOR15,                    0x10001458,__READ_WRITE,__rtor_bits);
#define BUCR15_bit    RTOR15_bit
#define BUCR15        RTOR15
__IO_REG32_BIT(CCNR15,                    0x1000145C,__READ_WRITE,__ccnr_bits);

/***************************************************************************
 **
 **  AUDMUX
 **
 ***************************************************************************/
__IO_REG32_BIT(HPCR1,                     0x10016000,__READ_WRITE,__hpcr_bits);
__IO_REG32_BIT(HPCR2,                     0x10016004,__READ_WRITE,__hpcr_bits);
__IO_REG32_BIT(HPCR3,                     0x10016008,__READ_WRITE,__hpcr_bits);
__IO_REG32_BIT(PPCR1,                     0x10016010,__READ_WRITE,__ppcr123_bits);
__IO_REG32_BIT(PPCR2,                     0x10016014,__READ_WRITE,__ppcr123_bits);
__IO_REG32_BIT(PPCR3,                     0x1001601C,__READ_WRITE,__ppcr123_bits);

/***************************************************************************
 **
 **  CSI
 **
 ***************************************************************************/
__IO_REG32_BIT(CSICR1,                    0x80000000,__READ_WRITE,__csicr1_bits);
__IO_REG32_BIT(CSICR2,                    0x80000004,__READ_WRITE,__csicr2_bits);
__IO_REG32_BIT(CSISR,                     0x80000008,__READ_WRITE,__csisr_bits);
__IO_REG32(    CSISTATFIFO,               0x8000000C,__READ      );
__IO_REG32(    CSIRFIFO,                  0x80000010,__READ      );
__IO_REG32_BIT(CSIRXCNT,                  0x80000014,__READ_WRITE,__csirxcnt_bits);
__IO_REG32_BIT(CSICR3,                    0x8000001C,__READ_WRITE,__csicr3_bits);

/***************************************************************************
 **
 **  Video Codec
 **
 ***************************************************************************/
__IO_REG32_BIT(VideoCodec_CodeRun,        0x10023000, __WRITE     ,__vcCodeRun_bits);
__IO_REG32_BIT(VideoCodec_CodeDownLoad,   0x10023004, __WRITE     ,__vcCodeDownLoad_bits);
__IO_REG32_BIT(VideoCodec_HostIntReq,     0x10023008, __WRITE     ,__vcHostIntReq_bits);
__IO_REG32_BIT(VideoCodec_BitIntClear,    0x1002300C, __WRITE     ,__vcBitIntClear_bits);
__IO_REG32_BIT(VideoCodec_BitIntSts,      0x10023010, __READ      ,__vcBitIntSts_bits);
__IO_REG32_BIT(VideoCodec_BitCodeReset,   0x10023014, __WRITE     ,__vcBitCodeReset_bits);
__IO_REG32_BIT(VideoCodec_BitCurPc,       0x10023018, __READ      ,__vcBitCurPc_bits);
__IO_REG32(    VideoCodec_WorkBufAddr,    0x10023100, __READ_WRITE);
__IO_REG32(    VideoCodec_CodeBufAddr,    0x10023104, __READ_WRITE);
__IO_REG32(    VideoCodec_BitStreamCtrl,  0x10023108, __READ_WRITE);
__IO_REG32(    VideoCodec_FrameMemCtrl,   0x1002310C, __READ_WRITE);
__IO_REG32(    VideoCodec_SramAddr,       0x10023110, __READ_WRITE);
__IO_REG32(    VideoCodec_SramSize,       0x10023114, __READ_WRITE);
__IO_REG32(    VideoCodec_BitStreamRdPtr, 0x10023140, __READ_WRITE);
__IO_REG32(    VideoCodec_BitStreamWrPtr, 0x10023144, __READ_WRITE);
__IO_REG32(    VideoCodec_FrameNum,       0x10023148, __READ_WRITE);
__IO_REG32(    VideoCodec_BusyFlag,       0x10023160, __READ_WRITE);
__IO_REG32(    VideoCodec_RunCommand,     0x10023164, __READ_WRITE);
__IO_REG32(    VideoCodec_RunIndex,       0x10023168, __READ_WRITE);
__IO_REG32(    VideoCodec_RunCodStd,      0x1002316C, __READ_WRITE);
__IO_REG32(    VideoCodec_BitBufAddr,     0x10023180, __READ_WRITE);
#define VideoCodec_FrameSrcAddrY  VideoCodec_BitBufAddr
__IO_REG32(    VideoCodec_BitBufSize,     0x10023184, __READ_WRITE);
#define VideoCodec_FrameSrcAddrCb VideoCodec_BitBufSize
__IO_REG32(    VideoCodec_FrameIntAddrY,  0x10023188, __READ_WRITE);
#define VideoCodec_FrameSrcAddrCr VideoCodec_FrameIntAddrY
__IO_REG32(    VideoCodec_FrameIntAddrCb, 0x1002318C, __READ_WRITE);
#define VideoCodec_FrameDecAddrY  VideoCodec_FrameIntAddrCb 
__IO_REG32(    VideoCodec_FrameIntAddrCr, 0x10023190, __READ_WRITE);
#define VideoCodec_FrameDecAddrCb VideoCodec_FrameIntAddrCr
__IO_REG32(    VideoCodec_EncCodStd,      0x10023194, __READ_WRITE);
#define VideoCodec_FrameDecAddrCr VideoCodec_EncCodStd
__IO_REG32(    VideoCodec_EncSrcFormat,   0x10023198, __READ_WRITE);
__IO_REG32(    VideoCodec_EncMp4Para,     0x1002319C, __READ_WRITE);
__IO_REG32(    VideoCodec_Enc263Para,     0x100231A0, __READ_WRITE);
__IO_REG32(    VideoCodec_Enc264Para,     0x100231A4, __READ_WRITE);
__IO_REG32(    VideoCodec_EncSliceMode,   0x100231A8, __READ_WRITE);
__IO_REG32(    VideoCodec_EncGopNum,      0x100231AC, __READ_WRITE);
__IO_REG32(    VideoCodec_EncPictureQs,   0x100231B0, __READ_WRITE);
__IO_REG32(    VideoCodec_RetStatus,      0x100231C0, __READ_WRITE);
__IO_REG32(    VideoCodec_RetSrcFormat,   0x100231C4, __READ_WRITE);
__IO_REG32(    VideoCodec_RetMp4Info,     0x100231C8, __READ_WRITE);
__IO_REG32(    VideoCodec_Ret263Info,     0x100231CC, __READ_WRITE);
__IO_REG32(    VideoCodec_Ret264Info,     0x100231D0, __READ_WRITE);
/*__IO_REG32(    VideoCodec_FrameSrcAddrY,  0x10023180, __READ_WRITE);*/
/*__IO_REG32(    VideoCodec_FrameSrcAddrCb, 0x10023184, __READ_WRITE);*/
/*__IO_REG32(    VideoCodec_FrameSrcAddrCr, 0x10023188, __READ_WRITE);*/
/*__IO_REG32(    VideoCodec_FrameDecAddrY,  0x1002318C, __READ_WRITE);*/
/*__IO_REG32(    VideoCodec_FrameDecAddrCb, 0x10023190, __READ_WRITE);*/
/*__IO_REG32(    VideoCodec_FrameDecAddrCr, 0x10023194, __READ_WRITE);*/

/***************************************************************************
 **
 **  eMMA Lt PP
 **
 ***************************************************************************/
__IO_REG32_BIT(PP_CNTL,                   0x10026000,__READ_WRITE,__pp_cntl_bits);
__IO_REG32_BIT(PP_INTRCNTL,               0x10026004,__READ_WRITE,__pp_intrcntl_bits);
__IO_REG32_BIT(PP_INTRSTATUS,             0x10026008,__READ_WRITE,__pp_intrstatus_bits);
__IO_REG32(    PP_SOURCE_Y_PTR,           0x1002600C,__READ_WRITE);
__IO_REG32(    PP_SOURCE_CB_PTR,          0x10026010,__READ_WRITE);
__IO_REG32(    PP_SOURCE_CR_PTR,          0x10026014,__READ_WRITE);
__IO_REG32(    PP_DEST_RGB_PTR,           0x10026018,__READ_WRITE);
__IO_REG32(    PP_QUANTIZER_PTR,          0x1002601C,__READ_WRITE);
__IO_REG32_BIT(PP_PROCESS_PARA,           0x10026020,__READ_WRITE,__pp_process_para_bits);
__IO_REG32_BIT(PP_FRAME_WIDTH,            0x10026024,__READ_WRITE,__pp_frame_width_bits);
__IO_REG32_BIT(PP_DISPLAY_WIDTH,          0x10026028,__READ_WRITE,__pp_display_width_bits);
__IO_REG32_BIT(PP_IMAGE_SIZE,             0x1002602C,__READ_WRITE,__pp_image_size_bits);
__IO_REG32_BIT(PP_DEST_FRAME_FMT_CNTL,    0x10026030,__READ_WRITE,__pp_dest_frame_format_cntl_bits);
__IO_REG32_BIT(PP_RESIZE_INDEX,           0x10026034,__READ_WRITE,__pp_resize_index_bits);
__IO_REG32_BIT(PP_CSC_COEF_123,           0x10026038,__READ_WRITE,__pp_csc_coef_123_bits);
__IO_REG32_BIT(PP_CSC_COEF_4,             0x1002603C,__READ_WRITE,__pp_csc_coef_4_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL0,       0x10026100,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL1,       0x10026104,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL2,       0x10026108,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL3,       0x1002610C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL4,       0x10026110,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL5,       0x10026114,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL6,       0x10026118,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL7,       0x1002611C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL8,       0x10026120,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL9,       0x10026124,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL10,      0x10026128,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL11,      0x1002612C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL12,      0x10026130,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL13,      0x10026134,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL14,      0x10026138,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL15,      0x1002613C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL16,      0x10026140,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL17,      0x10026144,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL18,      0x10026148,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL19,      0x1002614C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL20,      0x10026150,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL21,      0x10026154,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL22,      0x10026158,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL23,      0x1002615C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL24,      0x10026160,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL25,      0x10026164,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL26,      0x10026168,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL27,      0x1002616C,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL28,      0x10026170,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL29,      0x10026174,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL30,      0x10026178,__WRITE     ,__pp_resize_coef_tbl_bits);
__IO_REG32_BIT(PP_RESIZE_COEF_TBL31,      0x1002617C,__WRITE     ,__pp_resize_coef_tbl_bits);

/***************************************************************************
 **
 **  eMMA Lt PrP
 **
 ***************************************************************************/
__IO_REG32_BIT(PRP_CNTL,                  0x10026400,__READ_WRITE,__prp_cntl_bits);
__IO_REG32_BIT(PRP_INTR_CNTL,             0x10026404,__READ_WRITE,__prp_intr_cntl_bits);
__IO_REG32_BIT(PRP_INTRSTATUS,            0x10026408,__READ_WRITE,__prp_intrstatus_bits);
__IO_REG32(    PRP_SOURCE_Y_PTR,          0x1002640C,__READ_WRITE);
__IO_REG32(    PRP_SOURCE_CB_PTR,         0x10026410,__READ_WRITE);
__IO_REG32(    PRP_SOURCE_CR_PTR,         0x10026414,__READ_WRITE);
__IO_REG32(    PRP_DEST_RGB1_PTR,         0x10026418,__READ_WRITE);
__IO_REG32(    PRP_DEST_RGB2_PTR,         0x1002641C,__READ_WRITE);
__IO_REG32(    PRP_DEST_Y_PTR,            0x10026420,__READ_WRITE);
__IO_REG32(    PRP_DEST_CB_PTR,           0x10026424,__READ_WRITE);
__IO_REG32(    PRP_DEST_CR_PTR,           0x10026428,__READ_WRITE);
__IO_REG32_BIT(PRP_SRC_FRAME_SIZE,        0x1002642C,__READ_WRITE,__prp_src_frame_size_bits);
__IO_REG32_BIT(PRP_DEST_CH1_LINE_STRIDE,  0x10026430,__READ_WRITE,__prp_dest_ch1_line_stride_bits);
__IO_REG32_BIT(PRP_SRC_PIXEL_FORMAT_CNTL, 0x10026434,__READ_WRITE,__prp_src_pixel_format_cntl_bits);
__IO_REG32_BIT(PRP_CH1_PIXEL_FORMAT_CNTL, 0x10026438,__READ_WRITE,__prp_ch1_pixel_format_cntl_bits);
__IO_REG32_BIT(PRP_CH1_OUT_IMAGE_SIZE,    0x1002643C,__READ_WRITE,__prp_ch1_out_image_size_bits);
__IO_REG32_BIT(PRP_CH2_OUT_IMAGE_SIZE,    0x10026440,__READ_WRITE,__prp_ch2_out_image_size_bits);
__IO_REG32_BIT(PRP_SRC_LINE_STRIDE,       0x10026444,__READ_WRITE,__prp_src_line_stride_bits);
__IO_REG32_BIT(PRP_CSC_COEF_012,          0x10026448,__READ_WRITE,__prp_csc_coef_012_bits);
__IO_REG32_BIT(PRP_CSC_COEF_345,          0x1002644C,__READ_WRITE,__prp_csc_coef_345_bits);
__IO_REG32_BIT(PRP_CSC_COEF_678,          0x10026450,__READ_WRITE,__prp_csc_coef_678_bits);
__IO_REG32_BIT(PRP_CH1_RZ_HORI_COEF1,     0x10026454,__READ_WRITE,__prp_rz_hori_coef1_bits);
__IO_REG32_BIT(PRP_CH1_RZ_HORI_COEF2,     0x10026458,__READ_WRITE,__prp_rz_hori_coef2_bits);
__IO_REG32_BIT(PRP_CH1_RZ_HORI_VALID,     0x1002645C,__READ_WRITE,__prp_rz_hori_valid_bits);
__IO_REG32_BIT(PRP_CH1_RZ_VERT_COEF1,     0x10026460,__READ_WRITE,__prp_rz_vert_coef1_bits);
__IO_REG32_BIT(PRP_CH1_RZ_VERT_COEF2,     0x10026464,__READ_WRITE,__prp_rz_vert_coef2_bits);
__IO_REG32_BIT(PRP_CH1_RZ_VERT_VALID,     0x10026468,__READ_WRITE,__prp_rz_vert_valid_bits);
__IO_REG32_BIT(PRP_CH2_RZ_HORI_COEF1,     0x1002646C,__READ_WRITE,__prp_rz_hori_coef1_bits);
__IO_REG32_BIT(PRP_CH2_RZ_HORI_COEF2,     0x10026470,__READ_WRITE,__prp_rz_hori_coef2_bits);
__IO_REG32_BIT(PRP_CH2_RZ_HORI_VALID,     0x10026474,__READ_WRITE,__prp_rz_hori_valid_bits);
__IO_REG32_BIT(PRP_CH2_RZ_VERT_COEF1,     0x10026478,__READ_WRITE,__prp_rz_vert_coef1_bits);
__IO_REG32_BIT(PRP_CH2_RZ_VERT_COEF2,     0x1002647C,__READ_WRITE,__prp_rz_vert_coef2_bits);
__IO_REG32_BIT(PRP_CH2_RZ_VERT_VALID,     0x10026480,__READ_WRITE,__prp_rz_vert_valid_bits);

/***************************************************************************
 **
 **  SSI1
 **
 ***************************************************************************/
__IO_REG32(    SSI1_STX0,                 0x10010000,__READ_WRITE);
__IO_REG32(    SSI1_STX1,                 0x10010004,__READ_WRITE);
__IO_REG32(    SSI1_SRX0,                 0x10010008,__READ      );
__IO_REG32(    SSI1_SRX1,                 0x1001000C,__READ      );
__IO_REG32_BIT(SSI1_SCR,                  0x10010010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI1_SISR,                 0x10010014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI1_SIER,                 0x10010018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI1_STCR,                 0x1001001C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI1_SRCR,                 0x10010020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI1_STCCR,                0x10010024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SRCCR,                0x10010028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI1_SFCSR,                0x1001002C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI1_STR,                  0x10010030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI1_SOR,                  0x10010034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI1_SACNT,                0x10010038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI1_SACADD,               0x1001003C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI1_SACDAT,               0x10010040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI1_SATAG,                0x10010044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI1_STMSK,                0x10010048,__READ_WRITE);
__IO_REG32(    SSI1_SRMSK,                0x1001004C,__READ_WRITE);
__IO_REG32_BIT(SSI1_SACCST,               0x10010050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI1_SACCEN,               0x10010054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI1_SACCDIS,              0x10010058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  SSI2
 **
 ***************************************************************************/
__IO_REG32(    SSI2_STX0,                 0x10011000,__READ_WRITE);
__IO_REG32(    SSI2_STX1,                 0x10011004,__READ_WRITE);
__IO_REG32(    SSI2_SRX0,                 0x10011008,__READ      );
__IO_REG32(    SSI2_SRX1,                 0x1001100C,__READ      );
__IO_REG32_BIT(SSI2_SCR,                  0x10011010,__READ_WRITE,__scsr_bits);
__IO_REG32_BIT(SSI2_SISR,                 0x10011014,__READ      ,__sisr_bits);
__IO_REG32_BIT(SSI2_SIER,                 0x10011018,__READ_WRITE,__sier_bits);
__IO_REG32_BIT(SSI2_STCR,                 0x1001101C,__READ_WRITE,__stcr_bits);
__IO_REG32_BIT(SSI2_SRCR,                 0x10011020,__READ_WRITE,__srcr_bits);
__IO_REG32_BIT(SSI2_STCCR,                0x10011024,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SRCCR,                0x10011028,__READ_WRITE,__ssi_ccr_bits);
__IO_REG32_BIT(SSI2_SFCSR,                0x1001102C,__READ_WRITE,__ssi_sfcsr_bits);
__IO_REG32_BIT(SSI2_STR,                  0x10011030,__READ_WRITE,__ssi_str_bits);
__IO_REG32_BIT(SSI2_SOR,                  0x10011034,__READ_WRITE,__ssi_sor_bits);
__IO_REG32_BIT(SSI2_SACNT,                0x10011038,__READ_WRITE,__ssi_sacnt_bits);
__IO_REG32_BIT(SSI2_SACADD,               0x1001103C,__READ_WRITE,__ssi_sacadd_bits);
__IO_REG32_BIT(SSI2_SACDAT,               0x10011040,__READ_WRITE,__ssi_sacdat_bits);
__IO_REG32_BIT(SSI2_SATAG,                0x10011044,__READ_WRITE,__ssi_satag_bits);
__IO_REG32(    SSI2_STMSK,                0x10011048,__READ_WRITE);
__IO_REG32(    SSI2_SRMSK,                0x1001104C,__READ_WRITE);
__IO_REG32_BIT(SSI2_SACCST,               0x10011050,__READ      ,__ssi_saccst_bits);
__IO_REG32_BIT(SSI2_SACCEN,               0x10011054,__WRITE     ,__ssi_saccen_bits);
__IO_REG32_BIT(SSI2_SACCDIS,              0x10011058,__WRITE     ,__ssi_saccdis_bits);

/***************************************************************************
 **
 **  LCDC
 **
 ***************************************************************************/
__IO_REG32(    LSSAR,                     0x10021000,__READ_WRITE);
__IO_REG32_BIT(_LSR,                      0x10021004,__READ_WRITE,__lsr_bits);
__IO_REG32_BIT(LVPWR,                     0x10021008,__READ_WRITE,__lvpwr_bits);
__IO_REG32_BIT(LCPR,                      0x1002100C,__READ_WRITE,__lcpr_bits);
__IO_REG32_BIT(LCWHB,                     0x10021010,__READ_WRITE,__lcwhb_bits);
__IO_REG32_BIT(LCCMR,                     0x10021014,__READ_WRITE,__lccmr_bits);
__IO_REG32_BIT(LPCR,                      0x10021018,__READ_WRITE,__lpcr_bits);
__IO_REG32_BIT(LHCR,                      0x1002101C,__READ_WRITE,__lhcr_bits);
__IO_REG32_BIT(LVCR,                      0x10021020,__READ_WRITE,__lvcr_bits);
__IO_REG32_BIT(LPOR,                      0x10021024,__READ_WRITE,__lpor_bits);
__IO_REG32_BIT(LSCR,                      0x10021028,__READ_WRITE,__lscr_bits);
__IO_REG32_BIT(LPCCR,                     0x1002102C,__READ_WRITE,__lpccr_bits);
__IO_REG32_BIT(LDCR,                      0x10021030,__READ_WRITE,__ldcr_bits);
__IO_REG32_BIT(LRMCR,                     0x10021034,__READ_WRITE,__lrmcr_bits);
__IO_REG32_BIT(LICR,                      0x10021038,__READ_WRITE,__licr_bits);
__IO_REG32_BIT(LIER,                      0x1002103C,__READ_WRITE,__lier_bits);
__IO_REG32_BIT(LISR,                      0x10021040,__READ      ,__lisr_bits);
__IO_REG32(    LGWSAR,                    0x10021050,__READ_WRITE);
__IO_REG32_BIT(LGWSR,                     0x10021054,__READ_WRITE,__lgwsr_bits);
__IO_REG32_BIT(LGWVPWR,                   0x10021058,__READ_WRITE,__lgwvpwr_bits);
__IO_REG32_BIT(LGWPOR,                    0x1002105C,__READ_WRITE,__lgwpor_bits);
__IO_REG32_BIT(LGWPR,                     0x10021060,__READ_WRITE,__lgwpr_bits);
__IO_REG32_BIT(LGWCR,                     0x10021064,__READ_WRITE,__lgwcr_bits);
__IO_REG32_BIT(LGWDCR,                    0x10021068,__READ_WRITE,__lgwdcr_bits);
__IO_REG32_BIT(LAUSCR,                    0x10021080,__READ_WRITE,__lauscr_bits);
__IO_REG32_BIT(LAUSCCR,                   0x10021084,__READ_WRITE,__lausccr_bits);

/***************************************************************************
 **
 **  Smart Liquid Crystal Display Controller (SLCDC)
 **
 ***************************************************************************/
__IO_REG32(    DATA_BASE_ADDR,            0x10022000,__READ_WRITE);
#define DATABASEADR         DATA_BASE_ADDR
__IO_REG32_BIT(ATA_BUFF_SIZE,             0x10022004,__READ_WRITE,__data_buff_size_bits);
#define LCDDATABUFSIZE      ATA_BUFF_SIZE
#define LCDDATABUFSIZE_bit  ATA_BUFF_SIZE_bit
__IO_REG32(    CMD_BASE_ADDR,             0x10022008,__READ_WRITE);
#define COMBASEADR          CMD_BASE_ADDR
__IO_REG32_BIT(CMD_BUFF_SIZE,             0x1002200C,__READ_WRITE,__cmd_buff_size_bits);
#define COMBUFSIZ           CMD_BUFF_SIZE
#define COMBUFSIZ_bit       CMD_BUFF_SIZE_bit
__IO_REG32_BIT(STRING_SIZE,               0x10022010,__READ_WRITE,__string_size_bits);
#define LCDCOMSTRINGSIZ     STRING_SIZE
#define LCDCOMSTRINGSIZ_bit STRING_SIZE_bit
__IO_REG32_BIT(FIFO_CONFIG,               0x10022014,__READ_WRITE,__fifo_config_bits);
#define FIFOCONFIG          FIFO_CONFIG
#define FIFOCONFIG_bit      FIFO_CONFIG_bit
__IO_REG32_BIT(LCD_CONFIG,                0x10022018,__READ_WRITE,__lcd_config_bits);
#define LCDCONFIG           LCD_CONFIG
#define LCDCONFIG_bit       LCD_CONFIG_bit
__IO_REG32_BIT(LCDTRANSCONFIG,            0x1002201C,__READ_WRITE,__lcdtransconfig_bits);
__IO_REG32_BIT(DMA_CTRL_STAT,             0x10022020,__READ_WRITE,__dma_ctrl_stat_bits);
#define SLCDCCONTROL        DMA_CTRL_STAT
#define SLCDCCONTROL_bit    DMA_CTRL_STAT_bit
__IO_REG32_BIT(LCD_CLK_CONFIG,            0x10022024,__READ_WRITE,__lcd_clk_config_bits);
#define LCDCLOCKCONFIG      LCD_CLK_CONFIG
#define LCDCLOCKCONFIG_bit  LCD_CLK_CONFIG_bit
__IO_REG32_BIT(LCD_WRITE_DATA,            0x10022028,__READ_WRITE,__lcd_write_data_bits);

/* Assembler specific declarations  ****************************************/

#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__ */

/***************************************************************************
 **
 **  Interrupt vector table
 **
 ***************************************************************************/
#define RESETV        0x00  /* Reset                                       */
#define UNDEFV        0x04  /* Undefined instruction                       */
#define SWIV          0x08  /* Software interrupt                          */
#define PABORTV       0x0c  /* Prefetch abort                              */
#define DABORTV       0x10  /* Data abort                                  */
#define IRQV          0x18  /* Normal interrupt                            */
#define FIQV          0x1c  /* Fast interrupt                              */

/***************************************************************************
 **
 **   MCIX27 DMA channels
 **
 ***************************************************************************/
/*#define Reserved               0                 Unused*/
#define DMA_CSPI3_RX_FIFO      1
#define DMA_CSPI3_TX_FIFO      2
#define DMA_EXT_DMA            3
#define DMA_MSHC               4
/*#define Reserved               5                 Unused*/
#define DMA_SDHC2              6
#define DMA_SDHC1              7
#define DMA_SSI2_RX0_FIFO      8
#define DMA_SSI2_TX0_FIFO      9
#define DMA_SSI2_RX1_FIFO      10
#define DMA_SSI2_TX1_FIFO      11
#define DMA_SSI1_RX0_FIFO      12
#define DMA_SSI1_TX0_FIFO      13
#define DMA_SSI1_RX1_FIFO      14
#define DMA_SSI1_TX1_FIFO      15
#define DMA_CSPI2_RX_FIFO      16
#define DMA_CSPI2_TX_FIFO      17
#define DMA_CSPI1_RX_FIFO      18
#define DMA_CSPI1_TX_FIFO      19
#define DMA_UART4_RX_FIFO      20
#define DMA_UART4_TX_FIFO      21
#define DMA_UART3_RX_FIFO      22
#define DMA_UART3_TX_FIFO      23
#define DMA_UART2_RX_FIFO      24
#define DMA_UART2_TX_FIFO      25
#define DMA_UART1_RX_FIFO      26
#define DMA_UART1_TX_FIFO      27
#define DMA_ATA_TX_FIFO        28
#define DMA_ATA_RCV_FIFO       29
#define DMA_CSI_STAT_FIFO      30
#define DMA_CSI_RX_FIFO        31
#define DMA_UART5_TX_FIFO      32
#define DMA_UART5_RX_FIFO      33
#define DMA_UART6_TX_FIFO      34
#define DMA_UART6_RX_FIFO      35
#define DMA_SDHC3              36
#define DMA_NFC                37

/***************************************************************************
 **
 **   MCIX27 interrupt sources
 **
 ***************************************************************************/
/*#define Reserved               0                 Unused*/
#define INT_I2C2               1              /* I2C Bus Controller (I2C2)*/
#define INT_GPT6               2              /* General Purpose Timer (GPT6)*/
#define INT_GPT5               3              /* General Purpose Timer (GPT5)*/
#define INT_GPT4               4              /* General Purpose Timer (GPT4)*/
#define INT_RTIC               5              /* Real Time Integrity Checker (RTIC)*/
#define INT_CSPI3              6              /* Configurable SPI (CSPI3)*/
#define INT_SDHC               7              /* Secured Digital Host Controller (SDHC)*/
#define INT_GPIO               8              /* General Purpose Input/Output (GPIO)*/
#define INT_SDHC3              9              /* Secure Digital Host Controller (SDHC3)*/
#define INT_SDHC2              10             /* Secure Digital Host Controller (SDHC2)*/
#define INT_SDHC1              11             /* Secure Digital Host Controller (SDHC1)*/
#define INT_I2C1               12             /* I2C Bus Controller (I2C1)*/
#define INT_SS2                13             /* Synchronous Serial Interface (SSI2)*/
#define INT_SS1                14             /* Synchronous Serial Interface (SSI1)*/
#define INT_CSPI2              15             /* Configurable SPI (CSPI2)*/
#define INT_CSPI1              16             /* Configurable SPI (CSPI1)*/
#define INT_UART4              17             /* UART4*/
#define INT_UART3              18             /* UART3*/
#define INT_UART2              19             /* UART2*/
#define INT_UART1              20             /* UART1*/
#define INT_KPP                21             /* Key Pad Port (KPP)*/
#define INT_RTC                22             /* Real-Time Clock (RTC)*/
#define INT_PWM                23             /* Pulse Width Modulator (PWM)*/
#define INT_GPT3               24             /* General Purpose Timer (GPT3)*/
#define INT_GPT2               25             /* General Purpose Timer (GPT2)*/
#define INT_GPT1               26             /* General Purpose Timer (GPT1)*/
#define INT_WDOG               27             /* Watchdog (WDOG)*/
#define INT_PCMCIA             28             /* PCMCIA/CF Host Controller (PCMCIA)*/
#define INT_NFC                29             /* Nand Flash Controller (NFC)*/
#define INT_ATA                30             /* Advvanced Technology Attachment (ATA)*/
#define INT_CSI                31             /* CMOS Sensor Interface (CSI)*/
#define INT_DMACH0             32             /* DMA Channel 0*/
#define INT_DMACH1             33             /* DMA Channel 1*/
#define INT_DMACH2             34             /* DMA Channel 2*/
#define INT_DMACH3             35             /* DMA Channel 3*/
#define INT_DMACH4             36             /* DMA Channel 4*/
#define INT_DMACH5             37             /* DMA Channel 5*/
#define INT_DMACH6             38             /* DMA Channel 6*/
#define INT_DMACH7             39             /* DMA Channel 7*/
#define INT_DMACH8             40             /* DMA Channel 8*/
#define INT_DMACH9             41             /* DMA Channel 9*/
#define INT_DMACH10            42             /* DMA Channel 10*/
#define INT_DMACH11            43             /* DMA Channel 11*/
#define INT_DMACH12            44             /* DMA Channel 12*/
#define INT_DMACH13            45             /* DMA Channel 13*/
#define INT_DMACH14            46             /* DMA Channel 14*/
#define INT_DMACH15            47             /* DMA Channel 15*/
#define INT_UART6              48             /* UART6*/
#define INT_UART5              49             /* UART5*/
#define INT_FEC                50             /* Fast Ethernet Controller*/
#define INT_EMMAPRP            51             /* eMMA Pre Processor Interrupt*/
#define INT_EMMAPP             52             /* eMMA Post Processor Interrupt*/
#define INT_H264               53             /* H264*/
#define INT_USBHS1             54             /* USB HOST1*/
#define INT_USBHS2             55             /* USB HOST2*/
#define INT_USBOTG             56             /* USB OTG*/
#define INT_SMN                57             /* SCC SMN*/
#define INT_SCM                58             /* SCC SCM*/
#define INT_SAHARA             59             /* Run-Time Integrity Checker (RTIC)*/
#define INT_SLCDC              60             /* Smart LCD Controller (SLCDC)*/
#define INT_LCDC               61             /* LCD Controller (LCDC)*/
#define INT_IIM                62             /* IC Identify Module (IIM)*/
#define INT_DPTC               63             /* Dynamic Process Temperature Compensate (DPTC)*/

#endif    /* __MCIX27_H */
