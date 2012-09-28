/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Marvell PXA255
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2007
 **
 **    $Revision: 30251 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOPXA255_H
#define __IOPXA255_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    PXA255 SPECIAL FUNCTION REGISTERS
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

/* Power Manager Control Register (PMCR) */
typedef struct{
__REG32 IDAE              : 1;
__REG32                   :31;
} __pmcr_bits;

/* Power Manager Sleep Status Register (PSSR) */
typedef struct{
__REG32 SSS               : 1;
__REG32 BFS               : 1;
__REG32 VFS               : 1;
__REG32                   : 1;
__REG32 PH                : 1;
__REG32 RDH               : 1;
__REG32                   :26;
} __pssr_bits;

/* Power Manager Wake-Up Enable Register (PWER) */
typedef struct{
__REG32 WE0               : 1;
__REG32 WE1               : 1;
__REG32 WE2               : 1;
__REG32 WE3               : 1;
__REG32 WE4               : 1;
__REG32 WE5               : 1;
__REG32 WE6               : 1;
__REG32 WE7               : 1;
__REG32 WE8               : 1;
__REG32 WE9               : 1;
__REG32 WE10              : 1;
__REG32 WE11              : 1;
__REG32 WE12              : 1;
__REG32 WE13              : 1;
__REG32 WE14              : 1;
__REG32 WE15              : 1;
__REG32                   :15;
__REG32 WERTC             : 1;
} __pwer_bits;

/* Power Manager Rising-Edge Detect Enable Register (PRER) */
typedef struct{
__REG32 RE0               : 1;
__REG32 RE1               : 1;
__REG32 RE2               : 1;
__REG32 RE3               : 1;
__REG32 RE4               : 1;
__REG32 RE5               : 1;
__REG32 RE6               : 1;
__REG32 RE7               : 1;
__REG32 RE8               : 1;
__REG32 RE9               : 1;
__REG32 RE10              : 1;
__REG32 RE11              : 1;
__REG32 RE12              : 1;
__REG32 RE13              : 1;
__REG32 RE14              : 1;
__REG32 RE15              : 1;
__REG32                   :16;
} __prer_bits;

/* Power Manager Falling-Edge Detect Enable Register (PFER) */
typedef struct{
__REG32 FE0               : 1;
__REG32 FE1               : 1;
__REG32 FE2               : 1;
__REG32 FE3               : 1;
__REG32 FE4               : 1;
__REG32 FE5               : 1;
__REG32 FE6               : 1;
__REG32 FE7               : 1;
__REG32 FE8               : 1;
__REG32 FE9               : 1;
__REG32 FE10              : 1;
__REG32 FE11              : 1;
__REG32 FE12              : 1;
__REG32 FE13              : 1;
__REG32 FE14              : 1;
__REG32 FE15              : 1;
__REG32                   :16;
} __pfer_bits;

/* Power Manager Edge-Detect Status Register (PEDR) */
typedef struct{
__REG32 ED0               : 1;
__REG32 ED1               : 1;
__REG32 ED2               : 1;
__REG32 ED3               : 1;
__REG32 ED4               : 1;
__REG32 ED5               : 1;
__REG32 ED6               : 1;
__REG32 ED7               : 1;
__REG32 ED8               : 1;
__REG32 ED9               : 1;
__REG32 ED10              : 1;
__REG32 ED11              : 1;
__REG32 ED12              : 1;
__REG32 ED13              : 1;
__REG32 ED14              : 1;
__REG32 ED15              : 1;
__REG32                   :16;
} __pedr_bits;

/* Power Manager General Configuration Register (PCFR) */
typedef struct{
__REG32 OPDE              : 1;
__REG32 FP                : 1;
__REG32 FS                : 1;
__REG32                   :29;
} __pcfr_bits;

/* Power Manager GPIO Sleep-State Registers (PGSR0) */
typedef struct{
__REG32 SS0               : 1;
__REG32 SS1               : 1;
__REG32 SS2               : 1;
__REG32 SS3               : 1;
__REG32 SS4               : 1;
__REG32 SS5               : 1;
__REG32 SS6               : 1;
__REG32 SS7               : 1;
__REG32 SS8               : 1;
__REG32 SS9               : 1;
__REG32 SS10              : 1;
__REG32 SS11              : 1;
__REG32 SS12              : 1;
__REG32 SS13              : 1;
__REG32 SS14              : 1;
__REG32 SS15              : 1;
__REG32 SS16              : 1;
__REG32 SS17              : 1;
__REG32 SS18              : 1;
__REG32 SS19              : 1;
__REG32 SS20              : 1;
__REG32 SS21              : 1;
__REG32 SS22              : 1;
__REG32 SS23              : 1;
__REG32 SS24              : 1;
__REG32 SS25              : 1;
__REG32 SS26              : 1;
__REG32 SS27              : 1;
__REG32 SS28              : 1;
__REG32 SS29              : 1;
__REG32 SS30              : 1;
__REG32 SS31              : 1;
} __pgsr0_bits;

/* Power Manager GPIO Sleep-State Registers (PGSR1) */
typedef struct{
__REG32 SS32              : 1;
__REG32 SS33              : 1;
__REG32 SS34              : 1;
__REG32 SS35              : 1;
__REG32 SS36              : 1;
__REG32 SS37              : 1;
__REG32 SS38              : 1;
__REG32 SS39              : 1;
__REG32 SS40              : 1;
__REG32 SS41              : 1;
__REG32 SS42              : 1;
__REG32 SS43              : 1;
__REG32 SS44              : 1;
__REG32 SS45              : 1;
__REG32 SS46              : 1;
__REG32 SS47              : 1;
__REG32 SS48              : 1;
__REG32 SS49              : 1;
__REG32 SS50              : 1;
__REG32 SS51              : 1;
__REG32 SS52              : 1;
__REG32 SS53              : 1;
__REG32 SS54              : 1;
__REG32 SS55              : 1;
__REG32 SS56              : 1;
__REG32 SS57              : 1;
__REG32 SS58              : 1;
__REG32 SS59              : 1;
__REG32 SS60              : 1;
__REG32 SS61              : 1;
__REG32 SS62              : 1;
__REG32 SS63              : 1;
} __pgsr1_bits;

/* Power Manager GPIO Sleep-State Registers (PGSR2) */
typedef struct{
__REG32 SS64              : 1;
__REG32 SS65              : 1;
__REG32 SS66              : 1;
__REG32 SS67              : 1;
__REG32 SS68              : 1;
__REG32 SS69              : 1;
__REG32 SS70              : 1;
__REG32 SS71              : 1;
__REG32 SS72              : 1;
__REG32 SS73              : 1;
__REG32 SS74              : 1;
__REG32 SS75              : 1;
__REG32 SS76              : 1;
__REG32 SS77              : 1;
__REG32 SS78              : 1;
__REG32 SS79              : 1;
__REG32 SS80              : 1;
__REG32 SS81              : 1;
__REG32 SS82              : 1;
__REG32 SS83              : 1;
__REG32 SS84              : 1;
__REG32                   :11;
} __pgsr2_bits;

/* Reset Controller Status Register (RCSR) */
typedef struct{
__REG32 HWR               : 1;
__REG32 WDR               : 1;
__REG32 SMR               : 1;
__REG32 GPR               : 1;
__REG32                   :28;
} __rcsr_bits;

/* Power Manager Fast Sleep Walk-up Configuration Register (PMFW) */
typedef struct{
__REG32                   : 1;
__REG32 FWAKE             : 1;
__REG32                   :30;
} __pmfw_bits;

/* Core Clock Configuration Register (CCCR) */
typedef struct{
__REG32 L                 : 5;
__REG32 M                 : 2;
__REG32 N                 : 3;
__REG32                   :22;
} __cccr_bits;

/* Clock Enable Register (CKEN) */
typedef struct{
__REG32 PWM0_CLKEN          : 1;
__REG32 PWM1_CLKEN          : 1;
__REG32 AC97_CLKEN          : 1;
__REG32 SPI_CLKEN           : 1;
__REG32 HWUART_CLKEN        : 1;
__REG32 STUART_CLKEN        : 1;
__REG32 FFUART_CLKEN        : 1;
__REG32 BTUART_CLKEN        : 1;
__REG32 I2S_CLKEN           : 1;
__REG32 NSSP_CLKEN          : 1;
__REG32                     : 1;
__REG32 USB_CLKEN           : 1;
__REG32 MMC_CLKEN           : 1;
__REG32 FICP_CLKEN          : 1;
__REG32 I2C_CLKEN           : 1;
__REG32                     : 1;
__REG32 LCD_CLKEN           : 1;
__REG32                     :15;
} __cken_bits;

/* Oscillator Configuration Register (OSCC) */
typedef struct{
__REG32 OOK               : 1;
__REG32 OON               : 1;
__REG32                   :30;
} __oscc_bits;

/* GPIO Pin-Level Registers (GPLR0) */
typedef struct{
__REG32 PL0               : 1;
__REG32 PL1               : 1;
__REG32 PL2               : 1;
__REG32 PL3               : 1;
__REG32 PL4               : 1;
__REG32 PL5               : 1;
__REG32 PL6               : 1;
__REG32 PL7               : 1;
__REG32 PL8               : 1;
__REG32 PL9               : 1;
__REG32 PL10              : 1;
__REG32 PL11              : 1;
__REG32 PL12              : 1;
__REG32 PL13              : 1;
__REG32 PL14              : 1;
__REG32 PL15              : 1;
__REG32 PL16              : 1;
__REG32 PL17              : 1;
__REG32 PL18              : 1;
__REG32 PL19              : 1;
__REG32 PL20              : 1;
__REG32 PL21              : 1;
__REG32 PL22              : 1;
__REG32 PL23              : 1;
__REG32 PL24              : 1;
__REG32 PL25              : 1;
__REG32 PL26              : 1;
__REG32 PL27              : 1;
__REG32 PL28              : 1;
__REG32 PL29              : 1;
__REG32 PL30              : 1;
__REG32 PL31              : 1;
} __gplr0_bits;

/* GPIO Pin-Level Registers (GPLR1) */
typedef struct{
__REG32 PL32              : 1;
__REG32 PL33              : 1;
__REG32 PL34              : 1;
__REG32 PL35              : 1;
__REG32 PL36              : 1;
__REG32 PL37              : 1;
__REG32 PL38              : 1;
__REG32 PL39              : 1;
__REG32 PL40              : 1;
__REG32 PL41              : 1;
__REG32 PL42              : 1;
__REG32 PL43              : 1;
__REG32 PL44              : 1;
__REG32 PL45              : 1;
__REG32 PL46              : 1;
__REG32 PL47              : 1;
__REG32 PL48              : 1;
__REG32 PL49              : 1;
__REG32 PL50              : 1;
__REG32 PL51              : 1;
__REG32 PL52              : 1;
__REG32 PL53              : 1;
__REG32 PL54              : 1;
__REG32 PL55              : 1;
__REG32 PL56              : 1;
__REG32 PL57              : 1;
__REG32 PL58              : 1;
__REG32 PL59              : 1;
__REG32 PL60              : 1;
__REG32 PL61              : 1;
__REG32 PL62              : 1;
__REG32 PL63              : 1;
} __gplr1_bits;

/* GPIO Pin-Level Registers (GPLR2) */
typedef struct{
__REG32 PL64              : 1;
__REG32 PL65              : 1;
__REG32 PL66              : 1;
__REG32 PL67              : 1;
__REG32 PL68              : 1;
__REG32 PL69              : 1;
__REG32 PL70              : 1;
__REG32 PL71              : 1;
__REG32 PL72              : 1;
__REG32 PL73              : 1;
__REG32 PL74              : 1;
__REG32 PL75              : 1;
__REG32 PL76              : 1;
__REG32 PL77              : 1;
__REG32 PL78              : 1;
__REG32 PL79              : 1;
__REG32 PL80              : 1;
__REG32 PL81              : 1;
__REG32 PL82              : 1;
__REG32 PL83              : 1;
__REG32 PL84              : 1;
__REG32                   :11;
} __gplr2_bits;

/* GPIO Pin Direction Registers (GPDR0)
   GPIO Pin Bit-Wise Set Direction Registers (GSDR0)
   GPIO Pin Bit-Wise Clear Direction Registers (GCDR0)
   GPIO Bit-wise Set Rising-Edge (GSRER0) Detect-Enable Registers
   GPIO Bit-wise Clear Rising-Edge (GCRER0) Detect-Enable Registers
   GPIO Bit-wise Set Falling-Edge (GSFER0) Detect-Enable Registers
   GPIO Bit-wise Clear Falling-Edge (GCFER0) Detect-Enable Registers */
typedef struct{
__REG32 PD0               : 1;
__REG32 PD1               : 1;
__REG32 PD2               : 1;
__REG32 PD3               : 1;
__REG32 PD4               : 1;
__REG32 PD5               : 1;
__REG32 PD6               : 1;
__REG32 PD7               : 1;
__REG32 PD8               : 1;
__REG32 PD9               : 1;
__REG32 PD10              : 1;
__REG32 PD11              : 1;
__REG32 PD12              : 1;
__REG32 PD13              : 1;
__REG32 PD14              : 1;
__REG32 PD15              : 1;
__REG32 PD16              : 1;
__REG32 PD17              : 1;
__REG32 PD18              : 1;
__REG32 PD19              : 1;
__REG32 PD20              : 1;
__REG32 PD21              : 1;
__REG32 PD22              : 1;
__REG32 PD23              : 1;
__REG32 PD24              : 1;
__REG32 PD25              : 1;
__REG32 PD26              : 1;
__REG32 PD27              : 1;
__REG32 PD28              : 1;
__REG32 PD29              : 1;
__REG32 PD30              : 1;
__REG32 PD31              : 1;
} __gpdr0_bits;

/* GPIO Pin Direction Registers (GPDR1)
   GPIO Pin Bit-Wise Set Direction Registers (GSDR1)
   GPIO Pin Bit-Wise Clear Direction Registers (GCDR1)
   GPIO Bit-wise Set Rising-Edge (GSRER1) Detect-Enable Registers
   GPIO Bit-wise Clear Rising-Edge (GCRER1) Detect-Enable Registers
   GPIO Bit-wise Set Falling-Edge (GSFER1) Detect-Enable Registers
   GPIO Bit-wise Clear Falling-Edge (GCFER1) Detect-Enable Registers */
typedef struct{
__REG32 PD32              : 1;
__REG32 PD33              : 1;
__REG32 PD34              : 1;
__REG32 PD35              : 1;
__REG32 PD36              : 1;
__REG32 PD37              : 1;
__REG32 PD38              : 1;
__REG32 PD39              : 1;
__REG32 PD40              : 1;
__REG32 PD41              : 1;
__REG32 PD42              : 1;
__REG32 PD43              : 1;
__REG32 PD44              : 1;
__REG32 PD45              : 1;
__REG32 PD46              : 1;
__REG32 PD47              : 1;
__REG32 PD48              : 1;
__REG32 PD49              : 1;
__REG32 PD50              : 1;
__REG32 PD51              : 1;
__REG32 PD52              : 1;
__REG32 PD53              : 1;
__REG32 PD54              : 1;
__REG32 PD55              : 1;
__REG32 PD56              : 1;
__REG32 PD57              : 1;
__REG32 PD58              : 1;
__REG32 PD59              : 1;
__REG32 PD60              : 1;
__REG32 PD61              : 1;
__REG32 PD62              : 1;
__REG32 PD63              : 1;
} __gpdr1_bits;

/* GPIO Pin Direction Registers (GPDR2)
   GPIO Pin Bit-Wise Set Direction Registers (GSDR2)
   GPIO Pin Bit-Wise Clear Direction Registers (GCDR2)
   GPIO Bit-wise Set Rising-Edge (GSRER2) Detect-Enable Registers
   GPIO Bit-wise Clear Rising-Edge (GCRER2) Detect-Enable Registers
   GPIO Bit-wise Set Falling-Edge (GSFER2) Detect-Enable Registers
   GPIO Bit-wise Clear Falling-Edge (GCFER2) Detect-Enable Registers */
typedef struct{
__REG32 PD64              : 1;
__REG32 PD65              : 1;
__REG32 PD66              : 1;
__REG32 PD67              : 1;
__REG32 PD68              : 1;
__REG32 PD69              : 1;
__REG32 PD70              : 1;
__REG32 PD71              : 1;
__REG32 PD72              : 1;
__REG32 PD73              : 1;
__REG32 PD74              : 1;
__REG32 PD75              : 1;
__REG32 PD76              : 1;
__REG32 PD77              : 1;
__REG32 PD78              : 1;
__REG32 PD79              : 1;
__REG32 PD80              : 1;
__REG32 PD81              : 1;
__REG32 PD82              : 1;
__REG32 PD83              : 1;
__REG32 PD84              : 1;
__REG32                   :11;
} __gpdr2_bits;

/* GPIO Pin Output Set Registers (GPSR0) */
typedef struct{
__REG32 PS0               : 1;
__REG32 PS1               : 1;
__REG32 PS2               : 1;
__REG32 PS3               : 1;
__REG32 PS4               : 1;
__REG32 PS5               : 1;
__REG32 PS6               : 1;
__REG32 PS7               : 1;
__REG32 PS8               : 1;
__REG32 PS9               : 1;
__REG32 PS10              : 1;
__REG32 PS11              : 1;
__REG32 PS12              : 1;
__REG32 PS13              : 1;
__REG32 PS14              : 1;
__REG32 PS15              : 1;
__REG32 PS16              : 1;
__REG32 PS17              : 1;
__REG32 PS18              : 1;
__REG32 PS19              : 1;
__REG32 PS20              : 1;
__REG32 PS21              : 1;
__REG32 PS22              : 1;
__REG32 PS23              : 1;
__REG32 PS24              : 1;
__REG32 PS25              : 1;
__REG32 PS26              : 1;
__REG32 PS27              : 1;
__REG32 PS28              : 1;
__REG32 PS29              : 1;
__REG32 PS30              : 1;
__REG32 PS31              : 1;
} __gpsr0_bits;

/* GPIO Pin Output Set Registers (GPSR1) */
typedef struct{
__REG32 PS32              : 1;
__REG32 PS33              : 1;
__REG32 PS34              : 1;
__REG32 PS35              : 1;
__REG32 PS36              : 1;
__REG32 PS37              : 1;
__REG32 PS38              : 1;
__REG32 PS39              : 1;
__REG32 PS40              : 1;
__REG32 PS41              : 1;
__REG32 PS42              : 1;
__REG32 PS43              : 1;
__REG32 PS44              : 1;
__REG32 PS45              : 1;
__REG32 PS46              : 1;
__REG32 PS47              : 1;
__REG32 PS48              : 1;
__REG32 PS49              : 1;
__REG32 PS50              : 1;
__REG32 PS51              : 1;
__REG32 PS52              : 1;
__REG32 PS53              : 1;
__REG32 PS54              : 1;
__REG32 PS55              : 1;
__REG32 PS56              : 1;
__REG32 PS57              : 1;
__REG32 PS58              : 1;
__REG32 PS59              : 1;
__REG32 PS60              : 1;
__REG32 PS61              : 1;
__REG32 PS62              : 1;
__REG32 PS63              : 1;
} __gpsr1_bits;

/* GPIO Pin Output Set Registers (GPSR2) */
typedef struct{
__REG32 PS64              : 1;
__REG32 PS65              : 1;
__REG32 PS66              : 1;
__REG32 PS67              : 1;
__REG32 PS68              : 1;
__REG32 PS69              : 1;
__REG32 PS70              : 1;
__REG32 PS71              : 1;
__REG32 PS72              : 1;
__REG32 PS73              : 1;
__REG32 PS74              : 1;
__REG32 PS75              : 1;
__REG32 PS76              : 1;
__REG32 PS77              : 1;
__REG32 PS78              : 1;
__REG32 PS79              : 1;
__REG32 PS80              : 1;
__REG32 PS81              : 1;
__REG32 PS82              : 1;
__REG32 PS83              : 1;
__REG32 PS84              : 1;
__REG32                   :11;
} __gpsr2_bits;

/* GPIO Pin Output Clear Registers (GPCR0) */
typedef struct{
__REG32 PC0               : 1;
__REG32 PC1               : 1;
__REG32 PC2               : 1;
__REG32 PC3               : 1;
__REG32 PC4               : 1;
__REG32 PC5               : 1;
__REG32 PC6               : 1;
__REG32 PC7               : 1;
__REG32 PC8               : 1;
__REG32 PC9               : 1;
__REG32 PC10              : 1;
__REG32 PC11              : 1;
__REG32 PC12              : 1;
__REG32 PC13              : 1;
__REG32 PC14              : 1;
__REG32 PC15              : 1;
__REG32 PC16              : 1;
__REG32 PC17              : 1;
__REG32 PC18              : 1;
__REG32 PC19              : 1;
__REG32 PC20              : 1;
__REG32 PC21              : 1;
__REG32 PC22              : 1;
__REG32 PC23              : 1;
__REG32 PC24              : 1;
__REG32 PC25              : 1;
__REG32 PC26              : 1;
__REG32 PC27              : 1;
__REG32 PC28              : 1;
__REG32 PC29              : 1;
__REG32 PC30              : 1;
__REG32 PC31              : 1;
} __gpcr0_bits;

/* GPIO Pin Output Clear Registers (GPCR1) */
typedef struct{
__REG32 PC32              : 1;
__REG32 PC33              : 1;
__REG32 PC34              : 1;
__REG32 PC35              : 1;
__REG32 PC36              : 1;
__REG32 PC37              : 1;
__REG32 PC38              : 1;
__REG32 PC39              : 1;
__REG32 PC40              : 1;
__REG32 PC41              : 1;
__REG32 PC42              : 1;
__REG32 PC43              : 1;
__REG32 PC44              : 1;
__REG32 PC45              : 1;
__REG32 PC46              : 1;
__REG32 PC47              : 1;
__REG32 PC48              : 1;
__REG32 PC49              : 1;
__REG32 PC50              : 1;
__REG32 PC51              : 1;
__REG32 PC52              : 1;
__REG32 PC53              : 1;
__REG32 PC54              : 1;
__REG32 PC55              : 1;
__REG32 PC56              : 1;
__REG32 PC57              : 1;
__REG32 PC58              : 1;
__REG32 PC59              : 1;
__REG32 PC60              : 1;
__REG32 PC61              : 1;
__REG32 PC62              : 1;
__REG32 PC63              : 1;
} __gpcr1_bits;

/* GPIO Pin Output Clear Registers (GPCR2) */
typedef struct{
__REG32 PC64              : 1;
__REG32 PC65              : 1;
__REG32 PC66              : 1;
__REG32 PC67              : 1;
__REG32 PC68              : 1;
__REG32 PC69              : 1;
__REG32 PC70              : 1;
__REG32 PC71              : 1;
__REG32 PC72              : 1;
__REG32 PC73              : 1;
__REG32 PC74              : 1;
__REG32 PC75              : 1;
__REG32 PC76              : 1;
__REG32 PC77              : 1;
__REG32 PC78              : 1;
__REG32 PC79              : 1;
__REG32 PC80              : 1;
__REG32 PC81              : 1;
__REG32 PC82              : 1;
__REG32 PC83              : 1;
__REG32 PC84              : 1;
__REG32                   :11;
} __gpcr2_bits;

/* GPIO Rising-Edge Detect-Enable Registers (GRER0) */
typedef struct{
__REG32 RE0               : 1;
__REG32 RE1               : 1;
__REG32 RE2               : 1;
__REG32 RE3               : 1;
__REG32 RE4               : 1;
__REG32 RE5               : 1;
__REG32 RE6               : 1;
__REG32 RE7               : 1;
__REG32 RE8               : 1;
__REG32 RE9               : 1;
__REG32 RE10              : 1;
__REG32 RE11              : 1;
__REG32 RE12              : 1;
__REG32 RE13              : 1;
__REG32 RE14              : 1;
__REG32 RE15              : 1;
__REG32 RE16              : 1;
__REG32 RE17              : 1;
__REG32 RE18              : 1;
__REG32 RE19              : 1;
__REG32 RE20              : 1;
__REG32 RE21              : 1;
__REG32 RE22              : 1;
__REG32 RE23              : 1;
__REG32 RE24              : 1;
__REG32 RE25              : 1;
__REG32 RE26              : 1;
__REG32 RE27              : 1;
__REG32 RE28              : 1;
__REG32 RE29              : 1;
__REG32 RE30              : 1;
__REG32 RE31              : 1;
} __grer0_bits;

/* GPIO Rising-Edge Detect-Enable Registers (GRER1) */
typedef struct{
__REG32 RE32              : 1;
__REG32 RE33              : 1;
__REG32 RE34              : 1;
__REG32 RE35              : 1;
__REG32 RE36              : 1;
__REG32 RE37              : 1;
__REG32 RE38              : 1;
__REG32 RE39              : 1;
__REG32 RE40              : 1;
__REG32 RE41              : 1;
__REG32 RE42              : 1;
__REG32 RE43              : 1;
__REG32 RE44              : 1;
__REG32 RE45              : 1;
__REG32 RE46              : 1;
__REG32 RE47              : 1;
__REG32 RE48              : 1;
__REG32 RE49              : 1;
__REG32 RE50              : 1;
__REG32 RE51              : 1;
__REG32 RE52              : 1;
__REG32 RE53              : 1;
__REG32 RE54              : 1;
__REG32 RE55              : 1;
__REG32 RE56              : 1;
__REG32 RE57              : 1;
__REG32 RE58              : 1;
__REG32 RE59              : 1;
__REG32 RE60              : 1;
__REG32 RE61              : 1;
__REG32 RE62              : 1;
__REG32 RE63              : 1;
} __grer1_bits;

/* GPIO Rising-Edge Detect-Enable Registers (GRER2) */
typedef struct{
__REG32 RE64              : 1;
__REG32 RE65              : 1;
__REG32 RE66              : 1;
__REG32 RE67              : 1;
__REG32 RE68              : 1;
__REG32 RE69              : 1;
__REG32 RE70              : 1;
__REG32 RE71              : 1;
__REG32 RE72              : 1;
__REG32 RE73              : 1;
__REG32 RE74              : 1;
__REG32 RE75              : 1;
__REG32 RE76              : 1;
__REG32 RE77              : 1;
__REG32 RE78              : 1;
__REG32 RE79              : 1;
__REG32 RE80              : 1;
__REG32 RE81              : 1;
__REG32 RE82              : 1;
__REG32 RE83              : 1;
__REG32 RE84              : 1;
__REG32                   :11;
} __grer2_bits;

/* GPIO Falling-Edge Detect-Enable Registers (GFER0) */
typedef struct{
__REG32 FE0               : 1;
__REG32 FE1               : 1;
__REG32 FE2               : 1;
__REG32 FE3               : 1;
__REG32 FE4               : 1;
__REG32 FE5               : 1;
__REG32 FE6               : 1;
__REG32 FE7               : 1;
__REG32 FE8               : 1;
__REG32 FE9               : 1;
__REG32 FE10              : 1;
__REG32 FE11              : 1;
__REG32 FE12              : 1;
__REG32 FE13              : 1;
__REG32 FE14              : 1;
__REG32 FE15              : 1;
__REG32 FE16              : 1;
__REG32 FE17              : 1;
__REG32 FE18              : 1;
__REG32 FE19              : 1;
__REG32 FE20              : 1;
__REG32 FE21              : 1;
__REG32 FE22              : 1;
__REG32 FE23              : 1;
__REG32 FE24              : 1;
__REG32 FE25              : 1;
__REG32 FE26              : 1;
__REG32 FE27              : 1;
__REG32 FE28              : 1;
__REG32 FE29              : 1;
__REG32 FE30              : 1;
__REG32 FE31              : 1;
} __gfer0_bits;

/* GPIO Falling-Edge Detect-Enable Registers (GFER1) */
typedef struct{
__REG32 FE32              : 1;
__REG32 FE33              : 1;
__REG32 FE34              : 1;
__REG32 FE35              : 1;
__REG32 FE36              : 1;
__REG32 FE37              : 1;
__REG32 FE38              : 1;
__REG32 FE39              : 1;
__REG32 FE40              : 1;
__REG32 FE41              : 1;
__REG32 FE42              : 1;
__REG32 FE43              : 1;
__REG32 FE44              : 1;
__REG32 FE45              : 1;
__REG32 FE46              : 1;
__REG32 FE47              : 1;
__REG32 FE48              : 1;
__REG32 FE49              : 1;
__REG32 FE50              : 1;
__REG32 FE51              : 1;
__REG32 FE52              : 1;
__REG32 FE53              : 1;
__REG32 FE54              : 1;
__REG32 FE55              : 1;
__REG32 FE56              : 1;
__REG32 FE57              : 1;
__REG32 FE58              : 1;
__REG32 FE59              : 1;
__REG32 FE60              : 1;
__REG32 FE61              : 1;
__REG32 FE62              : 1;
__REG32 FE63              : 1;
} __gfer1_bits;

/* GPIO Falling-Edge Detect-Enable Registers (GFER2) */
typedef struct{
__REG32 FE64              : 1;
__REG32 FE65              : 1;
__REG32 FE66              : 1;
__REG32 FE67              : 1;
__REG32 FE68              : 1;
__REG32 FE69              : 1;
__REG32 FE70              : 1;
__REG32 FE71              : 1;
__REG32 FE72              : 1;
__REG32 FE73              : 1;
__REG32 FE74              : 1;
__REG32 FE75              : 1;
__REG32 FE76              : 1;
__REG32 FE77              : 1;
__REG32 FE78              : 1;
__REG32 FE79              : 1;
__REG32 FE80              : 1;
__REG32 FE81              : 1;
__REG32 FE82              : 1;
__REG32 FE83              : 1;
__REG32 FE84              : 1;
__REG32                   :11;
} __gfer2_bits;

/* GPIO Edge Detect Status Register (GEDR0) */
typedef struct{
__REG32 ED0               : 1;
__REG32 ED1               : 1;
__REG32 ED2               : 1;
__REG32 ED3               : 1;
__REG32 ED4               : 1;
__REG32 ED5               : 1;
__REG32 ED6               : 1;
__REG32 ED7               : 1;
__REG32 ED8               : 1;
__REG32 ED9               : 1;
__REG32 ED10              : 1;
__REG32 ED11              : 1;
__REG32 ED12              : 1;
__REG32 ED13              : 1;
__REG32 ED14              : 1;
__REG32 ED15              : 1;
__REG32 ED16              : 1;
__REG32 ED17              : 1;
__REG32 ED18              : 1;
__REG32 ED19              : 1;
__REG32 ED20              : 1;
__REG32 ED21              : 1;
__REG32 ED22              : 1;
__REG32 ED23              : 1;
__REG32 ED24              : 1;
__REG32 ED25              : 1;
__REG32 ED26              : 1;
__REG32 ED27              : 1;
__REG32 ED28              : 1;
__REG32 ED29              : 1;
__REG32 ED30              : 1;
__REG32 ED31              : 1;
} __gedr0_bits;

/* GPIO Edge Detect Status Register (GEDR1) */
typedef struct{
__REG32 ED32              : 1;
__REG32 ED33              : 1;
__REG32 ED34              : 1;
__REG32 ED35              : 1;
__REG32 ED36              : 1;
__REG32 ED37              : 1;
__REG32 ED38              : 1;
__REG32 ED39              : 1;
__REG32 ED40              : 1;
__REG32 ED41              : 1;
__REG32 ED42              : 1;
__REG32 ED43              : 1;
__REG32 ED44              : 1;
__REG32 ED45              : 1;
__REG32 ED46              : 1;
__REG32 ED47              : 1;
__REG32 ED48              : 1;
__REG32 ED49              : 1;
__REG32 ED50              : 1;
__REG32 ED51              : 1;
__REG32 ED52              : 1;
__REG32 ED53              : 1;
__REG32 ED54              : 1;
__REG32 ED55              : 1;
__REG32 ED56              : 1;
__REG32 ED57              : 1;
__REG32 ED58              : 1;
__REG32 ED59              : 1;
__REG32 ED60              : 1;
__REG32 ED61              : 1;
__REG32 ED62              : 1;
__REG32 ED63              : 1;
} __gedr1_bits;

/* GPIO Edge Detect Status Register (GEDR2) */
typedef struct{
__REG32 ED64              : 1;
__REG32 ED65              : 1;
__REG32 ED66              : 1;
__REG32 ED67              : 1;
__REG32 ED68              : 1;
__REG32 ED69              : 1;
__REG32 ED70              : 1;
__REG32 ED71              : 1;
__REG32 ED72              : 1;
__REG32 ED73              : 1;
__REG32 ED74              : 1;
__REG32 ED75              : 1;
__REG32 ED76              : 1;
__REG32 ED77              : 1;
__REG32 ED78              : 1;
__REG32 ED79              : 1;
__REG32 ED80              : 1;
__REG32 ED81              : 1;
__REG32 ED82              : 1;
__REG32 ED83              : 1;
__REG32 ED84              : 1;
__REG32                   :11;
} __gedr2_bits;

/* GPIO Alternate Function Register (GAFR0_L) */
typedef struct{
__REG32 AF0               : 2;
__REG32 AF1               : 2;
__REG32 AF2               : 2;
__REG32 AF3               : 2;
__REG32 AF4               : 2;
__REG32 AF5               : 2;
__REG32 AF6               : 2;
__REG32 AF7               : 2;
__REG32 AF8               : 2;
__REG32 AF9               : 2;
__REG32 AF10              : 2;
__REG32 AF11              : 2;
__REG32 AF12              : 2;
__REG32 AF13              : 2;
__REG32 AF14              : 2;
__REG32 AF15              : 2;
} __gafr0_l_bits;

/* GPIO Alternate Function Register (GAFR0_U) */
typedef struct{
__REG32 AF16              : 2;
__REG32 AF17              : 2;
__REG32 AF18              : 2;
__REG32 AF19              : 2;
__REG32 AF20              : 2;
__REG32 AF21              : 2;
__REG32 AF22              : 2;
__REG32 AF23              : 2;
__REG32 AF24              : 2;
__REG32 AF25              : 2;
__REG32 AF26              : 2;
__REG32 AF27              : 2;
__REG32 AF28              : 2;
__REG32 AF29              : 2;
__REG32 AF30              : 2;
__REG32 AF31              : 2;
} __gafr0_u_bits;

/* GPIO Alternate Function Register (GAFR1_L) */
typedef struct{
__REG32 AF32              : 2;
__REG32 AF33              : 2;
__REG32 AF34              : 2;
__REG32 AF35              : 2;
__REG32 AF36              : 2;
__REG32 AF37              : 2;
__REG32 AF38              : 2;
__REG32 AF39              : 2;
__REG32 AF40              : 2;
__REG32 AF41              : 2;
__REG32 AF42              : 2;
__REG32 AF43              : 2;
__REG32 AF44              : 2;
__REG32 AF45              : 2;
__REG32 AF46              : 2;
__REG32 AF47              : 2;
} __gafr1_l_bits;

/* GPIO Alternate Function Register (GAFR1_U) */
typedef struct{
__REG32 AF48              : 2;
__REG32 AF49              : 2;
__REG32 AF50              : 2;
__REG32 AF51              : 2;
__REG32 AF52              : 2;
__REG32 AF53              : 2;
__REG32 AF54              : 2;
__REG32 AF55              : 2;
__REG32 AF56              : 2;
__REG32 AF57              : 2;
__REG32 AF58              : 2;
__REG32 AF59              : 2;
__REG32 AF60              : 2;
__REG32 AF61              : 2;
__REG32 AF62              : 2;
__REG32 AF63              : 2;
} __gafr1_u_bits;

/* GPIO Alternate Function Register (GAFR2_L) */
typedef struct{
__REG32 AF64              : 2;
__REG32 AF65              : 2;
__REG32 AF66              : 2;
__REG32 AF67              : 2;
__REG32 AF68              : 2;
__REG32 AF69              : 2;
__REG32 AF70              : 2;
__REG32 AF71              : 2;
__REG32 AF72              : 2;
__REG32 AF73              : 2;
__REG32 AF74              : 2;
__REG32 AF75              : 2;
__REG32 AF76              : 2;
__REG32 AF77              : 2;
__REG32 AF78              : 2;
__REG32 AF79              : 2;
} __gafr2_l_bits;

/* GPIO Alternate Function Register (GAFR2_U) */
typedef struct{
__REG32 AF80              : 2;
__REG32 AF81              : 2;
__REG32 AF82              : 2;
__REG32 AF83              : 2;
__REG32 AF84              : 2;
__REG32                   :22;
} __gafr2_u_bits;

/* Interrupt Controller Pending Register (ICPR)
   Interrupt Controller IRQ Pending Registers (ICIP)
   Interrupt Controller FIQ Pending Registers (ICFP)
   Interrupt Controller Mask Registers (ICMR)
   Interrupt Controller Level Registers (ICLR) */
typedef struct{
__REG32                   : 7;
__REG32 UART              : 1;
__REG32 GPIO_0            : 1;
__REG32 GPIO_1            : 1;
__REG32 GPIO_X            : 1;
__REG32 USBC              : 1;
__REG32 PMU               : 1;
__REG32 I2S               : 1;
__REG32 AC97              : 1;
__REG32                   : 1;
__REG32 NSSP              : 1;
__REG32 LCD               : 1;
__REG32 I2C               : 1;
__REG32 ICP               : 1;
__REG32 STUART            : 1;
__REG32 BTUART            : 1;
__REG32 FFUART            : 1;
__REG32 MMC               : 1;
__REG32 SSP               : 1;
__REG32 DMAC              : 1;
__REG32 OST_0             : 1;
__REG32 OST_1             : 1;
__REG32 OST_2             : 1;
__REG32 OST_3             : 1;
__REG32 RTC_HZ            : 1;
__REG32 RTC_AL            : 1;
} __icpr_bits;

/* Interrupt Controller Control Register (ICCR) */
typedef struct{
__REG32 DIM               : 1;
__REG32                   :31;
} __iccr_bits;

/* RTC Trim Register (RTTR) */
typedef struct{
__REG32 CK_DIV            :16;
__REG32 DEL               :10;
__REG32                   : 5;
__REG32 LCK               : 1;
} __rttr_bits;

/* RTC Status Register (RTSR) */
typedef struct{
__REG32 AL                : 1;
__REG32 HZ                : 1;
__REG32 ALE               : 1;
__REG32 HZE               : 1;
__REG32                   :28;
} __rtsr_bits;

/* OS Timer Watchdog Match Enable Register (OWER) */
typedef struct{
__REG32 WME               : 1;
__REG32                   :31;
} __ower_bits;

/* OS Timer Interrupt Enable Register (OIER) */
typedef struct{
__REG32 E0                : 1;
__REG32 E1                : 1;
__REG32 E2                : 1;
__REG32 E3                : 1;
__REG32                   :28;
} __oier_bits;

/* OS Timer Status Register (OSSR) */
typedef struct{
__REG32 M0                : 1;
__REG32 M1                : 1;
__REG32 M2                : 1;
__REG32 M3                : 1;
__REG32                   :28;
} __ossr_bits;

/* PWM Control Registers (PWM_CTRLx) */
typedef struct {
  __REG32 PRESCALE         : 6;
  __REG32 SD               : 1;
  __REG32                  :25;
} __pwm_ctrl_bits;

/* PWM Duty Cycle Registers (PWM_DUTYx) */
typedef struct {
  __REG32 DCYCLE           :10;
  __REG32 FD               : 1;
  __REG32                  :21;
} __pwm_duty_bits;

/* PWM Period Control Registers (PWM_PERVALx) */
typedef struct {
  __REG32 PV               :10;
  __REG32                  :22;
} __pwm_perval_bits;

/* DMA Request to Channel Map Register (DRCMRx) */
typedef struct{
__REG32 CHLNUM            : 4;
__REG32                   : 3;
__REG32 MAPVLD            : 1;
__REG32                   :24;
} __drcmr_bits;

/* DMA Descriptor Address Registers (DDADRx) */
typedef struct{
__REG32 STOP              : 1;
__REG32                   : 3;
__REG32 DA                :28;
} __ddadr_bits;

/* DMA Command Registers (DCMDx) */
typedef struct{
__REG32 LEN               :13;
__REG32                   : 1;
__REG32 WIDTH             : 2;
__REG32 SIZE              : 2;
__REG32 ENDIAN            : 1;
__REG32                   : 2;
__REG32 ENDIRQEN          : 1;
__REG32 STARTIRQEN        : 1;
__REG32                   : 5;
__REG32 FLOWTRG           : 1;
__REG32 FLOWSRC           : 1;
__REG32 INCTRGADDR        : 1;
__REG32 INCSRCADDR        : 1;
} __dcmd_bits;

/* DMA Channel Control/Status Registers (DCSRx) */
typedef struct{
__REG32 BUSERRINTR        : 1;
__REG32 STARTINTR         : 1;
__REG32 ENDINTR           : 1;
__REG32 STOPINTR          : 1;
__REG32                   : 4;
__REG32 REQPEND           : 1;
__REG32                   :20;
__REG32 STOPIRQEN         : 1;
__REG32 NODESCFETCH       : 1;
__REG32 RUN               : 1;
} __dcsr_bits;

/* DMA Interrupt Register (DINT) */
typedef struct{
__REG32 CHLINTR0          : 1;
__REG32 CHLINTR1          : 1;
__REG32 CHLINTR2          : 1;
__REG32 CHLINTR3          : 1;
__REG32 CHLINTR4          : 1;
__REG32 CHLINTR5          : 1;
__REG32 CHLINTR6          : 1;
__REG32 CHLINTR7          : 1;
__REG32 CHLINTR8          : 1;
__REG32 CHLINTR9          : 1;
__REG32 CHLINTR10         : 1;
__REG32 CHLINTR11         : 1;
__REG32 CHLINTR12         : 1;
__REG32 CHLINTR13         : 1;
__REG32 CHLINTR14         : 1;
__REG32 CHLINTR15         : 1;
__REG32                   :16;
} __dint_bits;

/* SDRAM Configuration Register (MDCNFG) */
typedef struct{
__REG32 DE0               : 1;
__REG32 DE1               : 1;
__REG32 DWID0             : 1;
__REG32 DCAC0             : 2;
__REG32 DRAC0             : 2;
__REG32 DNB0              : 1;
__REG32 DTC0              : 2;
__REG32 DADDR0            : 1;
__REG32 DLATCH0           : 1;
__REG32 DSA1110_0         : 1;
__REG32                   : 3;
__REG32 DE2               : 1;
__REG32 DE3               : 1;
__REG32 DWID2             : 1;
__REG32 DCAC2             : 2;
__REG32 DRAC2             : 2;
__REG32 DNB2              : 1;
__REG32 DTC2              : 2;
__REG32 DADDR2            : 1;
__REG32 DLATCH2           : 1;
__REG32 DSA1110_2         : 1;
__REG32                   : 3;
} __mdcnfg_bits;

/* SDRAM Mode Register Set Configuration Register (MDMRS) */
typedef struct{
__REG32 MDBL0             : 3;
__REG32 MDADD0            : 1;
__REG32 MDCL0             : 3;
__REG32 MDMRS0            : 8;
__REG32                   : 1;
__REG32 MDBL2             : 3;
__REG32 MDADD2            : 1;
__REG32 MDCL2             : 3;
__REG32 MDMRS2            : 8;
__REG32                   : 1;
} __mdmrs_bits;

/* Special Low-Power SDRAM Mode Register Set Configuration Register (MDMRSLP) */
typedef struct{
__REG32 MDMRSLP0          :15;
__REG32 MDLPEN0           : 1;
__REG32 MDMRSLP2          :15;
__REG32 MDLPEN2           : 1;
} __mdmrslp_bits;

/* SDRAM Memory Device Refresh Register (MDREFR) */
typedef struct{
__REG32 DRI               :12;
__REG32 E0PIN             : 1;
__REG32 K0RUN             : 1;
__REG32 K0DB2             : 1;
__REG32 E1PIN             : 1;
__REG32 K1RUN             : 1;
__REG32 K1DB2             : 1;
__REG32 K2RUN             : 1;
__REG32 K2DB2             : 1;
__REG32 APD               : 1;
__REG32                   : 1;
__REG32 SLFRSH            : 1;
__REG32 K0FREE            : 1;
__REG32 K1FREE            : 1;
__REG32 K2FREE            : 1;
__REG32                   : 6;
} __mdrefr_bits;

/* Synchronous Static Memory Control Register (SXCNFG) */
typedef struct{
__REG32 SXEN0             : 2;
__REG32 SXCL0             : 3;
__REG32 SXRL0             : 3;
__REG32 SXRA0             : 2;
__REG32 SXCA0             : 2;
__REG32 SXTP0             : 2;
__REG32 SXLATCH0          : 1;
__REG32                   : 1;
__REG32 SXEN2             : 2;
__REG32 SXCL2             : 3;
__REG32 SXRL2             : 3;
__REG32 SXRA2             : 2;
__REG32 SXCA2             : 2;
__REG32 SXTP2             : 2;
__REG32 SXLATCH2          : 1;
__REG32                   : 1;
} __sxcnfg_bits;

/* Synchronous Static Memory Mode Register Set Configuration Register (SXMRS) */
typedef struct{
__REG32 SXMRS0            :15;
__REG32                   : 1;
__REG32 SXMRS2            :15;
__REG32                   : 1;
} __sxmrs_bits;

/* Static Memory SA-1111 Compatibility Configuration Register (SA1111CR) */
typedef struct{
__REG32 SA1110_0          : 1;
__REG32 SA1110_1          : 1;
__REG32 SA1110_2          : 1;
__REG32 SA1110_3          : 1;
__REG32 SA1110_4          : 1;
__REG32 SA1110_5          : 1;
__REG32                   :26;
} __sa1111cr_bits;

/* Static Memory Control Registers (MSC0,1,2) */
typedef struct{
__REG32 RT0_2_4           : 3;
__REG32 RBW0_2_4          : 1;
__REG32 RDF0_2_4          : 4;
__REG32 RDN0_2_4          : 4;
__REG32 RRR0_2_4          : 3;
__REG32 RBUFF0_2_4        : 1;
__REG32 RT1_3_5           : 3;
__REG32 RBW1_3_5          : 1;
__REG32 RDF1_3_5          : 4;
__REG32 RDN1_3_5          : 4;
__REG32 RRR1_3_5          : 3;
__REG32 RBUFF1_3_5        : 1;
} __msc_bits;

/* Expansion Memory Timing Configuration Registers (MCMEMx, MCATTx, MCIO0x) */
typedef struct{
__REG32 SET               : 7;
__REG32 ASST              : 5;
__REG32                   : 2;
__REG32 HOLD              : 6;
__REG32                   :12;
} __mcmem_bits;

/* Expansion Memory Configuration Register (MECR) */
typedef struct{
__REG32 NOS               : 1;
__REG32 CIT               : 1;
__REG32                   :30;
} __mecr_bits;

/* Boot Time Default Configuration Register (BOOT_DEF) */
typedef struct{
__REG32 BOOT_SEL          : 3;
__REG32 PKG_TYPE          : 1;
__REG32                   :28;
} __boot_def_bits;

/* LCD Controller Control Register 0 (LCCR0) */
typedef struct{
__REG32 ENB               : 1;
__REG32 CMS               : 1;
__REG32 SDS               : 1;
__REG32 LDM               : 1;
__REG32 SFM               : 1;
__REG32 IUM               : 1;
__REG32 EFM               : 1;
__REG32 PAS               : 1;
__REG32                   : 1;
__REG32 DPD               : 1;
__REG32 DIS               : 1;
__REG32 QDM               : 1;
__REG32 PDD               : 8;
__REG32 BM                : 1;
__REG32 OUM               : 1;
__REG32                   :10;
} __lccr0_bits;

/* LCD Controller Control Register 1 (LCCR1) */
typedef struct{
__REG32 PPL               :10;
__REG32 HSW               : 6;
__REG32 ELW               : 8;
__REG32 BLW               : 8;
} __lccr1_bits;

/* LCD Controller Control Register 2 (LCCR2) */
typedef struct{
__REG32 LPP               :10;
__REG32 VSW               : 6;
__REG32 EFW               : 8;
__REG32 BFW               : 8;
} __lccr2_bits;

/* LCD Controller Control Register 3 (LCCR3) */
typedef struct{
__REG32 PCD               : 8;
__REG32 ACB               : 8;
__REG32 API               : 4;
__REG32 VSP               : 1;
__REG32 HSP               : 1;
__REG32 PCP               : 1;
__REG32 OEP               : 1;
__REG32 BPP               : 3;
__REG32 DPC               : 1;
__REG32                   : 4;
} __lccr3_bits;

/* LCD DMA Command Register (LDCMDx) */
typedef struct{
__REG32 LENGTH            :21;
__REG32 EOFINT            : 1;
__REG32 SOFINT            : 1;
__REG32                   : 3;
__REG32 PAL               : 1;
__REG32                   : 5;
} __ldcmd_bits;

/* LCD DMA Frame Branch Registers (FBRx) */
typedef struct{
__REG32 BRA               : 1;
__REG32 BINT              : 1;
__REG32                   : 2;
__REG32 FBA               :28;
} __fbr_bits;

/* LCD Controller Status Register (LCSR) */
typedef struct{
__REG32 LDD               : 1;
__REG32 SOF               : 1;
__REG32 BER               : 1;
__REG32 ABC               : 1;
__REG32 IUL               : 1;
__REG32 IUU               : 1;
__REG32 OU                : 1;
__REG32 QD                : 1;
__REG32 _EOF              : 1;
__REG32 BS                : 1;
__REG32 SINT              : 1;
__REG32                   :21;
} __lcsr_bits;

/* LCD TMED RGB Seed Register (TRGBR) */
typedef struct{
__REG32 TRS               : 8;
__REG32 TGS               : 8;
__REG32 TBS               : 8;
__REG32                   : 8;
} __trgbr_bits;

/* LCD TMED Control Register (TCR) */
typedef struct{
__REG32 COAM              : 1;
__REG32 FNAM              : 1;
__REG32 COAE              : 1;
__REG32 FNAME             : 1;
__REG32 TVBS              : 4;
__REG32 THBS              : 4;
__REG32                   : 2;
__REG32 TED               : 1;
__REG32                   :17;
} __tcr_bits;

/* I2C Bus Monitor Register (IBMR) */
typedef struct {
  __REG32 SDA              : 1;
  __REG32 SCL              : 1;
  __REG32                  :30;
} __ibmr_bits;

/* I2C Data Buffer Register (IDBR) */
typedef struct {
  __REG32 IDB              : 8;
  __REG32                  :24;
} __idbr_bits;

/* I2C Control Register (ICR) */
typedef struct {
  __REG32 START            : 1;
  __REG32 STOP             : 1;
  __REG32 ACKNAK           : 1;
  __REG32 TB               : 1;
  __REG32 MA               : 1;
  __REG32 SCLE             : 1;
  __REG32 IUE              : 1;
  __REG32 GCD              : 1;
  __REG32 ITEIE            : 1;
  __REG32 IRFIE            : 1;
  __REG32 BEIE             : 1;
  __REG32 SSDIE            : 1;
  __REG32 ALDIE            : 1;
  __REG32 SADIE            : 1;
  __REG32 UR               : 1;
  __REG32 FM               : 1;
  __REG32                  :16;
} __icr_bits;

/* I2C Status Register (ISR) */
typedef struct {
  __REG32 RWM              : 1;
  __REG32 ACKNAK           : 1;
  __REG32 UB               : 1;
  __REG32 IBB              : 1;
  __REG32 SSD              : 1;
  __REG32 ALD              : 1;
  __REG32 ITE              : 1;
  __REG32 IRF              : 1;
  __REG32 GCAD             : 1;
  __REG32 SAD              : 1;
  __REG32 BED              : 1;
  __REG32                  :21;
} __isr_bits;

/* I2C Slave Address Register (ISAR) */
typedef struct {
  __REG32 ISA              : 7;
  __REG32                  :25;
} __isar_bits;

/* Receive Buffer Register (RBR) */
/* Transmit Holding Register (THR) */
/* Divisor Latch Register  Low (DLL) */
typedef union {
  /*FFRBR*/
  /*BTRBR*/
  /*STRBR*/
  /*HWRBR*/
  struct {
    __REG32 RBR              : 8;
    __REG32                  :24;
  } ;
  /*FFTHR*/
  /*BTTHR*/
  /*STTHR*/
  /*HWTHR*/
  struct {
    __REG32 THR              : 8;
    __REG32                  :24;
  } ;
  /*FFDLL*/
  /*BTDLL*/
  /*STDLL*/
  /*HWDLL*/
  struct {
    __REG32 DLL              : 8;
    __REG32                  :24;
  } ;
} __uartrbr_bits;

/* Interrupt Enable Register (IER) */
/* Divisor Latch Register  High (DLH) */
typedef union {
  /*FFIER*/
  /*BTIER*/
  /*STIER*/
  /*HWIER*/
  struct {
    __REG32 RAVIE            : 1;
    __REG32 TIE              : 1;
    __REG32 RLSE             : 1;
    __REG32 MIE              : 1;
    __REG32 RTOIE            : 1;
    __REG32 NRZE             : 1;
    __REG32 UUE              : 1;
    __REG32 DMAE             : 1;
    __REG32                  :24;
  } ;
  /*FFDLH*/
  /*BTDLH*/
  /*STDLH*/
  /*HWDLH*/
  struct {
    __REG32 DLH              : 8;
    __REG32                  :24;
  } ;
} __uartier_bits;

/* Interrupt Identification Register (IIR) */
/* FIFO Control Register (FCR) */
typedef union {
  /*FFIIR*/
  /*BTIIR*/
  /*STIIR*/
  struct {
    __REG32 IP               : 1;
    __REG32 IID              : 3;
    __REG32                  : 2;
    __REG32 FIFOES           : 2;
    __REG32                  :24;
  } ;
  /*FFFCR*/
  /*BTFCR*/
  /*STFCR*/
  struct {
    __REG32 TRFIFOE          : 1;
    __REG32 RESETRF          : 1;
    __REG32 RESETTF          : 1;
    __REG32                  : 3;
    __REG32 ITL              : 2;
    __REG32                  :24;
  } ;
} __uartiir_bits;

/* Interrupt Identification Register (IIR) (Hardware UART) */
/* FIFO Control Register (FCR) (Hardware UART) */
typedef union {
  /*HWIIR*/
  struct {
    __REG32 nIP              : 1;
    __REG32 IID              : 2;
    __REG32 TOD              : 1;
    __REG32 ABL              : 1;
    __REG32                  : 1;
    __REG32 FIFOES           : 2;
    __REG32                  :24;
  } ;
  /*HWFCR*/
  struct {
    __REG32 TRFIFOE          : 1;
    __REG32 RESETRF          : 1;
    __REG32 RESETTF          : 1;
    __REG32 TIL              : 1;
    __REG32                  : 2;
    __REG32 ITL              : 2;
    __REG32                  :24;
  } ;
} __hwiir_bits;

/* Line Control Register (LCR) */
typedef struct {
  __REG32 WLS              : 2;
  __REG32 STB              : 1;
  __REG32 PEN              : 1;
  __REG32 EPS              : 1;
  __REG32 STKYP            : 1;
  __REG32 SB               : 1;
  __REG32 DLAB             : 1;
  __REG32                  :24;
} __uartlcr_bits;

/*  Modem Control Register (MCR) */
typedef struct {
  __REG32 DTR              : 1;
  __REG32 RTS              : 1;
  __REG32 OUT1             : 1;
  __REG32 OUT2             : 1;
  __REG32 LOOP             : 1;
  __REG32                  :27;
} __uartmcr_bits;

/*  Modem Control Register (BTMCR) */
typedef struct {
  __REG32                  : 1;
  __REG32 RTS              : 1;
  __REG32                  : 1;
  __REG32 OUT2             : 1;
  __REG32 LOOP             : 1;
  __REG32                  :27;
} __btmcr_bits;

/*  Modem Control Register (HWMCR) */
typedef struct {
  __REG32                  : 1;
  __REG32 RTS              : 1;
  __REG32                  : 1;
  __REG32 OUT2             : 1;
  __REG32 LOOP             : 1;
  __REG32 AFE              : 1;
  __REG32                  :26;
} __hwmcr_bits;

/*  Modem Control Register (STMCR) */
typedef struct {
  __REG32                  : 3;
  __REG32 OUT2             : 1;
  __REG32 LOOP             : 1;
  __REG32                  :27;
} __stmcr_bits;

/* Line Status Register (LSR) */
typedef struct {
  __REG32 DR               : 1;
  __REG32 OE               : 1;
  __REG32 PE               : 1;
  __REG32 FE               : 1;
  __REG32 BI               : 1;
  __REG32 TDRQ             : 1;
  __REG32 TEMT             : 1;
  __REG32 FIFOE            : 1;
  __REG32                  :24;
} __uartlsr_bits;

/* Modem Status Register (MSR) */
typedef struct {
  __REG32 DCTS             : 1;
  __REG32 DDSR             : 1;
  __REG32 TERI             : 1;
  __REG32 DDCD             : 1;
  __REG32 CTS              : 1;
  __REG32 DSR              : 1;
  __REG32 RI               : 1;
  __REG32 DCD              : 1;
  __REG32                  :24;
} __uartmsr_bits;

/* Modem Status Register (BTMSR) */
typedef struct {
  __REG32 DCTS             : 1;
  __REG32                  : 3;
  __REG32 CTS              : 1;
  __REG32                  :27;
} __btmsr_bits;

/* Scratchpad Register (SPR) */
typedef struct {
  __REG32 SP               : 8;
  __REG32                  :24;
} __uartspr_bits;

/* Infrared Selection Register */
typedef struct {
  __REG32 XMITIR           : 1;
  __REG32 RCVEIR           : 1;
  __REG32 XMODE            : 1;
  __REG32 TXPL             : 1;
  __REG32 RXPL             : 1;
  __REG32                  :27;
} __uartisr_bits;

/* Receive FIFO Occupancy Register (FOR) */
typedef struct {
  __REG32 BYTE_COUNT       : 7;
  __REG32                  :25;
} __uartfor_bits;

/* Auto-Baud Control Register (ABR) */
typedef struct {
  __REG32 ABE              : 1;
  __REG32 ABLIE            : 1;
  __REG32 ABUP             : 1;
  __REG32 ABT              : 1;
  __REG32                  :28;
} __uartabr_bits;

/* Auto-Baud Count Register (ACR) */
typedef struct {
  __REG32 COUNT_VALUE      :16;
  __REG32                  :16;
} __uartacr_bits;

/* FICP Control Register 0 (ICCR0) */
typedef struct {
  __REG32 ITR              : 1;
  __REG32 LBM              : 1;
  __REG32 TUS              : 1;
  __REG32 TXE              : 1;
  __REG32 RXE              : 1;
  __REG32 RIE              : 1;
  __REG32 TIE              : 1;
  __REG32 AME              : 1;
  __REG32                  :24;
} __iccr0_bits;

/* FICP Control Register 1 (ICCR1) */
typedef struct {
  __REG32 AMV              : 8;
  __REG32                  :24;
} __iccr1_bits;

/* FICP Control Register 2 (ICCR2) */
typedef struct {
  __REG32 TRIG             : 2;
  __REG32 TXP              : 1;
  __REG32 RXP              : 1;
  __REG32                  :28;
} __iccr2_bits;

/* FICP Data Register (ICDR) */
typedef struct {
  __REG32 DATA             : 8;
  __REG32                  :24;
} __icdr_bits;

/* FICP Status Register 0 (ICSR0) */
typedef struct {
  __REG32 EIF              : 1;
  __REG32 TUR              : 1;
  __REG32 RAB              : 1;
  __REG32 TFS              : 1;
  __REG32 RFS              : 1;
  __REG32 FRE              : 1;
  __REG32                  :26;
} __icsr0_bits;

/* FICP Status Register 1 (ICSR1) */
typedef struct {
  __REG32 RSY              : 1;
  __REG32 TBY              : 1;
  __REG32 RNE              : 1;
  __REG32 TNF              : 1;
  __REG32 _EOF             : 1;
  __REG32 CRE              : 1;
  __REG32 ROR              : 1;
  __REG32                  :25;
} __icsr1_bits;

/* UDC Control Register (UDCCR) */
typedef struct{
__REG32 UDE               : 1;
__REG32 UDA               : 1;
__REG32 RSM               : 1;
__REG32 RESIR             : 1;
__REG32 SUSIR             : 1;
__REG32 SRM               : 1;
__REG32 RSTIR             : 1;
__REG32 REM               : 1;
__REG32                   :24;
} __udccr_bits;

/* UDC Control Function Register (UDCCFR) */
typedef struct{
__REG32 MB1L              : 2;
__REG32 ACM               : 1;
__REG32 MB1H              : 4;
__REG32 AREN              : 1;
__REG32                   :24;
} __udccfr_bits;

/* UDC Endpoint 0 Control/Status Register (UDCCS0) */
typedef struct{
__REG32 OPR               : 1;
__REG32 IPR               : 1;
__REG32 FTF               : 1;
__REG32 DRWF              : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 RNE               : 1;
__REG32 SA                : 1;
__REG32                   :24;
} __udccs0_bits;

/* UDC Endpoint x Control/Status Register (UDCCS1/6/11) */
typedef struct{
__REG32 TFS               : 1;
__REG32 TPC               : 1;
__REG32 FTF               : 1;
__REG32 TUR               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32                   : 1;
__REG32 TSP               : 1;
__REG32                   :24;
} __udccs1_bits;

/* UDC Endpoint x Control/Status Register (UDCCS2/7/12) */
typedef struct{
__REG32 RFS               : 1;
__REG32 RPC               : 1;
__REG32                   : 1;
__REG32 DME               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 RNE               : 1;
__REG32 RSP               : 1;
__REG32                   :24;
} __udccs2_bits;

/* UDC Endpoint x Control/Status Register (UDCCS3/8/13) */
typedef struct{
__REG32 TFS               : 1;
__REG32 TPC               : 1;
__REG32 FTF               : 1;
__REG32 TUR               : 1;
__REG32                   : 3;
__REG32 TSP               : 1;
__REG32                   :24;
} __udccs3_bits;

/* UDC Endpoint x Control/Status Register (UDCCS4/9/14) */
typedef struct{
__REG32 RFS               : 1;
__REG32 RPC               : 1;
__REG32 ROF               : 1;
__REG32 DME               : 1;
__REG32                   : 2;
__REG32 RNE               : 1;
__REG32 RSP               : 1;
__REG32                   :24;
} __udccs4_bits;

/* UDC Endpoint x Control/Status Register (UDCCS5/10/15) */
typedef struct{
__REG32 TFS               : 1;
__REG32 TPC               : 1;
__REG32 FTF               : 1;
__REG32 TUR               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32                   : 1;
__REG32 TSP               : 1;
__REG32                   :24;
} __udccs5_bits;

/* UDC Interrupt Control Register 0 (UICR0) */
typedef struct{
__REG32 IM0               : 1;
__REG32 IM1               : 1;
__REG32 IM2               : 1;
__REG32 IM3               : 1;
__REG32 IM4               : 1;
__REG32 IM5               : 1;
__REG32 IM6               : 1;
__REG32 IM7               : 1;
__REG32                   :24;
} __uicr0_bits;

/* UDC Interrupt Control Register 1 (UICR1) */
typedef struct{
__REG32 IM8               : 1;
__REG32 IM9               : 1;
__REG32 IM10              : 1;
__REG32 IM11              : 1;
__REG32 IM12              : 1;
__REG32 IM13              : 1;
__REG32 IM14              : 1;
__REG32 IM15              : 1;
__REG32                   :24;
} __uicr1_bits;

/* UDC Status/Interrupt Register 0 (USIR0) */
typedef struct{
__REG32 IR0               : 1;
__REG32 IR1               : 1;
__REG32 IR2               : 1;
__REG32 IR3               : 1;
__REG32 IR4               : 1;
__REG32 IR5               : 1;
__REG32 IR6               : 1;
__REG32 IR7               : 1;
__REG32                   :24;
} __usir0_bits;

/* UDC Status/Interrupt Register 1 (USIR1) */
typedef struct{
__REG32 IR8               : 1;
__REG32 IR9               : 1;
__REG32 IR10              : 1;
__REG32 IR11              : 1;
__REG32 IR12              : 1;
__REG32 IR13              : 1;
__REG32 IR14              : 1;
__REG32 IR15              : 1;
__REG32                   :24;
} __usir1_bits;

/* UDC Frame Number High Register (UFNHR) */
typedef struct{
__REG32 FNMSB             : 3;
__REG32 IPE4              : 1;
__REG32 IPE9              : 1;
__REG32 IPE14             : 1;
__REG32 SIM               : 1;
__REG32 SIR               : 1;
__REG32                   :24;
} __ufnhr_bits;

/* UDC Frame Number Low Register (UFNLR) */
typedef struct{
__REG32 FNLSB             : 8;
__REG32                   :24;
} __ufnlr_bits;

/* UDC Byte Count Register x (UBCR2/4/7/9/12/14) */
typedef struct{
__REG32 BC                : 8;
__REG32                   :24;
} __ubcr2_bits;

/* UDC Endpoint 0 Data Register (UDDR0)
   UDC Endpoint x Data Register (UDDRx) */
typedef struct{
__REG32 DATA              : 8;
__REG32                   :24;
} __uddr_bits;

/* PCM-Out Control Register (POCR) */
/* PCM-In Control Register (PICR) */
/* Mic-In Control Register (MCCR) */
/* MODEM-Out Control Register (MOCR) */
/* MODEM-In Control Register (MICR) */
typedef struct {
  __REG32                   : 3;
  __REG32 FEIE              : 1;
  __REG32                   :28;
} __pocr_bits;

/* Global Control Register (GCR) */
typedef struct {
  __REG32 GIE               : 1;
  __REG32 COLD_RST          : 1;
  __REG32 WARM_RST          : 1;
  __REG32 ACLINK_OFF        : 1;
  __REG32 PRIRES_IEN        : 1;
  __REG32 SECRES_IEN        : 1;
  __REG32                   : 2;
  __REG32 PRIRDY_IEN        : 1;
  __REG32 SECRDY_IEN        : 1;
  __REG32                   : 8;
  __REG32 SDONE_IE          : 1;
  __REG32 CDONE_IE          : 1;
  __REG32                   :12;
} __gcr_bits;

/* PCM-Out Status Register (POSR) */
/* MODEM-Out Status Register (MOSR) */
/* PCM_In Status Register (PISR) */
/* Mic-In Status Register (MCSR) */
/* MODEM-In Status Register (MISR) */
typedef struct {
  __REG32                   : 4;
  __REG32 FIFOE             : 1;
  __REG32                   :27;
} __posr_bits;

/* Global Status Register */
typedef struct {
  __REG32 GSCI              : 1;
  __REG32 MIINT             : 1;
  __REG32 MOINT             : 1;
  __REG32                   : 2;
  __REG32 PIINT             : 1;
  __REG32 POINT             : 1;
  __REG32 MINT              : 1;
  __REG32 PCR               : 1;
  __REG32 SCR               : 1;
  __REG32 PRIRES            : 1;
  __REG32 SECRES            : 1;
  __REG32 BIT1SLT12         : 1;
  __REG32 BIT2SLT12         : 1;
  __REG32 BIT3SLT12         : 1;
  __REG32 RDCS              : 1;
  __REG32                   : 2;
  __REG32 SDONE             : 1;
  __REG32 CDONE             : 1;
  __REG32                   :12;
} __gsr_bits;

/* CODEC Access Register (CAR) */
typedef struct {
  __REG32 CAIP              : 1;
  __REG32                   :31;
} __car_bits;

/* PCM Data Register (PCDR) */
typedef struct {
  __REG32 PCM_LDATA         :16;
  __REG32 PCM_RDATA         :16;
} __pcdr_bits;

/* Mic-In Data Register (MCDR) */
typedef struct {
  __REG32 MIC_IN_DAT        :16;
  __REG32                   :16;
} __mcdr_bits;

/* MODEM Data Register (MODR) */
typedef struct {
  __REG32 MODEM_DAT         :16;
  __REG32                   :16;
} __modr_bits;

/* Serial Audio Controller Global Control Register (SACR0) */
typedef struct{
__REG32 ENB               : 1;
__REG32                   : 1;
__REG32 BCKD              : 1;
__REG32 RST               : 1;
__REG32 EFWR              : 1;
__REG32 STRF              : 1;
__REG32                   : 2;
__REG32 TFTH              : 4;
__REG32 RFTH              : 4;
__REG32                   :16;
} __sacr0_bits;

/* Serial Audio Controller I2S/MSB-Justified Control Register (SACR1) */
typedef struct{
__REG32 AMSL              : 1;
__REG32                   : 2;
__REG32 DREC              : 1;
__REG32 DRPL              : 1;
__REG32 ENLBF             : 1;
__REG32                   :26;
} __sacr1_bits;

/* Serial Audio Controller I2S/MSB-Justified Status Register (SASR0) */
typedef struct{
__REG32 TNF               : 1;
__REG32 RNE               : 1;
__REG32 BSY               : 1;
__REG32 TFS               : 1;
__REG32 RFS               : 1;
__REG32 TUR               : 1;
__REG32 ROR               : 1;
__REG32                   : 1;
__REG32 TFL               : 4;
__REG32 RFL               : 4;
__REG32                   :16;
} __sasr0_bits;

/* Serial Audio Clock Divider Register (SADIV) */
typedef struct{
__REG32 SADIV             : 7;
__REG32                   :25;
} __sadiv_bits;

/* Serial Audio Interrupt Clear Register (SAICR) */
typedef struct{
__REG32                   : 5;
__REG32 TUR               : 1;
__REG32 ROR               : 1;
__REG32                   :25;
} __saicr_bits;

/* Serial Audio Interrupt Mask Register (SAIMR) */
typedef struct{
__REG32                   : 3;
__REG32 TFS               : 1;
__REG32 RFS               : 1;
__REG32 TUR               : 1;
__REG32 ROR               : 1;
__REG32                   :25;
} __saimr_bits;

/* Serial Audio Data Register (SADR) */
typedef struct{
__REG32 DTL               :16;
__REG32 DTH               :16;
} __sadr_bits;

/* MMC Clock Start/Stop Register (MMC_STRPCL) */
typedef struct{
__REG32  STRPCL           : 2;
__REG32                   :30;
} __mmc_strpcl_bits;

/* MMC Status Register (MMC_STAT) */
typedef struct{
__REG32 READ_TIME_OUT         : 1;
__REG32 TIME_OUT_RESPONSE     : 1;
__REG32 CRC_WRITE_ERROR       : 1;
__REG32 CRC_READ_ERROR        : 1;
__REG32 SPI_READ_ERROR_TOKEN  : 1;
__REG32 RES_CRC_ERR           : 1;
__REG32 XMIT_FIFO_EMPTY       : 1;
__REG32 RECV_FIFO_FULL        : 1;
__REG32 CLK_EN                : 1;
__REG32                       : 2;
__REG32 DATA_TRAN_DONE        : 1;
__REG32 PRG_DONE              : 1;
__REG32 END_CMD_RES           : 1;
__REG32                       :18;
} __mmc_stat_bits;

/* MMC Clock Rate Register (MMC_CLKRT) */
typedef struct{
__REG32 CLK_RATE          : 3;
__REG32                   :29;
} __mmc_clkrt_bits;

/* MMC_SPI Register (MMC_SPI) */
typedef struct{
__REG32 SPI_EN            : 1;
__REG32 CRC_ON            : 1;
__REG32 SPI_CS_EN         : 1;
__REG32 SPI_CS_ADDRESS    : 1;
__REG32                   :28;
} __mmc_spi_bits;

/* MMC_CMDAT Register (MMC_CMDAT) */
typedef struct{
__REG32 RESPONSE_FORMAT   : 2;
__REG32 DATA_EN           : 1;
__REG32 WR_RD             : 1;
__REG32 STRM_BLK          : 1;
__REG32 BUSY              : 1;
__REG32 INIT              : 1;
__REG32 DMA_EN            : 1;
__REG32                   :24;
} __mmc_cmdat_bits;

/* MMC_RESTO Register (MMC_RESTO) */
typedef struct{
__REG32 RES_TO            : 7;
__REG32                   :25;
} __mmc_resto_bits;

/*  MMC_RDTO Register (MMC_RDTO) */
typedef struct{
__REG32 READ_TO           :16;
__REG32                   :16;
} __mmc_rdto_bits;

/* MMC_BLKLEN Register (MMC_BLKLEN) */
typedef struct{
__REG32 BLK_LEN           :10;
__REG32                   :22;
} __mmc_blklen_bits;

/* MMC_NOB Register (MMC_NOB) */
typedef struct{
__REG32 MMC_NOB           :16;
__REG32                   :16;
} __mmc_nob_bits;

/* MMC_PRTBUF Register (MMC_PRTBUF) */
typedef struct{
__REG32 PRT_BUF           : 1;
__REG32                   :31;
} __mmc_prtbuf_bits;

/* MMC_I_MASK Register (MMC_I_MASK) */
typedef struct{
__REG32 DATA_TRAN_DONE    : 1;
__REG32 PRG_DONE          : 1;
__REG32 END_CMD_RES       : 1;
__REG32 STOP_CMD          : 1;
__REG32 CLK_IS_OFF        : 1;
__REG32 RXFIFO_RD_REQ     : 1;
__REG32 TXFIFO_WR_REQ     : 1;
__REG32                   :25;
} __mmc_i_mask_bits;

/* MMC_CMD Register (MMC_CMD) */
typedef struct{
__REG32 CMD_INDX          : 6;
__REG32                   :26;
} __mmc_cmd_bits;

/* MMC_ARGH Register (MMC_ARGH) */
typedef struct{
__REG32 ARG_H             :16;
__REG32                   :16;
} __mmc_argh_bits;

/* MMC_ARGL Register (MMC_ARGL) */
typedef struct{
__REG32 ARG_L             :16;
__REG32                   :16;
} __mmc_argl_bits;

/* MMC RESPONSE FIFO (MMC_RES) */
typedef struct{
__REG32 DATA              :16;
__REG32                   :16;
} __mmc_res_bits;

/* MMC RECEIVE FIFO (MMC_RXFIFO) */
typedef struct{
__REG32 DATA              : 8;
__REG32                   :24;
} __mmc_rxfifo_bits;

/* MMC TRANSMIT FIFO (MMC_TXFIFO) */
typedef struct{
__REG32 DATA              : 8;
__REG32                   :24;
} __mmc_txfifo_bits;

/* SSP Control Register 0 (SSCR0) */
typedef struct {
  __REG32 DSS               : 4;
  __REG32 FRF               : 2;
  __REG32                   : 1;
  __REG32 SSE               : 1;
  __REG32 SCR               :12;
  __REG32 EDSS              : 1;
  __REG32                   :11;
} __sscr0_bits;

/* SSP Control Register 1 (SSCR1) */
typedef struct {
  __REG32 RIE               : 1;
  __REG32 TIE               : 1;
  __REG32 LBM               : 1;
  __REG32 SPO               : 1;
  __REG32 SPH               : 1;
  __REG32 MWDS              : 1;
  __REG32 TFT               : 4;
  __REG32 RFT               : 4;
  __REG32 EFWR              : 1;
  __REG32 STRF              : 1;
  __REG32                   : 3;
  __REG32 TINTE             : 1;
  __REG32 RSRE              : 1;
  __REG32 TSRE              : 1;
  __REG32                   : 1;
  __REG32 RWOT              : 1;
  __REG32 SFRMDIR           : 1;
  __REG32 SCLKDIR           : 1;
  __REG32                   : 2;
  __REG32 SCFR              : 1;
  __REG32 EBCEI             : 1;
  __REG32 TTE               : 1;
  __REG32 TTELP             : 1;
} __sscr1_bits;

/* SSP Status Register (SSSR) */
typedef struct {
  __REG32                   : 2;
  __REG32 TNF               : 1;
  __REG32 RNE               : 1;
  __REG32 BSY               : 1;
  __REG32 TFS               : 1;
  __REG32 RFS               : 1;
  __REG32 ROR               : 1;
  __REG32 TFL               : 4;
  __REG32 RFL               : 4;
  __REG32                   : 3;
  __REG32 TINT              : 1;
  __REG32                   : 1;
  __REG32 TUR               : 1;
  __REG32 CSS               : 1;
  __REG32 BCE               : 1;
  __REG32                   : 8;
} __sssr_bits;

/* SSP Interrupt Test Register (SSITR) */
typedef struct {
  __REG32                   : 5;
  __REG32 TTFS              : 1;
  __REG32 TRFS              : 1;
  __REG32 TROR              : 1;
  __REG32                   :24;
} __ssitr_bits;

/* SSP Time Out Register (SSTO) */
typedef struct {
  __REG32 TIMEOUT           :24;
  __REG32                   : 8;
} __ssto_bits;

/* SSP Programmable Serial Protocol Register (SSPSP) */
typedef struct {
  __REG32 SCMODE            : 2;
  __REG32 SFRMP             : 1;
  __REG32 ETDS              : 1;
  __REG32 STRTDLY           : 3;
  __REG32 DMYSTRT           : 2;
  __REG32 SFRMDLY           : 7;
  __REG32 SFRMWDTH          : 6;
  __REG32                   : 1;
  __REG32 DMYSTOP           : 2;
  __REG32                   : 7;
} __sspsp_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** PM (Power Manager)
 **
 ***************************************************************************/
__IO_REG32_BIT(PMCR,                  0x40F00000,__READ_WRITE ,__pmcr_bits);
__IO_REG32_BIT(PSSR,                  0x40F00004,__READ_WRITE ,__pssr_bits);
__IO_REG32(    PSPR,                  0x40F00008,__READ_WRITE );
__IO_REG32_BIT(PWER,                  0x40F0000C,__READ_WRITE ,__pwer_bits);
__IO_REG32_BIT(PRER,                  0x40F00010,__READ_WRITE ,__prer_bits);
__IO_REG32_BIT(PFER,                  0x40F00014,__READ_WRITE ,__pfer_bits);
__IO_REG32_BIT(PEDR,                  0x40F00018,__READ_WRITE ,__pedr_bits);
__IO_REG32_BIT(PCFR,                  0x40F0001C,__READ_WRITE ,__pcfr_bits);
__IO_REG32_BIT(PGSR0,                 0x40F00020,__READ_WRITE ,__pgsr0_bits);
__IO_REG32_BIT(PGSR1,                 0x40F00024,__READ_WRITE ,__pgsr1_bits);
__IO_REG32_BIT(PGSR2,                 0x40F00028,__READ_WRITE ,__pgsr2_bits);
__IO_REG32_BIT(RCSR,                  0x40F00030,__READ_WRITE ,__rcsr_bits);
__IO_REG32_BIT(PMFW,                  0x40F00034,__READ_WRITE ,__pmfw_bits);
__IO_REG32_BIT(CCCR,                  0x41300000,__READ_WRITE ,__cccr_bits);
__IO_REG32_BIT(CKEN,                  0x41300004,__READ_WRITE ,__cken_bits);
__IO_REG32_BIT(OSCC,                  0x41300008,__READ_WRITE ,__oscc_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPLR0,                 0x40E00000,__READ       ,__gplr0_bits);
__IO_REG32_BIT(GPLR1,                 0x40E00004,__READ       ,__gplr1_bits);
__IO_REG32_BIT(GPLR2,                 0x40E00008,__READ       ,__gplr2_bits);
__IO_REG32_BIT(GPDR0,                 0x40E0000C,__READ_WRITE ,__gpdr0_bits);
__IO_REG32_BIT(GPDR1,                 0x40E00010,__READ_WRITE ,__gpdr1_bits);
__IO_REG32_BIT(GPDR2,                 0x40E00014,__READ_WRITE ,__gpdr2_bits);
__IO_REG32_BIT(GPSR0,                 0x40E00018,__WRITE      ,__gpsr0_bits);
__IO_REG32_BIT(GPSR1,                 0x40E0001C,__WRITE      ,__gpsr1_bits);
__IO_REG32_BIT(GPSR2,                 0x40E00020,__WRITE      ,__gpsr2_bits);
__IO_REG32_BIT(GPCR0,                 0x40E00024,__WRITE      ,__gpcr0_bits);
__IO_REG32_BIT(GPCR1,                 0x40E00028,__WRITE      ,__gpcr1_bits);
__IO_REG32_BIT(GPCR2,                 0x40E0002C,__WRITE      ,__gpcr2_bits);
__IO_REG32_BIT(GRER0,                 0x40E00030,__READ_WRITE ,__grer0_bits);
__IO_REG32_BIT(GRER1,                 0x40E00034,__READ_WRITE ,__grer1_bits);
__IO_REG32_BIT(GRER2,                 0x40E00038,__READ_WRITE ,__grer2_bits);
__IO_REG32_BIT(GFER0,                 0x40E0003C,__READ_WRITE ,__gfer0_bits);
__IO_REG32_BIT(GFER1,                 0x40E00040,__READ_WRITE ,__gfer1_bits);
__IO_REG32_BIT(GFER2,                 0x40E00044,__READ_WRITE ,__gfer2_bits);
__IO_REG32_BIT(GEDR0,                 0x40E00048,__READ_WRITE ,__gedr0_bits);
__IO_REG32_BIT(GEDR1,                 0x40E0004C,__READ_WRITE ,__gedr1_bits);
__IO_REG32_BIT(GEDR2,                 0x40E00050,__READ_WRITE ,__gedr2_bits);
__IO_REG32_BIT(GAFR0_L,               0x40E00054,__READ_WRITE ,__gafr0_l_bits);
__IO_REG32_BIT(GAFR0_U,               0x40E00058,__READ_WRITE ,__gafr0_u_bits);
__IO_REG32_BIT(GAFR1_L,               0x40E0005C,__READ_WRITE ,__gafr1_l_bits);
__IO_REG32_BIT(GAFR1_U,               0x40E00060,__READ_WRITE ,__gafr1_u_bits);
__IO_REG32_BIT(GAFR2_L,               0x40E00064,__READ_WRITE ,__gafr2_l_bits);
__IO_REG32_BIT(GAFR2_U,               0x40E00068,__READ_WRITE ,__gafr2_u_bits);

/***************************************************************************
 **
 ** IC (Interrup controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(ICPR,                  0x40D00010,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICIP,                  0x40D00000,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICFP,                  0x40D0000C,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICMR,                  0x40D00004,__READ_WRITE ,__icpr_bits);
__IO_REG32_BIT(ICLR,                  0x40D00008,__READ_WRITE ,__icpr_bits);
__IO_REG32_BIT(ICCR,                  0x40D00014,__READ_WRITE ,__iccr_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32(    RCNR,                  0x40900000,__READ_WRITE );
__IO_REG32(    RTAR,                  0x40900004,__READ_WRITE );
__IO_REG32_BIT(RTSR,                  0x40900008,__READ_WRITE ,__rtsr_bits);
__IO_REG32_BIT(RTTR,                  0x4090000C,__READ_WRITE ,__rttr_bits);

/***************************************************************************
 **
 ** OST (Operating System Timers)
 **
 ***************************************************************************/
__IO_REG32(    OSMR0,                 0x40A00000,__READ_WRITE );
__IO_REG32(    OSMR1,                 0x40A00004,__READ_WRITE );
__IO_REG32(    OSMR2,                 0x40A00008,__READ_WRITE );
__IO_REG32(    OSMR3,                 0x40A0000C,__READ_WRITE );
__IO_REG32_BIT(OWER,                  0x40A00018,__READ_WRITE ,__ower_bits);
__IO_REG32_BIT(OIER,                  0x40A0001C,__READ_WRITE ,__oier_bits);
__IO_REG32(    OSCR0,                 0x40A00010,__READ_WRITE );
__IO_REG32_BIT(OSSR,                  0x40A00014,__READ_WRITE ,__ossr_bits);

/***************************************************************************
 **
 ** PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_CTRL0,             0x40B00000,__READ_WRITE ,__pwm_ctrl_bits );
__IO_REG32_BIT(PWM_DUTY0,             0x40B00004,__READ_WRITE ,__pwm_duty_bits);
__IO_REG32_BIT(PWM_PERVAL0,           0x40B00008,__READ_WRITE ,__pwm_perval_bits);

/***************************************************************************
 **
 ** PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWM_CTRL1,             0x40C00000,__READ_WRITE ,__pwm_ctrl_bits );
__IO_REG32_BIT(PWM_DUTY1,             0x40C00004,__READ_WRITE ,__pwm_duty_bits);
__IO_REG32_BIT(PWM_PERVAL1,           0x40C00008,__READ_WRITE ,__pwm_perval_bits);

/***************************************************************************
 **
 ** DMA
 **
 ***************************************************************************/
__IO_REG32_BIT(DCSR0,                 0x40000000,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR1,                 0x40000004,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR2,                 0x40000008,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR3,                 0x4000000C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR4,                 0x40000010,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR5,                 0x40000014,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR6,                 0x40000018,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR7,                 0x4000001C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR8,                 0x40000020,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR9,                 0x40000024,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR10,                0x40000028,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR11,                0x4000002C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR12,                0x40000030,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR13,                0x40000034,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR14,                0x40000038,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR15,                0x4000003C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DINT,                  0x400000F0,__READ       ,__dint_bits);
__IO_REG32_BIT(DRCMR0,                0x40000100,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR2,                0x40000108,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR3,                0x4000010C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR4,                0x40000110,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR5,                0x40000114,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR6,                0x40000118,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR7,                0x4000011C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR8,                0x40000120,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR9,                0x40000124,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR10,               0x40000128,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR11,               0x4000012C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR12,               0x40000130,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR13,               0x40000134,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR14,               0x40000138,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR17,               0x40000144,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR18,               0x40000148,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR19,               0x4000014C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR20,               0x40000150,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR21,               0x40000154,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR22,               0x40000158,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR25,               0x40000164,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR26,               0x40000168,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR27,               0x4000016C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR28,               0x40000170,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR30,               0x40000178,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR31,               0x4000017C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR32,               0x40000180,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR33,               0x40000184,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR34,               0x40000188,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR35,               0x4000018C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR36,               0x40000190,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR37,               0x40000194,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR38,               0x40000198,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DDADR0,                0x40000200,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR0,                0x40000204,__READ_WRITE );
__IO_REG32(    DTADR0,                0x40000208,__READ_WRITE );
__IO_REG32_BIT( DCMD0,                0x4000020C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR1,                0x40000210,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR1,                0x40000214,__READ_WRITE );
__IO_REG32(    DTADR1,                0x40000218,__READ_WRITE );
__IO_REG32_BIT( DCMD1,                0x4000021C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR2,                0x40000220,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR2,                0x40000224,__READ_WRITE );
__IO_REG32(    DTADR2,                0x40000228,__READ_WRITE );
__IO_REG32_BIT( DCMD2,                0x4000022C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR3,                0x40000230,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR3,                0x40000234,__READ_WRITE );
__IO_REG32(    DTADR3,                0x40000238,__READ_WRITE );
__IO_REG32_BIT( DCMD3,                0x4000023C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR4,                0x40000240,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR4,                0x40000244,__READ_WRITE );
__IO_REG32(    DTADR4,                0x40000248,__READ_WRITE );
__IO_REG32_BIT( DCMD4,                0x4000024C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR5,                0x40000250,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR5,                0x40000254,__READ_WRITE );
__IO_REG32(    DTADR5,                0x40000258,__READ_WRITE );
__IO_REG32_BIT( DCMD5,                0x4000025C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR6,                0x40000260,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR6,                0x40000264,__READ_WRITE );
__IO_REG32(    DTADR6,                0x40000268,__READ_WRITE );
__IO_REG32_BIT( DCMD6,                0x4000026C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR7,                0x40000270,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR7,                0x40000274,__READ_WRITE );
__IO_REG32(    DTADR7,                0x40000278,__READ_WRITE );
__IO_REG32_BIT( DCMD7,                0x4000027C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR8,                0x40000280,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR8,                0x40000284,__READ_WRITE );
__IO_REG32(    DTADR8,                0x40000288,__READ_WRITE );
__IO_REG32_BIT( DCMD8,                0x4000028C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR9,                0x40000290,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR9,                0x40000294,__READ_WRITE );
__IO_REG32(    DTADR9,                0x40000298,__READ_WRITE );
__IO_REG32_BIT( DCMD9,                0x4000029C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR10,               0x400002A0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR10,               0x400002A4,__READ_WRITE );
__IO_REG32(    DTADR10,               0x400002A8,__READ_WRITE );
__IO_REG32_BIT( DCMD10,               0x400002AC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR11,               0x400002B0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR11,               0x400002B4,__READ_WRITE );
__IO_REG32(    DTADR11,               0x400002B8,__READ_WRITE );
__IO_REG32_BIT( DCMD11,               0x400002BC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR12,               0x400002C0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR12,               0x400002C4,__READ_WRITE );
__IO_REG32(    DTADR12,               0x400002C8,__READ_WRITE );
__IO_REG32_BIT( DCMD12,               0x400002CC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR13,               0x400002D0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR13,               0x400002D4,__READ_WRITE );
__IO_REG32(    DTADR13,               0x400002D8,__READ_WRITE );
__IO_REG32_BIT( DCMD13,               0x400002DC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR14,               0x400002E0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR14,               0x400002E4,__READ_WRITE );
__IO_REG32(    DTADR14,               0x400002E8,__READ_WRITE );
__IO_REG32_BIT( DCMD14,               0x400002EC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR15,               0x400002F0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR15,               0x400002F4,__READ_WRITE );
__IO_REG32(    DTADR15,               0x400002F8,__READ_WRITE );
__IO_REG32_BIT( DCMD15,               0x400002FC,__READ_WRITE ,__dcmd_bits);

/***************************************************************************
 **
 ** MC ( Memory Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(MDCNFG,                0x48000000,__READ_WRITE ,__mdcnfg_bits  );
__IO_REG32_BIT(MDREFR,                0x48000004,__READ_WRITE ,__mdrefr_bits  );
__IO_REG32_BIT(MDMRS,                 0x48000040,__READ_WRITE ,__mdmrs_bits   );
__IO_REG32_BIT(MDMRSLP,               0x48000058,__READ_WRITE ,__mdmrslp_bits );
__IO_REG32_BIT(SXCNFG,                0x4800001C,__READ_WRITE ,__sxcnfg_bits  );
__IO_REG32_BIT(SXMRS,                 0x48000024,__READ_WRITE ,__sxmrs_bits  );
__IO_REG32_BIT(SA1111CR,              0x48000064,__READ_WRITE ,__sa1111cr_bits  );
__IO_REG32_BIT(MSC0,                  0x48000008,__READ_WRITE ,__msc_bits     );
__IO_REG32_BIT(MSC1,                  0x4800000C,__READ_WRITE ,__msc_bits     );
__IO_REG32_BIT(MSC2,                  0x48000010,__READ_WRITE ,__msc_bits     );
__IO_REG32_BIT(MCMEM0,                0x48000028,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MCMEM1,                0x4800002C,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MCATT0,                0x48000030,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MCATT1,                0x48000034,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MCIO0,                 0x48000038,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MCIO1,                 0x4800003C,__READ_WRITE ,__mcmem_bits);
__IO_REG32_BIT(MECR,                  0x48000014,__READ_WRITE ,__mecr_bits);
__IO_REG32_BIT(BOOT_DEF,              0x48000044,__READ       ,__boot_def_bits);

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCCR0,                 0x44000000,__READ_WRITE ,__lccr0_bits);
__IO_REG32_BIT(LCCR1,                 0x44000004,__READ_WRITE ,__lccr1_bits);
__IO_REG32_BIT(LCCR2,                 0x44000008,__READ_WRITE ,__lccr2_bits);
__IO_REG32_BIT(LCCR3,                 0x4400000C,__READ_WRITE ,__lccr3_bits);
__IO_REG32_BIT(TRGBR,                 0x44000040,__READ_WRITE ,__trgbr_bits);
__IO_REG32_BIT(TCR,                   0x44000044,__READ_WRITE ,__tcr_bits);
__IO_REG32_BIT(FBR0,                  0x44000020,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR1,                  0x44000024,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(LCSR,                  0x44000038,__READ_WRITE ,__lcsr_bits);
__IO_REG32(    LIIDR,                 0x4400003C,__READ       );
__IO_REG32(    FDADR0,                0x44000200,__READ_WRITE );
__IO_REG32(    FSADR0,                0x44000204,__READ_WRITE );
__IO_REG32(    FIDR0,                 0x44000208,__READ       );
__IO_REG32_BIT(LDCMD0,                0x4400020C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR1,                0x44000210,__READ_WRITE );
__IO_REG32(    FSADR1,                0x44000214,__READ_WRITE );
__IO_REG32(    FIDR1,                 0x44000218,__READ       );
__IO_REG32_BIT(LDCMD1,                0x4400021C,__READ       ,__ldcmd_bits);

/***************************************************************************
 **
 ** I2C
 **
 ***************************************************************************/
__IO_REG32_BIT(IBMR,                  0x40301680,__READ       ,__ibmr_bits  );
__IO_REG32_BIT(IDBR,                  0x40301688,__READ_WRITE ,__idbr_bits  );
__IO_REG32_BIT(ICR,                   0x40301690,__READ_WRITE ,__icr_bits   );
__IO_REG32_BIT(ISR,                   0x40301698,__READ_WRITE ,__isr_bits   );
__IO_REG32_BIT(ISAR,                  0x403016A0,__READ_WRITE ,__isar_bits  );

/***************************************************************************
 **
 ** UART1 (Full Function UART)
 **
 ***************************************************************************/
__IO_REG32_BIT(FFRBR,                 0x40100000,__READ_WRITE ,__uartrbr_bits   );
#define FFTHR FFRBR
#define FFTHR_bit FFRBR_bit
#define FFDLL FFRBR
#define FFDLL_bit FFRBR_bit
__IO_REG32_BIT(FFIER,                 0x40100004,__READ_WRITE ,__uartier_bits   );
#define FFDLH FFIER
#define FFDLH_bit FFIER_bit
__IO_REG32_BIT(FFIIR,                 0x40100008,__READ_WRITE ,__uartiir_bits   );
#define FFFCR FFIIR
#define FFFCR_bit FFIIR_bit
__IO_REG32_BIT(FFLCR,                 0x4010000C,__READ_WRITE ,__uartlcr_bits   );
__IO_REG32_BIT(FFMCR,                 0x40100010,__READ_WRITE ,__uartmcr_bits   );
__IO_REG32_BIT(FFLSR,                 0x40100014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(FFMSR,                 0x40100018,__READ       ,__uartmsr_bits   );
__IO_REG32_BIT(FFSPR,                 0x4010001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(FFISR,                 0x40100020,__READ_WRITE ,__uartisr_bits   );

/***************************************************************************
 **
 ** UART2 (Bluetooth UART)
 **
 ***************************************************************************/
__IO_REG32_BIT(BTRBR,                 0x40200000,__READ_WRITE ,__uartrbr_bits   );
#define BTTHR BTRBR
#define BTTHR_bit BTRBR_bit
#define BTDLL BTRBR
#define BTDLL_bit BTRBR_bit
__IO_REG32_BIT(BTIER,                 0x40200004,__READ_WRITE ,__uartier_bits   );
#define BTDLH BTIER
#define BTDLH_bit BTIER_bit
__IO_REG32_BIT(BTIIR,                 0x40200008,__READ_WRITE ,__uartiir_bits   );
#define BTFCR BTIIR
#define BTFCR_bit BTIIR_bit
__IO_REG32_BIT(BTLCR,                 0x4020000C,__READ_WRITE ,__uartlcr_bits   );
__IO_REG32_BIT(BTMCR,                 0x40200010,__READ_WRITE ,__btmcr_bits     );
__IO_REG32_BIT(BTLSR,                 0x40200014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(BTMSR,                 0x40200018,__READ       ,__btmsr_bits     );
__IO_REG32_BIT(BTSPR,                 0x4020001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(BTISR,                 0x40200020,__READ_WRITE ,__uartisr_bits   );

/***************************************************************************
 **
 ** UART3 (Standard UART)
 **
 ***************************************************************************/
__IO_REG32_BIT(STRBR,                 0x40700000,__READ_WRITE ,__uartrbr_bits   );
#define STTHR STRBR
#define STTHR_bit STRBR_bit
#define STDLL STRBR
#define STDLL_bit STRBR_bit
__IO_REG32_BIT(STIER,                 0x40700004,__READ_WRITE ,__uartier_bits   );
#define STDLH STIER
#define STDLH_bit STIER_bit
__IO_REG32_BIT(STIIR,                 0x40700008,__READ_WRITE ,__uartiir_bits   );
#define STFCR STIIR
#define STFCR_bit STIIR_bit
__IO_REG32_BIT(STLCR,                 0x4070000C,__READ_WRITE ,__uartlcr_bits   );
__IO_REG32_BIT(STMCR,                 0x40700010,__READ_WRITE ,__stmcr_bits     );
__IO_REG32_BIT(STLSR,                 0x40700014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(STSPR,                 0x4070001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(STISR,                 0x40700020,__READ_WRITE ,__uartisr_bits   );

/***************************************************************************
 **
 ** UART4 (Hardware UART)
 **
 ***************************************************************************/
__IO_REG32_BIT(HWRBR,                 0x41600000,__READ_WRITE ,__uartrbr_bits   );
#define HWTHR HWRBR
#define HWTHR_bit HWRBR_bit
#define HWDLL HWRBR
#define HWDLL_bit HWRBR_bit
__IO_REG32_BIT(HWIER,                 0x41600004,__READ_WRITE ,__uartier_bits   );
#define HWDLH HWIER
#define HWDLH_bit HWIER_bit
__IO_REG32_BIT(HWIIR,                 0x41600008,__READ_WRITE ,__hwiir_bits     );
#define HWFCR HWIIR
#define HWFCR_bit HWIIR_bit
__IO_REG32_BIT(HWLCR,                 0x4160000C,__READ_WRITE ,__uartlcr_bits   );
__IO_REG32_BIT(HWMCR,                 0x41600010,__READ_WRITE ,__hwmcr_bits     );
__IO_REG32_BIT(HWLSR,                 0x41600014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(HWMSR,                 0x41600018,__READ       ,__btmsr_bits     );
__IO_REG32_BIT(HWSPR,                 0x4160001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(HWISR,                 0x41600020,__READ_WRITE ,__uartisr_bits   );
__IO_REG32_BIT(HWFOR,                 0x41600024,__READ_WRITE ,__uartfor_bits   );
__IO_REG32_BIT(HWABR,                 0x41600028,__READ_WRITE ,__uartabr_bits   );
__IO_REG32_BIT(HWACR,                 0x4160002C,__READ       ,__uartacr_bits   );

/***************************************************************************
 **
 ** FIRCP (Fast Infrared Communications Port)
 **
 ***************************************************************************/
__IO_REG32_BIT(ICCR0,                 0x40800000,__READ_WRITE ,__iccr0_bits);
__IO_REG32_BIT(ICCR1,                 0x40800004,__READ_WRITE ,__iccr1_bits);
__IO_REG32_BIT(ICCR2,                 0x40800008,__READ_WRITE ,__iccr2_bits);
__IO_REG32_BIT(ICDR,                  0x4080000C,__READ_WRITE ,__icdr_bits);
__IO_REG32_BIT(ICSR0,                 0x40800014,__READ_WRITE ,__icsr0_bits);
__IO_REG32_BIT(ICSR1,                 0x40800018,__READ       ,__icsr1_bits);

/***************************************************************************
 **
 ** UDC ( USB 1.1 Client Controller )
 **
 ***************************************************************************/
__IO_REG32_BIT(UDCCR,                 0x40600000,__READ_WRITE ,__udccr_bits);
__IO_REG32_BIT(UDCCFR,                0x40600008,__READ_WRITE ,__udccfr_bits);
__IO_REG32_BIT(UDCCS0,                0x40600010,__READ_WRITE ,__udccs0_bits);
__IO_REG32_BIT(UDCCS1,                0x40600014,__READ_WRITE ,__udccs1_bits);
__IO_REG32_BIT(UDCCS2,                0x40600018,__READ_WRITE ,__udccs2_bits);
__IO_REG32_BIT(UDCCS3,                0x4060001C,__READ_WRITE ,__udccs3_bits);
__IO_REG32_BIT(UDCCS4,                0x40600020,__READ_WRITE ,__udccs4_bits);
__IO_REG32_BIT(UDCCS5,                0x40600024,__READ_WRITE ,__udccs5_bits);
__IO_REG32_BIT(UDCCS6,                0x40600028,__READ_WRITE ,__udccs1_bits);
__IO_REG32_BIT(UDCCS7,                0x4060002C,__READ_WRITE ,__udccs2_bits);
__IO_REG32_BIT(UDCCS8,                0x40600030,__READ_WRITE ,__udccs3_bits);
__IO_REG32_BIT(UDCCS9,                0x40600034,__READ_WRITE ,__udccs4_bits);
__IO_REG32_BIT(UDCCS10,               0x40600038,__READ_WRITE ,__udccs5_bits);
__IO_REG32_BIT(UDCCS11,               0x4060003C,__READ_WRITE ,__udccs1_bits);
__IO_REG32_BIT(UDCCS12,               0x40600040,__READ_WRITE ,__udccs2_bits);
__IO_REG32_BIT(UDCCS13,               0x40600044,__READ_WRITE ,__udccs3_bits);
__IO_REG32_BIT(UDCCS14,               0x40600048,__READ_WRITE ,__udccs4_bits);
__IO_REG32_BIT(UDCCS15,               0x4060004C,__READ_WRITE ,__udccs5_bits);
__IO_REG32_BIT(UICR0,                 0x40600050,__READ_WRITE ,__uicr0_bits);
__IO_REG32_BIT(UICR1,                 0x40600054,__READ_WRITE ,__uicr1_bits);
__IO_REG32_BIT(USIR0,                 0x40600058,__READ_WRITE ,__usir0_bits);
__IO_REG32_BIT(USIR1,                 0x4060005C,__READ_WRITE ,__usir1_bits);
__IO_REG32_BIT(UFNHR,                 0x40600060,__READ_WRITE ,__ufnhr_bits);
__IO_REG32_BIT(UFNLR,                 0x40600064,__READ_WRITE ,__ufnlr_bits);
__IO_REG32_BIT(UBCR2,                 0x40600068,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UBCR4,                 0x4060006C,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UBCR7,                 0x40600070,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UBCR9,                 0x40600074,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UBCR12,                0x40600078,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UBCR14,                0x4060007C,__READ_WRITE ,__ubcr2_bits);
__IO_REG32_BIT(UDDR0,                 0x40600080,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR1,                 0x40600100,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR2,                 0x40600180,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR3,                 0x40600200,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR4,                 0x40600400,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR5,                 0x406000A0,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR6,                 0x40600600,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR7,                 0x40600680,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR8,                 0x40600700,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR9,                 0x40600900,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR10,                0x406000C0,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR11,                0x40600B00,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR12,                0x40600B80,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR13,                0x40600C00,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR14,                0x40600E00,__READ_WRITE ,__uddr_bits);
__IO_REG32_BIT(UDDR15,                0x406000E0,__READ_WRITE ,__uddr_bits);

/***************************************************************************
 **
 ** AC97 Controller
 **
 ***************************************************************************/
__IO_REG32_BIT(POCR,                  0x40500000,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(PICR,                  0x40500004,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MCCR,                  0x40500008,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(GCR,                   0x4050000C,__READ_WRITE ,__gcr_bits   );
__IO_REG32_BIT(POSR,                  0x40500010,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(PISR,                  0x40500014,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(MCSR,                  0x40500018,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(GSR,                   0x4050001C,__READ_WRITE ,__gsr_bits   );
__IO_REG32_BIT(CAR,                   0x40500020,__READ_WRITE ,__car_bits   );
__IO_REG32_BIT(PCDR,                  0x40500040,__READ_WRITE ,__pcdr_bits  );
__IO_REG32_BIT(MCDR,                  0x40500060,__READ       ,__mcdr_bits  );
__IO_REG32_BIT(MOCR,                  0x40500100,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MICR,                  0x40500108,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MOSR,                  0x40500110,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(MISR,                  0x40500118,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(MODR,                  0x40500140,__READ_WRITE ,__modr_bits  );

/***************************************************************************
 **
 ** I2S
 **
 ***************************************************************************/
__IO_REG32_BIT(SACR0,                 0x40400000,__READ_WRITE ,__sacr0_bits);
__IO_REG32_BIT(SACR1,                 0x40400004,__READ_WRITE ,__sacr1_bits);
__IO_REG32_BIT(SASR0,                 0x4040000C,__READ       ,__sasr0_bits);
__IO_REG32_BIT(SADIV,                 0x40400060,__READ_WRITE ,__sadiv_bits);
__IO_REG32_BIT(SAICR,                 0x40400018,__WRITE      ,__saicr_bits);
__IO_REG32_BIT(SAIMR,                 0x40400014,__READ_WRITE ,__saimr_bits);
__IO_REG32_BIT(SADR,                  0x40400080,__READ_WRITE ,__sadr_bits);

/***************************************************************************
 **
 ** MMC  
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC_STRPCL,            0x41100000,__READ_WRITE ,__mmc_strpcl_bits  );
__IO_REG32_BIT(MMC_STAT,              0x41100004,__READ       ,__mmc_stat_bits    );
__IO_REG32_BIT(MMC_CLKRT,             0x41100008,__READ_WRITE ,__mmc_clkrt_bits   );
__IO_REG32_BIT(MMC_SPI,               0x4110000C,__READ_WRITE ,__mmc_spi_bits     );
__IO_REG32_BIT(MMC_CMDAT,             0x41100010,__READ_WRITE ,__mmc_cmdat_bits   );
__IO_REG32_BIT(MMC_RESTO,             0x41100014,__READ_WRITE ,__mmc_resto_bits   );
__IO_REG32_BIT(MMC_RDTO,              0x41100018,__READ_WRITE ,__mmc_rdto_bits    );
__IO_REG32_BIT(MMC_BLKLEN,            0x4110001C,__READ_WRITE ,__mmc_blklen_bits  );
__IO_REG32_BIT(MMC_NOB,               0x41100020,__READ_WRITE ,__mmc_nob_bits  );
__IO_REG32_BIT(MMC_PRTBUF,            0x41100024,__READ_WRITE ,__mmc_prtbuf_bits  );
__IO_REG32_BIT(MMC_I_MASK,            0x41100028,__READ_WRITE ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC_I_REG,             0x4110002C,__READ       ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC_CMD,               0x41100030,__READ_WRITE ,__mmc_cmd_bits     );
__IO_REG32_BIT(MMC_ARGH,              0x41100034,__READ_WRITE ,__mmc_argh_bits    );
__IO_REG32_BIT(MMC_ARGL,              0x41100038,__READ_WRITE ,__mmc_argl_bits    );
__IO_REG32_BIT(MMC_RES,               0x4110003C,__READ       ,__mmc_res_bits     );
__IO_REG32_BIT(MMC_RXFIFO,            0x41100040,__READ       ,__mmc_rxfifo_bits  );
__IO_REG32_BIT(MMC_TXFIFO,            0x41100044,__WRITE      ,__mmc_txfifo_bits  );

/***************************************************************************
 **
 ** NSSP (Network SSP Serial Port)
 **
 ***************************************************************************/
__IO_REG32_BIT(NSSCR0,                0x41400000,__READ_WRITE ,__sscr0_bits);
__IO_REG32_BIT(NSSCR1,                0x41400004,__READ_WRITE ,__sscr1_bits);
__IO_REG32_BIT(NSSSR,                 0x41400008,__READ_WRITE ,__sssr_bits );
__IO_REG32_BIT(NSSITR,                0x4140000C,__READ_WRITE ,__ssitr_bits);
__IO_REG32(    NSSDR,                 0x41400010,__READ_WRITE                );
__IO_REG32_BIT(NSSTO,                 0x41400028,__READ_WRITE ,__ssto_bits );
__IO_REG32_BIT(NSSPSP,                0x4140002C,__READ_WRITE ,__sspsp_bits);

/***************************************************************************
 **  Assembler-specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

#endif    /* __IOPXA255_H */
