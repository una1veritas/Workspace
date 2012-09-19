/***************************************************************************
 **
 **    This file defines the Special Function Registers for
 **    Marvell PXA320
 **
 **    Used with ARM IAR C/C++ Compiler and Assembler.
 **
 **    (c) Copyright IAR Systems 2007
 **
 **    $Revision: 30251 $
 **
 **    Note: Only little endian addressing of 8 bit registers.
 ***************************************************************************/

#ifndef __IOPXA320_H
#define __IOPXA320_H

#if (((__TID__ >> 8) & 0x7F) != 0x4F)     /* 0x4F = 79 dec */
#error This file should only be compiled by ARM IAR compiler and assembler
#endif

#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **    PXA320 SPECIAL FUNCTION REGISTERS
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

/* MFPR Bit Definitions */
typedef struct{
__REG32 AF_SEL            : 3;
__REG32                   : 1;
__REG32 EDGE_RISE_EN      : 1;
__REG32 EDGE_FALL_EN      : 1;
__REG32 EDGE_CLEAR        : 1;
__REG32 SLEEP_OE_N        : 1;
__REG32 SLEEP_DATA        : 1;
__REG32 SLEEP_SEL         : 1;
__REG32 DRIVE             : 3;
__REG32 PULLDOWN_EN       : 1;
__REG32 PULLUP_EN         : 1;
__REG32 PULL_SEL          : 1;
__REG32                   :16;
} __mfpr_bits;

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
__REG32 PL85              : 1;
__REG32 PL86              : 1;
__REG32 PL87              : 1;
__REG32 PL88              : 1;
__REG32 PL89              : 1;
__REG32 PL90              : 1;
__REG32 PL91              : 1;
__REG32 PL92              : 1;
__REG32 PL93              : 1;
__REG32 PL94              : 1;
__REG32 PL95              : 1;
} __gplr2_bits;

/* GPIO Pin-Level Registers (GPLR3) */
typedef struct{
__REG32 PL96              : 1;
__REG32 PL97              : 1;
__REG32 PL98              : 1;
__REG32 PL99              : 1;
__REG32 PL100             : 1;
__REG32 PL101             : 1;
__REG32 PL102             : 1;
__REG32 PL103             : 1;
__REG32 PL104             : 1;
__REG32 PL105             : 1;
__REG32 PL106             : 1;
__REG32 PL107             : 1;
__REG32 PL108             : 1;
__REG32 PL109             : 1;
__REG32 PL110             : 1;
__REG32 PL111             : 1;
__REG32 PL112             : 1;
__REG32 PL113             : 1;
__REG32 PL114             : 1;
__REG32 PL115             : 1;
__REG32 PL116             : 1;
__REG32 PL117             : 1;
__REG32 PL118             : 1;
__REG32 PL119             : 1;
__REG32 PL120             : 1;
__REG32 PL121             : 1;
__REG32 PL122             : 1;
__REG32 PL123             : 1;
__REG32 PL124             : 1;
__REG32 PL125             : 1;
__REG32 PL126             : 1;
__REG32 PL127             : 1;
} __gplr3_bits;

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
__REG32 PD85              : 1;
__REG32 PD86              : 1;
__REG32 PD87              : 1;
__REG32 PD88              : 1;
__REG32 PD89              : 1;
__REG32 PD90              : 1;
__REG32 PD91              : 1;
__REG32 PD92              : 1;
__REG32 PD93              : 1;
__REG32 PD94              : 1;
__REG32 PD95              : 1;
} __gpdr2_bits;

/* GPIO Pin Direction Registers (GPDR3)
   GPIO Pin Bit-Wise Set Direction Registers (GSDR3)
   GPIO Pin Bit-Wise Clear Direction Registers (GCDR3)
   GPIO Bit-wise Set Rising-Edge (GSRER3) Detect-Enable Registers
   GPIO Bit-wise Clear Rising-Edge (GCRER3) Detect-Enable Registers
   GPIO Bit-wise Set Falling-Edge (GSFER3) Detect-Enable Registers
   GPIO Bit-wise Clear Falling-Edge (GCFER3) Detect-Enable Registers */
typedef struct{
__REG32 PD96              : 1;
__REG32 PD97              : 1;
__REG32 PD98              : 1;
__REG32 PD99              : 1;
__REG32 PD100             : 1;
__REG32 PD101             : 1;
__REG32 PD102             : 1;
__REG32 PD103             : 1;
__REG32 PD104             : 1;
__REG32 PD105             : 1;
__REG32 PD106             : 1;
__REG32 PD107             : 1;
__REG32 PD108             : 1;
__REG32 PD109             : 1;
__REG32 PD110             : 1;
__REG32 PD111             : 1;
__REG32 PD112             : 1;
__REG32 PD113             : 1;
__REG32 PD114             : 1;
__REG32 PD115             : 1;
__REG32 PD116             : 1;
__REG32 PD117             : 1;
__REG32 PD118             : 1;
__REG32 PD119             : 1;
__REG32 PD120             : 1;
__REG32 PD121             : 1;
__REG32 PD122             : 1;
__REG32 PD123             : 1;
__REG32 PD124             : 1;
__REG32 PD125             : 1;
__REG32 PD126             : 1;
__REG32 PD127             : 1;
} __gpdr3_bits;

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
__REG32 PS85              : 1;
__REG32 PS86              : 1;
__REG32 PS87              : 1;
__REG32 PS88              : 1;
__REG32 PS89              : 1;
__REG32 PS90              : 1;
__REG32 PS91              : 1;
__REG32 PS92              : 1;
__REG32 PS93              : 1;
__REG32 PS94              : 1;
__REG32 PS95              : 1;
} __gpsr2_bits;

/* GPIO Pin Output Set Registers (GPSR3) */
typedef struct{
__REG32 PS96              : 1;
__REG32 PS97              : 1;
__REG32 PS98              : 1;
__REG32 PS99              : 1;
__REG32 PS100             : 1;
__REG32 PS101             : 1;
__REG32 PS102             : 1;
__REG32 PS103             : 1;
__REG32 PS104             : 1;
__REG32 PS105             : 1;
__REG32 PS106             : 1;
__REG32 PS107             : 1;
__REG32 PS108             : 1;
__REG32 PS109             : 1;
__REG32 PS110             : 1;
__REG32 PS111             : 1;
__REG32 PS112             : 1;
__REG32 PS113             : 1;
__REG32 PS114             : 1;
__REG32 PS115             : 1;
__REG32 PS116             : 1;
__REG32 PS117             : 1;
__REG32 PS118             : 1;
__REG32 PS119             : 1;
__REG32 PS120             : 1;
__REG32 PS121             : 1;
__REG32 PS122             : 1;
__REG32 PS123             : 1;
__REG32 PS124             : 1;
__REG32 PS125             : 1;
__REG32 PS126             : 1;
__REG32 PS127             : 1;
} __gpsr3_bits;

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
__REG32 PC85              : 1;
__REG32 PC86              : 1;
__REG32 PC87              : 1;
__REG32 PC88              : 1;
__REG32 PC89              : 1;
__REG32 PC90              : 1;
__REG32 PC91              : 1;
__REG32 PC92              : 1;
__REG32 PC93              : 1;
__REG32 PC94              : 1;
__REG32 PC95              : 1;
} __gpcr2_bits;

/* GPIO Pin Output Clear Registers (GPCR3) */
typedef struct{
__REG32 PC96              : 1;
__REG32 PC97              : 1;
__REG32 PC98              : 1;
__REG32 PC99              : 1;
__REG32 PC100             : 1;
__REG32 PC101             : 1;
__REG32 PC102             : 1;
__REG32 PC103             : 1;
__REG32 PC104             : 1;
__REG32 PC105             : 1;
__REG32 PC106             : 1;
__REG32 PC107             : 1;
__REG32 PC108             : 1;
__REG32 PC109             : 1;
__REG32 PC110             : 1;
__REG32 PC111             : 1;
__REG32 PC112             : 1;
__REG32 PC113             : 1;
__REG32 PC114             : 1;
__REG32 PC115             : 1;
__REG32 PC116             : 1;
__REG32 PC117             : 1;
__REG32 PC118             : 1;
__REG32 PC119             : 1;
__REG32 PC120             : 1;
__REG32 PC121             : 1;
__REG32 PC122             : 1;
__REG32 PC123             : 1;
__REG32 PC124             : 1;
__REG32 PC125             : 1;
__REG32 PC126             : 1;
__REG32 PC127             : 1;
} __gpcr3_bits;

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
__REG32 RE85              : 1;
__REG32 RE86              : 1;
__REG32 RE87              : 1;
__REG32 RE88              : 1;
__REG32 RE89              : 1;
__REG32 RE90              : 1;
__REG32 RE91              : 1;
__REG32 RE92              : 1;
__REG32 RE93              : 1;
__REG32 RE94              : 1;
__REG32 RE95              : 1;
} __grer2_bits;

/* GPIO Rising-Edge Detect-Enable Registers (GRER3) */
typedef struct{
__REG32 RE96              : 1;
__REG32 RE97              : 1;
__REG32 RE98              : 1;
__REG32 RE99              : 1;
__REG32 RE100             : 1;
__REG32 RE101             : 1;
__REG32 RE102             : 1;
__REG32 RE103             : 1;
__REG32 RE104             : 1;
__REG32 RE105             : 1;
__REG32 RE106             : 1;
__REG32 RE107             : 1;
__REG32 RE108             : 1;
__REG32 RE109             : 1;
__REG32 RE110             : 1;
__REG32 RE111             : 1;
__REG32 RE112             : 1;
__REG32 RE113             : 1;
__REG32 RE114             : 1;
__REG32 RE115             : 1;
__REG32 RE116             : 1;
__REG32 RE117             : 1;
__REG32 RE118             : 1;
__REG32 RE119             : 1;
__REG32 RE120             : 1;
__REG32 RE121             : 1;
__REG32 RE122             : 1;
__REG32 RE123             : 1;
__REG32 RE124             : 1;
__REG32 RE125             : 1;
__REG32 RE126             : 1;
__REG32 RE127             : 1;
} __grer3_bits;

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
__REG32 FE85              : 1;
__REG32 FE86              : 1;
__REG32 FE87              : 1;
__REG32 FE88              : 1;
__REG32 FE89              : 1;
__REG32 FE90              : 1;
__REG32 FE91              : 1;
__REG32 FE92              : 1;
__REG32 FE93              : 1;
__REG32 FE94              : 1;
__REG32 FE95              : 1;
} __gfer2_bits;

/* GPIO Falling-Edge Detect-Enable Registers (GFER3) */
typedef struct{
__REG32 FE96              : 1;
__REG32 FE97              : 1;
__REG32 FE98              : 1;
__REG32 FE99              : 1;
__REG32 FE100             : 1;
__REG32 FE101             : 1;
__REG32 FE102             : 1;
__REG32 FE103             : 1;
__REG32 FE104             : 1;
__REG32 FE105             : 1;
__REG32 FE106             : 1;
__REG32 FE107             : 1;
__REG32 FE108             : 1;
__REG32 FE109             : 1;
__REG32 FE110             : 1;
__REG32 FE111             : 1;
__REG32 FE112             : 1;
__REG32 FE113             : 1;
__REG32 FE114             : 1;
__REG32 FE115             : 1;
__REG32 FE116             : 1;
__REG32 FE117             : 1;
__REG32 FE118             : 1;
__REG32 FE119             : 1;
__REG32 FE120             : 1;
__REG32 FE121             : 1;
__REG32 FE122             : 1;
__REG32 FE123             : 1;
__REG32 FE124             : 1;
__REG32 FE125             : 1;
__REG32 FE126             : 1;
__REG32 FE127             : 1;
} __gfer3_bits;

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
__REG32 ED85              : 1;
__REG32 ED86              : 1;
__REG32 ED87              : 1;
__REG32 ED88              : 1;
__REG32 ED89              : 1;
__REG32 ED90              : 1;
__REG32 ED91              : 1;
__REG32 ED92              : 1;
__REG32 ED93              : 1;
__REG32 ED94              : 1;
__REG32 ED95              : 1;
} __gedr2_bits;

/* GPIO Edge Detect Status Register (GEDR3) */
typedef struct{
__REG32 ED96              : 1;
__REG32 ED97              : 1;
__REG32 ED98              : 1;
__REG32 ED99              : 1;
__REG32 ED100             : 1;
__REG32 ED101             : 1;
__REG32 ED102             : 1;
__REG32 ED103             : 1;
__REG32 ED104             : 1;
__REG32 ED105             : 1;
__REG32 ED106             : 1;
__REG32 ED107             : 1;
__REG32 ED108             : 1;
__REG32 ED109             : 1;
__REG32 ED110             : 1;
__REG32 ED111             : 1;
__REG32 ED112             : 1;
__REG32 ED113             : 1;
__REG32 ED114             : 1;
__REG32 ED115             : 1;
__REG32 ED116             : 1;
__REG32 ED117             : 1;
__REG32 ED118             : 1;
__REG32 ED119             : 1;
__REG32 ED120             : 1;
__REG32 ED121             : 1;
__REG32 ED122             : 1;
__REG32 ED123             : 1;
__REG32 ED124             : 1;
__REG32 ED125             : 1;
__REG32 ED126             : 1;
__REG32 ED127             : 1;
} __gedr3_bits;

/* Oscillator Configuration Register (OSCC) */
typedef struct{
__REG32 VCXOST            : 8;
__REG32 TENS0             : 1;
__REG32 TENS2             : 1;
__REG32 TENS3             : 1;
__REG32 PEN               : 1;
__REG32 ROS               : 1;
__REG32                   : 3;
__REG32 TD                : 1;
__REG32                   :15;
} __oscc_bits;

/* Application Subsystem Clock Configuration Register (ACCR) */
typedef struct{
__REG32 XL                : 5;
__REG32                   : 3;
__REG32 XN                : 3;
__REG32 PCCE              : 1;
__REG32 DMCFS             : 2;
__REG32 HSS               : 2;
__REG32 XSPCLK            : 2;
__REG32 SFLFS             : 2;
__REG32                   : 3;
__REG32 SMCFS             : 3;
__REG32 D0CS              : 1;
__REG32                   : 3;
__REG32 SPDIS             : 1;
__REG32 XPDIS             : 1;
} __accr_bits;

/* Application Subsystem Clock Status Register (ACSR) */
typedef struct{
__REG32 XL_S              : 5;
__REG32                   : 3;
__REG32 XN_S              : 3;
__REG32                   : 1;
__REG32 DMC_S             : 2;
__REG32 HSS_S             : 2;
__REG32 XSPCLK_S          : 2;
__REG32 SFL_S             : 2;
__REG32                   : 3;
__REG32 SMC_S             : 3;
__REG32 RO_S              : 1;
__REG32                   : 1;
__REG32 SPLCK             : 1;
__REG32 XPLCK             : 1;
__REG32 SPDIS_S           : 1;
__REG32 XPDIS_S           : 1;
} __acsr_bits;

/* Application Subsystem Interrupt Control/Status Register (AICSR) */
typedef struct{
__REG32 FCIE              : 1;
__REG32 FCIS              : 1;
__REG32 TCIE              : 1;
__REG32 TCIS              : 1;
__REG32 PCIE              : 1;
__REG32 PCIS              : 1;
__REG32                   :26;
} __aicsr_bits;

/* D0 Mode Clock Enable Register A (D0CKEN_A) */
typedef struct{
__REG32                     : 1;
__REG32 LCD_CLKEN           : 1;
__REG32 USB_HOST_CLKEN      : 1;
__REG32 CAM_CLKEN           : 1;
__REG32 NAND_CLKEN          : 1;
__REG32                     : 1;
__REG32 USB_20_CLIENT_CLKEN : 1;
__REG32 GRAPH_CLKEN         : 1;
__REG32 DMC_CLKEN           : 1;
__REG32 SMC_CLKEN           : 1;
__REG32 ISRAM_CLKEN         : 1;
__REG32 BOOT_ROM_CLKEN      : 1;
__REG32 MMC0                : 1;
__REG32 MMC1                : 1;
__REG32 KBD_CLKEN           : 1;
__REG32 IR_CLKEN            : 1;
__REG32                     : 1;
__REG32 USIM0_CLKEN         : 1;
__REG32 USIM1_CLKEN         : 1;
__REG32                     : 1;
__REG32 UDC_CLKEN           : 1;
__REG32 UART2_CLKEN         : 1;
__REG32 UART1_CLKEN         : 1;
__REG32 UART3_CLKEN         : 1;
__REG32 AC97_CLKEN          : 1;
__REG32 TSI_CLKEN           : 1;
__REG32 SSP1_CLKEN          : 1;
__REG32 SSP2_CLKEN          : 1;
__REG32 SSP3_CLKEN          : 1;
__REG32 SSP4_CLKEN          : 1;
__REG32 MSL0_CLKEN          : 1;
__REG32                     : 1;
} __d0cken_a_bits;

/* D0 Mode Clock Enable Register B (D0CKEN_B) */
typedef struct{
__REG32 PWM01_CLKEN       : 1;
__REG32 PWM23_CLKEN       : 1;
__REG32                   : 2;
__REG32 I2C_CLKEN         : 1;
__REG32                   : 1;
__REG32 IC_CLKEN          : 1;
__REG32 GPIO_CLKEN        : 1;
__REG32 ONE_WIRE_CLKEN    : 1;
__REG32 HSIO2_CLKEN       : 1;
__REG32                   : 6;
__REG32 MINI_IM_CLKEN     : 1;
__REG32 MINI_LCD_CLKEN    : 1;
__REG32                   : 14;
} __d0cken_b_bits;

/* Power Management Unit Control Register (PMCR) */
typedef struct{
__REG32 BIE               : 1;
__REG32 BIS               : 1;
__REG32                   : 8;
__REG32 TIE               : 1;
__REG32 TIS               : 1;
__REG32 VIE               : 1;
__REG32 VIS               : 1;
__REG32                   :17;
__REG32 SWGR              : 1;
} __pmcr_bits;

/* Power Management Unit Status Register (PSR) */
typedef struct{
__REG32 SS2S              : 1;
__REG32 SS3S              : 1;
__REG32 BFS               : 1;
__REG32                   : 9;
__REG32 TSS               : 3;
__REG32                   :16;
__REG32 PTS               : 1;
} __psr_bits;

/* Power Management Unit General Configuration register (PCFR) */
typedef struct{
__REG32 GP_ROD            : 1;
__REG32 SL_ROD            : 1;
__REG32 PUDH1             : 1;
__REG32                   : 5;
__REG32 SWDD              : 1;
__REG32                   : 3;
__REG32 L0_EN             : 1;
__REG32 L1_DIS            : 1;
__REG32                   : 6;
__REG32 LPM_DEL           : 4;
__REG32 PWR_DEL           : 4;
__REG32 SYS_DEL           : 4;
} __pcfr_bits;

/* Power Manager Wake-Up Enable Register (PWER) */
typedef struct{
__REG32 WER0              : 1;
__REG32 WER1              : 1;
__REG32 WEF0              : 1;
__REG32 WEF1              : 1;
__REG32                   :27;
__REG32 WERTC             : 1;
} __pwer_bits;

/* Power Manager Wake-Up Status Register (PWSR) */
typedef struct{
__REG32 EDR0              : 1;
__REG32 EDR1              : 1;
__REG32 EDF0              : 1;
__REG32 EDF1              : 1;
__REG32                   :27;
__REG32 EERTC             : 1;
} __pwsr_bits;

/* Power Manager EXT_WAKEUP<1:0> Control Register (PECR) */
typedef struct{
__REG32 IN0               : 1;
__REG32 IN1               : 1;
__REG32                   :26;
__REG32 E0IE              : 1;
__REG32 E0IS              : 1;
__REG32 E1IE              : 1;
__REG32 E1IS              : 1;
} __pecr_bits;

/* Power Management Unit Voltage Change Control Register (PVCR) */
typedef struct{
__REG32 SA                : 7;
__REG32                   : 7;
__REG32 VCSA              : 1;
__REG32                   :14;
__REG32 TVE               : 1;
__REG32 PVE               : 1;
__REG32 FVE               : 1;
} __pvcr_bits;

/* Application Subsystem Power Status/Configuration Register (ASCR) */
typedef struct{
__REG32 D3S               : 1;
__REG32 D2S               : 1;
__REG32 D1S               : 1;
__REG32                   : 5;
__REG32 MTS_S             : 3;
__REG32                   : 1;
__REG32 MTS               : 3;
__REG32                   :16;
__REG32 RDH               : 1;
} __ascr_bits;

/* Application Subsystem Reset Status Register (ARSR) */
typedef struct{
__REG32 HWR               : 1;
__REG32 WDT               : 1;
__REG32 LPMR              : 1;
__REG32 GPR               : 1;
__REG32                   :28;
} __arsr_bits;

/* Application Subsystem Wake-Up from D3 Enable Register (AD3ER) 
   Application Subsystem Wake-Up from D2 to D0 State Enable Register (AD2D0ER)
   Application Subsystem Wake-Up from D1 to D0 State Enable Register (AD1D0ER) */
typedef struct{
__REG32 WE_EXTERNAL0      : 1;
__REG32 WE_EXTERNAL1      : 1;
__REG32 WE_GENERIC0       : 1;
__REG32 WE_GENERIC1       : 1;
__REG32 WE_GENERIC2       : 1;
__REG32 WE_GENERIC3       : 1;
__REG32 WE_GENERIC4       : 1;
__REG32 WE_GENERIC5       : 1;
__REG32 WE_GENERIC6       : 1;
__REG32 WE_GENERIC7       : 1;
__REG32 WE_GENERIC8       : 1;
__REG32 WE_GENERIC9       : 1;
__REG32 WE_GENERIC10      : 1;
__REG32 WE_GENERIC11      : 1;
__REG32 WE_GENERIC12      : 1;
__REG32 WE_GENERIC13      : 1;
__REG32 WE_OTG            : 1;
__REG32                   : 2;
__REG32 WEUSIM0           : 1;
__REG32 WEUSIM1           : 1;
__REG32 WEKP              : 1;
__REG32 WEDMUX2           : 1;
__REG32 WEDMUX3           : 1;
__REG32 WEMSL0            : 1;
__REG32                   : 1;
__REG32 WEUSB2            : 1;
__REG32                   : 1;
__REG32 WEUSBH            : 1;
__REG32 WETSI             : 1;
__REG32 WEOST             : 1;
__REG32 WERTC             : 1;
} __ad3er_bits;

/* Application Subsystem Wake-Up from D3 Status Register (AD3SR)
   Application Subsystem Wake-Up from D2 to D0 Status Register (AD2D0SR)
   Application Subsystem Wake-Up from D1 to D0 Status Register (AD1D0SR) */
typedef struct{
__REG32 WS_EXTERNAL0      : 1;
__REG32 WS_EXTERNAL1      : 1;
__REG32 WS_GENERIC0       : 1;
__REG32 WS_GENERIC1       : 1;
__REG32 WS_GENERIC2       : 1;
__REG32 WS_GENERIC3       : 1;
__REG32 WS_GENERIC4       : 1;
__REG32 WS_GENERIC5       : 1;
__REG32 WS_GENERIC6       : 1;
__REG32 WS_GENERIC7       : 1;
__REG32 WS_GENERIC8       : 1;
__REG32 WS_GENERIC9       : 1;
__REG32 WS_GENERIC10      : 1;
__REG32 WS_GENERIC11      : 1;
__REG32 WS_GENERIC12      : 1;
__REG32 WS_GENERIC13      : 1;
__REG32 WS_OTG            : 1;
__REG32                   : 2;
__REG32 WSUSIM0           : 1;
__REG32 WSUSIM1           : 1;
__REG32 WSKP              : 1;
__REG32 WSDMUX2           : 1;
__REG32 WSDMUX3           : 1;
__REG32 WSMSL0            : 1;
__REG32                   : 1;
__REG32 WSUSB2            : 1;
__REG32                   : 1;
__REG32 WSUSBH            : 1;
__REG32 WSTSI             : 1;
__REG32 WSOST             : 1;
__REG32 WSRTC             : 1;
} __ad3sr_bits;

/* Application Subsystem Wake-Up from D2 to D1 State Enable Register (AD2D1ER) */
typedef struct{
__REG32                   :31;
__REG32 WERTC             : 1;
} __ad2d1er_bits;

/* Application Subsystem Wake-Up from D2 to D1 Status Register (AD2D1SR) */
typedef struct{
__REG32                   :31;
__REG32 WSRTC             : 1;
} __ad2d1sr_bits;

/* Application Subsystem D3 Configuration Register (AD3R) */
typedef struct{
__REG32 AD3_R0            : 1;
__REG32 AD3_R1            : 1;
__REG32 AD3_R2            : 1;
__REG32 AD3_R3            : 1;
__REG32 AD3_R4            : 1;
__REG32 AD3_R5            : 1;
__REG32                   :26;
} __ad3r_bits;

/* Application Subsystem D2 Configuration Register (AD2R) */
typedef struct{
__REG32 AD2_R0            : 1;
__REG32 AD2_R1            : 1;
__REG32 AD2_R2            : 1;
__REG32 AD2_R3            : 1;
__REG32 AD2_R4            : 1;
__REG32 AD2_R5            : 1;
__REG32                   :26;
} __ad2r_bits;

/* Application Subsystem D1 Configuration Register (AD1R) */
typedef struct{
__REG32 AD1_R0            : 1;
__REG32 AD1_R1            : 1;
__REG32 AD1_R2            : 1;
__REG32 AD1_R3            : 1;
__REG32 AD1_R4            : 1;
__REG32 AD1_R5            : 1;
__REG32                   :26;
} __ad1r_bits;

/* 1-Wire Command Register (W1CMDR) */
typedef struct{
__REG32 _1WR              : 1;
__REG32 SRA               : 1;
__REG32 DQO               : 1;
__REG32 DQI               : 1;
__REG32                   :28;
} __w1cmdr_bits;

/* 1-Wire Transmit/Receive Buffer (W1TRR) */
typedef struct{
__REG32 DATA              : 8;
__REG32                   :24;
} __w1trr_bits;

/* 1-Wire Interrupt Register (W1INTR) */
typedef struct{
__REG32 PD                : 1;
__REG32 PDR               : 1;
__REG32 TBE               : 1;
__REG32 TEMP              : 1;
__REG32 RBF               : 1;
__REG32                   :27;
} __w1intr_bits;

/* 1-Wire Interrupt Enable Register (W1IER) */
typedef struct{
__REG32 EPD               : 1;
__REG32                   : 1;
__REG32 ETBE              : 1;
__REG32 ETMT              : 1;
__REG32 ERBF              : 1;
__REG32                   : 2;
__REG32 DQOE              : 1;
__REG32                   :24;
} __w1ier_bits;

/* 1-Wire Clock Divisor Register (W1CDR) */
typedef struct{
__REG32 DIVISOR           : 5;
__REG32                   :27;
} __w1cdr_bits;

/* DMA Request to Channel Map Register (DRCMRx) */
typedef struct{
__REG32 CHLNUM            : 5;
__REG32                   : 2;
__REG32 MAPVLD            : 1;
__REG32                   :24;
} __drcmr_bits;

/* DMA Descriptor Address Registers (DDADRx) */
typedef struct{
__REG32 STOP              : 1;
__REG32 BREN              : 1;
__REG32                   : 2;
__REG32 DA                :28;
} __ddadr_bits;

/* DMA Command Registers (DCMDx) */
typedef struct{
__REG32 LEN               :13;
__REG32                   : 1;
__REG32 WIDTH             : 2;
__REG32 SIZE              : 2;
__REG32                   : 3;
__REG32 ENDIRQEN          : 1;
__REG32 STARTIRQEN        : 1;
__REG32 ADDRMODE          : 1;
__REG32                   : 1;
__REG32 CMPEN             : 1;
__REG32                   : 2;
__REG32 FLOWTRG           : 1;
__REG32 FLOWSRC           : 1;
__REG32 INCTRGADDR        : 1;
__REG32 INCSRCADDR        : 1;
} __dcmd_bits;

/* DREQ Status Register (DRQSR0) */
typedef struct{
__REG32 REQPEND           : 5;
__REG32                   : 3;
__REG32 CLR               : 1;
__REG32                   :23;
} __drqsr0_bits;

/* DMA Channel Control/Status Registers (DCSRx) */
typedef struct{
__REG32 BUSERRINTR        : 1;
__REG32 STARTINTR         : 1;
__REG32 ENDINTR           : 1;
__REG32 STOPINTR          : 1;
__REG32 RASINTR           : 1;
__REG32                   : 3;
__REG32 REQPEND           : 1;
__REG32 EORINT            : 1;
__REG32 CMPST             : 1;
__REG32                   :11;
__REG32 MASKRUN           : 1;
__REG32 RASIRQEN          : 1;
__REG32 CLRCMPST          : 1;
__REG32 SETCMPST          : 1;
__REG32 EORSTOPEN         : 1;
__REG32 EORJMPEN          : 1;
__REG32 EORIRQEN          : 1;
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
__REG32 CHLINTR16         : 1;
__REG32 CHLINTR17         : 1;
__REG32 CHLINTR18         : 1;
__REG32 CHLINTR19         : 1;
__REG32 CHLINTR20         : 1;
__REG32 CHLINTR21         : 1;
__REG32 CHLINTR22         : 1;
__REG32 CHLINTR23         : 1;
__REG32 CHLINTR24         : 1;
__REG32 CHLINTR25         : 1;
__REG32 CHLINTR26         : 1;
__REG32 CHLINTR27         : 1;
__REG32 CHLINTR28         : 1;
__REG32 CHLINTR29         : 1;
__REG32 CHLINTR30         : 1;
__REG32 CHLINTR31         : 1;
} __dint_bits;

/* DMA Alignment Register (DALGN) */
typedef struct{
__REG32 DALGN0            : 1;
__REG32 DALGN1            : 1;
__REG32 DALGN2            : 1;
__REG32 DALGN3            : 1;
__REG32 DALGN4            : 1;
__REG32 DALGN5            : 1;
__REG32 DALGN6            : 1;
__REG32 DALGN7            : 1;
__REG32 DALGN8            : 1;
__REG32 DALGN9            : 1;
__REG32 DALGN10           : 1;
__REG32 DALGN11           : 1;
__REG32 DALGN12           : 1;
__REG32 DALGN13           : 1;
__REG32 DALGN14           : 1;
__REG32 DALGN15           : 1;
__REG32 DALGN16           : 1;
__REG32 DALGN17           : 1;
__REG32 DALGN18           : 1;
__REG32 DALGN19           : 1;
__REG32 DALGN20           : 1;
__REG32 DALGN21           : 1;
__REG32 DALGN22           : 1;
__REG32 DALGN23           : 1;
__REG32 DALGN24           : 1;
__REG32 DALGN25           : 1;
__REG32 DALGN26           : 1;
__REG32 DALGN27           : 1;
__REG32 DALGN28           : 1;
__REG32 DALGN29           : 1;
__REG32 DALGN30           : 1;
__REG32 DALGN31           : 1;
} __dalgn_bits;

/* DMA Programmed I/O Control Status Register (DPCSR) */
typedef struct{
__REG32 BRGBUSY           : 1;
__REG32                   :30;
__REG32 BRGSPLIT          : 1;
} __dpcsr_bits;

/* Interrupt Controller Pending Register (ICPR)
   Interrupt Controller IRQ Pending Registers (ICIP)
   Interrupt Controller FIQ Pending Registers (ICFP)
   Interrupt Controller Mask Registers (ICMR)
   Interrupt Controller Level Registers (ICLR) */
typedef struct{
__REG32 SSP3              : 1;
__REG32 MSL1              : 1;
__REG32 USBH2             : 1;
__REG32 USBH1             : 1;
__REG32 KEYPAD            : 1;
__REG32                   : 1;
__REG32 PWR_I2C           : 1;
__REG32 OST_4_11          : 1;
__REG32 GPIO_0            : 1;
__REG32 GPIO_1            : 1;
__REG32 GPIO_X            : 1;
__REG32 USBC              : 1;
__REG32 PML               : 1;
__REG32 SSP4              : 1;
__REG32 AC97              : 1;
__REG32 USIM1             : 1;
__REG32 SSP2              : 1;
__REG32 LCD               : 1;
__REG32 I2C               : 1;
__REG32                   : 1;
__REG32 UART3             : 1;
__REG32 UART2             : 1;
__REG32 UART1             : 1;
__REG32 MMC1              : 1;
__REG32 SSP1              : 1;
__REG32 DMAC              : 1;
__REG32 OST_0             : 1;
__REG32 OST_1             : 1;
__REG32 OST_2             : 1;
__REG32 OST_3             : 1;
__REG32 RTC_HZ            : 1;
__REG32 RTC_AL            : 1;
} __icpr_bits;

/* Interrupt Controller Pending Register (ICPR2)
   Interrupt Controller IRQ Pending Registers (ICIP2)
   Interrupt Controller FIQ Pending Registers (ICFP2)
   Interrupt Controller Mask Registers (ICMR2)
   Interrupt Controller Level Registers (ICLR2) */
typedef struct{
__REG32                   : 1;
__REG32 CIF               : 1;
__REG32 CONSUMERIR        : 1;
__REG32                   : 1;
__REG32 TSI               : 1;
__REG32                   : 1;
__REG32 USIM2             : 1;
__REG32 GRAPHICS          : 1;
__REG32                   : 1;
__REG32 MMC2              : 1;
__REG32                   : 2;
__REG32 ONE_WIRE          : 1;
__REG32 NAND_INF          : 1;
__REG32 USB2              : 1;
__REG32 SGP_MPMU          : 1;
__REG32                   : 1;
__REG32 WAKEUP0           : 1;
__REG32 WAKEUP1           : 1;
__REG32 DMEMC             : 1;
__REG32 BCCU              : 1;
__REG32                   :11;
} __icpr2_bits;

/* Interrupt Controller Control Register (ICCR) */
typedef struct{
__REG32 DIM               : 1;
__REG32                   :31;
} __iccr_bits;

/* Interrupt Controller Interrupt Priority Registers (IRP0 to IRP52) */
typedef struct{
__REG32 PID               : 6;
__REG32                   :25;
__REG32 VAL               : 1;
} __irp_bits;

/* Interrupt Control Highest Priority Register (ICHP) */
typedef struct{
__REG32 FIQ               : 6;
__REG32                   : 9;
__REG32 VAL_FIQ           : 1;
__REG32 IRQ               : 6;
__REG32                   : 9;
__REG32 VAL_IRQ           : 1;
} __ichp_bits;

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
__REG32 RDAL1             : 1;
__REG32 RDALE1            : 1;
__REG32 RDAL2             : 1;
__REG32 RDALE2            : 1;
__REG32 SWAL1             : 1;
__REG32 SWALE1            : 1;
__REG32 SWAL2             : 1;
__REG32 SWALE2            : 1;
__REG32 SWCE              : 1;
__REG32 PIAL              : 1;
__REG32 PIALE             : 1;
__REG32 PICE              : 1;
__REG32                   :16;
} __rtsr_bits;

/* Wristwatch Day Alarm Registers (RDARx) */
typedef struct{
__REG32 SECONDS           : 6;
__REG32 MINUTES           : 6;
__REG32 HOURS             : 5;
__REG32 DOW               : 3;
__REG32 WOM               : 3;
__REG32                   : 9;
} __rdar_bits;

/* Wristwatch Year Alarm Registers (RYARx) */
typedef struct{
__REG32 DOM               : 5;
__REG32 MONTH             : 4;
__REG32 YEAR              :12;
__REG32                   :11;
} __ryar_bits;

/* Stopwatch Alarm Registers (SWARx) */
typedef struct{
__REG32 HUNDRETHS         : 7;
__REG32 SECONDS           : 6;
__REG32 MINUTES           : 6;
__REG32 HOURS             : 5;
__REG32                   : 8;
} __swar_bits;

/* Periodic Interrupt Alarm Register (PIAR) */
typedef struct{
__REG32 MILLISECONDS      :16;
__REG32                   :16;
} __piar_bits;

/* RTC Day Counter Register (RDCR) */
typedef struct{
__REG32 SECONDS           : 6;
__REG32 MINUTES           : 6;
__REG32 HOURS             : 5;
__REG32 DOW               : 3;
__REG32 WOM               : 3;
__REG32                   : 9;
} __rdcr_bits;

/* RTC Year Counter Register (RYCR) */
typedef struct{
__REG32 DOM               : 5;
__REG32 MONTH             : 4;
__REG32 YEAR              :12;
__REG32                   :11;
} __rycr_bits;

/* Stopwatch Counter Register (SWCR) */
typedef struct{
__REG32 HUNDRETHS         : 7;
__REG32 SECONDS           : 6;
__REG32 MINUTES           : 6;
__REG32 HOURS             : 5;
__REG32                   : 8;
} __swcr_bits;

/* Periodic Interrupt Counter Register (RTCPICR) */
typedef struct{
__REG32 MILLISECONDS      :16;
__REG32                   :16;
} __rtcpicr_bits;

/* OS Match Control Registers (OMCRx) */
typedef struct{
__REG32 CRES              : 3;
__REG32 R                 : 1;
__REG32 S                 : 2;
__REG32 P                 : 1;
__REG32 C                 : 1;
__REG32                   :24;
} __omcr_bits;

/* OS Match Control Registers (OMCR8 and OMCR10) */
typedef struct{
__REG32 CRES              : 3;
__REG32 R                 : 1;
__REG32 S                 : 2;
__REG32 P                 : 1;
__REG32 C                 : 1;
__REG32 CRES3             : 1;
__REG32                   :23;
} __omcr8_bits;

/* OS Match Control Registers (OMCR9 and OMCR11) */
typedef struct{
__REG32 CRES              : 3;
__REG32 R                 : 1;
__REG32 S                 : 2;
__REG32 P                 : 1;
__REG32 C                 : 1;
__REG32 CRES3             : 1;
__REG32 N                 : 1;
__REG32                   :22;
} __omcr9_bits;

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
__REG32 E4                : 1;
__REG32 E5                : 1;
__REG32 E6                : 1;
__REG32 E7                : 1;
__REG32 E8                : 1;
__REG32 E9                : 1;
__REG32 E10               : 1;
__REG32 E11               : 1;
__REG32                   :20;
} __oier_bits;

/* OS Timer Status Register (OSSR) */
typedef struct{
__REG32 M0                : 1;
__REG32 M1                : 1;
__REG32 M2                : 1;
__REG32 M3                : 1;
__REG32 M4                : 1;
__REG32 M5                : 1;
__REG32 M6                : 1;
__REG32 M7                : 1;
__REG32 M8                : 1;
__REG32 M9                : 1;
__REG32 M10               : 1;
__REG32 M11               : 1;
__REG32                   :20;
} __ossr_bits;

/* Event Select Registers (PML_ESL_(7-0)) */
typedef struct{
__REG32 EN                : 7;
__REG32                   :25;
} __pml_esel_bits;

/* Breakpoint Register (MDU_XSCALE_BP) 
   MDU 2DG Stop Register (MDU_2DG_EVENT)
   MDU CW Match Signal Register (MDU_CW_MATCH) */
typedef struct{
__REG32                   : 2;
__REG32 _2DG_DB_EVENT     : 1;
__REG32 _2DG_CS_EVENT     : 1;
__REG32                   : 3;
__REG32 XDB_EVENT         : 1;
__REG32 XCS_EVENT         : 1;
__REG32                   :23;
} __mdu_xscale_bp_bits;

/* System Bus Arbiter Control Registers (ARB_CNTRL_1) */
typedef struct{
__REG32 SWITCH_WT         : 4;
__REG32 DMA_WT            : 4;
__REG32 LCD_WT            : 4;
__REG32 CAMERA_WT         : 4;
__REG32                   : 6;
__REG32 LOCK_FLAG         : 1;
__REG32 SWITCH_PARK       : 1;
__REG32 DMA_PARK          : 1;
__REG32 LCD_PARK          : 1;
__REG32 CI_PARK           : 1;
__REG32 USBH_PARK         : 1;
__REG32 SWITCH_SLV_PARK   : 1;
__REG32 DMA_SLV_PARK      : 1;
__REG32                   : 2;
} __arb_cntr1_bits;

/* System Bus Arbiter Control Registers (ARB_CNTRL_2) */
typedef struct{
__REG32 SWITCH_WT         : 4;
__REG32 _2DG_WT           : 4;
__REG32 USB2_WT           : 4;
__REG32                   :13;
__REG32 LOCK_FLAG         : 1;
__REG32 SWITCH_PARK       : 1;
__REG32 _2DG_PARK         : 1;
__REG32 USB2_PARK         : 1;
__REG32 SWITCH_SLV_PARK   : 1;
__REG32                   : 1;
__REG32 USB2_SLV_PARK     : 1;
} __arb_cntr2_bits;


/* SDRAM Configuration Register (MDCNFG) */
typedef struct{
__REG32 DCSE              : 2;
__REG32 DBW               : 1;
__REG32 DCAC              : 2;
__REG32 DRAC              : 2;
__REG32                   : 1;
__REG32 DTC               : 2;
__REG32                   :20;
__REG32 DMCEN             : 1;
__REG32                   : 1;
} __mdcnfg_bits;

/* SDRAM Refresh Control Register */
typedef struct{
__REG32 DRI               : 8;
__REG32 SWREF             : 1;
__REG32                   :23;
} __mdrefr_bits;

/* SDRAM Mode Register Set Configuration Register (MDMRS) */
typedef struct{
__REG32 MDMRS             :14;
__REG32                   : 1;
__REG32 MDBA              : 2;
__REG32                   :11;
__REG32 MDPEND            : 1;
__REG32 MDCOND            : 1;
__REG32 MDCS0             : 1;
__REG32 MDCS1             : 1;
} __mdmrs_bits;

/* DDR Hardware Calibration Register (DDR_HCAL) */
typedef struct{
__REG32 HCRNG             : 5;
__REG32                   : 1;
__REG32 HCOFF0            : 4;
__REG32 HCOFF1            : 4;
__REG32 HCOFF2            : 4;
__REG32 HCOFF3            : 4;
__REG32                   : 2;
__REG32                   : 4;
__REG32 HCPROG            : 1;
__REG32                   : 2;
__REG32 HCEN              : 1;
} __ddr_hcal_bits;

/* DDR Write Strobe Calibration Register (DDR_WCAL) */
typedef struct{
__REG32                   : 8;
__REG32 WCOFF             : 4;
__REG32                   : 4;
__REG32 WSDLV_STATUS      : 7;
__REG32                   : 8;
__REG32 WCEN              : 1;
} __ddr_wcal_bits;

/* Dynamic Memory Controller Interrupt Enable Register (DMCIER) */
typedef struct{
__REG32                   :29;
__REG32 EDLP              : 1;
__REG32 EORF              : 1;
__REG32 ERCI              : 1;
} __dmcier_bits;

/* Dynamic Memory Controller Interrupt Status Register (DMCISR) */
typedef struct{
__REG32 ORV               : 7;
__REG32 SLFREF            : 1;
__REG32 PDV               : 7;
__REG32 NCODE             : 7;
__REG32 PCODE             : 7;
__REG32 DLP               : 1;
__REG32 ORF               : 1;
__REG32 RCI               : 1;
} __dmcisr_bits;

/* Delay Line Status Register (DDR_DLS) */
typedef struct{
__REG32 SSDLV0            : 7;
__REG32                   : 1;
__REG32 SSDLV1            : 7;
__REG32                   : 1;
__REG32 SSDLV2            : 7;
__REG32                   : 1;
__REG32 SSDLV3            : 7;
__REG32                   : 1;
} __ddr_dls_bits;

/* External Memory Pin Interface Control Register (EMPI) */
typedef struct{
__REG32                   :28;
__REG32 SCHM_DMEM_EN      : 1;
__REG32 SCHM_CMD          : 1;
__REG32 PW_DQN            : 1;
__REG32 PD_DQS            : 1;
} __empi_bits;

/* Rcomp Control Register (RCOMP) */
typedef struct{
__REG32 REI               :20;
__REG32                   : 5;
__REG32 RCRNG             : 5;
__REG32 UPDATE            : 1;
__REG32 SWEVAL            : 1;
} __rcomp_bits;

/* PAD_MA Strength and Slew Settings Register (PAD_MA) */
/* PAD_MDMSB Strength and Slew Settings Register (PAD_MDMSB) */
/* PAD_MDLSB Strength and Slew Settings Register (PAD_MDLSB) */
/* PAD_SDRAM Strength and Slew Settings Register (PAD_SDRAM) */
/* PAD_SDCLK Strength and Slew Settings Register (PAD_SDCLK) */
/* PAD_SDCS Strength and Slew Settings Register (PAD_SDCS) */
/* PAD_SCLK Strength and Slew Settings Register (PAD_SCKL) */
typedef struct{
__REG32 NSLEW             : 4;
__REG32                   : 4;
__REG32 PSLEW             : 4;
__REG32                   : 4;
__REG32 NCODE             : 7;
__REG32                   : 1;
__REG32 PCODE             : 7;
__REG32                   : 1;
} __pad_ma_bits;

/* Static Memory Control Registers (MSC1) */
typedef struct{
__REG32 RT2               : 3;
__REG32                   : 1;
__REG32 RDF2              : 4;
__REG32 RDN2              : 4;
__REG32                   : 4;
__REG32 RT3               : 3;
__REG32                   : 1;
__REG32 RDF3              : 4;
__REG32 RDN3              : 4;
__REG32                   : 4;
} __msc1_bits;

/* Expansion Memory Configuration Register (MECR) */
typedef struct{
__REG32                   : 1;
__REG32 CIT               : 1;
__REG32                   :30;
} __mecr_bits;

/* Synchronous Static Memory Control Register (SXCNFG) */
typedef struct{
__REG32 SXEN0             : 2;
__REG32 SXCL0             : 3;
__REG32                   :10;
__REG32 SXCLEXT0          : 1;
__REG32 SXEN2             : 2;
__REG32 SXCL2             : 3;
__REG32 SXWRCL2           : 4;
__REG32                   : 6;
__REG32 SXCLEXT2          : 1;
} __sxcnfg_bits;

/* Expansion Memory Timing Configuration Register (MC<space>x) */
/* MCMEM0 */
/* MCATT0 */
/* MCIO0 */
typedef struct{
__REG32 SET               : 7;
__REG32 ASST              : 5;
__REG32                   : 2;
__REG32 HOLD              : 6;
__REG32                   :12;
} __mc_x_bits;

/* Clock Configuration Register (MEMCLKCFG) */
typedef struct{
__REG32                   :16;
__REG32 DF_CLKDIV         : 3;
__REG32                   :13;
} __memclkcfg_bits;

/* Address Configuration Registers (CSADRCFGx) */
typedef struct{
__REG32 INFTYPE           : 4;
__REG32 ADDRBASE          : 2;
__REG32                   : 2;
__REG32 ADDRSPLIT         : 4;
__REG32                   : 2;
__REG32 ADDRCONFIG        : 3;
__REG32 ALW               : 3;
__REG32 ALT               : 2;
__REG32                   :10;
} __csadrcfgx_bits;

/* Data Flash Control Register (NDCR) */
typedef struct{
__REG32 WRCMDREQM         : 1;
__REG32 RDDREQM           : 1;
__REG32 WRDREQM           : 1;
__REG32 SBERRM            : 1;
__REG32 DBERRM            : 1;
__REG32 CS1_BBDM          : 1;
__REG32 CS0_BBDM          : 1;
__REG32 CS1_CMDDM         : 1;
__REG32 CS0_CMDDM         : 1;
__REG32 CS1_PAGEDM        : 1;
__REG32 CS0_PAGEDM        : 1;
__REG32 RDYM              : 1;
__REG32 ND_ARB_EN         : 1;
__REG32                   : 1;
__REG32 PG_PER_BLK        : 1;
__REG32 RA_START          : 1;
__REG32 RD_ID_CNT         : 3;
__REG32 CLR_ECC           : 1;
__REG32 CLR_PG_CNT        : 1;
__REG32 ND_MODE           : 2;
__REG32 NCSX              : 1;
__REG32 PAGE_SZ           : 2;
__REG32 DWIDTH_M          : 1;
__REG32 DWIDTH_C          : 1;
__REG32 ND_RUN            : 1;
__REG32 DMA_EN            : 1;
__REG32 ECC_EN            : 1;
__REG32 SPARE_EN          : 1;
} __ndcr_bits;

/* NAND Interface Timing Parameter 0 Register (NDTR0CS0) */
typedef struct{
__REG32 tRP               : 3;
__REG32 tRH               : 3;
__REG32                   : 2;
__REG32 tWP               : 3;
__REG32 tWH               : 3;
__REG32                   : 2;
__REG32 tCS               : 3;
__REG32 tCH               : 3;
__REG32                   :10;
} __ndtr0cs0_bits;

/* NAND Interface Timing Parameter 1 Register (NDTR1CS0) */
typedef struct{
__REG32 tAR               : 4;
__REG32 tWHR              : 4;
__REG32                   : 8;
__REG32 tR                :16;
} __ndtr1cs0_bits;

/* NAND Controller Status Register (NDSR) */
typedef struct{
__REG32 WRCMDREQ          : 1;
__REG32 RDDREQ            : 1;
__REG32 WRDREQ            : 1;
__REG32 SBERR             : 1;
__REG32 DBERR             : 1;
__REG32 CS1_BBD           : 1;
__REG32 CS0_BBD           : 1;
__REG32 CS1_CMDD          : 1;
__REG32 CS0_CMDD          : 1;
__REG32 CS1_PAGED         : 1;
__REG32 CS0_PAGED         : 1;
__REG32 RDY               : 1;
__REG32                   :20;
} __ndsr_bits;

/* NAND Controller Page Count Register (NDPCR) */
typedef struct{
__REG32 PG_CNT_0          : 6;
__REG32                   :10;
__REG32 PG_CNT_1          : 6;
__REG32                   :10;
} __ndpcr_bits;

/* NAND Controller Command Buffer 0 (NDCB0) */
typedef struct{
__REG32 CMD1              : 8;
__REG32 CMD2              : 8;
__REG32 ADDR_CYC          : 3;
__REG32 DBC               : 1;
__REG32 NC                : 1;
__REG32 CMD_TYPE          : 3;
__REG32 CSEL              : 1;
__REG32 AUTO_RS           : 1;
__REG32                   : 6;
} __ndcb0_bits;

/* NAND Controller Command Buffer 1 (NDCB1) */
typedef struct{
__REG32  ADDR1            : 8;
__REG32  ADDR2            : 8;
__REG32  ADDR3            : 8;
__REG32  ADDR4            : 8;
} __ndcb1_bits;

/* NAND Controller Command Buffer 2 (NDCB2) */
typedef struct{
__REG32  ADDR5            : 8;
__REG32  PAGE_COUNT       : 6;
__REG32  ADDR3            :18;
} __ndcb2_bits;

/* IM Power Management Control Register (IMPMCR) */
typedef struct{
__REG32  DDT              : 8;
__REG32                   : 1;
__REG32  AW1              : 1;
__REG32  AW2              : 1;
__REG32  AW3              : 1;
__REG32  AW4              : 1;
__REG32  AW5              : 1;
__REG32                   :18;
} __impmcr_bits;

/* MMC Clock Start/Stop Register (MMC_STRPCL) */
typedef struct{
__REG32  STOP_CLK         : 1;
__REG32  STRT_CLK         : 1;
__REG32                   :30;
} __mmc_strpcl_bits;

/* MMC Status Register (MMC_STAT) */
typedef struct{
__REG32 TIME_OUT_READ     : 1;
__REG32 TIME_OUT_RES      : 1;
__REG32 CRC_WR_ERR        : 1;
__REG32 CRC_RD_ERR        : 1;
__REG32 DAT_ERR_TOKEN     : 1;
__REG32 RES_CRC_ERR       : 1;
__REG32                   : 2;
__REG32 CLK_EN            : 1;
__REG32 FLASH_ERR         : 1;
__REG32 SPI_WR_ERR        : 1;
__REG32 DATA_TRAN_DONE    : 1;
__REG32 PRG_DONE          : 1;
__REG32 END_CMD_RES       : 1;
__REG32 RD_STALLED        : 1;
__REG32 SDIO_INT          : 1;
__REG32 SDIO_SUSPEND_ACK  : 1;
__REG32                   :15;
} __mmc_stat_bits;

/* MMC Clock Rate Register (MMC_CLKRT) */
typedef struct{
__REG32 CLK_RATE          : 3;
__REG32                   :29;
} __mmc_clkrt_bits;

/* MMC_SPI Register (MMC_SPI) */
typedef struct{
__REG32 SPI_MODE          : 1;
__REG32 SPI_CRC_EN        : 1;
__REG32 SPI_CS_EN         : 1;
__REG32 SPI_CS_ADDRESS    : 1;
__REG32                   :28;
} __mmc_spi_bits;

/* MMC_CMDAT Register (MMC_CMDAT) */
typedef struct{
__REG32 RES_TYPE          : 2;
__REG32 DATA_EN           : 1;
__REG32 WR_RD             : 1;
__REG32 STRM_BLK          : 1;
__REG32 BUSY              : 1;
__REG32 INIT              : 1;
__REG32 DMA_EN            : 1;
__REG32 SD_4DAT           : 1;
__REG32                   : 1;
__REG32 STOP_TRAN         : 1;
__REG32 SDIO_INT_EN       : 1;
__REG32 SDIO_SUSPEND      : 1;
__REG32 SDIO_RESUME       : 1;
__REG32                   :18;
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
__REG32 BLK_LEN           :12;
__REG32                   :20;
} __mmc_blklen_bits;

/* MMC_NUMBLK Register (MMC_NUMBLK) */
typedef struct{
__REG32 NUM_BLK           :16;
__REG32                   :16;
} __mmc_numblk_bits;

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
__REG32 MMC_RXFIFO_RD_REQ : 1;
__REG32 MMC_TXFIFO_WR_REQ : 1;
__REG32 TINT              : 1;
__REG32 DAT_ERR           : 1;
__REG32 RES_ERR           : 1;
__REG32 RD_STALLED        : 1;
__REG32 SDIO_INT          : 1;
__REG32 SDIO_SUSPEND_ACK  : 1;
__REG32                   :19;
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

/* MMC READ_WAIT Register (MMC_RDWAIT) */
typedef struct{
__REG32 RD_WAIT_EN        : 1;
__REG32 RD_WAIT_START     : 1;
__REG32                   :30;
} __mmc_rdwait_bits;

/* MMC BLOCKS REMAINING Register (MMC_BLKS_REM) */
typedef struct{
__REG32 BLKS_REM          :16;
__REG32                   :16;
} __mmc_blks_rem_bits;

/* LCD Controller Control Register 0 (LCCR0) */
typedef struct{
__REG32 ENB               : 1;
__REG32                   : 2;
__REG32 LDM               : 1;
__REG32 SOFM0             : 1;
__REG32 IUM               : 1;
__REG32 EOFM0             : 1;
__REG32 PAS               : 1;
__REG32                   : 2;
__REG32 DIS               : 1;
__REG32 QDM               : 1;
__REG32 PDD               : 8;
__REG32 BSM0              : 1;
__REG32 OUM               : 1;
__REG32 LCDT              : 1;
__REG32 RDSTM             : 1;
__REG32 CMDIM             : 1;
__REG32                   : 1;
__REG32 LDDALT            : 1;
__REG32 DELAY_LBIAS       : 1;
__REG32                   : 4;
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
__REG32                   : 1;
__REG32 BPP3              : 1;
__REG32 PDFOR             : 2;
} __lccr3_bits;

/* LCD Controller Control Register 4 (LCCR4) */
typedef struct{
__REG32 K1                : 3;
__REG32 K2                : 3;
__REG32 K3                : 3;
__REG32 REOFM0            : 1;
__REG32 REOFM1            : 1;
__REG32 REOFM2            : 1;
__REG32 REOFM3            : 1;
__REG32 REOFM4            : 1;
__REG32 REOFM5            : 1;
__REG32 PAL_FOR           : 2;
__REG32                   : 9;
__REG32 REOFM6            : 1;
__REG32                   : 4;
__REG32 PCDDIV            : 1;
} __lccr4_bits;

/* LCD Controller Control Register 5 (LCCR5) */
typedef struct{
__REG32 SOFM1             : 1;
__REG32 SOFM2             : 1;
__REG32 SOFM3             : 1;
__REG32 SOFM4             : 1;
__REG32 SOFM5             : 1;
__REG32 SOFM6             : 1;
__REG32                   : 2;
__REG32 EOFM1             : 1;
__REG32 EOFM2             : 1;
__REG32 EOFM3             : 1;
__REG32 EOFM4             : 1;
__REG32 EOFM5             : 1;
__REG32 EOFM6             : 1;
__REG32                   : 2;
__REG32 BSM1              : 1;
__REG32 BSM2              : 1;
__REG32 BSM3              : 1;
__REG32 BSM4              : 1;
__REG32 BSM5              : 1;
__REG32 BSM6              : 1;
__REG32                   : 2;
__REG32 IUM1              : 1;
__REG32 IUM2              : 1;
__REG32 IUM3              : 1;
__REG32 IUM4              : 1;
__REG32 IUM5              : 1;
__REG32 IUM6              : 1;
__REG32                   : 2;
} __lccr5_bits;

/* LCD Controller Control Register 6 (LCCR6) */
typedef struct{
__REG32 B_BLUE            : 8;
__REG32 B_GREEN           : 8;
__REG32 B_RED             : 8;
__REG32                   : 7;
__REG32 BF_OFF            : 1;
} __lccr6_bits;

/* LCD Overlay 1 Control Register 1 (OVL1C1) */
typedef struct{
__REG32 PPL1              :10;
__REG32 LPO1              :10;
__REG32 BPP1              : 4;
__REG32                   : 7;
__REG32 O1EN              : 1;
} __ovl1c1_bits;

/* LCD Overlay 1 Control Register 2 (OVL1C2) */
typedef struct{
__REG32 O1XPOS            :10;
__REG32 O1YPOS            :10;
__REG32                   :12;
} __ovl1c2_bits;

/* LCD Overlay 2 Control Register 1 (OVL2C1) */
typedef struct{
__REG32 PPL2              :10;
__REG32 LPO2              :10;
__REG32 BPP2              : 4;
__REG32                   : 7;
__REG32 O2EN              : 1;
} __ovl2c1_bits;

/* LCD Overlay 2 Control Register 2 (OVL1C2) */
typedef struct{
__REG32 O2XPOS            :10;
__REG32 O2YPOS            :10;
__REG32 _FOR              : 3;
__REG32                   : 9;
} __ovl2c2_bits;

/* LCD Cursor Control Register (CCR) */
typedef struct{
__REG32 CURMS             : 3;
__REG32                   : 2;
__REG32 CXPOS             :10;
__REG32 CYPOS             :10;
__REG32                   : 6;
__REG32 CEN               : 1;
} __ccr_bits;

/* LCD Command Control Register (CMDCR) */
typedef struct{
__REG32 SYNC_CNT          : 8;
__REG32                   :24;
} __cmdcr_bits;

/* LCD TMED RGB Seed Register (TRGBR) */
typedef struct{
__REG32 TRS               : 8;
__REG32 TGS               : 8;
__REG32 TBS               : 8;
__REG32                   : 8;
} __trgbr_bits;

/* LCD TMED Control Register (TCR) */
typedef struct{
__REG32 TM2S              : 1;
__REG32 TM1S              : 1;
__REG32 TM2EN             : 1;
__REG32 TM1EN             : 1;
__REG32 TVBS              : 4;
__REG32 THBS              : 4;
__REG32 TSCS              : 2;
__REG32 TED               : 1;
__REG32                   :17;
} __tcr_bits;

/* LCD DMA Frame Branch Registers (FBRx) */
typedef struct{
__REG32 BRA               : 1;
__REG32 BINT              : 1;
__REG32                   : 2;
__REG32 SRCADDR           :28;
} __fbr_bits;

/* LCD Panel Read Status Register (PRSR) */
typedef struct{
__REG32 DATA              : 8;
__REG32 A0                : 1;
__REG32 ST_OK             : 1;
__REG32 CON_NT            : 1;
__REG32                   :21;
} __prsr_bits;

/* LCD Controller Status Register 0 (LCSR0) */
typedef struct{
__REG32 LDD               : 1;
__REG32 SOF0              : 1;
__REG32 BER               : 1;
__REG32 ABC               : 1;
__REG32 IU0               : 1;
__REG32 IU1               : 1;
__REG32 OU                : 1;
__REG32 QD                : 1;
__REG32 EOF0              : 1;
__REG32 BS0               : 1;
__REG32 SINT              : 1;
__REG32 RD_ST             : 1;
__REG32 CMD_INT           : 1;
__REG32 REOF0             : 1;
__REG32 REOF1             : 1;
__REG32 REOF2             : 1;
__REG32 REOF3             : 1;
__REG32 REOF4             : 1;
__REG32 REOF5             : 1;
__REG32 REOF6             : 1;
__REG32                   : 8;
__REG32 BER_CH            : 3;
__REG32                   : 1;
} __lcsr0_bits;

/* LCD Controller Status Register 1 (LCSR1) */
typedef struct{
__REG32 SOF1              : 1;
__REG32 SOF2              : 1;
__REG32 SOF3              : 1;
__REG32 SOF4              : 1;
__REG32 SOF5              : 1;
__REG32 SOF6              : 1;
__REG32                   : 2;
__REG32 EOF1              : 1;
__REG32 EOF2              : 1;
__REG32 EOF3              : 1;
__REG32 EOF4              : 1;
__REG32 EOF5              : 1;
__REG32 EOF6              : 1;
__REG32                   : 2;
__REG32 BS1               : 1;
__REG32 BS2               : 1;
__REG32 BS3               : 1;
__REG32 BS4               : 1;
__REG32 BS5               : 1;
__REG32 BS6               : 1;
__REG32                   : 3;
__REG32 IU2               : 1;
__REG32 IU3               : 1;
__REG32 IU4               : 1;
__REG32 IU5               : 1;
__REG32 IU6               : 1;
__REG32                   : 2;
} __lcsr1_bits;

/* LCD DMA Command Register (LDCMDx) */
typedef struct{
__REG32                   : 2;
__REG32 LENGTH            :19;
__REG32 EOFINT            : 1;
__REG32 SOFINT            : 1;
__REG32 LSTDES_EN         : 2;
__REG32                   : 1;
__REG32 PAL               : 1;
__REG32                   : 5;
} __ldcmd_bits;

/* Mini-LCD Controller Control Register 0 (MLCCR0) */
typedef struct{
__REG32 PCD               : 8;
__REG32 HSP               : 1;
__REG32 VSP               : 1;
__REG32 PCP               : 1;
__REG32 OEP               : 1;
__REG32                   :20;
} __mlccr0_bits;

/* Mini-LCD Controller Control Register 1 (MLCCR1) */
typedef struct{
__REG32 PPL               :10;
__REG32 HSW               : 6;
__REG32 ELW               : 8;
__REG32 BLW               : 8;
} __mlccr1_bits;

/* Mini-LCD Controller Control Register 2 (MLCCR2) */
typedef struct{
__REG32 LPP               :10;
__REG32 VSW               : 6;
__REG32 EFW               : 8;
__REG32 BFW               : 8;
} __mlccr2_bits;

/* Mini-LCD Frame Count Register (MLFRMCNT) */
typedef struct{
__REG32 FRCOUNT           :18;
__REG32                   :12;
__REG32 FWKUP_EN          : 1;
__REG32 WKUP_EN           : 1;
} __mlfrmcnt_bits;

/* Quick Capture Interface Control Register 0 (CICR0) */
typedef struct{
__REG32 FOM               : 1;
__REG32 EOFM              : 1;
__REG32 SOFM              : 1;
__REG32 CDM               : 1;
__REG32 QDM               : 1;
__REG32                   : 1;
__REG32 EOLM              : 1;
__REG32                   : 2;
__REG32 TOM               : 1;
__REG32 FUM               : 1;
__REG32 BSM               : 1;
__REG32 EOFXM             : 1;
__REG32                   : 8;
__REG32 GCU_EN            : 1;
__REG32 LCD_MODE          : 2;
__REG32 SIM               : 3;
__REG32 DIS               : 1;
__REG32 ENB               : 1;
__REG32 SL_CAP_EN         : 1;
__REG32                   : 2;
} __cicr0_bits;

/* Quick Capture Interface Control Register 1 (CICR1) */
typedef struct{
__REG32 DW                : 3;
__REG32 COLOR_SP          : 2;
__REG32 RAW_BPP           : 2;
__REG32                   : 3;
__REG32 YCBCR_F           : 1;
__REG32                   : 4;
__REG32 PPL               :12;
__REG32                   : 5;
} __cicr1_bits;

/* Quick Capture Interface Control Register 2 (CICR2) */
typedef struct{
__REG32 FSW               : 3;
__REG32 BFPW              : 6;
__REG32                   : 1;
__REG32 HSW               : 6;
__REG32 ELW               : 8;
__REG32 BLW               : 8;
} __cicr2_bits;

/* Quick Capture Interface Control Register 3 (CICR3) */
typedef struct{
__REG32 LPF               :11;
__REG32 VSW               : 5;
__REG32 EFW               : 8;
__REG32 BFW               : 8;
} __cicr3_bits;

/* Quick Capture Interface Control Register 4 (CICR4) */
typedef struct{
__REG32 DIV               : 8;
__REG32 FR_RATE           : 3;
__REG32                   : 8;
__REG32 MCLK_EN           : 1;
__REG32 VSP               : 1;
__REG32 HSP               : 1;
__REG32 PCP               : 1;
__REG32 PCLK_EN           : 1;
__REG32                   : 8;
} __cicr4_bits;

/* Quick Capture Interface Status Register (CISR) */
typedef struct{
__REG32                   : 3;
__REG32 _EOF              : 1;
__REG32 _SOF              : 1;
__REG32 CDD               : 1;
__REG32 CQD               : 1;
__REG32                   : 1;
__REG32 EOL               : 1;
__REG32 HST_INT           : 1;
__REG32 CGC_INT           : 1;
__REG32                   : 4;
__REG32 FTO               : 1;
__REG32                   :14;
__REG32 EOFX              : 1;
__REG32 SINT              : 1;
} __cisr_bits;

/* Quick Capture Interface Return Clock Delay Register (CIRCD) */
typedef struct{
__REG32 CELL8_DLY         : 3;
__REG32 CELL7_DLY         : 3;
__REG32 CELL6_DLY         : 3;
__REG32 CELL5_DLY         : 3;
__REG32 CELL4_DLY         : 3;
__REG32 CELL3_DLY         : 3;
__REG32 CELL2_DLY         : 3;
__REG32 CELL1_DLY         : 3;
__REG32 DLYCELL_SEL       : 4;
__REG32 DATA_SEL          : 1;
__REG32 LV_SEL            : 1;
__REG32 FV_SEL            : 1;
__REG32 CLK_SEL           : 1;
} __circd_bits;

/* Quick Capture Interface Receive Buffer Register x (CIBRx) */
typedef struct{
__REG32 BYTE0             : 8;
__REG32 BYTE1             : 8;
__REG32 BYTE2             : 8;
__REG32 BYTE3             : 8;
} __cibr_bits;

/* Quick Capture Interface Pixel Substitution Status Register (CIPSS) */
typedef struct{
__REG32 PXLCNT            :12;
__REG32 LNCNT             :12;
__REG32 RAMADDR           : 7;
__REG32 PSU_EN            : 1;
} __cipss_bits;

/* Quick Capture Interface Pixel Substitution Buffer (CIPBUF) */
typedef struct{
__REG32 DEADROW           :11;
__REG32                   : 5;
__REG32 DEADCOL           :12;
__REG32                   : 4;
} __cipbuf_bits;

/* Quick Capture Interface Histogram Configuration (CIHST) */
typedef struct{
__REG32 COLORSEL          : 4;
__REG32 SCALE             : 2;
__REG32 CLR_RAM           : 1;
__REG32                   :25;
} __cihst_bits;

/* Quick Capture Interface Companding Configuration Register (CICCR) */
typedef struct{
__REG32 EN                : 1;
__REG32 SCALE             : 2;
__REG32 BLC_CLUT          : 8;
__REG32 BLC               : 8;
__REG32 LUT_LD            : 1;
__REG32                   :12;
} __ciccr_bits;

/* Quick Capture Interface Spatial Scaling Configuration Register (CISSC) */
typedef struct{
__REG32 SCALE             : 2;
__REG32                   :30;
} __cissc_bits;

/* Quick Capture Interface Color Management Register (CICMR) */
typedef struct{
__REG32 DMODE             : 2;
__REG32                   :30;
} __cicmr_bits;

/* Quick Capture Interface Color Management Coefficients Registers (CICMC0) */
typedef struct{
__REG32 COF02             :10;
__REG32 COF01             :10;
__REG32 COF00             :10;
__REG32                   : 2;
} __cicmc0_bits;

/* Quick Capture Interface Color Management Coefficients Registers (CICMC1) */
typedef struct{
__REG32 COF12             :10;
__REG32 COF11             :10;
__REG32 COF10             :10;
__REG32                   : 2;
} __cicmc1_bits;

/* Quick Capture Interface Color Management Coefficients Registers (CICMC2) */
typedef struct{
__REG32 COF22             :10;
__REG32 COF21             :10;
__REG32 COF20             :10;
__REG32                   : 2;
} __cicmc2_bits;

/* Quick Capture Interface FIFO Status Register (CIFSR) */
typedef struct{
__REG32 IFO_0             : 1;
__REG32 IFO_1             : 1;
__REG32 IFO_2             : 1;
__REG32 IFO_3             : 1;
__REG32                   : 6;
__REG32 EOF3              : 1;
__REG32                   : 3;
__REG32 SOF0              : 1;
__REG32 SOF1              : 1;
__REG32 SOF2              : 1;
__REG32 SOF3              : 1;
__REG32                   : 3;
__REG32 BS0               : 1;
__REG32 BS1               : 1;
__REG32 BS2               : 1;
__REG32 BS3               : 1;
__REG32                   : 3;
__REG32 IFU_3             : 1;
__REG32                   : 3;
} __cifsr_bits;

/* Quick Capture Interface FIFO Control Registers (CIFR0) */
typedef struct{
__REG32 FEN0              : 1;
__REG32 FEN1              : 1;
__REG32 FEN2              : 1;
__REG32 RESETF            : 1;
__REG32                   : 4;
__REG32 FLVL0             : 8;
__REG32 FLVL1             : 7;
__REG32 FLVL2             : 7;
__REG32                   : 2;
} __cifr0_bits;

/* Quick Capture Interface FIFO Control Registers (CIFR1) */
typedef struct{
__REG32 FEN3              : 1;
__REG32 FLVL3             : 8;
__REG32                   :23;
} __cifr1_bits;

/* Quick Capture Interface DMA Command Registers (CICMDx) */
typedef struct{
__REG32                   : 3;
__REG32 LENGTH            :18;
__REG32                   : 1;
__REG32 SOFINT            : 1;
__REG32                   : 7;
__REG32 INCTRGADDR        : 1;
__REG32                   : 1;
} __cicmd_bits;

/* Quick Capture Interface DMA Branch Registers (CIDBRx) */
typedef struct{
__REG32 BRA               : 1;
__REG32 BINT              : 1;
__REG32                   : 2;
__REG32 SRCADDR           :28;
} __cidbr_bits;

/* Quick Capture Interface DMA Channel Control/Status Register (CIDCSRx) */
typedef struct{
__REG32 BUSERRINTR        : 1;
__REG32                   : 2;
__REG32 STOPINTR          : 1;
__REG32                   : 4;
__REG32 REQPEND           : 1;
__REG32                   :20;
__REG32 STOPIRQEN         : 1;
__REG32                   : 2;
} __cidcsr_bits;

/* Graphics Controller Configuration Register (GCCR) */
typedef struct{
__REG32 DEST              : 2;
__REG32 CURR_DEST         : 2;
__REG32 STOP              : 1;
__REG32                   : 1;
__REG32 ABORT             : 1;
__REG32                   : 1;
__REG32 BP_RST            : 1;
__REG32 SYNC_CLR          : 1;
__REG32 INT_ERR_EN        : 1;
__REG32                   :21;
} __gccr_bits;

/* Graphics Controller Interrupt Status Control Register (GCISCR) */
typedef struct{
__REG32 EOB_INTST         : 1;
__REG32 IN_INTST          : 1;
__REG32 BF_INTST          : 1;
__REG32 IOP_INTST         : 1;
__REG32 IIN_INTST         : 1;
__REG32 EEOB_INTST        : 1;
__REG32 PF_INTST          : 1;
__REG32 STOP_INTST        : 1;
__REG32                   : 4;
__REG32 IN_INT_ID         :20;
} __gciscr_bits;

/* Graphics Controller Interrupt Enable Control Register (GCIECR) */
typedef struct{
__REG32 EOB_INTEN         : 1;
__REG32 IN_INTEN          : 1;
__REG32 BF_INTEN          : 1;
__REG32 IOP_INTEN         : 1;
__REG32 IIN_INTEN         : 1;
__REG32 EEOB_INTEN        : 1;
__REG32 PF_INTEN          : 1;
__REG32 STOP_INTEN        : 1;
__REG32                   :24;
} __gciecr_bits;

/* Graphics Controller NOP ID Register (GCNOPID) */
typedef struct{
__REG32 NOP_ID            :20;
__REG32                   :12;
} __gcnopid_bits;

/* Graphics Controller Default Alpha Setting Register (GCALPHASET) */
typedef struct{
__REG32 ALPHA             :16;
__REG32 VALID             : 1;
__REG32                   :15;
} __gcalphaset_bits;

/* Graphics Controller Default Transparency Setting Register (GCTSET) */
typedef struct{
__REG32 TBIT              : 1;
__REG32                   :15;
__REG32 VALID             : 1;
__REG32                   :15;
} __gctset_bits;

/* Graphics Controller Ring Buffer Length Register (GCRBLR) */
typedef struct{
__REG32 LENGTH            :18;
__REG32                   :14;
} __gcrblr_bits;

/* Graphics Controller Destination Buffer  Step Size Register (GCDxSTP)
   Graphics Controller Step Size Register (GCSxSTP) */
typedef struct{
__REG32 STEP              : 4;
__REG32                   :28;
} __gcdstp_bits;

/* Graphics Controller Destination Buffer Stride Size Register (GCDxSTR)
   Graphics Controller Stride Size Register (GCSxSTR) */
typedef struct{
__REG32 STRIDE            :14;
__REG32                   :18;
} __gcdstr_bits;

/* Graphics Controller Destination Buffer Pixel Format Register (GCDxPF)
   Graphics Controller Pixel Format Register (GCSxPF) */
typedef struct{
__REG32 PFORM             : 4;
__REG32                   :28;
} __gcdpf_bits;

/* Keypad Control (KPC) Register */
typedef struct{
__REG32 DIE               : 1;
__REG32 DE                : 1;
__REG32 REE0              : 1;
__REG32                   : 1;
__REG32 RE_ZERO_DEB       : 1;
__REG32 DI                : 1;
__REG32 DKN               : 3;
__REG32 DK_DEB_SEL        : 1;
__REG32                   : 1;
__REG32 MIE               : 1;
__REG32 ME                : 1;
__REG32 MS0               : 1;
__REG32 MS1               : 1;
__REG32 MS2               : 1;
__REG32 MS3               : 1;
__REG32 MS4               : 1;
__REG32 MS5               : 1;
__REG32 MS6               : 1;
__REG32 MS7               : 1;
__REG32 IMKP              : 1;
__REG32 MI                : 1;
__REG32 MKCN              : 3;
__REG32 MKRN              : 3;
__REG32 ASACT             : 1;
__REG32 AS                : 1;
__REG32                   : 1;
} __kpc_bits;

/* Keypad Direct Key (KPDK) Register */
typedef struct{
__REG32 RA0_DK0           : 1;
__REG32 RB0_DK1           : 1;
__REG32 DK2               : 1;
__REG32 DK3               : 1;
__REG32 DK4               : 1;
__REG32 DK5               : 1;
__REG32 DK6               : 1;
__REG32 DK7               : 1;
__REG32                   :23;
__REG32 DKP               : 1;
} __kpdk_bits;

/* Keypad Rotary Encoder Count (KPREC) Register */
typedef struct{
__REG32 RE_COUNT0         : 8;
__REG32                   : 6;
__REG32 UF0               : 1;
__REG32 OF0               : 1;
__REG32                   :16;
} __kprec_bits;

/* Keypad Matrix Key (KPMK) Register */
typedef struct{
__REG32 MR0               : 1;
__REG32 MR1               : 1;
__REG32 MR2               : 1;
__REG32 MR3               : 1;
__REG32 MR4               : 1;
__REG32 MR5               : 1;
__REG32 MR6               : 1;
__REG32 MR7               : 1;
__REG32                   :23;
__REG32 MKP               : 1;
} __kpmk_bits;

/* Keypad Interface Automatic Scan (KPAS) Register */
typedef struct{
__REG32 CP                : 4;
__REG32 RP                : 4;
__REG32                   :18;
__REG32 MUKP              : 5;
__REG32 SO                : 1;
} __kpas_bits;

/* Keypad Interface Automatic Scan Multiple Keypress (KPASMKP0) */
typedef struct{
__REG32 MKC0              : 8;
__REG32                   : 8;
__REG32 MKC1              : 8;
__REG32                   : 7;
__REG32 SO                : 1;
} __kpasmkp0_bits;

/* Keypad Interface Automatic Scan Multiple Keypress (KPASMKP1) */
typedef struct{
__REG32 MKC2              : 8;
__REG32                   : 8;
__REG32 MKC3              : 8;
__REG32                   : 7;
__REG32 SO                : 1;
} __kpasmkp1_bits;

/* Keypad Interface Automatic Scan Multiple Keypress (KPASMKP2) */
typedef struct{
__REG32 MKC4              : 8;
__REG32                   : 8;
__REG32 MKC5              : 8;
__REG32                   : 7;
__REG32 SO                : 1;
} __kpasmkp2_bits;

/* Keypad Interface Automatic Scan Multiple Keypress (KPASMKP3) */
typedef struct{
__REG32 MKC6              : 8;
__REG32                   : 8;
__REG32 MKC7              : 8;
__REG32                   : 7;
__REG32 SO                : 1;
} __kpasmkp3_bits;

/* Keypad Key Debounce Interval (KPKDI) Register */
typedef struct{
__REG32 MKDI              : 8;
__REG32 DKDI              : 8;
__REG32                   :16;
} __kpkdi_bits;

/* ADC Data Register (ADCD) */
typedef struct{
__REG32 ADC_DATA0         :12;
__REG32 ADC_MUX0          : 3;
__REG32 ADC_DATA1         :12;
__REG32                   : 2;
__REG32 SV                : 1;
__REG32 VAL               : 1;
__REG32 SD                : 1;
} __adcd_bits;

/* ADC Setup Register (ADCS) */
typedef struct{
__REG32 ADC_DELAY         :16;
__REG32 ADC_MUX           : 3;
__REG32 RESOLUTION        : 1;
__REG32 PRECHARGE_DELAY   : 7;
__REG32 XY_MODE           : 1;
__REG32 VAL_EN            : 1;
__REG32 SD_EN             : 1;
__REG32 CC                : 1;
__REG32 RUN               : 1;
} __adcs_bits;

/* ADC Enable Register (ADCE) */
typedef struct{
__REG32 BIAS_EN           : 1;
__REG32                   : 6;
__REG32 P1                : 1;
__REG32 P2                : 1;
__REG32                   :23;
} __adce_bits;

/* ADC Pressure Register (ADCP) */
typedef struct{
__REG32 YP                : 6;
__REG32 XP                : 6;
__REG32 X_Y               : 6;
__REG32                   :14;
} __adcp_bits;

/* UDC Control Register (UDCCR) */
typedef struct{
__REG32 UDE               : 1;
__REG32 UDA               : 1;
__REG32 UDR               : 1;
__REG32 EMCE              : 1;
__REG32 SMAC              : 1;
__REG32 AAISN             : 3;
__REG32 AIN               : 3;
__REG32 ACN               : 2;
__REG32 PWRMD             : 1;
__REG32                   : 2;
__REG32 DWRE              : 1;
__REG32                   :11;
__REG32 BHNP              : 1;
__REG32 AHNP              : 1;
__REG32 AALTHNP           : 1;
__REG32 OEN               : 1;
} __udccr_bits;

/* UDC Interrupt Control Registers 0 (UDCICR0) */
typedef struct{
__REG32 IE0               : 2;
__REG32 IEA               : 2;
__REG32 IEB               : 2;
__REG32 IEC               : 2;
__REG32 IED               : 2;
__REG32 IEE               : 2;
__REG32 IEF               : 2;
__REG32 IEG               : 2;
__REG32 IEH               : 2;
__REG32 IEI               : 2;
__REG32 IEJ               : 2;
__REG32 IEK               : 2;
__REG32 IEL               : 2;
__REG32 IEM               : 2;
__REG32 IEN               : 2;
__REG32 IEP               : 2;
} __udcicr0_bits;

/* UDC Interrupt Control Registers 1 (UDCICR1) */
typedef struct{
__REG32 IEQ               : 2;
__REG32 IER               : 2;
__REG32 IES               : 2;
__REG32 IET               : 2;
__REG32 IEU               : 2;
__REG32 IEV               : 2;
__REG32 IEW               : 2;
__REG32 IEX               : 2;
__REG32                   :11;
__REG32 IERS              : 1;
__REG32 IESU              : 1;
__REG32 IERU              : 1;
__REG32 IESOF             : 1;
__REG32 IECC              : 1;
} __udccir1_bits;

/* UDC Interrupt Status Registers 0 (UDCISR0) */
typedef struct{
__REG32 IR0               : 2;
__REG32 IRA               : 2;
__REG32 IRB               : 2;
__REG32 IRC               : 2;
__REG32 IRD               : 2;
__REG32 IRE               : 2;
__REG32 IRF               : 2;
__REG32 IRG               : 2;
__REG32 IRH               : 2;
__REG32 IRI               : 2;
__REG32 IRJ               : 2;
__REG32 IRK               : 2;
__REG32 IRL               : 2;
__REG32 IRM               : 2;
__REG32 IRN               : 2;
__REG32 IRP               : 2;
} __udcisr0_bits;

/* UDC Interrupt Status Registers 1 (UDCISR1) */
typedef struct{
__REG32 IRQ               : 2;
__REG32 IRR               : 2;
__REG32 IRS               : 2;
__REG32 IRT               : 2;
__REG32 IRU               : 2;
__REG32 IRV               : 2;
__REG32 IRW               : 2;
__REG32 IRX               : 2;
__REG32                   :11;
__REG32 IRRS              : 1;
__REG32 IRSU              : 1;
__REG32 IRRU              : 1;
__REG32 IRSOF             : 1;
__REG32 IRCC              : 1;
} __udcisr1_bits;

/* UDC Frame Number Register (UDCFNR) */
typedef struct{
__REG32 FN                :11;
__REG32                   :21;
} __udcfnr_bits;

/* UDC On-the-Go Interrupt Control register(UDCOTGICR) */
typedef struct{
__REG32 IEIDF             : 1;
__REG32 IEIDR             : 1;
__REG32 IESDF             : 1;
__REG32 IESDR             : 1;
__REG32 IESVF             : 1;
__REG32 IESVR             : 1;
__REG32 IEVV44F           : 1;
__REG32 IEVV44R           : 1;
__REG32 IEVV40F           : 1;
__REG32 IEVV40R           : 1;
__REG32                   : 6;
__REG32 IEXF              : 1;
__REG32 IEXR              : 1;
__REG32                   : 6;
__REG32 IESF              : 1;
__REG32                   : 7;
} __udcotgicr_bits;

/*  UDC On-the-Go Interrupt Status register(UDCOTGISR) */
typedef struct{
__REG32 IRIDF             : 1;
__REG32 IRIDR             : 1;
__REG32 IRSDF             : 1;
__REG32 IRSDR             : 1;
__REG32 IRSVF             : 1;
__REG32 IRSVR             : 1;
__REG32 IRVV44F           : 1;
__REG32 IRVV44R           : 1;
__REG32 IRVV40F           : 1;
__REG32 IRVV40R           : 1;
__REG32                   : 6;
__REG32 IRXF              : 1;
__REG32 IRXR              : 1;
__REG32                   : 6;
__REG32 IRSF              : 1;
__REG32                   : 7;
} __udcotgisr_bits;

/* USB Port 2 Output Control Register (UP2OCR) */
typedef struct{
__REG32 CPVEN             : 1;
__REG32 CPVPE             : 1;
__REG32 DPPDE             : 1;
__REG32 DMPDE             : 1;
__REG32 DPPUE             : 1;
__REG32 DPSTAT            : 1;
__REG32 VPMBlockEnbN      : 1;
__REG32 DMSTAT            : 1;
__REG32 EXSP              : 1;
__REG32 EXSUS             : 1;
__REG32 IDON              : 1;
__REG32                   : 5;
__REG32 HXS               : 1;
__REG32 HXOE              : 1;
__REG32                   : 6;
__REG32 SEOS              : 3;
__REG32                   : 5;
} __up2ocr_bits;

/* USB Port 3 Output Control Register (UP3OCR) */
typedef struct{
__REG32 CFG               : 2;
__REG32                   :30;
} __up3ocr_bits;

/* UDC Endpoint 0 Control Status Register (UDCCSR0) */
typedef struct{
__REG32 OPC               : 1;
__REG32 IPR               : 1;
__REG32 FTF               : 1;
__REG32 DME               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 RNE               : 1;
__REG32 SA                : 1;
__REG32 AREN              : 1;
__REG32 ACM               : 1;
__REG32 ODFCLR            : 1;
__REG32                   :21;
} __udccsr0_bits;

/* UDC Endpoints A–X Control Status Registers (UDCCRSA-UDCCRSX) */
typedef struct{
__REG32 FS                : 1;
__REG32 PC                : 1;
__REG32 TRN               : 1;
__REG32 DME               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 BNE_BNF           : 1;
__REG32 SP                : 1;
__REG32 FEF               : 1;
__REG32 DPE               : 1;
__REG32 HBNE_HBNF         : 1;
__REG32                   :21;
} __udccsra_bits;

/* UDC Byte Count Registers (UDCBCR0 and UDCBCRA–UDCBCRX) */
typedef struct{
__REG32 BC                :10;
__REG32                   :22;
} __udcbcr_bits;

/* UDC Endpoint A–X Configuration Registers (UDCCRA–UDCCRX) */
typedef struct{
__REG32 EE                : 1;
__REG32 DE                : 1;
__REG32 MPS               :10;
__REG32 ED                : 1;
__REG32 ET                : 2;
__REG32 EN                : 4;
__REG32 AISN              : 3;
__REG32 IN                : 3;
__REG32 CN                : 2;
__REG32                   : 5;
} __udccra_bits;


/* U2D Control register U2DCR */
typedef struct{
__REG32 UDE               : 1;
__REG32 UDA               : 1;
__REG32 UDR               : 1;
__REG32 EMCE              : 1;
__REG32 AAISN             : 4;
__REG32 AIN               : 4;
__REG32 ACN               : 4;
__REG32 DWRE              : 1;
__REG32 SMAC              : 1;
__REG32 HS                : 1;
__REG32 CC                : 1;
__REG32 ADD               : 1;
__REG32 ABP               : 1;
__REG32                   :10;
} __u2dcr_bits;

/* U2D Interrupt Control Register (U2DICR) */
typedef struct{
__REG32 IE0               : 3;
__REG32 IEA               : 3;
__REG32 IEB               : 3;
__REG32 IEC               : 3;
__REG32 IED               : 3;
__REG32 IEE               : 3;
__REG32 IEF               : 3;
__REG32 IEG               : 3;
__REG32                   : 1;
__REG32 IEDPE             : 1;
__REG32 IERS              : 1;
__REG32 IESU              : 1;
__REG32 IERU              : 1;
__REG32 IEUSOF            : 1;
__REG32 IESOF             : 1;
__REG32 IECC              : 1;
} __u2dicr_bits;

/* U2D Interrupt Status Register (U2DISR) */
typedef struct{
__REG32 IR0               : 3;
__REG32 IRA               : 3;
__REG32 IRB               : 3;
__REG32 IRC               : 3;
__REG32 IRD               : 3;
__REG32 IRE               : 3;
__REG32 IRF               : 3;
__REG32 IRG               : 3;
__REG32                   : 1;
__REG32 IRDPE             : 1;
__REG32 IRRS              : 1;
__REG32 IRSU              : 1;
__REG32 IRRU              : 1;
__REG32 IRUSOF            : 1;
__REG32 IRSOF             : 1;
__REG32 IRCC              : 1;
} __u2disr_bits;

/* U2D Frame Number Register (U2DFNR) */
typedef struct{
__REG32 FN                :11;
__REG32                   :21;
} __u2dfnr_bits;

/* U2D Endpoint 0 Control Status Register (U2DCSR0) */
typedef struct{
__REG32 OPC               : 1;
__REG32 IPR               : 1;
__REG32 FTF               : 1;
__REG32 DME               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 RNE               : 1;
__REG32 SA                : 1;
__REG32 IPA               : 1;
__REG32                   :23;
} __u2dcsr0_bits;

/* U2D Endpoints A–P Control Status Registers (U2DCSRA–U2DCSRG) */
typedef struct{
__REG32 FS                : 1;
__REG32 PC                : 1;
__REG32 TRN               : 1;
__REG32 DME               : 1;
__REG32 SST               : 1;
__REG32 FST               : 1;
__REG32 BNE_BNF           : 1;
__REG32 SP                : 1;
__REG32 FEF               : 1;
__REG32 DPE               : 1;
__REG32 BF_BE             : 1;
__REG32                   :21;
} __u2dcsra_bits;

/* U2D Byte Count Register (U2DBCR0) */
typedef struct{
__REG32 BC                :11;
__REG32                   :21;
} __u2dbcr0_bits;

/* U2D Endpoint A-P Configuration Registers (U2DCRA–U2DCRG) */
typedef struct{
__REG32 EE                : 1;
__REG32 BS                :10;
__REG32                   :21;
} __u2dcra_bits;             

/* U2D Endpoint 0 Information Register (U2DEN0) */
typedef struct{
__REG32                   :19;
__REG32 MPS               :11;
__REG32                   : 2;
} __u2den0_bits;

/* U2D Endpoint Information Registers (U2DENA–U2DENG) */
typedef struct{
__REG32 EN                : 4;
__REG32 ED                : 1;
__REG32 ET                : 2;
__REG32 CN                : 4;
__REG32 IN                : 4;
__REG32 AISN              : 4;
__REG32 MPS               :11;
__REG32 HBW               : 2;
} __u2dena_bits;

/* U2DMA Channel Control/Status Registers (U2DMACSR0-7) */
typedef struct{
__REG32 BUSERRINTR        : 1;
__REG32 STARTINTR         : 1;
__REG32 ENDINTR           : 1;
__REG32 STOPINTR          : 1;
__REG32 RASIntr           : 1;
__REG32                   : 3;
__REG32 REQPEND           : 1;
__REG32 EORINTR           : 1;
__REG32 BUSERRTYPE        : 3;
__REG32 SCEMI             : 5;
__REG32 SCEMC             : 2;
__REG32                   : 2;
__REG32 MaskRun           : 1;
__REG32 RASIrqEn          : 1;
__REG32                   : 2;
__REG32 EORSTOPEN         : 1;
__REG32 EORJMPEN          : 1;
__REG32 EORIRQEN          : 1;
__REG32 STOPIRQEN         : 1;
__REG32                   : 1;
__REG32 RUN               : 1;
} __u2dmacsrx_bits;

/* U2DMA Control Register (U2DMACR)  */
typedef struct{
__REG32 MAXOCT            : 2;
__REG32 RETRYTOEN         : 1;
__REG32                   :29;
} __u2dmacr_bits;

/* U2DMA Descriptor Address Registers (U2DMADADR0-7) */
typedef struct{
__REG32 STOP              : 1;
__REG32                   : 3;
__REG32 DESCRIPA          :28;
} __u2dmadadrx_bits;

/* U2DMA Command Registers (U2DMACMD0-7) */
typedef struct{
__REG32 LEN               :11;
__REG32                   : 2;
__REG32 PACKCOMP          : 1;
__REG32                   : 7;
__REG32 EndIrqEn          : 1;
__REG32 STARTIRQEN        : 1;
__REG32 LSTDES_EN         : 2;
__REG32                   : 6;
__REG32 XFRDIR            : 1;
} __u2dmacmdx_bits;

/* UHC Host Control Register (UHCHCON) */
typedef struct {
  __REG32 CBSR              : 2;
  __REG32 PLE               : 1;
  __REG32 IE                : 1;
  __REG32 CLE               : 1;
  __REG32 BLE               : 1;
  __REG32 HCFS              : 2;
  __REG32 IR                : 1;
  __REG32 RWC               : 1;
  __REG32 RWE               : 1;
  __REG32                   :21;
} __hccontrol_bits;

/* UHC Command Status (UHCCOMS) */
typedef struct {
  __REG32 HCR               : 1;
  __REG32 CLF               : 1;
  __REG32 BLF               : 1;
  __REG32 OCR               : 1;
  __REG32                   :12;
  __REG32 SOC               : 2;
  __REG32                   :14;
} __hccommandstatus_bits;

/* UHC Interrupt Status (UHCINTS) */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32                   : 1;
} __hcinterruptstatus_bits;

/* UHC Interrupt Enable (UHCINTE)
   UHC Interrupt Disable (UHCINTD) */
typedef struct {
  __REG32 SO                : 1;
  __REG32 WDH               : 1;
  __REG32 SF                : 1;
  __REG32 RD                : 1;
  __REG32 UE                : 1;
  __REG32 FNO               : 1;
  __REG32 RHSC              : 1;
  __REG32                   :23;
  __REG32 OC                : 1;
  __REG32 MIE               : 1;
} __hcinterruptenable_bits;

/* UHC Host Controller Communication Area (UHCHCCA) */
typedef struct {
  __REG32                   : 8;
  __REG32 HCCA              :24;
} __hchcca_bits;

/* UHC Period Current Endpoint Descriptor (UHCPCED) */
typedef struct {
  __REG32                   : 4;
  __REG32 PCED              :28;
} __hcperiodcurrented_bits;

/* UHC Control Head Endpoint Descriptor (UHCCHED) */
typedef struct {
  __REG32                   : 4;
  __REG32 CHED              :28;
} __hccontrolheaded_bits;

/* UHC Control Current Endpoint Descriptor (UHCCCED) */
typedef struct {
  __REG32                   : 4;
  __REG32 CCED              :28;
} __hccontrolcurrented_bits;

/* UHC Bulk Head Endpoint Descriptor (UHCBHED) */
typedef struct {
  __REG32                   : 4;
  __REG32 BHED              :28;
} __hcbulkheaded_bits;

/* UHC Bulk Current Endpoint Descriptor (UHCBCED) */
typedef struct {
  __REG32                   : 4;
  __REG32 BCED              :28;
} __hcbulkcurrented_bits;

/* UHC Done Head (UHCDHEAD) */
typedef struct {
  __REG32                   : 4;
  __REG32 DHED              :28;
} __hcdonehead_bits;

/* UHC Frame Interval (UHCFMI) */
typedef struct {
  __REG32 FI                :14;
  __REG32                   : 2;
  __REG32 FSMPS             :15;
  __REG32 FIT               : 1;
} __hcfminterval_bits;

/* UHC Frame Remaining (UHCFMR) */
typedef struct {
  __REG32 FR                :14;
  __REG32                   :17;
  __REG32 FRT               : 1;
} __hcfmremaining_bits;

/* UHC Frame Number (UHCFMN) */
typedef struct {
  __REG32 FN                :16;
  __REG32                   :16;
} __hcfmnumber_bits;

/* UHC Periodic Start (UHCPERS) */
typedef struct {
  __REG32 PS                :14;
  __REG32                   :18;
} __hcperiodicstart_bits;

/* UHC Low-Speed Threshold (UHCLST) */
typedef struct {
  __REG32 LST               :12;
  __REG32                   :20;
} __hclsthreshold_bits;

/* UHC Root Hub Descriptor A (UHCRHDA) */
typedef struct {
  __REG32 NDP               : 8;
  __REG32 PSM               : 1;  /* ??*/
  __REG32 NPS               : 1;  /* ??*/
  __REG32 DT                : 1;
  __REG32 OCPM              : 1;
  __REG32 NOCP              : 1;
  __REG32                   :11;
  __REG32 POTPGT            : 8;
} __hcrhdescriptora_bits;

/* UHC Root Hub Descriptor B (UHCRHDB) */
typedef struct {
  __REG32 DR                :16;
  __REG32 PPCM              :16;
} __hcrhdescriptorb_bits;

/* UHC Root Hub Status (UHCRHS) */
typedef struct {
  __REG32 LPS               : 1;
  __REG32 OCI               : 1;
  __REG32                   :13;
  __REG32 DRWE              : 1;
  __REG32 LPSC              : 1;
  __REG32 OCIC              : 1;
  __REG32                   :13;
  __REG32 CRWE              : 1;
} __hcrhstatus_bits;

/* UHC Root Hub Port Status1/2/3 (UHCRHPS1, UHCRHPS2 and UHCRHPS3) */
typedef struct {
  __REG32 CCS               : 1;
  __REG32 PES               : 1;
  __REG32 PSS               : 1;
  __REG32 POCI              : 1;
  __REG32 PRS               : 1;
  __REG32                   : 3;
  __REG32 PPS               : 1;
  __REG32 LSDA              : 1;
  __REG32                   : 6;
  __REG32 CSC               : 1;
  __REG32 PESC              : 1;
  __REG32 PSSC              : 1;
  __REG32                   : 1;
  __REG32 PRSC              : 1;
  __REG32                   :11;
} __hcrhportstatus_bits;

/* UHC Status Register (UHCSTAT) */
typedef struct {
  __REG32                   : 7;
  __REG32 RWUE              : 1;
  __REG32 HBA               : 1;
  __REG32                   : 1;
  __REG32 HTA               : 1;
  __REG32 UPS1              : 1;
  __REG32                   : 1;
  __REG32 UPRI              : 1;
  __REG32 SBTAI             : 1;
  __REG32 SBMAI             : 1;
  __REG32                   :16;
} __uhcstat_bits;

/* UHC Reset Register (UHCHR) */
typedef struct {
  __REG32 FSBIR             : 1;
  __REG32 FHR               : 1;
  __REG32 CGR               : 1;
  __REG32 SSDC              : 1;
  __REG32 UIT               : 1;
  __REG32 SSE               : 1;
  __REG32 PSPL              : 1;
  __REG32 PCPL              : 1;
  __REG32                   : 1;
  __REG32 SSEP1             : 1;
  __REG32 SSEP2             : 1;
  __REG32 SSEP3             : 1;
  __REG32                   :20;
} __uhchr_bits;

/* UHC Interrupt Enable Register (UHCHIE) */
typedef struct {
  __REG32                   : 7;
  __REG32 RWIE              : 1;
  __REG32 HBAIE             : 1;
  __REG32                   : 1;
  __REG32 TAIE              : 1;
  __REG32 UPS1IE            : 1;
  __REG32 UPS2IE            : 1;
  __REG32 UPRIE             : 1;
  __REG32 UPS3IE            : 1;
  __REG32                   :17;
} __uhchie_bits;

/* UHC Interrupt Test Register (UHCHIT) */
typedef struct {
  __REG32                   : 7;
  __REG32 RWUT              : 1;
  __REG32 BAT               : 1;
  __REG32 IRQT              : 1;
  __REG32 TAT               : 1;
  __REG32 UPS1T             : 1;
  __REG32 UPS2T             : 1;
  __REG32 UPRT              : 1;
  __REG32 STAT              : 1;
  __REG32 SMAT              : 1;
  __REG32 UPS3T             : 1;
  __REG32                   :15;
} __uhchit_bits;

/* SSP Control Register 0 (SSCR0_x) */
typedef struct {
  __REG32 DSS               : 4;
  __REG32 FRF               : 2;
  __REG32 ECS               : 1;
  __REG32 SSE               : 1;
  __REG32 SCR               :12;
  __REG32 EDSS              : 1;
  __REG32 NCS               : 1;
  __REG32 RIM               : 1;
  __REG32 TIM               : 1;
  __REG32 FRDC              : 3;
  __REG32                   : 2;
  __REG32 FPCKE             : 1;
  __REG32 ACS               : 1;
  __REG32 MOD               : 1;
} __sscr0_x_bits;

/* SSP Control Register 1 (SSCR1_x) */
typedef struct {
  __REG32 RIE               : 1;
  __REG32 TIE               : 1;
  __REG32 LBM               : 1;
  __REG32 SPO               : 1;
  __REG32 SPH               : 1;
  __REG32                   : 1;
  __REG32 TFT               : 4;
  __REG32 RFT               : 4;
  __REG32 EFWR              : 1;
  __REG32 STRF              : 1;
  __REG32 IFS               : 1;
  __REG32                   : 1;
  __REG32 PINTE             : 1;
  __REG32 TINTE             : 1;
  __REG32 RSRE              : 1;
  __REG32 TSRE              : 1;
  __REG32 TRAIL             : 1;
  __REG32 RWOT              : 1;
  __REG32 SFRMDIR           : 1;
  __REG32 SCLKDIR           : 1;
  __REG32 ECRB              : 1;
  __REG32 ECRA              : 1;
  __REG32 SCFR              : 1;
  __REG32 EBCEI             : 1;
  __REG32 TTE               : 1;
  __REG32 TTELP             : 1;
} __sscr1_x_bits;

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
  __REG32                   : 2;
  __REG32 PINT              : 1;
  __REG32 TINT              : 1;
  __REG32 EOC               : 1;
  __REG32 TUR               : 1;
  __REG32 CSS               : 1;
  __REG32 BCE               : 1;
  __REG32                   : 7;
  __REG32 OSS               : 1;
} __sssr_x_bits;

/* SSP Interrupt Test Register (SSITR) */
typedef struct {
  __REG32                   : 5;
  __REG32 TTFS              : 1;
  __REG32 TRFS              : 1;
  __REG32 TROR              : 1;
  __REG32                   :24;
} __ssitr_x_bits;

/* SSP Time Out Register (SSTO) */
typedef struct {
  __REG32 TIMEOUT           :24;
  __REG32                   : 8;
} __ssto_x_bits;

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
  __REG32 FSRT              : 1;
  __REG32 EDMYSTRT          : 2;
  __REG32 EDMYSTOP          : 3;
  __REG32                   : 1;
} __sspsp_x_bits;

/* SSP TX Time Slot Active Register (SSTSA) */
typedef struct {
  __REG32 TTSA              : 8;
  __REG32                   :24;
} __sstsa_x_bits;

/* SSP RX Time Slot Active Register (SSRSA) */
typedef struct {
  __REG32 RTSA              : 8;
  __REG32                   :24;
} __ssrsa_x_bits;

/* SSP Time Slot Status Register (SSTSS) */
typedef struct {
  __REG32 TSS               : 3;
  __REG32                   :28;
  __REG32 NMBSY             : 1;
} __sstss_x_bits;

/* SSP Audio Clock Divider Register (SSACD) */
typedef struct {
  __REG32 ACDS              : 3;
  __REG32 SCDB              : 1;
  __REG32 ACPS              : 3;
  __REG32 SCDX8             : 1;
  __REG32                   :24;
} __ssacd_x_bits;

/* SSP Audio Clock Dither Divider Register (SSACDD) */
typedef struct {
  __REG32 DEN               :12;
  __REG32                   : 4;
  __REG32 NUM               :15;
  __REG32                   : 1;
} __ssacdd_x_bits;

/* PCM-Out Control Register (POCR) */
/* PCM-In Control Register (PCMICR) */
/* PCM-Surround-Out Control Register (PCSCR) */
/* PCM Center/LFE Control Register (PCCLCR) */
/* Mic-In Control Register (MCCR) */
/* MODEM-Out Control Register (MOCR) */
/* MODEM-In Control Register (MICR) */
typedef struct {
  __REG32                   : 1;
  __REG32 FSRIE             : 1;
  __REG32                   : 1;
  __REG32 FEIE              : 1;
  __REG32                   :28;
} __pocr_bits;

/* Global Control Register (GCR) */
typedef struct {
  __REG32 GPI_IE            : 1;
  __REG32 nCRST             : 1;
  __REG32 WRST              : 1;
  __REG32 ACOFF             : 1;
  __REG32 PRES_IE           : 1;
  __REG32 S1RES_IE          : 1;
  __REG32                   : 2;
  __REG32 PRDY_IE           : 1;
  __REG32 S1RDY_IE          : 1;
  __REG32                   : 8;
  __REG32 SDONE_IE          : 1;
  __REG32 CDONE_IE          : 1;
  __REG32                   : 4;
  __REG32 nDMAEN            : 1;
  __REG32                   : 5;
  __REG32 FRCRST            : 1;
  __REG32 CLKBPB            : 1;
} __gcr_bits;

/* PCM-Out Status Register (POSR) */
/* PCM Surround-Out Status Register (PCSSR) */
/* PCM Center/LFE Status Register (PCCLSR) */
/* MODEM-Out Status Register (MOSR) */
typedef struct {
  __REG32                   : 2;
  __REG32 FSR               : 1;
  __REG32                   : 1;
  __REG32 FIFOE             : 1;
  __REG32                   :27;
} __posr_bits;

/* PCM-In Status Register (PCMISR) */
/* Mic-In Status Register (MCSR) */
/* MODEM-In Status Register (MISR) */
typedef struct {
  __REG32                   : 2;
  __REG32 FSR               : 1;
  __REG32 EOC               : 1;
  __REG32 FIFOE             : 1;
  __REG32                   :27;
} __pcmisr_bits;

/* Global Status Register */
typedef struct {
  __REG32 GSCI              : 1;
  __REG32 MIINT             : 1;
  __REG32 MOINT             : 1;
  __REG32 ACOFFD            : 1;
  __REG32                   : 1;
  __REG32 PIINT             : 1;
  __REG32 POINT             : 1;
  __REG32 MCINT             : 1;
  __REG32 PCRDY             : 1;
  __REG32 S1CRDY            : 1;
  __REG32 PRESINT           : 1;
  __REG32 S1RESINT          : 1;
  __REG32 B1S12             : 1;
  __REG32 B2S12             : 1;
  __REG32 B3S12             : 1;
  __REG32 RCS               : 1;
  __REG32                   : 2;
  __REG32 SDONE             : 1;
  __REG32 CDONE             : 1;
  __REG32 PCLINT            : 1;
  __REG32 PSOINT            : 1;
  __REG32                   :10;
} __gsr_bits;

/* CODEC Access Register (CAR) */
typedef struct {
  __REG32 CAIP              : 1;
  __REG32                   :31;
} __car_bits;

/* PCM Surround Data Register (PCSDR) */
typedef struct {
  __REG32 PSML              :16;
  __REG32 PSMR              :16;
} __pcsdr_bits;

/* PCM Center/LFE Data Register (PCCLDR) */
typedef struct {
  __REG32 PCLML             :16;
  __REG32 PCLMR             :16;
} __pccldr_bits;

/* PCM Data Register (PCDR) */
typedef struct {
  __REG32 PCML              :16;
  __REG32 PCMR              :16;
} __pcdr_bits;

/* Mic-In Data Register (MCDR) */
typedef struct {
  __REG32 MCDAT             :16;
  __REG32                   :16;
} __mcdr_bits;

/* MODEM Data Register (MODR) */
typedef struct {
  __REG32 MODAT             :16;
  __REG32                   :16;
} __modr_bits;

/* CIR Pulse Width Comparator Register (CIRPW) */
typedef struct {
  __REG32 PW                :11;
  __REG32                   :21;
} __cirpw_bits;

/* CIR Modulation Period Comparator Register (CIRMP) */
typedef struct {
  __REG32 MP                :11;
  __REG32                   :21;
}__cirmp_bits;

/* CIR N0 Symbol Length Register (CIRN0) */
typedef struct {
  __REG32 N0S               : 7;
  __REG32                   :25;
}__cirn0_bits;

/* CIR N1 Symbol Length Register (CIRN1) */
typedef struct {
  __REG32 N1S               : 7;
  __REG32 N1_Unmod          : 1;
  __REG32                   :24;
}__cirn1_bits;

/* CIR S0 Symbol Length Register (CIRS0) */
typedef struct {
  __REG32 S0S               : 7;
  __REG32                   :25;
}__cirs0_bits;

/* CIR S1 Symbol Length Register (CIRS1) */
typedef struct {
  __REG32 S1S               : 7;
  __REG32 S1_Unmod          : 1;
  __REG32                   :24;
}__cirs1_bits;

/* CIR Number of Symbols Register (CIRNS) */
typedef struct {
  __REG32 SYMCNT            : 7;
  __REG32                   :25;
}__cirns_bits;

/* CIR Control Register (CIRCR) */
typedef struct {
  __REG32 CIR_Enable        : 1;
  __REG32 CLK_Enable        : 1;
  __REG32                   : 2;
  __REG32 Manchester        : 1;
  __REG32 SW_Reset          : 1;
  __REG32                   :26;
}__circr_bits;

/* CIR Interrupt Register (CIRIR) */
typedef struct {
  __REG32 EB_Mask           : 1;
  __REG32 EOT_Mask          : 1;
  __REG32                   : 6;
  __REG32 EB                : 1;
  __REG32 EOT               : 1;
  __REG32                   :22;
}__cirir_bits;

/* Receive Buffer Register (RBR) */
/* Transmit Holding Register (THR) */
/* Divisor Latch Register  Low (DLL) */
typedef union {
  /*FFRBR*/
  /*FFTHR*/
  /*BTRBR*/
  /*BTTHR*/
  /*STRBR*/
  /*STTHR*/
  struct {
    __REG32 Byte_0           : 8;
    __REG32 Byte_1           : 8;
    __REG32 Byte_2           : 8;
    __REG32 Byte_3           : 8;
  } ;
  /*FFDLL*/
  /*BTDLL*/
  /*STDLL*/
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
    __REG32 nIP              : 1;
    __REG32 IID              : 2;
    __REG32 TOD              : 1;
    __REG32 ABL              : 1;
    __REG32 EOC              : 1;
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
    __REG32 TIL              : 1;
    __REG32 TRAIL            : 1;
    __REG32 BUS              : 1;
    __REG32 ITL              : 2;
    __REG32                  :24;
  } ;
} __uartiir_bits;

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
  __REG32 AFE              : 1;
  __REG32                  :26;
} __uartmcr_bits;

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

/* Scratchpad Register (SPR) */
typedef struct {
  __REG32 SCRATCHPAD       : 8;
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
  __REG32 BYTE_COUNT       : 6;
  __REG32                  :26;
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

/* PWM Control Registers (PWMCRx) */
typedef struct {
  __REG32 PRESCALE         : 6;
  __REG32 SD               : 1;
  __REG32                  :25;
} __pwmcrx_bits;

/* PWM Duty Cycle Registers (PWMDCRx) */
typedef struct {
  __REG32 DCYCLE           :10;
  __REG32 FD               : 1;
  __REG32                  :21;
} __pwmdcrx_bits;

/* PWM Period Control Registers (PWMPCRx) */
typedef struct {
  __REG32 PV               :10;
  __REG32                  :22;
} __pwmpcrx_bits;

/* USIM Receive Buffer Register (RBR) */
typedef struct {
  __REG32 RB               : 8;
  __REG32 PERR             : 1;
  __REG32                  :23;
} __usimxrbr_bits;

/* USIM Transmit Holding Register (THR) */
typedef struct {
  __REG32 TB               : 8;
  __REG32                  :24;
} __usimxthr_bits;

/* USIM Interrupt Enable Register (IER) */
typedef struct {
  __REG32 OVRN             : 1;
  __REG32 PERR             : 1;
  __REG32 T0ERR            : 1;
  __REG32 FRAMERR          : 1;
  __REG32 TIMEO            : 1;
  __REG32 CWT              : 1;
  __REG32 BWT              : 1;
  __REG32                  : 1;
  __REG32 RDR              : 1;
  __REG32 TDR              : 1;
  __REG32 SmartCard_DET    : 1;
  __REG32                  : 2;
  __REG32 DMA_TIME         : 1;
  __REG32 DMA_RX           : 1;
  __REG32 DMA_TX           : 1;
  __REG32                  :16;
} __usimxier_bits;

/* USIM Interrupt Identification Register (IIR) */
typedef struct {
  __REG32 OVRN             : 1;
  __REG32 PERR             : 1;
  __REG32 T0ERR            : 1;
  __REG32 FRAMERR          : 1;
  __REG32 TIMEO            : 1;
  __REG32 CWT              : 1;
  __REG32 BWT              : 1;
  __REG32                  : 1;
  __REG32 RDR              : 1;
  __REG32 TDR              : 1;
  __REG32 SmartCard_DET    : 1;
  __REG32                  :21;
} __usimxiir_bits;

/* USIM FIFO Control Register (FCR) */
typedef struct {
  __REG32 RESETRF          : 1;
  __REG32 RESETTF          : 1;
  __REG32 TX_HOLD          : 1;
  __REG32 PEM              : 1;
  __REG32                  : 2;
  __REG32 RX_TL            : 2;
  __REG32 TX_TL            : 1;
  __REG32                  :23;
} __usimxfcr_bits;

/* USIM FIFO Status Register (FSR) */
typedef struct {
  __REG32 RX_LENGTH        : 5;
  __REG32 TX_LENGTH        : 5;
  __REG32 PERR_NUM         : 5;
  __REG32                  :17;
} __usimxfsr_bits;

/* USIM Error Control Register (ECR) */
typedef struct {
  __REG32 T0ERR_TL         : 2;
  __REG32                  : 1;
  __REG32 PE_TL            : 2;
  __REG32                  : 1;
  __REG32 T0_CLR           : 1;
  __REG32 T0_REPEAT        : 1;
  __REG32                  :24;
} __usimxecr_bits;

/* USIM Line Control Register (LCR) */
typedef struct {
  __REG32 INVERSE          : 1;
  __REG32 ORDER            : 1;
  __REG32 EPS              : 1;
  __REG32 RX_T1            : 1;
  __REG32 TX_T1            : 1;
  __REG32                  :27;
} __usimxlcr_bits;

/* USIM SmartCard Control Register (USCCR) */
typedef struct {
  __REG32 RST_SmartCard_N  : 1;
  __REG32 VCC              : 2;
  __REG32                  : 1;
  __REG32 TXD_FORCE        : 1;
  __REG32                  :27;
} __usimxusccr_bits;

/* USIM Line Status Register (LSR) */
typedef struct {
  __REG32 OVRN             : 1;
  __REG32 PERR             : 1;
  __REG32 T0ERR            : 1;
  __REG32 FRAMERR          : 1;
  __REG32 TIMEO            : 1;
  __REG32 CWT              : 1;
  __REG32 BWT              : 1;
  __REG32                  : 4;
  __REG32 TDR              : 1;
  __REG32 RX_EMPTY_N       : 1;
  __REG32 TX_WORKING       : 1;
  __REG32 RX_WORKING       : 1;
  __REG32 RXD              : 1;
  __REG32                  :16;
} __usimxlsr_bits;

/* USIM Extra Guard Time Register (EGTR) */
typedef struct {
  __REG32 EGTM             : 8;
  __REG32                  :24;
} __usimxegtr_bits;

/* USIM Block Guard Time Register (BGTR) */
typedef struct {
  __REG32 BGT              : 8;
  __REG32                  :24;
} __usimxbgtr_bits;

/* USIM Time-Out Register (TOR) */
typedef struct {
  __REG32 TO               : 8;
  __REG32                  :24;
} __usimxtor_bits;

/* USIM Clock Register (CLKR) */
typedef struct {
  __REG32 DIVISOR          : 8;
  __REG32                  : 4;
  __REG32 RQST             : 1;
  __REG32 STOP_UCLK        : 1;
  __REG32 STOP_LEVEL       : 1;
  __REG32 STOP_CLK_USIM    : 1;
  __REG32                  :16;
} __usimxclkr_bits;

/* USIM Divisor Latch Register (DLR) */
typedef struct {
  __REG32 DIVISOR          :16;
  __REG32                  :16;
} __usimxdlr_bits;

/* USIM Factor Latch Register (FLR) */
typedef struct {
  __REG32 FACTOR           : 8;
  __REG32                  :24;
} __usimxflr_bits;

/* USIM Character Waiting Time Register (CWTR) */
typedef struct {
  __REG32 CWT              :16;
  __REG32                  :16;
} __usimxcwtr_bits;

/* USIM Block Waiting Time Register (BWTR) */
typedef struct {
  __REG32 BWT              :16;
  __REG32                  :16;
} __usimxbwtr_bits;

/* I2C Bus Monitor Register (IBMR) */
typedef struct {
  __REG32 SDA              : 1;
  __REG32 SCL              : 1;
  __REG32                  :30;
} __ibmr_bits;

/* I2C Data Buffer Register (IDBR) */
typedef struct {
  __REG32 DATA             : 8;
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
  __REG32 DRFIE            : 1;
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
  __REG32 SA               : 7;
  __REG32                  :25;
} __isar_bits;

#endif    /* __IAR_SYSTEMS_ICC__ */

/* Declarations common to compiler and assembler **************************/

/***************************************************************************
 **
 ** MFP GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(MFPR_GPIO0,            0x40E10124,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO1,            0x40E10128,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO2,            0x40E1012C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO3,            0x40E10130,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO4,            0x40E10134,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_nXCVREN,          0x40E10138,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_CLE_NOE,       0x40E10204,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ALE_WE1,       0x40E10208,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_SCLK_E,        0x40E10210,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_nBE0,             0x40E10214,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_nBE1,             0x40E10218,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ALE_WE2,       0x40E1021C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_INT_RnB,       0x40E10220,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nCS0,          0x40E10224,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nCS1,          0x40E10228,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nWE,           0x40E1022C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nRE,           0x40E10230,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nLUA,          0x40E10234,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_nLLA,          0x40E10238,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ADDR0,         0x40E1023C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ADDR1,         0x40E10240,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ADDR2,         0x40E10244,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_ADDR3,         0x40E10248,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO0,           0x40E1024C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO8,           0x40E10250,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO1,           0x40E10254,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO9,           0x40E10258,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO2,           0x40E1025C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO10,          0x40E10260,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO3,           0x40E10264,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO11,          0x40E10268,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO4,           0x40E1026C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO12,          0x40E10270,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO5,           0x40E10274,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO13,          0x40E10278,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO6,           0x40E1027C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO14,          0x40E10280,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO7,           0x40E10284,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_DF_IO15,          0x40E10288,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO5,            0x40E1028C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO6,            0x40E10290,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO7,            0x40E10294,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO8,            0x40E10298,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO9,            0x40E1029C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO11,           0x40E102A0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO12,           0x40E102A4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO13,           0x40E102A8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO14,           0x40E102AC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO15,           0x40E102B0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO16,           0x40E102B4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO17,           0x40E102B8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO18,           0x40E102BC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO19,           0x40E102C0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO20,           0x40E102C4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO21,           0x40E102C8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO22,           0x40E102CC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO23,           0x40E102D0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO24,           0x40E102D4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO25,           0x40E102D8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO26,           0x40E102DC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO27,           0x40E10400,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO28,           0x40E10404,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO29,           0x40E10408,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO30,           0x40E1040C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO31,           0x40E10410,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO32,           0x40E10414,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO33,           0x40E10418,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO34,           0x40E1041C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO35,           0x40E10420,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO36,           0x40E10424,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO37,           0x40E10428,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO38,           0x40E1042C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO39,           0x40E10430,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO40,           0x40E10434,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO41,           0x40E10438,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO42,           0x40E1043C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO43,           0x40E10440,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO44,           0x40E10444,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO45,           0x40E10448,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO46,           0x40E1044C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO47,           0x40E10450,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO48,           0x40E10454,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO10,           0x40E10458,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO49,           0x40E1045C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO50,           0x40E10460,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO51,           0x40E10464,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO52,           0x40E10468,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO53,           0x40E1046C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO54,           0x40E10470,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO55,           0x40E10474,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO56,           0x40E10478,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO57,           0x40E1047C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO58,           0x40E10480,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO59,           0x40E10484,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO60,           0x40E10488,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO61,           0x40E1048C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO62,           0x40E10490,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO6_2,          0x40E10494,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO7_2,          0x40E10498,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO8_2,          0x40E1049C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO9_2,          0x40E104A0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO10_2,         0x40E104A4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO11_2,         0x40E104A8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO12_2,         0x40E104AC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO13_2,         0x40E104B0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO63,           0x40E104B4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO64,           0x40E104B8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO65,           0x40E104BC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO66,           0x40E104C0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO67,           0x40E104C4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO68,           0x40E104C8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO69,           0x40E104CC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO70,           0x40E104D0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO71,           0x40E104D4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO72,           0x40E104D8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO73,           0x40E104DC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO14_2,         0x40E104E0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO15_2,         0x40E104E4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO16_2,         0x40E104E8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO17_2,         0x40E104EC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO74,           0x40E104F0,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO75,           0x40E104F4,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO76,           0x40E104F8,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO77,           0x40E104FC,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO78,           0x40E10500,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO79,           0x40E10504,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO80,           0x40E10508,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO81,           0x40E1050C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO82,           0x40E10510,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO83,           0x40E10514,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO84,           0x40E10518,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO85,           0x40E1051C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO86,           0x40E10520,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO87,           0x40E10524,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO88,           0x40E10528,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO89,           0x40E1052C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO90,           0x40E10530,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO91,           0x40E10534,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO92,           0x40E10538,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO93,           0x40E1053C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO94,           0x40E10540,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO95,           0x40E10544,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO96,           0x40E10548,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO97,           0x40E1054C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO98,           0x40E10550,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO99,           0x40E10600,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO100,          0x40E10604,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO101,          0x40E10608,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO102,          0x40E1060C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO103,          0x40E10610,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO104,          0x40E10614,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO105,          0x40E10618,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO106,          0x40E1061C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO107,          0x40E10620,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO108,          0x40E10624,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO109,          0x40E10628,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO110,          0x40E1062C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO111,          0x40E10630,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO112,          0x40E10634,__READ_WRITE ,__mfpr_bits);
/*__IO_REG32_BIT(MFPR_TSI_XP,           ,__READ_WRITE ,__mfpr_bits);*/
/*__IO_REG32_BIT(MFPR_TSI_XM,           ,__READ_WRITE ,__mfpr_bits);*/
/*__IO_REG32_BIT(MFPR_TSI_YP,           ,__READ_WRITE ,__mfpr_bits);*/
/*__IO_REG32_BIT(MFPR_TSI_YM,           ,__READ_WRITE ,__mfpr_bits);*/
__IO_REG32_BIT(MFPR_GPIO113,          0x40E10638,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO114,          0x40E1063C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO115,          0x40E10640,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO116,          0x40E10644,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO117,          0x40E10648,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO118,          0x40E1064C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO119,          0x40E10650,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO120,          0x40E10654,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO121,          0x40E10658,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO122,          0x40E1065C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO123,          0x40E10660,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO124,          0x40E10664,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO125,          0x40E10668,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO126,          0x40E1066C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO127,          0x40E10670,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO0_2,          0x40E10674,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO1_2,          0x40E10678,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO2_2,          0x40E1067C,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO3_2,          0x40E10680,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO4_2,          0x40E10684,__READ_WRITE ,__mfpr_bits);
__IO_REG32_BIT(MFPR_GPIO5_2,          0x40E10688,__READ_WRITE ,__mfpr_bits);

/***************************************************************************
 **
 ** GPIO
 **
 ***************************************************************************/
__IO_REG32_BIT(GPLR0,                 0x40E00000,__READ       ,__gplr0_bits);
__IO_REG32_BIT(GPLR1,                 0x40E00004,__READ       ,__gplr1_bits);
__IO_REG32_BIT(GPLR2,                 0x40E00008,__READ       ,__gplr2_bits);
__IO_REG32_BIT(GPLR3,                 0x40E00100,__READ       ,__gplr3_bits);
__IO_REG32_BIT(GPDR0,                 0x40E0000C,__READ_WRITE ,__gpdr0_bits);
__IO_REG32_BIT(GPDR1,                 0x40E00010,__READ_WRITE ,__gpdr1_bits);
__IO_REG32_BIT(GPDR2,                 0x40E00014,__READ_WRITE ,__gpdr2_bits);
__IO_REG32_BIT(GPDR3,                 0x40E0010C,__READ_WRITE ,__gpdr3_bits);
__IO_REG32_BIT(GSDR0,                 0x40E00400,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GSDR1,                 0x40E00404,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GSDR2,                 0x40E00408,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GSDR3,                 0x40E0040C,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GCDR0,                 0x40E00420,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GCDR1,                 0x40E00424,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GCDR2,                 0x40E00428,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GCDR3,                 0x40E0042C,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GPSR0,                 0x40E00018,__WRITE      ,__gpsr0_bits);
__IO_REG32_BIT(GPSR1,                 0x40E0001C,__WRITE      ,__gpsr1_bits);
__IO_REG32_BIT(GPSR2,                 0x40E00020,__WRITE      ,__gpsr2_bits);
__IO_REG32_BIT(GPSR3,                 0x40E00118,__WRITE      ,__gpsr3_bits);
__IO_REG32_BIT(GPCR0,                 0x40E00024,__WRITE      ,__gpcr0_bits);
__IO_REG32_BIT(GPCR1,                 0x40E00028,__WRITE      ,__gpcr1_bits);
__IO_REG32_BIT(GPCR2,                 0x40E0002C,__WRITE      ,__gpcr2_bits);
__IO_REG32_BIT(GPCR3,                 0x40E00124,__WRITE      ,__gpcr3_bits);
__IO_REG32_BIT(GRER0,                 0x40E00030,__READ_WRITE ,__grer0_bits);
__IO_REG32_BIT(GRER1,                 0x40E00034,__READ_WRITE ,__grer1_bits);
__IO_REG32_BIT(GRER2,                 0x40E00038,__READ_WRITE ,__grer2_bits);
__IO_REG32_BIT(GRER3,                 0x40E00130,__READ_WRITE ,__grer3_bits);
__IO_REG32_BIT(GSRER0,                0x40E00440,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GSRER1,                0x40E00444,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GSRER2,                0x40E00448,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GSRER3,                0x40E0044C,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GCRER0,                0x40E00460,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GCRER1,                0x40E00464,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GCRER2,                0x40E00468,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GCRER3,                0x40E0046C,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GFER0,                 0x40E0003C,__READ_WRITE ,__gfer0_bits);
__IO_REG32_BIT(GFER1,                 0x40E00040,__READ_WRITE ,__gfer1_bits);
__IO_REG32_BIT(GFER2,                 0x40E00044,__READ_WRITE ,__gfer2_bits);
__IO_REG32_BIT(GFER3,                 0x40E0013C,__READ_WRITE ,__gfer3_bits);
__IO_REG32_BIT(GSFER0,                0x40E00480,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GSFER1,                0x40E00484,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GSFER2,                0x40E00488,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GSFER3,                0x40E0048C,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GCFER0,                0x40E004A0,__WRITE      ,__gpdr0_bits);
__IO_REG32_BIT(GCFER1,                0x40E004A4,__WRITE      ,__gpdr1_bits);
__IO_REG32_BIT(GCFER2,                0x40E004A8,__WRITE      ,__gpdr2_bits);
__IO_REG32_BIT(GCFER3,                0x40E004AC,__WRITE      ,__gpdr3_bits);
__IO_REG32_BIT(GEDR0,                 0x40E00048,__READ_WRITE ,__gedr0_bits);
__IO_REG32_BIT(GEDR1,                 0x40E0004C,__READ_WRITE ,__gedr1_bits);
__IO_REG32_BIT(GEDR2,                 0x40E00050,__READ_WRITE ,__gedr2_bits);
__IO_REG32_BIT(GEDR3,                 0x40E00148,__READ_WRITE ,__gedr3_bits);

/***************************************************************************
 **
 ** SERCCU (Services Clock Control Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(OSCC,                  0x41350000,__READ_WRITE ,__oscc_bits);

/***************************************************************************
 **
 ** SLACCU (Slave Clock Control Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(ACCR,                  0x41340000,__READ_WRITE ,__accr_bits);
__IO_REG32_BIT(ACSR,                  0x41340004,__READ       ,__acsr_bits);
__IO_REG32_BIT(AICSR,                 0x41340008,__READ_WRITE ,__aicsr_bits);
__IO_REG32_BIT(D0CKEN_A,              0x4134000C,__READ_WRITE ,__d0cken_a_bits);
__IO_REG32_BIT(D0CKEN_B,              0x41340010,__READ_WRITE ,__d0cken_b_bits);

/***************************************************************************
 **
 ** SERPMU (Services Power Management Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(PMCR,                  0x40F50000,__READ_WRITE ,__pmcr_bits);
__IO_REG32_BIT(PSR,                   0x40F50004,__READ_WRITE ,__psr_bits);
__IO_REG32(    PSPR,                  0x40F50008,__READ_WRITE );
__IO_REG32_BIT(PCFR,                  0x40F5000C,__READ_WRITE ,__pcfr_bits);
__IO_REG32_BIT(PWER,                  0x40F50010,__READ_WRITE ,__pwer_bits);
__IO_REG32_BIT(PWSR,                  0x40F50014,__READ_WRITE ,__pwsr_bits);
__IO_REG32_BIT(PECR,                  0x40F50018,__READ_WRITE ,__pecr_bits);
__IO_REG32_BIT(PVCR,                  0x40F50100,__READ_WRITE ,__pvcr_bits);

/***************************************************************************
 **
 ** SLAPMU (Slave Power Management Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(ASCR,                  0x40F40000,__READ_WRITE ,__ascr_bits);
__IO_REG32_BIT(ARSR,                  0x40F40004,__READ_WRITE ,__arsr_bits);
__IO_REG32_BIT(AD3ER,                 0x40F40008,__READ_WRITE ,__ad3er_bits);
__IO_REG32_BIT(AD3SR,                 0x40F4000C,__READ_WRITE ,__ad3sr_bits);
__IO_REG32_BIT(AD2D0ER,               0x40F40010,__READ_WRITE ,__ad3er_bits);
__IO_REG32_BIT(AD2D0SR,               0x40F40014,__READ_WRITE ,__ad3sr_bits);
__IO_REG32_BIT(AD2D1ER,               0x40F40018,__READ_WRITE ,__ad2d1er_bits);
__IO_REG32_BIT(AD2D1SR,               0x40F4001C,__READ_WRITE ,__ad2d1sr_bits);
__IO_REG32_BIT(AD1D0ER,               0x40F40020,__READ_WRITE ,__ad3er_bits);
__IO_REG32_BIT(AD1D0SR,               0x40F40024,__READ_WRITE ,__ad3sr_bits);
__IO_REG32(    AGENP,                 0x40F4002C,__READ_WRITE );
__IO_REG32_BIT(AD3R,                  0x40F40030,__READ_WRITE ,__ad3r_bits);
__IO_REG32_BIT(AD2R,                  0x40F40034,__READ_WRITE ,__ad2r_bits);
__IO_REG32_BIT(AD1R,                  0x40F40038,__READ_WRITE ,__ad1r_bits);

/***************************************************************************
 **
 ** 1-Wire
 **
 ***************************************************************************/
__IO_REG32_BIT(W1CMDR,                0x41B00000,__READ_WRITE ,__w1cmdr_bits);
__IO_REG32_BIT(W1TRR,                 0x41B00004,__READ_WRITE ,__w1trr_bits);
__IO_REG32_BIT(W1INTR,                0x41B00008,__READ       ,__w1intr_bits);
__IO_REG32_BIT(W1IER,                 0x41B0000C,__READ_WRITE ,__w1ier_bits);
__IO_REG32_BIT(W1CDR,                 0x41B00010,__READ_WRITE ,__w1cdr_bits);

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
__IO_REG32_BIT(DCSR16,                0x40000040,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR17,                0x40000044,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR18,                0x40000048,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR19,                0x4000004C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR20,                0x40000050,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR21,                0x40000054,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR22,                0x40000058,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR23,                0x4000005C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR24,                0x40000060,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR25,                0x40000064,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR26,                0x40000068,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR27,                0x4000006C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR28,                0x40000070,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR29,                0x40000074,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR30,                0x40000078,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DCSR31,                0x4000007C,__READ_WRITE ,__dcsr_bits);
__IO_REG32_BIT(DALGN,                 0x400000A0,__READ_WRITE ,__dalgn_bits);
__IO_REG32_BIT(DPCSR,                 0x400000A4,__READ_WRITE ,__dpcsr_bits);
__IO_REG32_BIT(DRQSR0,                0x400000E0,__READ_WRITE ,__drqsr0_bits);
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
__IO_REG32_BIT(DRCMR15,               0x4000013C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR16,               0x40000140,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR19,               0x4000014C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR20,               0x40000150,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR21,               0x40000154,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR22,               0x40000158,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR24,               0x40000160,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR25,               0x40000164,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR26,               0x40000168,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR27,               0x4000016C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR28,               0x40000170,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR29,               0x40000174,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR30,               0x40000178,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR31,               0x4000017C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR32,               0x40000180,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR33,               0x40000184,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR34,               0x40000188,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR35,               0x4000018C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR36,               0x40000190,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR37,               0x40000194,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR38,               0x40000198,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR39,               0x4000019C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR40,               0x400001A0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR41,               0x400001A4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR42,               0x400001A8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR43,               0x400001AC,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR44,               0x400001B0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR45,               0x400001B4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR46,               0x400001B8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR47,               0x400001BC,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR48,               0x400001C0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR49,               0x400001C4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR50,               0x400001C8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR51,               0x400001CC,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR52,               0x400001D0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR53,               0x400001D4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR54,               0x400001D8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR55,               0x400001DC,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR56,               0x400001E0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR57,               0x400001E4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR58,               0x400001E8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR59,               0x400001EC,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR60,               0x400001F0,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR61,               0x400001F4,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR62,               0x400001F8,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR63,               0x400001FC,__READ_WRITE ,__drcmr_bits);
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
__IO_REG32_BIT(DDADR16,               0x40000300,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR16,               0x40000304,__READ_WRITE );
__IO_REG32(    DTADR16,               0x40000308,__READ_WRITE );
__IO_REG32_BIT( DCMD16,               0x4000030C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR17,               0x40000310,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR17,               0x40000314,__READ_WRITE );
__IO_REG32(    DTADR17,               0x40000318,__READ_WRITE );
__IO_REG32_BIT( DCMD17,               0x4000031C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR18,               0x40000320,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR18,               0x40000324,__READ_WRITE );
__IO_REG32(    DTADR18,               0x40000328,__READ_WRITE );
__IO_REG32_BIT( DCMD18,               0x4000032C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR19,               0x40000330,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR19,               0x40000334,__READ_WRITE );
__IO_REG32(    DTADR19,               0x40000338,__READ_WRITE );
__IO_REG32_BIT( DCMD19,               0x4000033C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR20,               0x40000340,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR20,               0x40000344,__READ_WRITE );
__IO_REG32(    DTADR20,               0x40000348,__READ_WRITE );
__IO_REG32_BIT( DCMD20,               0x4000034C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR21,               0x40000350,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR21,               0x40000354,__READ_WRITE );
__IO_REG32(    DTADR21,               0x40000358,__READ_WRITE );
__IO_REG32_BIT( DCMD21,               0x4000035C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR22,               0x40000360,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR22,               0x40000364,__READ_WRITE );
__IO_REG32(    DTADR22,               0x40000368,__READ_WRITE );
__IO_REG32_BIT( DCMD22,               0x4000036C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR23,               0x40000370,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR23,               0x40000374,__READ_WRITE );
__IO_REG32(    DTADR23,               0x40000378,__READ_WRITE );
__IO_REG32_BIT( DCMD23,               0x4000037C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR24,               0x40000380,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR24,               0x40000384,__READ_WRITE );
__IO_REG32(    DTADR24,               0x40000388,__READ_WRITE );
__IO_REG32_BIT( DCMD24,               0x4000038C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR25,               0x40000390,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR25,               0x40000394,__READ_WRITE );
__IO_REG32(    DTADR25,               0x40000398,__READ_WRITE );
__IO_REG32_BIT( DCMD25,               0x4000039C,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR26,               0x400003A0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR26,               0x400003A4,__READ_WRITE );
__IO_REG32(    DTADR26,               0x400003A8,__READ_WRITE );
__IO_REG32_BIT( DCMD26,               0x400003AC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR27,               0x400003B0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR27,               0x400003B4,__READ_WRITE );
__IO_REG32(    DTADR27,               0x400003B8,__READ_WRITE );
__IO_REG32_BIT( DCMD27,               0x400003BC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR28,               0x400003C0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR28,               0x400003C4,__READ_WRITE );
__IO_REG32(    DTADR28,               0x400003C8,__READ_WRITE );
__IO_REG32_BIT( DCMD28,               0x400003CC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR29,               0x400003D0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(     DSADR29,               0x400003D4,__READ_WRITE );
__IO_REG32(     DTADR29,               0x400003D8,__READ_WRITE );
__IO_REG32_BIT( DCMD29,               0x400003DC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR30,               0x400003E0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(     DSADR30,               0x400003E4,__READ_WRITE );
__IO_REG32(    DTADR30,               0x400003E8,__READ_WRITE );
__IO_REG32_BIT( DCMD30,               0x400003EC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DDADR31,               0x400003F0,__READ_WRITE ,__ddadr_bits);
__IO_REG32(    DSADR31,               0x400003F4,__READ_WRITE );
__IO_REG32(    DTADR31,               0x400003F8,__READ_WRITE );
__IO_REG32_BIT( DCMD31,               0x400003FC,__READ_WRITE ,__dcmd_bits);
__IO_REG32_BIT(DRCMR66,               0x40001108,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR67,               0x4000110C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR91,               0x4000116C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR92,               0x40001170,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR93,               0x40001174,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR94,               0x40001178,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR95,               0x4000117C,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR96,               0x40001180,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR97,               0x40001184,__READ_WRITE ,__drcmr_bits);
__IO_REG32_BIT(DRCMR99,               0x4000118C,__READ_WRITE ,__drcmr_bits);

/***************************************************************************
 **
 ** IC (Interrup controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(ICPR,                  0x40D00010,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICPR2,                 0x40D000AC,__READ       ,__icpr2_bits);
__IO_REG32_BIT(ICIP,                  0x40D00000,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICIP2,                 0x40D0009C,__READ       ,__icpr2_bits);
__IO_REG32_BIT(ICFP,                  0x40D0000C,__READ       ,__icpr_bits);
__IO_REG32_BIT(ICFP2,                 0x40D000A8,__READ       ,__icpr2_bits);
__IO_REG32_BIT(ICMR,                  0x40D00004,__READ_WRITE ,__icpr_bits);
__IO_REG32_BIT(ICMR2,                 0x40D000A0,__READ_WRITE ,__icpr2_bits);
__IO_REG32_BIT(ICLR,                  0x40D00008,__READ_WRITE ,__icpr_bits);
__IO_REG32_BIT(ICLR2,                 0x40D000A4,__READ_WRITE ,__icpr2_bits);
__IO_REG32_BIT(ICCR,                  0x40D00014,__READ_WRITE ,__iccr_bits);
__IO_REG32_BIT(IPR0,                  0x40D0001C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR1,                  0x40D00020,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR2,                  0x40D00024,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR3,                  0x40D00028,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR4,                  0x40D0002C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR5,                  0x40D00030,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR6,                  0x40D00034,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR7,                  0x40D00038,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR8,                  0x40D0003C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR9,                  0x40D00040,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR10,                 0x40D00044,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR11,                 0x40D00048,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR12,                 0x40D0004C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR13,                 0x40D00050,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR14,                 0x40D00054,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR15,                 0x40D00058,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR16,                 0x40D0005C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR17,                 0x40D00060,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR18,                 0x40D00064,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR19,                 0x40D00068,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR20,                 0x40D0006C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR21,                 0x40D00070,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR22,                 0x40D00074,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR23,                 0x40D00078,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR24,                 0x40D0007C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR25,                 0x40D00080,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR26,                 0x40D00084,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR27,                 0x40D00088,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR28,                 0x40D0008C,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR29,                 0x40D00090,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR30,                 0x40D00094,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR31,                 0x40D00098,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR32,                 0x40D000B0,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR33,                 0x40D000B4,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR34,                 0x40D000B8,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR35,                 0x40D000BC,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR36,                 0x40D000C0,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR37,                 0x40D000C4,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR38,                 0x40D000C8,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR39,                 0x40D000CC,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR40,                 0x40D000D0,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR41,                 0x40D000D4,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR42,                 0x40D000D8,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR43,                 0x40D000DC,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR44,                 0x40D000E0,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR45,                 0x40D000E4,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR46,                 0x40D000E8,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR47,                 0x40D000EC,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR48,                 0x40D000F0,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR49,                 0x40D000F4,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR50,                 0x40D000F8,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR51,                 0x40D000FC,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(IPR52,                 0x40D00100,__READ_WRITE ,__irp_bits);
__IO_REG32_BIT(ICHP,                  0x40D00018,__READ       ,__ichp_bits);

/***************************************************************************
 **
 ** RTC
 **
 ***************************************************************************/
__IO_REG32_BIT(RTTR,                  0x4090000C,__READ_WRITE ,__rttr_bits);
__IO_REG32_BIT(RTSR,                  0x40900008,__READ_WRITE ,__rtsr_bits);
__IO_REG32(    RTAR,                  0x40900004,__READ_WRITE );
__IO_REG32_BIT(RDAR1,                 0x40900018,__READ_WRITE ,__rdar_bits);
__IO_REG32_BIT(RDAR2,                 0x40900020,__READ_WRITE ,__rdar_bits);
__IO_REG32_BIT(RYAR1,                 0x4090001C,__READ_WRITE ,__ryar_bits);
__IO_REG32_BIT(RYAR2,                 0x40900024,__READ_WRITE ,__ryar_bits);
__IO_REG32_BIT(SWAR1,                 0x4090002C,__READ_WRITE ,__swar_bits);
__IO_REG32_BIT(SWAR2,                 0x40900030,__READ_WRITE ,__swar_bits);
__IO_REG32_BIT(PIAR,                  0x40900038,__READ_WRITE ,__piar_bits);
__IO_REG32(    RCNR,                  0x40900000,__READ_WRITE );
__IO_REG32_BIT(RDCR,                  0x40900010,__READ_WRITE ,__rdcr_bits);
__IO_REG32_BIT(RYCR,                  0x40900014,__READ_WRITE ,__rycr_bits);
__IO_REG32_BIT(SWCR,                  0x40900028,__READ_WRITE ,__swcr_bits);
__IO_REG32_BIT(RTCPICR,               0x40900034,__READ_WRITE ,__rtcpicr_bits);

/***************************************************************************
 **
 ** OST (Operating System Timers)
 **
 ***************************************************************************/
__IO_REG32_BIT(OMCR4,                 0x40A000C0,__READ_WRITE ,__omcr_bits);
__IO_REG32_BIT(OMCR5,                 0x40A000C4,__READ_WRITE ,__omcr_bits);
__IO_REG32_BIT(OMCR6,                 0x40A000C8,__READ_WRITE ,__omcr_bits);
__IO_REG32_BIT(OMCR7,                 0x40A000CC,__READ_WRITE ,__omcr_bits);
__IO_REG32_BIT(OMCR8,                 0x40A000D0,__READ_WRITE ,__omcr8_bits);
__IO_REG32_BIT(OMCR9,                 0x40A000D4,__READ_WRITE ,__omcr9_bits);
__IO_REG32_BIT(OMCR10,                0x40A000D8,__READ_WRITE ,__omcr8_bits);
__IO_REG32_BIT(OMCR11,                0x40A000DC,__READ_WRITE ,__omcr9_bits);
__IO_REG32(    OSMR0,                 0x40A00000,__READ_WRITE );
__IO_REG32(    OSMR1,                 0x40A00004,__READ_WRITE );
__IO_REG32(    OSMR2,                 0x40A00008,__READ_WRITE );
__IO_REG32(    OSMR3,                 0x40A0000C,__READ_WRITE );
__IO_REG32(    OSMR4,                 0x40A00080,__READ_WRITE );
__IO_REG32(    OSMR5,                 0x40A00084,__READ_WRITE );
__IO_REG32(    OSMR6,                 0x40A00088,__READ_WRITE );
__IO_REG32(    OSMR7,                 0x40A0008C,__READ_WRITE );
__IO_REG32(    OSMR8,                 0x40A00090,__READ_WRITE );
__IO_REG32(    OSMR9,                 0x40A00094,__READ_WRITE );
__IO_REG32(    OSMR10,                0x40A00098,__READ_WRITE );
__IO_REG32(    OSMR11,                0x40A0009C,__READ_WRITE );
__IO_REG32_BIT(OWER,                  0x40A00018,__READ_WRITE ,__ower_bits);
__IO_REG32_BIT(OIER,                  0x40A0001C,__READ_WRITE ,__oier_bits);
__IO_REG32(    OSCR0,                 0x40A00010,__READ_WRITE );
__IO_REG32(    OSCR4,                 0x40A00040,__READ_WRITE );
__IO_REG32(    OSCR5,                 0x40A00044,__READ_WRITE );
__IO_REG32(    OSCR6,                 0x40A00048,__READ_WRITE );
__IO_REG32(    OSCR7,                 0x40A0004C,__READ_WRITE );
__IO_REG32(    OSCR8,                 0x40A00050,__READ_WRITE );
__IO_REG32(    OSCR9,                 0x40A00054,__READ_WRITE );
__IO_REG32(    OSCR10,                0x40A00058,__READ_WRITE );
__IO_REG32(    OSCR11,                0x40A0005C,__READ_WRITE );
__IO_REG32_BIT(OSSR,                  0x40A00014,__READ_WRITE ,__ossr_bits);
__IO_REG32(    OSNR,                  0x40A00020,__READ       );

/***************************************************************************
 **
 ** DBG
 **
 ***************************************************************************/
__IO_REG32_BIT(PML_ESEL_0,            0x4600FF00,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_1,            0x4600FF04,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_2,            0x4600FF08,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_3,            0x4600FF0C,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_4,            0x4600FF10,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_5,            0x4600FF14,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_6,            0x4600FF18,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(PML_ESEL_7,            0x4600FF1C,__WRITE      ,__pml_esel_bits);
__IO_REG32_BIT(MDU_XSCALE_BP,         0x4600FF40,__WRITE      ,__mdu_xscale_bp_bits);
__IO_REG32_BIT(MDU_2DG_EVENT,         0x4600FF54,__WRITE      ,__mdu_xscale_bp_bits);
__IO_REG32_BIT(MDU_CW_MATCH,          0x4600FF58,__WRITE      ,__mdu_xscale_bp_bits);

/***************************************************************************
 **
 ** SBA (System Bus Arbiters)
 **
 ***************************************************************************/
__IO_REG32_BIT(ARB_CNTRL_1,           0x4600FE00,__READ_WRITE ,__arb_cntr1_bits);
__IO_REG32_BIT(ARB_CNTRL_2,           0x4600FE80,__READ_WRITE ,__arb_cntr2_bits);

/***************************************************************************
 **
 ** DMC ( Dynamic Memory Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(MDCNFG,                0x48100000,__READ_WRITE ,__mdcnfg_bits  );
__IO_REG32_BIT(MDREFR,                0x48100004,__READ_WRITE ,__mdrefr_bits  );
__IO_REG32_BIT(MDMRS,                 0x48100040,__READ_WRITE ,__mdmrs_bits   );
__IO_REG32_BIT(DDR_HCAL,              0x48100060,__READ_WRITE ,__ddr_hcal_bits);
__IO_REG32_BIT(DDR_WCAL,              0x48100068,__READ_WRITE ,__ddr_wcal_bits);
__IO_REG32_BIT(DMCIER,                0x48100070,__READ_WRITE ,__dmcier_bits  );
__IO_REG32_BIT(DMCISR,                0x48100078,__READ_WRITE ,__dmcisr_bits  );
__IO_REG32_BIT(DDR_DLS,               0x48100080,__READ       ,__ddr_dls_bits );
__IO_REG32_BIT(EMPI,                  0x48100090,__READ_WRITE ,__empi_bits    );
__IO_REG32_BIT(RCOMP,                 0x48100100,__READ_WRITE ,__rcomp_bits   );

__IO_REG32_BIT(PAD_MA,                0x48100110,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_MDMSB,             0x48100114,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_MDLSB,             0x48100118,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_SDRAM,             0x4810011C,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_SDCLK,             0x48100120,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_SDCS,              0x48100124,__READ_WRITE ,__pad_ma_bits  );
__IO_REG32_BIT(PAD_SCLK,              0x4810012C,__READ_WRITE ,__pad_ma_bits  );

/***************************************************************************
 **
 ** SMC ( Static Memory Controller )
 **
 ***************************************************************************/
__IO_REG32_BIT(MSC1,                  0x4A00000C,__READ_WRITE ,__msc1_bits      );
__IO_REG32_BIT(MECR,                  0x4A000014,__READ_WRITE ,__mecr_bits      );
__IO_REG32_BIT(SXCNFG,                0x4A00001C,__READ_WRITE ,__sxcnfg_bits    );
__IO_REG32_BIT(MCMEM0,                0x4A000028,__READ_WRITE ,__mc_x_bits      );
__IO_REG32_BIT(MCATT0,                0x4A000030,__READ_WRITE ,__mc_x_bits      );
__IO_REG32_BIT(MCIO0,                 0x4A000038,__READ_WRITE ,__mc_x_bits      );
__IO_REG32_BIT(MEMCLKCFG,             0x4A000068,__READ_WRITE ,__memclkcfg_bits );
__IO_REG32_BIT(CSADRCFG0,             0x4A000080,__READ_WRITE ,__csadrcfgx_bits );
__IO_REG32_BIT(CSADRCFG1,             0x4A000084,__READ_WRITE ,__csadrcfgx_bits );
__IO_REG32_BIT(CSADRCFG2,             0x4A000088,__READ_WRITE ,__csadrcfgx_bits );
__IO_REG32_BIT(CSADRCFG3,             0x4A00008C,__READ_WRITE ,__csadrcfgx_bits );
__IO_REG32_BIT(CSADRCFG_P,            0x4A000090,__READ_WRITE ,__csadrcfgx_bits );
__IO_REG32(    CSMSADRCFG,            0x4A0000A0,__READ_WRITE                   );

/***************************************************************************
 **
 ** DFC ( Data Flash Controller )
 **
 ***************************************************************************/
__IO_REG32_BIT(NDCR,                  0x43100000,__READ_WRITE ,__ndcr_bits    );
__IO_REG32_BIT(NDTR0CS0,              0x43100004,__READ_WRITE ,__ndtr0cs0_bits);
__IO_REG32_BIT(NDTR1CS0,              0x4310000C,__READ_WRITE ,__ndtr1cs0_bits);
__IO_REG32_BIT(NDSR,                  0x43100014,__READ_WRITE ,__ndsr_bits    );
__IO_REG32_BIT(NDPCR,                 0x43100018,__READ       ,__ndpcr_bits   );
__IO_REG32(    NDBBR0,                0x4310001C,__READ                       );
__IO_REG32(    NDBBR1,                0x43100020,__READ                       );
__IO_REG32(    NDDB,                  0x43100040,__READ_WRITE                 );
__IO_REG32_BIT(NDCB0,                 0x43100048,__READ_WRITE ,__ndcb0_bits   );
__IO_REG32_BIT(NDCB1,                 0x4310004C,__READ       ,__ndcb1_bits   );
__IO_REG32_BIT(NDCB2,                 0x43100050,__READ       ,__ndcb2_bits   );

/***************************************************************************
 **
 ** IM ( Internal Memory )
 **
 ***************************************************************************/
__IO_REG32_BIT(IMPMCR,                0x58000000,__READ_WRITE ,__impmcr_bits  );

/***************************************************************************
 **
 ** MMC1 
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC1_STRPCL,            0x41100000,__READ_WRITE ,__mmc_strpcl_bits  );
__IO_REG32_BIT(MMC1_STAT,              0x41100004,__READ       ,__mmc_stat_bits    );
__IO_REG32_BIT(MMC1_CLKRT,             0x41100008,__READ_WRITE ,__mmc_clkrt_bits   );
__IO_REG32_BIT(MMC1_SPI,               0x4110000C,__READ_WRITE ,__mmc_spi_bits     );
__IO_REG32_BIT(MMC1_CMDAT,             0x41100010,__READ_WRITE ,__mmc_cmdat_bits   );
__IO_REG32_BIT(MMC1_RESTO,             0x41100014,__READ_WRITE ,__mmc_resto_bits   );
__IO_REG32_BIT(MMC1_RDTO,              0x41100018,__READ_WRITE ,__mmc_rdto_bits    );
__IO_REG32_BIT(MMC1_BLKLEN,            0x4110001C,__READ_WRITE ,__mmc_blklen_bits  );
__IO_REG32_BIT(MMC1_NUMBLK,            0x41100020,__READ_WRITE ,__mmc_numblk_bits  );
__IO_REG32_BIT(MMC1_PRTBUF,            0x41100024,__READ_WRITE ,__mmc_prtbuf_bits  );
__IO_REG32_BIT(MMC1_I_MASK,            0x41100028,__READ_WRITE ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC1_I_REG,             0x4110002C,__READ       ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC1_CMD,               0x41100030,__READ_WRITE ,__mmc_cmd_bits     );
__IO_REG32_BIT(MMC1_ARGH,              0x41100034,__READ_WRITE ,__mmc_argh_bits    );
__IO_REG32_BIT(MMC1_ARGL,              0x41100038,__READ_WRITE ,__mmc_argl_bits    );
__IO_REG32_BIT(MMC1_RES,               0x4110003C,__READ       ,__mmc_res_bits     );
__IO_REG32_BIT(MMC1_RXFIFO,            0x41100040,__READ       ,__mmc_rxfifo_bits  );
__IO_REG32_BIT(MMC1_TXFIFO,            0x41100044,__WRITE      ,__mmc_txfifo_bits  );
__IO_REG32_BIT(MMC1_RDWAIT,            0x41100048,__READ_WRITE ,__mmc_rdwait_bits  );
__IO_REG32_BIT(MMC1_BLKS_REM,          0x4110004C,__READ_WRITE ,__mmc_blks_rem_bits);

/***************************************************************************
 **
 ** MMC2 
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC2_STRPCL,            0x42000000,__READ_WRITE ,__mmc_strpcl_bits  );
__IO_REG32_BIT(MMC2_STAT,              0x42000004,__READ       ,__mmc_stat_bits    );
__IO_REG32_BIT(MMC2_CLKRT,             0x42000008,__READ_WRITE ,__mmc_clkrt_bits   );
__IO_REG32_BIT(MMC2_SPI,               0x4200000C,__READ_WRITE ,__mmc_spi_bits     );
__IO_REG32_BIT(MMC2_CMDAT,             0x42000010,__READ_WRITE ,__mmc_cmdat_bits   );
__IO_REG32_BIT(MMC2_RESTO,             0x42000014,__READ_WRITE ,__mmc_resto_bits   );
__IO_REG32_BIT(MMC2_RDTO,              0x42000018,__READ_WRITE ,__mmc_rdto_bits    );
__IO_REG32_BIT(MMC2_BLKLEN,            0x4200001C,__READ_WRITE ,__mmc_blklen_bits  );
__IO_REG32_BIT(MMC2_NUMBLK,            0x42000020,__READ_WRITE ,__mmc_numblk_bits  );
__IO_REG32_BIT(MMC2_PRTBUF,            0x42000024,__READ_WRITE ,__mmc_prtbuf_bits  );
__IO_REG32_BIT(MMC2_I_MASK,            0x42000028,__READ_WRITE ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC2_I_REG,             0x4200002C,__READ       ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC2_CMD,               0x42000030,__READ_WRITE ,__mmc_cmd_bits     );
__IO_REG32_BIT(MMC2_ARGH,              0x42000034,__READ_WRITE ,__mmc_argh_bits    );
__IO_REG32_BIT(MMC2_ARGL,              0x42000038,__READ_WRITE ,__mmc_argl_bits    );
__IO_REG32_BIT(MMC2_RES,               0x4200003C,__READ       ,__mmc_res_bits     );
__IO_REG32_BIT(MMC2_RXFIFO,            0x42000040,__READ       ,__mmc_rxfifo_bits  );
__IO_REG32_BIT(MMC2_TXFIFO,            0x42000044,__WRITE      ,__mmc_txfifo_bits  );
__IO_REG32_BIT(MMC2_RDWAIT,            0x42000048,__READ_WRITE ,__mmc_rdwait_bits  );
__IO_REG32_BIT(MMC2_BLKS_REM,          0x4200004C,__READ_WRITE ,__mmc_blks_rem_bits);

/***************************************************************************
 **
 ** MMC3
 **
 ***************************************************************************/
__IO_REG32_BIT(MMC3_STRPCL,            0x42500000,__READ_WRITE ,__mmc_strpcl_bits  );
__IO_REG32_BIT(MMC3_STAT,              0x42500004,__READ       ,__mmc_stat_bits    );
__IO_REG32_BIT(MMC3_CLKRT,             0x42500008,__READ_WRITE ,__mmc_clkrt_bits   );
__IO_REG32_BIT(MMC3_SPI,               0x4250000C,__READ_WRITE ,__mmc_spi_bits     );
__IO_REG32_BIT(MMC3_CMDAT,             0x42500010,__READ_WRITE ,__mmc_cmdat_bits   );
__IO_REG32_BIT(MMC3_RESTO,             0x42500014,__READ_WRITE ,__mmc_resto_bits   );
__IO_REG32_BIT(MMC3_RDTO,              0x42500018,__READ_WRITE ,__mmc_rdto_bits    );
__IO_REG32_BIT(MMC3_BLKLEN,            0x4250001C,__READ_WRITE ,__mmc_blklen_bits  );
__IO_REG32_BIT(MMC3_NUMBLK,            0x42500020,__READ_WRITE ,__mmc_numblk_bits  );
__IO_REG32_BIT(MMC3_PRTBUF,            0x42500024,__READ_WRITE ,__mmc_prtbuf_bits  );
__IO_REG32_BIT(MMC3_I_MASK,            0x42500028,__READ_WRITE ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC3_I_REG,             0x4250002C,__READ       ,__mmc_i_mask_bits  );
__IO_REG32_BIT(MMC3_CMD,               0x42500030,__READ_WRITE ,__mmc_cmd_bits     );
__IO_REG32_BIT(MMC3_ARGH,              0x42500034,__READ_WRITE ,__mmc_argh_bits    );
__IO_REG32_BIT(MMC3_ARGL,              0x42500038,__READ_WRITE ,__mmc_argl_bits    );
__IO_REG32_BIT(MMC3_RES,               0x4250003C,__READ       ,__mmc_res_bits     );
__IO_REG32_BIT(MMC3_RXFIFO,            0x42500040,__READ       ,__mmc_rxfifo_bits  );
__IO_REG32_BIT(MMC3_TXFIFO,            0x42500044,__WRITE      ,__mmc_txfifo_bits  );
__IO_REG32_BIT(MMC3_RDWAIT,            0x42500048,__READ_WRITE ,__mmc_rdwait_bits  );
__IO_REG32_BIT(MMC3_BLKS_REM,          0x4250004C,__READ_WRITE ,__mmc_blks_rem_bits);

/***************************************************************************
 **
 ** LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(LCCR0,                 0x44000000,__READ_WRITE ,__lccr0_bits);
__IO_REG32_BIT(LCCR1,                 0x44000004,__READ_WRITE ,__lccr1_bits);
__IO_REG32_BIT(LCCR2,                 0x44000008,__READ_WRITE ,__lccr2_bits);
__IO_REG32_BIT(LCCR3,                 0x4400000C,__READ_WRITE ,__lccr3_bits);
__IO_REG32_BIT(LCCR4,                 0x44000010,__READ_WRITE ,__lccr4_bits);
__IO_REG32_BIT(LCCR5,                 0x44000014,__READ_WRITE ,__lccr5_bits);
__IO_REG32_BIT(LCCR6,                 0x44000018,__READ_WRITE ,__lccr6_bits);
__IO_REG32_BIT(OVL1C1,                0x44000050,__READ_WRITE ,__ovl1c1_bits);
__IO_REG32_BIT(OVL1C2,                0x44000060,__READ_WRITE ,__ovl1c2_bits);
__IO_REG32_BIT(OVL2C1,                0x44000070,__READ_WRITE ,__ovl2c1_bits);
__IO_REG32_BIT(OVL2C2,                0x44000080,__READ_WRITE ,__ovl2c2_bits);
__IO_REG32_BIT(CCR,                   0x44000090,__READ_WRITE ,__ccr_bits);
__IO_REG32_BIT(CMDCR,                 0x44000100,__READ_WRITE ,__cmdcr_bits);
__IO_REG32_BIT(TRGBR,                 0x44000040,__READ_WRITE ,__trgbr_bits);
__IO_REG32_BIT(TCR,                   0x44000044,__READ_WRITE ,__tcr_bits);
__IO_REG32_BIT(FBR0,                  0x44000020,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR1,                  0x44000024,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR2,                  0x44000028,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR3,                  0x4400002C,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR4,                  0x44000030,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR5,                  0x44000110,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(FBR6,                  0x44000114,__READ_WRITE ,__fbr_bits);
__IO_REG32_BIT(PRSR,                  0x44000104,__READ_WRITE ,__prsr_bits);
__IO_REG32_BIT(LCSR0,                 0x44000038,__READ_WRITE ,__lcsr0_bits);
__IO_REG32_BIT(LCSR1,                 0x44000034,__READ_WRITE ,__lcsr1_bits);
__IO_REG32(    LIIDR,                 0x4400003C,__READ       );
__IO_REG32(    FDADR0,                0x44000200,__READ_WRITE );
__IO_REG32(    FSADR0,                0x44000204,__READ_WRITE );
__IO_REG32(    FIDR0,                 0x44000208,__READ       );
__IO_REG32_BIT(LDCMD0,                0x4400020C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR1,                0x44000210,__READ_WRITE );
__IO_REG32(    FSADR1,                0x44000214,__READ_WRITE );
__IO_REG32(    FIDR1,                 0x44000218,__READ       );
__IO_REG32_BIT(LDCMD1,                0x4400021C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR2,                0x44000220,__READ_WRITE );
__IO_REG32(    FSADR2,                0x44000224,__READ_WRITE );
__IO_REG32(    FIDR2,                 0x44000228,__READ       );
__IO_REG32_BIT(LDCMD2,                0x4400022C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR3,                0x44000230,__READ_WRITE );
__IO_REG32(    FSADR3,                0x44000234,__READ_WRITE );
__IO_REG32(    FIDR3,                 0x44000238,__READ       );
__IO_REG32_BIT(LDCMD3,                0x4400023C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR4,                0x44000240,__READ_WRITE );
__IO_REG32(    FSADR4,                0x44000244,__READ_WRITE );
__IO_REG32(    FIDR4,                 0x44000248,__READ       );
__IO_REG32_BIT(LDCMD4,                0x4400024C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR5,                0x44000250,__READ_WRITE );
__IO_REG32(    FSADR5,                0x44000254,__READ_WRITE );
__IO_REG32(    FIDR5,                 0x44000258,__READ       );
__IO_REG32_BIT(LDCMD5,                0x4400025C,__READ       ,__ldcmd_bits);
__IO_REG32(    FDADR6,                0x44000260,__READ_WRITE );
__IO_REG32(    FSADR6,                0x44000264,__READ_WRITE );
__IO_REG32(    FIDR6,                 0x44000268,__READ       );
__IO_REG32_BIT(LDCMD6,                0x4400026C,__READ       ,__ldcmd_bits);

/***************************************************************************
 **
 ** Mini-LCDC
 **
 ***************************************************************************/
__IO_REG32_BIT(MLCCR0,                0x46000000,__READ_WRITE ,__mlccr0_bits);
__IO_REG32_BIT(MLCCR1,                0x46000004,__READ_WRITE ,__mlccr1_bits);
__IO_REG32_BIT(MLCCR2,                0x46000008,__READ_WRITE ,__mlccr2_bits);
__IO_REG32(    MLSADD,                0x4600000C,__READ_WRITE );
__IO_REG32_BIT(MLFRMCNT,              0x46000010,__READ_WRITE ,__mlfrmcnt_bits);

/***************************************************************************
 **
 ** QCI (Quick Capture Interface)
 **
 ***************************************************************************/
__IO_REG32_BIT(CICR0,                 0x50000000,__READ_WRITE ,__cicr0_bits);
__IO_REG32_BIT(CICR1,                 0x50000004,__READ_WRITE ,__cicr1_bits);
__IO_REG32_BIT(CICR2,                 0x50000008,__READ_WRITE ,__cicr2_bits);
__IO_REG32_BIT(CICR3,                 0x5000000C,__READ_WRITE ,__cicr3_bits);
__IO_REG32_BIT(CICR4,                 0x50000010,__READ_WRITE ,__cicr4_bits);
__IO_REG32(    CITOR,                 0x5000001C,__READ_WRITE );
__IO_REG32_BIT(CISR,                  0x50000014,__READ_WRITE ,__cisr_bits);
__IO_REG32_BIT(CIRCD,                 0x50000044,__READ_WRITE ,__circd_bits);
__IO_REG32_BIT(CIBR0,                 0x50000028,__READ       ,__cibr_bits);
__IO_REG32_BIT(CIBR1,                 0x50000030,__READ       ,__cibr_bits);
__IO_REG32_BIT(CIBR2,                 0x50000038,__READ       ,__cibr_bits);
__IO_REG32_BIT(CIBR3,                 0x50000040,__READ       ,__cibr_bits);
__IO_REG32_BIT(CIPSS,                 0x50000064,__READ_WRITE ,__cipss_bits);
__IO_REG32_BIT(CIPBUF,                0x50000068,__READ_WRITE ,__cipbuf_bits);
__IO_REG32_BIT(CIHST,                 0x5000006C,__READ_WRITE ,__cihst_bits);
__IO_REG32(    CISUM,                 0x50000070,__READ_WRITE );
__IO_REG32_BIT(CICCR,                 0x50000074,__READ_WRITE ,__ciccr_bits);
__IO_REG32_BIT(CISSC,                 0x5000007C,__READ_WRITE ,__cissc_bits);
__IO_REG32_BIT(CICMR,                 0x50000090,__READ_WRITE ,__cicmr_bits);
__IO_REG32_BIT(CICMC0,                0x50000094,__READ_WRITE ,__cicmc0_bits);
__IO_REG32_BIT(CICMC1,                0x50000098,__READ_WRITE ,__cicmc1_bits);
__IO_REG32_BIT(CICMC2,                0x5000009C,__READ_WRITE ,__cicmc2_bits);
__IO_REG32_BIT(CIFSR,                 0x500000C0,__READ_WRITE ,__cifsr_bits);
__IO_REG32_BIT(CIFR0,                 0x500000B0,__READ_WRITE ,__cifr0_bits);
__IO_REG32_BIT(CIFR1,                 0x500000B4,__READ_WRITE ,__cifr1_bits);
__IO_REG32(    CIDADR0,               0x50000240,__READ_WRITE );
__IO_REG32(    CISADR0,               0x50000244,__READ       );
__IO_REG32(    CITSADR0,              0x50000248,__READ       );
__IO_REG32_BIT(CICMD0,                0x5000024C,__READ       ,__cicmd_bits);
__IO_REG32(    CIDADR1,               0x50000250,__READ_WRITE );
__IO_REG32(    CISADR1,               0x50000254,__READ       );
__IO_REG32(    CITSADR1,              0x50000258,__READ       );
__IO_REG32_BIT(CICMD1,                0x5000025C,__READ       ,__cicmd_bits);
__IO_REG32(    CIDADR2,               0x50000260,__READ_WRITE );
__IO_REG32(    CISADR2,               0x50000264,__READ       );
__IO_REG32(    CITSADR2,              0x50000268,__READ       );
__IO_REG32_BIT(CICMD2,                0x5000026C,__READ       ,__cicmd_bits);
__IO_REG32(    CIDADR3,               0x50000270,__READ_WRITE );
__IO_REG32(    CISADR3,               0x50000274,__READ       );
__IO_REG32(    CITSADR3,              0x50000278,__READ       );
__IO_REG32_BIT(CICMD3,                0x5000027C,__READ       ,__cicmd_bits);
__IO_REG32_BIT(CIDBR0,                0x50000220,__READ_WRITE ,__cidbr_bits);
__IO_REG32_BIT(CIDBR1,                0x50000224,__READ_WRITE ,__cidbr_bits);
__IO_REG32_BIT(CIDBR2,                0x50000228,__READ_WRITE ,__cidbr_bits);
__IO_REG32_BIT(CIDBR3,                0x5000022C,__READ_WRITE ,__cidbr_bits);
__IO_REG32_BIT(CIDCSR0,               0x50000200,__READ_WRITE ,__cidcsr_bits);
__IO_REG32_BIT(CIDCSR1,               0x50000204,__READ_WRITE ,__cidcsr_bits);
__IO_REG32_BIT(CIDCSR2,               0x50000208,__READ_WRITE ,__cidcsr_bits);
__IO_REG32_BIT(CIDCSR3,               0x5000020C,__READ_WRITE ,__cidcsr_bits);

/***************************************************************************
 **
 ** GC (Graphics Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(GCCR,                  0x54000000,__READ_WRITE ,__gccr_bits);
__IO_REG32_BIT(GCISCR,                0x54000004,__READ_WRITE ,__gciscr_bits);
__IO_REG32_BIT(GCIECR,                0x54000008,__READ_WRITE ,__gciecr_bits);
__IO_REG32_BIT(GCNOPID,               0x5400000C,__READ_WRITE ,__gcnopid_bits);
__IO_REG32_BIT(GCALPHASET,            0x54000010,__READ_WRITE ,__gcalphaset_bits);
__IO_REG32_BIT(GCTSET,                0x54000014,__READ_WRITE ,__gctset_bits);
__IO_REG32(    GCRBBR,                0x54000020,__READ_WRITE );
__IO_REG32_BIT(GCRBLR,                0x54000024,__READ_WRITE ,__gcrblr_bits);
__IO_REG32(    GCRBHR,                0x54000028,__READ       );
__IO_REG32(    GCRBTR,                0x5400002C,__READ_WRITE );
__IO_REG32(    GCRBEXHR,              0x54000030,__READ       );
__IO_REG32(    GCBBBR,                0x54000040,__READ       );
__IO_REG32(    GCBBHR,                0x54000044,__READ       );
__IO_REG32(    GCBBEXHR,              0x54000048,__READ       );
__IO_REG32(    GCD0BR,                0x54000060,__READ_WRITE );
__IO_REG32_BIT(GCD0STP,               0x54000064,__READ_WRITE ,__gcdstp_bits);
__IO_REG32_BIT(GCD0STR,               0x54000068,__READ_WRITE ,__gcdstr_bits);
__IO_REG32_BIT(GCD0PF,                0x5400006C,__READ_WRITE ,__gcdpf_bits);
__IO_REG32(    GCD1BR,                0x54000070,__READ_WRITE );
__IO_REG32_BIT(GCD1STP,               0x54000074,__READ_WRITE ,__gcdstp_bits);
__IO_REG32_BIT(GCD1STR,               0x54000078,__READ_WRITE ,__gcdstr_bits);
__IO_REG32_BIT(GCD1PF,                0x5400007C,__READ_WRITE ,__gcdpf_bits);
__IO_REG32(    GCD2BR,                0x54000080,__READ_WRITE );
__IO_REG32_BIT(GCD2STP,               0x54000084,__READ_WRITE ,__gcdstp_bits);
__IO_REG32_BIT(GCD2STR,               0x54000088,__READ_WRITE ,__gcdstr_bits);
__IO_REG32_BIT(GCD2PF,                0x5400008C,__READ_WRITE ,__gcdpf_bits);
__IO_REG32(    GCS0BR,                0x540000E0,__READ_WRITE );
__IO_REG32_BIT(GCS0STP,               0x540000E4,__READ_WRITE ,__gcdstp_bits);
__IO_REG32_BIT(GCS0STR,               0x540000E8,__READ_WRITE ,__gcdstr_bits);
__IO_REG32_BIT(GCS0PF,                0x540000EC,__READ_WRITE ,__gcdpf_bits);
__IO_REG32(    GCS1BR,                0x540000F0,__READ_WRITE );
__IO_REG32_BIT(GCS1STP,               0x540000F4,__READ_WRITE ,__gcdstp_bits);
__IO_REG32_BIT(GCS1STR,               0x540000F8,__READ_WRITE ,__gcdstr_bits);
__IO_REG32_BIT(GCS1PF,                0x540000FC,__READ_WRITE ,__gcdpf_bits);
__IO_REG32(    GCSC_0_WD0,            0x54000160,__READ_WRITE );
__IO_REG32(    GCSC_0_WD1,            0x54000164,__READ_WRITE );
__IO_REG32(    GCSC_1_WD0,            0x54000168,__READ_WRITE );
__IO_REG32(    GCSC_1_WD1,            0x5400016C,__READ_WRITE );
__IO_REG32(    GCSC_2_WD0,            0x54000170,__READ_WRITE );
__IO_REG32(    GCSC_2_WD1,            0x54000174,__READ_WRITE );
__IO_REG32(    GCSC_3_WD0,            0x54000178,__READ_WRITE );
__IO_REG32(    GCSC_3_WD1,            0x5400017C,__READ_WRITE );
__IO_REG32(    GCSC_4_WD0,            0x54000180,__READ_WRITE );
__IO_REG32(    GCSC_4_WD1,            0x54000184,__READ_WRITE );
__IO_REG32(    GCSC_5_WD0,            0x54000188,__READ_WRITE );
__IO_REG32(    GCSC_5_WD1,            0x5400018C,__READ_WRITE );
__IO_REG32(    GCSC_6_WD0,            0x54000190,__READ_WRITE );
__IO_REG32(    GCSC_6_WD1,            0x54000194,__READ_WRITE );
__IO_REG32(    GCSC_7_WD0,            0x54000198,__READ_WRITE );
__IO_REG32(    GCSC_7_WD1,            0x5400019C,__READ_WRITE );
__IO_REG32(    GCCABADDR,             0x540001E0,__READ       );
__IO_REG32(    GCTABADDR,             0x540001E4,__READ       );
__IO_REG32(    GCMABADDR,             0x540001E8,__READ       );

/***************************************************************************
 **
 ** KPDC (Keypad Controller)
 **
 ***************************************************************************/
__IO_REG32_BIT(KPC,                   0x41500000,__READ_WRITE ,__kpc_bits);
__IO_REG32_BIT(KPDK,                  0x41500008,__READ       ,__kpdk_bits);
__IO_REG32_BIT(KPREC,                 0x41500010,__READ_WRITE ,__kprec_bits);
__IO_REG32_BIT(KPMK,                  0x41500018,__READ       ,__kpmk_bits);
__IO_REG32_BIT(KPAS,                  0x41500020,__READ       ,__kpas_bits);
__IO_REG32_BIT(KPASMKP0,              0x41500028,__READ       ,__kpasmkp0_bits);
__IO_REG32_BIT(KPASMKP1,              0x41500030,__READ       ,__kpasmkp1_bits);
__IO_REG32_BIT(KPASMKP2,              0x41500038,__READ       ,__kpasmkp2_bits);
__IO_REG32_BIT(KPASMKP3,              0x41500040,__READ       ,__kpasmkp3_bits);
__IO_REG32_BIT(KPKDI,                 0x41500048,__READ_WRITE ,__kpkdi_bits);

/***************************************************************************
 **
 ** ADCTSI (ADC and Touch Screen Interface)
 **
 ***************************************************************************/
__IO_REG32_BIT(ADCD,                  0x41C00000,__READ       ,__adcd_bits);
__IO_REG32_BIT(ADCS,                  0x41C00004,__READ_WRITE ,__adcs_bits);
__IO_REG32_BIT(ADCE,                  0x41C00008,__READ_WRITE ,__adce_bits);
__IO_REG32_BIT(ADCP,                  0x41C0000C,__READ       ,__adcp_bits);

/***************************************************************************
 **
 ** UDC ( USB 1.1 Client Controller )
 **
 ***************************************************************************/
__IO_REG32_BIT(UDCCR,                 0x40600000,__READ_WRITE ,__udccr_bits    );
__IO_REG32_BIT(UDCICR0,               0x40600004,__READ_WRITE ,__udcicr0_bits  );
__IO_REG32_BIT(UDCCIR1,               0x40600008,__READ_WRITE ,__udccir1_bits  );
__IO_REG32_BIT(UDCISR0,               0x4060000C,__READ_WRITE ,__udcisr0_bits  );
__IO_REG32_BIT(UDCISR1,               0x40600010,__READ_WRITE ,__udcisr1_bits  );
__IO_REG32_BIT(UDCFNR,                0x40600014,__READ       ,__udcfnr_bits   );
__IO_REG32_BIT(UDCOTGICR,             0x40600018,__READ_WRITE ,__udcotgicr_bits);
__IO_REG32_BIT(UDCOTGISR,             0x4060001C,__READ_WRITE ,__udcotgisr_bits);
__IO_REG32_BIT(UP2OCR,                0x40600020,__READ_WRITE ,__up2ocr_bits   );
__IO_REG32_BIT(UP3OCR,                0x40600024,__READ_WRITE ,__up3ocr_bits   );
__IO_REG32_BIT(UDCCSR0,               0x40600100,__READ_WRITE ,__udccsr0_bits  );
__IO_REG32_BIT(UDCCSRA,               0x40600104,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRB,               0x40600108,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRC,               0x4060010C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRD,               0x40600110,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRE,               0x40600114,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRF,               0x40600118,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRG,               0x4060011C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRH,               0x40600120,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRI,               0x40600124,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRJ,               0x40600128,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRK,               0x4060012C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRL,               0x40600130,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRM,               0x40600134,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRN,               0x40600138,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRP,               0x4060013C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRQ,               0x40600140,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRR,               0x40600144,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRS,               0x40600148,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRT,               0x4060014C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRU,               0x40600150,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRV,               0x40600154,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRW,               0x40600158,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCCSRX,               0x4060015C,__READ_WRITE ,__udccsra_bits  );
__IO_REG32_BIT(UDCBCR0,               0x40600200,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRA,               0x40600204,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRB,               0x40600208,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRC,               0x4060020C,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRD,               0x40600210,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRE,               0x40600214,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRF,               0x40600218,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRG,               0x4060021C,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRH,               0x40600220,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRI,               0x40600224,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRJ,               0x40600228,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRK,               0x4060022C,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRL,               0x40600230,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRM,               0x40600234,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRN,               0x40600238,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRP,               0x4060023C,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRQ,               0x40600240,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRR,               0x40600244,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRS,               0x40600248,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRT,               0x4060024C,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRU,               0x40600250,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRV,               0x40600254,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRW,               0x40600258,__READ       ,__udcbcr_bits   );
__IO_REG32_BIT(UDCBCRX,               0x4060025C,__READ       ,__udcbcr_bits   );
__IO_REG32(    UDCDR0,                0x40600300,__READ_WRITE                  );
__IO_REG32(    UDCDRA,                0x40600304,__READ_WRITE                  );
__IO_REG32(    UDCDRB,                0x40600308,__READ_WRITE                  );
__IO_REG32(    UDCDRC,                0x4060030C,__READ_WRITE                  );
__IO_REG32(    UDCDRD,                0x40600310,__READ_WRITE                  );
__IO_REG32(    UDCDRE,                0x40600314,__READ_WRITE                  );
__IO_REG32(    UDCDRF,                0x40600318,__READ_WRITE                  );
__IO_REG32(    UDCDRG,                0x4060031C,__READ_WRITE                  );
__IO_REG32(    UDCDRH,                0x40600320,__READ_WRITE                  );
__IO_REG32(    UDCDRI,                0x40600324,__READ_WRITE                  );
__IO_REG32(    UDCDRJ,                0x40600328,__READ_WRITE                  );
__IO_REG32(    UDCDRK,                0x4060032C,__READ_WRITE                  );
__IO_REG32(    UDCDRL,                0x40600330,__READ_WRITE                  );
__IO_REG32(    UDCDRM,                0x40600334,__READ_WRITE                  );
__IO_REG32(    UDCDRN,                0x40600338,__READ_WRITE                  );
__IO_REG32(    UDCDRP,                0x4060033C,__READ_WRITE                  );
__IO_REG32(    UDCDRQ,                0x40600340,__READ_WRITE                  );
__IO_REG32(    UDCDRR,                0x40600344,__READ_WRITE                  );
__IO_REG32(    UDCDRS,                0x40600348,__READ_WRITE                  );
__IO_REG32(    UDCDRT,                0x4060034C,__READ_WRITE                  );
__IO_REG32(    UDCDRU,                0x40600350,__READ_WRITE                  );
__IO_REG32(    UDCDRV,                0x40600354,__READ_WRITE                  );
__IO_REG32(    UDCDRW,                0x40600358,__READ_WRITE                  );
__IO_REG32(    UDCDRX,                0x4060035C,__READ_WRITE                  );
__IO_REG32_BIT(UDCCRA,                0x40600404,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRB,                0x40600408,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRC,                0x4060040C,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRD,                0x40600410,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRE,                0x40600414,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRF,                0x40600418,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRG,                0x4060041C,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRH,                0x40600420,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRI,                0x40600424,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRJ,                0x40600428,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRK,                0x4060042C,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRL,                0x40600430,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRM,                0x40600434,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRN,                0x40600438,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRP,                0x4060043C,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRQ,                0x40600440,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRR,                0x40600444,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRS,                0x40600448,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRT,                0x4060044C,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRU,                0x40600450,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRV,                0x40600454,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRW,                0x40600458,__READ_WRITE ,__udccra_bits   );
__IO_REG32_BIT(UDCCRX,                0x4060045C,__READ_WRITE ,__udccra_bits   );
 
/***************************************************************************
 **
 ** U2D ( USB 2.0 Device Controller )
 **
 ***************************************************************************/
__IO_REG32_BIT(U2DCR,                 0x54100000,__READ_WRITE ,__u2dcr_bits     );
__IO_REG32_BIT(U2DICR,                0x54100004,__READ_WRITE ,__u2dicr_bits    );
__IO_REG32_BIT(U2DISR,                0x5410000C,__READ_WRITE ,__u2disr_bits    );
__IO_REG32_BIT(U2DFNR,                0x54100014,__READ       ,__u2dfnr_bits    );
__IO_REG32_BIT(U2DCSR0,               0x54100100,__READ_WRITE ,__u2dcsr0_bits   );
__IO_REG32_BIT(U2DCSRA,               0x54100104,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRB,               0x54100108,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRC,               0x5410010C,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRD,               0x54100110,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRE,               0x54100114,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRF,               0x54100118,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DCSRG,               0x5410011C,__READ_WRITE ,__u2dcsra_bits   );
__IO_REG32_BIT(U2DBCR0,               0x54100200,__READ       ,__u2dbcr0_bits   );
__IO_REG32(    U2DDR0,                0x54100300,__READ_WRITE                   );
__IO_REG32_BIT(U2DCRA,                0x54100404,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRB,                0x54100408,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRC,                0x5410040C,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRD,                0x54100410,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRE,                0x54100414,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRF,                0x54100418,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32_BIT(U2DCRG,                0x5410041C,__READ_WRITE ,__u2dcra_bits    );
__IO_REG32(    U2DSCA,                0x54100500,__READ                         );
__IO_REG32_BIT(U2DEN0,                0x54100504,__READ_WRITE ,__u2den0_bits    );
__IO_REG32_BIT(U2DENA,                0x54100508,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DENB,                0x5410050C,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DENC,                0x54100510,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DEND,                0x54100514,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DENE,                0x54100518,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DENF,                0x5410051C,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DENG,                0x54100520,__READ_WRITE ,__u2dena_bits    );
__IO_REG32_BIT(U2DMACSR0,             0x54101000,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR1,             0x54101004,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR2,             0x54101008,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR3,             0x5410100C,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR4,             0x54101010,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR5,             0x54101014,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR6,             0x54101018,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACSR7,             0x5410101C,__READ_WRITE ,__u2dmacsrx_bits );
__IO_REG32_BIT(U2DMACR,               0x54101080,__READ_WRITE ,__u2dmacr_bits   );
__IO_REG32(    U2DMAINT,              0x541010F0,__READ                         );
__IO_REG32_BIT(U2DMADADR0,            0x54101200,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR0,            0x54101204,__READ                         );
__IO_REG32(    U2DMATADR0,            0x54101208,__READ                         );
__IO_REG32_BIT(U2DMACMD0,             0x5410120C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR1,            0x54101210,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR1,            0x54101214,__READ                         );
__IO_REG32(    U2DMATADR1,            0x54101218,__READ                         );
__IO_REG32_BIT(U2DMACMD1,             0x5410121C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR2,            0x54101220,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR2,            0x54101224,__READ                         );
__IO_REG32(    U2DMATADR2,            0x54101228,__READ                         );
__IO_REG32_BIT(U2DMACMD2,             0x5410122C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR3,            0x54101230,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR3,            0x54101234,__READ                         );
__IO_REG32(    U2DMATADR3,            0x54101238,__READ                         );
__IO_REG32_BIT(U2DMACMD3,             0x5410123C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR4,            0x54101240,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR4,            0x54101244,__READ                         );
__IO_REG32(    U2DMATADR4,            0x54101248,__READ                         );
__IO_REG32_BIT(U2DMACMD4,             0x5410124C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR5,            0x54101250,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR5,            0x54101254,__READ                         );
__IO_REG32(    U2DMATADR5,            0x54101258,__READ                         );
__IO_REG32_BIT(U2DMACMD5,             0x5410125C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR6,            0x54101260,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR6,            0x54101264,__READ                         );
__IO_REG32(    U2DMATADR6,            0x54101268,__READ                         );
__IO_REG32_BIT(U2DMACMD6,             0x5410126C,__READ       ,__u2dmacmdx_bits );
__IO_REG32_BIT(U2DMADADR7,            0x54101270,__READ_WRITE ,__u2dmadadrx_bits);
__IO_REG32(    U2DMASADR7,            0x54101274,__READ                         );
__IO_REG32(    U2DMATADR7,            0x54101278,__READ                         );
__IO_REG32_BIT(U2DMACMD7,             0x5410127C,__READ       ,__u2dmacmdx_bits );

/***************************************************************************
 **
 ** USB HOST (OHCI)
 **
 ***************************************************************************/
__IO_REG32(    UHCREV,                0x4C000000,__READ       );
__IO_REG32_BIT(UHCHCON,               0x4C000004,__READ_WRITE ,__hccontrol_bits);
__IO_REG32_BIT(UHCCOMS,               0x4C000008,__READ_WRITE ,__hccommandstatus_bits);
__IO_REG32_BIT(UHCINTS,               0x4C00000C,__READ_WRITE ,__hcinterruptstatus_bits);
__IO_REG32_BIT(UHCINTE,               0x4C000010,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(UHCINTD,               0x4C000014,__READ_WRITE ,__hcinterruptenable_bits);
__IO_REG32_BIT(UHCHCCA,               0x4C000018,__READ_WRITE ,__hchcca_bits);
__IO_REG32_BIT(UHCPCED,               0x4C00001C,__READ       ,__hcperiodcurrented_bits);
__IO_REG32_BIT(UHCCHED,               0x4C000020,__READ_WRITE ,__hccontrolheaded_bits);
__IO_REG32_BIT(UHCCCED,               0x4C000024,__READ_WRITE ,__hccontrolcurrented_bits);
__IO_REG32_BIT(UHCBHED,               0x4C000028,__READ_WRITE ,__hcbulkheaded_bits);
__IO_REG32_BIT(UHCBCED,               0x4C00002C,__READ_WRITE ,__hcbulkcurrented_bits);
__IO_REG32_BIT(UHCDHEAD,              0x4C000030,__READ       ,__hcdonehead_bits);
__IO_REG32_BIT(UHCFMI,                0x4C000034,__READ_WRITE ,__hcfminterval_bits);
__IO_REG32_BIT(UHCFMR,                0x4C000038,__READ       ,__hcfmremaining_bits);
__IO_REG32_BIT(UHCFMN,                0x4C00003C,__READ       ,__hcfmnumber_bits);
__IO_REG32_BIT(UHCPERS,               0x4C000040,__READ_WRITE ,__hcperiodicstart_bits);
__IO_REG32_BIT(UHCLST,                0x4C000044,__READ_WRITE ,__hclsthreshold_bits);
__IO_REG32_BIT(UHCRHDA,               0x4C000048,__READ_WRITE ,__hcrhdescriptora_bits);
__IO_REG32_BIT(UHCRHDB,               0x4C00004C,__READ_WRITE ,__hcrhdescriptorb_bits);
__IO_REG32_BIT(UHCRHS,                0x4C000050,__READ_WRITE ,__hcrhstatus_bits);
__IO_REG32_BIT(UHCRHPS1,              0x4C000054,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(UHCRHPS2,              0x4C000058,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(UHCRHPS3,              0x4C00005C,__READ_WRITE ,__hcrhportstatus_bits);
__IO_REG32_BIT(UHCSTAT,               0x4C000060,__READ_WRITE ,__uhcstat_bits);
__IO_REG32_BIT(UHCHR,                 0x4C000064,__READ_WRITE ,__uhchr_bits);
__IO_REG32_BIT(UHCHIE,                0x4C000068,__READ_WRITE ,__uhchie_bits);
__IO_REG32_BIT(UHCHIT,                0x4C00006C,__READ_WRITE ,__uhchit_bits);

/***************************************************************************
 **
 ** SSP1
 **
 ***************************************************************************/
__IO_REG32_BIT(SSCR0_1,               0x41000000,__READ_WRITE ,__sscr0_x_bits);
__IO_REG32_BIT(SSCR1_1,               0x41000004,__READ_WRITE ,__sscr1_x_bits);
__IO_REG32_BIT(SSSR_1,                0x41000008,__READ_WRITE ,__sssr_x_bits );
__IO_REG32_BIT(SSITR_1,               0x4100000C,__READ_WRITE ,__ssitr_x_bits);
__IO_REG32(    SSDR_1,                0x41000010,__READ_WRITE                );
__IO_REG32_BIT(SSTO_1,                0x41000028,__READ_WRITE ,__ssto_x_bits );
__IO_REG32_BIT(SSPSP_1,               0x4100002C,__READ_WRITE ,__sspsp_x_bits);
__IO_REG32_BIT(SSTSA_1,               0x41000030,__READ_WRITE ,__sstsa_x_bits);
__IO_REG32_BIT(SSRSA_1,               0x41000034,__READ_WRITE ,__ssrsa_x_bits);
__IO_REG32_BIT(SSTSS_1,               0x41000038,__READ       ,__sstss_x_bits);
__IO_REG32_BIT(SSACD_1,               0x4100003C,__READ_WRITE ,__ssacd_x_bits);
__IO_REG32_BIT(SSACDD_1,              0x41000040,__READ_WRITE ,__ssacdd_x_bits);

/***************************************************************************
 **
 ** SSP2
 **
 ***************************************************************************/
__IO_REG32_BIT(SSCR0_2,               0x41700000,__READ_WRITE ,__sscr0_x_bits);
__IO_REG32_BIT(SSCR1_2,               0x41700004,__READ_WRITE ,__sscr1_x_bits);
__IO_REG32_BIT(SSSR_2,                0x41700008,__READ_WRITE ,__sssr_x_bits );
__IO_REG32_BIT(SSITR_2,               0x4170000C,__READ_WRITE ,__ssitr_x_bits);
__IO_REG32(    SSDR_2,                0x41700010,__READ_WRITE                );
__IO_REG32_BIT(SSTO_2,                0x41700028,__READ_WRITE ,__ssto_x_bits );
__IO_REG32_BIT(SSPSP_2,               0x4170002C,__READ_WRITE ,__sspsp_x_bits);
__IO_REG32_BIT(SSTSA_2,               0x41700030,__READ_WRITE ,__sstsa_x_bits);
__IO_REG32_BIT(SSRSA_2,               0x41700034,__READ_WRITE ,__ssrsa_x_bits);
__IO_REG32_BIT(SSTSS_2,               0x41700038,__READ       ,__sstss_x_bits);
__IO_REG32_BIT(SSACD_2,               0x4170003C,__READ_WRITE ,__ssacd_x_bits);
__IO_REG32_BIT(SSACDD_2,              0x41700040,__READ_WRITE ,__ssacdd_x_bits);

/***************************************************************************
 **
 ** SSP3
 **
 ***************************************************************************/
__IO_REG32_BIT(SSCR0_3,               0x41900000,__READ_WRITE ,__sscr0_x_bits);
__IO_REG32_BIT(SSCR1_3,               0x41900004,__READ_WRITE ,__sscr1_x_bits);
__IO_REG32_BIT(SSSR_3,                0x41900008,__READ_WRITE ,__sssr_x_bits );
__IO_REG32_BIT(SSITR_3,               0x4190000C,__READ_WRITE ,__ssitr_x_bits);
__IO_REG32(    SSDR_3,                0x41900010,__READ_WRITE                );
__IO_REG32_BIT(SSTO_3,                0x41900028,__READ_WRITE ,__ssto_x_bits );
__IO_REG32_BIT(SSPSP_3,               0x4190002C,__READ_WRITE ,__sspsp_x_bits);
__IO_REG32_BIT(SSTSA_3,               0x41900030,__READ_WRITE ,__sstsa_x_bits);
__IO_REG32_BIT(SSRSA_3,               0x41900034,__READ_WRITE ,__ssrsa_x_bits);
__IO_REG32_BIT(SSTSS_3,               0x41900038,__READ       ,__sstss_x_bits);
__IO_REG32_BIT(SSACD_3,               0x4190003C,__READ_WRITE ,__ssacd_x_bits);
__IO_REG32_BIT(SSACDD_3,              0x41900040,__READ_WRITE ,__ssacdd_x_bits);

/***************************************************************************
 **
 ** SSP4
 **
 ***************************************************************************/
__IO_REG32_BIT(SSCR0_4,               0x41A00000,__READ_WRITE ,__sscr0_x_bits);
__IO_REG32_BIT(SSCR1_4,               0x41A00004,__READ_WRITE ,__sscr1_x_bits);
__IO_REG32_BIT(SSSR_4,                0x41A00008,__READ_WRITE ,__sssr_x_bits );
__IO_REG32_BIT(SSITR_4,               0x41A0000C,__READ_WRITE ,__ssitr_x_bits);
__IO_REG32(    SSDR_4,                0x41A00010,__READ_WRITE                );
__IO_REG32_BIT(SSTO_4,                0x41A00028,__READ_WRITE ,__ssto_x_bits );
__IO_REG32_BIT(SSPSP_4,               0x41A0002C,__READ_WRITE ,__sspsp_x_bits);
__IO_REG32_BIT(SSTSA_4,               0x41A00030,__READ_WRITE ,__sstsa_x_bits);
__IO_REG32_BIT(SSRSA_4,               0x41A00034,__READ_WRITE ,__ssrsa_x_bits);
__IO_REG32_BIT(SSTSS_4,               0x41A00038,__READ       ,__sstss_x_bits);
__IO_REG32_BIT(SSACD_4,               0x41A0003C,__READ_WRITE ,__ssacd_x_bits);
__IO_REG32_BIT(SSACDD_4,              0x41A00040,__READ_WRITE ,__ssacdd_x_bits);

/***************************************************************************
 **
 ** AC97 Controller
 **
 ***************************************************************************/

__IO_REG32_BIT(POCR,                  0x40500000,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(PCMICR,                0x40500004,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MCCR,                  0x40500008,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(GCR,                   0x4050000C,__READ_WRITE ,__gcr_bits   );
__IO_REG32_BIT(POSR,                  0x40500010,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(PCMISR,                0x40500014,__READ_WRITE ,__pcmisr_bits);
__IO_REG32_BIT(MCSR,                  0x40500018,__READ_WRITE ,__pcmisr_bits);
__IO_REG32_BIT(GSR,                   0x4050001C,__READ_WRITE ,__gsr_bits   );
__IO_REG32_BIT(CAR,                   0x40500020,__READ_WRITE ,__car_bits   );
__IO_REG32_BIT(PCSCR,                 0x40500024,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(PCSSR,                 0x40500028,__READ_WRITE ,__posr_bits );
__IO_REG32_BIT(PCSDR,                 0x4050002C,__READ_WRITE ,__pcsdr_bits );
__IO_REG32_BIT(PCCLCR,                0x40500030,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(PCCLSR,                0x40500034,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(PCCLDR,                0x40500038,__READ_WRITE ,__pccldr_bits);
__IO_REG32_BIT(PCDR,                  0x40500040,__READ_WRITE ,__pcdr_bits  );
__IO_REG32_BIT(MCDR,                  0x40500060,__READ       ,__mcdr_bits  );
__IO_REG32_BIT(MOCR,                  0x40500100,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MICR,                  0x40500108,__READ_WRITE ,__pocr_bits  );
__IO_REG32_BIT(MOSR,                  0x40500110,__READ_WRITE ,__posr_bits  );
__IO_REG32_BIT(MISR,                  0x40500118,__READ_WRITE ,__pcmisr_bits);
__IO_REG32_BIT(MODR,                  0x40500140,__READ_WRITE ,__modr_bits  );

/***************************************************************************
 **
 ** CIR (Consumer Infrared Unit)
 **
 ***************************************************************************/
__IO_REG32_BIT(CIRPW,                 0x41D00000,__READ_WRITE ,__cirpw_bits);
__IO_REG32_BIT(CIRMP,                 0x41D00004,__READ_WRITE ,__cirmp_bits);
__IO_REG32_BIT(CIRN0,                 0x41D00008,__READ_WRITE ,__cirn0_bits);
__IO_REG32_BIT(CIRN1,                 0x41D0000C,__READ_WRITE ,__cirn1_bits);
__IO_REG32_BIT(CIRS0,                 0x41D00010,__READ_WRITE ,__cirs0_bits);
__IO_REG32_BIT(CIRS1,                 0x41D00014,__READ_WRITE ,__cirs1_bits);
__IO_REG32(    CIRBUFF,               0x41D00018,__READ_WRITE              );
__IO_REG32_BIT(CIRNS,                 0x41D0001C,__READ_WRITE ,__cirns_bits);
__IO_REG32_BIT(CIRCR,                 0x41D00020,__READ_WRITE ,__circr_bits);
__IO_REG32_BIT(CIRIR,                 0x41D00024,__READ_WRITE ,__cirir_bits);

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
__IO_REG32_BIT(FFFOR,                 0x40100024,__READ_WRITE ,__uartfor_bits   );
__IO_REG32_BIT(FFABR,                 0x40100028,__READ_WRITE ,__uartabr_bits   );
__IO_REG32_BIT(FFACR,                 0x4010002C,__READ       ,__uartacr_bits   );

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
__IO_REG32_BIT(BTMCR,                 0x40200010,__READ_WRITE ,__uartmcr_bits   );
__IO_REG32_BIT(BTLSR,                 0x40200014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(BTMSR,                 0x40200018,__READ       ,__uartmsr_bits   );
__IO_REG32_BIT(BTSPR,                 0x4020001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(BTISR,                 0x40200020,__READ_WRITE ,__uartisr_bits   );
__IO_REG32_BIT(BTFOR,                 0x40200024,__READ_WRITE ,__uartfor_bits   );
__IO_REG32_BIT(BTABR,                 0x40200028,__READ_WRITE ,__uartabr_bits   );
__IO_REG32_BIT(BTACR,                 0x4020002C,__READ       ,__uartacr_bits   );

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
__IO_REG32_BIT(STMCR,                 0x40700010,__READ_WRITE ,__uartmcr_bits   );
__IO_REG32_BIT(STLSR,                 0x40700014,__READ       ,__uartlsr_bits   );
__IO_REG32_BIT(STMSR,                 0x40700018,__READ       ,__uartmsr_bits   );
__IO_REG32_BIT(STSPR,                 0x4070001C,__READ_WRITE ,__uartspr_bits   );
__IO_REG32_BIT(STISR,                 0x40700020,__READ_WRITE ,__uartisr_bits   );
__IO_REG32_BIT(STFOR,                 0x40700024,__READ_WRITE ,__uartfor_bits   );
__IO_REG32_BIT(STABR,                 0x40700028,__READ_WRITE ,__uartabr_bits   );
__IO_REG32_BIT(STACR,                 0x4070002C,__READ       ,__uartacr_bits   );

/***************************************************************************
 **
 ** PWM0
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR0,                0x40B00000,__READ_WRITE ,__pwmcrx_bits );
__IO_REG32_BIT(PWMDCR0,               0x40B00004,__READ_WRITE ,__pwmdcrx_bits);
__IO_REG32_BIT(PWMPCR0,               0x40B00008,__READ_WRITE ,__pwmpcrx_bits);

/***************************************************************************
 **
 ** PWM2
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR2,                0x40B00010,__READ_WRITE ,__pwmcrx_bits );
__IO_REG32_BIT(PWMDCR2,               0x40B00014,__READ_WRITE ,__pwmdcrx_bits);
__IO_REG32_BIT(PWMPCR2,               0x40B00018,__READ_WRITE ,__pwmpcrx_bits);

/***************************************************************************
 **
 ** PWM1
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR1,                0x40C00000,__READ_WRITE ,__pwmcrx_bits );
__IO_REG32_BIT(PWMDCR1,               0x40C00004,__READ_WRITE ,__pwmdcrx_bits);
__IO_REG32_BIT(PWMPCR1,               0x40C00008,__READ_WRITE ,__pwmpcrx_bits);

/***************************************************************************
 **
 ** PWM3
 **
 ***************************************************************************/
__IO_REG32_BIT(PWMCR3,                0x40C00010,__READ_WRITE ,__pwmcrx_bits );
__IO_REG32_BIT(PWMDCR3,               0x40C00014,__READ_WRITE ,__pwmdcrx_bits);
__IO_REG32_BIT(PWMPCR3,               0x40C00018,__READ_WRITE ,__pwmpcrx_bits);

/***************************************************************************
 **
 ** USIM1
 **
 ***************************************************************************/
__IO_REG32_BIT(USIM1RBR,              0x41600000,__READ       ,__usimxrbr_bits  );
__IO_REG32_BIT(USIM1THR,              0x41600004,__WRITE      ,__usimxthr_bits  );
__IO_REG32_BIT(USIM1IER,              0x41600008,__READ_WRITE ,__usimxier_bits  );
__IO_REG32_BIT(USIM1IIR,              0x4160000C,__READ_WRITE ,__usimxiir_bits  );
__IO_REG32_BIT(USIM1FCR,              0x41600010,__WRITE      ,__usimxfcr_bits  );
__IO_REG32_BIT(USIM1FSR,              0x41600014,__READ       ,__usimxfsr_bits  );
__IO_REG32_BIT(USIM1ECR,              0x41600018,__READ_WRITE ,__usimxecr_bits  );
__IO_REG32_BIT(USIM1LCR,              0x4160001C,__READ_WRITE ,__usimxlcr_bits  );
__IO_REG32_BIT(USIM1USCCR,            0x41600020,__READ_WRITE ,__usimxusccr_bits);
__IO_REG32_BIT(USIM1LSR,              0x41600024,__READ       ,__usimxlsr_bits  );
__IO_REG32_BIT(USIM1EGTR,             0x41600028,__READ_WRITE ,__usimxegtr_bits );
__IO_REG32_BIT(USIM1BGTR,             0x4160002C,__READ_WRITE ,__usimxbgtr_bits );
__IO_REG32_BIT(USIM1TOR,              0x41600030,__READ_WRITE ,__usimxtor_bits  );
__IO_REG32_BIT(USIM1CLKR,             0x41600034,__READ_WRITE ,__usimxclkr_bits );
__IO_REG32_BIT(USIM1DLR,              0x41600038,__READ_WRITE ,__usimxdlr_bits  );
__IO_REG32_BIT(USIM1FLR,              0x4160003C,__READ_WRITE ,__usimxflr_bits  );
__IO_REG32_BIT(USIM1CWTR,             0x41600040,__READ_WRITE ,__usimxcwtr_bits );
__IO_REG32_BIT(USIM1BWTR,             0x41600044,__READ_WRITE ,__usimxbwtr_bits );

/***************************************************************************
 **
 ** USIM2
 **
 ***************************************************************************/
__IO_REG32_BIT(USIM2RBR,              0x42100000,__READ       ,__usimxrbr_bits  );
__IO_REG32_BIT(USIM2THR,              0x42100004,__WRITE      ,__usimxthr_bits  );
__IO_REG32_BIT(USIM2IER,              0x42100008,__READ_WRITE ,__usimxier_bits  );
__IO_REG32_BIT(USIM2IIR,              0x4210000C,__READ_WRITE ,__usimxiir_bits  );
__IO_REG32_BIT(USIM2FCR,              0x42100010,__WRITE      ,__usimxfcr_bits  );
__IO_REG32_BIT(USIM2FSR,              0x42100014,__READ       ,__usimxfsr_bits  );
__IO_REG32_BIT(USIM2ECR,              0x42100018,__READ_WRITE ,__usimxecr_bits  );
__IO_REG32_BIT(USIM2LCR,              0x4210001C,__READ_WRITE ,__usimxlcr_bits  );
__IO_REG32_BIT(USIM2USCCR,            0x42100020,__READ_WRITE ,__usimxusccr_bits);
__IO_REG32_BIT(USIM2LSR,              0x42100024,__READ       ,__usimxlsr_bits  );
__IO_REG32_BIT(USIM2EGTR,             0x42100028,__READ_WRITE ,__usimxegtr_bits );
__IO_REG32_BIT(USIM2BGTR,             0x4210002C,__READ_WRITE ,__usimxbgtr_bits );
__IO_REG32_BIT(USIM2TOR,              0x42100030,__READ_WRITE ,__usimxtor_bits  );
__IO_REG32_BIT(USIM2CLKR,             0x42100034,__READ_WRITE ,__usimxclkr_bits );
__IO_REG32_BIT(USIM2DLR,              0x42100038,__READ_WRITE ,__usimxdlr_bits  );
__IO_REG32_BIT(USIM2FLR,              0x4210003C,__READ_WRITE ,__usimxflr_bits  );
__IO_REG32_BIT(USIM2CWTR,             0x42100040,__READ_WRITE ,__usimxcwtr_bits );
__IO_REG32_BIT(USIM2BWTR,             0x42100044,__READ_WRITE ,__usimxbwtr_bits );

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
 **  Assembler-specific declarations
 ***************************************************************************/
#ifdef __IAR_SYSTEMS_ASM__
#endif    /* __IAR_SYSTEMS_ASM__ */

#endif    /* __IOPXA320_H */
