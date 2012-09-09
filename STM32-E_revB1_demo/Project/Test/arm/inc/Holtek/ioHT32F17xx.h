/***************************************************************************
**
**    This file defines the Special Function Registers for
**    HOLTEK ioHT32F17xx
**
**    Used with ARM IAR C/C++ Compiler and Assembler
**
**    (c) Copyright IAR Systems 2012
**
**    $Revision: 52631 $
**
***************************************************************************/
#ifndef __ioHT32F17xx_H__

#define __ioHT32F17xx_H__

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   ioHT32F17xx SPECIAL FUNCTION REGISTERS
 **
 ***************************************************************************
 ***************************************************************************
***************************************************************************/

/* C-compiler specific declarations  **************************************/

#ifdef __IAR_SYSTEMS_ICC__

#ifndef _SYSTEM_BUILD
    #pragma system_include
#endif

typedef struct {
    __REG32    INTLINESNUM : 5;
    __REG32                : 27;
} __ictr_bits;


typedef struct {
    __REG32    DISMCYCINT : 1;
    __REG32    DISDEFWBUF : 1;
    __REG32    DISFOLD    : 1;
    __REG32               : 29;
} __actlr_bits;


typedef struct {
    __REG32    SETENA0  : 1;
    __REG32    SETENA1  : 1;
    __REG32    SETENA2  : 1;
    __REG32    SETENA3  : 1;
    __REG32    SETENA4  : 1;
    __REG32    SETENA5  : 1;
    __REG32    SETENA6  : 1;
    __REG32    SETENA7  : 1;
    __REG32    SETENA8  : 1;
    __REG32    SETENA9  : 1;
    __REG32    SETENA10 : 1;
    __REG32    SETENA11 : 1;
    __REG32    SETENA12 : 1;
    __REG32    SETENA13 : 1;
    __REG32    SETENA14 : 1;
    __REG32    SETENA15 : 1;
    __REG32    SETENA16 : 1;
    __REG32    SETENA17 : 1;
    __REG32    SETENA18 : 1;
    __REG32    SETENA19 : 1;
    __REG32    SETENA20 : 1;
    __REG32    SETENA21 : 1;
    __REG32    SETENA22 : 1;
    __REG32    SETENA23 : 1;
    __REG32    SETENA24 : 1;
    __REG32    SETENA25 : 1;
    __REG32    SETENA26 : 1;
    __REG32    SETENA27 : 1;
    __REG32    SETENA28 : 1;
    __REG32    SETENA29 : 1;
    __REG32    SETENA30 : 1;
    __REG32    SETENA31 : 1;
} __iser0_bits;


typedef struct {
    __REG32    SETENA32 : 1;
    __REG32    SETENA33 : 1;
    __REG32    SETENA34 : 1;
    __REG32    SETENA35 : 1;
    __REG32    SETENA36 : 1;
    __REG32    SETENA37 : 1;
    __REG32    SETENA38 : 1;
    __REG32    SETENA39 : 1;
    __REG32    SETENA40 : 1;
    __REG32    SETENA41 : 1;
    __REG32    SETENA42 : 1;
    __REG32    SETENA43 : 1;
    __REG32    SETENA44 : 1;
    __REG32    SETENA45 : 1;
    __REG32    SETENA46 : 1;
    __REG32    SETENA47 : 1;
    __REG32    SETENA48 : 1;
    __REG32    SETENA49 : 1;
    __REG32    SETENA50 : 1;
    __REG32    SETENA51 : 1;
    __REG32    SETENA52 : 1;
    __REG32    SETENA53 : 1;
    __REG32    SETENA54 : 1;
    __REG32    SETENA55 : 1;
    __REG32    SETENA56 : 1;
    __REG32    SETENA57 : 1;
    __REG32    SETENA58 : 1;
    __REG32    SETENA59 : 1;
    __REG32    SETENA60 : 1;
    __REG32    SETENA61 : 1;
    __REG32    SETENA62 : 1;
    __REG32    SETENA63 : 1;
} __iser1_bits;


typedef struct {
    __REG32    SETENA64 : 1;
    __REG32    SETENA65 : 1;
    __REG32    SETENA66 : 1;
    __REG32    SETENA67 : 1;
    __REG32             : 28;
} __iser2_bits;


typedef struct {
    __REG32    CLRENA0  : 1;
    __REG32    CLRENA1  : 1;
    __REG32    CLRENA2  : 1;
    __REG32    CLRENA3  : 1;
    __REG32    CLRENA4  : 1;
    __REG32    CLRENA5  : 1;
    __REG32    CLRENA6  : 1;
    __REG32    CLRENA7  : 1;
    __REG32    CLRENA8  : 1;
    __REG32    CLRENA9  : 1;
    __REG32    CLRENA10 : 1;
    __REG32    CLRENA11 : 1;
    __REG32    CLRENA12 : 1;
    __REG32    CLRENA13 : 1;
    __REG32    CLRENA14 : 1;
    __REG32    CLRENA15 : 1;
    __REG32    CLRENA16 : 1;
    __REG32    CLRENA17 : 1;
    __REG32    CLRENA18 : 1;
    __REG32    CLRENA19 : 1;
    __REG32    CLRENA20 : 1;
    __REG32    CLRENA21 : 1;
    __REG32    CLRENA22 : 1;
    __REG32    CLRENA23 : 1;
    __REG32    CLRENA24 : 1;
    __REG32    CLRENA25 : 1;
    __REG32    CLRENA26 : 1;
    __REG32    CLRENA27 : 1;
    __REG32    CLRENA28 : 1;
    __REG32    CLRENA29 : 1;
    __REG32    CLRENA30 : 1;
    __REG32    CLRENA31 : 1;
} __icer0_bits;


typedef struct {
    __REG32    CLRENA32 : 1;
    __REG32    CLRENA33 : 1;
    __REG32    CLRENA34 : 1;
    __REG32    CLRENA35 : 1;
    __REG32    CLRENA36 : 1;
    __REG32    CLRENA37 : 1;
    __REG32    CLRENA38 : 1;
    __REG32    CLRENA39 : 1;
    __REG32    CLRENA40 : 1;
    __REG32    CLRENA41 : 1;
    __REG32    CLRENA42 : 1;
    __REG32    CLRENA43 : 1;
    __REG32    CLRENA44 : 1;
    __REG32    CLRENA45 : 1;
    __REG32    CLRENA46 : 1;
    __REG32    CLRENA47 : 1;
    __REG32    CLRENA48 : 1;
    __REG32    CLRENA49 : 1;
    __REG32    CLRENA50 : 1;
    __REG32    CLRENA51 : 1;
    __REG32    CLRENA52 : 1;
    __REG32    CLRENA53 : 1;
    __REG32    CLRENA54 : 1;
    __REG32    CLRENA55 : 1;
    __REG32    CLRENA56 : 1;
    __REG32    CLRENA57 : 1;
    __REG32    CLRENA58 : 1;
    __REG32    CLRENA59 : 1;
    __REG32    CLRENA60 : 1;
    __REG32    CLRENA61 : 1;
    __REG32    CLRENA62 : 1;
    __REG32    CLRENA63 : 1;
} __icer1_bits;


typedef struct {
    __REG32    CLRENA64 : 1;
    __REG32    CLRENA65 : 1;
    __REG32    CLRENA66 : 1;
    __REG32    CLRENA67 : 1;
    __REG32             : 28;
} __icer2_bits;


typedef struct {
    __REG32    SETPEND0  : 1;
    __REG32    SETPEND1  : 1;
    __REG32    SETPEND2  : 1;
    __REG32    SETPEND3  : 1;
    __REG32    SETPEND4  : 1;
    __REG32    SETPEND5  : 1;
    __REG32    SETPEND6  : 1;
    __REG32    SETPEND7  : 1;
    __REG32    SETPEND8  : 1;
    __REG32    SETPEND9  : 1;
    __REG32    SETPEND10 : 1;
    __REG32    SETPEND11 : 1;
    __REG32    SETPEND12 : 1;
    __REG32    SETPEND13 : 1;
    __REG32    SETPEND14 : 1;
    __REG32    SETPEND15 : 1;
    __REG32    SETPEND16 : 1;
    __REG32    SETPEND17 : 1;
    __REG32    SETPEND18 : 1;
    __REG32    SETPEND19 : 1;
    __REG32    SETPEND20 : 1;
    __REG32    SETPEND21 : 1;
    __REG32    SETPEND22 : 1;
    __REG32    SETPEND23 : 1;
    __REG32    SETPEND24 : 1;
    __REG32    SETPEND25 : 1;
    __REG32    SETPEND26 : 1;
    __REG32    SETPEND27 : 1;
    __REG32    SETPEND28 : 1;
    __REG32    SETPEND29 : 1;
    __REG32    SETPEND30 : 1;
    __REG32    SETPEND31 : 1;
} __ispr0_bits;


typedef struct {
    __REG32    SETPEND32 : 1;
    __REG32    SETPEND33 : 1;
    __REG32    SETPEND34 : 1;
    __REG32    SETPEND35 : 1;
    __REG32    SETPEND36 : 1;
    __REG32    SETPEND37 : 1;
    __REG32    SETPEND38 : 1;
    __REG32    SETPEND39 : 1;
    __REG32    SETPEND40 : 1;
    __REG32    SETPEND41 : 1;
    __REG32    SETPEND42 : 1;
    __REG32    SETPEND43 : 1;
    __REG32    SETPEND44 : 1;
    __REG32    SETPEND45 : 1;
    __REG32    SETPEND46 : 1;
    __REG32    SETPEND47 : 1;
    __REG32    SETPEND48 : 1;
    __REG32    SETPEND49 : 1;
    __REG32    SETPEND50 : 1;
    __REG32    SETPEND51 : 1;
    __REG32    SETPEND52 : 1;
    __REG32    SETPEND53 : 1;
    __REG32    SETPEND54 : 1;
    __REG32    SETPEND55 : 1;
    __REG32    SETPEND56 : 1;
    __REG32    SETPEND57 : 1;
    __REG32    SETPEND58 : 1;
    __REG32    SETPEND59 : 1;
    __REG32    SETPEND60 : 1;
    __REG32    SETPEND61 : 1;
    __REG32    SETPEND62 : 1;
    __REG32    SETPEND63 : 1;
} __ispr1_bits;


typedef struct {
    __REG32    SETPEND64 : 1;
    __REG32    SETPEND65 : 1;
    __REG32    SETPEND66 : 1;
    __REG32    SETPEND67 : 1;
    __REG32              : 28;
} __ispr2_bits;


typedef struct {
    __REG32    CLRPEND0  : 1;
    __REG32    CLRPEND1  : 1;
    __REG32    CLRPEND2  : 1;
    __REG32    CLRPEND3  : 1;
    __REG32    CLRPEND4  : 1;
    __REG32    CLRPEND5  : 1;
    __REG32    CLRPEND6  : 1;
    __REG32    CLRPEND7  : 1;
    __REG32    CLRPEND8  : 1;
    __REG32    CLRPEND9  : 1;
    __REG32    CLRPEND10 : 1;
    __REG32    CLRPEND11 : 1;
    __REG32    CLRPEND12 : 1;
    __REG32    CLRPEND13 : 1;
    __REG32    CLRPEND14 : 1;
    __REG32    CLRPEND15 : 1;
    __REG32    CLRPEND16 : 1;
    __REG32    CLRPEND17 : 1;
    __REG32    CLRPEND18 : 1;
    __REG32    CLRPEND19 : 1;
    __REG32    CLRPEND20 : 1;
    __REG32    CLRPEND21 : 1;
    __REG32    CLRPEND22 : 1;
    __REG32    CLRPEND23 : 1;
    __REG32    CLRPEND24 : 1;
    __REG32    CLRPEND25 : 1;
    __REG32    CLRPEND26 : 1;
    __REG32    CLRPEND27 : 1;
    __REG32    CLRPEND28 : 1;
    __REG32    CLRPEND29 : 1;
    __REG32    CLRPEND30 : 1;
    __REG32    CLRPEND31 : 1;
} __icpr0_bits;


typedef struct {
    __REG32    CLRPEND32 : 1;
    __REG32    CLRPEND33 : 1;
    __REG32    CLRPEND34 : 1;
    __REG32    CLRPEND35 : 1;
    __REG32    CLRPEND36 : 1;
    __REG32    CLRPEND37 : 1;
    __REG32    CLRPEND38 : 1;
    __REG32    CLRPEND39 : 1;
    __REG32    CLRPEND40 : 1;
    __REG32    CLRPEND41 : 1;
    __REG32    CLRPEND42 : 1;
    __REG32    CLRPEND43 : 1;
    __REG32    CLRPEND44 : 1;
    __REG32    CLRPEND45 : 1;
    __REG32    CLRPEND46 : 1;
    __REG32    CLRPEND47 : 1;
    __REG32    CLRPEND48 : 1;
    __REG32    CLRPEND49 : 1;
    __REG32    CLRPEND50 : 1;
    __REG32    CLRPEND51 : 1;
    __REG32    CLRPEND52 : 1;
    __REG32    CLRPEND53 : 1;
    __REG32    CLRPEND54 : 1;
    __REG32    CLRPEND55 : 1;
    __REG32    CLRPEND56 : 1;
    __REG32    CLRPEND57 : 1;
    __REG32    CLRPEND58 : 1;
    __REG32    CLRPEND59 : 1;
    __REG32    CLRPEND60 : 1;
    __REG32    CLRPEND61 : 1;
    __REG32    CLRPEND62 : 1;
    __REG32    CLRPEND63 : 1;
} __icpr1_bits;


typedef struct {
    __REG32    CLRPEND64 : 1;
    __REG32    CLRPEND65 : 1;
    __REG32    CLRPEND66 : 1;
    __REG32    CLRPEND67 : 1;
    __REG32              : 28;
} __icpr2_bits;


typedef struct {
    __REG32    ACTIVE0  : 1;
    __REG32    ACTIVE1  : 1;
    __REG32    ACTIVE2  : 1;
    __REG32    ACTIVE3  : 1;
    __REG32    ACTIVE4  : 1;
    __REG32    ACTIVE5  : 1;
    __REG32    ACTIVE6  : 1;
    __REG32    ACTIVE7  : 1;
    __REG32    ACTIVE8  : 1;
    __REG32    ACTIVE9  : 1;
    __REG32    ACTIVE10 : 1;
    __REG32    ACTIVE11 : 1;
    __REG32    ACTIVE12 : 1;
    __REG32    ACTIVE13 : 1;
    __REG32    ACTIVE14 : 1;
    __REG32    ACTIVE15 : 1;
    __REG32    ACTIVE16 : 1;
    __REG32    ACTIVE17 : 1;
    __REG32    ACTIVE18 : 1;
    __REG32    ACTIVE19 : 1;
    __REG32    ACTIVE20 : 1;
    __REG32    ACTIVE21 : 1;
    __REG32    ACTIVE22 : 1;
    __REG32    ACTIVE23 : 1;
    __REG32    ACTIVE24 : 1;
    __REG32    ACTIVE25 : 1;
    __REG32    ACTIVE26 : 1;
    __REG32    ACTIVE27 : 1;
    __REG32    ACTIVE28 : 1;
    __REG32    ACTIVE29 : 1;
    __REG32    ACTIVE30 : 1;
    __REG32    ACTIVE31 : 1;
} __iabr0_bits;


typedef struct {
    __REG32    ACTIVE32 : 1;
    __REG32    ACTIVE33 : 1;
    __REG32    ACTIVE34 : 1;
    __REG32    ACTIVE35 : 1;
    __REG32    ACTIVE36 : 1;
    __REG32    ACTIVE37 : 1;
    __REG32    ACTIVE38 : 1;
    __REG32    ACTIVE39 : 1;
    __REG32    ACTIVE40 : 1;
    __REG32    ACTIVE41 : 1;
    __REG32    ACTIVE42 : 1;
    __REG32    ACTIVE43 : 1;
    __REG32    ACTIVE44 : 1;
    __REG32    ACTIVE45 : 1;
    __REG32    ACTIVE46 : 1;
    __REG32    ACTIVE47 : 1;
    __REG32    ACTIVE48 : 1;
    __REG32    ACTIVE49 : 1;
    __REG32    ACTIVE50 : 1;
    __REG32    ACTIVE51 : 1;
    __REG32    ACTIVE52 : 1;
    __REG32    ACTIVE53 : 1;
    __REG32    ACTIVE54 : 1;
    __REG32    ACTIVE55 : 1;
    __REG32    ACTIVE56 : 1;
    __REG32    ACTIVE57 : 1;
    __REG32    ACTIVE58 : 1;
    __REG32    ACTIVE59 : 1;
    __REG32    ACTIVE60 : 1;
    __REG32    ACTIVE61 : 1;
    __REG32    ACTIVE62 : 1;
    __REG32    ACTIVE63 : 1;
} __iabr1_bits;


typedef struct {
    __REG32    ACTIVE64 : 1;
    __REG32    ACTIVE65 : 1;
    __REG32    ACTIVE66 : 1;
    __REG32    ACTIVE67 : 1;
    __REG32             : 28;
} __iabr2_bits;


typedef struct {
    __REG32    PRI_0 : 8;
    __REG32    PRI_1 : 8;
    __REG32    PRI_2 : 8;
    __REG32    PRI_3 : 8;
} __ip0_bits;


typedef struct {
    __REG32    PRI_4 : 8;
    __REG32    PRI_5 : 8;
    __REG32    PRI_6 : 8;
    __REG32    PRI_7 : 8;
} __ip1_bits;


typedef struct {
    __REG32    PRI_8  : 8;
    __REG32    PRI_9  : 8;
    __REG32    PRI_10 : 8;
    __REG32    PRI_11 : 8;
} __ip2_bits;


typedef struct {
    __REG32    PRI_12 : 8;
    __REG32    PRI_13 : 8;
    __REG32    PRI_14 : 8;
    __REG32    PRI_15 : 8;
} __ip3_bits;


typedef struct {
    __REG32    PRI_16 : 8;
    __REG32    PRI_17 : 8;
    __REG32    PRI_18 : 8;
    __REG32    PRI_19 : 8;
} __ip4_bits;


typedef struct {
    __REG32    PRI_20 : 8;
    __REG32    PRI_21 : 8;
    __REG32    PRI_22 : 8;
    __REG32    PRI_23 : 8;
} __ip5_bits;


typedef struct {
    __REG32    PRI_24 : 8;
    __REG32    PRI_25 : 8;
    __REG32    PRI_26 : 8;
    __REG32    PRI_27 : 8;
} __ip6_bits;


typedef struct {
    __REG32    PRI_28 : 8;
    __REG32    PRI_29 : 8;
    __REG32    PRI_30 : 8;
    __REG32    PRI_31 : 8;
} __ip7_bits;


typedef struct {
    __REG32    PRI_32 : 8;
    __REG32    PRI_33 : 8;
    __REG32    PRI_34 : 8;
    __REG32    PRI_35 : 8;
} __ip8_bits;


typedef struct {
    __REG32    PRI_36 : 8;
    __REG32    PRI_37 : 8;
    __REG32    PRI_38 : 8;
    __REG32    PRI_39 : 8;
} __ip9_bits;


typedef struct {
    __REG32    PRI_40 : 8;
    __REG32    PRI_41 : 8;
    __REG32    PRI_42 : 8;
    __REG32    PRI_43 : 8;
} __ip10_bits;


typedef struct {
    __REG32    PRI_44 : 8;
    __REG32    PRI_45 : 8;
    __REG32    PRI_46 : 8;
    __REG32    PRI_47 : 8;
} __ip11_bits;


typedef struct {
    __REG32    PRI_48 : 8;
    __REG32    PRI_49 : 8;
    __REG32    PRI_50 : 8;
    __REG32    PRI_51 : 8;
} __ip12_bits;


typedef struct {
    __REG32    PRI_52 : 8;
    __REG32    PRI_53 : 8;
    __REG32    PRI_54 : 8;
    __REG32    PRI_55 : 8;
} __ip13_bits;


typedef struct {
    __REG32    PRI_56 : 8;
    __REG32    PRI_57 : 8;
    __REG32    PRI_58 : 8;
    __REG32    PRI_59 : 8;
} __ip14_bits;


typedef struct {
    __REG32    PRI_60 : 8;
    __REG32    PRI_61 : 8;
    __REG32    PRI_62 : 8;
    __REG32    PRI_63 : 8;
} __ip15_bits;


typedef struct {
    __REG32    PRI_64 : 8;
    __REG32    PRI_65 : 8;
    __REG32    PRI_66 : 8;
    __REG32    PRI_67 : 8;
} __ip16_bits;


typedef struct {
    __REG32    REVISION    : 4;
    __REG32    PARTNO      : 12;
    __REG32                : 4;
    __REG32    VARIANT     : 4;
    __REG32    IMPLEMENTER : 8;
} __cpuid_bits;


typedef struct {
    __REG32    VECTACTIVE  : 9;
    __REG32                : 2;
    __REG32    RETTOBASE   : 1;
    __REG32    VECTPENDING : 10;
    __REG32    ISRPENDING  : 1;
    __REG32    ISRPREEMPT  : 1;
    __REG32                : 1;
    __REG32    PENDSTCLR   : 1;
    __REG32    PENDSTSET   : 1;
    __REG32    PENDSVCLR   : 1;
    __REG32    PENDSVSET   : 1;
    __REG32                : 2;
    __REG32    NMIPENDSET  : 1;
} __icsr_bits;


typedef struct {
    __REG32            : 7;
    __REG32    TBLOFF  : 22;
    __REG32    TBLBASE : 1;
    __REG32            : 2;
} __vtor_bits;


typedef struct {
    __REG32    VECTRESET     : 1;
    __REG32    VECTCLRACTIVE : 1;
    __REG32    SYSRESETREQ   : 1;
    __REG32                  : 5;
    __REG32    PRIGROUP      : 3;
    __REG32                  : 4;
    __REG32    ENDIANESS     : 1;
    __REG32    VECTKEY       : 16;
} __aircr_bits;


typedef struct {
    __REG32    PRI_4 : 8;
    __REG32    PRI_5 : 8;
    __REG32    PRI_6 : 8;
    __REG32    PRI_7 : 8;
} __shp0_bits;


typedef struct {
    __REG32    PRI_8  : 8;
    __REG32    PRI_9  : 8;
    __REG32    PRI_10 : 8;
    __REG32    PRI_11 : 8;
} __shp1_bits;


typedef struct {
    __REG32    PRI_12 : 8;
    __REG32    PRI_13 : 8;
    __REG32    PRI_14 : 8;
    __REG32    PRI_15 : 8;
} __shp2_bits;


typedef struct {
    __REG32    MEMFAULTACT    : 1;
    __REG32    BUSFAULTACT    : 1;
    __REG32                   : 1;
    __REG32    USGFAULTACT    : 1;
    __REG32                   : 3;
    __REG32    SVCALLACT      : 1;
    __REG32    MONITORACT     : 1;
    __REG32                   : 1;
    __REG32    PENDSVACT      : 1;
    __REG32    SYSTICKACT     : 1;
    __REG32    USGFAULTPENDED : 1;
    __REG32    MEMFAULTPENDED : 1;
    __REG32    BUSFAULTPENDED : 1;
    __REG32    SVCALLPENDED   : 1;
    __REG32    MEMFAULTENA    : 1;
    __REG32    BUSFAULTENA    : 1;
    __REG32    USGFAULTENA    : 1;
    __REG32                   : 13;
} __shcsr_bits;


typedef struct {
    __REG32    INTID : 9;
    __REG32          : 23;
} __stir_bits;


typedef struct {
    __REG32                : 1;
    __REG32    SLEEPONEXIT : 1;
    __REG32    SLEEPDEEP   : 1;
    __REG32                : 1;
    __REG32    SEVONPEND   : 1;
    __REG32                : 27;
} __scr_bits;


typedef struct {
    __REG32    NONEBASETHRDENA : 1;
    __REG32    USERSETMPEND    : 1;
    __REG32                    : 1;
    __REG32    UNALIGN_TRP     : 1;
    __REG32    DIV_0_TRP       : 1;
    __REG32                    : 3;
    __REG32    BFHFNMIGN       : 1;
    __REG32    STKALIGN        : 1;
    __REG32                    : 22;
} __ccr_bits;


typedef struct {
    __REG32    ENABLE    : 1;
    __REG32    TICKINT   : 1;
    __REG32    CLKSOURCE : 1;
    __REG32              : 13;
    __REG32    COUNTFLAG : 1;
    __REG32              : 15;
} __ctrl_bits;


typedef struct {
    __REG32    RELOAD : 24;
    __REG32           : 8;
} __load_bits;


typedef struct {
    __REG32    CURRENT : 24;
    __REG32            : 8;
} __val_bits;


typedef struct {
    __REG32    TENMS : 24;
    __REG32          : 6;
    __REG32    SKEW  : 1;
    __REG32    NOREF : 1;
} __calib_bits;


typedef union{
  /*CFSR*/
  struct {
    __REG32    _MFSR : 8;
    __REG32    _BFSR : 8;
    __REG32    _UFSR : 16;
  };
  struct
  {
    union
    {
      /*MFSR*/
      struct{
      __REG8    IACCVIOL  : 1;
      __REG8    DACCVIOL  : 1;
      __REG8              : 1;
      __REG8    MUNSTKERR : 1;
      __REG8    MSTKERR   : 1;
      __REG8              : 2;
      __REG8    MMARVALID : 1;
      } __byte0_bit;
      __REG8 __byte0;
    };
    union
    {
      /*BFSR*/
      struct{
      __REG8     IBUSERR     : 1;
      __REG8     PRECISERR   : 1;
      __REG8     IMPRECISERR : 1;
      __REG8     UNSTKERR    : 1;
      __REG8     STKERR      : 1;
      __REG8                 : 2;
      __REG8     BFARVALID   : 1;
      } __byte1_bit;
      __REG8 __byte1;
    };
    union
    {
      /*UFSR*/
      struct{
      __REG16    UNDEFINSTR : 1;
      __REG16    INVSTATE   : 1;
      __REG16    INVPC      : 1;
      __REG16    NOCP       : 1;
      __REG16               : 4;
      __REG16    UNALIGNED  : 1;
      __REG16    DIVBYZERO  : 1;
      __REG16               : 6;
      } __shortu_bit;
      __REG16 __shortu;
    };
  };
} __cfsr_bits;


typedef struct {
    __REG32    ADDRESS : 32;
} __mmfar_bits;


typedef struct {
    __REG32    ADDRESS : 32;
} __bfar_bits;


typedef struct {
    __REG32             : 1;
    __REG32    VECTTBL  : 1;
    __REG32             : 28;
    __REG32    FORCED   : 1;
    __REG32    DEBUGEVT : 1;
} __hfsr_bits;


typedef struct {
    __REG32    HALTED   : 1;
    __REG32    BKPT     : 1;
    __REG32    DWTTRAP  : 1;
    __REG32    VCATCH   : 1;
    __REG32    EXTERNAL : 1;
    __REG32             : 27;
} __dfsr_bits;


typedef struct {
    __REG32    IMPDEF : 32;
} __afsr_bits;


typedef struct {
    __REG32    TADB : 32;
} __fmc_tadr_bits;


typedef struct {
    __REG32    WRDB : 32;
} __fmc_wrdr_bits;


typedef struct {
    __REG32    CMD  : 4;
    __REG32         : 28;
} __fmc_ocmr_bits;


typedef struct {
    __REG32         : 1;
    __REG32    OPM  : 4;
    __REG32         : 27;
} __fmc_opcr_bits;


typedef struct {
    __REG32    ORFIEN  : 1;
    __REG32    ITADIEN : 1;
    __REG32    OBEIEN  : 1;
    __REG32    IOCMIEN : 1;
    __REG32    OREIEN  : 1;
    __REG32            : 27;
} __fmc_oier_bits;


typedef struct {
    __REG32    ORFF  : 1;
    __REG32    ITADF : 1;
    __REG32    OBEF  : 1;
    __REG32    IOCMF : 1;
    __REG32    OREF  : 1;
    __REG32          : 11;
    __REG32    RORFF : 1;
    __REG32    PPEF  : 1;
    __REG32          : 14;
} __fmc_oisr_bits;


typedef struct {
    __REG32    PPSB : 32;
} __fmc_ppsr0_bits;


typedef struct {
    __REG32    PPSB : 32;
} __fmc_ppsr1_bits;


typedef struct {
    __REG32    PPSB : 32;
} __fmc_ppsr2_bits;


typedef struct {
    __REG32    PPSB : 32;
} __fmc_ppsr3_bits;


typedef struct {
    __REG32    CPSB  : 1;
    __REG32    OBPSB : 1;
    __REG32          : 30;
} __fmc_cpsr_bits;


typedef struct {
    __REG32    VMCB : 2;
    __REG32         : 30;
} __fmc_vmcr_bits;


typedef struct {
    __REG32    WAIT    : 3;
    __REG32            : 1;
    __REG32    PFBE    : 1;
    __REG32            : 2;
    __REG32    DCDB    : 1;
    __REG32            : 4;
    __REG32    CE      : 1;
    __REG32            : 2;
    __REG32    FHLAEN  : 1;
    __REG32    FZWPSEN : 1;
    __REG32            : 15;
} __fmc_cfcr_bits;


typedef struct {
    __REG32    SBVT : 32;
} __fmc_sbvt0_bits;


typedef struct {
    __REG32    SBVT : 32;
} __fmc_sbvt1_bits;


typedef struct {
    __REG32    SBVT : 32;
} __fmc_sbvt2_bits;


typedef struct {
    __REG32    SBVT : 32;
} __fmc_sbvt3_bits;


typedef struct {
    __REG32    BAKPORF : 1;
    __REG32    PDF     : 1;
    __REG32            : 6;
    __REG32    WUPF    : 1;
    __REG32            : 23;
} __pwrcu_baksr_bits;


typedef struct {
    __REG32    BAKRST   : 1;
    __REG32             : 2;
    __REG32    LDOOFF   : 1;
    __REG32             : 3;
    __REG32    DMOSON   : 1;
    __REG32    WUPEN    : 1;
    __REG32    WUPIEN   : 1;
    __REG32             : 2;
    __REG32    V18RDYSC : 1;
    __REG32             : 2;
    __REG32    DMOSSTS  : 1;
    __REG32             : 16;
} __pwrcu_bakcr_bits;


typedef struct {
    __REG32    BAKTEST : 8;
    __REG32            : 24;
} __pwrcu_baktest_bits;


typedef struct {
    __REG32    HSIRCBL : 2;
    __REG32            : 30;
} __pwrcu_hsircr_bits;


typedef struct {
    __REG32    BODEN   : 1;
    __REG32    BODRIS  : 1;
    __REG32            : 1;
    __REG32    BODF    : 1;
    __REG32            : 12;
    __REG32    LVDEN   : 1;
    __REG32    LVDS    : 2;
    __REG32    LVDF    : 1;
    __REG32    LVDIWEN : 1;
    __REG32    LVDEWEN : 1;
    __REG32            : 10;
} __pwrcu_lvdcsr_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg0_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg1_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg2_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg3_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg4_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg5_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg6_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg7_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg8_bits;


typedef struct {
    __REG32    BAKREG : 32;
} __pwrcu_bakreg9_bits;


typedef struct {
    __REG32    CKOUTSRC : 3;
    __REG32    WDTSRC   : 1;
    __REG32             : 4;
    __REG32    PLLSRC   : 1;
    __REG32             : 11;
    __REG32    URPRE    : 2;
    __REG32    USBPRE   : 2;
    __REG32             : 5;
    __REG32    LPMOD    : 3;
} __ckcu_gcfgr_bits;


typedef struct {
    __REG32    SW     : 2;
    __REG32           : 7;
    __REG32    PLLEN  : 1;
    __REG32    HSEEN  : 1;
    __REG32    HSIEN  : 1;
    __REG32           : 4;
    __REG32    CKMEN  : 1;
    __REG32    PSRCEN : 1;
    __REG32           : 14;
} __ckcu_gccr_bits;


typedef struct {
    __REG32           : 1;
    __REG32    PLLRDY : 1;
    __REG32    HSERDY : 1;
    __REG32    HSIRDY : 1;
    __REG32    LSERDY : 1;
    __REG32    LSIRDY : 1;
    __REG32           : 26;
} __ckcu_gcsr_bits;


typedef struct {
    __REG32    CKSF     : 1;
    __REG32             : 1;
    __REG32    PLLRDYF  : 1;
    __REG32    HSERDYF  : 1;
    __REG32    HSIRDYF  : 1;
    __REG32    LSERDYF  : 1;
    __REG32    LSIRDYF  : 1;
    __REG32             : 9;
    __REG32    CKSIE    : 1;
    __REG32             : 1;
    __REG32    PLLRDYIE : 1;
    __REG32    HSERDYIE : 1;
    __REG32    HSIRDYIE : 1;
    __REG32    LSERDYIE : 1;
    __REG32    LSIRDYIE : 1;
    __REG32             : 9;
} __ckcu_gcir_bits;


typedef struct {
    __REG32         : 21;
    __REG32    POTD : 2;
    __REG32    PFBD : 6;
    __REG32         : 3;
} __ckcu_pllcfgr_bits;


typedef struct {
    __REG32           : 31;
    __REG32    PLLBPS : 1;
} __ckcu_pllcr_bits;


typedef struct {
    __REG32    AHBPRE : 2;
    __REG32           : 30;
} __ckcu_ahbcfgr_bits;


typedef struct {
    __REG32    FMCEN   : 1;
    __REG32            : 1;
    __REG32    SRAMEN  : 1;
    __REG32            : 1;
    __REG32    PDMAEN  : 1;
    __REG32    BMEN    : 1;
    __REG32    APB0EN  : 1;
    __REG32    APB1EN  : 1;
    __REG32            : 24;
} __ckcu_ahbccr_bits;


typedef struct {
    __REG32           : 16;
    __REG32    ADCDIV : 3;
    __REG32           : 13;
} __ckcu_apbcfgr_bits;


typedef struct {
    __REG32    I2C0EN : 1;
    __REG32    I2C1EN : 1;
    __REG32           : 2;
    __REG32    SPI0EN : 1;
    __REG32    SPI1EN : 1;
    __REG32           : 2;
    __REG32    UR0EN  : 1;
    __REG32    UR1EN  : 1;
    __REG32           : 4;
    __REG32    AFIOEN : 1;
    __REG32    EXTIEN : 1;
    __REG32    PAEN   : 1;
    __REG32    PBEN   : 1;
    __REG32    PCEN   : 1;
    __REG32    PDEN   : 1;
    __REG32    PEEN   : 1;
    __REG32           : 3;
    __REG32    SCIEN  : 1;
    __REG32           : 7;
} __ckcu_apbccr0_bits;


typedef struct {
    __REG32    MCTMEN  : 1;
    __REG32            : 3;
    __REG32    WDTEN   : 1;
    __REG32            : 1;
    __REG32    RTCEN   : 1;
    __REG32            : 1;
    __REG32    GPTM0EN : 1;
    __REG32    GPTM1EN : 1;
    __REG32            : 4;
    __REG32    USBEN   : 1;
    __REG32            : 1;
    __REG32    BFTM0EN : 1;
    __REG32    BFTM1EN : 1;
    __REG32            : 4;
    __REG32    OPA0EN  : 1;
    __REG32    OPA1EN  : 1;
    __REG32    ADCEN   : 1;
    __REG32            : 7;
} __ckcu_apbccr1_bits;


typedef struct {
    __REG32           : 8;
    __REG32    PLLST  : 4;
    __REG32           : 4;
    __REG32    HSEST  : 2;
    __REG32           : 6;
    __REG32    HSIST  : 3;
    __REG32           : 3;
    __REG32    CKSWST : 2;
} __ckcu_ckst_bits;


typedef struct {
    __REG32    BKISO    : 1;
    __REG32             : 7;
    __REG32    USBSLEEP : 1;
    __REG32             : 23;
} __ckcu_lpcr_bits;


typedef struct {
    __REG32    DBSLP   : 1;
    __REG32    DBDSLP1 : 1;
    __REG32    DBPD    : 1;
    __REG32    DBWDT   : 1;
    __REG32    DBMCTM  : 1;
    __REG32            : 1;
    __REG32    DBGPTM0 : 1;
    __REG32    DBGPTM1 : 1;
    __REG32    DBUR0   : 1;
    __REG32    DBUR1   : 1;
    __REG32    DBSPI0  : 1;
    __REG32    DBSPI1  : 1;
    __REG32    DBI2C0  : 1;
    __REG32    DBI2C1  : 1;
    __REG32    DBDSLP2 : 1;
    __REG32    DBDSCI  : 1;
    __REG32    DBBFTM0 : 1;
    __REG32    DBBFTM1 : 1;
    __REG32            : 14;
} __ckcu_mcudbgcr_bits;


typedef struct {
    __REG32    SYSRSTF : 1;
    __REG32    EXTRSTF : 1;
    __REG32    WDTRSTF : 1;
    __REG32    PORSTF  : 1;
    __REG32            : 28;
} __rstcu_grsr_bits;


typedef struct {
    __REG32    DMARST : 1;
    __REG32           : 31;
} __rstcu_ahbprstr_bits;


typedef struct {
    __REG32    I2C0RST : 1;
    __REG32    I2C1RST : 1;
    __REG32            : 2;
    __REG32    SPI0RST : 1;
    __REG32    SPI1RST : 1;
    __REG32            : 2;
    __REG32    UR0RST  : 1;
    __REG32    UR1RST  : 1;
    __REG32            : 4;
    __REG32    AFIORST : 1;
    __REG32    EXTIRST : 1;
    __REG32    PARST   : 1;
    __REG32    PBRST   : 1;
    __REG32    PCRST   : 1;
    __REG32    PDRST   : 1;
    __REG32    PERST   : 1;
    __REG32            : 3;
    __REG32    SCIRST  : 1;
    __REG32            : 7;
} __rstcu_apbprstr0_bits;


typedef struct {
    __REG32    MCTMRST  : 1;
    __REG32             : 3;
    __REG32    WDTRST   : 1;
    __REG32             : 3;
    __REG32    GPTM0RST : 1;
    __REG32    GPTM1RST : 1;
    __REG32             : 4;
    __REG32    USBRST   : 1;
    __REG32             : 1;
    __REG32    BFTM0RST : 1;
    __REG32    BFTM1RST : 1;
    __REG32             : 4;
    __REG32    OPA0RST  : 1;
    __REG32    OPA1RST  : 1;
    __REG32    ADCRST   : 1;
    __REG32             : 7;
} __rstcu_apbprstr1_bits;


typedef struct {
    __REG32    DIR0  : 1;
    __REG32    DIR1  : 1;
    __REG32    DIR2  : 1;
    __REG32    DIR3  : 1;
    __REG32    DIR4  : 1;
    __REG32    DIR5  : 1;
    __REG32    DIR6  : 1;
    __REG32    DIR7  : 1;
    __REG32    DIR8  : 1;
    __REG32    DIR9  : 1;
    __REG32    DIR10 : 1;
    __REG32    DIR11 : 1;
    __REG32    DIR12 : 1;
    __REG32    DIR13 : 1;
    __REG32    DIR14 : 1;
    __REG32    DIR15 : 1;
    __REG32          : 16;
} __gpioa_dircr_bits;


typedef struct {
    __REG32    INEN0  : 1;
    __REG32    INEN1  : 1;
    __REG32    INEN2  : 1;
    __REG32    INEN3  : 1;
    __REG32    INEN4  : 1;
    __REG32    INEN5  : 1;
    __REG32    INEN6  : 1;
    __REG32    INEN7  : 1;
    __REG32    INEN8  : 1;
    __REG32    INEN9  : 1;
    __REG32    INEN10 : 1;
    __REG32    INEN11 : 1;
    __REG32    INEN12 : 1;
    __REG32    INEN13 : 1;
    __REG32    INEN14 : 1;
    __REG32    INEN15 : 1;
    __REG32           : 16;
} __gpioa_iner_bits;


typedef struct {
    __REG32    PU0  : 1;
    __REG32    PU1  : 1;
    __REG32    PU2  : 1;
    __REG32    PU3  : 1;
    __REG32    PU4  : 1;
    __REG32    PU5  : 1;
    __REG32    PU6  : 1;
    __REG32    PU7  : 1;
    __REG32    PU8  : 1;
    __REG32    PU9  : 1;
    __REG32    PU10 : 1;
    __REG32    PU11 : 1;
    __REG32    PU12 : 1;
    __REG32    PU13 : 1;
    __REG32    PU14 : 1;
    __REG32    PU15 : 1;
    __REG32         : 16;
} __gpioa_pur_bits;


typedef struct {
    __REG32    PD0  : 1;
    __REG32    PD1  : 1;
    __REG32    PD2  : 1;
    __REG32    PD3  : 1;
    __REG32    PD4  : 1;
    __REG32    PD5  : 1;
    __REG32    PD6  : 1;
    __REG32    PD7  : 1;
    __REG32    PD8  : 1;
    __REG32    PD9  : 1;
    __REG32    PD10 : 1;
    __REG32    PD11 : 1;
    __REG32    PD12 : 1;
    __REG32    PD13 : 1;
    __REG32    PD14 : 1;
    __REG32    PD15 : 1;
    __REG32         : 16;
} __gpioa_pdr_bits;


typedef struct {
    __REG32    OD0  : 1;
    __REG32    OD1  : 1;
    __REG32    OD2  : 1;
    __REG32    OD3  : 1;
    __REG32    OD4  : 1;
    __REG32    OD5  : 1;
    __REG32    OD6  : 1;
    __REG32    OD7  : 1;
    __REG32    OD8  : 1;
    __REG32    OD9  : 1;
    __REG32    OD10 : 1;
    __REG32    OD11 : 1;
    __REG32    OD12 : 1;
    __REG32    OD13 : 1;
    __REG32    OD14 : 1;
    __REG32    OD15 : 1;
    __REG32         : 16;
} __gpioa_odr_bits;


typedef struct {
    __REG32    DV0  : 1;
    __REG32    DV1  : 1;
    __REG32    DV2  : 1;
    __REG32    DV3  : 1;
    __REG32    DV4  : 1;
    __REG32    DV5  : 1;
    __REG32    DV6  : 1;
    __REG32    DV7  : 1;
    __REG32         : 24;
} __gpioa_drvr_bits;


typedef struct {
    __REG32    LOCK0  : 1;
    __REG32    LOCK1  : 1;
    __REG32    LOCK2  : 1;
    __REG32    LOCK3  : 1;
    __REG32    LOCK4  : 1;
    __REG32    LOCK5  : 1;
    __REG32    LOCK6  : 1;
    __REG32    LOCK7  : 1;
    __REG32    LOCK8  : 1;
    __REG32    LOCK9  : 1;
    __REG32    LOCK10 : 1;
    __REG32    LOCK11 : 1;
    __REG32    LOCK12 : 1;
    __REG32    LOCK13 : 1;
    __REG32    LOCK14 : 1;
    __REG32    LOCK15 : 1;
    __REG32    LKEY   : 16;
} __gpioa_lockr_bits;


typedef struct {
    __REG32    DIN0  : 1;
    __REG32    DIN1  : 1;
    __REG32    DIN2  : 1;
    __REG32    DIN3  : 1;
    __REG32    DIN4  : 1;
    __REG32    DIN5  : 1;
    __REG32    DIN6  : 1;
    __REG32    DIN7  : 1;
    __REG32    DIN8  : 1;
    __REG32    DIN9  : 1;
    __REG32    DIN10 : 1;
    __REG32    DIN11 : 1;
    __REG32    DIN12 : 1;
    __REG32    DIN13 : 1;
    __REG32    DIN14 : 1;
    __REG32    DIN15 : 1;
    __REG32          : 16;
} __gpioa_dinr_bits;


typedef struct {
    __REG32    DOUT0  : 1;
    __REG32    DOUT1  : 1;
    __REG32    DOUT2  : 1;
    __REG32    DOUT3  : 1;
    __REG32    DOUT4  : 1;
    __REG32    DOUT5  : 1;
    __REG32    DOUT6  : 1;
    __REG32    DOUT7  : 1;
    __REG32    DOUT8  : 1;
    __REG32    DOUT9  : 1;
    __REG32    DOUT10 : 1;
    __REG32    DOUT11 : 1;
    __REG32    DOUT12 : 1;
    __REG32    DOUT13 : 1;
    __REG32    DOUT14 : 1;
    __REG32    DOUT15 : 1;
    __REG32           : 16;
} __gpioa_doutr_bits;


typedef struct {
    __REG32    SET0  : 1;
    __REG32    SET1  : 1;
    __REG32    SET2  : 1;
    __REG32    SET3  : 1;
    __REG32    SET4  : 1;
    __REG32    SET5  : 1;
    __REG32    SET6  : 1;
    __REG32    SET7  : 1;
    __REG32    SET8  : 1;
    __REG32    SET9  : 1;
    __REG32    SET10 : 1;
    __REG32    SET11 : 1;
    __REG32    SET12 : 1;
    __REG32    SET13 : 1;
    __REG32    SET14 : 1;
    __REG32    SET15 : 1;
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
} __gpioa_srr_bits;


typedef struct {
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
    __REG32          : 16;
} __gpioa_rr_bits;


typedef struct {
    __REG32    DIR0  : 1;
    __REG32    DIR1  : 1;
    __REG32    DIR2  : 1;
    __REG32    DIR3  : 1;
    __REG32    DIR4  : 1;
    __REG32    DIR5  : 1;
    __REG32    DIR6  : 1;
    __REG32    DIR7  : 1;
    __REG32    DIR8  : 1;
    __REG32    DIR9  : 1;
    __REG32    DIR10 : 1;
    __REG32    DIR11 : 1;
    __REG32    DIR12 : 1;
    __REG32    DIR13 : 1;
    __REG32    DIR14 : 1;
    __REG32    DIR15 : 1;
    __REG32          : 16;
} __gpiob_dircr_bits;


typedef struct {
    __REG32    INEN0  : 1;
    __REG32    INEN1  : 1;
    __REG32    INEN2  : 1;
    __REG32    INEN3  : 1;
    __REG32    INEN4  : 1;
    __REG32    INEN5  : 1;
    __REG32    INEN6  : 1;
    __REG32    INEN7  : 1;
    __REG32    INEN8  : 1;
    __REG32    INEN9  : 1;
    __REG32    INEN10 : 1;
    __REG32    INEN11 : 1;
    __REG32    INEN12 : 1;
    __REG32    INEN13 : 1;
    __REG32    INEN14 : 1;
    __REG32    INEN15 : 1;
    __REG32           : 16;
} __gpiob_iner_bits;


typedef struct {
    __REG32    PU0  : 1;
    __REG32    PU1  : 1;
    __REG32    PU2  : 1;
    __REG32    PU3  : 1;
    __REG32    PU4  : 1;
    __REG32    PU5  : 1;
    __REG32    PU6  : 1;
    __REG32    PU7  : 1;
    __REG32    PU8  : 1;
    __REG32    PU9  : 1;
    __REG32    PU10 : 1;
    __REG32    PU11 : 1;
    __REG32    PU12 : 1;
    __REG32    PU13 : 1;
    __REG32    PU14 : 1;
    __REG32    PU15 : 1;
    __REG32         : 16;
} __gpiob_pur_bits;


typedef struct {
    __REG32    PD0  : 1;
    __REG32    PD1  : 1;
    __REG32    PD2  : 1;
    __REG32    PD3  : 1;
    __REG32    PD4  : 1;
    __REG32    PD5  : 1;
    __REG32    PD6  : 1;
    __REG32    PD7  : 1;
    __REG32    PD8  : 1;
    __REG32    PD9  : 1;
    __REG32    PD10 : 1;
    __REG32    PD11 : 1;
    __REG32    PD12 : 1;
    __REG32    PD13 : 1;
    __REG32    PD14 : 1;
    __REG32    PD15 : 1;
    __REG32         : 16;
} __gpiob_pdr_bits;


typedef struct {
    __REG32    OD0  : 1;
    __REG32    OD1  : 1;
    __REG32    OD2  : 1;
    __REG32    OD3  : 1;
    __REG32    OD4  : 1;
    __REG32    OD5  : 1;
    __REG32    OD6  : 1;
    __REG32    OD7  : 1;
    __REG32    OD8  : 1;
    __REG32    OD9  : 1;
    __REG32    OD10 : 1;
    __REG32    OD11 : 1;
    __REG32    OD12 : 1;
    __REG32    OD13 : 1;
    __REG32    OD14 : 1;
    __REG32    OD15 : 1;
    __REG32         : 16;
} __gpiob_odr_bits;


typedef struct {
    __REG32    LOCK0  : 1;
    __REG32    LOCK1  : 1;
    __REG32    LOCK2  : 1;
    __REG32    LOCK3  : 1;
    __REG32    LOCK4  : 1;
    __REG32    LOCK5  : 1;
    __REG32    LOCK6  : 1;
    __REG32    LOCK7  : 1;
    __REG32    LOCK8  : 1;
    __REG32    LOCK9  : 1;
    __REG32    LOCK10 : 1;
    __REG32    LOCK11 : 1;
    __REG32    LOCK12 : 1;
    __REG32    LOCK13 : 1;
    __REG32    LOCK14 : 1;
    __REG32    LOCK15 : 1;
    __REG32    LKEY   : 16;
} __gpiob_lockr_bits;


typedef struct {
    __REG32    DIN0  : 1;
    __REG32    DIN1  : 1;
    __REG32    DIN2  : 1;
    __REG32    DIN3  : 1;
    __REG32    DIN4  : 1;
    __REG32    DIN5  : 1;
    __REG32    DIN6  : 1;
    __REG32    DIN7  : 1;
    __REG32    DIN8  : 1;
    __REG32    DIN9  : 1;
    __REG32    DIN10 : 1;
    __REG32    DIN11 : 1;
    __REG32    DIN12 : 1;
    __REG32    DIN13 : 1;
    __REG32    DIN14 : 1;
    __REG32    DIN15 : 1;
    __REG32          : 16;
} __gpiob_dinr_bits;


typedef struct {
    __REG32    DOUT0  : 1;
    __REG32    DOUT1  : 1;
    __REG32    DOUT2  : 1;
    __REG32    DOUT3  : 1;
    __REG32    DOUT4  : 1;
    __REG32    DOUT5  : 1;
    __REG32    DOUT6  : 1;
    __REG32    DOUT7  : 1;
    __REG32    DOUT8  : 1;
    __REG32    DOUT9  : 1;
    __REG32    DOUT10 : 1;
    __REG32    DOUT11 : 1;
    __REG32    DOUT12 : 1;
    __REG32    DOUT13 : 1;
    __REG32    DOUT14 : 1;
    __REG32    DOUT15 : 1;
    __REG32           : 16;
} __gpiob_doutr_bits;


typedef struct {
    __REG32    SET0  : 1;
    __REG32    SET1  : 1;
    __REG32    SET2  : 1;
    __REG32    SET3  : 1;
    __REG32    SET4  : 1;
    __REG32    SET5  : 1;
    __REG32    SET6  : 1;
    __REG32    SET7  : 1;
    __REG32    SET8  : 1;
    __REG32    SET9  : 1;
    __REG32    SET10 : 1;
    __REG32    SET11 : 1;
    __REG32    SET12 : 1;
    __REG32    SET13 : 1;
    __REG32    SET14 : 1;
    __REG32    SET15 : 1;
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
} __gpiob_srr_bits;


typedef struct {
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
    __REG32          : 16;
} __gpiob_rr_bits;


typedef struct {
    __REG32    DIR0  : 1;
    __REG32    DIR1  : 1;
    __REG32    DIR2  : 1;
    __REG32    DIR3  : 1;
    __REG32    DIR4  : 1;
    __REG32    DIR5  : 1;
    __REG32    DIR6  : 1;
    __REG32    DIR7  : 1;
    __REG32    DIR8  : 1;
    __REG32    DIR9  : 1;
    __REG32    DIR10 : 1;
    __REG32    DIR11 : 1;
    __REG32    DIR12 : 1;
    __REG32    DIR13 : 1;
    __REG32    DIR14 : 1;
    __REG32    DIR15 : 1;
    __REG32          : 16;
} __gpioc_dircr_bits;


typedef struct {
    __REG32    INEN0  : 1;
    __REG32    INEN1  : 1;
    __REG32    INEN2  : 1;
    __REG32    INEN3  : 1;
    __REG32    INEN4  : 1;
    __REG32    INEN5  : 1;
    __REG32    INEN6  : 1;
    __REG32    INEN7  : 1;
    __REG32    INEN8  : 1;
    __REG32    INEN9  : 1;
    __REG32    INEN10 : 1;
    __REG32    INEN11 : 1;
    __REG32    INEN12 : 1;
    __REG32    INEN13 : 1;
    __REG32    INEN14 : 1;
    __REG32    INEN15 : 1;
    __REG32           : 16;
} __gpioc_iner_bits;


typedef struct {
    __REG32    PU0  : 1;
    __REG32    PU1  : 1;
    __REG32    PU2  : 1;
    __REG32    PU3  : 1;
    __REG32    PU4  : 1;
    __REG32    PU5  : 1;
    __REG32    PU6  : 1;
    __REG32    PU7  : 1;
    __REG32    PU8  : 1;
    __REG32    PU9  : 1;
    __REG32    PU10 : 1;
    __REG32    PU11 : 1;
    __REG32    PU12 : 1;
    __REG32    PU13 : 1;
    __REG32    PU14 : 1;
    __REG32    PU15 : 1;
    __REG32         : 16;
} __gpioc_pur_bits;


typedef struct {
    __REG32    PD0  : 1;
    __REG32    PD1  : 1;
    __REG32    PD2  : 1;
    __REG32    PD3  : 1;
    __REG32    PD4  : 1;
    __REG32    PD5  : 1;
    __REG32    PD6  : 1;
    __REG32    PD7  : 1;
    __REG32    PD8  : 1;
    __REG32    PD9  : 1;
    __REG32    PD10 : 1;
    __REG32    PD11 : 1;
    __REG32    PD12 : 1;
    __REG32    PD13 : 1;
    __REG32    PD14 : 1;
    __REG32    PD15 : 1;
    __REG32         : 16;
} __gpioc_pdr_bits;


typedef struct {
    __REG32    OD0  : 1;
    __REG32    OD1  : 1;
    __REG32    OD2  : 1;
    __REG32    OD3  : 1;
    __REG32    OD4  : 1;
    __REG32    OD5  : 1;
    __REG32    OD6  : 1;
    __REG32    OD7  : 1;
    __REG32    OD8  : 1;
    __REG32    OD9  : 1;
    __REG32    OD10 : 1;
    __REG32    OD11 : 1;
    __REG32    OD12 : 1;
    __REG32    OD13 : 1;
    __REG32    OD14 : 1;
    __REG32    OD15 : 1;
    __REG32         : 16;
} __gpioc_odr_bits;


typedef struct {
    __REG32    LOCK0  : 1;
    __REG32    LOCK1  : 1;
    __REG32    LOCK2  : 1;
    __REG32    LOCK3  : 1;
    __REG32    LOCK4  : 1;
    __REG32    LOCK5  : 1;
    __REG32    LOCK6  : 1;
    __REG32    LOCK7  : 1;
    __REG32    LOCK8  : 1;
    __REG32    LOCK9  : 1;
    __REG32    LOCK10 : 1;
    __REG32    LOCK11 : 1;
    __REG32    LOCK12 : 1;
    __REG32    LOCK13 : 1;
    __REG32    LOCK14 : 1;
    __REG32    LOCK15 : 1;
    __REG32    LKEY   : 16;
} __gpioc_lockr_bits;


typedef struct {
    __REG32    DIN0  : 1;
    __REG32    DIN1  : 1;
    __REG32    DIN2  : 1;
    __REG32    DIN3  : 1;
    __REG32    DIN4  : 1;
    __REG32    DIN5  : 1;
    __REG32    DIN6  : 1;
    __REG32    DIN7  : 1;
    __REG32    DIN8  : 1;
    __REG32    DIN9  : 1;
    __REG32    DIN10 : 1;
    __REG32    DIN11 : 1;
    __REG32    DIN12 : 1;
    __REG32    DIN13 : 1;
    __REG32    DIN14 : 1;
    __REG32    DIN15 : 1;
    __REG32          : 16;
} __gpioc_dinr_bits;


typedef struct {
    __REG32    DOUT0  : 1;
    __REG32    DOUT1  : 1;
    __REG32    DOUT2  : 1;
    __REG32    DOUT3  : 1;
    __REG32    DOUT4  : 1;
    __REG32    DOUT5  : 1;
    __REG32    DOUT6  : 1;
    __REG32    DOUT7  : 1;
    __REG32    DOUT8  : 1;
    __REG32    DOUT9  : 1;
    __REG32    DOUT10 : 1;
    __REG32    DOUT11 : 1;
    __REG32    DOUT12 : 1;
    __REG32    DOUT13 : 1;
    __REG32    DOUT14 : 1;
    __REG32    DOUT15 : 1;
    __REG32           : 16;
} __gpioc_doutr_bits;


typedef struct {
    __REG32    SET0  : 1;
    __REG32    SET1  : 1;
    __REG32    SET2  : 1;
    __REG32    SET3  : 1;
    __REG32    SET4  : 1;
    __REG32    SET5  : 1;
    __REG32    SET6  : 1;
    __REG32    SET7  : 1;
    __REG32    SET8  : 1;
    __REG32    SET9  : 1;
    __REG32    SET10 : 1;
    __REG32    SET11 : 1;
    __REG32    SET12 : 1;
    __REG32    SET13 : 1;
    __REG32    SET14 : 1;
    __REG32    SET15 : 1;
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
} __gpioc_srr_bits;


typedef struct {
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
    __REG32          : 16;
} __gpioc_rr_bits;


typedef struct {
    __REG32    DIR0  : 1;
    __REG32    DIR1  : 1;
    __REG32    DIR2  : 1;
    __REG32    DIR3  : 1;
    __REG32    DIR4  : 1;
    __REG32    DIR5  : 1;
    __REG32    DIR6  : 1;
    __REG32    DIR7  : 1;
    __REG32    DIR8  : 1;
    __REG32    DIR9  : 1;
    __REG32    DIR10 : 1;
    __REG32    DIR11 : 1;
    __REG32    DIR12 : 1;
    __REG32    DIR13 : 1;
    __REG32    DIR14 : 1;
    __REG32    DIR15 : 1;
    __REG32          : 16;
} __gpiod_dircr_bits;


typedef struct {
    __REG32    INEN0  : 1;
    __REG32    INEN1  : 1;
    __REG32    INEN2  : 1;
    __REG32    INEN3  : 1;
    __REG32    INEN4  : 1;
    __REG32    INEN5  : 1;
    __REG32    INEN6  : 1;
    __REG32    INEN7  : 1;
    __REG32    INEN8  : 1;
    __REG32    INEN9  : 1;
    __REG32    INEN10 : 1;
    __REG32    INEN11 : 1;
    __REG32    INEN12 : 1;
    __REG32    INEN13 : 1;
    __REG32    INEN14 : 1;
    __REG32    INEN15 : 1;
    __REG32           : 16;
} __gpiod_iner_bits;


typedef struct {
    __REG32    PU0  : 1;
    __REG32    PU1  : 1;
    __REG32    PU2  : 1;
    __REG32    PU3  : 1;
    __REG32    PU4  : 1;
    __REG32    PU5  : 1;
    __REG32    PU6  : 1;
    __REG32    PU7  : 1;
    __REG32    PU8  : 1;
    __REG32    PU9  : 1;
    __REG32    PU10 : 1;
    __REG32    PU11 : 1;
    __REG32    PU12 : 1;
    __REG32    PU13 : 1;
    __REG32    PU14 : 1;
    __REG32    PU15 : 1;
    __REG32         : 16;
} __gpiod_pur_bits;


typedef struct {
    __REG32    PD0  : 1;
    __REG32    PD1  : 1;
    __REG32    PD2  : 1;
    __REG32    PD3  : 1;
    __REG32    PD4  : 1;
    __REG32    PD5  : 1;
    __REG32    PD6  : 1;
    __REG32    PD7  : 1;
    __REG32    PD8  : 1;
    __REG32    PD9  : 1;
    __REG32    PD10 : 1;
    __REG32    PD11 : 1;
    __REG32    PD12 : 1;
    __REG32    PD13 : 1;
    __REG32    PD14 : 1;
    __REG32    PD15 : 1;
    __REG32         : 16;
} __gpiod_pdr_bits;


typedef struct {
    __REG32    OD0  : 1;
    __REG32    OD1  : 1;
    __REG32    OD2  : 1;
    __REG32    OD3  : 1;
    __REG32    OD4  : 1;
    __REG32    OD5  : 1;
    __REG32    OD6  : 1;
    __REG32    OD7  : 1;
    __REG32    OD8  : 1;
    __REG32    OD9  : 1;
    __REG32    OD10 : 1;
    __REG32    OD11 : 1;
    __REG32    OD12 : 1;
    __REG32    OD13 : 1;
    __REG32    OD14 : 1;
    __REG32    OD15 : 1;
    __REG32         : 16;
} __gpiod_odr_bits;


typedef struct {
    __REG32    LOCK0  : 1;
    __REG32    LOCK1  : 1;
    __REG32    LOCK2  : 1;
    __REG32    LOCK3  : 1;
    __REG32    LOCK4  : 1;
    __REG32    LOCK5  : 1;
    __REG32    LOCK6  : 1;
    __REG32    LOCK7  : 1;
    __REG32    LOCK8  : 1;
    __REG32    LOCK9  : 1;
    __REG32    LOCK10 : 1;
    __REG32    LOCK11 : 1;
    __REG32    LOCK12 : 1;
    __REG32    LOCK13 : 1;
    __REG32    LOCK14 : 1;
    __REG32    LOCK15 : 1;
    __REG32    LKEY   : 16;
} __gpiod_lockr_bits;


typedef struct {
    __REG32    DIN0  : 1;
    __REG32    DIN1  : 1;
    __REG32    DIN2  : 1;
    __REG32    DIN3  : 1;
    __REG32    DIN4  : 1;
    __REG32    DIN5  : 1;
    __REG32    DIN6  : 1;
    __REG32    DIN7  : 1;
    __REG32    DIN8  : 1;
    __REG32    DIN9  : 1;
    __REG32    DIN10 : 1;
    __REG32    DIN11 : 1;
    __REG32    DIN12 : 1;
    __REG32    DIN13 : 1;
    __REG32    DIN14 : 1;
    __REG32    DIN15 : 1;
    __REG32          : 16;
} __gpiod_dinr_bits;


typedef struct {
    __REG32    DOUT0  : 1;
    __REG32    DOUT1  : 1;
    __REG32    DOUT2  : 1;
    __REG32    DOUT3  : 1;
    __REG32    DOUT4  : 1;
    __REG32    DOUT5  : 1;
    __REG32    DOUT6  : 1;
    __REG32    DOUT7  : 1;
    __REG32    DOUT8  : 1;
    __REG32    DOUT9  : 1;
    __REG32    DOUT10 : 1;
    __REG32    DOUT11 : 1;
    __REG32    DOUT12 : 1;
    __REG32    DOUT13 : 1;
    __REG32    DOUT14 : 1;
    __REG32    DOUT15 : 1;
    __REG32           : 16;
} __gpiod_doutr_bits;


typedef struct {
    __REG32    SET0  : 1;
    __REG32    SET1  : 1;
    __REG32    SET2  : 1;
    __REG32    SET3  : 1;
    __REG32    SET4  : 1;
    __REG32    SET5  : 1;
    __REG32    SET6  : 1;
    __REG32    SET7  : 1;
    __REG32    SET8  : 1;
    __REG32    SET9  : 1;
    __REG32    SET10 : 1;
    __REG32    SET11 : 1;
    __REG32    SET12 : 1;
    __REG32    SET13 : 1;
    __REG32    SET14 : 1;
    __REG32    SET15 : 1;
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
} __gpiod_srr_bits;


typedef struct {
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
    __REG32          : 16;
} __gpiod_rr_bits;


typedef struct {
    __REG32    DIR0  : 1;
    __REG32    DIR1  : 1;
    __REG32    DIR2  : 1;
    __REG32    DIR3  : 1;
    __REG32    DIR4  : 1;
    __REG32    DIR5  : 1;
    __REG32    DIR6  : 1;
    __REG32    DIR7  : 1;
    __REG32    DIR8  : 1;
    __REG32    DIR9  : 1;
    __REG32    DIR10 : 1;
    __REG32    DIR11 : 1;
    __REG32    DIR12 : 1;
    __REG32    DIR13 : 1;
    __REG32    DIR14 : 1;
    __REG32    DIR15 : 1;
    __REG32          : 16;
} __gpioe_dircr_bits;


typedef struct {
    __REG32    INEN0  : 1;
    __REG32    INEN1  : 1;
    __REG32    INEN2  : 1;
    __REG32    INEN3  : 1;
    __REG32    INEN4  : 1;
    __REG32    INEN5  : 1;
    __REG32    INEN6  : 1;
    __REG32    INEN7  : 1;
    __REG32    INEN8  : 1;
    __REG32    INEN9  : 1;
    __REG32    INEN10 : 1;
    __REG32    INEN11 : 1;
    __REG32    INEN12 : 1;
    __REG32    INEN13 : 1;
    __REG32    INEN14 : 1;
    __REG32    INEN15 : 1;
    __REG32           : 16;
} __gpioe_iner_bits;


typedef struct {
    __REG32    PU0  : 1;
    __REG32    PU1  : 1;
    __REG32    PU2  : 1;
    __REG32    PU3  : 1;
    __REG32    PU4  : 1;
    __REG32    PU5  : 1;
    __REG32    PU6  : 1;
    __REG32    PU7  : 1;
    __REG32    PU8  : 1;
    __REG32    PU9  : 1;
    __REG32    PU10 : 1;
    __REG32    PU11 : 1;
    __REG32    PU12 : 1;
    __REG32    PU13 : 1;
    __REG32    PU14 : 1;
    __REG32    PU15 : 1;
    __REG32         : 16;
} __gpioe_pur_bits;


typedef struct {
    __REG32    PD0  : 1;
    __REG32    PD1  : 1;
    __REG32    PD2  : 1;
    __REG32    PD3  : 1;
    __REG32    PD4  : 1;
    __REG32    PD5  : 1;
    __REG32    PD6  : 1;
    __REG32    PD7  : 1;
    __REG32    PD8  : 1;
    __REG32    PD9  : 1;
    __REG32    PD10 : 1;
    __REG32    PD11 : 1;
    __REG32    PD12 : 1;
    __REG32    PD13 : 1;
    __REG32    PD14 : 1;
    __REG32    PD15 : 1;
    __REG32         : 16;
} __gpioe_pdr_bits;


typedef struct {
    __REG32    OD0  : 1;
    __REG32    OD1  : 1;
    __REG32    OD2  : 1;
    __REG32    OD3  : 1;
    __REG32    OD4  : 1;
    __REG32    OD5  : 1;
    __REG32    OD6  : 1;
    __REG32    OD7  : 1;
    __REG32    OD8  : 1;
    __REG32    OD9  : 1;
    __REG32    OD10 : 1;
    __REG32    OD11 : 1;
    __REG32    OD12 : 1;
    __REG32    OD13 : 1;
    __REG32    OD14 : 1;
    __REG32    OD15 : 1;
    __REG32         : 16;
} __gpioe_odr_bits;


typedef struct {
    __REG32         : 5;
    __REG32    DV5  : 1;
    __REG32    DV6  : 1;
    __REG32    DV7  : 1;
    __REG32    DV8  : 1;
    __REG32    DV9  : 1;
    __REG32    DV10 : 1;
    __REG32         : 21;
} __gpioe_drvr_bits;


typedef struct {
    __REG32    LOCK0  : 1;
    __REG32    LOCK1  : 1;
    __REG32    LOCK2  : 1;
    __REG32    LOCK3  : 1;
    __REG32    LOCK4  : 1;
    __REG32    LOCK5  : 1;
    __REG32    LOCK6  : 1;
    __REG32    LOCK7  : 1;
    __REG32    LOCK8  : 1;
    __REG32    LOCK9  : 1;
    __REG32    LOCK10 : 1;
    __REG32    LOCK11 : 1;
    __REG32    LOCK12 : 1;
    __REG32    LOCK13 : 1;
    __REG32    LOCK14 : 1;
    __REG32    LOCK15 : 1;
    __REG32    LKEY   : 16;
} __gpioe_lockr_bits;


typedef struct {
    __REG32    DIN0  : 1;
    __REG32    DIN1  : 1;
    __REG32    DIN2  : 1;
    __REG32    DIN3  : 1;
    __REG32    DIN4  : 1;
    __REG32    DIN5  : 1;
    __REG32    DIN6  : 1;
    __REG32    DIN7  : 1;
    __REG32    DIN8  : 1;
    __REG32    DIN9  : 1;
    __REG32    DIN10 : 1;
    __REG32    DIN11 : 1;
    __REG32    DIN12 : 1;
    __REG32    DIN13 : 1;
    __REG32    DIN14 : 1;
    __REG32    DIN15 : 1;
    __REG32          : 16;
} __gpioe_dinr_bits;


typedef struct {
    __REG32    DOUT0  : 1;
    __REG32    DOUT1  : 1;
    __REG32    DOUT2  : 1;
    __REG32    DOUT3  : 1;
    __REG32    DOUT4  : 1;
    __REG32    DOUT5  : 1;
    __REG32    DOUT6  : 1;
    __REG32    DOUT7  : 1;
    __REG32    DOUT8  : 1;
    __REG32    DOUT9  : 1;
    __REG32    DOUT10 : 1;
    __REG32    DOUT11 : 1;
    __REG32    DOUT12 : 1;
    __REG32    DOUT13 : 1;
    __REG32    DOUT14 : 1;
    __REG32    DOUT15 : 1;
    __REG32           : 16;
} __gpioe_doutr_bits;


typedef struct {
    __REG32    SET0  : 1;
    __REG32    SET1  : 1;
    __REG32    SET2  : 1;
    __REG32    SET3  : 1;
    __REG32    SET4  : 1;
    __REG32    SET5  : 1;
    __REG32    SET6  : 1;
    __REG32    SET7  : 1;
    __REG32    SET8  : 1;
    __REG32    SET9  : 1;
    __REG32    SET10 : 1;
    __REG32    SET11 : 1;
    __REG32    SET12 : 1;
    __REG32    SET13 : 1;
    __REG32    SET14 : 1;
    __REG32    SET15 : 1;
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
} __gpioe_srr_bits;


typedef struct {
    __REG32    RST0  : 1;
    __REG32    RST1  : 1;
    __REG32    RST2  : 1;
    __REG32    RST3  : 1;
    __REG32    RST4  : 1;
    __REG32    RST5  : 1;
    __REG32    RST6  : 1;
    __REG32    RST7  : 1;
    __REG32    RST8  : 1;
    __REG32    RST9  : 1;
    __REG32    RST10 : 1;
    __REG32    RST11 : 1;
    __REG32    RST12 : 1;
    __REG32    RST13 : 1;
    __REG32    RST14 : 1;
    __REG32    RST15 : 1;
    __REG32          : 16;
} __gpioe_rr_bits;


typedef struct {
    __REG32    EXTI0PIN : 4;
    __REG32    EXTI1PIN : 4;
    __REG32    EXTI2PIN : 4;
    __REG32    EXTI3PIN : 4;
    __REG32    EXTI4PIN : 4;
    __REG32    EXTI5PIN : 4;
    __REG32    EXTI6PIN : 4;
    __REG32    EXTI7PIN : 4;
} __afio_essr0_bits;


typedef struct {
    __REG32    EXTI8PIN  : 4;
    __REG32    EXTI9PIN  : 4;
    __REG32    EXTI10PIN : 4;
    __REG32    EXTI11PIN : 4;
    __REG32    EXTI12PIN : 4;
    __REG32    EXTI13PIN : 4;
    __REG32    EXTI14PIN : 4;
    __REG32    EXTI15PIN : 4;
} __afio_essr1_bits;


typedef struct {
    __REG32    PACFG0  : 2;
    __REG32    PACFG1  : 2;
    __REG32    PACFG2  : 2;
    __REG32    PACFG3  : 2;
    __REG32    PACFG4  : 2;
    __REG32    PACFG5  : 2;
    __REG32    PACFG6  : 2;
    __REG32    PACFG7  : 2;
    __REG32    PACFG8  : 2;
    __REG32    PACFG9  : 2;
    __REG32    PACFG10 : 2;
    __REG32    PACFG11 : 2;
    __REG32    PACFG12 : 2;
    __REG32    PACFG13 : 2;
    __REG32    PACFG14 : 2;
    __REG32    PACFG15 : 2;
} __afio_gpacfgr_bits;


typedef struct {
    __REG32    PBCFG0  : 2;
    __REG32    PBCFG1  : 2;
    __REG32    PBCFG2  : 2;
    __REG32    PBCFG3  : 2;
    __REG32    PBCFG4  : 2;
    __REG32    PBCFG5  : 2;
    __REG32    PBCFG6  : 2;
    __REG32    PBCFG7  : 2;
    __REG32    PBCFG8  : 2;
    __REG32    PBCFG9  : 2;
    __REG32    PBCFG10 : 2;
    __REG32    PBCFG11 : 2;
    __REG32    PBCFG12 : 2;
    __REG32    PBCFG13 : 2;
    __REG32    PBCFG14 : 2;
    __REG32    PBCFG15 : 2;
} __afio_gpbcfgr_bits;


typedef struct {
    __REG32    PCCFG0  : 2;
    __REG32    PCCFG1  : 2;
    __REG32    PCCFG2  : 2;
    __REG32    PCCFG3  : 2;
    __REG32    PCCFG4  : 2;
    __REG32    PCCFG5  : 2;
    __REG32    PCCFG6  : 2;
    __REG32    PCCFG7  : 2;
    __REG32    PCCFG8  : 2;
    __REG32    PCCFG9  : 2;
    __REG32    PCCFG10 : 2;
    __REG32    PCCFG11 : 2;
    __REG32    PCCFG12 : 2;
    __REG32    PCCFG13 : 2;
    __REG32    PCCFG14 : 2;
    __REG32    PCCFG15 : 2;
} __afio_gpccfgr_bits;


typedef struct {
    __REG32    PDCFG0  : 2;
    __REG32    PDCFG1  : 2;
    __REG32    PDCFG2  : 2;
    __REG32    PDCFG3  : 2;
    __REG32    PDCFG4  : 2;
    __REG32    PDCFG5  : 2;
    __REG32    PDCFG6  : 2;
    __REG32    PDCFG7  : 2;
    __REG32    PDCFG8  : 2;
    __REG32    PDCFG9  : 2;
    __REG32    PDCFG10 : 2;
    __REG32    PDCFG11 : 2;
    __REG32    PDCFG12 : 2;
    __REG32    PDCFG13 : 2;
    __REG32    PDCFG14 : 2;
    __REG32    PDCFG15 : 2;
} __afio_gpdcfgr_bits;


typedef struct {
    __REG32    PECFG0  : 2;
    __REG32    PECFG1  : 2;
    __REG32    PECFG2  : 2;
    __REG32    PECFG3  : 2;
    __REG32    PECFG4  : 2;
    __REG32    PECFG5  : 2;
    __REG32    PECFG6  : 2;
    __REG32    PECFG7  : 2;
    __REG32    PECFG8  : 2;
    __REG32    PECFG9  : 2;
    __REG32    PECFG10 : 2;
    __REG32    PECFG11 : 2;
    __REG32    PECFG12 : 2;
    __REG32    PECFG13 : 2;
    __REG32    PECFG14 : 2;
    __REG32    PECFG15 : 2;
} __afio_gpecfgr_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr0_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr1_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr2_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr3_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr4_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr5_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr6_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr7_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr8_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr9_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr10_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr11_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr12_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr13_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr14_bits;


typedef struct {
    __REG32    DBCNT   : 28;
    __REG32    SRCTYPE : 3;
    __REG32    DBEN    : 1;
} __exti_cfgr15_bits;


typedef struct {
    __REG32    EXTI0EN  : 1;
    __REG32    EXTI1EN  : 1;
    __REG32    EXTI2EN  : 1;
    __REG32    EXTI3EN  : 1;
    __REG32    EXTI4EN  : 1;
    __REG32    EXTI5EN  : 1;
    __REG32    EXTI6EN  : 1;
    __REG32    EXTI7EN  : 1;
    __REG32    EXTI8EN  : 1;
    __REG32    EXTI9EN  : 1;
    __REG32    EXTI10EN : 1;
    __REG32    EXTI11EN : 1;
    __REG32    EXTI12EN : 1;
    __REG32    EXTI13EN : 1;
    __REG32    EXTI14EN : 1;
    __REG32    EXTI15EN : 1;
    __REG32             : 16;
} __exti_cr_bits;


typedef struct {
    __REG32    EXTI0EDF  : 1;
    __REG32    EXTI1EDF  : 1;
    __REG32    EXTI2EDF  : 1;
    __REG32    EXTI3EDF  : 1;
    __REG32    EXTI4EDF  : 1;
    __REG32    EXTI5EDF  : 1;
    __REG32    EXTI6EDF  : 1;
    __REG32    EXTI7EDF  : 1;
    __REG32    EXTI8EDF  : 1;
    __REG32    EXTI9EDF  : 1;
    __REG32    EXTI10EDF : 1;
    __REG32    EXTI11EDF : 1;
    __REG32    EXTI12EDF : 1;
    __REG32    EXTI13EDF : 1;
    __REG32    EXTI14EDF : 1;
    __REG32    EXTI15EDF : 1;
    __REG32              : 16;
} __exti_edgeflgr_bits;


typedef struct {
    __REG32    EXTI0EDS  : 1;
    __REG32    EXTI1EDS  : 1;
    __REG32    EXTI2EDS  : 1;
    __REG32    EXTI3EDS  : 1;
    __REG32    EXTI4EDS  : 1;
    __REG32    EXTI5EDS  : 1;
    __REG32    EXTI6EDS  : 1;
    __REG32    EXTI7EDS  : 1;
    __REG32    EXTI8EDS  : 1;
    __REG32    EXTI9EDS  : 1;
    __REG32    EXTI10EDS : 1;
    __REG32    EXTI11EDS : 1;
    __REG32    EXTI12EDS : 1;
    __REG32    EXTI13EDS : 1;
    __REG32    EXTI14EDS : 1;
    __REG32    EXTI15EDS : 1;
    __REG32              : 16;
} __exti_edgesr_bits;


typedef struct {
    __REG32    EXTI0SC  : 1;
    __REG32    EXTI1SC  : 1;
    __REG32    EXTI2SC  : 1;
    __REG32    EXTI3SC  : 1;
    __REG32    EXTI4SC  : 1;
    __REG32    EXTI5SC  : 1;
    __REG32    EXTI6SC  : 1;
    __REG32    EXTI7SC  : 1;
    __REG32    EXTI8SC  : 1;
    __REG32    EXTI9SC  : 1;
    __REG32    EXTI10SC : 1;
    __REG32    EXTI11SC : 1;
    __REG32    EXTI12SC : 1;
    __REG32    EXTI13SC : 1;
    __REG32    EXTI14SC : 1;
    __REG32    EXTI15SC : 1;
    __REG32             : 16;
} __exti_sscr_bits;


typedef struct {
    __REG32    EXTI0WEN  : 1;
    __REG32    EXTI1WEN  : 1;
    __REG32    EXTI2WEN  : 1;
    __REG32    EXTI3WEN  : 1;
    __REG32    EXTI4WEN  : 1;
    __REG32    EXTI5WEN  : 1;
    __REG32    EXTI6WEN  : 1;
    __REG32    EXTI7WEN  : 1;
    __REG32    EXTI8WEN  : 1;
    __REG32    EXTI9WEN  : 1;
    __REG32    EXTI10WEN : 1;
    __REG32    EXTI11WEN : 1;
    __REG32    EXTI12WEN : 1;
    __REG32    EXTI13WEN : 1;
    __REG32    EXTI14WEN : 1;
    __REG32    EXTI15WEN : 1;
    __REG32              : 15;
    __REG32    EVWUPIEN  : 1;
} __exti_wakupcr_bits;


typedef struct {
    __REG32    EXTI0POL  : 1;
    __REG32    EXTI1POL  : 1;
    __REG32    EXTI2POL  : 1;
    __REG32    EXTI3POL  : 1;
    __REG32    EXTI4POL  : 1;
    __REG32    EXTI5POL  : 1;
    __REG32    EXTI6POL  : 1;
    __REG32    EXTI7POL  : 1;
    __REG32    EXTI8POL  : 1;
    __REG32    EXTI9POL  : 1;
    __REG32    EXTI10POL : 1;
    __REG32    EXTI11POL : 1;
    __REG32    EXTI12POL : 1;
    __REG32    EXTI13POL : 1;
    __REG32    EXTI14POL : 1;
    __REG32    EXTI15POL : 1;
    __REG32              : 16;
} __exti_wakuppolr_bits;


typedef struct {
    __REG32    EXTI0WFL  : 1;
    __REG32    EXTI1WFL  : 1;
    __REG32    EXTI2WFL  : 1;
    __REG32    EXTI3WFL  : 1;
    __REG32    EXTI4WFL  : 1;
    __REG32    EXTI5WFL  : 1;
    __REG32    EXTI6WFL  : 1;
    __REG32    EXTI7WFL  : 1;
    __REG32    EXTI8WFL  : 1;
    __REG32    EXTI9WFL  : 1;
    __REG32    EXTI10WFL : 1;
    __REG32    EXTI11WFL : 1;
    __REG32    EXTI12WFL : 1;
    __REG32    EXTI13WFL : 1;
    __REG32    EXTI14WFL : 1;
    __REG32    EXTI15WFL : 1;
    __REG32              : 16;
} __exti_wakupflg_bits;


typedef struct {
    __REG32    ADRST : 1;
    __REG32          : 31;
} __adc_rst_bits;


typedef struct {
    __REG32    ADMODE : 2;
    __REG32           : 6;
    __REG32    ADSEQL : 3;
    __REG32           : 5;
    __REG32    ADSUBL : 3;
    __REG32           : 13;
} __adc_conv_bits;


typedef struct {
    __REG32    ADSEQ0 : 5;
    __REG32           : 3;
    __REG32    ADSEQ1 : 5;
    __REG32           : 3;
    __REG32    ADSEQ2 : 5;
    __REG32           : 3;
    __REG32    ADSEQ3 : 5;
    __REG32           : 3;
} __adc_lst0_bits;


typedef struct {
    __REG32    ADSEQ4 : 5;
    __REG32           : 3;
    __REG32    ADSEQ5 : 5;
    __REG32           : 3;
    __REG32    ADSEQ6 : 5;
    __REG32           : 3;
    __REG32    ADSEQ7 : 5;
    __REG32           : 3;
} __adc_lst1_bits;


typedef struct {
    __REG32    ADOF0   : 12;
    __REG32            : 3;
    __REG32    ADOFE0  : 1;
    __REG32            : 16;
} __adc_ofr0_bits;


typedef struct {
    __REG32    ADOF1  : 12;
    __REG32           : 3;
    __REG32    ADOFE1 : 1;
    __REG32           : 16;
} __adc_ofr1_bits;


typedef struct {
    __REG32    ADOF2  : 12;
    __REG32           : 3;
    __REG32    ADOFE2 : 1;
    __REG32           : 16;
} __adc_ofr2_bits;


typedef struct {
    __REG32    ADOF3  : 12;
    __REG32           : 3;
    __REG32    ADOFE3 : 1;
    __REG32           : 16;
} __adc_ofr3_bits;


typedef struct {
    __REG32    ADOF4  : 12;
    __REG32           : 3;
    __REG32    ADOFE4 : 1;
    __REG32           : 16;
} __adc_ofr4_bits;


typedef struct {
    __REG32    ADOF5  : 12;
    __REG32           : 3;
    __REG32    ADOFE5 : 1;
    __REG32           : 16;
} __adc_ofr5_bits;


typedef struct {
    __REG32    ADOF6  : 12;
    __REG32           : 3;
    __REG32    ADOFE6 : 1;
    __REG32           : 16;
} __adc_ofr6_bits;


typedef struct {
    __REG32    ADOF7  : 12;
    __REG32           : 3;
    __REG32    ADOFE7 : 1;
    __REG32           : 16;
} __adc_ofr7_bits;


typedef struct {
    __REG32    ADST0 : 8;
    __REG32          : 24;
} __adc_str0_bits;


typedef struct {
    __REG32    ADST1 : 8;
    __REG32          : 24;
} __adc_str1_bits;


typedef struct {
    __REG32    ADST2 : 8;
    __REG32          : 24;
} __adc_str2_bits;


typedef struct {
    __REG32    ADST3 : 8;
    __REG32          : 24;
} __adc_str3_bits;


typedef struct {
    __REG32    ADST4 : 8;
    __REG32          : 24;
} __adc_str4_bits;


typedef struct {
    __REG32    ADST5 : 8;
    __REG32          : 24;
} __adc_str5_bits;


typedef struct {
    __REG32    ADST6 : 8;
    __REG32          : 24;
} __adc_str6_bits;


typedef struct {
    __REG32    ADST7 : 8;
    __REG32          : 24;
} __adc_str7_bits;


typedef struct {
    __REG32    ADD0   : 16;
    __REG32           : 15;
    __REG32    ADVLD0 : 1;
} __adc_dr0_bits;


typedef struct {
    __REG32    ADD1   : 16;
    __REG32           : 15;
    __REG32    ADVLD1 : 1;
} __adc_dr1_bits;


typedef struct {
    __REG32    ADD2   : 16;
    __REG32           : 15;
    __REG32    ADVLD2 : 1;
} __adc_dr2_bits;


typedef struct {
    __REG32    ADD3   : 16;
    __REG32           : 15;
    __REG32    ADVLD3 : 1;
} __adc_dr3_bits;


typedef struct {
    __REG32    ADD4   : 16;
    __REG32           : 15;
    __REG32    ADVLD4 : 1;
} __adc_dr4_bits;


typedef struct {
    __REG32    ADD5   : 16;
    __REG32           : 15;
    __REG32    ADVLD5 : 1;
} __adc_dr5_bits;


typedef struct {
    __REG32    ADD6   : 16;
    __REG32           : 15;
    __REG32    ADVLD6 : 1;
} __adc_dr6_bits;


typedef struct {
    __REG32    ADD7   : 16;
    __REG32           : 15;
    __REG32    ADVLD7 : 1;
} __adc_dr7_bits;


typedef struct {
    __REG32    ADSW   : 1;
    __REG32    ADEXTI : 1;
    __REG32    TM     : 1;
    __REG32    BFTM   : 1;
    __REG32           : 28;
} __adc_tcr_bits;


typedef struct {
    __REG32    ADSC    : 1;
    __REG32            : 7;
    __REG32    ADEXTIS : 4;
    __REG32            : 4;
    __REG32    TMS     : 3;
    __REG32    BFTMS   : 1;
    __REG32            : 4;
    __REG32    TME     : 3;
    __REG32            : 5;
} __adc_tsr_bits;


typedef struct {
    __REG32    ADWLE  : 1;
    __REG32    ADWUE  : 1;
    __REG32    ADWALL : 1;
    __REG32           : 5;
    __REG32    ADWCH  : 4;
    __REG32           : 4;
    __REG32    ADLCH  : 4;
    __REG32           : 4;
    __REG32    ADUCH  : 4;
    __REG32           : 4;
} __adc_wcr_bits;


typedef struct {
    __REG32    ADLT : 12;
    __REG32         : 20;
} __adc_ltr_bits;


typedef struct {
    __REG32    ADUT : 12;
    __REG32         : 20;
} __adc_utr_bits;


typedef struct {
    __REG32    ADIMS  : 1;
    __REG32    ADIMG  : 1;
    __REG32    ADIMC  : 1;
    __REG32           : 13;
    __REG32    ADIML  : 1;
    __REG32    ADIMU  : 1;
    __REG32           : 6;
    __REG32    ADIMO  : 1;
    __REG32           : 7;
} __adc_imr_bits;


typedef struct {
    __REG32    ADIRAWS  : 1;
    __REG32    ADIRAWG  : 1;
    __REG32    ADIRAWC  : 1;
    __REG32             : 13;
    __REG32    ADIRAWL  : 1;
    __REG32    ADIRAWU  : 1;
    __REG32             : 6;
    __REG32    ADIRAWO  : 1;
    __REG32             : 7;
} __adc_iraw_bits;


typedef struct {
    __REG32    ADIMASKS  : 1;
    __REG32    ADIMASKG  : 1;
    __REG32    ADIMASKC  : 1;
    __REG32              : 13;
    __REG32    ADIMASKL  : 1;
    __REG32    ADIMASKU  : 1;
    __REG32              : 6;
    __REG32    ADIMASKO  : 1;
    __REG32              : 7;
} __adc_imask_bits;


typedef struct {
    __REG32    ADICLRS  : 1;
    __REG32    ADICLRG  : 1;
    __REG32    ADICLRC  : 1;
    __REG32             : 13;
    __REG32    ADICLRL  : 1;
    __REG32    ADICLRU  : 1;
    __REG32             : 6;
    __REG32    ADICLRO  : 1;
    __REG32             : 7;
} __adc_iclr_bits;


typedef struct {
    __REG32    ADDMAS : 1;
    __REG32    ADDMAG : 1;
    __REG32    ADDMAC : 1;
    __REG32           : 29;
} __adc_dmar_bits;


typedef struct {
    __REG32    OPA0EN : 1;
    __REG32    OPC0MS : 1;
    __REG32    A0OFM  : 1;
    __REG32    A0RS   : 1;
    __REG32           : 4;
    __REG32    CMP0S  : 1;
    __REG32           : 23;
} __opacr0_bits;


typedef struct {
    __REG32    A0OF : 6;
    __REG32         : 26;
} __ofvcr0_bits;


typedef struct {
    __REG32    CF0IEN : 1;
    __REG32    CR0IEN : 1;
    __REG32           : 30;
} __cmpier0_bits;


typedef struct {
    __REG32    CF0RAW : 1;
    __REG32    CR0RAW : 1;
    __REG32           : 30;
} __cmprsr0_bits;


typedef struct {
    __REG32    CF0IS : 1;
    __REG32    CR0IS : 1;
    __REG32          : 30;
} __cmpisr0_bits;


typedef struct {
    __REG32    CF0ICLR : 1;
    __REG32    CR0ICLR : 1;
    __REG32            : 30;
} __cmpiclr0_bits;


typedef struct {
    __REG32    OPA1EN : 1;
    __REG32    OPC1MS : 1;
    __REG32    A1OFM  : 1;
    __REG32    A1RS   : 1;
    __REG32           : 4;
    __REG32    CMP1S  : 1;
    __REG32           : 23;
} __opacr1_bits;


typedef struct {
    __REG32    A1OF : 6;
    __REG32         : 26;
} __ofvcr1_bits;


typedef struct {
    __REG32    CF1IEN : 1;
    __REG32    CR1IEN : 1;
    __REG32           : 30;
} __cmpier1_bits;


typedef struct {
    __REG32    CF1RAW : 1;
    __REG32    CR1RAW : 1;
    __REG32           : 30;
} __cmprsr1_bits;


typedef struct {
    __REG32    CF1IS : 1;
    __REG32    CR1IS : 1;
    __REG32          : 30;
} __cmpisr1_bits;


typedef struct {
    __REG32    CF1ICLR : 1;
    __REG32    CR1ICLR : 1;
    __REG32            : 30;
} __cmpiclr1_bits;


typedef struct {
    __REG32    UEV1DIS : 1;
    __REG32    UGDIS   : 1;
    __REG32            : 6;
    __REG32    CKDIV   : 2;
    __REG32            : 6;
    __REG32    CMSEL   : 2;
    __REG32            : 6;
    __REG32    DIR     : 1;
    __REG32            : 7;
} __mctm_cntcfr_bits;


typedef struct {
    __REG32    TSE    : 1;
    __REG32           : 7;
    __REG32    SMSEL  : 3;
    __REG32           : 5;
    __REG32    MMSEL  : 3;
    __REG32           : 5;
    __REG32    SPMSET : 1;
    __REG32           : 7;
} __mctm_mdcfr_bits;


typedef struct {
    __REG32    TRSEL  : 4;
    __REG32           : 4;
    __REG32    ETF    : 4;
    __REG32    ETIPSC : 2;
    __REG32           : 2;
    __REG32    ETIPOL : 1;
    __REG32           : 7;
    __REG32    ECME   : 1;
    __REG32           : 7;
} __mctm_trcfr_bits;


typedef struct {
    __REG32    TME    : 1;
    __REG32    CRBE   : 1;
    __REG32           : 6;
    __REG32    COMPRE : 1;
    __REG32    COMUS  : 1;
    __REG32           : 6;
    __REG32    CHCCDS : 1;
    __REG32           : 15;
} __mctm_ctr_bits;


typedef struct {
    __REG32    TI0F   : 4;
    __REG32           : 12;
    __REG32    CH0CCS : 2;
    __REG32    CH0PSC : 2;
    __REG32           : 11;
    __REG32    TI0SRC : 1;
} __mctm_ch0icfr_bits;


typedef struct {
    __REG32    TI1F   : 4;
    __REG32           : 12;
    __REG32    CH1CCS : 2;
    __REG32    CH1PSC : 2;
    __REG32           : 12;
} __mctm_ch1icfr_bits;


typedef struct {
    __REG32    TI2F   : 4;
    __REG32           : 12;
    __REG32    CH2CCS : 2;
    __REG32    CH2PSC : 2;
    __REG32           : 12;
} __mctm_ch2icfr_bits;


typedef struct {
    __REG32    TI3F   : 4;
    __REG32           : 12;
    __REG32    CH3CCS : 2;
    __REG32    CH3PSC : 2;
    __REG32           : 12;
} __mctm_ch3icfr_bits;


typedef struct {
    __REG32    CH0OM   : 3;
    __REG32    REF0CE  : 1;
    __REG32    CH0PRE  : 1;
    __REG32    CH0IMAE : 1;
    __REG32            : 26;
} __mctm_ch0ocfr_bits;


typedef struct {
    __REG32    CH1OM   : 3;
    __REG32    REF1CE  : 1;
    __REG32    CH1PRE  : 1;
    __REG32    CH1IMAE : 1;
    __REG32            : 26;
} __mctm_ch1ocfr_bits;


typedef struct {
    __REG32    CH2OM   : 3;
    __REG32    REF2CE  : 1;
    __REG32    CH2PRE  : 1;
    __REG32    CH2IMAE : 1;
    __REG32            : 26;
} __mctm_ch2ocfr_bits;


typedef struct {
    __REG32    CH3OM   : 3;
    __REG32    REF3CE  : 1;
    __REG32    CH3PRE  : 1;
    __REG32    CH3IMAE : 1;
    __REG32            : 26;
} __mctm_ch3ocfr_bits;


typedef struct {
    __REG32    CH0E  : 1;
    __REG32    CH0NE : 1;
    __REG32    CH1E  : 1;
    __REG32    CH1NE : 1;
    __REG32    CH2E  : 1;
    __REG32    CH2NE : 1;
    __REG32    CH3E  : 1;
    __REG32          : 25;
} __mctm_chctr_bits;


typedef struct {
    __REG32    CH0P  : 1;
    __REG32    CH0NP : 1;
    __REG32    CH1P  : 1;
    __REG32    CH1NP : 1;
    __REG32    CH2P  : 1;
    __REG32    CH2NP : 1;
    __REG32    CH3P  : 1;
    __REG32          : 25;
} __mctm_chpolr_bits;


typedef struct {
    __REG32    CH0OIS  : 1;
    __REG32    CH0OISN : 1;
    __REG32    CH1OIS  : 1;
    __REG32    CH1OISN : 1;
    __REG32    CH2OIS  : 1;
    __REG32    CH2OISN : 1;
    __REG32    CH3OIS  : 1;
    __REG32            : 25;
} __mctm_chbrkcfr_bits;


typedef struct {
    __REG32    BKE    : 1;
    __REG32    BKP    : 1;
    __REG32           : 2;
    __REG32    CHMOE  : 1;
    __REG32    CHAOE  : 1;
    __REG32           : 2;
    __REG32    BKF    : 4;
    __REG32           : 4;
    __REG32    LOCKLV : 2;
    __REG32           : 2;
    __REG32    CHOSSI : 1;
    __REG32    CHOSSR : 1;
    __REG32           : 2;
    __REG32    CHDTG  : 8;
} __mctm_chbrkctr_bits;


typedef struct {
    __REG32    CH0CCIE : 1;
    __REG32    CH1CCIE : 1;
    __REG32    CH2CCIE : 1;
    __REG32    CH3CCIE : 1;
    __REG32            : 4;
    __REG32    UEV1IE  : 1;
    __REG32    UEV2IE  : 1;
    __REG32    TEVIE   : 1;
    __REG32    BRKIE   : 1;
    __REG32            : 4;
    __REG32    CH0CCDE : 1;
    __REG32    CH1CCDE : 1;
    __REG32    CH2CCDE : 1;
    __REG32    CH3CCDE : 1;
    __REG32            : 4;
    __REG32    UEV1DE  : 1;
    __REG32    UEV2DE  : 1;
    __REG32    TEVDE   : 1;
    __REG32            : 5;
} __mctm_dictr_bits;


typedef struct {
    __REG32    CH0CCG : 1;
    __REG32    CH1CCG : 1;
    __REG32    CH2CCG : 1;
    __REG32    CH3CCG : 1;
    __REG32           : 4;
    __REG32    UEV1G  : 1;
    __REG32    UEV2G  : 1;
    __REG32    TEVG   : 1;
    __REG32    BRKG   : 1;
    __REG32           : 20;
} __mctm_evgr_bits;


typedef struct {
    __REG32    CH0CCIF : 1;
    __REG32    CH1CCIF : 1;
    __REG32    CH2CCIF : 1;
    __REG32    CH3CCIF : 1;
    __REG32    CH0OCF  : 1;
    __REG32    CH1OCF  : 1;
    __REG32    CH2OCF  : 1;
    __REG32    CH3OCF  : 1;
    __REG32    UEV1IF  : 1;
    __REG32    UEV2IF  : 1;
    __REG32    TEVIF   : 1;
    __REG32    BRKIF   : 1;
    __REG32            : 20;
} __mctm_intsr_bits;


typedef struct {
    __REG32    CNTV : 16;
    __REG32         : 16;
} __mctm_cntr_bits;


typedef struct {
    __REG32    PSCV : 16;
    __REG32         : 16;
} __mctm_pscr_bits;


typedef struct {
    __REG32    CRV : 16;
    __REG32        : 16;
} __mctm_crr_bits;


typedef struct {
    __REG32    REPV : 8;
    __REG32         : 24;
} __mctm_repr_bits;


typedef struct {
    __REG32    CH0CCV : 16;
    __REG32           : 16;
} __mctm_ch0ccr_bits;


typedef struct {
    __REG32    CH1CCV : 16;
    __REG32           : 16;
} __mctm_ch1ccr_bits;


typedef struct {
    __REG32    CH2CCV : 16;
    __REG32           : 16;
} __mctm_ch2ccr_bits;


typedef struct {
    __REG32    CH3CCV : 16;
    __REG32           : 16;
} __mctm_ch3ccr_bits;


typedef struct {
    __REG32    UEVDIS : 1;
    __REG32    UGDIS  : 1;
    __REG32           : 6;
    __REG32    CKDIV  : 2;
    __REG32           : 6;
    __REG32    CMSEL  : 2;
    __REG32           : 6;
    __REG32    DIR    : 1;
    __REG32           : 7;
} __gptm0_cntcfr_bits;


typedef struct {
    __REG32    TSE    : 1;
    __REG32           : 7;
    __REG32    SMSEL  : 3;
    __REG32           : 5;
    __REG32    MMSEL  : 3;
    __REG32           : 5;
    __REG32    SPMSET : 1;
    __REG32           : 7;
} __gptm0_mdcfr_bits;


typedef struct {
    __REG32    TRSEL  : 4;
    __REG32           : 4;
    __REG32    ETF    : 4;
    __REG32    ETIPSC : 2;
    __REG32           : 2;
    __REG32    ETIPOL : 1;
    __REG32           : 7;
    __REG32    ECME   : 1;
    __REG32           : 7;
} __gptm0_trcfr_bits;


typedef struct {
    __REG32    TME    : 1;
    __REG32    CRBE   : 1;
    __REG32           : 14;
    __REG32    CHCCDS : 1;
    __REG32           : 15;
} __gptm0_ctr_bits;


typedef struct {
    __REG32    TI0F   : 4;
    __REG32           : 12;
    __REG32    CH0CCS : 2;
    __REG32    CH0PSC : 2;
    __REG32           : 11;
    __REG32    TI0SRC : 1;
} __gptm0_ch0icfr_bits;


typedef struct {
    __REG32    TI1F   : 4;
    __REG32           : 12;
    __REG32    CH1CCS : 2;
    __REG32    CH1PSC : 2;
    __REG32           : 12;
} __gptm0_ch1icfr_bits;


typedef struct {
    __REG32    TI2F   : 4;
    __REG32           : 12;
    __REG32    CH2CCS : 2;
    __REG32    CH2PSC : 2;
    __REG32           : 12;
} __gptm0_ch2icfr_bits;


typedef struct {
    __REG32    TI3F   : 4;
    __REG32           : 12;
    __REG32    CH3CCS : 2;
    __REG32    CH3PSC : 2;
    __REG32           : 12;
} __gptm0_ch3icfr_bits;


typedef struct {
    __REG32    CH0OM   : 3;
    __REG32    REF0CE  : 1;
    __REG32    CH0PRE  : 1;
    __REG32    CH0IMAE : 1;
    __REG32            : 26;
} __gptm0_ch0ocfr_bits;


typedef struct {
    __REG32    CH1OM   : 3;
    __REG32    REF1CE  : 1;
    __REG32    CH1PRE  : 1;
    __REG32    CH1IMAE : 1;
    __REG32            : 26;
} __gptm0_ch1ocfr_bits;


typedef struct {
    __REG32    CH2OM   : 3;
    __REG32    REF2CE  : 1;
    __REG32    CH2PRE  : 1;
    __REG32    CH2IMAE : 1;
    __REG32            : 26;
} __gptm0_ch2ocfr_bits;


typedef struct {
    __REG32    CH3OM   : 3;
    __REG32    REF3CE  : 1;
    __REG32    CH3PRE  : 1;
    __REG32    CH3IMAE : 1;
    __REG32            : 26;
} __gptm0_ch3ocfr_bits;


typedef struct {
    __REG32    CH0E : 1;
    __REG32         : 1;
    __REG32    CH1E : 1;
    __REG32         : 1;
    __REG32    CH2E : 1;
    __REG32         : 1;
    __REG32    CH3E : 1;
    __REG32         : 25;
} __gptm0_chctr_bits;


typedef struct {
    __REG32    CH0P : 1;
    __REG32         : 1;
    __REG32    CH1P : 1;
    __REG32         : 1;
    __REG32    CH2P : 1;
    __REG32         : 1;
    __REG32    CH3P : 1;
    __REG32         : 25;
} __gptm0_chpolr_bits;


typedef struct {
    __REG32    CH0CCIE : 1;
    __REG32    CH1CCIE : 1;
    __REG32    CH2CCIE : 1;
    __REG32    CH3CCIE : 1;
    __REG32            : 4;
    __REG32    UEVIE   : 1;
    __REG32            : 1;
    __REG32    TEVIE   : 1;
    __REG32            : 5;
    __REG32    CH0CCDE : 1;
    __REG32    CH1CCDE : 1;
    __REG32    CH2CCDE : 1;
    __REG32    CH3CCDE : 1;
    __REG32            : 4;
    __REG32    UEVDE   : 1;
    __REG32            : 1;
    __REG32    TEVDE   : 1;
    __REG32            : 5;
} __gptm0_dictr_bits;


typedef struct {
    __REG32    CH0CCG : 1;
    __REG32    CH1CCG : 1;
    __REG32    CH2CCG : 1;
    __REG32    CH3CCG : 1;
    __REG32           : 4;
    __REG32    UEVG   : 1;
    __REG32           : 1;
    __REG32    TEVG   : 1;
    __REG32           : 21;
} __gptm0_evgr_bits;


typedef struct {
    __REG32    CH0CCIF : 1;
    __REG32    CH1CCIF : 1;
    __REG32    CH2CCIF : 1;
    __REG32    CH3CCIF : 1;
    __REG32    CH0OCF  : 1;
    __REG32    CH1OCF  : 1;
    __REG32    CH2OCF  : 1;
    __REG32    CH3OCF  : 1;
    __REG32    UEVIF   : 1;
    __REG32            : 1;
    __REG32    TEVIF   : 1;
    __REG32            : 21;
} __gptm0_intsr_bits;


typedef struct {
    __REG32    CNTV : 16;
    __REG32         : 16;
} __gptm0_cntr_bits;


typedef struct {
    __REG32    PSCV : 16;
    __REG32         : 16;
} __gptm0_pscr_bits;


typedef struct {
    __REG32    CRV : 16;
    __REG32        : 16;
} __gptm0_crr_bits;


typedef struct {
    __REG32    CH0CCV : 16;
    __REG32           : 16;
} __gptm0_ch0ccr_bits;


typedef struct {
    __REG32    CH1CCV : 16;
    __REG32           : 16;
} __gptm0_ch1ccr_bits;


typedef struct {
    __REG32    CH2CCV : 16;
    __REG32           : 16;
} __gptm0_ch2ccr_bits;


typedef struct {
    __REG32    CH3CCV : 16;
    __REG32           : 16;
} __gptm0_ch3ccr_bits;


typedef struct {
    __REG32    UEVDIS  : 1;
    __REG32    UGDIS   : 1;
    __REG32            : 6;
    __REG32    CKDIV   : 2;
    __REG32            : 6;
    __REG32    CMSEL   : 2;
    __REG32            : 6;
    __REG32    DIR     : 1;
    __REG32            : 7;
} __gptm1_cntcfr_bits;


typedef struct {
    __REG32    TSE    : 1;
    __REG32           : 7;
    __REG32    SMSEL  : 3;
    __REG32           : 5;
    __REG32    MMSEL  : 3;
    __REG32           : 5;
    __REG32    SPMSET : 1;
    __REG32           : 7;
} __gptm1_mdcfr_bits;


typedef struct {
    __REG32    TRSEL  : 4;
    __REG32           : 4;
    __REG32    ETF    : 4;
    __REG32    ETIPSC : 2;
    __REG32           : 2;
    __REG32    ETIPOL : 1;
    __REG32           : 7;
    __REG32    ECME   : 1;
    __REG32           : 7;
} __gptm1_trcfr_bits;


typedef struct {
    __REG32    TME    : 1;
    __REG32    CRBE   : 1;
    __REG32           : 14;
    __REG32    CHCCDS : 1;
    __REG32           : 15;
} __gptm1_ctr_bits;


typedef struct {
    __REG32    TI0F   : 4;
    __REG32           : 12;
    __REG32    CH0CCS : 2;
    __REG32    CH0PSC : 2;
    __REG32           : 11;
    __REG32    TI0SRC : 1;
} __gptm1_ch0icfr_bits;


typedef struct {
    __REG32    TI1F   : 4;
    __REG32           : 12;
    __REG32    CH1CCS : 2;
    __REG32    CH1PSC : 2;
    __REG32           : 12;
} __gptm1_ch1icfr_bits;


typedef struct {
    __REG32    TI2F   : 4;
    __REG32           : 12;
    __REG32    CH2CCS : 2;
    __REG32    CH2PSC : 2;
    __REG32           : 12;
} __gptm1_ch2icfr_bits;


typedef struct {
    __REG32    TI3F   : 4;
    __REG32           : 12;
    __REG32    CH3CCS : 2;
    __REG32    CH3PSC : 2;
    __REG32           : 12;
} __gptm1_ch3icfr_bits;


typedef struct {
    __REG32    CH0OM   : 3;
    __REG32    REF0CE  : 1;
    __REG32    CH0PRE  : 1;
    __REG32    CH0IMAE : 1;
    __REG32            : 26;
} __gptm1_ch0ocfr_bits;


typedef struct {
    __REG32    CH1OM   : 3;
    __REG32    REF1CE  : 1;
    __REG32    CH1PRE  : 1;
    __REG32    CH1IMAE : 1;
    __REG32            : 26;
} __gptm1_ch1ocfr_bits;


typedef struct {
    __REG32    CH2OM   : 3;
    __REG32    REF2CE  : 1;
    __REG32    CH2PRE  : 1;
    __REG32    CH2IMAE : 1;
    __REG32            : 26;
} __gptm1_ch2ocfr_bits;


typedef struct {
    __REG32    CH3OM   : 3;
    __REG32    REF3CE  : 1;
    __REG32    CH3PRE  : 1;
    __REG32    CH3IMAE : 1;
    __REG32            : 26;
} __gptm1_ch3ocfr_bits;


typedef struct {
    __REG32    CH0E : 1;
    __REG32         : 1;
    __REG32    CH1E : 1;
    __REG32         : 1;
    __REG32    CH2E : 1;
    __REG32         : 1;
    __REG32    CH3E : 1;
    __REG32         : 25;
} __gptm1_chctr_bits;


typedef struct {
    __REG32    CH0P : 1;
    __REG32         : 1;
    __REG32    CH1P : 1;
    __REG32         : 1;
    __REG32    CH2P : 1;
    __REG32         : 1;
    __REG32    CH3P : 1;
    __REG32         : 25;
} __gptm1_chpolr_bits;


typedef struct {
    __REG32    CH0CCIE : 1;
    __REG32    CH1CCIE : 1;
    __REG32    CH2CCIE : 1;
    __REG32    CH3CCIE : 1;
    __REG32            : 4;
    __REG32    UEVIE   : 1;
    __REG32            : 1;
    __REG32    TEVIE   : 1;
    __REG32            : 5;
    __REG32    CH0CCDE : 1;
    __REG32    CH1CCDE : 1;
    __REG32    CH2CCDE : 1;
    __REG32    CH3CCDE : 1;
    __REG32            : 4;
    __REG32    UEVDE   : 1;
    __REG32            : 1;
    __REG32    TEVDE   : 1;
    __REG32            : 5;
} __gptm1_dictr_bits;


typedef struct {
    __REG32    CH0CCG : 1;
    __REG32    CH1CCG : 1;
    __REG32    CH2CCG : 1;
    __REG32    CH3CCG : 1;
    __REG32           : 4;
    __REG32    UEVG   : 1;
    __REG32           : 1;
    __REG32    TEVG   : 1;
    __REG32           : 21;
} __gptm1_evgr_bits;


typedef struct {
    __REG32    CH0CCIF : 1;
    __REG32    CH1CCIF : 1;
    __REG32    CH2CCIF : 1;
    __REG32    CH3CCIF : 1;
    __REG32    CH0OCF  : 1;
    __REG32    CH1OCF  : 1;
    __REG32    CH2OCF  : 1;
    __REG32    CH3OCF  : 1;
    __REG32    UEVIF   : 1;
    __REG32            : 1;
    __REG32    TEVIF   : 1;
    __REG32            : 21;
} __gptm1_intsr_bits;


typedef struct {
    __REG32    CNTV : 16;
    __REG32         : 16;
} __gptm1_cntr_bits;


typedef struct {
    __REG32    PSCV : 16;
    __REG32         : 16;
} __gptm1_pscr_bits;


typedef struct {
    __REG32    CRV : 16;
    __REG32        : 16;
} __gptm1_crr_bits;


typedef struct {
    __REG32    CH0CCV : 16;
    __REG32           : 16;
} __gptm1_ch0ccr_bits;


typedef struct {
    __REG32    CH1CCV : 16;
    __REG32           : 16;
} __gptm1_ch1ccr_bits;


typedef struct {
    __REG32    CH2CCV : 16;
    __REG32           : 16;
} __gptm1_ch2ccr_bits;


typedef struct {
    __REG32    CH3CCV : 16;
    __REG32           : 16;
} __gptm1_ch3ccr_bits;


typedef struct {
    __REG32    MIEN : 1;
    __REG32    OSM  : 1;
    __REG32    CEN  : 1;
    __REG32         : 29;
} __bftm0_cr_bits;


typedef struct {
    __REG32    MIF  : 1;
    __REG32         : 31;
} __bftm0_sr_bits;


typedef struct {
    __REG32    CNTR : 32;
} __bftm0_cntr_bits;


typedef struct {
    __REG32    CMP : 32;
} __bftm0_cmpr_bits;


typedef struct {
    __REG32    MIEN : 1;
    __REG32    OSM  : 1;
    __REG32    CEN  : 1;
    __REG32         : 29;
} __bftm1_cr_bits;


typedef struct {
    __REG32    MIF  : 1;
    __REG32         : 31;
} __bftm1_sr_bits;


typedef struct {
    __REG32    CNTR : 32;
} __bftm1_cntr_bits;


typedef struct {
    __REG32    CMP : 32;
} __bftm1_cmpr_bits;


typedef struct {
    __REG32    RTCCNT : 32;
} __rtc_cnt_bits;


typedef struct {
    __REG32    RTCCMP : 32;
} __rtc_cmp_bits;


typedef struct {
    __REG32    RTCEN  : 1;
    __REG32    RTCSRC : 1;
    __REG32    LSIEN  : 1;
    __REG32    LSEEN  : 1;
    __REG32    CMPCLR : 1;
    __REG32    LSESM  : 1;
    __REG32           : 2;
    __REG32    RPRE   : 4;
    __REG32           : 4;
    __REG32    ROEN   : 1;
    __REG32    ROES   : 1;
    __REG32    ROWM   : 1;
    __REG32    ROAP   : 1;
    __REG32    ROLF   : 1;
    __REG32           : 11;
} __rtc_cr_bits;


typedef struct {
    __REG32    CSECFLAG : 1;
    __REG32    CMFLAG   : 1;
    __REG32    OVFLAG   : 1;
    __REG32             : 29;
} __rtc_sr_bits;


typedef struct {
    __REG32    CSECIEN : 1;
    __REG32    CMIEN   : 1;
    __REG32    OVIEN   : 1;
    __REG32            : 5;
    __REG32    CSECWEN : 1;
    __REG32    CMWEN   : 1;
    __REG32    OVWEN   : 1;
    __REG32            : 21;
} __rtc_iwen_bits;


typedef struct {
    __REG32    WDTRS : 1;
    __REG32          : 15;
    __REG32    RSKEY : 16;
} __wdt_cr_bits;


typedef struct {
    __REG32    WDTV     : 12;
    __REG32    WDTFIEN  : 1;
    __REG32    WDTRSTEN : 1;
    __REG32             : 18;
} __wdt_mr0_bits;


typedef struct {
    __REG32    WDTD : 12;
    __REG32    WPSC : 3;
    __REG32         : 17;
} __wdt_mr1_bits;


typedef struct {
    __REG32    WDTUF  : 1;
    __REG32    WDTERR : 1;
    __REG32           : 30;
} __wdt_sr_bits;


typedef struct {
    __REG32    PROTECT : 16;
    __REG32            : 16;
} __wdt_pr_bits;


typedef struct {
    __REG32    AA      : 1;
    __REG32    STOP    : 1;
    __REG32    GCEN    : 1;
    __REG32    I2CEN   : 1;
    __REG32            : 3;
    __REG32    ADRM    : 1;
    __REG32    TXDMAE  : 1;
    __REG32    RXDMAE  : 1;
    __REG32    DMANACK : 1;
    __REG32            : 1;
    __REG32    ENTOUT  : 1;
    __REG32            : 19;
} __i2c0_cr_bits;


typedef struct {
    __REG32    STAIE    : 1;
    __REG32    STOIE    : 1;
    __REG32    ADRSIE   : 1;
    __REG32    GCSIE    : 1;
    __REG32             : 4;
    __REG32    ARBLOSIE : 1;
    __REG32    RXNACKIE : 1;
    __REG32    BUSERRIE : 1;
    __REG32    TOUTIE   : 1;
    __REG32             : 4;
    __REG32    RXDNEIE  : 1;
    __REG32    TXDEIE   : 1;
    __REG32    RXBFIE   : 1;
    __REG32             : 13;
} __i2c0_ier_bits;


typedef struct {
    __REG32    ADDR : 10;
    __REG32         : 22;
} __i2c0_addr_bits;


typedef struct {
    __REG32    STA     : 1;
    __REG32    STO     : 1;
    __REG32    ADRS    : 1;
    __REG32    GCS     : 1;
    __REG32            : 4;
    __REG32    ARBLOS  : 1;
    __REG32    RXNACK  : 1;
    __REG32    BUSERR  : 1;
    __REG32    TOUTF   : 1;
    __REG32            : 4;
    __REG32    RXDNE   : 1;
    __REG32    TXDE    : 1;
    __REG32    RXBF    : 1;
    __REG32    BUSBUSY : 1;
    __REG32    MASTER  : 1;
    __REG32    TXNRX   : 1;
    __REG32            : 10;
} __i2c0_sr_bits;


typedef struct {
    __REG32    SHPG : 16;
    __REG32         : 16;
} __i2c0_shpgr_bits;


typedef struct {
    __REG32    SLPG : 16;
    __REG32         : 16;
} __i2c0_slpgr_bits;


typedef struct {
    __REG32    DATA : 8;
    __REG32         : 24;
} __i2c0_dr_bits;


typedef struct {
    __REG32    TAR  : 10;
    __REG32    RWD  : 1;
    __REG32         : 21;
} __i2c0_tar_bits;


typedef struct {
    __REG32    ADDMR : 10;
    __REG32          : 22;
} __i2c0_addmr_bits;


typedef struct {
    __REG32    ADDSR : 10;
    __REG32          : 22;
} __i2c0_addsr_bits;


typedef struct {
    __REG32    TOUT : 16;
    __REG32    PSC  : 3;
    __REG32         : 13;
} __i2c0_tout_bits;


typedef struct {
    __REG32    AA      : 1;
    __REG32    STOP    : 1;
    __REG32    GCEN    : 1;
    __REG32    I2CEN   : 1;
    __REG32            : 3;
    __REG32    ADRM    : 1;
    __REG32    TXDMAE  : 1;
    __REG32    RXDMAE  : 1;
    __REG32    DMANACK : 1;
    __REG32            : 1;
    __REG32    ENTOUT  : 1;
    __REG32            : 19;
} __i2c1_cr_bits;


typedef struct {
    __REG32    STAIE    : 1;
    __REG32    STOIE    : 1;
    __REG32    ADRSIE   : 1;
    __REG32    GCSIE    : 1;
    __REG32             : 4;
    __REG32    ARBLOSIE : 1;
    __REG32    RXNACKIE : 1;
    __REG32    BUSERRIE : 1;
    __REG32    TOUTIE   : 1;
    __REG32             : 4;
    __REG32    RXDNEIE  : 1;
    __REG32    TXDEIE   : 1;
    __REG32    RXBFIE   : 1;
    __REG32             : 13;
} __i2c1_ier_bits;


typedef struct {
    __REG32    ADDR : 10;
    __REG32         : 22;
} __i2c1_addr_bits;


typedef struct {
    __REG32    STA     : 1;
    __REG32    STO     : 1;
    __REG32    ADRS    : 1;
    __REG32    GCS     : 1;
    __REG32            : 4;
    __REG32    ARBLOS  : 1;
    __REG32    RXNACK  : 1;
    __REG32    BUSERR  : 1;
    __REG32    TOUTF   : 1;
    __REG32            : 4;
    __REG32    RXDNE   : 1;
    __REG32    TXDE    : 1;
    __REG32    RXBF    : 1;
    __REG32    BUSBUSY : 1;
    __REG32    MASTER  : 1;
    __REG32    TXNRX   : 1;
    __REG32            : 10;
} __i2c1_sr_bits;


typedef struct {
    __REG32    SHPG : 16;
    __REG32         : 16;
} __i2c1_shpgr_bits;


typedef struct {
    __REG32    SLPG : 16;
    __REG32         : 16;
} __i2c1_slpgr_bits;


typedef struct {
    __REG32    DATA : 8;
    __REG32         : 24;
} __i2c1_dr_bits;


typedef struct {
    __REG32    TAR  : 10;
    __REG32    RWD  : 1;
    __REG32         : 21;
} __i2c1_tar_bits;


typedef struct {
    __REG32    ADDMR : 10;
    __REG32          : 22;
} __i2c1_addmr_bits;


typedef struct {
    __REG32    ADDSR : 10;
    __REG32          : 22;
} __i2c1_addsr_bits;


typedef struct {
    __REG32    TOUT : 16;
    __REG32    PSC  : 3;
    __REG32         : 13;
} __i2c1_tout_bits;


typedef struct {
    __REG32    SPIEN  : 1;
    __REG32    TXDMAE : 1;
    __REG32    RXDMAE : 1;
    __REG32    SELOEN : 1;
    __REG32    SSELC  : 1;
    __REG32           : 27;
} __spi0_cr0_bits;


typedef struct {
    __REG32    DFL      : 4;
    __REG32             : 4;
    __REG32    FORMAT   : 3;
    __REG32    SELAP    : 1;
    __REG32    FIRSTBIT : 1;
    __REG32    SELM     : 1;
    __REG32    MODE     : 1;
    __REG32             : 17;
} __spi0_cr1_bits;


typedef struct {
    __REG32    TXBEIEN  : 1;
    __REG32    TXEIEN   : 1;
    __REG32    RXBNEIEN : 1;
    __REG32    WCIEN    : 1;
    __REG32    ROIEN    : 1;
    __REG32    MFIEN    : 1;
    __REG32    SAIEN    : 1;
    __REG32    TOIEN    : 1;
    __REG32             : 24;
} __spi0_ier_bits;


typedef struct {
    __REG32    CP : 16;
    __REG32       : 16;
} __spi0_cpr_bits;


typedef struct {
    __REG32    DR : 16;
    __REG32       : 16;
} __spi0_dr_bits;


typedef struct {
    __REG32    TXBE  : 1;
    __REG32    TXE   : 1;
    __REG32    RXBNE : 1;
    __REG32    WC    : 1;
    __REG32    RO    : 1;
    __REG32    MF    : 1;
    __REG32    SA    : 1;
    __REG32    TO    : 1;
    __REG32    BUSY  : 1;
    __REG32          : 23;
} __spi0_sr_bits;


typedef struct {
    __REG32    TXFTLS : 4;
    __REG32    RXFTLS : 4;
    __REG32    TFPR   : 1;
    __REG32    RFPR   : 1;
    __REG32    FIFOEN : 1;
    __REG32           : 21;
} __spi0_fcr_bits;


typedef struct {
    __REG32    TXFS : 4;
    __REG32    RXFS : 4;
    __REG32         : 24;
} __spi0_fsr_bits;


typedef struct {
    __REG32    TOC : 32;
} __spi0_ftocr_bits;


typedef struct {
    __REG32    SPIEN  : 1;
    __REG32    TXDMAE : 1;
    __REG32    RXDMAE : 1;
    __REG32    SELOEN : 1;
    __REG32    SSELC  : 1;
    __REG32           : 27;
} __spi1_cr0_bits;


typedef struct {
    __REG32    DFL      : 4;
    __REG32             : 4;
    __REG32    FORMAT   : 3;
    __REG32    SELAP    : 1;
    __REG32    FIRSTBIT : 1;
    __REG32    SELM     : 1;
    __REG32    MODE     : 1;
    __REG32             : 17;
} __spi1_cr1_bits;


typedef struct {
    __REG32    TXBEIEN  : 1;
    __REG32    TXEIEN   : 1;
    __REG32    RXBNEIEN : 1;
    __REG32    WCIEN    : 1;
    __REG32    ROIEN    : 1;
    __REG32    MFIEN    : 1;
    __REG32    SAIEN    : 1;
    __REG32    TOIEN    : 1;
    __REG32             : 24;
} __spi1_ier_bits;


typedef struct {
    __REG32    CP : 16;
    __REG32       : 16;
} __spi1_cpr_bits;


typedef struct {
    __REG32    DR : 16;
    __REG32       : 16;
} __spi1_dr_bits;


typedef struct {
    __REG32    TXBE  : 1;
    __REG32    TXE   : 1;
    __REG32    RXBNE : 1;
    __REG32    WC    : 1;
    __REG32    RO    : 1;
    __REG32    MF    : 1;
    __REG32    SA    : 1;
    __REG32    TO    : 1;
    __REG32    BUSY  : 1;
    __REG32          : 23;
} __spi1_sr_bits;


typedef struct {
    __REG32    TXFTLS : 4;
    __REG32    RXFTLS : 4;
    __REG32    TFPR   : 1;
    __REG32    RFPR   : 1;
    __REG32    FIFOEN : 1;
    __REG32           : 21;
} __spi1_fcr_bits;


typedef struct {
    __REG32    TXFS : 4;
    __REG32    RXFS : 4;
    __REG32         : 24;
} __spi1_fsr_bits;


typedef struct {
    __REG32    TOC : 32;
} __spi1_ftocr_bits;


typedef struct {
    __REG32    RD : 9;
    __REG32       : 23;
} __usart0_rbr_bits;


typedef struct {
    __REG32    TD : 9;
    __REG32       : 23;
} __usart0_tbr_bits;


typedef struct {
    __REG32    RFTLI_RTOIE : 1;
    __REG32    TFTLIE      : 1;
    __REG32    RLSIE       : 1;
    __REG32    MODSIE      : 1;
    __REG32                : 28;
} __usart0_ier_bits;


typedef struct {
    __REG32    NIP  : 1;
    __REG32    IID  : 3;
    __REG32         : 28;
} __usart0_iir_bits;


typedef struct {
    __REG32    FME    : 1;
    __REG32    RFR    : 1;
    __REG32    TFR    : 1;
    __REG32           : 1;
    __REG32    TFTL   : 2;
    __REG32    RFTL   : 2;
    __REG32    URTXEN : 1;
    __REG32    URRXEN : 1;
    __REG32           : 22;
} __usart0_fcr_bits;


typedef struct {
    __REG32    WLS  : 2;
    __REG32    NSB  : 1;
    __REG32    PBE  : 1;
    __REG32    EPE  : 1;
    __REG32    SPE  : 1;
    __REG32    BCB  : 1;
    __REG32         : 25;
} __usart0_lcr_bits;


typedef struct {
    __REG32    DTR   : 1;
    __REG32    RTS   : 1;
    __REG32    HFCEN : 1;
    __REG32          : 29;
} __usart0_modcr_bits;


typedef struct {
    __REG32    RFDR    : 1;
    __REG32    OEI     : 1;
    __REG32    PEI     : 1;
    __REG32    FEI     : 1;
    __REG32    BII     : 1;
    __REG32    TXFEMPT : 1;
    __REG32    TXEMPT  : 1;
    __REG32    ERRRX   : 1;
    __REG32    RSADDEF : 1;
    __REG32            : 23;
} __usart0_lsr_bits;


typedef struct {
    __REG32    DCTS : 1;
    __REG32    DDSR : 1;
    __REG32    DRI  : 1;
    __REG32    DDCD : 1;
    __REG32    CTSS : 1;
    __REG32    DSRS : 1;
    __REG32    RIS  : 1;
    __REG32    DCDS : 1;
    __REG32         : 24;
} __usart0_modsr_bits;


typedef struct {
    __REG32    RTOIC : 7;
    __REG32    RTOIE : 1;
    __REG32    TG    : 8;
    __REG32          : 16;
} __usart0_tpr_bits;


typedef struct {
    __REG32    MODE    : 2;
    __REG32    TRSM    : 1;
    __REG32            : 1;
    __REG32    TXDMAEN : 1;
    __REG32    RXDMAEN : 1;
    __REG32            : 26;
} __usart0_mdr_bits;


typedef struct {
    __REG32    IrDAEN  : 1;
    __REG32    IrDALP  : 1;
    __REG32    TXSEL   : 1;
    __REG32    LB      : 1;
    __REG32    TXINV   : 1;
    __REG32    RXINV   : 1;
    __REG32            : 2;
    __REG32    IrDAPSC : 8;
    __REG32            : 16;
} __usart0_irdacr_bits;


typedef struct {
    __REG32    TXENP    : 1;
    __REG32    RSNMM    : 1;
    __REG32    RSAAD    : 1;
    __REG32             : 5;
    __REG32    ADDMATCH : 8;
    __REG32             : 16;
} __usart0_rs485cr_bits;


typedef struct {
    __REG32    CLKEN : 1;
    __REG32          : 1;
    __REG32    CPS   : 1;
    __REG32    CPO   : 1;
    __REG32          : 28;
} __usart0_syncr_bits;


typedef struct {
    __REG32    TXFS : 5;
    __REG32         : 3;
    __REG32    RXFS : 5;
    __REG32         : 19;
} __usart0_fsr_bits;


typedef struct {
    __REG32    BRD : 16;
    __REG32        : 16;
} __usart0_dlr_bits;


typedef struct {
    __REG32    LBM  : 2;
    __REG32         : 30;
} __usart0_degtstr_bits;


typedef struct {
    __REG32    RD : 9;
    __REG32       : 23;
} __usart1_rbr_bits;


typedef struct {
    __REG32    TD : 9;
    __REG32       : 23;
} __usart1_tbr_bits;


typedef struct {
    __REG32    RFTLI_RTOIE : 1;
    __REG32    TFTLIE      : 1;
    __REG32    RLSIE       : 1;
    __REG32    MODSIE      : 1;
    __REG32                : 28;
} __usart1_ier_bits;


typedef struct {
    __REG32    NIP  : 1;
    __REG32    IID  : 3;
    __REG32         : 28;
} __usart1_iir_bits;


typedef struct {
    __REG32    FME    : 1;
    __REG32    RFR    : 1;
    __REG32    TFR    : 1;
    __REG32           : 1;
    __REG32    TFTL   : 2;
    __REG32    RFTL   : 2;
    __REG32    URTXEN : 1;
    __REG32    URRXEN : 1;
    __REG32           : 22;
} __usart1_fcr_bits;


typedef struct {
    __REG32    WLS  : 2;
    __REG32    NSB  : 1;
    __REG32    PBE  : 1;
    __REG32    EPE  : 1;
    __REG32    SPE  : 1;
    __REG32    BCB  : 1;
    __REG32         : 25;
} __usart1_lcr_bits;


typedef struct {
    __REG32    DTR   : 1;
    __REG32    RTS   : 1;
    __REG32    HFCEN : 1;
    __REG32          : 29;
} __usart1_modcr_bits;


typedef struct {
    __REG32    RFDR    : 1;
    __REG32    OEI     : 1;
    __REG32    PEI     : 1;
    __REG32    FEI     : 1;
    __REG32    BII     : 1;
    __REG32    TXFEMPT : 1;
    __REG32    TXEMPT  : 1;
    __REG32    ERRRX   : 1;
    __REG32    RSADDEF : 1;
    __REG32            : 23;
} __usart1_lsr_bits;


typedef struct {
    __REG32    DCTS : 1;
    __REG32    DDSR : 1;
    __REG32    DRI  : 1;
    __REG32    DDCD : 1;
    __REG32    CTSS : 1;
    __REG32    DSRS : 1;
    __REG32    RIS  : 1;
    __REG32    DCDS : 1;
    __REG32         : 24;
} __usart1_modsr_bits;


typedef struct {
    __REG32    RTOIC : 7;
    __REG32    RTOIE : 1;
    __REG32    TG    : 8;
    __REG32          : 16;
} __usart1_tpr_bits;


typedef struct {
    __REG32    MODE    : 2;
    __REG32    TRSM    : 1;
    __REG32            : 1;
    __REG32    TXDMAEN : 1;
    __REG32    RXDMAEN : 1;
    __REG32            : 26;
} __usart1_mdr_bits;


typedef struct {
    __REG32    IrDAEN  : 1;
    __REG32    IrDALP  : 1;
    __REG32    TXSEL   : 1;
    __REG32    LB      : 1;
    __REG32    TXINV   : 1;
    __REG32    RXINV   : 1;
    __REG32            : 2;
    __REG32    IrDAPSC : 8;
    __REG32            : 16;
} __usart1_irdacr_bits;


typedef struct {
    __REG32    TXENP    : 1;
    __REG32    RSNMM    : 1;
    __REG32    RSAAD    : 1;
    __REG32             : 5;
    __REG32    ADDMATCH : 8;
    __REG32             : 16;
} __usart1_rs485cr_bits;


typedef struct {
    __REG32    CLKEN : 1;
    __REG32          : 1;
    __REG32    CPS   : 1;
    __REG32    CPO   : 1;
    __REG32          : 28;
} __usart1_syncr_bits;


typedef struct {
    __REG32    TXFS : 5;
    __REG32         : 3;
    __REG32    RXFS : 5;
    __REG32         : 19;
} __usart1_fsr_bits;


typedef struct {
    __REG32    BRD : 16;
    __REG32        : 16;
} __usart1_dlr_bits;


typedef struct {
    __REG32    LBM  : 2;
    __REG32         : 30;
} __usart1_degtstr_bits;


typedef struct {
    __REG32    CONV     : 1;
    __REG32    CREP     : 1;
    __REG32    WTEN     : 1;
    __REG32    SCIM     : 1;
    __REG32    RETRY4_5 : 1;
    __REG32    ENSCI    : 1;
    __REG32    DETCNF   : 1;
    __REG32             : 1;
    __REG32    TXDMA    : 1;
    __REG32    RXDMA    : 1;
    __REG32             : 22;
} __sci_cr_bits;


typedef struct {
    __REG32    PARF  : 1;
    __REG32    RXCF  : 1;
    __REG32    TXCF  : 1;
    __REG32    WTF   : 1;
    __REG32          : 2;
    __REG32    CPREF : 1;
    __REG32    TXBEF : 1;
    __REG32          : 24;
} __sci_sr_bits;


typedef struct {
    __REG32           : 2;
    __REG32    CCLK   : 1;
    __REG32    CDIO   : 1;
    __REG32           : 3;
    __REG32    CLKSEL : 1;
    __REG32           : 24;
} __sci_ccr_bits;


typedef struct {
    __REG32    ETU     : 11;
    __REG32            : 4;
    __REG32    ETUCOMP : 1;
    __REG32            : 16;
} __sci_etu_bits;


typedef struct {
    __REG32    GT : 9;
    __REG32       : 23;
} __sci_gt_bits;


typedef struct {
    __REG32    WT : 24;
    __REG32       : 8;
} __sci_wt_bits;


typedef struct {
    __REG32    PARE    : 1;
    __REG32    RXCE    : 1;
    __REG32    TXCE    : 1;
    __REG32    WTE     : 1;
    __REG32            : 2;
    __REG32    CARDIRE : 1;
    __REG32    TXBEE   : 1;
    __REG32            : 24;
} __sci_ier_bits;


typedef struct {
    __REG32    PARP    : 1;
    __REG32    RXCP    : 1;
    __REG32    TXCP    : 1;
    __REG32    WTP     : 1;
    __REG32            : 2;
    __REG32    CARDIRP : 1;
    __REG32    TXBEP   : 1;
    __REG32            : 24;
} __sci_ipr_bits;


typedef struct {
    __REG32    TB : 8;
    __REG32       : 24;
} __sci_txb_bits;


typedef struct {
    __REG32    RB : 8;
    __REG32       : 24;
} __sci_rxb_bits;


typedef struct {
    __REG32    PSC : 6;
    __REG32        : 26;
} __sci_psc_bits;


typedef struct {
    __REG32           : 1;
    __REG32    FRES   : 1;
    __REG32    PDWN   : 1;
    __REG32    LPMODE : 1;
    __REG32           : 1;
    __REG32    GENRSM : 1;
    __REG32    RXDP   : 1;
    __REG32    RXDM   : 1;
    __REG32    ADRSET : 1;
    __REG32           : 23;
} __usb_csr_bits;


typedef struct {
    __REG32    UGIE   : 1;
    __REG32    SOFIE  : 1;
    __REG32    URSTIE : 1;
    __REG32    RSMIE  : 1;
    __REG32    SUSPIE : 1;
    __REG32    ESOFIE : 1;
    __REG32           : 2;
    __REG32    EP0IE  : 1;
    __REG32    EP1IE  : 1;
    __REG32    EP2IE  : 1;
    __REG32    EP3IE  : 1;
    __REG32    EP4IE  : 1;
    __REG32    EP5IE  : 1;
    __REG32    EP6IE  : 1;
    __REG32    EP7IE  : 1;
    __REG32           : 16;
} __usb_ier_bits;


typedef struct {
    __REG32           : 1;
    __REG32    SOFIF  : 1;
    __REG32    URSTIF : 1;
    __REG32    RSMIF  : 1;
    __REG32    SUSPIF : 1;
    __REG32    ESOFIF : 1;
    __REG32           : 2;
    __REG32    EP0IF  : 1;
    __REG32    EP1IF  : 1;
    __REG32    EP2IF  : 1;
    __REG32    EP3IF  : 1;
    __REG32    EP4IF  : 1;
    __REG32    EP5IF  : 1;
    __REG32    EP6IF  : 1;
    __REG32    EP7IF  : 1;
    __REG32           : 16;
} __usb_isr_bits;


typedef struct {
    __REG32    FRNUM  : 11;
    __REG32           : 5;
    __REG32    SOFLCK : 1;
    __REG32    LSOF   : 2;
    __REG32           : 13;
} __usb_fcr_bits;


typedef struct {
    __REG32    DEVA : 7;
    __REG32         : 25;
} __usb_devar_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32          : 26;
} __usb_ep0csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32    STRXIE : 1;
    __REG32    SDRXIE : 1;
    __REG32    SDERIE : 1;
    __REG32    ZLRXIE : 1;
    __REG32           : 20;
} __usb_ep0ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32    STRXIF : 1;
    __REG32    SDRXIF : 1;
    __REG32    SDERIF : 1;
    __REG32    ZLRXIF : 1;
    __REG32           : 20;
} __usb_ep0isr_bits;


typedef struct {
    __REG32    TXCNT : 7;
    __REG32          : 9;
    __REG32    RXCNT : 7;
    __REG32          : 9;
} __usb_ep0tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 7;
    __REG32           : 7;
    __REG32    EPADR  : 4;
    __REG32           : 3;
    __REG32    EPEN   : 1;
} __usb_ep0cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32          : 26;
} __usb_ep1csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep1ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep1isr_bits;


typedef struct {
    __REG32    TCNT : 9;
    __REG32         : 23;
} __usb_ep1tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 7;
    __REG32           : 7;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep1cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32          : 26;
} __usb_ep2csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep2ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep2isr_bits;


typedef struct {
    __REG32    TCNT : 9;
    __REG32         : 23;
} __usb_ep2tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 7;
    __REG32           : 7;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep2cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32          : 26;
} __usb_ep3csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep3ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep3isr_bits;


typedef struct {
    __REG32    TCNT : 9;
    __REG32         : 23;
} __usb_ep3tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 7;
    __REG32           : 7;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep3cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32    MDBTG : 1;
    __REG32    UDBTG : 1;
    __REG32          : 24;
} __usb_ep4csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep4ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep4isr_bits;


typedef struct {
    __REG32    TCNT0 : 10;
    __REG32          : 6;
    __REG32    TCNT1 : 10;
    __REG32          : 6;
} __usb_ep4tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 10;
    __REG32           : 3;
    __REG32    SDBS   : 1;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep4cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32    MDBTG : 1;
    __REG32    UDBTG : 1;
    __REG32          : 24;
} __usb_ep5csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep5ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep5isr_bits;


typedef struct {
    __REG32    TCNT0 : 10;
    __REG32          : 6;
    __REG32    TCNT1 : 10;
    __REG32          : 6;
} __usb_ep5tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 10;
    __REG32           : 3;
    __REG32    SDBS   : 1;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep5cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32    MDBTG : 1;
    __REG32    UDBTG : 1;
    __REG32          : 24;
} __usb_ep6csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep6ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep6isr_bits;


typedef struct {
    __REG32    TCNT0 : 10;
    __REG32          : 6;
    __REG32    TCNT1 : 10;
    __REG32          : 6;
} __usb_ep6tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 10;
    __REG32           : 3;
    __REG32    SDBS   : 1;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep6cfgr_bits;


typedef struct {
    __REG32    DTGTX : 1;
    __REG32    NAKTX : 1;
    __REG32    STLTX : 1;
    __REG32    DTGRX : 1;
    __REG32    NAKRX : 1;
    __REG32    STLRX : 1;
    __REG32    MDBTG : 1;
    __REG32    UDBTG : 1;
    __REG32          : 24;
} __usb_ep7csr_bits;


typedef struct {
    __REG32    OTRXIE : 1;
    __REG32    ODRXIE : 1;
    __REG32    ODOVIE : 1;
    __REG32    ITRXIE : 1;
    __REG32    IDTXIE : 1;
    __REG32    NAKIE  : 1;
    __REG32    STLIE  : 1;
    __REG32    UERIE  : 1;
    __REG32           : 24;
} __usb_ep7ier_bits;


typedef struct {
    __REG32    OTRXIF : 1;
    __REG32    ODRXIF : 1;
    __REG32    ODOVIF : 1;
    __REG32    ITRXIF : 1;
    __REG32    IDTXIF : 1;
    __REG32    NAKIF  : 1;
    __REG32    STLIF  : 1;
    __REG32    UERIF  : 1;
    __REG32           : 24;
} __usb_ep7isr_bits;


typedef struct {
    __REG32    TCNT0 : 10;
    __REG32          : 6;
    __REG32    TCNT1 : 10;
    __REG32          : 6;
} __usb_ep7tcr_bits;


typedef struct {
    __REG32    EPBUFA : 10;
    __REG32    EPLEN  : 10;
    __REG32           : 3;
    __REG32    SDBS   : 1;
    __REG32    EPADR  : 4;
    __REG32    EPDIR  : 1;
    __REG32    EPTYPE : 1;
    __REG32           : 1;
    __REG32    EPEN   : 1;
} __usb_ep7cfgr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch0cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch0sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch0dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch0cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch0tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch0ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch1cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch1sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch1dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch1cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch1tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch1ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch2cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch2sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch2dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch2cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch2tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch2ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch3cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch3sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch3dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch3cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch3tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch3ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch4cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch4sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch4dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch4cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch4tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch4ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch5cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch5sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch5dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch5cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch5tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch5ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch6cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch6sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch6dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch6cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch6tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch6ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch7cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch7sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch7dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch7cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch7tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch7ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch8cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch8sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch8dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch8cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch8tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch8ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch9cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch9sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch9dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch9cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch9tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch9ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch10cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch10sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch10dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch10cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch10tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch10ctsr_bits;


typedef struct {
    __REG32    CHEN    : 1;
    __REG32    SWTRIG  : 1;
    __REG32    DWIDTH  : 2;
    __REG32    DSTAINC : 1;
    __REG32    DSTAMOD : 1;
    __REG32    SRCAINC : 1;
    __REG32    SRCAMOD : 1;
    __REG32    CHPRI   : 2;
    __REG32    FIXAEN  : 1;
    __REG32    AUTORL  : 1;
    __REG32            : 20;
} __pdma_ch11cr_bits;


typedef struct {
    __REG32    SADR : 32;
} __pdma_ch11sadr_bits;


typedef struct {
    __REG32    DADR : 32;
} __pdma_ch11dadr_bits;


typedef struct {
    __REG32    CDADR : 16;
    __REG32    CSADR : 16;
} __pdma_ch11cadr_bits;


typedef struct {
    __REG32    BLKLEN : 16;
    __REG32    BLKCNT : 16;
} __pdma_ch11tsr_bits;


typedef struct {
    __REG32    CBLKLEN : 16;
    __REG32    CBLKCNT : 16;
} __pdma_ch11ctsr_bits;


typedef struct {
    __REG32    GEISTA0 : 1;
    __REG32    BEISTA0 : 1;
    __REG32    HTISTA0 : 1;
    __REG32    TCISTA0 : 1;
    __REG32    TEISTA0 : 1;
    __REG32    GEISTA1 : 1;
    __REG32    BEISTA1 : 1;
    __REG32    HTISTA1 : 1;
    __REG32    TCISTA1 : 1;
    __REG32    TEISTA1 : 1;
    __REG32    GEISTA2 : 1;
    __REG32    BEISTA2 : 1;
    __REG32    HTISTA2 : 1;
    __REG32    TCISTA2 : 1;
    __REG32    TEISTA2 : 1;
    __REG32    GEISTA3 : 1;
    __REG32    BEISTA3 : 1;
    __REG32    HTISTA3 : 1;
    __REG32    TCISTA3 : 1;
    __REG32    TEISTA3 : 1;
    __REG32    GEISTA4 : 1;
    __REG32    BEISTA4 : 1;
    __REG32    HTISTA4 : 1;
    __REG32    TCISTA4 : 1;
    __REG32    TEISTA4 : 1;
    __REG32    GEISTA5 : 1;
    __REG32    BEISTA5 : 1;
    __REG32    HTISTA5 : 1;
    __REG32    TCISTA5 : 1;
    __REG32    TEISTA5 : 1;
    __REG32            : 2;
} __pdma_isr0_bits;


typedef struct {
    __REG32    GEISTA6  : 1;
    __REG32    BEISTA6  : 1;
    __REG32    HTISTA6  : 1;
    __REG32    TCISTA6  : 1;
    __REG32    TEISTA6  : 1;
    __REG32    GEISTA7  : 1;
    __REG32    BEISTA7  : 1;
    __REG32    HTISTA7  : 1;
    __REG32    TCISTA7  : 1;
    __REG32    TEISTA7  : 1;
    __REG32    GEISTA8  : 1;
    __REG32    BEISTA8  : 1;
    __REG32    HTISTA8  : 1;
    __REG32    TCISTA8  : 1;
    __REG32    TEISTA8  : 1;
    __REG32    GEISTA9  : 1;
    __REG32    BEISTA9  : 1;
    __REG32    HTISTA9  : 1;
    __REG32    TCISTA9  : 1;
    __REG32    TEISTA9  : 1;
    __REG32    GEISTA10 : 1;
    __REG32    BEISTA10 : 1;
    __REG32    HTISTA10 : 1;
    __REG32    TCISTA10 : 1;
    __REG32    TEISTA10 : 1;
    __REG32    GEISTA11 : 1;
    __REG32    BEISTA11 : 1;
    __REG32    HTISTA11 : 1;
    __REG32    TCISTA11 : 1;
    __REG32    TEISTA11 : 1;
    __REG32             : 2;
} __pdma_isr1_bits;


typedef struct {
    __REG32    GEICLR0 : 1;
    __REG32    BEICLR0 : 1;
    __REG32    HTICLR0 : 1;
    __REG32    TCICLR0 : 1;
    __REG32    TEICLR0 : 1;
    __REG32    GEICLR1 : 1;
    __REG32    BEICLR1 : 1;
    __REG32    HTICLR1 : 1;
    __REG32    TCICLR1 : 1;
    __REG32    TEICLR1 : 1;
    __REG32    GEICLR2 : 1;
    __REG32    BEICLR2 : 1;
    __REG32    HTICLR2 : 1;
    __REG32    TCICLR2 : 1;
    __REG32    TEICLR2 : 1;
    __REG32    GEICLR3 : 1;
    __REG32    BEICLR3 : 1;
    __REG32    HTICLR3 : 1;
    __REG32    TCICLR3 : 1;
    __REG32    TEICLR3 : 1;
    __REG32    GEICLR4 : 1;
    __REG32    BEICLR4 : 1;
    __REG32    HTICLR4 : 1;
    __REG32    TCICLR4 : 1;
    __REG32    TEICLR4 : 1;
    __REG32    GEICLR5 : 1;
    __REG32    BEICLR5 : 1;
    __REG32    HTICLR5 : 1;
    __REG32    TCICLR5 : 1;
    __REG32    TEICLR5 : 1;
    __REG32            : 2;
} __pdma_iscr0_bits;


typedef struct {
    __REG32    GEICLR6  : 1;
    __REG32    BEICLR6  : 1;
    __REG32    HTICLR6  : 1;
    __REG32    TCICLR6  : 1;
    __REG32    TEICLR6  : 1;
    __REG32    GEICLR7  : 1;
    __REG32    BEICLR7  : 1;
    __REG32    HTICLR7  : 1;
    __REG32    TCICLR7  : 1;
    __REG32    TEICLR7  : 1;
    __REG32    GEICLR8  : 1;
    __REG32    BEICLR8  : 1;
    __REG32    HTICLR8  : 1;
    __REG32    TCICLR8  : 1;
    __REG32    TEICLR8  : 1;
    __REG32    GEICLR9  : 1;
    __REG32    BEICLR9  : 1;
    __REG32    HTICLR9  : 1;
    __REG32    TCICLR9  : 1;
    __REG32    TEICLR9  : 1;
    __REG32    GEICLR10 : 1;
    __REG32    BEICLR10 : 1;
    __REG32    HTICLR10 : 1;
    __REG32    TCICLR10 : 1;
    __REG32    TEICLR10 : 1;
    __REG32    GEICLR11 : 1;
    __REG32    BEICLR11 : 1;
    __REG32    HTICLR11 : 1;
    __REG32    TCICLR11 : 1;
    __REG32    TEICLR11 : 1;
    __REG32             : 2;
} __pdma_iscr1_bits;


typedef struct {
    __REG32    GEIE0 : 1;
    __REG32    BEIE0 : 1;
    __REG32    HTIE0 : 1;
    __REG32    TCIE0 : 1;
    __REG32    TEIE0 : 1;
    __REG32    GEIE1 : 1;
    __REG32    BEIE1 : 1;
    __REG32    HTIE1 : 1;
    __REG32    TCIE1 : 1;
    __REG32    TEIE1 : 1;
    __REG32    GEIE2 : 1;
    __REG32    BEIE2 : 1;
    __REG32    HTIE2 : 1;
    __REG32    TCIE2 : 1;
    __REG32    TEIE2 : 1;
    __REG32    GEIE3 : 1;
    __REG32    BEIE3 : 1;
    __REG32    HTIE3 : 1;
    __REG32    TCIE3 : 1;
    __REG32    TEIE3 : 1;
    __REG32    GEIE4 : 1;
    __REG32    BEIE4 : 1;
    __REG32    HTIE4 : 1;
    __REG32    TCIE4 : 1;
    __REG32    TEIE4 : 1;
    __REG32    GEIE5 : 1;
    __REG32    BEIE5 : 1;
    __REG32    HTIE5 : 1;
    __REG32    TCIE5 : 1;
    __REG32    TEIE5 : 1;
    __REG32          : 2;
} __pdma_ier0_bits;


typedef struct {
    __REG32    GEIE6  : 1;
    __REG32    BEIE6  : 1;
    __REG32    HTIE6  : 1;
    __REG32    TCIE6  : 1;
    __REG32    TEIE6  : 1;
    __REG32    GEIE7  : 1;
    __REG32    BEIE7  : 1;
    __REG32    HTIE7  : 1;
    __REG32    TCIE7  : 1;
    __REG32    TEIE7  : 1;
    __REG32    GEIE8  : 1;
    __REG32    BEIE8  : 1;
    __REG32    HTIE8  : 1;
    __REG32    TCIE8  : 1;
    __REG32    TEIE8  : 1;
    __REG32    GEIE9  : 1;
    __REG32    BEIE9  : 1;
    __REG32    HTIE9  : 1;
    __REG32    TCIE9  : 1;
    __REG32    TEIE9  : 1;
    __REG32    GEIE10 : 1;
    __REG32    BEIE10 : 1;
    __REG32    HTIE10 : 1;
    __REG32    TCIE10 : 1;
    __REG32    TEIE10 : 1;
    __REG32    GEIE11 : 1;
    __REG32    BEIE11 : 1;
    __REG32    HTIE11 : 1;
    __REG32    TCIE11 : 1;
    __REG32    TEIE11 : 1;
    __REG32           : 2;
} __pdma_ier1_bits;


#endif /*__IAR_SYSTEMS_ICC__                                              */

/* Declarations common to compiler and assembler  *************************/

__IO_REG32_BIT(ICTR,                    0xE000E004, __READ_WRITE , __ictr_bits);
__IO_REG32_BIT(ACTLR,                   0xE000E008, __READ_WRITE , __actlr_bits);
__IO_REG32_BIT(ISER0,                   0xE000E100, __READ_WRITE , __iser0_bits);
__IO_REG32_BIT(ISER1,                   0xE000E104, __READ_WRITE , __iser1_bits);
__IO_REG32_BIT(ISER2,                   0xE000E108, __READ_WRITE , __iser2_bits);
__IO_REG32_BIT(ICER0,                   0xE000E180, __READ_WRITE , __icer0_bits);
__IO_REG32_BIT(ICER1,                   0xE000E184, __READ_WRITE , __icer1_bits);
__IO_REG32_BIT(ICER2,                   0xE000E188, __READ_WRITE , __icer2_bits);
__IO_REG32_BIT(ISPR0,                   0xE000E200, __READ_WRITE , __ispr0_bits);
__IO_REG32_BIT(ISPR1,                   0xE000E204, __READ_WRITE , __ispr1_bits);
__IO_REG32_BIT(ISPR2,                   0xE000E208, __READ_WRITE , __ispr2_bits);
__IO_REG32_BIT(ICPR0,                   0xE000E280, __READ_WRITE , __icpr0_bits);
__IO_REG32_BIT(ICPR1,                   0xE000E284, __READ_WRITE , __icpr1_bits);
__IO_REG32_BIT(ICPR2,                   0xE000E288, __READ_WRITE , __icpr2_bits);
__IO_REG32_BIT(IABR0,                   0xE000E300, __READ_WRITE , __iabr0_bits);
__IO_REG32_BIT(IABR1,                   0xE000E304, __READ_WRITE , __iabr1_bits);
__IO_REG32_BIT(IABR2,                   0xE000E308, __READ_WRITE , __iabr2_bits);
__IO_REG32_BIT(IP0,                     0xE000E400, __READ_WRITE , __ip0_bits);
__IO_REG32_BIT(IP1,                     0xE000E404, __READ_WRITE , __ip1_bits);
__IO_REG32_BIT(IP2,                     0xE000E408, __READ_WRITE , __ip2_bits);
__IO_REG32_BIT(IP3,                     0xE000E40C, __READ_WRITE , __ip3_bits);
__IO_REG32_BIT(IP4,                     0xE000E410, __READ_WRITE , __ip4_bits);
__IO_REG32_BIT(IP5,                     0xE000E414, __READ_WRITE , __ip5_bits);
__IO_REG32_BIT(IP6,                     0xE000E418, __READ_WRITE , __ip6_bits);
__IO_REG32_BIT(IP7,                     0xE000E41C, __READ_WRITE , __ip7_bits);
__IO_REG32_BIT(IP8,                     0xE000E420, __READ_WRITE , __ip8_bits);
__IO_REG32_BIT(IP9,                     0xE000E424, __READ_WRITE , __ip9_bits);
__IO_REG32_BIT(IP10,                    0xE000E428, __READ_WRITE , __ip10_bits);
__IO_REG32_BIT(IP11,                    0xE000E42C, __READ_WRITE , __ip11_bits);
__IO_REG32_BIT(IP12,                    0xE000E430, __READ_WRITE , __ip12_bits);
__IO_REG32_BIT(IP13,                    0xE000E434, __READ_WRITE , __ip13_bits);
__IO_REG32_BIT(IP14,                    0xE000E438, __READ_WRITE , __ip14_bits);
__IO_REG32_BIT(IP15,                    0xE000E43C, __READ_WRITE , __ip15_bits);
__IO_REG32_BIT(IP16,                    0xE000E440, __READ_WRITE , __ip16_bits);
__IO_REG32_BIT(CPUID,                   0xE000ED00, __READ_WRITE , __cpuid_bits);
__IO_REG32_BIT(ICSR,                    0xE000ED04, __READ_WRITE , __icsr_bits);
__IO_REG32_BIT(VTOR,                    0xE000ED08, __READ_WRITE , __vtor_bits);
__IO_REG32_BIT(AIRCR,                   0xE000ED0C, __READ_WRITE , __aircr_bits);
__IO_REG32_BIT(SHP0,                    0xE000ED18, __READ_WRITE , __shp0_bits);
__IO_REG32_BIT(SHP1,                    0xE000ED1C, __READ_WRITE , __shp1_bits);
__IO_REG32_BIT(SHP2,                    0xE000ED20, __READ_WRITE , __shp2_bits);
__IO_REG32_BIT(SHCSR,                   0xE000ED24, __READ_WRITE , __shcsr_bits);
__IO_REG32_BIT(STIR,                    0xE000EF00, __READ_WRITE , __stir_bits);
__IO_REG32_BIT(SCR,                     0xE000ED10, __READ_WRITE , __scr_bits);
__IO_REG32_BIT(CCR,                     0xE000ED14, __READ_WRITE , __ccr_bits);
__IO_REG32_BIT(CTRL,                    0xE000E010, __READ_WRITE , __ctrl_bits);
__IO_REG32_BIT(LOAD,                    0xE000E014, __READ_WRITE , __load_bits);
__IO_REG32_BIT(VAL,                     0xE000E018, __READ_WRITE , __val_bits);
__IO_REG32_BIT(CALIB,                   0xE000E01C, __READ_WRITE , __calib_bits);
__IO_REG32_BIT(CFSR,                    0xE000ED28, __READ_WRITE , __cfsr_bits);
#define MFSR              CFSR_bit.__byte0
#define MFSR_bit          CFSR_bit.__byte0_bit
#define BFSR              CFSR_bit.__byte1
#define BFSR_bit          CFSR_bit.__byte1_bit
#define UFSR              CFSR_bit.__shortu
#define UFSR_bit          CFSR_bit.__shortu_bit
__IO_REG32_BIT(MMFAR,                   0xE000ED34, __READ_WRITE , __mmfar_bits);
__IO_REG32_BIT(BFAR,                    0xE000ED38, __READ_WRITE , __bfar_bits);
__IO_REG32_BIT(HFSR,                    0xE000ED2C, __READ_WRITE , __hfsr_bits);
__IO_REG32_BIT(DFSR,                    0xE000ED30, __READ_WRITE , __dfsr_bits);
__IO_REG32_BIT(AFSR,                    0xE000ED3C, __READ_WRITE , __afsr_bits);
__IO_REG32_BIT(FMC_TADR,                0x40080000, __READ_WRITE , __fmc_tadr_bits);
__IO_REG32_BIT(FMC_WRDR,                0x40080004, __READ_WRITE , __fmc_wrdr_bits);
__IO_REG32_BIT(FMC_OCMR,                0x4008000C, __READ_WRITE , __fmc_ocmr_bits);
__IO_REG32_BIT(FMC_OPCR,                0x40080010, __READ_WRITE , __fmc_opcr_bits);
__IO_REG32_BIT(FMC_OIER,                0x40080014, __READ_WRITE , __fmc_oier_bits);
__IO_REG32_BIT(FMC_OISR,                0x40080018, __READ_WRITE , __fmc_oisr_bits);
__IO_REG32_BIT(FMC_PPSR0,               0x40080020, __READ_WRITE , __fmc_ppsr0_bits);
__IO_REG32_BIT(FMC_PPSR1,               0x40080024, __READ_WRITE , __fmc_ppsr1_bits);
__IO_REG32_BIT(FMC_PPSR2,               0x40080028, __READ_WRITE , __fmc_ppsr2_bits);
__IO_REG32_BIT(FMC_PPSR3,               0x4008002C, __READ_WRITE , __fmc_ppsr3_bits);
__IO_REG32_BIT(FMC_CPSR,                0x40080030, __READ_WRITE , __fmc_cpsr_bits);
__IO_REG32_BIT(FMC_VMCR,                0x40080100, __READ_WRITE , __fmc_vmcr_bits);
__IO_REG32_BIT(FMC_CFCR,                0x40080200, __READ_WRITE , __fmc_cfcr_bits);
__IO_REG32_BIT(FMC_SBVT0,               0x40080300, __READ_WRITE , __fmc_sbvt0_bits);
__IO_REG32_BIT(FMC_SBVT1,               0x40080304, __READ_WRITE , __fmc_sbvt1_bits);
__IO_REG32_BIT(FMC_SBVT2,               0x40080308, __READ_WRITE , __fmc_sbvt2_bits);
__IO_REG32_BIT(FMC_SBVT3,               0x4008030C, __READ_WRITE , __fmc_sbvt3_bits);
__IO_REG32_BIT(PWRCU_BAKSR,             0x4006A100, __READ_WRITE , __pwrcu_baksr_bits);
__IO_REG32_BIT(PWRCU_BAKCR,             0x4006A104, __READ_WRITE , __pwrcu_bakcr_bits);
__IO_REG32_BIT(PWRCU_BAKTEST,           0x4006A108, __READ_WRITE , __pwrcu_baktest_bits);
__IO_REG32_BIT(PWRCU_HSIRCR,            0x4006A10C, __READ_WRITE , __pwrcu_hsircr_bits);
__IO_REG32_BIT(PWRCU_LVDCSR,            0x4006A110, __READ_WRITE , __pwrcu_lvdcsr_bits);
__IO_REG32_BIT(PWRCU_BAKREG0,           0x4006A200, __READ_WRITE , __pwrcu_bakreg0_bits);
__IO_REG32_BIT(PWRCU_BAKREG1,           0x4006A204, __READ_WRITE , __pwrcu_bakreg1_bits);
__IO_REG32_BIT(PWRCU_BAKREG2,           0x4006A208, __READ_WRITE , __pwrcu_bakreg2_bits);
__IO_REG32_BIT(PWRCU_BAKREG3,           0x4006A20C, __READ_WRITE , __pwrcu_bakreg3_bits);
__IO_REG32_BIT(PWRCU_BAKREG4,           0x4006A210, __READ_WRITE , __pwrcu_bakreg4_bits);
__IO_REG32_BIT(PWRCU_BAKREG5,           0x4006A214, __READ_WRITE , __pwrcu_bakreg5_bits);
__IO_REG32_BIT(PWRCU_BAKREG6,           0x4006A218, __READ_WRITE , __pwrcu_bakreg6_bits);
__IO_REG32_BIT(PWRCU_BAKREG7,           0x4006A21C, __READ_WRITE , __pwrcu_bakreg7_bits);
__IO_REG32_BIT(PWRCU_BAKREG8,           0x4006A220, __READ_WRITE , __pwrcu_bakreg8_bits);
__IO_REG32_BIT(PWRCU_BAKREG9,           0x4006A224, __READ_WRITE , __pwrcu_bakreg9_bits);
__IO_REG32_BIT(CKCU_GCFGR,              0x40088000, __READ_WRITE , __ckcu_gcfgr_bits);
__IO_REG32_BIT(CKCU_GCCR,               0x40088004, __READ_WRITE , __ckcu_gccr_bits);
__IO_REG32_BIT(CKCU_GCSR,               0x40088008, __READ_WRITE , __ckcu_gcsr_bits);
__IO_REG32_BIT(CKCU_GCIR,               0x4008800C, __READ_WRITE , __ckcu_gcir_bits);
__IO_REG32_BIT(CKCU_PLLCFGR,            0x40088018, __READ_WRITE , __ckcu_pllcfgr_bits);
__IO_REG32_BIT(CKCU_PLLCR,              0x4008801C, __READ_WRITE , __ckcu_pllcr_bits);
__IO_REG32_BIT(CKCU_AHBCFGR,            0x40088020, __READ_WRITE , __ckcu_ahbcfgr_bits);
__IO_REG32_BIT(CKCU_AHBCCR,             0x40088024, __READ_WRITE , __ckcu_ahbccr_bits);
__IO_REG32_BIT(CKCU_APBCFGR,            0x40088028, __READ_WRITE , __ckcu_apbcfgr_bits);
__IO_REG32_BIT(CKCU_APBCCR0,            0x4008802C, __READ_WRITE , __ckcu_apbccr0_bits);
__IO_REG32_BIT(CKCU_APBCCR1,            0x40088030, __READ_WRITE , __ckcu_apbccr1_bits);
__IO_REG32_BIT(CKCU_CKST,               0x40088034, __READ_WRITE , __ckcu_ckst_bits);
__IO_REG32_BIT(CKCU_LPCR,               0x40088300, __READ_WRITE , __ckcu_lpcr_bits);
__IO_REG32_BIT(CKCU_MCUDBGCR,           0x40088304, __READ_WRITE , __ckcu_mcudbgcr_bits);
__IO_REG32_BIT(RSTCU_GRSR,              0x40088100, __READ_WRITE , __rstcu_grsr_bits);
__IO_REG32_BIT(RSTCU_AHBPRSTR,          0x40088104, __READ_WRITE , __rstcu_ahbprstr_bits);
__IO_REG32_BIT(RSTCU_APBPRSTR0,         0x40088108, __READ_WRITE , __rstcu_apbprstr0_bits);
__IO_REG32_BIT(RSTCU_APBPRSTR1,         0x4008810C, __READ_WRITE , __rstcu_apbprstr1_bits);
__IO_REG32_BIT(GPIOA_DIRCR,             0x4001A000, __READ_WRITE , __gpioa_dircr_bits);
__IO_REG32_BIT(GPIOA_INER,              0x4001A004, __READ_WRITE , __gpioa_iner_bits);
__IO_REG32_BIT(GPIOA_PUR,               0x4001A008, __READ_WRITE , __gpioa_pur_bits);
__IO_REG32_BIT(GPIOA_PDR,               0x4001A00C, __READ_WRITE , __gpioa_pdr_bits);
__IO_REG32_BIT(GPIOA_ODR,               0x4001A010, __READ_WRITE , __gpioa_odr_bits);
__IO_REG32_BIT(GPIOA_DRVR,              0x4001A014, __READ_WRITE , __gpioa_drvr_bits);
__IO_REG32_BIT(GPIOA_LOCKR,             0x4001A018, __READ_WRITE , __gpioa_lockr_bits);
__IO_REG32_BIT(GPIOA_DINR,              0x4001A01C, __READ_WRITE , __gpioa_dinr_bits);
__IO_REG32_BIT(GPIOA_DOUTR,             0x4001A020, __READ_WRITE , __gpioa_doutr_bits);
__IO_REG32_BIT(GPIOA_SRR,               0x4001A024, __READ_WRITE , __gpioa_srr_bits);
__IO_REG32_BIT(GPIOA_RR,                0x4001A028, __READ_WRITE , __gpioa_rr_bits);
__IO_REG32_BIT(GPIOB_DIRCR,             0x4001B000, __READ_WRITE , __gpiob_dircr_bits);
__IO_REG32_BIT(GPIOB_INER,              0x4001B004, __READ_WRITE , __gpiob_iner_bits);
__IO_REG32_BIT(GPIOB_PUR,               0x4001B008, __READ_WRITE , __gpiob_pur_bits);
__IO_REG32_BIT(GPIOB_PDR,               0x4001B00C, __READ_WRITE , __gpiob_pdr_bits);
__IO_REG32_BIT(GPIOB_ODR,               0x4001B010, __READ_WRITE , __gpiob_odr_bits);
__IO_REG32_BIT(GPIOB_LOCKR,             0x4001B018, __READ_WRITE , __gpiob_lockr_bits);
__IO_REG32_BIT(GPIOB_DINR,              0x4001B01C, __READ_WRITE , __gpiob_dinr_bits);
__IO_REG32_BIT(GPIOB_DOUTR,             0x4001B020, __READ_WRITE , __gpiob_doutr_bits);
__IO_REG32_BIT(GPIOB_SRR,               0x4001B024, __READ_WRITE , __gpiob_srr_bits);
__IO_REG32_BIT(GPIOB_RR,                0x4001B028, __READ_WRITE , __gpiob_rr_bits);
__IO_REG32_BIT(GPIOC_DIRCR,             0x4001C000, __READ_WRITE , __gpioc_dircr_bits);
__IO_REG32_BIT(GPIOC_INER,              0x4001C004, __READ_WRITE , __gpioc_iner_bits);
__IO_REG32_BIT(GPIOC_PUR,               0x4001C008, __READ_WRITE , __gpioc_pur_bits);
__IO_REG32_BIT(GPIOC_PDR,               0x4001C00C, __READ_WRITE , __gpioc_pdr_bits);
__IO_REG32_BIT(GPIOC_ODR,               0x4001C010, __READ_WRITE , __gpioc_odr_bits);
__IO_REG32_BIT(GPIOC_LOCKR,             0x4001C018, __READ_WRITE , __gpioc_lockr_bits);
__IO_REG32_BIT(GPIOC_DINR,              0x4001C01C, __READ_WRITE , __gpioc_dinr_bits);
__IO_REG32_BIT(GPIOC_DOUTR,             0x4001C020, __READ_WRITE , __gpioc_doutr_bits);
__IO_REG32_BIT(GPIOC_SRR,               0x4001C024, __READ_WRITE , __gpioc_srr_bits);
__IO_REG32_BIT(GPIOC_RR,                0x4001C028, __READ_WRITE , __gpioc_rr_bits);
__IO_REG32_BIT(GPIOD_DIRCR,             0x4001D000, __READ_WRITE , __gpiod_dircr_bits);
__IO_REG32_BIT(GPIOD_INER,              0x4001D004, __READ_WRITE , __gpiod_iner_bits);
__IO_REG32_BIT(GPIOD_PUR,               0x4001D008, __READ_WRITE , __gpiod_pur_bits);
__IO_REG32_BIT(GPIOD_PDR,               0x4001D00C, __READ_WRITE , __gpiod_pdr_bits);
__IO_REG32_BIT(GPIOD_ODR,               0x4001D010, __READ_WRITE , __gpiod_odr_bits);
__IO_REG32_BIT(GPIOD_LOCKR,             0x4001D018, __READ_WRITE , __gpiod_lockr_bits);
__IO_REG32_BIT(GPIOD_DINR,              0x4001D01C, __READ_WRITE , __gpiod_dinr_bits);
__IO_REG32_BIT(GPIOD_DOUTR,             0x4001D020, __READ_WRITE , __gpiod_doutr_bits);
__IO_REG32_BIT(GPIOD_SRR,               0x4001D024, __READ_WRITE , __gpiod_srr_bits);
__IO_REG32_BIT(GPIOD_RR,                0x4001D028, __READ_WRITE , __gpiod_rr_bits);
__IO_REG32_BIT(GPIOE_DIRCR,             0x4001E000, __READ_WRITE , __gpioe_dircr_bits);
__IO_REG32_BIT(GPIOE_INER,              0x4001E004, __READ_WRITE , __gpioe_iner_bits);
__IO_REG32_BIT(GPIOE_PUR,               0x4001E008, __READ_WRITE , __gpioe_pur_bits);
__IO_REG32_BIT(GPIOE_PDR,               0x4001E00C, __READ_WRITE , __gpioe_pdr_bits);
__IO_REG32_BIT(GPIOE_ODR,               0x4001E010, __READ_WRITE , __gpioe_odr_bits);
__IO_REG32_BIT(GPIOE_DRVR,              0x4001E014, __READ_WRITE , __gpioe_drvr_bits);
__IO_REG32_BIT(GPIOE_LOCKR,             0x4001E018, __READ_WRITE , __gpioe_lockr_bits);
__IO_REG32_BIT(GPIOE_DINR,              0x4001E01C, __READ_WRITE , __gpioe_dinr_bits);
__IO_REG32_BIT(GPIOE_DOUTR,             0x4001E020, __READ_WRITE , __gpioe_doutr_bits);
__IO_REG32_BIT(GPIOE_SRR,               0x4001E024, __READ_WRITE , __gpioe_srr_bits);
__IO_REG32_BIT(GPIOE_RR,                0x4001E028, __READ_WRITE , __gpioe_rr_bits);
__IO_REG32_BIT(AFIO_ESSR0,              0x40022000, __READ_WRITE , __afio_essr0_bits);
__IO_REG32_BIT(AFIO_ESSR1,              0x40022004, __READ_WRITE , __afio_essr1_bits);
__IO_REG32_BIT(AFIO_GPACFGR,            0x40022008, __READ_WRITE , __afio_gpacfgr_bits);
__IO_REG32_BIT(AFIO_GPBCFGR,            0x4002200C, __READ_WRITE , __afio_gpbcfgr_bits);
__IO_REG32_BIT(AFIO_GPCCFGR,            0x40022010, __READ_WRITE , __afio_gpccfgr_bits);
__IO_REG32_BIT(AFIO_GPDCFGR,            0x40022014, __READ_WRITE , __afio_gpdcfgr_bits);
__IO_REG32_BIT(AFIO_GPECFGR,            0x40022018, __READ_WRITE , __afio_gpecfgr_bits);
__IO_REG32_BIT(EXTI_CFGR0,              0x40024000, __READ_WRITE , __exti_cfgr0_bits);
__IO_REG32_BIT(EXTI_CFGR1,              0x40024004, __READ_WRITE , __exti_cfgr1_bits);
__IO_REG32_BIT(EXTI_CFGR2,              0x40024008, __READ_WRITE , __exti_cfgr2_bits);
__IO_REG32_BIT(EXTI_CFGR3,              0x4002400C, __READ_WRITE , __exti_cfgr3_bits);
__IO_REG32_BIT(EXTI_CFGR4,              0x40024010, __READ_WRITE , __exti_cfgr4_bits);
__IO_REG32_BIT(EXTI_CFGR5,              0x40024014, __READ_WRITE , __exti_cfgr5_bits);
__IO_REG32_BIT(EXTI_CFGR6,              0x40024018, __READ_WRITE , __exti_cfgr6_bits);
__IO_REG32_BIT(EXTI_CFGR7,              0x4002401C, __READ_WRITE , __exti_cfgr7_bits);
__IO_REG32_BIT(EXTI_CFGR8,              0x40024020, __READ_WRITE , __exti_cfgr8_bits);
__IO_REG32_BIT(EXTI_CFGR9,              0x40024024, __READ_WRITE , __exti_cfgr9_bits);
__IO_REG32_BIT(EXTI_CFGR10,             0x40024028, __READ_WRITE , __exti_cfgr10_bits);
__IO_REG32_BIT(EXTI_CFGR11,             0x4002402C, __READ_WRITE , __exti_cfgr11_bits);
__IO_REG32_BIT(EXTI_CFGR12,             0x40024030, __READ_WRITE , __exti_cfgr12_bits);
__IO_REG32_BIT(EXTI_CFGR13,             0x40024034, __READ_WRITE , __exti_cfgr13_bits);
__IO_REG32_BIT(EXTI_CFGR14,             0x40024038, __READ_WRITE , __exti_cfgr14_bits);
__IO_REG32_BIT(EXTI_CFGR15,             0x4002403C, __READ_WRITE , __exti_cfgr15_bits);
__IO_REG32_BIT(EXTI_CR,                 0x40024040, __READ_WRITE , __exti_cr_bits);
__IO_REG32_BIT(EXTI_EDGEFLGR,           0x40024044, __READ_WRITE , __exti_edgeflgr_bits);
__IO_REG32_BIT(EXTI_EDGESR,             0x40024048, __READ_WRITE , __exti_edgesr_bits);
__IO_REG32_BIT(EXTI_SSCR,               0x4002404C, __READ_WRITE , __exti_sscr_bits);
__IO_REG32_BIT(EXTI_WAKUPCR,            0x40024050, __READ_WRITE , __exti_wakupcr_bits);
__IO_REG32_BIT(EXTI_WAKUPPOLR,          0x40024054, __READ_WRITE , __exti_wakuppolr_bits);
__IO_REG32_BIT(EXTI_WAKUPFLG,           0x40024058, __READ_WRITE , __exti_wakupflg_bits);
__IO_REG32_BIT(ADC_RST,                 0x40010004, __READ_WRITE , __adc_rst_bits);
__IO_REG32_BIT(ADC_CONV,                0x40010008, __READ_WRITE , __adc_conv_bits);
__IO_REG32_BIT(ADC_LST0,                0x40010010, __READ_WRITE , __adc_lst0_bits);
__IO_REG32_BIT(ADC_LST1,                0x40010014, __READ_WRITE , __adc_lst1_bits);
__IO_REG32_BIT(ADC_OFR0,                0x40010030, __READ_WRITE , __adc_ofr0_bits);
__IO_REG32_BIT(ADC_OFR1,                0x40010034, __READ_WRITE , __adc_ofr1_bits);
__IO_REG32_BIT(ADC_OFR2,                0x40010038, __READ_WRITE , __adc_ofr2_bits);
__IO_REG32_BIT(ADC_OFR3,                0x4001003C, __READ_WRITE , __adc_ofr3_bits);
__IO_REG32_BIT(ADC_OFR4,                0x40010040, __READ_WRITE , __adc_ofr4_bits);
__IO_REG32_BIT(ADC_OFR5,                0x40010044, __READ_WRITE , __adc_ofr5_bits);
__IO_REG32_BIT(ADC_OFR6,                0x40010048, __READ_WRITE , __adc_ofr6_bits);
__IO_REG32_BIT(ADC_OFR7,                0x4001004C, __READ_WRITE , __adc_ofr7_bits);
__IO_REG32_BIT(ADC_STR0,                0x40010070, __READ_WRITE , __adc_str0_bits);
__IO_REG32_BIT(ADC_STR1,                0x40010074, __READ_WRITE , __adc_str1_bits);
__IO_REG32_BIT(ADC_STR2,                0x40010078, __READ_WRITE , __adc_str2_bits);
__IO_REG32_BIT(ADC_STR3,                0x4001007C, __READ_WRITE , __adc_str3_bits);
__IO_REG32_BIT(ADC_STR4,                0x40010080, __READ_WRITE , __adc_str4_bits);
__IO_REG32_BIT(ADC_STR5,                0x40010084, __READ_WRITE , __adc_str5_bits);
__IO_REG32_BIT(ADC_STR6,                0x40010088, __READ_WRITE , __adc_str6_bits);
__IO_REG32_BIT(ADC_STR7,                0x4001008C, __READ_WRITE , __adc_str7_bits);
__IO_REG32_BIT(ADC_DR0,                 0x400100B0, __READ_WRITE , __adc_dr0_bits);
__IO_REG32_BIT(ADC_DR1,                 0x400100B4, __READ_WRITE , __adc_dr1_bits);
__IO_REG32_BIT(ADC_DR2,                 0x400100B8, __READ_WRITE , __adc_dr2_bits);
__IO_REG32_BIT(ADC_DR3,                 0x400100BC, __READ_WRITE , __adc_dr3_bits);
__IO_REG32_BIT(ADC_DR4,                 0x400100C0, __READ_WRITE , __adc_dr4_bits);
__IO_REG32_BIT(ADC_DR5,                 0x400100C4, __READ_WRITE , __adc_dr5_bits);
__IO_REG32_BIT(ADC_DR6,                 0x400100C8, __READ_WRITE , __adc_dr6_bits);
__IO_REG32_BIT(ADC_DR7,                 0x400100CC, __READ_WRITE , __adc_dr7_bits);
__IO_REG32_BIT(ADC_TCR,                 0x40010100, __READ_WRITE , __adc_tcr_bits);
__IO_REG32_BIT(ADC_TSR,                 0x40010104, __READ_WRITE , __adc_tsr_bits);
__IO_REG32_BIT(ADC_WCR,                 0x40010120, __READ_WRITE , __adc_wcr_bits);
__IO_REG32_BIT(ADC_LTR,                 0x40010124, __READ_WRITE , __adc_ltr_bits);
__IO_REG32_BIT(ADC_UTR,                 0x40010128, __READ_WRITE , __adc_utr_bits);
__IO_REG32_BIT(ADC_IMR,                 0x40010130, __READ_WRITE , __adc_imr_bits);
__IO_REG32_BIT(ADC_IRAW,                0x40010134, __READ_WRITE , __adc_iraw_bits);
__IO_REG32_BIT(ADC_IMASK,               0x40010138, __READ_WRITE , __adc_imask_bits);
__IO_REG32_BIT(ADC_ICLR,                0x4001013C, __READ_WRITE , __adc_iclr_bits);
__IO_REG32_BIT(ADC_DMAR,                0x40010140, __READ_WRITE , __adc_dmar_bits);
__IO_REG32_BIT(OPACR0,                  0x40018000, __READ_WRITE , __opacr0_bits);
__IO_REG32_BIT(OFVCR0,                  0x40018004, __READ_WRITE , __ofvcr0_bits);
__IO_REG32_BIT(CMPIER0,                 0x40018008, __READ_WRITE , __cmpier0_bits);
__IO_REG32_BIT(CMPRSR0,                 0x4001800C, __READ_WRITE , __cmprsr0_bits);
__IO_REG32_BIT(CMPISR0,                 0x40018010, __READ_WRITE , __cmpisr0_bits);
__IO_REG32_BIT(CMPICLR0,                0x40018014, __READ_WRITE , __cmpiclr0_bits);
__IO_REG32_BIT(OPACR1,                  0x40018100, __READ_WRITE , __opacr1_bits);
__IO_REG32_BIT(OFVCR1,                  0x40018104, __READ_WRITE , __ofvcr1_bits);
__IO_REG32_BIT(CMPIER1,                 0x40018108, __READ_WRITE , __cmpier1_bits);
__IO_REG32_BIT(CMPRSR1,                 0x4001810C, __READ_WRITE , __cmprsr1_bits);
__IO_REG32_BIT(CMPISR1,                 0x40018110, __READ_WRITE , __cmpisr1_bits);
__IO_REG32_BIT(CMPICLR1,                0x40018114, __READ_WRITE , __cmpiclr1_bits);
__IO_REG32_BIT(MCTM_CNTCFR,             0x4002C000, __READ_WRITE , __mctm_cntcfr_bits);
__IO_REG32_BIT(MCTM_MDCFR,              0x4002C004, __READ_WRITE , __mctm_mdcfr_bits);
__IO_REG32_BIT(MCTM_TRCFR,              0x4002C008, __READ_WRITE , __mctm_trcfr_bits);
__IO_REG32_BIT(MCTM_CTR,                0x4002C010, __READ_WRITE , __mctm_ctr_bits);
__IO_REG32_BIT(MCTM_CH0ICFR,            0x4002C020, __READ_WRITE , __mctm_ch0icfr_bits);
__IO_REG32_BIT(MCTM_CH1ICFR,            0x4002C024, __READ_WRITE , __mctm_ch1icfr_bits);
__IO_REG32_BIT(MCTM_CH2ICFR,            0x4002C028, __READ_WRITE , __mctm_ch2icfr_bits);
__IO_REG32_BIT(MCTM_CH3ICFR,            0x4002C02C, __READ_WRITE , __mctm_ch3icfr_bits);
__IO_REG32_BIT(MCTM_CH0OCFR,            0x4002C040, __READ_WRITE , __mctm_ch0ocfr_bits);
__IO_REG32_BIT(MCTM_CH1OCFR,            0x4002C044, __READ_WRITE , __mctm_ch1ocfr_bits);
__IO_REG32_BIT(MCTM_CH2OCFR,            0x4002C048, __READ_WRITE , __mctm_ch2ocfr_bits);
__IO_REG32_BIT(MCTM_CH3OCFR,            0x4002C04C, __READ_WRITE , __mctm_ch3ocfr_bits);
__IO_REG32_BIT(MCTM_CHCTR,              0x4002C050, __READ_WRITE , __mctm_chctr_bits);
__IO_REG32_BIT(MCTM_CHPOLR,             0x4002C054, __READ_WRITE , __mctm_chpolr_bits);
__IO_REG32_BIT(MCTM_CHBRKCFR,           0x4002C06C, __READ_WRITE , __mctm_chbrkcfr_bits);
__IO_REG32_BIT(MCTM_CHBRKCTR,           0x4002C070, __READ_WRITE , __mctm_chbrkctr_bits);
__IO_REG32_BIT(MCTM_DICTR,              0x4002C074, __READ_WRITE , __mctm_dictr_bits);
__IO_REG32_BIT(MCTM_EVGR,               0x4002C078, __READ_WRITE , __mctm_evgr_bits);
__IO_REG32_BIT(MCTM_INTSR,              0x4002C07C, __READ_WRITE , __mctm_intsr_bits);
__IO_REG32_BIT(MCTM_CNTR,               0x4002C080, __READ_WRITE , __mctm_cntr_bits);
__IO_REG32_BIT(MCTM_PSCR,               0x4002C084, __READ_WRITE , __mctm_pscr_bits);
__IO_REG32_BIT(MCTM_CRR,                0x4002C088, __READ_WRITE , __mctm_crr_bits);
__IO_REG32_BIT(MCTM_REPR,               0x4002C08C, __READ_WRITE , __mctm_repr_bits);
__IO_REG32_BIT(MCTM_CH0CCR,             0x4002C090, __READ_WRITE , __mctm_ch0ccr_bits);
__IO_REG32_BIT(MCTM_CH1CCR,             0x4002C094, __READ_WRITE , __mctm_ch1ccr_bits);
__IO_REG32_BIT(MCTM_CH2CCR,             0x4002C098, __READ_WRITE , __mctm_ch2ccr_bits);
__IO_REG32_BIT(MCTM_CH3CCR,             0x4002C09C, __READ_WRITE , __mctm_ch3ccr_bits);
__IO_REG32_BIT(GPTM0_CNTCFR,            0x4006E000, __READ_WRITE , __gptm0_cntcfr_bits);
__IO_REG32_BIT(GPTM0_MDCFR,             0x4006E004, __READ_WRITE , __gptm0_mdcfr_bits);
__IO_REG32_BIT(GPTM0_TRCFR,             0x4006E008, __READ_WRITE , __gptm0_trcfr_bits);
__IO_REG32_BIT(GPTM0_CTR,               0x4006E010, __READ_WRITE , __gptm0_ctr_bits);
__IO_REG32_BIT(GPTM0_CH0ICFR,           0x4006E020, __READ_WRITE , __gptm0_ch0icfr_bits);
__IO_REG32_BIT(GPTM0_CH1ICFR,           0x4006E024, __READ_WRITE , __gptm0_ch1icfr_bits);
__IO_REG32_BIT(GPTM0_CH2ICFR,           0x4006E028, __READ_WRITE , __gptm0_ch2icfr_bits);
__IO_REG32_BIT(GPTM0_CH3ICFR,           0x4006E02C, __READ_WRITE , __gptm0_ch3icfr_bits);
__IO_REG32_BIT(GPTM0_CH0OCFR,           0x4006E040, __READ_WRITE , __gptm0_ch0ocfr_bits);
__IO_REG32_BIT(GPTM0_CH1OCFR,           0x4006E044, __READ_WRITE , __gptm0_ch1ocfr_bits);
__IO_REG32_BIT(GPTM0_CH2OCFR,           0x4006E048, __READ_WRITE , __gptm0_ch2ocfr_bits);
__IO_REG32_BIT(GPTM0_CH3OCFR,           0x4006E04C, __READ_WRITE , __gptm0_ch3ocfr_bits);
__IO_REG32_BIT(GPTM0_CHCTR,             0x4006E050, __READ_WRITE , __gptm0_chctr_bits);
__IO_REG32_BIT(GPTM0_CHPOLR,            0x4006E054, __READ_WRITE , __gptm0_chpolr_bits);
__IO_REG32_BIT(GPTM0_DICTR,             0x4006E074, __READ_WRITE , __gptm0_dictr_bits);
__IO_REG32_BIT(GPTM0_EVGR,              0x4006E078, __READ_WRITE , __gptm0_evgr_bits);
__IO_REG32_BIT(GPTM0_INTSR,             0x4006E07C, __READ_WRITE , __gptm0_intsr_bits);
__IO_REG32_BIT(GPTM0_CNTR,              0x4006E080, __READ_WRITE , __gptm0_cntr_bits);
__IO_REG32_BIT(GPTM0_PSCR,              0x4006E084, __READ_WRITE , __gptm0_pscr_bits);
__IO_REG32_BIT(GPTM0_CRR,               0x4006E088, __READ_WRITE , __gptm0_crr_bits);
__IO_REG32_BIT(GPTM0_CH0CCR,            0x4006E090, __READ_WRITE , __gptm0_ch0ccr_bits);
__IO_REG32_BIT(GPTM0_CH1CCR,            0x4006E094, __READ_WRITE , __gptm0_ch1ccr_bits);
__IO_REG32_BIT(GPTM0_CH2CCR,            0x4006E098, __READ_WRITE , __gptm0_ch2ccr_bits);
__IO_REG32_BIT(GPTM0_CH3CCR,            0x4006E09C, __READ_WRITE , __gptm0_ch3ccr_bits);
__IO_REG32_BIT(GPTM1_CNTCFR,            0x4006F000, __READ_WRITE , __gptm1_cntcfr_bits);
__IO_REG32_BIT(GPTM1_MDCFR,             0x4006F004, __READ_WRITE , __gptm1_mdcfr_bits);
__IO_REG32_BIT(GPTM1_TRCFR,             0x4006F008, __READ_WRITE , __gptm1_trcfr_bits);
__IO_REG32_BIT(GPTM1_CTR,               0x4006F010, __READ_WRITE , __gptm1_ctr_bits);
__IO_REG32_BIT(GPTM1_CH0ICFR,           0x4006F020, __READ_WRITE , __gptm1_ch0icfr_bits);
__IO_REG32_BIT(GPTM1_CH1ICFR,           0x4006F024, __READ_WRITE , __gptm1_ch1icfr_bits);
__IO_REG32_BIT(GPTM1_CH2ICFR,           0x4006F028, __READ_WRITE , __gptm1_ch2icfr_bits);
__IO_REG32_BIT(GPTM1_CH3ICFR,           0x4006F02C, __READ_WRITE , __gptm1_ch3icfr_bits);
__IO_REG32_BIT(GPTM1_CH0OCFR,           0x4006F040, __READ_WRITE , __gptm1_ch0ocfr_bits);
__IO_REG32_BIT(GPTM1_CH1OCFR,           0x4006F044, __READ_WRITE , __gptm1_ch1ocfr_bits);
__IO_REG32_BIT(GPTM1_CH2OCFR,           0x4006F048, __READ_WRITE , __gptm1_ch2ocfr_bits);
__IO_REG32_BIT(GPTM1_CH3OCFR,           0x4006F04C, __READ_WRITE , __gptm1_ch3ocfr_bits);
__IO_REG32_BIT(GPTM1_CHCTR,             0x4006F050, __READ_WRITE , __gptm1_chctr_bits);
__IO_REG32_BIT(GPTM1_CHPOLR,            0x4006F054, __READ_WRITE , __gptm1_chpolr_bits);
__IO_REG32_BIT(GPTM1_DICTR,             0x4006F074, __READ_WRITE , __gptm1_dictr_bits);
__IO_REG32_BIT(GPTM1_EVGR,              0x4006F078, __READ_WRITE , __gptm1_evgr_bits);
__IO_REG32_BIT(GPTM1_INTSR,             0x4006F07C, __READ_WRITE , __gptm1_intsr_bits);
__IO_REG32_BIT(GPTM1_CNTR,              0x4006F080, __READ_WRITE , __gptm1_cntr_bits);
__IO_REG32_BIT(GPTM1_PSCR,              0x4006F084, __READ_WRITE , __gptm1_pscr_bits);
__IO_REG32_BIT(GPTM1_CRR,               0x4006F088, __READ_WRITE , __gptm1_crr_bits);
__IO_REG32_BIT(GPTM1_CH0CCR,            0x4006F090, __READ_WRITE , __gptm1_ch0ccr_bits);
__IO_REG32_BIT(GPTM1_CH1CCR,            0x4006F094, __READ_WRITE , __gptm1_ch1ccr_bits);
__IO_REG32_BIT(GPTM1_CH2CCR,            0x4006F098, __READ_WRITE , __gptm1_ch2ccr_bits);
__IO_REG32_BIT(GPTM1_CH3CCR,            0x4006F09C, __READ_WRITE , __gptm1_ch3ccr_bits);
__IO_REG32_BIT(BFTM0_CR,                0x40076000, __READ_WRITE , __bftm0_cr_bits);
__IO_REG32_BIT(BFTM0_SR,                0x40076004, __READ_WRITE , __bftm0_sr_bits);
__IO_REG32_BIT(BFTM0_CNTR,              0x40076008, __READ_WRITE , __bftm0_cntr_bits);
__IO_REG32_BIT(BFTM0_CMPR,              0x4007600C, __READ_WRITE , __bftm0_cmpr_bits);
__IO_REG32_BIT(BFTM1_CR,                0x40077000, __READ_WRITE , __bftm1_cr_bits);
__IO_REG32_BIT(BFTM1_SR,                0x40077004, __READ_WRITE , __bftm1_sr_bits);
__IO_REG32_BIT(BFTM1_CNTR,              0x40077008, __READ_WRITE , __bftm1_cntr_bits);
__IO_REG32_BIT(BFTM1_CMPR,              0x4007700C, __READ_WRITE , __bftm1_cmpr_bits);
__IO_REG32_BIT(RTC_CNT,                 0x4006A000, __READ_WRITE , __rtc_cnt_bits);
__IO_REG32_BIT(RTC_CMP,                 0x4006A004, __READ_WRITE , __rtc_cmp_bits);
__IO_REG32_BIT(RTC_CR,                  0x4006A008, __READ_WRITE , __rtc_cr_bits);
__IO_REG32_BIT(RTC_SR,                  0x4006A00C, __READ_WRITE , __rtc_sr_bits);
__IO_REG32_BIT(RTC_IWEN,                0x4006A010, __READ_WRITE , __rtc_iwen_bits);
__IO_REG32_BIT(WDT_CR,                  0x40068000, __READ_WRITE , __wdt_cr_bits);
__IO_REG32_BIT(WDT_MR0,                 0x40068004, __READ_WRITE , __wdt_mr0_bits);
__IO_REG32_BIT(WDT_MR1,                 0x40068008, __READ_WRITE , __wdt_mr1_bits);
__IO_REG32_BIT(WDT_SR,                  0x4006800C, __READ_WRITE , __wdt_sr_bits);
__IO_REG32_BIT(WDT_PR,                  0x40068010, __READ_WRITE , __wdt_pr_bits);
__IO_REG32_BIT(I2C0_CR,                 0x40048000, __READ_WRITE , __i2c0_cr_bits);
__IO_REG32_BIT(I2C0_IER,                0x40048004, __READ_WRITE , __i2c0_ier_bits);
__IO_REG32_BIT(I2C0_ADDR,               0x40048008, __READ_WRITE , __i2c0_addr_bits);
__IO_REG32_BIT(I2C0_SR,                 0x4004800C, __READ_WRITE , __i2c0_sr_bits);
__IO_REG32_BIT(I2C0_SHPGR,              0x40048010, __READ_WRITE , __i2c0_shpgr_bits);
__IO_REG32_BIT(I2C0_SLPGR,              0x40048014, __READ_WRITE , __i2c0_slpgr_bits);
__IO_REG32_BIT(I2C0_DR,                 0x40048018, __READ_WRITE , __i2c0_dr_bits);
__IO_REG32_BIT(I2C0_TAR,                0x4004801C, __READ_WRITE , __i2c0_tar_bits);
__IO_REG32_BIT(I2C0_ADDMR,              0x40048020, __READ_WRITE , __i2c0_addmr_bits);
__IO_REG32_BIT(I2C0_ADDSR,              0x40048024, __READ_WRITE , __i2c0_addsr_bits);
__IO_REG32_BIT(I2C0_TOUT,               0x40048028, __READ_WRITE , __i2c0_tout_bits);
__IO_REG32_BIT(I2C1_CR,                 0x40049000, __READ_WRITE , __i2c1_cr_bits);
__IO_REG32_BIT(I2C1_IER,                0x40049004, __READ_WRITE , __i2c1_ier_bits);
__IO_REG32_BIT(I2C1_ADDR,               0x40049008, __READ_WRITE , __i2c1_addr_bits);
__IO_REG32_BIT(I2C1_SR,                 0x4004900C, __READ_WRITE , __i2c1_sr_bits);
__IO_REG32_BIT(I2C1_SHPGR,              0x40049010, __READ_WRITE , __i2c1_shpgr_bits);
__IO_REG32_BIT(I2C1_SLPGR,              0x40049014, __READ_WRITE , __i2c1_slpgr_bits);
__IO_REG32_BIT(I2C1_DR,                 0x40049018, __READ_WRITE , __i2c1_dr_bits);
__IO_REG32_BIT(I2C1_TAR,                0x4004901C, __READ_WRITE , __i2c1_tar_bits);
__IO_REG32_BIT(I2C1_ADDMR,              0x40049020, __READ_WRITE , __i2c1_addmr_bits);
__IO_REG32_BIT(I2C1_ADDSR,              0x40049024, __READ_WRITE , __i2c1_addsr_bits);
__IO_REG32_BIT(I2C1_TOUT,               0x40049028, __READ_WRITE , __i2c1_tout_bits);
__IO_REG32_BIT(SPI0_CR0,                0x40004000, __READ_WRITE , __spi0_cr0_bits);
__IO_REG32_BIT(SPI0_CR1,                0x40004004, __READ_WRITE , __spi0_cr1_bits);
__IO_REG32_BIT(SPI0_IER,                0x40004008, __READ_WRITE , __spi0_ier_bits);
__IO_REG32_BIT(SPI0_CPR,                0x4000400C, __READ_WRITE , __spi0_cpr_bits);
__IO_REG32_BIT(SPI0_DR,                 0x40004010, __READ_WRITE , __spi0_dr_bits);
__IO_REG32_BIT(SPI0_SR,                 0x40004014, __READ_WRITE , __spi0_sr_bits);
__IO_REG32_BIT(SPI0_FCR,                0x40004018, __READ_WRITE , __spi0_fcr_bits);
__IO_REG32_BIT(SPI0_FSR,                0x4000401C, __READ_WRITE , __spi0_fsr_bits);
__IO_REG32_BIT(SPI0_FTOCR,              0x40004020, __READ_WRITE , __spi0_ftocr_bits);
__IO_REG32_BIT(SPI1_CR0,                0x40044000, __READ_WRITE , __spi1_cr0_bits);
__IO_REG32_BIT(SPI1_CR1,                0x40044004, __READ_WRITE , __spi1_cr1_bits);
__IO_REG32_BIT(SPI1_IER,                0x40044008, __READ_WRITE , __spi1_ier_bits);
__IO_REG32_BIT(SPI1_CPR,                0x4004400C, __READ_WRITE , __spi1_cpr_bits);
__IO_REG32_BIT(SPI1_DR,                 0x40044010, __READ_WRITE , __spi1_dr_bits);
__IO_REG32_BIT(SPI1_SR,                 0x40044014, __READ_WRITE , __spi1_sr_bits);
__IO_REG32_BIT(SPI1_FCR,                0x40044018, __READ_WRITE , __spi1_fcr_bits);
__IO_REG32_BIT(SPI1_FSR,                0x4004401C, __READ_WRITE , __spi1_fsr_bits);
__IO_REG32_BIT(SPI1_FTOCR,              0x40044020, __READ_WRITE , __spi1_ftocr_bits);
__IO_REG32_BIT(USART0_RBR,              0x40000000, __READ_WRITE , __usart0_rbr_bits);
#define USART0_TBR     USART0_RBR
__IO_REG32_BIT(USART0_IER,              0x40000004, __READ_WRITE , __usart0_ier_bits);
__IO_REG32_BIT(USART0_IIR,              0x40000008, __READ_WRITE , __usart0_iir_bits);
__IO_REG32_BIT(USART0_FCR,              0x4000000C, __READ_WRITE , __usart0_fcr_bits);
__IO_REG32_BIT(USART0_LCR,              0x40000010, __READ_WRITE , __usart0_lcr_bits);
__IO_REG32_BIT(USART0_MODCR,            0x40000014, __READ_WRITE , __usart0_modcr_bits);
__IO_REG32_BIT(USART0_LSR,              0x40000018, __READ_WRITE , __usart0_lsr_bits);
__IO_REG32_BIT(USART0_MODSR,            0x4000001C, __READ_WRITE , __usart0_modsr_bits);
__IO_REG32_BIT(USART0_TPR,              0x40000020, __READ_WRITE , __usart0_tpr_bits);
__IO_REG32_BIT(USART0_MDR,              0x40000024, __READ_WRITE , __usart0_mdr_bits);
__IO_REG32_BIT(USART0_IrDACR,           0x40000028, __READ_WRITE , __usart0_irdacr_bits);
__IO_REG32_BIT(USART0_RS485CR,          0x4000002C, __READ_WRITE , __usart0_rs485cr_bits);
__IO_REG32_BIT(USART0_SYNCR,            0x40000030, __READ_WRITE , __usart0_syncr_bits);
__IO_REG32_BIT(USART0_FSR,              0x40000034, __READ_WRITE , __usart0_fsr_bits);
__IO_REG32_BIT(USART0_DLR,              0x40000038, __READ_WRITE , __usart0_dlr_bits);
__IO_REG32_BIT(USART0_DEGTSTR,          0x40000040, __READ_WRITE , __usart0_degtstr_bits);
__IO_REG32_BIT(USART1_RBR,              0x40040000, __READ_WRITE , __usart1_rbr_bits);
#define USART1_TBR     USART1_RBR
__IO_REG32_BIT(USART1_IER,              0x40040004, __READ_WRITE , __usart1_ier_bits);
__IO_REG32_BIT(USART1_IIR,              0x40040008, __READ_WRITE , __usart1_iir_bits);
__IO_REG32_BIT(USART1_FCR,              0x4004000C, __READ_WRITE , __usart1_fcr_bits);
__IO_REG32_BIT(USART1_LCR,              0x40040010, __READ_WRITE , __usart1_lcr_bits);
__IO_REG32_BIT(USART1_MODCR,            0x40040014, __READ_WRITE , __usart1_modcr_bits);
__IO_REG32_BIT(USART1_LSR,              0x40040018, __READ_WRITE , __usart1_lsr_bits);
__IO_REG32_BIT(USART1_MODSR,            0x4004001C, __READ_WRITE , __usart1_modsr_bits);
__IO_REG32_BIT(USART1_TPR,              0x40040020, __READ_WRITE , __usart1_tpr_bits);
__IO_REG32_BIT(USART1_MDR,              0x40040024, __READ_WRITE , __usart1_mdr_bits);
__IO_REG32_BIT(USART1_IrDACR,           0x40040028, __READ_WRITE , __usart1_irdacr_bits);
__IO_REG32_BIT(USART1_RS485CR,          0x4004002C, __READ_WRITE , __usart1_rs485cr_bits);
__IO_REG32_BIT(USART1_SYNCR,            0x40040030, __READ_WRITE , __usart1_syncr_bits);
__IO_REG32_BIT(USART1_FSR,              0x40040034, __READ_WRITE , __usart1_fsr_bits);
__IO_REG32_BIT(USART1_DLR,              0x40040038, __READ_WRITE , __usart1_dlr_bits);
__IO_REG32_BIT(USART1_DEGTSTR,          0x40040040, __READ_WRITE , __usart1_degtstr_bits);
__IO_REG32_BIT(SCI_CR,                  0x40043000, __READ_WRITE , __sci_cr_bits);
__IO_REG32_BIT(SCI_SR,                  0x40043004, __READ_WRITE , __sci_sr_bits);
__IO_REG32_BIT(SCI_CCR,                 0x40043008, __READ_WRITE , __sci_ccr_bits);
__IO_REG32_BIT(SCI_ETU,                 0x4004300C, __READ_WRITE , __sci_etu_bits);
__IO_REG32_BIT(SCI_GT,                  0x40043010, __READ_WRITE , __sci_gt_bits);
__IO_REG32_BIT(SCI_WT,                  0x40043014, __READ_WRITE , __sci_wt_bits);
__IO_REG32_BIT(SCI_IER,                 0x40043018, __READ_WRITE , __sci_ier_bits);
__IO_REG32_BIT(SCI_IPR,                 0x4004301C, __READ_WRITE , __sci_ipr_bits);
__IO_REG32_BIT(SCI_TXB,                 0x40043020, __READ_WRITE , __sci_txb_bits);
__IO_REG32_BIT(SCI_RXB,                 0x40043024, __READ_WRITE , __sci_rxb_bits);
__IO_REG32_BIT(SCI_PSC,                 0x40043028, __READ_WRITE , __sci_psc_bits);
__IO_REG32_BIT(USB_CSR,                 0x4004E000, __READ_WRITE , __usb_csr_bits);
__IO_REG32_BIT(USB_IER,                 0x4004E004, __READ_WRITE , __usb_ier_bits);
__IO_REG32_BIT(USB_ISR,                 0x4004E008, __READ_WRITE , __usb_isr_bits);
__IO_REG32_BIT(USB_FCR,                 0x4004E00C, __READ_WRITE , __usb_fcr_bits);
__IO_REG32_BIT(USB_DEVAR,               0x4004E010, __READ_WRITE , __usb_devar_bits);
__IO_REG32_BIT(USB_EP0CSR,              0x4004E014, __READ_WRITE , __usb_ep0csr_bits);
__IO_REG32_BIT(USB_EP0IER,              0x4004E018, __READ_WRITE , __usb_ep0ier_bits);
__IO_REG32_BIT(USB_EP0ISR,              0x4004E01C, __READ_WRITE , __usb_ep0isr_bits);
__IO_REG32_BIT(USB_EP0TCR,              0x4004E020, __READ_WRITE , __usb_ep0tcr_bits);
__IO_REG32_BIT(USB_EP0CFGR,             0x4004E024, __READ_WRITE , __usb_ep0cfgr_bits);
__IO_REG32_BIT(USB_EP1CSR,              0x4004E028, __READ_WRITE , __usb_ep1csr_bits);
__IO_REG32_BIT(USB_EP1IER,              0x4004E02C, __READ_WRITE , __usb_ep1ier_bits);
__IO_REG32_BIT(USB_EP1ISR,              0x4004E030, __READ_WRITE , __usb_ep1isr_bits);
__IO_REG32_BIT(USB_EP1TCR,              0x4004E034, __READ_WRITE , __usb_ep1tcr_bits);
__IO_REG32_BIT(USB_EP1CFGR,             0x4004E038, __READ_WRITE , __usb_ep1cfgr_bits);
__IO_REG32_BIT(USB_EP2CSR,              0x4004E03C, __READ_WRITE , __usb_ep2csr_bits);
__IO_REG32_BIT(USB_EP2IER,              0x4004E040, __READ_WRITE , __usb_ep2ier_bits);
__IO_REG32_BIT(USB_EP2ISR,              0x4004E044, __READ_WRITE , __usb_ep2isr_bits);
__IO_REG32_BIT(USB_EP2TCR,              0x4004E048, __READ_WRITE , __usb_ep2tcr_bits);
__IO_REG32_BIT(USB_EP2CFGR,             0x4004E04C, __READ_WRITE , __usb_ep2cfgr_bits);
__IO_REG32_BIT(USB_EP3CSR,              0x4004E050, __READ_WRITE , __usb_ep3csr_bits);
__IO_REG32_BIT(USB_EP3IER,              0x4004E054, __READ_WRITE , __usb_ep3ier_bits);
__IO_REG32_BIT(USB_EP3ISR,              0x4004E058, __READ_WRITE , __usb_ep3isr_bits);
__IO_REG32_BIT(USB_EP3TCR,              0x4004E05C, __READ_WRITE , __usb_ep3tcr_bits);
__IO_REG32_BIT(USB_EP3CFGR,             0x4004E060, __READ_WRITE , __usb_ep3cfgr_bits);
__IO_REG32_BIT(USB_EP4CSR,              0x4004E064, __READ_WRITE , __usb_ep4csr_bits);
__IO_REG32_BIT(USB_EP4IER,              0x4004E068, __READ_WRITE , __usb_ep4ier_bits);
__IO_REG32_BIT(USB_EP4ISR,              0x4004E06C, __READ_WRITE , __usb_ep4isr_bits);
__IO_REG32_BIT(USB_EP4TCR,              0x4004E070, __READ_WRITE , __usb_ep4tcr_bits);
__IO_REG32_BIT(USB_EP4CFGR,             0x4004E074, __READ_WRITE , __usb_ep4cfgr_bits);
__IO_REG32_BIT(USB_EP5CSR,              0x4004E078, __READ_WRITE , __usb_ep5csr_bits);
__IO_REG32_BIT(USB_EP5IER,              0x4004E07C, __READ_WRITE , __usb_ep5ier_bits);
__IO_REG32_BIT(USB_EP5ISR,              0x4004E080, __READ_WRITE , __usb_ep5isr_bits);
__IO_REG32_BIT(USB_EP5TCR,              0x4004E084, __READ_WRITE , __usb_ep5tcr_bits);
__IO_REG32_BIT(USB_EP5CFGR,             0x4004E088, __READ_WRITE , __usb_ep5cfgr_bits);
__IO_REG32_BIT(USB_EP6CSR,              0x4004E08C, __READ_WRITE , __usb_ep6csr_bits);
__IO_REG32_BIT(USB_EP6IER,              0x4004E090, __READ_WRITE , __usb_ep6ier_bits);
__IO_REG32_BIT(USB_EP6ISR,              0x4004E094, __READ_WRITE , __usb_ep6isr_bits);
__IO_REG32_BIT(USB_EP6TCR,              0x4004E098, __READ_WRITE , __usb_ep6tcr_bits);
__IO_REG32_BIT(USB_EP6CFGR,             0x4004E09C, __READ_WRITE , __usb_ep6cfgr_bits);
__IO_REG32_BIT(USB_EP7CSR,              0x4004E0A0, __READ_WRITE , __usb_ep7csr_bits);
__IO_REG32_BIT(USB_EP7IER,              0x4004E0A4, __READ_WRITE , __usb_ep7ier_bits);
__IO_REG32_BIT(USB_EP7ISR,              0x4004E0A8, __READ_WRITE , __usb_ep7isr_bits);
__IO_REG32_BIT(USB_EP7TCR,              0x4004E0AC, __READ_WRITE , __usb_ep7tcr_bits);
__IO_REG32_BIT(USB_EP7CFGR,             0x4004E0B0, __READ_WRITE , __usb_ep7cfgr_bits);
__IO_REG32_BIT(PDMA_CH0CR,              0x40090000, __READ_WRITE , __pdma_ch0cr_bits);
__IO_REG32_BIT(PDMA_CH0SADR,            0x40090004, __READ_WRITE , __pdma_ch0sadr_bits);
__IO_REG32_BIT(PDMA_CH0DADR,            0x40090008, __READ_WRITE , __pdma_ch0dadr_bits);
__IO_REG32_BIT(PDMA_CH0CADR,            0x4009000C, __READ_WRITE , __pdma_ch0cadr_bits);
__IO_REG32_BIT(PDMA_CH0TSR,             0x40090010, __READ_WRITE , __pdma_ch0tsr_bits);
__IO_REG32_BIT(PDMA_CH0CTSR,            0x40090014, __READ_WRITE , __pdma_ch0ctsr_bits);
__IO_REG32_BIT(PDMA_CH1CR,              0x40090018, __READ_WRITE , __pdma_ch1cr_bits);
__IO_REG32_BIT(PDMA_CH1SADR,            0x4009001C, __READ_WRITE , __pdma_ch1sadr_bits);
__IO_REG32_BIT(PDMA_CH1DADR,            0x40090020, __READ_WRITE , __pdma_ch1dadr_bits);
__IO_REG32_BIT(PDMA_CH1CADR,            0x40090024, __READ_WRITE , __pdma_ch1cadr_bits);
__IO_REG32_BIT(PDMA_CH1TSR,             0x40090028, __READ_WRITE , __pdma_ch1tsr_bits);
__IO_REG32_BIT(PDMA_CH1CTSR,            0x4009002C, __READ_WRITE , __pdma_ch1ctsr_bits);
__IO_REG32_BIT(PDMA_CH2CR,              0x40090030, __READ_WRITE , __pdma_ch2cr_bits);
__IO_REG32_BIT(PDMA_CH2SADR,            0x40090034, __READ_WRITE , __pdma_ch2sadr_bits);
__IO_REG32_BIT(PDMA_CH2DADR,            0x40090038, __READ_WRITE , __pdma_ch2dadr_bits);
__IO_REG32_BIT(PDMA_CH2CADR,            0x4009003C, __READ_WRITE , __pdma_ch2cadr_bits);
__IO_REG32_BIT(PDMA_CH2TSR,             0x40090040, __READ_WRITE , __pdma_ch2tsr_bits);
__IO_REG32_BIT(PDMA_CH2CTSR,            0x40090044, __READ_WRITE , __pdma_ch2ctsr_bits);
__IO_REG32_BIT(PDMA_CH3CR,              0x40090048, __READ_WRITE , __pdma_ch3cr_bits);
__IO_REG32_BIT(PDMA_CH3SADR,            0x4009004C, __READ_WRITE , __pdma_ch3sadr_bits);
__IO_REG32_BIT(PDMA_CH3DADR,            0x40090050, __READ_WRITE , __pdma_ch3dadr_bits);
__IO_REG32_BIT(PDMA_CH3CADR,            0x40090054, __READ_WRITE , __pdma_ch3cadr_bits);
__IO_REG32_BIT(PDMA_CH3TSR,             0x40090058, __READ_WRITE , __pdma_ch3tsr_bits);
__IO_REG32_BIT(PDMA_CH3CTSR,            0x4009005C, __READ_WRITE , __pdma_ch3ctsr_bits);
__IO_REG32_BIT(PDMA_CH4CR,              0x40090060, __READ_WRITE , __pdma_ch4cr_bits);
__IO_REG32_BIT(PDMA_CH4SADR,            0x40090064, __READ_WRITE , __pdma_ch4sadr_bits);
__IO_REG32_BIT(PDMA_CH4DADR,            0x40090068, __READ_WRITE , __pdma_ch4dadr_bits);
__IO_REG32_BIT(PDMA_CH4CADR,            0x4009006C, __READ_WRITE , __pdma_ch4cadr_bits);
__IO_REG32_BIT(PDMA_CH4TSR,             0x40090070, __READ_WRITE , __pdma_ch4tsr_bits);
__IO_REG32_BIT(PDMA_CH4CTSR,            0x40090074, __READ_WRITE , __pdma_ch4ctsr_bits);
__IO_REG32_BIT(PDMA_CH5CR,              0x40090078, __READ_WRITE , __pdma_ch5cr_bits);
__IO_REG32_BIT(PDMA_CH5SADR,            0x4009007C, __READ_WRITE , __pdma_ch5sadr_bits);
__IO_REG32_BIT(PDMA_CH5DADR,            0x40090080, __READ_WRITE , __pdma_ch5dadr_bits);
__IO_REG32_BIT(PDMA_CH5CADR,            0x40090084, __READ_WRITE , __pdma_ch5cadr_bits);
__IO_REG32_BIT(PDMA_CH5TSR,             0x40090088, __READ_WRITE , __pdma_ch5tsr_bits);
__IO_REG32_BIT(PDMA_CH5CTSR,            0x4009008C, __READ_WRITE , __pdma_ch5ctsr_bits);
__IO_REG32_BIT(PDMA_CH6CR,              0x40090090, __READ_WRITE , __pdma_ch6cr_bits);
__IO_REG32_BIT(PDMA_CH6SADR,            0x40090094, __READ_WRITE , __pdma_ch6sadr_bits);
__IO_REG32_BIT(PDMA_CH6DADR,            0x40090098, __READ_WRITE , __pdma_ch6dadr_bits);
__IO_REG32_BIT(PDMA_CH6CADR,            0x4009009C, __READ_WRITE , __pdma_ch6cadr_bits);
__IO_REG32_BIT(PDMA_CH6TSR,             0x400900A0, __READ_WRITE , __pdma_ch6tsr_bits);
__IO_REG32_BIT(PDMA_CH6CTSR,            0x400900A4, __READ_WRITE , __pdma_ch6ctsr_bits);
__IO_REG32_BIT(PDMA_CH7CR,              0x400900A8, __READ_WRITE , __pdma_ch7cr_bits);
__IO_REG32_BIT(PDMA_CH7SADR,            0x400900AC, __READ_WRITE , __pdma_ch7sadr_bits);
__IO_REG32_BIT(PDMA_CH7DADR,            0x400900B0, __READ_WRITE , __pdma_ch7dadr_bits);
__IO_REG32_BIT(PDMA_CH7CADR,            0x400900B4, __READ_WRITE , __pdma_ch7cadr_bits);
__IO_REG32_BIT(PDMA_CH7TSR,             0x400900B8, __READ_WRITE , __pdma_ch7tsr_bits);
__IO_REG32_BIT(PDMA_CH7CTSR,            0x400900BC, __READ_WRITE , __pdma_ch7ctsr_bits);
__IO_REG32_BIT(PDMA_CH8CR,              0x400900C0, __READ_WRITE , __pdma_ch8cr_bits);
__IO_REG32_BIT(PDMA_CH8SADR,            0x400900C4, __READ_WRITE , __pdma_ch8sadr_bits);
__IO_REG32_BIT(PDMA_CH8DADR,            0x400900C8, __READ_WRITE , __pdma_ch8dadr_bits);
__IO_REG32_BIT(PDMA_CH8CADR,            0x400900CC, __READ_WRITE , __pdma_ch8cadr_bits);
__IO_REG32_BIT(PDMA_CH8TSR,             0x400900D0, __READ_WRITE , __pdma_ch8tsr_bits);
__IO_REG32_BIT(PDMA_CH8CTSR,            0x400900D4, __READ_WRITE , __pdma_ch8ctsr_bits);
__IO_REG32_BIT(PDMA_CH9CR,              0x400900D8, __READ_WRITE , __pdma_ch9cr_bits);
__IO_REG32_BIT(PDMA_CH9SADR,            0x400900DC, __READ_WRITE , __pdma_ch9sadr_bits);
__IO_REG32_BIT(PDMA_CH9DADR,            0x400900E0, __READ_WRITE , __pdma_ch9dadr_bits);
__IO_REG32_BIT(PDMA_CH9CADR,            0x400900E4, __READ_WRITE , __pdma_ch9cadr_bits);
__IO_REG32_BIT(PDMA_CH9TSR,             0x400900E8, __READ_WRITE , __pdma_ch9tsr_bits);
__IO_REG32_BIT(PDMA_CH9CTSR,            0x400900EC, __READ_WRITE , __pdma_ch9ctsr_bits);
__IO_REG32_BIT(PDMA_CH10CR,             0x400900F0, __READ_WRITE , __pdma_ch10cr_bits);
__IO_REG32_BIT(PDMA_CH10SADR,           0x400900F4, __READ_WRITE , __pdma_ch10sadr_bits);
__IO_REG32_BIT(PDMA_CH10DADR,           0x400900F8, __READ_WRITE , __pdma_ch10dadr_bits);
__IO_REG32_BIT(PDMA_CH10CADR,           0x400900FC, __READ_WRITE , __pdma_ch10cadr_bits);
__IO_REG32_BIT(PDMA_CH10TSR,            0x40090100, __READ_WRITE , __pdma_ch10tsr_bits);
__IO_REG32_BIT(PDMA_CH10CTSR,           0x40090104, __READ_WRITE , __pdma_ch10ctsr_bits);
__IO_REG32_BIT(PDMA_CH11CR,             0x40090108, __READ_WRITE , __pdma_ch11cr_bits);
__IO_REG32_BIT(PDMA_CH11SADR,           0x4009010C, __READ_WRITE , __pdma_ch11sadr_bits);
__IO_REG32_BIT(PDMA_CH11DADR,           0x40090110, __READ_WRITE , __pdma_ch11dadr_bits);
__IO_REG32_BIT(PDMA_CH11CADR,           0x40090114, __READ_WRITE , __pdma_ch11cadr_bits);
__IO_REG32_BIT(PDMA_CH11TSR,            0x40090118, __READ_WRITE , __pdma_ch11tsr_bits);
__IO_REG32_BIT(PDMA_CH11CTSR,           0x4009011C, __READ_WRITE , __pdma_ch11ctsr_bits);
__IO_REG32_BIT(PDMA_ISR0,               0x40090120, __READ_WRITE , __pdma_isr0_bits);
__IO_REG32_BIT(PDMA_ISR1,               0x40090124, __READ_WRITE , __pdma_isr1_bits);
__IO_REG32_BIT(PDMA_ISCR0,              0x40090128, __READ_WRITE , __pdma_iscr0_bits);
__IO_REG32_BIT(PDMA_ISCR1,              0x4009012C, __READ_WRITE , __pdma_iscr1_bits);
__IO_REG32_BIT(PDMA_IER0,               0x40090130, __READ_WRITE , __pdma_ier0_bits);
__IO_REG32_BIT(PDMA_IER1,               0x40090134, __READ_WRITE , __pdma_ier1_bits);

/* Assembler-specific declarations  ****************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__                                           */

/***************************************************************************
**
**  ioHT32F17xx Interrupt Lines
**
***************************************************************************/
#define MAIN_STACK     0
#define RESETI         1
#define NMII           2
#define HFI            3
#define MMI            4
#define BFI            5
#define UFI            6
#define SVCI           11
#define DMI            12
#define PSI            14
#define STI            15
#define CKRDY          16
#define LVD            17
#define BOD            18
#define WDT            19
#define RTC            20
#define FMC            21
#define EVWUP          22
#define LPWUP          23
#define EXTI0          24
#define EXTI1          25
#define EXTI2          26
#define EXTI3          27
#define EXTI4          28
#define EXTI5          29
#define EXTI6          30
#define EXTI7          31
#define EXTI8          32
#define EXTI9          33
#define EXTI10         34
#define EXTI11         35
#define EXTI12         36
#define EXTI13         37
#define EXTI14         38
#define EXTI15         39
#define COMP           40
#define ADC            41
#define MCTM_BRK       43
#define MCTM_UP        44
#define MCTM_TR        45
#define MCTM_CC        46
#define GPTM0          51
#define GPTM1          52
#define BFTM0          57
#define BFTM1          58
#define I2C0           59
#define I2C1           60
#define SPI0           61
#define SPI1           62
#define USART0         63
#define USART1         64
#define SCI            67
#define USB            69
#define PDMA_CH0       71
#define PDMA_CH1       72
#define PDMA_CH2       73
#define PDMA_CH3       74
#define PDMA_CH4       75
#define PDMA_CH5       76
#define PDMA_CH6       77
#define PDMA_CH7       78
#define PDMA_CH8       79
#define PDMA_CH9       80
#define PDMA_CH10      81
#define PDMA_CH11      82

#endif /*__ioHT32F17xx_H__                                                 */
