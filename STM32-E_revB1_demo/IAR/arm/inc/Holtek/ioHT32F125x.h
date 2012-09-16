/***************************************************************************
**
**    This file defines the Special Function Registers for
**    HOLTEK ioHT32F125x
**
**    Used with ARM IAR C/C++ Compiler and Assembler
**
**    (c) Copyright IAR Systems 2010
**
**    $Revision: 52631 $
**
***************************************************************************/
#ifndef __ioHT32F125x_H__

#define __ioHT32F125x_H__

#if (((__TID__ >> 8) & 0x7F) != 0x4F)
#error This file should only be compiled by ARM IAR compiler and assembler
#endif


#include "io_macros.h"

/***************************************************************************
 ***************************************************************************
 **
 **   ioHT32F125x SPECIAL FUNCTION REGISTERS
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
    __REG32             : 16;
} __iser1_bits;


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
    __REG32             : 16;
} __icer1_bits;


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
    __REG32              : 16;
} __ispr1_bits;


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
    __REG32              : 16;
} __icpr1_bits;


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
    __REG32             : 16;
} __iabr1_bits;


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
    __REG32    WAIT : 3;
    __REG32         : 1;
    __REG32    PFBE : 1;
    __REG32         : 1;
    __REG32    IPSE : 1;
    __REG32         : 25;
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
    __REG32            : 11;
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
    __REG32             : 7;
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
    __REG32    FMCEN  : 1;
    __REG32           : 1;
    __REG32    SRAMEN : 1;
    __REG32           : 29;
} __ckcu_ahbccr_bits;


typedef struct {
    __REG32           : 16;
    __REG32    ADCDIV : 3;
    __REG32           : 13;
} __ckcu_apbcfgr_bits;


typedef struct {
    __REG32    I2CEN  : 1;
    __REG32           : 3;
    __REG32    SPIEN  : 1;
    __REG32           : 3;
    __REG32    UREN   : 1;
    __REG32           : 5;
    __REG32    AFIOEN : 1;
    __REG32    EXTIEN : 1;
    __REG32    PAEN   : 1;
    __REG32    PBEN   : 1;
    __REG32           : 14;
} __ckcu_apbccr0_bits;


typedef struct {
    __REG32            : 4;
    __REG32    WDTEN   : 1;
    __REG32            : 1;
    __REG32    RTCEN   : 1;
    __REG32            : 1;
    __REG32    GPTM0EN : 1;
    __REG32    GPTM1EN : 1;
    __REG32            : 12;
    __REG32    OPA0EN  : 1;
    __REG32    OPA1EN  : 1;
    __REG32    ADCEN   : 1;
    __REG32            : 7;
} __ckcu_apbccr1_bits;


typedef struct {
    __REG32          : 8;
    __REG32    PLLST : 1;
    __REG32          : 7;
    __REG32    HSEST : 2;
    __REG32          : 6;
    __REG32    HSIST : 3;
    __REG32          : 5;
} __ckcu_ckst_bits;


typedef struct {
    __REG32    BKISO : 1;
    __REG32          : 31;
} __ckcu_lpcr_bits;


typedef struct {
    __REG32    DBSLP   : 1;
    __REG32    DBDSLP1 : 1;
    __REG32    DBPD    : 1;
    __REG32    DBWDT   : 1;
    __REG32            : 2;
    __REG32    DBGPTM0 : 1;
    __REG32    DBGPTM1 : 1;
    __REG32    DBUSART : 1;
    __REG32            : 1;
    __REG32    DBSPI   : 1;
    __REG32            : 3;
    __REG32    DBDSLP2 : 1;
    __REG32            : 17;
} __ckcu_mcudbgcr_bits;


typedef struct {
    __REG32    SYSRSTF : 1;
    __REG32    EXTRSTF : 1;
    __REG32    WDTRSTF : 1;
    __REG32    PORSTF  : 1;
    __REG32            : 28;
} __rstcu_grsr_bits;


typedef struct {
    __REG32    I2CRST  : 1;
    __REG32            : 3;
    __REG32    SPIRST  : 1;
    __REG32            : 3;
    __REG32    URRST   : 1;
    __REG32            : 5;
    __REG32    AFIORST : 1;
    __REG32    EXTIRST : 1;
    __REG32    PARST   : 1;
    __REG32    PBRST   : 1;
    __REG32            : 14;
} __rstcu_apbprstr0_bits;


typedef struct {
    __REG32             : 4;
    __REG32    WDTRST   : 1;
    __REG32             : 3;
    __REG32    GPTM0RST : 1;
    __REG32    GPTM1RST : 1;
    __REG32             : 12;
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
    __REG32    DV0  : 1;
    __REG32    DV1  : 1;
    __REG32    DV2  : 1;
    __REG32    DV3  : 1;
    __REG32    DV4  : 1;
    __REG32    DV5  : 1;
    __REG32    DV6  : 1;
    __REG32    DV7  : 1;
    __REG32         : 24;
} __gpiob_drvr_bits;


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
    __REG32    GPTM   : 1;
    __REG32           : 29;
} __adc_tcr_bits;


typedef struct {
    __REG32    ADSC    : 1;
    __REG32            : 7;
    __REG32    ADEXTIS : 4;
    __REG32            : 4;
    __REG32    GPTMS   : 3;
    __REG32            : 5;
    __REG32    GPTME   : 3;
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
    __REG32    UEVDIS : 1;
    __REG32    UGDIS   : 1;
    __REG32            : 6;
    __REG32    CKDIV   : 2;
    __REG32            : 6;
    __REG32    CMSEL   : 2;
    __REG32            : 6;
    __REG32    DIR     : 1;
    __REG32            : 7;
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
    __REG32    TME  : 1;
    __REG32    CRBE : 1;
    __REG32         : 30;
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
    __REG32            : 21;
} __gptm0_ictr_bits;


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
    __REG32    TME  : 1;
    __REG32    CRBE : 1;
    __REG32         : 30;
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
    __REG32            : 21;
} __gptm1_ictr_bits;


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
    __REG32    AA    : 1;
    __REG32    STOP  : 1;
    __REG32    GCEN  : 1;
    __REG32    I2CEN : 1;
    __REG32          : 3;
    __REG32    ADRM  : 1;
    __REG32          : 24;
} __i2c_cr_bits;


typedef struct {
    __REG32    STAIE    : 1;
    __REG32    STOIE    : 1;
    __REG32    ADRSIE   : 1;
    __REG32    GCSIE    : 1;
    __REG32             : 4;
    __REG32    ARBLOSIE : 1;
    __REG32    RXNACKIE : 1;
    __REG32    BUSERRIE : 1;
    __REG32             : 5;
    __REG32    RXDNEIE  : 1;
    __REG32    TXDEIE   : 1;
    __REG32    RXBFIE   : 1;
    __REG32             : 13;
} __i2c_ier_bits;


typedef struct {
    __REG32    ADDR : 10;
    __REG32         : 22;
} __i2c_addr_bits;


typedef struct {
    __REG32    STA     : 1;
    __REG32    STO     : 1;
    __REG32    ADRS    : 1;
    __REG32    GCS     : 1;
    __REG32            : 4;
    __REG32    ARBLOS  : 1;
    __REG32    RXNACK  : 1;
    __REG32    BUSERR  : 1;
    __REG32            : 5;
    __REG32    RXDNE   : 1;
    __REG32    TXDE    : 1;
    __REG32    RXBF    : 1;
    __REG32    BUSBUSY : 1;
    __REG32    MASTER  : 1;
    __REG32    TXNRX   : 1;
    __REG32            : 10;
} __i2c_sr_bits;


typedef struct {
    __REG32    SHPG : 16;
    __REG32         : 16;
} __i2c_shpgr_bits;


typedef struct {
    __REG32    SLPG : 16;
    __REG32         : 16;
} __i2c_slpgr_bits;


typedef struct {
    __REG32    DATA : 8;
    __REG32         : 24;
} __i2c_dr_bits;


typedef struct {
    __REG32    TAR  : 10;
    __REG32    RWD  : 1;
    __REG32         : 21;
} __i2c_tar_bits;


typedef struct {
    __REG32    SPIEN  : 1;
    __REG32           : 2;
    __REG32    SELOEN : 1;
    __REG32    SSELC  : 1;
    __REG32           : 27;
} __spi_cr0_bits;


typedef struct {
    __REG32    DFL      : 4;
    __REG32             : 4;
    __REG32    FORMAT   : 3;
    __REG32    SELAP    : 1;
    __REG32    FIRSTBIT : 1;
    __REG32    SELM     : 1;
    __REG32    MODE     : 1;
    __REG32             : 17;
} __spi_cr1_bits;


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
} __spi_ier_bits;


typedef struct {
    __REG32    CP : 16;
    __REG32       : 16;
} __spi_cpr_bits;


typedef struct {
    __REG32    DR : 16;
    __REG32       : 16;
} __spi_dr_bits;


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
} __spi_sr_bits;


typedef struct {
    __REG32    TXFTLS : 4;
    __REG32    RXFTLS : 4;
    __REG32    TFPR   : 1;
    __REG32    RFPR   : 1;
    __REG32    FIFOEN : 1;
    __REG32           : 21;
} __spi_fcr_bits;


typedef struct {
    __REG32    TXFS : 4;
    __REG32    RXFS : 4;
    __REG32         : 24;
} __spi_fsr_bits;


typedef struct {
    __REG32    TOC : 32;
} __spi_ftocr_bits;


typedef struct {
    __REG32    RD : 9;
    __REG32       : 23;
} __usart_rbr_bits;


typedef struct {
    __REG32    TD : 9;
    __REG32       : 23;
} __usart_tbr_bits;


typedef struct {
    __REG32    RFTLI_RTOIE : 1;
    __REG32    TFTLIE      : 1;
    __REG32    RLSIE       : 1;
    __REG32    MODSIE      : 1;
    __REG32                : 28;
} __usart_ier_bits;


typedef struct {
    __REG32    NIP  : 1;
    __REG32    IID  : 3;
    __REG32         : 28;
} __usart_iir_bits;


typedef struct {
    __REG32    FME  : 1;
    __REG32    RFR  : 1;
    __REG32    TFR  : 1;
    __REG32         : 1;
    __REG32    TFTL : 2;
    __REG32    RFTL : 2;
    __REG32         : 24;
} __usart_fcr_bits;


typedef struct {
    __REG32    WLS  : 2;
    __REG32    NSB  : 1;
    __REG32    PBE  : 1;
    __REG32    EPE  : 1;
    __REG32    SPE  : 1;
    __REG32    BCB  : 1;
    __REG32         : 25;
} __usart_lcr_bits;


typedef struct {
    __REG32    DTR  : 1;
    __REG32    RTS  : 1;
    __REG32         : 30;
} __usart_modcr_bits;


typedef struct {
    __REG32    RFDR    : 1;
    __REG32    OEI     : 1;
    __REG32    PEI     : 1;
    __REG32    FEI     : 1;
    __REG32    BII     : 1;
    __REG32    TXFEMPT : 1;
    __REG32    TXEMPT  : 1;
    __REG32    ERRRX   : 1;
    __REG32            : 24;
} __usart_lsr_bits;


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
} __usart_modsr_bits;


typedef struct {
    __REG32    RTOIC : 7;
    __REG32    RTOIE : 1;
    __REG32    TG    : 8;
    __REG32          : 16;
} __usart_tpr_bits;


typedef struct {
    __REG32    MODE : 2;
    __REG32    TRSM : 1;
    __REG32         : 29;
} __usart_mdr_bits;


typedef struct {
    __REG32    IrDAEN  : 1;
    __REG32    IrDALP  : 1;
    __REG32    TXSEL   : 1;
    __REG32    LB      : 1;
    __REG32            : 4;
    __REG32    IrDAPSC : 8;
    __REG32            : 16;
} __usart_irdacr_bits;


typedef struct {
    __REG32    TXENP : 1;
    __REG32          : 31;
} __usart_rs485cr_bits;


typedef struct {
    __REG32    CLKEN : 1;
    __REG32          : 1;
    __REG32    CPS   : 1;
    __REG32    CPO   : 1;
    __REG32          : 28;
} __usart_syncr_bits;


typedef struct {
    __REG32    LBM  : 2;
    __REG32         : 30;
} __usart_degtstr_bits;


typedef struct {
    __REG32    BRD : 16;
    __REG32        : 16;
} __usart_dlr_bits;


#endif /*__IAR_SYSTEMS_ICC__                                              */

/* Declarations common to compiler and assembler  *************************/

__IO_REG32_BIT(ICTR,                    0xE000E004, __READ_WRITE , __ictr_bits);
__IO_REG32_BIT(ACTLR,                   0xE000E008, __READ_WRITE , __actlr_bits);
__IO_REG32_BIT(ISER0,                   0xE000E100, __READ_WRITE , __iser0_bits);
__IO_REG32_BIT(ISER1,                   0xE000E104, __READ_WRITE , __iser1_bits);
__IO_REG32_BIT(ICER0,                   0xE000E180, __READ_WRITE , __icer0_bits);
__IO_REG32_BIT(ICER1,                   0xE000E184, __READ_WRITE , __icer1_bits);
__IO_REG32_BIT(ISPR0,                   0xE000E200, __READ_WRITE , __ispr0_bits);
__IO_REG32_BIT(ISPR1,                   0xE000E204, __READ_WRITE , __ispr1_bits);
__IO_REG32_BIT(ICPR0,                   0xE000E280, __READ_WRITE , __icpr0_bits);
__IO_REG32_BIT(ICPR1,                   0xE000E284, __READ_WRITE , __icpr1_bits);
__IO_REG32_BIT(IABR0,                   0xE000E300, __READ_WRITE , __iabr0_bits);
__IO_REG32_BIT(IABR1,                   0xE000E304, __READ_WRITE , __iabr1_bits);
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
__IO_REG32_BIT(GPIOB_DRVR,              0x4001B014, __READ_WRITE , __gpiob_drvr_bits);
__IO_REG32_BIT(GPIOB_LOCKR,             0x4001B018, __READ_WRITE , __gpiob_lockr_bits);
__IO_REG32_BIT(GPIOB_DINR,              0x4001B01C, __READ_WRITE , __gpiob_dinr_bits);
__IO_REG32_BIT(GPIOB_DOUTR,             0x4001B020, __READ_WRITE , __gpiob_doutr_bits);
__IO_REG32_BIT(GPIOB_SRR,               0x4001B024, __READ_WRITE , __gpiob_srr_bits);
__IO_REG32_BIT(GPIOB_RR,                0x4001B028, __READ_WRITE , __gpiob_rr_bits);
__IO_REG32_BIT(AFIO_ESSR0,              0x40022000, __READ_WRITE , __afio_essr0_bits);
__IO_REG32_BIT(AFIO_ESSR1,              0x40022004, __READ_WRITE , __afio_essr1_bits);
__IO_REG32_BIT(AFIO_GPACFGR,            0x40022008, __READ_WRITE , __afio_gpacfgr_bits);
__IO_REG32_BIT(AFIO_GPBCFGR,            0x4002200C, __READ_WRITE , __afio_gpbcfgr_bits);
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
__IO_REG32_BIT(GPTM0_ICTR,              0x4006E074, __READ_WRITE , __gptm0_ictr_bits);
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
__IO_REG32_BIT(GPTM1_ICTR,              0x4006F074, __READ_WRITE , __gptm1_ictr_bits);
__IO_REG32_BIT(GPTM1_EVGR,              0x4006F078, __READ_WRITE , __gptm1_evgr_bits);
__IO_REG32_BIT(GPTM1_INTSR,             0x4006F07C, __READ_WRITE , __gptm1_intsr_bits);
__IO_REG32_BIT(GPTM1_CNTR,              0x4006F080, __READ_WRITE , __gptm1_cntr_bits);
__IO_REG32_BIT(GPTM1_PSCR,              0x4006F084, __READ_WRITE , __gptm1_pscr_bits);
__IO_REG32_BIT(GPTM1_CRR,               0x4006F088, __READ_WRITE , __gptm1_crr_bits);
__IO_REG32_BIT(GPTM1_CH0CCR,            0x4006F090, __READ_WRITE , __gptm1_ch0ccr_bits);
__IO_REG32_BIT(GPTM1_CH1CCR,            0x4006F094, __READ_WRITE , __gptm1_ch1ccr_bits);
__IO_REG32_BIT(GPTM1_CH2CCR,            0x4006F098, __READ_WRITE , __gptm1_ch2ccr_bits);
__IO_REG32_BIT(GPTM1_CH3CCR,            0x4006F09C, __READ_WRITE , __gptm1_ch3ccr_bits);
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
__IO_REG32_BIT(I2C_CR,                  0x40048000, __READ_WRITE , __i2c_cr_bits);
__IO_REG32_BIT(I2C_IER,                 0x40048004, __READ_WRITE , __i2c_ier_bits);
__IO_REG32_BIT(I2C_ADDR,                0x40048008, __READ_WRITE , __i2c_addr_bits);
__IO_REG32_BIT(I2C_SR,                  0x4004800C, __READ_WRITE , __i2c_sr_bits);
__IO_REG32_BIT(I2C_SHPGR,               0x40048010, __READ_WRITE , __i2c_shpgr_bits);
__IO_REG32_BIT(I2C_SLPGR,               0x40048014, __READ_WRITE , __i2c_slpgr_bits);
__IO_REG32_BIT(I2C_DR,                  0x40048018, __READ_WRITE , __i2c_dr_bits);
__IO_REG32_BIT(I2C_TAR,                 0x4004801C, __READ_WRITE , __i2c_tar_bits);
__IO_REG32_BIT(SPI_CR0,                 0x40004000, __READ_WRITE , __spi_cr0_bits);
__IO_REG32_BIT(SPI_CR1,                 0x40004004, __READ_WRITE , __spi_cr1_bits);
__IO_REG32_BIT(SPI_IER,                 0x40004008, __READ_WRITE , __spi_ier_bits);
__IO_REG32_BIT(SPI_CPR,                 0x4000400C, __READ_WRITE , __spi_cpr_bits);
__IO_REG32_BIT(SPI_DR,                  0x40004010, __READ_WRITE , __spi_dr_bits);
__IO_REG32_BIT(SPI_SR,                  0x40004014, __READ_WRITE , __spi_sr_bits);
__IO_REG32_BIT(SPI_FCR,                 0x40004018, __READ_WRITE , __spi_fcr_bits);
__IO_REG32_BIT(SPI_FSR,                 0x4000401C, __READ_WRITE , __spi_fsr_bits);
__IO_REG32_BIT(SPI_FTOCR,               0x40004020, __READ_WRITE , __spi_ftocr_bits);
__IO_REG32_BIT(USART_RBR,               0x40000000, __READ_WRITE , __usart_rbr_bits);
#define USART_TBR     USART_RBR
__IO_REG32_BIT(USART_IER,               0x40000004, __READ_WRITE , __usart_ier_bits);
__IO_REG32_BIT(USART_IIR,               0x40000008, __READ_WRITE , __usart_iir_bits);
__IO_REG32_BIT(USART_FCR,               0x4000000C, __READ_WRITE , __usart_fcr_bits);
__IO_REG32_BIT(USART_LCR,               0x40000010, __READ_WRITE , __usart_lcr_bits);
__IO_REG32_BIT(USART_MODCR,             0x40000014, __READ_WRITE , __usart_modcr_bits);
__IO_REG32_BIT(USART_LSR,               0x40000018, __READ_WRITE , __usart_lsr_bits);
__IO_REG32_BIT(USART_MODSR,             0x4000001C, __READ_WRITE , __usart_modsr_bits);
__IO_REG32_BIT(USART_TPR,               0x40000020, __READ_WRITE , __usart_tpr_bits);
__IO_REG32_BIT(USART_MDR,               0x40000024, __READ_WRITE , __usart_mdr_bits);
__IO_REG32_BIT(USART_IrDACR,            0x40000028, __READ_WRITE , __usart_irdacr_bits);
__IO_REG32_BIT(USART_RS485CR,           0x4000002C, __READ_WRITE , __usart_rs485cr_bits);
__IO_REG32_BIT(USART_SYNCR,             0x40000030, __READ_WRITE , __usart_syncr_bits);
__IO_REG32_BIT(USART_DEGTSTR,           0x40000034, __READ_WRITE , __usart_degtstr_bits);
__IO_REG32_BIT(USART_DLR,               0x40000038, __READ_WRITE , __usart_dlr_bits);

/* Assembler-specific declarations  ***************************************/
#ifdef __IAR_SYSTEMS_ASM__

#endif    /* __IAR_SYSTEMS_ASM__                                          */

/***************************************************************************
**
**  ioHT32F125x Interrupt Lines
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
#define GPTM0          51
#define GPTM1          52
#define I2C            59
#define SPI            61
#define USART          63

#endif /*__ioHT32F125x_H__                                                */
